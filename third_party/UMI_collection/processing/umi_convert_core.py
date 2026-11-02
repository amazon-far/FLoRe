#!/usr/bin/env python3
"""Pure resampling / calibration core for the raw-recording -> UMI Zarr converter.

This module is deliberately dependency-light (numpy + scipy ONLY — NO zarr, av,
cv2, imagecodecs) so the numerically load-bearing logic is unit-testable with
just numpy + scipy. The heavy I/O + image decode + Zarr writing live in
``convert_to_umi.py``. DECISION-I: everything runs in the SINGLE collection
``.venv`` (bootstrap installs the conversion deps there too); there is no
separate ``.venv-proc``.

Everything here implements the conventions verified against the HuMI/UMI
reference pipelines and hardened by the plan review:

  * **Master timeline = the camera-frame timestamps.** Every other modality is
    resampled onto that grid; ``episode_timestamps`` IS that grid.
  * **CLAMP / CLIP everywhere — never assert-in-range.** HuMI's joint resampler
    asserts the query window lies inside the source span and crashes on a
    slightly-longer camera window; we clip query times to the source range and
    use ``fill_value=(first, last)`` so a build degrades gracefully instead.
  * **Rotation via scipy Slerp on ``Rotation`` objects** (NOT component-wise
    interpolation of raw quaternions, which is wrong). Position / gripper-width /
    F/T via ``interp1d(axis=0)``.
  * **Quaternion convention:** the VIVE bridge emits **wxyz**; scipy / UMI want
    **xyzw**. We convert here, at the single conversion point, and emit the UMI
    6-DOF pose ``[:3]=pos, [3:]=axis-angle (rotvec)``.
  * **F/T is the high-rate signal (250 Hz) downsampled to the ~15 Hz camera
    grid** — point-sampling would alias contact transients, so we
    window-average every raw sample that falls in each camera-frame interval
    (a cheap anti-alias) before assigning it to the grid.
  * **gripper_width is ABSOLUTE METRES** (UMI's sampler hard-codes a 0.08 m
    grasp threshold), produced from the motor encoder via a 2-point linear
    calibration — no ArUco.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation, Slerp


# ---------------------------------------------------------------------------
# Timeline hygiene
# ---------------------------------------------------------------------------
def make_strictly_increasing(t: np.ndarray) -> np.ndarray:
    """Return indices selecting a strictly-increasing subsequence of ``t``.

    The RealSense bridge re-sends its last frame (same seq, stale ``t``) on every
    client (re)connect, so a recorded camera-timestamp array can contain a
    backward jump or a duplicate at the episode boundary. Slerp / interp1d
    require strictly-increasing query knots, and the master timeline must be
    monotone, so we drop any frame whose timestamp does not advance past the
    running max. Returns the kept indices (into the original array) so the caller
    can drop the matching camera frames too.
    """
    t = np.asarray(t, dtype=np.float64)
    keep = np.zeros(t.shape[0], dtype=bool)
    last = -np.inf
    for i, ti in enumerate(t):
        if ti > last:
            keep[i] = True
            last = ti
    return np.nonzero(keep)[0]


def deglitch_tracker_positions(t: np.ndarray, pos: np.ndarray,
                               max_speed_mps: float = 3.0,
                               max_run: int = 30,
                               rejoin_dist_m: float = 0.10,
                               return_stats: bool = False):
    """Return indices selecting the tracker samples to KEEP after dropping
    position GLITCH EXCURSIONS (lighthouse occlusion / multipath / momentary
    world-frame wobble).

    The VIVE lighthouse tracker occasionally leaves the true trajectory for a
    short span: it jumps off at a physically-impossible speed (observed 8–96 m/s;
    a manipulating human hand tops out near ~2–3 m/s), sits at/near a WRONG place
    for 1 sample (a spike) up to ~24 samples (~300 ms of frozen/wandering drift),
    then SNAPS BACK onto the real path. Because the converter resamples the tracker
    onto the camera grid with a LINEAR ``interp1d`` (position) / ``Slerp``
    (rotation) that pass THROUGH every knot, an uncorrected excursion lands
    directly in the camera-grid ``robot0_eef_pos`` — i.e. the action/pose label.
    There is no other outlier rejection in the pipeline.

    ALGORITHM — a hot-step scan that brackets each excursion by the good samples on
    either side and drops what's between:
      1. A step ``k`` (between samples ``k`` and ``k+1``) is "hot" when
         ``|pos[k+1]-pos[k]| / (t[k+1]-t[k]) > max_speed_mps``.
      2. At the first hot OUT-jump (good sample ``k`` → excursion begins at ``k+1``)
         scan forward up to ``max_run`` samples for the RETURN: the nearest later
         sample ``r`` that has REJOINED the pre-glitch position, i.e.
         ``|pos[r]-pos[k]| <= rejoin_dist_m``. Drop the excursion samples
         ``k+1 .. r-1`` (whole rows) so the resampler bridges ``k → r`` linearly.
      3. If no rejoin is found within ``max_run`` samples, the span is either
         genuine sustained motion (it ended somewhere NEW) or an unbridgeable
         dropout — leave it untouched and advance one sample (so a later, separate
         glitch is still caught).

    Dropping WHOLE ROWS (the returned indices apply to ``t`` and the quaternion
    columns too) keeps the Slerp rotation knots aligned with the position knots.

    Deliberately CONSERVATIVE — it will not touch:
      * an excursion that never rejoins the pre-glitch position within ``max_run``
        samples (no clean anchor to interpolate from) — held as-is; if a whole
        episode's tracker is that broken, drop the EPISODE (see
        ``tracker_glitch_severity``), don't bridge;
      * the first / last sample — endpoints anchor the clamp and are never dropped
        (a bridge needs a good sample on BOTH sides);
      * anything when no step exceeds ``max_speed_mps``.

    The POSITION-REJOIN test (return sample lands back within ``rejoin_dist_m`` of
    where the excursion left) is what tells a teleport-and-return glitch apart from
    a real fast reposition: a genuine quick move ends somewhere NEW and never
    rejoins the departure point, so it is NOT dropped; a lighthouse glitch snaps
    back near where it left, so it IS bridged. A NET-speed test was rejected here
    because a long SLOW drift accumulates enough elapsed time that its average
    speed drops below the gate while the tracker is still parked at the wrong place
    — position-rejoin does not have that failure mode. ``max_speed_mps=3.0`` sits
    at/above human teleop hand speed yet ~3x below the smallest observed glitch;
    ``max_run=30`` (~0.4 s at ~77 Hz) covers the longest observed drift;
    ``rejoin_dist_m=0.10`` tolerates the small real hand drift during the occlusion.

    Returns kept indices into the original arrays (same contract as
    ``make_strictly_increasing``); a clean input returns ``arange(n)``. Requires
    ``t`` strictly increasing (call ``make_strictly_increasing`` first).

    When ``return_stats=True`` returns ``(kept_idx, stats)`` where ``stats`` is a
    dict counting what the SCAN did, so a QA gate never has to re-derive it from
    residual hot steps (which conflate genuine fast bridge steps with unbridged
    glitches). ``stats`` keys: ``n_bridged_excursions`` (hot out-jumps that found a
    rejoin and were dropped) and ``n_unbridged_excursions`` (hot out-jumps with NO
    rejoin within ``max_run`` — the residual the deglitch could not fix; ONE such
    excursion counts ONCE regardless of how many hot edges it spans), plus
    ``bridged_spans`` / ``unbridged_spans`` — the wall-clock brackets of each, so a
    consumer can mask the corrupted frames. BOTH span lists matter: bridging removes
    the bad samples but leaves the resampler interpolating a straight line across the
    hole, i.e. a fabricated ramp wherever the hand really moved during the occlusion.
    """
    t = np.asarray(t, dtype=np.float64)
    pos = np.asarray(pos, dtype=np.float64)
    n = t.shape[0]
    if n < 3 or max_run <= 0:
        return (np.arange(n), {"n_bridged_excursions": 0, "n_unbridged_excursions": 0,
                               "unbridged_spans": [], "bridged_spans": []}
                ) if return_stats else np.arange(n)
    dt = np.diff(t)
    dt = np.where(dt <= 0, 1e-9, dt)              # guard (t is pre-sanitized)
    step = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    hot = (step / dt) > max_speed_mps              # hot[k] = step k->k+1 implausible
    if not hot.any():
        return (np.arange(n), {"n_bridged_excursions": 0, "n_unbridged_excursions": 0,
                               "unbridged_spans": [], "bridged_spans": []}
                ) if return_stats else np.arange(n)

    drop = np.zeros(n, dtype=bool)
    n_bridged = 0
    n_unbridged = 0
    # Wall-clock (t_start, t_end) of every excursion the scan could NOT bridge, so a
    # consumer can mask exactly the frames it corrupts instead of discarding the whole
    # take. t_start is the last GOOD sample before the out-jump and t_end the first
    # sample after the excursion's hot run, i.e. the span bracketing the discontinuity
    # that survives into the resampled label.
    unbridged_spans: list[tuple[float, float]] = []
    # ...and of every excursion it DID bridge. Same reason, opposite case: dropping the
    # excursion leaves a HOLE, and the resampler draws a straight line across it, so the
    # camera-grid label ramps at up to 201 mm/frame (measured) wherever the hand actually
    # moved while the tracker was lost. Bracketed by the last good sample before the
    # out-jump and the sample that rejoined, i.e. exactly the interpolated span.
    bridged_spans: list[tuple[float, float]] = []
    k = 0
    n_steps = n - 1
    while k < n_steps:
        if not hot[k]:                             # not an out-jump; advance
            k += 1
            continue
        # Hot out-jump: good sample k, excursion starts at k+1. Find the return r
        # (k+2 .. k+1+max_run) that has rejoined the pre-glitch position pos[k].
        found = False
        r_max = min(k + 1 + max_run, n - 1)        # r must be a valid sample index
        for r in range(k + 2, r_max + 1):
            if float(np.linalg.norm(pos[r] - pos[k])) <= rejoin_dist_m:
                drop[k + 1:r] = True               # excursion samples k+1 .. r-1
                n_bridged += 1
                bridged_spans.append((float(t[k]), float(t[r])))
                k = r                               # resume from the clean landing
                found = True
                break
        if not found:
            # No rejoin within the window: one unbridgeable excursion. Skip PAST its
            # hot edges so its return-jump is not re-counted as a second excursion.
            n_unbridged += 1
            k0 = k
            k += 1
            while k < n_steps and hot[k]:
                k += 1
            unbridged_spans.append((float(t[k0]), float(t[min(k, n - 1)])))
    kept = np.nonzero(~drop)[0]
    if return_stats:
        return kept, {"n_bridged_excursions": n_bridged,
                      "n_unbridged_excursions": n_unbridged,
                      "unbridged_spans": unbridged_spans,
                      "bridged_spans": bridged_spans}
    return kept


def tracker_glitch_severity(t: np.ndarray, pos: np.ndarray,
                            max_speed_mps: float = 3.0,
                            max_run: int = 30) -> dict:
    """Summarize how badly a tracker stream glitches, to decide bridge-vs-drop.

    Runs the same excursion scan as :func:`deglitch_tracker_positions` and reports
    what it WOULD do, plus what it could NOT fix, as a plain dict:
      * ``n``                  — samples in (strictly-increasing) input
      * ``n_hot_steps``        — steps exceeding ``max_speed_mps`` (raw, pre-drop)
      * ``n_dropped``          — samples the deglitch drops (bridgeable excursions)
      * ``dropped_frac``       — ``n_dropped / n``
      * ``n_unbridged``        — EXCURSIONS with no rejoin within ``max_run`` that
                                 the deglitch could NOT fix (counted once each, from
                                 the scan itself — NOT residual hot-step count)
      * ``unbridged_spans``    — ``[(t_start, t_end), ...]`` wall-clock brackets of
                                 those excursions, so a consumer can mask exactly the
                                 frames they corrupt instead of dropping the take
      * ``bridged_spans``      — ``[(t_start, t_end), ...]`` brackets of the excursions
                                 that WERE bridged. The bridge removes the bad samples
                                 but the resampler interpolates straight across the
                                 resulting hole, so these spans hold a FABRICATED linear
                                 ramp (measured up to 201 mm in one camera frame). Mask
                                 them too — a bridged glitch is not a repaired glitch
      * ``max_jump_m``         — largest single-step position jump, raw (m)
      * ``peak_speed_mps``     — largest single-step speed, raw (m/s)
      * ``residual_max_jump_m``   — largest single-step jump on the KEPT stream (m);
                                    what actually reaches the resampled label
      * ``residual_peak_speed_mps`` — largest single-step speed on the KEPT stream

    Advisory only — the converter uses ``deglitch_tracker_positions`` to bridge; a
    review/QA gate uses this to flag episodes to exclude. Prefer gating on
    ``n_unbridged``, ``residual_max_jump_m`` or ``residual_peak_speed_mps`` (one
    unbridgeable teleport already yields an unphysical residual jump) rather than a
    raw hot-step count, which conflates genuine fast bridge steps with glitches."""
    t = np.asarray(t, dtype=np.float64)
    pos = np.asarray(pos, dtype=np.float64)
    n = t.shape[0]
    out = {"n": int(n), "n_hot_steps": 0, "n_dropped": 0, "dropped_frac": 0.0,
           "n_unbridged": 0, "unbridged_spans": [], "bridged_spans": [],
           "max_jump_m": 0.0, "peak_speed_mps": 0.0,
           "residual_max_jump_m": 0.0, "residual_peak_speed_mps": 0.0}
    if n < 3:
        return out
    dt = np.diff(t)
    dt = np.where(dt <= 0, 1e-9, dt)
    step = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    speed = step / dt
    hot = speed > max_speed_mps
    out["n_hot_steps"] = int(hot.sum())
    out["max_jump_m"] = float(step.max()) if step.size else 0.0
    out["peak_speed_mps"] = float(speed.max()) if speed.size else 0.0
    kept, stats = deglitch_tracker_positions(t, pos, max_speed_mps, max_run,
                                             return_stats=True)
    out["n_dropped"] = int(n - kept.shape[0])
    out["dropped_frac"] = out["n_dropped"] / n
    out["n_unbridged"] = int(stats["n_unbridged_excursions"])
    # Wall-clock spans of the excursions the deglitch could not fix. Lets the caller
    # mask exactly the corrupted frames (and only the windows that touch them) rather
    # than throwing the whole take away.
    out["unbridged_spans"] = list(stats.get("unbridged_spans", []))
    # Wall-clock brackets of every BRIDGED excursion. The bridge repairs the *shape* (the
    # excursion samples are gone) but the resampler still draws a straight line across the
    # hole, so the camera grid gets a fast linear RAMP wherever the hand really moved
    # during the occlusion. That ramp is a fabricated label, not measurement, and until
    # this key existed NOTHING flagged it: `unbridged_spans` is empty for these by
    # construction (they were bridged), which is exactly why clean94 episodes 30 and 31
    # carry 118/127 mm one-frame jumps with `robot0_pose_glitch == 0`.
    out["bridged_spans"] = list(stats.get("bridged_spans", []))
    # Residual on the KEPT stream — the discontinuity that actually survives into
    # the resampled label (a single unbridgeable teleport shows here as one big jump).
    if kept.shape[0] >= 2:
        kt = t[kept]
        kp = pos[kept]
        kdt = np.diff(kt)
        kdt = np.where(kdt <= 0, 1e-9, kdt)
        kstep = np.linalg.norm(np.diff(kp, axis=0), axis=1)
        out["residual_max_jump_m"] = float(kstep.max())
        out["residual_peak_speed_mps"] = float((kstep / kdt).max())
    return out


# Rotation-excursion scan thresholds. Measured on the 226-take 0822+0823 screwdriver set
# (each take's own max angular rate: median 87 deg/s, p95 254): every take that holds a
# there-and-back orientation excursion peaks at 519-1992 deg/s, and the worst take WITHOUT
# one peaks at 367 -- so 400 sits in a clean gap rather than on a guess.
ROT_EXCURSION_RATE_DPS = 400.0    # out-jump gate, deg/s on the RAW stream
ROT_EXCURSION_REJOIN_DEG = 12.0   # "landed back where it left" (analogue of rejoin_dist_m)
ROT_EXCURSION_MAX_RUN = 30        # samples to look ahead for the rejoin (~0.4 s at 77 Hz)
#: Amplitude below which an excursion is not worth excluding a take over: 10.2 deg is the
#: smallest one among the five hand-excluded takes of the shipped 0822 dataset, and once
#: Slerped onto the 30 Hz camera grid a smaller one is indistinguishable from real motion.
ROT_EXCURSION_MIN_AMP_DEG = 10.0


def rotation_excursions(t: np.ndarray, quat_wxyz: np.ndarray,
                        max_rate_dps: float = ROT_EXCURSION_RATE_DPS,
                        max_run: int = ROT_EXCURSION_MAX_RUN,
                        rejoin_deg: float = ROT_EXCURSION_REJOIN_DEG,
                        min_amp_deg: float = ROT_EXCURSION_MIN_AMP_DEG) -> list[dict]:
    """Find THERE-AND-BACK orientation excursions on a raw tracker stream — QA ONLY.

    The rotational twin of :func:`deglitch_tracker_positions`' excursion scan, with the
    same discriminator (a glitch leaves at an impossible rate and REJOINS where it left;
    genuine fast motion ends somewhere new). ``max_rate_dps`` replaces ``max_speed_mps``
    and ``rejoin_deg`` replaces ``rejoin_dist_m``.

    **Nothing in the conversion acts on this.** The deglitch is position-only by design
    (dropping rows on a rotation criterion would also delete good position knots), so a
    rotation glitch with a quiet position reaches the stored label; the only remedy is to
    ``--exclude`` the take. This function exists so the QA tools AGREE on which takes those
    are, instead of each carrying its own heuristic.

    Run on the RAW ~77 Hz stream, not on the camera-grid label, because the camera grid
    both attenuates and hides these: a 21 deg excursion inside a stretch of dropped frames
    showed up as 7.2 deg/frame on the label — under any per-frame magnitude floor — while
    its raw rate was 1452 deg/s. (Measured: take_371 of the 0823 session.)

    ``t`` must be strictly increasing (call :func:`make_strictly_increasing` first).
    ``quat_wxyz`` is [N,4]; only ``|dot|`` between rows is used, so any consistent
    component order (wxyz or xyzw) gives identical results.

    KNOWN LIMITATION: orientation alone cannot separate "jumped away and snapped back"
    from "spun a whole revolution" — both end AT the departure orientation. A sustained
    spin only rejoins within ``max_run`` samples above ~920 deg/s (2.6 rev/s of the
    wrist) and the fastest non-glitch take measured here peaks at 367 deg/s, so the gap
    is ~2.5x; if a task ever spins the tool that fast, unwrap the incremental rotation
    rather than raising the rate gate. Pinned by
    ``test_rotation_excursions_known_limit_a_full_turn_reads_as_a_rejoin``.

    Returns one dict per excursion whose peak amplitude reaches ``min_amp_deg``, in time
    order: ``t`` (wall time of the last GOOD sample before it), ``amp_deg`` (peak angle
    away from that sample), ``rate_dps`` (the out-jump rate) and ``n_samples`` (how many
    samples were off-path). An empty list means nothing above threshold.
    """
    t = np.asarray(t, dtype=np.float64)
    q = np.asarray(quat_wxyz, dtype=np.float64)
    n = q.shape[0]
    if n < 3 or t.shape[0] != n:
        return []

    def _ang(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Angle in degrees between quaternion rows, double-cover safe."""
        d = np.abs(np.sum(a * b, axis=-1))
        return np.degrees(2.0 * np.arccos(np.clip(d, 0.0, 1.0)))

    dt = np.diff(t)
    dt = np.where(dt <= 0, 1e-9, dt)
    rate = _ang(q[:-1], q[1:]) / dt

    out: list[dict] = []
    k = 0
    while k < n - 1:
        if rate[k] > max_rate_dps:
            rejoin = -1
            for r in range(k + 2, min(k + 2 + max_run, n)):
                if _ang(q[r], q[k]) <= rejoin_deg:
                    rejoin = r
                    break
            if rejoin > 0:
                amp = float(_ang(q[k + 1:rejoin], q[k]).max())
                if amp >= min_amp_deg:
                    out.append({"t": float(t[k]), "amp_deg": amp,
                                "rate_dps": float(rate[k]),
                                "n_samples": int(rejoin - k - 1)})
                k = rejoin          # resume past the excursion, as the position scan does
                continue
        k += 1
    return out


# ---------------------------------------------------------------------------
# Continuous-take -> UMI episodes: split the camera timeline at marker times
# ---------------------------------------------------------------------------
def split_take_into_episodes(cam_t: np.ndarray, marker_t: np.ndarray,
                             min_frames: int = 2) -> list[tuple[int, int]]:
    """Split a take's (strictly-increasing) camera timeline into UMI episodes at
    the episode-split MARKER wall-clock times (DECISION-E).

    The operator long-presses ``o+p`` to drop a marker at the boundary between two
    demonstrations; the recorder stores those wall-clock times in ``markers.npy``
    and we segment HERE, offline. A marker at time ``m`` ends the episode of all
    frames with ``cam_t < m`` and starts the next at the first frame ``>= m``.
    ``N`` markers therefore yield ``N+1`` episodes (no markers -> the whole take
    is one episode).

    Returns a list of ``(start, end)`` half-open index slices into ``cam_t``.
    Segments shorter than ``min_frames`` (e.g. a double-pressed marker, or a
    marker before the first / after the last frame) are DROPPED — a 0/1-frame
    "episode" can't be resampled or give a meaningful demo_start/demo_end. The
    caller should log how many were dropped (no silent truncation).
    """
    cam_t = np.asarray(cam_t, dtype=np.float64)
    n = cam_t.shape[0]
    marker_t = np.asarray(marker_t, dtype=np.float64).reshape(-1)
    if n == 0:
        return []
    if marker_t.shape[0] == 0:
        return [(0, n)] if n >= min_frames else []
    # Where each marker falls in the camera timeline (first frame at/after marker).
    cuts = np.searchsorted(cam_t, np.sort(marker_t), side="left")
    bounds = [0, *[int(c) for c in cuts], n]
    segments: list[tuple[int, int]] = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        if b - a >= min_frames:
            segments.append((a, b))
    return segments


# ---------------------------------------------------------------------------
# Continuous-take review: describe + sanity-flag the episodes a marker set would
# produce, BEFORE the irreversible Zarr write (the offline-review safety net).
#
# The cascade these guard against: a MISSED episode-split marker silently FUSES
# two demonstrations into one over-long episode (and renumbers every later one);
# a DOUBLED marker can leave a spurious very-short episode. ``split_take_into_..``
# already drops <2-frame slivers, but a real-but-wrong segmentation reaches
# training undetected today (``validate_episode_timeline`` only checks continuity,
# never a suspiciously long fused episode or an implausibly short one). These
# pure helpers let ``recording/review_take.py`` show EXACTLY what the converter
# will produce and auto-flag the implausible segments, with NO declared episode
# count required — so a single capture-time marker error is visible and fixable
# offline. They are deliberately numpy-only and reuse ``split_take_into_episodes``
# verbatim, so the preview equals the real conversion.
# ---------------------------------------------------------------------------
def summarize_segments(cam_t: np.ndarray, marker_t: np.ndarray,
                       min_frames: int = 2) -> list[dict]:
    """Describe the episodes ``split_take_into_episodes`` would emit for this
    marker set: one dict per KEPT segment with ``{index, start, end, n_frames,
    t_start, t_end, dur_s}`` (indices into ``cam_t``; times are whatever units
    ``cam_t`` carries — absolute wall-clock from the converter, or take-relative
    from the review tool). Built on the production splitter so it can never
    disagree with the real conversion."""
    cam_t = np.asarray(cam_t, dtype=np.float64)
    segs = split_take_into_episodes(cam_t, marker_t, min_frames=min_frames)
    out: list[dict] = []
    for i, (a, b) in enumerate(segs):
        out.append({
            "index": i,
            "start": int(a),
            "end": int(b),
            "n_frames": int(b - a),
            "t_start": float(cam_t[a]),
            "t_end": float(cam_t[b - 1]),
            "dur_s": float(cam_t[b - 1] - cam_t[a]),
        })
    return out


def grasp_close_times(gripper_t: np.ndarray, width_m: np.ndarray,
                      width_open_m: float, close_frac: float = 0.5,
                      hysteresis: float = 0.15) -> np.ndarray:
    """Return the wall-clock times at which the gripper STARTS a grasp (a falling
    edge: width crosses DOWN through ``close_frac*width_open_m``). Schmitt-trigger
    hysteresis (re-open above ``(close_frac+hysteresis)*width_open_m`` before the
    next close can fire) rejects chatter near the threshold, so one physical
    open->close->open grasp counts ONCE.

    A normal demonstration ends with one grasp, so the count of these events
    inside one episode window is a physical proxy for "how many demos are in
    here": >= 2 grasps in a single segment is a possible indicator of a MISSED
    marker that fused two demos.

    Advisory only — pick-and-hold / regrasp tasks break the one-grasp-per-demo
    assumption in BOTH directions, so the review tool treats this as a hint, not
    a gate, and ``outlier_long`` (duration) is the parallel signal:
      * Over-counting (false positive): a regrasp that fully re-opens past
        ``open_th`` and closes again within ONE demo adds an extra event.
      * Under-counting (false negative, the more dangerous direction): a regrasp
        that re-opens only INTO the hysteresis dead zone ``[close_th, open_th)``
        never re-arms the trigger, so every subsequent close is dropped — a fused
        two-demo episode can then show a count of 1 and raise NO ``multi_grasp``
        flag. When the per-demo gripper excursion is known to be small, lower
        ``hysteresis`` or ``close_frac`` to match the task's open fraction.
    A window with fewer than 2 samples always returns empty (no edge to observe)."""
    t = np.asarray(gripper_t, dtype=np.float64).reshape(-1)
    w = np.asarray(width_m, dtype=np.float64).reshape(-1)
    if t.shape[0] == 0 or w.shape[0] == 0 or width_open_m <= 0:
        return np.zeros((0,), dtype=np.float64)
    n = min(t.shape[0], w.shape[0])
    t, w = t[:n], w[:n]
    close_th = close_frac * width_open_m
    open_th = min(close_frac + hysteresis, 1.0) * width_open_m
    times: list[float] = []
    armed = w[0] >= close_th  # only fire a close once we've seen an "open" state
    for i in range(1, n):
        if armed and w[i] < close_th:
            times.append(float(t[i]))
            armed = False
        elif not armed and w[i] >= open_th:
            armed = True
    return np.asarray(times, dtype=np.float64)


def grasps_per_segment(summary: list[dict], close_times: np.ndarray) -> list[int]:
    """Count grasp-close events (from :func:`grasp_close_times`) inside each
    segment's ``[t_start, t_end]`` window. ``close_times`` and the segment
    ``t_*`` must be in the SAME time base (both take-relative or both absolute).

    A close event whose timestamp falls strictly between ``t_end`` of segment i
    (``cam_t[b-1]``) and ``t_start`` of segment i+1 (``cam_t[b]``) — the camera
    inter-frame gap at a marker boundary, ~66 ms at 15 Hz — is assigned to neither
    segment. This is a cosmetic artefact of the camera-grid-aligned window edges;
    the flag is advisory only, so no data is lost and the visual timeline review
    remains the real check."""
    ct = np.asarray(close_times, dtype=np.float64).reshape(-1)
    out: list[int] = []
    for s in summary:
        n = int(np.count_nonzero((ct >= s["t_start"]) & (ct <= s["t_end"])))
        out.append(n)
    return out


def flag_segments(summary: list[dict], grasp_counts: list[int] | None = None,
                  short_s: float = 0.5, long_k: float = 1.8,
                  min_segments_for_median: int = 3) -> list[dict]:
    """Annotate each segment with advisory plausibility flags (NO declared episode
    count needed). Returns a NEW list mirroring ``summary`` with two added keys:
    ``flags`` (list of short codes) and ``note`` (human string).

    Flags:
      * ``too_short``   — ``dur_s < short_s`` (a real demonstration is rarely
        sub-second; the likely cause is a DOUBLED marker leaving a sliver).
      * ``outlier_long``— ``dur_s > long_k * median(dur_s)`` (only when there are
        at least ``min_segments_for_median`` segments so the median is meaningful);
        a possible indicator of a MISSED marker (two demos fused) — inspect the
        gripper/F/T trace and grasp count to confirm.
      * ``multi_grasp`` — ``grasp_counts[i] >= 2`` (two grasp cycles inside one
        episode → probably two demos fused; only added when ``grasp_counts`` given).

    Heuristic and advisory by design: the operator's eyes on the timeline +
    gripper/F/T anchors are the real check. Thresholds are caller-tunable."""
    durs = np.asarray([s["dur_s"] for s in summary], dtype=np.float64)
    med = float(np.median(durs)) if durs.size else 0.0
    use_median = durs.size >= min_segments_for_median and med > 0.0
    out: list[dict] = []
    for i, s in enumerate(summary):
        flags: list[str] = []
        if s["dur_s"] < short_s:
            flags.append("too_short")
        if use_median and s["dur_s"] > long_k * med:
            flags.append("outlier_long")
        if grasp_counts is not None and i < len(grasp_counts) and grasp_counts[i] >= 2:
            flags.append("multi_grasp")
        note_bits = []
        if "too_short" in flags:
            note_bits.append(f"only {s['dur_s']:.2f}s — likely a doubled marker")
        if "outlier_long" in flags:
            note_bits.append(f"{s['dur_s']:.2f}s vs median {med:.2f}s — likely a MISSED marker (two demos fused)")
        if "multi_grasp" in flags:
            note_bits.append(f"{grasp_counts[i]} grasp cycles inside one episode — likely two demos fused")
        e = dict(s)
        e["flags"] = flags
        e["note"] = "; ".join(note_bits)
        out.append(e)
    return out


# ---------------------------------------------------------------------------
# Pose resampling (position interp1d + rotation Slerp), then TCP extrinsic
# ---------------------------------------------------------------------------
def quat_wxyz_to_xyzw(q_wxyz: np.ndarray) -> np.ndarray:
    """[...,4] wxyz -> [...,4] xyzw (scipy's convention). Normalizes."""
    q = np.asarray(q_wxyz, dtype=np.float64)
    q_xyzw = q[..., [1, 2, 3, 0]]
    n = np.linalg.norm(q_xyzw, axis=-1, keepdims=True)
    n = np.where(n < 1e-12, 1.0, n)
    return q_xyzw / n


def resample_position(src_t: np.ndarray, src_pos: np.ndarray, query_t: np.ndarray) -> np.ndarray:
    """Linear-interp xyz onto ``query_t``; query clamped to the source span,
    out-of-range held at the endpoint value (NEVER raises)."""
    src_t = np.asarray(src_t, dtype=np.float64)
    src_pos = np.asarray(src_pos, dtype=np.float64)
    if src_t.shape[0] == 1:
        return np.repeat(src_pos[:1], len(query_t), axis=0)
    f = interp1d(src_t, src_pos, axis=0, bounds_error=False,
                 fill_value=(src_pos[0], src_pos[-1]))
    qt = np.clip(np.asarray(query_t, dtype=np.float64), src_t[0], src_t[-1])
    return f(qt)


def resample_rotation_wxyz(src_t: np.ndarray, src_quat_wxyz: np.ndarray,
                           query_t: np.ndarray) -> Rotation:
    """Slerp wxyz quaternions onto ``query_t`` (clamped). Returns a scipy
    Rotation. Single-sample / degenerate inputs hold the one rotation."""
    src_t = np.asarray(src_t, dtype=np.float64)
    rot = Rotation.from_quat(quat_wxyz_to_xyzw(src_quat_wxyz))
    if src_t.shape[0] == 1:
        return Rotation.from_quat(np.repeat(rot.as_quat()[:1], len(query_t), axis=0))
    slerp = Slerp(src_t, rot)
    qt = np.clip(np.asarray(query_t, dtype=np.float64), src_t[0], src_t[-1])
    return slerp(qt)


def pose_to_6d(pos: np.ndarray, rot: Rotation) -> np.ndarray:
    """[T,3] pos + Rotation -> [T,6] UMI pose ([:3]=pos, [3:]=rotvec)."""
    pos = np.asarray(pos, dtype=np.float64)
    rotvec = rot.as_rotvec()
    return np.concatenate([pos, rotvec], axis=-1).astype(np.float64)


def apply_tcp_extrinsic(pos: np.ndarray, rot: Rotation,
                        T_tracker_tcp: np.ndarray) -> tuple[np.ndarray, Rotation]:
    """Compose the per-frame world<-tracker pose with a fixed tracker->TCP
    extrinsic (4x4) to get the world<-TCP pose UMI wants.

    ``world_T_tcp = world_T_tracker @ tracker_T_tcp``. With an identity extrinsic
    this is a no-op (and the converter stamps UNCALIBRATED — see convert_to_umi).
    Rotation couples into translation here, which is exactly why an identity
    placeholder silently poisons rotated poses and must be calibrated before
    training.
    """
    T_tracker_tcp = np.asarray(T_tracker_tcp, dtype=np.float64)
    pos = np.asarray(pos, dtype=np.float64)
    n = pos.shape[0]
    world_T_tracker = np.broadcast_to(np.eye(4), (n, 4, 4)).copy()
    world_T_tracker[:, :3, :3] = rot.as_matrix()
    world_T_tracker[:, :3, 3] = pos
    world_T_tcp = world_T_tracker @ T_tracker_tcp[None]
    tcp_pos = world_T_tcp[:, :3, 3]
    tcp_rot = Rotation.from_matrix(world_T_tcp[:, :3, :3])
    return tcp_pos, tcp_rot


# ---------------------------------------------------------------------------
# Gripper: encoder -> absolute width (metres), 2-point linear, clamped
# ---------------------------------------------------------------------------
@dataclass
class GripperWidthCalib:
    """Encoder (rad) -> physical jaw opening (m), 2-point linear + clamp.

    UMI's ``robot0_gripper_width`` is the absolute opening in metres and its
    sampler hard-codes a 0.08 m grasp threshold, so the SCALE matters — this maps
    the raw DM4310 encoder reading to true metres (NOT a 0..1 fraction).
    """

    enc_closed: float
    enc_open: float
    width_closed_m: float
    width_open_m: float

    def to_width_m(self, enc_rad: np.ndarray) -> np.ndarray:
        enc = np.asarray(enc_rad, dtype=np.float64)
        lo, hi = self.enc_closed, self.enc_open
        # Build a monotone-x interpolator regardless of which endpoint is larger
        # (the DM4310 re-references on power-cycle, so open may be < closed in rad).
        xs = np.array([lo, hi], dtype=np.float64)
        ys = np.array([self.width_closed_m, self.width_open_m], dtype=np.float64)
        order = np.argsort(xs)
        xs, ys = xs[order], ys[order]
        if abs(xs[1] - xs[0]) < 1e-12:
            return np.full_like(enc, ys[0])
        f = interp1d(xs, ys, bounds_error=False, fill_value=(ys[0], ys[1]))
        return f(np.clip(enc, xs[0], xs[1]))


def effective_gripper_calib(gripper: np.ndarray, base: "GripperWidthCalib | None",
                            min_span_rad: float = 0.2
                            ) -> tuple["GripperWidthCalib | None", str]:
    """Pick the encoder endpoints to map ``encoder_rad -> width_m`` with.

    The DM4310 is an ABSOLUTE encoder that RE-REFERENCES on every motor
    power-cycle, so the open/closed encoder readings shift between sessions (the
    repo ``gripper_range.json`` ships the i2rt *nominal* endpoints ``enc_open=4.9
    enc_closed=0.0`` and is explicitly "re-map per power-cycle"). The startup
    self-test homes the real stops each launch and the gripper bridge reports them
    on every frame, so the recorder stores the LIVE endpoints per-sample in
    ``gripper.npy`` columns ``open_rad`` (3) and ``closed_rad`` (4).

    If we map with the stale nominal endpoints instead, a session whose live
    encoder sits entirely outside ``[0, 4.9]`` (e.g. all-negative after a
    re-reference) CLAMPS every width to one rail — the whole gripper-width signal,
    which UMI thresholds at 0.08 m for grasp, is silently destroyed. So we PREFER
    the recorded endpoints, keeping the physical ``width_*_m`` (a real measurement)
    from ``base``.

    Returns ``(calib, source)`` where ``source`` is ``"recorded"`` /
    ``"json"`` / ``"none"`` for logging. Falls back to ``base`` (the JSON calib)
    when the recorded endpoints are absent or degenerate (``|open-closed| <
    min_span_rad``, e.g. a bridge that never reported them leaves both 0.0), and
    returns ``(None, "none")`` if there is no calib at all (width -> zeros, as
    before)."""
    if base is None:
        return None, "none"
    # Recorded endpoints live in cols 3 (open_rad) and 4 (closed_rad); they are set
    # once at homing so they're effectively constant within a take — use the median
    # to shrug off any stray row. Require both columns to be present.
    if gripper is not None and gripper.ndim == 2 and gripper.shape[1] >= 5 and gripper.shape[0] > 0:
        rec_open = float(np.median(gripper[:, 3]))
        rec_closed = float(np.median(gripper[:, 4]))
        if abs(rec_open - rec_closed) >= min_span_rad:
            return (GripperWidthCalib(enc_closed=rec_closed, enc_open=rec_open,
                                      width_closed_m=base.width_closed_m,
                                      width_open_m=base.width_open_m),
                    "recorded")
    return base, "json"


def resample_gripper_width(src_t: np.ndarray, width_m: np.ndarray,
                           query_t: np.ndarray) -> np.ndarray:
    """interp1d gripper width onto the camera grid (clamped). Returns [T,1]."""
    src_t = np.asarray(src_t, dtype=np.float64)
    width_m = np.asarray(width_m, dtype=np.float64).reshape(-1)
    if src_t.shape[0] == 1:
        out = np.full(len(query_t), width_m[0])
    else:
        f = interp1d(src_t, width_m, bounds_error=False,
                     fill_value=(width_m[0], width_m[-1]))
        out = f(np.clip(np.asarray(query_t, dtype=np.float64), src_t[0], src_t[-1]))
    return out.reshape(-1, 1).astype(np.float32)


# ---------------------------------------------------------------------------
# Secondary camera -> the master (LEFT-camera) grid
# ---------------------------------------------------------------------------
#: Largest tolerable |t_right - t_grid| before the pairing is called suspect, as a
#: FRACTION of the grid period. Above this the two views are showing different
#: instants and a policy conditioned on both sees an inconsistent scene. 0.75 rather
#: than 0.5 because a causal hold is expected to sit slightly behind by construction.
CAMERA_PAIR_SKEW_WARN_FRAC = 0.75


def resample_frame_index(src_t: np.ndarray, grid_t: np.ndarray,
                         causal: bool = True) -> tuple[np.ndarray, dict]:
    """Pick ONE source frame per grid time. Returns ``(idx[T], stats)``.

    Images cannot be interpolated, so a secondary camera is paired to the master grid
    by SELECTION. ``causal=True`` (the default) takes the most recent frame at or
    before each grid time — a zero-order hold, so the paired view never shows an
    instant the master grid has not reached yet. The alternative (nearest in time)
    halves the worst-case skew but can hand the model a frame from the FUTURE, which
    on the action-label grid is a leak; lag is the safer error.

    Grid times before the secondary camera's first frame have no causal answer; they
    clamp to index 0 and are counted in ``stats['n_pre']`` so the caller can warn
    instead of silently pairing a stale frame.

    ``stats``: n_pre, skew_median_s, skew_max_s (signed t_src - t_grid, so a causal
    hold gives NEGATIVE skew), and reused_max (how many grid frames shared one source
    frame — >1 means the secondary camera is slower than the master grid).
    """
    src_t = np.asarray(src_t, dtype=np.float64).reshape(-1)
    grid_t = np.asarray(grid_t, dtype=np.float64).reshape(-1)
    if src_t.size == 0:
        return np.zeros(0, dtype=np.int64), {"n_pre": 0, "skew_median_s": None,
                                             "skew_max_s": None, "reused_max": 0}
    if causal:
        idx = np.searchsorted(src_t, grid_t, side="right") - 1
        n_pre = int((idx < 0).sum())
        idx = np.clip(idx, 0, src_t.size - 1)
    else:
        idx = np.searchsorted(src_t, grid_t, side="left")
        idx = np.clip(idx, 0, src_t.size - 1)
        lo = np.clip(idx - 1, 0, src_t.size - 1)
        take_lo = np.abs(src_t[lo] - grid_t) <= np.abs(src_t[idx] - grid_t)
        idx = np.where(take_lo, lo, idx)
        n_pre = 0
    skew = src_t[idx] - grid_t
    _, counts = np.unique(idx, return_counts=True)
    return idx.astype(np.int64), {
        "n_pre": n_pre,
        "skew_median_s": float(np.median(skew)) if skew.size else None,
        "skew_max_s": float(skew[np.argmax(np.abs(skew))]) if skew.size else None,
        "reused_max": int(counts.max()) if counts.size else 0,
    }


# ---------------------------------------------------------------------------
# Force/torque: 250 Hz -> camera grid with a cheap anti-alias (interval mean)
# ---------------------------------------------------------------------------
def antialias_resample_wrench(src_t: np.ndarray, wrench: np.ndarray,
                              grid_t: np.ndarray) -> np.ndarray:
    """Downsample a high-rate wrench [N,6] to the camera grid [T] by averaging
    every raw sample whose timestamp falls in the half-open interval centred on
    each grid point (midpoints between neighbours). Intervals with no sample
    fall back to nearest-neighbour interp (clamped). This avoids the aliasing of
    naive point-sampling a 250 Hz contact signal down to ~15 Hz.

    WINDOW-CONFINED (continuous-take / marker-split, DECISION-E): the caller may
    pass the WHOLE take's F/T but only ONE episode's camera ``grid_t`` sub-window.
    Naive ±inf outer interval edges would then dump every sample BEFORE the
    episode (i.e. a previous demonstration's contact transients) into the first
    grid bin and every sample AFTER it into the last — cross-demo wrench bleed at
    every marker boundary. We therefore EXCLUDE samples outside
    ``[grid_t[0], grid_t[-1]]`` from the interval-mean; grid points left empty by
    that exclusion fall back to a clamped interp on the FULL stream (so an episode
    whose F/T starts slightly late still gets a sane held value, not a zero).

    Returns [T,6] float32.
    """
    src_t = np.asarray(src_t, dtype=np.float64)
    wrench = np.asarray(wrench, dtype=np.float64)
    grid_t = np.asarray(grid_t, dtype=np.float64)
    T = grid_t.shape[0]
    out = np.zeros((T, wrench.shape[1]), dtype=np.float64)

    if src_t.shape[0] == 0:
        return out.astype(np.float32)
    if T == 1:
        # Single grid point: mean of the in-window samples (fall back to all).
        in_win = src_t == src_t  # all-true; the window is a single instant
        out[:] = wrench.mean(axis=0)
        return out.astype(np.float32)

    # Confine the interval-mean to this episode's window so a neighbouring demo's
    # samples can't bleed across a marker boundary.
    in_win = (src_t >= grid_t[0]) & (src_t <= grid_t[-1])
    win_t = src_t[in_win]
    win_w = wrench[in_win]

    has = np.zeros(T, dtype=bool)
    if win_t.shape[0] >= 1:
        # Interval edges = midpoints between consecutive grid points, extended to
        # ±inf — but win_t is already clipped to [grid_t[0], grid_t[-1]], so the
        # outer bins only collect this window's edge samples.
        mids = 0.5 * (grid_t[:-1] + grid_t[1:])
        edges = np.concatenate([[-np.inf], mids, [np.inf]])
        bins = np.searchsorted(edges, win_t, side="right") - 1
        bins = np.clip(bins, 0, T - 1)
        sums = np.zeros((T, wrench.shape[1]), dtype=np.float64)
        counts = np.zeros(T, dtype=np.int64)
        np.add.at(sums, bins, win_w)
        np.add.at(counts, bins, 1)
        has = counts > 0
        out[has] = sums[has] / counts[has, None]

    # Empty intervals (or no in-window samples at all): clamped linear interp from
    # the FULL stream, so edges hold the nearest real value rather than zero.
    if not np.all(has):
        if src_t.shape[0] == 1:
            out[~has] = wrench[0]
        else:
            f = interp1d(src_t, wrench, axis=0, bounds_error=False,
                         fill_value=(wrench[0], wrench[-1]))
            qt = np.clip(grid_t[~has], src_t[0], src_t[-1])
            out[~has] = f(qt)
    return out.astype(np.float32)


# ---------------------------------------------------------------------------
# Force/torque: hi-rate per-step CAUSAL backward window -> [T, K, 6]
# ---------------------------------------------------------------------------
def resample_wrench_hires(src_t: np.ndarray, wrench: np.ndarray,
                          grid_t: np.ndarray, K: int) -> np.ndarray:
    """Per-camera-step CAUSAL hi-rate wrench stack ``[T, K, 6]``: for each camera
    grid point ``grid_t[i]`` take the K most-recent raw F/T samples at or before it
    (a backward-looking window ENDING at the frame timestamp). This is the MuSe-style
    hi-rate F/T target — each policy step carries a short *causal* wrench trajectory.

    Why a backward window (not the disjoint per-interval bins of
    :func:`antialias_resample_wrench`): consecutive steps' backward windows are
    NEAR-CONTIGUOUS (overlapping or with a small inter-window gap — measured ~48%
    overlap / ~52% small-gap on real takes), so K is decoupled from how many raw
    samples land in one ~11 Hz camera interval. At the measured ~178 Hz F/T / ~11.8 Hz
    camera (~15 raw/interval), a K=16 backward window is fully covered by real samples
    for every frame EXCEPT frame 0 of an episode (see EDGE below). Slot order is
    chronological: ``out[i, K-1]`` is the sample closest to ``grid_t[i]``, ``out[i, 0]``
    the oldest in the window.

    WINDOW-CONFINED (marker-split safety): only samples in ``[grid_t[0], grid_t[-1]]``
    are eligible, so a neighbouring demonstration's contact transients can never bleed
    across an episode boundary when the caller passes the WHOLE take's F/T with only
    one episode's ``grid_t`` sub-window (same discipline as
    :func:`antialias_resample_wrench`). Under the current per-episode recording model
    the recorder already trims F/T to the take, so this is a no-op there and load-bearing
    only for the legacy marker-split path.

    EDGE / STARTUP handling: a frame with fewer than K in-window samples at or before it
    is LEFT-PADDED by repeating the oldest available in-window sample — never fabricated,
    never pulled from before the episode. A frame with NO in-window sample at or before
    it — in practice frame 0 of nearly every episode, since window-confinement drops the
    pre-``grid_t[0]`` samples and the first IN-window F/T sample lands a few ms AFTER
    ``grid_t[0]`` (timestamp discreteness at ~178 Hz), so frame 0's newest slot is a
    ~7-12 ms boundary lookahead — left-holds the earliest in-window sample; a grid with no
    in-window samples at all falls back to a clamped interp on the FULL stream, held
    across all K slots (so it is a real value, not zero).

    Parameters mirror :func:`antialias_resample_wrench`. ``K <= 0`` returns a shaped
    ``[T, 0, D]`` array and the caller MUST omit the key entirely (see
    ``convert_to_umi`` wiring) so ``K == 0`` keeps old zarrs byte-identical.

    Returns ``[T, K, D]`` float32 (D == wrench.shape[1], normally 6).

    DOWNSTREAM CONTRACTS (not enforced here — for the trainer/serving that consume this):
      * ALIGNMENT: ``out[j]`` is the trailing window ENDING at camera frame j. A UVA-style
        model reuses this stream for BOTH force-input history (frames <= the obs anchor)
        and the future-force target (frames > the anchor); the phase mapping between frame
        index and action step lives in the trainer/sampler and is an off-by-one landmine —
        pin it with a test that force-target frame j aligns to action step j.
      * TARE TRANSIENT: the F/T is auto-tared in FREE SPACE at each episode START and the
        compensator settles over ~0.5 s (~6 camera frames at ~11.8 Hz). Those first frames
        are real-sample-covered (not padded) but are a converging-tare transient — the
        trainer should down-weight/mask the first ~0.5 s of force LOSS per episode.
      * DEPLOY PARITY: the window is defined in SAMPLE space (K most-recent samples), so at
        a different live F/T rate K spans a different wall-clock duration (K=16 ≈ 89 ms at
        the recorded ~178 Hz vs ~64 ms at a 250 Hz deploy buffer). Serving MUST either pin
        the deploy F/T rate to match training OR resample both sides onto a fixed-dt K grid,
        and MUST reimplement only the inner backward loop (window-confinement + the interp
        fallback are episode-scoped and have no online analog).
    """
    src_t = np.asarray(src_t, dtype=np.float64)
    wrench = np.asarray(wrench, dtype=np.float64)
    grid_t = np.asarray(grid_t, dtype=np.float64)
    T = grid_t.shape[0]
    D = wrench.shape[1] if wrench.ndim == 2 else 6

    # Guard K<=0 / empty / no-frame BEFORE any indexing so a (T,0,D) array never
    # reaches the zarr writer (a zero-width chunk trips a ZeroDivisionError there).
    if K <= 0 or T == 0 or src_t.shape[0] == 0:
        return np.zeros((T, max(K, 0), D), dtype=np.float32)

    # Window-confine to this episode's camera span (marker-split bleed guard). src_t
    # is already strictly increasing (caller runs make_strictly_increasing on ft[:,0]).
    in_win = (src_t >= grid_t[0]) & (src_t <= grid_t[-1])
    win_t = src_t[in_win]
    win_w = wrench[in_win]

    if win_t.shape[0] == 0:
        # No in-window sample at all: clamped interp from the FULL stream, held across
        # K (a real edge value rather than zero), matching antialias's empty fallback.
        if src_t.shape[0] == 1:
            held = np.repeat(wrench[0][None, :], T, axis=0)
        else:
            f = interp1d(src_t, wrench, axis=0, bounds_error=False,
                         fill_value=(wrench[0], wrench[-1]))
            held = f(np.clip(grid_t, src_t[0], src_t[-1]))
        return np.repeat(held[:, None, :], K, axis=1).astype(np.float32)

    # For each frame, the count of in-window samples AT OR BEFORE its timestamp
    # (causal). searchsorted side='right' => number of win_t <= grid_t[i].
    n_at_or_before = np.searchsorted(win_t, grid_t, side="right")  # (T,), in [0, len(win_t)]

    out = np.zeros((T, K, D), dtype=np.float64)
    n_win = win_t.shape[0]
    for i in range(T):
        end = int(n_at_or_before[i])          # exclusive end index into win_t
        if end == 0:
            # No in-window sample at/before this frame (F/T started after grid_t[i]):
            # left-hold the earliest in-window sample across the whole window.
            out[i, :] = win_w[0]
            continue
        start = max(0, end - K)               # take up to K most-recent, causal
        seg = win_w[start:end]                # (m, D), m <= K, chronological
        m = seg.shape[0]
        out[i, K - m:] = seg                  # right-align: newest at slot K-1
        if m < K:
            out[i, :K - m] = seg[0]           # left-pad by repeating the oldest real sample
    return out.astype(np.float32)


# ---------------------------------------------------------------------------
# HL-v2: fixed-dt UNIFORM grids (the contract the S1 high-level policy trains on)
# ---------------------------------------------------------------------------
# WHY a second grid exists alongside the camera grid.
#
# The camera grid is the master timeline, but it is NOT uniform: measured
# camera_dt_uniform=False, dt median 97.2 ms / mean 84.0 ms, p5..p95 66..101 ms,
# 48% outliers. Any window defined as "N camera frames" therefore spans a variable
# wall-clock duration — 16 frames is 1.06 s to 1.61 s depending on where it lands.
# The HL policy contract is defined in SECONDS ("1 s of history, 1 s of future"), so
# every model-facing quantity is resampled onto an explicit fixed-dt grid anchored
# at a camera frame. That makes the offline window and the online window the SAME
# object, which is what lets a deploy-time force buffer be a faithful mirror of
# training instead of a rate-dependent approximation (the pre-v2 hires window was
# defined in SAMPLE space and silently spanned 89 ms at 178 Hz vs 64 ms at 250 Hz).
#
# Two grids are emitted per episode, both anchored on the episode's camera frames:
#
#   1. A PAST grid, for the force HISTORY input: for camera frame i, the
#      FORCE_HIST_LEN bins tiling [t_i - span, t_i], causal (bin -1 ends AT t_i).
#      Written as [T, L, 6] by :func:`resample_wrench_uniform_past`.
#   2. A FUTURE grid, for the force/pose/gripper TARGETS: the H step times
#      t_i + k*dt, k=0..H-1, i.e. the half-open interval [t_i, t_i + span).
#      :func:`future_step_times` produces those times; the pose/gripper resamplers
#      already accept arbitrary query times, and
#      :func:`resample_wrench_uniform_future` bins the wrench onto them.
#
# T0-INCLUSIVE convention (DECISION-T0, UMI-on-Air parity): the future grid INCLUDES
# k=0 at t_i itself — the action AT the observation instant is the first predicted
# step, so the chunk is a strict shift of the observation rather than starting one dt
# ahead. In anchor-relative space that step-0 POSE row is the identity (it teaches
# nothing on its own), but the step-0 FORCE and GRIPPER carry real signal, and the
# alignment matches UMI's "current action == first action". The pre-v2 grid used
# ``arange(1, H+1)`` (the interval (t_i, t_i + span], EXCLUDING t_i) and burned no slot
# on the identity, but its chunk led the observation by one dt. A file carries
# ``attrs.hl_action_first_step`` = "t0" so a loader can tell the two grids apart — the
# shift is invisible in the values, so it must be provenance, not inferred.
#
# Both grids are ANTI-ALIASED (bin mean over the raw ~178 Hz stream), not
# point-sampled: a contact transient point-sampled at 16 Hz aliases. An empty bin never
# yields 0 (a zero wrench is indistinguishable from "no contact" and must not be
# fabricated) — the HISTORY holds the newest sample at or before the bin (a causal
# zero-order hold; interpolating would blend in the NEXT sample, which for a bin near t0
# is post-anchor), while a TARGET bin may interpolate, since looking at both neighbours
# of a future bin leaks nothing.
# CONTRACT v3 (2026-08-07): NO HISTORY, 30 Hz, 50-step chunk.
#
# The v2 contract was "1 s of force history (8 bins) -> 1 s of future (16 steps)" on a
# ~15 Hz camera. Both sides changed:
#
#   * FORCE_HIST_LEN 8 -> 1. The observation is the SINGLE anchor frame t0: the one
#     FORCE_HIST_DT-wide causal bin ENDING at t0. Not zero — the wiping task needs the
#     model to feel present contact — but no look-back. Every history mechanism
#     (causal binning, bin-exact occupancy, episode confinement, tare masking) is
#     UNCHANGED and simply operates on one bin, so the "was this measured" guarantee
#     that separates a real tared zero from a stale hold still holds. The span is now
#     the bin width itself: 1/30 s, matching the action grid, so the input force is
#     the same 33 ms average the first action step is scored against.
#   * ACTION_HORIZON 16 -> 50 and ACTION_SPAN_S 1.0 -> 5/3. 50 steps at 30 Hz spans
#     1.6667 s. ACTION_HZ is now 30.0 == the camera publish rate
#     (CAMERA_PUBLISH_HZ=30), so ACTION_DT (33.333 ms) matches the measured camera dt
#     (33.341 ms) to 0.024% — the action grid and the camera grid are the SAME grid to
#     within a quarter of a per-mille, which is why "nearest recorded sample" needs no
#     interpolation to be exact. (The resamplers are unchanged and still anti-alias;
#     at this dt each bin holds ~7.7 raw F/T samples and ~2.7 tracker samples, so bin
#     means remain well-defined rather than degenerating to point sampling.)
#
# The grid is still defined in SECONDS and resampled, NOT as a count of camera frames.
# That is load-bearing even now that the two rates coincide: a dropped camera frame or
# a session recorded at a different CAMERA_PUBLISH_HZ must not silently rescale the
# window, and the offline window must stay the same object as the online one.
# THE GRID IS MUTABLE STATE, AND THAT IS WHY IT IS NOT A SET OF LOOSE CONSTANTS.
# `convert_to_umi.py` re-pins the grid from CLI flags at runtime. Python captures
# default arguments at DEF time, so a function written as
#     def f(..., horizon: "int | None" = None)
# would keep the import-time 50 forever and silently IGNORE `--hl-action-horizon 16`
# (verified: rebinding the module attribute leaves the captured default at 50). So the
# grid lives in ONE mutable object, `GRID`, and every resampler/validator defaults its
# grid parameter to None and reads `GRID` at CALL time. A new call site that forgets to
# pass a grid therefore gets the CURRENT grid, not a stale snapshot.
@dataclass
class HLGrid:
    """The HL window contract: N causal input bins ending at t0, H output steps.

    Spans are DERIVED from (count, rate), never pinned separately — a span literal
    beside a rate literal is how the two drift apart.
    """

    force_hist_len: int = 1     # input bins: t0 only (was 8 over 1 s)
    action_horizon: int = 50    # output steps (was 16)
    action_hz: float = 30.0     # == CAMERA_PUBLISH_HZ=30; obs and action grids coincide
    #: Total look-back of the force input. ``None`` (the default) derives it as
    #: ``force_hist_len * action_dt``, i.e. each input bin is exactly ONE action step
    #: wide — so the force the model SEES at t0 is the same 33 ms average the first
    #: predicted step is scored against. It is a SEPARATE knob rather than always
    #: derived because the v2 contract was 8 bins over 1.0 s (dt 0.125 s), which is NOT
    #: 8 * action_dt at any action rate — deriving it unconditionally silently
    #: reproduced v2 at half its span.
    force_hist_span_s: "float | None" = None

    def __post_init__(self) -> None:
        if self.force_hist_len < 0:
            raise ValueError(f"force_hist_len must be >= 0, got {self.force_hist_len}")
        if self.action_horizon < 1:
            raise ValueError(f"action_horizon must be >= 1, got {self.action_horizon}")
        if not (math.isfinite(self.action_hz) and self.action_hz > 0):
            raise ValueError(f"action_hz must be a finite positive rate, got {self.action_hz}")
        if self.force_hist_span_s is not None and not (
                math.isfinite(self.force_hist_span_s) and self.force_hist_span_s > 0):
            raise ValueError("force_hist_span_s must be a finite positive duration or None, "
                             f"got {self.force_hist_span_s}")

    @property
    def action_dt(self) -> float:
        return 1.0 / float(self.action_hz)

    @property
    def action_span_s(self) -> float:
        return int(self.action_horizon) * self.action_dt

    @property
    def hist_span_s(self) -> float:
        """Resolved force look-back: explicit if given, else one action step per bin."""
        if self.force_hist_span_s is not None:
            return float(self.force_hist_span_s)
        return int(self.force_hist_len) * self.action_dt

    @property
    def force_hist_dt(self) -> float:
        if self.force_hist_len < 1:
            return self.action_dt
        return self.hist_span_s / int(self.force_hist_len)


#: The live grid. Rebound by ``convert_to_umi.main`` from ``--hl-*`` flags; read at call
#: time by every window function below. Mirrored in flore/s1_policy/data/layout.py, and a
#: mismatch is caught by a FLoRe-side test (test_hl_contract_matches_converter).
GRID = HLGrid()

#: The six grid quantities are ALSO exposed as module attributes, because callers and
#: tests read ``core.ACTION_DT`` etc. directly. They are served by ``__getattr__`` (a
#: module-level hook, PEP 562) rather than assigned, so they always reflect the CURRENT
#: ``GRID`` — an assigned constant would go stale the moment the CLI re-pins the grid,
#: which is the exact bug this indirection exists to prevent.
_GRID_ALIASES = {
    "FORCE_HIST_LEN": "force_hist_len",
    # NOTE: the RESOLVED span (``hist_span_s``), not the raw optional field — the field
    # is None by default and callers/attrs need the number the windows actually used.
    "FORCE_HIST_SPAN_S": "hist_span_s",
    "FORCE_HIST_DT": "force_hist_dt",
    "ACTION_HORIZON": "action_horizon",
    "ACTION_SPAN_S": "action_span_s",
    "ACTION_HZ": "action_hz",
    "ACTION_DT": "action_dt",
}


def __getattr__(name: str):
    """Serve the grid aliases off the live ``GRID`` (PEP 562)."""
    field = _GRID_ALIASES.get(name)
    if field is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(GRID, field)


def set_grid(force_hist_len: int, action_horizon: int, action_hz: float,
             force_hist_span_s: "float | None" = None) -> HLGrid:
    """Re-pin the live HL grid and return it. Validated by ``HLGrid.__post_init__``."""
    global GRID
    GRID = HLGrid(force_hist_len=int(force_hist_len),
                  action_horizon=int(action_horizon),
                  action_hz=float(action_hz),
                  force_hist_span_s=(None if force_hist_span_s is None
                                     else float(force_hist_span_s)))
    return GRID


def _future_grid(horizon: "int | None", span_s: "float | None") -> "tuple[int, float]":
    """Resolve a (horizon, span) pair against the LIVE grid. ``None`` means "the grid".

    Resolving here — at call time — rather than in a default argument is what makes
    ``--hl-action-horizon`` reach every resampler; see the GRID note above.
    """
    h = int(GRID.action_horizon if horizon is None else horizon)
    if h <= 0:
        raise ValueError(f"horizon must be >= 1, got {h}")
    # A span given WITHOUT a horizon (or vice versa) is honoured as passed: the tests
    # exercise odd spans directly. Only an omitted value falls back to the grid.
    s = float(GRID.action_span_s if span_s is None else span_s)
    if not (math.isfinite(s) and s > 0):
        raise ValueError(f"span_s must be a finite positive duration, got {s}")
    return h, s


def _past_grid(n_bins: "int | None", span_s: "float | None") -> "tuple[int, float]":
    """Resolve an (n_bins, span) pair for the causal history against the LIVE grid."""
    n = int(GRID.force_hist_len if n_bins is None else n_bins)
    if n <= 0:
        raise ValueError(f"n_bins must be >= 1, got {n}")
    s = float(GRID.hist_span_s if span_s is None else span_s)
    if not (math.isfinite(s) and s > 0):
        raise ValueError(f"span_s must be a finite positive duration, got {s}")
    return n, s


def future_step_times(anchor_t: np.ndarray, horizon: "int | None" = None,
                      span_s: "float | None" = None) -> np.ndarray:
    """Per-anchor action step times: ``anchor_t[i] + k*span/horizon``, k=0..H-1.

    Returns ``[T, horizon]`` float64 ABSOLUTE times. Step 0 lands ON the anchor
    (``t0``) and the last step lands at ``t0 + (H-1)*dt`` = ``t0 + span_s - dt``,
    i.e. the CLOSED-OPEN interval ``[anchor_t[i], anchor_t[i] + span_s)``.

    T0-INCLUSIVE convention (UMI-on-Air parity, DECISION-T0): step 0 is the action
    AT the observation instant. In anchor-relative space that pose row is the
    IDENTITY and teaches nothing on its own, but force/gripper at t0 carry real
    signal, and matching UMI's "current action is the first action" makes the chunk
    a strict shift of the observation rather than starting one dt in the future. The
    pre-v2 grid used ``arange(1, H+1)`` (the half-open ``(t0, t0+span]``), excluding
    t0; that burned no slot on the identity but broke UMI parity — see the module
    note above and ``resample_wrench_uniform_future``.

    These times may run past the episode's last camera frame; the caller decides
    what that means (this repo clamps the resample and flags the steps as padding,
    exactly as the pre-v2 frame-index chunk did).
    """
    horizon, span_s = _future_grid(horizon, span_s)
    anchor_t = np.asarray(anchor_t, dtype=np.float64).reshape(-1)
    dt = float(span_s) / int(horizon)
    offs = dt * np.arange(0, horizon, dtype=np.float64)      # (H,), step 0 == anchor
    return anchor_t[:, None] + offs[None, :]                 # (T, H)


def past_bin_edges(anchor_t: np.ndarray, n_bins: "int | None" = None,
                   span_s: "float | None" = None) -> np.ndarray:
    """Edges of the causal history bins: ``[T, n_bins+1]``, last edge == anchor.

    Bin ``j`` covers ``(edges[i,j], edges[i,j+1]]`` so bin ``n_bins-1`` ENDS exactly
    at the anchor timestamp and no bin contains a sample from the future.
    """
    n_bins, span_s = _past_grid(n_bins, span_s)
    anchor_t = np.asarray(anchor_t, dtype=np.float64).reshape(-1)
    dt = float(span_s) / int(n_bins)
    offs = dt * np.arange(-int(n_bins), 1, dtype=np.float64)  # (n_bins+1,), ends at 0
    return anchor_t[:, None] + offs[None, :]


def _bin_mean_with_interp_fallback(src_t: np.ndarray, values: np.ndarray,
                                   edges: np.ndarray, centers: np.ndarray,
                                   causal: bool = False) -> np.ndarray:
    """Anti-aliased bin means of ``values`` over ``edges``, with a fallback for
    empty bins. Returns ``[T, B, D]``.

    ``edges`` is ``[T, B+1]`` (monotone per row), ``centers`` is ``[T, B]`` (the
    representative time of each bin, used only by the fallback).

    Bin ``(i, j)`` is the mean of every ``src_t in (edges[i,j], edges[i,j+1]]``.
    An EMPTY bin — which happens when the bin is shorter than the raw sample period,
    when the sensor drops out, or when the bin lies outside the recorded span — takes
    a fallback value. It NEVER yields zero: a zero wrench reads as "no contact"
    downstream, so fabricating one would inject a false measurement.

    ``causal`` selects the fallback, and it is load-bearing:

      * ``causal=False`` (targets): clamped LINEAR INTERPOLATION at the bin centre —
        the most accurate estimate, and legitimate for a label.
      * ``causal=True`` (the force-history INPUT): a zero-order HOLD of the most
        recent sample at or before the bin's RIGHT EDGE. Interpolation is forbidden
        here because it blends the neighbouring sample on each side, and for a bin
        near the anchor the right-hand neighbour lies in the FUTURE — which would
        leak post-anchor information into an input feature. Demonstrated: with
        samples at t=9.0,9.4,9.8,10.2,10.6, anchor 10.0 and the post-anchor samples
        set to 1e6, the interpolating fallback returned 3.4e5 in the last history
        bin. A zero-order hold returns the t=9.8 value, as it must.

    When a causal bin has NO sample at or before it (it precedes the whole stream),
    the earliest sample is held — the only real value available. Such a window is
    reported unusable by :func:`force_history_valid` on coverage grounds, so it is
    masked out of training rather than silently trusted.
    """
    src_t = np.asarray(src_t, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    edges = np.asarray(edges, dtype=np.float64)
    centers = np.asarray(centers, dtype=np.float64)
    T, B = centers.shape
    D = values.shape[1] if values.ndim == 2 else 1
    values = values.reshape(values.shape[0], D)

    out = np.zeros((T, B, D), dtype=np.float64)
    if src_t.shape[0] == 0:
        # No raw stream at all. The caller MUST mark these invalid; we return zeros
        # because there is no real value to hold.
        return out.astype(np.float32)

    # Prefix sums let every bin mean be computed in O(1) from two searchsorted
    # lookups, so the whole [T, B] grid costs one vectorized pass instead of T*B
    # slices (T ~ 220/episode, B in {8, 16} -> ~3.5k bins/episode).
    csum = np.concatenate([np.zeros((1, D)), np.cumsum(values, axis=0)], axis=0)  # (N+1, D)
    ccnt = np.arange(src_t.shape[0] + 1, dtype=np.float64)                        # (N+1,)

    # side='right' on the LEFT edge and 'right' on the RIGHT edge gives the
    # half-open (lo, hi] convention, so a sample exactly AT the anchor lands in the
    # last past bin and never in a future bin.
    lo = np.searchsorted(src_t, edges[:, :-1], side="right")   # (T, B)
    hi = np.searchsorted(src_t, edges[:, 1:], side="right")    # (T, B)
    cnt = (ccnt[hi] - ccnt[lo])                                # (T, B)
    filled = cnt > 0
    with np.errstate(invalid="ignore", divide="ignore"):
        sums = csum[hi] - csum[lo]                             # (T, B, D)
        out = np.where(filled[..., None], sums / np.maximum(cnt, 1.0)[..., None], 0.0)

    if not filled.all():
        if src_t.shape[0] == 1:
            held = np.broadcast_to(values[0], (T, B, D))
        elif causal:
            # Zero-order hold of the newest sample at or before the bin's RIGHT edge.
            # NOT an interpolation: see the docstring — interpolating would pull in
            # the next sample, which for a bin near the anchor is in the future.
            idx = np.searchsorted(src_t, edges[:, 1:], side="right") - 1  # (T, B)
            held = values[np.clip(idx, 0, src_t.shape[0] - 1)]            # (T, B, D)
        else:
            f = interp1d(src_t, values, axis=0, bounds_error=False,
                         fill_value=(values[0], values[-1]))
            held = f(np.clip(centers, src_t[0], src_t[-1]))    # (T, B, D)
        out = np.where(filled[..., None], out, held)
    return out.astype(np.float32)


def resample_wrench_uniform_past(src_t: np.ndarray, wrench: np.ndarray,
                                 anchor_t: np.ndarray,
                                 n_bins: "int | None" = None,
                                 span_s: "float | None" = None,
                                 episode_start_t: "float | None" = None) -> np.ndarray:
    """CAUSAL fixed-dt force HISTORY per anchor: ``[T, n_bins, 6]`` float32.

    For camera frame ``i``, bin ``j`` is the anti-aliased mean wrench over
    ``(t_i - span + j*dt, t_i - span + (j+1)*dt]``, so ``out[i, -1]`` is the
    ``dt``-wide bin ENDING at the frame timestamp and ``out[i, 0]`` the oldest.
    Strictly causal: nothing after ``t_i`` can enter any bin.

    ``episode_start_t``, when given, CONFINES the window: no sample from before it may
    enter any bin. Pass the episode's first camera timestamp whenever the take was split
    at markers, because a marker boundary IS a demonstration boundary — without the
    confinement an anchor just after a marker inherits the PREVIOUS demo's contact (
    measured: 7 of 8 input bins carrying the prior episode's sustained wrench), and the
    policy learns "fresh scene + sustained contact -> do X" from a recording artifact.
    Bins left empty by the confinement are holds, and
    :func:`force_history_valid` — given the same ``episode_start_t`` — reports the
    window unusable, so those anchors are masked out rather than silently trusted.

    ``src_t`` must be strictly increasing (the caller runs ``make_strictly_increasing``).
    """
    src_t = np.asarray(src_t, dtype=np.float64)
    wrench = np.asarray(wrench, dtype=np.float64)
    if episode_start_t is not None and src_t.shape[0]:
        # Drop pre-episode samples outright, so neither the bin means nor the causal
        # hold can reach across a demonstration boundary.
        keep = src_t >= float(episode_start_t)
        src_t, wrench = src_t[keep], wrench[keep]
    edges = past_bin_edges(anchor_t, n_bins, span_s)               # (T, B+1)
    centers = 0.5 * (edges[:, :-1] + edges[:, 1:])                 # (T, B)
    # causal=True: an empty bin holds the newest sample at or before it, never an
    # interpolation (which would blend in the NEXT sample — post-anchor for the last
    # bins, i.e. label leakage into an input feature).
    return _bin_mean_with_interp_fallback(src_t, wrench, edges, centers, causal=True)


def resample_wrench_uniform_future(src_t: np.ndarray, wrench: np.ndarray,
                                   anchor_t: np.ndarray,
                                   horizon: "int | None" = None,
                                   span_s: "float | None" = None) -> np.ndarray:
    """Fixed-dt force TARGETS per anchor: ``[T, horizon, 6]`` float32.

    Step ``k`` is the anti-aliased mean wrench over the ``dt``-wide bin ENDING at
    ``future_step_times(anchor)[k]``, i.e. over ``(t_i + (k-1)*dt, t_i + k*dt]``.
    That makes force step ``k`` and action step ``k`` describe the SAME time bin,
    which is the alignment a force-predicting policy needs (the pre-v2 comment in
    :func:`resample_wrench_hires` flagged this exact phase mapping as an off-by-one
    landmine; here it is fixed by construction and pinned by a test).

    T0-INCLUSIVE (DECISION-T0): step 0 lands on the anchor, so its bin is
    ``(t_i - dt, t_i]`` — the most recent ``dt`` of wrench, i.e. the current contact.
    Unlike the step-0 POSE (an identity in anchor-relative space), this is a real,
    supervised measurement.

    Steps past the end of the recorded stream hold the last real value (clamped
    interp). The caller marks those steps as padding.
    """
    horizon, span_s = _future_grid(horizon, span_s)
    times = future_step_times(anchor_t, horizon, span_s)            # (T, H)
    dt = float(span_s) / int(horizon)
    edges = np.concatenate([times[:, :1] - dt, times], axis=1)     # (T, H+1)
    centers = times - 0.5 * dt
    return _bin_mean_with_interp_fallback(src_t, wrench, edges, centers)


def resample_pose_uniform_future(tracker: np.ndarray, anchor_t: np.ndarray,
                                 T_tracker_tcp: "np.ndarray | None",
                                 horizon: "int | None" = None,
                                 span_s: "float | None" = None) -> np.ndarray:
    """Absolute pose TARGETS on the fixed-dt future grid: ``[T, horizon, 6]`` float64.

    ``tracker`` is a sanitized raw tracker array (``TRACKER_COLS``: t, seq, xyz,
    wxyz, stale) — NOT the camera-grid pose. Resampling the targets from the RAW
    ~77 Hz stream matters: the camera grid is only ~11.9 Hz, so deriving 16 Hz
    targets from it would UPSAMPLE from the slower signal and invent motion that the
    tracker actually measured. Position is linearly interpolated and rotation
    Slerped (never component-wise on quaternions), then the fixed tracker->TCP
    extrinsic is composed — the same three operations, in the same order, as the
    camera-grid pose path, so a target and an anchor are numerically consistent.

    Pass ``T_tracker_tcp=None`` for the head/torso puck (a body reference with no
    puck->TCP transform).

    Returns ``[..., :3] = pos``, ``[..., 3:] = rotvec``, matching :func:`pose_to_6d`.
    Query times past the tracker's span are CLAMPED (endpoint-held); the caller flags
    those steps via :func:`future_step_valid`.
    """
    times = future_step_times(anchor_t, horizon, span_s)     # (T, H)
    flat = times.reshape(-1)
    tr_t = np.asarray(tracker[:, 0], dtype=np.float64)
    pos = resample_position(tr_t, tracker[:, 2:5], flat)
    rot = resample_rotation_wxyz(tr_t, tracker[:, 5:9], flat)
    if T_tracker_tcp is not None:
        pos, rot = apply_tcp_extrinsic(pos, rot, T_tracker_tcp)
    pose6 = pose_to_6d(pos, rot)                             # (T*H, 6)
    return pose6.reshape(times.shape[0], times.shape[1], 6)


def resample_gripper_uniform_future(src_t: np.ndarray, width_m: np.ndarray,
                                    anchor_t: np.ndarray,
                                    horizon: "int | None" = None,
                                    span_s: "float | None" = None) -> np.ndarray:
    """Gripper-width TARGETS on the fixed-dt future grid: ``[T, horizon]`` float32.

    Absolute metres (UMI's sampler hard-codes a 0.08 m grasp threshold, so the SCALE
    is load-bearing). Clamped linear interp from the raw ~29 Hz encoder stream, for
    the same reason the pose targets come from the raw tracker.
    """
    times = future_step_times(anchor_t, horizon, span_s)
    out = resample_gripper_width(src_t, width_m, times.reshape(-1))   # (T*H, 1)
    return out.reshape(times.shape[0], times.shape[1]).astype(np.float32)


#: How long BEFORE a tare the wrench is untrustworthy. The Bota compensator is
#: auto-tared in FREE SPACE at each episode start; the samples carrying the STALE
#: (previous) offset are the ones preceding ``tare_wall``, because the new offset takes
#: effect at it. Measured on session_2026.07.25: mean |F| in the 300 ms BEFORE the tare
#: is 0.474 N vs 0.245 N after, with pre > post in 82/97 takes, and the offset settles
#: to the recorded ``ft_tare_wrench`` on the POST side.
#:
#: An earlier version of this span pointed FORWARD (``[tare_wall, +0.5]``), which masked
#: the clean side and left 143 pre-tare force-target bins flagged valid, carrying up to
#: 1.45 N of pure offset error against a ~0.25 N session baseline. Do not "simplify" the
#: direction back.
TARE_SETTLE_S = 0.5


def tare_transient_spans(tare_wall: "np.ndarray | None",
                         settle_s: float = TARE_SETTLE_S) -> np.ndarray:
    """``[K, 2]`` (start, end) wall-clock spans where the wrench is tare-contaminated.

    Each span is ``[tare_wall - settle_s, tare_wall]``: BACKWARD from the tare, since
    the corrupted samples are the ones still carrying the previous offset (see
    :data:`TARE_SETTLE_S` for the measurement).

    ``tare_wall`` is the recorder's ``tare_events.npy`` first column, or the single
    ``take_meta.ft_tare_wall``. Empty/absent/non-finite input -> ``[0, 2]``.

    This exists because the recorded tare provenance was written for every take and
    then never read by the converter, so the transient reached the training label as if
    it were a measurement. A free-space tare converges toward ~0, which is exactly what
    "no contact" looks like, so the contamination is invisible in the values.
    """
    if tare_wall is None:
        return np.zeros((0, 2), dtype=np.float64)
    t = np.asarray(tare_wall, dtype=np.float64).reshape(-1)
    t = t[np.isfinite(t)]
    if t.size == 0:
        return np.zeros((0, 2), dtype=np.float64)
    return np.stack([t - float(settle_s), t], axis=1)


def window_overlaps_spans(win_lo: np.ndarray, win_hi: np.ndarray,
                          spans: np.ndarray) -> np.ndarray:
    """True where the window ``[win_lo, win_hi]`` intersects ANY row of ``spans``.

    ``win_lo``/``win_hi`` broadcast to a common shape; the result has that shape.
    Used to invalidate every force window that touches a tare transient — a window
    is only as trustworthy as its worst sample, so overlap (not containment) is the
    right test.
    """
    win_lo = np.asarray(win_lo, dtype=np.float64)
    win_hi = np.asarray(win_hi, dtype=np.float64)
    hit = np.zeros(np.broadcast(win_lo, win_hi).shape, dtype=bool)
    for lo, hi in np.asarray(spans, dtype=np.float64).reshape(-1, 2):
        hit |= (win_lo <= hi) & (win_hi >= lo)
    return hit


#: Peak plausible hand speed (m/s) for a target trajectory. Used to reject implausible
#: TARGET steps that survive the tracker's own glitch handling: deglitch BRIDGES an
#: isolated teleport by dropping+reinterpolating it, which repairs the anchor label but
#: leaves a fast ramp in the resampled trajectory that the per-anchor glitch mask does
#: not cover.
#:
#: MEASURED, not assumed. The old value (4.0) was reasoned from free-hand peak speed and
#: was DEAD CODE on real data: on clean94 it flagged 0 of 634,928 supervised steps while
#: five episodes carried 118-201 mm one-frame ramps. The gate has to be set against the
#: CAMERA-GRID step distribution it actually screens, which is far tighter than free-hand
#: speed because the label is a 15 Hz resample of a slow manipulation:
#:
#:   89 known-clean clean94 episodes (37,350 steps): p99 0.088, p99.9 0.138, MAX 0.506 m/s
#:   the 5 glitch-affected episodes:                 residual ramps 1.095 .. 3.010 m/s
#:
#: So there is a clean 2.2x gap between the fastest honest step ever recorded and the
#: slowest glitch residual. 1.0 m/s sits in that gap: it flags all 9 residual ramps with
#: ZERO false positives on the clean 89. Raising it back toward free-hand speed reopens
#: the hole (1.5 misses 3 ramps, 2.0 misses 6, 4.0 misses all 9); lowering it below
#: ~0.6 would start eating real fast reaches. Re-measure this if the task changes to
#: something genuinely faster than wiping.
MAX_TARGET_SPEED_MPS = 1.0


def implausible_step_mask(pose6: np.ndarray, dt: float,
                          max_speed: float = MAX_TARGET_SPEED_MPS) -> np.ndarray:
    """``[T, H]`` bool: True where a pose target step implies an impossible speed.

    ``pose6`` is ``[T, H, 6]`` (pos + rotvec) on a fixed-dt grid. A step is flagged when
    the distance from the PREVIOUS step (or, for step 0, nothing — it has no predecessor
    on the grid) exceeds ``max_speed * dt``. Both endpoints of an offending interval are
    flagged, because either one may be the corrupted sample.

    This is the last line of defence for pose labels: the take-level gate drops
    systemically-broken tracking and the per-anchor ``robot0_pose_glitch`` marks
    unbridged excursions, but a BRIDGED glitch leaves a physically impossible ramp in the
    interpolated target trajectory with nothing flagging it. Measured on the shipped
    session, 84 of 10,067 supervised steps across four glitch-affected takes implied
    >3.2 m/s before this mask.
    """
    pos = np.asarray(pose6, dtype=np.float64)[..., :3]
    if pos.shape[1] < 2:
        return np.zeros(pos.shape[:2], dtype=bool)
    step = np.linalg.norm(np.diff(pos, axis=1), axis=2)      # (T, H-1)
    bad = step > (float(max_speed) * float(dt))
    out = np.zeros(pos.shape[:2], dtype=bool)
    out[:, 1:] |= bad                                        # the arriving endpoint
    out[:, :-1] |= bad                                       # ...and the departing one
    return out


def force_history_valid(src_t: np.ndarray, anchor_t: np.ndarray,
                        span_s: "float | None" = None,
                        min_coverage: float = 0.75,
                        tare_spans: "np.ndarray | None" = None,
                        max_unmeasured_bins: int = 0,
                        episode_start_t: "float | None" = None,
                        n_bins: "int | None" = None) -> np.ndarray:
    """Per-anchor validity of the force HISTORY window: ``[T]`` bool.

    True when the raw F/T stream really measured the window ``[t_i - span, t_i]``.
    This is the mask that separates "the sensor measured no contact" (a real,
    trainable zero) from "we had no measurement and held a stale value" (not
    trainable). It is a REQUIRED companion to
    :func:`resample_wrench_uniform_past`, because a free-space-tared wrench is ≈0 by
    construction, so an all-zero test cannot distinguish the cases.

    THREE independent ways a window fails, all checked:

    1. **Span coverage** — the overlap of the window with the recorded span
       ``[src_t[0], src_t[-1]]`` must be at least ``min_coverage`` of it.
    2. **Bin-exact occupancy** — at most ``max_unmeasured_bins`` of the
       ``FORCE_HIST_LEN`` bins may be EMPTY (default 0: every bin must contain a real
       sample). This is the direct statement of "was this window measured", and it
       subsumes both failure shapes that span-coverage alone lets through:

         * a mid-window sensor dropout (samples exist on both sides, so the span test
           passes, but the bins inside the hole are stale holds), and
         * an episode-start anchor whose leading bins precede the stream entirely and
           are backfilled from the earliest sample.

       Both cases are silent in the VALUES, because a free-space-tared wrench is ≈0
       and a hold is indistinguishable from a real steady reading.
    3. **Tare transient** — the window must not touch ``tare_spans``
       (``[K, 2]`` from :func:`tare_transient_spans`).

    ``src_t`` must be strictly increasing.
    """
    n_bins, span_s = _past_grid(n_bins, span_s)
    anchor_t = np.asarray(anchor_t, dtype=np.float64).reshape(-1)
    src_t = np.asarray(src_t, dtype=np.float64)
    if episode_start_t is not None and src_t.shape[0]:
        # Must mirror resample_wrench_uniform_past's confinement EXACTLY, or the mask
        # would describe a window the resampler did not build.
        src_t = src_t[src_t >= float(episode_start_t)]
    if src_t.shape[0] == 0:
        return np.zeros(anchor_t.shape[0], dtype=bool)
    lo = anchor_t - float(span_s)

    # (1) span coverage
    overlap = np.minimum(anchor_t, src_t[-1]) - np.maximum(lo, src_t[0])
    ok = (overlap / float(span_s)) >= float(min_coverage)

    # (2) BIN-EXACT occupancy. This is the direct statement of "was this window
    # measured": count the bins that contain at least one raw sample, and require all
    # but ``max_unmeasured_bins`` of them. It supersedes an interior-gap heuristic
    # because it is exactly the set of bins whose value came from a hold rather than a
    # measurement — including the leading bins of an episode-start anchor, which a
    # span-coverage test passes (samples DO exist later in the window) while bin 0 is
    # entirely before the stream and is backfilled from the earliest sample.
    edges = past_bin_edges(anchor_t, n_bins, span_s)            # (T, B+1)
    lo_i = np.searchsorted(src_t, edges[:, :-1], side="right")
    hi_i = np.searchsorted(src_t, edges[:, 1:], side="right")
    n_unmeasured = ((hi_i - lo_i) == 0).sum(axis=1)
    ok &= n_unmeasured <= int(max_unmeasured_bins)

    # (3) tare transient
    if tare_spans is not None and np.asarray(tare_spans).size:
        ok &= ~window_overlaps_spans(lo, anchor_t, tare_spans)
    return ok


def future_step_valid(src_t: np.ndarray, anchor_t: np.ndarray, episode_end_t: float,
                      horizon: "int | None" = None,
                      span_s: "float | None" = None,
                      tare_spans: "np.ndarray | None" = None,
                      max_gap_s: "float | None" = None) -> np.ndarray:
    """Per-anchor, per-step validity of the FUTURE window: ``[T, horizon]`` bool.

    A step is valid when (a) its time is at or before the episode's last camera frame
    (``episode_end_t``, so a chunk never predicts across a boundary into the next
    demonstration), (b) it is at or before the source stream's last sample, and (c) its
    own ``dt``-wide bin CONTAINS a sample — i.e. it is not inside a mid-stream dropout,
    which the resamplers silently interpolate across. Invalid steps are the v2
    equivalent of the pre-v2 ``action_is_pad``: the loss must exclude them rather than
    train toward an interpolated or held value.

    EVERY pose/gripper caller passes a REAL ``src_t`` so terms (b) and (c) engage; pass
    ``src_t=np.zeros(0)`` ONLY for the shared episode-bound mask that is then AND-ed with
    each stream's own coverage. Do not "simplify" a caller to the empty form: dropping the
    coverage terms re-admits 1,840 measured pose-target steps that sit inside a >0.25 s
    tracker gap, which the resampler interpolates a straight line across.
    ``max_gap_s`` defaults to one step (``span_s/horizon``).

    Pose/gripper padding uses the ``episode_end_t`` term only; force padding needs
    both, which is why they are returned together and the caller AND-s in whichever
    terms apply to each quantity.

    ``tare_spans`` (``[K, 2]``, from :func:`tare_transient_spans`) additionally
    invalidates any step whose own ``dt``-wide bin touches a tare transient. This
    matters for the FIRST steps of an episode-start anchor, whose future bins sit
    inside the settling window.
    """
    horizon, span_s = _future_grid(horizon, span_s)
    times = future_step_times(anchor_t, horizon, span_s)            # (T, H)
    dt = float(span_s) / int(horizon)
    ok = times <= float(episode_end_t) + 1e-9
    src_t = np.asarray(src_t, dtype=np.float64)
    if src_t.shape[0] > 0:
        # Tail coverage: nothing past the last sample.
        ok &= times <= src_t[-1] + 1e-9
        # INTERIOR coverage: a step whose own dt-wide bin contains no sample sits in a
        # dropout, and the resampler interpolates a straight line / Slerp across it.
        # Checking only the tail missed 1840 real pose-target steps on the shipped
        # session (sanitized-tracker gaps up to 0.40 s, from genuine dropouts and from
        # deglitch_tracker_positions dropping up to 68 consecutive rows). This is the
        # same failure shape force_history_valid's bin-exact test catches.
        if max_gap_s is None:
            max_gap_s = dt
        lo_i = np.searchsorted(src_t, times - float(max_gap_s), side="right")
        hi_i = np.searchsorted(src_t, times, side="right")
        ok &= (hi_i - lo_i) > 0
    if tare_spans is not None and np.asarray(tare_spans).size:
        ok &= ~window_overlaps_spans(times - dt, times, tare_spans)
    return ok


# ---------------------------------------------------------------------------
# Image transform geometry (UMI: aspect-preserving resize + CENTER CROP)
# ---------------------------------------------------------------------------
@dataclass
class CropResizeParams:
    """Parameters for UMI's aspect-preserving resize + center crop.

    UMI does NOT naively squash to the output square. For input (in_w,in_h) ->
    output (out_w,out_h): scale so the SMALLER output ratio is met (cover), then
    center-crop the overflow. This keeps geometry undistorted at the cost of
    cropping the longer axis. Mirrors HuMI/UMI ``get_image_transform``.
    """

    resize_w: int
    resize_h: int
    crop_x: int
    crop_y: int
    out_w: int
    out_h: int


def compute_crop_resize(in_w: int, in_h: int, out_w: int, out_h: int) -> CropResizeParams:
    """Compute resize+center-crop params (cover fit). Pure integer geometry, so
    it's testable without cv2."""
    # Scale to COVER the output box (so both dims >= output), then center-crop.
    scale = max(out_w / in_w, out_h / in_h)
    resize_w = int(np.ceil(in_w * scale))
    resize_h = int(np.ceil(in_h * scale))
    # Guard against rounding leaving a dim 1px short of the output.
    resize_w = max(resize_w, out_w)
    resize_h = max(resize_h, out_h)
    crop_x = (resize_w - out_w) // 2
    crop_y = (resize_h - out_h) // 2
    return CropResizeParams(resize_w, resize_h, crop_x, crop_y, out_w, out_h)


# ---------------------------------------------------------------------------
# Episode validation (advisory; mirrors HuMI's stop-time checks)
# ---------------------------------------------------------------------------
def validate_episode_timeline(cam_t: np.ndarray, tracker_t: np.ndarray,
                              tracker_pos: np.ndarray,
                              time_jump_s: float = 0.1,
                              pos_jump_m: float = 0.1) -> dict:
    """Advisory continuity checks (NOT a hard gate). Reports camera-frame time
    gaps, tracker time gaps, and tracker position jumps above thresholds, plus
    whether the tracker span covers the camera span (so resampling won't be all
    edge-extrapolation)."""
    report: dict = {"ok": True, "warnings": []}
    cam_t = np.asarray(cam_t, dtype=np.float64)
    tr_t = np.asarray(tracker_t, dtype=np.float64)
    tr_p = np.asarray(tracker_pos, dtype=np.float64)

    if cam_t.shape[0] >= 2:
        dt = np.diff(cam_t)
        big = np.nonzero(dt > time_jump_s)[0]
        for i in big:
            report["warnings"].append(f"camera time jump {dt[i]:.3f}s at frame {i}->{i+1}")
    if tr_t.shape[0] >= 2:
        dt = np.diff(tr_t)
        big = np.nonzero(dt > time_jump_s)[0]
        for i in big:
            report["warnings"].append(f"tracker time jump {dt[i]:.3f}s at sample {i}->{i+1}")
    if tr_p.shape[0] >= 2:
        d = np.linalg.norm(np.diff(tr_p, axis=0), axis=1)
        big = np.nonzero(d > pos_jump_m)[0]
        for i in big:
            report["warnings"].append(f"tracker position jump {d[i]:.3f}m at sample {i}->{i+1}")
    # Coverage: does the tracker span the camera window?
    if cam_t.shape[0] and tr_t.shape[0]:
        if tr_t[0] > cam_t[0] + 1e-3 or tr_t[-1] < cam_t[-1] - 1e-3:
            report["warnings"].append(
                "tracker does not fully cover the camera window "
                f"(tracker [{tr_t[0]:.3f},{tr_t[-1]:.3f}] vs camera "
                f"[{cam_t[0]:.3f},{cam_t[-1]:.3f}]) — edges will be held constant"
            )
    report["ok"] = len(report["warnings"]) == 0
    return report


# ---------------------------------------------------------------------------
# camera-timeline uniformity verdict
# ---------------------------------------------------------------------------
def camera_grid_verdict(all_dt, take_grid_uniform=None, camera_t_source=None):
    """Judge whether a camera timeline is a usable UNIFORM grid.

    The camera timestamps are the MASTER GRID of the training dataset (every
    other modality is resampled onto them) and both training loaders index
    windows in FRAME units with no dt input -- so a non-uniform dt is
    multiplicative noise on every learned velocity and force time-derivative.

    Two ORTHOGONAL tests, because each is blind to what the other catches:

    * ``spread = (p95 - p5) / median <= 0.05`` catches a *systematically* jittery
      pacer. This is the failure that shipped 97 takes: a wall-clock publish
      throttle decimating 30 fps randomly by 2 or 3 frames -> dt alternating
      66.7 / 100.0 ms, spread 36%. Deliberately NOT ``round(dt / median)``: with
      median 97.2 ms every one of those ratios rounds to 1, the histogram
      collapses to a single bucket, and the gate reports UNIFORM on exactly the
      data it exists to catch.
    * ``outlier_frac`` (share of steps more than 20% off the median) catches rare
      but large holes -- a dropped frame, or a bridge reconnect (measured 1.6 s =
      24 publish periods). p5/p95 structurally cannot see <=5% of steps, so
      without this a 2 s mid-episode freeze passes.

    ``take_grid_uniform`` is the recorder's own verdict, propagated rather than
    re-derived: it can bucket against the true nominal CAPTURE period, which the
    converter never sees. ``False`` vetoes.

    Returns a dict of attrs (all JSON-safe) including ``camera_dt_uniform``.
    """
    import numpy as _np

    all_dt = _np.asarray(all_dt, dtype=_np.float64).reshape(-1)
    out = {}
    if camera_t_source is not None:
        out["camera_t_source"] = camera_t_source
    if all_dt.size == 0:
        out["camera_dt_uniform"] = None
        return out
    med = float(_np.median(all_dt))
    p5, p95 = (float(x) for x in _np.percentile(all_dt, [5, 95]))
    spread = (p95 - p5) / max(med, 1e-9)
    outlier_frac = float(_np.mean(_np.abs(all_dt / max(med, 1e-9) - 1.0) > 0.20))
    # HOLES: individual steps >1.5x the median. These are dropped frames or a
    # bridge reconnect (measured 1.6 s = 24 publish periods), not a pacer defect,
    # so they are reported and WARNED about separately rather than failing the
    # pacer verdict -- one dropped frame should not condemn a 75-episode session,
    # but it must not be invisible either (p5/p95 structurally cannot see it).
    holes = int((all_dt > 1.5 * max(med, 1e-9)).sum())
    out["camera_hz"] = round(1.0 / max(float(all_dt.mean()), 1e-9), 4)
    out["camera_dt_median"] = round(med, 6)
    out["camera_dt_mean"] = round(float(all_dt.mean()), 6)
    out["camera_dt_p5_p95"] = [round(p5, 6), round(p95, 6)]
    out["camera_dt_spread"] = round(float(spread), 4)
    out["camera_dt_outlier_frac"] = round(outlier_frac, 5)
    out["camera_dt_holes_n"] = holes
    out["camera_dt_max_over_median"] = round(float(all_dt.max() / max(med, 1e-9)), 3)
    if all_dt.size < 10:
        out["camera_dt_uniform"] = None      # too short to judge
        return out
    uniform = bool(spread <= 0.05 and outlier_frac <= 0.002)
    if take_grid_uniform is False:
        uniform = False
    out["camera_dt_uniform"] = uniform
    return out
