#!/usr/bin/env python
"""One-shot data-collection QA for a recorded session.

Loads every take's raw arrays directly (does NOT trust take_meta), reuses the
converter's own timeline validator, and checks the MuSe-force / K=16 hi-rate
contracts. Prints a per-take table + a session verdict. Read-only.

Usage:
    .venv/bin/python processing/audit_session.py data/recordings/session_XXX [--k 16]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import glob

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import umi_convert_core as core  # noqa: E402
# Import the converter's ACTUAL episode-drop thresholds so the audit's teleport verdict
# can never drift from the real KEEP/DROP decision (importing is side-effect-free — main()
# is under an __name__ guard). Fall back to the documented defaults if the module renames.
try:
    from convert_to_umi import (  # noqa: E402
        TRACKER_DROP_MAX_UNBRIDGED as _DROP_MAX_UNBRIDGED,
        TRACKER_DROP_RESIDUAL_JUMP_M as _DROP_RESIDUAL_JUMP_M,
    )
except Exception:  # noqa: BLE001
    _DROP_MAX_UNBRIDGED, _DROP_RESIDUAL_JUMP_M = 3, 0.30


def _load_base_calib():
    """Load the repo's gripper_range.json so width mapping keeps the physical
    width_open_m (effective_gripper_calib needs a non-None base or returns 'none')."""
    for p in ("robot_assets/calibration/gripper_range.json",
              *glob.glob("**/gripper_range.json", recursive=True)):
        if os.path.exists(p):
            import json as _json
            j = _json.load(open(p))
            return core.GripperWidthCalib(
                enc_closed=j.get("enc_closed", 0.0), enc_open=j.get("enc_open", 4.9),
                width_closed_m=j.get("width_closed_m", 0.0),
                width_open_m=j.get("width_open_m", 0.095))
    return None


_BASE_CALIB = _load_base_calib()

# ---- column layouts (verified against recording/recording_format.TRACKER_COLS) --
# ft_wrench.npy / ft_wrench_comp.npy: [N,7]  col0=wall_t, 1..6 = Fx Fy Fz Tx Ty Tz
# tracker*.npy: [N,10] TRACKER_COLS = (t, seq, x, y, z, qw, qx, qy, qz, stale):
#   col0=wall_t col1=seq(monotonic counter, NOT a valid flag) pos=2..4 quat_wxyz=5..8 col9=stale
# gripper.npy: [N,5]  col0=wall_t col1=enc_rad col2? col3=open_rad col4=closed_rad
# camera/frame_index.npy: [N,2] col0=wall_t col1=frame_number
# tare_events.npy: [M,7] col0=wall_t 1..6=tare wrench
GRASP_THRESH_M = 0.08          # UMI hard-coded grasp threshold
TARE_SETTLE_S = 0.5            # transient to mask from force loss
MIN_FRAMES = 30                # a usable episode floor (~2.5s @ 11.8Hz)
# Rotation-continuity gate (the check the position-only validator was blind to: a
# tracking re-localization can flip orientation with position barely moving). Measured
# on the training LABEL grid — the quaternion Slerped onto the camera timeline AFTER the
# SAME sanitize the converter runs (stale-drop + strictly-increasing + position deglitch)
# — because that, not the ~80 Hz native stream, is what a loader actually sees. NOTE the
# converter's deglitch is POSITION-only, so a rotation glitch with a quiet position (a
# re-localization flip) survives it into the label; this is the one the position-only
# validator misses. The DISCRIMINATOR is path/net (a there-and-back glitch traces a long
# path for ~zero net rotation), NOT absolute magnitude — so the step FLOOR is set just
# above genuine fast-wrist motion (measured genuine-motion max ~2.5 deg/frame on this
# session) and the ratio does the classifying. A step below the floor can't be a glitch
# worth flagging; above it, ratio>RATIO = glitch, else = fast rotation.
ROT_LABEL_STEP_FLOOR_DEG = 8.0     # only steps above this are candidates (>>2.5 genuine-motion max)
ROT_PATH_NET_RATIO = 3.0           # summed-path / net-rotation over a window: >this = there-and-back
ROT_GLITCH_WINDOW = 5              # camera frames each side of the worst step for the path/net test


def _load(take, name):
    p = os.path.join(take, name)
    return np.load(p) if os.path.exists(p) else None


#: Secondary keypoints (2pt/3pt/5pt) -> the zarr array a glitch on that puck corrupts.
#: 'left' is absent on purpose: it is the PRIMARY and gets the full block in
#: audit_take (rate, coverage, staleness, the KEEP/DROP gate, ...).
#:
#: The two 5pt FEET name no zarr array yet, and that is the honest label rather than a
#: placeholder key: ``convert_to_umi`` does not emit a foot keypoint (FLoRe's five-point
#: layout currently DERIVES ``wb_{left,right}_foot_*`` by FK from the joint target, and a
#: recorded foot replacing that derivation is a separate schema decision). The jump checks
#: below are pose checks and do not care — a teleporting puck is unusable data whatever it
#: is later called — so a 5pt session is audited today and the label sharpens later.
_SECONDARY_TRACKERS = (
    ("right", "tracker_right.npy", "robot1_eef_pos / robot1_eef_rot_axis_angle"),
    ("head", "tracker_head.npy", "head_pos / head_rot_axis_angle"),
    ("left_foot", "tracker_left_foot.npy", "the LEFT-foot keypoint (not yet converted)"),
    ("right_foot", "tracker_right_foot.npy", "the RIGHT-foot keypoint (not yet converted)"),
)


def _audit_secondary_tracker(r: dict, role: str, tr: np.ndarray, label_key: str,
                             cam_t: np.ndarray, nframes: int,
                             required: bool = False) -> bool:
    """Run the LABEL-CORRUPTING tracker checks on a secondary (2pt/3pt) puck.

    Returns True when this puck satisfies the no-jump requirement.

    ``audit_take``'s big tracker block only ever looked at ``tracker.npy``. The
    right-hand and chest pucks were LOADED and then never examined, so every
    lighthouse glitch on them was invisible to the one gate that decides which takes
    enter a "clean" dataset.

    Secondary pucks are in fact LESS protected than the primary, which is why this
    matters. For ``tracker.npy`` the converter computes a glitch severity, applies a
    KEEP/DROP gate, and writes a per-frame ``robot0_pose_glitch`` mask that the HL
    loader's ``drop_glitch_windows`` reads. For the secondaries it does none of that
    (``_sanitize_tracker``): it stale-drops, makes the stream monotone and runs the
    POSITION-only deglitch, then resamples. There is no ``robot1_pose_glitch`` /
    ``head_pose_glitch`` array at all — so an unbridged excursion ships an
    interpolation ramp, and a rotation glitch ships a corrupt rot6d, with NOTHING
    downstream able to mask either. The audit is the only line of defense.

    ``required`` is what the take's DECLARED MODE asks for (2pt -> left+right,
    3pt -> +head), and it decides the severity:

    * **required** -> a drop-gate-level glitch is an **ERR**, exactly like the primary.
      Recording in 2pt/3pt IS the statement that those keypoints are part of the
      dataset, so a take where any one of them teleports is unusable for it. The
      no-jump requirement is CONJUNCTIVE over the mode's roles — every role must pass,
      and `clean_roles` / `dirty_roles` record which did.
    * **not required** (a stray secondary file in a 1pt session) -> WARN only. It is
      informational: no declared dataset depends on that keypoint.

    Either way the message names the zarr key hit and that no mask exists downstream,
    because `--exclude` is the only remedy. Mirrors the primary's sanitize order
    (stale-drop -> strictly-increasing -> position deglitch) so what is measured is
    what a loader actually sees.
    """
    if tr is None or tr.shape[0] == 0 or not nframes:
        return False
    # Required roles escalate to err[]; optional ones stay advisory. Same THRESHOLDS
    # either way — only who gets told, and how loudly.
    # Since 2026-09-11 the converter MASKS a secondary role's glitches per role
    # (`<role>_pose_glitch` + `hl_<role>_action_valid`, rotation excursions dropped like
    # bridged position ones), so these findings cost WINDOWS, not the take: they are
    # advisory (warn) whether or not the role is declared. `dirty_roles` still names the
    # role so the tally shows where the windows went.
    sink = r["warn"]
    ok = True
    tr_t = tr[:, 0].astype(np.float64)
    tr_p = tr[:, 2:5].astype(np.float64)
    q = tr[:, 5:9].astype(np.float64)
    if tr.shape[0] >= 2:
        r[f"{role}_tracker_hz"] = float((tr.shape[0] - 1) / max(tr_t[-1] - tr_t[0], 1e-9))
    qn = np.linalg.norm(q, axis=1)
    if np.any(np.abs(qn - 1.0) > 0.05):
        r["warn"].append(f"{role} tracker quat norm off (min {qn.min():.3f} max {qn.max():.3f})")

    # Coverage of the camera window: a secondary that starts late / ends early is
    # EXTRAPOLATED flat by the resampler, which reads as a real, stationary keypoint.
    if tr_t[0] > cam_t[0] + 0.05:
        r["warn"].append(f"{role} tracker starts {(tr_t[0]-cam_t[0])*1000:.0f}ms AFTER frame0 "
                         f"({label_key} is held flat over the gap, indistinguishable from a "
                         f"still hand)")
    if tr_t[-1] < cam_t[-1] - 0.05:
        r["warn"].append(f"{role} tracker ends {(cam_t[-1]-tr_t[-1])*1000:.0f}ms BEFORE the last "
                         f"frame ({label_key} is held flat over the gap)")

    # Same sanitize order as the converter, so severity is measured on the kept stream.
    s_t, s_p, s_q = tr_t, tr_p, q
    st_col = 9
    if tr.shape[1] > st_col:
        live = tr[:, st_col] < 0.5
        r[f"{role}_stale_frac"] = float(1.0 - live.mean())
        if live.mean() < 0.95:
            r["warn"].append(f"{role} tracker stale {(1-live.mean())*100:.0f}%")
        if live.any():
            s_t, s_p, s_q = s_t[live], s_p[live], s_q[live]
    keep = core.make_strictly_increasing(s_t)
    s_t, s_p, s_q = s_t[keep], s_p[keep], s_q[keep]

    if s_t.shape[0] >= 3:
        sev = core.tracker_glitch_severity(s_t, s_p)
        keep_gl = core.deglitch_tracker_positions(s_t, s_p)
        s_t, s_p, s_q = s_t[keep_gl], s_p[keep_gl], s_q[keep_gl]
        r[f"{role}_unbridged"] = int(sev["n_unbridged"])
        r[f"{role}_residual_jump_mm"] = float(sev["residual_max_jump_m"] * 1000.0)
        if sev["n_unbridged"] > 0:
            # ANY unbridged excursion reaches the label unmasked (there is no
            # {role}_pose_glitch array), so unlike the primary there is no "sub-threshold
            # is recoverable" tier here — the loader cannot drop the straddling windows.
            # A required role therefore fails the no-jump gate on the first one.
            ok = False
            sink.append(
                f"{role} tracker: {sev['n_unbridged']} unbridged excursion(s), residual jump "
                f"{sev['residual_max_jump_m']*1000:.0f}mm -> a fabricated interpolation ramp in "
                f"{label_key}. The converter flags it in {role}_pose_glitch + masks the "
                f"straddling hl_{role}_action_valid steps (window-level loss, take KEPT)"
                + (f"; MODE requires {role} — --exclude only if the role is broken for most of the take"
                   if required else
                   f"; no declared keypoint depends on {role} in this mode (advisory)"))

    if s_t.shape[0] >= 2 and nframes >= 2:
        pos_lab = core.resample_position(s_t, s_p, cam_t)
        rot_lab = core.resample_rotation_wxyz(s_t, s_q, cam_t)
        dp_lab = np.linalg.norm(np.diff(pos_lab, axis=0), axis=1)
        r[f"{role}_pos_label_step_max_mm"] = float(dp_lab.max() * 1000.0) if dp_lab.size else 0.0
        rq = rot_lab.as_quat()
        dots = np.abs(np.sum(rq[:-1] * rq[1:], axis=1))
        ang_lab = np.degrees(2.0 * np.arccos(np.clip(dots, 0.0, 1.0)))
        r[f"{role}_rot_label_step_max_deg"] = float(ang_lab.max()) if ang_lab.size else 0.0
        if ang_lab.size and ang_lab.max() > ROT_LABEL_STEP_FLOOR_DEG:
            j = int(np.argmax(ang_lab))
            lo = max(0, j - ROT_GLITCH_WINDOW)
            hi = min(len(ang_lab) - 1, j + ROT_GLITCH_WINDOW)
            path = float(ang_lab[lo:hi + 1].sum())
            net = float(np.degrees(2.0 * np.arccos(
                np.clip(abs(float(np.dot(rq[lo], rq[hi + 1]))), 0.0, 1.0))))
            if path / max(net, 1e-6) > ROT_PATH_NET_RATIO:
                ok = False
                sink.append(
                    f"{role} rotation label glitch {ang_lab.max():.0f}deg/frame at frame "
                    f"{j}->{j+1} (path/net {path/max(net,1e-6):.1f}x = there-and-back); "
                    f"would reach {label_key} without the converter's rotation deglitch "
                    f"(default ON: the rows are dropped and the span masked in {role}_pose_glitch)"
                    + (f"; MODE requires {role}" if required
                       else " (advisory: no declared keypoint uses it in this mode)"))

    # Raw-stream rejoin test: the camera grid ATTENUATES an excursion that lands between
    # two frames, so this catches what the label-grid test above reads as benign (the
    # same reason the primary runs both).
    if s_t.shape[0] >= 3:
        exc = core.rotation_excursions(s_t, s_q)
        r[f"{role}_rot_excursions"] = len(exc)
        r[f"{role}_rot_excursion_amp_deg"] = max((e["amp_deg"] for e in exc), default=0.0)
        for e in exc:
            ok = False
            sink.append(
                f"{role} RAW rotation excursion {e['amp_deg']:.0f}deg peak over "
                f"{e['n_samples']} sample(s) (out-jump {e['rate_dps']:.0f} deg/s, then "
                f"REJOINS) — a lighthouse orientation glitch. The converter's rotation "
                f"deglitch drops these {e['n_samples']} row(s) and masks the span in "
                f"{role}_pose_glitch / hl_{role}_action_valid (take KEPT)"
                + (f"; MODE requires {role}" if required
                   else " (advisory: not a declared keypoint in this mode)"))
    return ok


def _primary_is_clean(r: dict, measured: bool) -> bool:
    """Did the PRIMARY puck satisfy the no-jump requirement?

    Derived from the metrics the primary block already recorded rather than from a new
    computation, so the two can never disagree. Note the asymmetry with the secondaries:
    the primary DOES get a per-frame ``robot0_pose_glitch`` mask and the HL loader drops
    the straddling windows, so a sub-threshold unbridged excursion is recoverable there
    and does not by itself fail the gate — the converter's own drop thresholds decide.

    ``measured`` must be False when the primary block did not run (no/empty
    ``tracker.npy``, or no camera frames). Absent metrics are NOT evidence of cleanliness:
    reading them through ``.get(..., 0)`` would score a take with no primary tracker at
    all as jump-free, which is the exact false pass this gate exists to prevent.
    """
    if not measured:
        return False
    if r.get("tracker_unbridged", 0) >= _DROP_MAX_UNBRIDGED:
        return False
    if r.get("tracker_residual_jump_mm", 0.0) >= _DROP_RESIDUAL_JUMP_M * 1000.0:
        return False
    return r.get("rot_excursions", 0) == 0


def audit_take(take_dir, K):
    r = {"take": os.path.basename(take_dir), "warn": [], "err": []}
    meta = {}
    mp = os.path.join(take_dir, "take_meta.json")
    if os.path.exists(mp):
        meta = json.load(open(mp))

    ft = _load(take_dir, "ft_wrench.npy")
    ftc = _load(take_dir, "ft_wrench_comp.npy")
    tr = _load(take_dir, "tracker.npy")
    trr = _load(take_dir, "tracker_right.npy")
    trh = _load(take_dir, "tracker_head.npy")
    trlf = _load(take_dir, "tracker_left_foot.npy")
    trrf = _load(take_dir, "tracker_right_foot.npy")
    grip = _load(take_dir, "gripper.npy")
    fidx = _load(take_dir, "camera/frame_index.npy")
    tare = _load(take_dir, "tare_events.npy")
    njpg = len([f for f in os.listdir(os.path.join(take_dir, "camera"))
                if f.endswith(".jpg")]) if os.path.isdir(os.path.join(take_dir, "camera")) else 0

    nframes = 0 if fidx is None else fidx.shape[0]
    r["frames"] = nframes
    r["njpg"] = njpg
    r["ft_valid"] = bool(meta.get("ft_valid", False))
    r["ft_comp"] = meta.get("ft_compensated", None)

    # --- 0. junk / too-short gate ---
    if nframes < MIN_FRAMES:
        r["err"].append(f"only {nframes} frames (<{MIN_FRAMES}) — likely aborted take")
        r["dur_s"] = 0.0
        return r

    # --- jpg vs index parity ---
    if njpg != nframes:
        r["err"].append(f"jpg count {njpg} != frame_index {nframes}")

    cam_t = fidx[:, 0].astype(np.float64)
    r["dur_s"] = float(cam_t[-1] - cam_t[0])

    # --- 1. camera cadence ---
    if nframes >= 2:
        cdt = np.diff(cam_t)
        r["cam_hz"] = float((nframes - 1) / (cam_t[-1] - cam_t[0]))
        big = np.nonzero(cdt > 0.1)[0]
        if big.size:
            r["warn"].append(f"{big.size} camera gaps>0.1s (max {cdt.max()*1000:.0f}ms)")
        if np.any(cdt <= 0):
            r["err"].append("non-monotonic camera timestamps")

    # --- 2. F/T presence, rate, coverage, NaN, saturation ---
    if ft is None or ft.shape[0] == 0:
        r["err"].append("no ft_wrench.npy")
    else:
        ft_t = ft[:, 0].astype(np.float64)
        ft_w = ft[:, 1:7].astype(np.float64)
        r["ft_hz"] = float((ft.shape[0] - 1) / (ft_t[-1] - ft_t[0])) if ft.shape[0] >= 2 else 0.0
        if not np.all(np.isfinite(ft)):
            r["err"].append("NaN/inf in ft_wrench")
        # coverage of camera window (backward window needs pre-frame-0 samples)
        lead = cam_t[0] - ft_t[0]        # how much F/T history exists before frame 0
        r["ft_lead_ms"] = lead * 1000.0
        if ft_t[0] > cam_t[0] + 1e-3:
            r["warn"].append(f"F/T starts {(-lead)*1000:.0f}ms AFTER frame0 (frame0 window left-padded)")
        if ft_t[-1] < cam_t[-1] - 1e-3:
            r["warn"].append(f"F/T ends {(cam_t[-1]-ft_t[-1])*1000:.0f}ms before last frame")
        # K-window sizing: min raw samples in any backward window of the camera grid
        n_le = np.searchsorted(ft_t, cam_t, side="right")
        per_step_new = np.diff(np.concatenate([[0], n_le]))  # new samples between frames
        r["ft_per_step_min"] = int(per_step_new.min())
        r["ft_per_step_med"] = float(np.median(per_step_new))
        # samples available behind the FIRST usable frame (frame at index >=1)
        r["ft_behind_f1"] = int(n_le[1]) if nframes > 1 else int(n_le[0])
        # magnitude sanity
        fmag = np.linalg.norm(ft_w[:, :3], axis=1)
        r["|F|_med"] = float(np.median(fmag))
        r["|F|_max"] = float(fmag.max())
        if fmag.max() < 0.5:
            r["warn"].append("|F| max<0.5N — sensor flat/no contact all take?")
        if fmag.max() > 200:
            r["warn"].append(f"|F| max {fmag.max():.0f}N — possible saturation/collision")
        if np.all(np.abs(ft_w).max(axis=0) < 1e-6):
            r["err"].append("ft_wrench all-zero")
        # duplicate/backward timestamps
        keep = core.make_strictly_increasing(ft_t)
        if keep.sum() < ft_t.shape[0]:
            r["warn"].append(f"{ft_t.shape[0]-keep.sum()} non-monotonic F/T timestamps (converter drops)")

    # --- 3. compensated stream present + re-zeroed ---
    if ftc is None:
        r["warn"].append("no ft_wrench_comp.npy (raw-only; zarr will use raw)")
    else:
        r["ft_comp_present"] = True

    # --- 4. tare: free-space? settled? ---
    if tare is None or tare.shape[0] == 0:
        r["warn"].append("no tare_events")
    else:
        r["n_tare"] = int(tare.shape[0])
    tw = meta.get("ft_tare_wall", None)
    twr = meta.get("ft_tare_wrench", None)
    if twr is not None:
        r["tare_|F|"] = float(np.linalg.norm(np.asarray(twr)[:3]))
    # transient: |dF| over first TARE_SETTLE_S after frame0 vs rest
    if ft is not None and ft.shape[0] > 5:
        ft_t = ft[:, 0].astype(np.float64)
        ft_w = ft[:, 1:7].astype(np.float64)
        early = ft_w[(ft_t >= cam_t[0]) & (ft_t < cam_t[0] + TARE_SETTLE_S)]
        late = ft_w[(ft_t >= cam_t[0] + TARE_SETTLE_S)]
        if early.shape[0] and late.shape[0]:
            early_amp = np.linalg.norm(early[:, :3], axis=1).max()
            late_med = np.median(np.linalg.norm(late[:, :3], axis=1))
            r["tare_transient_N"] = float(early_amp)
            if early_amp > late_med + 15:
                r["warn"].append(f"tare transient {early_amp:.0f}N in first {TARE_SETTLE_S}s (mask in loss)")

    # --- 4b. camera GRID UNIFORMITY (this used to be suppressed) ---
    # Deliberately OUTSIDE the tracker block below: a take with a tracker
    # dropout is the take you most want audited, and nesting this inside
    # `if tr is not None ...` silently dropped the camera verdict for it.
    # The old code folded every ~100 ms advisory away with the comment "at
    # ~11.8Hz the frame period is ~85ms, so ordinary jitter trips the 0.1s
    # threshold ... benign, not surfaced as a warning". That 11.8 Hz WAS the
    # bug (a wall-clock publish throttle randomly decimating 30fps by 2 or 3
    # frames), and folding it is why 97 takes shipped with a +-20% jittery
    # master timeline. Test uniformity DIRECTLY instead.
    #
    # NOT `round(dt/median)`: on the broken data (dt alternating 66.7/100.0 ms,
    # median 97.2) every ratio rounds to 1 and the histogram reports "uniform".
    if cam_t.size > 2:
        cam_dt = np.diff(cam_t.astype(np.float64))
        med = float(np.median(cam_dt))
        p5, p95 = (float(x) for x in np.percentile(cam_dt, [5, 95]))
        spread = (p95 - p5) / max(med, 1e-9)
        r["camera_hz"] = round(1.0 / max(float(cam_dt.mean()), 1e-9), 2)
        r["camera_dt_spread"] = round(spread, 4)
        if spread > 0.05:
            r["warn"].append(
                f"camera grid NON-UNIFORM: {r['camera_hz']:.2f} Hz, dt "
                f"{p5 * 1000:.0f}/{med * 1000:.0f}/{p95 * 1000:.0f} ms "
                f"(p5/med/p95, spread {spread * 100:.0f}%) — the camera grid is "
                f"the training dataset's MASTER TIMELINE and no loader has a dt "
                f"input; check the camera bridge pacing"
            )
    # Prefer the recorder's own verdict when the take carries it (it buckets
    # against the true nominal capture period).
    if meta.get("camera_grid_uniform") is False:
        r["warn"].append(
            f"take_meta camera_grid_uniform=false "
            f"(dt_k={meta.get('camera_dt_k')}, "
            f"lost frames={meta.get('camera_seq_gaps')})"
        )

    # --- 5. tracker coverage / jumps / staleness (converter validator) ---
    if tr is not None and tr.shape[0] and nframes:
        tr_t = tr[:, 0].astype(np.float64)
        tr_p = tr[:, 2:5].astype(np.float64)
        rep = core.validate_episode_timeline(cam_t, tr_t, tr_p)
        r["tracker_hz"] = float((tr.shape[0]-1)/(tr_t[-1]-tr_t[0])) if tr.shape[0] >= 2 else 0.0
        # Camera gaps well above the frame period = genuinely dropped frames.
        cam_jumps = [w for w in rep["warnings"] if "camera time jump" in w]
        real_cam_jumps = [w for w in cam_jumps
                          if float(w.split()[3].rstrip("s")) > 0.13]
        if real_cam_jumps:
            r["warn"].append(f"{len(real_cam_jumps)} real camera gaps>130ms (dropped frames)")

        # Fold tracker position jumps: isolated single-sample spikes = tracking
        # glitches (interp rides over one bad sample); a RUN of large steps = real
        # fast motion or a sustained dropout. Count + report the worst, don't spam.
        pos_jumps = [w for w in rep["warnings"] if "tracker position jump" in w]
        if pos_jumps:
            d = np.linalg.norm(np.diff(tr_p, axis=0), axis=1)
            worst = float(d.max())
            # Raw-stream count only — whether it actually corrupts the label is decided
            # by the label-grid teleport check below (5b), which fires an ERR on a
            # PERSISTENT step. An isolated return spike is folded there and stays benign.
            r["warn"].append(f"{len(pos_jumps)} tracker pos jumps>0.1m on raw stream "
                             f"(worst {worst*1000:.0f}mm; see 5b for label-grid verdict)")
        for w in rep["warnings"]:
            if "camera time jump" in w or "tracker position jump" in w:
                continue  # folded above
            r["warn"].append("tracker: " + w)
        # quat norm
        q = tr[:, 5:9].astype(np.float64)
        qn = np.linalg.norm(q, axis=1)
        if np.any(np.abs(qn - 1.0) > 0.05):
            r["warn"].append(f"tracker quat norm off (min {qn.min():.3f} max {qn.max():.3f})")

        # --- 5b. LABEL-GRID continuity (position teleport + rotation glitch) ---
        # The checks above run on the RAW ~80 Hz stream. Two failure modes SURVIVE
        # resampling onto the camera timeline and land as a corrupt training LABEL:
        #   (a) a position teleport the converter could NOT bridge (an unbridged
        #       excursion / large residual jump), and
        #   (b) a rotation glitch — which the position-only validator is entirely blind
        #       to, AND which the converter's deglitch (position-only) does NOT remove,
        #       so a re-localization that flips orientation with a quiet position ships
        #       straight into the label.
        # To measure what the loader ACTUALLY sees we must apply the converter's SAME
        # tracker sanitize BEFORE resampling — otherwise we (i) crash scipy Slerp on a
        # duplicate timestamp, and (ii) manufacture false teleports on glitches the
        # converter cleanly bridges. Mirror convert_to_umi's order exactly:
        # stale-drop -> make_strictly_increasing -> deglitch_tracker_positions.
        st_col = 9  # TRACKER_COLS index of 'stale' (see column-layout note above)
        s_t, s_p, s_q = tr_t, tr_p, q
        if tr.shape[1] > st_col:
            live = tr[:, st_col] < 0.5
            if live.any():
                s_t, s_p, s_q = s_t[live], s_p[live], s_q[live]
        keep = core.make_strictly_increasing(s_t)
        s_t, s_p, s_q = s_t[keep], s_p[keep], s_q[keep]
        # tracker_glitch_severity is the SAME verdict the converter's episode-drop gate
        # uses: how many excursions it could NOT bridge, and the worst residual jump that
        # survives on the KEPT stream. A surviving unbridged excursion is exactly a label
        # teleport. (This replaces a hand-rolled persistence heuristic that could not tell
        # a fast reach from a teleport.)
        if s_t.shape[0] >= 3:
            # tracker_glitch_severity is computed on the strictly-increasing stream
            # BEFORE deglitch — matching convert_to_umi.py:387 exactly.
            sev = core.tracker_glitch_severity(s_t, s_p)
            keep_gl = core.deglitch_tracker_positions(s_t, s_p)
            s_t, s_p, s_q = s_t[keep_gl], s_p[keep_gl], s_q[keep_gl]
            r["tracker_unbridged"] = int(sev["n_unbridged"])
            r["tracker_residual_jump_mm"] = float(sev["residual_max_jump_m"] * 1000.0)
            # Mirror the converter's ACTUAL episode-drop gate (convert_to_umi.py:421-423),
            # not a stricter n_unbridged>0 rule. The converter DROPS the take only at
            # >=TRACKER_DROP_MAX_UNBRIDGED unbridged OR residual>=TRACKER_DROP_RESIDUAL_JUMP_M;
            # below that it KEEPS the take, flags the frames in robot0_pose_glitch, and the
            # HL loader's drop_glitch_windows excludes only the straddling windows. So:
            #   - meets the drop gate  -> ERR  (converter itself drops the whole take)
            #   - 1..2 sub-threshold   -> warn (recoverable: loader drops ~a handful of windows)
            if sev["n_unbridged"] >= _DROP_MAX_UNBRIDGED or \
                    sev["residual_max_jump_m"] >= _DROP_RESIDUAL_JUMP_M:
                r["err"].append(
                    f"{sev['n_unbridged']} UNBRIDGED tracker excursion(s), residual jump "
                    f"{sev['residual_max_jump_m']*1000:.0f}mm — meets the converter's "
                    f"episode-DROP gate ({_DROP_MAX_UNBRIDGED}+ unbridged or "
                    f">={_DROP_RESIDUAL_JUMP_M*1000:.0f}mm); the take is unusable")
            elif sev["n_unbridged"] > 0:
                r["warn"].append(
                    f"{sev['n_unbridged']} unbridged tracker excursion(s), residual jump "
                    f"{sev['residual_max_jump_m']*1000:.0f}mm — below the drop gate; the "
                    f"converter KEEPS the take and the HL loader auto-drops only the windows "
                    f"straddling the flagged frame(s)")
        if s_t.shape[0] >= 2 and nframes >= 2:
            # Resample the SANITIZED stream onto the camera grid = the real training label.
            pos_lab = core.resample_position(s_t, s_p, cam_t)            # [T,3]
            rot_lab = core.resample_rotation_wxyz(s_t, s_q, cam_t)       # scipy Rotation [T]
            dp_lab = np.linalg.norm(np.diff(pos_lab, axis=0), axis=1)    # metres/frame
            r["pos_label_step_max_mm"] = float(dp_lab.max() * 1000.0) if dp_lab.size else 0.0
            # rotation: per-frame angular step via relative rotation magnitude (deg/frame)
            rq = rot_lab.as_quat()                                       # xyzw [T,4]
            dots = np.abs(np.sum(rq[:-1] * rq[1:], axis=1))             # |dot|, double-cover
            ang_lab = np.degrees(2.0 * np.arccos(np.clip(dots, 0.0, 1.0)))  # deg/frame
            r["rot_label_step_max_deg"] = float(ang_lab.max()) if ang_lab.size else 0.0
            if ang_lab.size and ang_lab.max() > ROT_LABEL_STEP_FLOOR_DEG:
                j = int(np.argmax(ang_lab))                             # worst step j->j+1
                lo = max(0, j - ROT_GLITCH_WINDOW)
                hi = min(len(ang_lab) - 1, j + ROT_GLITCH_WINDOW)      # last valid index in ang_lab
                path = float(ang_lab[lo:hi + 1].sum())                  # summed rotation over window
                q_lo, q_hi = rq[lo], rq[hi + 1]                         # net over the same window
                net = float(np.degrees(2.0 * np.arccos(
                    np.clip(abs(float(np.dot(q_lo, q_hi))), 0.0, 1.0))))
                ratio = path / max(net, 1e-6)
                # path/net is the discriminator: a there-and-back glitch traces a long
                # path for ~zero net rotation. Only THAT is flagged; real fast wrist
                # rotation (net ~= path) is left alone even above the floor.
                if ratio > ROT_PATH_NET_RATIO:
                    r["warn"].append(
                        f"rotation label glitch {ang_lab.max():.0f}deg/frame at frame "
                        f"{j}->{j+1} (path/net {ratio:.1f}x over +-{ROT_GLITCH_WINDOW} "
                        f"frames = there-and-back). Since 2026-09-11 the converter's rotation "
                        f"deglitch (default ON) drops the excursion rows and masks the span in "
                        f"robot0_pose_glitch / hl_left_action_valid, so this costs windows, "
                        f"not the take; --exclude only if it recurs through the whole take")
        # The label-grid test above is necessary but NOT sufficient: the camera grid
        # ATTENUATES an excursion that lands between two frames and stretches one that
        # falls inside a run of dropped frames, so a real 21 deg there-and-back can read
        # 7 deg/frame with a path/net of 2.3 and pass both gates (measured: take_366 and
        # take_371 of the 0823 session, whose RAW rates were 1061 and 1452 deg/s). Run the
        # SAME rejoin discriminator on the raw ~77 Hz stream, where the amplitude is the
        # amplitude. Kept as a second, independent warning rather than folded into the
        # first: they disagree only when the camera grid is the reason, and knowing that
        # is the point.
        if s_t.shape[0] >= 3:
            exc = core.rotation_excursions(s_t, s_q)
            r["rot_excursions"] = len(exc)
            r["rot_excursion_amp_deg"] = max((e["amp_deg"] for e in exc), default=0.0)
            for e in exc:
                r["warn"].append(
                    f"RAW rotation excursion {e['amp_deg']:.0f}deg peak over "
                    f"{e['n_samples']} tracker sample(s) (out-jump {e['rate_dps']:.0f} "
                    f"deg/s, then REJOINS where it left) — a lighthouse orientation "
                    f"glitch. The converter's rotation deglitch drops these rows and masks "
                    f"the span (robot0_pose_glitch); take KEPT")
    r["stale_frac"] = float(meta.get("tracker_stale_fraction", 0.0))
    if r["stale_frac"] > 0.05:
        r["warn"].append(f"tracker stale {r['stale_frac']*100:.0f}%")

    # --- 5c. EVERY keypoint the MODE declares must pass the no-jump gate ----------
    # The secondary pucks were loaded above and then never examined, so a lighthouse
    # glitch on the right hand or the chest passed the audit silently — and unlike the
    # primary they get no pose_glitch mask downstream.
    #
    # The requirement is CONJUNCTIVE over the mode's roles: recording in 2pt means both
    # hands are part of the dataset, 3pt means all three keypoints are and 5pt all five,
    # so a take is
    # acceptable only if EVERY declared role is jump-free. `mode` (stamped by the
    # recorder) is the statement of intent, and it decides severity: a declared role
    # failing the gate is an ERR, exactly like the primary; a stray secondary file in a
    # session that does not declare it stays a WARN.
    #
    # Additive for 1pt: `want` is ('left',) there, which is the primary's own block
    # above, so no 1pt take gains a key, a warning, or an error.
    mode = str(meta.get("mode", "")).strip()
    want = {"1pt": ("left",), "2pt": ("left", "right"),
            "3pt": ("left", "right", "head"),
            "5pt": ("left", "right", "head", "left_foot", "right_foot")}.get(mode)
    if want is None:
        # No (or unknown) mode stamp: fall back to "whatever actually recorded", so an
        # unlabelled take is still audited. Intent is unknown, so nothing is REQUIRED —
        # findings stay advisory rather than inventing a dataset contract.
        want = ()
    r["mode_roles"] = list(want)

    sec_present, clean, dirty = [], [], []
    # Did the primary block above actually measure anything? Same condition that guards it.
    primary_measured = tr is not None and tr.shape[0] > 0 and bool(nframes)
    if not primary_measured:
        # There was no error for this at all — `no ft_wrench.npy` was an err but a missing
        # PRIMARY tracker was silent, which is the more load-bearing of the two (it is the
        # robot0_* label). Every real take has one, so this cannot change a past verdict.
        r["err"].append(
            "no usable tracker.npy (missing, empty, or no camera frames) — there is no "
            "robot0_* pose label to build, so the take cannot serve ANY mode")
    if "left" in want:
        (clean if _primary_is_clean(r, primary_measured) else dirty).append("left")
    for role, _fname, label_key in _SECONDARY_TRACKERS:
        tr_sec = {"right": trr, "head": trh,
                  "left_foot": trlf, "right_foot": trrf}[role]
        if tr_sec is None or tr_sec.shape[0] == 0:
            continue
        sec_present.append(role)
        role_ok = _audit_secondary_tracker(
            r, role, tr_sec, label_key, cam_t, nframes, required=role in want)
        (clean if role_ok else dirty).append(role)
    r["secondary_trackers"] = sec_present

    # A declared role whose file is MISSING or EMPTY records nothing for that keypoint.
    # Silent before: the converter simply omits the key, so the take reads as clean
    # lower-mode data and the shortfall only surfaces at training time. This is an ERR
    # for a declared role — a keypoint that does not exist cannot satisfy the no-jump
    # requirement, so it can never be part of a conjunctive pass.
    missing = [role for role in want if role != "left" and role not in sec_present]
    if missing:
        dirty.extend(missing)
        r["err"].append(
            f"take_meta says mode={mode} but {', '.join(missing)} tracker(s) recorded "
            f"NOTHING — that keypoint is absent from the zarr entirely (the converter "
            f"omits the key), so this take cannot serve a {mode} dataset; it is really "
            f"{'1pt' if len(missing) == len(want) - 1 else 'lower-mode'} data")

    # The conjunctive verdict, recorded so the session summary can report it.
    r["clean_roles"] = sorted(set(clean) - set(dirty))
    r["dirty_roles"] = sorted(set(dirty))
    r["all_roles_clean"] = bool(want) and not r["dirty_roles"]

    # --- 6. gripper calibration + grasp signal ---
    if grip is not None and grip.shape[0]:
        try:
            cal, src = core.effective_gripper_calib(grip, _BASE_CALIB)
            r["grip_cal_src"] = src
            enc = grip[:, 1].astype(np.float64)
            if cal is not None:
                w = cal.to_width_m(enc)
                r["grip_w_min"] = float(w.min())
                r["grip_w_max"] = float(w.max())
                r["grip_crosses_grasp"] = bool(w.min() < GRASP_THRESH_M < w.max())
        except Exception as e:  # noqa: BLE001
            r["warn"].append(f"gripper calib probe failed: {e!r}")

    # --- 7. markers (should be 0 => single episode) ---
    r["n_episodes_meta"] = meta.get("n_episodes", None)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session")
    ap.add_argument("--k", type=int, default=16)
    args = ap.parse_args()

    takes = sorted(d for d in os.listdir(args.session) if d.startswith("take_"))
    # Never let ONE bad take abort QC for the whole session (that defeats the tool):
    # an unexpected raise is turned into a per-take error row and the audit continues.
    rows = []
    for t in takes:
        try:
            rows.append(audit_take(os.path.join(args.session, t), args.k))
        except Exception as e:  # noqa: BLE001
            # A crashed take is unusable: route it to the JUNK bucket (the "aborted take"
            # marker) and carry frames/dur_s so both the junk print (reads r['frames'])
            # and the good-row aggregates never KeyError on this fallback row.
            rows.append({"take": t, "warn": [], "frames": 0, "dur_s": 0.0,
                         "err": [f"aborted take: audit crashed: {type(e).__name__}: {e}"]})

    print(f"\n=== SESSION AUDIT: {args.session}  (K={args.k}) ===\n")
    hdr = (f"{'take':9s} {'frm':>4s} {'dur_s':>6s} {'cam_hz':>6s} {'ft_hz':>6s} "
           f"{'/step_min':>9s} {'behindF1':>8s} {'|F|med':>6s} {'|F|max':>7s} "
           f"{'gripΔ':>6s} {'stale%':>6s} {'valid':>5s}")
    print(hdr); print("-" * len(hdr))
    good, junk = [], []
    for r in rows:
        if r["err"] and "aborted take" in " ".join(r["err"]):
            junk.append(r); continue
        good.append(r)
        gr = ("Y" if r.get("grip_crosses_grasp") else "n") if "grip_crosses_grasp" in r else "?"
        print(f"{r['take']:9s} {r.get('frames',0):>4d} {r.get('dur_s',0):>6.1f} "
              f"{r.get('cam_hz',0):>6.1f} {r.get('ft_hz',0):>6.1f} "
              f"{r.get('ft_per_step_min','?'):>9} {r.get('ft_behind_f1','?'):>8} "
              f"{r.get('|F|_med',0):>6.1f} {r.get('|F|_max',0):>7.1f} "
              f"{gr:>6s} {r.get('stale_frac',0)*100:>5.1f} "
              f"{'Y' if r.get('ft_valid') else 'N':>5s}")
    # warnings / errors
    print("\n--- WARNINGS / ERRORS (real takes) ---")
    any_issue = False
    for r in good:
        if r["warn"] or r["err"]:
            any_issue = True
            for e in r["err"]:
                print(f"  [ERR ] {r['take']}: {e}")
            for w in r["warn"]:
                print(f"  [warn] {r['take']}: {w}")
    if not any_issue:
        print("  (none)")

    if junk:
        print("\n--- ABORTED / JUNK TAKES (exclude from training) ---")
        for r in junk:
            print(f"  {r['take']}: {r['frames']} frames — {'; '.join(r['err'])}")

    # aggregates
    print("\n--- SESSION AGGREGATES (real takes only) ---")
    ft_hz = [r["ft_hz"] for r in good if "ft_hz" in r]
    cam_hz = [r["cam_hz"] for r in good if "cam_hz" in r]
    durs = [r["dur_s"] for r in good if "dur_s" in r]
    pmin = [r["ft_per_step_min"] for r in good if "ft_per_step_min" in r]
    behind = [r["ft_behind_f1"] for r in good if "ft_behind_f1" in r]
    grasp_ok = sum(1 for r in good if r.get("grip_crosses_grasp"))
    print(f"  usable takes: {len(good)}   junk: {len(junk)}")
    if ft_hz:
        print(f"  F/T rate: {np.mean(ft_hz):.1f} Hz (min {min(ft_hz):.1f}, max {max(ft_hz):.1f})")
    if cam_hz:
        print(f"  camera rate: {np.mean(cam_hz):.1f} Hz")
    if durs:
        print(f"  episode dur: mean {np.mean(durs):.1f}s (min {min(durs):.1f}, max {max(durs):.1f}), total {sum(durs):.0f}s")
    if pmin:
        print(f"  F/T new-samples/camera-step: min-across-takes {min(pmin)} (median-of-medians ok for K)")
    if behind:
        print(f"  F/T samples behind frame1: min {min(behind)} (K={args.k} => "
              f"{'left-pad early frames' if min(behind) < args.k else 'full window from frame1'})")
    print(f"  gripper grasp signal present: {grasp_ok}/{len(good)} takes cross {GRASP_THRESH_M}m")

    # --- multi-keypoint no-jump verdict (2pt/3pt) --------------------------------
    # Printed only when some take declares more than one keypoint, so a 1pt session's
    # output is byte-identical. The number that matters for a 2pt/3pt dataset is how many
    # takes are clean on EVERY declared role at once — a per-role tally would let a
    # session read "95% clean on each" while few takes are clean on all of them.
    multi = [r for r in good if len(r.get("mode_roles", [])) > 1]
    if multi:
        modes = sorted({r.get("mode_roles") and "+".join(r["mode_roles"]) for r in multi})
        n_all = sum(1 for r in multi if r.get("all_roles_clean"))
        print(f"\n  keypoints declared: {', '.join(m for m in modes if m)}")
        print(f"  takes jump-free on ALL declared keypoints: {n_all}/{len(multi)}")
        if n_all < len(multi):
            per_role = {}
            for r in multi:
                for role in r.get("dirty_roles", []):
                    per_role[role] = per_role.get(role, 0) + 1
            print("    failing keypoint tally: "
                  + ", ".join(f"{k}={v}" for k, v in sorted(per_role.items())))
            print("    (since 2026-09-11 the converter masks EVERY role's glitches per role — "
                  "<role>_pose_glitch + hl_<role>_action_valid — and drops rotation excursions,\n"
                  "     so a dirty keypoint costs the straddling windows, not the take; only the "
                  "PRIMARY's unbridgeable excursions drop a take. The converter prints the totals.)")

    # K sizing verdict
    print(f"\n--- K={args.k} HI-RATE VERDICT ---")
    if pmin and behind:
        span_ms = args.k / np.mean(ft_hz) * 1000 if ft_hz else 0
        print(f"  Backward window K={args.k} spans ~{span_ms:.0f}ms at {np.mean(ft_hz):.0f}Hz.")
        print(f"  Overlapping windows => min new-samples/step ({min(pmin)}) does NOT bound K (OK).")
        print(f"  Only frame0 (and any frame with <K history) is left-padded — expected & edge-handled.")
    print()


if __name__ == "__main__":
    raise SystemExit(main())
