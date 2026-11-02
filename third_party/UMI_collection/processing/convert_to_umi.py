#!/usr/bin/env python3
"""Convert a raw recording session -> UMI Zarr ReplayBuffer (.zarr.zip).

Runs in the SAME ``.venv`` as collection (DECISION-I: one maintained env for
capture AND conversion; ``bootstrap.sh`` installs zarr/numcodecs/imagecodecs/av/
cv2 into ``.venv``). Reads the raw per-TAKE streams written by
``recording/recorder.py``, SPLITS each continuous take into UMI episodes at the
episode-split markers (DECISION-E), resamples every modality onto each episode's
CAMERA-FRAME timeline, and writes a single ``.zarr.zip`` in the layout the stock
UMI / diffusion_policy ``UmiDataset`` loader ingests:

  /meta/episode_ends           int64  [n_ep]      cumulative; [-1] == T_total
  /data/
    robot0_eef_pos             f32 [T,3]          TCP xyz (world<-TCP)
    robot0_eef_rot_axis_angle  f32 [T,3]          TCP rotvec
    robot0_gripper_width       f32 [T,1]          encoder -> absolute metres
    robot0_demo_start_pose     f32 [T,6]          episode first TCP 6D, broadcast
    robot0_demo_end_pose       f32 [T,6]          episode last  TCP 6D, broadcast
    robot0_ft_wrench           f32 [T,6]          Bota wrench (OUR extension; mandatory;
                                                   gravity/inertia-COMPENSATED when the take
                                                   has ft_wrench_comp.npy, else RAW — same
                                                   sensor frame/sign; see attrs.ft_compensated)
    robot0_ft_wrench_hires     f32 [T,K,6]        OUR extension; hi-rate per-step CAUSAL F/T
                                                   window (K most-recent raw samples at/before
                                                   each camera frame; K=--ft-hires-k, default 16;
                                                   omitted when K==0). Same provenance/frame/sign
                                                   as robot0_ft_wrench. MuSe-style hi-rate target.
    episode_timestamps         f64 [T]            camera frame timestamps, RELATIVE
                                                   (first frame of each episode == 0; DECISION-A)
    camera0_rgb                u8  [T,H,W,3]      JpegXl(99), chunks=(1,H,W,3)

  ADDITIVE 2pt/3pt keys — each written ONLY when the session actually recorded it, and
  all-or-nothing across the session (a key present in only some episodes RAISES rather
  than desyncing the columns). LEFT is robot0/camera0; the RIGHT hand is robot1/camera1;
  the CHEST is head_* (a body reference — no camera, F/T or gripper, and NO TCP extrinsic):
    robot1_eef_pos / _eef_rot_axis_angle  f32 [T,3]   RIGHT hand, SAME TCP extrinsic as robot0
    robot1_gripper_width       f32 [T,1]              RIGHT DM4310, its OWN homed endpoints
    robot1_ft_wrench           f32 [T,6]              RIGHT Bota, same resample as robot0's
    camera1_rgb                u8  [T,H,W,3]          RIGHT D405, paired onto camera0's grid
                                                       by CAUSAL frame selection (images cannot
                                                       be interpolated; a zero-order hold lags
                                                       but never shows a FUTURE instant, which
                                                       on the action-label grid is a leak).
                                                       Skew recorded in attrs camera1_pair_*.
    head_pos / head_rot_axis_angle f32 [T,3]          CHEST puck, RAW pose (no extrinsic)
  attrs `n_robots` / `n_cameras` are DERIVED from the above, and attrs `hands` records
  per-keypoint {pose, camera, ft, gripper} so downstream never infers capability from
  which keys happen to exist. NOTHING is zero-filled: "no sensor" must stay
  distinguishable from "the sensor read zero" (robot0_ft_wrench is the one exception —
  it is zero-filled + `ft_valid`-flagged for schema stability on the PRIMARY hand).

Notes locked in by the design review:
  * QUATERNION CONVENTION (DECISION-B): the raw tracker quaternions are WXYZ
    (libsurvive). They are converted to xyzw at EXACTLY ONE place —
    ``umi_convert_core.quat_wxyz_to_xyzw`` — invoked here. We also print a
    one-line wxyz reminder so the convention is never silently assumed.
  * Each take is split into N+1 episodes at its N markers; ``episode_timestamps``
    is made RELATIVE per episode (first frame 0) since training only uses frame
    deltas and the absolute wall-clock is kept in the raw files (DECISION-A).
  * NO ``joint_pos`` key — it is HuMI-G1-specific (sampler hard-slices [:12]); a
    bogus one corrupts training. Raw encoder is preserved in the RAW files only.
  * ``robot0_ft_wrench`` is written for EVERY episode (zero-filled + ``ft_valid``
    flag if Bota was absent) — a missing key on one episode makes the whole
    ReplayBuffer fail to load. It prefers the COMPENSATED sidecar
    (``ft_wrench_comp.npy``) when the take recorded one, else the RAW
    ``ft_wrench.npy``; ``attrs.ft_compensated`` is the session tri-state
    (compensated|raw|mixed). Raw is always kept on disk as the audit/safety net.
  * ``T_tracker_tcp`` identity -> the converter stamps ``UNCALIBRATED_DO_NOT_TRAIN``
    into the zarr attrs + session report (rotated poses are wrong otherwise).
  * Images use UMI's aspect-preserving resize + CENTER CROP (not a naive squash),
    out_res default (224, 224) — same as HuMI (generate_replay_buffer.py:64).

Usage:
    .venv/bin/python processing/convert_to_umi.py data/recordings/session_XXX \
        -o session_XXX.zarr.zip [--out-res 224 224]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "recording"))

import recording_format as fmt  # noqa: E402
import umi_convert_core as core  # noqa: E402


# Hi-rate F/T export width: K raw wrench samples per camera step, written as
# robot0_ft_wrench_hires [T, K, 6] via a CAUSAL backward window (see
# umi_convert_core.resample_wrench_hires). Default 16 — decoupled from the ~15
# raw-samples-per-camera-interval rate because consecutive backward windows are
# near-contiguous (overlapping or with a small gap), so every frame EXCEPT frame 0 of
# an episode is fully covered by real samples (measured ~178 Hz F/T / ~11.8 Hz camera).
# 0 disables the key entirely, keeping the zarr byte-identical to the pre-hi-rate
# schema. Overridden by the --ft-hires-k CLI arg.
FT_HIRES_K = 16

# ---------------------------------------------------------------------------
# HL-v2 high-level policy windows (--hands / --hl-windows)
# ---------------------------------------------------------------------------
# The high-level S1 policy contract is defined in SECONDS, not camera frames:
#   INPUT   image @ t0, gripper opening @ t0, 8 causal force bins over [t0-1s, t0]
#   OUTPUT  16 steps over (t0, t0+1s]: pose, force, gripper
# The camera grid is NOT uniform (attrs.camera_dt_uniform=False; measured 16-frame
# windows span 1.166..1.568 s, with 0% within 1.0+/-0.05 s), so every model-facing
# window is resampled onto an explicit fixed-dt grid anchored at each camera frame.
# The grid definitions + resamplers live in umi_convert_core (HL-v2 section) and are
# shared verbatim by the online deploy path, which is the only way the two can be
# guaranteed to agree.
#
# EVERY key is prefixed by ROLE — `hl_<role>_...` for role in {left, right, head} (see
# _hl_prefix). Written per camera frame t0, all additive; `--no-hl-windows` omits every
# one of them and keeps the zarr byte-identical to the pre-HL schema:
#
#   hl_<role>_force_history        f32 [T, 8, 6]   causal input bins over [t0-1s, t0]
#   hl_<role>_force_history_valid  f32 [T, 1]      1.0 = measured, gap-free, tare-clean
#   hl_<role>_action_pose          f32 [T, 16, 6]  ABSOLUTE pose targets (pos + rotvec)
#   hl_<role>_action_valid         f32 [T, 16]     1.0 = step inside the episode
#   hl_<role>_action_gripper       f32 [T, 16]     absolute width, metres
#   hl_<role>_action_gripper_valid f32 [T, 16]     1.0 = a live encoder stream backed it
#   hl_<role>_action_force         f32 [T, 16, 6]  force targets, bin k aligned to step k
#   hl_<role>_action_force_valid   f32 [T, 16]     1.0 = bin measured + tare-clean
#
# So `--hands left` emits exactly the eight `hl_left_*` arrays. Gripper keys exist only for
# a role in HL_GRIPPER_ROLES[mode] and force keys only for a role in HL_FT_ROLES[mode]
# (LEFT-only for every mode except `both_head_dual`, which declares BOTH hands); within a
# mode they are emitted unconditionally (never gated on a stream being present) because the
# zarr writer requires all episodes to agree on the key set — a dead stream is
# zero-filled and flagged invalid instead of making the session unconvertible.
#
# Poses are ABSOLUTE (aligned world) exactly like robot0_eef_pos; making them
# anchor-relative is the training dataset's job, so the zarr stays a faithful recording
# rather than baking in one policy's frame choice.
HL_WINDOWS = True

#: --hands mode -> the tracker roles it needs. MUST stay in sync with
#: flore/s1_policy/data/layout.py::tracker_roles_required (a FLoRe-side test asserts it).
#:
#: ``both_head_dual`` needs the SAME three pucks as ``both_head``; they differ only in the
#: per-hand HARDWARE the HL windows carry (see :data:`HL_FT_ROLES`), which is why it is a
#: separate mode name rather than a flag on ``both_head``: the two produce different key
#: SETS, and the FLoRe loader gates ``attrs.hands_mode`` exactly so that a dual-hardware
#: file can never be read as a left-only one (which would silently drop half the sensors).
HANDS_MODES: dict[str, tuple[str, ...]] = {
    "left": ("left",),
    "right": ("right",),
    "both": ("left", "right"),
    "both_head": ("left", "right", "head"),
    "both_head_dual": ("left", "right", "head"),
}
#: --hands mode -> which roles get FORCE keys in the **HL-v2 window arrays**
#: (``hl_<role>_force_history`` / ``hl_<role>_action_force``). :data:`HL_GRIPPER_ROLES` is
#: the same table for ``hl_<role>_action_gripper``.
#:
#: Every mode except ``both_head_dual`` is LEFT-ONLY, and deliberately stayed that way when
#: the rig gained a second Bota + a second DM4310. This is a TRAINING contract, not an
#: inventory: the shipped HL-v2 checkpoints are layout ``left_only``/``both_hands_head``
#: (one hand's force/gripper), and ``flore/s1_policy/data/layout.py`` asserts the key set
#: mode by mode — widening an EXISTING mode here would change its ``force_hist_channels``
#: (6 -> 12) and ``state_dim`` (19 -> 20) and so invalidate every checkpoint trained on it,
#: with no error at load time on the FLoRe side other than a width refusal.
#:
#: ``both_head_dual`` is that widening, done as a NEW mode: two 6-DoF wrenches and two
#: gripper widths, feeding FLoRe layout mode ``both_hands_head_dual``. The action COLUMNS do
#: not move (every hand block always reserved a gripper + force slot), so the 41-dim vector
#: is bit-compatible with ``both_head``'s — only the validity masks and the model's INPUT
#: widths change. That is the whole point of the reserved-vs-declared split.
#:
#: ``right`` gets NOTHING on purpose: there is one gripper and one Bota per hand, and a
#: right-only session means the operator recorded the right puck with the LEFT hand's
#: hardware unmoved. FLoRe's ``right_only`` layout declares a right-hand sensor on the
#: opposite assumption; ``tests/test_hl_contract.py`` pins that as the one known,
#: deliberate divergence so a NEW one still fails.
HL_FT_ROLES: dict[str, tuple[str, ...]] = {
    "left": ("left",),
    "right": (),
    "both": ("left",),
    "both_head": ("left",),
    "both_head_dual": ("left", "right"),
}
HL_GRIPPER_ROLES: dict[str, tuple[str, ...]] = {
    "left": ("left",),
    "right": (),
    "both": ("left",),
    "both_head": ("left",),
    "both_head_dual": ("left", "right"),
}
#: The role sets used when ``--hl-windows`` is given WITHOUT ``--hands`` (a supported
#: invocation that emits every tracker on disk). LEFT-only, i.e. the historical behavior:
#: an inferred conversion must not silently opt into the two-hand contract, because that
#: decision changes which FLoRe layout mode can read the file.
FT_ROLES = ("left",)
GRIPPER_ROLES = ("left",)


def hl_ft_roles(hands: "str | None") -> tuple[str, ...]:
    """Roles that get HL FORCE keys under ``--hands`` (None = the inferred default)."""
    if hands is None:
        return FT_ROLES
    try:
        return HL_FT_ROLES[hands]
    except KeyError:
        raise ValueError(f"unknown --hands {hands!r}; expected one of "
                         f"{sorted(HANDS_MODES)}") from None


def hl_gripper_roles(hands: "str | None") -> tuple[str, ...]:
    """Roles that get HL GRIPPER keys under ``--hands`` (None = the inferred default)."""
    if hands is None:
        return GRIPPER_ROLES
    try:
        return HL_GRIPPER_ROLES[hands]
    except KeyError:
        raise ValueError(f"unknown --hands {hands!r}; expected one of "
                         f"{sorted(HANDS_MODES)}") from None

# Tracker-glitch handling, in THREE tiers (shared with the audit tool so they can't drift):
#
#   1. deglitch_tracker_positions BRIDGES isolated lighthouse teleports that snap back.
#   2. An excursion corrupts the resampled eef_pos (and therefore the action) label at a
#      handful of frames. Those frames are FLAGGED per-frame in `robot0_pose_glitch` so a
#      training sampler can exclude only the WINDOWS that straddle them -- see the
#      measured justification below. This covers BOTH bridged and unbridged excursions
#      (see `_corrupt_spans`): bridging is not repairing.
#   3. Only a take whose tracker is *systemically* broken is dropped outright.
#
# Why tier 2 exists (measured on session_2026.07.25, 97 takes): a `>=1 unbridged`
# drop gate rejected 22 takes, and 20 of those 22 were rejected by the unbridged
# criterion ALONE (the residual-jump criterion never fired on its own). 15 of them had
# exactly ONE unbridged excursion. At a 32-frame training window and ~200 windows per
# take, one corrupted frame contaminates ~16% of that take's windows -- so dropping the
# take threw away the other 84% too, i.e. 20% of the whole dataset. Flagging the frames
# and excluding only the straddling windows recovers ~17% of the data at essentially no
# label cost. Takes with >=3 unbridged excursions (peak speeds 18-40 m/s) are
# systemically bad and stay dropped.
#
# Either of these still drops the take:
#   * >= TRACKER_DROP_MAX_UNBRIDGED unbridged excursions, OR
#   * a residual single-step jump on the KEPT stream above this many metres.
# 0.30 m in one ~13 ms tracker step is ~23 m/s — already far past human hand speed,
# so it can only be an unbridged glitch, not real motion.
TRACKER_DROP_MAX_UNBRIDGED = 3
TRACKER_DROP_RESIDUAL_JUMP_M = 0.30
# Frames within this many seconds of an excursion are flagged. The excursion bracket
# itself is already covered; the margin absorbs the linear interp/Slerp skirt that the
# discontinuity pulls on either side of it.
POSE_GLITCH_MARGIN_S = 0.10


def _corrupt_spans(tracker_sev: dict) -> list:
    """Wall-clock spans where the resampled pose label is NOT measurement.

    The union of the excursions the deglitch could not bridge AND the ones it did. Both
    corrupt the label, for opposite reasons:

    * UNBRIDGED — the bad samples are still in the stream, so the teleport itself
      interpolates straight into ``robot0_eef_pos``.
    * BRIDGED — the bad samples are gone, but ``interp1d``/``Slerp`` draw a straight line
      across the hole they left. Wherever the hand really moved during the occlusion that
      line is a FABRICATED ramp, not a measurement.

    Treating "bridged" as "fixed" was a real labelling hole: on clean94, episodes 30 and 31
    (takes 033/034) carry 118 mm and 127 mm ONE-FRAME jumps in ``robot0_eef_pos`` — 1.76
    and 1.91 m/s, ~3.5x the fastest honest step in the other 89 episodes — with
    ``robot0_pose_glitch == 0`` on every frame, because ``unbridged_spans`` is empty for a
    bridged excursion by construction. The bridges span 190-202 ms (15-16 dropped samples)
    while the hand moved 95-98 mm, so the ramp is entirely invented. Episodes 1/27/28 hit
    the same ramp but happened to ALSO carry an unbridged excursion nearby, which is the
    only reason they were flagged at all.

    Deliberately a UNION and not a re-derivation: both lists come from the one scan in
    ``deglitch_tracker_positions``, so this can never disagree with what the deglitch did.
    """
    return (list(tracker_sev.get("unbridged_spans", ()))
            + list(tracker_sev.get("bridged_spans", ()))
            # Rotation excursions dropped by `_drop_rotation_excursions`: the same
            # "bridged, therefore interpolated, therefore not a measurement" argument.
            + list(tracker_sev.get("rot_spans", ())))


# ---------------------------------------------------------------------------
# Calibration loading
# ---------------------------------------------------------------------------
# Repo-level default rig calibration (DECISION-D/H). A session's own
# calibration/ file wins; if it has none, we fall back to these so a conversion
# uses the measured rig values instead of silently defaulting to identity/zero.
_REPO_CALIB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "..", "robot_assets", "calibration")


def _resolve_calib(session_dir: str, filename: str) -> str | None:
    """Per-session calibration file if present, else the repo-level default."""
    p = os.path.join(session_dir, fmt.CALIB_DIR, filename)
    if os.path.exists(p):
        return p
    d = os.path.join(_REPO_CALIB_DIR, filename)
    if os.path.exists(d):
        print(f"[convert] using repo default calibration: {os.path.relpath(d)}")
        return d
    return None


def load_gripper_calib(session_dir: str) -> core.GripperWidthCalib | None:
    p = _resolve_calib(session_dir, fmt.GRIPPER_RANGE_FILE)
    if p is None:
        return None
    d = json.load(open(p))
    return core.GripperWidthCalib(
        enc_closed=float(d["enc_closed"]), enc_open=float(d["enc_open"]),
        width_closed_m=float(d["width_closed_m"]), width_open_m=float(d["width_open_m"]),
    )


def load_tcp_extrinsic(session_dir: str) -> tuple[np.ndarray, bool]:
    """Return (4x4 T_tracker_tcp, calibrated_bool). Identity + False if absent."""
    p = _resolve_calib(session_dir, fmt.TCP_EXTRINSIC_FILE)
    if p is None:
        return np.eye(4), False
    d = json.load(open(p))
    T = np.asarray(d.get("T_tracker_tcp", np.eye(4).tolist()), dtype=np.float64)
    calibrated = bool(d.get("calibrated", False)) and not np.allclose(T, np.eye(4))
    return T, calibrated


def load_head_extrinsic(session_dir: str) -> tuple[np.ndarray, bool]:
    """Return (4x4 T_tracker_head, calibrated_bool) for the CHEST puck. Identity + False
    if absent (= store the raw chest-puck axes, the pre-2026-08-26 behavior).

    The head's analogue of :func:`load_tcp_extrinsic`, resolved through the same
    per-session-then-repo-default path, and applied through the same
    ``core.apply_tcp_extrinsic`` compose. Two things make it NOT a copy of the hands':

      * it is a DIFFERENT rotation (a different puck on a different bracket) — see the
        ``_note`` in ``robot_assets/calibration/head_extrinsic.json``;
      * it is ROTATION-ONLY. A nonzero translation would be a lever the chest does not
        have, so this refuses one loudly rather than shifting ``head_pos`` by accident:
        a chest keypoint silently displaced by a few cm is invisible in every plot and
        poisons the torso target.
    """
    p = _resolve_calib(session_dir, fmt.HEAD_EXTRINSIC_FILE)
    if p is None:
        return np.eye(4), False
    d = json.load(open(p))
    T = np.asarray(d.get("T_tracker_head", np.eye(4).tolist()), dtype=np.float64)
    if not np.allclose(T[:3, 3], 0.0):
        raise ValueError(
            f"{p}: T_tracker_head must be ROTATION-ONLY (the chest keypoint has no TCP "
            f"lever), got translation {T[:3, 3].tolist()}")
    calibrated = bool(d.get("calibrated", False)) and not np.allclose(T, np.eye(4))
    return T, calibrated


def load_foot_extrinsic(session_dir: str) -> tuple[dict, bool]:
    """Return ``({"left_foot": 4x4, "right_foot": 4x4}, calibrated_bool)`` for the two FOOT
    pucks (``--feet recorded``). Identity + False if ``foot_extrinsic.json`` is absent —
    i.e. the RAW foot-puck axes are stored, which the recording format documents as a
    fact about the session rather than a default worth hiding.

    Same rotation-only rule as the chest's :func:`load_head_extrinsic`: a foot keypoint
    has no lever, so a translation is refused loudly.
    """
    ident = {r: np.eye(4) for r in fmt.FOOT_ROLES}
    p = _resolve_calib(session_dir, fmt.FOOT_EXTRINSIC_FILE)
    if p is None:
        return ident, False
    d = json.load(open(p))
    out = {}
    for r in fmt.FOOT_ROLES:
        T = np.asarray(d.get(f"T_tracker_{r}", np.eye(4).tolist()), dtype=np.float64)
        if not np.allclose(T[:3, 3], 0.0):
            raise ValueError(f"{p}: T_tracker_{r} must be ROTATION-ONLY (a foot keypoint "
                             f"has no lever), got translation {T[:3, 3].tolist()}")
        out[r] = T
    calibrated = bool(d.get("calibrated", False)) and not all(
        np.allclose(T, np.eye(4)) for T in out.values())
    return out, calibrated


#: Takes the converter refused to ingest, as ``(take_name, reason)`` — collected here
#: (not just printed) so the end-of-run statistics can name every one.
DROPPED_TAKES: list = []


def _note_drop(name: str, reason: str) -> None:
    DROPPED_TAKES.append((name, reason))


def _frames_in_spans(cam_t: np.ndarray, spans) -> np.ndarray:
    """``[T]`` float32 1.0 for camera frames within ``POSE_GLITCH_MARGIN_S`` of any span."""
    g = np.zeros(cam_t.shape[0], dtype=np.float32)
    for (gt0, gt1) in spans:
        lo = np.searchsorted(cam_t, gt0 - POSE_GLITCH_MARGIN_S, side="left")
        hi = np.searchsorted(cam_t, gt1 + POSE_GLITCH_MARGIN_S, side="right")
        g[lo:hi] = 1.0
    return g


def _drop_rotation_excursions(tracker: np.ndarray, label: str = "") -> tuple:
    """Drop the rows of every THERE-AND-BACK orientation excursion (a lighthouse
    orientation glitch: leaves at >400 deg/s and rejoins within ~0.4 s where it left).

    The rotational twin of the position deglitch, applied the same way — whole rows go,
    so the Slerp/interp knots stay aligned and the resampler bridges the hole. Without
    this a quiet-position rotation spike (measured 10-86 deg, 800-6000 deg/s on this
    rig) passes the position-only deglitch untouched and lands in the stored rotation
    label; the only remedy used to be ``--exclude`` of the whole take, which is what
    made 30-80% of a 5-puck session's takes "unusable" for one or two bad samples.

    Returns ``(tracker_kept, {"n_excursions", "n_samples", "spans"})`` where ``spans``
    are ``(t_start, t_end)`` wall-clock brackets of the dropped rows (masked downstream
    exactly like a BRIDGED position excursion — bridging is not repairing).
    """
    stats = {"n_excursions": 0, "n_samples": 0, "spans": []}
    if tracker is None or tracker.shape[0] < 3:
        return tracker, stats
    t = tracker[:, 0]
    exc = core.rotation_excursions(t, tracker[:, 5:9])
    if not exc:
        return tracker, stats
    drop = np.zeros(tracker.shape[0], dtype=bool)
    for e in exc:
        k = int(np.searchsorted(t, e["t"], side="left"))
        if k >= t.shape[0] or abs(t[k] - e["t"]) > 1e-9:
            continue
        a, b = k + 1, k + 1 + int(e["n_samples"])       # rows k+1 .. rejoin-1
        b = min(b, tracker.shape[0] - 1)                  # never drop the last row
        if b <= a:
            continue
        drop[a:b] = True
        stats["n_excursions"] += 1
        stats["n_samples"] += int(b - a)
        stats["spans"].append((float(t[k]), float(t[b])))
    if stats["n_excursions"]:
        print(f"  {label} rotation deglitch — dropped {stats['n_samples']} sample(s) in "
              f"{stats['n_excursions']} orientation excursion(s) "
              f"(peak {max(e['amp_deg'] for e in exc):.0f} deg)")
    return tracker[~drop], stats


def _sanitize_tracker_with_stats(tracker: "np.ndarray | None", stale_col: "int | None",
                                 label: str = "secondary",
                                 rot_deglitch: bool = True) -> tuple:
    """:func:`_sanitize_tracker` that ALSO returns the role's glitch severity dict (the
    same ``core.tracker_glitch_severity`` shape the primary hand gets, plus the
    ``rot_*`` keys), so a secondary role's corrupted spans can be MASKED per role
    instead of either ignored (the old behaviour) or answered by dropping the take.
    ``(None, None)`` when nothing live remains.
    """
    if tracker is None or tracker.shape[0] == 0:
        return None, None
    if stale_col is not None and tracker.shape[1] > stale_col:
        tracker = tracker[tracker[:, stale_col] < 0.5]
        if tracker.shape[0] == 0:
            print(f"  {label}: ALL frames stale -> dropping this keypoint entirely "
                  f"(a frozen pose would resample as a real, covered measurement)")
            return None, None
    tracker = tracker[core.make_strictly_increasing(tracker[:, 0])]
    sev = core.tracker_glitch_severity(tracker[:, 0], tracker[:, 2:5])
    if tracker.shape[0] >= 3:
        keep = core.deglitch_tracker_positions(tracker[:, 0], tracker[:, 2:5])
        n_gl = tracker.shape[0] - keep.shape[0]
        if n_gl > 0 or sev["n_unbridged"] > 0:
            print(f"  {label} tracker deglitch — dropped {n_gl} glitch sample(s), "
                  f"{sev['n_unbridged']} unbridged excursion(s), "
                  f"residual jump {sev['residual_max_jump_m']*100:.0f}cm")
        tracker = tracker[keep]
    rot = {"n_excursions": 0, "n_samples": 0, "spans": []}
    if rot_deglitch:
        tracker, rot = _drop_rotation_excursions(tracker, label=label)
    sev["rot_excursions"] = rot["n_excursions"]
    sev["rot_samples_dropped"] = rot["n_samples"]
    sev["rot_spans"] = rot["spans"]
    if tracker.shape[0] == 0:
        return None, None
    return tracker, sev


def _role_glitch_stats(sev: "dict | None", n_frames_flagged: int) -> dict:
    """One role's per-episode glitch provenance (aggregated over the session in main)."""
    sev = sev or {}
    return {
        "samples": int(sev.get("n", 0)),
        "pos_samples_dropped": int(sev.get("n_dropped", 0)),
        "pos_bridged_excursions": int(len(sev.get("bridged_spans", ()))),
        "pos_unbridged_excursions": int(sev.get("n_unbridged", 0)),
        "residual_jump_m_max": float(sev.get("residual_max_jump_m", 0.0)),
        "rot_excursions": int(sev.get("rot_excursions", 0)),
        "rot_samples_dropped": int(sev.get("rot_samples_dropped", 0)),
        "frames_flagged": int(n_frames_flagged),
    }


# ---------------------------------------------------------------------------
# Per-take loading + marker split -> one dict of UMI arrays per episode
# ---------------------------------------------------------------------------
def build_episodes_from_take(take_dir: str, gripper_calib, T_tracker_tcp,
                             out_res, jpeg_decode, image_resize,
                             *, hands: "str | None" = None,
                             hl_windows: bool = False,
                             T_tracker_head: "np.ndarray | None" = None,
                             feet: str = "ignore",
                             T_tracker_feet: "dict | None" = None,
                             rot_deglitch: bool = True) -> list[dict]:
    """Load one raw TAKE, split it into UMI episodes at its markers, resample each
    onto its own camera grid, and return a list of UMI data-array dicts (one per
    episode). Empty list if the take is unusable (no camera / no tracker).

    ``T_tracker_head`` is the CHEST puck's rotation-only body relabel
    (:func:`load_head_extrinsic`), applied to the 3pt ``head_*`` keypoint the way
    ``T_tracker_tcp`` is applied to the hands. ``None`` means "store the raw chest-puck
    axes" — the pre-2026-08-26 behavior, kept as the default so the LeRobot exporter
    (which calls this positionally and emits no head keypoint at all) is unaffected.

    ``hands`` selects which tracker roles to emit (see :data:`HANDS_MODES`):
    ``left`` | ``right`` | ``both`` | ``both_head``. ``None`` (the default) keeps the
    historical behavior EXACTLY — roles are inferred from which ``tracker_*.npy``
    files happen to exist. Setting it explicitly is what makes a mode a contract
    rather than an accident: a missing required tracker raises instead of silently
    producing a dataset with fewer keypoints than the caller asked for.

    ``hl_windows`` additionally emits the HL-v2 fixed-dt window arrays (see the
    HL_WINDOWS block at the top of this module). Off by default so this function's
    output stays byte-identical for the LeRobot exporter, which calls it positionally
    with a fixed signature (both new params are keyword-only for that reason).
    """
    cam_dir = os.path.join(take_dir, fmt.CAMERA_DIR)
    cam_index = np.load(os.path.join(cam_dir, fmt.CAMERA_INDEX_FILE))
    name = os.path.basename(take_dir)
    if cam_index.shape[0] == 0:
        print(f"  {name}: no camera frames, skipping")
        return []

    cam_t_all = cam_index[:, 0]
    # Enforce a strictly-increasing master timeline (drop stale-resend dups).
    keep = core.make_strictly_increasing(cam_t_all)
    cam_t = cam_t_all[keep]  # absolute wall-clock, strictly increasing
    # `t` comes from wall clock, which is NOT monotonic (chrony/NTP can step it).
    # A backward step of dt silently deletes ~dt*fps frames from the MASTER GRID;
    # a forward step punches a hole. Both used to be invisible here.
    if keep.size < cam_t_all.shape[0]:
        n_drop = int(cam_t_all.shape[0] - keep.size)
        print(f"  {name}: ⚠ camera timeline non-monotonic — "
              f"make_strictly_increasing DROPPED {n_drop}/{cam_t_all.shape[0]} frames "
              f"from the master grid (wall-clock step? duplicated timestamps?)")

    tracker = np.load(os.path.join(take_dir, fmt.TRACKER_FILE))
    gripper = np.load(os.path.join(take_dir, fmt.GRIPPER_FILE))
    # 2pt/3pt secondary trackers (LEFT-canonical: right hand + head/torso). Absent
    # in 1pt takes -> stay None and only robot0_* is written. Loaded as optional so
    # a legacy/1pt take is unaffected.
    tracker_right = _load_optional_npy(os.path.join(take_dir, fmt.TRACKER_RIGHT_FILE))
    tracker_head = _load_optional_npy(os.path.join(take_dir, fmt.TRACKER_HEAD_FILE))
    # 5pt FOOT pucks (``--feet recorded``): body references like the chest — pose only,
    # no camera / F-T / gripper — emitted as ``{left,right}_foot_pos/_rot_axis_angle``
    # and ``hl_{left,right}_foot_action_pose/_valid``. Ignored unless asked for, so
    # every existing session converts byte-identically.
    tracker_feet: dict = {}
    if feet == "recorded":
        tracker_feet = {
            "left_foot": _load_optional_npy(os.path.join(take_dir, fmt.TRACKER_LEFT_FOOT_FILE)),
            "right_foot": _load_optional_npy(os.path.join(take_dir, fmt.TRACKER_RIGHT_FOOT_FILE)),
        }
        _fmiss = [r for r, v in tracker_feet.items() if v is None or v.shape[0] == 0]
        if _fmiss:
            raise ValueError(
                f"{name}: --feet recorded needs tracker_{{left,right}}_foot.npy but "
                f"{_fmiss} are absent/empty in this take. Recorded mode was "
                f"{_load_take_meta(take_dir).get('mode')!r} (feet are 5pt only).")
    # RIGHT-hand per-hand MODALITIES (camera / F-T / gripper). Each is independent of
    # tracker_right: a puck needs no hardware, these do — so a 3pt take recorded before
    # the right-hand devices were wired up has tracker_right and none of these. All
    # optional, so every existing session converts byte-identically.
    gripper_right = _load_optional_npy(os.path.join(take_dir, fmt.GRIPPER_RIGHT_FILE))
    cam_right_dir = os.path.join(take_dir, fmt.CAMERA_RIGHT_DIR)
    cam_right_index = _load_optional_npy(os.path.join(cam_right_dir, fmt.CAMERA_INDEX_FILE))
    if cam_right_index is not None and cam_right_index.shape[0] == 0:
        cam_right_index = None

    # --- explicit --hands gating (fail loudly; never silently degrade) ----------
    # Without this the emitted keypoints are whatever files happen to be on disk, so
    # a mid-session tracker dropout quietly yields a smaller dataset than requested.
    if hands is not None:
        if hands not in HANDS_MODES:
            raise ValueError(f"unknown --hands {hands!r}; expected one of "
                             f"{sorted(HANDS_MODES)}")
        want = HANDS_MODES[hands]
        # 'left' is the PRIMARY role and always lives in tracker.npy; the other roles
        # are the sidecar files. (A right-only mode still reads tracker_right.npy —
        # the primary FILE is not re-pointed, so robot0_* keeps meaning "the primary
        # recorded hand" and only the HL/keypoint selection changes.)
        available = {"left": tracker if tracker.shape[0] else None,
                     "right": tracker_right, "head": tracker_head}
        missing = [r for r in want
                   if available.get(r) is None or available[r].shape[0] == 0]
        if missing:
            raise ValueError(
                f"{name}: --hands {hands} needs tracker role(s) {want} but "
                f"{missing} are absent/empty in this take. Recorded mode was "
                f"{_load_take_meta(take_dir).get('mode')!r}. Re-record, or pick a "
                f"--hands mode the session actually contains.")
        # Drop roles the mode does not want so they reach neither the zarr nor the
        # HL windows. The right hand's MODALITIES go with its tracker: emitting
        # camera1_rgb / robot1_ft_wrench for a hand whose pose was excluded would
        # produce force and pixels with no keypoint to attach them to.
        if "right" not in want:
            tracker_right = None
            gripper_right = None
            cam_right_index = None
            ft_right = None
        if "head" not in want:
            tracker_head = None
    # F/T: prefer the gravity/inertia-COMPENSATED sidecar when present (what
    # training wants); fall back to raw ft_wrench.npy. Both are the same sensor
    # frame/sign, so every downstream resample is value-agnostic. The raw file is
    # always kept on disk as the audit/safety net — we just don't train on it when
    # a compensated trace exists. take_meta.ft_compensated records which we used.
    take_meta = _load_take_meta(take_dir)
    ft_comp_path = os.path.join(take_dir, fmt.FT_COMP_FILE)
    ft_flag = take_meta.get("ft_compensated", False)  # True | False | "mixed" | absent
    if ft_flag is True and os.path.exists(ft_comp_path):
        ft = np.load(ft_comp_path)
        ft_source = "compensated"
    else:
        # raw-only, or 'mixed' provenance (fall back to the trustworthy raw file)
        ft = np.load(os.path.join(take_dir, fmt.FT_FILE))
        ft_source = "raw"
        if ft_flag == "mixed":
            print(f"  {name}: take_meta ft_compensated='mixed' (batch flag flipped "
                  f"mid-take); using RAW ft_wrench.npy")

    # RIGHT-hand F/T, same compensated-preferred rule but keyed on that hand's OWN
    # provenance (take_meta["hands"]["right"]["ft_compensated"]). The two sensors have
    # independent payload configs and tares, so reading the left hand's flag here is
    # exactly how an uncompensated right trace would get labelled compensated.
    ft_right = None
    ft_right_source = None
    _right_meta = (take_meta.get("hands") or {}).get("right") or {}
    _ft_right_raw_p = os.path.join(take_dir, fmt.FT_RIGHT_FILE)
    _ft_right_comp_p = os.path.join(take_dir, fmt.FT_COMP_RIGHT_FILE)
    if _right_meta.get("ft_compensated") is True and os.path.exists(_ft_right_comp_p):
        ft_right = np.load(_ft_right_comp_p)
        ft_right_source = "compensated"
    elif os.path.exists(_ft_right_raw_p):
        ft_right = np.load(_ft_right_raw_p)
        ft_right_source = "raw"
        if _right_meta.get("ft_compensated") == "mixed":
            print(f"  {name}: RIGHT take_meta ft_compensated='mixed'; using RAW "
                  f"{fmt.FT_RIGHT_FILE}")
    if ft_right is not None and ft_right.shape[0] == 0:
        ft_right, ft_right_source = None, None
    # Same non-finite assert as the primary: one NaN poisons every downstream mean.
    if ft_right is not None and not np.all(np.isfinite(ft_right[:, :7])):
        n_bad = int((~np.isfinite(ft_right[:, :7])).any(axis=1).sum())
        raise ValueError(
            f"{name}: {n_bad} non-finite RIGHT-hand F/T sample(s) in {ft_right_source} "
            f"wrench. Fix or drop the take (processing/audit_session.py reports these).")

    # F/T TARE provenance -> tare-transient spans. The compensator is auto-tared in
    # FREE SPACE at each episode start and settles over ~0.5 s, so the head of every
    # episode's wrench is a converging estimate. The recorder has written this
    # provenance for every take since day one and NOTHING read it, so the transient
    # reached the training label as if it were a measurement — and because a
    # free-space tare converges toward ~0 it is indistinguishable from "no contact"
    # in the values alone. Only meaningful for the COMPENSATED stream (a raw wrench
    # is never re-zeroed), so raw takes get no spans.
    def _tare_spans_for(events_file: str, arr: "np.ndarray | None", source: "str | None",
                        meta_wall) -> np.ndarray:
        """Tare-transient spans for ONE hand, from ITS OWN tare_events + wrench span."""
        tare_wall = None
        if source == "compensated":
            tare_events = _load_optional_npy(os.path.join(take_dir, events_file))
            if tare_events is not None and tare_events.shape[0]:
                # 96/97 takes carry TWO rows, and row 0 is a STALE LATCH from a previous
                # take (measured median -9.0 s, min -810 s relative to this take's first
                # camera frame). Keep only events inside this take's own F/T span, so a
                # span is never built around another take's wall-clock.
                tw = np.asarray(tare_events[:, 0], dtype=np.float64)
                if arr is not None and arr.shape[0]:
                    tw = tw[(tw >= arr[0, 0] - core.TARE_SETTLE_S) & (tw <= arr[-1, 0])]
                tare_wall = tw if tw.size else None
            if tare_wall is None and meta_wall is not None:
                tare_wall = np.asarray([meta_wall], dtype=np.float64)
        return core.tare_transient_spans(tare_wall)

    tare_spans = _tare_spans_for(fmt.TARE_EVENTS_FILE, ft, ft_source,
                                 take_meta.get("ft_tare_wall"))
    # The RIGHT hand tares INDEPENDENTLY (`_tare_all_ft` raises one request per sensor
    # under that sensor's own lock), so its transient is its own interval and its
    # provenance is its own file. Sharing the left hand's spans would mask the wrong
    # samples in both directions.
    tare_spans_right = _tare_spans_for(fmt.TARE_EVENTS_RIGHT_FILE, ft_right,
                                       ft_right_source,
                                       _right_meta.get("ft_tare_wall"))

    # A single non-finite F/T sample poisons a TAIL, not a point: the HL bin means are
    # computed from a cumulative sum, so one NaN makes every later bin NaN — measured
    # 133/144 history bins and 279/288 future bins from one NaN at index 100 — and the
    # validity masks are pure TIMESTAMP tests, so 17 history windows and 278 future bins
    # stayed flagged VALID. Real data is clean (0 non-finite across 388 arrays) and
    # audit_session checks it as a separate operator step, so this is a cheap assert
    # against a catastrophic-but-latent failure rather than a hot-path cost.
    if ft.shape[0] and not np.all(np.isfinite(ft[:, :7])):
        n_bad = int((~np.isfinite(ft[:, :7])).any(axis=1).sum())
        raise ValueError(
            f"{name}: {n_bad} non-finite F/T sample(s) in {ft_source} wrench. One NaN "
            f"propagates through the cumulative-sum bin means to every LATER bin while "
            f"the timestamp-based validity masks still report those windows valid. Fix "
            f"or drop the take (processing/audit_session.py reports these).")
    markers = _load_markers(take_dir)

    if tracker.shape[0] == 0:
        print(f"  {name}: tracker.npy empty, skipping")
        return []

    # The DM4310 absolute encoder re-references per power-cycle, so the repo
    # gripper_range.json endpoints (i2rt nominal) usually do NOT match this take's
    # live encoder — mapping with them clamps the whole width signal to one rail.
    # The recorder stored the homed endpoints per-sample (gripper.npy cols 3/4), so
    # prefer those, keeping the physical width_*_m from the JSON calib (H-fix).
    take_gripper_calib, calib_src = core.effective_gripper_calib(gripper, gripper_calib)
    if calib_src == "recorded":
        print(f"  {name}: gripper endpoints from RECORDED homing "
              f"(enc {take_gripper_calib.enc_closed:.3f}..{take_gripper_calib.enc_open:.3f} rad)")
    elif calib_src == "json" and gripper_calib is not None:
        print(f"  {name}: gripper endpoints from gripper_range.json "
              f"(no/degenerate recorded endpoints in gripper.npy)")

    # Drop stale-repeat tracker frames (col 'stale'==1) so a tracker dropout's
    # frozen-pose run isn't interpolated as real motion (H4). Keep the live ones.
    stale_col = fmt.TRACKER_COLS.index("stale") if "stale" in fmt.TRACKER_COLS else None
    if stale_col is not None and tracker.shape[1] > stale_col:
        live = tracker[:, stale_col] < 0.5
        if not live.any():
            # Same reasoning as _sanitize_tracker: keeping an all-stale stream yields one
            # frozen pose that resamples as a real, fully-covered eef label. The PRIMARY
            # pose is the action label, so drop the take rather than emit it.
            print(f"  {name}: DROPPING take — PRIMARY tracker is ALL stale "
                  f"({tracker.shape[0]} frames); a frozen pose would become the eef label.",
                  file=sys.stderr)
            _note_drop(name, f"primary tracker ALL stale ({tracker.shape[0]} frames)")
            return []
        tracker = tracker[live]
    # B1: EVERY source stream must be strictly increasing before Slerp/interp1d
    # (a single WS reconnect stale-resend duplicates a timestamp; scipy Slerp
    # raises ValueError on non-monotone knots). Sanitize each stream ONCE here.
    tracker = tracker[core.make_strictly_increasing(tracker[:, 0])]
    # Drop isolated lighthouse position glitches (physically-impossible teleports)
    # AFTER the monotone sanitize, BEFORE resampling — otherwise interp1d/Slerp
    # pass a bad pose straight into the camera-grid eef label. Rows dropped whole
    # (t+pos+quat together) so Slerp knots stay aligned; conservative (see core).
    # tracker_sev feeds the episode-drop gate + zarr provenance below.
    tracker_sev = core.tracker_glitch_severity(tracker[:, 0], tracker[:, 2:5])
    if tracker.shape[0] >= 3:
        keep_tr = core.deglitch_tracker_positions(tracker[:, 0], tracker[:, 2:5])
        n_gl = tracker.shape[0] - keep_tr.shape[0]
        if n_gl > 0 or tracker_sev["n_unbridged"] > 0:
            print(f"  {name}: tracker deglitch — dropped {n_gl} glitch sample(s), "
                  f"{tracker_sev['n_unbridged']} unbridged excursion(s), "
                  f"residual jump {tracker_sev['residual_max_jump_m']*100:.0f}cm")
            tracker = tracker[keep_tr]
    # ROTATION excursions on the primary: dropped rows + spans, masked like bridged
    # position excursions (they reach `_corrupt_spans` via tracker_sev["rot_spans"]).
    _rot = {"n_excursions": 0, "n_samples": 0, "spans": []}
    if rot_deglitch:
        tracker, _rot = _drop_rotation_excursions(tracker, label=f"{name} left")
    tracker_sev["rot_excursions"] = _rot["n_excursions"]
    tracker_sev["rot_samples_dropped"] = _rot["n_samples"]
    tracker_sev["rot_spans"] = _rot["spans"]
    if gripper.shape[0] > 0:
        gripper = gripper[core.make_strictly_increasing(gripper[:, 0])]
    if ft.shape[0] > 0:
        ft = ft[core.make_strictly_increasing(ft[:, 0])]
    # Same monotone hygiene for the RIGHT-hand streams (a WS reconnect stale-resend
    # duplicates a timestamp, and interp1d/searchsorted both need sorted knots).
    if gripper_right is not None and gripper_right.shape[0] > 0:
        gripper_right = gripper_right[core.make_strictly_increasing(gripper_right[:, 0])]
    else:
        gripper_right = None
    if ft_right is not None:
        ft_right = ft_right[core.make_strictly_increasing(ft_right[:, 0])]
    if cam_right_index is not None:
        cam_right_index = cam_right_index[core.make_strictly_increasing(cam_right_index[:, 0])]
        if cam_right_index.shape[0] == 0:
            cam_right_index = None
    # Same stale-drop + strictly-increasing hygiene for the secondary trackers, so a
    # right/head dropout's frozen run isn't interpolated as motion and Slerp never
    # sees a duplicate knot. Cleared to None if nothing live remains.
    tracker_right, sev_right = _sanitize_tracker_with_stats(
        tracker_right, stale_col, label=f"{name} right", rot_deglitch=rot_deglitch)
    tracker_head, sev_head = _sanitize_tracker_with_stats(
        tracker_head, stale_col, label=f"{name} head", rot_deglitch=rot_deglitch)
    sev_feet: dict = {}
    for _fr in list(tracker_feet):
        tracker_feet[_fr], sev_feet[_fr] = _sanitize_tracker_with_stats(
            tracker_feet[_fr], stale_col, label=f"{name} {_fr}", rot_deglitch=rot_deglitch)
    # Every emitted role's glitch severity, keyed by role. The PRIMARY's is the take-drop
    # gate above; the others are MASKED per role (windows + anchors) and never drop a take.
    role_sev: dict = {"left": tracker_sev}
    if tracker_right is not None:
        role_sev["right"] = sev_right
    if tracker_head is not None:
        role_sev["head"] = sev_head
    for _fr, _ftr in tracker_feet.items():
        if _ftr is not None:
            role_sev[_fr] = sev_feet[_fr]
    if hands is not None:
        role_sev = {r: v for r, v in role_sev.items()
                    if r in HANDS_MODES[hands] or r in tracker_feet}

    # Split the camera timeline into episode index-ranges at the markers.
    segments = core.split_take_into_episodes(cam_t, markers)
    n_drop = (len(markers) + 1) - len(segments)
    if not segments:
        print(f"  {name}: no usable episode segments after marker split, skipping")
        _note_drop(name, "no usable episode segments after marker split")
        return []
    if n_drop > 0:
        print(f"  {name}: dropped {n_drop} too-short segment(s) (<2 frames) at marker boundaries")

    # Whole-take tracker-glitch gate: if the PRIMARY tracker has an excursion the
    # deglitch could not bridge (an unbridged teleport, or a residual jump too big
    # to be real hand motion), the resampled eef_pos LABEL is corrupted and no
    # amount of downstream masking recovers it — DROP the take rather than train on
    # it. (Per-take here because the split is by markers; with 0 markers = 1 episode
    # this is the episode. The audit tool uses the SAME thresholds.)
    tracker_unbridgeable = (
        tracker_sev["n_unbridged"] >= TRACKER_DROP_MAX_UNBRIDGED
        or tracker_sev["residual_max_jump_m"] >= TRACKER_DROP_RESIDUAL_JUMP_M)
    if tracker_unbridgeable:
        print(f"  {name}: DROPPING take — unbridgeable tracker glitch "
              f"({tracker_sev['n_unbridged']} unbridged excursion(s), residual jump "
              f"{tracker_sev['residual_max_jump_m']*100:.0f}cm, peak "
              f"{tracker_sev['peak_speed_mps']:.0f} m/s). Corrupts eef_pos label.",
              file=sys.stderr)
        _note_drop(name, f"PRIMARY tracker unbridgeable: {tracker_sev['n_unbridged']} "
                         f"unbridged excursion(s), residual jump "
                         f"{tracker_sev['residual_max_jump_m']*100:.0f} cm")
        return []

    episodes: list[dict] = []
    for seg_i, (a, b) in enumerate(segments):
        seg_cam_t = cam_t[a:b]
        tag = f"{name} ep{seg_i}" if len(segments) > 1 else name
        # Advisory continuity/coverage check per EPISODE (review #9/#10): warn if the
        # tracker doesn't span this segment's camera window — then the pose resample
        # is edge-held (clamp-extrapolated), e.g. a tracker dropout localized to this
        # demo. Non-fatal; just surfaces silent bad coverage instead of hiding it.
        rep = core.validate_episode_timeline(seg_cam_t, tracker[:, 0], tracker[:, 2:5])
        if not rep["ok"]:
            for w in rep["warnings"]:
                if "does not fully cover" in w or "position jump" in w:
                    print(f"  WARN {tag}: {w}")
        arrs = _build_one_episode(
            seg_cam_t, keep[a:b], cam_dir, tracker, gripper, ft,
            take_gripper_calib, T_tracker_tcp, out_res, jpeg_decode, image_resize,
            tracker_right=tracker_right, gripper_right=gripper_right,
            tracker_head=tracker_head, T_tracker_head=T_tracker_head,
            ft_right=ft_right, cam_right_dir=cam_right_dir,
            cam_right_index=cam_right_index, take_tag=tag,
            tracker_feet=tracker_feet or None, T_tracker_feet=T_tracker_feet)
        # ===== HL-v2 fixed-dt policy windows (additive; off unless requested) =====
        # Built per EPISODE so a window never crosses a marker boundary in its
        # validity mask, but resampled from the TAKE-scoped raw streams so the force
        # history of an early anchor can legitimately see the physically-continuous
        # wrench just before the episode's first camera frame.
        if hl_windows:
            role_trackers = {"left": tracker}
            if tracker_right is not None and tracker_right.shape[0]:
                role_trackers["right"] = tracker_right
            if tracker_head is not None and tracker_head.shape[0]:
                role_trackers["head"] = tracker_head
            for _fr, _ftr in tracker_feet.items():
                if _ftr is not None and _ftr.shape[0]:
                    role_trackers[_fr] = _ftr
            if hands is not None:
                # The feet ride along with `--feet recorded` whatever the --hands mode:
                # HANDS_MODES names the hand/chest contract, the feet are orthogonal.
                role_trackers = {r: v for r, v in role_trackers.items()
                                 if r in HANDS_MODES[hands] or r in tracker_feet}
            g_src = None
            if gripper.shape[0] > 0 and take_gripper_calib is not None:
                g_src = (gripper[:, 0], take_gripper_calib.to_width_m(gripper[:, 1]))
            # The RIGHT hand's HL sources, supplied ONLY for a mode that declares that
            # hardware (`both_head_dual`); for every other mode the dicts are simply
            # never consulted, so the emitted key set is unchanged.
            _ft_roles = hl_ft_roles(hands)
            _grip_roles = hl_gripper_roles(hands)
            role_ft = {"left": (ft, bool(arrs["_ft_valid"]), tare_spans)}
            role_gsrc = {"left": g_src}
            if "right" in _ft_roles:
                role_ft["right"] = (ft_right, ft_right is not None and ft_right.shape[0] > 0,
                                    tare_spans_right)
            if "right" in _grip_roles:
                r_src = None
                if (gripper_right is not None and gripper_right.shape[0] > 0
                        and gripper_calib is not None):
                    # Same endpoint discipline as robot1_gripper_width: the RIGHT DM4310's
                    # OWN recorded homing, never the left hand's range (which would clamp
                    # every right width to one rail).
                    r_calib, _ = core.effective_gripper_calib(gripper_right, gripper_calib)
                    r_src = (gripper_right[:, 0], r_calib.to_width_m(gripper_right[:, 1]))
                role_gsrc["right"] = r_src
            arrs.update(_build_hl_windows(
                seg_cam_t, role_trackers, ft, g_src, T_tracker_tcp, tare_spans,
                ft_valid=bool(arrs["_ft_valid"]),
                glitch_spans=_corrupt_spans(tracker_sev),
                role_glitch_spans={r: _corrupt_spans(v) for r, v in role_sev.items()
                                   if v is not None},
                role_extr=T_tracker_feet,
                T_tracker_head=T_tracker_head,
                ft_roles=_ft_roles, gripper_roles=_grip_roles,
                role_ft=role_ft, role_gripper_src=role_gsrc))
        # ===== per-frame pose-glitch mask (tier 2 above) =====
        # 1.0 where this camera frame's eef_pos / action label is corrupted by a tracker
        # excursion — whether or not the deglitch bridged it (see _corrupt_spans). A
        # training sampler excludes the windows that straddle a flagged frame; everything
        # else in the take stays usable.
        glitch = _frames_in_spans(seg_cam_t, _corrupt_spans(tracker_sev))
        arrs["robot0_pose_glitch"] = glitch[:, None]     # [T, 1], like gripper_width
        # ===== per-ROLE pose-glitch masks (tier 2, every emitted tracker) =====
        # `<role>_pose_glitch` [T,1] per emitted role (left == robot0_pose_glitch by
        # construction) and their union `pose_glitch_any`. A secondary role's excursion
        # corrupts THAT role's anchor pose exactly as the primary's does, and used to be
        # answered only by `--exclude`-ing the take; a sampler that reads these drops
        # just the windows straddling the corrupted frames instead.
        any_glitch = glitch.copy()
        glitch_stats: dict = {}
        for _r, _sv in role_sev.items():
            g_r = glitch if _r == "left" else _frames_in_spans(seg_cam_t, _corrupt_spans(_sv))
            arrs[f"{_r}_pose_glitch"] = g_r[:, None]
            any_glitch = np.maximum(any_glitch, g_r)
            glitch_stats[_r] = _role_glitch_stats(_sv, int(g_r.sum()))
        arrs["pose_glitch_any"] = any_glitch[:, None]
        arrs["_glitch_stats"] = glitch_stats
        n_gl_frames = int(glitch.sum())
        if n_gl_frames:
            print(f"  {tag}: {n_gl_frames}/{glitch.shape[0]} frames flagged "
                  f"robot0_pose_glitch ({tracker_sev['n_unbridged']} unbridged + "
                  f"{len(tracker_sev.get('bridged_spans', []))} bridged tracker "
                  f"excursion(s)) -> the training sampler drops only the windows that "
                  f"straddle them")
        # Per-episode tracker-deglitch provenance (private; aggregated in main()).
        arrs["_tracker_dropped"] = int(tracker_sev["n_dropped"])
        arrs["_tracker_unbridged"] = int(tracker_sev["n_unbridged"])
        arrs["_tracker_residual_jump_m"] = float(tracker_sev["residual_max_jump_m"])
        # Per-episode F/T provenance (private; popped + aggregated in main()).
        arrs["_ft_source"] = ft_source
        # Which compensation engine produced the sidecar (only meaningful when we
        # actually used it, i.e. ft_source == "compensated") + whether its sign was
        # verified. Defaults keep raw/legacy takes uncontaminating the aggregate.
        arrs["_ft_algo"] = take_meta.get("ft_comp_algo") if ft_source == "compensated" else None
        arrs["_ft_sign_verified"] = take_meta.get("ft_sign_verified", True)
        # The recorder's own camera-grid verdict. It can bucket dt against the
        # true nominal CAPTURE period; the converter only sees the published
        # timestamps and cannot. None = pre-fix recording with no verdict.
        arrs["_camera_grid_uniform"] = take_meta.get("camera_grid_uniform", None)
        episodes.append(arrs)
        # Name the RIGHT-hand modalities that actually made it in, so a missing right
        # camera / Bota / gripper is visible per episode in the convert log rather than
        # only as an absent zarr key at the end.
        extra = [n for n, k in (("cam1", "camera1_rgb"), ("ft1", "robot1_ft_wrench"),
                                ("grip1", "robot1_gripper_width")) if k in arrs]
        rhs = f", right: {'+'.join(extra)}" if extra else ""
        print(f"  {tag}: T={arrs['camera0_rgb'].shape[0]} (F/T: {ft_source}"
              f"{f'/{ft_right_source}' if ft_right_source else ''}{rhs})")
    return episodes


def _resample_tracker_to_6d(tracker: np.ndarray, cam_t: np.ndarray,
                            T_tracker_tcp: "np.ndarray | None") -> np.ndarray:
    """Resample ONE sanitized tracker array (RAW world, wxyz cols 5:9) onto the
    camera grid and return its UMI 6-DOF pose [T,6] ([:3]=pos, [3:]=rotvec).

    ``T_tracker_tcp`` composes a fixed tracker->TCP extrinsic (the same helper the
    primary hand uses); for the head/torso puck the caller passes its rotation-only
    chest relabel instead (a body reference has no puck->TCP transform, but its puck is
    still bolted on rotated). Pass ``None`` to emit the RAW tracker pose with NO
    extrinsic at all. Same wxyz->xyzw conversion point +
    clamped Slerp/interp as the primary path, so a secondary keypoint is numerically
    consistent with robot0."""
    tr_t = tracker[:, 0]
    pos_grid = core.resample_position(tr_t, tracker[:, 2:5], cam_t)
    rot_grid = core.resample_rotation_wxyz(tr_t, tracker[:, 5:9], cam_t)
    if T_tracker_tcp is not None:
        pos_grid, rot_grid = core.apply_tcp_extrinsic(pos_grid, rot_grid, T_tracker_tcp)
    return core.pose_to_6d(pos_grid, rot_grid)  # [T,6]


def _hl_prefix(role: str) -> str:
    """Zarr key prefix for one role's HL-v2 window arrays."""
    return f"hl_{role}_"


def _hands_flag_for_roles(roles: "set | frozenset",
                          ft_roles: "set | frozenset | None" = None,
                          gripper_roles: "set | frozenset | None" = None) -> "str | None":
    """Reverse of :data:`HANDS_MODES`: the ``--hands`` value a role set corresponds to.

    Used to stamp ``attrs.hands_mode`` when the operator ran ``--hl-windows`` WITHOUT
    ``--hands`` (a supported invocation that emits every tracker on disk). Returns None
    for a combination no mode describes, so an unusual one is recorded as unknown rather
    than mislabelled as one of the modes.

    The tracker roles ALONE are no longer a unique key: ``both_head`` and
    ``both_head_dual`` need the identical three pucks and differ only in which hands carry
    force/gripper keys. So the match is over the full triple, and callers that know the
    emitted key set should pass ``ft_roles``/``gripper_roles`` too. Omitting them means
    "the inferred default" (:data:`FT_ROLES` / :data:`GRIPPER_ROLES`, left-only) — which
    is what an inference-mode conversion actually emits, and never the dual mode.
    """
    want = frozenset(roles)
    want_ft = frozenset(FT_ROLES if ft_roles is None else ft_roles) & want
    want_grip = frozenset(GRIPPER_ROLES if gripper_roles is None else gripper_roles) & want
    for flag, mode_roles in HANDS_MODES.items():
        if frozenset(mode_roles) != want:
            continue
        if (frozenset(HL_FT_ROLES[flag]) & want) != want_ft:
            continue
        if (frozenset(HL_GRIPPER_ROLES[flag]) & want) != want_grip:
            continue
        return flag
    return None


def _build_hl_windows(cam_t: np.ndarray, role_trackers: dict, ft: np.ndarray,
                      gripper_width_src: "tuple[np.ndarray, np.ndarray] | None",
                      T_tracker_tcp: np.ndarray, tare_spans: np.ndarray,
                      ft_valid: bool,
                      glitch_spans: "list | None" = None,
                      T_tracker_head: "np.ndarray | None" = None,
                      ft_roles: "tuple[str, ...] | None" = None,
                      gripper_roles: "tuple[str, ...] | None" = None,
                      role_ft: "dict | None" = None,
                      role_gripper_src: "dict | None" = None,
                      role_glitch_spans: "dict | None" = None,
                      role_extr: "dict | None" = None) -> dict:
    """Build every HL-v2 fixed-dt window array for ONE episode.

    ``cam_t`` is the episode's ABSOLUTE camera timeline (each frame is an anchor t0).
    ``role_trackers`` maps role -> sanitized raw tracker array; only those roles get
    keys, which is how ``--hands`` selection reaches the zarr. ``gripper_width_src``
    is ``(t, width_m)`` from the raw encoder stream, or None.

    ``ft_roles`` / ``gripper_roles`` are the roles that get force / gripper keys; they
    default to the module-level LEFT-only tuples so an existing caller is unchanged.
    ``role_ft`` / ``role_gripper_src`` supply the SECOND hand's streams:

    * ``role_ft[role] = (ft_array, ft_valid_bool, tare_spans)`` — each hand's own Bota has
      its own payload config, its own tare, and its own dropouts, so nothing here is
      shared between them. Reading the left hand's tare spans for the right wrench is
      exactly how a converging tare transient would be labelled a measurement.
    * ``role_gripper_src[role] = (t, width_m)`` — each DM4310 re-references per
      power-cycle, so each hand's endpoints come from ITS OWN recording.

    A role listed in ``ft_roles``/``gripper_roles`` with no entry in these dicts falls
    back to the primary ``ft``/``gripper_width_src`` positional arguments (which is how
    ``left`` is supplied), and a role whose stream is genuinely absent is ZERO-FILLED and
    flagged invalid — never omitted, because the zarr writer requires every episode to
    agree on the key set.

    Targets are resampled from the RAW streams (~77 Hz tracker, ~178 Hz F/T, ~29 Hz
    encoder), never from the ~11.9 Hz camera grid — see
    ``core.resample_pose_uniform_future``. Poses stay ABSOLUTE (aligned world); the
    anchor-relative transform is the trainer's choice, not the recording's.
    """
    out: dict = {}
    T = cam_t.shape[0]
    ep_start_t = float(cam_t[0])
    ep_end_t = float(cam_t[-1])
    ft_roles = FT_ROLES if ft_roles is None else tuple(ft_roles)
    gripper_roles = GRIPPER_ROLES if gripper_roles is None else tuple(gripper_roles)
    role_ft = dict(role_ft or {})
    role_gripper_src = dict(role_gripper_src or {})
    role_glitch_spans = dict(role_glitch_spans or {})
    role_extr = dict(role_extr or {})
    glitch_masked_steps: dict = {}

    # Pose/gripper step validity: inside the episode AND inside the source stream's
    # own coverage. The second term is load-bearing: resample_pose_uniform_future
    # CLAMPS past the tracker's span (endpoint-held), so a mid-episode tracker dropout
    # yields 16 identical frozen positions that would otherwise be fully supervised —
    # training the policy to predict "stop moving". Computed PER ROLE below, since a
    # right-hand dropout must not invalidate the left hand's targets.
    ep_valid = core.future_step_valid(np.zeros(0), cam_t, ep_end_t)
    # Per-step bin bounds, for span-overlap tests against glitch/tare intervals.
    step_hi = core.future_step_times(cam_t)                 # (T, H)
    step_lo = step_hi - core.ACTION_DT

    def _force_masks(f_arr: np.ndarray, f_ok: bool, spans: np.ndarray):
        """``(hist_valid[T], step_valid[T,H])`` for ONE hand's wrench stream.

        Force validity needs real F/T coverage AND tare-cleanliness. A hand with no F/T
        at all is entirely force-invalid rather than silently zero — a free-space-tared
        wrench is ~0 by construction, so "no data" and "no contact" are numerically
        indistinguishable and only the mask can tell them apart.
        """
        f_t = f_arr[:, 0] if f_arr is not None and f_arr.shape[0] else np.zeros(0)
        if not (f_ok and f_t.shape[0]):
            return (np.zeros(T, dtype=bool),
                    np.zeros((T, core.ACTION_HORIZON), dtype=bool))
        # episode_start_t CONFINES the history window: a marker boundary is a
        # demonstration boundary, so an anchor just after one must not inherit the
        # previous demo's contact. Passed to the mask and the resampler identically.
        return (core.force_history_valid(f_t, cam_t, tare_spans=spans,
                                        episode_start_t=ep_start_t),
                core.future_step_valid(f_t, cam_t, ep_end_t, tare_spans=spans))

    for role, tracker in role_trackers.items():
        p = _hl_prefix(role)
        # The head is a body reference: no puck->TCP extrinsic — it gets its OWN
        # rotation-only chest relabel instead (same rule as the camera-grid path, so
        # hl_head_action_pose and head_rot_axis_angle can never disagree).
        if role in role_extr:
            extr = role_extr[role]            # a FOOT: its own rotation-only relabel
        else:
            extr = T_tracker_head if role == "head" else T_tracker_tcp
        out[f"{p}action_pose"] = core.resample_pose_uniform_future(
            tracker, cam_t, extr).astype(np.float32)                 # [T, H, 6]
        # PER-ROLE validity: a step is trainable only if THIS role's tracker actually
        # covered it. Without the tracker term, a dropout's clamp-extrapolated frozen
        # pose is supervised as if measured.
        pose_valid = ep_valid & core.future_step_valid(
            tracker[:, 0], cam_t, ep_end_t)
        # ...and not inside a tracker excursion, BRIDGED OR NOT (``_corrupt_spans``).
        # robot0_pose_glitch is per-ANCHOR [T,1], but each anchor now owns 16 targets
        # spanning a full second, so an anchor OUTSIDE a glitch can still predict straight
        # through one. Measured on the shipped 92-episode session: 759 of 273,744
        # pose-target steps on 239 anchors (1.40%) were supervised at full loss weight
        # while inside an unbridged excursion, with inter-step jumps to 350 mm and 5 steps
        # implying >5 m/s — physically impossible labels. Reuse the same spans the
        # per-anchor mask is built from, so the two can never disagree.
        # The PRIMARY's spans (historical: every role was masked by them) UNIONED with
        # THIS role's own corrupted spans — a right/chest/foot excursion is masked where
        # it happened, never by dropping the take.
        _spans = list(glitch_spans or []) + list(role_glitch_spans.get(role, []))
        _before = pose_valid.copy()
        if _spans:
            pose_valid = pose_valid & ~core.window_overlaps_spans(
                step_lo, step_hi, np.asarray(_spans, dtype=np.float64))
        glitch_masked_steps[role] = int((_before & ~pose_valid).sum())
        # ...and physically plausible. This is the LAST line of defence and it catches a
        # failure the span masks structurally cannot: a ROTATION glitch. The deglitch is
        # POSITION-only, so a 19 deg one-frame orientation spike passes it untouched and
        # then the 0.23 m puck->TCP lever turns it into a 73 mm TCP jump (real: clean94
        # take_031 frame 2). Speed gate measured against the camera-grid step
        # distribution, not free-hand speed — see core.MAX_TARGET_SPEED_MPS.
        pose_valid = pose_valid & ~core.implausible_step_mask(
            out[f"{p}action_pose"], core.ACTION_DT)
        out[f"{p}action_valid"] = pose_valid.astype(np.float32)       # [T, H]

        if role in gripper_roles:
            # ALWAYS emit the key (never conditionally), paired with a validity mask.
            # The zarr writer requires every episode to agree on the key set, so a
            # conditional key made ONE take with a dead gripper stream abort the whole
            # session ("present in only SOME episodes"). Zero-fill + valid=0 instead:
            # the trainer masks those steps, and the rest of the session converts.
            g_src = role_gripper_src.get(role, gripper_width_src)
            if g_src is not None:
                g_t, g_w = g_src
                out[f"{p}action_gripper"] = core.resample_gripper_uniform_future(
                    g_t, g_w, cam_t)                                 # [T, H]
                # The encoder resample clamps too, so gate on ITS coverage — not the
                # tracker's. A dead encoder with a live tracker must mask only gripper.
                out[f"{p}action_gripper_valid"] = (
                    ep_valid & core.future_step_valid(g_t, cam_t, ep_end_t)
                ).astype(np.float32)
            else:
                out[f"{p}action_gripper"] = np.zeros((T, core.ACTION_HORIZON), np.float32)
                out[f"{p}action_gripper_valid"] = np.zeros((T, core.ACTION_HORIZON), np.float32)

        if role in ft_roles:
            # PER HAND: its own wrench array, its own ft_valid flag and its own tare
            # spans. `role_ft` carries the second hand; the primary falls through to the
            # positional `ft`/`ft_valid`/`tare_spans` so the single-hand path is unchanged.
            f_arr, f_ok, f_spans = role_ft.get(role, (ft, ft_valid, tare_spans))
            f_arr = np.zeros((0, 7)) if f_arr is None else np.asarray(f_arr)
            f_t = f_arr[:, 0] if f_arr.shape[0] else np.zeros(0)
            f_w = f_arr[:, 1:7] if f_arr.shape[0] else np.zeros((0, 6))
            f_hist_valid, f_step_valid = _force_masks(f_arr, bool(f_ok), f_spans)
            out[f"{p}force_history"] = core.resample_wrench_uniform_past(
                f_t, f_w, cam_t, episode_start_t=ep_start_t)         # [T, L, 6]
            out[f"{p}force_history_valid"] = f_hist_valid.astype(np.float32)[:, None]
            out[f"{p}action_force"] = core.resample_wrench_uniform_future(
                f_t, f_w, cam_t)                                     # [T, H, 6]
            out[f"{p}action_force_valid"] = f_step_valid.astype(np.float32)
    out["_glitch_masked_steps"] = glitch_masked_steps
    return out


def _build_one_episode(cam_t, cam_src_idx, cam_dir, tracker, gripper, ft,
                       gripper_calib, T_tracker_tcp, out_res,
                       jpeg_decode, image_resize,
                       tracker_right=None, gripper_right=None,
                       tracker_head=None, T_tracker_head=None, ft_right=None,
                       cam_right_dir=None,
                       cam_right_index=None, take_tag="",
                       tracker_feet=None, T_tracker_feet=None) -> dict:
    """Resample all modalities onto ONE episode's camera grid (cam_t, absolute
    wall-clock). ``cam_src_idx`` are the original JPEG file indices for the kept
    frames. ``episode_timestamps`` is emitted RELATIVE (first frame == 0).

    LEFT-canonical multi-keypoint (2pt/3pt): when ``tracker_right`` / ``tracker_head``
    are given (already sanitized: stale-dropped + strictly-increasing), this ALSO
    emits ``robot1_*`` (right hand, with the SAME tracker->TCP extrinsic as robot0 —
    the wrists are FK-symmetric per DECISION-D) and ``head_pos``/``head_rot_axis_angle``
    (head/chest — NO puck->TCP extrinsic; it gets its own rotation-only chest body
    relabel ``T_tracker_head``, which leaves ``head_pos`` untouched).

    RIGHT-hand per-hand modalities are each independent and each OPTIONAL:
      * ``gripper_right``   -> ``robot1_gripper_width``
      * ``ft_right``        -> ``robot1_ft_wrench``
      * ``cam_right_index`` -> ``camera1_rgb`` (paired to the LEFT/master grid by
        causal frame SELECTION — see core.resample_frame_index; images cannot be
        interpolated, and lag is a safer error than a future-frame leak)
    All three additionally require ``tracker_right``: a hand whose POSE was not emitted
    gets no force/pixels/width either, because those would be measurements with no
    keypoint to attach them to (and ``--hands left`` drops the whole hand for that reason).
    A missing one is simply not emitted. NONE of them is zero-filled: a zero-filled
    "present" key reads as a real measurement downstream (masked in, trained on), so
    "no right-hand sensor" and "the right-hand sensor read zero" must stay distinct.
    The one exception is ``robot0_ft_wrench``, which has always been zero-filled +
    flagged via ``ft_valid`` for schema stability on the PRIMARY hand.

    The additive keys are ignored by single-hand tooling (convert_to_lerobot, old UMI
    loaders) — robot0_* / camera0_rgb are untouched."""
    T = cam_t.shape[0]

    # --- pose: resample raw tracker (world, wxyz) then apply TCP extrinsic ---
    tr_t = tracker[:, 0]
    tr_pos = tracker[:, 2:5]
    tr_quat_wxyz = tracker[:, 5:9]
    pos_grid = core.resample_position(tr_t, tr_pos, cam_t)
    rot_grid = core.resample_rotation_wxyz(tr_t, tr_quat_wxyz, cam_t)
    tcp_pos, tcp_rot = core.apply_tcp_extrinsic(pos_grid, rot_grid, T_tracker_tcp)
    tcp6 = core.pose_to_6d(tcp_pos, tcp_rot)  # [T,6]

    eef_pos = tcp6[:, :3].astype(np.float32)
    eef_rot = tcp6[:, 3:].astype(np.float32)
    demo_start = np.broadcast_to(tcp6[0], (T, 6)).astype(np.float32).copy()
    demo_end = np.broadcast_to(tcp6[-1], (T, 6)).astype(np.float32).copy()

    # --- gripper width (absolute metres) ---
    if gripper.shape[0] > 0 and gripper_calib is not None:
        width_m = gripper_calib.to_width_m(gripper[:, 1])
        gw = core.resample_gripper_width(gripper[:, 0], width_m, cam_t)
    else:
        gw = np.zeros((T, 1), dtype=np.float32)

    # --- F/T wrench (anti-aliased downsample); ALWAYS present, length T ---
    ft_valid = ft.shape[0] > 0
    if ft_valid:
        wrench = core.antialias_resample_wrench(ft[:, 0], ft[:, 1:7], cam_t)
    else:
        wrench = np.zeros((T, 6), dtype=np.float32)

    # --- F/T hi-rate per-step CAUSAL window [T, K, 6] (MuSe-style target) ---
    # Emitted for EVERY episode when FT_HIRES_K > 0 (both the valid and zero-fill
    # branch) so the key's presence is all-or-nothing across a session; K == 0 omits
    # it entirely (old schema byte-identical). Shares the [T,6] wrench's provenance.
    wrench_hires = None
    if FT_HIRES_K > 0:
        if ft_valid:
            wrench_hires = core.resample_wrench_hires(ft[:, 0], ft[:, 1:7], cam_t, FT_HIRES_K)
        else:
            wrench_hires = np.zeros((T, FT_HIRES_K, 6), dtype=np.float32)

    # --- camera RGB: decode JPEG, aspect-preserving resize+crop, stack ---
    out_w, out_h = out_res
    rgb = np.zeros((T, out_h, out_w, 3), dtype=np.uint8)
    for out_i, src_i in enumerate(cam_src_idx):
        jpath = os.path.join(cam_dir, fmt.camera_frame_name(int(src_i)))
        img = jpeg_decode(jpath)  # HxWx3 uint8 RGB
        rgb[out_i] = image_resize(img, out_w, out_h)

    # DECISION-A: episode_timestamps RELATIVE to this episode's first frame (==0).
    rel_t = (cam_t - cam_t[0]).astype(np.float64)

    out = {
        "robot0_eef_pos": eef_pos,
        "robot0_eef_rot_axis_angle": eef_rot,
        "robot0_gripper_width": gw,
        "robot0_demo_start_pose": demo_start,
        "robot0_demo_end_pose": demo_end,
        "robot0_ft_wrench": wrench,
        "episode_timestamps": rel_t,
        "camera0_rgb": rgb,
        "_ft_valid": ft_valid,
    }
    if wrench_hires is not None:
        out["robot0_ft_wrench_hires"] = wrench_hires

    # --- 2pt/3pt secondary keypoints (LEFT-canonical additive schema) ----------
    # robot1_* = RIGHT hand, same TCP extrinsic as robot0 (FK-symmetric wrists).
    # A right gripper stream is normally absent (single physical gripper on the
    # primary hand) -> zero-fill, which the trainer's action_dim_mask drops.
    if tracker_right is not None and tracker_right.shape[0] > 0:
        r6 = _resample_tracker_to_6d(tracker_right, cam_t, T_tracker_tcp)
        out["robot1_eef_pos"] = r6[:, :3].astype(np.float32)
        out["robot1_eef_rot_axis_angle"] = r6[:, 3:].astype(np.float32)
        # robot1_gripper_width is written ONLY when a real right-hand gripper stream
        # exists. This rig has a SINGLE physical gripper (on the primary hand), so the
        # key is normally ABSENT — deliberately NOT zero-filled, because a zero-filled
        # "present" gripper would be masked real downstream and train the right gripper
        # to a constant 0 (a false signal). Absence == "no gripper here" to the loader.
        if gripper_right is not None and gripper_right.shape[0] > 0 and gripper_calib is not None:
            # The RIGHT gripper is a SECOND physical DM4310 with its own homed
            # endpoints, so prefer the ones recorded in ITS stream (cols 3/4) exactly
            # as effective_gripper_calib does for the primary. Falling back to the
            # left hand's encoder range would clamp the right width to one rail.
            right_calib, _src = core.effective_gripper_calib(gripper_right, gripper_calib)
            rw = right_calib.to_width_m(gripper_right[:, 1])
            out["robot1_gripper_width"] = core.resample_gripper_width(
                gripper_right[:, 0], rw, cam_t)
        # robot1_ft_wrench: the RIGHT Bota, anti-alias-resampled onto the master grid
        # exactly like robot0's. Emitted only when that sensor streamed (see docstring).
        if ft_right is not None and ft_right.shape[0] > 0:
            out["robot1_ft_wrench"] = core.antialias_resample_wrench(
                ft_right[:, 0], ft_right[:, 1:7], cam_t)
        # camera1_rgb: the RIGHT-hand D405, paired to the master grid by causal frame
        # selection. Decoded + cropped through the SAME jpeg_decode/image_resize as
        # camera0 so the two views are pixel-comparable.
        if cam_right_index is not None and cam_right_index.shape[0] > 0 and cam_right_dir:
            r_idx, skew = core.resample_frame_index(cam_right_index[:, 0], cam_t, causal=True)
            rgb1 = np.zeros((T, out_h, out_w, 3), dtype=np.uint8)
            for out_i, src_row in enumerate(r_idx):
                # Row -> the ORIGINAL JPEG file index (col 1 of frame_index is the
                # bridge seq, which is NOT the filename; the recorder writes
                # frame_%06d.jpg by ROW, so the row index IS the file index).
                jpath = os.path.join(cam_right_dir, fmt.camera_frame_name(int(src_row)))
                rgb1[out_i] = image_resize(jpeg_decode(jpath), out_w, out_h)
            out["camera1_rgb"] = rgb1
            out["_cam1_skew"] = skew
            # Loud when the pairing is not trustworthy: a right view showing a
            # different instant than the master grid is worse than no right view.
            grid_dt = float(np.median(np.diff(cam_t))) if cam_t.shape[0] > 1 else 0.0
            tol = core.CAMERA_PAIR_SKEW_WARN_FRAC * grid_dt
            if skew["n_pre"]:
                print(f"  WARN {take_tag}: camera1 has no frame at or before the first "
                      f"{skew['n_pre']} grid time(s) — those clamp to its earliest frame")
            if grid_dt > 0 and skew["skew_max_s"] is not None and abs(skew["skew_max_s"]) > tol:
                print(f"  WARN {take_tag}: camera1 pairing skew up to "
                      f"{skew['skew_max_s']*1e3:+.0f} ms (grid dt {grid_dt*1e3:.0f} ms, "
                      f"median {skew['skew_median_s']*1e3:+.0f} ms) — the two views are "
                      f"showing different instants; check the right camera's pacing")

    # head_* = HEAD/CHEST tracker. NO puck->TCP extrinsic (a body reference, not an
    # end-effector) — but the chest puck IS bolted on rotated, so it carries its own
    # ROTATION-ONLY body relabel: world_T_chest = world_T_puck @ T_tracker_head. That
    # leaves head_pos bit-identical (zero translation, enforced by load_head_extrinsic)
    # and only re-expresses head_rot_axis_angle in the chest body frame the operator
    # sees drawn in RAW_FRAME=1 MODE=3pt. No camera, no F/T, no gripper on the chest.
    if tracker_head is not None and tracker_head.shape[0] > 0:
        h6 = _resample_tracker_to_6d(tracker_head, cam_t, T_tracker_head)
        out["head_pos"] = h6[:, :3].astype(np.float32)
        out["head_rot_axis_angle"] = h6[:, 3:].astype(np.float32)

    # {left,right}_foot_* = the two FOOT pucks (5pt, ``--feet recorded``): body references
    # exactly like the chest — rotation-only relabel (identity = raw puck axes), no lever,
    # no camera / F/T / gripper. ABSOLUTE aligned world, like every other keypoint here;
    # a pelvis-frame stance is derivable downstream from these + head_*.
    for _fr, _ftr in (tracker_feet or {}).items():
        if _ftr is not None and _ftr.shape[0] > 0:
            f6 = _resample_tracker_to_6d(_ftr, cam_t, (T_tracker_feet or {}).get(_fr))
            out[f"{_fr}_pos"] = f6[:, :3].astype(np.float32)
            out[f"{_fr}_rot_axis_angle"] = f6[:, 3:].astype(np.float32)

    return out


def _load_take_meta(take_dir: str) -> dict:
    p = os.path.join(take_dir, fmt.TAKE_META_FILE)
    if not os.path.exists(p):
        return {}
    try:
        return json.load(open(p))
    except (ValueError, OSError):
        return {}


def _load_session_meta(session_dir: str) -> dict:
    """Session-level provenance (mode / primary_role / serial_role_map / clock).
    Empty dict for legacy sessions written before session_meta carried them."""
    p = os.path.join(session_dir, fmt.SESSION_META_FILE)
    if not os.path.exists(p):
        return {}
    try:
        return json.load(open(p))
    except (ValueError, OSError):
        return {}


def _load_optional_npy(path: str) -> "np.ndarray | None":
    """Load a .npy if it exists (2-D), else None. Used for the optional 2pt/3pt
    secondary tracker files, which are absent in 1pt/legacy takes."""
    if not os.path.exists(path):
        return None
    try:
        a = np.asarray(np.load(path), dtype=np.float64)
    except (ValueError, OSError):
        return None
    return a if a.ndim == 2 and a.shape[0] > 0 else None


def _sanitize_tracker(tracker: "np.ndarray | None", stale_col: "int | None",
                      label: str = "secondary") -> "np.ndarray | None":
    """Drop stale-repeat rows (col 'stale'==1) + enforce a strictly-increasing
    timestamp, mirroring the primary-tracker hygiene. Returns None if the input is
    None, empty, or ENTIRELY stale — in the last case keeping it would hand the resampler
    a single frozen pose that every validity mask then reports as a real, covered
    measurement (so the caller emits no secondary keypoint at all)."""
    if tracker is None or tracker.shape[0] == 0:
        return None
    if stale_col is not None and tracker.shape[1] > stale_col:
        # NO `if live.any()` guard: an ALL-stale tracker must yield zero rows and become
        # None, not fall through intact. The guard made the docstring false — an all-stale
        # input came back whole, a single frozen pose that the resampler happily
        # interpolated and every validity mask then reported as covered.
        tracker = tracker[tracker[:, stale_col] < 0.5]
        if tracker.shape[0] == 0:
            print(f"  {label}: ALL frames stale -> dropping this keypoint entirely "
                  f"(a frozen pose would resample as a real, covered measurement)")
            return None
    tracker = tracker[core.make_strictly_increasing(tracker[:, 0])]
    # Same lighthouse-glitch deglitch as the primary hand (right/head trackers
    # occlude too), applied after the monotone sanitize, before resampling. Report
    # drops so the secondary path is observable (not silent like before), matching
    # the primary. Secondary keypoints don't gate the take — a robot1/head glitch
    # is less load-bearing than the primary eef label — but the drop is logged.
    if tracker.shape[0] >= 3:
        keep = core.deglitch_tracker_positions(tracker[:, 0], tracker[:, 2:5])
        n_gl = tracker.shape[0] - keep.shape[0]
        if n_gl > 0:
            print(f"  {label} tracker deglitch — dropped {n_gl} glitch sample(s)")
        tracker = tracker[keep]
    return tracker if tracker.shape[0] > 0 else None


def _load_markers(take_dir: str) -> np.ndarray:
    p = os.path.join(take_dir, fmt.MARKER_FILE)
    if not os.path.exists(p):
        return np.zeros((0,), dtype=np.float64)
    m = np.load(p)
    return np.asarray(m, dtype=np.float64).reshape(-1)


# ---------------------------------------------------------------------------
# Image helpers (need cv2 / PIL; isolated so core stays dep-light)
# ---------------------------------------------------------------------------
def _make_image_fns():
    import cv2

    def jpeg_decode(path: str) -> np.ndarray:
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"failed to decode {path}")
        return bgr[:, :, ::-1]  # BGR->RGB

    def image_resize(img: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
        in_h, in_w = img.shape[:2]
        p = core.compute_crop_resize(in_w, in_h, out_w, out_h)
        resized = cv2.resize(img, (p.resize_w, p.resize_h), interpolation=cv2.INTER_AREA)
        return resized[p.crop_y:p.crop_y + out_h, p.crop_x:p.crop_x + out_w]

    return jpeg_decode, image_resize


# ---------------------------------------------------------------------------
# Zarr ReplayBuffer writer (minimal; stock-UMI-compatible layout)
# ---------------------------------------------------------------------------
#: Target DECOMPRESSED bytes per low-dim zarr chunk. Sets the read-amplification /
#: index-size tradeoff (see the chunking note in write_replay_buffer). 24 KiB keeps the
#: historical 4096-frame chunk for the narrow (T,1)..(T,6) arrays — 4096*6*4 = 96 KiB
#: exceeds it, so those land at 1024 frames, still a long sequential run — while the
#: wide (T,16,6) HL windows get 64 frames, measured 7x faster per random single-frame
#: read with no meaningful size cost.
_CHUNK_TARGET_BYTES = 24 * 1024


def write_replay_buffer(out_path: str, episodes: list[dict], out_res, attrs: dict,
                        take_grid_uniform: "bool | None" = None,
                        camera_t_source: "str | None" = None) -> None:
    import zarr
    from numcodecs import Blosc

    # Use the VENDORED UMI codec (copied verbatim from real-stanford UMI, same as
    # HuMI). Its stored config (id 'imagecodecs_jpegxl', keys level/effort/distance/
    # lossless/...) is exactly what stock UMI/diffusion_policy ReplayBuffer expects
    # on read — the modern imagecodecs.numcodecs.Jpegxl writes EXTRA keys
    # (bitspersample, ...) that the UMI reader's JpegXl.__init__ rejects. Using the
    # vendored codec keeps our .zarr.zip loadable by unmodified UMI training code.
    from imagecodecs_numcodecs import JpegXl, register_codecs
    register_codecs()

    out_w, out_h = out_res
    store = zarr.MemoryStore()
    root = zarr.group(store=store)
    data = root.require_group("data")
    meta = root.require_group("meta")

    low_dim_keys = ["robot0_eef_pos", "robot0_eef_rot_axis_angle", "robot0_gripper_width",
                    "robot0_demo_start_pose", "robot0_demo_end_pose", "robot0_ft_wrench",
                    "episode_timestamps"]
    # 2pt/3pt secondary keypoints (LEFT-canonical): robot1_* (right hand) and
    # head_* (head/torso). Additive — appended only when the episodes carry them, so
    # a 1pt session writes exactly the historical robot0-only layout. Every episode
    # MUST agree on these keys (a mid-session tracker dropout that drops right/head
    # in only some episodes would desync the columns); we assert that below.
    _secondary_keys = ["robot1_eef_pos", "robot1_eef_rot_axis_angle", "robot1_gripper_width",
                       # robot1_ft_wrench [T,6]: the RIGHT hand's Bota, resampled onto
                       # the master grid like robot0's. Present only when that sensor
                       # streamed for EVERY episode in the session (the all-or-nothing
                       # check below is what stops a half-covered force column).
                       "robot1_ft_wrench",
                       "head_pos", "head_rot_axis_angle",
                       # robot0_ft_wrench_hires [T,K,6]: additive, all-or-nothing across the
                       # session (emitted for every episode when FT_HIRES_K>0, absent when 0).
                       # The writer below is ndim-generic, so the 3D shape needs no special case.
                       "robot0_ft_wrench_hires",
                       # robot0_pose_glitch [T,1]: 1.0 where the eef_pos/action label is
                       # corrupted by an unbridgeable tracker excursion. Consumed by
                       # UmiLazyDataset(exclude_glitch_windows=True) to drop only the
                       # straddling windows.
                       "robot0_pose_glitch",
                       # {left,right}_foot_* [T,3]: the two FOOT pucks (--feet recorded),
                       # body references like head_*.
                       "left_foot_pos", "left_foot_rot_axis_angle",
                       "right_foot_pos", "right_foot_rot_axis_angle",
                       # pose_glitch_any [T,1]: union of every emitted role's mask.
                       "pose_glitch_any"]
    # <role>_pose_glitch [T,1] per emitted tracker role (left/right/head/left_foot/
    # right_foot): discovered like the hl_* keys so a new role needs no edit here.
    _secondary_keys += sorted({k for ep in episodes for k in ep
                               if k.endswith("_pose_glitch") and k != "robot0_pose_glitch"})
    # HL-v2 fixed-dt window arrays (additive, all-or-nothing across the session).
    # Discovered rather than hard-listed so adding a role/quantity in
    # _build_hl_windows needs no edit here. Take the UNION over every episode (not
    # just episode 0) so a key that appears late still reaches the all-or-nothing
    # check below and raises, instead of being silently dropped.
    _secondary_keys += sorted({k for ep in episodes for k in ep if k.startswith("hl_")})
    for key in _secondary_keys:
        present = [key in ep for ep in episodes]
        if all(present):
            low_dim_keys.append(key)
        elif any(present):
            raise ValueError(
                f"secondary key {key!r} present in only SOME episodes "
                f"({sum(present)}/{len(episodes)}); a partial secondary keypoint would "
                f"desync the zarr columns. Re-record or drop the incomplete take.")
    blosc = Blosc(cname="lz4", clevel=5, shuffle=Blosc.NOSHUFFLE)

    episode_ends = []
    cursor = 0
    # Pre-create datasets sized to total T (computed up front).
    total_T = sum(ep["camera0_rgb"].shape[0] for ep in episodes)
    for key in low_dim_keys:
        sample = episodes[0][key]
        shape = (total_T,) + sample.shape[1:]
        # Chunk length is a READ-AMPLIFICATION knob, and the wide hl_* arrays make it
        # load-bearing. A training sample reads ONE frame, and zarr2 has no chunk cache,
        # so a chunk is decompressed in full on every __getitem__ — even a repeat read of
        # the same frame. For a (T,16,6) f32 array a 4096-frame chunk is 1.5 MiB
        # decompressed to serve 384 B (4096x), measured at 2.17 ms/sample vs 0.31 ms at
        # 64 frames — enough to nearly double per-sample I/O against the 2.34 ms JpegXl
        # decode. So cap the chunk by BYTES rather than by frame count: narrow arrays
        # (T,1)/(T,3)/(T,6) keep long chunks (fast sequential scans, small index), while
        # wide per-frame windows get short ones. Costs ~0.25% of the zip size.
        row_bytes = int(np.prod(sample.shape[1:], dtype=np.int64)) * sample.dtype.itemsize
        chunk_len = max(1, min(int(total_T), _CHUNK_TARGET_BYTES // max(1, row_bytes)))
        chunks = (chunk_len,) + sample.shape[1:]
        data.create_dataset(key, shape=shape, chunks=chunks, dtype=sample.dtype,
                            compressor=blosc, overwrite=True)
    # camera0_rgb is the LEFT hand and the MASTER grid — always present. camera1_rgb is
    # the RIGHT hand and is additive: created only when EVERY episode carries it, under
    # the same all-or-nothing rule as the low-dim secondary keys (a session with the
    # right camera on for only some takes would desync the image column against the
    # pose columns, and unlike a low-dim array there is no cheap way to spot that later).
    cam_keys = ["camera0_rgb"]
    _cam1_present = [("camera1_rgb" in ep) for ep in episodes]
    if all(_cam1_present):
        cam_keys.append("camera1_rgb")
    elif any(_cam1_present):
        raise ValueError(
            f"camera1_rgb present in only SOME episodes ({sum(_cam1_present)}/"
            f"{len(episodes)}); a partial second camera would desync the zarr columns. "
            f"Re-record the affected takes, convert with --hands left, or exclude them.")
    cams = {
        key: data.create_dataset(
            key, shape=(total_T, out_h, out_w, 3), chunks=(1, out_h, out_w, 3),
            dtype=np.uint8, compressor=JpegXl(level=99, numthreads=1), overwrite=True)
        for key in cam_keys
    }

    for ep in episodes:
        T = ep["camera0_rgb"].shape[0]
        for key in low_dim_keys:
            data[key][cursor:cursor + T] = ep[key]
        for key in cam_keys:
            cams[key][cursor:cursor + T] = ep[key]
        cursor += T
        episode_ends.append(cursor)

    meta.create_dataset("episode_ends", data=np.asarray(episode_ends, dtype=np.int64),
                        chunks=(len(episode_ends),), dtype=np.int64, compressor=None,
                        overwrite=True)

    # ===== camera-timeline uniformity, recorded in attrs =====
    # See umi_convert_core.camera_grid_verdict for WHY the test is a dt spread +
    # outlier-fraction pair and NOT round(dt/median). Stamping the verdict here
    # means a downstream loader can gate on it instead of every consumer
    # re-deriving it (nothing did, for 97 takes).
    dts = [np.diff(ep["episode_timestamps"]) for ep in episodes
           if ep["episode_timestamps"].shape[0] > 1]
    all_dt = np.concatenate(dts) if dts else np.zeros(0)
    verdict = core.camera_grid_verdict(
        all_dt, take_grid_uniform=take_grid_uniform, camera_t_source=camera_t_source)
    attrs.update(verdict)
    if verdict.get("camera_dt_holes_n"):
        print(f"[convert] ⚠ {verdict['camera_dt_holes_n']} camera-timeline HOLE(s) "
              f"(single steps >1.5x the {verdict['camera_dt_median'] * 1000:.1f} ms "
              f"median; worst {verdict['camera_dt_max_over_median']:.1f}x). Dropped "
              f"frames or a camera-bridge reconnect. Every other modality is linearly "
              f"interpolated across a hole, so the pose/wrench labels there are "
              f"straight lines. Reported, not fatal -- see attrs camera_dt_holes_n.")
    if verdict.get("camera_dt_uniform") is False:
        print(f"[convert] ⚠ WARNING camera timeline NON-UNIFORM: "
              f"{verdict['camera_hz']:.2f} Hz mean, dt median "
              f"{verdict['camera_dt_median'] * 1000:.1f} ms, p5/p95 "
              f"{verdict['camera_dt_p5_p95'][0] * 1000:.1f}/"
              f"{verdict['camera_dt_p5_p95'][1] * 1000:.1f} ms "
              f"(spread {verdict['camera_dt_spread'] * 100:.1f}%, need <=5%; "
              f"outliers {verdict['camera_dt_outlier_frac'] * 100:.2f}%, need <=0.2%)"
              + (", take_meta says camera_grid_uniform=false"
                 if take_grid_uniform is False else "") + ".\n"
              f"           The camera grid is the MASTER TIMELINE and the training "
              f"loaders index FRAMES with no dt input, so this becomes multiplicative "
              f"noise on every velocity / dF-dt label. Root cause is usually the "
              f"camera bridge pacing -- see bridges/camera/realsense_bridge.py.\n"
              f"           attrs['camera_dt_uniform']=False is stamped on the zarr so "
              f"downstream can gate on it.")

    for k, v in attrs.items():
        root.attrs[k] = v

    # Save to a single .zip store (stock UMI loads via ReplayBuffer.copy_from_store).
    if os.path.exists(out_path):
        os.remove(out_path)
    with zarr.ZipStore(out_path, mode="w") as zip_store:
        zarr.copy_store(store, zip_store)
    print(f"[convert] wrote {out_path}  (episodes={len(episodes)}, T_total={total_T})")


def _print_glitch_summary(glitch_agg: dict, frames_total: int, n_takes: int,
                          dropped: list, excluded: list, n_episodes: int) -> None:
    """End-of-run tracker-glitch statistics: what was dropped, what was masked, per role."""
    print("[convert] ── tracker glitch statistics (kept episodes) ──────────────────────")
    print(f"[convert]   takes: {n_takes} listed, {n_takes - len(dropped)} ingested, "
          f"{len(dropped)} DROPPED by the converter, {len(excluded)} --exclude'd; "
          f"episodes written: {n_episodes}; camera frames: {frames_total}")
    for n, r in dropped:
        print(f"[convert]     dropped {n}: {r}")
    if glitch_agg:
        hdr = (f"[convert]   {'role':10s} {'samples':>8s} {'pos-drop':>8s} {'bridged':>7s} "
               f"{'unbridg':>7s} {'rot-exc':>7s} {'rot-drop':>8s} {'frames-flag':>11s} "
               f"{'hl-steps-masked':>15s} {'eps-flag':>8s}")
        print(hdr)
        for r, d in glitch_agg.items():
            ff = d["frames_flagged"]
            pct = f"{100.0 * ff / max(frames_total, 1):.1f}%"
            print(f"[convert]   {r:10s} {d['samples']:8d} {d['pos_samples_dropped']:8d} "
                  f"{d['pos_bridged_excursions']:7d} {d['pos_unbridged_excursions']:7d} "
                  f"{d['rot_excursions']:7d} {d['rot_samples_dropped']:8d} "
                  f"{ff:6d}({pct:>5s}) {d['hl_steps_masked']:15d} {d['episodes_flagged']:8d}")
        print("[convert]   (frames-flag = anchors whose window a sampler drops; hl-steps-masked = "
              "action steps zero-weighted; nothing else is lost — no episode is dropped for a "
              "secondary role's glitch)")


# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    # Declared global up front: the CLI override below is written back into the
    # module constant that _build_one_episode reads. (A module constant — not a
    # build_episodes_from_take kwarg — because convert_to_lerobot.py calls
    # build_episodes_from_take with a fixed signature.)
    global FT_HIRES_K
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("session_dir")
    p.add_argument("-o", "--output", required=True, help="output .zarr.zip path")
    p.add_argument("--out-res", type=int, nargs=2, default=(224, 224), metavar=("W", "H"))
    p.add_argument("--ft-hires-k", type=int, default=FT_HIRES_K,
                   help=f"K raw F/T samples per camera step for the causal hi-rate "
                        f"robot0_ft_wrench_hires [T,K,6] export (default {FT_HIRES_K}; "
                        f"0 disables the key, keeping the pre-hi-rate schema).")
    p.add_argument("--hands", choices=sorted(HANDS_MODES), default=None,
                   help="Which tracker roles to emit: left | right | both | both_head | "
                        "both_head_dual. Default (omitted) infers them from which "
                        "tracker_*.npy files exist, the historical behavior. Setting it "
                        "makes the mode a CONTRACT: a take missing a required tracker is "
                        "an error instead of a silently smaller dataset. both_head_dual "
                        "needs the same three pucks as both_head but ALSO folds the RIGHT "
                        "hand's Bota + DM4310 into the HL windows (hl_right_action_force / "
                        "_gripper / hl_right_force_history) — the two-hand force+gripper "
                        "contract that FLoRe layout mode 'both_hands_head_dual' reads.")
    p.add_argument("--hl-windows", dest="hl_windows", action="store_true", default=None,
                   help=f"Emit the HL fixed-dt policy windows (hl_* keys: "
                        f"{core.FORCE_HIST_LEN}-bin causal force history + "
                        f"{core.ACTION_HORIZON}-step pose/force/gripper targets on an "
                        f"exact {core.ACTION_HZ:g} Hz grid). Implied by --hands.")
    p.add_argument("--no-hl-windows", dest="hl_windows", action="store_false",
                   help="Suppress the hl_* keys even when --hands is given.")
    # --- HL grid overrides (contract v3 defaults; the v2 grid stays reachable) ------
    # These three re-pin the grid the hl_* windows are built on. They exist so the
    # PREVIOUS contract is still producible without editing code --
    #     --hl-force-hist-len 8 --hl-action-horizon 16 --hl-action-hz 16
    # reproduces the v2 "1 s history -> 1 s future @ 16 Hz" grid exactly. The values
    # are stamped into the zarr attrs, and the FLoRe loader gates every one of them
    # against its own layout constants, so a mismatched pair is a load-time error
    # rather than a silently mis-timed label.
    p.add_argument("--hl-force-hist-len", type=int, default=core.FORCE_HIST_LEN,
                   help=f"causal force-history INPUT bins ending at t0 (default "
                        f"{core.FORCE_HIST_LEN} = t0 only, no look-back; 8 = the v2 "
                        f"1 s history). 0 emits no force input at all.")
    p.add_argument("--hl-action-horizon", type=int, default=core.ACTION_HORIZON,
                   help=f"predicted action steps (default {core.ACTION_HORIZON}).")
    p.add_argument("--hl-action-hz", type=float, default=core.ACTION_HZ,
                   help=f"action grid rate in Hz (default {core.ACTION_HZ:g}, matching "
                        f"CAMERA_PUBLISH_HZ=30 so the obs and action grids coincide). "
                        f"The span is DERIVED as horizon/hz.")
    p.add_argument("--hl-force-hist-span-s", type=float, default=None,
                   help="total force-history look-back in seconds. Default (omitted) "
                        "derives it as force_hist_len/action_hz, i.e. one input bin per "
                        "action step. Pass 1.0 together with --hl-force-hist-len 8 "
                        "--hl-action-horizon 16 --hl-action-hz 16 to reproduce the v2 grid.")
    p.add_argument("--exclude", nargs="*", default=(), metavar="TAKE",
                   help="Take names to skip (e.g. take_005 take_025). Accepts the "
                        "basename with or without the take_ prefix / zero-padding "
                        "(take_005 == 005 == 5). Excluded takes are simply NOT "
                        "ingested — the raw recording is never modified. Names that "
                        "match no committed take are an error (typo guard).")
    p.add_argument("--feet", choices=("ignore", "recorded"), default="ignore",
                   help="5pt sessions only. 'recorded' ALSO emits the two FOOT pucks as "
                        "body-reference keypoints: {left,right}_foot_pos/_rot_axis_angle on "
                        "the camera grid and hl_{left,right}_foot_action_pose/_valid windows "
                        "(rotation-only relabel from calibration/foot_extrinsic.json, "
                        "identity = raw puck axes). Requires --hands. Default 'ignore' "
                        "leaves every existing conversion byte-identical.")
    p.add_argument("--no-rot-deglitch", action="store_true",
                   help="Disable dropping there-and-back ORIENTATION excursions "
                        "(lighthouse rotation glitches) from every tracker stream before "
                        "resampling. Default ON: the rows are dropped like a bridged position "
                        "glitch and the spans are masked in hl_*_action_valid + "
                        "<role>_pose_glitch, so a one-sample rotation spike costs a window, "
                        "not the take.")
    args = p.parse_args(argv if argv is not None else sys.argv[1:])
    if args.feet == "recorded" and args.hands is None:
        p.error("--feet recorded requires --hands (the feet ride along with an explicit "
                "hand/chest contract; attrs.hands_mode must not be inferred with feet present)")
    DROPPED_TAKES.clear()

    FT_HIRES_K = int(args.ft_hires_k)
    # Re-pin the HL grid ONCE, before any window is built. core.set_grid swaps the single
    # `core.GRID` object that every resampler/validator reads at CALL time — deliberately
    # not a set of module constants, because a function default is captured at DEF time
    # and would ignore this override entirely (see the GRID note in umi_convert_core).
    try:
        grid = core.set_grid(force_hist_len=args.hl_force_hist_len,
                             action_horizon=args.hl_action_horizon,
                             action_hz=args.hl_action_hz,
                             force_hist_span_s=args.hl_force_hist_span_s)
    except ValueError as e:
        p.error(str(e))
    # --hands implies the HL windows (asking for a specific hand set is asking for the
    # HL contract); --no-hl-windows still overrides. Without --hands the default is
    # OFF, so an existing conversion command produces the historical schema.
    hl_windows = args.hl_windows if args.hl_windows is not None else (args.hands is not None)
    if hl_windows:
        print(f"[convert] HL grid: {grid.force_hist_len} force-history bin(s) over "
              f"{grid.hist_span_s:.4f}s -> {grid.action_horizon} steps over "
              f"{grid.action_span_s:.4f}s @ {grid.action_hz:g} Hz "
              f"(action dt {grid.action_dt * 1000:.3f} ms)", flush=True)

    # DECISION-B reminder: raw tracker quaternions are WXYZ; we convert to xyzw here.
    print(f"[convert] raw tracker quaternions are {fmt.QUATERNION_CONVENTION.upper()}; "
          f"converting to xyzw at umi_convert_core.quat_wxyz_to_xyzw", flush=True)

    session_dir = os.path.abspath(args.session_dir)
    # Only ingest COMMITTED takes: a dir with take_meta.json (the recorder's
    # atomic-rename commit marker). This skips any leftover take_*.tmp from a
    # crash/SIGTERM mid-flush, so a partial write never aborts the run (M2).
    take_dirs = sorted(
        d for d in (os.path.join(session_dir, x) for x in os.listdir(session_dir))
        if os.path.isdir(d) and os.path.basename(d).startswith("take_")
        and not os.path.basename(d).endswith(".tmp")
        and os.path.exists(os.path.join(d, fmt.TAKE_META_FILE)))
    if not take_dirs:
        print(f"[convert] no committed take_* dirs (with {fmt.TAKE_META_FILE}) "
              f"in {session_dir}", file=sys.stderr)
        return 1

    # Exclude bad takes by name WITHOUT touching the raw recording: they are dropped
    # from the ingest list here, so nothing on disk changes. Normalize both the CLI
    # names and the on-disk basenames to a zero-padded canonical form so take_5, 005,
    # and take_005 all match take_005. A name that matches nothing is a typo — fail
    # loudly rather than silently converting the full set (the classic QC footgun).
    def _canon_take(name: str) -> str:
        # rstrip separators first so a tab-completed dir path ('.../take_012/') doesn't
        # basename to '' and silently become 'take_'. isdigit()+int() only zero-pads a
        # plain number; anything else (incl. odd inputs) passes through as 'take_<base>'
        # and simply won't match a real take -> the typo guard reports it, never crashes.
        base = os.path.basename(str(name).rstrip("/\\")).strip().lower()
        if base.startswith("take_"):
            base = base[len("take_"):]
        return f"take_{int(base):03d}" if (base.isdigit() and len(base) <= 9) else f"take_{base}"

    if args.exclude:
        want_drop = {_canon_take(x) for x in args.exclude}
        have = {os.path.basename(d): d for d in take_dirs}
        have_canon = {_canon_take(b): b for b in have}
        unknown = sorted(x for x in want_drop if x not in have_canon)
        if unknown:
            print(f"[convert] --exclude names match no committed take: "
                  f"{', '.join(unknown)} (have {len(have)} takes "
                  f"{min(have)}..{max(have)})", file=sys.stderr)
            return 1
        dropped = sorted(have_canon[x] for x in want_drop)
        take_dirs = [d for d in take_dirs if os.path.basename(d) not in dropped]
        print(f"[convert] excluding {len(dropped)} take(s) (raw untouched): "
              f"{', '.join(dropped)}", flush=True)
        if not take_dirs:
            print("[convert] every take was excluded — nothing to convert",
                  file=sys.stderr)
            return 1

    # Session-level tracking provenance (LEFT-canonical): mode + primary_role +
    # serial->role map, recorded by the recorder in session_meta. robot0_* is built
    # from tracker.npy, which is the PRIMARY (left) hand under LEFT-canonical; older
    # sessions have no 'mode' key and used the historical right-primary layout.
    session_meta = _load_session_meta(session_dir)
    session_mode = session_meta.get("mode")            # '1pt'|'2pt'|'3pt' or None (legacy)
    session_primary_role = session_meta.get("primary_role", "right" if session_mode is None else "left")

    gripper_calib = load_gripper_calib(session_dir)
    if gripper_calib is None:
        print("[convert] WARNING: no gripper_range.json — gripper_width will be ZERO. "
              "Run the gripper calibration before training.", file=sys.stderr)
    T_tracker_tcp, tcp_calibrated = load_tcp_extrinsic(session_dir)
    if not tcp_calibrated:
        print("[convert] WARNING: TCP extrinsic is identity/uncalibrated — poses are "
              "RAW TRACKER, not TCP. Stamping UNCALIBRATED_DO_NOT_TRAIN.", file=sys.stderr)
    # CHEST relabel: only reaches the zarr on a 3pt session (no head tracker -> no head_*
    # keys), but resolved unconditionally so the value lands in attrs either way. Loud in
    # BOTH directions, because "which chest convention is this dataset in" is invisible in
    # every downstream plot: head_pos is identical under both, only the rotation moves.
    T_tracker_head, head_calibrated = load_head_extrinsic(session_dir)
    if head_calibrated:
        print("[convert] chest (head_*) body relabel ACTIVE — head_rot_axis_angle is the "
              "chest BODY frame, not the raw chest-puck axes (head_pos unchanged)")
    else:
        print("[convert] WARNING: no head_extrinsic.json — head_rot_axis_angle keeps the "
              "RAW chest-puck axes (the pre-2026-08-26 convention). Fine for 1pt/2pt; on a "
              "3pt session it will NOT match a rig running the relabel.", file=sys.stderr)

    T_tracker_feet, feet_calibrated = None, False
    if args.feet == "recorded":
        T_tracker_feet, feet_calibrated = load_foot_extrinsic(session_dir)
        if feet_calibrated:
            print("[convert] FEET body relabel ACTIVE (foot_extrinsic.json)")
        else:
            print("[convert] NOTE: --feet recorded with no foot_extrinsic.json — "
                  "{left,right}_foot_rot_axis_angle keep the RAW foot-puck axes.")

    jpeg_decode, image_resize = _make_image_fns()

    episodes = []
    any_ft_invalid = False
    ft_sources: set[str] = set()
    ft_algos: set[str] = set()          # which compensation engine(s) across comp episodes
    any_unverified_sign = False         # any compensated episode with an unverified sign
    take_grid_uniform = None            # False if ANY take reported a non-uniform camera grid
    # Provenance of the camera timestamps. The bridge switched from a post-JPEG-encode
    # `time.time()` to the device global-time clock taken BEFORE encode, which moved the
    # camera grid ~10.8 ms earlier relative to every other stream (and collapsed its dt
    # jitter). Sessions from before and after the switch are therefore NOT time-consistent
    # to ~10.8 ms; recorded in code so re-converting can never lose the warning.
    camera_t_source = "device_global_time_pre_encode" if any(
        _load_take_meta(d).get("camera_capture_gaps") for d in take_dirs
    ) else "host_wall_post_encode"
    tracker_dropped_total = 0           # tracker glitch samples bridged across the session
    tracker_unbridged_total = 0         # residual unbridged excursions in KEPT episodes
    tracker_residual_jump_max = 0.0     # worst residual eef jump (m) in a KEPT episode
    glitch_agg: dict = {}               # per-role glitch statistics over KEPT episodes
    frames_total = 0
    excluded_takes = [] if not args.exclude else list(dropped)
    # RIGHT-hand camera pairing quality (see core.resample_frame_index). Only meaningful
    # when camera1_rgb is emitted; left at their neutral values otherwise.
    cam1_skew_max = 0.0                 # worst SIGNED (t_right - t_grid) across the session
    cam1_pre_total = 0                  # grid frames with no causal right frame (clamped)
    cam1_reused_max = 0                 # most grid frames served by ONE right frame
    for take_dir in take_dirs:
        for arrs in build_episodes_from_take(take_dir, gripper_calib, T_tracker_tcp,
                                             tuple(args.out_res), jpeg_decode, image_resize,
                                             hands=args.hands, hl_windows=hl_windows,
                                             T_tracker_head=T_tracker_head,
                                             feet=args.feet, T_tracker_feet=T_tracker_feet,
                                             rot_deglitch=not args.no_rot_deglitch):
            frames_total += int(arrs["camera0_rgb"].shape[0])
            _gm = arrs.pop("_glitch_masked_steps", {}) or {}
            for _r, _d in (arrs.pop("_glitch_stats", {}) or {}).items():
                acc = glitch_agg.setdefault(_r, {k: 0 for k in _d} | {"hl_steps_masked": 0,
                                                                     "episodes_flagged": 0})
                for k, v in _d.items():
                    acc[k] = max(acc[k], v) if k == "residual_jump_m_max" else acc[k] + v
                acc["hl_steps_masked"] += int(_gm.get(_r, 0))
                acc["episodes_flagged"] += int(_d.get("frames_flagged", 0) > 0)
            tracker_dropped_total += int(arrs.pop("_tracker_dropped", 0))
            tracker_unbridged_total += int(arrs.pop("_tracker_unbridged", 0))
            tracker_residual_jump_max = max(
                tracker_residual_jump_max, float(arrs.pop("_tracker_residual_jump_m", 0.0)))
            if not arrs.pop("_ft_valid"):
                any_ft_invalid = True
            src = arrs.pop("_ft_source", "raw")
            ft_sources.add(src)
            algo = arrs.pop("_ft_algo", None)
            sign_ok = arrs.pop("_ft_sign_verified", True)
            if arrs.pop("_camera_grid_uniform", None) is False:
                take_grid_uniform = False
            # RIGHT-hand camera pairing skew: aggregate the worst |skew| and any grid
            # frames with no causal right frame, so the zarr records how well the two
            # views line up instead of leaving it to be re-derived (or never checked).
            _sk = arrs.pop("_cam1_skew", None)
            if _sk is not None:
                cam1_pre_total += int(_sk.get("n_pre") or 0)
                if _sk.get("skew_max_s") is not None:
                    if abs(float(_sk["skew_max_s"])) > abs(cam1_skew_max):
                        cam1_skew_max = float(_sk["skew_max_s"])
                cam1_reused_max = max(cam1_reused_max, int(_sk.get("reused_max") or 0))
            if src == "compensated":
                # Only compensated episodes carry a meaningful engine identity; a
                # missing algo (older recording) defaults to bota (the pre-RNEA engine).
                ft_algos.add(algo or "bota")
                if sign_ok is False:
                    any_unverified_sign = True
            episodes.append(arrs)

    if not episodes:
        print("[convert] no usable episodes", file=sys.stderr)
        return 1

    # robot0_ft_wrench provenance, aggregated across the session's episodes. A
    # single session dir can legitimately MIX compensated + raw takes (relaunch
    # with --no-compensate, same --session-name), so collapse to a tri-state
    # rather than a bare bool — a trainer must not read a mixed session as all-one.
    if ft_sources == {"compensated"}:
        ft_provenance = "compensated"
    elif ft_sources == {"raw"} or not ft_sources:
        ft_provenance = "raw"
    else:
        ft_provenance = "mixed"

    # Which compensation ENGINE the compensated wrench came from, aggregated the same
    # way: a session can mix bota + rnea episodes (dropdown switched, or a per-take
    # 'mixed' flag), so collapse to a tri-state (bota|rnea|mixed) rather than assume
    # one — bota gravity-only and RNEA full-dynamic are DIFFERENT physics a trainer
    # must be able to tell apart. None when there is no compensated episode at all.
    if not ft_algos:
        ft_comp_algo = None
    elif ft_algos == {"bota"}:
        ft_comp_algo = "bota"
    elif ft_algos == {"rnea"}:
        ft_comp_algo = "rnea"
    else:
        ft_comp_algo = "mixed"

    # n_robots / n_cameras are DERIVED from what the episodes actually carry, not
    # hardcoded: they are the first thing a downstream loader reads to decide how many
    # keypoints/views exist, and a stale 1 next to a real robot1_*/camera1_rgb column is
    # exactly the kind of mismatch that gets discovered during training instead of here.
    _has_robot1 = all("robot1_eef_pos" in ep for ep in episodes)
    _has_cam1 = all("camera1_rgb" in ep for ep in episodes)
    _has_head = all("head_pos" in ep for ep in episodes)
    attrs = {
        "schema": "umi_replay_buffer_v1",
        "n_robots": 2 if _has_robot1 else 1,
        "n_cameras": 2 if _has_cam1 else 1,
        # Which per-hand modalities this dataset actually contains. The CHEST keypoint
        # (head_*) is a body reference with no camera / F-T / gripper by construction.
        "hands": {
            "left": {"pose": True, "camera": True, "ft": True, "gripper": True},
            "right": {"pose": _has_robot1,
                      "camera": _has_cam1,
                      "ft": all("robot1_ft_wrench" in ep for ep in episodes),
                      "gripper": all("robot1_gripper_width" in ep for ep in episodes)},
            "chest": {"pose": _has_head, "camera": False, "ft": False, "gripper": False},
        },
        # RIGHT-camera pairing quality: camera1_rgb is selected (causal hold) onto the
        # LEFT camera's grid, so these say how stale that view can be. skew is SIGNED
        # (negative = the right frame predates the grid time, which a causal hold
        # guarantees); pre_frames counts grid times with no causal right frame at all.
        **({"camera1_pair_skew_max_s": round(cam1_skew_max, 5),
            "camera1_pair_pre_frames": cam1_pre_total,
            "camera1_pair_reused_max": cam1_reused_max,
            "camera1_pair_rule": "causal_hold_on_camera0_grid"} if _has_cam1 else {}),
        "has_ft_wrench": True,
        # 'compensated' = gravity/inertia removed; 'raw' = native sensor (tool
        # gravity included); 'mixed' = per-episode varies (see take_meta per take).
        "ft_compensated": ft_provenance,
        # WHICH engine compensated (bota gravity-only | rnea full-dynamic | mixed |
        # null when no compensated episode). Distinct physics — kept for the trainer.
        "ft_comp_algo": ft_comp_algo,
        "tcp_calibrated": tcp_calibrated,
        "gripper_calibrated": gripper_calib is not None,
        "out_res": list(args.out_res),
        "image_transform": "aspect_preserving_resize_center_crop",
        "raw_quaternion_convention": fmt.QUATERNION_CONVENTION,  # provenance: 'wxyz'
        # Tracker-deglitch provenance: how many lighthouse position glitches were
        # BRIDGED (dropped + linearly re-interpolated) across the kept episodes, and
        # the worst residual eef jump that survived. Unbridgeable takes are dropped
        # upstream, so tracker_unbridged_kept should be 0 for a shipped dataset;
        # a nonzero value means a residual glitch reached the eef_pos label.
        "tracker_deglitch_samples_bridged": tracker_dropped_total,
        "tracker_unbridged_kept": tracker_unbridged_total,
        "tracker_residual_jump_m_max": round(tracker_residual_jump_max, 4),
        "episode_timestamps_relative": True,  # DECISION-A: each episode starts at t=0
        # LEFT-canonical tracking provenance. robot0_* comes from tracker.npy =
        # the PRIMARY hand: 'left' for sessions recorded with a 'mode' key, 'right'
        # for legacy sessions. tracking_mode is null for legacy recordings.
        "tracking_mode": session_mode,
        "robot0_primary_role": session_primary_role,
        "serial_role_map": session_meta.get("serial_role_map", {}),
        # ===== HL-v2 policy-window provenance =====
        # hands_mode is None for a legacy/inferred conversion; a downstream loader
        # should ASSERT on it rather than infer the keypoint set from key presence.
        # The role set ACTUALLY emitted, not the flag. `--hl-windows` without `--hands` is
        # a supported invocation that falls through to every tracker on disk, and stamping
        # the bare flag left this null — so a loader's mode gate (which must tolerate a
        # null from pre-provenance files) accepted EVERY mode against a 3-role file and
        # silently discarded the other roles' columns. Derived from the emitted keys so the
        # two can never disagree.
        "hands_mode": args.hands if args.hands is not None else _hands_flag_for_roles(
            {k.split("_")[1] for ep in episodes for k in ep if k.startswith("hl_")},
            # The FORCE / GRIPPER role sets are derived from the emitted keys for the same
            # reason the tracker roles are: `both_head` and `both_head_dual` need the
            # identical three pucks, so the tracker set alone cannot tell them apart, and
            # stamping the wrong one lets FLoRe's mode gate accept a file whose sensor
            # columns it will then mis-size.
            {k.split("_")[1] for ep in episodes for k in ep
             if k.startswith("hl_") and k.endswith("_force_history")},
            {k.split("_")[1] for ep in episodes for k in ep
             if k.startswith("hl_") and k.endswith("_action_gripper")}),
        "hl_windows": bool(hl_windows),
        "hl_force_hist_len": core.FORCE_HIST_LEN if hl_windows else None,
        "hl_force_hist_span_s": core.FORCE_HIST_SPAN_S if hl_windows else None,
        "hl_action_horizon": core.ACTION_HORIZON if hl_windows else None,
        "hl_action_span_s": core.ACTION_SPAN_S if hl_windows else None,
        "hl_action_hz": (core.ACTION_HORIZON / core.ACTION_SPAN_S) if hl_windows else None,
        # Whether the action chunk INCLUDES the observation instant t0 as step 0
        # ("t0", DECISION-T0, UMI-on-Air parity) or starts one dt ahead ("t0+dt", the
        # pre-v2 grid that excluded the identity step). The shift is invisible in the
        # values, so a loader MUST read this rather than infer it — an "excluded" model
        # served an "inclusive" chunk (or vice-versa) is off by one dt on every step.
        "hl_action_first_step": ("t0" if hl_windows else None),
        # Poses in the hl_* target arrays are ABSOLUTE aligned-world, like
        # robot0_eef_pos — the anchor-relative transform is the trainer's choice.
        "hl_pose_frame": "absolute_aligned_world" if hl_windows else None,
        # The wrench is NEVER rotated anywhere in this pipeline: it stays in the Bota
        # SENSOR frame end to end. Recorded explicitly because a force-conditioned
        # policy's frame assumption is otherwise invisible.
        "hl_force_frame": "bota_sensor" if hl_windows else None,
        "hl_tare_settle_s": core.TARE_SETTLE_S if hl_windows else None,
        # ===== frame reconstructibility =====
        # The world alignment R_world baked into every recorded pose, and the
        # tracker->TCP extrinsic composed on top. Without these, "absolute aligned
        # world" is a label, not a reproducible frame: two sessions taken under
        # different WORLD_RPY were byte-indistinguishable, and the extrinsic lived only
        # in a repo file that can later move (the LeRobot exporter already stored it —
        # the zarr path, which is what the policy trains on, had dropped it).
        # null world_align = a pre-provenance session, which is NOT a claim of identity.
        "world_align": session_meta.get("world_align"),
        "T_tracker_tcp": np.asarray(T_tracker_tcp, dtype=np.float64).tolist(),
        # The CHEST puck's rotation-only body relabel that produced head_rot_axis_angle
        # (identity == the raw chest-puck axes). Recorded even on 1pt/2pt, where there is
        # no head_* key, so "which convention" is answerable from the dataset alone —
        # head_pos is identical under both, so nothing else can tell them apart.
        "T_tracker_head": np.asarray(T_tracker_head, dtype=np.float64).tolist(),
        # The gripper_width -> [0,1] reference the trainer divides by, so a loader can
        # assert instead of assuming the value hardcoded on the training side.
        "gripper_width_open_m": (gripper_calib.width_open_m
                                 if gripper_calib is not None else None),
        # ===== FEET (5pt, --feet recorded) =====
        "feet_recorded": args.feet == "recorded",
        "foot_extrinsic_calibrated": bool(feet_calibrated),
        **({f"T_tracker_{r}": np.asarray(T, dtype=np.float64).tolist()
            for r, T in (T_tracker_feet or {}).items()}),
        # ===== glitch handling provenance =====
        # Rotation excursions dropped from every tracker stream (see --no-rot-deglitch),
        # per-role masks `<role>_pose_glitch` + `pose_glitch_any` written, and the
        # per-role statistics over the KEPT episodes (samples dropped, excursions, frames
        # flagged, hl_* steps masked). Takes the converter refused, with the reason.
        "rot_deglitch": not args.no_rot_deglitch,
        "pose_glitch_roles": sorted(glitch_agg),
        "glitch_stats": glitch_agg,
        "frames_total": frames_total,
        "takes_ingested": len(take_dirs) - len(DROPPED_TAKES),
        "takes_dropped": [{"take": n, "reason": r} for n, r in DROPPED_TAKES],
        "takes_excluded": list(excluded_takes),
    }
    if ft_provenance == "mixed":
        print("[convert] NOTE: F/T provenance is MIXED across episodes "
              "(some compensated, some raw) — see per-take take_meta ft_compensated.")
    if ft_comp_algo == "mixed":
        print("[convert] WARNING: compensated F/T MIXES algorithms across episodes "
              "(bota gravity-only + rnea full-dynamic) — these are different physics; "
              "see per-take take_meta ft_comp_algo before training.")
        attrs["ft_algo_note"] = "compensated wrench mixes bota + rnea across episodes (see take_meta ft_comp_algo)"
    if any_unverified_sign:
        print("[convert] WARNING: at least one COMPENSATED episode used an RNEA config "
              "with an UNVERIFIED wrench_sign — the compensated wrench may be sign-INVERTED. "
              "Validate with a static tilt or prefer the raw ft_wrench (see take_meta ft_sign_verified).")
        attrs["ft_sign_unverified"] = True
    if not tcp_calibrated:
        attrs["WARNING"] = "UNCALIBRATED_DO_NOT_TRAIN: TCP extrinsic is identity"
    if any_ft_invalid:
        attrs["ft_note"] = "some episodes had no F/T; zero-filled (see take_meta ft_valid)"

    _print_glitch_summary(glitch_agg, frames_total, len(take_dirs), DROPPED_TAKES,
                          excluded_takes, len(episodes))
    write_replay_buffer(args.output, episodes, tuple(args.out_res), attrs,
                        take_grid_uniform=take_grid_uniform,
                        camera_t_source=camera_t_source)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
