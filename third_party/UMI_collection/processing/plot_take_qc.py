#!/usr/bin/env python
"""Per-take RAW-data QC sheets for a recorded session (one PNG per take folder).

Adapted from a humanoid-UMI dataset visualizer — same "one row of
diagnostic panels per tracked frame + a right-margin stats block" style — but
targeted at OUR raw recordings (``data/recordings/session_*/take_*/``) instead of
a processed FDS shard, and EXTENDED with a force/torque row (the reference had no
F/T).

Why RAW (not the .zarr): the converted ``.zarr.zip`` is already de-glitched and
resampled onto the uniform camera grid, so a plot of it can't SHOW a glitch — the
whole point of a QC sheet. These panels plot the native-rate recorded streams
(tracker ~80 Hz, F/T ~230 Hz, gripper ~30 Hz) exactly as they land on disk, and
overlay the SAME de-glitch verdict the converter uses.

Load-bearing fidelity: the position-glitch dots, the KEEP/DROP badge, and the
"bridged / UNBRIDGED" counts are NOT a re-invented heuristic — they come straight
from ``umi_convert_core.deglitch_tracker_positions`` /
``tracker_glitch_severity`` run in the converter's exact sanitize order
(stale-drop -> ``make_strictly_increasing`` -> deglitch), gated against the
converter's own ``TRACKER_DROP_MAX_UNBRIDGED`` / ``TRACKER_DROP_RESIDUAL_JUMP_M``.
So a take this sheet marks DROP is a take ``convert_to_umi`` would drop, and the
red dots are the samples it would bridge over. This mirrors ``audit_session.py``
(the numeric twin of this visual tool).

Per take the figure has one row per tracked frame (1pt = one row) plus a
gripper/timing row and a force/torque row; columns per tracker row:
    [ raw pos xyz (+rolling median, drop dots) | raw orientation rpy |
      pos dev-from-median (cm) | rot dev-from-median (deg) | accel |2nd diff| (cm)
      | STATS text ]

Output: by default ``<take>/qc_<take>.png`` (one-to-one with the episode folder);
``--out-dir DIR`` redirects every PNG into DIR instead.

Run (no matplotlib in .venv — use uv, same as the reference scripts):
    UV_ONLY_BINARY=:all: uv run --no-project --python 3.12 \
      --with numpy --with scipy --with matplotlib \
      python processing/plot_take_qc.py data/recordings/session_2026.08.03_21.23.19

    # subset:            ... <session> --takes 000 076
    # all in one folder: ... <session> --out-dir data/qc/session_...
    # annotate shipped-dataset exclusions (badge only; still plotted):
    #                    ... <session> --dropped 011 013 025 037 048 058 072 074 075 076 083 088

Or just: ``./run_take_qc.sh <session> [args...]`` from the repo root.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

# --- repo imports (this file lives in processing/; recording_format is a sibling pkg) ---
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)                                     # processing/
sys.path.insert(0, os.path.join(_HERE, "..", "recording"))    # recording/
import recording_format as fmt  # noqa: E402
import umi_convert_core as core  # noqa: E402

# Import the converter's ACTUAL episode-drop thresholds so this sheet's KEEP/DROP
# badge can never drift from the real converter decision (import is side-effect
# free — convert_to_umi.main is under __name__ guard). Fall back to the documented
# defaults if the module ever renames them (same pattern as audit_session.py).
try:
    from convert_to_umi import (  # noqa: E402
        TRACKER_DROP_MAX_UNBRIDGED as _DROP_N,
        TRACKER_DROP_RESIDUAL_JUMP_M as _DROP_J,
    )
except Exception:  # noqa: BLE001
    _DROP_N, _DROP_J = 3, 0.30

# QC thresholds (kept identical to audit_session.py so the visual + numeric tools agree).
GRASP_THRESH_M = 0.08              # UMI hard-coded grasp threshold
MIN_FRAMES = 30                    # a usable episode floor (~2s @ 15 Hz camera)
DEAD_TRACKER_PATH_M = 0.10         # total path below this => tracker effectively frozen
POS_DEV_FLOOR_CM = 1.5             # visual floor line on the pos-dev panel (reference parity)
ROT_STEP_FLOOR_DEG = 8.0           # rotation-glitch candidate floor (audit ROT_LABEL_STEP_FLOOR_DEG)
ROT_GLITCH_WINDOW = 5              # camera frames each side of the worst step (audit parity)
ROT_PATH_NET_RATIO = 3.0           # summed-path / net-rotation over the window: >this = there-and-back
FLAT_FORCE_N = 0.5                 # |F| max below this => sensor flat / no contact all take
SAT_FORCE_N = 200.0                # |F| max above this => possible saturation / collision
MEDIAN_WIN = 7                     # rolling-median kernel (odd) for the dev-from-median panels

# Tracker-frame rows to look for (LEFT-canonical; 1pt has only the primary).
# Derived from fmt.TRACKER_FILE_BY_ROLE rather than listed, so a role added there (the two
# 5pt feet were) gets a QC row automatically. A sheet that silently omits a recorded
# keypoint is the failure this file exists to prevent.
TRACKER_ROWS = [(role, fname) for role, fname in fmt.TRACKER_FILE_BY_ROLE.items()]
FRAME_COLORS = {"left": "tab:blue", "right": "tab:orange", "head": "tab:green",
                "left_foot": "tab:purple", "right_foot": "tab:red"}


# --------------------------------------------------------------------------- IO


def _load(take_dir: str, name: str) -> "np.ndarray | None":
    p = os.path.join(take_dir, name)
    if not os.path.exists(p):
        return None
    a = np.load(p)
    return a if a.size else None


def _load_base_calib() -> "core.GripperWidthCalib | None":
    """Repo gripper_range.json -> a base calib carrying the physical width_open_m
    (effective_gripper_calib needs a non-None base or returns source 'none').
    Same loader as audit_session.py."""
    import glob as _glob
    for p in ("robot_assets/calibration/gripper_range.json",
              os.path.join(_HERE, "..", "robot_assets", "calibration", "gripper_range.json"),
              *_glob.glob("**/gripper_range.json", recursive=True)):
        if os.path.exists(p):
            j = json.load(open(p))
            return core.GripperWidthCalib(
                enc_closed=j.get("enc_closed", 0.0), enc_open=j.get("enc_open", 4.9),
                width_closed_m=j.get("width_closed_m", 0.0),
                width_open_m=j.get("width_open_m", 0.095))
    return None


_BASE_CALIB = _load_base_calib()


# ------------------------------------------------------------------- signal math


def _medfilt(x: np.ndarray, win: int = MEDIAN_WIN) -> np.ndarray:
    """1-D rolling median per column, edge-clamped. Small win, so a plain
    sliding window (no scipy dependency at import time). ``x`` is (N,) or (N,C)."""
    x = np.asarray(x, dtype=np.float64)
    one_d = x.ndim == 1
    if one_d:
        x = x[:, None]
    n = x.shape[0]
    win = max(1, min(win, n if n % 2 == 1 else n - 1))
    if win <= 1:
        return x[:, 0] if one_d else x
    half = win // 2
    pad = np.pad(x, ((half, half), (0, 0)), mode="edge")
    out = np.empty_like(x)
    for i in range(n):
        out[i] = np.median(pad[i:i + win], axis=0)
    return out[:, 0] if one_d else out


def _canonical_quat_xyzw(q_wxyz: np.ndarray) -> np.ndarray:
    """wxyz -> xyzw, then resolve the double-cover so consecutive quats live on
    the same hemisphere (component-wise median filtering is meaningless across a
    sign flip). Returns (N,4) xyzw."""
    q = core.quat_wxyz_to_xyzw(np.asarray(q_wxyz, dtype=np.float64))
    q = q / np.clip(np.linalg.norm(q, axis=1, keepdims=True), 1e-12, None)
    for i in range(1, q.shape[0]):
        if float(np.dot(q[i], q[i - 1])) < 0.0:
            q[i] = -q[i]
    return q


def _geodesic_deg(q_a_xyzw: np.ndarray, q_b_xyzw: np.ndarray) -> np.ndarray:
    """Per-row geodesic angle (deg) between two xyzw quaternion arrays."""
    dots = np.abs(np.sum(q_a_xyzw * q_b_xyzw, axis=1))
    return np.degrees(2.0 * np.arccos(np.clip(dots, 0.0, 1.0)))


def sanitize_tracker(tr: np.ndarray) -> dict:
    """Mirror the converter's tracker sanitize EXACTLY and return everything the
    QC panels need. Order (convert_to_umi): stale-drop -> make_strictly_increasing
    -> tracker_glitch_severity (pre-deglitch) -> deglitch_tracker_positions.

    Returns a dict with the strictly-increasing (pre-deglitch) stream the plots
    draw on, the boolean mask of samples the deglitch DROPS (the red dots), the
    unbridged spans (relative-time brackets to shade), and the severity dict.
    """
    t = tr[:, 0].astype(np.float64)
    pos = tr[:, 2:5].astype(np.float64)
    quat = tr[:, 5:9].astype(np.float64)               # wxyz
    stale = tr[:, 9].astype(np.float64) if tr.shape[1] > 9 else np.zeros(len(t))

    live = stale < 0.5
    if live.any():
        t, pos, quat = t[live], pos[live], quat[live]
    keep = core.make_strictly_increasing(t)
    t, pos, quat = t[keep], pos[keep], quat[keep]

    sev = {"n_unbridged": 0, "n_bridged_excursions": 0, "residual_max_jump_m": 0.0,
           "peak_speed_mps": 0.0, "max_jump_m": 0.0, "n_hot_steps": 0, "unbridged_spans": []}
    drop_mask = np.zeros(len(t), dtype=bool)
    if t.shape[0] >= 3:
        sev = core.tracker_glitch_severity(t, pos)
        kept_gl, stats = core.deglitch_tracker_positions(t, pos, return_stats=True)
        drop_mask[:] = True
        drop_mask[kept_gl] = False
        sev["n_bridged_excursions"] = stats["n_bridged_excursions"]
        sev["unbridged_spans"] = stats["unbridged_spans"]

    t0 = t[0] if len(t) else 0.0
    return {"t": t, "t0": t0, "trel": t - t0, "pos": pos, "quat_wxyz": quat,
            "drop_mask": drop_mask, "sev": sev, "n_stale_dropped": int((~live).sum())}


def rotation_label_glitch(s: dict, cam_t: "np.ndarray | None") -> dict:
    """The converter/audit label-grid rotation-glitch test, verbatim.

    A rotation glitch (tracker re-localization that flips orientation while
    position barely moves) survives the POSITION-only deglitch and lands in the
    resampled label. ``audit_session.py`` catches it by resampling the SANITIZED
    tracker quaternion onto the camera grid (the real training label), then, for
    the worst per-frame step ABOVE ``ROT_STEP_FLOOR_DEG``, comparing the summed
    path over +-ROT_GLITCH_WINDOW frames to the NET rotation across that window.
    A there-and-back glitch traces a long path for ~zero net (ratio high); genuine
    fast wrist rotation goes somewhere and stays (ratio ~1). Only ratio>RATIO is a
    glitch — raw step magnitude ALONE does not classify (takes 030/031/034 in
    clean94 reach ~14-23 deg/frame of GENUINE fast motion, ratio<2, and are kept).

    The label grid is necessary but not sufficient, so a SECOND, independent test runs
    ``core.rotation_excursions`` on the RAW stream: the camera grid attenuates an
    excursion that falls between two frames and stretches one inside a run of dropped
    frames, which is how a 21 deg there-and-back read 7 deg/frame at ratio 2.3 and passed
    both gates on the 0823 session. ``is_glitch`` is the OR of the two.

    Returns ``{max_step_deg, ratio, is_glitch, worst_frame, raw_amp_deg, raw_n}``;
    ``ratio``/``worst_frame`` are meaningful only when
    ``max_step_deg > ROT_STEP_FLOOR_DEG``, and ``raw_*`` come from the raw-stream scan.
    """
    out = {"max_step_deg": 0.0, "ratio": 0.0, "is_glitch": False, "worst_frame": -1,
           "raw_amp_deg": 0.0, "raw_n": 0}
    keep = ~s["drop_mask"]
    st, sq = s["t"][keep], s["quat_wxyz"][keep]
    if st.shape[0] >= 3:
        exc = core.rotation_excursions(st, sq)
        out["raw_n"] = len(exc)
        out["raw_amp_deg"] = max((e["amp_deg"] for e in exc), default=0.0)
        out["is_glitch"] = bool(exc)
    if cam_t is None or st.shape[0] < 2 or len(cam_t) < 2:
        return out
    rot = core.resample_rotation_wxyz(st, sq, np.asarray(cam_t, dtype=np.float64))
    rq = rot.as_quat()                                          # xyzw on the label grid
    step = _geodesic_deg(rq[:-1], rq[1:])
    if not step.size:
        return out
    out["max_step_deg"] = float(step.max())
    if out["max_step_deg"] <= ROT_STEP_FLOOR_DEG:
        return out
    j = int(np.argmax(step))
    lo = max(0, j - ROT_GLITCH_WINDOW)
    hi = min(len(step) - 1, j + ROT_GLITCH_WINDOW)              # last valid step index
    path = float(step[lo:hi + 1].sum())
    net = float(_geodesic_deg(rq[lo:lo + 1], rq[hi + 1:hi + 2])[0])
    out["ratio"] = path / max(net, 1e-6)
    # OR, never assign: the raw-stream scan above may already have flagged this take, and
    # a label-grid ratio under the gate is exactly the case it exists to cover.
    out["is_glitch"] = bool(out["is_glitch"] or out["ratio"] > ROT_PATH_NET_RATIO)
    out["worst_frame"] = j
    return out


def verdict_for(sev: dict, rot_glitch: dict, role: str = "left") -> tuple[str, str]:
    """KEEP / DROP badge from the converter's OWN drop gate, plus the rotation-glitch
    caveat audit_session flags (position-only pipeline can't mask a rot glitch).
    ``rot_glitch`` is ``rotation_label_glitch(...)`` — a there-and-back glitch
    (``is_glitch``), NOT mere raw step magnitude, is what earns CHECK.
    Returns (verdict, reason).

    ``role`` only changes the DROP wording. The converter's episode-drop gate applies to
    the PRIMARY tracker alone; for a secondary (2pt right hand / 3pt chest) it neither
    drops the take nor writes a pose_glitch mask, so "meets the drop gate" would read as
    "already handled downstream" when in fact nothing masks it. The primary's text is
    left byte-identical so the recorded per-session verdicts still reproduce."""
    if sev.get("n_unbridged", 0) >= _DROP_N or sev.get("residual_max_jump_m", 0.0) >= _DROP_J:
        if role != "left":
            return ("DROP", f"{sev['n_unbridged']} unbridged / residual "
                            f"{sev['residual_max_jump_m'] * 1000:.0f}mm — would meet the drop "
                            f"gate ({_DROP_N}+ or {_DROP_J * 1000:.0f}mm), but that gate is "
                            f"PRIMARY-only: this keypoint is neither dropped nor masked, so "
                            f"--exclude the take to keep it out of a {role}-using dataset")
        return ("DROP", f"{sev['n_unbridged']} unbridged / residual "
                        f"{sev['residual_max_jump_m'] * 1000:.0f}mm >= drop gate "
                        f"({_DROP_N}+ or {_DROP_J * 1000:.0f}mm)")
    if rot_glitch.get("is_glitch"):
        if rot_glitch.get("raw_n"):
            return ("CHECK", f"rot glitch {rot_glitch['raw_amp_deg']:.0f}deg RAW "
                             f"there-and-back excursion x{rot_glitch['raw_n']} "
                             f"(label grid reads {rot_glitch['max_step_deg']:.0f}deg/frame, "
                             f"path/net {rot_glitch['ratio']:.1f}x; position-only deglitch "
                             f"can't mask it)")
        return ("CHECK", f"rot glitch {rot_glitch['max_step_deg']:.0f}deg/frame, "
                         f"path/net {rot_glitch['ratio']:.1f}x (there-and-back; "
                         f"position-only deglitch can't mask it)")
    if sev.get("n_unbridged", 0) > 0:
        return ("KEEP*", f"{sev['n_unbridged']} sub-threshold unbridged "
                         f"(converter keeps; loader drops straddling windows)")
    return ("KEEP", "clean")


# ----------------------------------------------------------------------- plotting


def plot_take(take_dir: str, out_path: str, dropped_from_dataset: bool = False) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    take = os.path.basename(take_dir)
    meta = {}
    mp = os.path.join(take_dir, fmt.TAKE_META_FILE)
    if os.path.exists(mp):
        meta = json.load(open(mp))
    approx_hz = meta.get("approx_hz", {})

    fidx = _load(take_dir, os.path.join(fmt.CAMERA_DIR, fmt.CAMERA_INDEX_FILE))
    cam_t = fidx[:, 0].astype(np.float64) if fidx is not None else None
    nframes = 0 if cam_t is None else len(cam_t)

    trackers = [(role, tr) for role, f in TRACKER_ROWS
                if (tr := _load(take_dir, f)) is not None]
    grip = _load(take_dir, fmt.GRIPPER_FILE)
    ft = _load(take_dir, fmt.FT_FILE)
    ftc = _load(take_dir, fmt.FT_COMP_FILE)
    tare = _load(take_dir, fmt.TARE_EVENTS_FILE)

    nrows = len(trackers) + 2                       # trackers + gripper/timing + force/torque
    ncols = 6                                       # 5 plot columns + 1 text column
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 3.4 * nrows), squeeze=False)

    junk = nframes and nframes < MIN_FRAMES
    title = (f"{take}   T_cam={nframes}   mode={meta.get('mode', '?')}/"
             f"{meta.get('primary_role', '?')}   "
             f"cam {approx_hz.get('camera', 0):.1f}Hz  tr {approx_hz.get('tracker', 0):.1f}Hz  "
             f"ft {approx_hz.get('ft', 0):.0f}Hz")
    if junk:
        title += f"   [JUNK: {nframes} frames < {MIN_FRAMES}]"
    if dropped_from_dataset:
        # Dataset-agnostic on purpose: --dropped is passed per session, and a sheet that
        # names one old dataset ("clean94") reads as provenance while being wrong.
        title += "   [EXCLUDED from the shipped dataset]"
    fig.suptitle(title, fontsize=13, fontweight="bold",
                 color=("crimson" if (junk or dropped_from_dataset) else "black"))

    # ---- tracker rows -----------------------------------------------------
    for ri, (role, tr) in enumerate(trackers):
        s = sanitize_tracker(tr)
        trel, pos, sev = s["trel"], s["pos"], s["sev"]
        drop = s["drop_mask"]
        color = FRAME_COLORS.get(role, "tab:blue")
        row = axes[ri]

        # col0: raw pos xyz + rolling median + drop dots + unbridged span shading
        a = row[0]
        med = _medfilt(pos)
        for j, axis in enumerate("xyz"):
            a.plot(trel, pos[:, j], lw=0.9, label=axis)
            a.plot(trel, med[:, j], lw=0.6, ls="--", alpha=0.4, color="k")
        if drop.any():
            a.plot(trel[drop], pos[drop, 0], "r.", ms=4, label="deglitch drop")
            a.plot(trel[drop], pos[drop, 1], "r.", ms=4)
            a.plot(trel[drop], pos[drop, 2], "r.", ms=4)
        for (ts, te) in sev.get("unbridged_spans", []):
            a.axvspan(ts - s["t0"], te - s["t0"], color="red", alpha=0.15)
        a.set_ylabel(f"{role}\nraw pos (m)")
        a.set_title("raw position xyz (+median, drop dots)", fontsize=9)
        a.legend(loc="upper right", fontsize=6, ncol=2)

        # col1: raw orientation rpy (deg), unwrapped
        a = row[1]
        rpy = np.zeros((len(trel), 3))
        if len(trel) >= 1:
            from scipy.spatial.transform import Rotation
            q_xyzw = core.quat_wxyz_to_xyzw(s["quat_wxyz"])
            rpy = np.degrees(np.unwrap(np.radians(
                Rotation.from_quat(q_xyzw).as_euler("xyz", degrees=True)), axis=0))
        for j, nm in enumerate(("roll", "pitch", "yaw")):
            a.plot(trel, rpy[:, j], lw=0.9, label=nm)
        a.set_title("raw orientation rpy (deg)", fontsize=9)
        a.legend(loc="upper right", fontsize=6, ncol=3)

        # col2: position dev-from-median (cm) + floor line + drop dots
        a = row[2]
        dev_cm = np.linalg.norm(pos - med, axis=1) * 100.0
        a.plot(trel, dev_cm, lw=0.8, color=color)
        if drop.any():
            a.plot(trel[drop], dev_cm[drop], "r.", ms=4)
        a.axhline(POS_DEV_FLOOR_CM, color="crimson", lw=0.8, alpha=0.5)
        a.set_title(f"position glitch (dev-from-median, cm; floor {POS_DEV_FLOOR_CM})", fontsize=9)

        # col3: rotation dev-from-median (deg). The label-grid path/net verdict
        # (rot_g) is what drives the badge — the raw dev/step here is only the
        # visual signal.
        a = row[3]
        rot_step_max = 0.0
        rot_g = rotation_label_glitch(s, cam_t)
        if len(trel) >= 3:
            q_can = _canonical_quat_xyzw(s["quat_wxyz"])
            q_med = _medfilt(q_can)
            q_med = q_med / np.clip(np.linalg.norm(q_med, axis=1, keepdims=True), 1e-12, None)
            rot_dev = _geodesic_deg(q_can, q_med)
            a.plot(trel, rot_dev, lw=0.8, color=color)
            step = _geodesic_deg(q_can[:-1], q_can[1:])
            rot_step_max = float(step.max()) if step.size else 0.0
        ttl = "rotation glitch (dev-from-median, deg)"
        if rot_g["is_glitch"]:
            ttl += f"  [GLITCH path/net {rot_g['ratio']:.1f}x]"
        a.set_title(ttl, fontsize=9,
                    color=("crimson" if rot_g["is_glitch"] else "black"))

        # col4: accel |2nd diff| (cm) — jitter
        a = row[4]
        acc_max = 0.0
        if len(pos) >= 3:
            acc_cm = np.linalg.norm(np.diff(pos, n=2, axis=0), axis=1) * 100.0
            a.plot(trel[1:-1], acc_cm, lw=0.7, color=color)
            acc_max = float(acc_cm.max()) if acc_cm.size else 0.0
        a.set_title("accel |2nd diff| (cm)", fontsize=9)

        # col5: stats text + KEEP/DROP badge
        a = row[5]
        a.axis("off")
        path_m = float(np.linalg.norm(np.diff(pos, axis=0), axis=1).sum()) if len(pos) > 1 else 0.0
        step_all = np.linalg.norm(np.diff(pos, axis=0), axis=1) if len(pos) > 1 else np.zeros(1)
        dev_max = float(dev_cm.max()) if dev_cm.size else 0.0
        n_nonfinite = int((~np.isfinite(tr[:, 2:9])).any(axis=1).sum())
        verdict, reason = verdict_for(sev, rot_g, role)
        vcolor = {"KEEP": "green", "KEEP*": "darkgoldenrod",
                  "CHECK": "darkorange", "DROP": "crimson"}[verdict]
        lines = [
            f"samples      : {len(trel)}  (stale-drop {s['n_stale_dropped']})",
            f"path length  : {path_m:.3f} m  (dead <{DEAD_TRACKER_PATH_M})"
            + ("  ** DEAD **" if 0 < path_m < DEAD_TRACKER_PATH_M else ""),
            f"POS glitch   : {sev['n_bridged_excursions']} bridged, "
            f"{sev['n_unbridged']} UNBRIDGED",
            f"  max dev    : {dev_max:.2f} cm",
            f"  max step   : {step_all.max() * 100:.2f} cm (p99 {np.percentile(step_all, 99) * 100:.2f})",
            f"  peak speed : {sev['peak_speed_mps']:.1f} m/s (hot >3.0)",
            f"  residual   : {sev['residual_max_jump_m'] * 1000:.0f} mm (kept stream)",
            f"ROT raw step : {rot_step_max:.1f} deg (raw stream)",
            f"ROT label    : {rot_g['max_step_deg']:.1f} deg/frame, "
            f"path/net {rot_g['ratio']:.1f}x" + ("  ** GLITCH **" if rot_g["is_glitch"] else ""),
            f"max accel    : {acc_max:.2f} cm",
            f"non-finite   : {n_nonfinite} frames",
        ]
        a.text(0.0, 1.0, "\n".join(lines), family="monospace", fontsize=8.5,
               va="top", ha="left", transform=a.transAxes)
        a.text(0.0, 0.02, f"VERDICT: {verdict}\n{reason}", family="monospace",
               fontsize=9.5, fontweight="bold", color=vcolor, va="bottom", ha="left",
               transform=a.transAxes, wrap=True)

    # ---- gripper / timing row --------------------------------------------
    grow = axes[len(trackers)]
    # col0: encoder_rad + width_m (twin axis) + grasp threshold
    a = grow[0]
    if grip is not None:
        gt = grip[:, 0] - grip[0, 0]
        enc = grip[:, 1].astype(np.float64)
        a.plot(gt, enc, lw=0.8, color="tab:purple", label="encoder_rad")
        a.set_ylabel("encoder (rad)")
        cal, src = core.effective_gripper_calib(grip, _BASE_CALIB)
        if cal is not None:
            w = cal.to_width_m(enc)
            a2 = a.twinx()
            a2.plot(gt, w, lw=0.9, color="tab:green", label="width_m")
            a2.axhline(GRASP_THRESH_M, color="crimson", lw=0.7, alpha=0.6)
            a2.set_ylabel("width (m)")
            a.set_title(f"gripper (calib={src}, grasp line {GRASP_THRESH_M}m)", fontsize=9)
        else:
            a.set_title("gripper encoder (no calib)", fontsize=9)
    else:
        a.set_title("gripper (missing)", fontsize=9)
        a.axis("off")
    # col1: width step (Δ per sample)
    a = grow[1]
    if grip is not None and grip.shape[0] > 1:
        cal, _ = core.effective_gripper_calib(grip, _BASE_CALIB)
        if cal is not None:
            w = cal.to_width_m(grip[:, 1].astype(np.float64))
            a.plot((grip[:, 0] - grip[0, 0])[1:], np.abs(np.diff(w)) * 1000.0, lw=0.7, color="tab:green")
            a.set_title("gripper |Δwidth| (mm/sample)", fontsize=9)
    else:
        a.axis("off")
    # col2: camera cadence (dt ms) — master-timeline uniformity
    a = grow[2]
    if cam_t is not None and len(cam_t) > 1:
        cdt = np.diff(cam_t) * 1000.0
        a.plot(cam_t[1:] - cam_t[0], cdt, lw=0.7, color="tab:red")
        a.axhline(float(np.median(cdt)), color="k", lw=0.6, ls="--", alpha=0.5)
        a.set_title(f"camera dt (ms), median {np.median(cdt):.1f}", fontsize=9)
    else:
        a.axis("off")
    # col3: F/T new-samples per camera step (K-window coverage)
    a = grow[3]
    if ft is not None and cam_t is not None and len(cam_t) > 1:
        ft_t = ft[:, 0].astype(np.float64)
        n_le = np.searchsorted(ft_t, cam_t, side="right")
        per_step = np.diff(np.concatenate([[0], n_le]))
        a.plot(cam_t - cam_t[0], per_step, lw=0.7, color="tab:brown")
        a.set_title(f"F/T new-samples / cam step (min {int(per_step.min())})", fontsize=9)
    else:
        a.axis("off")
    # col4: gripper width histogram
    a = grow[4]
    if grip is not None:
        cal, _ = core.effective_gripper_calib(grip, _BASE_CALIB)
        if cal is not None:
            w = cal.to_width_m(grip[:, 1].astype(np.float64))
            a.hist(w, bins=30, color="tab:green", alpha=0.7)
            a.axvline(GRASP_THRESH_M, color="crimson", lw=0.8)
            a.set_title("gripper width dist (m)", fontsize=9)
        else:
            a.axis("off")
    else:
        a.axis("off")
    # col5: gripper stats
    a = grow[5]
    a.axis("off")
    glines = [f"cam frames   : {nframes}"]
    if grip is not None:
        cal, src = core.effective_gripper_calib(grip, _BASE_CALIB)
        if cal is not None:
            w = cal.to_width_m(grip[:, 1].astype(np.float64))
            crosses = bool(w.min() < GRASP_THRESH_M < w.max())
            glines += [f"grip calib   : {src}",
                       f"width range  : [{w.min() * 100:.1f}, {w.max() * 100:.1f}] cm",
                       f"crosses grasp: {'YES' if crosses else 'no'} ({GRASP_THRESH_M}m)"]
    if cam_t is not None and len(cam_t) > 1:
        cdt = np.diff(cam_t)
        spread = (np.percentile(cdt, 95) - np.percentile(cdt, 5)) / max(np.median(cdt), 1e-9)
        glines += [f"cam dt spread: {spread * 100:.1f}% "
                   + ("(NON-UNIFORM)" if spread > 0.05 else "(uniform)")]
    a.text(0.0, 1.0, "\n".join(glines), family="monospace", fontsize=8.5,
           va="top", ha="left", transform=a.transAxes)

    # ---- force / torque row ----------------------------------------------
    frow = axes[len(trackers) + 1]
    tare_t = tare[:, 0].astype(np.float64) if tare is not None else None

    def _plot_ft(a, arr, arrc, sl, title, ylab):
        """components (comp preferred, raw faint) for wrench columns ``sl``."""
        drew = False
        if arr is not None:
            t0 = arr[0, 0]
            for j in range(sl.start, sl.stop):
                a.plot(arr[:, 0] - t0, arr[:, j], lw=0.5, alpha=0.35)
            drew = True
        if arrc is not None:
            t0c = arrc[0, 0]
            for j, nm in zip(range(sl.start, sl.stop), ("x", "y", "z")):
                a.plot(arrc[:, 0] - t0c, arrc[:, j], lw=0.9, label=nm)
            drew = True
        a.set_title(title, fontsize=9)
        a.set_ylabel(ylab)
        if drew:
            a.legend(loc="upper right", fontsize=6, ncol=3)

    t0f = ft[0, 0] if ft is not None else (ftc[0, 0] if ftc is not None else 0.0)

    def _tare_lines(a):
        # Only tares that fall INSIDE the already-plotted data span. The session-level
        # tare fires ~70 s before a take starts (see take_meta tare_event_wall), so an
        # unguarded axvline at t=-70 would stretch the F/T axis and cram the real force
        # trace into the right margin. Snapshot the data-driven xlim, draw in-range
        # tares only, then restore it so the axis still frames the force data.
        if tare_t is None:
            return
        lo, hi = a.get_xlim()
        for tt in tare_t:
            x = tt - t0f
            if lo <= x <= hi:
                a.axvline(x, color="k", lw=0.6, ls=":", alpha=0.6)
        a.set_xlim(lo, hi)

    # col0: Fx Fy Fz
    _plot_ft(frow[0], ft, ftc, slice(1, 4), "force Fxyz (comp bold, raw faint) [N]", "F (N)")
    _tare_lines(frow[0])
    # col1: |F| raw vs comp + tare
    a = frow[1]
    if ft is not None:
        a.plot(ft[:, 0] - t0f, np.linalg.norm(ft[:, 1:4], axis=1), lw=0.6, alpha=0.5, label="|F| raw")
    if ftc is not None:
        a.plot(ftc[:, 0] - t0f, np.linalg.norm(ftc[:, 1:4], axis=1), lw=0.9, label="|F| comp")
    _tare_lines(a)
    a.set_title("|F| raw vs comp [N] (: = tare)", fontsize=9)
    a.legend(loc="upper right", fontsize=6)
    # col2: Tx Ty Tz
    _plot_ft(frow[2], ft, ftc, slice(4, 7), "torque Txyz (comp bold, raw faint) [Nm]", "T (Nm)")
    _tare_lines(frow[2])
    # col3: |T| raw vs comp + tare
    a = frow[3]
    if ft is not None:
        a.plot(ft[:, 0] - t0f, np.linalg.norm(ft[:, 4:7], axis=1), lw=0.6, alpha=0.5, label="|T| raw")
    if ftc is not None:
        a.plot(ftc[:, 0] - t0f, np.linalg.norm(ftc[:, 4:7], axis=1), lw=0.9, label="|T| comp")
    _tare_lines(a)
    a.set_title("|T| raw vs comp [Nm]", fontsize=9)
    a.legend(loc="upper right", fontsize=6)
    # col4: |F| jitter (|Δ|F|| per sample)
    a = frow[4]
    src = ftc if ftc is not None else ft
    if src is not None and src.shape[0] > 1:
        fmag = np.linalg.norm(src[:, 1:4], axis=1)
        a.plot((src[:, 0] - src[0, 0])[1:], np.abs(np.diff(fmag)), lw=0.5, color="tab:red")
        a.set_title("force jitter |Δ|F|| (N/sample)", fontsize=9)
    else:
        a.axis("off")
    # col5: force stats
    a = frow[5]
    a.axis("off")
    flines = []
    if ft is not None:
        fmag_raw = np.linalg.norm(ft[:, 1:4], axis=1)
        flines += [f"F/T samples  : {ft.shape[0]} @ {approx_hz.get('ft', 0):.0f}Hz",
                   f"|F| raw med  : {np.median(fmag_raw):.1f} N",
                   f"|F| raw max  : {fmag_raw.max():.1f} N"]
        if fmag_raw.max() < FLAT_FORCE_N:
            flines.append("  ** FLAT (no contact?) **")
        if fmag_raw.max() > SAT_FORCE_N:
            flines.append("  ** SATURATION? **")
    if ftc is not None:
        fmag_c = np.linalg.norm(ftc[:, 1:4], axis=1)
        tmag_c = np.linalg.norm(ftc[:, 4:7], axis=1)
        flines += [f"|F| comp med : {np.median(fmag_c):.1f} N",
                   f"|T| comp med : {np.median(tmag_c):.2f} Nm"]
    else:
        flines.append("comp         : ABSENT (raw only)")
    flines += [f"ft_valid     : {meta.get('ft_valid', '?')}",
               f"ft_comp_algo : {meta.get('ft_comp_algo', '?')}",
               f"n_tare       : {0 if tare is None else tare.shape[0]}"]
    a.text(0.0, 1.0, "\n".join(flines), family="monospace", fontsize=8.5,
           va="top", ha="left", transform=a.transAxes)

    for r in range(nrows):
        for c in range(ncols - 1):
            axes[r][c].set_xlabel("t (s)", fontsize=7)
            axes[r][c].tick_params(labelsize=7)

    fig.tight_layout(rect=(0, 0, 1, 0.985))
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------- main


def _canon(x: str) -> str:
    """'5' / '005' / 'take_005' -> 'take_005' (matches convert_to_umi._canon_take)."""
    b = os.path.basename(str(x))
    if b.startswith("take_"):
        return b
    d = "".join(ch for ch in b if ch.isdigit())
    return f"take_{int(d):03d}" if d else b


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session", help="data/recordings/session_* dir (or a single take_* dir)")
    ap.add_argument("--takes", nargs="*", default=None,
                    help="subset by name/number (000 / take_000 / 5); default = all takes")
    ap.add_argument("--out-dir", default=None,
                    help="write all PNGs here instead of into each take folder")
    ap.add_argument("--dropped", nargs="*", default=(),
                    help="take names excluded from the shipped dataset — badge only, still plotted")
    args = ap.parse_args(argv)

    session = os.path.abspath(args.session)
    if os.path.basename(session).startswith("take_"):
        take_dirs = [session]
    else:
        take_dirs = sorted(os.path.join(session, d) for d in os.listdir(session)
                           if d.startswith("take_") and os.path.isdir(os.path.join(session, d)))
    if args.takes:
        want = {_canon(x) for x in args.takes}
        take_dirs = [d for d in take_dirs if os.path.basename(d) in want]
    if not take_dirs:
        print(f"no take_* dirs under {session}", file=sys.stderr)
        return 1

    dropped = {_canon(x) for x in args.dropped}
    print(f"plotting {len(take_dirs)} take(s) from {session}")
    if dropped:
        print(f"  (annotating excluded-from-dataset: {sorted(dropped)})")
    ok = 0
    for td in take_dirs:
        take = os.path.basename(td)
        out = (os.path.join(args.out_dir, f"qc_{take}.png") if args.out_dir
               else os.path.join(td, f"qc_{take}.png"))
        try:
            plot_take(td, out, dropped_from_dataset=take in dropped)
            print(f"  wrote {out}")
            ok += 1
        except Exception as e:  # noqa: BLE001
            print(f"  [ERR] {take}: {type(e).__name__}: {e}", file=sys.stderr)
    print(f"done: {ok}/{len(take_dirs)} take(s)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
