#!/usr/bin/env python3
"""Replay a RAW recorded take in viser — the yam gripper, camera, and F/T as they
were captured, with a draggable timeline to stop at any timestamp.

This is the "watch what I just recorded" tool (the FIRST of the two replay
scripts). It reads a committed ``take_*`` from a session under
``data/recordings/`` and reconstructs the SAME gripper-only scene the live
collection viz draws for ``MODE=1pt``/``2pt`` (``DISPLAY_MODE=gripper``) — no
humanoid, no mink IK. It is the RAW-take twin of ``replay_zarr_viz.py`` (which
shows the converted ``.zarr.zip``); the two render the SAME cluster so a raw take
and its conversion can be eyeballed side by side.

For the PRIMARY hand it shows, riding one moving TCP frame:

  * the three yam-gripper meshes (base + two fingers, URDF colours baked in), with
    the jaws LERPed from the recorded encoder width (absolute metres via the take's
    homed endpoints — same calibration resolution as ``convert_to_umi``);
  * EXACTLY the three axis triads the collection viz draws — **TCP** (drawn at the
    fingertip grasp point), **tracker** (the physical VIVE puck origin), and
    **wrist_yaw_link** — nothing else;
  * the recorded D405 JPEGs on a camera frustum mounted off the wrist, and the Bota
    wrench as force/torque arrows + scrolling charts, with a "Compensated wrench"
    checkbox to switch between the RAW sensor wrench and the gravity/inertia-
    COMPENSATED sidecar (when the take recorded one), mirroring the live viz;
  * the full ``robot0_eef_pos`` (TCP) path as a GREEN polyline, so the trajectory
    is anchored on — and moves with — the TCP.

The TCP pose is derived from the recorded raw VIVE puck through the SAME
tracker->TCP extrinsic the converter uses (``umi_convert_core.apply_tcp_extrinsic``
with ``robot_assets/calibration/tcp_extrinsic.json``), so what you see here is the
exact ``robot0_eef_pos`` the ``.zarr.zip`` will store.

**Display defaults to the RAW recorded frame** (no home-anchor, no re-levelling),
matching ``RAW_FRAME=1 collect_data.sh``: the tracker triad IS the raw libsurvive
puck, and the TCP position + AXES follow the remapped extrinsic (whose rotation is
the measured ``-y,-x,-z`` puck->wrist relabel). This is the one convention shared by
collection, this viewer, and the zarr. ``--no-raw-pose`` re-enables the legacy
home-anchored view, which applies a world rotation the recording does not contain —
comfortable to look at, but NOT what was recorded, so never use it to judge fidelity.

**Playback runs at the tracker's NATIVE rate (~80 Hz), NOT the ~15 Hz camera grid.**
The camera grid is the converter's master clock, but decimating the MOTION to it makes
the replayed hand STEP between RGB frames; the tracker samples ~5x faster, so the moving
cluster + path + F/T + gripper play on the tracker's own timestamps (the "display grid")
and the RGB — which only exists at ~15 Hz — is HELD across the frames it spans. The
camera-grid resample is kept only for the converter-fidelity contract and the charts.

**Reviewing a whole session: the "Take" panel.** Point it at a session dir and the
Take panel pages through every committed take without a relaunch — a clickable ``Take``
dropdown labelled ``take_007  (687 fr, 25s)`` (the raw twin of the zarr viewer's Episode
dropdown), ⏮/⏭ Prev/Next take, and 🗑 **Delete this take**. Each switch tears the scene +
GUI down and rebuilds from the newly loaded take (~0.02 s), so nothing take-shaped can
survive into the next one. Delete is a REAL delete (one confirm click), and its two cases
differ — see :func:`delete_take`: a MERGED session's ``take_*`` is a symlink, so only the
link goes and the per-sitting recording is kept; a plain take dir is ``rmtree``'d. Either
way the converter's ``take_*`` glob stops seeing it, which is the point: prune while you
watch, then re-convert.

It is otherwise purely presentational and OFFLINE: it never connects to a bridge, never
writes anything, and is independent of the live ``RECORD=1`` pipeline. To keep
scrubbing instant, ALL per-frame quantities are PRE-COMPUTED at load, then the
timeline just indexes cached arrays. No mink import — the gripper cluster is placed
with pure numpy (``gripper_only_view``), exactly as ``DISPLAY_MODE=gripper`` does.

Run (no PYTHONPATH needed — there is no IK; run_replay_viz.sh calls it directly):
    .venv/bin/python replay/replay_raw_viz.py data/recordings/session_2026.06.20_15.08.35
    .venv/bin/python replay/replay_raw_viz.py data/recordings/session_XXX/take_000 --port 8083
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

import numpy as np
import viser
from viser import transforms as vtf

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.join(_HERE, "..")
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_REPO, "recording"))
sys.path.insert(0, os.path.join(_REPO, "processing"))
sys.path.insert(0, os.path.join(_REPO, "viz"))

import recording_format as fmt  # noqa: E402
import umi_convert_core as core  # noqa: E402
import convert_to_umi as conv  # noqa: E402
import replay_common as rc  # noqa: E402
# Gripper-cluster geometry + the home-anchored delta map — the SAME pure module the
# collection viz uses for DISPLAY_MODE=gripper (numpy + yourdfpy only, no mink).
import gripper_only_view as _gov  # noqa: E402
# The three display-triad offsets, the single source shared with collection_viz /
# replay_zarr_viz (wrist-relative). viz/ is on sys.path above.
from collection_viz import (  # noqa: E402
    _TCP_ALONG_WRIST_X_M, _TRACKER_TRUE_XYZ_M, _WRIST_YAW_XYZ_M,
)

_DEFAULT_YAM = os.path.join(_REPO, "robot_assets", "g1_yam", "g1_29dof_yam.urdf")

# Green — the collection GT trajectory colour, matching replay_zarr_viz.
_GT_GREEN = (90, 220, 120)
#: RIGHT-hand TCP path. Yellow, never green: the two hands' paths cross and overlap in
#: bimanual takes, and a same-colour pair is unreadable the moment they do.
_RIGHT_PATH_YELLOW = (250, 210, 40)
# PER-ROLE home pose for the anchored display, straight out of gripper_only_view: the
# G1's ZERO CONFIGURATION (forearms level along +x, wrists parallel, 0.297 m apart) for
# the hands and holosoma's torso point for the chest. Each keypoint is rebased onto its
# OWN home with its OWN first frame as the reference, so a home-anchored 2pt/3pt take
# starts in exactly the arrangement the collection viz and the teleop bridge start in.
# Shared constant, so the raw viewer, the zarr viewer and the live viz cannot drift.
_HOME_BY_ROLE = {
    role: (np.asarray(hp, dtype=np.float64), np.asarray(hq, dtype=np.float64))
    for role, (hp, hq) in _gov.HOME_WRIST_POSE.items()
}
# All three home rotations are identity today (upright, +z up like collection); kept as a
# named constant for the single-keypoint call paths that still take a bare quaternion.
_HOME_Q = np.array([1.0, 0.0, 0.0, 0.0])


# ---------------------------------------------------------------------------
# Load one take + pre-compute every per-frame quantity onto the CAMERA grid
# ---------------------------------------------------------------------------
class RawTake:
    """All streams of one committed take, on TWO strictly-increasing timelines.

    * The **camera grid** (``cam_t``, ~15 Hz) is the converter's master clock and is
      kept ONLY for the stored eef (:meth:`tcp_pose_grid`, the fidelity contract the
      tests pin) and the honest per-episode frame counts (``segments``).
    * The **display grid** (``disp_t``, the tracker's NATIVE ~80 Hz — falls back to the
      camera grid when a take has no usable tracker) is the PLAYBACK master: the moving
      cluster (gripper meshes, triads, camera pose, F/T arrows) and the scrub timeline
      ride it, so playback follows the real hand motion instead of stepping between RGB
      frames. The JPEG only exists at ~15 Hz, so each display frame HOLDS the most-recent
      camera frame (``disp_cam_idx``).

    Pose is resampled with Slerp/interp1d, gripper width with the take's homed
    endpoints, F/T anti-aliased — the SAME core helpers the converter uses, so a display
    frame that happens to land on a camera timestamp equals what conversion sampled."""

    def __init__(self, take_dir: str, session_dir: str):
        self.take_dir = os.path.abspath(take_dir)
        self.session_dir = os.path.abspath(session_dir)
        self.name = os.path.basename(self.take_dir)
        cam_dir = os.path.join(self.take_dir, fmt.CAMERA_DIR)
        cam_index = np.load(os.path.join(cam_dir, fmt.CAMERA_INDEX_FILE))
        if cam_index.shape[0] == 0:
            raise ValueError(f"{self.name}: no camera frames")
        keep = core.make_strictly_increasing(cam_index[:, 0])
        self.cam_t_abs = cam_index[keep, 0].astype(np.float64)
        # JPEGs are written by APPEND POSITION (recorder.py: `for i, jpeg in
        # enumerate(buf.cam_jpegs): open(camera_frame_name(i))`), so the file index
        # is the positional row in cam_index, i.e. `keep` — NOT column 1, which is
        # the camera bridge's SEQUENCE number (CAMERA_INDEX_COLS=("t","seq"); seq is
        # non-zero-based and has gaps from dropped frames). This mirrors exactly what
        # convert_to_umi.build_episodes_from_take passes (keep[a:b]).
        self.cam_src_idx = keep.astype(int)  # append-position == JPEG file index
        self.t0 = float(self.cam_t_abs[0])
        self.cam_t = self.cam_t_abs - self.t0
        self.K = int(self.cam_t.shape[0])
        self.cam_dir = cam_dir

        # markers (take-relative) for the episode summary readout.
        markers_abs = conv._load_markers(self.take_dir)
        self.markers = np.sort(markers_abs.astype(np.float64) - self.t0)
        self.segments = core.summarize_segments(self.cam_t, self.markers)

        # --- session-level tracking provenance (LEFT-canonical) ---------------
        # tracker.npy is the PRIMARY hand: 'left' for sessions recorded with a
        # 'mode' key, 'right' for legacy sessions (no mode). Secondary trackers
        # (tracker_right/tracker_head) exist for 2pt/3pt takes.
        sess_meta = conv._load_session_meta(session_dir)
        self.mode = sess_meta.get("mode")  # '1pt'|'2pt'|'3pt' or None (legacy)
        self.primary_role = sess_meta.get(
            "primary_role", "right" if self.mode is None else "left")

        # --- primary tracker (RAW world, wxyz) — sanitize + resample -----------
        # DEGLITCHED, matching the converter's PRIMARY path (stale-drop ->
        # strictly-increasing -> deglitch_tracker_positions), so the TCP path drawn
        # here is the exact eef the .zarr.zip stores.
        tracker = self._load_opt(fmt.TRACKER_FILE)  # degrade (no gripper) if absent
        self.has_tracker = tracker.ndim == 2 and tracker.shape[0] > 0
        if self.has_tracker:
            self.tr_pos_grid, self.tr_quat_grid = self._resample_tracker(tracker, deglitch=True)
            # ALSO keep the sanitized RAW-RATE samples (native ~80 Hz, NOT decimated
            # onto the ~15 Hz camera grid). take-relative time + world pos + wxyz. These
            # become the DISPLAY grid below: the overview path AND the moving cluster
            # (gripper/triads/camera pose/F/T) all ride them, so the replay follows the
            # real hand motion instead of stepping between RGB frames. tcp_pose_grid
            # (camera grid) is kept only for the converter-fidelity contract + charts.
            self.tr_raw_t, self.tr_raw_pos, self.tr_raw_quat = \
                self._sanitize_tracker_raw(tracker, deglitch=True)
        else:
            self.tr_pos_grid = np.zeros((self.K, 3))
            self.tr_quat_grid = np.tile([1.0, 0, 0, 0], (self.K, 1))
            self.tr_raw_t = np.zeros((0,))
            self.tr_raw_pos = np.zeros((0, 3))
            self.tr_raw_quat = np.zeros((0, 4))

        # --- DISPLAY grid: the tracker's NATIVE rate (~80 Hz), the PLAYBACK master --
        # The camera grid (~15 Hz) is the converter's clock, but decimating the MOTION
        # to it makes the replayed hand STEP between RGB frames (the "不够 smooth" the
        # operator saw). The tracker itself samples ~5x faster, so we play the moving
        # cluster + F/T + gripper on the tracker's own timestamps and just HOLD the RGB
        # (which only exists at ~15 Hz) across the frames it spans. Take-relative,
        # strictly increasing, already clipped to the camera span by _sanitize_tracker_raw
        # (so the scrub window still matches the recorded RGB). Falls back to the camera
        # grid when a take has no usable tracker (nothing denser to show).
        if self.tr_raw_t.shape[0] >= 2:
            self.disp_t = self.tr_raw_t
            # The display-grid tracker pose IS the native raw stream (resampling a stream
            # onto its own timestamps is identity), so the moving cluster uses it directly.
            self.tr_pos_disp, self.tr_quat_disp = self.tr_raw_pos, self.tr_raw_quat
            self.disp_is_raw = True
        else:
            self.disp_t = self.cam_t
            self.tr_pos_disp, self.tr_quat_disp = self.tr_pos_grid, self.tr_quat_grid
            self.disp_is_raw = False
        self.disp_t_abs = self.disp_t + self.t0
        self.D = int(self.disp_t.shape[0])
        self.disp_hz = self.D / max(self.disp_t[-1] - self.disp_t[0], 1e-6) if self.D > 1 else 0.0
        # Which camera frame each display frame shows: the most-recent RGB at/<= its time
        # (a zero-order hold — the image can't be interpolated). Clamped into [0, K-1].
        self.disp_cam_idx = np.clip(
            np.searchsorted(self.cam_t, self.disp_t, side="right") - 1, 0, self.K - 1).astype(int)
        # Episode boundaries ON THE DISPLAY GRID (for the readout's per-frame episode
        # label). ``segments`` (camera grid) stays the honest CONVERTER frame count shown
        # in the GUI; this is the same marker split re-indexed onto the denser grid.
        self.disp_segments = core.summarize_segments(self.disp_t, self.markers)

        # --- secondary trackers (2pt right / 3pt head), for triad display ------
        # Keyed by ROLE. The primary hand's role is self.primary_role; the extra
        # files carry the other roles. Each entry is (pos_grid[K,3], quat_grid[K,4]).
        # DEGLITCH, same as the primary: the converter's secondary path
        # (``convert_to_umi._sanitize_tracker``) runs ``deglitch_tracker_positions`` too
        # (right/head trackers occlude as well), so skipping it here would draw a path the
        # zarr does not contain. This branch used to pass ``deglitch=False`` on the strength
        # of a comment that said the converter didn't deglitch secondaries — stale since the
        # converter gained it. Measured divergence on the one 3pt session on disk
        # (session_2026.07.25_13.14.27): take_073/head 7.75 cm on the camera grid. It is
        # data-dependent (a bridged excursion outside the camera span shows 0), which is why
        # eyeballing most takes hid it. Resampled onto the DISPLAY grid (same as the primary
        # cluster) so the extra triads move as smoothly as the main hand.
        self.sec_trackers: dict[str, tuple] = {}
        # Every SECONDARY role fmt declares, so the two 5pt feet appear with no edit here
        # (a viewer that silently omits a recorded keypoint is how a mislabelled foot ships).
        for role, filename in ((r, f) for r, f in fmt.TRACKER_FILE_BY_ROLE.items()
                               if r != "left"):
            arr = self._load_opt(filename)
            if arr.ndim == 2 and arr.shape[0] > 0:
                self.sec_trackers[role] = self._resample_tracker(
                    arr, deglitch=True, query_t_abs=self.disp_t_abs)

        # --- RIGHT-hand per-hand MODALITIES (camera / F-T / gripper) -----------
        # Independent of tracker_right (a puck needs no hardware, these do), so a 3pt
        # take recorded before the right camera/Bota/gripper were wired up has the right
        # TRIAD and none of this. Each stays None when absent — nothing is zero-filled,
        # so the drawing can never imply a measurement that was not taken.
        self.cam_right_dir = os.path.join(self.take_dir, fmt.CAMERA_RIGHT_DIR)
        cam_right_idx_p = os.path.join(self.cam_right_dir, fmt.CAMERA_INDEX_FILE)
        self.cam_right_src_idx = None
        self.disp_cam_right_idx = None
        if os.path.exists(cam_right_idx_p):
            cri = np.asarray(np.load(cam_right_idx_p), dtype=np.float64)
            if cri.ndim == 2 and cri.shape[0] > 0:
                keep_r = core.make_strictly_increasing(cri[:, 0])
                self.cam_right_t_abs = cri[keep_r, 0].astype(np.float64)
                self.cam_right_src_idx = keep_r.astype(int)  # append position == file index
                # SAME causal hold the converter uses for camera1_rgb (see
                # core.resample_frame_index), so what you see here is what the .zarr.zip
                # paired — a fidelity checker that used a different rule would hide
                # exactly the pairing error it exists to catch. Indices are into
                # cam_right_src_idx, which maps to the JPEG file index.
                sel, self.cam_right_skew = core.resample_frame_index(
                    self.cam_right_t_abs, self.disp_t_abs, causal=True)
                self.disp_cam_right_idx = self.cam_right_src_idx[sel]
        self.has_cam_right = self.cam_right_src_idx is not None

        # RIGHT gripper: its OWN homed endpoints (cols 3/4 of its own stream), because it
        # is a second DM4310 whose absolute encoder re-references on its own power-cycle.
        gr = self._load_opt(fmt.GRIPPER_RIGHT_FILE)
        self.frac_right = None
        if gr.ndim == 2 and gr.shape[0] > 0:
            gr = gr[core.make_strictly_increasing(gr[:, 0])]
            calib_r, _srcr = core.effective_gripper_calib(gr, conv.load_gripper_calib(self.session_dir))
            if calib_r is not None:
                wr = np.asarray(calib_r.to_width_m(gr[:, 1]), dtype=np.float64)
                wr_disp = core.resample_gripper_width(gr[:, 0], wr, self.disp_t_abs).reshape(-1)
                self.width_disp_right = wr_disp
                self.frac_right = np.clip(wr_disp / max(float(calib_r.width_open_m), 1e-6), 0.0, 1.0)
        self.has_gripper_right = self.frac_right is not None

        # --- tare events (R5): wall-clock times, take-relative for the readout -
        tare = self._load_opt(fmt.TARE_EVENTS_FILE)
        if tare.ndim == 2 and tare.shape[0] > 0:
            self.tare_times = np.sort(tare[:, 0].astype(np.float64) - self.t0)
        else:
            self.tare_times = np.zeros((0,), dtype=np.float64)

        # --- gripper: width (m) + open fraction. width_grid stays on the CAMERA grid
        # (the chart shows the converter's view); width_disp is on the DISPLAY grid so
        # the jaws LERP as smoothly as the moving cluster. ---
        gripper = self._load_opt(fmt.GRIPPER_FILE)
        base_calib = conv.load_gripper_calib(self.session_dir)
        calib, self.calib_src = core.effective_gripper_calib(gripper, base_calib)
        self.width_open_m = float(calib.width_open_m) if calib is not None else 0.095
        if gripper.shape[0] > 0 and calib is not None:
            gripper = gripper[core.make_strictly_increasing(gripper[:, 0])]
            w = np.asarray(calib.to_width_m(gripper[:, 1]), dtype=np.float64)
            self.width_grid = core.resample_gripper_width(gripper[:, 0], w, self.cam_t_abs).reshape(-1)
            self.width_disp = core.resample_gripper_width(gripper[:, 0], w, self.disp_t_abs).reshape(-1)
            self.has_gripper = True
        else:
            self.width_grid = np.zeros(self.K)
            self.width_disp = np.zeros(self.D)
            self.has_gripper = False
        # open fraction in [0,1] for the mesh jaw LERP (clamped), on the display grid.
        wo = max(self.width_open_m, 1e-6)
        self.frac_grid = np.clip(self.width_disp / wo, 0.0, 1.0)

        # --- F/T wrench (anti-aliased onto the cam grid, same as the converter) ---
        # Load BOTH traces the take recorded so the browser can switch between them
        # live (the user's ask), the same way the live viz's "Compensated wrench"
        # checkbox does: ``ft_wrench.npy`` is ALWAYS the RAW sensor wrench, and
        # ``ft_wrench_comp.npy`` is the gravity/inertia-COMPENSATED trace written
        # only when a Bota payload config was active (take_meta.ft_compensated).
        # Same sensor frame/sign either way, so the arrows rotate identically. Each
        # grid is None when that trace is absent; ``has_ft`` is true if either is.
        # Each trace is resampled onto BOTH grids: ``*_grid_*`` on the camera grid feeds
        # the scrolling charts (the converter's view), ``*_disp_*`` on the display grid
        # feeds the per-frame arrows + |F|/|T| readouts so they move at the native rate.
        meta = conv._load_take_meta(self.take_dir)
        self.wrench_grid_raw = self._resample_wrench_file(fmt.FT_FILE, self.cam_t_abs)
        self.wrench_disp_raw = self._resample_wrench_file(fmt.FT_FILE, self.disp_t_abs)
        comp_recorded = meta.get("ft_compensated") is True
        self.wrench_grid_comp = (
            self._resample_wrench_file(fmt.FT_COMP_FILE, self.cam_t_abs) if comp_recorded else None)
        self.wrench_disp_comp = (
            self._resample_wrench_file(fmt.FT_COMP_FILE, self.disp_t_abs) if comp_recorded else None)
        self.has_ft = (self.wrench_grid_raw is not None
                       or self.wrench_grid_comp is not None)
        self.has_comp = self.wrench_grid_comp is not None
        # Default to the compensated trace when present (matches the live viz default
        # and what you usually want to see); fall back to raw, then to zeros.
        self.wrench_default_comp = self.wrench_grid_comp is not None
        if not self.has_ft:
            # Keep render code branch-free: a zero grid + a "none" label on both grids.
            self.wrench_grid_raw = np.zeros((self.K, 6), dtype=np.float32)
            self.wrench_disp_raw = np.zeros((self.D, 6), dtype=np.float32)

        # RIGHT-hand wrench, resolved from that hand's OWN provenance
        # (take_meta["hands"]["right"]["ft_compensated"]) — the two Bota sensors have
        # independent payload configs and tares, so reading the left flag here is how an
        # uncompensated right trace would be drawn as if it were compensated. Only the
        # DISPLAY grid is needed (arrows + readout; the charts stay the primary hand's,
        # which is the converter's view of the action label).
        right_meta = (meta.get("hands") or {}).get("right") or {}
        self.wrench_disp_right = None
        self.wrench_right_source = None
        if right_meta.get("ft_compensated") is True:
            self.wrench_disp_right = self._resample_wrench_file(fmt.FT_COMP_RIGHT_FILE, self.disp_t_abs)
            self.wrench_right_source = "compensated" if self.wrench_disp_right is not None else None
        if self.wrench_disp_right is None:
            self.wrench_disp_right = self._resample_wrench_file(fmt.FT_RIGHT_FILE, self.disp_t_abs)
            if self.wrench_disp_right is not None:
                self.wrench_right_source = "raw"
        self.has_ft_right = self.wrench_disp_right is not None
        # The SAME right-hand trace on the CAMERA grid, for that hand's own Force/Torque
        # charts (the converter's view of the right F/T, like the left charts are of the
        # left). Resolved from the SAME provenance choice as the display grid above, so the
        # right arrows, readout and charts can never show two different traces.
        self.wrench_grid_right = None
        if self.has_ft_right:
            self.wrench_grid_right = self._resample_wrench_file(
                fmt.FT_COMP_RIGHT_FILE if self.wrench_right_source == "compensated" else fmt.FT_RIGHT_FILE,
                self.cam_t_abs)

    def tcp_pose_grid(self, T_tracker_tcp: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Primary puck grid -> TCP (eef) pose grid via the converter's own extrinsic.

        Composes ``world_T_tcp = world_T_puck @ T_tracker_tcp`` per frame with the
        SAME helper ``convert_to_umi`` uses (``core.apply_tcp_extrinsic``), so the
        TCP path drawn here IS the ``robot0_eef_pos`` the ``.zarr.zip`` stores.
        Returns (pos[K,3], wxyz[K,4])."""
        from scipy.spatial.transform import Rotation
        rot = Rotation.from_quat(self.tr_quat_grid[:, [1, 2, 3, 0]])  # wxyz -> xyzw
        tcp_pos, tcp_rot = core.apply_tcp_extrinsic(self.tr_pos_grid, rot, T_tracker_tcp)
        tcp_wxyz = np.atleast_2d(tcp_rot.as_quat())[:, [3, 0, 1, 2]]  # xyzw -> wxyz
        return tcp_pos, tcp_wxyz

    def tcp_pose_raw(self, T_tracker_tcp: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """RAW-RATE (native ~80 Hz, NOT camera-grid) TCP pose, via the same extrinsic.

        Same compose as :meth:`tcp_pose_grid` but from the sanitized full-rate tracker
        samples, so the overview trajectory polyline keeps the hand detail the ~15 Hz
        camera-grid resample smooths away. Returns (pos[N,3], wxyz[N,4]); empty if the
        take has no tracker."""
        if self.tr_raw_pos.shape[0] == 0:
            return np.zeros((0, 3)), np.zeros((0, 4))
        from scipy.spatial.transform import Rotation
        rot = Rotation.from_quat(self.tr_raw_quat[:, [1, 2, 3, 0]])  # wxyz -> xyzw
        tcp_pos, tcp_rot = core.apply_tcp_extrinsic(self.tr_raw_pos, rot, T_tracker_tcp)
        tcp_wxyz = np.atleast_2d(tcp_rot.as_quat())[:, [3, 0, 1, 2]]  # xyzw -> wxyz
        return tcp_pos, tcp_wxyz

    def tcp_pose_disp(self, T_tracker_tcp: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """TCP (eef) pose on the DISPLAY grid (native ~80 Hz when a tracker exists, else
        the camera grid) — what the moving cluster rides so playback is smooth. Same
        extrinsic compose as :meth:`tcp_pose_grid`; equals it wherever the two grids
        share a timestamp. Returns (pos[D,3], wxyz[D,4])."""
        from scipy.spatial.transform import Rotation
        rot = Rotation.from_quat(self.tr_quat_disp[:, [1, 2, 3, 0]])  # wxyz -> xyzw
        tcp_pos, tcp_rot = core.apply_tcp_extrinsic(self.tr_pos_disp, rot, T_tracker_tcp)
        tcp_wxyz = np.atleast_2d(tcp_rot.as_quat())[:, [3, 0, 1, 2]]  # xyzw -> wxyz
        return tcp_pos, tcp_wxyz

    def sec_pose_disp(self, role: str, T_tracker_tcp: np.ndarray,
                      T_tracker_head: "np.ndarray | None" = None
                      ) -> tuple[np.ndarray, np.ndarray]:
        """A SECONDARY keypoint's display pose, with the extrinsic applied per the
        converter's rule: hands get the puck->TCP transform, the chest gets its OWN
        rotation-only body relabel.

        ``sec_trackers`` holds the RAW puck pose, which is what neither keypoint stores:

          * the RIGHT hand is treated exactly like robot0 ("same TCP extrinsic as robot0
            (FK-symmetric wrists)"). Drawing that cluster on the raw puck put it
            **0.2305 m and 180 deg** off the pose stored in ``robot1_eef_pos`` — the two
            hands rendered in different frames, and the right gripper meshes hung on a
            puck frame while the left's hung on a wrist frame.
          * the CHEST has no TCP and no lever, but its puck IS bolted on rotated, so
            ``convert_to_umi`` composes a ROTATION-ONLY chest relabel into
            ``head_rot_axis_angle`` (``load_head_extrinsic``). Position is unaffected, so
            skipping it draws the chest triad's AXES in the raw-puck convention while the
            zarr viewer (which reads the stored key) draws the body frame — the two
            viewers disagreeing on the one keypoint neither can cross-check.

        ``T_tracker_head=None`` reproduces a session with no ``head_extrinsic.json``. The
        two 5pt FEET currently take that SAME matrix, which is the hardware claim
        ``collection_viz._BODY_RELABEL_M`` records (the ankle pucks are mounted like the
        chest one). When a session starts carrying its own
        ``calibration/foot_extrinsic.json`` (``fmt.FOOT_EXTRINSIC_FILE``), this is the one
        place to read it per role.
        Returns (pos[D,3], wxyz[D,4]) on the same DISPLAY grid as the primary cluster."""
        pos, wxyz = self.sec_trackers[role]
        # A BODY REFERENCE (chest, or a 5pt foot) gets the rotation-only body relabel; a
        # HAND gets the puck->TCP transform. Branching on the ROLE KIND, not on the string
        # "head", is what keeps a foot from being handed the 0.242 m TCP lever — which
        # would draw it 24 cm out with nothing to notice it by.
        extr = T_tracker_head if role in fmt.BODY_ROLES else T_tracker_tcp
        if extr is None:
            return pos, wxyz
        from scipy.spatial.transform import Rotation
        rot = Rotation.from_quat(np.asarray(wxyz)[:, [1, 2, 3, 0]])  # wxyz -> xyzw
        tcp_pos, tcp_rot = core.apply_tcp_extrinsic(np.asarray(pos), rot, extr)
        return tcp_pos, np.atleast_2d(tcp_rot.as_quat())[:, [3, 0, 1, 2]]  # xyzw -> wxyz

    def wrench_grid(self, compensated: bool) -> np.ndarray:
        """Return the active CAMERA-grid wrench (for the charts), falling back to
        whichever trace exists (so a degenerate take never crashes the render)."""
        if compensated and self.wrench_grid_comp is not None:
            return self.wrench_grid_comp
        if not compensated and self.wrench_grid_raw is not None:
            return self.wrench_grid_raw
        # Requested mode absent: return the other one if it exists, else zeros.
        if self.wrench_grid_comp is not None:
            return self.wrench_grid_comp
        if self.wrench_grid_raw is not None:
            return self.wrench_grid_raw
        return np.zeros((self.K, 6), dtype=np.float32)

    def wrench_disp(self, compensated: bool) -> np.ndarray:
        """Return the active DISPLAY-grid wrench (for the per-frame arrows + |F|/|T|
        readouts), same fallback order as :meth:`wrench_grid`."""
        if compensated and self.wrench_disp_comp is not None:
            return self.wrench_disp_comp
        if not compensated and self.wrench_disp_raw is not None:
            return self.wrench_disp_raw
        if self.wrench_disp_comp is not None:
            return self.wrench_disp_comp
        if self.wrench_disp_raw is not None:
            return self.wrench_disp_raw
        return np.zeros((self.D, 6), dtype=np.float32)

    def _resample_tracker(self, tracker: np.ndarray, deglitch: bool = False,
                          query_t_abs: "np.ndarray | None" = None) -> tuple:
        """Sanitize + resample one tracker array onto a query grid, mirroring the
        converter's hygiene so the drawn pose == the stored one. Returns
        (pos[Q,3], quat[Q,4] wxyz) on ``query_t_abs`` (defaults to the camera grid).

        Order (exactly as convert_to_umi): drop stale-repeat rows (col 'stale'==1) ->
        make_strictly_increasing -> (PRIMARY only) deglitch_tracker_positions, which
        bridges isolated lighthouse teleports before Slerp/interp1d. ``deglitch``
        gates that last step: True for the primary hand (matches the eef label),
        False for secondary keypoints (matches _sanitize_tracker). ``query_t_abs``
        picks the output grid: the camera grid for the fidelity-pinned eef, the display
        grid for the smooth moving cluster (both are exact resamples of the same clean
        stream, so a display sample landing on a camera time equals the grid value)."""
        if query_t_abs is None:
            query_t_abs = self.cam_t_abs
        stale_col = fmt.TRACKER_COLS.index("stale")
        live = tracker[:, stale_col] < 0.5
        if live.any():
            tracker = tracker[live]
        tracker = tracker[core.make_strictly_increasing(tracker[:, 0])]
        if deglitch and tracker.shape[0] >= 3:
            keep_tr = core.deglitch_tracker_positions(tracker[:, 0], tracker[:, 2:5])
            tracker = tracker[keep_tr]
        t = tracker[:, 0]
        pos_grid = core.resample_position(t, tracker[:, 2:5], query_t_abs)
        rot = core.resample_rotation_wxyz(t, tracker[:, 5:9], query_t_abs)
        quat_grid = rot.as_quat()[:, [3, 0, 1, 2]]  # xyzw -> wxyz
        return pos_grid, quat_grid

    def _sanitize_tracker_raw(self, tracker: np.ndarray, deglitch: bool = False) -> tuple:
        """Same sanitize as _resample_tracker (stale-drop -> strictly-increasing ->
        optional deglitch) but WITHOUT the camera-grid resample — returns the tracker's
        NATIVE-rate samples so the overview path keeps its full fidelity. Clipped to the
        take's camera time span so the drawn path matches the scrubbable timeline (the
        tracker stream can start slightly before / end slightly after the first/last RGB
        frame). Returns (t_rel[N], pos[N,3], quat_wxyz[N,4]) in take-relative seconds."""
        stale_col = fmt.TRACKER_COLS.index("stale")
        live = tracker[:, stale_col] < 0.5
        if live.any():
            tracker = tracker[live]
        tracker = tracker[core.make_strictly_increasing(tracker[:, 0])]
        if deglitch and tracker.shape[0] >= 3:
            keep_tr = core.deglitch_tracker_positions(tracker[:, 0], tracker[:, 2:5])
            tracker = tracker[keep_tr]
        t_abs = tracker[:, 0]
        # Keep only the samples inside the camera span [t0, cam_t_abs[-1]] so the path
        # covers exactly the replayable window (a small margin lets the endpoints reach
        # the first/last camera frame instead of stopping short).
        lo, hi = self.cam_t_abs[0], self.cam_t_abs[-1]
        keep = (t_abs >= lo) & (t_abs <= hi)
        if keep.sum() >= 2:
            tracker = tracker[keep]
        pos = np.asarray(tracker[:, 2:5], dtype=np.float64)
        quat = np.asarray(tracker[:, 5:9], dtype=np.float64)  # wxyz, as stored
        return tracker[:, 0] - self.t0, pos, quat

    def _resample_wrench_file(self, filename: str, grid_t_abs: np.ndarray) -> "np.ndarray | None":
        """Load + anti-alias one wrench file onto ``grid_t_abs`` (None if absent or
        empty). Same core helper the converter uses, so a camera-grid resample here ==
        conversion; the display grid gets the same interval-mean anti-alias at ~80 Hz."""
        ft = self._load_opt(filename)
        if ft.ndim != 2 or ft.shape[0] == 0:
            return None
        ft = ft[core.make_strictly_increasing(ft[:, 0])]
        return core.antialias_resample_wrench(ft[:, 0], ft[:, 1:7], grid_t_abs)

    def _load_opt(self, filename: str) -> np.ndarray:
        p = os.path.join(self.take_dir, filename)
        if not os.path.exists(p):
            return np.zeros((0, 0), dtype=np.float64)
        return np.asarray(np.load(p), dtype=np.float64)

    def jpeg_path(self, i: int) -> str:
        return os.path.join(self.cam_dir, fmt.camera_frame_name(int(self.cam_src_idx[i])))

    def jpeg_path_right(self, i: int) -> "str | None":
        """RIGHT-hand JPEG for DISPLAY-grid frame ``i``, or None when there is no right
        camera. Same append-position rule as the left (``camera_frame_name(row)``)."""
        if self.cam_right_src_idx is None:
            return None
        row = int(self.disp_cam_right_idx[i])
        return os.path.join(self.cam_right_dir, fmt.camera_frame_name(row))


# ---------------------------------------------------------------------------
def wrist_pose_from_stored(T_tracker_tcp: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """WRIST pose relative to the STORED eef (TCP) frame: (offset[3], quat_wxyz[4]).

    IDENTICAL to ``replay_zarr_viz.wrist_pose_from_stored`` (kept in sync so the raw and
    zarr tools hang the same cluster). The converter stored
    ``world_T_eef = world_T_puck @ T_tracker_tcp``, so the eef frame relates to the wrist
    frame through the extrinsic's ROTATION as well as its translation:

        eef_T_wrist = (puck_T_eef)^-1 @ puck_T_wrist
                    = T_tracker_tcp^-1 @ [I | _TRACKER_TRUE_XYZ_M_in_puck]

    The puck seat ``_TRACKER_TRUE_XYZ_M`` is defined in the WRIST frame, so it must be
    rotated into the puck frame by the extrinsic's rotation before it can be composed
    with a puck-frame translation. The OLD helper returned ``-(_TRACKER_TRUE_XYZ_M + t)``,
    i.e. it ADDED a puck-frame translation to a wrist-frame vector — valid ONLY while the
    extrinsic rotation was identity (puck axes == wrist axes). With the measured
    ``-y,-x,-z`` puck->wrist relabel that assumption is false and the cluster landed
    ~33 cm off; the same conflation is what made the old pure-translation extrinsic
    ``(0.225,0,-0.05)`` (a WRIST-frame lever) wrong when applied in the PUCK frame.

    Deriving the full 4x4 inverse here is exact for ANY extrinsic (rotating or not), so
    the TCP triad lands on the trajectory and the "tracker" triad returns to the physical
    puck — verified to machine precision against the raw puck in test_replay_logic."""
    T = np.asarray(T_tracker_tcp, dtype=np.float64)
    R_ext, t_ext = T[:3, :3], T[:3, 3]
    # puck_T_wrist: the wrist origin/axes expressed in the puck frame. The puck sits at
    # _TRACKER_TRUE_XYZ_M in WRIST coords, so wrist-from-puck = -R_ext @ that vector
    # (R_ext is exactly the puck->wrist axis relabel for this rig's extrinsic).
    puck_R_wrist = R_ext
    puck_t_wrist = -(R_ext @ np.asarray(_TRACKER_TRUE_XYZ_M, dtype=np.float64))
    # eef_T_wrist = inv(puck_T_eef) @ puck_T_wrist
    eef_R_wrist = R_ext.T @ puck_R_wrist
    eef_t_wrist = R_ext.T @ (puck_t_wrist - t_ext)
    return eef_t_wrist, _gov._rotmat_to_wxyz(eef_R_wrist)


def puck_triad_local_quat(T_tracker_tcp: np.ndarray) -> np.ndarray:
    """Local (wrist-relative) ROTATION for the "tracker" (puck) triad — wxyz.

    The puck triad is a CHILD of the wrist frame, so without an explicit rotation it
    inherits the WRIST axes — which is wrong, because the whole point of the extrinsic is
    that the puck axes differ from the wrist axes (by ``R_ext``, the measured -y,-x,-z
    relabel). Drawing it unrotated made the "tracker" triad identical to the TCP triad,
    a 180deg error the operator caught by eye.

    The wrist frame satisfies ``R_wrist_world = R_puck_world @ R_ext``, so to render the
    triad back at the raw puck orientation the local rotation must be ``R_ext^-1``::

        R_wrist_world @ R_ext^-1 = R_puck_world

    Identity for a legacy pure-translation extrinsic, so old datasets are unaffected.
    Same value as ``collection_viz._puck_child_quat`` (the live view, both paths)."""
    R_ext = np.asarray(T_tracker_tcp, dtype=np.float64)[:3, :3]
    return _gov._rotmat_to_wxyz(R_ext.T)


def fingertip_grasp_point_wrist(geom) -> np.ndarray:
    """Wrist-frame grasp point = midpoint between the two CLOSED fingertips, computed
    from the actual URDF finger meshes.

    The operator wants the TCP marker + trajectory drawn at the point the gripper
    actually grasps — the gap between the fingertips — NOT at the calibrated eef
    (wrist +x 0.21 m), which after the +30 mm URDF gripper swap renders ~4.5 cm behind
    the tips (mid-finger). Deriving it from the mesh (not a hard-coded 0.255) keeps it
    correct across URDF changes: the tips moved 0.226 -> 0.255 once already. The grasp
    CENTRE is jaw-independent (the fingers open symmetrically about it), so the closed
    pose is representative. Returns an (x, y, z) point in the wrist-yaw frame."""
    from scipy.spatial.transform import Rotation
    tips = []
    for pc in (geom.finger1, geom.finger2):
        R = Rotation.from_quat(np.asarray(pc.quat, dtype=np.float64)[[1, 2, 3, 0]]).as_matrix()
        v = (np.asarray(pc.mesh.vertices, dtype=np.float64) @ R.T) + np.asarray(pc.pos_closed)
        tips.append(v[v[:, 0] > v[:, 0].max() - 0.01].mean(axis=0))  # verts within 1cm of the tip
    return np.mean(tips, axis=0)


def _rebase(pos: np.ndarray, wxyz: np.ndarray, home_p: np.ndarray, home_q: np.ndarray,
            on: bool, reach_scale: float = 1.0,
            ref: "tuple | None" = None) -> tuple[np.ndarray, np.ndarray]:
    """Home-anchored display poses (or the raw poses when ``on`` is False).

    ON re-bases every frame onto an upright home via ``gripper_only_view.home_anchor``
    (the SAME math the collection viz + replay_zarr_viz use), ref = the take's FIRST
    frame:  R_disp = (R_cur·R_ref⁻¹)·R_home ,  p_disp = p_home + reach·(p_cur − p_ref).
    So the whole trajectory rides an upright, z-up home like collection — display
    only, the underlying arrays and the numeric readouts are untouched.

    ``ref`` overrides the reference pose (else frame 0 of ``pos``/``wxyz``). The
    RAW-RATE path passes the CAMERA-GRID frame-0 pose here so it anchors to the SAME
    reference the moving gripper does and overlays it exactly (its own first sample is
    a hair earlier in time, so anchoring to its own frame 0 would offset the line)."""
    pos = np.asarray(pos, dtype=np.float64)
    wxyz = np.asarray(wxyz, dtype=np.float64)
    if not on or pos.shape[0] == 0:
        return pos, wxyz
    ref_p, ref_q = ref if ref is not None else (pos[0], wxyz[0])
    dp = np.empty_like(pos)
    dq = np.empty_like(wxyz)
    for k in range(pos.shape[0]):
        dp[k], dq[k] = _gov.home_anchor(home_p, home_q, ref_p, ref_q, pos[k], wxyz[k],
                                        reach_scale=reach_scale)
    return dp, dq


def list_takes(session_dir: str) -> list[str]:
    """Every committed ``take_*`` dir under a session, natural-sorted.

    The prev/next buttons page through this list. Filters the same way ``find_take``
    does (a ``take_meta.json`` commit marker, no ``.tmp``); ``os.path.isdir`` follows
    symlinks, so a MERGED session (whose ``take_*`` are symlinks into the per-sitting
    sessions) enumerates correctly. Names are zero-padded to 3 digits, so a lexical
    sort is already the numeric order."""
    session_dir = os.path.abspath(session_dir)
    return sorted(
        os.path.join(session_dir, x) for x in os.listdir(session_dir)
        if os.path.isdir(os.path.join(session_dir, x)) and x.startswith("take_")
        and not x.endswith(".tmp")
        and os.path.exists(os.path.join(session_dir, x, fmt.TAKE_META_FILE)))


def find_take(path: str) -> tuple[str, str]:
    """Resolve a session dir (-> its first committed take) or a take dir directly."""
    path = os.path.abspath(path)
    base = os.path.basename(path)
    if base.startswith("take_") and os.path.exists(os.path.join(path, fmt.TAKE_META_FILE)):
        return os.path.dirname(path), path
    takes = list_takes(path)
    if not takes:
        raise SystemExit(f"[replay] no committed take_* dirs in {path}")
    return path, takes[0]


def take_summary(take_dir: str) -> tuple[int, float]:
    """``(camera_frames, duration_s)`` of a take, read from its ``take_meta.json``.

    Cheap on purpose (a ~2 kB json per take, not a ``RawTake`` load) so the Take
    dropdown can label all 226 takes of a merged session with their frame count at
    startup. Falls back to counting the camera JPEGs, then to ``(0, 0.0)`` — a take
    whose meta is unreadable still gets a selectable dropdown entry."""
    meta_path = os.path.join(take_dir, fmt.TAKE_META_FILE)
    try:
        with open(meta_path) as f:
            meta = json.load(f)
        frames = int(meta.get("counts", {}).get("camera", 0) or 0)
        dur = float(meta.get("stop_wall", 0.0) or 0.0) - float(meta.get("start_wall", 0.0) or 0.0)
    except Exception:  # noqa: BLE001 — a bad meta must not break the take list
        frames, dur = 0, 0.0
    if frames <= 0:
        cam_dir = os.path.join(take_dir, fmt.CAMERA_DIR)
        try:
            frames = sum(1 for e in os.scandir(cam_dir) if e.name.endswith(".jpg"))
        except OSError:
            frames = 0
    return frames, max(dur, 0.0)


def take_labels(take_dirs: list[str]) -> list[str]:
    """Dropdown labels: ``take_007  (687 fr, 25s)`` — the raw-take twin of the zarr
    viewer's ``ep 3  (412 fr)`` episode dropdown, so both tools read alike. Take names
    are unique, so the labels are too (the callback maps label -> index by position)."""
    out = []
    for d in take_dirs:
        frames, dur = take_summary(d)
        out.append(f"{os.path.basename(d)}  ({frames} fr, {dur:.0f}s)")
    return out


def delete_take(take_dir: str) -> str:
    """DELETE a take from the session for real. Returns a one-line description.

    Irreversible — there is no undo and nothing is moved aside, which is what the
    operator asked for. Two cases, and the difference matters:

    * a plain committed take dir -> ``shutil.rmtree``: the recording is GONE.
    * a MERGED session's ``take_*`` -> ``os.unlink`` of the SYMLINK only. The bytes
      live in the per-sitting session the merge pointed at; removing that too would
      destroy data every OTHER merged session / already-converted dataset built from
      the same sitting also references. So the merged view loses the take (the
      converter's ``take_*`` glob no longer sees it, which is the point) and the
      original sitting keeps it.
    """
    take_dir = os.path.abspath(take_dir)
    name = os.path.basename(take_dir)
    if os.path.islink(take_dir):
        target = os.path.realpath(take_dir)
        os.unlink(take_dir)
        return (f"unlinked merged-session take {name} (the sitting's own copy at "
                f"{target} is untouched)")
    shutil.rmtree(take_dir)
    return f"rmtree'd {name} — the recording is permanently gone"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("path", help="a session dir (uses take_000) or a single take_NNN dir")
    p.add_argument("--take", default=None, help="take name to load (e.g. take_002) when given a session dir")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8083)
    p.add_argument("--yam-urdf", default=_DEFAULT_YAM)
    p.add_argument("--reach-scale", type=float, default=1.0,
                   help="scale the home-anchored position delta (display only; default 1.0)")
    p.add_argument("--raw-pose", action=argparse.BooleanOptionalAction, default=True,
                   help="show the RAW absolute world pose — the DEFAULT, matching "
                        "RAW_FRAME=1 collect_data.sh: no home-anchor, no re-levelling, the "
                        "recorded libsurvive frame as-is. --no-raw-pose re-enables the old "
                        "home-anchored display (a world-frame rotation the recording does not "
                        "contain; useful only for a comfortable eyeball, never for fidelity).")
    p.add_argument("--cam-offset", type=float, nargs=3, default=(0.10, 0.0, 0.04))
    p.add_argument("--cam-rpy", type=float, nargs=3, default=(4.7124, 1.5708, 0.0))
    p.add_argument("--cam-fov-deg", type=float, default=58.0)
    # Accepted for backward compatibility (there is no IK any more — gripper mode).
    p.add_argument("--no-ik", action="store_true", help=argparse.SUPPRESS)
    args = p.parse_args(argv if argv is not None else sys.argv[1:])
    if args.no_ik:
        print("[replay] note: --no-ik is a no-op now (this tool is gripper-only, no mink IK).")

    session_dir, first_take = find_take(args.path)
    if args.take is not None:
        first_take = os.path.join(session_dir, args.take)
    take_dirs = list_takes(session_dir) or [os.path.abspath(first_take)]
    try:
        take_index = take_dirs.index(os.path.abspath(first_take))
    except ValueError:
        take_index = 0
    # Labels for the Take dropdown, built ONCE here (a json read per take) and carried
    # across rebuilds — re-deriving them on every switch would re-stat 226 takes.
    labels = take_labels(take_dirs)

    # ONE viser server for the whole session; each take rebuilds the scene+GUI into it
    # (reset between takes) so the Take dropdown / ⏮ / ⏭ page through takes and 🗑 removes
    # one without a relaunch or a new browser URL. The blocking per-take render loop
    # returns when a nav button calls Timeline.stop(); we then act on the request HERE —
    # the delete included, off the render/callback thread (buttons raise flags; the main
    # thread mutates) — and rebuild for the new take.
    server = viser.ViserServer(host=args.host, port=args.port, label="UMI raw take replay")
    print(f"[replay] http://{args.host}:{args.port}  — {len(take_dirs)} take(s); "
          "Take dropdown / ⏮ Prev / ⏭ Next / 🗑 Delete in the 'Take' panel", flush=True)
    while 0 <= take_index < len(take_dirs):
        server.scene.reset()
        server.gui.reset()
        nav = view_take(server, args, session_dir, take_dirs[take_index], take_dirs,
                        take_index, labels)
        action = nav["action"]
        if action == "quit":
            break
        if action == "prev":
            take_index = max(0, take_index - 1)
        elif action == "next":
            take_index = min(len(take_dirs) - 1, take_index + 1)
        elif action == "goto":
            take_index = int(np.clip(int(nav["index"]), 0, len(take_dirs) - 1))
        elif action == "delete":
            gone = take_dirs[take_index]
            try:
                what = delete_take(gone)
            except OSError as e:
                print(f"[replay] DELETE FAILED for {os.path.basename(gone)}: {e!r}",
                      flush=True)
                continue  # keep the same take on screen; nothing was removed
            print(f"[replay] DELETED (irreversible): {what}", flush=True)
            del take_dirs[take_index]
            del labels[take_index]
            if not take_dirs:
                print("[replay] no takes left in this session — exiting.", flush=True)
                break
            take_index = min(take_index, len(take_dirs) - 1)
    return 0


def view_take(server, args, session_dir, take_dir, take_dirs, take_index, labels):
    """Build the scene + GUI for ONE take, run the blocking timeline, return a nav dict.

    Assumes the caller reset the server's scene + GUI (clean slate). Returns
    ``{"action": "prev"|"next"|"delete"|"quit"}`` — the ⏮/⏭/🗑 buttons only set that
    flag and stop the timeline; :func:`main` performs the file move + rebuild on the main
    thread (buttons raise flags; the render/main thread mutates)."""
    nav = {"action": "quit"}
    print(f"[replay] raw take: {os.path.relpath(take_dir)}  "
          f"(take {take_index + 1}/{len(take_dirs)})")
    print(f"[replay] DECISION-B: raw tracker quaternions are {fmt.QUATERNION_CONVENTION.upper()}")
    take = RawTake(take_dir, session_dir)
    primary_role = take.primary_role
    print(f"[replay] {take.K} camera frames over {take.cam_t[-1]:.1f}s, "
          f"{len(take.markers)} markers -> {len(take.segments)} episode(s)")
    print(f"[replay] playback grid: {take.D} frames @ ~{take.disp_hz:.0f} Hz "
          f"({'native tracker' if take.disp_is_raw else 'camera grid — no tracker'}), "
          f"RGB held across ~{take.D/max(take.K,1):.1f} display frames each")
    ft_avail = ("comp+raw" if (take.has_comp and take.wrench_grid_raw is not None)
                else "comp" if take.has_comp
                else "raw" if take.has_ft else "none")
    print(f"[replay] gripper calib source: {take.calib_src}; "
          f"ft={ft_avail}, tracker={'yes' if take.has_tracker else 'no'}, "
          f"mode={take.mode or 'legacy'} (primary={primary_role})")

    # --- tracker->TCP extrinsic (the SAME the converter uses) -> the TCP (eef) path.
    T_ext, tcp_calibrated = conv.load_tcp_extrinsic(session_dir)
    if not np.allclose(T_ext[:3, :3], np.eye(3), atol=1e-6):
        # Expected for the CURRENT extrinsic: the puck body axes differ from the wrist
        # axes by the measured -y,-x,-z relabel. wrist_pose_from_stored handles the
        # rotation exactly (full 4x4 inverse), so the cluster is correct, not "slightly off".
        print("[replay] extrinsic has a rotation (puck axes != wrist axes) — handled "
              "exactly by the wrist-pose inverse; TCP axes follow the remapped extrinsic.")
    if not tcp_calibrated:
        print("[replay] NOTE: tcp extrinsic is UNCALIBRATED (identity) — the TCP path "
              "equals the raw puck path until a calibration exists.")
    # CHEST body relabel (3pt only), rotation-only: the same file+compose the converter
    # applies to head_rot_axis_angle, so the drawn chest AXES are the stored keypoint's.
    T_head, head_calibrated = conv.load_head_extrinsic(session_dir)
    if not head_calibrated:
        print("[replay] NOTE: no head_extrinsic.json — the chest triad draws the RAW "
              "chest-puck axes (this session's stored head_rot_axis_angle convention).")
    # The DISPLAY-grid TCP pose (native ~80 Hz when a tracker exists) drives BOTH the
    # moving cluster AND the green overview path — same dense grid, so the two overlay by
    # construction and the replayed hand follows the real motion instead of stepping
    # between RGB frames. tcp_pose_grid (camera grid) is used only for the honest
    # converter-view charts + the fidelity test; the numeric readout uses this grid's
    # raw-world pose (an actual sample, not a decimation).
    tcp_pos_disp, tcp_wxyz_disp = take.tcp_pose_disp(T_ext)

    # The gripper cluster hangs off the wrist. The converter stored world_T_eef =
    # world_T_puck @ T_ext, so the wrist pose relative to the stored eef is the full 4x4
    # inverse (rotation included — the extrinsic relabels puck->wrist axes). Identical to
    # replay_zarr_viz.wrist_pose_from_stored, so the TCP triad lands ON the trajectory and
    # the "tracker" triad lands back on the physical puck.
    wrist_off, wrist_q = wrist_pose_from_stored(T_ext)
    tri_trk = np.asarray(_TRACKER_TRUE_XYZ_M)
    tri_wrist = np.asarray(_WRIST_YAW_XYZ_M)
    home_p, _ = _HOME_BY_ROLE.get(primary_role, _HOME_BY_ROLE["left"])

    # Gripper geometry (meshes + wrist-relative endpoint poses), loaded once. Built here
    # (before the scene) so the fingertip grasp point can be derived from it.
    grip_geom = _gov.build_gripper_geom(primary_role, urdf_path=args.yam_urdf)

    # TCP triad + trajectory land at the FINGERTIP GRASP POINT (the gap between the
    # closed fingertips), per the operator: the calibrated eef (wrist +x 0.21 m) renders
    # ~4.5 cm behind the tips after the +30 mm URDF gripper swap, which reads as "TCP not
    # between the fingers". ``grasp_wrist`` is measured from the mesh (auto-tracks a URDF
    # change); ``grasp_off_eef`` is that point expressed in the /eef frame, i.e. the rigid
    # forward offset applied to BOTH the TCP triad (a wrist child at grasp_wrist) and the
    # green trajectory line (each eef sample pushed +R·grasp_off_eef), so the two stay
    # exactly coincident. The numeric readout still reports the stored eef (the trained
    # robot0_eef_pos), so the DATA stays honest — only the on-screen marker/line move to
    # where the gripper actually grasps.
    grasp_wrist = fingertip_grasp_point_wrist(grip_geom)
    tri_tcp = grasp_wrist
    # eef-frame vector eef-origin -> grasp point. grasp_wrist is in WRIST coords, so it
    # must be ROTATED by eef_R_wrist before being added to the eef-frame wrist offset
    # (the old `wrist_off + grasp_wrist` silently assumed eef axes == wrist axes, which
    # is false for a rotating extrinsic — same frame conflation as the old wrist offset).
    grasp_off_eef = wrist_off + vtf.SO3(np.asarray(wrist_q, dtype=np.float64)).as_matrix() @ grasp_wrist
    print(f"[replay] TCP marker + path drawn at the fingertip grasp point "
          f"(wrist +x {grasp_wrist[0]:.3f} m; {grasp_off_eef[0]*100:.1f} cm ahead of the "
          f"calibrated eef at {_TCP_ALONG_WRIST_X_M:.3f} m).")

    # --- scene --- (server is created by main(); its scene/GUI were reset before this call)
    server.scene.set_up_direction("+z")
    server.scene.add_grid("/grid", plane="xy", shadow_opacity=0.0, infinite_grid=True, cell_size=0.1)

    # /eef = the exact TCP pose (moved per-frame). Its children reproduce the
    # collection-viz cluster: a wrist child carrying the three yam-gripper meshes, the
    # three axis triads (TCP, tracker, wrist_yaw_link) and NOTHING ELSE, and the camera.
    eef = server.scene.add_frame("/eef", show_axes=False, origin_radius=0.0)
    # The wrist frame carries BOTH the offset and the ROTATION relative to the stored eef
    # (the extrinsic rotates puck->wrist axes; see wrist_pose_from_stored). Without the
    # rotation the whole cluster hangs at the wrong attitude.
    server.scene.add_frame("/eef/wrist", show_axes=False, position=tuple(wrist_off),
                           wxyz=tuple(wrist_q), origin_radius=0.0)
    server.scene.add_frame("/eef/wrist/tcp", show_axes=True, axes_length=0.08,
                           axes_radius=0.004, origin_radius=0.01, position=tuple(tri_tcp))
    server.scene.add_label("/eef/wrist/tcp/label", "TCP")
    # The puck triad needs its OWN rotation (R_ext^-1): the puck axes are NOT the wrist
    # axes — they differ by exactly the extrinsic's relabel. Drawn unrotated it inherited
    # the wrist axes and read identical to the TCP triad (a 180deg error).
    server.scene.add_frame("/eef/wrist/tracker", show_axes=True, axes_length=0.07,
                           axes_radius=0.0035, origin_radius=0.009, position=tuple(tri_trk),
                           wxyz=tuple(puck_triad_local_quat(T_ext)))
    server.scene.add_label("/eef/wrist/tracker/label", "tracker")
    server.scene.add_frame("/eef/wrist/wrist_yaw", show_axes=True, axes_length=0.07,
                           axes_radius=0.0035, origin_radius=0.009, position=tuple(tri_wrist))
    server.scene.add_label("/eef/wrist/wrist_yaw/label", "wrist_yaw_link")

    # Three yam-gripper meshes (base + 2 fingers) under the wrist frame at their
    # wrist-relative poses (URDF colours baked in), same as collection_viz gripper mode.
    # grip_geom was loaded above (to derive the grasp point).
    grip_nodes = {}
    for piece in grip_geom.pieces():
        grip_nodes[piece.name] = server.scene.add_mesh_trimesh(
            f"/eef/wrist/grip_{piece.name}", piece.mesh,
            position=tuple(piece.pos_closed), wxyz=tuple(piece.quat))

    # Camera frustum on the wrist, composed Rx(roll)·Ry(yaw) so +z (viser OpenCV
    # frustum forward) points OUTWARD along wrist +x — identical to collection_viz /
    # replay_zarr_viz. A single from_rpy_radians(r, y, 0) would mis-roll the image ~90°.
    rpy = np.asarray(args.cam_rpy, float)
    cam_rot = (vtf.SO3.from_rpy_radians(float(rpy[0]), 0.0, 0.0)
               .multiply(vtf.SO3.from_rpy_radians(0.0, float(rpy[1]), 0.0)))
    server.scene.add_frame("/eef/wrist/cam", show_axes=False,
                           position=tuple(np.asarray(args.cam_offset, float)),
                           wxyz=tuple(cam_rot.wxyz))
    frustum = server.scene.add_camera_frustum(
        "/eef/wrist/cam/view", fov=np.deg2rad(args.cam_fov_deg), aspect=4.0 / 3.0,
        scale=0.12, color=(40, 200, 255))
    # Wrench arrows ride the TCP pose (sensor-frame wrench, rotates with the shown pose).
    wrench = rc.WrenchArrows(server, name="/wrench") if take.has_ft else None

    # Secondary tracker triads (2pt right / 3pt head): the extra pucks, home-anchored
    # to their own per-role home so they stay in the scene alongside the primary
    # cluster. RAW puck pose (no gripper mesh / camera — those are single, on primary).
    # The CHEST frame carries its own triad (a body reference IS the drawn point). The
    # RIGHT hand does not: it gets the same /wrist child + three triads the left hand has
    # (below), so drawing an extra triad on its root would duplicate the TCP one.
    sec_frames = {
        role: server.scene.add_frame(f"/tracker_{role}",
                                     show_axes=(role in fmt.BODY_ROLES),
                                     axes_length=0.07, axes_radius=0.004,
                                     origin_radius=0.01 if role in fmt.BODY_ROLES else 0.0)
        for role in take.sec_trackers
    }
    for _role, _label in (("head", "chest"), ("left_foot", "foot L"),
                          ("right_foot", "foot R")):
        if _role in sec_frames:
            server.scene.add_label(f"/tracker_{_role}/label", _label)
    # Secondary display poses, extrinsic applied PER ROLE (hands: puck->TCP; chest: its
    # own rotation-only body relabel) — the same rule the converter uses, so
    # /tracker_right rides the pose stored in robot1_eef_pos and /tracker_head the axes
    # stored in head_rot_axis_angle.
    sec_pose = {role: take.sec_pose_disp(role, T_ext, T_head) for role in take.sec_trackers}
    # role -> (home_pos, home_quat). The CHEST has its own entry now ("head" -> the
    # torso point); before it fell through to the LEFT WRIST home, which drew the chest
    # triad inside the left hand.
    sec_home = {role: _HOME_BY_ROLE.get(role, _HOME_BY_ROLE["left"])
                for role in take.sec_trackers}

    # ===== RIGHT hand: the SAME cluster the left hand gets ========================
    # The right hand is a complete hand assembly (tracker + camera + F/T + gripper), so a
    # bare puck triad does not show enough to QC it. It gets the IDENTICAL frame chain as
    # the left — /tracker_right (at the TCP) -> /wrist (the inverse extrinsic) -> the three
    # triads + three yam meshes + the camera — because the two hands are the same physical
    # assembly with the same puck mount, and the converter treats them identically.
    #
    # Everything below is measured from the SAME constants the left cluster uses (wrist_off,
    # wrist_q, tri_tcp/tri_trk/tri_wrist, args.cam_offset, cam_rot): nothing is mirrored by
    # hand, so the two clusters cannot drift apart. Only the gripper MESHES come from the
    # right-hand URDF links, and the frustum colour differs so L/R stay tellable apart.
    #
    # Each piece is gated on ITS OWN stream: a right hand with a puck and no camera gets
    # meshes with no frustum, which is exactly what that take contains.
    grip_nodes_r: dict = {}
    grip_geom_r = frustum_r = wrench_r = None
    if "right" in take.sec_trackers:
        wrist_r = "/tracker_right/wrist"
        server.scene.add_frame(wrist_r, show_axes=False, position=tuple(wrist_off),
                               wxyz=tuple(wrist_q), origin_radius=0.0)
        server.scene.add_frame(f"{wrist_r}/tcp", show_axes=True, axes_length=0.08,
                               axes_radius=0.004, origin_radius=0.01, position=tuple(tri_tcp))
        server.scene.add_label(f"{wrist_r}/tcp/label", "TCP R")
        server.scene.add_frame(f"{wrist_r}/tracker", show_axes=True, axes_length=0.07,
                               axes_radius=0.0035, origin_radius=0.009, position=tuple(tri_trk),
                               wxyz=tuple(puck_triad_local_quat(T_ext)))
        server.scene.add_label(f"{wrist_r}/tracker/label", "tracker R")
        server.scene.add_frame(f"{wrist_r}/wrist_yaw", show_axes=True, axes_length=0.07,
                               axes_radius=0.0035, origin_radius=0.009, position=tuple(tri_wrist))
        server.scene.add_label(f"{wrist_r}/wrist_yaw/label", "wrist_yaw_link R")
        grip_geom_r = _gov.build_gripper_geom("right", urdf_path=args.yam_urdf)
        for piece in grip_geom_r.pieces():
            grip_nodes_r[piece.name] = server.scene.add_mesh_trimesh(
                f"{wrist_r}/grip_{piece.name}", piece.mesh,
                position=tuple(piece.pos_closed), wxyz=tuple(piece.quat))
        if take.has_cam_right:
            server.scene.add_frame(f"{wrist_r}/cam", show_axes=False,
                                   position=tuple(np.asarray(args.cam_offset, float)),
                                   wxyz=tuple(cam_rot.wxyz))
            frustum_r = server.scene.add_camera_frustum(
                f"{wrist_r}/cam/view", fov=np.deg2rad(args.cam_fov_deg), aspect=4.0 / 3.0,
                scale=0.12, color=(255, 170, 40))   # orange: never confuse it with camera L
            server.scene.add_label(f"{wrist_r}/cam/view/label", "camera R")
            server.scene.add_label("/eef/wrist/cam/view/label", "camera L")
        if take.has_ft_right:
            # World-root node like /wrench: WrenchArrows builds world-space endpoints, so
            # parenting under a moving frame would apply that transform twice.
            wrench_r = rc.WrenchArrows(server, name="/wrench_right")

    # --- GUI: info + per-frame readouts + charts ---
    gui = server.gui
    gui.configure_theme(control_layout="collapsible", control_width="large")
    with gui.add_folder("Take"):
        gui.add_text("Name", initial_value=take.name, disabled=True)
        gui.add_text("RGB frames", initial_value=f"{take.K} @ ~{take.K/max(take.cam_t[-1],1e-6):.0f} Hz",
                     disabled=True)
        # The playback timeline runs at the tracker's native rate (falls back to the RGB
        # grid on a take with no tracker) — the "not smooth" fix.
        gui.add_text("Playback", initial_value=f"{take.D} @ ~{take.disp_hz:.0f} Hz "
                     f"({'tracker' if take.disp_is_raw else 'RGB'})", disabled=True)
        seg_str = ", ".join(f"ep{s['index']}:{s['n_frames']}fr" for s in take.segments)
        gui.add_text("Episodes", initial_value=seg_str or "1", disabled=True)
        gui.add_text("Gripper calib", initial_value=take.calib_src, disabled=True)
        gui.add_text("Mode", initial_value=f"{take.mode or 'legacy'} (primary={primary_role})",
                     disabled=True)
        # WHICH per-hand modalities this take actually recorded. Spelled out because the
        # viewer draws only what is there, and "the right camera isn't shown" has two very
        # different causes (not wired up vs broken viewer) that this row separates.
        _kp = ["left: pose+cam" + ("+ft" if take.has_ft else "")
               + ("+grip" if take.has_gripper else "")]
        if "right" in take.sec_trackers:
            _kp.append("right: pose"
                       + ("+cam" if take.has_cam_right else "")
                       + ("+ft" if take.has_ft_right else "")
                       + ("+grip" if take.has_gripper_right else ""))
        for _role, _name in (("head", "chest"), ("left_foot", "foot L"),
                             ("right_foot", "foot R")):
            if _role in take.sec_trackers:
                _kp.append(f"{_name}: pose")
        gui.add_text("Keypoints", initial_value=" · ".join(_kp), disabled=True)
        if take.has_cam_right:
            _sk = take.cam_right_skew
            gui.add_text("camera R pair skew",
                         initial_value=f"{(_sk['skew_max_s'] or 0.0)*1e3:+.0f} ms max "
                                       f"(reuse x{_sk['reused_max']})", disabled=True)
        gui.add_text("TCP calibrated", initial_value=str(tcp_calibrated), disabled=True)
        # R5: F/T tare events during the take, as take-relative times (a tare re-zeroes
        # the wrench). Listed here since the uPlot charts have no vertical-marker API.
        if take.tare_times.size:
            gui.add_text("F/T tares (s)",
                         initial_value=", ".join(f"{t:.1f}" for t in take.tare_times),
                         disabled=True)
        # --- take navigation: page through the session and delete while watching ---
        # The controls only set nav[...] and stop the timeline; main() (on the main thread)
        # does the delete + rebuild. ``timeline`` is bound later in this function and
        # resolved at click time via the closure. Prev/Next are disabled at the ends.
        gui.add_text("Take #", initial_value=f"{take_index + 1} / {len(take_dirs)}",
                     disabled=True)
        # The clickable take picker, carrying each take's frame count + duration — the
        # raw-take twin of the zarr viewer's Episode dropdown.
        take_dd = gui.add_dropdown("Take", options=tuple(labels),
                                   initial_value=labels[take_index])
        prev_btn = gui.add_button("⏮ Prev take", disabled=(take_index == 0))
        next_btn = gui.add_button("⏭ Next take", disabled=(take_index >= len(take_dirs) - 1))
        del_btn = gui.add_button("🗑 Delete this take")
        # ONE confirm click, because the delete is irreversible and this button sits next
        # to the two navigation buttons an operator clicks dozens of times per session.
        # The arm state lives only in this page build, so ANY navigation disarms it.
        del_armed = {"on": False}
        del_hint = gui.add_text("delete", initial_value="permanent — click twice",
                               disabled=True)

        @take_dd.on_update
        def _(_e) -> None:
            nav["action"] = "goto"
            nav["index"] = labels.index(take_dd.value)
            timeline.stop()

        @prev_btn.on_click
        def _(_e) -> None:
            nav["action"] = "prev"
            timeline.stop()

        @next_btn.on_click
        def _(_e) -> None:
            nav["action"] = "next"
            timeline.stop()

        @del_btn.on_click
        def _(_e) -> None:
            if not del_armed["on"]:
                del_armed["on"] = True
                del_btn.label = f"🗑 CONFIRM delete {take.name}"
                del_hint.value = ("click again to delete for good — no undo, "
                                  "or change take to cancel")
                return
            nav["action"] = "delete"
            timeline.stop()

    # F/T display mode: True = gravity/inertia-COMPENSATED, False = RAW. Both traces
    # are loaded; the checkbox swaps which one the arrows, charts, and |F|/|T| readouts
    # show — the SAME live RAW<->COMP switch the collection viz has. ``comp`` is read on
    # the loop thread; the callback only writes it, and the chart refill happens on the
    # loop thread via ``before_render`` (never mutate scene from the GUI callback thread).
    ft_mode = {"comp": bool(take.wrench_default_comp), "charts_comp": None}
    with gui.add_folder("Frame readout"):
        rd_t = gui.add_text("t (s)", initial_value="0.000", disabled=True)
        rd_ep = gui.add_text("episode", initial_value="-", disabled=True)
        rd_pos = gui.add_text("TCP xyz", initial_value="-", disabled=True)
        rd_grip = gui.add_text("gripper width", initial_value="-", disabled=True)
        rd_ftsrc = gui.add_text(
            "F/T source",
            initial_value=("compensated" if ft_mode["comp"] else "raw") if take.has_ft else "none",
            disabled=True)
        rd_force = gui.add_text("|F| (N)", initial_value="-", disabled=True)
        rd_torque = gui.add_text("|T| (Nm)", initial_value="-", disabled=True)
    # RIGHT-hand readouts, and ONLY for streams this take has. An always-present row
    # showing "-" is how a missing modality stops being noticed.
    rd_r_grip = rd_r_force = None
    if take.has_gripper_right or take.has_ft_right:
        with gui.add_folder("Frame readout — RIGHT"):
            if take.has_gripper_right:
                rd_r_grip = gui.add_text("gripper width", initial_value="-", disabled=True)
            if take.has_ft_right:
                rd_r_force = gui.add_text(
                    f"|F| (N, {take.wrench_right_source})", initial_value="-", disabled=True)
    with gui.add_folder("View"):
        show_cam = gui.add_checkbox("Show camera", initial_value=True)
        show_path = gui.add_checkbox("Show TCP path", initial_value=True)
        # ARROWS ARE F^cmd, CHARTS ARE F_meas — see the same note in replay_zarr_viz.py. The
        # arrow carries WrenchArrows' meas->cmd negation (the direction a robot would push to
        # reproduce this contact); the charts stay the recorded wrench verbatim, raw or
        # compensated per the toggle below, which is what makes this the fidelity view.
        show_wrench = gui.add_checkbox("Show F^cmd arrows (= -F_meas)", initial_value=bool(take.has_ft))
        gui.add_text("arrows vs charts",
                     initial_value="arrows = F^cmd (negated, deploy sign) · charts = recorded F_meas",
                     disabled=True)
        show_mesh = gui.add_checkbox("Show gripper mesh", initial_value=True)
        # Compensated <-> raw F/T toggle. Disabled (stuck on raw) when the take did NOT
        # record a compensated sidecar — same gating as the live viz.
        show_comp = gui.add_checkbox(
            "Compensated wrench", initial_value=bool(take.wrench_default_comp),
            disabled=not take.has_comp)
        # Home-anchored view (default ON): re-base each frame onto an upright home so the
        # gripper reads like the collection viz (z-up), instead of the raw absolute puck
        # pose. DISPLAY-ONLY — the stored data and the readouts are unchanged. OFF =
        # strict fidelity (raw stored world pose). --raw-pose starts this OFF.
        home_anchor_view = gui.add_checkbox("Home-anchored view (like collection)",
                                            initial_value=not args.raw_pose)

        @show_comp.on_update
        def _(_e) -> None:
            ft_mode["comp"] = bool(show_comp.value)

        @home_anchor_view.on_update
        def _(_e) -> None:
            # Flag a rebuild consumed on the loop thread (the path polyline + display
            # poses depend on the anchor mode); do NOT mutate the scene here.
            st["rebuild"] = True

    with gui.add_folder("Charts"):
        grip_chart = rc.make_single_chart(gui, "Gripper width (m)", "width", color_idx=1,
                                          y_range=(0.0, max(take.width_open_m * 1.05, 0.1)))
        rc.set_full_single(grip_chart, take.cam_t, take.width_grid)
        if take.has_ft:
            force_chart = rc.make_xyz_chart(gui, "Force LEFT (N)", ("Fx", "Fy", "Fz"))
            torque_chart = rc.make_xyz_chart(gui, "Torque LEFT (Nm)", ("Tx", "Ty", "Tz"))
    # RIGHT-hand F/T charts, ONLY when the take recorded a right Bota (the same absence rule
    # as the right readout: an always-present empty chart is how a missing sensor goes
    # unnoticed). The right hand has ONE trace (its own compensated sidecar when its Bota
    # was compensated, else raw — named in the folder title), so no raw<->comp toggle and
    # a single fill here. Charts stay the recorded F_meas, like the left ones.
    force_chart_r = torque_chart_r = None
    if take.has_ft_right and take.wrench_grid_right is not None:
        with gui.add_folder(f"Charts — RIGHT (F/T {take.wrench_right_source})"):
            force_chart_r = rc.make_xyz_chart(gui, "Force RIGHT (N)", ("Fx", "Fy", "Fz"))
            torque_chart_r = rc.make_xyz_chart(gui, "Torque RIGHT (Nm)", ("Tx", "Ty", "Tz"))
            rc.set_full_xyz(force_chart_r, take.cam_t, take.wrench_grid_right[:, 0:3])
            rc.set_full_xyz(torque_chart_r, take.cam_t, take.wrench_grid_right[:, 3:6])

    # Display state (recomputed on the anchor toggle). disp_* are the DRAWN cluster poses
    # on the DISPLAY grid; path_pos is the overview polyline = the same trajectory pushed
    # to the fingertip grasp point. tcp_pos_disp stays the raw-world readout source.
    st: dict = {"rebuild": False, "disp_pos": None, "disp_wxyz": None, "sec_disp": {},
                "path_pos": None, "path_pos_right": None,
                "last_cam_i": -1, "last_cam_right_i": -1}

    def _shift_to_grasp(pos: np.ndarray, wxyz: np.ndarray) -> np.ndarray:
        """Push each eef point forward to the fingertip grasp point: p + R(wxyz)·
        grasp_off_eef. The grasp offset is rigid in the eef frame, so rotating it per
        sample keeps the drawn line ON the TCP triad (which is the same offset, as a
        child of /eef/wrist) for every orientation."""
        if pos.shape[0] == 0:
            return pos
        R = vtf.SO3(np.asarray(wxyz, dtype=np.float64)).as_matrix()  # (N,3,3)
        return pos + np.einsum("nij,j->ni", R, grasp_off_eef)

    def recompute_display() -> None:
        on = home_anchor_view.value
        # The moving cluster rides the DISPLAY grid (native ~80 Hz). The overview path is
        # the SAME trajectory pushed forward to the fingertip grasp point, so it overlays
        # the TCP marker by construction — no separate raw-rate source / external anchor
        # any more (cluster and path are one grid now).
        st["disp_pos"], st["disp_wxyz"] = _rebase(tcp_pos_disp, tcp_wxyz_disp, home_p,
                                                  _HOME_Q, on, args.reach_scale)
        st["path_pos"] = _shift_to_grasp(st["disp_pos"], st["disp_wxyz"])
        st["sec_disp"] = {}
        # sec_pose, NOT sec_trackers: the right hand's poses have the puck->TCP extrinsic
        # applied (the chest's do not), so /tracker_right rides the same frame the left
        # cluster and robot1_eef_pos do.
        for role, (sp, sq) in sec_pose.items():
            _shp, _shq = sec_home[role]
            st["sec_disp"][role] = _rebase(sp, sq, _shp, _shq, on, args.reach_scale)
        # RIGHT-hand path: the same grasp-point push as the left, off the right hand's
        # DISPLAY pose (eef frame, extrinsic applied), so it overlays the right TCP marker
        # by construction exactly as the green line overlays the left one.
        if "right" in st["sec_disp"]:
            sp_r, sq_r = st["sec_disp"]["right"]
            st["path_pos_right"] = _shift_to_grasp(sp_r, sq_r)
        else:
            st["path_pos_right"] = None

    def set_path() -> None:
        # TCP path — GREEN (the GT trajectory colour). Drawn on the DISPLAY grid (the
        # tracker's native ~80 Hz, not the ~15 Hz camera grid) with add_line_segments,
        # i.e. the EXACT dense polyline: one straight segment between every pair of
        # consecutive samples, NO client-side resampling. (add_spline_catmull_rom was
        # wrong here: viser forwards segments=None -> drei's default 20 -> three's
        # getPoints(20) redraws the WHOLE curve as just 21 vertices, so a tortuosity-6+
        # path collapsed to a near-straight line REGARDLESS of how many points we fed it —
        # the "很直" the user saw.) At 80 Hz the segments read perfectly smooth. Same grid
        # as the moving cluster, so line == marker for every frame.
        def _draw(name: str, pos: np.ndarray, color) -> None:
            if show_path.value and pos.shape[0] >= 2:
                p = pos.astype(np.float32)
                segs = np.stack([p[:-1], p[1:]], axis=1)  # (N-1, 2, 3) start/end pairs
                server.scene.add_line_segments(name, points=segs, colors=color, line_width=4.0)
            else:
                # Hidden (path off, or <2 points): a single degenerate segment, invisible.
                zp = (pos[:1] if pos.shape[0] else np.zeros((1, 3))).astype(np.float32)
                server.scene.add_line_segments(name, points=np.stack([zp, zp], axis=1),
                                               colors=color, line_width=4.0, visible=False)
        _draw("/eef_path", st["path_pos"], _GT_GREEN)
        # RIGHT hand, YELLOW, same toggle, same grid, same grasp-point anchor. Only when the
        # take has a right tracker (a 1pt take draws exactly what it drew before).
        if st["path_pos_right"] is not None:
            _draw("/eef_path_right", st["path_pos_right"], _RIGHT_PATH_YELLOW)

    def refill_ft_charts() -> None:
        """Fill the force/torque charts with the currently-selected wrench trace. Runs
        on the loop thread. uPlot auto-scales y, so raw (~14 N) vs compensated (~0.4 N)
        both render legibly after a switch."""
        if not take.has_ft:
            return
        wg = take.wrench_grid(ft_mode["comp"])
        rc.set_full_xyz(force_chart, take.cam_t, wg[:, 0:3])
        rc.set_full_xyz(torque_chart, take.cam_t, wg[:, 3:6])
        ft_mode["charts_comp"] = ft_mode["comp"]

    recompute_display()
    set_path()
    refill_ft_charts()  # initial fill on the main thread, before the loop starts

    # episode lookup for the readout — i indexes the DISPLAY grid now, so use the
    # display-grid segments (same marker split, denser boundaries).
    def episode_of(i: int) -> str:
        for s in take.disp_segments:
            if s["start"] <= i < s["end"]:
                return str(s["index"])
        return "-"

    def render_at(i: int) -> None:
        # i indexes the DISPLAY grid (native ~80 Hz). Drive the /eef frame to the display
        # TCP pose; the wrist/mesh/triad/camera cluster rides it rigidly (children of
        # /eef/wrist). The camera IMAGE only exists at ~15 Hz, so it is held via
        # disp_cam_idx (the most-recent RGB) — everything else moves at the native rate.
        eef.position = tuple(st["disp_pos"][i])
        eef.wxyz = tuple(st["disp_wxyz"][i])
        # secondary tracker triads (2pt/3pt only)
        for role, frame in sec_frames.items():
            sp, sq = st["sec_disp"][role]
            frame.position = tuple(sp[i])
            frame.wxyz = tuple(sq[i])
        # gripper meshes: visibility + jaw LERP from the recorded open fraction
        for name in grip_nodes:
            grip_nodes[name].visible = show_mesh.value
        if show_mesh.value:
            frac = float(take.frac_grid[i])
            for pc in (grip_geom.finger1, grip_geom.finger2):
                grip_nodes[pc.name].position = tuple(pc.pos_at(frac))
        # RIGHT-hand meshes + jaws, from that hand's OWN recorded fraction. With no right
        # gripper stream they hold CLOSED rather than mirroring the left, so the drawing
        # never implies a grasp nothing measured.
        for name in grip_nodes_r:
            grip_nodes_r[name].visible = show_mesh.value
        if show_mesh.value and grip_geom_r is not None:
            frac_r = float(take.frac_right[i]) if take.has_gripper_right else 0.0
            for pc in (grip_geom_r.finger1, grip_geom_r.finger2):
                grip_nodes_r[pc.name].position = tuple(pc.pos_at(frac_r))
        # camera frame: the RGB is ~15 Hz, so map this display frame to the RGB it holds
        # (disp_cam_idx) and decode only when THAT index changes (so ~5 display frames
        # reuse one decode). Guard a missing/corrupt JPEG so playback never silently
        # freezes on a stale image — mark the slot done either way so we don't retry the
        # same bad file every loop iteration.
        cam_i = int(take.disp_cam_idx[i])
        frustum.visible = show_cam.value
        if show_cam.value and cam_i != st["last_cam_i"]:
            st["last_cam_i"] = cam_i
            jp = take.jpeg_path(cam_i)
            img = None
            if os.path.exists(jp):
                with open(jp, "rb") as f:
                    img = rc.decode_jpeg(f.read())
            else:
                print(f"[replay] missing camera frame {jp}", flush=True)
            if img is not None:
                h, w = img.shape[:2]
                frustum.aspect = w / max(h, 1)
                frustum.image = img
        # RIGHT camera: its own decode cache keyed on ITS OWN chosen row (the two cameras
        # are paired to the display grid independently, so one index cannot serve both).
        if frustum_r is not None:
            frustum_r.visible = show_cam.value
            cam_ri = int(take.disp_cam_right_idx[i])
            if show_cam.value and cam_ri != st.get("last_cam_right_i"):
                st["last_cam_right_i"] = cam_ri
                jpr = take.jpeg_path_right(i)
                img_r = None
                if jpr and os.path.exists(jpr):
                    with open(jpr, "rb") as f:
                        img_r = rc.decode_jpeg(f.read())
                else:
                    print(f"[replay] missing RIGHT camera frame {jpr}", flush=True)
                if img_r is not None:
                    hr, wr = img_r.shape[:2]
                    frustum_r.aspect = wr / max(hr, 1)
                    frustum_r.image = img_r
        # F/T arrows ride the fingertip grasp point (same anchor as the TCP triad + line),
        # so the wrench originates where the gripper grasps, not 4.5 cm back at the eef.
        # Display-grid wrench so the arrows move at the native rate with the cluster. The RAW
        # sensor triple goes in: WrenchArrows applies the measured sensor->TCP relabel itself.
        wg = take.wrench_disp(ft_mode["comp"])
        if wrench is not None:
            wrench.set_visible(show_wrench.value)
            if show_wrench.value:
                R = vtf.SO3(np.asarray(st["disp_wxyz"][i], dtype=np.float64)).as_matrix()
                grasp_origin = st["disp_pos"][i] + R @ grasp_off_eef
                wrench.update(grasp_origin, st["disp_wxyz"][i], wg[i, 0:3], wg[i, 3:6])
        # RIGHT-hand wrench, anchored on ITS grasp point (the same wrist->grasp offset,
        # rotated by the RIGHT puck's shown orientation) so the arrows sit on the right
        # fingertips instead of the left ones.
        if wrench_r is not None:
            wrench_r.set_visible(show_wrench.value)
            if show_wrench.value:
                sp_r, sq_r = st["sec_disp"]["right"]
                R_r = vtf.SO3(np.asarray(sq_r[i], dtype=np.float64)).as_matrix()
                grasp_origin_r = sp_r[i] + R_r @ grasp_off_eef
                wr_now = take.wrench_disp_right
                wrench_r.update(grasp_origin_r, sq_r[i], wr_now[i, 0:3], wr_now[i, 3:6])
        # readouts (RAW data, independent of the home-anchor DISPLAY toggle)
        rd_t.value = f"{take.disp_t[i]:.3f}"
        rd_ep.value = episode_of(i)
        rd_pos.value = np.array2string(tcp_pos_disp[i], precision=3, suppress_small=True)
        gw = take.width_disp[i]
        rd_grip.value = f"{gw*1000:.1f} mm ({'grasp' if gw < 0.08 else 'open'})"
        if rd_r_grip is not None:
            gwr = take.width_disp_right[i]
            rd_r_grip.value = f"{gwr*1000:.1f} mm ({'grasp' if gwr < 0.08 else 'open'})"
        if rd_r_force is not None:
            rd_r_force.value = f"{np.linalg.norm(take.wrench_disp_right[i, 0:3]):.2f}"
        if take.has_ft:
            rd_ftsrc.value = ("compensated" if wg is take.wrench_disp_comp else "raw")
        rd_force.value = f"{np.linalg.norm(wg[i, 0:3]):.2f}"
        rd_torque.value = f"{np.linalg.norm(wg[i, 3:6]):.3f}"

    # The scrub timeline runs on the DISPLAY grid (native ~80 Hz), so play advances the
    # moving cluster at the true recorded hand rate — the "not smooth" fix. (Timeline
    # advances by wall-clock*speed and indexes by searchsorted, so a non-uniform ~80 Hz
    # grid plays in real time exactly as the ~15 Hz one did.)
    timeline = rc.Timeline(server, take.disp_t, render_at, folder_label="Playback")

    def before_render() -> None:
        # Home-anchor toggle: recompute display poses + rebuild the path polyline on the
        # loop thread (all scene mutation stays here), then force a repaint so a paused
        # frame updates immediately.
        if st["rebuild"]:
            st["rebuild"] = False
            recompute_display()
            set_path()
            timeline.request_render()
        # Compensated<->raw toggle: refill charts + force a re-render so the ARROWS,
        # |F|/|T| readouts, and the "F/T source" label flip together with the charts
        # (render_at is skipped while PAUSED on an unchanged frame otherwise).
        if take.has_ft and ft_mode["charts_comp"] != ft_mode["comp"]:
            refill_ft_charts()
            timeline.request_render()

    timeline.before_render = before_render
    print("[replay] drag 'Time (s)' to scrub, ⏵ to play; ⏮/⏭ change take, 🗑 deletes it",
          flush=True)
    timeline.run()
    return nav


if __name__ == "__main__":
    raise SystemExit(main())
