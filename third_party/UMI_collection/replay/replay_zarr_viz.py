#!/usr/bin/env python3
"""Replay a CONVERTED UMI Zarr (``.zarr.zip``) in viser — the conversion-fidelity
checker (the SECOND of the two replay scripts).

Where ``replay_raw_viz.py`` shows the RAW capture, this tool shows the data
EXACTLY as it now lives in the ``.zarr.zip`` the trainer will load — nothing is
re-solved or re-derived. That is the whole point: if the converter dropped,
aliased, or mis-aligned anything, you see it HERE, against the raw tool, instead
of discovering it after a training run.

It loads the stock-UMI ReplayBuffer layout written by ``convert_to_umi.py``
(``meta/episode_ends`` + ``data/{robot0_eef_pos, robot0_eef_rot_axis_angle,
robot0_gripper_width, robot0_ft_wrench, camera0_rgb, episode_timestamps, …}``)
and, for ONE selectable episode:

MULTI-KEYPOINT (2pt/3pt): when the file carries them, this ALSO draws the RIGHT hand
(``robot1_*`` + ``camera1_rgb`` + ``robot1_ft_wrench`` + ``robot1_gripper_width``) as a
second, identical gripper cluster in ORANGE, and the CHEST (``head_*``) as a violet triad
— no gripper/camera/F-T there, because that keypoint is a body reference (it is the third
point of holosoma's exp:g1-wbt-3pt-hybrid-unified[-anneal]: torso_link + 0.35 z). Each is
gated on its own key, so a file with a right pose but no right camera draws the right
cluster WITHOUT a frustum. Nothing is drawn that is not in the file.

  * draws the TCP as a moving triad and its full ``robot0_eef_pos`` path as a
    polyline (so a resampling glitch shows as a kink);
  * shows the STORED ``camera0_rgb`` (224x224, JpegXl-decoded by zarr) on a
    frustum riding the TCP — the same frames the policy will train on;
  * draws ``robot0_ft_wrench`` as force/torque arrows + charts and
    ``robot0_gripper_width`` as a readout + chart (with UMI's 0.08 m grasp line);
  * marks the stored ``robot0_demo_start_pose`` / ``robot0_demo_end_pose`` so you
    can confirm they coincide with the episode's first/last TCP pose.

The episode is chosen from a CLICKABLE dropdown in the browser (the user asked to
pick an episode from the web page). ``episode_timestamps`` is RELATIVE per episode
(DECISION-A: each starts at 0), so the timeline runs 0..episode_duration and plays
back in true relative time.

Deliberately does NOT run the mink IK: re-solving would re-introduce robot motion
and could MASK a conversion artifact (e.g. a frozen tracker run), defeating the
purpose. It only renders what the file contains.

Run:
    .venv/bin/python replay/replay_zarr_viz.py session_XXX.zarr.zip            # → :8084
    .venv/bin/python replay/replay_zarr_viz.py out.zarr.zip --episode 2 --port 8084
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import viser
from viser import transforms as vtf

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.join(_HERE, "..")
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_REPO, "processing"))

import replay_common as rc  # noqa: E402
from force_recovery_overlay import ForceRecoveryOverlay  # noqa: E402
from motion_completion_overlay import MotionCompletionOverlay  # noqa: E402

_GRASP_THRESH_M = 0.08  # UMI's hard-coded grasp threshold

# --- gripper-mesh display, to match the collection viz (RECORD=1 MODE=1pt) ------
# The zarr replay used to draw only a bare /tcp triad — the user asked it to show the
# SAME scene the collection viz does: the three yam-gripper meshes + EXACTLY the three
# axis triads (TCP, tracker, wrist_yaw_link) and NOTHING else. The gripper geometry +
# the triad wrist-relative offsets live in the viz package (single source), so import
# them rather than re-deriving. viz/ is added to sys.path below.
sys.path.insert(0, os.path.join(_REPO, "viz"))
import gripper_only_view as _gov  # noqa: E402
from collection_viz import (  # noqa: E402
    _TCP_ALONG_WRIST_X_M, _TRACKER_TRUE_XYZ_M, _WRIST_YAW_XYZ_M,
)

# Total jaw opening (m) at gripper_width == this maps to the finger LERP fraction 1.0.
# Single source is the repo gripper calibration (width_open_m, DECISION-H); the stored
# robot0_gripper_width is absolute metres, so frac = width / _WIDTH_OPEN_M.
_WIDTH_OPEN_M = 0.095

# Green (collection GT trajectory colour the user expects), replacing the old yellow.
_GT_GREEN = (90, 220, 120)
# Secondary keypoint path colours. Deliberately far from _GT_GREEN (the LEFT/primary
# path) in hue: the whole point of drawing three keypoints at once is telling them apart.
_R_ORANGE = (255, 170, 40)     # RIGHT hand (robot1_*), matches its camera frustum
_CHEST_VIOLET = (170, 120, 255)  # CHEST (head_*), the body reference


# Camera mount, IDENTICAL to collection_viz / replay_raw_viz: offset (0.10,0,0.04) from
# the WRIST, rotation composed Rx(roll)·Ry(yaw) so +z (viser OpenCV frustum forward)
# points OUTWARD along wrist +x. A single from_rpy_radians(r, y, 0) would compose
# intrinsically X→Y→Z and mis-roll the image ~90° (the bug the user saw).
_CAM_OFFSET_XYZ = (0.10, 0.0, 0.04)
_CAM_RPY = (4.7124, 1.5708, 0.0)

# Display defaults to the RAW STORED pose — the same convention as
# RAW_FRAME=1 collect_data.sh and replay_raw_viz, so this viewer and the raw-take viewer
# are directly comparable frame-for-frame (that comparison IS the fidelity check).
# The stored eef poses are ABSOLUTE world poses in the recorded libsurvive frame; held in
# the hand the puck's +z points at the floor, so a faithful render can look "upside down"
# — that is the data, not a bug. The optional "Home-anchored view" checkbox re-bases every
# frame onto a fixed upright home:
#     R_disp = (R_cur · R_ref⁻¹) · R_home ,   p_disp = p_home + (p_cur − p_ref)
# with R_home = identity (z-up) and ref = the episode's first frame — the SAME math as
# gripper_only_view.home_anchor (imported, single source). That is DISPLAY-ONLY and applies
# a world rotation the DATA does not contain, so it is for comfortable eyeballing only,
# never for judging fidelity. Either way the stored arrays + readouts are untouched.
# PER-ROLE home, straight out of gripper_only_view: the G1's zero configuration (both
# forearms level along +x, wrists parallel, 0.297 m apart) for the hands, and holosoma's
# torso point for the chest. Shared with the collection viz and the raw-take viewer so
# all three draw a home-anchored 2pt/3pt session in the SAME starting arrangement.
_HOME_BY_ROLE = {
    role: (np.asarray(hp, dtype=np.float64), np.asarray(hq, dtype=np.float64))
    for role, (hp, hq) in _gov.HOME_WRIST_POSE.items()
}
_HOME_P, _HOME_Q = _HOME_BY_ROLE["left"]  # the primary hand, for single-keypoint callers


# ---------------------------------------------------------------------------
class ZarrEpisodes:
    """Lazy per-episode view over a UMI ReplayBuffer .zarr.zip.

    Slices each ``data/*`` array on the episode boundaries from
    ``meta/episode_ends`` (cumulative). Arrays are read on demand per episode so a
    big dataset doesn't all land in RAM at once; one episode's RGB (T x 224 x 224
    x 3 uint8) is the only large block held."""

    def __init__(self, zarr_path: str):
        import zarr
        from imagecodecs_numcodecs import register_codecs
        register_codecs()  # so camera0_rgb (JpegXl) decodes on read
        self.path = zarr_path
        self.g = zarr.open(zarr_path, "r")
        self.attrs = dict(self.g.attrs)
        self.ends = np.asarray(self.g["meta/episode_ends"][:], dtype=np.int64)
        self.starts = np.concatenate([[0], self.ends[:-1]]).astype(np.int64) if len(self.ends) else np.zeros(0, np.int64)
        self.n_ep = int(len(self.ends))
        self.data_keys = sorted(self.g["data"].keys())
        self.has_ft = "robot0_ft_wrench" in self.data_keys
        # ---- 2pt/3pt secondary keypoints + per-hand modalities -------------------
        # Every one of these is INDEPENDENTLY optional (the converter emits each only
        # when that device streamed), so each gets its own flag rather than one
        # "is3pt" — a 3pt session recorded before the right camera was wired up has
        # robot1_eef_pos and no camera1_rgb, and the viewer must draw exactly what is
        # in the file. That is the whole job of this tool.
        self.has_robot1 = "robot1_eef_pos" in self.data_keys
        self.has_cam1 = "camera1_rgb" in self.data_keys
        self.has_ft1 = "robot1_ft_wrench" in self.data_keys
        self.has_grip1 = "robot1_gripper_width" in self.data_keys
        # head_* is the CHEST keypoint: a body pose with NO puck->TCP extrinsic (only its
        # own rotation-only chest relabel, so head_pos IS the raw puck position) and no
        # camera / F-T / gripper. It is the third point of holosoma's
        # exp:g1-wbt-3pt-hybrid-unified[-anneal] (torso_link + 0.35 z). Drawn AS STORED,
        # so which chest convention a dataset carries is visible here without a flag —
        # cross-check attrs.T_tracker_head (identity == the pre-2026-08-26 raw axes).
        self.has_head = "head_pos" in self.data_keys
        # HL-v2 window arrays, if this dataset has them. They hold the PER-ANCHOR action
        # chunks the policy is trained on, which is what the force-recovery overlay augments —
        # so the overlay funnels onto the real training target instead of a re-derivation.
        self.hl_role = str(self.attrs.get("robot0_primary_role") or "left")
        self.hl_force_key = f"hl_{self.hl_role}_action_force"
        self.hl_valid_key = f"hl_{self.hl_role}_action_force_valid"
        self.has_hl_force = (bool(self.attrs.get("hl_windows"))
                             and self.hl_force_key in self.data_keys)
        self.hl_horizon = int(self.attrs.get("hl_action_horizon", 0) or 0)
        self.hl_dt = 1.0 / float(self.attrs.get("hl_action_hz", 30.0) or 30.0)

    def action_force_chunk(self, ep: int, local_frame: int) -> np.ndarray | None:
        """The stored ``hl_<role>_action_force`` chunk at one anchor, or None if unusable.

        Indexed GLOBALLY (``episode_start + local_frame``) because the ``hl_*`` arrays are
        dataset-wide like every other ``data/*`` array. Returns None when any step of the
        chunk is an unsupervised target — augmenting a window the trainer itself masks would
        put a funnel onto a label that does not exist."""
        if not self.has_hl_force:
            return None
        g_idx = int(self.starts[ep]) + int(local_frame)
        if not 0 <= g_idx < int(self.ends[-1]):
            return None
        valid = np.asarray(self.g[f"data/{self.hl_valid_key}"][g_idx], dtype=np.float32)
        if not np.all(valid > 0.5):
            return None
        return np.asarray(self.g[f"data/{self.hl_force_key}"][g_idx], dtype=np.float64)

    def slice(self, ep: int, key: str) -> np.ndarray:
        a, b = int(self.starts[ep]), int(self.ends[ep])
        return np.asarray(self.g[f"data/{key}"][a:b])

    def episode_lengths(self) -> list[int]:
        return [int(b - a) for a, b in zip(self.starts, self.ends)]


def rotvec_to_wxyz(rotvec: np.ndarray) -> np.ndarray:
    """[T,3] axis-angle -> [T,4] wxyz for viser frames."""
    from scipy.spatial.transform import Rotation
    q_xyzw = Rotation.from_rotvec(np.asarray(rotvec, dtype=np.float64)).as_quat()
    return np.atleast_2d(q_xyzw)[:, [3, 0, 1, 2]]


def wrist_pose_from_stored(T_tracker_tcp: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """WRIST pose relative to the STORED eef (TCP) frame: (offset[3], quat_wxyz[4]).

    IDENTICAL to ``replay_raw_viz.wrist_pose_from_stored`` (kept in sync, pinned by
    test_replay_logic, so the raw and zarr viewers hang the SAME cluster).

    The converter stored ``world_T_eef = world_T_puck @ T_tracker_tcp`` — the extrinsic
    acts on the RECORDED RAW POSE, the physical VIVE puck, NOT the wrist. So the eef
    frame relates to the wrist frame through the extrinsic's ROTATION as well as its
    translation::

        eef_T_wrist = inv(T_tracker_tcp) @ [R_ext | -R_ext·_TRACKER_TRUE_XYZ_M]

    (``_TRACKER_TRUE_XYZ_M`` is the puck's seat in WRIST coords, so it must be rotated
    into the puck frame before composing with the puck-frame extrinsic translation.)

    The OLD helper returned ``-(_TRACKER_TRUE_XYZ_M + t)``, ADDING a puck-frame
    translation to a wrist-frame vector — valid ONLY while the extrinsic rotation was
    identity. The measured extrinsic now carries the ``-y,-x,-z`` puck->wrist relabel, and
    under that assumption the cluster sat ~33 cm off (the same frame conflation that made
    the old pure-translation ``(0.225,0,-0.05)`` — a WRIST-frame lever — wrong when
    applied in the PUCK frame). The full 4x4 inverse used here is exact for ANY extrinsic.

    Self-describing: reads whatever extrinsic the dataset was built with, so an older
    identity-rotation dataset still renders correctly (the rotation term degenerates).
    For the current extrinsic it yields exactly ``(-0.242, 0, 0)`` with identity rotation
    — i.e. the wrist sits 0.242 m back along the TCP's own +x, the measured
    wrist->fingertip-midpoint datum, so the TCP triad lands on the trajectory and the
    "tracker" triad returns to the physical puck."""
    T = np.asarray(T_tracker_tcp, dtype=np.float64)
    R_ext, t_ext = T[:3, :3], T[:3, 3]
    puck_t_wrist = -(R_ext @ np.asarray(_TRACKER_TRUE_XYZ_M, dtype=np.float64))
    # puck_R_wrist == R_ext for this rig (the extrinsic's rotation IS the puck->wrist axis
    # relabel), so eef_R_wrist = R_ext^T @ puck_R_wrist collapses to IDENTITY: the stored
    # eef already shares the wrist's axes. Kept as the explicit product so a future
    # extrinsic whose rotation is NOT the puck->wrist relabel still composes correctly.
    puck_R_wrist = R_ext
    eef_R_wrist = R_ext.T @ puck_R_wrist
    eef_t_wrist = R_ext.T @ (puck_t_wrist - t_ext)
    return eef_t_wrist, _gov._rotmat_to_wxyz(eef_R_wrist)


def puck_triad_local_quat(T_tracker_tcp: np.ndarray) -> np.ndarray:
    """Local (wrist-relative) ROTATION for the "tracker" (puck) triad — wxyz.

    IDENTICAL to ``replay_raw_viz.puck_triad_local_quat`` (pinned by test_replay_logic).
    The triad is a CHILD of the wrist frame, so unrotated it would inherit the WRIST axes
    — wrong, since the puck axes differ from the wrist axes by ``R_ext`` (the measured
    -y,-x,-z relabel). Since ``R_wrist_world = R_puck_world @ R_ext``, the local rotation
    that renders the triad back at the raw puck orientation is ``R_ext^-1``. Identity for
    a legacy pure-translation extrinsic, so older datasets are unaffected."""
    R_ext = np.asarray(T_tracker_tcp, dtype=np.float64)[:3, :3]
    return _gov._rotmat_to_wxyz(R_ext.T)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("zarr_path", help="a converted .zarr.zip (from processing/convert_to_umi.py)")
    p.add_argument("--episode", type=int, default=0, help="initial episode index to show")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8084)
    p.add_argument("--cam-fov-deg", type=float, default=58.0)
    # OPT-IN, and the only thing that changes anything: without it this viewer is
    # byte-for-byte the gripper-only replay it has always been — no humanoid, no G1 URDF
    # load, no model process, no extra GUI panel. With it, the recorded 3 tracker points are
    # completed to a 29-DoF whole-body motion ONLINE (one window per play-head step, in a
    # Motion_Completion backend under its own conda env) and drawn as this rig's own
    # g1_29dof_yam.urdf. See replay/motion_completion_overlay.py.
    p.add_argument("--motion-completion", action="store_true",
                   help="also draw the G1 whole-body motion completed online from the "
                        "recorded 3pt tracker stream (needs a Motion_Completion checkout, "
                        "its g1_mosyn conda env and a downloaded checkpoint; not part of the FLoRe release)")
    p.add_argument("--mc-urdf", default=None,
                   help="URDF for the completed humanoid; default is this repo's "
                        "robot_assets/g1_yam/g1_29dof_yam.urdf (yam gripper + F/T sensor), "
                        "NOT Motion_Completion's assembly.urdf")
    # Initial ABSOLUTE PLACEMENT, for the same reason as the post-process toggles below: the
    # placement is the single biggest thing to compare (commanded hand height is a CONDITIONED
    # channel, so it decides the posture), and comparing it means several viewers side by side
    # — which a clicked-together dropdown cannot reproduce. Defaults are the panel's own, so
    # an existing launch is unchanged.
    p.add_argument("--mc-cond-mode", default="home", choices=("home", "absolute"),
                   help="where the completion STARTS. 'home' (default) re-bases the take onto "
                        "the G1's 29-joints-zero pose and needs no guessed origin height; "
                        "'absolute' keeps the take's world poses and needs the z lift below "
                        "(the fidelity view — hands back on the recorded eef path)")
    p.add_argument("--mc-dz-anchor", default="chest", choices=("chest", "hand", "train"),
                   help="how the 'absolute' z lift is derived; ignored for --mc-cond-mode home. "
                        "'chest' (default) normalises against the operator's standing chest "
                        "height, 'hand' against the hands, 'train' against the checkpoint's own "
                        "training band (measured: that one commands 0.645 m and the model "
                        "answers by CROUCHING to pelvis 0.451)")
    # Initial state of the two post-process toggles, so an A/B is two reproducible launches
    # rather than a click sequence. Both are also switchable live in the panel.
    p.add_argument("--mc-no-foot-ik", action="store_true",
                   help="start with the causal foot IK OFF (feet slide ~0.031 m/s, but the "
                        "window cost drops from ~1800 ms to ~32 ms)")
    p.add_argument("--mc-no-wrist-ik", action="store_true",
                   help="start with the per-window wrist IK OFF (raw flow hands)")
    p.add_argument("--mc-foot-ik-iters", type=int, default=None,
                   help="leg-IK gradient steps in the foot pass; below ~80 it does not "
                        "converge and buys nothing (default 150, MC's own)")
    args = p.parse_args(argv if argv is not None else sys.argv[1:])

    if not os.path.exists(args.zarr_path):
        raise SystemExit(f"[replay-zarr] not found: {args.zarr_path}")
    z = ZarrEpisodes(args.zarr_path)
    if z.n_ep == 0:
        raise SystemExit("[replay-zarr] no episodes (empty episode_ends)")
    # Clamp BOTH ends: min() alone lets a negative --episode index from the back.
    ep0 = max(0, min(args.episode, z.n_ep - 1))
    print(f"[replay-zarr] {os.path.basename(args.zarr_path)}: {z.n_ep} episodes, "
          f"lengths {z.episode_lengths()}")
    if z.attrs.get("WARNING"):
        print(f"[replay-zarr] ATTR WARNING: {z.attrs['WARNING']}")
    if not z.attrs.get("tcp_calibrated", True):
        print("[replay-zarr] NOTE: tcp_calibrated=False — eef poses are RAW tracker, not TCP.")

    # The stored eef pose corresponds to whatever point the converter's extrinsic maps
    # to (the +0.21 TCP grasp point for current datasets; a brief experiment used a
    # tracker_offset point instead). Read it from the dataset's own provenance so the
    # gripper mesh + triads sit right for EITHER; fall back to identity if absent.
    T_ext = np.asarray(z.attrs.get("T_tracker_tcp", np.eye(4).tolist()), dtype=np.float64)
    if T_ext.shape != (4, 4):
        T_ext = np.eye(4)
    if not np.allclose(T_ext[:3, :3], np.eye(3), atol=1e-6):
        # Expected for datasets built with the CURRENT extrinsic (the puck body axes differ
        # from the wrist axes by the measured -y,-x,-z relabel). Handled exactly by the
        # full 4x4 inverse in wrist_pose_from_stored — not "slightly off".
        print("[replay-zarr] extrinsic has a rotation (puck axes != wrist axes) — handled "
              "exactly by the wrist-pose inverse.")
    # wrist pose (offset AND rotation) relative to the stored eef point
    wrist_off, wrist_q = wrist_pose_from_stored(T_ext)
    # Triad offsets are WRIST-relative (same as collection_viz parents them under
    # /tracker). They are drawn as children of /eef/wrist — which is ALREADY positioned
    # at wrist_off — so we use the RAW wrist-relative offsets here, exactly like the
    # gripper meshes and camera (also children of /eef/wrist with raw offsets). Adding
    # wrist_off here too would double-count it and float the triads ~3 cm off the mesh.
    tri_tcp = np.asarray([_TCP_ALONG_WRIST_X_M, 0.0, 0.0])
    tri_trk = np.asarray(_TRACKER_TRUE_XYZ_M)
    tri_wrist = np.asarray(_WRIST_YAW_XYZ_M)  # (0,0,0): the wrist-yaw link origin itself

    # --- scene ---
    server = viser.ViserServer(host=args.host, port=args.port, label="UMI zarr replay")
    server.scene.set_up_direction("+z")
    server.scene.add_grid("/grid", plane="xy", shadow_opacity=0.0, infinite_grid=True, cell_size=0.1)

    # /eef = the exact stored pose (moved per-frame). Its children reproduce the
    # collection-viz cluster: a wrist child carrying the three yam-gripper meshes, the
    # three axis triads (TCP, tracker, wrist_yaw_link) and NOTHING ELSE, and the camera.
    eef = server.scene.add_frame("/eef", show_axes=False, origin_radius=0.0)
    # Carries BOTH the offset and the ROTATION relative to the stored eef (identity for
    # the current extrinsic, but composed explicitly — see wrist_pose_from_stored).
    wrist = server.scene.add_frame("/eef/wrist", show_axes=False,
                                   position=tuple(wrist_off), wxyz=tuple(wrist_q),
                                   origin_radius=0.0)
    # The three requested triads (and only these), children of the wrist frame so they
    # ride the stored pose rigidly, exactly like collection_viz.
    server.scene.add_frame("/eef/wrist/tcp", show_axes=True, axes_length=0.08,
                           axes_radius=0.004, origin_radius=0.01, position=tuple(tri_tcp))
    server.scene.add_label("/eef/wrist/tcp/label", "TCP")
    # The puck triad needs its OWN rotation (R_ext^-1): puck axes != wrist axes. Drawn
    # unrotated it inherited the wrist axes and read identical to the TCP triad.
    server.scene.add_frame("/eef/wrist/tracker", show_axes=True, axes_length=0.07,
                           axes_radius=0.0035, origin_radius=0.009, position=tuple(tri_trk),
                           wxyz=tuple(puck_triad_local_quat(T_ext)))
    server.scene.add_label("/eef/wrist/tracker/label", "tracker")
    server.scene.add_frame("/eef/wrist/wrist_yaw", show_axes=True, axes_length=0.07,
                           axes_radius=0.0035, origin_radius=0.009, position=tuple(tri_wrist))
    server.scene.add_label("/eef/wrist/wrist_yaw/label", "wrist_yaw_link")

    # Three yam-gripper meshes (base + 2 fingers) under the wrist frame at their
    # wrist-relative poses (URDF colours baked in), same as collection_viz gripper mode.
    grip_geom = _gov.build_gripper_geom("left")
    grip_nodes = {}
    for piece in grip_geom.pieces():
        grip_nodes[piece.name] = server.scene.add_mesh_trimesh(
            f"/eef/wrist/grip_{piece.name}", piece.mesh,
            position=tuple(piece.pos_closed), wxyz=tuple(piece.quat))

    # Camera frustum on the wrist, composed Rx(roll)·Ry(yaw) so +z points OUTWARD along
    # wrist +x — identical to collection_viz / replay_raw_viz (fixes the ~90° roll).
    cam_rot = (vtf.SO3.from_rpy_radians(_CAM_RPY[0], 0.0, 0.0)
               .multiply(vtf.SO3.from_rpy_radians(0.0, _CAM_RPY[1], 0.0)))
    server.scene.add_frame("/eef/wrist/cam", show_axes=False, position=tuple(_CAM_OFFSET_XYZ),
                           wxyz=tuple(cam_rot.wxyz))
    frustum = server.scene.add_camera_frustum(
        "/eef/wrist/cam/view", fov=np.deg2rad(args.cam_fov_deg), aspect=1.0, scale=0.12,
        color=(40, 200, 255))
    # Wrench arrows: WORLD-ROOT node (NOT a child of /eef). WrenchArrows.update() is
    # handed the WORLD-space origin (disp_pos) + world-space arrow endpoints it builds
    # itself; parenting under /eef — which render_at() drives to (disp_pos, disp_wxyz)
    # every frame — would re-apply that transform to already-world coords, floating the
    # cones a whole home-anchor offset up-and-forward off the TCP. Root parenting lands
    # the origin exactly on disp_pos (== the TCP triad), matching replay_raw_viz's /wrench.
    wrench = rc.WrenchArrows(server, name="/wrench") if z.has_ft else None

    # ===== RIGHT hand (robot1_*) + CHEST (head_*) ==============================
    # Built ONLY from what the file contains. The right hand gets the SAME cluster as
    # the left — wrist frame, 3 yam meshes, TCP/tracker/wrist_yaw triads, camera frustum
    # — because it is the same physical hand assembly and the point of this viewer is
    # that a converted right hand must look exactly as trustworthy as the left. The
    # chest gets a TRIAD ONLY: it is a body reference with no TCP extrinsic, no camera,
    # no F/T and no gripper, so drawing a gripper there would invent geometry.
    eef1 = grip1_geom = frustum1 = wrench1 = None
    grip1_nodes: dict = {}
    chest = None
    if z.has_robot1:
        eef1 = server.scene.add_frame("/eef1", show_axes=False, origin_radius=0.0)
        # Same wrist offset/rotation as the left: robot1_eef_pos is produced by the SAME
        # tracker->TCP extrinsic (DECISION-D, FK-symmetric wrists), so the cluster
        # geometry is identical and must not be re-derived per hand.
        server.scene.add_frame("/eef1/wrist", show_axes=False, position=tuple(wrist_off),
                               wxyz=tuple(wrist_q), origin_radius=0.0)
        server.scene.add_frame("/eef1/wrist/tcp", show_axes=True, axes_length=0.08,
                               axes_radius=0.004, origin_radius=0.01, position=tuple(tri_tcp))
        server.scene.add_label("/eef1/wrist/tcp/label", "TCP R")
        server.scene.add_frame("/eef1/wrist/tracker", show_axes=True, axes_length=0.07,
                               axes_radius=0.0035, origin_radius=0.009, position=tuple(tri_trk),
                               wxyz=tuple(puck_triad_local_quat(T_ext)))
        server.scene.add_label("/eef1/wrist/tracker/label", "tracker R")
        server.scene.add_frame("/eef1/wrist/wrist_yaw", show_axes=True, axes_length=0.07,
                               axes_radius=0.0035, origin_radius=0.009, position=tuple(tri_wrist))
        grip1_geom = _gov.build_gripper_geom("right")
        for piece in grip1_geom.pieces():
            grip1_nodes[piece.name] = server.scene.add_mesh_trimesh(
                f"/eef1/wrist/grip_{piece.name}", piece.mesh,
                position=tuple(piece.pos_closed), wxyz=tuple(piece.quat))
        if z.has_cam1:
            server.scene.add_frame("/eef1/wrist/cam", show_axes=False,
                                   position=tuple(_CAM_OFFSET_XYZ), wxyz=tuple(cam_rot.wxyz))
            frustum1 = server.scene.add_camera_frustum(
                "/eef1/wrist/cam/view", fov=np.deg2rad(args.cam_fov_deg), aspect=1.0,
                scale=0.12, color=(255, 170, 40))   # orange: never confuse it with camera L
            server.scene.add_label("/eef1/wrist/cam/view/label", "camera R")
            server.scene.add_label("/eef/wrist/cam/view/label", "camera L")
        if z.has_ft1:
            # Own world-root node, same reasoning as /wrench (see above): the arrows are
            # built in WORLD space, so parenting under /eef1 would re-apply its transform.
            wrench1 = rc.WrenchArrows(server, name="/wrench1")
    if z.has_head:
        chest = server.scene.add_frame("/chest", show_axes=True, axes_length=0.09,
                                       axes_radius=0.0045, origin_radius=0.012)
        server.scene.add_label("/chest/label", "chest (head_*)")

    gui = server.gui
    gui.configure_theme(control_layout="collapsible", control_width="large")

    # --- episode picker (clickable, in the browser) ---
    with gui.add_folder("Dataset"):
        gui.add_text("File", initial_value=os.path.basename(args.zarr_path), disabled=True)
        gui.add_text("Episodes", initial_value=str(z.n_ep), disabled=True)
        gui.add_text("TCP calibrated", initial_value=str(z.attrs.get("tcp_calibrated", "?")),
                     disabled=True)
        # WHAT this file actually contains, spelled out. The viewer draws only what is
        # present, so this row is how the operator distinguishes "the right hand isn't
        # drawn because the file has no right hand" from "the right hand isn't drawn
        # because the viewer is broken".
        _kp = ["left(pose+cam+ft+grip)"]
        if z.has_robot1:
            _kp.append("right(pose"
                       + ("+cam" if z.has_cam1 else "")
                       + ("+ft" if z.has_ft1 else "")
                       + ("+grip" if z.has_grip1 else "") + ")")
        if z.has_head:
            _kp.append("chest(pose)")
        gui.add_text("Keypoints", initial_value=" · ".join(_kp), disabled=True)
        if z.has_cam1 and z.attrs.get("camera1_pair_skew_max_s") is not None:
            # How well camera1 lines up with the master grid it was paired onto. A large
            # value means the two drawn views are showing different instants.
            gui.add_text("camera R pair skew",
                         initial_value=f"{float(z.attrs['camera1_pair_skew_max_s'])*1e3:+.0f} ms max"
                                       f"  (reuse x{z.attrs.get('camera1_pair_reused_max', '?')})",
                         disabled=True)
        ep_options = [f"ep {i}  ({n} fr)" for i, n in enumerate(z.episode_lengths())]
        ep_dd = gui.add_dropdown("Episode", ep_options, initial_value=ep_options[ep0])
    with gui.add_folder("Frame readout" + (" — LEFT" if z.has_robot1 else "")):
        rd_t = gui.add_text("t (s, rel)", initial_value="0.000", disabled=True)
        rd_pos = gui.add_text("eef xyz", initial_value="-", disabled=True)
        rd_grip = gui.add_text("gripper width", initial_value="-", disabled=True)
        rd_force = gui.add_text("|F| (N)", initial_value="-", disabled=True)
        rd_torque = gui.add_text("|T| (Nm)", initial_value="-", disabled=True)
    # Per-keypoint readouts, and ONLY for what the file has. A row that always exists and
    # always says "-" is how a missing modality stops being noticed; a row that is absent
    # says "this dataset does not contain it" without anyone having to interpret a dash.
    rd1_pos = rd1_grip = rd1_force = rd_chest = None
    if z.has_robot1:
        with gui.add_folder("Frame readout — RIGHT"):
            rd1_pos = gui.add_text("eef xyz", initial_value="-", disabled=True)
            if z.has_grip1:
                rd1_grip = gui.add_text("gripper width", initial_value="-", disabled=True)
            if z.has_ft1:
                rd1_force = gui.add_text("|F| (N)", initial_value="-", disabled=True)
    if z.has_head:
        with gui.add_folder("Frame readout — CHEST"):
            rd_chest = gui.add_text("head xyz", initial_value="-", disabled=True)
    with gui.add_folder("View"):
        show_cam = gui.add_checkbox("Show camera", initial_value=True)
        show_path = gui.add_checkbox("Show eef path", initial_value=True)
        # ARROWS ARE F^cmd, CHARTS ARE F_meas — deliberately, and stated because the pair
        # would otherwise look like a bug. The arrow answers a DEPLOY question ("which way
        # would the robot push to reproduce this?") and so carries WrenchArrows' meas->cmd
        # negation; the charts answer a DATA question and stay the stored `robot0_ft_wrench`
        # verbatim, which is what makes this viewer a fidelity check. Same in replay_raw_viz.
        show_wrench = gui.add_checkbox("Show F^cmd arrows (= -F_meas)", initial_value=bool(z.has_ft))
        gui.add_text("arrows vs charts",
                     initial_value="arrows = F^cmd (negated, deploy sign) · charts = stored F_meas",
                     disabled=True)
        show_mesh = gui.add_checkbox("Show gripper mesh", initial_value=True)
        # Home-anchored view — default OFF, i.e. the RAW STORED pose, matching
        # RAW_FRAME=1 collect_data.sh and replay_raw_viz. That makes this viewer directly
        # comparable to the raw-take viewer frame-for-frame (same convention, same axes),
        # which is the whole point of having both. Turning it ON re-bases every frame onto
        # an upright home — a world rotation the DATA does not contain; convenient to look
        # at, but it must never be used to judge fidelity. DISPLAY-ONLY either way: the
        # stored arrays and the numeric readouts below are untouched.
        home_anchor_view = gui.add_checkbox("Home-anchored view (NOT the stored frame)",
                                            initial_value=False)

        @home_anchor_view.on_update
        def _(_e) -> None:
            # Flag a rebuild: the loop thread rebuilds the static scene (the path
            # polyline depends on the anchor mode) and forces a frame re-render, WITHOUT
            # resetting the timeline cursor. Consumed in poll_pending on the loop thread.
            st["rebuild"] = True

    # charts are (re)filled per episode
    with gui.add_folder("Charts"):
        grip_chart = rc.make_single_chart(gui, "gripper_width (m)", "width", color_idx=1)
        if z.has_ft:
            force_chart = rc.make_xyz_chart(gui, "ft_wrench force (N)", ("Fx", "Fy", "Fz"))
            torque_chart = rc.make_xyz_chart(gui, "ft_wrench torque (Nm)", ("Tx", "Ty", "Tz"))

    # Force-recovery funnel overlay (purple). Built AFTER the charts so its panel sits at the
    # bottom of the sidebar, and it is entirely additive: with no FLoRe checkout to import the
    # funnel from, the panel says so and nothing else in this viewer changes.
    # Online motion completion (opt-in). Constructed unconditionally so the call sites below
    # need no branching, but with --motion-completion absent it is inert: `available` is
    # False, `build_gui` adds nothing, and every hook returns immediately. Built BEFORE the
    # funnel panel's GUI so the (much longer) startup log lands before the "ready" line.
    mc = MotionCompletionOverlay(server, enabled=bool(args.motion_completion),
                                 urdf_path=args.mc_urdf,
                                 wrist_ik=not args.mc_no_wrist_ik,
                                 foot_ik=not args.mc_no_foot_ik,
                                 foot_ik_iters=args.mc_foot_ik_iters,
                                 init_cond_mode=args.mc_cond_mode,
                                 init_dz_anchor=args.mc_dz_anchor,
                                 log=lambda m: print(f"[replay-zarr][mc] {m}", flush=True))
    if args.motion_completion and not mc.available:
        print(f"[replay-zarr] motion completion DISABLED: {mc.disabled_reason}")
    mc.build_gui(gui)

    aug = ForceRecoveryOverlay(server, horizon=z.hl_horizon or 50, dt=z.hl_dt)
    aug.build_gui(gui)
    if aug.available and not z.has_hl_force:
        print("[replay-zarr] force-recovery overlay: this dataset has no hl_*_action_force "
              "arrays (attrs.hl_windows unset), so Generate will report 'no usable window'.")
    elif not aug.available:
        print(f"[replay-zarr] force-recovery overlay DISABLED: {aug.disabled_reason}")

    # mutable episode state. ONE Timeline is reused across episodes (retargeted via
    # set_series); the dropdown callback only sets pending_ep, applied by the loop.
    st: dict = {"ep": ep0, "data": None, "last_cam_i": -1, "last_cam1_i": -1, "pending_ep": None,
                "rebuild": False}

    def display_poses(d: dict, pos=None, wxyz=None,
                      role: str = "left") -> tuple[np.ndarray, np.ndarray]:
        """Return (pos[T,3], wxyz[T,4]) to DRAW, honouring the home-anchor toggle.

        OFF -> the raw stored absolute world-aligned poses (strict fidelity check).
        ON  -> re-based onto an upright home per gripper_only_view.home_anchor (the SAME
        math the collection viz uses), ref = that keypoint's OWN first frame:
            R_disp = (R_cur·R_ref⁻¹)·R_home ,  p_disp = p_home + (p_cur − p_ref)
        so the whole trajectory rides an upright, z-up home like collection — display
        only, the underlying stored arrays and the readouts are untouched.

        PER-ROLE home + PER-ROLE reference. ``role`` picks the home out of
        HOME_WRIST_POSE, so with the anchor ON each keypoint starts the episode exactly
        at the G1's ZERO-CONFIGURATION pose for that keypoint: the two hands parallel,
        0.297 m apart, the chest at the torso point. That is the same anchor the teleop
        bridge and the collection viz apply per hand, so a home-anchored replay shows the
        rig starting where the robot starts.

        It therefore does NOT preserve the operator's measured hand-to-hand offset — each
        hand is independently re-zeroed, so only the CHANGE in their relative pose
        survives. That is not a loss: the true measured geometry is exactly what the
        anchor-OFF default draws, and OFF is the fidelity check. Anchoring all keypoints
        to the LEFT hand's reference (what this used to do) preserved the offset but then
        no hand starts at its own home, which is the property the operator needs to see."""
        pos = d["pos"] if pos is None else pos
        wxyz = d["wxyz"] if wxyz is None else wxyz
        if not home_anchor_view.value or pos.shape[0] == 0:
            return pos, wxyz
        home_p, home_q = _HOME_BY_ROLE.get(role, _HOME_BY_ROLE["left"])
        ref_p, ref_q = pos[0], wxyz[0]
        dp = np.empty_like(pos)
        dq = np.empty_like(wxyz)
        for k in range(pos.shape[0]):
            tp, tq = _gov.home_anchor(home_p, home_q, ref_p, ref_q, pos[k], wxyz[k])
            dp[k], dq[k] = tp, tq
        return dp, dq

    def load_episode(ep: int) -> dict:
        d = {
            "pos": z.slice(ep, "robot0_eef_pos").astype(np.float64),
            "rot": z.slice(ep, "robot0_eef_rot_axis_angle").astype(np.float64),
            "width": z.slice(ep, "robot0_gripper_width").reshape(-1).astype(np.float64),
            "ts": z.slice(ep, "episode_timestamps").reshape(-1).astype(np.float64),
            "rgb": z.slice(ep, "camera0_rgb"),
        }
        d["wxyz"] = rotvec_to_wxyz(d["rot"])
        d["ft"] = z.slice(ep, "robot0_ft_wrench").astype(np.float64) if z.has_ft else None
        # ---- secondary keypoints (each independently optional) -------------------
        if z.has_robot1:
            d["pos1"] = z.slice(ep, "robot1_eef_pos").astype(np.float64)
            d["wxyz1"] = rotvec_to_wxyz(z.slice(ep, "robot1_eef_rot_axis_angle").astype(np.float64))
            d["rgb1"] = z.slice(ep, "camera1_rgb") if z.has_cam1 else None
            d["ft1"] = z.slice(ep, "robot1_ft_wrench").astype(np.float64) if z.has_ft1 else None
            d["width1"] = (z.slice(ep, "robot1_gripper_width").reshape(-1).astype(np.float64)
                           if z.has_grip1 else None)
        if z.has_head:
            d["head_pos"] = z.slice(ep, "head_pos").astype(np.float64)
            d["head_wxyz"] = rotvec_to_wxyz(z.slice(ep, "head_rot_axis_angle").astype(np.float64))
        # Display poses (raw or home-anchored). Recomputed on episode load AND on a
        # rebuild (the toggle) so the path + per-frame render share one coherent array.
        d["disp_pos"], d["disp_wxyz"] = display_poses(d)
        # Each secondary point is rebased onto ITS OWN home with ITS OWN first frame as
        # the reference — so with the anchor ON the episode starts in the G1's zero
        # configuration (hands parallel, 0.297 m apart; chest at the torso point), which
        # is what the operator is asked to start from and what the teleop bridge does per
        # hand. The operator's measured hand-to-hand offset is therefore NOT preserved
        # here; anchor OFF (the default) is what shows the true stored geometry, and that
        # is the fidelity check. A single shared reference would preserve the offset but
        # then no hand starts at its own home.
        if z.has_robot1:
            d["disp_pos1"], d["disp_wxyz1"] = display_poses(d, d["pos1"], d["wxyz1"],
                                                            role="right")
        if z.has_head:
            d["disp_head_pos"], d["disp_head_wxyz"] = display_poses(
                d, d["head_pos"], d["head_wxyz"], role="head")
        return d

    def set_static_scene(d: dict) -> None:
        # full eef path — GREEN (the GT trajectory colour). Uses the DISPLAY poses so it
        # matches the drawn frames (raw or home-anchored). Drawn with add_line_segments
        # (the EXACT stored polyline, one segment per stored step) NOT a Catmull-Rom
        # spline: viser forwards segments=None -> drei's default 20 -> three's
        # getPoints(20) would redraw the whole episode as just 21 vertices, both hiding
        # any conversion kink (this is the FIDELITY checker — a kink MUST show) and
        # flattening the curve. Exact segments keep every stored point.
        pos = d["disp_pos"]
        if show_path.value and pos.shape[0] >= 2:
            p = pos.astype(np.float32)
            segs = np.stack([p[:-1], p[1:]], axis=1)  # (T-1, 2, 3)
            server.scene.add_line_segments("/eef_path", points=segs, colors=_GT_GREEN,
                                           line_width=2.0)
        else:
            zp = (pos[:1] if pos.shape[0] else np.zeros((1, 3))).astype(np.float32)
            server.scene.add_line_segments("/eef_path", points=np.stack([zp, zp], axis=1),
                                           colors=_GT_GREEN, line_width=2.0, visible=False)
        # Secondary paths, same exact-polyline rule as the left (one segment per stored
        # step, no spline): a conversion kink on the RIGHT hand has to be as visible as
        # one on the left. Distinct colours so the three keypoints read apart at a glance.
        for _key, _name, _col, _present in (
            ("disp_pos1", "/eef1_path", _R_ORANGE, eef1 is not None),
            ("disp_head_pos", "/chest_path", _CHEST_VIOLET, chest is not None),
        ):
            p2 = d.get(_key)
            if show_path.value and _present and p2 is not None and p2.shape[0] >= 2:
                q = p2.astype(np.float32)
                server.scene.add_line_segments(_name, points=np.stack([q[:-1], q[1:]], axis=1),
                                               colors=_col, line_width=2.0)
            elif _present:
                zq = (p2[:1] if (p2 is not None and p2.shape[0]) else np.zeros((1, 3))).astype(np.float32)
                server.scene.add_line_segments(_name, points=np.stack([zq, zq], axis=1),
                                               colors=_col, line_width=2.0, visible=False)
        # NOTE: no demo start/end triads — the user asked for EXACTLY the three axis
        # triads (TCP, tracker, wrist_yaw_link), nothing else.
        # charts (whole episode at once)
        rc.set_full_single(grip_chart, d["ts"], d["width"])
        if z.has_ft and d["ft"] is not None:
            rc.set_full_xyz(force_chart, d["ts"], d["ft"][:, 0:3])
            rc.set_full_xyz(torque_chart, d["ts"], d["ft"][:, 3:6])

    def render_at(i: int) -> None:
        d = st["data"]
        # Drive the /eef frame to the DISPLAY pose (raw or home-anchored); the wrist/
        # mesh/triad/camera cluster rides it rigidly (children of /eef/wrist).
        eef.position = tuple(d["disp_pos"][i])
        eef.wxyz = tuple(d["disp_wxyz"][i])
        # Animate the jaws from the stored absolute gripper width (LERP fingers).
        for piece in grip_nodes:
            node = grip_nodes[piece]
            node.visible = show_mesh.value
        if show_mesh.value:
            frac = float(np.clip(d["width"][i] / _WIDTH_OPEN_M, 0.0, 1.0))
            for pc in (grip_geom.finger1, grip_geom.finger2):
                grip_nodes[pc.name].position = tuple(pc.pos_at(frac))
        frustum.visible = show_cam.value
        if show_cam.value and i != st["last_cam_i"]:
            img = d["rgb"][i]
            h, w = img.shape[:2]
            frustum.aspect = w / max(h, 1)
            frustum.image = np.asarray(img)
            st["last_cam_i"] = i
        if wrench is not None:
            wrench.set_visible(show_wrench.value)
            if show_wrench.value and d["ft"] is not None:
                # Ride the DISPLAY pose so the arrows track the drawn gripper. The RAW
                # sensor-frame wrench goes in as-is: WrenchArrows.update applies the measured
                # sensor->TCP relabel AND the shown orientation, so pre-rotating here would
                # apply the relabel twice.
                wrench.update(d["disp_pos"][i], d["disp_wxyz"][i], d["ft"][i, 0:3], d["ft"][i, 3:6])
        # ---- RIGHT hand + CHEST, driven from the same frame index -----------------
        # Every one of these is gated on its own node being present, so a file with a
        # right pose but no right camera draws the right cluster with no frustum, which
        # is exactly what that dataset contains.
        if eef1 is not None:
            eef1.position = tuple(d["disp_pos1"][i])
            eef1.wxyz = tuple(d["disp_wxyz1"][i])
            for _n in grip1_nodes.values():
                _n.visible = show_mesh.value
            if show_mesh.value:
                # Jaws from the RIGHT hand's own width when the dataset has one; a hand
                # with no recorded gripper holds CLOSED rather than mirroring the left,
                # so the drawing can never imply a grasp nothing measured.
                frac1 = (float(np.clip(d["width1"][i] / _WIDTH_OPEN_M, 0.0, 1.0))
                         if d.get("width1") is not None else 0.0)
                for pc in (grip1_geom.finger1, grip1_geom.finger2):
                    grip1_nodes[pc.name].position = tuple(pc.pos_at(frac1))
            if frustum1 is not None:
                frustum1.visible = show_cam.value
                if show_cam.value and i != st.get("last_cam1_i"):
                    img1 = d["rgb1"][i]
                    h1, w1 = img1.shape[:2]
                    frustum1.aspect = w1 / max(h1, 1)
                    frustum1.image = np.asarray(img1)
                    st["last_cam1_i"] = i
            if wrench1 is not None and d.get("ft1") is not None:
                wrench1.set_visible(show_wrench.value)
                if show_wrench.value:
                    wrench1.update(d["disp_pos1"][i], d["disp_wxyz1"][i],
                                   d["ft1"][i, 0:3], d["ft1"][i, 3:6])
        if chest is not None:
            chest.position = tuple(d["disp_head_pos"][i])
            chest.wxyz = tuple(d["disp_head_wxyz"][i])
        # The purple recovery arrow rides the same frame index; it hides itself outside the
        # span of whatever chunk was last generated.
        aug.render_at(i)
        # The completed humanoid rides it too (no-op unless --motion-completion). It draws
        # only frames whose window has already been run, so early in playback it simply
        # holds — that hold IS the online behaviour, not a stall.
        mc.render_at(i)
        rd_t.value = f"{d['ts'][i]:.3f}"
        # Readout is the RAW stored eef pos (what the trainer sees), independent of the
        # home-anchor DISPLAY toggle — so the numbers stay a faithful data readout.
        rd_pos.value = np.array2string(d["pos"][i], precision=3, suppress_small=True)
        gw = float(d["width"][i])
        rd_grip.value = f"{gw*1000:.1f} mm ({'grasp' if gw < _GRASP_THRESH_M else 'open'})"
        if d["ft"] is not None:
            rd_force.value = f"{np.linalg.norm(d['ft'][i,0:3]):.2f}"
            rd_torque.value = f"{np.linalg.norm(d['ft'][i,3:6]):.3f}"
        # Secondary readouts: RAW stored values (what the trainer sees), independent of
        # the home-anchor DISPLAY toggle — same rule as the left hand above.
        if rd1_pos is not None:
            rd1_pos.value = np.array2string(d["pos1"][i], precision=3, suppress_small=True)
        if rd1_grip is not None:
            gw1 = float(d["width1"][i])
            rd1_grip.value = f"{gw1*1000:.1f} mm ({'grasp' if gw1 < _GRASP_THRESH_M else 'open'})"
        if rd1_force is not None:
            rd1_force.value = f"{np.linalg.norm(d['ft1'][i,0:3]):.2f}"
        if rd_chest is not None:
            rd_chest.value = np.array2string(d["head_pos"][i], precision=3, suppress_small=True)

    def apply_episode(ep: int, timeline) -> None:
        """Load + show episode ``ep`` and retarget the shared timeline. Runs on the
        loop thread (the only thread that calls render_at), so there is no race
        with playback."""
        st["ep"] = ep
        st["data"] = load_episode(ep)
        st["last_cam_i"] = -1
        st["last_cam1_i"] = -1   # force a right-camera repaint on the new episode too
        set_static_scene(st["data"])
        # Drop any generated funnel: it was anchored to a pose in the OLD episode, and leaving
        # it on screen would silently attribute one episode's wrench plan to another's frames.
        aug.clear()
        # Same reasoning for the humanoid: its completion belongs to the old episode's
        # tracker stream. clear() then on_episode() resets the streaming state so the new
        # episode starts from window 0 with a fresh episode canon.
        mc.clear()
        mc.on_episode(z, ep)
        timeline.set_series(st["data"]["ts"])

    @ep_dd.on_update
    def _(_e) -> None:
        # GUI callback thread: only record the request. The loop thread applies it
        # (loads arrays + rebuilds scene + retargets the timeline) so episode data
        # is never swapped underneath an in-flight render.
        st["pending_ep"] = ep_options.index(ep_dd.value)

    print(f"[replay-zarr] http://{args.host}:{args.port}  — pick an Episode, drag Time to scrub",
          flush=True)
    st["data"] = load_episode(st["ep"])
    set_static_scene(st["data"])
    mc.on_episode(z, st["ep"])
    timeline = rc.Timeline(server, st["data"]["ts"], render_at, folder_label="Playback")

    # Apply a pending episode switch at the top of each loop iteration, ON THE LOOP
    # THREAD (the only thread that renders), so episode arrays are never swapped
    # mid-render. The dropdown callback merely sets st["pending_ep"]; we consume it
    # here exactly once.
    def poll_pending() -> None:
        # Home-anchor toggle: recompute display poses for the CURRENT episode + rebuild
        # the path polyline, without touching the timeline cursor. Force a re-render so
        # a paused frame repaints immediately. Consumed before the episode switch.
        if st["rebuild"]:
            st["rebuild"] = False
            d = st["data"]
            d["disp_pos"], d["disp_wxyz"] = display_poses(d)
            set_static_scene(d)
            timeline.request_render()
        # Force-recovery funnel: the button/sliders only set a flag, so the generate + redraw
        # happen HERE, on the loop thread, like every other scene mutation in this file. The
        # chunk is read at the cursor's frame and pinned to that frame's DISPLAY pose, so the
        # bundle stays put while the timeline scrubs through it.
        if aug.available:
            d = st["data"]
            # timeline.current_frame(), not the last RENDERED frame: before_render runs before
            # run() computes the index, so a scrub immediately followed by Generate would
            # otherwise anchor one frame behind where the user is looking.
            i = int(np.clip(timeline.current_frame(), 0, max(d["disp_pos"].shape[0] - 1, 0)))
            if aug.poll(ep=st["ep"], frame=i,
                        demo_chunk=z.action_force_chunk(st["ep"], i),
                        anchor_pos=d["disp_pos"][i], anchor_wxyz=d["disp_wxyz"][i],
                        # The RED arrow's own source, so the panel can report how far the
                        # funnel's target sits from it (two resamplings of one wrench).
                        stored_wrench=None if d["ft"] is None else d["ft"][i]):
                timeline.request_render()

        # The online motion-completion loop: hand it the cursor, it schedules the windows the
        # play head has reached and absorbs the ones that finished. On the loop thread like
        # every other scene mutation here; the model call itself is on the overlay's own
        # worker, so this returns in microseconds. No-op without --motion-completion.
        if mc.poll(timeline.current_frame()):
            timeline.request_render()

        ep = st["pending_ep"]
        if ep is None:
            return
        st["pending_ep"] = None
        if ep != st["ep"]:
            apply_episode(ep, timeline)

    timeline.before_render = poll_pending
    try:
        timeline.run()
    finally:
        # Reap the backend process. Without this a Ctrl-C leaves a GPU-resident python
        # holding ~2 GB of VRAM until the shell session ends.
        mc.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
