#!/usr/bin/env python3
"""Replay a CONVERTED LeRobot v2.1 dataset (Psi0/GR00T layout) in viser — the
LeRobot-export viewer, the THIRD replay tool alongside ``replay_raw_viz.py`` (raw
take) and ``replay_zarr_viz.py`` (UMI zarr).

Like ``replay_raw_viz.py`` (and UNLIKE the deliberately-no-IK ``replay_zarr_viz.py``),
this tool puts the FULL G1 HUMANOID back in the scene: it drives the SAME mink IK
the live viz uses (``viz/g1_right_hand_ik.py``) and mounts the recorded camera on
the REAL forward-kinematics wrist (``right_wrist_yaw_link``). That matters for the
camera image: the wrist-mounted frustum's orientation is taken from the URDF link,
so the picture sits UPRIGHT relative to the ground — mounting the frustum on the
bare TCP frame (as an earlier version did) tilts the image, because the camera
mount offset is calibrated relative to the wrist link, not the TCP.

The dataset stores the **TCP** pose (``states`` cols eef_pos|eef_rot =
``world_T_tcp``). To drive the home-anchored IK exactly as the raw-take replay
does, we recover the tracker world pose by INVERTING the TCP extrinsic
(``world_T_tracker = world_T_tcp @ inv(T_tracker_tcp)`` — the exact inverse of
``umi_convert_core.apply_tcp_extrinsic``), then run the identical IK. The converter
EMBEDS the exact extrinsic matrix it used in ``info.json`` (``tcp_extrinsic``), so
the recovery is EXACT even for a custom (non-repo-default) extrinsic. Older
datasets without that field fall back to the repo-default
``robot_assets/calibration/tcp_extrinsic.json`` (loudly warning if even that can't
be resolved while ``tcp_calibrated`` is True); ``tcp_calibrated=False`` means an
identity extrinsic was used so TCP == tracker.

It shows the data EXACTLY as it lives on disk otherwise (parquet ``states``/
``action`` + the h264 ``observation.images.egocentric`` mp4), one selectable
episode at a time, with a clickable dropdown. The IK is re-solved per episode on
the loop thread (the only thread that touches the mink ``Configuration``, so the
live-viz threading invariant is preserved). Pass ``--no-ik`` to skip the robot and
show only the stored TCP triad + camera + F/T (the old behavior, no humanoid).

The dataset layout it reads (written by ``processing/convert_to_lerobot.py``):

  meta/info.json            fps, features, tcp_calibrated + provenance flags
  meta/episodes.jsonl       per-episode length + instruction
  meta/modality.json        state/action sub-slices (eef_pos|eef_rot|gripper|ft)
  data/chunk-*/episode_*.parquet   states[13], action[7], timestamp, ...
  videos/chunk-*/egocentric/episode_*.mp4   the wrist RGB

Run (humi_ik must be on PYTHONPATH for the IK import — run_replay_lerobot_viz.sh
does this):
    PYTHONPATH=third_party/humi_ik .venv/bin/python replay/replay_lerobot_viz.py \
        data/lerobot/session_XXX                         # → :8085
    PYTHONPATH=third_party/humi_ik .venv/bin/python replay/replay_lerobot_viz.py \
        data/lerobot/session_XXX --episode 2 --no-ik     # TCP triad only, no robot
"""
from __future__ import annotations

import argparse
import json
import os
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

import convert_to_lerobot as cl  # noqa: E402  (state/action slices — single source of truth)
import convert_to_umi as conv  # noqa: E402  (calib loaders, shared with the converter)
import replay_common as rc  # noqa: E402

_GRASP_THRESH_M = 0.08  # UMI's hard-coded grasp threshold
_DEFAULT_YAM = os.path.join(_REPO, "robot_assets", "g1_yam", "g1_29dof_yam.urdf")
_JAW_OPEN_M = 0.0475  # URDF jaw prismatic upper limit (matches collection_viz / raw replay)


# ---------------------------------------------------------------------------
# TCP -> tracker recovery (exact inverse of umi_convert_core.apply_tcp_extrinsic)
# ---------------------------------------------------------------------------
def recover_tracker_poses(eef_pos: np.ndarray, eef_rot_rotvec: np.ndarray,
                          T_tracker_tcp: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Recover the per-frame tracker world pose from the STORED TCP pose.

    The converter stored ``world_T_tcp = world_T_tracker @ T_tracker_tcp`` (see
    ``apply_tcp_extrinsic``). Inverting: ``world_T_tracker = world_T_tcp @
    inv(T_tracker_tcp)``. Returns (tracker_pos[T,3], tracker_wxyz[T,4]) ready to
    feed the home-anchored IK — so this replay reproduces the SAME robot motion as
    the raw-take replay, which drives the IK from the live tracker pose."""
    from scipy.spatial.transform import Rotation

    pos = np.asarray(eef_pos, dtype=np.float64)
    rotvec = np.asarray(eef_rot_rotvec, dtype=np.float64)
    n = pos.shape[0]
    R_tcp = Rotation.from_rotvec(rotvec)
    world_T_tcp = np.broadcast_to(np.eye(4), (n, 4, 4)).copy()
    world_T_tcp[:, :3, :3] = R_tcp.as_matrix()
    world_T_tcp[:, :3, 3] = pos
    inv_ext = np.linalg.inv(np.asarray(T_tracker_tcp, dtype=np.float64))
    world_T_tracker = world_T_tcp @ inv_ext[None]
    tr_pos = world_T_tracker[:, :3, 3]
    tr_xyzw = Rotation.from_matrix(world_T_tracker[:, :3, :3]).as_quat()
    tr_wxyz = np.atleast_2d(tr_xyzw)[:, [3, 0, 1, 2]]
    return tr_pos, tr_wxyz


# ---------------------------------------------------------------------------
class LeRobotEpisodes:
    """Lazy per-episode view over a LeRobot v2.1 dataset dir.

    Reads ``meta/info.json`` + ``meta/episodes.jsonl`` once; each episode's parquet
    + mp4 are read on demand (one episode's RGB is the only large block held)."""

    def __init__(self, root: str):
        self.root = os.path.abspath(root)
        info_path = os.path.join(self.root, "meta", "info.json")
        if not os.path.exists(info_path):
            raise SystemExit(f"[replay-lerobot] not a LeRobot dataset (no meta/info.json): {root}")
        self.info = json.load(open(info_path))
        self.fps = float(self.info.get("fps", 15.0))
        self.attrs = self.info  # info.json carries our provenance flags too
        self.data_tmpl = self.info.get("data_path")
        self.video_tmpl = self.info.get("video_path")
        if not self.data_tmpl or not self.video_tmpl:
            raise SystemExit("[replay-lerobot] info.json missing data_path/video_path "
                             "(not a LeRobot v2.1 dataset?)")
        self.chunks_size = int(self.info.get("chunks_size", 1000))
        self.tcp_calibrated = bool(self.info.get("tcp_calibrated", False))
        self.eps = self._load_episodes_jsonl()
        self.n_ep = len(self.eps)
        # state/action slice maps from the WRITER (never re-hardcode them here).
        self._ss = cl._group_slices(cl.STATE_GROUPS)
        self._as = cl._group_slices(cl.ACTION_GROUPS)
        self.has_ft = "ft_wrench" in self._ss

    def _load_episodes_jsonl(self) -> list[dict]:
        p = os.path.join(self.root, "meta", "episodes.jsonl")
        rows: list[dict] = []
        if os.path.exists(p):
            with open(p) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        rows.append(json.loads(line))
        rows.sort(key=lambda r: r.get("episode_index", 0))
        return rows

    def episode_lengths(self) -> list[int]:
        return [int(r.get("length", 0)) for r in self.eps]

    def instruction(self, ep: int) -> str:
        return str(self.eps[ep].get("instruction", ""))

    def _parquet_path(self, ep: int) -> str:
        chunk = ep // self.chunks_size
        return os.path.join(self.root, self.data_tmpl.format(episode_chunk=chunk, episode_index=ep))

    def _video_path(self, ep: int) -> str:
        chunk = ep // self.chunks_size
        return os.path.join(self.root, self.video_tmpl.format(episode_chunk=chunk, episode_index=ep))

    def load_episode(self, ep: int) -> dict:
        import pyarrow.parquet as pq

        table = pq.read_table(self._parquet_path(ep))
        cols = table.column_names
        state_col = cl.STATE_COL if cl.STATE_COL in cols else "observation.state"
        state = np.asarray(table[state_col].to_pylist(), dtype=np.float64)  # [T,13]
        action = np.asarray(table[cl.ACTION_COL].to_pylist(), dtype=np.float64)  # [T,7]
        ts = np.asarray(table["timestamp"].to_pylist(), dtype=np.float64) if "timestamp" in cols \
            else np.arange(state.shape[0]) / self.fps
        if state.ndim != 2 or state.shape[1] < cl.STATE_DIM:
            raise SystemExit(f"[replay-lerobot] episode {ep}: '{state_col}' has shape "
                             f"{state.shape}, expected [T,{cl.STATE_DIM}] — wrong dataset/schema?")

        sp, sr = self._ss["eef_pos"], self._ss["eef_rot"]
        sg, sf = self._ss["gripper"], self._ss.get("ft_wrench")
        pos = state[:, sp[0]:sp[1]]
        rot = state[:, sr[0]:sr[1]]
        width = state[:, sg[0]:sg[1]].reshape(-1)
        ft = state[:, sf[0]:sf[1]] if (sf and self.has_ft) else None

        rgb = self._decode_video(self._video_path(ep), expect=state.shape[0])
        return {
            "pos": pos, "rot": rot, "width": width, "ft": ft, "ts": ts,
            "wxyz": rotvec_to_wxyz(rot), "rgb": rgb, "action": action,
        }

    def _decode_video(self, path: str, expect: int) -> list[np.ndarray]:
        """Decode the episode mp4 to a list of HxWx3 uint8 RGB frames. Returns a
        list (frames may be 1 off the parquet length after re-encode; the render
        clamps the index, and we WARN on a mismatch rather than crash)."""
        if not os.path.exists(path):
            print(f"[replay-lerobot] WARN: missing video {path}; frustum will be blank")
            return []
        import av

        frames: list[np.ndarray] = []
        container = av.open(path)
        try:
            for frame in container.decode(video=0):
                frames.append(frame.to_ndarray(format="rgb24"))
        finally:
            container.close()
        if expect and abs(len(frames) - expect) > 1:
            print(f"[replay-lerobot] WARN: {os.path.basename(path)} has {len(frames)} video "
                  f"frames but parquet has {expect} rows (re-encode mismatch).")
        return frames


def rotvec_to_wxyz(rotvec: np.ndarray) -> np.ndarray:
    """[T,3] axis-angle -> [T,4] wxyz for viser frames."""
    from scipy.spatial.transform import Rotation
    q_xyzw = Rotation.from_rotvec(np.asarray(rotvec, dtype=np.float64)).as_quat()
    return np.atleast_2d(q_xyzw)[:, [3, 0, 1, 2]]


def _rotvec_single(rotvec3: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation
    q = Rotation.from_rotvec(np.asarray(rotvec3, dtype=np.float64)).as_quat()  # xyzw
    return np.asarray([q[3], q[0], q[1], q[2]])


# ---------------------------------------------------------------------------
# Pre-solve the IK over ONE episode so scrubbing never calls mink (mirrors
# replay_raw_viz.presolve_ik, but per-episode and driven by the recovered tracker
# pose). Runs on the loop thread (the only thread that touches the IK config).
# ---------------------------------------------------------------------------
def presolve_episode(tr_pos: np.ndarray, tr_wxyz: np.ndarray, width_m: np.ndarray,
                     width_open_m: float, ik, mapper):
    """Solve the right-hand IK once per frame of this episode and cache the yam
    config + wrist pose at each. Returns (cfgs[K,n_yam], wrist_pos[K,3],
    wrist_wxyz[K,4]). Recalibrates the home anchor at the episode's first frame
    (each episode is independent), then steps toward each recovered tracker pose."""
    K = int(tr_pos.shape[0])
    n_yam = len(mapper._yam_names)
    cfgs = np.zeros((K, n_yam))
    wrist_pos = np.zeros((K, 3))
    wrist_wxyz = np.tile([1.0, 0, 0, 0], (K, 1)).astype(float)
    wo = max(float(width_open_m), 1e-6)
    if K == 0:
        return cfgs, wrist_pos, wrist_wxyz
    ik.recalibrate(tr_pos[0], tr_wxyz[0])
    for i in range(K):
        q = ik.step(tr_pos[i], tr_wxyz[i])
        jaw = float(np.clip(width_m[i] / wo, 0.0, 1.0)) * _JAW_OPEN_M
        cfgs[i] = mapper.to_yam(q, left_jaw_m=0.0, right_jaw_m=jaw)
        wrist = ik._solver.configuration.get_transform_frame_to_world(
            frame_name="right_wrist_yaw_link", frame_type="body")
        wrist_pos[i] = wrist.translation()
        wrist_wxyz[i] = wrist.rotation().wxyz
    return cfgs, wrist_pos, wrist_wxyz


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("dataset_dir", help="a converted LeRobot dataset dir (from processing/convert_to_lerobot.py)")
    p.add_argument("--episode", type=int, default=0, help="initial episode index to show")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8085)
    p.add_argument("--no-ik", action="store_true",
                   help="skip the mink IK / humanoid: show the stored TCP triad + camera + F/T only")
    p.add_argument("--yam-urdf", default=_DEFAULT_YAM)
    p.add_argument("--ik-config", default="proposal")
    p.add_argument("--reach-scale", type=float, default=1.0)
    # Camera mount RELATIVE TO THE WRIST link (same convention + defaults as the raw
    # replay / live viz), so the recorded image sits upright on the FK wrist.
    p.add_argument("--cam-offset", type=float, nargs=3, default=(0.10, 0.0, 0.04))
    p.add_argument("--cam-rpy", type=float, nargs=3, default=(4.7124, 1.5708, 0.0))
    p.add_argument("--cam-fov-deg", type=float, default=58.0)
    args = p.parse_args(argv if argv is not None else sys.argv[1:])

    z = LeRobotEpisodes(args.dataset_dir)
    if z.n_ep == 0:
        raise SystemExit("[replay-lerobot] no episodes (empty episodes.jsonl)")
    ep0 = max(0, min(args.episode, z.n_ep - 1))
    print(f"[replay-lerobot] {os.path.basename(z.root)}: {z.n_ep} episodes, "
          f"lengths {z.episode_lengths()}, fps={z.fps}")
    if z.info.get("WARNING"):
        print(f"[replay-lerobot] ATTR WARNING: {z.info['WARNING']}")
    if not z.tcp_calibrated:
        print("[replay-lerobot] NOTE: tcp_calibrated=False — eef poses are RAW tracker, not TCP.")
    print(f"[replay-lerobot] F/T provenance: {z.info.get('ft_compensated', 'unknown')}")

    # --- TCP extrinsic for tracker recovery (so IK reproduces the raw-take robot) ---
    # PREFER the exact matrix the converter STORED in info.json (``tcp_extrinsic``):
    # world_T_tcp = world_T_tracker @ T_tracker_tcp, so inv() of it maps the stored
    # TCP pose back to the tracker the IK needs — exact even for a custom extrinsic.
    # Only needed when the IK is on; computed lazily so --no-ik never depends on it.
    def _resolve_extrinsic() -> np.ndarray:
        stored = z.info.get("tcp_extrinsic")
        if stored is not None:
            T = np.asarray(stored, dtype=np.float64)
            if T.shape == (4, 4):
                print(f"[replay-lerobot] recovering tracker pose with the dataset's STORED "
                      f"TCP extrinsic (+x {float(T[0, 3]):.3f} m) — exact.")
                return T
            print(f"[replay-lerobot] WARNING: info.tcp_extrinsic has bad shape {T.shape}; "
                  f"falling back.", flush=True)
        # Older dataset without a stored matrix: fall back to the repo default when
        # the dataset says it was calibrated (load_tcp_extrinsic returns identity+False
        # if the repo file is missing — flag that loudly, the recovery would be wrong).
        if z.tcp_calibrated:
            T, cal = conv.load_tcp_extrinsic(z.root)
            if cal:
                print(f"[replay-lerobot] no stored extrinsic; using repo-default TCP extrinsic "
                      f"(+x {float(T[0, 3]):.3f} m). Exact only for repo-default sessions.")
            else:
                print(f"[replay-lerobot] WARNING: dataset is tcp_calibrated but has NO stored "
                      f"extrinsic AND the repo default could not be resolved — recovering with "
                      f"IDENTITY, so the humanoid will be MIS-PLACED. Re-run the converter to "
                      f"embed tcp_extrinsic, or use --no-ik.", flush=True)
            return T
        return np.eye(4)  # uncalibrated -> TCP == tracker

    # gripper width_open_m for the jaw mapping (repo default 0.095 if absent).
    gcal = conv.load_gripper_calib(z.root)
    width_open_m = float(gcal.width_open_m) if gcal is not None else 0.095

    # --- IK + mapper (optional) ---
    ik = mapper = None
    T_tracker_tcp = np.eye(4)
    if not args.no_ik:
        try:
            from g1_right_hand_ik import RightHandIK, YamConfigMapper, yam_actuated_joint_names
            ik = RightHandIK(config_name=args.ik_config, reach_scale=args.reach_scale)
            yam_names = yam_actuated_joint_names(args.yam_urdf)
            mapper = YamConfigMapper(ik._humi_joint_names, yam_names)
            T_tracker_tcp = _resolve_extrinsic()  # only needed once the IK is live
        except ImportError as e:
            print(f"[replay-lerobot] WARNING: could not import the mink IK ({e}); "
                  f"falling back to --no-ik (TCP triad, no humanoid). Is "
                  f"PYTHONPATH=third_party/humi_ik set? (run_replay_lerobot_viz.sh does this)",
                  flush=True)
            ik = mapper = None
        except Exception as e:  # noqa: BLE001 - bad --ik-config etc.: degrade, don't crash
            print(f"[replay-lerobot] WARNING: could not init the mink IK ({e!r}); "
                  f"falling back to --no-ik (TCP triad, no humanoid). Check --ik-config / "
                  f"--yam-urdf.", flush=True)
            ik = mapper = None

    # --- scene ---
    server = viser.ViserServer(host=args.host, port=args.port, label="LeRobot replay")
    server.scene.set_up_direction("+z")
    server.scene.add_grid("/grid", plane="xy", shadow_opacity=0.0, infinite_grid=True, cell_size=0.1)

    urdf = None
    if ik is not None:
        from pathlib import Path
        from viser.extras import ViserUrdf
        base_xyz = np.asarray(ik.home_q()[0:3], dtype=float).copy()
        server.scene.add_frame("/robot", show_axes=False, position=tuple(base_xyz))
        urdf = ViserUrdf(server, urdf_or_path=Path(args.yam_urdf), root_node_name="/robot")

    # /follow rides the FK wrist (IK) or the stored TCP (no-ik). The camera mounts
    # off it with the WRIST-relative offset+rpy (raw-replay convention), so the image
    # is upright. F/T arrows also anchor here.
    follow = server.scene.add_frame("/follow", show_axes=True, axes_length=0.06,
                                    axes_radius=0.004, origin_radius=0.01)
    rpy = np.asarray(args.cam_rpy, float)
    cam_rot = (vtf.SO3.from_rpy_radians(float(rpy[0]), 0.0, 0.0)
               .multiply(vtf.SO3.from_rpy_radians(0.0, float(rpy[1]), 0.0)))
    server.scene.add_frame("/follow/cam", show_axes=False,
                           position=tuple(np.asarray(args.cam_offset, float)),
                           wxyz=tuple(cam_rot.wxyz))
    frustum = server.scene.add_camera_frustum(
        "/follow/cam/view", fov=np.deg2rad(args.cam_fov_deg), aspect=4.0 / 3.0, scale=0.12,
        color=(40, 200, 255))
    wrench = rc.WrenchArrows(server, name="/follow/wrench") if z.has_ft else None

    gui = server.gui
    gui.configure_theme(control_layout="collapsible", control_width="large")

    with gui.add_folder("Dataset"):
        gui.add_text("Dir", initial_value=os.path.basename(z.root), disabled=True)
        gui.add_text("Episodes", initial_value=str(z.n_ep), disabled=True)
        gui.add_text("fps", initial_value=str(z.fps), disabled=True)
        gui.add_text("Robot (IK)", initial_value=("on" if ik is not None else "off (--no-ik)"),
                     disabled=True)
        gui.add_text("TCP calibrated", initial_value=str(z.tcp_calibrated), disabled=True)
        gui.add_text("F/T", initial_value=str(z.info.get("ft_compensated", "?")), disabled=True)
        ep_options = [f"ep {i}  ({n} fr)" for i, n in enumerate(z.episode_lengths())]
        ep_dd = gui.add_dropdown("Episode", ep_options, initial_value=ep_options[ep0])
        rd_task = gui.add_text("Task", initial_value=z.instruction(ep0), disabled=True)
    with gui.add_folder("Frame readout"):
        rd_t = gui.add_text("t (s, rel)", initial_value="0.000", disabled=True)
        rd_pos = gui.add_text("eef xyz", initial_value="-", disabled=True)
        rd_grip = gui.add_text("gripper width", initial_value="-", disabled=True)
        rd_force = gui.add_text("|F| (N)", initial_value="-", disabled=True)
        rd_torque = gui.add_text("|T| (Nm)", initial_value="-", disabled=True)
    with gui.add_folder("View"):
        show_cam = gui.add_checkbox("Show camera", initial_value=True)
        show_path = gui.add_checkbox("Show TCP path", initial_value=True)
        show_wrench = gui.add_checkbox("Show F/T arrows", initial_value=bool(z.has_ft))
    with gui.add_folder("Charts"):
        grip_chart = rc.make_single_chart(gui, "gripper_width (m)", "width", color_idx=1)
        if z.has_ft:
            force_chart = rc.make_xyz_chart(gui, "ft_wrench force (N)", ("Fx", "Fy", "Fz"))
            torque_chart = rc.make_xyz_chart(gui, "ft_wrench torque (Nm)", ("Tx", "Ty", "Tz"))

    # st["data"] holds the loaded episode arrays; st["ik"] holds the per-episode
    # IK presolve (cfgs/wrist_pos/wrist_wxyz) or None when --no-ik.
    st: dict = {"ep": ep0, "data": None, "ik": None, "last_cam_i": -1, "pending_ep": None}

    def solve_ik_for(d: dict) -> dict | None:
        """Recover the tracker pose from the stored TCP pose and presolve the IK
        for this episode. Returns the cached arrays, or None when IK is disabled.
        Runs on the loop thread only (sole owner of the mink Configuration)."""
        if ik is None:
            return None
        tr_pos, tr_wxyz = recover_tracker_poses(d["pos"], d["rot"], T_tracker_tcp)
        cfgs, wpos, wwxyz = presolve_episode(tr_pos, tr_wxyz, d["width"], width_open_m, ik, mapper)
        return {"cfgs": cfgs, "wrist_pos": wpos, "wrist_wxyz": wwxyz}

    def set_static_scene(d: dict) -> None:
        pos = d["pos"]
        if show_path.value and pos.shape[0] >= 2:
            server.scene.add_spline_catmull_rom("/tcp_path", points=pos.astype(np.float32),
                                                color=(250, 200, 80), line_width=2.0)
        else:
            server.scene.add_spline_catmull_rom("/tcp_path", points=pos[:1].astype(np.float32),
                                                visible=False)
        rc.set_full_single(grip_chart, d["ts"], d["width"])
        if z.has_ft and d["ft"] is not None:
            rc.set_full_xyz(force_chart, d["ts"], d["ft"][:, 0:3])
            rc.set_full_xyz(torque_chart, d["ts"], d["ft"][:, 3:6])

    def render_at(i: int) -> None:
        d = st["data"]
        if d["pos"].shape[0] == 0:  # empty episode (corrupt parquet) — nothing to draw
            return
        i = int(np.clip(i, 0, d["pos"].shape[0] - 1))
        ik_arr = st["ik"]
        if ik_arr is not None:
            # robot driven by the per-episode IK presolve; /follow rides the FK wrist.
            urdf.update_cfg(ik_arr["cfgs"][i])
            origin = ik_arr["wrist_pos"][i]
            o_wxyz = ik_arr["wrist_wxyz"][i]
        else:
            # no-ik: /follow rides the stored TCP pose (the old triad behavior).
            origin = d["pos"][i]
            o_wxyz = d["wxyz"][i]
        follow.position = tuple(origin)
        follow.wxyz = tuple(o_wxyz)
        frustum.visible = show_cam.value
        if show_cam.value and i != st["last_cam_i"] and d["rgb"]:
            ci = int(np.clip(i, 0, len(d["rgb"]) - 1))
            img = d["rgb"][ci]
            h, w = img.shape[:2]
            frustum.aspect = w / max(h, 1)
            frustum.image = np.asarray(img)
            st["last_cam_i"] = i
        if wrench is not None:
            wrench.set_visible(show_wrench.value)
            if show_wrench.value and d["ft"] is not None:
                wrench.update(origin, o_wxyz, d["ft"][i, 0:3], d["ft"][i, 3:6])
        rd_t.value = f"{d['ts'][i]:.3f}"
        rd_pos.value = np.array2string(d["pos"][i], precision=3, suppress_small=True)
        gw = float(d["width"][i])
        rd_grip.value = f"{gw*1000:.1f} mm ({'grasp' if gw < _GRASP_THRESH_M else 'open'})"
        if d["ft"] is not None:
            rd_force.value = f"{np.linalg.norm(d['ft'][i,0:3]):.2f}"
            rd_torque.value = f"{np.linalg.norm(d['ft'][i,3:6]):.3f}"

    def apply_episode(ep: int, timeline) -> None:
        st["ep"] = ep
        st["data"] = z.load_episode(ep)
        st["ik"] = solve_ik_for(st["data"])  # presolve on the loop thread
        st["last_cam_i"] = -1
        rd_task.value = z.instruction(ep)
        set_static_scene(st["data"])
        timeline.set_series(st["data"]["ts"])

    @ep_dd.on_update
    def _(_e) -> None:
        st["pending_ep"] = ep_options.index(ep_dd.value)

    if ik is not None:
        print(f"[replay-lerobot] pre-solving IK for episode {ep0}…", flush=True)
    st["data"] = z.load_episode(st["ep"])
    st["ik"] = solve_ik_for(st["data"])
    rd_task.value = z.instruction(st["ep"])
    set_static_scene(st["data"])
    timeline = rc.Timeline(server, st["data"]["ts"], render_at, folder_label="Playback")

    def poll_pending() -> None:
        ep = st["pending_ep"]
        if ep is None:
            return
        st["pending_ep"] = None
        if ep != st["ep"]:
            if ik is not None:
                print(f"[replay-lerobot] pre-solving IK for episode {ep}…", flush=True)
            apply_episode(ep, timeline)

    timeline.before_render = poll_pending
    print(f"[replay-lerobot] http://{args.host}:{args.port}  — pick an Episode, drag Time to scrub",
          flush=True)
    timeline.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
