#!/usr/bin/env python3
"""Convert a raw recording session -> a LeRobot v2.1 dataset (Psi0 / GR00T layout).

This is the SECOND first-class export target, at the SAME level as
``processing/convert_to_umi.py`` (raw -> UMI Zarr): both read the raw per-take
streams written by ``recording/recorder.py``, split each take into episodes at
its markers, and resample every modality onto each episode's CAMERA-FRAME
timeline. They INTENTIONALLY share the resampling/calibration core: this module
calls ``convert_to_umi.build_episodes_from_take`` verbatim, so the LeRobot and
Zarr exports are produced from numerically IDENTICAL per-frame arrays — a
divergence between the two would be a bug, not a design choice.

Why a separate exporter (and why not the stock Psi0 ``postprocess_psi0_sonic.py``):
  * The Psi-0 / GR00T ingest format is **LeRobot v2.1 on disk** — a directory of
    ``data/chunk-*/episode_*.parquet`` + ``videos/chunk-*/egocentric/episode_*.mp4``
    + ``meta/{info,tasks,episodes,episodes_stats}.jsonl`` + ``meta/modality.json``
    (GR00T grouping) + ``meta/stats.json``. We write exactly that here, matching
    the canonical writer ``Psi0/scripts/data/raw_to_lerobot.py``
    field-for-field (column names
    ``states``/``action``, the ``egocentric`` video dir, the modality + stats
    layout) so the Psi0 ``finetune-real-psi0`` loader ingests it.
  * But Psi0's CANONICAL ``states``/``action`` are a **whole-body 36-DOF G1 joint
    vector** (legs+waist+arms+hands + base nav/height). OUR rig records
    END-EFFECTOR data only (a single VIVE-driven TCP 6-DOF pose + gripper width +
    Bota F/T + one wrist RGB) — there is NO whole-body joint stream. Feeding our
    EE data into the 36-DOF schema (what ``postprocess_psi0_sonic.py`` expects)
    would SILENTLY zero/garble most channels. So we emit an HONEST end-effector
    schema (the same physical signals as our Zarr) and describe it in
    ``modality.json`` — a downstream G1 whole-body policy is NOT directly trainable
    on this; an EE/Cartesian policy (UMI/diffusion-policy-style, or GR00T with a
    CUSTOM modality config naming these 3 action keys) is.

Schema we emit (flat LeRobot columns; ``modality.json`` carries the GR00T slices):

  states                    f32 [13]  eef_pos(3) | eef_rot_axis_angle(3) |
                                       gripper_width_m(1) | ft_wrench(6)
  action                    f32 [7]   eef_pos(3) | eef_rot_axis_angle(3) |
                                       gripper_width_m(1)            (ABSOLUTE EE pose)
  observation.images.egocentric video [H,W,3]  the wrist RGB (out-res, default 224x224)
  timestamp                 f32       uniform i/fps (LeRobot derives its timeline from fps)
  frame_index/episode_index/index/task_index  int64
  next.done                 bool      True on each episode's last frame

  * STATE vs ACTION (read before you train): this is END-EFFECTOR imitation data.
    ``action[t]`` is the ABSOLUTE TCP pose+gripper AT frame t — the SAME value as
    the pose part of ``states[t]``. That is intentional and is exactly how UMI /
    diffusion-policy treat an EE trajectory: the SE(3) pose array is BOTH the
    proprioceptive observation and the action target, and the "predict the future"
    relationship lives in the policy's ACTION-CHUNK / horizon sampler (which reads
    ``action[t : t+H]`` = future absolute poses), NOT in a stored one-frame shift.
    There is no separate commanded-vs-achieved signal in this rig — the VIVE pose
    IS the operator command. CONSEQUENCE: train a chunked/horizon absolute-pose
    policy. A naive single-step ``π(obs_t) -> action_t`` config would see
    ``action_t == pose_t`` and learn the identity (a do-nothing controller); that
    is a config error, not a data error. F/T is a SENSED signal (observation only),
    so it is in ``states`` but NOT in ``action``; ``modality.json`` flags action
    ``absolute``.
  * QUATERNION (DECISION-B) + TCP extrinsic (DECISION-D) + gripper metres
    (DECISION-H) + F/T provenance (record-both) are all handled INSIDE the shared
    core / ``build_episodes_from_take`` — see convert_to_umi.py. The
    UNCALIBRATED-TCP and F/T-compensated provenance are surfaced here too (info.json
    + a printed banner), so a LeRobot consumer sees the same warnings the Zarr does.
  * fps: LeRobot derives its timeline from a SINGLE nominal ``fps`` while our camera
    grid is mildly irregular (~D405 12-15 Hz). We default ``--fps`` to the median
    camera rate measured across the session (rounded), and the mp4 is encoded at the
    same integer fps so video frame i lands at i/fps (matches LeRobot's
    ``timestamp = frame_index/fps`` within tolerance). The TRUE per-frame wall-clock
    grid stays in the raw take (and in the Zarr's relative ``episode_timestamps``).

Usage:
    .venv/bin/python processing/convert_to_lerobot.py data/recordings/session_XXX \
        -o data/lerobot/session_XXX --task "pick and place" [--fps 15] [--out-res 224 224]

Needs ``pyarrow`` + ``av`` (PyAV, h264) in the .venv (env/requirements-processing.txt).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "recording"))

import convert_to_umi as umi  # noqa: E402  (reuse the EXACT per-episode builder + calib loaders)
import recording_format as fmt  # noqa: E402

CODE_VERSION = "v2.1"          # LeRobot codebase_version string the Psi0 loader expects
CHUNKS_SIZE = 1000             # episodes per data/video chunk dir (LeRobot convention)
STATE_COL = "states"           # parquet/feature column (Psi0 reference uses `states`, not observation.state)
ACTION_COL = "action"
VIDEO_FEATURE = "observation.images.egocentric"  # the feature key + modality original_key
VIDEO_DIR = "egocentric"       # on-disk video sub-dir (matches the Psi0 reference layout)

# Flat layout of states / action (single source of truth for the parquet builder
# AND modality.json AND the replay viewer, so they never drift). Each entry:
# (group_name, length). Cumulative offsets give the modality slices.
STATE_GROUPS = (("eef_pos", 3), ("eef_rot", 3), ("gripper", 1), ("ft_wrench", 6))
ACTION_GROUPS = (("eef_pos", 3), ("eef_rot", 3), ("gripper", 1))
STATE_DIM = sum(n for _, n in STATE_GROUPS)    # 13
ACTION_DIM = sum(n for _, n in ACTION_GROUPS)  # 7


# ---------------------------------------------------------------------------
# Pure schema helpers (no I/O — unit-tested in test_convert_lerobot.py)
# ---------------------------------------------------------------------------
def _group_slices(groups) -> dict[str, tuple[int, int]]:
    """[(name,len),...] -> {name: (start,end)} half-open cumulative slices."""
    out: dict[str, tuple[int, int]] = {}
    cur = 0
    for name, n in groups:
        out[name] = (cur, cur + n)
        cur += n
    return out


def episode_state_action(ep: dict) -> tuple[np.ndarray, np.ndarray]:
    """Assemble the flat ``states`` [T,13] and ``action`` [T,7] arrays from one
    episode dict (the structure ``build_episodes_from_take`` returns).

    states = eef_pos | eef_rot_axis_angle | gripper_width | ft_wrench
    action = eef_pos | eef_rot_axis_angle | gripper_width   (absolute EE target;
             SAME-frame as the state — see the module docstring's STATE vs ACTION)

    Both float32. F/T is observation-only (sensed, not commanded), so it is in
    states but not action. Lengths come from the per-episode camera grid (T)."""
    pos = np.asarray(ep["robot0_eef_pos"], dtype=np.float32)            # [T,3]
    rot = np.asarray(ep["robot0_eef_rot_axis_angle"], dtype=np.float32)  # [T,3]
    grip = np.asarray(ep["robot0_gripper_width"], dtype=np.float32).reshape(-1, 1)  # [T,1]
    ft = np.asarray(ep["robot0_ft_wrench"], dtype=np.float32)            # [T,6]
    state = np.concatenate([pos, rot, grip, ft], axis=1).astype(np.float32)
    action = np.concatenate([pos, rot, grip], axis=1).astype(np.float32)
    assert state.shape[1] == STATE_DIM, (state.shape, STATE_DIM)
    assert action.shape[1] == ACTION_DIM, (action.shape, ACTION_DIM)
    return state, action


def modality_dict() -> dict:
    """GR00T/Psi0 ``meta/modality.json``: name the sub-slices of state/action +
    the video + the language annotation. Mirrors ``postprocess_psi0_sonic.py``'s
    ``modality_dict`` (every entry carries ``original_key``/``rotation_type``/
    ``absolute``/``dtype``); ``rotation_type='axis_angle'`` marks the EE rotation
    block as a 3-vector rotvec. ``original_key`` points groups at the flat
    parquet columns (``states``/``action``), which is how GR00T resolves them."""
    s = _group_slices(STATE_GROUPS)
    a = _group_slices(ACTION_GROUPS)

    def entry(slc, original_key, rotation_type=None, absolute=True):
        start, end = slc
        return {"start": int(start), "end": int(end), "rotation_type": rotation_type,
                "absolute": absolute, "dtype": "float32", "original_key": original_key}

    return {
        "state": {
            "eef_pos": entry(s["eef_pos"], STATE_COL),
            "eef_rot": entry(s["eef_rot"], STATE_COL, rotation_type="axis_angle"),
            "gripper": entry(s["gripper"], STATE_COL),
            "ft_wrench": entry(s["ft_wrench"], STATE_COL),
        },
        "action": {
            "eef_pos": entry(a["eef_pos"], ACTION_COL),
            "eef_rot": entry(a["eef_rot"], ACTION_COL, rotation_type="axis_angle"),
            "gripper": entry(a["gripper"], ACTION_COL),
        },
        "video": {"egocentric": {"original_key": VIDEO_FEATURE}},
        "annotation": {"human.task_description": {"original_key": "task_index"}},
    }


def stats_block(arr: np.ndarray) -> dict:
    """min/max/mean/std/q01/q99 over axis 0 (GR00T calc_modality_stats format)."""
    a = np.asarray(arr, dtype=np.float32)
    if a.ndim == 1:
        a = a.reshape(-1, 1)
    return {
        "min": a.min(axis=0).astype(np.float32).tolist(),
        "max": a.max(axis=0).astype(np.float32).tolist(),
        "mean": a.mean(axis=0).astype(np.float32).tolist(),
        "std": a.std(axis=0).astype(np.float32).tolist(),
        "q01": np.quantile(a, 0.01, axis=0).astype(np.float32).tolist(),
        "q99": np.quantile(a, 0.99, axis=0).astype(np.float32).tolist(),
    }


def timestamp_stats(n: int, fps: float) -> dict:
    """Closed-form stats for a uniform 0..(n-1)/fps timestamp column (matches the
    Psi0 scripts' analytic timestamp stats so a stats consumer sees the same shape)."""
    n = int(n)
    return {
        "min": [0.0], "max": [(n - 1) / fps if n > 1 else 0.0],
        "mean": [((n - 1) / 2) / fps if n > 1 else 0.0],
        "std": [n / (2 * fps * math.sqrt(3)) if n > 1 else 0.0],
        "q01": [0.0], "q99": [(n - 1) / fps if n > 1 else 0.0],
        "count": [n],
    }


def info_features(out_w: int, out_h: int, fps: float) -> dict:
    """The ``features`` block of meta/info.json (LeRobot v2.1)."""
    return {
        VIDEO_FEATURE: {
            "dtype": "video",
            "shape": [out_h, out_w, 3],
            "names": ["height", "width", "channel"],
            "video_info": {
                "video.fps": float(fps),
                "video.codec": "h264",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "has_audio": False,
            },
        },
        STATE_COL: {"dtype": "float32", "shape": [STATE_DIM], "names": ["state"]},
        ACTION_COL: {"dtype": "float32", "shape": [ACTION_DIM], "names": ["action"]},
        "timestamp": {"dtype": "float32", "shape": [1]},
        "frame_index": {"dtype": "int64", "shape": [1]},
        "episode_index": {"dtype": "int64", "shape": [1]},
        "index": {"dtype": "int64", "shape": [1]},
        "next.done": {"dtype": "bool", "shape": [1]},
        "task_index": {"dtype": "int64", "shape": [1]},
    }


# ---------------------------------------------------------------------------
# I/O helpers (parquet + mp4)
# ---------------------------------------------------------------------------
def _write_jsonl(path: str, rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            json.dump(r, f, separators=(",", ":"))
            f.write("\n")


def write_episode_parquet(path: str, state: np.ndarray, action: np.ndarray,
                          timestamp: np.ndarray, episode_index: int, task_index: int,
                          index_base: int) -> int:
    """Write ONE episode's low-dim table as LeRobot parquet. Returns the number of
    frames written. Uses pyarrow directly (``datasets`` is intentionally not a
    .venv dep). ``states``/``action`` go in as list<float32> columns (LeRobot's
    Sequence(Value('float32')) layout)."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    n = int(state.shape[0])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    done = np.zeros(n, dtype=bool)
    if n > 0:
        done[-1] = True
    table = pa.table({
        STATE_COL: pa.array(state.tolist(), type=pa.list_(pa.float32())),
        ACTION_COL: pa.array(action.tolist(), type=pa.list_(pa.float32())),
        "timestamp": pa.array(np.asarray(timestamp, dtype=np.float32)),
        "frame_index": pa.array(np.arange(n, dtype=np.int64)),
        "episode_index": pa.array(np.full(n, episode_index, dtype=np.int64)),
        "index": pa.array(np.arange(index_base, index_base + n, dtype=np.int64)),
        "task_index": pa.array(np.full(n, task_index, dtype=np.int64)),
        "next.done": pa.array(done),
    })
    pq.write_table(table, path)
    return n


def encode_mp4(path: str, frames: np.ndarray, fps: float) -> None:
    """Encode [T,H,W,3] uint8 RGB -> h264 mp4 (yuv420p) with PyAV. The caller writes
    to a tmp path then renames. h264 needs even dims (validated up front in main)."""
    import av

    frames = np.asarray(frames)
    if frames.ndim != 4 or frames.shape[0] == 0:
        raise ValueError(f"frames must be [T,H,W,3] with T>0, got {frames.shape}")
    h, w = int(frames.shape[1]), int(frames.shape[2])
    if h % 2 or w % 2:
        raise ValueError(f"h264 yuv420p needs even dims, got {w}x{h}")
    rate = max(1, int(round(fps)))  # integer stream rate; fps is validated >=1 in main
    os.makedirs(os.path.dirname(path), exist_ok=True)
    container = av.open(path, mode="w")
    try:
        stream = container.add_stream("libx264", rate=rate)
        stream.width = w
        stream.height = h
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "18"}  # near-visually-lossless; RGB is re-encoded
        for i in range(frames.shape[0]):
            frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(frames[i]), format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():  # flush
            container.mux(packet)
    finally:
        container.close()


# ---------------------------------------------------------------------------
# Up-front validation (fail before writing anything — partial datasets are worse
# than a clean abort)
# ---------------------------------------------------------------------------
def _check_encoders() -> str | None:
    """Return an error string if pyarrow / PyAV+libx264 are unavailable, else None."""
    try:
        import pyarrow  # noqa: F401
        import pyarrow.parquet  # noqa: F401
    except ImportError as e:
        return (f"pyarrow is required to write LeRobot parquet but is missing ({e}). "
                f"Install the conversion deps: uv pip install --python .venv/bin/python "
                f"-r env/requirements-processing.txt")
    try:
        import av
        av.codec.Codec("libx264", "w")
    except Exception as e:  # noqa: BLE001 - missing PyAV or libx264 both land here
        return (f"PyAV with an h264 (libx264) encoder is required for the episode "
                f"videos but is unavailable ({e}). Install/repair: uv pip install "
                f"--python .venv/bin/python -r env/requirements-processing.txt")
    return None


def _refuse_unsafe_overwrite(out_dir: str) -> str | None:
    """Return an error string if --overwrite would rmtree something dangerous.

    Guards against a typo (or input/output confusion) wiping the repo, $HOME, the
    cwd, an ANCESTOR of either, or — most insidiously — a raw recordings dir / the
    very session being converted. We only ever delete a dir that (a) is none of
    those and (b) looks like a LeRobot dataset WE wrote (has meta/info.json). A
    non-existent out_dir is fine (nothing to delete)."""
    if not os.path.exists(out_dir):
        return None
    real = os.path.realpath(out_dir)
    home = os.path.realpath(os.path.expanduser("~"))
    cwd = os.path.realpath(os.getcwd())
    repo = os.path.realpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    sep = os.sep
    # out_dir is a protected root, or an ancestor of the cwd/repo (deleting it would
    # take the working tree / repo with it).
    if real in {home, cwd, repo, os.path.realpath(os.sep)}:
        return f"refusing --overwrite on a protected directory: {real}"
    if (cwd + sep).startswith(real + sep) or (repo + sep).startswith(real + sep):
        return f"refusing --overwrite: {real} is an ancestor of the repo/working dir"
    # Exists but is NOT a dataset we'd recognize -> almost certainly the wrong path
    # (e.g. a raw session dir). Require our own commit marker before deleting.
    if not os.path.exists(os.path.join(out_dir, "meta", "info.json")):
        return (f"refusing --overwrite: {real} exists but is not a LeRobot dataset "
                f"(no meta/info.json) — it looks like the wrong path. Delete it "
                f"yourself if you really mean to.")
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("session_dir")
    p.add_argument("-o", "--output", required=True, help="output LeRobot dataset dir")
    p.add_argument("--task", default="manipulation", help="natural-language task string for every episode")
    p.add_argument("--fps", type=int, default=None,
                   help="dataset fps (LeRobot timeline; must be >=1). Default: median camera rate from the data")
    p.add_argument("--out-res", type=int, nargs=2, default=(224, 224), metavar=("W", "H"))
    p.add_argument("--overwrite", action="store_true", help="remove an existing output dir first (guarded)")
    args = p.parse_args(argv if argv is not None else sys.argv[1:])

    # ---- up-front validation: fail loudly BEFORE writing anything ----
    enc_err = _check_encoders()
    if enc_err:
        print(f"[lerobot] ERROR: {enc_err}", file=sys.stderr)
        return 2
    out_w, out_h = int(args.out_res[0]), int(args.out_res[1])
    if out_w % 2 or out_h % 2 or out_w <= 0 or out_h <= 0:
        print(f"[lerobot] ERROR: --out-res must be positive EVEN dims (h264 yuv420p); "
              f"got {out_w}x{out_h}", file=sys.stderr)
        return 2
    if args.fps is not None and args.fps < 1:
        print(f"[lerobot] ERROR: --fps must be >= 1, got {args.fps}", file=sys.stderr)
        return 2

    session_dir = os.path.abspath(args.session_dir)
    if not os.path.isdir(session_dir):
        print(f"[lerobot] ERROR: session dir not found: {session_dir}", file=sys.stderr)
        return 1
    out_dir = os.path.abspath(args.output)

    print(f"[lerobot] raw tracker quaternions are {fmt.QUATERNION_CONVENTION.upper()}; "
          f"converting to xyzw at umi_convert_core.quat_wxyz_to_xyzw (shared core)", flush=True)

    if os.path.exists(out_dir):
        if not args.overwrite:
            print(f"[lerobot] ERROR: output exists: {out_dir} (pass --overwrite to replace)",
                  file=sys.stderr)
            return 3
        unsafe = _refuse_unsafe_overwrite(out_dir)
        if unsafe:
            print(f"[lerobot] ERROR: {unsafe}", file=sys.stderr)
            return 3

    # Only ingest COMMITTED takes (have take_meta.json) — skip leftover *.tmp (M2),
    # identical to convert_to_umi.
    take_dirs = sorted(
        d for d in (os.path.join(session_dir, x) for x in os.listdir(session_dir))
        if os.path.isdir(d) and os.path.basename(d).startswith("take_")
        and not os.path.basename(d).endswith(".tmp")
        and os.path.exists(os.path.join(d, fmt.TAKE_META_FILE)))
    if not take_dirs:
        print(f"[lerobot] no committed take_* dirs (with {fmt.TAKE_META_FILE}) in {session_dir}",
              file=sys.stderr)
        return 1

    gripper_calib = umi.load_gripper_calib(session_dir)
    if gripper_calib is None:
        print("[lerobot] WARNING: no gripper_range.json — gripper_width will be ZERO.",
              file=sys.stderr)
    T_tracker_tcp, tcp_calibrated = umi.load_tcp_extrinsic(session_dir)
    if not tcp_calibrated:
        print("[lerobot] WARNING: TCP extrinsic is identity/uncalibrated — poses are RAW "
              "TRACKER, not TCP. info.json will carry UNCALIBRATED_DO_NOT_TRAIN.", file=sys.stderr)

    jpeg_decode, image_resize = umi._make_image_fns()

    # First pass: build every episode's resampled arrays (SHARED core) + measure the
    # camera rate so we can pick a sane default fps. Episodes are held in RAM as
    # numpy (same as convert_to_umi); a session is O(hundreds) of short episodes.
    episodes: list[dict] = []
    ft_sources: set[str] = set()
    ft_algos: set[str] = set()
    any_unverified_sign = False
    any_ft_invalid = False
    rates: list[float] = []
    for take_dir in take_dirs:
        for ep in umi.build_episodes_from_take(take_dir, gripper_calib, T_tracker_tcp,
                                               (out_w, out_h), jpeg_decode, image_resize):
            src = ep.pop("_ft_source", "raw")
            ft_sources.add(src)
            algo = ep.pop("_ft_algo", None)
            sign_ok = ep.pop("_ft_sign_verified", True)
            if src == "compensated":
                ft_algos.add(algo or "bota")
                if sign_ok is False:
                    any_unverified_sign = True
            if not ep.pop("_ft_valid", True):
                any_ft_invalid = True
            ts = np.asarray(ep["episode_timestamps"], dtype=np.float64)
            if ts.shape[0] > 1 and ts[-1] > ts[0]:
                rates.append((ts.shape[0] - 1) / (ts[-1] - ts[0]))
            episodes.append(ep)

    if not episodes:
        print("[lerobot] no usable episodes", file=sys.stderr)
        return 1

    fps = int(args.fps) if args.fps else max(1, int(round(float(np.median(rates)) if rates else 15.0)))
    if rates:
        print(f"[lerobot] camera rate across {len(episodes)} episode(s): "
              f"median {np.median(rates):.2f} Hz, range [{min(rates):.2f}, {max(rates):.2f}] Hz "
              f"-> dataset fps={fps} (true per-frame times kept in the raw take)")

    # F/T provenance tri-state (compensated|raw|mixed), same logic as convert_to_umi.
    if ft_sources == {"compensated"}:
        ft_provenance = "compensated"
    elif ft_sources == {"raw"} or not ft_sources:
        ft_provenance = "raw"
    else:
        ft_provenance = "mixed"

    # Compensation-engine tri-state (bota|rnea|mixed|None), same logic as convert_to_umi.
    if not ft_algos:
        ft_comp_algo = None
    elif ft_algos == {"bota"}:
        ft_comp_algo = "bota"
    elif ft_algos == {"rnea"}:
        ft_comp_algo = "rnea"
    else:
        ft_comp_algo = "mixed"
    if ft_comp_algo == "mixed":
        print("[lerobot] WARNING: compensated F/T MIXES algorithms across episodes "
              "(bota gravity-only + rnea full-dynamic) — different physics; see take_meta ft_comp_algo.")
    if any_unverified_sign:
        print("[lerobot] WARNING: a COMPENSATED episode used an RNEA config with an "
              "UNVERIFIED wrench_sign — the compensated wrench may be sign-INVERTED "
              "(see take_meta ft_sign_verified).")

    # Second pass writes into a SIBLING .tmp dir, then atomically renames to out_dir
    # only after meta/ is fully written (the recorder's commit-marker pattern). A
    # crash/exception therefore never leaves a partial dataset at the real path, and
    # a re-run is always clean (no --overwrite needed to recover from a failure).
    tmp_dir = out_dir + ".tmp"
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    try:
        n_episodes, total_frames = _write_dataset(
            tmp_dir, episodes, fps, out_w, out_h, args.task, ft_provenance,
            tcp_calibrated, gripper_calib is not None, any_ft_invalid, T_tracker_tcp,
            ft_comp_algo=ft_comp_algo, any_unverified_sign=any_unverified_sign)
        if os.path.exists(out_dir):
            shutil.rmtree(out_dir)  # guarded above; safe to replace our own dataset
        os.replace(tmp_dir, out_dir)
    finally:
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)  # clean up on failure; no partial at the real path

    print(f"[lerobot] wrote {n_episodes} episode(s), {total_frames} frames -> {out_dir}")
    print(f"[lerobot] schema: {STATE_COL}[{STATE_DIM}] (eef_pos|eef_rot|gripper|ft) + "
          f"{ACTION_COL}[{ACTION_DIM}] (eef_pos|eef_rot|gripper) + {VIDEO_FEATURE} {out_w}x{out_h} h264")
    print(f"[lerobot] F/T provenance: {ft_provenance} | TCP calibrated: {tcp_calibrated} | "
          f"gripper calibrated: {gripper_calib is not None}")
    if not tcp_calibrated:
        print("[lerobot] NOTE: info.json carries UNCALIBRATED_DO_NOT_TRAIN (identity TCP extrinsic).")
    return 0


def _write_dataset(out_dir: str, episodes: list[dict], fps: int, out_w: int, out_h: int,
                   task: str, ft_provenance: str, tcp_calibrated: bool,
                   gripper_calibrated: bool, any_ft_invalid: bool,
                   T_tracker_tcp: np.ndarray, *,
                   ft_comp_algo: "str | None" = None,
                   any_unverified_sign: bool = False) -> tuple[int, int]:
    """Write the whole LeRobot dataset tree into ``out_dir`` (a .tmp staging dir).
    Returns (n_episodes, total_frames). Raises on any I/O failure (the caller cleans
    up the staging dir, so no partial lands at the final path)."""
    for sub in ("data", "videos", "meta"):
        os.makedirs(os.path.join(out_dir, sub), exist_ok=True)

    episodes_meta: list[dict] = []
    episodes_stats: list[dict] = []
    # Accumulate every low-dim column so meta/stats.json carries per-feature stats
    # for ALL of them (LeRobot/GR00T normalization reads stats.json; missing keys
    # KeyError on load — matches postprocess_psi0_sonic.py's full stats block).
    agg = {STATE_COL: [], ACTION_COL: [], "timestamp": [], "frame_index": [],
           "episode_index": [], "index": [], "task_index": [], "next.done": []}
    total_frames = 0
    index_base = 0
    for ep_i, ep in enumerate(episodes):
        chunk = ep_i // CHUNKS_SIZE
        state, action = episode_state_action(ep)
        n = int(state.shape[0])
        # LeRobot derives its timeline from fps; emit a UNIFORM timestamp column at
        # the nominal fps (its loader asserts timestamp ~= frame_index/fps).
        uniform_t = (np.arange(n, dtype=np.float32) / fps)

        parquet_path = os.path.join(out_dir, "data", f"chunk-{chunk:03d}",
                                    f"episode_{ep_i:06d}.parquet")
        write_episode_parquet(parquet_path, state, action, uniform_t,
                              episode_index=ep_i, task_index=0, index_base=index_base)

        vid_dir = os.path.join(out_dir, "videos", f"chunk-{chunk:03d}", VIDEO_DIR)
        vid_path = os.path.join(vid_dir, f"episode_{ep_i:06d}.mp4")
        tmp_path = vid_path + ".tmp.mp4"
        encode_mp4(tmp_path, ep["camera0_rgb"], fps)
        os.replace(tmp_path, vid_path)

        episodes_meta.append({
            "episode_index": ep_i, "tasks": [task], "length": n,
            "dataset_from_index": total_frames, "dataset_to_index": total_frames + n - 1,
            "instruction": task,
        })
        episodes_stats.append({
            "episode_index": ep_i,
            "stats": {
                STATE_COL: {**stats_block(state), "count": [n]},
                ACTION_COL: {**stats_block(action), "count": [n]},
                "timestamp": timestamp_stats(n, fps),
            },
        })
        agg[STATE_COL].append(state)
        agg[ACTION_COL].append(action)
        agg["timestamp"].append(uniform_t)
        agg["frame_index"].append(np.arange(n, dtype=np.int64))
        agg["episode_index"].append(np.full(n, ep_i, dtype=np.int64))
        agg["index"].append(np.arange(index_base, index_base + n, dtype=np.int64))
        agg["task_index"].append(np.zeros(n, dtype=np.int64))
        done = np.zeros(n, dtype=bool)
        if n:
            done[-1] = True
        agg["next.done"].append(done)
        total_frames += n
        index_base += n
        print(f"  episode {ep_i}: {n} frames -> {os.path.relpath(parquet_path, out_dir)}")

    n_episodes = len(episodes)
    n_chunks = math.ceil(n_episodes / CHUNKS_SIZE)

    meta = os.path.join(out_dir, "meta")
    _write_jsonl(os.path.join(meta, "episodes.jsonl"), episodes_meta)
    _write_jsonl(os.path.join(meta, "episodes_stats.jsonl"), episodes_stats)
    _write_jsonl(os.path.join(meta, "tasks.jsonl"),
                 [{"task_index": 0, "task": task, "description": task}])

    info = {
        "codebase_version": CODE_VERSION,
        "robot_type": "g1_umi_ee",  # NOT a whole-body g1 schema — EE/Cartesian (see module docstring)
        "total_episodes": n_episodes,
        "total_frames": int(total_frames),
        "total_tasks": 1,
        "total_videos": n_episodes,
        "total_chunks": n_chunks,
        "chunks_size": CHUNKS_SIZE,
        "fps": fps,
        "splits": {"train": f"0:{n_episodes}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/" + VIDEO_DIR + "/episode_{episode_index:06d}.mp4",
        "features": info_features(out_w, out_h, fps),
        # OUR provenance (mirrors the zarr attrs so a LeRobot consumer sees the same flags):
        "raw_quaternion_convention": fmt.QUATERNION_CONVENTION,
        "tcp_calibrated": bool(tcp_calibrated),
        # Store the ACTUAL extrinsic matrix used (world_T_tcp = world_T_tracker @
        # T_tracker_tcp), so the replay viewer can recover the tracker pose EXACTLY
        # (inverting it) to re-drive the IK — even for a custom non-repo-default
        # extrinsic, and even if the repo calibration file later moves. Identity
        # when uncalibrated (TCP == tracker).
        "tcp_extrinsic": np.asarray(T_tracker_tcp, dtype=np.float64).tolist(),
        "gripper_calibrated": bool(gripper_calibrated),
        "ft_compensated": ft_provenance,
        # Which engine compensated (bota|rnea|mixed|null) — different physics, mirrored
        # from the zarr attrs so a LeRobot consumer sees the same provenance.
        "ft_comp_algo": ft_comp_algo,
        "schema_note": ("END-EFFECTOR schema (TCP 6-DOF + gripper_m + F/T), NOT the "
                        "Psi0 whole-body 36-DOF joint schema. Train an EE/Cartesian "
                        "policy; see meta/modality.json for the state/action slices."),
    }
    if not tcp_calibrated:
        info["WARNING"] = "UNCALIBRATED_DO_NOT_TRAIN: TCP extrinsic is identity"
    if any_ft_invalid:
        info["ft_note"] = "some episodes had no F/T; zero-filled (F/T slice of state is 0)"
    if ft_comp_algo == "mixed":
        info["ft_algo_note"] = "compensated wrench mixes bota + rnea across episodes (see take_meta ft_comp_algo)"
    if any_unverified_sign:
        info["ft_sign_unverified"] = True
    with open(os.path.join(meta, "info.json"), "w") as f:
        json.dump(info, f, indent=4)

    with open(os.path.join(meta, "modality.json"), "w") as f:
        json.dump(modality_dict(), f, indent=2)

    # Aggregate stats.json over the whole dataset for EVERY low-dim column (GR00T
    # calc_modality_stats format). stats_psi0.json is a copy (Psi0 finetune reads it).
    stats = {}
    for key, parts in agg.items():
        cat = np.concatenate(parts, axis=0)
        stats[key] = {**stats_block(cat), "count": [int(total_frames)]}
    with open(os.path.join(meta, "stats.json"), "w") as f:
        json.dump(stats, f, indent=4)
    with open(os.path.join(meta, "stats_psi0.json"), "w") as f:
        json.dump(stats, f, indent=4)

    return n_episodes, total_frames


if __name__ == "__main__":
    raise SystemExit(main())
