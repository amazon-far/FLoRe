#!/usr/bin/env python3
"""Tests for processing/convert_to_lerobot.py — the raw -> LeRobot v2.1 exporter.

Two layers, like the rest of the pipeline's tests:
  * PURE LOGIC: the schema helpers (slice math, state/action assembly, modality
    map, stats) — numpy only, no I/O.
  * ROUND-TRIP: build a tiny synthetic raw take with the SAME helper the converter
    uses, run main() end-to-end into a temp dir, and assert the on-disk LeRobot
    dataset is well-formed (parquet columns + shapes, mp4 present + decodable,
    meta json consistency, state/action slices match the writer's own map, and the
    F/T slice equals what the shared zarr core produced).

Prints "LOGIC OK" on success. Run in the collection .venv:
    .venv/bin/python processing/test_convert_lerobot.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "..", "recording"))

import convert_to_lerobot as cl
import convert_to_umi as umi
import recording_format as fmt


def _ok(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print(f"  ok: {msg}")


# ---------------------------------------------------------------------------
# Pure-logic schema tests
# ---------------------------------------------------------------------------
def test_slices():
    s = cl._group_slices(cl.STATE_GROUPS)
    a = cl._group_slices(cl.ACTION_GROUPS)
    _ok(s["eef_pos"] == (0, 3), "state eef_pos slice 0:3")
    _ok(s["eef_rot"] == (3, 6), "state eef_rot slice 3:6")
    _ok(s["gripper"] == (6, 7), "state gripper slice 6:7")
    _ok(s["ft_wrench"] == (7, 13), "state ft_wrench slice 7:13")
    _ok(a["gripper"] == (6, 7), "action gripper slice 6:7")
    _ok(cl.STATE_DIM == 13 and cl.ACTION_DIM == 7, "state/action dims 13/7")
    # No gap / overlap: groups tile [0, DIM).
    _ok(max(e for _, e in s.values()) == cl.STATE_DIM, "state slices tile to STATE_DIM")
    _ok(max(e for _, e in a.values()) == cl.ACTION_DIM, "action slices tile to ACTION_DIM")


def _fake_episode(T):
    rng = np.random.default_rng(0)
    return {
        "robot0_eef_pos": rng.standard_normal((T, 3)).astype(np.float32),
        "robot0_eef_rot_axis_angle": rng.standard_normal((T, 3)).astype(np.float32),
        "robot0_gripper_width": rng.uniform(0, 0.1, (T, 1)).astype(np.float32),
        "robot0_ft_wrench": rng.standard_normal((T, 6)).astype(np.float32),
        "episode_timestamps": np.arange(T, dtype=np.float64) / 15.0,
        "camera0_rgb": rng.integers(0, 255, (T, 8, 8, 3), dtype=np.uint8),
    }


def test_state_action_assembly():
    ep = _fake_episode(5)
    state, action = cl.episode_state_action(ep)
    _ok(state.shape == (5, 13), "state shape [T,13]")
    _ok(action.shape == (5, 7), "action shape [T,7]")
    _ok(state.dtype == np.float32 and action.dtype == np.float32, "float32")
    # state pose+gripper part equals action exactly (absolute EE target).
    _ok(np.array_equal(state[:, :7], action), "action == state[:, :7] (eef_pos|eef_rot|gripper)")
    # ft slice equals the raw wrench input.
    s = cl._group_slices(cl.STATE_GROUPS)["ft_wrench"]
    _ok(np.array_equal(state[:, s[0]:s[1]], ep["robot0_ft_wrench"]), "state ft slice == input wrench")
    # gripper column carries the raw width.
    _ok(np.array_equal(state[:, 6:7], ep["robot0_gripper_width"]), "state gripper col == input width")


def test_modality_consistency():
    m = cl.modality_dict()
    _ok(set(m) == {"state", "action", "video", "annotation"}, "modality top keys")
    _ok(m["state"]["ft_wrench"]["start"] == 7 and m["state"]["ft_wrench"]["end"] == 13,
        "modality ft slice 7:13 matches STATE_GROUPS")
    _ok(m["state"]["eef_rot"]["rotation_type"] == "axis_angle", "eef_rot rotation_type axis_angle")
    _ok("ft_wrench" not in m["action"], "action has NO ft (observation-only)")
    _ok(m["video"]["egocentric"]["original_key"] == cl.VIDEO_FEATURE, "video original_key")
    # original_key points state/action groups at the flat parquet columns.
    _ok(m["state"]["eef_pos"]["original_key"] == cl.STATE_COL, "state original_key == STATE_COL")
    _ok(m["action"]["eef_pos"]["original_key"] == cl.ACTION_COL, "action original_key == ACTION_COL")
    # modality slices must agree with the live slice map.
    s = cl._group_slices(cl.STATE_GROUPS)
    for name, (a, b) in s.items():
        _ok(m["state"][name]["start"] == a and m["state"][name]["end"] == b,
            f"modality.state.{name} == STATE_GROUPS")


def test_stats_block():
    a = np.array([[0.0, 10.0], [2.0, 20.0], [4.0, 30.0]], dtype=np.float32)
    b = cl.stats_block(a)
    _ok(b["min"] == [0.0, 10.0] and b["max"] == [4.0, 30.0], "stats min/max per-col")
    _ok(abs(b["mean"][0] - 2.0) < 1e-6 and abs(b["mean"][1] - 20.0) < 1e-6, "stats mean per-col")
    _ok(len(b["q01"]) == 2 and len(b["q99"]) == 2, "stats quantiles per-col")
    # 1-D input is treated as a single column.
    b1 = cl.stats_block(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    _ok(b1["mean"] == [2.0], "stats 1-D -> single column")


def test_timestamp_stats():
    s = cl.timestamp_stats(10, 15.0)
    _ok(s["min"] == [0.0], "timestamp min 0")
    _ok(abs(s["max"][0] - 9 / 15.0) < 1e-6, "timestamp max (n-1)/fps")
    _ok(s["count"] == [10], "timestamp count")
    s1 = cl.timestamp_stats(1, 15.0)
    _ok(s1["max"] == [0.0] and s1["std"] == [0.0], "single-frame timestamp stats degenerate to 0")


# ---------------------------------------------------------------------------
# Round-trip: synthetic raw take -> main() -> on-disk dataset
# ---------------------------------------------------------------------------
def _write_synthetic_take(session_dir, n_cam=12, compensated=True):
    """Write one committed raw take using the recorder's real synth JPEG so the
    converter's cv2 decode path is exercised, mirroring test_ft_compensated."""
    import recorder as R

    take_dir = os.path.join(session_dir, fmt.take_dir_name(0))
    cam_dir = os.path.join(take_dir, fmt.CAMERA_DIR)
    os.makedirs(cam_dir, exist_ok=True)
    t0 = 1000.0
    cam_t = t0 + np.arange(n_cam) / 15.0
    # tracker at ~120 Hz spanning the camera window (cols: t,seq,x,y,z,qw,qx,qy,qz,stale)
    n_tr = 80
    tr_t = np.linspace(cam_t[0] - 0.01, cam_t[-1] + 0.01, n_tr)
    tracker = np.zeros((n_tr, 10), dtype=np.float64)
    tracker[:, 0] = tr_t
    tracker[:, 1] = np.arange(n_tr)
    tracker[:, 2] = np.linspace(0, 0.2, n_tr)     # x ramp
    tracker[:, 5] = 1.0                            # qw=1 (identity wxyz)
    np.save(os.path.join(take_dir, fmt.TRACKER_FILE), tracker)
    # gripper (cols: t, enc, cmd, open_rad, closed_rad) with a real homed span
    n_g = 30
    g_t = np.linspace(cam_t[0], cam_t[-1], n_g)
    gripper = np.zeros((n_g, 5), dtype=np.float64)
    gripper[:, 0] = g_t
    gripper[:, 1] = np.linspace(0.0, 4.0, n_g)     # encoder sweep
    gripper[:, 3] = 4.0                             # open_rad
    gripper[:, 4] = 0.0                             # closed_rad
    np.save(os.path.join(take_dir, fmt.GRIPPER_FILE), gripper)
    # F/T raw (always) + compensated sidecar
    n_ft = 200
    ft_t = np.linspace(cam_t[0], cam_t[-1], n_ft)
    ft_raw = np.zeros((n_ft, 7)); ft_raw[:, 0] = ft_t; ft_raw[:, 1] = 9.0   # raw Fx=9
    np.save(os.path.join(take_dir, fmt.FT_FILE), ft_raw)
    if compensated:
        ft_c = np.zeros((n_ft, 7)); ft_c[:, 0] = ft_t; ft_c[:, 1] = 0.2     # comp Fx=0.2
        np.save(os.path.join(take_dir, fmt.FT_COMP_FILE), ft_c)
    np.save(os.path.join(take_dir, fmt.MARKER_FILE), np.zeros((0,), dtype=np.float64))
    # camera frames (real synth jpeg) + index
    for i in range(n_cam):
        with open(os.path.join(cam_dir, fmt.camera_frame_name(i)), "wb") as f:
            f.write(R._synth_jpeg(float(i)))  # recorder's real synth frame (64x48)
    cam_index = np.zeros((n_cam, 2), dtype=np.float64)
    cam_index[:, 0] = cam_t
    cam_index[:, 1] = np.arange(n_cam)
    np.save(os.path.join(cam_dir, fmt.CAMERA_INDEX_FILE), cam_index)
    meta = {"task": "t", "counts": {"camera": n_cam}, "ft_valid": True,
            "ft_compensated": bool(compensated), "n_markers": 0, "n_episodes": 1}
    with open(os.path.join(take_dir, fmt.TAKE_META_FILE), "w") as f:
        json.dump(meta, f)


def _write_calib(session_dir):
    """A non-identity TCP extrinsic + gripper range so the dataset is 'calibrated'
    (exercises the calibrated path; the zarr tests cover the identity warning)."""
    cdir = os.path.join(session_dir, fmt.CALIB_DIR)
    os.makedirs(cdir, exist_ok=True)
    T = np.eye(4); T[0, 3] = 0.21
    json.dump({"T_tracker_tcp": T.tolist(), "calibrated": True},
              open(os.path.join(cdir, fmt.TCP_EXTRINSIC_FILE), "w"))
    json.dump({"enc_closed": 0.0, "enc_open": 4.9, "width_closed_m": 0.0, "width_open_m": 0.095},
              open(os.path.join(cdir, fmt.GRIPPER_RANGE_FILE), "w"))


def test_round_trip():
    import pyarrow.parquet as pq

    with tempfile.TemporaryDirectory() as tmp:
        session = os.path.join(tmp, "session")
        os.makedirs(session)
        _write_synthetic_take(session, n_cam=12, compensated=True)
        _write_calib(session)
        out = os.path.join(tmp, "lerobot_out")

        rc = cl.main([session, "-o", out, "--task", "pick block", "--fps", "15"])
        _ok(rc == 0, "main() returns 0")

        # meta files present
        for f in ("info.json", "tasks.jsonl", "episodes.jsonl",
                  "episodes_stats.jsonl", "modality.json", "stats.json", "stats_psi0.json"):
            _ok(os.path.exists(os.path.join(out, "meta", f)), f"meta/{f} written")

        info = json.load(open(os.path.join(out, "meta", "info.json")))
        _ok(info["total_episodes"] == 1, "info total_episodes == 1")
        _ok(info["fps"] == 15, "info fps == 15")
        _ok(info["codebase_version"] == "v2.1", "info codebase_version v2.1")
        _ok(info["ft_compensated"] == "compensated", "info ft_compensated compensated")
        _ok(info["tcp_calibrated"] is True, "info tcp_calibrated True")
        _ok("WARNING" not in info, "no UNCALIBRATED warning (calibrated extrinsic)")
        _ok(info["splits"] == {"train": "0:1"}, "info splits train 0:1")
        # the exact TCP extrinsic must be embedded (so the replay can recover the
        # tracker pose for the IK without guessing the matrix); it's the +0.21 m one.
        _ok("tcp_extrinsic" in info, "info embeds tcp_extrinsic matrix")
        Text = np.asarray(info["tcp_extrinsic"], dtype=np.float64)
        _ok(Text.shape == (4, 4) and abs(Text[0, 3] - 0.21) < 1e-9,
            "info tcp_extrinsic is the 4x4 +0.21 m matrix")
        _ok(info["features"]["states"]["shape"] == [13], "info states feature dim 13")
        _ok(info["features"]["action"]["shape"] == [7], "info action feature dim 7")
        _ok(info["video_path"].split("/")[2] == "egocentric", "video_path uses egocentric dir")

        # parquet
        pqs = []
        for root, _d, files in os.walk(os.path.join(out, "data")):
            pqs += [os.path.join(root, f) for f in files if f.endswith(".parquet")]
        _ok(len(pqs) == 1, "exactly one episode parquet")
        table = pq.read_table(pqs[0])
        for col in ("states", "action", "timestamp", "frame_index",
                    "episode_index", "index", "task_index", "next.done"):
            _ok(col in table.column_names, f"parquet has column {col}")
        state = np.asarray(table["states"].to_pylist(), dtype=np.float32)
        action = np.asarray(table["action"].to_pylist(), dtype=np.float32)
        T = state.shape[0]
        _ok(state.shape == (T, 13), "parquet state [T,13]")
        _ok(action.shape == (T, 7), "parquet action [T,7]")
        done = np.asarray(table["next.done"].to_pylist())
        _ok(done[-1] and not done[:-1].any(), "next.done only on last frame")
        fidx = np.asarray(table["frame_index"].to_pylist())
        _ok(np.array_equal(fidx, np.arange(T)), "frame_index 0..T-1")
        # timestamp column is uniform at fps
        ts = np.asarray(table["timestamp"].to_pylist(), dtype=np.float64)
        _ok(np.allclose(ts, np.arange(T) / 15.0, atol=1e-5), "timestamp uniform at fps")

        # The F/T slice of state must be the COMPENSATED value (Fx~0.2), NOT raw 9.0
        # (the shared core preferred the sidecar) — same assertion as the zarr test.
        ftx = state[:, 7]
        _ok(np.all(np.abs(ftx - 0.2) < 1e-3), f"state F/T slice is COMPENSATED (~0.2), got {ftx[0]:.3f}")

        # gripper width within [0, 0.095] and not all one rail (calib applied)
        gw = state[:, 6]
        _ok(gw.min() >= -1e-6 and gw.max() <= 0.095 + 1e-6, "gripper width within physical range")
        _ok(gw.max() - gw.min() > 0.01, "gripper width spans (not clamped to a rail)")

        # video present + decodable + frame count ~ T
        vids = []
        for root, _d, files in os.walk(os.path.join(out, "videos")):
            vids += [os.path.join(root, f) for f in files if f.endswith(".mp4")]
        _ok(len(vids) == 1, "exactly one episode mp4")
        import av
        c = av.open(vids[0])
        nframes = sum(1 for _ in c.decode(video=0))
        c.close()
        _ok(abs(nframes - T) <= 1, f"mp4 frame count ~ T ({nframes} vs {T})")

        # episodes.jsonl consistency
        eps = [json.loads(l) for l in open(os.path.join(out, "meta", "episodes.jsonl")) if l.strip()]
        _ok(len(eps) == 1 and eps[0]["length"] == T, "episodes.jsonl length == T")
        _ok(eps[0]["instruction"] == "pick block", "episodes.jsonl instruction")
        _ok(eps[0]["tasks"] == ["pick block"], "episodes.jsonl tasks is [task string]")
        _ok(eps[0]["dataset_from_index"] == 0 and eps[0]["dataset_to_index"] == T - 1,
            "episodes.jsonl dataset_from/to index")

        # stats.json carries EVERY low-dim column (LeRobot/GR00T normalization reads it)
        stats = json.load(open(os.path.join(out, "meta", "stats.json")))
        for key in ("states", "action", "timestamp", "frame_index", "episode_index",
                    "index", "task_index", "next.done"):
            _ok(key in stats, f"stats.json has key {key}")
        _ok(len(stats["states"]["mean"]) == 13, "stats states mean dim 13")
        _ok(len(stats["action"]["mean"]) == 7, "stats action mean dim 7")
        _ok(os.path.exists(os.path.join(out, "meta", "stats_psi0.json")), "stats_psi0.json copy written")

        # cross-check: the SAME raw take through convert_to_umi yields identical
        # eef_pos/rot/gripper/ft arrays (the two exporters share the core).
        gcal = umi.load_gripper_calib(session)
        Tt, _cal = umi.load_tcp_extrinsic(session)
        jd, ir = umi._make_image_fns()
        umi_eps = umi.build_episodes_from_take(
            os.path.join(session, fmt.take_dir_name(0)), gcal, Tt, (224, 224), jd, ir)
        ue = umi_eps[0]
        _ok(np.allclose(state[:, 0:3], ue["robot0_eef_pos"], atol=1e-5),
            "lerobot eef_pos == zarr-core eef_pos (shared builder)")
        _ok(np.allclose(state[:, 7:13], ue["robot0_ft_wrench"], atol=1e-5),
            "lerobot ft == zarr-core ft (shared builder)")

        # --overwrite re-runs cleanly
        rc2 = cl.main([session, "-o", out, "--task", "pick block", "--fps", "15", "--overwrite"])
        _ok(rc2 == 0, "main() --overwrite re-runs to 0")


def test_raw_only_provenance():
    """A raw-only take (no sidecar) -> info.ft_compensated == 'raw'."""
    import pyarrow.parquet as pq
    with tempfile.TemporaryDirectory() as tmp:
        session = os.path.join(tmp, "session")
        os.makedirs(session)
        _write_synthetic_take(session, n_cam=10, compensated=False)
        _write_calib(session)
        out = os.path.join(tmp, "out")
        _ok(cl.main([session, "-o", out, "--fps", "15"]) == 0, "raw-only main() ok")
        info = json.load(open(os.path.join(out, "meta", "info.json")))
        _ok(info["ft_compensated"] == "raw", "raw-only -> ft_compensated 'raw'")
        pqs = [os.path.join(r, f) for r, _d, fs in os.walk(os.path.join(out, "data"))
               for f in fs if f.endswith(".parquet")]
        state = np.asarray(pq.read_table(pqs[0])["states"].to_pylist(), dtype=np.float32)
        _ok(np.all(np.abs(state[:, 7] - 9.0) < 1e-3), "raw-only F/T slice is RAW (~9.0)")
        # video lives under the egocentric/ dir (canonical Psi0 layout)
        _ok(os.path.isdir(os.path.join(out, "videos", "chunk-000", "egocentric")),
            "video dir is .../egocentric/")


def test_validation_guards():
    """Up-front validation must fail BEFORE writing a partial dataset."""
    with tempfile.TemporaryDirectory() as tmp:
        session = os.path.join(tmp, "session")
        os.makedirs(session)
        _write_synthetic_take(session, n_cam=8, compensated=False)
        _write_calib(session)
        # odd --out-res rejected up front (no output dir created)
        out = os.path.join(tmp, "odd")
        _ok(cl.main([session, "-o", out, "--out-res", "223", "223"]) == 2, "odd out-res -> rc 2")
        _ok(not os.path.exists(out), "odd out-res leaves NO partial output dir")
        _ok(not os.path.exists(out + ".tmp"), "odd out-res leaves no .tmp")
        # fps < 1 rejected
        _ok(cl.main([session, "-o", out, "--fps", "0"]) == 2, "fps 0 -> rc 2")
        _ok(cl.main([session, "-o", out, "--fps", "-5"]) == 2, "fps -5 -> rc 2")
        # bogus session dir -> rc 1, not a traceback
        _ok(cl.main([os.path.join(tmp, "nope"), "-o", out]) == 1, "missing session -> rc 1")
        # exists without --overwrite -> rc 3
        good = os.path.join(tmp, "good")
        _ok(cl.main([session, "-o", good, "--fps", "15"]) == 0, "first convert ok")
        _ok(cl.main([session, "-o", good, "--fps", "15"]) == 3, "exists w/o --overwrite -> rc 3")
        # --overwrite refuses a non-dataset dir (the rmtree footgun guard)
        precious = os.path.join(tmp, "precious")
        os.makedirs(precious)
        open(os.path.join(precious, "important.txt"), "w").write("keep me")
        _ok(cl.main([session, "-o", precious, "--fps", "15", "--overwrite"]) == 3,
            "--overwrite on a non-dataset dir refused (rc 3)")
        _ok(os.path.exists(os.path.join(precious, "important.txt")), "precious file survived")
        # --overwrite on our OWN dataset is allowed
        _ok(cl.main([session, "-o", good, "--fps", "15", "--overwrite"]) == 0,
            "--overwrite on our own dataset ok")


def test_multi_episode_markers():
    """A take with 2 markers -> 3 episodes; verify per-episode parquet + index/
    dataset_from_index accounting across the multi-episode loop."""
    import pyarrow.parquet as pq
    with tempfile.TemporaryDirectory() as tmp:
        session = os.path.join(tmp, "session")
        os.makedirs(session)
        _write_synthetic_take(session, n_cam=30, compensated=True)
        _write_calib(session)
        # two markers inside the camera span -> 3 segments
        take = os.path.join(session, fmt.take_dir_name(0))
        cam_index = np.load(os.path.join(take, fmt.CAMERA_DIR, fmt.CAMERA_INDEX_FILE))
        cam_t = cam_index[:, 0]
        markers = np.array([cam_t[10], cam_t[20]], dtype=np.float64)
        np.save(os.path.join(take, fmt.MARKER_FILE), markers)
        out = os.path.join(tmp, "out")
        _ok(cl.main([session, "-o", out, "--fps", "15", "--task", "multi"]) == 0, "multi-ep main ok")
        info = json.load(open(os.path.join(out, "meta", "info.json")))
        _ok(info["total_episodes"] == 3, "3 episodes from 2 markers")
        eps = [json.loads(l) for l in open(os.path.join(out, "meta", "episodes.jsonl")) if l.strip()]
        _ok(len(eps) == 3, "episodes.jsonl has 3 rows")
        # contiguous, non-overlapping dataset index ranges
        cursor = 0
        ok_idx = True
        global_index = []
        for e in eps:
            ok_idx = ok_idx and e["dataset_from_index"] == cursor
            cursor = e["dataset_to_index"] + 1
            chunk = e["episode_index"] // cl.CHUNKS_SIZE
            pqp = os.path.join(out, "data", f"chunk-{chunk:03d}",
                               f"episode_{e['episode_index']:06d}.parquet")
            global_index += pq.read_table(pqp)["index"].to_pylist()
        _ok(ok_idx, "dataset_from_index contiguous across episodes")
        _ok(global_index == list(range(cursor)), "global 'index' column monotonic 0..N-1")
        _ok(info["total_frames"] == cursor, "total_frames == sum of episode lengths")


def test_uncalibrated_tcp_warning():
    """No calibration at all -> identity TCP -> info carries UNCALIBRATED_DO_NOT_TRAIN
    and the repo-default fallback is NOT present (temp session has no calib dir)."""
    with tempfile.TemporaryDirectory() as tmp:
        session = os.path.join(tmp, "session")
        os.makedirs(session)
        _write_synthetic_take(session, n_cam=8, compensated=True)
        # NO _write_calib: rely on repo defaults. The repo ships a calibrated TCP
        # extrinsic, so to test the UNCALIBRATED path we write an explicit identity.
        cdir = os.path.join(session, fmt.CALIB_DIR)
        os.makedirs(cdir, exist_ok=True)
        json.dump({"T_tracker_tcp": np.eye(4).tolist(), "calibrated": False},
                  open(os.path.join(cdir, fmt.TCP_EXTRINSIC_FILE), "w"))
        json.dump({"enc_closed": 0.0, "enc_open": 4.9, "width_closed_m": 0.0, "width_open_m": 0.095},
                  open(os.path.join(cdir, fmt.GRIPPER_RANGE_FILE), "w"))
        out = os.path.join(tmp, "out")
        _ok(cl.main([session, "-o", out, "--fps", "15"]) == 0, "uncalibrated main ok")
        info = json.load(open(os.path.join(out, "meta", "info.json")))
        _ok(info["tcp_calibrated"] is False, "tcp_calibrated False")
        _ok(info.get("WARNING") == "UNCALIBRATED_DO_NOT_TRAIN: TCP extrinsic is identity",
            "info carries UNCALIBRATED_DO_NOT_TRAIN")


if __name__ == "__main__":
    test_slices()
    test_state_action_assembly()
    test_modality_consistency()
    test_stats_block()
    test_timestamp_stats()
    test_round_trip()
    test_raw_only_provenance()
    test_validation_guards()
    test_multi_episode_markers()
    test_uncalibrated_tcp_warning()
    print("LOGIC OK")
