"""--feet recorded + per-role rotation deglitch + glitch statistics (2026-09-11).

A synthetic 5pt session: five pucks, a there-and-back ORIENTATION glitch injected into the
CHEST stream (quiet position, 40 deg for 3 samples at ~77 Hz), one into the LEFT foot.
Pins that

* the two feet come out as body-reference keypoints (camera grid + hl_* windows),
* the rotation glitch is DROPPED from the label (no spike in head_rot_axis_angle) and its
  frames are MASKED (head_pose_glitch / pose_glitch_any / hl_head_action_valid) — while the
  take is KEPT (the old answer was --exclude),
* --no-rot-deglitch restores the spike (the contrast that proves the mechanism),
* attrs carry the per-role statistics + drop list.
"""
from __future__ import annotations

import json
import math
import os
import sys
import tempfile
import shutil

import numpy as np
import pytest
import zarr
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "recording"))
import recording_format as fmt  # noqa: E402
import convert_to_umi as conv  # noqa: E402
import umi_convert_core as core  # noqa: E402


def _tracker(tt, cx, cy, cz, amp, yaw_rate=0.7):
    rows = []
    for i, t in enumerate(tt):
        a = 0.3 * math.sin(yaw_rate * (t - tt[0]))
        rows.append([t, i, cx + amp * math.sin(0.5 * (t - tt[0])),
                     cy + amp * math.cos(0.5 * (t - tt[0])), cz + 0.05 * math.sin(t - tt[0]),
                     math.cos(a / 2), 0.0, 0.0, math.sin(a / 2), 0.0])
    return np.asarray(rows, dtype=np.float64)


def _inject_rot_glitch(tr: np.ndarray, k: int, deg: float, n: int = 3) -> None:
    """Rotate rows k+1..k+n by `deg` about x (a pure orientation excursion that rejoins)."""
    h = math.radians(deg) / 2
    qg = np.array([math.cos(h), math.sin(h), 0.0, 0.0])
    for i in range(k + 1, k + 1 + n):
        w1, x1, y1, z1 = qg
        w2, x2, y2, z2 = tr[i, 5:9]
        tr[i, 5:9] = [w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                      w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                      w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                      w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2]


def _write_synth_5pt(session_dir: str, head_glitch_k: int = 60, foot_glitch_k: int = 100) -> None:
    os.makedirs(os.path.join(session_dir, fmt.CALIB_DIR), exist_ok=True)
    json.dump({"mode": "5pt", "primary_role": "left", "quaternion_convention": "wxyz",
               "serial_role_map": {"LHR-L": "left", "LHR-R": "right", "LHR-H": "head",
                                   "LHR-LF": "left_foot", "LHR-RF": "right_foot"},
               "schema_version": 3, "clock": "time.time()"},
              open(os.path.join(session_dir, fmt.SESSION_META_FILE), "w"), indent=2)
    T = np.eye(4); T[0, 3] = 0.21
    json.dump({"T_tracker_tcp": T.tolist(), "calibrated": True, "frame": "wrist"},
              open(os.path.join(session_dir, fmt.CALIB_DIR, fmt.TCP_EXTRINSIC_FILE), "w"))
    json.dump({"enc_closed": -6.53, "enc_open": -1.25, "width_closed_m": 0.0, "width_open_m": 0.095},
              open(os.path.join(session_dir, fmt.CALIB_DIR, fmt.GRIPPER_RANGE_FILE), "w"))
    td = os.path.join(session_dir, "take_000")
    os.makedirs(os.path.join(td, fmt.CAMERA_DIR), exist_ok=True)
    N, rate, t0 = 240, 77.0, 1000.0
    tt = t0 + np.arange(N) / rate
    head = _tracker(tt, 0.0, 0.0, 1.3, 0.02)
    _inject_rot_glitch(head, head_glitch_k, 40.0, n=3)
    lf = _tracker(tt, -0.1, 0.1, 0.05, 0.01)
    _inject_rot_glitch(lf, foot_glitch_k, 25.0, n=2)
    np.save(os.path.join(td, fmt.TRACKER_FILE), _tracker(tt, 0.1, 0.3, 0.9, 0.08))
    np.save(os.path.join(td, fmt.TRACKER_RIGHT_FILE), _tracker(tt, 0.4, -0.2, 0.85, 0.06))
    np.save(os.path.join(td, fmt.TRACKER_HEAD_FILE), head)
    np.save(os.path.join(td, fmt.TRACKER_LEFT_FOOT_FILE), lf)
    np.save(os.path.join(td, fmt.TRACKER_RIGHT_FOOT_FILE), _tracker(tt, -0.1, -0.1, 0.05, 0.01))
    grip = [[t, -6.5 + 0.5 * (1 + math.sin(0.8 * (t - t0))) * 5.0, 0.0, -1.25, -6.53] for t in tt[::2]]
    np.save(os.path.join(td, fmt.GRIPPER_FILE), np.asarray(grip, dtype=np.float64))
    ftr = [[t, math.sin(2 * (t - t0)), math.cos(2 * (t - t0)), 1.0, 0.01, 0.02, 0.0]
           for t in np.arange(t0, tt[-1], 1 / 200.0)]
    np.save(os.path.join(td, fmt.FT_FILE), np.asarray(ftr, dtype=np.float64))
    np.save(os.path.join(td, fmt.MARKER_FILE), np.zeros((0, 1)))
    cam_idx = []
    for i, t in enumerate(tt[::3]):   # ~25.7 Hz camera grid
        Image.fromarray((np.random.rand(48, 64, 3) * 255).astype(np.uint8), "RGB").save(
            os.path.join(td, fmt.CAMERA_DIR, fmt.camera_frame_name(i)), "JPEG")
        cam_idx.append([t, i])
    np.save(os.path.join(td, fmt.CAMERA_DIR, fmt.CAMERA_INDEX_FILE), np.asarray(cam_idx, dtype=np.float64))
    json.dump({"task": "synth", "mode": "5pt", "primary_role": "left", "start_wall": float(t0),
               "stop_wall": float(tt[-1]), "ft_compensated": False, "quaternion_convention": "wxyz",
               "counts": {"tracker": N, "camera": len(cam_idx)}},
              open(os.path.join(td, fmt.TAKE_META_FILE), "w"), indent=2)


def _rot_step_deg(rotvec: np.ndarray) -> np.ndarray:
    """Per-frame angular change (deg) of a [T,3] rotvec series — a glitch is a spike here."""
    from scipy.spatial.transform import Rotation as R
    r = R.from_rotvec(rotvec.astype(np.float64))
    d = (r[1:] * r[:-1].inv()).magnitude()
    return np.degrees(d)


@pytest.fixture()
def sess():
    tmp = tempfile.mkdtemp()
    try:
        s = os.path.join(tmp, "session_synth_5pt")
        _write_synth_5pt(s)
        yield s
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _convert(sess, out, *extra):
    # No --hl-action-horizon override: core.set_grid is process-global and would leak into
    # the other test modules' contract-constant checks.
    rc = conv.main([sess, "-o", out, "--hands", "both_head_dual", "--out-res", "32", "32", *extra])
    assert rc == 0
    return zarr.open(zarr.ZipStore(out, mode="r"), mode="r")


def test_feet_recorded_emits_foot_keypoints_and_windows(sess, tmp_path):
    root = _convert(sess, str(tmp_path / "a.zarr.zip"), "--feet", "recorded")
    d = root["data"]
    for k in ("left_foot_pos", "left_foot_rot_axis_angle", "right_foot_pos", "right_foot_rot_axis_angle",
              "hl_left_foot_action_pose", "hl_left_foot_action_valid",
              "hl_right_foot_action_pose", "hl_right_foot_action_valid"):
        assert k in d, k
    T = d["camera0_rgb"].shape[0]
    assert d["left_foot_pos"].shape == (T, 3)
    assert d["hl_left_foot_action_pose"].shape == (T, core.ACTION_HORIZON, 6)
    # A foot is a body reference: NO TCP lever (its x stays at the raw puck x, unlike the hands).
    assert abs(float(np.mean(d["left_foot_pos"][:, 0])) - (-0.1)) < 0.03
    assert abs(float(np.mean(d["robot0_eef_pos"][:, 0])) - 0.1) > 0.15   # hands DO carry +0.21
    # No force/gripper keys for a foot.
    assert "hl_left_foot_action_force" not in d and "hl_left_foot_action_gripper" not in d
    a = root.attrs.asdict()
    assert a["feet_recorded"] is True and a["hands_mode"] == "both_head_dual"
    assert a["foot_extrinsic_calibrated"] is False
    assert a["T_tracker_left_foot"] == np.eye(4).tolist()


def test_feet_default_ignore_is_unchanged(sess, tmp_path):
    root = _convert(sess, str(tmp_path / "b.zarr.zip"))
    assert "left_foot_pos" not in root["data"] and "hl_left_foot_action_pose" not in root["data"]
    assert root.attrs.asdict()["feet_recorded"] is False


def test_feet_recorded_requires_hands(sess, tmp_path):
    with pytest.raises(SystemExit):
        conv.main([sess, "-o", str(tmp_path / "c.zarr.zip"), "--feet", "recorded"])


def test_rotation_glitch_is_dropped_masked_and_the_take_kept(sess, tmp_path):
    on = _convert(sess, str(tmp_path / "on.zarr.zip"), "--feet", "recorded")
    off = _convert(sess, str(tmp_path / "off.zarr.zip"), "--feet", "recorded", "--no-rot-deglitch")
    step_on = _rot_step_deg(on["data"]["head_rot_axis_angle"][:])
    step_off = _rot_step_deg(off["data"]["head_rot_axis_angle"][:])
    # The 40 deg spike reaches the label without the deglitch and is gone with it.
    assert step_off.max() > 15.0, f"contrast: spike expected in --no-rot-deglitch, got {step_off.max():.1f}"
    assert step_on.max() < 5.0, f"spike survived the rotation deglitch: {step_on.max():.1f} deg/frame"
    # The take survives (one episode, full length), i.e. NOT the old --exclude answer.
    assert on["data"]["camera0_rgb"].shape[0] == off["data"]["camera0_rgb"].shape[0] > 50
    # ...and the corrupted frames are MASKED per role: head + left_foot flagged, hands clean.
    d = on["data"]
    for k in ("head_pose_glitch", "left_foot_pose_glitch", "right_pose_glitch",
              "left_pose_glitch", "pose_glitch_any", "robot0_pose_glitch"):
        assert k in d, k
    assert d["head_pose_glitch"][:].sum() > 0
    assert d["left_foot_pose_glitch"][:].sum() > 0
    assert d["right_pose_glitch"][:].sum() == 0 and d["left_pose_glitch"][:].sum() == 0
    assert np.array_equal(d["left_pose_glitch"][:], d["robot0_pose_glitch"][:])
    assert (d["pose_glitch_any"][:] >= d["head_pose_glitch"][:]).all()
    # hl windows: the head's steps inside the excursion are zero-weighted, the hands' are not.
    assert d["hl_head_action_valid"][:].sum() < d["hl_left_action_valid"][:].sum()
    a = on.attrs.asdict()
    gs = a["glitch_stats"]
    assert gs["head"]["rot_excursions"] >= 1 and gs["head"]["rot_samples_dropped"] >= 3
    assert gs["left_foot"]["rot_excursions"] >= 1
    assert gs["left"]["rot_excursions"] == 0 and gs["right"]["rot_excursions"] == 0
    assert gs["head"]["frames_flagged"] > 0 and gs["head"]["hl_steps_masked"] > 0
    assert a["rot_deglitch"] is True and a["takes_dropped"] == [] and a["takes_ingested"] == 1
    assert sorted(a["pose_glitch_roles"]) == ["head", "left", "left_foot", "right", "right_foot"]
    assert off.attrs.asdict()["glitch_stats"]["head"]["rot_excursions"] == 0
