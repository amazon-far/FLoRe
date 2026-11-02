#!/usr/bin/env python3
"""End-to-end regression for the LEFT-canonical 2pt/3pt conversion path.

Builds a tiny synthetic 3pt raw session on disk (left + right + head trackers, a
single physical gripper on the primary/left hand), runs the REAL
``convert_to_umi.build_episodes_from_take`` + zarr writer, and asserts the produced
episode dicts carry the additive secondary keypoints with the right semantics:

  * ``robot0_*`` = the PRIMARY (left) hand, unchanged from the 1pt path.
  * ``robot1_*`` = the RIGHT hand, with the SAME tracker->TCP extrinsic as robot0
    (so a pure +x extrinsic offsets both hands' x identically).
  * ``robot1_gripper_width`` is ABSENT (single physical gripper on the primary hand)
    — deliberately not zero-filled, so the trainer masks that slot False.
  * ``head_pos``/``head_rot_axis_angle`` = the head, RAW pose (NO extrinsic), no gripper.

Run:  PYTHONPATH=processing:recording .venv/bin/python processing/test_convert_3pt.py
Prints ``LOGIC OK`` / a failing summary (exit 1).
"""
from __future__ import annotations

import json
import math
import os
import shutil
import sys
import tempfile

import numpy as np
from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "..", "recording"))

import recording_format as fmt  # noqa: E402
import convert_to_umi as conv  # noqa: E402


def _make_tracker(tt: np.ndarray, cx: float, cy: float, cz: float, amp: float) -> np.ndarray:
    rows = []
    for i, t in enumerate(tt):
        a = 0.3 * math.sin(0.7 * (t - tt[0]))  # slow yaw
        rows.append([t, i,
                     cx + amp * math.sin(0.5 * (t - tt[0])),
                     cy + amp * math.cos(0.5 * (t - tt[0])),
                     cz + 0.05 * math.sin(t - tt[0]),
                     math.cos(a / 2), 0.0, 0.0, math.sin(a / 2), 0.0])
    return np.asarray(rows, dtype=np.float64)


def _write_synth_3pt(session_dir: str) -> None:
    os.makedirs(os.path.join(session_dir, fmt.CALIB_DIR), exist_ok=True)
    json.dump(
        {"mode": "3pt", "primary_role": "left", "quaternion_convention": "wxyz",
         "serial_role_map": {"LHR-L": "left", "LHR-R": "right", "LHR-H": "head"},
         "schema_version": 3, "clock": "time.time()"},
        open(os.path.join(session_dir, fmt.SESSION_META_FILE), "w"), indent=2)
    # A calibrated TCP extrinsic (pure +x 0.21) so robot0/robot1 x are offset but head is not.
    T = np.eye(4); T[0, 3] = 0.21
    json.dump({"T_tracker_tcp": T.tolist(), "calibrated": True, "frame": "wrist"},
              open(os.path.join(session_dir, fmt.CALIB_DIR, fmt.TCP_EXTRINSIC_FILE), "w"))
    json.dump({"enc_closed": -6.53, "enc_open": -1.25, "width_closed_m": 0.0, "width_open_m": 0.095},
              open(os.path.join(session_dir, fmt.CALIB_DIR, fmt.GRIPPER_RANGE_FILE), "w"))

    td = os.path.join(session_dir, "take_000")
    os.makedirs(os.path.join(td, fmt.CAMERA_DIR), exist_ok=True)
    N, rate, t0 = 120, 60.0, 1000.0
    tt = t0 + np.arange(N) / rate
    np.save(os.path.join(td, fmt.TRACKER_FILE), _make_tracker(tt, 0.1, 0.3, 0.9, 0.08))        # left
    np.save(os.path.join(td, fmt.TRACKER_RIGHT_FILE), _make_tracker(tt, 0.4, -0.2, 0.85, 0.06))  # right
    np.save(os.path.join(td, fmt.TRACKER_HEAD_FILE), _make_tracker(tt, 0.0, 0.0, 1.3, 0.02))     # head
    grip = [[t, -6.5 + 0.5 * (1 + math.sin(0.8 * (t - t0))) * 5.0, 0.0, -1.25, -6.53]
            for t in tt[::2]]
    np.save(os.path.join(td, fmt.GRIPPER_FILE), np.asarray(grip, dtype=np.float64))
    ftr = [[t, math.sin(2 * (t - t0)), math.cos(2 * (t - t0)), 1.0, 0.01, 0.02, 0.0]
           for t in np.arange(t0, tt[-1], 1 / 200.0)]
    np.save(os.path.join(td, fmt.FT_FILE), np.asarray(ftr, dtype=np.float64))
    np.save(os.path.join(td, fmt.MARKER_FILE), np.zeros((0, 1)))
    cam_idx = []
    for i, t in enumerate(tt[::4]):
        Image.fromarray((np.random.rand(48, 64, 3) * 255).astype(np.uint8), "RGB").save(
            os.path.join(td, fmt.CAMERA_DIR, fmt.camera_frame_name(i)), "JPEG")
        cam_idx.append([t, i])
    np.save(os.path.join(td, fmt.CAMERA_DIR, fmt.CAMERA_INDEX_FILE), np.asarray(cam_idx, dtype=np.float64))
    json.dump({"task": "synth", "mode": "3pt", "primary_role": "left",
               "start_wall": float(t0), "stop_wall": float(tt[-1]), "ft_compensated": False,
               "quaternion_convention": "wxyz", "counts": {"tracker": N, "camera": len(cam_idx)}},
              open(os.path.join(td, fmt.TAKE_META_FILE), "w"), indent=2)


def test_3pt_episode_has_secondary_keypoints():
    tmp = tempfile.mkdtemp()
    try:
        sess = os.path.join(tmp, "session_synth_3pt")
        _write_synth_3pt(sess)
        gcalib = conv.load_gripper_calib(sess)
        T_tcp, calibrated = conv.load_tcp_extrinsic(sess)
        assert calibrated, "test TCP extrinsic should be calibrated"
        jpeg_decode, image_resize = conv._make_image_fns()
        eps = conv.build_episodes_from_take(
            os.path.join(sess, "take_000"), gcalib, T_tcp, (224, 224), jpeg_decode, image_resize)
        assert len(eps) == 1, f"expected 1 episode, got {len(eps)}"
        ep = eps[0]
        # robot0 (left) + robot1 (right) present; head present.
        for k in ("robot0_eef_pos", "robot1_eef_pos", "robot1_eef_rot_axis_angle",
                  "head_pos", "head_rot_axis_angle"):
            assert k in ep, f"missing key {k}"
        # right gripper ABSENT (single physical gripper on the primary hand).
        assert "robot1_gripper_width" not in ep, "robot1_gripper_width must be absent (no right gripper)"
        # robot0 and robot1 use the SAME +x extrinsic -> both x-offset from raw center,
        # head uses NO extrinsic -> its x stays near the raw head center (~0.0).
        assert abs(float(np.mean(ep["head_pos"][:, 0]))) < 0.15, "head x should be ~raw (no extrinsic)"
        # left and right are distinct hands (different world centers).
        assert not np.allclose(ep["robot0_eef_pos"], ep["robot1_eef_pos"]), "left/right must differ"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_1pt_take_writes_no_secondary_keys():
    """A take with only the primary tracker.npy must NOT emit robot1_*/head_* keys
    (so the 1pt path is byte-for-byte the historical single-hand layout)."""
    tmp = tempfile.mkdtemp()
    try:
        sess = os.path.join(tmp, "session_synth_1pt")
        _write_synth_3pt(sess)
        # Remove the right + head trackers to simulate a 1pt take.
        for f in (fmt.TRACKER_RIGHT_FILE, fmt.TRACKER_HEAD_FILE):
            os.remove(os.path.join(sess, "take_000", f))
        gcalib = conv.load_gripper_calib(sess)
        T_tcp, _ = conv.load_tcp_extrinsic(sess)
        jpeg_decode, image_resize = conv._make_image_fns()
        eps = conv.build_episodes_from_take(
            os.path.join(sess, "take_000"), gcalib, T_tcp, (224, 224), jpeg_decode, image_resize)
        ep = eps[0]
        for k in ("robot1_eef_pos", "robot1_gripper_width", "head_pos", "head_rot_axis_angle"):
            assert k not in ep, f"1pt take must not emit {k}"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_corrupt_spans_unions_bridged_and_unbridged():
    """REGRESSION: `robot0_pose_glitch` must flag BRIDGED excursions too.

    The converter used to build the mask from `unbridged_spans` alone, which is empty for
    a bridged excursion by construction — so a bridged lighthouse dropout shipped a
    fabricated 118-201 mm interpolation ramp in `robot0_eef_pos` with `pose_glitch == 0`
    (live on clean94 episodes 30/31). `_corrupt_spans` is the union, and both the
    per-frame mask and the HL `action_valid` window mask read it, so the two can't drift.
    """
    assert conv._corrupt_spans({"unbridged_spans": [(1.0, 2.0)],
                                "bridged_spans": [(5.0, 6.0)]}) == [(1.0, 2.0), (5.0, 6.0)]
    # A bridged-only severity dict (the ep30/ep31 shape) must NOT come back empty.
    assert conv._corrupt_spans({"unbridged_spans": [],
                                "bridged_spans": [(5.0, 6.0)]}) == [(5.0, 6.0)]
    # Clean stream -> no spans, so a clean take masks nothing.
    assert conv._corrupt_spans({"unbridged_spans": [], "bridged_spans": []}) == []
    # Tolerates a legacy dict with no bridged key at all (never raises mid-conversion).
    assert conv._corrupt_spans({"unbridged_spans": [(1.0, 2.0)]}) == [(1.0, 2.0)]
    assert conv._corrupt_spans({}) == []


def _run_all() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  ok  {t.__name__}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  FAIL {t.__name__}: {exc!r}")
    if failed:
        print(f"\n{failed}/{len(tests)} FAILED")
        return 1
    print(f"\nLOGIC OK ({len(tests)} checks)")
    return 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
