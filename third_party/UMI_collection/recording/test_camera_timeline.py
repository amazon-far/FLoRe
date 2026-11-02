#!/usr/bin/env python3
"""Recorder-side guard for the camera-timeline uniformity invariant.

The camera grid is the MASTER TIMELINE of the training dataset and both training
loaders index windows in FRAME units with no dt input, so a non-uniform grid is
multiplicative noise on every velocity / dF-dt label. A 11.90 Hz grid with dt
alternating between 2 and 3 capture periods shipped 97 takes without a single
warning -- `approx_hz.camera = 11.8` was written into all 97 take_meta.json files
and nothing read it. These tests pin the detector so that cannot recur.

Run:  .venv/bin/python recording/test_camera_timeline.py     (prints "TIMELINE OK")
"""
from __future__ import annotations

import base64
import io
import json
import os
import shutil
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from recorder import Recorder  # noqa: E402

JPEG = base64.b64encode(b"\xff\xd8\xff\xd9").decode("ascii")  # not decoded by the recorder


def _feed(rec, cam_k, n=200, fps=30.0, t0=1_800_000_000.0):
    """Feed n camera frames whose capture gaps follow the `cam_k` sequence."""
    t = t0
    cap = 1000
    for i in range(n):
        k = cam_k[i % len(cam_k)]
        rec.on_camera({"t": t, "seq": i + 1, "capture_seq": cap, "w": 8, "h": 8,
                       "jpeg_b64": JPEG})
        t += k / fps
        cap += k
    # A take needs BOTH camera and tracker frames to be saved at all.
    for j in range(400):
        # legacy flat tracker format: key is "quat" (WXYZ), not "quat_wxyz"
        rec.on_tracker({"t": t0 + j * (1.0 / 77.7), "pos": [0.0, 0.0, 0.0],
                        "quat": [1.0, 0.0, 0.0, 0.0]})


def _run(cam_k, n=200):
    tmp = tempfile.mkdtemp(prefix="camtl_")
    try:
        rec = Recorder(tmp, session_name="s", mode="1pt", camera_fps_nominal=30.0)
        rec.start(task="t")
        _feed(rec, cam_k, n=n)
        rec.stop()
        take = os.path.join(tmp, "s", "take_000")
        with open(os.path.join(take, "take_meta.json")) as f:
            return json.load(f)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_uniform_grid_is_accepted():
    m = _run([2])
    assert m["camera_grid_uniform"] is True, m
    assert m["camera_dt_k"] == {"2": 199}, m["camera_dt_k"]
    assert m["camera_capture_gaps"] == {"2": 199}, m["camera_capture_gaps"]
    assert m["camera_seq_gaps"] == 0
    assert abs(m["camera_dt_median"] - 2 / 30.0) < 1e-6
    print("  uniform 15 Hz grid accepted: OK")


def test_the_shipped_defect_is_detected():
    """The exact pattern measured on disk: 2/3 capture-period alternation."""
    rng = np.random.default_rng(0)
    cam_k = [2 if x else 3 for x in rng.integers(0, 2, size=400)]
    m = _run(cam_k, n=300)
    assert m["camera_grid_uniform"] is False, m
    ks = m["camera_dt_k"]
    assert set(ks) == {"2", "3"}, ks
    assert set(m["camera_capture_gaps"]) == {"2", "3"}, m["camera_capture_gaps"]
    print(f"  2/3 alternation detected: dt_k={ks} uniform=False: OK")


def test_lost_frames_are_detected():
    """A `seq` gap means the bridge produced an image that never arrived."""
    tmp = tempfile.mkdtemp(prefix="camtl_")
    try:
        rec = Recorder(tmp, session_name="s", mode="1pt", camera_fps_nominal=30.0)
        rec.start(task="t")
        t = 1_800_000_000.0
        seq = 0
        cap = 1000
        for i in range(100):
            seq += 2 if i == 50 else 1  # one frame lost at i==50
            rec.on_camera({"t": t, "seq": seq, "capture_seq": cap, "w": 8, "h": 8,
                           "jpeg_b64": JPEG})
            t += 2 / 30.0
            cap += 2
        for j in range(200):
            rec.on_tracker({"t": 1_800_000_000.0 + j / 77.7, "pos": [0, 0, 0],
                            "quat": [1, 0, 0, 0]})
        rec.stop()
        with open(os.path.join(tmp, "s", "take_000", "take_meta.json")) as f:
            m = json.load(f)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    assert m["camera_seq_gaps"] == 1, m["camera_seq_gaps"]
    assert m["camera_grid_uniform"] is False, "a lost frame must fail the gate"
    print("  lost frame (seq gap) detected and fails the gate: OK")


def main() -> int:
    test_uniform_grid_is_accepted()
    test_the_shipped_defect_is_detected()
    test_lost_frames_are_detected()
    print("TIMELINE OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
