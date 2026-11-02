#!/usr/bin/env python
"""Headless eval gate for bota_ft_viz.

Starts the viser server on an ephemeral port, runs the SensorReader (live by
default, or synthetic with --mock), and asserts:
  * the reader decodes >= 1 frame,
  * the integrated orientation quaternion stays unit-norm,
  * the scene + GUI build and a draw pass runs without raising.

Prints ``SMOKE OK`` on success (exit 0), ``SMOKE FAIL: ...`` otherwise (exit 1).

Usage:
    python smoke_test.py --mock        # no hardware needed
    python smoke_test.py               # against the live sensor
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import viser

import bota_ft_viz as v


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--seconds", type=float, default=8.0, help="frame-wait deadline (hw init takes ~3s)")
    args = ap.parse_args()

    # 1) Quaternion math sanity (independent of hardware).
    q = np.array([1.0, 0.0, 0.0, 0.0])
    for _ in range(1000):
        q = v.integrate_gyro(q, np.array([0.0, 0.0, 1.0]), 0.001)
    assert abs(np.linalg.norm(q) - 1.0) < 1e-6, "quaternion drifted off unit norm"

    # 2) Build server + scene FIRST (matches main(): a busy reader during viser
    #    startup deadlocks via the GIL), then start the reader.
    state = v.SensorState()
    server = viser.ViserServer(host="127.0.0.1", port=0, label="smoke")
    scene = v.ForceTorqueScene(server, state)
    # Auto-detect the sensor (live mode), same as main().
    config_path = v._DEFAULT_CONFIG
    if not args.mock:
        port = v.find_bota_port()
        if port is None:
            print("SMOKE FAIL: no Bota sensor found under /dev/serial/by-id (try --mock)")
            return 1
        config_path = v.resolve_driver_config(v._DEFAULT_CONFIG, port)
    reader = v.SensorReader(state, config_path, mock=args.mock)
    reader.start()

    try:
        deadline = time.time() + args.seconds
        while time.time() < deadline:
            s = state.snapshot()
            if s.n_frames > 0:
                break
            if s.last_error:
                print(f"SMOKE FAIL: reader error: {s.last_error}")
                return 1
            time.sleep(0.05)

        sample = state.snapshot()
        if sample.n_frames == 0:
            print("SMOKE FAIL: no frames decoded (sensor busy/absent? try --mock)")
            return 1
        assert abs(np.linalg.norm(sample.quat_wxyz) - 1.0) < 1e-5, "live quaternion not unit-norm"

        # 3) A draw pass runs against live state without raising.
        for _ in range(5):
            scene.update(state.snapshot())
            time.sleep(0.02)
    finally:
        reader.stop()
        reader.join(timeout=2.0)
        server.stop()

    print(
        f"SMOKE OK  frames={sample.n_frames} rate={sample.rate_hz:.0f}Hz "
        f"F={v._fmt(sample.force)} T={v._fmt(sample.torque)} "
        f"acc={v._fmt(sample.acc)} gyro={v._fmt(sample.gyro)} "
        f"|q|={np.linalg.norm(sample.quat_wxyz):.6f}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
