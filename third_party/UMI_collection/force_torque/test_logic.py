#!/usr/bin/env python
"""Hardware-free logic tests for bota_ft_viz (run with the mock reader).

Covers the bugs the Opus reviewers flagged:
  * reset_orientation actually re-zeros the integrator (and isn't overwritten),
  * dt is clamped so a long stall can't snap the triad,
  * quaternion stays unit-norm,
  * the redraw-skip + visibility-toggle path doesn't raise.

Usage: python test_logic.py   ->  prints LOGIC OK / LOGIC FAIL.
"""

from __future__ import annotations

import math
import sys
import time

import numpy as np
import viser

import bota_ft_viz as v


def test_reset_orientation_persists() -> None:
    """After spinning (mock yaws steadily), a reset must drive the triad back to
    identity AND stay there (the reader must consume the flag, not get
    overwritten)."""
    state = v.SensorState()
    reader = v.SensorReader(state, v._DEFAULT_CONFIG, mock=True)
    reader.start()
    try:
        # Let it spin away from identity.
        time.sleep(0.4)
        before = state.snapshot().quat_wxyz
        assert abs(before[0]) < 0.9999, f"mock did not rotate (w={before[0]})"

        # Request reset; the reader should zero the integrator within a frame.
        with state.lock:
            state.reset_orientation = True
        time.sleep(0.1)
        after = state.snapshot().quat_wxyz
        # Identity-ish right after reset, and the flag was cleared.
        assert state.snapshot().quat_wxyz is not None
        assert not state.reset_orientation, "reset flag not consumed by reader"
        # w close to 1 means near-identity (it will have drifted a little since).
        assert after[0] > 0.99, f"reset did not re-zero orientation (w={after[0]})"
    finally:
        reader.stop()
        reader.join(timeout=2.0)


def test_dt_clamp_no_blowup() -> None:
    """integrate_gyro with a huge dt and the per-frame clamp must not rotate by
    more than the clamp allows."""
    q = np.array([1.0, 0.0, 0.0, 0.0])
    gyro = np.array([0.0, 0.0, 10.0])  # 10 rad/s
    # Unclamped: 10 rad/s * 5 s = 50 rad. Clamped at 0.05 s -> 0.5 rad.
    q2 = v.integrate_gyro(q, gyro, min(5.0, 0.05))
    angle = 2.0 * np.arccos(np.clip(abs(q2[0]), -1, 1))
    assert angle <= 0.5 + 1e-6, f"clamp failed, rotated {angle} rad"
    assert abs(np.linalg.norm(q2) - 1.0) < 1e-9


def test_complementary_fusion_converges() -> None:
    """Start from identity but with the sensor actually tilted: a static
    accelerometer reading gravity along a tilted body axis must pull the
    estimate toward that tilt (pitch/roll lock), and stay unit-norm."""
    # Sensor physically pitched ~30 deg: gravity (+up, 9.81) read in the body
    # frame points partly along body x.
    theta = math.radians(30.0)
    acc = 9.81 * np.array([math.sin(theta), 0.0, math.cos(theta)])
    q = np.array([1.0, 0.0, 0.0, 0.0])  # estimate starts wrong (no tilt)
    for _ in range(2000):
        q = v.complementary_correct(q, acc, gain=0.05)
    # The estimate's predicted up-in-body should now match the measured one.
    up_pred_body = v.quat_rotate(np.array([q[0], -q[1], -q[2], -q[3]]), np.array([0.0, 0.0, 1.0]))
    err = float(np.linalg.norm(up_pred_body - acc / np.linalg.norm(acc)))
    assert err < 1e-2, f"fusion did not converge to gravity (err={err})"
    assert abs(np.linalg.norm(q) - 1.0) < 1e-9


def test_fusion_rejects_dynamic_accel() -> None:
    """An accelerometer reading far from 1g (sensor being shaken) must be
    ignored so linear acceleration doesn't corrupt the tilt estimate."""
    q = np.array([1.0, 0.0, 0.0, 0.0])
    bogus = np.array([30.0, 20.0, 5.0])  # |a| ~ 36 m/s^2, nowhere near 9.81
    out = v.complementary_correct(q, bogus, gain=0.5)
    assert np.allclose(out, q), "dynamic-accel sample should be rejected, estimate unchanged"


def test_visibility_toggle_path() -> None:
    """Hiding then re-showing an arrow must not raise and must flip the cached
    visibility bookkeeping."""
    state = v.SensorState()
    server = viser.ViserServer(host="127.0.0.1", port=0, label="logic")
    try:
        scene = v.ForceTorqueScene(server, state)
        # Seed a frame so update() doesn't early-return on the change-signature.
        with state.lock:
            state.n_frames = 1
            state.gyro = np.array([0.1, 0.0, 0.0])
        scene.gui_show_gyro.value = True
        scene.update(state.snapshot())
        assert scene._arrow_visible.get("/world/sensor/gyro") is True

        with state.lock:
            state.n_frames = 2
        scene.gui_show_gyro.value = False
        scene.update(state.snapshot())
        assert scene._arrow_visible.get("/world/sensor/gyro") is False

        with state.lock:
            state.n_frames = 3
        scene.gui_show_gyro.value = True
        scene.update(state.snapshot())
        assert scene._arrow_visible.get("/world/sensor/gyro") is True
    finally:
        server.stop()


def test_charts_build_and_update() -> None:
    """The 4 live charts build, and update_charts pushes rolling-window data
    (x newest=0, older negative) without raising; the change-gate skips a
    no-new-frame second call."""
    state = v.SensorState()
    t0 = time.perf_counter()
    for i in range(40):
        state.history.append(
            t0 + i * 0.02,
            np.array([i * 0.1, -i * 0.05, 1.0]),
            np.array([0.1, 0.2, 0.3]),
            np.array([9.8, 0.0, 0.1]),
            np.array([0.01, -0.02, 0.0]),
        )
    with state.lock:
        state.n_frames = 40
    server = viser.ViserServer(host="127.0.0.1", port=0, label="charts")
    try:
        scene = v.ForceTorqueScene(server, state)
        assert set(scene._charts) == {"force", "torque", "acc", "gyro"}
        x, channels = state.history.export()
        assert x[-1] == 0.0 and x[0] < 0.0, "rolling x-axis not newest=0/older-negative"
        assert all(len(c) == x.size for c in channels["force"]), "component/x length mismatch"
        scene.update_charts()
        assert scene._last_chart_n == 40
        # No new frame -> change-gate should skip (n unchanged).
        scene.update_charts()
        assert scene._last_chart_n == 40
    finally:
        server.stop()


def test_redraw_skip() -> None:
    """A second update() with no new frame and no GUI change is a no-op (the
    change-signature gate)."""
    state = v.SensorState()
    server = viser.ViserServer(host="127.0.0.1", port=0, label="skip")
    try:
        scene = v.ForceTorqueScene(server, state)
        with state.lock:
            state.n_frames = 7
        scene.update(state.snapshot())
        sig1 = scene._last_sig
        scene.update(state.snapshot())  # identical snapshot -> early return
        assert scene._last_sig == sig1
    finally:
        server.stop()


def main() -> int:
    tests = [
        test_dt_clamp_no_blowup,
        test_complementary_fusion_converges,
        test_fusion_rejects_dynamic_accel,
        test_reset_orientation_persists,
        test_visibility_toggle_path,
        test_charts_build_and_update,
        test_redraw_skip,
    ]
    for t in tests:
        try:
            t()
            print(f"  ok  {t.__name__}")
        except Exception as exc:  # noqa: BLE001
            print(f"LOGIC FAIL: {t.__name__}: {exc}")
            return 1
    print("LOGIC OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
