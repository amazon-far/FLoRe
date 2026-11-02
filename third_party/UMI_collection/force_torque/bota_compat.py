#!/usr/bin/env python
"""Bota Control Toolkit generation-A (>= 1.0) compatibility helpers.

The rig was written against ``bota_payload_utils`` **0.0.2**, which shipped a
``create_ft_sensor_hwi(driver, imu_offset=...)`` wrapper: it owned the driver
lifecycle, did the IMU -> sensor-frame transform internally, and fed the
estimator/compensator via ``connect_input_block`` + a no-argument ``update()``.

``bota_payload_utils`` **1.0.2** (the Nov-2025 "Bota Control Toolkit syntax"
demo line) REMOVED that wrapper. The caller now:

  * drives the ``BotaDriver`` lifecycle directly (``configure``/``activate``/
    ``read_frame_blocking``/``deactivate``/``shutdown``), and
  * assembles the 12-element vector itself -
    ``[fx, fy, fz, tx, ty, tz, ax, ay, az, wx, wy, wz]`` - where the wrench is
    taken straight from the driver frame and the IMU (accel + gyro) is
    transformed from the IMU frame into the SENSOR frame via
    ``bota_tf_utils.BotaTfImu`` (the same fixed ``[0, 0, -0.0257] m`` SensONE/M8
    offset the Bota demo uses), then
  * calls ``estimator.update(timestamp, wrench_imu_raw12)`` /
    ``compensator.update(timestamp, wrench_imu_raw12, wrench_imu_compensated12)``.

The underlying EKF is unchanged, so the compensated wrench is numerically the
same as 0.0.2 produced - this is purely an API-shape migration.

TWO sharp edges this module exists to centralize:

  1. ``update()`` now returns a ``bota_control.BotaControlReturnCode`` ENUM, not a
     bool. The enum is **always truthy** (``bool(RC.FatalError) is True``) and is
     **not int-convertible**, so the old ``if not comp.update(): raise`` pattern
     would NEVER fire. Use :func:`rc_is_fatal` / compare to ``RC.OK`` instead.
  2. ``tare_compensated_wrench()`` is now **asynchronous**: it collects a window
     (~500 samples / ~0.5 s at 1 kHz) before the compensated output is valid;
     until then ``update()`` returns ``Stale`` and writes zeros. Call
     :func:`warm_up_tare` after ``tare_compensated_wrench()`` before trusting the
     compensated output (or capturing tare provenance).

This module imports ``bota_control`` + ``bota_tf_utils`` at import time, so it is
only importable where the gen-A toolkit is installed (the rig ``.venv``).
"""
from __future__ import annotations

import time

import numpy as np

import bota_control as bc
import bota_tf_utils as tf

# wrench_frame -> imu_frame offset (m) for the M8-series; matches the Bota demo
# (``01_/02_bota_payload_*_demo.py``). Leave as-is unless Bota specifies a
# different value for your sensor.
IMU_OFFSET = (0.0, 0.0, -0.0257)

_RC = bc.BotaControlReturnCode

# A transient bad dt makes update() return FatalError for a sample or two, then
# RECOVERS on the next clean timestamp (verified: the internal EMA dt filter
# returns to OK immediately once the timestamps are sane again). This happens
# when frames arrive faster than the clock advances - e.g. a buffered-frame burst
# right after the driver activates, or right after a recalibration drained the
# stream. So a SINGLE fatal must NOT tear down the reader thread (0.0.2 had no
# such exposure). Callers tolerate up to this many CONSECUTIVE fatals (~0.2 s at
# the 1 kHz sensor rate) before escalating to a genuine "stream is broken" error.
FATAL_TOLERANCE = 200


def rc_is_fatal(rc) -> bool:
    """True only for ``FatalError`` (the single code the Bota demo stops on)."""
    return rc == _RC.FatalError


def rc_is_ok(rc) -> bool:
    """True for a fully-valid ``OK`` sample (tare settled, no degradation)."""
    return rc == _RC.OK


def make_imu_tf(imu_offset=IMU_OFFSET, name: str = "imuF_in_imuF_to_sensor_in_sensor"):
    """Build the IMU-frame -> sensor-frame transform block, configured EXACTLY as
    the Bota demo (identity rotation, ``imu_offset`` reference-point translation)."""
    cfg = tf.BotaTfConfig()
    cfg.name = name
    cfg.policy = bc.BotaControlBlockPolicy.StrictRt
    block = tf.BotaTfImu(cfg)
    block.set_input_reference_point("fixed_frame", np.asarray(imu_offset, dtype=np.float64))
    block.set_input_expression_axes("fixed_frame", np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64))
    block.set_output_reference_point("fixed_frame", np.array([0.0, 0.0, 0.0], dtype=np.float64))
    block.set_output_expression_axes("fixed_frame", np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64))
    return block


class FtChain:
    """IMU transform + reusable buffers for the gen-A wrench-assembly path.

    Owns the ``BotaTfImu`` block (which may keep internal filter state, so make
    ONE per session and reuse it across the live loop AND any recalibration -
    re-creating it mid-stream resets that state). ``assemble(frame)`` returns the
    12-vector ``[wrench(6), imu_in_sensor(6)]`` to hand to ``estimator.update`` /
    ``compensator.update``. Buffers are reused (no per-frame allocation, matching
    the demo's RT loop); the returned array is overwritten on the next call, so
    consume it before calling ``assemble`` again."""

    def __init__(self, imu_offset=IMU_OFFSET):
        self._tf = make_imu_tf(imu_offset)
        self._imu_in = np.zeros(6, dtype=np.float64)
        self._imu_out = np.zeros(6, dtype=np.float64)
        self._raw12 = np.zeros(12, dtype=np.float64)
        # Scratch the compensator writes its 12-vector output into.
        self.comp12 = np.zeros(12, dtype=np.float64)
        # Return code of the most recent IMU TF update(); ``assemble`` records it
        # so callers can fold a TF failure into their fatal-tolerance logic (the
        # upstream demo guards this update with loop_react). FatalError here means
        # the IMU-in-sensor block is garbage -> the 12-vector's IMU half is bad.
        self.last_tf_rc = _RC.OK

    def assemble(self, frame) -> np.ndarray:
        """Fill + return the raw 12-vector for ``frame`` (a ``BotaDriver`` frame).

        Transforms the IMU into the sensor frame via the TF block, so the 12-vec
        layout is the one the estimator/compensator expect. The TF update's return
        code is stored on ``self.last_tf_rc`` (the demo guards it via loop_react);
        callers escalate via :func:`tf_is_fatal`."""
        self._imu_in[:3] = frame.acceleration
        self._imu_in[3:] = frame.angular_rate
        self.last_tf_rc = self._tf.update(self._imu_in, self._imu_out)
        self._raw12[:3] = frame.force
        self._raw12[3:6] = frame.torque
        self._raw12[6:] = self._imu_out
        return self._raw12

    def tf_is_fatal(self) -> bool:
        """True if the last IMU TF update returned FatalError (IMU half is garbage)."""
        return self.last_tf_rc == _RC.FatalError


def warm_up_tare(driver, comp, chain: FtChain, timeout_s: float = 3.0,
                 stop_event=None, frame_sink=None) -> bool:
    """Pump frames through ``comp`` until its asynchronous tare settles.

    1.0.2's ``tare_compensated_wrench()`` defers: it averages a window before the
    compensated output is meaningful, returning ``Stale`` (zeros) until then.
    This drains frames until ``update()`` returns ``OK`` (tare settled) or
    ``timeout_s`` elapses. Returns True if the tare settled. Safe to call right
    after ``comp.tare_compensated_wrench()``; the discarded samples are the brief
    (~0.5 s) startup window.

    ``frame_sink`` (optional): called with each drained ``frame`` BEFORE it is
    consumed by the tare. The recorder uses this to keep the RAW wrench flowing
    during the ~0.5 s tare window (which otherwise leaves a hole at every episode
    start); the compensated output is Stale/zeros here so only raw is meaningful,
    which is exactly what the sink records for this window."""
    deadline = time.perf_counter() + timeout_s
    while time.perf_counter() < deadline:
        if stop_event is not None and stop_event.is_set():
            return False
        frame = driver.read_frame_blocking()
        if frame_sink is not None:
            frame_sink(frame)
        rc = comp.update(time.perf_counter(), chain.assemble(frame), chain.comp12)
        if rc_is_ok(rc):
            return True
        if rc_is_fatal(rc):
            return False
    return False
