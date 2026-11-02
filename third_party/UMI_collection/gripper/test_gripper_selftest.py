#!/usr/bin/env python3
"""Unit tests for the pure decision logic of the gripper startup self-test.

Covers the hardware-free pieces of ``gripper_selftest``: stall detection,
endpoint classification, full-stroke traversal check, and the machine-readable
RESULT line the launcher parses. The hardware runner (auto-home + verify) needs a
real DM4310 and is exercised by ``run_collection_viz.sh`` on launch, not here.

Run:
    PYTHONPATH=third_party/i2rt:gripper .venv/bin/python -m pytest gripper/test_gripper_selftest.py -q
"""
from __future__ import annotations

import math

import pytest

from gripper_selftest import (
    RESULT_PREFIX,
    SelfTestResult,
    _safe_temp,
    classify_endpoints,
    is_stalled,
    traversed_ok,
)


class _TempState:
    """Minimal stand-in for a MotorInfo with rotor/MOS temps."""

    def __init__(self, temp_rotor=-1, temp_mos=-1):
        self.temp_rotor = temp_rotor
        self.temp_mos = temp_mos


class TestSafeTemp:
    """_safe_temp must keep a genuine 0 C distinct from the -1 'no telemetry'
    sentinel — the `0.0 or -1` idiom folded them together and silently disarmed the
    over-temp guard for a 0-reading sample."""

    def test_zero_is_kept_not_folded_to_minus_one(self):
        assert _safe_temp(_TempState(temp_rotor=0.0), "temp_rotor") == 0.0

    def test_real_value_passthrough(self):
        assert _safe_temp(_TempState(temp_rotor=42.0), "temp_rotor") == 42.0

    def test_minus_one_sentinel_is_missing(self):
        assert _safe_temp(_TempState(temp_rotor=-1), "temp_rotor") == -1.0

    def test_none_is_missing(self):
        assert _safe_temp(_TempState(temp_rotor=None), "temp_rotor") == -1.0

    def test_missing_attr_is_missing(self):
        assert _safe_temp(object(), "temp_rotor") == -1.0


class TestIsStalled:
    def test_needs_full_window(self):
        # Too few samples -> can't decide -> not stalled.
        assert is_stalled([0.0], frozen_streak=50, min_push_ticks=25, eps=0.01) is False

    def test_frozen_while_pushing_long_enough_is_stall(self):
        # Frozen-while-pushing long enough AND the window confirms no motion -> stop.
        window = [-1.250, -1.251, -1.250, -1.249, -1.250]
        assert is_stalled(window, frozen_streak=30, min_push_ticks=25, eps=0.01, pushing=True) is True

    def test_frozen_but_short_streak_is_not_stall(self):
        # Encoder frozen but only just started -> NOT a stall: this is the breakaway
        # window, give the jaw time to move.
        window = [-1.250, -1.251, -1.250, -1.249, -1.250]
        assert is_stalled(window, frozen_streak=5, min_push_ticks=25, eps=0.01, pushing=True) is False

    def test_moving_window_is_not_stall(self):
        # The jaw is still travelling across the window -> not a stop yet.
        window = [0.0, 0.1, 0.2, 0.3, 0.4]
        assert is_stalled(window, frozen_streak=50, min_push_ticks=25, eps=0.01, pushing=True) is False

    def test_not_pushing_is_not_stall(self):
        # Lead not saturated (not commanding into anything) -> can't be a hard stop.
        window = [1.0, 1.0, 1.0, 1.0, 1.0]
        assert is_stalled(window, frozen_streak=50, min_push_ticks=25, eps=0.01, pushing=False) is False

    def test_zero_min_push_ticks_disables_requirement(self):
        # min_push_ticks<=0 degrades to "stall on a no-motion window alone" (still
        # requires pushing).
        window = [2.0, 2.0, 2.0]
        assert is_stalled(window, frozen_streak=0, min_push_ticks=0, eps=0.01, pushing=True) is True


class TestStallLocalGrace:
    """The grace must be LOCAL to each obstruction (reset on encoder motion), not
    global to the sweep. A jaw lagging the ramp the whole stroke must still get a
    fresh min_push_ticks countdown at each new sticky spot, or a mid-stroke freeze
    becomes a phantom truncated endpoint (the false-PASS the re-review found)."""

    def test_breakaway_before_min_push_is_not_stall(self):
        # Frozen but the streak hasn't persisted -> hold off (jaw should break free).
        frozen = [-1.305, -1.305, -1.305, -1.305, -1.305]
        assert is_stalled(frozen, frozen_streak=20, min_push_ticks=25, eps=0.01, pushing=True) is False

    def test_stop_very_close_to_start_still_detected(self):
        # A stop only ~0.05 rad from the sweep start (the hardware case that broke a
        # distance-based guard): frozen-while-pushing persists -> correctly a stall.
        near_start = [-1.254, -1.254, -1.253, -1.254, -1.254]
        assert is_stalled(near_start, frozen_streak=30, min_push_ticks=25, eps=0.01, pushing=True) is True

    def test_mid_stroke_fresh_freeze_gets_a_fresh_countdown(self):
        # A jaw that just resumed motion mid-stroke (so frozen_streak was reset to a
        # small value by the advance) must NOT be called stopped on a brief freeze,
        # even though it has been pushing since the sweep start. This is exactly the
        # ramp-lagging-jaw false-PASS: low frozen_streak -> not a stall.
        window = [2.50, 2.50, 2.50, 2.50, 2.50]
        assert is_stalled(window, frozen_streak=8, min_push_ticks=25, eps=0.01, pushing=True) is False


class TestClassifyEndpoints:
    def test_open_is_larger_closed_is_smaller(self):
        # sim-frame convention: positive = more open.
        assert classify_endpoints(5.1, 0.0, min_travel=1.0) == (5.1, 0.0)
        assert classify_endpoints(0.0, 5.1, min_travel=1.0) == (5.1, 0.0)

    def test_negative_frame_after_drift(self):
        # After a power-cycle the whole range can be negative; open is still the
        # larger (less negative) value.
        assert classify_endpoints(-1.25, -6.53, min_travel=1.0) == (-1.25, -6.53)

    def test_too_small_travel_is_none(self):
        # Jaws never moved a real stroke (stuck / dead / something held) -> fail.
        assert classify_endpoints(0.02, 0.0, min_travel=1.0) is None
        assert classify_endpoints(-1.25, -1.25, min_travel=1.0) is None

    def test_too_large_travel_is_none(self):
        # Implausibly long discovery (missed stop / slipping) -> fail.
        assert classify_endpoints(-1.0, -9.0, min_travel=4.5, max_travel=6.5) is None

    def test_in_band_passes(self):
        # A real ~5.3 rad stroke is inside [4.5, 6.5] -> accepted.
        assert classify_endpoints(-1.25, -6.53, min_travel=4.5, max_travel=6.5) == (-1.25, -6.53)

    def test_partial_jam_rejected_at_default_min_travel(self):
        # A ~1.5 rad partial stroke (something jammed) is BELOW the 3.5 rad default
        # floor (real YAM stroke ~5 rad) -> rejected, so a truncated-but-self-
        # consistent discovery can't false-PASS. (~1.77 rad of truncation at the old
        # 3.5 floor is the case the re-review flagged; the 4.5 default tightens it.)
        assert classify_endpoints(-1.25, -3.50, min_travel=4.5, max_travel=6.5) is None
        # A near-full ~5.3 rad stroke clears the band.
        assert classify_endpoints(-1.25, -6.53, min_travel=4.5, max_travel=6.5) == (-1.25, -6.53)


class TestTraversedOk:
    def test_full_sweep_passes(self):
        # Encoder covered the whole [0, 3.5] stroke -> pass.
        assert traversed_ok(0.0, 3.5, open_pos=3.5, closed_pos=0.0, frac_required=0.8) is True

    def test_partial_sweep_above_threshold_passes(self):
        # Covered 0.1..3.4 of a 3.5 stroke = 3.3 >= 0.8*3.5 -> pass (settle margin).
        assert traversed_ok(0.1, 3.4, open_pos=3.5, closed_pos=0.0, frac_required=0.8) is True

    def test_twitch_fails(self):
        # Barely moved (stuck/dead) -> fail.
        assert traversed_ok(0.0, 0.2, open_pos=3.5, closed_pos=0.0, frac_required=0.8) is False

    def test_zero_stroke_fails(self):
        assert traversed_ok(0.0, 0.0, open_pos=1.0, closed_pos=1.0, frac_required=0.8) is False

    def test_negative_frame_full_sweep_passes(self):
        assert traversed_ok(-4.75, -1.25, open_pos=-1.25, closed_pos=-4.75, frac_required=0.8) is True


class TestResultLine:
    def test_pass_line_format_is_parseable(self):
        r = SelfTestResult("PASS", open_pos=-1.2500, closed_pos=-4.7500, travel=3.5)
        line = r.result_line()
        assert line.startswith(RESULT_PREFIX)
        assert "status=PASS" in line
        # The launcher greps open=/closed= with a sed [-0-9.] capture; pin the shape.
        assert "open=-1.2500" in line
        assert "closed=-4.7500" in line

    def test_fail_line_includes_reason(self):
        r = SelfTestResult("FAIL", float("nan"), float("nan"), float("nan"), "stall during homing")
        line = r.result_line()
        assert "status=FAIL" in line
        assert "reason=" in line

    def test_pass_endpoints_roundtrip_through_sed_capture(self):
        # Emulate the launcher's sed extraction so a format change is caught here.
        import re
        r = SelfTestResult("PASS", open_pos=2.0, closed_pos=-1.5, travel=3.5)
        line = r.result_line()
        m_open = re.search(r"open=([-0-9.]+)", line)
        m_closed = re.search(r"closed=([-0-9.]+)", line)
        assert m_open and float(m_open.group(1)) == pytest.approx(2.0)
        assert m_closed and float(m_closed.group(1)) == pytest.approx(-1.5)
