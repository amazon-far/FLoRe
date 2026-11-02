"""Unit tests for the 2-key keyboard gripper teleop.

Covers the pure, hardware-free pieces of
``holosoma_hand_pipeline.keyboard_gripper``: the per-tick velocity clamp
(``_step_toward``), the key→target mapping (``_key_to_target``), and the
``_DryRunGripper`` stub contract. The curses loop + CAN bring-up aren't
exercised here (no TTY / no hardware in CI) — same split the rest of the
package uses (see test_gripper_writer.py).

The module imports cleanly without rclpy / gmp_i2rt: it only pulls in
``defaults`` and stdlib at import time (LinearGripperDriver is a lazy
in-function import), so no conftest stubbing is needed.
"""

from __future__ import annotations

import pytest

from holosoma_hand_pipeline.defaults import GRIPPER_CLOSED_POS, GRIPPER_OPEN_POS
from holosoma_hand_pipeline import keyboard_gripper as kg
from holosoma_hand_pipeline.keyboard_gripper import (
    _CLOSE_KEYS,
    _OPEN_KEYS,
    _DryRunGripper,
    _EncoderPublisher,
    _clamp_to_actual_lead,
    _gripper_motor_ok,
    _key_direction,
    _key_to_target,
    _momentary_target,
    _parse_args,
    _resolve_backend,
    _resolve_endpoints,
    _resolve_gains,
    _run_headless,
    _step_toward,
    main,
)


class _CountingEvent:
    """A threading.Event stand-in whose is_set() returns False for the first
    ``n`` polls then True forever — lets a control loop run exactly ``n`` ticks
    in a unit test without real time / threads."""

    def __init__(self, n: int):
        self._left = int(n)

    def is_set(self) -> bool:
        if self._left <= 0:
            return True
        self._left -= 1
        return False

    def set(self) -> None:
        self._left = 0


class TestStepToward:
    def test_small_delta_snaps_to_target(self):
        # Within one step → land exactly on target (no overshoot).
        assert _step_toward(0.0, 0.05, max_delta=0.10) == pytest.approx(0.05)

    def test_large_positive_delta_clamped(self):
        # Full open from closed is clamped to a single +max_delta step.
        out = _step_toward(GRIPPER_CLOSED_POS, GRIPPER_OPEN_POS, max_delta=0.10)
        assert out == pytest.approx(GRIPPER_CLOSED_POS + 0.10)

    def test_large_negative_delta_clamped(self):
        out = _step_toward(GRIPPER_OPEN_POS, GRIPPER_CLOSED_POS, max_delta=0.10)
        assert out == pytest.approx(GRIPPER_OPEN_POS - 0.10)

    def test_at_target_is_fixed_point(self):
        # Already parked → no motion (so the loop skips redundant writes).
        assert _step_toward(1.234, 1.234, max_delta=0.10) == pytest.approx(1.234)

    def test_converges_in_expected_tick_count(self):
        # Full travel at 0.10 rad/tick should reach target monotonically
        # and stop exactly there — pins the ramp behavior a tap relies on.
        current = GRIPPER_CLOSED_POS
        target = GRIPPER_OPEN_POS
        for _ in range(1000):
            nxt = _step_toward(current, target, max_delta=0.10)
            # Never overshoots past the target.
            assert min(current, target) <= nxt <= max(current, target)
            current = nxt
            if current == target:
                break
        assert current == pytest.approx(target)


class TestKeyToTarget:
    def test_open_keys_latch_open(self):
        for key in _OPEN_KEYS:
            assert _key_to_target(key, GRIPPER_OPEN_POS, GRIPPER_CLOSED_POS, 1.0) == GRIPPER_OPEN_POS

    def test_close_keys_latch_closed(self):
        for key in _CLOSE_KEYS:
            assert _key_to_target(key, GRIPPER_OPEN_POS, GRIPPER_CLOSED_POS, 1.0) == GRIPPER_CLOSED_POS

    def test_unknown_key_keeps_current_target(self):
        sentinel = 2.71828
        assert _key_to_target(ord("z"), GRIPPER_OPEN_POS, GRIPPER_CLOSED_POS, sentinel) == sentinel

    def test_o_and_p_are_the_two_control_keys(self):
        # The macropad emits 'o' and 'p'; pin that they are the open /
        # close keys so a refactor doesn't silently remap them.
        assert ord("o") in _OPEN_KEYS
        assert ord("p") in _CLOSE_KEYS


class TestKeyDirection:
    def test_open_keys_are_positive(self):
        for key in _OPEN_KEYS:
            assert _key_direction(key) == +1

    def test_close_keys_are_negative(self):
        for key in _CLOSE_KEYS:
            assert _key_direction(key) == -1

    def test_other_keys_are_zero(self):
        assert _key_direction(ord("z")) == 0
        assert _key_direction(ord(" ")) == 0


class TestMomentaryTarget:
    """Hold-to-move semantics: while held, the target leads current by one
    step toward the endpoint; on release the target pins to current."""

    OPEN, CLOSED, MD = 4.9, 0.0, 0.10

    def test_held_open_leads_toward_open(self):
        out = _momentary_target(+1, 1.0, self.OPEN, self.CLOSED, self.MD)
        assert out == pytest.approx(1.0 + self.MD)

    def test_held_close_leads_toward_closed(self):
        out = _momentary_target(-1, 1.0, self.OPEN, self.CLOSED, self.MD)
        assert out == pytest.approx(1.0 - self.MD)

    def test_release_pins_to_current(self):
        # held_direction 0 → freeze exactly where we are (no drift).
        assert _momentary_target(0, 2.345, self.OPEN, self.CLOSED, self.MD) == pytest.approx(2.345)

    def test_clamps_at_open_endpoint(self):
        # Holding open at the endpoint never commands past it.
        out = _momentary_target(+1, self.OPEN, self.OPEN, self.CLOSED, self.MD)
        assert out == pytest.approx(self.OPEN)

    def test_clamps_at_closed_endpoint(self):
        out = _momentary_target(-1, self.CLOSED, self.OPEN, self.CLOSED, self.MD)
        assert out == pytest.approx(self.CLOSED)

    def test_works_with_inverted_endpoint_order(self):
        # open < closed numerically (e.g. negative-rad open): +1 must still
        # move toward 'open' even though it's the smaller number.
        out_open = _momentary_target(+1, 0.0, -1.0, 1.0, 0.1)
        assert out_open == pytest.approx(-0.1)
        out_close = _momentary_target(-1, 0.0, -1.0, 1.0, 0.1)
        assert out_close == pytest.approx(0.1)


class TestDryRunGripper:
    def test_initial_position_held(self):
        g = _DryRunGripper(initial=GRIPPER_OPEN_POS)
        assert g.get_position() == GRIPPER_OPEN_POS

    def test_set_position_reflected(self):
        g = _DryRunGripper()
        g.set_position(1.5)
        assert g.get_position() == 1.5

    def test_close_is_safe(self):
        _DryRunGripper().close()  # must not raise


class TestBackendResolution:
    def test_auto_picks_i2rt_for_channel(self):
        # A bare --channel (CANable on can0) → the direct i2rt backend.
        args = _parse_args(["--channel", "can0"])
        assert _resolve_backend(args) == "i2rt"

    def test_auto_picks_gmp_i2rt_for_serial(self):
        # A PEAK serial (robot) → the gmp_i2rt MotorChainRobot path.
        args = _parse_args(["--serial", "peak_device_id_6"])
        assert _resolve_backend(args) == "gmp_i2rt"

    def test_explicit_backend_wins(self):
        args = _parse_args(["--backend", "gmp_i2rt", "--channel", "can0"])
        assert _resolve_backend(args) == "gmp_i2rt"


class TestGainResolution:
    """Per-backend default gains: the direct-i2rt MIT PD path needs a
    firmer kp than the gmp_i2rt force-mapped path (measured — kp=1 only
    reaches ~60% of travel on the direct path; kp=3 tracks the full
    stroke)."""

    def test_i2rt_default_gains(self):
        args = _parse_args(["--channel", "can0"])  # kp/kd unset
        assert _resolve_gains(args, "i2rt") == (3.0, 0.3)

    def test_gmp_i2rt_default_gains(self):
        args = _parse_args(["--serial", "peak_device_id_6"])
        assert _resolve_gains(args, "gmp_i2rt") == (1.0, 0.2)

    def test_explicit_kp_kd_override_both_backends(self):
        args = _parse_args(["--channel", "can0", "--kp", "5.0", "--kd", "0.4"])
        assert _resolve_gains(args, "i2rt") == (5.0, 0.4)
        assert _resolve_gains(args, "gmp_i2rt") == (5.0, 0.4)


class TestEndpointResolution:
    """Per-backend default travel endpoints. i2rt is MEASURED on the laptop
    YAM linear_4310 (stops at ~-0.08 / ~+5.02 rad → safe 4.90 / 0.0);
    gmp_i2rt keeps the holosoma sim-frame convention (3.5 / 0.0)."""

    def test_i2rt_default_endpoints_reach_near_full_stroke(self):
        args = _parse_args(["--channel", "can0"])  # open/closed unset
        open_pos, closed_pos = _resolve_endpoints(args, "i2rt")
        assert (open_pos, closed_pos) == (4.90, 0.0)
        # Must open meaningfully further than the old 3.5 default.
        assert open_pos > GRIPPER_OPEN_POS

    def test_gmp_i2rt_default_endpoints_are_sim_convention(self):
        args = _parse_args(["--serial", "peak_device_id_6"])
        assert _resolve_endpoints(args, "gmp_i2rt") == (GRIPPER_OPEN_POS, GRIPPER_CLOSED_POS)

    def test_explicit_open_closed_override(self):
        args = _parse_args(["--channel", "can0", "--open", "5.0", "--closed", "0.1"])
        assert _resolve_endpoints(args, "i2rt") == (5.0, 0.1)


class TestEncoderPublisher:
    """The optional read-only encoder WebSocket publisher (default OFF). These
    pin the cross-process wire contract collection_viz consumes + the localhost
    guard, WITHOUT a TTY or CAN (so they run in CI like the rest of the suite)."""

    def test_encoder_ws_off_by_default(self):
        # No --encoder-ws-port -> the feature is disabled; host defaults local.
        args = _parse_args(["--channel", "can0"])
        assert args.encoder_ws_port is None
        assert args.encoder_ws_host == "127.0.0.1"

    def test_encoder_ws_port_parsed(self):
        args = _parse_args(["--channel", "can0", "--encoder-ws-port", "8774"])
        assert args.encoder_ws_port == 8774

    def test_snapshot_wire_contract(self):
        # The exact field set + values collection_viz._on_gripper_msg reads. A
        # silent rename here would make the viz mirror a frozen gripper, so pin it.
        import json

        pub = _EncoderPublisher("127.0.0.1", 8774, open_pos=4.90, closed_pos=0.0)
        pub.set_position(1.0, 0.9)
        frame = json.loads(pub._snapshot())
        assert set(frame.keys()) == {"seq", "t", "open", "closed", "cmd", "actual", "kbd_online", "motor_ok"}
        assert frame["cmd"] == 1.0
        assert frame["actual"] == 0.9
        assert frame["open"] == 4.90
        assert frame["closed"] == 0.0
        # No keyboard heartbeat seen yet -> the 2-key bridge reports offline.
        assert frame["kbd_online"] is False
        # Motor is healthy until the control thread is observed dead.
        assert frame["motor_ok"] is True

    def test_snapshot_seq_increments(self):
        import json

        pub = _EncoderPublisher("127.0.0.1", 8774, 4.90, 0.0)
        pub.set_position(0.1, 0.1)
        s1 = json.loads(pub._snapshot())["seq"]
        pub.set_position(0.2, 0.2)
        s2 = json.loads(pub._snapshot())["seq"]
        assert s2 == s1 + 1

    def test_non_localhost_encoder_host_rejected(self, monkeypatch):
        # main() must refuse a routable encoder WS host (security guard) without
        # ever starting a server / touching a TTY. Guard curses just in case.
        monkeypatch.setattr(kg.curses, "wrapper", lambda *a, **k: None)
        rc = main(["--dry-run", "--encoder-ws-port", "8774", "--encoder-ws-host", "0.0.0.0"])
        assert rc == 2


class TestKeyboardLiveness:
    """The 2-key bridge connects to the gripper WS (never to the viz), so its
    liveness is relayed through the gripper snapshot's ``kbd_online`` field. A
    frame tagged src="kbd" (heartbeat or real intent) marks it alive; the browser's
    own untagged control frames must NOT masquerade as the keyboard."""

    def _pub(self):
        return _EncoderPublisher("127.0.0.1", 8774, open_pos=4.90, closed_pos=0.0)

    def test_kbd_offline_before_any_frame(self):
        assert self._pub()._kbd_online() is False

    def test_kbd_heartbeat_marks_online(self):
        import json

        pub = self._pub()
        pub._apply_inbound('{"cmd": "ping", "src": "kbd"}')
        assert pub._kbd_online() is True
        assert json.loads(pub._snapshot())["kbd_online"] is True

    def test_kbd_intent_also_marks_online_and_still_moves_motor(self):
        # A real o/p intent from the keyboard both marks liveness AND drives the
        # motor (the ping branch only suppresses the bare heartbeat).
        pub = self._pub()
        pub._apply_inbound('{"cmd": "open", "src": "kbd"}')
        assert pub._kbd_online() is True
        assert pub.get_target_frac() == pytest.approx(1.0)

    def test_browser_frame_does_not_mark_keyboard(self):
        # The browser's own control frames carry no src -> must not be mistaken
        # for the keyboard bridge being online.
        pub = self._pub()
        pub._apply_inbound('{"cmd": "open"}')
        assert pub._kbd_online() is False

    def test_kbd_goes_offline_after_timeout(self):
        pub = self._pub()
        pub._apply_inbound('{"cmd": "ping", "src": "kbd"}')
        assert pub._kbd_online() is True
        # Backdate the heartbeat past the timeout window -> offline again.
        pub._kbd_last_t -= (pub._kbd_timeout + 1.0)
        assert pub._kbd_online() is False


class TestWebControl:
    """The headless, browser-driven control path: inbound open/close/stop +
    target_frac latching, and the --web-control CLI contract. No TTY / CAN, so
    these run in CI alongside the rest of the suite."""

    def _pub(self):
        return _EncoderPublisher("127.0.0.1", 8774, open_pos=4.90, closed_pos=0.0)

    def test_target_frac_defaults_none(self):
        # No command yet -> freeze (the headless loop holds position).
        assert self._pub().get_target_frac() is None

    def test_cmd_open_close_stop(self):
        pub = self._pub()
        pub._apply_inbound('{"cmd": "open"}')
        assert pub.get_target_frac() == pytest.approx(1.0)
        pub._apply_inbound('{"cmd": "close"}')
        assert pub.get_target_frac() == pytest.approx(0.0)
        pub._apply_inbound('{"cmd": "stop"}')
        assert pub.get_target_frac() is None

    def test_target_frac_clamped(self):
        pub = self._pub()
        pub._apply_inbound('{"target_frac": 0.42}')
        assert pub.get_target_frac() == pytest.approx(0.42)
        pub._apply_inbound('{"target_frac": 9.0}')  # clamp high
        assert pub.get_target_frac() == pytest.approx(1.0)
        pub._apply_inbound('{"target_frac": -3.0}')  # clamp low
        assert pub.get_target_frac() == pytest.approx(0.0)

    def test_malformed_inbound_is_ignored(self):
        # Garbage / wrong-type / unknown cmd must never crash or move the motor.
        pub = self._pub()
        pub.set_target_frac(0.5)
        for bad in ("not json", "[1,2,3]", '{"cmd": "spin"}', '{"target_frac": "x"}', "42"):
            pub._apply_inbound(bad)
        assert pub.get_target_frac() == pytest.approx(0.5)  # unchanged

    def test_nan_inf_inbound_frame_rejected(self):
        # SAFETY: json.loads accepts NaN/Infinity by default, and NaN defeats the
        # clamp (every NaN comparison is False) — left unchecked it would creep
        # the motor shut a step/tick forever. parse_constant rejects the whole
        # frame, so a non-finite literal NEVER reaches the motor: the prior valid
        # target is left intact (same contract as any other malformed frame), and
        # crucially the latched target is always finite (never nan/inf).
        import math

        for bad in ('{"target_frac": NaN}', '{"target_frac": Infinity}', '{"target_frac": -Infinity}'):
            pub = self._pub()
            pub.set_target_frac(0.5)  # a prior valid target
            pub._apply_inbound(bad)
            tf = pub.get_target_frac()
            assert tf == pytest.approx(0.5)  # unchanged
            assert tf is None or math.isfinite(tf)  # never non-finite

    def test_set_target_frac_rejects_non_finite_directly(self):
        pub = self._pub()
        pub.set_target_frac(0.5)
        pub.set_target_frac(float("nan"))
        assert pub.get_target_frac() is None
        pub.set_target_frac(0.3)
        pub.set_target_frac(float("inf"))
        assert pub.get_target_frac() is None


class TestRunHeadless:
    """The headless web-control runtime: frac -> rad -> per-tick-clamped motor
    write. Driven hardware-free against _DryRunGripper + _EncoderPublisher, with
    a stop_evt that trips after N ticks (so the loop exits deterministically).
    Pins the load-bearing mapping the curses path's TestMomentaryTarget covers
    for its own target computation."""

    def _args(self, max_delta=0.10, initial_frac=-1.0):
        # initial_frac=-1 (freeze, no startup seed) by DEFAULT here so the tests below
        # isolate the frac->rad mapping. The shipped default is 1.0 = OPEN and is pinned
        # separately by test_startup_seeds_the_declared_open_state; seeding inside these
        # would clobber the target they pre-set on the publisher.
        a = _parse_args(
            [
                "--channel",
                "can0",
                "--encoder-ws-port",
                "8774",
                "--web-control",
                "--max-delta",
                str(max_delta),
                "--rate-hz",
                "1000",
                "--initial-frac",
                str(initial_frac),
            ]
        )
        a.open, a.closed = _resolve_endpoints(a, "i2rt")  # (4.90, 0.0)
        return a

    def _run_n(self, gripper, args, pub, n):
        # A stop event that returns True after n is_set() polls, so _run_headless
        # executes exactly n control ticks then exits.
        evt = _CountingEvent(n)
        _run_headless(gripper, args, pub, evt)

    def test_frac_one_ramps_toward_open_one_step_per_tick(self):
        g = _DryRunGripper(initial=0.0)
        pub = _EncoderPublisher("127.0.0.1", 8774, 4.90, 0.0)
        pub.set_target_frac(1.0)
        self._run_n(g, self._args(max_delta=0.10), pub, n=3)
        # 3 ticks * 0.10 from 0.0 toward open (4.90) -> 0.30, never overshooting.
        assert g.get_position() == pytest.approx(0.30)

    def test_frac_half_converges_to_midpoint(self):
        g = _DryRunGripper(initial=0.0)
        pub = _EncoderPublisher("127.0.0.1", 8774, 4.90, 0.0)
        pub.set_target_frac(0.5)  # target = 0.0 + 0.5*4.90 = 2.45 rad
        self._run_n(g, self._args(max_delta=0.10), pub, n=200)
        assert g.get_position() == pytest.approx(2.45)

    def test_none_target_freezes_at_current(self):
        g = _DryRunGripper(initial=1.234)
        pub = _EncoderPublisher("127.0.0.1", 8774, 4.90, 0.0)
        pub.set_target_frac(None)  # freeze
        self._run_n(g, self._args(max_delta=0.10), pub, n=50)
        assert g.get_position() == pytest.approx(1.234)  # never moved

    def test_startup_seeds_the_declared_open_state(self):
        """The startup jaw state must be DECLARED, not inherited.

        Without this the loop sat at target_frac=None (freeze) until a page command arrived,
        so "the initial state" meant whatever the jaw happened to be: the self-test's last
        goal, the previous session's last position, or an unknown one under SELFTEST=0. A jaw
        that comes up shut reads as a jaw HOLDING something, so both hands start OPEN.
        """
        assert _parse_args(["--channel", "can0"]).initial_frac == 1.0  # the shipped default
        g = _DryRunGripper(initial=0.0)
        pub = _EncoderPublisher("127.0.0.1", 8774, 4.90, 0.0)
        assert pub.get_target_frac() is None  # nothing commanded yet
        self._run_n(g, self._args(max_delta=0.10, initial_frac=1.0), pub, n=200)
        assert pub.get_target_frac() == pytest.approx(1.0)
        assert g.get_position() == pytest.approx(4.90)  # ramped all the way OPEN

    def test_negative_initial_frac_keeps_the_old_freeze(self):
        """The opt-out has to still work: resuming a grasp on an object already held is the
        one case where inheriting the jaw position is right."""
        g = _DryRunGripper(initial=1.234)
        pub = _EncoderPublisher("127.0.0.1", 8774, 4.90, 0.0)
        self._run_n(g, self._args(max_delta=0.10, initial_frac=-1.0), pub, n=50)
        assert pub.get_target_frac() is None
        assert g.get_position() == pytest.approx(1.234)

    def test_web_control_requires_encoder_port(self, monkeypatch):
        # --web-control without --encoder-ws-port is a usage error (rc=2), and it
        # must bail BEFORE any gripper bring-up (the control channel IS the
        # encoder socket). Spy _make_gripper to pin "no bring-up on usage error".
        called = []
        monkeypatch.setattr(kg, "_make_gripper", lambda *a, **k: called.append(1))
        rc = main(["--dry-run", "--web-control"])
        assert rc == 2
        assert called == []  # never touched the (dry-run) gripper / CAN path

    def test_localhost_encoder_host_accepted(self, monkeypatch):
        # The localhost-only allowlist accepts BOTH spellings. Pin that
        # "localhost" (not just 127.0.0.1) is allowed, so a future tightening of
        # the allowlist can't silently break the documented contract. Stub the
        # publisher + headless loop so no real server / motion happens.
        monkeypatch.setattr(kg._EncoderPublisher, "start", lambda self: None)
        monkeypatch.setattr(kg, "_run_headless", lambda *a, **k: None)
        rc = main(["--dry-run", "--encoder-ws-port", "8774", "--encoder-ws-host", "localhost", "--web-control"])
        assert rc == 0  # accepted, not rejected with rc=2

    def test_web_control_flag_parsed(self):
        args = _parse_args(["--channel", "can0", "--encoder-ws-port", "8774", "--web-control"])
        assert args.web_control is True
        # Off by default so the curses path is unchanged when not requested.
        assert _parse_args(["--channel", "can0"]).web_control is False


class TestOverheatGuard:
    """`_clamp_to_actual_lead` is the overheat guard: it caps how far the command
    may lead the measured encoder, bounding stall torque (torque ≈ kp·(cmd−actual))
    so a hard-stop stall can never run current away into over-temperature."""

    def test_no_clamp_when_within_lead(self):
        # cmd within max_lead of actual -> unchanged (normal tracking untouched).
        assert _clamp_to_actual_lead(1.05, 1.00, max_lead=0.6) == pytest.approx(1.05)

    def test_clamps_positive_lead(self):
        # cmd far ABOVE a frozen actual (stalled opening) -> clamped to actual+lead.
        assert _clamp_to_actual_lead(4.90, -1.25, max_lead=0.6) == pytest.approx(-1.25 + 0.6)

    def test_clamps_negative_lead(self):
        # cmd far BELOW actual (stalled closing) -> clamped to actual-lead.
        assert _clamp_to_actual_lead(-4.0, 0.0, max_lead=0.6) == pytest.approx(-0.6)

    def test_zero_or_negative_max_lead_disables(self):
        # <=0 disables the guard (command passes through unchanged).
        assert _clamp_to_actual_lead(100.0, 0.0, max_lead=0.0) == pytest.approx(100.0)
        assert _clamp_to_actual_lead(100.0, 0.0, max_lead=-1.0) == pytest.approx(100.0)

    def test_default_max_lead_is_positive(self):
        # The shipped default must actually enable the guard (regression: a 0 default
        # would silently leave the motor unprotected).
        assert _parse_args(["--channel", "can0"]).max_lead > 0.0

    def test_guard_bounds_stall_in_headless_loop(self):
        # End-to-end: command full-open against a gripper whose encoder is FROZEN
        # at the drifted rest (-1.25). Without the guard `current` would ramp to the
        # 4.90 endpoint; with it, the written command never leads the frozen actual
        # by more than max_lead.
        class _FrozenGripper:
            def __init__(self, frozen):
                self._frozen = frozen
                self.last_cmd = None
            def get_position(self):
                return self._frozen
            def set_position(self, p):
                self.last_cmd = p
            def close(self):
                pass

        g = _FrozenGripper(frozen=-1.25)
        pub = _EncoderPublisher("127.0.0.1", 8774, 4.90, 0.0)
        pub.set_target_frac(1.0)  # full open
        a = _parse_args(["--channel", "can0", "--encoder-ws-port", "8774", "--web-control",
                         "--max-delta", "0.10", "--rate-hz", "1000", "--max-lead", "0.6"])
        a.open, a.closed = _resolve_endpoints(a, "i2rt")
        _run_headless(g, a, pub, _CountingEvent(500))
        # The last command written can never exceed frozen + max_lead.
        assert g.last_cmd is not None
        assert g.last_cmd <= -1.25 + 0.6 + 1e-9


class _FaultGripper:
    """A gripper stub that reports motor_ok()=True for the first ``alive_ticks``
    motor_ok polls, then False forever (simulating the i2rt control thread dying)."""

    def __init__(self, alive_ticks: int, pos: float = -1.25):
        self._alive = int(alive_ticks)
        self._pos = float(pos)
        self.closed = False
        self.set_calls = 0

    def get_position(self) -> float:
        return self._pos

    def set_position(self, p: float) -> None:
        self.set_calls += 1
        self._pos = float(p)  # a live motor would track; the dead one shouldn't be written

    def motor_ok(self) -> bool:
        if self._alive <= 0:
            return False
        self._alive -= 1
        return True

    def close(self) -> None:
        self.closed = True


class TestMotorOkProbe:
    def test_missing_method_is_ok(self):
        # A stub without motor_ok() (dry-run / gmp_i2rt) is never falsely faulted.
        assert _gripper_motor_ok(_DryRunGripper()) is True

    def test_true_and_false_passthrough(self):
        assert _gripper_motor_ok(_FaultGripper(alive_ticks=5)) is True
        assert _gripper_motor_ok(_FaultGripper(alive_ticks=0)) is False

    def test_raising_probe_is_fault(self):
        class _Raises:
            def motor_ok(self):
                raise RuntimeError("bus dead")
        assert _gripper_motor_ok(_Raises()) is False


class TestDeadMotorDetection:
    """Once the underlying control thread dies, the headless loop must STOP
    publishing a phantom moving command (the bug that green-lit a dead motor),
    de-energize, and latch motor_ok=False on the snapshot."""

    def test_fault_de_energizes_and_flags_motor_not_ok(self):
        import json

        g = _FaultGripper(alive_ticks=3)  # dies on the 4th tick
        pub = _EncoderPublisher("127.0.0.1", 8774, 4.90, 0.0)
        pub.set_target_frac(1.0)  # commanding open the whole time
        a = _parse_args(["--channel", "can0", "--encoder-ws-port", "8774", "--web-control",
                         "--max-delta", "0.10", "--rate-hz", "1000", "--initial-frac", "-1"])
        a.open, a.closed = _resolve_endpoints(a, "i2rt")
        _run_headless(g, a, pub, _CountingEvent(50))
        # On fault: de-energized once, and the snapshot reports the motor faulted.
        assert g.closed is True
        assert json.loads(pub._snapshot())["motor_ok"] is False

    def test_no_writes_after_fault(self):
        # After the fault latches, the loop must not keep writing set_position (a
        # phantom ramp over a dead motor). Count writes after death stays bounded.
        g = _FaultGripper(alive_ticks=2)
        pub = _EncoderPublisher("127.0.0.1", 8774, 4.90, 0.0)
        pub.set_target_frac(1.0)
        a = _parse_args(["--channel", "can0", "--encoder-ws-port", "8774", "--web-control",
                         "--max-delta", "0.10", "--rate-hz", "1000", "--initial-frac", "-1"])
        a.open, a.closed = _resolve_endpoints(a, "i2rt")
        _run_headless(g, a, pub, _CountingEvent(100))
        # Only the pre-fault ticks could have written; after fault it freezes. With
        # ~2 alive ticks, far fewer than the 100 loop iterations.
        assert g.set_calls <= 5

    def test_healthy_motor_runs_normally(self):
        # A never-faulting stub must move exactly like before (no regression).
        g = _DryRunGripper(initial=0.0)
        pub = _EncoderPublisher("127.0.0.1", 8774, 4.90, 0.0)
        pub.set_target_frac(0.5)  # midpoint = 2.45
        a = _parse_args(["--channel", "can0", "--encoder-ws-port", "8774", "--web-control",
                         "--max-delta", "0.10", "--rate-hz", "1000", "--max-lead", "0",
                         "--initial-frac", "-1"])
        a.open, a.closed = _resolve_endpoints(a, "i2rt")
        _run_headless(g, a, pub, _CountingEvent(200))
        assert g.get_position() == pytest.approx(2.45)

    def test_snapshot_includes_motor_ok_true_by_default(self):
        import json
        pub = _EncoderPublisher("127.0.0.1", 8774, 4.90, 0.0)
        assert json.loads(pub._snapshot())["motor_ok"] is True
