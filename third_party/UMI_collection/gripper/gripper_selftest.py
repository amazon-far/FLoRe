#!/usr/bin/env python3
"""Startup auto-home + open/close self-test for the DM4310 YAM linear_4310 gripper.

WHY THIS EXISTS
---------------
The DM4310 uses an ABSOLUTE encoder that RE-REFERENCES on every power loss, so the
open/closed rad endpoints DRIFT after each motor power-cycle. If the teleop loop is
launched with stale endpoints (the launcher's defaults, or a value from a previous
power-cycle), commanding "open" drives the jaws PAST the real mechanical stop; with
the teleop kp the motor then holds large current into the stall → OVER-TEMPERATURE
→ the i2rt control thread dies, and the gripper looks "connected" (green) while
being completely dead. That is exactly the failure this test prevents on EVERY
launch.

WHAT IT DOES (auto-home + verify)
---------------------------------
1. AUTO-HOME — gently drive the jaws to BOTH hard stops to discover the TRUE
   current endpoints (so the drift is corrected automatically, no manual
   re-measure). Homing is SAFE BY CONSTRUCTION:
     * a LOW homing kp (``--kp-home``, default 1.0 vs teleop 3.0), and
     * the same OVERHEAT-GUARD lead clamp the teleop loop uses, so the command can
       never lead the encoder by more than ``--max-lead`` rad. The worst-case
       holding torque against a stop is therefore bounded to ``kp_home*max_lead``
       (tiny), and
     * stall detection that fires within ~``stall_ticks`` ticks of hitting a stop
       and IMMEDIATELY stops commanding that direction, plus a per-direction
       timeout and a rotor/MOS temperature guard that ABORT the whole test.
2. VERIFY — run ONE full open→close→open cycle against the discovered endpoints and
   confirm the encoder actually TRAVERSED the range (not a stuck/dead motor).
3. REPORT — print a machine-readable ``RESULT`` line the launcher parses to pick up
   the discovered endpoints, and exit 0 (PASS) / non-zero (FAIL). The motor is
   powered DOWN on every exit path.

The pure decision logic (stall detection, endpoint classification, traversal
check) is split into hardware-free functions so it is unit-tested without a motor
(see gripper/test_gripper_selftest.py). The hardware runner talks to i2rt's
``DMChainCanInterface`` directly — the SAME path keyboard_gripper's i2rt backend
uses, so the discovered sim-frame endpoints are directly usable as
``GRIPPER_OPEN``/``GRIPPER_CLOSED``.

This test OWNS the CAN bus while it runs and releases it (powers the motor down,
closes the chain) BEFORE the gripper bridge starts — they never run concurrently,
so the single-CAN-writer invariant is preserved.

USAGE (can0 must be UP; run while NO other process owns the gripper, nothing held):
    PYTHONPATH=third_party/i2rt:gripper \
      .venv/bin/python gripper/gripper_selftest.py --channel can0
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass


# Sentinel the launcher greps for. Kept stable so a wording change to the human
# log never breaks the launcher's endpoint pickup.
RESULT_PREFIX = "[selftest] RESULT"


@dataclass
class SelfTestResult:
    status: str            # "PASS" | "FAIL"
    open_pos: float        # discovered OPEN endpoint (sim-frame rad), NaN on fail
    closed_pos: float      # discovered CLOSED endpoint (sim-frame rad), NaN on fail
    travel: float          # |open - closed| (rad)
    reason: str = ""       # human explanation (esp. on FAIL)

    def result_line(self) -> str:
        return (f"{RESULT_PREFIX} status={self.status} open={self.open_pos:.4f} "
                f"closed={self.closed_pos:.4f} travel={self.travel:.4f}"
                + (f" reason={self.reason!r}" if self.reason else ""))


def is_stalled(window: list[float], frozen_streak: int, min_push_ticks: int, eps: float,
               pushing: bool = True) -> bool:
    """Decide whether the jaws have hit a hard stop.

    ``window`` is the most recent encoder positions (rad); ``frozen_streak`` is the
    number of CONSECUTIVE recent ticks the encoder has NOT moved (advanced less than
    one ``eps`` between samples) WHILE commanding hard; ``min_push_ticks`` is how
    long that frozen-while-pushing state must persist before a stall may be
    declared; ``eps`` is the no-motion threshold; ``pushing`` is whether the command
    lead is currently saturated.

    A stall is: the encoder has stayed frozen WHILE we push for at least
    ``min_push_ticks`` ticks AND the window confirms no motion.

    Why "frozen-since" and not "lead-saturated-since" or distance-from-start:
      * Instantaneous "lead saturated + window full" fires the tick full torque
        first arrives — misreading a jaw whose breakaway merely LAGS the ramp as a
        hard stop (phantom start endpoint / false-FAIL reversal sweep).
      * A distance-from-start arming guard is wrong too: a real stop can be closer
        to the sweep start than any fixed arming distance, which then never arms and
        times out — observed on hardware.
      * Counting ticks-since-the-lead-first-saturated is ALSO wrong: for a jaw that
        lags the ramp the WHOLE sweep the lead pins from the start, the streak grows
        monotonically, and by mid-stroke a fresh sticky spot gets ZERO grace — a
        ~min_push_ticks freeze is misread as a stop → phantom TRUNCATED endpoint
        (false-PASS). So the grace must be LOCAL to each obstruction: reset the
        counter whenever the encoder ADVANCES, so every fresh freeze starts a new
        ``min_push_ticks`` countdown no matter where in the stroke it happens. A
        free-but-sticky jaw breaks away within that window (counter resets on the
        advance); only a jaw against a real stop stays frozen the whole countdown.

    Needs a full ``window`` to decide (returns False before then) so a single noisy
    sample can't trip it. ``min_push_ticks <= 0`` disables the persistence
    requirement (legacy / test behavior: stall on a no-motion window alone)."""
    if len(window) < 2:
        return False
    if not pushing:
        return False  # not commanding into anything — can't be a hard stop
    if frozen_streak < min_push_ticks:
        return False  # not frozen-while-pushing long enough to call it stopped
    spread = max(window) - min(window)
    return spread < eps


def classify_endpoints(a: float, b: float, min_travel: float,
                       max_travel: float = float("inf")) -> tuple[float, float] | None:
    """Map two discovered stop positions to (open, closed) or None if the travel is
    implausible.

    Convention (matches defaults.py + measure_gripper_endpoints.py + the i2rt
    backend's sim-frame): positive sim rad = more OPEN, so OPEN is the larger value
    and CLOSED the smaller. The discovered stroke must fall in the TWO-SIDED band
    ``[min_travel, max_travel]`` or it's rejected (→ None, FAIL):
      * below ``min_travel`` → the motor never moved a real stroke (stuck, dead, or
        something held in the jaws), OR a mid-stroke false-stall truncated it;
      * above ``max_travel`` → an implausibly long discovery (e.g. a missed stop, a
        slipping/unbounded mechanism) that shouldn't be trusted as an endpoint.
    The band is the only guard against a truncated-but-self-consistent discovery
    (``traversed_ok`` measures against the DISCOVERED range, so it can't catch one)."""
    travel = abs(a - b)
    if travel < min_travel or travel > max_travel:
        return None
    return (max(a, b), min(a, b))


def traversed_ok(min_seen: float, max_seen: float, open_pos: float, closed_pos: float,
                 frac_required: float = 0.8) -> bool:
    """During the open→close→open verify, did the encoder cover enough of the range?

    ``min_seen``/``max_seen`` are the encoder extremes observed during the cycle.
    PASS if the observed span is at least ``frac_required`` of the discovered
    [closed, open] stroke — i.e. the jaws genuinely opened AND closed, not just
    twitched. ``frac_required`` < 1 tolerates the lead-clamp / settle margin so a
    healthy gripper isn't failed for stopping a few mrad short of each stop."""
    full = abs(open_pos - closed_pos)
    if full <= 0:
        return False
    seen = abs(max_seen - min_seen)
    return seen >= frac_required * full


def _safe_temp(state, attr: str) -> float:
    """Read a rotor/MOS temperature, mapping ONLY the missing-telemetry sentinel to
    -1 — a genuine 0 °C (or 0x00 glitch) reads as 0.0, not -1. (The naive
    ``getattr(...) or -1`` folds 0.0 into -1 because ``0.0 or -1 == -1``, which would
    silently disarm the over-temp check for that sample.) The MotorInfo dataclass
    default is -1, so None/-1 mean 'no reading'; any real byte (incl. 0) is kept."""
    v = getattr(state, attr, -1)
    if v is None:
        return -1.0
    try:
        return float(v)
    except (TypeError, ValueError):
        return -1.0


def _drive_until_stall(chain, motor_id, motor_type, np, *, start_pos, direction, step, kp, kd,
                       max_lead, rate_hz, stall_eps, stall_ticks, min_push_ticks, timeout_s,
                       temp_limit_c, label):
    """Gently drive in ``direction`` (+1/-1 in sim rad) until a hard stop, returning
    (endpoint_rad, aborted_reason | None). Bounded torque + stall + timeout + temp."""
    dt = 1.0 / max(rate_hz, 1.0)
    command = float(start_pos)
    target = command + direction * 100.0  # far past any real stop; the clamp bounds it
    window: list[float] = []
    frozen_streak = 0  # consecutive ticks the encoder has NOT advanced while pushing
    prev_actual = float(start_pos)
    t0 = time.monotonic()
    last_print = 0.0
    while True:
        # read encoder + temps
        states = chain.read_states()
        actual = float(states[0].pos)
        temp_rotor = _safe_temp(states[0], "temp_rotor")
        temp_mos = _safe_temp(states[0], "temp_mos")
        if temp_limit_c > 0 and (temp_rotor >= temp_limit_c or temp_mos >= temp_limit_c):
            return actual, f"over-temp during homing (rotor={temp_rotor:.0f}C mos={temp_mos:.0f}C)"

        # one ramp step toward the far target, then clamp the lead (overheat guard)
        delta = target - command
        s = abs(step)
        command = command + (s if delta > 0 else -s) if abs(delta) > s else target
        lo, hi = actual - max_lead, actual + max_lead
        command = float(min(max(command, lo), hi))
        chain.set_commands(torques=np.zeros(1), pos=np.array([command]), vel=np.zeros(1),
                           kp=np.array([kp]), kd=np.array([kd]))

        window.append(actual)
        if len(window) > stall_ticks:
            window.pop(0)
        lead = command - actual
        pushing = abs(lead) >= 0.9 * max_lead if max_lead > 0 else True
        # FROZEN streak = ticks the encoder has not advanced WHILE pushing. Reset it
        # the instant the encoder advances (> stall_eps since the last sample) — so
        # the min_push_ticks grace is LOCAL to each obstruction: a fresh sticky spot
        # anywhere in the stroke gets a full new countdown, and a free-but-sticky jaw
        # breaks away within it (the advance resets the streak). Only a jaw against a
        # real stop stays frozen the whole countdown. Counting since the lead first
        # saturated (the previous design) gave a ramp-lagging jaw zero mid-stroke
        # grace → a mid-stroke freeze became a phantom truncated endpoint (false-PASS).
        advanced = abs(actual - prev_actual) > stall_eps
        prev_actual = actual
        if pushing and not advanced:
            frozen_streak += 1
        else:
            frozen_streak = 0
        if len(window) >= stall_ticks and is_stalled(window, frozen_streak, min_push_ticks, stall_eps, pushing):
            return actual, None  # hit the stop

        now = time.monotonic()
        if now - last_print > 0.5:
            last_print = now
            print(f"[selftest] homing {label}: actual={actual:+.3f} cmd={command:+.3f} "
                  f"lead={lead:+.3f} temp={temp_rotor:.0f}/{temp_mos:.0f}C", flush=True)
        if now - t0 > timeout_s:
            # Never reached a clean stall within the budget. If it never moved at
            # all the motor is likely dead; report the last position + reason.
            return actual, f"homing {label} timed out after {timeout_s:.1f}s (no clean stall)"
        time.sleep(dt)


def run_selftest(args) -> SelfTestResult:
    """Hardware runner: auto-home both stops, verify a full open/close cycle, and
    return a PASS/FAIL result. Powers the motor down on every exit path."""
    import math

    import numpy as np
    from i2rt.motor_drivers.dm_driver import DMChainCanInterface, MotorCmd, ReceiveMode

    nan = float("nan")
    print(f"[selftest] opening {args.motor_type} id={args.motor_id} on {args.channel} "
          f"(motor_direction={args.motor_direction}, kp_home={args.kp_home}, max_lead={args.max_lead})",
          flush=True)
    chain = DMChainCanInterface(
        motor_list=[[args.motor_id, args.motor_type]],
        motor_offset=[0.0],
        motor_direction=[args.motor_direction],
        channel=args.channel,
        receive_mode=ReceiveMode.p16,
        motor_chain_name="gripper_selftest",
        start_thread=False,
    )
    mi = chain.motor_interface
    # off -> clean_error -> on clears a timeout-latched 'disabled' the same way the
    # teleop bring-up does; ride past the post-enable transient before commanding.
    try:
        mi.motor_off(args.motor_id)
    except Exception:  # noqa: BLE001
        pass
    time.sleep(0.05)
    try:
        mi.clean_error(args.motor_id)
    except Exception:  # noqa: BLE001
        pass
    time.sleep(0.05)
    try:
        info = mi.motor_on(args.motor_id, args.motor_type)
    except Exception as exc:  # noqa: BLE001
        print(f"[selftest] motor_on warning: {exc!r} (continuing)", flush=True)
        info = None
    # WARM-UP (mirrors _I2rtGripper._enable_and_warmup): the DM4310 reports the
    # 'disabled' (0x0) state for the first ~2 ticks after re-enable (the CAN
    # command-timeout register latches it when idle). Starting the control thread
    # immediately loses that transient race and the thread dies on the first error
    # — exactly a spurious "control thread died during bring-up". Send a handful of
    # synchronous zero-/low-gain hold commands FIRST so the motor is solidly enabled
    # (err==0x1) before the background loop starts. Seed `start` from the warm-up
    # read so the thread spins up holding the current position (no snap).
    start = 0.0
    if info is not None:
        raw_pos = info.position
        warm_ok = 0
        for _ in range(10):
            try:
                fi = mi.set_control(motor_id=args.motor_id, motor_type=args.motor_type,
                                    pos=raw_pos, vel=0.0, kp=0.5, kd=0.1, torque=0.0)
                if fi.error_code == "0x1":
                    warm_ok += 1
                raw_pos = fi.position
            except Exception:  # noqa: BLE001
                pass
            time.sleep(0.01)
        # Convert the final RAW read to sim-frame for the hold seed (direction/offset).
        start = float(chain._joint_position_real_to_sim_idx(raw_pos, 0))
        print(f"[selftest] motor enabled (warmup {warm_ok}/10 ok), start={start:+.4f} rad", flush=True)
    else:
        try:
            start = float(chain.read_states()[0].pos)
        except Exception:  # noqa: BLE001
            pass
    chain.commands = [MotorCmd(torque=0.0, pos=start, vel=0.0, kp=0.0, kd=0.0)]
    chain.start_thread()
    time.sleep(0.2)

    def _fail(reason: str) -> SelfTestResult:
        print(f"[selftest] FAIL: {reason}", file=sys.stderr, flush=True)
        return SelfTestResult("FAIL", nan, nan, nan, reason)

    try:
        if not getattr(chain, "running", True):
            return _fail("control thread died during bring-up (motor fault before homing)")

        # --- AUTO-HOME both stops ---
        start = float(chain.read_states()[0].pos)
        print(f"[selftest] start pos {start:+.3f} rad; homing toward OPEN (+) first", flush=True)
        ep_a, abort = _drive_until_stall(
            chain, args.motor_id, args.motor_type, np, start_pos=start, direction=+1,
            step=args.step, kp=args.kp_home, kd=args.kd_home, max_lead=args.max_lead,
            rate_hz=args.rate_hz, stall_eps=args.stall_eps, stall_ticks=args.stall_ticks,
            min_push_ticks=args.min_push_ticks, timeout_s=args.dir_timeout,
            temp_limit_c=args.temp_limit, label="OPEN(+)")
        if abort:
            return _fail(abort)
        if not getattr(chain, "running", True):
            return _fail("control thread died while homing toward OPEN")
        print(f"[selftest] OPEN stop at {ep_a:+.3f} rad; homing toward CLOSED (-)", flush=True)
        ep_b, abort = _drive_until_stall(
            chain, args.motor_id, args.motor_type, np, start_pos=ep_a, direction=-1,
            step=args.step, kp=args.kp_home, kd=args.kd_home, max_lead=args.max_lead,
            rate_hz=args.rate_hz, stall_eps=args.stall_eps, stall_ticks=args.stall_ticks,
            min_push_ticks=args.min_push_ticks, timeout_s=args.dir_timeout,
            temp_limit_c=args.temp_limit, label="CLOSED(-)")
        if abort:
            return _fail(abort)
        if not getattr(chain, "running", True):
            return _fail("control thread died while homing toward CLOSED")

        eps = classify_endpoints(ep_a, ep_b, args.min_travel, args.max_travel)
        if eps is None:
            return _fail(f"travel {abs(ep_a - ep_b):.3f} rad outside the plausible band "
                         f"[{args.min_travel}, {args.max_travel}] rad (jaws didn't move a real stroke — "
                         "stuck, dead, something held, a mid-stroke false-stall, or a missed stop?)")
        open_pos, closed_pos = eps
        print(f"[selftest] discovered endpoints: OPEN={open_pos:+.3f}  CLOSED={closed_pos:+.3f}  "
              f"travel={abs(open_pos - closed_pos):.3f} rad", flush=True)

        # --- VERIFY: one full open -> close -> open cycle, watch the encoder span ---
        print("[selftest] verifying: full open -> close -> open cycle", flush=True)
        min_seen = float("inf")
        max_seen = float("-inf")
        span = open_pos - closed_pos
        dt = 1.0 / max(args.rate_hz, 1.0)
        # We are currently AT the closed stop. Sweep to open, back to closed, to open.
        for goal in (open_pos, closed_pos, open_pos):
            command = float(chain.read_states()[0].pos)
            t0 = time.monotonic()
            while True:
                states = chain.read_states()
                actual = float(states[0].pos)
                min_seen = min(min_seen, actual)
                max_seen = max(max_seen, actual)
                temp_rotor = _safe_temp(states[0], "temp_rotor")
                temp_mos = _safe_temp(states[0], "temp_mos")
                if args.temp_limit > 0 and (temp_rotor >= args.temp_limit or temp_mos >= args.temp_limit):
                    return _fail(f"over-temp during verify (rotor={temp_rotor:.0f}C mos={temp_mos:.0f}C)")
                if not getattr(chain, "running", True):
                    return _fail("control thread died during verify cycle")
                d = goal - command
                s = abs(args.step)
                command = command + (s if d > 0 else -s) if abs(d) > s else goal
                command = float(min(max(command, actual - args.max_lead), actual + args.max_lead))
                chain.set_commands(torques=np.zeros(1), pos=np.array([command]), vel=np.zeros(1),
                                   kp=np.array([args.kp_home]), kd=np.array([args.kd_home]))
                if abs(actual - goal) <= args.settle_tol:
                    break
                if time.monotonic() - t0 > args.dir_timeout:
                    break  # tolerated: traversed_ok below is the real gate
                time.sleep(dt)

        if not traversed_ok(min_seen, max_seen, open_pos, closed_pos, args.verify_frac):
            return _fail(f"verify cycle covered only {abs(max_seen - min_seen):.3f} rad of the "
                         f"{abs(span):.3f} rad stroke (jaws did not fully open+close)")

        print(f"[selftest] verify OK: encoder spanned {abs(max_seen - min_seen):.3f} rad "
              f"of {abs(span):.3f} rad", flush=True)
        return SelfTestResult("PASS", open_pos, closed_pos, abs(open_pos - closed_pos))
    finally:
        # De-energize on EVERY exit path (pass, fail, or exception). ORDER MATTERS:
        # STOP the control thread BEFORE motor_off. The i2rt control loop auto-
        # recovers a motor it sees disabled (_try_recover_motors → clean_error +
        # motor_on, dm_driver.py:632-643), so if we motor_off() while the thread is
        # still running, its very next ~4 ms tick reads "disabled" and RE-ENABLES the
        # motor we just turned off — leaving it energized. So: (1) flip running=False
        # and let the loop exit, (2) THEN motor_off (now nothing re-enables it), (3)
        # THEN close the bus. The sleep both lets the loop drain its current iteration
        # (avoids i2rt's post-close fd race) and ensures it has observed running=False
        # before we power down.
        try:
            chain.running = False
        except Exception:  # noqa: BLE001
            pass
        time.sleep(0.1)  # let the control loop see running=False and exit its while
        try:
            mi.motor_off(args.motor_id)
        except Exception:  # noqa: BLE001
            pass
        try:
            chain.close()
        except Exception:  # noqa: BLE001
            pass


def _parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--channel", default="can0", help="SocketCAN channel (must be UP)")
    p.add_argument("--motor-id", type=int, default=7)
    p.add_argument("--motor-type", default="DM4310")
    p.add_argument("--motor-direction", type=int, default=-1,
                   help="must match keyboard_gripper (default -1) so endpoints are in the same frame")
    # Homing safety knobs. kp-home MATCHES the teleop i2rt kp (3.0): the overheat
    # guard (--max-lead) is what bounds torque now (to kp*max_lead ≈ 1.5 Nm — the
    # same torque a 0.5-rad teleop tracking error produces, demonstrably safe since
    # teleop tracks the full stroke without overheating; the original failure was at
    # SATURATED torque from an unclamped ~6-rad error). A weaker kp here would risk a
    # FALSE-FAIL: too little torque to overcome static friction → the jaw never moves
    # → homing reads an instant "stall" at the start → travel≈0 → spurious FAIL.
    p.add_argument("--kp-home", type=float, default=3.0,
                   help="homing gain (matches teleop 3.0; torque is bounded by --max-lead, not by a weak kp)")
    p.add_argument("--kd-home", type=float, default=0.3)
    p.add_argument("--max-lead", type=float, default=0.5,
                   help="overheat guard: max rad the command may lead the encoder (bounds stall torque to kp*max_lead)")
    p.add_argument("--step", type=float, default=0.05, help="per-tick ramp (rad/tick)")
    p.add_argument("--rate-hz", type=float, default=100.0)
    p.add_argument("--stall-eps", type=float, default=0.01,
                   help="no-motion threshold (rad) across the stall window = hit a stop")
    p.add_argument("--stall-ticks", type=int, default=10, help="ticks of no motion to call a stall")
    # ANTI-FALSE-STALL: a stall is only declared after the encoder has stayed FROZEN
    # WHILE pushing for this many consecutive ticks (the counter resets the instant
    # the encoder advances, so the grace is LOCAL to each obstruction — a fresh
    # sticky spot anywhere in the stroke gets a full new countdown, and a free-but-
    # sticky jaw breaks away within it). This rides out a slow static-friction
    # breakaway WITHOUT a distance-from-start arming guard (a real stop can be nearer
    # the start than any fixed arming distance, which would then time out — observed
    # on hardware). At 100 Hz, 25 ticks ≈ 0.25 s frozen-while-pushing before stopped.
    p.add_argument("--min-push-ticks", type=int, default=25,
                   help="consecutive frozen-while-pushing ticks before the encoder counts as stopped")
    p.add_argument("--dir-timeout", type=float, default=8.0, help="max seconds per homing/verify sweep")
    p.add_argument("--temp-limit", type=float, default=70.0, help="abort if rotor/MOS temp >= this (C); <=0 off")
    # Two-sided plausibility band on the discovered stroke. The real YAM linear_4310
    # stroke is ~5.1-5.3 rad. The FLOOR rejects a truncated discovery (partial jam,
    # or a mid-stroke false-stall) — verify_frac only checks the DISCOVERED range so
    # it can't catch truncation, making this band the sole guard. The CEILING rejects
    # an implausibly long discovery (missed stop / slipping mechanism). Override both
    # for a mechanically different gripper.
    p.add_argument("--min-travel", type=float, default=4.5,
                   help="discovered stroke FLOOR (rad); below this the test FAILs (dead/stuck/jammed/truncated)")
    p.add_argument("--max-travel", type=float, default=6.5,
                   help="discovered stroke CEILING (rad); above this the test FAILs (missed stop/slipping)")
    p.add_argument("--settle-tol", type=float, default=0.05, help="verify: 'reached goal' tolerance (rad)")
    p.add_argument("--verify-frac", type=float, default=0.8,
                   help="verify PASS needs the cycle to cover this fraction of the discovered stroke")
    return p.parse_args(argv if argv is not None else sys.argv[1:])


def main(argv=None) -> int:
    args = _parse_args(argv)
    try:
        res = run_selftest(args)
    except KeyboardInterrupt:
        print("[selftest] interrupted", file=sys.stderr, flush=True)
        return 130
    except Exception as exc:  # noqa: BLE001 - any bring-up/CAN failure is a FAIL, not a crash
        res = SelfTestResult("FAIL", float("nan"), float("nan"), float("nan"), f"exception: {exc!r}")
    # The machine-readable line the launcher parses (always printed, last).
    print(res.result_line(), flush=True)
    return 0 if res.status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
