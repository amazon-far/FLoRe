#!/usr/bin/env python3
"""Two-key keyboard teleop for a single YAM linear_4310 gripper.

Drives ONE ``LinearGripperDriver`` from a 2-key macro keyboard so an
operator can open / close the gripper by hand, with no ROS and no
holosoma in the loop. It is the interactive cousin of ``gripper_test``
(which just runs a fixed open/close sweep): here you decide.

Key map (a 2-key macropad that emits ``o`` and ``p``) — MOMENTARY /
hold-to-move::

    HOLD o  →  keep OPENING  (toward the open endpoint)
    HOLD p  →  keep CLOSING  (toward the closed endpoint)
    release →  STOP / freeze where it is
    q / ESC / Ctrl-C  →  quit (hold at current pos, then release)

While a key is held the commanded position ramps toward that endpoint at
``--max-delta`` rad/tick (the same per-tick velocity clamp the ROS hand
pipeline uses); the instant you let go it freezes. "Held" is inferred
from X keyboard autorepeat: a held key keeps delivering events, and the
loop treats ``--hold-timeout`` seconds of silence as a release. ``main``
shrinks the autorepeat initial delay on startup (via ``xset``) so a hold
responds immediately, and restores it on exit. Because a bare 2-key pad
usually can't send ``q``, **Ctrl-C in the launching terminal also quits
cleanly** (curses runs in cbreak mode, so SIGINT is still delivered).

Two driver backends (``--backend``):

* ``i2rt`` — talk to i2rt's ``DMChainCanInterface`` directly over a
  SocketCAN channel (a generic CANable / gs_usb adapter shows up as
  ``can0``). This is the API the YAM manual documents; it needs no
  ``gmp_i2rt`` PEAK layer and no MotorChainRobot (so no mujoco). The
  channel must already be UP::

      sudo ip link set can0 up type can bitrate 1000000

* ``gmp_i2rt`` — an optional robot-side path (its driver package is not part of this release)
  (``gmp_i2rt.LinearGripperDriver``): resolves PEAK device-id serials,
  wraps a MotorChainRobot, and applies the YAM linear_4310 joint-limit
  override (its ``detect_gripper_limits`` under-reports travel as
  ~0.07 rad, so we force ``[closed, open]``).

``--backend auto`` (default) picks ``gmp_i2rt`` when ``--serial`` is set,
``i2rt`` otherwise.

Examples::

    # CANable on can0 (auto-selects the i2rt backend), o=open / p=close
    sudo ip link set can0 up type can bitrate 1000000   # once
    python -m holosoma_hand_pipeline.keyboard_gripper --channel can0

    # robot PEAK adapter (auto-selects gmp_i2rt)
    python -m holosoma_hand_pipeline.keyboard_gripper --serial peak_device_id_5

    # iterate the CLI / key handling away from the robot
    HOLOSOMA_GRIPPER_DRY_RUN=1 python -m holosoma_hand_pipeline.keyboard_gripper

Env knobs:
    ``HOLOSOMA_GRIPPER_DRY_RUN=1`` — skip CAN bring-up; drive an
    in-memory stub instead. Useful for checking the key loop + ramp on a
    laptop with no gripper attached.
"""

from __future__ import annotations

import argparse
import curses
import os
import signal
import sys
import threading
import time
from typing import Any

from holosoma_hand_pipeline.defaults import GRIPPER_CLOSED_POS, GRIPPER_OPEN_POS

# Key codes for the two control keys (lower + upper so CapsLock / a
# shifted macropad layer still work). ESC is 27; Ctrl-C is 3 (delivered
# as a keycode only if curses were in raw mode — we run cbreak, so it
# arrives as KeyboardInterrupt, but we accept the code defensively too).
_OPEN_KEYS = (ord("o"), ord("O"))
_CLOSE_KEYS = (ord("p"), ord("P"))
_QUIT_KEYS = (ord("q"), ord("Q"), 27, 3)


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


class _EncoderPublisher:
    """Optional localhost WebSocket broadcaster of the live encoder reading.

    OFF by default (``--encoder-ws-port`` unset): when enabled, a daemon thread
    runs a tiny ``websockets`` server on its own asyncio loop and broadcasts the
    latest commanded/actual gripper position so a read-only consumer (the
    data-collection viz) can mirror the gripper without ever touching the CAN
    bus. Keeping it inside the teleop process preserves the single-CAN-writer
    invariant — the same process that owns the motor is the one publishing its
    state, so there is no second reader contending on ``can0``.

    The control loop calls :meth:`set_position` each tick (cheap: just latches
    two floats under a lock); the broadcaster ships them at its own rate. The
    wire frame is ``{"seq", "t", "open", "closed", "cmd", "actual"}`` where
    ``open``/``closed`` are the travel endpoints (so the consumer can normalize
    to a 0..1 open fraction without hard-coding the range).

    The socket is **bidirectional**: when ``--web-control`` is on, the browser
    sends control intents on the SAME connection and :meth:`get_target_frac`
    hands the latched target to the headless control loop. Accepted inbound
    frames (JSON object): ``{"cmd": "open"|"close"|"stop"}`` (open/close ramp to
    an endpoint, stop freezes) or ``{"target_frac": 0.0..1.0}`` (ramp to a
    fraction of the open stroke). This keeps the control channel inside the
    process that already owns the CAN bus — the browser never touches CAN.
    """

    def __init__(self, host: str, port: int, open_pos: float, closed_pos: float):
        self._host = host
        self._port = int(port)
        self._open = float(open_pos)
        self._closed = float(closed_pos)
        self._lock = threading.Lock()
        self._cmd = float(closed_pos)
        self._actual = float(closed_pos)
        self._seq = 0
        # Latched browser target (open fraction in [0,1]); None == hold/freeze
        # at the current position. Read each tick by the headless control loop.
        self._target_frac: float | None = None
        # 2-key keyboard bridge liveness. That bridge is a SEPARATE process that
        # connects to THIS WS to send o/p intents, so it is otherwise invisible to
        # the viz. It stamps a heartbeat (or tags its intents src="kbd") and we
        # surface kbd_online in the snapshot the viz already reads — no extra
        # socket, and this process stays the sole CAN writer.
        self._kbd_last_t = 0.0
        self._kbd_timeout = 2.0
        # Motor liveness, surfaced to the viz + health probe. Starts True; the
        # headless loop flips it False the moment the underlying i2rt control thread
        # dies on an unrecoverable fault (e.g. a hard-stop stall overheats the motor)
        # so the page shows the gripper RED instead of green-lighting a dead motor.
        self._motor_ok = True
        self._loop = None
        self._stop = None
        self._thread = None
        # Bind handshake: the WS server binds on the loop thread, so start()
        # waits on _bound to confirm the socket is actually listening before
        # returning. _start_error carries the bind failure (e.g. port in use)
        # back to the caller so --web-control can fail fast instead of running a
        # control loop that can never receive a command.
        self._bound = threading.Event()
        self._start_error: BaseException | None = None

    def set_position(self, cmd: float, actual: float) -> None:
        with self._lock:
            self._cmd = float(cmd)
            self._actual = float(actual)
            self._seq += 1

    def set_target_frac(self, frac: float | None) -> None:
        """Latch the browser's commanded open fraction (None == freeze).

        Non-finite inputs (NaN/inf) are rejected as freeze: NaN would defeat the
        clamp (every NaN comparison is False, so ``min(max(nan,0),1)`` is nan)
        and then drive the motor a fixed step toward "nowhere" every tick — i.e.
        a single malformed frame could creep the gripper shut. Treat it as a
        no-command freeze so a bad frame can never move the motor."""
        import math  # noqa: PLC0415

        if frac is None or not math.isfinite(float(frac)):
            with self._lock:
                self._target_frac = None
            return
        with self._lock:
            self._target_frac = float(min(max(frac, 0.0), 1.0))

    def get_target_frac(self) -> float | None:
        with self._lock:
            return self._target_frac

    def set_motor_ok(self, ok: bool) -> None:
        """Latch the motor's liveness (False once the control thread has died), so
        the broadcast snapshot reports it and the viz/health-probe see the fault."""
        with self._lock:
            self._motor_ok = bool(ok)

    def mark_kbd_alive(self) -> None:
        """Stamp the 2-key bridge as alive (called on any frame it sends, incl. a
        bare heartbeat). The snapshot reports it online until ``_kbd_timeout`` of
        silence — so the viz shows the keyboard going offline when its process
        dies, even though it never moves the gripper while you're not pressing."""
        with self._lock:
            self._kbd_last_t = time.time()

    def _kbd_online(self) -> bool:
        with self._lock:
            return self._kbd_last_t > 0.0 and (time.time() - self._kbd_last_t) < self._kbd_timeout

    def _apply_inbound(self, raw) -> None:
        """Parse one inbound control frame from the browser and latch a target.

        Tolerant by design: anything unparseable / unknown is ignored so a
        malformed frame can never crash the control channel or move the motor.
        """
        import json  # noqa: PLC0415

        def _reject_non_finite(_c):
            # parse_constant fires on the JSON NaN/Infinity/-Infinity literals
            # (json.loads accepts them by default). Raising rejects the WHOLE
            # frame so a non-finite value anywhere can never reach the motor —
            # belt-and-suspenders with set_target_frac's isfinite guard.
            raise ValueError("non-finite JSON constant")

        try:
            msg = json.loads(raw, parse_constant=_reject_non_finite)
        except (ValueError, TypeError):
            return
        if not isinstance(msg, dict):
            return
        # Any well-formed frame from the 2-key bridge stamps its liveness; a frame
        # tagged src="kbd" (its periodic heartbeat or a real o/p intent) keeps the
        # viz's keyboard indicator online between key presses. The browser's own
        # control frames carry no "src", so they don't masquerade as the keyboard.
        if msg.get("src") == "kbd":
            self.mark_kbd_alive()
        # A bare heartbeat (no command payload) only marks liveness — nothing to do.
        if msg.get("cmd") == "ping":
            return
        if "target_frac" in msg:
            try:
                self.set_target_frac(float(msg["target_frac"]))
            except (ValueError, TypeError):
                pass
            return
        cmd = msg.get("cmd")
        if cmd == "open":
            self.set_target_frac(1.0)
        elif cmd == "close":
            self.set_target_frac(0.0)
        elif cmd == "stop":
            self.set_target_frac(None)

    def _snapshot(self) -> str:
        import json  # noqa: PLC0415

        with self._lock:
            cmd, actual, seq, motor_ok = self._cmd, self._actual, self._seq, self._motor_ok
        return json.dumps(
            {"seq": seq, "t": time.time(), "open": self._open, "closed": self._closed, "cmd": cmd, "actual": actual,
             "kbd_online": self._kbd_online(), "motor_ok": motor_ok}
        )

    def start(self) -> None:
        import asyncio  # noqa: PLC0415

        import websockets  # noqa: PLC0415  (optional dep; only needed when enabled)

        def _run() -> None:
            loop = asyncio.new_event_loop()
            self._loop = loop
            asyncio.set_event_loop(loop)
            clients: set = set()

            async def handler(ws):
                clients.add(ws)
                try:
                    async for raw in ws:
                        # Inbound = browser control intents (open/close/stop or
                        # target_frac). Latched for the control loop to apply.
                        self._apply_inbound(raw)
                except Exception:  # noqa: BLE001
                    pass
                finally:
                    clients.discard(ws)

            async def broadcast():
                self._stop = asyncio.Event()
                async with websockets.serve(handler, self._host, self._port):
                    print(f"[keyboard_gripper] encoder WS at ws://{self._host}:{self._port}")
                    self._bound.set()  # signal start() the bind succeeded
                    while not self._stop.is_set():
                        if clients:
                            websockets.broadcast(clients, self._snapshot())
                        await asyncio.sleep(1.0 / 30.0)

            try:
                loop.run_until_complete(broadcast())
            except Exception as exc:  # noqa: BLE001
                # Bind failed (port in use, etc.) or the server died — record it
                # and release start()'s wait so the caller sees the failure.
                self._start_error = exc
                print(f"[keyboard_gripper] encoder WS stopped: {exc!r}")
            finally:
                self._bound.set()
                loop.close()

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()
        # Wait for the bind to confirm (or fail). Raise so --web-control can
        # fail fast rather than enter a control loop with a dead channel.
        if not self._bound.wait(timeout=5.0):
            raise RuntimeError(f"encoder WS did not bind on {self._host}:{self._port} within 5s")
        if self._start_error is not None:
            raise self._start_error

    def stop(self) -> None:
        loop, ev = self._loop, self._stop
        if loop is not None and ev is not None:
            loop.call_soon_threadsafe(ev.set)


def _set_autorepeat_fast() -> bool:
    """Shrink X's key autorepeat delay so a HELD key starts repeating
    almost immediately (default initial delay is ~660 ms, which the
    momentary loop would misread as a release at the start of a hold).

    Sets ~80 ms initial delay, ~50 repeats/s. Returns True if applied
    (so the caller knows to restore on exit). No-op off X / no xset.
    """
    if not os.environ.get("DISPLAY"):
        return False
    import shutil  # noqa: PLC0415
    import subprocess  # noqa: PLC0415

    if shutil.which("xset") is None:
        return False
    try:
        subprocess.run(["xset", "r", "rate", "80", "50"], check=True, capture_output=True, timeout=3)
        return True
    except Exception:  # noqa: BLE001
        return False


def _restore_autorepeat() -> None:
    """Restore X's default autorepeat (660 ms delay, 25/s). Best-effort."""
    if not os.environ.get("DISPLAY"):
        return
    import shutil  # noqa: PLC0415
    import subprocess  # noqa: PLC0415

    if shutil.which("xset") is None:
        return
    try:
        subprocess.run(["xset", "r", "rate", "660", "25"], check=False, capture_output=True, timeout=3)
    except Exception:  # noqa: BLE001
        pass


def _step_toward(current: float, target: float, max_delta: float) -> float:
    """Move ``current`` toward ``target`` by at most ``max_delta``.

    The per-tick velocity clamp — identical in spirit to
    ``HandPipelineNode._apply`` and ``gripper_test._ramp_to``: a YAM
    linear_4310 snap-write can over-drive the DM4310, so even though a
    key tap latches a full-travel goal we only ever advance one
    ``max_delta`` step per control tick. Returns the new commanded
    position (== ``target`` once within one step).
    """
    delta = target - current
    step = abs(max_delta)
    if abs(delta) <= step:
        return target
    return current + (step if delta > 0 else -step)


def _clamp_to_actual_lead(commanded: float, actual: float, max_lead: float) -> float:
    """Cap the commanded position so it never leads the measured encoder by more
    than ``max_lead`` rad. This is the OVERHEAT GUARD.

    In MIT/PD position mode the motor torque is ``kp·(cmd − actual) + …``. When the
    jaws hit a hard stop (e.g. the absolute encoder re-referenced on a power-cycle
    so the configured endpoints sit past the real travel — see the launcher note),
    ``_step_toward`` keeps ramping ``cmd`` away from a frozen ``actual``; the error
    grows unbounded and ``kp·error`` saturates torque into the stall →
    over-temperature → the i2rt control thread dies. Clamping the error to
    ``±max_lead`` bounds the worst-case holding torque to ``kp·max_lead`` so a
    stall can never cook the motor, while leaving normal tracking untouched (a
    freely-moving jaw keeps ``actual`` within one ``max_delta`` of ``cmd``, far
    inside the window). ``max_lead <= 0`` disables the guard.
    """
    if max_lead is None or max_lead <= 0.0:
        return commanded
    lo = actual - max_lead
    hi = actual + max_lead
    return float(min(max(commanded, lo), hi))


def _gripper_motor_ok(gripper: Any) -> bool:
    """Best-effort liveness of the underlying motor driver.

    Returns the driver's ``motor_ok()`` when it exposes one (the i2rt backend flips
    it False once its background control thread dies on an unrecoverable fault), and
    ``True`` for backends/stubs without the method (dry-run, gmp_i2rt) so they are
    never falsely reported faulted. Any exception is treated as a fault."""
    probe = getattr(gripper, "motor_ok", None)
    if probe is None:
        return True
    try:
        return bool(probe())
    except Exception:  # noqa: BLE001 - a raising health check IS a fault
        return False


def _key_to_target(key: int, open_pos: float, closed_pos: float, current_target: float) -> float:
    """Map a key code to the latched full-travel target.

    ``o`` → open, ``p`` → close, anything else → unchanged. Kept pure
    (no curses / CAN) so the key semantics are unit-testable.

    NOTE: this is the LATCH semantics (a tap drives to the endpoint). The
    interactive loop uses the MOMENTARY semantics below instead; this
    helper is retained for tests / potential reuse.
    """
    if key in _OPEN_KEYS:
        return open_pos
    if key in _CLOSE_KEYS:
        return closed_pos
    return current_target


def _key_direction(key: int) -> int:
    """Momentary direction for a key: +1 toward open, -1 toward close, 0
    otherwise. Pure (no curses / CAN) so it's unit-testable."""
    if key in _OPEN_KEYS:
        return +1
    if key in _CLOSE_KEYS:
        return -1
    return 0


def _momentary_target(
    held_direction: int, current: float, open_pos: float, closed_pos: float, max_delta: float
) -> float:
    """Compute the next target for momentary (hold-to-move) control.

    While a direction is held the target leads ``current`` by one
    ``max_delta`` step toward the corresponding endpoint, so the ramp in
    ``_step_toward`` keeps the motor moving that way. On release
    (``held_direction == 0``) the target is pinned to ``current`` — the
    gripper freezes wherever it is. Never commands past the endpoints.

    ``open_pos``/``closed_pos`` may be in either numeric order (open can
    be the larger or smaller value); the step is taken in the numeric
    direction from closed→open for +1 and open→closed for -1.
    """
    if held_direction == 0:
        return current
    toward = open_pos if held_direction > 0 else closed_pos
    other = closed_pos if held_direction > 0 else open_pos
    lo, hi = (other, toward) if toward >= other else (toward, other)
    step = abs(max_delta)
    nxt = current + (step if toward >= other else -step)
    return float(min(max(nxt, lo), hi))


class _DryRunGripper:
    """In-memory stand-in for ``LinearGripperDriver`` (no CAN touch).

    Mirrors ``set_position`` / ``get_position`` / ``close`` so the key
    loop + ramp can be exercised with ``HOLOSOMA_GRIPPER_DRY_RUN=1`` on a
    laptop with no gripper attached.
    """

    def __init__(self, initial: float = GRIPPER_CLOSED_POS, name: str = "dry_run_gripper") -> None:
        self._name = name
        self._position = float(initial)

    def set_position(self, position: float) -> None:
        self._position = float(position)

    def get_position(self) -> float:
        return self._position

    def close(self) -> None:
        return


class _I2rtGripper:
    """Drive a single DM motor over a SocketCAN channel via i2rt directly.

    Wraps i2rt's ``DMChainCanInterface`` (the API the YAM manual
    documents) so we can run a generic CANable / gs_usb adapter on
    ``can0`` with no gmp_i2rt PEAK-serial layer and no MotorChainRobot
    (so no mujoco dependency). The chain runs its own background control
    thread; ``set_position`` just latches the PD target it re-sends —
    the same pattern as i2rt's ``single_motor_position_pd_control``
    example. ``motor_direction`` is applied inside the chain on both the
    command and the read-back, so the open/closed sim-frame values are
    used directly (no joint-limit override needed — this path has no
    ``MotorChainRobot`` clipping).

    Bring-up does an explicit ``motor_off → clean_error → motor_on``
    cycle plus a short warm-up *before* starting the control thread. The
    DM4310 ships with its CAN command-timeout register set (8000 ms), so
    after sitting idle it latches the ``disabled`` (0x0) state; a plain
    ``start_thread=True`` construct then dies because the chain's own
    recovery loses the ~2-tick transient race right after re-enable
    (the motor still reports ``disabled`` for a couple of ticks). Doing
    the enable + warm-up holds synchronously first pushes past that
    window so the thread starts against an already-enabled motor.
    """

    def __init__(
        self,
        channel: str,
        motor_id: int,
        motor_type: str,
        motor_direction: int,
        kp: float,
        kd: float,
        name: str = "keyboard_gripper",
    ) -> None:
        import numpy as np  # noqa: PLC0415
        from i2rt.motor_drivers.dm_driver import DMChainCanInterface, MotorCmd, ReceiveMode  # noqa: PLC0415

        self._np = np
        self._kp = float(kp)
        self._kd = float(kd)
        self._name = name
        self._motor_id = int(motor_id)
        self._motor_type = str(motor_type)
        # close() may be called twice (the headless loop de-energizes on a motor
        # fault, then main()'s finally closes again on exit); guard it idempotent so
        # the second call can't double-shutdown the CAN bus.
        self._is_closed = False

        # Construct WITHOUT the background thread so we can hand-enable
        # the motor and warm it up before the control loop starts.
        self._chain = DMChainCanInterface(
            motor_list=[[motor_id, motor_type]],
            motor_offset=[0.0],
            motor_direction=[motor_direction],
            channel=channel,
            receive_mode=ReceiveMode.p16,
            motor_chain_name=name,
            start_thread=False,
        )

        hold = self._enable_and_warmup()

        # Seed the chain's latched command to "hold at current" so the
        # control thread doesn't snap the motor when it spins up, then
        # start it. After this, set_position()/get_position() drive it.
        self._chain.commands = [MotorCmd(torque=0.0, pos=hold, vel=0.0, kp=self._kp, kd=self._kd)]
        self._chain.start_thread()
        time.sleep(0.2)

    def _enable_and_warmup(self) -> float:
        """Explicitly enable the DM motor and warm it past the post-enable
        ``disabled`` transient. Returns the held position (rad)."""
        mi = self._chain.motor_interface
        mid = self._motor_id
        # off -> clean_error -> on: the sequence that reliably clears a
        # timeout-latched disabled state (a bare motor_on does not).
        try:
            mi.motor_off(mid)
        except Exception:  # noqa: BLE001
            pass
        time.sleep(0.05)
        try:
            mi.clean_error(mid)
        except Exception:  # noqa: BLE001
            pass
        time.sleep(0.05)
        info = mi.motor_on(mid, self._motor_type)
        # read-back here is in RAW motor frame; convert to sim frame for
        # the hold seed (direction/offset applied by the chain elsewhere).
        hold = self._chain._joint_position_real_to_sim_idx(info.position, 0)
        # Warm-up: a handful of direct hold commands so the motor is
        # solidly enabled (err==0x1) before the background thread starts.
        warm_ok = 0
        for _ in range(10):
            try:
                fi = mi.set_control(
                    motor_id=mid, motor_type=self._motor_type, pos=info.position, vel=0.0, kp=0.5, kd=0.1, torque=0.0
                )
                if fi.error_code == "0x1":
                    warm_ok += 1
            except Exception:  # noqa: BLE001
                pass
            time.sleep(0.01)
        print(f"[{self._name}] motor {mid} enabled (warmup {warm_ok}/10 ok), hold={hold:+.4f} rad")
        return float(hold)

    def set_position(self, position: float) -> None:
        # PD position command (torque ff = 0). kp/kd applied per write.
        self._chain.set_commands(
            torques=self._np.zeros(1),
            pos=self._np.array([float(position)]),
            vel=self._np.zeros(1),
            kp=self._np.array([self._kp]),
            kd=self._np.array([self._kd]),
        )

    def get_position(self) -> float:
        states = self._chain.read_states()
        return float(states[0].pos)

    def motor_ok(self) -> bool:
        """True while the chain's background control thread is alive and the motor
        has not latched an unrecoverable fault.

        i2rt's ``_set_torques_and_update_state`` sets ``self._chain.running = False``
        and re-raises on ANY unrecoverable control-loop error (e.g. the
        over-temperature a hard-stop stall produces) — every except branch in that
        loop flips ``running`` before exiting, so a False ``running`` is the
        authoritative "the control thread has died" signal. Once it dies,
        ``set_position`` writes are silently dropped and ``get_position`` returns a
        frozen value. The headless loop polls this so it can stop publishing a
        phantom moving command over a dead motor, de-energize, and surface the fault
        to the viz instead of green-lighting it."""
        try:
            return bool(getattr(self._chain, "running", False))
        except Exception:  # noqa: BLE001
            return False

    def close(self) -> None:
        # Idempotent: close() can be called twice (headless loop de-energizes on a
        # motor fault, then main()'s finally closes again on exit). The second call
        # is a no-op so it can't double-shutdown the CAN bus.
        if self._is_closed:
            return
        self._is_closed = True
        # Latch a hold-at-current command, then stop the chain. A short
        # settle lets the background thread drain before the socket is
        # torn down (avoids i2rt's post-close fd race on the loop thread).
        try:
            hold = self.get_position()
            self.set_position(hold)
            time.sleep(0.1)
        except Exception:  # noqa: BLE001
            pass
        # STOP the control thread BEFORE motor_off. The i2rt loop auto-recovers a
        # motor it sees disabled (_try_recover_motors → clean_error + motor_on), so
        # motor_off() while the thread still runs would be undone on its next tick,
        # leaving the gripper energized. Flip running=False and let it exit first,
        # THEN power down, THEN close the bus.
        try:
            self._chain.running = False
        except Exception:  # noqa: BLE001
            pass
        time.sleep(0.05)
        # Power the motor down so it's not left energized, and so the
        # next run's enable cycle starts from a known off state.
        try:
            self._chain.motor_interface.motor_off(self._motor_id)
        except Exception:  # noqa: BLE001
            pass
        self._chain.close()


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    # Backend selection:
    #   i2rt     — talk to i2rt's DMChainCanInterface directly over a
    #              SocketCAN channel (e.g. a CANable/gs_usb adapter on
    #              can0). This is the manual's documented API and needs
    #              no gmp_i2rt / no PEAK serial layer / no mujoco.
    #   gmp_i2rt — an optional robot-side path: gmp_i2rt.LinearGripperDriver,
    #              which resolves PEAK device-id serials and wraps a
    #              MotorChainRobot (driver package not part of this release).
    #   auto     — gmp_i2rt when --serial is set, i2rt when --channel is.
    p.add_argument(
        "--backend",
        choices=("auto", "i2rt", "gmp_i2rt"),
        default="auto",
        help="driver backend (default: auto — i2rt for --channel, gmp_i2rt for --serial)",
    )
    # Device selection. A generic CANable/gs_usb adapter shows up as a
    # SocketCAN channel (can0) — use --channel for it. PEAK adapters on
    # the robot resolve via --serial (peak_device_id_N) through gmp_i2rt.
    p.add_argument(
        "--serial",
        dest="serial",
        default="",
        help="CAN device serial for the gmp_i2rt backend (e.g. peak_device_id_6).",
    )
    p.add_argument(
        "--channel",
        dest="channel",
        default="can0",
        help="SocketCAN channel for the i2rt backend (default: can0). Must already be UP.",
    )
    # open/closed default per backend (resolved in _make_gripper):
    #   i2rt     → open=4.90 closed=0.00 — MEASURED on the laptop YAM
    #              linear_4310: the mechanical stops are at ~-0.08 (closed)
    #              and ~+5.02 (open) rad, a ~5.1 rad stroke; 4.90/0.00 sits
    #              just inside both stops so the motor isn't fighting them.
    #   gmp_i2rt → open=3.50 closed=0.00 — the holosoma sim-frame convention
    #              the hand pipeline uses on the robot.
    # Pass --open/--closed to override either default.
    p.add_argument("--open", type=float, default=None, help="motor position for fully open (rad)")
    p.add_argument("--closed", type=float, default=None, help="motor position for fully closed (rad)")
    p.add_argument("--rate-hz", type=float, default=50.0, help="control loop rate (Hz)")
    # 0.075 rad/tick at 50 Hz = 3.75 rad/s → full ~5 rad travel in ~1.3 s while held.
    # This is the OPEN/CLOSE speed while a key is held down (momentary control), and it
    # also bounds how fast the jaws track an absolute slider/target, since both go through
    # _step_toward. 1.5x the previous 0.05 (operator asked for a snappier grasp during
    # collection); still below HandPipelineNode's 0.10 original.
    #
    # ONE default serves EVERY hand and mode: the launcher never passes --max-delta, and
    # both the LEFT and RIGHT gripper processes are this same module, so 1pt/2pt/3pt all
    # move at this speed and cannot drift apart.
    #
    # Safe against the overheat guard below: that clamps how far cmd may LEAD the measured
    # encoder (--max-lead 0.6 rad), and a freely-moving jaw stays within ONE max_delta of
    # cmd — 0.075 is still 8x inside the clamp, so a stall trips the guard exactly as
    # before. Raising this much further would start eating that margin.
    p.add_argument("--max-delta", type=float, default=0.075, help="per-tick Δq move speed while held (rad/tick)")
    # OVERHEAT GUARD: cap how far the commanded position may lead the measured
    # encoder. In PD mode torque ≈ kp·(cmd−actual); if the jaws stall at a hard
    # stop (drifted endpoints command past the real travel) cmd ramps away from a
    # frozen actual and kp·error cooks the motor. Clamping the lead bounds the
    # worst-case stall torque to kp·max_lead. 0.6 rad with the i2rt kp=3.0 keeps a
    # stall well under the motor's continuous torque while never touching normal
    # tracking (a free jaw stays within one max_delta of cmd). <=0 disables.
    p.add_argument(
        "--max-lead",
        type=float,
        default=0.6,
        help="overheat guard: max rad the command may lead the encoder (bounds stall torque to kp*max_lead; <=0 off)",
    )
    # Momentary control: a key is "held" as long as X autorepeat keeps
    # delivering its events. Once no key arrives for this long, treat it as
    # released and freeze. 0.09 s spans ~4.5x the gap between repeats at the
    # autorepeat rate we set on startup (~50/s = 20 ms apart) — enough to
    # ride out a dropped repeat without a false release, while keeping the
    # post-release coast small (≈ hold_timeout × speed ≈ 0.09 s × 2.5 rad/s).
    p.add_argument("--hold-timeout", type=float, default=0.09, help="seconds of no key event = released (freeze)")
    # Gains default per backend (resolved in _make_gripper) because the
    # two paths have different torque semantics:
    #   gmp_i2rt → kp=1.0/kd=0.2  (MotorChainRobot force-mapped path; kp=20
    #              there saturates the DM4310 and strands the motor short)
    #   i2rt     → kp=3.0/kd=0.3  (direct MIT PD; measured to track the full
    #              0 → 3.5 rad stroke to within ~0.05 rad — kp=1 is too soft
    #              here and only reaches ~60% of travel)
    # Pass --kp/--kd to override either default.
    p.add_argument("--kp", type=float, default=None, help="position gain (default: i2rt 3.0 / gmp_i2rt 1.0)")
    p.add_argument("--kd", type=float, default=None, help="velocity gain (default: i2rt 0.3 / gmp_i2rt 0.2)")
    p.add_argument("--motor-id", type=int, default=7)
    p.add_argument("--motor-type", default="DM4310")
    # motor_direction=-1 matches the hand pipeline: positive motor rad =
    # open at the mechanism, so open=3.5 / closed=0.0 are sim-frame.
    p.add_argument("--motor-direction", type=int, default=-1)
    p.add_argument("--dry-run", action="store_true", help="skip CAN bring-up; drive an in-memory stub")
    # Optional read-only encoder publisher (OFF unless a port is given). Lets the
    # data-collection viz mirror the gripper over a localhost WebSocket WITHOUT a
    # second CAN reader — this process stays the sole CAN writer/reader.
    p.add_argument(
        "--encoder-ws-port",
        type=int,
        default=None,
        help="if set, broadcast the live encoder position over a localhost WebSocket on this port",
    )
    p.add_argument("--encoder-ws-host", default="127.0.0.1", help="encoder WS bind host (localhost only)")
    # Headless, browser-driven control. With --web-control the process runs NO
    # curses TTY loop and does NOT touch X key autorepeat; it ramps toward the
    # open fraction the viser page pushes over the encoder WebSocket (open/close/
    # stop buttons + slider). This lets the gripper run as a detached background
    # process and be controlled entirely from the web page — no focused terminal,
    # no global `xset` autorepeat change. Implies / requires --encoder-ws-port
    # (the same socket carries control IN and encoder state OUT).
    p.add_argument(
        "--web-control",
        action="store_true",
        help="headless: drive from the viser page over the encoder WebSocket (no TTY, no xset). "
        "Requires --encoder-ws-port.",
    )
    p.add_argument(
        "--initial-frac",
        type=float,
        default=1.0,
        help="[--web-control] startup open fraction, 0=closed 1=OPEN (default 1.0). NEGATIVE "
        "freezes at whatever the jaw was, which is the old behaviour and is only right when "
        "resuming a grasp on an object already held.",
    )
    return p.parse_args(argv)


def _resolve_backend(args: argparse.Namespace) -> str:
    """Pick the concrete backend for ``--backend auto``: gmp_i2rt when a
    serial is given (robot PEAK path), i2rt otherwise (SocketCAN)."""
    if args.backend != "auto":
        return args.backend
    return "gmp_i2rt" if args.serial else "i2rt"


# Per-backend default gains (overridable via --kp/--kd). See _parse_args.
_DEFAULT_GAINS = {"i2rt": (3.0, 0.3), "gmp_i2rt": (1.0, 0.2)}
# Per-backend default (open, closed) travel endpoints (overridable via
# --open/--closed). i2rt is MEASURED on the laptop hardware (stops at
# ~-0.08 / ~+5.02 rad); gmp_i2rt keeps the holosoma sim-frame convention.
_DEFAULT_ENDPOINTS = {"i2rt": (4.90, 0.0), "gmp_i2rt": (GRIPPER_OPEN_POS, GRIPPER_CLOSED_POS)}


def _resolve_gains(args: argparse.Namespace, backend: str) -> tuple[float, float]:
    kp_def, kd_def = _DEFAULT_GAINS[backend]
    kp = kp_def if args.kp is None else float(args.kp)
    kd = kd_def if args.kd is None else float(args.kd)
    return kp, kd


def _resolve_endpoints(args: argparse.Namespace, backend: str) -> tuple[float, float]:
    """Resolve (open, closed) for the backend, applying --open/--closed
    overrides. Returns the sim-frame travel endpoints in rad."""
    open_def, closed_def = _DEFAULT_ENDPOINTS[backend]
    open_pos = open_def if args.open is None else float(args.open)
    closed_pos = closed_def if args.closed is None else float(args.closed)
    return open_pos, closed_pos


def _make_gripper(args: argparse.Namespace, dry_run: bool):
    """Bring up the gripper (or a dry-run stub) for the chosen backend.

    Assumes ``args.open`` / ``args.closed`` have already been resolved to
    concrete floats (``main`` does this via ``_resolve_endpoints``)."""
    name = "keyboard_gripper"
    backend = _resolve_backend(args)
    kp, kd = _resolve_gains(args, backend)

    if dry_run:
        print(f"[{name}] dry-run (backend={backend} serial={args.serial!r} channel={args.channel!r}) — no CAN")
        return _DryRunGripper(initial=args.closed, name=name)

    if backend == "i2rt":
        # Direct SocketCAN path (CANable/gs_usb on can0). No PEAK serial,
        # no MotorChainRobot/mujoco. The channel must already be UP
        # (sudo ip link set <ch> up type can bitrate 1000000).
        print(f"[{name}] backend=i2rt channel={args.channel!r} kp={kp} kd={kd} (DMChainCanInterface, direct SocketCAN)")
        gripper = _I2rtGripper(
            channel=args.channel,
            motor_id=args.motor_id,
            motor_type=args.motor_type,
            motor_direction=args.motor_direction,
            kp=kp,
            kd=kd,
            name=name,
        )
        print(f"[{name}] initialized at q={gripper.get_position():.4f} rad")
        return gripper

    # gmp_i2rt backend (optional robot-side PEAK adapters; package not part of this release).
    from gmp_i2rt.linear_gripper_driver import LinearGripperDriver  # noqa: PLC0415

    print(f"[{name}] backend=gmp_i2rt serial={args.serial!r} channel={args.channel!r} kp={kp} kd={kd}")
    driver = LinearGripperDriver(
        can_serial=args.serial or None,
        can_channel=args.channel or None,
        motor_direction=args.motor_direction,
        motor_id=args.motor_id,
        motor_type=args.motor_type,
        motor_offset=0.0,
        kp=kp,
        kd=kd,
        gripper_name=name,
    )

    # gmp_i2rt wraps a MotorChainRobot, whose detect_gripper_limits
    # under-reports YAM linear_4310 travel (~0.07 rad vs the real
    # ~3.5 rad) — without this override _clip_robot_joint_pos_command
    # clips every write to that tiny range. (The i2rt backend above has
    # no MotorChainRobot, so it needs no override.)
    try:
        import numpy as _np  # noqa: PLC0415

        ad = getattr(driver, "_actuator_drive", None)
        if ad is not None:
            jl = getattr(ad, "_joint_limits", None)
            lo = float(min(args.open, args.closed))
            hi = float(max(args.open, args.closed))
            forced = _np.array([[lo, hi]])
            if jl is not None:
                print(
                    f"[{name}] joint_limits {jl.tolist()} -> {forced.tolist()} "
                    "(override: detect_gripper_limits under-reports on YAM linear_4310)"
                )
            ad._joint_limits = forced
    except Exception as exc:  # noqa: BLE001
        print(f"[{name}] joint_limits override failed: {exc!r}")

    print(f"[{name}] initialized at q={driver.get_position():.4f} rad")
    return driver


def _run_curses(stdscr: Any, gripper: Any, args: argparse.Namespace, publisher: Any = None) -> None:
    """Momentary (hold-to-move) key loop: while ``o`` is held the gripper
    opens, while ``p`` is held it closes, and on release it freezes where
    it is. Redraws a status panel. Patterned on i2rt's
    single_motor_position_pd_control, but driven by held-key direction
    rather than a latched endpoint.

    "Held" is inferred from X autorepeat: a held key keeps delivering
    key events; once none arrive for ``args.hold_timeout`` seconds the key
    is treated as released and the target is pinned to the current
    position. ``main`` shrinks the autorepeat delay on startup so the
    first repeat lands within the timeout (no false release at the start
    of a hold).
    """
    curses.curs_set(0)
    stdscr.nodelay(True)
    stdscr.timeout(0)

    dt = 1.0 / max(args.rate_hz, 1.0)
    hold_timeout = float(args.hold_timeout)
    # Seed from the live position so we hold the gripper exactly where it
    # is — no startup snap.
    current = float(gripper.get_position())
    target = current
    last_actual = current  # last CAN-read actual; refreshed in the 0.1 s status block
    held_direction = 0  # +1 opening, -1 closing, 0 released/idle
    last_dir_label = "(idle)"
    last_key_time = 0.0
    last_sent: float | None = None
    last_print = 0.0
    running = True

    while running:
        t0 = time.monotonic()

        # Drain all buffered key events this tick. The LAST directional
        # key seen this tick wins; any directional key also refreshes the
        # held timer. Quit keys exit immediately.
        saw_direction = False
        while True:
            key = stdscr.getch()
            if key == -1:
                break
            if key in _QUIT_KEYS:
                running = False
                break
            direction = _key_direction(key)
            if direction != 0:
                held_direction = direction
                saw_direction = True
                last_dir_label = "o → OPENING" if direction > 0 else "p → CLOSING"
        if saw_direction:
            last_key_time = t0

        # Release detection: no directional key within hold_timeout →
        # freeze. (last_key_time starts at 0, so before the first press
        # t0 - 0 > timeout → held_direction stays 0 → frozen at current.)
        if held_direction != 0 and (t0 - last_key_time) > hold_timeout:
            held_direction = 0
            last_dir_label = "(released — frozen)"

        # Momentary target: lead current by one step toward the held
        # endpoint, or pin to current when released.
        target = _momentary_target(held_direction, current, args.open, args.closed, args.max_delta)
        current = _step_toward(current, target, args.max_delta)
        # OVERHEAT GUARD: never let the command lead the encoder past max_lead, so
        # holding a key into a hard stop can't run torque away into over-temp.
        # last_actual is the most recent CAN read (refreshed in the status block).
        current = _clamp_to_actual_lead(current, last_actual, float(getattr(args, "max_lead", 0.0)))

        # Skip redundant writes: once parked the motor chain's background
        # thread re-sends the latched command (see HandPipelineNode._apply).
        if last_sent is None or current != last_sent:
            gripper.set_position(current)
            last_sent = current

        now = time.monotonic()
        # Publish the live position every tick (cheap latch) when enabled, so the
        # viz chart is smooth. ``actual`` is refreshed only in the 0.1 s status
        # block below (a CAN read), so reuse the last-known value between reads.
        if publisher is not None:
            publisher.set_position(current, last_actual)
        if now - last_print > 0.1:
            actual = float(gripper.get_position())
            last_actual = actual
            if held_direction > 0:
                status = "OPENING"
            elif held_direction < 0:
                status = "CLOSING"
            else:
                status = "HOLDING"
            stdscr.erase()
            stdscr.addstr(0, 0, "YAM linear_4310 — 2-key keyboard gripper teleop (hold to move)")
            stdscr.addstr(
                1, 0, f"Device     : backend={_resolve_backend(args)} channel={args.channel!r} serial={args.serial!r}"
            )
            stdscr.addstr(2, 0, f"Commanded  : {current:+.4f} rad   (actual {actual:+.4f})")
            stdscr.addstr(3, 0, f"Range      : OPEN={args.open:.2f}   CLOSED={args.closed:.2f} rad")
            stdscr.addstr(4, 0, f"Status     : {status}")
            stdscr.addstr(5, 0, f"Last action: {last_dir_label}")
            stdscr.addstr(7, 0, "HOLD [o] to open   •   HOLD [p] to close   •   release = stop")
            stdscr.addstr(8, 0, "Quit: [q] / ESC / Ctrl-C")
            stdscr.refresh()
            last_print = now

        elapsed = time.monotonic() - t0
        if elapsed < dt:
            time.sleep(dt - elapsed)

    # Hold at the last commanded position on the way out so the gripper
    # doesn't lurch when the control loop stops.
    try:
        gripper.set_position(current)
    except Exception:  # noqa: BLE001
        pass


def _run_headless(gripper: Any, args: argparse.Namespace, publisher: Any, stop_evt: threading.Event) -> None:
    """Headless control loop driven by the browser over the encoder WebSocket.

    The web-control counterpart to :func:`_run_curses`: instead of reading X
    key events from a focused TTY, it reads the latched open-fraction target the
    browser pushed onto ``publisher`` (``open``/``close``/``stop`` buttons or a
    slider). This needs NO terminal focus and does NOT touch X autorepeat, so it
    can run as a detached background process while you control the gripper from
    the viser page. Same per-tick velocity clamp + skip-redundant-write logic as
    the curses loop; ``stop_evt`` (set by SIGINT/SIGTERM) ends it cleanly.

    Target semantics: ``publisher.get_target_frac()`` returns an open fraction in
    [0,1] (mapped to ``[closed, open]`` rad) to ramp toward, or ``None`` to
    freeze at the current position (the ``stop`` intent / no command yet).

    Two safety layers run every tick:
      * OVERHEAT GUARD — the command is clamped so it never leads the measured
        encoder by more than ``args.max_lead`` rad (:func:`_clamp_to_actual_lead`),
        bounding stall torque so a drifted endpoint past a hard stop can't cook the
        motor.
      * DEAD-MOTOR DETECTION — if the i2rt control thread dies on an unrecoverable
        fault, we STOP publishing a phantom moving command (which would green-light
        a dead motor), latch ``motor_ok=False`` on the snapshot, de-energize, and
        freeze. The loop keeps running (still broadcasting the fault) until
        ``stop_evt`` so the viz can show the gripper RED rather than silently lying.
    """
    dt = 1.0 / max(args.rate_hz, 1.0)
    open_pos, closed_pos = float(args.open), float(args.closed)
    span = open_pos - closed_pos
    max_lead = float(getattr(args, "max_lead", 0.0))
    current = float(gripper.get_position())
    last_sent: float | None = None
    last_print = 0.0
    faulted = False  # latched once the motor dies; never green-lights again

    # STARTUP STATE IS A DECLARATION, NOT AN INHERITANCE. Without this the loop sat at
    # `target_frac=None` — freeze — until a command arrived, so "the initial state" meant
    # whatever the jaw happened to be: the self-test's last goal, the previous session's last
    # position, or an unknown one under SELFTEST=0. A jaw that comes up shut reads as a jaw
    # HOLDING something. Both hands must start OPEN, on every low level.
    initial_frac = float(getattr(args, "initial_frac", 1.0))
    if publisher is not None and initial_frac >= 0.0:
        publisher.set_target_frac(min(max(initial_frac, 0.0), 1.0))
        print(f"[keyboard_gripper] web-control: startup target_frac={initial_frac:.2f} "
              f"({'OPEN' if initial_frac >= 0.999 else 'partial'}) — ramping there before any "
              "page command", flush=True)
    else:
        print("[keyboard_gripper] web-control: startup FROZEN at the current position "
              "(--initial-frac < 0)", flush=True)

    print("[keyboard_gripper] web-control: waiting for open/close/stop from the viser page", flush=True)
    while not stop_evt.is_set():
        t0 = time.monotonic()

        # Read the live encoder ONCE per tick: it feeds both the overheat-guard
        # clamp and the published 'actual'. Fall back to last command if the read
        # raises (a dead chain) — _gripper_motor_ok below is the authoritative fault
        # signal, not a single failed read.
        try:
            actual = float(gripper.get_position())
        except Exception:  # noqa: BLE001
            actual = current

        # DEAD-MOTOR DETECTION: once the underlying control thread has died, stop
        # commanding (a phantom ramp over a frozen encoder is exactly what
        # green-lit a dead motor before). Latch the fault, de-energize once, and
        # keep broadcasting motor_ok=False so the page shows it RED.
        if not faulted and not _gripper_motor_ok(gripper):
            faulted = True
            print("[keyboard_gripper] MOTOR FAULT: control thread died (likely a stall/over-temp); "
                  "freezing + de-energizing. Re-measure endpoints and relaunch.", file=sys.stderr, flush=True)
            if publisher is not None:
                publisher.set_motor_ok(False)
            try:
                gripper.close()  # powers the motor down
            except Exception:  # noqa: BLE001
                pass

        if not faulted:
            frac = publisher.get_target_frac() if publisher is not None else None
            if frac is None:
                target = current  # freeze where we are
            else:
                target = closed_pos + frac * span
            current = _step_toward(current, target, args.max_delta)
            # OVERHEAT GUARD: never let the command lead the encoder past max_lead.
            current = _clamp_to_actual_lead(current, actual, max_lead)

            if last_sent is None or current != last_sent:
                gripper.set_position(current)
                last_sent = current
        else:
            frac = None  # for the status line below

        now = time.monotonic()
        if publisher is not None:
            publisher.set_position(current, actual)
            if now - last_print > 1.0:
                last_print = now
                state = "FAULTED" if faulted else ("freeze" if frac is None else f"{frac:.2f}")
                print(
                    f"[keyboard_gripper] q={current:+.3f} rad (actual {actual:+.3f}) target_frac={state}",
                    flush=True,
                )

        elapsed = time.monotonic() - t0
        if elapsed < dt:
            time.sleep(dt - elapsed)

    if not faulted:
        try:
            gripper.set_position(current)
        except Exception:  # noqa: BLE001
            pass


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])

    if args.web_control and args.encoder_ws_port is None:
        print("[keyboard_gripper] --web-control requires --encoder-ws-port (the control channel).", file=sys.stderr)
        return 2

    backend = _resolve_backend(args)
    # Resolve per-backend default endpoints, then write the concrete
    # floats back onto args so the loop / status panel use them directly.
    args.open, args.closed = _resolve_endpoints(args, backend)

    if args.open == args.closed:
        print(
            f"[keyboard_gripper] open ({args.open}) must differ from closed ({args.closed}); "
            "equal values disable motion.",
            file=sys.stderr,
        )
        return 2

    dry_run = args.dry_run or _env_bool("HOLOSOMA_GRIPPER_DRY_RUN")
    print(f"[keyboard_gripper] dry_run={dry_run} backend={backend}")
    print(f"[keyboard_gripper] serial={args.serial!r} channel={args.channel!r}")
    print(f"[keyboard_gripper] range: OPEN={args.open:.2f} CLOSED={args.closed:.2f} rad")
    if args.web_control:
        print("[keyboard_gripper] web-control mode: open/close/stop from the viser page (no TTY, no xset)")
    else:
        print("[keyboard_gripper] HOLD o=open / HOLD p=close / release=stop / q,ESC,Ctrl-C=quit")

    try:
        gripper = _make_gripper(args, dry_run)
    except Exception as exc:  # noqa: BLE001
        print(f"[keyboard_gripper] bring-up failed: {exc!r}", file=sys.stderr)
        return 1

    # Optional read-only encoder WS publisher (single-CAN-owner safe).
    publisher = None
    if args.encoder_ws_port is not None:
        if args.encoder_ws_host not in ("127.0.0.1", "localhost"):
            print(
                f"[keyboard_gripper] refusing non-localhost encoder WS host {args.encoder_ws_host!r}", file=sys.stderr
            )
            return 2
        try:
            publisher = _EncoderPublisher(args.encoder_ws_host, args.encoder_ws_port, args.open, args.closed)
            publisher.start()
        except Exception as exc:  # noqa: BLE001
            print(f"[keyboard_gripper] encoder WS failed to start: {exc!r}", file=sys.stderr)
            publisher = None

    # In --web-control the WS is the SOLE control channel — if it never bound
    # (e.g. port already in use), a headless loop would freeze forever and the
    # operator's buttons would do nothing. Fail fast and power the motor down,
    # rather than silently holding an energized gripper no one can command.
    if args.web_control and publisher is None:
        print("[keyboard_gripper] --web-control but the encoder WS is unavailable; aborting.", file=sys.stderr)
        try:
            gripper.close()
        except Exception as exc:  # noqa: BLE001
            print(f"[keyboard_gripper] close raised: {exc!r}", file=sys.stderr)
        return 1

    # web-control: headless loop driven by the viser page — NO curses TTY and
    # NO global `xset` autorepeat change (the two reasons the curses path needs a
    # focused terminal and bleeds fast key-repeat into the whole X session). Runs
    # until SIGINT/SIGTERM, so it can be launched as a detached background process.
    if args.web_control:
        stop_evt = threading.Event()

        def _sig(_s, _f):
            stop_evt.set()

        signal.signal(signal.SIGINT, _sig)
        signal.signal(signal.SIGTERM, _sig)
        try:
            _run_headless(gripper, args, publisher, stop_evt)
        except KeyboardInterrupt:
            print("[keyboard_gripper] interrupted", file=sys.stderr)
        except Exception as exc:  # noqa: BLE001
            print(f"[keyboard_gripper] loop failed: {exc!r}", file=sys.stderr)
            return 1
        finally:
            if publisher is not None:
                publisher.stop()
            try:
                gripper.close()
            except Exception as exc:  # noqa: BLE001
                print(f"[keyboard_gripper] close raised: {exc!r}", file=sys.stderr)
        return 0

    # Speed up key autorepeat so a held key responds immediately; restore
    # the user's default on exit. Skipped under dry-run on a headless box.
    autorepeat_changed = _set_autorepeat_fast()

    try:
        # curses.wrapper restores the terminal even on exception. SIGINT
        # (Ctrl-C) still raises KeyboardInterrupt under cbreak — caught
        # here so a 2-key pad without a 'q' can still quit cleanly.
        curses.wrapper(_run_curses, gripper, args, publisher)
    except KeyboardInterrupt:
        print("[keyboard_gripper] interrupted", file=sys.stderr)
    except Exception as exc:  # noqa: BLE001
        print(f"[keyboard_gripper] loop failed: {exc!r}", file=sys.stderr)
        return 1
    finally:
        if publisher is not None:
            publisher.stop()
        if autorepeat_changed:
            _restore_autorepeat()
        try:
            gripper.close()
        except Exception as exc:  # noqa: BLE001
            print(f"[keyboard_gripper] close raised: {exc!r}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
