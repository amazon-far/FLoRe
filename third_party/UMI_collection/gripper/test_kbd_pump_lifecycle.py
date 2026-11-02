#!/usr/bin/env python
"""Lifecycle tests for the keyboard bridge's evdev pumps (no hardware needed).

Pins the fix for the traceback the collection console printed after a long idle
pause — one copy per keyboard node, on the first keypress that ended the pause:

    Exception in callback ReadIterator.__anext__.<locals>.next_batch_ready(...)
      .../evdev/eventio_async.py:48  future.set_result(next(self.current_batch))
    asyncio.exceptions.InvalidStateError: invalid state
    During handling of the above exception, another exception occurred:
      .../evdev/eventio_async.py:50  future.set_exception(e)
    asyncio.exceptions.InvalidStateError: invalid state

``async_read_loop()`` is not cancel-safe: cancelling a pump parked on it cancels
only the future evdev handed us, leaving evdev's ``loop.add_reader`` registration
and its done-callback armed. The next readable event then calls set_result() on a
cancelled future and evdev's own handler re-raises the same error out of
set_exception(). The trigger in production is the 30 s silent-device watchdog, so
the pump must release its device on EVERY exit path, cancellation included.

The fake device is a real ``evdev.eventio_async.EventIO`` over a pipe: only
``read()`` is stubbed, so ReadIterator, the loop reader and the done-callback under
test are all the shipped evdev code. ``close()`` deliberately does NOT close the
pipe (the real ``InputDevice.close()`` would close the fd) — keeping it writable
lets us prove the READER is gone rather than just that the fd is.

Run:
    .venv/bin/python gripper/test_kbd_pump_lifecycle.py
"""
from __future__ import annotations

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from evdev import InputEvent, ecodes  # noqa: E402
from evdev.eventio_async import EventIO  # noqa: E402

from keyboard_gripper_ws import (  # noqa: E402
    _CLOSE,
    _OPEN,
    GestureState,
    _pump,
    _recycle_stale_pump,
    make_intent_sender,
)

_PATH = "/dev/input/fake0"

# The LEFT hand's two key CODES, resolved from the bridge's own map rather than named
# here. The pair and which one means open/close are both operator preferences that have
# changed more than once; a test that hardcodes KEY_F breaks on the next swap and says
# nothing about the pump logic it is actually covering.
_K_OPEN = getattr(ecodes, _OPEN)
_K_CLOSE = getattr(ecodes, _CLOSE)


def _key(code: int, value: int) -> InputEvent:
    return InputEvent(0, 0, ecodes.EV_KEY, code, value)


class FakeDev(EventIO):
    """An evdev async device backed by a pipe (see the module docstring).

    Subclasses the async mixin, NOT InputDevice, whose ioctl-heavy ``__init__``
    needs a genuine /dev/input node. Everything the pump touches — the
    ReadIterator, the loop reader, the done-callback, ``close()`` — is real."""

    def __init__(self, path: str = _PATH, name: str = "fake pad"):
        self._r, self._w = os.pipe()
        self.fd = self._r
        self.path = path
        self.name = name
        self.close_calls = 0
        self.read_error: Exception | None = None
        self._queue: list[InputEvent] = []

    def feed(self, *events: InputEvent) -> None:
        """Queue events AND make the fd readable, i.e. simulate a keypress."""
        self._queue.extend(events)
        os.write(self._w, b"x")

    def read(self):
        os.read(self._r, 1)
        if self.read_error is not None:
            raise self.read_error
        batch, self._queue = self._queue, []
        return iter(batch)

    def close(self) -> None:
        self.close_calls += 1
        super().close()  # the load-bearing half: loop.remove_reader(self.fd)

    def dispose(self) -> None:
        for fd in (self._r, self._w):
            try:
                os.close(fd)
            except OSError:
                pass


class _Sink:
    """Collects what a pump would have sent."""

    def __init__(self):
        self.intents: list[str] = []
        self.hands: list[str] = []          # the hand each intent was routed to
        self.records: list[str] = []

    async def send_intent(self, cmd: str, hand: str = "left") -> None:
        self.intents.append(cmd)
        self.hands.append(hand)

    async def send_record_cmd(self, cmd: str) -> None:
        self.records.append(cmd)


def _drive(coro_factory):
    """Run one coroutine on a private loop; return (result, loop_level_errors)."""
    errors: list[dict] = []
    loop = asyncio.new_event_loop()
    loop.set_exception_handler(lambda _loop, ctx: errors.append(ctx))
    try:
        result = loop.run_until_complete(coro_factory())
        # Let any callback armed on the loop fire before we judge the run.
        loop.run_until_complete(asyncio.sleep(0.05))
    finally:
        loop.close()
    return result, errors


# --------------------------------------------------------------------------- #
# _pump
# --------------------------------------------------------------------------- #

def test_pump_delivers_a_keypress():
    """Guard against the rest of this file passing vacuously: the fake device must
    really drive _pump's shipped code path (evdev categorize + edge detect)."""
    dev, sink, gesture, last_event = FakeDev(), _Sink(), GestureState(), {}

    async def scenario():
        task = asyncio.create_task(_pump(dev, sink.send_intent, gesture,
                                         sink.send_record_cmd, last_event))
        await asyncio.sleep(0.02)
        dev.feed(_key(ecodes.KEY_S, 1))   # 's' = start this episode
        await asyncio.sleep(0.05)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    _, errors = _drive(scenario)
    dev.dispose()
    assert sink.records == ["start"], sink.records
    assert last_event[_PATH] > 0
    assert not errors, errors


def test_pump_releases_the_device_when_cancelled():
    """THE regression: cancel a parked pump (what the watchdog does), then press a
    key. Before the fix evdev's orphaned callback fired here and the loop printed
    InvalidStateError; the fix closes the device, unregistering the reader."""
    dev, sink, gesture = FakeDev(), _Sink(), GestureState()

    async def scenario():
        task = asyncio.create_task(_pump(dev, sink.send_intent, gesture,
                                         sink.send_record_cmd, {}))
        await asyncio.sleep(0.02)          # let it park inside async_read_loop()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        dev.feed(_key(ecodes.KEY_S, 1))    # the keypress that ends the pause

    _, errors = _drive(scenario)
    dev.dispose()
    assert not errors, f"orphaned evdev callback fired: {errors}"
    assert dev.close_calls >= 1, "a cancelled pump must release its device"
    # Cancellation is transparent: it must not fake a gripper 'stop' or a record
    # command on the shared state (that would freeze a hold on another keyboard).
    assert sink.intents == [] and sink.records == []


def test_pump_releases_the_device_when_unplugged():
    """The OSError path releases too — and before it awaits send_intent('stop'),
    so the node cannot wake the loop while the handler is suspended."""
    dev, sink, gesture = FakeDev(), _Sink(), GestureState()
    dev.read_error = OSError(19, "No such device")

    async def scenario():
        task = asyncio.create_task(_pump(dev, sink.send_intent, gesture,
                                         sink.send_record_cmd, {}))
        await asyncio.sleep(0.02)
        dev.feed()                        # readable -> read() raises ENODEV
        await asyncio.wait_for(task, timeout=1.0)

    _, errors = _drive(scenario)
    dev.dispose()
    assert dev.close_calls >= 1, "an unplugged device must be released"
    # One 'stop' per HAND, not one total: the vanished keyboard could have been holding
    # either key pair, the two grippers are separate processes, and each LATCHES its
    # target with no timeout — so stopping only the left hand would leave a held right
    # jaw closing forever. Sending a redundant stop is a no-op on the gripper side.
    assert sink.intents == ["stop", "stop"], sink.intents
    assert sorted(sink.hands) == ["left", "right"], sink.hands
    assert not errors, errors


# --------------------------------------------------------------------------- #
# _recycle_stale_pump
# --------------------------------------------------------------------------- #

def test_recycle_keeps_reading_the_old_node_while_it_swaps_in_the_new_one():
    """No gap, tested as the property that matters rather than as call order.

    The watchdog can only fire after a silence, so the keypress at risk is exactly
    the one that ENDS the operator's pause — the 's' that starts the next take. Here
    it lands mid-swap: with the stale pump cancelled first, nothing is reading that
    node and the press is lost; opening first keeps it covered. (Both fds being
    briefly live is safe: each evdev fd gets its own copy of the event stream.)"""
    stale_dev, fresh_dev = FakeDev(), FakeDev()
    sink, gesture = _Sink(), GestureState()

    async def scenario():
        stale = asyncio.create_task(_pump(stale_dev, sink.send_intent, gesture,
                                          sink.send_record_cmd, {}))
        pumps = {_PATH: stale}
        last_event = {_PATH: 1.0}
        made: list = []
        await asyncio.sleep(0.02)  # let the stale pump park inside async_read_loop()

        def opener(path):
            assert path == _PATH
            # Runs in the executor thread. Sleep first so that a cancel-then-open
            # order has definitely been processed by the loop, THEN press a key on
            # the node being recycled.
            import time as _t
            _t.sleep(0.05)
            stale_dev.feed(_key(ecodes.KEY_S, 1))
            _t.sleep(0.05)
            return fresh_dev

        def mk_pump(d):
            made.append(d)
            return asyncio.create_task(asyncio.sleep(3600))

        ok = await _recycle_stale_pump(_PATH, pumps, last_event, mk_pump,
                                       open_device=opener)
        assert ok is True
        assert made == [fresh_dev], "the same path must be re-attached"
        assert pumps[_PATH] is not stale
        assert _PATH not in last_event, "the heartbeat is re-seeded by mk_pump"
        pumps[_PATH].cancel()
        await asyncio.gather(stale, pumps[_PATH], return_exceptions=True)
        assert stale.cancelled(), "the stale pump must be retired"

    _, errors = _drive(scenario)
    stale_dev.dispose()
    fresh_dev.dispose()
    assert sink.records == ["start"], "the mid-swap keypress was dropped in the gap"
    assert stale_dev.close_calls >= 1, "...and the stale fd must still be released"
    assert not errors, errors


def test_recycle_survives_a_node_that_really_vanished():
    """Unplugged during the pause: report it, retire the stale pump, leave the path
    free so the 2 s rescan re-attaches it if the keyboard comes back."""
    async def scenario():
        stale = asyncio.create_task(asyncio.sleep(3600))
        pumps = {_PATH: stale}

        def opener(path):
            raise FileNotFoundError(2, "No such file or directory", path)

        ok = await _recycle_stale_pump(_PATH, pumps, {}, lambda d: None,
                                       open_device=opener)
        assert ok is False
        assert _PATH not in pumps, "a path left in pumps blocks the rescan"
        await asyncio.gather(stale, return_exceptions=True)
        assert stale.cancelled()

    _, errors = _drive(scenario)
    assert not errors, errors


def test_recycle_still_attaches_when_the_stale_pump_already_finished():
    """The pump can end on its own (unplug) between the watchdog's scan and here —
    with no task left to cancel, the re-attach must still happen."""
    fresh, sentinel = FakeDev(), object()

    async def scenario():
        pumps: dict = {}
        ok = await _recycle_stale_pump(_PATH, pumps, {}, lambda d: sentinel,
                                       open_device=lambda p: fresh)
        assert ok is True and pumps[_PATH] is sentinel

    _, errors = _drive(scenario)
    fresh.dispose()
    assert not errors, errors


# --------------------------------------------------------------------------- #
# the intent stream itself
# --------------------------------------------------------------------------- #

def test_a_repeated_press_is_queued_and_not_coalesced_away():
    """The sender must pass every intent through. The old `last_sent` guard dropped
    one equal to the previous intent; being process-global across every attached
    keyboard, that silently subtracts a press the operator made (see
    ``make_intent_sender``). Intents are idempotent downstream, so there is nothing
    to trade for it."""
    async def scenario():
        queue: asyncio.Queue = asyncio.Queue()
        send = make_intent_sender(queue)
        for cmd in ("open", "open", "open", "stop", "stop"):
            await send(cmd)
        return [queue.get_nowait() for _ in range(queue.qsize())]

    got, errors = _drive(scenario)
    assert got == ["open", "open", "open", "stop", "stop"], (
        f"the intent sender dropped a repeat: {got}")
    assert not errors, errors


def test_pump_emits_one_intent_per_key_event_it_acts_on():
    """The pump is a pure translator with no state of its own: three key-downs are
    three `open` intents. (A real node never delivers that — the kernel collapses a
    repeated key-down — which is the point: the pump must not add a second layer of
    filtering on top of the one guarantee it already has.)"""
    dev, sink, gesture = FakeDev(), _Sink(), GestureState()

    async def scenario():
        task = asyncio.create_task(_pump(dev, sink.send_intent, gesture,
                                         sink.send_record_cmd, {}))
        await asyncio.sleep(0.02)
        # A press whose release never arrives (pad swallows the key-up), then two
        # more presses of the SAME key.
        dev.feed(_key(_K_OPEN, 1))
        await asyncio.sleep(0.03)
        dev.feed(_key(_K_OPEN, 1))
        await asyncio.sleep(0.03)
        dev.feed(_key(_K_OPEN, 1))
        await asyncio.sleep(0.05)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    _, errors = _drive(scenario)
    dev.dispose()
    assert sink.intents == ["open", "open", "open"], (
        f"a repeated press must not be swallowed, got {sink.intents}")
    assert not errors, errors


def test_a_release_still_stops_and_repeated_stops_are_harmless():
    """The release is what freezes the jaws; a duplicate stop is a no-op downstream
    (set_target_frac(None) twice = freeze twice), so it needs no special-casing."""
    dev, sink, gesture = FakeDev(), _Sink(), GestureState()

    async def scenario():
        task = asyncio.create_task(_pump(dev, sink.send_intent, gesture,
                                         sink.send_record_cmd, {}))
        await asyncio.sleep(0.02)
        for ev in (_key(_K_OPEN, 1), _key(_K_OPEN, 0),
                   _key(_K_CLOSE, 1), _key(_K_CLOSE, 0)):
            dev.feed(ev)
            await asyncio.sleep(0.03)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    _, errors = _drive(scenario)
    dev.dispose()
    assert sink.intents == ["open", "stop", "close", "stop"], sink.intents
    assert not errors, errors


def test_autorepeat_never_becomes_an_intent():
    """value==2 is the kernel's autorepeat; acting on it is what made a held key
    emit a stream of characters into the whole X session."""
    dev, sink, gesture = FakeDev(), _Sink(), GestureState()

    async def scenario():
        task = asyncio.create_task(_pump(dev, sink.send_intent, gesture,
                                         sink.send_record_cmd, {}))
        await asyncio.sleep(0.02)
        for ev in (_key(_K_OPEN, 1), _key(_K_OPEN, 2), _key(_K_OPEN, 2),
                   _key(_K_OPEN, 0)):
            dev.feed(ev)
            await asyncio.sleep(0.03)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    _, errors = _drive(scenario)
    dev.dispose()
    assert sink.intents == ["open", "stop"], sink.intents
    assert not errors, errors


def test_gripper_keys_avoid_visers_camera_controls():
    """No gripper key may be one viser moves the camera with.

    The operator HOLDS these while watching the collection page, and this bridge reads
    /dev/input directly — the browser receives the very same keystroke and nothing here
    can swallow it. So a gripper key inside viser's CAMERA_MOVEMENT_KEYS (KeyW KeyA KeyS
    KeyD KeyQ KeyE) flies the view for as long as the jaw moves. That is exactly what made
    the old q/w RIGHT-hand pair unusable, so it is pinned rather than left to review.

    Also checks the routing still works and that the retired keys are inert, so a partial
    rename (constants moved, `on_key_hand` not) cannot pass."""
    import keyboard_gripper_ws as kb

    for hand, pair in kb._HAND_KEYS.items():
        for key in pair:
            assert key not in kb._VISER_CAMERA_KEYS, f"{hand} key {key} is a viser camera key"
    # the record/home keys must stay distinct from the gripper keys, or a grasp would
    # start/stop an episode
    flat = [k for pair in kb._HAND_KEYS.values() for k in pair]
    assert len(set(flat)) == 4, flat
    assert not set(flat) & {kb._REC_START, kb._REC_STOP, kb._HOME}, flat

    g = kb.GestureState()
    assert g.on_key_hand(kb._OPEN, True) == ("left", "open")
    assert g.on_key_hand(kb._OPEN, False) == ("left", "stop")
    assert g.on_key_hand(kb._CLOSE, True) == ("left", "close")
    assert g.on_key_hand(kb._OPEN_R, True) == ("right", "open")
    assert g.on_key_hand(kb._CLOSE_R, True) == ("right", "close")
    # retired pairs are inert: no jaw motion AND no record command
    for dead in ("KEY_O", "KEY_P", "KEY_Q", "KEY_W"):
        assert g.on_key_hand(dead, True) == ("", ""), dead
        assert g.take_record_cmd() is None, dead


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
