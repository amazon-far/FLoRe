#!/usr/bin/env python3
"""Global keyboard -> gripper WebSocket bridge (hold-to-move f/g + j/k).

The keyboard counterpart to the viser page's Open/Close/Stop buttons: it reads
the keyboard at the DEVICE level (``evdev``) and sends the SAME control intents
(``{"cmd": "open"|"close"|"stop"}``) to the headless ``keyboard_gripper
--web-control`` process over its localhost WebSocket (default ``:8774``). So you
can drive the gripper from the keyboard without:

  * a focused terminal — evdev reads the device globally, so it works no matter
    which window (the browser, an editor, nothing) has focus; and
  * the global ``xset`` autorepeat hack the old curses teleop used — evdev events
    carry ``value`` (0=release, 1=press, 2=autorepeat), so we detect a real HOLD
    from the press/release pair and simply IGNORE autorepeat. No more "one
    keypress emits three characters" bleeding into your whole X session.

Hold-to-move semantics (matches the gripper's momentary loop):

    HOLD g  ->  send {"cmd":"open"}   (LEFT gripper keeps OPENING while held)
    HOLD f  ->  send {"cmd":"close"}  (LEFT gripper keeps CLOSING while held)
    release ->  send {"cmd":"stop"}   (freeze where it is)

TWO HANDS (2pt/3pt): the RIGHT gripper gets its OWN key pair on its OWN connection:

    HOLD k / HOLD j  ->  the same open/close intents to --gripper-right-url

The pairs are INDEPENDENT, not mirrored. Mirroring the left pair onto both hands would
close the idle hand on every grasp, and the recorded ``robot1_gripper_width`` would then
carry a real-but-task-irrelevant signal a policy learns to reproduce. With no
--gripper-right-url the j/k keys are inert and this bridge behaves exactly as before.

The four keys must stay OUT of viser's camera-movement set (w/a/s/d/q/e) — the operator
holds them while watching the collection page, and a held camera key flies the view for as
long as the jaw moves. That is why this is f/g + j/k and not the older o/p + q/w; see the
key-map block below, and ``--key-left-open`` &co if a fixed-code macropad forces a change.

The gripper process stays the SOLE CAN writer — this bridge only emits JSON
intents (identical to the web buttons), exactly like the rest of the hub-and-
spoke stack. It never touches CAN.

Device access: ``evdev`` needs read access to ``/dev/input/event*``. Either run
this under ``sudo`` or add yourself to the ``input`` group
(``sudo usermod -aG input $USER`` then re-login). With no ``--device`` it grabs
every keyboard carrying a configured hand's pair (f/g or j/k), so any keyboard works.

Run (under .venv, which has evdev + websockets):
    .venv/bin/python gripper/keyboard_gripper_ws.py --gripper-url ws://127.0.0.1:8774
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time

# ---------------------------------------------------------------------------
# Key map. MUST avoid viser's camera-movement keys.
# ---------------------------------------------------------------------------
# The operator watches the collection page in a browser while pressing these, and viser
# binds SIX keys to camera movement (viser/client/src/CameraControls.tsx,
# CAMERA_MOVEMENT_KEYS): KeyW KeyA KeyS KeyD KeyQ KeyE. Any gripper key drawn from that
# set silently flies the camera on every grasp, because this bridge reads /dev/input
# directly while the browser ALSO gets the keystroke — nothing here can swallow it.
# So the gripper keys are chosen OUTSIDE that set, and `_assert_viser_safe` below fails
# fast if a future edit (or a --key-* override) puts one back in.
#
# f/g and j/k: the two index fingers' home-row keys, so the left hand's pair sits under
# the left hand and the right hand's under the right hand. Both on every keyboard, so the
# no---device auto-grab still finds a node for each hand.
#
# WITHIN each pair the direction is OPERATOR PREFERENCE, and the two hands are deliberately
# NOT the same way round: LEFT is f close / g open (the index-finger HOME key closes), RIGHT
# is j OPEN / k CLOSE (the home key opens) — swapped 2026-08-30 at the operator's request,
# from the j close / k open both-hands-alike map. Do not "restore" the symmetry; the map is
# muscle memory, and the startup line + --help print it from the constants below so the two
# never disagree.
#
# Overridable (--key-left-open etc.) for the case a 2-key macropad emits fixed codes that
# cannot be reprogrammed: the pad decides, and then the map has to follow the hardware.
_OPEN = "KEY_G"
_CLOSE = "KEY_F"
# RIGHT-hand gripper (2pt/3pt): its OWN key pair, deliberately NOT a mirror of the left's.
# Mirroring would close the idle hand on every grasp, and `robot1_gripper_width` would
# then record a real-but-task-irrelevant signal that a policy learns to reproduce —
# worse than no right gripper at all. Inert unless --gripper-right-url is set.
_OPEN_R = "KEY_J"
_CLOSE_R = "KEY_K"
#: hand -> (open key, close key). The single source of truth for the routing table.
_HAND_KEYS = {"left": (_OPEN, _CLOSE), "right": (_OPEN_R, _CLOSE_R)}
# Recording control keys (manual per-episode start/stop) + 'c' return-to-home.
# Deliberately DISTINCT from the gripper keys so the hold-to-move is never disturbed.
# KEY_S=31, KEY_E=18, KEY_C=46. All three go to the recorder control WS as
# {"cmd":"start"|"stop"|"home"}; 'home' is relayed to the viz (return to initial
# pose, R4) — the gripper bridge stays the sole CAN writer and never sees them.
#
# NOTE s and e ARE in viser's CAMERA_MOVEMENT_KEYS, so a press also nudges the camera
# (s = back, e = up) while the page has focus. Left as-is on purpose: these are the
# recording protocol's keys, documented and in the operator's muscle memory, and one tap
# is a single small jog rather than the continuous fly a HELD gripper key produces. They
# are excluded from _assert_viser_safe for that reason — remap them deliberately (and
# update the runbooks) if the jog ever matters.
_REC_START = "KEY_S"
_REC_STOP = "KEY_E"
_HOME = "KEY_C"

#: viser's keyboard camera-movement set, as evdev key names. Mirrors
#: ``viser/client/src/CameraControls.tsx::CAMERA_MOVEMENT_KEYS`` = {KeyW, KeyA, KeyS,
#: KeyD, KeyQ, KeyE}. A HELD gripper key from this set flies the camera for as long as
#: the jaw moves, which is what made q/w unusable.
_VISER_CAMERA_KEYS = frozenset({"KEY_W", "KEY_A", "KEY_S", "KEY_D", "KEY_Q", "KEY_E"})


def _assert_viser_safe(hand_keys: dict[str, tuple[str, str]]) -> None:
    """Refuse a gripper key that viser drives the camera with.

    Checked at startup rather than trusted, because the failure is silent and only shows
    up as "the view keeps drifting while I grasp" — hard to attribute to a key map. The
    recording keys (s/e/c) are deliberately NOT checked; see the note above."""
    bad = [(hand, k) for hand, pair in hand_keys.items() for k in pair
           if k in _VISER_CAMERA_KEYS]
    if bad:
        pretty = ", ".join(f"{hand}:{k[4:].lower()}" for hand, k in bad)
        raise SystemExit(
            f"[kbd] ERROR: gripper key(s) collide with viser's camera controls: {pretty}\n"
            f"[kbd]        viser moves the camera on w/a/s/d/q/e, so holding one of those to\n"
            f"[kbd]        work a jaw also flies the view. Pick keys outside that set.")


class GestureState:
    """Translate the macropad keys into gripper intents + recording-control events.

    Two independent concerns share one state object (one physical pad, possibly
    several evdev nodes):

      * GRIPPER hold-to-move (unchanged): HOLD g -> 'open', HOLD f -> 'close',
        release -> 'stop'. Autorepeat is dropped upstream in ``_pump``.
      * RECORDING control (manual per-episode model): a single PRESS of ``s``
        starts a new episode, a single PRESS of ``e`` stops + saves it. One
        Start..Stop span is exactly ONE episode (the offline converter, given no
        markers, emits the whole take as a single episode) — so there is no
        long-press marker gesture any more; segmentation is the start/stop pair.
      * HOME control: a single PRESS of ``c`` requests a return to the initial
        pose (R4), relayed via the recorder to the viz (which owns the IK reset).

    ``on_key`` returns a GRIPPER intent ('open'/'close'/'stop'/'') for f/g; the s/e
    record keys return '' for the gripper and instead surface a record COMMAND via
    :meth:`take_record_cmd` (consumed once by the bridge), so the two paths never
    cross — the s/e keys never move the jaws, and f/g never touch the recorder.

    All updates run on the single asyncio loop thread, so no lock is needed. Time
    is injectable (``now``) only for symmetry with the old API / tests.
    """

    def __init__(self, longpress_s: float = 1.5):
        # longpress_s kept for signature compatibility; unused now (no chord gesture).
        self.longpress_s = longpress_s
        self.o_down = False
        self.p_down = False
        # RIGHT-hand hold state (j/k). Separate flags, because the two hands' holds are
        # independent — a shared pair of flags is how one hand's release would stop the
        # other's jaws mid-grasp.
        self.o_down_r = False
        self.p_down_r = False
        # Edge-detect the record keys so a HOLD (or autorepeat that slips through)
        # fires the command only ONCE per physical press.
        self._s_down = False
        self._e_down = False
        self._c_down = False
        self._pending_cmd: str | None = None   # "start" | "stop" | "home", consumed by the bridge

    def on_key(self, keyname: str, pressed: bool, now: float | None = None) -> str:
        """LEFT-hand view of :meth:`on_key_hand` — the historical signature.

        Returns the intent for the LEFT (primary) gripper only, and '' for anything
        else (record keys, home, the right-hand pair). Kept because it is what the
        existing tests and any single-hand caller use."""
        hand, intent = self.on_key_hand(keyname, pressed, now)
        return intent if hand == "left" else ""

    def on_key_hand(self, keyname: str, pressed: bool,
                    now: float | None = None) -> "tuple[str, str]":
        """Update key state; return ``(hand, intent)`` where hand is 'left'|'right'|''.

        intent is 'open'/'close' on a hold, 'stop' on release, and '' for a record /
        home / unknown key (those latch a command retrievable via
        :meth:`take_record_cmd` instead). The hand is what routes the intent to the
        right gripper bridge — each hand has its OWN key pair (see _HAND_KEYS), so one
        hand's hold can never move the other's jaws."""
        if keyname == _OPEN:
            self.o_down = pressed
            return "left", ("open" if pressed else "stop")
        if keyname == _CLOSE:
            self.p_down = pressed
            return "left", ("close" if pressed else "stop")
        if keyname == _OPEN_R:
            self.o_down_r = pressed
            return "right", ("open" if pressed else "stop")
        if keyname == _CLOSE_R:
            self.p_down_r = pressed
            return "right", ("close" if pressed else "stop")
        if keyname == _REC_START:
            # Fire on the press EDGE only (ignore release + any repeat while held).
            if pressed and not self._s_down:
                self._pending_cmd = "start"
            self._s_down = pressed
            return "", ""
        if keyname == _REC_STOP:
            if pressed and not self._e_down:
                self._pending_cmd = "stop"
            self._e_down = pressed
            return "", ""
        if keyname == _HOME:
            # 'c' -> return to initial pose (edge-triggered, once per press).
            if pressed and not self._c_down:
                self._pending_cmd = "home"
            self._c_down = pressed
            return "", ""
        return "", ""

    def take_record_cmd(self) -> str | None:
        """Return + clear a pending record command ('start'/'stop'), else None.
        Consumed once by the device pump so each press maps to exactly one command."""
        cmd = self._pending_cmd
        self._pending_cmd = None
        return cmd

    def reset(self) -> None:
        """Clear all key state. Called when a keyboard pump ends (device unplugged
        mid-hold): otherwise the down-flags stay latched on the SHARED state and a
        later real press couldn't edge-detect. Cheap and idempotent."""
        self.o_down = False
        self.p_down = False
        self.o_down_r = False
        self.p_down_r = False
        self._s_down = False
        self._e_down = False
        self._c_down = False
        self._pending_cmd = None


def _find_keyboards(explicit: list[str] | None):
    """Return InputDevice handles for the keyboard(s) to listen on.

    With ``explicit`` paths, open exactly those. Otherwise auto-pick every device
    exposing a COMPLETE hand's key pair — o+p (left) or q+w (right) — i.e. real
    keyboards, not mice / FIDO keys that only emit a couple of codes.

    Requiring o+p specifically would make a minimal macropad that reports ONLY its two
    right-hand keys invisible to the bridge: it has no left pair to match on, so the right
    gripper would sit inert with no error anywhere. Either pair qualifies a device; both
    hands' keys are then routed from whatever node sends them (see _HAND_KEYS). Today's
    FQ-K002 pads report a full 163-key descriptor and match either way."""
    import evdev
    from evdev import ecodes

    if explicit:
        return [evdev.InputDevice(p) for p in explicit]

    found = []
    for path in evdev.list_devices():
        try:
            d = evdev.InputDevice(path)
        except Exception:  # noqa: BLE001 - permission / transient
            continue
        keys = d.capabilities().get(ecodes.EV_KEY, [])
        # Derived from _HAND_KEYS, not hardcoded: a --key-* override has to change WHICH
        # devices are grabbed too, or the bridge would listen on a node that cannot send
        # the configured keys.
        def _has(pair):
            return all(getattr(ecodes, k) in keys for k in pair)
        if any(_has(pair) for pair in _HAND_KEYS.values()):
            found.append(d)
    return found


def _open_device(path: str):
    """Open ONE known /dev/input node (evdev imported lazily, as in _find_keyboards)."""
    import evdev

    return evdev.InputDevice(path)


def _release_device(dev) -> None:
    """Close ``dev`` and UNREGISTER its epoll reader from the asyncio loop.

    Load-bearing on every exit path of :func:`_pump`, cancellation included, and
    the fix for the traceback storm the collection console printed after a long
    idle pause:

        Exception in callback ReadIterator.__anext__.<locals>.next_batch_ready(...)
          .../evdev/eventio_async.py:48 in next_batch_ready -> future.set_result(...)
        asyncio.exceptions.InvalidStateError: invalid state
        During handling of the above exception, another exception occurred:
          .../evdev/eventio_async.py:50 in next_batch_ready -> future.set_exception(e)
        asyncio.exceptions.InvalidStateError: invalid state

    evdev's ``async_read_loop()`` is NOT cancel-safe. While a pump is parked on it,
    evdev has (a) a reader registered on the loop via ``loop.add_reader(fd, ...)``
    and (b) a done-callback bound to the future that ``__anext__`` just handed us.
    Cancelling the task cancels only that future — the reader and the callback stay
    live, and the closure keeps this device object (and its fd) alive. So when the
    node next becomes readable — the operator's very next keypress — evdev calls
    ``set_result()`` on the cancelled future, gets ``InvalidStateError``, and its own
    handler re-raises the SAME error out of ``set_exception()``, uncaught, inside an
    asyncio callback. One traceback per recycled node, per recycle. The user-visible
    trigger is therefore the 30 s silent-device watchdog below, which is why it only
    ever showed up after a wait.

    ``InputDevice.close()`` does ``loop.remove_reader(fd)`` then closes the fd, so
    the callback can never fire. It is idempotent (guarded on ``fd > -1``), hence
    safe to call from a handler AND from the ``finally``."""
    try:
        dev.close()
    except Exception:  # noqa: BLE001 - already closed / no loop / fd gone: nothing to undo
        pass


def make_intent_sender(queue):
    """Return the coroutine the pumps call to ship a gripper intent.

    It enqueues EVERY intent. Do NOT put a "skip it if it equals the last one" guard
    back in. One used to live here, to absorb duplicates from a device with several
    evdev nodes, and it can only ever SUBTRACT a press the operator made:

      * the guard is process-global, shared by every attached keyboard — and
        auto-detect grabs EVERY keyboard with a hand's key pair (10 nodes on the rig, the
        operator's main keyboard among them). Consecutive identical intents
        therefore do arise ACROSS devices, and dropping one is not harmless: tap
        `o` on a second keyboard during a pad hold and its "open" is swallowed on
        the way in while its release still sends `stop` — the gripper freezes with
        the pad key still held, and the pad's own release is swallowed too, so it
        takes another press to get moving. (Within ONE node it never fires: the
        kernel collapses a repeated key-down, so down/up strictly alternate.)
      * and any `stop` that never made it out — a WS drop, a bridge restart, an
        evdev recycle — leaves the guard one intent behind, so it eats the next
        identical press.

    There is no upside to weigh against that: the intents are idempotent on the
    gripper side (`open` -> ``target_frac`` 1.0, `close` -> 0.0, `stop` -> freeze
    where you are), so re-sending one is a no-op. Note the gripper LATCHES that
    target and never expires it (``get_target_frac`` has no timeout;
    ``--hold-timeout`` belongs to the curses loop, not web-control), so
    hold-to-move works only because the release sends `stop` — which is exactly why
    an intent must never be dropped on a guess.

    ``queue`` may be a single asyncio.Queue (single-hand, historical) or a
    ``{hand: Queue}`` mapping. With a mapping, an intent for a hand that has no queue
    (no ``--gripper-right-url`` given) is DROPPED — the key is inert rather than being
    silently rerouted to the other hand's jaws."""

    async def send_intent(cmd: str, hand: str = "left") -> None:
        if isinstance(queue, dict):
            q = queue.get(hand)
            if q is None:
                return  # that hand has no bridge configured; the key does nothing
            await q.put(cmd)
            return
        await queue.put(cmd)

    return send_intent


async def _drain(ws) -> None:
    """Consume + DISCARD everything a server broadcasts on an outbound client conn.

    The bridge connects to the gripper (:8774, 30 Hz status) and recorder (:8776,
    5 Hz status) only to SEND intents — it has no use for their status frames. But
    a websockets client that never recv()s lets its incoming Assembler fill past
    ``max_queue`` (default 16); the library then ``transport.pause_reading()``, after
    which it can't read the server's keepalive PING and the connection is closed with
    1011 'keepalive ping timeout' (~40 s). The ensuing ~1 s reconnect drops input and
    is the root of the intermittent multi-second key→action lag. Draining keeps reads
    flowing so PONGs are sent and the connection never times out. Returns when the
    connection closes (the sender's reconnect loop then re-establishes it)."""
    try:
        async for _ in ws:
            pass  # discard: we never act on server status, only emit intents
    except Exception:  # noqa: BLE001 - connection closed; the sender loop reconnects
        pass


async def _pump(dev, send_intent, gesture, send_record_cmd, last_event=None) -> None:
    """Read one device's event stream and translate keys to intents/commands.

    ``value`` semantics from the Linux input layer: 1=key down (press),
    0=key up (release), 2=autorepeat. We act ONLY on press/release and DROP
    autorepeat (value==2) — that is what makes hold-to-move work without the
    ``xset`` autorepeat hack and without the triple-character side effect.

    Key state is funnelled through the SHARED ``gesture`` (a GestureState):
      * f/g and j/k -> gripper hold-to-move intents (sent via ``send_intent``);
      * s/e -> recording start/stop COMMANDS (sent via ``send_record_cmd``),
        edge-detected so a single press maps to exactly one command.

    Per-device errors are CONTAINED here: if this keyboard is unplugged mid-run
    the kernel raises ``OSError(ENODEV)`` from the read; we log it and return so
    ONLY this pump ends. The other keyboards + the WS sender keep running. On
    release we emit a final 'stop' so the gripper can't be left latched.

    ``last_event`` (optional dict path->monotonic ts) is the liveness heartbeat
    the supervisor's watchdog reads: we stamp it on EVERY event (before the
    key-type filter, so any traffic counts as alive). A USB device that
    silently autosuspends stalls this ``async_read_loop`` WITHOUT raising — so
    the pump neither ends nor gets re-opened; the watchdog uses this stamp to
    detect and recycle such a dead-but-held node (the 2-key macropad symptom).

    The device is ALWAYS released on the way out (see :func:`_release_device`) —
    including on cancellation, which is how the watchdog recycles a node. Skipping
    that leaks the fd and leaves evdev's loop reader armed on a cancelled future,
    which is the ``InvalidStateError`` traceback storm."""
    from evdev import categorize, ecodes

    name = getattr(dev, "name", "?")
    try:
        async for event in dev.async_read_loop():
            if last_event is not None:
                last_event[dev.path] = time.monotonic()
            if event.type != ecodes.EV_KEY:
                continue
            key = categorize(event)
            keyname = key.keycode if isinstance(key.keycode, str) else (key.keycode[0] if key.keycode else "")
            if keyname not in (_OPEN, _CLOSE, _OPEN_R, _CLOSE_R, _REC_START, _REC_STOP, _HOME):
                continue
            if event.value not in (0, 1):  # value==2 (autorepeat) -> ignore
                continue
            hand, intent = gesture.on_key_hand(keyname, pressed=(event.value == 1))
            # A record key may have latched a start/stop command this event.
            cmd = gesture.take_record_cmd()
            # TRACE every key we act on, always on. An intermittent "I pressed o and
            # the gripper did nothing" is otherwise undiagnosable: this line is the
            # one place that says whether the press reached the bridge at all, which
            # keyboard sent it (auto-detect attaches EVERY matching keyboard, so a stray
            # 'o' typed in another window shows up here too), and what went out on
            # the wire. Pair it with the gripper's own `target_frac=` line to pin the
            # hop that dropped it. Wall-clock so it can be matched against a take's
            # timestamp and the operator's "it broke around …". Volume is human-scale
            # — the gripper + record keys only, and autorepeat was already dropped above.
            print(f"[kbd-gripper] {time.strftime('%H:%M:%S')} {name!r} {keyname} "
                  f"{'down' if event.value == 1 else 'up'} -> "
                  f"{(f'{hand}:{intent}' if intent else '') or cmd or '(nothing)'}",
                  flush=True)
            if intent:
                await send_intent(intent, hand)
            if cmd:
                await send_record_cmd(cmd)
    except OSError as exc:  # device unplugged / read error -> drop just this pump
        _release_device(dev)  # release before the await below, not just in `finally`
        print(f"[kbd-gripper] device {name!r} dropped ({exc}); other keyboards still active", flush=True)
        gesture.reset()            # clear any half-held key so it can't latch (#6)
        # BOTH hands: a device dropped mid-hold could have been holding either
        # pair, and a latched target never expires on the gripper side.
        for _h in _HAND_KEYS:
            await send_intent("stop", _h)  # don't leave a gripper latched mid-hold
    except Exception as exc:  # noqa: BLE001 - keep the rest of the bridge alive
        _release_device(dev)  # ditto: the node may still be readable while we await
        print(f"[kbd-gripper] device {name!r} pump error: {exc!r}", flush=True)
        gesture.reset()
        for _h in _HAND_KEYS:
            await send_intent("stop", _h)
    finally:
        _release_device(dev)  # idempotent; covers the normal-return path too


async def _recycle_stale_pump(path, pumps, last_event, mk_pump, *, open_device=None) -> bool:
    """Re-open a silent evdev node with a FRESH fd, then retire its stale pump.

    Called by the watchdog in :func:`_run`. Returns True if the node was re-attached.

    The ORDER is the whole point — open first, cancel second:

      * NO GAP. The rescan in ``_run`` runs every 2 s, so handing the re-open off to
        it leaves up to 2 s with nobody reading this keyboard. And the watchdog can
        only fire after a silence, so the keypress at risk is precisely the one that
        ENDS the operator's pause — the ``s`` that starts the next take. Overlapping
        the two fds instead is safe: each evdev fd gets its OWN copy of the event
        stream, and a duplicate press is already absorbed by the shared GestureState
        edge-detect plus ``send_intent``'s coalescing (auto-detect routinely runs >1
        node of one pad through one gesture, so double delivery is a supported case).
      * The rescan is AUTO-MODE ONLY. With an explicit ``--device`` a recycled pump
        was never re-attached at all, so a pinned keyboard that idled 30 s went dead
        for the rest of the session.

    Cancelling the stale pump is what releases its fd and unregisters its asyncio
    reader (see :func:`_release_device`); leaving that reader armed on a cancelled
    future is the ``InvalidStateError`` traceback storm.

    ``open_device`` is injectable for the tests; production passes None -> evdev."""
    stale = pumps.pop(path, None)
    last_event.pop(path, None)
    opener = open_device or _open_device
    attached = False
    try:
        fresh = await asyncio.get_running_loop().run_in_executor(None, opener, path)
        pumps[path] = mk_pump(fresh)
        attached = True
    except Exception as exc:  # noqa: BLE001 - the supervisor must survive a bad node
        # Usually OSError: the node really went away (unplugged during the pause).
        print(f"[kbd-gripper] could not re-open {path!r} ({exc!r}); "
              f"the rescan will pick it up if it returns", flush=True)
    finally:
        # ALWAYS retire the stale pump, and only after the re-open above (that
        # ordering is the no-gap property — see the docstring).
        if stale is not None:
            stale.cancel()
    return attached


async def _run(args) -> int:
    import websockets

    devices = _find_keyboards(args.device or None)
    if not devices:
        # A USB macropad can enumerate a few SECONDS after this bridge starts (the
        # launcher spawns everything at once) — exiting here is exactly how the
        # keyboard silently ended up dead while the rest of the stack ran. So in
        # auto-detect mode we DON'T quit: fall through with zero pumps and let the
        # re-enumeration loop below attach the keyboard the moment it appears. The
        # sender task is the anchor that keeps the process alive meanwhile. With an
        # explicit --device (a fixed path the operator chose) we keep the old hard
        # error, and --no-wait-for-keyboard restores the strict exit for scripts.
        msg = ("[kbd-gripper] no keyboard with the gripper keys found yet. "
               "In the 'input' group? (sudo usermod -aG input $USER, then re-login), else run under sudo.")
        if args.device or not args.wait_for_keyboard:
            print(msg + " Exiting (no --wait-for-keyboard).", file=sys.stderr)
            return 1
        print(msg, file=sys.stderr)
        print("[kbd-gripper] waiting for a keyboard to appear (auto-attach on plug-in)…", flush=True)
    else:
        print(f"[kbd-gripper] listening on: {', '.join(repr(d.name) for d in devices)}", flush=True)
    print(f"[kbd-gripper] HOLD {_OPEN[4:].lower()} = open / HOLD {_CLOSE[4:].lower()} = close"
          f" / release = stop  (Ctrl-C to quit)", flush=True)
    if args.recorder_url:
        print(f"[kbd-gripper] recording: s = START episode / e = STOP+save / c = return to home "
              f"-> {args.recorder_url}", flush=True)

    # A single shared WS connection, reconnecting if the gripper bridge restarts.
    # Intents are latched into an asyncio.Queue by the device pumps and shipped by
    # the sender, so a momentary disconnect never blocks the input loop.
    # One queue + one sender task PER HAND. Separate connections, because they are
    # separate bridge processes on separate CAN buses: a right-hand bridge that is down
    # (or absent) must not block or reorder the left hand's intents.
    hand_urls = {"left": args.gripper_url}
    if getattr(args, "gripper_right_url", ""):
        hand_urls["right"] = args.gripper_right_url
        print(f"[kbd-gripper] RIGHT gripper on {args.gripper_right_url} "
              f"— keys {_OPEN_R[4:].lower()}=open / {_CLOSE_R[4:].lower()}=close", flush=True)
    else:
        print(f"[kbd-gripper] no RIGHT gripper (--gripper-right-url unset) — "
              f"{_OPEN_R[4:].lower()}/{_CLOSE_R[4:].lower()} keys are inert", flush=True)
    queues: dict[str, asyncio.Queue] = {h: asyncio.Queue() for h in hand_urls}
    send_intent = make_intent_sender(queues)

    async def sender(hand: str = "left") -> None:
        url = hand_urls[hand]
        queue = queues[hand]
        while True:
            try:
                async with websockets.connect(url, max_size=None) as ws:
                    print(f"[kbd-gripper] connected to {hand} gripper {url}", flush=True)
                    # DRAIN the inbound side. The gripper server BROADCASTS its encoder
                    # status to every client at 30 Hz; if we never recv() it, websockets'
                    # incoming Assembler fills past max_queue (default 16, ~0.5 s) and
                    # calls transport.pause_reading() — after which we can no longer read
                    # the server's keepalive PING, so the connection is force-closed with
                    # 1011 'keepalive ping timeout' (~ping_interval+ping_timeout = ~40 s).
                    # The reconnect backoff below then drops input for ~1 s, and any key
                    # pressed in that window fires SECONDS late (the reported lag). We
                    # don't need the status frames — just discard them so reads never
                    # pause and PONGs keep flowing. (Verified: NODRAIN -> ~10 s key
                    # latency + periodic 1011 close; DRAIN -> <1 ms, 0 closes.)
                    drain_task = asyncio.create_task(_drain(ws))
                    try:
                        # A periodic heartbeat (src="kbd") keeps the gripper bridge —
                        # and through it the viz "Keyboard" indicator — marking this
                        # bridge online even when no key is pressed. The intents below
                        # carry the same src tag so they double as liveness pings. Drain
                        # the queue with a 1 s timeout so the heartbeat fires on idle.
                        while True:
                            try:
                                cmd = await asyncio.wait_for(queue.get(), timeout=1.0)
                                await ws.send(json.dumps({"cmd": cmd, "src": "kbd"}))
                            except asyncio.TimeoutError:
                                await ws.send(json.dumps({"cmd": "ping", "src": "kbd"}))
                    finally:
                        drain_task.cancel()
            except Exception:  # noqa: BLE001 - gripper bridge down / not up yet
                await asyncio.sleep(1.0)

    # Recording start/stop commands go to the recorder process's control WS (NOT
    # the gripper) — the gripper stays the sole CAN writer and never learns about
    # recording. Best-effort: a separate reconnecting connection; if the recorder
    # is down the command is dropped (the GUI Start/Stop buttons remain a backup).
    rec_queue: asyncio.Queue = asyncio.Queue()

    async def send_record_cmd(cmd: str) -> None:
        # cmd is "start"/"stop" (one manual episode per Start..Stop) or "home"
        # (return to initial pose, relayed by the recorder to the viz).
        if args.recorder_url:
            await rec_queue.put(cmd)

    async def recorder_sender() -> None:
        if not args.recorder_url:
            return
        while True:
            try:
                async with websockets.connect(args.recorder_url, max_size=None) as ws:
                    print(f"[kbd-gripper] connected to recorder {args.recorder_url}", flush=True)
                    # DRAIN: the recorder server broadcasts a status frame to every
                    # client at 5 Hz; same keepalive-timeout-then-reconnect trap as
                    # sender() above if we never recv() it. This connection drives the
                    # s/e record keys, so an undrained stall here is exactly the
                    # "press s/e, nothing for seconds" symptom. Discard the frames.
                    drain_task = asyncio.create_task(_drain(ws))
                    try:
                        while True:
                            cmd = await rec_queue.get()
                            await ws.send(json.dumps({"cmd": cmd}))
                            print(f"[kbd-gripper] sent record {cmd!r}", flush=True)
                    finally:
                        drain_task.cancel()
            except Exception:  # noqa: BLE001 - recorder down / not up yet
                await asyncio.sleep(1.0)

    # ONE shared gesture state across all keyboards (the pad is a single device,
    # but auto-detect may surface >1 node; sharing makes the record-key edge
    # detection coherent across nodes).
    gesture = GestureState()

    # Per-device liveness heartbeat (path -> monotonic ts of the last event),
    # read by the watchdog in the supervisor loop. Seeded at attach time so a
    # legitimately-idle keyboard isn't treated as stale.
    last_event: dict = {}

    def _mk_pump(d):
        last_event[d.path] = time.monotonic()  # seed: attached == alive right now
        return asyncio.create_task(_pump(d, send_intent, gesture, send_record_cmd, last_event))

    # The sender is the anchor task: it never returns (reconnect loop), so the
    # bridge stays alive even if every keyboard pump drops (each pump returns on
    # its device's unplug, see _pump). A periodic re-scan re-attaches keyboards
    # that come back (or appear) without restarting the process.
    # One sender task per configured hand. sender_tasks['left'] is the historical
    # single connection; a right-hand task exists only when its URL was given. The
    # liveness check and the cancel path below both iterate the dict, so there is no
    # separate handle for the left one.
    sender_tasks = {h: asyncio.create_task(sender(h)) for h in hand_urls}
    rec_sender_task = asyncio.create_task(recorder_sender())
    pumps: dict = {d.path: _mk_pump(d) for d in devices}
    explicit = bool(args.device)
    # Watchdog: recycle a pump whose device has delivered NO events for this long.
    # A USB device that autosuspends (or any silent stall) leaves async_read_loop
    # parked WITHOUT raising, so the pump is neither done() (not reaped below) nor
    # re-opened by the rescan (its path is still in `pumps`) — it goes dead-but-held
    # until a physical re-plug. The recycle re-opens the SAME node with a fresh fd
    # (re-arms the epoll reader and, on many HID devices, resumes the USB endpoint)
    # and only THEN cancels the stale pump, so no keypress falls in a gap. The udev rule
    # (85-fq-macropad.rules) removes the autosuspend TRIGGER for the known pad; this
    # is the pure BACKUP for any silent stall (BT sleep, cable glitch) that keeps
    # the same evdev path.
    #
    # TWO deliberate design choices (an earlier version got both wrong):
    #  1. The recycle is TRANSPARENT: it does NOT touch the SHARED gesture state or
    #     emit an intent. That state is process-global across all keyboards, so
    #     reset()+'stop' here would freeze an in-progress hold on a DIFFERENT
    #     keyboard. It's also
    #     unnecessary: a real hold emits autorepeat (value==2) which stamps
    #     last_event every ~30 ms, so a genuinely-held device can NEVER go stale;
    #     a device that IS stale is by definition sending nothing, so it has no
    #     key "down" to clear. We just re-open its node and leave the gripper alone.
    #  2. STALE_S is LARGE (30 s). evdev keyboards emit NO idle heartbeat, so
    #     "no events" is indistinguishable from "healthy but idle" — the operator
    #     legitimately pauses between keypresses, and a whole take (12-40 s of tracker
    #     motion between the 's' and the 'e' press) sends nothing at all. So a
    #     false fire is NOT rare, it is routine, and the fix is to make it harmless
    #     rather than to chase a threshold that cannot be right: the recycle is
    #     transparent (no gripper disturbance) and gapless (re-open before cancel).
    #     Getting the gapless part wrong is what printed the InvalidStateError
    #     tracebacks on the console — see _release_device.
    STALE_S = 30.0
    try:
        while True:
            # Reap finished pumps (a dropped/unplugged device).
            for path in [p for p, t in pumps.items() if t.done()]:
                pumps.pop(path, None)
                last_event.pop(path, None)
            # Watchdog: force-recycle a pump that's gone silent (dead-but-held).
            # TRANSPARENT — do not touch the shared gesture state (see note above).
            now_mono = time.monotonic()
            for path in [p for p, t in pumps.items()
                         if not t.done() and now_mono - last_event.get(p, now_mono) > STALE_S]:
                print(f"[kbd-gripper] device {path!r} silent >{STALE_S:.0f}s "
                      f"(autosuspend/stall?) — re-opening it with a fresh fd", flush=True)
                await _recycle_stale_pump(path, pumps, last_event, _mk_pump)
            # Re-enumerate (auto mode only): pick up keyboards that (re)appeared.
            # Run the scan in a THREAD, not on the event loop: _find_keyboards opens
            # every /dev/input/event* node and reads its capabilities, which on a busy
            # host with many input nodes measures ~0.4-0.6 s (and can stall for seconds
            # on a flaky USB/BT device). Done synchronously here it freezes the shared
            # loop every 2 s — injecting a few-hundred-ms tail onto whatever key lands
            # in the scan window. Offloading keeps key handling responsive.
            if not explicit:
                found = await asyncio.get_running_loop().run_in_executor(
                    None, _find_keyboards, None)
                for d in found:
                    if d.path not in pumps:
                        print(f"[kbd-gripper] (re)attached {d.name!r}", flush=True)
                        pumps[d.path] = _mk_pump(d)
            if any(t.done() for t in sender_tasks.values()):  # only ends on a fatal error
                break
            await asyncio.sleep(2.0)
    except asyncio.CancelledError:
        pass
    finally:
        for t in [*sender_tasks.values(), rec_sender_task, *pumps.values()]:
            t.cancel()
    return 0


def _apply_key_overrides(args) -> None:
    """Rebind the module-level gripper key map from --key-* and validate it.

    Module globals rather than plumbed state because every consumer (GestureState.on_key,
    the _pump filter, _find_keyboards) already reads these names, and a single map keeps
    "which key does what" answerable in one place. Rejects an unknown evdev name and any
    key viser drives the camera with, both BEFORE a device is opened, so a bad override is
    a startup error and not a session of mysterious view drift."""
    global _OPEN, _CLOSE, _OPEN_R, _CLOSE_R, _HAND_KEYS
    from evdev import ecodes

    wanted = {"left": (args.key_left_open, args.key_left_close),
              "right": (args.key_right_open, args.key_right_close)}
    for hand, pair in wanted.items():
        for k in pair:
            if not hasattr(ecodes, k):
                raise SystemExit(f"[kbd] ERROR: {hand} key {k!r} is not an evdev key name "
                                 f"(want e.g. KEY_F, KEY_LEFTBRACE)")
    flat = [k for pair in wanted.values() for k in pair]
    if len(set(flat)) != len(flat):
        raise SystemExit(f"[kbd] ERROR: the four gripper keys must be distinct, got {flat}")
    clash = sorted(set(flat) & {_REC_START, _REC_STOP, _HOME})
    if clash:
        raise SystemExit(f"[kbd] ERROR: gripper key(s) {clash} are the record/home keys "
                         f"(s=start, e=stop, c=home) — pick others")
    _assert_viser_safe(wanted)
    _OPEN, _CLOSE = wanted["left"]
    _OPEN_R, _CLOSE_R = wanted["right"]
    _HAND_KEYS = {"left": (_OPEN, _CLOSE), "right": (_OPEN_R, _CLOSE_R)}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--gripper-url", default="ws://127.0.0.1:8774",
                   help="LEFT (primary) headless keyboard_gripper --web-control WebSocket (default :8774)")
    p.add_argument("--gripper-right-url", default="",
                   help=f"RIGHT-hand gripper bridge WebSocket (e.g. ws://127.0.0.1:8778). "
                        f"Empty (default) = single-hand: the "
                        f"{_OPEN_R[4:].lower()}/{_CLOSE_R[4:].lower()} keys are inert and "
                        f"nothing extra is connected, so a 1pt session is unchanged.")
    p.add_argument("--device", action="append", default=[],
                   help=f"explicit /dev/input/eventN to listen on (repeatable). Default: "
                   f"auto-detect every keyboard carrying a configured hand's key pair "
                   f"({_OPEN[4:].lower()}/{_CLOSE[4:].lower()} or "
                   f"{_OPEN_R[4:].lower()}/{_CLOSE_R[4:].lower()}).")
    # Key overrides. Only needed when a 2-key macropad emits codes that cannot be
    # reprogrammed — then the hardware decides and the map must follow it. Values are
    # evdev key NAMES (KEY_F, KEY_LEFTBRACE, ...); validated against viser's camera set
    # so an override cannot silently reintroduce the drifting-view bug.
    for _flag, _dest, _default, _what in (
            ("--key-left-open", "key_left_open", _OPEN, "LEFT gripper OPEN"),
            ("--key-left-close", "key_left_close", _CLOSE, "LEFT gripper CLOSE"),
            ("--key-right-open", "key_right_open", _OPEN_R, "RIGHT gripper OPEN"),
            ("--key-right-close", "key_right_close", _CLOSE_R, "RIGHT gripper CLOSE")):
        p.add_argument(_flag, dest=_dest, default=_default, metavar="KEY_X",
                       help=f"evdev key name for {_what} (default {_default})")
    p.add_argument("--recorder-url", default="",
                   help="recorder control WS for the s=start / e=stop per-episode record keys "
                   "(e.g. ws://127.0.0.1:8776). Empty = record keys disabled.")
    p.add_argument("--longpress", type=float, default=1.5,
                   help="(deprecated, unused) kept for launcher backward-compat; the "
                   "long-press o+p marker gesture was replaced by s/e start/stop keys")
    # Default ON: don't exit when no keyboard is present at startup — wait and
    # auto-attach when one appears (a USB macropad often enumerates a few seconds
    # after the launcher spawns this bridge). --no-wait-for-keyboard restores the
    # strict "exit 1 if none found" behavior for scripted/CI use.
    p.add_argument("--wait-for-keyboard", dest="wait_for_keyboard", action="store_true", default=True,
                   help="(default) keep running and auto-attach when a keyboard appears, instead of exiting")
    p.add_argument("--no-wait-for-keyboard", dest="wait_for_keyboard", action="store_false",
                   help="exit with an error if no gripper keyboard is found at startup (old behavior)")
    args = p.parse_args(argv if argv is not None else sys.argv[1:])
    _apply_key_overrides(args)
    try:
        return asyncio.run(_run(args))
    except KeyboardInterrupt:
        print("\n[kbd-gripper] bye", file=sys.stderr)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
