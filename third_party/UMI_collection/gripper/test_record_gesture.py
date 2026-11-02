#!/usr/bin/env python
"""Logic tests for the manual per-episode record keys (s=start / e=stop).

No evdev / hardware needed. The manual record model replaced the old long-press
o+p episode-split marker gesture: one Start..Stop span is exactly ONE episode, so
s/e edge-detect a single press into a recorder "start"/"stop" command while o/p
stay pure gripper hold-to-move keys.

Run:
    .venv/bin/python gripper/test_record_gesture.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from keyboard_gripper_ws import GestureState, _OPEN, _CLOSE, _REC_START, _REC_STOP  # noqa: E402


def test_single_key_is_normal_hold_to_move():
    g = GestureState()
    assert g.on_key(_OPEN, True, now=0.0) == "open"
    assert g.on_key(_OPEN, False, now=0.1) == "stop"
    assert g.on_key(_CLOSE, True, now=0.2) == "close"
    assert g.on_key(_CLOSE, False, now=0.3) == "stop"


def test_record_keys_emit_no_gripper_intent():
    # s/e must NEVER move the jaws — on_key returns '' for them.
    g = GestureState()
    assert g.on_key(_REC_START, True) == ""
    assert g.on_key(_REC_START, False) == ""
    assert g.on_key(_REC_STOP, True) == ""
    assert g.on_key(_REC_STOP, False) == ""


def test_start_key_press_latches_start_cmd_once():
    g = GestureState()
    g.on_key(_REC_START, True)
    assert g.take_record_cmd() == "start"
    # Consumed -> no second command from the same press.
    assert g.take_record_cmd() is None


def test_stop_key_press_latches_stop_cmd_once():
    g = GestureState()
    g.on_key(_REC_STOP, True)
    assert g.take_record_cmd() == "stop"
    assert g.take_record_cmd() is None


def test_hold_or_autorepeat_does_not_refire():
    # A HOLD (press, then a stray repeat that slips through as another press)
    # must fire the command on the press EDGE only — not once per repeat.
    g = GestureState()
    g.on_key(_REC_START, True)
    assert g.take_record_cmd() == "start"
    g.on_key(_REC_START, True)          # simulate a duplicate "down" while held
    assert g.take_record_cmd() is None  # no refire while still held


def test_release_then_repress_arms_a_new_cmd():
    g = GestureState()
    g.on_key(_REC_START, True)
    assert g.take_record_cmd() == "start"
    g.on_key(_REC_START, False)         # release
    assert g.take_record_cmd() is None
    g.on_key(_REC_START, True)          # fresh press -> new start
    assert g.take_record_cmd() == "start"


def test_start_then_stop_sequence():
    # The normal one-episode flow: press s, then press e.
    g = GestureState()
    g.on_key(_REC_START, True)
    assert g.take_record_cmd() == "start"
    g.on_key(_REC_START, False)
    g.on_key(_REC_STOP, True)
    assert g.take_record_cmd() == "stop"
    g.on_key(_REC_STOP, False)
    assert g.take_record_cmd() is None


def test_gripper_and_record_keys_are_independent():
    # Holding o (gripper open) and pressing s (start) at once: the gripper intent
    # comes from on_key, the record cmd from take_record_cmd — they don't collide.
    g = GestureState()
    assert g.on_key(_OPEN, True) == "open"
    assert g.take_record_cmd() is None        # o is not a record key
    assert g.on_key(_REC_START, True) == ""   # s emits no gripper intent
    assert g.take_record_cmd() == "start"
    # o is still held -> releasing it still stops the gripper.
    assert g.on_key(_OPEN, False) == "stop"


def test_reset_clears_pending_cmd_and_key_state_on_unplug():
    # Device unplugged mid-press -> _pump calls gesture.reset(); a pending record
    # command and the down-flags must clear so nothing fires spuriously and a later
    # real press can edge-detect again.
    g = GestureState()
    g.on_key(_REC_START, True)            # latches a pending "start"
    g.reset()                             # simulate the unplug handler
    assert g.take_record_cmd() is None    # pending cmd cleared
    assert not g._s_down and not g._e_down and not g.o_down and not g.p_down
    # A fresh real press still works after reset.
    g.on_key(_REC_STOP, True)
    assert g.take_record_cmd() == "stop"


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
