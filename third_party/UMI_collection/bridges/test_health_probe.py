#!/usr/bin/env python3
"""health_probe.py: the PER-ROLE tracker gate, against a real WebSocket server.

``--require-keys trackers:left,right[,head]`` is what stops a multi-keypoint session
recording an empty tracker file. The tracker wire format is ``{"trackers": {role: pose}}``
with a role simply ABSENT until its puck locks, so "frames are flowing" says nothing about
whether the MODE's roles are all live: a 3pt run with the chest puck powered off streams
left+right forever, passes a bare probe, and the operator discovers the empty
``tracker_head.npy`` at conversion time.

Run:  PYTHONPATH=bridges .venv/bin/python bridges/test_health_probe.py
"""
from __future__ import annotations

import asyncio
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from health_probe import _probe  # noqa: E402

_CHECKS = 0


def check(cond, msg: str) -> None:
    global _CHECKS
    _CHECKS += 1
    if not cond:
        raise AssertionError(msg)
    print(f"  ok  {msg}")


async def _serve(roles, port: int):
    """A stand-in tracker bridge broadcasting only ``roles``, like a partial lock."""
    import websockets

    async def handler(ws):
        try:
            while True:
                await ws.send(json.dumps({"mode": "3pt", "t": 0.0,
                                          "trackers": {r: {"pos": [0, 0, 0]} for r in roles}}))
                await asyncio.sleep(0.02)
        except Exception:  # noqa: BLE001 - the probe disconnects as soon as it decides
            return

    return await websockets.serve(handler, "127.0.0.1", port)


async def _rc(roles, require_keys, port: int, timeout: float = 1.2) -> int:
    srv = await _serve(roles, port)
    try:
        return await _probe(f"ws://127.0.0.1:{port}", timeout, None, require_keys)
    finally:
        srv.close()
        await srv.wait_closed()


async def main() -> int:
    port = 8891
    # (live roles, required roles, expected rc, label)
    cases = [
        (["left", "right", "head"], "trackers:left,right,head", 0,
         "MODE=3pt with all three pucks locked -> healthy"),
        (["left", "right"], "trackers:left,right", 0,
         "MODE=2pt with both hands locked -> healthy (today's rig)"),
        (["left", "right"], "trackers:left,right,head", 1,
         "MODE=3pt with the CHEST puck off -> DEGRADED, not a false pass"),
        (["left"], "trackers:left,right", 1,
         "MODE=2pt with the RIGHT puck off -> DEGRADED"),
        (["left"], "trackers:left", 0,
         "MODE=1pt is unaffected (one role, one puck)"),
    ]
    for roles, req, want, label in cases:
        rc = await _rc(roles, req, port)
        port += 1
        check(rc == want, f"{label}  (rc={rc})")

    # A bare probe (no --require-keys) must behave EXACTLY as before: any frame passes,
    # even a partial role map. This is the regression guard on every other probe call
    # site in the launcher (camera, gripper, F/T).
    rc = await _rc(["left"], None, port)
    check(rc == 0, "a bare probe still passes on a partial role map (unchanged behaviour)")
    port += 1

    # An absent sub-dict must not crash — it reads as "every key missing".
    srv = await _serve([], port)
    try:
        rc = await _probe(f"ws://127.0.0.1:{port}", 1.2, None, "nosuchdict:left")
    finally:
        srv.close()
        await srv.wait_closed()
    check(rc == 1, "requiring keys of an ABSENT sub-dict is a clean timeout, not a crash")

    print(f"\nLOGIC OK ({_CHECKS} checks)")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
