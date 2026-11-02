#!/usr/bin/env python3
"""Active WebSocket health probe for the data-collection bridges.

Connect to a bridge's localhost WS, confirm frames actually FLOW within a
timeout (vs the launcher's old ``kill -0``, which a stuck ``sudo`` wrapper could
pass while its python child was dead), and optionally require a named field to be
truthy. Used by ``run_collection_viz.sh`` after it spawns the bridges:

    health_probe.py --url ws://127.0.0.1:8772 --timeout 6
    health_probe.py --url ws://127.0.0.1:8774 --timeout 6 --require kbd_online
    health_probe.py --url ws://127.0.0.1:8772 --timeout 8 --require-keys trackers:left,right

The gripper WS broadcasts a snapshot (incl. ``kbd_online``) to any client at
~30 Hz WITHOUT the client sending anything, so this probe is read-only — it never
opens CAN and never sends an intent (the gripper stays the sole CAN writer). The
same is true of the tracker/camera bridges (broadcast-only).

``--require-keys DICT:k1,k2`` requires a named sub-dict to contain ALL of the listed
keys. That exists for the tracker bridge, whose wire format is ``{"trackers": {role:
pose}}`` with a role simply ABSENT until its puck has produced a first pose. "A frame
arrived" therefore says nothing about whether the mode's roles are all live: a 3pt
session with the chest puck powered off streams left+right forever and passes a bare
probe, so the take records an EMPTY tracker_head.npy and the operator finds out at
conversion time. On a partial match the probe prints ``MISSING <dict>: <keys>`` to stdout
so the caller can name them.

Exit codes (the launcher captures them with ``|| rc=$?`` so a failed probe never
trips ``set -e``):
    0  healthy   — a frame arrived (and --require / --require-keys were satisfied)
    1  connected but no qualifying frame within --timeout
    2  could not connect within --timeout (bridge not up yet / refused)
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys


async def _probe(url: str, timeout: float, require: str | None,
                 require_keys: str | None = None) -> int:
    import websockets

    want_dict, want_keys = "", []
    if require_keys:
        want_dict, _, keys = require_keys.partition(":")
        want_keys = [k.strip() for k in keys.split(",") if k.strip()]
    # Best partial match seen, so a timeout can report exactly which keys never showed
    # up rather than a bare "no qualifying frame".
    best_missing: list[str] | None = None

    # Connect with its own timeout so a refused/not-yet-up bridge fails fast as
    # "could not connect" (rc=2) rather than masquerading as "no frame" (rc=1).
    try:
        ws = await asyncio.wait_for(
            websockets.connect(url, open_timeout=timeout, max_size=None), timeout=timeout)
    except Exception:  # noqa: BLE001 - not up yet / refused / handshake failed
        return 2

    # Connected: now wait (within the budget) for a qualifying frame. The timeout is caught
    # INSIDE `async with ws` so the socket closes normally (code 1000): leaving that block on
    # an exception closes it with 1011 "internal error", and a bridge polled every 5 s by
    # the launcher would log a "connection handler failed" traceback per poll.
    timed_out = False
    try:
        async with ws:
            try:
                async with asyncio.timeout(timeout):
                    while True:
                        raw = await ws.recv()
                        if require is None and not want_keys:
                            return 0
                        try:
                            msg = json.loads(raw)
                        except Exception:  # noqa: BLE001 - non-JSON frame, keep reading
                            continue
                        if not isinstance(msg, dict):
                            continue
                        if want_keys:
                            have = msg.get(want_dict) or {}
                            missing = [k for k in want_keys if k not in have]
                            if best_missing is None or len(missing) < len(best_missing):
                                best_missing = missing
                            if missing:
                                continue  # keep reading: roles appear as their pucks lock
                        if require is not None and not bool(msg.get(require)):
                            continue
                        return 0
            except asyncio.TimeoutError:
                timed_out = True
    except Exception:  # noqa: BLE001 - socket dropped mid-read
        return 1
    if timed_out and best_missing:
        print(f"MISSING {want_dict}: {','.join(best_missing)}")
    return 1  # connected, but no qualifying frame in time


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--url", required=True, help="bridge WebSocket URL")
    p.add_argument("--timeout", type=float, default=6.0, help="seconds to wait for a qualifying frame")
    p.add_argument("--require", default=None,
                   help="require this top-level JSON field to be truthy (e.g. kbd_online)")
    p.add_argument("--require-keys", default=None, metavar="DICT:k1,k2",
                   help="require the named sub-dict to contain ALL these keys "
                        "(e.g. trackers:left,right,head). Prints 'MISSING dict: keys' on "
                        "timeout so the caller can report which ones never appeared.")
    args = p.parse_args(argv if argv is not None else sys.argv[1:])
    try:
        return asyncio.run(_probe(args.url, args.timeout, args.require, args.require_keys))
    except KeyboardInterrupt:
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
