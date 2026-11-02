#!/usr/bin/env python3
"""Opt-in lossless F/T sample publisher (source-side ring -> localhost WS).

NOTE the name: this is the FULL-RATE lossless sink, NOT an "uncompensated" one.
It carries the RAW wrench always, and ALSO the gravity/inertia-COMPENSATED wrench
when a Bota payload config is active (``compensated=True``). "Raw" in the class
name means "every sample, unaliased", not "no compensation".

The Bota ``SensorReader`` overwrites ``state.force/torque`` in place every read
(~250 Hz) and only bumps a counter — there is NO history a poller can drain
without aliasing. To record the FULL-RATE wrench trace, the reader pushes every
sample into THIS object's bounded ring (a microsecond ``deque.append`` on the
reader thread — it does not perturb the estimator EKF), and a tiny WebSocket
server thread DRAINS the ring and broadcasts samples in BATCHES to the
out-of-process recorder. Batching (not one-message-per-sample) is what lets a
localhost socket keep up with 250 Hz.

Wire frame (text JSON, matches recording_format.FT_WS_PORT_DEFAULT):
    {"type": "ft_batch",
     "compensated": bool,                            # samples_comp present + meaningful?
     "samples":      [[t, Fx,Fy,Fz, Tx,Ty,Tz], ...], # RAW wrench (always)
     "samples_comp": [[t, Fx,Fy,Fz, Tx,Ty,Tz], ...], # COMPENSATED wrench (only if compensated)
     "ft_comp_algo": "bota"|"rnea",                   # which engine produced samples_comp (compensated only)
     "ft_sign_verified": bool}                        # active engine's global sign trusted on this hand

``compensated`` is set ONCE after construction (the viz knows whether a payload
config is active) and is NOT mutated by the recompensation path — within a
session it is fixed, so every batch agrees (the recorder relies on this; see the
F/T provenance banner in recording_format.py). The raw + comp rows in one batch
are 1:1 and share ``t``.

``t`` is wall-clock ``time.time()`` (the single rig clock), stamped at the sensor
read so it lines up with the tracker/gripper/camera streams the recorder also
timestamps on the same clock.

This object owns ONLY its ring + its own asyncio loop/thread; it never touches
the Bota HWI or the SensorReader's SensorState, so it does not cross the
"Bota HWI touched only on the reader thread" invariant.
"""
from __future__ import annotations

import collections
import threading
import time


class FtRawPublisher:
    """Bounded SPSC-ish ring + localhost WS broadcaster for raw F/T samples."""

    def __init__(self, host: str = "127.0.0.1", port: int = 8775,
                 maxlen: int = 4000, drain_hz: float = 50.0):
        self._host = host
        self._port = port
        self._lock = threading.Lock()
        # maxlen=4000 ~= 16 s at 250 Hz; the recorder drains at drain_hz so this
        # only fills if no recorder is connected (then oldest samples drop — fine,
        # nothing is recording them). Each ring row is (raw7, comp7|None) so the
        # raw + compensated batches stay 1:1 and share timestamps.
        self._ring: collections.deque = collections.deque(maxlen=maxlen)
        # Whether a Bota payload compensator is active this session. Set ONCE by
        # the viz right after construction (do NOT toggle mid-session — the
        # recorder latches it per take; see recording_format's F/T invariant).
        self.compensated: bool = False
        # Optional callable -> (tare_wrench|None, tare_wall|None): the compensator
        # DC offset + its wall-clock, attached to each frame for recorder
        # provenance. Set by the viz when compensated. None = omit from the frame.
        self.tare_getter = None
        # Optional callable -> (algo_name|None, sign_verified: bool): the LIVE
        # compensation-algorithm identity ("bota"/"rnea") + whether its global sign
        # is trusted on this hand, attached to each compensated frame so the recorder
        # can latch WHICH engine compensated each take (a dropdown switched mid-session
        # otherwise silently blends two compensation models). Set by the viz when
        # compensated. None (or a None name) = omit from the frame.
        self.algo_getter = None
        self._drain_period = 1.0 / drain_hz if drain_hz > 0 else 0.02
        self._loop = None
        self._stop = None
        self._thread: threading.Thread | None = None
        self._bound = threading.Event()
        self._start_error: Exception | None = None

    # -- producer side (called on the SensorReader thread) ------------------
    def push(self, t_wall: float, force, torque,
             comp_force=None, comp_torque=None) -> None:
        """Append one sample. MICROSECOND cost (deque.append under a lock); safe
        to call from the high-rate reader loop. ``force/torque`` is the RAW wrench
        (always). ``comp_force/comp_torque`` is the compensated wrench at the same
        timestamp; pass None on the raw-only loop (nothing to compensate)."""
        t = float(t_wall)
        raw = (t, float(force[0]), float(force[1]), float(force[2]),
               float(torque[0]), float(torque[1]), float(torque[2]))
        comp = None
        if comp_force is not None and comp_torque is not None:
            comp = (t, float(comp_force[0]), float(comp_force[1]), float(comp_force[2]),
                    float(comp_torque[0]), float(comp_torque[1]), float(comp_torque[2]))
        with self._lock:
            self._ring.append((raw, comp))

    def _drain(self) -> tuple:
        """Return (raw_rows, comp_rows). comp_rows is the compensated samples for
        the rows that had them (1:1 with raw_rows when self.compensated)."""
        with self._lock:
            if not self._ring:
                return [], []
            rows = list(self._ring)
            self._ring.clear()
        raw_rows = [r for (r, _c) in rows]
        comp_rows = [c for (_r, c) in rows if c is not None]
        return raw_rows, comp_rows

    # -- consumer side (own asyncio thread; never touches Bota) -------------
    def start(self) -> None:
        import asyncio

        import websockets

        def _run() -> None:
            loop = asyncio.new_event_loop()
            self._loop = loop
            asyncio.set_event_loop(loop)
            clients: set = set()

            async def handler(ws):
                clients.add(ws)
                try:
                    async for _ in ws:  # publisher is broadcast-only; drain input
                        pass
                except Exception:  # noqa: BLE001
                    pass
                finally:
                    clients.discard(ws)

            async def broadcast():
                self._stop = asyncio.Event()
                async with websockets.serve(handler, self._host, self._port, max_size=None):
                    print(f"[ft-pub] full-rate F/T WS at ws://{self._host}:{self._port} "
                          f"(compensated={self.compensated})", flush=True)
                    self._bound.set()
                    import json
                    while not self._stop.is_set():
                        raw_rows, comp_rows = self._drain()
                        if raw_rows and clients:
                            frame = {"type": "ft_batch",
                                     "compensated": bool(self.compensated),
                                     "samples": raw_rows}
                            # Only attach comp rows when this session is compensated
                            # (else the reader pushed None for every sample).
                            if self.compensated and comp_rows:
                                frame["samples_comp"] = comp_rows
                            if self.compensated and self.tare_getter is not None:
                                try:
                                    tw, twall = self.tare_getter()
                                    if tw is not None:
                                        frame["tare_wrench"] = list(tw)
                                        frame["tare_wall"] = twall
                                except Exception:  # noqa: BLE001 - provenance is advisory
                                    pass
                            if self.compensated and self.algo_getter is not None:
                                try:
                                    algo, sign_ok = self.algo_getter()
                                    if algo is not None:
                                        frame["ft_comp_algo"] = str(algo)
                                        frame["ft_sign_verified"] = bool(sign_ok)
                                except Exception:  # noqa: BLE001 - provenance is advisory
                                    pass
                            websockets.broadcast(clients, json.dumps(frame))
                        await asyncio.sleep(self._drain_period)

            try:
                loop.run_until_complete(broadcast())
            except Exception as exc:  # noqa: BLE001
                self._start_error = exc
                print(f"[ft-pub] stopped: {exc!r}", flush=True)
            finally:
                self._bound.set()
                loop.close()

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()
        if not self._bound.wait(timeout=5.0):
            raise RuntimeError(f"F/T publisher did not bind on {self._host}:{self._port}")
        if self._start_error is not None:
            raise self._start_error

    def stop(self) -> None:
        loop, ev = self._loop, self._stop
        if loop is not None and ev is not None:
            loop.call_soon_threadsafe(ev.set)
