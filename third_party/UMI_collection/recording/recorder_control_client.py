#!/usr/bin/env python3
"""Thin client the viz GUI uses to drive the out-of-process recorder.

The recorder (``recording/recorder.py``) exposes a bidirectional control WS
(:8776): the viz sends ``{"cmd":"start"|"stop"|"marker"|"discard","task":...}``
and receives ``{"type":"status",...}`` frames. This client runs the WS on its own
asyncio thread (exactly like ``_WSReader`` in collection_viz.py) so GUI callbacks
stay non-blocking: a button only enqueues an intent; the latest status is cached
under a lock for the render loop to display.

Kept dependency-light (websockets only) and import-safe under .venv. If the
recorder process isn't up, sends are dropped and status stays "recorder: offline"
— recording is simply unavailable, the viz is otherwise unaffected.
"""
from __future__ import annotations

import json
import threading


class RecorderControlClient(threading.Thread):
    def __init__(self, url: str = "ws://127.0.0.1:8776"):
        super().__init__(daemon=True)
        self.url = url
        self._lock = threading.Lock()
        self._status: dict = {}
        self._connected = False
        self._loop = None
        self._ws = None
        self._stop = threading.Event()

    def stop(self) -> None:
        self._stop.set()

    @property
    def connected(self) -> bool:
        with self._lock:
            return self._connected

    def status(self) -> dict:
        with self._lock:
            return dict(self._status)

    def send(self, obj: dict) -> None:
        """Thread-safe best-effort send (no-op if recorder is down)."""
        import asyncio
        loop, ws = self._loop, self._ws
        if loop is None or ws is None:
            return
        try:
            asyncio.run_coroutine_threadsafe(ws.send(json.dumps(obj)), loop)
        except Exception:  # noqa: BLE001
            pass

    # convenience wrappers
    def start_episode(self, task: str = "") -> None:
        self.send({"cmd": "start", "task": task})

    def stop_episode(self) -> None:
        self.send({"cmd": "stop"})

    def mark_boundary(self) -> None:
        """Drop an episode-split marker into the continuous take (DECISION-E)."""
        self.send({"cmd": "marker"})

    def return_home(self) -> None:
        """Relay a 'c'-key return-to-initial-pose intent (R4). The recorder bumps
        home_seq in status; the viz observes it and resets the robot on its render
        thread. No-op if the recorder is down."""
        self.send({"cmd": "home"})

    def discard_episode(self) -> None:
        self.send({"cmd": "discard"})

    def run(self) -> None:
        import asyncio

        import websockets

        async def loop():
            while not self._stop.is_set():
                try:
                    async with websockets.connect(self.url, max_size=None) as ws:
                        self._ws = ws
                        with self._lock:
                            self._connected = True
                        async for raw in ws:
                            if self._stop.is_set():
                                break
                            try:
                                msg = json.loads(raw)
                            except (ValueError, TypeError):
                                continue
                            if msg.get("type") == "status":
                                with self._lock:
                                    self._status = msg
                except Exception:  # noqa: BLE001 - recorder down / not up yet
                    pass
                finally:
                    self._ws = None
                    with self._lock:
                        self._connected = False
                if not self._stop.is_set():
                    await asyncio.sleep(1.0)

        try:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            self._loop.run_until_complete(loop())
        except Exception:  # noqa: BLE001
            pass
