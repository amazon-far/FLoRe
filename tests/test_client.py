"""Tests for the FLoRe client layer: s0_driver.

These tests verify the WebSocket server broadcast. No GPU, no network (except
localhost WS).
"""

from __future__ import annotations

import json
import time

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# s0_driver WS server
# ---------------------------------------------------------------------------


def test_s0_driver_broadcasts():
    """S0Driver starts a WS server; a client connects and receives 3pt JSON."""
    import websockets.sync.client

    from flore.client.s0_driver import S0Driver

    driver = S0Driver(host="127.0.0.1", port=18770)  # non-default port for test isolation
    driver.FRESHNESS_TIMEOUT_S = 10.0  # keep broadcasting throughout the test (no stale-stop race)
    driver.start()
    time.sleep(0.2)

    pos = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], dtype=np.float32)
    orn = np.arange(12, dtype=np.float32) * 0.1

    with websockets.sync.client.connect("ws://127.0.0.1:18770") as ws:
        # Send AFTER the client connects so the payload is guaranteed fresh and
        # a broadcast lands on this connection (avoids a connect/send race under
        # concurrent CPU load). Generous recv timeout swamps the 5ms interval.
        time.sleep(0.1)
        driver.send(pos, orn)
        msg_str = ws.recv(timeout=3.0)
        msg = json.loads(msg_str)
        assert msg["mode"] == "3pt"
        assert len(msg["pos"]) == 9
        assert len(msg["orn"]) == 12
        np.testing.assert_allclose(msg["pos"], pos.tolist(), atol=1e-5)
        np.testing.assert_allclose(msg["orn"], orn.tolist(), atol=1e-5)

    driver.stop()


def test_s0_driver_freshness_timeout_stops_rebroadcast():
    """Safety: after FRESHNESS_TIMEOUT_S with no send(), the driver STOPS
    re-broadcasting so holosoma's stale gate can fire (a hung control loop must
    not freeze S0 on the last target while looking alive)."""
    import websockets.sync.client

    from flore.client.s0_driver import S0Driver

    driver = S0Driver(host="127.0.0.1", port=18771)
    driver.FRESHNESS_TIMEOUT_S = 0.3  # shortened for the test (default 0.5)
    driver.start()
    time.sleep(0.2)

    pos = np.ones(9, dtype=np.float32)
    orn = np.zeros(12, dtype=np.float32)

    with websockets.sync.client.connect("ws://127.0.0.1:18771") as ws:
        time.sleep(0.1)
        driver.send(pos, orn)
        # Fresh: at least one frame arrives promptly.
        msg = json.loads(ws.recv(timeout=2.0))
        assert msg["mode"] == "3pt"

        # Go silent well past the freshness window; broadcasts must STOP.
        # (Generous margins so the test is robust under concurrent CPU load —
        # the broadcast interval is 5ms, so a recv() timeout >> that is a
        # definitive "broadcaster is quiet" signal.)
        time.sleep(1.0)
        # Drain any in-flight frames. A recv timeout of 0.3s is 60x the 5ms
        # broadcast interval, so a timeout genuinely means the stream stopped
        # (not just a scheduling gap mid-stream).
        try:
            while True:
                ws.recv(timeout=0.3)
        except TimeoutError:
            pass
        # After draining, no NEW frame should arrive (broadcaster is quiet).
        with pytest.raises(TimeoutError):
            ws.recv(timeout=0.5)

    driver.stop()
