#!/usr/bin/env python3
"""Regression tests for the camera-bridge PACING invariant.

Why this file exists
--------------------
The recorded camera timeline is the MASTER GRID of the training dataset
(`processing/convert_to_umi.py` takes `cam_t = cam_index[:, 0]` and resamples
every other modality onto it), and both training loaders index windows in FRAME
units with no dt input. A non-uniform camera dt therefore turns into
multiplicative noise on every learned velocity and force time-derivative.

The bridge originally throttled publishing with

    if (now - last_pub) < publish_period: continue
    last_pub = now

i.e. a wall-clock period test whose reference is reset to the ACTUAL publish
instant. With `--fps 30` (33.3439 ms measured on the D405, std 0.3 us) and
`--publish-hz 15` (66.6667 ms) the decision margin is +21 us while host arrival
jitter is 300-470 us -- 14-22x larger. The accept/reject decision was therefore
made by scheduler noise, and because the reference re-anchored on every publish
the phase random-walked. Result on disk: 97 takes at 11.90 Hz with dt alternating
between 2 and 3 capture periods (47.9% / 52.1%), +-20% rms per-step jitter.

These tests pin the replacement (count-based decimation on the SDK frame number)
AND pin the old logic as a known-bad reference, so the bug cannot come back
disguised as a "simplification".

Run:  .venv-vive/bin/python bridges/camera/test_pacing.py     (prints "PACING OK")
      (any interpreter with numpy works; no camera and no pyrealsense2 needed)
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from realsense_bridge import resolve_decimation  # noqa: E402

FPS = 30.0
CAPTURE_PERIOD = 1.0 / 29.99  # measured D405 sensor period, not the nominal 1/30


def _arrivals(n: int, rho: float = 0.8, jitter_s: float = 4.0e-4, seed: int = 0):
    """Synthetic host arrival times for `n` frames at FPS with AR(1) jitter.

    Real arrival jitter is positively autocorrelated (OS scheduling / USB
    batching); rho ~= 0.8 is what reproduces the measured 47.9/52.1 split.
    """
    rng = np.random.default_rng(seed)
    e = 0.0
    out = np.empty(n)
    t = 0.0
    for i in range(n):
        e = rho * e + np.sqrt(1 - rho * rho) * rng.standard_normal()
        t = i * CAPTURE_PERIOD + e * jitter_s
        out[i] = t
    return out


def _old_throttle(arrivals: np.ndarray, publish_hz: float):
    """The ORIGINAL logic, kept as a known-bad reference."""
    period = 1.0 / publish_hz
    last_pub = 0.0
    published = []
    for i, now in enumerate(arrivals):
        if period and (now - last_pub) < period:
            continue
        last_pub = now
        published.append(i)
    return np.asarray(published)


def _new_decimation(frame_numbers: np.ndarray, decim: int):
    """The REPLACEMENT: count-based on the SDK frame number."""
    base = frame_numbers[0]
    return frame_numbers[(frame_numbers - base) % decim == 0]


def test_resolve_decimation():
    assert resolve_decimation(30, 15.0) == 2
    assert resolve_decimation(30, 30.0) == 1
    assert resolve_decimation(30, 10.0) == 3
    assert resolve_decimation(30, 0) == 1
    assert resolve_decimation(90, 15.0) == 6
    for bad in (12.0, 11.9, 20.0, 7.0):
        try:
            resolve_decimation(30, bad)
        except ValueError as exc:
            assert "does not divide" in str(exc)
        else:
            raise AssertionError(
                f"resolve_decimation(30, {bad}) must refuse a non-integer decimation"
            )
    print("  resolve_decimation: OK")


def test_new_decimation_is_exactly_uniform():
    """Every published gap must be exactly `decim`, for every fps/publish combo."""
    for fps, pub in ((30, 15.0), (30, 10.0), (30, 30.0), (90, 15.0), (60, 20.0)):
        decim = resolve_decimation(fps, pub)
        fns = np.arange(1000, 1000 + 600)
        pub_fns = _new_decimation(fns, decim)
        gaps = np.unique(np.diff(pub_fns))
        assert gaps.tolist() == [decim], (
            f"fps={fps} pub={pub}: gaps {gaps.tolist()} != [{decim}]"
        )
        assert len(pub_fns) == len(fns) // decim
    print("  new decimation uniform for all fps/publish combos: OK")


def test_new_decimation_is_immune_to_arrival_jitter():
    """Count-based decimation cannot be perturbed by host timing at all."""
    fns = np.arange(0, 900)
    for seed in range(5):
        # jitter only affects *when* frames arrive, never their numbers
        _ = _arrivals(len(fns), seed=seed)
        gaps = np.unique(np.diff(_new_decimation(fns, 2)))
        assert gaps.tolist() == [2]
    print("  new decimation immune to arrival jitter: OK")


def test_sensor_drop_becomes_a_hole_not_a_phase_slide():
    """Keying on the SDK frame number is deliberate.

    A sensor-side drop must leave a gap of exactly 2*decim (detectable
    downstream) instead of silently shifting the publish phase, which is what a
    local counter would do.
    """
    fns = np.concatenate([np.arange(0, 20), np.arange(21, 40)])  # frame 20 dropped
    pub = _new_decimation(fns, 2)
    gaps = np.diff(pub)
    assert set(gaps.tolist()) <= {2, 4}, gaps.tolist()
    assert (gaps == 4).sum() == 1, f"expected exactly one 2*decim hole, got {gaps.tolist()}"
    # phase preserved: every published frame number is still even
    assert np.all((pub - pub[0]) % 2 == 0)

    # A LOCAL COUNTER (publish every 2nd frame we happen to RECEIVE) slides the
    # phase across the drop: published frames end up on BOTH parities of the
    # capture grid, so the publish instants stop being exact multiples of the
    # capture period. Frame-number keying keeps them all on one parity.
    local = fns[::2]
    assert len(set(((local - local[0]) % 2).tolist())) == 2, (
        "expected a local counter to slide the publish phase across the drop"
    )
    assert len(set(((pub - pub[0]) % 2).tolist())) == 1, (
        "frame-number keying must keep every published frame on one parity"
    )
    print("  sensor drop -> visible 2*decim hole (frame-number keyed), "
          "local counter would hide it: OK")


def test_old_throttle_is_reproducibly_non_uniform():
    """Regression pin on the ORIGINAL bug.

    Asserts the old logic produces MORE than one distinct gap and a rate well
    below the requested one -- i.e. that the defect we measured on disk is a
    property of that code, not of the hardware.
    """
    arrivals = _arrivals(6000, rho=0.8, seed=1)
    pub = _old_throttle(arrivals, 15.0)
    gaps = np.diff(pub)
    ks, counts = np.unique(gaps, return_counts=True)
    eff_hz = len(pub) / (arrivals[-1] - arrivals[0])
    assert len(ks) > 1, (
        "the old throttle produced a uniform grid in simulation -- the regression "
        "pin is broken"
    )
    assert 2 in ks and 3 in ks, f"expected the 2/3 alternation, got {ks.tolist()}"
    assert eff_hz < 13.0, f"expected the old throttle to undershoot 15 Hz, got {eff_hz:.2f}"
    frac = dict(zip(ks.tolist(), (counts / counts.sum()).round(3).tolist()))
    print(f"  old throttle reproduces the defect: {eff_hz:.2f} Hz, gaps {frac}: OK")


def test_new_beats_old_on_the_measured_metric():
    """The metric that actually matters: spread of a 16-frame window span."""
    arrivals = _arrivals(12000, rho=0.8, seed=2)
    old = arrivals[_old_throttle(arrivals, 15.0)]
    fns = np.arange(len(arrivals))
    new = arrivals[_new_decimation(fns, resolve_decimation(30, 15.0))]

    def spread(t, w=16):
        s = t[w:] - t[:-w]
        return (np.percentile(s, 95) - np.percentile(s, 5)) / np.median(s)

    def step_rms(t):
        d = np.diff(t)
        return float(np.sqrt(np.mean((d / d.mean() - 1) ** 2)))

    s_old, s_new = spread(old), spread(new)
    r_old, r_new = step_rms(old), step_rms(new)
    assert s_new < s_old / 5, f"window spread old {s_old:.3f} new {s_new:.3f}"
    assert r_new < r_old / 20, f"per-step rms old {r_old:.3f} new {r_new:.4f}"
    print(f"  16-frame window spread {s_old * 100:.1f}% -> {s_new * 100:.2f}%; "
          f"per-step rms {r_old * 100:.1f}% -> {r_new * 100:.3f}%: OK")




# --------------------------------------------------------------------------- #
# END-TO-END: run the REAL bridge (mock camera) and measure the REAL wire grid.
# The pure-logic tests above re-implement the decimation, so they pass even if
# `_stream_once` / `_run_mock` is reverted to the wall-clock throttle. This one
# executes the shipped code path.
# --------------------------------------------------------------------------- #


def test_end_to_end_mock_bridge_is_uniform():
    import json
    import subprocess

    try:
        import websockets  # noqa: F401
    except Exception:  # noqa: BLE001
        print("  e2e: SKIPPED (no websockets in this interpreter)")
        return

    here = os.path.dirname(os.path.abspath(__file__))
    bridge = os.path.join(here, "realsense_bridge.py")
    port = 8791
    proc = subprocess.Popen(
        [sys.executable, bridge, "--mock", "--port", str(port),
         "--fps", "30", "--publish-hz", "15", "--width", "64", "--height", "48",
         "--report-every", "1"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        import asyncio
        import time as _time

        async def collect():
            import websockets as ws
            for _ in range(50):
                try:
                    conn = await ws.connect(f"ws://127.0.0.1:{port}")
                    break
                except Exception:  # noqa: BLE001
                    await asyncio.sleep(0.2)
            else:
                raise RuntimeError("bridge never came up")
            ts, caps, seqs = [], [], []
            async with conn:
                t0 = _time.time()
                while _time.time() - t0 < 6.0:
                    m = json.loads(await conn.recv())
                    ts.append(m["t"])
                    caps.append(m["capture_seq"])
                    seqs.append(m["seq"])
            return ts, caps, seqs

        ts, caps, seqs = asyncio.run(collect())
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except Exception:  # noqa: BLE001
            proc.kill()

    assert len(ts) > 40, f"only {len(ts)} frames"
    dt = np.diff(ts)
    med = float(np.median(dt))
    spread = (np.percentile(dt, 95) - np.percentile(dt, 5)) / med
    cap_gaps = sorted(set(np.diff(caps).tolist()))
    seq_gaps = sorted(set(np.diff(seqs).tolist()))
    hz = len(dt) / dt.sum()
    assert cap_gaps == [2], f"capture_seq gaps {cap_gaps} != [2] -> decimation broken"
    assert seq_gaps == [1], f"wire seq gaps {seq_gaps} != [1] -> frames lost on the WS"
    assert spread < 0.05, f"wire dt spread {spread * 100:.2f}% >= 5%"
    assert abs(hz - 15.0) < 0.5, f"achieved {hz:.3f} Hz != 15"
    print(f"  e2e real bridge (mock camera): {hz:.3f} Hz, dt med {med * 1000:.2f} ms, "
          f"spread {spread * 100:.2f}%, capture gaps {cap_gaps}, seq gaps {seq_gaps}: OK")


def test_converter_gate():
    """The converter-side verdict, on the scenarios that matter."""
    sys.path.insert(
        0,
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))), "processing"),
    )
    import umi_convert_core as core

    rng = np.random.default_rng(0)
    # The MEASURED proportion on disk (47.9% at 2 capture periods, 52.1% at 3), so
    # the median lands on the 3-period value -- which is exactly what made the
    # naive round(dt/median) gate collapse to one bucket and report "uniform".
    bad = np.where(rng.random(3000) < 0.479, 2 / 29.99, 3 / 29.99)
    good = np.full(3000, 2 / 29.99) + rng.normal(0, 5e-4, 3000)

    assert core.camera_grid_verdict(bad)["camera_dt_uniform"] is False
    assert core.camera_grid_verdict(good)["camera_dt_uniform"] is True
    # the round(dt/median) degeneracy that a previous version of this gate had
    med = float(np.median(bad))
    assert len(set(np.round(bad / med).astype(int).tolist())) == 1, (
        "sanity: round(dt/median) really does collapse on the bad data"
    )
    # short episode -> None, not a false "uniform"
    assert core.camera_grid_verdict(np.full(5, 2 / 29.99))["camera_dt_uniform"] is None
    assert core.camera_grid_verdict(np.zeros(0))["camera_dt_uniform"] is None
    # recorder veto
    assert core.camera_grid_verdict(good, take_grid_uniform=False)["camera_dt_uniform"] is False
    # holes are reported even when the pacer is fine
    stall = np.concatenate([np.full(2999, 2 / 29.99), [2.0]])
    v = core.camera_grid_verdict(stall)
    assert v["camera_dt_uniform"] is True and v["camera_dt_holes_n"] == 1
    assert v["camera_dt_max_over_median"] > 20
    # 5% dropped frames -> fails
    assert core.camera_grid_verdict(
        np.where(rng.random(3000) < 0.05, 4 / 29.99, 2 / 29.99)
    )["camera_dt_uniform"] is False
    print("  converter gate (bad/good/short/veto/holes/5%-drop): OK")


def main() -> int:
    test_resolve_decimation()
    test_new_decimation_is_exactly_uniform()
    test_new_decimation_is_immune_to_arrival_jitter()
    test_sensor_drop_becomes_a_hole_not_a_phase_slide()
    test_old_throttle_is_reproducibly_non_uniform()
    test_new_beats_old_on_the_measured_metric()
    test_converter_gate()
    test_end_to_end_mock_bridge_is_uniform()
    print("PACING OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
