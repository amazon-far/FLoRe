#!/usr/bin/env python3
"""Pre-viz Bota F/T connectivity self-check (R2).

The startup counterpart to ``gripper/gripper_selftest.py`` for the force/torque
sensor: BEFORE the viser scene comes up, prove the Bota sensor is discoverable
and actually STREAMING a finite wrench, then FULLY release it so the in-process
viz ``SensorReader`` can claim it (the sensor is single-owner — only one reader
at a time). Prints a single ``[ft-selftest] RESULT ...`` line and exits 0 (PASS)
or non-zero (FAIL) so the launcher can gate on it like the gripper self-test.

Runs under the SAME py3.12 venv as the viz (``bota_driver`` + ``bota_ft_viz``),
distinct from ``bridges/preflight.py`` which is pure-stdlib and can only check the
by-id symlink EXISTS — it cannot open the sensor. This does.

Usage (from the launcher, under $BOTA_PY):
    python force_torque/ft_selfcheck.py [--timeout 6] [--config <bota_binary.json>]

Exit: 0 = a finite streaming wrench was seen and the sensor released cleanly;
      2 = no sensor found;  3 = found but no finite streaming sample in --timeout.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

# bota_ft_viz is this same directory; import it directly.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bota_ft_viz as bft  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--timeout", type=float, default=6.0,
                    help="seconds to wait for a finite streaming wrench (default 6)")
    ap.add_argument("--config", default=bft._DEFAULT_CONFIG,
                    help="Bota driver JSON config template")
    ap.add_argument("--serial", default=None,
                    help="PIN a specific sensor by its USB serial substring (e.g. "
                         "3875355D3532). MANDATORY once two sensors are attached: the "
                         "unpinned pick is the first SORTED /dev/serial/by-id match, and "
                         "3472… sorts before 3875… — so a bare run self-checks the RIGHT "
                         "hand's sensor and reports it as the left hand's.")
    ap.add_argument("--label", default="", help="hand tag for the log lines, e.g. 'L' / 'R'")
    args = ap.parse_args(argv if argv is not None else sys.argv[1:])

    tag = f"ft-selftest{'-' + args.label if args.label else ''}"
    port = bft.find_bota_port(args.serial)
    if port is None:
        pinned = f" matching serial {args.serial}" if args.serial else ""
        print(f"[{tag}] RESULT FAIL: no Bota F/T sensor found{pinned} (check the "
              "USB/EtherCAT connection + power).", flush=True)
        return 2
    print(f"[{tag}] Bota F/T sensor: {port}", flush=True)

    cfg = bft.resolve_driver_config(args.config, port)
    state = bft.SensorState()
    # RAW loop only (no compensator/estimator): we just need to confirm the stream
    # is alive and finite. Low read rate — this is a brief pre-flight, not the live
    # reader. No raw_sink (nothing records here).
    reader = bft.SensorReader(state, cfg, mock=False, read_hz=100.0)
    reader.start()

    deadline = time.monotonic() + args.timeout
    ok = False
    saw_connected = False
    try:
        while time.monotonic() < deadline:
            with state.lock:
                connected = state.connected
                n = state.n_frames
                force = state.force.copy()
                torque = state.torque.copy()
            saw_connected = saw_connected or connected
            # PASS when the reader reports connected, has advanced past the first
            # frame, and the wrench is finite (a dead/garbage stream reads NaN/inf
            # or never advances n_frames).
            if connected and n >= 2 and np.all(np.isfinite(force)) and np.all(np.isfinite(torque)):
                fmag = float(np.linalg.norm(force))
                tmag = float(np.linalg.norm(torque))
                print(f"[{tag}] RESULT PASS: streaming finite wrench "
                      f"(|F|={fmag:.2f}N |T|={tmag:.3f}Nm, {n} frames).", flush=True)
                ok = True
                break
            time.sleep(0.1)
    finally:
        # ALWAYS release the sensor so the in-process viz reader can claim it
        # (single-owner hardware). Join so the driver is fully shut down before we
        # return and the launcher spawns the viz.
        reader.stop()
        reader.join(timeout=3.0)

    if not ok:
        why = ("connected but no finite streaming sample" if saw_connected
               else "sensor found but never started streaming")
        print(f"[{tag}] RESULT FAIL: {why} within {args.timeout:.0f}s.", flush=True)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
