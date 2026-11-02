#!/usr/bin/env python
"""Live Bota payload COMPENSATION readout (step 2 of 2), terminal version.

Loads the payload estimated by estimate_payload.py (the compensator JSON) and
prints, in real time, BOTH the raw wrench and the COMPENSATED wrench - i.e. the
raw force/torque with the tool's gravity (and, with dynamic compensation, its
inertia) removed. The compensated wrench should hover near ZERO when nothing is
touching the tool - in ANY orientation - and jump only when you actually push on
it. That is the whole point vs. a one-shot tare.

Use it to sanity-check the estimate before wiring compensation into the viz:
  1. Hold the sensor+tool free (no contact). Compensated F/T should read ~0.
  2. Slowly rotate it through orientations - compensated F/T should STAY ~0
     (raw will swing with gravity; compensated shouldn't). This is what a tare
     can't do.
  3. Press a finger on the tool - the compensated force should jump to the
     contact force, then return to ~0 when you let go.

    .venv/bin/python force_torque/compensate_live.py
    .venv/bin/python force_torque/compensate_live.py --no-tare   # skip startup tare
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import time

import numpy as np
import bota_driver
import bota_payload_utils as bpu

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bota_compat import FATAL_TOLERANCE, IMU_OFFSET, FtChain, rc_is_fatal, warm_up_tare  # noqa: E402
from bota_ft_viz import find_bota_port, list_bota_ports, resolve_driver_config  # noqa: E402

# UMI_collection/force_torque/ -> repo root one level up; configs in robot_assets/bota_config/.
_BOTA_CFG = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "robot_assets", "bota_config")
)
_DRIVER_CFG = os.path.join(_BOTA_CFG, "bota_driver", "bota_binary.json")
_COMP_CFG = os.path.join(_BOTA_CFG, "bota_payload_utils", "payload_compensator.json")
_IMU_OFFSET = IMU_OFFSET

_stop = False


def _sig(_s, _f):
    global _stop
    _stop = True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port-path", default=None)
    ap.add_argument("--driver-config", default=_DRIVER_CFG)
    ap.add_argument("--compensator-config", default=_COMP_CFG)
    ap.add_argument("--no-tare", action="store_true", help="skip the startup tare of the compensated wrench")
    ap.add_argument("--rate-hz", type=float, default=5.0, help="terminal print rate")
    args = ap.parse_args()

    signal.signal(signal.SIGINT, _sig)

    port = args.port_path or find_bota_port()
    if port is None:
        print("ERROR: no Bota sensor found under /dev/serial/by-id/.")
        if list_bota_ports():
            print(f"  candidates: {list_bota_ports()}")
        return 1
    print(f"Using sensor: {port}")
    print(f"Payload config: {args.compensator_config}")
    drv_cfg = resolve_driver_config(args.driver_config, port)

    driver = bota_driver.BotaDriver(drv_cfg)
    comp = bpu.BotaPayloadCompensator(args.compensator_config)
    # gen-A: caller owns the IMU transform + 12-vector assembly (no HWI wrapper).
    chain = FtChain(_IMU_OFFSET)

    try:
        if not driver.configure():
            raise RuntimeError("driver.configure() failed")
        if not driver.activate():
            raise RuntimeError("driver.activate() failed")

        if not args.no_tare:
            # Tare zeroes any residual constant offset on top of the model-based
            # compensation. Do it with the tool free / not in contact. 1.0.2's tare
            # is ASYNC (collects a ~0.5 s window before the compensated output is
            # valid) - warm it up so the first printed numbers aren't startup zeros.
            if not comp.tare_compensated_wrench():
                print("WARNING: tare_compensated_wrench() returned False")
            if not warm_up_tare(driver, comp, chain):
                print("WARNING: compensated tare did not settle within timeout (showing raw output)")

        print("\n" + "=" * 78)
        print("  RAW vs COMPENSATED wrench. Compensated should stay ~0 when free,")
        print("  in ANY orientation, and jump only on real contact. Ctrl-C to stop.")
        print("=" * 78)

        loop_period = 1.0 / 1000.0
        next_t = time.time()
        last_print = 0.0
        print_period = 1.0 / args.rate_hz
        consec_fatal = 0
        while not _stop:
            frame = driver.read_frame_blocking()
            raw12 = chain.assemble(frame)
            rc = comp.update(time.perf_counter(), raw12, chain.comp12)
            if rc_is_fatal(rc):
                # Transient bad-dt fatal recovers next clean sample; bail only on a
                # sustained run.
                consec_fatal += 1
                if consec_fatal > FATAL_TOLERANCE:
                    print(f"compensator.update() FatalError x{consec_fatal} (stream broken)")
                    break
                continue
            consec_fatal = 0

            now = time.time()
            if now - last_print >= print_period:
                last_print = now
                # raw wrench = first 6 of the assembled input; compensated = first
                # 6 of the compensator output (both in the sensor frame).
                rf, rt = raw12[:3], raw12[3:6]
                cf, ct = chain.comp12[:3], chain.comp12[3:6]
                print(
                    f"RAW  F[{rf[0]:+6.2f} {rf[1]:+6.2f} {rf[2]:+6.2f}] "
                    f"T[{rt[0]:+5.2f} {rt[1]:+5.2f} {rt[2]:+5.2f}]   ||   "
                    f"COMP F[{cf[0]:+6.2f} {cf[1]:+6.2f} {cf[2]:+6.2f}] "
                    f"T[{ct[0]:+5.2f} {ct[1]:+5.2f} {ct[2]:+5.2f}]   |Fc|={np.linalg.norm(cf):4.2f}N"
                )

            next_t += loop_period
            dt = next_t - time.time()
            if dt > 0:
                time.sleep(dt)
            elif dt < -loop_period:
                next_t = time.time()

        driver.deactivate()
        driver.shutdown()
        print("\nStopped cleanly.")
        return 0

    except Exception as exc:  # noqa: BLE001
        try:
            driver.deactivate()
            driver.shutdown()
        except Exception:  # noqa: BLE001
            pass
        print(f"\nFATAL: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
