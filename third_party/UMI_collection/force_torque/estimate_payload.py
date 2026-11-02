#!/usr/bin/env python
"""Guided Bota payload ESTIMATION (step 1 of 2 for compensation).

Estimates the mass / center-of-mass / inertia of whatever tool is bolted to the
sensor, by running Bota's EKF (BotaPayloadEstimator) while YOU slowly rotate the
sensor through several orientations. On success it writes the result into the
compensator config so step 2 (compensate_live / the viz) can use it.

HOW TO USE:
  1. Make sure the tool is mounted but NOTHING is touching it (no contact!) -
     the sensor + tool must hang/move freely.
  2. Run this script. When it says "ROTATE NOW", pick the sensor up and slowly
     turn it through many orientations (roll it, pitch it, yaw it - get gravity
     pointing along +/-x, +/-y, +/-z at some point). Move smoothly, ~15-30 s.
  3. It stops automatically when the EKF converges (between min_time and
     max_time from the estimator config) and writes the payload to the
     compensator JSON.

The serial port is auto-detected (any Bota sensor). The committed driver config
is never modified (a temp copy is used).

    .venv/bin/python force_torque/estimate_payload.py
    .venv/bin/python force_torque/estimate_payload.py --max-time 60   # longer rotate window
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import sys
import time

import bota_driver
import bota_payload_utils as bpu

# Reuse the auto-detect + config-resolution helpers from the viz tool, plus the
# gen-A (>=1.0) toolkit compat shim (12-vector assembly + IMU TF + enum handling).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bota_compat import FATAL_TOLERANCE, IMU_OFFSET, FtChain, rc_is_fatal  # noqa: E402
from bota_ft_viz import find_bota_port, list_bota_ports, resolve_driver_config  # noqa: E402

# UMI_collection/force_torque/ -> repo root one level up; configs in robot_assets/bota_config/.
_BOTA_CFG = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "robot_assets", "bota_config")
)
_DRIVER_CFG = os.path.join(_BOTA_CFG, "bota_driver", "bota_binary.json")
_EST_CFG = os.path.join(_BOTA_CFG, "bota_payload_utils", "payload_estimator.json")
_COMP_CFG = os.path.join(_BOTA_CFG, "bota_payload_utils", "payload_compensator.json")

# IMU offset (wrench_frame -> imu_frame, meters) for the M8-series; matches the
# Bota demo. Sourced from bota_compat so all three F/T tools agree.
_IMU_OFFSET = IMU_OFFSET

_stop = False


def _sig(_signum, _frame):
    global _stop
    _stop = True


def _fmt_payload(comp_path: str) -> str:
    p = json.load(open(comp_path))["payload_compensator_config"]["payload_properties"]
    com = p["com"]
    return (
        f"  mass    = {p['mass']:.4f} kg\n"
        f"  com     = [{com['x']:+.4f}, {com['y']:+.4f}, {com['z']:+.4f}] m\n"
        f"  inertia = Ixx {p['inertia']['Ixx']:+.6f}  Iyy {p['inertia']['Iyy']:+.6f}  "
        f"Izz {p['inertia']['Izz']:+.6f}  (kg m^2)"
    )


def _force_dynamic_compensation(comp_path: str) -> None:
    """Force ``enable_dynamic_compensation = false`` in the just-written compensator
    config. The estimator HARDCODES this flag into the file it serializes, and NO
    config/API knob controls it. This rig runs GRAVITY-ONLY compensation by choice,
    so we pin the flag to ``false`` here regardless of what the estimator .so wrote
    (the installed bota_payload_utils 1.0.2 writes ``false``; the older 0.0.2 wrote
    ``true``). WITHOUT this guard a bpu 0.0.2 estimator — or a future downgrade —
    would silently RE-ENABLE inertia (dynamic) compensation. Log loudly when we flip
    it so a regression is visible. Best effort: a malformed file is left untouched
    (logged)."""
    try:
        cfg = json.load(open(comp_path))
        pc = cfg["payload_compensator_config"]
        if pc.get("enable_dynamic_compensation") is not False:
            print(f"[payload] WARNING: estimator wrote enable_dynamic_compensation="
                  f"{pc.get('enable_dynamic_compensation')!r}; forcing false "
                  f"(this rig runs gravity-only compensation by design).", flush=True)
        pc["enable_dynamic_compensation"] = False
        with open(comp_path, "w") as f:
            json.dump(cfg, f, indent=2)
    except (OSError, KeyError, ValueError) as exc:
        print(f"[payload] WARNING: could not force enable_dynamic_compensation in "
              f"{comp_path}: {exc!r}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port-path", default=None, help="serial device (default: auto-detect)")
    ap.add_argument("--driver-config", default=_DRIVER_CFG)
    ap.add_argument("--estimator-config", default=_EST_CFG)
    ap.add_argument("--out", default=_COMP_CFG, help="compensator JSON to write the result into")
    ap.add_argument("--max-time", type=float, default=None, help="override estimator max_time (s)")
    ap.add_argument("--min-time", type=float, default=None, help="override estimator min_time (s)")
    args = ap.parse_args()

    signal.signal(signal.SIGINT, _sig)

    port = args.port_path or find_bota_port()
    if port is None:
        print("ERROR: no Bota sensor found under /dev/serial/by-id/.")
        if list_bota_ports():
            print(f"  candidates: {list_bota_ports()}")
        return 1
    print(f"Using sensor: {port}")
    drv_cfg = resolve_driver_config(args.driver_config, port)

    out_dir = os.path.dirname(os.path.abspath(args.out))
    prev_cwd = os.getcwd()
    drv_cfg = os.path.abspath(drv_cfg)

    # The estimator .so writes its result to ``output_config_file_name``. Empirically
    # (bpu 1.0.2) it resolves that name relative to the directory of the ESTIMATOR
    # CONFIG, NOT the process CWD — so a temp config in /tmp made the library write
    # /tmp/payload_compensator.json while this script looked in out_dir, found the
    # STALE seed already there, and copied that (silently discarding the real
    # estimate). To be robust to whichever base the .so uses, we ALWAYS: (a) write
    # the (possibly time-overridden) estimator config INTO out_dir, and (b) force a
    # UNIQUE output name that cannot pre-exist, so the file we later read is provably
    # the one this run just wrote (verified by existence + mtime below).
    est_out_name = f".payload_estimate_{os.getpid()}.json"
    ecfg = json.load(open(args.estimator_config))
    ec = ecfg["payload_estimator_config"]
    if args.min_time is not None:
        ec["min_time"] = args.min_time
    if args.max_time is not None:
        ec["max_time"] = args.max_time
    if args.min_time is not None or args.max_time is not None:
        print(f"Estimator window: min_time={ec['min_time']}s  max_time={ec['max_time']}s")
    ec["output_config_file_name"] = est_out_name
    est_cfg_path = os.path.join(out_dir, f".payload_estimator_cfg_{os.getpid()}.json")
    with open(est_cfg_path, "w") as f:
        json.dump(ecfg, f)

    # Candidate locations the .so might write the unique-named result to. We chdir
    # into out_dir before constructing the estimator, so BOTH the config-dir base
    # and the CWD base point inside out_dir — a single path, but we keep the list
    # shape in case a future bpu resolves elsewhere. Snapshot the start time to
    # reject any pre-existing stale file.
    _t_start = time.time()
    _out_candidates = [os.path.join(out_dir, est_out_name)]
    for _c in _out_candidates:
        # A leftover from a crashed prior run would look "fresh enough"; remove it so
        # the mtime check below is unambiguous.
        try:
            if os.path.exists(_c):
                os.remove(_c)
        except OSError:
            pass

    # Back up the existing compensator config (the user's --out target) before we
    # overwrite it with the new estimate.
    if os.path.exists(args.out):
        backup = args.out + ".bak"
        shutil.copy(args.out, backup)
        print(f"Backed up previous compensator config -> {backup}")

    # chdir into out_dir so the CWD-relative branch also lands inside out_dir.
    os.chdir(out_dir)

    os.chdir(out_dir)
    driver = bota_driver.BotaDriver(drv_cfg)
    estimator = bpu.BotaPayloadEstimator(est_cfg_path)
    # gen-A: the caller owns the IMU transform + 12-vector assembly (no HWI wrapper).
    chain = FtChain(_IMU_OFFSET)

    try:
        if not driver.configure():
            raise RuntimeError("driver.configure() failed")
        if not driver.activate():
            raise RuntimeError("driver.activate() failed")

        if not estimator.start_payload_estimation():
            raise RuntimeError("start_payload_estimation() failed")

        print("\n" + "=" * 64)
        print("  >>> ROTATE NOW <<<  pick up the sensor+tool (NOTHING touching")
        print("  it) and slowly turn it through many orientations: roll, pitch,")
        print("  yaw, so gravity points along each axis at some point.")
        print("  Keep moving smoothly for ~15-30 s. It stops on its own.")
        print("=" * 64 + "\n")

        loop_period = 1.0 / 1000.0
        next_t = time.time()
        t0 = time.time()
        last_log = 0.0
        consec_fatal = 0
        while estimator.is_estimator_running() and not _stop:
            frame = driver.read_frame_blocking()
            rc = estimator.update(time.perf_counter(), chain.assemble(frame))
            if rc_is_fatal(rc):
                # A transient bad-dt fatal recovers on the next clean sample; only a
                # sustained run means the stream is genuinely broken.
                consec_fatal += 1
                if consec_fatal > FATAL_TOLERANCE:
                    raise RuntimeError(
                        f"estimator.update() returned FatalError for {consec_fatal} "
                        f"consecutive samples (stream broken)"
                    )
                continue
            consec_fatal = 0
            now = time.time()
            if now - last_log >= 1.0:
                last_log = now
                print(f"  estimating... {now - t0:4.1f}s elapsed (rotate through more poses)")
            next_t += loop_period
            dt = next_t - time.time()
            if dt > 0:
                time.sleep(dt)
            elif dt < -loop_period:
                next_t = time.time()

        if _stop:
            print("\nInterrupted before convergence - no result written.")
            return 1

        ok = estimator.is_estimation_successful()

        # Resolve the UNIQUE-named result file the estimator wrote THIS run. We
        # accept a candidate only if it exists AND was modified after we started
        # (mtime >= _t_start) — this is what makes it impossible to silently copy a
        # stale seed again (the /tmp bug). Cover both resolution bases.
        written = None
        if ok:
            for cand in _out_candidates:
                if os.path.exists(cand) and os.path.getmtime(cand) >= _t_start - 1.0:
                    written = cand
                    break

        os.chdir(prev_cwd)
        driver.deactivate()
        driver.shutdown()

        # Clean up the temp estimator config we wrote into out_dir.
        try:
            os.remove(est_cfg_path)
        except OSError:
            pass

        if not ok:
            print("\nEstimation did NOT converge (move through more varied poses, "
                  "keep it out of contact, and retry).")
            return 1

        if written is None:
            # The estimator claimed success but we cannot find a freshly-written
            # result file. Refuse to write --out rather than leave a stale value —
            # this is the failure that previously copied the seed silently.
            print("\nFATAL: estimation reported success but no freshly-written result "
                  f"file was found (looked for {est_out_name} in {out_dir} / CWD, "
                  f"mtime >= run start). NOT writing {args.out} to avoid persisting a "
                  "stale payload. Please retry.")
            return 1

        if os.path.abspath(written) != os.path.abspath(args.out):
            shutil.copy(written, args.out)
        # Remove the intermediate unique-named file now that --out holds it.
        try:
            if os.path.abspath(written) != os.path.abspath(args.out):
                os.remove(written)
        except OSError:
            pass

        # Belt-and-suspenders: the estimator .so bakes enable_dynamic_compensation
        # into the file it writes (the installed 1.0.2 writes false; 0.0.2 wrote
        # true). Force false so this rig always runs GRAVITY-ONLY compensation,
        # even if run against a bpu 0.0.2 estimator.
        _force_dynamic_compensation(args.out)

        print("\n" + "=" * 64)
        print("  ESTIMATION SUCCESS - payload written to:")
        print(f"  {args.out}")
        print("=" * 64)
        print(_fmt_payload(args.out))
        print("\nNext: run the live compensation -")
        print("  python compensate_live.py            (numbers in the terminal)")
        print("  or feed it into the viz (step 2).")
        return 0

    except Exception as exc:  # noqa: BLE001
        os.chdir(prev_cwd)
        try:
            driver.deactivate()
            driver.shutdown()
        except Exception:  # noqa: BLE001
            pass
        print(f"\nFATAL: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
