#!/usr/bin/env python3
"""Measure the DM4310 gripper's OPEN / CLOSED encoder endpoints by hand.

WHY: the DM4310 uses an ABSOLUTE encoder that RE-REFERENCES on every power loss,
so after a motor power-cycle the open/closed rad values drift and the launcher's
defaults (open=4.90 closed=0.0) no longer match the real travel. Commanding the
old endpoints then drives the motor into a mechanical stall → kp holds large
current → OVER-TEMPERATURE shutdown. This tool reads the *current* endpoints so
you can pass them back as ``GRIPPER_OPEN=/GRIPPER_CLOSED=`` (no code edit).

SAFE BY DESIGN: it commands the motor with kp=0 kd=0 torque=0 — i.e. ZERO torque,
so the jaws are FREELY back-drivable by hand and the motor cannot heat up. It only
READS the encoder. It never holds a position. This is the opposite of the teleop
control loop (which uses kp=3.0 and would fight a stall).

USAGE (can0 must be UP; run while NO other process owns the gripper):
    PYTHONPATH=third_party/i2rt:gripper \
      .venv/bin/python gripper/measure_gripper_endpoints.py --channel can0

Then, with the jaws free:
  1. push the jaws FULLY OPEN  by hand, read the steady value  -> that's GRIPPER_OPEN
  2. push the jaws FULLY CLOSED by hand, read the steady value -> that's GRIPPER_CLOSED
  3. relaunch:  GRIPPER_OPEN=<open> GRIPPER_CLOSED=<closed> RECORD=1 ./run_collection_viz.sh

Ctrl-C to quit (the motor is powered down on exit).
"""
from __future__ import annotations

import argparse
import sys
import time


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--channel", default="can0", help="SocketCAN channel (must be UP)")
    p.add_argument("--motor-id", type=int, default=7)
    p.add_argument("--motor-type", default="DM4310")
    p.add_argument("--motor-direction", type=int, default=-1,
                   help="must match keyboard_gripper (default -1) so the sign matches teleop")
    p.add_argument("--hz", type=float, default=10.0, help="print rate")
    args = p.parse_args(argv if argv is not None else sys.argv[1:])

    import numpy as np
    from i2rt.motor_drivers.dm_driver import DMChainCanInterface, MotorCmd, ReceiveMode

    print(f"[measure] opening {args.motor_type} id={args.motor_id} on {args.channel} "
          f"(motor_direction={args.motor_direction})", flush=True)
    chain = DMChainCanInterface(
        motor_list=[[args.motor_id, args.motor_type]],
        motor_offset=[0.0],
        motor_direction=[args.motor_direction],
        channel=args.channel,
        receive_mode=ReceiveMode.p16,
        motor_chain_name="measure_endpoints",
        start_thread=False,
    )
    # Enable so the encoder reports, but seed a ZERO-GAIN command so the motor
    # applies NO torque — the jaws stay freely movable by hand and cannot overheat.
    try:
        chain.motor_interface.motor_on(args.motor_id, args.motor_type)
    except Exception as exc:  # noqa: BLE001
        print(f"[measure] motor_on warning: {exc!r} (continuing — read may still work)", flush=True)
    chain.commands = [MotorCmd(torque=0.0, pos=0.0, vel=0.0, kp=0.0, kd=0.0)]
    chain.start_thread()
    time.sleep(0.2)

    print("[measure] MOTOR IS FREE (zero torque). Push the jaws by hand:", flush=True)
    print("[measure]   * fully OPEN  -> note the value as GRIPPER_OPEN", flush=True)
    print("[measure]   * fully CLOSED-> note the value as GRIPPER_CLOSED", flush=True)
    print("[measure] Ctrl-C to quit.\n", flush=True)

    lo = float("inf")
    hi = float("-inf")
    period = 1.0 / max(args.hz, 1.0)
    try:
        while True:
            # Keep re-seeding the zero-gain command so the chain's bg thread never
            # latches a holding torque, then read the live encoder position.
            chain.set_commands(
                torques=np.zeros(1), pos=np.zeros(1), vel=np.zeros(1),
                kp=np.zeros(1), kd=np.zeros(1))
            pos = float(chain.read_states()[0].pos)
            lo = min(lo, pos)
            hi = max(hi, pos)
            print(f"\r[measure] pos = {pos:+.4f} rad     (range seen: min {lo:+.4f}  max {hi:+.4f})   ",
                  end="", flush=True)
            time.sleep(period)
    except KeyboardInterrupt:
        print("\n[measure] done. Range seen this session:", flush=True)
        print(f"[measure]   min {lo:+.4f} rad   max {hi:+.4f} rad", flush=True)
        print(f"[measure] pick which is open vs closed for your mechanism, then:", flush=True)
        print(f"[measure]   GRIPPER_OPEN=<open> GRIPPER_CLOSED=<closed> RECORD=1 ./run_collection_viz.sh", flush=True)
    finally:
        try:
            chain.motor_interface.motor_off(args.motor_id)
        except Exception:  # noqa: BLE001
            pass
        try:
            chain.close()
        except Exception:  # noqa: BLE001
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
