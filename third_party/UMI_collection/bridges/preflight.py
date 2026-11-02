#!/usr/bin/env python3
"""Pre-launch device preflight for the data-collection rig.

Resolve every REMOVABLE device by its STABLE identifier (by-id symlink / USB
serial) BEFORE the bridges spawn, so a USB-port move, a tty renumber, a sensor
swap, or a second-of-a-kind device fails FAST here with the exact fix — instead
of deep in the stack where the error points at the wrong subsystem (e.g. a CAN
bus on the wrong adapter surfaces as the gripper self-test's misleading
"stall / dead motor" banner).

Pure stdlib — no hardware libraries — so it runs in the main .venv (the camera
lib pyrealsense2 lives only in .venv-vive, so the D405 is checked via lsusb USB
id and is WARN-only here). Reads the SAME env knobs the launcher uses
(BOTA_SERIAL, CAN_SERIAL, CAMERA_SERIAL, RIGHT_SERIAL, GRIPPER) so behavior is
consistent with run_collection_viz.sh.

    bridges/preflight.py                 # check everything, print one line/device
    GRIPPER=0 bridges/preflight.py       # skip the CAN/gripper check
    BOTA_SERIAL=3875355D3532 bridges/preflight.py   # pin a specific Bota
    CAN_SERIAL=0014002733354B0432333837 bridges/preflight.py  # require this CANable

Prints ``PREFLIGHT OK`` (exit 0) when every HARD check passes (warnings allowed),
or ``PREFLIGHT FAIL: ...`` (exit 1) listing the hard failures. HARD checks:
exactly-one Bota present; and, unless GRIPPER=0, a CAN iface present (plus a
serial match when CAN_SERIAL is pinned). Camera + tracker are WARN-only (the
launcher's own health probes are the live gate for those).
"""
from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys

# Stable by-id symlink for any Bota FT sensor (vendor-string match, serial-agnostic
# — same glob as force_torque/bota_ft_viz.py:_BOTA_BY_ID_GLOB).
_BOTA_BY_ID_GLOB = "/dev/serial/by-id/*Bota_Systems*Force_Torque_Sensor*-if00"
# RealSense D405 + Valve Watchman (tracker dongle) USB ids, for an lsusb presence
# check that needs no vendor library.
_D405_USB_ID = "8086:0b5b"
_WATCHMAN_USB_ID = "28de:2101"


def _run(cmd: list[str]) -> str:
    """Best-effort capture of a command's stdout (empty on any failure)."""
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=5).stdout
    except Exception:  # noqa: BLE001 - tool missing / timeout: treat as no output
        return ""


def _lsusb_has(usb_id: str) -> bool:
    return _lsusb_count(usb_id) > 0


def _lsusb_count(usb_id: str) -> int:
    """How many devices with this USB vendor:product id are on the bus.

    A COUNT, not a boolean, because the rig now expects two of some kinds (two D405s,
    two CANables) and "at least one present" cannot distinguish a complete two-hand rig
    from a one-hand rig with the second unit unplugged."""
    return sum(1 for line in _run(["lsusb"]).splitlines() if usb_id in line)


def _can_ifaces() -> list[str]:
    """All SocketCAN interface names, e.g. ['can-umi'] or ['can0', 'can1']."""
    out = _run(["ip", "-o", "link", "show", "type", "can"])
    names = []
    for line in out.splitlines():
        # "8: can-umi: <NOARP,UP,...> ..."  ->  field 1 is "can-umi" (strip @ if any)
        parts = line.split(": ", 2)
        if len(parts) >= 2:
            names.append(parts[1].split("@")[0].strip())
    return names


def _can_serial(iface: str) -> str | None:
    """The CANable's stable USB serial (ID_USB_SERIAL_SHORT) for an iface, or None."""
    out = _run(["udevadm", "info", "-p", f"/sys/class/net/{iface}"])
    for line in out.splitlines():
        if "ID_USB_SERIAL_SHORT=" in line:
            return line.split("ID_USB_SERIAL_SHORT=", 1)[1].strip()
    return None


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--no-gripper", action="store_true",
                   help="skip the CAN/gripper check (mirrors GRIPPER=0)")
    p.add_argument("--no-ft", action="store_true",
                   help="downgrade a missing Bota to a WARN (mirrors the viz --no-ft / NO_FT=1)")
    args = p.parse_args(argv if argv is not None else sys.argv[1:])

    fails: list[str] = []  # hard failures -> exit 1
    warns: list[str] = []  # advisory -> still exit 0

    # --- Bota F/T (HARD, unless --no-ft / NO_FT=1 -> WARN) --------------------
    # PER-HAND once MODE is 2pt/3pt. BOTA_SERIAL is the legacy single-sensor pin and
    # still wins when set; otherwise the LEFT hand is checked against FT_LEFT_SERIAL and
    # (in 2pt/3pt) the RIGHT against FT_RIGHT_SERIAL. "Two sensors found, using the
    # first" is NOT an acceptable answer with two hands attached: the glob is sorted by
    # serial and 3472… sorts before 3875…, so the unpinned first match is the RIGHT
    # hand's sensor — the launcher would have opened it as the left hand's.
    bota_serial = os.environ.get("BOTA_SERIAL", "").strip()
    ft_left = os.environ.get("FT_LEFT_SERIAL", "").strip()
    ft_right = os.environ.get("FT_RIGHT_SERIAL", "").strip()
    mode = os.environ.get("MODE", "1pt").strip() or "1pt"
    want_right_ft = mode != "1pt" and os.environ.get("RIGHT_FT", "1") != "0"
    no_ft = args.no_ft or os.environ.get("NO_FT") == "1"
    present = sorted(glob.glob(_BOTA_BY_ID_GLOB))
    left_pin = bota_serial or ft_left
    matched = [m for m in present if left_pin in m] if left_pin else present
    if not present:
        # The viz runs fine with F/T disabled (--no-ft); only HARD-fail when F/T is wanted.
        msg = ("Bota F/T: NO sensor under /dev/serial/by-id "
               "(check the USB cable / power; the sensor may be on another port).")
        (warns if no_ft else fails).append(msg)
    elif left_pin and not matched:
        label = "BOTA_SERIAL" if bota_serial else "FT_LEFT_SERIAL"
        msg = (f"Bota F/T (L): {label}={left_pin} not found; "
               f"present: {[os.path.basename(m) for m in present]}")
        (warns if no_ft else fails).append(msg)
    else:
        chosen = matched[0]
        if len(matched) > 1 and not left_pin:
            warns.append(f"Bota F/T: {len(matched)} sensors found, using "
                         f"{os.path.basename(chosen)}; set FT_LEFT_SERIAL to pin the left hand.")
        print(f"[preflight] Bota F/T (L): {os.path.basename(chosen)}")
    # RIGHT hand: optional by design, so absence is a WARN — the session still records
    # the left hand and the converter simply emits no robot1_ft_wrench.
    if want_right_ft:
        if not ft_right:
            warns.append("Bota F/T (R): no FT_RIGHT_SERIAL set — right wrench NOT recorded.")
        elif ft_right == left_pin:
            fails.append(f"Bota F/T: FT_LEFT_SERIAL == FT_RIGHT_SERIAL ({ft_right}) — "
                         "both hands would open the SAME sensor.")
        elif not any(ft_right in m for m in present):
            warns.append(f"Bota F/T (R): FT_RIGHT_SERIAL={ft_right} not present — "
                         "right wrench NOT recorded (left hand unaffected).")
        else:
            print(f"[preflight] Bota F/T (R): ...{ft_right}-if00")

    # --- CAN / gripper (HARD unless skipped) ---------------------------------
    if args.no_gripper or os.environ.get("GRIPPER") == "0":
        print("[preflight] CAN/gripper : skipped (GRIPPER=0)")
    else:
        can_serial = os.environ.get("CAN_SERIAL", "").strip()
        pinned = os.environ.get("CAN_CHANNEL", "").strip()
        ifaces = _can_ifaces()
        if not ifaces:
            fails.append("CAN: no SocketCAN interface present "
                         "(USB-CAN adapter unplugged? expected gs_usb 'can-umi' or 'can0').")
        elif pinned and pinned not in ifaces:
            # The launcher uses a non-empty CAN_CHANNEL VERBATIM (no auto-detect), so a
            # pin to an absent iface would fail later at the CAN-ready gate. Mirror that
            # here instead of silently falling through to can-umi/first (false OK).
            fails.append(f"CAN: pinned CAN_CHANNEL={pinned!r} is not present; "
                         f"have {ifaces}.")
        else:
            # Mirror the launcher's resolution: explicit pin > serial-pinned can-umi > first.
            if pinned and pinned in ifaces:
                iface = pinned
            elif "can-umi" in ifaces:
                iface = "can-umi"
            else:
                iface = ifaces[0]
            serial = _can_serial(iface)
            if can_serial and serial != can_serial:
                fails.append(f"CAN: iface {iface} has serial {serial!r} but "
                             f"CAN_SERIAL={can_serial!r} was required (wrong/!pinned adapter).")
            else:
                extra = f" serial={serial}" if serial else ""
                multi = f"  (+{len(ifaces) - 1} other CAN iface)" if len(ifaces) > 1 else ""
                print(f"[preflight] CAN/gripper : {iface}{extra}{multi}")
            if iface != "can-umi" and "can-umi" not in ifaces:
                warns.append("CAN: using an enumeration-order name (not the serial-pinned "
                             "'can-umi'); a USB-port move can rename it. See docs/HARDWARE_SETUP.md.")
            # RIGHT gripper's adapter, resolved the same way the launcher does: by USB
            # serial, never by canN name. WARN-only — the right gripper is optional.
            if mode != "1pt" and os.environ.get("RIGHT_GRIPPER", "1") != "0":
                r_pin = os.environ.get("GRIPPER_RIGHT_CHANNEL", "").strip()
                r_ser = os.environ.get("GRIPPER_RIGHT_CAN_SERIAL", "").strip()
                r_iface = r_pin or next(
                    (i for i in ifaces if r_ser and _can_serial(i) == r_ser), "")
                if not r_iface:
                    warns.append(f"CAN (R): no adapter with serial {r_ser or '(unset)'} — "
                                 "right gripper NOT recorded (left hand unaffected).")
                elif r_iface == iface:
                    fails.append(f"CAN: the right gripper resolved to {r_iface}, the SAME "
                                 "iface as the left — two writers on one bus breaks the "
                                 "single-CAN-writer rule.")
                else:
                    print(f"[preflight] CAN (R)     : {r_iface}"
                          + (f" serial={r_ser}" if r_ser else ""))

    # --- Camera (WARN-only; pyrealsense2 is in .venv-vive) -------------------
    # Serials can only be READ with pyrealsense2 (.venv-vive), so from here we can count
    # D405s on the USB bus and echo the pins — enough to catch "the right camera is not
    # plugged in", which is the failure that silently costs a session its camera1_rgb.
    n_d405 = _lsusb_count(_D405_USB_ID)
    want_right_cam = mode != "1pt" and os.environ.get("RIGHT_CAMERA", "1") != "0"
    if n_d405:
        cam = os.environ.get("CAMERA_SERIAL", "").strip()
        cam_r = os.environ.get("RIGHT_CAMERA_SERIAL", "").strip()
        pins = [f"L={cam or 'first-device'}"]
        if want_right_cam:
            pins.append(f"R={cam_r or 'UNSET'}")
        print(f"[preflight] camera D405 : {n_d405} on the bus ({', '.join(pins)})")
        if not cam and n_d405 > 1:
            warns.append(f"camera: {n_d405} D405s present and CAMERA_SERIAL is unset — "
                         "'first device' is enumeration order, so the LEFT bridge may open "
                         "the RIGHT camera. Pin CAMERA_SERIAL.")
        if want_right_cam and cam_r and cam_r == cam:
            fails.append(f"camera: CAMERA_SERIAL == RIGHT_CAMERA_SERIAL ({cam}) — "
                         "two bridges cannot stream the same device.")
        if want_right_cam and n_d405 < 2:
            warns.append(f"camera: MODE={mode} wants a RIGHT camera but only {n_d405} D405 "
                         "on the bus — no camera1_rgb will be recorded.")
    else:
        warns.append("camera: no RealSense D405 (USB 8086:0b5b) on the bus.")

    # --- VIVE tracker (WARN-only) -------------------------------------------
    if _lsusb_has(_WATCHMAN_USB_ID):
        # LEFT is the primary hand (LEFT-canonical); show it plus any right serial.
        left = os.environ.get("LEFT_SERIAL", "").strip()
        right = os.environ.get("RIGHT_SERIAL", "").strip()
        _labels = [f"left={left}"] if left else []
        if right:
            _labels.append(f"right={right}")
        print("[preflight] VIVE tracker: Watchman dongle present"
              + (f" ({', '.join(_labels)})" if _labels else ""))
    else:
        warns.append("tracker: no Valve Watchman dongle (USB 28de:2101) on the bus.")

    for w in warns:
        print(f"[preflight] WARN: {w}", file=sys.stderr)
    if fails:
        for f in fails:
            print(f"PREFLIGHT FAIL: {f}", file=sys.stderr)
        return 1
    print("PREFLIGHT OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
