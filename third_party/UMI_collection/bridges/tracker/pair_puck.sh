#!/usr/bin/env bash
# Pair a Vive Tracker to a Watchman dongle from the terminal -- no SteamVR window needed.
#
#   bridges/tracker/pair_puck.sh [DONGLE_SERIAL] [WINDOW_S=40]
#
# Uses SteamVR's own `lighthouse_console` (tools/lighthouse/bin/linux64), whose interactive
# `serial <dongle>` + `pair` puts THAT dongle into pairing mode; the puck must be blinking
# blue (hold its button ~2 s) inside the window. Verified 2026-09-09 on this rig:
# E296900B95 <-> LHR-C68495D2. The flag form `/serial X /pair` SEGFAULTS (exit 139) and the
# Linux SteamVR UI ignores a dongle plugged after vrserver started, which is why the
# "I want to pair a different type of controller" entry never appeared -- so this goes
# through the console. Refuses while the libsurvive bridge is up: the dongle is single-claim.
set -euo pipefail
SERIAL="${1:-}"; WINDOW="${2:-40}"
if pgrep -f "vive_tracker_bridge.py|survive-cli|survive-websocketd" >/dev/null; then
  echo "[pair] REFUSED: the tracker bridge / libsurvive is running and holds the dongles. Stop it first." >&2; exit 2
fi
LHC=""
for c in "$HOME/.local/share/Steam/steamapps/common/SteamVR" "$HOME/.steam/steam/steamapps/common/SteamVR"; do
  [ -x "$c/tools/lighthouse/bin/linux64/lighthouse_console" ] && LHC="$c/tools/lighthouse/bin/linux64/lighthouse_console" && break
done
[ -n "$LHC" ] || { echo "[pair] REFUSED: SteamVR's lighthouse_console not found (install SteamVR, appid 250820)." >&2; exit 2; }
mapfile -t DONGLES < <(for d in /sys/bus/usb/devices/*; do
  [ -f "$d/idProduct" ] || continue
  [ "$(cat "$d/idVendor")" = 28de ] && [ "$(cat "$d/idProduct")" = 2101 ] && cat "$d/serial"; done)
echo "[pair] Watchman dongles on USB: ${#DONGLES[@]} [${DONGLES[*]:-none}]"
if [ -z "$SERIAL" ]; then
  [ "${#DONGLES[@]}" -eq 1 ] || { echo "[pair] REFUSED: pass the dongle serial explicitly when more than one is plugged (a puck pairs to ONE dongle)." >&2; exit 2; }
  SERIAL="${DONGLES[0]}"
fi
printf '%s\n' "${DONGLES[@]}" | grep -qx "$SERIAL" || { echo "[pair] REFUSED: dongle $SERIAL is not on USB." >&2; exit 2; }
# lighthouse_console reaches the dongle through /dev/hidraw*, i.e. through the KERNEL usbhid
# driver. libsurvive reaches it through libusb and DETACHES that driver -- and does not give
# it back on exit (measured: after one --list-serials the interface reads driver=NONE, no
# hidraw node, and the console answers "No connected lighthouse device found" although
# lsusb still lists the dongle). Re-plugging restores the binding; so does
#   echo <bus-port>:1.0 | sudo tee /sys/bus/usb/drivers/usbhid/bind
for d in /sys/bus/usb/devices/*; do
  [ -f "$d/serial" ] && [ "$(cat "$d/serial")" = "$SERIAL" ] || continue
  IF="$(basename "$d"):1.0"
  if [ ! -e "$d/$IF/driver" ]; then
    echo "[pair] REFUSED: dongle $SERIAL has NO kernel HID driver bound ($IF: driver=NONE), so lighthouse_console cannot open it." >&2
    echo "[pair]   -> UNPLUG and RE-PLUG the dongle (or: echo $IF | sudo tee /sys/bus/usb/drivers/usbhid/bind), then re-run." >&2
    echo "[pair]   This is libsurvive's doing (libusb detached usbhid and never re-attached), typically the --list-serials check at the end of the previous run." >&2
    exit 3
  fi
done
echo "[pair] dongle $SERIAL -> pairing mode for ${WINDOW}s. NOW hold the puck's button ~2 s until it BLINKS BLUE (idle pairing times out after 30 s and the puck powers off; brief press = on again)."
# `deviceinfo` after the window: the console reports the puck now on this dongle (serial /
# "Wireless controller ... connected") WITHOUT touching the kernel driver, so a retry works.
printf 'serial %s\npair\nsleep %d\ndeviceinfo\nsleep 1500\nquit\n' "$SERIAL" "$((WINDOW * 1000))" \
  | timeout "$((WINDOW + 30))" "$LHC" 2>&1 \
  | sed 's/\x1b\[[0-9;]*m//g' | grep -v -E '^lh> *$|Attempting HID|IMU HID opened|keepalive' || true
if [ "${VERIFY_LIBSURVIVE:-0}" != "1" ]; then
  echo "[pair] window closed. (VERIFY_LIBSURVIVE=1 also runs a 15 s libsurvive --list-serials -- it DETACHES the dongle's kernel HID driver, so re-plug before pairing again.)"
  exit 0
fi
echo "[pair] window closed. Checking the radio with libsurvive (15 s) -- this detaches usbhid; re-plug the dongle before another pairing..."
cd "$(dirname "$0")/../.."
PY=".venv-vive/bin/python"; [ -x "$PY" ] || PY=python3
timeout 60 "$PY" bridges/tracker/vive_tracker_bridge.py --list-serials --list-seconds 15 2>&1 \
  | sed 's/\x1b\[[0-9;]*m//g' | grep -E 'tracker WM|dongles on USB|NO puck' || true
echo "[pair] a 'tracker WM.. serial=LHR-........ <- UNASSIGNED' line is the newly paired puck; that serial is your LEFT_FOOT_SERIAL / RIGHT_FOOT_SERIAL."
