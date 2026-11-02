# Hardware Setup

One-time host setup for the four hardware singletons (VIVE tracker, RealSense
D405, i2rt DM4310 gripper, Bota F/T sensor) plus a hardware-swap quick
reference. Do these once per machine; the per-swap table at the bottom covers
day-to-day equipment changes.

## 1. CAN bring-up (`can0`)

The gripper bridge is the **only** process that writes CAN. The DM4310 motor
hangs off a CANable / gs_usb adapter, which enumerates as a SocketCAN channel
(`can0`).

Bring the link up once per boot:

```bash
sudo ip link set can0 up type can bitrate 1000000
```

`run_collection_viz.sh` auto-detects the first CAN interface and brings it up
via `sudo` for you, so you normally don't run this by hand. Override the
defaults with env vars:

```bash
CAN_CHANNEL=can1 CAN_BITRATE=1000000 ./run_collection_viz.sh
```

Verify the link is up:

```bash
ip -details link show can0   # state should read UP, bitrate 1000000
```

### 1a. Stable CAN name (`can-umi`) — survives a USB-port move

`canN` is assigned by **enumeration order** (gs_usb), so moving the CANable to a
different USB port, or plugging a **second** CAN adapter, can silently rename it
— and the launcher's "first CAN interface" pick then binds the wrong/absent bus
(which surfaces downstream as the gripper self-test's misleading "stall / dead
motor"). Pin the adapter to a fixed name keyed on its **stable USB serial** so it
is always `can-umi` regardless of port. Once per machine, as root:

```bash
# Find this adapter's serial:
udevadm info -p /sys/class/net/can0 | grep ID_USB_SERIAL_SHORT
#   E: ID_USB_SERIAL_SHORT=0014002733354B0432333837   <- yours will differ

# Create the rename rule (substitute YOUR serial):
sudo tee /etc/systemd/network/10-can-umi.link >/dev/null <<'EOF'
[Match]
Property=ID_USB_SERIAL_SHORT=0014002733354B0432333837

[Link]
Name=can-umi
EOF

# Apply (a reload alone does NOT rename an already-up netdev):
sudo udevadm control --reload
sudo ip link set can0 down            # kernel won't rename an UP iface
sudo udevadm trigger -c add /sys/class/net/can0
#   …or just replug the adapter / reboot. Verify:
ip -o link show can-umi
```

The launcher auto-detect **prefers `can-umi`** when present, falling back to the
first CAN iface then `can0` — so this is optional and the rig still runs without
it. Pin a different adapter with `CAN_CHANNEL=...`; require a specific serial at
preflight with `CAN_SERIAL=...`.

### 1b. Device preflight

Before spawning the bridges, `run_collection_viz.sh` runs `bridges/preflight.py`,
which resolves every removable device by its **stable id** (Bota by-id symlink,
CAN USB serial, RealSense + Watchman USB ids) and **fails fast with the exact
fix** instead of deep in the stack. It HARD-fails on a missing Bota (unless
`--no-ft`) or a missing/serial-mismatched CAN iface (unless `GRIPPER=0`);
camera/tracker are warn-only. Skip it with `PREFLIGHT=0`. Run it standalone:

```bash
.venv/bin/python bridges/preflight.py        # prints PREFLIGHT OK / PREFLIGHT FAIL
```

## 2. evdev keyboard access

The `keyboard_gripper_ws` bridge reads raw key events from `/dev/input/event*`
via evdev, which requires membership in the `input` group:

```bash
sudo usermod -aG input $USER
```

**You must log out and back in** (or reboot) for the new group to take effect —
group membership is only applied at session start. Until you re-login,
`run_collection_viz.sh` falls back to launching the keyboard bridge under
`sudo`.

## 3. Bota payload bootstrap

After physically mounting the end-effector tool, capture the payload so the
F/T charts read net (tool-compensated) force. Hold the rig still during the
run:

```bash
.venv/bin/python force_torque/estimate_payload.py --max-time 60
```

This writes a per-rig `payload_compensator.json` (gitignored). **Regenerate it
per rig** — the payload is mass/CoG-specific to the mounted tool. `bootstrap.sh`
seeds an initial payload during setup; this step refreshes it for the actual
tool you've mounted.

## 4. libsurvive — REQUIRED source build at the pinned commit

The tracker bridge follows the reference VIVE stack (`bridges/tracker/vive_tracker_bridge.py`
documents the port): libsurvive **built from source** at commit
`5588cadf5ab764b51e50f9b08996009d1b9f836e` (2026-05-14) against the **system** libusb, with
the ctypes bindings taken from the same checkout. The PyPI `pysurvive` wheel (1.1.21, uploaded
2023-03-11, bundling libsurvive 1.2.7 and its own libusb) is **refused** by the bridge — it is
three years behind on Gen2-lighthouse OOTX/tracking fixes and its libusb segfaults on some
hosts. `bootstrap.sh` performs the build (idempotent); by hand:

```bash
sudo apt-get install -y libusb-1.0-0-dev liblapacke-dev libopenblas-dev cmake build-essential

git clone https://github.com/cntools/libsurvive.git ~/libsurvive
git -C ~/libsurvive checkout 5588cadf5ab764b51e50f9b08996009d1b9f836e
git -C ~/libsurvive submodule update --init --recursive
make -C ~/libsurvive -j"$(nproc)"     # -> bin/libsurvive.so + bin/plugins/ + bindings/python/pysurvive
```

`run_collection_viz.sh` exports, for the tracker process only:

```bash
export LD_LIBRARY_PATH=$LIBSURVIVE_DIR/bin:$LD_LIBRARY_PATH        # LIBSURVIVE_DIR defaults to ~/libsurvive
export PYTHONPATH=$LIBSURVIVE_DIR/bindings/python:$PYTHONPATH
```

Every launch resets the base-station cache and passes `--force-calibrate 1 --force-ootx 1
--lighthouse-update-velocity 0.5` (the reference recorder flow): every station
is re-read and re-solved, every puck the MODE needs is simply powered and in view, and the
launch blocks until every role streams (`TRACKER_STARTUP_WAIT`, 120 s). Two base stations is
the floor (`MIN_STATIONS`); `STATIONS=N` declares how many you powered for a count comparison.
`TRACKER_CALIB=ootx-only|reuse` are escape hatches, not a workflow. A role that never streams is diagnosed **per puck**
in `data/logs/tracker.log` (a table + VERDICT lines every 5 s and on FATAL, quoted by the
launcher): `no-radio` / `no-light` / `stale` / `(unassigned)` name THAT puck; "no base station
has a solved position yet" is the room solve and blocks every puck alike. The udev rules (`81-vive.rules`) give the `/dev/bus/usb` nodes `MODE=0666`; after a **replug** the `uaccess` ACL is
not re-applied automatically, so if the launcher reports a node without rw access run
`sudo udevadm control --reload-rules && sudo udevadm trigger --action=add --subsystem-match=usb && sudo udevadm settle`.

## Hardware-swap quick reference

| Swap | What to do |
|---|---|
| **Different VIVE tracker** | `LEFT_SERIAL` defaults to this rig's LEFT (primary) unit (`LHR-75106789`). For a different unit set `LEFT_SERIAL=LHR-xxxx`, or empty for first-seen latch. **With several trackers present, first-seen is non-deterministic** and can latch the wrong one (the hand then never moves). To find a unit's serial, shake it and watch the `[tracker] tracker … serial=…` log line. Every puck that produces a pose is logged with its serial (`[tracker] tracker WM.. serial=LHR-… produced its first pose`), and a configured serial that never appears is **named** by the startup wait (`waiting for left=LHR-… (remaining Ns; seen so far: WM0=LHR-…)`) and on the `VIVE trackers` board row — so a stale default reads as "configured X, saw Y", not as a dead tracker. |
| **Different Bota F/T sensor** | Nothing for connectivity (by-id auto-detect), but **regenerate the payload** with `force_torque/estimate_payload.py` (see §3). |
| **2-key keyboard (USB-C ↔ USB-A)** | Nothing — evdev auto-detects any device exposing `KEY_O` + `KEY_P`. |
| **CANable moved to another USB port** | `canN` is enumeration-order, so a port move can rename it. Pin it to `can-umi` by USB serial once (see §1a); the launcher then prefers `can-umi` regardless of port. Without the pin, the launcher still auto-detects the first CAN iface. |
| **Gripper not connecting / CAN down** | The launcher auto-brings-up the CAN iface (`can-umi` if pinned, else first/`can0`) via `sudo`. After a motor **power-cycle** the DM4310 encoder re-references, but the **startup self-test auto-homes the new endpoints** — you don't re-map by hand. **The jaws move during the self-test (auto-home + open/close); keep hands clear and nothing held.** A self-test FAIL **aborts the launch** (motor powered down). To pin endpoints manually, set `GRIPPER_OPEN=.. GRIPPER_CLOSED=..`; to skip the self-test, `SELFTEST=0`. See `gripper/README.md`. |
| **Different RealSense camera** | Nothing — first-device pick. To pin one, pass `--serial`. For non-color models, pass `--stream left-ir` as a fallback. |
