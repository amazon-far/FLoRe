#!/usr/bin/env bash
# Launch the full data-collection viz stack: the hardware bridges (each under the
# venv that can talk to its device) + the viser app. Self-contained: all paths are
# relative to THIS repo (no holosoma_extensions, no conda).
#
# Topology (all localhost WebSocket JSON). The tracker bridge is SINGLE (one pysurvive
# USB singleton serves every role in one role-keyed frame); the camera and gripper are
# per-hand DEVICES, so 2pt/3pt spawn a second bridge for each:
#   vive_tracker_bridge   (.venv-vive py3.9 + SOURCE-BUILT libsurvive @5588cadf, the reference stack)
#                                                       ws://127.0.0.1:8772  ALL roles' SE3
#   realsense_bridge  L   (.venv-vive, py3.9 pyrealsense) ws://127.0.0.1:8773  LEFT D405 RGB JPEG
#   realsense_bridge  R   (.venv-vive, py3.9 pyrealsense) ws://127.0.0.1:8777  RIGHT D405 (2pt/3pt)
#   keyboard_gripper  L   (.venv, py3.12 + i2rt + can0)   ws://127.0.0.1:8774  encoder rad + control
#   keyboard_gripper  R   (.venv, py3.12 + i2rt + can1)   ws://127.0.0.1:8778  RIGHT jaws (2pt/3pt)
#       ^ HEADLESS --web-control: no curses TTY, no `xset`. Driven by the
#         Open/Close/Stop buttons + slider on the viser page over the encoder WS.
#   keyboard_gripper_ws   (.venv, py3.12 + evdev)         ws-> :8774 + :8778   f/g = LEFT jaws,
#                                                                              j/k = RIGHT jaws
#   collection_viz        (.venv, py3.12 viser+mink)      http://localhost:8080  the scene
#       ^ ALSO owns the Bota F/T reader(s) in-process and republishes the full-rate
#         wrench for the recorder: :8775 (LEFT) and :8779 (RIGHT).
#       ^ Its PANEL is per-hand in 2pt/3pt: each present hand gets its own
#         `Gripper (left|right)` folder (fraction + encoder rad + Open/Close/Stop +
#         slider, driving that hand only), its own `F/T L`/`F/T R` row + wrench
#         charts, and its own `F/T compensation (left|right)` folder — Tare, the
#         raw/compensated toggle, the bota/rnea dropdown and Recompensation, each
#         writing only into that sensor's state and its OWN payload_compensator_*.json.
#         Only Recompensation is one-hand-at-a-time. `s`/Start auto-tares BOTH.
#
# PER-HAND LAYOUT (what MODE selects):
#   LEFT  hand : tracker + camera + Bota F/T + DM4310 gripper
#   RIGHT hand : tracker + camera + Bota F/T + DM4310 gripper   (2pt/3pt)
#   CHEST      : tracker ONLY — a body reference, the third point of holosoma's
#                exp:g1-wbt-3pt-hybrid-unified[-anneal] (torso_link + 0.35 z)  (3pt)
# Every RIGHT-hand DEVICE is optional: a missing one is ⚠️ degraded on the status board
# and simply records no frames (never zero-filled). RIGHT_CAMERA/RIGHT_FT/RIGHT_GRIPPER=0
# skip one deliberately. See the RIGHT-hand config block below for serials/channels.
#
# Usage:
#   ./run_collection_viz.sh                 # prompts MODE (1/2/3pt), then the full stack
#   MODE=3pt ./run_collection_viz.sh        # skip the mode prompt (1pt|2pt|3pt); LEFT-canonical
#   ./run_collection_viz.sh --mock          # viz only, synthetic everything (no hardware)
#   LEFT_SERIAL=LHR-xxxx ./run_collection_viz.sh    # override the LEFT (primary) tracker (default LHR-75106789)
#   GRIPPER=0 ./run_collection_viz.sh       # skip the gripper bridge (no CAN/motor)
#   KEYBOARD=0 ./run_collection_viz.sh      # skip the f/g + j/k keyboard bridge
#   DISPLAY_MODE=humanoid ./run_collection_viz.sh   # force the full G1 (auto = gripper in EVERY
#   DISPLAY_MODE=gripper  ./run_collection_viz.sh   # mode). gripper = yam meshes + camera per hand
#                                           # (+ a chest-puck triad in 3pt), no mink IK.
#                                           # Display-only; recording unchanged.
#   FT_ALGO=rnea ./run_collection_viz.sh    # start F/T compensation on the independent RNEA
#                                           # algorithm (default bota); switchable live in the GUI
#                                           # ("Compensation algorithm" dropdown). Same as --ft-algo rnea.
#
# Gripper endpoints drift after a motor power-cycle (the DM4310 absolute encoder
# re-references). If the jaws stall or don't fully open/close, re-map the stops
# and pass them WITHOUT editing code (the i2rt defaults are open=4.90 closed=0.0;
# the values below are an EXAMPLE — yours differ after each power-cycle):
#   GRIPPER_OPEN=-1.35 GRIPPER_CLOSED=-6.40 ./run_collection_viz.sh
#
# Stop with Ctrl-C; all spawned bridges are killed on exit (the gripper's SIGTERM
# handler powers the motor down first).
set -eo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HUMI_SRC="$REPO_ROOT/third_party/humi_ik"
I2RT_SRC="$REPO_ROOT/third_party/i2rt"
GRIPPER_PKG="$REPO_ROOT/gripper"

# Two uv venvs (see bootstrap.sh). py3.12 hosts viz+IK+bota+gripper+keyboard;
# py3.9 hosts the pyrealsense camera (cp39-only wheel) and the tracker bridge, whose
# pysurvive is NOT a wheel: the source-built libsurvive (LIBSURVIVE_DIR below) is
# put on LD_LIBRARY_PATH/PYTHONPATH for that one process.
VENV="$REPO_ROOT/.venv"
VENV_VIVE="$REPO_ROOT/.venv-vive"
BOTA_PY="${BOTA_PY:-$VENV/bin/python}"        # viz + bota + gripper + keyboard
PIKA_PY="${PIKA_PY:-$VENV_VIVE/bin/python}"   # tracker + camera
I2RT_PY="${I2RT_PY:-$VENV/bin/python}"        # gripper + keyboard bridge

# --- tracking mode (1pt / 2pt / 3pt / 5pt) — LEFT-canonical ----------
# The PRIMARY hand is the LEFT hand in EVERY mode (this is the single canonical
# serial->role convention that removes the old "same tracker is 'right' in 1pt
# but 'left' in 3pt" ~180° yaw flip):
#   1pt: left=LHR-75106789 (the only tracker)
#   2pt: + right=LHR-60CDD21F
#   3pt: + head=LHR-3DC276BA
#   5pt: + left_foot=LHR-C68495D2 (dongle E296900B95)  right_foot=LHR-FF1D3DE2 (dongle 3641D31EB2)
# LEFT_SERIAL is fixed across modes; RIGHT/HEAD/the FEET are added by mode. Override any
# with LEFT_SERIAL/RIGHT_SERIAL/HEAD_SERIAL/LEFT_FOOT_SERIAL/RIGHT_FOOT_SERIAL. MODE is
# prompted interactively when unset on a TTY (see the prompt block below); set
# MODE=1pt|2pt|3pt|5pt to skip it.
#
# The two FOOT pucks were paired 2026-09-09 (bridges/tracker/pair_puck.sh) and ASSIGNED by
# the operator: LEFT foot = LHR-C68495D2 (its dongle E296900B95), RIGHT foot = LHR-FF1D3DE2
# (dongle 3641D31EB2). Like the other three these are executable defaults for THIS rig;
# override with LEFT_FOOT_SERIAL= / RIGHT_FOOT_SERIAL= if a puck is swapped. A wrong foot
# serial does not fail — it records a mirrored stance with every gate green — so re-check
# with `.venv-vive/bin/python bridges/tracker/vive_tracker_bridge.py --list-serials` (read-only)
# whenever a puck is replaced.
MODE="${MODE:-}"                                   # resolved by the prompt block below
LEFT_SERIAL="${LEFT_SERIAL:-LHR-75106789}"         # primary hand, all modes
RIGHT_SERIAL="${RIGHT_SERIAL:-LHR-60CDD21F}"       # 2pt/3pt/5pt
HEAD_SERIAL="${HEAD_SERIAL:-LHR-3DC276BA}"         # 3pt/5pt
LEFT_FOOT_SERIAL="${LEFT_FOOT_SERIAL:-LHR-C68495D2}"   # 5pt (paired 2026-09-09, dongle E296900B95)
RIGHT_FOOT_SERIAL="${RIGHT_FOOT_SERIAL:-LHR-FF1D3DE2}"  # 5pt (paired 2026-09-09, dongle 3641D31EB2)
YAW_DEG="${YAW_DEG:-0}"
# FULL libsurvive-world -> robot-world alignment "ROLL PITCH YAW" (deg), applied to
# BOTH tracker pos + quat. Use this (NOT YAW_DEG) when a physical PITCH shows up as
# a YAW in the viz — a world TILT that a pure yaw structurally cannot correct. Derive
# the numbers first:
#   .venv-vive/bin/python bridges/tracker/vive_tracker_bridge.py --measure-frame
# (hold the tracker still, then do a slow pure pitch/yaw/roll; read the axis).
# Example once measured:  WORLD_RPY="0 90 0" ./run_collection_viz.sh
#
# RIG DEFAULT = "0 0 180": on this rig the libsurvive world +x points OPPOSITE the
# robot's forward, so an uncorrected pose reads with forward/back (+/-x) AND left/
# right (+/-y) REVERSED, and rotation about x/y reversed (z / yaw are correct — a
# pure yaw, verified: both z channels are clean, so it is NOT a lighthouse tilt).
# Rz(180) maps pos (x,y,z)->(-x,-y,z) and flips the rotation axis x->-x,y->-y,z->z,
# which is exactly that symptom. This is baked into every recorded pose by the
# bridge and filed as provenance to the recorder; training targets are anchor-
# relative so a global world rotation cancels in the deltas. Set WORLD_RPY="0 0 0"
# to DISABLE (back to raw libsurvive world), or re-measure if the rig/lighthouses move.
WORLD_RPY="${WORLD_RPY:-0 0 180}"
# AXISCHECK=1 runs the interactive +X/+Y/+Z direction check after the bridges are up
# (bridges/tracker/axis_check.py) and ABORTS the launch if an axis is mis-mapped. It
# prints the corrective WORLD_RPY to relaunch with. Opt-in (default 0) because it needs
# the operator to physically move the tracker; run it after any re-mount, lighthouse
# move, or WORLD_RPY change, and before recording a fresh training session.
AXISCHECK="${AXISCHECK:-0}"
# --- VIVE tracker stack: the reference recorder flow (umi_collection)
# libsurvive is SOURCE-BUILT at the pinned commit with the system libusb and driven
# through its own ctypes bindings; the PyPI pysurvive wheel (2023, libsurvive 1.2.7 + a
# vendored libusb) is REFUSED by the bridge. ONE launch, ONE libsurvive process, EVERY puck
# the MODE needs powered (solid green) and in view of the stations before launch — that is
# The reference recorder's whole ritual (`umi_collection/sensors/vive_tracker.py`, `scripts/record_session.py`):
#   * the cache is reset and libsurvive gets --force-calibrate 1 --force-ootx 1
#     --lighthouse-update-velocity 0.5: every station is re-read (OOTX) and RE-SOLVED at every
#     launch, so each launch is its own world frame (WORLD_RPY survives because libsurvive
#     anchors the frame on the lowest hardware BaseStationID, not on discovery order).
#   * the launch BLOCKS until every role has produced a pose (the reference recorder's wait_until_tracking /
#     --vive-warmup-timeout; here TRACKER_STARTUP_WAIT, default 120 s) and only THEN starts
#     the cameras (they share USB bandwidth with the dongles). Nothing to do while it waits;
#     a reference Orin with 2 stations and 5 pucks takes ~20-25 s.
#   * a role that never streams FAILS the launch and is named per puck (bridge table + VERDICT).
# There is no wave gate, no calibration pass and no cache reuse by default. FRESH_CALIB /
# FRESH_OOTX / EXPECT_LIGHTHOUSES / TRACKER_GSS / TRACKER_STACK_KB are RETIRED (ignored with
# a notice; the station knobs are STATIONS / MIN_STATIONS below).
LIBSURVIVE_DIR="${LIBSURVIVE_DIR:-$HOME/libsurvive}"
LIBSURVIVE_COMMIT="5588cadf5ab764b51e50f9b08996009d1b9f836e"
# TRACKER_CALIB — an ESCAPE HATCH; the default `fresh` is the reference recorder's flag pair on a reset cache.
# A station's cache entry has two independent halves (survive.c): its factory blob (fcal +
# id + channel, cleared by --force-ootx) and its solved room pose (cleared by --force-calibrate).
#   fresh      both re-derived every launch (the reference flow)                       <- default
#   ootx-only  keep the factory blobs, re-solve the room
#   reuse      keep both, re-derive nothing (bridges/tracker/libsurvive_cache.py first repairs
#              the slot-numbering gap config_save leaves after a partial run). Only for a room a
#              previous launch solved and nothing has moved since.
# `auto` (the old default) is accepted and means fresh.
TRACKER_CALIB="${TRACKER_CALIB:-fresh}"
case "$TRACKER_CALIB" in
  fresh|ootx-only|reuse) ;;
  auto) echo "[run] NOTE: TRACKER_CALIB=auto now means fresh — every launch re-reads and re-solves the stations" >&2
        TRACKER_CALIB=fresh ;;
  *) echo "[run] ERROR: TRACKER_CALIB must be fresh|ootx-only|reuse (got '$TRACKER_CALIB')" >&2; exit 1 ;;
esac
TRACKER_STARTUP_WAIT="${TRACKER_STARTUP_WAIT:-120}"
case "$TRACKER_STARTUP_WAIT" in
  ''|*[!0-9]*) echo "[run] ERROR: bad TRACKER_STARTUP_WAIT='$TRACKER_STARTUP_WAIT' (want seconds)" >&2; exit 2 ;;
esac
# BASE STATIONS. The set libsurvive uses is whatever the pucks can SEE when the launch starts
# (a station joins the first time a puck reports a sync on its channel, and the set only
# grows), and libsurvive attempts NO room solve until every station seen has decoded its OOTX
# blob — so power exactly the stations you mean to use; fewer lock faster.
#   MIN_STATIONS  (default 2) the floor: fewer SOLVED stations at READY is a blocking ❌
#                 (a one-station solve yields poses but no parallax).
#   STATIONS      (optional) how many you powered, for a 2/3/4/6 comparison: the board prints
#                 `solved of seen` + the time to READY and warns when seen != STATIONS.
MIN_STATIONS="${MIN_STATIONS:-2}"
STATIONS="${STATIONS:-}"
case "$MIN_STATIONS" in
  ''|*[!0-9]*) echo "[run] ERROR: bad MIN_STATIONS='$MIN_STATIONS' (want a station count)" >&2; exit 2 ;;
esac
case "$STATIONS" in
  *[!0-9]*) echo "[run] ERROR: bad STATIONS='$STATIONS' (want a station count, or unset)" >&2; exit 2 ;;
esac
for _retired in FRESH_CALIB FRESH_OOTX EXPECT_LIGHTHOUSES TRACKER_GSS TRACKER_STACK_KB; do
  if [ -n "${!_retired:-}" ]; then
    echo "[run] NOTE: $_retired=${!_retired} is IGNORED — the tracker stack follows the reference recorder: fresh OOTX +" >&2
    echo "[run]       room solve every launch, every station the pucks see. Station knobs: STATIONS= / MIN_STATIONS=." >&2
  fi
done
# LEFT-hand D405. PINNED, not "first device": with two D405s attached, librealsense's
# device order is enumeration order, so an unpinned LEFT bridge is a coin flip that can
# hand the left hand the RIGHT camera and mislabel a whole session. (Confirmed 2026-08-25
# via `rs-enumerate-devices -s`; this is the serial bridges/camera/realsense_bridge.py was
# developed against.) Set CAMERA_SERIAL= (empty) to go back to first-device.
CAMERA_SERIAL="${CAMERA_SERIAL-323622272456}"
CAMERA_STREAM="${CAMERA_STREAM:-}" # color (default) | left-ir for non-color models
# ---- RIGHT-hand per-hand devices (2pt/3pt only) ------------------------------
# The rig's 3pt configuration is: LEFT hand = tracker+camera+F/T+gripper, RIGHT hand =
# tracker+camera+F/T+gripper, CHEST = tracker ONLY. The right TRACKER needs no extra
# process (one pysurvive bridge serves all three roles on :8772); the right camera,
# Bota and gripper are separate DEVICES and so get their own bridge/reader each.
#
# EVERY ONE OF THESE IS OPTIONAL AND ABSENCE IS NOT AN ERROR. A device that is not
# plugged in is reported ⚠️ degraded on the status board and the take simply records no
# frames for it (recording_format's PER-HAND banner) — it never blocks a session and
# never zero-fills a fake reading. Set the *_RIGHT=0 knobs to skip one deliberately.
#
# Camera L/R assignment is PHYSICAL and cannot be auto-detected — which D405 is taped
# to which handheld unit is a fact about the room. Both serials below are PINNED (see
# CAMERA_SERIAL above for why "first device" is unsafe once a second camera exists); the
# viz labels each frustum `camera L` (cyan) / `camera R` (orange) so a swap is a
# 5-second visual check. These are this rig's two handheld D405s, NOT holosoma's
# robot-mounted CAM0/CAM1 pair (323622270532 / 353322271384) — different cameras.
RIGHT_CAMERA="${RIGHT_CAMERA:-1}"                       # 0 = skip the right-hand camera
RIGHT_CAMERA_SERIAL="${RIGHT_CAMERA_SERIAL:-353322270760}"
# CAMERA_WS_PORT / CAMERA_RIGHT_WS_PORT — the two camera bridges' WS ports: 8873 / 8877,
# deliberately NOT realsense_bridge.py's own default (8773) nor 8777, because the deploy
# stack's robot-mounted camera bridges (holosoma_extensions 6_/8_*.sh, hl_serving_*) own
# 8773 / 8777 and the two stacks run on the same box. Measured 2026-09-10 00:25 with both
# on the old shared numbers: collection's bridges opened their handheld D405s fine and
# died two seconds later on `OSError: [Errno 98] address already in use`, into a log nobody
# was tailing — the session ran on with no images at all. recording_format.py's
# CAMERA_*_PORT_DEFAULT, the recorder's and the viz's --camera-url defaults say the same.
CAMERA_WS_PORT="${CAMERA_WS_PORT:-8873}"
CAMERA_RIGHT_WS_PORT="${CAMERA_RIGHT_WS_PORT:-8877}"
# Bota serial->hand for the two HANDHELD sensors, resolved through /dev/serial/by-id/ so
# a USB re-enumeration cannot silently swap the hands. (holosoma's robot-mounted pair is
# 3870355D3532 / 386B355D3532 — different sensors, don't copy those here.)
#
# PINNING IS MANDATORY, and note the direction of the trap changed: find_bota_port()
# takes the first *sorted* by-id match, and 3472… sorts BEFORE 3875…, so with both
# handheld sensors attached an unpinned LEFT hand opens the RIGHT sensor. A pinned
# serial that is absent returns None rather than falling back to the other hand.
RIGHT_FT="${RIGHT_FT:-1}"                               # 0 = skip the right-hand Bota
FT_LEFT_SERIAL="${FT_LEFT_SERIAL:-3875355D3532}"
FT_RIGHT_SERIAL="${FT_RIGHT_SERIAL:-347237663335}"
FT_PUB_RIGHT_PORT="${FT_PUB_RIGHT_PORT:-8779}"
# Right gripper = a second DM4310 on its own CAN bus. Both CANable adapters are resolved
# BY USB SERIAL, not by interface name: gs_usb names them canN in ENUMERATION order, so a
# replug can swap can0/can1 and quietly point a hand at the other hand's motor. The LEFT
# adapter already has a systemd .link pinning it to `can-umi` (see CAN_CHANNEL below);
# the right one has no such rule, so it arrives as canN and we look it up by serial.
# GRIPPER_RIGHT_CHANNEL= (explicit iface) still wins if you set it.
RIGHT_GRIPPER="${RIGHT_GRIPPER:-1}"                     # 0 = skip the right-hand gripper
GRIPPER_RIGHT_CAN_SERIAL="${GRIPPER_RIGHT_CAN_SERIAL:-001D005433354B0332333837}"
GRIPPER_RIGHT_CHANNEL="${GRIPPER_RIGHT_CHANNEL:-}"      # empty = resolve by the serial above
GRIPPER_RIGHT_WS_PORT="${GRIPPER_RIGHT_WS_PORT:-8778}"
GRIPPER_RIGHT_OPEN="${GRIPPER_RIGHT_OPEN:-}"   # pin the right endpoints (rad); else self-test homes
GRIPPER_RIGHT_CLOSED="${GRIPPER_RIGHT_CLOSED:-}"
# What the viz DRAWS (does NOT change what is recorded — recording is untouched):
#   auto (default) = gripper-only in EVERY mode. The rig IS two isolated handheld
#                    grippers, so 1pt/2pt/3pt draw the same cluster(s) and the RIGHT hand
#                    is drawn identically to the LEFT (same meshes, same TCP/tracker/
#                    wrist_yaw triads, same puck binding). 3pt adds ONE object: a bare
#                    axis triad for the chest puck.
#                    NOTE this is also what keeps RAW_FRAME=1 alive in 3pt — raw frame is
#                    a gripper-only feature, so an auto->humanoid 3pt silently dropped it.
#   humanoid       = force the full G1 (mink IK) in any mode. The only display that
#                    exercises the torso/waist-yaw task; ~8 ms of solve per frame.
#   gripper        = force gripper-only (3 yam meshes + camera per hand, NO mink IK) in
#                    any mode; same as auto now.
# Named DISPLAY_MODE (not DISPLAY) on purpose: $DISPLAY is the X11 server var and is
# set on every desktop session — reusing it would clobber X11 and make auto-vs-forced
# ambiguous. The camera-to-gripper geometry is identical in both display modes.
DISPLAY_MODE="${DISPLAY_MODE:-auto}"
case "$DISPLAY_MODE" in
  auto|humanoid|gripper) ;;
  *) echo "[run] ERROR: bad DISPLAY_MODE='$DISPLAY_MODE' (want auto|humanoid|gripper)" >&2; exit 2 ;;
esac
# RAW_FRAME=1 (gripper-only display only): draw the gripper cluster at the puck's RAW
# libsurvive pose with NO home-anchor — TCP position+axes = the fused -y,-x,-z
# remapped-TCP, wrist_yaw at TCP-0.242x — a faithful twin of the recorded tracker.npy
# frame. Ignored in humanoid mode. Display only; recording is untouched.
# DEFAULT ON (1) since 2026-09-11: with the base stations re-solved on every launch the
# world yaw is whatever libsurvive picked this time, and the home-anchored view applies a
# WORLD-frame delta onto a +x-facing home -- a pitch the operator makes reads as a
# pitch+roll mix there, while the raw frame draws the axes the recording actually
# contains (what replay_raw_viz shows afterwards). RAW_FRAME=0 restores the home-anchored
# view; DISPLAY_MODE=humanoid needs RAW_FRAME=0 (see below).
RAW_FRAME="${RAW_FRAME:-1}"
case "$RAW_FRAME" in
  0|1) ;;
  *) echo "[run] ERROR: bad RAW_FRAME='$RAW_FRAME' (want 0|1)" >&2; exit 2 ;;
esac
# RAW_FRAME=1 + an EXPLICIT humanoid display is a contradiction: raw frame is a
# gripper-only feature, so the viz drops it and you silently get the home-anchored view.
# That combination is how a 3pt session quietly lost its raw frame back when auto resolved
# 3pt to humanoid. Refuse it rather than print a NOTE nobody reads in the startup scroll —
# `auto` needs no help here (it is gripper in every mode), so this only fires when someone
# asked for both by name.
if [ "$RAW_FRAME" = "1" ] && [ "$DISPLAY_MODE" = "humanoid" ]; then
  echo "[run] ERROR: RAW_FRAME=1 (the default) cannot be drawn in DISPLAY_MODE=humanoid (raw" >&2
  echo "[run]        frame is a gripper-only view). Drop DISPLAY_MODE=humanoid, or pass RAW_FRAME=0." >&2
  exit 2
fi
# CAMERA_VIEW=0 boots the viz with its "Show camera view" checkbox UNCHECKED. That
# checkbox is the viz's single biggest CPU knob and it is not obvious why: a live frustum
# image is a per-frame JPEG decode PLUS a JPEG *re*-encode (viser's CameraFrustumHandle
# re-encodes on every assignment) for EACH camera, on the render thread, whether or not a
# browser tab is even open. Measured on this 3pt rig with two 640x480@30Hz D405s:
# collection_viz 50.9% -> 26.0% of one core with it off.
# DISPLAY ONLY. The recorder is a separate process with its own subscription to the same
# camera bridges and writes their JPEG bytes verbatim (recording/recorder.py), so this
# cannot change one byte of a recorded take — and the bridges keep running either way.
# Default 1 (shown), because the frustums are how you confirm which physical D405 is on
# which hand, which cannot be auto-detected. The normal flow is: launch with it on, check
# the two frustums, then UNCHECK it in the sidebar before recording — the toggle stays
# live, this env var only picks the state you boot into (e.g. an unattended session).
CAMERA_VIEW="${CAMERA_VIEW:-1}"
case "$CAMERA_VIEW" in
  0|1) ;;
  *) echo "[run] ERROR: bad CAMERA_VIEW='$CAMERA_VIEW' (want 0|1)" >&2; exit 2 ;;
esac
# CAMERA_FPS / CAMERA_PUBLISH_HZ — the D405 capture rate and the bridges' PUBLISH rate,
# both 30 by default. The publish rate IS the training dataset's master timeline (the
# converter's action grid), so it is declared here rather than buried in a default, and it
# must be an integer divisor of CAMERA_FPS (the bridge refuses otherwise: publishing is
# count-based decimation so the grid stays uniform). 15 still records and converts; it
# only makes the action grid coarser. Never lower it to save CPU.
CAMERA_FPS="${CAMERA_FPS:-30}"
CAMERA_PUBLISH_HZ="${CAMERA_PUBLISH_HZ:-30}"
for _v in CAMERA_FPS CAMERA_PUBLISH_HZ; do
  case "${!_v}" in
    ''|0|*[!0-9]*) echo "[run] ERROR: bad $_v='${!_v}' (want a positive integer)" >&2; exit 2 ;;
  esac
done
if [ $(( CAMERA_FPS % CAMERA_PUBLISH_HZ )) -ne 0 ]; then
  echo "[run] ERROR: CAMERA_PUBLISH_HZ=$CAMERA_PUBLISH_HZ must divide CAMERA_FPS=$CAMERA_FPS (uniform count-based decimation)" >&2
  exit 2
fi
# CAMERA_PREVIEW_HZ — how often the frustum RGB is REPAINTED. This is NOT
# CAMERA_PUBLISH_HZ and the two must not be confused:
#   CAMERA_PUBLISH_HZ  the bridges' publish rate. It IS the training dataset's master
#                      timeline, must be an integer divisor of CAMERA_FPS, and must stay
#                      UNIFORM (count-based decimation). Never lower it for CPU.
#   CAMERA_PREVIEW_HZ  how often the VIZ decodes one of those frames to paint it. Display
#                      only — `cam_jpeg` has exactly two consumers and both are the two
#                      frustum paint guards, while the recorder has its own subscription
#                      and writes the bridges' JPEG bytes verbatim.
# Receiving a frame costs ~0.12 ms; painting one costs ~1.8 ms (PIL decode + viser's
# unconditional JPEG re-encode). So this is where the CPU is, and 10 Hz answers the only
# question the preview exists for — which D405 is on which hand, and is the lens clear.
# 0 = repaint every published frame (the historical behaviour).
CAMERA_PREVIEW_HZ="${CAMERA_PREVIEW_HZ:-10}"
# RENDER_HZ — scene/GUI update rate. 30 by default: the tracker bridge broadcasts at
# 90 Hz but nobody sees past 30 on a triad, the cameras publish at 30, the charts are
# throttled to 15, and the loop has NO overrun backoff — so the 16.7 ms budget 60 Hz
# leaves for ~10 ms of work (plus ~9.8 ms of mink IK in DISPLAY_MODE=humanoid) is what
# made ticks run back-to-back. Display only; see collection_viz's --render-hz comment for
# the one recording coupling (the auto-tare edge, already quantised by the recorder's
# 5 Hz status broadcast).
RENDER_HZ="${RENDER_HZ:-30}"
for _v in CAMERA_PREVIEW_HZ RENDER_HZ; do
  eval "_val=\$$_v"
  case "$_val" in
    ''|*[!0-9.]*|*.*.*) echo "[run] ERROR: bad $_v='$_val' (want a number)" >&2; exit 2 ;;
  esac
done
if [ "${RENDER_HZ%%.*}" = "0" ]; then
  echo "[run] ERROR: RENDER_HZ=$RENDER_HZ would stop the render loop entirely" >&2; exit 2
fi
GRIPPER="${GRIPPER:-1}"            # 0 = skip the gripper bridge
KEYBOARD="${KEYBOARD:-1}"          # 0 = skip the keyboard->gripper bridge
CAN_CHANNEL="${CAN_CHANNEL:-}"     # empty = prefer can-umi, then CAN_LEFT_SERIAL, then first CAN iface
CAN_LEFT_SERIAL="${CAN_LEFT_SERIAL:-0014002733354B0432333837}"  # the LEFT CANable (== 10-can-umi.link)
PREFLIGHT="${PREFLIGHT:-1}"        # 1 = resolve every removable device by stable id before spawning (0 skips)
GRIPPER_WS_PORT="${GRIPPER_WS_PORT:-8774}"
CAN_BITRATE="${CAN_BITRATE:-1000000}"
GRIPPER_OPEN="${GRIPPER_OPEN:-}"   # override the open endpoint (rad) after a power-cycle
GRIPPER_CLOSED="${GRIPPER_CLOSED:-}"
# Data recording (UMI). RECORD=1 spawns the out-of-process recorder, tells the
# viz to publish the full-rate raw F/T + show the Recording panel, and wires the
# o+p long-press episode-split marker gesture into the keyboard bridge. Recording
# is CONTINUOUS (one Start..Stop = one take); markers split it into UMI episodes
# offline. Off by default so the pure-viz workflow is unchanged.
RECORD="${RECORD:-0}"
REC_OUT="${REC_OUT:-$REPO_ROOT/data/recordings}"   # session root
FT_PUB_PORT="${FT_PUB_PORT:-8775}"                 # raw F/T publisher (viz -> recorder)
RECORDER_CTRL_PORT="${RECORDER_CTRL_PORT:-8776}"   # recorder control WS
# Bridge log directory. The tracker (1 Hz heartbeat) and camera (per-batch rate
# line) + libsurvive's own "Info: Global solve …" churn would otherwise SPAM the
# terminal at ~1-2 lines/s and BURY the interactive gates (the calibration-wave
# ENTER prompt and the go/no-go confirm) — the launch then only LOOKS hung while
# it is really just waiting on `read`. So those two bridges' stdout+stderr are
# redirected to per-bridge log files here (nothing downstream parses their
# stdout; liveness is the WS health probe). LOG_TRACKER=0 restores inline output.
LOG_DIR="${LOG_DIR:-$REPO_ROOT/data/logs}"
# PORT_PRECHECK=0 skips the "is every port I am about to bind free?" gate. Off only for
# debugging: a held port is not degraded operation, it is a stream that will not exist.
PORT_PRECHECK="${PORT_PRECHECK:-1}"
LOG_BRIDGES="${LOG_BRIDGES:-1}"                    # 0 = keep tracker/camera output inline (old behavior)

MOCK=""
NO_FT=""
for a in "$@"; do
  [ "$a" = "--mock" ] && MOCK="1"
  [ "$a" = "--no-ft" ] && NO_FT="1"   # forwarded to preflight so a missing Bota is only a WARN
done

# --- one-time re-exec under the 'input' group so the gripper keyboard bridge runs as
# a DIRECT child (no sudo, no per-bridge `sg`), which keeps it killable by the
# cleanup trap. The common case (this rig): the operator is a CONFIGURED member of
# `input` (getent shows it) but the group is not ACTIVE in this login session, so
# evdev can't read the macropad without re-acquiring it — and that is exactly how
# the keyboard silently fell back to sudo (whose wrapper then masked a dead child).
# Fires at most ONCE: guarded by _UMI_SG_REEXEC (belt) AND idempotent anyway —
# after the re-exec `id -nG` already contains `input`, so the `! ... grep -qx
# input` clause is false (suspenders). Skipped under --mock (no hardware) and
# KEYBOARD=0 (no bridge to host). If `sg` is missing or the operator is truly not
# a member, every clause fails and we fall through to the existing input-group-
# else-sudo spawn — never a hard fail. STOP THE STACK WITH Ctrl-C (or
# `kill -TERM -<pgid>`): a direct `kill -TERM <launcher-pid>` does not traverse
# `sg` to the inner bash trap (the cleanup() hardening below covers the rest).
if [ -z "${_UMI_SG_REEXEC:-}" ] && [ -z "$MOCK" ] && [ "${KEYBOARD:-1}" != "0" ] \
   && command -v sg >/dev/null 2>&1 \
   && getent group input 2>/dev/null | cut -d: -f4 | tr ',' '\n' | grep -qx "$(id -un)" \
   && ! id -nG 2>/dev/null | tr ' ' '\n' | grep -qx input; then
  echo "[run] re-exec under 'input' group (sg) so the keyboard bridge runs sudo-free (Ctrl-C to stop)" >&2
  export _UMI_SG_REEXEC=1
  exec sg input -c "$(printf '%q ' "${BASH_SOURCE[0]}" "$@")"
fi

# --- interactive startup prompts (MODE, then VIVE recalibration) --------------
# Placed AFTER the sg re-exec so they fire exactly ONCE (the re-exec re-runs the
# whole script), and BEFORE any bridge spawns so their stdin is a clean TTY.
# BOTH are skipped when the value is already set in the env (so the non-interactive
# FLoRe wrapper — `MODE=3pt bash scripts/collect_data.sh` — never blocks) or when
# stdin is not a TTY (piped/cron). Order per the collection spec: MODE first (R6),
# then the VIVE base-station recalibration y/n (R1).
if [ -z "$MODE" ]; then
  if [ -t 0 ]; then
    printf '[run] tracking MODE? [1]=1pt (left hand) / [2]=2pt (+right) / [3]=3pt (+head) / [5]=5pt (+both feet): ' >&2
    read -r _ans || _ans=1
    case "$_ans" in
      2|2pt) MODE=2pt ;;
      3|3pt) MODE=3pt ;;
      5|5pt) MODE=5pt ;;
      *)     MODE=1pt ;;
    esac
  else
    MODE=1pt   # non-interactive default (matches the historical default)
  fi
fi
case "$MODE" in
  1pt|2pt|3pt|5pt) ;;
  *) echo "[run] ERROR: bad MODE='$MODE' (want 1pt|2pt|3pt|5pt)" >&2; exit 2 ;;
esac
# 5pt needs the two FOOT serials up front. Refused HERE — before any bridge, any sudo
# prompt and any calibration wave — because the failure mode without them is not a crash:
# the tracker bridge would exit 2 thirty seconds in, and the operator would have waved for
# nothing. There is deliberately no fallback (first-seen / "the remaining pucks"): with
# five pucks on five dongles, "whichever answers next" is a coin flip between the two feet,
# and a swapped pair records a mirrored stance that nothing downstream can detect.
if [ "$MODE" = "5pt" ] && [ -z "$MOCK" ]; then
  _missing_feet=""
  [ -z "$LEFT_FOOT_SERIAL" ] && _missing_feet="LEFT_FOOT_SERIAL"
  [ -z "$RIGHT_FOOT_SERIAL" ] && _missing_feet="${_missing_feet:+$_missing_feet }RIGHT_FOOT_SERIAL"
  if [ -n "$_missing_feet" ]; then
    echo "[run] ERROR: MODE=5pt needs $_missing_feet (the defaults were edited away)." >&2
    echo "[run]   Find the serials (read-only, no wave, does not touch the rig's calib):" >&2
    echo "[run]     .venv-vive/bin/python bridges/tracker/vive_tracker_bridge.py --list-serials" >&2
    echo "[run]   Power ONE new puck at a time to tell left from right, then:" >&2
    echo "[run]     LEFT_FOOT_SERIAL=LHR-xxxxxxxx RIGHT_FOOT_SERIAL=LHR-yyyyyyyy MODE=5pt ./run_collection_viz.sh" >&2
    exit 2
  fi
fi
if [ "$MODE" = "5pt" ] && [ "$DISPLAY_MODE" = "humanoid" ]; then
  # Same refusal collection_viz.resolve_display_mode raises, made here so it costs nothing
  # instead of appearing after the bridges are up: the mink IK has hand + torso tasks only,
  # so a 5pt humanoid draws two foot triads beside legs that ignore them.
  echo "[run] ERROR: DISPLAY_MODE=humanoid is not available for MODE=5pt — the mink IK has" >&2
  echo "[run]        no foot task, so the two foot pucks would float beside legs that do" >&2
  echo "[run]        not follow them. Use the default gripper-only display (it draws all" >&2
  echo "[run]        five pucks), or MODE=3pt for the humanoid view." >&2
  exit 2
fi
echo "[run] MODE=$MODE (primary hand = LEFT; left=$LEFT_SERIAL$([ "$MODE" != 1pt ] && echo "  right=$RIGHT_SERIAL")$([ "$MODE" = 3pt ] || [ "$MODE" = 5pt ] && echo "  head=$HEAD_SERIAL")$([ "$MODE" = 5pt ] && echo "  left_foot=$LEFT_FOOT_SERIAL  right_foot=$RIGHT_FOOT_SERIAL"))"

# The tracker ROLES this MODE requires, resolved ONCE here and used by every tracker
# gate below (the calibration wave gate + the module health check). Both must ask the
# same question — "are ALL of this mode's pucks live", not "is anything streaming" —
# because a partially-locked rig records an EMPTY tracker file for the missing role and
# nothing downstream notices until conversion.
case "$MODE" in
  1pt) WANT_ROLES="left" ;;
  2pt) WANT_ROLES="left,right" ;;
  3pt) WANT_ROLES="left,right,head" ;;
  *)   WANT_ROLES="left,right,head,left_foot,right_foot" ;;
esac


# Sanity: the venvs must exist (run bootstrap.sh first), unless --mock with the
# default py3.12 venv only.
if [ ! -x "$BOTA_PY" ]; then
  echo "[run] ERROR: $BOTA_PY not found. Run ./bootstrap.sh first." >&2
  exit 1
fi

# --- pre-launch self-check board + go/no-go gate ------------------------------
# Every hardware self-check below (device preflight, VIVE calibration, gripper
# self-test, F/T self-check, per-bridge stream health, axis directions) now
# records ONE row here instead of hard-aborting on the spot. The operator then
# sees the WHOLE picture — what passed, what only warned, what failed — on a
# single board right before recording starts, and explicitly confirms go/no-go.
# A failure no longer buries the later checks (you learn the F/T is dead AND the
# camera is slow in one pass, not one relaunch each). Two safety invariants are
# preserved regardless of the gate: a gripper whose self-test FAILED is never
# energized (its teleop bridge is simply not spawned), and any blocking FAIL row
# defaults the gate to ABORT (you must type 'yes' to override).
_CHK_NAME=(); _CHK_STATUS=(); _CHK_ICON=(); _CHK_DETAIL=()
# TTY-gated ANSI colors (empty when piped/redirected so logs stay clean).
if [ -t 1 ]; then
  _C_GRN=$'\033[32m'; _C_YLW=$'\033[33m'; _C_RED=$'\033[31m'
  _C_DIM=$'\033[2m'; _C_BLD=$'\033[1m'; _C_RST=$'\033[0m'
else
  _C_GRN=; _C_YLW=; _C_RED=; _C_DIM=; _C_BLD=; _C_RST=
fi
_record() {  # $1=STATUS $2=icon $3=name $4=detail  (appends one board row)
  _CHK_STATUS+=("$1"); _CHK_ICON+=("$2"); _CHK_NAME+=("$3"); _CHK_DETAIL+=("$4")
}
record_pass() { _record OK   "✅" "$1" "$2"; }   # green  — verified good
record_warn() { _record WARN "⚠️" "$1" "$2"; }   # yellow — degraded but non-blocking
record_fail() { _record FAIL "❌" "$1" "$2"; }   # red    — blocking (gate defaults to abort)
record_skip() { _record SKIP "⏭️" "$1" "$2"; }   # dim    — intentionally not run
print_status_board() {
  local i col
  echo >&2
  echo "[run] ${_C_BLD}══════════ PRE-LAUNCH SELF-CHECK  (MODE=$MODE${RECORD:+, RECORD=$RECORD}) ══════════${_C_RST}" >&2
  for i in ${_CHK_STATUS[@]+"${!_CHK_STATUS[@]}"}; do
    case "${_CHK_STATUS[$i]}" in
      OK)   col="$_C_GRN" ;; WARN) col="$_C_YLW" ;;
      FAIL) col="$_C_RED" ;; *)    col="$_C_DIM" ;;
    esac
    printf '[run]   %s%-4s%s %-2s %-21s %s\n' \
      "$col" "${_CHK_STATUS[$i]}" "$_C_RST" "${_CHK_ICON[$i]}" \
      "${_CHK_NAME[$i]}" "${_CHK_DETAIL[$i]}" >&2
  done
  # Point the operator at the redirected bridge logs (tracker/camera/gripper/keyboard)
  # so the clean terminal doesn't hide where their live output went.
  [ "$LOG_BRIDGES" != "0" ] && [ -n "${LOG_DIR:-}" ] \
    && echo "[run]   ${_C_DIM}bridge logs: $LOG_DIR/{tracker,camera,gripper,keyboard}.log  (tail -f to watch)${_C_RST}" >&2
  echo "[run] ${_C_BLD}═══════════════════════════════════════════════════════════════════${_C_RST}" >&2
}
# The go/no-go gate: print the board, then let the operator decide. Blocking
# FAILs default to abort (type 'yes' to override); otherwise a plain [Y/n]
# confirm. Non-interactive (piped/cron): proceed unless there is a blocking FAIL,
# in which case abort unless FORCE_START=1. Exiting here fires the cleanup trap,
# so every already-spawned bridge is shut down (the gripper de-energizes).
confirm_go_no_go() {
  print_status_board
  local i n_fail=0 _go
  for i in ${_CHK_STATUS[@]+"${!_CHK_STATUS[@]}"}; do
    [ "${_CHK_STATUS[$i]}" = "FAIL" ] && n_fail=$((n_fail + 1))
  done
  if [ "$n_fail" -gt 0 ]; then
    echo "[run] ${_C_RED}${_C_BLD}✋ $n_fail blocking failure(s) above — data collection is NOT recommended.${_C_RST}" >&2
    if [ ! -t 0 ]; then
      if [ "${FORCE_START:-0}" = "1" ]; then
        echo "[run] FORCE_START=1 (non-interactive) -> starting DESPITE the failure(s)." >&2
        return 0
      fi
      echo "[run] non-interactive + blocking failure(s) -> ABORTING (set FORCE_START=1 to override)." >&2
      exit 1
    fi
    printf '[run] Type %syes%s to start ANYWAY, or press ENTER to abort: ' "$_C_BLD" "$_C_RST" >&2
    read -r _go || _go=""
    case "$_go" in
      yes|YES) echo "[run] override accepted — starting despite the failure(s)." >&2; return 0 ;;
      *)       echo "[run] aborting (spawned bridges will be shut down)." >&2; exit 1 ;;
    esac
  fi
  if [ ! -t 0 ]; then
    echo "[run] non-interactive, no blocking failures -> proceeding." >&2
    return 0
  fi
  local _prompt
  if [ "$RECORD" = "1" ]; then _prompt="Start DATA COLLECTION now? [Y/n] "
  else _prompt="Bring up the viz now? [Y/n] "; fi
  printf '[run] %s%s%s' "$_C_BLD" "$_prompt" "$_C_RST" >&2
  read -r _go || _go="y"
  case "$_go" in
    n|N|no|NO) echo "[run] aborting on operator request (spawned bridges will be shut down)." >&2; exit 0 ;;
    *)         echo "[run] confirmed — launching." >&2 ;;
  esac
}

# --- port preflight: every localhost port this launcher is about to BIND, checked
# against who already holds it, BEFORE a single bridge spawns.
#
# This is the check whose absence cost a whole session on 2026-09-10, when the deploy stack's
# robot-mounted camera bridges used the SAME default ports as collection's (8773 / 8777;
# collection now defaults to 8873 / 8877, but the gate stays — any stray holder kills a session).
# With both up, collection's two bridges opened their handheld D405s, then died two
# seconds later on `OSError: [Errno 98] address already in use` — into data/logs/camera*.log,
# which nobody was tailing. Every other gate passed, the viser page came up, and the
# session ran with NO IMAGES: the recorder refuses a take with no LEFT camera, so the
# operator's takes silently did not exist. A port clash is also the one failure a device
# preflight structurally cannot see, because it is not about devices.
#
# FAIL, not warn: there is no degraded mode here. A held port means that stream is absent
# for the whole session. The gate defaults to abort on any FAIL row, and the remedy is
# printed as the exact variable to set.
if [ "$PORT_PRECHECK" != "0" ] && [ -z "$MOCK" ]; then
  _port_owner() {  # $1=port -> "pid cmd" of whoever is LISTENING, empty if free
    ss -ltnpH 2>/dev/null | awk -v p=":$1\$" '$4 ~ p {print $NF; exit}' \
      | sed -E 's/.*pid=([0-9]+).*/\1/' | while read -r _pid; do
          [ -n "$_pid" ] && printf '%s %s' "$_pid" "$(tr '\0' ' ' < "/proc/$_pid/cmdline" 2>/dev/null | cut -c1-90)"
        done
  }
  # One entry per LINE, `name|port|remedy` — never a space-separated list: "camera L"
  # contains a space and word-splitting turned the label into "L". Only ports THIS
  # launcher binds; :8092 / :8770 / :8771 / :8784 / :8785 are the deploy stack's and are
  # none of our business.
  _busy=""
  while IFS='|' read -r _pname _pport _pvar; do
    [ -n "$_pport" ] || continue
    _own="$(_port_owner "$_pport")"
    [ -n "$_own" ] || continue
    _busy="$_busy $_pname:$_pport"
    echo "[run]   ${_C_RED}port $_pport ($_pname) is ALREADY BOUND${_C_RST} by pid ${_own%% *}: ${_own#* }" >&2
    if [ "$_pvar" = "-" ]; then
      echo "[run]     this launcher has no variable for it — stop pid ${_own%% *}, or this stream cannot exist this session" >&2
    else
      echo "[run]     move ours with ${_C_BLD}$_pvar=<free port>${_C_RST}, or stop pid ${_own%% *}" >&2
    fi
  done <<PORTS
viser page|8080|-
tracker|8772|-
camera L|$CAMERA_WS_PORT|CAMERA_WS_PORT
$([ "$MODE" != "1pt" ] && echo "camera R|$CAMERA_RIGHT_WS_PORT|CAMERA_RIGHT_WS_PORT")
gripper L|$GRIPPER_WS_PORT|GRIPPER_WS_PORT
gripper R|$GRIPPER_RIGHT_WS_PORT|GRIPPER_RIGHT_WS_PORT
$([ "$RECORD" = "1" ] && printf 'F/T pub L|%s|FT_PUB_PORT\nF/T pub R|%s|FT_PUB_RIGHT_PORT\nrecorder ctrl|%s|RECORDER_CTRL_PORT' "$FT_PUB_PORT" "$FT_PUB_RIGHT_PORT" "$RECORDER_CTRL_PORT")
PORTS
  if [ -n "$_busy" ]; then
    record_fail "Port availability" "already bound:${_busy} — that bridge would die on EADDRINUSE and its stream would be absent all session"
  else
    record_pass "Port availability" "every localhost port this launcher binds is free"
  fi
elif [ -n "$MOCK" ]; then
  # --mock binds no bridge port at all (the viz synthesizes every stream in-process), so
  # a held 8773 is irrelevant here and flagging it would teach the operator to ignore the row.
  record_skip "Port availability" "--mock binds no bridge ports"
fi

# --- device preflight: resolve every removable device by its STABLE id BEFORE the
# bridges spawn, so a USB-port move / tty renumber / sensor swap / wrong CAN adapter
# fails FAST here with the exact fix, instead of deep in the stack (where e.g. a
# wrong CAN bus surfaces as the gripper self-test's misleading "stall/dead motor").
# HARD-fails only on a missing Bota or (unless GRIPPER=0) a missing/serial-mismatched
# CAN iface; camera/tracker are warn-only (their live WS health probes are the gate).
# Skipped under --mock (no hardware) and PREFLIGHT=0. Pure stdlib, never opens CAN.
if [ -z "$MOCK" ] && [ "$PREFLIGHT" != "0" ]; then
  # Capture combined output so a FAIL becomes a board row (with the exact reason)
  # instead of a hard exit — the go/no-go gate below owns the abort decision. The
  # `|| pf_rc=$?` keeps a non-zero preflight from tripping `set -e` here.
  pf_log="$(mktemp)"; pf_rc=0
  # MODE + the RIGHT_* knobs let preflight check the second hand's devices with the SAME
  # resolution the launcher uses below (serial, never canN / "first device").
  GRIPPER="$GRIPPER" CAN_CHANNEL="$CAN_CHANNEL" NO_FT="$NO_FT" MODE="$MODE" \
    BOTA_SERIAL="${BOTA_SERIAL:-}" CAN_SERIAL="${CAN_SERIAL:-}" \
    CAMERA_SERIAL="$CAMERA_SERIAL" LEFT_SERIAL="$LEFT_SERIAL" RIGHT_SERIAL="$RIGHT_SERIAL" \
    CAN_LEFT_SERIAL="$CAN_LEFT_SERIAL" \
    FT_LEFT_SERIAL="$FT_LEFT_SERIAL" FT_RIGHT_SERIAL="$FT_RIGHT_SERIAL" RIGHT_FT="$RIGHT_FT" \
    RIGHT_CAMERA="$RIGHT_CAMERA" RIGHT_CAMERA_SERIAL="$RIGHT_CAMERA_SERIAL" \
    RIGHT_GRIPPER="$RIGHT_GRIPPER" GRIPPER_RIGHT_CHANNEL="$GRIPPER_RIGHT_CHANNEL" \
    GRIPPER_RIGHT_CAN_SERIAL="$GRIPPER_RIGHT_CAN_SERIAL" \
    "$BOTA_PY" "$REPO_ROOT/bridges/preflight.py" >"$pf_log" 2>&1 || pf_rc=$?
  cat "$pf_log" >&2   # keep the operator's live view identical to before
  if [ "$pf_rc" = 0 ]; then
    # Summarize the HARD devices (Bota + CAN) for the row; camera/tracker get their
    # own live health rows below, so don't double-report them here.
    # `[^:]*` tolerates the per-hand "(L)"/"(R)" labels; without it the whole row
    # silently degraded to "Bota=?" the moment preflight became hand-aware.
    pf_bota="$(sed -n 's/.*Bota F\/T (L)[^:]*: *//p' "$pf_log" | tail -1)"
    pf_bota_r="$(sed -n 's/.*Bota F\/T (R)[^:]*: *//p' "$pf_log" | tail -1)"
    pf_can="$(sed -n 's/.*CAN\/gripper *: *//p' "$pf_log" | tail -1)"
    pf_can_r="$(sed -n 's/.*CAN (R)[^:]*: *//p' "$pf_log" | tail -1)"
    pf_detail="Bota=${pf_bota:-?}${pf_bota_r:+ +R}${pf_can:+ · CAN=$pf_can}${pf_can_r:+ +R}"
    if grep -q '^\[preflight\] WARN:' "$pf_log"; then
      record_warn "Devices (USB/CAN)" "$pf_detail (see WARN above)"
    else
      record_pass "Devices (USB/CAN)" "$pf_detail"
    fi
  else
    pf_why="$(sed -n 's/^PREFLIGHT FAIL: *//p' "$pf_log" | head -1)"
    record_fail "Devices (USB/CAN)" "${pf_why:-preflight failed}  (fix, or PREFLIGHT=0 to skip)"
  fi
  rm -f "$pf_log"
else
  record_skip "Devices (USB/CAN)" "$([ -n "$MOCK" ] && echo '(--mock)' || echo '(PREFLIGHT=0)')"
fi

pids=()
# Parallel array of human names for each pid in `pids`, so the startup summary
# can say "tracker: up / gripper: DIED" instead of a bare PID. Appended in
# lock-step with every `pids+=(...)` below.
pnames=()
viz_pid=""
cleanup() {
  echo "[run] stopping bridges..."
  [ -n "$viz_pid" ] && kill "$viz_pid" 2>/dev/null || true
  if [ "${#pids[@]}" -gt 0 ]; then
    for i in "${!pids[@]}"; do
      p="${pids[$i]}"
      [ -z "$p" ] && continue
      # The keyboard bridge MAY have been spawned via sudo (operator not yet an
      # ACTIVE 'input' member and no `sg` re-exec). $p is then the sudo PID; a
      # plain kill won't reach its python child, which would LEAK and double-grab
      # the macropad next run. sudo-kill that one slot first; a normal kill covers
      # every direct child (incl. the keyboard once it's a direct child).
      if [ "${i}" = "${kbd_pid_idx:-}" ] && [ "${KBD_VIA_SUDO:-0}" = "1" ]; then
        sudo kill "$p" 2>/dev/null || kill "$p" 2>/dev/null || true
      else
        kill "$p" 2>/dev/null || true
      fi
    done
  fi
}
trap cleanup EXIT INT TERM

# Spawn the 2-key keyboard bridge, preferring a DIRECT child (trap-killable) when
# the input group is active (it is, after the sg re-exec above), else a sudo
# fallback. Sets $kbd_pid; KBD_VIA_SUDO=1 marks the sudo path so cleanup/restart
# can reach the python child THROUGH the sudo wrapper. Single source of the spawn
# (used for both the initial start and the health-check auto-restart).
start_keyboard_bridge() {
  # Log it like the tracker/camera/gripper bridges. The bridge traces EVERY f/g/j/k/s/e/c
  # key it acts on (which keyboard, and the intent it emitted), and that trace is the
  # only record of an intermittent "I pressed o and the gripper didn't move" — without
  # a file it scrolls off the launcher terminal and the next report has no evidence.
  # It also keeps the per-key lines from burying the go/no-go prompt. LOG_BRIDGES=0
  # restores inline output. `sudo tee` is not needed: only the python child runs as
  # root and the shell owns the redirect.
  local kbd_out=""
  # RIGHT-hand gripper keys (j/k). Passed ONLY when that bridge is actually up, so
  # the keys stay inert rather than queueing intents at a port nobody serves.
  # start_keyboard_bridge is also called by the health-check restart path, and
  # gripper_right_ok is already resolved by then, so both call sites agree.
  local kbd_right=()
  if [ "${gripper_right_ok:-0}" = "1" ]; then
    kbd_right=(--gripper-right-url "ws://127.0.0.1:$GRIPPER_RIGHT_WS_PORT")
  fi
  if [ "$LOG_BRIDGES" != "0" ] && [ -n "${LOG_DIR:-}" ]; then
    mkdir -p "$LOG_DIR"; kbd_out="$LOG_DIR/keyboard.log"
    echo "[run]   keyboard output -> $kbd_out"
    # APPEND, with a dated banner: the health check restarts this bridge into the
    # same slot, and an intermittent key fault is worth comparing across sessions,
    # so truncating would throw away the very history we want.
    { echo; echo "=== $(date -Is) keyboard_gripper_ws start ==="; } >>"$kbd_out"
  fi
  if id -nG 2>/dev/null | tr ' ' '\n' | grep -qx input; then
    echo "[run] starting keyboard_gripper_ws (evdev, input group)"
    if [ -n "$kbd_out" ]; then
      "$I2RT_PY" "$REPO_ROOT/gripper/keyboard_gripper_ws.py" \
        --gripper-url "ws://127.0.0.1:$GRIPPER_WS_PORT" "${kbd_right[@]}" "${KBD_REC_ARGS[@]}" >>"$kbd_out" 2>&1 &
    else
      "$I2RT_PY" "$REPO_ROOT/gripper/keyboard_gripper_ws.py" \
        --gripper-url "ws://127.0.0.1:$GRIPPER_WS_PORT" "${kbd_right[@]}" "${KBD_REC_ARGS[@]}" &
    fi
    kbd_pid="$!"; KBD_VIA_SUDO=0
  else
    echo "[run] starting keyboard_gripper_ws (evdev, via sudo — join 'input' group to drop sudo)"
    if [ -n "$kbd_out" ]; then
      sudo "$I2RT_PY" "$REPO_ROOT/gripper/keyboard_gripper_ws.py" \
        --gripper-url "ws://127.0.0.1:$GRIPPER_WS_PORT" "${kbd_right[@]}" "${KBD_REC_ARGS[@]}" >>"$kbd_out" 2>&1 &
    else
      sudo "$I2RT_PY" "$REPO_ROOT/gripper/keyboard_gripper_ws.py" \
        --gripper-url "ws://127.0.0.1:$GRIPPER_WS_PORT" "${kbd_right[@]}" "${KBD_REC_ARGS[@]}" &
    fi
    kbd_pid="$!"; KBD_VIA_SUDO=1
  fi
}

if [ -z "$MOCK" ]; then
  # --- pre-acquire sudo BEFORE any bridge spawns --------------------------------
  # The tracker bridge below starts streaming pose to stdout at 90 Hz the instant it
  # comes up. The FIRST privileged step after that is the CAN bring-up
  # (`sudo ip link set ... up`), whose `[sudo] password for ...:` prompt then lands
  # UNDER the tracker's pose spam — the launch LOOKS hung when it is only blocked on
  # stdin waiting for the password (the exact symptom that sent us here). So prompt
  # NOW, on a clean screen, before anything streams; the bring-up below (and the
  # keyboard evdev sudo fallback, if any) then reuse the cached credential silently.
  # Only when a privileged step is actually coming (GRIPPER!=0) AND sudo isn't already
  # cached. Non-fatal on auth failure: the CAN-ready gate below still aborts with a
  # clear reason if the bus truly can't come up.
  if [ "$GRIPPER" != "0" ] && ! sudo -n true 2>/dev/null; then
    echo "[run] ── sudo required (CAN bring-up) — enter password NOW ──"
    echo "[run]   Prompting HERE, before the tracker starts streaming at 90 Hz —"
    echo "[run]   otherwise this prompt gets BURIED under the tracker's pose output"
    echo "[run]   and the launch only LOOKS hung (it is just waiting for the password)."
    sudo -v || echo "[run] WARNING: sudo auth failed; CAN bring-up may fail below (then use GRIPPER=0)" >&2
    echo "[run] ──────────────────────────────────────────────────"
  fi

  # --- tracker bridge: the reference libsurvive stack ---------------------------------------
  # Source-built libsurvive @ $LIBSURVIVE_COMMIT + its own ctypes bindings, run by the py3.9
  # venv's interpreter (numpy + websockets from there, pysurvive from PYTHONPATH — the exact
  # env the reference runbook exports). Procedure = that runbook's steps 0-4, automated:
  #   0. kill stale holders  (pysurvive is a per-process singleton -> LIBUSB_ERROR_BUSY)
  #   1. a 28de USB device (Watchman dongle / wired tracker) is enumerated at all
  #   2. its /dev/bus/usb node is rw for us (81-vive.rules MODE=0666; libsurvive uses libusb)
  #   3. rm the calibration cache -> from-scratch OOTX + base-station solve
  #   4. start the bridge with the reference flags and BLOCK until every role streams
  if [ -x "$PIKA_PY" ]; then
    if [ ! -f "$LIBSURVIVE_DIR/bin/libsurvive.so" ] || [ ! -f "$LIBSURVIVE_DIR/bindings/python/pysurvive/__init__.py" ]; then
      echo "[run] ERROR: no source-built libsurvive at $LIBSURVIVE_DIR (need bin/libsurvive.so +" >&2
      echo "[run]        bindings/python/pysurvive). Build the pinned commit:" >&2
      echo "[run]          ./bootstrap.sh          # clone + checkout ${LIBSURVIVE_COMMIT:0:12} + make" >&2
      echo "[run]        or point LIBSURVIVE_DIR=<checkout>. The PyPI pysurvive wheel is NOT accepted." >&2
      exit 1
    fi
    _lsv_head="$(git -C "$LIBSURVIVE_DIR" rev-parse HEAD 2>/dev/null || true)"
    if [ -n "$_lsv_head" ] && [ "$_lsv_head" != "$LIBSURVIVE_COMMIT" ]; then
      echo "[run] WARNING: $LIBSURVIVE_DIR is at ${_lsv_head:0:12}, not the pin ${LIBSURVIVE_COMMIT:0:12}" >&2
    fi
    TRACKER_ENV=(env "LD_LIBRARY_PATH=$LIBSURVIVE_DIR/bin${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
                     "PYTHONPATH=$LIBSURVIVE_DIR/bindings/python${PYTHONPATH:+:$PYTHONPATH}")
    # 0. singleton. pgrep -f matches COMMAND LINES, and a wrapper shell whose command line
    #    quotes one of these names (a `bash -c "... vive_tracker_bridge.py ..."`, an agent's
    #    tool shell) is an ANCESTOR of this launcher — killing it kills us. Exclude the
    #    ancestor chain; everything else matching is a real holder of the dongles.
    _anc=" "; _p="$$"
    while [ "${_p:-1}" -gt 1 ] 2>/dev/null; do
      _anc="$_anc$_p "; _p="$(awk '{print $4}' "/proc/$_p/stat" 2>/dev/null || echo 1)"
    done
    _stale_pids=""
    for _pid in $(pgrep -f 'vive_tracker_bridge\.py|survive-cli|survive-websocketd|pose_check|vive_tracker_viser' 2>/dev/null || true); do
      case "$_anc" in *" $_pid "*) continue ;; esac
      _stale_pids="$_stale_pids $_pid"
    done
    if [ -n "$_stale_pids" ]; then
      echo "[run] killing stale libsurvive holder(s):$_stale_pids (pysurvive is a per-process singleton)" >&2
      # shellcheck disable=SC2086  # intentional word-split of the pid list
      kill -9 $_stale_pids 2>/dev/null || true
      sleep 0.5
    fi
    # 1. USB
    if ! lsusb 2>/dev/null | grep -qi ' 28de:'; then
      echo "[run] ERROR: no VIVE USB device (28de:*) enumerated. Plug the Watchman dongle(s) / wired" >&2
      echo "[run]        tracker in with a DATA cable (a charge-only cable lights the LED and never" >&2
      echo "[run]        enumerates — nothing in dmesg either). Check: lsusb | grep 28de" >&2
      exit 1
    fi
    # 2. permissions
    _noperm=""
    for _d in /sys/bus/usb/devices/*; do
      [ "$(cat "$_d/idVendor" 2>/dev/null)" = "28de" ] || continue
      _node="/dev/bus/usb/$(printf %03d "$(cat "$_d/busnum")")/$(printf %03d "$(cat "$_d/devnum")")"
      { [ -r "$_node" ] && [ -w "$_node" ]; } || _noperm="$_noperm $_node"
    done
    if [ -n "$_noperm" ]; then
      echo "[run] ERROR: VIVE USB node(s) not rw for $USER:$_noperm" >&2
      echo "[run]        81-vive.rules must be in /etc/udev/rules.d/ and re-triggered after a replug:" >&2
      echo "[run]          sudo udevadm control --reload-rules && sudo udevadm trigger --action=add --subsystem-match=usb && sudo udevadm settle" >&2
      exit 1
    fi
    # 3. cache. The bridge owns it (bridges/tracker/libsurvive_cache.py): `fresh` RESETS it
    # (the reference flow: every station re-read + re-solved); the escape hatches repair the slot-numbering
    # gap config_save leaves after a partial run before anything is reused.
    TRACKER_CALIB_EFF="$TRACKER_CALIB"
    _cache_line="$("$PIKA_PY" "$REPO_ROOT/bridges/tracker/libsurvive_cache.py" --show 2>/dev/null | head -1)"
    [ -n "$_cache_line" ] && echo "[run] base-station cache before launch: $_cache_line"
    case "$TRACKER_CALIB_EFF" in
      fresh)
        echo "[run] libsurvive: --force-calibrate 1 --force-ootx 1 --lighthouse-update-velocity 0.5 on a reset cache"
        echo "[run]   (the reference recorder flow: every station re-read + re-solved; this launch is its own world frame)"
        ;;
      ootx-only)
        echo "[run] TRACKER_CALIB=ootx-only: keeping each station's FACTORY blob, re-solving the room"
        ;;
      reuse)
        echo "[run] TRACKER_CALIB=reuse: reusing the cached blobs AND room poses — valid only if no station"
        echo "[run]   moved since the launch that solved them"
        ;;
    esac
    # 4. bridge. WORLD_RPY (full roll/pitch/yaw alignment) OVERRIDES the legacy yaw-only
    # YAW_DEG when set — rig geometry, applied at the boundary, unchanged.
    WORLD_ARG=(--yaw-deg "$YAW_DEG")
    if [ -n "$WORLD_RPY" ]; then
      # shellcheck disable=SC2206  # intentional word-split of "ROLL PITCH YAW"
      WORLD_ARG=(--world-rpy-deg $WORLD_RPY)
      echo "[run] tracker world alignment: --world-rpy-deg $WORLD_RPY (overrides YAW_DEG)"
    fi
    # LEFT-canonical serial args by mode: left (primary) is required in EVERY mode;
    # right is added for 2pt/3pt/5pt; head for 3pt/5pt; the two feet only for 5pt. The
    # bridge's --mode enforces it (it exits naming the flag if one is missing).
    MODE_ARGS=(--mode "$MODE")
    MODE_ARGS+=(${LEFT_SERIAL:+--left-serial "$LEFT_SERIAL"})
    if [ "$MODE" != "1pt" ]; then
      MODE_ARGS+=(${RIGHT_SERIAL:+--right-serial "$RIGHT_SERIAL"})
    fi
    if [ "$MODE" = "3pt" ] || [ "$MODE" = "5pt" ]; then
      MODE_ARGS+=(${HEAD_SERIAL:+--head-serial "$HEAD_SERIAL"})
    fi
    if [ "$MODE" = "5pt" ]; then
      MODE_ARGS+=(${LEFT_FOOT_SERIAL:+--left-foot-serial "$LEFT_FOOT_SERIAL"})
      MODE_ARGS+=(${RIGHT_FOOT_SERIAL:+--right-foot-serial "$RIGHT_FOOT_SERIAL"})
    fi
    echo "[run] tracker mode=$MODE  left=$LEFT_SERIAL$([ "$MODE" != 1pt ] && echo "  right=$RIGHT_SERIAL")$([ "$MODE" = 3pt ] || [ "$MODE" = 5pt ] && echo "  head=$HEAD_SERIAL")$([ "$MODE" = 5pt ] && echo "  left_foot=$LEFT_FOOT_SERIAL  right_foot=$RIGHT_FOOT_SERIAL")"
    echo "[run] starting vive_tracker_bridge: libsurvive $LIBSURVIVE_DIR (${_lsv_head:0:12})," \
         "--calib-mode $TRACKER_CALIB_EFF --lighthouse-update-velocity 0.5"
    # libsurvive's own output (OOTX decode, "Global solve" churn) goes to a log so it cannot
    # bury the status board; LOG_BRIDGES=0 keeps it inline (the reference run_umi launcher shows all of it).
    # During the startup wait the solve lines are echoed from the log regardless (below).
    if [ "$LOG_BRIDGES" != "0" ]; then
      mkdir -p "$LOG_DIR"
      TRACKER_LOG="$LOG_DIR/tracker.log"
      echo "[run]   tracker output -> $TRACKER_LOG  (tail -f to watch libsurvive)"
      "${TRACKER_ENV[@]}" "$PIKA_PY" "$REPO_ROOT/bridges/tracker/vive_tracker_bridge.py" \
          "${MODE_ARGS[@]}" "${WORLD_ARG[@]}" --startup-wait "$TRACKER_STARTUP_WAIT" \
          --calib-mode "$TRACKER_CALIB_EFF" >"$TRACKER_LOG" 2>&1 &
    else
      TRACKER_LOG=""
      "${TRACKER_ENV[@]}" "$PIKA_PY" "$REPO_ROOT/bridges/tracker/vive_tracker_bridge.py" \
          "${MODE_ARGS[@]}" "${WORLD_ARG[@]}" --startup-wait "$TRACKER_STARTUP_WAIT" \
          --calib-mode "$TRACKER_CALIB_EFF" &
    fi
    TRACKER_PID="$!"
    pids+=("$TRACKER_PID"); pnames+=("tracker")

    # the reference run_umi launcher shows libsurvive's own progress on the terminal as it happens — above all
    # "Global solve with N scenes for <LH> ... (acc err 0.0310)" (acc err = the solved station's
    # tilt against its own accelerometer; poser_mpfit.c), and "MPFIT success / status failure",
    # the OOTX decodes, the objects attaching. With the log redirect those would be invisible
    # until the FAIL verdict, so echo exactly those lines from the log for the startup wait.
    TRACKER_TAIL_PID=""
    if [ -n "${TRACKER_LOG:-}" ]; then
      _LSV_SHOW='Global solve|MPFIT (success|status failure)|Got OOTX packet|Adding tracked object|Adding lighthouse ch'
      _LSV_SHOW="$_LSV_SHOW"'|Using LH [0-9]+ as reference|Attempting to solve for|Can'"'"'t solve for position'
      _LSV_SHOW="$_LSV_SHOW"'|produced its first pose|^\[tracker\] (READY|cache|calibration mode)'
      tail --pid="$TRACKER_PID" -n +1 -F "$TRACKER_LOG" 2>/dev/null \
        | grep --line-buffered -aE "$_LSV_SHOW" | sed -u 's/^/[run]     libsurvive | /' >&2 &
      TRACKER_TAIL_PID="$!"
    fi
    _stop_tracker_tail() {   # end the echo (tail exits by itself when the bridge does)
      [ -n "${TRACKER_TAIL_PID:-}" ] || return 0
      pkill -f "^tail --pid=$TRACKER_PID -n \+1 -F " 2>/dev/null || true   # anchored: the tail itself, no shell
      kill "$TRACKER_TAIL_PID" 2>/dev/null || true
      TRACKER_TAIL_PID=""
    }

    # --- BLOCK until every role this MODE needs streams (trackers first, cameras after)
    # The wire is {"trackers": {role: pose}} with a role ABSENT until its puck locks, so the
    # probe asks for ALL of $WANT_ROLES — a 5pt run with one ankle puck asleep must not
    # pass because four stream. Probe rc 2 = bridge not up yet, rc 1 = up but roles missing
    # (it prints which). The bridge runs the SAME deadline and, on its own FATAL, prints a
    # per-puck table + VERDICT (vive_tracker_bridge.py::diagnose — off the radio / sees no
    # station / lost its pose / unassigned serial, versus the SHARED room solve) and exits
    # 1: the normal failure path here is therefore the "bridge EXITED" branch quoting that
    # verdict. The hard deadline 15 s later is only for a bridge that hangs.
    _tracker_verdict() {   # $1 = the tracker log: the last FATAL block, else the last STATUS block
      [ -n "$1" ] && [ -f "$1" ] || return 1
      local start
      start="$(grep -an '^\[tracker\] FATAL' "$1" | tail -1 | cut -d: -f1)"
      [ -n "$start" ] || start="$(grep -an '^\[tracker\] STATUS' "$1" | tail -1 | cut -d: -f1)"
      [ -n "$start" ] || return 1
      tail -n "+$start" "$1" | grep -a '^\[tracker\]' | grep -av '^\[tracker\] SUMMARY:' \
        | sed 's/^\[tracker\] */[run]     | /'
    }
    _tracker_summary() {   # $1 = the tracker log: the bridge's one-line verdict, if it wrote one
      [ -n "$1" ] && [ -f "$1" ] && grep -a '^\[tracker\] SUMMARY: ' "$1" | tail -1 | sed 's/^\[tracker\] SUMMARY: //'
    }
    echo "[run] ── VIVE trackers: waiting for $WANT_ROLES (up to ${TRACKER_STARTUP_WAIT}s) ──────"
    echo "[run]   nothing to do: keep every puck powered (solid green) in view of the base stations."
    echo "[run]   libsurvive decodes each station's OOTX blob, then solves the room from the pucks' own"
    echo "[run]   light${STATIONS:+ (STATIONS=$STATIONS powered)}; MIN_STATIONS=$MIN_STATIONS solved is the floor. The table below names"
    echo "[run]   a puck that is off / dark / mis-serialled, or says the shared room solve is the wait."
    _t0="$(date +%s)"; _tracker_ready=""; _probe_missing=""
    while :; do
      _elapsed=$(( $(date +%s) - _t0 ))
      _left=$(( TRACKER_STARTUP_WAIT - _elapsed )); [ "$_left" -lt 0 ] && _left=0
      if ! kill -0 "$TRACKER_PID" 2>/dev/null; then
        _sum="$(_tracker_summary "${TRACKER_LOG:-}")"
        if [ -n "$_sum" ]; then
          echo "[run]   ${_C_RED}${_C_BLD}VIVE trackers: not every role got a pose in ${TRACKER_STARTUP_WAIT}s — the bridge's verdict:${_C_RST}" >&2
          _tracker_verdict "$TRACKER_LOG" >&2 || true
          record_fail "VIVE trackers" "$_sum"
        else
          echo "[run]   ${_C_RED}${_C_BLD}tracker bridge (pid $TRACKER_PID) EXITED during startup${_C_RST}" >&2
          if [ -n "${TRACKER_LOG:-}" ] && [ -f "$TRACKER_LOG" ]; then
            grep -avE "Preamble found|Bad sync bit" "$TRACKER_LOG" | tail -n 8 | sed 's/^/[run]     | /' >&2 || true
          fi
          record_fail "VIVE trackers" "bridge exited during startup (see ${TRACKER_LOG:-the terminal}) — nothing can be recorded"
        fi
        break
      fi
      if [ "$_elapsed" -ge $(( TRACKER_STARTUP_WAIT + 15 )) ]; then
        _sum="$(_tracker_summary "${TRACKER_LOG:-}")"
        echo "[run]   ${_C_RED}${_C_BLD}VIVE trackers: no pose for ${_probe_missing:-$WANT_ROLES} and the bridge did not exit — its last status:${_C_RST}" >&2
        _tracker_verdict "${TRACKER_LOG:-}" >&2 || true
        record_fail "VIVE trackers" "${_sum:-role(s) ${_probe_missing:-$WANT_ROLES} produced no pose in ${TRACKER_STARTUP_WAIT}s (see ${TRACKER_LOG:-the terminal} for which puck)}"
        break
      fi
      _pt=5; [ "$_left" -gt 0 ] && [ "$_left" -lt 5 ] && _pt="$_left"
      _out=""
      if _out="$("$BOTA_PY" "$REPO_ROOT/bridges/health_probe.py" --url ws://127.0.0.1:8772 \
                   --timeout "$_pt" --require-keys "trackers:$WANT_ROLES" 2>/dev/null)"; then
        _tracker_ready=1; break
      fi
      _probe_missing="$(printf '%s' "$_out" | sed -n 's/^MISSING trackers: *//p' | tail -1)"
      echo "[run]   waiting for ${_probe_missing:-$WANT_ROLES} (remaining ${_left}s)" >&2
      if [ -n "${TRACKER_LOG:-}" ] && [ -f "$TRACKER_LOG" ]; then
        # the bridge's one-line verdict so far: which puck is dark / off the radio, or "room solve"
        grep -a '^\[tracker\] STATUS' "$TRACKER_LOG" | tail -1 | sed 's/^\[tracker\] STATUS [^|]*| /[run]     /' >&2 || true
      fi
    done
    sleep 0.3   # let the echo flush the READY / last solve line before it is stopped
    _stop_tracker_tail
    if [ -n "$_tracker_ready" ]; then
      _dt=$(( $(date +%s) - _t0 ))
      echo "[run]   ${_C_BLD}trackers READY: $WANT_ROLES (${_dt}s)${_C_RST}"
      # The bridge's READY line carries libsurvive's per-station acc err (the number the
      # reference run_umi launcher shows) and `stations positioned=N seen=M` — the two numbers the 2/3/4/6
      # station comparison reads, next to the time to READY above.
      _ready_line="$(grep -a '^\[tracker\] READY' "${TRACKER_LOG:-/dev/null}" 2>/dev/null | tail -1 || true)"
      [ -n "$_ready_line" ] && printf '%s\n' "$_ready_line" | sed 's/^\[tracker\] /[run]   /'
      _n_pos="$(printf '%s' "$_ready_line" | sed -n 's/.*stations positioned=\([0-9]*\).*/\1/p')"
      _n_seen="$(printf '%s' "$_ready_line" | sed -n 's/.*stations positioned=[0-9]* seen=\([0-9]*\).*/\1/p')"
      _st="${_n_pos:-?} solved of ${_n_seen:-?} seen${STATIONS:+ (STATIONS=$STATIONS powered)}"
      if [ -n "$_n_pos" ] && [ "$_n_pos" -lt "$MIN_STATIONS" ]; then
        record_fail "VIVE trackers" "all roles streaming after ${_dt}s but only $_st — MIN_STATIONS=$MIN_STATIONS is the floor (one station gives poses but no parallax); check the other station(s)' power and line of sight"
      elif [ -n "$STATIONS" ] && [ -n "$_n_seen" ] && [ "$_n_seen" != "$STATIONS" ]; then
        record_warn "VIVE trackers" "all roles streaming after ${_dt}s; base stations $_st — libsurvive saw a different count than you powered (it uses what the pucks could see at launch)"
      else
        record_pass "VIVE trackers" "all roles streaming after ${_dt}s; base stations $_st"
      fi
    fi

    # --- BASE-STATION GATE: the room is solved -- show the live acc err and WAIT for ENTER ----
    # acc err = |(R_lh·â)_z − 1| per station (poser_mpfit.c's own number, recomputed live by
    # the bridge from BaseStationData every --acc-report-s; 0 = the solved pose agrees with the
    # station's tilt sensor, implied tilt acos(1−acc err)). libsurvive keeps refining the
    # station poses for a while after READY, so the operator watches the numbers settle (or
    # not: a station that keeps jumping is a line-of-sight or a moving-puck problem) BEFORE
    # the launcher goes on to the gripper self-test, cameras and F/T. ENTER continues.
    # Interactive terminals only (like the MODE / calibration prompts); STATION_GATE=0 skips it.
    STATION_GATE="${STATION_GATE:-1}"
    if [ -n "$_tracker_ready" ] && [ "$STATION_GATE" = "1" ] && [ -t 0 ]; then
      echo "[run] ── base stations solved — live acc err (|up_z−1| vs each station's tilt sensor; ~0 is good) ──"
      _acc_last="$(grep -a '^\[tracker\] ACC' "${TRACKER_LOG:-/dev/null}" 2>/dev/null | tail -1 || true)"
      if [ -n "$_acc_last" ]; then
        printf '%s\n' "$_acc_last" | sed 's/^\[tracker\] /[run]     /' >&2
      else
        echo "[run]     (no ACC line from the bridge yet — it prints one every 2 s once the stations are solved)" >&2
      fi
      echo "[run]   ${_C_BLD}watch it settle, then press ENTER to continue (gripper self-test, cameras, F/T …)${_C_RST}"
      _GATE_TAIL_PID=""
      if [ -n "${TRACKER_LOG:-}" ]; then
        tail --pid="$TRACKER_PID" -n 0 -F "$TRACKER_LOG" 2>/dev/null \
          | grep --line-buffered -aE '^\[tracker\] ACC|Global solve|Position for LH' \
          | sed -u 's/^\[tracker\] /[run]     /' >&2 &
        _GATE_TAIL_PID="$!"
      fi
      read -r _gate_ans || true
      if [ -n "$_GATE_TAIL_PID" ]; then
        pkill -f "^tail --pid=$TRACKER_PID -n 0 -F " 2>/dev/null || true
        kill "$_GATE_TAIL_PID" 2>/dev/null || true
      fi
      echo "[run]   continuing."
    fi

    # --- camera bridge (.venv-vive + pyrealsense2) ---
    # CAMERA_SERIAL pins a specific D405 (else first device); CAMERA_STREAM picks
    # color (default) or left-ir for non-color models — no code edit needed.
    #
    # CAMERA_FPS / CAMERA_PUBLISH_HZ are passed EXPLICITLY rather than left to the
    # bridge's argparse defaults, because the camera timeline is the MASTER GRID of
    # the training dataset and the recording rate must be visible in this log, not
    # buried in a default. CAMERA_PUBLISH_HZ must be an integer divisor of
    # CAMERA_FPS (the bridge refuses to start otherwise): publishing is count-based
    # decimation so the recorded dt is uniform. 30/30 -> decim 1, exactly 30.000 Hz.
    #
    # DEFAULT 30 Hz (was 15) since the HL contract v3 (2026-08-07): the policy grid is
    # 30 Hz == this rate, so the observation grid and the action grid COINCIDE and the
    # converter needs no interpolation to pick "the sample nearest this step". The D405
    # already CAPTURED at 30 fps and the old default threw every second frame away.
    # Cost of the change: JPEG encode doubles (~40% -> ~80% of one core at 640x480 q80)
    # and each take is 2x the frames on disk; the F/T anti-alias average per camera step
    # halves (~15 -> ~8 samples at 231 Hz), which is why the hi-rate force window is a
    # BACKWARD window (decoupled from the camera interval) rather than a per-interval bin.
    # CAMERA_PUBLISH_HZ=15 still works and still converts; it just makes the action grid
    # denser than the image grid again.
    echo "[run] starting realsense_bridge (py3.9) ${CAMERA_FPS}fps -> ${CAMERA_PUBLISH_HZ}Hz uniform"
    # Same rationale as the tracker: the camera's per-batch rate line would also
    # scroll the gates off-screen. Redirect to a log unless LOG_BRIDGES=0.
    if [ "$LOG_BRIDGES" != "0" ]; then
      mkdir -p "$LOG_DIR"
      CAMERA_LOG="$LOG_DIR/camera.log"
      echo "[run]   camera output -> $CAMERA_LOG"
      "$PIKA_PY" "$REPO_ROOT/bridges/camera/realsense_bridge.py" \
          --port "$CAMERA_WS_PORT" \
          --fps "$CAMERA_FPS" --publish-hz "$CAMERA_PUBLISH_HZ" \
          ${CAMERA_SERIAL:+--serial "$CAMERA_SERIAL"} ${CAMERA_STREAM:+--stream "$CAMERA_STREAM"} >"$CAMERA_LOG" 2>&1 &
    else
      CAMERA_LOG=""
      "$PIKA_PY" "$REPO_ROOT/bridges/camera/realsense_bridge.py" \
          --port "$CAMERA_WS_PORT" \
          --fps "$CAMERA_FPS" --publish-hz "$CAMERA_PUBLISH_HZ" \
          ${CAMERA_SERIAL:+--serial "$CAMERA_SERIAL"} ${CAMERA_STREAM:+--stream "$CAMERA_STREAM"} &
    fi
    pids+=("$!"); pnames+=("camera")

    # --- RIGHT-hand camera bridge (2pt/3pt only) -----------------------------
    # A SECOND realsense_bridge process on its own port. One process per device: the
    # bridge holds a pyrealsense pipeline for one serial, and its pacing/health must be
    # attributable to that camera alone (each hand's grid is checked separately in
    # take_meta["hands"][role]).
    #
    # Same fps/publish rate as the LEFT camera on purpose. The LEFT grid is the master
    # timeline and camera1_rgb is paired onto it by causal SELECTION (no interpolation),
    # so a slower right camera would mean each of its frames is held across several grid
    # steps — legal, recorded in attrs camera1_pair_reused_max, but not what you want.
    #
    # Spawned only in 2pt/3pt, only when RIGHT_CAMERA=1, and only with an explicit
    # serial: "first device" would race the LEFT bridge for the same camera (three D405s
    # are attached to this box). A failure to open it is DEGRADED, never fatal.
    CAMERA_RIGHT_LOG=""
    if [ "$MODE" != "1pt" ] && [ "$RIGHT_CAMERA" = "1" ] && [ -n "$RIGHT_CAMERA_SERIAL" ]; then
      if [ "$RIGHT_CAMERA_SERIAL" = "$CAMERA_SERIAL" ]; then
        echo "[run] ERROR: RIGHT_CAMERA_SERIAL == CAMERA_SERIAL ($CAMERA_SERIAL) — the two" >&2
        echo "[run]        hands would fight over ONE camera. Set distinct serials." >&2
        exit 2
      fi
      echo "[run] starting realsense_bridge RIGHT (py3.9) serial=$RIGHT_CAMERA_SERIAL -> ws:$CAMERA_RIGHT_WS_PORT"
      if [ "$LOG_BRIDGES" != "0" ]; then
        mkdir -p "$LOG_DIR"
        CAMERA_RIGHT_LOG="$LOG_DIR/camera_right.log"
        echo "[run]   right camera output -> $CAMERA_RIGHT_LOG"
        "$PIKA_PY" "$REPO_ROOT/bridges/camera/realsense_bridge.py" \
            --port "$CAMERA_RIGHT_WS_PORT" \
            --fps "$CAMERA_FPS" --publish-hz "$CAMERA_PUBLISH_HZ" \
            --serial "$RIGHT_CAMERA_SERIAL" ${CAMERA_STREAM:+--stream "$CAMERA_STREAM"} \
            >"$CAMERA_RIGHT_LOG" 2>&1 &
      else
        "$PIKA_PY" "$REPO_ROOT/bridges/camera/realsense_bridge.py" \
            --port "$CAMERA_RIGHT_WS_PORT" \
            --fps "$CAMERA_FPS" --publish-hz "$CAMERA_PUBLISH_HZ" \
            --serial "$RIGHT_CAMERA_SERIAL" ${CAMERA_STREAM:+--stream "$CAMERA_STREAM"} &
      fi
      pids+=("$!"); pnames+=("camera_right")
    elif [ "$MODE" != "1pt" ] && [ "$RIGHT_CAMERA" = "1" ]; then
      echo "[run] RIGHT camera SKIPPED — RIGHT_CAMERA_SERIAL is empty. Set it (or"
      echo "[run]   RIGHT_CAMERA=0 to silence this); 'first device' is unsafe with 3 D405s attached."
    fi

  else
    echo "[run] WARNING: $PIKA_PY missing — skipping tracker + camera (run bootstrap.sh)" >&2
    record_warn "VIVE trackers" "tracker bridge skipped ($PIKA_PY missing — run bootstrap.sh)"
  fi

  # Resolve a SocketCAN iface name from a CANable USB serial, e.g.
  # 0014002733354B0432333837 -> can-umi. Walks /sys/class/net/*/ and reads the parent
  # USB device's serial, so it is immune to the canN enumeration order that a replug or
  # a second adapter shuffles. Prints nothing and returns 1 when no iface matches.
  _can_iface_for_serial() {
    local want="$1" net ser
    [ -n "$want" ] || return 1
    for net in /sys/class/net/*; do
      [ -e "$net/device" ] || continue
      # gs_usb: /sys/class/net/canX/device -> .../<usb-dev>/<intf>, so the serial lives
      # one level UP from the interface node.
      ser="$(cat "$(readlink -f "$net/device")/../serial" 2>/dev/null || true)"
      if [ "$ser" = "$want" ]; then basename "$net"; return 0; fi
    done
    return 1
  }

  # --- gripper bridge (.venv + i2rt + CAN), HEADLESS web-control ---
  # gripper_ok gates the ENERGIZED teleop bridge: it stays 1 only while the CAN bus
  # is up AND the self-test proves a safe open/close. A FAIL flips it to 0, records
  # a board row, and SKIPS the bridge spawn — so a stalled/absent motor is reported
  # on the board but NEVER driven with stale endpoints (the over-temp safety
  # invariant that used to be enforced by the hard `exit 1`).
  gripper_ok=1
  if [ "$GRIPPER" != "0" ]; then
    # Auto-detect the CAN interface if not pinned, then bring it up (idempotent).
    # PREFER the serial-pinned name (can-umi) created by /etc/systemd/network/
    # 10-can-umi.link — it is stable across a USB-port move and disambiguates a 2nd
    # CAN adapter (gs_usb otherwise names canN by ENUMERATION order, so the bare
    # first-CAN pick below can bind the wrong/absent bus). Falls back to the first
    # CAN iface (then can0) when the .link rename isn't applied — see docs/HARDWARE_SETUP.md.
    if [ -z "$CAN_CHANNEL" ]; then
      if ip -o link show can-umi >/dev/null 2>&1; then
        CAN_CHANNEL="can-umi"
      # Second choice: this adapter's USB SERIAL. The old fallback went straight to
      # "first CAN iface", which was safe with one adapter and is NOT safe with two —
      # it would happily hand the LEFT gripper the RIGHT hand's bus if the .link rename
      # is missing. Serial lookup gives the same answer the .link rule would.
      elif CAN_CHANNEL="$(_can_iface_for_serial "$CAN_LEFT_SERIAL")" && [ -n "$CAN_CHANNEL" ]; then
        echo "[run] CAN(L): resolved $CAN_CHANNEL by serial $CAN_LEFT_SERIAL (no can-umi rename applied)"
      else
        CAN_CHANNEL="$(ip -o link show type can 2>/dev/null | awk -F': ' 'NR==1{print $2}')"
        CAN_CHANNEL="${CAN_CHANNEL:-can0}"
        echo "[run] WARNING: CAN(L) fell back to first-iface '$CAN_CHANNEL' — with two" >&2
        echo "[run]   adapters attached this can be the WRONG hand's bus. Apply the" >&2
        echo "[run]   10-can-umi.link rename or set CAN_CHANNEL= explicitly." >&2
      fi
    fi
    if ! ip -details link show "$CAN_CHANNEL" 2>/dev/null | grep -q "state UP"; then
      echo "[run] bringing up $CAN_CHANNEL @ ${CAN_BITRATE} (sudo)"
      sudo ip link set "$CAN_CHANNEL" up type can bitrate "$CAN_BITRATE" \
        || echo "[run] WARNING: could not bring up $CAN_CHANNEL; set GRIPPER=0 to skip" >&2
    fi

    # --- CAN-ready gate: abort with a CLEAR reason if the bus is still down ----------
    # If $CAN_CHANNEL is not state UP at this point (no sudo/tty to bring it up, the
    # USB-CAN adapter unplugged, or the motor unpowered), EVERYTHING gripper-side fails
    # downstream: the self-test throws "Network is down" and its FAIL banner then blames
    # a "stall / dead motor / something held in the jaws" — burying the real cause (the
    # bus never came up) in the bring-up WARNING far above. Even with SELFTEST=0 the
    # keyboard_gripper bridge can't open the socket either. So gate HERE, before the
    # self-test, regardless of SELFTEST, with the actual fix — never a misleading abort.
    if ! ip -details link show "$CAN_CHANNEL" 2>/dev/null | grep -q "state UP"; then
      echo "[run] ──────────────────────────────────────────────────" >&2
      echo "[run] ERROR: CAN bus '$CAN_CHANNEL' is not UP — cannot talk to the gripper motor." >&2
      echo "[run]   (This is the REAL cause of any 'self-test FAILED / stall / dead motor' below.)" >&2
      echo "[run]   Bring it up, then relaunch:" >&2
      echo "[run]     sudo ip link set $CAN_CHANNEL up type can bitrate $CAN_BITRATE" >&2
      echo "[run]   Also check: USB-CAN adapter plugged in, and the DM4310 motor is POWERED." >&2
      echo "[run]   Or skip the gripper entirely: GRIPPER=0 ./run_collection_viz.sh" >&2
      echo "[run] ──────────────────────────────────────────────────" >&2
      # Record + skip (no bridge spawn) instead of aborting: the board shows the
      # real cause and the go/no-go gate decides. Nothing gripper-side is energized.
      record_fail "CAN bus / gripper (L)" "'$CAN_CHANNEL' not UP — motor unreachable (adapter/power? or GRIPPER=0)"
      gripper_ok=0
    fi

    # --- gripper self-test: auto-home + full open/close BEFORE the bridge starts ---
    # The DM4310 absolute encoder re-references on every power-cycle, so the real
    # open/closed rad endpoints DRIFT. Launching the teleop loop with stale endpoints
    # drives the jaws past the hard stop → stall → over-temp → dead motor that still
    # shows green. So on every launch we first run a SAFE auto-home (teleop kp bounded
    # by the lead clamp + stall/temp guards) that discovers the TRUE current endpoints
    # and proves one full open→close→open cycle. It OWNS the bus, then releases it
    # (motor powered down) before the bridge starts — they never run concurrently
    # (single CAN writer). On PASS we pick up the discovered endpoints (unless
    # GRIPPER_OPEN/CLOSED pin them); on FAIL we ABORT the launch (per design) with the
    # re-measure hint. SELFTEST=0 skips it (e.g. a known-good rig in a hurry); the
    # over-temp risk is then back. NOTE: the jaws MOVE during this test — keep hands
    # clear and nothing held.
    SELFTEST="${SELFTEST:-1}"
    if [ "$gripper_ok" = "0" ]; then
      # CAN never came up — skip the self-test (it would only throw "Network is down"
      # and mislead). The CAN row above already captured the real cause.
      echo "[run] gripper self-test skipped (CAN bus down — see the board)" >&2
    elif [ "$SELFTEST" != "0" ]; then
      echo "[run] ── gripper self-test (auto-home + open/close) ─────────"
      echo "[run]   the jaws will MOVE to find both stops — keep hands clear, nothing held"
      # Run the test capturing output to a temp file for both the live echo AND the
      # RESULT-line parse. CRITICAL: a FAIL exits non-zero, and under `set -eo
      # pipefail` a non-zero command-substitution pipeline would trip `set -e` AT THE
      # ASSIGNMENT and abort before the guidance block runs. So we drop `set -e`
      # around the run, read PIPESTATUS immediately, and `|| true` the grep — the
      # explicit st_rc / st_result checks below are the real gate.
      st_log="$(mktemp)"
      set +e
      PYTHONPATH="$I2RT_SRC:$GRIPPER_PKG" "$I2RT_PY" "$REPO_ROOT/gripper/gripper_selftest.py" \
        --channel "$CAN_CHANNEL" 2>&1 | tee "$st_log"
      st_rc=${PIPESTATUS[0]}
      set -e
      st_result="$(grep -F '[selftest] RESULT' "$st_log" | tail -1 || true)"
      rm -f "$st_log"
      if [ "$st_rc" != "0" ] || [ -z "$st_result" ]; then
        echo "[run] ──────────────────────────────────────────────────" >&2
        echo "[run] ERROR: gripper self-test FAILED — motor powered down, bridge NOT started." >&2
        echo "[run]   The jaws did not complete a full open/close (likely a stall, dead motor," >&2
        echo "[run]   or something held in the jaws). Clear the jaws and retry, or measure by hand:" >&2
        echo "[run]     PYTHONPATH=third_party/i2rt:gripper $I2RT_PY gripper/measure_gripper_endpoints.py --channel $CAN_CHANNEL" >&2
        echo "[run]   then relaunch with GRIPPER_OPEN=<open> GRIPPER_CLOSED=<closed>, or SELFTEST=0 to skip the test." >&2
        # Over-temp safety: DO NOT energize the teleop bridge with unproven endpoints.
        # Record + flip gripper_ok=0 (skips the spawn); the go/no-go gate decides.
        record_fail "Gripper (L) self-test" "did not complete open/close — motor NOT energized (clear jaws / measure endpoints)"
        gripper_ok=0
      else
        # Pick up the auto-homed endpoints unless the operator pinned them explicitly.
        st_open="$(printf '%s\n' "$st_result" | sed -n 's/.*open=\([-0-9.]*\).*/\1/p')"
        st_closed="$(printf '%s\n' "$st_result" | sed -n 's/.*closed=\([-0-9.]*\).*/\1/p')"
        if [ -z "$GRIPPER_OPEN" ] && [ -n "$st_open" ]; then GRIPPER_OPEN="$st_open"; fi
        if [ -z "$GRIPPER_CLOSED" ] && [ -n "$st_closed" ]; then GRIPPER_CLOSED="$st_closed"; fi
        echo "[run]   self-test PASSED — using endpoints OPEN=$GRIPPER_OPEN CLOSED=$GRIPPER_CLOSED rad"
        echo "[run] ──────────────────────────────────────────────────"
        record_pass "Gripper (L) self-test" "open/close OK — OPEN=$GRIPPER_OPEN CLOSED=$GRIPPER_CLOSED rad"
      fi
    else
      echo "[run] SELFTEST=0 -> skipping gripper auto-home/open-close test (over-temp risk if endpoints drifted)" >&2
      record_warn "Gripper (L) self-test" "SKIPPED (SELFTEST=0) — over-temp risk if endpoints drifted"
    fi

    if [ "$gripper_ok" != "0" ]; then
      echo "[run] starting keyboard_gripper --web-control (py3.12, $CAN_CHANNEL -> ws:$GRIPPER_WS_PORT)"
      # The teleop bridge prints a 1 Hz "q=… rad" status line in web-control mode,
      # which (like the tracker/camera) would bury the go/no-go prompt below. Log
      # it unless LOG_BRIDGES=0. A MOTOR FAULT still surfaces on the viz page (RED)
      # + the health probe, so nothing safety-relevant is hidden by the redirect.
      if [ "$LOG_BRIDGES" != "0" ]; then
        mkdir -p "$LOG_DIR"; GRIPPER_LOG="$LOG_DIR/gripper.log"
        echo "[run]   gripper output -> $GRIPPER_LOG"
        PYTHONPATH="$I2RT_SRC:$GRIPPER_PKG" \
          "$I2RT_PY" -m holosoma_hand_pipeline.keyboard_gripper \
            --channel "$CAN_CHANNEL" --encoder-ws-port "$GRIPPER_WS_PORT" --web-control \
            ${GRIPPER_OPEN:+--open "$GRIPPER_OPEN"} ${GRIPPER_CLOSED:+--closed "$GRIPPER_CLOSED"} >"$GRIPPER_LOG" 2>&1 &
      else
        PYTHONPATH="$I2RT_SRC:$GRIPPER_PKG" \
          "$I2RT_PY" -m holosoma_hand_pipeline.keyboard_gripper \
            --channel "$CAN_CHANNEL" --encoder-ws-port "$GRIPPER_WS_PORT" --web-control \
            ${GRIPPER_OPEN:+--open "$GRIPPER_OPEN"} ${GRIPPER_CLOSED:+--closed "$GRIPPER_CLOSED"} &
      fi
      pids+=("$!"); pnames+=("gripper")
    fi

    # --- RIGHT-hand gripper (2pt/3pt): a SECOND DM4310 on its own CAN bus ------
    # Independent of the left hand at every step — its own bus, its own self-test, its
    # own encoder endpoints, its own bridge port. A right-hand failure NEVER touches the
    # left gripper (gripper_ok stays whatever the left hand earned) because a bimanual
    # session must still be recordable with one working hand: the take then carries no
    # gripper_right.npy and the converter emits no robot1_gripper_width.
    #
    # SAME over-temp invariant as the left: the bridge is only spawned after a PASSING
    # self-test (or an explicit SELFTEST=0), so the motor is never energized with stale
    # endpoints from a previous power-cycle.
    gripper_right_ok=0
    if [ "$MODE" != "1pt" ] && [ "$RIGHT_GRIPPER" = "1" ]; then
      # Resolve the right adapter BY USB SERIAL unless pinned explicitly. Naming it
      # "can1" would be a guess: gs_usb assigns canN in enumeration order, so the right
      # adapter is can0 on this box today (the left one is renamed can-umi by the
      # systemd .link and so never takes a canN slot).
      if [ -z "$GRIPPER_RIGHT_CHANNEL" ]; then
        GRIPPER_RIGHT_CHANNEL="$(_can_iface_for_serial "$GRIPPER_RIGHT_CAN_SERIAL" || true)"
        if [ -n "$GRIPPER_RIGHT_CHANNEL" ]; then
          echo "[run] CAN(R): resolved $GRIPPER_RIGHT_CHANNEL by serial $GRIPPER_RIGHT_CAN_SERIAL"
        else
          echo "[run] RIGHT gripper: no CAN adapter with serial $GRIPPER_RIGHT_CAN_SERIAL — skipping" >&2
          echo "[run]   (adapter unplugged, or set GRIPPER_RIGHT_CHANNEL=canN / GRIPPER_RIGHT_CAN_SERIAL=)" >&2
          record_warn "Gripper (R)" "CAN adapter $GRIPPER_RIGHT_CAN_SERIAL absent — right gripper NOT recorded"
        fi
      fi
    fi
    if [ "$MODE" != "1pt" ] && [ "$RIGHT_GRIPPER" = "1" ] && [ -n "$GRIPPER_RIGHT_CHANNEL" ]; then
      if [ "$GRIPPER_RIGHT_CHANNEL" = "$CAN_CHANNEL" ]; then
        echo "[run] ERROR: GRIPPER_RIGHT_CHANNEL == CAN_CHANNEL ($CAN_CHANNEL) — two bridges" >&2
        echo "[run]        would both write that bus, breaking the single-CAN-writer rule." >&2
        exit 2
      fi
      if ! ip -details link show "$GRIPPER_RIGHT_CHANNEL" 2>/dev/null | grep -q "state UP"; then
        echo "[run] bringing up $GRIPPER_RIGHT_CHANNEL @ ${CAN_BITRATE} (sudo)"
        sudo ip link set "$GRIPPER_RIGHT_CHANNEL" up type can bitrate "$CAN_BITRATE" \
          || echo "[run] WARNING: could not bring up $GRIPPER_RIGHT_CHANNEL" >&2
      fi
      if ! ip -details link show "$GRIPPER_RIGHT_CHANNEL" 2>/dev/null | grep -q "state UP"; then
        echo "[run] RIGHT gripper: CAN bus '$GRIPPER_RIGHT_CHANNEL' is not UP — skipping" >&2
        echo "[run]   (adapter unplugged / motor unpowered / no sudo. Left hand is unaffected.)" >&2
        record_warn "Gripper (R)" "'$GRIPPER_RIGHT_CHANNEL' not UP — right gripper NOT recorded"
      elif [ "${SELFTEST:-1}" != "0" ]; then
        echo "[run] ── RIGHT gripper self-test on $GRIPPER_RIGHT_CHANNEL (auto-home + open/close) ──"
        echo "[run]   the RIGHT jaws will MOVE to find both stops — keep hands clear, nothing held"
        st_log_r="$(mktemp)"
        set +e
        PYTHONPATH="$I2RT_SRC:$GRIPPER_PKG" "$I2RT_PY" "$REPO_ROOT/gripper/gripper_selftest.py" \
          --channel "$GRIPPER_RIGHT_CHANNEL" 2>&1 | tee "$st_log_r"
        st_rc_r=${PIPESTATUS[0]}
        set -e
        st_result_r="$(grep -F '[selftest] RESULT' "$st_log_r" | tail -1 || true)"
        rm -f "$st_log_r"
        if [ "$st_rc_r" != "0" ] || [ -z "$st_result_r" ]; then
          echo "[run] RIGHT gripper self-test FAILED — motor de-energized, bridge NOT started." >&2
          echo "[run]   Left hand is unaffected; this session records no right gripper." >&2
          record_warn "Gripper (R)" "self-test failed — right motor NOT energized, no gripper_right.npy"
        else
          st_open_r="$(printf '%s\n' "$st_result_r" | sed -n 's/.*open=\([-0-9.]*\).*/\1/p')"
          st_closed_r="$(printf '%s\n' "$st_result_r" | sed -n 's/.*closed=\([-0-9.]*\).*/\1/p')"
          if [ -z "$GRIPPER_RIGHT_OPEN" ] && [ -n "$st_open_r" ]; then GRIPPER_RIGHT_OPEN="$st_open_r"; fi
          if [ -z "$GRIPPER_RIGHT_CLOSED" ] && [ -n "$st_closed_r" ]; then GRIPPER_RIGHT_CLOSED="$st_closed_r"; fi
          echo "[run]   RIGHT self-test PASSED — OPEN=$GRIPPER_RIGHT_OPEN CLOSED=$GRIPPER_RIGHT_CLOSED rad"
          record_pass "Gripper (R) self-test" "open/close OK — OPEN=$GRIPPER_RIGHT_OPEN CLOSED=$GRIPPER_RIGHT_CLOSED rad"
          gripper_right_ok=1
        fi
      else
        record_warn "Gripper (R) self-test" "SKIPPED (SELFTEST=0) — over-temp risk if endpoints drifted"
        gripper_right_ok=1
      fi
      if [ "$gripper_right_ok" = "1" ]; then
        echo "[run] starting keyboard_gripper RIGHT ($GRIPPER_RIGHT_CHANNEL -> ws:$GRIPPER_RIGHT_WS_PORT)"
        if [ "$LOG_BRIDGES" != "0" ]; then
          mkdir -p "$LOG_DIR"; GRIPPER_RIGHT_LOG="$LOG_DIR/gripper_right.log"
          echo "[run]   right gripper output -> $GRIPPER_RIGHT_LOG"
          PYTHONPATH="$I2RT_SRC:$GRIPPER_PKG" \
            "$I2RT_PY" -m holosoma_hand_pipeline.keyboard_gripper \
              --channel "$GRIPPER_RIGHT_CHANNEL" --encoder-ws-port "$GRIPPER_RIGHT_WS_PORT" --web-control \
              ${GRIPPER_RIGHT_OPEN:+--open "$GRIPPER_RIGHT_OPEN"} \
              ${GRIPPER_RIGHT_CLOSED:+--closed "$GRIPPER_RIGHT_CLOSED"} >"$GRIPPER_RIGHT_LOG" 2>&1 &
        else
          PYTHONPATH="$I2RT_SRC:$GRIPPER_PKG" \
            "$I2RT_PY" -m holosoma_hand_pipeline.keyboard_gripper \
              --channel "$GRIPPER_RIGHT_CHANNEL" --encoder-ws-port "$GRIPPER_RIGHT_WS_PORT" --web-control \
              ${GRIPPER_RIGHT_OPEN:+--open "$GRIPPER_RIGHT_OPEN"} \
              ${GRIPPER_RIGHT_CLOSED:+--closed "$GRIPPER_RIGHT_CLOSED"} &
        fi
        pids+=("$!"); pnames+=("gripper_right")
      fi
    elif [ "$MODE" != "1pt" ]; then
      record_skip "Gripper (R)" "(RIGHT_GRIPPER=0)"
    fi

    if [ "$gripper_ok" = "0" ]; then
      echo "[run] gripper teleop bridge NOT started (CAN/self-test failed above) — motor stays de-energized" >&2
      [ "$RECORD" = "1" ] && echo "[run] NOTE: no gripper bridge -> s/e episode keys unavailable; use the GUI 'Start episode (s)' / 'Stop + save episode (e)' buttons" >&2
    fi

    # --- keyboard -> gripper bridge (.venv + evdev) ---
    # HOLD o=open / p=close / release=stop. Global (any window), drops autorepeat
    # (no triple-char bug). Sends the SAME WS intents the buttons do — the gripper
    # stays the sole CAN writer. evdev needs /dev/input access: works directly in
    # the `input` group (sudo usermod -aG input $USER, then re-login), else sudo.
    # Only when the gripper bridge is actually up (nothing to bridge to otherwise).
    if [ "$gripper_ok" != "0" ] && [ "$KEYBOARD" != "0" ]; then
      # With RECORD=1, also wire the o+p long-press episode-split marker gesture to
      # the recorder control WS (the gripper stays the sole CAN writer; recorder separate).
      KBD_REC_ARGS=()
      [ "$RECORD" = "1" ] && KBD_REC_ARGS=(--recorder-url "ws://127.0.0.1:$RECORDER_CTRL_PORT")
      # --wait-for-keyboard is DEFAULT ON in the bridge (late-macropad auto-attach),
      # so it is not passed here (single source of truth). The health check below
      # restarts this bridge into the SAME pids[] slot if the keyboard isn't online.
      start_keyboard_bridge
      pids+=("$kbd_pid"); pnames+=("keyboard(2-key)")
      kbd_pid_idx=$((${#pids[@]} - 1))
    fi
  else
    echo "[run] GRIPPER=0 -> skipping gripper + keyboard bridges"
    # The s/e per-episode start/stop keys live in the keyboard bridge, which only
    # starts inside the GRIPPER!=0 block. Make the loss explicit so the operator
    # falls back to the GUI "Start episode (s)" / "Stop + save episode (e)" buttons.
    [ "$RECORD" = "1" ] && echo "[run] NOTE: GRIPPER=0 -> s/e episode keys unavailable; use the GUI 'Start episode (s)' / 'Stop + save episode (e)' buttons" >&2
  fi

  sleep 1
  # Active health check: connect to each bridge WS and confirm frames actually
  # FLOW within a timeout — vs the old `kill -0`, which a stuck sudo wrapper could
  # pass while its python child was dead (exactly today's silent keyboard
  # failure). For the gripper we additionally require kbd_online=true, which proves
  # the 2-key bridge is attached to the macropad; if it isn't, we RESTART the
  # keyboard bridge (bounded) and re-probe. Every probe is captured with `|| rc=$?`
  # so a failed/absent module never trips `set -e`. Off the --mock path entirely.
  echo "[run] ── module health check ───────────────────"
  PROBE=("$BOTA_PY" "$REPO_ROOT/bridges/health_probe.py")
  KBD_MAX_RESTARTS="${KBD_MAX_RESTARTS:-3}"

  # tracker / camera: warn-only (non-fatal; the page shows live status regardless).
  if [ -x "$PIKA_PY" ]; then
    # PER-ROLE, not just "frames are flowing". The wire format is {"trackers": {role:
    # pose}} with a role ABSENT until its puck locks, so a 3pt run with the chest puck
    # powered off streams left+right forever and would pass a bare probe — recording an
    # EMPTY tracker_head.npy that only shows up at conversion time. Name the roles this
    # MODE needs and report exactly which ones never appeared.
    rc=0; probe_out="$("${PROBE[@]}" --url "ws://127.0.0.1:8772" --timeout 8 \
      --require-keys "trackers:$WANT_ROLES" 2>/dev/null)" || rc=$?
    _missing="$(printf '%s' "$probe_out" | sed -n 's/^MISSING trackers: *//p' | tail -1)"
    if [ "$rc" = 0 ]; then
      echo "[run]   tracker            healthy (roles: $WANT_ROLES)"
      record_pass "Tracker stream" "healthy (ws:8772, roles $WANT_ROLES)"
    elif [ -n "$_missing" ]; then
      # Some roles ARE live and others never locked — the actionable case. Degraded, not
      # blocking: the operator may deliberately be recording with a puck off the radio.
      echo "[run]   tracker            MISSING role(s): $_missing (powered off / out of lighthouse view?)" >&2
      echo "[run]     -> that keypoint records NOTHING. Power the puck, or drop to a lower MODE." >&2
      record_warn "Tracker stream" "role(s) $_missing NEVER locked — that keypoint records nothing (MODE=$MODE)"
    else
      echo "[run]   tracker            no frames in 8s (will show offline on the page)" >&2
      record_warn "Tracker stream" "no frames in 8s (shows offline on the page)"
    fi
    rc=0; "${PROBE[@]}" --url "ws://127.0.0.1:$CAMERA_WS_PORT" --timeout 8 >/dev/null 2>&1 || rc=$?
    if [ "$rc" = 0 ]; then echo "[run]   camera L           healthy"; record_pass "Camera stream (L)" "healthy (ws:$CAMERA_WS_PORT)"
    else echo "[run]   camera L           no frames in 8s (D405 can enumerate slowly)" >&2
         record_warn "Camera stream (L)" "no frames in 8s (D405 enumerates slowly)"; fi
    # RIGHT camera: same warn-only discipline. A missing right camera is DEGRADED, not
    # blocking — the take records camera0 only and the converter emits no camera1_rgb.
    # Reported as ⏭️ when deliberately off so "not wired up yet" reads differently from
    # "wired up and silent", which is the whole point of the board during bring-up.
    if [ "$MODE" = "1pt" ]; then
      :   # no right hand in 1pt; nothing to report
    elif [ "$RIGHT_CAMERA" != "1" ]; then
      record_skip "Camera stream (R)" "(RIGHT_CAMERA=0)"
    elif [ -z "$RIGHT_CAMERA_SERIAL" ]; then
      record_warn "Camera stream (R)" "no RIGHT_CAMERA_SERIAL set — right camera NOT recorded"
    else
      rc=0; "${PROBE[@]}" --url "ws://127.0.0.1:$CAMERA_RIGHT_WS_PORT" --timeout 8 >/dev/null 2>&1 || rc=$?
      if [ "$rc" = 0 ]; then echo "[run]   camera R           healthy"
           record_pass "Camera stream (R)" "healthy (ws:$CAMERA_RIGHT_WS_PORT, serial $RIGHT_CAMERA_SERIAL)"
      else echo "[run]   camera R           no frames in 8s — right hand records NO camera" >&2
           record_warn "Camera stream (R)" "no frames in 8s (serial $RIGHT_CAMERA_SERIAL not attached?) — no camera1_rgb"; fi
    fi
  else
    record_warn "Camera stream (L)" "camera bridge skipped ($PIKA_PY missing)"
  fi

  if [ "$GRIPPER" != "0" ] && [ "$gripper_ok" != "0" ]; then
    rc=0; "${PROBE[@]}" --url "ws://127.0.0.1:$GRIPPER_WS_PORT" --timeout 6 >/dev/null 2>&1 || rc=$?
    # WARN, not FAIL: the load-bearing over-temp safety is already gated by the
    # gripper self-test + gripper_ok (a failed self-test never energizes the motor).
    # This stream probe reaching here means the self-test PASSED, so a "no frames in
    # 6s" is almost always a benign slow-first-frame — matching the tracker/camera/
    # keyboard rows (all warn-only) and preserving the pre-board non-fatal behavior
    # (a blocking FAIL would abort non-interactive/cron launches for a transient).
    if [ "$rc" = 0 ]; then echo "[run]   gripper            healthy"; record_pass "Gripper (L) stream" "healthy (ws:$GRIPPER_WS_PORT)"
    else echo "[run]   gripper            no frames in 6s (check CAN / motor power)" >&2
         record_warn "Gripper (L) stream" "no frames in 6s (self-test passed; check motor power if persistent)"; fi

    if [ "$KEYBOARD" != "0" ]; then
      # The macropad can enumerate seconds after launch; the bridge waits + auto-
      # attaches (default --wait-for-keyboard). Probe kbd_online; if it isn't true,
      # restart the bridge (bounded) and re-probe. The restart respawns into the
      # SAME pids[] slot so cleanup still covers the latest pid.
      # BOTH hands' connections. One keyboard bridge process serves both gripper
      # bridges (--gripper-url + --gripper-right-url), and each gripper reports its OWN
      # kbd_online. Probing only the left would pass while the RIGHT connection is dead,
      # i.e. a ✅ board with inert j/k keys — the operator finds out by pressing them at
      # a grasp. A restart re-establishes both, so the retry loop covers either failure.
      _kbd_probe() {
        "${PROBE[@]}" --url "ws://127.0.0.1:$GRIPPER_WS_PORT" --timeout 6 \
          --require kbd_online >/dev/null 2>&1 || return 1
        if [ "${gripper_right_ok:-0}" = "1" ]; then
          "${PROBE[@]}" --url "ws://127.0.0.1:$GRIPPER_RIGHT_WS_PORT" --timeout 6 \
            --require kbd_online >/dev/null 2>&1 || return 2
        fi
        return 0
      }
      _kbd_keys="f/g"
      [ "${gripper_right_ok:-0}" = "1" ] && _kbd_keys="f/g + j/k"
      tries=0
      while :; do
        rc=0
        _kbd_probe || rc=$?
        if [ "$rc" = 0 ]; then
          echo "[run]   keyboard(2-key)    healthy (kbd_online, $_kbd_keys)"
          record_pass "Keyboard (gripper keys)" "attached (kbd_online) — $_kbd_keys live"
          break
        fi
        _which_hand="LEFT (f/g)"; [ "$rc" = 2 ] && _which_hand="RIGHT (j/k)"
        if [ "$tries" -ge "$KBD_MAX_RESTARTS" ]; then
          echo "[run]   keyboard(2-key)    $_which_hand OFFLINE after $tries restart(s) — plug in the macropad; the GUI 'Start episode (s)' / 'Stop + save episode (e)' buttons still work" >&2
          record_warn "Keyboard (gripper keys)" "$_which_hand OFFLINE after $tries restart(s) — GUI Start/Stop still works"
          break
        fi
        tries=$((tries + 1))
        echo "[run]   keyboard(2-key)    $_which_hand not online — restart $tries/$KBD_MAX_RESTARTS" >&2
        # Reap the old bridge in its slot (sudo path needs sudo kill).
        old="${pids[$kbd_pid_idx]:-}"
        if [ -n "$old" ]; then
          if [ "${KBD_VIA_SUDO:-0}" = "1" ]; then sudo kill "$old" 2>/dev/null || kill "$old" 2>/dev/null || true
          else kill "$old" 2>/dev/null || true; fi
        fi
        start_keyboard_bridge
        pids[$kbd_pid_idx]="$kbd_pid"
        sleep 2
      done
    fi
  elif [ "$GRIPPER" = "0" ]; then
    record_skip "Gripper (L)" "(GRIPPER=0)"
  fi

  # --- F/T sensor self-check (R2): prove the Bota streams a finite wrench BEFORE
  # the viz claims it, then release it (single-owner sensor). Complements the
  # gripper self-test (open/close, above) so both the gripper AND the F/T are
  # verified before viser. Skipped with --no-ft or SELFCHECK=0; a hard FAIL aborts
  # the launch with the fix, like the gripper self-test. preflight already checked
  # the by-id symlink exists; THIS opens the sensor (pure-stdlib preflight can't).
  # RIGHT-hand Bota PRESENCE. The sensor itself is opened by the viz (it owns the Bota
  # HWI and republishes the full-rate wrench), so all the launcher can honestly check is
  # whether the device NODE exists — but that is exactly the check that matters, because
  # it is what decides whether the recorder is told to expect a right F/T stream at all.
  # Serial-keyed via /dev/serial/by-id/: two identical Bota sensors on ttyACM0/ttyACM1
  # swap on a re-enumeration, and a swapped pair silently mislabels every wrench.
  ft_right_ok=0
  if [ "$MODE" = "1pt" ] || [ -n "$NO_FT" ]; then
    :
  elif [ "$RIGHT_FT" != "1" ]; then
    record_skip "F/T sensor (R)" "(RIGHT_FT=0)"
  elif ! ls /dev/serial/by-id/ 2>/dev/null | grep -q "$FT_RIGHT_SERIAL"; then
    record_warn "F/T sensor (R)" "serial $FT_RIGHT_SERIAL not in /dev/serial/by-id — no robot1_ft_wrench"
  elif [ "$SELFCHECK" = "0" ]; then
    ft_right_ok=1
    record_warn "F/T sensor (R)" "device present, NOT stream-verified (SELFCHECK=0)"
  else
    # SAME streaming self-check the left hand gets — open the sensor, prove a finite
    # wrench, release it (single-owner) — rather than the node-exists check this used to
    # do. "The symlink is there" and "the sensor streams" are different facts, and the
    # operator scanning the board before pressing Y cannot tell them apart. WARN, never
    # FAIL: the right hand is optional by design, so a bad right Bota degrades the
    # session to left-only instead of blocking it (the left's check stays blocking).
    echo "[run] ── F/T sensor (R) self-check (streaming wrench) ─────"
    ftr_log="$(mktemp)"; ftr_rc=0
    "$BOTA_PY" "$REPO_ROOT/force_torque/ft_selfcheck.py" --label R \
      --serial "$FT_RIGHT_SERIAL" --timeout "${FT_SELFCHECK_TIMEOUT:-6}" 2>&1 \
      | tee "$ftr_log" || ftr_rc=${PIPESTATUS[0]}
    if [ "$ftr_rc" = 0 ]; then
      ft_right_ok=1
      ftr_detail="$(sed -n 's/.*RESULT PASS: *//p' "$ftr_log" | tail -1)"
      record_pass "F/T sensor (R)" "${ftr_detail:-streaming finite wrench} (serial $FT_RIGHT_SERIAL)"
    else
      ftr_why="$(sed -n 's/.*RESULT FAIL: *//p' "$ftr_log" | tail -1)"
      echo "[run]   RIGHT F/T self-check FAILED — session continues WITHOUT robot1_ft_wrench" >&2
      record_warn "F/T sensor (R)" "${ftr_why:-no finite wrench} — no robot1_ft_wrench (left hand unaffected)"
    fi
    rm -f "$ftr_log"
  fi
  SELFCHECK="${SELFCHECK:-1}"
  if [ -n "$NO_FT" ]; then
    record_skip "F/T sensor (L)" "(--no-ft)"
  elif [ "$SELFCHECK" = "0" ]; then
    record_skip "F/T sensor (L)" "(SELFCHECK=0)"
  else
    echo "[run] ── F/T sensor self-check (streaming wrench) ─────────"
    # Capture the RESULT line for the board; the check itself streams to the tty.
    ft_log="$(mktemp)"; ft_rc=0
    # --serial PINS the LEFT sensor. Unpinned, ft_selfcheck takes the first SORTED by-id
    # match — and 3472… sorts before 3875… — so with both handheld sensors attached this
    # check opened the RIGHT hand's Bota and reported it as the left hand's ✅.
    "$BOTA_PY" "$REPO_ROOT/force_torque/ft_selfcheck.py" --label L \
      ${FT_LEFT_SERIAL:+--serial "$FT_LEFT_SERIAL"} \
      --timeout "${FT_SELFCHECK_TIMEOUT:-6}" 2>&1 | tee "$ft_log" || ft_rc=${PIPESTATUS[0]}
    if [ "$ft_rc" = 0 ]; then
      echo "[run]   F/T self-check PASSED"
      ft_detail="$(sed -n 's/.*RESULT PASS: *//p' "$ft_log" | tail -1)"
      record_pass "F/T sensor (L)" "${ft_detail:-streaming finite wrench}"
    else
      echo "[run] ──────────────────────────────────────────────────" >&2
      echo "[run] ERROR: F/T sensor self-check FAILED." >&2
      echo "[run]   Check the Bota USB/EtherCAT connection + power, then relaunch." >&2
      echo "[run]   Skip the F/T entirely: --no-ft ./run_collection_viz.sh" >&2
      echo "[run]   Or skip only this check: SELFCHECK=0 ./run_collection_viz.sh" >&2
      echo "[run] ──────────────────────────────────────────────────" >&2
      ft_why="$(sed -n 's/.*RESULT FAIL: *//p' "$ft_log" | tail -1)"
      record_fail "F/T sensor (L)" "${ft_why:-no finite wrench} (check USB/EtherCAT + power, or --no-ft)"
    fi
    rm -f "$ft_log"
  fi

  # --- AXIS-DIRECTION check (interactive; opt-IN via AXISCHECK=1) -------------
  # Verifies that pushing the tracker FORWARD records as +x, to the robot's LEFT as
  # +y, and UP as +z — i.e. that the world alignment agrees with the humanoid's frame.
  # This is the one rig error nothing downstream can detect: every recorded pose is
  # multiplied by R_world, so a wrong alignment yields a self-consistent dataset that
  # trains a policy which reaches the wrong way. Read-only (a WS subscriber), so it
  # coexists with the running bridges — unlike --measure-frame, which needs exclusive
  # ownership of the pysurvive context. Opt-in because it needs a human to move the
  # tracker; run it whenever the rig is re-mounted or the lighthouses move, and after
  # any WORLD_RPY change.
  AXISCHECK="${AXISCHECK:-0}"
  if [ "$AXISCHECK" = "1" ]; then
    echo "[run] ── tracker axis-direction check (interactive) ──────"
    AXIS_ROLES=(left)
    [ "$MODE" != "1pt" ] && AXIS_ROLES+=(right)
    { [ "$MODE" = "3pt" ] || [ "$MODE" = "5pt" ]; } && AXIS_ROLES+=(head)
    [ "$MODE" = "5pt" ] && AXIS_ROLES+=(left_foot right_foot)
    # Pass the CURRENT alignment so the printed correction is a drop-in relaunch value.
    # Without it the tool can only print the raw delta, which is correct ONLY from an
    # identity start — and Euler triples cannot be composed by hand.
    AXIS_CUR=()
    if [ -n "$WORLD_RPY" ]; then
      AXIS_CUR=(--current-rpy-deg $WORLD_RPY)
    elif [ "${YAW_DEG:-0}" != "0" ]; then
      AXIS_CUR=(--current-rpy-deg 0 0 "$YAW_DEG")
    fi
    if "$BOTA_PY" "$REPO_ROOT/bridges/tracker/axis_check.py" \
         --roles "${AXIS_ROLES[@]}" "${AXIS_CUR[@]+"${AXIS_CUR[@]}"}" \
         --json "$REC_OUT/axis_check.json"; then
      echo "[run]   axis check PASSED"
      record_pass "Axis directions" "+x/+y/+z verified for: ${AXIS_ROLES[*]}"
    else
      echo "[run] ──────────────────────────────────────────────────" >&2
      echo "[run] ERROR: tracker axis check FAILED." >&2
      echo "[run]   Relaunch with the WORLD_RPY the check printed (it already composes" >&2
      echo "[run]   the fix onto the alignment this run used), e.g.:" >&2
      echo "[run]     WORLD_RPY=\"0 0 90\" MODE=$MODE ./run_collection_viz.sh" >&2
      echo "[run]   Then re-run the check to confirm. Skip it: AXISCHECK=0" >&2
      echo "[run] ──────────────────────────────────────────────────" >&2
      record_fail "Axis directions" "mis-mapped axis — relaunch with the printed WORLD_RPY (poisons every pose)"
    fi
  else
    record_skip "Axis directions" "(AXISCHECK=0 — run after any re-mount / lighthouse move)"
  fi
  echo "[run] ────────────────────────────────────────"

  # --- GO / NO-GO GATE ------------------------------------------------------
  # Every check above recorded a row instead of aborting; now show the whole
  # board and let the operator confirm. A blocking FAIL defaults to abort (type
  # 'yes' to override). Exiting here fires the cleanup trap, so every bridge
  # already spawned is shut down (the gripper de-energizes). Placed BEFORE the
  # recorder + viz spawn so nothing records until the operator says go.
  confirm_go_no_go
fi

# --- recorder (.venv; out-of-process, subscribes to the bridges + the viz's
# raw-F/T publisher). It writes raw per-episode files under REC_OUT; convert them
# to a UMI Zarr afterwards with processing/convert_to_umi.py.
# NOTE: under --mock the viz binds NO bridge ports (tracker/camera/gripper/F/T),
# so a recorder would capture nothing — we skip it and tell the operator. To
# rehearse the record->convert flow without hardware, use the recorder's OWN
# self-synthesizing mode instead: recording/recorder.py --mock --out <dir>. ---
if [ "$RECORD" = "1" ] && [ -z "$MOCK" ]; then
  echo "[run] starting recorder (py3.12) -> $REC_OUT  (control ws:$RECORDER_CTRL_PORT)"
  # Forward MODE + the LEFT-canonical serials so session_meta/take_meta record the
  # tracking mode + serial->role map (the recorder itself only files streams).
  REC_MODE_ARGS=(--mode "$MODE" ${LEFT_SERIAL:+--left-serial "$LEFT_SERIAL"})
  if [ "$MODE" != "1pt" ]; then
    REC_MODE_ARGS+=(${RIGHT_SERIAL:+--right-serial "$RIGHT_SERIAL"})
  fi
  if [ "$MODE" = "3pt" ] || [ "$MODE" = "5pt" ]; then
    REC_MODE_ARGS+=(${HEAD_SERIAL:+--head-serial "$HEAD_SERIAL"})
  fi
  if [ "$MODE" = "5pt" ]; then
    REC_MODE_ARGS+=(${LEFT_FOOT_SERIAL:+--left-foot-serial "$LEFT_FOOT_SERIAL"})
    REC_MODE_ARGS+=(${RIGHT_FOOT_SERIAL:+--right-foot-serial "$RIGHT_FOOT_SERIAL"})
  fi
  # Forward the WORLD ALIGNMENT as provenance. The bridge multiplies every recorded
  # pose by it, and it used to be recorded nowhere — so a session taken with
  # WORLD_RPY="0 90 0" was byte-indistinguishable from a default one and its frame was
  # unreconstructible. The recorder never transforms anything; it just files the fact.
  if [ -n "$WORLD_RPY" ]; then
    REC_MODE_ARGS+=(--world-rpy-deg $WORLD_RPY)
  elif [ "${YAW_DEG:-0}" != "0" ]; then
    REC_MODE_ARGS+=(--world-rpy-deg 0 0 "$YAW_DEG")
  fi
  # RIGHT-hand modality URLs. Passed ONLY for the devices that actually came up, because
  # to the recorder a URL is the ASSERTION "this device exists": with no URL it opens no
  # socket and the take records zero right-hand frames (which is the honest record of a
  # hand whose camera/Bota/gripper is not wired up), instead of retrying a dead port and
  # leaving an empty ft_wrench_right.npy that looks like a silent sensor.
  REC_RIGHT_ARGS=()
  if [ "$MODE" != "1pt" ]; then
    [ "$RIGHT_CAMERA" = "1" ] && [ -n "$RIGHT_CAMERA_SERIAL" ] \
      && REC_RIGHT_ARGS+=(--camera-right-url "ws://127.0.0.1:$CAMERA_RIGHT_WS_PORT")
    [ "${ft_right_ok:-0}" = "1" ] \
      && REC_RIGHT_ARGS+=(--ft-right-url "ws://127.0.0.1:$FT_PUB_RIGHT_PORT")
    [ "${gripper_right_ok:-0}" = "1" ] \
      && REC_RIGHT_ARGS+=(--gripper-right-url "ws://127.0.0.1:$GRIPPER_RIGHT_WS_PORT")
    echo "[run] recorder RIGHT-hand streams: ${REC_RIGHT_ARGS[*]:-(none — right hand records tracker only)}"
  fi
  "$BOTA_PY" "$REPO_ROOT/recording/recorder.py" \
    --out "$REC_OUT" --ctrl-port "$RECORDER_CTRL_PORT" \
    --ft-url "ws://127.0.0.1:$FT_PUB_PORT" \
    --camera-url "ws://127.0.0.1:$CAMERA_WS_PORT" \
    --gripper-url "ws://127.0.0.1:$GRIPPER_WS_PORT" \
    "${REC_RIGHT_ARGS[@]}" \
    --camera-fps "$CAMERA_FPS" "${REC_MODE_ARGS[@]}" &
  rec_pid="$!"; pids+=("$rec_pid"); pnames+=("recorder")
  # Surface a recorder that dies immediately (bad port / import error) instead of
  # leaving the panel stuck at "offline" with no hint.
  sleep 0.5
  kill -0 "$rec_pid" 2>/dev/null || echo "[run] WARNING: recorder (pid $rec_pid) exited at startup" >&2
elif [ "$RECORD" = "1" ]; then
  echo "[run] RECORD=1 + --mock: skipping recorder (no bridge ports bound under --mock)." >&2
  echo "[run]   To rehearse recording offline: recording/recorder.py --mock --out <dir>" >&2
fi

# --- viz app (.venv) ---
# Backgrounded + `wait` (NOT exec, NOT a plain foreground run): exec would replace
# this shell and destroy the cleanup trap; a plain foreground run would DEFER a
# SIGTERM behind the child. `& + wait` keeps the shell responsive so any exit path
# fires cleanup, which SIGTERMs the viz AND every bridge (the gripper's own handler
# then powers the motor down). humi_ik goes on PYTHONPATH for the mink IK import.
# With RECORD=1 the viz publishes the full-rate raw F/T + shows the Recording panel.
VIZ_REC_ARGS=()
[ "$RECORD" = "1" ] && VIZ_REC_ARGS=(--record --ft-pub-port "$FT_PUB_PORT" --recorder-url "ws://127.0.0.1:$RECORDER_CTRL_PORT")
# Forward the initial F/T compensation algorithm from the env (the GUI dropdown
# switches it live). collection_viz's --ft-algo also defaults to $FT_ALGO, so this
# is belt-and-suspenders across the sg re-exec; an explicit --ft-algo in "$@" wins.
VIZ_FT_ARGS=()
[ -n "$FT_ALGO" ] && VIZ_FT_ARGS=(--ft-algo "$FT_ALGO")
VIZ_MODE_ARGS=(--tracking-mode "$MODE" --display-mode "$DISPLAY_MODE")
[ "$RAW_FRAME" = "1" ] && VIZ_MODE_ARGS+=(--raw-frame)
# Passed EXPLICITLY rather than left to collection_viz's own CAMERA_VIEW env default,
# for the same reason as --ft-algo above: the sg re-exec is not guaranteed to carry the
# caller's environment, and a silently-ignored CAMERA_VIEW=0 looks exactly like "the
# knob does nothing". An explicit --no-camera-view in "$@" still wins (argparse
# store_true), so this cannot override a hand-passed flag.
[ "$CAMERA_VIEW" = "0" ] && VIZ_MODE_ARGS+=(--no-camera-view)
# Both passed EXPLICITLY, same reason: the sg re-exec is not guaranteed to carry the
# caller's environment, so relying on collection_viz's own env defaults would make a
# CAMERA_PREVIEW_HZ=... look ignored. An explicit flag in "$@" still wins (argparse takes
# the last occurrence).
VIZ_MODE_ARGS+=(--camera-preview-hz "$CAMERA_PREVIEW_HZ" --render-hz "$RENDER_HZ")
# The LEFT Bota is pinned by serial in EVERY mode, 1pt INCLUDED. This line used to live
# inside the right-hand block below, which made MODE=1pt pass no serial at all — and the
# viz then falls back to find_bota_port(None) = the FIRST SORTED by-id match. With both
# handheld sensors attached that is the RIGHT one (3472… sorts before 3875…), so a 1pt
# session recorded the RIGHT sensor's wrench as the left hand's `ft_wrench.npy` while the
# launcher's own left self-check — which DID pass --serial "$FT_LEFT_SERIAL" (see the F/T
# (L) block above) — reported ✅ on the correct sensor. A ✅ board over a mislabeled wrench
# is the worst shape this rig has: nothing downstream can tell the two hands apart.
# Unchanged for 2pt/3pt: same line, same value, same position ahead of --ft-right-serial.
# There is deliberately no "unpin the left sensor" escape hatch — FT_LEFT_SERIAL uses `:-`
# (unlike CAMERA_SERIAL's `-`), so an empty value takes the default back rather than
# reverting to first-device. Pass --ft-left-serial '' after the script's own args to
# override; nothing legitimate wants that with two sensors on the bus.
[ -z "$MOCK" ] && VIZ_MODE_ARGS+=(--ft-left-serial "$FT_LEFT_SERIAL")
# RIGHT-hand devices the viz should read/draw (2pt/3pt). The RIGHT Bota is read
# IN-PROCESS here exactly like the left one (it is the viz that owns the Bota HWI and
# republishes the full-rate wrench for the recorder), so the viz — not the launcher —
# is where the second sensor is opened. Serials are resolved through
# /dev/serial/by-id/ so a USB re-enumeration cannot swap the two hands.
# --mock has no hardware, so declaring right-hand devices there would draw an empty
# 'camera R' frustum and an offline 'Gripper R' row for things that cannot exist — the
# opposite of what the status rows are for. Live path only.
if [ "$MODE" != "1pt" ] && [ -z "$MOCK" ]; then
  if [ "$RIGHT_FT" = "1" ]; then
    VIZ_MODE_ARGS+=(--ft-right-serial "$FT_RIGHT_SERIAL")
    [ "$RECORD" = "1" ] && VIZ_MODE_ARGS+=(--ft-pub-right-port "$FT_PUB_RIGHT_PORT")
  fi
  [ "$RIGHT_CAMERA" = "1" ] && [ -n "$RIGHT_CAMERA_SERIAL" ] \
    && VIZ_MODE_ARGS+=(--camera-right-url "ws://127.0.0.1:$CAMERA_RIGHT_WS_PORT")
  [ "${gripper_right_ok:-0}" = "1" ] \
    && VIZ_MODE_ARGS+=(--gripper-right-url "ws://127.0.0.1:$GRIPPER_RIGHT_WS_PORT")
fi
echo "[run] starting collection_viz (py3.12) on http://localhost:8080 (display=$DISPLAY_MODE$([ "$RAW_FRAME" = 1 ] && echo ", RAW puck frame"))"
PYTHONPATH="$HUMI_SRC:${PYTHONPATH:-}" "$BOTA_PY" "$REPO_ROOT/viz/collection_viz.py" \
  --camera-url "ws://127.0.0.1:$CAMERA_WS_PORT" \
  --gripper-url "ws://127.0.0.1:$GRIPPER_WS_PORT" "${VIZ_REC_ARGS[@]}" "${VIZ_FT_ARGS[@]}" "${VIZ_MODE_ARGS[@]}" "$@" &
viz_pid="$!"
# --- session watchdog. Until the viz exits, poll every child once a second and say so
# LOUDLY when one dies. Before this the launcher sat in a bare `wait "$viz_pid"`, so a
# bridge that crashed mid-session (the tracker bridge did, 2026-09-10 00:07, ~2 min into
# 6-station streaming, with nothing in its log) showed up only as pucks frozen on the
# page — "viser lost the stream" — with no line anywhere naming the dead process. Each
# death is reported ONCE; the recorder keeps running (a take with no tracker is refused
# at STOP, not silently saved). `wait` at the end still reaps the viz for the EXIT trap.
_reported=()
while kill -0 "$viz_pid" 2>/dev/null; do
  for _i in "${!pids[@]}"; do
    _p="${pids[$_i]}"; _n="${pnames[$_i]}"
    [ -n "$_p" ] || continue
    case " ${_reported[*]:-} " in *" $_p "*) continue ;; esac
    if ! kill -0 "$_p" 2>/dev/null; then
      _reported+=("$_p")
      echo "" >&2
      echo "[run] ${_C_RED}${_C_BLD}$_n (pid $_p) EXITED mid-session${_C_RST} — its stream is GONE until you relaunch." >&2
      if [ "$_n" = "tracker" ]; then
        echo "[run]   the pucks on the page are FROZEN, not lost by the page: the libsurvive bridge died." >&2
        if [ -n "${TRACKER_LOG:-}" ] && [ -f "$TRACKER_LOG" ]; then
          echo "[run]   last lines of $TRACKER_LOG:" >&2
          grep -avE "Preamble found|Bad sync bit|streaming .* roles=" "$TRACKER_LOG" | tail -n 6 | sed 's/^/[run]     | /' >&2 || true
        fi
        echo "[run]   If it died with no Python traceback the crash was inside libsurvive's own C" >&2
        echo "[run]   thread; name it with:  sudo journalctl -k | grep segfault | tail -1" >&2
        echo "[run]   Relaunch: the tracker stack is fresh on every launch — the pucks" >&2
        echo "[run]   re-lock once they are moved in view of the base stations." >&2
        echo "[run]   Nothing recorded while it was down is usable: the recorder refuses a take with no tracker." >&2
      fi
      echo "[run]   Ctrl-C this launcher to stop the rest cleanly." >&2
    fi
  done
  sleep 1
done
wait "$viz_pid"
