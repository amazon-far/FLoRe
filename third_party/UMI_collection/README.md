# UMI_collection — FLoRe's robot-free data collection stack

The software of FLoRe's handheld data collection interface (the 3D-printable parts are in
[`UMI_Hardware.zip`](../../UMI_Hardware.zip)): a browser (**viser**) visualization of the
**data-collection rig** in one live 3D scene, plus the recorder/converter pipeline that
turns a session into the **UMI Zarr** dataset the high-level policy (`flore/`) trains on
(see [Recording a data-collection session](#recording-a-data-collection-session)).

From the FLoRe root, [`scripts/collect_data.sh`](../../scripts/collect_data.sh) is the entry
point: a thin wrapper that selects the tracking `MODE` (`1pt` / `2pt` / `3pt` / `5pt`) and
runs `run_collection_viz.sh` from this directory.

> **What is vendored here.** The parts of the rig's collection code that live collection,
> conversion, QC and replay need. Not included: the motion-completion integration (its
> model and checkpoint are not released, so the zarr viewer's `--motion-completion` panel
> reports itself unavailable) and the real-robot visualizer. Device serials, ports and
> calibration files below are those of the authors' rig; every one can be overridden
> without code edits.

What you see, live:

1. Each hand's **yam gripper cluster** (meshes, TCP / tracker / wrist triads, camera
   frustum) at its tracker pose — the default display (`DISPLAY_MODE=auto`). With
   `DISPLAY_MODE=humanoid` (1pt/2pt/3pt): a **G1 humanoid** (`g1_29dof_yam.urdf`) whose
   **right hand is driven by a single VIVE tracker** through HuMI's mink IK (right-hand frame task only; the left
   hand / feet / pelvis / base hold the `stand` pose, so the floating base never
   drifts). A **Recalibrate** button anchors your current hand pose to the
   robot's home hand.
2. **RealSense D405 RGB** on a camera frustum that **follows the gripper**.
3. **i2rt DM4310 gripper** open/close mirrored in the URDF jaws + a live encoder
   chart, driven from the **web page** (Open / Close / Stop + slider) *and* the
   **keyboard** (hold `f` = close / `g` = open; the RIGHT hand is `j` = open /
   `k` = close).
4. **Bota force/torque charts** (compensated wrench) + **Tare** and
   **Recompensation (90 s)** buttons.

In 2pt/3pt the panel is **symmetric**: each hand that is present gets its own
`Gripper (left|right)` folder (fraction, encoder rad, Open/Close/Stop + slider —
driving that hand only, never mirrored), its own `F/T L`/`F/T R` row and wrench
charts, and its own `F/T compensation (left|right)` folder with Tare, the
raw↔compensated toggle, the `bota`/`rnea` engine dropdown and Recompensation. The
two sensors share nothing: separate device, reader, payload + RNEA config and tare
offset. Only **Recompensation** is arbitrated to one hand at a time — its overlay is
a fullscreen "rotate *the* tool" prompt and its estimator needs a steady feed.
Pressing **`s`** (or Start) auto-tares **every** attached sensor.

## Architecture — hub & spoke

No single Python interpreter can hold viser + pysurvive + pyrealsense2 + CAN +
bota_driver (conflicting native-lib / Python-version needs — notably `pysurvive`
ships a **cp39-only** wheel). So each hardware singleton gets its own **bridge
process** in the venv that can talk to it, emitting localhost-WebSocket JSON, and
the viz is a thin WS **client** (the hub):

```
[vive_tracker_bridge]  ws://127.0.0.1:8772 ─┐  .venv-vive (py3.9)   right-hand SE3
[realsense_bridge]     ws://127.0.0.1:8873 ─┼─▶ viz/collection_viz   D405 RGB JPEG
[keyboard_gripper      ws://127.0.0.1:8774 ◀┤  (.venv, py3.12:       encoder rad OUT
   --web-control]                           │   viser+mink+bota)    + control IN
[keyboard_gripper_ws]  ─ws→ :8774 ──────────┘  .venv (py3.12)        f/g key intents
                                                http://localhost:8080  the scene
```

| Process | venv | Owns |
|---|---|---|
| `viz/collection_viz.py` | `.venv` (py3.12) | the scene, IK, GUI, charts, in-process Bota reader |
| `bridges/tracker/vive_tracker_bridge.py` | `.venv-vive` (py3.9 + pysurvive) | the VIVE tracker |
| `bridges/camera/realsense_bridge.py` | `.venv-vive` (py3.9 + pyrealsense2) | the D405 camera |
| `gripper/…/keyboard_gripper.py --web-control` | `.venv` (py3.12 + i2rt + CAN) | the DM4310 motor (**sole CAN writer**) |
| `gripper/keyboard_gripper_ws.py` | `.venv` (py3.12 + evdev) | the keyboard → WS intents (no CAN) |

**Invariants:** exactly one process writes CAN (the gripper); the browser and
keyboard bridge only send JSON intents; the `:8774` socket is bidirectional
(encoder OUT + control IN); all WS binds are localhost-only; the mink IK
`Configuration` is mutated only on the render thread; the Bota HWI/estimator is
touched only on the `SensorReader` thread. See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Installation (from the FLoRe checkout → first run)

Everything any `run_*.sh` needs is either pip-installed by `bootstrap.sh` or vendored as
plain files in this directory: `third_party/humi_ik/` (the mink IK stack the humanoid
display and the LeRobot replay import), `third_party/i2rt/` (the DM4310 SocketCAN driver
subset the gripper uses), `third_party/bota_demo_py/examples/rnea_dynamic_compensator_demo/`
(the RNEA F/T compensator) and `robot_assets/` (the G1+yam URDF and meshes, the Bota configs
and the rig calibration). libsurvive is built from source by `bootstrap.sh`; nothing else
needs fetching. These environments are separate from FLoRe's own `uv` environment at the
repository root — never install one into the other.

Prerequisites: Linux (tested on Ubuntu 24.04) and, for the real rig, libsurvive's build
dependencies:

```bash
sudo apt install -y libusb-1.0-0-dev liblapacke-dev libopenblas-dev cmake build-essential
```

Follow these steps in order. Steps 1–2 are all you need for the no-hardware UI;
3–6 bring up the real rig.

### 1 · Build the environments

```bash
cd third_party/UMI_collection        # from the FLoRe root
./bootstrap.sh                 # uv builds both venvs + libsurvive, seeds the Bota payload
#   ./bootstrap.sh --viz-only  # only the py3.12 venv (UI-mock; no tracker/camera hw)
```

`bootstrap.sh` is idempotent (safe to re-run), auto-installs
[uv](https://docs.astral.sh/uv/) if missing, and builds **two** venvs — no conda:

| venv | python | holds | why split |
|---|---|---|---|
| `.venv` | 3.12 | viz hub + mink IK + Bota + gripper + keyboard + **UMI/LeRobot conversion** | the main env |
| `.venv-vive` | 3.9 | RealSense camera bridge + the tracker bridge's numpy/websockets | `pyrealsense2` ships a **cp39-only** wheel. pysurvive is **not** installed here: the tracker bridge uses a **source-built** libsurvive (`~/libsurvive` @ `5588cadf`, built by `bootstrap.sh`) via `LD_LIBRARY_PATH`/`PYTHONPATH` and refuses the PyPI wheel |

Pinned dep lists live in [`env/`](env/). **libsurvive is built from source** by
`bootstrap.sh` at the pinned commit (`5588cadf`, 2026-05-14; `LIBSURVIVE_DIR` to move it;
needs `libusb-1.0-0-dev liblapacke-dev libopenblas-dev cmake build-essential`). The PyPI
`pysurvive` wheel (2023, libsurvive 1.2.7) is refused by the tracker bridge — see
[`docs/HARDWARE_SETUP.md` §4](docs/HARDWARE_SETUP.md) and the bridge's docstring for why.

### 2 · Verify with no hardware (do this first)

```bash
./run_collection_viz.sh --mock        # → http://localhost:8080  (synthetic everything)
bash scripts/collect_data.sh --mock   # the same, from the FLoRe root through the MODE wrapper
```

If the scene loads in the browser, the `.venv` + vendored IK + URDF are wired
correctly. Equivalent direct form:
`PYTHONPATH=third_party/humi_ik .venv/bin/python viz/collection_viz.py --mock`.

### 3 · One-time host setup (real hardware only)

Do these **once per machine** — full detail (and the stable-CAN rename) in
[`docs/HARDWARE_SETUP.md`](docs/HARDWARE_SETUP.md):

```bash
sudo usermod -aG input "$USER"   # evdev access for the gripper keyboard bridge — then LOG OUT and back IN
```

- **CAN bus** — the launcher auto-brings-up the interface via `sudo` on every
  run (you don't `ip link` by hand). It asks for your password **once, right at
  the start** (before the tracker begins streaming), so enter it when prompted —
  pre-cache it with `sudo -v` if you'd rather not be asked. Recommended once: pin
  the CANable to a stable `can-umi` name by its USB serial so a USB-port move
  can't misroute it ([`docs/HARDWARE_SETUP.md` §1a](docs/HARDWARE_SETUP.md)).
- Until you've re-logged-in for the `input` group, the launcher still runs the
  keyboard bridge under `sudo` (works, just noisier).

### 4 · Per-rig hardware config — env vars, no code edits

All set on the `run_collection_viz.sh` command line; the defaults already suit
this rig:

| Var | Default | Use |
|---|---|---|
| `LEFT_SERIAL` | `LHR-75106789` | LEFT (primary) VIVE tracker, all modes — **pin it** when several trackers are present (empty = first-seen, non-deterministic); `LEFT_SERIAL=LHR-xxxx` for another unit |
| `CAMERA_SERIAL` / `CAMERA_STREAM` | first device / `color` | pin a specific D405 / use `left-ir` for non-color models |
| `CAN_CHANNEL` / `CAN_SERIAL` / `CAN_BITRATE` | auto (`can-umi`→first→`can0`) / — / `1000000` | pin / require a specific CAN adapter |
| `GRIPPER_OPEN` / `GRIPPER_CLOSED` | auto-homed (step 5) | pin endpoints (rad) instead of the self-test |
| `GRIPPER=0` / `KEYBOARD=0` | both on | skip the gripper / keyboard bridge |
| `SELFTEST=0` / `PREFLIGHT=0` | all checks on | skip the gripper self-test / device preflight |
| `TRACKER_STARTUP_WAIT` | `120` (s) | how long the launch BLOCKS for every tracker role the MODE needs to stream before the cameras start (the reference recorder's `--vive-warmup-timeout` is 90 s; 2 stations lock in ~20-25 s there). A role still missing is named on the `VIVE trackers` board row and the gate defaults to abort |
| `MIN_STATIONS` | `2` | the floor: fewer base stations with a SOLVED pose at READY is a blocking ❌ (one station gives poses but no parallax) |
| `STATIONS` | unset | how many base stations you powered, for a 2/3/4/6 comparison: the board prints `solved of seen` + the time to READY and warns when the seen count differs |
| `TRACKER_CALIB` | `fresh` | escape hatch. `fresh` = the reference recorder's `--force-calibrate 1 --force-ootx 1` on a reset cache (every station re-read + re-solved, every launch). `ootx-only` keeps the factory blobs and re-solves the room; `reuse` re-derives nothing — only for a room a previous launch solved and nothing has moved since |
| `LIBSURVIVE_DIR` | `~/libsurvive` | the source-built libsurvive checkout (the pinned commit `5588cadf`) the tracker bridge runs on |

### The tracker bring-up (the reference recorder flow)

This is the flow of the reference `umi_collection` recorder
(`sensors/vive_tracker.py` + `scripts/record_session.py`), ported as-is:

1. Power **every** puck the MODE needs (solid green) and put it in view of the base
   stations. Nothing else — no wave, no hold-still, no calibration pass.
2. Launch. libsurvive starts with `--force-calibrate 1 --force-ootx 1
   --lighthouse-update-velocity 0.5` on a reset cache, so every station is re-read (OOTX) and
   **re-solved** at every process start. Each launch is its own world frame; `WORLD_RPY`
   survives because the frame is anchored on the station with the lowest hardware
   `BaseStationID`, not on discovery order.
3. The launch blocks until every role has produced a pose (`TRACKER_STARTUP_WAIT`, 120 s),
   then the cameras start. A role that never streams FAILS the launch and is named per puck
   (the bridge's table + VERDICT: off the radio / sees no station / lost its pose / serial
   mismatch, versus the shared room solve).

```bash
# from the FLoRe root
MODE=5pt RECORD=1 bash scripts/collect_data.sh                      # all five pucks on, then record
MODE=5pt STATIONS=2 GRIPPER=0 KEYBOARD=0 bash scripts/collect_data.sh   # station-count test, trackers only
```

**Base stations are the lever.** libsurvive attempts *no* room solve until every station it
has seen has decoded its OOTX blob, and the set it sees is whatever the pucks can see from
where they are at launch (it only grows). So power exactly the stations you mean to use:
two is the floor (`MIN_STATIONS=2`, blocking), and fewer lock faster. Measured here with six
powered and five pucks: 4 seen, 2 decoded, 0 solved in 150 s. `STATIONS=N` declares how many
you powered so the board can print `solved of seen` and the time to READY for a 2/3/4/6
comparison.

**Measured 2026-09-11, 2 stations (ch4/ch8) x 5 pucks lying untouched on the bench, 5 fresh launches back to back:** READY in **19.0 / 28.5 / 25.6 / 17.6 s** and one **FAIL at 120 s**. Nobody waved. The failure was ch4's OOTX decoder landing on the left_foot puck, which received **47%** of the light the other pucks did (5472 vs ~11 500 syncs): its blob restarted 8 times and never completed, while the four runs whose pins landed on well-lit pucks decoded both stations in 10-20 s. The startup table now names that puck (`gets only 47% of the light the best-placed puck gets — turn THAT puck to face the stations`) and counts a repeated preamble as a lost frame. So: no waving, but every puck must face the stations before launch — the one it decodes through is picked arbitrarily.

**Measured 2026-09-11, 4 stations (ch2/ch4/ch5/ch8) x the same 5 untouched pucks, 4 fresh launches:** **4 of 4 FAILED at 120 s**, 1 blob of 16 decoded, 32-34 lost frames per launch. Not a placement problem this time: with 4 stations powered the best puck caught only **68-71%** of each station's ~50 Hz sweeps and the worst 37-42%, where with 2 stations the well-placed pucks caught **100%**. A blob is one bit per sweep and needs ~300 consecutive bits intact, so at 70% it never completes, while the pose solver tolerates the same loss. So OOTX cannot be decoded live with 4 stations on this rig; 2 stations is what the reference flow supports. Running 4 means the factory blobs must come from the cache (`TRACKER_CALIB=ootx-only`, a ONE-TIME decode with two stations powered at a time; the room pose is still re-solved fresh every launch), or a libsurvive change that pools OOTX bits across pucks.

**3 stations (ch4/ch5/ch8), same pucks, 4 fresh launches: 1 of 4 READY (73.4 s; blobs at 25/40/70 s), 3 FAILED with 2/3, 2/3 and 0/3 blobs.** Sweep catch rate 87-94% on the best puck, 55-66% on the worst; the one success was the launch whose pins landed on the two best-lit pucks. Summary of the station-count sweep (5 pucks untouched on the bench, `--startup-wait 120`): **2 stations → 4/5 READY in 18-29 s; 3 → 1/4 in 73 s; 4 → 0/4.** The lever is the per-puck sweep catch rate (100% → ~90% → ~70% on the best puck), and a blob needs ~300 sweeps in a row.

`ootx-ignore-sync-error` is written into libsurvive's config (default on): every
completed blob is CRC32-checked, so a tolerated sync bit cannot admit a corrupt one, and it
removed the frame restarts that were stretching OOTX here.

### 5 · Calibrate

Three **independent** calibrations (full guide in
[`docs/CALIBRATION.md`](docs/CALIBRATION.md)):

1. **Bota tool payload — offline, viz must be STOPPED** (it holds the sensor's
   serial port). After mounting the tool, with nothing touching it:
   ```bash
   .venv/bin/python force_torque/estimate_payload.py --max-time 60
   ```
   On `>>> ROTATE NOW <<<`, slowly turn the sensor through many orientations
   (gravity along ±x/±y/±z) for ~15–30 s; it writes the gitignored
   `payload_compensator.json`. `bootstrap.sh` already seeds this file from the
   **example** payload, so the charts read *compensated* and the in-browser
   "Recompensation (90 s)" button is present from the first launch — but using a
   **placeholder** tool's mass/CoM. **You must run this once for your actual
   tool** (and regenerate after any tool/mount change) before trusting the
   wrench. (Delete the file to fall back to the RAW wrench — that hand's button
   is then greyed with the reason in its `Payload` line.) In 2pt/3pt the RIGHT
   hand has its own config — pin its sensor and its output file:
   ```bash
   .venv/bin/python force_torque/estimate_payload.py --max-time 60 \
     --port-path /dev/serial/by-id/*347237663335*-if00 \
     --out robot_assets/bota_config/bota_payload_utils/payload_compensator_right.json
   ```
   Once each hand has a config, both `Recompensation (90 s)` buttons work in the
   browser and each rewrites only its own hand's files.
2. **Gripper endpoints — automatic.** The startup **self-test auto-homes** both
   stops every launch (the DM4310 absolute encoder re-references on each
   power-cycle). ⚠️ **The jaws move during the self-test — keep hands clear,
   nothing held.** A FAIL **aborts the launch** (motor powered down).
3. **IK hand anchor + lighthouse — at launch.** On an interactive terminal the
   launcher first prompts the tracking **MODE** (1pt/2pt/3pt). The base stations are
   then solved **from scratch on every launch** (the libsurvive cache is
   removed and `--force-calibrate 1 --force-ootx 1` passed — there is no fresh-vs-cached
   prompt any more) and the launch **blocks** until every tracker role the MODE needs
   streams a pose (`TRACKER_STARTUP_WAIT`, 120 s). **Power every puck the MODE needs, in view
   of the stations, before launching — then nothing** (the reference recorder flow; two stations lock
   in ~20-25 s, and fewer stations lock faster). If a role never streams, the launcher prints the bridge's
   per-puck **VERDICT** (`data/logs/tracker.log`): which puck is off the radio / sees no
   station / lost its pose, versus the shared room solve — fix only what it names. While it
   waits, libsurvive's own solve lines are echoed as in the reference `run_umi` launcher (`Got OOTX packet`,
   `MPFIT success/failure`, `Global solve … (acc err …)` = each station's tilt against its
   accelerometer), and the READY line repeats the acc err per station. Then click **"Recalibrate (hold rest pose)"** in the browser
   to anchor your current hand pose to the robot's home pose.

### 6 · Launch the full stack

```bash
./run_collection_viz.sh                  # → http://localhost:8080
RECORD=1 ./run_collection_viz.sh         # + the recorder & Recording panel (see below)
```

The launcher runs these in order, failing fast with the exact fix at each gate:
re-exec into the `input` group → **device preflight** (`bridges/preflight.py`,
HARD-fails on a missing Bota or CAN) → **prompt for `sudo`** (once, up front, for
the CAN bring-up) → spawn the **tracker + camera** bridges → bring up CAN →
**gripper self-test** (auto-home + open/close) → spawn the **gripper + keyboard**
bridges → **health-check** each WebSocket → (with `RECORD=1`) the recorder → the
viz. A startup status box, then the page's live **Status** panel, show each module
online / stale / offline. Stop with **Ctrl-C** — the cleanup trap powers the
gripper motor down and kills every bridge.

> **If it seems to hang right after the tracker lines start scrolling**, it's
> almost certainly waiting for your **sudo password** — the prompt now fires
> *before* the tracker streams, but if you skipped it (or `GRIPPER=0` wasn't set
> and CAN was already down) just type the password. Pre-cache with `sudo -v`
> before launching to avoid the prompt entirely.

### 7 · The other run scripts (offline, read-only)

These replay an already-recorded session; each needs only `.venv` (plus the
vendored `humi_ik` where IK is involved) and touches no hardware (details under
[Replaying a recording](#replaying-a-recording-three-viewers)):

```bash
./run_replay_viz.sh         data/recordings/session_XXX   # raw take,        :8083  (gripper cluster + TCP path + camera + F/T)
./run_replay_zarr_viz.sh    session_XXX.zarr.zip          # converted Zarr,  :8084  (fidelity check)
./run_replay_lerobot_viz.sh data/lerobot/session_XXX      # converted LeRobot, :8085
```

## Recording a data-collection session

Add `RECORD=1` to launch the full stack **with recording** — it spawns the
out-of-process recorder (`recording/recorder.py`, a separate process so disk I/O
never GIL-starves the Bota EKF) and adds a **Recording** panel to the page:

```bash
RECORD=1 ./run_collection_viz.sh        # → http://localhost:8080
```

On startup the launcher prints a **per-module status box** so you can see, before
the page even loads, which bridges came up:

```
[run] ── module startup status ──────────────
[run]   tracker            up   (pid 12345)
[run]   camera             up   (pid 12346)
[run]   gripper            up   (pid 12347)
[run]   keyboard(2-key)    up   (pid 12348)
[run]   recorder           up   (pid 12350)
[run] ────────────────────────────────────────
```

The page's **Status** panel then shows each module's **live** online state —
`● online <Hz>` / `⚠ stale` / `○ offline (Ns)` — based on frame arrival, so a
bridge that dies mid-take flips to offline instead of looking forever-connected.
(The 2-key keyboard has no stream of its own; its liveness is relayed through the
gripper bridge as a heartbeat, so it appears in Status like the rest.)

**Manual per-episode start/stop** (one **`s`..`e`** span = exactly ONE episode,
saved as `take_NNN/`; capture stays dumb, no in-take segmentation):

1. Check the **Recording** panel reads `idle (next episode 0)`, not `○ offline`.
2. Type a **Task** name (default `pick_place`).
3. **Start an episode**: press **`s`** on the macropad (same intent as the
   **`● Start episode (s)`** button). A 2 s center-screen banner confirms it and
   capture begins — all four streams at wire rate. The status line shows live
   per-stream frame counts `trk…/grp…/ft…/cam…`; watch that **all four climb**.
4. Do the demo, then **stop+save**: press **`e`** (same as **`■ Stop + save
   episode (e)`**). A second banner fires; the episode flushes atomically (`.tmp`
   + rename; `take_meta.json` is the commit marker). **`Discard last episode`**
   throws away a botched one.
5. Repeat 3–4 for each demo. So 10 grasps = 10 × (`s` → demo → `e`).

`s`/`e` are wired through the keyboard bridge, so with `GRIPPER=0`/`KEYBOARD=0`
use the GUI **`● Start episode (s)`** / **`■ Stop + save episode (e)`** buttons
instead. (Older recordings used a continuous take split by `o+p` markers; that
model is gone — `markers.npy` is still written, but empty, and a markerless take
converts to a single episode.)

**Review the episode boundaries before converting (the safety net).** A missed
`s`/`e` keypress fuses two demos into one over-long episode (or leaves a sliver)
and shifts every later episode number. Because the conversion split is a pure,
re-runnable function of the plain `markers.npy`, any boundary error is 100%
fixable offline *before* the irreversible Zarr write:

```bash
.venv/bin/python recording/review_take.py data/recordings/session_XXX   # viser :8082
```

It draws each take's timeline (gripper width + |F| + episode boundaries +
per-episode frame/duration + auto-flagged implausible segments), reproduces the
**exact** split `convert_to_umi.py` will produce, and lets you add/move/delete
boundaries then save. Headless: `--headless` (print the flagged summary) or
`--set-markers 2.0,4.0,6.0` (write take-relative seconds).

**Convert the raw session → a UMI Zarr ReplayBuffer** (for diffusion-policy
training; runs in the same `.venv`):

```bash
.venv/bin/python processing/convert_to_umi.py data/recordings/session_XXX -o session_XXX.zarr.zip
.venv/bin/python processing/inspect_umi.py session_XXX.zarr.zip      # sanity-check the output
# a 5pt session: carry the two FOOT pucks as keypoints, and every hl_* window at H=200
.venv/bin/python processing/convert_to_umi.py data/recordings/session_XXX -o session_XXX.zarr.zip \
    --hands both_head_dual --feet recorded --hl-action-horizon 200
```

The converter never needs an `--exclude` list for a lighthouse glitch any more (2026-09-11): a
there-and-back ROTATION excursion is dropped from every tracker's stream like a bridged position
glitch (`--no-rot-deglitch` restores the old label), and every emitted role gets its own
`<role>_pose_glitch` mask (+ `pose_glitch_any`) with its `hl_<role>_action_valid` steps masked, so a
right/chest/foot glitch costs the straddling windows, not the take — only the PRIMARY hand's
unbridgeable excursions drop a take. The run ends with a per-role table (samples, position drops,
bridged/unbridged excursions, rotation excursions, frames flagged, hl steps masked) and the list of
dropped takes with reasons; the same numbers are in `attrs.glitch_stats` / `takes_dropped`.

`inspect_umi.py` is the fast numeric check (per-episode shapes, timestamps
monotone, `gripper_width` straddling the 0.08 m grasp threshold). For an actual
look at the motion, use the replay viewers below.

To instead export a **LeRobot v2.1** dataset (Psi0/GR00T layout — needed by
replay viewer 3, and numerically identical to the Zarr since both share the same
conversion core):

```bash
.venv/bin/python processing/convert_to_lerobot.py data/recordings/session_XXX \
    -o data/lerobot/session_XXX --task "pick and place"
```

No hardware? Rehearse the whole record→review→convert flow with
`recording/recorder.py --mock --out <dir>`. (The mock synthesizes a take with 2
markers → 3 episodes purely to exercise the converter's marker-split path; LIVE
capture instead uses the `s`/`e` per-episode keys above, where 0 markers ⇒ the
whole take is one episode.) The design decisions the code tags as `DECISION-*` are
summarized under [Design decisions](#design-decisions).

## Replaying a recording (three viewers)

Three **independent, offline, read-only** viser viewers (none touches the live
`RECORD=1 ./run_collection_viz.sh` pipeline — no bridges, no CAN, no writes). Use
them in this order to verify a recording before training on it:

**1. Raw-take replay** — watch what you captured in the SAME gripper-only scene the
collection viz shows for `MODE=1pt/2pt` (the three yam-gripper meshes + the three
axis triads TCP/tracker/wrist_yaw_link + the wrist camera + F/T arrows/charts + the
green TCP path the gripper rides), with a **draggable timeline** to stop at any
timestamp. No mink IK: the TCP pose is derived from the recorded puck through the
same `tracker->TCP` extrinsic the converter uses, so this matches the zarr replay
below:

```bash
./run_replay_viz.sh data/recordings/session_XXX                   # → http://localhost:8083
./run_replay_viz.sh data/recordings/session_XXX/take_002          # a specific take
./run_replay_viz.sh data/recordings/session_XXX --raw-pose        # raw world pose (skip the home-anchor)
```

**2. Zarr replay** — the **conversion-fidelity check**: replays the produced
`.zarr.zip` *exactly as stored* (no IK re-solve, so an artifact can't be masked).
**Pick an episode from the dropdown in the browser**, scrub the timeline, and
compare against the raw replay — TCP path, the stored 224×224 RGB, resampled F/T,
and `gripper_width` (with the 0.08 m grasp line):

```bash
./run_replay_zarr_viz.sh session_XXX.zarr.zip               # → http://localhost:8084
./run_replay_zarr_viz.sh session_XXX.zarr.zip --episode 2
```

**`--motion-completion` is not available in this release.** The flag stays (FLoRe's
offline inference viewer reuses the same panel), but the motion-completion model, its conda
environment and its checkpoint are not released, so the panel reports itself unavailable and
the viewer runs exactly as without the flag.

What to look for (the conversion can silently lose information — see
[`docs/REPLAY_AND_FIDELITY.md`](docs/REPLAY_AND_FIDELITY.md)): a `gripper_width`
that never crosses 0.08 m (a stale gripper calibration zeroing the signal), a TCP
path with a kink or a frozen run (a tracker dropout interpolated as motion), or
camera/pose that drift apart over an episode (time misalignment). The full table
of checks, the gripper-endpoint gotcha, and the items that still need a human
measurement are in that doc.

**3. LeRobot replay** — replays a converted **LeRobot v2.1** dataset dir
(Psi0/GR00T layout from `processing/convert_to_lerobot.py`) one episode at a
time. Unlike the Zarr viewer it puts the **full G1 humanoid** back via the same
mink IK (recovered from the stored TCP pose) with the camera on the real FK
wrist, so the image sits upright. **Pick an episode from the dropdown**; `--no-ik`
drops the robot to the stored TCP triad:

```bash
./run_replay_lerobot_viz.sh data/lerobot/session_XXX               # → http://localhost:8085
./run_replay_lerobot_viz.sh data/lerobot/session_XXX --episode 2
./run_replay_lerobot_viz.sh data/lerobot/session_XXX --no-ik       # TCP triad only
```

## Repository layout

| Path | What | README |
|---|---|---|
| `viz/` | the hub: `collection_viz.py` + `g1_right_hand_ik.py` + tests | [viz/README.md](viz/README.md) |
| `bridges/tracker/` | VIVE tracker → WS bridge | [bridges/tracker/README.md](bridges/tracker/README.md) |
| `bridges/camera/` | RealSense D405 → WS bridge | [bridges/camera/README.md](bridges/camera/README.md) |
| `gripper/` | i2rt DM4310 control + evdev keyboard bridge | [gripper/README.md](gripper/README.md) |
| `force_torque/` | Bota F/T viz + payload estimation/compensation tools | [force_torque/README.md](force_torque/README.md) |
| `recording/` `processing/` | the recorder + UMI Zarr converter/review pipeline | — |
| `replay/` | offline viser replay of a raw take + a converted `.zarr.zip` + a converted LeRobot dataset dir | — |
| `robot_assets/` | `g1_yam/` URDF + meshes, `bota_config/` driver + payload JSON | — |
| `third_party/humi_ik/` | vendored HuMI `ikumi` mink-IK stack (on `PYTHONPATH`) | — |
| `third_party/i2rt/` | vendored i2rt SocketCAN motor-driver subset (on `PYTHONPATH`) | — |
| `third_party/bota_demo_py/…/rnea_dynamic_compensator_demo/` | the pure-numpy RNEA F/T compensator (`rnea` engine) | [README](third_party/bota_demo_py/examples/rnea_dynamic_compensator_demo/README.md) |
| `env/` | pinned `requirements-*.txt` | — |
| `docs/` | architecture, calibration, hardware setup | [docs/](docs/) |

## Design decisions

Code comments refer to these by tag:

| Tag | Decision |
|---|---|
| `DECISION-A` | `episode_timestamps` are RELATIVE: each episode's first frame is t = 0 (the absolute wall clock stays in the raw take). |
| `DECISION-B` | Raw `tracker*.npy` quaternions are **WXYZ** (libsurvive); they are converted to xyzw at exactly one place, `umi_convert_core.quat_wxyz_to_xyzw`. |
| `DECISION-D` | Rig calibration defaults live in `robot_assets/calibration/` (the puck→TCP extrinsic, FK-symmetric wrists); a session's own calibration takes precedence. |
| `DECISION-E` | A take may be split into episodes by markers; with zero markers the whole take is one episode, which is what the live `s`/`e` per-episode keys produce. |
| `DECISION-H` | Gripper width is absolute metres from a 2-point encoder calibration (`gripper_range.json`, `width_open_m` = 0.095 m); the encoder endpoints come from the recording. |
| `DECISION-I` | One `.venv` runs both capture and conversion. |
| `DECISION-K` | The Zarr is the primary format; the LeRobot export is secondary. |
| `DECISION-T0` | The action grid is T0-inclusive: future step 0 lands on the anchor frame. |

## Third-party code in this directory

`third_party/humi_ik/` is from [HuMI](https://github.com/Richard-coder-Nai/HuMI) (MIT; the
G1 description inside it, like `robot_assets/g1_yam/`, derives from Unitree's G1 model,
BSD-3-Clause), `third_party/i2rt/` and the yam gripper meshes are from
[i2rt](https://github.com/i2rt-robotics/i2rt) (MIT), and `processing/imagecodecs_numcodecs.py`
is UMI's copy of the imagecodecs JPEG-XL codec (BSD-3-Clause). Notices and modification
notes: [`THIRD_PARTY_NOTICES`](../../THIRD_PARTY_NOTICES) at the FLoRe root.

## Ports & key defaults

| Thing | Value |
|---|---|
| viser scene | `http://localhost:8080` |
| tracker WS (ALL roles, one frame) | `8772` |
| camera WS — LEFT / RIGHT | `8873` / `8877` (`CAMERA_WS_PORT` / `CAMERA_RIGHT_WS_PORT`; 8773/8777 are the deploy stack's robot-mounted cameras, so both stacks can run on one box) |
| gripper WS — LEFT / RIGHT (bidirectional) | `8774` / `8778` |
| F/T republish WS — LEFT / RIGHT | `8775` / `8779` |
| recorder control WS | `8776` |
| Bota dashboard (opt-in `--dashboard`) | `8081` |
| gripper gains (i2rt) / endpoints | `kp=3.0 kd=0.3` / **auto-homed each launch** by the startup self-test (drift-corrected); defaults open `4.90`, closed `0.0` rad |
| gripper overheat guard | `--max-lead 0.6` rad (bounds stall torque to `kp·max_lead`) |
| jaw prismatic travel | `[0, 0.0475] m` (q=0 closed) |
| reach scale | `1.0` (position only; rotation 1:1) |
| camera mount offset / rpy | `(0.10, 0, 0.04) m` / `(4.7124, 1.5708, 0)` — **UNCALIBRATED, measure for your rig** |
| F/T read rate | `--ft-read-hz 250` |

## Hardware swaps (no code edits)

The stack is built to self-adapt — see [`docs/HARDWARE_SETUP.md`](docs/HARDWARE_SETUP.md):

- **VIVE tracker** — `LEFT_SERIAL` defaults to this rig's LEFT (primary) unit
  (`LHR-75106789`); set it to a different `LHR-xxxx`, or empty for first-seen latch.
  With several trackers present, first-seen is non-deterministic and can grab the wrong
  one (the arm then never moves) — pin the serial.
- **Bota F/T sensor** — auto-detected by `by-id` glob; **regenerate the payload**
  (`force_torque/estimate_payload.py`).
- **2-key keyboard (USB-C → USB-A)** — evdev auto-detects any `o`+`p` keyboard.
- **Gripper / CAN** — launcher auto-brings-up `can0`; after a motor power-cycle
  the DM4310 encoder re-references, but the **startup self-test auto-homes the new
  endpoints** (the jaws move to find both stops + verify a full open/close — keep
  hands clear). A self-test **FAIL aborts the launch** (motor powered down). Pin
  endpoints with `GRIPPER_OPEN=… GRIPPER_CLOSED=…`, or skip with `SELFTEST=0`. See
  [`gripper/README.md`](gripper/README.md).
- **RealSense camera** — PINNED by serial (`CAMERA_SERIAL`, default `323622272456` =
  LEFT; `RIGHT_CAMERA_SERIAL`, default `353322270760` = RIGHT); `--stream left-ir` for
  non-default models. Set `CAMERA_SERIAL=` (empty) for the old first-device pick — but
  with two D405s attached that is enumeration order, so the LEFT bridge can open the
  RIGHT camera. Which physical camera is on which hand cannot be auto-detected — the viz
  labels the frustums `camera L` (cyan) / `camera R` (orange) so you can confirm it in
  five seconds.

### Two hands (MODE=2pt / 3pt / 5pt)

`MODE` selects the KEYPOINTS, and 2pt/3pt/5pt bring up the RIGHT hand's own camera / Bota /
gripper alongside its tracker:

| | tracker | camera (D405) | Bota F/T | gripper (CANable serial) |
|---|---|---|---|---|
| **LEFT** (primary) | `LHR-75106789` | `323622272456` | `3875355D3532` | `can-umi` (`0014002733354B0432333837`) |
| **RIGHT** | `LHR-60CDD21F` | `353322270760` | `347237663335` | by serial `001D005433354B0332333837` |
| **CHEST** (3pt) | `LHR-3DC276BA` | — | — | — |
| **LEFT FOOT** (5pt) | `LHR-C68495D2` (dongle `E296900B95`) | — | — | — |
| **RIGHT FOOT** (5pt) | `LHR-FF1D3DE2` (dongle `3641D31EB2`) | — | — | — |

Every one is resolved by a STABLE id (serial / by-id), never by enumeration order —
`bridges/preflight.py` prints all six before the bridges spawn. These are the HANDHELD
units, not holosoma's robot-mounted cameras/Botas.

The CHEST point is a *body reference* — no camera, F/T or gripper — and is the third
point of holosoma's `exp:g1-wbt-3pt-hybrid-unified[-anneal]` (`torso_link` + 0.35 z).

**MODE=5pt** adds the 4th and 5th keypoints, the two FEET, as two more body references
handled by the chest's exact code paths (`recording_format.BODY_ROLES`): a puck each,
`tracker_{left,right}_foot.npy`, a bare axis triad at the frame origin with no TCP lever,
the chest's rotation-only puck→body relabel *shared by declaration* (the ankle brackets
hold the pucks like the chest one — `collection_viz._BODY_RELABEL_M` is the one place to
fix if that stops being true), and no camera/F-T/gripper. Roles are `left_foot` /
`right_foot`, FLoRe's own five-keypoint names. The two pucks were paired 2026-09-09 with
`bridges/tracker/pair_puck.sh` and assigned by the operator — LEFT `LHR-C68495D2`, RIGHT
`LHR-FF1D3DE2` — and are the launcher's defaults like the other three (override with
`LEFT_FOOT_SERIAL=` / `RIGHT_FOOT_SERIAL=`; a wrong one records a mirrored stance that
passes every gate, so re-check with `--list-serials` after swapping a puck). 5pt is gripper-only (`DISPLAY_MODE=humanoid` is refused: the mink IK has no
foot task). `convert_to_umi.py --feet recorded` emits the two feet as body-reference keypoints (`{left,right}_foot_pos/_rot_axis_angle` + `hl_{left,right}_foot_action_pose/_valid`, raw puck axes unless `calibration/foot_extrinsic.json` exists); FLoRe's `fivepoint_*` layout does not train on them yet.

**Every RIGHT-hand device is optional; its absence is recorded as a fact, not an error.**
A 3pt session taken before the right camera/Bota/gripper are wired up still produces
valid takes (`tracker_right.npy`, no `camera_right/`), each missing device is ⚠️ degraded
on the status board, and the converter emits no `camera1_rgb` / `robot1_ft_wrench`.
Nothing is zero-filled. Skip one deliberately with `RIGHT_CAMERA=0` / `RIGHT_FT=0` /
`RIGHT_GRIPPER=0`.

Two things worth knowing:

- **Every removable device is pinned by serial, not by order.** Bota
  (`FT_LEFT_SERIAL`/`FT_RIGHT_SERIAL`: `by-id` sorts by serial and `3472…` precedes
  `3875…`), camera (`CAMERA_SERIAL`/`RIGHT_CAMERA_SERIAL`: no more "first device"), and
  the right gripper's CANable (`GRIPPER_RIGHT_CAN_SERIAL`: `gs_usb` names adapters
  `canN` in enumeration order). An unpinned pick hands a hand the other hand's device.
- **Gripper keys are independent, not mirrored**: `f`/`g` = LEFT jaws (close/open),
  `j`/`k` = RIGHT jaws (open/close).
  Mirroring would close the idle hand on every grasp and record a task-irrelevant
  `robot1_gripper_width` a policy learns to reproduce.

Both replay viewers draw whatever the take/zarr contains: the right hand gets the same
gripper cluster + camera frustum + wrench arrows as the left, the chest gets a triad, and
a "Keypoints" row spells out which modalities are actually present. Regression:
`processing/test_dual_hand_3pt.py`.

## Tests

```bash
PY=.venv/bin/python
PYTHONPATH=third_party/humi_ik $PY viz/test_logic_collection.py      # LOGIC OK (12)
PYTHONPATH=third_party/humi_ik $PY viz/smoke_test_collection.py      # SMOKE OK (real scene + IK + ViserUrdf)
PYTHONPATH=third_party/i2rt:gripper $PY -m pytest \
    gripper/holosoma_hand_pipeline/tests/test_keyboard_gripper.py -q # 65 passed
PYTHONPATH=gripper $PY gripper/test_kbd_pump_lifecycle.py            # LOGIC OK (10, evdev pump lifecycle)
$PY force_torque/test_logic.py                                       # LOGIC OK
$PY force_torque/smoke_test.py --mock                                # SMOKE OK
PYTHONPATH=recording:processing:replay $PY \
    processing/test_dual_hand_3pt.py                                 # LOGIC OK (90, dual-hand 2pt/3pt)
PYTHONPATH=bridges .venv/bin/python \
    bridges/test_health_probe.py                                     # LOGIC OK (7, per-role tracker gate)
```

## Calibration

Two **distinct** procedures — don't conflate them (full guide in
[`docs/CALIBRATION.md`](docs/CALIBRATION.md)):

- **IK "Recalibrate (hold rest pose)"** — re-anchors the tracker → hand mapping.
- **F/T "Recompensation (90 s)"** — re-estimates the Bota tool payload
  (mass / CoM / inertia) so gravity compensation stays accurate after a
  tool/mount change.

### Bota payload estimation (offline)

The in-browser **Recompensation** button is only ENABLED when that hand has a
payload config: `payload_compensator.json` for the LEFT hand (gitignored;
`bootstrap.sh` seeds it from `.example`) and `payload_compensator_right.json` for
the RIGHT one. Without it that hand's charts show the **RAW** wrench and its button
is greyed with the reason in the `Payload` line. A recompensation only ever rewrites
its OWN hand's `payload_compensator_*.json` (+ `rnea_compensator_*.json`), so
re-estimating one hand never disturbs the other. Run the offline estimator once to
create a missing one:

```bash
# 1) STOP the viz first — it reads the Bota sensor in-process and holds the serial
#    port; the estimator can't open it while the viz (or any bota tool) is running.
# 2) Mount the tool but leave NOTHING touching it — the sensor+tool must hang free.
.venv/bin/python force_torque/estimate_payload.py --max-time 60
#    (drop --max-time to use the config default 15–30 s)
```

When it prints **`>>> ROTATE NOW <<<`**, pick the sensor up and slowly turn it
through many orientations (roll/pitch/yaw so gravity points along ±x, ±y, ±z) for
~15–30 s. It stops on its own when the EKF converges and writes the payload into
`payload_compensator.json` (backing up the previous one to `.bak`). Re-launch with
`./run_collection_viz.sh` and the charts read net (tool-gravity-removed) wrench
with the Recompensation button present. Regenerate after **any** tool/mount change.
