# Architecture

`UMI_collection` is a browser ([viser](https://viser.studio/)) **visualization** of
a teleop data-collection rig. One viser scene shows, live: a G1 humanoid
(`g1_29dof_yam.urdf`) whose **right hand** is driven by a single VIVE tracker
through HuMI mink IK; a RealSense **D405** RGB feed on a camera frustum that
follows the gripper; the **i2rt DM4310** gripper open/close mirrored in the URDF
jaws plus a live encoder chart; and **Bota** force/torque charts with a payload
"Recompensation" button. Recording is a later step — this repo is the viz only.

## Hub-and-spoke topology

The viz (`viz/collection_viz.py`) is a thin WebSocket **client** — the hub. Each
hardware singleton runs in its **own process**, in the venv that can actually
talk to that device, and emits localhost-only WebSocket JSON. No single
interpreter holds viser + pysurvive + pyrealsense2 + CAN + bota_driver at once.

```
                          .venv-vive (py3.9)
                       ┌──────────────────────────┐
   VIVE tracker  ───── │ vive_tracker_bridge       │ ──ws JSON──┐  :8772
   (pysurvive)         │   bridges/tracker/        │            │  (pose OUT)
                       └──────────────────────────┘            │
                       ┌──────────────────────────┐            │
   D405 (pyrealsense2) │ realsense_bridge          │ ──ws JSON──┤  :8873
                       │   bridges/camera/         │            │  (jpeg OUT)
                       └──────────────────────────┘            │
                                                               ▼
                          .venv (py3.12)                ┌──────────────────────┐
                       ┌──────────────────────────┐     │ collection_viz.py    │
   DM4310 on can0  ─── │ keyboard_gripper          │◄────┤  (HUB / WS client)   │
   (i2rt / SocketCAN)  │   --web-control           │ ws  │  viser + mink IK +   │
   SOLE CAN WRITER     │   gripper/...             │─────►│  bota reader + GUI   │
                       └──────────────────────────┘ :8774│  + charts            │
                          .venv (py3.12)            (BIDIR)└──────────────────────┘
                       ┌──────────────────────────┐            │  :8080
   keyboard (evdev)─── │ keyboard_gripper_ws       │ ──ws JSON──┘  viser web UI
   NO CAN              │   gripper/keyboard_..._ws │  (intents IN → :8774)
                       └──────────────────────────┘
                       ┌──────────────────────────┐
   Bota F/T  ───────── │ bota_ft_viz (optional)   │  dashboard :8081
   (in-process reader  │   force_torque/          │  (the live F/T charts are
    inside the viz)    └──────────────────────────┘   also read in-process)
```

Ports: tracker **8772**, camera **8873** (RIGHT **8877**; 8773/8777 are the deploy stack's robot-mounted cameras), gripper **8774** (bidirectional),
viser web UI **8080**, optional bota dashboard **8081**. All binds are
`localhost`-only.

## Processes and ownership

| Process | venv / python | Owns / links against | Role |
|---|---|---|---|
| `collection_viz.py` | `.venv` (py3.12) | viser, mink, mujoco, yourdfpy, bota_driver, bota_payload_utils | Hub. viser scene, mink right-hand IK, GUI + charts, in-process Bota F/T reader |
| `vive_tracker_bridge` | `.venv-vive` (py3.9) + `~/libsurvive` on `LD_LIBRARY_PATH`/`PYTHONPATH` | source-built libsurvive `5588cadf` (the pinned commit) via its ctypes bindings | Reads every VIVE tracker (one shared `SimpleContext`, serial-keyed), emits the role-keyed pose JSON on `:8772` |
| `realsense_bridge` | `.venv-vive` (py3.9) | pyrealsense2, opencv | Reads the D405, emits JPEG frames on `:8773` |
| `keyboard_gripper --web-control` | `.venv` (py3.12) | i2rt, python-can | **Sole CAN writer.** Drives the DM4310 on `can0`; encoder OUT + control IN on `:8774` |
| `keyboard_gripper_ws` | `.venv` (py3.12) | evdev | Reads keyboard; sends open/close **intents** to `:8774`. Touches no CAN |

### Why two venvs

`pyrealsense2` ships a **cp39-only** wheel, so the camera bridge must run under py3.9
(`.venv-vive`); the tracker bridge shares that interpreter for numpy/websockets and gets
pysurvive from a **source-built** libsurvive (`~/libsurvive` @ `5588cadf`, pure-ctypes
bindings on `PYTHONPATH`, `bin/` on `LD_LIBRARY_PATH`) — the PyPI wheel is refused. Everything
else — viser, mink, bota — runs under py3.12 (`.venv`). The split is forced by the
native-lib / python-version constraints, not by preference: a single interpreter cannot
satisfy viser + pyrealsense2 + i2rt/CAN + bota_driver simultaneously. The
hub-and-spoke design turns each incompatible dependency into an isolated process
that publishes JSON, so the hub only depends on a WebSocket client.

## Invariants (load-bearing)

These are the rules that keep the rig safe and race-free. Do not violate them.

- **Exactly one process writes CAN** — `keyboard_gripper --web-control`. The
  browser and the keyboard bridge only send JSON **intents**; they never open a
  CAN socket. The **startup self-test** (`gripper/gripper_selftest.py`) also
  writes CAN, but it runs **before** the bridge and **releases the bus** (motor
  off + `chain.close()`) before the bridge starts — they never run concurrently,
  so the single-writer rule holds.
- **A position command may never lead the encoder by more than `--max-lead`.**
  This **overheat guard** (`_clamp_to_actual_lead`) bounds stall torque to
  `kp·max_lead`. Without it, a stale/over-far endpoint drives the motor into a
  hard stop at saturated torque → over-temperature → the i2rt control thread
  dies, and the bridge then green-lights a dead motor. `motor_ok` on `:8774`
  (from `chain.running`) surfaces such a death so the viz shows `✖ MOTOR FAULT`.
- **`:8774` is bidirectional** — encoder/state flows OUT to the viz, control
  commands flow IN. It is the only socket with two-way traffic.
- **All WS binds are `localhost`-only.** Nothing is exposed off-host.
- **The mink IK `Configuration` is mutated ONLY on the render thread.** The
  right-hand-frame task is the only task that tracks the tracker; left
  hand/feet/pelvis/base hold the stand pose so the floating base never drifts.
  WS callbacks must not touch the `Configuration` directly.
- **The bota HWI / estimator is touched ONLY on the `SensorReader` thread.** The
  GUI reads cached values; it does not call into the driver.
- **Buttons raise flags; the render loop mutates.** GUI callbacks (e.g. the
  "Recompensation" button) run on a **viser callback thread**, not the render
  thread. A callback must only flip a phase flag under the lock — it must
  **never** null a modal handle or mutate scene/IK/driver state directly. The
  render loop observes the flag and performs the mutation. A close button that
  nulls a modal handle from the viser callback thread will race the render
  thread's read of that handle.

## Launcher contract (`run_collection_viz.sh`)

The launcher starts the four background bridges plus the viz, with a cleanup
trap that powers the energized gripper down on any exit. Two gotchas are
load-bearing:

- **Background bridges need a real session — do NOT `nohup` them.** They are
  launched into the launcher's session so they receive the shutdown signal;
  `nohup` detaches them from `SIGHUP` and leaks orphaned bridge processes.
- **The viz runs as `… & wait`, NOT `exec`.** `exec` replaces the launcher
  process and **destroys the cleanup trap**, so a crashed viz would leave the
  gripper energized. A plain foreground (non-`&`) run is also wrong: bash
  **defers a trapped `SIGTERM` until the foreground child returns**, so shutdown
  hangs. Running the viz with `& wait` lets the trap fire promptly and route any
  exit path through the gripper's own `SIGTERM` handler, de-energizing the
  motor.
- **The gripper self-test gates the launch.** Before spawning the gripper bridge,
  the launcher runs `gripper/gripper_selftest.py` (auto-home both stops + verify a
  full open/close). On PASS it feeds the discovered endpoints to the bridge; on
  **FAIL it aborts the launch** with the motor powered down — so the stack never
  comes up over a stalled/dead gripper. The jaws move during this test (hands
  clear, nothing held). `SELFTEST=0` skips it (removes the startup over-temp gate).
  The self-test runs the python pipeline under `set +e` and reads `PIPESTATUS` so a
  FAIL surfaces its guidance instead of `set -e` aborting silently.

`./run_collection_viz.sh --mock` brings up the UI only (no hardware bridges).

## Vendored third-party trees

- **`third_party/humi_ik/`** — vendored HuMI `ikumi` IK (the mink-based
  differential IK used for the right-hand task). On `PYTHONPATH`.
- **`third_party/i2rt/`** — the minimal SocketCAN motor-driver subset for the
  DM4310. On `PYTHONPATH`.

`robot_assets/` holds the render URDF + meshes (`g1_yam/`, `g1_29dof_yam.urdf`)
and the Bota configs (`bota_config/`).

## Setup & run

```bash
./bootstrap.sh                 # create both uv venvs + seed bota payload
./run_collection_viz.sh        # full stack (all bridges + viz)
./run_collection_viz.sh --mock # UI only, no hardware
```
