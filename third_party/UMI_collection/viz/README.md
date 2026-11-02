# `viz/` — the hub (collection_viz.py + g1_right_hand_ik.py)

This is the **hub** of the hub-and-spoke rig (see the repo root README for the
spoke bridges). `collection_viz.py` is a thin **WebSocket client** that:

- subscribes to the three bridge feeds — tracker `8772`, camera `8873`, gripper
  `8774` (bidirectional: encoder JSON in, control intents out) — and re-renders
  one [viser](https://viser.studio) scene on `http://localhost:8080`;
- runs the **right-hand-only mink IK** (`g1_right_hand_ik.py`) on the render
  thread to pose the G1 (`g1_29dof_yam.urdf`);
- rides a **camera frustum** on the gripper with the live D405 RGB feed;
- mirrors the i2rt gripper open/close into the URDF jaws + a live encoder chart;
- reads the **Bota F/T sensor in-process** (NOT a bridge) and draws force/torque
  charts with a payload **Recompensation** button.

It is the only process that holds viser; hardware singletons that can't coexist
in one interpreter (pysurvive, pyrealsense2, CAN) live in their own bridges.

## Run it standalone (UI work, no hardware)

```bash
PYTHONPATH=third_party/humi_ik .venv/bin/python viz/collection_viz.py --mock
# then open http://localhost:8080
```

`--mock` substitutes a synthetic tracker / gripper / camera so the full scene,
GUI, IK, and charts come up with no bridges and no hardware. `humi_ik` must be
on `PYTHONPATH` (the vendored HuMI IK is imported, not pip-installed).

## The IK (`g1_right_hand_ik.py`)

The HuMI mink `IKSolver` drives **only `_right_hand_task`** — left hand, feet,
pelvis, and base all hold the stand pose, so the floating base never drifts.

- **Joint remap by name.** HuMI's 25 hinge joints are mapped onto the 30
  actuated joints of the yam URDF via `YamConfigMapper` (name-based, not
  index-based). The 4 yam-only revolutes — `waist_roll`, `waist_pitch`, and both
  `wrist_yaw` — default to `0`. `finger2` is a `<mimic>` joint resolved by
  `yourdfpy`; only `finger1` is set, from the gripper open fraction.
- **Home-anchored, world-frame delta calibration.** On Recalibrate the operator
  hand pose is captured as the reference (`p_ref`, `R_ref`). Each frame the IK
  target is

  ```
  p_target = p_home + reach_scale * (p_cur - p_ref)
  R_target = R_cur            # rotation mirrored 1:1
  ```

  `--reach-scale` scales **position only**; rotation is always 1:1. The robot
  hand therefore lives in robot-space anchored at `p_home`, not at the physical
  tracker location.
- The **Recalibrate** GUI button re-captures `p_ref`/`R_ref`. Under `--mock` the
  first frame auto-calibrates (unless `--no-auto-calibrate`).

> **CAMERA ≠ WRIST ≠ TRACKER is a feature, not a bug.** The IK targets
> `right_hand_site`. The camera rides `right_wrist_yaw_link` (~0.235 m back along
> `+x`) plus its own `--cam-offset`/`--cam-rpy` mount transform. The home-anchored
> remap places the robot hand in robot-space, *not* at the physical tracker. All
> three frames are intentionally distinct.

## CLI flags (defaults)

| Flag | Default | Notes |
|---|---|---|
| `--mock` | off | synthetic tracker/gripper/camera, no bridges, UI only |
| `--no-ik` | off | render home pose, only move the tracker frame |
| `--no-ft` | off | skip the in-process Bota F/T sensor + charts |
| `--reach-scale` | `1.0` | scales IK target **position** delta only |
| `--ik-config` | `proposal` | HuMI `IKConfig` name |
| `--cam-offset` | `0.10 0.0 0.04` | **UNCALIBRATED — measure for your rig** |
| `--cam-rpy` | `4.7124 1.5708 0.0` | **UNCALIBRATED — measure for your rig** |
| `--gripper-url` | `ws://127.0.0.1:8774` | bidirectional encoder/control |
| `--ft-read-hz` | `250.0` | F/T reader rate; see gotcha below |

Other flags: `--host` (`0.0.0.0`), `--port` (`8080`), `--render-hz` (`60`),
`--yam-urdf`, `--tracker-url` (`8772`), `--camera-url` (`8873`),
`--robot-height`, `--cam-fov-deg` (`58.0`, D405 color), `--no-auto-calibrate`,
`--no-compensate`, `--recal-countdown` (`5.0`), `--recal-max-time` (`60.0`).

## Gotchas

- **`ViserUrdf` needs a `pathlib.Path`.** Passing a `str` trips an internal
  assert. Always wrap the URDF path in `Path(...)`.
- **Render `/robot` at the IK base z (~0.76).** Otherwise the pelvis sits on the
  grid and the feet hang below it.
- **viser 1.0.30 is right-panel-only** (`control_width` caps at `"large"`,
  ~384 px) and has **no keyboard API** — that's why gripper control needs the
  external evdev bridge, not an in-page key handler.
- **IK is ~9.8 ms/step.** It runs on the render thread; keep per-frame work off
  the hot path.
- **`--ft-read-hz` defaults to `250`.** A 1 kHz F/T reader starves the IK loop —
  do not crank it without re-measuring frame time.
- **Thread invariants:** the mink IK `Configuration` is mutated **only** on the
  render thread; the Bota HWI/estimator is touched **only** on the
  `SensorReader` thread.

## Tests

```bash
PYTHONPATH=third_party/humi_ik .venv/bin/python viz/test_logic_collection.py    # prints "LOGIC OK" (12 checks)
PYTHONPATH=third_party/humi_ik .venv/bin/python viz/smoke_test_collection.py     # prints "SMOKE OK"
```

`smoke_test_collection.py` builds the real scene + IK + `ViserUrdf` and asserts
the hand moves, the base holds, and the meshes load.

## Two distinct recal handshakes

There are **two separate** recalibration flows — do not conflate them:

- **IK Recalibrate** — re-captures the operator hand reference (`p_ref`/`R_ref`).
- **F/T Recompensation** — re-zeros the Bota payload compensation.

See [`docs/CALIBRATION.md`](../docs/CALIBRATION.md) for the full procedure.
