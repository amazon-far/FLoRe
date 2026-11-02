# `force_torque/` — Bota F/T tooling

Bota Systems force/torque sensor support: payload-gravity compensation and a
standalone viser visualizer, plus the helper tools that feed the main
collection viz.

The main collection viz (`viz/collection_viz.py`) reads the Bota sensor
**in-process** — the `.venv` already ships `bota-driver` + `bota-payload-utils`,
so the Force/Torque charts and the **Recompensation** button live inside the
main viser scene, not behind a bridge. This module additionally ships the
standalone tools below for calibration and bring-up.

## Sensor auto-detect (hardware-swap robustness)

The serial device is auto-detected from the by-id glob:

```
/dev/serial/by-id/*Bota_Systems*Force_Torque_Sensor*-if00
```

Swapping sensors or USB ports needs **no config change**. Override with
`--port-path` if needed. The `com_port` / `serial` fields in the committed
`bota_binary.json` are **irrelevant** — they are overridden at runtime by the
by-id resolution.

## Tools

| File | Purpose |
|---|---|
| `bota_ft_viz.py` | Standalone viser viz (force/torque charts + orientation triad) |
| `estimate_payload.py` | Estimate tool mass / COM / inertia → write compensator JSON |
| `compensate_live.py` | Print RAW vs COMPENSATED wrench to sanity-check the estimate |
| `dashboard.py` | Bottom-dock dashboard layout (iframe-wraps viser, second port). Imported by `bota_ft_viz.py --dashboard`; not standalone. |
| `test_logic.py` | Pure-logic unit test |
| `smoke_test.py` | `--mock` smoke test |

## Gotchas

- **One owner of the serial port.** Only one process can hold `/dev/ttyACM*` at
  a time — close any Chrome WebSerial tab holding the port before launching a
  tool.
- **Reader throttling.** `read_frame_blocking` holds the GIL, so a tight ~1 kHz
  reader starves viser. Throttle with `--read-hz` (standalone default `200`; the
  collection viz uses `--ft-read-hz 250`).
- **Orientation triad is a complementary filter.** Pitch/roll are absolute and
  drift-free; **yaw is gyro-only and drifts**. Use **Reset** to rezero.
- **Estimator output is CWD-relative at construction.** The estimator writes its
  output relative to the CWD when it is constructed; the tools always pass
  absolute paths to avoid surprises.

## Payload compensation pipeline

**Tare vs Compensation:** Tare subtracts a *fixed constant* — only correct at
the exact pose where you click it. Compensation removes the tool gravity in
*any* orientation, so the wrench stays zeroed as the gripper rotates.

### Step 1 — estimate the payload

```bash
.venv/bin/python force_torque/estimate_payload.py --max-time 60
```

The 30 s default is often too short by hand, so pass `--max-time 60`. When it
prints **ROTATE NOW**, lift the tool free of contact and slowly rotate through
many orientations so gravity points along ±x, ±y, ±z. A good estimate has
**positive diagonal inertia**.

It writes (per-rig):

```
robot_assets/bota_config/bota_payload_utils/payload_compensator.json
```

This file is **gitignored** and is **per-rig** — regenerate it after any tool or
mount change. `bootstrap.sh` seeds it from
`payload_compensator.example.json`.

### Step 2 (optional) — verify

```bash
.venv/bin/python force_torque/compensate_live.py
```

Prints RAW vs COMPENSATED so you can confirm the compensated wrench sits near
zero across orientations.

### Step 3 — use it

The viz shows the **compensated** wrench whenever the compensator JSON exists. If
`payload_compensator.json` is **missing**, the F/T reader runs the RAW loop and the
in-browser **Recompensation** button is **hidden** (it needs both the compensator
and estimator configs) — so a vanished Recompensation button means "run Step 1".

## Standalone viz

```bash
.venv/bin/python force_torque/bota_ft_viz.py
# open http://localhost:8080
```

| Flag | Effect |
|---|---|
| `--compensate` | Start in compensated mode |
| `--dashboard` | Bottom-dock layout; needs a **second** port `8081` (it iframe-wraps viser) |
| `--mock` | Synthetic data, no hardware |
| `--read-hz` | Reader throttle (default `200`) |
| `--port-path` | Override sensor by-id auto-detect |

## Tests

```bash
.venv/bin/python force_torque/test_logic.py     # LOGIC OK
.venv/bin/python force_torque/smoke_test.py --mock   # SMOKE OK
```

## See also

`docs/CALIBRATION.md` covers the in-browser **Recompensation** (90 s) procedure
and the IK-pause convergence gotcha.
