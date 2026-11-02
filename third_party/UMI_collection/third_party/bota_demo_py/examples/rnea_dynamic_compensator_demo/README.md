# RNEA Dynamic Compensator (independent, parallel to Bota's official path)

An independent, pure-numpy dynamic payload compensator for a wrist-mounted Bota F/T
sensor with an onboard IMU. It removes the gripper's **gravity + motion (inertial)**
wrench so that fast free-space motion doesn't produce phantom force spikes that would
make a force controller shake.

It runs **parallel to** Bota's official `bota_payload_utils` compensator (same input,
same output shape, **zero shared compensation code**), so you can run each on the same
sensor and compare.

## Why this exists / what's different from Bota's

- **Same running logic** as Bota's compensator: read the raw 12-vector
  `[fx,fy,fz, tx,ty,tz, ax,ay,az, wx,wy,wz]`, subtract the payload's predicted
  gravity+inertial wrench, output the compensated wrench.
- **Different, open, testable math**: because the Bota sensor's IMU already gives
  sensor-frame specific force + angular rate, the payload distal to the sensor is a
  **single rigid body**, so RNEA collapses to a one-body Newton–Euler pass in the
  sensor frame. No robot model, no floating-base kinematics, no finite-differenced
  joint accelerations. The only derivative is `α = dω/dt`, filtered.

Physics (all in the sensor frame S, payload mass `m`, COM offset `c`, inertia `I`
about the COM, IMU point `r_imu`):

```
f_com = f_imu + α×r_ic + ω×(ω×r_ic),   r_ic = c − r_imu
F_O   = m · f_com
M_O   = I·α + ω×(I·ω) + c×F_O
compensated = raw − bias − wrench_sign · [F_O, M_O]
```

`bias` is set by the startup tare; `wrench_sign` matches Bota's reading convention
(**−1** for this rig — see Calibration). See `rnea_core.py` for the derivation and the
documented assumptions.

## Files, and how the collection stack uses them

| File | What |
|---|---|
| `rnea_core.py` | The compensator + α-estimator. Pure numpy, **no bota_* imports**. |
| `test_rnea_core.py` | Offline analytic physics tests (no hardware). |

`force_torque/bota_ft_viz.py` imports `rnea_core` from this directory for the `rnea`
engine of each hand's **F/T compensation** panel (the alternative to Bota's own `bota`
engine). Its per-hand configs live in
`robot_assets/bota_config/rnea_dynamic_compensator/rnea_compensator{,_left,_right}.json`.
Only this directory is vendored: the standalone Bota demo harness the numbers below were
measured with (Bota Systems' `bota_demo_py`, see https://code.botasys.com) is not
redistributed here.

Offline self-test, from the FLoRe checkout (no hardware):

```bash
third_party/UMI_collection/.venv/bin/python \
    third_party/UMI_collection/third_party/bota_demo_py/examples/rnea_dynamic_compensator_demo/test_rnea_core.py
```

## Calibration (do this once per sensor/tool)

1. **Payload params** (`rnea_compensator*.json` → `payload_properties`): mass, COM,
   inertia. Get them from the Bota payload estimator (`force_torque/estimate_payload.py`,
   or the collection viz's per-hand **Recompensation** button, which rewrites that hand's
   file) or from CAD.
   - **Mass and COM are what matter for the force residual.** The inertia tensor only
     affects the (small) torque residual — verified: swapping inertia leaves the force
     residual unchanged. Don't over-invest in estimating inertia precisely; a positive-
     definite estimate (or even zeros) is fine.
2. **`wrench_sign`** (`conventions.wrench_sign`): Bota reports the wrench with the
   opposite sign to our payload prediction, so this rig uses **−1**. Verify from a
   static tilt: with a correct sign, the compensated wrench stays ~0 as you reorient;
   if it grows worse when tilted, flip the sign.
3. **`imu_offset`** (`geometry.imu_offset`): `[0, 0, -0.0257]` m for the M8 series.
4. **`accel_is_specific_force`**: must be `true` (the Bota accelerometer reports
   specific force, i.e. includes gravity). `false` raises `NotImplementedError`.

## What to expect (measured, left hand, 50 Hz / 12 Hz anti-alias)

| | RNEA | Bota official |
|---|---|---|
| moving \|F\| residual mean | **1.4 N** | 2.2 N |
| moving \|F\| residual p99 | **3.2 N** | 5.3 N |
| >4 N false-trigger rate | **0.5%** | 6.3% |
| static \|F\| residual | 0.3 N | 0.3 N |
| torque residual | 0.47 Nm | 0.20 Nm |

With a **4 N** contact threshold on a 50 Hz controller, RNEA's decimated output sits
safely under 4 N in free space (0.5% excursions), while Bota's un-decimated output
exceeds 4 N ~6% of the time. RNEA wins on force here **because** its residual is higher-
frequency (α noise) and the anti-alias filter removes exactly that; Bota's residual is
lower-frequency and survives decimation. Bota is better on torque (RNEA's inertia is
weak), but torque is not the shaking concern here.

## Honest limits (not bugs — physics)

- The residual floor (~3 N p99 while moving, at 50 Hz) is **IMU accelerometer
  noise/bias in the `m·a` term**, not a tuning problem. It does **not** shrink with
  better payload params or a lower α cutoff (verified: lowering the α cutoff makes the
  *force* residual slightly worse). Bota's own note: "the IMU is not a perfect sensor."
- Tare removes a constant bias at one pose; a small **pose-dependent** residual remains
  as you reorient (both paths have it).
- For a hard 4 N threshold, add a short **debounce** in the controller (require ≥2
  consecutive over-threshold samples) to reject the rare single-sample excursion.
