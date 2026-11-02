# Calibration

The collection viz has **two independent recalibration handshakes**. They are
unrelated and are a common source of confusion, so keep them straight:

| Button | State flag | What it re-anchors | Owner thread |
|---|---|---|---|
| **Recalibrate (hold rest pose)** | `CollectionState.recal_request` | the tracker → right-hand IK mapping | render thread |
| **Recompensation (90s)** | `SensorState.recal_request` | the Bota tool **payload** (mass / CoM / inertia) for gravity compensation | bota `SensorReader` thread |

The IK recal moves the *robot hand*; the F/T recompensation changes how the
*force/torque* charts subtract gravity. Neither one touches the other.

---

## (A) IK recalibration — "Recalibrate (hold rest pose)"

Re-anchors the tracker-to-hand mapping so the robot right hand mirrors **your
displacement and rotation relative to a reference pose you choose**, not the raw
tracker world pose.

**Procedure**

1. Hold your hand (the one wearing the VIVE tracker) at a comfortable **rest
   pose** that matches the robot's home hand pose.
2. Click **Recalibrate (hold rest pose)**.
3. From that moment, the robot right hand follows:
   - your **translation** away from the reference, scaled by `reach_scale`
     (default `1.0`), and
   - your **rotation** relative to the reference, 1:1.

The mink IK then solves the right-hand frame task to that target while the left
hand, feet, pelvis, and base hold the stand pose (so the floating base never
drifts).

Under `--mock` there is no live tracker, so the viz **auto-calibrates at the
home pose** on startup; the button is effectively a no-op there.

---

## (B) F/T recompensation — "Recompensation (90s)"

Re-estimates the Bota **tool payload** (mass, center of mass, inertia) so gravity
compensation stays accurate after you change the gripper, mount, or anything
rigidly attached past the sensor.

**Visibility.** The button appears **only** when all of these hold:

- a Bota sensor is present,
- **both** `payload_compensator.json` **and** `payload_estimator.json` exist, and
- `--no-compensate` is **off** (compensation is enabled).

**Procedure (one click → fullscreen overlay)**

1. **Countdown (~5 s, `--recal-countdown` default `5`).** **Lift the tool and
   keep it FREE OF CONTACT.** Nothing should be touching the payload.
2. **Estimation (~90 s, `--recal-max-time` default `90`).** **Rotate the tool
   SLOWLY through many orientations** so gravity points down every axis (roll,
   pitch, yaw). The Bota EKF needs gravity to load each axis to solve the
   payload.
3. On success the viz **swaps in the new compensator and re-tares**.

A **good estimate has positive diagonal inertia**. If the diagonal inertia comes
back non-positive, treat the estimate as bad and redo it (slower, fuller range
of motion, confirm the tool was free of contact).

---

## Tare vs. Recompensation

Two different ways to zero the force/torque reading — do not confuse them.

- **Tare wrench** (the "Tare wrench" button): subtracts the **current** reading
  as a constant offset. Instant, but only correct **at that one pose** — as the
  sensor rotates, gravity creeps back into the reading because the offset is
  fixed.
- **Recompensation**: removes gravity in **any** orientation by modeling the
  payload. Slow (~95 s total) but orientation-correct.

**When to use which**

- Use **Tare** for a fast zero at the current pose.
- Use **Recompensation** when the **tool or mount changes** (different gripper,
  added fixture, etc.).
- **Bootstrap the FIRST payload offline** with `estimate_payload.py` before you
  ever rely on the in-viz button — see `force_torque/README.md`. The in-viz
  recompensation refines/replaces an already-seeded payload; it is not the
  initial seed.

---

## Threading rules (load-bearing — do not violate)

- The bota **`SensorReader` thread is the ONLY toucher of the Bota HWI /
  estimator.** All sensor reads, the EKF, the compensator swap, and the tare
  happen there.
- The **render thread owns the WHOLE modal lifecycle** — it creates the
  fullscreen overlay, drives the countdown/estimation phases, and tears the
  modal down.
- The **Close button only flips `recal_phase` under the lock.** It must **NEVER
  null the modal handle from the viser callback thread** — that race kills the
  render loop. Closing is requested via the phase flag; the render thread
  performs the actual teardown.

---

## Convergence gotcha — "did not converge" in-viz but `estimate_payload.py` works

**Symptom.** Recompensation runs the full ~90 s and then fails with **"did not
converge"**, yet the standalone `force_torque/estimate_payload.py` succeeds on
the same hardware.

**Cause.** The Bota EKF needs a **high, steady sample rate**. The ~60 Hz mink IK
solve (~8 ms/step) competes for the GIL and **starves the estimator** of samples,
so the EKF never accumulates enough clean data to converge.

**Fix (baked in).** While `recal_phase` is in `countdown` or `estimating`, the
viz **skips `ik.step()` AND the per-frame JPEG decode**, handing the interpreter
to the estimator thread. The arm simply **holds its last pose for the ~90 s** of
the procedure. No action required — this happens automatically.

**The physical prerequisite still applies.** Skipping IK only frees the GIL; you
must still **lift the tool free of contact and rotate slowly through
roll/pitch/yaw**. If you hold it still or let it touch something, it will fail to
converge no matter what the GIL is doing.

**Reading the logs.** The `[ESTIMATOR] Estimating, keep moving the payload...`
lines are the **Bota library's own runtime messages**. A convergence failure is
therefore the **EKF itself** rejecting the data — not the viz wiring or the WS
bridge. Redo the motion slower and through a fuller range of orientations.
