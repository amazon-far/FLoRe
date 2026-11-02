"""
Offline sanity tests for rnea_core — NO hardware, NO bota imports required.

Run:  python examples/rnea_dynamic_compensator_demo/test_rnea_core.py
Exits non-zero on failure. These are analytic checks, not a substitute for
on-hardware validation (docs/04), but they pin down the physics/sign conventions.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rnea_core import (
    RneaDynamicCompensator, RneaCompensatorConfig, AlphaEstimatorConfig, skew,
)

G = 9.80665
FAILS = []


def check(name, cond):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    if not cond:
        FAILS.append(name)


def base_cfg(**kw):
    d = dict(
        mass=2.0,
        com=np.array([0.0, 0.0, 0.05]),
        inertia=np.diag([0.01, 0.02, 0.03]),
        imu_offset=np.zeros(3),
        accel_is_specific_force=True,
        wrench_sign=1.0,
        # deterministic α: no filter lag so the analytic torque checks are exact
        alpha=AlphaEstimatorConfig(omega_lowpass_cutoff_hz=0.0, alpha_lowpass_cutoff_hz=0.0),
    )
    d.update(kw)
    return RneaCompensatorConfig(**d)


def test_skew():
    v = np.array([1.0, 2.0, 3.0])
    x = np.array([-4.0, 5.0, 6.0])
    check("skew(v)@x == cross(v,x)", np.allclose(skew(v) @ x, np.cross(v, x)))


def test_static_gravity():
    """At rest, specific force reads +g along +z_sensor; F should be m*g*z, and
    torque = c x F. This is pure gravity, no motion."""
    cfg = base_cfg()
    comp = RneaDynamicCompensator(cfg)
    accel = np.array([0.0, 0.0, G])   # sensor at rest, z up
    omega = np.zeros(3)
    w = comp.predict_payload_wrench(0.0, accel, omega)
    F_expected = cfg.mass * accel
    M_expected = np.cross(cfg.com, F_expected)
    check("static gravity force = m*g", np.allclose(w[:3], F_expected))
    check("static gravity torque = c x F", np.allclose(w[3:], M_expected))


def test_pure_linear_accel():
    """Add a horizontal acceleration on top of gravity; force tracks m*(a)."""
    cfg = base_cfg(com=np.zeros(3))  # no COM offset -> zero torque
    comp = RneaDynamicCompensator(cfg)
    accel = np.array([3.0, 0.0, G])
    w = comp.predict_payload_wrench(0.0, accel, np.zeros(3))
    check("linear force = m*a_specific", np.allclose(w[:3], cfg.mass * accel))
    check("no torque when com=0 & no rotation", np.allclose(w[3:], 0.0))


def test_centripetal_com():
    """Constant ω, zero α, COM offset along z, IMU at origin. The COM sees
    centripetal accel ω x (ω x r_ic). Verify f_com transport term."""
    cfg = base_cfg(inertia=np.zeros((3, 3)))
    comp = RneaDynamicCompensator(cfg)
    omega = np.array([0.0, 0.0, 5.0])   # spin about z
    accel = np.array([0.0, 0.0, 0.0])   # ignore gravity to isolate the term
    r_ic = cfg.com - cfg.imu_offset
    expected_f_com = np.cross(omega, np.cross(omega, r_ic))
    w = comp.predict_payload_wrench(0.0, accel, omega)
    check("centripetal force = m*(ω×(ω×r))", np.allclose(w[:3], cfg.mass * expected_f_com))


def test_gyroscopic_torque():
    """Constant ω (α=0), verify torque = ω x (I ω) with a non-diagonal ω."""
    I = np.diag([0.01, 0.02, 0.03])
    cfg = base_cfg(inertia=I, com=np.zeros(3), mass=0.0)  # kill force to isolate torque
    comp = RneaDynamicCompensator(cfg)
    omega = np.array([1.0, 2.0, 3.0])
    w = comp.predict_payload_wrench(0.0, np.zeros(3), omega)
    expected = np.cross(omega, I @ omega)
    check("gyroscopic torque = ω×(Iω)", np.allclose(w[3:], expected))


def test_angular_accel_torque():
    """Feed ω ramp so α is a known constant; torque should include I·α.
    Uses unfiltered α (cutoffs=0) and exact dt."""
    I = np.diag([0.01, 0.02, 0.03])
    cfg = base_cfg(inertia=I, com=np.zeros(3), mass=0.0)
    comp = RneaDynamicCompensator(cfg)
    dt = 0.001
    alpha_true = np.array([10.0, -5.0, 2.0])
    # step 0 seeds; step 1 yields α = (ω1-ω0)/dt
    comp.predict_payload_wrench(0.0, np.zeros(3), np.zeros(3))
    omega1 = alpha_true * dt
    w = comp.predict_payload_wrench(dt, np.zeros(3), omega1)
    # ω is tiny here so ω×Iω ~ 0; torque ≈ I·α
    check("angular-accel torque ≈ I·α", np.allclose(w[3:], I @ alpha_true, atol=1e-6))


def test_full_update_and_tare():
    """End-to-end update() with the 12-vector, and tare zeroing at static pose."""
    cfg = base_cfg()
    comp = RneaDynamicCompensator(cfg)
    raw = np.zeros(12)
    raw[6:9] = np.array([0.0, 0.0, G])      # static, z up
    # Suppose the sensor reads exactly the payload wrench + a bias offset:
    predicted = comp.predict_payload_wrench(0.0, raw[6:9], raw[9:12])
    bias = np.array([1.0, -2.0, 3.0, 0.1, -0.1, 0.2])
    raw[:6] = cfg.wrench_sign * predicted + bias
    # Tare should recover exactly this bias, so compensated -> 0.
    comp.tare_from_prediction(raw[:6], predicted)
    out = np.zeros(12)
    comp.update(0.001, raw, out)
    check("post-tare compensated ≈ 0 at static pose", np.allclose(out[:6], 0.0, atol=1e-6))
    check("IMU channels passed through", np.allclose(out[6:], raw[6:]))


def test_sign_flip():
    """wrench_sign = -1 should flip the subtracted prediction."""
    cfg = base_cfg(wrench_sign=-1.0, com=np.zeros(3))
    comp = RneaDynamicCompensator(cfg)
    raw = np.zeros(12)
    raw[6:9] = np.array([0.0, 0.0, G])
    out = np.zeros(12)
    comp.update(0.0, raw, out)
    # compensated = raw(0) - bias(0) - (-1)*predicted = +predicted
    check("wrench_sign=-1 adds prediction", np.allclose(out[:3], cfg.mass * raw[6:9]))


def test_specific_force_false_rejected():
    """accel_is_specific_force=False must fail loudly, not silently drop gravity."""
    cfg = base_cfg(accel_is_specific_force=False)
    raised = False
    try:
        RneaDynamicCompensator(cfg)
    except NotImplementedError:
        raised = True
    check("accel_is_specific_force=False raises NotImplementedError", raised)


if __name__ == "__main__":
    print("Running rnea_core offline tests...")
    for fn in [test_skew, test_static_gravity, test_pure_linear_accel,
               test_centripetal_com, test_gyroscopic_torque, test_angular_accel_torque,
               test_full_update_and_tare, test_sign_flip,
               test_specific_force_false_rejected]:
        print(f"\n{fn.__name__}:")
        fn()
    print("\n" + ("ALL PASSED" if not FAILS else f"FAILURES: {FAILS}"))
    sys.exit(1 if FAILS else 0)
