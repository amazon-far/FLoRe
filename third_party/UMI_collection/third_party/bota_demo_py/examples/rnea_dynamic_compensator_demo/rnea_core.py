"""
rnea_core.py — Independent single-body Newton-Euler (RNEA) dynamic wrench compensator.

This module is a *parallel, independent* re-implementation of dynamic payload
compensation. It shares NOTHING with the Bota `bota_payload_utils` code path:
no bota imports here, pure numpy. The running *logic* is deliberately the same
as Bota's compensator — read a raw wrench+IMU vector, compute the payload's
gravity + inertial contribution, subtract it, and hand back a compensated wrench —
so the two can run side by side on the same inputs and be compared directly.

------------------------------------------------------------------------------
PHYSICS (why this is "RNEA" for our case)
------------------------------------------------------------------------------
RNEA (Recursive Newton-Euler Algorithm) is the general inverse-dynamics recursion
over a kinematic chain. For a wrist-mounted F/T sensor whose *entire* downstream
payload (gripper + adapter + camera + rigidly-held object) is one rigid body,
the recursion has exactly ONE link, so it collapses to a single Newton-Euler
pass. Crucially, the Bota sensor carries its own IMU, so we get the sensor-frame
proper acceleration and angular rate DIRECTLY from the reading — we do NOT need
the robot model, floating-base kinematics, or finite-differenced joint
accelerations (q̈), which is the dominant noise source in a full-robot RNEA.

The only time derivative we still need is angular acceleration α = dω/dt, which
we estimate from the gyro with an explicit, tunable filter (see AlphaEstimator).

Let everything be expressed in the SENSOR frame S:
  ω          : angular velocity of the (rigid) sensor+payload body           [gyro]
  α          : angular acceleration = dω/dt                                  [estimated]
  f_imu      : accelerometer *specific force* at the IMU point               [accel]
               (specific force = a_coordinate - g; at rest it reads +g "up",
                so gravity is ALREADY folded in — we never add m·g separately)
  c          : vector from sensor origin O to payload COM                    [config]
  r_imu      : vector from sensor origin O to the IMU/accelerometer point    [config]
  m, I       : payload mass and inertia tensor about its COM (in S axes)     [config]

Specific force transported from the IMU point to the COM:
  f_com = f_imu + α × r_ic + ω × (ω × r_ic),   r_ic = c - r_imu

Single-body Newton-Euler → wrench the sensor structure exerts on the payload,
taken at the sensor origin O:
  F_O = m · f_com
  M_O = I·α + ω × (I·ω) + c × F_O

This [F_O, M_O] is the payload's gravity+inertial contribution to what the
sensor reads (up to a global sign / frame convention that MUST be calibrated
against static data — see `wrench_sign` and docs/03). The compensated (contact)
wrench is:
  W_contact ≈ W_raw − W_bias − wrench_sign · [F_O, M_O]

------------------------------------------------------------------------------
HONESTY / ASSUMPTIONS (each is called out in docs/ for you to confirm)
------------------------------------------------------------------------------
1. `frame.acceleration` is *specific force* (includes gravity), in sensor axes.
   Verified indirectly: Bota's own compensator consumes accel in the same 12-vector.
   Only this specific-force path is IMPLEMENTED. A gravity-removed (coordinate-
   acceleration) IMU is NOT supported here, because subtracting/adding gravity needs a
   sensor-frame gravity vector, which needs orientation this IMU-only path deliberately
   avoids. Constructing with accel_is_specific_force=False raises NotImplementedError
   rather than silently dropping m*g.  -> confirm which your sensor reports in docs/01
2. IMU and payload are ONE rigid body (rigid mounting, no compliant adapter).
3. Inertia I is expressed about the COM, in sensor axes. -> docs/02
4. Global sign/frame convention of the Bota reading is unknown here and is
   fixed empirically by a static calibration. -> docs/03
5. α is finite-differenced from ω → noisy; filtered here, still the weakest link.
   Prefer a lower cutoff and validate. -> docs/04
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import numpy as np


def skew(v: np.ndarray) -> np.ndarray:
    """Return the 3x3 skew-symmetric matrix M such that M @ x == np.cross(v, x)."""
    x, y, z = float(v[0]), float(v[1]), float(v[2])
    return np.array([[0.0, -z, y],
                     [z, 0.0, -x],
                     [-y, x, 0.0]], dtype=np.float64)


@dataclass
class AlphaEstimatorConfig:
    # First-order low-pass cutoff (Hz) applied to ω BEFORE differentiation.
    # <= 0 disables filtering (raw backward difference — noisy, not recommended).
    omega_lowpass_cutoff_hz: float = 20.0
    # Extra low-pass on the differentiated α (Hz). <= 0 disables.
    alpha_lowpass_cutoff_hz: float = 20.0
    # Clamp |dt| into [min_dt, max_dt] to survive timestamp glitches (seconds).
    min_dt: float = 1.0e-4   # 10 kHz ceiling
    max_dt: float = 1.0e-2   # 100 Hz floor


class AlphaEstimator:
    """
    Estimates angular acceleration α = dω/dt from a stream of (timestamp, ω).

    Pipeline:  ω -> [optional LP] -> backward finite difference -> [optional LP] -> α
    First call returns α = 0 (no history yet).

    NOTE: This is the single largest error source in dynamic compensation. Finite
    differencing amplifies gyro noise by ~1/dt. Filtering trades phase lag for noise.
    If your controller can supply a model-based α (e.g. from commanded trajectory),
    prefer that and bypass this estimator entirely.
    """

    def __init__(self, cfg: AlphaEstimatorConfig | None = None) -> None:
        self.cfg = cfg or AlphaEstimatorConfig()
        self._prev_t: float | None = None
        self._omega_f = np.zeros(3, dtype=np.float64)   # filtered ω state
        self._alpha_f = np.zeros(3, dtype=np.float64)   # filtered α state
        self._have_omega = False

    @staticmethod
    def _lp_alpha(cutoff_hz: float, dt: float) -> float:
        """First-order low-pass smoothing factor in [0, 1] for step dt."""
        if cutoff_hz <= 0.0:
            return 1.0  # pass-through
        tau = 1.0 / (2.0 * np.pi * cutoff_hz)
        return dt / (tau + dt)

    def reset(self) -> None:
        self._prev_t = None
        self._omega_f[:] = 0.0
        self._alpha_f[:] = 0.0
        self._have_omega = False

    def update(self, timestamp: float, omega: np.ndarray) -> np.ndarray:
        """Return the current α estimate (rad/s^2) in the same axes as ω."""
        omega = np.asarray(omega, dtype=np.float64)

        if self._prev_t is None:
            # Seed filter state; no derivative available on the first sample.
            self._omega_f[:] = omega
            self._prev_t = timestamp
            self._have_omega = True
            return self._alpha_f.copy()

        dt = timestamp - self._prev_t
        # Guard against zero/negative/huge dt from timestamp jitter.
        dt = float(np.clip(dt, self.cfg.min_dt, self.cfg.max_dt))
        self._prev_t = timestamp

        # 1) low-pass ω
        a_w = self._lp_alpha(self.cfg.omega_lowpass_cutoff_hz, dt)
        omega_prev = self._omega_f.copy()
        self._omega_f += a_w * (omega - self._omega_f)

        # 2) backward difference of the *filtered* ω
        alpha_raw = (self._omega_f - omega_prev) / dt

        # 3) low-pass α
        a_a = self._lp_alpha(self.cfg.alpha_lowpass_cutoff_hz, dt)
        self._alpha_f += a_a * (alpha_raw - self._alpha_f)

        return self._alpha_f.copy()


@dataclass
class RneaCompensatorConfig:
    # --- payload inertial parameters (single rigid body distal to the sensor) ---
    mass: float = 0.0                                   # kg
    com: np.ndarray = field(default_factory=lambda: np.zeros(3))   # sensor origin -> COM, sensor axes [m]
    inertia: np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))  # about COM, sensor axes [kg m^2]

    # --- geometry ---
    imu_offset: np.ndarray = field(default_factory=lambda: np.zeros(3))    # sensor origin -> IMU point [m]

    # --- conventions (see docs/03) ---
    accel_is_specific_force: bool = True   # True: frame.acceleration already includes gravity
    wrench_sign: float = 1.0               # global sign matching Bota's reading convention (+1/-1)

    # --- toggles for validation / ablation (see docs/04) ---
    enable_linear_inertia: bool = True     # m·(f terms)  — gravity is inside f when specific force
    enable_rotational_inertia: bool = True # I·α + ω×(I·ω)
    enable_com_coupling: bool = True        # c × F, and α×r_ic + ω×(ω×r_ic) transport to COM

    # --- bias (set by tare) ---
    bias: np.ndarray = field(default_factory=lambda: np.zeros(6))  # [fx..tz] in reading convention

    alpha: AlphaEstimatorConfig = field(default_factory=AlphaEstimatorConfig)

    @staticmethod
    def from_json(path: str) -> "RneaCompensatorConfig":
        with open(path, "r") as fh:
            raw = json.load(fh)
        c = raw["rnea_compensator_config"]
        pp = c["payload_properties"]
        com = pp["com"]
        I = pp["inertia"]
        inertia = np.array([
            [I["Ixx"], I["Ixy"], I["Ixz"]],
            [I["Ixy"], I["Iyy"], I["Iyz"]],
            [I["Ixz"], I["Iyz"], I["Izz"]],
        ], dtype=np.float64)
        geo = c.get("geometry", {})
        conv = c.get("conventions", {})
        tog = c.get("toggles", {})
        af = c.get("alpha_estimator", {})
        return RneaCompensatorConfig(
            mass=float(pp["mass"]),
            com=np.array([com["x"], com["y"], com["z"]], dtype=np.float64),
            inertia=inertia,
            imu_offset=np.array(geo.get("imu_offset", [0.0, 0.0, 0.0]), dtype=np.float64),
            accel_is_specific_force=bool(conv.get("accel_is_specific_force", True)),
            wrench_sign=float(conv.get("wrench_sign", 1.0)),
            enable_linear_inertia=bool(tog.get("enable_linear_inertia", True)),
            enable_rotational_inertia=bool(tog.get("enable_rotational_inertia", True)),
            enable_com_coupling=bool(tog.get("enable_com_coupling", True)),
            alpha=AlphaEstimatorConfig(
                omega_lowpass_cutoff_hz=float(af.get("omega_lowpass_cutoff_hz", 20.0)),
                alpha_lowpass_cutoff_hz=float(af.get("alpha_lowpass_cutoff_hz", 20.0)),
                min_dt=float(af.get("min_dt", 1.0e-4)),
                max_dt=float(af.get("max_dt", 1.0e-2)),
            ),
        )


class RneaDynamicCompensator:
    """
    Single-body Newton-Euler dynamic wrench compensator.

    Interface intentionally mirrors bota_payload_utils.BotaPayloadCompensator so the
    two run in parallel on identical inputs:

        comp = RneaDynamicCompensator(cfg)
        comp.update(timestamp, wrench_imu_raw12, wrench_imu_compensated12)

    where the 12-vector layout is [fx,fy,fz, tx,ty,tz, ax,ay,az, wx,wy,wz] in
    SENSOR frame. The compensated output copies IMU channels through unchanged and
    writes the compensated contact wrench into slots [0:6].

    This class does NOT return a BotaControlReturnCode (that type lives in bota_control,
    which we deliberately do not import). It returns a plain status string; the demo
    maps it onto the same reactor used by the Bota demos.
    """

    def __init__(self, cfg: RneaCompensatorConfig) -> None:
        if not cfg.accel_is_specific_force:
            # The gravity-removed path would need a sensor-frame gravity vector
            # (i.e. orientation) threaded through update(); this IMU-only single-body
            # path does not have that. Fail loudly instead of silently dropping m*g.
            raise NotImplementedError(
                "accel_is_specific_force=False (gravity-removed accelerometer) is not "
                "supported: this path has no sensor-frame gravity vector. Keep "
                "accel_is_specific_force=true (the Bota IMU reports specific force), or "
                "implement f_imu = accel + gravity_in_sensor and plumb a per-step gravity "
                "vector through update()."
            )
        self.cfg = cfg
        self.alpha_est = AlphaEstimator(cfg.alpha)
        self._last_predicted = np.zeros(6, dtype=np.float64)

    # ---- calibration ----------------------------------------------------------
    def set_bias(self, bias6: np.ndarray) -> None:
        self.cfg.bias = np.asarray(bias6, dtype=np.float64).copy()

    def tare_from_prediction(self, wrench_raw6: np.ndarray, predicted6: np.ndarray) -> None:
        """
        Zero the compensated output for the current (static) pose:
        bias := raw - wrench_sign * predicted, so that update() returns ~0 now.
        Call ONLY while stationary and free of contact.
        """
        self.cfg.bias = (np.asarray(wrench_raw6, dtype=np.float64)
                         - self.cfg.wrench_sign * np.asarray(predicted6, dtype=np.float64))

    # ---- core physics ---------------------------------------------------------
    def predict_payload_wrench(self, timestamp: float, accel: np.ndarray,
                               omega: np.ndarray) -> np.ndarray:
        """
        Predict the payload's gravity+inertial wrench at the sensor origin, in the
        sensor frame, in the *reading* convention (before wrench_sign is applied).
        Returns a length-6 vector [Fx,Fy,Fz, Mx,My,Mz].
        """
        cfg = self.cfg
        m = cfg.mass
        c = cfg.com
        I = cfg.inertia

        accel = np.asarray(accel, dtype=np.float64)
        omega = np.asarray(omega, dtype=np.float64)
        alpha = self.alpha_est.update(timestamp, omega)

        # Specific force at the IMU point (includes gravity). Only this path is
        # supported; accel_is_specific_force=False is rejected in __init__, so accel
        # here is always the sensor's specific-force reading.
        f_imu = accel

        # Transport specific force from IMU point to COM.
        r_ic = c - cfg.imu_offset
        if cfg.enable_com_coupling:
            f_com = f_imu + np.cross(alpha, r_ic) + np.cross(omega, np.cross(omega, r_ic))
        else:
            f_com = f_imu

        # Force: F = m * f_com  (gravity already inside f_com via specific force)
        if cfg.enable_linear_inertia:
            F = m * f_com
        else:
            F = np.zeros(3, dtype=np.float64)

        # Torque about the sensor origin O.
        M = np.zeros(3, dtype=np.float64)
        if cfg.enable_rotational_inertia:
            M = M + I @ alpha + np.cross(omega, I @ omega)
        if cfg.enable_com_coupling:
            M = M + np.cross(c, F)   # moment of F (applied at O) transported through COM offset

        out = np.empty(6, dtype=np.float64)
        out[:3] = F
        out[3:] = M
        self._last_predicted = out
        return out

    # ---- main entry point ------------------------------------------------------
    def update(self, timestamp: float, wrench_imu_raw: np.ndarray,
               wrench_imu_compensated: np.ndarray) -> str:
        """
        Mirror of BotaPayloadCompensator.update. Reads [f(3),t(3),a(3),ω(3)] from
        wrench_imu_raw, writes compensated [f,t] into wrench_imu_compensated[0:6]
        and passes IMU channels through. Writes in place. Returns a status string
        ("OK" / "DEGRADED:<reason>").

        compensated = raw - bias - wrench_sign * predicted_payload_wrench
        """
        raw = np.asarray(wrench_imu_raw, dtype=np.float64)
        if raw.shape[0] != 12:
            raise ValueError(f"wrench_imu_raw must be length 12, got {raw.shape[0]}")

        wrench6 = raw[:6]
        accel = raw[6:9]
        omega = raw[9:12]

        predicted = self.predict_payload_wrench(timestamp, accel, omega)

        comp6 = wrench6 - self.cfg.bias - self.cfg.wrench_sign * predicted

        wrench_imu_compensated[:6] = comp6
        wrench_imu_compensated[6:] = raw[6:]   # pass IMU through, like the Bota demo buffer

        status = "OK"
        if not np.all(np.isfinite(comp6)):
            status = "DEGRADED:non_finite_output"
        return status

    @property
    def last_predicted_wrench(self) -> np.ndarray:
        return self._last_predicted.copy()
