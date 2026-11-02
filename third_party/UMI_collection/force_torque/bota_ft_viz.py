#!/usr/bin/env python
"""Real-time viser visualization of a live Bota force/torque sensor.

Reads the RAW Bota driver stream (force, torque, IMU linear acceleration and
angular rate) and renders it in a browser via viser:

  * three FORCE arrows      (Fx, Fy, Fz)   - red-ish, drawn from the sensor origin
  * three TORQUE arrows     (Tx, Ty, Tz)   - blue-ish
  * one ACCELERATION arrow  (IMU lin. acc) - green
  * one ANGULAR-RATE arrow  (IMU gyro)     - magenta
  * a coordinate-axis triad showing the sensor's CURRENT ORIENTATION, integrated
    live from the gyro (so physically rotating the sensor rotates the triad).
  * four LIVE scrolling time-series charts (viser uPlot) in the GUI panel -
    Force, Torque, Accel, Gyro - each a ~2 s rolling window with x/y/z as
    red/green/blue lines (same style as the Bota reference dashboard).

This is a pure visualizer of the RAW stream - it performs NO payload
gravity/inertia compensation (that is a separate task). It does NOT import any
humanoid robot model.

Note on the orientation triad: it is a complementary filter - gyro integration
for fast motion, continuously corrected toward the accelerometer's gravity
direction so PITCH and ROLL are absolute and drift-free. YAW (heading) has no
gravity reference, so it is still gyro-only and slowly drifts; the "Reset
orientation" button re-zeros heading. Set ``--fusion-gain 0`` for raw gyro-only
dead reckoning.

Architecture (mirrors kimodo/demo/app.py + bota_driver_py_example):
  * one dedicated READER thread runs ``read_frame_blocking()`` and writes the
    latest sample into a lock-guarded ``SensorState``. The sensor streams at
    ~1011 Hz; ``--read-hz`` (default 200) throttles the loop with a GIL-yielding
    sleep so the reader doesn't starve viser's render/websocket threads (pass
    ``--read-hz 0`` to read at the full rate). Decoupling read from render also
    keeps the driver from logging "frames skipped".
  * the main RENDER loop snapshots the state at ~60 Hz and redraws the scene.

The sensor's serial device is AUTO-DETECTED from /dev/serial/by-id (any Bota FT
sensor, regardless of serial number), so swapping sensors / USB ports needs no
config change. Override with --port-path if needed.

Run (live sensor, auto-detected):
    .venv/bin/python force_torque/bota_ft_viz.py --dashboard

Run (no hardware, synthetic data - good for a quick UI check):
    .venv/bin/python force_torque/bota_ft_viz.py --mock
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import shutil
import signal
import tempfile
import threading
import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np
import viser
from viser import uplot

# ----------------------------------------------------------------------------
# Defaults
# ----------------------------------------------------------------------------

# The driver config template (app_mode/baudrate/etc). Its ``com_port`` is
# overridden at runtime by the auto-detected device, so the committed value
# doesn't matter and the tool works across sensor swaps / USB-port changes.
# UMI_collection/force_torque/ -> repo root is one level up; configs live under
# robot_assets/bota_config/ (writable; recalibration overwrites the compensator).
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_BOTA_CFG = os.path.join(_REPO_ROOT, "robot_assets", "bota_config")
_DEFAULT_CONFIG = os.path.join(_BOTA_CFG, "bota_driver", "bota_binary.json")

# Payload compensator config (written by estimate_payload.py) used by --compensate.
_DEFAULT_COMP_CONFIG = os.path.join(_BOTA_CFG, "bota_payload_utils", "payload_compensator.json")
# Estimator config (priors / min-max time) used by in-browser recalibration.
_DEFAULT_EST_CONFIG = os.path.join(_BOTA_CFG, "bota_payload_utils", "payload_estimator.json")
# Independent RNEA (single-body Newton-Euler) dynamic-compensator config, consumed
# by the RNEA algorithm option (rnea_core, vendored under third_party/bota_demo_py).
# The estimator does NOT write this — RNEA payload params are synced from a Bota
# recompensation (see _sync_rnea_payload); conventions (wrench_sign, imu_offset)
# stay fixed. Per-side files (rnea_compensator_<side>.json) resolved by the viz.
_RNEA_CFG_DIR = os.path.join(_BOTA_CFG, "rnea_dynamic_compensator")
_DEFAULT_RNEA_CONFIG = os.path.join(_RNEA_CFG_DIR, "rnea_compensator.json")
# third_party/bota_demo_py/examples/rnea_dynamic_compensator_demo holds rnea_core.py
# (pure numpy, no bota imports) -- the single reference implementation.
_RNEA_CORE_DIR = os.path.join(
    _REPO_ROOT, "third_party", "bota_demo_py", "examples", "rnea_dynamic_compensator_demo"
)


def _import_rnea_core():
    """Lazily import the vendored rnea_core module (pure numpy, no bota deps).

    Returns the module, or raises ImportError. Kept lazy + guarded so a missing /
    broken vendored copy disables the RNEA option without touching the Bota path."""
    import sys as _sys

    if _RNEA_CORE_DIR not in _sys.path:
        _sys.path.insert(0, _RNEA_CORE_DIR)
    import rnea_core  # noqa: PLC0415

    return rnea_core

# udev exposes every Bota FT sensor as a stable by-id symlink like
# /dev/serial/by-id/usb-Bota_Systems_..._<serial>-if00 . Match on the vendor
# string so ANY Bota sensor is found regardless of serial number.
_BOTA_BY_ID_GLOB = "/dev/serial/by-id/*Bota_Systems*Force_Torque_Sensor*-if00"


def find_bota_port(serial: "str | None" = None) -> str | None:
    """Return the stable by-id path of a connected Bota FT sensor, or None.

    Prefers the by-id symlink (survives replug / renumbering).

    ``serial`` PINS a specific sensor by its serial substring (e.g. '3870355D3532'),
    which is MANDATORY once two Bota sensors are attached: the by-id names sort by
    serial, so the unpinned "first sorted match" below silently depends on which serial
    happens to sort lower — 386B… sorts BEFORE 3870…, so the left hand would have
    quietly opened the right hand's sensor. Returns None if that serial is not present,
    rather than falling back to the other sensor: opening the wrong hand's F/T is worse
    than opening none (a mislabeled wrench is indistinguishable from a real one).

    With no ``serial`` the historical behavior is kept exactly: first sorted match."""
    matches = sorted(glob.glob(_BOTA_BY_ID_GLOB))
    if serial:
        hit = [m for m in matches if serial in m]
        return hit[0] if hit else None
    return matches[0] if matches else None


def list_bota_ports() -> list[str]:
    return sorted(glob.glob(_BOTA_BY_ID_GLOB))


def resolve_driver_config(template_path: str, port: str | None) -> str:
    """Return a path to a driver config whose ``com_port`` is ``port``.

    Loads the template JSON, patches ``driver_config.communication_interface_
    params.com_port``, and writes it to a temp file (leaving the committed
    config untouched). If ``port`` is None, returns the template unchanged."""
    if port is None:
        return template_path
    with open(template_path) as f:
        cfg = json.load(f)
    cfg["driver_config"]["communication_interface_params"]["com_port"] = port
    tmp = tempfile.NamedTemporaryFile("w", suffix=".json", prefix="bota_cfg_", delete=False)
    json.dump(cfg, tmp)
    tmp.close()
    return tmp.name

# Colors (R, G, B) 0-255. Force uses the pure R/G/B axis family; torque uses a
# distinct cyan/magenta/yellow family so the six arrows (all drawn from the same
# origin) never read as the same hue when force & torque point the same way.
_COL_FX = (230, 50, 50)
_COL_FY = (50, 210, 70)
_COL_FZ = (60, 110, 250)
_COL_TX = (50, 210, 210)
_COL_TY = (230, 70, 210)
_COL_TZ = (235, 200, 40)
_COL_ACC = (250, 140, 30)
_COL_GYR = (150, 90, 240)


def _hex(rgb: tuple) -> str:
    return "#{:02x}{:02x}{:02x}".format(*rgb)


# Per-axis line colors for the x/y/z traces in each time-series plot. Force and
# torque reuse the same R/G/B-per-axis convention as the reference dashboard;
# acc/gyro use the same so x/y/z read consistently across all four charts.
_AXIS_COLORS = (_hex((220, 50, 50)), _hex((40, 180, 70)), _hex((60, 110, 240)))


# ----------------------------------------------------------------------------
# Shared state between reader thread and render loop
# ----------------------------------------------------------------------------


class TimeSeriesBuffer:
    """Fixed-window rolling history of the wrench + IMU channels for the live
    plots. The reader appends (under the SensorState lock) at a capped cadence;
    the chart updater exports time-relative numpy arrays for uPlot.

    Each channel keeps three component deques (x, y, z). ``t`` holds absolute
    perf_counter timestamps; on export they're shifted so the newest sample is
    at 0 and older samples are negative seconds (matching the reference's
    -2s..0s axis)."""

    def __init__(self, window_s: float = 2.0, max_points: int = 240):
        # max_points caps the deque; at the 50 Hz append rate, window_s (2 s ->
        # ~100 pts) is what's actually shown, with headroom so a brief overrun
        # never truncates the visible window.
        self.window_s = window_s
        self.t: deque[float] = deque(maxlen=max_points)
        # 4 channels x 3 components.
        self._ch = {
            name: tuple(deque(maxlen=max_points) for _ in range(3))
            for name in ("force", "torque", "acc", "gyro")
        }

    def append(self, t: float, force, torque, acc, gyro) -> None:
        self.t.append(t)
        for name, vec in (("force", force), ("torque", torque), ("acc", acc), ("gyro", gyro)):
            comps = self._ch[name]
            comps[0].append(float(vec[0]))
            comps[1].append(float(vec[1]))
            comps[2].append(float(vec[2]))

    def export(self) -> tuple[np.ndarray, dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]] | None:
        """Return (x_rel_seconds, {channel: (x_arr, y_arr, z_arr)}) or None if empty."""
        if not self.t:
            return None
        t = np.asarray(self.t, dtype=float)
        x = t - t[-1]  # newest at 0, older negative
        # Clip to the visible rolling window (e.g. last 2 s).
        keep = x >= -self.window_s
        x = x[keep]
        out = {}
        for name, comps in self._ch.items():
            out[name] = tuple(np.asarray(c, dtype=float)[keep] for c in comps)
        return x, out


@dataclass
class SensorState:
    """Latest decoded sample plus integrated orientation. Lock-guarded."""

    lock: threading.Lock = field(default_factory=threading.Lock)
    history: TimeSeriesBuffer = field(default_factory=TimeSeriesBuffer)
    # ALWAYS-raw rolling history (independent of the show_compensated display
    # toggle), so the viz can plot the live RAW wrench alongside the compensated
    # one for debugging. In the raw-only loop it equals ``history``; in the
    # compensated loop it holds the uncompensated wrench. Same tare offset applied
    # so a tare zeroes this curve too (R5). Only appended by the reader.
    raw_history: TimeSeriesBuffer = field(default_factory=TimeSeriesBuffer)
    raw_n_frames: int = 0
    force: np.ndarray = field(default_factory=lambda: np.zeros(3))
    torque: np.ndarray = field(default_factory=lambda: np.zeros(3))
    acc: np.ndarray = field(default_factory=lambda: np.zeros(3))
    gyro: np.ndarray = field(default_factory=lambda: np.zeros(3))
    temperature: float = 0.0
    # Orientation of the sensor body relative to its start pose, as a wxyz
    # quaternion, integrated from the gyro.
    quat_wxyz: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0]))
    # Wrench tare offsets (subtracted before display), set by the GUI tare button.
    force_offset: np.ndarray = field(default_factory=lambda: np.zeros(3))
    torque_offset: np.ndarray = field(default_factory=lambda: np.zeros(3))
    # Set by the GUI "Reset orientation" button; the reader consumes it and
    # re-zeros its integrator (so the reset isn't overwritten next frame).
    reset_orientation: bool = False
    # When the compensator chain is running, the reader computes BOTH the raw and
    # the gravity/inertia-compensated wrench every tick; this flag (toggled live
    # by the GUI) selects which one is published to force/torque + the charts.
    # ``compensation_available`` says whether the toggle is meaningful at all.
    show_compensated: bool = True
    compensation_available: bool = False
    # Recalibration handshake (browser/GUI -> reader -> browser):
    #   recal_request : set by the UI to ask the reader to re-estimate the payload
    #   recal_phase   : reader-published state machine - "idle"/"estimating"/
    #                   "done"/"failed"
    #   recal_elapsed : seconds since estimation started (for the overlay)
    #   recal_message : short human status for the overlay
    recal_request: bool = False
    recal_phase: str = "idle"
    recal_elapsed: float = 0.0
    recal_message: str = ""
    # One-shot "tare now" handshake (set by the UI, consumed by the reader like
    # recal_request). Fired automatically on each recording START edge (the s key
    # or the GUI Start button). On a
    # COMPENSATED stream the reader re-tares the compensator -> re-zeroes the
    # gravity/inertia-compensated wrench (this CHANGES the recorded ft_wrench_comp
    # and refreshes the tare provenance) and then zeros the display; on a RAW/mock
    # stream (no compensator) it only zeros the display offsets. MUST be triggered
    # in FREE SPACE: contact force present at tare is baked in as an episode-long
    # bias (the captured provenance makes a contaminated tare detectable offline).
    tare_request: bool = False
    # Compensation-algorithm switch (browser/GUI -> reader -> browser), mirroring the
    # recal_request/tare_request one-shot handshake so the reader thread (sole toucher
    # of the driver/compensator) is the only one that swaps the live engine:
    #   algo_request : set by the UI dropdown to ask the reader to switch algorithm
    #                  ("bota" | "rnea"); consumed+cleared at the loop head.
    #   algo         : reader-published CURRENT live algorithm (what's compensating now)
    #   algo_status  : reader->UI note about the active algo (e.g. an RNEA wrench_sign
    #                  UNVERIFIED warning for a hand seeded from the other one).
    # Only meaningful on the COMPENSATED path (compensation_available); the raw/mock
    # loops ignore them.
    algo_request: str | None = None
    algo: str = "bota"
    algo_status: str = ""
    n_frames: int = 0
    rate_hz: float = 0.0
    connected: bool = False
    last_error: str = ""

    def snapshot(self) -> "SensorSample":
        with self.lock:
            return SensorSample(
                force=self.force - self.force_offset,
                torque=self.torque - self.torque_offset,
                acc=self.acc.copy(),
                gyro=self.gyro.copy(),
                temperature=self.temperature,
                quat_wxyz=self.quat_wxyz.copy(),
                n_frames=self.n_frames,
                rate_hz=self.rate_hz,
                connected=self.connected,
                last_error=self.last_error,
            )


@dataclass
class SensorSample:
    force: np.ndarray
    torque: np.ndarray
    acc: np.ndarray
    gyro: np.ndarray
    temperature: float
    quat_wxyz: np.ndarray
    n_frames: int
    rate_hz: float
    connected: bool
    last_error: str


# ----------------------------------------------------------------------------
# Quaternion helpers (wxyz convention, matching viser)
# ----------------------------------------------------------------------------


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product of two wxyz quaternions."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ]
    )


def quat_from_rotvec(rotvec: np.ndarray) -> np.ndarray:
    """wxyz quaternion from a rotation vector (axis * angle, radians)."""
    angle = float(np.linalg.norm(rotvec))
    if angle < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    axis = rotvec / angle
    half = angle * 0.5
    s = math.sin(half)
    return np.array([math.cos(half), axis[0] * s, axis[1] * s, axis[2] * s])


def integrate_gyro(quat: np.ndarray, gyro: np.ndarray, dt: float) -> np.ndarray:
    """Advance ``quat`` (wxyz, body->world) by body-frame angular rate ``gyro``."""
    dq = quat_from_rotvec(gyro * dt)
    out = quat_mul(quat, dq)
    n = np.linalg.norm(out)
    return out / n if n > 0 else np.array([1.0, 0.0, 0.0, 0.0])


def quat_rotate(quat: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """Rotate ``vec`` by wxyz ``quat`` (body->world): v_world = R(quat) @ v_body."""
    w, x, y, z = quat
    # Standard quaternion rotation, expanded.
    t = 2.0 * np.cross([x, y, z], vec)
    return vec + w * t + np.cross([x, y, z], t)


def complementary_correct(quat: np.ndarray, acc: np.ndarray, gain: float) -> np.ndarray:
    """Nudge a gyro-integrated body->world ``quat`` so the gravity direction it
    predicts in the body frame lines up with the measured accelerometer.

    Pure gyro integration drifts; the accelerometer (when the sensor isn't
    strongly accelerating) points along +gravity and pins absolute pitch/roll
    (NOT yaw - gravity carries no heading). ``gain`` in [0, 1] is the per-update
    fraction of the tilt error we correct (small = smooth/slow, 1 = snap).

    Returns ``quat`` unchanged if ``acc`` is unusable (near zero, or far from 1g
    -> sensor is accelerating, so gravity estimate is unreliable).
    """
    g = float(np.linalg.norm(acc))
    if g < 1e-3:
        return quat
    # Reject samples that aren't ~gravity-only (dynamic acceleration corrupts tilt).
    if abs(g - 9.81) > 4.0:
        return quat
    # World "up" seen from the body frame, per the current orientation estimate.
    up_world = np.array([0.0, 0.0, 1.0])
    up_pred_body = quat_rotate(np.array([quat[0], -quat[1], -quat[2], -quat[3]]), up_world)
    meas_up_body = acc / g  # accelerometer reads +up when static
    # Tilt error between predicted and measured up, as an axis-angle in the body
    # frame. Right-multiplying ``quat`` by ``dq`` rotates the body-frame predicted
    # up by R(dq)^-1, so we apply the rotation that takes MEASURED -> PREDICTED
    # (axis = meas x pred); its inverse then pulls predicted toward measured.
    axis = np.cross(meas_up_body, up_pred_body)
    s = float(np.linalg.norm(axis))
    if s < 1e-8:
        return quat
    angle = math.atan2(s, float(np.dot(up_pred_body, meas_up_body)))
    dq = quat_from_rotvec((axis / s) * (angle * gain))
    out = quat_mul(quat, dq)
    n = np.linalg.norm(out)
    return out / n if n > 0 else quat


# ----------------------------------------------------------------------------
# Compensation-algorithm adapter (Bota official vs independent RNEA)
# ----------------------------------------------------------------------------


def _sync_rnea_payload(rnea_config_path: str, comp_config_path: str) -> bool:
    """Copy the payload mass/com/inertia from a Bota compensator JSON into an RNEA
    config JSON, preserving RNEA's geometry/conventions/toggles/alpha_estimator.

    Called after a Bota recompensation so BOTH algorithms compensate the SAME freshly
    estimated payload (the user's mental model: recompensation produces the params,
    the algorithm switch just changes the math). wrench_sign + imu_offset are
    CONVENTIONS, not estimated, so they are left untouched. Best effort: returns True
    on success, False (logged) on any malformed/missing file — never raises into the
    reader loop."""
    try:
        with open(comp_config_path) as fh:
            src = json.load(fh)["payload_compensator_config"]["payload_properties"]
        with open(rnea_config_path) as fh:
            dst = json.load(fh)
        dst.setdefault("rnea_compensator_config", {})["payload_properties"] = {
            "mass": src["mass"],
            "com": src["com"],
            "inertia": src["inertia"],
        }
        tmp = rnea_config_path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(dst, fh, indent=2)
        os.replace(tmp, rnea_config_path)  # atomic swap so a concurrent read never sees a half file
        return True
    except (OSError, KeyError, ValueError) as exc:
        print(f"[rnea] WARNING: could not sync payload {comp_config_path} -> "
              f"{rnea_config_path}: {exc!r}", flush=True)
        return False


class _AlgoCompensator:
    """Holds BOTH compensation engines and exposes ONE algorithm-agnostic interface
    to the reader loop so switching the live algorithm is a single ``.switch(name)``.

    Both engines consume the SAME 12-vector the reader already assembles via
    ``FtChain.assemble`` (raw wrench + IMU transported to the SENSOR ORIGIN) and write
    the compensated wrench in place into the SAME ``chain.comp12`` buffer, so nothing
    downstream (display toggle, recorder sink, charts) changes.

    IMPORTANT (physics): UMI's FtChain transports the IMU to the sensor origin BEFORE
    packing raw12[6:12]. rnea_core, by contrast, expects RAW IMU at the IMU point and
    transports IMU->COM itself using its ``imu_offset``. To avoid double-counting the
    lever arm we build the RNEA compensator with ``imu_offset`` forced to [0,0,0]:
    the COM is already sensor-origin-referenced, so r_ic = com - 0 = com is correct.
    (Gyro is frame-invariant; sensor & IMU axes are aligned, translation-only offset.)

    Return codes are normalized to bota_control.BotaControlReturnCode so the reader's
    existing rc_is_fatal/rc_is_ok / fatal-tolerance logic is untouched: RNEA's string
    status "OK" -> OK, anything else -> Degraded (RNEA has no Stale/Fatal concept)."""

    def __init__(self, bota_comp, rnea_config_path, active, imu_offset):
        self._bota = bota_comp                # bpu.BotaPayloadCompensator (always built)
        self._rnea_config_path = rnea_config_path
        self._imu_offset = imu_offset         # for reference / provenance only
        self._rnea = None                     # built lazily on first RNEA activation
        self._rnea_mod = None
        self._rc = None                       # bota_control.BotaControlReturnCode (set in _ensure_rc)
        self.status_note = ""                 # e.g. wrench_sign UNVERIFIED warning
        # Whether the ACTIVE RNEA config's wrench_sign was verified on THIS hand
        # (recorded provenance: a seeded-from-the-other-hand RNEA config carries an
        # UNVERIFIED sign that can invert the compensated wrench). Meaningless for
        # Bota (the official EKF has no wrench_sign concept), so the ``sign_verified``
        # property reports True there. Set in _build_rnea from the config conventions.
        self._rnea_sign_verified = True
        # Start on the requested engine iff an RNEA config exists; else Bota.
        self.active = "rnea" if (active == "rnea" and rnea_config_path) else "bota"
        if self.active == "rnea":
            self._build_rnea()  # may fall back to "bota" and set status_note on failure

    # ---- return-code normalization ----
    def _ensure_rc(self):
        if self._rc is None:
            import bota_control as _bc  # noqa: PLC0415
            self._rc = _bc.BotaControlReturnCode
        return self._rc

    def _rnea_rc(self, status: str):
        rc = self._ensure_rc()
        return rc.OK if status == "OK" else rc.Degraded

    # ---- RNEA lifecycle ----
    def _build_rnea(self) -> bool:
        """Construct the RNEA compensator from its per-side config, with imu_offset
        zeroed (see class docstring). On any failure, fall back to Bota and record a
        status note. Returns True iff RNEA is now active."""
        try:
            if self._rnea_mod is None:
                self._rnea_mod = _import_rnea_core()
            cfg = self._rnea_mod.RneaCompensatorConfig.from_json(self._rnea_config_path)
            # The 12-vector IMU is ALREADY transported to the sensor origin by
            # FtChain's BotaTfImu -> do NOT let RNEA re-apply the imu->origin lever
            # arm (com is sensor-origin-referenced already, so r_ic = com - 0 = com).
            # CAVEAT (minor, pre-existing): BotaTfImu applies the centripetal term
            # ω×(ω×r) but NOT the tangential α×r term for the ~2.57 cm imu->origin leg,
            # so that small tangential accel is dropped (~1 N of force error only at
            # very high angular acceleration ~50 rad/s²). It affects the standalone
            # rnea demo path differently (there imu_offset≠0 covers the whole leg), but
            # here both engines see the same pre-transported IMU, and it does not touch
            # wrench_sign (a global sign). Acceptable for the force-anti-shake goal.
            cfg.imu_offset = np.zeros(3, dtype=np.float64)
            self._rnea = self._rnea_mod.RneaDynamicCompensator(cfg)
            self.active = "rnea"
            # Surface an UNVERIFIED-sign warning (a hand seeded from the other one)
            # AND record the flag for recorded provenance (a wrong global sign
            # INVERTS the compensated wrench; downstream must be able to segregate
            # such takes offline). Default to verified=True only if the config
            # actively says so; a missing/unreadable flag is treated as UNVERIFIED
            # (conservative — an unknown sign is not a safe sign).
            note = ""
            verified = False
            try:
                with open(self._rnea_config_path) as fh:
                    conv = json.load(fh)["rnea_compensator_config"].get("conventions", {})
                verified = bool(conv.get("wrench_sign_verified") is True)
                if not verified:
                    note = ("RNEA wrench_sign UNVERIFIED for this hand — validate with a "
                            "static tilt (compensated |F| should stay ~0 as you reorient).")
            except (OSError, KeyError, ValueError):
                note = "RNEA wrench_sign UNVERIFIED (config conventions unreadable)."
            self._rnea_sign_verified = verified
            self.status_note = note
            return True
        except Exception as exc:  # noqa: BLE001 - never let a bad RNEA config kill the loop
            print(f"[rnea] WARNING: could not build RNEA compensator from "
                  f"{self._rnea_config_path}: {exc!r}; staying on Bota.", flush=True)
            self._rnea = None
            self.active = "bota"
            self.status_note = f"RNEA unavailable ({type(exc).__name__}); using Bota."
            return False

    def rebuild_rnea(self) -> None:
        """Rebuild the RNEA engine from its (possibly just-synced) config. No-op if
        RNEA was never built. Keeps the active engine unchanged unless it fails."""
        if self._rnea is not None or self.active == "rnea":
            was_active = self.active == "rnea"
            self._rnea = None
            self._build_rnea()
            if not was_active and self.active == "rnea":
                # We only wanted to refresh the model, not switch to it.
                self.active = "bota"

    def switch(self, name: str) -> bool:
        """Switch the active engine. Returns True if the active engine changed."""
        name = "rnea" if name == "rnea" else "bota"
        if name == self.active:
            return False
        if name == "rnea":
            if self._rnea_config_path is None:
                self.status_note = "no RNEA config for this hand; staying on Bota."
                return False
            if self._rnea is None:
                self._build_rnea()
            else:
                self.active = "rnea"
            return self.active == "rnea"
        # -> bota
        self.active = "bota"
        self.status_note = ""
        return True

    @property
    def rnea_available(self) -> bool:
        return self._rnea_config_path is not None

    @property
    def sign_verified(self) -> bool:
        """Whether the ACTIVE engine's global sign is trusted on this hand. Always
        True for Bota (no wrench_sign concept); the RNEA config's
        wrench_sign_verified flag when RNEA is active. Recorded per take so an
        unverified-sign RNEA episode is detectable offline."""
        return True if self.active != "rnea" else self._rnea_sign_verified

    # ---- live compensation (identical signature to BotaPayloadCompensator.update) ----
    def update(self, timestamp, raw12, comp12):
        if self.active == "rnea" and self._rnea is not None:
            return self._rnea_rc(self._rnea.update(timestamp, raw12, comp12))
        return self._bota.update(timestamp, raw12, comp12)

    # ---- tare (each engine tares differently) ----
    def tare(self, driver, chain, stop_event=None, frame_sink=None) -> None:
        """Re-zero the ACTIVE engine's compensated output in free space.

        Bota: async tare_compensated_wrench() + warm_up_tare (drain until it settles).
        RNEA: synchronous — average a short window of (raw6, predicted6) and set bias
        via tare_from_prediction (bias valid immediately; no warm-up needed).

        ``frame_sink`` (optional): called with each frame this tare drains from the
        driver, BEFORE it is consumed. The recorder passes it so the RAW wrench keeps
        flowing during the ~0.5 s (Bota) / ~1 s (RNEA) tare window — otherwise the
        drained frames never reach the recorder and every episode opens with an F/T
        hole. Only raw is meaningful here (the compensated output is Stale/settling),
        which is exactly what the sink records for this window."""
        from bota_compat import warm_up_tare  # noqa: PLC0415

        if self.active == "rnea" and self._rnea is not None:
            n = 0
            acc_raw = np.zeros(6, dtype=np.float64)
            acc_pred = np.zeros(6, dtype=np.float64)
            deadline = time.perf_counter() + 1.0   # ~1 s static window, like the demo
            while time.perf_counter() < deadline:
                if stop_event is not None and stop_event.is_set():
                    break
                frame = driver.read_frame_blocking()
                if frame_sink is not None:
                    frame_sink(frame)
                raw12 = chain.assemble(frame)
                pred = self._rnea.predict_payload_wrench(
                    time.perf_counter(), raw12[6:9], raw12[9:12])
                acc_raw += raw12[:6]
                acc_pred += pred
                n += 1
            if n > 0:
                self._rnea.tare_from_prediction(acc_raw / n, acc_pred / n)
            return
        # Bota
        self._bota.tare_compensated_wrench()
        if not warm_up_tare(driver, self._bota, chain, stop_event=stop_event,
                            frame_sink=frame_sink):
            print("[bota] WARNING: compensated tare did not settle within timeout.",
                  flush=True)

    def tare_offset6(self, driver, chain):
        """Return the 6-vector DC offset the active engine tared out, for recorder
        provenance, or None if it can't be determined on this sample.

        Defined IDENTICALLY for both engines as ``raw12 - comp12`` on one live sample
        (the wrench the compensator actually removes at the tare pose). This is the
        near-full gravity wrench for a clean free-space tare and, crucially, carries
        any CONTACT force present at tare — which is exactly what the offline
        contaminated-tare detector inspects. Reporting cfg.bias for RNEA instead would
        give a ~0 residual (gravity lives in the prediction, not the bias), a quantity
        semantically different from Bota's by ~one g of force — so the single untagged
        recorder ft_tare_wrench field would mean different things per engine. Keep them
        the same. (Bota needs a fully-settled OK sample; RNEA's synchronous tare is
        already valid, so any finite comp sample is fine.)"""
        from bota_compat import rc_is_ok  # noqa: PLC0415

        frame = driver.read_frame_blocking()
        raw12 = chain.assemble(frame)
        rc = self.update(time.perf_counter(), raw12, chain.comp12)
        if not rc_is_ok(rc):
            return None
        return [float(raw12[i] - chain.comp12[i]) for i in range(6)]


# ----------------------------------------------------------------------------
# Sensor reader thread
# ----------------------------------------------------------------------------


class SensorReader(threading.Thread):
    """Dedicated thread: reads the Bota stream as fast as possible.

    With ``mock=True`` it synthesizes a plausible signal instead of opening
    hardware, so the UI/eval can run with no sensor attached.
    """

    def __init__(
        self,
        state: SensorState,
        config_path: str,
        mock: bool = False,
        read_hz: float = 200.0,
        fusion_gain: float = 0.02,
        compensator_config: str | None = None,
        estimator_config: str | None = None,
        recal_countdown_s: float = 5.0,
        recal_max_time: float = 60.0,
        raw_sink=None,
        *,
        rnea_config: str | None = None,
        algo: str = "bota",
    ):
        super().__init__(daemon=True)
        self.state = state
        # Optional lossless sink for the FULL-RATE wrench (used by the data
        # recorder). Called every read with (t_wall, raw_force, raw_torque,
        # comp_force, comp_torque); must be microsecond-cheap (it runs on this
        # high-rate reader thread). None = off. The RAW wrench is ALWAYS passed (the
        # recorder keeps it as the irreversible audit/safety net); the COMPENSATED
        # wrench is passed too when this reader runs the payload-compensated loop
        # (else None). NOTE: compensation is NOT reproducible offline — it needs the
        # sensor IMU orientation per sample, which the recorder doesn't store — so
        # the live compensated trace is captured here rather than recomputed later.
        self.raw_sink = raw_sink
        # Most-recent compensator tare offset + its wall-clock time, for the
        # recorder's provenance (a startup tare taken with the tool in contact
        # silently DC-biases every compensated sample — recording the offset/time
        # makes a bad tare detectable). Updated at startup tare, each
        # recompensation, AND each recording-start auto-tare.
        #
        # Stored as ONE tuple ``(wrench, wall)`` (or None), NOT two fields: the
        # reader thread writes it while the publisher thread reads it via
        # ``tare_getter`` on EVERY ft_batch (now on every episode start, not just
        # once at startup). Two separate attributes would let the reader get a torn
        # pair — (new_wrench, old_wall) or vice-versa — corrupting the safety
        # provenance the offline pipeline trusts. A single tuple object is published
        # with one atomic reference store, so the getter always sees a consistent
        # pair (or the whole previous one). ``tare_wrench``/``tare_wall`` are exposed
        # as read-only properties derived from it for existing callers.
        self._tare = None            # (offs6, wall) the compensator tared out, or None
        # Live compensation-algorithm provenance for the recorder, as ONE atomic
        # tuple ``(algo_name, sign_verified)`` (single reference store, same
        # discipline as ``_tare``): the reader thread writes it whenever the active
        # engine changes; the publisher WS thread reads it per ft_batch via
        # ``algo_getter``. Recording which engine (bota|rnea) compensated each take —
        # and whether the RNEA sign was verified — is what lets the offline pipeline
        # segregate takes that silently mixed two compensation models (a switched
        # dropdown mid-session). None until the compensated loop publishes it.
        self._algo_prov = None       # (algo_name, sign_verified) or None
        self.config_path = config_path
        self.mock = mock
        # If set, run the Bota payload compensator: force/torque published to the
        # state become the COMPENSATED wrench (tool gravity + inertia removed),
        # so arrows + charts show real external contact force. None = raw stream.
        self.compensator_config = compensator_config
        # For in-browser recalibration: the estimator config + how long the
        # "get ready / rotate" countdown overlay runs before estimation starts.
        self.estimator_config = estimator_config
        # Independent RNEA dynamic-compensator config (per-side rnea_compensator_*.json)
        # + the algorithm to START with. Both only matter on the compensated path;
        # the live engine is swapped by the algo_request handshake. algo is normalized
        # to "rnea" only when an RNEA config is actually available (else "bota").
        self.rnea_config = rnea_config
        self.algo = "rnea" if (str(algo).lower() == "rnea" and rnea_config) else "bota"
        self.recal_countdown_s = recal_countdown_s
        # Max estimation window for recalibration (overrides the config's
        # max_time; 0 = use the config as-is).
        self.recal_max_time = recal_max_time
        # Complementary-filter strength for accel gravity correction of the
        # orientation triad (0 = gyro-only dead reckoning, drifts; higher =
        # snappier pitch/roll lock but noisier). Scaled by dt in the loop.
        self.fusion_gain = fusion_gain
        # ``read_frame_blocking`` is a C call that holds the GIL while it waits
        # (~1 ms at the 1 kHz sensor rate). A tight loop therefore starves the
        # viser render/websocket threads, which freezes the browser. A viz only
        # needs ~60 Hz, so we throttle reads with a GIL-yielding sleep. <=0
        # disables the throttle (full rate - only safe if nothing else runs).
        self.read_period = (1.0 / read_hz) if read_hz and read_hz > 0 else 0.0
        # Append to the plot history at most ~50 Hz: a 2 s window then holds
        # ~100 points/channel, plenty for smooth curves without flooding the
        # rolling deques (or the websocket) at the full read rate.
        self._plot_period = 1.0 / 50.0
        self._last_plot_t = 0.0
        self._stop_event = threading.Event()
        self._driver = None

    def stop(self) -> None:
        self._stop_event.set()

    # Provenance readers (cross-thread): derived from the single ``_tare`` tuple so
    # a concurrent publisher-thread read never sees a torn (wrench, wall) pair.
    @property
    def tare_wrench(self):
        t = self._tare  # one atomic reference load
        return list(t[0]) if t is not None else None

    @property
    def tare_wall(self):
        t = self._tare
        return t[1] if t is not None else None

    # -- main loop ----------------------------------------------------------

    def run(self) -> None:
        try:
            if self.mock:
                self._run_mock()
            else:
                self._run_hardware()
        except Exception as exc:  # noqa: BLE001 - surface any driver error to the UI
            with self.state.lock:
                self.state.connected = False
                self.state.last_error = str(exc)

    def _run_hardware(self) -> None:
        if self.compensator_config:
            self._run_hardware_compensated()
            return

        import bota_driver

        driver = bota_driver.BotaDriver(self.config_path)
        self._driver = driver
        try:
            if not driver.configure():
                raise RuntimeError("driver.configure() failed")
            if not driver.activate():
                raise RuntimeError("driver.activate() failed")

            with self.state.lock:
                self.state.connected = True

            quat = np.array([1.0, 0.0, 0.0, 0.0])
            last_t = time.perf_counter()
            rate_ema = 0.0
            while not self._stop_event.is_set():
                # One-shot tare (recording START edge). No
                # compensator on this raw path, so it's display-only (B): zero the
                # offsets subtracted before charting. Recorded raw wrench unchanged.
                if self._consume_tare_request():
                    self._zero_display_offset()
                frame = driver.read_frame_blocking()
                now = time.perf_counter()
                dt = now - last_t
                last_t = now
                # Clamp dt before deriving an instantaneous rate: the very first
                # iteration (and any scheduler hiccup) can yield a near-zero dt
                # that spikes the EMA to absurd values.
                if dt > 1e-4:
                    inst = 1.0 / dt
                    rate_ema = 0.98 * rate_ema + 0.02 * inst if rate_ema else inst
                # ...and clamp it before integrating orientation, so a long
                # first-frame / stall latency can't snap the triad by a big angle.
                dt_int = min(dt, 0.05)

                gyro = np.asarray(frame.angular_rate, dtype=float)
                acc = np.asarray(frame.acceleration, dtype=float)
                quat = integrate_gyro(quat, gyro, dt_int)
                # Complementary fusion: pull pitch/roll toward the accelerometer's
                # gravity direction so the triad doesn't drift (yaw stays gyro-only).
                if self.fusion_gain > 0.0:
                    quat = complementary_correct(quat, acc, self.fusion_gain * min(dt_int / 0.005, 1.0))

                force = np.asarray(frame.force, dtype=float)
                torque = np.asarray(frame.torque, dtype=float)
                # Lossless full-rate sink for the recorder (RAW wrench, wall-clock).
                if self.raw_sink is not None:
                    self.raw_sink(time.time(), force, torque)
                with self.state.lock:
                    # Honor a GUI reset: re-zero the integrator here (in the
                    # reader) so the reset persists instead of being overwritten.
                    if self.state.reset_orientation:
                        quat = np.array([1.0, 0.0, 0.0, 0.0])
                        self.state.reset_orientation = False
                    self.state.force = force
                    self.state.torque = torque
                    self.state.acc = acc
                    self.state.gyro = gyro
                    self.state.temperature = float(frame.temperature)
                    self.state.quat_wxyz = quat
                    self.state.n_frames += 1
                    self.state.rate_hz = rate_ema
                    if now - self._last_plot_t >= self._plot_period:
                        self._last_plot_t = now
                        self.state.history.append(
                            now, force - self.state.force_offset, torque - self.state.torque_offset, acc, gyro
                        )
                        # Raw-only loop: the raw curve equals the display curve here
                        # (force/torque ARE the raw wrench). Keep raw_history filled
                        # so the viz's raw plot works uniformly across both loops.
                        self.state.raw_history.append(
                            now, force - self.state.force_offset, torque - self.state.torque_offset, acc, gyro
                        )
                        self.state.raw_n_frames += 1

                # Yield the GIL so the render/websocket threads aren't starved.
                if self.read_period:
                    time.sleep(self.read_period)
        finally:
            try:
                driver.deactivate()
                driver.shutdown()
            except Exception:  # noqa: BLE001
                pass
            with self.state.lock:
                self.state.connected = False

    def _run_hardware_compensated(self) -> None:
        """Runs the driver + compensator chain (gen-A API: caller-driven driver +
        bota_compat.FtChain 12-vector assembly, no HWI wrapper). The compensator is an
        _AlgoCompensator holding BOTH the Bota official engine and (when a per-side
        RNEA config is present) the independent RNEA engine; the LIVE algorithm is
        switched by the ``algo_request`` handshake, exactly like recal/tare. The LIVE
        state shows EITHER the raw or the gravity/inertia-compensated wrench, selected
        by ``state.show_compensated`` (GUI toggle), but the recorder sink ALWAYS gets
        both raw and compensated (recording is decoupled from the display toggle). The
        orientation triad's IMU (accel/gyro) comes straight from the raw driver frame,
        so it is unaffected by compensation."""
        import bota_driver
        import bota_payload_utils as bpu

        from bota_compat import FATAL_TOLERANCE, IMU_OFFSET, FtChain, rc_is_fatal, rc_is_ok

        driver = bota_driver.BotaDriver(self.config_path)
        bota_comp = bpu.BotaPayloadCompensator(self.compensator_config)
        # gen-A (>=1.0) toolkit: no create_ft_sensor_hwi wrapper. The caller drives
        # the driver lifecycle and assembles the 12-vector (raw wrench + IMU
        # transformed into the sensor frame) itself; ``chain`` owns that transform
        # and its reusable buffers. One chain for the whole session (shared with
        # recalibration) so the TF block's internal state isn't reset mid-stream.
        chain = FtChain(IMU_OFFSET)
        # The algorithm adapter: Bota engine always built; RNEA built lazily on first
        # activation. ``comp`` is the sole live-compensator handle for this thread.
        comp = _AlgoCompensator(bota_comp, self.rnea_config, self.algo, IMU_OFFSET)
        try:
            if not driver.configure():
                raise RuntimeError("driver.configure() failed")
            if not driver.activate():
                raise RuntimeError("driver.activate() failed")
            # Tare the ACTIVE engine (Bota async warm-up, or RNEA synchronous average).
            comp.tare(driver, chain, stop_event=self._stop_event, frame_sink=self._tare_frame_sink())
            self._capture_tare_algo(comp, driver, chain)
            self._publish_algo_prov(comp)

            with self.state.lock:
                self.state.connected = True
                self.state.compensation_available = True
                self.state.algo = comp.active
                self.state.algo_status = comp.status_note

            quat = np.array([1.0, 0.0, 0.0, 0.0])
            last_t = time.perf_counter()
            rate_ema = 0.0
            consec_fatal = 0
            consec_nonok = 0
            while not self._stop_event.is_set():
                # In-browser recalibration: re-estimate the payload on this same
                # driver, then swap in a fresh compensator. Blocks the live view for
                # the duration (countdown + estimation), driving the overlay. Check
                # BEFORE reading a frame so a recal request isn't paired with a
                # now-discarded compensation sample.
                with self.state.lock:
                    do_recal = self.state.recal_request
                    self.state.recal_request = False
                if do_recal and self.estimator_config:
                    # Re-estimate the Bota payload AND sync it into the RNEA config so
                    # both engines compensate the same fresh payload; rebuild the RNEA
                    # engine from the refreshed config.
                    new_bota = self._recalibrate(driver, bpu, chain)
                    if new_bota is not None:
                        comp._bota = new_bota  # swap the Bota engine held by the adapter
                        if self.rnea_config:
                            _sync_rnea_payload(self.rnea_config, self.compensator_config)
                            comp.rebuild_rnea()
                        # _recalibrate already tared+captured provenance for the fresh
                        # Bota engine (async warm-up). So RE-tare here ONLY when the
                        # active engine is RNEA (rebuild_rnea produced an untared RNEA
                        # engine); taring Bota again would be a redundant ~1s warm-up
                        # whose provenance capture just overwrites _recalibrate's.
                        if comp.active == "rnea":
                            comp.tare(driver, chain, stop_event=self._stop_event, frame_sink=self._tare_frame_sink())
                            self._capture_tare_algo(comp, driver, chain)
                        self._reset_display_offset()
                        self._publish_algo_prov(comp)
                        with self.state.lock:
                            self.state.algo = comp.active
                            self.state.algo_status = comp.status_note
                    last_t = time.perf_counter()
                    consec_fatal = 0
                    continue

                # One-shot algorithm switch (GUI dropdown). Reader-thread-only swap of
                # the live engine, re-taring the newly-active engine in free space so
                # its compensated output starts at ~0 (each engine has its own bias).
                with self.state.lock:
                    algo_req = self.state.algo_request
                    self.state.algo_request = None
                if algo_req is not None and algo_req != comp.active:
                    comp.switch(algo_req)
                    comp.tare(driver, chain, stop_event=self._stop_event, frame_sink=self._tare_frame_sink())
                    self._capture_tare_algo(comp, driver, chain)
                    self._reset_display_offset()
                    self._publish_algo_prov(comp)
                    with self.state.lock:
                        self.state.algo = comp.active
                        self.state.algo_status = comp.status_note
                    print(f"[ft] compensation algorithm -> {comp.active}"
                          + (f"  ({comp.status_note})" if comp.status_note else ""), flush=True)
                    last_t = time.perf_counter()
                    consec_fatal = 0
                    continue

                # One-shot tare (fired on each recording START edge).
                # Compensated stream -> re-tare the ACTIVE engine (A: changes
                # the recorded compensated wrench + refreshes provenance) THEN reset
                # the display offset to zero (B: the signal itself is now re-zeroed,
                # so the offset must be 0, NOT the stale pre-tare wrench). Checked at
                # the loop head like recal so we're the sole toucher of the HWI. The
                # tare may drain its own window, so reset the dt clock afterwards and
                # skip a frame.
                if self._consume_tare_request():
                    comp.tare(driver, chain, stop_event=self._stop_event, frame_sink=self._tare_frame_sink())
                    self._capture_tare_algo(comp, driver, chain)
                    self._reset_display_offset()
                    last_t = time.perf_counter()
                    consec_fatal = 0
                    continue

                frame = driver.read_frame_blocking()
                raw12 = chain.assemble(frame)  # [wrench(6), imu_in_sensor(6)]
                rc = comp.update(time.perf_counter(), raw12, chain.comp12)
                if rc_is_fatal(rc) or chain.tf_is_fatal():
                    # A transient bad dt (frame burst vs clock) yields a fatal (on the
                    # IMU TF and/or the compensator) that recovers on the next clean
                    # sample - skip it, don't kill the thread. Only a SUSTAINED run
                    # means the stream is truly broken.
                    consec_fatal += 1
                    if consec_fatal > FATAL_TOLERANCE:
                        raise RuntimeError(
                            f"compensator.update() returned FatalError for "
                            f"{consec_fatal} consecutive samples (stream broken)"
                        )
                    last_t = time.perf_counter()
                    continue
                consec_fatal = 0
                # A non-OK-but-non-fatal sample (Stale while a re-tare collects its
                # window, or Degraded) means comp12 is zeros / reduced-quality, not a
                # real compensated wrench. We still publish (raw stays the always-valid
                # audit net, and the multi-rate converter prefers the recorded comp
                # only when ft_compensated says so), but a SUSTAINED run is silent
                # degradation — surface it with a throttled log so it's not invisible.
                if not rc_is_ok(rc):
                    consec_nonok += 1
                    if consec_nonok == 1 or consec_nonok % 1000 == 0:
                        print(f"[bota] compensator non-OK (rc={rc}) for {consec_nonok} "
                              f"samples; compensated wrench may be zeros/degraded.", flush=True)
                else:
                    consec_nonok = 0
                now = time.perf_counter()
                dt = now - last_t
                last_t = now
                if dt > 1e-4:
                    inst = 1.0 / dt
                    rate_ema = 0.98 * rate_ema + 0.02 * inst if rate_ema else inst
                dt_int = min(dt, 0.05)

                # Orientation triad uses the RAW driver-frame IMU (same as the raw
                # reader path); the sensor-frame-transformed IMU only feeds the
                # compensator's 12-vector.
                gyro = np.asarray(frame.angular_rate, dtype=float)
                acc = np.asarray(frame.acceleration, dtype=float)
                quat = integrate_gyro(quat, gyro, dt_int)
                if self.fusion_gain > 0.0:
                    quat = complementary_correct(quat, acc, self.fusion_gain * min(dt_int / 0.005, 1.0))

                # raw12 + chain.comp12 are REUSED buffers (overwritten next tick);
                # copy the wrench slices before they reach shared state / the sink.
                raw_f = np.array(raw12[:3])
                raw_t = np.array(raw12[3:6])
                cmp_f = np.array(chain.comp12[:3])
                cmp_t = np.array(chain.comp12[3:6])
                # Lossless full-rate sink for the recorder: BOTH the raw wrench
                # (kept as the audit/safety net) AND the live compensated wrench
                # (gravity/inertia removed — what training prefers; can't be
                # recomputed offline without per-sample IMU orientation). Decoupled
                # from the show_compensated display toggle on purpose.
                if self.raw_sink is not None:
                    self.raw_sink(time.time(), raw_f, raw_t, cmp_f, cmp_t)
                with self.state.lock:
                    if self.state.reset_orientation:
                        quat = np.array([1.0, 0.0, 0.0, 0.0])
                        self.state.reset_orientation = False
                    # Live RAW/COMPENSATED selection (GUI toggle).
                    if self.state.show_compensated:
                        force, torque = cmp_f, cmp_t
                    else:
                        force, torque = raw_f, raw_t
                    self.state.force = force
                    self.state.torque = torque
                    self.state.acc = acc
                    self.state.gyro = gyro
                    self.state.temperature = float(frame.temperature)
                    self.state.quat_wxyz = quat
                    self.state.n_frames += 1
                    self.state.rate_hz = rate_ema
                    if now - self._last_plot_t >= self._plot_period:
                        self._last_plot_t = now
                        self.state.history.append(
                            now, force - self.state.force_offset, torque - self.state.torque_offset, acc, gyro
                        )
                        # Always-raw curve (compensated loop): plot the RAW wrench
                        # regardless of the display toggle, same tare offset applied.
                        self.state.raw_history.append(
                            now, raw_f - self.state.force_offset, raw_t - self.state.torque_offset, acc, gyro
                        )
                        self.state.raw_n_frames += 1

                if self.read_period:
                    time.sleep(self.read_period)
        finally:
            try:
                driver.deactivate()
                driver.shutdown()
            except Exception:  # noqa: BLE001
                pass
            with self.state.lock:
                self.state.connected = False

    def _capture_tare(self, driver, comp, chain) -> None:
        """Record the DC offset the compensator just tared out + the wall-clock,
        for recorder provenance. The compensator subtracts a fixed offset captured
        at the current pose/contact state; if the tool was in contact at tare, every
        compensated sample carries that bias. We read it back as (raw - compensated)
        right after the tare so a contaminated tare is detectable offline. Best
        effort: any driver/shape hiccup leaves tare_wrench None (logged once).

        REQUIRES a fully-settled (rc_is_ok) sample: while the 1.0.2 async tare is
        still collecting its window, update() returns Stale and writes ZEROS into
        comp12, so (raw - 0) would latch the entire raw wrench as the bogus 'tare
        offset' — poisoning the very provenance the offline pipeline trusts to
        detect a contaminated tare. So only capture on OK; on a non-OK sample leave
        tare_wrench None (logged) rather than record a wrong value."""
        try:
            from bota_compat import rc_is_ok

            frame = driver.read_frame_blocking()
            raw12 = chain.assemble(frame)
            rc = comp.update(time.perf_counter(), raw12, chain.comp12)
            if not rc_is_ok(rc):
                # Tare not settled / degraded — a captured offset would be wrong.
                print(f"[bota] tare-provenance capture skipped: compensator not OK "
                      f"(rc={rc}); leaving tare offset unrecorded.", flush=True)
                return
            offs = [float(raw12[i] - chain.comp12[i]) for i in range(6)]
            # Publish offset+time as ONE tuple (single atomic store) so the
            # publisher thread's tare_getter can't read a torn (wrench, wall) pair.
            self._tare = (offs, time.time())
        except Exception as exc:  # noqa: BLE001 - provenance is advisory, never fatal
            print(f"[bota] tare-provenance capture skipped: {exc!r}", flush=True)

    def _capture_tare_algo(self, comp, driver, chain) -> None:
        """Algorithm-agnostic tare-provenance capture: record the DC offset the ACTIVE
        engine tared out + the wall-clock, as one atomic (offs6, wall) tuple (same
        contract as _capture_tare). Delegates to _AlgoCompensator.tare_offset6 (Bota:
        raw-comp on an OK sample; RNEA: cfg.bias). None -> leave provenance unrecorded
        (logged). Provenance is advisory; never fatal."""
        try:
            offs = comp.tare_offset6(driver, chain)
            if offs is None:
                print(f"[ft] tare-provenance capture skipped for algo={comp.active} "
                      "(engine not OK); leaving tare offset unrecorded.", flush=True)
                return
            self._tare = (offs, time.time())
        except Exception as exc:  # noqa: BLE001 - provenance is advisory, never fatal
            print(f"[ft] tare-provenance capture skipped: {exc!r}", flush=True)

    def _tare_frame_sink(self):
        """Build a frame_sink for comp.tare that keeps the RAW wrench flowing to the
        recorder during the tare's frame-drain window (else every episode opens with
        a ~0.5-1 s F/T hole). Returns None when there's no raw_sink (nothing records),
        so the tare drains as before. Pushes ONLY raw (comp_force/torque=None): the
        compensated output is Stale/settling across the tare, so the recorder writes
        no compensated sample for this window — raw is the meaningful trace here and
        stays gap-free. Uses wall-clock time.time() to match the live loop's sink
        timestamps (the recorder keys on wall-clock)."""
        if self.raw_sink is None:
            return None

        def _sink(frame):
            try:
                self.raw_sink(time.time(),
                              np.asarray(frame.force, dtype=float),
                              np.asarray(frame.torque, dtype=float))
            except Exception:  # noqa: BLE001 - a sink hiccup must never break the tare
                pass
        return _sink

    def _publish_algo_prov(self, comp) -> None:
        """Store the active engine's algorithm-provenance tuple (name, sign_verified)
        for the recorder, as ONE atomic reference store (like ``_tare``) so the
        publisher WS thread never reads a torn pair. Called on the reader thread
        wherever the active engine is (re)established or switched."""
        self._algo_prov = (comp.active, bool(comp.sign_verified))

    def _consume_tare_request(self) -> bool:
        """Atomically read+clear state.tare_request. Reader-thread only.

        Mirrors the recal_request consume at the loop head so a UI-set flag is
        acted on exactly once by the sole toucher of the Bota HWI."""
        with self.state.lock:
            do_tare = self.state.tare_request
            self.state.tare_request = False
        return do_tare

    def _zero_display_offset(self) -> None:
        """Display-only tare (B) for the RAW/mock path (no compensator): copy the
        current published wrench into the SensorState offsets so the charts/header
        read zero from here on. Identical to the GUI 'Tare wrench' button; changes
        NOTHING that is recorded (the recorder captures the raw/compensated wrench,
        not these display offsets)."""
        with self.state.lock:
            self.state.force_offset = self.state.force.copy()
            self.state.torque_offset = self.state.torque.copy()

    def _reset_display_offset(self) -> None:
        """Display tare (B) for the COMPENSATED path: RESET the display offsets to
        zero. On this path the compensator re-tare (A) has already re-zeroed the
        COMPENSATED signal itself, so the freshly-compensated wrench is ~0 in free
        space; the chart shows it directly. Capturing the current reading here
        instead would be WRONG — state.force still holds the STALE pre-tare wrench
        (warm_up_tare reads its ~0.5 s window locally and never updates state.force),
        so the next real frame (~0) minus that stale offset would display -old_wrench
        rather than zero. Zeroing the offset lets the new compensated ~0 show as ~0."""
        with self.state.lock:
            self.state.force_offset = np.zeros(3)
            self.state.torque_offset = np.zeros(3)

    def _set_recal(self, phase: str, elapsed: float = 0.0, message: str = "") -> None:
        with self.state.lock:
            self.state.recal_phase = phase
            self.state.recal_elapsed = elapsed
            self.state.recal_message = message

    def _recalibrate(self, driver, bpu, chain):
        """Re-estimate the payload on the live driver, then return a fresh
        compensator. Drives state.recal_* for the browser overlay. Returns the new
        compensator on success, or None (keep the old one). ``chain`` is the shared
        FtChain (IMU transform + 12-vector buffers) so the TF block isn't reset."""
        import json as _json
        import tempfile as _tempfile

        from bota_compat import FATAL_TOLERANCE, rc_is_fatal, warm_up_tare

        out_tmp = None
        etmp_name = None
        try:
            # 1) Countdown so the operator can grab the sensor and free it of
            #    contact. Inside the try so a stream hiccup here can't escape and
            #    wedge the overlay at "countdown".
            t0 = time.perf_counter()
            while True:
                remaining = self.recal_countdown_s - (time.perf_counter() - t0)
                if remaining <= 0 or self._stop_event.is_set():
                    break
                driver.read_frame_blocking()  # keep the stream alive during the countdown
                self._set_recal(
                    "countdown", remaining, f"Get ready - lift the tool, no contact. Starting in {remaining:0.0f}s"
                )
                time.sleep(0.1)

            # 2) Run the estimator on the same driver stream. Write to an absolute
            #    path (no chdir races), into a temp file, then copy onto the config.
            ecfg = _json.load(open(self.estimator_config))
            ec = ecfg["payload_estimator_config"]
            # The estimator only needs the PATH to write into; create+close so we
            # don't leak the open fd (the estimator opens/writes it itself).
            _otmp = _tempfile.NamedTemporaryFile("w", suffix=".json", prefix="recal_out_", delete=False)
            out_tmp = _otmp.name
            _otmp.close()
            ec["output_config_file_name"] = out_tmp
            # Extend the estimation window (default 30s is often too short to
            # converge by hand; 60s matches what worked in manual estimation).
            if self.recal_max_time:
                ec["max_time"] = self.recal_max_time
                ec["min_time"] = min(float(ec.get("min_time", 15.0)), self.recal_max_time)
            etmp = _tempfile.NamedTemporaryFile("w", suffix=".json", prefix="recal_est_", delete=False)
            etmp_name = etmp.name
            _json.dump(ecfg, etmp)
            etmp.close()
            max_t = float(ec.get("max_time", 30.0))

            estimator = bpu.BotaPayloadEstimator(etmp_name)
            if not estimator.start_payload_estimation():
                self._set_recal("failed", 0.0, "could not start estimation")
                return None

            # Pace the estimator feed to ~1 kHz like the standalone estimate_payload.py
            # (which converges): a fixed `time.sleep(0.001)` EVERY iteration roughly
            # HALVES the effective rate and the EKF then can't converge. Sleep only
            # the remainder to the next 1 ms tick, and throttle the overlay update +
            # its state.lock to ~10 Hz so it isn't contending at the feed rate. Log
            # the measured rate each second so a convergence failure is diagnosable
            # (rate-starved vs insufficient physical excitation).
            est_t0 = time.perf_counter()
            loop_period = 1.0 / 1000.0
            next_t = time.perf_counter()
            n_updates = 0
            consec_fatal = 0
            last_overlay = 0.0
            last_rate_log = est_t0
            while estimator.is_estimator_running() and not self._stop_event.is_set():
                frame = driver.read_frame_blocking()
                rc = estimator.update(time.perf_counter(), chain.assemble(frame))
                if rc_is_fatal(rc):
                    # Tolerate a transient bad-dt fatal (recovers next clean sample);
                    # only a sustained run means the stream is broken.
                    consec_fatal += 1
                    if consec_fatal > FATAL_TOLERANCE:
                        self._set_recal("failed", time.perf_counter() - est_t0,
                                        "estimator returned FatalError (stream broken)")
                        return None
                    continue
                consec_fatal = 0
                n_updates += 1
                now = time.perf_counter()
                el = now - est_t0
                if now - last_overlay >= 0.1:
                    last_overlay = now
                    self._set_recal(
                        "estimating", el, f"ROTATE the tool slowly through many poses ({el:0.0f}/{int(max_t)}s)"
                    )
                if now - last_rate_log >= 1.0:
                    print(f"[recal] estimating {el:0.0f}/{int(max_t)}s  feed={n_updates / el:0.0f} Hz", flush=True)
                    last_rate_log = now
                next_t += loop_period
                dt = next_t - time.perf_counter()
                if dt > 0:
                    time.sleep(dt)
                elif dt < -loop_period:
                    next_t = time.perf_counter()  # fell behind; resync, don't spiral

            if not estimator.is_estimation_successful():
                print(f"[recal] FAILED to converge after {time.perf_counter() - est_t0:0.0f}s "
                      f"(mean feed {n_updates / max(time.perf_counter() - est_t0, 1e-6):0.0f} Hz)", flush=True)
                self._set_recal("failed", time.perf_counter() - est_t0,
                                "did not converge - rotate through more poses, keep free of contact, retry")
                return None

            # 3) Install the new payload + build a fresh compensator.
            shutil.copy(out_tmp, self.compensator_config)
            # Force enable_dynamic_compensation=false BEFORE building the compensator:
            # the estimator .so hardcodes this flag into the file it writes (the
            # installed bpu 1.0.2 writes false; 0.0.2 wrote true), and no config/API
            # knob controls it. This rig runs GRAVITY-ONLY compensation by design, so
            # pin it false regardless of what the estimator wrote — else a bpu 0.0.2
            # estimator (or a downgrade) would silently re-enable inertia compensation.
            # Log loudly if it had to flip so a regression is visible; best effort
            # (never wedge the recal).
            try:
                _cfg = _json.load(open(self.compensator_config))
                _pc = _cfg["payload_compensator_config"]
                if _pc.get("enable_dynamic_compensation") is not False:
                    print(f"[recal] WARNING: estimator wrote enable_dynamic_compensation="
                          f"{_pc.get('enable_dynamic_compensation')!r}; forcing false "
                          f"(this rig runs gravity-only compensation by design)", flush=True)
                _pc["enable_dynamic_compensation"] = False
                with open(self.compensator_config, "w") as _f:
                    _json.dump(_cfg, _f, indent=2)
            except (OSError, KeyError, ValueError) as exc:
                print(f"[recal] WARNING: could not force enable_dynamic_compensation: {exc!r}", flush=True)
            new_comp = bpu.BotaPayloadCompensator(self.compensator_config)
            new_comp.tare_compensated_wrench()
            # 1.0.2 tare is async; drain frames until it settles before capturing
            # provenance / returning the compensator to the live loop.
            if not warm_up_tare(driver, new_comp, chain, stop_event=self._stop_event):
                print("[recal] WARNING: new compensator's tare did not settle; "
                      "tare provenance may be skipped this take.", flush=True)
            # Refresh tare provenance: a recompensation re-tares (and swaps the
            # payload model), so the offset baked into subsequent compensated
            # samples changes. The recorder, if recording, flags this discontinuity.
            self._capture_tare(driver, new_comp, chain)
            self._set_recal("done", 0.0, "recalibration complete")
            return new_comp
        except Exception as exc:  # noqa: BLE001
            self._set_recal("failed", 0.0, str(exc))
            return None
        finally:
            for p in (out_tmp, etmp_name):
                if p:
                    try:
                        os.unlink(p)
                    except OSError:
                        pass

    def _run_mock(self) -> None:
        with self.state.lock:
            self.state.connected = True
        quat = np.array([1.0, 0.0, 0.0, 0.0])
        last_t = time.perf_counter()
        t0 = last_t
        while not self._stop_event.is_set():
            # One-shot tare (recording START edge). Mock has no
            # compensator, so display-only (B): keeps the flag from dangling and the
            # chart zeroes on episode start just like the real raw path.
            if self._consume_tare_request():
                self._zero_display_offset()
            now = time.perf_counter()
            dt = now - last_t
            last_t = now
            t = now - t0
            force = np.array([6.0 * math.sin(t), 4.0 * math.cos(0.7 * t), -9.81 + math.sin(2 * t)])
            torque = np.array([0.5 * math.sin(0.5 * t), 0.3 * math.cos(t), 0.2 * math.sin(1.5 * t)])
            acc = np.array([9.81, 0.5 * math.sin(t), 0.5 * math.cos(t)])
            gyro = np.array([0.0, 0.0, 0.6])  # steady yaw spin so the triad rotates
            quat = integrate_gyro(quat, gyro, min(dt, 0.05))
            with self.state.lock:
                if self.state.reset_orientation:
                    quat = np.array([1.0, 0.0, 0.0, 0.0])
                    self.state.reset_orientation = False
                self.state.force = force
                self.state.torque = torque
                self.state.acc = acc
                self.state.gyro = gyro
                self.state.temperature = 37.0
                self.state.quat_wxyz = quat
                self.state.n_frames += 1
                self.state.rate_hz = 1000.0
                if now - self._last_plot_t >= self._plot_period:
                    self._last_plot_t = now
                    self.state.history.append(
                        now, force - self.state.force_offset, torque - self.state.torque_offset, acc, gyro
                    )
                    # Mock reader has no separate compensated wrench; the raw curve
                    # mirrors the display curve so the viz's raw plot still animates.
                    self.state.raw_history.append(
                        now, force - self.state.force_offset, torque - self.state.torque_offset, acc, gyro
                    )
                    self.state.raw_n_frames += 1
            time.sleep(0.001)


# ----------------------------------------------------------------------------
# Viser scene
# ----------------------------------------------------------------------------


class ForceTorqueScene:
    """Owns the viser server, scene primitives, and GUI controls."""

    def __init__(
        self,
        server: viser.ViserServer,
        state: SensorState,
        panel_charts: bool = True,
        mode: str = "RAW",
        compensation_available: bool = False,
        rnea_available: bool = False,
    ):
        self.server = server
        self.state = state
        # When the viser scene is embedded in the bottom-dock dashboard, the
        # charts are drawn by that page instead, so we skip the in-panel uPlot
        # charts to avoid duplicating them.
        self.panel_charts = panel_charts
        # "RAW" or "COMPENSATED" - shown in the GUI so it's clear what the
        # arrows/charts represent.
        self.mode = mode
        # Whether the RAW<->COMPENSATED toggle is usable (compensator running).
        self.compensation_available = compensation_available
        # Whether the RNEA algorithm option should be offered (RNEA config present).
        self.rnea_available = rnea_available
        self.gui_algo = None
        # Echo-suppress flag while the render thread reconciles gui_algo.value back to
        # the reader's actual state.algo (so that write doesn't re-request a swap).
        self._algo_echo = False

        server.scene.set_up_direction("+z")
        server.scene.add_frame(
            "/world",
            show_axes=False,
            position=(0.0, 0.0, 0.0),
        )
        # A faint reference triad at the origin (the sensor's START pose).
        server.scene.add_frame(
            "/world/ref",
            show_axes=True,
            axes_length=0.25,
            axes_radius=0.004,
            position=(0.0, 0.0, 0.0),
        )
        # The sensor body whose orientation we update live from the gyro.
        self.body = server.scene.add_frame(
            "/world/sensor",
            show_axes=True,
            axes_length=0.6,
            axes_radius=0.012,
            origin_radius=0.03,
            position=(0.0, 0.0, 0.0),
        )
        # A small puck to visually anchor the sensor origin.
        server.scene.add_icosphere(
            "/world/sensor/puck",
            radius=0.04,
            color=(120, 120, 130),
            subdivisions=2,
        )

        # Cached arrow handles + last-visibility, so a hidden vector is toggled
        # via handle.visible (one message, on change) instead of re-creating an
        # invisible arrow every render tick.
        self._arrows: dict[str, viser.ArrowsHandle] = {}
        self._arrow_visible: dict[str, bool] = {}
        # Signature of the last draw; skip the whole redraw when nothing changed
        # (no new sensor frame and no GUI control moved).
        self._last_sig: tuple | None = None
        # n_frames at the last chart push; skip re-pushing identical arrays.
        self._last_chart_n: int = -1

        self._build_gui()

    # -- GUI ----------------------------------------------------------------

    def _build_gui(self) -> None:
        gui = self.server.gui
        # control_width="large" = the widest the stock viser API offers (~384px,
        # roughly a third of a laptop screen). The right panel holds the four live
        # charts in single-port mode, so make it as wide as possible.
        gui.configure_theme(control_layout="collapsible", control_width="large")

        with gui.add_folder("Status"):
            self.gui_mode = gui.add_text("Wrench", initial_value=self.mode, disabled=True)
            self.gui_conn = gui.add_text("Connection", initial_value="connecting...", disabled=True)
            self.gui_rate = gui.add_text("Rate (Hz)", initial_value="0", disabled=True)
            self.gui_temp = gui.add_text("Temp (C)", initial_value="0", disabled=True)

        with gui.add_folder("Force / Torque (N, Nm)"):
            self.gui_force = gui.add_text("Force", initial_value="", disabled=True)
            self.gui_torque = gui.add_text("Torque", initial_value="", disabled=True)
            self.gui_fmag = gui.add_text("|F| / |T|", initial_value="", disabled=True)

        with gui.add_folder("IMU"):
            self.gui_acc = gui.add_text("Lin acc (m/s^2)", initial_value="", disabled=True)
            self.gui_gyr = gui.add_text("Ang rate (rad/s)", initial_value="", disabled=True)

        with gui.add_folder("Legend / notes", expand_by_default=False):
            gui.add_markdown(
                "**Force** R=x G=y B=z &nbsp; **Torque** cyan=x magenta=y yellow=z\n\n"
                "**Acc** orange (incl. gravity) &nbsp; **Ang-rate** purple\n\n"
                "Triad = orientation: gyro + **accelerometer gravity fusion**, so "
                "pitch/roll are absolute & drift-free; yaw is gyro-only (drifts, "
                "use Reset). Faint triad = start pose."
            )

        with gui.add_folder("Display"):
            self.gui_force_scale = gui.add_slider("Force len/N", min=0.005, max=0.2, step=0.005, initial_value=0.05)
            self.gui_torque_scale = gui.add_slider("Torque len/Nm", min=0.05, max=2.0, step=0.05, initial_value=0.5)
            self.gui_acc_scale = gui.add_slider("Acc len/(m/s^2)", min=0.005, max=0.2, step=0.005, initial_value=0.03)
            self.gui_gyr_scale = gui.add_slider("Gyro len/(rad/s)", min=0.05, max=2.0, step=0.05, initial_value=0.5)
            self.gui_show_force = gui.add_checkbox("Show force", initial_value=True)
            self.gui_show_torque = gui.add_checkbox("Show torque", initial_value=True)
            self.gui_show_acc = gui.add_checkbox("Show acceleration", initial_value=True)
            self.gui_show_gyro = gui.add_checkbox("Show angular rate", initial_value=True)
            self.gui_combined = gui.add_checkbox("Combined F/T vectors", initial_value=False)

        with gui.add_folder("Controls"):
            tare_btn = gui.add_button("Tare wrench")
            reset_btn = gui.add_button("Reset orientation")
            # Live RAW <-> COMPENSATED toggle. Only meaningful when the
            # compensator chain is running (i.e. launched with --compensate);
            # otherwise the reader publishes raw only and this is disabled.
            self.gui_compensate = gui.add_checkbox(
                "Compensated wrench",
                initial_value=self.compensation_available,
                disabled=not self.compensation_available,
            )
            # Live compensation-ALGORITHM switch (Bota official vs independent RNEA).
            # The recompensation button is unchanged; this only changes which math
            # compensates the wrench. Only offered when an RNEA config is present.
            if self.rnea_available:
                with self.state.lock:
                    _algo0 = self.state.algo
                self.gui_algo = gui.add_dropdown(
                    "Compensation algorithm", ("bota", "rnea"), initial_value=_algo0)
            self.gui_pause = gui.add_checkbox("Pause", initial_value=False)

        @tare_btn.on_click
        def _(_event) -> None:
            with self.state.lock:
                self.state.force_offset = self.state.force.copy()
                self.state.torque_offset = self.state.torque.copy()

        @reset_btn.on_click
        def _(_event) -> None:
            with self.state.lock:
                self.state.reset_orientation = True

        @self.gui_compensate.on_update
        def _(_event) -> None:
            with self.state.lock:
                self.state.show_compensated = self.gui_compensate.value
            # Reflect the new mode in the status readout immediately.
            self.gui_mode.value = "COMPENSATED" if self.gui_compensate.value else "RAW"

        if self.gui_algo is not None:
            @self.gui_algo.on_update
            def _(_event) -> None:
                # Ignore the echo when the render thread reconciles the value back to
                # the actual engine (after a fallback); only a real user pick requests
                # a swap. Hand the request to the reader thread (sole toucher of HWI).
                if self._algo_echo:
                    return
                with self.state.lock:
                    self.state.algo_request = self.gui_algo.value

        # Live scrolling time-series charts (uPlot) - one per channel, stacked
        # at the bottom of the panel: Force, Torque, Accel, Gyro. Each shows the
        # last ~2 s with x/y/z as red/green/blue lines (same style as the Bota
        # reference dashboard). Updated from the rolling history buffer.
        self._charts: dict[str, viser.GuiUplotHandle] = {}
        if self.panel_charts:
            with gui.add_folder("Live plots"):
                self._charts["force"] = self._make_chart("Force (N)", ("Fx", "Fy", "Fz"))
                self._charts["torque"] = self._make_chart("Torque (Nm)", ("Tx", "Ty", "Tz"))
                self._charts["acc"] = self._make_chart("Accel (m/s^2)", ("Ax", "Ay", "Az"))
                self._charts["gyro"] = self._make_chart("Gyro (rad/s)", ("Gx", "Gy", "Gz"))

    def _make_chart(self, title: str, labels: tuple[str, str, str]) -> viser.GuiUplotHandle:
        """Create an empty 3-line scrolling plot (x = time, y = x/y/z)."""
        x0 = np.array([-2.0, 0.0])
        zero = np.zeros(2)
        return self.server.gui.add_uplot(
            data=(x0, zero, zero, zero),
            series=(
                uplot.Series(label="t"),
                uplot.Series(label=labels[0], stroke=_AXIS_COLORS[0], width=1.5),
                uplot.Series(label=labels[1], stroke=_AXIS_COLORS[1], width=1.5),
                uplot.Series(label=labels[2], stroke=_AXIS_COLORS[2], width=1.5),
            ),
            scales={"x": uplot.Scale(time=False)},
            title=title,
            aspect=2.2,
            height=120,
        )

    def update_charts(self) -> None:
        with self.state.lock:
            n = self.state.n_frames
            exported = self.state.history.export()
        # Skip the push when no new sample has arrived since the last chart
        # update (avoids re-sending identical arrays at the chart rate when idle).
        if n == self._last_chart_n:
            return
        self._last_chart_n = n
        if exported is None:
            return
        x, channels = exported
        if x.size < 2:
            return
        for name, handle in self._charts.items():
            cx, cy, cz = channels[name]
            handle.data = (x, cx, cy, cz)

    # -- per-frame draw -----------------------------------------------------

    @staticmethod
    def _component_arrows(vec: np.ndarray, scale: float, cols: tuple) -> tuple[np.ndarray, np.ndarray]:
        """Three axis-aligned arrows (one per component), origin -> component."""
        pts = np.zeros((3, 2, 3))
        for i in range(3):
            pts[i, 1, i] = vec[i] * scale
        return pts, np.array(cols)

    def _draw_arrow(self, name: str, pts: np.ndarray, cols: np.ndarray, shown: bool, **kw) -> None:
        """Update one arrow group: overwrite geometry when shown (handles expose
        no geometry setter), otherwise just toggle the cached handle invisible -
        so a hidden vector costs one message on change, not one per tick."""
        if shown:
            self._arrows[name] = self.server.scene.add_arrows(name, pts, cols, **kw)
            self._arrow_visible[name] = True
        elif self._arrow_visible.get(name, True):
            handle = self._arrows.get(name)
            if handle is not None:
                handle.visible = False
            self._arrow_visible[name] = False

    def update(self, sample: SensorSample) -> None:
        # Skip the entire redraw when nothing changed: no new sensor frame AND
        # no GUI control moved. (The orientation triad only changes with a new
        # frame, so this is safe.) Keeps the websocket quiet when the sensor is
        # idle or the tab is in the background.
        # Reconcile the algorithm dropdown + mode readout to the engine the reader
        # ACTUALLY runs (after an RNEA build/switch fallback to bota the widget would
        # otherwise lie; the ⚠ note surfaces the reason). Done every call (cheap; the
        # widget only re-renders on a real change) and folded into the redraw signature
        # so an algo change alone forces a redraw.
        with self.state.lock:
            live_algo = self.state.algo
            live_algo_status = self.state.algo_status
            live_show_comp = self.state.show_compensated and self.state.compensation_available
        if self.gui_algo is not None and self.gui_algo.value != live_algo:
            self._algo_echo = True
            try:
                self.gui_algo.value = live_algo
            finally:
                self._algo_echo = False
        if self.compensation_available:
            base = "COMPENSATED" if live_show_comp else "RAW"
            self.gui_mode.value = (f"{base} [{live_algo}]" + ("  ⚠" if (live_show_comp and live_algo_status) else "")
                                   if self.rnea_available else base)

        sig = (
            sample.n_frames,
            self.gui_show_force.value,
            self.gui_show_torque.value,
            self.gui_show_acc.value,
            self.gui_show_gyro.value,
            self.gui_combined.value,
            self.gui_force_scale.value,
            self.gui_torque_scale.value,
            self.gui_acc_scale.value,
            self.gui_gyr_scale.value,
            live_algo,
        )
        if sig == self._last_sig:
            return
        self._last_sig = sig

        # Orientation triad.
        self.body.wxyz = sample.quat_wxyz

        # Force / torque: three per-component arrows, or one combined resultant.
        # All parented under /world/sensor so they rotate with the body frame
        # (the Bota stream is expressed in the sensor frame).
        if self.gui_combined.value:
            f_pts = np.array([[[0, 0, 0], list(sample.force * self.gui_force_scale.value)]], dtype=float)
            self._draw_arrow(
                "/world/sensor/force", f_pts, np.array([_COL_FX]), self.gui_show_force.value,
                shaft_radius=0.012, head_radius=0.028,
            )
            t_pts = np.array([[[0, 0, 0], list(sample.torque * self.gui_torque_scale.value)]], dtype=float)
            self._draw_arrow(
                "/world/sensor/torque", t_pts, np.array([_COL_TX]), self.gui_show_torque.value,
                shaft_radius=0.012, head_radius=0.028,
            )
        else:
            f_pts, f_cols = self._component_arrows(
                sample.force, self.gui_force_scale.value, (_COL_FX, _COL_FY, _COL_FZ)
            )
            self._draw_arrow(
                "/world/sensor/force", f_pts, f_cols, self.gui_show_force.value,
                shaft_radius=0.01, head_radius=0.024,
            )
            t_pts, t_cols = self._component_arrows(
                sample.torque, self.gui_torque_scale.value, (_COL_TX, _COL_TY, _COL_TZ)
            )
            self._draw_arrow(
                "/world/sensor/torque", t_pts, t_cols, self.gui_show_torque.value,
                shaft_radius=0.008, head_radius=0.02,
            )

        # IMU linear-acceleration arrow.
        a_pts = np.array([[[0, 0, 0], list(sample.acc * self.gui_acc_scale.value)]], dtype=float)
        self._draw_arrow(
            "/world/sensor/acc", a_pts, np.array([_COL_ACC]), self.gui_show_acc.value,
            shaft_radius=0.009, head_radius=0.022,
        )

        # IMU angular-rate arrow.
        g_pts = np.array([[[0, 0, 0], list(sample.gyro * self.gui_gyr_scale.value)]], dtype=float)
        self._draw_arrow(
            "/world/sensor/gyro", g_pts, np.array([_COL_GYR]), self.gui_show_gyro.value,
            shaft_radius=0.009, head_radius=0.022,
        )

        # GUI numeric readouts.
        self.gui_conn.value = "connected" if sample.connected else (sample.last_error or "disconnected")
        self.gui_rate.value = f"{sample.rate_hz:.0f}"
        self.gui_temp.value = f"{sample.temperature:.1f}"
        self.gui_force.value = _fmt(sample.force)
        self.gui_torque.value = _fmt(sample.torque)
        self.gui_fmag.value = f"{np.linalg.norm(sample.force):.2f} / {np.linalg.norm(sample.torque):.3f}"
        self.gui_acc.value = _fmt(sample.acc)
        self.gui_gyr.value = _fmt(sample.gyro)


def _fmt(v: np.ndarray) -> str:
    return f"[{v[0]:+.3f}, {v[1]:+.3f}, {v[2]:+.3f}]"


# ----------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=_DEFAULT_CONFIG, help="Bota driver JSON config template")
    parser.add_argument(
        "--port-path",
        default=None,
        help="serial device path for the sensor; default = auto-detect any "
        "connected Bota sensor via /dev/serial/by-id (works across sensor "
        "swaps / serial numbers)",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--render-hz", type=float, default=60.0, help="scene redraw rate")
    parser.add_argument(
        "--read-hz",
        type=float,
        default=200.0,
        help="cap on sensor-read rate; throttles the reader so it doesn't starve "
        "the viser threads via the GIL (<=0 = full rate, ~1 kHz)",
    )
    parser.add_argument(
        "--fusion-gain",
        type=float,
        default=0.02,
        help="accelerometer gravity-fusion strength for the orientation triad "
        "(0 = gyro-only dead reckoning; ~0.02 = drift-corrected pitch/roll)",
    )
    parser.add_argument("--mock", action="store_true", help="synthetic data, no hardware")
    parser.add_argument(
        "--dashboard",
        action="store_true",
        help="serve a bottom-dock dashboard page (3D scene on top 2/3 via iframe, "
        "Force/Torque + Accel/Gyro charts docked across the bottom 1/3) instead of "
        "the in-panel charts. Opens on --dashboard-port.",
    )
    parser.add_argument("--dashboard-port", type=int, default=8081, help="port for the bottom-dock dashboard")
    parser.add_argument(
        "--compensate",
        action="store_true",
        help="show the payload-COMPENSATED wrench (tool gravity + inertia removed "
        "via BotaPayloadCompensator) instead of the raw stream. Requires a payload "
        "estimated by estimate_payload.py.",
    )
    parser.add_argument(
        "--compensator-config",
        default=_DEFAULT_COMP_CONFIG,
        help="payload compensator JSON (from estimate_payload.py)",
    )
    parser.add_argument(
        "--ft-algo",
        choices=("bota", "rnea"),
        default=os.environ.get("FT_ALGO", "bota"),
        help="INITIAL compensation algorithm: 'bota' (official bota_payload_utils, "
        "gravity-only as this rig runs it) or 'rnea' (independent single-body "
        "Newton-Euler dynamic compensation). Switchable live in the GUI. Default bota.",
    )
    parser.add_argument(
        "--rnea-config",
        default=_DEFAULT_RNEA_CONFIG,
        help="RNEA compensator JSON (payload synced from a Bota recompensation; "
        "conventions like wrench_sign are fixed). Enables the RNEA GUI option.",
    )
    parser.add_argument(
        "--recal-countdown",
        type=float,
        default=5.0,
        help="seconds of fullscreen countdown before in-browser recalibration starts",
    )
    parser.add_argument(
        "--recal-max-time",
        type=float,
        default=60.0,
        help="max seconds for the recalibration estimation window (default 60; the "
        "stock 30s is often too short to converge by hand)",
    )
    args = parser.parse_args()

    # Resolve the serial device: explicit --port-path wins, else auto-detect any
    # connected Bota sensor (so a new sensor / different serial just works). The
    # detected port is patched into a temp copy of the config template; the
    # committed config is never modified. Skipped in --mock.
    config_path = args.config
    if not args.mock:
        port = args.port_path or find_bota_port()
        if port is None:
            others = list_bota_ports()
            print("ERROR: no Bota sensor found under /dev/serial/by-id/.")
            print("  - check the USB cable / power, or pass --port-path /dev/ttyACMx")
            print("  - or run with --mock to test the UI without hardware")
            if others:
                print(f"  (candidates seen: {others})")
            return
        print(f"Using sensor: {port}")
        config_path = resolve_driver_config(args.config, port)

    state = SensorState()

    # Run the compensator chain whenever a payload config exists (live mode), so
    # the RAW<->COMPENSATED toggle is always usable in the GUI. --compensate only
    # picks the INITIAL display mode. Skipped in --mock (no real wrench).
    comp_cfg = None
    comp_available = False
    if not args.mock and args.compensator_config and os.path.exists(args.compensator_config):
        comp_cfg = args.compensator_config
        comp_available = True
    elif args.compensate and not args.mock:
        print(f"NOTE: --compensate but no payload config at {args.compensator_config}; showing RAW only.")
    if args.compensate and args.mock:
        print("NOTE: --compensate ignored in --mock (no real wrench to compensate)")
    state.show_compensated = bool(args.compensate and comp_available)
    _mode = "COMPENSATED" if state.show_compensated else "RAW"

    # RNEA algorithm option: only offered when the compensator chain runs AND an
    # RNEA config exists (pure-numpy engine, vendored under third_party/bota_demo_py).
    rnea_cfg = args.rnea_config if (comp_available and args.rnea_config
                                    and os.path.exists(args.rnea_config)) else None
    algo0 = args.ft_algo if rnea_cfg else "bota"
    if args.ft_algo == "rnea" and not rnea_cfg:
        print(f"NOTE: --ft-algo rnea but no RNEA config at {args.rnea_config}; starting on bota.")
    state.algo = algo0

    # Build the viser server BEFORE starting the reader. The reader's C-level
    # ``read_frame_blocking`` holds the GIL, and a busy reader during server
    # startup will deadlock viser's bootstrap threads.
    server = viser.ViserServer(host=args.host, port=args.port, label="Bota F/T Sensor")
    # In dashboard mode the charts are drawn by the wrapper page, so skip the
    # in-panel uPlot charts (avoids duplicating them inside the iframe).
    scene = ForceTorqueScene(
        server, state, panel_charts=not args.dashboard, mode=_mode,
        compensation_available=comp_available, rnea_available=bool(rnea_cfg),
    )

    dashboard = None
    if args.dashboard:
        from dashboard import DashboardServer

        # The iframe loads viser from the user's browser, so it must point at a
        # host the browser can reach. localhost works for same-machine viewing.
        viser_url = f"http://localhost:{args.port}"
        dashboard = DashboardServer(state, args.host, args.dashboard_port, viser_url, mode=_mode)
        dashboard.start()

    reader = SensorReader(
        state,
        config_path,
        mock=args.mock,
        read_hz=args.read_hz,
        fusion_gain=args.fusion_gain,
        compensator_config=comp_cfg,
        estimator_config=(_DEFAULT_EST_CONFIG if comp_available else None),
        recal_countdown_s=args.recal_countdown,
        recal_max_time=args.recal_max_time,
        rnea_config=rnea_cfg,
        algo=algo0,
    )
    reader.start()

    stop = threading.Event()

    def _sig(_signum, _frame):
        stop.set()

    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)

    mode = "COMPENSATED (gravity/inertia removed)" if comp_cfg else "RAW"
    print(f"Wrench mode: {mode}")
    if comp_cfg:
        print(f"  payload: {comp_cfg}")
    if args.dashboard:
        print(f"Dashboard:  http://localhost:{args.dashboard_port}   (3D scene + bottom-dock charts)")
        print(f"(viser 3D scene also at http://localhost:{args.port})")
    else:
        print(f"Viser serving on http://localhost:{args.port}  (mock={args.mock})")
    period = 1.0 / args.render_hz
    chart_period = 1.0 / 15.0  # charts are heavier; refresh slower than the 3D scene
    last_chart_t = 0.0
    try:
        while not stop.is_set():
            t = time.perf_counter()
            sample = state.snapshot()
            if not scene.gui_pause.value:
                scene.update(sample)
                if t - last_chart_t >= chart_period:
                    last_chart_t = t
                    scene.update_charts()
            sleep = period - (time.perf_counter() - t)
            if sleep > 0:
                time.sleep(sleep)
    finally:
        reader.stop()
        reader.join(timeout=2.0)
        if dashboard is not None:
            dashboard.stop()
        server.stop()
        # Remove the temp config we wrote for the auto-detected port.
        if config_path != args.config:
            try:
                os.unlink(config_path)
            except OSError:
                pass
        print("Shut down cleanly.")


if __name__ == "__main__":
    main()
