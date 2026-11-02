#!/usr/bin/env python3
"""VIVE tracker bridge — a source-built libsurvive stack, on this rig's WebSocket wire.

WHAT THIS IS. The tracker handling here follows a reference UMI data-collection stack
(its ``HeadViveTracker`` / ``ViveTracker`` readers, the per-role startup waits, the
``--list-serials`` detector, the libsurvive cache reset and the library pin).

Its rules, kept verbatim:

  1. libsurvive is SOURCE-BUILT at commit ``5588cadf`` (2026-05-14) against the SYSTEM
     libusb, and the ctypes bindings come from the SAME checkout
     (``LD_LIBRARY_PATH=~/libsurvive/bin`` + ``PYTHONPATH=~/libsurvive/bindings/python``).
     The PyPI ``pysurvive`` wheel (1.1.21, uploaded 2023-03-11, bundling libsurvive
     "1.2.7" and its own libusb) is REFUSED at startup: three years of Gen2-lighthouse
     OOTX / tracking fixes are missing from it, and its libusb segfaults on some hosts.
  2. ONE ``SimpleContext`` per process, shared by every tracker — pysurvive is a
     per-process singleton (a second holder gets ``LIBUSB_ERROR_BUSY``).
  3. Flags ``--force-calibrate 1 --force-ootx 1 --lighthouse-update-velocity 0.5`` on a
     reset ``~/.config/libsurvive/config.json`` -- ``--calib-mode fresh``, the default and
     the reference recorder flow (``umi_collection/sensors/vive_tracker.py``:
     "lighthouses are RE-SOLVED at every process start"). Every station is re-read and
     re-solved, so each launch is its own world frame; a stale cached station POSE would
     silently yield wrong poses in recorded data. ``ootx-only`` / ``reuse`` (see
     :data:`CALIB_MODES`) are escape hatches, not a workflow.
     Measured here 2026-09-11 with SIX stations x 5 pucks: ``poser_mpfit.c`` attempts NO
     room solve until EVERY station it has seen has its OOTX blob, so 4 seen / 2 decoded /
     0 solved in 150 s -- the count of powered stations is the lever (2 is the floor; the
     launcher's ``STATIONS`` / ``MIN_STATIONS``), not a second calibration pass.
  4. A tracker is keyed by its stable ``LHR-`` SERIAL, read with
     ``simple_serial_number(obj.ptr)``; the internal ``T20``/``WM0`` names are assigned
     per run and can swap between two physical pucks.
  5. Startup BLOCKS until every required tracker has produced a pose -- the reference recorder's
     ``wait_until_tracking`` (``--vive-warmup-timeout``, 90 s there; 120 s here); a
     reference gripper driver waits up to 150 s per wrist tracker -- logging every 5 s, and
     FAILS if one never does, because a recording with a missing role is worthless.
     Nothing for the operator to do meanwhile beyond keeping every puck powered and in the
     stations' view. Cameras start only after (they share USB bandwidth with the dongles).
  6. Every pose carries ``received_at``; one older than the stale window is INVALID,
     never silently re-used (reference: 1.0 s).
  7. NO base-station counting, NO expected-station pin, NO wave gate, NO solver or
     stack-size knobs: libsurvive uses every station whose channel it sees.

WHAT IS OURS (the boundary, byte-identical to before): the role <-> serial CLI
(``--mode`` + ``--*-serial``), the world alignment (``--world-rpy-deg``: one
``R_world`` left-applied to position and quaternion — rig geometry, not tracker
handling), and the WebSocket wire on ``:8772`` that ``recording/recorder.py``,
``viz/collection_viz.py``, ``bridges/health_probe.py`` and
``bridges/tracker/axis_check.py`` consume::

    {"mode": "1pt"|"2pt"|"3pt"|"5pt", "t": float,
     "trackers": {role: {"seq": int, "t": float, "pos": [x, y, z],
                         "quat": [w, x, y, z], "stale": bool}},
     "lighthouses": {"count": int, "names": [...]}}     # optional; provenance only

A role is ABSENT until its tracker has produced a first pose (consumers tolerate a
partial map during startup). ``quat`` is WXYZ, pysurvive's native order — the
recorder stores it verbatim. ``stale`` defaults to the recorder's historical 0.25 s
window (a DATA flag; the reference 1.0 s is the startup-validity threshold, ``--max-age``).

Why ``WORLD_RPY`` survives a fresh solve on every launch: libsurvive anchors its world on
the station with the LOWEST hardware ``BaseStationID`` (``src/poser.c``, "Using LH %d as
reference lighthouse"; overridable with ``--survive-arg --reference-basestation
--survive-arg <id>``), not on discovery order — so the same physical station defines the
axes every time until it is moved.

What the startup wait adds on top of the reference flow (diagnosis, not behaviour): every 5 s and
on FATAL it prints a per-ROLE table — on the radio? has a pose? which stations' light has
it synced to? — and a VERDICT that separates a puck fault (off the radio / sees no station
/ lost its pose / on the radio under a serial no role names) from the SHARED blocker
nobody can get a pose before (no base station positioned yet: OOTX x/y, "move the pucks").
The light counts come from libsurvive's own ``sync`` and ``log`` hooks (:class:`LightMonitor`),
never from its structs — see that class for why. Nothing here gates anything (rule 7).

One deliberate deviation from the reference collector loop: ``NextUpdated()`` is
non-blocking (``survive_simple_get_next_updated`` returns 0 when nothing changed), so
their bare ``continue`` spins a core; an empty poll here sleeps 0.5 ms. Same data,
same order, same latency at the 240 Hz a puck reports.

Run (the launcher sets the env; run bare, the bridge re-execs itself with it when
``$LIBSURVIVE_DIR`` — default ``~/libsurvive`` — holds the source build)::

    export LD_LIBRARY_PATH=$HOME/libsurvive/bin:$LD_LIBRARY_PATH
    export PYTHONPATH=$HOME/libsurvive/bindings/python:$PYTHONPATH
    .venv-vive/bin/python bridges/tracker/vive_tracker_bridge.py --mode 5pt \
        --left-serial LHR-... --right-serial LHR-... --head-serial LHR-... \
        --left-foot-serial LHR-... --right-foot-serial LHR-... --world-rpy-deg 0 0 180
    .venv-vive/bin/python bridges/tracker/vive_tracker_bridge.py --list-serials   # who is on the radio
"""

from __future__ import annotations

import argparse
import asyncio
import ctypes
import json
import math
import os
import re
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field

import numpy as np

import libsurvive_cache

# --- reference constants ---------------------------------------------------------------
#: libsurvive flags passed on every launch (pika ViveTracker.default_args,
#: HeadViveTracker.args, PikaSenseGripper.connect).
SURVIVE_ARGS = ["--force-calibrate", "1", "--force-ootx", "1", "--lighthouse-update-velocity", "0.5"]
#: Always passed, whatever the calibration mode.
SURVIVE_ARGS_BASE = ["--lighthouse-update-velocity", "0.5"]
#: The two INDEPENDENT halves of a station's cache entry, and which flag clears which
#: (survive.c). ``fresh`` is the reference pair.
#:   --force-ootx 1      -> OOTXSet = 0, memset(fcal)  : the FACTORY calibration + identity
#:   --force-calibrate 1 -> PositionSet = 0, Pose = {} : WHERE the station is in the room
#: Only the second can go stale while the hardware is unchanged, so ``ootx-only`` re-solves
#: the room without re-reading a single radio blob. That matters because poser_mpfit.c
#: attempts NO room solve until EVERY station it has seen has OOTX, so on a 6-station rig one
#: station whose blob never completes blocks the whole room for the entire run.
CALIB_MODES = {
    "fresh": ["--force-calibrate", "1", "--force-ootx", "1"],   # re-read blobs AND re-solve
    "ootx-only": ["--force-calibrate", "1"],                    # keep the factory blobs, re-solve the room
    "reuse": [],                                                # keep both: the room is already solved
}
#: libsurvive's OOTX decoder restarts the whole 33-byte frame on a bad 17th (sync) bit
#: unless this is set; every completed frame is CRC32-checked either way
#: (ootx_decoder.c), so tolerating the bit cannot admit corrupt calibration -- it only
#: stops one dropped sweep from throwing away a frame that was nearly complete. Read from
#: the CONFIG FILE (survive_configi(..., SC_SETCONFIG, ...)), not from argv.
OOTX_IGNORE_SYNC_ERROR = True
#: The pinned libsurvive source commit.
LIBSURVIVE_COMMIT = "5588cadf5ab764b51e50f9b08996009d1b9f836e"
#: the reference recorder's record_session.py blocks 90 s (--vive-warmup-timeout) for every configured puck;
#: a reference gripper driver waits "up to 150 seconds (tracker config resolution can take
#: 50-65s, slower under camera load)".
STARTUP_WAIT_S = 120.0   # the reference recorder's --vive-warmup-timeout is 90 s; 2 stations lock in ~20-25 s there
#: The reference pose validity window (pose_stale_warn_s / absolute_pose_stale_warn_s).
MAX_AGE_S = 1.0
#: umi_state_node's per-role wait logs every 5 s.
WAIT_LOG_EVERY_S = 5.0
#: libsurvive's own "is this object moving?" test, replicated so the table agrees with the
#: gate it reports on (survive_sensor_activations.c: an IMU sample counts as movement when
#: |gyro| > move-threshold-gyro or |accel - accel_ema| > move-threshold-acc, and the
#: running average is 0.98/0.02).
MOVE_THRESHOLD_GYRO = 0.075
MOVE_THRESHOLD_ACC = 0.03
ACCEL_EMA_ALPHA = 0.02
#: poser_mpfit.c feeds an UNPOSITIONED station's light into the solve only while the object
#: has been stationary for longer than its timebase (1 s).
STATIONARY_MIN_S = 1.0
#: how recently a station must have joined the run for the table to call it out.
NEW_STATION_NOTE_S = 30.0

# --- our wire -----------------------------------------------------------------------
WS_HOST = "127.0.0.1"
WS_PORT = 8772  # distinct from the policy bridge's 8770
#: canonical role order (holosoma's vr_3point / vr_5point order); a mode's roles are a
#: PREFIX of this.
ROLES_ALL = ("left", "right", "head", "left_foot", "right_foot")
ROLES_BY_MODE = {
    "1pt": ("left",),
    "2pt": ("left", "right"),
    "3pt": ("left", "right", "head"),
    "5pt": ("left", "right", "head", "left_foot", "right_foot"),
}
#: recorder contract: the per-row `stale` flag window (unchanged).
STALE_S = 0.25


def log(msg: str) -> None:
    print("".join(f"[tracker] {line}\n" for line in str(msg).split("\n")), end="", flush=True)


# --- the source-build env, made automatic -----------------------------------------------------
ENV_REEXEC_MARK = "VIVE_TRACKER_BRIDGE_REEXEC"


def source_build_env(libsurvive_dir: str | None = None) -> dict[str, str] | None:
    """The two variables the reference runbook exports, if a source checkout exists; else None."""
    root = os.path.expanduser(libsurvive_dir or os.environ.get("LIBSURVIVE_DIR", "~/libsurvive"))
    so = os.path.join(root, "bin", "libsurvive.so")
    bindings = os.path.join(root, "bindings", "python")
    if not (os.path.isfile(so) and os.path.isfile(os.path.join(bindings, "pysurvive", "__init__.py"))):
        return None
    env = dict(os.environ)
    ld = env.get("LD_LIBRARY_PATH", "")
    py = env.get("PYTHONPATH", "")
    if os.path.join(root, "bin") not in ld.split(":"):
        env["LD_LIBRARY_PATH"] = os.path.join(root, "bin") + (":" + ld if ld else "")
    if bindings not in py.split(":"):
        env["PYTHONPATH"] = bindings + (":" + py if py else "")
    return env


def ensure_source_build_env() -> None:
    """Re-exec once with LD_LIBRARY_PATH/PYTHONPATH pointing at the source build when they
    do not already — so a bare ``python vive_tracker_bridge.py --list-serials`` (pair_puck.sh,
    a shell) gets the same interpreter environment the launcher gives the bridge. dlopen
    reads LD_LIBRARY_PATH at process start, which is why this must be an exec and not an
    ``os.environ`` edit. No-op when the env is already right or no source build exists
    (the import below then fails with the build recipe)."""
    if os.environ.get(ENV_REEXEC_MARK):
        return
    env = source_build_env()
    if env is None:
        return
    if env.get("LD_LIBRARY_PATH") == os.environ.get("LD_LIBRARY_PATH") \
            and env.get("PYTHONPATH") == os.environ.get("PYTHONPATH"):
        return
    env[ENV_REEXEC_MARK] = "1"
    log(f"re-exec with the source-build env: LD_LIBRARY_PATH+={os.path.dirname(env['LD_LIBRARY_PATH'].split(':')[0])}/bin "
        f"PYTHONPATH+={env['PYTHONPATH'].split(':')[0]}")
    os.execve(sys.executable, [sys.executable] + sys.argv, env)


# --- pysurvive provenance (rule 1) -------------------------------------------
def pysurvive_provenance(mod) -> tuple[str, bool]:
    """Return (package_dir, is_pypi_wheel).

    The manylinux wheel unpacks to ``site-packages/pysurvive/`` with a
    ``site-packages/pysurvive.libs/`` sibling holding its vendored libusb/libudev; the
    source checkout's ``bindings/python/pysurvive/`` has no such sibling and no ``.so``
    of its own (it dlopens ``libsurvive.so`` off ``LD_LIBRARY_PATH``).
    """
    pkg_dir = os.path.dirname(os.path.realpath(mod.__file__))
    wheel = os.path.isdir(os.path.join(os.path.dirname(pkg_dir), "pysurvive.libs"))
    return pkg_dir, wheel


def import_pysurvive(allow_wheel: bool = False):
    try:
        import pysurvive  # noqa: PLC0415 - deliberately late: the env must be set first
    except ImportError as exc:
        raise SystemExit(
            "[tracker] ERROR: pysurvive is not importable. the source-build env (source-built libsurvive):\n"
            "    export LD_LIBRARY_PATH=$HOME/libsurvive/bin:$LD_LIBRARY_PATH\n"
            "    export PYTHONPATH=$HOME/libsurvive/bindings/python:$PYTHONPATH\n"
            f"  (build: git clone https://github.com/cntools/libsurvive ~/libsurvive && "
            f"git -C ~/libsurvive checkout {LIBSURVIVE_COMMIT} && make -C ~/libsurvive)\n"
            f"  import error: {exc}"
        ) from exc
    pkg_dir, wheel = pysurvive_provenance(pysurvive)
    if wheel and not allow_wheel:
        raise SystemExit(
            f"[tracker] ERROR: refusing the PyPI pysurvive wheel at {pkg_dir}\n"
            "  It bundles a 2023 libsurvive (1.2.7) + its own libusb; the rule is a SOURCE build\n"
            f"  at {LIBSURVIVE_COMMIT[:12]} with the system libusb. Put the source bindings FIRST:\n"
            "    export LD_LIBRARY_PATH=$HOME/libsurvive/bin:$LD_LIBRARY_PATH\n"
            "    export PYTHONPATH=$HOME/libsurvive/bindings/python:$PYTHONPATH\n"
            "  (--allow-wheel overrides, for diagnosis only)"
        )
    log(f"pysurvive from {pkg_dir} ({'PyPI WHEEL (allowed by flag)' if wheel else 'source-built bindings'})")
    return pysurvive


# --- quaternion helpers (wxyz, Hamilton) --------------------------------------------
def quat_normalize(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    n = float(np.linalg.norm(q))
    return q / n if n > 0 else np.array([1.0, 0.0, 0.0, 0.0])


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


def quat_from_rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Intrinsic Z(yaw)*Y(pitch)*X(roll) as a wxyz quaternion — the convention
    ``bridges/tracker/axis_check.py`` (``rpy_deg_to_rot``) prints ``WORLD_RPY`` in."""
    qx = np.array([math.cos(roll / 2), math.sin(roll / 2), 0.0, 0.0])
    qy = np.array([math.cos(pitch / 2), 0.0, math.sin(pitch / 2), 0.0])
    qz = np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])
    return quat_normalize(quat_mul(qz, quat_mul(qy, qx)))


def rot_by_quat(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    u = np.array([x, y, z])
    v = np.asarray(v, dtype=float)
    return v + 2.0 * np.cross(u, np.cross(u, v) + w * v)


# --- the reference reader -----------------------------------------------------------------
@dataclass(frozen=True)
class TrackerPose:
    device_name: str      # libsurvive's per-run name (T20, WM0, ...)
    serial: str           # the stable LHR- serial ("" until read)
    timestamp: float      # libsurvive's own clock, Pose()[1]
    received_at: float    # time.time() when the collector saw it (freshness)
    pos: np.ndarray       # (3,) raw libsurvive world
    quat_wxyz: np.ndarray  # (4,) raw, pysurvive's order


# --- startup diagnostics: WHICH puck, or the room? -------------------------------------
# libsurvive's own progress lines (survive_process_gen2.c / ootx_decoder.c / poser.c wording
# at 5588cadf), parsed from its log hook.
RE_OOTX_DONE = re.compile(r"Got OOTX packet (\d+) ([0-9a-fA-F]+)")
RE_OOTX_ATTACH = re.compile(r"OOTX not set for LH in channel (\d+); attaching ootx decoder using device (\S+)")
RE_OOTX_RESTART = re.compile(r"\((\d+)\) OOTX Decoder: Bad sync bit")
#: A station repeats its 33-byte blob back to back, so a SECOND preamble on a channel whose blob has
#: not decoded yet means the previous frame was lost (a missed sweep, or a CRC failure -- libsurvive
#: logs that one only at verbosity 200). Measured: the frame a well-placed puck completes in ~10 s
#: restarted 8 times in 120 s through a puck that saw half the light.
RE_OOTX_PREAMBLE = re.compile(r"\((\d+)\) Preamble found")
#: A pending channel pinned to a puck receiving less than this share of the best-placed puck's
#: light is called out by name: that puck's placement is the blocker (trial 1 of 5: 47%, FAIL@120 s;
#: the four runs whose pins landed on well-lit pucks locked in 18-29 s).
LIGHT_STARVED_SHARE = 0.6
RE_REFERENCE_LH = re.compile(r"Using LH (\d+) as reference lighthouse")
#: survive.c::survive_get_bsd_idx — a station enters the run the first time a puck REPORTS a
#: sync from its channel, and ``activeLighthouses`` only ever GROWS. So the set is whatever the
#: pucks could SEE from where they were lying, not how many stations are powered: measured on
#: this rig, three runs minutes apart saw {1,2,4,5}, {1,2,4,5,8,10} and {1,4,8,10} with all six
#: on. Carrying a puck across the room to give the solve parallax therefore ADDS stations
#: mid-wait, and the all-OOTX gate immediately covers each new one.
RE_ADD_LH = re.compile(r"Adding lighthouse ch (\d+) \(idx: (\d+), cnt: (\d+)\)")
#: poser_mpfit.c: the room solve. "acc err" = |up_z - 1| with up = the station's accelerometer
#: gravity rotated by its SOLVED pose — 0 means the solved station stands exactly as its own
#: accelerometer says; the implied tilt is acos(1 - acc_err). the reference run_umi launcher shows this line.
_NUM = r"([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?|nan|inf|-inf)"
RE_GLOBAL_SOLVE = re.compile(r"Global solve with (\d+) scenes for (\d+) with error of " + _NUM + "/" + _NUM
                             + r" \(acc err\s*" + _NUM + r"\)")
RE_MPFIT_OK = re.compile(r"MPFIT success " + _NUM + "/" + _NUM + "/" + _NUM
                         + r" \((\d+) measurements, (-?\d+), (\w+), (\d+) iters, up err\s*" + _NUM)
RE_MPFIT_FAIL = re.compile(r"MPFIT status failure " + _NUM + "/" + _NUM + "/" + _NUM + r" \((\d+) measurements, (-?\d+), (\w+)\)")

UNASSIGNED = "(unassigned)"


def _cstr(x) -> str:
    """bytes / ctypesgen ``String`` / str -> str."""
    if x is None:
        return ""
    if isinstance(x, bytes):
        return x.decode("utf-8", "replace")
    data = getattr(x, "data", None)
    if isinstance(data, bytes):
        return data.decode("utf-8", "replace")
    try:
        return str(x, "utf-8")
    except TypeError:
        return str(x)


@dataclass
class LightState:
    """What each puck has SEEN and how far each station's OOTX has got (a copy)."""
    available: bool            # False: hooks not installed -> the table shows '?'
    syncs: dict                # puck name -> {channel: light syncs}
    ootx_done: dict            # channel -> base-station id (hex)
    ootx_decoder: dict         # channel -> the puck whose light decodes its OOTX
    ootx_restarts: dict        # channel -> blob frame restarts (2nd preamble before decode, or a "Bad sync bit" reset)
    reference_lh: int | None = None   # the poser's world anchor (LH index), once chosen
    solves: dict = field(default_factory=dict)   # LH index -> {scenes, err, acc_err, t}: last Global solve
    motion: dict = field(default_factory=dict)   # puck name -> seconds since it last MOVED
    added: dict = field(default_factory=dict)    # channel -> wall clock it FIRST entered the run
    mpfit_ok: int = 0                 # room-solve attempts that converged
    mpfit_fail: int = 0               # ... that did not (MP_MAXITER, too few measurements, ...)
    last_mpfit: str = ""              # "success (886 meas, 12 iters, up err 0.0031)" / "failure MP_MAXITER (886 meas)"
    last_mpfit_t: float | None = None

    def channels_of(self, name: str) -> list:
        return sorted(self.syncs.get(name, {}))

    def still_s(self, name: str) -> float | None:
        """Seconds this puck has been stationary by libsurvive's OWN criterion, or None."""
        return self.motion.get(name)

    def is_still(self, name: str, min_s: float = STATIONARY_MIN_S) -> bool | None:
        v = self.still_s(name)
        return None if v is None else v >= min_s

    def total_syncs(self, name: str) -> int:
        return sum(self.syncs.get(name, {}).values())

    def light_share(self, name: str) -> float | None:
        """This puck's light as a fraction of the best-lit puck's (None with nothing counted)."""
        best = max((self.total_syncs(n) for n in self.syncs), default=0)
        return None if best <= 0 else self.total_syncs(name) / best

    def channels_seen(self) -> list:
        chans = set(self.ootx_done) | set(self.ootx_decoder) | set(self.ootx_restarts)
        for per in self.syncs.values():
            chans |= set(per)
        return sorted(chans)

    def ootx_pending(self) -> list:
        return [c for c in self.channels_seen() if c not in self.ootx_done]


def unavailable_light_state() -> LightState:
    return LightState(False, {}, {}, {}, {}, None)


class LightMonitor:
    """Per-puck, per-channel light SYNC counts + OOTX decoder progress, from libsurvive's hooks.

    This is what lets a stalled startup say WHICH puck is the problem — or that none is and
    the room solve is. Why hooks and not ``so->stats.hit_from_lhs``: ``pysurvive_generated.py``
    (2022-01) predates the ``survive.h`` the library is built from (2023-03 -> 5588cadf):
    ``PoserFnData`` was removed from ``SurviveObject`` ahead of ``stats`` and ``accepted_data``
    was added inside it, so the ctypes struct reads the wrong offsets. The hook TYPEDEFS
    (``sync_process_func``, ``log_process_func``) are unchanged over that span, and the one
    struct field read here is ``SurviveObject.codename`` — field #2, same offset in both.

    Cost: the sync hook fires once per station rotation per puck (~50 Hz x pucks x stations,
    ~1.5 k/s on this rig: a few ms of Python per second); the log hook is rare lines. Both
    call libsurvive's default handler afterwards, so nothing it printed before is lost.
    """

    def __init__(self, pysurvive):
        self._ps = pysurvive
        self._lock = threading.Lock()
        self._syncs: dict[str, dict[int, int]] = {}
        self._ootx_done: dict[int, str] = {}
        self._ootx_decoder: dict[int, str] = {}
        self._ootx_restarts: dict[int, int] = {}
        self._ootx_preambles: dict[int, int] = {}
        self._reference_lh: int | None = None
        self._added: dict[int, float] = {}        # channel -> when libsurvive first allocated it a slot
        self._accel_ema: dict[str, list] = {}     # per puck, libsurvive's 0.98/0.02 running average
        self._last_move: dict[str, float] = {}    # per puck, wall clock of the last MOVING sample
        self._solves: dict[int, dict] = {}
        self._mpfit_ok = 0
        self._mpfit_fail = 0
        self._last_mpfit = ""
        self._last_mpfit_t: float | None = None
        self._keep: list = []          # CFUNCTYPE objects must outlive the context
        self._default_sync = None
        self._default_log = None
        self._default_imu = None
        self.attached = False
        self.error = ""

    def attach(self, simple_ctx) -> bool:
        try:
            gen = sys.modules.get("pysurvive.pysurvive_generated")
            if gen is None:
                gen = self._ps.pysurvive_generated
            lib = gen._libs["survive"]
            get_ctx = lib.get("survive_simple_get_ctx", "cdecl")
            get_ctx.argtypes = [ctypes.c_void_p]
            get_ctx.restype = ctypes.c_void_p
            raw = get_ctx(ctypes.cast(simple_ctx.ptr, ctypes.c_void_p))
            if not raw:
                raise RuntimeError("survive_simple_get_ctx returned NULL")
            ctx = ctypes.cast(raw, ctypes.POINTER(gen.SurviveContext))
            self._default_sync = gen.survive_default_sync_process
            self._default_log = gen.survive_default_log_process
            self._default_imu = gen.survive_default_imu_process
            sync_cb = gen.sync_process_func(self._on_sync)
            log_cb = gen.log_process_func(self._on_log)
            imu_cb = gen.imu_process_func(self._on_imu)
            self._keep += [ctx, sync_cb, log_cb, imu_cb]
            gen.survive_install_sync_fn(ctx, sync_cb)
            gen.survive_install_log_fn(ctx, log_cb)
            gen.survive_install_imu_fn(ctx, imu_cb)
        except Exception as exc:  # noqa: BLE001 - diagnostics are optional; the wait still works
            self.error = f"{type(exc).__name__}: {exc}"
            return False
        self.attached = True
        return True

    # -- hooks (libsurvive's thread) --
    def _on_sync(self, so, channel, timeofsync, ootx, gen2):
        try:
            name = _cstr(so.contents.codename) or "?"
            ch = int(channel)
            with self._lock:
                per = self._syncs.setdefault(name, {})
                per[ch] = per.get(ch, 0) + 1
        except Exception:  # noqa: BLE001 - never let a stats read break libsurvive's pipeline
            pass
        self._default_sync(so, channel, timeofsync, ootx, gen2)

    def _on_imu(self, so, mask, accelgyro, timecode, obj_id):
        """Per-puck stationary time. Replicates survive_sensor_activations.c's own test so
        the table's 'still' column is the quantity poser_mpfit.c gates the room solve on;
        the real one lives in ``SurviveObject.activations``, at an offset the 2022 ctypes
        layout gets wrong (see the class docstring)."""
        try:
            name = _cstr(so.contents.codename) or "?"
            acc = [float(accelgyro[i]) for i in range(3)]
            gyro = [float(accelgyro[i]) for i in range(3, 6)]
            now = time.time()
            with self._lock:
                ema = self._accel_ema.get(name)
                if ema is None:
                    self._accel_ema[name] = list(acc)
                    self._last_move[name] = now
                else:
                    moved = (math.sqrt(sum(g * g for g in gyro)) > MOVE_THRESHOLD_GYRO
                             or math.sqrt(sum((a - b) ** 2 for a, b in zip(ema, acc))) > MOVE_THRESHOLD_ACC)
                    for i in range(3):
                        ema[i] = (1.0 - ACCEL_EMA_ALPHA) * ema[i] + ACCEL_EMA_ALPHA * acc[i]
                    if moved:
                        self._last_move[name] = now
        except Exception:  # noqa: BLE001 - never break libsurvive's IMU pipeline over a stat
            pass
        self._default_imu(so, mask, accelgyro, timecode, obj_id)

    def _on_log(self, ctx, level, msg):
        try:
            self.feed_log_line(_cstr(msg))
        except Exception:  # noqa: BLE001
            pass
        self._default_log(ctx, level, msg)

    def feed_log_line(self, text: str, now: float | None = None) -> None:
        """Parse one libsurvive log line (also the unit-test entry point)."""
        m = RE_GLOBAL_SOLVE.search(text)
        if m:
            with self._lock:
                self._solves[int(m.group(2))] = {"scenes": int(m.group(1)), "err": float(m.group(4)),
                                                 "acc_err": float(m.group(5)),
                                                 "t": time.time() if now is None else now}
            return
        m = RE_MPFIT_OK.search(text)
        if m:
            with self._lock:
                self._mpfit_ok += 1
                self._last_mpfit = f"success ({m.group(4)} meas, {m.group(7)} iters, up err {float(m.group(8)):.4f})"
                self._last_mpfit_t = time.time() if now is None else now
            return
        m = RE_MPFIT_FAIL.search(text)
        if m:
            with self._lock:
                self._mpfit_fail += 1
                self._last_mpfit = f"failure {m.group(6)} ({m.group(4)} meas)"
                self._last_mpfit_t = time.time() if now is None else now
            return
        m = RE_ADD_LH.search(text)
        if m:
            with self._lock:
                self._added.setdefault(int(m.group(1)), time.time() if now is None else now)
            return
        m = RE_OOTX_DONE.search(text)
        if m:
            with self._lock:
                self._ootx_done[int(m.group(1))] = m.group(2)
            return
        m = RE_OOTX_ATTACH.search(text)
        if m:
            with self._lock:
                self._ootx_decoder[int(m.group(1))] = m.group(2)
            return
        m = RE_OOTX_RESTART.search(text)
        if m:
            with self._lock:
                ch = int(m.group(1))
                self._ootx_restarts[ch] = self._ootx_restarts.get(ch, 0) + 1
            return
        m = RE_OOTX_PREAMBLE.search(text)
        if m:
            with self._lock:
                ch = int(m.group(1))
                n = self._ootx_preambles.get(ch, 0)
                self._ootx_preambles[ch] = n + 1
                if n and ch not in self._ootx_done:   # a frame started after one already had: the first was lost
                    self._ootx_restarts[ch] = self._ootx_restarts.get(ch, 0) + 1
            return
        m = RE_REFERENCE_LH.search(text)
        if m:
            with self._lock:
                self._reference_lh = int(m.group(1))

    def state(self) -> LightState:
        with self._lock:
            # keyword-only on purpose: this list has grown twice and a positional slip put the
            # motion dict into `solves` (silently: both are dicts keyed by something plausible)
            return LightState(available=self.attached,
                              syncs={n: dict(per) for n, per in self._syncs.items()},
                              ootx_done=dict(self._ootx_done),
                              ootx_decoder=dict(self._ootx_decoder),
                              ootx_restarts=dict(self._ootx_restarts),
                              reference_lh=self._reference_lh,
                              added=dict(self._added),
                              solves={i: dict(v) for i, v in self._solves.items()},
                              motion={n: max(0.0, time.time() - t) for n, t in self._last_move.items()},
                              mpfit_ok=self._mpfit_ok, mpfit_fail=self._mpfit_fail,
                              last_mpfit=self._last_mpfit, last_mpfit_t=self._last_mpfit_t)


def format_solves(light: LightState, lh_labels: dict | None = None, now: float | None = None) -> str:
    """The room solve so far, the reference run_umi launcher terminal in one line:
    ``MPFIT ok 2 / failed 5 (last success (...), 3s ago); acc err LH0(ch10) 0.0310 (~14°, 12 scenes, err 0.0021), ...``
    ("" when libsurvive has not tried yet)."""
    lh_labels = lh_labels or {}
    bits = []
    if light.mpfit_ok or light.mpfit_fail:
        s = f"MPFIT ok {light.mpfit_ok} / failed {light.mpfit_fail}"
        if light.last_mpfit:
            age = "" if (now is None or light.last_mpfit_t is None) else f", {max(0.0, now - light.last_mpfit_t):.0f}s ago"
            s += f" (last {light.last_mpfit}{age})"
        bits.append(s)
    if light.solves:
        parts = []
        for idx in sorted(light.solves):
            sv = light.solves[idx]
            name = f"LH{idx}"
            tilt = math.degrees(math.acos(max(-1.0, min(1.0, 1.0 - sv["acc_err"]))))
            parts.append(f"{lh_labels.get(name, name)} {sv['acc_err']:.4f} (~{tilt:.0f}°, {sv['scenes']} scenes, "
                         f"err {sv['err']:.3g})")
        bits.append("acc err " + ", ".join(parts))
    return "; ".join(bits)


# --- live acc err: the number libsurvive prints as "(acc err X)", recomputed on demand ----------
# poser_mpfit.c:1002-1006 (5588cadf): up = normalize(bsd.accel); err = R_lh2world @ up;
# acc err = |err_z - 1|. libsurvive only PRINTS it from the GlobalSceneSolver path, which a
# 2-station room solved through the per-object MPFIT never takes (measured 2026-09-11: READY in
# 51 s, zero "Global solve" lines) -- so read the two inputs off BaseStationData instead and
# compute the same number. Field offsets verified against the C header with offsetof() the same
# day: Pose 8, BaseStationID 68, fcal 72, accel 192, mode 216 -- identical to the ctypes
# binding's, because the header only APPENDS fields after `tracker`. Both are re-checked at
# runtime against the binding and the read is disabled on any mismatch.
BSD_POSE_OFFSET = 8      # SurvivePose {Pos[3], Rot[4]} doubles; Rot (wxyz) starts at +24
BSD_ACCEL_OFFSET = 192   # LinmathPoint3d: the station's OOTX tilt sensor, its own "up"


def acc_err(rot_wxyz, accel) -> float | None:
    """|(R_lh2world · â)_z − 1| for one solved station (None when accel is all-zero, i.e. no
    OOTX yet). 0 = the solved pose agrees with the station's own tilt sensor; the implied
    tilt is acos(1 − acc_err) (:func:`tilt_deg`)."""
    a = np.asarray(accel, dtype=float)
    n = float(np.linalg.norm(a))
    if n < 1e-9:
        return None
    up = rot_by_quat(quat_normalize(np.asarray(rot_wxyz, dtype=float)), a / n)
    return abs(float(up[2]) - 1.0)


def tilt_deg(acc: float) -> float:
    return math.degrees(math.acos(max(-1.0, min(1.0, 1.0 - acc))))


def format_acc_err(errs: dict, lh_labels: dict | None = None) -> str:
    """``LH0(ch4) 0.0545 (~19°), LH1(ch8) 0.0980 (~26°), LH2 unsolved`` from name -> acc_err|None."""
    lh_labels = lh_labels or {}
    parts = []
    for name in sorted(errs):
        v = errs[name]
        label = lh_labels.get(name, name)
        parts.append(f"{label} unsolved" if v is None else f"{label} {v:.4f} (~{tilt_deg(v):.0f}°)")
    return ", ".join(parts)


@dataclass
class RoleRow:
    role: str
    serial: str
    name: str | None      # libsurvive's WM0/T20 name; None = not on the radio
    pose: str             # "fresh" | "stale" | "none"
    age_s: float | None
    channels: list | None  # channels this puck has synced to; None = unknown
    state: str            # ok | stale | no-radio | no-light | blocked-room | no-pose | unassigned
    note: str
    still_s: float | None = None   # seconds stationary by libsurvive's own test; None = no IMU seen


@dataclass
class Diagnosis:
    rows: list
    positioned: list       # LH names with a solved position
    light: LightState
    shared_blocker: str | None
    puck_problems: list    # roles whose own puck is the problem
    hints: list            # extra VERDICT lines (serial mismatch, unreadable serials)
    lh_labels: dict = field(default_factory=dict)   # "LH2" -> "LH2(ch10)" where the OOTX id is known
    now: float | None = None

    def _roles(self):
        return [r for r in self.rows if r.role != UNASSIGNED]

    def summary(self) -> str:
        roles = self._roles()
        n = len(roles)
        posed = sum(1 for r in roles if r.pose == "fresh")
        radio = sum(1 for r in roles if r.name is not None)
        faults = ", ".join(f"{r.role}({r.state})" for r in roles if r.role in self.puck_problems) or "none"
        shared = "none" if self.shared_blocker is None else self.shared_blocker.split(" — ")[0]
        return (f"puck fault(s) = {faults}; shared blocker = {shared}; "
                f"pose {posed}/{n}, on the radio {radio}/{n}, stations positioned {len(self.positioned)}")

    def lines(self) -> list:
        out = []
        w = max([len(r.role) for r in self.rows] + [4])
        out.append(f"  {'role':<{w}}  {'serial':<13} {'puck':<5} {'radio':<5} {'pose':<9} {'still':<8} "
                   "sees stations (light syncs)")
        for r in self.rows:
            if r.name is None:
                sees = "-"
            elif r.channels is None:
                sees = "?"
            elif not r.channels:
                sees = "NONE"
            else:
                total = sum(self.light.syncs.get(r.name, {}).values())
                sees = " ".join(f"ch{c}" for c in r.channels) + f" ({total} syncs)"
            pose = f"stale {r.age_s:.0f}s" if r.pose == "stale" and r.age_s is not None else r.pose
            if r.name is None:
                still = "-"
            elif r.still_s is None:
                still = "?"
            elif r.still_s >= STATIONARY_MIN_S:
                still = f"{min(r.still_s, 999):.0f}s"
            else:
                still = "MOVING"
            if r.role in self.puck_problems:
                mark = "  <-- " + r.note
            else:
                mark = "  " + r.note if r.note else ""
            out.append(f"  {r.role:<{w}}  {r.serial:<13} {(r.name or '-'):<5} {('yes' if r.name else 'NO'):<5} "
                       f"{pose:<9} {still:<8} {sees}{mark}")
        done = self.light.ootx_done
        pending = self.light.ootx_pending()
        if self.light.available or done or pending:
            def _pend(c):
                via = self.light.ootx_decoder.get(c)
                n = self.light.ootx_restarts.get(c, 0)
                bits = ([f"via {via}"] if via else []) + ([f"{n} restart{'s' if n != 1 else ''}"] if n else [])
                return f"ch{c}" + (f"[{', '.join(bits)}]" if bits else "")
            line = ("  stations: positioned " + (", ".join(self.positioned) if self.positioned else "NONE")
                    + f"; OOTX done {len(done)}/{len(done) + len(pending)} seen"
                    + (" (" + " ".join(f"ch{c}" for c in sorted(done)) + ")" if done else "")
                    + ("; pending " + " ".join(_pend(c) for c in pending) if pending else ""))
            if self.light.added and self.now is not None:
                newest_ch, newest_t = max(self.light.added.items(), key=lambda kv: kv[1])
                age = self.now - newest_t
                if len(self.light.added) > 1 and age <= NEW_STATION_NOTE_S:
                    line += (f". ch{newest_ch} only entered this run {age:.0f}s ago — the set is whatever the "
                             "pucks could SEE, and it only grows; the OOTX gate now covers it too")
            out.append(line)
        solve = format_solves(self.light, self.lh_labels, self.now)
        if solve:
            out.append("  solve: " + solve)
        if self.shared_blocker:
            out.append(f"  VERDICT shared: {self.shared_blocker}")
        for r in self.rows:
            if r.role in self.puck_problems:
                where = r.serial + (f", {r.name}" if r.name else "")
                out.append(f"  VERDICT puck: {r.role} ({where}) {r.note}")
        out.extend(f"  VERDICT: {h}" for h in self.hints)
        if not self.puck_problems and not self.hints and self.shared_blocker:
            waiting = [r.role for r in self._roles() if r.pose != "fresh"]
            out.append(f"  VERDICT: no puck-specific fault — {', '.join(waiting)} are all waiting on the room solve")
        return out


def diagnose(serial_by_role: dict, radio: dict, poses: dict, positioned: list, light: LightState,
             now: float | None = None, max_age_s: float = MAX_AGE_S, include_unassigned: bool = True,
             lh_labels: dict | None = None) -> Diagnosis:
    """Classify every role — THIS puck's fault, or the shared room solve?

    radio: libsurvive name -> serial ("" if unreadable) for every puck it attached, pose or
    not; poses: serial -> latest TrackerPose (or None); positioned: LH names with a pose.
    A puck on the radio under a serial no role names is listed as ``(unassigned)`` — the
    usual shape of a ``--*-serial`` typo or a swapped puck.
    """
    now = time.time() if now is None else now
    name_by_serial = {s: n for n, s in radio.items() if s}
    items = list(serial_by_role.items())
    if include_unassigned:
        assigned = set(serial_by_role.values())
        items += [(UNASSIGNED, s) for n, s in sorted(radio.items()) if s and s not in assigned]
    rows: list = []
    puck_problems: list = []
    room_blocked = False
    for role, serial in items:
        pose = poses.get(serial)
        name = name_by_serial.get(serial) or (pose.device_name if pose is not None else None)
        channels = light.channels_of(name) if (light.available and name) else None
        age = None if pose is None else now - pose.received_at
        still = light.still_s(name) if name else None
        if pose is not None and age <= max_age_s:
            rows.append(RoleRow(role, serial, name, "fresh", age, channels, "ok", "", still))
            continue
        if role == UNASSIGNED:
            rows.append(RoleRow(role, serial, name, "stale" if pose is not None else "none", age, channels,
                                "unassigned", "on the radio, but no role names this serial", still))
            continue
        if pose is not None:
            state = "stale"
            note = f"had a pose, silent for {age:.0f}s: out of every station's view / covered — check THIS puck"
        elif name is None:
            state = "no-radio"
            note = "NOT on the radio: off / asleep (press its button) / not paired to a dongle — check THIS puck"
        elif channels is not None and not channels:
            state = "no-light"
            note = "on the radio but sees NO base station: face-down / covered / pointed away — check THIS puck"
        elif not positioned:
            state = "blocked-room"
            note = "waiting on the room solve (shared, not this puck)"
            room_blocked = True
        else:
            state = "no-pose"
            seen = f"{len(channels)}" if channels else "?"
            note = (f"sees {seen} station(s) but the solver has not placed it yet: it needs >=2 positioned "
                    "stations in view at once — check THIS puck's line of sight")
        if state != "blocked-room":
            puck_problems.append(role)
        rows.append(RoleRow(role, serial, name, "stale" if pose is not None else "none", age, channels, state,
                            note, still))

    hints: list = []
    off_radio = [r for r in rows if r.state == "no-radio"]
    unassigned = [r for r in rows if r.role == UNASSIGNED]
    unreadable = sorted(n for n, s in radio.items() if not s)
    if off_radio and unassigned:
        hints.append(", ".join(r.role for r in off_radio) + " is not on the radio while "
                     + ", ".join(f"{r.name} ({r.serial})" for r in unassigned)
                     + " is and no role names it — a --*-serial mismatch / swapped puck?")
    if off_radio and unreadable:
        hints.append(", ".join(unreadable) + " is on the radio with no readable serial yet — it may be "
                     + ", ".join(r.role for r in off_radio))

    role_by_name = {}
    for r, sn in serial_by_role.items():
        n = name_by_serial.get(sn)
        if n:
            role_by_name[n] = r

    def _who(puck: str) -> str:
        return f"{puck} ({role_by_name[puck]})" if puck in role_by_name else puck

    shared = None
    if room_blocked:
        done = sorted(light.ootx_done)
        pending = light.ootx_pending()
        n_seen = len(done) + len(pending)
        if pending:
            # THE gate: poser_mpfit.c breaks out of its solve setup on the first station without
            # OOTX, so one undecoded blob blocks the room no matter how good every other one is.
            shared = (f"OOTX {len(done)}/{n_seen} stations decoded (pending "
                      + " ".join(f"ch{c}" for c in pending)
                      + ') — libsurvive attempts NO room solve until EVERY station it has seen has its '
                      'blob (poser_mpfit.c: "Wait til this thing gets OOTX"), so those block the whole '
                      "room and no puck can get a pose. A blob is 1 bit per sync sweep, so it needs an "
                      "unbroken line of sight from the puck it decodes through")
            stuck = [(c, light.ootx_decoder[c], light.ootx_restarts.get(c, 0)) for c in pending
                     if c in light.ootx_decoder]
            if stuck:
                shared += (". Each pending channel decodes through ONE puck for the life of this process "
                           "(survive_process_gen2.c pins decoderContext->user on first contact): ")
                shared += "; ".join(
                    f"ch{c} through {_who(via)}" + (f", {n} restart{'s' if n != 1 else ''}" if n else "")
                    for c, via, n in stuck)
                moving = [(c, via) for c, via, _ in stuck if light.is_still(via) is False]
                if moving:
                    shared += ". " + "; ".join(f"{_who(via)} is MOVING and ch{c} decodes only through it"
                                               for c, via in moving)
                starved = []
                for c, via, _ in stuck:
                    share = light.light_share(via)
                    if share is not None and share < LIGHT_STARVED_SHARE:
                        starved.append(f"{_who(via)} gets only {share:.0%} of the light the best-placed puck gets")
                if starved:
                    shared += (". " + "; ".join(starved) + " — turn THAT puck to face the stations (no waving: the "
                               "blob needs a steady, unbroken view from the one puck it decodes through)")
                shared += (". Give those pucks a clear view of those stations; a station you do not need is "
                           "faster powered OFF than decoded")
        else:
            shared = (f"every station has OOTX ({n_seen}/{n_seen}) but none has a solved POSITION yet — the "
                      "solve needs a puck that sees >=2 stations at once (its light enters the solve while it "
                      "is still, so a puck lying on the bench is fine); check line of sight from where the "
                      "pucks are")
        shared += ". This blocks every puck alike and is not a puck fault"
    return Diagnosis(rows, list(positioned), light, shared, puck_problems, hints, dict(lh_labels or {}), now)


class ViveTracker:
    """One shared libsurvive context; the latest pose per tracker, keyed by serial.

    ``HeadViveTracker`` (head_zed_node.py) generalised to N serials, with pika
    ``ViveTracker``'s device bookkeeping and the blocking ``wait_for`` of
    ``PikaSenseGripper.connect`` / ``umi_state_node._wait_for_*_tracker_pose``.
    """

    def __init__(self, pysurvive, args: list[str] | None = None):
        self._ps = pysurvive
        self.args = list(SURVIVE_ARGS) if args is None else list(args)
        self.context = None
        self._running = False
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._latest_by_serial: dict[str, TrackerPose] = {}
        self._latest_by_name: dict[str, TrackerPose] = {}
        self._serial_to_name: dict[str, str] = {}
        self._lighthouses: dict[str, str] = {}   # LH name -> LHB- serial ("" if unread)
        self._updates: dict[str, int] = {}
        self.monitor: LightMonitor | None = None
        self._get_bsd = None      # survive_simple_get_bsd, loaded lazily (None = unavailable)

    # -- lifecycle --
    def connect(self) -> None:
        survive_args = sys.argv[:1] + self.args
        log(f"pysurvive.SimpleContext({survive_args[1:]})")
        self.context = self._ps.SimpleContext(survive_args)
        if not self.context:
            raise RuntimeError(
                "failed to initialize pysurvive SimpleContext — is another libsurvive process "
                "(a previous bridge, survive-cli, pose_check) holding the dongles?"
            )
        if os.environ.get("TRACKER_LIGHT_MONITOR", "1") != "0":   # the escape hatch, never needed so far
            self.monitor = LightMonitor(self._ps)
            if not self.monitor.attach(self.context):
                log(f"per-puck light diagnostics unavailable ({self.monitor.error}); the startup table "
                    "cannot tell 'sees no station' from 'waiting on the room solve'")
        else:
            log("TRACKER_LIGHT_MONITOR=0: per-puck light diagnostics off")
        self._running = True
        self._thread = threading.Thread(target=self._collector, name="vive-tracker-collector", daemon=True)
        self._thread.start()

    def disconnect(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self.context = None
        for name, n in sorted(self._updates.items()):
            log(f"  {name}: {n} updates")

    # -- reads --
    def get_pose(self, serial: str) -> TrackerPose | None:
        with self._lock:
            pose = self._latest_by_serial.get(serial)
            if pose is not None:
                return pose
            name = self._serial_to_name.get(serial)
            if name:
                return self._latest_by_name.get(name)
            return self._latest_by_name.get(serial)   # a bare T20/WM0 name also resolves

    def is_fresh(self, serial: str, max_age_s: float = MAX_AGE_S, now: float | None = None) -> bool:
        pose = self.get_pose(serial)
        if pose is None:
            return False
        return ((time.time() if now is None else now) - pose.received_at) <= max_age_s

    def devices(self) -> dict[str, str]:
        """serial -> libsurvive name, for every tracker that has produced a pose."""
        with self._lock:
            return dict(self._serial_to_name)

    def lighthouses(self) -> list[str]:
        with self._lock:
            return sorted(self._lighthouses)

    def lighthouse_serials(self) -> dict[str, str]:
        with self._lock:
            return dict(self._lighthouses)

    def _walk_objects(self) -> list:
        """(name, serial) for every object libsurvive has attached (pose or not) — a LIVE walk.

        ``SimpleContext.Objects()`` is a snapshot taken in ``__init__``, before the driver
        thread has added a single puck (the "Adding tracked object WM0" lines print AFTER
        it returns), so it is empty for the life of the process; pika's 1 Hz device
        monitor built on it never saw anything either. The underlying list functions are
        live, so walk them; fall back to the snapshot if a binding lacks them.
        """
        ctx = self.context
        if ctx is None:
            return []
        out: list = []
        try:
            cur = self._ps.simple_get_first_object(ctx.ptr)
            while cur:
                out.append((_cstr(self._ps.simple_object_name(cur)), self._serial_of(cur)))
                cur = self._ps.simple_get_next_object(ctx.ptr, cur)
        except Exception:  # noqa: BLE001 - older/newer bindings: use the snapshot
            try:
                out = [(_cstr(o.Name()), self._read_serial(o)) for o in ctx.Objects()]
            except Exception:  # noqa: BLE001
                out = []
        return out

    def radio_devices(self) -> list[str]:
        """Names of every PUCK libsurvive has attached (pose or not)."""
        return sorted(n for n, _ in self._walk_objects() if not n.startswith("LH"))

    def radio_serials(self) -> dict[str, str]:
        """puck name -> LHR- serial ("" if unreadable) for every puck on the radio."""
        return {n: s for n, s in self._walk_objects() if not n.startswith("LH")}

    def _station_positioned(self, sao) -> bool | None:
        """``BaseStationData.PositionSet`` for an LH simple object; None if unreadable.

        Read as bit 0 of byte 0 through ``survive_simple_get_bsd`` — the struct's FIRST field,
        so this does not depend on the stale ctypes layouts (see :class:`LightMonitor`).
        """
        if self._get_bsd is None:
            try:
                gen = sys.modules.get("pysurvive.pysurvive_generated") or self._ps.pysurvive_generated
                fn = gen._libs["survive"].get("survive_simple_get_bsd", "cdecl")
                fn.argtypes = [ctypes.c_void_p]
                fn.restype = ctypes.c_void_p
                self._get_bsd = fn
            except Exception:  # noqa: BLE001 - fall back to the collector's view
                self._get_bsd = False
        if not self._get_bsd:
            return None
        try:
            raw = self._get_bsd(ctypes.cast(sao, ctypes.c_void_p))
            if not raw:
                return None
            return bool(ctypes.cast(raw, ctypes.POINTER(ctypes.c_uint8))[0] & 1)
        except Exception:  # noqa: BLE001
            return None

    def positioned_stations(self) -> list[str]:
        """LH names whose base station has a SOLVED position (``PositionSet``).

        An LH object alone is not proof: besides ``lh_fn`` (a solve), the simple API creates
        one at init for every station in a cached config (``survive_api.c``
        ``create_lighthouse`` for ``i < activeLighthouses``) — a launch that skipped the
        cache ``rm`` lists them unsolved. Where the bit cannot be read, fall back to the
        collector's list: libsurvive marks an LH object updated only when it has a pose.
        """
        ctx = self.context
        if ctx is None:
            return self.lighthouses()
        names: list = []
        readable = True
        try:
            cur = self._ps.simple_get_first_object(ctx.ptr)
            while cur:
                name = _cstr(self._ps.simple_object_name(cur))
                if name.startswith("LH"):
                    bit = self._station_positioned(cur)
                    if bit is None:
                        readable = False
                    elif bit:
                        names.append(name)
                cur = self._ps.simple_get_next_object(ctx.ptr, cur)
        except Exception:  # noqa: BLE001
            readable = False
        return sorted(names) if readable else self.lighthouses()

    def station_acc_errs(self) -> dict:
        """LH name -> live :func:`acc_err` (None = not solved / no OOTX yet); {} when the
        BaseStationData read is unavailable (no ``survive_simple_get_bsd``, or a binding whose
        field offsets differ from the header's -- see BSD_POSE_OFFSET)."""
        ctx = self.context
        if ctx is None:
            return {}
        try:
            gen = sys.modules.get("pysurvive.pysurvive_generated") or self._ps.pysurvive_generated
            bsd_t = gen.struct_BaseStationData
            if bsd_t.Pose.offset != BSD_POSE_OFFSET or bsd_t.accel.offset != BSD_ACCEL_OFFSET:
                return {}
        except Exception:  # noqa: BLE001 - binding without the struct: cannot vouch for offsets
            return {}
        out: dict = {}
        try:
            cur = self._ps.simple_get_first_object(ctx.ptr)
            while cur:
                name = _cstr(self._ps.simple_object_name(cur))
                if name.startswith("LH"):
                    positioned = self._station_positioned(cur)
                    if positioned is None:
                        return {}
                    if not positioned:
                        out[name] = None
                    else:
                        raw = self._get_bsd(ctypes.cast(cur, ctypes.c_void_p))
                        pose = (ctypes.c_double * 7).from_address(raw + BSD_POSE_OFFSET)
                        accel = (ctypes.c_double * 3).from_address(raw + BSD_ACCEL_OFFSET)
                        out[name] = acc_err(list(pose)[3:7], list(accel))
                cur = self._ps.simple_get_next_object(ctx.ptr, cur)
        except Exception:  # noqa: BLE001
            return {}
        return out

    def light_state(self) -> LightState:
        return self.monitor.state() if self.monitor is not None else unavailable_light_state()

    def diagnose(self, serial_by_role: dict[str, str], max_age_s: float = MAX_AGE_S,
                 include_unassigned: bool = True) -> Diagnosis:
        """Per-role verdict (:func:`diagnose`) from the live radio walk, the latest poses,
        the positioned stations and the light monitor."""
        radio = self.radio_serials()
        with self._lock:
            poses = dict(self._latest_by_serial)
        for serial in serial_by_role.values():
            if serial not in poses:
                poses[serial] = self.get_pose(serial)
        light = self.light_state()
        labels = self.lh_labels(light)
        positioned = [labels.get(n, n) for n in self.positioned_stations()]
        return diagnose(serial_by_role, radio, poses, positioned, light,
                        max_age_s=max_age_s, include_unassigned=include_unassigned, lh_labels=labels)

    def lh_labels(self, light: LightState | None = None) -> dict:
        """"LH2" -> "LH2(ch10)": the LH serial is LHB-<BaseStationID hex> and the "Got OOTX packet
        <channel> <id>" line printed the same id, so a station's index maps to its channel."""
        light = self.light_state() if light is None else light
        ch_by_id = {}
        for ch, hexid in light.ootx_done.items():
            try:
                ch_by_id[int(hexid, 16)] = ch
            except ValueError:
                pass
        lh_serials = {n: s for n, s in self._walk_objects() if n.startswith("LH")}
        lh_serials.update({n: s for n, s in self.lighthouse_serials().items() if s})
        labels = {}
        for name, serial in lh_serials.items():
            labels[name] = name
            try:
                ch = ch_by_id.get(int(serial.split("-", 1)[1], 16))
            except (IndexError, ValueError):
                ch = None
            if ch is not None:
                labels[name] = f"{name}(ch{ch})"
        return labels

    def update_counts(self) -> dict[str, int]:
        with self._lock:
            return dict(self._updates)

    # -- the reference startup wait --
    def wait_for(self, serial_by_role: dict[str, str], timeout_s: float,
                 max_age_s: float = MAX_AGE_S, log_every_s: float = WAIT_LOG_EVERY_S) -> float:
        """Block until EVERY serial has a fresh pose; raise naming the ones that never did —
        and, per puck, WHY (:func:`diagnose`): off the radio / sees no station / lost its
        pose, versus the shared room solve nobody can get a pose before."""
        t0 = time.time()
        deadline = t0 + max(0.0, timeout_s)
        last_log = t0   # first table after log_every_s: the driver attaches the pucks 1-2 s in
        while True:
            now = time.time()
            missing = [r for r, s in serial_by_role.items() if not self.is_fresh(s, max_age_s, now)]
            if not missing:
                return now - t0
            if now >= deadline:
                diag = self.diagnose(serial_by_role, max_age_s)
                raise RuntimeError(
                    f"tracker role(s) not reporting after {timeout_s:.0f}s: "
                    + ", ".join(f"{r}={serial_by_role[r]}" for r in missing)
                    + "\n" + "\n".join(diag.lines())
                    + f"\nSUMMARY: {diag.summary()}"
                )
            if now - last_log >= log_every_s:
                diag = self.diagnose(serial_by_role, max_age_s)
                log(f"STATUS {deadline - now:.0f}s left, waiting for {', '.join(missing)} | {diag.summary()}\n"
                    + "\n".join(diag.lines()))
                last_log = now
            time.sleep(0.1)

    # -- the collector (HeadViveTracker._collector) --
    def _serial_of(self, ptr) -> str:
        try:
            return _cstr(self._ps.simple_serial_number(ptr))
        except Exception:  # noqa: BLE001 - the reference reader swallows it too; the name path still works
            return ""

    def _read_serial(self, obj) -> str:
        return self._serial_of(obj.ptr)

    def _collector(self) -> None:
        assert self.context is not None
        while self._running and self.context.Running():
            try:
                updated = self.context.NextUpdated()
                if not updated:
                    time.sleep(0.0005)   # see the module docstring: NextUpdated is non-blocking
                    continue
                name = str(updated.Name(), "utf-8")
                serial = self._read_serial(updated)
                if name.startswith("LH"):
                    with self._lock:
                        if name not in self._lighthouses:
                            log(f"lighthouse {name} serial={serial or '?'}")
                        self._lighthouses[name] = serial
                    continue
                pose_obj = updated.Pose()
                pd = pose_obj[0]
                ts = float(pose_obj[1])
                pos = np.array([pd.Pos[0], pd.Pos[1], pd.Pos[2]], dtype=float)
                quat = np.array([pd.Rot[0], pd.Rot[1], pd.Rot[2], pd.Rot[3]], dtype=float)  # wxyz
                pose = TrackerPose(name, serial, ts, time.time(), pos, quat)
                with self._lock:
                    if name not in self._updates:
                        log(f"tracker {name} serial={serial or '?'} produced its first pose")
                        self._updates[name] = 0
                    self._updates[name] += 1
                    self._latest_by_name[name] = pose
                    if serial:
                        self._serial_to_name[serial] = name
                        self._latest_by_serial[serial] = pose
            except Exception:  # noqa: BLE001 - keep the thread alive, as the reference reader does
                log("collector error:\n" + traceback.format_exc())
                time.sleep(0.1)


# --- our wire ------------------------------------------------------------------------
class TrackerBridge:
    """Role-keyed, world-aligned snapshots of a :class:`ViveTracker` for the WS."""

    def __init__(self, tracker: ViveTracker, mode: str, serial_by_role: dict[str, str],
                 q_world: np.ndarray, stale_s: float = STALE_S):
        self.tracker = tracker
        self.mode = mode
        self.serial_by_role = dict(serial_by_role)
        self.q_world = quat_normalize(q_world)
        self.stale_s = float(stale_s)
        self._seq = {role: 0 for role in serial_by_role}

    def payload(self, pose: TrackerPose, seq: int, now: float) -> dict:
        p = rot_by_quat(self.q_world, pose.pos)
        q = quat_normalize(quat_mul(self.q_world, quat_normalize(pose.quat_wxyz)))
        age = now - pose.received_at
        return {
            "seq": seq,
            "t": now,
            "pos": [float(p[0]), float(p[1]), float(p[2])],
            "quat": [float(q[0]), float(q[1]), float(q[2]), float(q[3])],
            "stale": bool(self.stale_s > 0 and age > self.stale_s),
        }

    def snapshot(self, now: float | None = None) -> dict | None:
        now = time.time() if now is None else now
        trackers = {}
        for role, serial in self.serial_by_role.items():
            pose = self.tracker.get_pose(serial)
            if pose is None:
                continue
            self._seq[role] += 1
            trackers[role] = self.payload(pose, self._seq[role], now)
        if not trackers:
            return None
        names = self.tracker.lighthouses()
        return {"mode": self.mode, "t": now, "trackers": trackers,
                "lighthouses": {"count": len(names), "names": names}}


async def serve(bridge: TrackerBridge, host: str, port: int, rate_hz: float, stop: threading.Event,
                heartbeat_s: float = 10.0) -> None:
    import websockets  # noqa: PLC0415

    clients: set = set()

    async def handler(ws):
        clients.add(ws)
        try:
            async for _ in ws:   # consumers only listen; drain so pings flow
                pass
        except websockets.exceptions.ConnectionClosed:
            pass   # a consumer/probe that left, however it closed — not a server error
        finally:
            clients.discard(ws)

    period = 1.0 / max(1e-3, rate_hz)
    async with websockets.serve(handler, host, port, max_size=None):
        log(f"WebSocket serving at ws://{host}:{port} (broadcast {rate_hz:.0f} Hz)")
        next_hb = time.time() + heartbeat_s
        while not stop.is_set():
            snap = bridge.snapshot()
            if snap is not None and clients:
                websockets.broadcast(clients, json.dumps(snap))
            now = time.time()
            if now >= next_hb:
                next_hb = now + heartbeat_s
                roles = sorted(snap["trackers"]) if snap else []
                log(f"streaming {bridge.mode} roles={roles} lighthouses={len(bridge.tracker.lighthouses())} "
                    f"clients={len(clients)}")
            await asyncio.sleep(period)


def _wait_then_report(tracker: ViveTracker, serial_by_role: dict[str, str], timeout_s: float,
                      max_age_s: float, stop: threading.Event, result: dict,
                      acc_report_s: float = 0.0) -> None:
    """The reference blocking startup wait, on a side thread so the WS can already serve.

    After READY, with ``acc_report_s > 0``, keep printing ``ACC <per-station acc err>`` every
    that many seconds for the life of the process: the launcher's base-station gate tails
    these lines so the operator can watch the solve settle before pressing ENTER, and the
    tracker log keeps a drift record of the station poses across the session."""
    try:
        elapsed = tracker.wait_for(serial_by_role, timeout_s, max_age_s)
    except RuntimeError as exc:
        log(f"FATAL: {exc}")
        result["ok"] = False
        stop.set()
        return
    result["ok"] = True
    light = tracker.light_state()
    solve = format_solves(light, tracker.lh_labels(), time.time())
    # `stations positioned=N seen=M` is what the launcher's MIN_STATIONS floor and STATIONS=
    # comparison read: N = stations with a solved pose, M = every station a puck reported a
    # sync from (falls back to libsurvive's LH objects when the light monitor is off).
    n_pos = len(tracker.positioned_stations())
    n_seen = len(light.channels_seen()) if light.available else len(tracker.lighthouses())
    labels = tracker.lh_labels(light)
    live = tracker.station_acc_errs()
    log("READY all roles streaming after %.1fs: %s | stations positioned=%d seen=%d | lighthouses: %s%s%s" % (
        elapsed, ", ".join(f"{r}={s}" for r, s in serial_by_role.items()), n_pos, max(n_pos, n_seen),
        ", ".join(f"{n}={s or '?'}" for n, s in sorted(tracker.lighthouse_serials().items())) or "none",
        f" | {solve}" if solve else "",
        f" | acc err (live) {format_acc_err(live, labels)}" if live else ""))
    if acc_report_s <= 0:
        return
    if not live:
        log("ACC unavailable: BaseStationData not readable through this binding (no live acc err)")
        return
    while not stop.wait(acc_report_s):
        live = tracker.station_acc_errs()
        if live:
            log("ACC " + format_acc_err(live, tracker.lh_labels()))


# --- --list-serials (detect_vive_trackers_minimal.py) --------------------------------
def usb_28de_devices() -> list[tuple[str, str, str]]:
    """(sysfs name, idProduct, serial) for every Valve/HTC USB device — runbook step 1."""
    out = []
    root = "/sys/bus/usb/devices"
    try:
        entries = sorted(os.listdir(root))
    except OSError:
        return out
    for d in entries:
        try:
            with open(os.path.join(root, d, "idVendor")) as f:
                if f.read().strip() != "28de":
                    continue
            with open(os.path.join(root, d, "idProduct")) as f:
                pid = f.read().strip()
            serial = ""
            try:
                with open(os.path.join(root, d, "serial")) as f:
                    serial = f.read().strip()
            except OSError:
                pass
            out.append((d, pid, serial))
        except OSError:
            continue
    return out


def list_serials(pysurvive, seconds: float, role_by_serial: dict[str, str]) -> int:
    dongles = usb_28de_devices()
    log(f"dongles on USB: {len(dongles)}" + "".join(f"  {d}={s or pid}" for d, pid, s in dongles))
    tracker = ViveTracker(pysurvive)
    tracker.connect()
    deadline = time.time() + max(0.0, seconds)
    while time.time() < deadline and tracker.context.Running():
        time.sleep(0.25)
    devices = tracker.devices()
    trackers = sorted(s for s in devices if s.startswith("LHR-"))
    print("DETECT_SERIALS:", ",".join(trackers) if trackers else "(none)", flush=True)
    for serial, name in sorted(devices.items(), key=lambda kv: kv[1]):
        role = role_by_serial.get(serial)
        log(f"tracker {name} serial={serial} <- {role.upper() if role else 'UNASSIGNED'}")
    for name, serial in sorted(tracker.lighthouse_serials().items()):
        log(f"lighthouse {name} serial={serial or '?'}")
    log("\n".join(tracker.diagnose({r: s for s, r in role_by_serial.items()}).lines()))
    if not trackers:
        log(f"NO puck produced a pose in {seconds:.0f}s — power/wake it facing a base station "
            f"(OOTX takes ~25-35 s per station on a fresh start)")
    tracker.disconnect()
    return 0


# --- CLI -----------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default=WS_HOST, help="WS bind host (localhost only)")
    ap.add_argument("--port", type=int, default=WS_PORT, help="WS bind port")
    ap.add_argument("--rate-hz", type=float, default=90.0, help="WS broadcast rate")
    ap.add_argument("--mode", choices=tuple(ROLES_BY_MODE), default="1pt",
                    help="which roles this rig has (each needs its --<role>-serial)")
    ap.add_argument("--left-serial", default=None, help="LHR- serial of the LEFT (primary) hand tracker")
    ap.add_argument("--right-serial", default=None, help="LHR- serial of the RIGHT hand tracker (2pt+)")
    ap.add_argument("--head-serial", default=None, help="LHR- serial of the CHEST tracker (3pt+)")
    ap.add_argument("--left-foot-serial", default=None, help="LHR- serial of the LEFT FOOT tracker (5pt)")
    ap.add_argument("--right-foot-serial", default=None, help="LHR- serial of the RIGHT FOOT tracker (5pt)")
    ap.add_argument("--yaw-deg", type=float, default=0.0,
                    help="legacy yaw-only world alignment (ignored when --world-rpy-deg is given)")
    ap.add_argument("--world-rpy-deg", type=float, nargs=3, default=None, metavar=("ROLL", "PITCH", "YAW"),
                    help="libsurvive-world -> robot-world rotation, intrinsic Z*Y*X degrees, applied to "
                         "pos AND quat (the rig default is '0 0 180')")
    ap.add_argument("--stale-timeout", type=float, default=STALE_S,
                    help="wire `stale` flag: pose older than this many seconds (recorder contract)")
    ap.add_argument("--startup-wait", type=float, default=STARTUP_WAIT_S,
                    help="block until EVERY role has a fresh pose, then exit 1 if one never does "
                         "(the reference recorder's wait_until_tracking; 120 s). 0 = do not wait / never fail")
    ap.add_argument("--acc-report-s", type=float, default=2.0, metavar="S",
                    help="after READY, print 'ACC <per-station acc err>' every S seconds for the life of "
                         "the process (the launcher's base-station gate tails it; 0 = off)")
    ap.add_argument("--max-age", type=float, default=MAX_AGE_S,
                    help="a pose older than this is not 'fresh' for the startup wait (reference: 1.0 s)")
    ap.add_argument("--survive-arg", action="append", default=[], metavar="ARG",
                    help="extra libsurvive argument(s), appended after the default flags")
    ap.add_argument("--allow-wheel", action="store_true",
                    help="accept the PyPI pysurvive wheel (diagnosis only; refused by default)")
    ap.add_argument("--calib-mode", choices=tuple(CALIB_MODES), default="fresh",
                    help="fresh = the reference flow (re-read every factory blob AND re-solve the room); "
                         "ootx-only = keep the cached factory blobs, re-solve the room; "
                         "reuse = keep both (the room is already solved)")
    ap.add_argument("--ootx-ignore-sync-error", choices=("0", "1"),
                    default="1" if OOTX_IGNORE_SYNC_ERROR else "0",
                    help="tolerate a bad OOTX sync bit instead of restarting the 33-byte frame "
                         "(every frame is CRC32-checked either way)")
    ap.add_argument("--clear-cache", action="store_true",
                    help="rm ~/.config/libsurvive/{config,null}.json first (the launcher does this itself)")
    ap.add_argument("--list-serials", action="store_true",
                    help="open the context, wait --list-seconds, print every LHR-/LHB- serial seen, exit")
    ap.add_argument("--list-seconds", type=float, default=35.0,
                    help="how long --list-serials listens (reference detect script: 35 s)")
    return ap


def resolve_roles(args) -> dict[str, str]:
    """Role -> serial for the mode, refusing a missing serial by flag name."""
    flag_by_role = {"left": "--left-serial", "right": "--right-serial", "head": "--head-serial",
                    "left_foot": "--left-foot-serial", "right_foot": "--right-foot-serial"}
    attr_by_role = {r: f.lstrip("-").replace("-", "_") for r, f in flag_by_role.items()}
    out: dict[str, str] = {}
    for role in ROLES_BY_MODE[args.mode]:
        serial = getattr(args, attr_by_role[role])
        if not serial:
            raise SystemExit(f"[tracker] ERROR: --mode {args.mode} needs {flag_by_role[role]} (role '{role}')")
        out[role] = serial
    dup = {s for s in out.values() if list(out.values()).count(s) > 1}
    if dup:
        raise SystemExit(f"[tracker] ERROR: one serial assigned to several roles: {sorted(dup)}")
    return out


def world_quat(args) -> np.ndarray:
    if args.world_rpy_deg is not None:
        r, p, y = (math.radians(v) for v in args.world_rpy_deg)
        return quat_from_rpy(r, p, y)
    return quat_from_rpy(0.0, 0.0, math.radians(args.yaw_deg))


def clear_cache() -> None:
    for name in ("config.json", "null.json"):
        path = os.path.join(os.path.expanduser("~"), ".config", "libsurvive", name)
        try:
            os.remove(path)
            log(f"removed {path}")
        except FileNotFoundError:
            pass


def prepare_cache(calib_mode: str, ignore_sync_error: bool = OOTX_IGNORE_SYNC_ERROR,
                  path: str | None = None) -> None:
    """Put ``~/.config/libsurvive/config.json`` into the state ``calib_mode`` promises.

    ``fresh`` replaces it with globals only — no cached station at all, so both the factory
    blobs and the positions are re-derived (the reference flow). The file is not simply DELETED because
    ``ootx-ignore-sync-error`` is read from it, not from argv.

    ``ootx-only`` / ``reuse`` repair what is there: drop blocks with no factory calibration,
    drop a duplicated channel, and close the slot-numbering gap that
    ``config_save`` leaves whenever a run decoded some stations but not others. That gap is
    not cosmetic — see :mod:`libsurvive_cache`. ``ootx-only`` additionally clears every
    solved POSITION here rather than relying on ``--force-calibrate``, so what the file says
    is what libsurvive will use.
    """
    path = path or libsurvive_cache.DEFAULT_PATH
    null_json = os.path.join(os.path.dirname(path), "null.json")
    try:
        os.remove(null_json)
    except OSError:
        pass
    if calib_mode == "fresh":
        libsurvive_cache.write_globals_only(path, ignore_sync_error)
        log(f"cache: reset {path} — every station re-reads its factory blob AND is re-solved ('fresh')")
        return
    cfg, notes = libsurvive_cache.repair(path, clear_poses=(calib_mode == "ootx-only"),
                                        ootx_ignore_sync_error=ignore_sync_error)
    if cfg is None:
        libsurvive_cache.write_globals_only(path, ignore_sync_error)
        log(f"cache: none at {path} — starting from scratch, so this run behaves like 'fresh' "
            "(it must decode every station's blob before ANY room solve is attempted)")
        return
    for n in notes:
        log("cache: " + n)
    cfg.save(path)
    if calib_mode == "reuse" and not cfg.positioned():
        log("cache: WARNING no station has a solved POSITION, so --calib-mode reuse has nothing to reuse "
            "and this run still needs a full room solve — use --calib-mode fresh (the default)")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.allow_wheel:
        ensure_source_build_env()
    pysurvive = import_pysurvive(allow_wheel=args.allow_wheel)
    if args.clear_cache:
        clear_cache()
    prepare_cache(args.calib_mode, args.ootx_ignore_sync_error == "1")
    survive_args = CALIB_MODES[args.calib_mode] + list(SURVIVE_ARGS_BASE) + list(args.survive_arg)
    log(f"calibration mode: {args.calib_mode} — libsurvive args {' '.join(survive_args) or '(none)'}")

    if args.list_serials:
        role_by_serial = {}
        for role in ROLES_ALL:
            s = getattr(args, f"{role}_serial")
            if s:
                role_by_serial[s] = role
        return list_serials(pysurvive, args.list_seconds, role_by_serial)

    serial_by_role = resolve_roles(args)
    log(f"mode={args.mode} " + " ".join(f"{r}={s}" for r, s in serial_by_role.items()))
    q_world = world_quat(args)
    if args.world_rpy_deg is not None:
        log("world alignment --world-rpy-deg %s (intrinsic Z*Y*X)" % " ".join(f"{v:g}" for v in args.world_rpy_deg))
    elif args.yaw_deg:
        log(f"world alignment --yaw-deg {args.yaw_deg:g}")

    tracker = ViveTracker(pysurvive, survive_args)
    tracker.connect()
    bridge = TrackerBridge(tracker, args.mode, serial_by_role, q_world, args.stale_timeout)

    stop = threading.Event()
    result: dict = {}
    if args.startup_wait > 0:
        log(f"blocking until every role streams (up to {args.startup_wait:.0f}s; the reference recorder's wait_until_tracking) — "
            "keep every puck powered and in view of the base stations; nothing else to do")
        threading.Thread(target=_wait_then_report, name="vive-startup-wait", daemon=True,
                         args=(tracker, serial_by_role, args.startup_wait, args.max_age, stop, result,
                               args.acc_report_s)).start()
    try:
        asyncio.run(serve(bridge, args.host, args.port, args.rate_hz, stop))
    except KeyboardInterrupt:
        pass
    finally:
        tracker.disconnect()
    return 0 if result.get("ok", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
