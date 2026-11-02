#!/usr/bin/env python3
"""Interactive +X/+Y/+Z axis-direction check for the world-aligned tracker frame.

WHY THIS EXISTS
---------------
Every recorded pose is destructively multiplied by the bridge's world-alignment
rotation ``R_world`` (``vive_tracker_bridge.py``: ``--yaw-deg`` / ``--world-rpy-deg`` /
``--world-quat``). If that rotation is wrong, the dataset's axes do not agree with the
humanoid's: pushing the hand FORWARD may record as -x, or as +y. Nothing downstream can
detect this — the poses are self-consistent, the policy trains happily, and the error
only shows up as a robot that reaches the wrong way at deploy time. (The known instance:
a reversed L/R serial mapping showed up as a ~180 degrees yaw flip.)

The only pre-existing tool is ``vive_tracker_bridge.py --measure-frame``, which is
ROTATION-only, needs exclusive ownership of the pysurvive context (so it cannot run
while the rig is up), and requires the operator to eyeball which libsurvive axis a
physical rotation maps to. There was no translation test anywhere in the repo — i.e.
nothing checked the thing that actually matters for "push forward == +x".

WHAT IT DOES
------------
Subscribes to the LIVE tracker bridge on ws://127.0.0.1:8772 (read-only, coexists with
the running rig) and walks the operator through three prompted motions per role:

    move the tracker ~+30 cm along the humanoid's +X (forward)  -> expect +x dominant
    ...                                     +Y (robot's LEFT)   -> expect +y dominant
    ...                                     +Z (up)             -> expect +z dominant

For each, it captures a reference pose, waits for a displacement above
``--min-travel``, and reports the dominant axis and sign of the observed delta. A PASS
requires the dominant axis to be the expected one AND the sign to be positive AND the
off-axis leakage to stay under ``--max-cross`` of the primary component.

On failure it SOLVES for the corrective rotation from the three measured directions
(orthonormalized via SVD — the nearest rotation matrix to the observed axis triple),
COMPOSES it onto the alignment the bridge is currently running, and prints the exact
``WORLD_RPY`` / ``--world-quat`` to relaunch with. That closes the loop the old tool left
open ("paste the three axis lines back and I'll solve it").

Pass ``--current-rpy-deg`` / ``--current-quat`` whenever the rig is NOT on an identity
alignment (the launcher does this automatically from WORLD_RPY / YAW_DEG). The solved
delta alone is a valid relaunch value only from identity, and the operator cannot compose
it by hand: adding Euler triples is wrong in general.

Convention reminder: this rig's world frame is Z-up with +x FORWARD, matching the
humanoid's pelvis frame. "+Y is the robot's LEFT" follows from a right-handed frame.

Exit codes (mirrors gripper_selftest.py / ft_selfcheck.py so the launcher can gate):
    0  PASS  — every checked role's axes are correct
    1  FAIL  — an axis is mis-mapped (a corrective WORLD_RPY is printed), or timeout
    2  could not connect to the tracker bridge

Run:
    .venv/bin/python bridges/tracker/axis_check.py                 # left only
    .venv/bin/python bridges/tracker/axis_check.py --roles left right head
    .venv/bin/python bridges/tracker/axis_check.py --roles left right head left_foot right_foot
    .venv/bin/python bridges/tracker/axis_check.py --json out.json # machine-readable
    .venv/bin/python bridges/tracker/axis_check.py --current-rpy-deg 0 90 0
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys

import numpy as np

WS_URL_DEFAULT = "ws://127.0.0.1:8772"

#: Max frames to discard per drain pass (see TrackerStream.run). ~90 Hz for 5 s of prompt
#: dwell is ~450; 4096 covers a long dwell while bounding the loop against a publisher
#: that outruns us.
_DRAIN_MAX = 4096

#: The three checks, in order. ``vec`` is the expected unit direction of the measured
#: displacement in the world-aligned frame.
AXES = (
    ("+X", np.array([1.0, 0.0, 0.0]), "FORWARD, away from you (the humanoid's +x)"),
    ("+Y", np.array([0.0, 1.0, 0.0]), "to the robot's LEFT (your right, facing it)"),
    ("+Z", np.array([0.0, 0.0, 1.0]), "straight UP"),
)


def _fmt_vec(v: np.ndarray) -> str:
    return "[" + " ".join(f"{x:+.3f}" for x in v) + "]"


def dominant_axis(delta: np.ndarray) -> tuple[int, float]:
    """Index of the largest-|.| component of ``delta`` and its signed value."""
    i = int(np.argmax(np.abs(delta)))
    return i, float(delta[i])


def cross_leakage(delta: np.ndarray) -> float:
    """Ratio of the largest OFF-dominant component to the dominant one.

    A clean straight push has a small ratio; 0.5 means the motion was ~27 degrees off
    the axis, which usually means the operator moved diagonally rather than that the
    frame is wrong — hence a separate, looser threshold from the sign/axis test.
    """
    i, val = dominant_axis(delta)
    others = np.delete(np.abs(delta), i)
    return float(others.max() / max(abs(val), 1e-9))


def solve_correction(measured: dict[str, np.ndarray]) -> np.ndarray | None:
    """Nearest rotation matrix mapping the MEASURED directions onto the intended axes.

    ``measured[name]`` is the observed unit displacement when the operator moved along
    the intended axis ``name``. We want ``C`` with ``C @ measured[k] ~= intended[k]``
    for all k; the least-squares solution over a full triple is the orthogonal Procrustes
    problem, whose answer is ``C = U V^T`` from the SVD of ``intended @ measured^T``.
    Requires all three axes; returns None otherwise (two axes underdetermine the roll
    about the third, and guessing it silently would be worse than declining).

    ``C`` LEFT-multiplies: it composes onto the existing ``R_world``, so the corrected
    launch value is ``C @ R_world_current``. The caller prints the composed result.
    """
    if not all(name in measured for name, _v, _h in AXES):
        return None
    intended = np.stack([v for _n, v, _h in AXES], axis=1)          # (3, 3) columns
    observed = np.stack([measured[n] for n, _v, _h in AXES], axis=1)
    u, _s, vt = np.linalg.svd(intended @ observed.T)
    c = u @ vt
    if np.linalg.det(c) < 0:                 # reflection -> force a proper rotation
        u[:, -1] *= -1.0
        c = u @ vt
    return c


def rot_to_rpy_deg(r: np.ndarray) -> tuple[float, float, float]:
    """Intrinsic Z(yaw)*Y(pitch)*X(roll) Euler angles in degrees.

    Matches ``vive_tracker_bridge.quat_from_rpy``, so the printed triple can be passed
    straight back as ``WORLD_RPY`` / ``--world-rpy-deg``.
    """
    pitch = -math.asin(max(-1.0, min(1.0, r[2, 0])))
    if abs(math.cos(pitch)) < 1e-6:          # gimbal lock: fold roll into yaw
        roll = 0.0
        yaw = math.atan2(-r[0, 1], r[1, 1])
    else:
        roll = math.atan2(r[2, 1], r[2, 2])
        yaw = math.atan2(r[1, 0], r[0, 0])
    return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)


def rpy_deg_to_rot(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Intrinsic Z(yaw)*Y(pitch)*X(roll) rotation matrix from degrees.

    Exact inverse of :func:`rot_to_rpy_deg`, and the same convention the bridge's
    ``quat_from_rpy`` uses — so a WORLD_RPY passed in here means what the bridge applied.
    """
    r, p, y = (math.radians(v) for v in (roll, pitch, yaw))
    rx = np.array([[1, 0, 0], [0, math.cos(r), -math.sin(r)], [0, math.sin(r), math.cos(r)]])
    ry = np.array([[math.cos(p), 0, math.sin(p)], [0, 1, 0], [-math.sin(p), 0, math.cos(p)]])
    rz = np.array([[math.cos(y), -math.sin(y), 0], [math.sin(y), math.cos(y), 0], [0, 0, 1]])
    return rz @ ry @ rx


def quat_wxyz_to_rot(q: np.ndarray) -> np.ndarray:
    """wxyz quaternion -> rotation matrix (the bridge's ``--world-quat`` format)."""
    q = np.asarray(q, dtype=float)
    n = float(np.linalg.norm(q))
    w, x, y, z = q / (n if n > 1e-12 else 1.0)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def rot_to_quat_wxyz(r: np.ndarray) -> np.ndarray:
    """Rotation matrix -> wxyz quaternion (the bridge's ``--world-quat`` format)."""
    tr = float(np.trace(r))
    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0
        w = 0.25 * s
        x = (r[2, 1] - r[1, 2]) / s
        y = (r[0, 2] - r[2, 0]) / s
        z = (r[1, 0] - r[0, 1]) / s
    else:
        i = int(np.argmax(np.diag(r)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = math.sqrt(max(1e-12, 1.0 + r[i, i] - r[j, j] - r[k, k])) * 2.0
        q = [0.0, 0.0, 0.0]
        q[i] = 0.25 * s
        q[j] = (r[j, i] + r[i, j]) / s
        q[k] = (r[k, i] + r[i, k]) / s
        w = (r[k, j] - r[j, k]) / s
        x, y, z = q
    v = np.array([w, x, y, z], dtype=float)
    n = float(np.linalg.norm(v))
    return v / (n if n > 1e-12 else 1.0)


class TrackerStream:
    """Read-only latest-pose register over the bridge's role-keyed WS payload."""

    def __init__(self, url: str) -> None:
        self.url = url
        self._latest: dict[str, tuple[np.ndarray, bool]] = {}
        self._mode: str | None = None

    async def run(self, stop: asyncio.Event) -> None:
        import websockets

        async with websockets.connect(self.url, open_timeout=5.0, max_size=None) as ws:
            while not stop.is_set():
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=2.0)
                except asyncio.TimeoutError:
                    continue
                # DRAIN TO LATEST. The bridge broadcasts ~90 Hz; if this task was starved
                # (a blocking prompt, a slow disk) the socket holds a backlog of old poses.
                # Replaying it would report motion the operator made WHILE READING the
                # prompt, so keep only the newest frame — this register is a latest-value
                # snapshot, not a history. Bounded so a faster-than-us publisher cannot
                # spin here forever.
                for _ in range(_DRAIN_MAX):
                    try:
                        msg = await asyncio.wait_for(ws.recv(), timeout=0.002)
                    except asyncio.TimeoutError:
                        break            # caught up: `msg` is the newest frame
                d = json.loads(msg)
                self._mode = d.get("mode", self._mode)
                # All modes emit the role-keyed superset; the flat legacy payload is
                # never emitted by the current bridge but is tolerated for old logs.
                trackers = d.get("trackers")
                if trackers is None and "pos" in d:
                    trackers = {"left": d}
                for role, t in (trackers or {}).items():
                    self._latest[role] = (
                        np.asarray(t["pos"], dtype=float), bool(t.get("stale", False)))

    def pose(self, role: str) -> np.ndarray | None:
        """Latest NON-STALE position for ``role``, or None."""
        got = self._latest.get(role)
        if got is None:
            return None
        pos, stale = got
        return None if stale else pos

    @property
    def mode(self) -> str | None:
        return self._mode

    @property
    def roles_seen(self) -> tuple[str, ...]:
        return tuple(self._latest)


async def _await_role(stream: TrackerStream, role: str, timeout: float) -> bool:
    for _ in range(int(timeout * 10)):
        if stream.pose(role) is not None:
            return True
        await asyncio.sleep(0.1)
    return False


async def _measure_axis(stream: TrackerStream, role: str, label: str, hint: str,
                        min_travel: float, timeout: float) -> np.ndarray | None:
    """Prompt, capture a reference, and return the unit displacement of the motion.

    The reference is the pose at the moment the operator presses Enter; the motion ends
    when the displacement first exceeds ``min_travel``, then settles for ``--settle`` so
    the reported vector is the END of the push rather than a mid-stroke sample.

    ⚠️ The prompt MUST be read on a thread (``run_in_executor``), never with a bare
    ``input()``. A bare call blocks the whole asyncio event loop, so ``TrackerStream.run``
    — the only WS consumer — stops draining while the operator reads the prompt. The
    socket then buffers seconds of ~90 Hz poses, the reference is captured from a stale
    register, and the poll loop replays that BACKLOG as if it were the push: the tool
    reports motion the operator made while reading, and can report a 40 cm move for a
    tracker that never moved. That is not just a false FAIL — it can produce a false PASS
    on a genuinely mis-aligned rig, i.e. the exact opposite of this tool's purpose.
    """
    loop = asyncio.get_event_loop()
    prompt = (f"\n  [{role} {label}] hold the tracker STILL, then press Enter and move it "
              f"~{min_travel * 100:.0f} cm {hint}: ")
    try:
        await loop.run_in_executor(None, input, prompt)
    except EOFError:
        # Non-interactive stdin (piped / cron). The launcher documents this check as
        # interactive, so refuse rather than "measure" a stream nobody is moving.
        print(f"\n  [{role} {label}] stdin is not interactive — this check needs an "
              f"operator to move the tracker. Run it from a terminal, or set AXISCHECK=0.",
              file=sys.stderr)
        return None
    # Let the reader drain anything that queued during the prompt, so the reference is
    # the pose NOW and not a stale register value.
    await asyncio.sleep(0.25)
    ref = stream.pose(role)
    if ref is None:
        print(f"  [{role} {label}] tracker went stale at the reference — retry.", file=sys.stderr)
        return None
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        await asyncio.sleep(0.05)
        cur = stream.pose(role)
        if cur is None:
            continue
        if float(np.linalg.norm(cur - ref)) >= min_travel:
            await asyncio.sleep(0.3)                 # let the operator stop moving
            end = stream.pose(role)
            delta = (end if end is not None else cur) - ref
            n = float(np.linalg.norm(delta))
            print(f"  [{role} {label}] moved {n * 100:5.1f} cm  delta={_fmt_vec(delta)}")
            return delta / n
    print(f"  [{role} {label}] TIMEOUT: never travelled {min_travel * 100:.0f} cm.",
          file=sys.stderr)
    return None


async def _check_role(stream: TrackerStream, role: str, args) -> dict:
    print(f"\n=== role: {role} ===")
    result: dict = {"role": role, "axes": {}, "pass": True}
    measured: dict[str, np.ndarray] = {}
    for label, expect, hint in AXES:
        unit = await _measure_axis(stream, role, label, hint, args.min_travel, args.timeout)
        if unit is None:
            result["pass"] = False
            result["axes"][label] = {"ok": False, "reason": "no motion / stale"}
            continue
        i, val = dominant_axis(unit)
        want_i = int(np.argmax(expect))
        leak = cross_leakage(unit)
        # THREE outcomes, not two. A wrong axis or an inverted sign means the FRAME is
        # wrong. Excessive off-axis leakage usually means the OPERATOR pushed diagonally —
        # so it must not feed solve_correction, which would fit a rotation to a sloppy
        # triple and print a confident, wrong WORLD_RPY (reproduced: a perfect rig with
        # every push 28 deg off-axis printed (-15.9, -20.96, -15.9) and aborted the launch).
        frame_ok = (i == want_i) and (val > 0.0)
        sloppy = leak > args.max_cross
        ok = frame_ok and not sloppy
        result["axes"][label] = {
            "ok": bool(ok), "unit": [round(float(x), 4) for x in unit],
            "dominant": "xyz"[i], "sign": "+" if val > 0 else "-",
            "cross_leakage": round(leak, 3),
            "frame_ok": bool(frame_ok), "sloppy": bool(sloppy),
        }
        if ok:
            print(f"  [{role} {label}] OK   dominant={'xyz'[i]}{'+' if val > 0 else '-'} "
                  f"cross={leak:.2f}")
            measured[label] = unit
        elif frame_ok:
            # Axis and sign are RIGHT; the push was just diagonal. Do not treat this as a
            # frame error and do not let it into the correction fit.
            result["pass"] = False
            result["needs_redo"] = True
            print(f"  [{role} {label}] REDO motion {leak:.2f} off-axis "
                  f"(>{args.max_cross}) — the axis and sign are CORRECT, so this is a "
                  f"sloppy push, not a frame error. Push straighter and re-run.",
                  file=sys.stderr)
        else:
            result["pass"] = False
            measured[label] = unit
            why = "wrong axis" if i != want_i else "INVERTED sign"
            print(f"  [{role} {label}] FAIL {why}: measured {'xyz'[i]}"
                  f"{'+' if val > 0 else '-'}, expected {label}", file=sys.stderr)

    any_frame_error = any(not a.get("frame_ok", True) for a in result["axes"].values())
    if not result["pass"] and not any_frame_error:
        print(f"\n  [{role}] every axis is correct in DIRECTION and SIGN — no frame "
              f"correction is needed. Re-run and push straighter along each axis.",
              file=sys.stderr)
    elif not result["pass"]:
        c = solve_correction(measured)
        if c is not None:
            # The solved C maps MEASURED -> INTENDED, so the value to relaunch with is
            # C @ R_world_current, not C. Printing bare C is only right when the current
            # alignment is identity; otherwise following the launcher's "relaunch with the
            # WORLD_RPY the check printed" leaves the rig broken (measured residual 1.6-2.0
            # for non-identity starts). And the composition CANNOT be done by the operator
            # from Euler triples — adding them is wrong in general (R_cur=(0,0,90) with
            # C=(40,0,0) gives a true (0,-40,90), not (40,0,90)) — so the tool must be
            # told the current alignment and do it.
            r_cur = args.current_rot                    # identity when not supplied
            composed = c @ r_cur
            roll, pitch, yaw = rot_to_rpy_deg(composed)
            quat = rot_to_quat_wxyz(composed)
            result["correction_rpy_deg"] = [round(roll, 2), round(pitch, 2), round(yaw, 2)]
            result["correction_quat_wxyz"] = [round(float(x), 6) for x in quat]
            result["delta_rpy_deg"] = [round(v, 2) for v in rot_to_rpy_deg(c)]
            result["assumed_current_rpy_deg"] = [round(v, 2) for v in rot_to_rpy_deg(r_cur)]
            print(f"\n  [{role}] CORRECTION — RELAUNCH with this (it already composes the "
                  f"fix onto the alignment you launched with):")
            print(f"     WORLD_RPY=\"{roll:.2f} {pitch:.2f} {yaw:.2f}\"")
            print(f"     or --world-quat {quat[0]:.6f} {quat[1]:.6f} {quat[2]:.6f} "
                  f"{quat[3]:.6f}")
            if not args.assumed_identity:
                print(f"     (composed onto the --current-rpy-deg you passed)")
            else:
                print(f"     NOTE: assuming the rig currently runs an IDENTITY alignment. "
                      f"If you launched with WORLD_RPY/YAW_DEG set, re-run this check with "
                      f"--current-rpy-deg <R P Y> (or --current-quat) so the printed value "
                      f"is composed correctly — replacing a non-identity alignment with a "
                      f"bare delta leaves the rig wrong.")
            print(f"     Relaunch, then re-run this check to confirm.")
        else:
            print(f"\n  [{role}] cannot solve a correction: all three axes must be "
                  f"measured (two underdetermine the roll about the third).",
                  file=sys.stderr)
    return result


async def _main(args) -> int:
    stream = TrackerStream(args.url)
    stop = asyncio.Event()
    task = asyncio.create_task(stream.run(stop))
    # Fail fast and distinctly on "bridge not up" (rc=2) vs "up but wrong" (rc=1).
    await asyncio.sleep(0.5)
    if task.done() and task.exception() is not None:
        print(f"[axis_check] ERROR: cannot connect to {args.url} — is the tracker "
              f"bridge running? ({task.exception()})", file=sys.stderr)
        return 2

    print(f"[axis_check] connected to {args.url}; world frame is Z-up with +x FORWARD.")
    print("[axis_check] This is READ-ONLY — it never touches the tracker or the robot.")

    results = []
    for role in args.roles:
        if not await _await_role(stream, role, timeout=8.0):
            print(f"[axis_check] ERROR: no live (non-stale) pose for role {role!r}. "
                  f"Seen: {stream.roles_seen or '(none)'} (bridge mode={stream.mode}). "
                  f"Is that tracker powered and in view of the lighthouses?",
                  file=sys.stderr)
            stop.set()
            return 2
        results.append(await _check_role(stream, role, args))

    stop.set()
    task.cancel()

    ok = all(r["pass"] for r in results)
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"pass": ok, "url": args.url, "mode": stream.mode,
                       "min_travel_m": args.min_travel, "max_cross": args.max_cross,
                       "roles": results}, f, indent=2)
        print(f"[axis_check] wrote {args.json}")
    print(f"\n[axis_check] RESULT {'PASS' if ok else 'FAIL'} "
          f"({sum(r['pass'] for r in results)}/{len(results)} roles correct)")
    if not ok:
        print("[axis_check] Do NOT record training data until this passes: every "
              "recorded pose is multiplied by the world alignment, and a wrong axis is "
              "undetectable downstream.", file=sys.stderr)
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", default=WS_URL_DEFAULT, help=f"tracker bridge WS (default {WS_URL_DEFAULT})")
    p.add_argument("--roles", nargs="+", default=["left"],
                   choices=["left", "right", "head", "left_foot", "right_foot"],
                   help="roles to check (default: left, the primary hand). The 5pt FEET "
                        "are checkable like any other role — the check reads the wire, so "
                        "it needs no per-role geometry — but a foot bolted to an ankle is "
                        "awkward to push a clean 15 cm along one axis; hold the puck in "
                        "your hand for the check, then mount it.")
    p.add_argument("--min-travel", type=float, default=0.15,
                   help="metres of displacement that count as a deliberate move (default 0.15)")
    p.add_argument("--max-cross", type=float, default=0.5,
                   help="max off-axis/on-axis ratio before the push is called sloppy "
                        "(default 0.5, i.e. ~27 deg)")
    p.add_argument("--timeout", type=float, default=30.0,
                   help="seconds to wait for each motion (default 30)")
    p.add_argument("--json", default="", help="also write a machine-readable report here")
    p.add_argument("--current-rpy-deg", type=float, nargs=3, default=None,
                   metavar=("ROLL", "PITCH", "YAW"),
                   help="the world alignment the TRACKER BRIDGE is currently running "
                        "(WORLD_RPY / YAW_DEG). Required for a correct relaunch value "
                        "whenever it is non-identity: the printed correction is composed "
                        "onto it, and Euler triples cannot be composed by hand.")
    p.add_argument("--current-quat", type=float, nargs=4, default=None,
                   metavar=("W", "X", "Y", "Z"),
                   help="same as --current-rpy-deg, as the bridge's wxyz --world-quat. "
                        "Wins over the rpy form.")
    args = p.parse_args(argv if argv is not None else sys.argv[1:])

    # Resolve the current alignment to a rotation matrix once, so _check_role can compose
    # without re-parsing. assumed_identity drives the "you may need --current-rpy-deg"
    # warning, which exists because silently assuming identity is how a non-identity rig
    # gets a bare delta printed at it.
    if args.current_quat:
        args.current_rot = quat_wxyz_to_rot(np.asarray(args.current_quat, dtype=float))
        args.assumed_identity = False
    elif args.current_rpy_deg:
        args.current_rot = rpy_deg_to_rot(*args.current_rpy_deg)
        args.assumed_identity = False
    else:
        args.current_rot = np.eye(3)
        args.assumed_identity = True
    try:
        return asyncio.run(_main(args))
    except KeyboardInterrupt:
        print("\n[axis_check] interrupted.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
