#!/usr/bin/env python
"""Logic tests for the axis-direction check's pure math (no hardware, no WS).

The load-bearing claim is that when the check FAILS it prints a rotation that actually
fixes the frame. These tests inject known-wrong world alignments, run the measured
directions a tracker would report under them, and assert the solved correction composes
back to identity.

Run:
    .venv/bin/python bridges/tracker/test_axis_check.py
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import axis_check as ac  # noqa: E402


def _rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Intrinsic Z(yaw)*Y(pitch)*X(roll), degrees — matches the bridge's convention."""
    r, p, y = (math.radians(v) for v in (roll, pitch, yaw))
    rx = np.array([[1, 0, 0], [0, math.cos(r), -math.sin(r)], [0, math.sin(r), math.cos(r)]])
    ry = np.array([[math.cos(p), 0, math.sin(p)], [0, 1, 0], [-math.sin(p), 0, math.cos(p)]])
    rz = np.array([[math.cos(y), -math.sin(y), 0], [math.sin(y), math.cos(y), 0], [0, 0, 1]])
    return rz @ ry @ rx


def test_dominant_axis_and_sign():
    assert ac.dominant_axis(np.array([0.9, 0.1, 0.0])) == (0, 0.9)
    i, v = ac.dominant_axis(np.array([-0.8, 0.2, 0.1]))
    assert (i, round(v, 6)) == (0, -0.8)          # sign is preserved, not abs'd
    assert ac.dominant_axis(np.array([0.0, 0.0, -1.0])) == (2, -1.0)


def test_cross_leakage_measures_off_axis_fraction():
    assert ac.cross_leakage(np.array([1.0, 0.0, 0.0])) == 0.0
    assert abs(ac.cross_leakage(np.array([1.0, 0.5, 0.0])) - 0.5) < 1e-9
    # A 45-degree diagonal push leaks 100%: ambiguous, must not pass.
    d = np.array([1.0, 1.0, 0.0]) / math.sqrt(2)
    assert abs(ac.cross_leakage(d) - 1.0) < 1e-9


def test_rpy_roundtrip_matches_the_bridge_convention():
    """The printed WORLD_RPY must reproduce the rotation it describes, in the same
    convention vive_tracker_bridge.quat_from_rpy uses (else the operator's relaunch
    applies a different rotation than the one solved)."""
    for rpy in [(0, 0, 0), (0, 0, 90), (0, 90, 0), (90, 0, 0), (10, -20, 30), (0, 0, 180)]:
        r = _rpy(*rpy)
        back = ac.rot_to_rpy_deg(r)
        assert np.allclose(_rpy(*back), r, atol=1e-9), (rpy, back)


def test_quat_roundtrip_for_all_trace_branches():
    """rot_to_quat_wxyz has a positive-trace branch and three diagonal-max branches;
    exercise all of them (180-degree rotations are the negative-trace cases)."""
    for rpy in [(0, 0, 0), (0, 0, 90), (180, 0, 0), (0, 180, 0), (0, 0, 180),
                (37, -14, 122)]:
        r = _rpy(*rpy)
        q = ac.rot_to_quat_wxyz(r)
        assert abs(np.linalg.norm(q) - 1.0) < 1e-9, rpy
        # Rebuild the matrix from the quaternion and compare.
        w, x, y, z = q
        rebuilt = np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
        assert np.allclose(rebuilt, r, atol=1e-8), rpy


def _measured_under(bad: np.ndarray) -> dict:
    """What the check would MEASURE if the world alignment were wrong by ``bad``.

    The operator moves along the true axes; the bridge reports those displacements
    rotated by the erroneous alignment.
    """
    return {name: bad @ v for name, v, _h in ac.AXES}


def test_correction_recovers_a_90_degree_yaw_error():
    """The classic failure: pushing FORWARD records as +y (or -y). The solved
    correction must map the measurements back onto the intended axes."""
    bad = _rpy(0, 0, 90)
    measured = _measured_under(bad)
    # Sanity: the wrong alignment really does mis-map +X.
    i, val = ac.dominant_axis(measured["+X"])
    assert i == 1 and val > 0, measured["+X"]
    c = ac.solve_correction(measured)
    assert c is not None
    assert np.allclose(c @ bad, np.eye(3), atol=1e-9)
    for name, v, _h in ac.AXES:
        assert np.allclose(c @ measured[name], v, atol=1e-9), name


def test_correction_recovers_the_180_degree_flip_and_a_tilt():
    """The two historically-observed shapes: a ~180 degrees yaw (reversed L/R serials)
    and a world TILT, which a pure yaw structurally cannot fix."""
    for rpy in [(0, 0, 180), (0, 90, 0), (0, 90, 45), (12, -33, 200)]:
        bad = _rpy(*rpy)
        c = ac.solve_correction(_measured_under(bad))
        assert c is not None, rpy
        assert np.allclose(c @ bad, np.eye(3), atol=1e-8), rpy
        assert abs(np.linalg.det(c) - 1.0) < 1e-9, f"{rpy}: not a proper rotation"


def test_correction_is_a_proper_rotation_even_from_noisy_pushes():
    """Hand motions are not perfectly axis-aligned. The SVD must still return a proper
    rotation (det=+1, orthonormal), not a reflection or a shear."""
    rng = np.random.default_rng(0)
    bad = _rpy(0, 0, 90)
    measured = {}
    for name, v, _h in ac.AXES:
        d = bad @ v + rng.normal(scale=0.12, size=3)     # sloppy ~7 degrees pushes
        measured[name] = d / np.linalg.norm(d)
    c = ac.solve_correction(measured)
    assert c is not None
    assert np.allclose(c.T @ c, np.eye(3), atol=1e-9)
    assert abs(np.linalg.det(c) - 1.0) < 1e-9
    # Still close to the true correction despite the noise.
    assert np.allclose(c @ bad, np.eye(3), atol=0.25)


def test_correction_rejects_a_reflection():
    """A mirrored measurement set (e.g. a sign error on one axis) must not be 'fixed'
    with a reflection — that would silently flip handedness."""
    measured = {"+X": np.array([1.0, 0, 0]), "+Y": np.array([0, -1.0, 0]),
                "+Z": np.array([0, 0, 1.0])}
    c = ac.solve_correction(measured)
    assert c is not None
    assert abs(np.linalg.det(c) - 1.0) < 1e-9, "must stay a proper rotation"


def test_correction_declines_with_fewer_than_three_axes():
    """Two axes underdetermine the roll about the third; guessing would be worse than
    declining, so the tool says so instead."""
    assert ac.solve_correction({"+X": np.array([0, 1.0, 0])}) is None
    assert ac.solve_correction({"+X": np.array([0, 1.0, 0]),
                                "+Y": np.array([-1.0, 0, 0])}) is None


def test_identity_alignment_passes_every_axis():
    measured = _measured_under(np.eye(3))
    for name, expect, _h in ac.AXES:
        unit = measured[name]
        i, val = ac.dominant_axis(unit)
        assert i == int(np.argmax(expect)) and val > 0, name
        assert ac.cross_leakage(unit) < 1e-9


def test_tracker_stream_parses_role_keyed_and_legacy_payloads():
    """The bridge emits the role-keyed superset; old logs carry a flat payload. Both
    must parse, and a STALE pose must never be handed out as a measurement."""
    import asyncio
    import json as _json

    s = ac.TrackerStream("ws://unused")

    class FakeWS:
        def __init__(self, msgs):
            self._m = list(msgs)

        async def recv(self):
            if not self._m:
                raise asyncio.CancelledError
            return self._m.pop(0)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

    msgs = [
        _json.dumps({"mode": "3pt", "t": 1.0, "trackers": {
            "left": {"pos": [1, 2, 3], "stale": False},
            "right": {"pos": [4, 5, 6], "stale": True}}}),
        _json.dumps({"seq": 1, "t": 2.0, "pos": [7, 8, 9], "stale": False}),  # legacy flat
    ]

    async def drive():
        ws = FakeWS(msgs)
        stop = asyncio.Event()
        # Inline the recv loop (the real one owns the connection).
        while True:
            try:
                msg = await ws.recv()
            except asyncio.CancelledError:
                break
            d = _json.loads(msg)
            s._mode = d.get("mode", s._mode)
            trackers = d.get("trackers")
            if trackers is None and "pos" in d:
                trackers = {"left": d}
            for role, t in (trackers or {}).items():
                s._latest[role] = (np.asarray(t["pos"], dtype=float),
                                   bool(t.get("stale", False)))
        return stop

    asyncio.run(drive())
    assert s.mode == "3pt"
    # The legacy flat frame overwrote 'left'.
    assert np.allclose(s.pose("left"), [7, 8, 9])
    # A stale pose is withheld, not returned as if measured.
    assert s.pose("right") is None
    assert s.pose("head") is None


class _ScriptedStream:
    """A TrackerStream stand-in that replays a scripted motion per axis, in-process.

    Avoids binding a socket (and avoids a real bridge) while still driving the REAL
    ``_measure_axis`` / ``_check_role`` code paths, including the reference capture, the
    min-travel gate, and the settle delay.
    """

    def __init__(self, world: np.ndarray, roles=("left",), stale_roles=()):
        self.world = world
        self._roles = roles
        self._stale = set(stale_roles)
        self._axis = 0          # which AXES entry is being performed
        self._t = 0.0
        self.mode = "1pt"

    @property
    def roles_seen(self):
        return self._roles

    def next_axis(self) -> None:
        self._axis += 1
        self._t = 0.0

    def pose(self, role: str):
        if role in self._stale or role not in self._roles:
            return None
        # Ramp 30 cm along the intended axis; the first call (t==0) is the reference.
        self._t += 0.06
        d = ac.AXES[min(self._axis, len(ac.AXES) - 1)][1]
        return self.world @ (d * 0.30 * min(1.0, self._t))


def _run_check(world: np.ndarray, **kw) -> dict:
    """Drive the real _check_role against a scripted stream, auto-answering prompts."""
    import asyncio
    import builtins

    stream = _ScriptedStream(world)
    calls = {"n": 0}

    def fake_input(_prompt=""):
        # Each prompt starts a new axis (except the first) and resets the ramp.
        if calls["n"]:
            stream.next_axis()
        calls["n"] += 1
        stream._t = 0.0
        return ""

    kw.setdefault("current_rot", np.eye(3))
    kw.setdefault("assumed_identity", True)
    args = argparse.Namespace(min_travel=0.10, max_cross=0.5, timeout=5.0, **kw)
    real_input = builtins.input
    builtins.input = fake_input
    try:
        return asyncio.run(ac._check_role(stream, "left", args))
    finally:
        builtins.input = real_input


def test_end_to_end_check_passes_on_a_correct_frame():
    """The full interactive path (reference capture -> travel gate -> verdict) with an
    identity world alignment must PASS all three axes."""
    r = _run_check(np.eye(3))
    assert r["pass"] is True, r
    for label, _v, _h in ac.AXES:
        a = r["axes"][label]
        assert a["ok"] and a["sign"] == "+" and a["dominant"] == label[1].lower(), (label, a)
        assert a["cross_leakage"] < 1e-6


def test_end_to_end_check_fails_and_solves_a_yaw90_frame():
    """With a 90-degree yaw error, pushing FORWARD reads as +y. The check must fail AND
    print a correction that composes back to identity."""
    bad = _rpy(0, 0, 90)
    r = _run_check(bad)
    assert r["pass"] is False
    assert r["axes"]["+X"]["dominant"] == "y", r["axes"]["+X"]
    assert "correction_rpy_deg" in r, r
    c = _rpy(*r["correction_rpy_deg"])
    assert np.allclose(c @ bad, np.eye(3), atol=1e-6), r["correction_rpy_deg"]


def test_end_to_end_check_detects_an_inverted_axis():
    """A 180-degree yaw inverts +X and +Y while leaving +Z correct: the per-axis verdicts
    must isolate exactly that."""
    r = _run_check(_rpy(0, 0, 180))
    assert r["pass"] is False
    assert r["axes"]["+X"]["sign"] == "-" and r["axes"]["+X"]["dominant"] == "x"
    assert r["axes"]["+Y"]["sign"] == "-"
    assert r["axes"]["+Z"]["ok"] is True, r["axes"]["+Z"]


def test_rpy_and_quat_to_rot_roundtrip():
    """The inverses used to parse a CURRENT alignment must match the bridge's convention,
    or the composition below fixes the wrong rotation."""
    for rpy in [(0, 0, 0), (0, 0, 90), (0, 90, 0), (12, -33, 57), (180, 0, 0)]:
        r = _rpy(*rpy)
        assert np.allclose(ac.rpy_deg_to_rot(*rpy), r, atol=1e-12), rpy
        assert np.allclose(ac.quat_wxyz_to_rot(ac.rot_to_quat_wxyz(r)), r, atol=1e-9), rpy
        # ...and round-tripping through the Euler extraction is stable.
        assert np.allclose(ac.rpy_deg_to_rot(*ac.rot_to_rpy_deg(r)), r, atol=1e-9), rpy


def test_correction_must_be_composed_onto_the_current_alignment():
    """REGRESSION. The solved C maps MEASURED -> INTENDED, so the RELAUNCH value is
    C @ R_current, not C. Printing the bare delta is correct only from an identity start;
    measured residuals for a non-identity start were 0.88-1.0 (i.e. still broken). And the
    operator cannot compose it by hand: adding Euler triples is wrong in general —
    R_cur=(0,0,90) with C=(40,0,0) gives a true (0,-40,90), not (40,0,90)."""
    for cur in [(0, 0, 0), (0, 0, 90), (0, 90, 0), (12, -33, 57)]:
        r_cur = _rpy(*cur)
        # The bridge currently applies r_cur; the CORRECT map is identity, so the operator
        # measures r_cur @ intended.
        measured = {n: r_cur @ v for n, v, _h in ac.AXES}
        c = ac.solve_correction(measured)
        assert c is not None
        # The composed value fixes the rig exactly...
        assert np.allclose(c @ r_cur, np.eye(3), atol=1e-9), cur
        # ...while the bare delta does NOT, unless the start was identity.
        if not np.allclose(r_cur, np.eye(3)):
            assert not np.allclose(c, np.eye(3), atol=1e-6), cur

    # Euler addition is genuinely wrong, so the tool must do the composition.
    r_cur = _rpy(0, 0, 90)
    c = _rpy(40, 0, 0)
    true = ac.rot_to_rpy_deg(c @ r_cur)
    assert not np.allclose(true, (40, 0, 90), atol=1.0), true


def test_prompt_is_read_off_the_event_loop():
    """REGRESSION (CRITICAL). A bare ``input()`` inside the coroutine blocks the whole
    asyncio loop, so the only WS consumer stops draining while the operator reads the
    prompt. The socket then buffers seconds of ~90 Hz poses, the reference is captured
    from a stale register, and the poll loop replays that BACKLOG as if it were the push —
    reproduced as a reported 40 cm move for a tracker that never moved, and as a false
    FAIL on a correctly-aligned rig. Worse, the same mechanism can produce a false PASS on
    a mis-aligned one, i.e. the exact opposite of this tool's purpose.

    Pinned structurally: the prompt must go through run_in_executor, and the reader must
    drain to the newest frame rather than replay history."""
    import inspect

    src = inspect.getsource(ac._measure_axis)
    assert "run_in_executor(None, input" in src, "prompt must not block the event loop"
    # A bare input(...) call must not remain.
    assert "\n    input(" not in src and " input(f" not in src, src

    run_src = inspect.getsource(ac.TrackerStream.run)
    assert "_DRAIN_MAX" in run_src, "reader must drain the backlog, not replay it"


def test_prompt_handles_non_interactive_stdin():
    """The launcher documents this check as interactive. With piped stdin it must say so
    and return None, not die on an uncaught EOFError traceback."""
    import asyncio
    import builtins

    stream = _ScriptedStream(np.eye(3))
    real = builtins.input

    def raise_eof(_p=""):
        raise EOFError

    builtins.input = raise_eof
    try:
        got = asyncio.run(ac._measure_axis(stream, "left", "+X", "fwd",
                                           min_travel=0.1, timeout=1.0))
    finally:
        builtins.input = real
    assert got is None


def test_stale_tracker_at_the_reference_is_refused():
    """A stale pose must never become the reference — the whole measurement hangs off it."""
    import asyncio
    import builtins

    stream = _ScriptedStream(np.eye(3), stale_roles=("left",))
    real = builtins.input
    builtins.input = lambda _p="": ""
    try:
        got = asyncio.run(ac._measure_axis(stream, "left", "+X", "fwd",
                                           min_travel=0.1, timeout=1.0))
    finally:
        builtins.input = real
    assert got is None



def test_sloppy_push_is_a_redo_not_a_frame_correction():
    """REGRESSION. Excessive off-axis leakage means the OPERATOR pushed diagonally, not
    that the frame is wrong. Folding it into the same boolean as axis/sign let a PERFECT
    rig with 28-degree-off pushes reach solve_correction and print a confident, wrong
    WORLD_RPY (-15.9, -20.96, -15.9), aborting the launch."""
    import asyncio
    import builtins

    # A correct frame, but every push leaks 45 degrees into the next axis.
    class Diagonal(_ScriptedStream):
        def pose(self, role):
            p = super().pose(role)
            if p is None:
                return None
            d = ac.AXES[min(self._axis, 2)][1]
            skew = np.roll(d, 1) * 0.6          # large off-axis component
            return p + skew * float(np.linalg.norm(p))

    stream = Diagonal(np.eye(3))
    calls = {"n": 0}

    def fake_input(_p=""):
        if calls["n"]:
            stream.next_axis()
        calls["n"] += 1
        stream._t = 0.0
        return ""

    args = argparse.Namespace(min_travel=0.10, max_cross=0.5, timeout=5.0,
                              current_rot=np.eye(3), assumed_identity=True)
    real = builtins.input
    builtins.input = fake_input
    try:
        r = asyncio.run(ac._check_role(stream, "left", args))
    finally:
        builtins.input = real

    assert r["pass"] is False
    # Every axis is right in DIRECTION and SIGN...
    assert all(a["frame_ok"] for a in r["axes"].values()), r["axes"]
    assert all(a["sloppy"] for a in r["axes"].values()), r["axes"]
    # ...so NO correction is offered — that is the whole point.
    assert "correction_rpy_deg" not in r, r.get("correction_rpy_deg")
    assert r.get("needs_redo") is True


def test_a_real_frame_error_still_gets_a_correction():
    """The counterpart: a genuine axis/sign error must still solve and print."""
    r = _run_check(_rpy(0, 0, 90))
    assert r["pass"] is False
    assert any(not a["frame_ok"] for a in r["axes"].values())
    assert "correction_rpy_deg" in r


def _run_all() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  ok  {t.__name__}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  FAIL {t.__name__}: {exc!r}")
    if failed:
        print(f"\n{failed}/{len(tests)} FAILED")
        return 1
    print(f"\nLOGIC OK ({len(tests)} checks)")
    return 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
