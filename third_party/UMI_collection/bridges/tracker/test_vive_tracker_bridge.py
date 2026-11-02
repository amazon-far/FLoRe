"""Unit tests for the tracker bridge, against a fake pysurvive.

No hardware, no libsurvive: a stub module stands in for ``pysurvive`` with the exact
API surface the bridge uses (``SimpleContext(argv)``, ``.Running()``, ``.NextUpdated()``,
``obj.Name()``/``.Pose()``/``.ptr``, ``simple_serial_number(ptr)``). Run with

    .venv/bin/python -m pytest bridges/tracker/test_vive_tracker_bridge.py -q
"""

from __future__ import annotations

import importlib
import math
import os
import re
import sys
import threading
import time
import types

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

vtb = importlib.import_module("vive_tracker_bridge")


# --- fake pysurvive ----------------------------------------------------------------
class _PoseData:
    def __init__(self, pos, rot_wxyz):
        self.Pos = list(pos)
        self.Rot = list(rot_wxyz)


class FakeObj:
    def __init__(self, ptr, name, pos=(0.0, 0.0, 0.0), rot_wxyz=(1.0, 0.0, 0.0, 0.0), ts=123.456):
        self.ptr = ptr
        self._name = name
        self.pos = pos
        self.rot = rot_wxyz
        self.ts = ts

    def Name(self):  # noqa: N802 - pysurvive's API
        return self._name.encode("utf-8")

    def Pose(self):  # noqa: N802
        return (_PoseData(self.pos, self.rot), self.ts)


class FakeContext:
    ptr = 1   # the bridge hands ctx.ptr to the live object-walk functions

    def __init__(self, argv):
        self.argv = list(argv)
        self._q: list = []
        self.attached: list = []   # what the driver has added (pose or not) — radio_devices() walks it
        self._lock = threading.Lock()
        self.running = True

    def Running(self):  # noqa: N802
        return self.running

    def Objects(self):  # noqa: N802
        return []

    def NextUpdated(self):  # noqa: N802
        with self._lock:
            return self._q.pop(0) if self._q else None

    def push(self, obj):
        with self._lock:
            if obj not in self.attached:
                self.attached.append(obj)
            self._q.append(obj)

    def attach(self, obj):
        """The driver added a tracked object that has not produced a pose yet."""
        with self._lock:
            if obj not in self.attached:
                self.attached.append(obj)


def make_fake_pysurvive(wheel: bool = False, tmp_path=None):
    mod = types.ModuleType("pysurvive")
    if tmp_path is None:
        mod.__file__ = "/fake/libsurvive/bindings/python/pysurvive/__init__.py"
    else:
        base = tmp_path / ("site-packages" if wheel else os.path.join("bindings", "python"))
        pkg = base / "pysurvive"
        pkg.mkdir(parents=True)
        (pkg / "__init__.py").write_text("")
        if wheel:
            (base / "pysurvive.libs").mkdir()
        mod.__file__ = str(pkg / "__init__.py")
    created: list[FakeContext] = []
    serials: dict[int, bytes] = {}

    def SimpleContext(argv):  # noqa: N802
        ctx = FakeContext(argv)
        created.append(ctx)
        return ctx

    def simple_serial_number(ptr):
        return serials.get(getattr(ptr, "ptr", ptr))   # the live walk hands the object itself

    def simple_get_first_object(ctx_ptr):
        ctx = created[-1]
        return ctx.attached[0] if ctx.attached else 0

    def simple_get_next_object(ctx_ptr, cur):
        ctx = created[-1]
        i = ctx.attached.index(cur) + 1
        return ctx.attached[i] if i < len(ctx.attached) else 0

    def simple_object_name(obj):
        return obj.Name()

    mod.SimpleContext = SimpleContext
    mod.simple_serial_number = simple_serial_number
    mod.simple_get_first_object = simple_get_first_object
    mod.simple_get_next_object = simple_get_next_object
    mod.simple_object_name = simple_object_name
    mod._created = created
    mod._serials = serials
    return mod


def _wait_until(pred, timeout=2.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if pred():
            return True
        time.sleep(0.01)
    return False


@pytest.fixture
def tracker():
    ps = make_fake_pysurvive()
    tr = vtb.ViveTracker(ps)
    tr.connect()
    yield tr, ps, ps._created[0]
    tr.disconnect()


# --- reference rules ------------------------------------------------------------------
def test_default_survive_args(tracker):
    tr, ps, ctx = tracker
    assert ctx.argv[0] == sys.argv[0]
    assert ctx.argv[1:] == ["--force-calibrate", "1", "--force-ootx", "1", "--lighthouse-update-velocity", "0.5"]
    # 120 s: the reference recorder waits 90 s (--vive-warmup-timeout); 2 stations lock in ~20-25 s there
    assert vtb.STARTUP_WAIT_S == 120.0 and vtb.MAX_AGE_S == 1.0
    assert vtb.LIBSURVIVE_COMMIT == "5588cadf5ab764b51e50f9b08996009d1b9f836e"


def test_collector_keys_by_serial_and_notes_lighthouses(tracker):
    tr, ps, ctx = tracker
    ps._serials.update({1: b"LHR-AAAA1111", 2: b"LHB-0000BEEF"})
    ctx.push(FakeObj(1, "T20", pos=(0.1, 0.2, 0.3), rot_wxyz=(0.5, 0.5, 0.5, 0.5)))
    ctx.push(FakeObj(2, "LH0"))
    assert _wait_until(lambda: "LHR-AAAA1111" in tr.devices() and tr.lighthouses())
    assert tr.devices() == {"LHR-AAAA1111": "T20"}
    assert tr.lighthouses() == ["LH0"]
    assert tr.lighthouse_serials() == {"LH0": "LHB-0000BEEF"}
    pose = tr.get_pose("LHR-AAAA1111")
    assert pose.device_name == "T20" and pose.serial == "LHR-AAAA1111"
    np.testing.assert_allclose(pose.pos, [0.1, 0.2, 0.3])
    np.testing.assert_allclose(pose.quat_wxyz, [0.5, 0.5, 0.5, 0.5])   # wxyz preserved
    assert pose.timestamp == 123.456
    assert abs(time.time() - pose.received_at) < 2.0
    # the bare libsurvive name resolves too (the reference HeadViveTracker.get_pose fallback)
    assert tr.get_pose("T20") is pose
    assert tr.update_counts() == {"T20": 1}


def test_wait_for_raises_naming_only_the_missing_roles(tracker):
    tr, ps, ctx = tracker
    ps._serials[1] = b"LHR-LEFT0001"
    ctx.push(FakeObj(1, "T20"))
    assert _wait_until(lambda: tr.get_pose("LHR-LEFT0001") is not None)
    with pytest.raises(RuntimeError) as ei:
        tr.wait_for({"left": "LHR-LEFT0001", "right": "LHR-RIGHT001"}, timeout_s=0.4)
    msg = str(ei.value)
    assert "right=LHR-RIGHT001" in msg and "left=" not in msg
    assert "LHR-LEFT0001" in msg   # what WAS seen is named


def test_wait_for_names_pucks_on_the_radio_without_a_pose(tracker):
    tr, ps, ctx = tracker
    ps._serials[7] = b"LHR-L"
    ctx.attach(FakeObj(7, "WM3"))            # added by the driver, no pose yet
    assert tr.radio_devices() == ["WM3"] and tr.radio_serials() == {"WM3": "LHR-L"}
    with pytest.raises(RuntimeError) as ei:
        tr.wait_for({"left": "LHR-L"}, timeout_s=0.2)
    msg = str(ei.value)
    # on the radio, no pose, no station positioned: the ROOM is the blocker, not this puck
    assert re.search(r"left\s+LHR-L\s+WM3\s+yes\s+none", msg)
    # no station has a POSITION and none is pending OOTX either (no light monitor): the room is
    # the blocker, and the advice is line of sight — not a hold-still or one-puck ritual
    assert "VERDICT shared: every station has OOTX" in msg and "none has a solved POSITION yet" in msg
    assert "sees >=2 stations at once" in msg
    assert "HOLD" not in msg and "ONE PUCK" not in msg
    assert "SUMMARY: puck fault(s) = none; shared blocker = every station has OOTX" in msg


def test_wait_for_flags_a_serial_mismatch(tracker):
    tr, ps, ctx = tracker
    ps._serials[7] = b"LHR-OTHER"
    ctx.attach(FakeObj(7, "WM3"))            # a puck IS on the radio — under a serial no role names
    with pytest.raises(RuntimeError) as ei:
        tr.wait_for({"left": "LHR-L"}, timeout_s=0.2)
    msg = str(ei.value)
    assert "VERDICT puck: left (LHR-L) NOT on the radio" in msg
    assert "(unassigned)  LHR-OTHER" in msg and "--*-serial mismatch" in msg
    assert "SUMMARY: puck fault(s) = left(no-radio)" in msg


# --- diagnose(): which puck, or the room? ------------------------------------------
def _light(syncs, done=None, decoder=None, restarts=None, available=True):
    return vtb.LightState(available, syncs, done or {}, decoder or {}, restarts or {}, None)


def test_diagnose_room_solve_is_shared_not_a_puck_fault():
    roles = {"left": "LHR-A", "right": "LHR-B"}
    radio = {"WM0": "LHR-A", "WM1": "LHR-B"}
    light = _light({"WM0": {1: 100, 8: 90}, "WM1": {2: 50}}, done={1: "ebfe746a"},
                   decoder={2: "WM1", 4: "WM0"}, restarts={4: 3})
    d = vtb.diagnose(roles, radio, {}, [], light, now=1000.0)
    assert [r.state for r in d.rows] == ["blocked-room", "blocked-room"]
    assert d.puck_problems == []
    # the gate, named: one undecoded blob blocks the room however good the rest is
    assert "OOTX 1/4 stations decoded (pending ch2 ch4 ch8)" in d.shared_blocker
    assert "attempts NO room solve until EVERY station" in d.shared_blocker
    assert "unbroken line of sight from the puck it decodes through" in d.shared_blocker
    # the pin is permanent, so the puck a pending channel decodes through is the one to aim
    assert "ch4 through WM0 (left), 3 restarts" in d.shared_blocker
    assert "ch2 through WM1 (right)" in d.shared_blocker
    # the reference flow: every puck stays awake; the remedy is line of sight (or fewer stations), never one puck
    assert "faster powered OFF than decoded" in d.shared_blocker
    assert "ONE PUCK" not in d.shared_blocker and "MODE=1pt" not in d.shared_blocker
    text = "\n".join(d.lines())
    assert "ch1 ch8 (190 syncs)" in text and "no puck-specific fault" in text
    assert "pending ch2[via WM1] ch4[via WM0, 3 restarts] ch8" in text
    assert d.summary().startswith("puck fault(s) = none; shared blocker = OOTX 1/4 stations decoded")


def test_diagnose_names_the_dark_puck_and_the_absent_one():
    roles = {"left": "LHR-A", "head": "LHR-C", "right_foot": "LHR-E"}
    radio = {"WM0": "LHR-A", "WM2": "LHR-C"}     # right_foot's puck never attached
    light = _light({"WM0": {1: 10}})               # head (WM2) has synced to nothing
    d = vtb.diagnose(roles, radio, {}, [], light, now=1000.0)
    assert {r.role: r.state for r in d.rows} == {"left": "blocked-room", "head": "no-light", "right_foot": "no-radio"}
    assert d.puck_problems == ["head", "right_foot"]
    text = "\n".join(d.lines())
    assert "VERDICT puck: head (LHR-C, WM2) on the radio but sees NO base station" in text
    assert "VERDICT puck: right_foot (LHR-E) NOT on the radio" in text
    assert re.search(r"head\s+LHR-C\s+WM2\s+yes\s+none\s+\?\s+NONE\s+<--", text)   # still='?': no IMU seen
    assert "puck fault(s) = head(no-light), right_foot(no-radio)" in d.summary()


def test_diagnose_with_the_room_solved_blames_the_puck():
    roles = {"left": "LHR-A", "right": "LHR-B"}
    radio = {"WM0": "LHR-A", "WM1": "LHR-B"}
    now = time.time()
    poses = {"LHR-A": _pose("LHR-A", (0, 0, 0), received_at=now),
             "LHR-B": _pose("LHR-B", (0, 0, 0), received_at=now - 5.0)}
    light = _light({"WM0": {1: 10, 2: 10}, "WM1": {1: 3}})
    d = vtb.diagnose(roles, radio, poses, ["LH0", "LH1"], light, now=now)
    assert {r.role: r.state for r in d.rows} == {"left": "ok", "right": "stale"}
    assert d.shared_blocker is None and d.puck_problems == ["right"]
    assert "stale 5s" in "\n".join(d.lines())
    # sees light, never got a pose, while stations ARE positioned: its own problem
    d2 = vtb.diagnose(roles, radio, {"LHR-A": poses["LHR-A"]}, ["LH0"], light, now=now)
    assert {r.role: r.state for r in d2.rows}["right"] == "no-pose" and d2.puck_problems == ["right"]


def test_diagnose_without_the_light_monitor_still_separates_radio_from_room():
    roles = {"left": "LHR-A", "right": "LHR-B"}
    d = vtb.diagnose(roles, {"WM0": "LHR-A"}, {}, [], vtb.unavailable_light_state(), now=1.0)
    assert {r.role: r.state for r in d.rows} == {"left": "blocked-room", "right": "no-radio"}
    text = "\n".join(d.lines())
    assert re.search(r"left\s+LHR-A\s+WM0\s+yes\s+none\s+\?\s+\?", text)   # still '?', sees '?'
    assert "stations:" not in text   # no per-station line without the monitor


def test_light_monitor_parses_libsurvive_lines():
    mon = vtb.LightMonitor(None)
    for line in ["Info: OOTX not set for LH in channel 10; attaching ootx decoder using device WM1",
                 "Info: OOTX not set for LH in channel 1; attaching ootx decoder using device WM4",
                 "Info: Got OOTX packet 1 ebfe746a", "Info: (10) OOTX Decoder: Bad sync bit",
                 "Info: (10) OOTX Decoder: Bad sync bit", "Info: (10) Preamble found",
                 "Info: Using LH 2 as reference lighthouse", "Info: Adding tracked object WM0 from HTC",
                 "   87.08 Warning: MPFIT status failure 744430459400.026733/19726552.840980/0.015027 "
                 "(886 measurements, 5, MP_MAXITER)",
                 "Info: MPFIT success 0.531/0.0001234567/0.0021000 (1126 measurements, 1, MP_OK_CHI, 12 iters, "
                 "up err 0.0031000, trace 0.0000010)",
                 "Info: Global solve with 12 scenes for 0 with error of 0.531000/0.0001234567 (acc err 0.0310)",
                 "Info: Global solve with 12 scenes for 2 with error of 0.531000/0.0001234567 (acc err 0.0012)"]:
        mon.feed_log_line(line, now=100.0)
    st = mon.state()
    assert st.available is False   # never attached to a context here
    assert st.ootx_done == {1: "ebfe746a"} and st.ootx_decoder == {10: "WM1", 1: "WM4"}
    assert st.ootx_restarts == {10: 2} and st.reference_lh == 2
    assert st.channels_seen() == [1, 10] and st.ootx_pending() == [10]
    # the room solve: the reference "acc err" line, per station, plus the attempt tally
    assert st.mpfit_fail == 1 and st.mpfit_ok == 1
    assert st.last_mpfit == "success (1126 meas, 12 iters, up err 0.0031)" and st.last_mpfit_t == 100.0
    assert st.solves == {0: {"scenes": 12, "err": 0.0001234567, "acc_err": 0.031, "t": 100.0},
                         2: {"scenes": 12, "err": 0.0001234567, "acc_err": 0.0012, "t": 100.0}}
    line = vtb.format_solves(st, {"LH0": "LH0(ch10)"}, now=103.0)
    assert line == ("MPFIT ok 1 / failed 1 (last success (1126 meas, 12 iters, up err 0.0031), 3s ago); "
                    "acc err LH0(ch10) 0.0310 (~14°, 12 scenes, err 0.000123), LH2 0.0012 (~3°, 12 scenes, err 0.000123)")
    assert vtb.format_solves(vtb.unavailable_light_state()) == ""


def test_light_monitor_tracks_when_each_station_entered_the_run():
    mon = vtb.LightMonitor(None)
    mon.feed_log_line("Info: Adding lighthouse ch 4 (idx: 0, cnt: 1)", now=100.0)
    mon.feed_log_line("Info: Adding lighthouse ch 10 (idx: 1, cnt: 2)", now=140.0)
    mon.feed_log_line("Info: Adding lighthouse ch 4 (idx: 0, cnt: 1)", now=200.0)   # re-log: keep the first
    st = mon.state()
    assert st.added == {4: 100.0, 10: 140.0}


def test_diagnose_calls_out_a_station_that_only_just_joined():
    roles = {"left": "LHR-A"}
    light = _light({"WM0": {4: 10, 10: 3}}, decoder={4: "WM0", 10: "WM0"})
    light.added.update({4: 100.0, 10: 148.0})
    d = vtb.diagnose(roles, {"WM0": "LHR-A"}, {}, [], light, now=150.0)
    text = "\n".join(d.lines())
    assert "ch10 only entered this run 2s ago" in text and "it only grows" in text
    # ... and stays quiet once it is old news
    light.added[10] = 100.0
    d2 = vtb.diagnose(roles, {"WM0": "LHR-A"}, {}, [], light, now=200.0)
    assert "only entered this run" not in "\n".join(d2.lines())


def test_diagnose_reports_stationarity_and_names_a_moving_ootx_decoder():
    roles = {"left": "LHR-A", "right": "LHR-B"}
    radio = {"WM0": "LHR-A", "WM1": "LHR-B"}
    light = _light({"WM0": {8: 50}, "WM1": {8: 20}}, decoder={8: "WM0"})
    light.motion.update({"WM0": 0.2, "WM1": 4.0})     # left is being waved, right is held
    d = vtb.diagnose(roles, radio, {}, [], light, now=1000.0)
    text = "\n".join(d.lines())
    assert re.search(r"left\s+LHR-A\s+WM0\s+yes\s+none\s+MOVING", text)
    assert re.search(r"right\s+LHR-B\s+WM1\s+yes\s+none\s+4s", text)
    # the ONE puck ch8's blob can ever decode through is the one in motion
    assert "WM0 (left) is MOVING and ch8 decodes only through it" in d.shared_blocker


def test_diagnose_shows_the_solve_line_and_labels_positioned_stations_by_channel():
    roles = {"left": "LHR-A"}
    light = vtb.LightState(True, {"WM0": {10: 5}}, {10: "e66022c0"}, {}, {}, 0,
                           solves={0: {"scenes": 9, "err": 0.002, "acc_err": 0.031, "t": 50.0}},
                           mpfit_ok=1, mpfit_fail=4, last_mpfit="success (900 meas, 8 iters, up err 0.0100)",
                           last_mpfit_t=50.0)
    d = vtb.diagnose(roles, {"WM0": "LHR-A"}, {}, ["LH0(ch10)"], light, now=52.0, lh_labels={"LH0": "LH0(ch10)"})
    text = "\n".join(d.lines())
    assert "stations: positioned LH0(ch10); OOTX done 1/1 seen (ch10)" in text
    assert "solve: MPFIT ok 1 / failed 4 (last success (900 meas, 8 iters, up err 0.0100), 2s ago); " \
           "acc err LH0(ch10) 0.0310 (~14°, 9 scenes, err 0.002)" in text
    # the room IS solved and this puck sees light: its own problem now, not the room's
    assert {r.role: r.state for r in d.rows} == {"left": "no-pose"} and d.shared_blocker is None


def test_wait_for_returns_once_every_role_is_fresh(tracker):
    tr, ps, ctx = tracker
    ps._serials.update({1: b"LHR-L", 2: b"LHR-R"})
    ctx.push(FakeObj(1, "T20"))
    ctx.push(FakeObj(2, "T21"))
    elapsed = tr.wait_for({"left": "LHR-L", "right": "LHR-R"}, timeout_s=2.0)
    assert 0.0 <= elapsed < 2.0


def test_wait_for_treats_an_old_pose_as_not_fresh(tracker):
    tr, ps, ctx = tracker
    ps._serials[1] = b"LHR-L"
    ctx.push(FakeObj(1, "T20"))
    assert _wait_until(lambda: tr.get_pose("LHR-L") is not None)
    time.sleep(0.06)
    with pytest.raises(RuntimeError):
        tr.wait_for({"left": "LHR-L"}, timeout_s=0.2, max_age_s=0.01)


# --- our wire ----------------------------------------------------------------------
class _StaticTracker:
    """A ViveTracker stand-in with poses set directly (no thread)."""

    def __init__(self):
        self.poses: dict[str, vtb.TrackerPose] = {}
        self.lhs: list[str] = []

    def get_pose(self, serial):
        return self.poses.get(serial)

    def lighthouses(self):
        return list(self.lhs)


def _pose(serial, pos, quat_wxyz=(1, 0, 0, 0), received_at=None):
    return vtb.TrackerPose("T20", serial, 1.0, time.time() if received_at is None else received_at,
                           np.array(pos, float), np.array(quat_wxyz, float))


def test_snapshot_wire_shape_roles_absent_until_seen_and_yaw180():
    st = _StaticTracker()
    q_world = vtb.quat_from_rpy(0.0, 0.0, math.pi)   # the rig default WORLD_RPY="0 0 180"
    br = vtb.TrackerBridge(st, "2pt", {"left": "LHR-L", "right": "LHR-R"}, q_world)
    assert br.snapshot() is None
    st.poses["LHR-L"] = _pose("LHR-L", (1.0, 0.0, 0.0))
    st.lhs = ["LH0", "LH1"]
    snap = br.snapshot()
    assert set(snap) == {"mode", "t", "trackers", "lighthouses"}
    assert snap["mode"] == "2pt" and set(snap["trackers"]) == {"left"}     # right ABSENT, not zero
    left = snap["trackers"]["left"]
    assert set(left) == {"seq", "t", "pos", "quat", "stale"}
    np.testing.assert_allclose(left["pos"], [-1.0, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(left["quat"], [0.0, 0.0, 0.0, 1.0], atol=1e-12)  # q_world * identity
    assert left["stale"] is False and left["seq"] == 1
    assert snap["lighthouses"] == {"count": 2, "names": ["LH0", "LH1"]}
    assert br.snapshot()["trackers"]["left"]["seq"] == 2


def test_stale_flag_follows_received_at():
    st = _StaticTracker()
    br = vtb.TrackerBridge(st, "1pt", {"left": "LHR-L"}, np.array([1.0, 0, 0, 0]), stale_s=0.25)
    st.poses["LHR-L"] = _pose("LHR-L", (0, 0, 0), received_at=time.time() - 1.0)
    assert br.snapshot()["trackers"]["left"]["stale"] is True
    br0 = vtb.TrackerBridge(st, "1pt", {"left": "LHR-L"}, np.array([1.0, 0, 0, 0]), stale_s=0.0)
    assert br0.snapshot()["trackers"]["left"]["stale"] is False


def test_quat_from_rpy_matches_axis_check_convention():
    axis_check = importlib.import_module("axis_check")
    rng = np.random.default_rng(0)
    for _ in range(20):
        r, p, y = rng.uniform(-170, 170, size=3)
        rot = axis_check.rpy_deg_to_rot(r, p, y)
        q = vtb.quat_from_rpy(math.radians(r), math.radians(p), math.radians(y))
        for v in np.eye(3):
            np.testing.assert_allclose(vtb.rot_by_quat(q, v), rot @ v, atol=1e-9)


# --- provenance + CLI ---------------------------------------------------------------
def test_wheel_provenance_detection(tmp_path):
    wheel = make_fake_pysurvive(wheel=True, tmp_path=tmp_path / "w")
    src = make_fake_pysurvive(wheel=False, tmp_path=tmp_path / "s")
    assert vtb.pysurvive_provenance(wheel)[1] is True
    assert vtb.pysurvive_provenance(src)[1] is False


def test_import_pysurvive_refuses_the_wheel(monkeypatch, tmp_path):
    wheel = make_fake_pysurvive(wheel=True, tmp_path=tmp_path)
    monkeypatch.setitem(sys.modules, "pysurvive", wheel)
    with pytest.raises(SystemExit) as ei:
        vtb.import_pysurvive()
    assert "refusing the PyPI pysurvive wheel" in str(ei.value)
    assert vtb.import_pysurvive(allow_wheel=True) is wheel


def test_import_pysurvive_accepts_source_bindings(monkeypatch, tmp_path):
    src = make_fake_pysurvive(wheel=False, tmp_path=tmp_path)
    monkeypatch.setitem(sys.modules, "pysurvive", src)
    assert vtb.import_pysurvive() is src


def test_resolve_roles_needs_every_flag_of_the_mode():
    ap = vtb.build_parser()
    with pytest.raises(SystemExit, match="--right-serial"):
        vtb.resolve_roles(ap.parse_args(["--mode", "2pt", "--left-serial", "LHR-A"]))
    roles = vtb.resolve_roles(ap.parse_args(["--mode", "3pt", "--left-serial", "LHR-A",
                                             "--right-serial", "LHR-B", "--head-serial", "LHR-C"]))
    assert roles == {"left": "LHR-A", "right": "LHR-B", "head": "LHR-C"}
    with pytest.raises(SystemExit, match="several roles"):
        vtb.resolve_roles(ap.parse_args(["--mode", "2pt", "--left-serial", "LHR-A", "--right-serial", "LHR-A"]))
    five = vtb.resolve_roles(ap.parse_args(
        ["--mode", "5pt", "--left-serial", "a", "--right-serial", "b", "--head-serial", "c",
         "--left-foot-serial", "d", "--right-foot-serial", "e"]))
    assert tuple(five) == vtb.ROLES_BY_MODE["5pt"]


def test_world_quat_prefers_rpy_over_yaw():
    ap = vtb.build_parser()
    q = vtb.world_quat(ap.parse_args(["--yaw-deg", "90", "--world-rpy-deg", "0", "0", "180"]))
    np.testing.assert_allclose(q, [0, 0, 0, 1], atol=1e-12)
    q = vtb.world_quat(ap.parse_args(["--yaw-deg", "180"]))
    np.testing.assert_allclose(q, [0, 0, 0, 1], atol=1e-12)


def test_list_serials_prints_the_detect_line_and_pair_puck_markers(capsys):
    ps = make_fake_pysurvive()
    ps._serials.update({1: b"LHR-AAAA1111", 2: b"LHR-BBBB2222", 3: b"LHB-00000001"})
    orig = ps.SimpleContext

    def SimpleContext(argv):  # noqa: N802 - pre-load the queue the collector will drain
        ctx = orig(argv)
        ctx.push(FakeObj(1, "WM0"))
        ctx.push(FakeObj(2, "WM1"))
        ctx.push(FakeObj(3, "LH0"))
        return ctx

    ps.SimpleContext = SimpleContext
    rc = vtb.list_serials(ps, 0.3, {"LHR-AAAA1111": "left"})
    out = capsys.readouterr().out
    assert rc == 0
    assert "DETECT_SERIALS: LHR-AAAA1111,LHR-BBBB2222" in out
    assert "tracker WM0 serial=LHR-AAAA1111 <- LEFT" in out
    assert "tracker WM1 serial=LHR-BBBB2222 <- UNASSIGNED" in out   # what pair_puck.sh greps for
    assert "dongles on USB" in out
    assert "lighthouse LH0 serial=LHB-00000001" in out


def test_source_build_env_points_at_the_source_build(tmp_path, monkeypatch):
    assert vtb.source_build_env(str(tmp_path / "nowhere")) is None
    root = tmp_path / "libsurvive"
    (root / "bin").mkdir(parents=True)
    (root / "bin" / "libsurvive.so").write_bytes(b"")
    (root / "bindings" / "python" / "pysurvive").mkdir(parents=True)
    (root / "bindings" / "python" / "pysurvive" / "__init__.py").write_text("")
    monkeypatch.setenv("LD_LIBRARY_PATH", "/opt/x")
    monkeypatch.delenv("PYTHONPATH", raising=False)
    env = vtb.source_build_env(str(root))
    assert env["LD_LIBRARY_PATH"] == f"{root}/bin:/opt/x"
    assert env["PYTHONPATH"] == f"{root}/bindings/python"
    # already set -> unchanged (this is what makes ensure_source_build_env a no-op under the launcher)
    monkeypatch.setenv("LD_LIBRARY_PATH", f"{root}/bin:/opt/x")
    monkeypatch.setenv("PYTHONPATH", f"{root}/bindings/python")
    env2 = vtb.source_build_env(str(root))
    assert env2["LD_LIBRARY_PATH"] == os.environ["LD_LIBRARY_PATH"]
    assert env2["PYTHONPATH"] == os.environ["PYTHONPATH"]


def test_light_monitor_counts_a_second_preamble_before_decode_as_a_restart():
    """A station repeats its blob back to back: a 2nd preamble on an undecoded channel = a lost frame.
    Measured trial 1/5 (2 stations, 5 pucks): 8 preambles on ch4 and no packet in 120 s."""
    mon = vtb.LightMonitor.__new__(vtb.LightMonitor)
    mon._lock = __import__("threading").Lock()
    mon._syncs, mon._ootx_done, mon._ootx_decoder, mon._ootx_restarts = {}, {}, {}, {}
    mon._ootx_preambles, mon._added, mon._solves, mon._motion = {}, {}, {}, {}
    mon._reference_lh, mon._mpfit_ok, mon._mpfit_fail, mon._last_mpfit, mon._last_mpfit_t = None, 0, 0, "", None
    mon.feed_log_line("Info: (4) Preamble found")           # the first frame start is not a restart
    assert mon._ootx_restarts == {}
    mon.feed_log_line("Info: (4) Preamble found")
    mon.feed_log_line("Info: (4) Preamble found")
    assert mon._ootx_restarts == {4: 2}
    mon.feed_log_line("Info: Got OOTX packet 4 97154a7d")
    mon.feed_log_line("Info: (4) Preamble found")           # decoded: later preambles are not restarts
    assert mon._ootx_restarts == {4: 2}
    mon.feed_log_line("Info: (8) OOTX Decoder: Bad sync bit")   # the old reset line still counts
    assert mon._ootx_restarts == {4: 2, 8: 1}


def test_diagnose_names_the_light_starved_puck_a_pending_channel_is_pinned_to():
    """Trial 1/5 measured: ch4 pinned to the left_foot puck at 47% of the best puck's light -> FAIL@120 s,
    while the same room with the pin on a well-lit puck locked in 18-29 s. The table must say which puck."""
    roles = {"left": "LHR-A", "left_foot": "LHR-F"}
    radio = {"WM3": "LHR-A", "WM4": "LHR-F"}
    light = _light({"WM3": {4: 5800, 8: 5800}, "WM4": {4: 2700, 8: 2700}}, done={8: "ad3c6894"},
                   decoder={4: "WM4", 8: "WM3"}, restarts={4: 8})
    assert light.light_share("WM4") == pytest.approx(2700 / 5800)
    d = vtb.diagnose(roles, radio, {}, [], light, now=1000.0)
    assert "ch4 through WM4 (left_foot), 8 restarts" in d.shared_blocker
    assert "WM4 (left_foot) gets only 47% of the light the best-placed puck gets" in d.shared_blocker
    assert "turn THAT puck to face the stations (no waving" in d.shared_blocker
    # a pin on the best-lit puck is not called out
    light2 = _light({"WM3": {4: 5800, 8: 5800}, "WM4": {4: 2700, 8: 2700}}, done={8: "ad3c6894"},
                    decoder={4: "WM3", 8: "WM3"})
    d2 = vtb.diagnose(roles, radio, {}, [], light2, now=1000.0)
    assert "gets only" not in d2.shared_blocker


def test_acc_err_matches_poser_mpfit_formula():
    # A station whose solved rotation is identity and whose tilt sensor says "up is +z": no error.
    assert vtb.acc_err([1, 0, 0, 0], [0, 0, 1]) == pytest.approx(0.0)
    # No OOTX yet (all-zero accel): undefined, not 1.0.
    assert vtb.acc_err([1, 0, 0, 0], [0, 0, 0]) is None
    # The two stations of session_2026.09.11_16.38.51 (config.json poses + OOTX accel), the
    # numbers the launcher's gate should have shown: 0.0545 (~19°) and 0.0980 (~26°).
    e0 = vtb.acc_err([0.4592, 0.0391, 0.3073, 0.8326], [-6, 127, 106])
    e1 = vtb.acc_err([-0.7861, -0.2130, -0.0480, -0.5782], [-3, 127, 120])
    assert e0 == pytest.approx(0.0545, abs=2e-3) and e1 == pytest.approx(0.0980, abs=2e-3)
    assert vtb.tilt_deg(e0) == pytest.approx(19.0, abs=1.0)
    # A 10 deg tilt of the pose against the sensor reads 1 - cos(10deg).
    q = vtb.quat_from_rpy(math.radians(10.0), 0.0, 0.0)
    assert vtb.tilt_deg(vtb.acc_err(q, [0, 0, 1])) == pytest.approx(10.0, abs=1e-6)


def test_format_acc_err_labels_channels_and_names_unsolved_stations():
    line = vtb.format_acc_err({"LH1": 0.0980, "LH0": 0.0545, "LH2": None},
                              lh_labels={"LH0": "LH0(ch4)", "LH1": "LH1(ch8)"})
    assert line == "LH0(ch4) 0.0545 (~19°), LH1(ch8) 0.0980 (~26°), LH2 unsolved"
