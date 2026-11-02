#!/usr/bin/env python
"""Panel-symmetry tests for the DUAL-HAND collection viz (collection_viz.py).

The rig grew a second hand: a second VIVE puck, a second D405, a second DM4310
gripper and a second Bota F/T sensor. The panel had followed only partway — the
right hand got a status row and a mesh, while the F/T charts, the tare, the
raw/compensated toggle, the engine dropdown and the payload recompensation all
still belonged to the single primary sensor. These tests pin the SYMMETRY, and in
particular the three ways a two-hand panel goes wrong silently:

  1. **Crossed controls.** Every control here is built in a `for hand in ...`
     loop, so a callback that closes over the LOOP VARIABLE instead of binding it
     drives whichever hand happened to be last. That failure is invisible in the
     UI (the button works, it just moves the other gripper) and would put a real
     grasp on the idle hand mid-take. Tests CLICK the real viser callbacks.
  2. **Shared state.** A tare / algo switch / recompensation that reaches the
     other hand's SensorState corrupts a wrench nobody is watching. Each control
     is asserted to touch its OWN state and to leave the sibling untouched.
  3. **Two concurrent recompensations.** The overlay is a fullscreen prompt to
     rotate *the* tool and the estimator EKF needs a steady ~1 kHz feed; two at
     once is an instruction nobody can follow and a race for the interpreter.

Hardware-free: ``bota_ft_viz.SensorState`` is a plain dataclass (the bota driver
imports live inside SensorReader's methods), so the REAL state objects and the
REAL scene/GUI are used — only the sensor readers and the CAN bridges are absent.
Display is gripper-only so no mink IK / URDF solve is needed.

Run:  .venv/bin/python viz/test_dual_hand_panel.py
Prints ``PANEL OK  N passed`` on success, ``PANEL FAIL ...`` (exit 1) otherwise.
"""
from __future__ import annotations

import inspect
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_ROOT, "force_torque"))

import viser  # noqa: E402

import bota_ft_viz as bft  # noqa: E402
import collection_viz as cv  # noqa: E402

# Ports well clear of the live rig (8080/8772-8779) and of the other smoke tests
# (8096-8101) so a test run can never talk to a running collection session.
_PORT = 8150


# ---------------------------------------------------------------------------
# GUI helpers: find + fire the REAL viser handles the scene built
# ---------------------------------------------------------------------------


def _handles(server):
    """[(folder_label, widget_label, handle)] for every GUI input on the page."""
    folders = {}
    for uuid, c in server.gui._container_handle_from_uuid.items():
        props = getattr(getattr(c, "_impl", None), "props", None)
        folders[uuid] = getattr(props, "label", None) if props is not None else None
    out = []
    for h in server.gui._gui_input_handle_from_uuid.values():
        impl = h._impl
        out.append((folders.get(impl.parent_container_id),
                    getattr(impl.props, "label", None), h))
    return out


def _find(server, folder: str, label: str):
    hits = [h for f, w, h in _handles(server) if f == folder and w == label]
    assert len(hits) == 1, f"{folder!r}/{label!r}: expected 1 handle, found {len(hits)}"
    return hits[0]


def _folders(server) -> list:
    out = []
    for c in server.gui._container_handle_from_uuid.values():
        props = getattr(getattr(c, "_impl", None), "props", None)
        lbl = getattr(props, "label", None) if props is not None else None
        if lbl:
            out.append(lbl)
    return out


def _click(handle) -> None:
    """Fire a button's registered on_click callbacks, exactly as a browser click
    does (viser dispatches `_impl.update_cb` on the GuiUpdateMessage)."""
    assert handle._impl.update_cb, "handle has no registered callback"
    for cb in list(handle._impl.update_cb):
        cb(None)


def _set(handle, value) -> None:
    """Assign a widget value the way a user drag/click does, then fire its
    callbacks (a bare `.value =` write from Python does not dispatch them)."""
    handle._impl.value = value
    handle._impl.props.value = value
    for cb in list(handle._impl.update_cb):
        cb(None)


# ---------------------------------------------------------------------------
# Rig builder
# ---------------------------------------------------------------------------


class _Rig:
    """A real CollectionScene with the requested hands' devices, plus recording
    sinks so the gripper control frames and the recorder status can be asserted."""

    def __init__(self, *, mode="3pt", left_ft=True, right_ft=True,
                 right_gripper=True, comp=True, recomp=True):
        global _PORT
        _PORT += 1
        self.sent = {"left": [], "right": []}
        self.state = cv.CollectionState()
        self.ft = {}
        for hand, want in (("left", left_ft), ("right", right_ft)):
            if not want:
                continue
            st = bft.SensorState()
            st.connected = True
            st.compensation_available = comp
            st.show_compensated = comp
            st.algo = "bota"
            self.ft[hand] = st
        self.server = viser.ViserServer(host="127.0.0.1", port=_PORT, label="panel-test")
        self.scene = cv.CollectionScene(
            self.server, self.state, ik=None, mapper=None, yam_urdf=cv._DEFAULT_YAM,
            cam_offset_xyz=(0.10, 0.0, 0.04), cam_offset_rpy=(4.7124, 1.5708, 0.0),
            cam_fov_deg=58.0, tracking_mode=mode, display_mode="gripper",
            ft_state=self.ft.get("left"), ft_state_right=self.ft.get("right"),
            gripper_send=lambda m: self.sent["left"].append(m),
            gripper_right_send=lambda m: self.sent["right"].append(m),
            recomp_enabled=recomp and comp and "left" in self.ft,
            recomp_enabled_right=recomp and comp and "right" in self.ft,
            rnea_available=True, rnea_available_right=True,
            has_right_camera=False, has_right_gripper=right_gripper,
            has_right_ft="right" in self.ft,
        )

    def close(self):
        self.server.stop()

    # -- convenience ------------------------------------------------------
    def ft_folder(self, hand: str) -> str:
        return (f"F/T compensation ({hand})" if len(self.ft) > 1
                else "F/T compensation")

    def grip_folder(self, hand: str) -> str:
        return f"Gripper ({hand})" if self.scene._has_right_gripper else "Gripper"

    def feed_gripper(self, hand: str, *, actual: float, frac_seq: int = 1,
                     motor_ok: bool = True) -> None:
        """Push one encoder frame through the REAL bridge handler for ``hand``."""
        msg = {"open": 4.90, "closed": 0.0, "actual": actual, "cmd": actual,
               "seq": frac_seq, "motor_ok": motor_ok, "kbd_online": True}
        if hand == "left":
            cv._on_gripper_msg(self.state, msg)
        else:
            cv._on_gripper_right_msg(self.state, msg)


# ---------------------------------------------------------------------------
# Tests: build-out
# ---------------------------------------------------------------------------


def test_both_hands_get_the_same_ft_controls():
    rig = _Rig()
    try:
        want = ("Tare wrench", "Compensated wrench", "Compensation algorithm",
                "Payload", "Recompensation (90s)")
        for hand in ("left", "right"):
            folder = rig.ft_folder(hand)
            for label in want:
                _find(rig.server, folder, label)  # raises unless exactly one
        assert len(rig.scene._ft_panels) == 2
        assert [p.hand for p in rig.scene._ft_panels] == ["left", "right"], "left-first order"
        # Each panel is bound to its OWN state object, not a shared one.
        assert rig.scene._ft_panels[0].state is rig.ft["left"]
        assert rig.scene._ft_panels[1].state is rig.ft["right"]
        assert rig.ft["left"] is not rig.ft["right"]
    finally:
        rig.close()


def test_two_sensors_suffix_BOTH_status_rows():
    rig = _Rig()
    try:
        labels = [w for f, w, _h in _handles(rig.server) if f == "Status"]
        assert "F/T L" in labels and "F/T R" in labels, labels
        # Leaving the left unsuffixed would make it read as "the" sensor.
        assert "F/T sensor" not in labels, labels
    finally:
        rig.close()


def test_one_sensor_keeps_the_historical_labels():
    rig = _Rig(mode="1pt", right_ft=False, right_gripper=False)
    try:
        labels = [w for f, w, _h in _handles(rig.server) if f == "Status"]
        assert "F/T sensor" in labels, labels
        assert "F/T L" not in labels and "F/T R" not in labels, labels
        assert "F/T compensation" in _folders(rig.server)
        assert "Gripper" in _folders(rig.server)
        # No right-hand folder/row at all: an empty one reads as broken hardware.
        assert "Gripper (right)" not in _folders(rig.server)
        assert rig.scene._enc_chart_right is None
        assert len(rig.scene._ft_panels) == 1
    finally:
        rig.close()


def test_each_hand_gets_its_own_wrench_charts():
    rig = _Rig()
    try:
        left, right = rig.scene._ft_panels
        charts = [left._force_chart, left._torque_chart, left._raw_force_chart,
                  left._raw_torque_chart, right._force_chart, right._torque_chart,
                  right._raw_force_chart, right._raw_torque_chart]
        assert all(c is not None for c in charts), "a chart was not created"
        assert len({id(c) for c in charts}) == 8, "two hands share a chart handle"
        # And the right gripper's width gets its own trace (separate bridges, so
        # no shared x grid to put two series on).
        assert rig.scene._enc_chart_right is not None
        assert rig.scene._enc_chart_right is not rig.scene._enc_chart
    finally:
        rig.close()


# ---------------------------------------------------------------------------
# Tests: F/T controls are per hand
# ---------------------------------------------------------------------------


def test_tare_button_zeroes_only_its_own_hand():
    rig = _Rig()
    try:
        for hand in ("right", "left"):
            other = "left" if hand == "right" else "right"
            for st in rig.ft.values():
                st.tare_request = False
            _click(_find(rig.server, rig.ft_folder(hand), "Tare wrench"))
            assert rig.ft[hand].tare_request is True, f"{hand} tare did not fire"
            assert rig.ft[other].tare_request is False, f"{hand} tare reached {other}"
    finally:
        rig.close()


def test_raw_only_tare_falls_back_to_the_display_offset():
    # No payload config -> the reader runs the RAW loop, which never consumes
    # tare_request; the tare must zero the DISPLAY offsets instead or the button
    # would look like it worked and do nothing.
    rig = _Rig(comp=False, recomp=False)
    try:
        st = rig.ft["right"]
        st.force = np.array([1.0, -2.0, 3.0])
        st.torque = np.array([0.1, 0.2, -0.3])
        _click(_find(rig.server, rig.ft_folder("right"), "Tare wrench"))
        assert st.tare_request is False, "raw loop would never consume the flag"
        assert np.allclose(st.force_offset, [1.0, -2.0, 3.0]), st.force_offset
        assert np.allclose(st.torque_offset, [0.1, 0.2, -0.3]), st.torque_offset
        assert np.allclose(rig.ft["left"].force_offset, 0.0), "reached the left hand"
    finally:
        rig.close()


def test_compensated_toggle_and_algo_dropdown_are_per_hand():
    rig = _Rig()
    try:
        _set(_find(rig.server, rig.ft_folder("right"), "Compensated wrench"), False)
        assert rig.ft["right"].show_compensated is False
        assert rig.ft["left"].show_compensated is True, "toggle crossed hands"

        _set(_find(rig.server, rig.ft_folder("right"), "Compensation algorithm"), "rnea")
        assert rig.ft["right"].algo_request == "rnea"
        assert rig.ft["left"].algo_request is None, "algo switch crossed hands"
    finally:
        rig.close()


def test_recompensation_arms_only_its_own_hand():
    rig = _Rig()
    try:
        _click(_find(rig.server, rig.ft_folder("right"), "Recompensation (90s)"))
        assert rig.ft["right"].recal_request is True
        assert rig.ft["right"].recal_phase == "countdown"
        assert rig.ft["left"].recal_request is False, "recomp crossed hands"
        assert rig.ft["left"].recal_phase == "idle"
    finally:
        rig.close()


def test_second_recompensation_is_refused_while_one_runs():
    rig = _Rig()
    try:
        _click(_find(rig.server, rig.ft_folder("left"), "Recompensation (90s)"))
        rig.ft["left"].recal_phase = "estimating"
        _click(_find(rig.server, rig.ft_folder("right"), "Recompensation (90s)"))
        assert rig.ft["right"].recal_request is False, "two estimates started at once"
        assert rig.ft["right"].recal_phase == "idle"
        status = _find(rig.server, rig.ft_folder("right"), "Payload")
        assert "LEFT" in status.value, status.value
        # ...and it becomes available again once the left cycle ends.
        rig.ft["left"].recal_phase = "idle"
        _click(_find(rig.server, rig.ft_folder("right"), "Recompensation (90s)"))
        assert rig.ft["right"].recal_request is True
    finally:
        rig.close()


def test_recompensation_refuses_a_sensor_that_is_not_streaming():
    rig = _Rig()
    try:
        rig.ft["right"].connected = False
        _click(_find(rig.server, rig.ft_folder("right"), "Recompensation (90s)"))
        # Arming a dead reader wedges the modal at countdown forever (the reader
        # is the only consumer of recal_request).
        assert rig.ft["right"].recal_request is False
        assert rig.ft["right"].recal_phase == "idle"
        assert "not streaming" in _find(rig.server, rig.ft_folder("right"), "Payload").value
    finally:
        rig.close()


def test_disabled_recompensation_cannot_arm():
    rig = _Rig(recomp=False)
    try:
        for hand in ("left", "right"):
            assert rig.scene._ft_panels[0 if hand == "left" else 1].recomp_enabled is False
            _click(_find(rig.server, rig.ft_folder(hand), "Recompensation (90s)"))
            assert rig.ft[hand].recal_request is False
            assert "no payload config" in _find(rig.server, rig.ft_folder(hand), "Payload").value
    finally:
        rig.close()


# ---------------------------------------------------------------------------
# Tests: the recompensation overlay
# ---------------------------------------------------------------------------


def test_overlay_is_per_hand_and_names_the_hand_to_rotate():
    rig = _Rig()
    try:
        left, right = rig.scene._ft_panels
        _click(_find(rig.server, rig.ft_folder("right"), "Recompensation (90s)"))
        rig.ft["right"].recal_phase = "estimating"
        rig.ft["right"].recal_elapsed = 3.0
        rig.ft["right"].recal_message = "ROTATE the tool slowly"
        rig.scene._update_recomp_overlay()
        assert right.recomp_modal is not None, "no modal opened"
        assert left.recomp_modal is None, "the idle hand opened a modal"
        # The prompt is the operator's ONLY instruction; rotating the wrong tool
        # converges a payload for the sensor that never moved.
        assert "right" in right.recomp_md.content, right.recomp_md.content
        assert "estimating" in right.gui_recomp_status.value
        # ...and the left hand's compact readout is still pumped meanwhile.
        assert left.gui_recomp_status.value == "ready", left.gui_recomp_status.value

        # done -> lingers, then auto-closes.
        rig.ft["right"].recal_phase = "done"
        rig.scene._update_recomp_overlay()
        assert right.recomp_modal is not None, "closed before the operator could read it"
        right.recomp_done_at = time.perf_counter() - 5.0
        rig.scene._update_recomp_overlay()      # flips the phase to idle
        rig.scene._update_recomp_overlay()      # ...and closes on the next tick
        assert rig.ft["right"].recal_phase == "idle"
        assert right.recomp_modal is None, "modal outlived its cycle"
        assert right.recomp_armed_at == 0.0 and right.recomp_done_at == 0.0, "timers not re-armed"
    finally:
        rig.close()


def test_a_stalled_reader_fails_the_cycle_instead_of_wedging():
    rig = _Rig()
    try:
        right = rig.scene._ft_panels[1]
        _click(_find(rig.server, rig.ft_folder("right"), "Recompensation (90s)"))
        rig.scene._update_recomp_overlay()          # arms the clock (render thread)
        assert right.recomp_armed_at > 0.0
        right.recomp_armed_at = time.perf_counter() - 30.0   # reader never advanced
        rig.scene._update_recomp_overlay()
        assert rig.ft["right"].recal_phase == "failed"
        assert "not responding" in rig.ft["right"].recal_message
        assert right.recomp_modal is not None, "no Close button for the operator"
    finally:
        rig.close()


def test_busy_panel_reports_either_hand():
    rig = _Rig()
    try:
        assert rig.scene._recomp_busy_panel() is None
        rig.ft["right"].recal_phase = "estimating"
        assert rig.scene._recomp_busy_panel() is rig.scene._ft_panels[1]
        rig.ft["right"].recal_phase = "idle"
        rig.ft["left"].recal_phase = "countdown"
        assert rig.scene._recomp_busy_panel() is rig.scene._ft_panels[0]
    finally:
        rig.close()


# ---------------------------------------------------------------------------
# Tests: the recording-edge auto-tare
# ---------------------------------------------------------------------------


def test_record_start_and_stop_tare_BOTH_hands():
    rig = _Rig()
    try:
        for st in rig.ft.values():
            st.tare_request = False
        rig.scene._update_recording_overlay({"recording": True, "take_idx": 4})
        assert all(st.tare_request for st in rig.ft.values()), "a hand was not tared at start"
        assert "左右手" in rig.scene._rec_md.content, rig.scene._rec_md.content

        for st in rig.ft.values():
            st.tare_request = False
        rig.scene._update_recording_overlay(
            {"recording": False, "take_idx": 5, "last_saved": "/x/take_004"})
        assert all(st.tare_request for st in rig.ft.values()), "a hand was not tared at stop"
        assert "左右手" in rig.scene._rec_md.content, rig.scene._rec_md.content
    finally:
        rig.close()


def test_the_banner_names_only_the_hands_that_actually_have_a_sensor():
    # A promise about "both hands" when only one sensor came up would hide an
    # untared hand's DC bias in the take.
    rig = _Rig(right_ft=False)
    try:
        rig.scene._update_recording_overlay({"recording": True, "take_idx": 0})
        assert "左手" in rig.scene._rec_md.content
        assert "左右手" not in rig.scene._rec_md.content, rig.scene._rec_md.content
    finally:
        rig.close()
    rig = _Rig(left_ft=False, right_ft=False)
    try:
        rig.scene._update_recording_overlay({"recording": True, "take_idx": 0})
        md = rig.scene._rec_md.content
        assert "tare" not in md, md          # no sensor -> promise no tare
        assert "力觉" not in md, md           # ...and don't claim it is recorded
        rig.scene._update_recording_overlay(
            {"recording": False, "take_idx": 1, "last_saved": "/x/take_000"})
        assert "tare" not in rig.scene._rec_md.content, rig.scene._rec_md.content
    finally:
        rig.close()


def test_the_banner_describes_the_MODE_not_one_hand():
    rig = _Rig(mode="3pt")
    try:
        rig.scene._update_recording_overlay({"recording": True, "take_idx": 0})
        md = rig.scene._rec_md.content
        assert "双手位姿" in md and "胸部" in md, md
    finally:
        rig.close()
    rig = _Rig(mode="1pt", right_ft=False, right_gripper=False)
    try:
        rig.scene._update_recording_overlay({"recording": True, "take_idx": 0})
        md = rig.scene._rec_md.content
        assert "左手位姿" in md and "胸部" not in md, md
    finally:
        rig.close()


# ---------------------------------------------------------------------------
# Tests: the gripper folders
# ---------------------------------------------------------------------------


def test_gripper_controls_drive_only_their_own_hand():
    # The late-binding trap: four callbacks per hand are built in a loop, so a
    # closure over the loop variable makes EVERY button drive the right hand.
    rig = _Rig()
    try:
        _click(_find(rig.server, "Gripper (left)", "Open gripper"))
        assert rig.sent["left"] == [{"target_frac": 1.0}], rig.sent
        assert rig.sent["right"] == [], rig.sent

        _click(_find(rig.server, "Gripper (right)", "Close gripper"))
        assert rig.sent["right"] == [{"target_frac": 0.0}], rig.sent
        assert len(rig.sent["left"]) == 1, rig.sent

        _click(_find(rig.server, "Gripper (right)", "Stop"))
        assert rig.sent["right"][-1] == {"cmd": "stop"}, rig.sent

        # The slider syncs its OWN hand's widget (and only sends on a real drag).
        _set(_find(rig.server, "Gripper (left)", "Target open %"), 40)
        assert rig.sent["left"][-1] == {"target_frac": 0.4}, rig.sent
        assert rig.scene.gui_grip_slider["left"].value == 40
        assert rig.scene.gui_grip_slider["right"].value == 0, "slider crossed hands"
    finally:
        rig.close()


def test_button_sync_writes_its_own_slider_without_resending():
    rig = _Rig()
    try:
        _click(_find(rig.server, "Gripper (right)", "Open gripper"))
        assert rig.scene.gui_grip_slider["right"].value == 100
        assert rig.sent["right"] == [{"target_frac": 1.0}], "the sync double-sent"
        assert rig.scene.gui_grip_slider["left"].value == 0
        assert rig.sent["left"] == []
        assert rig.scene._suppress_slider_send == {"left": False, "right": False}
    finally:
        rig.close()


def test_a_hand_with_no_bridge_gets_readouts_but_no_controls():
    # "device declared, but no send channel" — what --mock does. The readouts must
    # still exist (they are fed by the WS stream) while the controls, which would
    # write to a None callable, must not be built at all.
    global _PORT
    _PORT += 1
    state = cv.CollectionState()
    server = viser.ViserServer(host="127.0.0.1", port=_PORT, label="panel-test")
    try:
        cv.CollectionScene(
            server, state, ik=None, mapper=None, yam_urdf=cv._DEFAULT_YAM,
            cam_offset_xyz=(0.10, 0.0, 0.04), cam_offset_rpy=(4.7124, 1.5708, 0.0),
            cam_fov_deg=58.0, tracking_mode="3pt", display_mode="gripper",
            gripper_send=None, gripper_right_send=None, has_right_gripper=True,
        )
        labels = [w for f, w, _h in _handles(server) if f == "Gripper (right)"]
        assert "Open fraction" in labels and "Encoder (rad)" in labels, labels
        assert "Open gripper" not in labels and "Target open %" not in labels, labels
    finally:
        server.stop()


def test_right_gripper_readouts_show_the_RIGHT_hands_numbers():
    rig = _Rig()
    try:
        rig.feed_gripper("left", actual=0.98)     # ~20% of the 4.90 rad span
        rig.feed_gripper("right", actual=3.92)    # ~80%
        rig.scene.update()
        assert rig.scene.gui_grip_rad["left"].value == "+0.980"
        assert rig.scene.gui_grip_rad["right"].value == "+3.920"
        assert rig.scene.gui_grip_frac["left"].value.startswith("20%")
        assert rig.scene.gui_grip_frac["right"].value.startswith("80%")
        assert "80% open" in rig.scene.gui_gripper_right.value
        # The right hand's endpoints are kept per hand (its absolute encoder
        # re-references on its own power-cycle).
        with rig.state.lock:
            assert rig.state.gripper_open_rad_right == 4.90
            assert rig.state.gripper_actual_rad_right == 3.92
            assert rig.state.gripper_actual_rad == 0.98
    finally:
        rig.close()


def test_right_motor_fault_is_as_loud_as_the_left():
    rig = _Rig()
    try:
        rig.feed_gripper("right", actual=2.0, motor_ok=False)
        rig.scene.update()
        assert "MOTOR FAULT" in rig.scene.gui_gripper_right.value, rig.scene.gui_gripper_right.value
    finally:
        rig.close()


def test_right_gripper_chart_advances_on_its_own_stream():
    rig = _Rig()
    try:
        for i in range(4):
            rig.feed_gripper("right", actual=4.9 * i / 3.0, frac_seq=i + 1)
            time.sleep(0.002)
        rig.scene.update_chart()
        n_r = rig.scene._last_chart_n_right
        assert n_r == 4, n_r
        assert rig.scene._last_chart_n == -1, "the LEFT chart advanced on right frames"
    finally:
        rig.close()


# ---------------------------------------------------------------------------
# Tests: wiring that cannot be exercised without the sensors
# ---------------------------------------------------------------------------


def test_right_recompensation_writes_the_RIGHT_configs():
    """A recompensation must not overwrite the other hand's payload.

    ``SensorReader._recalibrate`` redirects the estimator's output to a temp file
    and then copies it onto ``self.compensator_config``, and
    ``_run_hardware_compensated`` syncs the RNEA payload from that same file — so
    which files a right-hand recompensation rewrites is decided ENTIRELY by what
    resolve_right_ft_configs returns. Handing it the shared
    payload_compensator.json would silently repoint the LEFT hand's compensation
    at the right tool's mass."""
    comp, est, rnea = cv.resolve_right_ft_configs(bft)
    assert comp is not None and os.path.basename(comp) == "payload_compensator_right.json", comp
    assert rnea is not None and os.path.basename(rnea) == "rnea_compensator_right.json", rnea
    assert comp != bft._DEFAULT_COMP_CONFIG, "the right hand would overwrite the left's payload"
    assert rnea != bft._DEFAULT_RNEA_CONFIG, "the right hand would overwrite the left's RNEA"
    # The estimator config is shared on purpose (priors, not a result) and is safe
    # to share only because _recalibrate redirects its output.
    assert est == bft._DEFAULT_EST_CONFIG, est
    recal = inspect.getsource(bft.SensorReader._recalibrate)
    assert 'ec["output_config_file_name"] = out_tmp' in recal, \
        "the estimator writes to its config's own path -> both hands would share it"
    assert "shutil.copy(out_tmp, self.compensator_config)" in recal
    loop = inspect.getsource(bft.SensorReader._run_hardware_compensated)
    assert "_sync_rnea_payload(self.rnea_config, self.compensator_config)" in loop
    # --no-compensate / a missing payload file -> RAW loop, and then NO estimator:
    # the raw loop never consumes recal_request, so an armed button would wedge.
    assert cv.resolve_right_ft_configs(bft, compensate=False) == (None, None, None)
    # main() must actually use the seam, and gate the button on both halves.
    main_src = inspect.getsource(cv.main)
    right = main_src[main_src.index('ft_right_serial", ""'):]
    assert "resolve_right_ft_configs(" in right, right[:400]
    assert "estimator_config=est_cfg_r" in right
    assert "recomp_enabled_right = bool(est_cfg_r) and bool(comp_cfg_r)" in right


def test_the_render_loop_throttles_for_EITHER_hand():
    main_src = inspect.getsource(cv.main)
    assert "recomp_busy = scene._recomp_busy_panel() is not None" in main_src, \
        "a right-hand estimate would run against a 60 Hz render loop"
    assert "elif recomp_enabled or recomp_enabled_right:" in main_src, \
        "Pause during a right-hand recomp would freeze its overlay"


def test_the_right_gripper_reader_is_shared_not_duplicated():
    """One WS connection per bridge: the control channel and the recorded stream
    must be the SAME reader, else the panel drives a socket whose frames nobody
    reads (and the bridge sees two subscribers)."""
    main_src = inspect.getsource(cv.main)
    assert main_src.count('_on_gripper_right_msg(state, m)') == 1, \
        "two readers on the right gripper bridge"
    assert "readers.append(gripper_reader_right)" in main_src


# ---------------------------------------------------------------------------
# End-to-end: the REAL main() with the launcher's own arguments
# ---------------------------------------------------------------------------


def zz_test_main_wires_both_sensors_end_to_end():
    """Run the actual ``main()`` with two Botas faked at the discovery layer.

    The scene tests above build CollectionScene directly, so they cannot catch a
    ``main()`` that resolves the configs correctly and then forgets to hand
    ``ft_state_right`` / ``recomp_enabled_right`` to the scene — the panel would
    silently fall back to one hand. This drives the real entry point end to end:
    only ``find_bota_port`` (no sensors attached) and ``SensorReader`` (no
    bota_driver) are stubbed; the config resolution, the SensorReader kwargs, the
    scene construction and the GUI build are all the production code.

    Runs LAST and in the MAIN thread: ``main()`` installs SIGINT/SIGTERM handlers,
    which Python only allows from the main thread, so it is stopped by delivering
    a real SIGINT rather than by a flag. Named ``zz_`` so the collector orders it
    after every other test.
    """
    import signal
    import threading

    global _PORT
    _PORT += 1
    left_dev = "/dev/serial/by-id/usb-Bota_Systems_x_3875355D3532-if00"
    right_dev = "/dev/serial/by-id/usb-Bota_Systems_x_347237663335-if00"
    made = []
    captured = {}

    class _StubReader:
        def __init__(self, state, config_path, **kw):
            self.state = state
            self.config_path = config_path
            self.kw = kw
            self._tare = None
            self._algo_prov = None
            made.append(self)

        def start(self):
            # The panel's header/arm gate keys off `connected`; a real reader sets
            # it once the driver activates.
            with self.state.lock:
                self.state.connected = True

        def stop(self):
            pass

        def join(self, timeout=None):
            pass

    orig_find, orig_reader, orig_scene = bft.find_bota_port, bft.SensorReader, cv.CollectionScene

    class _Scene(orig_scene):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            captured["scene"] = self

    bft.find_bota_port = lambda serial=None: (
        left_dev if serial and "3875355D3532" in serial else
        right_dev if serial and "347237663335" in serial else None)
    bft.SensorReader = _StubReader
    cv.CollectionScene = _Scene
    argv = sys.argv
    # The exact right-hand arguments run_collection_viz.sh passes for MODE=3pt.
    sys.argv = ["collection_viz.py", "--port", str(_PORT), "--no-ik",
                "--tracking-mode", "3pt", "--display-mode", "gripper", "--raw-frame",
                "--tracker-url", "ws://127.0.0.1:9971",
                "--camera-url", "ws://127.0.0.1:9972",
                "--gripper-url", "ws://127.0.0.1:9973",
                "--gripper-right-url", "ws://127.0.0.1:9978",
                "--ft-left-serial", "3875355D3532",
                "--ft-right-serial", "347237663335"]
    stopper = threading.Timer(6.0, lambda: os.kill(os.getpid(), signal.SIGINT))
    stopper.daemon = True
    try:
        stopper.start()
        try:
            cv.main()
        except KeyboardInterrupt:
            pass                    # SIGINT landed before main() installed its handler
    finally:
        stopper.cancel()
        sys.argv = argv
        bft.find_bota_port, bft.SensorReader = orig_find, orig_reader
        cv.CollectionScene = orig_scene

    scene = captured.get("scene")
    assert scene is not None, "main() never built a scene"
    # Two readers, each on its OWN device and its OWN per-side configs.
    assert len(made) == 2, f"{len(made)} readers built"
    left, right = made
    assert left.config_path != right.config_path, "both readers opened one device"
    assert os.path.basename(right.kw["compensator_config"]) == "payload_compensator_right.json"
    assert os.path.basename(left.kw["compensator_config"]) == "payload_compensator.json"
    assert right.kw["estimator_config"] == bft._DEFAULT_EST_CONFIG, \
        "no estimator -> the right Recompensation button would be dead"
    assert os.path.basename(right.kw["rnea_config"]) == "rnea_compensator_right.json"
    # ...and the panel actually got them.
    assert [p.hand for p in scene._ft_panels] == ["left", "right"], \
        [p.hand for p in scene._ft_panels]
    assert scene._ft_panels[1].state is right.state, "the right panel reads the wrong sensor"
    assert scene._ft_panels[1].recomp_enabled is True, "right Recompensation not enabled"
    assert scene._ft_panels[1].rnea_available is True, "right RNEA dropdown not offered"
    assert scene._gripper_right_send is not None, "right gripper controls have no channel"
    assert "Gripper (right)" in _folders(scene.server)
    assert "F/T compensation (right)" in _folders(scene.server)


def _run() -> int:
    tests = [(n, f) for n, f in sorted(globals().items())
             if (n.startswith("test_") or n.startswith("zz_test_")) and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  ok   {name}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  FAIL {name}: {type(exc).__name__}: {exc}")
    if failed:
        print(f"PANEL FAIL  {failed}/{len(tests)} failed")
        return 1
    print(f"PANEL OK  {len(tests)} passed")
    return 0


if __name__ == "__main__":
    sys.exit(_run())
