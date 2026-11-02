#!/usr/bin/env python
"""Tests for the camera-preview switch — the collection viz's biggest CPU knob.

A live frustum image is NOT a cheap read-only overlay. Per frame, per camera, on the
render thread, it costs a JPEG *de*code (PIL) plus a JPEG *re*-encode, because viser's
``CameraFrustumHandle.image`` setter re-encodes unconditionally with no dirty check —
so the pixels the camera bridge already encoded get decoded and encoded right back.
Measured on the 3pt rig with two 640x480@30Hz D405s: ``collection_viz`` 50.9% -> 26.0%
of one core when the preview is off. It is paid even with no browser tab open, because
the re-encode happens on assignment rather than on send.

These tests pin the switch and the four ways it goes wrong quietly:

  1. **A half-wired switch.** The GUI checkbox used to gate only the render loop, so the
     reader threads kept base64-decoding every frame regardless. The flag has to reach
     ``CollectionState.preview_images`` for the readers to see it.
  2. **A dark status row.** Skipping a frame's pixels must NOT skip its liveness
     bookkeeping — the operator still needs ``camera ● online 30Hz``, and a bridge that
     dies mid-session still has to flip to offline. This is why a corrupt payload and a
     switched-off preview are different cases (``_DECODE_FAILED`` vs ``None``).
  3. **A stale repaint.** Frames held while the preview is off would repaint the instant
     it came back on — arbitrarily old if the bridge stopped meanwhile.
  4. **A half-hidden scene.** The RIGHT frustum's ``.image`` was gated on the checkbox
     but its ``.visible`` was never assigned at all, so unchecking the box hid the left
     camera and left the right one floating, frozen on its last frame. On a 3pt rig that
     reads as "the right camera is stuck", which is the opposite of the truth.

And the load-bearing safety claim, pinned structurally: turning the preview off cannot
change a recorded byte. The recorder is a SEPARATE process with its own subscription to
the same camera bridges, and it writes their JPEG bytes verbatim.

Hardware-free: the WS handlers are module-level functions over a plain dataclass, and
display is gripper-only so no mink IK / URDF solve is needed.

Run:  .venv/bin/python viz/test_camera_preview.py
Prints ``PREVIEW OK  N passed`` on success, ``PREVIEW FAIL ...`` (exit 1) otherwise.
"""
from __future__ import annotations

import base64
import io
import os
import re
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_ROOT, "force_torque"))

import viser  # noqa: E402

import collection_viz as cv  # noqa: E402

# The viser-handle helpers are imported rather than copied: they are generic page
# introspection (walk the container/input registries, fire the real update_cb), not
# domain logic, and a second copy would drift. Importing a sibling test module is safe
# here because these files are hand-rolled scripts with `if __name__ == "__main__"`
# runners — there is no pytest collection to double-run anything.
from test_dual_hand_panel import _find, _folders, _set  # noqa: E402

# Clear of the live rig (8080, 8772-8779) and of the other smoke tests (8096-8101,
# 8150+).
_PORT = 8210

_CAMERA_FOLDER = "Camera"
_CAMERA_LABEL = "Show camera view (~24% of a core)"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _jpeg_b64(w: int = 64, h: int = 48) -> str:
    """A real, decodable JPEG as the bridge sends it: base64 of JPEG bytes."""
    from PIL import Image

    rgb = np.zeros((h, w, 3), np.uint8)
    rgb[:, :, 0] = np.linspace(0, 255, w, dtype=np.uint8)[None, :]
    bio = io.BytesIO()
    Image.fromarray(rgb, "RGB").save(bio, "JPEG", quality=70)
    return base64.b64encode(bio.getvalue()).decode("ascii")


def _frame(seq: int, *, b64: str | None = None, w: int = 64, h: int = 48) -> dict:
    return {"jpeg_b64": _jpeg_b64(w, h) if b64 is None else b64,
            "w": w, "h": h, "seq": seq}


class _Rig:
    """A real CollectionScene in gripper display mode, optionally with a right camera
    so the second frustum actually exists."""

    def __init__(self, *, right_camera: bool = False, show_camera_view: bool = True,
                 camera_preview_hz: float = 0.0):
        global _PORT
        _PORT += 1
        self.state = cv.CollectionState()
        self.server = viser.ViserServer(host="127.0.0.1", port=_PORT, label="preview-test")
        self.scene = cv.CollectionScene(
            self.server, self.state, ik=None, mapper=None, yam_urdf=cv._DEFAULT_YAM,
            cam_offset_xyz=(0.10, 0.0, 0.04), cam_offset_rpy=(4.7124, 1.5708, 0.0),
            cam_fov_deg=58.0, tracking_mode="3pt", display_mode="gripper",
            has_right_camera=right_camera,
            show_camera_view=show_camera_view,
            # 0 = repaint every frame, so the tests ABOVE (which assert on a single
            # update() painting a single frame) are unaffected by the throttle's
            # default. The throttle has its own tests below and opts in explicitly.
            camera_preview_hz=camera_preview_hz,
        )

    def close(self) -> None:
        self.server.stop()

    @property
    def checkbox(self):
        return _find(self.server, _CAMERA_FOLDER, _CAMERA_LABEL)


# ---------------------------------------------------------------------------
# The switch itself
# ---------------------------------------------------------------------------


def test_the_camera_folder_offers_exactly_one_preview_switch():
    rig = _Rig()
    try:
        assert _CAMERA_FOLDER in _folders(rig.server), _folders(rig.server)
        # _find asserts there is exactly one handle with this folder/label pair, so a
        # duplicated control (two switches disagreeing about one flag) fails here.
        assert rig.checkbox.value is True
    finally:
        rig.close()


def test_the_switch_label_states_its_cost():
    """The whole point of the label is discoverability: nothing about a 'Show camera
    view' checkbox suggests it is worth ~24% of a core, so the number is IN the label.
    Pinned because a well-meaning tidy-up of the wording would silently un-teach it."""
    rig = _Rig()
    try:
        label = _CAMERA_LABEL
        assert re.search(r"\d+\s*%", label), f"no cost in the label: {label!r}"
        assert rig.checkbox is not None  # the label is the real one on the page
    finally:
        rig.close()


def test_toggling_the_switch_reaches_the_reader_threads():
    """The bug this exists to prevent: gating only the render loop. The WS readers run
    in their own threads and consult state.preview_images, so the GUI callback must
    write THAT, not just its own widget value."""
    rig = _Rig()
    try:
        assert rig.state.preview_images is True
        _set(rig.checkbox, False)
        assert rig.state.preview_images is False, "the readers never heard about it"
        _set(rig.checkbox, True)
        assert rig.state.preview_images is True
    finally:
        rig.close()


# ---------------------------------------------------------------------------
# The reader-thread half: skip the decode, keep the liveness
# ---------------------------------------------------------------------------


def test_preview_on_decodes_and_stores_the_frame():
    st = cv.CollectionState()
    st.preview_images = True
    cv._on_camera_msg(st, _frame(1))
    assert st.cam_jpeg is not None and st.cam_jpeg[:2] == b"\xff\xd8", "not a JPEG"
    assert (st.cam_w, st.cam_h, st.cam_seq) == (64, 48, 1)
    assert st.cam_connected is True and st.cam_recv_t > 0.0


def test_preview_off_skips_the_decode_but_keeps_the_status_row_alive():
    """The saving AND the thing that must not break with it. cam_jpeg stays None (no
    base64 decode, nothing stashed) while seq/connected/recv_t still advance — else a
    perfectly healthy bridge would read '○ offline' the moment the box is unchecked."""
    for handler, jpeg_attr, seq_attr, conn_attr, t_attr in (
            (cv._on_camera_msg, "cam_jpeg", "cam_seq", "cam_connected", "cam_recv_t"),
            (cv._on_camera_right_msg, "cam_jpeg_right", "cam_seq_right",
             "cam_connected_right", "cam_recv_t_right")):
        st = cv.CollectionState()
        st.preview_images = False
        handler(st, _frame(7))
        assert getattr(st, jpeg_attr) is None, f"{jpeg_attr}: decoded anyway"
        assert getattr(st, seq_attr) == 7, f"{seq_attr}: liveness lost"
        assert getattr(st, conn_attr) is True, f"{conn_attr}: reads as never-seen"
        assert getattr(st, t_attr) > 0.0, f"{t_attr}: status row would go offline"


def test_a_corrupt_payload_drops_the_frame_INCLUDING_its_liveness():
    """Why _DECODE_FAILED is a distinct sentinel rather than reusing None: a broken
    stream must NOT be counted as a live frame. If the two collapsed, a bridge emitting
    garbage would show as '● online' forever."""
    st = cv.CollectionState()
    st.preview_images = True
    cv._on_camera_msg(st, _frame(3, b64="!!! not base64 !!!"))
    assert st.cam_jpeg is None
    assert st.cam_seq == 0, "a corrupt frame was counted as live"
    assert st.cam_connected is False
    assert st.cam_recv_t == 0.0


def test_switching_off_drops_frames_already_held():
    """Otherwise the next tick after switching back ON repaints whatever was stashed —
    arbitrarily stale if the camera stopped in between."""
    rig = _Rig(right_camera=True)
    try:
        cv._on_camera_msg(rig.state, _frame(1))
        cv._on_camera_right_msg(rig.state, _frame(1))
        assert rig.state.cam_jpeg is not None and rig.state.cam_jpeg_right is not None
        _set(rig.checkbox, False)
        assert rig.state.cam_jpeg is None, "left frame still pinned"
        assert rig.state.cam_jpeg_right is None, "right frame still pinned"
    finally:
        rig.close()


# ---------------------------------------------------------------------------
# The render-loop half
# ---------------------------------------------------------------------------


def test_update_does_not_touch_the_frustum_image_while_the_preview_is_off():
    """Belt to the reader's braces: even if a frame reached cam_jpeg some other way,
    update() must not run the decode + viser re-encode."""
    rig = _Rig()
    try:
        _set(rig.checkbox, False)
        # Plant a frame directly, bypassing the reader's own gate.
        cv._on_camera_msg(rig.state, _frame(11))
        rig.state.cam_jpeg = base64.b64decode(_jpeg_b64())
        before = rig.scene._last_cam_seq
        rig.scene.update()
        assert rig.scene._last_cam_seq == before, "the frame was decoded + re-encoded"
        assert rig.scene._frustum.image is None, "an image was pushed to the frustum"
    finally:
        rig.close()


def test_update_paints_the_frustum_while_the_preview_is_on():
    """The other direction, so 'off' passing cannot be a scene that never paints."""
    rig = _Rig()
    try:
        cv._on_camera_msg(rig.state, _frame(11))
        rig.scene.update()
        assert rig.scene._last_cam_seq == 11
        img = rig.scene._frustum.image
        assert img is not None and img.shape == (48, 64, 3), getattr(img, "shape", None)
    finally:
        rig.close()


def test_BOTH_frustums_follow_the_switch():
    """The half-hidden-scene bug. The right frustum's `.visible` was never assigned, so
    unchecking the box hid the left camera and left the right one frozen in the scene."""
    rig = _Rig(right_camera=True)
    try:
        assert rig.scene._frustum_right is not None, "test needs the right frustum"
        rig.scene.update()
        assert rig.scene._frustum.visible is True
        assert rig.scene._frustum_right.visible is True
        assert rig.scene._cam_mount.visible is True

        _set(rig.checkbox, False)
        rig.scene.update()
        assert rig.scene._frustum.visible is False, "left frustum still shown"
        assert rig.scene._frustum_right.visible is False, "RIGHT frustum still shown"
        assert rig.scene._cam_mount.visible is False, "camera mount still shown"

        _set(rig.checkbox, True)
        rig.scene.update()
        assert rig.scene._frustum_right.visible is True, "right frustum never came back"
    finally:
        rig.close()


def test_visibility_is_assigned_only_when_it_CHANGES():
    """A small saving, asserted for what it actually is. viser's props_setattr already
    drops an unchanged value before the wire, so this never removed wire traffic — it
    removes the per-assignment cast+compare that runs first, three times per render tick.
    The test that matters is that the memo cannot SWALLOW a real toggle."""
    rig = _Rig(right_camera=True)
    try:
        nodes = (rig.scene._frustum, rig.scene._frustum_right, rig.scene._cam_mount)
        seen: list[object] = []
        classes = {type(n) for n in nodes}
        # Save exactly what each class had in its OWN __dict__, so the restore cannot
        # delete a __setattr__ viser really defines there (that would corrupt the class
        # for the rest of the process) nor leave a shim behind if it inherited one.
        saved = {c: ("__setattr__" in c.__dict__, c.__dict__.get("__setattr__"))
                 for c in classes}
        real = {c: c.__setattr__ for c in classes}

        def _make(cls):
            def counting(self, name, value):
                if name == "visible" and any(self is n for n in nodes):
                    seen.append(value)
                real[cls](self, name, value)
            return counting

        for c in classes:
            c.__setattr__ = _make(c)
        try:
            rig.scene.update()          # first tick: _last_show_cam is None -> writes
            assert len(seen) == 3, f"first tick should push all three: {seen}"
            seen.clear()
            for _ in range(5):          # steady state: nothing changed
                rig.scene.update()
            assert seen == [], f"redundant visible= writes: {seen}"

            _set(rig.checkbox, False)   # a REAL change must still get through
            rig.scene.update()
            assert seen == [False, False, False], seen
        finally:
            for c, (had_own, prev) in saved.items():
                if had_own:
                    c.__setattr__ = prev
                else:
                    del c.__setattr__
    finally:
        rig.close()


# ---------------------------------------------------------------------------
# Repaint throttle: --camera-preview-hz
# ---------------------------------------------------------------------------


def test_the_first_frame_of_each_camera_paints_immediately():
    """Deadlines start at -inf on purpose. An operator waiting for the first picture
    must not have to wait out a period — on a 1 Hz preview that reads as a dead camera."""
    rig = _Rig(right_camera=True, camera_preview_hz=1.0)
    try:
        cv._on_camera_msg(rig.state, _frame(1))
        cv._on_camera_right_msg(rig.state, _frame(1))
        rig.scene.update()
        assert rig.scene._frustum.image is not None, "left camera waited out a period"
        assert rig.scene._frustum_right.image is not None, "right camera waited out a period"
    finally:
        rig.close()


def test_frames_arriving_inside_a_period_are_NOT_repainted():
    """The saving. The bridges keep publishing at 30 Hz and the reader keeps taking every
    frame; what the throttle drops is the decode + re-encode, which is ~15x the cost of
    receiving one."""
    rig = _Rig(camera_preview_hz=1.0)   # 1 s period: nothing else can paint this test
    try:
        cv._on_camera_msg(rig.state, _frame(1))
        rig.scene.update()
        painted_first = rig.scene._last_cam_seq
        assert painted_first == 1
        # 20 more frames, all well inside the 1 s window.
        for seq in range(2, 22):
            cv._on_camera_msg(rig.state, _frame(seq))
            rig.scene.update()
        assert rig.scene._last_cam_seq == 1, \
            f"repainted inside the period (seq now {rig.scene._last_cam_seq})"
        # ...and the LATEST frame is still sitting in state, so the next eligible paint
        # shows current pixels rather than the stale one it skipped to.
        assert rig.state.cam_seq == 21
    finally:
        rig.close()


def test_the_next_frame_after_the_period_expires_IS_repainted():
    rig = _Rig(camera_preview_hz=50.0)   # 20 ms period, cheap to wait out
    try:
        cv._on_camera_msg(rig.state, _frame(1))
        rig.scene.update()
        assert rig.scene._last_cam_seq == 1
        time.sleep(0.03)
        cv._on_camera_msg(rig.state, _frame(2))
        rig.scene.update()
        assert rig.scene._last_cam_seq == 2, "the throttle never reopened"
    finally:
        rig.close()


def test_the_two_cameras_throttle_INDEPENDENTLY():
    """One shared deadline would let whichever stream's frame arrived first inside a
    window win it and push the deadline out for BOTH — the other camera's preview would
    then update at half the rate, or stall entirely if the first one is faster."""
    rig = _Rig(right_camera=True, camera_preview_hz=1.0)
    try:
        # LEFT paints first and claims its own window.
        cv._on_camera_msg(rig.state, _frame(1))
        rig.scene.update()
        assert rig.scene._last_cam_seq == 1
        assert rig.scene._last_cam_seq_right == -1, "right painted without a frame"

        # RIGHT's first frame arrives inside the window LEFT just claimed. With ONE
        # shared deadline it would be dropped; with per-camera deadlines it paints.
        cv._on_camera_right_msg(rig.state, _frame(1))
        rig.scene.update()
        assert rig.scene._last_cam_seq_right == 1, "the right camera shared LEFT's deadline"

        # ...and each camera's paint must advance its OWN deadline, which is the half a
        # one-directional test misses: writing the left's timestamp from the right's
        # paint leaves the right permanently unthrottled (and drags the left's window
        # forward on every right frame), yet still passes everything above.
        cv._on_camera_msg(rig.state, _frame(2))
        cv._on_camera_right_msg(rig.state, _frame(2))
        rig.scene.update()
        assert rig.scene._last_cam_seq == 1, "LEFT repainted inside its own window"
        assert rig.scene._last_cam_seq_right == 1, \
            "RIGHT repainted inside its own window — its paint did not advance its own deadline"
    finally:
        rig.close()


def test_preview_hz_0_keeps_the_historical_every_frame_behaviour():
    rig = _Rig(camera_preview_hz=0.0)
    try:
        for seq in (1, 2, 3, 4, 5):
            cv._on_camera_msg(rig.state, _frame(seq))
            rig.scene.update()
            assert rig.scene._last_cam_seq == seq, f"seq {seq} was throttled at hz=0"
    finally:
        rig.close()


def test_the_throttle_never_touches_the_bridges_publish_rate():
    """The one confusion that WOULD cost data: CAMERA_PUBLISH_HZ is the training
    dataset's master timeline and must stay a uniform integer divisor of CAMERA_FPS.
    CAMERA_PREVIEW_HZ is a viz-side repaint deadline. They must not be wired together."""
    sh = open(os.path.join(_ROOT, "run_collection_viz.sh")).read()
    assert re.search(r'CAMERA_PREVIEW_HZ="\$\{CAMERA_PREVIEW_HZ:-10\}"', sh)
    # No realsense_bridge spawn may take CAMERA_PREVIEW_HZ / RENDER_HZ ...
    for m in re.finditer(r"realsense_bridge\.py", sh):
        window = sh[m.start():m.start() + 900]
        for bad in ("CAMERA_PREVIEW_HZ", "RENDER_HZ"):
            assert bad not in window, f"a bridge spawn reads {bad} — that is the timeline"
    # ...and --publish-hz must still come from CAMERA_PUBLISH_HZ alone.
    assert re.search(r'--publish-hz "\$\{CAMERA_PUBLISH_HZ:-30\}"', sh), \
        "the bridges' publish rate is no longer CAMERA_PUBLISH_HZ"
    # ...and the recorder's own camera rate stays CAMERA_FPS, not a preview knob.
    assert re.search(r'--camera-fps "\$\{CAMERA_FPS:-30\}"', sh), \
        "the recorder's camera fps is no longer CAMERA_FPS"


def test_render_hz_default_is_30_and_the_launcher_exposes_it():
    src = open(os.path.join(_HERE, "collection_viz.py")).read()
    assert re.search(r'"--render-hz", type=float, default=30\.0', src), \
        "--render-hz default moved; the tick-budget reasoning in its comment is stale"
    sh = open(os.path.join(_ROOT, "run_collection_viz.sh")).read()
    assert re.search(r'RENDER_HZ="\$\{RENDER_HZ:-30\}"', sh)
    assert "--render-hz" in sh, "the launcher does not pass --render-hz across the sg re-exec"
    assert "would stop the render loop entirely" in sh, "RENDER_HZ=0 is not refused"


# ---------------------------------------------------------------------------
# Boot state: --no-camera-view / CAMERA_VIEW=0
# ---------------------------------------------------------------------------


def test_booting_with_the_preview_off_also_silences_the_readers():
    """The checkbox's on_update only fires on operator clicks, never at construction.
    So a --no-camera-view launch that set the widget's initial_value alone would keep
    decoding every frame until somebody happened to toggle it twice."""
    rig = _Rig(show_camera_view=False)
    try:
        assert rig.state.preview_images is False, "readers still decoding after boot-off"
        assert rig.checkbox.value is False, "the widget disagrees with the state"
        cv._on_camera_msg(rig.state, _frame(1))
        assert rig.state.cam_jpeg is None
        # ...and the nodes are actually hidden on the first tick, even though viser
        # creates them visible (this is what _last_show_cam=None buys).
        rig.scene.update()
        assert rig.scene._frustum.visible is False
    finally:
        rig.close()


def test_boot_off_can_still_be_switched_back_on_at_runtime():
    """--no-camera-view picks the state you boot into, not a permanent lock — the
    operator has to be able to look at a camera mid-session to diagnose one."""
    rig = _Rig(show_camera_view=False)
    try:
        _set(rig.checkbox, True)
        assert rig.state.preview_images is True
        cv._on_camera_msg(rig.state, _frame(4))
        rig.scene.update()
        assert rig.scene._frustum.visible is True
        assert rig.scene._frustum.image is not None
    finally:
        rig.close()


def test_the_cli_flag_and_env_knob_are_wired():
    src = open(os.path.join(_HERE, "collection_viz.py")).read()
    assert '"--no-camera-view"' in src, "no CLI flag"
    assert 'os.environ.get("CAMERA_VIEW"' in src, "CAMERA_VIEW env default missing"
    assert 'show_camera_view=not getattr(args, "no_camera_view"' in src, \
        "the flag never reaches CollectionScene"


def test_the_launcher_passes_the_flag_only_when_CAMERA_VIEW_is_0():
    """collection_viz has its own CAMERA_VIEW env default, but the launcher re-execs
    under `sg` and is not guaranteed to carry the caller's environment — a silently
    ignored CAMERA_VIEW=0 looks exactly like 'the knob does nothing'."""
    sh = open(os.path.join(_ROOT, "run_collection_viz.sh")).read()
    assert re.search(r'CAMERA_VIEW="\$\{CAMERA_VIEW:-1\}"', sh), "no CAMERA_VIEW default"
    assert re.search(r'\[\s*"\$CAMERA_VIEW"\s*=\s*"0"\s*\]\s*&&\s*VIZ_MODE_ARGS\+=\('
                     r'--no-camera-view\)', sh), "the launcher never passes the flag"
    # Refuse a bad value rather than silently treating it as "shown".
    assert 'bad CAMERA_VIEW=' in sh, "CAMERA_VIEW is not validated"


# ---------------------------------------------------------------------------
# The safety claim: recording cannot be affected
# ---------------------------------------------------------------------------


def test_the_recorder_never_consults_the_preview_switch():
    """This is what makes the switch safe to flip mid-session, and it is a claim about
    a DIFFERENT process, so it is pinned structurally rather than trusted."""
    import ast

    path = os.path.join(_ROOT, "recording", "recorder.py")
    rec = open(path).read()
    assert "preview_images" not in rec, "the recorder reads the viz's preview flag"
    # IMPORTS, not a text search: recorder.py's module docstring legitimately names
    # collection_viz to explain why it is a separate process in the first place, so
    # grepping the source would fail on the very comment that documents the split.
    imported = set()
    for node in ast.walk(ast.parse(rec, filename=path)):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert "collection_viz" not in imported, f"the recorder imports the viz: {imported}"
    # It keeps the bridge's bytes and writes them straight out — no decode, so nothing
    # the viz does to its own copy can reach the take.
    assert "verbatim from the bridge" in rec
    assert re.search(r"for i, jpeg in enumerate\(cam\w*\.jpegs\):", rec), \
        "the verbatim-write loop moved; re-verify the no-decode claim"


def test_the_camera_bridges_are_untouched_by_the_switch():
    """The switch is viz-side only: the bridges keep encoding and publishing, because
    the recorder needs those frames. Turning the preview off must not be mistaken for
    a way to stop the cameras (that would silently record nothing)."""
    sh = open(os.path.join(_ROOT, "run_collection_viz.sh")).read()
    # The bridge spawn block must not be gated on CAMERA_VIEW.
    for m in re.finditer(r"realsense_bridge\.py", sh):
        window = sh[max(0, m.start() - 900):m.start()]
        assert "CAMERA_VIEW" not in window, \
            "a realsense_bridge spawn is gated on CAMERA_VIEW — recording would break"


def _run() -> int:
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  ok   {name}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  FAIL {name}: {type(exc).__name__}: {exc}")
    if failed:
        print(f"PREVIEW FAIL  {failed}/{len(tests)} failed")
        return 1
    print(f"PREVIEW OK  {len(tests)} passed")
    return 0


if __name__ == "__main__":
    sys.exit(_run())
