#!/usr/bin/env python3
"""Shared viser building blocks for the two OFFLINE replay tools.

Two separate, independent scripts sit on top of this module (neither touches the
live data-collection pipeline — they only READ already-committed files):

  * ``replay/replay_raw_viz.py``   — replays a RAW recorded take
    (``data/recordings/session_*/take_*``): the G1 driven by the recorded VIVE
    pose through the same mink IK, gripper jaws from the recorded encoder, the
    recorded D405 JPEGs on the follow-camera, F/T as arrows + charts. A draggable
    timeline scrubs to any timestamp.
  * ``replay/replay_zarr_viz.py``  — replays a CONVERTED ``.zarr.zip`` UMI
    ReplayBuffer, one selectable episode at a time, to CONFIRM the conversion kept
    the information (TCP path, the stored 224x224 RGB, resampled F/T + gripper
    width) and didn't introduce time-misalignment or weird motion.

This module deliberately holds only the pieces BOTH share so the two tools stay
small and obviously independent:

  * :class:`Timeline` — the playback controller (time slider + play/pause + speed
    + step + loop) and its render-loop thread. Crucially there is NO live IK in
    the loop: every per-frame quantity is PRE-COMPUTED into arrays at load, so the
    render callback only sets viser scene properties from cached numpy. That means
    none of the live-viz threading invariants (render-thread-owns-IK etc.) apply
    here — the loop is pure presentation.
  * :func:`decode_jpeg`, :class:`WrenchArrows`, :func:`make_xyz_chart` /
    :func:`set_full_xyz` — small scene/GUI helpers mirrored from
    ``viz/collection_viz.py`` so the look matches the live tool.

Dependencies: viser + numpy + Pillow only (all already in ``.venv``).
"""
from __future__ import annotations

import io
import threading
import time
from typing import Callable

import numpy as np
import viser
from viser import transforms as vtf
from viser import uplot

# Match the live viz palette (viz/collection_viz.py:_AXIS_COLORS).
AXIS_COLORS = ("#dc3232", "#28b446", "#3c6ef0")


# ---------------------------------------------------------------------------
# Image
# ---------------------------------------------------------------------------
def decode_jpeg(jpeg: bytes) -> np.ndarray | None:
    """Decode JPEG bytes -> HxWx3 uint8 RGB (None on failure). Mirrors
    ``viz/collection_viz.py:_decode_jpeg`` so both tools handle the D405 stream
    identically."""
    if not jpeg:
        return None
    try:
        from PIL import Image

        return np.asarray(Image.open(io.BytesIO(jpeg)).convert("RGB"))
    except Exception:  # noqa: BLE001 - a corrupt frame must not kill playback
        return None


# ---------------------------------------------------------------------------
# Force / torque arrows (a rough 3D cue; the exact signal is the charts/readouts)
# ---------------------------------------------------------------------------
#: MEASURED bota-sensor -> yam-TCP axis relabel: ``f_tcp = SENSOR_TO_TCP @ f_sensor``, i.e.
#: sensor +x is the TCP's +z, sensor +y is the TCP's −y, sensor +z is the TCP's +x. The Bota is
#: bolted on rotated, so the sensor frame and the TCP/wrist frame do NOT share an orientation —
#: exactly the same class of mistake as the puck→wrist relabel documented in
#: ``docs/REPLAY_AND_FIDELITY.md``, and it was live in these arrows until 2026-08-13.
#:
#: A proper rotation (det ``+1``, 180° about ``(1,0,1)/√2``) and SYMMETRIC, hence its own
#: inverse. Verified on ``session_2026.08.12_merged_v3``: fitting the sensor→TCP rotation and a
#: fixed world direction jointly across 3965 contact frames recovers a surface normal 7.7° off
#: world-vertical (the wiped table) and lands 19.9° from this matrix versus 169.0° from
#: identity — i.e. the un-relabelled arrow pointed INTO the table instead of out of it. Mirrors
#: FLoRe ``flore/utils/force_frame.py::TCP_FROM_SENSOR``, pinned against it by that repo's
#: ``tests/test_force_frame.py`` (the two cannot be imported across the venv split).
SENSOR_TO_TCP = np.array([[0.0, 0.0, 1.0],
                          [0.0, -1.0, 0.0],
                          [1.0, 0.0, 0.0]], dtype=np.float64)

#: ``F^cmd = CMD_FROM_MEAS_SIGN · F_meas`` — the measured→COMMANDED sign flip, applied by
#: :class:`WrenchArrows` so every replay draws the arrow the DEPLOY direction, not the sensor's
#: reaction. Everything these tools read is a MEASUREMENT (what the Bota felt while the human
#: worked), but the low-level force controller consumes a COMMAND, and it closes
#: ``e = F^cmd + F_meas → 0`` (holosoma ``force_deploy_math.servo_step``; UniFP Eq.2
#: ``x_target = x_contact + (F_cmd + F_ext)/K``). At that fixed point the measurement OPPOSES
#: the command, so the command that would reproduce a demonstrated press is the NEGATIVE of the
#: wrench the demo measured — a press reads ``F_meas`` pointing OUT of the table, and the
#: command that presses into it points IN.
#:
#: A SIGN, not a frame: it commutes with :data:`SENSOR_TO_TCP` and is its own inverse. Mirrors
#: FLoRe ``flore/utils/force_frame.py::CMD_FROM_MEAS_SIGN``, pinned against it by that repo's
#: ``tests/test_force_frame.py`` (the two cannot be imported across the venv split).
CMD_FROM_MEAS_SIGN = -1.0


class WrenchArrows:
    """Draw force (and optionally torque) as arrows anchored at a moving frame.

    The Bota wrench is expressed in the SENSOR frame (whether raw or
    gravity/inertia-compensated — both share the same frame/sign, so this rotation
    is valid either way; do NOT re-add a gravity term here). Two rotations get it into
    the scene, and BOTH are needed:

      1. :data:`SENSOR_TO_TCP` — the measured sensor→TCP axis relabel. Skipping it (as this
         class did until 2026-08-13) silently asserts that the Bota shares the gripper's axes,
         which is wrong by a 180° turn: the drawn force pointed *into* the wiped surface.
      2. the orientation of the frame we're showing (the wrist for the raw tool, the TCP for
         the zarr tool — they share an orientation, the TCP being a pure +x translation), so
         the arrow points sensibly relative to the gripper.

    ...and, since 2026-08-16, ONE SIGN: :data:`CMD_FROM_MEAS_SIGN` turns the stored measurement
    into the ``F^cmd`` a deploy would publish (``as_command``, default True). Pass
    ``as_command=False`` for the raw measured direction. Like the relabel it is applied HERE and
    callers must not pre-apply it — a second flip is invisible (two negations = identity),
    exactly the failure mode that hid the missing relabel for so long.

    Torque is rotated by the same matrix, and flipped by the same sign, which is correct as a
    change of AXES but does NOT re-reference it to the TCP origin (no ``r × f`` lever term) — so
    the torque arrow is "the sensor's torque, in TCP axes, negated". Force needs no such caveat.

    This is an APPROXIMATE visual cue — the precise values live in the numeric readout and
    the uPlot charts, which stay in the RAW sensor frame **and the RAW measured SIGN** (they are
    the fidelity view of the stored array; only the arrow is re-expressed for deploy). Each
    viewer states that split next to its arrow toggle. Arrow length is ``magnitude * scale``
    (metres per N / Nm), clamped so a spike can't fill the scene."""

    #: Default arrow colours (force, torque). Overridable per instance so a page showing TWO
    #: wrenches — e.g. FLoRe's planner, with a commanded and a measured one — can distinguish
    #: them while every other tool keeps the historical pair byte-identically.
    FORCE_RGB = (240, 80, 80)
    TORQUE_RGB = (80, 160, 240)

    def __init__(self, server: viser.ViserServer, name: str = "/wrench",
                 force_scale: float = 0.01, torque_scale: float = 0.05,
                 max_len_m: float = 0.4, show_torque: bool = True,
                 as_command: bool = True,
                 force_color=None, torque_color=None):
        self.server = server
        self.name = name
        self.force_scale = force_scale
        self.torque_scale = torque_scale
        self.max_len_m = max_len_m
        self.show_torque = show_torque
        self.as_command = as_command
        self.force_color = self.FORCE_RGB if force_color is None else tuple(force_color)
        self.torque_color = self.TORQUE_RGB if torque_color is None else tuple(torque_color)
        self.visible = True

    def _arrow(self, sub: str, origin: np.ndarray, vec: np.ndarray, scale: float,
               color) -> None:
        mag = float(np.linalg.norm(vec))
        name = f"{self.name}/{sub}"
        if not self.visible or mag < 1e-6:
            # Hide by drawing a zero-length (invisible) arrow.
            self.server.scene.add_arrows(name, points=np.zeros((1, 2, 3), dtype=np.float32),
                                         colors=color, visible=False)
            return
        length = min(mag * scale, self.max_len_m)
        end = origin + vec / mag * length
        pts = np.asarray([[origin, end]], dtype=np.float32)  # (1, 2, 3)
        self.server.scene.add_arrows(name, points=pts, colors=color,
                                     shaft_radius=0.004, head_radius=0.01,
                                     head_length=0.02, visible=True)

    def update(self, origin_xyz, wxyz, force_xyz, torque_xyz) -> None:
        """``force_xyz``/``torque_xyz`` are the RAW MEASURED SENSOR-frame triples; ``wxyz`` is the
        TCP/wrist-oriented frame the arrow is anchored to. Both rotations AND the measured→command
        sign are applied here (sign, relabel, then frame) — callers must NOT pre-apply either, or
        it lands twice (and both are involutions, so the second application is invisible)."""
        # R_world_tcp @ SENSOR_TO_TCP, times the meas→cmd sign: ONE composed matrix, so force and
        # torque cannot end up taking different paths (or different signs) into the scene.
        R = vtf.SO3(np.asarray(wxyz, dtype=np.float64)).as_matrix() @ SENSOR_TO_TCP
        if self.as_command:
            R = CMD_FROM_MEAS_SIGN * R
        self.update_world(
            origin_xyz,
            R @ np.asarray(force_xyz, dtype=np.float64),
            None if torque_xyz is None else R @ np.asarray(torque_xyz, dtype=np.float64),
        )

    def update_world(self, origin_xyz, force_world, torque_world=None) -> None:
        """Draw an ALREADY-RESOLVED world-frame wrench — no relabel, no sign, no frame rotation.

        For a source that is already expressed in the scene's own frame, where :meth:`update`'s
        sensor→TCP→world chain would be wrong rather than merely redundant. FLoRe's planner uses
        it for the low level's PELVIS-frame ``F_meas`` (its viser world *is* the pelvis frame,
        identity orientation), while its commanded arrow still goes through :meth:`update`.

        :meth:`update` delegates here, so both paths share one scale/clamp/threshold policy —
        which is what lets two arrows on the same page be compared by length.
        """
        origin = np.asarray(origin_xyz, dtype=np.float64)
        self._arrow("force", origin, np.asarray(force_world, dtype=np.float64),
                    self.force_scale, self.force_color)
        if self.show_torque and torque_world is not None:
            self._arrow("torque", origin, np.asarray(torque_world, dtype=np.float64),
                        self.torque_scale, self.torque_color)
        else:
            # Hide rather than strand the last drawn torque arrow when torque is switched off or
            # a source carries none (the low level's F_meas buffer is force-only).
            self.server.scene.add_arrows(f"{self.name}/torque",
                                         points=np.zeros((1, 2, 3), dtype=np.float32),
                                         colors=self.torque_color, visible=False)

    def set_torque_visible(self, vis: bool) -> None:
        """Toggle the torque arrow alone. The next :meth:`update_world` hides it when off, so a
        page can offer force-only (the default for FLoRe's planner, matching the offline inference
        viewer) without losing the torque cue entirely."""
        self.show_torque = bool(vis)
        if not vis:
            self.server.scene.add_arrows(f"{self.name}/torque",
                                         points=np.zeros((1, 2, 3), dtype=np.float32),
                                         colors=self.torque_color, visible=False)

    def set_visible(self, vis: bool) -> None:
        self.visible = vis
        if not vis:
            for sub in ("force", "torque"):
                self.server.scene.add_arrows(f"{self.name}/{sub}",
                                             points=np.zeros((1, 2, 3), dtype=np.float32),
                                             colors=(0, 0, 0), visible=False)


# ---------------------------------------------------------------------------
# uPlot helpers (a windowed scrolling x/y/z chart, like the live viz)
# ---------------------------------------------------------------------------
def make_xyz_chart(gui, title: str, labels, height: int = 110):
    """A 3-line (x/y/z) uPlot chart, matching ``collection_viz._make_xyz_chart``."""
    x0 = np.array([0.0, 1.0])
    zero = np.zeros(2)
    return gui.add_uplot(
        data=(x0, zero, zero, zero),
        series=(
            uplot.Series(label="t"),
            uplot.Series(label=labels[0], stroke=AXIS_COLORS[0], width=1.5),
            uplot.Series(label=labels[1], stroke=AXIS_COLORS[1], width=1.5),
            uplot.Series(label=labels[2], stroke=AXIS_COLORS[2], width=1.5),
        ),
        scales={"x": uplot.Scale(time=False)},
        title=title, aspect=2.2, height=height,
    )


def make_single_chart(gui, title: str, label: str, color_idx: int = 1,
                      y_range: tuple[float, float] | None = None, height: int = 110):
    x0 = np.array([0.0, 1.0])
    scales = {"x": uplot.Scale(time=False)}
    if y_range is not None:
        scales["y"] = uplot.Scale(range=y_range)
    return gui.add_uplot(
        data=(x0, np.zeros(2)),
        series=(uplot.Series(label="t"),
                uplot.Series(label=label, stroke=AXIS_COLORS[color_idx], width=1.5)),
        scales=scales, title=title, aspect=2.2, height=height,
    )


def set_full_xyz(chart, t: np.ndarray, xyz: np.ndarray) -> None:
    """Show the WHOLE (t, x, y, z) series at once. Used by the episode tools where
    the entire trajectory is known up front."""
    t = np.asarray(t, dtype=np.float64).reshape(-1)
    xyz = np.asarray(xyz, dtype=np.float64)
    if t.shape[0] < 2:
        return
    chart.data = (t, xyz[:, 0], xyz[:, 1], xyz[:, 2])


def set_full_single(chart, t: np.ndarray, y: np.ndarray) -> None:
    t = np.asarray(t, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if t.shape[0] < 2:
        return
    chart.data = (t, y)


# ---------------------------------------------------------------------------
# Playback controller — the draggable timeline shared by both tools
# ---------------------------------------------------------------------------
class Timeline:
    """A draggable time scrubber + play/pause/speed/step controls and the render
    loop that drives them.

    The owner supplies ``times`` (per-frame timestamps in SECONDS, monotonic
    non-decreasing, length K) and a ``render_at(i)`` callback. The slider is in
    real seconds so you can stop at a specific timestamp (the user's ask); play
    advances by WALL-CLOCK elapsed * speed, so non-uniform frame spacing (e.g. the
    camera grid) plays back in true real time. ``render_at`` must only set viser
    scene/GUI properties from pre-computed data — there is no live solve here, so
    it is safe to call from the loop thread or a slider callback (a lock serializes
    the two)."""

    def __init__(self, server: viser.ViserServer, times: np.ndarray,
                 render_at: Callable[[int], None], *, folder_label: str = "Playback",
                 loop_default: bool = True, max_fps: float = 60.0):
        self.server = server
        self.times = np.asarray(times, dtype=np.float64).reshape(-1)
        self.K = int(self.times.shape[0])
        self.render_at = render_at
        self.max_fps = max_fps
        self.dur = float(self.times[-1] - self.times[0]) if self.K > 1 else 0.0
        self._t0 = float(self.times[0]) if self.K else 0.0

        self._lock = threading.Lock()
        self._playing = False
        self._speed = 1.0
        self._t = 0.0               # current position, take-relative seconds
        self._force = True          # render needed (initial)
        self._last_i = -1
        self._suppress = False      # ignore slider callbacks we triggered ourselves
        self._stop = False          # run() returns after the current iteration
        # Optional callback run at the top of each loop iteration ON THE LOOP THREAD
        # (the zarr tool uses it to apply a pending episode switch safely). May call
        # set_series(); it runs before the frame index is computed.
        self.before_render: Callable[[], None] | None = None

        gui = server.gui
        step = self._nice_step()
        with gui.add_folder(folder_label):
            self.t_slider = gui.add_slider(
                "Time (s)", min=0.0, max=max(self.dur, step), step=step, initial_value=0.0)
            self.frame_txt = gui.add_text("Frame", initial_value=f"0 / {max(self.K-1,0)}",
                                          disabled=True)
            self.play_btn = gui.add_button("⏵ Play")
            row_prev = gui.add_button("⏮ Prev frame")
            row_next = gui.add_button("⏭ Next frame")
            self.speed_dd = gui.add_dropdown(
                "Speed", ("0.25x", "0.5x", "1x", "2x", "4x"), initial_value="1x")
            self.loop_cb = gui.add_checkbox("Loop", initial_value=loop_default)

        @self.t_slider.on_update
        def _(_e) -> None:
            if self._suppress:
                return
            with self._lock:
                self._t = float(self.t_slider.value)
                self._playing = False
                self.play_btn.label = "⏵ Play"
                self._force = True

        @self.play_btn.on_click
        def _(_e) -> None:
            with self._lock:
                self._playing = not self._playing
                self.play_btn.label = "⏸ Pause" if self._playing else "⏵ Play"

        @row_prev.on_click
        def _(_e) -> None:
            self._step_frame(-1)

        @row_next.on_click
        def _(_e) -> None:
            self._step_frame(+1)

        @self.speed_dd.on_update
        def _(_e) -> None:
            with self._lock:
                self._speed = float(self.speed_dd.value.rstrip("x"))

    def set_series(self, times: np.ndarray) -> None:
        """Retarget this timeline to a new per-frame timestamp array (e.g. when the
        zarr tool switches episode), reusing the SAME GUI handles so no duplicate
        controls are created and the render loop keeps running. Resets the cursor
        to 0 and forces a re-render of the first frame."""
        times = np.asarray(times, dtype=np.float64).reshape(-1)
        with self._lock:
            self.times = times
            self.K = int(times.shape[0])
            self.dur = float(times[-1] - times[0]) if self.K > 1 else 0.0
            self._t0 = float(times[0]) if self.K else 0.0
            self._t = 0.0
            self._playing = False
            self._last_i = -1
            self._force = True
        step = self._nice_step()
        # Update slider bounds for the new duration (setter is safe; suppress the echo).
        self._suppress = True
        try:
            self.t_slider.max = max(self.dur, step)
            self.t_slider.step = step
            self.t_slider.value = 0.0
        finally:
            self._suppress = False
        self.play_btn.label = "⏵ Play"

    def stop(self) -> None:
        """Ask :meth:`run` to return after the current iteration finishes.

        Lets the owner rebuild the whole viewer for a NEW take (the raw viewer's
        take switch tears down every scene node + GUI folder and re-enters ``main``
        with the new take) without racing the render loop: the loop checks this flag
        at the top of each iteration under the same lock as the play/scrub state, so
        it never mutates the scene after the owner has begun tearing it down. Safe
        from a viser callback thread (it only sets a bool)."""
        with self._lock:
            self._stop = True

    def request_render(self) -> None:
        """Force ``render_at`` to re-run on the next loop iteration even if the
        frame index hasn't changed (e.g. the owner switched a *display mode* while
        PAUSED, so the cached frame must be repainted). Sets the same ``_force``
        flag a scrub does, under the lock, so it is safe to call from the loop
        thread's ``before_render`` hook (the intended caller). Does NOT move the
        cursor or change play state."""
        with self._lock:
            self._force = True

    def current_frame(self) -> int:
        """The frame index the cursor is on RIGHT NOW, safe from any thread.

        ``before_render`` runs at the top of a loop iteration, *before* ``run()`` computes the
        index and calls ``render_at``. So a hook that needs "which frame is the user looking
        at" must ask here rather than remember the last rendered one: a scrub followed
        immediately by a button press would otherwise be serviced against the previous
        frame. Reads the same lock-guarded ``_t`` the loop reads."""
        with self._lock:
            return self._idx_for_t(self._t)

    def _nice_step(self) -> float:
        if self.K <= 1 or self.dur <= 0:
            return 1.0
        med_dt = float(np.median(np.diff(self.times)))
        return max(round(med_dt, 4), 1e-3)

    def _idx_for_t(self, t: float) -> int:
        if self.K <= 1:
            return 0
        abs_t = self._t0 + t
        i = int(np.searchsorted(self.times, abs_t, side="right") - 1)
        return int(np.clip(i, 0, self.K - 1))

    def _step_frame(self, d: int) -> None:
        with self._lock:
            i = int(np.clip(self._idx_for_t(self._t) + d, 0, self.K - 1))
            self._t = float(self.times[i] - self._t0)
            self._playing = False
            self.play_btn.label = "⏵ Play"
            self._force = True

    def _set_slider(self, t: float) -> None:
        # _suppress stops our own programmatic slider write (during playback) from
        # being echoed back through on_update as a "user scrubbed" event. It is set
        # on the loop thread and read on the viser callback thread WITHOUT the lock:
        # the only cost of the tiny race is that a real user scrub landing in this
        # ~microsecond window is dropped (the loop overwrites the slider next frame
        # anyway). All actual playback state (_t/_playing/_force) IS lock-guarded, so
        # the race cannot corrupt state — it's a benign, at-most-one-dropped-scrub.
        self._suppress = True
        try:
            self.t_slider.value = float(np.clip(t, 0.0, self.t_slider.max))
        finally:
            self._suppress = False

    def run(self) -> None:
        """Blocking render loop. Renders the initial frame, then advances while
        playing and re-renders on scrub/step."""
        period = 1.0 / self.max_fps
        last = time.perf_counter()
        while True:
            with self._lock:
                if self._stop:
                    return
            now = time.perf_counter()
            dt = now - last
            last = now
            if self.before_render is not None:
                try:
                    self.before_render()
                except Exception as e:  # noqa: BLE001 - keep the loop alive
                    print(f"[replay] before_render error: {e!r}", flush=True)
            with self._lock:
                playing, speed, t, force = self._playing, self._speed, self._t, self._force
                self._force = False
            if playing and self.dur > 0:
                t += dt * speed
                if t >= self.dur:
                    if self.loop_cb.value:
                        t = 0.0
                    else:
                        t = self.dur
                        with self._lock:
                            self._playing = False
                        self.play_btn.label = "⏵ Play"
                with self._lock:
                    self._t = t
                self._set_slider(t)
            i = self._idx_for_t(t)
            if force or i != self._last_i:
                try:
                    self.render_at(i)
                except Exception as e:  # noqa: BLE001 - keep the loop alive; surface once
                    print(f"[replay] render error at frame {i}: {e!r}", flush=True)
                self._last_i = i
                self.frame_txt.value = f"{i} / {self.K - 1}  (t={self.times[i]-self._t0:.2f}s)"
            time.sleep(period)
