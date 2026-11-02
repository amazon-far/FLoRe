#!/usr/bin/env python3
"""The PURPLE force-recovery funnel drawn on top of the zarr replay scene.

**Everything this module draws lives in FORCE space, not in the room.** A point is the TIP of a
wrench vector, scaled at some metres-per-newton and pinned to the anchor frame's TCP because that
is where the force acts — the curves are *not* a path the gripper travels, and they carry no
information about where the tracker went. The scene legend and the in-scene ``FORCE SPACE (N)``
label exist so that cannot be misread, and a labelled reference tick gives the newton scale.

Press **Generate** at any frame and this draws, anchored at that frame's TCP:

  * every generated RECOVERY trajectory (purple) — each starting at its own out-of-distribution
    wrench and funnelling onto the TARGET, so the bundle is wide at the mouth and pinched at the
    throat. Under ``anchor`` each purple curve is a *straight* segment in wrench space and that
    is not a coincidence: one fixed target + a linear solve means the funnel weights can only set
    the SPEED along the start->target line, never bend it (measured 2e-14 N off the line). Under
    ``demo`` the target moves every step, so purple genuinely curves;
  * the TARGET (green arrow + green dot). Under the default ``target_mode="anchor"`` this is the
    single wrench at **t0** — the nominal force for this contact — so every trajectory converges
    to one point, which is 1001-Demos' own shape ("converge consistently to **the same**
    pre-contact pose"). It is the same vector the viewer's own RED ``F^cmd`` arrow draws, so
    parking the timeline on t0 and pressing Play shows the purple arrow closing onto the red one.
  * the DEMO force chunk (amber line) — read from the zarr, never generated. Under
    ``"anchor"`` it is **context** (what the demonstration went on to do), not the target: over
    one 1.667 s chunk the demonstrated force turns a median 86 deg away from its t0 value.
    Under ``target_mode="demo"`` the target becomes this curve step by step instead — and then,
    and only then, the green target arrow rides the amber line. Amber, not green, precisely so
    "the demo" and "the target" cannot be read as one thing under ``anchor``.
  * optionally the naive straight line from the same start (grey), which is what a
    linear-interpolation "recovery" would look like and is visibly *not* the demo's shape;
  * ``Play force arrow`` walks k on its own loop, independent of the main timeline, so the
    recovery can be watched without scrubbing.

Red and green are two resamplings of one physical wrench — the red arrow reads
``robot0_ft_wrench`` (the converter's camera-grid mean) and the target reads
``hl_*_action_force`` (the 30 Hz action grid) — so they agree to about **0.08 N** (p50; p99
0.175 N), not exactly. The panel prints the live difference rather than leaving it to be
guessed at.

Why the maths is imported instead of copied
-------------------------------------------
The funnel lives in FLoRe's :mod:`flore.utils.force_funnel`, which is deliberately numpy-only for
this reason. This repo normally keeps its own copies of shared constants (see
``replay_common.SENSOR_TO_TCP``) because it must stand alone, and a 3x3 matrix pinned by a test
on the other side is cheap to duplicate. A ~400-line solver is not: a drifting copy would make
this viewer draw a *different* funnel than the one training consumes, and since this viewer IS
the acceptance test for that augmentation, that failure would be both silent and maximally
expensive. So the import is optional and degrades honestly — no FLoRe on disk means the panel
says exactly why it is disabled, and the rest of the viewer is untouched.

Frame convention, matching the existing red arrow exactly
--------------------------------------------------------
The stored wrench is in the **bota sensor** frame. A scene point is
``anchor_pos + scale * (sign * R_world_tcp @ SENSOR_TO_TCP) @ F``, i.e. the same relabel and
the same measured->commanded sign :class:`replay_common.WrenchArrows` applies, so a purple curve
and the red arrow live in one space. Two deliberate choices:

* ``R_world_tcp`` is the orientation at the **anchor** frame, held for the whole chunk. The
  chunk is a plan made at t0; re-rotating each step by where the hand later went would fold the
  robot's own motion into a wrench plot and make a straight force look curved.
* The tip of the drawn arrow is read out of the *same array* as the curve, rather than
  recomputed, so the arrow cannot drift off the line it is meant to follow.

Read-only and additive: it never writes the zarr and never changes what the rest of the viewer
draws.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

import replay_common as rc

#: The funnel bundle. Purple because the user asked for purple; distinct from the amber demo
#: chunk, the green target and the red/blue stored-wrench arrows already in the scene.
_AUG_PURPLE = (178, 102, 240)
#: Dimmer, for the torque bundle when it is switched on.
_AUG_PURPLE_DIM = (128, 88, 168)
#: The funnel's TARGET — the thing purple converges onto.
_TARGET_GREEN = (90, 220, 120)
#: The demo chunk. Deliberately NOT green: under ``target_mode="anchor"`` the demo is *not* the
#: target, and painting both green made a reader ask which of the two the funnel was generating.
#: Also not green because the scene's stored-eef path is already green, in the room.
_DEMO_AMBER = (235, 185, 70)
#: The naive linear "recovery", for contrast.
_LINEAR_GREY = (150, 150, 150)
#: The newton-scale reference tick.
_SCALE_WHITE = (230, 230, 230)

#: Legend, kept next to the colours it describes so the two cannot drift apart. Also states the
#: two things a viewer of this scene has to know and cannot see: that the space is force, and
#: that the scene's pre-existing red/blue arrows are the STORED wrench (which is why the panel
#: says what they are — they belong to ``replay_common.WrenchArrows``, not to this module).
_LEGEND_MD = """\
**All curves below are FORCE space (N), anchored at the TCP — not a path in the room.**

| | what it is | generated? |
|---|---|---|
| 🟡 **amber line** | the DEMO force chunk over the next H steps, read from the zarr | **no — data** |
| 🟢 **green arrow + dot** | the funnel's **TARGET**, what purple closes onto | **no — data** |
| 🟣 **purple lines** | the OOD→recovery trajectories (the funnel) | **yes** |
| 🟣 **purple arrow** | the selected recovery at step *k* | **yes** |
| ⬜ **grey lines** | the naive straight-line "recovery", for contrast | yes |
| ⬛ **white tick** | scale reference, length = the newtons printed beside it | — |

**Only the purple is generated.** Generate draws amber and green too because the viewer has no
other way to show them: the red arrow is one instant, and these are that instant's *next 1.67 s*.

`target mode = anchor` (default): the target is the ONE wrench at **t0**, held — the green arrow
never moves, it **coincides with the 🔴 red arrow** when the timeline sits on t0, and every purple
line is a straight run from its own OOD start back to it. The amber line is then only *context*
(what the demo went on to do — a median 86° away), NOT the target.
`target mode = demo`: the target becomes the amber line step by step, so the green arrow rides
amber and purple bends to follow it, ending far from the red arrow.

Already in this scene, from the viewer itself (not this panel):
🔴 **red arrow = stored FORCE** `F^cmd = -F_meas` · 🔵 **blue arrow = stored torque**, both at
the frame the timeline is on. Red comes from `robot0_ft_wrench`, amber/green from
`hl_*_action_force` — two resamplings of one wrench, so they agree to ~0.08 N, not exactly.
"""


def _import_force_funnel():
    """Import FLoRe's funnel module off the source tree, or return ``(None, reason)``.

    ``replay/`` -> UMI_collection root -> ``third_party/`` -> the FLoRe checkout, i.e. three
    levels up, which is where this repo sits when it is vendored inside FLoRe. Standalone
    clones simply get the panel disabled.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    flore_root = os.path.abspath(os.path.join(here, "..", "..", ".."))
    marker = os.path.join(flore_root, "flore", "utils", "force_funnel.py")
    if not os.path.exists(marker):
        return None, f"flore/utils/force_funnel.py not found under {flore_root}"
    if flore_root not in sys.path:
        sys.path.append(flore_root)
    try:
        from flore.utils import force_funnel
    except Exception as exc:  # noqa: BLE001 - a viewer must not die on an optional import
        return None, f"import failed: {type(exc).__name__}: {exc}"
    return force_funnel, ""


class ForceRecoveryOverlay:
    """GUI panel + scene nodes for the funnel augmentation. One instance per viewer."""

    def __init__(self, server, *, horizon: int, dt: float, name: str = "/force_aug"):
        self.server = server
        self.name = name
        self.horizon = int(horizon)
        self.dt = float(dt)
        self.ff, self.disabled_reason = _import_force_funnel()
        # Drawn-line handles, kept so a changed n_aug removes the stale ones instead of
        # leaving orphans behind (viser keeps a node until it is explicitly removed).
        self._lines: list = []
        self._arrow_name = f"{name}/arrow"
        self._target_arrow_name = f"{name}/target_arrow"
        self._tip_name = f"{name}/tip"
        # Generation state, all written on the loop thread only.
        self._batch = None            # flore RecoveryBatch
        self._pts = None              # (n, H, 3) scene-space points for the force curves
        self._target_pts = None       # (H, 3) same, for the funnel TARGET (the green arrow)
        self._anchor_pos = None       # (3,) where the bundle is pinned
        self._anchor_frame = -1       # LOCAL frame index inside the episode
        self._anchor_ep = -1
        self._stored_wrench = None    # robot0_ft_wrench[t0]: what the RED arrow draws
        self._request = False         # set by the button, consumed by poll()
        # Independent arrow animation: a step cursor advanced by the loop thread, so the
        # recovery can be watched without dragging the shared timeline (which would also move
        # the robot, the RGB and the stored-wrench arrows and bury the thing being looked at).
        self._anim_on = False
        self._anim_k = 0.0
        self._anim_last_t = None
        #: Last integer step actually drawn. At 20 steps/s against a 60 fps loop the same step
        #: comes round ~3x, and redrawing it re-sends two arrows, a point cloud and two GUI
        #: text updates for an identical picture. Reset to -1 by :meth:`_draw` so a scale or
        #: selection change still refreshes the arrows.
        self._last_drawn_k = -1

    # -- GUI ---------------------------------------------------------------
    @property
    def available(self) -> bool:
        return self.ff is not None

    def build_gui(self, gui) -> None:
        with gui.add_folder("Force recovery aug (1001-Demos funnel)"):
            if not self.available:
                gui.add_text("status", initial_value="DISABLED", disabled=True)
                gui.add_text("reason", initial_value=self.disabled_reason[:90], disabled=True)
                gui.add_text("what it needs",
                             initial_value="FLoRe checkout with flore/utils/force_funnel.py",
                             disabled=True)
                self.gen_btn = None
                return
            self._add_legend(gui)
            self.gen_btn = gui.add_button("Generate at this frame")
            self.clear_btn = gui.add_button("Clear")
            self.play_btn = gui.add_button("▶ Play force arrow")
            self.anim_hz = gui.add_slider("arrow steps/s", min=2.0, max=60.0, step=2.0,
                                          initial_value=15.0)
            self.n_aug = gui.add_slider("trajectories", min=1, max=32, step=1, initial_value=8)
            self.target_mode = gui.add_dropdown("target mode", tuple(self.ff.TARGET_MODES),
                                                initial_value=self.ff.TARGET_MODES[0])
            # Named after the paper's own knobs so the panel and §3.2 read the same.
            self.radius_scale = gui.add_slider("force radius x ramp", min=0.25, max=6.0,
                                               step=0.25, initial_value=2.0)
            self.cone_deg = gui.add_slider("torque cone (deg)", min=5.0, max=180.0, step=5.0,
                                           initial_value=50.0)
            self.funnel_power = gui.add_slider("funnel power (k/T)^p", min=1.0, max=8.0,
                                               step=0.5, initial_value=3.0)
            self.lambda_acc = gui.add_slider("smoothness lambda", min=0.5, max=64.0, step=0.5,
                                             initial_value=4.0)
            self.seed = gui.add_number("seed", initial_value=0, step=1)
            self.scale = gui.add_slider("draw scale (m per N)", min=0.002, max=0.05,
                                       step=0.002, initial_value=0.01)
            self.pick = gui.add_slider("arrow follows #", min=0, max=31, step=1, initial_value=0)
            self.show_demo = gui.add_checkbox("show demo chunk (amber, from zarr)",
                                              initial_value=True)
            self.show_linear = gui.add_checkbox("show naive linear (grey)", initial_value=False)
            self.show_torque = gui.add_checkbox("show torque bundle", initial_value=False)
            self.show_legend_3d = gui.add_checkbox("in-scene labels", initial_value=True)
            self.rd_target = gui.add_text("target vs red arrow", initial_value="-", disabled=True)
            self.rd_step = gui.add_text("arrow at step k", initial_value="-", disabled=True)
            self.rd_close = gui.add_text("purple-to-target |dF|", initial_value="-",
                                         disabled=True)
            self.rd_gap = gui.add_text("start gap |dF| (N)", initial_value="-", disabled=True)
            self.rd_merge = gui.add_text("merge step", initial_value="-", disabled=True)
            self.rd_rej = gui.add_text("draws rejected", initial_value="-", disabled=True)
            self.rd_anchor = gui.add_text("anchored at", initial_value="-", disabled=True)

            # Buttons only raise a flag; the loop thread does the work (same discipline as the
            # viewer's episode switch and its home-anchor toggle).
            @self.gen_btn.on_click
            def _(_e) -> None:
                self._request = True

            @self.clear_btn.on_click
            def _(_e) -> None:
                self._request = "clear"

            @self.play_btn.on_click
            def _(_e) -> None:
                # Flag only; the loop thread owns _anim_on and the cursor. Restart from the
                # mouth on every press, so "play" always shows the whole recovery.
                self._request = "toggle_play"

            for handle in (self.n_aug, self.radius_scale, self.cone_deg, self.funnel_power,
                           self.lambda_acc, self.seed, self.target_mode):
                @handle.on_update
                def _(_e) -> None:
                    # A knob change re-generates only if something is already on screen, so
                    # dragging a slider before the first Generate stays inert.
                    if self._batch is not None:
                        self._request = True

            for handle in (self.scale, self.show_demo, self.show_linear, self.show_torque,
                           self.show_legend_3d):
                @handle.on_update
                def _(_e) -> None:
                    if self._batch is not None:
                        self._request = "redraw"

    def _add_legend(self, gui) -> None:
        """The colour key. Markdown if this viser has it, plain text otherwise — the legend is
        the whole point of the user's ask, so it must not depend on an optional GUI widget."""
        try:
            gui.add_markdown(_LEGEND_MD)
            return
        except Exception:  # noqa: BLE001 - fall back rather than lose the legend
            pass
        for line in _LEGEND_MD.splitlines():
            text = line.strip().strip("|").replace("**", "").replace("`", "")
            if text and not set(text) <= set("-| "):
                gui.add_text(" ", initial_value=text[:90], disabled=True)

    # -- generation, on the loop thread ------------------------------------
    def poll(self, *, ep: int, frame: int, demo_chunk, anchor_pos, anchor_wxyz,
             stored_wrench=None) -> bool:
        """Consume a pending request. Returns True if the caller should force a re-render.

        ``demo_chunk`` is ``(H, 6)`` for the CURRENT frame (or None if this frame has no usable
        window), ``anchor_pos``/``anchor_wxyz`` are that frame's DISPLAY pose, and
        ``stored_wrench`` is ``robot0_ft_wrench[frame]`` — the vector the RED arrow draws — used
        only to report how far the funnel's target sits from it. Must be called
        from the render-loop thread — it mutates scene nodes and the overlay's own state.
        """
        if not self.available:
            return False
        request, self._request = self._request, False
        # Advance the arrow animation FIRST and unconditionally: it has to tick on every loop
        # iteration, not only on the iterations that happen to carry a button press.
        animated = self._tick_animation()
        if not request:
            return animated
        if request == "clear":
            self._clear()
            return True
        if request == "toggle_play":
            self._anim_on = not self._anim_on and self._batch is not None
            self._anim_k = 0.0
            self._anim_last_t = None
            self.play_btn.label = "⏸ Pause force arrow" if self._anim_on else "▶ Play force arrow"
            if not self._anim_on and self._batch is not None:
                self._draw_arrows(0)      # park it back at the mouth rather than mid-flight
            return True
        if request == "redraw":
            # Only ever a re-DRAW. Falling through to generation when nothing is on screen
            # would turn a scale-slider nudge into an unrequested Generate; the handler already
            # guards on `_batch`, and this makes it true regardless of thread interleaving.
            if self._batch is not None:
                self._draw()
            return True
        if demo_chunk is None:
            self.rd_anchor.value = f"frame {frame}: no usable window here"
            self._clear()
            return True

        cfg = self.ff.FunnelConfig(
            n_aug=int(self.n_aug.value),
            force_radius_scale=float(self.radius_scale.value),
            torque_cone_deg=float(self.cone_deg.value),
            funnel_power=float(self.funnel_power.value),
            lambda_acc=float(self.lambda_acc.value),
            target_mode=str(self.target_mode.value),
        )
        demo = np.asarray(demo_chunk, dtype=np.float64)
        self._batch = self.ff.generate_recovery_chunks(
            demo, dt=self.dt, cfg=cfg, rng=int(self.seed.value) + frame)
        self._anchor_pos = np.asarray(anchor_pos, dtype=np.float64)
        self._anchor_frame = int(frame)
        self._anchor_ep = int(ep)
        self._stored_wrench = (None if stored_wrench is None
                               else np.asarray(stored_wrench, dtype=np.float64).reshape(-1))
        # ONE composed matrix, identical to WrenchArrows.update's chain (sign, relabel, then
        # the shown frame), so the purple curves and the red arrow share a space.
        from viser import transforms as vtf
        rot = vtf.SO3(np.asarray(anchor_wxyz, dtype=np.float64)).as_matrix() @ rc.SENSOR_TO_TCP
        self._rot = rc.CMD_FROM_MEAS_SIGN * rot
        self._draw()
        return True

    def _tick_animation(self) -> bool:
        """Advance the independent arrow cursor. Returns True if a re-render is needed.

        Wall-clock paced (``arrow steps/s``), not one step per loop iteration, so the speed
        means something and does not ride the render loop's frame rate."""
        if not self._anim_on or self._pts is None:
            return False
        now = time.monotonic()
        if self._anim_last_t is None:
            self._anim_last_t = now
        self._anim_k += (now - self._anim_last_t) * float(self.anim_hz.value)
        self._anim_last_t = now
        horizon = int(self._pts.shape[1])
        if self._anim_k >= horizon:
            self._anim_k = 0.0                # loop, so the funnel can be watched repeatedly
        step = int(self._anim_k)
        if step == self._last_drawn_k:
            return False
        self._draw_arrows(step)
        return True

    def clear(self) -> None:
        """Remove everything drawn. Called on an episode switch, since the bundle is pinned to
        a pose in the episode it was generated from. Loop thread only, like :meth:`poll`."""
        if self.available:
            self._anim_on = False
            self.play_btn.label = "▶ Play force arrow"
            self._clear()

    def _clear(self) -> None:
        for handle in self._lines:
            handle.remove()
        self._lines.clear()
        self._hide(self._arrow_name)
        self._hide(self._target_arrow_name)
        self.server.scene.add_point_cloud(self._tip_name, points=np.zeros((1, 3), np.float32),
                                          colors=np.zeros((1, 3), np.uint8), point_size=0.001,
                                          visible=False)
        self._batch = None
        self._pts = None
        self._target_pts = None
        self._last_drawn_k = -1
        for rd in ("rd_gap", "rd_merge", "rd_rej", "rd_step", "rd_close", "rd_target"):
            getattr(self, rd).value = "-"

    def _hide(self, name: str) -> None:
        self.server.scene.add_arrows(name, points=np.zeros((1, 2, 3), dtype=np.float32),
                                     colors=(0, 0, 0), visible=False)

    #: Torque is drawn at this multiple of the force scale, matching
    #: ``WrenchArrows``' own 0.05 / 0.01 ratio. Without it the torque bundle (~0.7 N.m against
    #: ~5 N) would be 7x shorter than the force bundle and effectively invisible.
    TORQUE_SCALE_RATIO = 5.0

    def _scene_points(self, wrench_xyz: np.ndarray, *, torque: bool = False) -> np.ndarray:
        """``(..., 3)`` wrench triples -> scene points, anchored and scaled."""
        vec = np.asarray(wrench_xyz, dtype=np.float64) @ self._rot.T
        scale = float(self.scale.value) * (self.TORQUE_SCALE_RATIO if torque else 1.0)
        return self._anchor_pos + scale * vec

    def _polyline(self, name: str, pts: np.ndarray, color, width: float) -> None:
        p = np.asarray(pts, dtype=np.float32)
        segs = np.stack([p[:-1], p[1:]], axis=1)          # (H-1, 2, 3), every stored step kept
        self._lines.append(self.server.scene.add_line_segments(
            name, points=segs, colors=color, line_width=width))

    def _draw(self) -> None:
        """Rebuild every line. Cheap enough to redo wholesale (32 x 50 points)."""
        for handle in self._lines:
            handle.remove()
        self._lines.clear()
        # A scale/selection change must repaint the arrows, so forget what was last drawn.
        self._last_drawn_k = -1
        batch = self._batch
        if batch is None or batch.n == 0:
            self.rd_rej.value = f"{getattr(batch, 'n_rejected', 0)} (nothing admissible)"
            return
        fs, ts = self.ff.FORCE_SLICE, self.ff.TORQUE_SLICE

        # Cached so the arrows read their tips out of the SAME arrays the curves were drawn
        # from — they cannot drift off the lines they are meant to follow.
        self._pts = self._scene_points(batch.chunks[:, :, fs])
        # The TARGET the arrows are compared against is the REFERENCE, not the demo. Under
        # anchor mode those are different arrays, and drawing the demo as "the target" would
        # point the operator at a curve the funnel never aimed at.
        self._target_pts = self._scene_points(batch.reference[:, fs])
        for k in range(batch.n):
            self._polyline(f"{self.name}/traj_{k}", self._pts[k], _AUG_PURPLE, 1.5)
        if self.show_torque.value:
            tpts = self._scene_points(batch.chunks[:, :, ts], torque=True)
            for k in range(batch.n):
                self._polyline(f"{self.name}/torq_{k}", tpts[k], _AUG_PURPLE_DIM, 1.0)
        if self.show_demo.value:
            self._polyline(f"{self.name}/demo", self._scene_points(batch.demo[:, fs]),
                           _DEMO_AMBER, 2.0)
        # The funnel's throat, as a dot: under anchor mode every trajectory ends exactly here.
        self._lines.append(self.server.scene.add_point_cloud(
            f"{self.name}/target", points=np.asarray([self._target_pts[-1]], dtype=np.float32),
            colors=np.asarray([_TARGET_GREEN], dtype=np.uint8), point_size=0.016))
        self._draw_scale_and_labels()
        if self.show_linear.value:
            for k in range(batch.n):
                line = self.ff.linear_reference(batch.demo, batch.starts[k],
                                                str(self.target_mode.value))
                self._polyline(f"{self.name}/lin_{k}", self._scene_points(line[:, fs]),
                               _LINEAR_GREY, 1.0)

        gap = np.linalg.norm(batch.starts[:, fs] - batch.reference[0, fs], axis=1)
        self.rd_gap.value = (f"p50 {np.percentile(gap, 50):.2f}  max {gap.max():.2f}  "
                             f"(train |F| peaks ~7.7 N)")
        self.rd_merge.value = (f"p50 {np.percentile(batch.merge_steps, 50):.0f} / "
                               f"{self.horizon - 1}  ({self.dt * float(np.percentile(batch.merge_steps, 50)):.2f} s)")
        self.rd_rej.value = f"{batch.n_rejected}  {batch.reject_reasons or ''}"
        self.rd_anchor.value = (f"ep {self._anchor_ep} frame {self._anchor_frame}, n={batch.n}, "
                                f"mode={self.target_mode.value}")
        if self._stored_wrench is None:
            self.rd_target.value = "no stored wrench passed in"
        else:
            # The cross-check the two pipelines make possible: the funnel's target and the red
            # arrow are the same physical wrench read off two different resamplings.
            d = float(np.linalg.norm(batch.reference[-1, fs] - self._stored_wrench[fs]))
            self.rd_target.value = (f"{d:.3f} N apart  (|target|="
                                    f"{float(np.linalg.norm(batch.reference[-1, fs])):.2f} N)")

    def _draw_scale_and_labels(self) -> None:
        """A labelled newton tick plus the in-scene ``FORCE SPACE`` caption.

        Without a scale bar a 3-D curve in force space has no readable magnitude — and without
        the caption the bundle is easy to mistake for a path the gripper travels, which is the
        one thing it is not."""
        show = bool(self.show_legend_3d.value)
        # Tick along the scene's +z from the anchor, at a round newton value the bundle spans.
        span_n = 5.0
        tip = self._anchor_pos + np.array([0.0, 0.0, span_n * float(self.scale.value)])
        seg = np.stack([[self._anchor_pos, tip]]).astype(np.float32)
        # Appended to _lines, like the curves, so Clear and the next _draw() remove them —
        # otherwise a stale "5 N" caption would hang in the air over an emptied scene.
        self._lines.append(self.server.scene.add_line_segments(
            f"{self.name}/scale", points=seg, colors=_SCALE_WHITE, line_width=2.0, visible=show))
        self._lines.append(self.server.scene.add_label(
            f"{self.name}/scale_label", f"{span_n:.0f} N", position=tuple(tip), visible=show))
        self._lines.append(self.server.scene.add_label(
            f"{self.name}/space_label", "FORCE SPACE (N) - not a path in the room",
            position=tuple(self._anchor_pos + np.array([0.0, 0.0, -0.03])), visible=show))

    # -- per-frame -----------------------------------------------------
    def _draw_arrows(self, k: int) -> None:
        """Purple = the selected recovery at step ``k``; green = the funnel's TARGET at ``k``.

        Drawing them as a pair is what makes the recovery legible: the purple arrow closes onto
        the green one, and at ``k = H-1`` they coincide exactly (the terminal constraint). Under
        ``target_mode="anchor"`` the green arrow does not move — it is the t0 wrench — so it is
        also the viewer's own red ``F^cmd`` arrow when the timeline sits on t0."""
        self._last_drawn_k = int(k)
        idx = int(np.clip(self.pick.value, 0, self._pts.shape[0] - 1))
        end = self._pts[idx, k]
        self.server.scene.add_arrows(
            self._arrow_name,
            points=np.asarray([[self._anchor_pos, end]], dtype=np.float32),
            colors=_AUG_PURPLE, shaft_radius=0.005, head_radius=0.012, head_length=0.024,
            visible=True)
        target_end = self._target_pts[k]
        self.server.scene.add_arrows(
            self._target_arrow_name,
            points=np.asarray([[self._anchor_pos, target_end]], dtype=np.float32),
            colors=_TARGET_GREEN, shaft_radius=0.004, head_radius=0.010, head_length=0.020,
            visible=True)
        # A dot on the curve at "now", so the arrow's tip is unambiguous against N overlapping
        # purple lines.
        self.server.scene.add_point_cloud(
            self._tip_name, points=np.asarray([end], dtype=np.float32),
            colors=np.asarray([_AUG_PURPLE], dtype=np.uint8), point_size=0.012, visible=True)
        # Report the gap in NEWTONS, from the wrench arrays rather than the drawn metres, so the
        # number does not silently change when the draw scale is dragged.
        chunk, reference = self._batch.chunks[idx], self._batch.reference
        fs = self.ff.FORCE_SLICE
        gap_n = float(np.linalg.norm(chunk[k, fs] - reference[k, fs]))
        self.rd_step.value = f"{k} / {self._pts.shape[1] - 1}  (t0+{k * self.dt:.2f}s)"
        self.rd_close.value = (f"{gap_n:.2f} N  (was "
                               f"{float(np.linalg.norm(chunk[0, fs] - reference[0, fs])):.2f} N at k=0)")

    def render_at(self, frame: int) -> None:
        """Move the arrows to the main timeline's frame — unless the independent Play is running.

        Hidden outside the anchored chunk's span: there is no prediction for a frame the plan
        does not cover, and an arrow left frozen at the boundary would read as one.
        """
        if not self.available or self._pts is None or self._batch is None:
            return
        if self._anim_on:
            return          # the animation owns the arrows; don't fight it from the timeline
        k = int(frame) - self._anchor_frame
        if k < 0 or k >= self._pts.shape[1]:
            self._hide(self._arrow_name)
            self._hide(self._target_arrow_name)
            return
        self._draw_arrows(k)
