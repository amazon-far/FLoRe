#!/usr/bin/env python3
"""The G1 humanoid drawn on top of the zarr replay, driven by ONLINE motion completion.

Additive, opt-in, and off unless ``replay_zarr_viz.py --motion-completion`` is passed:
without the flag this module is constructed but never starts a backend, adds no scene nodes
and no GUI panel, and the replay is byte-for-byte the gripper-only viewer it has always
been. Same contract (and same shape) as ``replay/force_recovery_overlay.py``.

What it draws
=============
* the **G1 humanoid**, from ``robot_assets/g1_yam/g1_29dof_yam.urdf`` — THIS repo's URDF,
  with the yam gripper and the Bota F/T sensor, not Motion_Completion's ``assembly.urdf``.
  That is safe because all 29 canonical joint origins are bit-identical between the two
  (see :mod:`motion_completion.mc_frames`), so the completed 29 DoF FK to the same place in
  either — and it means the drawn robot is the one this rig actually deploys, down to the
  jaws, which are driven from the take's own recorded gripper width.
* the **conditioning the model saw** — L/R ``gripper_base`` triads + trails and the root
  (chest) path + heading triad, in Motion_Completion's own viewer colours (blue L, pink R,
  orange root) so the two views read the same way.
* the **keypoints handed to the low-level controller** for the selected mode
  (2pt / 3pt / 5pt / 29dof), each at holosoma's own lever off its body — the "then pick"
  step, drawn so it can be checked rather than trusted.

Online, and what that costs
===========================
Windows are run **as the play head reaches them**, one at a time, on a worker thread — not
precomputed. The schedule is Motion_Completion's own (T=30, overlap 10, stride 20, anchor
seed from the previous window's tail), so what runs here is what a deployment loop runs.

The honest caveat, stated in the panel because it is a property of the CHECKPOINT and not of
this integration: its conditioning is dense over the whole 30-frame window, so a window
cannot be run until its LAST cond frame exists. On a robot that is a command lag of one
window — 30 frames, 1.00 s at 30 Hz — before adding the ~30 ms of compute. The default view
draws the completion at the play head (aligned with the gripper cluster, which is what makes
the frame chain checkable by eye); the ``deployment lag`` checkbox shifts it back by that
one window so the deploy question can be looked at too. Both come from the identical
online loop; only which frame is DISPLAYED changes.

The z lift, and why the robot stands below the grid
==================================================
The recorded world frame's z=0 is the libsurvive calibration origin, not the floor; the
model's frame's z=0 IS the floor. :mod:`motion_completion.mc_frames` lifts the conditioning
by ``dz`` (≈ +0.60 m on the reference session) and this overlay subtracts the same ``dz``
back off the output, so the humanoid's hands land ON the take's recorded eef path and its
feet land on the REAL floor — which is ``dz`` below the grid, because the grid is drawn at
the libsurvive origin. That is not the robot floating: it is the grid not being the floor,
which was already true of this scene. ``Stand on the grid`` keeps the lift instead.

Threading
=========
This repo's rule is "buttons raise flags, the loop thread mutates". Every scene write here
happens in :meth:`poll` or :meth:`render_at`, both called from the timeline's loop thread.
The worker thread ONLY talks to the backend over the pipe and posts results into a queue;
it never touches viser. GUI callbacks only set flags.
"""
from __future__ import annotations

import os
import queue
import sys
import threading
import time
from pathlib import Path

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, ".."))
for _p in (os.path.join(_REPO, "motion_completion"), os.path.join(_REPO, "viz")):
    if _p not in sys.path:
        sys.path.append(_p)

#: Motion_Completion's viewer colours, deliberately reused: someone who has looked at
#: ``visualizer_app/viser_motion_completion.py`` should recognise this scene on sight.
_COLOR_L = (51, 140, 255)
_COLOR_R = (255, 102, 140)
_COLOR_ROOT = (255, 140, 51)
#: How far above the floor the root conditioning is drawn — MC's own 0.02 m
#: (``viser_motion_completion.py``'s pelvis path / pelvis_axes), so the two viewers agree.
_ROOT_DRAW_Z = 0.02
#: ``ankle_roll_link`` sits ~0.03 m above the floor on a planted G1 foot. Compared against the
#: LOWER of the two ankles, never one of them: a stepping foot legitimately swings to 0.26 m,
#: so a per-ankle threshold would call every step a fault.
_ANKLE_FLOAT_M = 0.075
#: The keypoint markers handed to the low-level controller. Bright and distinct from the
#: scene's existing green (recorded eef path) / orange (right hand) / violet (chest).
_COLOR_KP = (255, 255, 90)

_DEFAULT_URDF = os.path.join(_REPO, "robot_assets", "g1_yam", "g1_29dof_yam.urdf")

#: Total jaw opening (m) at which the drawn fingers are fully open. Same single source as
#: the rest of the viewer: the URDF prismatic limit, doubled (aperture = 2q).
_JAW_OPEN_M = 0.0475
_WIDTH_OPEN_M = 2 * _JAW_OPEN_M

_MODES = ("3pt", "2pt", "5pt", "29dof")
_HAND_POINTS = ("tcp", "lever")
_ROT_AXES = ("y", "x", "z")
#: Mirror ``CondEpisode.COND_MODES`` / ``DZ_ANCHORS`` — best first, so each dropdown's initial
#: value is also the recommended one.
_COND_MODES = ("home", "absolute")
_DZ_ANCHORS = ("chest", "hand", "train")

_PANEL_MD = """\
**Online G1 motion completion** — the recorded 3 tracker points completed to a 29-DoF
whole-body motion, one window at a time as the timeline plays.

| | what it is |
|---|---|
| 🤖 **G1** | the completion, drawn in THIS rig's `g1_29dof_yam.urdf` (yam jaws + F/T) |
| 🔵 / 🩷 **triads + trails** | the L / R `gripper_base` pose the model was CONDITIONED on |
| 🟠 **path + triad** | the root cond: chest XY + heading (the only chest channels used) |
| 🟡 **markers** | the keypoints handed to the low-level controller for the selected mode |

The blue/pink triads are **input**, the robot is **output**, and they must coincide — that
agreement IS the coordinate check, and `wrist err` below measures it.

In the default `home` start the triads sit at the **robot's** hands, not the operator's: the
motion is re-based onto the G1's 29-joints-zero pose, so the robot stands on the grid and the
recorded gripper cluster is offset below it. Switch `start from` to `absolute` to put the
triads back onto the recorded cluster (the fidelity view).
"""


class MotionCompletionOverlay:
    """GUI panel + scene nodes + the streaming driver. One instance per viewer."""

    def __init__(self, server, *, enabled: bool = False, urdf_path: str | None = None,
                 log=None, name: str = "/mc", wrist_ik: bool = True, foot_ik: bool = True,
                 foot_ik_iters: int | None = None, init_cond_mode: str = "home",
                 init_dz_anchor: str = "chest", panel_md: str | None = None,
                 folder_label: str = "Motion completion (G1 whole body)"):
        self.server = server
        self.name = name
        self.enabled = bool(enabled)
        self.urdf_path = urdf_path or _DEFAULT_URDF
        self.log = log or (lambda m: print(f"[mc] {m}", flush=True))
        # HOST-FACING presentation + placement default. All three exist for FLoRe's offline
        # inference viewer (flore/s1_policy/inference/mc_humanoid.py), which drives this panel
        # from a POLICY's predicted chunk rather than a recording:
        #   * init_cond_mode — that viewer's display frame has ALREADY normalised the take to
        #     the rig's standing heights (its DISPLAY_HOME_POS *is* RIG_{WRIST,CHEST}_STAND_Z),
        #     so `absolute` + the chest anchor derives dz ~= 0 there and puts the hands on the
        #     drawn chunk. Here there is a real libsurvive origin to bridge, so `home` stays
        #     the default and this repo's behaviour is unchanged.
        #   * panel_md / folder_label — the intro text names the DATA ("the recorded 3 tracker
        #     points"), which is wrong for a predicted chunk.
        if init_cond_mode not in _COND_MODES:
            raise ValueError(f"init_cond_mode must be one of {_COND_MODES}, "
                             f"got {init_cond_mode!r}")
        self._init_cond_mode = str(init_cond_mode)
        # The z anchor is the OTHER half of the placement, so it gets the same treatment: the
        # constructor arg is the panel's initial value AND the no-panel fallback, or a headless
        # driver asking for `absolute`+`hand` would silently be completed at `chest` — a
        # different commanded hand height, i.e. a different posture.
        if init_dz_anchor not in _DZ_ANCHORS:
            raise ValueError(f"init_dz_anchor must be one of {_DZ_ANCHORS}, "
                             f"got {init_dz_anchor!r}")
        self._init_dz_anchor = str(init_dz_anchor)
        self._panel_md = _PANEL_MD if panel_md is None else str(panel_md)
        self._folder_label = str(folder_label)
        # Initial state of the two post-process toggles. Settable from the CLI so an A/B is a
        # second launch rather than a click sequence — the foot IK's effect is the kind of thing
        # you compare in two windows side by side, and a clicked-together comparison is not
        # reproducible.
        self._init_wrist_ik = bool(wrist_ik)
        self._init_foot_ik = bool(foot_ik)
        self._init_foot_iters = foot_ik_iters
        self.backend = None
        self.disabled_reason = "" if self.enabled else "not requested (--motion-completion)"
        self.contract: dict = {}
        self.body_names: list[str] = []
        self.joint_names: list[str] = []

        # scene handles, created on the first episode
        self._robot = None
        self._robot_root = None
        self._urdf_joint_names: list[str] = []
        self._cond_nodes: dict = {}
        self._static: list = []
        self._kp_axes = None
        self._kp_labels: list = []

        # episode state, written on the loop thread only
        self.cond = None
        self._qpos = None
        self._fk_pos = None
        self._fk_quat = None
        self._filled = None
        self._next_w = 0
        self._last_drawn = -1
        self._lat: list = []
        self._errors: list = []
        #: Bumped on every :meth:`on_episode`. Jobs and results carry it, and a result whose
        #: epoch is stale is DROPPED — otherwise a window still in flight when the operator
        #: switches episode lands in the new episode's buffer, and two same-length episodes
        #: would make that silent (the shape guard alone cannot catch it).
        self._epoch = 0

        # worker
        self._jobs: queue.Queue = queue.Queue()
        self._results: queue.Queue = queue.Queue()
        self._worker: threading.Thread | None = None
        self._stop = threading.Event()

        # GUI flags
        self._restart_requested = False
        self._gui = {}
        #: How to REBUILD the current conditioning, so the "Re-run" button can apply changed
        #: knobs from the loop thread. Set by :meth:`on_episode` (zarr) or :meth:`on_cond`
        #: (any other producer) — one mechanism, because the knobs it must re-read are the
        #: same either way.
        self._restart_build = None

        if not self.enabled:
            return
        from mc_backend_client import MCBackend  # noqa: PLC0415
        self.backend = MCBackend(log=self.log)
        if not self.backend.available:
            self.disabled_reason = self.backend.disabled_reason
            self.log(f"DISABLED: {self.disabled_reason}")
            return
        if not self.backend.start():
            self.disabled_reason = self.backend.disabled_reason
            return
        self.contract = dict(self.backend.contract)
        self.body_names = list(self.contract["body_names"])
        self.joint_names = list(self.contract["joint_names"])
        self._start_worker()

    # ------------------------------------------------------------------ status
    @property
    def available(self) -> bool:
        return self.enabled and self.backend is not None and not self.disabled_reason

    # ------------------------------------------------------------------ worker
    def _start_worker(self) -> None:
        self._worker = threading.Thread(target=self._work, daemon=True, name="mc-windows")
        self._worker.start()

    def _work(self) -> None:
        """The ONLY thread that talks to the backend. Never touches viser."""
        pushed = 0
        n = 0
        cond = None
        epoch = -1
        while not self._stop.is_set():
            try:
                job = self._jobs.get(timeout=0.2)
            except queue.Empty:
                continue
            kind = job[0]
            try:
                if kind == "episode":
                    _, epoch, cond, pp = job
                    n, pushed = cond.n, 0
                    # The post-process settings are read on the LOOP thread and carried in the
                    # job, not read off the GUI here: this is the worker thread, and a checkbox
                    # flipped mid-episode would change the pass halfway through a stream.
                    self.backend.start_episode(n, **pp)
                elif kind == "window":
                    _, job_epoch, w = job
                    if job_epoch != epoch:
                        continue      # queued before the episode switch; the state is gone
                    if cond is None:
                        raise RuntimeError("window before episode")
                    # Push exactly the cond frames this window needs and no more — the
                    # streaming property is what makes this an online emulation rather than
                    # a batch run dressed up as one.
                    need = min(cond.starts[w] + int(self.contract["window"]), n)
                    if need > pushed:
                        sl = slice(pushed, need)
                        self.backend.push(cond.left_pos[sl], cond.left_quat[sl],
                                          cond.right_pos[sl], cond.right_quat[sl],
                                          cond.root_pos[sl], cond.root_quat[sl])
                        pushed = need
                    t0 = time.perf_counter()
                    res = self.backend.run_window(w)
                    res["wall_ms"] = (time.perf_counter() - t0) * 1e3
                    self._results.put(("window", epoch, w, res))
                elif kind == "quit":
                    return
            except Exception as exc:  # noqa: BLE001 - report, keep the worker alive
                self._results.put(("error", job, f"{type(exc).__name__}: {exc}"))

    # ------------------------------------------------------------------ GUI
    def build_gui(self, gui) -> None:
        if not self.enabled:
            return                      # no flag -> no panel at all
        with gui.add_folder(self._folder_label):
            if not self.available:
                gui.add_text("status", initial_value="DISABLED", disabled=True)
                gui.add_text("reason", initial_value=self.disabled_reason[:110], disabled=True)
                gui.add_text("what it needs", disabled=True, initial_value=(
                    "a Motion_Completion checkout + its g1_mosyn conda env + a downloaded "
                    "checkpoint (MOTION_COMPLETION_{DIR,PY,CKPT})"))
                return
            gui.add_markdown(self._panel_md)
            c = self.contract
            gui.add_text("checkpoint", initial_value=str(c["ckpt"])[:44], disabled=True)
            gui.add_text("contract", disabled=True, initial_value=(
                f"T={c['window']} ov={c['overlap']} stride={c['stride']} "
                f"{c['flow_infer_strategy']} {c['wrist_body']} {c['device']}"))
            gui.add_text("deploy command lag", disabled=True, initial_value=(
                f"{c['deployment_lag_frames']} frames (~"
                f"{c['deployment_lag_frames'] / 30.0:.2f} s) — dense cond over the window"))
            gui.add_text("post-process", disabled=True, initial_value=(
                f"wrist IK {'on' if c.get('wrist_ik') else 'off'} · foot IK "
                f"{c.get('foot_fix', '?')} ({c.get('foot_ik_iters', '?')} it, look-back "
                f"{c.get('foot_lookback', '?')} fr)"))
            g = self._gui
            g["z_lift"] = gui.add_text("z lift", initial_value="-", disabled=True)
            g["progress"] = gui.add_text("windows", initial_value="0/0", disabled=True)
            g["latency"] = gui.add_text("latency", initial_value="-", disabled=True)
            g["wrist_err"] = gui.add_text("wrist err (this frame)", initial_value="-",
                                          disabled=True)
            g["posture"] = gui.add_text("posture", initial_value="-", disabled=True)
            g["foot"] = gui.add_text("foot slide", initial_value="-", disabled=True)

            with gui.add_folder("Post-process (needs a re-run)"):
                gui.add_markdown(
                    "**wrist IK** solves the arms onto the commanded `gripper_base` pose each "
                    "window — it is what turns a few-mm flow output into the 0.3 mm `wrist "
                    "err` below. ~20 ms.\n\n"
                    "**foot IK** is MC's contact-lock + inertialize + leg-IK pass, adapted to "
                    "run **causally**: a rolling look-back of already-corrected frames plus the "
                    "new ones, keeping only the new tail, so it never rewrites a frame already "
                    "sent. It only touches leg DoFs, so it cannot undo the wrist IK.\n\n"
                    "⚠️ `foot IK iters` is a **convergence threshold, not a dial**. Measured "
                    "end to end: off &rarr; slide 0.031 m/s at 6% duty · 40 &rarr; 0.032 at "
                    "88% (costs 13&times;, buys nothing) · 80 &rarr; 0.0077 at 163% · **150 "
                    "(MC's own) &rarr; 0.0036 at 331%**. Below ~80 the leg IK never reaches "
                    "the locked targets. So no real-time setting helps, and at the default "
                    "the humanoid **trails the play head** — play once and let it catch up.")
                g["wrist_ik"] = gui.add_checkbox("wrist IK",
                                                 initial_value=self._init_wrist_ik)
                g["foot_ik"] = gui.add_checkbox("foot IK (causal)",
                                                initial_value=self._init_foot_ik)
                g["foot_iters"] = gui.add_slider(
                    "foot IK iters", 5, 200, 5,
                    int(self._init_foot_iters or c.get("foot_ik_iters", 150)))
                g["foot_lookback"] = gui.add_slider("foot look-back (fr)", 0, 90, 5,
                                                    int(c.get("foot_lookback", 45)))

            with gui.add_folder("Draw"):
                g["show_robot"] = gui.add_checkbox("G1 humanoid", initial_value=True)
                g["show_cond"] = gui.add_checkbox("conditioning (in)", initial_value=True)
                g["show_trails"] = gui.add_checkbox("cond trails", initial_value=True)
                g["show_kp"] = gui.add_checkbox("controller keypoints (out)",
                                                initial_value=True)
                g["lag_view"] = gui.add_checkbox(
                    f"deployment lag view ({c['deployment_lag_frames']} fr)",
                    initial_value=False)
                # Inert in `home` mode, where dz is 0 and the robot already stands on the grid.
                g["stand_on_grid"] = gui.add_checkbox(
                    "stand on the grid (absolute only)", initial_value=False)

            with gui.add_folder("Low-level handoff"):
                g["mode"] = gui.add_dropdown("keypoints", _MODES, initial_value="3pt")
                g["hand_point"] = gui.add_dropdown("hand point", _HAND_POINTS,
                                                  initial_value="tcp")
                gui.add_text("hand point", disabled=True, initial_value=(
                    "tcp = 0.242 m fingertip (every current ckpt) · lever = 0.18 m +-25 mm"))

            with gui.add_folder("Conditioning (needs a re-run)"):
                gui.add_markdown(
                    "**Where the motion starts.** The commanded hand height is a "
                    "*conditioned* channel, so getting the absolute placement wrong is not a "
                    "display problem — it tells the model to reach somewhere it cannot, and "
                    "the model obeys by crouching or straining.\n\n"
                    "`home` (default) starts the completion at the G1's **29 joints = 0** "
                    "pose — forearms level along +x, jaws parallel 0.297 m apart, pelvis "
                    "standing at 0.7919 — placed at the chest puck's frame-0 XY + heading, "
                    "and transfers only the operator's *change* in pose from there. Nothing "
                    "is guessed: the floor is the robot's own, so it stands ON the grid. The "
                    "price is that the anchor is per hand, so the operator's measured "
                    "hand-to-hand offset is replaced by the robot's — the same trade the "
                    "rig's own display home-anchor makes.\n\n"
                    "`absolute` keeps the take's world poses and bridges the two floors with "
                    "a rigid **z lift** = how high the libsurvive origin (the grid) sat above "
                    "the floor, which nobody measured. It is the FIDELITY mode: the hands land "
                    "back on the recorded eef path. Measured over 22 windows, sweeping only "
                    "the lift: hands at 0.645 m &rarr; pelvis 0.45 (knee 1.8); 0.95 &rarr; "
                    "0.785; 1.25 &rarr; 0.800. Posture saturates above ~0.9 so the model "
                    "cannot pick the anchor for you. The `z anchor` / `pin z lift` controls "
                    "below apply to **`absolute` only**.")
                g["cond_mode"] = gui.add_dropdown("start from", _COND_MODES,
                                                  initial_value=self._init_cond_mode)
                g["dz_anchor"] = gui.add_dropdown("z anchor (absolute)", _DZ_ANCHORS,
                                                  initial_value=self._init_dz_anchor)
                g["dz_mode"] = gui.add_checkbox("pin z lift (absolute)", initial_value=False)
                g["dz"] = gui.add_slider("z lift (m)", -0.5, 2.5, 0.01, 1.20)
                gui.add_markdown(
                    "The rotation offset is MC's own `--umi_wrist_rot_axis/deg`, kept as a "
                    "probe of the grip convention. The chain is EXACT at 0 deg (sub-mm round "
                    "trip); non-zero also rotates the drawn grip and moves the TCP keypoint "
                    "~0.34 m, so it is a diagnostic, not fidelity.")
                g["rot_axis"] = gui.add_dropdown("rot axis (gripper local)", _ROT_AXES,
                                                 initial_value="y")
                g["rot_deg"] = gui.add_slider("rot deg", -180.0, 180.0, 5.0, 0.0)
                g["restart"] = gui.add_button("Re-run this episode")

                @g["restart"].on_click
                def _(_e) -> None:
                    # Flag only; the loop thread performs the reset (this repo's rule).
                    self._restart_requested = True

    # ------------------------------------------------------------------ episode
    def on_episode(self, z, ep: int) -> None:
        """Build the conditioning for episode ``ep`` and reset the streaming state.

        Loop thread. Safe to call when unavailable (it becomes a no-op), so the caller does
        not have to branch.
        """
        from mc_episode import CondEpisode  # noqa: PLC0415

        self.on_cond(lambda **kw: CondEpisode.from_zarr(z, ep, **kw))

    def on_cond(self, build) -> None:
        """Build the conditioning through ``build`` and reset the streaming state.

        The PRODUCER-AGNOSTIC entry point :meth:`on_episode` is a two-line wrapper around.
        ``build`` is called with the panel's live conditioning knobs as keyword arguments —
        ``cond_mode``, ``dz_wrist``, ``dz_anchor``, ``window``, ``overlap`` — i.e. exactly
        :meth:`CondEpisode.from_zarr`'s / :meth:`~CondEpisode.from_poses`' signature, so a
        caller passes a partial and does not re-read the GUI itself. It is retained for the
        "Re-run" button, which must rebuild with the CHANGED knobs from this thread.

        Existing to let FLoRe's offline inference viewer drive this panel from a policy's
        predicted action chunk without a second copy of the streaming loop.

        Loop thread. A no-op when unavailable, so the caller does not have to branch.
        """
        if not self.available:
            return
        import mc_frames as F  # noqa: PLC0415

        self._restart_build = build
        self._drain_results()
        g = self._gui
        dz = float(g["dz"].value) if (g and g["dz_mode"].value) else None
        anchor = str(g["dz_anchor"].value) if g else self._init_dz_anchor
        # The no-panel fallback is `self._init_cond_mode`, NOT a hardcoded "home": the
        # constructor arg has to mean the same thing with and without a GUI, or a headless
        # driver (a smoke test, a batch export) silently gets the OTHER placement — which
        # changes the posture, since the commanded hand height is a conditioned channel. The
        # same reasoning covers the post-process settings a few lines below.
        mode = str(g["cond_mode"].value) if g else self._init_cond_mode
        try:
            cond = build(
                cond_mode=mode, dz_wrist=dz, dz_anchor=anchor,
                window=int(self.contract["window"]),
                overlap=int(self.contract["overlap"]))
        except SystemExit as exc:
            self.disabled_reason = str(exc)
            self.log(f"DISABLED: {self.disabled_reason}")
            return
        for w in cond.warnings():
            self.log(f"WARNING: {w}")

        deg = float(g["rot_deg"].value) if g else 0.0
        axis = str(g["rot_axis"].value) if g else "y"
        if deg != 0.0:
            cond = _rotate_cond(cond, axis, deg, F)
            self.log(f"conditioning rotated {deg:+.0f} deg about gripper-local {axis}")

        self.cond = cond
        self._epoch += 1
        n, nb = cond.n, len(self.body_names)
        self._qpos = np.zeros((n, 36), dtype=np.float64)
        self._fk_pos = np.zeros((n, nb, 3), dtype=np.float64)
        self._fk_quat = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (n, nb, 1))
        self._filled = np.zeros(n, dtype=bool)
        self._next_w = 0
        self._last_drawn = -1
        self._lat, self._errors = [], []
        self.log(cond.describe())
        if g:
            if cond.cond_mode == "home":
                sep = float(np.linalg.norm(cond.wrist_left_pos[0] - cond.wrist_right_pos[0]))
                g["z_lift"].value = (
                    f"home · start = 29 joints at 0 (wrists {sep * 1e3:.0f} mm apart, pelvis "
                    f"{F.PELVIS_STAND_Z:.3f}) · commanded hands "
                    f"{cond.left_pos[:, 2].mean():.2f} m · floor IS the grid")
            else:
                g["z_lift"].value = (
                    f"absolute · {cond.dz_wrist:+.3f} m · {cond.dz_anchor} · commanded hands "
                    f"{cond.left_pos[:, 2].mean():.2f} m · floor at z="
                    f"{-cond.dz_wrist:+.3f} in the recorded frame")
        self._ensure_scene()
        self._draw_static()
        if g:
            pp = {"wrist_ik": bool(g["wrist_ik"].value),
                  "foot_fix": "inertialize" if g["foot_ik"].value else "none",
                  "foot_ik_iters": int(g["foot_iters"].value),
                  "foot_lookback": int(g["foot_lookback"].value)}
        else:
            # Same rule as `mode` above: with no panel the CONSTRUCTOR's settings apply. An
            # empty dict leaves the BACKEND's launch defaults, which are already this
            # integration's chosen ones (wrist_ik=True, foot_fix=inertialize — MC's upstream
            # defaults are the opposite), so this is not about turning the passes on. It is
            # about a caller that asked to turn one OFF: wrist_ik=False / foot_ik=False /
            # foot_ik_iters=N were silently dropped whenever no panel had been built, so a
            # headless A/B measured the same configuration twice.
            pp = {"wrist_ik": bool(self._init_wrist_ik),
                  "foot_fix": "inertialize" if self._init_foot_ik else "none"}
            if self._init_foot_iters is not None:
                pp["foot_ik_iters"] = int(self._init_foot_iters)
        self.log(f"post-process: wrist IK {'on' if pp['wrist_ik'] else 'off'} · foot IK "
                 f"{pp['foot_fix']} ({pp.get('foot_ik_iters', 'ckpt default')} it, look-back "
                 f"{pp.get('foot_lookback', 'ckpt default')} fr)")
        self._jobs.put(("episode", self._epoch, cond, pp))

    def clear(self) -> None:
        """Hide everything. Called when the viewer switches episodes."""
        if not self.available:
            return
        self.cond = None
        self._filled = None
        for h in self._static:
            _safe_remove(h)
        self._static = []
        for h in list(self._cond_nodes.values()):
            if hasattr(h, "visible"):
                h.visible = False
        if self._robot_root is not None:
            self._robot_root.visible = False
        if self._kp_axes is not None:
            self._kp_axes.visible = False

    def stop(self) -> None:
        self._stop.set()
        self._jobs.put(("quit",))
        if self.backend is not None:
            self.backend.stop()

    # ------------------------------------------------------------------ scene
    def _ensure_scene(self) -> None:
        s = self.server.scene
        if self._robot is None:
            from viser.extras import ViserUrdf  # noqa: PLC0415
            # ViserUrdf needs a pathlib.Path (a str is treated as a yourdfpy.URDF object
            # and trips an assert) — the same trap collection_viz documents.
            self._robot_root = s.add_frame(f"{self.name}/robot", show_axes=False)
            self._robot = ViserUrdf(self.server, urdf_or_path=Path(self.urdf_path),
                                    root_node_name=f"{self.name}/robot")
            self._urdf_joint_names = list(self._robot.get_actuated_joint_names())
            self.log(f"URDF {os.path.basename(self.urdf_path)}: "
                     f"{len(self._urdf_joint_names)} actuated joints "
                     f"(model drives {len(self.joint_names)}, by NAME)")
        if not self._cond_nodes:
            for key, col, lab in (("L", _COLOR_L, "cond L gripper_base"),
                                  ("R", _COLOR_R, "cond R gripper_base")):
                self._cond_nodes[key] = s.add_frame(
                    f"{self.name}/cond_{key}", show_axes=True, axes_length=0.12,
                    axes_radius=0.006, origin_radius=0.012)
                s.add_label(f"{self.name}/cond_{key}/label", lab)
            self._cond_nodes["root"] = s.add_frame(
                f"{self.name}/cond_root", show_axes=True, axes_length=0.16,
                axes_radius=0.007, origin_radius=0.014)
            s.add_label(f"{self.name}/cond_root/label",
                        "cond root (chest XY + yaw, on the floor)")

    def _draw_static(self) -> None:
        """The cond trails. Rebuilt per episode / per display-mode change."""
        for h in self._static:
            _safe_remove(h)
        self._static = []
        g = self._gui
        if not (g and g["show_cond"].value and g["show_trails"].value):
            return
        c = self.cond
        for key, arr, col in (("L", c.left_pos, _COLOR_L), ("R", c.right_pos, _COLOR_R),
                              ("root", c.root_pos, _COLOR_ROOT)):
            to_disp = self._root_to_display if key == "root" else self._to_display
            p = to_disp(arr).astype(np.float32)
            if p.shape[0] < 2:
                continue
            self._static.append(self.server.scene.add_line_segments(
                f"{self.name}/trail_{key}", points=np.stack([p[:-1], p[1:]], axis=1),
                colors=col, line_width=2.0))

    # ------------------------------------------------------------------ driving
    def _to_display(self, p: np.ndarray) -> np.ndarray:
        """Model-frame position -> the frame the rest of the scene is drawn in."""
        g = self._gui
        if g and g["stand_on_grid"].value:
            return np.asarray(p, dtype=np.float64)
        return self.cond.unlift(p)

    def _floor_z(self) -> float:
        """Where the model's z=0 (the FLOOR) lands in the frame we are drawing in."""
        return float(self._to_display(np.zeros((1, 3)))[0, 2])

    def _root_to_display(self, p: np.ndarray) -> np.ndarray:
        """The root cond, drawn ON THE GROUND — the same choice MC's own viewer makes.

        The model reads only XY + heading off this slot (its z is zeroed by the cond mask,
        verified by sweeping the z and getting the same output), so drawing it at the chest
        puck's height states a conditioning input that does not exist and invites exactly the
        wrong diagnosis. ``visualizer_app/viser_motion_completion.py`` draws the pelvis path
        and its heading triads at z=0.02; this keeps that offset and moves it to wherever the
        floor is in the current display frame.
        """
        out = self._to_display(p)
        out[..., 2] = self._floor_z() + _ROOT_DRAW_Z
        return out

    def _drain_results(self) -> bool:
        """Absorb finished windows into the episode buffers. Loop thread. True if anything
        changed (the caller then forces a re-render)."""
        changed = False
        while True:
            try:
                item = self._results.get_nowait()
            except queue.Empty:
                break
            if item[0] == "error":
                _, job, msg = item
                self._errors.append(msg)
                self.log(f"window job {job[:3]} failed: {msg}")
                continue
            _, epoch, w, res = item
            if epoch != self._epoch or self._qpos is None:
                continue                       # in flight when the episode switched
            a = res["arrays"]
            lo, k = int(res["start"]), int(res["n"])
            if lo + k > self._qpos.shape[0]:
                continue
            self._qpos[lo:lo + k, :3] = a["root_pos"]
            self._qpos[lo:lo + k, 3:7] = a["root_quat"]
            self._qpos[lo:lo + k, 7:36] = a["joint_angles"]
            self._fk_pos[lo:lo + k] = a["fk_pos"]
            self._fk_quat[lo:lo + k] = a["fk_quat"]
            self._filled[lo:lo + k] = True
            self._lat.append(res)
            changed = True
        return changed

    def poll(self, head_frame: int) -> bool:
        """Advance the online loop. Call from the viewer's loop thread each iteration.

        Returns True when the scene should be re-rendered (a window landed).
        """
        if not self.available:
            return False
        if self._restart_requested:
            self._restart_requested = False
            if self._restart_build is not None:
                self.log("re-running the episode with the current conditioning knobs")
                self.on_cond(self._restart_build)
                return True
        if self.cond is None:
            return False
        changed = self._drain_results()

        import mc_frames as F  # noqa: PLC0415
        T, ov = int(self.contract["window"]), int(self.contract["overlap"])
        w_target = F.window_for_frame(int(head_frame), self.cond.starts)
        # Windows MUST run in order: window w's anchor seed is built by w-1, and running out
        # of order silently drops the seam continuity the anchor-mix ckpt was trained for.
        while self._next_w <= w_target:
            self._jobs.put(("window", self._epoch, self._next_w))
            self._next_w += 1
            changed = True
        if self._gui:
            done = int(self._filled.sum()) if self._filled is not None else 0
            self._gui["progress"].value = (
                f"{self._next_w}/{len(self.cond.starts)} run · {done}/{self.cond.n} frames"
                + (f" · {len(self._errors)} ERR" if self._errors else ""))
            if self._lat:
                def _mean(k):
                    return float(np.mean([r.get(k, 0.0) for r in self._lat]))
                tot = _mean("t_total_ms")
                span_ms = (T - ov) / 30.0 * 1e3
                duty = tot / span_ms * 100.0
                self._gui["latency"].value = (
                    f"{tot:.0f} ms/window (flow {_mean('t_flow_ms'):.0f} + wrist IK "
                    f"{_mean('t_ik_ms'):.0f} + foot IK {_mean('t_foot_ms'):.0f}) · "
                    f"{duty:.0f}% duty of the {span_ms:.0f} ms it covers"
                    + ("  ← over real time" if duty > 100.0 else ""))
                # The foot pass's own before/after, so the cost above has a benefit beside it.
                sb = [r["slide_before"] for r in self._lat if r.get("foot")]
                sa = [r["slide_after"] for r in self._lat if r.get("foot")]
                self._gui["foot"].value = (
                    f"{float(np.median(sb)):.4f} → {float(np.median(sa)):.4f} m/s planar "
                    f"(MC eval: 0.0574 unfixed → 0.0246 fixed)" if sb else
                    "foot IK off — feet slide (MC eval: 0.0574 m/s unfixed)")
        return changed

    def render_at(self, frame: int) -> None:
        """Draw the completion at ``frame``. Loop thread."""
        if not self.available or self.cond is None or self._filled is None:
            return
        import mc_frames as F  # noqa: PLC0415
        g = self._gui
        n = self.cond.n
        eff = int(frame)
        if g and g["lag_view"].value:
            eff -= int(self.contract["deployment_lag_frames"])
        eff = int(np.clip(eff, 0, n - 1))
        if not self._filled[eff]:
            # Nothing completed for this frame yet. Hold the last drawn pose rather than
            # snapping to the origin, and say so.
            if g:
                g["wrist_err"].value = "waiting for its window"
            return

        show_robot = bool(g["show_robot"].value) if g else True
        if self._robot_root is not None:
            self._robot_root.visible = show_robot
        if show_robot:
            q = self._qpos[eff]
            self._robot_root.position = tuple(self._to_display(q[None, :3])[0])
            self._robot_root.wxyz = tuple(q[3:7])
            # The jaws come from the TAKE, not the model: the completion has no gripper
            # channel, and drawing them shut would hide the grasp the demo actually made.
            jaws = {}
            for side, key in (("left", "left"), ("right", "right")):
                wid = self.cond.grip_width.get(key)
                # No jaw channel for this hand -> render it OPEN (1.0), never 0.0. 0.0 is the
                # CLOSED end of the travel, so it drew an unmeasured hand as a hand HOLDING
                # something. Same reasoning as mc_frames.urdf_cfg_from_qpos's jaw default.
                frac = (float(np.clip(wid[min(eff, len(wid) - 1)] / _WIDTH_OPEN_M, 0.0, 1.0))
                        if wid is not None else 1.0)
                jaws[side] = frac * _JAW_OPEN_M
            self._robot.update_cfg(F.urdf_cfg_from_qpos(
                q[7:36], self.joint_names, self._urdf_joint_names,
                jaw_left=jaws["left"], jaw_right=jaws["right"]))

        show_cond = bool(g["show_cond"].value) if g else True
        for key, pos, quat in (("L", self.cond.left_pos, self.cond.left_quat),
                               ("R", self.cond.right_pos, self.cond.right_quat),
                               ("root", self.cond.root_pos, self.cond.root_quat)):
            node = self._cond_nodes.get(key)
            if node is None:
                continue
            node.visible = show_cond
            if show_cond:
                to_disp = self._root_to_display if key == "root" else self._to_display
                node.position = tuple(to_disp(pos[eff][None])[0])
                node.wxyz = tuple(quat[eff])

        self._render_keypoints(eff, F)
        self._render_readouts(eff, F)
        self._last_drawn = eff

    def _render_keypoints(self, eff: int, F) -> None:
        g = self._gui
        if not (g and g["show_kp"].value):
            if self._kp_axes is not None:
                self._kp_axes.visible = False
            for h in self._kp_labels:
                h.visible = False
            return
        kp = F.select_keypoints(self._fk_pos[eff:eff + 1], self._fk_quat[eff:eff + 1],
                                self.body_names, mode=str(g["mode"].value),
                                hand_point=str(g["hand_point"].value))
        pos = self._to_display(kp["pos"][0]).astype(np.float32)
        wxyz = kp["quat"][0].astype(np.float32)
        # One batched-axes node re-sent per frame: viser has no "move a batch" op, and n is
        # <= 5, so a rebuild is cheaper than five individual frames' worth of messages.
        self._kp_axes = self.server.scene.add_batched_axes(
            f"{self.name}/keypoints", batched_wxyzs=wxyz, batched_positions=pos,
            axes_length=0.14, axes_radius=0.008)
        if len(self._kp_labels) != len(kp["names"]):
            for h in self._kp_labels:
                _safe_remove(h)
            self._kp_labels = [
                self.server.scene.add_label(f"{self.name}/kp_label_{i}", nm)
                for i, nm in enumerate(kp["names"])]
        for h, p in zip(self._kp_labels, pos):
            h.visible = True
            h.position = (float(p[0]), float(p[1]), float(p[2]) + 0.06)

    def _render_readouts(self, eff: int, F) -> None:
        g = self._gui
        if not g:
            return
        # The round trip: FK wrist -> mount forward -> compare to the cond we sent. This is
        # the number that says the chain is right; ~57 mm means the mount was double-applied
        # or skipped, ~242 mm means the TCP lever was missed.
        parts = []
        for side, body, tgt in (("L", "left_wrist_yaw_link", "left"),
                                ("R", "right_wrist_yaw_link", "right")):
            b = self.body_names.index(body)
            p_g, R_g = F.mc_gripper_from_wrist(
                self._fk_pos[eff:eff + 1, b],
                F.quat_wxyz_to_R(self._fk_quat[eff:eff + 1, b]), tgt)
            tp = getattr(self.cond, f"{tgt}_pos")[eff]
            tR = F.quat_wxyz_to_R(getattr(self.cond, f"{tgt}_quat")[eff])
            dp = float(np.linalg.norm(p_g[0] - tp)) * 1e3
            dR = R_g[0] @ tR.T
            ang = float(np.rad2deg(np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1))))
            parts.append(f"{side} {dp:.1f} mm / {ang:.2f} deg")
        g["wrist_err"].value = " · ".join(parts)
        pel = float(self._qpos[eff, 2])
        kn = 0.5 * (self._qpos[eff, 7 + self.joint_names.index("left_knee_joint")]
                    + self._qpos[eff, 7 + self.joint_names.index("right_knee_joint")])
        # The PLANTED foot, i.e. the lower of the two — a stepping foot swings to 0.26 m.
        ank = min(float(self._fk_pos[eff, self.body_names.index(f"{s}_ankle_roll_link"), 2])
                  for s in ("left", "right"))
        # Both directions the commanded hand height can be wrong, named, because that is what
        # an operator can act on. NOTE the asymmetry: a crouch means the z lift is too low and
        # is a calibration bug; tiptoe at a lift you have verified is the CHECKPOINT reaching
        # above its training band (gripper z 0.61-0.72), which no lift can fix.
        verdict = ("CROUCH — commanded hands too low (check the z lift)" if pel < 0.70
                   else "on tiptoe — hands above the ckpt's training band"
                   if ank > _ANKLE_FLOAT_M else "standing")
        g["posture"].value = (f"pelvis z {pel:.3f} (G1 stands ~0.79) · knee {kn:+.2f} rad "
                              f"· planted foot z {ank:+.3f} · {verdict}")


# ---------------------------------------------------------------------------
def _rotate_cond(cond, axis: str, deg: float, F):
    """Post-multiply both hands' cond rotation by ``R_axis(deg)`` in the GRIPPER's frame.

    The equivalent of MC's ``--umi_wrist_rot_axis`` / ``--umi_wrist_rot_deg``, and applied
    the same way (post-multiplied, i.e. in the body's LOCAL frame). Note the SIGN differs
    from MC's real-UMI recipe (they pass ``y -90``): theirs acts on the RAW PUCK frame to
    convert it into the gripper convention, while ours acts on an already-correct
    ``gripper_base`` frame. Different starting frame, so the numbers are not comparable —
    which is exactly why this is a separate, separately-labelled knob rather than a reuse.
    """
    th = np.deg2rad(float(deg))
    c, s = np.cos(th), np.sin(th)
    Rm = {"x": np.array([[1, 0, 0], [0, c, -s], [0, s, c]]),
          "y": np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]),
          "z": np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])}[str(axis).lower()]
    for side in ("left", "right"):
        key = f"{side}_quat"
        R = F.quat_wxyz_to_R(getattr(cond, key))
        setattr(cond, key, F.R_to_quat_wxyz(np.einsum("tij,jk->tik", R, Rm)))
    return cond


def _safe_remove(handle) -> None:
    try:
        handle.remove()
    except Exception:  # noqa: BLE001 - a viewer must not die removing a stale node
        pass


__all__ = ["MotionCompletionOverlay"]
