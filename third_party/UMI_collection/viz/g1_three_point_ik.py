"""Three-point (left hand + right hand + head/torso) mink IK for the 3pt
collection viz — the MODE=3pt counterpart of :class:`g1_right_hand_ik.RightHandIK`.

Design constraints (from the humi_ik solver internals):

- The stock :class:`ikumi.solver.IKSolver` is hardwired to FIVE targets
  (``HandRootFootKeyPoints``: 2 hands + 2 FEET + root). We do NOT call its
  ``solve_one_step``; like ``RightHandIK`` we drive the task targets directly
  and call ``mink.solve_ik`` ourselves. The foot/pelvis/COM tasks keep the home
  (stand) targets they were built with — that IS the "constant stand-pose foot
  targets" synthesis.
- There is NO torso/waist task in the stock solver, and the G1 has no neck
  joint (``head_link`` is a fixed geom on ``torso_link``). So "head tracking"
  here means: a NEW mink ``FrameTask`` on the ``torso_link`` body, orientation-
  dominant, fed from the head tracker via the same home-anchored delta mapping
  the hands use. Physically the third tracker sits on the upper torso/head;
  holosoma's 3pt controller tracks ``torso_link + (0,0,0.35)`` the same way.

  ⚠️ **Yaw-only in THIS MJCF.** The HuMI MJCF's waist chain has ONLY
  ``waist_yaw_joint`` (waist_roll/waist_pitch are yam-URDF-only inert links
  here). With the base pinned, ``torso_link`` is controllable in YAW only:
  head-tracker pitch/roll and translation produce ~zero-Jacobian residuals the
  QP silently ignores (harmless — the soft cost just goes unmet). The recorded
  head-tracker DATA is still full 6-DoF (the recorder stores the raw pose);
  only the live *visualization* is yaw-limited. The real 3pt-hybrid controller
  on the robot tracks the full orientation through whole-body motion.
- Both arms + the waist hinges move; the LEG hinges are velocity-frozen and the
  floating base is hard re-asserted to home each inner step (same technique and
  rationale as ``RightHandIK(right_arm_only=True)`` — the QP otherwise recruits
  the base/legs long before the posture regularizer resists).
- Calibration: per-role anchors. ``recalibrate`` captures all three tracker
  poses at once (operator holds the calibration pose: hands in front separated
  L/R, head level facing +x — same as the holosoma 3pt bridge) and maps each
  onto its robot home frame (left_hand_site / right_hand_site / torso_link).

Consumed by ``collection_viz.py --tracking-mode 3pt``; launched via
``MODE=3pt ./run_collection_viz.sh``.
"""
from __future__ import annotations

import numpy as np

from g1_right_hand_ik import quat_inv, quat_mul

# Lazy heavyweight imports (mirrors g1_right_hand_ik): populated in __init__.
mujoco = None
SE3 = SO3 = solve_ik = NoSolutionFound = FrameTask = None

#: role -> (robot home frame, frame_type) each tracker maps onto.
_ROLE_FRAMES = {
    "left": ("left_hand_site", "site"),
    "right": ("right_hand_site", "site"),
    "head": ("torso_link", "body"),
}

ROLES = ("left", "right", "head")

# ~0 but nonzero so ConfigurationLimit/collision always keep feasibility slack
# (identical to RightHandIK's freeze). Exact 0 can render the QP infeasible.
_FREEZE_VEL = 1e-4  # rad/s


class ThreePointIK:
    """HuMI mink IK driving both hands + the torso from three VIVE trackers."""

    def __init__(self, config_name: str = "proposal", reach_scale: float = 1.0,
                 arm_max_velocity: float = 3.0,
                 torso_position_cost: float = 2.0,
                 torso_orientation_cost: float = 10.0,
                 active_roles: tuple[str, ...] = ROLES):
        # active_roles selects which of ('left','right','head') this instance
        # drives — a PREFIX of ROLES with the primary hand ('left') first, so the
        # SAME class serves 1pt (('left',)), 2pt (('left','right')) and 3pt (all).
        # Inactive roles' tasks simply hold their home target (never re-targeted),
        # and recalibrate/step only require/consume the active ones. The primary
        # (main-arm) role is always active_roles[0] == 'left' (LEFT-canonical).
        bad = [r for r in active_roles if r not in ROLES]
        if bad or not active_roles or active_roles[0] != "left":
            raise ValueError(
                f"active_roles must be a non-empty prefix of {ROLES} starting with "
                f"'left' (LEFT-canonical); got {active_roles!r}")
        self._active_roles = tuple(active_roles)
        global mujoco, SE3, SO3, solve_ik, NoSolutionFound, FrameTask
        import mujoco  # noqa: PLC0415
        from mink import SE3, SO3  # noqa: PLC0415
        from mink import solve_ik  # noqa: PLC0415
        from mink.exceptions import NoSolutionFound  # noqa: PLC0415
        from mink.tasks import FrameTask  # noqa: PLC0415
        from humi_data_collection.ikumi.config import get_config  # noqa: PLC0415

        self._cfg = get_config(config_name)
        # Same guard as RightHandIK(right_arm_only=True): a pinned base + frozen
        # legs is only consistent with a standing config (no COM driver).
        com_xy = float(getattr(self._cfg, "com_xy_cost", 0.0))
        com_z = float(getattr(self._cfg, "com_z_cost", 0.0))
        assert com_xy == 0.0 and com_z == 0.0, (
            "ThreePointIK pins the base and freezes the legs, which is only valid "
            f"with the COM task disabled; config '{config_name}' enables COM "
            f"(com_xy_cost={com_xy}, com_z_cost={com_z})."
        )
        self._solver = self._cfg.create()
        self._reach_scale = float(reach_scale)
        self._arm_max_velocity = float(arm_max_velocity)
        self._home_base_q = self._solver._home_q[:7].copy()

        # NEW torso task (orientation-dominant): the stock solver has no torso/
        # waist task at all. Append it to the solver's task list so our solve_ik
        # call sees it alongside hands/feet/posture/pelvis.
        self._torso_task = FrameTask(
            frame_name="torso_link",
            frame_type="body",
            position_cost=float(torso_position_cost),
            orientation_cost=float(torso_orientation_cost),
            gain=self._cfg.hand_gain,
            lm_damping=self._cfg.hand_damping,
        )
        self._torso_task.set_target(
            self._solver.configuration.get_transform_frame_to_world(
                frame_name="torso_link", frame_type="body"))
        self._solver._tasks.append(self._torso_task)

        self._role_task = {
            "left": self._solver._left_hand_task,
            "right": self._solver._right_hand_task,
            "head": self._torso_task,
        }
        # Robot home frame pose per role (world), from the stand keyframe.
        self._home_pose = {
            role: self._solver.configuration.get_transform_frame_to_world(
                frame_name=frame, frame_type=ftype)
            for role, (frame, ftype) in _ROLE_FRAMES.items()
        }

        # Per-role calibration anchors (None until recalibrate()).
        self._ref_pos: dict[str, np.ndarray] = {}
        self._ref_quat_inv: dict[str, np.ndarray] = {}
        # Per-role puck->wrist MOUNT rotation, captured by reset_axes_to_wrist(). Empty
        # until then, and every consumer treats "absent" as identity, so the default
        # behavior is bit-identical to before this existed.
        self._mount_quat: dict[str, np.ndarray] = {}
        self._mount_quat_inv: dict[str, np.ndarray] = {}
        self._calibrated = False

        # Compat attrs that collection_viz.py reads on any IK instance (used for
        # YamConfigMapper + mock feeder home pos). Expose the same names as
        # RightHandIK so the caller doesn't need to branch.
        self._humi_joint_names = self._collect_humi_joint_names()
        self._home_rh = self._home_pose["right"]

        self._rebuild_velocity_limit()

    def _collect_humi_joint_names(self) -> list[str]:
        """HuMI hinge joint names (same helper as RightHandIK)."""
        names = []
        m = self._solver.model
        for jid in range(m.njnt):
            jname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, jid)
            if jname is None:
                continue
            if int(m.jnt_type[jid]) in (int(mujoco.mjtJoint.mjJNT_FREE),
                                        int(mujoco.mjtJoint.mjJNT_BALL)):
                continue
            names.append(jname)
        return names

    @property
    def active_roles(self) -> tuple[str, ...]:
        """The roles this instance drives (prefix of ROLES, primary 'left' first)."""
        return self._active_roles

    @property
    def home_right_hand(self):
        """Robot home right-hand SE3 (compat with RightHandIK consumers)."""
        return self._home_rh

    @property
    def home_primary_hand(self):
        """Robot home SE3 of the PRIMARY (main-arm) hand. LEFT-canonical, so this
        is the left-hand site — what the mock feeder circles + the camera rides."""
        return self._home_pose["left"]

    # -- velocity limit: arms + waist free, legs frozen ----------------------
    def _rebuild_velocity_limit(self) -> None:
        from mink.limits import VelocityLimit  # noqa: PLC0415

        m = self._solver.model
        free_prefixes = ("left_shoulder_", "left_elbow_", "left_wrist_",
                         "right_shoulder_", "right_elbow_", "right_wrist_",
                         "waist_")
        velocities: dict[str, np.ndarray] = {}
        for jid in range(m.njnt):
            jname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, jid)
            if jname is None:
                continue
            if int(m.jnt_type[jid]) in (int(mujoco.mjtJoint.mjJNT_FREE),
                                        int(mujoco.mjtJoint.mjJNT_BALL)):
                continue  # free base: pinned in step(), not velocity-limited
            if any(jname.startswith(p) for p in free_prefixes):
                if self._arm_max_velocity > 0:
                    velocities[jname] = np.full(1, self._arm_max_velocity, dtype=np.float64)
            else:
                # Legs (and any future hinge outside arms+waist) are frozen.
                velocities[jname] = np.full(1, _FREEZE_VEL, dtype=np.float64)
        new_limit = VelocityLimit(model=m, velocities=velocities)
        # Replace in the solver's limit list (same slot juggling as RightHandIK).
        sol = self._solver
        replaced = [new_limit if lim is sol._velocity_limit else lim for lim in sol._limits]
        # Guard the swap actually landed: if a humi_ik refactor stops registering
        # _velocity_limit in _limits, this would otherwise silently no-op and the
        # legs would unfreeze (the QP then recruits them against the pinned base).
        assert any(lim is new_limit for lim in replaced), (
            "velocity-limit swap failed: solver._velocity_limit not found in "
            "solver._limits (humi_ik internals changed?)"
        )
        sol._limits = replaced
        sol._velocity_limit = new_limit

    # -- calibration ----------------------------------------------------------
    @property
    def calibrated(self) -> bool:
        return self._calibrated

    def recalibrate(self, poses: dict[str, tuple[np.ndarray, np.ndarray]]) -> None:
        """Capture the ACTIVE tracker poses as the reference mapping onto the
        robot home frames. ``poses`` = {role: (pos[3], quat_wxyz[4])} and must
        contain every role in :attr:`active_roles` (a partial calibration of an
        active limb would leave it anchored to garbage). Inactive roles are
        ignored and keep their home target."""
        missing = [r for r in self._active_roles if r not in poses]
        if missing:
            raise ValueError(f"recalibrate needs all active roles {self._active_roles}; missing {missing}")
        for role in self._active_roles:
            pos, quat = poses[role]
            self._ref_pos[role] = np.asarray(pos, dtype=float).copy()
            self._ref_quat_inv[role] = quat_inv(np.asarray(quat, dtype=float))
        self._solver.reset()
        self._calibrated = True

    def reset_axes_to_wrist(self, poses: dict[str, tuple[np.ndarray, np.ndarray]]
                            ) -> dict[str, list[float]]:
        """Re-zero each active role's VISUALIZED axes onto the wrist, axes aligned.

        The puck is bolted to the handle at an arbitrary orientation, so its raw axes
        do not line up with the wrist's — a tracker triad drawn in the scene points
        somewhere diagonal, and the operator has no way to tell a mount rotation from a
        world-alignment error. This captures the CURRENT tracker orientation as the
        role's mount rotation, so from here on the displayed triad (and the recorded
        pose, once the extrinsic is applied offline) reads as axis-aligned when the hand
        is held in this reference posture.

        Returns ``{role: quat_wxyz}`` — the captured mount rotation per role — so the
        caller can PERSIST it with the take (a mount rotation that lives only in RAM is
        lost on restart, and a session recorded under a different mount is
        indistinguishable from one recorded under this mount).

        This composes with, and does not replace, :meth:`recalibrate`: the home-anchored
        delta mapping already absorbs a mount TRANSLATION, but the rotation delta is
        taken in the WORLD frame, so a rotated mount is NOT cancelled (which is exactly
        why the world-alignment knob exists). Calling this makes the reference posture's
        orientation the identity for the display.
        """
        missing = [r for r in self._active_roles if r not in poses]
        if missing:
            raise ValueError(f"reset_axes_to_wrist needs all active roles "
                             f"{self._active_roles}; missing {missing}")
        out: dict[str, list[float]] = {}
        for role in self._active_roles:
            _pos, quat = poses[role]
            q = np.asarray(quat, dtype=float)
            n = float(np.linalg.norm(q))
            q = q / (n if n > 1e-9 else 1.0)
            self._mount_quat[role] = q.copy()
            self._mount_quat_inv[role] = quat_inv(q)
            out[role] = [float(x) for x in q]
        return out

    def mount_quats(self) -> dict[str, list[float]]:
        """Captured mount rotations, ``{role: quat_wxyz}``. Empty until a reset."""
        return {r: [float(x) for x in q] for r, q in self._mount_quat.items()}

    def display_quat(self, role: str, quat_wxyz: np.ndarray) -> np.ndarray:
        """Tracker orientation with the captured mount rotation removed.

        Identity-preserving: with no captured mount this returns the input unchanged,
        so the scene is bit-identical to before any reset.
        """
        inv = self._mount_quat_inv.get(role)
        if inv is None:
            return np.asarray(quat_wxyz, dtype=float)
        return quat_mul(np.asarray(quat_wxyz, dtype=float), inv)

    def reset_pose(self) -> None:
        self._solver.reset()

    def invalidate_calibration(self) -> None:
        self._calibrated = False

    def current_q(self) -> np.ndarray:
        return self._solver.configuration.q.copy()

    def home_q(self) -> np.ndarray:
        return self._solver.home_configuration.q.copy()

    # -- per-step targets + solve --------------------------------------------
    def _tracker_to_target(self, role: str, pos: np.ndarray, quat_wxyz: np.ndarray):
        """Home-anchored delta mapping (same math as RightHandIK, per role)."""
        cur_p = np.asarray(pos, dtype=float)
        cur_q = np.asarray(quat_wxyz, dtype=float)
        delta_world = quat_mul(cur_q, self._ref_quat_inv[role])
        home = self._home_pose[role]
        target_q = quat_mul(delta_world, home.rotation().wxyz)
        nq = float(np.linalg.norm(target_q))
        if nq > 1e-9:
            target_q = target_q / nq
        target_p = home.translation() + self._reach_scale * (cur_p - self._ref_pos[role])
        return SE3.from_rotation_and_translation(
            rotation=SO3(wxyz=target_q), translation=target_p)

    def step(self, poses: dict[str, tuple[np.ndarray, np.ndarray]],
             dt: float | None = None) -> np.ndarray:
        """Advance the IK toward the live 3-tracker targets; return HuMI q (32).

        ``poses`` = {role: (pos[3], quat_wxyz[4])}. A role may be ABSENT or None
        (its tracker went stale/missing): that task simply keeps its previous
        target — hold semantics per limb, matching the holosoma bridge.
        """
        if not self._calibrated:
            return self._solver.configuration.q.copy()
        for role in self._active_roles:
            entry = poses.get(role)
            if entry is None:
                continue
            pos, quat = entry
            self._role_task[role].set_target(self._tracker_to_target(role, pos, quat))
        outer_dt = float(self._cfg.dt if dt is None else dt)
        inner_dt = outer_dt / max(self._cfg.inner_steps, 1)
        q_before = self._solver.configuration.q.copy()
        try:
            for _ in range(self._cfg.inner_steps):
                v = solve_ik(
                    configuration=self._solver.configuration,
                    tasks=self._solver._tasks,
                    dt=inner_dt,
                    solver=self._cfg.solver,
                    limits=self._solver._limits,
                    damping=self._cfg.damping,
                )
                self._solver.configuration.integrate_inplace(v, inner_dt)
                # Hard-pin the floating base (see RightHandIK.step for the full
                # rationale: VelocityLimit can't cap a free joint; PostureTask
                # zeros it; without the pin the base drifts under tracker yaw).
                q = self._solver.configuration.q.copy()
                q[:7] = self._home_base_q
                self._solver.configuration.update(q=q)
        except NoSolutionFound:
            self._solver.configuration.update(q=q_before)
        return self._solver.configuration.q.copy()


__all__ = ["ThreePointIK", "ROLES"]
