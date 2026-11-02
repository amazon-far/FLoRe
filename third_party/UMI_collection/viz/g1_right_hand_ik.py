#!/usr/bin/env python3
"""Right-hand-only G1 IK driven by a single VIVE tracker, rendered on the yam URDF.

Wraps HuMI's mink ``IKSolver`` (vendored at ``third_party/humi_ik``) but
drives ONLY the right-hand frame task: the left hand, both feet, the pelvis/root
and the COM tasks keep the targets that ``IKSolver.__init__`` set from the
``stand`` keyframe, so the whole body except the right arm holds the standing
pose and the floating base never drifts (verified: a +8 cm right-hand step moves
the hand and leaves ``q[0:3]`` at the stand height).

Two coordinate facts make this work with one tracker:

  * **Home-anchoring lives here, not in the bridge.** The tracker bridge streams
    a raw yaw-aligned world SE3. On calibration we capture (tracker_ref,
    robot_home_right_hand_site) and thereafter map the live tracker pose to a
    right-hand target by mirroring its displacement/rotation since the reference
    onto the robot's home hand pose:

        R_target = ΔR_world · R_home ,  ΔR_world = R_cur · R_ref⁻¹
        p_target = p_home + reach_scale · (p_cur − p_ref)

    So at the calibration instant the target == the home hand pose (robot holds
    still), and live motion is mirrored 1:1 (world-frame delta, like the policy
    bridge's orientation handling). No puck-mounting measurement is needed for
    position; orientation tracks the puck's own rotation about world axes.

  * **25 → yam joint remap.** HuMI's MJCF exposes 25 actuated hinge joints; the
    yam URDF has 30 actuated (non-mimic) joints — it adds waist_roll, waist_pitch
    and both wrist_yaw (revolute) plus the two gripper finger1 prismatic joints
    (finger2 is ``<mimic>``-driven, set by yourdfpy automatically). We map HuMI's
    q[7:] by NAME into the yam actuated order, default the 4 yam-only revolutes
    to 0 (their stand value), and fill finger1 from the live gripper fraction.

Used by ``collection_viz.py``. Requires ``mink`` + ``mujoco`` (installed into the
bota env) and ``humi_data_collection`` on ``PYTHONPATH``.
"""
from __future__ import annotations

import os as _os
from dataclasses import dataclass

import numpy as np

# NOTE: mink / mujoco / humi_data_collection are imported LAZILY inside
# ``RightHandIK`` (the only part that needs them), so ``GripperRange``,
# ``YamConfigMapper`` and ``yam_actuated_joint_names`` stay usable with just
# yourdfpy — handy for the eval-gate's gripper-mapping checks and unit tests.

# The yam URDF the viz renders. The right-hand site geometry comes from HuMI's
# MJCF (which the solver loads); we only need the joint-name set from the URDF.
# Resolved relative to this file (UMI_collection/viz/) so it is CWD-independent.
YAM_URDF_DEFAULT = _os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)),
    "..", "robot_assets", "g1_yam", "g1_29dof_yam.urdf",
)

# yam-only revolute joints HuMI's 25-joint solver does NOT produce. Their stand
# value is 0, so a static default keeps the rendered pose consistent.
_YAM_ONLY_REVOLUTE_DEFAULTS = {
    "waist_roll_joint": 0.0,
    "waist_pitch_joint": 0.0,
    "left_wrist_yaw_joint": 0.0,
    "right_wrist_yaw_joint": 0.0,
}

# Gripper prismatic joints (finger1 is the driven DOF; finger2 is <mimic>).
_GRIPPER_FINGER1 = ("left_gripper_finger1_joint", "right_gripper_finger1_joint")


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


def quat_inv(q: np.ndarray) -> np.ndarray:
    """Inverse of a unit wxyz quaternion (bare conjugate)."""
    return np.array([q[0], -q[1], -q[2], -q[3]])


@dataclass
class GripperRange:
    """Encoder travel endpoints -> URDF prismatic [0, jaw_open_m] mapping.

    q=0 is fingertips CLOSED, q=jaw_open_m is OPEN (confirmed from the URDF
    comment). The encoder reports sim-frame rad in [closed_rad, open_rad]; we map
    the open FRACTION linearly onto the jaw stroke.
    """

    open_rad: float = 4.90
    closed_rad: float = 0.0
    jaw_open_m: float = 0.0475  # URDF prismatic upper limit

    def fraction(self, encoder_rad: float) -> float:
        """Open fraction in [0, 1] from a live encoder reading."""
        span = self.open_rad - self.closed_rad
        if abs(span) < 1e-9:
            return 0.0
        f = (encoder_rad - self.closed_rad) / span
        return float(min(max(f, 0.0), 1.0))

    def jaw_m(self, encoder_rad: float) -> float:
        """URDF finger1 prismatic value (m) for a live encoder reading."""
        return self.fraction(encoder_rad) * self.jaw_open_m


def yam_actuated_joint_names(urdf_path: str = YAM_URDF_DEFAULT) -> list[str]:
    """Actuated (non-mimic) joint names of the yam URDF, in yourdfpy order.

    This is the order ``ViserUrdf.update_cfg`` expects for an array config.
    """
    import os

    import yourdfpy

    robot = yourdfpy.URDF.load(
        urdf_path,
        load_meshes=False,
        build_scene_graph=False,
        filename_handler=lambda fname: os.path.join(os.path.dirname(urdf_path), fname),
    )
    return list(robot.actuated_joint_names)


class RightHandIK:
    """HuMI mink IK, driving only the right-hand frame task from one tracker."""

    def __init__(self, config_name: str = "proposal", reach_scale: float = 1.0,
                 arm_max_velocity: float = 3.0, right_arm_only: bool = True):
        # Lazy import: only RightHandIK needs the heavy IK stack.
        global mujoco, SE3, SO3, solve_ik, NoSolutionFound
        import mujoco  # noqa: PLC0415
        from mink import SE3, SO3, solve_ik  # noqa: PLC0415
        from mink.exceptions import NoSolutionFound  # noqa: PLC0415

        from humi_data_collection.ikumi.config import get_config  # noqa: PLC0415

        # 'proposal' uses home=stand, posture=stand, com disabled — a clean
        # standing base for upper-body-only tracking.
        cfg = get_config(config_name)
        self._cfg = cfg
        self._solver = self._cfg.create()
        self._reach_scale = float(reach_scale)
        self._arm_max_velocity = float(arm_max_velocity)

        # right_arm_only pins the whole body EXCEPT the right arm: every other
        # hinge (left arm, both legs, waist incl. waist_yaw) is velocity-frozen
        # in the QP, and the floating base is hard re-asserted to its home pose
        # after every inner integrate. WHY: the stock HuMI whole-body solver is a
        # single unconstrained weighted-least-squares QP over all nv=31 DOFs in
        # which left_hand/feet/pelvis are SOFT costs and posture is a negligible
        # regularizer that structurally cannot touch the free base (mink's
        # PostureTask zeros the free-joint DOFs in its error+Jacobian). The
        # right-hand FrameTask Jacobian is nonzero on the base + waist_yaw, so the
        # cost-900 hand pull trivially overpowers the cost-16 pelvis pin and
        # recruits waist_yaw + the unpinned free base — visibly dragging the left
        # hand and both legs even though only the right tracker moved. Freezing
        # the non-right-arm hinges + pinning the base enforces the viz's "the
        # floating base never drifts" invariant as a near-hard constraint instead
        # of that soft tug-of-war. Set False to recover stock whole-body behaviour.
        #
        # The freeze is only CORRECT when the base is meant to stand, i.e. the COM
        # driver is disabled (proposal). A walk/squat config re-enables COM/foot
        # drivers, and "frozen base + active COM driver" is genuinely infeasible —
        # guard so a future config can't silently misbehave.
        self._right_arm_only = bool(right_arm_only)
        if self._right_arm_only:
            com_xy = float(getattr(self._cfg, "com_xy_cost", 0.0))
            com_z = float(getattr(self._cfg, "com_z_cost", 0.0))
            assert com_xy == 0.0 and com_z == 0.0, (
                "right_arm_only freezes the legs/waist and pins the floating base, "
                "which is only valid with the COM task disabled (a standing base); "
                f"config '{config_name}' enables COM (com_xy_cost={com_xy}, "
                f"com_z_cost={com_z}). Pass right_arm_only=False for whole-body configs."
            )
            # Free-base home slice q[0:7] = [x y z | qw qx qy qz], captured once for
            # the hard base re-assert in step(). Taken from the solver's stand home.
            self._home_base_q = self._solver._home_q[:7].copy()

        # Cap the arm/waist joint speed in the IK QP. The stock HuMI solver only
        # velocity-limits the LOWER-LIMB joints (see solver._build_velocity_limit),
        # so when the right-hand target is pushed to the arm's reach boundary the
        # QP can jump to a flipped shoulder solution in a single 0.02 s outer step
        # — the operator sees the hand rise, then "snap" down past a threshold.
        # Capping arm+waist velocity makes the boundary graceful (the hand stops
        # at max extension and holds); continuous in-reach teleop deltas are far
        # below the cap, so normal motion is unchanged. Verified: eliminates the
        # snap on both a pure vertical lift and an orientation-coupled lift.
        # Rebuild the VelocityLimit when we either cap the arm (snap protection)
        # OR freeze the non-right-arm hinges (right_arm_only) — both live in the
        # same rebuild. With right_arm_only and arm_max_velocity<=0 the right-arm
        # joints are left uncapped but every other hinge is still frozen.
        if self._arm_max_velocity > 0 or self._right_arm_only:
            self._augment_arm_velocity_limit()

        # Robot home right-hand site pose (world), from the stand keyframe.
        self._home_rh: SE3 = self._solver.configuration.get_transform_frame_to_world(
            frame_name="right_hand_site", frame_type="site"
        )

        # Calibration reference (set by recalibrate()); identity until first cal.
        self._ref_pos = np.zeros(3)
        self._ref_quat_inv = np.array([1.0, 0.0, 0.0, 0.0])
        self._calibrated = False

        # HuMI joint names in q[7:] order (the 25 hinge joints).
        self._humi_joint_names = self._collect_humi_joint_names()

    # -- HuMI joint introspection ------------------------------------------
    def _collect_humi_joint_names(self) -> list[str]:
        m = self._solver.model
        names = []
        for i in range(m.njnt):
            if int(m.jnt_type[i]) == int(mujoco.mjtJoint.mjJNT_HINGE):
                n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i)
                if n:
                    names.append(n)
        return names

    def _augment_arm_velocity_limit(self) -> None:
        """Add the arm + waist joints to the solver's VelocityLimit.

        The stock ``IKSolver._build_velocity_limit`` only caps the lower-limb
        joints (hips/knees/ankles). The right-arm chain is therefore unbounded in
        the QP, which lets a reach-boundary singularity snap the shoulder to a
        flipped solution in one outer step. We rebuild the VelocityLimit with the
        SAME lower-limb caps PLUS the arm + waist joints at ``arm_max_velocity``,
        and swap it into the live ``_limits`` list in place (the solver appends the
        velocity limit at index 1, after the configuration limit)."""
        from mink.limits import VelocityLimit  # noqa: PLC0415

        sol = self._solver
        m = sol.model
        lower_prefixes = ("left_hip_", "left_knee_", "left_ankle_",
                          "right_hip_", "right_knee_", "right_ankle_")
        arm_prefixes = ("left_shoulder_", "left_elbow_", "left_wrist_",
                        "right_shoulder_", "right_elbow_", "right_wrist_", "waist_")
        # The six right-arm hinges (HuMI MJCF) — the ONLY joints allowed to move
        # under a right-hand-only tracker. right_wrist_yaw does NOT exist in the
        # MJCF (yam-URDF-only, defaulted to 0 by YamConfigMapper), so it is
        # correctly absent here. Matched by prefix.
        right_arm_prefixes = ("right_shoulder_", "right_elbow_", "right_wrist_")
        # ~0 but nonzero so the joint_limit_gain=0.95 ConfigurationLimit and the
        # collision limit always retain feasibility slack (a ±FREEZE_VEL·dt box
        # always contains Δq=0); an exact 0 cap could force QP infeasibility.
        FREEZE_VEL = 1e-4  # rad/s
        velocities: dict[str, np.ndarray] = {}
        for jid in range(m.njnt):
            jname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, jid)
            if jname is None:
                continue
            if int(m.jnt_type[jid]) in (int(mujoco.mjtJoint.mjJNT_FREE),
                                        int(mujoco.mjtJoint.mjJNT_BALL)):
                # The free base CANNOT be velocity-limited (mink VelocityLimit
                # raises on free joints); it is pinned in step() instead.
                continue
            if self._right_arm_only:
                # Only the right-arm hinges keep a real cap; EVERY other hinge
                # (left arm, both legs, waist incl. waist_yaw) is frozen to ~0.
                # Any future MJCF hinge not in the right-arm set defaults to
                # frozen — the safe direction.
                if any(jname.startswith(p) for p in right_arm_prefixes):
                    # Cap the right arm only when arm_max_velocity>0 (snap
                    # protection); otherwise leave it uncapped (no dict entry).
                    if self._arm_max_velocity > 0:
                        velocities[jname] = np.full(
                            1, self._arm_max_velocity, dtype=np.float64)
                else:
                    velocities[jname] = np.full(1, FREEZE_VEL, dtype=np.float64)
            else:
                if any(jname.startswith(p) for p in lower_prefixes):
                    velocities[jname] = np.full(1, self._cfg.max_joint_velocity, dtype=np.float64)
                elif any(jname.startswith(p) for p in arm_prefixes):
                    velocities[jname] = np.full(1, self._arm_max_velocity, dtype=np.float64)
        new_limit = VelocityLimit(model=m, velocities=velocities)
        # Replace the solver's velocity limit (kept at index 1 of _limits). solve_ik
        # reads sol._limits, NOT sol._velocity_limit, so the swap MUST land in the
        # list. Assert it did: if a future humi_ik refactor stopped placing the same
        # VelocityLimit object into _limits, this would silently no-op and the arm
        # would be left uncapped — the reach-boundary snap-drop would silently return.
        replaced = False
        for i, lim in enumerate(sol._limits):
            if lim is sol._velocity_limit:
                sol._limits[i] = new_limit
                replaced = True
                break
        assert replaced, (
            "RightHandIK: could not find the solver's VelocityLimit in _limits to "
            "augment with arm/waist caps; the arm would be left velocity-unbounded. "
            "Did humi_ik's IKSolver change how _limits is built?"
        )
        sol._velocity_limit = new_limit

    @property
    def home_right_hand(self) -> SE3:
        return self._home_rh

    @property
    def calibrated(self) -> bool:
        return self._calibrated

    # -- calibration --------------------------------------------------------
    def recalibrate(self, tracker_pos: np.ndarray, tracker_quat_wxyz: np.ndarray) -> None:
        """Capture the current tracker pose as the reference that maps to the
        robot's home right-hand pose. Call while the operator holds the hand at
        the rest pose. Also resets the solver to the home configuration so the
        next solve starts from standing."""
        self._ref_pos = np.asarray(tracker_pos, dtype=float).copy()
        self._ref_quat_inv = quat_inv(np.asarray(tracker_quat_wxyz, dtype=float))
        self._solver.reset()
        self._calibrated = True

    def reset_pose(self) -> None:
        """Reset the solver to the home (stand) configuration."""
        self._solver.reset()

    def invalidate_calibration(self) -> None:
        """Clear the calibration flag so :meth:`step` holds home until re-anchored.

        Used on an episode reset: after this, ``calibrated`` is False and the next
        live tracker pose must re-``recalibrate`` before the hand tracks again."""
        self._calibrated = False

    def current_q(self) -> np.ndarray:
        """The LAST SOLVED configuration q (length 32), held live in the solver.

        Unlike :meth:`home_q` (a fresh stand keyframe), this is the pose the solver
        most recently integrated to -- so a consumer can HOLD the current hand pose
        on a stale tracker frame instead of snapping back to stand (the viz
        renders ``_last_q`` on a stale frame for exactly this reason)."""
        return self._solver.configuration.q.copy()

    # -- per-step target + solve -------------------------------------------
    def _tracker_to_target(self, tracker_pos: np.ndarray, tracker_quat_wxyz: np.ndarray) -> SE3:
        """Map a live tracker SE3 to a right-hand world target via the home anchor."""
        cur_p = np.asarray(tracker_pos, dtype=float)
        cur_q = np.asarray(tracker_quat_wxyz, dtype=float)
        # World-frame relative rotation since calibration, applied onto home.
        delta_world = quat_mul(cur_q, self._ref_quat_inv)
        home_q = self._home_rh.rotation().wxyz
        target_q = quat_mul(delta_world, home_q)
        nq = float(np.linalg.norm(target_q))
        if nq > 1e-9:
            target_q = target_q / nq
        # Mirror displacement onto the home position.
        target_p = self._home_rh.translation() + self._reach_scale * (cur_p - self._ref_pos)
        return SE3.from_rotation_and_translation(
            rotation=SO3(wxyz=target_q), translation=target_p
        )

    def step(self, tracker_pos: np.ndarray, tracker_quat_wxyz: np.ndarray,
             dt: float | None = None) -> np.ndarray:
        """Advance the IK one outer step toward the live tracker target and
        return the full HuMI configuration q (length 32). Only the right-hand
        task target is updated; all other tasks hold the stand pose."""
        if not self._calibrated:
            # Without a reference, hold home — don't chase an un-anchored pose.
            return self._solver.configuration.q.copy()
        target = self._tracker_to_target(tracker_pos, tracker_quat_wxyz)
        self._solver._right_hand_task.set_target(target)
        outer_dt = float(self._cfg.dt if dt is None else dt)
        inner_dt = outer_dt / max(self._cfg.inner_steps, 1)
        # Snapshot the current configuration so an infeasible QP at the reach
        # boundary (NoSolutionFound) holds the last pose instead of crashing the
        # caller — the live viz (collection_viz.py) calls step() unguarded on the
        # render thread.
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
                if self._right_arm_only:
                    # Hard-pin the floating base: VelocityLimit can't cap a free
                    # joint and PostureTask zeros it in its error+Jacobian, so the
                    # QP can integrate a tiny base velocity each inner step that
                    # accumulates into visible drift (worst under tracker yaw).
                    # Overwrite q[0:7] back to the stand base and re-run FK so the
                    # next inner solve sees the pinned base; the right-arm hinges
                    # keep whatever they just solved. update() runs the same
                    # mj_kinematics the NoSolutionFound branch uses, so collision/
                    # limit FK stays consistent. The re-assert lives INSIDE the
                    # try, so a mid-loop infeasibility still rolls back to q_before.
                    q = self._solver.configuration.q.copy()
                    q[:7] = self._home_base_q
                    self._solver.configuration.update(q=q)
        except NoSolutionFound:
            self._solver.configuration.update(q=q_before)
        return self._solver.configuration.q.copy()

    def home_q(self) -> np.ndarray:
        return self._solver.home_configuration.q.copy()


class YamConfigMapper:
    """Maps HuMI q (32 = 7 base + 25 hinge) onto the yam URDF actuated vector.

    Output is an array in ``yam_actuated_joint_names`` order, ready for
    ``ViserUrdf.update_cfg``. yam-only revolutes default to the stand value (0);
    the two gripper finger1 joints come from the live gripper open fraction.
    """

    def __init__(self, humi_joint_names: list[str], yam_joint_names: list[str]):
        self._humi_names = humi_joint_names
        self._yam_names = yam_joint_names
        # index of each yam joint in HuMI's q[7:], or None if yam-only.
        self._humi_idx_for_yam = []
        for yn in yam_joint_names:
            self._humi_idx_for_yam.append(humi_joint_names.index(yn) if yn in humi_joint_names else None)

    def to_yam(self, humi_q: np.ndarray, left_jaw_m: float = 0.0, right_jaw_m: float = 0.0) -> np.ndarray:
        """Build the yam actuated config vector from a HuMI q + gripper jaws."""
        hinge = humi_q[7:]  # 25 values in HuMI joint order
        out = np.zeros(len(self._yam_names), dtype=np.float64)
        for i, yn in enumerate(self._yam_names):
            hidx = self._humi_idx_for_yam[i]
            if hidx is not None:
                out[i] = hinge[hidx]
            elif yn == _GRIPPER_FINGER1[0]:
                out[i] = left_jaw_m
            elif yn == _GRIPPER_FINGER1[1]:
                out[i] = right_jaw_m
            else:
                out[i] = _YAM_ONLY_REVOLUTE_DEFAULTS.get(yn, 0.0)
        return out


__all__ = [
    "GripperRange",
    "RightHandIK",
    "YamConfigMapper",
    "yam_actuated_joint_names",
    "YAM_URDF_DEFAULT",
]
