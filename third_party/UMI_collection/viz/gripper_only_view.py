#!/usr/bin/env python3
"""Gripper-only display geometry for the data-collection viz (no humanoid, no IK).

``collection_viz.py`` normally renders the whole G1 humanoid driven by mink IK.
In ``--display-mode gripper`` (the default for MODE=1pt/2pt) we instead show ONLY
the three yam-gripper meshes per active hand + the gripper's own frame + the
camera frustum — no arm, no torso, no legs, and crucially **no mink import**.

This module is the pure, hardware-free half of that feature:

  * :func:`build_gripper_geom` pulls the three gripper link meshes (base +
    two fingers) for one hand out of the yam URDF via ``yourdfpy`` — WITH the
    URDF material colours already baked into the trimesh, so the gripper looks
    byte-identical to how ``ViserUrdf`` draws it in humanoid mode — together with
    each link's pose RELATIVE TO THAT HAND'S WRIST-YAW LINK. Because the two
    fingers slide on prismatic joints whose axis is fixed, each finger's
    wrist-relative rotation is CONSTANT across the jaw stroke and its translation
    is EXACTLY LINEAR in the jaw opening. Both are kinematic facts of a fixed-axis
    prismatic joint on a static base, so capturing the closed/open translation
    endpoints and LERPing per frame is exact — no per-frame full-URDF forward
    kinematics on the render hot path. :func:`build_gripper_geom` asserts the two
    PREMISES the LERP rests on (finger rotation constant + base static, checked at
    the two jaw endpoints); given those, the translation linearity follows and needs
    no separate midpoint check.

  * :func:`home_anchor` is the SAME home-anchored delta mapping the IK uses
    (``g1_three_point_ik.ThreePointIK._tracker_to_target``): capture a reference
    tracker pose, then map the live tracker pose to a target by mirroring its
    world-frame displacement/rotation since the reference onto a fixed "home"
    wrist pose. Extracted here as a pure function (numerically identical to the
    IK's, pinned by a test) so the gripper-only scene can place the gripper
    cluster the same way WITHOUT constructing the mink solver.

Kept deliberately free of ``viser`` and ``mink``/``mujoco`` so it imports under
just ``numpy`` + ``yourdfpy`` and is unit-testable with no server or hardware.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from g1_right_hand_ik import GripperRange, quat_inv, quat_mul

# The yam URDF the viz renders (same default as g1_right_hand_ik / collection_viz).
_YAM_URDF_DEFAULT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "robot_assets", "g1_yam", "g1_29dof_yam.urdf",
)

# The three gripper links per hand, and the finger prismatic joints (finger2 is a
# <mimic> of finger1 — yourdfpy sets it automatically, but MuJoCo's loader drops
# <mimic>, so we set BOTH explicitly here to stay loader-independent).
_BASE_LINK = "{side}_rubber_hand"
_FINGER1_LINK = "{side}_gripper_finger1_link"
_FINGER2_LINK = "{side}_gripper_finger2_link"
_FINGER1_JOINT = "{side}_gripper_finger1_joint"
_FINGER2_JOINT = "{side}_gripper_finger2_joint"

# URDF prismatic upper limit (m); q=0 CLOSED, q=_JAW_OPEN_M OPEN. Single source is
# GripperRange.jaw_open_m in g1_right_hand_ik — imported (not re-typed) so a change
# to the URDF stroke there flows here automatically. g1_right_hand_ik lazy-imports
# mink, so this pulls in only numpy/yourdfpy-level code.
_JAW_OPEN_M = GripperRange().jaw_open_m


def _rotmat_to_wxyz(R: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix -> unit wxyz quaternion (Shepperd's method).

    Kept local so this module needs neither viser.transforms nor scipy — it must
    import under just numpy + yourdfpy for the unit tests."""
    R = np.asarray(R, dtype=float)
    t = R[0, 0] + R[1, 1] + R[2, 2]
    if t > 0.0:
        s = 0.5 / np.sqrt(t + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] >= R[1, 1] and R[0, 0] >= R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] >= R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=float)
    n = float(np.linalg.norm(q))
    return q / (n if n > 1e-12 else 1.0)


@dataclass
class _Piece:
    """One gripper mesh + its pose relative to the hand's wrist-yaw link.

    ``mesh`` is a link-local ``trimesh.Trimesh`` (URDF colours baked in). ``quat``
    is the constant wrist-relative rotation (wxyz). ``pos_closed``/``pos_open`` are
    the wrist-relative translations at jaw=0 / jaw=_JAW_OPEN_M; a static base link
    has them equal, and :meth:`pos_at` LERPs between them by the open fraction."""

    name: str
    mesh: object                 # trimesh.Trimesh (kept as object to avoid a hard import here)
    quat: np.ndarray             # wxyz, constant across the jaw stroke
    pos_closed: np.ndarray       # xyz at jaw closed (fraction 0)
    pos_open: np.ndarray         # xyz at jaw open (fraction 1); == pos_closed for the base
    is_finger: bool

    def pos_at(self, open_frac: float) -> np.ndarray:
        """Wrist-relative translation at the given open fraction [0,1] (LERP)."""
        f = float(min(max(open_frac, 0.0), 1.0))
        return self.pos_closed + f * (self.pos_open - self.pos_closed)


@dataclass
class GripperGeom:
    """The three gripper pieces for one hand, all poses wrist-relative."""

    side: str                    # "left" or "right"
    base: _Piece
    finger1: _Piece
    finger2: _Piece

    def pieces(self):
        return (self.base, self.finger1, self.finger2)


def _load_urdf(urdf_path: str):
    import yourdfpy

    return yourdfpy.URDF.load(
        urdf_path, load_meshes=True, build_scene_graph=True,
        # collision meshes not needed; visual scene carries the baked colours.
        load_collision_meshes=False,
    )


def build_gripper_geom(side: str, urdf_path: str = _YAM_URDF_DEFAULT,
                       urdf=None, jaw_open_m: float = _JAW_OPEN_M) -> GripperGeom:
    """Extract one hand's three gripper meshes + wrist-relative poses from the URDF.

    ``side`` is "left" or "right". Pass a preloaded ``yourdfpy.URDF`` as ``urdf``
    to avoid re-parsing (both hands share one load); otherwise it loads from
    ``urdf_path``. Returns a :class:`GripperGeom` whose finger pieces carry the
    closed AND open wrist-relative translations so the caller can LERP the jaw
    without any further FK.

    Asserts the two invariants the LERP relies on — finger rotation constant and
    translation linear across the jaw stroke — so a URDF change that broke them
    fails loudly here instead of silently mis-drawing the jaws.
    """
    if side not in ("left", "right"):
        raise ValueError(f"side must be 'left' or 'right', got {side!r}")
    if urdf is None:
        urdf = _load_urdf(urdf_path)

    wrist = f"{side}_wrist_yaw_link"
    base_link = _BASE_LINK.format(side=side)
    f1_link = _FINGER1_LINK.format(side=side)
    f2_link = _FINGER2_LINK.format(side=side)
    f1_joint = _FINGER1_JOINT.format(side=side)
    f2_joint = _FINGER2_JOINT.format(side=side)

    def _set_jaw(q: float) -> None:
        cfg = {n: 0.0 for n in urdf.actuated_joint_names}
        # Set BOTH finger joints (finger2 mimics finger1, but yourdfpy's mimic
        # handling shouldn't be relied on for the endpoint capture).
        cfg[f1_joint] = q
        cfg[f2_joint] = q
        urdf.update_cfg(cfg)

    def _wrist_rel(link: str) -> np.ndarray:
        # 4x4 transform of `link` expressed in the wrist-yaw frame.
        return np.asarray(urdf.get_transform(link, wrist), dtype=float).copy()

    def _link_local_mesh(link: str):
        # The coloured visual mesh in the LINK's own frame: the URDF-scene geometry
        # (colours baked into face_colors, exactly what ViserUrdf renders) copied and
        # pushed through the visual <origin> so it sits in the link frame. Parenting
        # this under a node placed at the link's wrist-relative pose reproduces the
        # humanoid-mode gripper appearance byte-for-byte.
        visual = urdf.link_map[link].visuals[0]
        gname = visual.geometry.mesh.filename  # e.g. "meshes/yam_gripper_base.STL"
        base = os.path.basename(gname)
        # yourdfpy keys scene geometry by the mesh basename; the right hand is the
        # second instance ("<name>.STL_1"). Pick the instance whose scene-graph
        # parent frame is THIS link so left/right never cross-wire.
        sc = urdf.scene
        chosen = None
        for node in sc.graph.nodes_geometry:
            _T, gkey = sc.graph[node]
            if os.path.basename(str(gkey)).split(".STL")[0] == base.split(".STL")[0]:
                parent = sc.graph.transforms.parents.get(node)
                if parent == link:
                    chosen = gkey
                    break
        if chosen is None:
            raise KeyError(f"no scene geometry found for link {link!r} (mesh {base!r})")
        mesh = sc.geometry[chosen].copy()
        origin = np.asarray(visual.origin, dtype=float) if visual.origin is not None else np.eye(4)
        mesh.apply_transform(origin)
        return mesh

    # Capture wrist-relative poses at both jaw endpoints.
    _set_jaw(0.0)
    base_c = _wrist_rel(base_link)
    f1_c = _wrist_rel(f1_link)
    f2_c = _wrist_rel(f2_link)
    _set_jaw(jaw_open_m)
    base_o = _wrist_rel(base_link)
    f1_o = _wrist_rel(f1_link)
    f2_o = _wrist_rel(f2_link)

    # Premises the LERP rests on (see module docstring): each finger's wrist-relative
    # ROTATION is constant across the stroke, and the BASE link is static. Both are
    # checked at the two jaw endpoints below. Given a fixed-axis prismatic joint these
    # imply the finger TRANSLATION is exactly linear in the jaw opening, so the
    # two-endpoint LERP is exact and no midpoint sample is needed.
    for nm, Tc, To in (("finger1", f1_c, f1_o), ("finger2", f2_c, f2_o)):
        dR = float(np.abs(Tc[:3, :3] - To[:3, :3]).max())
        assert dR < 1e-6, f"{side} {nm}: wrist-relative rotation not constant across jaw (max|dR|={dR:.2e})"
    dbase = float(np.abs(base_c[:3, 3] - base_o[:3, 3]).max())
    assert dbase < 1e-9, f"{side} base link moved with the jaw (|d|={dbase:.2e}) — not a static base"

    def _piece(name, link, Tc, To, is_finger):
        return _Piece(
            name=name,
            mesh=_link_local_mesh(link),
            quat=_rotmat_to_wxyz(Tc[:3, :3]),
            pos_closed=Tc[:3, 3].copy(),
            pos_open=To[:3, 3].copy(),
            is_finger=is_finger,
        )

    return GripperGeom(
        side=side,
        base=_piece("base", base_link, base_c, base_o, is_finger=False),
        finger1=_piece("finger1", f1_link, f1_c, f1_o, is_finger=True),
        finger2=_piece("finger2", f2_link, f2_c, f2_o, is_finger=True),
    )


def home_anchor(home_pos: np.ndarray, home_quat: np.ndarray,
                ref_pos: np.ndarray, ref_quat: np.ndarray,
                cur_pos: np.ndarray, cur_quat: np.ndarray,
                reach_scale: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Home-anchored delta mapping — the SAME math as the IK, minus the solver.

    Mirrors ``ThreePointIK._tracker_to_target`` exactly (pinned numerically by a
    test): the live tracker's world-frame displacement/rotation since the captured
    reference is applied to a fixed "home" wrist pose:

        R_target = (cur * ref^-1) * R_home
        p_target = p_home + reach_scale * (p_cur - p_ref)

    so at ``cur == ref`` the target IS the home pose (gripper holds still on
    calibration), and live motion mirrors 1:1. Returns ``(pos[3], quat_wxyz[4])``.
    """
    cur_q = np.asarray(cur_quat, dtype=float)
    delta_world = quat_mul(cur_q, quat_inv(np.asarray(ref_quat, dtype=float)))
    target_q = quat_mul(delta_world, np.asarray(home_quat, dtype=float))
    nq = float(np.linalg.norm(target_q))
    if nq > 1e-9:
        target_q = target_q / nq
    target_p = (np.asarray(home_pos, dtype=float)
                + float(reach_scale) * (np.asarray(cur_pos, dtype=float) - np.asarray(ref_pos, dtype=float)))
    return target_p, target_q


# Default display-home wrist poses for gripper-only mode, per hand role. There is
# no robot to anchor to, so these are just comfortable, deterministic spots that
# put the gripper in front at ~table height, approach along world +x (matching the
# wrist frame convention the camera mount + gripper mesh offsets assume). The live
# tracker delta is mirrored onto these (see home_anchor); identity rotation keeps
# the gripper pointing +x. Left is front-left, right is front-right.
#
# "head" is the CHEST/torso body reference (holosoma's 3rd point), NOT a wrist — it has no
# gripper, camera or TCP. It exists so a 3pt chest puck anchors to its own spot instead of
# falling back to the LEFT WRIST home, which is what a missing key does (consumers look it
# up as ``.get(role, HOME_WRIST_POSE["left"])``). The value is FLoRe's committed
# ``display.py::DISPLAY_HOME_POS["head"]``, so both replay viewers and the offline inference
# viewer re-base a RECORDED 3pt take onto the SAME arrangement; a test pins the two dicts
# equal, and motion completion's ``mc_frames.RIG_CHEST_STAND_Z`` (not part of the FLoRe
# release) IS this z (the chest anchor
# the completion normalises a take's height against). It is the STANDING OPERATOR's chest —
# 35 cm above the hands — and must stay that: do not repoint it at the robot's body.
#
# The LIVE collection viz is the one consumer that deliberately does NOT use this entry: its
# "Reset robot to home (c)" snaps the rig to the G1's OWN home arrangement, so its chest
# home is the URDF's pelvis instead (``collection_viz._HEAD_HOME_XYZ_M`` /
# ``display_home_pose``, re-derived from the URDF by a test there). Both hands still come
# from here verbatim. Only read in the home-anchored path — RAW_FRAME draws the raw puck
# pose and never looks a home up.
# The two 5pt FEET are the same kind of entry as "head" and for the same reason: a body
# reference with no gripper, camera or TCP, which needs its OWN spot or it renders on the
# left hand (every consumer looks a home up as ``.get(role, HOME_WRIST_POSE["left"])``).
# The values are FLoRe's committed ``display.DISPLAY_HOME_POS["left_foot"/"right_foot"]``
# verbatim — the same pin the three hand/chest entries carry, so a take drawn by the
# replay viewers and by FLoRe's offline inference viewer lands in ONE arrangement. z=0.10
# is a display height, not a floor measurement: this dict is a comfortable, deterministic
# cluster and never a robot configuration (FLoRe's own note on the same table).
HOME_WRIST_POSE = {
    "left": (np.array([0.35, 0.18, 0.95]), np.array([1.0, 0.0, 0.0, 0.0])),
    "right": (np.array([0.35, -0.18, 0.95]), np.array([1.0, 0.0, 0.0, 0.0])),
    "head": (np.array([0.30, 0.0, 1.30]), np.array([1.0, 0.0, 0.0, 0.0])),
    "left_foot": (np.array([0.30, 0.12, 0.10]), np.array([1.0, 0.0, 0.0, 0.0])),
    "right_foot": (np.array([0.30, -0.12, 0.10]), np.array([1.0, 0.0, 0.0, 0.0])),
}


__all__ = ["GripperGeom", "build_gripper_geom", "home_anchor", "HOME_WRIST_POSE"]
