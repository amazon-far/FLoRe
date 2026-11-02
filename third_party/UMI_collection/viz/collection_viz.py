#!/usr/bin/env python
"""Data-collection visualization: G1 + single-tracker right hand + D405 + gripper.

A viser app (built on the ``bota_ft_viz`` patterns) that, for now, VISUALIZES the
data-collection rig — the recording pipeline is a later step. In one 3D scene it:

  1. Renders the G1 humanoid (``g1_29dof_yam.urdf``) and drives ONLY the RIGHT
     hand live from a single VIVE tracker, via HuMI's mink IK (right-hand frame
     task only; left hand / feet / pelvis / base hold the ``stand`` pose). A
     **Recalibrate** button in the right panel captures the operator's current
     hand pose as the reference that maps to the robot's home hand pose.
  2. Shows the RealSense D405 RGB feed on a **camera frustum that follows the
     tracker** (the camera is mounted on the gripper), with a configurable
     camera-to-wrist mount offset.
  3. Mirrors the real gripper open/close in the URDF jaws (encoder fraction ->
     prismatic [0, 0.0475] m) and plots the **live encoder reading** as a
     scrolling chart in the right panel, in the same style as the bota force
     chart.

Topology (hub-and-spoke; this is the hub / WS client):

    [vive_tracker_bridge]  ws://127.0.0.1:8772  ─┐
    [realsense_bridge]     ws://127.0.0.1:8873  ─┼─▶  collection_viz  (viser :8080)
    [keyboard_gripper      ws://127.0.0.1:8774  ─┘     (bota env: viser+mink+yourdfpy)
       --encoder-ws-port]

Each bridge owns one hardware singleton in the env that can talk to it (pika for
tracker+camera, i2rt-kbd for the gripper); this app just consumes their JSON.
Run under the bota env (which now has viser + mink + mujoco + yourdfpy):

    PYTHONPATH=third_party/humi_ik \
      .venv/bin/python viz/collection_viz.py

``--mock`` (no bridges / no IK import) renders the home pose with synthetic
tracker + gripper + camera so the UI runs with nothing attached.

``--display-mode`` selects WHAT is drawn (never what is recorded):
  * ``humanoid`` — the full G1 above, driven by mink IK (historical behavior).
  * ``gripper``  — ONLY the three yam-gripper meshes + the camera frustum per
    active hand, placed by a home-anchored tracker delta with NO mink IK (no
    arm/torso/legs solve, no mink import). See ``gripper_only_view.py``.
  * ``auto`` (default) — ``gripper`` in EVERY tracking mode. 1pt/2pt/3pt all draw
    the same isolated hand cluster(s); 3pt just adds ONE more triad for the chest
    puck. The humanoid is opt-in via ``--display-mode humanoid``.
The camera-to-gripper geometry is the URDF's in BOTH modes (the frustum is
parented under the same follow frame as the gripper meshes).

PER-HAND PANEL (2pt/3pt). The rig has TWO of everything but the chest puck, and
the sidebar is symmetric to match: each hand that is actually present gets its own

  * ``Gripper (left|right)`` folder — open fraction, encoder rad, and
    Open/Close/Stop + a target slider that drive ONLY that hand's jaws (the two
    keyboard_gripper bridges both run ``--web-control``; each is the sole CAN
    writer on its own bus).
  * ``F/T L`` / ``F/T R`` status row and four wrench charts.
  * ``F/T compensation (left|right)`` folder — Tare, the raw/compensated display
    toggle, the bota/rnea engine dropdown and Recompensation, each writing only
    into that sensor's own ``SensorState``. See ``_FTPanel``.

Nothing is shared between the two hands: separate device, separate reader thread,
separate payload + RNEA config, separate tare offset, separate publisher port. The
one exception is RECOMPENSATION, which is arbitrated to one hand at a time (its
overlay is a fullscreen "rotate the tool" prompt and its estimator needs a steady
~1 kHz feed). Recording START and STOP auto-tare EVERY attached sensor, so both
hands' wrench begins each episode zeroed at their own free-space pose.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import math
import os
import signal
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np
import viser
from viser import uplot

# Resolve repo paths relative to this file (UMI_collection/viz/).
_VIZ_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_VIZ_DIR, ".."))

# This dir on sys.path ONCE, guarded, so the sibling `import gripper_only_view` inside the
# per-frame helpers below is a plain sys.modules hit. They used to insert on every call,
# which in RAW_FRAME mode is 3 inserts per rendered frame — sys.path grows without bound
# and every subsequent import walks a longer list.
if _VIZ_DIR not in sys.path:
    sys.path.insert(0, _VIZ_DIR)
_DEFAULT_YAM = os.path.join(
    _REPO_ROOT, "robot_assets", "g1_yam", "g1_29dof_yam.urdf",
)

_AXIS_COLORS = ("#dc3232", "#28b446", "#3c6ef0")

# F/T charts open with a fixed symmetric y-window of [-10, 10] (N and Nm), then
# PEAK-HOLD: the axis only ever widens to track a new magnitude peak, never shrinks
# back (so a transient spike stays visible and the baseline doesn't jitter the axis).
_FT_Y_START = 10.0

# URDF jaw prismatic upper limit (m) — the single source is GripperRange in
# g1_right_hand_ik; take it from there (this file's dir is on sys.path both when run
# as a script and when imported by the tests) so the open-fraction -> jaw conversion
# can't desync from the URDF stroke. Fall back to the literal only if the sibling
# module isn't importable yet. q=0 CLOSED, _JAW_OPEN_M OPEN.
try:
    import os as _os_jaw
    import sys as _sys_jaw
    _sys_jaw.path.insert(0, _os_jaw.path.dirname(os.path.abspath(__file__)))
    from g1_right_hand_ik import GripperRange as _GripperRange
    _JAW_OPEN_M = _GripperRange().jaw_open_m
except Exception:  # noqa: BLE001 - keep the literal as a safe fallback
    _JAW_OPEN_M = 0.0475

# TCP (grasp point) offset along the wrist +x axis, in metres — DISPLAY-ONLY constant for
# drawing the axis triads; the converter/training pipeline reads the PUCK->TCP transform from
# robot_assets/calibration/tcp_extrinsic.json INDEPENDENTLY ((0.225, 0, -0.05) — a DIFFERENT
# LEG) and never imports this value.
#
# 0.242 m is the MEASURED wrist_yaw_link->fingertip-midpoint distance, and the yam URDF agrees:
# FK puts the fingertip meshes at x in [0.1624, 0.2562] in the wrist frame (the closed pinch
# band starts at x ~ 0.2386), so 0.242 lands on the pad. The previous value here was 0.21,
# which sat ~35 mm BEHIND the pad, inside the finger shanks — that is why the triad labelled
# "tcp" did not appear between the drawn fingertips. It is the reference the other two triads
# are measured FROM, so it must not move once correct.
_TCP_ALONG_WRIST_X_M = 0.242

# Two extra display triads, measured RELATIVE TO THE TCP. All are children of /tracker (the
# wrist-yaw follow frame), so their stored positions are in the WRIST frame = TCP_offset +
# (value rel TCP). Display only — nothing recorded changes.
#   * "tracker" — the physical VIVE puck's most accurate origin: from the TCP, back along -x
#     22.5 cm and up +z 5 cm (the measured puck->TCP extrinsic). Lands at wrist + (+0.017, 0,
#     +0.05), which is the puck's true wrist-relative seat.
#   * "wrist_yaw_link" — from the TCP, back along -x 24.2 cm, i.e. EXACTLY (0, 0, 0): the
#     wrist-yaw link origin itself. With the corrected 0.242 datum this coincides with the
#     follow-frame origin (and so with the gripper mesh's own wrist origin) by construction.
#     The old pair (0.21, -0.032) instead left the meshes 32 mm ahead of this triad, which read
#     as a "measurement gap" that was really just the wrong TCP datum.
_TRACKER_TRUE_XYZ_M = (_TCP_ALONG_WRIST_X_M - 0.225, 0.0, 0.05)   # (+0.017, 0, 0.05)
_WRIST_YAW_XYZ_M = (_TCP_ALONG_WRIST_X_M - 0.242, 0.0, 0.0)       # (0, 0, 0)

# Live-tracking BIND POINT (Problem 2A). The home-anchored delta (gripper_only_view.
# home_anchor) yields the TRACKED pose — where the physical VIVE puck is. The operator
# wants that pose to sit on the axis LABELLED "tracker" (the puck's most-accurate
# origin, _TRACKER_TRUE_XYZ_M), not on the wrist origin the follow frame used to carry.
# Since every triad + the gripper meshes + the camera are rigid CHILDREN of the follow
# frame (the wrist origin), we place the follow frame so the puck child lands exactly on
# the tracked pose: follow = tracked ⊕ (wrist-from-puck offset), where the wrist sits at
# the NEGATED puck local offset. Everything else keeps its fixed wrist-relative pose and
# rides along rigidly — the whole cluster now pivots about the puck (physically correct:
# the puck reports its own origin). Display only; recording is a separate process.
_TRACKER_FOLLOW_OFFSET_M = (
    -_TRACKER_TRUE_XYZ_M[0], -_TRACKER_TRUE_XYZ_M[1], -_TRACKER_TRUE_XYZ_M[2],
)  # wrist origin expressed in the puck frame == (0.01, 0, -0.05)

# ---------------------------------------------------------------------------
# RAW-FRAME display option (RAW_FRAME=1 / --raw-frame; gripper-only mode only).
#
# The DEFAULT gripper-only view is home-anchored: the first live tracker frame is
# mapped onto a comfortable HOME_WRIST_POSE in front of the robot (home_anchor +
# follow_pose_binding_puck), so the operator sees the gripper at a pleasant spot
# regardless of where libsurvive seats its world. RAW mode drops ALL of that and
# draws the cluster at the puck's RAW libsurvive pose — the frame the recorder
# actually stores in tracker.npy — so the viz is a faithful twin of the recording:
#
#   * the "tracker" triad sits at the RAW puck pose (position AND orientation),
#   * the yam-gripper TCP position + TCP AXES are the RAW puck put through the
#     -y,-x,-z body relabel + the puck->TCP lever, i.e. the SAME single fused
#     transform verified on the 8088 inspector (tcp_extrinsic_remap):
#         TCP  = puck ∘ [R=M | t=puck->TCP],   with M = relabel(-y,-x,-z)
#   * wrist_yaw_link sits at TCP − 0.242 m along the TCP's own +x (the measured
#     wrist->fingertip datum _TCP_ALONG_WRIST_X_M).
#
# Implementation: the gripper cluster hangs (unchanged) off the /tracker follow
# frame at fixed WRIST-relative child offsets. So in RAW mode we seat that follow
# frame at the WRIST pose derived from the raw puck, and rotate ONLY the puck
# ("tracker") child triad by M^-1 so it reads back the raw puck orientation:
#     follow_q = puck_q ⊗ M_quat            (wrist orientation = puck ∘ M)
#     follow_p = puck_p + R(puck_q)·PUCK_TO_WRIST
#     PUCK_TO_WRIST = −(M · _TRACKER_TRUE_XYZ_M)   (wrist origin in the puck frame)
# Every other child (TCP triad at +0.242x, wrist_yaw at 0, gripper meshes, camera)
# keeps its fixed wrist-relative pose and rides along — proven == the raw-puck
# ground truth to ~1e-15 m / 1e-14 deg over random poses (test_raw_frame_*).
#
# The puck->TCP lever uses the REPO-canonical _TRACKER_TRUE_XYZ_M (puck at wrist
# +(0.017,0,0.05)); the 8088 scratch json used (0,-0.224,0.05), which differs by
# exactly 1 mm — the durable code follows the committed calibration, not the
# scratch file. M is symmetric (M==M.T), a proper 180° rotation (det=+1).
_RAW_RELABEL_M = np.array([[0.0, -1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])

# CHEST puck -> chest BODY frame relabel (the 3pt head keypoint's own mount rotation).
#
# The chest puck is bolted onto the harness rotated, so its RAW libsurvive body axes are
# not the torso reference the 3rd keypoint is meant to express. The measured relabel is
# 90 deg about the puck's own +y, then 90 deg about the resulting +x:
#
#     H = Ry(90) @ Rx(90)      (a proper rotation, det=+1)
#
# i.e. the drawn/stored axes move  x_new = -z_old, y_new = +x_old, z_new = -y_old, which
# puts the chest frame on the SAME axis convention the hands' TCP triads use.
#
# Applied as a BODY-frame right-multiplication, exactly like the hands' puck->TCP
# extrinsic: ``world_T_chest = world_T_puck @ [H | 0]``. The translation is ZERO — the
# chest has no lever (its triad IS the frame origin; there is no wrist or TCP to reach),
# so ``head_pos`` is bit-identical with and without this and ONLY ``head_rot_axis_angle``
# moves.
#
# NOT interchangeable with the hands' M: a different puck on a different bracket, and
# unlike M (symmetric, M@M == I) this one has ORDER 3 (H@H@H == I), so H^-1 != H.
# Kept equal to robot_assets/calibration/head_extrinsic.json — which is what the
# converter and both replay viewers read — by test_logic_collection.
# ``test_head_relabel_matches_the_committed_head_extrinsic``.
_HEAD_RELABEL_M = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, -1.0], [-1.0, 0.0, 0.0]])

# ---------------------------------------------------------------------------
# TRACKING MODES and the BODY-REFERENCE roles.
#
# Literal, not fmt.TRACKER_ROLES_BY_MODE: the viz deliberately does not import
# recording_format (it is a viz process, not a recorder) — same rule as the mount-rotation
# filename below. Kept in sync by test_logic_collection.test_mode_roles_match_the_recorder.
#
# A BODY REFERENCE is a puck and nothing else: no camera, no F/T, no gripper, no gripper
# meshes, no +0.242 m TCP lever, a triad AT the frame origin, and a rotation-only
# puck->body relabel. The chest (3pt) was the first; the two 5pt FEET are the second and
# third, and they are handled by the SAME code paths for exactly that reason — every
# branch below asks "is this role a body reference" instead of comparing to the string
# "head", because a fifth keypoint bolted on as "another head" is one forgotten `elif`
# away from being drawn at libsurvive's raw world origin with nothing to notice it by.
_ROLES_BY_MODE = {
    "1pt": ("left",),
    "2pt": ("left", "right"),
    "3pt": ("left", "right", "head"),
    "5pt": ("left", "right", "head", "left_foot", "right_foot"),
}
#: Roles that carry a gripper cluster + camera (the two HANDS).
_GRIP_ROLES_ALL = ("left", "right")
#: Roles that are body references (see the banner above), in wire order.
_BODY_ROLES_ALL = ("head", "left_foot", "right_foot")
#: What the drawn triad is LABELLED. The label names the FRAME the recording expresses
#: (chest/foot BODY), not the puck, which is why the hands' raw-puck triad is the one
#: called "tracker" while these say what body part they are.
_BODY_ROLE_LABEL = {"head": "chest", "left_foot": "foot L", "right_foot": "foot R"}

#: role -> its puck->BODY relabel (rotation-only; see _HEAD_RELABEL_M).
#:
#: ⚠️ The two FEET currently SHARE the chest's matrix, and that is a DECLARATION about
#: the hardware, not a derivation: the operator's instruction when the 4th and 5th pucks
#: were added was "mount them like the chest one, so the axes are the same". If a foot
#: bracket turns out to hold its puck differently, this is the ONE place to fix — and it
#: must be fixed here rather than compensated downstream, because the same matrix is what
#: `<session>/calibration/foot_extrinsic.json` will carry into the converter.
#: A wrong relabel is invisible in POSITION (it is rotation-only) and shows up only as a
#: foot whose axes point somewhere unexpected, so check the drawn triads in RAW_FRAME=1
#: against the physical pucks before recording a 5pt session.
_BODY_RELABEL_M = {
    "head": _HEAD_RELABEL_M,
    "left_foot": _HEAD_RELABEL_M,
    "right_foot": _HEAD_RELABEL_M,
}

# ---------------------------------------------------------------------------
# CHEST display home — the collection viz's OWN, and the only role it overrides.
#
# The two HANDS use ``gripper_only_view.HOME_WRIST_POSE`` verbatim. The chest does not.
# That dict's ``head`` entry (0.30, 0, 1.30) is holosoma's torso+0.35z of a standing
# OPERATOR — 35 cm ABOVE the hands — and it is the right number where it is used: the two
# replay viewers and FLoRe's offline inference viewer re-base a RECORDED take onto it (a
# person's chest IS above their hands), and the motion-completion chest anchor
# ``mc_frames.RIG_CHEST_STAND_Z`` IS that z. All of that is left untouched.
#
# It is the wrong spot for the LIVE rig's "Reset robot to home (c)". What the operator
# wants to snap back to is the G1's OWN home arrangement — the body reference BETWEEN and
# BEHIND the two grippers at about their height — not a point floating over them. So the
# chest's home here is the PELVIS of the G1's zero configuration, placed relative to the
# hand homes by the URDF's own pelvis->wrist_yaw_link vector:
#
#     head_home = mid(wrist_home_L, wrist_home_R) − mid(pelvis→wrist)_URDF
#               = (0.333, 0, 0.900) − (0.19977428, ~0, 0.09523283)
#               = (0.13322572, 0, 0.80476717)
#
# The wrist homes are the puck homes minus the puck's wrist-local seat
# (``_TRACKER_TRUE_XYZ_M``; the homes carry identity rotation, so it is a plain
# subtraction). The URDF vector is FK at q=0 — at zero every revolute joint contributes
# only its origin, so composing the joint origins IS the zero-configuration forward
# kinematics. That puts the chest 20 cm behind the wrist (44 cm behind the gripper's TCP)
# and 9.5 cm below it, and lands z within 1.3 cm of ``mc_frames.PELVIS_STAND_Z`` (0.79186)
# — the cross-check that this really is a standing G1's pelvis and not a picked spot.
#
# ROTATION IS IDENTITY, exactly like every ``HOME_WRIST_POSE`` entry. The home-anchored
# path applies the live WORLD-frame delta on top (``gripper_only_view.home_anchor``) and a
# body relabel cancels out of that delta (see :func:`raw_chest_pose_from_puck`), so there
# is nothing here to rotate — this moves the chest triad's position and NOTHING else.
#
# y is exactly 0: the URDF's 1e-5 m left/right asymmetry is round-off in the joint origins
# (``mc_frames`` documents the same choice) and "between the two grippers" is the intent.
# Re-derived from the URDF by ``test_logic_collection.test_chest_home_is_the_urdfs_pelvis``.
_HEAD_HOME_XYZ_M = (0.13322572, 0.0, 0.80476717)

# The two 5pt FEET's display homes, derived the SAME way and for the same reason as the
# chest's: "Reset robot to home (c)" must put every triad where the ROBOT's own body is,
# not where a standing operator's is. So each foot home is the chest (== pelvis) home plus
# the URDF's own pelvis -> {side}_ankle_roll_link vector at q=0:
#
#     foot_home = _HEAD_HOME_XYZ_M + (-0.000002, ±0.118506, -0.756864)
#               = (0.133223, ±0.118506, 0.047903)
#
# At q=0 the ankle chain's rotation is EXACTLY identity (max|R-I| = 2e-18), so this is a
# plain vector add, and the two sides are mirrored to 0 in y — no round-off to swallow,
# unlike the wrist chain. The resulting z sits 1.29 cm above the floor rather than on it,
# which is the SAME 1.29 cm by which the chest home (a wrist-midpoint derivation) sits
# above mc_frames.PELVIS_STAND_Z: the whole cluster is one rigid arrangement, so the feet
# are consistent with the chest and hands instead of being separately grounded.
#
# NOT gripper_only_view.HOME_WRIST_POSE's foot entries (0.30, ±0.12, 0.10): those are the
# shared display homes the two replay viewers and FLoRe's offline inference viewer re-base a
# RECORDED take onto, and they stay untouched — the same split the chest already has.
# Re-derived from the URDF by test_logic_collection.test_foot_homes_are_the_urdfs_ankles.
_FOOT_HOME_XYZ_M = {
    "left_foot": (0.13322372, 0.11850600, 0.04790317),
    "right_foot": (0.13322372, -0.11850600, 0.04790317),
}


def display_home_pose(role: str):
    """Home pose for one view role in the home-anchored gripper display.

    THE one place the collection viz resolves a home, so the startup seat, the "Reset robot
    to home (c)" re-seat and the per-frame live anchor cannot drift apart — if they did, a
    reset would visibly jump the instant tracking resumed (``home_anchor`` at ``cur == ref``
    must reproduce exactly the pose the reset just wrote).

    Hands come straight from ``gripper_only_view.HOME_WRIST_POSE``; the chest is overridden
    with :data:`_HEAD_HOME_XYZ_M` and the two 5pt feet with :data:`_FOOT_HOME_XYZ_M` (see
    them for why). An unknown future role still falls back
    to the LEFT hand's home rather than stacking on (0,0,0). Returns ``(pos[3], quat_wxyz[4])``
    as fresh arrays — callers hand them to viser, which must not alias a module constant.
    """
    import gripper_only_view as gov

    if role == "head":
        return (np.asarray(_HEAD_HOME_XYZ_M, dtype=float),
                np.array([1.0, 0.0, 0.0, 0.0]))
    if role in _FOOT_HOME_XYZ_M:
        return (np.asarray(_FOOT_HOME_XYZ_M[role], dtype=float),
                np.array([1.0, 0.0, 0.0, 0.0]))
    hp, hq = gov.HOME_WRIST_POSE.get(role, gov.HOME_WRIST_POSE["left"])
    return np.asarray(hp, dtype=float).copy(), np.asarray(hq, dtype=float).copy()


def _rotmat_to_wxyz_local(R: np.ndarray) -> np.ndarray:
    """3x3 rotation -> unit wxyz quaternion; delegates to gripper_only_view's Shepperd
    implementation (single source) so the RAW-mode quats match the mesh-placement math."""
    import gripper_only_view as gov
    return gov._rotmat_to_wxyz(R)


def _rot_vec_by_wxyz(q, v) -> np.ndarray:
    """Rotate 3-vector ``v`` by a unit wxyz quaternion ``q`` (q ⊗ [0,v] ⊗ q*).

    Uses the branch-free ``v + 2w(u×v) + 2u×(u×v)`` form (u = q's vector part), so it
    needs only numpy — no viser.transforms / scipy import on the render hot path."""
    q = np.asarray(q, dtype=float)
    v = np.asarray(v, dtype=float)
    u = q[1:]
    t = 2.0 * np.cross(u, v)
    return v + q[0] * t + np.cross(u, t)


def follow_pose_binding_puck(tgt_p, tgt_q):
    """Map a TRACKED wrist pose to the follow-frame pose that lands the puck triad on it.

    The gripper-only follow frame carries every triad + the gripper meshes + the camera
    as rigid children at fixed wrist-relative offsets. To make the "tracker" (puck) child
    — at local ``_TRACKER_TRUE_XYZ_M`` — coincide with the tracked pose ``(tgt_p, tgt_q)``
    for ANY orientation, seat the follow frame at:

        follow_q = tgt_q
        follow_p = tgt_p + R(tgt_q) · _TRACKER_FOLLOW_OFFSET_M     (== tgt_p − R(tgt_q)·puck_local)

    so the puck child's world pose = follow_p + R(follow_q)·puck_local = tgt_p, with
    orientation tgt_q. Every other child keeps its wrist-relative pose and rides along;
    the cluster now pivots about the puck. Pure translation of the parent — child local
    offsets are untouched. Returns ``(follow_p ndarray[3], follow_q ndarray[4])``."""
    tgt_q = np.asarray(tgt_q, dtype=float)
    follow_p = np.asarray(tgt_p, dtype=float) + _rot_vec_by_wxyz(tgt_q, _TRACKER_FOLLOW_OFFSET_M)
    return follow_p, tgt_q


def raw_follow_pose_from_puck(puck_p, puck_q):
    """RAW-mode follow-frame pose: seat the wrist frame directly on the raw puck.

    In RAW_FRAME mode the gripper cluster is NOT home-anchored — it is drawn at the
    puck's raw libsurvive pose. The follow frame carries the cluster at fixed WRIST-
    relative child offsets, so to place the WRIST from the raw puck we compose the
    fused puck->wrist transform (the -y,-x,-z body relabel M, then back the wrist
    lever out of the puck->TCP offset):

        follow_q = puck_q ⊗ M_quat
        follow_p = puck_p + R(puck_q) · PUCK_TO_WRIST,   PUCK_TO_WRIST = −(M·_TRACKER_TRUE_XYZ_M)

    The TCP triad (child at +0.242 x), wrist_yaw triad (child at 0), gripper meshes,
    and camera then land at exactly the fused-transform poses, and the puck triad —
    rotated by M^-1 (see _puck_child_quat) — reads back the raw puck.
    Returns ``(follow_p ndarray[3], follow_q ndarray[4])``. Pure; unit-tested."""
    puck_q = np.asarray(puck_q, dtype=float)
    m_quat = _rotmat_to_wxyz_local(_RAW_RELABEL_M)
    puck_to_wrist = -(_RAW_RELABEL_M @ np.asarray(_TRACKER_TRUE_XYZ_M, dtype=float))
    follow_p = np.asarray(puck_p, dtype=float) + _rot_vec_by_wxyz(puck_q, puck_to_wrist)
    follow_q = _quat_mul_wxyz(puck_q, m_quat)
    return follow_p, follow_q


def raw_body_pose_from_puck(role: str, puck_p, puck_q):
    """RAW-mode BODY-REFERENCE pose: the body frame derived from that role's raw puck.

    The chest's analogue of :func:`raw_follow_pose_from_puck` — and the reason both exist
    is the same: a follow frame must carry the frame the RECORDING expresses, not the raw
    puck it was measured with. For a hand that frame is the wrist (puck ∘ M, minus the
    lever); for the chest it is the torso body reference (puck ∘ H, NO lever, since the
    chest triad IS the frame origin):

        chest_q = puck_q ⊗ H_quat
        chest_p = puck_p                        (H is rotation-only)

    So the drawn triad reads exactly the ``head_pos`` / ``head_rot_axis_angle`` the
    converter stores (``world_T_chest = world_T_puck @ [H|0]``, the same compose through
    the same matrix), which is what makes RAW mode a faithful twin of the recording.

    ONLY the RAW path needs this. The home-anchored path applies a WORLD-frame rotation
    delta (``gripper_only_view.home_anchor``: ``R_target = cur ⊗ ref^-1 ⊗ R_home``) and a
    body relabel CANCELS out of it — ``(cur⊗H) ⊗ (ref⊗H)^-1 == cur ⊗ ref^-1`` — so an
    anchored chest triad is already home-referenced and right-multiplying H there would
    only spin it off its home convention. This is the opposite of ``_puck_child_quat``,
    which is unconditional because it encodes a WITHIN-cluster relative orientation
    (puck vs wrist) that survives any rigid rebasing; H is an ABSOLUTE frame convention
    on the raw pose and there is no second chest triad to be relative to. Pure;
    unit-tested. Returns ``(body_p ndarray[3], body_q ndarray[4])``.

    ``role`` picks the relabel out of :data:`_BODY_RELABEL_M` (the two 5pt FEET currently
    share the chest's — read that table's warning). A role with no entry gets the raw puck
    pose UNCHANGED rather than the chest's matrix, so a future body reference cannot
    silently inherit a bracket rotation that was never measured for it."""
    H = _BODY_RELABEL_M.get(role)
    p_out = np.asarray(puck_p, dtype=float).copy()
    q_in = np.asarray(puck_q, dtype=float)
    if H is None:
        return p_out, q_in.copy()
    return p_out, _quat_mul_wxyz(q_in, _rotmat_to_wxyz_local(H))


def raw_chest_pose_from_puck(puck_p, puck_q):
    """The CHEST's :func:`raw_body_pose_from_puck`, kept as a name.

    ``convert_to_umi``, the raw replay viewer and the tests all reach for the chest case
    by this name; a role-keyed call is the generalization, not a replacement."""
    return raw_body_pose_from_puck("head", puck_p, puck_q)


def _quat_mul_wxyz(a, b) -> np.ndarray:
    """Hamilton product of two wxyz quaternions (local copy of g1_right_hand_ik.quat_mul,
    kept here so the pure follow-pose helpers need no humi_ik import in gripper mode)."""
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


def _puck_child_quat() -> np.ndarray:
    """Local (wrist-relative) rotation for the puck 'tracker' triad. ALWAYS applied.

    Every follow frame in this viz is a WRIST frame (``follow_pose_binding_puck`` takes a
    tracked WRIST pose; ``raw_follow_pose_from_puck`` composes puck→wrist explicitly), and
    the physical VIVE puck is bolted onto the tool ROTATED — the two body frames differ by
    the measured relabel M. So an unrotated child would inherit the WRIST axes and draw the
    puck triad with the same orientation as the TCP triad, which is not what the hardware
    does. Carrying M^-1 renders it at the puck's true orientation:
    ``follow_q ⊗ M^-1 = (puck_q ⊗ M) ⊗ M^-1 = puck_q``.

    This is a MECHANICAL constant, so it is identical in the home-anchored and RAW_FRAME
    paths — home-anchoring rigidly rebases the whole cluster and cannot change a
    within-cluster relative orientation. It used to be gated on RAW_FRAME, which left the
    default path drawing tracker == TCP.

    Agrees with all three other consumers of the same contract, and M is its own inverse
    (symmetric, det=+1) so they are literally the same matrix:
      * ``hl_serving._TRIAD_LOCAL_ROT  = {"tracker": _PUCK_TO_WRIST_RELABEL}``
      * ``reinfer_hl_take``            : ``PUCK_TO_WRIST_RELABEL if name == "tracker"``
      * ``replay_{zarr,raw}_viz.puck_triad_local_quat`` : ``R_ext^-1`` off the dataset
    """
    m_quat = _rotmat_to_wxyz_local(_RAW_RELABEL_M)
    return np.array([m_quat[0], -m_quat[1], -m_quat[2], -m_quat[3]])   # inverse (unit quat conj)


def resolve_display_mode(display_mode: str, tracking_mode: str) -> str:
    """Resolve the display mode, expanding 'auto' by the tracking mode.

    Spec: gripper-only in EVERY tracking mode — 1pt, 2pt, 3pt AND 5pt all draw just the
    isolated yam-gripper cluster(s) with no humanoid and no mink IK. 3pt is 2pt plus
    ONE extra triad for the chest puck and 5pt is 3pt plus two foot triads; neither is a
    different display. 3pt used to
    resolve to 'humanoid', which meant the operator saw a whole IK'd G1 (and paid the
    ~8 ms/frame solve) for a rig that is physically two handheld grippers.

    'humanoid'/'gripper' pass through unchanged, so the full G1 is still one env var
    away (DISPLAY_MODE=humanoid) — that is the only place the torso FrameTask is
    exercised. Kept a pure function so the auto mapping is unit-testable.

    ⚠️ ``humanoid`` is REFUSED for 5pt, by name. ``ThreePointIK`` has three tasks (two
    hand FrameTasks + the yaw-only torso one) and NO foot task, so a 5pt humanoid would
    draw the two feet as raw triads floating beside a G1 whose legs ignore them — a
    picture that looks like the feet are being tracked wrong rather than not tracked at
    all. Raising here is what makes the launcher's own refusal and this one the same
    decision; gripper-only draws all five pucks honestly."""
    if tracking_mode == "5pt" and display_mode == "humanoid":
        raise ValueError(
            "DISPLAY_MODE=humanoid is not available for MODE=5pt: the mink IK "
            "(g1_three_point_ik.ThreePointIK) has hand + torso tasks only, so the two "
            "foot pucks would be drawn as free triads next to legs that do not follow "
            "them. Use the default gripper-only display (DISPLAY_MODE=auto), which draws "
            "all five pucks, or drop to MODE=3pt for the humanoid view.")
    if display_mode == "auto":
        return "gripper"
    return display_mode


# ----------------------------------------------------------------------------
# Shared state (lock-guarded), mirroring bota_ft_viz.SensorState
# ----------------------------------------------------------------------------


class EncoderBuffer:
    """Rolling history of the gripper open-fraction for the live chart."""

    def __init__(self, window_s: float = 4.0, max_points: int = 400):
        self.window_s = window_s
        self.t: deque[float] = deque(maxlen=max_points)
        self.frac: deque[float] = deque(maxlen=max_points)

    def append(self, t: float, frac: float) -> None:
        self.t.append(t)
        self.frac.append(float(frac))

    def export(self) -> tuple[np.ndarray, np.ndarray] | None:
        if not self.t:
            return None
        t = np.asarray(self.t, dtype=float)
        x = t - t[-1]
        keep = x >= -self.window_s
        return x[keep], np.asarray(self.frac, dtype=float)[keep]


@dataclass
class CollectionState:
    """Latest tracker / gripper / camera data + handshake flags. Lock-guarded."""

    lock: threading.Lock = field(default_factory=threading.Lock)

    # PRIMARY tracker (robot-aligned world frame, from vive_tracker_bridge). This
    # is the LEFT hand in EVERY mode (LEFT-canonical): it drives the main IK arm,
    # the camera-follow frame, and the jaw. The unprefixed name is kept because it
    # is the primary slot the whole update()/render path revolves around.
    tracker_pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    tracker_quat: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0]))
    tracker_seq: int = 0
    tracker_t: float = 0.0
    tracker_stale: bool = True
    tracker_connected: bool = False
    # perf_counter() of the last frame RECEIVED, per stream. Unlike the sticky
    # *_connected flags (true forever after the first frame), these let update()
    # report a module that connected then went SILENT — a bridge dying mid-take is
    # the failure data collection most needs to see. Arrival rate is derived from
    # the seq delta over the same window.
    tracker_recv_t: float = 0.0
    tracker_hz: float = 0.0

    # Secondary RIGHT-hand tracker (2pt/3pt; 1pt leaves these at defaults).
    right_tracker_pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    right_tracker_quat: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0]))
    right_tracker_stale: bool = True
    right_tracker_connected: bool = False
    right_tracker_recv_t: float = 0.0

    # BODY-REFERENCE trackers: the chest (3pt) and the two feet (5pt). ONE dict keyed by
    # role rather than a parallel set of fields per role — the chest's five fields were
    # already copied from the right hand's, and copying them twice more is how a fifth
    # puck ends up read in three places and written in four. Every value is
    # ``{"pos","quat","stale","connected","recv_t"}``; a role appears only once its puck
    # has produced a first pose, exactly like the wire (see :func:`_body_tracker_slot`).
    body_tracker: dict = field(default_factory=dict)

    # Gripper encoder (from keyboard_gripper --encoder-ws-port).
    gripper_frac: float = 0.0  # 0=closed, 1=open
    gripper_cmd_rad: float = 0.0
    gripper_actual_rad: float = 0.0
    gripper_open_rad: float = 4.90
    gripper_closed_rad: float = 0.0
    gripper_seq: int = 0
    gripper_connected: bool = False
    gripper_recv_t: float = 0.0
    gripper_hz: float = 0.0
    # Motor health from the bridge: True until the i2rt control thread dies on an
    # unrecoverable fault (e.g. a hard-stop stall overheats the motor). The bridge
    # keeps broadcasting after the fault, so this distinguishes a LIVE gripper from
    # a connected-but-dead one — the page shows it RED rather than a phantom green.
    gripper_motor_ok: bool = True
    # 2-key keyboard bridge liveness, relayed THROUGH the gripper bridge (the
    # keyboard talks only to the gripper WS, never to the viz). True while that
    # bridge's heartbeat is fresh on the gripper side.
    kbd_online: bool = False
    history: EncoderBuffer = field(default_factory=EncoderBuffer)

    # Preview-image master switch, driven by the GUI's "Show camera view" checkbox
    # and read by BOTH camera WS reader threads. False means "nobody is going to look
    # at these pixels", which lets the readers skip the base64 decode and the render
    # loop skip a JPEG decode + viser's JPEG RE-encode per frame per camera — measured
    # at roughly half of this process's CPU with two 640x480@30Hz D405s (see the
    # `Camera` GUI folder). A plain bool, not lock-guarded, matching `show_compensated`:
    # a torn read costs at most one frame of preview either way.
    #
    # It does NOT gate RECORDING. The recorder is a separate process with its own
    # subscription to the same camera bridges and writes the bridge's JPEG bytes
    # verbatim (recording/recorder.py), so turning the preview off cannot change a
    # single recorded byte.
    preview_images: bool = True
    # Camera (from realsense_bridge): latest JPEG bytes + size.
    cam_jpeg: bytes | None = None
    cam_w: int = 0
    cam_h: int = 0
    cam_seq: int = 0
    cam_connected: bool = False
    cam_recv_t: float = 0.0
    cam_hz: float = 0.0
    # RIGHT-hand D405 (2pt/3pt). A parallel set rather than a role dict, because the
    # render loop snapshots these fields individually under the lock and a dict would
    # need a copy per frame. All zero/None when no right camera is configured, which is
    # exactly the "not connected" state the status line already knows how to render.
    cam_jpeg_right: bytes | None = None
    cam_w_right: int = 0
    cam_h_right: int = 0
    cam_seq_right: int = 0
    cam_connected_right: bool = False
    cam_recv_t_right: float = 0.0
    cam_hz_right: float = 0.0
    # RIGHT-hand gripper (2pt/3pt). A PARALLEL set of the left's fields, not a
    # subset: the panel shows the same readouts per hand (fraction + encoder rad +
    # its own measured endpoints) and drives the same Open/Close/slider controls, so
    # an operator can no longer read a right-hand number that is really the left's.
    # The endpoints are kept PER HAND on purpose — the right DM4310's absolute
    # encoder re-references on its own power-cycle, so normalising it against the
    # left's range would clamp the right width to one rail.
    gripper_frac_right: float = 0.0
    gripper_cmd_rad_right: float = 0.0
    gripper_actual_rad_right: float = 0.0
    gripper_open_rad_right: float = 4.90
    gripper_closed_rad_right: float = 0.0
    gripper_seq_right: int = 0
    gripper_motor_ok_right: bool = True
    gripper_connected_right: bool = False
    gripper_recv_t_right: float = 0.0
    gripper_hz_right: float = 0.0
    history_right: EncoderBuffer = field(default_factory=EncoderBuffer)

    # Recalibration / reset handshake: the render loop performs the IK mutation
    # (it owns the solver), the GUI buttons just raise these flags under the lock.
    # This keeps ALL _solver.reset()/step() calls on the single render thread —
    # mink/mujoco's Configuration is not thread-safe.
    recal_request: bool = False
    reset_request: bool = False
    # "Reset tracker axes to wrist": captures the current puck orientation as the role's
    # MOUNT rotation, so the visualized triad reads axis-aligned in the reference posture.
    # Same flag discipline — the render thread owns the IK object.
    axes_reset_request: bool = False


# ----------------------------------------------------------------------------
# WebSocket reader threads (consume the three bridges)
# ----------------------------------------------------------------------------


class _WSReader(threading.Thread):
    """Connects to a bridge WS and feeds frames into CollectionState. Auto-reconnects.

    Bidirectional: :meth:`send_json` lets the GUI thread push a control frame back
    to the bridge over the same connection (used to drive the gripper open/close/
    stop + slider from the viser page). Sends are best-effort — dropped silently
    while the bridge is down, since the gripper holds its last position anyway.
    """

    def __init__(self, state: CollectionState, url: str, on_message, name: str):
        super().__init__(daemon=True)
        self.state = state
        self.url = url
        self._on_message = on_message
        self._name = name
        self._stop = threading.Event()
        # Set inside run() so the GUI thread can schedule sends onto the reader's
        # asyncio loop; None while disconnected.
        self._loop = None
        self._ws = None

    def stop(self):
        self._stop.set()

    def send_json(self, obj) -> None:
        """Thread-safe: schedule a JSON frame to the bridge (no-op if down)."""
        import asyncio

        loop, ws = self._loop, self._ws
        if loop is None or ws is None:
            return
        try:
            data = json.dumps(obj)
            asyncio.run_coroutine_threadsafe(ws.send(data), loop)
        except Exception:  # noqa: BLE001 - bridge dropped mid-send
            pass

    def run(self):
        import asyncio

        import websockets

        async def loop():
            while not self._stop.is_set():
                try:
                    async with websockets.connect(self.url, max_size=None) as ws:
                        self._ws = ws
                        print(f"[viz] connected to {self._name} {self.url}", flush=True)
                        async for raw in ws:
                            if self._stop.is_set():
                                break
                            try:
                                msg = json.loads(raw)
                            except (ValueError, TypeError):
                                continue
                            self._on_message(msg)
                except Exception:  # noqa: BLE001 - bridge down / not up yet
                    pass
                finally:
                    self._ws = None
                if not self._stop.is_set():
                    await asyncio.sleep(1.0)  # reconnect backoff

        try:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            self._loop.run_until_complete(loop())
        except Exception:  # noqa: BLE001
            pass


def _ewma_hz(prev_hz: float, prev_recv_t: float, now: float) -> float:
    """Exponentially-smoothed arrival rate from the gap since the last frame.
    First frame (prev_recv_t==0) seeds nothing (returns 0); afterward a 0.3 EWMA
    keeps the displayed Hz steady without a sliding window."""
    if prev_recv_t <= 0.0:
        return 0.0
    dt = now - prev_recv_t
    if dt <= 0.0:
        return prev_hz
    inst = 1.0 / dt
    return inst if prev_hz <= 0.0 else (0.7 * prev_hz + 0.3 * inst)


def _body_tracker_slot(state: CollectionState, role: str) -> dict:
    """The mutable per-role slot for a BODY-REFERENCE tracker, created on first sight.

    Defaults match what the old per-role fields defaulted to (origin, identity, stale,
    disconnected), so a role that has never streamed reads exactly as it did before —
    ``connected=False`` is what every consumer gates on. Callers hold ``state.lock``."""
    slot = state.body_tracker.get(role)
    if slot is None:
        slot = {"pos": np.zeros(3), "quat": np.array([1.0, 0.0, 0.0, 0.0]),
                "stale": True, "connected": False, "recv_t": 0.0}
        state.body_tracker[role] = slot
    return slot


def _on_tracker_msg(state: CollectionState, msg: dict) -> None:
    """Handle the role-keyed tracker WS message (all modes) + the legacy flat one.

    Role-keyed (1pt/2pt/3pt/5pt): {"mode":..., "t":float,
        "trackers":{"left":{..}, "right":{..}, "head":{..}, "left_foot":{..}, ...}}.
    LEFT-canonical: the PRIMARY hand is ``left`` -> the unprefixed ``tracker_*`` state
    (main IK arm, camera, jaw); ``right`` -> ``right_tracker_*`` (2pt/3pt/5pt); every BODY
    REFERENCE (``head`` + the two 5pt feet) -> ``state.body_tracker[role]``.
    Legacy flat (old logs): {"seq","t","pos","quat","stale"} -> primary slot.
    """
    # Discriminate on the "trackers" key — every role-keyed message has it.
    trackers = msg.get("trackers")
    if trackers is not None:
        now = time.perf_counter()
        with state.lock:
            for role, data in trackers.items():
                pos = data.get("pos")
                quat = data.get("quat")
                if pos is None or quat is None:
                    continue
                if role == "left":  # PRIMARY hand (main IK arm / camera / jaw)
                    state.tracker_pos = np.asarray(pos, dtype=float)
                    state.tracker_quat = np.asarray(quat, dtype=float)
                    state.tracker_seq = int(data.get("seq", state.tracker_seq + 1))
                    state.tracker_t = float(data.get("t", time.time()))
                    state.tracker_stale = bool(data.get("stale", False))
                    state.tracker_connected = True
                    state.tracker_hz = _ewma_hz(state.tracker_hz, state.tracker_recv_t, now)
                    state.tracker_recv_t = now
                elif role == "right":
                    state.right_tracker_pos = np.asarray(pos, dtype=float)
                    state.right_tracker_quat = np.asarray(quat, dtype=float)
                    state.right_tracker_stale = bool(data.get("stale", False))
                    state.right_tracker_connected = True
                    state.right_tracker_recv_t = now
                elif role in _BODY_ROLES_ALL:
                    slot = _body_tracker_slot(state, role)
                    slot["pos"] = np.asarray(pos, dtype=float)
                    slot["quat"] = np.asarray(quat, dtype=float)
                    slot["stale"] = bool(data.get("stale", False))
                    slot["connected"] = True
                    slot["recv_t"] = now
        return
    # Legacy flat message -> primary slot (kept for old logs / UMI-on-Air).
    pos = msg.get("pos")
    quat = msg.get("quat")
    if pos is None or quat is None:
        return
    now = time.perf_counter()
    with state.lock:
        state.tracker_pos = np.asarray(pos, dtype=float)
        state.tracker_quat = np.asarray(quat, dtype=float)
        state.tracker_seq = int(msg.get("seq", state.tracker_seq + 1))
        state.tracker_t = float(msg.get("t", time.time()))
        state.tracker_stale = bool(msg.get("stale", False))
        state.tracker_connected = True
        state.tracker_hz = _ewma_hz(state.tracker_hz, state.tracker_recv_t, now)
        state.tracker_recv_t = now


def _on_gripper_msg(state: CollectionState, msg: dict) -> None:
    open_rad = float(msg.get("open", 4.90))
    closed_rad = float(msg.get("closed", 0.0))
    actual = float(msg.get("actual", msg.get("cmd", 0.0)))
    # Use the single tested encoder->fraction mapping (endpoints come over the
    # wire, so the bridge's measured range drives the normalization).
    from g1_right_hand_ik import GripperRange

    frac = GripperRange(open_rad=open_rad, closed_rad=closed_rad).fraction(actual)
    now = time.perf_counter()
    with state.lock:
        state.gripper_open_rad = open_rad
        state.gripper_closed_rad = closed_rad
        state.gripper_cmd_rad = float(msg.get("cmd", actual))
        state.gripper_actual_rad = actual
        state.gripper_frac = frac
        state.gripper_seq = int(msg.get("seq", state.gripper_seq + 1))
        state.gripper_connected = True
        state.gripper_hz = _ewma_hz(state.gripper_hz, state.gripper_recv_t, now)
        state.gripper_recv_t = now
        # Default True so an older bridge (no motor_ok field) isn't shown faulted.
        state.gripper_motor_ok = bool(msg.get("motor_ok", True))
        # The 2-key keyboard's liveness rides in on the gripper frame (the gripper
        # bridge relays its heartbeat). Default missing -> False so an older bridge
        # simply shows the keyboard as offline rather than crashing.
        state.kbd_online = bool(msg.get("kbd_online", False))
        state.history.append(now, frac)


def _on_gripper_right_msg(state: CollectionState, msg: dict) -> None:
    """RIGHT-hand gripper encoder frames (2pt/3pt).

    Its own endpoints: the right DM4310 is a second motor whose absolute encoder
    re-references on its own power-cycle, so normalising it with the left hand's range
    would clamp the right width to one rail. Deliberately a MIRROR of
    _on_gripper_msg (not a reduced copy) so the panel's right-hand readouts mean
    exactly what the left's do; the only field it does NOT mirror is ``kbd_online``,
    which is a single global bridge relayed on the LEFT frame."""
    open_rad = float(msg.get("open", 4.90))
    closed_rad = float(msg.get("closed", 0.0))
    actual = float(msg.get("actual", msg.get("cmd", 0.0)))
    from g1_right_hand_ik import GripperRange

    frac = GripperRange(open_rad=open_rad, closed_rad=closed_rad).fraction(actual)
    now = time.perf_counter()
    with state.lock:
        state.gripper_open_rad_right = open_rad
        state.gripper_closed_rad_right = closed_rad
        state.gripper_cmd_rad_right = float(msg.get("cmd", actual))
        state.gripper_actual_rad_right = actual
        state.gripper_frac_right = frac
        state.gripper_seq_right = int(msg.get("seq", state.gripper_seq_right + 1))
        state.gripper_connected_right = True
        state.gripper_hz_right = _ewma_hz(state.gripper_hz_right, state.gripper_recv_t_right, now)
        state.gripper_recv_t_right = now
        # Default True so an older bridge (no motor_ok field) isn't shown faulted.
        state.gripper_motor_ok_right = bool(msg.get("motor_ok", True))
        state.history_right.append(now, frac)


# Sentinel for "this base64 payload is corrupt" — distinct from the legitimate None
# that `preview_images = False` produces. Without it the two cases collapse and a
# preview-off frame would be dropped BEFORE the liveness bookkeeping, so the status
# row would read "camera ○ offline" for a bridge that is streaming perfectly.
_DECODE_FAILED = object()


def _maybe_b64decode(state: CollectionState, b64: str):
    """base64 -> JPEG bytes, or None when the preview is switched off.

    The camera WS reader threads call this per frame per camera. With the preview off
    nobody will ever look at the pixels, so decoding them is pure cost — and so is
    KEEPING them: a stashed frame would repaint the instant the checkbox came back on,
    showing a stale image (arbitrarily old if the bridge died meanwhile). Returning
    None both skips the work and lets the render loop's `cam_jpeg is not None` guard
    hold off until a genuinely fresh frame arrives.

    Returns ``_DECODE_FAILED`` on a corrupt payload so the caller can drop the frame
    entirely, liveness included — that IS a broken stream and should show as one."""
    if not state.preview_images:
        return None
    try:
        return base64.b64decode(b64)
    except Exception:  # noqa: BLE001
        return _DECODE_FAILED


def _on_camera_msg(state: CollectionState, msg: dict) -> None:
    b64 = msg.get("jpeg_b64")
    if b64 is None:
        return
    jpeg = _maybe_b64decode(state, b64)
    if jpeg is _DECODE_FAILED:
        return
    now = time.perf_counter()
    with state.lock:
        state.cam_jpeg = jpeg
        state.cam_w = int(msg.get("w", 0))
        state.cam_h = int(msg.get("h", 0))
        state.cam_seq = int(msg.get("seq", state.cam_seq + 1))
        state.cam_connected = True
        state.cam_hz = _ewma_hz(state.cam_hz, state.cam_recv_t, now)
        state.cam_recv_t = now


def _on_camera_right_msg(state: CollectionState, msg: dict) -> None:
    """RIGHT-hand D405 frames. Same wire format, its OWN state fields — the two
    cameras are separate bridge processes and each hand's frustum shows its own view,
    so mixing them into one latest-frame register would silently draw one camera twice."""
    b64 = msg.get("jpeg_b64")
    if b64 is None:
        return
    jpeg = _maybe_b64decode(state, b64)
    if jpeg is _DECODE_FAILED:
        return
    now = time.perf_counter()
    with state.lock:
        state.cam_jpeg_right = jpeg
        state.cam_w_right = int(msg.get("w", 0))
        state.cam_h_right = int(msg.get("h", 0))
        state.cam_seq_right = int(msg.get("seq", state.cam_seq_right + 1))
        state.cam_connected_right = True
        state.cam_hz_right = _ewma_hz(state.cam_hz_right, state.cam_recv_t_right, now)
        state.cam_recv_t_right = now


# ----------------------------------------------------------------------------
# Mock feeders (no bridges / no hardware)
# ----------------------------------------------------------------------------


class _MockFeeder(threading.Thread):
    """Synthesizes tracker + gripper + camera frames so --mock runs standalone.

    ``active_roles`` selects which trackers to synthesize: the primary ('left')
    always moves; every other role (2pt right, 3pt chest, the two 5pt feet) gets its
    OWN distinct motion so the operator can verify per-role frame mapping (C00: each
    synthetic source moves only its assigned target)."""

    def __init__(self, state: CollectionState, home_pos: np.ndarray,
                 active_roles: tuple[str, ...] = ("left",)):
        super().__init__(daemon=True)
        self.state = state
        self._home = home_pos.copy()
        self._active_roles = tuple(active_roles)
        self._stop = threading.Event()

    def stop(self):
        self._stop.set()

    def run(self):
        t0 = time.perf_counter()
        yy, xx = np.meshgrid(np.linspace(0, 1, 240), np.linspace(0, 1, 320), indexing="ij")
        while not self._stop.is_set():
            t = time.perf_counter() - t0
            # tracker: small circle around the captured reference
            pos = self._home + np.array([0.08 * math.sin(0.5 * t), 0.08 * math.cos(0.5 * t), 0.05 * math.sin(t)])
            ang = 0.3 * math.sin(0.7 * t)
            quat = np.array([math.cos(ang / 2), 0.0, 0.0, math.sin(ang / 2)])
            frac = 0.5 * (1 + math.sin(0.8 * t))  # gripper opening/closing
            # camera: moving gradient
            r = (127 * (1 + np.sin(2 * math.pi * (xx + 0.1 * t)))).astype(np.uint8)
            g = (127 * (1 + np.sin(2 * math.pi * (yy + 0.13 * t)))).astype(np.uint8)
            b = np.full_like(r, int(80 + 80 * math.sin(t)))
            from PIL import Image
            bio = io.BytesIO()
            Image.fromarray(np.stack([r, g, b], 2), "RGB").save(bio, "JPEG", quality=70)
            now = time.perf_counter()
            with self.state.lock:
                self.state.tracker_pos = pos
                self.state.tracker_quat = quat
                self.state.tracker_seq += 1
                self.state.tracker_t = time.time()
                self.state.tracker_stale = False
                self.state.tracker_connected = True
                self.state.tracker_hz = _ewma_hz(self.state.tracker_hz, self.state.tracker_recv_t, now)
                self.state.tracker_recv_t = now
                # Secondary trackers (2pt/3pt): each a DISTINCT motion offset from
                # home so an L/R swap or wrong-target mapping is visible (C00).
                if "right" in self._active_roles:
                    r_ang = 0.3 * math.cos(0.7 * t)
                    self.state.right_tracker_pos = self._home + np.array(
                        [0.4, 0.10 * math.cos(0.6 * t), 0.06 * math.sin(0.9 * t)])
                    self.state.right_tracker_quat = np.array([math.cos(r_ang / 2), 0.0, 0.0, math.sin(r_ang / 2)])
                    self.state.right_tracker_stale = False
                    self.state.right_tracker_connected = True
                    self.state.right_tracker_recv_t = now
                # BODY REFERENCES: a slow, small motion each (they are body points, not
                # hand orbits), on DISTINCT offsets so a role landing in the wrong slot
                # reads as a wrong trace rather than as two identical ones. The feet sit
                # BELOW the home and barely move, which is what a stance looks like.
                for _role, _off, _amp, _rate in (
                        ("head", (0.2, 0.0, 0.35), 0.03, 0.4),
                        ("left_foot", (-0.05, 0.12, -0.85), 0.01, 0.3),
                        ("right_foot", (-0.05, -0.12, -0.85), 0.01, 0.35)):
                    if _role not in self._active_roles:
                        continue
                    slot = _body_tracker_slot(self.state, _role)
                    slot["pos"] = self._home + np.array(
                        [_off[0], _off[1], _off[2] + _amp * math.sin(_rate * t)])
                    b_ang = 0.2 * math.sin(0.5 * t)
                    slot["quat"] = np.array([math.cos(b_ang / 2), 0.0, 0.0, math.sin(b_ang / 2)])
                    slot["stale"] = False
                    slot["connected"] = True
                    slot["recv_t"] = now
                self.state.gripper_frac = frac
                self.state.gripper_actual_rad = frac * 4.90
                self.state.gripper_cmd_rad = frac * 4.90
                self.state.gripper_seq += 1
                self.state.gripper_connected = True
                self.state.gripper_motor_ok = True  # mock: motor always "healthy"
                self.state.gripper_hz = _ewma_hz(self.state.gripper_hz, self.state.gripper_recv_t, now)
                self.state.gripper_recv_t = now
                self.state.kbd_online = True  # mock: pretend the 2-key pad is present
                self.state.history.append(now, frac)
                self.state.cam_jpeg = bio.getvalue()
                self.state.cam_w, self.state.cam_h = 320, 240
                self.state.cam_seq += 1
                self.state.cam_connected = True
                self.state.cam_hz = _ewma_hz(self.state.cam_hz, self.state.cam_recv_t, now)
                self.state.cam_recv_t = now
            time.sleep(1.0 / 30.0)


# ----------------------------------------------------------------------------
# Per-hand F/T panel
# ----------------------------------------------------------------------------


class _FTPanel:
    """One Bota sensor's whole GUI block: status row, four charts, tare /
    compensated / algorithm / recompensation controls.

    ONE instance per physical sensor, so the left and right hands get IDENTICAL
    controls by construction rather than by two copies of the same code kept in
    sync by hand. The two sensors share NOTHING but this class: each has its own
    ``bota_ft_viz.SensorState``, its own reader thread (the sole toucher of its
    HWI), its own compensator/RNEA config, and therefore its own tare offset and
    its own payload estimate. Every control here writes only into ``self.state``
    under that state's lock, which is exactly the handshake the reader consumes —
    so a right-hand click can never reach the left hand's sensor.

    The ONE thing arbitrated across hands is RECOMPENSATION: its overlay is a
    screen-filling prompt telling the operator to rotate *the* tool through many
    orientations, and the estimator EKF needs a steady ~1 kHz feed to converge, so
    two at once is both an instruction nobody can follow and a race for the
    interpreter. ``CollectionScene._recomp_gate`` serializes the check-then-arm and
    a click is REFUSED while either hand is mid-cycle. Everything else — taring,
    the raw/compensated toggle, the engine dropdown, the charts — is per hand and
    fully concurrent.

    This mirrors the ``FtSide`` of the bimanual deploy viz (not part of this release), which solved
    the same two-sensor problem first; the difference is only that the logic lives
    on the panel here instead of on the scene.
    """

    def __init__(self, server, state, hand: str, recomp_enabled: bool,
                 rnea_available: bool):
        self.server = server
        self.state = state          # bota_ft_viz.SensorState (this sensor's)
        self.hand = hand            # "left" | "right"
        # Which RNEA config file this hand reads, for the disabled-dropdown hint.
        # The LEFT hand keeps the SHARED rnea_compensator.json it has always used;
        # only the right hand is per-side (see main()).
        self._rnea_basename = ("rnea_compensator.json" if hand == "left"
                               else f"rnea_compensator_{hand}.json")
        self.recomp_enabled = bool(recomp_enabled)
        self.rnea_available = bool(rnea_available)
        # GUI handles (created by the build_* methods, in _build_gui's order).
        self.gui_status = None
        self.gui_show_comp = None
        self.gui_algo = None
        self.gui_recomp_status = None
        self._algo_echo = False
        # Charts + their peak-held y half-ranges.
        self._force_chart = None
        self._torque_chart = None
        self._raw_force_chart = None
        self._raw_torque_chart = None
        self._force_y_peak = _FT_Y_START
        self._torque_y_peak = _FT_Y_START
        self._raw_force_y_peak = _FT_Y_START
        self._raw_torque_y_peak = _FT_Y_START
        self._last_ft_n = -1
        self._last_raw_ft_n = -1
        # Recompensation overlay: this hand's own modal, opened on the first
        # non-idle recal phase and closed when the cycle returns to idle. The
        # cross-hand gate guarantees at most one is ever up. All four fields are
        # RENDER-THREAD-ONLY (the click callback never touches them):
        # recomp_armed_at is the stuck-reader clock, armed lazily on the first
        # countdown frame; recomp_done_at is the success linger timer.
        self.recomp_modal = None
        self.recomp_md = None
        self.recomp_armed_at = 0.0
        self.recomp_done_at = 0.0

    # -- naming -------------------------------------------------------------
    @property
    def label(self) -> str:
        """Human label for prompts/modals: 'LEFT' / 'RIGHT'."""
        return self.hand.upper()

    # -- build --------------------------------------------------------------
    def build_status_row(self, gui, dual: bool) -> None:
        """Add this hand's Status-folder row.

        ``dual`` (two sensors this session) suffixes BOTH rows 'F/T L' / 'F/T R';
        a single-sensor session keeps the historical unsuffixed 'F/T sensor'."""
        name = f"F/T {self.label[0]}" if dual else "F/T sensor"
        self.gui_status = gui.add_text(name, initial_value="○ waiting…", disabled=True)

    def build_charts(self, scene: "CollectionScene", dual: bool) -> None:
        """Add the four wrench charts (compensated-or-raw pair + always-raw pair).

        The y-axis STARTS fixed at [-10, 10] (N and Nm) and PEAK-HOLDs: it widens
        to track a new magnitude peak but never shrinks back (see _FT_Y_START +
        _peak_hold_y). The ALWAYS-RAW pair is a live view of the uncompensated
        wrench independent of the Compensated toggle: in a raw-only session it
        mirrors the pair above, with compensation on it shows the tool weight the
        compensator removes."""
        sfx = f" {self.label[0]}" if dual else ""
        self._force_chart = scene._make_xyz_chart(
            f"Force{sfx} (N)", ("Fx", "Fy", "Fz"), y_half_range=_FT_Y_START)
        self._torque_chart = scene._make_xyz_chart(
            f"Torque{sfx} (Nm)", ("Tx", "Ty", "Tz"), y_half_range=_FT_Y_START)
        self._raw_force_chart = scene._make_xyz_chart(
            f"Force{sfx} RAW (N)", ("Fx", "Fy", "Fz"), y_half_range=_FT_Y_START)
        self._raw_torque_chart = scene._make_xyz_chart(
            f"Torque{sfx} RAW (Nm)", ("Tx", "Ty", "Tz"), y_half_range=_FT_Y_START)

    def build_controls(self, gui, scene: "CollectionScene", dual: bool) -> None:
        """Add this hand's 'F/T compensation' folder: Tare, the raw/compensated
        toggle, the engine dropdown and Recompensation.

        Tare works on ANY sensor (it just subtracts the current reading);
        Recompensation needs the estimator+compensator configs and so is gated by
        ``recomp_enabled``."""
        title = f"F/T compensation ({self.hand})" if dual else "F/T compensation"
        with gui.add_folder(title):
            # Tare: zero the displayed wrench at the current pose. On a
            # COMPENSATED stream this raises the reader's one-shot tare_request
            # (the reader owns the HWI and re-tares the active engine, which also
            # refreshes the recorder's tare provenance); on a raw-only stream the
            # reader has no tare hook, so fall back to copying the live wrench into
            # the display offsets — the same mechanism the standalone bota tool
            # uses. Quick relative zero; distinct from Recompensation (full
            # gravity model).
            tare_btn = gui.add_button("Tare wrench")

            @tare_btn.on_click
            def _(_event) -> None:
                self.request_tare()

            # Live RAW <-> COMPENSATED toggle (mirrors bota_ft_viz). The reader
            # computes BOTH wrenches every tick when the compensator chain is
            # running and publishes whichever state.show_compensated selects, so
            # this is a pure display switch (no reader restart). Only meaningful
            # when a payload config exists (compensation_available); otherwise the
            # reader publishes raw-only and the checkbox is disabled. The charts +
            # the header |F| readout follow it on the next frame. NOTE: this is the
            # DISPLAY wrench only — recording always captures the RAW signal
            # (ft_raw_publisher), so compensation stays reproducible offline.
            comp_avail = bool(self.state.compensation_available)
            self.gui_show_comp = gui.add_checkbox(
                "Compensated wrench",
                initial_value=comp_avail,
                disabled=not comp_avail,
            )

            @self.gui_show_comp.on_update
            def _(_event) -> None:
                with self.state.lock:
                    self.state.show_compensated = bool(self.gui_show_comp.value)

            # Live compensation-ALGORITHM switch (Bota official vs independent
            # RNEA). Distinct from the Recompensation button (payload re-estimate);
            # this just changes which math compensates. ALWAYS SHOWN (R3): when no
            # RNEA config exists it is DISABLED (greyed) and pinned to 'bota', so
            # the selector is discoverable and FT_ALGO=rnea's absence is visible
            # rather than silent. The reader thread does the swap+re-tare.
            with self.state.lock:
                _algo0 = self.state.algo
            _rnea_hint = (f"no {self._rnea_basename} — RNEA unavailable "
                          "(run estimate_payload / seed the config)")
            self.gui_algo = gui.add_dropdown(
                "Compensation algorithm", ("bota", "rnea"),
                initial_value=_algo0 if self.rnea_available else "bota",
                disabled=not self.rnea_available,
                hint=None if self.rnea_available else _rnea_hint)

            @self.gui_algo.on_update
            def _(_event) -> None:
                # Ignore the echo when WE reconcile the value back to the actual
                # engine; only a real user pick should request a swap. A disabled
                # dropdown can't be changed, so this only fires when RNEA exists.
                if self._algo_echo or not self.rnea_available:
                    return
                with self.state.lock:
                    self.state.algo_request = self.gui_algo.value

            # Recompensation (payload re-estimate). ALWAYS SHOWN (R3); DISABLED
            # when no estimator/compensator config is wired (a raw-only reader
            # never consumes recal_request, so it would wedge). The status text
            # says why when disabled, so the control is discoverable.
            self.gui_recomp_status = gui.add_text(
                "Payload", initial_value="ready" if self.recomp_enabled
                else "no payload config — run estimate_payload.py to enable", disabled=True)
            recomp_btn = gui.add_button("Recompensation (90s)", disabled=not self.recomp_enabled)

            @recomp_btn.on_click
            def _(_event) -> None:
                self.arm_recomp(scene)

    # -- controls -----------------------------------------------------------
    def request_tare(self) -> None:
        """Zero this hand's wrench. Safe from any thread (flag under the lock).

        Also used by the recording START/STOP edges, so both hands begin every
        episode zeroed at their own free-space pose."""
        with self.state.lock:
            if self.state.compensation_available:
                # Compensated stream: the reader re-tares the ACTIVE engine and
                # then zeroes the display offset itself. Setting the offsets here
                # too would double-subtract for the ~0.5 s until the tare settles.
                self.state.tare_request = True
            else:
                self.state.force_offset = self.state.force.copy()
                self.state.torque_offset = self.state.torque.copy()

    def arm_recomp(self, scene: "CollectionScene") -> None:
        """Ask this hand's reader thread to re-estimate its payload.

        Refuses while EITHER hand is mid-cycle. The whole check-then-arm runs
        under the scene's gate so two near-simultaneous L+R clicks (two browser
        callback threads) cannot both pass the check and start."""
        if not self.recomp_enabled:
            return
        with scene._recomp_gate:
            busy = scene._recomp_busy_panel()
            if busy is not None:
                # Same hand -> a double-click on a running cycle: silently ignore.
                if busy is not self:
                    self.gui_recomp_status.value = (
                        f"{busy.label} hand is recompensating — wait for it to finish")
                return
            # Refuse if the reader isn't streaming, else recal_request is never
            # consumed and the overlay wedges at countdown forever (the reader is
            # its only consumer). Reset elapsed/message so the first overlay frame
            # shows a sane value, not a stale number from the previous run.
            with self.state.lock:
                if not self.state.connected:
                    self.gui_recomp_status.value = "sensor not streaming — cannot recompensate"
                    return
                self.state.recal_request = True
                self.state.recal_phase = "countdown"
                self.state.recal_elapsed = 0.0
                self.state.recal_message = ""
        # recomp_armed_at (the stuck-detector clock) is armed LAZILY on the render
        # thread in update_recomp_overlay — never written from this callback thread.

    # -- per-frame ----------------------------------------------------------
    def recomp_active(self) -> bool:
        """True while this hand's reader is running a payload estimate."""
        with self.state.lock:
            return self.state.recal_phase in ("countdown", "estimating")

    def refresh_status(self) -> None:
        """Update the Status row + reconcile the engine dropdown. Render thread."""
        with self.state.lock:
            conn = self.state.connected
            fmag = float(np.linalg.norm(self.state.force))
            rate = self.state.rate_hz
            comp = self.state.compensation_available and self.state.show_compensated
            algo = self.state.algo
            algo_status = self.state.algo_status
        # Tag the header with the live wrench mode + the ACTIVE compensation algo
        # so it's unambiguous which signal the |F| / charts show (raw vs
        # payload-compensated, and which engine). A ⚠ marks any reader caveat
        # (RNEA wrench_sign unverified, or an RNEA->bota fallback).
        mode = (f"comp:{algo}" if comp else "raw")
        warn = "  ⚠" if (comp and algo_status) else ""
        if self.gui_status is not None:
            self.gui_status.value = (f"● online  |F|={fmag:.1f}N  {rate:.0f}Hz  [{mode}]{warn}"
                                     if conn else "○ offline")
        # Reconcile the dropdown to the engine the reader ACTUALLY runs (after a
        # fallback the widget would otherwise lie). Echo-suppressed so it doesn't
        # re-request a swap; viser only re-renders on a real change.
        if self.gui_algo is not None and self.gui_algo.value != algo:
            self._algo_echo = True
            try:
                self.gui_algo.value = algo
            finally:
                self._algo_echo = False

    def update_recomp_overlay(self) -> None:
        """Drive this hand's compact Payload readout + its fullscreen modal.

        SINGLE-OWNER: the render thread owns the WHOLE modal lifecycle (open,
        text, close) and the two timers. The reader thread only WRITES
        recal_phase/elapsed/message, and the Close button only flips the phase
        back to "idle" under the lock — it never touches the modal handle. That
        mirrors the file's established "buttons raise flags, the render loop
        mutates" pattern and avoids the cross-thread race where the callback nulls
        the handle mid-deref."""
        with self.state.lock:
            phase = self.state.recal_phase
            elapsed = self.state.recal_elapsed
            message = self.state.recal_message

        # Stuck-detector: while armed, if the reader never advances out of
        # countdown (dead/stalled thread — it is the only consumer of
        # recal_request), force a 'failed' phase so the operator gets a Close
        # button instead of a perpetual fullscreen modal. Generous window:
        # countdown + a few seconds before the estimator's first tick. The clock is
        # armed LAZILY here (render thread) on the first countdown frame, not in
        # the click callback, so recomp_armed_at stays single-owner.
        if phase == "countdown":
            if self.recomp_armed_at == 0.0:
                self.recomp_armed_at = time.perf_counter()
            if (time.perf_counter() - self.recomp_armed_at) > 12.0:
                with self.state.lock:
                    if self.state.recal_phase == "countdown":  # still stuck
                        self.state.recal_phase = phase = "failed"
                        self.state.recal_message = message = (
                            "sensor not responding — is the F/T reader still streaming?")

        # Compact panel readout (mirrors the overlay but always visible).
        if self.gui_recomp_status is not None:
            if phase == "countdown":
                self.gui_recomp_status.value = f"starting in {math.ceil(elapsed) if elapsed > 0 else 0}s…"
            elif phase == "estimating":
                self.gui_recomp_status.value = f"estimating… {int(elapsed)}s (rotate the tool)"
            elif phase == "failed":
                self.gui_recomp_status.value = "last run failed — retry"
            elif phase in ("idle", "done"):
                self.gui_recomp_status.value = "compensated" if phase == "done" else "ready"

        active = phase in ("countdown", "estimating", "done", "failed")

        # Open the modal when a cycle starts (idle -> active).
        if active and self.recomp_modal is None:
            self.recomp_modal = self.server.gui.add_modal(
                f"Payload recompensation ({self.hand})")
            self.recomp_done_at = 0.0  # reset the done-linger timer for this cycle
            with self.recomp_modal:
                self.recomp_md = self.server.gui.add_markdown("Starting…")
                close_btn = self.server.gui.add_button("Close")

                @close_btn.on_click
                def _(_e) -> None:
                    # Thread-safe: only flip the phase to idle (under the lock).
                    # The render thread sees idle next tick and closes the modal —
                    # the callback thread never touches the modal handle.
                    with self.state.lock:
                        if self.state.recal_phase in ("done", "failed"):
                            self.state.recal_phase = "idle"

        if self.recomp_modal is not None and self.recomp_md is not None:
            self.recomp_md.content = _recomp_markdown(phase, elapsed, message, hand=self.hand)

        # Auto-close shortly after a successful done (failed waits for Close so the
        # operator can read the error). 'done' lingers ~1.5 s, then we flip to idle
        # and the branch below closes it next tick.
        if phase == "done":
            if self.recomp_done_at == 0.0:
                self.recomp_done_at = time.perf_counter()
            elif (time.perf_counter() - self.recomp_done_at) > 1.5:
                with self.state.lock:
                    self.state.recal_phase = "idle"

        # Close + tear down when the cycle ends. This is the ONLY place the handle
        # is nulled, always on the render thread.
        if not active and self.recomp_modal is not None:
            self.close_recomp_overlay()

    def close_recomp_overlay(self) -> None:
        """Close + null this hand's modal and re-arm its timers. Render-thread only."""
        if self.recomp_modal is not None:
            try:
                self.recomp_modal.close()
            except Exception:  # noqa: BLE001
                pass
        self.recomp_modal = None
        self.recomp_md = None
        self.recomp_armed_at = 0.0
        self.recomp_done_at = 0.0

    def refresh_charts(self) -> None:
        """Push new wrench samples into this hand's four charts. Render thread."""
        with self.state.lock:
            ft_n = self.state.n_frames
            ft_exported = self.state.history.export()
        if ft_n != self._last_ft_n and ft_exported is not None:
            self._last_ft_n = ft_n
            fx, ch = ft_exported
            if fx.size >= 2:
                f = ch["force"]
                t = ch["torque"]
                self._force_chart.data = (fx, f[0], f[1], f[2])
                self._torque_chart.data = (fx, t[0], t[1], t[2])
                # Peak-hold the y-axis: start at [-10, 10], widen to the running
                # magnitude peak across the visible window, never shrink. Only
                # re-push .scales when the peak actually grows (avoids per-frame
                # client churn while the signal stays inside the window).
                self._force_y_peak = CollectionScene._peak_hold_y(
                    self._force_chart, self._force_y_peak, f)
                self._torque_y_peak = CollectionScene._peak_hold_y(
                    self._torque_chart, self._torque_y_peak, t)

        # Always-raw wrench charts (R3), from the reader's raw_history buffer.
        with self.state.lock:
            raw_n = self.state.raw_n_frames
            raw_exported = self.state.raw_history.export()
        if raw_n != self._last_raw_ft_n and raw_exported is not None:
            self._last_raw_ft_n = raw_n
            rfx, rch = raw_exported
            if rfx.size >= 2:
                rf = rch["force"]
                rt = rch["torque"]
                self._raw_force_chart.data = (rfx, rf[0], rf[1], rf[2])
                self._raw_torque_chart.data = (rfx, rt[0], rt[1], rt[2])
                self._raw_force_y_peak = CollectionScene._peak_hold_y(
                    self._raw_force_chart, self._raw_force_y_peak, rf)
                self._raw_torque_y_peak = CollectionScene._peak_hold_y(
                    self._raw_torque_chart, self._raw_torque_y_peak, rt)


# ----------------------------------------------------------------------------
# Scene
# ----------------------------------------------------------------------------


class CollectionScene:
    """Owns the viser server, the G1 URDF, the camera frustum, and the GUI."""

    def __init__(self, server: viser.ViserServer, state: CollectionState, ik, mapper,
                 yam_urdf: str, cam_offset_xyz, cam_offset_rpy, cam_fov_deg: float,
                 robot_height: float = 0.0, auto_calibrate: bool = True, ft_state=None,
                 gripper_send=None, recomp_enabled: bool = False, recorder_client=None,
                 rnea_available: bool = False, tracking_mode: str = "1pt",
                 display_mode: str = "humanoid", reach_scale: float = 1.0,
                 raw_frame: bool = False, has_right_camera: bool = False,
                 has_right_gripper: bool = False, has_right_ft: bool = False,
                 ft_state_right=None, gripper_right_send=None,
                 recomp_enabled_right: bool = False, rnea_available_right: bool = False,
                 show_camera_view: bool = True, camera_preview_hz: float = 10.0):
        self.server = server
        self.state = state
        # Initial state of the "Show camera view" checkbox (--no-camera-view / CAMERA_VIEW=0).
        # Mirrored into state.preview_images HERE rather than only in the checkbox's
        # on_update, because on_update fires on operator clicks — not at construction —
        # so a --no-camera-view launch would otherwise start with the readers still
        # decoding every frame until the first manual toggle.
        self._show_camera_view = bool(show_camera_view)
        state.preview_images = self._show_camera_view
        # Last value written to the frustums' `.visible`. None = never written, so the
        # first render tick always pushes the real state (including a --no-camera-view
        # boot, where the nodes are created visible and must be hidden once).
        self._last_show_cam: bool | None = None
        # Frustum REPAINT throttle (--camera-preview-hz). <= 0 means "every frame the
        # bridges publish", i.e. the historical behaviour. Deadlines start at -inf so the
        # first frame of each camera paints immediately — an operator watching for the
        # first picture must not wait out a period, and on a 1 Hz preview that would look
        # like a dead camera.
        self._cam_paint_period = (1.0 / camera_preview_hz) if camera_preview_hz > 0 else 0.0
        self._last_cam_paint_t = float("-inf")
        self._last_cam_paint_t_right = float("-inf")
        self.ik = ik          # ThreePointIK (all modes, restricted by active_roles) or None (gripper-only / mock-without-IK)
        self.mapper = mapper  # YamConfigMapper or None
        # Display mode selects WHAT is drawn (not what is recorded — recording is
        # untouched by this). "humanoid": the full G1 URDF driven by mink IK (the
        # historical behavior). "gripper": ONLY the three yam-gripper meshes per
        # active hand + the camera frustum, placed by a home-anchored delta with NO
        # mink IK (ik is None). See gripper_only_view.py. Camera-to-gripper geometry
        # is the URDF's in BOTH modes (the frustum is parented under the same follow
        # frame as the gripper meshes), so it is preserved by construction.
        self._display_mode = display_mode
        # RAW-FRAME display (gripper-only mode only): draw the cluster at the puck's
        # RAW libsurvive pose + the fused remapped-TCP, with NO home-anchor. Ignored
        # in humanoid mode (which has no gripper-only follow frames). Display only —
        # recording is a separate process and always stores raw tracker.npy.
        self._raw_frame = bool(raw_frame) and (display_mode == "gripper")
        self._yam_urdf_path = yam_urdf
        # reach_scale scales the home-anchored position delta in gripper mode (the
        # IK owns its own copy in humanoid mode; this is only read when ik is None).
        self._reach_scale = float(reach_scale)
        # Tracking mode (1pt/2pt/3pt/5pt) + the resolved active roles. LEFT-canonical:
        # the PRIMARY (main-arm) hand is 'left' in every mode; it drives the main
        # IK arm, the camera-follow frame, and the jaw. 2pt adds 'right', 3pt adds
        # 'head', 5pt adds the two FEET. Secondary trackers render as raw-pose triads
        # (operator feedback).
        self.tracking_mode = tracking_mode
        self._active_roles = _ROLES_BY_MODE.get(tracking_mode, ("left",))
        # The BODY REFERENCES active this session (chest + feet), in wire order. Used by
        # every triad / status / anchor loop instead of `role == "head"`.
        self._body_roles = tuple(r for r in self._active_roles if r in _BODY_ROLES_ALL)
        # WHICH right-hand devices exist this session (2pt/3pt). Passed in rather than
        # inferred from the URLs so the scene has one truth to draw from, and so the
        # right-hand frustum / status rows appear only for hardware that is actually
        # there — an empty frustum floating on the right hand reads as a broken camera.
        # A BODY REFERENCE never has any of these (chest, and the two 5pt feet).
        self._has_right_camera = bool(has_right_camera) and "right" in self._active_roles
        self._has_right_gripper = bool(has_right_gripper) and "right" in self._active_roles
        self._has_right_ft = bool(has_right_ft) and "right" in self._active_roles
        # The robot wrist/site/jaw the PRIMARY hand drives. LEFT-canonical -> left.
        self._primary_wrist_link = "left_wrist_yaw_link"
        self._primary_hand_site = "left_hand_site"
        # Optional out-of-process recorder control client (None = recording UI off).
        # The GUI buttons only send intents over its WS; the recorder process owns
        # all disk I/O, keeping recording off the viz interpreter (GIL) entirely.
        self._recorder = recorder_client
        # Callable(dict) that ships a control frame to the gripper bridge over the
        # encoder WS (open/close/stop + slider). None when there is no gripper
        # connection (--mock), in which case the Gripper controls are hidden. The
        # right hand's is the SAME contract on the right bridge's own WS — that
        # bridge runs with --web-control too, so the panel drives each hand's jaws
        # independently (never mirrored: the idle hand must not record a grasp).
        self._gripper_send = gripper_send
        self._gripper_right_send = gripper_right_send if self._has_right_gripper else None
        # Optional bota F/T SensorStates (read in-process via bota_ft_viz), one per
        # hand. When present, that hand gets a status row, Force/Torque charts fed
        # from its rolling history, and its OWN tare / compensated / algorithm /
        # recompensation controls — see _FTPanel, which owns all of it. None -> that
        # hand has no F/T block at all (sensor absent / --no-ft / 1pt).
        self.ft_state = ft_state              # LEFT (primary); the historical field
        self.ft_state_right = ft_state_right if self._has_right_ft else None
        # Whether each Recompensation button is meaningful (a Bota estimator config
        # was wired to that reader). Set by main() from the resolved configs.
        self._recomp_enabled = recomp_enabled
        self._recomp_enabled_right = recomp_enabled_right
        # Whether the RNEA compensation-algorithm option should be offered for that
        # hand (its per-side RNEA config exists).
        self._rnea_available = rnea_available
        self._rnea_available_right = rnea_available_right
        # The per-hand F/T panels, built in _build_gui. Ordered left-then-right so
        # every folder/chart/row lands in a stable place in the sidebar. Each owns
        # its own controls, charts and recompensation modal (see _FTPanel).
        self._ft_panels: list = []
        self._ft_panel_left = None
        self._ft_panel_right = None
        # Serializes the "only ONE hand recompensates at a time" check-then-arm.
        # Two browser callback threads (an L and an R click in the same instant)
        # would otherwise both pass the check and start two 90 s estimates that
        # fight for the interpreter — neither converging. Same gate as
        # the bimanual deploy viz's ``_recomp_gate``.
        self._recomp_gate = threading.Lock()
        # Recording-event center overlay (mirrors the recompensation modal pattern,
        # single-owner = render thread). A 2 s fullscreen "REC ● 开始录制 episode N" /
        # "✓ 已保存 episode N" banner fired on the recorder's idle<->recording edge
        # (whether triggered by the s/e keys or the GUI buttons). _rec_modal holds
        # the open handle (None == closed); _rec_overlay_until is its expiry time;
        # _rec_prev_recording tracks the last seen recording flag for edge detection.
        self._rec_modal = None
        self._rec_md = None
        self._rec_overlay_until = 0.0
        self._rec_prev_recording = False
        self._rec_overlay_secs = 2.0
        # 'c'-key home observer: last seen recorder home_seq (-1 = no baseline yet,
        # so the first status snapshot is absorbed rather than firing a reset).
        self._last_home_seq = -1
        self._cam_offset_xyz = np.asarray(cam_offset_xyz, dtype=float)
        self._cam_offset_rpy = np.asarray(cam_offset_rpy, dtype=float)
        self._cam_fov = math.radians(cam_fov_deg)
        self._last_cam_seq = -1
        self._last_cam_seq_right = -1
        self._frustum_right = None   # created in _build_gripper_nodes when configured
        self._last_chart_n = -1
        self._last_chart_n_right = -1
        self._enc_chart_right = None   # created in _build_gui when a right gripper exists
        # Last solved HuMI q (None until first IK step); lets the jaws render from
        # the live gripper fraction even when arm tracking is paused/stale.
        self._last_q = None
        # Auto-calibrate on the first live (non-stale) tracker frame so the hand
        # tracks immediately without the operator having to click Recalibrate.
        # (Re-armed by Reset; the button still lets you re-anchor any time.)
        self._auto_calibrate = auto_calibrate
        self._did_auto_calibrate = False
        # Captured puck->wrist MOUNT rotations, {role: quat_wxyz}, from the
        # "Reset tracker axes to wrist" button. Empty until then; every consumer treats
        # absent as identity, so the default scene is unchanged.
        self._mount_quats: dict = {}

        server.scene.set_up_direction("+z")
        server.scene.add_grid("/grid", plane="xy", shadow_opacity=0.0, infinite_grid=True, cell_size=0.25)

        # Whether the full humanoid is drawn. "gripper" mode draws ONLY the yam
        # gripper meshes (no URDF, no mink IK — ik is None); "humanoid" draws the
        # whole G1 driven by IK, exactly as before. Set once here so the rest of
        # __init__ / update() branch on a single flag.
        self._display_humanoid = (self._display_mode != "gripper")
        # Which active roles actually carry a gripper (+ camera + the three-triad
        # cluster). A BODY-REFERENCE puck (chest, feet) has no gripper, camera or TCP, so
        # it gets no meshes — only a bare axis triad.
        self._grip_roles = tuple(r for r in self._active_roles if r in _GRIP_ROLES_ALL)
        # Which roles get a home-anchored FOLLOW FRAME in gripper-only mode. This is
        # every active role, not just the gripper ones: the chest triad has to live in
        # the same display world as the two hands, or it renders at libsurvive's raw
        # world pose metres away from them and reads as a broken tracker. So 3pt =
        # 2 gripper clusters + 1 chest triad, all anchored by the same delta.
        self._view_roles = tuple(self._active_roles)
        # Gripper-only anchor state (mirrors the IK's per-role recalibrate refs, but
        # with no solver): role -> (ref_pos, ref_quat) captured on Recalibrate /
        # first live frame; the live pose is mapped onto a fixed home wrist pose by
        # the SAME home-anchored delta the IK uses (gripper_only_view.home_anchor).
        self._grip_ref: dict = {}
        self._grip_calibrated = False
        self._grip_geom: dict = {}       # role -> GripperGeom (meshes + wrist-rel poses)
        self._grip_nodes: dict = {}      # role -> {"base":h, "finger1":h, "finger2":h}
        self._grip_follow: dict = {}     # role -> viser frame the meshes hang under

        from pathlib import Path

        if self._display_humanoid:
            # Robot URDF, rendered at the stand pose.
            from viser.extras import ViserUrdf

            # Render the URDF root (pelvis) at the IK SOLVER base position so that
            # solver-world == viser-world: the IK'd wrist/site coords (used to place
            # the camera-follow frame) then line up with the rendered robot, AND the
            # feet sit on the grid instead of the pelvis (ViserUrdf only sets joint
            # angles, not the floating base, so without this the pelvis renders at
            # z=0 and the grid cuts through the waist). robot_height is an extra trim.
            if ik is not None:
                base_xyz = np.asarray(ik.home_q()[0:3], dtype=float).copy()
                base_xyz[2] += robot_height
            else:
                base_xyz = np.array([0.0, 0.0, robot_height])
            self._robot_frame = server.scene.add_frame(
                "/robot", show_axes=False, position=tuple(base_xyz)
            )
            # ViserUrdf requires a pathlib.Path (a str is treated as a yourdfpy.URDF
            # object and trips an assert); meshes resolve relative to the urdf file.
            self._urdf = ViserUrdf(server, urdf_or_path=Path(yam_urdf), root_node_name="/robot")
            self._yam_names = list(self._urdf.get_actuated_joint_names())
        else:
            # Gripper-only: no /robot, no ViserUrdf, no mink. Load the per-hand
            # gripper geometry (meshes + wrist-relative endpoint poses) once from the
            # URDF via yourdfpy; the mesh scene nodes are created AFTER the wrist
            # follow-frames exist (see _build_gripper_nodes below). One URDF load is
            # shared across both hands.
            self._robot_frame = None
            self._urdf = None
            self._yam_names = []
            sys.path.insert(0, os.path.dirname(__file__))
            import gripper_only_view as gov

            shared_urdf = gov._load_urdf(yam_urdf)
            for role in self._grip_roles:
                self._grip_geom[role] = gov.build_gripper_geom(role, urdf=shared_urdf)

        # Wrist follow-frame: a small triad placed at the live right WRIST-YAW
        # link pose (the gripper mounts rigidly off this link), with the camera
        # frustum parented under it so the camera moves with the gripper.
        self._tracker_frame = server.scene.add_frame(
            "/tracker", show_axes=False, axes_length=0.08, axes_radius=0.004,
            origin_radius=0.01,
        )
        # Local rotation of the puck ("tracker") child triad every HAND follow frame
        # carries (see _add_hand_triads). The follow frame is a WRIST frame in BOTH paths
        # and the puck is bolted on rotated by M, so this is unconditional — see
        # _puck_child_quat. Computed ONCE here and handed to every hand, so 2pt/3pt cannot
        # diverge from the validated MODE=1pt hand.
        self._puck_child_q = _puck_child_quat()
        self._tracker_triad, self._tracker_true_triad, self._wrist_yaw_triad = \
            self._add_hand_triads("/tracker")
        # Camera mount: rigidly fixed to the wrist-yaw frame. The wrist frame's
        # +x is the gripper approach axis (the hand site sits +0.235 m along +x),
        # so an offset of +x ~0.10 m sits beside the yam gripper and +z lifts it
        # a touch above the barrel. viser frustums use the OpenCV convention
        # (+z = view/forward); the mount rotation makes +z point OUTWARD along
        # wrist +x (looking the way the gripper points). The default also rolls
        # +90deg about x (CCW) so the image is oriented as requested. The rotation
        # is composed Rx(roll)*Ry(yaw) from --cam-rpy = (roll_x, yaw_y, _).
        from viser import transforms as vtf

        rpy = self._cam_offset_rpy
        cam_rot = (vtf.SO3.from_rpy_radians(float(rpy[0]), 0.0, 0.0)
                   .multiply(vtf.SO3.from_rpy_radians(0.0, float(rpy[1]), 0.0)))
        cam_wxyz = cam_rot.wxyz
        self._cam_mount = server.scene.add_frame(
            "/tracker/cam", show_axes=False, position=tuple(self._cam_offset_xyz), wxyz=tuple(cam_wxyz),
        )
        # Initial aspect (updated per-frame from the actual camera dimensions).
        self._cam_aspect = 4.0 / 3.0  # 640x480 default
        self._frustum = server.scene.add_camera_frustum(
            "/tracker/cam/view", fov=self._cam_fov, aspect=self._cam_aspect,
            scale=0.12, color=(40, 200, 255),
        )
        # Labelled L only when there IS a right camera to confuse it with, so a 1pt
        # scene is unchanged. Which physical D405 is 'left' cannot be auto-detected —
        # this label plus the orange 'camera R' is the operator's 5-second L/R check.
        if self._has_right_camera:
            server.scene.add_label("/tracker/cam/view/label", "camera L")

        # Secondary tracker triads (2pt adds RIGHT, 3pt also HEAD). These are RAW
        # world-pose axes (operator feedback that the extra trackers are live +
        # oriented right), NOT IK-driven links — the primary /tracker frame above
        # is the only camera/jaw follow-frame. Created only for the active roles so
        # 1pt shows just the primary tracker.
        self._sec_frames: dict[str, object] = {}
        for role in self._active_roles:
            if role == "left":
                continue  # primary hand == /tracker frame above
            self._sec_frames[role] = server.scene.add_frame(
                f"/tracker_{role}", show_axes=False, axes_length=0.07, axes_radius=0.004,
                origin_radius=0.01,
            )
            if role in self._grip_roles:
                # A HAND: the SAME three labelled triads the primary hand carries (TCP,
                # puck, wrist-yaw), at the same offsets. The right hand used to get a
                # single unlabelled TCP triad even though its follow frame is seated by
                # the same follow_pose_binding_puck() map — which assumes a puck child
                # at _TRACKER_TRUE_XYZ_M — so the seat the binding pivots about was
                # invisible on that hand only. Both hands are now identical.
                self._add_hand_triads(f"/tracker_{role}")
            elif self._display_humanoid:
                # CHEST, humanoid display: unchanged historical cosmetic +x offset.
                server.scene.add_frame(
                    f"/tracker_{role}/triad", show_axes=True, axes_length=0.07,
                    axes_radius=0.004, origin_radius=0.01,
                    position=(_TCP_ALONG_WRIST_X_M, 0.0, 0.0),
                )
            else:
                # A BODY REFERENCE, gripper-only display: the ONE thing 3pt adds over 2pt
                # (the chest), and the two more 5pt adds (the feet). A bare triad AT THE
                # FRAME ORIGIN — there is no gripper, camera, TCP or wrist on any of them,
                # so the +0.242 m TCP lever the hands carry would draw the axis 24 cm out
                # in front of the body part.
                #
                # No local rotation: the mount relabel H lives in the FRAME
                # (raw_body_pose_from_puck / the home anchor), not in this child, so the
                # triad reads whatever frame the recording expresses. Labelled "chest" /
                # "foot L" / "foot R" rather than "... tracker" because in RAW mode it is
                # the BODY frame (== the stored keypoint), not the puck's own axes — the
                # hands' naming, where the raw-puck triad is the one called "tracker".
                server.scene.add_frame(
                    f"/tracker_{role}/triad", show_axes=True, axes_length=0.09,
                    axes_radius=0.004, origin_radius=0.011,
                )
                server.scene.add_label(f"/tracker_{role}/triad/label",
                                       _BODY_ROLE_LABEL.get(role, role))
            # RIGHT hand only: its own camera mount + frustum, geometrically IDENTICAL
            # to the primary hand's (same offset, same rpy, same fov) because it is the
            # same D405 on the same 3D-printed mount — only the parent frame differs.
            # A DIFFERENT colour so the two views are never confused, and a text label,
            # since which physical camera is 'right' is a PHYSICAL fact the operator has
            # to confirm (serials cannot tell you which hand a camera is taped to).
            if role == "right" and self._has_right_camera:
                server.scene.add_frame(
                    "/tracker_right/cam", show_axes=False,
                    position=tuple(self._cam_offset_xyz), wxyz=tuple(cam_wxyz),
                )
                self._frustum_right = server.scene.add_camera_frustum(
                    "/tracker_right/cam/view", fov=self._cam_fov, aspect=self._cam_aspect,
                    scale=0.12, color=(255, 170, 40),
                )
                server.scene.add_label("/tracker_right/cam/view/label", "camera R")

        # Gripper-only meshes: parent the three gripper pieces per hand UNDER the
        # same follow frame the camera hangs on (primary -> /tracker, right ->
        # /tracker_right), so the camera-to-gripper geometry is identical to
        # humanoid mode and moves together. Done here, after those frames exist.
        if not self._display_humanoid:
            self._build_gripper_nodes()

        self._build_gui()
        # Initial pose = home.
        self._apply_home_pose()

    def _add_hand_triads(self, parent: str):
        """Add the three labelled child triads ONE hand's follow frame carries.

        **MODE=1pt's LEFT hand is the reference and it is validated — do not "fix" the
        numbers here.** Every hand in every mode is drawn by this one call with the same
        offsets, the same labels and the same `_puck_child_q`, so 2pt/3pt are copies of
        that hand by construction rather than by agreement. smoke_test_collection.py
        section 7 pins it against a REAL 1pt scene, child node by child node:

          * ``triad``        — the GRIPPER CENTRE (TCP/grasp point), wrist +x
            ``_TCP_ALONG_WRIST_X_M`` = 0.242 m, matching the offline inference viewer's
            EE triad. NOT the wrist-yaw origin.
          * ``tracker_true`` — the physical VIVE puck's most accurate origin (from the
            TCP, back 22.5 cm and up 5 cm), carrying ``_puck_child_q`` = M^-1 in EVERY
            path (the puck is bolted on rotated). This is also the point the
            live follow-frame binding pivots about (``follow_pose_binding_puck``), so it is
            the one triad that reads the pose the recorder stores.
          * ``wrist_yaw``    — the wrist-yaw link origin, i.e. (0,0,0) in this frame:
            the frame the gripper meshes / TCP / camera all hang off.

        All three are CHILDREN of ``parent``, so they ride the hand and inherit its
        orientation; only the drawn axes move. Display only — the meshes, camera and
        everything recorded are untouched. Returns ``(tcp, puck, wrist_yaw)`` handles.
        """
        scene = self.server.scene
        tcp = scene.add_frame(
            f"{parent}/triad", show_axes=True, axes_length=0.08, axes_radius=0.004,
            origin_radius=0.01, position=(_TCP_ALONG_WRIST_X_M, 0.0, 0.0),
        )
        scene.add_label(f"{parent}/triad/label", "TCP")
        # Slightly thinner/shorter axes than the TCP triad, each labelled, so the three
        # are distinguishable at a glance.
        puck = scene.add_frame(
            f"{parent}/tracker_true", show_axes=True, axes_length=0.07, axes_radius=0.0035,
            origin_radius=0.009, position=_TRACKER_TRUE_XYZ_M, wxyz=tuple(self._puck_child_q),
        )
        scene.add_label(f"{parent}/tracker_true/label", "tracker")
        wrist_yaw = scene.add_frame(
            f"{parent}/wrist_yaw", show_axes=True, axes_length=0.07, axes_radius=0.0035,
            origin_radius=0.009, position=_WRIST_YAW_XYZ_M,
        )
        scene.add_label(f"{parent}/wrist_yaw/label", "wrist_yaw_link")
        return tcp, puck, wrist_yaw

    def _build_gripper_nodes(self) -> None:
        """Register the gripper-mode follow frames + create the mesh nodes per hand.

        Two things, one pass over the view roles:

          * ``_grip_follow[role]`` — the frame the role's cluster hangs under and that
            ``_update_gripper_only`` seats each frame: the PRIMARY hand ('left') under
            ``/tracker`` (same frame the camera frustum uses), the RIGHT hand under
            ``/tracker_right``, the CHEST under ``/tracker_head``. The chest is in this
            dict too — it needs the same home-anchored placement as the hands even
            though it carries no meshes.
          * mesh nodes, for the two HANDS only. Placed at their WRIST-RELATIVE poses
            (so the cluster reads exactly like the URDF gripper); the fingers are
            re-positioned per frame from the live open fraction (LERP). Colours are the
            URDF's, baked into the trimesh by gripper_only_view."""
        for role in self._view_roles:
            follow = self._tracker_frame if role == "left" else self._sec_frames.get(role)
            if follow is None:
                continue
            self._grip_follow[role] = follow
            geom = self._grip_geom.get(role)
            if geom is None:
                continue          # chest: a bare puck triad, no gripper meshes
            parent = "/tracker" if role == "left" else f"/tracker_{role}"
            nodes = {}
            for piece in geom.pieces():
                nodes[piece.name] = self.server.scene.add_mesh_trimesh(
                    f"{parent}/grip_{piece.name}", piece.mesh,
                    position=tuple(piece.pos_closed), wxyz=tuple(piece.quat),
                )
            self._grip_nodes[role] = nodes

    def _seat_view_frame(self, role: str, follow, tgt_p, tgt_q) -> None:
        """Seat one gripper-mode follow frame so its PUCK triad lands on (tgt_p, tgt_q).

        A HAND's follow frame is the WRIST origin and carries the puck as a child at
        ``_TRACKER_TRUE_XYZ_M``, so the frame has to be offset back off the tracked pose
        (``follow_pose_binding_puck``) for the puck child to coincide with it.

        The CHEST has no wrist, no TCP and no mesh cluster — its triad IS the frame
        origin — so the binding is the identity. Running the hand offset on it would
        displace the chest axis by the puck->wrist lever for no reason."""
        if role in self._grip_roles:
            fp, fq = follow_pose_binding_puck(tgt_p, tgt_q)
        else:
            fp, fq = np.asarray(tgt_p, dtype=float), np.asarray(tgt_q, dtype=float)
        follow.position = tuple(fp)
        follow.wxyz = tuple(fq)

    def _apply_home_pose(self) -> None:
        if self.ik is not None and self.mapper is not None:
            q = self.ik.home_q()
            cfg = self.mapper.to_yam(q, 0.0, 0.0)
            self._urdf.update_cfg(np.asarray(cfg))
        # Seat the camera-follow frame on the WRIST at home, so the frustum is
        # mounted on the gripper from the very first render. /tracker is added with
        # no position (world origin = on the ground); without this, the camera sits
        # on the floor until the FIRST live tracker frame moves it — i.e. forever in
        # --mock-without-IK, while "Track right hand" is off, during recompensation,
        # or before the VIVE tracker streams. _place_follow_frame_at_wrist() is the
        # same pose the per-frame IK branch writes.
        self._place_follow_frame_at_wrist()
        # Gripper-only mode: no IK, so seat each view follow frame on its display HOME
        # (the anchor point the live tracker delta mirrors onto — a wrist for a hand, the
        # G1's pelvis for the chest; see display_home_pose), so the gripper + camera sit in
        # front from the first frame instead of on the floor. Same role as
        # _place_follow_frame_at_wrist() in humanoid mode.
        if not self._display_humanoid:
            for role in self._view_roles:
                follow = self._grip_follow.get(role)
                if follow is None:
                    continue
                if self._raw_frame:
                    # RAW mode has no comfortable "home": the cluster tracks the raw puck.
                    # Before any live frame it holds at the world origin (the puck child
                    # then reads (0,0,0)); the first live frame seats it at the raw pose.
                    follow.position = (0.0, 0.0, 0.0)
                    follow.wxyz = (1.0, 0.0, 0.0, 0.0)
                    continue
                hp, hq = display_home_pose(role)
                # Seat through the SAME puck-binding map as the live drive (Problem 2A):
                # the home target IS a tracked pose, so the puck triad must land on it at
                # home too. Applying it here (and at the reset re-seat) keeps home == the
                # first live frame (cur==ref -> tgt==home), with no ~5 cm jump when
                # tracking starts. Display only — child offsets / meshes unchanged.
                self._seat_view_frame(role, follow, hp, hq)

    def _persist_mount_quats(self, mounts: dict) -> None:
        """Write the captured mount rotations into <session>/calibration/.

        A rotation held only in RAM dies on restart, and a session recorded under a
        different mount is otherwise indistinguishable from one recorded under this one.
        The converter's ``_resolve_calib`` already prefers a per-session calibration file
        over the repo default, so this is the lowest-friction persistence path. Best-effort
        and loud on failure: losing the file must not take the rig down mid-session, but it
        must not be silent either.
        """
        # The recorder owns the session root and publishes it on its status frame; the
        # viz has no filesystem view of it otherwise (they are separate processes).
        session = None
        if self._recorder is not None:
            session = (self._recorder.status() or {}).get("session_dir")
        if not session:
            print("[viz] mount rotations captured but NOT persisted: no recording session "
                  "yet (start a take first, or they are lost on restart).", flush=True)
            return
        try:
            d = os.path.join(session, "calibration")
            os.makedirs(d, exist_ok=True)
            # Literal, not fmt.TRACKER_MOUNT_FILE: the viz deliberately does not import
            # recording_format (it is a viz process, not a recorder). Kept in sync by
            # test_axes_reset_persists_the_mount_rotation.
            path = os.path.join(d, "tracker_mount.json")
            with open(path, "w") as f:
                json.dump({"mount_quat_wxyz": mounts,
                           "captured_wall": time.time(),
                           "_note": "puck->wrist mount rotation per role, from the viz "
                                    "'Reset tracker axes to wrist' button"}, f, indent=2)
            print(f"[viz] mount rotations persisted -> {path}", flush=True)
        except OSError as exc:
            print(f"[viz] WARNING: could not persist mount rotations: {exc}", flush=True)

    def _display_quat(self, role: str, quat_wxyz) -> "np.ndarray":
        """Tracker orientation with the captured mount rotation removed (identity-safe).

        Delegates to the IK object when it exists (single source of the math); falls back
        to the raw pose under --mock --no-ik, where there is no IK to hold the mount."""
        if self.ik is not None and hasattr(self.ik, "display_quat"):
            return self.ik.display_quat(role, np.asarray(quat_wxyz, dtype=float))
        return np.asarray(quat_wxyz, dtype=float)

    def _place_follow_frame_at_wrist(self) -> bool:
        """Move the camera-follow frame (/tracker) to the current solved PRIMARY
        WRIST-YAW link pose (LEFT-canonical -> left wrist). Returns True if it
        placed it (IK available). The frustum is parented under /tracker, so this
        keeps the camera on the gripper even when the per-frame IK branch is
        skipped."""
        if self.ik is None:
            return False
        wrist = self.ik._solver.configuration.get_transform_frame_to_world(
            frame_name=self._primary_wrist_link, frame_type="body"
        )
        self._tracker_frame.position = tuple(wrist.translation())
        self._tracker_frame.wxyz = tuple(wrist.rotation().wxyz)
        return True

    def _update_gripper_only(self, do_recal: bool, do_axes: bool, do_reset: bool,
                             live_poses) -> str:
        """Per-frame gripper-only placement (no humanoid, no mink IK).

        Drives EVERY view role — both hands AND, in 3pt, the chest puck (which gets a
        bare axis triad instead of a mesh cluster, but the same anchor, so it lands in
        the same display world as the hands rather than at libsurvive's raw pose).

        Two placement modes share the jaw-LERP tail (_animate_gripper_jaws):
          * DEFAULT (home-anchored): handles the SAME calibration handshake the
            humanoid path does (Recalibrate / first-frame auto-cal / Reset re-arms),
            then places each role's follow frame (a hand's carries the camera + the
            three gripper meshes) at ``home ⊕ (live tracker delta since the
            reference)`` — the SAME home-anchored mapping the IK uses, minus the solver.
          * RAW (self._raw_frame; early-return branch): NO anchor, NO calibration —
            seat the follow frame directly on the WRIST pose derived from the live RAW
            puck (raw_follow_pose_from_puck) so the puck/TCP/wrist triads + meshes read
            the recorded tracker.npy frame + the fused remapped-TCP. The chest has no
            wrist to derive, so it sits on the raw puck pose itself.
        Finger meshes are LERPed from the live open fraction. Returns a short pos
        string for the GUI readout.

        Moves ONLY scene nodes; never touches the recorder, gripper bridge, or CAN.
        A stale/paused tracker simply holds the last pose: ``live_poses()`` already
        drops stale/disconnected roles (see _live_poses), so a stale role is absent
        from ``poses`` and its follow frame is left untouched — the same hold-on-stale
        behavior the humanoid path gets from ThreePointIK.step()."""
        import gripper_only_view as gov

        poses = live_poses()
        view_active = [r for r in self._view_roles]

        def _all_view_live(p: dict) -> bool:
            return all(r in p for r in view_active)

        # RAW-FRAME mode: no home-anchor, no calibration, no Reset re-seat. Seat each
        # follow frame directly on the WRIST pose derived from the LIVE RAW puck (see
        # raw_follow_pose_from_puck); the puck/TCP/wrist child triads + gripper meshes +
        # camera ride along at their fixed wrist-relative offsets. Absent (stale/
        # disconnected) roles hold their last pose, same as the anchored path. The jaw
        # LERP below still runs. Recalibrate/Reset are inert here (nothing to anchor).
        if self._raw_frame:
            if do_recal or do_reset:
                # Recalibrate and 'c' (reset to home) are inert here BY DESIGN — raw frame
                # draws each puck's own libsurvive pose, so there is no home to snap back
                # to. Name the knob to drop: "no anchor" alone reads like a failure, and an
                # operator who presses 'c' and sees nothing move has no other way to learn
                # that RAW_FRAME=1 is what disabled it.
                self.gui_calib.value = (f"raw frame: no home to reset to "
                                        f"(drop RAW_FRAME=1) — {self.tracking_mode}")
            if do_axes:
                self.gui_calib.value = "axes reset N/A in raw-frame mode"
            rh_pos_str = ""
            if self.gui_track_enable.value:
                for role in view_active:
                    follow = self._grip_follow.get(role)
                    entry = poses.get(role)
                    if follow is None or entry is None:
                        continue
                    cur_p, cur_q = entry
                    if role in self._grip_roles:
                        fp, fq = raw_follow_pose_from_puck(cur_p, cur_q)
                    else:
                        # A BODY REFERENCE (chest, or a 5pt foot): no wrist and no lever,
                        # but its puck IS bolted on rotated, so the frame carries the BODY
                        # pose (puck ∘ H) — the same compose the converter applies to
                        # head_rot_axis_angle. Position is the raw puck's (H is
                        # rotation-only). The matrix is looked up PER ROLE, because "the
                        # feet are mounted like the chest" is a hardware claim that may
                        # stop being true for one bracket (see _BODY_RELABEL_M).
                        fp, fq = raw_body_pose_from_puck(role, cur_p, cur_q)
                    follow.position = tuple(fp)
                    follow.wxyz = tuple(fq)
                    if role == "left":
                        rh_pos_str = _fmt(np.asarray(cur_p, dtype=float))
            self._animate_gripper_jaws(self._grip_roles)
            return rh_pos_str

        # Reset re-arms auto-cal + drops the anchors (frames snap back to home on the
        # next _apply_home_pose-equivalent below).
        if do_reset:
            self._grip_ref = {}
            self._grip_calibrated = False
            self._did_auto_calibrate = False
            for role in view_active:
                follow = self._grip_follow.get(role)
                if follow is None:
                    continue
                hp, hq = display_home_pose(role)
                # Same puck-binding map as home/live (Problem 2A) so the reset lands the
                # puck triad on the home target, matching the first live frame.
                self._seat_view_frame(role, follow, hp, hq)

        # Recalibrate / first-frame auto-cal: capture every view role's current tracker
        # pose as the reference. Refuse a partial/stale capture (same gate as the IK
        # path) so we never anchor to a (0,0,0)/identity default. The chest is in the
        # gate too, matching ThreePointIK.recalibrate's all-active-roles requirement:
        # anchoring the hands while the chest puck is dark would leave that triad
        # referenced to nothing, and the launcher's wave gate already demands all three.
        want_cal = do_recal or (self._auto_calibrate and not self._did_auto_calibrate)
        if want_cal:
            if _all_view_live(poses):
                for role in view_active:
                    pos, quat = poses[role]
                    self._grip_ref[role] = (np.asarray(pos, dtype=float).copy(),
                                            np.asarray(quat, dtype=float).copy())
                self._grip_calibrated = True
                self._did_auto_calibrate = True
                tag = "calibrated" if do_recal else "auto-calibrated"
                self.gui_calib.value = f"{tag} (gripper, {self.tracking_mode})"
            elif do_recal:
                missing = [r for r in view_active if r not in poses]
                self.gui_calib.value = f"recal skipped: need live {'+'.join(missing)}"

        # "Reset tracker axes to wrist" has no mount-rotation math without the IK; be
        # explicit rather than silently doing nothing (the humanoid path owns it).
        if do_axes:
            self.gui_calib.value = "axes reset N/A in gripper mode (no IK)"

        # Place each view follow frame at home ⊕ live delta. Uncalibrated, a
        # stale/absent tracker, or tracking toggled off -> hold (leave the frame
        # where it is), matching the humanoid path's hold semantics.
        rh_pos_str = ""
        if self._grip_calibrated and self.gui_track_enable.value:
            for role in view_active:
                follow = self._grip_follow.get(role)
                entry = poses.get(role)
                ref = self._grip_ref.get(role)
                if follow is None or entry is None or ref is None:
                    continue
                hp, hq = display_home_pose(role)
                cur_p, cur_q = entry
                ref_p, ref_q = ref
                tgt_p, tgt_q = gov.home_anchor(hp, hq, ref_p, ref_q, cur_p, cur_q,
                                               reach_scale=self._reach_scale)
                # Bind the TRACKED pose to the "tracker" (puck) triad, not the wrist
                # origin (Problem 2A): seat the follow frame so the puck child lands on
                # (tgt_p, tgt_q). The readout below still reports tgt_p (the puck pose),
                # NOT follow.position — the follow frame is now offset from it.
                self._seat_view_frame(role, follow, tgt_p, tgt_q)
                if role == "left":
                    rh_pos_str = _fmt(tgt_p)

        self._animate_gripper_jaws(self._grip_roles)
        return rh_pos_str

    def _animate_gripper_jaws(self, grip_active) -> None:
        """LERP the finger meshes from the live open fraction (gripper-only mode).

        Decoupled from tracking so the jaws mirror the real gripper even while
        paused/stale/raw. Each hand is driven by its OWN encoder: the primary (LEFT)
        gripper by gripper_frac, the RIGHT one by gripper_frac_right when that second
        motor is present. Without a right gripper its jaws hold CLOSED — deliberately a
        fixed pose rather than a mirror of the left, so the scene never suggests a hand
        is grasping when nothing there is reporting. Moves only mesh scene nodes."""
        with self.state.lock:
            grip_frac = self.state.gripper_frac
            grip_frac_r = self.state.gripper_frac_right if self._has_right_gripper else 0.0
        for role in grip_active:
            nodes = self._grip_nodes.get(role)
            geom = self._grip_geom.get(role)
            if not nodes or geom is None:
                continue
            frac = grip_frac if role == "left" else grip_frac_r
            for piece in (geom.finger1, geom.finger2):
                nodes[piece.name].position = tuple(piece.pos_at(frac))

    def _build_gui(self) -> None:
        gui = self.server.gui
        gui.configure_theme(control_layout="collapsible", control_width="large")

        with gui.add_folder("Status"):
            # Each module shows a real-time online/offline dot + arrival rate, not
            # the old sticky "connected forever". A module that connects then goes
            # silent (bridge died, USB dropped) flips back to ○ offline within its
            # timeout — the failure data collection most needs to catch live.
            self.gui_tracker = gui.add_text("Tracker L (primary)", initial_value="○ waiting…", disabled=True)
            # Secondary tracker status lines, one per extra active role (2pt/3pt).
            self.gui_sec_tracker = {}
            if "right" in self._active_roles:
                self.gui_sec_tracker["right"] = gui.add_text("Tracker R", initial_value="○ waiting…", disabled=True)
            # One row per BODY REFERENCE, in wire order: "Tracker head" keeps its exact
            # historical label so a 3pt board is unchanged; the feet read "Tracker foot L/R".
            for _role in self._body_roles:
                self.gui_sec_tracker[_role] = gui.add_text(
                    "Tracker head" if _role == "head"
                    else f"Tracker {_BODY_ROLE_LABEL.get(_role, _role)}",
                    initial_value="○ waiting…", disabled=True)
            self.gui_gripper = gui.add_text(
                "Gripper L" if self._has_right_gripper else "Gripper",
                initial_value="○ waiting…", disabled=True)
            # RIGHT-hand rows exist ONLY when that device is present. An always-present
            # "○ offline" row for hardware that was never wired up trains the operator to
            # ignore the offline dot — which is the one signal that has to keep meaning
            # "something broke" during a recording session.
            self.gui_gripper_right = (
                gui.add_text("Gripper R", initial_value="○ waiting…", disabled=True)
                if self._has_right_gripper else None)
            self.gui_kbd = gui.add_text("Keyboard (2-key)", initial_value="○ waiting…", disabled=True)
            self.gui_cam = gui.add_text(
                "Camera L" if self._has_right_camera else "Camera",
                initial_value="○ waiting…", disabled=True)
            self.gui_cam_right = (
                gui.add_text("Camera R", initial_value="○ waiting…", disabled=True)
                if self._has_right_camera else None)
            self.gui_calib = gui.add_text("Calibration", initial_value="not calibrated", disabled=True)
            # One F/T row per sensor. Build the panels here (left first) so the rest
            # of _build_gui can add each one's charts + controls in order. With two
            # sensors BOTH rows are suffixed 'F/T L' / 'F/T R' — leaving the left one
            # unsuffixed would make it read as "the" sensor and the right as an extra.
            _dual_ft = self.ft_state is not None and self.ft_state_right is not None
            if self.ft_state is not None:
                self._ft_panel_left = _FTPanel(
                    self.server, self.ft_state, "left",
                    recomp_enabled=self._recomp_enabled, rnea_available=self._rnea_available)
                self._ft_panels.append(self._ft_panel_left)
            if self.ft_state_right is not None:
                self._ft_panel_right = _FTPanel(
                    self.server, self.ft_state_right, "right",
                    recomp_enabled=self._recomp_enabled_right,
                    rnea_available=self._rnea_available_right)
                self._ft_panels.append(self._ft_panel_right)
            for panel in self._ft_panels:
                panel.build_status_row(gui, _dual_ft)
            # Historical alias: the LEFT row. Kept because it is the one the
            # single-sensor code path (and the smoke test) knows by name.
            self.gui_ft = self._ft_panel_left.gui_status if self._ft_panel_left is not None else None

        # Header + readout label reflect the display mode: in gripper-only mode
        # there is no IK and the readout is the gripper follow-frame position (not a
        # robot hand site). The Recalibrate / Track toggle work in BOTH modes.
        _hand_hdr = (f"Primary hand: LEFT (tracker -> IK)  [{self.tracking_mode}]"
                     if self._display_humanoid
                     else f"Primary hand: LEFT (gripper-only, no IK)  [{self.tracking_mode}]")
        with gui.add_folder(_hand_hdr):
            self.gui_track_pos = gui.add_text("Tracker pos", initial_value="", disabled=True)
            self.gui_rh_pos = gui.add_text(
                "Robot hand pos" if self._display_humanoid else "Gripper pos",
                initial_value="", disabled=True)
            recal_btn = gui.add_button("Recalibrate (hold rest pose)")
            axes_btn = gui.add_button("Reset tracker axes to wrist")
            self.gui_track_enable = gui.add_checkbox("Track primary hand", initial_value=True)

        # One Gripper folder per hand. Keyed dicts (not two sets of _right fields)
        # so the readouts and the control wiring exist EXACTLY once — the same
        # per-side shape the bimanual deploy viz uses. A hand with no gripper bridge gets no
        # folder at all: an empty "Gripper R" reads as broken hardware.
        self.gui_grip_frac = {}
        self.gui_grip_rad = {}
        self.gui_grip_slider = {}
        self._suppress_slider_send = {}
        _dual_grip = self._has_right_gripper
        for _hand, _send in (("left", self._gripper_send), ("right", self._gripper_right_send)):
            if _hand == "right" and not self._has_right_gripper:
                continue
            with gui.add_folder(f"Gripper ({_hand})" if _dual_grip else "Gripper"):
                self.gui_grip_frac[_hand] = gui.add_text("Open fraction", initial_value="", disabled=True)
                self.gui_grip_rad[_hand] = gui.add_text("Encoder (rad)", initial_value="", disabled=True)
                # Browser-driven control (only when that hand's gripper bridge is
                # connected — both run with --web-control). These send intents back
                # over the encoder WS; the gripper process is the only CAN writer,
                # we never touch the bus from here. This is why the gripper no
                # longer needs a focused terminal or `xset`. Each hand drives ONLY
                # its own jaws — never mirrored, so the idle hand cannot record a
                # task-irrelevant grasp (same rule as the f/g vs j/k key pairs).
                if _send is None:
                    continue
                grip_open_btn = gui.add_button("Open gripper")
                grip_close_btn = gui.add_button("Close gripper")
                grip_stop_btn = gui.add_button("Stop")
                self.gui_grip_slider[_hand] = gui.add_slider(
                    "Target open %", min=0, max=100, step=1, initial_value=0
                )

                # Buttons send the intent DIRECTLY (so a click always fires, even
                # right after Stop), and also sync the slider for display. Viser
                # skips on_update when .value is assigned its current value, so
                # without the direct send "click Open again after Stop" (slider
                # already 100) would be a silent no-op. The _suppress_slider_send
                # flag stops the programmatic .value write from ALSO firing the
                # slider's on_update (which would double-send the same target).
                self._suppress_slider_send[_hand] = False

                # ``h=_hand, send=_send`` bind THIS iteration's hand into every
                # callback. Without the default-arg capture all four closures would
                # read the loop variable at click time — i.e. every button would
                # drive the RIGHT hand (the last iteration), silently crossing the
                # two grippers.
                def _send_frac_and_sync(frac: float, h=_hand, send=_send) -> None:
                    send({"target_frac": frac})
                    self._suppress_slider_send[h] = True
                    try:
                        self.gui_grip_slider[h].value = int(round(frac * 100))
                    finally:
                        self._suppress_slider_send[h] = False

                @grip_open_btn.on_click
                def _(_e, sync=_send_frac_and_sync) -> None:
                    sync(1.0)

                @grip_close_btn.on_click
                def _(_e, sync=_send_frac_and_sync) -> None:
                    sync(0.0)

                @grip_stop_btn.on_click
                def _(_e, send=_send) -> None:
                    # Freeze where it is; leave the slider display untouched.
                    send({"cmd": "stop"})

                @self.gui_grip_slider[_hand].on_update
                def _(_e, h=_hand, send=_send) -> None:
                    # Manual drags send; programmatic syncs (from the buttons) don't.
                    if self._suppress_slider_send[h]:
                        return
                    send({"target_frac": float(self.gui_grip_slider[h].value) / 100.0})

        with gui.add_folder("Camera"):
            # THE CPU LEVER, and the label says so because it is not guessable from the
            # widget: a live frustum image is not a cheap read-only overlay, it is a
            # per-frame JPEG decode (PIL) plus a JPEG RE-encode (viser's
            # CameraFrustumHandle.image setter re-encodes unconditionally, with no dirty
            # check) for EACH camera. Measured on the 3pt rig with two 640x480@30Hz
            # D405s: ~50% of one core with this on, ~26% with it off — i.e. the pixels
            # cost about as much as everything else this process does put together, and
            # they cost it even with no browser tab open, because the re-encode happens
            # on assignment rather than on send.
            #
            # Unchecking it does NOT touch recording: the recorder is its own process
            # with its own subscription to the same bridges and writes their JPEG bytes
            # verbatim. Turn it off once the frustums have confirmed which camera is on
            # which hand — after that the tracker axes are what you actually watch.
            self.gui_show_cam = gui.add_checkbox(
                "Show camera view (~24% of a core)", initial_value=self._show_camera_view)

            @self.gui_show_cam.on_update
            def _(_e) -> None:
                on = bool(self.gui_show_cam.value)
                # Tell the two reader threads to stop base64-decoding frames nobody
                # will look at. Set BEFORE dropping the stash so a frame landing in
                # between cannot re-stash on the way out.
                self.state.preview_images = on
                if not on:
                    # Drop what is already held. The readers stop refilling it, but a
                    # camera that stalls right now would otherwise leave its last frame
                    # pinned and repaint it — arbitrarily stale — on the way back on.
                    with self.state.lock:
                        self.state.cam_jpeg = None
                        self.state.cam_jpeg_right = None

        with gui.add_folder("Live plots"):
            self._enc_chart = self._make_enc_chart(
                "Gripper open fraction (left)" if _dual_grip else "Gripper open fraction")
            # Right jaw width gets its OWN chart rather than a second series on the
            # left's: the two encoder streams are separate bridges on separate
            # timers, so they have no common x grid to share.
            self._enc_chart_right = (
                self._make_enc_chart("Gripper open fraction (right)")
                if self._has_right_gripper else None)
            # Bota F/T charts (same style as bota_ft_viz): 3 x/y/z lines each, four
            # charts per SENSOR, owned by that hand's _FTPanel. With two sensors the
            # titles carry an L/R so a wrench trace can never be read as the other
            # hand's.
            for panel in self._ft_panels:
                panel.build_charts(self, _dual_ft)

        # F/T compensation controls, ONE folder per sensor: Tare, the
        # raw/compensated toggle, the engine dropdown and Recompensation. All of it
        # lives on _FTPanel so the right hand gets the identical set by
        # construction — see that class for the per-control semantics and for why
        # recompensation (only) is arbitrated across hands.
        for panel in self._ft_panels:
            panel.build_controls(gui, self, _dual_ft)


        # Recording controls (only when a recorder control client is wired). The
        # buttons just send intents to the out-of-process recorder over its WS;
        # the recorder owns all disk I/O. Status (REC ●, frame counts, elapsed) is
        # refreshed from the recorder's broadcast in update().
        if self._recorder is not None:
            with gui.add_folder("Recording"):
                self.gui_rec_task = gui.add_text("Task", initial_value="pick_place")
                self.gui_rec_status = gui.add_text("Recorder", initial_value="offline", disabled=True)
                # Manual per-episode model: one Start..Stop span is exactly ONE
                # episode (saved as take_NNN/; the converter emits the whole take as
                # a single episode when there are no markers). The keyboard's s/e
                # keys send the SAME start/stop control intents these buttons do; a
                # 2 s center banner fires on each edge (see _update_recording_overlay).
                rec_start_btn = gui.add_button("● Start episode (s)")
                rec_stop_btn = gui.add_button("■ Stop + save episode (e)")
                rec_discard_btn = gui.add_button("Discard last episode")

                @rec_start_btn.on_click
                def _(_e) -> None:
                    self._recorder.start_episode(task=str(self.gui_rec_task.value))

                @rec_stop_btn.on_click
                def _(_e) -> None:
                    self._recorder.stop_episode()

                @rec_discard_btn.on_click
                def _(_e) -> None:
                    self._recorder.discard_episode()

        with gui.add_folder("Controls"):
            self.gui_pause = gui.add_checkbox("Pause", initial_value=False)
            reset_btn = gui.add_button("Reset robot to home (c)")

        @recal_btn.on_click
        def _(_event) -> None:
            with self.state.lock:
                self.state.recal_request = True

        @axes_btn.on_click
        def _(_event) -> None:
            # Flag only: the render loop owns the IK object (see CollectionState).
            with self.state.lock:
                self.state.axes_reset_request = True

        @reset_btn.on_click
        def _(_event) -> None:
            # Route through the flag handshake (not a direct ik.reset_pose()) so
            # the _solver mutation happens on the render thread, never racing
            # update()'s ik.step() on the shared mink/mujoco Configuration.
            with self.state.lock:
                self.state.reset_request = True

    def _make_enc_chart(self, title: str):
        """A single-trace 0..1 open-fraction chart (one per gripper)."""
        return self.server.gui.add_uplot(
            data=(np.array([-4.0, 0.0]), np.zeros(2)),
            series=(
                uplot.Series(label="t"),
                uplot.Series(label="open frac", stroke=_AXIS_COLORS[2], width=1.5),
            ),
            scales={"x": uplot.Scale(time=False), "y": uplot.Scale(range=(-0.05, 1.05))},
            title=title,
            aspect=2.2,
            height=120,
        )

    def _make_xyz_chart(self, title: str, labels, y_half_range: float | None = None):
        """A 3-line (x/y/z) scrolling uPlot chart, matching bota_ft_viz style.

        ``y_half_range`` pins the initial y-axis to the symmetric ``[-h, +h]``
        (else uPlot auto-scales). The F/T charts use this with a PEAK-HOLD policy
        applied in :meth:`_refresh_charts` — the axis starts at the fixed range and
        only ever WIDENS to track a new magnitude peak, never shrinking back, so a
        brief spike stays visible and the baseline doesn't jitter the axis."""
        x0 = np.array([-2.0, 0.0])
        zero = np.zeros(2)
        scales = {"x": uplot.Scale(time=False)}
        if y_half_range is not None:
            scales["y"] = uplot.Scale(range=(-y_half_range, y_half_range))
        return self.server.gui.add_uplot(
            data=(x0, zero, zero, zero),
            series=(
                uplot.Series(label="t"),
                uplot.Series(label=labels[0], stroke=_AXIS_COLORS[0], width=1.5),
                uplot.Series(label=labels[1], stroke=_AXIS_COLORS[1], width=1.5),
                uplot.Series(label=labels[2], stroke=_AXIS_COLORS[2], width=1.5),
            ),
            scales=scales,
            title=title,
            aspect=2.2,
            height=120,
        )

    # -- per-frame ----------------------------------------------------------
    def update(self) -> None:
        with self.state.lock:
            # PRIMARY hand = LEFT (drives the main IK arm / camera / jaw).
            tracker_pos = self.state.tracker_pos.copy()
            tracker_quat = self.state.tracker_quat.copy()
            tracker_conn = self.state.tracker_connected
            tracker_stale = self.state.tracker_stale
            # Secondary RIGHT + HEAD tracker state (2pt/3pt; defaults if absent).
            right_pos = self.state.right_tracker_pos.copy()
            right_quat = self.state.right_tracker_quat.copy()
            right_conn = self.state.right_tracker_connected
            right_stale = self.state.right_tracker_stale
            right_recv_t = self.state.right_tracker_recv_t
            # Every BODY REFERENCE this mode declares, copied out under the lock as one
            # dict (chest + the two 5pt feet). A role whose puck has never streamed gets
            # the never-seen defaults, so downstream reads exactly as it did before.
            body = {}
            for _role in self._body_roles:
                _slot = _body_tracker_slot(self.state, _role)
                body[_role] = {"pos": np.asarray(_slot["pos"], dtype=float).copy(),
                               "quat": np.asarray(_slot["quat"], dtype=float).copy(),
                               "connected": bool(_slot["connected"]),
                               "stale": bool(_slot["stale"]),
                               "recv_t": float(_slot["recv_t"])}
            grip_frac = self.state.gripper_frac
            grip_actual = self.state.gripper_actual_rad
            grip_conn = self.state.gripper_connected
            grip_motor_ok = self.state.gripper_motor_ok
            # RIGHT gripper, snapshotted HERE under the same lock as everything
            # else rather than in a second `with self.state.lock` further down: one
            # frame must render one consistent view of the rig.
            grip_frac_r = self.state.gripper_frac_right
            grip_actual_r = self.state.gripper_actual_rad_right
            grip_conn_r = self.state.gripper_connected_right
            grip_motor_ok_r = self.state.gripper_motor_ok_right
            grip_recv_t_r, grip_hz_r = self.state.gripper_recv_t_right, self.state.gripper_hz_right
            cam_conn = self.state.cam_connected
            cam_seq = self.state.cam_seq
            cam_jpeg = self.state.cam_jpeg
            cam_w, cam_h = self.state.cam_w, self.state.cam_h
            # Liveness: receive-time + smoothed rate per stream, for real online/
            # offline (vs the sticky *_connected). kbd rides on the gripper frame.
            tracker_recv_t, tracker_hz = self.state.tracker_recv_t, self.state.tracker_hz
            grip_recv_t, grip_hz = self.state.gripper_recv_t, self.state.gripper_hz
            cam_recv_t, cam_hz = self.state.cam_recv_t, self.state.cam_hz
            cam_conn_r = self.state.cam_connected_right
            cam_seq_r = self.state.cam_seq_right
            cam_jpeg_r = self.state.cam_jpeg_right
            cam_w_r, cam_h_r = self.state.cam_w_right, self.state.cam_h_right
            cam_recv_t_r, cam_hz_r = self.state.cam_recv_t_right, self.state.cam_hz_right
            kbd_online = self.state.kbd_online
            do_recal = self.state.recal_request
            self.state.recal_request = False
            do_reset = self.state.reset_request
            self.state.reset_request = False
            do_axes = self.state.axes_reset_request
            self.state.axes_reset_request = False

        # Is a Bota payload recompensation in progress? While it runs, the bota
        # SensorReader thread is driving the estimator's EKF, which needs a high,
        # STEADY sample rate to converge. The mink IK solve is ~8 ms and at 60 Hz
        # holds the GIL ~half of every second — that starves/jitters the estimator
        # and is why an in-viz recompensation fails to converge while the
        # standalone estimate_payload.py (dedicated ~1 kHz loop, nothing else)
        # succeeds. So we SKIP the IK step during recal to hand the interpreter to
        # the estimator (the arm just holds its last pose for ~90 s).
        # EITHER hand's estimate gates this: both readers live in this interpreter,
        # so a right-hand recompensation needs the GIL just as much as a left-hand
        # one does.
        recomp_active = self._recomp_busy_panel() is not None

        # Assemble the live per-role tracker poses (LEFT-canonical). Only roles
        # active for this mode AND currently live (connected + not stale) are
        # included. 'left' is the primary hand held in the unprefixed tracker_*.
        def _live_poses() -> dict:
            live = {}
            if "left" in self._active_roles and tracker_conn and not tracker_stale:
                live["left"] = (tracker_pos, tracker_quat)
            if "right" in self._active_roles and right_conn and not right_stale:
                live["right"] = (right_pos, right_quat)
            for _role, _b in body.items():
                if _b["connected"] and not _b["stale"]:
                    live[_role] = (_b["pos"], _b["quat"])
            return live

        def _all_active_live(poses: dict) -> bool:
            return all(r in poses for r in self._active_roles)

        # Recalibration capture (render thread owns the IK). Refuse to anchor
        # unless EVERY active tracker is live — capturing the (0,0,0)/identity
        # defaults would silently report "calibrated" then fling a limb when a
        # real frame lands. ThreePointIK.recalibrate needs all active roles at once.
        if do_recal and self.ik is not None:
            poses = _live_poses()
            if _all_active_live(poses):
                self.ik.recalibrate(poses)
                self._did_auto_calibrate = True
                self.gui_calib.value = f"calibrated ({self.tracking_mode})"
            else:
                missing = [r for r in self._active_roles if r not in poses]
                self.gui_calib.value = f"recal skipped: need live {'+'.join(missing)}"
        # Reset the tracker AXES onto the wrist: capture the current puck orientation as
        # the mount rotation. Same live-pose gate as recalibrate — capturing a stale
        # (0,0,0)/identity default would silently report success and then mis-orient every
        # displayed triad. Persisted with the take via the recorder's calibration hop.
        if do_axes and self.ik is not None:
            poses = _live_poses()
            if _all_active_live(poses):
                mounts = self.ik.reset_axes_to_wrist(poses)
                self._mount_quats = mounts
                self._persist_mount_quats(mounts)
                self.gui_calib.value = (f"axes reset to wrist "
                                        f"({'+'.join(sorted(mounts))})")
                print(f"[viz] tracker axes reset to wrist: "
                      f"{ {r: [round(v, 4) for v in q] for r, q in mounts.items()} }",
                      flush=True)
            else:
                missing = [r for r in self._active_roles if r not in poses]
                self.gui_calib.value = f"axes reset skipped: need live {'+'.join(missing)}"

        # Reset to home (also render-thread-owned, via the same handshake).
        if do_reset and self.ik is not None:
            self.ik.reset_pose()
            self._last_q = None
            self._did_auto_calibrate = False  # re-arm auto-cal after a reset

        # Auto-calibrate once on the first fully-live frame, so the hand tracks
        # right away (the operator holds a comfortable rest pose at start).
        if (self._auto_calibrate and not self._did_auto_calibrate and self.ik is not None):
            poses = _live_poses()
            if _all_active_live(poses):
                self.ik.recalibrate(poses)
                self._did_auto_calibrate = True
                self.gui_calib.value = f"auto-calibrated ({self.tracking_mode}, first frame)"

        # IK step. Drive the main (LEFT) arm always; 2pt adds the right arm, 3pt
        # also the torso — each from its live tracker (a stale/missing secondary
        # simply holds its last target, per ThreePointIK.step). Skipped while a
        # recompensation runs so the estimator EKF gets the interpreter.
        rh_pos_str = ""
        # ---- Gripper-only display (no humanoid, no mink IK) --------------------
        # Self-contained: this block moves ONLY scene nodes (the gripper follow
        # frames + finger meshes). It never touches the recorder, the gripper
        # bridge, or any IK object (there is none). The three gripper meshes hang
        # under the SAME follow frame the camera rides, so camera-to-gripper
        # geometry is identical to humanoid mode. Placement uses the SAME
        # home-anchored delta the IK uses (gripper_only_view.home_anchor).
        if not self._display_humanoid:
            rh_pos_str = self._update_gripper_only(
                do_recal, do_axes, do_reset, _live_poses)
        elif (self.ik is not None and self.mapper is not None and self.gui_track_enable.value
                and not tracker_stale and not recomp_active):
            self._last_q = self.ik.step(_live_poses())
            # Re-seat the camera-follow frame on the freshly-solved wrist pose.
            self._place_follow_frame_at_wrist()
            rh = self.ik._solver.configuration.get_transform_frame_to_world(
                frame_name=self._primary_hand_site, frame_type="site"
            )
            rh_pos_str = _fmt(rh.translation())
        elif self.ik is not None and self.mapper is not None:
            # Tracking gated off (paused / stale tracker / recompensation): HOLD the
            # camera on the wrist at the last solved pose instead of letting it fall
            # back to the world origin (the ground). The frustum stays mounted on
            # the gripper; it just doesn't move until tracking resumes.
            self._place_follow_frame_at_wrist()
        elif self.ik is None and self.mapper is None:
            # mock-without-IK (humanoid layout, --no-ik): move the tracker frame from the
            # raw pose. Gripper-only mode is handled above, so this is reached only in the
            # humanoid-with-no-IK path. Seated through the puck->wrist transform, NOT with
            # the raw puck pose written straight on: /tracker is a WRIST frame everywhere
            # else (its children are the TCP / puck / wrist_yaw triads at wrist-relative
            # offsets, and the puck child carries M^-1), so writing a puck pose into it
            # made all three triads mean the wrong thing in this debug path.
            fp, fq = raw_follow_pose_from_puck(tracker_pos, tracker_quat)
            self._tracker_frame.position = tuple(fp)
            self._tracker_frame.wxyz = tuple(fq)

        # Render the URDF (incl. jaws) every frame from the latest solved pose +
        # the LIVE gripper fraction — decoupled from the arm-tracking gate so the
        # jaws keep mirroring the real gripper even when tracking is paused/stale.
        # The single physical gripper mounts on the PRIMARY (LEFT) hand, so its
        # fraction drives left_jaw_m; the (unused) right jaw stays closed.
        if self.ik is not None and self.mapper is not None:
            q_render = self._last_q if self._last_q is not None else self.ik.home_q()
            jaw = grip_frac * _JAW_OPEN_M
            cfg = self.mapper.to_yam(q_render, left_jaw_m=jaw, right_jaw_m=0.0)
            self._urdf.update_cfg(np.asarray(cfg))

        # Secondary tracker triads (2pt right, 3pt head): draw the RAW world pose
        # so the operator can confirm the extra trackers are live + oriented (they
        # are not IK follow-frames — only the primary /tracker drives an arm here).
        # Secondary triads show the RAW world pose, with the captured mount rotation
        # removed so the axes read wrist-aligned in the reference posture (identity when
        # nothing has been captured, so the default scene is unchanged).
        # In gripper-only mode EVERY secondary frame is a home-anchored FOLLOW frame
        # driven by _update_gripper_only (the right hand carries the gripper cluster, the
        # chest a bare puck triad), NOT a raw-pose triad — so skip them here. Writing the
        # raw pose would fight the anchored placement every frame and fling the chest
        # axis off to libsurvive's world origin.
        for role, frame in self._sec_frames.items():
            if not self._display_humanoid and role in self._grip_follow:
                continue  # driven by the gripper-only block above
            if role == "right":
                frame.position = tuple(right_pos)
                frame.wxyz = tuple(self._display_quat(role, right_quat))
                frame.visible = right_conn
            elif role in body:
                frame.position = tuple(body[role]["pos"])
                frame.wxyz = tuple(self._display_quat(role, body[role]["quat"]))
                frame.visible = body[role]["connected"]

        # Camera frame: repaint the frustum image when a new frame arrived. Skip
        # the JPEG decode during recompensation (PIL decode is ~1-2 ms of
        # GIL-holding work at 30 Hz; the estimator EKF needs the interpreter).
        # Show/hide the frustums, on CHANGE only. These were unconditional writes at
        # --render-hz (60/s). The saving is SMALL and worth naming honestly: viser's
        # props_setattr already short-circuits an unchanged value before it reaches the
        # wire (_assignable_props_api.py, "Skip update if value hasn't changed"), so this
        # was never sending 180 redundant messages a second. What it does avoid is the
        # per-assignment _cast_value_recursive + getattr + compare that runs BEFORE that
        # check — a couple of tenths of a percent, not the reason to make this edit.
        #
        # The reason is the next paragraph.
        #
        # The RIGHT frustum is included here, which it was not before: its `.image` was
        # gated on the checkbox but its `.visible` was never touched at all, so
        # unchecking "Show camera view" hid the left camera and left the right one
        # floating in the scene frozen on its last decoded frame. On a 3pt rig that
        # reads as "the right camera is stuck", which is the opposite of the truth.
        _show_cam = bool(self.gui_show_cam.value)
        if _show_cam != self._last_show_cam:
            self._last_show_cam = _show_cam
            self._frustum.visible = _show_cam
            self._cam_mount.visible = _show_cam
            if self._frustum_right is not None:
                self._frustum_right.visible = _show_cam
        # Preview REPAINT RATE, throttled independently of the bridges' publish rate
        # (--camera-preview-hz). The expensive part is not receiving a frame, it is
        # painting one: PIL decode + viser's unconditional JPEG re-encode, ~1.8 ms per
        # camera. Repainting at the full 30 Hz publish rate spends that 60 times a
        # second on two cameras for a picture whose only job is "which D405 is on which
        # hand, and is the lens clear" — a question 10 Hz answers identically.
        #
        # Per camera, not shared: the two D405s are separate bridge processes on
        # separate timers, so one shared deadline would let the faster stream starve
        # the other's repaints (whoever's frame arrived first inside a window would win
        # it and push the deadline out for both).
        #
        # This CANNOT touch recorded data. `cam_jpeg` has exactly two consumers, both of
        # them these two paint guards; the recorder is a separate process with its OWN
        # subscription to the same bridges and writes their JPEG bytes verbatim. The
        # bridges' publish rate — which IS the training dataset's master timeline and
        # must stay uniform — is CAMERA_PUBLISH_HZ and is untouched by this.
        _t_paint = time.perf_counter()
        if (_show_cam and cam_jpeg is not None and cam_seq != self._last_cam_seq
                and _t_paint - self._last_cam_paint_t >= self._cam_paint_period
                and not recomp_active):
            self._last_cam_seq = cam_seq
            self._last_cam_paint_t = _t_paint
            img = _decode_jpeg(cam_jpeg)
            if img is not None:
                # Match the frustum aspect to the actual frame so the RGB isn't
                # stretched on non-4:3 D405 modes (aspect IS a settable prop).
                if cam_w > 0 and cam_h > 0:
                    self._frustum.aspect = cam_w / cam_h
                self._frustum.image = img
        # RIGHT-hand view, on its own frustum under /tracker_right (created only when a
        # right camera was configured). Same show/hide toggle + recomp gate as the left.
        if (self._frustum_right is not None and _show_cam and cam_jpeg_r is not None
                and cam_seq_r != self._last_cam_seq_right
                and _t_paint - self._last_cam_paint_t_right >= self._cam_paint_period
                and not recomp_active):
            self._last_cam_seq_right = cam_seq_r
            self._last_cam_paint_t_right = _t_paint
            img_r = _decode_jpeg(cam_jpeg_r)
            if img_r is not None:
                if cam_w_r > 0 and cam_h_r > 0:
                    self._frustum_right.aspect = cam_w_r / cam_h_r
                self._frustum_right.image = img_r

        # GUI readouts. Module status is REAL-TIME online/offline now: a stream is
        # "online" only while frames keep arriving (last frame within its timeout),
        # so a bridge that dies mid-take flips to ○ offline instead of staying a
        # sticky "connected". Generous per-stream timeouts ride out a hiccup.
        now = time.perf_counter()
        self.gui_tracker.value = _status_str(
            "tracker", tracker_conn, tracker_recv_t, now, tracker_hz, timeout=1.0, stale=tracker_stale)
        # Secondary trackers (2pt/3pt): online/stale/offline, no rate (feedback only).
        if "right" in self.gui_sec_tracker:
            self.gui_sec_tracker["right"].value = _status_str(
                "right", right_conn, right_recv_t, now, 0.0, timeout=1.0, stale=right_stale)
        for _role, _b in body.items():
            _row = self.gui_sec_tracker.get(_role)
            if _row is not None:
                _row.value = _status_str(_role, _b["connected"], _b["recv_t"], now, 0.0,
                                         timeout=1.0, stale=_b["stale"])
        # A connected gripper whose motor thread has died (e.g. a hard-stop stall
        # overheated it) keeps streaming frames, so the plain online check would
        # show it green. Override with a loud FAULT so a dead motor is never
        # mistaken for a working one — the exact failure that hid the over-temp.
        grip_live = grip_conn and grip_recv_t > 0 and (now - grip_recv_t) < 2.0
        if grip_live and not grip_motor_ok:
            self.gui_gripper.value = "✖ MOTOR FAULT — re-measure endpoints + relaunch"
        else:
            self.gui_gripper.value = _status_str(
                "gripper", grip_conn, grip_recv_t, now, grip_hz, timeout=2.0)
        # The keyboard has no stream of its own — the gripper bridge reports its
        # heartbeat freshness as kbd_online, which is only meaningful while the
        # gripper frame itself is arriving (grip_live computed just above).
        if not grip_live:
            self.gui_kbd.value = "○ unknown (gripper offline)"
        else:
            self.gui_kbd.value = "● online" if kbd_online else "○ offline"
        self.gui_cam.value = _status_str(
            "camera", cam_conn, cam_recv_t, now, cam_hz, timeout=2.0, suffix=f"  {cam_w}x{cam_h}")
        if self.gui_cam_right is not None:
            self.gui_cam_right.value = _status_str(
                "camera_right", cam_conn_r, cam_recv_t_r, now, cam_hz_r, timeout=2.0,
                suffix=f"  {cam_w_r}x{cam_h_r}")
        # RIGHT gripper: the same MOTOR-FAULT override as the left. Without it a
        # right DM4310 that overheated on a hard-stop stall keeps streaming frames
        # and the row stays green — the exact failure the left-hand override exists
        # to catch, and the right hand records a jaw width that no longer moves.
        if self.gui_gripper_right is not None:
            grip_live_r = grip_conn_r and grip_recv_t_r > 0 and (now - grip_recv_t_r) < 2.0
            if grip_live_r and not grip_motor_ok_r:
                self.gui_gripper_right.value = "✖ MOTOR FAULT — re-measure endpoints + relaunch"
            else:
                self.gui_gripper_right.value = _status_str(
                    "gripper_right", grip_conn_r, grip_recv_t_r, now, grip_hz_r, timeout=2.0,
                    suffix=f"  {grip_frac_r * 100:.0f}% open")
        self.gui_track_pos.value = _fmt(tracker_pos)
        self.gui_rh_pos.value = rh_pos_str
        # Per-hand gripper readouts (the dicts _build_gui filled; a hand with no
        # gripper folder simply has no entry).
        for _h, _frac, _rad in (("left", grip_frac, grip_actual),
                                ("right", grip_frac_r, grip_actual_r)):
            if _h not in self.gui_grip_frac:
                continue
            self.gui_grip_frac[_h].value = f"{_frac * 100:.0f}%  ({'open' if _frac > 0.5 else 'closed'})"
            self.gui_grip_rad[_h].value = f"{_rad:+.3f}"
        # Per-sensor F/T header + engine reconcile + recompensation overlay. Each
        # panel reads only its OWN SensorState, so the two hands' readouts can never
        # be swapped, and each drives its own modal (at most one is ever up).
        for panel in self._ft_panels:
            panel.refresh_status()
            if panel.recomp_enabled:
                panel.update_recomp_overlay()

        # Recorder status readout (REC ● / idle + frame counts), from the
        # recorder's broadcast. Display-only; the recorder owns the actual state.
        # Manual per-episode model: one Start..Stop span == one episode (= one
        # take_NNN dir; the converter emits the whole take as a single episode
        # when there are no markers). On the recording edge we also fire a 2 s
        # center overlay (see _update_recording_overlay).
        if self._recorder is not None:
            if not self._recorder.connected:
                self.gui_rec_status.value = "○ offline (start recorder.py)"
            else:
                st = self._recorder.status()
                if st.get("recording"):
                    fr = st.get("frames", {})
                    self.gui_rec_status.value = (
                        f"REC ● episode {st.get('take_idx', 0)}  {st.get('elapsed', 0):.0f}s  "
                        f"trk{fr.get('tracker', 0)}/grp{fr.get('gripper', 0)}/"
                        f"ft{fr.get('ft', 0)}/cam{fr.get('camera', 0)}"
                    )
                else:
                    saved = st.get("last_saved", "")
                    self.gui_rec_status.value = f"idle (next episode {st.get('take_idx', 0)})" + (
                        f"  last: {os.path.basename(saved)}" if saved else "")
                self._update_recording_overlay(st)
                # 'c' key -> return to initial pose (R4). The macropad 'c' sends
                # {"cmd":"home"} to the recorder, which surfaces a monotonically
                # increasing home_seq in its status. Observe an increment here and
                # raise the SAME reset_request the "Reset robot to home" button uses,
                # so the IK reset stays render-thread-owned (never from a callback).
                home_seq = int(st.get("home_seq", 0))
                if home_seq != self._last_home_seq:
                    if self._last_home_seq >= 0:  # ignore the first snapshot (baseline)
                        with self.state.lock:
                            self.state.reset_request = True
                    self._last_home_seq = home_seq

    def _recomp_busy_panel(self):
        """The panel whose reader is mid-payload-estimate, or None.

        Cross-hand check used by BOTH the arm gate (refuse a second click) and the
        render loop (throttle everything while one runs)."""
        for panel in self._ft_panels:
            if panel.recomp_active():
                return panel
        return None

    def _update_recomp_overlay(self) -> None:
        """Pump every hand's recompensation overlay (compact readout + modal).

        Called from update() and, while a run is in progress, directly from the
        throttled render loop — so the countdown keeps ticking and a finished
        cycle still auto-closes even though the 3D scene is frozen."""
        for panel in self._ft_panels:
            if panel.recomp_enabled:
                panel.update_recomp_overlay()

    def _update_recording_overlay(self, st: dict) -> None:
        """Fire a 2 s center-screen banner on each recording start/stop edge.

        Mirrors the recompensation overlay's single-owner discipline: this runs
        ONLY on the render thread (called from update()), so it is the sole toucher
        of the _rec_modal handle — the s/e keys and GUI buttons only change the
        recorder's state, never this modal. On the idle->recording edge we show
        ``REC ● 开始录制 episode N``; on recording->idle (a saved episode) we show
        ``✓ 已保存 episode N``. Each banner auto-closes after _rec_overlay_secs.

        ``st`` is the recorder's latest status dict. ``take_idx`` is the index of
        the NEXT episode while idle and of the CURRENT one while recording, so on
        START we label with ``take_idx`` and on STOP with ``take_idx - 1`` (the
        episode that was just saved)."""
        recording = bool(st.get("recording"))
        idx = int(st.get("take_idx", 0))
        now = time.perf_counter()

        if recording and not self._rec_prev_recording:
            # idle -> recording: this episode's index is the current take_idx.
            # Auto-tare EVERY F/T sensor at each episode start so both hands' wrench
            # begins zeroed at their own current (free-space) pose. We only raise a
            # flag under each state's lock; that hand's SensorReader thread (sole
            # toucher of its Bota HWI) does the actual tare on its next loop
            # iteration — a compensator re-tare (changes the recorded compensated
            # wrench + refreshes tare provenance) on a compensated stream, else a
            # display-only zero. The tare settles ~0.5 s AFTER start; the recorder
            # latches tare provenance "last non-null wins" PER ROLE, so it still
            # lands in this take's meta for each hand independently.
            self._tare_all_ft()
            # The tare sentence is DROPPED when no sensor is attached: claiming a
            # tare that never happened is exactly the sentence an operator would
            # trust before a take with a biased (or absent) wrench.
            _tare_note = (f"已自动 tare {self._tare_hands_zh()}力觉(请确保开始时未接触物体)。"
                          if self._ft_panels else "")
            self._open_recording_banner(
                f"# ⏺ 开始录制 episode {idx}\n\n正在记录{self._recording_streams_zh()}。"
                f"{_tare_note}按 **e** 结束并保存。",
                now)
        elif not recording and self._rec_prev_recording:
            # recording -> idle: the saved episode was take_idx-1 (the counter has
            # already advanced). A discarded/incomplete take leaves last_saved empty.
            # Auto-tare EVERY F/T on the STOP edge too (R5): re-zeroes the live view
            # / next take's baseline between demonstrations, visible on the curves.
            # (Semantics: this tare settles ~0.5 s AFTER cmd=stop, so it lands in the
            # NEXT take's provenance, NOT the just-saved one — it is a between-take
            # convenience zero, matching the recorder ordering; see recorder.py.)
            self._tare_all_ft()
            saved = st.get("last_saved", "")
            _tare_note = (f"已自动 tare {self._tare_hands_zh()}力觉（两段之间归零）。"
                          if self._ft_panels else "")
            if saved:
                self._open_recording_banner(
                    f"# ✓ 已保存 episode {max(idx - 1, 0)}\n\n`{os.path.basename(saved)}`\n\n{_tare_note}", now)
            else:
                self._open_recording_banner(
                    "# ⚠ 未保存\n\n该段缺少相机或 tracker 数据，未写入磁盘。", now)
        self._rec_prev_recording = recording

        # Auto-close once the linger window expires (render-thread only).
        if self._rec_modal is not None and now >= self._rec_overlay_until:
            self._close_recording_banner()

    def _tare_all_ft(self) -> None:
        """Zero EVERY attached F/T sensor (both hands). Called on the recording
        START and STOP edges.

        Each panel writes only into its own SensorState under that state's lock, so
        this is one independent request per sensor — never a shared one. A hand
        whose sensor is absent simply has no panel."""
        for panel in self._ft_panels:
            panel.request_tare()

    def _tare_hands_zh(self) -> str:
        """'左右手' / '左手' / '右手' — which hands the auto-tare covered.

        Named in the banner because on a two-sensor rig "已自动 tare 力觉" alone
        would read as a promise about both hands even when only one sensor came up,
        and an untared hand's wrench carries a silent DC bias into the take."""
        hands = [p.hand for p in self._ft_panels]
        if len(hands) > 1:
            return "左右手"
        if hands == ["left"]:
            return "左手"
        if hands == ["right"]:
            return "右手"
        return ""

    def _recording_streams_zh(self) -> str:
        """Which streams the banner claims are being recorded, per MODE.

        The old copy hardcoded '右手位姿 / 夹爪 / 力觉 / 相机' — wrong on two counts
        on this rig: the primary hand is the LEFT one (LEFT-canonical), and in
        2pt/3pt/5pt the take also carries the second hand, the chest and (5pt) the two
        feet. 力觉 is listed only when a sensor is actually attached."""
        body = "双手位姿" if "right" in self._active_roles else "左手位姿"
        if "head" in self._active_roles:
            body += " / 胸部"
        if any(r in self._active_roles for r in ("left_foot", "right_foot")):
            body += " / 双脚"
        body += " / 夹爪"
        if self._ft_panels:
            body += " / 力觉"
        return body + " / 相机"

    def _open_recording_banner(self, markdown: str, now: float) -> None:
        """(Re)open the center banner with ``markdown`` for _rec_overlay_secs.
        Render-thread only. Closing any prior banner first keeps a single handle."""
        self._close_recording_banner()
        self._rec_overlay_until = now + self._rec_overlay_secs
        self._rec_modal = self.server.gui.add_modal("Recording")
        with self._rec_modal:
            self._rec_md = self.server.gui.add_markdown(markdown)

    def _close_recording_banner(self) -> None:
        """Close + null the recording banner. Render-thread only (single owner)."""
        if self._rec_modal is not None:
            try:
                self._rec_modal.close()
            except Exception:  # noqa: BLE001
                pass
            self._rec_modal = None
            self._rec_md = None

    def update_chart(self) -> None:
        # Gripper open-fraction chart, one trace per hand (a single chart with two
        # series would need both hands to share an x grid, which two independently
        # timed encoder streams do not).
        with self.state.lock:
            n = self.state.gripper_seq
            exported = self.state.history.export()
            n_r = self.state.gripper_seq_right
            exported_r = self.state.history_right.export()
        if n != self._last_chart_n and exported is not None:
            self._last_chart_n = n
            x, y = exported
            if x.size >= 2:
                self._enc_chart.data = (x, y)
        if self._enc_chart_right is not None and n_r != self._last_chart_n_right and exported_r is not None:
            self._last_chart_n_right = n_r
            x_r, y_r = exported_r
            if x_r.size >= 2:
                self._enc_chart_right.data = (x_r, y_r)

        # Bota Force / Torque charts, per sensor (from each in-process
        # SensorReader's rolling history).
        for panel in self._ft_panels:
            panel.refresh_charts()

    @staticmethod
    def _peak_hold_y(chart, held_peak: float, chans) -> float:
        """Widen ``chart``'s symmetric y-range to fit the current magnitude peak,
        holding (never shrinking) the previous peak. Returns the new held peak.

        ``chans`` is the (3, N) x/y/z block for the visible window. Re-pushes the
        scale only when the peak grows past the held value, so a quiet signal leaves
        the axis untouched."""
        cur = float(np.nanmax(np.abs(chans))) if np.size(chans) else 0.0
        if cur <= held_peak:
            return held_peak
        chart.scales = {"x": uplot.Scale(time=False),
                        "y": uplot.Scale(range=(-cur, cur))}
        return cur


def _fmt(v: np.ndarray) -> str:
    return f"[{v[0]:+.3f}, {v[1]:+.3f}, {v[2]:+.3f}]"


def _status_str(name: str, connected: bool, recv_t: float, now: float, hz: float,
                timeout: float, stale: bool = False, suffix: str = "") -> str:
    """One module's real-time status line: ``● online 90Hz`` while frames keep
    arriving, ``⚠ stale`` for a tracker whose pose froze, ``○ offline (Ns)`` once
    the last frame is older than ``timeout``, or ``○ waiting…`` before the first.

    ``connected`` (the sticky first-frame flag) only distinguishes "never seen"
    from "seen then went silent" — the live/offline call is purely the arrival
    age, so a dead bridge is always caught regardless of the old flag."""
    if not connected or recv_t <= 0.0:
        return "○ waiting…"
    age = now - recv_t
    if age >= timeout:
        return f"○ offline ({age:.0f}s){suffix}"
    if stale:
        return f"⚠ stale{suffix}"
    return f"● online  {hz:.0f}Hz{suffix}"


def _recomp_markdown(phase: str, elapsed: float, message: str, hand: str = "") -> str:
    """Big-text markdown for the recompensation overlay, mirroring the dashboard
    overlay's phase copy (countdown digit / rotate prompt / done / failed).

    ``hand`` names WHICH gripper to lift/rotate. It must be in the copy on a
    two-sensor rig: the prompt is the operator's only instruction, and rotating
    the wrong hand's tool produces a converged-but-wrong payload for the sensor
    that never moved. Empty (single-sensor) keeps the historical wording."""
    which = f" {hand}" if hand else ""
    if phase == "countdown":
        return (f"# ⏳ {math.ceil(elapsed) if elapsed > 0 else 0}\n\n"
                f"## Get ready — lift the{which} tool, keep it free of contact\n\n"
                f"{message}")
    if phase == "estimating":
        return ("# ↻ Estimating payload…\n\n"
                f"## ROTATE the{which} tool slowly through many orientations\n\n"
                f"{message}")
    if phase == "done":
        return "# ✓ Recompensation complete\n\nThe payload was re-estimated and the compensator updated."
    if phase == "failed":
        return ("# ✕ Recompensation failed\n\n"
                f"{message}\n\nRotate through more poses, keep the tool free of contact, and retry. "
                "Click **Close** to dismiss.")
    return f"{message}"


def resolve_right_ft_configs(bft, *, compensate: bool = True) -> tuple:
    """Resolve the RIGHT hand's ``(compensator, estimator, rnea)`` config paths.

    A named seam rather than inline code in ``main()`` because getting it wrong is
    silent and destructive: ``SensorReader._recalibrate`` copies the fresh payload
    onto its OWN ``compensator_config``, so handing the right reader the SHARED
    ``payload_compensator.json`` would make a right-hand recompensation overwrite
    the LEFT hand's payload — the left charts would then quietly compensate with
    the right tool's mass forever after, with nothing anywhere to catch it. Same
    for the RNEA file, which ``_run_hardware_compensated`` syncs from whichever
    compensator config this reader holds.

    * compensator — PER SIDE (``payload_compensator_right.json``). None when the
      file is absent or ``compensate`` is False -> the reader runs the RAW loop.
    * estimator — SHARED (``payload_estimator.json``): it is priors + convergence
      thresholds, not a per-sensor result, and ``_recalibrate`` redirects its
      output to a temp file. None unless the compensated loop will run, because
      the raw loop never consumes ``recal_request`` and the button would wedge.
    * rnea — PER SIDE, and only meaningful alongside the compensator.

    The LEFT hand deliberately keeps the SHARED ``payload_compensator.json`` /
    ``rnea_compensator.json`` it has always used: repointing it at a ``_left`` file
    would silently change the compensation of every existing single-hand workflow.
    """
    comp = os.path.join(os.path.dirname(bft._DEFAULT_COMP_CONFIG),
                        "payload_compensator_right.json")
    comp = comp if (compensate and os.path.exists(comp)) else None
    rnea = os.path.join(os.path.dirname(bft._DEFAULT_RNEA_CONFIG),
                        "rnea_compensator_right.json")
    rnea = rnea if (comp and os.path.exists(rnea)) else None
    est = bft._DEFAULT_EST_CONFIG if (comp and os.path.exists(bft._DEFAULT_EST_CONFIG)) else None
    return comp, est, rnea


def _decode_jpeg(jpeg: bytes):
    try:
        from PIL import Image

        return np.asarray(Image.open(io.BytesIO(jpeg)).convert("RGB"))
    except Exception:  # noqa: BLE001
        return None


# ----------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8080)
    # 30, not the historical 60. Nothing this loop draws benefits from 60: the tracker
    # bridge broadcasts at 90 Hz but a human watching triads cannot see past 30, the
    # cameras publish at 30, and the charts are throttled to 15. What 60 DID buy was a
    # 16.7 ms tick budget for work that measures ~10 ms (and ~9.8 ms of mink IK on top in
    # humanoid mode) — and the loop has no overrun backoff (`sleep = period - elapsed;
    # if sleep > 0`), so an over-budget tick just runs back-to-back. 30 Hz doubles the
    # budget to 33.3 ms.
    #
    # Recording is a separate process, so this is not on the data path. The ONE coupling
    # is that the render thread observes the recorder's idle->recording edge and fires
    # the auto-tare there — and that observation is already quantised by the recorder's
    # own 5 Hz status broadcast (recorder.py, `sleep(0.2)`), so halving the render rate
    # adds at most 16.7 ms (mean ~8) on top of a 0-200 ms window and a ~0.5 s async tare
    # settle. It cannot change which take the tare lands in, the provenance latching, or
    # any recorded array's grid or shape.
    p.add_argument("--render-hz", type=float, default=30.0,
                   help="scene/GUI update rate (default 30). Display only — the recorder "
                        "is a separate process.")
    # Frustum REPAINT rate, deliberately decoupled from the bridges' CAMERA_PUBLISH_HZ.
    # Painting a frame costs a PIL decode + viser's unconditional JPEG re-encode (~1.8 ms
    # per camera); receiving one costs ~0.12 ms. So the publish rate — which is the
    # training timeline and must stay uniform — is the wrong knob, and this is the right
    # one. 0 = repaint every published frame (historical behaviour).
    p.add_argument("--camera-preview-hz", type=float,
                   default=float(os.environ.get("CAMERA_PREVIEW_HZ", "10")),
                   help="frustum RGB repaint rate (default 10; 0 = every published frame). "
                        "Independent of --publish-hz on the bridges, which is the recorded "
                        "timeline and is NOT affected. Same as CAMERA_PREVIEW_HZ.")
    p.add_argument("--yam-urdf", default=_DEFAULT_YAM)
    p.add_argument("--tracker-url", default="ws://127.0.0.1:8772")
    p.add_argument("--camera-url", default="ws://127.0.0.1:8873",
                   help="LEFT-hand D405 bridge (8873: 8773/8777 belong to the deploy stack's robot cameras)")
    p.add_argument("--gripper-url", default="ws://127.0.0.1:8774", help="LEFT-hand gripper bridge")
    # ---- RIGHT-hand per-hand devices (2pt/3pt). All default EMPTY: an unset URL/serial
    # means "this hand has no such device", so a 1pt run and a 3pt run before the right
    # hardware is wired up both behave exactly as before — nothing extra is opened and
    # nothing is drawn for it. See recording_format's PER-HAND banner.
    p.add_argument("--camera-right-url", default="",
                   help="RIGHT-hand D405 bridge (e.g. ws://127.0.0.1:8877); empty = no right camera")
    p.add_argument("--gripper-right-url", default="",
                   help="RIGHT-hand gripper bridge (e.g. ws://127.0.0.1:8778); empty = no right gripper")
    p.add_argument("--ik-config", default="proposal", help="HuMI IKConfig name")
    p.add_argument("--reach-scale", type=float, default=1.0)
    p.add_argument("--no-ik", action="store_true", help="skip mink IK (render home pose, move tracker frame only)")
    p.add_argument("--robot-height", type=float, default=0.0,
                   help="extra z offset for the whole robot in the scene (m)")
    # Camera mount offset relative to the right WRIST-YAW frame (metres / rad).
    # Default: 10 cm forward along the gripper approach (wrist +x) — a touch
    # beyond the yam gripper — and 4 cm up; yaw -90deg about y so the frustum's
    # view axis (-z) points along wrist +x (forward, same as the gripper).
    p.add_argument("--cam-offset", type=float, nargs=3, default=(0.10, 0.0, 0.04),
                   metavar=("X", "Y", "Z"), help="camera position offset from the wrist (m)")
    p.add_argument("--cam-rpy", type=float, nargs=3, default=(4.7124, 1.5708, 0.0),
                   metavar=("ROLL_X", "YAW_Y", "_"), help="camera mount rotation: composed "
                   "Rx(roll)*Ry(yaw). Default (3pi/2, pi/2) points the frustum OUTWARD along the "
                   "gripper (wrist +x) with a 270deg roll about x (image upright for this mount). "
                   "3rd value unused.")
    p.add_argument("--cam-fov-deg", type=float, default=58.0, help="frustum vertical FOV (D405 color ~58 deg)")
    # Start with the frustum RGB off. Same switch as the GUI's "Show camera view"
    # checkbox, which stays live either way — this only picks the state you boot into,
    # for a long unattended recording session where nobody is going to look at the
    # pixels. Display only: the recorder is a separate process with its own camera
    # subscription and writes the bridge's JPEG bytes verbatim, so this cannot change
    # what lands on disk. Env default so the launcher can expose it as CAMERA_VIEW=0
    # without every wrapper having to thread a flag.
    p.add_argument("--no-camera-view", action="store_true",
                   default=os.environ.get("CAMERA_VIEW", "1") == "0",
                   help="boot with 'Show camera view' UNCHECKED (saves ~24%% of one core with two "
                        "640x480@30Hz cameras: no per-frame JPEG decode + viser re-encode). "
                        "Toggleable live in the GUI. Recording is unaffected. Same as CAMERA_VIEW=0.")
    p.add_argument("--no-auto-calibrate", action="store_true",
                   help="do NOT auto-calibrate on the first tracker frame (use the Recalibrate button)")
    p.add_argument("--no-ft", action="store_true", help="skip the in-process Bota F/T sensor + charts")
    p.add_argument("--ft-read-hz", type=float, default=250.0,
                   help="Bota F/T read rate. The sensor streams ~1kHz; we can't drain it fully AND "
                   "run the IK loop in one process, so the driver logs benign 'frames skipped' "
                   "warnings — the charts only need ~15Hz and always get the latest read.")
    # Bota payload compensation: when a compensator config exists, the F/T charts
    # show the gravity/inertia-COMPENSATED wrench (tool weight removed), and the
    # "Recompensation" button can re-estimate it live. Defaults to the shared
    # bota_demo_py config the standalone tool writes.
    p.add_argument("--no-compensate", action="store_true",
                   help="show the RAW wrench (skip Bota payload compensation + the Recompensation button)")
    p.add_argument("--ft-algo", choices=("bota", "rnea"), default=os.environ.get("FT_ALGO", "bota"),
                   help="INITIAL compensation algorithm: 'bota' (official) or 'rnea' (independent "
                        "single-body Newton-Euler dynamic). Switchable live in the GUI. Default bota.")
    p.add_argument("--recal-countdown", type=float, default=5.0,
                   help="seconds of 'get ready, lift the tool' countdown before recompensation estimation")
    p.add_argument("--recal-max-time", type=float, default=90.0,
                   help="max payload-estimation window for recompensation (s); overrides the config's max_time")
    p.add_argument("--tracking-mode", choices=("1pt", "2pt", "3pt", "5pt"), default="1pt",
                   help="LEFT-canonical: 1pt = LEFT hand only; 2pt = left+right hands; "
                        "3pt = left+right+head (both arms + torso); 5pt = 3pt + both FEET "
                        "(two more BODY-REFERENCE pucks — no camera/F-T/gripper, and no IK "
                        "task, so 5pt is gripper-only). 1pt/2pt/3pt use ThreePointIK with "
                        "the matching active roles. Match the bridge's MODE.")
    p.add_argument("--display-mode", choices=("auto", "humanoid", "gripper"), default="auto",
                   help="WHAT to draw (does NOT change what is recorded). 'humanoid' = the "
                        "full G1 driven by mink IK; 'gripper' = ONLY the yam gripper meshes + "
                        "camera per active hand (plus a chest-puck triad in 3pt and two "
                        "foot triads in 5pt), "
                        "home-anchored with NO IK; 'auto' (default) = gripper in EVERY "
                        "tracking mode. Camera-to-gripper geometry is identical in both "
                        "modes.")
    p.add_argument("--raw-frame", action="store_true",
                   help="gripper-only display ONLY: draw the cluster at the puck's RAW "
                        "libsurvive pose (no home-anchor), with the yam TCP position+axes "
                        "as the fused -y,-x,-z remapped-TCP and wrist_yaw at TCP-0.242x. A "
                        "faithful twin of the recorded tracker.npy frame; ignored in "
                        "humanoid mode. Display only — recording is unchanged.")
    p.add_argument("--mock", action="store_true", help="synthetic tracker/gripper/camera, no bridges")
    # Data recording: when --record is set, the viz (a) publishes the FULL-RATE
    # raw Bota wrench on a localhost WS so the out-of-process recorder captures it
    # losslessly, and (b) shows a Recording panel wired to the recorder's control
    # WS. The recorder itself is a SEPARATE process (recording/recorder.py) — the
    # viz only feeds F/T and drives start/stop, keeping disk I/O off this interpreter.
    p.add_argument("--record", action="store_true",
                   help="enable data-recording integration (F/T publisher + Recording panel)")
    p.add_argument("--ft-pub-port", type=int, default=8775, help="localhost port for the raw F/T publisher")
    # Bota serial PINS, mandatory once two sensors are attached: the by-id names sort by
    # serial, so an unpinned pick silently depends on which serial sorts lower (386B…
    # before 3870…) and the left hand would open the right hand's sensor.
    p.add_argument("--ft-left-serial", default="",
                   help="pin the LEFT Bota by serial substring (e.g. 3870355D3532). "
                        "Empty = first by-id match (single-sensor behavior).")
    p.add_argument("--ft-right-serial", default="",
                   help="pin the RIGHT Bota by serial substring (e.g. 386B355D3532). "
                        "Empty = no right-hand F/T is opened at all.")
    p.add_argument("--ft-pub-right-port", type=int, default=8779,
                   help="localhost port for the RIGHT-hand raw F/T publisher (needs --record "
                        "and --ft-right-serial)")
    p.add_argument("--recorder-url", default="ws://127.0.0.1:8776",
                   help="control WS of the out-of-process recorder")
    args = p.parse_args()

    state = CollectionState()

    # IK + mapper (skipped in --no-ik; still needed for the yam joint order).
    ik = None
    mapper = None
    tracking_mode = getattr(args, "tracking_mode", "1pt")
    _active_roles = _ROLES_BY_MODE[tracking_mode]
    # Resolve the display mode: 'auto' -> gripper in EVERY tracking mode. 'gripper'
    # draws ONLY the yam gripper meshes per hand + (in 3pt) a chest-puck triad — no
    # humanoid, NO mink IK; it never affects what the recorder captures — that path
    # is untouched.
    try:
        display_mode = resolve_display_mode(getattr(args, "display_mode", "auto"), tracking_mode)
    except ValueError as exc:
        # A refused combination (today: 5pt + humanoid) is an operator mistake, not a bug:
        # print the reason and exit, rather than showing a traceback from inside a resolver.
        print(f"[viz] ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from None
    # In gripper mode there is NO IK (the whole point: no arm/torso solve, no mink
    # import). --no-ik forces the same, and 5pt is refused above. Otherwise build the
    # ThreePointIK — over the mode's HAND+TORSO roles only, since it has no foot task.
    if not args.no_ik and display_mode != "gripper":
        sys.path.insert(0, os.path.dirname(__file__))
        # LEFT-canonical: ONE IK (ThreePointIK) drives every mode, restricted to the
        # mode's active roles (1pt = left only). This is the single-source fix for
        # the old "1pt drives the RIGHT arm" ~180° yaw flip — the primary hand is
        # always the left arm now.
        from g1_three_point_ik import ThreePointIK
        ik = ThreePointIK(config_name=args.ik_config, reach_scale=args.reach_scale,
                          active_roles=_active_roles)
        # yam joint order comes from the URDF the scene loads (set after scene build).
    _raw = getattr(args, "raw_frame", False) and display_mode == "gripper"
    print(f"Display mode: {display_mode} (tracking_mode={tracking_mode}, ik={ik is not None}"
          f"{', RAW puck frame + remapped-TCP' if _raw else ''})")
    if getattr(args, "raw_frame", False) and display_mode != "gripper":
        print("  NOTE: --raw-frame is ignored in humanoid display mode (gripper-only feature).")

    # In-process Bota F/T reader (bota env has bota_driver; the sensor is on the
    # same gripper rig). Skipped under --mock / --no-ft / when no sensor is found.
    ft_state = None
    ft_reader = None
    ft_publisher = None     # raw F/T WS publisher (only when --record + a sensor)
    ft_raw_sink = None      # callable pushed into SensorReader (None = no recording sink)
    # RIGHT-hand Bota (2pt/3pt only, and only when --ft-right-serial is given). Fully
    # independent of the left: own device, own compensator config, own publisher port.
    ft_state_right = None
    ft_reader_right = None
    ft_publisher_right = None
    recomp_enabled = False  # set True when an estimator config is wired to the reader
    rnea_available = False  # set True when a per-hand RNEA config enables the algo switch
    recomp_enabled_right = False
    rnea_available_right = False
    if not args.mock and not args.no_ft:
        # bota_ft_viz lives in the sibling force_torque/ module.
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "force_torque"))
        import bota_ft_viz as bft

        # PIN the LEFT sensor by serial when the launcher supplied one. With two Bota
        # sensors attached, the unpinned first-sorted-match picks whichever serial sorts
        # lower (386B… before 3870…) — i.e. the RIGHT hand's sensor would have become the
        # left hand's wrench, with nothing anywhere to catch it. No serial given (1pt, or
        # a manual run) keeps the historical single-sensor behavior.
        ft_port = bft.find_bota_port(getattr(args, "ft_left_serial", "") or None)
        if ft_port is None and getattr(args, "ft_left_serial", ""):
            print(f"NOTE: LEFT Bota serial {args.ft_left_serial} not attached — NOT falling back "
                  f"to the other sensor (a mislabeled wrench is worse than none). "
                  f"Present: {bft.list_bota_ports()}")
        if ft_port is not None:
            ft_cfg = bft.resolve_driver_config(bft._DEFAULT_CONFIG, ft_port)
            ft_state = bft.SensorState()
            # Payload compensation: if a compensator config already exists, run the
            # compensated chain (charts show tool-gravity-removed wrench) AND enable
            # the Recompensation button (re-estimates the payload live). The
            # estimator config is only meaningful when compensation is active.
            comp_cfg = None
            est_cfg = None
            if not args.no_compensate and os.path.exists(bft._DEFAULT_COMP_CONFIG):
                comp_cfg = bft._DEFAULT_COMP_CONFIG
                est_cfg = bft._DEFAULT_EST_CONFIG
                ft_state.compensation_available = True
                print(f"Bota payload compensation ON ({comp_cfg})")
            elif not args.no_compensate:
                # No compensator config yet: the reader runs the RAW loop (which
                # never consumes recal_request), so the Recompensation button is
                # hidden. Bootstrap the first payload offline with estimate_payload.py.
                print(f"NOTE: no payload config at {bft._DEFAULT_COMP_CONFIG}; F/T shown RAW "
                      "(Recompensation hidden). Run estimate_payload.py once to create it.")
            # When recording, publish the FULL-RATE wrench losslessly to the
            # out-of-process recorder via a source-side ring (the reader pushes
            # each sample; a WS thread drains+broadcasts). Microsecond push cost,
            # so the estimator EKF is unaffected. Off (None) when not recording.
            # The recorder always stores the RAW wrench; when a payload config is
            # active it ALSO stores the live compensated wrench (publisher carries
            # both + a `compensated` flag the recorder latches per take).
            if args.record:
                sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "force_torque"))
                from ft_raw_publisher import FtRawPublisher
                ft_publisher = FtRawPublisher(port=args.ft_pub_port)
                # Fixed for the session: are we running the compensated loop? (the
                # reader picks the compensated loop iff comp_cfg is set). The
                # recorder uses this to know ft_wrench_comp.npy is meaningful.
                ft_publisher.compensated = bool(comp_cfg)
                ft_raw_sink = ft_publisher.push
            # Lower read rate than the standalone bota tool: this process also runs
            # the IK render loop, so a ~1 kHz reader would starve it (and the
            # sensor logs "frames skipped"). 80 Hz is plenty for the charts.
            # RNEA algorithm option: only when the compensator chain runs AND an RNEA
            # config exists. collection_viz is single-sensor, so it uses the shared
            # rnea_compensator.json (no per-side split like the bimanual viz).
            rnea_cfg = None
            ft_algo = "bota"
            if comp_cfg is not None:
                if os.path.exists(bft._DEFAULT_RNEA_CONFIG):
                    rnea_cfg = bft._DEFAULT_RNEA_CONFIG
                    ft_algo = (getattr(args, "ft_algo", None) or "bota").lower()
                    ft_algo = "rnea" if ft_algo == "rnea" else "bota"
                    print(f"Bota F/T: RNEA algorithm option ON ({os.path.basename(rnea_cfg)}); "
                          f"initial algo={ft_algo}.")
                else:
                    # No shared RNEA config: RNEA dropdown is hidden, Bota only. Say so
                    # loudly so `FT_ALGO=rnea` doesn't silently do nothing.
                    print(f"NOTE: no RNEA config at {bft._DEFAULT_RNEA_CONFIG}; RNEA algorithm "
                          "option disabled (F/T compensation stays on Bota). Create it (e.g. copy "
                          "a per-side rnea_compensator_<side>.json) to enable the switch.")
            ft_reader = bft.SensorReader(
                ft_state, ft_cfg, mock=False, read_hz=args.ft_read_hz,
                compensator_config=comp_cfg, estimator_config=est_cfg,
                recal_countdown_s=args.recal_countdown, recal_max_time=args.recal_max_time,
                raw_sink=ft_raw_sink,
                rnea_config=rnea_cfg, algo=ft_algo,
            )
            ft_state.algo = ft_algo if rnea_cfg else "bota"
            rnea_available = bool(rnea_cfg)
            # Surface the compensator's tare offset + time to the recorder for
            # provenance (a tare taken with the tool in contact silently biases the
            # compensated trace; raw is unaffected). The reader sets these after its
            # startup tare / each recompensation / each recording-start auto-tare;
            # the publisher includes them in the batch frame so they land in
            # take_meta. Read the reader's single ``_tare`` tuple with ONE atomic
            # reference load (not the two derived properties) so the cross-thread
            # read can't tear the (wrench, wall) pair.
            if ft_publisher is not None and comp_cfg:
                def _tare_getter(r=ft_reader):
                    t = r._tare  # one atomic load: (offs6, wall) or None
                    return (list(t[0]), t[1]) if t is not None else (None, None)
                ft_publisher.tare_getter = _tare_getter

                # Which engine (bota|rnea) compensated + whether its sign is trusted,
                # so the recorder latches per-take algo provenance (a mid-session
                # dropdown switch otherwise silently blends two compensation models).
                # ONE atomic load of the reader's _algo_prov tuple (no torn pair).
                def _algo_getter(r=ft_reader):
                    a = r._algo_prov  # one atomic load: (name, sign_verified) or None
                    # Default sign to False (conservative "unknown sign is not a safe
                    # sign", matching _build_rnea) for defense-in-depth: the publisher
                    # only attaches the fields when name is not None, so this default
                    # is currently discarded, but it stays safe if that gate loosens.
                    return (a[0], a[1]) if a is not None else (None, False)
                ft_publisher.algo_getter = _algo_getter
            # Recompensation needs BOTH an estimator config AND the compensated
            # run loop (the recal is only dispatched there). The reader only runs
            # the compensated loop when compensator_config is set, so require both.
            recomp_enabled = bool(est_cfg) and bool(comp_cfg)
            print(f"Bota F/T sensor (LEFT): {ft_port}")
        else:
            print("NOTE: no Bota F/T sensor found; force/torque charts disabled.")

        # ===== RIGHT-hand Bota (2pt/3pt) =========================================
        # A SECOND, fully independent reader + publisher, with the SAME capabilities
        # as the left: own status row, own four charts, own tare / raw-compensated
        # toggle / engine dropdown / recompensation button (see _FTPanel). Nothing is
        # shared — own device, own compensator + RNEA config, own estimator run, own
        # tare offset, own publisher port — so a control on one hand can never move
        # the other hand's numbers. It gives the dataset the right hand's full-rate
        # wrench, published on its own port, recorded to ft_wrench_right.npy and
        # converted to robot1_ft_wrench.
        #
        # Compensation uses the PER-SIDE configs (payload_compensator_right.json /
        # rnea_compensator_right.json). The left hand keeps the SHARED
        # payload_compensator.json it has always used — repointing it at _left would
        # silently change the compensation on every existing single-hand workflow.
        #
        # ⚠️ The right hand's RNEA sign is SEEDED/UNVALIDATED on this rig (same caveat
        # holosoma's 2_ft_bridge.sh carries). It is published with
        # ``sign_verified=False``, so the recorder latches that into take_meta and a
        # suspect trace can be segregated offline instead of trusted silently.
        if getattr(args, "ft_right_serial", ""):
            ft_port_r = bft.find_bota_port(args.ft_right_serial)
            if ft_port_r is None:
                print(f"NOTE: RIGHT Bota serial {args.ft_right_serial} not attached — the right "
                      f"hand records NO F/T (no robot1_ft_wrench). Present: {bft.list_bota_ports()}")
            elif ft_port_r == ft_port:
                print(f"ERROR: LEFT and RIGHT Bota resolved to the SAME device ({ft_port_r}); "
                      f"skipping the right sensor rather than reading one sensor as two hands.")
            else:
                comp_cfg_r, est_cfg_r, rnea_cfg_r = resolve_right_ft_configs(
                    bft, compensate=not args.no_compensate)
                ft_state_right = bft.SensorState()
                if comp_cfg_r:
                    ft_state_right.compensation_available = True
                if args.record and getattr(args, "ft_pub_right_port", 0):
                    from ft_raw_publisher import FtRawPublisher
                    ft_publisher_right = FtRawPublisher(port=args.ft_pub_right_port)
                    ft_publisher_right.compensated = bool(comp_cfg_r)
                    ft_raw_sink_right = ft_publisher_right.push
                else:
                    ft_publisher_right, ft_raw_sink_right = None, None
                _algo_r = "bota"
                if rnea_cfg_r:
                    _algo_r = (getattr(args, "ft_algo", None) or "bota").lower()
                    _algo_r = "rnea" if _algo_r == "rnea" else "bota"
                ft_reader_right = bft.SensorReader(
                    ft_state_right, bft.resolve_driver_config(bft._DEFAULT_CONFIG, ft_port_r),
                    mock=False, read_hz=args.ft_read_hz,
                    compensator_config=comp_cfg_r, estimator_config=est_cfg_r,
                    recal_countdown_s=args.recal_countdown, recal_max_time=args.recal_max_time,
                    raw_sink=ft_raw_sink_right,
                    rnea_config=rnea_cfg_r, algo=_algo_r,
                )
                ft_state_right.algo = _algo_r if rnea_cfg_r else "bota"
                rnea_available_right = bool(rnea_cfg_r)
                recomp_enabled_right = bool(est_cfg_r) and bool(comp_cfg_r)
                if ft_publisher_right is not None and comp_cfg_r:
                    def _tare_getter_r(r=ft_reader_right):
                        t = r._tare
                        return (list(t[0]), t[1]) if t is not None else (None, None)
                    ft_publisher_right.tare_getter = _tare_getter_r

                    def _algo_getter_r(r=ft_reader_right):
                        a = r._algo_prov
                        return (a[0], a[1]) if a is not None else (None, False)
                    ft_publisher_right.algo_getter = _algo_getter_r
                print(f"Bota F/T sensor (RIGHT): {ft_port_r}  "
                      f"comp={'on' if comp_cfg_r else 'RAW'} algo={ft_state_right.algo}  "
                      f"recomp={'on' if recomp_enabled_right else 'off'}"
                      f"{f' -> ws:{args.ft_pub_right_port}' if ft_publisher_right else ''}")

    # Build the viser server + scene BEFORE starting readers (bota_ft_viz lesson:
    # a busy reader during bootstrap can starve viser's threads).
    server = viser.ViserServer(host=args.host, port=args.port, label="G1 Data-Collection Viz")

    # The gripper reader is created up front (not started yet) so its bidirectional
    # send_json can drive the GUI's Open/Close/Stop + slider. Live path only —
    # --mock has no gripper bridge, so the controls are hidden there.
    gripper_reader = None
    gripper_reader_right = None
    if not args.mock:
        gripper_reader = _WSReader(state, args.gripper_url, lambda m: _on_gripper_msg(state, m), "gripper")
        # Same for the RIGHT bridge when the launcher asserted it came up (it also
        # runs with --web-control, so its encoder WS accepts the identical control
        # frames). Created here, not in the readers list below, purely so its
        # send_json can back that hand's Open/Close/Stop + slider.
        if getattr(args, "gripper_right_url", ""):
            gripper_reader_right = _WSReader(
                state, args.gripper_right_url, lambda m: _on_gripper_right_msg(state, m),
                "gripper_right")

    # Recorder control client (only with --record): drives the out-of-process
    # recorder's control WS from the Recording panel. Started after the scene.
    recorder_client = None
    if args.record:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "recording"))
        from recorder_control_client import RecorderControlClient
        recorder_client = RecorderControlClient(url=args.recorder_url)

    # mapper needs the yam joint names; build it from the scene's URDF after load.
    scene = CollectionScene(
        server, state, ik=ik, mapper=None, yam_urdf=args.yam_urdf,
        cam_offset_xyz=args.cam_offset, cam_offset_rpy=args.cam_rpy, cam_fov_deg=args.cam_fov_deg,
        robot_height=args.robot_height, auto_calibrate=not args.no_auto_calibrate, ft_state=ft_state,
        gripper_send=(gripper_reader.send_json if gripper_reader is not None else None),
        recomp_enabled=recomp_enabled, recorder_client=recorder_client,
        rnea_available=rnea_available, tracking_mode=tracking_mode,
        display_mode=display_mode, reach_scale=args.reach_scale,
        raw_frame=getattr(args, "raw_frame", False),
        # Right-hand hardware actually present this session. camera/gripper are asserted
        # by their URL (the launcher only passes one for a device that came up); the F/T
        # by a reader that really opened. All False in 1pt / --mock -> nothing changes.
        has_right_camera=bool(getattr(args, "camera_right_url", "")),
        has_right_gripper=bool(getattr(args, "gripper_right_url", "")),
        has_right_ft=ft_reader_right is not None,
        # RIGHT hand's own F/T + gripper, so its panel gets the identical controls.
        ft_state_right=ft_state_right,
        gripper_right_send=(gripper_reader_right.send_json
                            if gripper_reader_right is not None else None),
        recomp_enabled_right=recomp_enabled_right,
        rnea_available_right=rnea_available_right,
        show_camera_view=not getattr(args, "no_camera_view", False),
        camera_preview_hz=float(getattr(args, "camera_preview_hz", 10.0)),
    )
    if ik is not None:
        from g1_right_hand_ik import YamConfigMapper

        mapper = YamConfigMapper(ik._humi_joint_names, scene._yam_names)
        scene.mapper = mapper
        scene._apply_home_pose()

    # Start the raw F/T publisher (must bind before the reader pushes into it).
    if ft_publisher is not None:
        try:
            ft_publisher.start()
            print(f"Raw F/T publisher on ws://127.0.0.1:{args.ft_pub_port}")
        except Exception as exc:  # noqa: BLE001 - recording is best-effort
            print(f"NOTE: raw F/T publisher failed to start ({exc!r}); F/T won't be recorded.")
            ft_publisher = None
    # RIGHT-hand publisher, same best-effort discipline: a failure here costs the right
    # hand's wrench (no robot1_ft_wrench) and nothing else.
    if ft_publisher_right is not None:
        try:
            ft_publisher_right.start()
            print(f"Raw F/T publisher (RIGHT) on ws://127.0.0.1:{args.ft_pub_right_port}")
        except Exception as exc:  # noqa: BLE001 - recording is best-effort
            print(f"NOTE: RIGHT raw F/T publisher failed to start ({exc!r}); "
                  f"the right hand's F/T won't be recorded.")
            ft_publisher_right = None
    # Start the recorder control client (connects to the recorder process if up).
    if recorder_client is not None:
        recorder_client.start()

    # Start the in-process Bota reader(s) now that the server/scene exist.
    if ft_reader is not None:
        ft_reader.start()
    if ft_reader_right is not None:
        ft_reader_right.start()

    # Readers / mock feeder.
    readers = []
    mock_feeder = None
    if args.mock:
        # Primary (LEFT) hand home = what the mock circles + the camera rides.
        home = ik.home_primary_hand.translation() if ik is not None else np.array([0.4, 0.2, 0.8])
        mock_feeder = _MockFeeder(state, home, active_roles=_active_roles)
        mock_feeder.start()
        # In mock, auto-calibrate at the home so IK tracks immediately, and
        # reflect that in the GUI (the mock path bypasses the recal button). The
        # reference for EVERY active role is (home, identity) — the feeder's
        # per-role offsets are then mirrored as home-anchored deltas.
        if ik is not None:
            ident = np.array([1.0, 0.0, 0.0, 0.0])
            ik.recalibrate({r: (home, ident) for r in _active_roles})
            scene.gui_calib.value = f"calibrated (mock, {tracking_mode})"
    else:
        readers = [
            _WSReader(state, args.tracker_url, lambda m: _on_tracker_msg(state, m), "tracker"),
            _WSReader(state, args.camera_url, lambda m: _on_camera_msg(state, m), "camera"),
            gripper_reader,  # created above so its send_json could drive the GUI
        ]
        # RIGHT-hand streams, subscribed only when the launcher asserted the device
        # exists (a URL == "this is plugged in"). With no URL nothing is opened, so a
        # 1pt session — and a 3pt session before the right hardware is wired up — keeps
        # exactly the historical three readers.
        if getattr(args, "camera_right_url", ""):
            readers.append(_WSReader(state, args.camera_right_url,
                                     lambda m: _on_camera_right_msg(state, m), "camera_right"))
        if gripper_reader_right is not None:
            # Created above so its send_json could back the right hand's Open/Close/
            # Stop + slider. The j/k keys and that bridge's own page keep working —
            # the bridge is the single CAN writer and simply serves whichever intent
            # arrives, so there is no second owner of the bus.
            readers.append(gripper_reader_right)
        for r in readers:
            r.start()

    stop = threading.Event()

    def _sig(_s, _f):
        stop.set()

    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)

    print(f"Viser serving on http://localhost:{args.port}  "
          f"(mock={args.mock}, display={display_mode}, ik={ik is not None})")
    if not args.mock:
        print(f"  tracker: {args.tracker_url}   camera: {args.camera_url}   gripper: {args.gripper_url}")
        _right_bits = []
        if getattr(args, "camera_right_url", ""):
            _right_bits.append(f"camera_R: {args.camera_right_url}")
        if getattr(args, "gripper_right_url", ""):
            _right_bits.append(f"gripper_R: {args.gripper_right_url}")
        if ft_reader_right is not None:
            _right_bits.append("ft_R: in-process")
        print(f"  RIGHT hand: {'   '.join(_right_bits) if _right_bits else '(tracker only)'}")
    period = 1.0 / args.render_hz
    chart_period = 1.0 / 15.0
    last_chart = 0.0
    try:
        while not stop.is_set():
            t = time.perf_counter()
            # While a payload recompensation runs on EITHER hand, throttle the whole
            # render loop right down so that hand's bota reader thread gets the
            # interpreter to feed the estimator EKF at ~1 kHz (it needs a high,
            # steady rate to converge — this is what the standalone
            # estimate_payload.py gets for free). We still pump the overlays
            # (countdown / done auto-close) at this slow rate. The OTHER sensor's
            # reader keeps streaming: it sleeps ~1/read_hz between frames and so
            # holds the GIL for only a low-single-digit percentage, unlike this
            # 60 Hz loop.
            recomp_busy = scene._recomp_busy_panel() is not None
            if recomp_busy:
                scene._update_recomp_overlay()
                time.sleep(0.2)  # ~5 Hz: barely competes with the estimator feed
                continue
            if not scene.gui_pause.value:
                scene.update()
                if t - last_chart >= chart_period:
                    last_chart = t
                    scene.update_chart()
            elif recomp_enabled or recomp_enabled_right:
                # Pause freezes the 3D scene + charts, but the recompensation
                # overlays must keep updating (countdown / auto-close) — pure UI
                # bookkeeping independent of the IK render, and otherwise a
                # mid-recal Pause would leave a stale fullscreen modal.
                scene._update_recomp_overlay()
            sleep = period - (time.perf_counter() - t)
            if sleep > 0:
                time.sleep(sleep)
    finally:
        if mock_feeder is not None:
            mock_feeder.stop()
        for r in readers:
            r.stop()
        if ft_reader is not None:
            ft_reader.stop()
            ft_reader.join(timeout=2.0)
        if ft_reader_right is not None:
            ft_reader_right.stop()
            ft_reader_right.join(timeout=2.0)
        if ft_publisher is not None:
            ft_publisher.stop()
        if ft_publisher_right is not None:
            ft_publisher_right.stop()
        if recorder_client is not None:
            recorder_client.stop()
        server.stop()
        print("Shut down cleanly.")


if __name__ == "__main__":
    main()
