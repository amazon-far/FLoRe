#!/usr/bin/env python
"""Hardware-free logic tests for the data-collection viz.

Covers the pure pieces that don't need a viser server or hardware:
  * GripperRange encoder->jaw mapping (clamp, endpoints, linearity).
  * YamConfigMapper: 25 HuMI joints -> yam actuated order, defaults, gripper.
  * EncoderBuffer rolling-window export (newest at x=0).
  * RightHandIK home-anchored tracker->target math (hold-still + mirror).

Run (no pytest needed — runs as a script too):
  PYTHONPATH=third_party/humi_ik \
    .venv/bin/python viz/test_logic_collection.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from g1_right_hand_ik import GripperRange, YamConfigMapper, yam_actuated_joint_names  # noqa: E402


# --------------------------------------------------------------------------
# GripperRange
# --------------------------------------------------------------------------
def test_gripper_endpoints():
    gr = GripperRange(open_rad=4.90, closed_rad=0.0, jaw_open_m=0.0475)
    assert abs(gr.jaw_m(0.0) - 0.0) < 1e-12          # closed -> jaws touch
    assert abs(gr.jaw_m(4.90) - 0.0475) < 1e-12      # open -> full stroke
    assert abs(gr.jaw_m(2.45) - 0.02375) < 1e-9      # half travel -> half jaw


def test_gripper_clamps_out_of_range():
    gr = GripperRange()
    assert gr.fraction(-5.0) == 0.0
    assert gr.fraction(99.0) == 1.0
    assert gr.jaw_m(-5.0) == 0.0
    assert abs(gr.jaw_m(99.0) - 0.0475) < 1e-12


def test_gripper_degenerate_span():
    gr = GripperRange(open_rad=1.0, closed_rad=1.0)
    assert gr.fraction(1.0) == 0.0  # no span -> safe 0, not div-by-zero


# --------------------------------------------------------------------------
# YamConfigMapper
# --------------------------------------------------------------------------
def _fake_humi_names():
    # The 25 HuMI hinge joints, in the MJCF q[7:] order.
    return [
        "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint",
        "left_ankle_pitch_joint", "left_ankle_roll_joint",
        "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint",
        "right_ankle_pitch_joint", "right_ankle_roll_joint",
        "waist_yaw_joint",
        "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
        "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint",
        "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
        "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint",
    ]


def test_mapper_matches_yam_actuated_count():
    humi = _fake_humi_names()
    yam = yam_actuated_joint_names()
    mapper = YamConfigMapper(humi, yam)
    q = np.zeros(7 + len(humi))
    cfg = mapper.to_yam(q, left_jaw_m=0.0, right_jaw_m=0.0)
    assert len(cfg) == len(yam)


def test_mapper_copies_shared_joints_by_name():
    humi = _fake_humi_names()
    yam = yam_actuated_joint_names()
    mapper = YamConfigMapper(humi, yam)
    # set a recognizable value on right_elbow in HuMI's q[7:]
    q = np.zeros(7 + len(humi))
    q[7 + humi.index("right_elbow_joint")] = 0.321
    cfg = mapper.to_yam(q)
    assert abs(cfg[yam.index("right_elbow_joint")] - 0.321) < 1e-12


def test_mapper_defaults_yam_only_joints_to_zero():
    humi = _fake_humi_names()
    yam = yam_actuated_joint_names()
    mapper = YamConfigMapper(humi, yam)
    q = np.full(7 + len(humi), 9.9)  # everything HuMI = 9.9
    cfg = mapper.to_yam(q)
    # yam-only revolutes (not in HuMI) must be 0, not 9.9
    for name in ("waist_roll_joint", "waist_pitch_joint", "left_wrist_yaw_joint", "right_wrist_yaw_joint"):
        assert abs(cfg[yam.index(name)]) < 1e-12, f"{name} should default 0"


def test_mapper_injects_gripper_fingers():
    humi = _fake_humi_names()
    yam = yam_actuated_joint_names()
    mapper = YamConfigMapper(humi, yam)
    cfg = mapper.to_yam(np.zeros(7 + len(humi)), left_jaw_m=0.01, right_jaw_m=0.0475)
    assert abs(cfg[yam.index("left_gripper_finger1_joint")] - 0.01) < 1e-12
    assert abs(cfg[yam.index("right_gripper_finger1_joint")] - 0.0475) < 1e-12


def test_yam_excludes_mimic_finger2():
    yam = yam_actuated_joint_names()
    assert "left_gripper_finger2_joint" not in yam
    assert "right_gripper_finger2_joint" not in yam


# --------------------------------------------------------------------------
# EncoderBuffer
# --------------------------------------------------------------------------
def test_encoder_buffer_window_and_zero():
    from collection_viz import EncoderBuffer

    buf = EncoderBuffer(window_s=2.0, max_points=100)
    for i in range(50):
        buf.append(float(i) * 0.1, frac=i / 50.0)  # 0..5 s, 0.1 s apart
    x, y = buf.export()
    assert x[-1] == 0.0, "newest sample at x=0"
    assert x.min() >= -2.0 - 1e-9, "clipped to the 2 s window"
    assert y[-1] > y[0], "fraction increasing"


# --------------------------------------------------------------------------
# RightHandIK home-anchoring math (needs mink+mujoco+HuMI)
# --------------------------------------------------------------------------
def test_ik_hold_still_at_calibration():
    from g1_right_hand_ik import RightHandIK

    ik = RightHandIK()
    home = ik.home_right_hand.translation().copy()
    ref_q = np.array([1.0, 0.0, 0.0, 0.0])
    ik.recalibrate(home + np.array([0.3, -0.1, 0.2]), ref_q)
    # feed the EXACT reference -> target should equal home (hold still)
    tgt = ik._tracker_to_target(home + np.array([0.3, -0.1, 0.2]), ref_q)
    assert np.allclose(tgt.translation(), home, atol=1e-9), "calib pose must map to home"


def test_ik_mirrors_displacement():
    from g1_right_hand_ik import RightHandIK

    ik = RightHandIK(reach_scale=1.0)
    home = ik.home_right_hand.translation().copy()
    ref = np.array([0.5, -0.2, 1.0])
    ref_q = np.array([1.0, 0.0, 0.0, 0.0])
    ik.recalibrate(ref, ref_q)
    disp = np.array([0.05, 0.10, -0.07])
    tgt = ik._tracker_to_target(ref + disp, ref_q)
    assert np.allclose(tgt.translation(), home + disp, atol=1e-9), "1:1 displacement mirror"


def test_ik_base_does_not_drift():
    from g1_right_hand_ik import RightHandIK

    ik = RightHandIK()
    home = ik.home_right_hand.translation().copy()
    base0 = ik.home_q()[0:3].copy()
    ik.recalibrate(home, np.array([1.0, 0.0, 0.0, 0.0]))
    for _ in range(20):
        q = ik.step(home + np.array([0.0, 0.0, 0.08]), np.array([1.0, 0.0, 0.0, 0.0]))
    assert abs(q[2] - base0[2]) < 0.03, "right-hand-only IK must not move the base"


# --------------------------------------------------------------------------
# Runner (no pytest required)
# --------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Tracker axes reset-to-wrist (mount rotation)
# ---------------------------------------------------------------------------
def _rpy_q(roll, pitch, yaw):
    """wxyz quaternion for intrinsic Z(yaw)*Y(pitch)*X(roll), degrees."""
    import math

    from g1_three_point_ik import quat_mul
    r, p, y = (math.radians(v) for v in (roll, pitch, yaw))
    qz = np.array([math.cos(y / 2), 0.0, 0.0, math.sin(y / 2)])
    qy = np.array([math.cos(p / 2), 0.0, math.sin(p / 2), 0.0])
    qx = np.array([math.cos(r / 2), math.sin(r / 2), 0.0, 0.0])
    return quat_mul(qz, quat_mul(qy, qx))


def test_mount_rotation_zeroes_the_reference_posture():
    """"Reset the tracker axes onto the wrist" == capture the current puck orientation
    as the mount rotation, so the displayed triad reads axis-aligned in the reference
    posture. The puck is bolted on at an arbitrary angle, and without this the operator
    cannot tell a mount rotation from a world-alignment error."""
    from g1_three_point_ik import quat_inv, quat_mul

    mount = _rpy_q(37, 0, -22)                 # an arbitrarily bolted-on puck
    inv = quat_inv(mount)
    disp = quat_mul(mount, inv)
    assert np.allclose(np.abs(disp), [1, 0, 0, 0], atol=1e-9), disp


def test_mount_rotation_passes_real_motion_through_undistorted():
    """Removing the mount rotation must not distort SUBSEQUENT motion: a real 90-degree
    yaw of the hand still displays as exactly a 90-degree yaw."""
    from g1_three_point_ik import quat_inv, quat_mul

    mount = _rpy_q(37, 0, -22)
    inv = quat_inv(mount)
    for rpy in [(0, 0, 90), (0, 45, 0), (30, 0, 0), (10, -20, 70)]:
        moved = quat_mul(_rpy_q(*rpy), mount)   # a world-frame rotation of the hand
        disp = quat_mul(moved, inv)
        assert np.allclose(np.abs(disp), np.abs(_rpy_q(*rpy)), atol=1e-9), rpy


def test_mount_rotation_absent_is_identity():
    """No captured mount == unchanged behavior, bit for bit (the default path)."""
    import g1_three_point_ik as ik

    class _Stub:
        _mount_quat_inv: dict = {}
        display_quat = ik.ThreePointIK.display_quat

    q = _rpy_q(12, 34, 56)
    assert np.allclose(_Stub().display_quat("left", q), q)


def test_axes_reset_is_wired_flag_button_and_render_consumption():
    """The human asked for a tracker reset; a method with no caller is not that. Pins the
    full flag handshake: a GUI button raises a flag under the lock, and the RENDER loop
    (which owns the IK object — mink's Configuration is not thread-safe) consumes it and
    calls reset_axes_to_wrist behind the same live-pose gate as Recalibrate."""
    import pathlib

    src = pathlib.Path(__file__).with_name("collection_viz.py").read_text()
    assert "axes_reset_request: bool = False" in src
    assert "self.state.axes_reset_request = True" in src
    assert "do_axes = self.state.axes_reset_request" in src
    assert "self.state.axes_reset_request = False" in src
    assert 'gui.add_button("Reset tracker axes to wrist")' in src
    assert "self.ik.reset_axes_to_wrist(poses)" in src
    i = src.index("if do_axes and self.ik is not None:")
    block = src[i:i + 900]
    assert "_all_active_live(poses)" in block, "must refuse a partial/stale capture"
    assert "skipped" in block, "must tell the operator why it did nothing"
    # ...and the captured rotation actually reaches the displayed triads — the RIGHT hand
    # by name, and EVERY body reference (chest + the two 5pt feet) through the role dict,
    # which is what keeps a fifth puck from being drawn on its un-reset axes.
    assert "self._display_quat(role, right_quat)" in src
    assert 'self._display_quat(role, body[role]["quat"])' in src


def test_axes_reset_returns_the_quats_and_leaves_the_solver_alone():
    """It must RETURN the captured rotations (a mount that lives only in RAM is lost on
    restart, and a session recorded under a different mount is indistinguishable), and it
    must NOT touch the solver — only the render thread may mutate that."""
    import inspect

    from g1_three_point_ik import ThreePointIK

    assert "poses" in inspect.signature(ThreePointIK.reset_axes_to_wrist).parameters
    assert "PERSIST" in (ThreePointIK.reset_axes_to_wrist.__doc__ or "").upper()
    src = inspect.getsource(ThreePointIK.reset_axes_to_wrist)
    assert "return out" in src
    assert "_solver" not in src, "reset_axes_to_wrist must not mutate the IK solver"



def test_axes_reset_persists_the_mount_rotation():
    """A mount rotation held only in RAM dies on restart, and a session recorded under a
    different mount is indistinguishable from one recorded under this one — so the capture
    must reach disk WITH the take. Pins the whole hop: the recorder publishes session_dir
    on its status frame (the viz is a separate process with no filesystem view of it), and
    the viz writes <session>/calibration/tracker_mount.json, which the converter's
    _resolve_calib already prefers over the repo default."""
    import pathlib

    viz = pathlib.Path(__file__).with_name("collection_viz.py").read_text()
    rec = (pathlib.Path(__file__).parents[1] / "recording" / "recorder.py").read_text()
    fmt = (pathlib.Path(__file__).parents[1] / "recording" / "recording_format.py").read_text()

    assert '"session_dir": self._session_dir,' in rec, "recorder must publish the session root"
    assert "_persist_mount_quats(mounts)" in viz, "capture must call the persister"
    assert '.get("session_dir")' in viz, "viz must read it off the status frame"
    # The filename is duplicated (the viz does not import recording_format) — keep in sync.
    assert 'TRACKER_MOUNT_FILE = "tracker_mount.json"' in fmt
    assert '"tracker_mount.json"' in viz
    # Failure to persist must be loud, not silent, and must not take the rig down.
    i = viz.index("def _persist_mount_quats")
    body = viz[i:viz.index("def _display_quat", i)]
    assert "except OSError" in body and "WARNING" in body
    assert "NOT persisted" in body, "must say so when there is no session yet"


# ---------------------------------------------------------------------------
# Gripper-only display mode (DISPLAY_MODE=gripper): the yam gripper meshes +
# camera follow the tracker via a home-anchored delta, with NO mink IK.
# ---------------------------------------------------------------------------
def test_gripper_geom_extracts_three_colored_pieces_per_hand():
    """build_gripper_geom pulls the base + two fingers for a hand out of the URDF
    WITH the URDF material colours baked in (so gripper-only mode looks identical to
    humanoid mode), and left/right never cross-wire."""
    import gripper_only_view as gov

    u = gov._load_urdf(gov._YAM_URDF_DEFAULT)
    for side in ("left", "right"):
        g = gov.build_gripper_geom(side, urdf=u)
        assert g.side == side
        names = {p.name for p in g.pieces()}
        assert names == {"base", "finger1", "finger2"}
        for p in g.pieces():
            assert len(p.mesh.vertices) > 1000, f"{side} {p.name} mesh empty"
            # colours baked into face_colors (green base ~[28,72,3], blue tip ~[151,200,235])
            assert hasattr(p.mesh.visual, "face_colors"), f"{side} {p.name} no baked colour"
        assert not g.base.is_finger and g.finger1.is_finger and g.finger2.is_finger


def test_gripper_geom_jaw_lerp_is_linear_and_base_static():
    """Finger wrist-relative translation is LINEAR in the jaw opening (so we can LERP
    the two endpoints per frame instead of re-running FK), and the base link is
    static across the stroke."""
    import gripper_only_view as gov

    g = gov.build_gripper_geom("left")
    # base doesn't move with the jaw
    assert np.allclose(g.base.pos_at(0.0), g.base.pos_at(1.0), atol=1e-9)
    # finger midpoint == mean of the endpoints (linearity)
    for f in (g.finger1, g.finger2):
        assert np.allclose(f.pos_at(0.5), 0.5 * (f.pos_closed + f.pos_open), atol=1e-9)
        # clamp out of range
        assert np.allclose(f.pos_at(-1.0), f.pos_closed)
        assert np.allclose(f.pos_at(2.0), f.pos_open)


def test_home_anchor_matches_ik_tracker_to_target():
    """gripper_only_view.home_anchor MUST be numerically identical to the IK's
    ThreePointIK._tracker_to_target — it's the SAME home-anchored delta mapping,
    just without the mink solver. If they drift, gripper mode and humanoid mode
    would place the gripper differently for the same tracker motion."""
    import gripper_only_view as gov
    from g1_right_hand_ik import quat_inv
    from g1_three_point_ik import ThreePointIK

    ik = ThreePointIK(active_roles=("left",))
    home = ik._home_pose["left"]
    hp, hq = home.translation().copy(), home.rotation().wxyz.copy()
    ref_p = np.array([0.5, -0.2, 1.0])
    ref_q = np.array([0.92388, 0.0, 0.0, 0.38268])  # 45 deg yaw
    ik._ref_pos["left"] = ref_p.copy()
    ik._ref_quat_inv["left"] = quat_inv(ref_q)
    cur_p = ref_p + np.array([0.05, 0.10, -0.07])
    cur_q = np.array([0.70711, 0.70711, 0.0, 0.0])  # 90 deg roll
    se3 = ik._tracker_to_target("left", cur_p, cur_q)
    gp, gq = gov.home_anchor(hp, hq, ref_p, ref_q, cur_p, cur_q, reach_scale=1.0)
    assert np.allclose(gp, se3.translation(), atol=1e-9), (gp, se3.translation())
    qb = se3.rotation().wxyz
    qa = gq / np.linalg.norm(gq)
    qb = qb / np.linalg.norm(qb)
    assert np.allclose(qa, qb, atol=1e-9) or np.allclose(qa, -qb, atol=1e-9), (qa, qb)


def test_home_wrist_pose_matches_fujis_committed_display_homes():
    """HOME_WRIST_POSE must equal FLoRe's ``display.py::DISPLAY_HOME_POS``, key for key.

    Both dicts place the SAME session: the collection viz + both replay viewers read this
    one, FLoRe's offline inference viewer reads that one. If they drift, the same take is
    drawn metres apart depending on which tool you opened, which is exactly the confusion
    that costs a debugging round. Every role must be present in both — a role missing HERE
    silently falls back to the LEFT WRIST home (every consumer looks it up as
    ``.get(role, HOME_WRIST_POSE["left"])``), so a 3pt chest would render on the left hand.

    Skips rather than fails when FLoRe is not alongside, since the submodule is usable on
    its own. Only the home-anchored path reads these; RAW_FRAME never looks a home up."""
    import importlib.util

    import gripper_only_view as gov

    cand = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "..", "..", "flore", "s1_policy", "inference", "display.py")
    if not os.path.isfile(cand):
        print("  skip  (FLoRe display.py not alongside)")
        return
    spec = importlib.util.spec_from_file_location("_fuji_display", cand)
    fd = importlib.util.module_from_spec(spec)
    sys.modules["_fuji_display"] = fd          # dataclasses in that module need this
    spec.loader.exec_module(fd)

    assert set(gov.HOME_WRIST_POSE) == set(fd.DISPLAY_HOME_POS), \
        (sorted(gov.HOME_WRIST_POSE), sorted(fd.DISPLAY_HOME_POS))
    for role, (hp, hq) in gov.HOME_WRIST_POSE.items():
        assert np.allclose(hp, fd.DISPLAY_HOME_POS[role], atol=1e-12), (role, hp)
        # Identity rotation is what makes the anchor frame render +z up, and (for the two
        # hands) what makes the jaws start parallel.
        assert np.allclose(hq, [1.0, 0.0, 0.0, 0.0], atol=1e-12), (role, hq)
    # The two hands differ only in y, straddling the centreline.
    lp, rp = gov.HOME_WRIST_POSE["left"][0], gov.HOME_WRIST_POSE["right"][0]
    assert np.allclose(lp[[0, 2]], rp[[0, 2]], atol=1e-9), "hands differ only in y"
    assert abs(lp[1] + rp[1]) < 1e-9, "hands straddle the centreline"


def test_chest_home_is_the_urdfs_pelvis():
    """The collection viz's CHEST home is the G1's pelvis, re-derived from the yam URDF.

    "Reset robot to home (c)" must put the 3pt chest triad where the ROBOT's body reference
    is — between and behind the two grippers, at about their height — not at the standing
    OPERATOR's chest. That second thing is what the shared display home
    (``gripper_only_view.HOME_WRIST_POSE["head"]``) means, and it has to keep meaning it:
    both replay viewers and FLoRe's offline inference viewer re-base a RECORDED take onto it
    and motion completion's ``mc_frames.RIG_CHEST_STAND_Z`` (not part of the FLoRe release)
    IS its z. Hence the override in
    ``collection_viz``, and hence this test — re-derive the override from the URDF, and
    assert in the same breath that the shared dict did NOT move.

    At q=0 a revolute joint contributes only its origin, so composing the joint origins from
    ``pelvis`` to ``{side}_wrist_yaw_link`` IS the zero-configuration forward kinematics —
    the same derivation ``mc_frames.ZERO_CFG_WRIST_REL_PELVIS`` is pinned by, done here with
    a local parser so this file needs neither yourdfpy nor the motion-completion module.
    """
    import xml.etree.ElementTree as ET

    import collection_viz as cv
    import gripper_only_view as gov

    urdf = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                        "robot_assets", "g1_yam", "g1_29dof_yam.urdf")
    joints = []
    for j in ET.parse(urdf).getroot().findall("joint"):
        o = j.find("origin")
        get = (lambda k: [float(v) for v in (o.get(k) or "0 0 0").split()]) if o is not None \
            else (lambda k: [0.0, 0.0, 0.0])
        joints.append((j.find("parent").get("link"), j.find("child").get("link"),
                       np.array(get("xyz")), np.array(get("rpy"))))

    def _rpy_to_R(rpy):
        """URDF rpy -> Rz(yaw) Ry(pitch) Rx(roll) (the fixed-axis convention)."""
        r, p, y = (float(v) for v in rpy)
        cr, sr, cp, sp, cy, sy = (np.cos(r), np.sin(r), np.cos(p), np.sin(p),
                                  np.cos(y), np.sin(y))
        return (np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
                @ np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
                @ np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]]))

    def _from_pelvis(link: str):
        """(R, t) of ``link`` relative to ``pelvis`` with every joint at 0."""
        seq, cur = [], link
        while cur != "pelvis":
            hit = next((j for j in joints if j[1] == cur), None)
            assert hit is not None, f"no parent joint for {cur}"
            seq.append(hit)
            cur = hit[0]
        R, t = np.eye(3), np.zeros(3)
        for _parent, _child, xyz, rpy in reversed(seq):
            t = t + R @ xyz
            R = R @ _rpy_to_R(rpy)
        return R, t

    # The URDF leg: pelvis -> wrist_yaw_link, mirrored to within round-off.
    wrist_rel_pelvis = {}
    for side in ("left", "right"):
        R, t = _from_pelvis(f"{side}_wrist_yaw_link")
        # Near-identity is what makes the zero config's forearms level along +x; the
        # constant is USED as exact identity, so bound the error being ignored.
        assert np.abs(R - np.eye(3)).max() < 5e-4, (side, R)
        wrist_rel_pelvis[side] = t
    assert abs(wrist_rel_pelvis["left"][1] - -wrist_rel_pelvis["right"][1]) < 2e-5, \
        "the URDF's two arms are not mirrored in y"

    # The viz's hand homes are PUCK poses (identity rotation), so the wrist frames sit a
    # plain _TRACKER_TRUE_XYZ_M behind them; the chest home is their midpoint minus the
    # URDF's pelvis->wrist vector.
    puck_local = np.asarray(cv._TRACKER_TRUE_XYZ_M, dtype=float)
    mid_wrist = 0.5 * (gov.HOME_WRIST_POSE["left"][0] + gov.HOME_WRIST_POSE["right"][0]) \
        - puck_local
    want = mid_wrist - 0.5 * (wrist_rel_pelvis["left"] + wrist_rel_pelvis["right"])
    got, got_q = cv.display_home_pose("head")
    assert abs(got[0] - want[0]) < 1e-6 and abs(got[2] - want[2]) < 1e-6, (got, want)
    # y is EXACTLY the centreline: the URDF's 1e-5 m left/right asymmetry is round-off in
    # the joint origins and "between the two grippers" is the intent.
    assert got[1] == 0.0, got
    # "a bit behind, at about the same height": behind the wrist, and much closer to the
    # hands than the operator's chest is (the shared home is 35 cm above them).
    lp = gov.HOME_WRIST_POSE["left"][0]
    assert got[0] < lp[0] - 0.15, ("the chest home is not behind the grippers", got, lp)
    assert abs(got[2] - lp[2]) < 0.20, ("the chest home is not at gripper height", got, lp)
    # The cross-check that this IS a standing G1's pelvis and not a picked spot: within a
    # couple of cm of the pelvis stand height derived from the foot's own collision solids.
    _, ankle = _from_pelvis("left_ankle_roll_link")
    pelvis_stand_z = -ankle[2] + 0.035          # mc_frames.SOLE_BELOW_ANKLE_ROLL
    assert abs(got[2] - pelvis_stand_z) < 0.02, (got[2], pelvis_stand_z)
    # NO weird rotation: identity, exactly like every HOME_WRIST_POSE entry. The
    # home-anchored path right-multiplies the live world delta onto this.
    assert np.allclose(got_q, [1.0, 0.0, 0.0, 0.0], atol=1e-12), got_q
    # The HANDS are untouched pass-throughs, and the SHARED chest home has not moved.
    for role in ("left", "right"):
        hp, hq = cv.display_home_pose(role)
        assert np.allclose(hp, gov.HOME_WRIST_POSE[role][0], atol=1e-12), role
        assert np.allclose(hq, gov.HOME_WRIST_POSE[role][1], atol=1e-12), role
    assert np.allclose(gov.HOME_WRIST_POSE["head"][0], [0.30, 0.0, 1.30], atol=1e-12), \
        "the SHARED chest display home moved; the replay viewers and mc_frames read it"


def test_display_home_pose_returns_fresh_arrays():
    """Callers hand the result straight to viser, so it must not alias a module constant.

    ``_seat_view_frame`` writes ``follow.position = tuple(...)`` today, but the home is also
    fed to ``home_anchor`` and a future in-place ``+=`` on a shared array would silently walk
    the home a little further every frame — the kind of drift that looks like tracker creep."""
    import collection_viz as cv

    for role in ("left", "right", "head", "unknown_future_role"):
        a, _ = cv.display_home_pose(role)
        b, _ = cv.display_home_pose(role)
        assert a is not b, role
        a += 1.0
        c, _ = cv.display_home_pose(role)
        assert np.allclose(b, c, atol=1e-12), role


def test_home_anchor_holds_still_at_reference():
    """At cur == ref the target IS the home pose (gripper doesn't jump on calibrate)."""
    import gripper_only_view as gov

    hp = np.array([0.35, 0.18, 0.95])
    hq = np.array([1.0, 0.0, 0.0, 0.0])
    ref_p = np.array([0.6, -0.1, 1.1])
    ref_q = np.array([0.8, 0.0, 0.6, 0.0]) / np.linalg.norm([0.8, 0.0, 0.6, 0.0])
    gp, gq = gov.home_anchor(hp, hq, ref_p, ref_q, ref_p, ref_q)
    assert np.allclose(gp, hp, atol=1e-9)
    assert np.allclose(np.abs(gq), np.abs(hq), atol=1e-9)


def test_puck_triad_carries_the_relabel_in_EVERY_path():
    """The drawn tracker->TCP relative rotation is M, unconditionally.

    This is a MECHANICAL constant: the VIVE puck is bolted onto the tool rotated by
    ``_RAW_RELABEL_M``, so the angle between the puck axes and the TCP/wrist axes is a
    property of the hardware. Home-anchoring rigidly rebases the whole cluster and cannot
    change a within-cluster relative orientation, so the puck child's local rotation must
    NOT be gated on RAW_FRAME. It used to be, which left the default path drawing the
    "tracker" triad with the same orientation as the "TCP" triad, and made the live viz
    disagree with all three other consumers of the same contract
    (hl_serving._TRIAD_LOCAL_ROT, reinfer_hl_take, replay_*_viz's puck_triad_local_quat).

    Asserted as (a) the value, and (b) the absence of the gate in the source, so a
    re-introduced ``if self._raw_frame`` fails even though the helper stays correct."""
    import inspect

    import collection_viz as cv

    M = cv._RAW_RELABEL_M
    assert np.allclose(M @ M, np.eye(3)), "M must be its own inverse for M^-1 == M"
    q = cv._puck_child_quat()
    for v in np.eye(3):
        assert np.allclose(cv._rot_vec_by_wxyz(q, v), M.T @ v, atol=1e-12), v
    src = inspect.getsource(cv.CollectionScene.__init__)
    assert "self._puck_child_q = _puck_child_quat()" in src, \
        "the puck child quat is no longer unconditional"
    assert "_puck_child_quat() if self._raw_frame" not in src, "the RAW_FRAME gate came back"


def test_relabel_matches_the_committed_tcp_extrinsic():
    """collection_viz._RAW_RELABEL_M must BE the rotation block of the committed
    robot_assets/calibration/tcp_extrinsic.json.

    The live viz hard-codes the relabel (it draws the current rig and has no dataset to
    read), while the converter and both replay viewers read that file. This test is the
    only thing stopping the two from drifting — and a drift would silently mis-orient
    every 'tracker' triad the operator uses to sanity-check the rig, in the one direction
    nothing downstream can detect (the trajectory stays bit-identical; only axes spin)."""
    import json
    import pathlib

    import collection_viz as cv

    path = pathlib.Path(__file__).parents[1] / "robot_assets" / "calibration" / "tcp_extrinsic.json"
    T = np.asarray(json.loads(path.read_text())["T_tracker_tcp"], dtype=float)
    assert np.allclose(T[:3, :3], cv._RAW_RELABEL_M, atol=1e-12), (T[:3, :3], cv._RAW_RELABEL_M)
    # The translation leg is the SAME physical lever, expressed in PUCK coords:
    # M @ (wrist-frame TCP - puck) == the file's t.
    lever_wrist = np.array([cv._TCP_ALONG_WRIST_X_M, 0.0, 0.0]) - np.asarray(cv._TRACKER_TRUE_XYZ_M)
    assert np.allclose(cv._RAW_RELABEL_M @ lever_wrist, T[:3, 3], atol=1e-9), \
        (cv._RAW_RELABEL_M @ lever_wrist, T[:3, 3])


def test_chest_frame_is_seated_without_the_hands_puck_lever():
    """_seat_view_frame must bind a HAND and the CHEST differently.

    A hand's follow frame is the WRIST origin and carries the puck as a child at
    _TRACKER_TRUE_XYZ_M, so the frame is pushed off the tracked pose for that child to
    land on it. The chest has no wrist and no mesh cluster — its triad IS the frame
    origin — so applying the same lever would draw the chest axis ~5 cm off the pose the
    puck actually reports, on every frame, with nothing to notice it by."""
    import collection_viz as cv

    class _F:
        position = None
        wxyz = None

    # No server / URDF / IK needed: _seat_view_frame reads only _grip_roles.
    scene = object.__new__(cv.CollectionScene)
    scene._grip_roles = ("left", "right")
    tgt_p = np.array([0.4, 0.1, 1.0])
    tgt_q = np.array([0.70711, 0.0, 0.70711, 0.0])   # 90 deg pitch, so the lever rotates

    for hand in ("left", "right"):
        fh = _F()
        cv.CollectionScene._seat_view_frame(scene, hand, fh, tgt_p, tgt_q)
        exp_p, exp_q = cv.follow_pose_binding_puck(tgt_p, tgt_q)
        assert np.allclose(fh.position, exp_p), (hand, fh.position, exp_p)
        assert np.allclose(fh.wxyz, exp_q), (hand, fh.wxyz, exp_q)
        # the invariant that binding exists for: the puck CHILD is back on the pose
        puck = np.asarray(fh.position) + cv._rot_vec_by_wxyz(fh.wxyz, cv._TRACKER_TRUE_XYZ_M)
        assert np.allclose(puck, tgt_p, atol=1e-12), (hand, puck, tgt_p)
        # and the lever is not a no-op, so the chest check below is a real difference
        assert not np.allclose(fh.position, tgt_p)

    fc = _F()
    cv.CollectionScene._seat_view_frame(scene, "head", fc, tgt_p, tgt_q)
    assert np.allclose(fc.position, tgt_p), fc.position
    assert np.allclose(fc.wxyz, tgt_q), fc.wxyz


def test_head_relabel_matches_the_committed_head_extrinsic():
    """collection_viz._HEAD_RELABEL_M must BE the rotation block of the committed
    robot_assets/calibration/head_extrinsic.json — the file the CONVERTER and the raw
    replay viewer read for the chest keypoint.

    Same drift guard as ``test_relabel_matches_the_committed_tcp_extrinsic``, and it
    matters more here: the chest relabel is rotation-ONLY, so a live-vs-stored drift moves
    NOTHING in any position plot or path overlay — ``head_pos`` is bit-identical either
    way. The only witness is the drawn triad, which is exactly what the operator uses to
    decide the chest puck is mounted right.

    Also pins the three properties that make it a distinct object from the hands' M, each
    of which is a way to get it wrong by "reusing the one we already have":
      * ZERO translation (the chest has no TCP lever),
      * NOT equal to M (a different puck on a different bracket),
      * ORDER 3 (H@H@H == I), where M is its own inverse — so ``H^-1 != H`` and an
        inverse-vs-forward slip does NOT cancel the way it silently would for M.
    """
    import json
    import pathlib

    import collection_viz as cv

    path = pathlib.Path(__file__).parents[1] / "robot_assets" / "calibration" / "head_extrinsic.json"
    T = np.asarray(json.loads(path.read_text())["T_tracker_head"], dtype=float)
    H = cv._HEAD_RELABEL_M
    assert np.allclose(T[:3, :3], H, atol=1e-12), (T[:3, :3], H)
    assert np.allclose(T[:3, 3], 0.0), f"chest relabel must be rotation-only, got t={T[:3, 3]}"
    assert np.isclose(np.linalg.det(H), 1.0) and np.allclose(H @ H.T, np.eye(3)), H
    assert not np.allclose(H, cv._RAW_RELABEL_M), "the chest reused the hands' M"
    assert np.allclose(H @ H @ H, np.eye(3)) and not np.allclose(H @ H, np.eye(3)), \
        "H must have order 3 (so H^-1 != H, unlike the hands' symmetric M)"


def test_chest_raw_pose_is_the_body_frame_the_converter_stores():
    """RAW-mode chest: position is the raw puck's, orientation is puck ∘ H.

    The operator's spec, pinned as the AXIS MAPPING they gave (90 deg about the drawn
    green +y, then 90 deg about the resulting red +x):

        x_new = -z_old,   y_new = +x_old,   z_new = -y_old

    which is a BODY-frame (right) multiplication — the same compose
    ``world_T_chest = world_T_puck @ [H|0]`` that ``convert_to_umi`` applies to
    ``head_rot_axis_angle``, so the drawn triad and the stored keypoint cannot disagree.

    And it must stay RAW-ONLY. ``home_anchor`` applies a WORLD-frame rotation delta
    (``R_target = cur ⊗ ref^-1 ⊗ R_home``), so a body relabel CANCELS out of it —
    asserted numerically below. Right-multiplying H in the anchored path would therefore
    not "also fix" that view, it would spin the chest triad off its home convention.
    """
    import inspect

    import collection_viz as cv
    import gripper_only_view as gov

    puck_p = np.array([0.1, -0.2, 1.25])
    puck_q = np.array([0.5, 0.5, 0.5, 0.5])          # a generic, non-axis-aligned pose
    puck_q = puck_q / np.linalg.norm(puck_q)
    p, q = cv.raw_chest_pose_from_puck(puck_p, puck_q)

    assert np.allclose(p, puck_p, atol=0), "rotation-only: the chest keeps the puck's position"
    # The drawn axes, expressed in the RAW puck's own axes.
    old = np.stack([cv._rot_vec_by_wxyz(puck_q, v) for v in np.eye(3)])   # x,y,z of the puck
    new = np.stack([cv._rot_vec_by_wxyz(q, v) for v in np.eye(3)])        # x,y,z as drawn
    for label, got, want in (("x_new = -z_old", new[0], -old[2]),
                             ("y_new = +x_old", new[1], +old[0]),
                             ("z_new = -y_old", new[2], -old[1])):
        assert np.allclose(got, want, atol=1e-12), f"{label}: {got} != {want}"

    # ...and the anchored path is untouched BECAUSE the world-frame delta cancels H.
    ref_q = np.array([0.6, 0.0, 0.8, 0.0]); ref_q /= np.linalg.norm(ref_q)
    hq = np.array([1.0, 0.0, 0.0, 0.0])
    hp = ref_p = np.zeros(3)
    _, plain = gov.home_anchor(hp, hq, ref_p, ref_q, puck_p, puck_q)
    _, relabelled = gov.home_anchor(
        hp, hq, ref_p, cv.raw_chest_pose_from_puck(ref_p, ref_q)[1],
        puck_p, cv.raw_chest_pose_from_puck(puck_p, puck_q)[1])
    assert np.allclose(np.abs(plain), np.abs(relabelled), atol=1e-12), \
        "a body relabel must cancel out of home_anchor's world-frame delta"

    src = inspect.getsource(cv.CollectionScene._update_gripper_only)
    # The RAW branch seats EVERY body reference on its own body frame, role-keyed — the
    # chest and, in 5pt, the two feet, whose relabel may stop being the chest's.
    assert "raw_body_pose_from_puck(role, cur_p, cur_q)" in src, \
        "the RAW branch no longer seats the body references on their body frame"
    assert src.count("raw_body_pose_from_puck") == 1, \
        "the body relabel leaked out of the RAW branch (it cancels in the anchored one)"
    assert "raw_chest_pose_from_puck" not in src, \
        "the RAW branch must go through the role-keyed helper, not the chest alias"


def test_mode_roles_match_the_recorder():
    """The viz's own ``_ROLES_BY_MODE`` must equal ``recording_format``'s.

    The viz deliberately does not import recording_format (it is a viz process, not a
    recorder), so the two tables are duplicated — and a mode whose role list disagrees
    means the take carries a puck the page never drew, or the page anchors a role the
    recorder files nowhere. Both are silent: nothing fails, the picture is just missing a
    keypoint. Same duplication + same pin as the tracker_mount.json filename."""
    import sys as _sys

    import collection_viz as cv

    rec = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "recording")
    if rec not in _sys.path:
        _sys.path.insert(0, rec)
    import recording_format as fmt

    assert cv._ROLES_BY_MODE == fmt.TRACKER_ROLES_BY_MODE, \
        (cv._ROLES_BY_MODE, fmt.TRACKER_ROLES_BY_MODE)
    assert cv._BODY_ROLES_ALL == fmt.BODY_ROLES, (cv._BODY_ROLES_ALL, fmt.BODY_ROLES)
    assert cv._GRIP_ROLES_ALL == fmt.HAND_ROLES, (cv._GRIP_ROLES_ALL, fmt.HAND_ROLES)
    # Every body reference is drawable (a label) and has a declared relabel — a role in
    # the mode table with neither would render as an unnamed triad on raw puck axes.
    for role in cv._BODY_ROLES_ALL:
        assert role in cv._BODY_ROLE_LABEL, role
        assert role in cv._BODY_RELABEL_M, role
    # 5pt is 3pt plus the two feet, in that order (the wire order the low level reads).
    assert cv._ROLES_BY_MODE["5pt"][:3] == cv._ROLES_BY_MODE["3pt"], cv._ROLES_BY_MODE
    assert cv._ROLES_BY_MODE["5pt"][3:] == ("left_foot", "right_foot"), cv._ROLES_BY_MODE


def test_five_pt_humanoid_is_refused_by_name():
    """MODE=5pt + DISPLAY_MODE=humanoid must RAISE, not draw.

    ``ThreePointIK`` has two hand FrameTasks and a yaw-only torso task and NO foot task,
    so a 5pt humanoid would show two foot triads floating beside legs that ignore them —
    which reads as "the feet are tracked wrong" rather than "the feet are not tracked".
    Every other combination still passes through."""
    import collection_viz as cv
    from g1_three_point_ik import ROLES as IK_ROLES

    # The premise: the IK really has no foot task (if one is ever added, relax this).
    assert not any(r.endswith("_foot") for r in IK_ROLES), IK_ROLES
    try:
        cv.resolve_display_mode("humanoid", "5pt")
    except ValueError as exc:
        assert "5pt" in str(exc) and "humanoid" in str(exc), str(exc)
    else:
        raise AssertionError("5pt + humanoid must be refused")
    assert cv.resolve_display_mode("auto", "5pt") == "gripper"
    assert cv.resolve_display_mode("gripper", "5pt") == "gripper"
    for tm in ("1pt", "2pt", "3pt"):
        assert cv.resolve_display_mode("humanoid", tm) == "humanoid", tm


def test_foot_homes_are_the_urdfs_ankles():
    """The two 5pt FOOT display homes are the chest (pelvis) home + the URDF's own
    pelvis -> ankle_roll_link vector at q=0 — the same derivation the chest home uses.

    "Reset robot to home (c)" has to put every triad where the ROBOT's body is. A foot
    left on ``gripper_only_view.HOME_WRIST_POSE`` (0.30, ±0.12, 0.10) would sit 17 cm in
    front of the robot's ankles, and a foot with no entry at all would render ON the left
    hand (every consumer looks a home up as ``.get(role, ...["left"])``). Also asserts the
    SHARED display homes did not move, since the replay + offline viewers read those."""
    import xml.etree.ElementTree as ET

    import collection_viz as cv
    import gripper_only_view as gov

    urdf = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                        "robot_assets", "g1_yam", "g1_29dof_yam.urdf")
    joints = []
    for j in ET.parse(urdf).getroot().findall("joint"):
        o = j.find("origin")
        get = (lambda k: [float(v) for v in (o.get(k) or "0 0 0").split()]) if o is not None \
            else (lambda k: [0.0, 0.0, 0.0])
        joints.append((j.find("parent").get("link"), j.find("child").get("link"),
                       np.array(get("xyz")), np.array(get("rpy"))))

    def _rpy_to_R(rpy):
        r, p, y = (float(v) for v in rpy)
        cr, sr, cp, sp, cy, sy = (np.cos(r), np.sin(r), np.cos(p), np.sin(p),
                                  np.cos(y), np.sin(y))
        return (np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
                @ np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
                @ np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]]))

    def _from_pelvis(link: str):
        seq, cur = [], link
        while cur != "pelvis":
            hit = next((j for j in joints if j[1] == cur), None)
            assert hit is not None, f"no parent joint for {cur}"
            seq.append(hit)
            cur = hit[0]
        R, t = np.eye(3), np.zeros(3)
        for _parent, _child, xyz, rpy in reversed(seq):
            t = t + R @ xyz
            R = R @ _rpy_to_R(rpy)
        return R, t

    pelvis_home = np.asarray(cv._HEAD_HOME_XYZ_M, dtype=float)
    for role, side in (("left_foot", "left"), ("right_foot", "right")):
        R, t = _from_pelvis(f"{side}_ankle_roll_link")
        # At q=0 the ankle chain is EXACTLY identity, so the home is a plain vector add
        # (unlike the wrist chain, which is only near-identity).
        assert np.abs(R - np.eye(3)).max() < 1e-12, (side, R)
        got, got_q = cv.display_home_pose(role)
        np.testing.assert_allclose(got, pelvis_home + t, atol=1e-6)
        # Identity rotation, exactly like every other home: the anchored path
        # right-multiplies the live world delta onto this.
        assert np.allclose(got_q, [1.0, 0.0, 0.0, 0.0], atol=1e-12), got_q
        # Below the chest home and (just) above the floor: one rigid arrangement with the
        # hands and chest, NOT separately grounded, so it sits the same 1.29 cm high the
        # chest home does above the standing pelvis.
        assert got[2] < pelvis_home[2] - 0.5, (got, pelvis_home)
        assert 0.0 < got[2] < 0.10, got
    # mirrored in y, and DISTINCT (a shared home would hide a left/right foot swap)
    lf, _ = cv.display_home_pose("left_foot")
    rf, _ = cv.display_home_pose("right_foot")
    assert lf[1] == -rf[1] != 0.0, (lf, rf)
    assert np.allclose(lf[[0, 2]], rf[[0, 2]], atol=0.0), (lf, rf)
    # The SHARED display homes (replay + FLoRe offline viewer) are untouched.
    assert np.allclose(gov.HOME_WRIST_POSE["left_foot"][0], [0.30, 0.12, 0.10], atol=1e-12)
    assert np.allclose(gov.HOME_WRIST_POSE["right_foot"][0], [0.30, -0.12, 0.10], atol=1e-12)


def test_display_mode_auto_resolution_behavior():
    """The spec's whole point, pinned BEHAVIORALLY (not by source string-grep):
    auto -> gripper in EVERY tracking mode (the rig is two isolated handheld grippers;
    3pt is 2pt plus one chest triad, not a different display). Explicit modes pass
    through, so DISPLAY_MODE=humanoid still brings up the full G1.
    Runs the real resolver so a logic change fails here regardless of how the source
    is spelled."""
    from collection_viz import resolve_display_mode

    for tm in ("1pt", "2pt", "3pt"):
        assert resolve_display_mode("auto", tm) == "gripper", tm
        # explicit overrides pass straight through, in every tracking mode
        assert resolve_display_mode("humanoid", tm) == "humanoid"
        assert resolve_display_mode("gripper", tm) == "gripper"


def test_display_mode_wired_and_no_x11_clash():
    """Pin the wiring that the behavior test can't see: gripper mode builds NO IK,
    the scene receives the mode, and the launcher knob is DISPLAY_MODE (NOT the X11
    $DISPLAY var) passed as --display-mode."""
    import pathlib

    viz = pathlib.Path(__file__).with_name("collection_viz.py").read_text()
    # gripper mode builds NO IK; scene receives the resolved mode
    assert 'if not args.no_ik and display_mode != "gripper":' in viz
    assert "display_mode=display_mode" in viz
    assert "resolve_display_mode(" in viz, "main() must use the shared resolver"
    # launcher uses DISPLAY_MODE, not DISPLAY, and passes --display-mode
    launcher = (pathlib.Path(__file__).parents[1] / "run_collection_viz.sh").read_text()
    assert 'DISPLAY_MODE="${DISPLAY_MODE:-auto}"' in launcher
    assert "--display-mode" in launcher
    assert 'DISPLAY="${DISPLAY' not in launcher, "must not reuse the X11 $DISPLAY var"


def test_gripper_mode_does_not_touch_recording_or_can():
    """Gripper-only is DISPLAY-only: its per-frame update must not call the recorder,
    the gripper bridge send, or any IK/CAN path. Pin that _update_gripper_only moves
    only scene nodes (reads gripper_frac, writes frame/mesh positions)."""
    import inspect

    import collection_viz as cv

    src = inspect.getsource(cv.CollectionScene._update_gripper_only)
    jaw = inspect.getsource(cv.CollectionScene._animate_gripper_jaws)
    # BOTH the per-frame update AND the extracted jaw-animation helper must be
    # display-only (the jaw LERP moved into _animate_gripper_jaws, shared by the
    # anchored and RAW paths — the safety property must cover it too).
    for forbidden in ("_recorder", "_gripper_send", "self.ik.", "update_cfg", "_solver"):
        assert forbidden not in src, f"gripper-only update must not reference {forbidden!r}"
        assert forbidden not in jaw, f"jaw animation must not reference {forbidden!r}"
    # the jaw helper DOES read the live open fraction; the update still places frames
    # via the home-anchored mapping (anchored path) and calls the jaw helper.
    assert "gripper_frac" in jaw
    assert "home_anchor" in src
    assert "_animate_gripper_jaws" in src, "update must delegate the jaw LERP to the helper"


def _mat_to_wxyz(R):
    """3x3 rotation -> wxyz (scipy), for building random puck orientations in the tests."""
    from scipy.spatial.transform import Rotation as Rot
    q = Rot.from_matrix(R).as_quat()  # xyzw
    return np.array([q[3], q[0], q[1], q[2]])


def _child_world(fp, fq, lp, lq):
    """World pose of a child at local (lp, lq) under follow frame (fp, fq)."""
    import collection_viz as cv
    wp = np.asarray(fp, float) + cv._rot_vec_by_wxyz(fq, lp)
    wq = cv._quat_mul_wxyz(fq, lq)
    return wp, wq


def test_raw_follow_places_puck_triad_back_at_raw_pose():
    """RAW mode invariant #1: the 'tracker' (puck) child triad must render at the
    EXACT raw puck pose (position + orientation) for any puck orientation — that is
    the whole point of raw-frame mode (a faithful twin of the recorded tracker.npy)."""
    import collection_viz as cv
    from scipy.spatial.transform import Rotation as Rot

    puck_local_q = cv._puck_child_quat()              # M^-1, carried by the puck child in EVERY path
    rng = np.random.default_rng(7)
    for _ in range(200):
        p = rng.normal(size=3)
        R = Rot.random(random_state=rng).as_matrix()
        q = _mat_to_wxyz(R)
        fp, fq = cv.raw_follow_pose_from_puck(p, q)
        # puck child sits at the fixed wrist-relative _TRACKER_TRUE_XYZ_M, rotated M^-1
        pk_p, pk_q = _child_world(fp, fq, np.asarray(cv._TRACKER_TRUE_XYZ_M), puck_local_q)
        assert np.allclose(pk_p, p, atol=1e-9), f"puck pos {pk_p} != raw {p}"
        Rr = Rot.from_quat(np.asarray(pk_q)[[1, 2, 3, 0]]).as_matrix()
        ang = np.degrees(Rot.from_matrix(Rr @ R.T).magnitude())
        assert ang < 1e-6, f"puck orient off by {ang} deg"


def test_raw_follow_tcp_is_the_fused_remapped_tcp():
    """RAW mode invariant #2: the TCP triad position AND axes equal the fused
    remapped-TCP (puck ∘ [R=M | t=puck->TCP]), and wrist_yaw sits at TCP-0.242 along
    the TCP's own +x. Checked against an INDEPENDENT two-hop ground truth."""
    import collection_viz as cv
    from scipy.spatial.transform import Rotation as Rot

    M = cv._RAW_RELABEL_M
    tcp_back = cv._TCP_ALONG_WRIST_X_M
    # Independent puck-frame puck->TCP lever from the documented WRIST-frame child
    # offsets: (TCP - puck) in the wrist frame, rotated into the puck frame by M.
    # == (0, -0.225, 0.05) for the repo-canonical constants (1 mm from the 8088
    # scratch (0,-0.224,0.05), as disclosed).
    tcp_wrist = np.array([tcp_back, 0.0, 0.0])
    puck_wrist = np.asarray(cv._TRACKER_TRUE_XYZ_M, float)
    t_ext = M @ (tcp_wrist - puck_wrist)
    assert np.allclose(t_ext, [0.0, -0.225, 0.05], atol=1e-9), t_ext
    rng = np.random.default_rng(9)
    for _ in range(200):
        p = rng.normal(size=3)
        R = Rot.random(random_state=rng).as_matrix()
        q = _mat_to_wxyz(R)
        fp, fq = cv.raw_follow_pose_from_puck(p, q)
        # scene children: TCP triad at +0.242x (identity local), wrist_yaw at origin
        tcp_p, tcp_q = _child_world(fp, fq, [tcp_back, 0, 0], [1, 0, 0, 0])
        wy_p, _ = _child_world(fp, fq, [0, 0, 0], [1, 0, 0, 0])
        # ground truth: TCP = puck ∘ [R=M | t=t_ext] (the fused remapped-TCP)
        R_tcp = R @ M
        p_tcp = p + R @ t_ext
        assert np.allclose(tcp_p, p_tcp, atol=1e-9), f"TCP pos {tcp_p} != {p_tcp}"
        Rr = Rot.from_quat(np.asarray(tcp_q)[[1, 2, 3, 0]]).as_matrix()
        ang = np.degrees(Rot.from_matrix(Rr @ R_tcp.T).magnitude())
        assert ang < 1e-6, f"TCP axes off by {ang} deg (must be puck∘M)"
        # wrist_yaw = TCP - 0.242 along TCP +x
        p_wrist = p_tcp - R_tcp @ np.array([tcp_back, 0, 0])
        assert np.allclose(wy_p, p_wrist, atol=1e-9), f"wrist {wy_p} != TCP-0.242x {p_wrist}"


def test_raw_frame_wired_and_gated_to_gripper_mode():
    """Pin the wiring: --raw-frame reaches the scene, is GATED to gripper display mode
    (ignored in humanoid), and the launcher exposes RAW_FRAME as --raw-frame."""
    import pathlib

    viz = pathlib.Path(__file__).with_name("collection_viz.py").read_text()
    assert 'p.add_argument("--raw-frame"' in viz
    assert "raw_frame=getattr(args, \"raw_frame\", False)" in viz
    # gated: only meaningful in gripper mode
    assert 'bool(raw_frame) and (display_mode == "gripper")' in viz
    launcher = (pathlib.Path(__file__).parents[1] / "run_collection_viz.sh").read_text()
    assert 'RAW_FRAME="${RAW_FRAME:-0}"' in launcher
    assert "--raw-frame" in launcher


def _run() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  ok   {t.__name__}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  FAIL {t.__name__}: {type(exc).__name__}: {exc}")
    if failed:
        print(f"LOGIC FAIL  {failed}/{len(tests)} failed")
        return 1
    print(f"LOGIC OK  {len(tests)} passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(_run())
