#!/usr/bin/env python
"""Eval gate for the data-collection viz (collection_viz.py).

Hardware-free: builds the viser server + scene FIRST (the bota_ft_viz lesson),
then drives synthetic tracker / gripper / camera frames through the SAME code
path the live app uses, asserting the load-bearing pieces actually work — not
just that imports succeed. The trap this guards against is a "green" run where
ViserUrdf silently fails to resolve meshes or the IK never moves the hand.

Checks:
  * ViserUrdf loads g1_29dof_yam.urdf (proves yourdfpy + all 45 mesh refs).
  * The HuMI mink IKSolver constructs and a calibrated PRIMARY-hand step MOVES the
    left_hand_site toward the tracker while the base does NOT drift. LEFT-canonical:
    the live 1pt viz drives ThreePointIK(active_roles=("left",)), so the smoke test
    exercises the SAME class + role-keyed dict API (not the legacy RightHandIK).
  * The 25->yam joint remap produces a vector matching the URDF actuated count,
    with the LEFT gripper finger1 set from the open fraction.
  * A camera JPEG decodes and is accepted by the frustum.
  * The encoder chart accepts a rolling update.
  * Gripper-only display (what DISPLAY_MODE=auto selects in EVERY tracking mode) for
    1pt / 2pt / 3pt: the yam meshes track the puck with no URDF and no mink; every hand's
    transform chain is diffed against a REAL MODE=1pt scene (the validated reference) so
    2pt/3pt cannot drift from it, while each hand keeps its own side's URDF gripper links;
    and 3pt's chest puck gets a bare home-anchored axis triad — no meshes and none of the
    hands' wrist/TCP lever.
  * MODE=3pt on the forced HUMANOID display, the only path exercising the torso task.

Run:  PYTHONPATH=third_party/humi_ik \
        .venv/bin/python viz/smoke_test_collection.py
Prints ``SMOKE OK ...`` on success, ``SMOKE FAIL ...`` (exit 1) otherwise.
"""
from __future__ import annotations

import io
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main() -> int:
    try:
        import viser  # noqa: F401
        from PIL import Image

        import collection_viz as cv
        from g1_right_hand_ik import GripperRange, YamConfigMapper
        from g1_three_point_ik import ThreePointIK

        # 1) gripper mapping invariants (no hardware, no HuMI).
        gr = GripperRange()
        assert abs(gr.jaw_m(gr.closed_rad)) < 1e-9, "closed -> 0 jaw"
        assert abs(gr.jaw_m(gr.open_rad) - gr.jaw_open_m) < 1e-9, "open -> full jaw"
        assert gr.fraction(-99) == 0.0 and gr.fraction(99) == 1.0, "fraction clamps"

        # 2) build the real scene (server + ViserUrdf + IK). LEFT-canonical: the
        # live 1pt viz uses ThreePointIK restricted to the primary ('left') role;
        # mirror that exactly (dict recalibrate/step, tracking_mode="1pt").
        import viser as _viser

        state = cv.CollectionState()
        ik = ThreePointIK(active_roles=("left",))
        server = _viser.ViserServer(host="127.0.0.1", port=8096, label="smoke")
        try:
            scene = cv.CollectionScene(
                server, state, ik=ik, mapper=None, yam_urdf=cv._DEFAULT_YAM,
                cam_offset_xyz=(0.09, 0.0, 0.03), cam_offset_rpy=(0.0, 0.44, 0.0), cam_fov_deg=58.0,
                tracking_mode="1pt",
            )
            mapper = YamConfigMapper(ik._humi_joint_names, scene._yam_names)
            scene.mapper = mapper
            scene._apply_home_pose()
            assert len(scene._yam_names) >= 29, f"too few yam joints: {len(scene._yam_names)}"

            # 3) calibrated primary(LEFT)-hand step moves the hand, base holds.
            home = ik.home_primary_hand.translation().copy()
            base0 = ik.home_q()[0:3].copy()
            ik.recalibrate({"left": (home, np.array([1.0, 0.0, 0.0, 0.0]))})
            bio = io.BytesIO()
            Image.fromarray((np.random.rand(240, 320, 3) * 255).astype("uint8"), "RGB").save(bio, "JPEG")
            for i in range(8):
                with state.lock:
                    state.tracker_pos = home + np.array([0.0, 0.0, 0.08 * i / 7])
                    state.tracker_quat = np.array([1.0, 0.0, 0.0, 0.0])
                    state.tracker_seq += 1
                    state.tracker_connected = True
                    state.tracker_stale = False
                    state.gripper_frac = i / 7.0
                    state.gripper_connected = True
                    state.gripper_seq += 1
                    state.history.append(time.perf_counter(), i / 7.0)
                    state.cam_jpeg = bio.getvalue()
                    state.cam_w, state.cam_h = 320, 240
                    state.cam_seq += 1
                    state.cam_connected = True
                scene.update()
                scene.update_chart()

            lh = ik._solver.configuration.get_transform_frame_to_world(
                frame_name="left_hand_site", frame_type="site"
            ).translation()
            base1 = ik._solver.configuration.q[0:3]
            assert lh[2] > home[2] + 0.04, f"left hand did not rise: {lh[2]:.3f} vs home {home[2]:.3f}"
            assert abs(base1[2] - base0[2]) < 0.03, f"base drifted: {base1[2]:.3f} vs {base0[2]:.3f}"

            # 4) remap shape + gripper injection. LEFT-canonical: the single physical
            # gripper is on the LEFT hand, so its fraction drives left_jaw_m.
            q = ik._solver.configuration.q.copy()
            cfg = mapper.to_yam(q, left_jaw_m=gr.jaw_m(4.90), right_jaw_m=0.0)
            assert len(cfg) == len(scene._yam_names), "cfg length mismatch"
            lidx = scene._yam_names.index("left_gripper_finger1_joint")
            assert abs(cfg[lidx] - 0.0475) < 1e-9, "gripper finger1 not set from fraction"

            print(
                f"SMOKE OK  yam_joints={len(scene._yam_names)} humi_joints={len(ik._humi_joint_names)} "
                f"LH z {home[2]:.3f}->{lh[2]:.3f} base_z {base0[2]:.3f}->{base1[2]:.3f} "
                f"gripper_finger1={cfg[lidx]:.4f}m"
            )
        finally:
            server.stop()

        # 5) gripper-only display mode (DISPLAY_MODE=gripper): the real scene with
        # NO URDF and NO mink IK — the yam gripper meshes + camera follow the tracker
        # via the home-anchored delta, and the LEFT jaws animate from the live open
        # fraction. Guards the trap where the gripper-only path silently draws nothing
        # (mesh extraction failed), leaves the cluster on the floor, or mis-parents a
        # hand. Both supported gripper configs are exercised: 1pt (left only) and 2pt
        # (left + right meshes under /tracker_right).
        import gripper_only_view as gov

        def _smoke_gripper(mode, roles, port):
            state2 = cv.CollectionState()
            server2 = _viser.ViserServer(host="127.0.0.1", port=port, label=f"smoke-gripper-{mode}")
            try:
                gscene = cv.CollectionScene(
                    server2, state2, ik=None, mapper=None, yam_urdf=cv._DEFAULT_YAM,
                    cam_offset_xyz=(0.10, 0.0, 0.04), cam_offset_rpy=(4.7124, 1.5708, 0.0),
                    cam_fov_deg=58.0, tracking_mode=mode, display_mode="gripper",
                )
                assert gscene._urdf is None and gscene.ik is None, "gripper mode built a URDF/IK"
                assert set(gscene._grip_nodes) == set(roles), (mode, gscene._grip_nodes)
                for role in roles:
                    assert set(gscene._grip_geom[role].__dict__) >= {"base", "finger1", "finger2"}
                hp = {r: gov.HOME_WRIST_POSE[r][0] for r in roles}
                for i in range(6):
                    with state2.lock:
                        state2.tracker_pos = hp["left"] + np.array([0.0, 0.0, 0.06 * i / 5])
                        state2.tracker_quat = np.array([1.0, 0.0, 0.0, 0.0])
                        state2.tracker_seq += 1
                        state2.tracker_connected = True
                        state2.tracker_stale = False
                        if "right" in roles:
                            state2.right_tracker_pos = hp["right"] + np.array([0.0, 0.03 * i / 5, 0.0])
                            state2.right_tracker_quat = np.array([1.0, 0.0, 0.0, 0.0])
                            state2.right_tracker_connected = True
                            state2.right_tracker_stale = False
                        state2.gripper_frac = i / 5.0
                        state2.gripper_connected = True
                        state2.gripper_seq += 1
                    gscene.update()
                # Problem 2A: the TRACKED pose is now bound to the "tracker" (puck) triad
                # (local _TRACKER_TRUE_XYZ_M under the follow frame), not the follow-frame
                # origin — so the follow frame itself is seated at an OFFSET from the
                # tracked pose. Reconstruct the puck triad's WORLD position (the pose the
                # operator sees the puck axis at) and assert THAT tracked the tracker's
                # +0.06 m upward motion. The tracker moved from hp["left"] to
                # hp["left"]+[0,0,0.06] with identity orientation, so at cur==ref->..->last
                # the puck world pose must equal that final tracked pose exactly (this is
                # the 2A invariant: puck triad world pose == home_anchor target).
                fp = np.asarray(gscene._tracker_frame.position)
                fq = np.asarray(gscene._tracker_frame.wxyz)
                puck_world = fp + cv._rot_vec_by_wxyz(fq, cv._TRACKER_TRUE_XYZ_M)
                follow_z = float(puck_world[2])
                assert follow_z > hp["left"][2] + 0.02, f"[{mode}] puck triad did not track: {follow_z:.3f}"
                expected_puck = np.asarray(hp["left"], dtype=float) + np.array([0.0, 0.0, 0.06])
                assert np.allclose(puck_world, expected_puck, atol=1e-6), (
                    f"[{mode}] puck triad not bound to tracked pose: {puck_world} vs {expected_puck}")
                # LEFT jaws open to full stroke (the single physical gripper is on left).
                geom = gscene._grip_geom["left"]
                f1 = np.asarray(gscene._grip_nodes["left"]["finger1"].position)
                assert np.allclose(f1, geom.finger1.pos_at(1.0), atol=1e-6), f"[{mode}] left jaw did not open"
                if "right" in roles:
                    # RIGHT meshes exist + are parented under /tracker_right (its follow
                    # frame moved), and its jaws HOLD closed (no 2nd encoder).
                    assert gscene._sec_frames["right"] is gscene._grip_follow["right"]
                    rf1 = np.asarray(gscene._grip_nodes["right"]["finger1"].position)
                    assert np.allclose(rf1, gscene._grip_geom["right"].finger1.pos_at(0.0), atol=1e-6), \
                        f"[{mode}] right jaw should hold closed"
                assert gscene._grip_calibrated, f"[{mode}] never auto-calibrated"
                print(f"SMOKE OK (gripper {mode})  roles={tuple(roles)}  follow_z {hp['left'][2]:.3f}->{follow_z:.3f}  "
                      f"no_urdf={gscene._urdf is None}")
            finally:
                server2.stop()

        _smoke_gripper("1pt", ("left",), 8097)
        _smoke_gripper("2pt", ("left", "right"), 8098)

        # 6) MODE=3pt on the HUMANOID display (DISPLAY_MODE=humanoid — NO LONGER what
        # 'auto' selects; see section 7). Still the only path that exercises the torso
        # FrameTask, so it stays covered: everything else here is 1pt-humanoid or
        # gripper-only, and a broken chest task would otherwise be invisible. Asserts the
        # third tracker really drives the torso, and that it does so in YAW (the HuMI
        # MJCF's waist chain has only waist_yaw_joint, so with the base pinned that is the
        # one controllable DoF — head pitch/roll are near-zero Jacobian and MUST NOT be
        # mistaken for a broken task).
        state3 = cv.CollectionState()
        ik3 = ThreePointIK(active_roles=("left", "right", "head"))
        server3 = _viser.ViserServer(host="127.0.0.1", port=8099, label="smoke-3pt")
        try:
            scene3 = cv.CollectionScene(
                server3, state3, ik=ik3, mapper=None, yam_urdf=cv._DEFAULT_YAM,
                cam_offset_xyz=(0.09, 0.0, 0.03), cam_offset_rpy=(0.0, 0.44, 0.0),
                cam_fov_deg=58.0, tracking_mode="3pt", display_mode="humanoid",
            )
            scene3.mapper = YamConfigMapper(ik3._humi_joint_names, scene3._yam_names)
            scene3._apply_home_pose()
            assert scene3._display_humanoid, "explicit display_mode='humanoid' must hold"
            assert set(scene3._sec_frames) == {"right", "head"}, \
                f"3pt needs BOTH secondary triads, got {sorted(scene3._sec_frames)}"

            ident = np.array([1.0, 0.0, 0.0, 0.0])
            hl = ik3.home_primary_hand.translation().copy()
            hr = ik3._solver.configuration.get_transform_frame_to_world(
                frame_name="right_hand_site", frame_type="site").translation().copy()
            ht = ik3._solver.configuration.get_transform_frame_to_world(
                frame_name="torso_link", frame_type="body").translation().copy()

            def _torso_yaw_deg():
                R = ik3._solver.configuration.get_transform_frame_to_world(
                    frame_name="torso_link", frame_type="body").rotation().as_matrix()
                return float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))

            def _feed(head_quat, n):
                for _ in range(n):
                    with state3.lock:
                        state3.tracker_pos, state3.tracker_quat = hl, ident
                        state3.tracker_connected, state3.tracker_stale = True, False
                        state3.tracker_seq += 1
                        state3.right_tracker_pos, state3.right_tracker_quat = hr, ident
                        state3.right_tracker_connected, state3.right_tracker_stale = True, False
                        _h3 = cv._body_tracker_slot(state3, "head")
                        _h3.update({"pos": ht, "quat": head_quat,
                                    "connected": True, "stale": False})
                    scene3.update()

            # Hold the NEUTRAL pose first. update() auto-calibrates once on the first
            # fully-live frame, so the reference must be captured at neutral — feeding the
            # rotated pose from frame 0 anchors the yaw INTO the reference and the delta is
            # then identity forever (the torso never moves, and it looks like a dead task).
            # This is the same reason the operator holds the calibration pose.
            _feed(ident, 3)
            yaw0 = _torso_yaw_deg()
            # Now rotate the HEAD tracker 25 deg about z; hands hold still at their anchors.
            # mink is an iterative velocity solver — the isolated IK needs ~30 steps to
            # converge on this, so a 10-frame loop would read as "barely moved".
            a = np.radians(25.0) / 2.0
            _feed(np.array([np.cos(a), 0.0, 0.0, np.sin(a)]), 40)
            yaw1 = _torso_yaw_deg()
            assert abs(yaw1 - yaw0) > 5.0, (
                f"the HEAD tracker did not rotate the torso: yaw {yaw0:.1f} -> {yaw1:.1f} deg "
                f"(the torso FrameTask is not being driven)")
            # Both hands must still be held at their anchors — the torso task must not
            # drag them off target (that is what the position/orientation cost split is for).
            lh3 = ik3._solver.configuration.get_transform_frame_to_world(
                frame_name="left_hand_site", frame_type="site").translation()
            rh3 = ik3._solver.configuration.get_transform_frame_to_world(
                frame_name="right_hand_site", frame_type="site").translation()
            for nm, got, want in (("left", lh3, hl), ("right", rh3, hr)):
                err = float(np.linalg.norm(got - want))
                assert err < 0.05, f"the torso task dragged the {nm} hand {err*100:.1f} cm off its anchor"
            print(f"SMOKE OK (3pt humanoid)  torso yaw {yaw0:.1f}->{yaw1:.1f} deg  "
                  f"hands held (L {np.linalg.norm(lh3-hl)*100:.1f} cm, "
                  f"R {np.linalg.norm(rh3-hr)*100:.1f} cm)  sec_frames={sorted(scene3._sec_frames)}")
        finally:
            server3.stop()

        # 7) MODE=3pt on what DISPLAY_MODE=auto NOW selects: gripper-only. The rig is two
        # isolated handheld grippers, so 3pt draws the SAME two clusters as 2pt plus ONE
        # extra object — a bare axis triad for the chest puck. The traps this guards:
        #   * the chest getting gripper MESHES (it has no gripper) or the hands' +0.242 m
        #     TCP lever / puck binding (it has no wrist), either of which draws the chest
        #     axis tens of cm off the pose the puck reports, self-consistently;
        #   * the chest frame still being written the RAW libsurvive pose by update()'s
        #     secondary-triad loop, which would fight the anchored placement each frame;
        #   * the right hand not being drawn identically to the left (the whole point) —
        #     checked by comparing the two hands' full child-node sets.
        state4 = cv.CollectionState()
        server4 = _viser.ViserServer(host="127.0.0.1", port=8100, label="smoke-3pt-gripper")
        try:
            assert cv.resolve_display_mode("auto", "3pt") == "gripper", \
                "3pt must default to gripper-only"
            scene4 = cv.CollectionScene(
                server4, state4, ik=None, mapper=None, yam_urdf=cv._DEFAULT_YAM,
                cam_offset_xyz=(0.10, 0.0, 0.04), cam_offset_rpy=(4.7124, 1.5708, 0.0),
                cam_fov_deg=58.0, tracking_mode="3pt",
                display_mode=cv.resolve_display_mode("auto", "3pt"),
                has_right_camera=True, has_right_gripper=True,
            )
            assert scene4._urdf is None and scene4.ik is None, "3pt auto built a URDF/IK"
            # meshes: hands only. follow frames: hands AND chest (it needs the anchor).
            assert set(scene4._grip_nodes) == {"left", "right"}, sorted(scene4._grip_nodes)
            assert set(scene4._grip_follow) == {"left", "right", "head"}, sorted(scene4._grip_follow)
            assert scene4._grip_follow["head"] is scene4._sec_frames["head"]
            # BOTH hands drawn identically: same DIRECT child node names under each
            # follow frame (the right hand used to get one unlabelled TCP triad and no
            # puck/wrist_yaw axes at all, on a frame seated by the puck binding).
            def _children(node):
                kids = server4.scene._children_from_node_name.get(node, ())
                return {n.rsplit("/", 1)[-1] for n in kids}
            lchild, rchild = _children("/tracker"), _children("/tracker_right")
            assert lchild == rchild, f"hands differ: left-only={lchild - rchild} right-only={rchild - lchild}"
            assert {"triad", "tracker_true", "wrist_yaw"} <= lchild, sorted(lchild)
            # The chest carries a triad and NOTHING else (no meshes, no camera).
            hchild = _children("/tracker_head")
            assert hchild == {"triad"}, sorted(hchild)

            # THE invariant: 3pt's RIGHT hand is a byte-for-byte copy of MODE=1pt's LEFT
            # hand (the validated reference display), and 3pt's own left hand is unchanged
            # from 1pt's. Compared as the full child subtree — every frame's/mesh's LOCAL
            # (position, wxyz) keyed by relative path — so this catches a divergence in ANY
            # of the offsets, not just the ones a hand-written assertion thought to list.
            # Checked against a REAL 1pt scene built here, not against restated constants.
            def _local_poses(scene_obj, root):
                h, kids = scene_obj._handle_from_node_name, scene_obj._children_from_node_name
                out, stack = {}, [root]
                while stack:
                    node = stack.pop()
                    for child in kids.get(node, ()):
                        stack.append(child)
                        if child.endswith("/label"):
                            continue    # cosmetic; a 1pt run has no 'camera L' label to match
                        hd = h[child]
                        out[child[len(root):]] = (np.asarray(hd.position, dtype=float).round(12),
                                                 np.asarray(hd.wxyz, dtype=float).round(12))
                return out

            server1 = _viser.ViserServer(host="127.0.0.1", port=8101, label="smoke-1pt-ref")
            try:
                ref = cv.CollectionScene(
                    server1, cv.CollectionState(), ik=None, mapper=None, yam_urdf=cv._DEFAULT_YAM,
                    cam_offset_xyz=(0.10, 0.0, 0.04), cam_offset_rpy=(4.7124, 1.5708, 0.0),
                    cam_fov_deg=58.0, tracking_mode="1pt",
                    display_mode=cv.resolve_display_mode("auto", "1pt"),
                )
                ref_left = _local_poses(server1.scene, "/tracker")
                assert ref_left, "1pt reference cluster is empty"
                assert ref._display_humanoid is False
            finally:
                server1.stop()

            def _diff(a, b, what, keys=None):
                ka, kb = set(a), set(b)
                assert ka == kb, (f"{what}: node sets differ — extra={sorted(kb - ka)} "
                                  f"missing={sorted(ka - kb)}")
                for k in (ka if keys is None else ka & keys):
                    assert np.allclose(a[k][0], b[k][0], atol=1e-12), (what, k, "pos", a[k][0], b[k][0])
                    assert np.allclose(a[k][1], b[k][1], atol=1e-12), (what, k, "wxyz", a[k][1], b[k][1])

            # The TRANSFORM CHAIN — the three triads and the camera mount — is what must be
            # copied verbatim: identical child paths, offsets and rotations (including the
            # puck triad's _puck_child_q, identity here and M^-1 only under RAW_FRAME=1).
            CHAIN = {"/triad", "/tracker_true", "/wrist_yaw", "/cam", "/cam/view"}
            left3 = _local_poses(server4.scene, "/tracker")
            right3 = _local_poses(server4.scene, "/tracker_right")
            assert CHAIN <= set(ref_left), sorted(ref_left)
            _diff(ref_left, left3, "3pt LEFT vs the 1pt reference")            # full subtree
            _diff(ref_left, right3, "3pt RIGHT vs the 1pt LEFT reference", keys=CHAIN)

            # The gripper MESHES are the exception, and deliberately so: the two yam
            # grippers are not translations of each other (the right-side URDF links are
            # the left's mirrored in y AND z with finger1/finger2 swapped), so each hand
            # must carry its OWN side's geometry. Assert exactly that — it is also the
            # cross-wire check for _link_local_mesh, which picks a mesh instance by scene
            # parent and would otherwise happily hand the right hand the left's links.
            for role, prefix in (("left", "/tracker"), ("right", "/tracker_right")):
                geom = gov.build_gripper_geom(role, urdf_path=cv._DEFAULT_YAM)
                assert scene4._grip_geom[role].side == role
                for piece in geom.pieces():
                    hd = server4.scene._handle_from_node_name[f"{prefix}/grip_{piece.name}"]
                    # atol on the quat is 1e-8, not 1e-12: `geom` is a SECOND URDF load, and
                    # _rotmat_to_wxyz's near-identity branch differs by ~2e-10 (≈2e-8 rad)
                    # between loads. Positions are bit-exact.
                    assert np.allclose(hd.position, piece.pos_closed, atol=1e-12), \
                        (role, piece.name, "pos", np.asarray(hd.position), piece.pos_closed)
                    assert np.allclose(hd.wxyz, piece.quat, atol=1e-8), \
                        (role, piece.name, "wxyz", np.asarray(hd.wxyz), piece.quat)
            assert not np.allclose(scene4._grip_geom["left"].base.pos_closed,
                                   scene4._grip_geom["right"].base.pos_closed), \
                "both hands got the SAME side's gripper links (cross-wired)"

            ident = np.array([1.0, 0.0, 0.0, 0.0])
            # The VIZ's homes, not gripper_only_view's dict: the chest is deliberately NOT
            # that dict's entry (which is the standing operator's torso, the right anchor for
            # re-basing a RECORDED take and for motion completion) but the G1's own pelvis, so
            # 'c' snaps the rig back to the robot's home arrangement. Asserting the difference
            # here is what stops the next reader "fixing" one into the other.
            hp = {r: cv.display_home_pose(r)[0] for r in ("left", "right", "head")}
            for r in ("left", "right"):
                assert np.allclose(hp[r], gov.HOME_WRIST_POSE[r][0], atol=1e-12), r
            assert not np.allclose(hp["head"], gov.HOME_WRIST_POSE["head"][0], atol=1e-3), \
                "the chest home stopped being the viz's own pelvis spot"

            def _feed4(dl, dr, dh, n=3):
                for _ in range(n):
                    with state4.lock:
                        state4.tracker_pos, state4.tracker_quat = hp["left"] + dl, ident
                        state4.tracker_connected, state4.tracker_stale = True, False
                        state4.tracker_seq += 1
                        state4.right_tracker_pos, state4.right_tracker_quat = hp["right"] + dr, ident
                        state4.right_tracker_connected, state4.right_tracker_stale = True, False
                        _h4 = cv._body_tracker_slot(state4, "head")
                        _h4.update({"pos": hp["head"] + dh, "quat": ident,
                                    "connected": True, "stale": False})
                    scene4.update()

            z = np.zeros(3)
            _feed4(z, z, z)                     # auto-calibrates at the reference pose
            assert scene4._grip_calibrated, "3pt gripper mode never auto-calibrated"
            assert set(scene4._grip_ref) == {"left", "right", "head"}, sorted(scene4._grip_ref)
            # At cur == ref the chest triad sits EXACTLY on its home (identity binding).
            # A hand's frame is offset by the puck lever instead, and its puck CHILD is
            # what lands on home — assert both, so a copy-paste of either binding fails.
            # .copy() throughout, and it is not defensive: viser's position setter writes
            # `self._impl.position[:] = ...` IN PLACE, so np.asarray() aliases the live array
            # and a name read now would silently report a LATER frame's pose.
            head_p = np.asarray(scene4._sec_frames["head"].position).copy()
            assert np.allclose(head_p, hp["head"], atol=1e-9), (head_p, hp["head"])
            for role, frame in (("left", scene4._tracker_frame),
                                ("right", scene4._sec_frames["right"])):
                fp, fq = np.asarray(frame.position), np.asarray(frame.wxyz)
                puck = fp + cv._rot_vec_by_wxyz(fq, cv._TRACKER_TRUE_XYZ_M)
                assert np.allclose(puck, hp[role], atol=1e-9), (role, puck, hp[role])
                assert not np.allclose(fp, hp[role]), f"{role} lost the puck lever"
            # Now move ONLY the chest puck: its triad follows 1:1 and the hands hold.
            dh = np.array([0.0, 0.0, 0.07])
            _feed4(z, z, dh, n=3)
            head_p2 = np.asarray(scene4._sec_frames["head"].position).copy()
            assert np.allclose(head_p2, hp["head"] + dh, atol=1e-9), (head_p2, hp["head"] + dh)
            lp2 = np.asarray(scene4._tracker_frame.position).copy()
            puck_l = lp2 + cv._rot_vec_by_wxyz(np.asarray(scene4._tracker_frame.wxyz),
                                               cv._TRACKER_TRUE_XYZ_M)
            assert np.allclose(puck_l, hp["left"], atol=1e-9), "chest motion dragged the left hand"

            # 'c' / "Reset robot to home": ALL THREE roles snap back, the chest included.
            # The chest is the one that used to be an afterthought — it carries no gripper,
            # so a reset loop written for the hands can silently skip it and leave the triad
            # 7 cm up wherever the operator's chest happened to be. Then re-feed at the SAME
            # displaced pose: the reset re-arms auto-cal, so cur becomes the new ref and every
            # role must still read its home — a reset that jumps on the next frame is not one.
            with state4.lock:
                state4.reset_request = True
            scene4.update()
            for role, want in (("head", hp["head"]), ("left", hp["left"]), ("right", hp["right"])):
                frame = (scene4._tracker_frame if role == "left"
                         else scene4._sec_frames[role])
                got = np.asarray(frame.position).copy()
                if role != "head":                     # a hand's frame carries the puck lever
                    got = got + cv._rot_vec_by_wxyz(np.asarray(frame.wxyz),
                                                    cv._TRACKER_TRUE_XYZ_M)
                assert np.allclose(got, want, atol=1e-9), (f"reset missed {role}", got, want)
            # The re-arm is CONSUMED in the same update(): dropping the anchors leaves
            # auto-cal armed, and every role is live, so the reference is re-captured at the
            # current (displaced) pose before this same frame draws. That is what puts the
            # frames back on home — assert the new reference, not a transient flag.
            assert scene4._grip_calibrated, "reset left the gripper view uncalibrated"
            assert np.allclose(scene4._grip_ref["head"][0], hp["head"] + dh, atol=1e-9), \
                ("the reset did not re-anchor the chest to the live pose",
                 scene4._grip_ref["head"][0])
            _feed4(z, z, dh, n=2)
            head_p3 = np.asarray(scene4._sec_frames["head"].position).copy()
            assert np.allclose(head_p3, hp["head"], atol=1e-9), \
                ("the chest jumped off home right after the reset", head_p3)

            print(f"SMOKE OK (3pt gripper)  meshes={sorted(scene4._grip_nodes)}  "
                  f"chest triad {hp['head'][2]:.3f}->{head_p2[2]:.3f} (home-anchored, no lever)"
                  f" -> {head_p3[2]:.3f} after 'c'  "
                  f"both hands == the 1pt LEFT reference ({len(lchild)} children each, "
                  f"chain {len(CHAIN)} nodes bit-equal, meshes per-side)")
        finally:
            server4.stop()

        # 8) MODE=5pt: 3pt plus TWO MORE BODY REFERENCES (the feet), which must be the
        # chest's code paths and not a third shape. The traps this guards are the ones a
        # "copy the head" change actually hits:
        #   * a foot getting gripper MESHES or the hands' +0.242 m TCP lever / puck binding
        #     (it has neither a gripper nor a wrist), which draws it 24 cm out, consistently
        #     enough to look deliberate;
        #   * a foot falling through to the LEFT WRIST home (every consumer looks a home up
        #     as `.get(role, ...["left"])`), i.e. rendered inside the left hand;
        #   * the two feet SHARING one home or one state slot, which hides a left/right swap;
        #   * 'c' resetting the hands and chest and skipping the feet — the exact
        #     afterthought the chest itself used to be.
        # The 5pt scene must be the 3pt scene PLUS two triads: asserted by comparing the
        # three shared roles' node sets against scene4's.
        state5 = cv.CollectionState()
        server5 = _viser.ViserServer(host="127.0.0.1", port=8101, label="smoke-5pt-gripper")
        try:
            assert cv.resolve_display_mode("auto", "5pt") == "gripper"
            scene5 = cv.CollectionScene(
                server5, state5, ik=None, mapper=None, yam_urdf=cv._DEFAULT_YAM,
                cam_offset_xyz=(0.10, 0.0, 0.04), cam_offset_rpy=(4.7124, 1.5708, 0.0),
                cam_fov_deg=58.0, tracking_mode="5pt",
                display_mode=cv.resolve_display_mode("auto", "5pt"),
                has_right_camera=True, has_right_gripper=True,
            )
            assert scene5._urdf is None and scene5.ik is None, "5pt built a URDF/IK"
            assert scene5._active_roles == ("left", "right", "head", "left_foot", "right_foot")
            assert scene5._body_roles == ("head", "left_foot", "right_foot")
            # MESHES: the two hands only. FOLLOW FRAMES: all five (a body reference needs
            # the same anchor or it renders at libsurvive's raw world pose).
            assert set(scene5._grip_nodes) == {"left", "right"}, sorted(scene5._grip_nodes)
            assert set(scene5._grip_follow) == set(scene5._active_roles), sorted(scene5._grip_follow)
            for r in ("head", "left_foot", "right_foot"):
                assert scene5._grip_follow[r] is scene5._sec_frames[r], r

            def _children5(node):
                kids = server5.scene._children_from_node_name.get(node, ())
                return {n.rsplit("/", 1)[-1] for n in kids}

            # Each foot carries a triad and NOTHING else — same as the chest, byte for byte.
            for r in ("head", "left_foot", "right_foot"):
                assert _children5(f"/tracker_{r}") == {"triad"}, (r, sorted(_children5(f"/tracker_{r}")))
            # ...and the two HANDS are untouched by 5pt (identical child sets to 3pt's).
            assert _children5("/tracker") == lchild, "5pt changed the LEFT hand"
            assert _children5("/tracker_right") == rchild, "5pt changed the RIGHT hand"
            # A status row per body reference, so a dark foot puck is visible on the page.
            assert set(scene5.gui_sec_tracker) == {"right", "head", "left_foot", "right_foot"}, \
                sorted(scene5.gui_sec_tracker)

            # Homes: five DISTINCT spots, the feet from the URDF's ankles (not the shared
            # display dict, and not the left hand's).
            hp5 = {r: cv.display_home_pose(r)[0] for r in scene5._active_roles}
            for a in range(len(scene5._active_roles)):
                for b in range(a + 1, len(scene5._active_roles)):
                    ra, rb = scene5._active_roles[a], scene5._active_roles[b]
                    assert not np.allclose(hp5[ra], hp5[rb], atol=1e-6), (ra, rb, hp5[ra])
            assert hp5["left_foot"][2] < hp5["head"][2] - 0.5, hp5
            for r in ("left_foot", "right_foot"):
                assert not np.allclose(hp5[r], gov.HOME_WRIST_POSE[r][0], atol=1e-3), \
                    f"{r} fell back to the SHARED display home"

            def _feed5(deltas, n=3):
                for _ in range(n):
                    with state5.lock:
                        state5.tracker_pos = hp5["left"] + deltas.get("left", z)
                        state5.tracker_quat = ident
                        state5.tracker_connected, state5.tracker_stale = True, False
                        state5.tracker_seq += 1
                        state5.right_tracker_pos = hp5["right"] + deltas.get("right", z)
                        state5.right_tracker_quat = ident
                        state5.right_tracker_connected = True
                        state5.right_tracker_stale = False
                        for _r in ("head", "left_foot", "right_foot"):
                            _sl = cv._body_tracker_slot(state5, _r)
                            _sl.update({"pos": hp5[_r] + deltas.get(_r, z), "quat": ident,
                                        "connected": True, "stale": False})
                    scene5.update()

            _feed5({})
            assert scene5._grip_calibrated, "5pt gripper mode never auto-calibrated"
            assert set(scene5._grip_ref) == set(scene5._active_roles), sorted(scene5._grip_ref)
            # At cur == ref every BODY triad sits EXACTLY on its own home (identity binding).
            for r in ("head", "left_foot", "right_foot"):
                got = np.asarray(scene5._sec_frames[r].position).copy()
                assert np.allclose(got, hp5[r], atol=1e-9), (r, got, hp5[r])

            # Move ONLY the LEFT foot: it follows 1:1 and NOTHING else moves — the check
            # that the five roles have five independent state slots and five references.
            dlf = np.array([0.0, 0.0, 0.06])
            _feed5({"left_foot": dlf}, n=3)
            lf_p = np.asarray(scene5._sec_frames["left_foot"].position).copy()
            assert np.allclose(lf_p, hp5["left_foot"] + dlf, atol=1e-9), (lf_p, hp5["left_foot"] + dlf)
            for r in ("head", "right_foot"):
                got = np.asarray(scene5._sec_frames[r].position).copy()
                assert np.allclose(got, hp5[r], atol=1e-9), (f"left-foot motion moved {r}", got)
            puck_l5 = (np.asarray(scene5._tracker_frame.position)
                       + cv._rot_vec_by_wxyz(np.asarray(scene5._tracker_frame.wxyz),
                                             cv._TRACKER_TRUE_XYZ_M))
            assert np.allclose(puck_l5, hp5["left"], atol=1e-9), "foot motion dragged the left hand"

            # 'c' snaps ALL FIVE back, feet included.
            with state5.lock:
                state5.reset_request = True
            scene5.update()
            for r in scene5._active_roles:
                frame = scene5._tracker_frame if r == "left" else scene5._sec_frames[r]
                got = np.asarray(frame.position).copy()
                if r in ("left", "right"):
                    got = got + cv._rot_vec_by_wxyz(np.asarray(frame.wxyz), cv._TRACKER_TRUE_XYZ_M)
                assert np.allclose(got, hp5[r], atol=1e-9), (f"reset missed {r}", got, hp5[r])
            print(f"SMOKE OK (5pt gripper)  roles={scene5._active_roles}  "
                  f"body={scene5._body_roles}  meshes={sorted(scene5._grip_nodes)}  "
                  f"L-foot triad {hp5['left_foot'][2]:.3f}->{lf_p[2]:.3f} (home-anchored, no "
                  f"lever), chest+R-foot held, hands byte-identical to 3pt")
        finally:
            server5.stop()
        return 0
    except Exception as exc:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        print(f"SMOKE FAIL  {type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
