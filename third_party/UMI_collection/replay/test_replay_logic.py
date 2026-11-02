#!/usr/bin/env python3
"""Pure-logic tests for the replay tools — the bits that don't need a browser.

Covers the Timeline frame-indexing + episode-retarget math (the trickiest shared
code) and the zarr episode slicing. Run in the collection .venv:

    .venv/bin/python replay/test_replay_logic.py        # prints "LOGIC OK"

We construct a real ``viser.ViserServer`` on a high, unused port to build the GUI
handles (no browser needed); the test exercises the pure index/seek math, not the
render loop. A real converted zarr is produced on the fly from the mock recorder
so the slicing test runs anywhere.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.join(_HERE, "..")
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_REPO, "processing"))
sys.path.insert(0, os.path.join(_REPO, "recording"))

_checks = 0


def ok(cond: bool, msg: str) -> None:
    global _checks
    assert cond, f"FAIL: {msg}"
    _checks += 1
    print(f"  ok  {msg}")


def test_timeline_index_and_retarget():
    import viser
    import replay_common as rc

    server = viser.ViserServer(host="127.0.0.1", port=8099, label="test")
    try:
        # Non-uniform timestamps (like a jittery ~15 Hz camera grid), absolute-ish.
        t = np.array([100.0, 100.07, 100.13, 100.21, 100.30, 100.40], dtype=np.float64)
        seen = {"i": []}
        tl = rc.Timeline(server, t, lambda i: seen["i"].append(i))

        ok(tl.K == 6, "K reflects frame count")
        ok(abs(tl.dur - 0.40) < 1e-9, "dur is last-first regardless of t0 offset")
        # _idx_for_t uses take-relative seconds; clamps both ends.
        ok(tl._idx_for_t(0.0) == 0, "t=0 -> frame 0")
        ok(tl._idx_for_t(0.40) == 5, "t=dur -> last frame")
        ok(tl._idx_for_t(-1.0) == 0, "negative clamps to 0")
        ok(tl._idx_for_t(100.0) == 5, "beyond end clamps to last")
        # A time between knots picks the frame at/just-before it (searchsorted right-1).
        ok(tl._idx_for_t(0.10) == 1, "0.10s falls in frame 1 (knot 0.07<=0.10<0.13)")
        ok(tl._idx_for_t(0.13) == 2, "exact knot 0.13 -> frame 2")

        # _step_frame moves one frame and pins the cursor to that knot.
        tl._t = 0.0
        tl._step_frame(+1)
        ok(tl._idx_for_t(tl._t) == 1, "step +1 from frame 0 -> frame 1")
        tl._step_frame(-1)
        ok(tl._idx_for_t(tl._t) == 0, "step -1 back -> frame 0")
        tl._step_frame(-1)
        ok(tl._idx_for_t(tl._t) == 0, "step -1 at start stays at 0 (clamp)")

        # set_series retargets to a new (shorter) episode, resets bounds + cursor.
        t2 = np.array([5.0, 5.05, 5.10], dtype=np.float64)
        tl.set_series(t2)
        ok(tl.K == 3, "set_series updates K")
        ok(abs(tl.dur - 0.10) < 1e-9, "set_series updates dur")
        ok(abs(tl.t_slider.max - 0.10) < 1e-6, "slider max retargeted to new dur")
        ok(tl._idx_for_t(0.0) == 0 and tl._idx_for_t(0.10) == 2,
           "indexing correct on the new series")
        ok(tl._t == 0.0 and tl._last_i == -1 and tl._force,
           "set_series resets cursor + forces re-render")

        # Single-frame series must not crash the math.
        tl.set_series(np.array([9.0], dtype=np.float64))
        ok(tl._idx_for_t(0.0) == 0, "single-frame series -> frame 0")
        ok(tl.dur == 0.0, "single-frame dur is 0")
    finally:
        server.stop()


def test_force_overlay_draws_the_target_and_the_demo_in_DIFFERENT_colours():
    """The funnel panel's two non-generated objects must be visually distinct.

    Both were green in the first cut — a line for the demo chunk, an arrow for the funnel's
    target — and that reads as one object, so the first person to open it asked which of the two
    Generate had produced. Under the default ``target_mode="anchor"`` they are not even the same
    kind of thing: the target is the single t0 wrench (== the viewer's own red ``F^cmd`` arrow),
    while the demo chunk wanders a median 86 deg away from it. Pinned here because a colour is
    exactly the sort of thing a later edit unifies for tidiness.

    Also pins the claim the panel makes in newtons: with the stored wrench passed in equal to the
    chunk's t0 row, ``target vs red arrow`` must read 0.000 N — i.e. the funnel really does aim at
    the vector the red arrow draws, no dataset needed to see it.
    """
    import viser

    import force_recovery_overlay as FRO

    if not FRO.ForceRecoveryOverlay(None, horizon=50, dt=1 / 30).available:
        print("  skip  FLoRe's flore/utils/force_funnel.py not importable here")
        return
    ok(FRO._DEMO_AMBER != FRO._TARGET_GREEN, "demo and target are different colours")
    ok(FRO._AUG_PURPLE not in (FRO._DEMO_AMBER, FRO._TARGET_GREEN),
       "neither is the generated purple")

    H = 50
    # A plausible contact chunk: force ramps to ~4 N, torque = r x F about a fixed lever, so the
    # window is admissible without depending on any dataset being present.
    ramp = np.linspace(0.0, 1.0, H)[:, None]
    demo = np.zeros((H, 6), dtype=np.float64)
    demo[:, :3] = ramp * np.array([4.0, -1.5, 0.8]) + np.array([1.6, -0.6, 0.3])
    demo[:, 3:] = np.cross(np.array([0.0, 0.0, 0.2]), demo[:, :3])

    server = viser.ViserServer(host="127.0.0.1", port=8097, label="test-force-aug")
    try:
        aug = FRO.ForceRecoveryOverlay(server, horizon=H, dt=1 / 30)
        aug.build_gui(server.gui)
        arrows: dict[str, tuple] = {}
        real_add_arrows = server.scene.add_arrows

        def spy(name, *a, **kw):                       # capture the colour, then draw for real
            if "colors" in kw:
                arrows[name] = tuple(np.asarray(kw["colors"]).ravel()[:3])
            return real_add_arrows(name, *a, **kw)

        server.scene.add_arrows = spy
        aug._request = True                            # what the Generate button's callback sets
        drew = aug.poll(ep=0, frame=10, demo_chunk=demo, anchor_pos=np.zeros(3),
                        anchor_wxyz=np.array([1.0, 0.0, 0.0, 0.0]), stored_wrench=demo[0])
        ok(drew, "poll() reports it drew something")
        ok(aug._batch is not None and aug._batch.n > 0,
           f"the synthetic contact window is admissible (n={getattr(aug._batch, 'n', 0)})")

        drawn = {h.name: tuple(np.asarray(h.colors).ravel()[:3]) for h in aug._lines
                 if hasattr(h, "colors")}
        ok(drawn.get("/force_aug/demo") == FRO._DEMO_AMBER,
           f"demo chunk drawn amber {FRO._DEMO_AMBER}")
        ok(drawn.get("/force_aug/target") == FRO._TARGET_GREEN,
           f"target dot drawn green {FRO._TARGET_GREEN}")
        ok(drawn.get("/force_aug/traj_0") == FRO._AUG_PURPLE, "trajectories drawn purple")

        aug.render_at(10)                              # k=0: paints both arrows
        ok(arrows.get("/force_aug/arrow") == FRO._AUG_PURPLE, "recovery arrow is purple")
        ok(arrows.get("/force_aug/target_arrow") == FRO._TARGET_GREEN, "target arrow is green")

        # The two claims the panel prints, checked in newtons rather than pixels.
        ok(aug.rd_target.value.startswith("0.000 N apart"),
           f"target IS the red arrow's vector ({aug.rd_target.value})")
        ok(np.allclose(aug._batch.chunks[:, -1, :], demo[0], atol=1e-9),
           "every trajectory ends on the t0 wrench")
    finally:
        server.stop()


def test_zarr_episode_slicing():
    import replay_zarr_viz as Z

    # Produce a small real zarr from the mock recorder (3 episodes).
    with tempfile.TemporaryDirectory() as td:
        rec_out = os.path.join(td, "rec")
        subprocess.run(
            [sys.executable, os.path.join(_REPO, "recording", "recorder.py"),
             "--mock", "--out", rec_out, "--mock-seconds", "4"],
            check=True, capture_output=True, cwd=_REPO)
        sess = [os.path.join(rec_out, x) for x in os.listdir(rec_out) if x.startswith("session_")][0]
        zpath = os.path.join(td, "m.zarr.zip")
        subprocess.run(
            [sys.executable, os.path.join(_REPO, "processing", "convert_to_umi.py"),
             sess, "-o", zpath], check=True, capture_output=True, cwd=_REPO)

        z = Z.ZarrEpisodes(zpath)
        ok(z.n_ep >= 2, f"mock zarr has >=2 episodes (got {z.n_ep})")
        lens = z.episode_lengths()
        ok(sum(lens) == int(z.ends[-1]), "episode lengths sum to total T (episode_ends[-1])")
        ok(int(z.starts[0]) == 0, "first episode starts at index 0")
        # slicing boundaries are contiguous and non-overlapping
        for e in range(z.n_ep):
            a, b = int(z.starts[e]), int(z.ends[e])
            pos = z.slice(e, "robot0_eef_pos")
            ts = z.slice(e, "episode_timestamps").reshape(-1)
            ok(pos.shape[0] == b - a, f"ep{e} slice length matches boundaries")
            ok(abs(float(ts[0])) < 1e-6, f"ep{e} timestamps relative (start at 0)")
            ok(np.all(np.diff(ts) > 0) if len(ts) > 1 else True,
               f"ep{e} timestamps strictly increasing")
        # rotvec->wxyz round-trips a known rotation
        wxyz = Z.rotvec_to_wxyz(np.array([[0.0, 0.0, 0.0]]))
        ok(np.allclose(wxyz[0], [1, 0, 0, 0]), "zero rotvec -> identity wxyz")


def test_lerobot_tracker_recovery():
    """The LeRobot replay drives the SAME mink IK as the raw replay by recovering
    the tracker world pose from the STORED TCP pose. That recovery
    (``recover_tracker_poses``) MUST be the EXACT inverse of the converter's
    ``apply_tcp_extrinsic`` — otherwise the reconstructed humanoid (and the
    wrist-mounted camera) would be driven from a wrong pose and the image would
    tilt, the very bug this re-adds the robot to fix. Verify to machine precision,
    including ROTATION coupling (a translation-only check would miss a transposed
    rotation block)."""
    import umi_convert_core as core
    import replay_lerobot_viz as R
    from scipy.spatial.transform import Rotation

    rng = np.random.default_rng(7)
    n = 40
    tr_pos = rng.standard_normal((n, 3))
    tr_rot = Rotation.from_rotvec(rng.standard_normal((n, 3)) * 0.6)

    for label, T in (("repo-default (+x 0.21, identity rot)",
                      np.array([[1, 0, 0, 0.21], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], float)),
                     ("identity (tcp == tracker)", np.eye(4)),
                     ("rotated+translated extrinsic", None)):
        if T is None:
            T = np.eye(4)
            T[:3, :3] = Rotation.from_rotvec([0.3, -0.2, 0.5]).as_matrix()
            T[:3, 3] = [0.21, 0.05, -0.03]
        # forward (converter): tracker -> tcp
        tcp_pos, tcp_rot = core.apply_tcp_extrinsic(tr_pos, tr_rot, T)
        # inverse (replay): tcp -> tracker
        rec_pos, rec_wxyz = R.recover_tracker_poses(tcp_pos, tcp_rot.as_rotvec(), T)
        rec_rot = Rotation.from_quat(np.atleast_2d(rec_wxyz)[:, [1, 2, 3, 0]])  # wxyz->xyzw
        pos_err = float(np.abs(rec_pos - tr_pos).max())
        rot_err = float((rec_rot * tr_rot.inv()).magnitude().max())
        ok(pos_err < 1e-9, f"recover pos exact inverse [{label}] (err {pos_err:.1e})")
        ok(rot_err < 1e-9, f"recover rot exact inverse [{label}] (err {rot_err:.1e})")

    # identity extrinsic: recovered tracker == stored tcp (tcp_calibrated=False path)
    tcp_pos = rng.standard_normal((5, 3))
    tcp_rotvec = rng.standard_normal((5, 3)) * 0.4
    rec_pos, _ = R.recover_tracker_poses(tcp_pos, tcp_rotvec, np.eye(4))
    ok(np.allclose(rec_pos, tcp_pos), "identity extrinsic -> tracker == tcp")


def test_lerobot_episode_loader():
    """The LeRobot replay's dataset loader (parquet + mp4 + meta) over a real
    converted dataset: episode count/lengths, the states->pos/rot/grip/ft slicing
    (via the writer's own slice map), and the mp4 frame count matching the parquet."""
    import convert_to_lerobot as cl
    import replay_lerobot_viz as R

    with tempfile.TemporaryDirectory() as td:
        rec_out = os.path.join(td, "rec")
        subprocess.run(
            [sys.executable, os.path.join(_REPO, "recording", "recorder.py"),
             "--mock", "--out", rec_out, "--mock-seconds", "4"],
            check=True, capture_output=True, cwd=_REPO)
        sess = [os.path.join(rec_out, x) for x in os.listdir(rec_out) if x.startswith("session_")][0]
        ds = os.path.join(td, "lr")
        subprocess.run(
            [sys.executable, os.path.join(_REPO, "processing", "convert_to_lerobot.py"),
             sess, "-o", ds, "--task", "mock"], check=True, capture_output=True, cwd=_REPO)

        z = R.LeRobotEpisodes(ds)
        ok(z.n_ep >= 2, f"mock lerobot dataset has >=2 episodes (got {z.n_ep})")
        d = z.load_episode(0)
        n = d["pos"].shape[0]
        ok(d["pos"].shape == (n, 3), "ep0 pos [T,3]")
        ok(d["rot"].shape == (n, 3), "ep0 rot [T,3]")
        ok(d["width"].shape == (n,), "ep0 gripper width [T]")
        ok(d["ft"] is not None and d["ft"].shape == (n, 6), "ep0 ft [T,6] from states slice")
        ok(d["ts"].shape == (n,), "ep0 timestamps [T]")
        ok(len(d["rgb"]) == 0 or abs(len(d["rgb"]) - n) <= 1, "ep0 mp4 frames ~ T")
        ok(z.episode_lengths()[0] == n, "episodes.jsonl length matches parquet rows")


def test_raw_replay_cluster_matches_converter_and_zarr():
    """The raw replay hangs the gripper cluster off the TCP (eef) pose so that the TCP
    triad lands ON the stored trajectory and the "tracker" triad lands back on the
    physical puck — the property that makes the raw and zarr replays render the SAME
    cluster. Pin it to machine precision against the converter's own
    ``apply_tcp_extrinsic`` (the forward map the wrist offset must invert), and assert
    the raw and zarr tools compute the identical wrist offset."""
    import umi_convert_core as core
    import replay_raw_viz as RR
    import replay_zarr_viz as RZ
    from collection_viz import _TCP_ALONG_WRIST_X_M, _TRACKER_TRUE_XYZ_M
    from scipy.spatial.transform import Rotation

    TRK = np.asarray(_TRACKER_TRUE_XYZ_M, dtype=np.float64)
    tcp_local = np.asarray([_TCP_ALONG_WRIST_X_M, 0.0, 0.0], dtype=np.float64)

    # The CURRENT repo extrinsic carries the measured -y,-x,-z puck->wrist relabel; the
    # legacy identity-rotation forms must still render correctly (self-describing tool).
    M = np.array([[0.0, -1, 0], [-1, 0, 0], [0, 0, -1]], float)
    remap_default = np.eye(4)
    remap_default[:3, :3] = M
    remap_default[:3, 3] = M @ (tcp_local - TRK)          # == (0,-0.225,0.05)
    legacy_translation = np.array([[1, 0, 0, 0.225], [0, 1, 0, 0], [0, 0, 1, -0.05], [0, 0, 0, 1]], float)
    for label, T in ((" current remap (-y,-x,-z, t=(0,-0.225,0.05))", remap_default),
                     ("legacy pure-translation (+0.225,0,-0.05)", legacy_translation),
                     ("identity (tcp == puck)", np.eye(4))):
        wo_raw, wq_raw = RR.wrist_pose_from_stored(T)
        wo_zarr, wq_zarr = RZ.wrist_pose_from_stored(T)
        ok(np.allclose(wo_raw, wo_zarr) and np.allclose(wq_raw, wq_zarr),
           f"raw & zarr wrist pose identical [{label}]")
        pq_raw = RR.puck_triad_local_quat(T)
        pq_zarr = RZ.puck_triad_local_quat(T)
        ok(np.allclose(pq_raw, pq_zarr), f"raw & zarr puck-triad rotation identical [{label}]")

        # The load-bearing UNIVERSAL invariant: raw puck -> TCP via the converter, then
        # the "tracker" triad hung under the wrist frame (offset AND rotation) off the TCP
        # pose must return to the puck — POSITION *AND* ORIENTATION, to machine precision.
        # NOTE (2026-08-06): this block previously only compared POSITIONS while its
        # comment claimed both. That vacuity let a real bug ship — the puck triad was
        # drawn with no local rotation, so it inherited the WRIST axes and rendered 180deg
        # off (identical to the TCP triad), which the operator caught by eye. The
        # orientation assertion below is the one that would have failed.
        rng = np.random.default_rng(3)
        p_puck = rng.standard_normal((6, 3))
        R_puck = Rotation.from_rotvec(rng.standard_normal((6, 3)) * 0.7)
        tcp_pos, tcp_rot = core.apply_tcp_extrinsic(p_puck, R_puck, T)
        Rm = tcp_rot.as_matrix()
        Rw_local = Rotation.from_quat(np.asarray(wq_raw)[[1, 2, 3, 0]]).as_matrix()
        wrist_p = tcp_pos + np.einsum("nij,j->ni", Rm, wo_raw)
        wrist_R = np.einsum("nij,jk->nik", Rm, Rw_local)
        trk_world = wrist_p + np.einsum("nij,j->ni", wrist_R, TRK)
        err = float(np.abs(trk_world - p_puck).max())
        ok(err < 1e-9, f"tracker triad POSITION lands on the puck [{label}] (err {err:.1e})")
        # ORIENTATION: the triad's local rotation must bring the wrist axes back to the
        # raw puck axes for EVERY sampled pose.
        Rp_local = Rotation.from_quat(np.asarray(pq_raw)[[1, 2, 3, 0]]).as_matrix()
        trk_R = np.einsum("nij,jk->nik", wrist_R, Rp_local)
        ang = np.degrees(Rotation.from_matrix(
            np.einsum("nij,nkj->nik", trk_R, R_puck.as_matrix())).magnitude()).max()
        ok(ang < 1e-6,
           f"tracker triad ORIENTATION == raw puck axes [{label}] (max {ang:.1e} deg)")
        # ...and for a ROTATING extrinsic the puck triad must NOT read the same as the TCP
        # triad (which carries the wrist/eef axes) — the exact symptom that was observed.
        tcp_vs_puck = np.degrees(Rotation.from_matrix(
            np.einsum("nij,nkj->nik", wrist_R, trk_R)).magnitude()).max()
        if not np.allclose(T[:3, :3], np.eye(3), atol=1e-9):
            ok(tcp_vs_puck > 1.0,
               f"rotating extrinsic: puck triad differs from the TCP/wrist triad "
               f"[{label}] ({tcp_vs_puck:.1f} deg apart)")

    # Current-extrinsic consistency: the TCP triad sits at the eef origin (eef-frame
    # offset 0) and so rides the stored trajectory. With the rotating extrinsic this is
    # the statement that the wrist sits exactly _TCP_ALONG_WRIST_X_M back along the eef's
    # own +x — the user-specified "wrist = TCP - 0.242 x" — which the math must produce
    # on its own, not by construction.
    wo_cur, wq_cur = RR.wrist_pose_from_stored(remap_default)
    ok(np.allclose(wq_cur, [1.0, 0.0, 0.0, 0.0], atol=1e-12),
       "current remap: stored eef shares the WRIST axes (eef->wrist rotation is identity)")
    ok(np.allclose(wo_cur, [-_TCP_ALONG_WRIST_X_M, 0.0, 0.0], atol=1e-12),
       f"current remap: wrist sits at TCP -{_TCP_ALONG_WRIST_X_M} along its own +x "
       f"(got {np.round(wo_cur, 6)})")
    Rw = Rotation.from_quat(np.asarray(wq_cur)[[1, 2, 3, 0]]).as_matrix()
    ok(np.allclose(wo_cur + Rw @ tcp_local, np.zeros(3), atol=1e-12),
       "current remap: TCP triad rides the stored trajectory (eef-frame offset 0)")


def test_raw_replay_tcp_marker_at_fingertip_grasp():
    """The operator wants the TCP marker + green trajectory at the FINGERTIP grasp point
    (the gap between the closed fingertips), not the calibrated eef (wrist +x
    ``_TCP_ALONG_WRIST_X_M``). Verify:
      (a) the mesh-derived grasp point is well forward of the eef and centred (y,z ~ 0);
      (b) the eef-frame grasp offset applied to BOTH the TCP triad (a wrist child) and the
          green line (per-sample p + R·grasp_off_eef) lands them on the SAME point as the
          fingertips — to machine precision, under arbitrary rotation (so line == triad ==
          grasp for every pose, not just the identity)."""
    import convert_to_umi as conv
    import gripper_only_view as gov
    import replay_raw_viz as RR
    from collection_viz import _TCP_ALONG_WRIST_X_M, _TRACKER_TRUE_XYZ_M
    from scipy.spatial.transform import Rotation

    geom = gov.build_gripper_geom("left", urdf_path=RR._DEFAULT_YAM)
    grasp = RR.fingertip_grasp_point_wrist(geom)  # wrist-frame grasp point
    # The grasp point is the CENTROID of the distal tip band, so it sits a few mm ahead of the
    # 0.242 m fingertip-midpoint datum (the pad is an ~18 mm long band, not a point) and well
    # inside the measured mesh span x in [0.1624, 0.2562]. The bound used to be
    # "> datum + 0.02" — only satisfiable while the datum was the wrong 0.21, which put the TCP
    # triad ~35 mm behind the pads. Assert AGREEMENT with the datum now, not a large gap.
    ok(_TCP_ALONG_WRIST_X_M < grasp[0] <= 0.2562,
       f"grasp point is on the pad, at/ahead of the TCP datum "
       f"({_TCP_ALONG_WRIST_X_M} < {grasp[0]:.4f} <= 0.2562)")
    ok(grasp[0] - _TCP_ALONG_WRIST_X_M < 0.02,
       f"TCP datum is within 20 mm of the mesh grasp point "
       f"({grasp[0] - _TCP_ALONG_WRIST_X_M:.4f} m) — it must land ON the fingertips")
    ok(abs(grasp[1]) < 0.01 and abs(grasp[2]) < 0.01,
       f"grasp point centred between the jaws (y={grasp[1]:.3f}, z={grasp[2]:.3f})")

    # Independently recompute the fingertip grasp point in the WRIST frame from the meshes
    # (mirror of the helper) to cross-check, then verify triad/line/fingertip coincide in
    # WORLD under a random eef pose.
    def _tip_centroid(pc):
        R = Rotation.from_quat(np.asarray(pc.quat)[[1, 2, 3, 0]]).as_matrix()
        v = (np.asarray(pc.mesh.vertices) @ R.T) + np.asarray(pc.pos_closed)
        return v[v[:, 0] > v[:, 0].max() - 0.01].mean(axis=0)
    gc_wrist = np.mean([_tip_centroid(geom.finger1), _tip_centroid(geom.finger2)], axis=0)
    ok(np.allclose(gc_wrist, grasp, atol=1e-9), "helper grasp point matches direct mesh computation")

    T_ext, _ = conv.load_tcp_extrinsic("/nonexistent")  # repo default
    wrist_off, wrist_q = RR.wrist_pose_from_stored(T_ext)
    R_w = Rotation.from_quat(np.asarray(wrist_q)[[1, 2, 3, 0]]).as_matrix()
    # grasp is in WRIST coords, so it must be rotated by the eef->wrist rotation before
    # being added to the eef-frame wrist offset (mirrors replay_raw_viz exactly).
    grasp_off_eef = wrist_off + R_w @ grasp
    # TCP triad world (child of /eef/wrist at grasp) and the shifted line point both equal
    # p_eef + R_eef @ grasp_off_eef. The fingertip world = p_eef + R_eef@(wrist_off +
    # R_w@gc_wrist) — the SAME vector — so all three coincide for ANY eef pose.
    rng = np.random.default_rng(21)
    max_err = 0.0
    for _ in range(8):
        p_eef = rng.standard_normal(3)
        R_eef = Rotation.from_rotvec(rng.standard_normal(3) * 0.8).as_matrix()
        triad_w = p_eef + R_eef @ grasp_off_eef
        line_w = p_eef + R_eef @ grasp_off_eef
        finger_w = p_eef + R_eef @ (wrist_off + R_w @ gc_wrist)
        max_err = max(max_err, float(np.abs(triad_w - finger_w).max()),
                      float(np.abs(line_w - finger_w).max()))
    ok(max_err < 1e-12,
       f"TCP triad == green line == fingertip grasp, any pose (max {max_err:.1e} m)")


def test_raw_replay_rebase_matches_gripper_only_view():
    """The raw replay's home-anchor (``_rebase`` ON) must be numerically identical to
    ``gripper_only_view.home_anchor`` (what the collection viz + zarr replay use), so the
    three tools draw the trajectory the same way; and OFF must be a pass-through of the
    raw poses (the strict-fidelity view)."""
    import gripper_only_view as gov
    import replay_raw_viz as RR
    from scipy.spatial.transform import Rotation

    rng = np.random.default_rng(11)
    n = 12
    pos = rng.standard_normal((n, 3))
    wxyz = Rotation.from_rotvec(rng.standard_normal((n, 3)) * 0.5).as_quat()[:, [3, 0, 1, 2]]
    home_p = np.asarray([0.35, 0.18, 0.95]); home_q = np.array([1.0, 0.0, 0.0, 0.0])

    # OFF -> raw pass-through.
    dp, dq = RR._rebase(pos, wxyz, home_p, home_q, on=False)
    ok(dp is pos and dq is wxyz, "_rebase OFF returns the raw arrays unchanged")

    # ON -> per-frame == gov.home_anchor with ref = frame 0.
    dp, dq = RR._rebase(pos, wxyz, home_p, home_q, on=True)
    ref_p, ref_q = pos[0], wxyz[0]
    max_p = max_q = 0.0
    for k in range(n):
        gp, gq = gov.home_anchor(home_p, home_q, ref_p, ref_q, pos[k], wxyz[k])
        max_p = max(max_p, float(np.abs(dp[k] - gp).max()))
        # quaternion sign-agnostic geodesic
        max_q = max(max_q, float(min(np.abs(dq[k] - gq).max(), np.abs(dq[k] + gq).max())))
    ok(max_p < 1e-12, f"_rebase ON pos == gov.home_anchor (max {max_p:.1e})")
    ok(max_q < 1e-12, f"_rebase ON quat == gov.home_anchor (max {max_q:.1e})")
    # At the reference frame the anchored pose IS the home pose.
    ok(np.allclose(dp[0], home_p), "_rebase ON: frame 0 sits at the home position")

    # ref override: the full-rate PATH anchors to an EXTERNAL reference (the camera-grid
    # frame 0), not its own frame 0, so it overlays the cluster. A sample AT that ref
    # pose must map exactly onto home; the same physical pose fed through either
    # reference lands identically (proves the overlay).
    ext_ref = (pos[3].copy(), wxyz[3].copy())
    dp2, _ = RR._rebase(pos, wxyz, home_p, home_q, on=True, ref=ext_ref)
    ok(np.allclose(dp2[3], home_p),
       "_rebase ON with ref override: the ref-matching sample sits at home")
    # A pose rebased with ref=X equals gov.home_anchor(...,ref=X,...) — independent of
    # which array it came from, so the path (ref=grid[0]) and cluster (ref=grid[0]) agree.
    gp3, _ = gov.home_anchor(home_p, home_q, ext_ref[0], ext_ref[1], pos[7], wxyz[7])
    ok(np.allclose(dp2[7], gp3), "_rebase ref override matches gov.home_anchor with that ref")


def test_raw_replay_tcp_equals_converter_eef():
    """END-TO-END: the raw replay's TCP grid (RawTake.tcp_pose_grid) MUST equal the
    converter's stored ``robot0_eef_pos`` for the same take — the whole point of the
    tool is to match the zarr replay. This pins the shared sanitize sequence
    (stale-drop -> strictly-increasing -> PRIMARY deglitch) + the extrinsic compose;
    an earlier build skipped the deglitch and drew a 1.5 cm-off path. Runs on a real
    converted-from-mock take so it exercises the actual pipeline, not a re-derivation."""
    import convert_to_umi as conv
    import replay_raw_viz as RR
    from scipy.spatial.transform import Rotation

    with tempfile.TemporaryDirectory() as td:
        rec_out = os.path.join(td, "rec")
        subprocess.run(
            [sys.executable, os.path.join(_REPO, "recording", "recorder.py"),
             "--mock", "--out", rec_out, "--mock-seconds", "4"],
            check=True, capture_output=True, cwd=_REPO)
        sess = [os.path.join(rec_out, x) for x in os.listdir(rec_out) if x.startswith("session_")][0]
        takes = sorted(x for x in os.listdir(sess) if x.startswith("take_")
                       and os.path.isdir(os.path.join(sess, x)))
        take_dir = os.path.join(sess, takes[0])

        take = RR.RawTake(take_dir, sess)
        T_ext, _ = conv.load_tcp_extrinsic(sess)
        tcp_pos, tcp_wxyz = take.tcp_pose_grid(T_ext)
        gcal = conv.load_gripper_calib(sess)
        eps = list(conv.build_episodes_from_take(
            take_dir, gcal, T_ext, out_res=(224, 224),
            jpeg_decode=lambda b: np.zeros((224, 224, 3), np.uint8),
            image_resize=lambda im, w, h: np.zeros((h, w, 3), np.uint8)))
        ok(len(eps) >= 1, f"converter emitted >=1 episode ({len(eps)})")
        # Concatenate the converter's per-episode eef back to a take-length grid: the
        # mock take is markerless (one episode) but stay robust if that changes.
        conv_pos = np.concatenate([np.asarray(e["robot0_eef_pos"]) for e in eps], axis=0)
        conv_rv = np.concatenate([np.asarray(e["robot0_eef_rot_axis_angle"]) for e in eps], axis=0)
        n = min(len(tcp_pos), len(conv_pos))
        ok(n > 0 and abs(len(tcp_pos) - len(conv_pos)) <= 1,
           f"raw-replay frames ({len(tcp_pos)}) ~= converter frames ({len(conv_pos)})")
        perr = float(np.abs(tcp_pos[:n] - conv_pos[:n]).max())
        raw_rv = Rotation.from_quat(tcp_wxyz[:n][:, [1, 2, 3, 0]]).as_rotvec()
        rerr = float((Rotation.from_rotvec(raw_rv)
                      * Rotation.from_rotvec(conv_rv[:n]).inv()).magnitude().max())
        # Tolerances are float wxyz<->xyzw round-trip noise, NOT the 1.5 cm a missing
        # deglitch (or wrong extrinsic) would produce.
        ok(perr < 1e-6, f"raw-replay TCP pos == converter robot0_eef_pos (err {perr:.1e} m)")
        ok(rerr < 1e-5, f"raw-replay TCP rot == converter eef rot (err {rerr:.1e} rad)")


def test_raw_replay_path_is_full_rate_and_overlays_grid():
    """The overview trajectory polyline is drawn from the RAW-RATE tracker (~80 Hz),
    NOT the camera-grid resample (~15 Hz), so it doesn't read straight between RGB
    frames — but it must still overlay the camera-grid path (same source, denser
    sampling). Assert: (a) raw path is strictly denser than the grid; (b) it stays
    inside the camera time span; (c) interpolated back onto the camera times it matches
    the grid path to sub-mm; (d) its arc length is >= the grid's (detail restored)."""
    import convert_to_umi as conv
    import replay_raw_viz as RR
    from scipy.interpolate import interp1d

    with tempfile.TemporaryDirectory() as td:
        rec_out = os.path.join(td, "rec")
        subprocess.run(
            [sys.executable, os.path.join(_REPO, "recording", "recorder.py"),
             "--mock", "--out", rec_out, "--mock-seconds", "4"],
            check=True, capture_output=True, cwd=_REPO)
        sess = [os.path.join(rec_out, x) for x in os.listdir(rec_out) if x.startswith("session_")][0]
        takes = sorted(x for x in os.listdir(sess) if x.startswith("take_")
                       and os.path.isdir(os.path.join(sess, x)))
        take = RR.RawTake(os.path.join(sess, takes[0]), sess)
        T_ext, _ = conv.load_tcp_extrinsic(sess)
        grid_pos, _ = take.tcp_pose_grid(T_ext)
        raw_pos, _ = take.tcp_pose_raw(T_ext)

        ok(raw_pos.shape[0] > grid_pos.shape[0],
           f"raw path denser than grid ({raw_pos.shape[0]} > {grid_pos.shape[0]})")
        ok(take.tr_raw_t[0] >= take.cam_t[0] - 1e-6 and take.tr_raw_t[-1] <= take.cam_t[-1] + 1e-6,
           "raw path clipped to the camera time span")
        # Overlay: interp raw path onto camera times, compare to the grid path.
        f = interp1d(take.tr_raw_t, raw_pos, axis=0, bounds_error=False,
                     fill_value="extrapolate")
        overlay_err = float(np.linalg.norm(f(take.cam_t) - grid_pos, axis=1).max())
        ok(overlay_err < 5e-3, f"raw path overlays the grid path (max {overlay_err*1000:.2f} mm)")
        # NOTE: no arc-length assertion here. On real jittery hardware the denser raw
        # path IS longer (it restores hand curves the ~15 Hz grid smooths — measured
        # 0.744 vs 0.704 m on session_2026.08.03), which is the whole point; but the
        # mock recorder emits a SMOOTH analytic tracker, on which the grid's interp1d
        # can tie or even marginally exceed the raw arc. Density + sub-mm overlay are
        # the invariants that hold on both, so the "less straight" win is asserted
        # there, not via a length compare that only separates on real noise.


def test_raw_replay_playback_grid_is_native_rate():
    """The MOVING cluster (not just the static path) plays on the tracker's NATIVE
    ~80 Hz grid, so the replayed hand doesn't STEP at the ~15 Hz camera rate (the
    operator's "不够 smooth"). Assert the whole display contract:
      (a) the playback grid ``disp_t`` is strictly denser than the camera grid and
          equals the native raw-tracker times (``disp_is_raw``);
      (b) the RGB zero-order-hold ``disp_cam_idx`` is monotone, in [0, K-1], starts at
          0, and hits every camera frame (so no RGB is skipped, just held);
      (c) ``tcp_pose_disp`` is a true SUPERSET of ``tcp_pose_grid`` — at the display
          samples that coincide with camera times it equals the fidelity-pinned grid
          pose to float round-trip (the denser grid restores motion, it does NOT invent
          a different trajectory);
      (d) the gripper-width + wrench display grids are length-D (aligned to the cluster),
          and the display-grid episode split covers the same span as the camera one."""
    import convert_to_umi as conv
    import replay_raw_viz as RR

    with tempfile.TemporaryDirectory() as td:
        rec_out = os.path.join(td, "rec")
        subprocess.run(
            [sys.executable, os.path.join(_REPO, "recording", "recorder.py"),
             "--mock", "--out", rec_out, "--mock-seconds", "4"],
            check=True, capture_output=True, cwd=_REPO)
        sess = [os.path.join(rec_out, x) for x in os.listdir(rec_out) if x.startswith("session_")][0]
        takes = sorted(x for x in os.listdir(sess) if x.startswith("take_")
                       and os.path.isdir(os.path.join(sess, x)))
        take = RR.RawTake(os.path.join(sess, takes[0]), sess)
        T_ext, _ = conv.load_tcp_extrinsic(sess)

        # (a) native-rate playback grid.
        ok(take.disp_is_raw, "display grid is the native tracker stream (not camera fallback)")
        ok(take.D > take.K, f"playback grid denser than camera ({take.D} > {take.K})")
        ok(np.array_equal(take.disp_t, take.tr_raw_t),
           "disp_t == native raw-tracker times")
        ok(take.disp_hz > 1.5 * (take.K / max(take.cam_t[-1], 1e-6)),
           f"playback ~{take.disp_hz:.0f} Hz >> camera ~{take.K/max(take.cam_t[-1],1e-6):.0f} Hz")

        # (b) RGB zero-order hold: valid, monotone, contiguous. The very last camera
        # frame may be held-out when the tracker stream ends a few ms BEFORE the last RGB
        # (the two streams don't stop at the same instant) — an honest boundary effect,
        # not a skip, so we assert contiguity up to ci.max() and allow the final frame to
        # be the only one dropped.
        ci = take.disp_cam_idx
        ok(ci.shape[0] == take.D, "disp_cam_idx has one entry per display frame")
        ok(ci[0] == 0, "disp_cam_idx starts on the first camera frame")
        ok(0 <= ci.min() and ci.max() <= take.K - 1, "disp_cam_idx stays in [0, K-1]")
        ok(ci.max() >= take.K - 2, "hold reaches the last (or 2nd-last) camera frame")
        ok(np.all(np.diff(ci) >= 0), "disp_cam_idx is monotone non-decreasing (a hold)")
        ok(set(ci.tolist()) == set(range(int(ci.max()) + 1)),
           "no INTERIOR camera frame is skipped (every frame 0..max is held)")

        # (c) superset fidelity: display pose == grid pose at coincident times.
        disp_pos, disp_wxyz = take.tcp_pose_disp(T_ext)
        grid_pos, grid_wxyz = take.tcp_pose_grid(T_ext)
        ok(disp_pos.shape[0] == take.D and grid_pos.shape[0] == take.K,
           "display pose length D, grid pose length K")
        # For each camera time, find the display sample sharing it (native grid contains
        # the camera-coincident instants only where they align; compare the nearest).
        from scipy.interpolate import interp1d
        f = interp1d(take.disp_t, disp_pos, axis=0, bounds_error=False, fill_value="extrapolate")
        superset_err = float(np.linalg.norm(f(take.cam_t) - grid_pos, axis=1).max())
        ok(superset_err < 5e-3,
           f"tcp_pose_disp interpolates onto the fidelity-pinned grid pose "
           f"(max {superset_err*1000:.2f} mm)")

        # (d) slower streams resampled to the cluster length; episode span preserved.
        ok(take.width_disp.shape[0] == take.D, "gripper width on the display grid")
        if take.has_ft:
            ok(take.wrench_disp(False).shape[0] == take.D, "wrench arrows on the display grid")
        ok(take.disp_segments[0]["start"] == 0
           and take.disp_segments[-1]["end"] == take.D,
           "display-grid episodes cover the whole playback grid")


def test_raw_replay_playback_grid_falls_back_without_tracker():
    """A take with NO usable primary tracker must fall back to the camera grid for
    playback (there is nothing denser to show) — disp_t == cam_t, disp_is_raw False,
    and the zero-order hold becomes the identity map. This keeps the tool working on a
    degenerate/camera-only take instead of crashing on an empty native grid."""
    import replay_raw_viz as RR

    with tempfile.TemporaryDirectory() as td:
        rec_out = os.path.join(td, "rec")
        subprocess.run(
            [sys.executable, os.path.join(_REPO, "recording", "recorder.py"),
             "--mock", "--out", rec_out, "--mock-seconds", "4"],
            check=True, capture_output=True, cwd=_REPO)
        sess = [os.path.join(rec_out, x) for x in os.listdir(rec_out) if x.startswith("session_")][0]
        takes = sorted(x for x in os.listdir(sess) if x.startswith("take_")
                       and os.path.isdir(os.path.join(sess, x)))
        take_dir = os.path.join(sess, takes[0])
        # Remove the primary tracker so has_tracker is False.
        import recording_format as fmt
        os.remove(os.path.join(take_dir, fmt.TRACKER_FILE))

        take = RR.RawTake(take_dir, sess)
        ok(not take.has_tracker, "no primary tracker after removal")
        ok(not take.disp_is_raw, "display grid falls back (disp_is_raw False)")
        ok(take.D == take.K and np.array_equal(take.disp_t, take.cam_t),
           "disp_t == cam_t on the fallback")
        ok(np.array_equal(take.disp_cam_idx, np.arange(take.K)),
           "fallback RGB hold is the identity map")
        ok(take.width_disp.shape[0] == take.D, "width still aligned to D on the fallback")


def _run_all() -> int:
    for fn in (test_timeline_index_and_retarget,
               test_force_overlay_draws_the_target_and_the_demo_in_DIFFERENT_colours,
               test_zarr_episode_slicing,
               test_raw_replay_cluster_matches_converter_and_zarr,
               test_raw_replay_tcp_marker_at_fingertip_grasp,
               test_raw_replay_rebase_matches_gripper_only_view,
               test_raw_replay_tcp_equals_converter_eef,
               test_raw_replay_path_is_full_rate_and_overlays_grid,
               test_raw_replay_playback_grid_is_native_rate,
               test_raw_replay_playback_grid_falls_back_without_tracker,
               test_lerobot_tracker_recovery, test_lerobot_episode_loader):
        print(f"- {fn.__name__}")
        fn()
    print(f"\nLOGIC OK ({_checks} checks)")
    return 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
