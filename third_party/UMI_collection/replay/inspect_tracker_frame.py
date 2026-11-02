#!/usr/bin/env python3
"""Bare-bones RAW tracker coordinate-frame inspector (viser).

PURPOSE: check the tracker's OWN coordinate system, nothing else. Draws ONLY the
raw ``tracker.npy`` — the physical VIVE puck pose in libsurvive world, EXACTLY as
recorded:
  * NO ground grid (free-drag/rotate the scene to inspect from any angle),
  * NO TCP extrinsic, NO home_anchor, NO re-root — pure raw world pose,
  * the full trajectory as a polyline at the tracker positions,
  * a world-origin triad (reference) + per-sample tracker triads every --stride
    frames so you can SEE how the puck frame rotates along the path,
  * frame-0 (green) and last-frame (red) triads highlighted.

Columns are the authoritative recording_format.TRACKER_COLS:
    t, seq, x,y,z, qw,qx,qy,qz, stale     (quaternion is WXYZ, libsurvive)

Usage:
    inspect_tracker_frame.py data/recordings/session_XXX [--take take_000]
                             [--stride 40] [--triad-len 0.05] [--port 8088]
"""
from __future__ import annotations
import argparse
import json
import os
import sys

import numpy as np
import viser
import viser.transforms as vtf
from scipy.spatial.transform import Rotation as Rot

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, os.pardir, "recording"))
import recording_format as fmt  # noqa: E402  (authoritative TRACKER_COLS)


def load_raw_tracker(take_dir: str):
    """Return (t, pos[N,3], quat_wxyz[N,4], stale[N]) straight from tracker.npy."""
    tr = np.load(os.path.join(take_dir, fmt.TRACKER_FILE))
    assert tr.shape[1] == len(fmt.TRACKER_COLS), (
        f"tracker.npy has {tr.shape[1]} cols, expected {len(fmt.TRACKER_COLS)} "
        f"({fmt.TRACKER_COLS})")
    t = tr[:, 0]
    pos = tr[:, 2:5]                 # x,y,z
    quat_wxyz = tr[:, 5:9]           # qw,qx,qy,qz  (WXYZ)
    stale = tr[:, 9]
    return t, pos, quat_wxyz, stale


def find_take(path: str, take: str | None) -> str:
    path = os.path.abspath(path)
    if os.path.basename(path).startswith("take_"):
        return path
    if take:
        return os.path.join(path, take)
    takes = sorted(
        os.path.join(path, x) for x in os.listdir(path)
        if x.startswith("take_") and not x.endswith(".tmp")
        and os.path.exists(os.path.join(path, x, fmt.TAKE_META_FILE)))
    if not takes:
        raise SystemExit(f"[inspect] no committed take_* in {path}")
    return takes[0]


def add_triad(server, name, p, R: Rot, length: float, radius: float):
    """Draw an XYZ axis triad at world pose (p, R) using three colored arrows.

    Returns the three ArrowsHandle so a caller can toggle their visibility."""
    M = R.as_matrix()
    colors = [(230, 40, 40), (40, 200, 40), (40, 90, 230)]  # x=red y=green z=blue
    handles = []
    for ax in range(3):
        end = p + M[:, ax] * length
        handles.append(server.scene.add_arrows(
            f"{name}/ax{ax}", points=np.asarray([[p, end]], dtype=np.float32),
            colors=colors[ax], shaft_radius=radius, head_radius=radius * 2.5,
            head_length=length * 0.25))
    return handles


def _rotation_from_axis_map(tokens) -> Rot:
    """Build a proper rotation from an axis remap like ['-y','-x','-z'].

    Each token names which CURRENT axis the NEW x/y/z should point along, so the
    matrix has those signed unit vectors as its COLUMNS. Rejects a reflection
    (det=-1): only a proper rotation is a legal fixed re-orientation."""
    basis = {"x": np.array([1.0, 0, 0]), "y": np.array([0, 1.0, 0]), "z": np.array([0, 0, 1.0])}
    cols = []
    for tok in tokens:
        t = tok.strip().lower()
        sign = -1.0 if t[0] == "-" else 1.0
        key = t[-1]
        if key not in basis or t.lstrip("+-") not in basis:
            raise SystemExit(f"[inspect] bad --axis-map token {tok!r} (want +x/-x/+y/-y/+z/-z)")
        cols.append(sign * basis[key])
    M = np.column_stack(cols)
    if not np.allclose(M @ M.T, np.eye(3)):
        raise SystemExit(f"[inspect] --axis-map {tokens} is not orthonormal (repeated axis?)")
    if not np.isclose(np.linalg.det(M), 1.0):
        raise SystemExit(f"[inspect] --axis-map {tokens} has det={np.linalg.det(M):+.0f} "
                         f"(a reflection, not a rotation) — flip one sign.")
    return Rot.from_matrix(M)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("path", help="session dir (uses take_000) or a take_NNN dir")
    ap.add_argument("--take", default=None)
    ap.add_argument("--stride", type=int, default=40,
                    help="draw a tracker triad every N samples (default 40 ~= 0.5s)")
    ap.add_argument("--triad-len", type=float, default=0.05, help="axis length (m)")
    ap.add_argument("--align-first", action=argparse.BooleanOptionalAction, default=True,
                    help="apply ONE fixed rotation R0^-1 so frame-0 puck is upright at the "
                         "origin (canonical x-fwd/y-left/z-up); --no-align-first shows the "
                         "raw libsurvive Z-up world")
    ap.add_argument("--extra-rpy", type=float, nargs=3, default=None, metavar=("R", "P", "Y"),
                    help="extra fixed rotation (deg, intrinsic xyz) applied AFTER align-first "
                         "— spin forward/left to taste without touching the data pipeline")
    ap.add_argument("--axis-map", default=None, metavar="NEWX,NEWY,NEWZ",
                    help="extra fixed rotation given as an axis REMAP: a COMMA-separated "
                         "triple naming which CURRENT axis the new x/y/z point along, each "
                         "one of {+x,-x,+y,-y,+z,-z}. Use '=' so the leading '-' isn't parsed "
                         "as a flag, e.g. --axis-map=-y,-x,-z. Applied AFTER align-first; "
                         "rejected unless it is a proper rotation (det=+1).")
    ap.add_argument("--tcp", action=argparse.BooleanOptionalAction, default=True,
                    help="overlay the TCP trajectory + triads, derived from the puck via the "
                         "calibrated tracker->TCP extrinsic (world_T_tcp = world_T_puck @ "
                         "tracker_T_tcp), then the SAME fixed rotation as the tracker. "
                         "--no-tcp draws the tracker only.")
    ap.add_argument("--tcp-extrinsic",
                    default=os.path.join(_HERE, os.pardir, "robot_assets", "calibration",
                                         "tcp_extrinsic.json"),
                    help="tracker->TCP extrinsic json (T_tracker_tcp 4x4). Default = repo calib.")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8088)
    args = ap.parse_args(argv)

    take_dir = find_take(args.path, args.take)
    t, pos, quat_wxyz, stale = load_raw_tracker(take_dir)
    live = stale < 0.5
    R = Rot.from_quat(quat_wxyz[:, [1, 2, 3, 0]])  # wxyz -> xyzw
    N = pos.shape[0]

    # ONE fixed world rotation R_fix (NOT home_anchor, NOT per-frame): apply the SAME
    # rotation to every position and orientation, so the whole trajectory rotates
    # rigidly — shape unchanged. This is just a display re-orientation of the raw data.
    #   --align-first (default): R_fix = R0^-1, so frame-0 puck becomes identity at the
    #     origin, giving a canonical upright x-fwd/y-left/z-up start.
    #   --extra-rpy R P Y: an additional constant spin applied after, to taste.
    R_fix = Rot.identity()
    p_shift = np.zeros(3)
    if args.align_first:
        R_fix = R[0].inv()
        p_shift = pos[0].copy()          # translate frame-0 to the origin too
    if args.extra_rpy is not None:
        R_fix = Rot.from_euler("xyz", args.extra_rpy, degrees=True) * R_fix
    if args.axis_map is not None:
        toks = [t for t in args.axis_map.replace(" ", ",").split(",") if t]
        if len(toks) != 3:
            raise SystemExit(f"[inspect] --axis-map wants 3 comma-separated tokens, got {toks}")
        R_fix = _rotation_from_axis_map(toks) * R_fix

    # TCP pose (before the display R_fix): compose the calibrated extrinsic in the puck
    # BODY frame — world_T_tcp = world_T_puck @ tracker_T_tcp — the SAME math the
    # converter uses. Rotation couples into translation here, which is why we must use
    # the puck ORIENTATION (not just add a world offset). Then the SAME fixed rotation /
    # shift as the tracker is applied, so tracker & TCP are compared in one frame.
    tcp_pos = tcp_R = None
    tcp_remap_pos = tcp_remap_R = None
    T_fused = None
    if args.tcp:
        T_ext = np.asarray(json.load(open(args.tcp_extrinsic))["T_tracker_tcp"], dtype=np.float64)
        R_ext = Rot.from_matrix(T_ext[:3, :3]); t_ext = T_ext[:3, 3]
        tcp_pos_raw = pos + R.apply(t_ext)   # world_T_puck.p + R_puck @ t_ext
        tcp_R = R * R_ext
        tcp_pos = R_fix.apply(tcp_pos_raw - p_shift)
        tcp_R = R_fix * tcp_R

        # FOLD the two steps into ONE rigid transform tracker->remapped-TCP:
        #   (1) tracker->TCP calibrated extrinsic T_tracker_tcp = [R_ext | t_ext]
        #   (2) TCP body-axis relabel new x=-y / y=-x / z=-z == a body-frame RIGHT
        #       multiply by M (M's columns are the new axes in the old body frame).
        # Compose:  T_tracker_remap = T_tracker_tcp @ diag(M,1) = [ R_ext@M | t_ext ].
        # The relabel is a pure rotation ABOUT the TCP origin, so the translation is
        # UNCHANGED (t_c == t_ext); only the rotation columns turn. Applying this ONE
        # 4x4 to the raw puck gives the remapped-TCP in a single hop.
        M = _rotation_from_axis_map(["-y", "-x", "-z"]).as_matrix()   # det=+1, validated
        R_c = R_ext.as_matrix() @ M
        t_c = t_ext.copy()
        T_fused = np.eye(4); T_fused[:3, :3] = R_c; T_fused[:3, 3] = t_c
        Rc = Rot.from_matrix(R_c)
        remap_pos_raw = pos + R.apply(t_c)                # SINGLE fused transform on raw puck
        tcp_remap_pos = R_fix.apply(remap_pos_raw - p_shift)
        tcp_remap_R = R_fix * (R * Rc)
        # PROOF the fold is exact: the one-hop result must equal the two-hop (TCP then M).
        assert np.allclose(tcp_remap_pos, tcp_pos, atol=1e-9), "fused pos != TCP pos"
        assert np.allclose(tcp_remap_R.as_matrix(),
                           (tcp_R * Rot.from_matrix(M)).as_matrix(), atol=1e-9), \
            "fused R != two-step (extrinsic-then-M) R"

    pos = R_fix.apply(pos - p_shift)
    R = R_fix * R

    print(f"[inspect] {take_dir}")
    print(f"[inspect]   {N} samples ({int(live.sum())} live), t span {t[-1]-t[0]:.2f}s")
    fixmode = ("align-first (R0^-1)" if args.align_first else "raw libsurvive Z-up")
    if args.extra_rpy is not None:
        fixmode += f" + extra rpy {args.extra_rpy}"
    if args.axis_map is not None:
        fixmode += f" + axis-map {args.axis_map}"
    print(f"[inspect]   fixed rotation: {fixmode}")
    print(f"[inspect]   pos0={np.round(pos[0],4)}  frame0 rot={np.degrees(R[0].magnitude()):.1f}deg from identity")
    M0 = R[0].as_matrix()
    print(f"[inspect]   frame0 puck axes in world:  +x={np.round(M0[:,0],3)}  "
          f"+y={np.round(M0[:,1],3)}  +z={np.round(M0[:,2],3)}")

    server = viser.ViserServer(host=args.host, port=args.port, label="RAW tracker frame")
    server.scene.set_up_direction("+z")
    # NO grid — user wants a clean scene to free-drag/rotate.

    # World-origin reference triad (bigger) with a label at the tip of EACH axis so
    # the world frame's X/Y/Z are unambiguous (x=red fwd, y=green left, z=blue up).
    wlen = args.triad_len * 2.0
    add_triad(server, "/world_origin", np.zeros(3), Rot.identity(),
              length=wlen, radius=0.004)
    server.scene.add_label("/world_origin/label", "world (0,0,0)")
    for ax, nm in ((0, "world +X"), (1, "world +Y"), (2, "world +Z")):
        tip = np.zeros(3); tip[ax] = wlen * 1.12
        server.scene.add_label(f"/world_origin/{nm}", nm, position=tuple(tip))

    # Full raw trajectory polyline at the tracker (ORANGE) positions.
    p = pos.astype(np.float32)
    segs = np.stack([p[:-1], p[1:]], axis=1)  # (N-1,2,3)
    server.scene.add_line_segments("/tracker_path", points=segs,
                                   colors=(255, 170, 0), line_width=2.5)
    server.scene.add_label("/tracker_path/label", "tracker (orange)", position=tuple(pos[N // 3]))

    # Per-sample tracker triads every --stride frames (see the frame rotate).
    for i in range(0, N, max(1, args.stride)):
        add_triad(server, f"/tracker_{i:05d}", pos[i], R[i],
                  length=args.triad_len, radius=0.0022)

    # Highlight first (green ball) and last (red ball) + full triads.
    server.scene.add_icosphere("/first_pt", radius=0.008, color=(40, 220, 40), position=tuple(pos[0]))
    server.scene.add_label("/first_pt/label", "frame 0")
    server.scene.add_icosphere("/last_pt", radius=0.008, color=(220, 40, 40), position=tuple(pos[-1]))
    server.scene.add_label("/last_pt/label", f"frame {N-1}")
    add_triad(server, "/first_triad", pos[0], R[0], length=args.triad_len * 1.6, radius=0.0035)
    add_triad(server, "/last_triad", pos[-1], R[-1], length=args.triad_len * 1.6, radius=0.0035)

    # TCP overlay (CYAN path) — same frame as the tracker, derived via the calibrated
    # extrinsic. Thin grey segments connect each shown puck to its TCP so the fixed
    # tracker->TCP lever (the +z 5cm / -y 22.4cm you read off) is visible per frame.
    if tcp_pos is not None:
        tp = tcp_pos.astype(np.float32)
        tsegs = np.stack([tp[:-1], tp[1:]], axis=1)
        server.scene.add_line_segments("/tcp_path", points=tsegs, colors=(0, 210, 210),
                                       line_width=2.5)
        server.scene.add_label("/tcp_path/label", "TCP (cyan)", position=tuple(tcp_pos[N // 3]))
        lever = np.stack([pos[::max(1, args.stride)], tcp_pos[::max(1, args.stride)]], axis=1)
        server.scene.add_line_segments("/tcp_lever", points=lever.astype(np.float32),
                                       colors=(150, 150, 150), line_width=1.2)
        for i in range(0, N, max(1, args.stride)):
            add_triad(server, f"/tcp_{i:05d}", tcp_pos[i], tcp_R[i],
                      length=args.triad_len, radius=0.0022)
        add_triad(server, "/tcp_first", tcp_pos[0], tcp_R[0], length=args.triad_len * 1.6,
                  radius=0.0035)
        server.scene.add_icosphere("/tcp_first_pt", radius=0.008, color=(0, 210, 210),
                                   position=tuple(tcp_pos[0]))
        server.scene.add_label("/tcp_first_pt/label", "TCP frame 0")
        d0 = np.linalg.norm(tcp_pos[0] - pos[0])
        print(f"[inspect]   TCP overlay ON: |TCP-tracker| frame0 = {d0*100:.1f} cm "
              f"(extrinsic |t|={np.linalg.norm(np.asarray(json.load(open(args.tcp_extrinsic))['T_tracker_tcp'])[:3,3])*100:.1f} cm)")

    # Axis-RELABELED TCP overlay (toggle) — the FUSED tracker->remapped-TCP transform
    # (tracker->TCP extrinsic then the -y,-x,-z body relabel, folded into ONE 4x4
    # T_fused above) applied to the raw puck. Positions equal the cyan TCP path
    # (the relabel is a pure rotation about the TCP origin, t unchanged), so the
    # magenta path MUST overlap cyan and only the triads reorient. Hidden by default;
    # the GUI checkbox reveals it to A/B it.
    remap_handles = []
    if tcp_remap_pos is not None:
        rp = tcp_remap_pos.astype(np.float32)
        rsegs = np.stack([rp[:-1], rp[1:]], axis=1)
        remap_handles.append(server.scene.add_line_segments(
            "/tcp_remap_path", points=rsegs, colors=(255, 0, 220), line_width=2.5))
        remap_handles.append(server.scene.add_label(
            "/tcp_remap_path/label", "TCP remapped -y,-x,-z (magenta)",
            position=tuple(tcp_remap_pos[2 * N // 3])))
        for i in range(0, N, max(1, args.stride)):
            remap_handles += add_triad(server, f"/tcp_remap_{i:05d}", tcp_remap_pos[i],
                                       tcp_remap_R[i], length=args.triad_len, radius=0.0022)
        remap_handles += add_triad(server, "/tcp_remap_first", tcp_remap_pos[0],
                                   tcp_remap_R[0], length=args.triad_len * 1.6, radius=0.0035)
        for h in remap_handles:
            h.visible = False                                    # start hidden; checkbox reveals
        A0 = tcp_R[0].as_matrix(); B0 = tcp_remap_R[0].as_matrix()
        ok = (np.allclose(B0[:, 0], -A0[:, 1]) and np.allclose(B0[:, 1], -A0[:, 0])
              and np.allclose(B0[:, 2], -A0[:, 2]))
        with np.printoptions(precision=4, suppress=True):
            print(f"[inspect]   FUSED tracker->remapped-TCP T (one 4x4):\n{T_fused}")
        print(f"[inspect]   remap-TCP overlay READY (hidden): path == cyan TCP (same "
              f"positions); frame0 new_x=-old_y & new_y=-old_x & new_z=-old_z -> {ok}")

    gui = server.gui
    with gui.add_folder("RAW tracker frame"):
        gui.add_text("take", initial_value=os.path.basename(take_dir), disabled=True)
        gui.add_text("samples", initial_value=str(N), disabled=True)
        gui.add_text("frame0 rot (deg)", initial_value=f"{np.degrees(R[0].magnitude()):.1f}", disabled=True)
        gui.add_text("puck +z in world", initial_value=np.array2string(np.round(M0[:, 2], 2)), disabled=True)

    if remap_handles:
        with gui.add_folder("Axis relabel (TCP)"):
            cb = gui.add_checkbox("show remapped -y,-x,-z", initial_value=False)

            @cb.on_update
            def _toggle_remap(_) -> None:
                for h in remap_handles:
                    h.visible = cb.value
    print(f"[inspect] http://{args.host}:{args.port}  — no grid; drag to rotate. "
          f"triads every {args.stride} frames (x=red y=green z=blue). "
          f"tracker=orange TCP={'cyan' if tcp_pos is not None else 'off'}", flush=True)
    import time
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    raise SystemExit(main())
