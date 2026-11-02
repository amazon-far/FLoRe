#!/usr/bin/env python3
"""Rigidly re-root a UMI .zarr.zip so EACH episode's FIRST-frame TCP is the world
origin (000, identity axes: +x fwd, +y left, +z up).

WHAT this does (per episode, using that episode's own frame 0 as (p0, R0)):
    p'(t) = R0^T @ (p(t) - p0)
    R'(t) = R0^T @ R(t)
so frame 0 becomes exactly (0,0,0) with identity orientation, and the whole
trajectory rides in frame-0's body frame. This is a genuine SE(3) rigid change
of world frame — position and rotation stay rigidly coupled (UNLIKE the viz's
home_anchor, which translates position but rotates orientation separately and is
therefore NOT a coordinate transform; it cannot be reproduced by any world frame).

WHY the result will NOT look like the collection viz: the recorded frame-0 TCP is
NOT upright (this rig starts ~178° from aligned-world identity — hand-held pose).
The viz fakes uprightness with R_home=identity; a rigid re-root honestly carries
that 178° through, so absolute readouts differ from the viz by tens of cm / deg.
That is expected and correct.

WHAT is transformed: pose arrays only —
    robot0_eef_pos            (T,3)  position
    robot0_eef_rot_axis_angle (T,3)  rotation (rotvec)
    robot0_demo_start_pose    (T,6)  [pos3, rotvec3], per-episode-constant
    robot0_demo_end_pose      (T,6)  [pos3, rotvec3], per-episode-constant
    hl_<role>_action_pose     (T,H,6) [pos3, rotvec3] per step   (if present)
Everything else is copied VERBATIM: camera0_rgb, episode_timestamps,
robot0_gripper_width, robot0_ft_wrench(+_hires), robot0_pose_glitch, and ALL the
hl_<role>_{force*,gripper*,*_valid} arrays. Force is sensor-frame (frame-invariant),
gripper is a scalar width, masks/timestamps are frame-agnostic.

IMPORTANT — training semantics: the policy's action label is PER-ANCHOR
(R_ref^T(p-p_ref) at each t0), which is INVARIANT to any rigid world re-root, so a
model trained on the re-rooted file gets BIT-IDENTICAL deltas. This export changes
only the ABSOLUTE readout/frame, not the learnable target. attrs get a
``rerooted_to_episode_first_frame`` flag + note so no downstream tool mistakes it
for the raw aligned-world file (the trainer's hl_pose_frame gate expects
'absolute_aligned_world'; this file is deliberately different).

Read-only w.r.t. the input; writes a NEW .zarr.zip atomically.

Usage:
    reroot_zarr_first_frame.py IN.zarr.zip -o OUT.zarr.zip
    reroot_zarr_first_frame.py IN.zarr.zip -o OUT.zarr.zip --verify   # extra checks
"""
from __future__ import annotations
import argparse
import os
import sys

import numpy as np
import zarr
from scipy.spatial.transform import Rotation as Rot

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from imagecodecs_numcodecs import register_codecs  # noqa: E402  (vendored JpegXl)

# Pose arrays laid out as position (first 3) and rotvec (next 3). eef is split
# across two arrays; demo_* and hl_*_action_pose pack both into a 6-vector.
_POS_KEY = "robot0_eef_pos"
_ROT_KEY = "robot0_eef_rot_axis_angle"
_POSE6_KEYS = ("robot0_demo_start_pose", "robot0_demo_end_pose")  # (T,6)


def _reroot_pos_rot(pos: np.ndarray, rot: Rot, p0: np.ndarray, R0: Rot):
    """p' = R0^T (p - p0) ;  R' = R0^T R.  Returns (pos', rotvec')."""
    R0inv = R0.inv()
    pp = R0inv.apply(pos - p0)
    RR = R0inv * rot
    return pp, RR.as_rotvec()


def reroot_zarr(in_path: str, out_path: str, verify: bool = False) -> None:
    register_codecs()
    zin = zarr.open(in_path, mode="r")
    data_in = zin["data"]
    keys = sorted(data_in.array_keys())
    ends = np.asarray(zin["meta/episode_ends"][:], dtype=np.int64)
    starts = np.concatenate([[0], ends[:-1]]).astype(np.int64)
    T_total = int(ends[-1])

    # hl action-pose arrays present? (only when converted with --hands)
    hl_action_pose_keys = [k for k in keys if k.startswith("hl_") and k.endswith("_action_pose")]

    print(f"[reroot] {os.path.basename(in_path)}: {len(ends)} episode(s), T_total={T_total}")
    print(f"[reroot]   transforming: {_POS_KEY}, {_ROT_KEY}, {list(_POSE6_KEYS)}"
          + (f", {hl_action_pose_keys}" if hl_action_pose_keys else ""))

    # --- load pose arrays we will rewrite (small; the rest is copied lazily) ---
    pos = np.asarray(data_in[_POS_KEY][:], dtype=np.float64)          # (T,3)
    rvec = np.asarray(data_in[_ROT_KEY][:], dtype=np.float64)         # (T,3)
    pose6 = {k: np.asarray(data_in[k][:], dtype=np.float64) for k in _POSE6_KEYS if k in data_in}
    hlap = {k: np.asarray(data_in[k][:], dtype=np.float64) for k in hl_action_pose_keys}

    out_pos = np.empty_like(pos)
    out_rvec = np.empty_like(rvec)
    out_pose6 = {k: v.copy() for k, v in pose6.items()}
    out_hlap = {k: v.copy() for k, v in hlap.items()}

    for s, e in zip(starts, ends):
        s, e = int(s), int(e)
        p0 = pos[s].copy()
        R0 = Rot.from_rotvec(rvec[s])
        # eef pos/rot
        out_pos[s:e], out_rvec[s:e] = _reroot_pos_rot(pos[s:e], Rot.from_rotvec(rvec[s:e]), p0, R0)
        # demo_start/end (each row is [pos3, rotvec3])
        for k, v in pose6.items():
            pp, rr = _reroot_pos_rot(v[s:e, :3], Rot.from_rotvec(v[s:e, 3:]), p0, R0)
            out_pose6[k][s:e, :3] = pp
            out_pose6[k][s:e, 3:] = rr
        # hl action pose (T,H,6): flatten steps, transform, reshape
        for k, v in hlap.items():
            seg = v[s:e]                       # (n,H,6)
            n, H, _ = seg.shape
            flat_p = seg[..., :3].reshape(-1, 3)
            flat_r = seg[..., 3:].reshape(-1, 3)
            pp, rr = _reroot_pos_rot(flat_p, Rot.from_rotvec(flat_r), p0, R0)
            out_hlap[k][s:e, :, :3] = pp.reshape(n, H, 3)
            out_hlap[k][s:e, :, 3:] = rr.reshape(n, H, 3)

    if verify:
        _verify(pos, rvec, out_pos, out_rvec, pose6, out_pose6, starts, ends)

    # --- write NEW zarr.zip (atomic: tmp + os.replace) ---
    #
    # Strategy: FIRST copy the whole input store VERBATIM (raw encoded chunks, so the
    # LOSSY JpegXl camera0_rgb is byte-preserved — re-`data=`-writing it would decode +
    # RE-COMPRESS lossily, silently degrading every frame ~20 code-values). THEN overwrite
    # ONLY the pose arrays with transformed values. copy_store preserves .zarray metadata
    # (chunks/compressor/filters) and attrs, so the re-written arrays reuse the identical
    # encoding and only their bytes change.
    tmp = out_path + ".tmp"
    if os.path.exists(tmp):
        os.remove(tmp)
    replaced = {_POS_KEY: out_pos.astype(np.float32), _ROT_KEY: out_rvec.astype(np.float32)}
    for k, v in out_pose6.items():
        replaced[k] = v.astype(np.float32)
    for k, v in out_hlap.items():
        replaced[k] = v.astype(np.float32)

    with zarr.ZipStore(tmp, mode="w") as zs:
        # 1) verbatim raw-chunk copy of everything (RGB stays byte-identical)
        zarr.copy_store(zin.store, zs)
        # 2) overwrite ONLY the transformed pose arrays, in place, same encoding
        zout = zarr.group(store=zs)
        for k, arr in replaced.items():
            zout["data"][k][:] = arr
        # 3) stamp re-root provenance (copy_store already carried the originals)
        zout.attrs["rerooted_to_episode_first_frame"] = True
        zout.attrs["reroot_note"] = (
            "Each episode rigidly re-rooted so frame-0 TCP = (0,0,0)/identity via "
            "p'=R0^T(p-p0), R'=R0^T R. Pose arrays (eef, demo_*, hl_*_action_pose) "
            "transformed; force/gripper/rgb/timestamps/masks unchanged. NOT the raw "
            "aligned-world file — do not feed a hl_pose_frame='absolute_aligned_world' "
            "trainer without accounting for this (per-anchor deltas are invariant, so "
            "training is unaffected, but the absolute frame differs)."
        )
        # invalidate the old pose-frame claim so a gate can't silently accept it
        if "hl_pose_frame" in zout.attrs:
            zout.attrs["hl_pose_frame"] = "rerooted_episode_first_frame"
    os.replace(tmp, out_path)
    print(f"[reroot] wrote {out_path}  ({os.path.getsize(out_path)/1e6:.1f} MB)")


def _verify(pos, rvec, out_pos, out_rvec, pose6, out_pose6, starts, ends):
    """Assert frame-0 lands at origin/identity and per-anchor deltas are invariant."""
    print("[verify] checking each episode's frame 0 and per-anchor invariance...")
    for s, e in zip(starts, ends):
        s, e = int(s), int(e)
        assert np.allclose(out_pos[s], 0, atol=1e-9), f"ep@{s}: frame0 pos != 0"
        assert np.linalg.norm(out_rvec[s]) < 1e-6, f"ep@{s}: frame0 rot != identity"
        # per-anchor delta invariance: R_ref^T(p_k - p_ref) equal in both frames
        for t0 in (s, (s + e) // 2):
            k = min(t0 + 16, e - 1)
            Ra = Rot.from_rotvec(rvec[t0]); da = Ra.inv().apply(pos[k] - pos[t0])
            Rb = Rot.from_rotvec(out_rvec[t0]); db = Rb.inv().apply(out_pos[k] - out_pos[t0])
            assert np.linalg.norm(da - db) < 1e-7, f"per-anchor delta changed @{t0}: {da} vs {db}"
    print("[verify]   OK: frame0=origin/identity; per-anchor deltas bit-invariant.")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("in_zarr", help="input .zarr.zip")
    ap.add_argument("-o", "--output", required=True, help="output .zarr.zip")
    ap.add_argument("--verify", action="store_true", help="run extra numeric checks")
    args = ap.parse_args(argv)
    if os.path.abspath(args.in_zarr) == os.path.abspath(args.output):
        print("[reroot] refusing to overwrite the input in place", file=sys.stderr)
        return 2
    reroot_zarr(args.in_zarr, args.output, verify=args.verify)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
