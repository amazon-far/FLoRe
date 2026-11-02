#!/usr/bin/env python3
"""Quick sanity inspector for a produced UMI Zarr ReplayBuffer.

Prints per-episode shapes, ranges, timeline stats, and the load-bearing
invariants (all data keys length-T, episode_ends cumulative, timestamps strictly
increasing, gripper_width straddling the 0.08 m UMI grasp threshold). A full 3D
viser replay is a follow-up; this is the fast "did the conversion produce sane
data" check.

Run (in the collection .venv — DECISION-I, one env for capture + conversion):
    .venv/bin/python processing/inspect_umi.py session.zarr.zip
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("zarr_path")
    args = p.parse_args(argv if argv is not None else sys.argv[1:])

    import zarr
    from imagecodecs_numcodecs import register_codecs
    register_codecs()

    g = zarr.open(args.zarr_path, "r")
    print(f"== {args.zarr_path} ==")
    print("attrs:")
    for k, v in dict(g.attrs).items():
        flag = "  <-- WARNING" if k == "WARNING" else ""
        print(f"  {k}: {v}{flag}")

    ends = g["meta/episode_ends"][:]
    T = int(ends[-1]) if len(ends) else 0
    print(f"\nepisodes: {len(ends)}  | T_total: {T}")
    print(f"episode_ends: {list(ends)}")
    print(f"episode_lengths: {list(np.diff(np.concatenate([[0], ends])))}")

    print("\ndata arrays:")
    all_T = True
    for k in sorted(g["data"].keys()):
        a = g["data"][k]
        ok = a.shape[0] == T
        all_T = all_T and ok
        print(f"  {k:30s} {str(a.shape):18s} {a.dtype}  chunks={a.chunks}"
              f"{'' if ok else '  <-- LENGTH != T!'}")

    # episode_timestamps are RELATIVE per episode (DECISION-A): each episode
    # restarts at 0, so the GLOBAL array is intentionally NOT monotone. Check
    # monotonicity WITHIN each episode and that each episode starts at ~0 instead.
    ts = g["data/episode_timestamps"][:]
    starts = np.concatenate([[0], ends[:-1]]).astype(int)
    per_ep_mono = True
    per_ep_zero = True
    for a, b in zip(starts, ends):
        a, b = int(a), int(b)
        seg = ts[a:b]
        if len(seg) > 1 and not np.all(np.diff(seg) > 0):
            per_ep_mono = False
        if len(seg) and abs(float(seg[0])) > 1e-6:
            per_ep_zero = False
    gw = g["data/robot0_gripper_width"][:].reshape(-1)
    print("\nsanity:")
    print(f"  all data length == T: {all_T}")
    print(f"  episode_timestamps per-episode strictly increasing: {per_ep_mono}")
    print(f"  episode_timestamps per-episode start at 0 (relative): {per_ep_zero}")
    mono = per_ep_mono and per_ep_zero
    print(f"  gripper_width range (m): [{gw.min():.4f}, {gw.max():.4f}]")
    if gw.max() < 0.08:
        print("    NOTE: max gripper_width < 0.08 m (UMI's grasp threshold) — "
              "every frame reads as 'grasping'. Check the gripper calibration / "
              "override gripper_width_threshold in training.")
    pos = g["data/robot0_eef_pos"][:]
    print(f"  eef_pos bbox (m): min={np.round(pos.min(0),3)} max={np.round(pos.max(0),3)}")
    if "robot0_ft_wrench" in g["data"]:
        w = g["data/robot0_ft_wrench"][:]
        print(f"  ft_wrench |F| mean/max (N): {np.linalg.norm(w[:,:3],axis=1).mean():.2f} / "
              f"{np.linalg.norm(w[:,:3],axis=1).max():.2f}")
    return 0 if (all_T and mono) else 1


if __name__ == "__main__":
    raise SystemExit(main())
