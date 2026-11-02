#!/usr/bin/env python3
"""Empirically measure the effective F/T rate and per-camera-interval sample
counts on RECORDED takes, to size K for a hi-rate wrench export.

The nominal Bota read rate is 250 Hz and the camera ~15 Hz, but USB/EKF jitter
means neither holds. This script replicates the EXACT interval binning that
``umi_convert_core.antialias_resample_wrench`` uses (midpoint edges, window-
confined to [grid_t[0], grid_t[-1]]) and reports, per camera interval, how many
raw F/T samples actually land in it — the ceiling on how many hi-rate samples/
step we can export WITHOUT interpolating up.

Usage:
    .venv/bin/python processing/measure_ft_rate.py [session_dir ...]
    (defaults to every session under data/recordings/)
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np


def camera_grid_t(take_dir: str) -> np.ndarray:
    """Camera-frame wall-clock timestamps (the resample grid), sorted."""
    fi = np.load(os.path.join(take_dir, "camera", "frame_index.npy"))
    return np.sort(fi[:, 0].astype(np.float64))


def counts_per_interval(ft_t: np.ndarray, grid_t: np.ndarray) -> np.ndarray:
    """Replicate antialias_resample_wrench binning: midpoint edges, window-
    confined to [grid_t[0], grid_t[-1]]. Returns per-grid-point raw-sample count."""
    in_win = (ft_t >= grid_t[0]) & (ft_t <= grid_t[-1])
    wt = ft_t[in_win]
    mids = 0.5 * (grid_t[:-1] + grid_t[1:])
    edges = np.concatenate([[-np.inf], mids, [np.inf]])
    bins = np.clip(np.searchsorted(edges, wt, side="right") - 1, 0, len(grid_t) - 1)
    c = np.zeros(len(grid_t), dtype=np.int64)
    np.add.at(c, bins, 1)
    return c


def measure(take_dir: str):
    grid = camera_grid_t(take_dir)
    ft = np.load(os.path.join(take_dir, "ft_wrench.npy"))
    ft_t = ft[:, 0].astype(np.float64)
    c = counts_per_interval(ft_t, grid)
    interior = c[1:-1]  # drop the two edge bins that collect window-edge samples
    ft_hz = (len(ft_t) - 1) / (ft_t[-1] - ft_t[0]) if len(ft_t) > 1 else 0.0
    cam_hz = (len(grid) - 1) / (grid[-1] - grid[0]) if len(grid) > 1 else 0.0
    return interior, ft_hz, cam_hz, len(grid), len(ft_t)


def main(argv):
    sessions = argv[1:] or sorted(glob.glob("data/recordings/session_*"))
    pooled = []
    print(f"{'take':>28} {'ncam':>5} {'nft':>6} {'ft_hz':>6} {'cam_hz':>6} | min p1 p5 p10 med p90 max")
    for sess in sessions:
        for take in sorted(glob.glob(os.path.join(sess, "take_*"))):
            if not os.path.exists(os.path.join(take, "ft_wrench.npy")):
                continue
            if not os.path.exists(os.path.join(take, "camera", "frame_index.npy")):
                continue
            interior, ft_hz, cam_hz, ncam, nft = measure(take)
            if len(interior) == 0:
                continue
            pooled.append(interior)
            p = lambda a, q: int(np.percentile(a, q))
            label = os.path.join(os.path.basename(sess), os.path.basename(take))
            print(f"{label:>28} {ncam:>5} {nft:>6} {ft_hz:6.1f} {cam_hz:6.1f} | "
                  f"{interior.min():>3} {p(interior,1):>3} {p(interior,5):>3} "
                  f"{p(interior,10):>3} {p(interior,50):>3} {p(interior,90):>3} {interior.max():>3}")
    if not pooled:
        print("No takes with F/T + camera/frame_index.npy found.")
        return
    allc = np.concatenate(pooled)
    print(f"\n=== POOLED interior intervals (n={len(allc)}) ===")
    for q in [0, 1, 5, 10, 25, 50, 75, 90, 100]:
        print(f"  p{q:>3}: {np.percentile(allc, q):.1f}")
    print(f"  mean {allc.mean():.2f}  std {allc.std():.2f}  empty-intervals {int((allc==0).sum())}")
    print("  K safety (fraction of intervals with >= K raw samples):")
    for K in [4, 6, 8, 10, 12, 16]:
        print(f"    K={K:>2}: {(allc>=K).mean()*100:6.2f}%   min-interval={allc.min()}")
    # Recommended K = the largest K covered by 100% of intervals (min count).
    print(f"\n  => 100%-safe K (== min interval count): {allc.min()}")


if __name__ == "__main__":
    main(sys.argv)
