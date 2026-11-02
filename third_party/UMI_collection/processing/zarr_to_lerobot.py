#!/usr/bin/env python3
"""Optional one-way bridge: our UMI Zarr ReplayBuffer (.zarr.zip) -> LeRobot dataset.

DECISION-K: Zarr is the PRIMARY format (UMI / diffusion_policy ingest it
natively; see processing/convert_to_umi.py). This exporter is
NOT run by default — it exists so a produced ``.zarr.zip`` can be converted to a
LeRobot dataset on demand (e.g. for LeRobot-based training / HF hub upload).

Adapted from HuMI's ``zarr_to_lerobot.py`` with ONE addition: our schema carries
an extra ``robot0_ft_wrench [T,6]`` key (Bota force/torque) that stock HuMI does
not — it is mapped to a plain float32 feature so the wrench survives the bridge.

``lerobot`` is a HEAVY dep (torch, ffmpeg, datasets) and is INTENTIONALLY NOT in
the collection/conversion ``.venv``. It is imported lazily here, so this module
imports fine without it; if you actually run the export, install lerobot first:

    uv pip install --python .venv/bin/python lerobot

Then:

    .venv/bin/python processing/zarr_to_lerobot.py -i session_XXX.zarr.zip \
        -o my_dataset --task "pick and place" [--fps 15] [--vcodec h264]

Notes:
  * RGB is RE-ENCODED to video (MP4) by LeRobot — lossy, unlike the Zarr's
    JpegXl. Default ``--vcodec auto`` may pick NVENC; pass ``--vcodec h264`` on a
    box with no NVIDIA GPU.
  * The Zarr ``episode_timestamps`` (relative, DECISION-A) is NOT emitted as a
    feature; LeRobot derives its own timeline from ``--fps``. Set ``--fps`` to
    the camera rate the take was recorded at (D405 ~15 Hz) for correct playback.
  * Pose math is already done in the Zarr (TCP 6-DOF, gripper metres); this is a
    pure key->feature remap, so the UNCALIBRATED warning (if any) carries over —
    check the source Zarr's attrs before training.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# Zarr-key -> (lerobot feature name, shape, dtype). camera keys handled separately.
def _build_features(data_group) -> dict:
    """Map the Zarr data keys to LeRobot feature specs. Unknown low-dim keys are
    passed through as float32 with their stored trailing shape, so OUR extra
    ``robot0_ft_wrench`` (and any future extension) is preserved automatically."""
    features: dict = {}
    for key in data_group.keys():
        shape = tuple(data_group[key].shape[1:])
        if key.startswith("camera") and key.endswith("_rgb"):
            features[f"observation.images.{key}"] = {
                "dtype": "video", "shape": shape,
                "names": ["height", "width", "channels"],
            }
        elif key == "episode_timestamps":
            continue  # LeRobot builds its own timeline from fps
        else:
            # All low-dim keys (eef_pos/rot, gripper_width, demo_*, AND our
            # robot0_ft_wrench) -> float32 feature, same name, stored shape.
            feat_shape = shape if shape else (1,)
            features[key] = {"dtype": "float32", "shape": feat_shape, "names": [key]}
    return features


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("-i", "--input", required=True, help="input .zarr.zip (from convert_to_umi.py)")
    p.add_argument("-o", "--repo-id", required=True, help="LeRobot repo_id / output dataset name")
    p.add_argument("--task", default="manipulation", help="natural-language task string for every frame")
    p.add_argument("--fps", type=int, default=15, help="dataset fps (set to the camera rate, D405 ~15 Hz)")
    p.add_argument("--root", default=None, help="output root dir (default: HF_LEROBOT_HOME/repo_id)")
    p.add_argument("--vcodec", default="auto", help="video codec; use 'h264' on a box without NVENC")
    args = p.parse_args(argv if argv is not None else sys.argv[1:])

    import zarr
    from imagecodecs_numcodecs import register_codecs
    register_codecs()

    # Open + validate the input and build the feature map FIRST (cheap, and fails
    # fast on a bad path) before requiring the heavy lerobot dep.
    g = zarr.open(args.input, "r")
    src_attrs = dict(g.attrs)
    if src_attrs.get("WARNING"):
        print(f"[zarr->lerobot] NOTE: source Zarr carries WARNING: {src_attrs['WARNING']}", file=sys.stderr)
    print(f"[zarr->lerobot] raw quaternion provenance: "
          f"{src_attrs.get('raw_quaternion_convention', 'unknown')} (already xyzw in poses)")
    # robot0_ft_wrench is physically different raw vs compensated (tool gravity in
    # vs out) — surface it so a downstream trainer doesn't mix conventions blindly.
    print(f"[zarr->lerobot] F/T wrench provenance: "
          f"{src_attrs.get('ft_compensated', 'unknown')} "
          f"(compensated=gravity removed | raw=native | mixed=per-episode varies)")

    data = g["data"]
    ends = g["meta/episode_ends"][:]
    keys = list(data.keys())
    features = _build_features(data)

    # Lazy heavy import — keeps this module import-safe without lerobot installed.
    try:
        from lerobot.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset
    except ImportError:
        print("[zarr->lerobot] ERROR: lerobot is not installed in this venv.\n"
              "  This exporter is optional (Zarr is the primary format). To use it:\n"
              "    uv pip install --python .venv/bin/python lerobot\n"
              "  The LeRobotDataset.create / add_frame / save_episode API has shifted\n"
              "  across lerobot releases; if create() rejects a kwarg below, pin the\n"
              "  version your training stack uses (see env/requirements-processing.txt).",
              file=sys.stderr)
        return 2

    # Fail-fast on an existing output dir (review #15): LeRobotDataset.create does
    # mkdir(exist_ok=False) and would raise a confusing FileExistsError mid-run.
    import pathlib
    out_dir = pathlib.Path(args.root).expanduser() if args.root else (HF_LEROBOT_HOME / args.repo_id)
    if out_dir.exists():
        print(f"[zarr->lerobot] ERROR: output already exists: {out_dir}\n"
              f"  Pass a fresh --repo-id/--root or delete it first.", file=sys.stderr)
        return 3

    # `vcodec=` is accepted by some lerobot versions and not others (review #14):
    # try with it, fall back to the minimal signature so the export works on stock
    # PyPI lerobot too.
    create_kwargs = dict(repo_id=args.repo_id, fps=args.fps, features=features,
                         root=args.root, use_videos=True)
    try:
        dataset = LeRobotDataset.create(vcodec=args.vcodec, **create_kwargs)
    except TypeError:
        print("[zarr->lerobot] NOTE: this lerobot ignores vcodec= on create(); "
              "using its default video encoder.", file=sys.stderr)
        dataset = LeRobotDataset.create(**create_kwargs)

    feature_keys = [k for k in keys if k != "episode_timestamps"]
    starts = np.concatenate([[0], ends[:-1]]).astype(int)
    print(f"[zarr->lerobot] {len(ends)} episodes, T_total={int(ends[-1]) if len(ends) else 0}")
    for ep_i, (a, b) in enumerate(zip(starts, ends)):
        a, b = int(a), int(b)
        # Bulk-load the episode's arrays once (review #16): one slice per key (one
        # JpegXl chunk decode per frame still, but no per-(key,t) zip re-seek).
        seg = {k: data[k][a:b] for k in feature_keys}
        for i in range(b - a):
            frame: dict = {"task": args.task}
            for key in feature_keys:
                val = seg[key][i]
                if key.startswith("camera") and key.endswith("_rgb"):
                    frame[f"observation.images.{key}"] = val
                else:
                    arr = np.asarray(val, dtype=np.float32)
                    frame[key] = arr if arr.ndim else np.array([arr], dtype=np.float32)
            try:
                dataset.add_frame(frame)
            except RuntimeError as e:
                if "nvenc" in str(e).lower():
                    raise RuntimeError("NVENC failed; retry with --vcodec h264") from e
                raise
        # LeRobot maps shape-(1,) features to a scalar HF Value; squeeze 1-elem arrays.
        if getattr(dataset, "episode_buffer", None) is not None:
            for key, ft in dataset.features.items():
                if ft.get("shape") == (1,):
                    bufk = dataset.episode_buffer.get(key)
                    if isinstance(bufk, list) and bufk and isinstance(bufk[0], np.ndarray):
                        dataset.episode_buffer[key] = [x.item() for x in bufk]
        dataset.save_episode()
        print(f"  episode {ep_i}: {b - a} frames")

    if hasattr(dataset, "finalize"):
        dataset.finalize()
    print(f"[zarr->lerobot] done -> {getattr(dataset, 'root', args.root or args.repo_id)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
