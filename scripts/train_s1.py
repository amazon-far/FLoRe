#!/usr/bin/env python3
"""Tyro CLI entry point for S1 training (diffusion).

Examples
--------
    # HL-v2 diffusion (DDPM/DDIM, EMA default-on):
    uv run python scripts/train_s1.py --algo diffusion \
        --hl-data.zarr-paths session.zarr.zip \
        --hl-data.val-last-episodes 4 \
        --hl-data.norm-stats-path checkpoints/norm_stats_hl_left_clean99.json \
        --diffusion.diffusion-net unet --batch-size 32 --max-steps 100000

The model's knobs live under the ``--diffusion.*`` prefix, and its W&B loss namespace is
``train/diffusion_loss/*``.
"""

from __future__ import annotations

import logging

import tyro

from flore.s1_policy.training.config import TrainConfig
from flore.s1_policy.training.trainer import Trainer


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    config = tyro.cli(TrainConfig)
    if not config.hl_data.zarr_paths:
        raise SystemExit("No data: pass --hl-data.zarr-paths <one or more .zarr.zip files>")
    Trainer(config).train()


if __name__ == "__main__":
    main()
