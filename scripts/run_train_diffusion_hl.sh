#!/usr/bin/env bash
# Full HL-v2 Diffusion run: ViT-B + 1D UNet, epsilon, 16-step DDIM, EMA, GPU 1.
# W&B project: FLoRe
#
# Everything the run needs is set HERE, not inherited from the launching shell:
# `tmux new` starts a fresh login shell and only propagates the variables in
# tmux's update-environment list, so outer-shell DS=/STATS=/WANDB_* would arrive
# empty. An empty --hl-data.zarr-paths does NOT trip the "No data" guard (tyro
# turns "" into the truthy tuple ('',)) and dies ~2 min later inside the loader.
set -euo pipefail

cd "$(dirname "$0")/.."

DS=third_party/UMI_collection/data/datasets/session_2026.08.03_clean94.zarr.zip
STATS=checkpoints/norm_stats_hl_left_clean94.json
RUN=hl_left_clean94_diffusion_unet

# W&B: your own `wandb login` is used (export WANDB_BASE_URL for a self-hosted server).
# export WANDB_ENTITY=<your-entity>   # optional: no --wandb-entity flag exists, the entity is env-only

[[ -f "$DS" ]]    || { echo "FATAL: dataset missing: $DS" >&2; exit 1; }
[[ -f "$STATS" ]] || { echo "FATAL: norm stats missing: $STATS — run scripts/fit_hl_norm_stats.py" >&2; exit 1; }

# Diffusion defaults to clip_sample=True, which REQUIRES normalized actions; the
# missing-stats failure is loud but arrives late. Check CUDA here too: trainer.py
# downgrades to CPU when CUDA is unavailable and exits 0, which wastes days.
CUDA_VISIBLE_DEVICES=1 .venv/bin/python - <<'PY'
import sys, torch
if not torch.cuda.is_available():
    sys.exit("FATAL: CUDA unavailable on GPU 1 — refusing to train on CPU")
print("GPU 1:", torch.cuda.get_device_name(0))
PY

mkdir -p logs

set -o pipefail
CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/train_s1.py \
    --algo diffusion --pipeline hl \
    --hl-data.zarr-paths "$DS" \
    --hl-data.mode left_only \
    --hl-data.val-last-episodes 4 \
    --hl-data.norm-stats-path "$STATS" \
    --diffusion.vit vit_b \
    --diffusion.diffusion-net unet \
    --diffusion.prediction-type epsilon \
    --diffusion.num-train-timesteps 50 \
    --diffusion.num-inference-steps 16 \
    --diffusion.input-pertub 0.1 \
    --diffusion.num-draws 1 \
    --batch-size 32 --num-workers 8 \
    --lr 1e-4 --weight-decay 1e-4 --grad-clip 1.0 \
    --warmup-steps 1000 --max-steps 100000 \
    --save-every 5000 --log-every 50 \
    --eval-seeds 0 1 2 3 \
    --sample-every-epochs 5 \
    --sample-diagnostic-batches 1 --sample-diagnostic-samples 4 \
    --artifact-every-epochs 50 \
    --epoch-artifact-every-epochs 10 \
    --wandb --wandb-project FLoRe --wandb-run-name "$RUN" \
    --output-dir "checkpoints/$RUN" \
    "$@" \
    2>&1 | tee -a "logs/$RUN.log"
