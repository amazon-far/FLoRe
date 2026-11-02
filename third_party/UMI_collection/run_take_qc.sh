#!/usr/bin/env bash
# Render a per-take RAW-data QC sheet (one PNG per take folder) for a recorded
# session. Adapted from the humanoid-UMI dataset visualizers' "diagnostic panels
# per frame + stats block" style, extended with a force/torque row, and wired to
# OUR converter's own de-glitch verdict (umi_convert_core) so the KEEP/DROP badge
# and red glitch dots match convert_to_umi exactly. Read-only; OFFLINE.
#
# Plots the RAW recorded streams (tracker ~80Hz, F/T ~230Hz, gripper ~30Hz) — NOT
# the converted .zarr (which is already de-glitched/resampled and can't SHOW a
# glitch). This is the visual twin of processing/audit_session.py.
#
#   ./run_take_qc.sh data/recordings/session_2026.08.03_21.23.19
#   ./run_take_qc.sh data/recordings/session_2026.08.03_21.23.19 --takes 000 076
#   ./run_take_qc.sh <session> --out-dir data/qc/session_2026.08.03   # all PNGs in one dir
#   # badge the clean94 exclusions (still plotted, marked EXCLUDED):
#   ./run_take_qc.sh <session> --dropped 011 013 025 037 048 058 072 074 075 076 083 088
#
# .venv has no matplotlib, so this runs via `uv run` in an ephemeral env
# (numpy+scipy+matplotlib) — the same pattern the reference visualizers use.
# Extra args after the session are forwarded to processing/plot_take_qc.py.
set -eo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ "$#" -lt 1 ]; then
  echo "usage: ./run_take_qc.sh <session_dir_or_take_dir> [--takes ...] [--out-dir DIR] [--dropped ...]" >&2
  exit 2
fi

cd "$REPO_ROOT"
echo "[qc] rendering per-take QC sheets (uv ephemeral env: numpy+scipy+matplotlib)"
UV_ONLY_BINARY=:all: uv run --no-project --python 3.12 --index-strategy unsafe-best-match \
  --with "numpy>=2" --with scipy --with matplotlib \
  python processing/plot_take_qc.py "$@"
