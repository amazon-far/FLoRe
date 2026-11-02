#!/usr/bin/env bash
# Replay a CONVERTED LeRobot v2.1 dataset (Psi0/GR00T layout) in viser. Like the
# RAW-take replay (run_replay_viz.sh), this puts the FULL G1 HUMANOID in the scene:
# the arm is driven by the SAME mink IK (recovered from the stored TCP pose) and
# the recorded camera rides the REAL forward-kinematics wrist, so the image sits
# UPRIGHT relative to the ground (a bare TCP-mounted frustum tilts it). Pick an
# episode from the dropdown in the browser, scrub the timeline. OFFLINE, read-only.
#
# It renders what the dataset contains (states/action + the egocentric mp4 + F/T)
# with the robot reconstructed via IK. Pass --no-ik to drop the humanoid and show
# only the stored TCP triad + camera + F/T.
#
# First produce a dataset from a raw session:
#   .venv/bin/python processing/convert_to_lerobot.py data/recordings/session_XXX \
#       -o data/lerobot/session_XXX --task "pick and place"
# then:
#   ./run_replay_lerobot_viz.sh data/lerobot/session_XXX
#   ./run_replay_lerobot_viz.sh data/lerobot/session_XXX --episode 2
#   ./run_replay_lerobot_viz.sh data/lerobot/session_XXX --no-ik     # TCP triad only
#   PORT=8087 ./run_replay_lerobot_viz.sh data/lerobot/session_XXX
#
# Extra args after the path are forwarded to replay_lerobot_viz.py
# (--episode, --port, --no-ik, --reach-scale, --cam-offset, --cam-rpy, --cam-fov-deg).
set -eo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$REPO_ROOT/.venv"
PY="${REPLAY_PY:-$VENV/bin/python}"
HUMI_SRC="$REPO_ROOT/third_party/humi_ik"   # mink IK import (same as the live viz / raw replay)
PORT="${PORT:-8085}"

if [ "$#" -lt 1 ]; then
  echo "usage: ./run_replay_lerobot_viz.sh <lerobot_dataset_dir> [extra args]" >&2
  echo "   e.g. ./run_replay_lerobot_viz.sh data/lerobot/session_XXX --episode 0" >&2
  exit 2
fi

# PYTHONPATH=humi_ik is REQUIRED here (unlike the zarr replay): this tool drives the
# mink IK to reconstruct the humanoid, importing viz/g1_right_hand_ik.py. The script
# also needs processing/ (state/action slices), which replay_lerobot_viz.py adds to
# sys.path internally.
echo "[run] lerobot replay on http://localhost:$PORT  (Ctrl-C to stop)"
PYTHONPATH="$HUMI_SRC:${PYTHONPATH:-}" "$PY" "$REPO_ROOT/replay/replay_lerobot_viz.py" \
  --port "$PORT" "$@"
