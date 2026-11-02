#!/usr/bin/env bash
# Replay a RAW recorded take in viser (OFFLINE — no hardware, no bridges, reads
# only already-committed files). Draws the SAME gripper-only scene the live
# collection viz shows for MODE=1pt/2pt (DISPLAY_MODE=gripper): the three yam-gripper
# meshes with the jaws driven by the recorded encoder, EXACTLY the three axis triads
# (TCP, tracker, wrist_yaw_link), the recorded D405 camera, the Bota F/T, and the TCP
# (robot0_eef_pos) path as a green polyline the gripper rides — with a draggable
# timeline to stop at any timestamp.
#
# This is the RAW-take twin of run_replay_zarr_viz.sh: both render the same cluster,
# so a raw take and its .zarr.zip conversion can be eyeballed side by side. NO mink
# IK (gripper-only display), so — unlike before — no humi_ik on PYTHONPATH is needed.
#
# This does NOT touch the live data-collection pipeline (RECORD=1
# ./run_collection_viz.sh). It is a separate viewer for what you already recorded.
#
# Point it at a SESSION and review the whole thing without relaunching: the "Take"
# panel has a clickable Take dropdown (each entry labelled with its frame count +
# duration, like the zarr viewer's Episode dropdown), ⏮ Prev take / ⏭ Next take, and
# a 🗑 Delete this take button. Delete is a REAL, IRREVERSIBLE delete of the session's
# take (one confirm click: the label flips to "CONFIRM delete take_NNN"). On a MERGED
# session the take_* entries are symlinks, so it removes the LINK and the per-sitting
# session keeps its copy; on a plain session it rmtree's the recording. Either way the
# converter's take_* glob stops seeing it, so a re-convert simply excludes it.
#
# Usage:
#   ./run_replay_viz.sh data/recordings/session_2026.06.20_15.08.35     # take_000
#   ./run_replay_viz.sh data/recordings/session_XXX/take_002            # a specific take
#   PORT=8085 ./run_replay_viz.sh data/recordings/session_XXX           # pick the port
#   ./run_replay_viz.sh data/recordings/session_XXX --raw-pose          # raw world pose (no home-anchor)
#
# Any extra args after the path are forwarded to replay_raw_viz.py
# (--take, --port, --raw-pose, --reach-scale, --cam-offset, --cam-rpy, --cam-fov-deg, …).
set -eo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$REPO_ROOT/.venv"
PY="${REPLAY_PY:-$VENV/bin/python}"
PORT="${PORT:-8083}"

if [ "$#" -lt 1 ]; then
  echo "usage: ./run_replay_viz.sh <session_dir|take_dir> [extra args]" >&2
  echo "   e.g. ./run_replay_viz.sh data/recordings/session_2026.06.20_15.08.35" >&2
  exit 2
fi

# No PYTHONPATH=humi_ik: this tool is gripper-only (no mink IK). It DOES need viz/ +
# processing/ for the gripper geometry, the TCP extrinsic, and the JpegXl codec, which
# replay_raw_viz.py adds to sys.path internally.
echo "[run] raw-take replay on http://localhost:$PORT  (Ctrl-C to stop)"
"$PY" "$REPO_ROOT/replay/replay_raw_viz.py" --port "$PORT" "$@"
