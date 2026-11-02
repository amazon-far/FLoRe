#!/usr/bin/env bash
# Replay a CONVERTED UMI Zarr (.zarr.zip) in viser to CONFIRM the conversion kept
# the information (no lost frames / aliasing / time misalignment / weird motion).
# Pick an episode from the dropdown in the browser, scrub the timeline. OFFLINE,
# read-only; independent of the live pipeline and of run_replay_viz.sh.
#
# It renders ONLY what the file contains (TCP path, the stored 224x224 RGB,
# resampled F/T + gripper width) — no IK re-solve, by design, so a conversion
# artifact is visible here instead of being masked.
#
# First produce a .zarr.zip from a raw session (see README), then:
#   ./run_replay_zarr_viz.sh session_XXX.zarr.zip
#   ./run_replay_zarr_viz.sh out.zarr.zip --episode 2
#   PORT=8086 ./run_replay_zarr_viz.sh out.zarr.zip
#
# OPTIONAL: add --motion-completion to ALSO draw a full G1 humanoid, completed ONLINE from
# the recorded 3pt tracker stream (two hands + chest) by the Motion_Completion model, in
# THIS repo's g1_29dof_yam.urdf (yam jaws + F/T sensor):
#   ./run_replay_zarr_viz.sh out.zarr.zip --motion-completion
# It needs a 3pt dataset (convert with --hands both_head), a Motion_Completion checkout, its
# g1_mosyn conda env and a downloaded checkpoint; a missing piece disables the panel with a
# reason instead of failing the launch. WITHOUT the flag this viewer is byte-for-byte the
# gripper-only replay it has always been — no humanoid, no model process, no extra panel.
# Motion completion (the model, its env and its checkpoint) is NOT part of the FLoRe
# release; env overrides MOTION_COMPLETION_{DIR,CKPT,PY,STEP}.
#
# The panel's `start from` decides the absolute placement, and it is not a display knob: the
# commanded hand height is a CONDITIONED channel, so placing the take too low makes the model
# crouch. `home` (default) starts the completion at the G1's 29-joints-zero pose and needs no
# guessed number — the humanoid stands on the grid. `absolute` keeps the take's world poses and
# needs the z lift = how high the libsurvive origin sat above the floor; it puts the hands back
# on the recorded eef path, so it is the fidelity view. Pin the lift if you measured it.
#
# Extra args after the path are forwarded to replay_zarr_viz.py
# (--episode, --port, --cam-fov-deg, --motion-completion, --mc-urdf).
set -eo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$REPO_ROOT/.venv"
PY="${REPLAY_PY:-$VENV/bin/python}"
PORT="${PORT:-8084}"

if [ "$#" -lt 1 ]; then
  echo "usage: ./run_replay_zarr_viz.sh <path.zarr.zip> [extra args]" >&2
  echo "   e.g. ./run_replay_zarr_viz.sh session_XXX.zarr.zip --episode 0" >&2
  exit 2
fi

# No PYTHONPATH=humi_ik here: the zarr tool never imports the IK (it replays the
# file as-is). It does need the processing/ dir for the JpegXl codec, which
# replay_zarr_viz.py adds to sys.path internally.
echo "[run] zarr replay on http://localhost:$PORT  (Ctrl-C to stop)"
"$PY" "$REPO_ROOT/replay/replay_zarr_viz.py" --port "$PORT" "$@"
