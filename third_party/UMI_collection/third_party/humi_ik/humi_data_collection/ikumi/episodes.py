from __future__ import annotations

import json
from pathlib import Path

from .config import IKConfig
from .keypoints import (
    HandRootFootKeyPoints,
    load_target_from_frame,
)
from .solution import IKSolution


def load_target_episode(
    json_path: Path,
) -> list[HandRootFootKeyPoints]:
    """Load an episode containing target keypoint sequences.

    Returns:
        List of target keypoints for each frame in the episode.
    """
    with json_path.open("r") as f:
        data = json.load(f)

    if isinstance(data, dict) and "episode" in data:
        data = data["episode"]
    elif not isinstance(data, list):
        raise ValueError(
            "Expected a list of frames or a dict with an 'episode' key."
        )

    return [load_target_from_frame(frame) for frame in data]


def save_ik_episode(
    episode: list[IKSolution],
    ik_config: IKConfig,
    output_path: Path,
) -> None:
    """Save a solved IK episode to a JSON file, extending targets with solutions."""
    json_data = [sol.to_frame_dict() for sol in episode]
    out_dict = {
        "mjcf_path": str(ik_config.mjcf_path.absolute()),
        "episode": json_data,
    }
    with output_path.open("w") as f:
        json.dump(out_dict, f, indent=4)


def load_ik_episode(
    json_path: Path,
) -> list[IKSolution]:
    """Load an episode containing full IK solutions."""
    with json_path.open("r") as f:
        data = json.load(f)

    if isinstance(data, dict) and "episode" in data:
        data = data["episode"]
    elif not isinstance(data, list):
        raise ValueError(
            "Expected a list of frames or a dict with an 'episode' key."
        )

    return [IKSolution.from_frame_dict(frame) for frame in data]


__all__ = [
    "load_target_episode",
    "save_ik_episode",
    "load_ik_episode",
]
