"""Small helpers for the EXP-018/019 tests: synthetic STG-NF pose JSON with known gaps."""

import json
from pathlib import Path

import numpy as np


def toy_track(rng, frames, start=(300.0, 200.0), speed=(1.5, 0.5)) -> dict:
    """{frame key: {keypoints, scores}} of a walking person: 17 keypoints on a rigid body that drifts across the image."""
    body = rng.uniform(-30, 30, size=(17, 2))
    track = {}
    for frame in frames:
        centre = np.array(start) + np.array(speed) * frame + rng.normal(0, 0.3, 2)
        xy = centre + body + rng.normal(0, 0.5, size=(17, 2))
        conf = rng.uniform(0.3, 0.9, size=(17, 1))
        track[f"{frame:04d}"] = {"keypoints": [float(v) for v in np.hstack([xy, conf]).reshape(-1)],
                                 "scores": float(rng.uniform(0.5, 0.9))}
    return track


def write_toy_clip(path: Path, rng, tracks: dict[str, list[int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({pid: toy_track(rng, frames) for pid, frames in tracks.items()}))


def write_toy_dir(directory: Path, seed: int = 0) -> Path:
    """Two clips: person 1 with all frames, person 2 with a 1-frame and a 3-frame hole, a short person 3 (too short for a window)."""
    rng = np.random.default_rng(seed)
    holes = [f for f in range(5, 120) if f not in (40, 80, 81, 82)]
    write_toy_clip(directory / "01_001_alphapose_tracked_person.json", rng,
                   {"1": list(range(0, 100)), "2": holes, "3": list(range(10, 30))})
    write_toy_clip(directory / "02_005_alphapose_tracked_person.json", rng, {"1": list(range(3, 90))})
    return directory
