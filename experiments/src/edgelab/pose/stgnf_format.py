"""STG-NF pose JSON format (what their dataset.py reads), and conversion from tracked YOLO-pose detections.

File `<scene>_<clip>_<source>_tracked_person.json`:
    {"<person_id>": {"<frame, 4-digit, 0-based>": {"keypoints": [x, y, conf] * 17 (COCO-17 order, pixels),
                                                   "scores": <detection score>}}}
STG-NF itself adds the neck point (COCO-17 -> 18) when loading, so we store plain COCO-17.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

NUM_KEYPOINTS = 17


@dataclass
class TrackedPose:
    frame: int
    track_id: int
    keypoints: np.ndarray  # (17, 3): x, y in pixels of the original frame, keypoint confidence
    score: float  # detection (box) confidence


def frame_key(frame: int) -> str:
    return f"{frame:04d}"


def to_stgnf_dict(poses: list[TrackedPose]) -> dict:
    clip: dict[str, dict[str, dict]] = {}
    for pose in sorted(poses, key=lambda p: (p.track_id, p.frame)):
        if pose.keypoints.shape != (NUM_KEYPOINTS, 3):
            raise ValueError(f"expected (17, 3) keypoints, got {pose.keypoints.shape}")
        clip.setdefault(str(pose.track_id), {})[frame_key(pose.frame)] = {
            "keypoints": [float(v) for v in pose.keypoints.reshape(-1)],
            "scores": float(pose.score),
        }
    return clip


def write_stgnf_json(poses: list[TrackedPose], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_stgnf_dict(poses)))
    return path


def read_person_track(clip: dict, person_id: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(frames, keypoints (T, 17, 3), scores) for one person, frames sorted the way STG-NF sorts them."""
    person = clip[person_id]
    if isinstance(person, list):  # some released files store a list of single-frame dicts
        merged = {}
        for part in person:
            merged.update(part)
        person = merged
    keys = sorted(person.keys())
    frames = np.array([int(k) for k in keys])
    keypoints = np.stack([np.asarray(person[k]["keypoints"], dtype=float).reshape(-1, 3) for k in keys])
    scores = np.array([float(person[k]["scores"]) for k in keys])
    return frames, keypoints, scores


def read_stgnf_json(path: str | Path) -> dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    clip = json.loads(Path(path).read_text())
    return {int(pid): read_person_track(clip, pid) for pid in clip}


def poses_from_stgnf_json(path: str | Path) -> list[TrackedPose]:
    poses = []
    for pid, (frames, keypoints, scores) in read_stgnf_json(path).items():
        poses += [TrackedPose(int(f), pid, k, float(s)) for f, k, s in zip(frames, keypoints, scores)]
    return poses


def poses_from_ultralytics_result(result, frame: int) -> list[TrackedPose]:
    """One ultralytics `Results` (pose model, from model.track) -> poses with a confirmed track id."""
    boxes = result.boxes
    if boxes is None or boxes.id is None or result.keypoints is None:
        return []
    ids = boxes.id.int().cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    keypoints = result.keypoints.data.cpu().numpy()  # (N, 17, 3)
    return [TrackedPose(frame, int(i), k.astype(float), float(c)) for i, c, k in zip(ids, confs, keypoints)]
