"""Fill gaps in STG-NF pose JSON (EXP-009 plan §1): B = interpolate between two detections, P = extrapolate forward.

    uv run --extra pose python -m edgelab.pose.fill --in-dir <pose dir> --out-dir <dir> --mode B --gap 8

Data shapes (the STG-NF JSON, read as plain dicts):
    clip  = {"<person id>": track}
    track = {"<frame, 4 digits>": {"keypoints": [x, y, conf] * 17, "scores": <detection score>}}

The filled JSON keeps exactly this format, so STG-NF reads it unchanged. Which poses were made up is NOT written
into it: it goes to a separate sidecar file per video (in `prov_dir`):
    {"<person id>": {"<frame>": age}}     one entry per predicted pose, age = frames since the last detection
A pose that is not in the sidecar was detected (age 0).
"""

import argparse
import json
from pathlib import Path

import numpy as np

from edgelab.pose.stgnf_eval import SEG_LENS

DAMPING = 0.8
SUFFIX = "tracked_person.json"
PROVENANCE_SUFFIX = ".provenance.json"


def _as_array(pose: dict) -> np.ndarray:
    return np.asarray(pose["keypoints"], dtype=float).reshape(-1, 3)


def key_format(track: dict):
    """frame -> key text in the track's own style. Released files use 3 digits ('068') in one video; STG-NF sorts the
    keys as TEXT, so a new key must have the same width as the existing ones or the order of the poses would change."""
    width = min(len(k) for k in track) if track else 4
    return lambda frame: f"{frame:0{width}d}"


def _make_pose(keypoints: np.ndarray, score: float) -> dict:
    return {"keypoints": [float(v) for v in keypoints.reshape(-1)], "scores": float(score)}


def fill_B(track: dict, max_gap: int) -> tuple[dict, dict]:
    """Linear interpolation inside every gap of at most `max_gap` missing frames between two detected frames.

    Interpolates x, y and keypoint confidence (and the detection score) of every keypoint. Longer gaps stay empty.
    Returns (filled track, {frame key: age}) where age = frames since the earlier detection.
    """
    filled, predicted = dict(track), {}
    key = key_format(track)
    frames = sorted(int(k) for k in track)
    for before, after in zip(frames, frames[1:]):
        gap = after - before - 1
        if gap < 1 or gap > max_gap:
            continue
        pose_before, pose_after = track[key(before)], track[key(after)]
        kp_before, kp_after = _as_array(pose_before), _as_array(pose_after)
        for frame in range(before + 1, after):
            t = (frame - before) / (after - before)
            keypoints = (1 - t) * kp_before + t * kp_after
            score = (1 - t) * float(pose_before["scores"]) + t * float(pose_after["scores"])
            filled[key(frame)] = _make_pose(keypoints, score)
            predicted[key(frame)] = frame - before
    return dict(sorted(filled.items())), predicted


def fill_P(track: dict, max_gap: int, damping: float = DAMPING, last_frame: int | None = None) -> tuple[dict, dict]:
    """Causal forward prediction after every detected frame, at most `max_gap` frames, stopping at the next detection.

    Step k (k = 1, 2, ...) after the detected frame: position += velocity * damping**k, where velocity = (x, y) change
    per frame between the last two detected frames of the track (zero if there is only one). Keypoint confidence and
    detection score = those of the last detection * damping**k. `last_frame`: never predict beyond the video's end.
    """
    filled, predicted = dict(track), {}
    key = key_format(track)
    frames = sorted(int(k) for k in track)
    for i, frame in enumerate(frames):
        pose = track[key(frame)]
        keypoints = _as_array(pose)
        velocity = np.zeros((len(keypoints), 2))
        if i > 0:
            earlier = _as_array(track[key(frames[i - 1])])
            velocity = (keypoints[:, :2] - earlier[:, :2]) / (frame - frames[i - 1])
        next_detection = frames[i + 1] if i + 1 < len(frames) else None
        position = keypoints[:, :2].copy()
        for k in range(1, max_gap + 1):
            target = frame + k
            if next_detection is not None and target >= next_detection:
                break
            if last_frame is not None and target > last_frame:
                break
            position = position + velocity * damping ** k
            predicted_keypoints = np.column_stack([position, keypoints[:, 2] * damping ** k])
            filled[key(target)] = _make_pose(predicted_keypoints, float(pose["scores"]) * damping ** k)
            predicted[key(target)] = k
    return dict(sorted(filled.items())), predicted


def fill_clip(clip: dict, mode: str, max_gap: int, last_frame: int | None = None) -> tuple[dict, dict]:
    """Fill every track of one video. mode 'B' or 'P'. Returns (filled clip, {person id: {frame key: age}})."""
    filled, provenance = {}, {}
    for person_id, track in clip.items():
        track = merge_parts(track)
        if mode == "B":
            filled[person_id], predicted = fill_B(track, max_gap)
        elif mode == "P":
            filled[person_id], predicted = fill_P(track, max_gap, last_frame=last_frame)
        else:
            raise ValueError(f"mode must be 'B' or 'P', got {mode!r}")
        if predicted:
            provenance[person_id] = predicted
    return filled, provenance


def merge_parts(track):
    """Some released files store a track as a list of single-frame dicts; STG-NF merges them, so do we."""
    if isinstance(track, list):
        merged = {}
        for part in track:
            merged.update(part)
        return merged
    return track


def video_of_file(name: str, videos) -> str | None:
    """The video (longest name in `videos`) that a pose file name starts with, e.g. '01_0014_x_tracked_person.json' -> '01_0014'."""
    found = [v for v in videos if name.startswith(v + "_")]
    return max(found, key=len) if found else None


def provenance_path(prov_dir: Path, pose_file: Path | str) -> Path:
    return Path(prov_dir) / (Path(pose_file).name.removesuffix(".json") + PROVENANCE_SUFFIX)


def fill_dir(in_dir: Path, out_dir: Path, mode: str, max_gap: int, prov_dir: Path | None = None,
             n_frames: dict[str, int] | None = None) -> dict:
    """Fill every *tracked_person.json of in_dir -> out_dir (same file names), sidecars to prov_dir.

    prov_dir defaults to a sibling folder `<out_dir name>_prov` (so STG-NF never sees the sidecars).
    n_frames: {video: frame count}; P stops at the video's last frame (B never leaves the span of detections).
    Returns totals: detected poses, predicted poses.
    """
    out_dir, prov_dir = Path(out_dir), Path(prov_dir) if prov_dir else Path(out_dir).with_name(Path(out_dir).name + "_prov")
    out_dir.mkdir(parents=True, exist_ok=True)
    prov_dir.mkdir(parents=True, exist_ok=True)
    detected = predicted = 0
    for path in sorted(Path(in_dir).glob(f"*{SUFFIX}")):
        last_frame = None
        if n_frames is not None:
            video = video_of_file(path.name, n_frames)
            if video is None:
                raise ValueError(f"no frame count for pose file {path.name}")
            last_frame = n_frames[video] - 1
        clip = json.loads(path.read_text())
        filled, provenance = fill_clip(clip, mode, max_gap, last_frame)
        (out_dir / path.name).write_text(json.dumps(filled))
        provenance_path(prov_dir, path).write_text(json.dumps(provenance))
        n_predicted = sum(len(p) for p in provenance.values())
        predicted += n_predicted
        detected += sum(len(merge_parts(t)) for t in clip.values())
    stats = {"mode": mode, "max_gap": max_gap, "detected_poses": detected, "predicted_poses": predicted}
    (prov_dir / "fill_stats.json").write_text(json.dumps(stats))
    return stats


def segment_predicted_shares(metadata: np.ndarray, pose_dir: Path, prov_dir: Path | None, dataset: str = "ShanghaiTech") -> np.ndarray:
    """Share of predicted poses in each scored 24-frame (UBnormal: 16) window; one value per row of segment metadata.

    metadata rows = [scene, clip, person, start_frame] as returned by score_segments. A window is the seg_len poses of the
    person's track from start_frame on (the same poses STG-NF cut). prov_dir None = nothing was filled (all zero).
    """
    seg_len = SEG_LENS[dataset]
    if prov_dir is None:
        return np.zeros(len(metadata))
    index = {}  # (scene, clip) -> (pose file, sidecar file)
    for path in Path(pose_dir).glob(f"*{SUFFIX}"):
        index[_scene_clip(path.name, dataset)] = path
    tracks: dict[tuple, dict] = {}
    shares = np.zeros(len(metadata))
    for row, (scene, clip, person, start) in enumerate(metadata):
        key = (str(int(scene)), str(clip) if dataset == "UBnormal" else str(int(clip)))
        if key not in tracks:
            path = index[key]
            clip_json = json.loads(path.read_text())
            side = json.loads(provenance_path(prov_dir, path).read_text())
            tracks[key] = {pid: (sorted(int(k) for k in merge_parts(t)), {int(f) for f in side.get(pid, {})})
                           for pid, t in clip_json.items()}
        frames, predicted = tracks[key][str(int(person))]
        first = frames.index(int(start))
        window = frames[first:first + seg_len]
        shares[row] = sum(f in predicted for f in window) / len(window)
    return shares


def _scene_clip(file_name: str, dataset: str) -> tuple[str, str]:
    """(scene, clip) strings as they appear in segment metadata, read from a pose file name the way STG-NF does."""
    if dataset == "UBnormal":
        from edgelab.pose.ubnormal import parse_video_name, video_of_pose_file
        scene, clip = parse_video_name(video_of_pose_file(file_name))
        return str(int(scene)), clip
    scene, clip = file_name.split("_")[:2]
    return str(int(scene)), str(int(clip))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=["B", "P"], required=True)
    parser.add_argument("--gap", type=int, required=True)
    args = parser.parse_args()
    print(json.dumps(fill_dir(args.in_dir, args.out_dir, args.mode, args.gap)))


if __name__ == "__main__":
    main()
