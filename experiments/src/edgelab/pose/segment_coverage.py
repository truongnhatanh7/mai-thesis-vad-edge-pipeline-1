"""Label-free coverage of unlabeled videos: share of ALL frames that STG-NF would score (EXP-014, choice of alpha).

    uv run --extra pose python -m edgelab.pose.segment_coverage --cache <detcache dir of held-out training videos> \
        --c 0.25 --out coverage.json

STG-NF needs no network here, only its segmenting rule (third_party/STG-NF utils/pose_utils.py, test pass):
a track with T poses (sorted frame ids) is cut into windows of 24 consecutive poses, window k = poses k .. k+23 for
k = 0 .. T-25 (their loop stops one window early); a window is used when at least 22 of its 24 frames lie in
[start frame, start frame + 24) (at most 2 frames missing). The window's score lands on frame start + 12 (paper
variant) or start + 23 (live variant). A frame is covered when some track has such a window landing on it.
This is the same `covered` mask score_dir.py builds from the real segments, but without running STG-NF.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from edgelab.pose.detcache import load_detections
from edgelab.pose.replay import replay_video

SEG_LEN = 24
MISSING_ALLOWED = 2
OFFSETS = {"paper": SEG_LEN // 2, "live": SEG_LEN - 1}


def scored_frames(track_frames: list[int], offset: int, seg_len: int = SEG_LEN, missing_allowed: int = MISSING_ALLOWED) -> set[int]:
    """Frames that receive a score from the windows of one track (frame ids, any order)."""
    keys = sorted(track_frames)
    frames = set()
    for start in range(max(0, len(keys) - seg_len)):  # STG-NF: ceil((T - seg_len) / 1) windows
        window = keys[start:start + seg_len]
        inside = sum(1 for k in window if keys[start] <= k < keys[start] + seg_len)
        if inside >= seg_len - missing_allowed:
            frames.add(keys[start] + offset)
    return frames


def video_coverage(tracks: dict[int, list[int]], n_frames: int, offset: int) -> np.ndarray:
    """Boolean mask over the video's frames: scored by at least one track."""
    mask = np.zeros(n_frames, dtype=bool)
    for frames in tracks.values():
        for frame in scored_frames(frames, offset):
            if frame < n_frames:
                mask[frame] = True
    return mask


def cache_coverage(cache: Path, c: float, tracker: str = "botsort") -> dict:
    """Replay every video of a detector cache at gate c; return per-variant coverage over all frames of all videos."""
    covered = {v: 0 for v in OFFSETS}
    total = 0
    per_video = {}
    for path in sorted(cache.glob("*.npz")):
        frames, data = load_detections(path)
        poses, _ = replay_video(frames, data["frame_ids"], c, tracker)
        tracks: dict[int, list[int]] = {}
        for pose in poses:
            tracks.setdefault(pose.track_id, []).append(pose.frame)
        n = len(frames)
        per_video[path.stem] = {}
        for variant, offset in OFFSETS.items():
            hit = int(video_coverage(tracks, n, offset).sum())
            covered[variant] += hit
            per_video[path.stem][variant] = hit / n
        total += n
    return {"videos": len(per_video), "frames": total, "c": c,
            **{f"coverage_{v}": covered[v] / total for v in OFFSETS}, "per_video": per_video}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--c", type=float, default=0.25)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = cache_coverage(args.cache, args.c)
    args.out.write_text(json.dumps(result, indent=1))
    print(json.dumps({k: v for k, v in result.items() if k != "per_video"}))


if __name__ == "__main__":
    main()
