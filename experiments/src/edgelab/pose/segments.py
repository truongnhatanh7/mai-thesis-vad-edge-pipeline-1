"""Training segments of STG-NF, built from pose JSON as arrays (same windows as third_party/STG-NF dataset.py).

STG-NF cuts every track (a person's poses, sorted by frame) into windows of `seg_len` consecutive POSES every `stride`
poses, and keeps a window if at most 2 of the frames start_key .. start_key + seg_len - 1 are missing
(utils/pose_utils.is_seg_continuous). tests/test_pose_segments.py checks that this module cuts exactly the same windows.

We need the same windows plus some frames of context around them, because EXP-018 drops frames of a window and
predicts them from the detections *before* the window. So a bank keeps all poses once (`xy`) and, per window, a grid:

    grid[w, g] = row in `xy` of the pose at frame (start_key - ctx + g), or -1 if the track has no pose on that frame
    g = 0 .. ctx - 1                 context before the window
    g = ctx .. ctx + seg_len + 1     the window's own frames (+ 2 spare columns: a window may spread over seg_len + 2 frames)

Only x, y are kept (the STG-NF model reads nothing else).
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from edgelab.pose.fill import SUFFIX, merge_parts

SPARE = 2  # a kept window has >= seg_len - 2 of its first seg_len frames, so its poses lie within seg_len + 2 frames
MISSING_TOLERANCE = 2


@dataclass
class SegmentBank:
    xy: np.ndarray  # (P, 17, 2) float32, every pose of every track
    grid: np.ndarray  # (N, ctx + seg_len + SPARE) int32, see module docstring
    file_id: np.ndarray  # (N,) int32, index into `files`
    files: list[str]
    ctx: int
    seg_len: int

    def __len__(self):
        return len(self.grid)

    def subset(self, files_wanted: set[int]) -> np.ndarray:
        """Window indices that come from the given file ids."""
        return np.flatnonzero(np.isin(self.file_id, list(files_wanted)))


def read_tracks(path: Path) -> list[tuple[np.ndarray, np.ndarray]]:
    """[(frames (n,) ints sorted, keypoints (n, 17, 3))] of one pose file, persons in numeric id order (as STG-NF)."""
    clip = json.loads(Path(path).read_text())
    tracks = []
    for person in sorted(clip, key=int):
        poses = merge_parts(clip[person])
        keys = sorted(poses, key=int)
        frames = np.array([int(k) for k in keys])
        keypoints = np.stack([np.asarray(poses[k]["keypoints"], dtype=np.float64).reshape(-1, 3) for k in keys])
        tracks.append((frames, keypoints))
    return tracks


def window_starts(frames: np.ndarray, seg_len: int, stride: int) -> np.ndarray:
    """Pose indices where STG-NF starts a window of this (sorted) track and keeps it."""
    n = len(frames)
    if n <= seg_len:
        return np.zeros(0, dtype=int)
    starts = np.arange(int(np.ceil((n - seg_len) / stride))) * stride
    keep = []
    for start in starts:
        inside = frames[start:start + seg_len]
        keep.append(np.count_nonzero((inside >= frames[start]) & (inside < frames[start] + seg_len)) >= seg_len - MISSING_TOLERANCE)
    return starts[np.array(keep, dtype=bool)]


def build_bank(pose_dir: Path, seg_len: int, stride: int, ctx: int = 16, files: list[Path] | None = None) -> SegmentBank:
    files = files if files is not None else sorted(Path(pose_dir).glob(f"*{SUFFIX}"))
    xy_parts, grids, file_ids, base = [], [], [], 0
    width = ctx + seg_len + SPARE
    for file_index, path in enumerate(files):
        for frames, keypoints in read_tracks(path):
            starts = window_starts(frames, seg_len, stride)
            if len(starts) == 0:
                continue
            wanted = frames[starts][:, None] - ctx + np.arange(width)[None, :]  # (S, width) frame numbers
            at = np.searchsorted(frames, wanted).clip(max=len(frames) - 1)
            grids.append(np.where(frames[at] == wanted, at + base, -1).astype(np.int32))
            file_ids.append(np.full(len(starts), file_index, dtype=np.int32))
            xy_parts.append(keypoints[:, :, :2].astype(np.float32))
            base += len(frames)
    if not grids:
        raise ValueError(f"no training window in {pose_dir}")
    return SegmentBank(np.concatenate(xy_parts), np.concatenate(grids), np.concatenate(file_ids),
                       [p.name for p in files], ctx, seg_len)
