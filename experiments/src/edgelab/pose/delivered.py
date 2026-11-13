"""Delivered scores (EXP-004 plan §3.1 step 2): what a live system that skipped frames would actually output.

STG-NF ran on a re-indexed pose folder (replay.py `reindex=True`): its segments live in processed-frame numbers 0..K-1.
Here each processed frame's score is placed back on the real timeline and held until the next processed frame, then
the live variant's causal half-Gaussian runs on the real-frame series, and the AUC is taken against the real-frame GT.
"""

from pathlib import Path

import numpy as np

from edgelab.pose.smoothing import SMOOTHERS
from edgelab.pose.stgnf_eval import _clips, score_frame_offsets


def processed_normality(normality: np.ndarray, metadata: np.ndarray, scene, clip, n_processed: int,
                        frame_offset: int) -> np.ndarray:
    """(n_processed,) normality of the least-normal person on each processed frame; +inf = nobody scored there.

    Same placement rule as stgnf_eval.clip_normality, but on the processed-frame axis. scene / clip may be ints
    (ShanghaiTech) or STG-NF's strings (UBnormal); they are compared as strings.
    """
    rows = (metadata[:, 0].astype(str) == str(scene)) & (metadata[:, 1].astype(str) == str(clip))
    persons = metadata[:, 2].astype(int)
    out = np.full(n_processed, np.inf)
    for person in sorted(set(persons[rows])):
        person_rows = np.flatnonzero(rows & (persons == person))
        frames = metadata[person_rows, 3].astype(int) + frame_offset
        inside = frames < n_processed
        np.minimum.at(out, frames[inside], normality[person_rows][inside])
    return out


def hold_last(per_processed: np.ndarray, frame_map: np.ndarray, n_frames: int) -> np.ndarray:
    """Real-frame series: frame f gets the value of the latest processed frame at or before f (+inf before the first)."""
    latest = np.searchsorted(frame_map, np.arange(n_frames), side="right") - 1
    return np.where(latest >= 0, per_processed[np.maximum(latest, 0)], np.inf)


def delivered_variant_scores(normality: np.ndarray, metadata: np.ndarray, frame_maps: dict[str, np.ndarray],
                             gt_dir: Path, offset_name: str = "end", smoother_name: str = "causal_half_gaussian",
                             dataset: str = "ShanghaiTech", clips: list[str] | None = None):
    """-> (names, labels, anomaly scores per real frame). Defaults = the live variant of EXP-001/EXP-003.

    dataset / clips / gt_dir as in stgnf_eval.clip_normality (UBnormal: segment length 16, so the live score frame is 15).

    Frames with no output yet, or with nobody scored, get the dataset's most normal score (same rule as variant_scores).
    """
    names, labels, held = [], [], []
    frame_offset = score_frame_offsets(dataset)[offset_name]
    for name, scene, clip, gt in _clips(dataset, gt_dir, clips):
        frame_map = np.asarray(frame_maps[name])
        per_processed = processed_normality(normality, metadata, scene, clip, len(frame_map), frame_offset)
        names.append(name)
        labels.append(gt)
        held.append(hold_last(per_processed, frame_map, len(gt)))
    finite = np.concatenate(held)
    finite = finite[np.isfinite(finite)]
    if len(finite) == 0:
        raise ValueError("no scored segment in any video: nothing to deliver")
    most_normal = finite.max()
    scores = [SMOOTHERS[smoother_name](-np.where(np.isfinite(h), h, most_normal)) for h in held]
    return names, labels, scores
