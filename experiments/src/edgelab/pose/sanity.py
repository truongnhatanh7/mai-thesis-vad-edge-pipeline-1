"""Compare two pose sources per video: tracks, poses, track lengths, keypoint confidences, usable 24-frame segments.

    uv run python -m edgelab.pose.sanity --a <released dir> --b <yolo dir> --out <csv>
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from edgelab.pose.stgnf_format import read_stgnf_json

SEG_LEN = 24


def _video_id(path: Path) -> str:
    return "_".join(path.name.split("_")[:2])


def _usable_segments(frames: np.ndarray, missing_allowed: int = 2) -> int:
    """Segments STG-NF keeps at test time (stride 1): >= 22 of 24 expected frames present."""
    frames = np.sort(frames)
    count = 0
    for start_index in range(max(0, int(np.ceil(len(frames) - SEG_LEN)))):
        start = frames[start_index]
        window = frames[start_index:start_index + SEG_LEN]
        if np.sum((window >= start) & (window < start + SEG_LEN)) >= SEG_LEN - missing_allowed:
            count += 1
    return count


def summarize_dir(pose_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(pose_dir.glob("*_tracked_person.json")):
        tracks = read_stgnf_json(path)
        lengths = [len(f) for f, _, _ in tracks.values()]
        kp_conf = np.concatenate([k[..., 2].ravel() for _, k, _ in tracks.values()]) if tracks else np.array([])
        rows.append({"video": _video_id(path), "tracks": len(tracks), "poses": int(sum(lengths)),
                     "median_track_len": float(np.median(lengths)) if lengths else 0.0,
                     "tracks_ge_24": int(sum(l >= SEG_LEN for l in lengths)),
                     "segments": int(sum(_usable_segments(f) for f, _, _ in tracks.values())),
                     "kp_conf_mean": float(kp_conf.mean()) if kp_conf.size else np.nan,
                     "kp_conf_p10": float(np.quantile(kp_conf, 0.1)) if kp_conf.size else np.nan})
    return pd.DataFrame(rows).set_index("video")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", type=Path, required=True)
    parser.add_argument("--b", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    a, b = summarize_dir(args.a), summarize_dir(args.b)
    joined = a.add_suffix("_a").join(b.add_suffix("_b"), how="inner")
    joined.to_csv(args.out)
    print(joined.sum(numeric_only=True)[[c for c in joined.columns if c.startswith(("tracks", "poses", "segments"))]])
    print(joined[["kp_conf_mean_a", "kp_conf_mean_b", "median_track_len_a", "median_track_len_b"]].median())


if __name__ == "__main__":
    main()
