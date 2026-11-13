"""Separates 'missed people' from 'worse poses': which frames does each pose source score at all?

A frame is covered if at least one person has a segment whose score lands on it; uncovered frames get the
'most normal' score. Coverage-matched AUC compares both sources on the frames both cover (exploratory diagnostic).

    uv run python -m edgelab.pose.coverage --a artifacts/EXP-001/released --b artifacts/EXP-001/yolov8n-pose-botsort
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from edgelab.pose.stgnf_eval import SEG_LEN, STGNF_ROOT, assemble_clip_scores

GT_DIR = STGNF_ROOT / "data/ShanghaiTech/gt/test_frame_mask"


def covered_frames(metadata: np.ndarray, gt_dir: Path, frame_offset: int = SEG_LEN // 2) -> list[np.ndarray]:
    masks = []
    for gt_file in sorted(gt_dir.glob("*.npy")):
        scene_id, clip_id = (int(x) for x in gt_file.stem.split("_"))
        mask = np.zeros(len(np.load(gt_file)), dtype=bool)
        frames = metadata[(metadata[:, 0] == scene_id) & (metadata[:, 1] == clip_id), 3] + frame_offset
        mask[frames[frames < len(mask)]] = True
        masks.append(mask)
    return masks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", type=Path, required=True, help="evaluate.py output dir (has segments.pkl)")
    parser.add_argument("--b", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    sources = {}
    for name, folder in (("a", args.a), ("b", args.b)):
        with open(folder / "segments.pkl", "rb") as f:
            segments = pickle.load(f)
        _, labels, scores = assemble_clip_scores(segments.normality, segments.metadata, GT_DIR)
        sources[name] = (np.concatenate(labels), np.concatenate(scores),
                         np.concatenate(covered_frames(segments.metadata, GT_DIR)))
    labels = sources["a"][0]
    report = {"a": str(args.a), "b": str(args.b), "smoothing": "none", "score_frame": "middle"}
    for name, (_, scores, covered) in sources.items():
        report[f"coverage_anomalous_{name}"] = float(covered[labels == 1].mean())
        report[f"coverage_normal_{name}"] = float(covered[labels == 0].mean())
        report[f"auc_all_frames_{name}"] = float(roc_auc_score(labels, scores))
    both = sources["a"][2] & sources["b"][2]
    report["frames_covered_by_both"] = int(both.sum())
    for name, (_, scores, _) in sources.items():
        report[f"auc_on_frames_covered_by_both_{name}"] = float(roc_auc_score(labels[both], scores[both]))
    out = args.out or args.b / "coverage_vs_a.json"
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
