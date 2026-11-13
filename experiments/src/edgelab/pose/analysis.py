"""Per-scene / per-video breakdown of two score files, plus per-video mean luma (to look for a dark-video pattern).

    uv run python -m edgelab.pose.analysis --a <frame_scores_A.npz> --b <frame_scores_B.npz> \
        --frames-root <testing/frames> --out <csv>
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score


def load_scores(npz_path: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    data = np.load(npz_path)
    return {str(n): (data[f"label_{n}"], data[f"score_{n}"]) for n in data["names"]}


def mean_luma(frame_dir: Path, every: int = 10) -> float:
    frames = sorted(frame_dir.glob("*.jpg"))[::every]
    return float(np.mean([cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2YCrCb)[..., 0].mean() for f in frames]))


def per_video_table(a: dict, b: dict, frames_root: Path) -> pd.DataFrame:
    rows = []
    for name, (labels, score_a) in a.items():
        score_b = b[name][1]
        both = len(np.unique(labels)) == 2
        rows.append({"video": name, "scene": name.split("_")[0], "frames": len(labels),
                     "anomalous_frac": float(labels.mean()), "luma": mean_luma(frames_root / name),
                     "auc_a": roc_auc_score(labels, score_a) if both else np.nan,
                     "auc_b": roc_auc_score(labels, score_b) if both else np.nan})
    table = pd.DataFrame(rows).set_index("video")
    table["auc_drop"] = table["auc_a"] - table["auc_b"]
    return table


def per_scene_auc(scores: dict) -> pd.Series:
    by_scene: dict[str, list] = {}
    for name, (labels, score) in scores.items():
        by_scene.setdefault(name.split("_")[0], []).append((labels, score))
    out = {}
    for scene, items in sorted(by_scene.items()):
        labels = np.concatenate([l for l, _ in items])
        score = np.concatenate([s for _, s in items])
        out[scene] = roc_auc_score(labels, score) if len(np.unique(labels)) == 2 else np.nan
    return pd.Series(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", type=Path, required=True)
    parser.add_argument("--b", type=Path, required=True)
    parser.add_argument("--frames-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    a, b = load_scores(args.a), load_scores(args.b)
    table = per_video_table(a, b, args.frames_root)
    table.to_csv(args.out)
    scenes = pd.DataFrame({"auc_a": per_scene_auc(a), "auc_b": per_scene_auc(b),
                           "videos": table.groupby("scene").size(),
                           "luma": table.groupby("scene")["luma"].mean()})
    scenes.to_csv(args.out.with_name(args.out.stem + "_scenes.csv"))
    print(scenes.round(3))
    valid = table.dropna(subset=["auc_drop"])
    rho, p = stats.spearmanr(valid["luma"], valid["auc_drop"])
    print(f"videos with both classes: {len(valid)}; Spearman(luma, AUC drop) = {rho:.3f} (p = {p:.3f})")
    print(f"luma quantiles over 107 videos: {np.quantile(table['luma'], [0, 0.1, 0.5, 0.9, 1]).round(1)}")


if __name__ == "__main__":
    main()
