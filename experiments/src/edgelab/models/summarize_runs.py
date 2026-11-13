"""Print EXP-002 result tables straight from MLflow (numbers in results.md are pasted from this output).

uv run python -m edgelab.models.summarize_runs --dataset hockey-fight
"""

import argparse

import mlflow
import numpy as np

from edgelab.tracking.mlflow_run import DEFAULT_TRACKING_URI

COLUMNS = ["test_video_acc", "test_video_acc_ci_lo", "test_video_acc_ci_hi", "test_clip_acc", "test_clip_acc_ci_lo",
           "test_clip_acc_ci_hi", "test_clip_f1", "test_clip_auc", "epochs_run", "best_epoch", "mean_epoch_seconds"]


def finished_runs(dataset: str):
    mlflow.set_tracking_uri(DEFAULT_TRACKING_URI)
    runs = mlflow.search_runs(experiment_names=["EXP-002"], filter_string=f"params.dataset = '{dataset}'")
    runs = runs[runs["status"] == "FINISHED"]
    runs = runs[~runs["params.seed"].isin(["99"])]  # smoke run
    return runs.dropna(subset=["metrics.test_clip_acc"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="hockey-fight")
    args = parser.parse_args()
    runs = finished_runs(args.dataset).sort_values(["params.protocol", "params.seed"])
    print("| protocol | seed | run id | " + " | ".join(COLUMNS) + " |")
    print("|" + "---|" * (len(COLUMNS) + 3))
    for _, r in runs.iterrows():
        values = [f"{r[f'metrics.{c}']:.3f}" if c not in ("epochs_run", "best_epoch") else f"{r[f'metrics.{c}']:.0f}"
                  for c in COLUMNS]
        print(f"| {r['params.protocol']} | {r['params.seed']} | `{r['run_id']}` | " + " | ".join(values) + " |")

    by = {p: runs[runs["params.protocol"] == p].set_index("params.seed") for p in ("video", "leaky")}
    for protocol, frame in by.items():
        if len(frame):
            acc = frame["metrics.test_clip_acc"]
            print(f"\n{protocol}: clip acc mean {acc.mean():.4f} sd {acc.std(ddof=1) if len(acc) > 1 else float('nan'):.4f}"
                  f" (n={len(acc)})")
    common = sorted(set(by["video"].index) & set(by["leaky"].index))
    if common:
        gaps = np.array([by["leaky"].loc[s, "metrics.test_clip_acc"] - by["video"].loc[s, "metrics.test_clip_acc"]
                         for s in common])
        print(f"H-B gap (leaky - video, clip acc) per seed {dict(zip(common, gaps.round(4)))}; mean {gaps.mean():.4f}")


if __name__ == "__main__":
    main()
