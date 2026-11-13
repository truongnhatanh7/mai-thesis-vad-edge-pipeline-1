"""Deadline replay for one (P, G, O, B, light) cell (EXP-004 plan §3.1 step 2), on the host.

    uv run --extra pose python -m edgelab.pipeline.deadline_replay --cache <EXP-003 detcache dir> \
        --trace run1.csv run2.csv ... --c 0.25 --out <dir> --level L3 --enhance E0 --label O1-B2

timing trace (VM csv) -> block-bootstrap service times -> keep_mask per video -> detection cache replayed with the mask
(processed frames re-indexed 0..K-1) -> frozen STG-NF -> scores held on the real frames -> delivered AUC (live variant).
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from edgelab.metrics.scores import bootstrap_ci_over_videos, frame_level_auc
from edgelab.pipeline.deadline_sim import keep_masks_for_videos
from edgelab.pose.delivered import delivered_variant_scores
from edgelab.pose.score_dir import DATASET_DEFAULTS
from edgelab.pose.replay import FRAME_MAPS_FILE, replay_dir
from edgelab.pose.stgnf_eval import score_segments
from edgelab.tracking.mlflow_run import tracked_run

BLOCK = 24


def load_trace(csv_files: list[Path], column: str = "service_ms") -> np.ndarray:
    """Pool the per-frame times of several repetitions. Each video is cut to whole 24-frame blocks first,
    so a bootstrap block never mixes two videos."""
    parts = []
    for path in csv_files:
        for _, video in pd.read_csv(path, usecols=["video", column]).groupby("video", sort=False):
            values = video[column].to_numpy(float)
            parts.append(values[: len(values) // BLOCK * BLOCK])
    return np.concatenate(parts)


def cache_frame_counts(cache: Path, videos: list[str] | None = None) -> dict[str, int]:
    files = sorted(cache.glob("*.npz")) if videos is None else [cache / f"{v}.npz" for v in sorted(videos)]
    return {p.stem: len(np.load(p)["frame_ids"]) for p in files}


def run_cell(cache: Path, trace_ms: np.ndarray, c: float, out: Path, config: dict, fps: float = 24.0, seed: int = 0,
             n_resamples: int = 2000, threads: int | None = None, tracking_uri: str | None = None,
             gt_dir: Path | None = None, dataset: str = "ShanghaiTech", videos: list[str] | None = None,
             experiment: str | None = None, checkpoint: Path | None = None) -> dict:
    """videos: only these videos (default: every file in the cache; UBnormal: its 211 test videos).
    dataset picks the checkpoint, GT, segment length and pose-file naming; experiment = MLflow name (EXP-004 / EXP-006)."""
    default_checkpoint, default_gt_dir = DATASET_DEFAULTS[dataset]
    checkpoint, gt_dir = checkpoint or default_checkpoint, gt_dir or default_gt_dir
    experiment = experiment or ("EXP-004" if dataset == "ShanghaiTech" else "EXP-006")
    if videos is None and dataset == "UBnormal":
        from edgelab.pose.ubnormal import load_test_list
        videos = load_test_list()
    out.mkdir(parents=True, exist_ok=True)
    counts = cache_frame_counts(cache, videos)
    masks = keep_masks_for_videos(counts, trace_ms, fps, seed)
    np.savez_compressed(out / "keep_masks.npz", **masks)
    drop_rate = 1 - sum(int(m.sum()) for m in masks.values()) / sum(counts.values())

    pose_dir = out / "poses"
    replay_dir(cache, c, pose_dir, keep_masks=masks, reindex=True, videos=videos, dataset=dataset)
    segments = score_segments(pose_dir, checkpoint, device="cpu", threads=threads, dataset=dataset)
    frame_maps = dict(np.load(pose_dir / FRAME_MAPS_FILE))
    names, labels, scores = delivered_variant_scores(segments.normality, segments.metadata, frame_maps, gt_dir,
                                                         dataset=dataset, clips=videos)
    auc = frame_level_auc(labels, scores)
    low, high = bootstrap_ci_over_videos(labels, scores, n_resamples=n_resamples)
    np.savez_compressed(out / "delivered_scores_live.npz", names=np.array(names),
                        **{f"score_{n}": s for n, s in zip(names, scores)},
                        **{f"label_{n}": y for n, y in zip(names, labels)})
    result = {"auc": auc, "ci95_low": low, "ci95_high": high, "drop_rate": drop_rate,
              "num_segments": segments.num_segments, "num_videos": len(names), "trace_frames": len(trace_ms)}
    run_config = {**config, "c": c, "fps": fps, "seed": seed, "block": BLOCK, "cache": str(cache),
                  "variant": "live (end frame, causal half-Gaussian)", "bootstrap_resamples": n_resamples}
    if dataset != "ShanghaiTech":  # ShanghaiTech configs keep their EXP-004 content
        run_config["dataset"] = dataset
    import mlflow
    with tracked_run(experiment, f"{config.get('label', 'cell')}-c{c}", run_config,
                     input_files={"checkpoint": checkpoint}, tracking_uri=tracking_uri):
        mlflow.log_metrics({k: v for k, v in result.items() if isinstance(v, (int, float))})
        mlflow.log_artifact(str(out / "delivered_scores_live.npz"))
    (out / "result.json").write_text(json.dumps({**result, "config": run_config}, indent=1))
    print(f"delivered AUC={auc * 100:.2f}  CI95=[{low * 100:.2f}, {high * 100:.2f}]  drop_rate={drop_rate:.3f}  "
          f"segments={segments.num_segments}")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True, help="EXP-003 detector cache of the light level and P")
    parser.add_argument("--trace", type=Path, nargs="+", required=True, help="timing csv(s) of this (P, G, O, B, light)")
    parser.add_argument("--c", type=float, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--label", default="cell", help="free text for MLflow, e.g. O1-B2")
    parser.add_argument("--level", default="unknown")
    parser.add_argument("--enhance", default="unknown")
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-resamples", type=int, default=2000)
    parser.add_argument("--threads", type=int, default=None, help="torch threads for STG-NF on the host")
    parser.add_argument("--dataset", default="ShanghaiTech", choices=list(DATASET_DEFAULTS))
    parser.add_argument("--gt-dir", type=Path, default=None, help="default: STG-NF's GT folder of the dataset")
    parser.add_argument("--videos", nargs="*", default=None, help="only these videos (smoke runs)")
    parser.add_argument("--experiment", default=None, help="MLflow experiment (default EXP-004; UBnormal: EXP-006)")
    args = parser.parse_args()
    config = {"label": args.label, "level": args.level, "enhance": args.enhance, "tracker": "botsort",
              "trace_files": [str(p) for p in args.trace]}
    run_cell(args.cache, load_trace(args.trace), args.c, args.out, config, args.fps, args.seed, args.n_resamples,
             args.threads, gt_dir=args.gt_dir, dataset=args.dataset, videos=args.videos, experiment=args.experiment)


if __name__ == "__main__":
    main()
