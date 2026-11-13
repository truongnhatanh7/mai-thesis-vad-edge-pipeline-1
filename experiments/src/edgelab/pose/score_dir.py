"""Score a folder of *tracked_person.json with the frozen STG-NF checkpoint -> AUC for both pre-registered variants.

    uv run python -m edgelab.pose.score_dir --pose-dir <dir> --out-dir <dir> --level L2 --enhance E1 --c 0.25

paper = score on the middle frame of the 24-frame segment + offline Gaussian smoothing (uses the future);
live  = score on the last frame + causal half-Gaussian (past only). One MLflow run per variant
(experiment EXP-003 by default; --dataset UBnormal --experiment EXP-006 for the second dataset).
"""

import argparse
from pathlib import Path

import mlflow
import numpy as np

from edgelab.metrics.scores import bootstrap_ci_over_videos, frame_level_auc
from edgelab.pose.evaluate import CHECKPOINT, GT_DIR
from edgelab.pose.smoothing import SMOOTHERS
from edgelab.pose.stgnf_eval import clip_normality, score_frame_offsets, score_segments
from edgelab.pose import ubnormal
from edgelab.tracking.mlflow_run import tracked_run

# (checkpoint, gt dir) of each dataset; ShanghaiTech = the EXP-001 values
DATASET_DEFAULTS = {"ShanghaiTech": (CHECKPOINT, GT_DIR),
                    "UBnormal": (ubnormal.CHECKPOINT, ubnormal.GT_DIR)}

VARIANTS = {"paper": ("middle", "offline_gaussian"), "live": ("end", "causal_half_gaussian")}


def coverage(covered_per_video: list[np.ndarray], labels_per_video: list[np.ndarray]) -> tuple[float, float]:
    """(fraction of anomalous frames, fraction of normal frames) that have at least one scored person."""
    covered, labels = np.concatenate(covered_per_video), np.concatenate(labels_per_video)
    return float(covered[labels == 1].mean()), float(covered[labels == 0].mean())


def variant_scores(segments, gt_dir: Path, offset_name: str, smoother_name: str, dataset: str = "ShanghaiTech",
                   clips: list[str] | None = None):
    """-> (names, labels, smoothed anomaly scores, covered masks) for one variant."""
    names, labels, normality = clip_normality(segments.normality, segments.metadata, gt_dir,
                                              score_frame_offsets(dataset)[offset_name], dataset, clips)
    covered = [np.isfinite(n) for n in normality]
    flat = np.concatenate(normality)
    most_normal = flat[np.isfinite(flat)].max()  # frames with nobody get the most normal score (STG-NF's rule)
    raw = [-np.where(np.isfinite(n), n, most_normal) for n in normality]
    return names, labels, [SMOOTHERS[smoother_name](s) for s in raw], covered


def score_pose_dir(pose_dir: Path, out_dir: Path, config: dict, threads: int | None = None, n_resamples: int = 2000,
                   checkpoint: Path | None = None, gt_dir: Path | None = None, experiment: str = "EXP-003",
                   tracking_uri: str | None = None, dataset: str = "ShanghaiTech",
                   clips: list[str] | None = None) -> dict[str, dict]:
    """config: free-form tags logged with every run, e.g. {level, enhance, c, tracker, device}.

    checkpoint / gt_dir default to the dataset's own (DATASET_DEFAULTS). clips: UBnormal test videos (default: all 211).
    """
    default_checkpoint, default_gt_dir = DATASET_DEFAULTS[dataset]
    checkpoint, gt_dir = checkpoint or default_checkpoint, gt_dir or default_gt_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        segments = score_segments(pose_dir, checkpoint, device="cpu", threads=threads, dataset=dataset)
    except ValueError as error:  # STG-NF's loader crashes with "need at least one array" when no track reaches 24 frames
        raise ValueError(f"no 24-frame person track in any video of {pose_dir}: nothing to score") from error
    results = {}
    for variant, (offset_name, smoother_name) in VARIANTS.items():
        names, labels, scores, covered = variant_scores(segments, gt_dir, offset_name, smoother_name, dataset, clips)
        auc = frame_level_auc(labels, scores)
        low, high = bootstrap_ci_over_videos(labels, scores, n_resamples=n_resamples)
        cov_anomalous, cov_normal = coverage(covered, labels)
        scores_file = out_dir / f"frame_scores_{variant}.npz"
        np.savez_compressed(scores_file, names=np.array(names),
                            **{f"score_{n}": s for n, s in zip(names, scores)},
                            **{f"label_{n}": l for n, l in zip(names, labels)},
                            **{f"covered_{n}": c for n, c in zip(names, covered)})
        run_config = {**config, "variant": variant, "score_frame": offset_name, "smoothing": smoother_name,
                      "pose_dir": str(pose_dir), "checkpoint": str(checkpoint), "bootstrap_resamples": n_resamples,
                      "dataset": dataset}
        with tracked_run(experiment, f"{config.get('level')}-{config.get('enhance')}-c{config.get('c')}-{variant}",
                         run_config, input_files={"checkpoint": checkpoint}, tracking_uri=tracking_uri) as run:
            mlflow.set_tags({k: str(config[k]) for k in ("level", "enhance", "c", "tracker", "device") if k in config})
            results[variant] = {"run_id": run.info.run_id, "auc": auc, "ci95_low": low, "ci95_high": high,
                                "coverage_anomalous": cov_anomalous, "coverage_normal": cov_normal,
                                "num_segments": segments.num_segments, "num_videos": len(names),
                                "scores_file": str(scores_file)}
            mlflow.log_metrics({k: v for k, v in results[variant].items() if isinstance(v, (int, float))})
            mlflow.log_artifact(str(scores_file))
        print(f"{variant:6s} AUC={auc * 100:.2f}  CI95=[{low * 100:.2f}, {high * 100:.2f}]  "
              f"coverage anomalous={cov_anomalous:.3f} normal={cov_normal:.3f}  segments={segments.num_segments}")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pose-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    for name in ("level", "enhance", "tracker", "device"):
        parser.add_argument(f"--{name}", default="unknown")
    parser.add_argument("--c", type=float, default=None)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--n-resamples", type=int, default=2000)
    parser.add_argument("--dataset", default="ShanghaiTech", choices=list(DATASET_DEFAULTS))
    parser.add_argument("--checkpoint", type=Path, default=None, help="default: the dataset's released checkpoint")
    parser.add_argument("--gt-dir", type=Path, default=None, help="default: STG-NF's GT folder of the dataset")
    parser.add_argument("--experiment", default="EXP-003", help="MLflow experiment name")
    parser.add_argument("--videos", nargs="*", default=None, help="UBnormal only: score just these test videos (smoke runs)")
    args = parser.parse_args()
    config = {k: getattr(args, k) for k in ("level", "enhance", "c", "tracker", "device")}
    score_pose_dir(args.pose_dir, args.out_dir, config, args.threads, args.n_resamples, args.checkpoint, args.gt_dir,
                   args.experiment, dataset=args.dataset, clips=args.videos)


if __name__ == "__main__":
    main()
