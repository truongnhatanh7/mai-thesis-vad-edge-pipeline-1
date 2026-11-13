"""EXP-001 evaluation: frozen STG-NF on one pose source, every (score frame x smoothing) variant, one MLflow run each.

    uv run python -m edgelab.pose.evaluate --source released --pose-dir <dir with *_tracked_person.json>
"""

import argparse
import json
import pickle
from pathlib import Path

import mlflow
import numpy as np

from edgelab.metrics.scores import bootstrap_ci_over_videos, frame_level_auc
from edgelab.pose.smoothing import SMOOTHERS
from edgelab.pose.stgnf_eval import SEG_LEN, STGNF_ROOT, assemble_clip_scores, score_segments
from edgelab.tracking.mlflow_run import EXPERIMENTS_ROOT, tracked_run

CHECKPOINT = STGNF_ROOT / "checkpoints/ShanghaiTech_85_9.tar"
GT_DIR = STGNF_ROOT / "data/ShanghaiTech/gt/test_frame_mask"
# middle = paper (uses 12 future frames); end = score only exists once the 24-frame segment is complete (live).
SCORE_FRAME_OFFSETS = {"middle": SEG_LEN // 2, "end": SEG_LEN - 1}


def evaluate(source: str, pose_dir: Path, out_dir: Path, threads: int | None, n_resamples: int,
             checkpoint: Path = CHECKPOINT) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    segments = score_segments(pose_dir, checkpoint, device="cpu", threads=threads)
    with open(out_dir / "segments.pkl", "wb") as f:
        pickle.dump(segments, f)

    rows = []
    for offset_name, offset in SCORE_FRAME_OFFSETS.items():
        names, labels, raw_scores = assemble_clip_scores(segments.normality, segments.metadata, GT_DIR, offset)
        for smoother_name, smoother in SMOOTHERS.items():
            scores = [smoother(s) for s in raw_scores]
            config = {"source": source, "pose_dir": str(pose_dir), "checkpoint": str(checkpoint),
                      "score_frame": offset_name, "score_frame_offset": offset, "smoothing": smoother_name,
                      "seg_len": SEG_LEN, "test_stride": 1, "device": "cpu", "torch_threads": threads,
                      "bootstrap_resamples": n_resamples}
            run_name = f"{source}-{offset_name}-{smoother_name}"
            with tracked_run("EXP-001", run_name, config, input_files={"checkpoint": checkpoint}) as run:
                auc = frame_level_auc(labels, scores)
                low, high = bootstrap_ci_over_videos(labels, scores, n_resamples=n_resamples)
                mlflow.log_metrics({"frame_auc": auc, "ci95_low": low, "ci95_high": high,
                                    "num_segments": segments.num_segments, "num_videos": len(names),
                                    "num_frames": int(sum(len(l) for l in labels)),
                                    "seconds_stgnf_model": segments.seconds_model,
                                    "seconds_stgnf_load": segments.seconds_load})
                scores_file = out_dir / f"frame_scores_{offset_name}_{smoother_name}.npz"
                np.savez_compressed(scores_file, names=np.array(names), **{f"score_{n}": s for n, s in zip(names, scores)},
                                    **{f"label_{n}": l for n, l in zip(names, labels)})
                mlflow.log_artifact(str(scores_file))
                rows.append({"run_id": run.info.run_id, "run_name": run_name, "frame_auc": auc,
                             "ci95": [low, high], "score_frame": offset_name, "smoothing": smoother_name})
                print(f"{run_name:45s} AUC={auc * 100:.2f}  CI95=[{low * 100:.2f}, {high * 100:.2f}]  run={run.info.run_id}")
    with open(out_dir / "results.jsonl", "a") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="label, e.g. released or yolov8n-botsort")
    parser.add_argument("--pose-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--n-resamples", type=int, default=2000)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    args = parser.parse_args()
    out_dir = args.out_dir or EXPERIMENTS_ROOT / "artifacts" / "EXP-001" / args.source
    evaluate(args.source, args.pose_dir, out_dir, args.threads, args.n_resamples, args.checkpoint)


if __name__ == "__main__":
    main()
