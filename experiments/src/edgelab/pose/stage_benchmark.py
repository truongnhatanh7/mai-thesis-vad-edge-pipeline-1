"""Per-stage CPU cost of the pose pipeline on one video, torch intra-op threads 1/2/4. Mac numbers: comparative only.

    uv run python -m edgelab.pose.stage_benchmark --frames-dir <testing/frames/01_0014> --pose-json <released json>

Stages: pose model (ultralytics pre + inference + post), tracker (track-call wall minus the three model stages, so it
includes framework overhead), STG-NF flow (single 24-frame segment, batch 1, and a whole-video batch).
"""

import argparse
import time
from pathlib import Path

import cv2
import mlflow
import numpy as np
import torch

from edgelab.pose.extract import DEFAULTS
from edgelab.pose.stgnf_eval import SEG_LEN, STGNF_ROOT, _import_stgnf
from edgelab.tracking.mlflow_run import tracked_run

CHECKPOINT = STGNF_ROOT / "checkpoints/ShanghaiTech_85_9.tar"


def _quantiles(values_ms) -> dict:
    v = np.asarray(values_ms)
    return {"mean": float(v.mean()), "p50": float(np.quantile(v, 0.5)), "p95": float(np.quantile(v, 0.95))}


def bench_pose_and_tracker(images, model_name, tracker, imgsz, threads, warmup=10):
    from ultralytics import YOLO
    model = YOLO(model_name)
    for image in images[:warmup]:
        model.predict(image, imgsz=imgsz, device="cpu", verbose=False)
    model = YOLO(model_name)
    pose_ms, tracker_ms = [], []
    for index, image in enumerate(images):
        if index == 1:
            torch.set_num_threads(threads)  # Ultralytics select_device() resets it to min(8, cpus-1) on setup
            assert torch.get_num_threads() == threads
        t0 = time.perf_counter()
        result = model.track(image, persist=index > 0, tracker=tracker, imgsz=imgsz, conf=DEFAULTS["conf"],
                             iou=DEFAULTS["iou"], device="cpu", verbose=False)[0]
        wall_ms = (time.perf_counter() - t0) * 1e3
        model_ms = result.speed["preprocess"] + result.speed["inference"] + result.speed["postprocess"]
        pose_ms.append(model_ms)
        tracker_ms.append(wall_ms - model_ms)
    return _quantiles(pose_ms[1:]), _quantiles(tracker_ms[1:])  # first track call builds the tracker


def load_stgnf_model():
    _import_stgnf()
    from models.STG_NF.model_pose import STG_NF
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    model = STG_NF(pose_shape=(2, SEG_LEN, 18), hidden_channels=0, K=8, L=1, R=3.0, actnorm_scale=1.0,
                   flow_permutation="permute", flow_coupling="affine", LU_decomposed=True, learn_top=False,
                   edge_importance=False, temporal_kernel_size=None, strategy="uniform", max_hops=8, device="cpu")
    model.load_state_dict(checkpoint["state_dict"], strict=False)
    model.set_actnorm_init()
    return model.eval()


def bench_stgnf(model, num_video_segments, repeats=200):
    single = torch.randn(1, 2, SEG_LEN, 18)
    batch = torch.randn(num_video_segments, 2, SEG_LEN, 18)
    ones = torch.ones(1)
    with torch.no_grad():
        for _ in range(20):
            model(single, label=ones, score=ones)
        single_ms = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            model(single, label=ones, score=ones)
            single_ms.append((time.perf_counter() - t0) * 1e3)
        t0 = time.perf_counter()
        model(batch, label=torch.ones(num_video_segments), score=torch.ones(num_video_segments))
        batch_ms = (time.perf_counter() - t0) * 1e3
    return _quantiles(single_ms), batch_ms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames-dir", type=Path, required=True)
    parser.add_argument("--num-video-segments", type=int, default=786, help="segments STG-NF builds for the video")
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--imgsz", type=int, default=DEFAULTS["imgsz"])
    args = parser.parse_args()

    images = [cv2.imread(str(p)) for p in sorted(args.frames_dir.glob("*.jpg"))]
    for threads in args.threads:
        torch.set_num_threads(threads)
        cv2.setNumThreads(1)
        config = {"video": args.frames_dir.name, "frames": len(images), "torch_threads": threads,
                  "opencv_threads": 1, "pose_model": DEFAULTS["model"], "tracker": DEFAULTS["tracker"],
                  "imgsz": args.imgsz, "machine_note": "Mac M4 Pro, comparative only"}
        with tracked_run("EXP-001", f"stage-cost-mac-threads{threads}", config) as run:
            pose, tracker = bench_pose_and_tracker(images, DEFAULTS["model"], DEFAULTS["tracker"], args.imgsz, threads)
            mlflow.log_metric("torch_threads_effective", torch.get_num_threads())
            flow_single, flow_batch_ms = bench_stgnf(load_stgnf_model(), args.num_video_segments)
            metrics = {**{f"pose_ms_{k}": v for k, v in pose.items()},
                       **{f"tracker_ms_{k}": v for k, v in tracker.items()},
                       **{f"stgnf_segment_ms_{k}": v for k, v in flow_single.items()},
                       "stgnf_video_batch_ms": flow_batch_ms}
            mlflow.log_metrics(metrics)
            print(threads, run.info.run_id, {k: round(v, 3) for k, v in metrics.items()})


if __name__ == "__main__":
    main()
