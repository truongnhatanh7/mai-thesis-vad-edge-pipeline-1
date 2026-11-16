"""STEP 4: Online Real-Time Pipeline Runner on Physical Raspberry Pi 5 (`nanh@raspberrypi.local`).

Read this file fourth (~180 lines). It shows the exact online loop that runs on the Raspberry Pi 5:
  - Frames arrive on a strict 24.0 FPS wall-clock schedule (`arrival_ms = frame_idx * 41.67 ms`).
  - A single FIFO worker processes each frame:
      `start_ms = max(arrival_ms, worker_free_ms)`
      `queue_wait_ms = start_ms - arrival_ms`
  - If `use_guard=True` and `queue_wait_ms > 400 ms`, the Hysteresis Guard sheds alternate frames until wait < 150 ms.
  - Otherwise:
      1. Decode frame (`cv2.imread` / `VideoCapture`)
      2. If `use_cache=True`, check `StaticValidityCacheC1` (40x24 thumbnail diff). If static, reuse last detections;
         else run `PoseDetector` (`NCNN FP16` or `PyTorch FP32`).
      3. Update `BoT-SORT` (`default` or `two_stage`).
      4. Push tracked skeletons into `OnlineCircularWindowScorer` (`STG-NF` or `LoRaD`) and emit normality score.
  - Strict On-Time Delivery Rule:
      A score finishing at `finish_ms = start_ms + service_ms` is ONLY delivered on time if `finish_ms - arrival_ms <= 2000 ms`
      (within the 2.0-second rolling buffer cap); otherwise it is dropped as late (`dropped_late += 1`)!
  - Meanwhile, `Pi5TelemetryMonitor` samples `vcgencmd pmic_read_adc` (12 power rails -> Watts & Joules/frame),
    SoC temperature (`°C`), ARM clock (`MHz`), and throttling hex (`0x0`).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "experiments" / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "experiments" / "src"))

from edgelab.telemetry.pi5_telemetry import Pi5TelemetryMonitor, read_board_info  # noqa: E402
from step1_models import PoseDetector, SkeletonNormalizingFlow, load_scorer_checkpoint  # noqa: E402
from step2_tracker_cache import (  # noqa: E402
    HysteresisQueueGuard,
    OnlineCircularWindowScorer,
    StaticValidityCacheC1,
    build_botsort_tracker,
    step_tracker,
)
from step3_offline_eval import smooth_causal_one_sided  # noqa: E402


def run_online_pi5_stream(
    name: str,
    frames_list: list[tuple[str, int, Path]],
    detector: PoseDetector,
    flow_model: torch.nn.Module,
    two_stage_tracker: bool = True,
    use_cache_c1: bool = True,
    use_queue_guard: bool = True,
    burst_period_w: int = 1,
    train_sentinel: float = -0.5122,
) -> dict:
    """Execute one online pipeline arm on the frame stream and return latency, FPS, queue wait, power, and scores."""
    dt_ms = 1000.0 / 24.0  # 24.0 FPS camera arrival interval (41.67 ms)
    cache_c1 = StaticValidityCacheC1()
    guard = HysteresisQueueGuard(high_ms=400.0, low_ms=150.0)
    monitor = Pi5TelemetryMonitor(interval_s=0.25).start()

    cur_clip = ""
    tracker = None
    window_scorer = None
    worker_free_ms = 0.0

    service_ms_all, pose_ms_all, track_ms_all, score_ms_all, wait_ms_all = [], [], [], [], []
    delivered_normality = np.full(len(frames_list), train_sentinel, dtype=np.float64)
    c1_hits, guard_sheds, windows_scored, windows_dropped_late = 0, 0, 0, 0
    last_valid_norm = train_sentinel
    hold_age = 999
    max_hold = max(0, burst_period_w - 1)

    for idx, (clip_name, clip_fidx, jpg_path) in enumerate(frames_list):
        if clip_name != cur_clip:
            cur_clip = clip_name
            tracker = build_botsort_tracker(two_stage=two_stage_tracker)
            window_scorer = OnlineCircularWindowScorer(flow_model, seg_len=24)
            cache_c1.reset()
            last_valid_norm = train_sentinel
            hold_age = 999

        arrival_ms = idx * dt_ms
        start_ms = max(arrival_ms, worker_free_ms)
        wait_ms = start_ms - arrival_ms
        wait_ms_all.append(wait_ms)

        # 0. Intact-Window Burst Schedule (W > 24 skips inter-burst frames cleanly) or Queue Guard Shedding
        in_burst = True if burst_period_w <= 24 else ((clip_fidx % burst_period_w) < 24)
        if not in_burst or (use_queue_guard and guard.should_shed(clip_fidx, wait_ms)):
            if not in_burst:
                pass
            else:
                guard_sheds += 1
            if hold_age < max_hold:
                delivered_normality[idx] = last_valid_norm
                hold_age += 1
            continue

        # 1. Decode Frame
        t0 = time.perf_counter()
        img_bgr = cv2.imread(str(jpg_path))

        # 2. Pose Estimation (with or without C1 Static-Validity Cache)
        tp0 = time.perf_counter()
        if use_cache_c1:
            boxes, scores, kps, reused = cache_c1.check_or_compute(clip_fidx, img_bgr, detector)
            if reused:
                c1_hits += 1
        else:
            boxes, scores, kps, _ = detector(img_bgr)
        tp1 = time.perf_counter()

        # 3. Multi-Object Tracking (BoT-SORT)
        tracked_poses = step_tracker(tracker, boxes, scores, kps, img_bgr, two_stage=two_stage_tracker)
        tt1 = time.perf_counter()

        # 4. Circular Window Buffer & Normalizing Flow Scoring (STG-NF or LoRaD)
        win_norms = window_scorer.step(clip_fidx, tracked_poses)
        ts1 = time.perf_counter()

        svc_ms = (ts1 - t0) * 1000.0
        finish_ms = start_ms + svc_ms
        worker_free_ms = finish_ms

        service_ms_all.append(svc_ms)
        pose_ms_all.append((tp1 - tp0) * 1000.0)
        track_ms_all.append((tt1 - tp1) * 1000.0)
        score_ms_all.append((ts1 - tt1) * 1000.0)

        # 5. Strict Causal Delivery Check (2.0-second rolling buffer latency bound)
        if len(win_norms) > 0:
            windows_scored += len(win_norms)
            if (finish_ms - arrival_ms) <= 2000.0:
                last_valid_norm = float(np.min(win_norms))
                delivered_normality[idx] = last_valid_norm
                hold_age = 0
            else:
                windows_dropped_late += len(win_norms)
        elif hold_age < max_hold:
            delivered_normality[idx] = last_valid_norm
            hold_age += 1

    tel = monitor.stop()
    mean_svc = float(np.mean(service_ms_all)) if service_ms_all else 0.0
    eff_fps = 1000.0 / max(0.1, mean_svc)
    mean_w = tel["power_total_w_mean"]

    return {
        "arm": name,
        "frames": len(frames_list),
        "service_ms_mean": round(mean_svc, 2),
        "pose_ms_mean": round(float(np.mean(pose_ms_all)), 2) if pose_ms_all else 0.0,
        "track_ms_mean": round(float(np.mean(track_ms_all)), 2) if track_ms_all else 0.0,
        "score_ms_mean": round(float(np.mean(score_ms_all)), 2) if score_ms_all else 0.0,
        "throughput_fps": round(eff_fps, 2),
        "wait_p99_ms": round(float(np.percentile(wait_ms_all, 99)), 1),
        "c1_hit_pct": round(100.0 * c1_hits / max(1, len(frames_list)), 1),
        "guard_shed_pct": round(100.0 * guard_sheds / max(1, len(frames_list)), 1),
        "windows_scored": windows_scored,
        "windows_dropped_late": windows_dropped_late,
        "power_w": round(mean_w, 2) if np.isfinite(mean_w) else 0.0,
        "energy_j_per_frame": round(mean_w * (mean_svc / 1000.0), 3) if np.isfinite(mean_w) else 0.0,
        "temp_c_mean": tel["temp_c_mean"],
        "throttled": tel["throttled_codes_seen"],
        "smoothed_anomaly_scores": smooth_causal_one_sided(-delivered_normality),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Clean Option A Online Runner (works on Pi 5 or Mac)")
    ap.add_argument("--frames-dir", type=Path, default=Path("~/thesis-data/frames").expanduser())
    ap.add_argument("--models-dir", type=Path, default=Path("~/thesis-data/models").expanduser())
    ap.add_argument("--max-frames", type=int, default=240, help="Number of frames to run per arm (default: 240 = 10s)")
    args = ap.parse_args()

    # Fall back to vm-exchange paths when run locally on Mac
    if not args.frames_dir.exists():
        args.frames_dir = Path("~/thesis-data/vm-exchange/frames").expanduser()
    if not args.models_dir.exists():
        args.models_dir = Path("~/thesis-data/vm-exchange/models").expanduser()

    clips = sorted(p for p in args.frames_dir.iterdir() if p.is_dir())
    frames_list = []
    for cdir in clips:
        for idx, jp in enumerate(sorted(cdir.glob("*.jpg"))):
            frames_list.append((cdir.name, idx, jp))
            if len(frames_list) >= args.max_frames:
                break
        if len(frames_list) >= args.max_frames:
            break

    stgnf_ckpt = REPO_ROOT / "experiments/third_party/STG-NF/checkpoints/ShanghaiTech_85_9.tar"
    if not stgnf_ckpt.exists():
        stgnf_ckpt = Path("~/edgelab/third_party/STG-NF/checkpoints/ShanghaiTech_85_9.tar").expanduser()
    stgnf_model = load_scorer_checkpoint(stgnf_ckpt)
    lorad_model = SkeletonNormalizingFlow(n_frames=24, steps=8, causal=True).eval()
    for m in lorad_model.modules():
        if hasattr(m, "inited"):
            m.inited = True

    det_stock320 = PoseDetector("ncnn", args.models_dir / "yolov8n-pose-320-320x320-ncnn", threads=4)
    det_stock640 = PoseDetector("ncnn", args.models_dir / "yolov8n-pose-640-384x640-ncnn", threads=4)
    det_ml0_640 = PoseDetector("ncnn", args.models_dir / "exp020/ML0-640-384x640-ncnn", threads=4)

    print(f"\n=== Option A Online Stream Benchmark ({len(frames_list)} frames @ 24.0 FPS arrival) ===")
    for name, det, flow, two_stg, use_c1, use_grd in [
        ("1. Huang RPi* (Stock NCNN 320, Default Track)", det_stock320, stgnf_model, False, False, False),
        ("2. Huang/Stock NCNN 640 (Full-Rate, No Cache)", det_stock640, stgnf_model, False, False, False),
        ("3. LoRaD Causal Flow (Stock NCNN 640, Full-Rate)", det_stock640, lorad_model, False, False, False),
        ("4. LoRaD + Cache-C1+Guard (Bursty Reuse)", det_stock640, lorad_model, False, True, True),
        ("5. Ours ML0 + 2-Stage Track (Full-Rate, No Cache)", det_ml0_640, stgnf_model, True, False, False),
        ("6. Ours ML0 + 2-Stage Track + Cache-C1+Guard", det_ml0_640, stgnf_model, True, True, True),
    ]:
        r = run_online_pi5_stream(name, frames_list, det, flow, two_stg, use_c1, use_grd)
        print(
            f"{r['arm']:50s} | {r['throughput_fps']:5.1f} FPS (svc={r['service_ms_mean']:5.1f}ms) | "
            f"wait_p99={r['wait_p99_ms']:7.1f}ms | C1={r['c1_hit_pct']:4.1f}% | late_drop={r['windows_dropped_late']:3d}"
        )


if __name__ == "__main__":
    main()
