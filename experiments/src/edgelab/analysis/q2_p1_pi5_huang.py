"""Q2-P1-Pi5-Huang-Repro: Physical Raspberry Pi 5 reproduction of Huang et al. (IEEE ESL 2026) vs. Our Pipeline.

Runs directly on the Raspberry Pi 5 (Cortex-A76, 8GB RAM) and streams live telemetry, step metrics,
artifacts, and MLflow Traces (hierarchical spans for Decode -> Pose -> BoT-SORT -> STG-NF -> Telemetry)
to the Mac MLflow server via `MLFLOW_TRACKING_URI=http://127.0.0.1:5000` (SSH reverse tunnel).

Evaluates:
1. `Huang_RPi_PyTorch_640`: Default PyTorch YOLOv8n-pose (640) + BoT-SORT (default) + STG-NF (Huang Table III/IV `RPi`)
2. `Huang_RPi_Star_NCNN_320x320`: NCNN YOLOv8n-pose (320x320) + BoT-SORT (default) + STG-NF (Huang Table III/IV `RPi*`)
3. `Huang_RPi_Star_NCNN_192x320`: NCNN YOLOv8n-pose (192x320 letterbox) + BoT-SORT (default) + STG-NF
4. `Stock_NCNN_384x640`: Stock NCNN YOLOv8n-pose (384x640) + BoT-SORT (default) + STG-NF
5. `Ours_ML0_NCNN_640_FullRate`: ML0 NCNN (384x640) + 2-Stage BoT-SORT (c=0.20, low=0.05) + STG-NF (W=1)
6. `Ours_ML0_NCNN_640_CacheC1_Guard`: ML0 NCNN (384x640) + 2-Stage BoT-SORT + C1 static-validity detector reuse + Guard
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"

import cv2
import mlflow
import numpy as np
import torch
from mlflow.entities import SpanType

from edgelab.pipeline.ncnn_pose import NcnnPose
from edgelab.pipeline.stgnf_live import LiveStgnf, load_model
from edgelab.pose.replay import build_tracker
from edgelab.telemetry.pi5_telemetry import (
    Pi5TelemetryMonitor,
    read_board_info,
    read_pmic_power,
    read_soc_temp_c,
    read_throttled,
)

PI_DATA_DIR = Path("~/thesis-data").expanduser()
PI_EDGELAB_DIR = Path("~/edgelab").expanduser()
FRAMES_ROOT = PI_DATA_DIR / "frames"
MODELS_ROOT = PI_DATA_DIR / "models"
STGNF_CKPT = PI_EDGELAB_DIR / "third_party/STG-NF/checkpoints/ShanghaiTech_85_9.tar"


class _DetBoxes:
    """Minimal Ultralytics Boxes-compatible wrapper for BoT-SORT update."""

    def __init__(self, xyxy: np.ndarray, conf: np.ndarray):
        self.xyxy = xyxy
        self.conf = conf
        self.cls = np.zeros(len(conf), dtype=np.float32)
        cx = 0.5 * (xyxy[:, 0] + xyxy[:, 2])
        cy = 0.5 * (xyxy[:, 1] + xyxy[:, 3])
        w = xyxy[:, 2] - xyxy[:, 0]
        h = xyxy[:, 3] - xyxy[:, 1]
        self.xywh = np.stack([cx, cy, w, h], axis=1) if len(conf) else np.zeros((0, 4), dtype=np.float32)

    def __len__(self) -> int:
        return len(self.conf)

    def __getitem__(self, idx) -> "_DetBoxes":
        return _DetBoxes(self.xyxy[idx], self.conf[idx])


def make_two_stage_tracker(c_high: float = 0.20, c_low: float = 0.05, buffer: int = 30, match: float = 0.85):
    from ultralytics.trackers.track import TRACKER_MAP
    from ultralytics.utils import ROOT, YAML, IterableSimpleNamespace

    cfg = YAML.load(ROOT / "cfg/trackers/botsort.yaml")
    cfg.update(
        track_high_thresh=c_high,
        new_track_thresh=c_high,
        track_low_thresh=c_low,
        track_buffer=buffer,
        match_thresh=match,
        fuse_score=True,
        gmc_method="none",
        with_reid=False,
    )
    return TRACKER_MAP[cfg["tracker_type"]](IterableSimpleNamespace(**cfg))


def match_tracks_to_keypoints(tracks_out: np.ndarray, boxes: np.ndarray, keypoints: np.ndarray) -> dict[int, np.ndarray]:
    """Match BoT-SORT output rows [x1, y1, x2, y2, track_id, ...] back to detector keypoints (N, 17, 3)."""
    if len(tracks_out) == 0 or len(boxes) == 0:
        return {}
    out: dict[int, np.ndarray] = {}
    for row in tracks_out:
        tx1, ty1, tx2, ty2, tid = float(row[0]), float(row[1]), float(row[2]), float(row[3]), int(row[4])
        # IoU with detector boxes
        ix1 = np.maximum(tx1, boxes[:, 0])
        iy1 = np.maximum(ty1, boxes[:, 1])
        ix2 = np.minimum(tx2, boxes[:, 2])
        iy2 = np.minimum(ty2, boxes[:, 3])
        inter = np.maximum(0.0, ix2 - ix1) * np.maximum(0.0, iy2 - iy1)
        a_t = max(1e-6, (tx2 - tx1) * (ty2 - ty1))
        a_b = np.maximum(1e-6, (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]))
        iou = inter / (a_t + a_b - inter + 1e-6)
        best = int(np.argmax(iou))
        if iou[best] >= 0.3:
            out[tid] = keypoints[best]
    return out


def c1_thumbnail(bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (40, 24), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0


def c1_static_check(prev_thumb: np.ndarray, cur_thumb: np.ndarray, prev_boxes: np.ndarray, h_img: int = 480, w_img: int = 856) -> tuple[bool, float, float]:
    diff = np.abs(cur_thumb - prev_thumb)
    # Exterior border (top/bottom 3 rows, left/right 4 cols)
    mask = np.zeros_like(diff, dtype=bool)
    mask[:3, :] = True
    mask[-3:, :] = True
    mask[:, :4] = True
    mask[:, -4:] = True
    ext_p95 = float(np.percentile(diff[mask], 95))
    roi_max = 0.0
    for box in prev_boxes:
        x1 = max(0, min(39, int(box[0] / w_img * 40)))
        y1 = max(0, min(23, int(box[1] / h_img * 24)))
        x2 = max(x1 + 1, min(40, int(np.ceil(box[2] / w_img * 40))))
        y2 = max(y1 + 1, min(24, int(np.ceil(box[3] / h_img * 24))))
        roi_max = max(roi_max, float(diff[y1:y2, x1:x2].mean()))
    # Locked E03 SHT level 3 static thresholds (q60)
    is_static = (roi_max <= 0.0225) and (ext_p95 <= 0.0350)
    return is_static, roi_max, ext_p95


@mlflow.trace(name="pi5_traced_frame_inspection", span_type=SpanType.CHAIN)
def traced_sample_frame(
    arm_name: str,
    frame_idx: int,
    clip_name: str,
    decode_ms: float,
    pose_ms: float,
    track_ms: float,
    score_ms: float,
    n_dets: int,
    n_tracks: int,
    n_windows: int,
    min_normality: float | None,
    c1_reused: bool,
    telemetry_snap: dict[str, Any],
) -> dict[str, Any]:
    """Emit a rich hierarchical MLflow Trace for representative frames so the MLflow UI Trace tab shows full stage I/O."""
    span = mlflow.get_current_active_span()
    if span:
        span.set_attribute("description", f"Pi 5 End-to-End Frame Trace ({arm_name}, clip {clip_name}, frame {frame_idx})")
    mlflow.update_current_trace(
        metadata={
            "mlflow.trace.user": "nanh",
            "mlflow.trace.session": f"pi5-{arm_name}",
        }
    )
    with mlflow.start_span(name="1_decode_frame", span_type=SpanType.TOOL) as s_dec:
        s_dec.set_inputs({"clip": clip_name, "frame_idx": frame_idx, "resolution": "856x480"})
        s_dec.set_outputs({"decode_ms": round(decode_ms, 3), "status": "ok"})

    with mlflow.start_span(name="2_pose_detector", span_type=SpanType.LLM) as s_pose:
        s_pose.set_inputs({"arm": arm_name, "c1_reused": c1_reused})
        s_pose.set_outputs({"pose_ms": round(pose_ms, 3), "n_detections": n_dets, "c1_cache_hit": c1_reused})

    with mlflow.start_span(name="3_botsort_tracker", span_type=SpanType.TOOL) as s_trk:
        s_trk.set_inputs({"n_detections_in": n_dets})
        s_trk.set_outputs({"track_ms": round(track_ms, 3), "n_active_tracks": n_tracks})

    with mlflow.start_span(name="4_stgnf_anomaly_scorer", span_type=SpanType.LLM) as s_scr:
        s_scr.set_inputs({"n_completed_24f_windows": n_windows})
        s_scr.set_outputs({"score_ms": round(score_ms, 3), "min_normality": min_normality})

    with mlflow.start_span(name="5_pi5_hardware_telemetry", span_type=SpanType.TOOL) as s_tel:
        s_tel.set_inputs({"sensor_source": "vcgencmd_pmic_and_sysfs"})
        s_tel.set_outputs(telemetry_snap)

    total_ms = decode_ms + pose_ms + track_ms + score_ms
    return {
        "arm": arm_name,
        "clip": clip_name,
        "frame_idx": frame_idx,
        "total_service_ms": round(total_ms, 3),
        "n_dets": n_dets,
        "n_tracks": n_tracks,
        "n_windows": n_windows,
        "c1_reused": c1_reused,
        "temp_c": telemetry_snap.get("temp_c"),
        "power_total_w": telemetry_snap.get("power_total_w"),
    }


def load_frame_list(max_frames: int = 720) -> list[tuple[str, int, Path]]:
    """Collect ordered frames across the 12 ShanghaiTech clips on disk."""
    clips = sorted(p for p in FRAMES_ROOT.iterdir() if p.is_dir())
    items: list[tuple[str, int, Path]] = []
    for clip_dir in clips:
        jpgs = sorted(clip_dir.glob("*.jpg"))
        for idx, jp in enumerate(jpgs):
            items.append((clip_dir.name, idx, jp))
            if len(items) >= max_frames:
                return items
    return items


def run_pi5_pipeline_arm(
    arm_cfg: dict[str, Any],
    frames_list: list[tuple[str, int, Path]],
    stgnf_model: torch.nn.Module,
    out_dir: Path,
) -> dict[str, Any]:
    name = arm_cfg["name"]
    threads = int(arm_cfg.get("threads", 4))
    os.environ["OMP_NUM_THREADS"] = str(threads)
    torch.set_num_threads(threads)

    # Initialize detector
    backend = arm_cfg["backend"]
    yolo_pt = None
    ncnn_det = None
    if backend == "pytorch":
        from ultralytics import YOLO

        yolo_pt = YOLO(str(arm_cfg["model_path"]))
        torch.set_num_threads(threads)
    elif backend == "ncnn":
        ncnn_det = NcnnPose(arm_cfg["model_path"], threads=threads, fp16=True)
    else:
        raise ValueError(f"Unknown backend {backend}")

    # Warmup detector (5 frames)
    sample_img = cv2.imread(str(frames_list[0][2]))
    for _ in range(5):
        if backend == "pytorch":
            yolo_pt.predict(sample_img, imgsz=arm_cfg["imgsz"], conf=0.25, verbose=False, device="cpu")
            torch.set_num_threads(threads)
        else:
            ncnn_det(sample_img, conf=0.05)

    # Tracker & state per clip
    two_stage = bool(arm_cfg.get("two_stage_tracker", False))
    use_c1 = bool(arm_cfg.get("use_c1", False))
    use_guard = bool(arm_cfg.get("use_guard", False))
    gate_c = float(arm_cfg.get("gate_c", 0.20 if two_stage else 0.25))
    low_c = float(arm_cfg.get("low_c", 0.05 if two_stage else gate_c))

    monitor = Pi5TelemetryMonitor(interval_s=0.20).start()
    dt_ms = 1000.0 / 24.0  # 24 FPS stream arrival cadence

    decode_ms_list: list[float] = []
    pose_ms_list: list[float] = []
    track_ms_list: list[float] = []
    score_ms_list: list[float] = []
    c1_ms_list: list[float] = []
    total_ms_list: list[float] = []
    queue_wait_ms_list: list[float] = []
    n_dets_list: list[int] = []
    n_tracks_list: list[int] = []
    n_wins_list: list[int] = []
    c1_hits = 0
    guard_sheds = 0
    late_drops = 0

    cur_clip = ""
    tracker = None
    live_scorer = None
    prev_thumb: np.ndarray | None = None
    prev_boxes = np.zeros((0, 4), dtype=np.float32)
    prev_scores = np.zeros((0,), dtype=np.float32)
    prev_kps = np.zeros((0, 17, 3), dtype=np.float32)
    c1_age = 999
    guard_active = False
    worker_free_ms = 0.0

    t_wall_start = time.perf_counter()
    cpu_start = time.process_time()

    for global_idx, (clip_name, clip_fidx, jpg_path) in enumerate(frames_list):
        if clip_name != cur_clip:
            cur_clip = clip_name
            tracker = make_two_stage_tracker(gate_c, low_c) if two_stage else build_tracker(gate_c, "botsort")
            live_scorer = LiveStgnf(stgnf_model)
            prev_thumb = None
            c1_age = 999

        arrival_ms = global_idx * dt_ms
        start_ms = max(arrival_ms, worker_free_ms)
        wait_ms = start_ms - arrival_ms
        queue_wait_ms_list.append(wait_ms)

        # Hysteresis queue guard
        if use_guard:
            if not guard_active and wait_ms > 400.0:
                guard_active = True
            elif guard_active and wait_ms < 150.0:
                guard_active = False
            if guard_active and (clip_fidx % 2 == 1):
                guard_sheds += 1
                decode_ms_list.append(0.0)
                pose_ms_list.append(0.0)
                track_ms_list.append(0.0)
                score_ms_list.append(0.0)
                c1_ms_list.append(0.0)
                total_ms_list.append(0.0)
                n_dets_list.append(0)
                n_tracks_list.append(0)
                n_wins_list.append(0)
                continue

        # 1. Decode
        t0 = time.perf_counter()
        img = cv2.imread(str(jpg_path))
        t1 = time.perf_counter()
        dec_ms = (t1 - t0) * 1000.0

        # 2. C1 check + Pose estimation
        c1_reused = False
        c1_ms = 0.0
        if use_c1:
            tc0 = time.perf_counter()
            cur_thumb = c1_thumbnail(img)
            if prev_thumb is not None and c1_age < 3 and (clip_fidx % 12 != 0):
                is_static, _, _ = c1_static_check(prev_thumb, cur_thumb, prev_boxes, img.shape[0], img.shape[1])
                if is_static:
                    c1_reused = True
            tc1 = time.perf_counter()
            c1_ms = (tc1 - tc0) * 1000.0
            prev_thumb = cur_thumb

        tp0 = time.perf_counter()
        if c1_reused:
            boxes, scores, kps = prev_boxes, prev_scores, prev_kps
            c1_age += 1
            c1_hits += 1
            p_ms = 0.0
        else:
            if backend == "pytorch":
                res = yolo_pt.predict(img, imgsz=arm_cfg["imgsz"], conf=low_c, verbose=False, device="cpu")[0]
                torch.set_num_threads(threads)
                if res.boxes is not None and len(res.boxes):
                    boxes = res.boxes.xyxy.cpu().numpy().astype(np.float32)
                    scores = res.boxes.conf.cpu().numpy().astype(np.float32)
                    kps = res.keypoints.data.cpu().numpy().astype(np.float32)
                else:
                    boxes = np.zeros((0, 4), dtype=np.float32)
                    scores = np.zeros((0,), dtype=np.float32)
                    kps = np.zeros((0, 17, 3), dtype=np.float32)
            else:
                boxes, scores, kps, _ = ncnn_det(img, conf=low_c)
            tp1 = time.perf_counter()
            p_ms = (tp1 - tp0) * 1000.0
            prev_boxes, prev_scores, prev_kps = boxes, scores, kps
            c1_age = 0

        # 3. Tracker (BoT-SORT)
        tt0 = time.perf_counter()
        keep = scores >= (low_c if two_stage else gate_c)
        b_in, s_in, k_in = boxes[keep], scores[keep], kps[keep]
        det_obj = _DetBoxes(b_in, s_in)
        tracks_out = tracker.update(det_obj, img)
        matched_kps = match_tracks_to_keypoints(tracks_out, b_in, k_in)
        tt1 = time.perf_counter()
        trk_ms = (tt1 - tt0) * 1000.0

        # 4. STG-NF Scoring on completed 24-frame windows
        ts0 = time.perf_counter()
        scored_tids, scores_out = live_scorer.update(clip_fidx, matched_kps)
        min_norm = float(np.min(scores_out)) if len(scores_out) > 0 else None
        ts1 = time.perf_counter()
        scr_ms = (ts1 - ts0) * 1000.0

        svc_ms = dec_ms + c1_ms + p_ms + trk_ms + scr_ms
        finish_ms = start_ms + svc_ms
        worker_free_ms = finish_ms
        if finish_ms - arrival_ms > 2000.0 and len(scores_out) > 0:
            late_drops += len(scores_out)

        decode_ms_list.append(dec_ms)
        c1_ms_list.append(c1_ms)
        pose_ms_list.append(p_ms)
        track_ms_list.append(trk_ms)
        score_ms_list.append(scr_ms)
        total_ms_list.append(svc_ms)
        n_dets_list.append(int(len(b_in)))
        n_tracks_list.append(int(len(tracks_out)))
        n_wins_list.append(int(len(scores_out)))

        # Live MLflow step metrics & trace every 60 frames
        if global_idx % 60 == 0 or global_idx == len(frames_list) - 1:
            snap = monitor.sample_once()
            mlflow.log_metrics(
                {
                    "live_service_ms": round(svc_ms, 2),
                    "live_pose_ms": round(p_ms, 2),
                    "live_track_ms": round(trk_ms, 2),
                    "live_score_ms": round(scr_ms, 2),
                    "live_queue_wait_ms": round(wait_ms, 2),
                    "live_temp_c": snap["temp_c"],
                    "live_power_total_w": snap["power_total_w"],
                    "live_power_core_w": snap["power_core_w"],
                    "live_rss_mb": snap["rss_mb"],
                    "live_sys_cpu_pct": snap["sys_cpu_pct"],
                },
                step=global_idx,
            )
            if global_idx in (0, len(frames_list) // 2, len(frames_list) - 1):
                traced_sample_frame(
                    name,
                    global_idx,
                    clip_name,
                    dec_ms,
                    p_ms,
                    trk_ms,
                    scr_ms,
                    int(len(b_in)),
                    int(len(tracks_out)),
                    int(len(scores_out)),
                    min_norm,
                    c1_reused,
                    {
                        "temp_c": snap["temp_c"],
                        "power_total_w": snap["power_total_w"],
                        "power_core_w": snap["power_core_w"],
                        "rss_mb": snap["rss_mb"],
                        "throttled_hex": snap["throttled_hex"],
                    },
                )

    wall_total_s = time.perf_counter() - t_wall_start
    cpu_total_s = time.process_time() - cpu_start
    tel_summary = monitor.stop()

    n_f = len(frames_list)
    mean_svc_ms = float(np.mean(total_ms_list))
    throughput_fps = 1000.0 / max(1e-6, mean_svc_ms)
    video_seconds = n_f / 24.0
    cpu_ms_per_video_s = (cpu_total_s * 1000.0) / video_seconds

    summary = {
        "arm": name,
        "description": arm_cfg["description"],
        "backend": backend,
        "imgsz": str(arm_cfg["imgsz"]),
        "threads": threads,
        "two_stage_tracker": two_stage,
        "use_c1": use_c1,
        "use_guard": use_guard,
        "frames": n_f,
        "wall_total_s": round(wall_total_s, 2),
        "cpu_total_s": round(cpu_total_s, 2),
        "throughput_fps": round(throughput_fps, 2),
        "Sustainable_24fps": bool(mean_svc_ms < (1000.0 / 24.0) and np.percentile(queue_wait_ms_list, 99) <= 2000.0),
        "decode_ms_mean": round(float(np.mean(decode_ms_list)), 3),
        "c1_check_ms_mean": round(float(np.mean(c1_ms_list)), 3),
        "pose_ms_mean": round(float(np.mean(pose_ms_list)), 3),
        "pose_ms_p50": round(float(np.median(pose_ms_list)), 3),
        "pose_ms_p95": round(float(np.percentile(pose_ms_list, 95)), 3),
        "track_ms_mean": round(float(np.mean(track_ms_list)), 3),
        "score_ms_mean": round(float(np.mean(score_ms_list)), 3),
        "service_ms_mean": round(mean_svc_ms, 3),
        "service_ms_p50": round(float(np.median(total_ms_list)), 3),
        "service_ms_p95": round(float(np.percentile(total_ms_list, 95)), 3),
        "service_ms_p99": round(float(np.percentile(total_ms_list, 99)), 3),
        "queue_wait_ms_p50": round(float(np.median(queue_wait_ms_list)), 2),
        "queue_wait_ms_p95": round(float(np.percentile(queue_wait_ms_list, 95)), 2),
        "queue_wait_ms_p99": round(float(np.percentile(queue_wait_ms_list, 99)), 2),
        "queue_wait_ms_max": round(float(np.max(queue_wait_ms_list)), 2),
        "cpu_ms_per_video_s": round(cpu_ms_per_video_s, 1),
        "c1_hit_rate_pct": round(100.0 * c1_hits / max(1, n_f), 2),
        "guard_shed_pct": round(100.0 * guard_sheds / max(1, n_f), 2),
        "late_windows_dropped": late_drops,
        "mean_detections_per_frame": round(float(np.mean(n_dets_list)), 2),
        "mean_tracks_per_frame": round(float(np.mean(n_tracks_list)), 2),
        "mean_windows_per_frame": round(float(np.mean(n_wins_list)), 2),
        "offline_auc_sht": arm_cfg["offline_auc_sht"],
        "causal_ideal_auc_sht": arm_cfg["causal_ideal_auc_sht"],
        "telemetry": tel_summary,
        "energy_j_per_frame": round(tel_summary["power_total_w_mean"] * (mean_svc_ms / 1000.0), 4),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{name}_summary.json").write_text(json.dumps(summary, indent=2))
    (out_dir / f"{name}_telemetry.json").write_text(json.dumps(monitor.samples, indent=2))
    np.savez_compressed(
        out_dir / f"{name}_trace.npz",
        decode_ms=np.array(decode_ms_list, dtype=np.float32),
        c1_ms=np.array(c1_ms_list, dtype=np.float32),
        pose_ms=np.array(pose_ms_list, dtype=np.float32),
        track_ms=np.array(track_ms_list, dtype=np.float32),
        score_ms=np.array(score_ms_list, dtype=np.float32),
        service_ms=np.array(total_ms_list, dtype=np.float32),
        queue_wait_ms=np.array(queue_wait_ms_list, dtype=np.float32),
        n_dets=np.array(n_dets_list, dtype=np.int32),
        n_tracks=np.array(n_tracks_list, dtype=np.int32),
        n_wins=np.array(n_wins_list, dtype=np.int32),
    )
    return summary


def get_arms() -> list[dict[str, Any]]:
    return [
        {
            "name": "Huang_RPi_PyTorch_640",
            "description": "Huang et al. `RPi` config: Stock PyTorch YOLOv8n-pose (640x640) + default BoT-SORT + STG-NF",
            "backend": "pytorch",
            "model_path": PI_EDGELAB_DIR / "artifacts/EXP-001/models/yolov8n-pose.pt",
            "imgsz": [640, 640],
            "threads": 4,
            "two_stage_tracker": False,
            "use_c1": False,
            "use_guard": False,
            "offline_auc_sht": 68.53,
            "causal_ideal_auc_sht": 66.03,
            "max_frames": 240,  # PyTorch @ 640 is ~3-6 FPS (~40-80s for 240 frames)
        },
        {
            "name": "Huang_RPi_Star_NCNN_320x320",
            "description": "Huang et al. `RPi*` config: Stock NCNN YOLOv8n-pose (320x320) + default BoT-SORT + STG-NF",
            "backend": "ncnn",
            "model_path": MODELS_ROOT / "yolov8n-pose-320-320x320-ncnn",
            "imgsz": [320, 320],
            "threads": 4,
            "two_stage_tracker": False,
            "use_c1": False,
            "use_guard": False,
            "offline_auc_sht": 57.28,
            "causal_ideal_auc_sht": 59.18,
            "max_frames": 720,
        },
        {
            "name": "Huang_RPi_Star_NCNN_192x320",
            "description": "Huang et al. `RPi*` 16:9 letterbox variant: Stock NCNN YOLOv8n-pose (192x320) + default BoT-SORT + STG-NF",
            "backend": "ncnn",
            "model_path": MODELS_ROOT / "yolov8n-pose-320-192x320-ncnn",
            "imgsz": [192, 320],
            "threads": 4,
            "two_stage_tracker": False,
            "use_c1": False,
            "use_guard": False,
            "offline_auc_sht": 57.28,
            "causal_ideal_auc_sht": 59.18,
            "max_frames": 720,
        },
        {
            "name": "Stock_NCNN_384x640",
            "description": "Stock NCNN FP16 YOLOv8n-pose (384x640) + default BoT-SORT + STG-NF",
            "backend": "ncnn",
            "model_path": MODELS_ROOT / "yolov8n-pose-640-384x640-ncnn",
            "imgsz": [384, 640],
            "threads": 4,
            "two_stage_tracker": False,
            "use_c1": False,
            "use_guard": False,
            "offline_auc_sht": 68.53,
            "causal_ideal_auc_sht": 66.03,
            "max_frames": 720,
        },
        {
            "name": "Ours_ML0_NCNN_640_FullRate",
            "description": "Our ML0 NCNN FP16 (384x640) + Locked Two-Stage BoT-SORT (c=0.20, low=0.05) + STG-NF (Full Rate)",
            "backend": "ncnn",
            "model_path": MODELS_ROOT / "exp020/ML0-640-384x640-ncnn",
            "imgsz": [384, 640],
            "threads": 4,
            "two_stage_tracker": True,
            "use_c1": False,
            "use_guard": False,
            "offline_auc_sht": 81.62,
            "causal_ideal_auc_sht": 76.82,
            "max_frames": 720,
        },
        {
            "name": "Ours_ML0_NCNN_640_CacheC1_Guard",
            "description": "Our Locked Winner: ML0 NCNN FP16 (384x640) + Two-Stage BoT-SORT + C1 Detector Reuse + Queue Guard",
            "backend": "ncnn",
            "model_path": MODELS_ROOT / "exp020/ML0-640-384x640-ncnn",
            "imgsz": [384, 640],
            "threads": 4,
            "two_stage_tracker": True,
            "use_c1": True,
            "use_guard": True,
            "offline_auc_sht": 81.62,
            "causal_ideal_auc_sht": 76.82,
            "max_frames": 720,
        },
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=PI_DATA_DIR / "pi5_huang_repro")
    parser.add_argument("--tracking-uri", type=str, default=os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    args = parser.parse_args()

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment("Pi5-Huang-Repro-and-Comparison")

    board = read_board_info()
    print(f"[Pi5] Board info: {json.dumps(board)}", flush=True)
    stgnf_model = load_model(STGNF_CKPT)

    all_summaries = []
    for arm in get_arms():
        summary_file = args.out_dir / f"{arm['name']}_summary.json"
        if summary_file.exists():
            cached = json.loads(summary_file.read_text())
            print(f"[Pi5] Loaded cached {arm['name']}: {cached['throughput_fps']} FPS", flush=True)
            all_summaries.append(cached)
            continue
        frames = load_frame_list(max_frames=arm["max_frames"])
        print(f"[Pi5] Running {arm['name']} on {len(frames)} frames ...", flush=True)
        with mlflow.start_run(run_name=arm["name"], description=arm["description"]) as run:
            mlflow.set_tags({
                "machine": "pi5",
                "board_model": board["board_model"],
                "cpu_governor": board["cpu_governor"],
                "arm_name": arm["name"],
                "backend": arm["backend"],
            })
            mlflow.log_params({
                "arm": arm["name"],
                "backend": arm["backend"],
                "imgsz": str(arm["imgsz"]),
                "threads": arm["threads"],
                "two_stage_tracker": arm["two_stage_tracker"],
                "use_c1": arm["use_c1"],
                "use_guard": arm["use_guard"],
                "frames": len(frames),
            })
            res = run_pi5_pipeline_arm(arm, frames, stgnf_model, args.out_dir)
            res["mlflow_run_id"] = run.info.run_id
            tel = res["telemetry"]
            mlflow.log_metrics({
                "throughput_fps": res["throughput_fps"],
                "service_ms_mean": res["service_ms_mean"],
                "service_ms_p95": res["service_ms_p95"],
                "pose_ms_mean": res["pose_ms_mean"],
                "track_ms_mean": res["track_ms_mean"],
                "score_ms_mean": res["score_ms_mean"],
                "decode_ms_mean": res["decode_ms_mean"],
                "c1_check_ms_mean": res["c1_check_ms_mean"],
                "queue_wait_ms_p99": res["queue_wait_ms_p99"],
                "queue_wait_ms_max": res["queue_wait_ms_max"],
                "cpu_ms_per_video_s": res["cpu_ms_per_video_s"],
                "c1_hit_rate_pct": res["c1_hit_rate_pct"],
                "offline_auc_sht": res["offline_auc_sht"],
                "causal_ideal_auc_sht": res["causal_ideal_auc_sht"],
                "temp_c_mean": tel["temp_c_mean"],
                "temp_c_max": tel["temp_c_max"],
                "power_total_w_mean": tel["power_total_w_mean"],
                "power_total_w_max": tel["power_total_w_max"],
                "power_core_w_mean": tel["power_core_w_mean"],
                "energy_j_per_frame": res["energy_j_per_frame"],
                "rss_mb_peak": tel["rss_mb_peak"],
                "sys_cpu_pct_mean": tel["sys_cpu_pct_mean"],
                "sys_cpu_pct_peak": tel["sys_cpu_pct_peak"],
            })
            mlflow.log_artifact(str(args.out_dir / f"{arm['name']}_summary.json"))
            mlflow.log_artifact(str(args.out_dir / f"{arm['name']}_telemetry.json"))
            all_summaries.append(res)
            print(
                f"[Pi5] DONE {arm['name']}: {res['throughput_fps']} FPS "
                f"(service={res['service_ms_mean']:.1f}ms, pose={res['pose_ms_mean']:.1f}ms, "
                f"track={res['track_ms_mean']:.2f}ms, score={res['score_ms_mean']:.2f}ms, "
                f"power={tel['power_total_w_mean']:.2f}W, temp={tel['temp_c_mean']:.1f}C, "
                f"RSS={tel['rss_mb_peak']:.0f}MB, run_id={run.info.run_id})",
                flush=True,
            )

    mlflow.flush_trace_async_logging()
    (args.out_dir / "pi5_huang_repro_all.json").write_text(
        json.dumps({"board": board, "arms": all_summaries}, indent=2)
    )


if __name__ == "__main__":
    main()
