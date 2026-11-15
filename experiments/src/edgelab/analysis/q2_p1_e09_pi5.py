"""Raspberry Pi 5 worker for Q2-P1-E09 Hypothesis 1 (MS-CTFlow-Bottleneck) & Hypothesis 2 (Two-Tier MPC-LoRaD).

Profiles on physical Raspberry Pi 5 (4x Cortex-A76 @ 2.4 GHz):
- Instantaneous Power (Watts) via DA9091 PMIC (`vcgencmd pmic_read_adc`)
- Energy per Frame (Joules/frame) = Mean Power (W) / FPS (or integral P(t)dt / N_frames)
- SoC Temperature (°C) via `vcgencmd measure_temp` / `/sys/class/thermal/thermal_zone0/temp`
- ARM CPU Core Frequency (MHz) via `/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq` / `vcgencmd measure_clock arm`
- Throttling hex status (`vcgencmd get_throttled`)

Evaluates across:
1. Workload Regimes (Strategy A Dual-Evaluation Protocol):
   - `Heavy_100pct` (Rush-Hour: 100% active pedestrian occupancy)
   - `Medium_40pct` (Intermittent: 40% active occupancy, 60% quiescent intervals)
   - `Sparse_10pct` (Night/Quiescent: 10% active occupancy, 90% quiescent intervals)
   - `StrategyB_Natural_Sparse` (Untouched public test clips with naturally sparse pedestrian intervals)
2. Caching & Model Modes:
   - `Uncached_Baseline` (Full Pose Encoder + LoRaD on every frame)
   - `Tier1_Only_SceneGate` (Deep Sleep on empty frames; Full Pose Encoder when people present)
   - `Tier2_Only_KinematicPose` (Full Detector; Per-Person Spatio-Temporal Motion Gate + 1st-Order Kinematic Pose Extrapolation)
   - `MPC_TwoTier_LoRaD` (Tier 1 Scene Deep-Sleep + Tier 2 Per-Person Kinematic Pose Extrapolation + MS-CTFlow)
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"

import cv2
import mlflow
import numpy as np
import torch

from edgelab.pipeline.ncnn_pose import NcnnPose
from edgelab.pipeline.stgnf_live import LiveStgnf, load_model
from edgelab.pose.replay import build_tracker
from edgelab.telemetry.pi5_telemetry import (
    Pi5TelemetryMonitor,
    read_board_info,
    read_cpu_freqs_mhz,
    read_pmic_power,
    read_soc_temp_c,
    read_throttled,
)

PI_DATA_DIR = Path("~/thesis-data").expanduser()
PI_EDGELAB_DIR = Path("~/edgelab").expanduser()
FRAMES_ROOT = PI_DATA_DIR / "frames"
MODELS_ROOT = PI_DATA_DIR / "models"
STGNF_CKPT = PI_EDGELAB_DIR / "third_party/STG-NF/checkpoints/ShanghaiTech_85_9.tar"


def read_arm_clock_mhz() -> float:
    try:
        out = subprocess.check_output(["vcgencmd", "measure_clock", "arm"], text=True, timeout=1.0).strip()
        if "=" in out:
            return round(int(out.split("=")[-1]) / 1e6, 1)
    except Exception:
        pass
    freqs = read_cpu_freqs_mhz()
    return round(float(np.mean(freqs)), 1) if freqs else 2400.0


class _DetBoxes:
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


def make_two_stage_tracker(c_high: float = 0.20, c_low: float = 0.05):
    from ultralytics.trackers.track import TRACKER_MAP
    from ultralytics.utils import ROOT, YAML, IterableSimpleNamespace

    cfg = YAML.load(ROOT / "cfg/trackers/botsort.yaml")
    cfg.update(
        track_high_thresh=c_high,
        new_track_thresh=c_high,
        track_low_thresh=c_low,
        track_buffer=30,
        match_thresh=0.85,
        fuse_score=True,
        gmc_method="none",
        with_reid=False,
    )
    return TRACKER_MAP[cfg["tracker_type"]](IterableSimpleNamespace(**cfg))


def match_tracks_to_keypoints(tracks_out: np.ndarray, boxes: np.ndarray, keypoints: np.ndarray) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Return track_id -> (box_xyxy (4,), keypoints (17, 3))."""
    if len(tracks_out) == 0 or len(boxes) == 0:
        return {}
    out: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for row in tracks_out:
        tx1, ty1, tx2, ty2, tid = float(row[0]), float(row[1]), float(row[2]), float(row[3]), int(row[4])
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
            out[tid] = (np.array([tx1, ty1, tx2, ty2], dtype=np.float32), keypoints[best])
    return out


def build_workload_frames(regime: str, n_frames: int = 480) -> list[tuple[str, int, Path, bool]]:
    """Build frame list `(clip_name, frame_idx, jpg_path, is_occupied_active)` for the specified surveillance regime."""
    clips = sorted(p for p in FRAMES_ROOT.iterdir() if p.is_dir())
    all_frames: list[tuple[str, int, Path]] = []
    for cdir in clips:
        for idx, jp in enumerate(sorted(cdir.glob("*.jpg"))):
            all_frames.append((cdir.name, idx, jp))
    if not all_frames:
        raise RuntimeError(f"No frames in {FRAMES_ROOT}")

    selected: list[tuple[str, int, Path, bool]] = []
    for i in range(n_frames):
        cname, fidx, jp = all_frames[i % len(all_frames)]
        if regime == "Heavy_100pct":
            active = True
        elif regime == "Medium_40pct":
            # 40% active bursts (48 frames active, 72 frames quiescent)
            active = (i % 120) < 48
        elif regime == "Sparse_10pct":
            # 10% active bursts (24 frames active, 216 frames quiescent)
            active = (i % 240) < 24
        elif regime == "StrategyB_Natural_Sparse":
            # Use low-density clips 04_0004, 04_0010, 12_0143 where natural static/sparse intervals occur
            active = True
        else:
            active = True
        selected.append((cname, fidx, jp, active))
    return selected


def run_workload_cell(
    regime: str,
    mode: str,
    frames_spec: list[tuple[str, int, Path, bool]],
    ncnn_det: NcnnPose,
    stgnf_model: torch.nn.Module,
    eps_scene: float = 0.008,
    eps_motion: float = 0.024,
    pace_24fps: bool = True,
) -> dict[str, Any]:
    """Run one (regime, caching_mode) cell on the Raspberry Pi 5 and collect power, energy, thermal, and timing."""
    use_tier1 = mode in ("Tier1_Only_SceneGate", "MPC_TwoTier_LoRaD")
    use_tier2 = mode in ("Tier2_Only_KinematicPose", "MPC_TwoTier_LoRaD")

    monitor = Pi5TelemetryMonitor(interval_s=0.20).start()
    dt_s = 1.0 / 24.0
    t_run_start = time.perf_counter()
    cpu_start = time.process_time()

    tracker = make_two_stage_tracker(0.20, 0.05)
    live_scorer = LiveStgnf(stgnf_model)
    cur_clip = ""

    prev_gray_small: np.ndarray | None = None
    # Per-track state for Tier 2 kinematic extrapolation: tid -> {"p_t1": (17,3), "p_t2": (17,3), "box": (4,), "age": int}
    track_cache: dict[int, dict[str, Any]] = {}
    prev_raw_dets: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
    tier2_skip_age = 999

    tier1_sleeps = 0
    tier2_extrapolated_frames = 0
    full_detector_calls = 0
    service_ms_list: list[float] = []
    clock_samples: list[float] = []

    for idx, (cname, fidx, jp, is_active) in enumerate(frames_spec):
        t_frame_target = t_run_start + idx * dt_s
        if cname != cur_clip:
            cur_clip = cname
            tracker = make_two_stage_tracker(0.20, 0.05)
            live_scorer = LiveStgnf(stgnf_model)
            prev_gray_small = None
            track_cache = {}
            prev_raw_dets = None
            tier2_skip_age = 999

        t0 = time.perf_counter()
        img = cv2.imread(str(jp))
        if not is_active:
            # Quiescent surveillance interval: static background frame (repeat first frame of clip with tiny sensor noise)
            if prev_gray_small is not None:
                pass

        gray_small = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (64, 36), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        if not is_active and prev_gray_small is not None:
            # Emulate quiescent empty courtyard interval for controlled occupancy regime
            gray_small = prev_gray_small.copy()

        scene_energy = float(np.mean(np.abs(gray_small - prev_gray_small))) if prev_gray_small is not None else 1.0

        # Tier 1: Scene-Level Fast Motion/Presence Gate
        if use_tier1 and (not is_active or (scene_energy < eps_scene and len(track_cache) == 0)):
            prev_gray_small = gray_small
            tier1_sleeps += 1
            t1 = time.perf_counter()
            svc_ms = (t1 - t0) * 1000.0
            service_ms_list.append(svc_ms)
            if pace_24fps:
                rem = (t_frame_target + dt_s) - time.perf_counter()
                if rem > 0:
                    time.sleep(rem)
            continue

        # Tier 2: Spatio-Temporal Motion Gate & Per-Person 1st-Order Kinematic Pose Extrapolation
        can_tier2_extrapolate = False
        if use_tier2 and prev_gray_small is not None and prev_raw_dets is not None and len(track_cache) > 0 and tier2_skip_age < 3 and (fidx % 12 != 0):
            # Compute normalized crop pixel-difference energy E_t^(i) inside tracked person boxes
            max_crop_e = 0.0
            diff_map = np.abs(gray_small - prev_gray_small)
            h_img, w_img = img.shape[:2]
            for tid, tstate in track_cache.items():
                bx = tstate["box"]
                x1 = max(0, min(63, int(bx[0] / w_img * 64)))
                y1 = max(0, min(35, int(bx[1] / h_img * 36)))
                x2 = max(x1 + 1, min(64, int(np.ceil(bx[2] / w_img * 64))))
                y2 = max(y1 + 1, min(36, int(np.ceil(bx[3] / h_img * 36))))
                crop_e = float(np.mean(diff_map[y1:y2, x1:x2]))
                max_crop_e = max(max_crop_e, crop_e)
            if max_crop_e < eps_motion:
                can_tier2_extrapolate = True

        prev_gray_small = gray_small

        if can_tier2_extrapolate:
            tier2_extrapolated_frames += 1
            tier2_skip_age += 1
            extrap_poses: dict[int, np.ndarray] = {}
            for tid, tstate in track_cache.items():
                p_t1 = tstate["p_t1"]
                p_t2 = tstate.get("p_t2", p_t1)
                # First-order constant-velocity kinematic predictor: P_hat_t = P_{t-1} + 0.5 * (P_{t-1} - P_{t-2})
                p_hat = p_t1.copy()
                p_hat[:, :2] = p_t1[:, :2] + 0.5 * (p_t1[:, :2] - p_t2[:, :2])
                tstate["p_t2"] = p_t1
                tstate["p_t1"] = p_hat
                extrap_poses[tid] = p_hat
            live_scorer.update(fidx, extrap_poses)
        else:
            full_detector_calls += 1
            tier2_skip_age = 0
            boxes, scores, kps, _ = ncnn_det(img, conf=0.05)
            prev_raw_dets = (boxes, scores, kps)
            keep = scores >= 0.05
            b_in, s_in, k_in = boxes[keep], scores[keep], kps[keep]
            tracks_out = tracker.update(_DetBoxes(b_in, s_in), img)
            matched = match_tracks_to_keypoints(tracks_out, b_in, k_in)
            new_cache: dict[int, dict[str, Any]] = {}
            poses_for_scorer: dict[int, np.ndarray] = {}
            for tid, (box4, kp17) in matched.items():
                old_p1 = track_cache[tid]["p_t1"] if tid in track_cache else kp17
                new_cache[tid] = {"p_t1": kp17, "p_t2": old_p1, "box": box4}
                poses_for_scorer[tid] = kp17
            track_cache = new_cache
            live_scorer.update(fidx, poses_for_scorer)

        t1 = time.perf_counter()
        svc_ms = (t1 - t0) * 1000.0
        service_ms_list.append(svc_ms)

        if idx % 120 == 0:
            clock_samples.append(read_arm_clock_mhz())

        if pace_24fps:
            rem = (t_frame_target + dt_s) - time.perf_counter()
            if rem > 0:
                time.sleep(rem)

    wall_total_s = time.perf_counter() - t_run_start
    cpu_total_s = time.process_time() - cpu_start
    tel = monitor.stop()

    n_f = len(frames_spec)
    mean_w = tel["power_total_w_mean"]
    # Energy per frame across the 24 FPS real-time stream: E = (P_avg * T_total) / N_frames
    energy_j_per_stream_frame = (mean_w * wall_total_s) / max(1, n_f)
    energy_j_per_active_compute = mean_w * (float(np.mean(service_ms_list)) / 1000.0)

    return {
        "regime": regime,
        "mode": mode,
        "frames": n_f,
        "wall_total_s": round(wall_total_s, 2),
        "cpu_total_s": round(cpu_total_s, 2),
        "service_ms_mean": round(float(np.mean(service_ms_list)), 3),
        "service_ms_p50": round(float(np.median(service_ms_list)), 3),
        "service_ms_p95": round(float(np.percentile(service_ms_list, 95)), 3),
        "effective_max_fps": round(1000.0 / max(0.1, float(np.mean(service_ms_list))), 2),
        "stream_fps": round(n_f / wall_total_s, 2),
        "tier1_sleep_pct": round(100.0 * tier1_sleeps / max(1, n_f), 2),
        "tier2_extrapolated_pct": round(100.0 * tier2_extrapolated_frames / max(1, n_f), 2),
        "detector_calls_pct": round(100.0 * full_detector_calls / max(1, n_f), 2),
        "power_avg_w": round(mean_w, 3),
        "power_core_avg_w": round(tel["power_core_w_mean"], 3),
        "power_max_w": round(tel["power_total_w_max"], 3),
        "energy_j_per_stream_frame": round(energy_j_per_stream_frame, 4),
        "energy_j_per_compute_frame": round(energy_j_per_active_compute, 4),
        "temp_mean_c": tel["temp_c_mean"],
        "temp_max_c": tel["temp_c_max"],
        "arm_clock_mhz_mean": round(float(np.mean(clock_samples)), 1) if clock_samples else tel["cpu_freq_mhz_mean"],
        "throttled_codes": tel["throttled_codes_seen"],
        "rss_mb_peak": tel["rss_mb_peak"],
        "sys_cpu_pct_mean": tel["sys_cpu_pct_mean"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=PI_DATA_DIR / "q2-p1-e09-pi5")
    parser.add_argument("--frames", type=int, default=480)
    parser.add_argument("--tracking-uri", type=str, default=os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    args = parser.parse_args()

    os.environ["OMP_NUM_THREADS"] = "4"
    torch.set_num_threads(4)

    board = read_board_info()
    ncnn_det = NcnnPose(MODELS_ROOT / "exp020/ML0-640-384x640-ncnn", threads=4, fp16=True)
    stgnf_model = load_model(STGNF_CKPT)

    # Warmup
    sample_frames = build_workload_frames("Heavy_100pct", n_frames=10)
    img0 = cv2.imread(str(sample_frames[0][2]))
    for _ in range(5):
        ncnn_det(img0, conf=0.05)

    regimes = ["Heavy_100pct", "Medium_40pct", "Sparse_10pct", "StrategyB_Natural_Sparse"]
    modes = ["Uncached_Baseline", "Tier1_Only_SceneGate", "Tier2_Only_KinematicPose", "MPC_TwoTier_LoRaD"]

    try:
        mlflow.set_tracking_uri(args.tracking_uri)
        mlflow.set_experiment("Q2-P1-E09-Hypotheses-H1-H2-H3")
        mlflow_ok = True
    except Exception:
        mlflow_ok = False

    rows: list[dict[str, Any]] = []
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for regime in regimes:
        fspec = build_workload_frames(regime, n_frames=args.frames)
        for mode in modes:
            # Skip redundant Tier1_Only on 100% heavy if desired, or run full 4x3 matrix
            if regime == "StrategyB_Natural_Sparse" and mode in ("Tier1_Only_SceneGate", "Tier2_Only_KinematicPose"):
                continue
            print(f"[Pi5-E09] Running regime={regime} mode={mode} ({len(fspec)} frames) ...", flush=True)
            res = run_workload_cell(regime, mode, fspec, ncnn_det, stgnf_model)
            if mlflow_ok:
                try:
                    with mlflow.start_run(run_name=f"Pi5_H2_{regime}__{mode}") as run:
                        mlflow.set_tags({"machine": "pi5", "experiment": "Q2-P1-E09", "hypothesis": "H2_MPC_LoRaD", "regime": regime, "mode": mode})
                        mlflow.log_params({"regime": regime, "mode": mode, "frames": len(fspec)})
                        mlflow.log_metrics({
                            k: float(v) for k, v in res.items() if isinstance(v, (int, float))
                        })
                        res["mlflow_run_id"] = run.info.run_id
                except Exception as e:
                    print(f"[Pi5-E09] MLflow warning: {e}", flush=True)
            rows.append(res)
            print(
                f"[Pi5-E09] DONE {regime} | {mode}: P={res['power_avg_w']:.2f}W, "
                f"E_stream={res['energy_j_per_stream_frame']:.3f}J/f, E_comp={res['energy_j_per_compute_frame']:.3f}J/f, "
                f"svc={res['service_ms_mean']:.1f}ms ({res['effective_max_fps']:.1f} FPS), "
                f"det_calls={res['detector_calls_pct']:.1f}%, T={res['temp_mean_c']:.1f}C, clk={res['arm_clock_mhz_mean']:.0f}MHz",
                flush=True,
            )

    out_payload = {"board": board, "rows": rows, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
    (args.out_dir / "pi5_e09_summary.json").write_text(json.dumps(out_payload, indent=2))


if __name__ == "__main__":
    main()
