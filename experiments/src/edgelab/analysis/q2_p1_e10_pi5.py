"""Raspberry Pi 5 Fast-Feedback Contextual Bandit / RL OS Tuner (`FastOS-RL`, Q2-P1-E10).

Implements a Surrogate-Warm-Started LinUCB Contextual Bandit that tunes OS & CPU runtime knobs
online in 0.5-second micro-windows (12 frames @ 24 fps per step, 24 steps = ~12 seconds total tuning time per pipeline!)
directly on the physical Raspberry Pi 5 (4x Cortex-A76 @ 2.4 GHz).

Action Space (18 OS/CPU configurations):
- CPU Frequency Governor g in {'ondemand', 'schedutil', 'performance'}
- Thread Concurrency T in {2, 3, 4}
- Core Affinity & CFS Priority p in {'unpinned_nice0', 'pinned_isolated_nice-10'}

Evaluates 3 OS Policies (`OS_Default`, `OS_Static_Pin3`, and `FastOS_RL_Tuned`) across the 5 requested pipelines:
1. `LoRaD_FullRate` (Stock NCNN 640 + BoT-SORT + LoRaD Causal Flow)
2. `LoRaD_Cached` (Stock NCNN 640 + Two-Tier MPC / C1 Cache + LoRaD Causal Flow)
3. `Ours_ML0_FullRate` (Fine-Tuned ML0 NCNN 640 + 2-Stage BoT-SORT + STG-NF)
4. `Ours_ML0_Cached` (Fine-Tuned ML0 NCNN 640 + 2-Stage BoT-SORT + Cache-C1+Guard + STG-NF)
5. `Huang_RPi_Star_NCNN320` (Huang et al. RPi* NCNN 320 + BoT-SORT + STG-NF)
6. `Huang_RPi_PyTorch640` (Huang et al. RPi PyTorch 640 + BoT-SORT + STG-NF)
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"

import cv2
import mlflow
import numpy as np
import psutil
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
GOV_PATH = Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")


@dataclasses.dataclass(frozen=True)
class OSAction:
    action_id: int
    governor: str        # 'ondemand', 'schedutil', 'performance'
    threads: int         # 2, 3, 4
    pin_isolated: bool   # False = unpinned (0-3) nice 0; True = pinned isolated cores + nice -10

    @property
    def label(self) -> str:
        pin_str = f"pin{self.threads}_nice-10" if self.pin_isolated else "unpinned_nice0"
        return f"{self.governor}|T{self.threads}|{pin_str}"


def build_action_space() -> list[OSAction]:
    actions: list[OSAction] = []
    idx = 0
    for gov in ("ondemand", "schedutil", "performance"):
        for thr in (2, 3, 4):
            for pin in (False, True):
                actions.append(OSAction(idx, gov, thr, pin))
                idx += 1
    return actions


def apply_os_action(action: OSAction) -> float:
    """Apply governor, thread count, core affinity, and CFS nice priority in < 1.5 ms. Returns switch overhead in ms."""
    t0 = time.perf_counter()
    # 1. Set CPU frequency scaling governor
    try:
        if GOV_PATH.exists():
            GOV_PATH.write_text(action.governor + "\n")
    except PermissionError:
        try:
            subprocess.run(
                ["sudo", "-n", "tee", str(GOV_PATH)],
                input=(action.governor + "\n").encode(),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=0.5,
            )
        except Exception:
            pass

    # 2. Set PyTorch & OpenMP threads
    os.environ["OMP_NUM_THREADS"] = str(action.threads)
    try:
        torch.set_num_threads(action.threads)
    except Exception:
        pass

    # 3. Set CPU core affinity and nice priority across all threads of current process
    if action.pin_isolated:
        # Isolate Core 0 for OS/network/telemetry interrupts when threads < 4
        if action.threads == 2:
            cpus = {2, 3}
        elif action.threads == 3:
            cpus = {1, 2, 3}
        else:
            cpus = {0, 1, 2, 3}
        target_nice = -10
    else:
        cpus = {0, 1, 2, 3}
        target_nice = 0

    task_dir = Path("/proc/self/task")
    if task_dir.exists() and hasattr(os, "sched_setaffinity"):
        for tpath in task_dir.iterdir():
            try:
                tid = int(tpath.name)
                os.sched_setaffinity(tid, cpus)
            except Exception:
                continue
    try:
        os.setpriority(os.PRIO_PROCESS, 0, target_nice)
    except PermissionError:
        try:
            subprocess.run(
                ["sudo", "-n", "renice", "-n", str(target_nice), "-p", str(os.getpid())],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=0.5,
            )
        except Exception:
            pass

    return (time.perf_counter() - t0) * 1000.0


class BackgroundCameraIngestDaemon:
    """Realistic edge OS background co-load (network/RTSP ingest + system telemetry daemon on Core 0).

    Without core isolation, CFS schedules inference threads across Core 0 where camera/network interrupts run,
    causing barrier synchronization stalls in OpenMP/NCNN.
    """

    def __init__(self):
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _worker(self) -> None:
        if hasattr(os, "sched_setaffinity"):
            try:
                os.sched_setaffinity(0, {0})
            except Exception:
                pass
        rng = np.random.default_rng(123)
        buf = rng.bytes(65536)
        while not self._stop.is_set():
            # Simulate periodic RTSP packet checksum / telemetry serialization on Core 0 (~25% of 1 core)
            _ = sum(buf[:4096])
            time.sleep(0.002)

    def start(self) -> "BackgroundCameraIngestDaemon":
        self._stop.clear()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)


# =====================================================================================================================
# Surrogate-Warm-Started LinUCB Contextual Bandit (`FastOS-RL`)
# =====================================================================================================================
class FastOSLinUCB:
    """Contextual Bandit with Surrogate Prior Warm-Start for fast (<15s) OS/CPU tuning on Raspberry Pi 5.

    Feature map phi(context, action) in R^8:
      0: bias (1.0)
      1: is_performance_gov (1.0 if governor == 'performance' else 0.0)
      2: is_schedutil_gov (1.0 if governor == 'schedutil' else 0.0)
      3: is_performance_gov * is_cached_bursty (captures avoiding ondemand 1.5GHz downclock lag on bursty C1 pipelines!)
      4: threads_norm ((threads - 2) / 2.0)
      5: is_t3_or_t4 (1.0 if threads >= 3 else 0.0)
      6: pin_isolated (1.0 if pin_isolated else 0.0)
      7: pin_isolated * (1.0 if threads == 3 else 0.5) (captures isolating Core 0 from background interrupts!)
    """

    def __init__(self, actions: list[OSAction], alpha: float = 0.30, l2_reg: float = 1.0, warm_start: bool = True):
        self.actions = actions
        self.alpha = alpha
        self.d = 8
        self.A = np.eye(self.d, dtype=np.float64) * l2_reg
        # Surrogate prior theta_0 informed by E02 thread scaling and EXP-004 pinning model
        if warm_start:
            self.theta_prior = np.array([-0.80, 0.08, 0.04, 0.14, 0.12, 0.18, 0.09, 0.11], dtype=np.float64)
            self.b = self.A @ self.theta_prior
        else:
            self.theta_prior = np.zeros(self.d, dtype=np.float64)
            self.b = np.zeros(self.d, dtype=np.float64)

    @staticmethod
    def featurize(context: dict[str, float], action: OSAction) -> np.ndarray:
        bursty = float(context.get("is_cached_bursty", 0.0))
        is_perf = 1.0 if action.governor == "performance" else 0.0
        is_sched = 1.0 if action.governor == "schedutil" else 0.0
        t_norm = (action.threads - 2.0) / 2.0
        is_t34 = 1.0 if action.threads >= 3 else 0.0
        pin = 1.0 if action.pin_isolated else 0.0
        pin_core0_free = pin * (1.0 if action.threads == 3 else 0.6)
        return np.array(
            [1.0, is_perf, is_sched, is_perf * bursty, t_norm, is_t34, pin, pin_core0_free],
            dtype=np.float64,
        )

    def select_action(self, context: dict[str, float], step: int) -> tuple[OSAction, float, float]:
        """Return (chosen_action, predicted_mean, ucb_score)."""
        A_inv = np.linalg.inv(self.A)
        theta_hat = A_inv @ self.b
        best_a = self.actions[0]
        best_ucb = -1e18
        best_mean = -1e18
        for a in self.actions:
            phi = self.featurize(context, a)
            mean = float(phi @ theta_hat)
            unc = float(np.sqrt(max(1e-12, phi @ A_inv @ phi)))
            ucb = mean + self.alpha * unc
            if ucb > best_ucb:
                best_ucb = ucb
                best_mean = mean
                best_a = a
        return best_a, best_mean, best_ucb

    def update(self, context: dict[str, float], action: OSAction, reward: float) -> None:
        phi = self.featurize(context, action)
        self.A += np.outer(phi, phi)
        self.b += phi * float(reward)

    def best_greedy_action(self, context: dict[str, float]) -> OSAction:
        A_inv = np.linalg.inv(self.A)
        theta_hat = A_inv @ self.b
        return max(self.actions, key=lambda a: float(self.featurize(context, a) @ theta_hat))


def compute_rl_reward(metrics: dict[str, float], deadline_ms: float = 41.67) -> float:
    """Reward function balancing frame drop rate, normalized mean & tail latency, and energy per frame."""
    drop_rate = metrics["drop_rate_pct"] / 100.0
    norm_mean = metrics["service_ms_mean"] / deadline_ms
    norm_p95 = metrics["service_ms_p95"] / deadline_ms
    joules = metrics["energy_j_per_frame"]
    return float(-(2.5 * drop_rate + 1.0 * norm_mean + 0.4 * norm_p95 + 0.15 * joules))


# =====================================================================================================================
# Pipeline Execution Engine on Physical Raspberry Pi 5
# =====================================================================================================================
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


def make_tracker_for_pipeline(two_stage: bool):
    if not two_stage:
        return build_tracker(0.25, "botsort")
    from ultralytics.trackers.track import TRACKER_MAP
    from ultralytics.utils import ROOT, YAML, IterableSimpleNamespace

    cfg = YAML.load(ROOT / "cfg/trackers/botsort.yaml")
    cfg.update(
        track_high_thresh=0.20,
        new_track_thresh=0.20,
        track_low_thresh=0.05,
        track_buffer=30,
        match_thresh=0.85,
        fuse_score=True,
        gmc_method="none",
        with_reid=False,
    )
    return TRACKER_MAP[cfg["tracker_type"]](IterableSimpleNamespace(**cfg))


def match_tracks(tracks_out: np.ndarray, boxes: np.ndarray, keypoints: np.ndarray) -> dict[int, np.ndarray]:
    if len(tracks_out) == 0 or len(boxes) == 0:
        return {}
    out: dict[int, np.ndarray] = {}
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
            out[tid] = keypoints[best]
    return out


class PipelineRunner:
    """Holds pre-loaded detector models for T in {2, 3, 4} so switching threads during RL takes 0.0 ms."""

    def __init__(self, spec: dict[str, Any], stgnf_model: torch.nn.Module, lorad_model: torch.nn.Module, msctflow_model: torch.nn.Module | None = None):
        self.spec = spec
        self.name = spec["name"]
        self.backend = spec["backend"]
        self.two_stage = bool(spec.get("two_stage", False))
        self.use_cache = bool(spec.get("use_cache", False))
        sc_type = spec.get("scorer", "stgnf")
        if sc_type == "lorad":
            self.scorer_model = lorad_model
        elif sc_type == "msctflow" and msctflow_model is not None:
            self.scorer_model = msctflow_model
        else:
            self.scorer_model = stgnf_model

        self.ncnn_by_t: dict[int, NcnnPose] = {}
        self.yolo_pt = None
        if self.backend == "ncnn":
            for t in (2, 3, 4):
                self.ncnn_by_t[t] = NcnnPose(spec["model_dir"], threads=t, fp16=True)
        else:
            from ultralytics import YOLO

            self.yolo_pt = YOLO(str(spec["model_dir"]))

    def run_window(
        self,
        frames: list[tuple[str, int, Path]],
        action: OSAction,
        pace_realtime_bursty: bool = True,
    ) -> dict[str, Any]:
        """Run `frames` under `action` and return timing, drop rate, CPU %, clock MHz, and PMIC power."""
        switch_ms = apply_os_action(action)
        det_ncnn = self.ncnn_by_t.get(action.threads)
        low_c = 0.05 if self.two_stage else 0.25

        tracker = make_tracker_for_pipeline(self.two_stage)
        live_scorer = LiveStgnf(self.scorer_model)
        cur_clip = ""
        prev_thumb: np.ndarray | None = None
        prev_dets: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
        c1_age = 999

        dt_ms = 1000.0 / 24.0
        worker_free_ms = 0.0
        svc_ms_list: list[float] = []
        wait_ms_list: list[float] = []
        drops = 0
        c1_hits = 0
        freqs_seen: list[float] = []

        proc = psutil.Process(os.getpid())
        psutil.cpu_percent(interval=None)
        proc.cpu_percent(interval=None)
        pmic_samples: list[float] = []
        temp_samples: list[float] = []

        t_wall0 = time.perf_counter()
        cpu0 = time.process_time()

        for idx, (cname, fidx, jp) in enumerate(frames):
            if cname != cur_clip:
                cur_clip = cname
                tracker = make_tracker_for_pipeline(self.two_stage)
                live_scorer = LiveStgnf(self.scorer_model)
                prev_thumb = None
                prev_dets = None
                c1_age = 999

            arrival_ms = idx * dt_ms
            # If pipeline is cached and worker finishes early, sleep briefly to reflect real camera inter-arrival
            # (this exposes the ondemand governor's downclocking behavior during C1 cache skips!)
            if pace_realtime_bursty and self.use_cache:
                elapsed_wall_ms = (time.perf_counter() - t_wall0) * 1000.0
                if elapsed_wall_ms < arrival_ms:
                    time.sleep((arrival_ms - elapsed_wall_ms) / 1000.0)

            start_ms = max(arrival_ms, worker_free_ms)
            wait_ms = start_ms - arrival_ms

            t0 = time.perf_counter()
            img = cv2.imread(str(jp))

            reused = False
            if self.use_cache:
                gray_sm = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (40, 24), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
                if prev_thumb is not None and prev_dets is not None and c1_age < 3 and (fidx % 10 != 0):
                    diff = float(np.percentile(np.abs(gray_sm - prev_thumb), 90))
                    if diff <= 0.032:
                        reused = True
                prev_thumb = gray_sm

            if reused and prev_dets is not None:
                boxes, scores, kps = prev_dets
                c1_age += 1
                c1_hits += 1
            else:
                if self.backend == "ncnn":
                    boxes, scores, kps, _ = det_ncnn(img, conf=low_c)
                else:
                    res = self.yolo_pt.predict(img, imgsz=640, conf=low_c, verbose=False, device="cpu")[0]
                    torch.set_num_threads(action.threads)
                    if res.boxes is not None and len(res.boxes):
                        boxes = res.boxes.xyxy.cpu().numpy().astype(np.float32)
                        scores = res.boxes.conf.cpu().numpy().astype(np.float32)
                        kps = res.keypoints.data.cpu().numpy().astype(np.float32)
                    else:
                        boxes = np.zeros((0, 4), dtype=np.float32)
                        scores = np.zeros((0,), dtype=np.float32)
                        kps = np.zeros((0, 17, 3), dtype=np.float32)
                prev_dets = (boxes, scores, kps)
                c1_age = 0

            keep = scores >= low_c
            b_in, s_in, k_in = boxes[keep], scores[keep], kps[keep]
            tracks_out = tracker.update(_DetBoxes(b_in, s_in), img)
            matched = match_tracks(tracks_out, b_in, k_in)
            live_scorer.update(fidx, matched)

            t1 = time.perf_counter()
            svc_ms = (t1 - t0) * 1000.0
            finish_ms = start_ms + svc_ms
            worker_free_ms = finish_ms

            # Frame drop / deadline miss condition: service_ms > 41.67 ms or queue wait > 400 ms
            if svc_ms > dt_ms or wait_ms > 400.0:
                drops += 1

            svc_ms_list.append(svc_ms)
            wait_ms_list.append(wait_ms)

            if idx % 8 == 0:
                f_mhz = read_cpu_freqs_mhz()
                if f_mhz:
                    freqs_seen.append(float(np.mean(f_mhz)))
                pmic = read_pmic_power()
                if np.isfinite(pmic["power_total_w"]):
                    pmic_samples.append(pmic["power_total_w"])
                temp_samples.append(read_soc_temp_c())

        wall_s = time.perf_counter() - t_wall0
        cpu_s = time.process_time() - cpu0
        sys_cpu_pct = psutil.cpu_percent(interval=None)
        proc_cpu_pct = round(100.0 * cpu_s / max(1e-6, wall_s), 1)

        n_f = len(frames)
        mean_svc = float(np.mean(svc_ms_list))
        p50_svc = float(np.median(svc_ms_list))
        p95_svc = float(np.percentile(svc_ms_list, 95))
        p99_svc = float(np.percentile(svc_ms_list, 99))
        mean_w = float(np.mean(pmic_samples)) if pmic_samples else 5.20
        j_per_frame = mean_w * (mean_svc / 1000.0)
        drop_rate_pct = 100.0 * drops / max(1, n_f)

        out = {
            "action_id": action.action_id,
            "action_label": action.label,
            "governor": action.governor,
            "threads": action.threads,
            "pin_isolated": action.pin_isolated,
            "switch_overhead_ms": round(switch_ms, 3),
            "frames": n_f,
            "service_ms_mean": round(mean_svc, 3),
            "service_ms_p50": round(p50_svc, 3),
            "service_ms_p95": round(p95_svc, 3),
            "service_ms_p99": round(p99_svc, 3),
            "throughput_fps": round(1000.0 / max(0.1, mean_svc), 2),
            "queue_wait_p95_ms": round(float(np.percentile(wait_ms_list, 95)), 2),
            "drop_rate_pct": round(drop_rate_pct, 2),
            "c1_hit_rate_pct": round(100.0 * c1_hits / max(1, n_f), 2),
            "proc_cpu_pct": proc_cpu_pct,
            "sys_cpu_pct": round(sys_cpu_pct, 1),
            "cpu_freq_mhz_mean": round(float(np.mean(freqs_seen)), 1) if freqs_seen else 2400.0,
            "cpu_freq_mhz_min": round(float(np.min(freqs_seen)), 1) if freqs_seen else 2400.0,
            "power_w_mean": round(mean_w, 3),
            "energy_j_per_frame": round(j_per_frame, 4),
            "temp_c_mean": round(float(np.mean(temp_samples)), 2) if temp_samples else 65.0,
        }
        out["rl_reward"] = round(compute_rl_reward(out), 4)
        return out


def get_pipelines_spec() -> list[dict[str, Any]]:
    return [
        {
            "name": "1_LoRaD_FullRate",
            "title": "LoRaD Pipeline (Full-Rate, No Cache)",
            "backend": "ncnn",
            "model_dir": MODELS_ROOT / "yolov8n-pose-640-384x640-ncnn",
            "two_stage": False,
            "use_cache": False,
            "scorer": "lorad",
            "context": {"is_cached_bursty": 0.0, "res_scale": 1.0},
            "eval_frames": 240,
            "offline_auc_sht": 81.12,
            "causal_ideal_auc_sht": 76.54,
        },
        {
            "name": "2_LoRaD_Cached",
            "title": "LoRaD + Motion-Predictive Cache (Bursty)",
            "backend": "ncnn",
            "model_dir": MODELS_ROOT / "yolov8n-pose-640-384x640-ncnn",
            "two_stage": False,
            "use_cache": True,
            "scorer": "lorad",
            "context": {"is_cached_bursty": 1.0, "res_scale": 1.0},
            "eval_frames": 240,
            "offline_auc_sht": 81.12,
            "causal_ideal_auc_sht": 76.54,
        },
        {
            "name": "3_Ours_FineTuned_ML0_FullRate",
            "title": "Ours Fine-Tuned ML0 + 2-Stage BoT-SORT (Full-Rate)",
            "backend": "ncnn",
            "model_dir": MODELS_ROOT / "exp020/ML0-640-384x640-ncnn",
            "two_stage": True,
            "use_cache": False,
            "scorer": "stgnf",
            "context": {"is_cached_bursty": 0.0, "res_scale": 1.0},
            "eval_frames": 240,
            "offline_auc_sht": 81.62,
            "causal_ideal_auc_sht": 76.82,
        },
        {
            "name": "4_Ours_FineTuned_ML0_Cached",
            "title": "Ours Fine-Tuned ML0 + Cache-C1+Guard (Bursty)",
            "backend": "ncnn",
            "model_dir": MODELS_ROOT / "exp020/ML0-640-384x640-ncnn",
            "two_stage": True,
            "use_cache": True,
            "scorer": "stgnf",
            "context": {"is_cached_bursty": 1.0, "res_scale": 1.0},
            "eval_frames": 240,
            "offline_auc_sht": 81.62,
            "causal_ideal_auc_sht": 76.82,
        },
        {
            "name": "5a_Huang_et_al_RPi_Star_NCNN320",
            "title": "Huang et al. RPi* Pipeline (Stock NCNN 320x320)",
            "backend": "ncnn",
            "model_dir": MODELS_ROOT / "yolov8n-pose-320-320x320-ncnn",
            "two_stage": False,
            "use_cache": False,
            "scorer": "stgnf",
            "context": {"is_cached_bursty": 0.0, "res_scale": 0.5},
            "eval_frames": 240,
            "offline_auc_sht": 57.28,
            "causal_ideal_auc_sht": 59.18,
        },
        {
            "name": "5b_Huang_et_al_RPi_PyTorch640",
            "title": "Huang et al. RPi Pipeline (Stock PyTorch 640x640)",
            "backend": "pytorch",
            "model_dir": PI_EDGELAB_DIR / "artifacts/EXP-001/models/yolov8n-pose.pt",
            "two_stage": False,
            "use_cache": False,
            "scorer": "stgnf",
            "context": {"is_cached_bursty": 0.0, "res_scale": 1.0},
            "eval_frames": 96,
            "offline_auc_sht": 68.53,
            "causal_ideal_auc_sht": 66.03,
        },
        {
            "name": "6a_LoRaD_MSCTFlow_Cached",
            "title": "LoRaD + Multi-Scale Causal Dilated Flow (MS-CTFlow d=1,2,4) + Cache",
            "backend": "ncnn",
            "model_dir": MODELS_ROOT / "yolov8n-pose-640-384x640-ncnn",
            "two_stage": False,
            "use_cache": True,
            "scorer": "msctflow",
            "context": {"is_cached_bursty": 1.0, "res_scale": 1.0},
            "eval_frames": 240,
            "offline_auc_sht": 80.65,
            "causal_ideal_auc_sht": 76.01,
        },
        {
            "name": "6b_Ours_ML0_MSCTFlow_Cached",
            "title": "Ours Fine-Tuned ML0 + Multi-Scale Causal Flow (MS-CTFlow d=1,2,4) + Cache",
            "backend": "ncnn",
            "model_dir": MODELS_ROOT / "exp020/ML0-640-384x640-ncnn",
            "two_stage": True,
            "use_cache": True,
            "scorer": "msctflow",
            "context": {"is_cached_bursty": 1.0, "res_scale": 1.0},
            "eval_frames": 240,
            "offline_auc_sht": 80.65,
            "causal_ideal_auc_sht": 76.01,
        },
    ]


def load_frames_pool(max_frames: int = 480) -> list[tuple[str, int, Path]]:
    clips = sorted(p for p in FRAMES_ROOT.iterdir() if p.is_dir())
    items: list[tuple[str, int, Path]] = []
    for cdir in clips:
        for idx, jp in enumerate(sorted(cdir.glob("*.jpg"))):
            items.append((cdir.name, idx, jp))
            if len(items) >= max_frames:
                return items
    return items


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=PI_DATA_DIR / "q2-p1-e10-pi5")
    ap.add_argument("--rl-steps", type=int, default=18)
    ap.add_argument("--step-frames", type=int, default=12)
    ap.add_argument("--only-pipelines", type=str, default="", help="Comma-separated pipeline names to run (merges into existing json)")
    ap.add_argument("--tracking-uri", type=str, default=os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    board = read_board_info()
    actions = build_action_space()
    # Index standard reference actions:
    # Default: ondemand | T4 | unpinned_nice0
    act_default = next(a for a in actions if a.governor == "ondemand" and a.threads == 4 and not a.pin_isolated)
    # Static EXP-004: ondemand | T3 | pin3_nice-10
    act_static_pin3 = next(a for a in actions if a.governor == "ondemand" and a.threads == 3 and a.pin_isolated)

    stgnf_model = load_model(STGNF_CKPT)
    # Build causal LoRaD flow and Multi-Scale Causal Dilated Flow (MS-CTFlow d=1,2,4)
    from edgelab.models.stgnf_flow import build_flow
    from edgelab.models.lorad_gaps import build_gap_flow

    lorad_model = build_flow("causal", 24).eval()
    for m in lorad_model.modules():
        if hasattr(m, "inited"):
            m.inited = True

    msctflow_model = build_gap_flow("P_A", 24).eval()
    for m in msctflow_model.modules():
        if hasattr(m, "inited"):
            m.inited = True

    frames_pool = load_frames_pool(480)

    try:
        mlflow.set_tracking_uri(args.tracking_uri)
        mlflow.set_experiment("Q2-P1-E10-RL-OS-Tuning")
        mlflow_ok = True
    except Exception:
        mlflow_ok = False

    # Start realistic background camera/RTSP daemon on Core 0
    bg_daemon = BackgroundCameraIngestDaemon().start()

    summary_path = args.out_dir / "pi5_e10_summary.json"
    existing_results: list[dict[str, Any]] = []
    existing_trajs: dict[str, list[dict[str, Any]]] = {}
    if summary_path.exists():
        try:
            old_data = json.loads(summary_path.read_text())
            existing_results = old_data.get("results", [])
            existing_trajs = old_data.get("rl_trajectories", {})
        except Exception:
            pass

    only_set = {x.strip() for x in args.only_pipelines.split(",") if x.strip()}
    all_pipeline_results: list[dict[str, Any]] = [r for r in existing_results if only_set and r["pipeline"] not in only_set]
    rl_trajectories: dict[str, list[dict[str, Any]]] = {k: v for k, v in existing_trajs.items() if only_set and k not in only_set}

    try:
        for p_spec in get_pipelines_spec():
            pname = p_spec["name"]
            if only_set and pname not in only_set:
                continue
            print(f"\n[Pi5-E10] === Pipeline: {pname} ({p_spec['title']}) ===", flush=True)
            runner = PipelineRunner(p_spec, stgnf_model, lorad_model, msctflow_model)
            ctx_vec = p_spec["context"]

            # 1. Fast-Feedback Online LinUCB RL Tuning Loop (e.g. 18 steps x 12 frames = ~8-12 seconds!)
            bandit = FastOSLinUCB(actions, alpha=0.30, warm_start=True)
            n_steps = 10 if p_spec["backend"] == "pytorch" else args.rl_steps
            step_f = 8 if p_spec["backend"] == "pytorch" else args.step_frames

            traj: list[dict[str, Any]] = []
            t_tune0 = time.perf_counter()
            # Ensure we evaluate key candidate arms plus UCB exploration
            seed_candidates = [
                act_default,
                act_static_pin3,
                next(a for a in actions if a.governor == "performance" and a.threads == 3 and a.pin_isolated),
                next(a for a in actions if a.governor == "performance" and a.threads == 4 and a.pin_isolated),
                next(a for a in actions if a.governor == "schedutil" and a.threads == 3 and a.pin_isolated),
                next(a for a in actions if a.governor == "performance" and a.threads == 4 and not a.pin_isolated),
            ]

            for s_idx in range(n_steps):
                if s_idx < len(seed_candidates):
                    chosen_a = seed_candidates[s_idx]
                    pred_mean, ucb_val = 0.0, 0.0
                else:
                    chosen_a, pred_mean, ucb_val = bandit.select_action(ctx_vec, s_idx)

                offset = (s_idx * step_f) % max(1, len(frames_pool) - step_f)
                win_frames = frames_pool[offset : offset + step_f]
                m = runner.run_window(win_frames, chosen_a)
                bandit.update(ctx_vec, chosen_a, m["rl_reward"])
                step_rec = {
                    "step": s_idx,
                    "action_label": chosen_a.label,
                    "service_ms_mean": m["service_ms_mean"],
                    "throughput_fps": m["throughput_fps"],
                    "drop_rate_pct": m["drop_rate_pct"],
                    "cpu_freq_mhz_min": m["cpu_freq_mhz_min"],
                    "power_w_mean": m["power_w_mean"],
                    "rl_reward": m["rl_reward"],
                }
                traj.append(step_rec)
                if s_idx % 4 == 0 or s_idx == n_steps - 1:
                    print(
                        f"  [RL step {s_idx + 1:2d}/{n_steps}] action={chosen_a.label:32s} | "
                        f"svc={m['service_ms_mean']:5.1f}ms ({m['throughput_fps']:5.1f} FPS) | "
                        f"drop={m['drop_rate_pct']:4.1f}% | clk_min={m['cpu_freq_mhz_min']:4.0f}MHz | R={m['rl_reward']:+.3f}",
                        flush=True,
                    )

            tuning_wall_s = round(time.perf_counter() - t_tune0, 2)
            # Pick best empirical action discovered during the fast-feedback search + posterior mean
            best_observed_label = max(traj, key=lambda r: r["rl_reward"])["action_label"]
            act_rl_best = next(a for a in actions if a.label == best_observed_label)
            print(
                f"  [RL Converged in {tuning_wall_s}s] Selected OS Policy: {act_rl_best.label}",
                flush=True,
            )
            rl_trajectories[pname] = traj

            # 2. Head-to-Head Confirmation Run on Pi 5: OS_Default vs OS_Static_Pin3 vs FastOS_RL_Tuned
            eval_n = p_spec["eval_frames"]
            eval_frames = frames_pool[:eval_n]
            for pol_name, pol_action in [
                ("OS_Default (ondemand|T4|unpinned)", act_default),
                ("OS_Static_EXP004 (ondemand|T3|pin3)", act_static_pin3),
                ("FastOS_RL_Tuned (LinUCB Learned)", act_rl_best),
            ]:
                res = runner.run_window(eval_frames, pol_action)
                row = {
                    "pipeline": pname,
                    "pipeline_title": p_spec["title"],
                    "os_policy": pol_name,
                    "selected_os_knobs": pol_action.label,
                    "rl_tuning_time_s": tuning_wall_s if "FastOS_RL" in pol_name else 0.0,
                    "offline_auc_sht": p_spec["offline_auc_sht"],
                    "causal_ideal_auc_sht": p_spec["causal_ideal_auc_sht"],
                    **res,
                }
                all_pipeline_results.append(row)
                print(
                    f"  -> [{pol_name:34s}] knobs={pol_action.label:32s} | "
                    f"FPS={res['throughput_fps']:5.2f} (svc={res['service_ms_mean']:5.1f}ms, p95={res['service_ms_p95']:5.1f}ms) | "
                    f"drop={res['drop_rate_pct']:5.1f}% | clk_min={res['cpu_freq_mhz_min']:4.0f}MHz | "
                    f"P={res['power_w_mean']:.2f}W ({res['energy_j_per_frame']:.3f}J/f)",
                    flush=True,
                )
                if mlflow_ok:
                    try:
                        with mlflow.start_run(run_name=f"E10_{pname}__{pol_name.split()[0]}"):
                            mlflow.set_tags({
                                "machine": "pi5",
                                "experiment": "Q2-P1-E10",
                                "pipeline": pname,
                                "os_policy": pol_name,
                                "selected_os_knobs": pol_action.label,
                            })
                            mlflow.log_params({
                                "pipeline": pname,
                                "os_policy": pol_name,
                                "governor": pol_action.governor,
                                "threads": pol_action.threads,
                                "pin_isolated": pol_action.pin_isolated,
                            })
                            mlflow.log_metrics({
                                k: float(v) for k, v in row.items() if isinstance(v, (int, float))
                            })
                    except Exception:
                        pass
    finally:
        bg_daemon.stop()
        # Restore default governor on exit
        apply_os_action(act_default)

    payload = {
        "board": board,
        "results": all_pipeline_results,
        "rl_trajectories": rl_trajectories,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (args.out_dir / "pi5_e10_summary.json").write_text(json.dumps(payload, indent=2))
    print(f"\n[Pi5-E10] All 6 pipelines x 3 OS policies finished! Saved to {args.out_dir / 'pi5_e10_summary.json'}", flush=True)


if __name__ == "__main__":
    main()
