"""STEP 1: Models — Pose Estimators (NCNN / PyTorch) & Anomaly Scorers (STG-NF / LoRaD).

Read this file first (~180 lines). It defines the two neural network stages of the pipeline:
  1. Pose Estimator (`PoseDetector`):
     - Runs YOLOv8n-pose on 1 frame -> returns person boxes (N, 4), confidences (N,), and 17 keypoints (N, 17, 3).
     - Supports Huang et al.'s `RPi` (PyTorch 640), `RPi*` (NCNN 320), Stock NCNN 640, and our `ML0` NCNN 640.
  2. Skeleton Anomaly Scorer (`build_stgnf_or_lorad` + `normalize_window_n0` + `ema_normalize_track_n1`):
     - Takes a 24-frame skeleton trajectory of 1 person -> outputs a normality score (-NLL; higher = more normal).
     - Shows the exact difference between Original `STG-NF` (symmetric temporal conv) and `LoRaD` (causal left-padded conv),
       and between per-window normalization (`N0`) and streaming EMA prefix normalization (`N1`).
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from scipy.signal import lfilter

# Point to the pinned third_party/STG-NF graph/actnorm primitives (read-only)
REPO_ROOT = Path(__file__).resolve().parents[1]
STGNF_ROOT = REPO_ROOT / "experiments" / "third_party" / "STG-NF"
if str(STGNF_ROOT) not in sys.path:
    sys.path.insert(0, str(STGNF_ROOT))

from models.STG_NF.graph import Graph  # noqa: E402
from models.STG_NF.modules_pose import ActNorm2d, Permute2d, gaussian_likelihood  # noqa: E402
from models.STG_NF.stgcn import ConvTemporalGraphical, st_gcn  # noqa: E402

# OpenPose-18 joint order expected by STG-NF (COCO-17 + neck = midpoint of shoulders 5 & 6)
COCO18_ORDER = [0, 17, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3]
VID_RES = np.array([856.0, 480.0], dtype=np.float64)


# =====================================================================
# PART A: Pose Normalization (Why N0 vs N1 Matters for Caching)
# =====================================================================
def coco17_to_openpose18(xy17: np.ndarray) -> np.ndarray:
    """Convert 17 COCO keypoints (..., 17, 2) to 18 OpenPose keypoints (..., 18, 2) by adding the neck joint."""
    neck = 0.5 * (xy17[..., 5, :] + xy17[..., 6, :])
    return np.concatenate([xy17, neck[..., None, :]], axis=-2)[..., COCO18_ORDER, :]


def normalize_window_n0(windows_xy17: np.ndarray) -> torch.Tensor:
    """N0 (Per-Window Normalization used by STG-NF & LoRaD-Window):
    Subtracts the 24-frame window's mean (x, y) and divides by the window's vertical std_y.
    Input: (B, T=24, 17, 2) pixel coords -> Output: (B, C=2, T=24, V=18) tensor for the normalizing flow.
    """
    p = coco17_to_openpose18(np.asarray(windows_xy17, dtype=np.float64)) / VID_RES  # (B, T, 18, 2)
    mean_xy = p.mean(axis=(1, 2), keepdims=True)
    std_y = p[..., 1].std(axis=(1, 2), keepdims=True)[..., None] + 1e-6
    normed = (p - mean_xy) / std_y
    return torch.from_numpy(np.ascontiguousarray(normed.transpose(0, 3, 1, 2), dtype=np.float32))


def ema_normalize_track_n1(track_xy17: np.ndarray, seg_len: int = 24) -> np.ndarray:
    """N1 (Streaming Prefix EMA Normalization tested in LoRaD A3/A3s):
    Normalizes each frame ONCE as it arrives using only past frames <= t via an Exponential Moving Average.
    Allows O(1) circular-buffer streaming, but loses ~4-5 AUC points vs N0 because scale/centroid drift!
    """
    alpha = 2.0 / (seg_len + 1.0)
    p = coco17_to_openpose18(np.asarray(track_xy17, dtype=np.float64)) / VID_RES  # (N, 18, 2)
    centroid = p.mean(axis=1)  # (N, 2)
    y = p[..., 1]
    moments = np.stack([y.mean(axis=1), (y ** 2).mean(axis=1)], axis=1)
    zi_c = ((1.0 - alpha) * centroid[:1]).astype(np.float64)
    zi_m = ((1.0 - alpha) * moments[:1]).astype(np.float64)
    c_ema, _ = lfilter([alpha], [1.0, -(1.0 - alpha)], centroid, axis=0, zi=zi_c)
    m_ema, _ = lfilter([alpha], [1.0, -(1.0 - alpha)], moments, axis=0, zi=zi_m)
    sigma = np.sqrt(np.maximum(m_ema[:, 1] - m_ema[:, 0] ** 2, 1e-6))
    return ((p - c_ema[:, None, :]) / sigma[:, None, None]).astype(np.float32)


# =====================================================================
# PART B: Anomaly Scorers — Original STG-NF vs. LoRaD Causal Flow
# =====================================================================
class CausalSTGCN(nn.Module):
    """LoRaD's Causal Spatio-Temporal Graph Block:
    Standard STG-NF uses symmetric temporal padding (k-1)//2, which reads future frames inside the window.
    LoRaD pads ONLY on the left by (k - 1) frames (`F.pad(h, (0, 0, k - 1, 0))`), so frame t only sees <= t.
    """

    def __init__(self, in_ch: int, out_ch: int, kernel_t: int, kernel_s: int, residual: bool = True):
        super().__init__()
        self.k = kernel_t
        self.gcn = ConvTemporalGraphical(in_ch, out_ch, kernel_s)
        self.bn1, self.relu = nn.BatchNorm2d(out_ch), nn.ReLU(inplace=False)
        self.tconv = nn.Conv2d(out_ch, out_ch, kernel_size=(kernel_t, 1), padding=0)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.res = (lambda x: 0) if not residual else (
            (lambda x: x) if in_ch == out_ch else nn.Sequential(nn.Conv2d(in_ch, out_ch, 1), nn.BatchNorm2d(out_ch))
        )

    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        h, _ = self.gcn(x, A)
        h = self.relu(self.bn1(h))
        h = self.tconv(F.pad(h, (0, 0, self.k - 1, 0)))  # Left-only causal padding!
        return F.relu(self.bn2(h) + self.res(x))


class FlowCouplingStep(nn.Module):
    """One invertible Normalizing Flow step: ActNorm -> Channel Permute -> Affine Coupling (z2' = (z2 + shift) * scale)."""

    def __init__(self, n_frames: int, first: bool, causal: bool):
        super().__init__()
        self.actnorm = ActNorm2d(2, 1.0)
        self.permute = Permute2d(2, shuffle=False)
        A = torch.from_numpy(Graph(strategy="uniform", max_hop=8).A).float()
        self.register_buffer("A", A)
        kt = n_frames // 2 + 1
        self.causal = causal
        self.gcn = CausalSTGCN(1, 2, kt, A.size(0), residual=not first) if causal else st_gcn(1, 2, (kt, A.size(0)), 1, residual=not first)

    def forward(self, x: torch.Tensor, logdet: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z, logdet = self.actnorm(x, logdet=logdet, reverse=False)
        z = self.permute(z, reverse=False)
        z1, z2 = z[:, 0:1], z[:, 1:2]
        h = self.gcn(z1, self.A) if self.causal else self.gcn(z1, self.A)[0]
        shift, scale = h[:, 0:1], torch.sigmoid(h[:, 1:2] + 2.0) + 1e-6
        z2 = (z2 + shift) * scale
        return torch.cat([z1, z2], dim=1), logdet + torch.log(scale).sum(dim=[1, 2, 3])


class SkeletonNormalizingFlow(nn.Module):
    """8-step Spatio-Temporal Graph Normalizing Flow (~616 parameters).
    `causal=False` -> Original STG-NF; `causal=True` -> LoRaD Causal Pose Flow.
    """

    def __init__(self, n_frames: int = 24, steps: int = 8, causal: bool = False):
        super().__init__()
        self.steps = nn.ModuleList([FlowCouplingStep(n_frames, first=(k == 0), causal=causal) for k in range(steps)])
        self.prior_mean = 3.0

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logdet = torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)
        z = x
        for step in self.steps:
            z, logdet = step(z, logdet)
        log_p = gaussian_likelihood(torch.full_like(z, self.prior_mean), torch.zeros_like(z), z) + logdet
        nll = -log_p / (math.log(2.0) * z[0].numel())
        return z, nll


def load_scorer_checkpoint(path: Path, causal: bool = False, seg_len: int = 24) -> nn.Module:
    """Load either the official STG-NF checkpoint (`ShanghaiTech_85_9.tar`) or our trained LoRaD checkpoint (`final.pt`)."""
    from edgelab.pipeline.stgnf_live import load_model as load_official_stgnf
    from edgelab.models.lorad import load_flow as load_lorad_flow

    if path.name.endswith(".tar"):
        return load_official_stgnf(path)
    flow, _, _ = load_lorad_flow(path)
    return flow


# =====================================================================
# PART C: Pose Detector Wrapper (NCNN FP16 & PyTorch FP32)
# =====================================================================
class PoseDetector:
    """Unified wrapper for YOLOv8n-pose across:
      - Huang `RPi` (PyTorch FP32 @ 640x640)
      - Huang `RPi*` (NCNN FP16 @ 320x320)
      - Stock NCNN FP16 @ 384x640
      - Ours Fine-Tuned `ML0` NCNN FP16 @ 384x640 (trained with box-only small people kept)
    """

    def __init__(self, backend: str, model_path: str | Path, threads: int = 4, imgsz: int = 640):
        self.backend = backend
        self.threads = threads
        self.imgsz = imgsz
        if backend == "ncnn":
            from edgelab.pipeline.ncnn_pose import NcnnPose

            self.model = NcnnPose(model_path, threads=threads, fp16=True)
        elif backend == "pytorch":
            from ultralytics import YOLO

            self.model = YOLO(str(model_path))
            torch.set_num_threads(threads)
        else:
            raise ValueError(f"Unsupported backend: {backend}")

    def __call__(self, img_bgr: np.ndarray, conf_thresh: float = 0.05) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """Run pose inference on 1 BGR frame -> (boxes_xyxy, scores, keypoints_17x3, infer_ms)."""
        t0 = time.perf_counter()
        if self.backend == "ncnn":
            boxes, scores, kps, _ = self.model(img_bgr, conf=conf_thresh)
        else:
            res = self.model.predict(img_bgr, imgsz=self.imgsz, conf=conf_thresh, verbose=False, device="cpu")[0]
            torch.set_num_threads(self.threads)
            if res.boxes is not None and len(res.boxes):
                boxes = res.boxes.xyxy.cpu().numpy().astype(np.float32)
                scores = res.boxes.conf.cpu().numpy().astype(np.float32)
                kps = res.keypoints.data.cpu().numpy().astype(np.float32)
            else:
                boxes = np.zeros((0, 4), dtype=np.float32)
                scores = np.zeros((0,), dtype=np.float32)
                kps = np.zeros((0, 17, 3), dtype=np.float32)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        return boxes, scores, kps, dt_ms
