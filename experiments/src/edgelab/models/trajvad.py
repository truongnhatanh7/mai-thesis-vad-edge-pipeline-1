"""TrajVAD-T / TrajVAD-P (Song & Lee, "Bounding-Box Trajectories Matter for Video Anomaly Detection", ECCV 2026) for the edge pose pipeline (Q2-P1-E07).

Ported from the authors' repository (https://github.com/songinpyo/TrajVAD-ECCV2026, models/trajvad_{layers,t,p,pose_utils}.py, utils/bbox_features.py,
data/dataset_loader.py) with the changes the edge pipeline needs. Model classes are the authors' (device handling removed, nothing else changed):

    TrajVAD-T  27 kinematic features of one box trajectory (+ class embedding) -> normalising flow of K = 6 causal-conv couplings, hidden 96
    TrajVAD-P  the same trajectory flow (K = 18, hidden 64) + a conditional pose flow (K_pose = 18) conditioned on z_traj, 17 x 3 pose input
               (pelvis-centred, one body scale per window), joint NLL gated by the keypoint confidence of the window

What differs from the authors' data pipeline (all documented in Q2-P1-E07/plan.md):
    * the box of a pose track is the min / max of its keypoints (train: released AlphaPose poses carry no boxes; used for test tracks too, so train / test match);
    * every feature is CAUSAL: the Kalman filter is a 15-line constant-velocity filter in NumPy (no `filterpy`), first differences start at 0 (the
      authors forward-fill frame 0 from frame 1: a one-frame look-ahead), no gap interpolation (interpolation reads the right end of a gap);
      a pose track therefore yields the same feature for pose t whatever happens after t;
    * only the person class exists (class id 0): the class embedding is constant, so what the flow sees of "multi-class" boxes is not available:
      ShanghaiTech anomalies by bicycles / vehicles are invisible to a pose-only frontend (limitation, reported);
    * `conf` = mean keypoint confidence of the pose.
Windows: ShanghaiTech 16 poses, UBnormal 12 (the authors' paper configuration); train stride 4, scoring stride 1; landing / protocol are E00's.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from edgelab.pose.segments import read_tracks, window_starts

SEG_LEN = {"ShanghaiTech": 16, "UBnormal": 12}
VID_RES = (856.0, 480.0)
FEATURE_NAMES = ["cx", "cy", "w", "h", "vx", "vy", "speed", "ax", "ay", "acc", "area", "ratio", "h_norm_speed", "w_norm_speed",
                 "area_norm_speed", "h_norm_acc", "w_norm_acc", "area_norm_acc", "sin_theta", "cos_theta", "jerk", "curvature",
                 "expansion_rate", "ar_velocity", "kinetic_energy", "path_efficiency", "conf"]
N_FEATURES = len(FEATURE_NAMES)  # 27
TRAIN_STRIDE = 4
CONFIGS = {  # the authors' paper configurations (configs/paper/TrajVAD-{T,P}/*.yaml); weight decay T 0, P 1e-5
    "T": dict(hidden_channels=96, K=6, weight_decay=0.0),
    "P": dict(hidden_channels=64, K=18, K_pose=18, pose_dim=51, lambda_pose=1.0, weight_decay=1e-5),
}


# =====================================================================================================================
# Causal features
# =====================================================================================================================
def kalman_cv(boxes: np.ndarray, frames: np.ndarray, q: float = 1.0, r: float = 10.0, p0: float = 1000.0) -> np.ndarray:
    """Causal constant-velocity Kalman filter over (cx, cy, w, h): the authors' 8-state filter (filterpy defaults Q = qI, R = rI,
    P0 = 1000 I) in NumPy. A frame gap g makes g prediction steps before the update. Row 0 is returned unfiltered (as the authors do)."""
    n = len(boxes)
    out = np.empty((n, 4))
    if n == 0:
        return out
    F = np.eye(8)
    F[:4, 4:] = np.eye(4)
    H = np.zeros((4, 8))
    H[:, :4] = np.eye(4)
    Q, R, P = q * np.eye(8), r * np.eye(4), p0 * np.eye(8)
    x = np.concatenate([boxes[0], np.zeros(4)])
    out[0] = boxes[0]
    gaps = np.diff(np.asarray(frames)).clip(1, 30)
    for t in range(1, n):
        for _ in range(int(gaps[t - 1])):
            x = F @ x
            P = F @ P @ F.T + Q
        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)
        x = x + K @ (boxes[t] - H @ x)
        P = (np.eye(8) - K @ H) @ P
        out[t] = x[:4]
    return out


def diff0(x: np.ndarray, frames: np.ndarray | None = None) -> np.ndarray:
    """First difference with a zero first row (causal; the authors copy row 1 into row 0). Per frame when `frames` is given."""
    out = np.zeros_like(x, dtype=np.float64)
    if len(x) > 1:
        d = x[1:] - x[:-1]
        if frames is not None:
            dt = np.diff(frames).clip(1, 30).astype(np.float64)
            d = d / dt.reshape((-1,) + (1,) * (x.ndim - 1))
        out[1:] = d
    return out


def pose_boxes(keypoints: np.ndarray) -> np.ndarray:
    """(n, 17, 3) -> (n, 4) cx, cy, w, h of the keypoint min / max box in pixels."""
    x, y = keypoints[:, :, 0], keypoints[:, :, 1]
    x1, x2, y1, y2 = x.min(1), x.max(1), y.min(1), y.max(1)
    return np.stack([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], axis=1)


def path_efficiency(pos: np.ndarray, window: int = 12) -> np.ndarray:
    out = np.ones(len(pos))
    step = np.linalg.norm(np.diff(pos, axis=0), axis=1) if len(pos) > 1 else np.zeros(0)
    csum = np.concatenate([[0.0], np.cumsum(step)])
    for t in range(1, len(pos)):
        s = max(0, t - window + 1)
        total = csum[t] - csum[s]
        out[t] = np.linalg.norm(pos[t] - pos[s]) / total if total > 1e-6 and t - s >= 1 else 1.0
    return out


def trajectory_features(frames: np.ndarray, keypoints: np.ndarray, kalman: bool = True, vid_res: tuple = VID_RES) -> np.ndarray:
    """One pose track -> (n, 27) causal features (the authors' `extract_all_features` on the Kalman-smoothed keypoint box)."""
    boxes = pose_boxes(keypoints)
    if kalman:
        boxes = kalman_cv(boxes, frames)
    vw, vh = vid_res
    cx, cy, w, h = boxes[:, 0] / vw, boxes[:, 1] / vh, boxes[:, 2] / vw, boxes[:, 3] / vh
    pos = np.stack([cx, cy], axis=1)
    vel = diff0(pos, frames)
    acc_v = diff0(vel, frames)
    if len(acc_v) > 1:
        acc_v[1] = 0.0  # needs two velocities
    speed, acc = np.linalg.norm(vel, axis=1), np.linalg.norm(acc_v, axis=1)
    orient = np.where(speed[:, None] > 1e-6, vel / np.maximum(speed[:, None], 1e-12), 0.0)
    area, ratio = w * h, w / (h + 1e-6)
    jerk_v = diff0(acc_v, frames)
    jerk_v[:3] = 0.0
    jerk = np.linalg.norm(jerk_v, axis=1)
    sq = vel[:, 0] ** 2 + vel[:, 1] ** 2
    curv = np.where(sq > 1e-6, np.abs(vel[:, 0] * acc_v[:, 1] - vel[:, 1] * acc_v[:, 0]) / np.maximum(sq, 1e-12) ** 1.5, 0.0)
    expansion = diff0(np.log(np.maximum(area, 1e-6)), frames)
    ar_vel = diff0(ratio, frames)
    conf = keypoints[:, :, 2].mean(axis=1)
    feats = np.stack([cx, cy, w, h, vel[:, 0], vel[:, 1], speed, acc_v[:, 0], acc_v[:, 1], acc, area, ratio, speed / (h + 1e-6),
                      speed / (w + 1e-6), speed / (area + 1e-6), acc / (h + 1e-6), acc / (w + 1e-6), acc / (area + 1e-6), orient[:, 0],
                      orient[:, 1], jerk, curv, expansion, ar_vel, area * speed ** 2, path_efficiency(pos), conf], axis=1)
    return feats.astype(np.float32)


def robust_normalize_pose(pose: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(B, T, 17, 3) pixels -> ((B, T, 51), (B,) mask): pelvis-centred, one body-scale per window (authors' `robust_normalize_pose`)."""
    pose = pose.float().clone()
    valid = (pose[..., 2] > 0).any(dim=-1)
    mask = valid.any(dim=1)
    pelvis = pose[:, :, [11, 12], :2].mean(dim=2, keepdim=True)
    coords = pose[..., :2] - pelvis
    scale = coords[..., 1].std(dim=(1, 2), keepdim=True).unsqueeze(-1).clamp_min(1e-6)
    pose[..., :2] = coords / scale
    pose[~valid] = 0
    return pose.reshape(pose.shape[0], pose.shape[1], 51), mask


# =====================================================================================================================
# Models (authors' code)
# =====================================================================================================================
class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=self.padding, dilation=dilation)

    def forward(self, x):
        out = self.conv(x)
        return out[:, :, :-self.padding] if self.padding > 0 else out


class PoseCoupling(nn.Module):
    """z_target = (x_target + shift(cond)) * scale(cond), shift / scale from two causal convs over the condition (TrajVADPoseCoupling)."""

    def __init__(self, target_dim, cond_dim, hidden_dim, dilation_pattern=(1, 2)):
        super().__init__()
        d1, d2 = dilation_pattern
        self.net = nn.Sequential(CausalConv1d(cond_dim, hidden_dim, 3, d1), nn.ReLU(), CausalConv1d(hidden_dim, hidden_dim, 3, d2), nn.ReLU(),
                                 nn.Conv1d(hidden_dim, target_dim * 2, kernel_size=1))

    def forward(self, target, condition):
        shift, scale = self.net(condition).chunk(2, dim=1)
        scale = torch.sigmoid(scale + 2.0) + 1e-6
        return (target + shift) * scale, torch.sum(torch.log(scale), dim=[1, 2])


class ChannelCoupling(nn.Module):
    """TrajVADChannelCoupling: split halves, causal-conv subnet of one half gives shift / scale of the other."""

    def __init__(self, left_dim, right_dim, hidden_dim, direction="left_to_right", dilation_pattern=(1, 2)):
        super().__init__()
        self.direction = direction
        in_c, out_c = (left_dim, right_dim) if direction == "left_to_right" else (right_dim, left_dim)
        d1, d2 = dilation_pattern
        self.net = nn.Sequential(CausalConv1d(in_c, hidden_dim, 3, d1), nn.ReLU(), CausalConv1d(hidden_dim, hidden_dim, 3, d2), nn.ReLU(),
                                 nn.Conv1d(hidden_dim, out_c * 2, kernel_size=1))

    def forward(self, z_left, z_right):
        src, tgt = (z_left, z_right) if self.direction == "left_to_right" else (z_right, z_left)
        shift, scale = self.net(src).chunk(2, dim=1)
        scale = torch.sigmoid(scale + 2.0) + 1e-6
        new = (tgt + shift) * scale
        logdet = torch.sum(torch.log(scale), dim=[1, 2])
        return (z_left, new, logdet) if self.direction == "left_to_right" else (new, z_right, logdet)


def _dilation(layer_index: int, max_dilation: int = 12) -> tuple[int, int]:
    d = min(2 ** layer_index, max_dilation)
    return d, d


class TrajVADT(nn.Module):
    def __init__(self, cont_dim=27, emb_dim=3, num_classes=80, hidden_channels=96, K=6, prior_mean=3.0, seg_len=12, max_dilation=12, **_):
        super().__init__()
        self.cont_dim, self.emb_dim, self.prior_mean, self.seg_len = cont_dim, emb_dim, prior_mean, seg_len
        self.input_dim = cont_dim + emb_dim
        self.split_a, self.split_b = list(range(self.input_dim // 2)), list(range(self.input_dim // 2, self.input_dim))
        self.class_emb = nn.Embedding(num_classes, emb_dim)
        self.layers = nn.ModuleList(ChannelCoupling(len(self.split_a), len(self.split_b), hidden_channels,
                                                    "left_to_right" if i % 2 == 0 else "right_to_left", _dilation(i, max_dilation))
                                    for i in range(K))
        self.actnorm_scale = nn.Parameter(torch.ones(1, self.input_dim, 1))
        self.actnorm_bias = nn.Parameter(torch.zeros(1, self.input_dim, 1))
        self.actnorm_initialized = False
        self.register_buffer("prior_mean_tensor", torch.full((1,), prior_mean))

    def forward(self, x_cont, x_cat=None):
        b, t, _ = x_cont.shape
        if x_cat is not None:
            x_cat = torch.mode(x_cat.cpu(), dim=1, keepdim=True).values.to(x_cat.device).expand_as(x_cat)  # mps lacks aten::mode
            x = torch.cat([x_cont, self.class_emb(x_cat.squeeze(-1))], dim=2)
        else:
            x = x_cont
        x = x.permute(0, 2, 1)
        if not self.actnorm_initialized:
            with torch.no_grad():
                self.actnorm_bias.data.copy_(-x.mean(dim=[0, 2], keepdim=True))
                self.actnorm_scale.data.copy_(1.0 / (x.std(dim=[0, 2], keepdim=True) + 1e-6))
            self.actnorm_initialized = True
        logdet = torch.sum(torch.log(torch.abs(self.actnorm_scale))) * t
        latent = (x + self.actnorm_bias) * self.actnorm_scale
        total = logdet.repeat(b)
        a, c = latent[:, self.split_a], latent[:, self.split_b]
        for layer in self.layers:
            a, c, ld = layer(a, c)
            total = total + ld
        z = torch.cat([a, c], dim=1)
        logp = (-0.5 * ((z - self.prior_mean_tensor) ** 2) - 0.5 * math.log(2 * math.pi)).sum(dim=[1, 2])
        return z, -(logp + total) / (t * self.input_dim)


class TrajVADP(nn.Module):
    def __init__(self, cont_dim=27, emb_dim=3, num_classes=80, hidden_channels=64, K=18, K_pose=18, prior_mean=3.0, seg_len=12, pose_dim=51,
                 lambda_pose=1.0, max_dilation=12, **_):
        super().__init__()
        self.pose_dim, self.lambda_pose = pose_dim, float(lambda_pose)
        self.base_model = TrajVADT(cont_dim, emb_dim, num_classes, hidden_channels, K, prior_mean, seg_len, max_dilation)
        self.base_latent_dim = self.base_model.input_dim
        self.pose_actnorm_scale = nn.Parameter(torch.ones(1, pose_dim, 1))
        self.pose_actnorm_bias = nn.Parameter(torch.zeros(1, pose_dim, 1))
        self.pose_actnorm_initialized = False
        self.pose_layers = nn.ModuleList(PoseCoupling(pose_dim, self.base_latent_dim, hidden_channels, _dilation(i, max_dilation))
                                         for i in range(K_pose))
        self.register_buffer("prior_mean_tensor", torch.full((1,), prior_mean))

    def forward(self, x_cont, x_cat=None, x_pose=None, pose_mask=None):
        b, t, _ = x_cont.shape
        z_base, nll_base = self.base_model(x_cont, x_cat)
        g_person = torch.ones(b)
        if x_cat is not None:
            g_person = (torch.mode(x_cat.squeeze(-1).cpu(), dim=1).values.to(x_cat.device) == 0).float()
        g_pose = torch.ones(b) if pose_mask is None else pose_mask.float().view(b)
        g_quality = x_pose.reshape(b, t, 17, 3)[..., 2].mean(dim=(1, 2)).clamp(0, 1)
        g = g_person * g_pose * g_quality
        xp = x_pose.permute(0, 2, 1)
        if not self.pose_actnorm_initialized:
            with torch.no_grad():
                self.pose_actnorm_bias.data.copy_(-xp.mean(dim=[0, 2], keepdim=True))
                self.pose_actnorm_scale.data.copy_(1.0 / (xp.std(dim=[0, 2], keepdim=True) + 1e-6))
            self.pose_actnorm_initialized = True
        total = (torch.sum(torch.log(torch.abs(self.pose_actnorm_scale))) * t).repeat(b)
        zp = (xp + self.pose_actnorm_bias) * self.pose_actnorm_scale
        for layer in self.pose_layers:
            zp, ld = layer(zp, z_base)
            total = total + ld
        logp_pose = (-0.5 * ((zp - self.prior_mean_tensor) ** 2) - 0.5 * math.log(2 * math.pi)).sum(dim=[1, 2])
        pose_term = (logp_pose + total) * g
        raw_total = -nll_base * (t * self.base_latent_dim) + self.lambda_pose * pose_term
        d_eff = self.base_latent_dim + self.pose_dim * g
        return torch.cat([z_base, zp], dim=1), -raw_total / (t * d_eff)


def build_model(kind: str, seg_len: int) -> nn.Module:
    cfg = {k: v for k, v in CONFIGS[kind].items() if k != "weight_decay"}
    return TrajVADT(seg_len=seg_len, **cfg) if kind == "T" else TrajVADP(seg_len=seg_len, **cfg)


def mark_initialised(model: nn.Module) -> None:
    for m in model.modules():
        if hasattr(m, "actnorm_initialized"):
            m.actnorm_initialized = True
        if hasattr(m, "pose_actnorm_initialized"):
            m.pose_actnorm_initialized = True


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


# =====================================================================================================================
# Training windows, scaler, training
# =====================================================================================================================
@dataclass
class Scaler:
    mean: np.ndarray
    std: np.ndarray

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mean) / self.std).astype(np.float32)


@dataclass
class TrajTrainSet:
    feats: np.ndarray  # (P, 27) raw (unscaled) causal features of every training pose
    kp: np.ndarray  # (P, 17, 3) keypoints
    starts: np.ndarray  # (N,) first pose index of each window
    seg_len: int

    def __len__(self) -> int:
        return len(self.starts)


def build_trainset(pose_dir: Path, seg_len: int, stride: int = TRAIN_STRIDE, cache: Path | None = None) -> TrajTrainSet:
    if cache is not None and Path(cache).exists():
        z = np.load(cache)
        return TrajTrainSet(z["feats"], z["kp"], z["starts"], int(z["seg_len"]))
    feats, kps, starts, base = [], [], [], 0
    for path in sorted(Path(pose_dir).glob("*_tracked_person.json")):
        for frames, kp in read_tracks(path):
            s = window_starts(frames, seg_len, stride)
            kp = kp.astype(np.float32)
            feats.append(trajectory_features(frames, kp))
            kps.append(kp)
            if len(s):
                starts.append(s.astype(np.int64) + base)
            base += len(frames)
    ts = TrajTrainSet(np.concatenate(feats), np.concatenate(kps), np.concatenate(starts), seg_len)
    if cache is not None:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, feats=ts.feats, kp=ts.kp, starts=ts.starts, seg_len=ts.seg_len)
    return ts


def fit_scaler(ts: TrajTrainSet) -> Scaler:
    idx = np.unique((ts.starts[:, None] + np.arange(ts.seg_len)[None, :]).ravel())
    f = ts.feats[idx].astype(np.float64)
    std = f.std(axis=0)
    return Scaler(f.mean(axis=0), np.where(std < 1e-8, 1.0, std))


def window_batch(feats_scaled: np.ndarray, kp: np.ndarray, starts: np.ndarray, seg_len: int, kind: str):
    idx = starts[:, None] + np.arange(seg_len)[None, :]
    x_cont = torch.from_numpy(feats_scaled[idx])
    x_cat = torch.zeros(len(starts), seg_len, 1, dtype=torch.long)
    if kind == "T":
        return x_cont, x_cat, None, None
    pose, mask = robust_normalize_pose(torch.from_numpy(kp[idx]))
    return x_cont, x_cat, pose, mask


def train_trajvad(kind: str, dataset: str, seed: int, out_dir: Path, ts: TrajTrainSet, epochs: int = 10, batch_size: int = 256, lr: float = 5e-4,
                  threads: int = 2, max_batches: int | None = None, log=print, device: str = "cpu") -> Path:
    """Authors' recipe: RMSprop lr 5e-4 momentum 0.9 (weight decay T 0 / P 1e-5), batch 256, grad-clip 3, 10 epochs, mean NLL."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / "DONE").exists():
        return out_dir / "final.pt"
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    seg_len = ts.seg_len
    scaler = fit_scaler(ts)
    feats = scaler(ts.feats)
    dev = torch.device(device)
    model = build_model(kind, seg_len).to(dev)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, weight_decay=CONFIGS[kind]["weight_decay"], momentum=0.9)
    n, t0, losses_by_epoch = len(ts), time.perf_counter(), []
    for epoch in range(epochs):
        rng = np.random.default_rng([seed, epoch])
        order = rng.permutation(n)
        model.train()
        losses, seen = [], 0
        for b, begin in enumerate(range(0, n, batch_size)):
            if max_batches is not None and b >= max_batches:
                break
            pick = ts.starts[order[begin:begin + batch_size]]
            x_cont, x_cat, pose, mask = window_batch(feats, ts.kp, pick, seg_len, kind)
            x_cont, x_cat = x_cont.to(dev), x_cat.to(dev)
            if kind == "P":
                pose, mask = pose.to(dev), mask.to(dev)
            _, nll = model(x_cont, x_cat) if kind == "T" else model(x_cont, x_cat, pose, mask)
            loss = nll.mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            optimizer.step()
            losses.append(loss.item() * len(pick))
            seen += len(pick)
        losses_by_epoch.append(float(np.sum(losses) / max(seen, 1)))
        log(f"[TrajVAD-{kind} {dataset} s{seed}] epoch {epoch + 1}/{epochs} loss {losses_by_epoch[-1]:.4f} ({(time.perf_counter() - t0) / 60:.1f} min)")
    config = {"kind": kind, "dataset": dataset, "seg_len": seg_len, "seed": seed, "epochs": epochs, "batch_size": batch_size, "lr": lr,
              "n_windows": n, "params": count_parameters(model), "epoch_losses": losses_by_epoch,
              "train_minutes": (time.perf_counter() - t0) / 60, "threads": threads, "device": device}
    model = model.cpu()
    torch.save({"kind": kind, "state_dict": model.state_dict(), "scaler": {"mean": scaler.mean, "std": scaler.std}, "config": config},
               out_dir / "final.pt")
    (out_dir / "config.json").write_text(json.dumps(config, indent=1))
    (out_dir / "DONE").write_text(time.strftime("%Y-%m-%d %H:%M:%S\n"))
    return out_dir / "final.pt"


def load_trajvad(path: Path) -> tuple[nn.Module, Scaler, str, int]:
    saved = torch.load(path, map_location="cpu", weights_only=False)
    cfg = saved["config"]
    model = build_model(saved["kind"], cfg["seg_len"])
    model.load_state_dict(saved["state_dict"])
    mark_initialised(model)
    return model.eval(), Scaler(saved["scaler"]["mean"], saved["scaler"]["std"]), saved["kind"], cfg["seg_len"]


# =====================================================================================================================
# Scorer (track interface of q2_p1_eval)
# =====================================================================================================================
class TrajVadScorer:
    """normality = -NLL of every complete window (L consecutive poses of a track, <= 2 missing frames)."""

    def __init__(self, model: nn.Module, scaler: Scaler, kind: str, seg_len: int, chunk: int = 4096, device: str = "cpu"):
        self.device = torch.device(device)
        self.model, self.scaler, self.kind, self.seg_len, self.chunk = model.eval().to(self.device), scaler, kind, seg_len, chunk
        self.name = f"TrajVAD-{kind}"

    @classmethod
    def from_checkpoint(cls, path: Path, device: str = "cpu") -> "TrajVadScorer":
        model, scaler, kind, seg_len = load_trajvad(path)
        return cls(model, scaler, kind, seg_len, device=device)

    def track_windows(self, frames: np.ndarray, kp: np.ndarray, starts: np.ndarray):
        feats = self.scaler(trajectory_features(frames, kp))
        return feats, starts

    def score_tracks(self, tracks: list[tuple[np.ndarray, np.ndarray]]) -> list[np.ndarray]:
        from edgelab.models.lorad import starts_of

        L = self.seg_len
        conts, poses, masks, counts = [], [], [], []
        for frames, kp in tracks:
            s = starts_of(frames, L)
            counts.append(len(s))
            if len(s) == 0:
                continue
            feats = self.scaler(trajectory_features(frames, kp))
            x_cont, _, pose, mask = window_batch(feats, kp.astype(np.float32), s, L, self.kind)
            conts.append(x_cont)
            if self.kind == "P":
                poses.append(pose)
                masks.append(mask)
        if not conts:
            return [np.zeros(0) for _ in tracks]
        x_cont = torch.cat(conts)
        pose = torch.cat(poses) if poses else None
        mask = torch.cat(masks) if masks else None
        out = []
        with torch.no_grad():
            for i in range(0, len(x_cont), self.chunk):
                sl = slice(i, i + self.chunk)
                dev = self.device
                x_cat = torch.zeros(len(x_cont[sl]), L, 1, dtype=torch.long, device=dev)
                if self.kind == "T":
                    _, nll = self.model(x_cont[sl].to(dev), x_cat)
                else:
                    _, nll = self.model(x_cont[sl].to(dev), x_cat, pose[sl].to(dev), mask[sl].to(dev))
                out.append(nll.cpu().numpy().astype(np.float64))
        nll = np.concatenate(out)
        res, at = [], 0
        for c in counts:
            res.append(-nll[at:at + c])
            at += c
        return res


# =====================================================================================================================
# Online (per-pose) TrajVAD for the edge runner: incremental features, ring buffer of the last L poses, one batched flow call per frame
# =====================================================================================================================
class _TrackState:
    """Incremental version of `trajectory_features` for one track: O(1) per pose, identical output (unit-tested)."""

    def __init__(self, L: int):
        self.n = 0
        self.frame_prev = 0
        self.x = np.zeros(8)
        self.P = 1000.0 * np.eye(8)
        self.pos_prev = np.zeros(2)
        self.vel_prev = np.zeros(2)
        self.acc_prev = np.zeros(2)
        self.log_area_prev = 0.0
        self.ratio_prev = 0.0
        self.pos_hist: list[np.ndarray] = []  # last 12 positions
        self.step_hist: list[float] = []  # last 11 step norms
        self.feats: list[np.ndarray] = []  # last L scaled feature rows
        self.kps: list[np.ndarray] = []
        self.frames: list[int] = []
        self.L = L
        F = np.eye(8)
        F[:4, 4:] = np.eye(4)
        self.F, self.H = F, np.hstack([np.eye(4), np.zeros((4, 4))])
        self.Q, self.R, self.I8 = np.eye(8), 10.0 * np.eye(4), np.eye(8)

    def push(self, frame: int, kp: np.ndarray, vid_res=VID_RES) -> np.ndarray:
        """Raw 27-d feature row of this pose."""
        box = pose_boxes(kp[None].astype(np.float64))[0]
        n = self.n
        if n == 0:
            self.x = np.concatenate([box, np.zeros(4)])
            b = box.copy()
            dt = 1.0
        else:
            dt = float(np.clip(frame - self.frame_prev, 1, 30))
            for _ in range(int(dt)):
                self.x = self.F @ self.x
                self.P = self.F @ self.P @ self.F.T + self.Q
            S = self.H @ self.P @ self.H.T + self.R
            K = self.P @ self.H.T @ np.linalg.inv(S)
            self.x = self.x + K @ (box - self.H @ self.x)
            self.P = (self.I8 - K @ self.H) @ self.P
            b = self.x[:4].copy()
        vw, vh = vid_res
        cx, cy, w, h = b[0] / vw, b[1] / vh, b[2] / vw, b[3] / vh
        pos = np.array([cx, cy])
        area, ratio = w * h, w / (h + 1e-6)
        if n == 0:
            vel, acc_v, jerk_v, expansion, ar_vel = (np.zeros(2),) * 3 + (0.0, 0.0)
        else:
            vel = (pos - self.pos_prev) / dt
            acc_v = (vel - self.vel_prev) / dt
            if n == 1:
                acc_v = np.zeros(2)
            jerk_v = (acc_v - self.acc_prev) / dt
            if n < 3:
                jerk_v = np.zeros(2)
            expansion = (math.log(max(area, 1e-6)) - self.log_area_prev) / dt
            ar_vel = (ratio - self.ratio_prev) / dt
        speed, acc = float(np.linalg.norm(vel)), float(np.linalg.norm(acc_v))
        orient = vel / max(speed, 1e-12) if speed > 1e-6 else np.zeros(2)
        jerk = float(np.linalg.norm(jerk_v))
        sq = vel[0] ** 2 + vel[1] ** 2
        curv = abs(vel[0] * acc_v[1] - vel[1] * acc_v[0]) / max(sq, 1e-12) ** 1.5 if sq > 1e-6 else 0.0
        # path efficiency over the last 12 positions (window 12 -> start index max(0, t - 11))
        if n > 0:
            self.step_hist.append(float(np.linalg.norm(pos - self.pos_prev)))
            self.step_hist = self.step_hist[-11:]
        self.pos_hist.append(pos)
        self.pos_hist = self.pos_hist[-12:]
        total = sum(self.step_hist)
        eff = float(np.linalg.norm(pos - self.pos_hist[0]) / total) if (n > 0 and total > 1e-6 and len(self.pos_hist) > 1) else 1.0
        conf = float(kp[:, 2].mean())
        row = np.array([cx, cy, w, h, vel[0], vel[1], speed, acc_v[0], acc_v[1], acc, area, ratio, speed / (h + 1e-6), speed / (w + 1e-6),
                        speed / (area + 1e-6), acc / (h + 1e-6), acc / (w + 1e-6), acc / (area + 1e-6), orient[0], orient[1], jerk, curv,
                        expansion, ar_vel, area * speed ** 2, eff, conf], dtype=np.float32)
        self.pos_prev, self.vel_prev, self.acc_prev = pos, vel, acc_v
        self.log_area_prev, self.ratio_prev = math.log(max(area, 1e-6)), ratio
        self.frame_prev = frame
        self.n += 1
        return row


class OnlineTrajVAD:
    """Live TrajVAD-T/P: `update_frame(frame_id, {track_id: keypoints17x3})` -> [(track_id, normality)] for the windows completed at this frame.

    State per track: Kalman + difference state (O(1)), ring buffers of the last L scaled feature rows / keypoints / frame ids.
    The windows completed by one frame are scored in ONE batched flow call. STG-NF continuity rule (>= L - 2 poses within L frames) applied."""

    def __init__(self, model: nn.Module, scaler: Scaler, kind: str, seg_len: int):
        self.model, self.scaler, self.kind, self.L = model.eval(), scaler, kind, seg_len
        self.tracks: dict[int, _TrackState] = {}

    @classmethod
    def from_checkpoint(cls, path: Path) -> "OnlineTrajVAD":
        model, scaler, kind, seg_len = load_trajvad(path)
        return cls(model, scaler, kind, seg_len)

    def forget(self, live_ids: set[int]) -> None:
        for tid in [t for t in self.tracks if t not in live_ids]:
            del self.tracks[tid]

    @torch.no_grad()
    def update_frame(self, frame_id: int, poses: dict[int, np.ndarray]) -> list[tuple[int, float]]:
        L, ready, conts, kps = self.L, [], [], []
        for tid, kp in poses.items():
            st = self.tracks.setdefault(tid, _TrackState(L))
            raw = st.push(frame_id, np.asarray(kp, dtype=np.float32))
            st.feats = (st.feats + [self.scaler(raw[None])[0]])[-L:]
            st.kps = (st.kps + [np.asarray(kp, dtype=np.float32)])[-L:]
            st.frames = (st.frames + [frame_id])[-L:]
            if len(st.feats) < L:
                continue
            frames = np.asarray(st.frames)
            if np.count_nonzero(frames < frames[0] + L) < L - 2:
                continue
            ready.append(tid)
            conts.append(np.stack(st.feats))
            kps.append(np.stack(st.kps))
        if not ready:
            return []
        x_cont = torch.from_numpy(np.stack(conts))
        x_cat = torch.zeros(len(ready), L, 1, dtype=torch.long)
        if self.kind == "T":
            _, nll = self.model(x_cont, x_cat)
        else:
            pose, mask = robust_normalize_pose(torch.from_numpy(np.stack(kps)))
            _, nll = self.model(x_cont, x_cat, pose, mask)
        return [(tid, -float(v)) for tid, v in zip(ready, nll.numpy())]
