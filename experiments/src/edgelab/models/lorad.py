"""LoRaD causal pose flow (Zhang et al., JRTIP 2026): causal ST-GCN normalising flow + online circular buffer (Q2-P1-E06).

The flow is `edgelab.models.stgnf_flow.CouplingFlow` with `CausalGraphSubnet` coupling networks (left-only temporal padding,
frame t depends on frames <= t). Two ways to normalise a track before it enters the flow:

    N0  per-window   x -> (x - mean_window) / std_y_window     the STG-NF rule: needs the whole window, so every new frame
                                                               re-normalises (and re-scores) all L frames of the window
    N1  causal EMA   x_t -> (x_t - C_t) / sigma_t              C_t, sigma_t = exponential moving averages (span L, started
                                                               at the first pose of the TRACK) of the per-pose centroid and
                                                               the first / second moment of y. A pose is normalised ONCE,
                                                               the moment it arrives, and never changes: O(1) per frame

Three ways to turn a causal flow into a window score (window = L consecutive poses of one track, completed at its last pose):

    window mode   run the flow on the L normalised poses of the window (zero padding at the window start). N0 and N1.
    stream mode   N1 only: run the flow over the whole track prefix, one pose at a time, with a ring buffer of k - 1 frames per
                  coupling step (`CausalFlowStreamer`), and average the last L per-frame NLLs in a circular buffer
                  (`OnlineLoRaD`). O(steps * k) work per frame, independent of L and of the track length, and it equals the
                  whole-track batch evaluation exactly (tests/test_q2_p1_e06.py).

Everything here is NumPy / PyTorch; no Ultralytics. Normalisation uses STG-NF's 856 x 480 constants for both datasets.
"""

from __future__ import annotations

import json
import math
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from scipy.signal import lfilter

from edgelab.models.stgnf_flow import CouplingFlow, build_flow, count_parameters, frame_nll
from edgelab.pose.augment import COCO18_ORDER, VID_RES, to_flow_input
from edgelab.pose.segments import read_tracks, window_starts
from edgelab.pose.stgnf_eval import SEG_LENS

STRIDE = 6  # STG-NF training windows: every 6th pose of a track
EPS = 1e-6
VID = np.array(VID_RES, dtype=np.float64)  # (856, 480)


# =====================================================================================================================
# Normalisation
# =====================================================================================================================
def to_coco18(xy17: np.ndarray) -> np.ndarray:
    """(..., 17, 2) -> (..., 18, 2) in STG-NF's joint order (neck = mean of the shoulders)."""
    neck = 0.5 * (xy17[..., 5, :] + xy17[..., 6, :])
    return np.concatenate([xy17, neck[..., None, :]], axis=-2)[..., COCO18_ORDER, :]


def ema_alpha(seg_len: int) -> float:
    """EMA span = window length: alpha = 2 / (L + 1)."""
    return 2.0 / (seg_len + 1.0)


def ema(x: np.ndarray, alpha: float) -> np.ndarray:
    """y_0 = x_0, y_t = alpha x_t + (1 - alpha) y_{t-1}, along axis 0 (causal, O(1) state)."""
    if len(x) == 0:
        return x.copy()
    zi = ((1.0 - alpha) * x[:1]).astype(np.float64)
    y, _ = lfilter([alpha], [1.0, -(1.0 - alpha)], x.astype(np.float64), axis=0, zi=zi)
    return y


def ema_normalize_track(xy17: np.ndarray, alpha: float) -> np.ndarray:
    """N1: (n, 17, 2) pixel keypoints of one track -> (n, 18, 2) normalised, every pose using only poses <= its own."""
    p = to_coco18(np.asarray(xy17, dtype=np.float64)) / VID  # (n, 18, 2)
    centroid = p.mean(axis=1)  # (n, 2)
    y = p[..., 1]
    moments = np.stack([y.mean(axis=1), (y ** 2).mean(axis=1)], axis=1)  # (n, 2)
    c, m = ema(centroid, alpha), ema(moments, alpha)
    sigma = np.sqrt(np.maximum(m[:, 1] - m[:, 0] ** 2, EPS))
    return (p - c[:, None, :]) / sigma[:, None, None]


class EmaNormalizer:
    """The streaming form of `ema_normalize_track`: one `update(xy17)` per pose, O(1) state (centroid, two y moments)."""

    def __init__(self, alpha: float):
        self.alpha = alpha
        self.c: np.ndarray | None = None
        self.m: np.ndarray | None = None

    def update(self, xy17: np.ndarray) -> np.ndarray:
        p = to_coco18(np.asarray(xy17, dtype=np.float64)) / VID  # (18, 2)
        centroid, y = p.mean(axis=0), p[:, 1]
        moments = np.array([y.mean(), (y ** 2).mean()])
        if self.c is None:
            self.c, self.m = centroid, moments
        else:
            self.c = self.alpha * centroid + (1 - self.alpha) * self.c
            self.m = self.alpha * moments + (1 - self.alpha) * self.m
        sigma = math.sqrt(max(self.m[1] - self.m[0] ** 2, EPS))
        return (p - self.c) / sigma


def window_input(window_xy17: np.ndarray) -> torch.Tensor:
    """N0: (B, T, 17, 2) pixel windows -> (B, 2, T, 18) flow input, each window by its own mean / std (STG-NF)."""
    return to_flow_input(torch.from_numpy(np.ascontiguousarray(window_xy17, dtype=np.float32)))


def stack_normalized(windows_n: np.ndarray) -> torch.Tensor:
    """N1: (B, T, 18, 2) already-normalised poses -> (B, 2, T, 18) flow input."""
    return torch.from_numpy(np.ascontiguousarray(windows_n.transpose(0, 3, 1, 2), dtype=np.float32))


# =====================================================================================================================
# Training windows
# =====================================================================================================================
@dataclass
class TrainSet:
    xy: np.ndarray  # (P, 17, 2) float32 pixel keypoints of every pose of every training track
    xn: np.ndarray  # (P, 18, 2) float32 N1-normalised poses (EMA over the whole track prefix)
    starts: np.ndarray  # (N,) int64 index into xy / xn of each window's first pose (a window = seg_len consecutive poses)
    seg_len: int

    def __len__(self) -> int:
        return len(self.starts)

    def batch(self, rows: np.ndarray, norm: str, flip: np.ndarray) -> torch.Tensor:
        idx = self.starts[rows][:, None] + np.arange(self.seg_len)[None, :]
        if norm == "N0":
            xy = self.xy[idx].copy()  # (B, T, 17, 2)
            if flip.any():
                xy[flip, ..., 0] = -xy[flip, ..., 0]  # mirror about x = 0 before centring == negate centred x
            return window_input(xy)
        x = stack_normalized(self.xn[idx])
        if flip.any():
            x[torch.from_numpy(flip), 0] = -x[torch.from_numpy(flip), 0]
        return x


def build_trainset(pose_dir: Path, seg_len: int, stride: int = STRIDE, cache: Path | None = None) -> TrainSet:
    """All training tracks of a pose folder -> `TrainSet` (cut exactly as STG-NF cuts them, see edgelab.pose.segments.window_starts)."""
    if cache is not None and Path(cache).exists():
        z = np.load(cache)
        return TrainSet(z["xy"], z["xn"], z["starts"], int(z["seg_len"]))
    xy_parts, xn_parts, starts, base = [], [], [], 0
    alpha = ema_alpha(seg_len)
    for path in sorted(Path(pose_dir).glob("*_tracked_person.json")):
        for frames, kp in read_tracks(path):
            s = window_starts(frames, seg_len, stride)
            xy = kp[:, :, :2].astype(np.float32)
            xy_parts.append(xy)
            xn_parts.append(ema_normalize_track(xy, alpha).astype(np.float32))
            if len(s):
                starts.append(s.astype(np.int64) + base)
            base += len(frames)
    ts = TrainSet(np.concatenate(xy_parts), np.concatenate(xn_parts), np.concatenate(starts), seg_len)
    if cache is not None:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, xy=ts.xy, xn=ts.xn, starts=ts.starts, seg_len=ts.seg_len)
    return ts


# =====================================================================================================================
# Model files and training
# =====================================================================================================================
def lorad_kind(arm: str) -> tuple[str, str]:
    """arm -> (flow kind, normalisation). A1 symmetric STG-NF + N0, A2 causal + N0, A3 causal + N1, A2L causal-lite + N0."""
    table = {"A1": ("graph", "N0"), "A2": ("causal", "N0"), "A3": ("causal", "N1"), "A2L": ("causal-lite", "N0"),
             "A3L": ("causal-lite", "N1")}
    return table[arm]


def mark_actnorm(model: torch.nn.Module) -> None:
    from models.STG_NF.modules_pose import ActNorm2d

    for module in model.modules():
        if isinstance(module, ActNorm2d):
            module.inited = True


def train_flow(arm: str, dataset: str, seed: int, out_dir: Path, trainset: TrainSet, epochs: int = 10, batch_size: int = 256,
               lr: float = 5e-4, threads: int = 2, max_batches: int | None = None, log=print) -> Path:
    """STG-NF recipe (Adamax lr 5e-4 x 0.99^epoch, grad-clip 100, identity + mirror augmentation, prior N(3, 1)); one folder per run."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / "DONE").exists():
        return out_dir / "final.pt"
    kind, norm = lorad_kind(arm)
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    seg_len = trainset.seg_len
    model = build_flow(kind, seg_len)
    optimizer = torch.optim.Adamax(model.parameters(), lr=lr)
    n, t0, losses_by_epoch = len(trainset), time.perf_counter(), []
    start_epoch = 0
    if (out_dir / "last.pt").exists():
        saved = torch.load(out_dir / "last.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(saved["state_dict"])
        optimizer.load_state_dict(saved["optimizer"])
        mark_actnorm(model)
        start_epoch, losses_by_epoch = saved["epoch"], saved["losses"]
    for epoch in range(start_epoch, epochs):
        rng = np.random.default_rng([seed, epoch])
        rows = np.concatenate([np.arange(n), np.arange(n)])
        flips = np.concatenate([np.zeros(n, dtype=bool), np.ones(n, dtype=bool)])
        order = rng.permutation(2 * n)
        model.train()
        losses = []
        for b, begin in enumerate(range(0, 2 * n, batch_size)):
            if max_batches is not None and b >= max_batches:
                break
            pick = order[begin:begin + batch_size]
            x = trainset.batch(rows[pick], norm, flips[pick])
            _, nll = model(x)
            loss = nll.mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 100)
            optimizer.step()
            optimizer.zero_grad()
            losses.append(loss.item())
        for group in optimizer.param_groups:
            group["lr"] = lr * 0.99 ** epoch
        losses_by_epoch.append(float(np.mean(losses)))
        torch.save({"state_dict": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch + 1,
                    "losses": losses_by_epoch}, out_dir / "last.pt.tmp")
        (out_dir / "last.pt.tmp").replace(out_dir / "last.pt")
        log(f"[{arm} {dataset} s{seed}] epoch {epoch + 1}/{epochs} loss {losses_by_epoch[-1]:.4f} ({(time.perf_counter() - t0) / 60:.1f} min)")
    config = {"arm": arm, "kind": kind, "norm": norm, "dataset": dataset, "seg_len": seg_len, "seed": seed, "epochs": epochs,
              "batch_size": batch_size, "lr": lr, "n_windows": n, "params": count_parameters(model),
              "epoch_losses": losses_by_epoch, "train_minutes": (time.perf_counter() - t0) / 60, "threads": threads}
    torch.save({"kind": kind, "norm": norm, "state_dict": model.state_dict(), "config": config}, out_dir / "final.pt")
    (out_dir / "config.json").write_text(json.dumps(config, indent=1))
    (out_dir / "DONE").write_text(time.strftime("%Y-%m-%d %H:%M:%S\n"))
    return out_dir / "final.pt"


def load_flow(path: Path) -> tuple[CouplingFlow, str, int]:
    """final.pt -> (flow in eval mode, norm 'N0' | 'N1', seg_len)."""
    saved = torch.load(path, map_location="cpu", weights_only=False)
    cfg = saved["config"]
    model = build_flow(saved["kind"], cfg["seg_len"])
    model.load_state_dict(saved["state_dict"])
    mark_actnorm(model)
    return model.eval(), saved.get("norm", cfg.get("norm", "N0")), cfg["seg_len"]


# =====================================================================================================================
# Scoring tracks (window table engine)
# =====================================================================================================================
def starts_of(frames: np.ndarray, seg_len: int, max_missing: int = 2) -> np.ndarray:
    """Pose indices where an STG-NF-complete window starts (same rule as q2_p1_e00.complete_window_starts)."""
    frames = np.asarray(frames)
    n = len(frames)
    if n < seg_len:
        return np.zeros(0, dtype=np.int64)
    s = np.arange(n - seg_len + 1)
    in_span = np.searchsorted(frames, frames[s] + seg_len, side="left") - s
    return s[np.minimum(in_span, seg_len) >= seg_len - max_missing]


class FlowScorer:
    """Scores the complete windows of tracks with any `CouplingFlow` + normalisation + mode. normality = -NLL (bits / dim), higher = more normal.

    mode 'window' (default) or 'stream' (N1 + causal flow only: whole-track causal evaluation, window = mean of its L frame NLLs).
    """

    def __init__(self, flow: CouplingFlow, norm: str, seg_len: int, mode: str = "window", chunk: int = 4096):
        if mode == "stream" and norm != "N1":
            raise ValueError("stream mode needs the causal EMA normalisation N1")
        self.flow, self.norm, self.seg_len, self.mode, self.chunk = flow, norm, seg_len, mode, chunk
        self.alpha = ema_alpha(seg_len)

    @classmethod
    def from_checkpoint(cls, path: Path, mode: str = "window") -> "FlowScorer":
        flow, norm, seg_len = load_flow(path)
        return cls(flow, norm, seg_len, mode)

    def _nll(self, x: torch.Tensor) -> np.ndarray:
        out = []
        with torch.no_grad():
            for i in range(0, len(x), self.chunk):
                _, nll = self.flow(x[i:i + self.chunk])
                out.append(nll.numpy().astype(np.float64))
        return np.concatenate(out) if out else np.zeros(0)

    def score_tracks(self, tracks: list[tuple[np.ndarray, np.ndarray]]) -> list[np.ndarray]:
        """tracks: [(frames (n,), keypoints (n, 17, 3))] -> per track the normality of every complete window (in start order)."""
        L = self.seg_len
        if self.mode == "stream":
            return [self._stream_track(frames, kp) for frames, kp in tracks]
        wins, counts = [], []
        for frames, kp in tracks:
            s = starts_of(frames, L)
            counts.append(len(s))
            if len(s) == 0:
                continue
            idx = s[:, None] + np.arange(L)[None, :]
            if self.norm == "N0":
                wins.append(kp[:, :, :2][idx])  # (m, L, 17, 2) pixels
            else:
                wins.append(ema_normalize_track(kp[:, :, :2], self.alpha)[idx])  # (m, L, 18, 2) normalised once per pose
        if not wins:
            return [np.zeros(0) for _ in tracks]
        allw = np.concatenate(wins)
        x = window_input(allw) if self.norm == "N0" else stack_normalized(allw)
        nll = self._nll(x)
        out, at = [], 0
        for c in counts:
            out.append(-nll[at:at + c])
            at += c
        return out

    def _stream_track(self, frames: np.ndarray, kp: np.ndarray) -> np.ndarray:
        L = self.seg_len
        s = starts_of(frames, L)
        if len(s) == 0:
            return np.zeros(0)
        xn = ema_normalize_track(kp[:, :, :2], self.alpha)  # (n, 18, 2)
        x = stack_normalized(xn[None])  # (1, 2, n, 18)
        with torch.no_grad():
            per_frame = frame_nll(self.flow, x)[0].numpy().astype(np.float64)  # (n,)
        csum = np.concatenate([[0.0], np.cumsum(per_frame)])
        return -(csum[s + L] - csum[s]) / L


# =====================================================================================================================
# O(1) streaming: ring buffers per coupling step + circular buffer of per-frame NLLs
# =====================================================================================================================
class CausalFlowStreamer:
    """Evaluates a causal `CouplingFlow` one pose at a time. State per track: k - 1 past frames of the (post-BN, post-ReLU)
    graph-conv output of every coupling step. `step(x_t)` returns frame t's NLL share (bits / dim) of the whole-prefix evaluation."""

    def __init__(self, flow: CouplingFlow):
        self.flow = flow.eval()
        self.buffers: list[torch.Tensor | None] = [None] * len(flow.steps)
        self.t = 0

    def reset(self) -> None:
        self.buffers = [None] * len(self.flow.steps)
        self.t = 0

    @torch.no_grad()
    def step(self, x_t: torch.Tensor) -> float:
        """x_t: (2, 18) normalised pose -> NLL of this frame in bits per dimension."""
        h = x_t[None, :, None, :]  # (1, 2, 1, V)
        v = h.shape[3]
        logdet = 0.0
        for i, step in enumerate(self.flow.steps):
            z = step.permute(step.actnorm_frame(h), reverse=False)
            z1, z2 = z[:, 0:1], z[:, 1:2]
            gcn = step.subnet.gcn
            residual = gcn.residual(z1)
            g, _ = gcn.gcn(z1, step.subnet.A)
            g = gcn.relu1(gcn.bn1(g))  # (1, 2, 1, V): this frame's input to the temporal convolution
            past = self.buffers[i]
            full = g if past is None else torch.cat([past, g], dim=2)  # <= k frames
            window = full if full.shape[2] >= gcn.k else torch.nn.functional.pad(full, (0, 0, gcn.k - full.shape[2], 0))
            self.buffers[i] = full[:, :, -(gcn.k - 1):] if gcn.k > 1 else None  # keep the last k - 1 frames: the ring buffer
            out = torch.relu(gcn.bn2(gcn.tconv(window)) + residual)  # (1, 2, 1, V)
            shift, scale = out[:, 0:1], torch.sigmoid(out[:, 1:2] + 2.0) + 1e-6
            z2 = (z2 + shift) * scale
            logdet = logdet + float(torch.log(scale).sum()) + float(step.actnorm.logs.sum()) * v
            h = torch.cat([z1, z2], dim=1)
        log_p = float((-0.5 * (((h - self.flow.prior_mean) ** 2) / (1.0 + 1e-6) + math.log(2 * math.pi))).sum())
        self.t += 1
        return -(log_p + logdet) / (math.log(2.0) * 2 * v)


class CircularMean:
    """Mean of the last `size` values with a ring buffer and a running sum: O(1) per push."""

    def __init__(self, size: int):
        self.size, self.buf, self.total, self.i, self.n = size, np.zeros(size), 0.0, 0, 0

    def push(self, value: float) -> float | None:
        if self.n == self.size:
            self.total -= self.buf[self.i]
        else:
            self.n += 1
        self.buf[self.i] = value
        self.total += value
        self.i = (self.i + 1) % self.size
        return self.total / self.size if self.n == self.size else None


class OnlineLoRaD:
    """Live LoRaD for the edge pipeline: `update(track_id, frame_id, keypoints17x3)` -> normality of the window completed by this pose, or None.

    N1 stream mode (default): per track a `EmaNormalizer`, a `CausalFlowStreamer` and a `CircularMean` of the last L frame NLLs:
    O(1) per pose. N0 window mode (`norm='N0'`): per track a deque of the last L poses, re-normalised and re-scored in full each frame.
    The STG-NF continuity rule (>= L - 2 poses within L frames of the window's first) is applied to the window's frame ids.
    """

    def __init__(self, flow: CouplingFlow, norm: str, seg_len: int, mode: str = "stream"):
        if mode == "stream" and norm != "N1":
            raise ValueError("stream mode needs N1")
        self.flow, self.norm, self.L, self.mode = flow.eval(), norm, seg_len, mode
        self.alpha = ema_alpha(seg_len)
        self.tracks: dict[int, dict] = {}

    def forget(self, live_ids: set[int]) -> None:
        for tid in [t for t in self.tracks if t not in live_ids]:
            del self.tracks[tid]

    def update(self, track_id: int, frame_id: int, keypoints: np.ndarray) -> float | None:
        st = self.tracks.get(track_id)
        if st is None:
            st = {"frames": deque(maxlen=self.L)}
            if self.mode == "stream":
                st.update(norm=EmaNormalizer(self.alpha), flow=CausalFlowStreamer(self.flow), nll=CircularMean(self.L))
            else:
                st["xy"] = deque(maxlen=self.L)
            self.tracks[track_id] = st
        st["frames"].append(frame_id)
        if self.mode == "stream":
            xn = st["norm"].update(keypoints[:, :2])
            window_nll = st["nll"].push(st["flow"].step(torch.from_numpy(xn.T.astype(np.float32))))
        else:
            st["xy"].append(np.asarray(keypoints[:, :2], dtype=np.float32))
            window_nll = None
            if len(st["xy"]) == self.L:
                with torch.no_grad():
                    _, nll = self.flow(window_input(np.stack(st["xy"])[None]))
                window_nll = float(nll[0])
        if window_nll is None:
            return None
        frames = np.fromiter(st["frames"], dtype=np.int64)
        if np.count_nonzero(frames < frames[0] + self.L) < self.L - 2:
            return None
        return -float(window_nll)
