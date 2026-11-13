"""Training batches for STG-NF variants: windows from a SegmentBank, optionally with random frame drops + filling (EXP-018 C1).

    batch = make_batch(bank, rows, fill=None | "cv" | "gru", predictor, rng, flip)  ->  (x (B, 2, T, 18), provenance (B, T) | None)

fill=None: the original windows (the EXP-001 recipe). Otherwise every window gets its own random keep mask over its grid
(context + window): rate r ~ U(0, 0.85), made by the latest-frame deadline simulator with random service times (a slow
frame is followed by a burst of drops, like a real CPU); frames the mask drops count as missing, the filler predicts them
(G = 8, the same rules as at test time, see edgelab.pose.predictor), and the window is then cut the way STG-NF cuts a
track: the first T poses from the first one at or after the window start, kept if >= T - 2 of them lie within T frames.
Windows that do not survive are dropped (at test time they would not be scored either). provenance: 1 = detected, 0 = predicted.

Normalisation is STG-NF's (dataset.py + utils/data_utils.normalize_pose): x / 856, y / 480, minus the mean of the window,
divided by the std of y; tests/test_pose_augment.py checks it against their function.
"""

import numpy as np
import torch

from edgelab.pipeline.deadline_sim import simulate_latest_frame
from edgelab.pose.predictor import FILL_G, fill_batch
from edgelab.pose.segments import SegmentBank

VID_RES = (856.0, 480.0)  # STG-NF's default, used for ShanghaiTech and for UBnormal alike
COCO18_ORDER = [0, 17, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3]  # utils: keypoints17_to_coco18
MAX_DROP_RATE = 0.85
WARMUP = 16  # frames simulated before the grid starts, so the first frame is not always a processed one


def random_keep_masks(n: int, length: int, rng: np.random.Generator, max_rate: float = MAX_DROP_RATE) -> np.ndarray:
    """(n, length) bool, True = processed. Row rate r ~ U(0, max_rate): service times average 1 / (1 - r) frame periods
    (log-normal, sigma 0.5, independent per frame), fed to the latest-frame simulator of EXP-004."""
    masks = np.empty((n, length), dtype=bool)
    rates = rng.uniform(0.0, max_rate, n)
    for i, rate in enumerate(rates):
        mean_frames = 1.0 / (1.0 - rate)
        service_frames = mean_frames * np.exp(0.5 * rng.standard_normal(length + WARMUP) - 0.125)
        masks[i] = simulate_latest_frame(service_frames * 1000.0 / 24.0, length + WARMUP, fps=24.0)[WARMUP:]
    return masks


def select_windows(xy: torch.Tensor, valid: torch.Tensor, predicted: torch.Tensor, ctx: int, seg_len: int):
    """Cut STG-NF windows out of (B, L, 17, 2) poses of a grid. -> (windows (B', T, 17, 2), provenance (B', T), kept rows (B',))."""
    region = valid[:, ctx:]
    rank = torch.cumsum(region.long(), dim=1) - 1
    wanted = torch.arange(seg_len)[None, :, None]
    one_hot = (rank[:, None, :] == wanted) & region[:, None, :]  # (B, T, W): the r-th valid pose of the region sits at column w
    exists = one_hot.any(dim=2)
    position = one_hot.float().argmax(dim=2)  # (B, T)
    ok = exists.all(dim=1) & ((position < position[:, :1] + seg_len).sum(dim=1) >= seg_len - 2)
    column = ctx + position
    windows = torch.gather(xy, 1, column[..., None, None].expand(-1, -1, 17, 2))
    provenance = 1.0 - torch.gather(predicted, 1, column).float()
    return windows[ok], provenance[ok], torch.nonzero(ok).flatten()


def to_flow_input(window: torch.Tensor, flip: torch.Tensor | None = None) -> torch.Tensor:
    """(B, T, 17, 2) pixel keypoints -> (B, 2, T, 18) normalised, as STG-NF feeds its flow. flip: (B,) bool, x -> -x first."""
    window = window.double()
    if flip is not None:
        sign = torch.where(flip, -1.0, 1.0).double()
        window = torch.stack([window[..., 0] * sign[:, None, None], window[..., 1]], dim=-1)
    neck = 0.5 * (window[:, :, 5] + window[:, :, 6])
    window = torch.cat([window, neck[:, :, None]], dim=2)[:, :, COCO18_ORDER]
    return normalise(window).float().permute(0, 3, 1, 2)


def normalise(window: torch.Tensor) -> torch.Tensor:
    """(B, T, V, 2) pixels -> (B, T, V, 2): / video size, minus the mean over (T, V), divided by the std of y over (T, V)."""
    scaled = window / torch.tensor(VID_RES, dtype=window.dtype)
    centred = scaled - scaled.mean(dim=(1, 2), keepdim=True)
    return centred / scaled[..., 1].std(dim=(1, 2), unbiased=False)[:, None, None, None]


def make_batch(bank: SegmentBank, rows: np.ndarray, fill: str | None, predictor, rng: np.random.Generator,
               flip: np.ndarray, max_gap: int = FILL_G):
    """(x (B', 2, T, 18), provenance (B', T) or None) for the windows `rows` of the bank; B' <= len(rows)."""
    grid = torch.from_numpy(bank.grid[rows].astype(np.int64))
    present = grid >= 0
    xy = torch.from_numpy(bank.xy[grid.clamp(min=0).numpy()])  # (B, L, 17, 2)
    flip = torch.from_numpy(np.asarray(flip, dtype=bool))
    if fill is None:
        windows, _, kept = select_windows(xy, present, torch.zeros_like(present), bank.ctx, bank.seg_len)
        return to_flow_input(windows, flip[kept]), None
    keep = torch.from_numpy(random_keep_masks(len(rows), grid.shape[1], rng))
    detected = present & keep
    pose, filled = fill_batch(fill, predictor, xy, detected, max_gap)
    windows, provenance, kept = select_windows(pose, detected | filled, filled, bank.ctx, bank.seg_len)
    return to_flow_input(windows, flip[kept]), provenance
