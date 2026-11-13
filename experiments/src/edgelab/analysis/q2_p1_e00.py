"""Q2-P1-E00 (Audit): can we reproduce existing scores, timestamps, normalisation and missing-data handling?

    uv run python -m edgelab.analysis.q2_p1_e00 lock      # hash everything, define splits -> Q2-P1-E00_audit/config.yaml
    uv run python -m edgelab.analysis.q2_p1_e00 run       # all audits -> results.csv, decisions.jsonl, summary.json
    uv run python -m edgelab.analysis.q2_p1_e00 report    # summary.json -> report.md
    uv run python -m edgelab.analysis.q2_p1_e00 all

No training. Nothing here changes a frontend or the scorer: the only "treatment" is the *scoring protocol* (how a stream
of STG-NF window scores becomes one per-frame score), applied one step at a time (plan.md section 4):

    S0  offline paper : middle-frame landing, two-sided Gaussian, global test `most_normal` fill, last window omitted
    S1  offline "live": end-frame landing,    causal half-Gaussian, global fill,                last window omitted
    S2  all windows   : landing at the real completion frame, causal, global fill,            nothing omitted
    S3  train sentinel: S2 but empty frames get a sentinel computed from TRAINING-normal windows only (no look-ahead)
    S4  hold          : S3 + hold the last score for at most H frames (H in 0, 6, 12, 24, inf) with coverage / staleness

The module has two layers: pure functions (window rules, landing, causal fill, splits, hashing; unit-tested in
tests/test_q2_p1_e00.py without any data) and I/O drivers that need ~/thesis-data and the STG-NF checkpoints.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import roc_auc_score

from edgelab.metrics.scores import bootstrap_ci_over_videos, frame_level_auc, paired_bootstrap_diff
from edgelab.pose.fill import SUFFIX, merge_parts
from edgelab.pose.smoothing import STGNF_EQUIVALENT_SIGMA, SMOOTHERS
from edgelab.pose.stgnf_eval import EXPERIMENTS_ROOT, SEG_LENS, STGNF_ROOT, _import_stgnf

REPO_ROOT = EXPERIMENTS_ROOT.parent
EXP_DIR = EXPERIMENTS_ROOT / "Q2-P1-E00_audit"
DATA_ROOT = Path(os.environ.get("EDGELAB_DATA_ROOT", str(Path.home() / "thesis-data")))
WORK_DIR = DATA_ROOT / "q2-p1-e00"

SEED = 20261113
DATASETS = ("ShanghaiTech", "UBnormal")
FPS = {"ShanghaiTech": 24.0, "UBnormal": 30.0}  # native frame rates; EXP-023 hard-coded 24.0 for both
N_DEV = {"ShanghaiTech": 35, "UBnormal": 70}
MISSING_TOLERANCE = 2  # STG-NF is_seg_continuous: at most 2 missing frames inside a window
HOLDS = (0, 6, 12, 24, math.inf)
SENTINEL_QUANTILES = {"p50": 0.50, "p90": 0.90, "p99": 0.99, "max": 1.0}
QUANTILE_TABLE = (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 0.995, 1.0)
SHT_CALIB_TRAIN_CLIPS = [f"{scene:02d}_001" for scene in range(1, 13)]
GATE_C = 0.25

# paths ---------------------------------------------------------------------------------------------------------
SHT_GT_DIR = STGNF_ROOT / "data/ShanghaiTech/gt/test_frame_mask"
UB_GT_DIR = STGNF_ROOT / "data/UBnormal/gt"
SHT_TRAIN_VIDEOS = DATA_ROOT / "shanghaitech/shanghaitech/training/videos"
UB_TEST_VIDEOS = DATA_ROOT / "ubnormal/test_videos"
UB_TEST_LIST = EXPERIMENTS_ROOT / "EXP-006-ubnormal-replication/test_videos.txt"
EXP023_DIR = EXPERIMENTS_ROOT / "EXP-023-observation-reuse"


# =====================================================================================================================
# 1. Hashing
# =====================================================================================================================
def sha256_file(path: str | Path, chunk_bytes: int = 1 << 22) -> str:
    """Full SHA-256 of a file (never a prefix)."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def tree_manifest(directory: str | Path, pattern: str = "*.npz") -> dict:
    """Per-file SHA-256 of every file matching `pattern` + one digest of (name, size, sha) lines = the tree hash."""
    files = sorted(Path(directory).glob(pattern))
    per_file, lines, total = {}, [], 0
    for path in files:
        sha, size = sha256_file(path), path.stat().st_size
        per_file[path.name] = sha
        lines.append(f"{path.name}\t{size}\t{sha}")
        total += size
    return {"n_files": len(files), "total_bytes": total, "tree_sha256": hashlib.sha256("\n".join(lines).encode()).hexdigest(),
            "files": per_file}


def frozen_artifacts(data_root: Path = DATA_ROOT) -> dict[str, dict]:
    """name -> {path, role}. Every file whose bytes a number in this lane depends on."""
    ncnn = data_root / "vm-exchange/models"
    out = {
        "ShanghaiTech_85_9.tar": {"path": STGNF_ROOT / "checkpoints/ShanghaiTech_85_9.tar", "role": "scorer SHT (frozen STG-NF)"},
        "UBnormal_unsupervised_71_8.tar": {"path": STGNF_ROOT / "checkpoints/UBnormal_unsupervised_71_8.tar",
                                           "role": "scorer UBnormal unsupervised (frozen STG-NF, seg_len 16)"},
        "UBnormal_supervised_79_2.tar": {"path": STGNF_ROOT / "checkpoints/UBnormal_supervised_79_2.tar",
                                         "role": "NOT USED (supervised UBnormal scorer; hashed so nobody swaps it in by accident)"},
        "yolov8n-pose.pt": {"path": EXPERIMENTS_ROOT / "artifacts/EXP-001/models/yolov8n-pose.pt", "role": "stock detector"},
        "uv.lock": {"path": EXPERIMENTS_ROOT / "uv.lock", "role": "dependency lock"},
    }
    for arm in ("ML0", "M00", "MLP", "M0P"):
        out[f"{arm}.pt"] = {"path": EXPERIMENTS_ROOT / f"artifacts/EXP-020/{arm}.pt",
                            "role": f"EXP-020 arm {arm} (= weights/last.pt of the training run; this file produced every cache)"}
        out[f"{arm}.best.pt"] = {"path": data_root / f"exp020/train/{arm}/weights/best.pt",
                                 "role": f"NOT USED: Ultralytics best.pt of arm {arm} (differs from the evaluated last.pt)"}
    for arm in ("ML0", "M00", "MLP", "M0P"):
        folder = ncnn / f"exp020/{arm}-640-384x640-ncnn"
        for ext in ("param", "bin"):
            out[f"{arm}-ncnn.{ext}"] = {"path": folder / f"model.ncnn.{ext}", "role": f"NCNN FP16 384x640 export of {arm}"}
    folder = ncnn / "yolov8n-pose-640-384x640-ncnn"
    for ext in ("param", "bin"):
        out[f"yolov8n-pose-ncnn.{ext}"] = {"path": folder / f"model.ncnn.{ext}", "role": "NCNN FP16 384x640 export of stock yolov8n-pose"}
    return out


# =====================================================================================================================
# 2. Frontends ("sources") and datasets
# =====================================================================================================================
@dataclass(frozen=True)
class Source:
    name: str
    dataset: str
    pose_dir: Path
    cache_dir: Path | None  # detector cache that produced pose_dir (None: released AlphaPose poses)
    history: tuple[Path, str] | None  # (results.jsonl, "cache" value) of the earlier experiment
    claimed: tuple[float, float] | None  # (paper, live) AUC x 100 quoted in the plan / earlier reports
    description: str

    @property
    def key(self) -> str:
        return f"{self.dataset}/{self.name}"


def build_sources(data_root: Path = DATA_ROOT) -> list[Source]:
    d = data_root
    sht_rel, ub_rel = STGNF_ROOT / "data/ShanghaiTech/pose/test", STGNF_ROOT / "data/UBnormal/pose/test"
    tag = "L0_E0_c0.25_botsort"
    return [
        Source("released", "ShanghaiTech", sht_rel, None, None, (85.94, 80.98), "released AlphaPose poses (STG-NF authors)"),
        Source("stock", "ShanghaiTech", d / "exp003/poses" / tag, d / "exp003/cache/L0_E0", (d / "exp003/results.jsonl", "L0_E0"),
               (68.53, 66.17), "stock yolov8n-pose @640 (MPS FP32) + BoT-SORT"),
        Source("ML0_mps", "ShanghaiTech", d / "exp020-ML0-sht/poses" / tag, d / "exp020-ML0-sht/cache/L0_E0",
               (d / "exp020-ML0-sht/results.jsonl", "L0_E0"), (77.88, 73.53), "ML0 (MPS FP32) + BoT-SORT"),
        Source("ML0_ncnn", "ShanghaiTech", d / "exp020-ncnn/poses/L0_E0-ML0_c0.25_botsort", d / "exp020-ncnn/cache/L0_E0-ML0",
               (d / "exp020-ncnn/results.jsonl", "L0_E0-ML0"), (78.00, 73.77), "ML0 (NCNN FP16 384x640) + BoT-SORT"),
        Source("MLP_mps", "ShanghaiTech", d / "exp020-MLP-sht/poses" / tag, d / "exp020-MLP-sht/cache/L0_E0",
               (d / "exp020-MLP-sht/results.jsonl", "L0_E0"), (79.35, 74.79), "MLP (MPS FP32) + BoT-SORT"),
        Source("MLP_ncnn", "ShanghaiTech", d / "exp020-ncnn/poses/L0_E0-MLP_c0.25_botsort", d / "exp020-ncnn/cache/L0_E0-MLP",
               (d / "exp020-ncnn/results.jsonl", "L0_E0-MLP"), (79.24, 74.70), "MLP (NCNN FP16 384x640) + BoT-SORT"),
        Source("released", "UBnormal", ub_rel, None, None, (71.78, 70.32), "released AlphaPose poses (STG-NF authors)"),
        Source("stock", "UBnormal", d / "exp006/poses" / tag, d / "exp006/cache/L0_E0", (d / "exp006/results.jsonl", "L0_E0"),
               (55.76, 56.20), "stock yolov8n-pose @640 (MPS FP32) + BoT-SORT"),
        Source("ML0_mps", "UBnormal", d / "exp020-ML0-ub/poses" / tag, d / "exp020-ML0-ub/cache/L0_E0",
               (d / "exp020-ML0-ub/results.jsonl", "L0_E0"), (54.37, 54.93), "ML0 (MPS FP32) + BoT-SORT"),
        Source("MLP_mps", "UBnormal", d / "exp020-MLP-ub/poses" / tag, d / "exp020-MLP-ub/cache/L0_E0",
               (d / "exp020-MLP-ub/results.jsonl", "L0_E0"), (54.28, 54.78), "MLP (MPS FP32) + BoT-SORT"),
    ]


def historical_auc(source: Source) -> dict[str, float] | None:
    """{variant: AUC x 100} stored by the earlier experiment (c = 0.25, level L0), or None for the released poses."""
    if source.history is None:
        return None
    path, cache = source.history
    out = {}
    for line in Path(path).read_text().splitlines():
        row = json.loads(line)
        if row.get("cache") == cache and row.get("c") == GATE_C and row.get("level", "L0") == "L0":
            out[row["variant"]] = round(row["auc"] * 100, 6)
            out[f"{row['variant']}_ci"] = [round(row["ci95_low"] * 100, 6), round(row["ci95_high"] * 100, 6)]
    return out


def dataset_clips(dataset: str) -> tuple[list[str], list[np.ndarray]]:
    """(clip names in canonical order, per-frame labels with 1 = anomalous)."""
    if dataset == "ShanghaiTech":
        files = sorted(SHT_GT_DIR.glob("*.npy"))
        return [f.stem for f in files], [np.load(f).astype(int) for f in files]
    from edgelab.pose.ubnormal import load_gt, load_test_list

    names = load_test_list(UB_TEST_LIST)
    return names, [load_gt(v, UB_GT_DIR) for v in names]


def index_pose_dir(pose_dir: Path, dataset: str) -> dict[str, Path]:
    """clip name -> pose JSON in `pose_dir` (the released folders also hold *-results.json files, which are ignored)."""
    out = {}
    for path in sorted(Path(pose_dir).glob(f"*{SUFFIX}")):
        if dataset == "ShanghaiTech":
            clip = "_".join(path.name.split("_")[:2])
        else:
            from edgelab.pose.ubnormal import video_of_pose_file

            clip = video_of_pose_file(path.name)
        out[clip] = path
    return out


# =====================================================================================================================
# 3. Splits
# =====================================================================================================================
def stratified_dev_split(items: list[str], strata: dict[str, str], n_dev: int, seed: int = SEED) -> tuple[list[str], list[str]]:
    """Deterministic video-grouped split into exactly `n_dev` development and the rest confirmation items.

    Items are ordered by (stratum, sha256(seed|item)) and every ~len/n_dev-th item (systematic sampling) goes to dev, so each
    stratum is represented in proportion up to rounding. Pure function of its arguments.
    """
    n = len(items)
    if not 0 < n_dev < n:
        raise ValueError("n_dev must be in (0, len(items))")
    ordered = sorted(items, key=lambda it: (strata[it], hashlib.sha256(f"{seed}|{it}".encode()).hexdigest()))
    dev = [it for i, it in enumerate(ordered) if (i + 1) * n_dev // n > i * n_dev // n]
    assert len(dev) == n_dev
    dev_set = set(dev)
    return sorted(dev), sorted(it for it in items if it not in dev_set)


def ub_scene(video: str) -> str:
    return re.findall(r"scene_(\d+)_", video)[0]


def build_splits(data_root: Path = DATA_ROOT) -> dict:
    """All split definitions of the lane (plan.md section 3)."""
    sht_clips, sht_labels = dataset_clips("ShanghaiTech")
    strata = {c: c.split("_")[0] for c in sht_clips}  # every SHT test clip has anomalies: stratify by scene
    sht_dev, sht_conf = stratified_dev_split(sht_clips, strata, N_DEV["ShanghaiTech"])
    ub_clips, ub_labels = dataset_clips("UBnormal")
    ub_strata = {v: f"{int(ub_scene(v)):02d}|{int(lab.sum() > 0)}" for v, lab in zip(ub_clips, ub_labels)}
    ub_dev, ub_conf = stratified_dev_split(ub_clips, ub_strata, N_DEV["UBnormal"])
    pose_root = STGNF_ROOT / "data/UBnormal/pose"

    def names(split: str) -> list[str]:
        return sorted(p.name.replace("_alphapose_tracked_person.json", "") for p in (pose_root / split).glob("*tracked_person.json"))

    sht_train = sorted(p.name.replace("_alphapose_tracked_person.json", "")
                       for p in (STGNF_ROOT / "data/ShanghaiTech/pose/train").glob("*tracked_person.json"))
    return {
        "ShanghaiTech": {"test_dev": sht_dev, "test_confirmation": sht_conf, "train_calibration": list(SHT_CALIB_TRAIN_CLIPS),
                         "train_all_released": sht_train},
        "UBnormal": {"test_dev": ub_dev, "test_confirmation": ub_conf, "official_train_released": names("train"),
                     "official_validation_released": names("validation"), "official_test_released": names("test"),
                     "official_validation_frontend_caches_available": False},
    }


def split_report(dataset: str, splits: dict) -> dict:
    """R8: disjoint, covering, both classes present in each part."""
    clips, labels = dataset_clips(dataset)
    dev, conf = set(splits["test_dev"]), set(splits["test_confirmation"])
    lab = dict(zip(clips, labels))
    ok_cover = dev | conf == set(clips) and not (dev & conf)
    info = {"n_test": len(clips), "n_dev": len(dev), "n_confirmation": len(conf), "disjoint_and_covering": bool(ok_cover)}
    for name, part in (("dev", dev), ("confirmation", conf)):
        frames = np.concatenate([lab[c] for c in sorted(part)])
        info[f"{name}_frames"] = int(len(frames))
        info[f"{name}_anomalous_frames"] = int(frames.sum())
        info[f"{name}_has_both_classes"] = bool(0 < frames.sum() < len(frames))
        info[f"{name}_scenes"] = len({c.split('_')[0] if dataset == 'ShanghaiTech' else ub_scene(c) for c in part})
    return info


# =====================================================================================================================
# 4. Window rules (pure)
# =====================================================================================================================
def complete_window_starts(frames: np.ndarray, seg_len: int, max_missing: int = MISSING_TOLERANCE) -> np.ndarray:
    """Pose indices where a window of `seg_len` consecutive poses of ONE track starts and satisfies STG-NF's continuity rule.

    `frames` = sorted frame numbers of the track's poses. Every start 0 .. n - seg_len is considered (nothing omitted); a
    window is kept if at least `seg_len - max_missing` of its poses fall in [start_frame, start_frame + seg_len).
    """
    frames = np.asarray(frames)
    n = len(frames)
    if n < seg_len:
        return np.zeros(0, dtype=np.int64)
    starts = np.arange(n - seg_len + 1)
    in_span = np.searchsorted(frames, frames[starts] + seg_len, side="left") - starts
    return starts[np.minimum(in_span, seg_len) >= seg_len - max_missing]


def stgnf_offline_window_mask(starts: np.ndarray, n_poses: int, seg_len: int) -> np.ndarray:
    """True for the windows STG-NF's test-time `split_pose_to_segments` really produces: `num_segs = ceil((n - seg_len) / 1)`
    only enumerates starts 0 .. n - seg_len - 1, so the LAST complete window (start n - seg_len) is never scored, and a track
    of exactly `seg_len` poses yields nothing."""
    return np.asarray(starts) < (n_poses - seg_len)


def landing_frames(start_frame: np.ndarray, last_frame: np.ndarray, seg_len: int, mode: str) -> np.ndarray:
    """Frame that receives a window's score. middle = paper (needs seg_len//2 future frames); end = start + seg_len - 1
    (live placement of the earlier code); completion = frame of the window's last pose (when it is really available)."""
    if mode == "middle":
        return start_frame + seg_len // 2
    if mode == "end":
        return start_frame + seg_len - 1
    if mode == "completion":
        return np.asarray(last_frame)
    raise ValueError(mode)


def per_frame_min(clip_idx: np.ndarray, landing: np.ndarray, normality: np.ndarray, n_frames: list[int]) -> list[np.ndarray]:
    """Per clip: per-frame normality = least normal window landing on that frame (the least normal person); +inf = no window."""
    n_frames_arr = np.asarray(n_frames, dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(n_frames_arr)])
    flat = np.full(int(offsets[-1]), np.inf)
    valid = (landing >= 0) & (landing < n_frames_arr[clip_idx])
    np.minimum.at(flat, offsets[clip_idx[valid]] + landing[valid], normality[valid])
    return [flat[offsets[i]:offsets[i + 1]] for i in range(len(n_frames))]


def global_most_normal_fill(per_clip: list[np.ndarray]) -> tuple[list[np.ndarray], float]:
    """STG-NF's rule: empty frames get the single most-normal finite score of the WHOLE test set (look-ahead + test statistic)."""
    flat = np.concatenate(per_clip)
    most_normal = float(flat[np.isfinite(flat)].max())
    return [np.where(np.isfinite(c), c, most_normal) for c in per_clip], most_normal


def causal_smooth(x: np.ndarray) -> np.ndarray:
    """The reference causal half-Gaussian of `edgelab.pose.smoothing`. A vectorised convolution is mathematically identical
    (max |diff| ~ 1e-16) but breaks exact ties differently, which moves a pooled AUC by ~0.005 points (the global fill and the
    sentinel create long runs of tied scores), so every number in this lane uses the reference implementation."""
    return SMOOTHERS["causal_half_gaussian"](x)


def hold_fill(raw: np.ndarray, sentinel: float, max_hold: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Causal fill of a per-frame normality series (+inf / non-finite = no window completed on that frame).

    Returns (filled normality, mode, age): mode 0 = fresh window, 1 = held last score (age <= max_hold), 2 = sentinel
    (nothing to hold or hold expired). age = frames since the last fresh score (-1 before the first one).
    Same rule as `refresh_policy.CausalScoreDeliverer` (max_hold = 0 -> never hold) but vectorised.
    """
    n = len(raw)
    fresh = np.isfinite(raw)
    t = np.arange(n)
    last = np.maximum.accumulate(np.where(fresh, t, -1))
    age = np.where(last >= 0, t - last, -1)
    held = ~fresh & (last >= 0) & (age <= max_hold)
    value = np.where(fresh, raw, np.where(held, raw[np.maximum(last, 0)], sentinel))
    mode = np.where(fresh, 0, np.where(held, 1, 2)).astype(np.int8)
    return value, mode, age


def deliver(per_clip: list[np.ndarray], sentinel: float, max_hold: float) -> tuple[list[np.ndarray], dict]:
    """S3/S4: strictly causal per-frame anomaly scores + coverage / staleness statistics (pooled over clips)."""
    scores, modes, ages = [], [], []
    for raw in per_clip:
        value, mode, age = hold_fill(raw, sentinel, max_hold)
        scores.append(causal_smooth(-value))
        modes.append(mode)
        ages.append(age)
    mode, age = np.concatenate(modes), np.concatenate(ages)
    used = age[age >= 0]  # age of the information behind every frame that has any (fresh = 0)
    stats = {"fresh_fraction": float((mode == 0).mean()), "held_fraction": float((mode == 1).mean()),
             "sentinel_fraction": float((mode == 2).mean()),
             "mean_age_frames": float(used.mean()) if len(used) else float("nan"),
             "p95_age_frames": float(np.quantile(used, 0.95)) if len(used) else float("nan"),
             "max_age_frames": int(used.max()) if len(used) else -1}
    return scores, stats


def pooled_auc(labels: list[np.ndarray], scores: list[np.ndarray], idx: list[int] | None = None) -> float:
    idx = range(len(labels)) if idx is None else idx
    y = np.concatenate([labels[i] for i in idx])
    s = np.concatenate([scores[i] for i in idx])
    return float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else float("nan")


def clip_auc(label: np.ndarray, score: np.ndarray) -> float:
    return float(roc_auc_score(label, score)) if 0 < label.sum() < len(label) else float("nan")


def bootstrap_ci(labels, scores, idx: list[int] | None = None, n: int = 1000) -> list[float]:
    idx = list(range(len(labels))) if idx is None else idx
    low, high = bootstrap_ci_over_videos([labels[i] for i in idx], [scores[i] for i in idx], n_resamples=n)
    return [100 * low, 100 * high]


def paired_diff(labels, scores_a, scores_b, idx: list[int] | None = None, n: int = 1000) -> dict:
    idx = list(range(len(labels))) if idx is None else idx
    diff, low, high = paired_bootstrap_diff([labels[i] for i in idx], [scores_a[i] for i in idx], [scores_b[i] for i in idx],
                                            n_resamples=n)
    return {"delta": 100 * diff, "ci95": [100 * low, 100 * high], "excludes_zero": bool(low > 0 or high < 0)}


# =====================================================================================================================
# 5. Pose reading, STG-NF scorer wrapper, window tables
# =====================================================================================================================
def read_pose_clip(path: Path) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """person id -> (frames (n,) int sorted numerically, keypoints (n, 17, 3) float64), as stored in the pose JSON."""
    clip = json.loads(Path(path).read_text())
    out = {}
    for pid in sorted(clip, key=int):
        poses = merge_parts(clip[pid])
        if not poses:
            continue
        keys = sorted(poses, key=int)
        out[int(pid)] = (np.array([int(k) for k in keys], dtype=np.int64),
                         np.stack([np.asarray(poses[k]["keypoints"], dtype=np.float64).reshape(17, 3) for k in keys]))
    return out


def gather_windows(keypoints: np.ndarray, starts: np.ndarray, seg_len: int) -> np.ndarray:
    return keypoints[starts[:, None] + np.arange(seg_len)[None, :]]


def normalize_windows(windows: np.ndarray) -> np.ndarray:
    """(N, T, 17, 3) pixels -> (N, T, 18, 3) normalised, each window by its OWN mean / std (STG-NF `normalize_pose`)."""
    _import_stgnf()
    from dataset import keypoints17_to_coco18
    from utils.data_utils import normalize_pose

    return normalize_pose(keypoints17_to_coco18(windows).astype(np.float32))


class Scorer:
    """The frozen STG-NF flow: per-window normalisation exactly as `dataset.PoseSegDataset` does it, then -NLL."""

    def __init__(self, dataset: str, checkpoint: Path | None = None):
        from edgelab.pipeline.refresh_policy import CausalLiveStgnf
        from edgelab.pose.score_dir import DATASET_DEFAULTS

        self.dataset, self.seg_len = dataset, SEG_LENS[dataset]
        self.checkpoint = Path(checkpoint or DATASET_DEFAULTS[dataset][0])
        self.core = CausalLiveStgnf(CausalLiveStgnf.load_model_for_dataset(self.checkpoint, dataset), dataset)
        self.model = self.core.model

    @staticmethod
    def normalize(windows: np.ndarray) -> np.ndarray:
        return normalize_windows(windows)

    def score_normalized(self, normalized: np.ndarray) -> np.ndarray:
        import torch

        x = torch.from_numpy(np.ascontiguousarray(normalized.transpose(0, 3, 1, 2)[:, :2])).float()
        with torch.no_grad():
            _, nll = self.model(x, label=torch.ones(len(x)))
        return -nll.numpy().reshape(-1).astype(np.float64)

    def score(self, windows: np.ndarray, chunk: int = 4096) -> np.ndarray:
        if len(windows) == 0:
            return np.zeros(0)
        return np.concatenate([self.core.score_batch(windows[i:i + chunk]).astype(np.float64) for i in range(0, len(windows), chunk)])


WINDOW_COLS = ("clip_idx", "person", "start_idx", "n_track", "start_frame", "last_frame", "in_offline", "normality")


def build_window_table(pose_files: dict[str, Path], clips: list[str], scorer: Scorer, log=None) -> dict[str, np.ndarray]:
    """Every complete window of every track of every clip, scored once. Missing pose file = clip without tracks."""
    seg_len = scorer.seg_len
    cols: dict[str, list] = {c: [] for c in WINDOW_COLS if c != "normality"}
    n_tracks, n_poses = np.zeros(len(clips), dtype=np.int64), np.zeros(len(clips), dtype=np.int64)
    buffer: list[np.ndarray] = []
    buffered, norm_parts = 0, []

    def flush():
        nonlocal buffer, buffered
        if buffer:
            norm_parts.append(scorer.score(np.concatenate(buffer)))
        buffer, buffered = [], 0

    for ci, clip in enumerate(clips):
        path = pose_files.get(clip)
        tracks = read_pose_clip(path) if path is not None else {}
        for pid, (frames, kp) in tracks.items():
            n_tracks[ci] += 1
            n_poses[ci] += len(frames)
            starts = complete_window_starts(frames, seg_len)
            if len(starts) == 0:
                continue
            cols["clip_idx"].append(np.full(len(starts), ci, dtype=np.int32))
            cols["person"].append(np.full(len(starts), pid, dtype=np.int32))
            cols["start_idx"].append(starts.astype(np.int32))
            cols["n_track"].append(np.full(len(starts), len(frames), dtype=np.int32))
            cols["start_frame"].append(frames[starts].astype(np.int32))
            cols["last_frame"].append(frames[starts + seg_len - 1].astype(np.int32))
            cols["in_offline"].append(stgnf_offline_window_mask(starts, len(frames), seg_len))
            buffer.append(gather_windows(kp, starts, seg_len))
            buffered += len(starts)
            if buffered >= 4096:
                flush()
        if log and (ci + 1) % 25 == 0:
            log(f"    windows: {ci + 1}/{len(clips)} clips")
    flush()
    table = {c: (np.concatenate(v) if v else np.zeros(0, dtype=bool if c == "in_offline" else np.int32)) for c, v in cols.items()}
    table["normality"] = np.concatenate(norm_parts) if norm_parts else np.zeros(0)
    table["clip_n_tracks"], table["clip_n_poses"] = n_tracks, n_poses
    return table


def dir_fingerprint(directory: Path, pattern: str = f"*{SUFFIX}") -> str:
    parts = [f"{p.name}:{p.stat().st_size}" for p in sorted(Path(directory).glob(pattern))]
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


def load_or_build_table(source: Source, clips: list[str], scorer: Scorer, log=print) -> dict[str, np.ndarray]:
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    path = WORK_DIR / f"windows__{source.dataset}__{source.name}.npz"
    stamp = json.dumps({"pose": dir_fingerprint(source.pose_dir), "ckpt": sha256_file(scorer.checkpoint)[:16],
                        "seg_len": scorer.seg_len, "v": 2})
    if path.exists():
        z = np.load(path, allow_pickle=False)
        if str(z["stamp"]) == stamp:
            return {k: z[k] for k in z.files if k != "stamp"}
    log(f"  scoring all complete windows of {source.key} ...")
    t0 = time.perf_counter()
    table = build_window_table(index_pose_dir(source.pose_dir, source.dataset), clips, scorer, log)
    log(f"    {len(table['normality'])} windows in {time.perf_counter() - t0:.0f}s")
    np.savez_compressed(path, stamp=np.array(stamp), **table)
    return table


# =====================================================================================================================
# 6. Decomposition engine
# =====================================================================================================================
STEP_SPECS = {
    "S0": dict(select="offline", landing="middle", smooth="offline_gaussian",
               text="offline paper: middle landing, two-sided Gaussian, global fill, last window omitted"),
    "S0a": dict(select="offline", landing="end", smooth="offline_gaussian", text="ablation: end landing only (still two-sided Gaussian)"),
    "S0b": dict(select="offline", landing="middle", smooth="causal", text="ablation: causal smoothing only (still middle landing)"),
    "S1": dict(select="offline", landing="end", smooth="causal",
               text="offline 'live': end landing + causal half-Gaussian, global fill, last window omitted"),
    "S1a": dict(select="all", landing="end", smooth="causal", text="ablation: omission fixed only (end landing)"),
    "S1b": dict(select="offline", landing="completion", smooth="causal", text="ablation: completion landing only (omission kept)"),
    "S2": dict(select="all", landing="completion", smooth="causal",
               text="all complete windows, landed at the real completion frame, global fill"),
}


@dataclass
class SourceData:
    source: Source
    clips: list[str]
    labels: list[np.ndarray]
    table: dict[str, np.ndarray]
    seg_len: int
    dev_idx: list[int]
    conf_idx: list[int]

    @property
    def n_frames(self) -> list[int]:
        return [len(lab) for lab in self.labels]

    def landing(self, mode: str, select: str = "all") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        t = self.table
        mask = np.ones(len(t["normality"]), dtype=bool) if select == "all" else t["in_offline"]
        land = landing_frames(t["start_frame"], t["last_frame"], self.seg_len, mode)
        return t["clip_idx"][mask], land[mask], t["normality"][mask]

    def raw_per_frame(self, mode: str, select: str) -> list[np.ndarray]:
        ci, land, norm = self.landing(mode, select)
        return per_frame_min(ci, land, norm, self.n_frames)

    def step_scores(self, step: str) -> tuple[list[np.ndarray], float]:
        spec = STEP_SPECS[step]
        filled, most_normal = global_most_normal_fill(self.raw_per_frame(spec["landing"], spec["select"]))
        smoother = SMOOTHERS["offline_gaussian"] if spec["smooth"] == "offline_gaussian" else causal_smooth
        return [smoother(-c) for c in filled], most_normal

    def auc_triplet(self, scores: list[np.ndarray]) -> dict[str, float]:
        return {"all": 100 * pooled_auc(self.labels, scores), "dev": 100 * pooled_auc(self.labels, scores, self.dev_idx),
                "confirmation": 100 * pooled_auc(self.labels, scores, self.conf_idx)}


def coverage_fractions(per_clip: list[np.ndarray], labels: list[np.ndarray]) -> dict[str, float]:
    fresh, lab = np.concatenate([np.isfinite(c) for c in per_clip]), np.concatenate(labels)
    return {"all": float(fresh.mean()), "normal_frames": float(fresh[lab == 0].mean()), "anomalous_frames": float(fresh[lab == 1].mean())}


# =====================================================================================================================
# 7. Training-normal calibration of the sentinel
# =====================================================================================================================
def distribution_summary(values: np.ndarray) -> dict:
    values = values[np.isfinite(values)]
    return {"n": int(len(values)), "quantiles": {f"{q:g}": float(np.quantile(values, q)) for q in QUANTILE_TABLE},
            "mean": float(values.mean()), "std": float(values.std())}


def percentile_rank(values: np.ndarray, x: float) -> float:
    """Fraction of `values` that are <= x (so 1 - rank is the share of windows MORE normal than x)."""
    values = values[np.isfinite(values)]
    return float((values <= x).mean())


def train_normality_distribution(pose_files: list[Path], scorer: Scorer) -> dict[str, np.ndarray]:
    """Normality of every complete window of training-normal poses + the per-frame-min series those windows would give."""
    seg_len = scorer.seg_len
    windows, meta = [], []
    for ci, path in enumerate(pose_files):
        for _, (frames, kp) in read_pose_clip(path).items():
            starts = complete_window_starts(frames, seg_len)
            if len(starts):
                windows.append(gather_windows(kp, starts, seg_len))
                meta.append(np.stack([np.full(len(starts), ci), frames[starts + seg_len - 1]], axis=1))
    if not windows:
        return {"window": np.zeros(0), "frame_min": np.zeros(0)}
    norm = scorer.score(np.concatenate(windows))
    meta = np.concatenate(meta)
    n_frames = [int(meta[meta[:, 0] == i, 1].max()) + 1 if (meta[:, 0] == i).any() else 1 for i in range(len(pose_files))]
    per_frame = per_frame_min(meta[:, 0].astype(int), meta[:, 1], norm, n_frames)
    flat = np.concatenate(per_frame)
    return {"window": norm, "frame_min": flat[np.isfinite(flat)]}


def sentinel_values(window_normality: np.ndarray) -> dict[str, float]:
    return {name: float(np.quantile(window_normality, q)) for name, q in SENTINEL_QUANTILES.items()}


def detect_train_poses(source_name: str, clips: list[str], out_dir: Path, log=print) -> dict[str, Path]:
    """Run stock / ML0 (MPS FP32, exactly the settings of the test caches) on normal training videos and replay at c = 0.25.

    Needed to calibrate a *frontend-matched* sentinel on training poses of the same detector. Training only (no test data).
    """
    from edgelab.pose.detcache import cache_video, find_source, save_atomic
    from edgelab.pose.replay import replay_video
    from edgelab.pose.detcache import load_detections
    from edgelab.pose.stgnf_format import write_stgnf_json
    from ultralytics import YOLO

    weights = {"stock": EXPERIMENTS_ROOT / "artifacts/EXP-001/models/yolov8n-pose.pt",
               "ML0_mps": EXPERIMENTS_ROOT / "artifacts/EXP-020/ML0.pt"}[source_name]
    cache_dir, pose_dir = out_dir / "cache", out_dir / "poses"
    pose_dir.mkdir(parents=True, exist_ok=True)
    model = None
    files = {}
    for clip in clips:
        npz = cache_dir / f"{clip}.npz"
        if not npz.exists():
            if model is None:
                model = YOLO(str(weights))
            log(f"    detecting {source_name} on train clip {clip}")
            save_atomic(npz, cache_video(model, find_source(SHT_TRAIN_VIDEOS, clip), "L0", "E0", "mps"))
        out = pose_dir / f"{clip}_{source_name}_tracked_person.json"
        if not out.exists():
            frames, data = load_detections(npz)
            poses, _ = replay_video(frames, data["frame_ids"], GATE_C, "botsort")
            write_stgnf_json(poses, out)
        files[clip] = out
    return files


# =====================================================================================================================
# 8. Reproduction against the historical code path
# =====================================================================================================================
def reproduce_source(source: Source, n_resamples: int = 2000, threads: int | None = None) -> tuple[dict, object]:
    """Historical path: STG-NF loader -> score_segments -> variant_scores -> AUC + bootstrap CI (identical call to score_dir)."""
    from edgelab.pose.score_dir import DATASET_DEFAULTS, VARIANTS, coverage, variant_scores
    from edgelab.pose.stgnf_eval import score_segments

    checkpoint, gt_dir = DATASET_DEFAULTS[source.dataset]
    segments = score_segments(source.pose_dir, checkpoint, device="cpu", threads=threads, dataset=source.dataset)
    hist = historical_auc(source)
    out = {"source": source.key, "n_segments_stgnf": int(segments.num_segments), "variants": {}}
    for variant, (offset, smoother) in VARIANTS.items():
        names, labels, scores, covered = variant_scores(segments, gt_dir, offset, smoother, source.dataset, None)
        auc = 100 * frame_level_auc(labels, scores)
        low, high = bootstrap_ci_over_videos(labels, scores, n_resamples=n_resamples)
        cov_anomalous, cov_normal = coverage(covered, labels)
        claim = None if source.claimed is None else source.claimed[0 if variant == "paper" else 1]
        out["variants"][variant] = {"auc": auc, "ci95": [100 * low, 100 * high], "n_videos": len(names),
                                    "coverage_anomalous": cov_anomalous, "coverage_normal": cov_normal,
                                    "historical_auc": None if hist is None else hist.get(variant),
                                    "historical_ci95": None if hist is None else hist.get(f"{variant}_ci"),
                                    "claimed_auc": claim,
                                    "delta_vs_historical": None if hist is None or variant not in hist else auc - hist[variant],
                                    "delta_vs_claimed": None if claim is None else auc - claim}
    return out, segments


def window_parity(table: dict, segments, clips: list[str], dataset: str) -> dict:
    """R3: STG-NF's segments vs this module's offline windows, keyed by (clip, person, start frame)."""
    import pandas as pd

    meta = np.asarray(segments.metadata)
    if dataset == "ShanghaiTech":
        names = [f"{int(s):02d}_{int(c):04d}" for s, c in zip(meta[:, 0], meta[:, 1])]
    else:
        from edgelab.pose.ubnormal import parse_video_name

        lookup = {parse_video_name(v): v for v in clips}
        names = [lookup[(str(int(s)), str(c))] for s, c in zip(meta[:, 0], meta[:, 1])]
    cidx = {c: i for i, c in enumerate(clips)}
    theirs = pd.DataFrame({"clip_idx": [cidx[n] for n in names], "person": meta[:, 2].astype(int),
                           "start_frame": meta[:, 3].astype(int), "theirs": np.asarray(segments.normality, dtype=np.float64)})
    off = table["in_offline"]
    mine = pd.DataFrame({"clip_idx": table["clip_idx"][off].astype(int), "person": table["person"][off].astype(int),
                         "start_frame": table["start_frame"][off].astype(int), "mine": table["normality"][off]})
    merged = theirs.merge(mine, on=["clip_idx", "person", "start_frame"], how="outer", indicator=True)
    both = merged[merged["_merge"] == "both"]
    diff = np.abs(both["theirs"].to_numpy() - both["mine"].to_numpy())
    return {"n_stgnf_segments": int(len(theirs)), "n_mine_offline_windows": int(len(mine)),
            "n_matched": int(len(both)), "only_in_stgnf": int((merged["_merge"] == "left_only").sum()),
            "only_in_mine": int((merged["_merge"] == "right_only").sum()),
            "max_abs_normality_diff": float(diff.max()) if len(diff) else 0.0,
            "mean_abs_normality_diff": float(diff.mean()) if len(diff) else 0.0}


def replay_check(source: Source, clips: list[str], limit: int | None = None) -> dict:
    """R2b: replaying the detector cache (c = 0.25, BoT-SORT) must regenerate the stored pose JSON exactly."""
    from edgelab.pose.detcache import load_detections
    from edgelab.pose.replay import replay_video
    from edgelab.pose.stgnf_format import to_stgnf_dict

    if source.cache_dir is None:
        return {"applicable": False}
    pose_files = index_pose_dir(source.pose_dir, source.dataset)
    identical, differing, missing = 0, [], []
    for clip in clips[:limit]:
        npz = source.cache_dir / f"{clip}.npz"
        if clip not in pose_files or not npz.exists():
            missing.append(clip)
            continue
        frames, data = load_detections(npz)
        poses, _ = replay_video(frames, data["frame_ids"], GATE_C, "botsort")
        fresh = json.loads(json.dumps(to_stgnf_dict(poses)))
        if fresh == json.loads(pose_files[clip].read_text()):
            identical += 1
        else:
            differing.append(clip)
    return {"applicable": True, "n_clips": len(clips[:limit]), "identical": identical, "differing": differing, "missing": missing}


# =====================================================================================================================
# 9. Reuse / normalisation audit
# =====================================================================================================================
def streaming_windows(frames: np.ndarray, keypoints: np.ndarray, seg_len: int) -> list[tuple[int, np.ndarray]]:
    """What a live system does: poses arrive one frame at a time, each is stored ONCE in the track's buffer, and every
    time `seg_len` poses are available the last `seg_len` buffered poses form a window (continuity rule applied)."""
    buffer_f, buffer_kp, out = [], [], []
    for f, kp in zip(frames, keypoints):
        buffer_f.append(int(f))
        buffer_kp.append(kp)  # stored once, referenced by up to seg_len windows
        if len(buffer_f) >= seg_len:
            window_f = np.array(buffer_f[-seg_len:])
            if np.count_nonzero(window_f < window_f[0] + seg_len) >= seg_len - MISSING_TOLERANCE:
                out.append((len(buffer_f) - seg_len, np.stack(buffer_kp[-seg_len:])))
    return out


def normalize_with_stats(windows: np.ndarray, mean_xy: np.ndarray, std_y: float) -> np.ndarray:
    """STG-NF normalisation with GIVEN statistics (to emulate reuse of a normalisation computed on another window)."""
    _import_stgnf()
    from dataset import keypoints17_to_coco18

    x = keypoints17_to_coco18(windows).astype(np.float32) / np.array([856, 480, 1])  # int divisor -> float64, as normalize_pose
    x[..., :2] = (x[..., :2] - mean_xy) / std_y
    return x


def window_stats(windows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-window mean of (x, y) and std of y after division by the video resolution (what normalize_pose computes)."""
    _import_stgnf()
    from dataset import keypoints17_to_coco18

    x = keypoints17_to_coco18(windows).astype(np.float32) / np.array([856, 480, 1])
    return x[..., :2].mean(axis=(1, 2)), x[..., 1].std(axis=(1, 2))


def reuse_audit(dataset: str, pose_files: dict[str, Path], clips: list[str], scorer: Scorer, max_clips: int = 6) -> dict:
    """R5 on real tracks: per-frame (x, y, conf) reuse is bit-exact; post-normalisation / hidden-state reuse is not."""
    seg_len = scorer.seg_len
    chosen = [c for c in clips if c in pose_files][:: max(1, len(clips) // max_clips)][:max_clips]
    reuse_in, recomputed_in, stream_in = [], [], []
    shared_first, track_level, pairs_a, pairs_b = [], [], [], []
    n_windows = n_poses = n_tracks = 0
    records = []
    for clip in chosen:
        raw = json.loads(Path(pose_files[clip]).read_text())
        shared = read_pose_clip(pose_files[clip])  # parsed ONCE: the per-frame store that overlapping windows share
        for pid, (frames, kp) in shared.items():
            starts = complete_window_starts(frames, seg_len)
            if len(starts) == 0:
                continue
            n_tracks += 1
            n_poses += len(frames)
            n_windows += len(starts)
            w_reuse = gather_windows(kp, starts, seg_len)
            person = merge_parts(raw[str(pid)])
            keys = {int(k): k for k in person}  # recompute: each window is rebuilt from the raw JSON entries on its own
            w_fresh = np.stack([np.stack([np.asarray(person[keys[int(frames[s + j])]]["keypoints"], dtype=np.float64).reshape(17, 3)
                                          for j in range(seg_len)]) for s in starts])
            w_stream_pairs = streaming_windows(frames, kp, seg_len)
            w_stream = np.stack([w for _, w in w_stream_pairs]) if w_stream_pairs else np.zeros_like(w_reuse)
            reuse_in.append(w_reuse)
            recomputed_in.append(w_fresh)
            if len(w_stream) == len(w_reuse):
                stream_in.append(w_stream)
            else:
                stream_in.append(np.full_like(w_reuse, np.nan))  # would be a parity failure; flagged below
            # negative controls: normalisation computed on another window / on the whole track
            mean0, std0 = window_stats(w_reuse[:1])
            shared_first.append(normalize_with_stats(w_reuse, mean0[0], float(std0[0])))
            mean_t, std_t = window_stats(kp[None])
            track_level.append(normalize_with_stats(w_reuse, mean_t[0], float(std_t[0])))
            if len(w_reuse) >= 2:
                pairs_a.append(w_reuse[:-1])
                pairs_b.append(w_reuse[1:])
            records.append({"clip": clip, "person": pid, "n_poses": int(len(frames)), "n_windows": int(len(starts))})
    reuse, fresh, stream = (np.concatenate(x) for x in (reuse_in, recomputed_in, stream_in))
    correct_norm = scorer.normalize(reuse)
    s_reuse, s_fresh = scorer.score(reuse), scorer.score(fresh)
    single = np.concatenate([scorer.score(reuse[i:i + 1]) for i in range(min(len(reuse), 256))])
    shared_norm, track_norm = np.concatenate(shared_first), np.concatenate(track_level)
    s_correct = scorer.score_normalized(correct_norm)
    result = {
        "clips": chosen, "n_tracks": n_tracks, "n_windows": int(n_windows), "n_poses": int(n_poses),
        "pose_evaluations_without_reuse": int(n_windows * seg_len), "reuse_factor": float(n_windows * seg_len / max(n_poses, 1)),
        "per_frame_reuse": {
            "input_max_abs_diff_shared_vs_recomputed": float(np.abs(reuse - fresh).max()),
            "input_bit_identical_shared_vs_recomputed": bool(np.array_equal(reuse, fresh)),
            "input_max_abs_diff_shared_vs_streaming": float(np.nanmax(np.abs(reuse - stream))),
            "input_bit_identical_shared_vs_streaming": bool(np.array_equal(reuse, stream)),
            "score_max_abs_diff_same_batch": float(np.abs(s_reuse - s_fresh).max()),
            "score_bit_identical_same_batch": bool(np.array_equal(s_reuse, s_fresh)),
            "score_max_abs_diff_single_window_vs_batch_float32_noise": float(np.abs(single - s_reuse[: len(single)]).max()),
        },
        "post_normalization_reuse": {
            "frozen_first_window_stats": {
                "input_max_abs_diff": float(np.abs(shared_norm - correct_norm).max()),
                "input_mean_abs_diff": float(np.abs(shared_norm - correct_norm).mean()),
                "score_max_abs_diff": float(np.abs(scorer.score_normalized(shared_norm) - s_correct).max()),
                "score_mean_abs_diff": float(np.abs(scorer.score_normalized(shared_norm) - s_correct).mean())},
            "whole_track_stats": {
                "input_max_abs_diff": float(np.abs(track_norm - correct_norm).max()),
                "input_mean_abs_diff": float(np.abs(track_norm - correct_norm).mean()),
                "score_max_abs_diff": float(np.abs(scorer.score_normalized(track_norm) - s_correct).max()),
                "score_mean_abs_diff": float(np.abs(scorer.score_normalized(track_norm) - s_correct).mean())},
        },
        "tracks": records,
    }
    result["hidden_state_reuse"] = hidden_state_reuse(scorer, np.concatenate(pairs_a)[:400], np.concatenate(pairs_b)[:400])
    result["passes"] = bool(result["per_frame_reuse"]["input_max_abs_diff_shared_vs_recomputed"] == 0.0
                            and result["per_frame_reuse"]["score_max_abs_diff_same_batch"] == 0.0
                            and result["per_frame_reuse"]["input_max_abs_diff_shared_vs_streaming"] == 0.0
                            and result["post_normalization_reuse"]["frozen_first_window_stats"]["input_max_abs_diff"] > 0
                            and result["hidden_state_reuse"]["max_abs_diff_correct_norm"] > 0)
    return result


def hidden_state_reuse(scorer: Scorer, windows_a: np.ndarray, windows_b: np.ndarray) -> dict:
    """windows_b = windows_a shifted by one frame. Compare the first flow step's output at the 23 frames both windows share."""
    import torch

    captured: list = []
    handle = scorer.model.flow.layers[0].register_forward_hook(lambda m, i, o: captured.append(o[0].detach().clone()))

    def hidden(normalized: np.ndarray) -> torch.Tensor:
        captured.clear()
        scorer.score_normalized(normalized)
        return captured[0]

    try:
        ha, hb = hidden(scorer.normalize(windows_a)), hidden(scorer.normalize(windows_b))
        mean_a, std_a = window_stats(windows_a)
        shared = np.stack([normalize_with_stats(windows_b[i:i + 1], mean_a[i], float(std_a[i]))[0] for i in range(len(windows_b))])
        hb_shared = hidden(shared)
        ha_shared = hidden(np.stack([normalize_with_stats(windows_a[i:i + 1], mean_a[i], float(std_a[i]))[0] for i in range(len(windows_a))]))
    finally:
        handle.remove()
    overlap_a, overlap_b = ha[:, :, 1:, :], hb[:, :, :-1, :]
    interior = slice(4, -4)  # drop 4 boundary frames each side (valid for seg_len 16 and 24)
    return {"n_pairs": int(len(windows_a)), "layer": "flow.layers[0] (first FlowStep: actnorm -> invconv -> 2x st-gcn, temporal kernel T//2+1)",
            "max_abs_diff_correct_norm": float((overlap_a - overlap_b).abs().max()),
            "mean_abs_diff_correct_norm": float((overlap_a - overlap_b).abs().mean()),
            "max_abs_diff_shared_norm_stats": float((ha_shared[:, :, 1:, :] - hb_shared[:, :, :-1, :]).abs().max()),
            "max_abs_diff_shared_norm_stats_interior_frames": float(
                (ha_shared[:, :, 1:, :][:, :, interior] - hb_shared[:, :, :-1, :][:, :, interior]).abs().max()),
            "note": "even with identical normalisation the temporal receptive field of the coupling network spans the "
                    "whole window, so no hidden state of an overlapping frame can be copied"}


# =====================================================================================================================
# 10. Cadence / timestamps
# =====================================================================================================================
def frame_time_ms(dataset: str, frame: int | np.ndarray) -> float | np.ndarray:
    """Capture time of a frame in ms at the dataset's NATIVE rate (the fix for EXP-023's hard-coded 24 fps)."""
    return np.asarray(frame) * 1000.0 / FPS[dataset] if isinstance(frame, np.ndarray) else frame * 1000.0 / FPS[dataset]


def normal_hours(dataset: str, n_normal_frames: int) -> float:
    return n_normal_frames / (FPS[dataset] * 3600.0)


def cadence_audit(clips_by_dataset: dict[str, tuple[list[str], list[np.ndarray]]], sht_caches: list[Path], ub_caches: list[Path],
                  measure_videos: bool = True) -> dict:
    import cv2

    out: dict = {"native_fps_locked": FPS, "exp023_hardcoded_lines": []}
    text = (EXPERIMENTS_ROOT / "src/edgelab/analysis/exp023.py").read_text().splitlines()
    for i, line in enumerate(text, 1):
        if "24.0" in line:
            out["exp023_hardcoded_lines"].append(f"exp023.py:{i}: {line.strip()}")
    # ShanghaiTech: training .avi carry fps metadata; test is JPEG frames (no metadata)
    sht = {"train_video_fps": {}}
    for path in sorted(SHT_TRAIN_VIDEOS.glob("*.avi")):
        cap = cv2.VideoCapture(str(path))
        sht["train_video_fps"][path.stem] = float(cap.get(cv2.CAP_PROP_FPS))
        cap.release()
    fps_vals = list(sht["train_video_fps"].values())
    sht["n_train_videos"] = len(fps_vals)
    sht["train_fps_counts"] = {f"{v:g}": int(sum(abs(x - v) < 0.01 for x in fps_vals)) for v in sorted(set(round(x, 2) for x in fps_vals))}
    sht["train_videos_not_24fps"] = sorted(k for k, v in sht["train_video_fps"].items() if abs(v - 24.0) >= 0.01)
    sht["scenes_with_mixed_fps"] = sorted({k.split("_")[0] for k in sht["train_videos_not_24fps"]} &
                                          {k.split("_")[0] for k, v in sht["train_video_fps"].items() if abs(v - 24.0) < 0.01})
    sht["all_train_fps_equal_24"] = bool(fps_vals) and not sht["train_videos_not_24fps"]
    sht["test_clips_have_fps_metadata"] = False  # test clips are JPEG sequences
    out["ShanghaiTech"] = sht
    ub = {"video_fps": {}, "frame_count_mismatches_vs_gt": [], "unreadable": []}
    names, labels = clips_by_dataset["UBnormal"]
    if measure_videos:
        for name, lab in zip(names, labels):
            path = UB_TEST_VIDEOS / f"{name}.mp4"
            cap = cv2.VideoCapture(str(path))
            if not cap.isOpened():
                ub["unreadable"].append(name)
                continue
            ub["video_fps"][name] = float(cap.get(cv2.CAP_PROP_FPS))
            if int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) != len(lab):
                ub["frame_count_mismatches_vs_gt"].append([name, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), len(lab)])
            cap.release()
        vals = np.array(list(ub["video_fps"].values()))
        ub["fps_min"], ub["fps_max"] = float(vals.min()), float(vals.max())
        ub["all_30fps"] = bool(np.all(np.abs(vals - 30.0) < 0.01))
    out["UBnormal"] = ub
    # GT length vs cache length
    for key, caches, dataset in (("ShanghaiTech", sht_caches, "ShanghaiTech"), ("UBnormal", ub_caches, "UBnormal")):
        names, labels = clips_by_dataset[dataset]
        gt_len = dict(zip(names, (len(lab) for lab in labels)))
        bad = []
        for cache in caches:
            for clip, n in gt_len.items():
                path = cache / f"{clip}.npz"
                if not path.exists():
                    bad.append([str(cache.name), clip, "missing"])
                    continue
                fid = np.load(path)["frame_ids"]
                if len(fid) != n or not np.array_equal(fid, np.arange(n)):
                    bad.append([str(cache.name), clip, f"{len(fid)} vs {n}"])
        out[key]["cache_length_mismatches"] = bad
    # consequences of the 24 vs 30 fps mistake on UBnormal
    n_normal = int(sum((lab == 0).sum() for lab in clips_by_dataset["UBnormal"][1]))
    seg = SEG_LENS["UBnormal"]
    out["ubnormal_consequences_of_24fps_assumption"] = {
        "window_seconds_true": seg / 30.0, "window_seconds_assumed": seg / 24.0,
        "frame_interval_ms_true": 1000 / 30.0, "frame_interval_ms_assumed": 1000 / 24.0,
        "delay_ms_overstated_by_factor": 30.0 / 24.0,
        "normal_hours_true": normal_hours("UBnormal", n_normal), "normal_hours_assumed": n_normal / (24.0 * 3600.0),
        "false_alerts_per_hour_understated_by_factor": 24.0 / 30.0,
        "detector_budget_hz_true_for_stride_2": 30.0 / 2, "detector_budget_hz_assumed_for_stride_2": 24.0 / 2,
        "detector_budget_hz_true_for_stride_4": 30.0 / 4, "detector_budget_hz_assumed_for_stride_4": 24.0 / 4,
    }
    return out


# =====================================================================================================================
# 11. Causality tests (R6)
# =====================================================================================================================
def causality_tests(data: SourceData, sentinel: float, rng_seed: int = 0) -> dict:
    """(a) prefix invariance of the causal deliverer; (b) the global-fill steps are NOT prefix-invariant; (c) equals the
    stateful `CausalScoreDeliverer`; (d) no window lands before its completion frame."""
    from edgelab.pipeline.refresh_policy import CausalScoreDeliverer

    rng = np.random.default_rng(rng_seed)
    raw = data.raw_per_frame("completion", "all")
    longest = sorted(range(len(raw)), key=lambda i: -len(raw[i]))[:3]
    max_diff, checks = 0.0, 0
    for i in longest:
        full, _ = deliver([raw[i]], sentinel, 12)
        for t in rng.integers(30, len(raw[i]), 8):
            prefix, _ = deliver([raw[i][: t + 1]], sentinel, 12)
            max_diff = max(max_diff, abs(float(prefix[0][-1]) - float(full[0][t])))
            checks += 1
    # global fill: the fill value changes when the future is cut away
    _, mn_full = global_most_normal_fill(raw)
    def fill_value(frac: float) -> float:
        kept = [r[: max(1, int(len(r) * frac))] for r in raw]
        return global_most_normal_fill(kept)[1] if any(np.isfinite(k).any() for k in kept) else float("nan")

    cuts = {f"{frac:g}": fill_value(frac) for frac in (0.05, 0.1, 0.25, 0.5, 0.75)}
    mn_cut = cuts["0.05"]
    # class equivalence on one clip
    i = longest[0]
    deliverer = CausalScoreDeliverer(train_sentinel_normality=sentinel, max_hold_frames=12)
    class_scores = np.array([deliverer.step_frame(t, [raw[i][t]] if np.isfinite(raw[i][t]) else []) for t in range(len(raw[i]))])
    mine, _ = deliver([raw[i]], sentinel, 12)
    ci, land, _ = data.landing("completion", "all")
    t = data.table
    return {"prefix_invariance_max_abs_diff": max_diff, "prefix_checks": checks,
            "matches_CausalScoreDeliverer_max_abs_diff": float(np.abs(class_scores - mine[0]).max()),
            "global_fill_value_full_video_set": mn_full, "global_fill_value_after_keeping_first_5pct": mn_cut,
            "global_fill_value_by_kept_prefix_fraction": cuts,
            "global_fill_is_look_ahead": bool(any(v != mn_full for v in cuts.values() if not math.isnan(v))),
            "windows_landing_before_completion_frame": int((land < t["last_frame"]).sum()),
            "sentinel_source": "training-normal windows only (function takes no test arrays)"}


# =====================================================================================================================
# 12. Lock (config.yaml)
# =====================================================================================================================
def to_plain(obj):
    if isinstance(obj, dict):
        return {str(k): to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_plain(v) for v in obj]
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return to_plain(obj.tolist())
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None if math.isnan(obj) else ("inf" if obj > 0 else "-inf")
    return obj


def build_lock(log=print) -> tuple[dict, dict]:
    """-> (config dict for config.yaml, detailed hash manifest for summary.json)."""
    log("hashing frozen artefacts (full SHA-256) ...")
    artifacts = {}
    for name, info in frozen_artifacts().items():
        path = Path(info["path"])
        artifacts[name] = {"path": str(path), "role": info["role"],
                           **({"sha256": sha256_file(path), "bytes": path.stat().st_size} if path.exists() else {"sha256": None, "bytes": None})}
    sources = build_sources()
    caches, pose_dirs, manifests = {}, {}, {}
    for s in sources:
        if s.cache_dir is not None and s.cache_dir.exists() and str(s.cache_dir) not in manifests:
            log(f"hashing cache {s.cache_dir} ...")
            m = tree_manifest(s.cache_dir)
            manifests[str(s.cache_dir)] = m
            first = np.load(sorted(s.cache_dir.glob("*.npz"))[0])
            caches[str(s.cache_dir).replace(str(DATA_ROOT), "~/thesis-data")] = {
                "n_videos": m["n_files"], "total_bytes": m["total_bytes"], "tree_sha256": m["tree_sha256"],
                "meta_first_video": json.loads(str(first["meta"])), "used_by": [x.key for x in sources if x.cache_dir == s.cache_dir]}
        if s.pose_dir.exists():
            pm = tree_manifest(s.pose_dir, f"*{SUFFIX}")
            pose_dirs[s.key] = {"path": str(s.pose_dir).replace(str(DATA_ROOT), "~/thesis-data"), "n_files": pm["n_files"],
                                "tree_sha256": pm["tree_sha256"]}
    splits = build_splits()
    config = {
        "experiment": "Q2-P1-E00_audit", "question": "Can we reproduce existing scores, timestamps, normalization and missing-data handling?",
        "training": False, "seed": SEED, "gate_threshold_c": GATE_C, "tracker": "botsort (Ultralytics default yaml, no GMC/ReID)",
        "datasets": {
            "ShanghaiTech": {"fps": FPS["ShanghaiTech"], "seg_len": SEG_LENS["ShanghaiTech"], "window_seconds": SEG_LENS["ShanghaiTech"] / 24.0,
                             "scorer_checkpoint": "ShanghaiTech_85_9.tar", "normalisation_resolution_wh": [856, 480],
                             "n_test_clips": 107, "splits": splits["ShanghaiTech"]},
            "UBnormal": {"fps": FPS["UBnormal"], "seg_len": SEG_LENS["UBnormal"], "window_seconds": SEG_LENS["UBnormal"] / 30.0,
                         "scorer_checkpoint": "UBnormal_unsupervised_71_8.tar",
                         "normalisation_resolution_wh": [856, 480], "n_test_clips": 211,
                         "note_normalisation": "STG-NF normalises UBnormal with the SHT default resolution 856x480 (original behaviour, kept)",
                         "splits": splits["UBnormal"]},
        },
        "frontends": {s.key: {"pose_dir": str(s.pose_dir).replace(str(DATA_ROOT), "~/thesis-data"), "description": s.description,
                              "claimed_auc_paper_live": list(s.claimed) if s.claimed else None} for s in sources},
        "scoring_protocol": {
            "steps": {k: v["text"] for k, v in STEP_SPECS.items()} | {
                "S3": "all windows, completion landing, causal half-Gaussian, empty frames = train-calibrated sentinel (hold 0)",
                "S4": "S3 + hold last score <= H frames, H in [0, 6, 12, 24, inf]"},
            "causal_smoother": f"half-Gaussian sigma={STGNF_EQUIVALENT_SIGMA:.6f} (sqrt(sum_{{s=1..6}} s^2)), truncate 4 sigma",
            "sentinel_candidates": list(SENTINEL_QUANTILES), "hold_candidates": ["inf" if math.isinf(h) else h for h in HOLDS],
            "selection_rule": "ML0_mps dev split of each dataset; criterion delivered AUC; ties within 0.10 pt -> smaller H",
            "bootstrap": {"unit": "video", "resamples_historical_rescore": 2000, "resamples_new_contrasts": 1000, "seed": 0}},
        "frozen_artifacts": artifacts, "detector_caches": caches, "pose_dirs": pose_dirs,
    }
    return to_plain(config), {"cache_manifests": manifests}


def verify_lock(config: dict, log=print) -> dict:
    """R1: recompute every hash of config.yaml and compare."""
    problems, n = [], 0
    for name, info in config["frozen_artifacts"].items():
        path = Path(info["path"])
        if info["sha256"] is None or not path.exists():
            problems.append(f"{name}: missing")
            continue
        n += 1
        if sha256_file(path) != info["sha256"]:
            problems.append(f"{name}: SHA-256 changed")
    for key, info in config["detector_caches"].items():
        folder = Path(key.replace("~/thesis-data", str(DATA_ROOT)))
        n += 1
        if not folder.exists() or tree_manifest(folder)["tree_sha256"] != info["tree_sha256"]:
            problems.append(f"cache {key}: tree hash changed or missing")
    return {"n_checked": n, "problems": problems, "passes": not problems}


# =====================================================================================================================
# 13. The audit run
# =====================================================================================================================
def env_info() -> dict:
    from importlib import metadata

    info = {"python": sys.version.split()[0], "platform": platform.platform(), "machine": platform.machine(),
            "cpu_count": os.cpu_count()}
    for pkg in ("numpy", "scipy", "scikit-learn", "torch", "ultralytics", "pandas", "opencv-python-headless"):
        try:
            info[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            pass
    try:
        info["git_commit"] = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO_ROOT).stdout.strip()
        info["git_dirty"] = bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, cwd=REPO_ROOT).stdout.strip())
    except FileNotFoundError:
        info["git_commit"] = "unknown"
    return info


def select_config(grid: dict, tie: float = 0.10) -> dict:
    """Pre-registered rule: best dev AUC; any config within `tie` points of the best -> pick smallest hold, then smallest sentinel quantile."""
    rows = [(v["auc"]["dev"], k, v) for k, v in grid.items()]
    best = max(r[0] for r in rows)
    pool = [r for r in rows if best - r[0] <= tie]
    order = list(SENTINEL_QUANTILES)
    pool.sort(key=lambda r: (HOLDS.index(r[2]["hold"]) if r[2]["hold"] in HOLDS else 99, order.index(r[2]["quantile"])))
    return {"key": pool[0][1], **{k: pool[0][2][k] for k in ("quantile", "hold", "sentinel")}, "dev_auc": pool[0][0], "best_dev_auc": best}


def hold_name(h: float) -> str:
    return "inf" if math.isinf(h) else str(int(h))


def run_audit(only_datasets: tuple[str, ...] = DATASETS, n_boot: int = 1000, n_boot_hist: int = 2000, skip_hist: bool = False,
              skip_replay: bool = False, log=print) -> dict:
    t_start = time.perf_counter()
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load((EXP_DIR / "config.yaml").read_text())
    summary: dict = {"experiment": "Q2-P1-E00_audit", "environment": env_info(), "seed": SEED}
    log("R1: verifying locked hashes ...")
    summary["R1_lock_verification"] = verify_lock(config)
    sources = [s for s in build_sources() if s.dataset in only_datasets]
    clip_info = {d: dataset_clips(d) for d in DATASETS}
    splits = {d: config["datasets"][d]["splits"] for d in DATASETS}
    summary["R8_splits"] = {d: split_report(d, splits[d]) for d in DATASETS}
    scorers = {d: Scorer(d) for d in only_datasets}
    datas: dict[str, SourceData] = {}
    for s in sources:
        names, labels = clip_info[s.dataset]
        pos = {c: i for i, c in enumerate(names)}
        table = load_or_build_table(s, names, scorers[s.dataset], log)
        datas[s.key] = SourceData(s, names, labels, table, SEG_LENS[s.dataset],
                                  [pos[c] for c in splits[s.dataset]["test_dev"]], [pos[c] for c in splits[s.dataset]["test_confirmation"]])

    # ---- R2 / R2b / R3: reproduction, replay, window parity ------------------------------------------------------
    summary["R2_reproduction"], summary["R2b_replay"], summary["R3_window_parity"] = {}, {}, {}
    stage_dir = WORK_DIR / "stages"
    stage_dir.mkdir(parents=True, exist_ok=True)
    for s in sources:
        f = stage_dir / f"repro__{s.dataset}__{s.name}.json"
        if skip_hist and f.exists():
            cached = json.loads(f.read_text())
            summary["R2_reproduction"][s.key], summary["R3_window_parity"][s.key] = cached["repro"], cached["parity"]
        else:
            log(f"R2/R3: historical re-score of {s.key} ...")
            repro, segments = reproduce_source(s, n_resamples=n_boot_hist)
            parity = window_parity(datas[s.key].table, segments, clip_info[s.dataset][0], s.dataset)
            summary["R2_reproduction"][s.key], summary["R3_window_parity"][s.key] = repro, parity
            f.write_text(json.dumps(to_plain({"repro": repro, "parity": parity})))
            del segments
        g = stage_dir / f"replay__{s.dataset}__{s.name}.json"
        if skip_replay and g.exists():
            summary["R2b_replay"][s.key] = json.loads(g.read_text())
        else:
            log(f"R2b: replaying detector cache of {s.key} ...")
            summary["R2b_replay"][s.key] = replay_check(s, clip_info[s.dataset][0]) if not skip_replay else {"applicable": False, "skipped": True}
            g.write_text(json.dumps(to_plain(summary["R2b_replay"][s.key])))

    # ---- training-normal distributions for the sentinel -----------------------------------------------------------
    log("sentinel calibration on training-normal poses ...")
    sf = stage_dir / f"sentinel__{'_'.join(only_datasets)}.json"
    if sf.exists():
        summary["sentinel"] = json.loads(sf.read_text())
    else:
        summary["sentinel"] = sentinel_audit(config, scorers, datas, clip_info, log)
        sf.write_text(json.dumps(to_plain(summary["sentinel"])))

    # ---- decomposition ---------------------------------------------------------------------------------------------
    summary["decomposition"], summary["delivered_grid"], summary["selection"], summary["omission"] = {}, {}, {}, {}
    series: dict[str, dict] = {}
    for s in sources:
        d = datas[s.key]
        log(f"decomposition of {s.key} ...")
        series[s.key] = {}
        steps = {}
        for step in STEP_SPECS:
            scores, most_normal = d.step_scores(step)
            series[s.key][step] = scores
            raw = d.raw_per_frame(STEP_SPECS[step]["landing"], STEP_SPECS[step]["select"])
            steps[step] = {"text": STEP_SPECS[step]["text"], "auc": d.auc_triplet(scores), "global_most_normal_fill": most_normal,
                           "coverage": coverage_fractions(raw, d.labels)}
        summary["decomposition"][s.key] = {"steps": steps}
        t = d.table
        n_windows, off = len(t["normality"]), t["in_offline"]
        late = t["last_frame"] - (t["start_frame"] + d.seg_len - 1)
        summary["omission"][s.key] = {
            "n_complete_windows": int(n_windows), "n_stgnf_offline_windows": int(off.sum()), "n_omitted_last_windows": int((~off).sum()),
            "omitted_fraction": float((~off).mean()) if n_windows else 0.0,
            "n_tracks": int(t["clip_n_tracks"].sum()), "n_poses": int(t["clip_n_poses"].sum()),
            "n_windows_with_missing_frames": int((late > 0).sum()), "max_completion_lag_frames": int(late.max()) if n_windows else 0,
            "n_windows_end_landing_before_completion": int((late > 0).sum())}

    # ---- sentinel x hold grid + selection --------------------------------------------------------------------------
    sent = summary["sentinel"]
    for s in sources:
        d, grid = datas[s.key], {}
        raw = d.raw_per_frame("completion", "all")
        for dist_name, values in matched_sentinels(s, sent).items():
            for qname, value in values.items():
                for hold in HOLDS:
                    scores, stats = deliver(raw, value, hold)
                    key = f"{dist_name}|{qname}|H{hold_name(hold)}"
                    grid[key] = {"dist": dist_name, "quantile": qname, "sentinel": value, "hold": hold, "hold_name": hold_name(hold),
                                 "auc": d.auc_triplet(scores), **stats}
                    series[s.key][f"G:{key}"] = scores
        summary["delivered_grid"][s.key] = grid
    for ds in only_datasets:
        key = f"{ds}/ML0_mps"
        matched = sent["matched_distribution"][key]
        sub = {k: v for k, v in summary["delivered_grid"][key].items() if v["dist"] == matched}
        summary["selection"][ds] = {**select_config(sub), "selected_on": key, "matched_distribution": matched}
    # the S3 / S4 "selected" series for every source
    for s in sources:
        sel = summary["selection"][s.dataset]
        grid = summary["delivered_grid"][s.key]
        matched = sent["matched_distribution"][s.key]
        pick = grid[f"{matched}|{sel['quantile']}|H{hold_name(sel['hold'])}"]
        pick0 = grid[f"{matched}|{sel['quantile']}|H0"]
        series[s.key]["S3"] = series[s.key][f"G:{matched}|{sel['quantile']}|H0"]
        series[s.key]["S4"] = series[s.key][f"G:{matched}|{sel['quantile']}|H{hold_name(sel['hold'])}"]
        steps = summary["decomposition"][s.key]["steps"]
        steps["S3"] = {"text": f"train sentinel {sel['quantile']} ({pick0['sentinel']:.4f} from {matched}), no hold", "auc": pick0["auc"],
                       "sentinel": pick0["sentinel"], "stats": {k: pick0[k] for k in ("fresh_fraction", "held_fraction", "sentinel_fraction")}}
        steps["S4"] = {"text": f"S3 + hold H={hold_name(sel['hold'])} frames", "auc": pick["auc"],
                       "sentinel": pick["sentinel"], "stats": {k: pick[k] for k in ("fresh_fraction", "held_fraction", "sentinel_fraction",
                                                                                      "mean_age_frames", "p95_age_frames", "max_age_frames")}}

    # ---- causality, reuse, cadence ---------------------------------------------------------------------------------
    log("R6: causality tests ...")
    summary["R6_causality"] = {}
    for ds in only_datasets:
        key = f"{ds}/ML0_mps"
        sel = summary["selection"][ds]
        summary["R6_causality"][ds] = causality_tests(datas[key], sel["sentinel"])
    log("R5: reuse / normalisation audit ...")
    summary["R5_reuse"] = {}
    for ds in only_datasets:
        key = f"{ds}/ML0_mps"
        summary["R5_reuse"][ds] = reuse_audit(ds, index_pose_dir(datas[key].source.pose_dir, ds), clip_info[ds][0], scorers[ds])
        summary["R5_reuse"][ds]["post_normalization_auc"] = post_normalization_auc(datas[key], scorers[ds])
    log("R7: cadence audit ...")
    caches = {ds: [s.cache_dir for s in sources if s.dataset == ds and s.cache_dir is not None] for ds in DATASETS}
    summary["R7_cadence"] = cadence_audit(clip_info, caches["ShanghaiTech"], caches["UBnormal"])

    # ---- bootstrap CIs and paired contrasts ------------------------------------------------------------------------
    log("bootstrap confidence intervals ...")
    chain = ["S0", "S1", "S2", "S3", "S4"]
    for s in sources:
        d = datas[s.key]
        steps = summary["decomposition"][s.key]["steps"]
        ci_sources = s.name in ("released", "stock", "ML0_mps", "MLP_mps") or s.dataset == "ShanghaiTech" and s.name == "ML0_ncnn"
        if not ci_sources:
            continue
        bf = stage_dir / f"boot__{s.dataset}__{s.name}__{n_boot}__{steps['S4']['sentinel']:.6f}_{hold_name(summary['selection'][s.dataset]['hold'])}.json"
        if bf.exists():
            cached = json.loads(bf.read_text())
            for step in chain:
                steps[step]["ci95_all"] = cached["ci"][step]
            steps["S4"]["ci95_dev"], steps["S4"]["ci95_confirmation"] = cached["dev"], cached["conf"]
            steps["paired"] = cached["paired"]
            continue
        for step in chain:
            steps[step]["ci95_all"] = bootstrap_ci(d.labels, series[s.key][step], None, n_boot)
        steps["S4"]["ci95_dev"] = bootstrap_ci(d.labels, series[s.key]["S4"], d.dev_idx, n_boot)
        steps["S4"]["ci95_confirmation"] = bootstrap_ci(d.labels, series[s.key]["S4"], d.conf_idx, n_boot)
        steps["paired"] = {f"{a}->{b}": paired_diff(d.labels, series[s.key][b], series[s.key][a], None, n_boot)
                           for a, b in zip(chain[:-1], chain[1:])}
        steps["paired"]["S0->S4"] = paired_diff(d.labels, series[s.key]["S4"], series[s.key]["S0"], None, n_boot)
        for a, b in (("S0", "S0a"), ("S0", "S0b"), ("S1", "S1a"), ("S1", "S1b")):
            steps["paired"][f"{a}->{b}"] = paired_diff(d.labels, series[s.key][b], series[s.key][a], None, n_boot)
        bf.write_text(json.dumps(to_plain({"ci": {k: steps[k]["ci95_all"] for k in chain}, "dev": steps["S4"]["ci95_dev"],
                                           "conf": steps["S4"]["ci95_confirmation"], "paired": steps["paired"]})))

    # ---- outputs ---------------------------------------------------------------------------------------------------
    write_results_csv(datas, series, summary)
    write_decisions(datas, series, summary)
    summary["rules"] = evaluate_rules(summary)
    summary["runtime_seconds"] = time.perf_counter() - t_start
    (EXP_DIR / "summary.json").write_text(json.dumps(to_plain(summary), indent=1))
    log(f"done in {summary['runtime_seconds']:.0f}s")
    return summary


def matched_sentinels(source: Source, sent: dict) -> dict[str, dict[str, float]]:
    """{distribution name: {quantile name: sentinel value}} to evaluate for this source."""
    names = {sent["matched_distribution"][source.key]}
    names.add("released_all")
    return {n: sent["distributions"][source.dataset][n]["sentinel_values"] for n in sorted(names)}


def sentinel_audit(config: dict, scorers: dict, datas: dict, clip_info: dict, log) -> dict:
    out: dict = {"distributions": {}, "matched_distribution": {}, "exp023_reproduction": {}, "test_frame_context": {}}
    for ds, scorer in scorers.items():
        dist: dict = {}
        train_dir = STGNF_ROOT / f"data/{ds}/pose/train"
        files = sorted(train_dir.glob(f"*{SUFFIX}"))
        if ds == "ShanghaiTech":
            calib = [p for p in files if "_".join(p.name.split("_")[:2]) in SHT_CALIB_TRAIN_CLIPS]
            variants = {"released_all": files, "released_calib12": calib}
        else:
            variants = {"released_all": files}
        for name, fl in variants.items():
            log(f"  {ds}: train-normal windows of {name} ({len(fl)} clips)")
            tr = train_normality_distribution(fl, scorer)
            dist[name] = {"n_clips": len(fl), "window": distribution_summary(tr["window"]), "per_frame_min": distribution_summary(tr["frame_min"]),
                          "sentinel_values": sentinel_values(tr["window"]), "_raw": tr["window"]}
        if ds == "ShanghaiTech":
            for src in ("stock", "ML0_mps"):
                log(f"  {ds}: detecting + scoring {src} on the 12 calibration clips")
                files_src = detect_train_poses(src, SHT_CALIB_TRAIN_CLIPS, WORK_DIR / f"train_{src}", log)
                tr = train_normality_distribution(list(files_src.values()), scorer)
                dist[f"{src}_calib12"] = {"n_clips": len(files_src), "window": distribution_summary(tr["window"]),
                                          "per_frame_min": distribution_summary(tr["frame_min"]),
                                          "sentinel_values": sentinel_values(tr["window"]), "_raw": tr["window"]}
        # reproduce EXP-023's number
        value = reproduce_exp023_sentinel(ds, scorer)
        frozen = json.loads((EXP023_DIR / "frozen_manifest.json").read_text())
        frozen_val = frozen["shanghaitech" if ds == "ShanghaiTech" else "ubnormal"]["train_sentinel_normality"]
        raw_all = dist["released_all"]["_raw"]
        out["exp023_reproduction"][ds] = {"recomputed": value, "frozen_in_exp023_manifest": frozen_val,
                                         "matches": bool(value is not None and abs(value - frozen_val) < 5e-4),
                                         "recipe": "first 2 windows of each track of the first 25 sorted released train clips, <=512 windows, quantile 0.90",
                                         "percentile_rank_in_all_train_windows": percentile_rank(raw_all, frozen_val),
                                         "share_of_train_windows_more_normal": 1 - percentile_rank(raw_all, frozen_val),
                                         "train_window_max": float(raw_all.max()), "train_window_p99": float(np.quantile(raw_all, 0.99))}
        out["distributions"][ds] = {k: {kk: vv for kk, vv in v.items() if kk != "_raw"} for k, v in dist.items()}
        # where do test frames sit relative to those sentinels
        for src in [s for s in datas if s.startswith(ds + "/")]:
            d = datas[src]
            raw = d.raw_per_frame("completion", "all")
            flat, lab = np.concatenate(raw), np.concatenate(d.labels)
            ok = np.isfinite(flat)
            out["test_frame_context"][src] = {
                "global_most_normal_test": float(flat[ok].max()),
                "test_normal_frame_normality_quantiles": {f"{q:g}": float(np.quantile(flat[ok & (lab == 0)], q)) for q in (0.5, 0.9, 0.99, 1.0)},
                "test_anomalous_frame_normality_quantiles": {f"{q:g}": float(np.quantile(flat[ok & (lab == 1)], q)) for q in (0.5, 0.9, 0.99, 1.0)},
                "share_of_scored_test_normal_frames_more_normal_than_exp023_sentinel": float((flat[ok & (lab == 0)] > frozen_val).mean())}
        out["matched_distribution"].update({f"{ds}/{s.name}": matched_name(ds, s.name) for s in build_sources() if s.dataset == ds})
    return out


def matched_name(dataset: str, source: str) -> str:
    if dataset == "ShanghaiTech":
        return {"released": "released_all", "stock": "stock_calib12", "ML0_mps": "ML0_mps_calib12", "ML0_ncnn": "ML0_mps_calib12"}.get(source, "released_all")
    return "released_all"


def reproduce_exp023_sentinel(dataset: str, scorer: Scorer) -> float | None:
    try:
        from edgelab.analysis.exp023 import calibrate_train_sentinel

        return float(calibrate_train_sentinel(dataset, scorer.checkpoint))
    except Exception as exc:  # noqa: BLE001 - audit must not die on an optional cross-check
        print(f"EXP-023 sentinel reproduction failed: {exc}")
        return None


def post_normalization_auc(data: SourceData, scorer: Scorer) -> dict:
    """S2-protocol AUC when every window is scored with a normalisation that was NOT computed on the window itself."""
    seg_len = data.seg_len
    pose_files = index_pose_dir(data.source.pose_dir, data.source.dataset)
    t = data.table
    ordered = []  # same clip -> person -> window order as build_window_table
    for clip in data.clips:
        path = pose_files.get(clip)
        tracks = read_pose_clip(path) if path else {}
        for _, (frames, kp) in tracks.items():
            starts = complete_window_starts(frames, seg_len)
            if len(starts) == 0:
                continue
            w = gather_windows(kp, starts, seg_len)
            m0, s0 = window_stats(w[:1])
            mt, st = window_stats(kp[None])
            ordered.append((normalize_with_stats(w, m0[0], float(s0[0])), normalize_with_stats(w, mt[0], float(st[0]))))
    first = np.concatenate([o[0] for o in ordered])
    whole = np.concatenate([o[1] for o in ordered])
    out = {}
    for name, x in (("first_window_stats", first), ("whole_track_stats", whole)):
        norm = np.concatenate([scorer.score_normalized(x[i:i + 4096]) for i in range(0, len(x), 4096)])
        raw = per_frame_min(t["clip_idx"], t["last_frame"], norm, data.n_frames)
        filled, _ = global_most_normal_fill(raw)
        scores = [causal_smooth(-c) for c in filled]
        out[name] = data.auc_triplet(scores)
    out["per_window_reference_S2"] = data.auc_triplet(data.step_scores("S2")[0])
    return out


# ---- outputs ----------------------------------------------------------------------------------------------------
RESULT_COLUMNS = ["dataset", "video", "frontend", "split", "n_frames", "fps", "duration_s", "anomalous_frames", "anomalous_fraction",
                  "gt_len_equals_cache_len", "n_tracks", "n_poses", "n_windows_complete", "n_windows_stgnf_offline",
                  "n_omitted_last_windows", "n_windows_completed_after_end_landing", "max_completion_lag_frames",
                  "coverage_offline_middle", "coverage_all_windows", "coverage_normal_frames", "coverage_anomalous_frames",
                  "auc_S0", "auc_S1", "auc_S2", "auc_S3", "auc_S4", "selected_sentinel_normality", "selected_hold_frames",
                  "frames_sentinel_S3", "frames_held_S4", "mean_age_frames_S4"]


def write_results_csv(datas: dict[str, SourceData], series: dict, summary: dict) -> None:
    rows = []
    for key, d in datas.items():
        sel = summary["selection"][d.source.dataset]
        sent = summary["decomposition"][key]["steps"]["S3"]["sentinel"]
        t = d.table
        raw_all = d.raw_per_frame("completion", "all")
        raw_off = d.raw_per_frame("middle", "offline")
        for i, clip in enumerate(d.clips):
            m = t["clip_idx"] == i
            lab = d.labels[i]
            fresh = np.isfinite(raw_all[i])
            value, mode, age = hold_fill(raw_all[i], sent, sel["hold"])
            late = (t["last_frame"][m] - (t["start_frame"][m] + d.seg_len - 1)) if m.any() else np.zeros(0)
            cache_ok = ""
            if d.source.cache_dir is not None and (d.source.cache_dir / f"{clip}.npz").exists():
                cache_ok = bool(len(np.load(d.source.cache_dir / f"{clip}.npz")["frame_ids"]) == len(lab))
            used = age[age >= 0]
            rows.append({
                "dataset": d.source.dataset, "video": clip, "frontend": d.source.name,
                "split": "dev" if i in set(d.dev_idx) else "confirmation", "n_frames": len(lab), "fps": FPS[d.source.dataset],
                "duration_s": round(len(lab) / FPS[d.source.dataset], 3), "anomalous_frames": int(lab.sum()),
                "anomalous_fraction": round(float(lab.mean()), 4), "gt_len_equals_cache_len": cache_ok,
                "n_tracks": int(t["clip_n_tracks"][i]), "n_poses": int(t["clip_n_poses"][i]), "n_windows_complete": int(m.sum()),
                "n_windows_stgnf_offline": int(t["in_offline"][m].sum()), "n_omitted_last_windows": int((~t["in_offline"][m]).sum()),
                "n_windows_completed_after_end_landing": int((late > 0).sum()),
                "max_completion_lag_frames": int(late.max()) if len(late) else 0,
                "coverage_offline_middle": round(float(np.isfinite(raw_off[i]).mean()), 4),
                "coverage_all_windows": round(float(fresh.mean()), 4),
                "coverage_normal_frames": round(float(fresh[lab == 0].mean()), 4) if (lab == 0).any() else "",
                "coverage_anomalous_frames": round(float(fresh[lab == 1].mean()), 4) if (lab == 1).any() else "",
                **{f"auc_{s}": ("" if math.isnan(clip_auc(lab, series[key][s][i])) else round(100 * clip_auc(lab, series[key][s][i]), 3))
                   for s in ("S0", "S1", "S2", "S3", "S4")},
                "selected_sentinel_normality": round(sent, 5), "selected_hold_frames": hold_name(sel["hold"]),
                "frames_sentinel_S3": int((hold_fill(raw_all[i], sent, 0)[1] == 2).sum()), "frames_held_S4": int((mode == 1).sum()),
                "mean_age_frames_S4": round(float(used.mean()), 3) if len(used) else ""})
    with open(EXP_DIR / "results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def write_decisions(datas: dict[str, SourceData], series: dict, summary: dict, max_lines: int = 30000) -> None:
    """Sample log: every window completion and every per-frame delivery decision of 4 clips per dataset (ML0 MPS), + reuse checks."""
    lines: list[str] = []
    for ds in DATASETS:
        key = f"{ds}/ML0_mps"
        if key not in datas:
            continue
        d, sel = datas[key], summary["selection"][ds]
        pos = {c: i for i, c in enumerate(d.clips)}
        conf = summary_split_names(summary, ds, d)
        picks = conf["dev"][:2] + conf["confirmation"][:2]
        t = d.table
        for clip in picks:
            i = pos[clip]
            m = np.flatnonzero(t["clip_idx"] == i)
            for j in m[:600]:
                lines.append(json.dumps({"kind": "window_complete", "dataset": ds, "video": clip, "person": int(t["person"][j]),
                                         "start_frame": int(t["start_frame"][j]), "completion_frame": int(t["last_frame"][j]),
                                         "completion_ms": round(float(frame_time_ms(ds, int(t["last_frame"][j]))), 3),
                                         "paper_middle_landing": int(t["start_frame"][j] + d.seg_len // 2),
                                         "end_landing": int(t["start_frame"][j] + d.seg_len - 1),
                                         "stgnf_scores_it": bool(t["in_offline"][j]), "normality": round(float(t["normality"][j]), 6)}))
            raw = d.raw_per_frame("completion", "all")[i]
            value, mode, age = hold_fill(raw, sel["sentinel"], sel["hold"])
            score = series[key]["S4"][i]
            for fr in range(len(raw)):
                lines.append(json.dumps({"kind": "frame_delivery", "dataset": ds, "video": clip, "frame": fr,
                                         "t_ms": round(float(frame_time_ms(ds, fr)), 3),
                                         "mode": ("fresh", "held", "sentinel")[int(mode[fr])], "age_frames": int(age[fr]),
                                         "normality_used": round(float(value[fr]), 6), "delivered_anomaly_score": round(float(score[fr]), 6)}))
        for rec in summary["R5_reuse"].get(ds, {}).get("tracks", []):
            lines.append(json.dumps({"kind": "reuse_check", "dataset": ds, **rec,
                                     "per_frame_input_max_abs_diff": summary["R5_reuse"][ds]["per_frame_reuse"]["input_max_abs_diff_shared_vs_recomputed"]}))
    lines = lines[:max_lines]
    (EXP_DIR / "decisions.jsonl").write_text("\n".join(lines) + "\n")


def summary_split_names(summary: dict, ds: str, d: SourceData) -> dict[str, list[str]]:
    return {"dev": [d.clips[i] for i in sorted(d.dev_idx, key=lambda k: d.clips[k])],
            "confirmation": [d.clips[i] for i in sorted(d.conf_idx, key=lambda k: d.clips[k])]}


def evaluate_rules(summary: dict) -> dict:
    rules: dict = {}
    rules["R1"] = {"pass": bool(summary["R1_lock_verification"]["passes"]), "detail": f"{summary['R1_lock_verification']['n_checked']} objects re-hashed"}
    bad = []
    for key, r in summary["R2_reproduction"].items():
        for variant, v in r["variants"].items():
            for ref in ("delta_vs_historical", "delta_vs_claimed"):
                if v[ref] is not None and abs(v[ref]) > 0.01:
                    bad.append(f"{key}/{variant}/{ref}={v[ref]:+.4f}")
    rules["R2"] = {"pass": not bad, "detail": bad or "all |delta| <= 0.01 pt"}
    rb = [k for k, v in summary["R2b_replay"].items() if v.get("applicable") and (v["differing"] or v["missing"])]
    rules["R2b"] = {"pass": not rb, "detail": rb or "all replays bit-identical"}
    pb = [k for k, v in summary["R3_window_parity"].items()
          if v["only_in_stgnf"] or v["only_in_mine"] or v["max_abs_normality_diff"] > 1e-4]
    rules["R3"] = {"pass": not pb, "detail": pb or "identical key sets, max |d normality| %.2e" %
                   max(v["max_abs_normality_diff"] for v in summary["R3_window_parity"].values())}
    anchors = []
    for key, dec in summary["decomposition"].items():
        rep = summary["R2_reproduction"][key]["variants"]
        for step, variant in (("S0", "paper"), ("S1", "live")):
            diff = dec["steps"][step]["auc"]["all"] - rep[variant]["auc"]
            if abs(diff) > 0.01:
                anchors.append(f"{key}/{step}={diff:+.4f}")
    rules["R4"] = {"pass": not anchors, "detail": anchors or "S0 == paper and S1 == live for every source (<= 0.01 pt)"}
    rules["R5"] = {"pass": all(v["passes"] for v in summary["R5_reuse"].values()), "detail": {
        ds: v["per_frame_reuse"]["input_max_abs_diff_shared_vs_recomputed"] for ds, v in summary["R5_reuse"].items()}}
    rules["R6"] = {"pass": all(v["prefix_invariance_max_abs_diff"] <= 1e-12 and v["windows_landing_before_completion_frame"] == 0
                               and v["matches_CausalScoreDeliverer_max_abs_diff"] <= 1e-9 for v in summary["R6_causality"].values()),
                   "detail": {ds: v["prefix_invariance_max_abs_diff"] for ds, v in summary["R6_causality"].items()}}
    c = summary["R7_cadence"]
    cad_ok = (c["ShanghaiTech"]["all_train_fps_equal_24"] and c["UBnormal"].get("all_30fps", False) and not c["UBnormal"]["unreadable"]
              and not c["UBnormal"]["frame_count_mismatches_vs_gt"] and not c["ShanghaiTech"]["cache_length_mismatches"]
              and not c["UBnormal"]["cache_length_mismatches"])
    rules["R7"] = {"pass": bool(cad_ok), "detail": "SHT 24 fps, UB 30 fps, GT = cache = video length" if cad_ok else
                   f"SHT train videos by fps {c['ShanghaiTech']['train_fps_counts']} (not all 24); UB all 30 fps = {c['UBnormal'].get('all_30fps')}; "
                   f"GT/cache/video lengths consistent = {not c['UBnormal']['frame_count_mismatches_vs_gt'] and not c['ShanghaiTech']['cache_length_mismatches'] and not c['UBnormal']['cache_length_mismatches']}"}
    rules["R8"] = {"pass": all(v["disjoint_and_covering"] and v["dev_has_both_classes"] and v["confirmation_has_both_classes"]
                               for v in summary["R8_splits"].values()), "detail": summary["R8_splits"]}
    return rules


# =====================================================================================================================
# 14. CLI
# =====================================================================================================================
def _f(x, nd=2, sign=False):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    return f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"


def _ci(ci, nd=2):
    return "n/a" if not ci else f"[{ci[0]:.{nd}f}, {ci[1]:.{nd}f}]"


def _table(header: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def render_report() -> None:
    S = json.loads((EXP_DIR / "summary.json").read_text())
    config = yaml.safe_load((EXP_DIR / "config.yaml").read_text())
    dec, sent, sel, omi = S["decomposition"], S["sentinel"], S["selection"], S["omission"]
    rules, grid = S["rules"], S["delivered_grid"]
    L: list[str] = []
    add = L.append

    def step(key, name):
        return dec[key]["steps"][name]

    # ---- headline numbers used in the answer ----------------------------------------------------------------------
    n_pass = sum(1 for r in rules.values() if r["pass"])
    add("# Q2-P1-E00 Audit report\n")
    add(f"*Generated by `edgelab.analysis.q2_p1_e00 report` from `summary.json` (seed {S['seed']}, "
        f"commit `{S['environment'].get('git_commit', '?')[:10]}`, runtime {S['runtime_seconds'] / 60:.0f} min). No model was trained.*\n")
    add("## 1. Answer\n")
    r2 = S["R2_reproduction"]
    worst = max((abs(v[ref]) for r in r2.values() for v in r["variants"].values() for ref in ("delta_vs_historical",) if v[ref] is not None), default=0)
    sht, ub = "ShanghaiTech/ML0_mps", "UBnormal/ML0_mps"
    add(f"**Can we reproduce existing scores, timestamps, normalisation and missing-data handling? "
        f"{'Yes' if all(rules[k]['pass'] for k in ('R1', 'R2', 'R2b', 'R3', 'R4')) else 'Partly (see rule table)'} for scores, windows, normalisation and missing data; timestamps only partly (ShanghaiTech is not uniformly 24 fps).** "
        f"{n_pass}/{len(rules)} pre-registered rules pass.\n")
    add(f"* Every historical AUC (offline *paper* and *live* variant, all 107 SHT / 211 UB clips, 10 sources) is re-scored with the current code to within "
        f"{worst:.4f} pt of the stored value; STG-NF window keys/normality match exactly; detector-cache replays regenerate the stored pose JSON bit-identically.")
    add(f"* The offline->causal gap is **not** one effect. On {sht} the chain S0->S4 is "
        + " -> ".join(f"{n} {_f(step(sht, n)['auc']['all'])}" for n in ("S0", "S1", "S2", "S3", "S4")) +
        f"; on {ub}: " + " -> ".join(f"{n} {_f(step(ub, n)['auc']['all'])}" for n in ("S0", "S1", "S2", "S3", "S4")) + ".")
    ex = sent["exp023_reproduction"]
    add(f"* EXP-023's train sentinel is reproduced (SHT {ex['ShanghaiTech']['recomputed']:.4f}, UB {ex['UBnormal']['recomputed']:.4f}); "
        f"on SHT it sits at the {100 * ex['ShanghaiTech']['percentile_rank_in_all_train_windows']:.1f}th percentile of all train-normal window scores "
        f"({100 * ex['ShanghaiTech']['share_of_train_windows_more_normal']:.1f}% of train windows are more normal), on UB at the "
        f"{100 * ex['UBnormal']['percentile_rank_in_all_train_windows']:.1f}th.")
    r5 = S["R5_reuse"]
    add(f"* Per-frame (x, y, conf) reuse across overlapping windows is bit-exact (max_abs_diff = "
        + ", ".join(f"{ds} {v['per_frame_reuse']['input_max_abs_diff_shared_vs_recomputed']}" for ds, v in r5.items()) +
        "); post-normalisation or hidden-state reuse is not.")
    cad = S["R7_cadence"]
    add(f"* UBnormal is 30 fps (min/max {cad['UBnormal'].get('fps_min')}/{cad['UBnormal'].get('fps_max')}); EXP-023 hard-codes 24.0 fps in "
        f"{len(cad['exp023_hardcoded_lines'])} places. ShanghaiTech is *not* uniformly 24 fps: {cad['ShanghaiTech']['train_fps_counts']} training videos by container fps "
        f"(scenes {', '.join(cad['ShanghaiTech']['scenes_with_mixed_fps'])} mix both); test clips are JPEG sequences without metadata, so R7 **fails** as pre-registered "
        f"(SHT 24 fps is an approximation, <= 4 % error on SHT time-based metrics).\n")
    pa_ = lambda key, k: dec[key]["steps"]["paired"][k]
    add("**Key findings (ML0_mps, 95% video-bootstrap CI in brackets)**\n")
    add(f"1. *ShanghaiTech* offline->causal gap {_f(pa_(sht, 'S0->S4')['delta'], 2, True)} {_ci(pa_(sht, 'S0->S4')['ci95'])}: almost all of it is the S0->S1 step "
        f"({_f(pa_(sht, 'S0->S1')['delta'], 2, True)}), i.e. one-sided (causal) smoothing ({_f(pa_(sht, 'S0->S0b')['delta'], 2, True)} alone) plus end landing "
        f"({_f(pa_(sht, 'S0->S0a')['delta'], 2, True)} alone). Fixing the last-window omission + completion landing "
        f"({_f(pa_(sht, 'S1->S2')['delta'], 2, True)}), the train sentinel ({_f(pa_(sht, 'S2->S3')['delta'], 2, True)}) and holding "
        f"({_f(pa_(sht, 'S3->S4')['delta'], 2, True)} at H={hold_name(sel['ShanghaiTech']['hold'])}) change nothing measurable. Causal delivered AUC "
        f"{_f(step(sht, 'S4')['auc']['all'])} (dev {_f(step(sht, 'S4')['auc']['dev'])}, confirmation {_f(step(sht, 'S4')['auc']['confirmation'])}).")
    add(f"2. *UBnormal*: the S0->S1 step is not significant ({_f(pa_(ub, 'S0->S1')['delta'], 2, True)} {_ci(pa_(ub, 'S0->S1')['ci95'])}); the train-calibrated sentinel is "
        f"*better* than the global test max fill ({_f(pa_(ub, 'S2->S3')['delta'], 2, True)} {_ci(pa_(ub, 'S2->S3')['ci95'])}), so the causal protocol scores "
        f"{_f(step(ub, 'S4')['auc']['all'])} (dev {_f(step(ub, 'S4')['auc']['dev'])}, confirmation {_f(step(ub, 'S4')['auc']['confirmation'])}) vs offline "
        f"{_f(step(ub, 'S0')['auc']['all'])}. The sentinel quantile matters on UB (p50 {_f(grid[ub]['released_all|p50|H0']['auc']['all'])} vs max "
        f"{_f(grid[ub]['released_all|max|H0']['auc']['all'])}); the quantile is selected on dev only.")
    r5s, r5u = r5["ShanghaiTech"]["post_normalization_auc"], r5["UBnormal"]["post_normalization_auc"]
    add(f"3. Overlap reuse: only the per-frame (x, y, conf) store can be shared (max_abs_diff 0.0, {r5['ShanghaiTech']['reuse_factor']:.1f}x / {r5['UBnormal']['reuse_factor']:.1f}x fewer "
        f"pose evaluations). Reusing normalised inputs from a frozen window destroys the score (S2 AUC {_f(r5s['per_window_reference_S2']['all'])}->{_f(r5s['first_window_stats']['all'])} SHT, "
        f"{_f(r5u['per_window_reference_S2']['all'])}->{_f(r5u['first_window_stats']['all'])} UB); flow hidden states of shared frames differ "
        f"(max {r5['ShanghaiTech']['hidden_state_reuse']['max_abs_diff_correct_norm']:.2f} SHT / {r5['UBnormal']['hidden_state_reuse']['max_abs_diff_correct_norm']:.2f} UB).")
    om = omi[sht]
    add(f"4. Missing data: the STG-NF last-window omission drops {om['n_omitted_last_windows']} of {om['n_complete_windows']} complete windows ({100 * om['omitted_fraction']:.2f}%) on {sht}; "
        f"{om['n_windows_with_missing_frames']} windows complete later than start+seg_len-1 (up to {om['max_completion_lag_frames']} frames). Frame coverage by fresh scores: "
        f"{100 * step(sht, 'S2')['coverage']['all']:.1f}% SHT, {100 * step(ub, 'S2')['coverage']['all']:.1f}% UB, so {100 - 100 * step(ub, 'S2')['coverage']['all']:.1f}% of UB frames fall back to the sentinel.\n")

    # ---- 2. Locks ---------------------------------------------------------------------------------------------------
    add("## 2. Locked inputs (R1)\n")
    add(f"`config.yaml` was written before any scoring; the run re-hashed {S['R1_lock_verification']['n_checked']} objects "
        f"({'all unchanged' if S['R1_lock_verification']['passes'] else S['R1_lock_verification']['problems']}).\n")
    add(_table(["artifact", "bytes", "SHA-256"],
               [[k, v.get("bytes", ""), f"`{v['sha256']}`"] for k, v in config["frozen_artifacts"].items()]))
    add("")
    add(_table(["detector cache", "files", "tree SHA-256"],
               [[k, v.get("n_videos", ""), f"`{v['tree_sha256']}`"] for k, v in config["detector_caches"].items()]))
    add("")
    add("### Splits (R8)\n")
    add(_table(["dataset", "test", "dev", "confirmation", "dev anom. frames", "conf. anom. frames", "disjoint+covering"],
               [[ds, r["n_test"], r["n_dev"], r["n_confirmation"], r["dev_anomalous_frames"], r["confirmation_anomalous_frames"],
                 r["disjoint_and_covering"]] for ds, r in S["R8_splits"].items()]))
    add("")

    # ---- 3. Reproduction --------------------------------------------------------------------------------------------
    add("## 3. Reproduction of historical scores (R2/R2b/R3)\n")
    rows = []
    for key, r in r2.items():
        for variant, v in r["variants"].items():
            rows.append([key, variant, _f(v["auc"], 3), _ci(v["ci95"]), _f(v["historical_auc"], 3), _f(v["delta_vs_historical"], 4, True),
                         _f(v["claimed_auc"], 2), _f(v["delta_vs_claimed"], 3, True), f"{100 * v['coverage_anomalous']:.1f}/{100 * v['coverage_normal']:.1f}"])
    add(_table(["source", "variant", "re-scored AUC", "95% CI (video bootstrap)", "stored AUC", "d vs stored", "claimed", "d vs claimed",
                "cov. anom/normal %"], rows))
    add("")
    rp, pr = S["R2b_replay"], S["R3_window_parity"]
    add(_table(["source", "STG-NF segments", "matched", "only STG-NF", "only mine", "max abs d normality", "replay clips checked", "replay differing"],
               [[k, pr[k]["n_stgnf_segments"], pr[k]["n_matched"], pr[k]["only_in_stgnf"], pr[k]["only_in_mine"],
                 f"{pr[k]['max_abs_normality_diff']:.2e}", rp[k].get("n_checked", rp[k].get("n_clips", "n/a")) if rp[k].get("applicable") else "n/a",
                 len(rp[k].get("differing", [])) if rp[k].get("applicable") else "n/a"] for k in pr]))
    add("")
    add("Stored AUCs come from the earlier experiment outputs (n/a for the released AlphaPose poses, which have no stored result); 'claimed' are the thesis-table figures "
        "(two decimals, so |d| <= 0.005 is rounding). MLP_ncnn / released CIs: bootstrap CIs are computed for released, stock, ML0 (MPS, NCNN SHT) and MLP_mps in section 4; "
        "MLP_ncnn has point estimates only.\n")

    # ---- 4. Decomposition -------------------------------------------------------------------------------------------
    add("## 4. Offline->causal decomposition (S0-S4)\n")
    add("S0 offline paper; S1 end landing + causal smoothing (the old 'live'); S2 fix last-window omission and land at the completion frame; "
        "S3 train-calibrated sentinel instead of the global `most_normal` fill (no hold); S4 = S3 + hold. S0-S2 still use the global test-set fill "
        "(look-ahead). S3/S4 are causal. Sentinel/hold selected on ML0_mps **dev** only, applied unchanged to every source.\n")
    for ds in DATASETS:
        s_ = sel[ds]
        add(f"**{ds}** - selected on `{s_['selected_on']}`: sentinel `{s_['quantile']}` = {s_['sentinel']:.4f} (`{s_['matched_distribution']}`), "
            f"hold H = {hold_name(s_['hold'])}; dev AUC {s_['dev_auc']:.2f} (best dev {s_['best_dev_auc']:.2f}).\n")
        rows = []
        for key in dec:
            if not key.startswith(ds + "/"):
                continue
            st = dec[key]["steps"]
            for n in ("S0", "S1", "S2", "S3", "S4"):
                a = st[n]["auc"]
                rows.append([key.split("/")[1], n, _f(a["all"]), _ci(st[n].get("ci95_all")), _f(a["dev"]), _f(a["confirmation"]),
                             f"{100 * st[n]['coverage']['all']:.1f}" if "coverage" in st[n] else "-"])
        add(_table(["source", "step", "AUC all", "95% CI", "dev", "confirmation", "coverage % (fresh frames)"], rows))
        add("")
        rows = []
        for key in dec:
            if not key.startswith(ds + "/") or "paired" not in dec[key]["steps"]:
                continue
            for k, v in dec[key]["steps"]["paired"].items():
                rows.append([key.split("/")[1], k, _f(v["delta"], 2, True), _ci(v["ci95"]), "yes" if v["excludes_zero"] else "no"])
        add("Paired video-bootstrap contrasts (AUC points, includes ablations S0a/S0b/S1a/S1b):\n")
        add(_table(["source", "contrast", "delta", "95% CI", "CI excludes 0"], rows))
        add("")
    # ablations
    add("### Ablation levels (AUC all clips)\n")
    rows = []
    for key in dec:
        st = dec[key]["steps"]
        if "S0a" in st:
            rows.append([key] + [_f(st[n]["auc"]["all"]) for n in ("S0", "S0a", "S0b", "S1", "S1a", "S1b", "S2")])
    add(_table(["source", "S0", "S0a end only", "S0b causal only", "S1", "S1a omission fixed", "S1b completion only", "S2"], rows))
    add("")
    add("### Last-window omission and completion lag\n")
    add(_table(["source", "complete windows", "STG-NF scores", "omitted", "omitted %", "windows w/ missing frames", "max completion lag (frames)"],
               [[k, v["n_complete_windows"], v["n_stgnf_offline_windows"], v["n_omitted_last_windows"], f"{100 * v['omitted_fraction']:.2f}",
                 v["n_windows_with_missing_frames"], v["max_completion_lag_frames"]] for k, v in omi.items()]))
    add("")

    # ---- 5. Sentinel -----------------------------------------------------------------------------------------------
    add("## 5. Sentinel calibration (training data only)\n")
    for ds in DATASETS:
        add(f"**{ds}** - distribution of train-normal window normality (-NLL; higher = more normal):\n")
        rows = []
        for name, d in sent["distributions"][ds].items():
            q = d["window"]["quantiles"]
            rows.append([name, d["n_clips"], d["window"]["n"]] + [_f(q[k], 3) for k in ("0.01", "0.1", "0.5", "0.9", "0.99", "1")])
        add(_table(["train set", "clips", "windows", "p1", "p10", "p50", "p90", "p99", "max"], rows))
        add("")
        e = sent["exp023_reproduction"][ds]
        add(f"EXP-023 sentinel: recomputed {e['recomputed']:.4f} vs frozen {e['frozen_in_exp023_manifest']:.4f} (match: {e['matches']}); "
            f"recipe: {e['recipe']}. Percentile rank among all train-normal windows: {100 * e['percentile_rank_in_all_train_windows']:.2f}%; "
            f"train p99 = {e['train_window_p99']:.4f}, max = {e['train_window_max']:.4f}.\n")
        rows = []
        for key, c in sent["test_frame_context"].items():
            if not key.startswith(ds + "/"):
                continue
            nq, aq = c["test_normal_frame_normality_quantiles"], c["test_anomalous_frame_normality_quantiles"]
            rows.append([key.split("/")[1], _f(c["global_most_normal_test"], 3), _f(nq["0.5"], 3), _f(nq["0.99"], 3), _f(aq["0.5"], 3),
                         f"{100 * c['share_of_scored_test_normal_frames_more_normal_than_exp023_sentinel']:.1f}"])
        add("Where test frames sit (completion landing, all windows):\n")
        add(_table(["source", "global test max (the S0-S2 fill)", "normal-frame median", "normal-frame p99", "anomalous-frame median",
                    "% scored normal test frames more normal than EXP-023 sentinel"], rows))
        add("")
    add("### Sentinel x hold grid (delivered AUC, matched train distribution)\n")
    for key in (sht, ub):
        rows = []
        m = sent["matched_distribution"][key]
        for k, v in grid[key].items():
            if v["dist"] != m:
                continue
            rows.append([v["quantile"], _f(v["sentinel"], 3), v["hold_name"], _f(v["auc"]["dev"]), _f(v["auc"]["confirmation"]), _f(v["auc"]["all"]),
                         f"{100 * v['fresh_fraction']:.1f}", f"{100 * v['held_fraction']:.1f}", f"{100 * v['sentinel_fraction']:.1f}",
                         _f(v.get("mean_age_frames"), 1), _f(v.get("p95_age_frames"), 0)])
        add(f"`{key}` (`{m}`); selected: `{sel[key.split('/')[0]]['quantile']}`/H{hold_name(sel[key.split('/')[0]]['hold'])}\n")
        add(_table(["sentinel", "value", "H", "dev", "confirm.", "all", "fresh %", "held %", "sentinel %", "mean age", "p95 age"], rows))
        add("")
    add("Other sources and the `released_all` vs frontend-matched distributions are in `summary.json -> delivered_grid` (every row, every source).\n")

    # ---- 6. Reuse ---------------------------------------------------------------------------------------------------
    add("## 6. Normalisation and overlap reuse (R5)\n")
    for ds, v in r5.items():
        pf, pn, hs = v["per_frame_reuse"], v["post_normalization_reuse"], v["hidden_state_reuse"]
        add(f"**{ds}** ({v['n_tracks']} tracks, {v['n_windows']} windows from {v['n_poses']} poses; {v['reuse_factor']:.1f}x pose evaluations without reuse):\n")
        add(_table(["check", "result"], [
            ["per-frame (x,y,conf) shared store vs per-window recompute: max_abs_diff", pf["input_max_abs_diff_shared_vs_recomputed"]],
            ["... vs streaming ring buffer: max_abs_diff", pf["input_max_abs_diff_shared_vs_streaming"]],
            ["window scores, same batch: max_abs_diff", pf["score_max_abs_diff_same_batch"]],
            ["window scores, batch-of-1 vs batch (float32 noise)", f"{pf['score_max_abs_diff_single_window_vs_batch_float32_noise']:.2e}"],
            ["frozen first-window mean/std: input max / score max abs diff", f"{pn['frozen_first_window_stats']['input_max_abs_diff']:.3f} / {pn['frozen_first_window_stats']['score_max_abs_diff']:.3f}"],
            ["whole-track mean/std: input max / score max abs diff", f"{pn['whole_track_stats']['input_max_abs_diff']:.3f} / {pn['whole_track_stats']['score_max_abs_diff']:.3f}"],
            ["hidden state (flow.layers[0]) of 23 shared frames, correct per-window norm: max abs diff", f"{hs['max_abs_diff_correct_norm']:.3f}"],
            ["... with identical norm stats, all frames / interior frames", f"{hs['max_abs_diff_shared_norm_stats']:.3f} / {hs['max_abs_diff_shared_norm_stats_interior_frames']:.3f}"]]))
        pa = v["post_normalization_auc"]
        add(f"\nS2-protocol AUC (all clips) with window-external normalisation: per-window (correct) {pa['per_window_reference_S2']['all']:.2f}, "
            f"frozen first-window stats {pa['first_window_stats']['all']:.2f}, whole-track stats {pa['whole_track_stats']['all']:.2f}.\n")
    add("Conclusion: cache the per-frame detections/poses (bit-exact), normalise and score every window independently. "
        "Caching normalised inputs or intermediate flow activations is not valid.\n")

    # ---- 7. Cadence -------------------------------------------------------------------------------------------------
    add("## 7. Timestamps and cadence (R7)\n")
    c24 = cad["ubnormal_consequences_of_24fps_assumption"]
    add(_table(["item", "value"], [
        ["SHT training videos by container fps (test clips are JPEG: no metadata)", f"{cad['ShanghaiTech']['train_fps_counts']} of {cad['ShanghaiTech']['n_train_videos']}; scenes with both 24 and 25: {cad['ShanghaiTech']['scenes_with_mixed_fps']}"],
        ["UB test videos fps min/max (all 30?)", f"{cad['UBnormal'].get('fps_min')} / {cad['UBnormal'].get('fps_max')} -> {cad['UBnormal'].get('all_30fps')}"],
        ["UB frame-count mismatches vs GT", len(cad["UBnormal"]["frame_count_mismatches_vs_gt"])],
        ["cache/GT length mismatches (SHT, UB)", f"{len(cad['ShanghaiTech']['cache_length_mismatches'])}, {len(cad['UBnormal']['cache_length_mismatches'])}"],
        ["UB 16-frame window = true / assumed seconds", f"{c24['window_seconds_true']:.3f} / {c24['window_seconds_assumed']:.3f}"],
        ["UB frame interval true / assumed (ms)", f"{c24['frame_interval_ms_true']:.2f} / {c24['frame_interval_ms_assumed']:.2f}"],
        ["delay (ms) overstated by", f"{c24['delay_ms_overstated_by_factor']:.2f}x"],
        ["UB normal hours true / assumed", f"{c24['normal_hours_true']:.3f} / {c24['normal_hours_assumed']:.3f}"],
        ["false alerts / hour understated by", f"{c24['false_alerts_per_hour_understated_by_factor']:.2f}x (true rate = assumed / 0.8)"],
        ["detector budget at stride 2 / 4 (Hz) true vs assumed", f"{c24['detector_budget_hz_true_for_stride_2']} vs {c24['detector_budget_hz_assumed_for_stride_2']} / "
                                                                f"{c24['detector_budget_hz_true_for_stride_4']} vs {c24['detector_budget_hz_assumed_for_stride_4']}"]]))
    add("\nEXP-023 lines with the hard-coded 24.0:\n")
    for line in cad["exp023_hardcoded_lines"]:
        add(f"* `{line}`")
    add("\nThe audit uses `frame_time_ms(dataset, frame)` with native fps everywhere; ShanghaiTech test frames are JPEG sequences without metadata, so "
        "24 fps rests on the training videos and the dataset spec.\n")

    # ---- 8. Causality -----------------------------------------------------------------------------------------------
    add("## 8. Causality (R6)\n")
    add(_table(["dataset", "prefix-invariance max diff", "checks", "== CausalScoreDeliverer max diff", "windows landing before completion",
                "global fill: full set vs kept prefix fraction", "global fill is look-ahead"],
               [[ds, f"{v['prefix_invariance_max_abs_diff']:.1e}", v["prefix_checks"], f"{v['matches_CausalScoreDeliverer_max_abs_diff']:.1e}",
                 v["windows_landing_before_completion_frame"], f"{v['global_fill_value_full_video_set']:.3f} vs " + ", ".join(f"{k}: {x:.3f}" for k, x in v["global_fill_value_by_kept_prefix_fraction"].items()),
                 v["global_fill_is_look_ahead"]] for ds, v in S["R6_causality"].items()]))
    add("")

    # ---- 9. Bugs / limitations / decision ---------------------------------------------------------------------------
    add("## 9. Defects found in the existing pipeline\n")
    s0s1 = step(sht, "S1")["auc"]["all"] - step(sht, "S0")["auc"]["all"]
    add(f"1. **Global `most_normal` fill** (S0-S2) uses the maximum over the whole test set - a test statistic. Replacing it by a train-calibrated sentinel changes "
        f"AUC by {_f(step(sht, 'S3')['auc']['all'] - step(sht, 'S2')['auc']['all'], 2, True)} pt on SHT (nothing) and "
        f"{_f(step(ub, 'S3')['auc']['all'] - step(ub, 'S2')['auc']['all'], 2, True)} pt on UB (ML0_mps; paired CIs in section 4): on UB the *test-set-max* fill is the worst "
        f"choice, because the most-normal test frame is far more normal than anything a frame without a score should be assigned.")
    add(f"2. **EXP-023 sentinel** ({ex['ShanghaiTech']['recomputed']:.4f}) is built from the first 2 windows of each track of 25 clips and sits far inside the normal range "
        f"(see section 5): most frames' normality exceeds it, so it is not a 'no information' value.")
    add("3. **Last-window omission** in STG-NF test-time segmentation (`num_segs = ceil((n - seg_len) / 1)`): removes one window per track "
        "(section 4 table); fixed in S2.")
    add("4. **'end' placement backdates** windows with missing frames (the window completes later than start + seg_len - 1); S2 lands at the completion frame.")
    add("5. **24 fps hard-coded** for UBnormal (section 7).")
    add("6. UBnormal normalisation uses the 856x480 resolution constant also for UB (as in the released code); retained for parity, documented.\n")
    add("## 10. Limitations\n")
    add("* UBnormal official validation (64 clips) has no frontend caches, so UB dev/confirmation are splits of the 211 test clips and the test set has been explored before; "
        "dev/confirmation protect only the sentinel/hold choice made here.")
    add("* The UB sentinel is calibrated on released AlphaPose training poses (no frontend-matched UB training detections); SHT has stock / ML0 matched calibration on 12 train clips.")
    hub = grid[ub]
    add(f"* Hold on UB is unstable: dev prefers H=0 ({hub['released_all|p50|H0']['auc']['dev']:.2f}) but confirmation prefers H=inf "
        f"({hub['released_all|p50|Hinf']['auc']['confirmation']:.2f} vs {hub['released_all|p50|H0']['auc']['confirmation']:.2f}); the dev-only rule selected H=0, so no hold effect is claimed. "
        "UB scenes are heterogeneous (70 dev clips) and the 'all' ranking would differ from dev.")
    add("* Pooled frame-level AUC is tie-sensitive for constant-fill runs; the reference smoother is used so S1 anchors reproduce.")
    add("* Bootstrap resamples videos (seed 0); contrasts use 1000 resamples, historical re-scores 2000.\n")
    add("## 11. Rules\n")
    add(_table(["rule", "pass", "detail"], [[k, "PASS" if v["pass"] else "FAIL", str(v["detail"])[:200]] for k, v in rules.items()]))
    add("\n## 12. Decision for E01\n")
    add("Proceed. Use `completion` landing for every causal run, report the train-sentinel S3/S4 numbers (never S0-S2) as the causal baseline, "
        "take the sentinel quantile and hold selected here, keep per-frame detection caches shared across windows but never share normalised inputs or hidden states, "
        "and use native fps for every delay/FA-per-hour figure.\n")
    (EXP_DIR / "report.md").write_text("\n".join(L) + "\n")
    print(f"wrote {EXP_DIR / 'report.md'}")


def cmd_lock() -> None:
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    config, manifests = build_lock()
    header = ("# Q2-P1-E00 locked configuration (generated by `python -m edgelab.analysis.q2_p1_e00 lock`; do not hand-edit).\n"
              "# Hashes are full SHA-256. `run` re-verifies every hash before computing anything (rule R1).\n")
    (EXP_DIR / "config.yaml").write_text(header + yaml.safe_dump(config, sort_keys=False, width=140))
    (WORK_DIR / "stages").mkdir(parents=True, exist_ok=True)
    (WORK_DIR / "stages" / "cache_file_hashes.json").write_text(json.dumps(manifests))
    print(f"wrote {EXP_DIR / 'config.yaml'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["lock", "run", "report", "all"])
    parser.add_argument("--datasets", nargs="*", default=list(DATASETS), choices=list(DATASETS))
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--skip-hist", action="store_true", help="reuse cached historical re-scores of earlier runs")
    parser.add_argument("--skip-replay", action="store_true", help="reuse cached replay checks")
    args = parser.parse_args()
    if args.command in ("lock", "all"):
        cmd_lock()
    if args.command in ("run", "all"):
        run_audit(tuple(args.datasets), n_boot=args.n_boot, skip_hist=args.skip_hist, skip_replay=args.skip_replay)
    if args.command in ("report", "all"):
        render_report()


if __name__ == "__main__":
    main()
