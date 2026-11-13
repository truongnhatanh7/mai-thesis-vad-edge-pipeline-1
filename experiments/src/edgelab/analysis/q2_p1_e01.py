"""Q2-P1-E01 (Quality): what accuracy can the frozen frontend / scorer reach before CPU constraints?

    uv run --extra pose python -m edgelab.analysis.q2_p1_e01 lock     # hash new caches/pose folders -> Q2-P1-E01_quality/config.yaml
    uv run --extra pose python -m edgelab.analysis.q2_p1_e01 run      # all analyses -> results.csv, decisions.jsonl, summary.json
    uv run --extra pose python -m edgelab.analysis.q2_p1_e01 report   # summary.json -> report.md
    uv run --extra pose python -m edgelab.analysis.q2_p1_e01 all

No training. Builds directly on Q2-P1-E00 (locked splits `test_dev` / `test_confirmation`, hashes, the causal S3/S4 protocol):
this module *imports* E00's window engine, scorer wrapper, protocol steps and bootstrap and re-implements none of them.

    A  frontend / resolution comparison, offline paper S0 and causal S3/S4, dev / confirmation / all, paired video bootstraps
    B  recall by person height + EXP-022 miss decomposition (+ how many misses sit in the raw cache at 0.05 <= conf < 0.25)
    C  tracking gaps and stranded windows (tracks, lengths, stranded short tracks, within-track gaps, scorable coverage)
    D  tracker / gate sweep on `test_dev` ONLY (c, true two-stage BoT-SORT, track_buffer, match_thresh), verified on confirmation
    ->  `locked_for_E02`

The module has pure functions (rules, selection, track statistics, matching; unit-tested without data in
tests/test_q2_p1_e01.py) and I/O drivers that need ~/thesis-data and the STG-NF checkpoints.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import multiprocessing as mp
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from edgelab.analysis import q2_p1_e00 as e00

DATASETS = e00.DATASETS
SEG_LENS = e00.SEG_LENS
SEED = e00.SEED
FPS = e00.FPS
E00_DIR = e00.EXP_DIR
EXP_DIR = e00.EXPERIMENTS_ROOT / "Q2-P1-E01_quality"
DATA_ROOT = e00.DATA_ROOT
WORK_DIR = DATA_ROOT / "q2-p1-e01"
N_BOOT = 1000  # video-bootstrap resamples (E00: same), seed 0
N_BOOT_RECALL = 2000
CHAIN = ("S0", "S3", "S4")

# ---- decision-rule constants (plan.md section 3) --------------------------------------------------------------------
FRONTEND_MARGIN = 2.0  # AUC points a costlier candidate must add (EXP-020 P1/P2 margin) ...
TRACKER_MIN_DELTA = 0.5  # AUC points a non-default tracker winner must add on the primary dev series
TIE_BAND = 0.10  # AUC points: cells within this of the best are tied -> closest to default wins
SMALL_BINS = ("<32", "32-48")
LARGE_BINS = ("96-160", ">=160")
LOW_CONF_LO, LOW_CONF_HI = 0.05, 0.25  # the band that a two-stage tracker / lower gate could use
LOW_CONF_SHARE_THRESHOLD = 0.20
INTERMITTENT_THRESHOLD = 0.30  # EXP-022 rule
LEN_EDGES = (1, 8, 16, 24, 48, 96, np.inf)
GAP_EDGES = (1, 2, 3, 6, 12, 24, np.inf)


def edge_names(edges: tuple, last: str = "+") -> list[str]:
    names = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        hi_i = int(hi) - 1 if np.isfinite(hi) else None
        names.append(f"{int(lo)}{last}" if hi_i is None else (str(int(lo)) if hi_i == int(lo) else f"{int(lo)}-{hi_i}"))
    return names


LEN_NAMES, GAP_NAMES = edge_names(LEN_EDGES), edge_names(GAP_EDGES)


def plain(obj):
    return e00.to_plain(obj)


def now() -> str:
    return time.strftime("%H:%M:%S")


# =====================================================================================================================
# 1. Arms (frontends x resolution x runtime)
# =====================================================================================================================
@dataclass(frozen=True)
class Arm:
    name: str  # e.g. stock, stock_r800, ML0_ncnn   ("stock" = yolov8n-pose @640 MPS, E00's name)
    dataset: str
    label: str
    group: str  # released | stock | M00 | ML0 | MLP
    runtime: str  # alphapose | mps_fp32 | ncnn_fp16
    imgsz: int | None
    cost: float | None  # nominal FLOPs relative to stock@640 (EXP-020 amendment 1.3 for the P2 head)
    sentinel_dist: str  # E00 sentinel distribution (plan A6)
    pose_dir: Path
    cache_dir: Path | None
    in_e00: bool

    @property
    def key(self) -> str:
        return f"{self.dataset}/{self.name}"


def build_arms(d: Path = DATA_ROOT) -> list[Arm]:
    SHT, UB = "ShanghaiTech", "UBnormal"
    src = {s.key: s for s in e00.build_sources(d)}
    tag = "L0_E0_c0.25_botsort"
    arms: list[Arm] = []

    def old(ds, name, label, group, runtime, imgsz, cost, dist):
        s = src[f"{ds}/{name}"]
        arms.append(Arm(name, ds, label, group, runtime, imgsz, cost, dist, s.pose_dir, s.cache_dir, True))

    def new(ds, name, label, group, runtime, imgsz, cost, dist, pose_dir, cache_dir):
        arms.append(Arm(name, ds, label, group, runtime, imgsz, cost, dist, pose_dir, cache_dir, False))

    old(SHT, "released", "released AlphaPose (oracle reference)", "released", "alphapose", None, None, "released_all")
    new(SHT, "stock_r480", "stock @480 (MPS FP32)", "stock", "mps_fp32", 480, 0.5625, "stock_calib12",
        d / "exp003/poses/L0_E0-r480_c0.25_botsort", d / "exp003/cache/L0_E0-r480")
    old(SHT, "stock", "stock @640 (MPS FP32)", "stock", "mps_fp32", 640, 1.0, "stock_calib12")
    new(SHT, "stock_r800", "stock @800 (MPS FP32)", "stock", "mps_fp32", 800, 1.5625, "stock_calib12",
        d / "exp003/poses/L0_E0-r800_c0.25_botsort", d / "exp003/cache/L0_E0-r800")
    new(SHT, "stock_r960", "stock @960 (MPS FP32)", "stock", "mps_fp32", 960, 2.25, "stock_calib12",
        d / "exp003/poses/L0_E0-r960_c0.25_botsort", d / "exp003/cache/L0_E0-r960")
    new(SHT, "stock_ncnn800", "stock @800 (NCNN FP16)", "stock", "ncnn_fp16", 800, 1.5625, "stock_calib12",
        d / "exp012/poses/L0_E0-ncnn16-800_c0.25_botsort", d / "exp012/cache/L0_E0-ncnn16-800")
    new(SHT, "M00_mps", "M00 @640 (standard labels, 4 ep; MPS FP32)", "M00", "mps_fp32", 640, 1.0, "released_all",
        d / "exp020-M00-sht/poses" / tag, d / "exp020-M00-sht/cache/L0_E0")
    new(SHT, "M00_ncnn", "M00 @640 (NCNN FP16)", "M00", "ncnn_fp16", 640, 1.0, "released_all",
        d / "exp020-ncnn/poses/L0_E0-M00_c0.25_botsort", d / "exp020-ncnn/cache/L0_E0-M00")
    old(SHT, "ML0_mps", "ML0 @640 (partial labels; MPS FP32)", "ML0", "mps_fp32", 640, 1.0, "ML0_mps_calib12")
    old(SHT, "ML0_ncnn", "ML0 @640 (NCNN FP16)", "ML0", "ncnn_fp16", 640, 1.0, "ML0_mps_calib12")
    old(SHT, "MLP_mps", "MLP @640 (partial labels + P2 head; MPS FP32)", "MLP", "mps_fp32", 640, 1.92, "released_all")
    old(SHT, "MLP_ncnn", "MLP @640 (NCNN FP16)", "MLP", "ncnn_fp16", 640, 1.92, "released_all")

    old(UB, "released", "released AlphaPose (oracle reference)", "released", "alphapose", None, None, "released_all")
    new(UB, "stock_r480", "stock @480 (MPS FP32)", "stock", "mps_fp32", 480, 0.5625, "released_all",
        d / "exp006/poses/L0-480_E0_c0.25_botsort", d / "exp006/cache/L0-480_E0")
    old(UB, "stock", "stock @640 (MPS FP32)", "stock", "mps_fp32", 640, 1.0, "released_all")
    new(UB, "stock_r960", "stock @960 (MPS FP32)", "stock", "mps_fp32", 960, 2.25, "released_all",
        d / "exp006/poses/L0-960_E0_c0.25_botsort", d / "exp006/cache/L0-960_E0")
    new(UB, "M00_mps", "M00 @640 (standard labels, 4 ep; MPS FP32)", "M00", "mps_fp32", 640, 1.0, "released_all",
        d / "exp020-M00-ub/poses" / tag, d / "exp020-M00-ub/cache/L0_E0")
    old(UB, "ML0_mps", "ML0 @640 (partial labels; MPS FP32)", "ML0", "mps_fp32", 640, 1.0, "released_all")
    old(UB, "MLP_mps", "MLP @640 (partial labels + P2 head; MPS FP32)", "MLP", "mps_fp32", 640, 1.92, "released_all")
    return arms


def arm_by_key(arms: list[Arm]) -> dict[str, Arm]:
    return {a.key: a for a in arms}


# contrasts (plan: ML0 - stock@640, ML0 - stock@800, ML0 - M00, MLP - ML0, ML0_ncnn - ML0_mps + resolution / lock pairs) ----------
def contrast_specs() -> list[dict]:
    S, U = "ShanghaiTech", "UBnormal"
    c = lambda cid, ds, a, b, text, full=False: dict(id=cid, dataset=ds, a=a, b=b, text=text, full=full)  # noqa: E731
    return [
        c("sht_ML0mps_stock", S, "ML0_mps", "stock", "ML0 - stock@640 (MPS FP32)", True),
        c("sht_ML0ncnn_stock", S, "ML0_ncnn", "stock", "ML0 (NCNN) - stock@640 (MPS)", True),
        c("sht_ML0mps_stock800", S, "ML0_mps", "stock_r800", "ML0 - stock@800 (MPS FP32)", True),
        c("sht_ML0ncnn_stockncnn800", S, "ML0_ncnn", "stock_ncnn800", "ML0 - stock@800, both NCNN FP16 (equal runtime)", True),
        c("sht_ML0mps_M00mps", S, "ML0_mps", "M00_mps", "ML0 - M00 (labels vs 4 extra epochs; MPS)", True),
        c("sht_ML0ncnn_M00ncnn", S, "ML0_ncnn", "M00_ncnn", "ML0 - M00 (NCNN FP16)", True),
        c("sht_MLPmps_ML0mps", S, "MLP_mps", "ML0_mps", "MLP - ML0 (P2 head; MPS)", True),
        c("sht_MLPncnn_ML0ncnn", S, "MLP_ncnn", "ML0_ncnn", "MLP - ML0 (P2 head; NCNN FP16)", True),
        c("sht_ML0ncnn_ML0mps", S, "ML0_ncnn", "ML0_mps", "ML0 NCNN FP16 - ML0 MPS FP32 (quantisation)", True),
        c("sht_MLPncnn_MLPmps", S, "MLP_ncnn", "MLP_mps", "MLP NCNN FP16 - MLP MPS FP32", True),
        c("sht_MLPncnn_stockncnn800", S, "MLP_ncnn", "stock_ncnn800", "MLP@640 - stock@800 (both NCNN FP16)", True),
        c("sht_stock480_stock", S, "stock_r480", "stock", "stock@480 - stock@640", True),
        c("sht_stock800_stock", S, "stock_r800", "stock", "stock@800 - stock@640", True),
        c("sht_stock960_stock", S, "stock_r960", "stock", "stock@960 - stock@640", True),
        c("sht_stockncnn800_stock800", S, "stock_ncnn800", "stock_r800", "stock@800 NCNN FP16 - MPS FP32", True),
        c("sht_M00mps_stock", S, "M00_mps", "stock", "M00 - stock@640 (4 extra epochs, standard labels)", True),
        c("ub_ML0mps_stock", U, "ML0_mps", "stock", "ML0 - stock@640", True),
        c("ub_ML0mps_stock960", U, "ML0_mps", "stock_r960", "ML0 - stock@960", True),
        c("ub_ML0mps_M00mps", U, "ML0_mps", "M00_mps", "ML0 - M00", True),
        c("ub_MLPmps_ML0mps", U, "MLP_mps", "ML0_mps", "MLP - ML0", True),
        c("ub_stock480_stock", U, "stock_r480", "stock", "stock@480 - stock@640", True),
        c("ub_stock960_stock", U, "stock_r960", "stock", "stock@960 - stock@640", True),
        c("ub_M00mps_stock", U, "M00_mps", "stock", "M00 - stock@640", True),
    ]


# =====================================================================================================================
# 2. Locked E00 inputs
# =====================================================================================================================
def load_e00(path_dir: Path = E00_DIR) -> tuple[dict, dict]:
    config = yaml.safe_load((path_dir / "config.yaml").read_text())
    summary = json.loads((path_dir / "summary.json").read_text())
    return config, summary


def split_hash(names: list[str]) -> str:
    return hashlib.sha256("\n".join(sorted(names)).encode()).hexdigest()


def locked_protocol(e00_summary: dict, arms: list[Arm]) -> dict:
    """Per dataset: quantile, hold, and the sentinel value of every distribution an arm may use (E00's numbers, not re-tuned)."""
    out = {}
    for ds in DATASETS:
        sel = e00_summary["selection"][ds]
        dists = e00_summary["sentinel"]["distributions"][ds]
        out[ds] = {"quantile": sel["quantile"], "hold": sel["hold"], "selected_on": sel["selected_on"],
                   "e00_dev_auc": sel["dev_auc"],
                   "sentinel_values": {name: v["sentinel_values"][sel["quantile"]] for name, v in dists.items()},
                   "arm_sentinel": {a.name: a.sentinel_dist for a in arms if a.dataset == ds}}
    return plain(out)


def arm_sentinel(proto: dict, arm: Arm, common: bool = False) -> float:
    dist = "released_all" if common else arm.sentinel_dist
    return float(proto[arm.dataset]["sentinel_values"][dist])


def hold_of(proto: dict, dataset: str) -> float:
    h = proto[dataset]["hold"]
    return math.inf if h == "inf" else float(h)


# =====================================================================================================================
# 3. Track statistics (part C) - pure
# =====================================================================================================================
COUNT_FIELDS = ("n_tracks", "n_poses", "n_stranded_short", "n_stranded_gappy", "n_exact_seg_len", "gaps_1", "gaps_2", "gaps_ge3",
                "gap_frames", "poses_scorable", "poses_unscorable_stranded", "poses_unscorable_initial", "poses_unscorable_post_gap",
                "n_windows")
FRAME_FIELDS = ("n_frames", "anomalous_frames", "n_normal_frames", "scorable_frames", "scorable_normal", "scorable_anomalous")


def clip_track_stats(tracks: dict, seg_len: int) -> tuple[dict, np.ndarray, np.ndarray]:
    """Counts of one clip from {person: (frames, keypoints)} + the track lengths and the within-track gap lengths (frames).

    Every pose of a track falls in exactly one class: scorable (the last pose of >= 1 complete window), in a stranded track
    (< seg_len poses, or >= seg_len poses but no complete window), initial warm-up (first seg_len - 1 poses of a scorable
    track) or post-gap re-warm-up (every other unscorable pose of a scorable track).
    """
    c = dict.fromkeys(COUNT_FIELDS, 0)
    lengths, gap_lengths = [], []
    for frames, _ in tracks.values():
        n = len(frames)
        lengths.append(n)
        c["n_tracks"] += 1
        c["n_poses"] += n
        gaps = np.diff(frames) - 1
        gaps = gaps[gaps > 0]
        gap_lengths.append(gaps)
        c["gaps_1"] += int((gaps == 1).sum())
        c["gaps_2"] += int((gaps == 2).sum())
        c["gaps_ge3"] += int((gaps >= 3).sum())
        c["gap_frames"] += int(gaps.sum())
        c["n_exact_seg_len"] += int(n == seg_len)
        starts = e00.complete_window_starts(frames, seg_len)
        c["n_windows"] += len(starts)
        if n < seg_len:
            c["n_stranded_short"] += 1
            c["poses_unscorable_stranded"] += n
        elif len(starts) == 0:
            c["n_stranded_gappy"] += 1
            c["poses_unscorable_stranded"] += n
        else:
            c["poses_scorable"] += len(starts)
            c["poses_unscorable_initial"] += seg_len - 1
            c["poses_unscorable_post_gap"] += n - len(starts) - (seg_len - 1)
    return (c, np.asarray(lengths, dtype=np.int64),
            np.concatenate(gap_lengths).astype(np.int64) if gap_lengths else np.zeros(0, np.int64))


def histogram(values: np.ndarray, edges: tuple) -> list[int]:
    return [int(((values >= lo) & (values < hi)).sum()) for lo, hi in zip(edges[:-1], edges[1:])]


def sum_fields(rows: list[dict], fields: tuple[str, ...]) -> dict:
    return {f: int(sum(r[f] for r in rows)) for f in fields}


def aggregate_track_rows(rows: list[dict], seg_len: int) -> dict:
    """Pool per-clip count rows (all fields are additive) and derive the ratios the report shows."""
    t = sum_fields(rows, COUNT_FIELDS + FRAME_FIELDS)
    poses = max(t["n_poses"], 1)
    t["stranded_short_share_of_tracks"] = t["n_stranded_short"] / max(t["n_tracks"], 1)
    t["stranded_poses_share"] = t["poses_unscorable_stranded"] / poses
    t["scorable_pose_share"] = t["poses_scorable"] / poses
    t["initial_warmup_pose_share"] = t["poses_unscorable_initial"] / poses
    t["post_gap_pose_share"] = t["poses_unscorable_post_gap"] / poses
    t["gaps_total"] = t["gaps_1"] + t["gaps_2"] + t["gaps_ge3"]
    t["gaps_per_1000_poses"] = 1000 * t["gaps_total"] / poses
    t["gaps_ge3_per_1000_poses"] = 1000 * t["gaps_ge3"] / poses
    t["coverage_all"] = t["scorable_frames"] / max(t["n_frames"], 1)
    t["coverage_normal"] = t["scorable_normal"] / t["n_normal_frames"] if t["n_normal_frames"] else float("nan")
    t["coverage_anomalous"] = t["scorable_anomalous"] / t["anomalous_frames"] if t["anomalous_frames"] else float("nan")
    t["windows_per_frame"] = t["n_windows"] / max(t["n_frames"], 1)
    t["windows_per_scorable_frame"] = t["n_windows"] / t["scorable_frames"] if t["scorable_frames"] else float("nan")
    t["tracks_per_frame"] = t["n_tracks"] / max(t["n_frames"], 1)
    t["seg_len"] = seg_len
    return t


def length_summary(lengths: np.ndarray, seg_len: int) -> dict:
    if len(lengths) == 0:
        return {"n": 0}
    q = np.quantile(lengths, [0.1, 0.25, 0.5, 0.75, 0.9])
    return {"n": int(len(lengths)), "mean": float(lengths.mean()), "p10": float(q[0]), "p25": float(q[1]), "median": float(q[2]),
            "p75": float(q[3]), "p90": float(q[4]), "max": int(lengths.max()), "share_below_seg_len": float((lengths < seg_len).mean()),
            "hist": dict(zip(LEN_NAMES, histogram(lengths, LEN_EDGES)))}


def gap_summary(gap_lengths: np.ndarray) -> dict:
    if len(gap_lengths) == 0:
        return {"n": 0, "hist": dict.fromkeys(GAP_NAMES, 0)}
    return {"n": int(len(gap_lengths)), "mean": float(gap_lengths.mean()), "median": float(np.median(gap_lengths)),
            "p90": float(np.quantile(gap_lengths, 0.9)), "max": int(gap_lengths.max()), "share_ge3": float((gap_lengths >= 3).mean()),
            "hist": dict(zip(GAP_NAMES, histogram(gap_lengths, GAP_EDGES)))}


# =====================================================================================================================
# 4. Tracker cells, replay and in-memory window tables (part D)
# =====================================================================================================================
@dataclass(frozen=True)
class TrackerCell:
    c: float = 0.25  # track_high_thresh = new_track_thresh = detector gate when two-stage is disabled
    low: float | None = None  # None: disabled (boxes below c never reach the tracker = the E00 / replay.py behaviour)
    buffer: int = 30
    match: float = 0.8

    @property
    def gate(self) -> float:
        return self.c if self.low is None else self.low

    @property
    def track_low_thresh(self) -> float:
        return min(0.1, self.c / 2) if self.low is None else self.low  # disabled: the value replay.build_tracker used

    @property
    def label(self) -> str:
        low = "off" if self.low is None else f"{self.low:.2f}"
        return f"c{self.c:.2f}|low-{low}|b{self.buffer}|m{self.match:.2f}"

    def n_changed(self) -> int:
        d = DEFAULT_CELL
        return int(self.c != d.c) + int(self.low != d.low) + int(self.buffer != d.buffer) + int(self.match != d.match)


DEFAULT_CELL = TrackerCell()
GRID = {"c": (0.20, 0.25, 0.30), "low": (None, 0.10, 0.05), "buffer": (30, 45), "match": (0.80, 0.85)}


def grid_cells() -> list[TrackerCell]:
    return [TrackerCell(c, low, b, m) for c, low, b, m in itertools.product(GRID["c"], GRID["low"], GRID["buffer"], GRID["match"])]


def build_tracker_cell(cell: TrackerCell):
    """Ultralytics default BoT-SORT yaml (no GMC / ReID), thresholds from `cell`. The default cell == replay.build_tracker(0.25)."""
    from ultralytics.trackers.track import TRACKER_MAP
    from ultralytics.utils import ROOT, YAML, IterableSimpleNamespace

    cfg = YAML.load(ROOT / "cfg/trackers/botsort.yaml")
    cfg.update(track_high_thresh=cell.c, new_track_thresh=cell.c, track_low_thresh=cell.track_low_thresh,
               track_buffer=cell.buffer, match_thresh=cell.match, gmc_method="none", with_reid=False)
    return TRACKER_MAP[cfg["tracker_type"]](IterableSimpleNamespace(**cfg))


def replay_cell(frames: list[dict], frame_ids: np.ndarray, cell: TrackerCell) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Detector-cache frames -> {track id: (frames, keypoints)} with the tracker of `cell`.

    The detector gate is `cell.gate`: with two-stage disabled it is c (exactly replay.replay_video); with two-stage enabled boxes down
    to `low` reach BoT-SORT, which may use them only to continue live tracks (new tracks need conf >= new_track_thresh = c)."""
    from edgelab.pose.replay import Detections

    tracker = build_tracker_cell(cell)
    per: dict[int, list] = {}
    for frame, frame_id in zip(frames, frame_ids):
        keep = frame["conf"] >= cell.gate
        kept = {k: v[keep] for k, v in frame.items()}
        for row in tracker.update(Detections(kept["boxes"], kept["conf"], kept["cls"]), None):
            per.setdefault(int(row[4]), []).append((int(frame_id), kept["keypoints"][int(row[7])].astype(float)))
    return {pid: (np.array([f for f, _ in items], dtype=np.int64), np.stack([k for _, k in items])) for pid, items in sorted(per.items())}


def tracks_from_poses(poses: list) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    per: dict[int, list] = {}
    for p in sorted(poses, key=lambda q: (q.track_id, q.frame)):
        per.setdefault(int(p.track_id), []).append((int(p.frame), np.asarray(p.keypoints, dtype=float)))
    return {pid: (np.array([f for f, _ in v], dtype=np.int64), np.stack([k for _, k in v])) for pid, v in per.items()}


def track_partition(tracks: dict) -> set:
    """Id-agnostic identity of a clip's tracks: the set of (frames, exact keypoint bytes)."""
    return {(tuple(int(f) for f in frames), hashlib.blake2b(np.ascontiguousarray(kp, dtype=np.float64).tobytes(), digest_size=12).hexdigest())
            for frames, kp in tracks.values()}


def track_key(frames: np.ndarray, keypoints: np.ndarray) -> bytes:
    h = hashlib.blake2b(digest_size=16)
    h.update(np.ascontiguousarray(frames, dtype=np.int64).tobytes())
    h.update(np.ascontiguousarray(keypoints, dtype=np.float64).tobytes())
    return h.digest()


def window_table_from_tracks(tracks_by_clip: list[dict], seg_len: int, scorer, cache: dict | None = None) -> dict[str, np.ndarray]:
    """Same table as `e00.build_window_table` but from in-memory tracks; identical tracks are scored once (`cache`: track hash -> normality).

    Window scoring is a pure function of the window's poses, so reusing the normality of an identical track across tracker cells is exact."""
    cache = {} if cache is None else cache
    pending: dict[bytes, np.ndarray] = {}
    for tracks in tracks_by_clip:
        for frames, kp in tracks.values():
            starts = e00.complete_window_starts(frames, seg_len)
            if len(starts) == 0:
                continue
            key = track_key(frames, kp)
            if key not in cache and key not in pending:
                pending[key] = e00.gather_windows(kp, starts, seg_len)
    keys = list(pending)
    for i in range(0, len(keys), 512):  # score in large batches
        chunk = keys[i:i + 512]
        norm = scorer.score(np.concatenate([pending[k] for k in chunk]))
        offset = 0
        for k in chunk:
            n = len(pending[k])
            cache[k] = norm[offset:offset + n]
            offset += n
    cols: dict[str, list] = {c: [] for c in e00.WINDOW_COLS if c != "normality"}
    norms = []
    n_tracks, n_poses = np.zeros(len(tracks_by_clip), dtype=np.int64), np.zeros(len(tracks_by_clip), dtype=np.int64)
    for ci, tracks in enumerate(tracks_by_clip):
        for pid, (frames, kp) in tracks.items():
            n_tracks[ci] += 1
            n_poses[ci] += len(frames)
            starts = e00.complete_window_starts(frames, seg_len)
            if len(starts) == 0:
                continue
            cols["clip_idx"].append(np.full(len(starts), ci, dtype=np.int32))
            cols["person"].append(np.full(len(starts), pid, dtype=np.int32))
            cols["start_idx"].append(starts.astype(np.int32))
            cols["n_track"].append(np.full(len(starts), len(frames), dtype=np.int32))
            cols["start_frame"].append(frames[starts].astype(np.int32))
            cols["last_frame"].append(frames[starts + seg_len - 1].astype(np.int32))
            cols["in_offline"].append(e00.stgnf_offline_window_mask(starts, len(frames), seg_len))
            norms.append(cache[track_key(frames, kp)])
    table = {c: (np.concatenate(v) if v else np.zeros(0, dtype=bool if c == "in_offline" else np.int32)) for c, v in cols.items()}
    table["normality"] = np.concatenate(norms) if norms else np.zeros(0)
    table["clip_n_tracks"], table["clip_n_poses"] = n_tracks, n_poses
    return table


# =====================================================================================================================
# 5. Selection rules (pure; plan.md section 3)
# =====================================================================================================================
def select_tracker_cell(dev_auc: dict[str, float], cells: dict[str, TrackerCell], tie: float = TIE_BAND) -> dict:
    """T1: highest dev AUC; cells within `tie` of it are tied -> fewest changed factors, then smaller |c - 0.25|, then low disabled."""
    best = max(dev_auc.values())
    pool = [k for k, v in dev_auc.items() if best - v <= tie]
    pool.sort(key=lambda k: (cells[k].n_changed(), abs(cells[k].c - DEFAULT_CELL.c), 0 if cells[k].low is None else 1, k))
    argmax = max(dev_auc, key=lambda k: (dev_auc[k], -cells[k].n_changed()))
    return {"winner": pool[0], "best": best, "argmax": argmax, "tied": sorted(pool), "winner_auc": dev_auc[pool[0]]}


def adopt_tracker(winner: str, default: str, primary: dict, vetoes: dict[str, dict], min_delta: float = TRACKER_MIN_DELTA) -> dict:
    """T2. `primary` / each veto: {"delta": winner - default, "ci95": [lo, hi]} on dev. Adopt only if (a) delta >= min_delta,
    (b) primary CI excludes 0 (lower > 0), (c) no veto CI lies entirely below 0."""
    if winner == default:
        return {"adopted": False, "locked": default, "reason": "the default cell is the (tie-resolved) dev winner"}
    reasons = []
    if primary["delta"] < min_delta:
        reasons.append(f"delta {primary['delta']:+.2f} < +{min_delta:.1f} pt")
    if not primary["ci95"][0] > 0:
        reasons.append(f"primary dev CI [{primary['ci95'][0]:+.2f}, {primary['ci95'][1]:+.2f}] does not exclude 0")
    harmed = [k for k, v in vetoes.items() if v["ci95"][1] < 0]
    if harmed:
        reasons.append("significantly harmful on " + ", ".join(harmed))
    ok = not reasons
    return {"adopted": ok, "locked": winner if ok else default, "reason": "all T2 conditions hold" if ok else "; ".join(reasons)}


def verify_tracker(locked: str, default: str, confirmation_primary: dict) -> dict:
    """T3: a non-default lock whose confirmation CI (locked - default, primary series) lies entirely below 0 fails verification."""
    if locked == default:
        return {"failed": False, "final": default, "reason": "default locked; nothing to verify"}
    failed = confirmation_primary["ci95"][1] < 0
    return {"failed": failed, "final": default if failed else locked,
            "reason": ("confirmation CI entirely below 0 -> revert to default" if failed else
                       f"confirmation delta {confirmation_primary['delta']:+.2f} [{confirmation_primary['ci95'][0]:+.2f}, {confirmation_primary['ci95'][1]:+.2f}]")}


def select_frontend(candidates: list[dict], dev_auc: dict[str, float], pair: dict[tuple[str, str], dict], ub_harm: dict[str, bool],
                    margin: float = FRONTEND_MARGIN) -> dict:
    """F1-F3. candidates sorted by cost ascending: {name, cost}. pair[(costly, cheaper)] = {"delta", "ci95"} on dev S4 (SHT).
    A costlier candidate replaces the incumbent only if dev AUC >= incumbent + margin and CI lower > 0 and no UB harm veto."""
    order = sorted(candidates, key=lambda c: c["cost"])
    incumbent, steps = order[0]["name"], []
    for cand in order[1:]:
        name = cand["name"]
        p = pair[(name, incumbent)]
        gain = dev_auc[name] - dev_auc[incumbent]
        ok = gain >= margin and p["ci95"][0] > 0
        vetoed = ok and ub_harm.get(name, False)
        steps.append({"candidate": name, "incumbent": incumbent, "dev_gain": gain, "ci95": p["ci95"], "meets_margin_and_ci": bool(ok),
                      "ub_veto": bool(vetoed), "replaces": bool(ok and not vetoed)})
        if ok and not vetoed:
            incumbent = name
    return {"locked": incumbent, "steps": steps}


# =====================================================================================================================
# 6. Recall by height + miss decomposition (part B) - pure matching
# =====================================================================================================================
LOW_BUCKETS = (("0.10-0.25", 0.10, 0.25), ("0.05-0.10", 0.05, 0.10), ("0.01-0.05", 0.01, 0.05))
BUCKET_NAMES = [b[0] for b in LOW_BUCKETS] + ["none"]
KINDS = ("never_seen", "intermittent")


def bucket_of_best_conf(best: float) -> str:
    """Name of the confidence band holding the best cached box (IoU >= 0.3) of a missed person-frame ('none': no box >= 0.01)."""
    if best is None or not np.isfinite(best):
        return "none"
    for name, lo, hi in LOW_BUCKETS:
        if lo <= best < hi:
            return name
    return "none"


def best_conf_per_entry(entries: list, boxes_by_frame: dict, iou_min: float = 0.3) -> np.ndarray:
    """For each reference (frame, height, box): max conf over cached boxes (conf >= 0.01) with IoU >= iou_min on that frame; nan if none."""
    from edgelab.analysis.exp009 import iou_with_boxes

    out = np.full(len(entries), np.nan)
    for i, (frame, _, box) in enumerate(entries):
        found = boxes_by_frame.get(frame)
        if found is None:
            continue
        boxes, conf = found
        if len(boxes) == 0:
            continue
        m = iou_with_boxes(box, boxes) >= iou_min
        if m.any():
            out[i] = conf[m].max()
    return out


def hit_flags(best: np.ndarray, conf: float = 0.25) -> np.ndarray:
    """EXP-011 matcher: found = a cached box with conf >= 0.25 and IoU >= 0.3, i.e. best >= 0.25."""
    return np.nan_to_num(best, nan=-1.0) >= conf


def miss_breakdown(tracks: dict, best: dict, conf: float = 0.25) -> dict:
    """EXP-022 never-seen / intermittent split of missed person-frames x confidence band of the best cached box x height bin.

    tracks: video -> {pid: [(frame, height, box)]}; best: video -> {pid: best-conf array}.
    """
    from edgelab.analysis.exp011 import BIN_NAMES, height_bin

    out = {"missed": 0, "by_kind": {k: {b: 0 for b in BUCKET_NAMES} for k in KINDS},
           "by_bin": {n: {b: 0 for b in BUCKET_NAMES} for n in BIN_NAMES}}
    for video, people in tracks.items():
        for pid, entries in people.items():
            b = best[video][pid]
            hit = hit_flags(b, conf)
            kind = "never_seen" if hit.sum() == 0 else ("intermittent" if (~hit).any() else None)
            if kind is None:
                continue
            for (_, height, _), h, bc in zip(entries, hit, b):
                if h:
                    continue
                name = bucket_of_best_conf(bc)
                out["missed"] += 1
                out["by_kind"][kind][name] += 1
                out["by_bin"][BIN_NAMES[height_bin(height)]][name] += 1
    m = max(out["missed"], 1)
    in_band = lambda d: sum(d[n] for n in ("0.10-0.25", "0.05-0.10"))  # noqa: E731  (0.05 <= conf < 0.25)
    out["in_band_total"] = in_band(out["by_kind"]["never_seen"]) + in_band(out["by_kind"]["intermittent"])
    out["share_in_band"] = out["in_band_total"] / m
    out["two_stage_recoverable"] = in_band(out["by_kind"]["intermittent"])  # a low box can only continue a live track
    out["share_two_stage_recoverable"] = out["two_stage_recoverable"] / m
    out["gate_only_recoverable"] = in_band(out["by_kind"]["never_seen"])  # needs a lower gate: spawning tracks from low boxes
    out["share_gate_only_recoverable"] = out["gate_only_recoverable"] / m
    out["share_below_0.05_or_none"] = 1 - out["share_in_band"]
    return out


def bootstrap_count_diff(counts_a: np.ndarray, counts_b: np.ndarray, bins: list[int], n: int = N_BOOT_RECALL, seed: int = 0) -> dict:
    """Paired video bootstrap of pooled recall(a) - recall(b) over the height bins `bins`. counts: (videos, bins, 2) = persons, found."""
    def rec(c, idx):
        s = c[idx][:, bins].sum(axis=(0, 1))
        return s[1] / s[0] if s[0] else float("nan")

    rng = np.random.default_rng(seed)
    nv = counts_a.shape[0]
    full = np.arange(nv)
    diffs = []
    for _ in range(n):
        idx = rng.integers(0, nv, nv)
        d = rec(counts_a, idx) - rec(counts_b, idx)
        if np.isfinite(d):
            diffs.append(d)
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    return {"delta": float(rec(counts_a, full) - rec(counts_b, full)), "ci95": [float(lo), float(hi)]}


# =====================================================================================================================
# 7. Process pool (fork) - globals are inherited, never pickled
# =====================================================================================================================
_G: dict = {}


def pmap(fn, items: list, procs: int) -> list:
    if procs <= 1 or len(items) <= 1:
        return [fn(x) for x in items]
    with mp.get_context("fork").Pool(min(procs, len(items))) as pool:
        return pool.map(fn, items, chunksize=1)


def _boot_worker(job: tuple):
    kind, ds, a, b, step, subset, n = job
    labels = _G["labels"][ds]
    idx = {"all": None, "dev": _G["dev_idx"][ds], "confirmation": _G["conf_idx"][ds]}[subset]
    sa = _G["series"][(ds, a, step)]
    if kind == "ci":
        return job, e00.bootstrap_ci(labels, sa, idx, n)
    return job, e00.paired_diff(labels, sa, _G["series"][(ds, b, step)], idx, n)  # = AUC(a) - AUC(b)


def _replay_worker(cell: TrackerCell):
    frames_by_clip, ids_by_clip = _G["det_frames"], _G["det_ids"]
    return cell.label, [replay_cell(f, i, cell) for f, i in zip(frames_by_clip, ids_by_clip)]


# =====================================================================================================================
# 8. Lock (config.yaml) and verification (R1)
# =====================================================================================================================
def short(path: Path) -> str:
    return str(path).replace(str(DATA_ROOT), "~/thesis-data")


def expand(path: str) -> Path:
    return Path(path.replace("~/thesis-data", str(DATA_ROOT)))


def build_config(log=print) -> dict:
    from edgelab.analysis.exp011 import BIN_NAMES

    e00cfg, e00sum = load_e00()
    arms = build_arms()
    proto = locked_protocol(e00sum, arms)
    frontends, cache_seen = [], {}
    for a in arms:
        entry = {"key": a.key, "dataset": a.dataset, "label": a.label, "group": a.group, "runtime": a.runtime, "imgsz": a.imgsz,
                 "relative_flops_vs_stock640": a.cost, "sentinel_distribution": a.sentinel_dist,
                 "sentinel_value": arm_sentinel(proto, a), "in_e00": a.in_e00,
                 "pose_dir": short(a.pose_dir), "cache_dir": None if a.cache_dir is None else short(a.cache_dir)}
        log(f"hashing pose folder {a.key} ...")
        pm = e00.tree_manifest(a.pose_dir, f"*{e00.SUFFIX}")
        entry["pose_dir_files"], entry["pose_dir_tree_sha256"] = pm["n_files"], pm["tree_sha256"]
        if a.cache_dir is not None:
            ck = short(a.cache_dir)
            if ck not in cache_seen:
                log(f"hashing cache {ck} ...")
                cm = e00.tree_manifest(a.cache_dir)
                first = np.load(sorted(a.cache_dir.glob("*.npz"))[0])
                cache_seen[ck] = {"n_videos": cm["n_files"], "total_bytes": cm["total_bytes"], "tree_sha256": cm["tree_sha256"],
                                  "meta_first_video": json.loads(str(first["meta"])), "used_by": []}
            cache_seen[ck]["used_by"].append(a.key)
            entry["cache_tree_sha256"] = cache_seen[ck]["tree_sha256"]
        old = e00cfg["detector_caches"].get(short(a.cache_dir)) if a.cache_dir else None
        entry["matches_e00_cache_hash"] = None if old is None else bool(old["tree_sha256"] == entry["cache_tree_sha256"])
        frontends.append(entry)
    splits = {ds: {name: {"n": len(e00cfg["datasets"][ds]["splits"][name]), "sha256": split_hash(e00cfg["datasets"][ds]["splits"][name])}
                   for name in ("test_dev", "test_confirmation")} for ds in DATASETS}
    cells = grid_cells()
    config = {
        "experiment": "Q2-P1-E01_quality",
        "question": "What accuracy can the frozen frontend/scorer reach before CPU constraints? Compare stock and existing ML0; inspect missed small "
                    "people and tracking gaps.",
        "training": False, "seed": SEED, "bootstrap": {"unit": "video", "resamples": N_BOOT, "resamples_recall": N_BOOT_RECALL, "seed": 0},
        "depends_on_E00": {"config": "Q2-P1-E00_audit/config.yaml", "config_sha256": e00.sha256_file(E00_DIR / "config.yaml"),
                           "summary_sha256": e00.sha256_file(E00_DIR / "summary.json"),
                           "report_sha256": e00.sha256_file(E00_DIR / "report.md")},
        "splits_from_E00": splits,
        "scoring_protocol": {"landing": "completion", "smoother": "causal half-Gaussian (E00)", "steps_reported": list(CHAIN),
                             "offline_S0": "middle landing, two-sided Gaussian, global most-normal fill, last window omitted (paper protocol)",
                             "causal_S3_S4": "completion landing, causal half-Gaussian, train-calibrated sentinel, hold from E00",
                             "per_dataset": proto,
                             "sentinel_sensitivity": "every arm is also scored with the common released_all sentinel"},
        "scorers": {"ShanghaiTech": "ShanghaiTech_85_9.tar", "UBnormal": "UBnormal_unsupervised_71_8.tar",
                    "sha256_from_E00_config": {k: e00cfg["frozen_artifacts"][k]["sha256"] for k in
                                               ("ShanghaiTech_85_9.tar", "UBnormal_unsupervised_71_8.tar", "yolov8n-pose.pt", "ML0.pt", "M00.pt",
                                                "MLP.pt", "ML0-ncnn.bin", "M00-ncnn.bin", "MLP-ncnn.bin", "yolov8n-pose-ncnn.bin")}},
        "frontends": frontends,
        "resolution_caches": {
            "ShanghaiTech": {"stock_480": "exp003/cache/L0_E0-r480", "stock_640": "exp003/cache/L0_E0", "stock_800": "exp003/cache/L0_E0-r800",
                             "stock_960": "exp003/cache/L0_E0-r960", "stock_ncnn16_800": "exp012/cache/L0_E0-ncnn16-800"},
            "UBnormal": {"stock_480": "exp006/cache/L0-480_E0", "stock_640": "exp006/cache/L0_E0", "stock_960": "exp006/cache/L0-960_E0",
                         "note": "no stock@800 cache exists for UBnormal: stock@960 is the higher-resolution comparator"}},
        "detector_caches": cache_seen,
        "contrasts": [{k: c[k] for k in ("id", "dataset", "a", "b", "text")} for c in contrast_specs()],
        "recall": {"reference": "released AlphaPose test poses (ShanghaiTech)", "height_bins_px": list(BIN_NAMES),
                   "matcher": "cached box conf >= 0.25 and IoU >= 0.3 with the keypoint box on the same frame (EXP-011 / EXP-022)",
                   "low_confidence_buckets_best_box": [b[0] for b in LOW_BUCKETS], "two_stage_band": [LOW_CONF_LO, LOW_CONF_HI],
                   "decomposition_arms": ["stock", "stock_r800", "ML0_ncnn", "MLP_ncnn"]},
        "tracking_gap": {"stranded_short": "poses < seg_len (SHT 24, UB 16): no STG-NF window", "bridged_gap_frames": [1, 2],
                         "breaking_gap_frames": ">= 3 (needs seg_len - 1 further poses to re-warm up)",
                         "length_bins": LEN_NAMES, "gap_bins": GAP_NAMES},
        "tracker_sweep": {
            "tracker": "botsort (Ultralytics 8.4.x default yaml, GMC none, ReID off); IDs per tracker instance",
            "default_cell": DEFAULT_CELL.label, "n_cells": len(cells), "cells": [c.label for c in cells],
            "grid": {"c (track_high_thresh = new_track_thresh = gate)": list(GRID["c"]),
                     "two_stage low (track_low_thresh; None = disabled, pre-filter at c as in replay.py)": [None if x is None else x for x in GRID["low"]],
                     "track_buffer": list(GRID["buffer"]), "match_thresh": list(GRID["match"])},
            "dev_only": True,
            "primary_series": "ShanghaiTech/ML0_ncnn", "veto_series": ["ShanghaiTech/stock", "UBnormal/ML0_mps", "UBnormal/stock"],
            "verification_series": ["ShanghaiTech/ML0_ncnn", "ShanghaiTech/stock", "ShanghaiTech/MLP_ncnn", "UBnormal/ML0_mps", "UBnormal/stock"],
            "rules": {"tie_band_pt": TIE_BAND, "min_delta_pt": TRACKER_MIN_DELTA}},
        "frontend_lock": {"candidates": {"ML0_ncnn": 1.0, "stock_ncnn800": 1.5625, "MLP_ncnn": 1.92}, "margin_pt": FRONTEND_MARGIN,
                          "ub_veto": "MLP_mps - ML0_mps dev S4 on UBnormal with paired CI entirely below 0"},
        "rules": ["QA1", "QA2", "QA3", "QA4", "QA5", "QA6", "QA7", "QB1", "QB2", "QB3", "QB4", "R1", "R2", "R3", "R4", "R5", "R6", "T1", "T2", "T3", "F2"],
    }
    return plain(config)


def cmd_lock(log=print) -> None:
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    config = build_config(log)
    header = ("# Q2-P1-E01 locked configuration (generated by `python -m edgelab.analysis.q2_p1_e01 lock` BEFORE `run`; do not hand-edit).\n"
              "# Full SHA-256 / tree hashes of every cache and pose folder; split lists are those of Q2-P1-E00 (referenced by hash).\n")
    (EXP_DIR / "config.yaml").write_text(header + yaml.safe_dump(config, sort_keys=False, width=140))
    print(f"wrote {EXP_DIR / 'config.yaml'}")


def verify_lock(config: dict, log=print) -> dict:
    """R1: E00 config / summary hashes, split lists, every cache and pose-folder tree hash."""
    problems, n = [], 0
    e00cfg, _ = load_e00()
    dep = config["depends_on_E00"]
    for key, path in (("config_sha256", "config.yaml"), ("summary_sha256", "summary.json"), ("report_sha256", "report.md")):
        n += 1
        if e00.sha256_file(E00_DIR / path) != dep[key]:
            problems.append(f"E00 {path} changed")
    for ds, parts in config["splits_from_E00"].items():
        for name, info in parts.items():
            n += 1
            got = e00cfg["datasets"][ds]["splits"][name]
            if split_hash(got) != info["sha256"] or len(got) != info["n"]:
                problems.append(f"split {ds}/{name} differs from the lock")
    for key, info in config["detector_caches"].items():
        n += 1
        folder = expand(key)
        if not folder.exists() or e00.tree_manifest(folder)["tree_sha256"] != info["tree_sha256"]:
            problems.append(f"cache {key}: tree hash changed or missing")
    for fe in config["frontends"]:
        n += 1
        folder = expand(fe["pose_dir"])
        if not folder.exists() or e00.tree_manifest(folder, f"*{e00.SUFFIX}")["tree_sha256"] != fe["pose_dir_tree_sha256"]:
            problems.append(f"pose folder {fe['key']}: tree hash changed or missing")
    for name, sha in config["scorers"]["sha256_from_E00_config"].items():
        n += 1
        if e00cfg["frozen_artifacts"][name]["sha256"] != sha:
            problems.append(f"E00 frozen artefact {name} hash differs")
    return {"n_checked": n, "problems": problems, "passes": not problems}


# =====================================================================================================================
# 9. Evaluation of a set of tracks (arms and tracker cells use the same code)
# =====================================================================================================================
@dataclass
class Ctx:
    e00cfg: dict
    e00sum: dict
    config: dict
    proto: dict
    arms: list[Arm]
    scorers: dict
    names: dict
    labels: dict
    dev_idx: dict
    conf_idx: dict
    procs: int
    log: object


def build_ctx(procs: int, log=print) -> Ctx:
    e00cfg, e00sum = load_e00()
    config = yaml.safe_load((EXP_DIR / "config.yaml").read_text())
    arms = build_arms()
    names, labels, dev_idx, conf_idx = {}, {}, {}, {}
    for ds in DATASETS:
        names[ds], labels[ds] = e00.dataset_clips(ds)
        pos = {c: i for i, c in enumerate(names[ds])}
        dev_idx[ds] = [pos[c] for c in e00cfg["datasets"][ds]["splits"]["test_dev"]]
        conf_idx[ds] = [pos[c] for c in e00cfg["datasets"][ds]["splits"]["test_confirmation"]]
    return Ctx(e00cfg, e00sum, config, locked_protocol(e00sum, arms), arms, {ds: e00.Scorer(ds) for ds in DATASETS}, names, labels,
               dev_idx, conf_idx, procs, log)


def read_tracks(pose_dir: Path, dataset: str, clips: list[str]) -> list[dict]:
    files = e00.index_pose_dir(pose_dir, dataset)
    return [e00.read_pose_clip(files[c]) if c in files else {} for c in clips]


def auc_subsets(labels: list, scores: list, dev_idx: list[int], conf_idx: list[int]) -> dict:
    f = lambda idx: 100 * e00.pooled_auc(labels, scores, idx) if (idx is None or len(idx)) else float("nan")  # noqa: E731
    return {"all": f(None), "dev": f(dev_idx), "confirmation": f(conf_idx)}


def evaluate_tracks(tracks_by_clip: list[dict], table: dict, clips: list[str], labels: list, dataset: str, sentinel: float, hold: float,
                    dev_idx: list[int], conf_idx: list[int], with_s0: bool = True, extra_sentinels: dict | None = None) -> dict:
    seg = SEG_LENS[dataset]
    stub = e00.Source("e01", dataset, Path("."), None, None, None, "")
    d = e00.SourceData(stub, clips, labels, table, seg, dev_idx, conf_idx)
    raw = d.raw_per_frame("completion", "all")
    series: dict[str, list] = {}
    series["S3"], st3 = e00.deliver(raw, sentinel, 0)
    series["S4"], st4 = (series["S3"], st3) if hold == 0 else e00.deliver(raw, sentinel, hold)
    if with_s0:
        series["S0"], _ = d.step_scores("S0")
    auc = {s: auc_subsets(labels, v, dev_idx, conf_idx) for s, v in series.items()}
    extra = {name: auc_subsets(labels, e00.deliver(raw, v, hold)[0], dev_idx, conf_idx) for name, v in (extra_sentinels or {}).items()}
    dev_set = set(dev_idx)
    rows, all_lengths, all_gaps = [], [], []
    for i, clip in enumerate(clips):
        stats, lengths, gaps = clip_track_stats(tracks_by_clip[i], seg)
        all_lengths.append(lengths)
        all_gaps.append(gaps)
        lab, fresh = labels[i], np.isfinite(raw[i])
        row = {"video": clip, "split": "dev" if i in dev_set else "confirmation", **stats,
               "n_frames": len(lab), "anomalous_frames": int(lab.sum()), "n_normal_frames": int((lab == 0).sum()),
               "scorable_frames": int(fresh.sum()), "scorable_normal": int(fresh[lab == 0].sum()),
               "scorable_anomalous": int(fresh[lab == 1].sum()),
               "track_len_mean": float(lengths.mean()) if len(lengths) else float("nan"),
               "track_len_median": float(np.median(lengths)) if len(lengths) else float("nan")}
        for s, v in series.items():
            a = e00.clip_auc(lab, v[i])
            row[f"auc_{s}"] = float("nan") if math.isnan(a) else 100 * a
        rows.append(row)
    if sum(r["n_windows"] for r in rows) != len(table["normality"]):
        raise AssertionError("per-clip window counts do not add up to the window table")
    agg = {"all": aggregate_track_rows(rows, seg)}
    for split in ("dev", "confirmation"):
        part = [r for r in rows if r["split"] == split]
        if part:
            agg[split] = aggregate_track_rows(part, seg)
    lengths, gaps = np.concatenate(all_lengths), np.concatenate(all_gaps)
    return {"series": series, "auc": auc, "auc_extra": extra, "coverage": e00.coverage_fractions(raw, labels), "delivery": st4, "rows": rows,
            "agg": agg, "length": length_summary(lengths, seg), "gap": gap_summary(gaps), "_d": d, "n_windows": len(table["normality"])}


def arm_table(arm: Arm, clips: list[str], scorer, log) -> dict:
    if arm.in_e00:
        stub = e00.Source(arm.name, arm.dataset, arm.pose_dir, arm.cache_dir, None, None, "")
        return e00.load_or_build_table(stub, clips, scorer, log)
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    path = WORK_DIR / f"windows__{arm.dataset}__{arm.name}.npz"
    stamp = json.dumps({"pose": e00.dir_fingerprint(arm.pose_dir), "ckpt": e00.sha256_file(scorer.checkpoint)[:16], "seg_len": scorer.seg_len, "v": 2})
    if path.exists():
        z = np.load(path, allow_pickle=False)
        if str(z["stamp"]) == stamp:
            return {k: z[k] for k in z.files if k != "stamp"}
    log(f"  scoring all complete windows of {arm.key} ...")
    t0 = time.perf_counter()
    table = e00.build_window_table(e00.index_pose_dir(arm.pose_dir, arm.dataset), clips, scorer, log)
    log(f"    {len(table['normality'])} windows in {time.perf_counter() - t0:.0f}s")
    np.savez_compressed(path, stamp=np.array(stamp), **table)
    return table


# ---- series persistence (bootstrap stage reads them) --------------------------------------------------------------------
def save_series(series: dict, path: Path) -> None:
    flat = {f"{ds}|{name}|{step}": np.concatenate(v) for (ds, name, step), v in series.items()}
    np.savez_compressed(path, **flat)


def load_series(path: Path, labels: dict) -> dict:
    z = np.load(path)
    out = {}
    for k in z.files:
        ds, name, step = k.split("|")
        offs = np.concatenate([[0], np.cumsum([len(x) for x in labels[ds]])])
        out[(ds, name, step)] = [z[k][offs[i]:offs[i + 1]] for i in range(len(labels[ds]))]
    return out


# =====================================================================================================================
# 10. Stages
# =====================================================================================================================
def cached_stage(name: str, fn, force: set[str], stamp: str):
    path = WORK_DIR / "stages" / f"e01__{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and name not in force and "all" not in force:
        cached = json.loads(path.read_text())
        if cached.get("_stamp") == stamp:
            print(f"[{now()}] stage {name}: cached")
            return cached["result"]
    t0 = time.perf_counter()
    result = plain(fn())
    path.write_text(json.dumps({"_stamp": stamp, "result": result}))
    print(f"[{now()}] stage {name}: done in {time.perf_counter() - t0:.0f}s")
    return result


# ---- A + C: every arm ----------------------------------------------------------------------------------------------------
def stage_arms(ctx: Ctx) -> dict:
    results, series = {}, {}
    for arm in ctx.arms:
        ds = arm.dataset
        ctx.log(f"[{now()}] arm {arm.key}")
        names, labels = ctx.names[ds], ctx.labels[ds]
        table = arm_table(arm, names, ctx.scorers[ds], ctx.log)
        tracks = read_tracks(arm.pose_dir, ds, names)
        sentinel, hold = arm_sentinel(ctx.proto, arm), hold_of(ctx.proto, ds)
        ev = evaluate_tracks(tracks, table, names, labels, ds, sentinel, hold, ctx.dev_idx[ds], ctx.conf_idx[ds], True,
                             {"common_released_all_sentinel": arm_sentinel(ctx.proto, arm, True)})
        d = ev.pop("_d")
        for step, v in ev.pop("series").items():
            series[(ds, arm.name, step)] = v
        causal = e00.causality_tests(d, sentinel)
        e00_steps = ctx.e00sum["decomposition"].get(arm.key, {}).get("steps")
        parity = None
        if e00_steps is not None:
            parity = {s: {k: ev["auc"][s][k] - e00_steps[s]["auc"][k] for k in ("all", "dev", "confirmation")} for s in CHAIN}
        results[arm.key] = {"dataset": ds, "name": arm.name, "label": arm.label, "group": arm.group, "runtime": arm.runtime,
                            "imgsz": arm.imgsz, "cost": arm.cost, "sentinel_dist": arm.sentinel_dist, "sentinel": sentinel,
                            "hold": "inf" if math.isinf(hold) else hold, **ev, "causality": causal, "e00_parity": parity}
    save_series(series, WORK_DIR / "series.npz")
    return results


# ---- A: bootstraps -------------------------------------------------------------------------------------------------------
def stage_boot(ctx: Ctx) -> dict:
    _G.update(labels=ctx.labels, dev_idx=ctx.dev_idx, conf_idx=ctx.conf_idx, series=load_series(WORK_DIR / "series.npz", ctx.labels))
    jobs = []
    for arm in ctx.arms:
        jobs.append(("ci", arm.dataset, arm.name, None, "S0", "all", N_BOOT))
        jobs += [("ci", arm.dataset, arm.name, None, "S4", sub, N_BOOT) for sub in ("all", "dev", "confirmation")]
    for c in contrast_specs():
        jobs.append(("diff", c["dataset"], c["a"], c["b"], "S0", "all", N_BOOT))
        jobs += [("diff", c["dataset"], c["a"], c["b"], "S4", sub, N_BOOT) for sub in ("all", "dev", "confirmation")]
    ctx.log(f"[{now()}] bootstrapping {len(jobs)} jobs on {ctx.procs} processes")
    done = pmap(_boot_worker, jobs, ctx.procs)
    ci, diff = {}, {}
    for (kind, ds, a, b, step, sub, _), res in done:
        if kind == "ci":
            ci[f"{ds}/{a}|{step}|{sub}"] = res
        else:
            diff[f"{ds}|{a}|{b}|{step}|{sub}"] = res
    out = {"ci": ci, "diff": {}}
    for c in contrast_specs():
        out["diff"][c["id"]] = {"dataset": c["dataset"], "a": c["a"], "b": c["b"], "text": c["text"],
                                **{f"{step}|{sub}": diff[f"{c['dataset']}|{c['a']}|{c['b']}|{step}|{sub}"]
                                   for step, sub in (("S0", "all"), ("S4", "all"), ("S4", "dev"), ("S4", "confirmation"))}}
    _G.clear()
    return out


# ---- B: recall by height, miss decomposition, low-confidence boxes ---------------------------------------------------------
DECOMP_ARMS = ("stock", "stock_r800", "ML0_ncnn", "MLP_ncnn")
RECALL_CONTRASTS = (("ML0_ncnn", "stock"), ("ML0_ncnn", "stock_ncnn800"), ("ML0_mps", "stock_r800"), ("ML0_ncnn", "M00_ncnn"),
                    ("MLP_ncnn", "ML0_ncnn"), ("ML0_ncnn", "ML0_mps"), ("ML0_mps", "M00_mps"), ("stock_r800", "stock"))
BIN_GROUPS = {"<48": [0, 1], "48-96": [2, 3], ">=96": [4, 5], "all": [0, 1, 2, 3, 4, 5]}


def load_boxes(cache_file: Path) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    from edgelab.pose.detcache import load_detections

    frames, data = load_detections(cache_file)
    return {int(f): (fr["boxes"], fr["conf"]) for f, fr in zip(data["frame_ids"], frames)}


def stage_recall(ctx: Ctx) -> dict:
    from edgelab.analysis import exp022
    from edgelab.analysis.exp011 import BIN_NAMES, height_bin, recall, wilson

    tracks = exp022.reference_tracks()
    videos = sorted(tracks)
    pos = {c: i for i, c in enumerate(ctx.names["ShanghaiTech"])}
    split_of = {c: ("dev" if i in set(ctx.dev_idx["ShanghaiTech"]) else "confirmation") for c, i in pos.items()}
    n_ref = sum(len(e) for p in tracks.values() for e in p.values())
    ctx.log(f"[{now()}] recall: {len(videos)} reference videos, {n_ref} reference person-frames")
    out = {"n_reference_person_frames": n_ref, "n_reference_tracks": sum(len(p) for p in tracks.values()), "arms": {}, "contrasts": {},
           "parity_with_exp022_matcher": None}
    counts_by_arm = {}
    for arm in [a for a in ctx.arms if a.dataset == "ShanghaiTech" and a.cache_dir is not None]:
        ctx.log(f"[{now()}]   recall {arm.name}")
        best = {}
        for video in videos:
            boxes = load_boxes(arm.cache_dir / f"{video}.npz")
            best[video] = {pid: best_conf_per_entry(entries, boxes) for pid, entries in tracks[video].items()}
        flags = {v: {pid: hit_flags(b) for pid, b in best[v].items()} for v in best}
        counts = np.zeros((len(videos), len(BIN_NAMES), 2), dtype=np.int64)
        for vi, video in enumerate(videos):
            for pid, entries in tracks[video].items():
                for (_, height, _), hit in zip(entries, flags[video][pid]):
                    counts[vi, height_bin(height), 0] += 1
                    counts[vi, height_bin(height), 1] += int(hit)
        counts_by_arm[arm.name] = counts

        def table(mask) -> dict:
            s = counts[mask].sum(axis=0)
            t = {n: {"persons": int(s[i, 0]), "found": int(s[i, 1])} for i, n in enumerate(BIN_NAMES)}
            t["all"] = {"persons": int(s[:, 0].sum()), "found": int(s[:, 1].sum())}
            return t

        all_mask = np.ones(len(videos), bool)
        dev_mask = np.array([split_of[v] == "dev" for v in videos])
        entry = {"recall": table(all_mask), "recall_dev": table(dev_mask), "recall_confirmation": table(~dev_mask),
                 "wilson": {n: list(wilson(table(all_mask)[n])) for n in BIN_NAMES}}
        pooled = lambda bins, t=table(all_mask): (sum(t[b]["found"] for b in bins), sum(t[b]["persons"] for b in bins))  # noqa: E731
        for grp, bins in (("small_lt48", SMALL_BINS), ("large_ge96", LARGE_BINS), ("mid_48_96", ("48-64", "64-96"))):
            f, n = pooled(bins)
            entry[f"pooled_{grp}"] = {"found": f, "persons": n, "recall": f / n if n else float("nan"),
                                      "wilson": list(wilson({"found": f, "persons": n}))}
        if arm.name == "stock":  # parity of the extended matcher with exp022's own functions
            old = exp022.match_flags(tracks, arm.cache_dir)
            out["parity_with_exp022_matcher"] = bool(all(np.array_equal(old[v][p], flags[v][p]) for v in old for p in old[v]))
        if arm.name in DECOMP_ARMS:
            cls, cls_long = exp022.miss_classes(tracks, flags), exp022.miss_classes(tracks, flags, min_length=exp022.LONG_TRACK)
            entry["miss_classes"], entry["miss_classes_long_tracks"] = cls, cls_long
            entry["low_conf"] = miss_breakdown(tracks, best)
            entry["intermittent_share"] = cls["intermittent_share"]
        out["arms"][arm.name] = entry
        # per-video counts for results.csv
        entry["_video_counts"] = {v: counts[vi].tolist() for vi, v in enumerate(videos)}
    for a, b in RECALL_CONTRASTS:
        if a in counts_by_arm and b in counts_by_arm:
            out["contrasts"][f"{a}-{b}"] = {grp: bootstrap_count_diff(counts_by_arm[a], counts_by_arm[b], bins) for grp, bins in BIN_GROUPS.items()}
    return out


# ---- R2: the replay of every arm's cache reproduces its stored pose folder ----------------------------------------------------
def stage_replay_check(ctx: Ctx, per_arm: int = 5) -> dict:
    from edgelab.pose.detcache import load_detections

    out = {}
    for arm in ctx.arms:
        if arm.cache_dir is None:
            continue
        ds, names = arm.dataset, ctx.names[arm.dataset]
        files = e00.index_pose_dir(arm.pose_dir, ds)
        step = max(1, len(names) // per_arm)
        sample = names[::step][:per_arm]
        equal, bad = 0, []
        for clip in sample:
            frames, data = load_detections(arm.cache_dir / f"{clip}.npz")
            mine = replay_cell(frames, data["frame_ids"], DEFAULT_CELL)
            stored = e00.read_pose_clip(files[clip]) if clip in files else {}
            if track_partition(mine) == track_partition(stored):
                equal += 1
            else:
                bad.append(clip)
        out[arm.key] = {"clips_checked": len(sample), "identical": equal, "differing": bad}
    return out


# ---- D: tracker sweep (dev only) and verification ----------------------------------------------------------------------------
SWEEP_SERIES = (("ShanghaiTech", "ML0_ncnn", "primary"), ("ShanghaiTech", "stock", "veto"), ("UBnormal", "ML0_mps", "veto"),
                ("UBnormal", "stock", "veto"))
VERIFY_SERIES = (("ShanghaiTech", "ML0_ncnn"), ("ShanghaiTech", "stock"), ("ShanghaiTech", "MLP_ncnn"), ("UBnormal", "ML0_mps"), ("UBnormal", "stock"))
GATE_FLOOR = min(x for x in GRID["low"] if x is not None)


def load_frames_floor(cache_dir: Path, clips: list[str], floor: float) -> tuple[list, list]:
    from edgelab.pose.detcache import load_detections

    frames_by_clip, ids_by_clip = [], []
    for clip in clips:
        frames, data = load_detections(cache_dir / f"{clip}.npz")
        frames_by_clip.append([{k: v[fr["conf"] >= floor] for k, v in fr.items()} for fr in frames])
        ids_by_clip.append(data["frame_ids"])
    return frames_by_clip, ids_by_clip


def replay_cells(cells: list[TrackerCell], frames_by_clip: list, ids_by_clip: list, procs: int, consume) -> None:
    """Replay `cells` in batches of `procs` (bounded memory) and hand (label, tracks_by_clip) to `consume` one by one."""
    _G["det_frames"], _G["det_ids"] = frames_by_clip, ids_by_clip
    try:
        for i in range(0, len(cells), max(procs, 1)):
            for label, tracks in pmap(_replay_worker, cells[i:i + max(procs, 1)], procs):
                consume(label, tracks)
    finally:
        _G.pop("det_frames", None)
        _G.pop("det_ids", None)


def stage_sweep(ctx: Ctx) -> dict:
    """Part D on `test_dev` clips ONLY: the clip lists handed to the sweep never contain a confirmation clip (rule R4)."""
    arms = arm_by_key(ctx.arms)
    cells = grid_cells()
    by_label = {c.label: c for c in cells}
    out = {"cells": [c.label for c in cells], "series": {}, "dev_clip_counts": {}}
    for ds, name, role in SWEEP_SERIES:
        arm = arms[f"{ds}/{name}"]
        dev_names = [ctx.names[ds][i] for i in ctx.dev_idx[ds]]
        assert set(dev_names).isdisjoint(ctx.names[ds][i] for i in ctx.conf_idx[ds])  # R4
        labels = [ctx.labels[ds][i] for i in ctx.dev_idx[ds]]
        sentinel, hold = arm_sentinel(ctx.proto, arm), hold_of(ctx.proto, ds)
        ctx.log(f"[{now()}] sweep {arm.key} ({role}): {len(dev_names)} dev clips x {len(cells)} cells")
        frames_by_clip, ids_by_clip = load_frames_floor(arm.cache_dir, dev_names, GATE_FLOOR)
        cache: dict = {}
        res, series = {}, {}

        def consume(label, tracks):
            table = window_table_from_tracks(tracks, SEG_LENS[ds], ctx.scorers[ds], cache)
            ev = evaluate_tracks(tracks, table, dev_names, labels, ds, sentinel, hold, list(range(len(dev_names))), [], with_s0=False)
            ev.pop("_d")
            series[label] = ev.pop("series")["S4"]
            ev["dev_auc_S4"] = ev["auc"]["S4"]["all"]  # all == the dev clips
            res[label] = ev

        replay_cells(cells, frames_by_clip, ids_by_clip, ctx.procs, consume)
        del frames_by_clip
        default = DEFAULT_CELL.label
        # paired dev bootstrap vs the default: every cell on the primary series, only the winner / argmax on veto series (below)
        _G.update(labels={ds: labels}, dev_idx={ds: None}, conf_idx={ds: None},
                  series={(ds, lab, "S4"): s for lab, s in series.items()})
        jobs = [("diff", ds, lab, default, "S4", "all", N_BOOT) for lab in series if lab != default] if role == "primary" else []
        done = pmap(_boot_worker, jobs, ctx.procs)
        _G.clear()
        deltas = {j[2]: r for j, r in done}
        out["series"][f"{ds}/{name}"] = {"role": role, "cells": res, "paired_vs_default": deltas, "sentinel": sentinel,
                                         "n_dev_clips": len(dev_names)}
        out.setdefault("_series_arrays", {})[f"{ds}/{name}"] = series
    dev_auc = {lab: out["series"]["ShanghaiTech/ML0_ncnn"]["cells"][lab]["dev_auc_S4"] for lab in out["cells"]}
    sel = select_tracker_cell(dev_auc, by_label)
    out["T1"] = sel
    # veto-series CIs for the winner and the argmax
    ver = {}
    for pick in sorted({sel["winner"], sel["argmax"]} - {DEFAULT_CELL.label}):
        per = {}
        for ds, name, role in SWEEP_SERIES:
            key = f"{ds}/{name}"
            labels = [ctx.labels[ds][i] for i in ctx.dev_idx[ds]]
            arrays = out["_series_arrays"][key]
            _G.update(labels={ds: labels}, dev_idx={ds: None}, conf_idx={ds: None}, series={(ds, l, "S4"): s for l, s in arrays.items()})
            (_, r), = pmap(_boot_worker, [("diff", ds, pick, DEFAULT_CELL.label, "S4", "all", N_BOOT)], 1)
            _G.clear()
            per[key] = r
        ver[pick] = per
    out["paired_winner_argmax_all_series"] = ver
    primary_delta = {p: ver[p]["ShanghaiTech/ML0_ncnn"] for p in ver}
    vetoes = {p: {k: v for k, v in ver[p].items() if k != "ShanghaiTech/ML0_ncnn"} for p in ver}
    if sel["winner"] == DEFAULT_CELL.label:
        out["T2"] = adopt_tracker(sel["winner"], DEFAULT_CELL.label, {"delta": 0.0, "ci95": [0.0, 0.0]}, {})
    else:
        out["T2"] = adopt_tracker(sel["winner"], DEFAULT_CELL.label, primary_delta[sel["winner"]], vetoes[sel["winner"]])
    # marginal effects: mean dev AUC of the cells at every factor level
    marg = {}
    for key, info in out["series"].items():
        marg[key] = {}
        for factor, getter in (("c", lambda c: c.c), ("low", lambda c: "off" if c.low is None else c.low), ("buffer", lambda c: c.buffer),
                               ("match", lambda c: c.match)):
            levels: dict = {}
            for lab, r in info["cells"].items():
                levels.setdefault(str(getter(by_label[lab])), []).append(r["dev_auc_S4"])
            marg[key][factor] = {lv: float(np.mean(v)) for lv, v in levels.items()}
    out["marginal_means"] = marg
    out.pop("_series_arrays")
    return out


def stage_verify(ctx: Ctx, sweep: dict) -> dict:
    """T3: default / locked / dev-arg-max cells on the full clip sets. The locked cell was fixed from dev data before this runs."""
    arms = arm_by_key(ctx.arms)
    by_label = {c.label: c for c in grid_cells()}
    locked = sweep["T2"]["locked"]
    argmax, winner = sweep["T1"]["argmax"], sweep["T1"]["winner"]
    picks = list(dict.fromkeys([DEFAULT_CELL.label, locked, winner, argmax]))
    out = {"picks": {"default": DEFAULT_CELL.label, "locked_after_T2": locked, "dev_winner_tie_resolved": winner, "dev_argmax": argmax},
           "series": {}}
    for ds, name in VERIFY_SERIES:
        arm = arms[f"{ds}/{name}"]
        names, labels = ctx.names[ds], ctx.labels[ds]
        sentinel, hold = arm_sentinel(ctx.proto, arm), hold_of(ctx.proto, ds)
        ctx.log(f"[{now()}] verify {arm.key}: {len(names)} clips x {len(picks)} cells")
        frames_by_clip, ids_by_clip = load_frames_floor(arm.cache_dir, names, GATE_FLOOR)
        cache: dict = {}
        res, series = {}, {}

        def consume(label, tracks):
            table = window_table_from_tracks(tracks, SEG_LENS[ds], ctx.scorers[ds], cache)
            ev = evaluate_tracks(tracks, table, names, labels, ds, sentinel, hold, ctx.dev_idx[ds], ctx.conf_idx[ds], with_s0=True)
            ev.pop("_d")
            for s, v in ev.pop("series").items():
                series[(label, s)] = v
            res[label] = ev

        replay_cells([by_label[p] for p in picks], frames_by_clip, ids_by_clip, ctx.procs, consume)
        del frames_by_clip
        _G.update(labels=ctx.labels, dev_idx=ctx.dev_idx, conf_idx=ctx.conf_idx, series={(ds, lab, s): v for (lab, s), v in series.items()})
        jobs = [("diff", ds, p, DEFAULT_CELL.label, "S4", sub, N_BOOT) for p in picks if p != DEFAULT_CELL.label
                for sub in ("all", "dev", "confirmation")]
        jobs += [("diff", ds, p, DEFAULT_CELL.label, "S0", "all", N_BOOT) for p in picks if p != DEFAULT_CELL.label]
        done = pmap(_boot_worker, jobs, ctx.procs)
        _G.clear()
        paired = {}
        for (_, _, a, _, step, sub, _), r in done:
            paired.setdefault(a, {})[f"{step}|{sub}"] = r
        out["series"][arm.key] = {"cells": res, "paired_vs_default": paired}
    primary = out["series"]["ShanghaiTech/ML0_ncnn"]["paired_vs_default"].get(locked, {}).get("S4|confirmation")
    out["T3"] = verify_tracker(locked, DEFAULT_CELL.label, primary) if primary is not None else \
        verify_tracker(locked, DEFAULT_CELL.label, {"delta": 0.0, "ci95": [0.0, 0.0]})
    out["final_cell"] = out["T3"]["final"]
    return out


# =====================================================================================================================
# 11. Decisions: frontend lock, rules
# =====================================================================================================================
def negate(r: dict) -> dict:
    return {"delta": -r["delta"], "ci95": [-r["ci95"][1], -r["ci95"][0]], "excludes_zero": r["excludes_zero"]}


def decide_frontend(arms: dict, boot: dict) -> dict:
    diff = boot["diff"]
    cand = [{"name": n, "cost": c} for n, c in (("ML0_ncnn", 1.0), ("stock_ncnn800", 1.5625), ("MLP_ncnn", 1.92))]
    dev_auc = {n["name"]: arms[f"ShanghaiTech/{n['name']}"]["auc"]["S4"]["dev"] for n in cand}
    pair = {("stock_ncnn800", "ML0_ncnn"): negate(diff["sht_ML0ncnn_stockncnn800"]["S4|dev"]),
            ("MLP_ncnn", "ML0_ncnn"): diff["sht_MLPncnn_ML0ncnn"]["S4|dev"],
            ("MLP_ncnn", "stock_ncnn800"): diff["sht_MLPncnn_stockncnn800"]["S4|dev"]}
    ub = diff["ub_MLPmps_ML0mps"]["S4|dev"]
    sel = select_frontend(cand, dev_auc, pair, {"MLP_ncnn": bool(ub["ci95"][1] < 0)})
    sel["dev_auc"] = dev_auc
    sel["ub_proxy_MLPmps_minus_ML0mps_dev"] = ub
    sel["evidence"] = {n: {"S4": arms[f"ShanghaiTech/{n}"]["auc"]["S4"], "S0": arms[f"ShanghaiTech/{n}"]["auc"]["S0"]} for n in dev_auc}
    return sel


def evaluate_rules(S: dict) -> dict:
    rules: dict = {}
    d, arms = S["bootstrap"]["diff"], S["arms"]

    def setr(rid, ok, detail, info=False):
        rules[rid] = {"status": "INFO" if info else ("PASS" if ok else "FAIL"), "pass": None if info else bool(ok), "detail": detail}

    def fmt(r):
        return f"{r['delta']:+.2f} [{r['ci95'][0]:+.2f}, {r['ci95'][1]:+.2f}]"

    r = d["sht_ML0ncnn_stock"]["S4|all"]
    setr("QA1", r["delta"] > 0 and r["ci95"][0] > 0, f"ML0_ncnn - stock@640, SHT S4 all: {fmt(r)}")
    r = d["sht_ML0ncnn_stockncnn800"]["S4|all"]
    setr("QA2", r["delta"] > 0 and r["ci95"][0] > 0, f"ML0_ncnn - stock_ncnn800, SHT S4 all: {fmt(r)}")
    r = d["sht_ML0mps_M00mps"]["S4|all"]
    setr("QA3", r["delta"] > 0 and r["ci95"][0] > 0, f"ML0_mps - M00_mps, SHT S4 all: {fmt(r)}")
    r = d["sht_ML0ncnn_ML0mps"]["S4|all"]
    setr("QA4", abs(r["delta"]) <= 0.5 and r["ci95"][0] <= 0 <= r["ci95"][1], f"ML0_ncnn - ML0_mps, SHT S4 all: {fmt(r)}")
    a, b = d["sht_stock960_stock"]["S4|all"], d["sht_ML0mps_stock800"]["S4|all"]
    setr("QA5", a["delta"] > 0 and b["delta"] > 0, f"stock@960 - stock@640: {fmt(a)}; ML0_mps - stock@800: {fmt(b)}")
    r = d["ub_ML0mps_stock"]["S4|all"]
    setr("QA6", r["ci95"][0] > 0, f"UB ML0_mps - stock@640 S4 all: {fmt(r)} (supported = CI lower > 0)", info=True)
    r = d["sht_MLPncnn_ML0ncnn"]["S4|all"]
    setr("QA7", True, f"MLP_ncnn - ML0_ncnn, SHT S4 all: {fmt(r)}; dev {fmt(d['sht_MLPncnn_ML0ncnn']['S4|dev'])}", info=True)
    rc = S["recall"]["arms"]
    ml0, stock = rc["ML0_ncnn"], rc["stock"]
    small, large = ml0["pooled_small_lt48"], ml0["pooled_large_ge96"]
    setr("QB1", small["recall"] <= 0.5 * large["recall"],
         f"ML0_ncnn recall <48 px {small['recall']:.3f} vs >=96 px {large['recall']:.3f} (ratio {small['recall'] / large['recall']:.2f})")
    ss = stock["pooled_small_lt48"]
    setr("QB2", small["recall"] > ss["recall"] and small["wilson"][0] > ss["wilson"][1],
         f"recall <48 px: ML0_ncnn {small['recall']:.3f} [{small['wilson'][0]:.3f}, {small['wilson'][1]:.3f}] vs stock {ss['recall']:.3f} "
         f"[{ss['wilson'][0]:.3f}, {ss['wilson'][1]:.3f}]")
    setr("QB3", ml0["intermittent_share"] >= INTERMITTENT_THRESHOLD,
         "intermittent share of missed person-frames: " + ", ".join(f"{k} {rc[k]['intermittent_share']:.1%}" for k in DECOMP_ARMS if k in rc))
    lc = ml0["low_conf"]
    setr("QB4", lc["share_in_band"] >= LOW_CONF_SHARE_THRESHOLD,
         f"ML0_ncnn: {lc['share_in_band']:.1%} of missed person-frames have a cached box with 0.05 <= conf < 0.25; two-stage-recoverable (intermittent) "
         f"{lc['share_two_stage_recoverable']:.1%}, gate-only (never-seen) {lc['share_gate_only_recoverable']:.1%}")
    setr("R1", S["R1_lock_verification"]["passes"], f"{S['R1_lock_verification']['n_checked']} objects re-hashed; {S['R1_lock_verification']['problems'] or 'no problems'}")
    bad = {k: v for k, v in S["replay_check"].items() if v["differing"]}
    setr("R2", not bad, "replay == stored pose folders on a clip sample for every arm" if not bad else f"differs: {bad}")
    worst = max((abs(v) for a in arms.values() if a["e00_parity"] for s in a["e00_parity"].values() for v in s.values() if v is not None and not math.isnan(v)),
                default=0.0)
    setr("R3", worst <= 0.01, f"max |AUC(E01) - AUC(E00)| over every E00 arm, step and split = {worst:.4f} pt")
    sp = S["splits"]
    setr("R4", all(v["disjoint_and_covering"] for v in sp.values()) and S["tracker_sweep"]["dev_only"],
         f"splits disjoint+covering {{ {', '.join(f'{k}: {v['disjoint_and_covering']}' for k, v in sp.items())} }}; sweep clip lists = dev only "
         f"({S['tracker_sweep']['dev_clip_counts']})")
    caus = max(a["causality"]["prefix_invariance_max_abs_diff"] for a in arms.values())
    land = sum(a["causality"]["windows_landing_before_completion_frame"] for a in arms.values())
    setr("R5", caus <= 1e-12 and land == 0, f"prefix-invariance max diff {caus:.1e}; windows landing before completion: {land}")
    setr("R6", S["pipeline_consistency"]["passes"], S["pipeline_consistency"]["detail"])
    t = S["tracker_sweep"]
    setr("T1", True, f"dev winner (tie-resolved) {t['T1']['winner']} ({t['T1']['winner_auc']:.2f}); argmax {t['T1']['argmax']} ({t['T1']['best']:.2f}); tied: {len(t['T1']['tied'])} cells", info=True)
    setr("T2", True, f"adopted={t['T2']['adopted']} -> {t['T2']['locked']}: {t['T2']['reason']}", info=True)
    v = S["tracker_verification"]["T3"]
    setr("T3", not v["failed"], f"{v['reason']}")
    f = S["frontend_selection"]
    setr("F2", True, f"locked {f['locked']}: " + "; ".join(f"{s['candidate']} vs {s['incumbent']}: dev gain {s['dev_gain']:+.2f}, CI lower {s['ci95'][0]:+.2f} -> "
                                                         f"{'replaces' if s['replaces'] else ('UB veto' if s['ub_veto'] else 'does not replace')}" for s in f["steps"]), info=True)
    return rules


def pipeline_consistency(S: dict) -> dict:
    """R6: (a) tracker pipeline at the default cell (in-memory replay) == stored-pose pipeline of the arms stage (AUC, all splits, S0 and S4);
    (b) the dev-only sweep AUC of the default cell == the dev AUC of the full-set evaluation of that cell."""
    diffs, notes = [], []
    for key, ver in S["tracker_verification"]["series"].items():
        cell = ver["cells"][DEFAULT_CELL.label]
        for step in ("S0", "S4"):
            for sub in ("all", "dev", "confirmation"):
                diffs.append(abs(cell["auc"][step][sub] - S["arms"][key]["auc"][step][sub]))
        sw = S["tracker_sweep"]["series"].get(key)
        if sw:
            diffs.append(abs(sw["cells"][DEFAULT_CELL.label]["dev_auc_S4"] - cell["auc"]["S4"]["dev"]))
    worst = max(diffs) if diffs else 0.0
    return {"passes": worst <= 0.02, "max_abs_diff_pt": worst,
            "detail": f"max |ΔAUC| = {worst:.4f} pt between the in-memory tracker pipeline (default cell), the stored-pose pipeline and the dev-only sweep (tolerance 0.02 pt: "
                      "float32 batch-composition noise in the STG-NF scores)"}


def locked_for_e02(S: dict) -> dict:
    fe = S["frontend_selection"]["locked"]
    arm = S["arms"][f"ShanghaiTech/{fe}"]
    ub_proxy = {"ML0_ncnn": "ML0_mps", "MLP_ncnn": "MLP_mps", "stock_ncnn800": "stock_r960"}[fe]
    final = S["tracker_verification"]["final_cell"]
    cell = {c.label: c for c in grid_cells()}[final]
    return {
        "frontend": {"name": fe, "label": arm["label"], "relative_flops_vs_stock640": arm["cost"], "runtime": arm["runtime"],
                     "ubnormal_proxy_arm": ub_proxy, "ubnormal_note": "no NCNN cache exists for UBnormal: the MPS FP32 arm is the quality proxy",
                     "S4_ShanghaiTech": arm["auc"]["S4"], "S0_ShanghaiTech": arm["auc"]["S0"],
                     "S4_UBnormal_proxy": S["arms"][f"UBnormal/{ub_proxy}"]["auc"]["S4"] if f"UBnormal/{ub_proxy}" in S["arms"] else None},
        "tracker": {"label": final, "c": cell.c, "two_stage_low_thresh": cell.low, "track_buffer": cell.buffer, "match_thresh": cell.match,
                    "is_default_e00_setting": final == DEFAULT_CELL.label, "T2": S["tracker_sweep"]["T2"], "T3": S["tracker_verification"]["T3"]},
        "scoring_protocol": {"landing": "completion", "smoother": "causal half-Gaussian", "per_dataset": S["protocol"],
                             "note": "S3/S4 only; never report S0-S2 as the delivered baseline (E00 decision)"},
        "runner_up": next((s["candidate"] for s in S["frontend_selection"]["steps"] if not s["replaces"]), None),
    }


# =====================================================================================================================
# 12. Outputs
# =====================================================================================================================
CSV_COLUMNS = ["dataset", "frontend", "tracker", "video", "split", "n_frames", "anomalous_frames", "n_tracks", "n_poses", "track_len_mean",
               "track_len_median", "n_stranded_short", "n_stranded_gappy", "n_exact_seg_len", "gaps_1", "gaps_2", "gaps_ge3", "gap_frames",
               "poses_scorable", "poses_unscorable_stranded", "poses_unscorable_initial", "poses_unscorable_post_gap", "n_windows",
               "windows_per_frame", "coverage_all", "coverage_normal", "coverage_anomalous", "auc_S0", "auc_S3", "auc_S4", "ref_persons",
               "ref_found", "ref_persons_lt48", "ref_found_lt48"]


def csv_row(dataset: str, frontend: str, tracker: str, r: dict, ref: list | None = None) -> dict:
    nf, nn, na = max(r["n_frames"], 1), r["n_normal_frames"], r["anomalous_frames"]
    row = {k: r.get(k, "") for k in CSV_COLUMNS if k in r}
    row.update(dataset=dataset, frontend=frontend, tracker=tracker, windows_per_frame=r["n_windows"] / nf,
               coverage_all=r["scorable_frames"] / nf, coverage_normal=r["scorable_normal"] / nn if nn else "",
               coverage_anomalous=r["scorable_anomalous"] / na if na else "")
    if ref is not None:
        row.update(ref_persons=sum(x[0] for x in ref), ref_found=sum(x[1] for x in ref), ref_persons_lt48=sum(x[0] for x in ref[:2]),
                   ref_found_lt48=sum(x[1] for x in ref[:2]))
    for k, v in list(row.items()):
        if isinstance(v, float):
            row[k] = "" if math.isnan(v) else round(v, 5)
    return row


def write_results_csv(S: dict, recall_counts: dict, sweep_rows: dict, verify_rows: dict) -> int:
    rows = []
    for key, a in S["arms"].items():
        ds, name = a["dataset"], a["name"]
        counts = recall_counts.get(name) if ds == "ShanghaiTech" else None
        for r in a["rows"]:
            rows.append(csv_row(ds, name, "stored-default", r, None if counts is None else counts.get(r["video"])))
    for (key, label), rs in sweep_rows.items():
        ds, name = key.split("/")
        rows += [csv_row(ds, name, "sweep:" + label, r) for r in rs]
    for (key, label), rs in verify_rows.items():
        ds, name = key.split("/")
        rows += [csv_row(ds, name, "verify:" + label, r) for r in rs]
    with open(EXP_DIR / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        w.writeheader()
        w.writerows(rows)
    return len(rows)


def write_decisions(S: dict) -> int:
    lines = []
    add = lambda kind, **kw: lines.append(json.dumps(plain({"kind": kind, **kw})))  # noqa: E731
    for key, a in S["arms"].items():
        add("arm_result", arm=key, label=a["label"], sentinel_dist=a["sentinel_dist"], sentinel=a["sentinel"], hold=a["hold"], auc=a["auc"],
            auc_common_sentinel=a["auc_extra"], coverage=a["coverage"], e00_parity=a["e00_parity"])
        add("track_stats", arm=key, agg=a["agg"]["all"], length=a["length"], gap=a["gap"])
    for cid, c in S["bootstrap"]["diff"].items():
        add("contrast", id=cid, **c)
    for name, a in S["recall"]["arms"].items():
        add("recall", arm=name, recall=a["recall"], pooled={k: a[k] for k in a if k.startswith("pooled_")})
        if "low_conf" in a:
            add("miss_decomposition", arm=name, classes=a["miss_classes"], classes_long=a["miss_classes_long_tracks"], low_conf=a["low_conf"])
    for key, info in S["tracker_sweep"]["series"].items():
        for lab, c in info["cells"].items():
            add("tracker_cell", series=key, role=info["role"], cell=lab, dev_auc_S4=c["dev_auc_S4"], coverage=c["coverage"],
                n_tracks=c["agg"]["all"]["n_tracks"], stranded_short=c["agg"]["all"]["n_stranded_short"], gaps_ge3=c["agg"]["all"]["gaps_ge3"],
                paired_vs_default=info["paired_vs_default"].get(lab))
    add("tracker_selection", T1=S["tracker_sweep"]["T1"], T2=S["tracker_sweep"]["T2"], paired_winner_argmax=S["tracker_sweep"]["paired_winner_argmax_all_series"],
        T3=S["tracker_verification"]["T3"], picks=S["tracker_verification"]["picks"])
    add("frontend_selection", **S["frontend_selection"])
    for rid, r in S["rules"].items():
        add("rule", id=rid, **r)
    add("lock", **S["locked_for_E02"])
    (EXP_DIR / "decisions.jsonl").write_text("\n".join(lines) + "\n")
    return len(lines)


def strip_rows(obj):
    """summary.json keeps aggregates only: per-video rows live in results.csv."""
    if isinstance(obj, dict):
        return {k: strip_rows(v) for k, v in obj.items() if k not in ("rows", "_video_counts")}
    if isinstance(obj, list):
        return [strip_rows(v) for v in obj]
    return obj


def run_all(procs: int, force: set[str], log=print) -> dict:
    t_start = time.perf_counter()
    if not (EXP_DIR / "config.yaml").exists():
        raise SystemExit("run `lock` first: Q2-P1-E01_quality/config.yaml is the pre-registered, hashed configuration")
    config = yaml.safe_load((EXP_DIR / "config.yaml").read_text())
    stamp = e00.sha256_file(EXP_DIR / "config.yaml")[:12]
    force = set(force)
    if "sweep" in force:
        force.add("verify")
    if "arms" in force:
        force |= {"boot"}
    log(f"[{now()}] R1: verifying locked hashes ...")
    r1 = verify_lock(config)
    ctx = build_ctx(procs, log)
    arms = cached_stage("arms", lambda: stage_arms(ctx), force, stamp)
    boot = cached_stage("boot", lambda: stage_boot(ctx), force, stamp)
    recall = cached_stage("recall", lambda: stage_recall(ctx), force, stamp)
    rcheck = cached_stage("replay_check", lambda: stage_replay_check(ctx), force, stamp)
    sweep = cached_stage("sweep", lambda: stage_sweep(ctx), force, stamp)
    verify = cached_stage("verify", lambda: stage_verify(ctx, sweep), force, stamp)

    sweep_rows = {(key, lab): c["rows"] for key, info in sweep["series"].items() for lab, c in info["cells"].items()}
    verify_rows = {(key, lab): c["rows"] for key, info in verify["series"].items() for lab, c in info["cells"].items()}
    recall_counts = {name: a.get("_video_counts", {}) for name, a in recall["arms"].items()}
    S = {"experiment": "Q2-P1-E01_quality", "environment": e00.env_info(), "seed": SEED, "R1_lock_verification": r1, "protocol": ctx.proto,
         "splits": ctx.e00sum["R8_splits"], "arms": arms, "bootstrap": boot, "recall": recall, "replay_check": rcheck, "tracker_sweep": sweep,
         "tracker_verification": verify}
    S["tracker_sweep"]["dev_only"] = True
    S["tracker_sweep"]["dev_clip_counts"] = {k: v["n_dev_clips"] for k, v in sweep["series"].items()}
    S["frontend_selection"] = decide_frontend(arms, boot)
    S["pipeline_consistency"] = pipeline_consistency(S)
    S["rules"] = evaluate_rules(S)
    S["locked_for_E02"] = locked_for_e02(S)
    n_rows = write_results_csv(S, recall_counts, sweep_rows, verify_rows)
    n_dec = write_decisions(S)
    S["runtime_seconds"] = time.perf_counter() - t_start
    S["outputs"] = {"results_csv_rows": n_rows, "decisions_jsonl_lines": n_dec}
    (EXP_DIR / "summary.json").write_text(json.dumps(plain(strip_rows(S)), indent=1))
    log(f"[{now()}] done in {S['runtime_seconds']:.0f}s: results.csv {n_rows} rows, decisions.jsonl {n_dec} lines")
    return S

# =====================================================================================================================
# 13. report.md
# =====================================================================================================================
def md_table(header: list[str], rows: list[list], align: str | None = None) -> str:
    align = align or ("l" + "r" * (len(header) - 1))
    sep = "|" + "|".join({"l": ":--", "r": "--:", "c": ":-:"}[a] for a in align) + "|"
    return "\n".join(["| " + " | ".join(header) + " |", sep] + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows])


def f2(x, nd: int = 2) -> str:
    return "-" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


def pct(x, nd: int = 1) -> str:
    return "-" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{100 * x:.{nd}f}%"


def with_ci(v, ci) -> str:
    return f"{f2(v)} [{f2(ci[0])}, {f2(ci[1])}]" if ci else f2(v)


def dci(r: dict | None, nd: int = 2) -> str:
    """paired difference `delta [lo, hi]`, '*' when the 95% CI excludes 0."""
    if not r:
        return "-"
    star = "\\*" if r["ci95"][0] > 0 or r["ci95"][1] < 0 else ""
    return f"{r['delta']:+.{nd}f} [{r['ci95'][0]:+.{nd}f}, {r['ci95'][1]:+.{nd}f}]{star}"


def render_report_text(S: dict) -> str:
    arms, boot, rc = S["arms"], S["bootstrap"], S["recall"]
    ci, diff = boot["ci"], boot["diff"]
    sw, ver = S["tracker_sweep"], S["tracker_verification"]
    lock, fe = S["locked_for_E02"], S["frontend_selection"]
    SHT, UB = "ShanghaiTech", "UBnormal"
    dflt = DEFAULT_CELL.label
    A = lambda ds, n: arms[f"{ds}/{n}"]  # noqa: E731
    D = lambda cid, key: diff[cid][key]  # noqa: E731
    out: list[str] = []
    w = out.append

    # ---------------------------------------------------------------------------------------------------------------
    w("# Q2-P1-E01 — Quality: what the frozen frontend/scorer can reach before CPU constraints\n")
    w(f"Generated by `edgelab.analysis.q2_p1_e01` (git `{str(S['environment'].get('git_commit', ''))[:8]}`); pre-registered in `plan.md`, locked in `config.yaml`; "
      f"per-video numbers in `results.csv`, every decision in `decisions.jsonl`, all aggregates in `summary.json`. No training was done: every model is an "
      "existing checkpoint / detector cache (E00 locked splits, hashes and causal S3/S4 protocol). Scores are frame-level AUC x 100 pooled over the clips "
      "of the stated split; bootstrap CIs resample *videos* (1000 resamples, seed 0, paired where it is a difference).\n")

    # ---------------------------------------------------------------------------------------------------------------
    w("## 1. Answer\n")
    w(S["_headline"])
    w("")

    # ---------------------------------------------------------------------------------------------------------------
    w("## 2. Locked inputs and rules\n")
    w(f"* Splits (E00): ShanghaiTech {S['splits'][SHT]['n_dev']} dev / {S['splits'][SHT]['n_confirmation']} confirmation clips, "
      f"UBnormal {S['splits'][UB]['n_dev']} / {S['splits'][UB]['n_confirmation']}; disjoint and covering: {', '.join(f'{k} {v['disjoint_and_covering']}' for k, v in S['splits'].items())}.")
    p = S["protocol"]
    w(f"* Causal protocol (E00): completion landing, half-Gaussian smoother, train-calibrated sentinel, hold {p[SHT]['hold']} (SHT) / {p[UB]['hold']} (UB) — with "
      f"hold 0 the delivered series S3 and S4 are identical, so one column `S4` stands for both. SHT sentinel quantile `{p[SHT]['quantile']}`, "
      f"UB `{p[UB]['quantile']}`.")
    w("* **Tuning discipline.** The only tuned object (tracker/gate setting) was selected on `test_dev` (plan section 3, rules T1-T3). The frontend lock (F1-F3) "
      "uses dev AUC and the dev-paired CI only. Confirmation and full-set numbers are reported alongside, never used for selection.")
    w(f"* Costs (nominal FLOPs vs stock@640, EXP-020 amendment): ML0 1.00x, stock@800 1.5625x, MLP 1.92x, stock@960 2.25x, stock@480 0.5625x.\n")

    # ---------------------------------------------------------------------------------------------------------------
    w("## 3. Frontend / resolution comparison (offline S0 and causal S4)\n")
    w("S0 = the paper's offline STG-NF protocol (middle landing, two-sided Gaussian, global most-normal fill, last window dropped); S4 = the deployable causal "
      "protocol. Columns: dev / confirmation / all (AUC x 100); CIs are 95% video-bootstrap on the all-clips set. `ΔS0→S4` is the cost of causality on all "
      "clips. The sentinel for each arm is the distribution E00 locked for it (stock family: `stock_calib12`; ML0: `ML0_mps_calib12`; everything else, "
      "and all of UBnormal: `released_all`); the last column re-scores S4 with the common `released_all` sentinel to show the result does not hinge on it.\n")
    for ds, names in ((SHT, ["released", "stock_r480", "stock", "stock_r800", "stock_ncnn800", "stock_r960", "M00_mps", "M00_ncnn", "ML0_mps", "ML0_ncnn",
                             "MLP_mps", "MLP_ncnn"]),
                      (UB, ["released", "stock_r480", "stock", "stock_r960", "M00_mps", "ML0_mps", "MLP_mps"])):
        n_dev = S["tracker_sweep"]["dev_clip_counts"].get(f"{ds}/ML0_ncnn" if ds == SHT else f"{ds}/ML0_mps")
        w(f"### {ds}{'' if ds == SHT else ' (no NCNN or @800 caches exist; MPS FP32 arms only)'}\n")
        rows = []
        for n in names:
            a = A(ds, n)
            au = a["auc"]
            cs0, cs4 = ci.get(f"{ds}/{n}|S0|all"), ci.get(f"{ds}/{n}|S4|all")
            rows.append([f"`{n}`", a["runtime"].replace("_", " "), "-" if a["cost"] is None else f"{a['cost']:.2f}x",
                         f2(au["S0"]["dev"]), f2(au["S0"]["confirmation"]), with_ci(au["S0"]["all"], cs0),
                         f2(au["S4"]["dev"]), f2(au["S4"]["confirmation"]), with_ci(au["S4"]["all"], cs4),
                         f"{au['S4']['all'] - au['S0']['all']:+.2f}", f2(a["auc_extra"]["common_released_all_sentinel"]["all"]), pct(a["coverage"]["all"], 0)])
        w(md_table(["arm", "runtime", "cost", "S0 dev", "S0 conf", "S0 all [95% CI]", "S4 dev", "S4 conf", "S4 all [95% CI]", "ΔS0→S4", "S4 all, common sentinel",
                    "scored frames"], rows))
        w("")
    w("Reading guide: `released` is the oracle (authors' AlphaPose poses; the paper's 85.94 is reproduced at S0). `scored frames` is the share of frames that "
      "receive a fresh window score (the rest get the sentinel); this is the tracking-side limit on every causal number.\n")

    w("### 3.1 Paired contrasts (video bootstrap, difference in AUC points; `*` = 95% CI excludes 0)\n")
    rows = []
    for cid, c in diff.items():
        rows.append([("SHT" if c["dataset"] == SHT else "UB") + f" `{c['a']}` − `{c['b']}`", c["text"], dci(c["S0|all"]), dci(c["S4|dev"]),
                     dci(c["S4|confirmation"]), dci(c["S4|all"])])
    w(md_table(["pair", "meaning", "S0 all", "S4 dev", "S4 confirmation", "S4 all"], rows, "lllrrr"))
    w("")

    # ---------------------------------------------------------------------------------------------------------------
    w("## 4. Recall of small people (ShanghaiTech, released-AlphaPose reference; EXP-011/022 protocol)\n")
    w(f"Reference: {rc['n_reference_person_frames']:,} person-frames in {rc['n_reference_tracks']:,} released-AlphaPose tracks. A reference person-frame is *found* "
      "when the frontend's cached box has conf ≥ 0.25 and IoU ≥ 0.3 with it (the EXP-011 matcher; re-implemented here with identical flags to EXP-022's "
      f"functions: parity = {rc['parity_with_exp022_matcher']}). Recall by reference box height (px):\n")
    bins = ["<32", "32-48", "48-64", "64-96", "96-160", ">=160", "all"]
    order = [n for n in ("stock_r480", "stock", "stock_r800", "stock_ncnn800", "stock_r960", "M00_mps", "M00_ncnn", "ML0_mps", "ML0_ncnn", "MLP_mps", "MLP_ncnn")
             if n in rc["arms"]]
    rows = []
    npers = {b: rc["arms"][order[0]]["recall"][b]["persons"] for b in bins}
    for n in order:
        r = rc["arms"][n]["recall"]
        rows.append([f"`{n}`"] + [pct(r[b]["found"] / r[b]["persons"]) for b in bins])
    rows.append(["person-frames"] + [f"{npers[b]:,}" for b in bins])
    w(md_table(["arm"] + bins, rows))
    w("")
    w("Dev vs confirmation recall (all heights / <48 px) and Wilson 95% intervals for the pooled <48 and ≥96 px groups:\n")
    rows = []
    for n in order:
        a = rc["arms"][n]
        g = lambda t, bs: sum(a[t][b]["found"] for b in bs) / max(sum(a[t][b]["persons"] for b in bs), 1)  # noqa: E731
        sm, lg = a["pooled_small_lt48"], a["pooled_large_ge96"]
        rows.append([f"`{n}`", pct(g("recall_dev", ["all"])), pct(g("recall_confirmation", ["all"])), pct(g("recall_dev", ["<32", "32-48"])),
                     pct(g("recall_confirmation", ["<32", "32-48"])), f"{pct(sm['recall'])} [{pct(sm['wilson'][0])}, {pct(sm['wilson'][1])}]",
                     f"{pct(lg['recall'])} [{pct(lg['wilson'][0])}, {pct(lg['wilson'][1])}]"])
    w(md_table(["arm", "all, dev", "all, conf", "<48, dev", "<48, conf", "<48 px, all [Wilson]", "≥96 px, all [Wilson]"], rows))
    w("")
    w("Paired video-bootstrap recall differences (percentage points, 2000 resamples; `*` = CI excludes 0):\n")
    rows = []
    for k, v in rc["contrasts"].items():
        rows.append([f"`{k.replace('-', '` − `')}`"] + [(f"{100 * v[g]['delta']:+.1f} [{100 * v[g]['ci95'][0]:+.1f}, {100 * v[g]['ci95'][1]:+.1f}]" +
                                                      ("\\*" if v[g]["ci95"][0] > 0 or v[g]["ci95"][1] < 0 else "")) for g in ("<48", "48-96", ">=96", "all")])
    w(md_table(["pair", "<48 px", "48-96 px", "≥96 px", "all"], rows, "lrrrr"))
    w("")

    w("### 4.1 Miss decomposition (EXP-022): never-seen vs intermittent, and what the raw cache holds\n")
    w("A reference track is *never seen* if no frame is matched, *intermittent* if some frames are. Each missed person-frame is also classed by the confidence of the "
      "best raw-cache box that overlaps it (IoU ≥ 0.3, conf ≥ 0.01): boxes with 0.05 ≤ conf < 0.25 exist in the cache but are removed by the 0.25 gate. A low box can "
      "only *continue* a live track in BoT-SORT's second association (`two-stage-recoverable` = such boxes inside intermittent tracks); in never-seen tracks it "
      "would need a lower gate that also spawns tracks (`gate-only`).\n")
    rows = []
    for n in DECOMP_ARMS:
        a = rc["arms"][n]
        c, lc = a["miss_classes"], a["low_conf"]
        rows.append([f"`{n}`", f"{c['missed_person_frames']:,}", pct(c["missed_person_frames"] / c["person_frames"]), pct(c["never_seen_share"]), pct(c["intermittent_share"]),
                     f"{c['tracks_never_seen']}/{c['tracks_intermittent']}/{c['tracks_fully_matched']}", pct(c["intermittent_share_match_rate_ge_half"]),
                     f"{lc['in_band_total']:,}", pct(lc["share_in_band"]), pct(lc["share_two_stage_recoverable"]), pct(lc["share_gate_only_recoverable"])])
    w(md_table(["arm", "missed person-frames", "of reference", "never-seen share", "intermittent share", "tracks never/interm./full", "intermittent in tracks matched ≥50%",
                "misses with 0.05≤conf<0.25 box", "share of misses", "two-stage-recoverable", "gate-only"], rows))
    w("")
    w("Best raw-cache box confidence for missed person-frames (share of all misses), by kind:\n")
    rows = []
    for n in DECOMP_ARMS:
        lc = rc["arms"][n]["low_conf"]
        for kind in KINDS:
            rows.append([f"`{n}`", kind.replace("_", "-")] + [pct(lc["by_kind"][kind][b] / max(lc["missed"], 1)) for b in BUCKET_NAMES])
    w(md_table(["arm", "kind"] + [f"best box {b}" for b in BUCKET_NAMES[:-1]] + ["no box ≥0.01"], rows, "llrrrr"))
    w("")
    w("Missed person-frames by height bin (ML0 NCNN; never-seen / intermittent), and the share that has a 0.05-0.25 box:\n")
    rows = []
    for n in ("ML0_ncnn", "MLP_ncnn", "stock"):
        a = rc["arms"][n]
        mb = a["miss_classes"]["missed_by_height_bin"]
        for b in bins[:-1]:
            tot = a["low_conf"]["by_bin"][b]
            ntot = sum(tot.values())
            rows.append([f"`{n}`", b, f"{mb[b]['missed_never_seen']:,}", f"{mb[b]['missed_intermittent']:,}", f"{ntot:,}",
                         pct((tot["0.10-0.25"] + tot["0.05-0.10"]) / ntot) if ntot else "-"])
    w(md_table(["arm", "height", "never-seen", "intermittent", "all misses", "share with 0.05≤conf<0.25 box"], rows))
    w("")

    # ---------------------------------------------------------------------------------------------------------------
    w("## 5. Tracking gaps and stranded windows\n")
    w("Tracks are the BoT-SORT output at the E00 setting (c = 0.25, no two-stage low threshold, buffer 30, match 0.8). `stranded short` = tracks with fewer than "
      "`seg_len` poses (24 SHT / 16 UB): they can never fill a window. `warm-up` = the first `seg_len − 1` poses of each scorable run can only feed later windows' "
      "context, not receive their own score. `gaps` are runs of missing frames inside a track (STG-NF tolerates ≤ 2 per window). Person-window density = windows per frame.\n")
    tr_arms = {SHT: ["released", "stock", "stock_r800", "ML0_ncnn", "MLP_ncnn"], UB: ["released", "stock", "stock_r960", "ML0_mps", "MLP_mps"]}
    for ds in (SHT, UB):
        seg = SEG_LENS[ds]
        w(f"### {ds} (seg_len {seg}{'; UBnormal has no NCNN/@800 cache, so the substitutes stock@960, ML0 MPS and MLP MPS are shown' if ds == UB else ''})\n")
        rows = []
        for n in tr_arms[ds]:
            a = A(ds, n)
            g, ln, gp = a["agg"]["all"], a["length"], a["gap"]
            rows.append([f"`{n}`", f"{g['n_tracks']:,}", f"{ln['median']:.0f}", f"{ln['mean']:.0f}", f"{g['n_stranded_short']:,} ({pct(g['stranded_short_share_of_tracks'], 0)})",
                         f"{g['n_stranded_gappy']}", f"{g['n_exact_seg_len']}", f"{g['gaps_1']:,}/{g['gaps_2']:,}/{g['gaps_ge3']:,}", f"{g['gaps_ge3_per_1000_poses']:.1f}",
                         pct(g["scorable_pose_share"], 0), pct(g["stranded_poses_share"]), pct(g["initial_warmup_pose_share"], 0), pct(g["post_gap_pose_share"], 0),
                         f"{g['windows_per_frame']:.2f}", pct(g["coverage_all"], 1), pct(g["coverage_normal"], 1), pct(g["coverage_anomalous"], 1)])
        w(md_table(["arm", "tracks", "median len", "mean len", "stranded short", "stranded gappy", "exact seg_len", "gaps 1/2/≥3", "≥3-gaps /1k poses",
                    "poses scorable", "poses stranded", "poses in warm-up", "poses in post-gap warm-up", "windows /frame", "scorable frames", "normal", "anomalous"], rows))
        w("")
        w(f"Track-length histogram (poses; {ds}) and gap-length histogram (frames):\n")
        rows = []
        for n in tr_arms[ds]:
            a = A(ds, n)
            rows.append([f"`{n}`", "tracks"] + [f"{a['length']['hist'][k]:,}" for k in a["length"]["hist"]])
        hk = list(A(ds, tr_arms[ds][0])["length"]["hist"])
        w(md_table(["arm", ""] + hk, rows))
        w("")
        rows = []
        for n in tr_arms[ds]:
            a = A(ds, n)
            rows.append([f"`{n}`", "gaps"] + [f"{a['gap']['hist'][k]:,}" for k in a["gap"]["hist"]])
        hk = list(A(ds, tr_arms[ds][0])["gap"]["hist"])
        w(md_table(["arm", ""] + hk, rows))
        w("")

    # ---------------------------------------------------------------------------------------------------------------
    w("## 6. Tracker / gate sweep (development clips only)\n")
    w(f"Grid (locked in `config.yaml`): c ∈ {{0.20, 0.25, 0.30}} × two-stage low threshold ∈ {{disabled, 0.10, 0.05}} × track_buffer ∈ {{30, 45}} × match_thresh ∈ "
      f"{{0.8, 0.85}} = 36 cells. With a low threshold, boxes down to that confidence enter BoT-SORT's second association (they may continue live tracks only; a new "
      f"track still needs conf ≥ c). Primary series: SHT ML0_ncnn S4 on the {sw['dev_clip_counts']['ShanghaiTech/ML0_ncnn']} dev clips; vetoes: SHT stock, UB ML0_mps, UB stock "
      f"({', '.join(f'{k.split('/')[0]} {v}' for k, v in sw['dev_clip_counts'].items())} dev clips). The sweep reads dev clips only (rule R4).\n")
    prim = sw["series"]["ShanghaiTech/ML0_ncnn"]
    cells = prim["cells"]
    ranked = sorted(cells, key=lambda k: -cells[k]["dev_auc_S4"])
    w(f"Dev S4 AUC of the default cell `{dflt}`: **{cells[dflt]['dev_auc_S4']:.2f}**; dev best `{ranked[0]}` {cells[ranked[0]]['dev_auc_S4']:.2f}; "
      f"worst `{ranked[-1]}` {cells[ranked[-1]]['dev_auc_S4']:.2f}. Cells within the {TIE_BAND:.2f}-pt tie band of the best: {len(sw['T1']['tied'])}.\n")
    show = list(dict.fromkeys(ranked[:10] + [dflt] + ranked[-3:]))
    rows = []
    for lab in show:
        c = cells[lab]
        g = c["agg"]["all"]
        other = [sw["series"][k]["cells"][lab]["dev_auc_S4"] - sw["series"][k]["cells"][dflt]["dev_auc_S4"] for k in sw["series"] if k != "ShanghaiTech/ML0_ncnn"]
        rows.append([f"`{lab}`" + (" (default)" if lab == dflt else ""), f2(c["dev_auc_S4"]), dci(prim["paired_vs_default"].get(lab)) if lab != dflt else "-",
                     f"{g['n_tracks']:,}", f"{g['n_stranded_short']:,}", f"{g['gaps_ge3']:,}", pct(c["coverage"]["all"], 1)] + [f"{x:+.2f}" for x in other])
    heads = ["cell", "dev S4 AUC", "paired Δ vs default [95% CI]", "tracks", "stranded short", "gaps ≥3", "scored frames"] + \
            [f"Δ {k.split('/')[0][:3]} {k.split('/')[1]}" for k in sw["series"] if k != "ShanghaiTech/ML0_ncnn"]
    w("Top 10, default and bottom 3 cells by primary dev AUC (Δ columns: dev S4 AUC minus the default cell on the veto series):\n")
    w(md_table(heads, rows))
    w("")
    w("Marginal means of the dev S4 AUC over the cells at each factor level (averaging the other three factors):\n")
    rows = []
    for key in sw["series"]:
        mm = sw["marginal_means"][key]
        for factor in ("c", "low", "buffer", "match"):
            rows.append([f"{key}", factor] + [f"{lv}: {v:.2f}" for lv, v in mm[factor].items()])
    w(md_table(["series", "factor", "level 1", "level 2", "level 3"], [r + [""] * (5 - len(r)) for r in rows], "llrrr"))
    w("")
    t1, t2, t3 = sw["T1"], sw["T2"], ver["T3"]
    w(f"**T1** (dev winner, ties within {TIE_BAND:.2f} pt resolved toward the default): `{t1['winner']}` ({t1['winner_auc']:.2f}); plain arg-max `{t1['argmax']}` ({t1['best']:.2f}).  "
      f"**T2** (adopt only if ≥ +{TRACKER_MIN_DELTA} pt, primary dev CI lower > 0, no significantly harmful veto series): adopted = **{t2['adopted']}** -> locked `{t2['locked']}` — {t2['reason']}.\n")
    if sw.get("paired_winner_argmax_all_series"):
        rows = []
        for pick, per in sw["paired_winner_argmax_all_series"].items():
            rows.append([f"`{pick}`"] + [dci(per[k]) for k in sw["series"]])
        w("Paired dev difference of the winner / arg-max against the default on every sweep series (S4):\n")
        w(md_table(["cell"] + list(sw["series"]), rows, "l" + "r" * len(sw["series"])))
        w("")

    w("### 6.1 Verification on the full clip sets (dev -> confirmation)\n")
    w(f"Cells replayed on all clips: default, locked-after-T2 `{ver['picks']['locked_after_T2']}`, dev winner `{ver['picks']['dev_winner_tie_resolved']}`, dev arg-max "
      f"`{ver['picks']['dev_argmax']}`. The dev numbers equal those of the sweep (R6). **T3**: {t3['reason']} -> final cell `{ver['final_cell']}`.\n")
    rows = []
    for key, info in ver["series"].items():
        for lab, c in info["cells"].items():
            pr = info["paired_vs_default"].get(lab, {})
            rows.append([key, f"`{lab}`" + (" (default)" if lab == dflt else ""), f2(c["auc"]["S4"]["dev"]), f2(c["auc"]["S4"]["confirmation"]), f2(c["auc"]["S4"]["all"]),
                         f2(c["auc"]["S0"]["all"]), dci(pr.get("S4|dev")) if pr else "-", dci(pr.get("S4|confirmation")) if pr else "-", dci(pr.get("S4|all")) if pr else "-",
                         f"{c['agg']['all']['n_tracks']:,}", pct(c["coverage"]["all"], 1)])
    w(md_table(["series", "cell", "S4 dev", "S4 conf", "S4 all", "S0 all", "Δ dev", "Δ conf", "Δ all", "tracks", "scored frames"], rows, "llrrrrrrrrr"))
    w("")

    # ---------------------------------------------------------------------------------------------------------------
    w("## 7. Locked configuration for E02\n")
    f_ = lock["frontend"]
    t_ = lock["tracker"]
    w(f"* **Frontend: `{f_['name']}`** — {f_['label']}, {f_['relative_flops_vs_stock640']:.2f}x stock@640 FLOPs, runtime `{f_['runtime']}`. Causal S4 AUC on ShanghaiTech: dev "
      f"{f_['S4_ShanghaiTech']['dev']:.2f} / confirmation {f_['S4_ShanghaiTech']['confirmation']:.2f} / all {f_['S4_ShanghaiTech']['all']:.2f} (S0 all {f_['S0_ShanghaiTech']['all']:.2f}). "
      f"UBnormal proxy (`{f_['ubnormal_proxy_arm']}`, MPS FP32 — no NCNN UB cache): S4 all {f_['S4_UBnormal_proxy']['all']:.2f}." if f_["S4_UBnormal_proxy"] else
      f"* **Frontend: `{f_['name']}`**.")
    fin = ver["final_cell"]
    mlv, ubv = ver["series"]["ShanghaiTech/ML0_ncnn"]["cells"], ver["series"]["UBnormal/ML0_mps"]["cells"]
    w(f"  * With the locked tracker: ShanghaiTech S4 dev {mlv[fin]['auc']['S4']['dev']:.2f} / confirmation {mlv[fin]['auc']['S4']['confirmation']:.2f} / all "
      f"{mlv[fin]['auc']['S4']['all']:.2f} (S0 all {mlv[fin]['auc']['S0']['all']:.2f}); UBnormal ML0 MPS proxy S4 dev {ubv[fin]['auc']['S4']['dev']:.2f} / confirmation "
      f"{ubv[fin]['auc']['S4']['confirmation']:.2f} / all {ubv[fin]['auc']['S4']['all']:.2f}. Selection-bias note: the dev gain of the locked cell "
      f"({mlv[fin]['auc']['S4']['dev'] - mlv[dflt]['auc']['S4']['dev']:+.2f} pt) is partly winner's curse; the confirmation gain "
      f"({mlv[fin]['auc']['S4']['confirmation'] - mlv[dflt]['auc']['S4']['confirmation']:+.2f} pt) is the unbiased estimate to plan with.")
    for s in fe["steps"]:
        w(f"  * F-step `{s['candidate']}` vs `{s['incumbent']}`: dev gain {s['dev_gain']:+.2f}, dev CI [{s['ci95'][0]:+.2f}, {s['ci95'][1]:+.2f}] -> "
          f"{'replaces the incumbent' if s['replaces'] else ('blocked by UB veto' if s['ub_veto'] else 'does not clear +2.0 pt and CI lower > 0')}.")
    w(f"* **Tracker: `{t_['label']}`** — c = {t_['c']}, two-stage low threshold {t_['two_stage_low_thresh']}, track_buffer {t_['track_buffer']}, match_thresh {t_['match_thresh']}; "
      f"{'identical to the E00 default' if t_['is_default_e00_setting'] else 'differs from the E00 default'}. T2: {t_['T2']['reason']}; T3: {t_['T3']['reason']}.")
    w(f"* **Scoring**: S3/S4 only (completion landing, causal half-Gaussian), sentinel/hold per E00 ({', '.join(f'{k}: {v['quantile']}/hold {v['hold']}' for k, v in lock['scoring_protocol']['per_dataset'].items())}). "
      "Never use S0-S2 as the delivered baseline.")
    w(f"* Runner-up if E02's CPU budget excludes the lock: `{lock['runner_up']}`.\n")

    # ---------------------------------------------------------------------------------------------------------------
    w("## 8. Pre-registered rules\n")
    rows = [[rid, r["status"], str(r["detail"]).replace("|", "\\|")] for rid, r in S["rules"].items()]
    w(md_table(["rule", "status", "evidence"], rows, "lll"))
    w("")
    w(S["_limits"])
    w("")
    return "\n".join(out)

def build_headline(S: dict) -> str:
    arms, boot, rc = S["arms"], S["bootstrap"], S["recall"]
    d, sw, ver, lock = boot["diff"], S["tracker_sweep"], S["tracker_verification"], S["locked_for_E02"]
    fe = S["frontend_selection"]
    A = lambda ds, n: arms[f"{ds}/{n}"]["auc"]  # noqa: E731
    SHT, UB = "ShanghaiTech", "UBnormal"
    ml, st, ml_m, mlp = A(SHT, "ML0_ncnn"), A(SHT, "stock"), A(SHT, "ML0_mps"), A(SHT, "MLP_ncnn")
    rel = A(SHT, "released")
    fmt = lambda r: dci(r)  # noqa: E731
    lines = []
    ad = lambda s: lines.append(s)  # noqa: E731

    ad(f"**1. Offline ceiling vs the deployable causal protocol (ShanghaiTech).** With the paper's offline protocol (S0) the released AlphaPose poses reproduce "
       f"{rel['S0']['all']:.2f}; the best frozen detector (ML0 @640, NCNN FP16) reaches {ml['S0']['all']:.2f} and MLP {mlp['S0']['all']:.2f}, against {st['S0']['all']:.2f} for "
       f"stock YOLOv8n-pose @640 and {A(SHT, 'stock_r800')['S0']['all']:.2f} @800. Under the causal protocol that E02 must beat (S4) the numbers are "
       f"released {rel['S4']['all']:.2f}, ML0 {ml['S4']['all']:.2f} (dev {ml['S4']['dev']:.2f} / confirmation {ml['S4']['confirmation']:.2f}), MLP {mlp['S4']['all']:.2f}, "
       f"stock@640 {st['S4']['all']:.2f}, stock@800 {A(SHT, 'stock_r800')['S4']['all']:.2f}, stock@960 {A(SHT, 'stock_r960')['S4']['all']:.2f}. "
       f"Causality costs {ml['S0']['all'] - ml['S4']['all']:.1f} pt for ML0; the gap to the oracle poses is {rel['S4']['all'] - ml['S4']['all']:.1f} pt (S4) / "
       f"{rel['S0']['all'] - ml['S0']['all']:.1f} pt (S0) and comes from the frontend (pose quality and tracking, items 5-6): the scorer is identical.")
    r1, r2, r3 = d["sht_ML0ncnn_stock"], d["sht_ML0ncnn_stockncnn800"], d["sht_ML0mps_M00mps"]
    ad(f"**2. Existing ML0 vs stock (ShanghaiTech, S4 all, paired video bootstrap).** ML0@640 − stock@640 = {fmt(r1['S4|all'])} (dev {fmt(r1['S4|dev'])}, confirmation "
       f"{fmt(r1['S4|confirmation'])}); ML0@640 − stock@800 at equal runtime (both NCNN FP16) = {fmt(r2['S4|all'])} (dev {fmt(r2['S4|dev'])}, confirmation "
       f"{fmt(r2['S4|confirmation'])}). The partial-label recipe therefore beats *more resolution* by a margin larger than stock@480→960 buys "
       f"({fmt(d['sht_stock960_stock']['S4|all'])} for 640→960), and ML0 − M00 = {fmt(r3['S4|all'])} shows the gain comes from the labels, not from the extra training epochs "
       f"(M00 is *worse* than stock: {fmt(d['sht_M00mps_stock']['S4|all'])}). NCNN FP16 vs MPS FP32 is lossless: ML0 {fmt(d['sht_ML0ncnn_ML0mps']['S4|all'])}, "
       f"MLP {fmt(d['sht_MLPncnn_MLPmps']['S4|all'])}, stock@800 {fmt(d['sht_stockncnn800_stock800']['S4|all'])}.")
    p1, p2 = d["sht_MLPncnn_ML0ncnn"], d["sht_MLPmps_ML0mps"]
    ad(f"**3. The P2 head (MLP − ML0) is small and not confirmed.** NCNN: {fmt(p1['S4|all'])} (dev {fmt(p1['S4|dev'])}, confirmation {fmt(p1['S4|confirmation'])}); MPS: "
       f"{fmt(p2['S4|all'])} (dev {fmt(p2['S4|dev'])}, confirmation {fmt(p2['S4|confirmation'])}) for 1.92x the FLOPs. Frontend lock (F1-F3, dev data only): "
       f"**`{lock['frontend']['name']}`**" + (f" — MLP misses the pre-registered +{FRONTEND_MARGIN:.1f} pt dev margin (dev gain {fe['steps'][-1]['dev_gain']:+.2f}, a shortfall of "
       f"{FRONTEND_MARGIN - fe['steps'][-1]['dev_gain']:.2f} pt although its dev CI excludes 0) and its advantage vanishes on held-out clips and under the locked tracker "
       f"(SHT S4 all with the locked tracker: MLP {ver['series']['ShanghaiTech/MLP_ncnn']['cells'][ver['final_cell']]['auc']['S4']['all']:.2f} vs ML0 "
       f"{ver['series']['ShanghaiTech/ML0_ncnn']['cells'][ver['final_cell']]['auc']['S4']['all']:.2f}; point estimates, not a pre-registered test). The P2 head's recall gain is limited to "
       f"the 48-96 px bins ({fmt_pp(rc['contrasts']['MLP_ncnn-ML0_ncnn']['48-96'])} pt)." if lock["frontend"]["name"] == "ML0_ncnn" else "."))
    ub1, ub2 = d["ub_ML0mps_stock"], d["ub_ML0mps_stock960"]
    ad(f"**4. UBnormal does not show the ML0 advantage.** ML0 − stock@640 = {fmt(ub1['S4|all'])} (dev {fmt(ub1['S4|dev'])}, confirmation {fmt(ub1['S4|confirmation'])}), "
       f"ML0 − stock@960 = {fmt(ub2['S4|all'])}; MLP − ML0 = {fmt(d['ub_MLPmps_ML0mps']['S4|all'])}. UBnormal S4 AUCs are released {A(UB, 'released')['S4']['all']:.2f}, "
       f"stock {A(UB, 'stock')['S4']['all']:.2f}, ML0 {A(UB, 'ML0_mps')['S4']['all']:.2f}, stock@960 {A(UB, 'stock_r960')['S4']['all']:.2f} (S0: "
       f"{A(UB, 'released')['S0']['all']:.2f} / {A(UB, 'stock')['S0']['all']:.2f} / {A(UB, 'ML0_mps')['S0']['all']:.2f} / {A(UB, 'stock_r960')['S0']['all']:.2f}). "
       "The ShanghaiTech advantage of the partial-label recipe does not transfer to UBnormal, so the quality lock rests on ShanghaiTech and UBnormal is used as a no-harm check "
       "(MPS proxies: no NCNN cache exists).")
    a_ml, a_st = rc["arms"]["ML0_ncnn"], rc["arms"]["stock"]
    g = lambda a, b: a["recall"][b]["found"] / a["recall"][b]["persons"]  # noqa: E731
    lc, lcs = a_ml["low_conf"], rc["arms"]["stock"]["low_conf"]
    ad(f"**5. Small people.** Recall against the released-AlphaPose reference (ShanghaiTech, {rc['n_reference_person_frames']:,} person-frames): ML0 {pct(g(a_ml, '<32'))} / "
       f"{pct(g(a_ml, '32-48'))} / {pct(g(a_ml, '48-64'))} / {pct(g(a_ml, '64-96'))} / {pct(g(a_ml, '96-160'))} for <32 / 32-48 / 48-64 / 64-96 / 96-160 px "
       f"(≥160: {pct(g(a_ml, '>=160'))}), stock@640 {pct(g(a_st, '<32'))} / {pct(g(a_st, '32-48'))} / {pct(g(a_st, '48-64'))} / {pct(g(a_st, '64-96'))} / "
       f"{pct(g(a_st, '96-160'))}; overall {pct(g(a_ml, 'all'))} vs {pct(g(a_st, 'all'))}. ML0 fixes the mid-size people (48-96 px) and half of the 32-48 px ones but still "
       f"finds only {pct(g(a_ml, '<32'))} of the <32 px people; stock@800 and the P2 head move the small-person recall by only "
       f"{100 * (g(rc['arms']['stock_r800'], '32-48') - g(a_st, '32-48')):+.1f} pt and {100 * (g(rc['arms']['MLP_ncnn'], '32-48') - g(a_ml, '32-48')):+.1f} pt in the 32-48 px bin. "
       f"{pct(a_ml['intermittent_share'])} of ML0's {a_ml['miss_classes']['missed_person_frames']:,} missed person-frames are *intermittent* (people it sees in other frames of the "
       f"same track; stock: {pct(a_st['intermittent_share'])}), and {pct(lc['share_in_band'])} of the ML0 misses have a raw-cache box with 0.05 ≤ conf < 0.25 "
       f"({pct(lc['share_two_stage_recoverable'])} inside intermittent tracks where a two-stage association could use it, {pct(lc['share_gate_only_recoverable'])} in never-seen tracks).")
    gml, grel, gst = A_agg(arms, SHT, "ML0_ncnn"), A_agg(arms, SHT, "released"), A_agg(arms, SHT, "stock")
    ad(f"**6. Tracking gaps and stranded windows.** ML0 produces {gml['n_tracks']:,} tracks on ShanghaiTech (released {grel['n_tracks']:,}, stock {gst['n_tracks']:,}); "
       f"{pct(gml['stranded_short_share_of_tracks'], 0)} of them are shorter than 24 poses (released {pct(grel['stranded_short_share_of_tracks'], 0)}, stock "
       f"{pct(gst['stranded_short_share_of_tracks'], 0)}) and hold {pct(gml['stranded_poses_share'])} of the poses; {gml['gaps_ge3']:,} within-track gaps are ≥3 frames "
       f"(released {grel['gaps_ge3']:,}) and {pct(gml['post_gap_pose_share'], 0)} of the poses sit in post-gap warm-up. Net: {pct(gml['coverage_all'], 1)} of frames receive a "
       f"fresh window score with ML0 (released {pct(grel['coverage_all'], 1)}, stock {pct(gst['coverage_all'], 1)}; anomalous frames {pct(gml['coverage_anomalous'], 1)} vs "
       f"{pct(gst['coverage_anomalous'], 1)}). The ML0 detector's extra recall comes with more track fragmentation, which is the lever the tracker sweep tests.")
    t1, t2, t3 = sw["T1"], sw["T2"], ver["T3"]
    prim = sw["series"]["ShanghaiTech/ML0_ncnn"]["cells"]
    dflt = DEFAULT_CELL.label
    fin = ver["final_cell"]
    vs = ver["series"]["ShanghaiTech/ML0_ncnn"]
    sp = lambda key, step="S4|all": ver["series"][key]["paired_vs_default"].get(fin, {}).get(step)  # noqa: E731
    ad(f"**7. Tracker / gate sweep (dev only, 36 cells).** Default `{dflt}` scores {prim[dflt]['dev_auc_S4']:.2f} on the dev primary series (SHT ML0_ncnn S4); the best cell "
       f"`{t1['argmax']}` {t1['best']:.2f} ({t1['best'] - prim[dflt]['dev_auc_S4']:+.2f} pt); {len(t1['tied'])} cells lie within the {TIE_BAND:.2f} pt tie band of the best. "
       f"Tie-resolved winner `{t1['winner']}`; T2 adoption: **{t2['adopted']}** — {t2['reason']}. Locked tracker: **`{fin}`**" +
       (" (the E00 default: no setting is reliably better). " if fin == dflt else
        f". Verification on the full sets (default → locked, S4): SHT ML0_ncnn dev {vs['cells'][dflt]['auc']['S4']['dev']:.2f} → {vs['cells'][fin]['auc']['S4']['dev']:.2f} "
        f"({fmt(sp('ShanghaiTech/ML0_ncnn', 'S4|dev'))}), **confirmation {vs['cells'][dflt]['auc']['S4']['confirmation']:.2f} → {vs['cells'][fin]['auc']['S4']['confirmation']:.2f} "
        f"({fmt(sp('ShanghaiTech/ML0_ncnn', 'S4|confirmation'))})**, all {vs['cells'][dflt]['auc']['S4']['all']:.2f} → {vs['cells'][fin]['auc']['S4']['all']:.2f} "
        f"({fmt(sp('ShanghaiTech/ML0_ncnn'))}); the dev gain shrinks on held-out clips (selection on 35 clips), but the sign and a CI above 0 hold. T3: {t3['reason']}. "
        f"Other series, same cell vs default (S4 all): SHT stock {fmt(sp('ShanghaiTech/stock'))}, SHT MLP_ncnn {fmt(sp('ShanghaiTech/MLP_ncnn'))}, UB ML0_mps "
        f"{fmt(sp('UBnormal/ML0_mps'))}, UB stock {fmt(sp('UBnormal/stock'))}. Mechanism (factor marginals, section 6): the two-stage low threshold carries the effect "
        f"(dev mean over cells: disabled {sw['marginal_means']['ShanghaiTech/ML0_ncnn']['low']['off']:.2f}, 0.10 {sw['marginal_means']['ShanghaiTech/ML0_ncnn']['low']['0.1']:.2f}, "
        f"0.05 {sw['marginal_means']['ShanghaiTech/ML0_ncnn']['low']['0.05']:.2f}); the lock also lowers c to 0.20 and raises match_thresh to 0.85; track_buffer (30 vs 45) is irrelevant "
        f"({sw['marginal_means']['ShanghaiTech/ML0_ncnn']['buffer']['30']:.2f} vs {sw['marginal_means']['ShanghaiTech/ML0_ncnn']['buffer']['45']:.2f}). Tracks drop from "
        f"{vs['cells'][dflt]['agg']['all']['n_tracks']:,} to {vs['cells'][fin]['agg']['all']['n_tracks']:,}, scored-frame coverage rises {pct(vs['cells'][dflt]['coverage']['all'], 1)} → "
        f"{pct(vs['cells'][fin]['coverage']['all'], 1)}. "
        f"The frontend ordering is unchanged under the locked tracker (SHT S4 all: ML0 {vs['cells'][fin]['auc']['S4']['all']:.2f}, MLP "
        f"{ver['series']['ShanghaiTech/MLP_ncnn']['cells'][fin]['auc']['S4']['all']:.2f}, stock@640 {ver['series']['ShanghaiTech/stock']['cells'][fin]['auc']['S4']['all']:.2f})."))
    ml_fin, ub_fin = vs["cells"][fin]["auc"], ver["series"]["UBnormal/ML0_mps"]["cells"][fin]["auc"]
    ad(f"**8. Locked configuration for E02.** Frontend `{lock['frontend']['name']}` ({lock['frontend']['relative_flops_vs_stock640']:.2f}x stock@640 FLOPs), tracker `{fin}`, S3/S4 "
       f"causal scoring with E00's sentinel/hold. Quality reference to beat or approach under CPU limits (locked frontend + tracker): ShanghaiTech S4 {ml_fin['S4']['all']:.2f} "
       f"(dev {ml_fin['S4']['dev']:.2f}, confirmation {ml_fin['S4']['confirmation']:.2f}; S0 {ml_fin['S0']['all']:.2f}); UBnormal (ML0 MPS proxy) S4 {ub_fin['S4']['all']:.2f} "
       f"(dev {ub_fin['S4']['dev']:.2f}, confirmation {ub_fin['S4']['confirmation']:.2f}). With the E00 default tracker the same numbers are {ml['S4']['all']:.2f} and "
       f"{A(UB, 'ML0_mps')['S4']['all']:.2f}. Rules: "
       f"{sum(1 for r in S['rules'].values() if r['status'] == 'PASS')} pass, {sum(1 for r in S['rules'].values() if r['status'] == 'FAIL')} fail, "
       f"{sum(1 for r in S['rules'].values() if r['status'] == 'INFO')} informational.")
    return "\n\n".join(lines)


def fmt_pp(r: dict) -> str:
    return f"{100 * r['delta']:+.1f} [{100 * r['ci95'][0]:+.1f}, {100 * r['ci95'][1]:+.1f}]"


def A_agg(arms: dict, ds: str, name: str) -> dict:
    return arms[f"{ds}/{name}"]["agg"]["all"]


LIMITS = """## 9. Limitations and honest caveats

* **Dev is small.** ShanghaiTech has 35 dev clips (10 scenes); dev CIs are wide (about ±4 pt on a single series) and the dev-only choices (frontend lock, tracker sweep) can only separate effects of several points. That is why the rules require a margin *and* a CI, and why the confirmation and full-set numbers are always shown next to the dev ones.
* **Single seed, fixed checkpoints.** No model was retrained; every number is one deterministic replay of a cached detector (seed 0 only fixes the bootstrap). Differences between MPS FP32 and NCNN FP16 detections are tiny (section 3.1) but caches were produced once.
* **UBnormal has no NCNN, no @800 and no ML0/MLP NCNN cache**; its quality claims use MPS FP32 proxies and stock@960 as the high-resolution comparator. Its stranded-window analysis for NCNN arms is therefore absent.
* **Reference for recall is a model, not ground truth.** Recall is measured against released AlphaPose person boxes (EXP-011/022 protocol); a person the reference misses cannot be a miss here, and a box that matches a reference box is not necessarily a *good pose*. Pose quality is reflected only through the scorer.
* **"Recoverable" is an upper bound.** A 0.05-0.25 cached box that overlaps a missed person (IoU ≥ 0.3) may be a duplicate, a part, or a false positive; the sweep, not the count, decides whether it helps. Second-stage boxes here may only continue live tracks (BoT-SORT semantics); spawning tracks from low-confidence boxes is the `c` factor of the grid.
* **Tracker settings are replayed from the detector cache** (conf ≥ 0.01, NMS IoU 0.7 fixed at detection time): any setting that would change NMS or input resolution is a different frontend, not a tracker cell. GMC and ReID stay disabled (as in E00).
* **Wilson intervals (QB2) treat person-frames as independent**, which they are not (the same person appears in hundreds of frames), so they are far too narrow; the paired video-bootstrap recall differences in section 4 are the honest uncertainty, and they also exclude 0 for ML0 vs stock in the <48 px and 48-96 px groups.
* **Selection bias of the tracker lock.** The cell was chosen as the best of 36 on 35 dev clips; the dev gain (+5.0) shrank to +2.0 on confirmation (CI lower bound +0.2). The confirmation-set effect is the number to plan with; the confirmation clips informed only T3 (a revert-only check), not the choice.
* **The frontend lock was decided at the E00 default tracker** (pre-registered order: frontend first, tracker second). Section 6.1 shows the frontend ordering under the locked tracker as point estimates only; no re-selection was done.
* **No cost claims.** FLOPs ratios are nominal; wall-clock, memory and CPU feasibility are E02's question, not measured here.
* **The scorer is frozen** (trained on the released AlphaPose poses); E01 does not separate how much of the released-vs-detector gap is missed people, tracking fragmentation or a pose-domain shift of the scorer's input. Retraining the scorer is out of scope.
"""


def render_report() -> None:
    S = json.loads((EXP_DIR / "summary.json").read_text())
    S["_headline"], S["_limits"] = build_headline(S), LIMITS
    text = render_report_text(S)
    (EXP_DIR / "report.md").write_text(text)
    print(f"wrote {EXP_DIR / 'report.md'} ({len(text.splitlines())} lines)")


# =====================================================================================================================
# 14. CLI
# =====================================================================================================================
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["lock", "run", "report", "all"])
    parser.add_argument("--procs", type=int, default=max(1, min(8, (os.cpu_count() or 2) - 2)), help="fork-pool size (bootstrap, replay)")
    parser.add_argument("--force", nargs="*", default=[], help="stages to recompute: arms boot recall replay_check sweep verify (or all)")
    args = parser.parse_args()
    if args.command in ("lock", "all"):
        cmd_lock()
    if args.command in ("run", "all"):
        run_all(args.procs, set(args.force))
    if args.command in ("report", "all"):
        render_report()


if __name__ == "__main__":
    main()
