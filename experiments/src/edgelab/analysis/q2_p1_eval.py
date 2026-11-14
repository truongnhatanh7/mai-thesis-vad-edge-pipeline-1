"""Shared evaluation engine of Q2-P1-E06 / E07: tracks from a detector cache -> window table of any scorer -> S0 / S4 AUC + fast paired bootstrap.

Everything is built on the E00 protocol code (`q2_p1_e00`: window rule, landing, causal smoothing, sentinel hold, global fill) and the E01 tracker
replay; nothing of the protocol is re-implemented here except

    * `replay_tracks`   BoT-SORT (E01 cell) or TrackTrack (Ultralytics 8.4.201 `tracktrack`) over a detector cache, with an on-disk track cache
    * `build_table`     window table for any scorer with `score_tracks` (frozen STG-NF, LoRaD flows, TrajVAD) and any segment length
    * `evaluate_table`  S0 / S4 series, AUCs (all / dev / confirmation), coverage, track statistics (E01 `clip_track_stats`)
    * `AucBoot`         exact pooled-AUC video bootstrap in O(R n^2) from a per-video-pair rank matrix (same statistic as sklearn's pooled AUC)
"""

from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from edgelab.analysis import q2_p1_e00 as e00
from edgelab.analysis import q2_p1_e01 as e01

DATA_ROOT = e00.DATA_ROOT
DATASETS = e00.DATASETS
EXPERIMENTS_ROOT = e00.EXPERIMENTS_ROOT
STGNF_TRAIN = {"ShanghaiTech": e00.STGNF_ROOT / "data/ShanghaiTech/pose/train", "UBnormal": e00.STGNF_ROOT / "data/UBnormal/pose/train"}
STGNF_TEST = {"ShanghaiTech": e00.STGNF_ROOT / "data/ShanghaiTech/pose/test", "UBnormal": e00.STGNF_ROOT / "data/UBnormal/pose/test"}
QUANTILE = {"ShanghaiTech": "max", "UBnormal": "p50"}  # E00 locked sentinel quantile
HOLD = 0  # E00 locked hold (both datasets)


# =====================================================================================================================
# Dataset context (splits, labels)
# =====================================================================================================================
@dataclass
class DatasetCtx:
    dataset: str
    names: list[str]
    labels: list[np.ndarray]
    dev_idx: list[int]
    conf_idx: list[int]
    fps: float
    seg_default: int


def load_ctx() -> dict[str, DatasetCtx]:
    cfg = yaml.safe_load((e00.EXP_DIR / "config.yaml").read_text())
    out = {}
    for ds in DATASETS:
        names, labels = e00.dataset_clips(ds)
        pos = {c: i for i, c in enumerate(names)}
        out[ds] = DatasetCtx(ds, names, labels, [pos[c] for c in cfg["datasets"][ds]["splits"]["test_dev"]],
                             [pos[c] for c in cfg["datasets"][ds]["splits"]["test_confirmation"]], e00.FPS[ds], e00.SEG_LENS[ds])
    return out


# =====================================================================================================================
# Trackers and replay
# =====================================================================================================================
@dataclass(frozen=True)
class TrackerSpec:
    kind: str = "botsort"  # botsort | tracktrack
    c: float = 0.25  # track_high_thresh = new_track_thresh
    low: float | None = None  # track_low_thresh (None: BoT-SORT single-stage default of E00)
    buffer: int = 30
    match: float = 0.80
    min_track_len: int = 3  # tracktrack only

    @property
    def label(self) -> str:
        if self.kind == "botsort":
            return e01.TrackerCell(self.c, self.low, self.buffer, self.match).label.replace("|", "_")
        low = "off" if self.low is None else f"{self.low:.2f}"
        return f"tt_c{self.c:.2f}_low{low}_m{self.match:.2f}_b{self.buffer}_n{self.min_track_len}"

    @property
    def gate(self) -> float:
        if self.kind == "botsort":
            return e01.TrackerCell(self.c, self.low, self.buffer, self.match).gate
        return self.c if self.low is None else self.low


BOTSORT_DEFAULT = TrackerSpec("botsort", 0.25, None, 30, 0.80)  # E00 default
BOTSORT_LOCKED = TrackerSpec("botsort", 0.20, 0.05, 30, 0.85)  # E01 locked two-stage
TRACKTRACK_DEFAULT = TrackerSpec("tracktrack", 0.6, 0.25, 30, 0.7)  # Ultralytics tracktrack.yaml (MOT17-tuned)


def build_tracktrack(spec: TrackerSpec):
    from ultralytics.trackers.track import TRACKER_MAP
    from ultralytics.utils import ROOT, YAML, IterableSimpleNamespace

    cfg = YAML.load(ROOT / "cfg/trackers/tracktrack.yaml")
    cfg.update(track_high_thresh=spec.c, new_track_thresh=spec.c, track_low_thresh=0.0 if spec.low is None else spec.low,
               track_buffer=spec.buffer, match_thresh=spec.match, min_track_len=spec.min_track_len, gmc_method="none", with_reid=False)
    return TRACKER_MAP["tracktrack"](IterableSimpleNamespace(**cfg))


def build_tracker(spec: TrackerSpec):
    return build_tracktrack(spec) if spec.kind == "tracktrack" else e01.build_tracker_cell(e01.TrackerCell(spec.c, spec.low, spec.buffer, spec.match))


def replay_clip(frames: list[dict], frame_ids: np.ndarray, spec: TrackerSpec) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Detector-cache frames -> {track id: (frame ids, keypoints (n, 17, 3))}; boxes below the gate never reach the tracker."""
    from edgelab.pose.replay import Detections

    tracker = build_tracker(spec)
    per: dict[int, list] = {}
    gate = spec.gate
    for frame, frame_id in zip(frames, frame_ids):
        keep = frame["conf"] >= gate
        kept = {k: v[keep] for k, v in frame.items()}
        for row in tracker.update(Detections(kept["boxes"], kept["conf"], kept["cls"]), None):
            per.setdefault(int(row[4]), []).append((int(frame_id), kept["keypoints"][int(row[7])].astype(float)))
    return {pid: (np.array([f for f, _ in v], dtype=np.int64), np.stack([k for _, k in v])) for pid, v in sorted(per.items())}


def _replay_worker(job: tuple) -> tuple[str, dict]:
    from edgelab.pose.detcache import load_detections

    clip, path, spec = job
    frames, data = load_detections(Path(path))
    return clip, replay_clip(frames, data["frame_ids"], spec)


def replay_tracks(cache_dir: Path, clips: list[str], spec: TrackerSpec, procs: int = 6, work: Path | None = None, tag: str | None = None) -> list[dict]:
    """Tracks of every clip (missing cache file = no tracks); pickled under `work` when given (key: tag + spec label + file sizes)."""
    cache_dir = Path(cache_dir)
    store = None
    if work is not None:
        stamp = hashlib.sha256((str(cache_dir) + spec.label + "|".join(clips) + "|".join(f"{p.name}:{p.stat().st_size}" for p in sorted(cache_dir.glob("*.npz")))).encode()).hexdigest()[:12]
        store = Path(work) / "tracks" / f"{tag or cache_dir.parent.parent.name + '_' + cache_dir.name}__{spec.label}__{stamp}.pkl"
        if store.exists():
            return pickle.loads(store.read_bytes())
    jobs = [(c, cache_dir / f"{c}.npz", spec) for c in clips if (cache_dir / f"{c}.npz").exists()]
    with mp.get_context("spawn").Pool(procs) as pool:
        done = dict(pool.imap_unordered(_replay_worker, jobs, chunksize=2))
    out = [done.get(c, {}) for c in clips]
    if store is not None:
        store.parent.mkdir(parents=True, exist_ok=True)
        store.write_bytes(pickle.dumps(out))
    return out


def released_tracks(dataset: str, clips: list[str], pose_dir: Path | None = None) -> list[dict]:
    files = e00.index_pose_dir(pose_dir or STGNF_TEST[dataset], dataset)
    return [e00.read_pose_clip(files[c]) if c in files else {} for c in clips]


def train_tracks(dataset: str) -> list[dict]:
    """Released AlphaPose training tracks (normal clips): used to calibrate the sentinel (and to train everything)."""
    files = sorted(STGNF_TRAIN[dataset].glob("*_tracked_person.json"))
    return [e00.read_pose_clip(f) for f in files]


# =====================================================================================================================
# Window tables
# =====================================================================================================================
def track_key(frames: np.ndarray, kp: np.ndarray) -> bytes:
    return e01.track_key(frames, kp)


def tracks_hash(tracks_by_clip: list[dict]) -> str:
    h = hashlib.sha256()
    for tracks in tracks_by_clip:
        for pid, (frames, kp) in tracks.items():
            h.update(track_key(frames, kp))
        h.update(b"|")
    return h.hexdigest()[:16]


class FrozenStgnfScorer:
    """Frozen released STG-NF (E00 `Scorer`), per-window normalisation, via the track interface."""

    name = "STG-NF"

    def __init__(self, dataset: str):
        self.dataset, self.seg_len = dataset, e00.SEG_LENS[dataset]
        self.core = e00.Scorer(dataset)

    def score_tracks(self, tracks: list[tuple[np.ndarray, np.ndarray]]) -> list[np.ndarray]:
        wins, counts = [], []
        for frames, kp in tracks:
            s = e00.complete_window_starts(frames, self.seg_len)
            counts.append(len(s))
            if len(s):
                wins.append(e00.gather_windows(kp, s, self.seg_len))
        norm = self.core.score(np.concatenate(wins)) if wins else np.zeros(0)
        out, at = [], 0
        for c in counts:
            out.append(norm[at:at + c])
            at += c
        return out


def build_table(tracks_by_clip: list[dict], scorer, cache: dict | None = None, chunk: int = 256) -> dict[str, np.ndarray]:
    """Same table as `e00.build_window_table` (columns `e00.WINDOW_COLS`) for any scorer with `.seg_len` and `.score_tracks`.

    A window is scored once per distinct track content (hash of frames + keypoints): `cache` maps hash -> normality."""
    seg = scorer.seg_len
    cache = {} if cache is None else cache
    pending: dict[bytes, tuple] = {}
    for tracks in tracks_by_clip:
        for frames, kp in tracks.values():
            if len(e00.complete_window_starts(frames, seg)) == 0:
                continue
            key = track_key(frames, kp)
            if key not in cache and key not in pending:
                pending[key] = (frames, kp)
    keys = list(pending)
    for i in range(0, len(keys), chunk):
        part = keys[i:i + chunk]
        for k, v in zip(part, scorer.score_tracks([pending[k] for k in part])):
            cache[k] = np.asarray(v, dtype=np.float64)
    cols: dict[str, list] = {c: [] for c in e00.WINDOW_COLS if c != "normality"}
    norms = []
    n_tracks, n_poses = np.zeros(len(tracks_by_clip), dtype=np.int64), np.zeros(len(tracks_by_clip), dtype=np.int64)
    for ci, tracks in enumerate(tracks_by_clip):
        for pid, (frames, kp) in tracks.items():
            n_tracks[ci] += 1
            n_poses[ci] += len(frames)
            starts = e00.complete_window_starts(frames, seg)
            if len(starts) == 0:
                continue
            cols["clip_idx"].append(np.full(len(starts), ci, dtype=np.int32))
            cols["person"].append(np.full(len(starts), pid, dtype=np.int32))
            cols["start_idx"].append(starts.astype(np.int32))
            cols["n_track"].append(np.full(len(starts), len(frames), dtype=np.int32))
            cols["start_frame"].append(frames[starts].astype(np.int32))
            cols["last_frame"].append(frames[starts + seg - 1].astype(np.int32))
            cols["in_offline"].append(e00.stgnf_offline_window_mask(starts, len(frames), seg))
            norms.append(cache[track_key(frames, kp)])
    table = {c: (np.concatenate(v) if v else np.zeros(0, dtype=bool if c == "in_offline" else np.int32)) for c, v in cols.items()}
    table["normality"] = np.concatenate(norms) if norms else np.zeros(0)
    table["clip_n_tracks"], table["clip_n_poses"] = n_tracks, n_poses
    return table


def cached_table(path: Path, tracks_by_clip: list[dict], scorer, scorer_id: str) -> dict[str, np.ndarray]:
    """`build_table` with an on-disk npz keyed by (track content hash, scorer id)."""
    stamp = f"{tracks_hash(tracks_by_clip)}|{scorer_id}|{scorer.seg_len}|v1"
    path = Path(path)
    if path.exists():
        z = np.load(path, allow_pickle=False)
        if str(z["stamp"]) == stamp:
            return {k: z[k] for k in z.files if k != "stamp"}
    table = build_table(tracks_by_clip, scorer)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, stamp=np.array(stamp), **table)
    return table


def train_sentinels(scorer, dataset: str, cache_path: Path | None = None) -> dict[str, float]:
    """Quantiles (p50 / p90 / p99 / max) of the scorer's normality over every complete window of the released-pose training tracks (E00 rule)."""
    if cache_path is not None and Path(cache_path).exists():
        return json.loads(Path(cache_path).read_text())
    tracks = [t for clip in train_tracks(dataset) for t in clip.values()]
    parts = []
    for i in range(0, len(tracks), 512):
        parts += scorer.score_tracks(tracks[i:i + 512])
    norm = np.concatenate([p for p in parts if len(p)])
    values = e00.sentinel_values(norm)
    if cache_path is not None:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        Path(cache_path).write_text(json.dumps(values))
    return values


# =====================================================================================================================
# Evaluation
# =====================================================================================================================
def evaluate_table(table: dict, tracks_by_clip: list[dict], ctx: DatasetCtx, seg: int, sentinel: float, hold: float = HOLD,
                   with_s0: bool = True) -> dict:
    """S0 (offline paper) and S4 (strict causal) per-clip scores and AUCs (all / dev / confirmation) + coverage + track stats."""
    stub = e00.Source("q2", ctx.dataset, Path("."), None, None, None, "")
    d = e00.SourceData(stub, ctx.names, ctx.labels, table, seg, ctx.dev_idx, ctx.conf_idx)
    raw = d.raw_per_frame("completion", "all")
    series = {"S4": e00.deliver(raw, sentinel, hold)[0]}
    delivery = e00.deliver(raw, sentinel, hold)[1]
    if with_s0:
        series["S0"], _ = d.step_scores("S0")
    auc = {s: e01.auc_subsets(ctx.labels, v, ctx.dev_idx, ctx.conf_idx) for s, v in series.items()}
    rows, lengths_all, gaps_all = [], [], []
    dev = set(ctx.dev_idx)
    for i, clip in enumerate(ctx.names):
        stats, lengths, gaps = e01.clip_track_stats(tracks_by_clip[i], seg)
        lengths_all.append(lengths)
        gaps_all.append(gaps)
        lab, fresh = ctx.labels[i], np.isfinite(raw[i])
        rows.append({"video": clip, "split": "dev" if i in dev else "confirmation", **stats, "n_frames": len(lab),
                     "anomalous_frames": int(lab.sum()), "n_normal_frames": int((lab == 0).sum()), "scorable_frames": int(fresh.sum()),
                     "scorable_normal": int(fresh[lab == 0].sum()), "scorable_anomalous": int(fresh[lab == 1].sum())})
    agg = {"all": e01.aggregate_track_rows(rows, seg)}
    for split in ("dev", "confirmation"):
        agg[split] = e01.aggregate_track_rows([r for r in rows if r["split"] == split], seg)
    return {"series": series, "auc": auc, "coverage": e00.coverage_fractions(raw, ctx.labels), "delivery": delivery, "rows": rows, "agg": agg,
            "length": e01.length_summary(np.concatenate(lengths_all), seg), "gap": e01.gap_summary(np.concatenate(gaps_all)),
            "n_windows": int(len(table["normality"]))}


# =====================================================================================================================
# Fast video bootstrap of the pooled AUC
# =====================================================================================================================
def pair_matrix(labels: list[np.ndarray], scores: list[np.ndarray], idx: list[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """U[i, j] = #(positive frame of clip i scored above a negative frame of clip j) + 0.5 #ties; P[i], N[j] = positive / negative counts.

    The pooled AUC of any multiset of clips with multiplicities w is (w' U w) / ((w . P)(w . N)): identical to sklearn's roc_auc_score on the
    concatenated frames (ties count one half)."""
    n = len(idx)
    pos = [scores[i][labels[i] == 1] for i in idx]
    neg = [np.sort(scores[i][labels[i] == 0]) for i in idx]
    P = np.array([len(p) for p in pos], dtype=np.float64)
    N = np.array([len(x) for x in neg], dtype=np.float64)
    allpos = np.concatenate(pos) if P.sum() else np.zeros(0)
    owner = np.repeat(np.arange(n), P.astype(int))
    U = np.zeros((n, n))
    for j in range(n):
        if N[j] == 0:
            continue
        lo = np.searchsorted(neg[j], allpos, side="left")
        hi = np.searchsorted(neg[j], allpos, side="right")
        U[:, j] = np.bincount(owner, weights=lo + 0.5 * (hi - lo), minlength=n)
    return U, P, N


def resample_weights(n: int, P: np.ndarray, N: np.ndarray, n_resamples: int = 2000, seed: int = 0) -> np.ndarray:
    """(R, n) multiplicities of resampled clips; resamples without both classes are redrawn (as `bootstrap_ci_over_videos` does)."""
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < n_resamples:
        w = np.bincount(rng.integers(0, n, n), minlength=n).astype(np.float64)
        if (w @ P) > 0 and (w @ N) > 0:
            out.append(w)
    return np.stack(out)


def boot_auc(U: np.ndarray, P: np.ndarray, N: np.ndarray, W: np.ndarray) -> np.ndarray:
    num = np.einsum("ri,ij,rj->r", W, U, W)
    return num / ((W @ P) * (W @ N))


def full_auc(U: np.ndarray, P: np.ndarray, N: np.ndarray) -> float:
    w = np.ones(len(P))
    return float(w @ U @ w / (P.sum() * N.sum()))


def percentile_ci(x: np.ndarray) -> list[float]:
    return [float(100 * np.quantile(x, 0.025)), float(100 * np.quantile(x, 0.975))]


@dataclass
class SeriesStats:
    """Pair matrices of one (arm, protocol) series for the three clip subsets; `seeds` stacked (the statistic is the seed mean)."""
    mats: dict[str, list[tuple]] = field(default_factory=dict)  # subset -> [(U, P, N) per seed]


class AucBoot:
    """Bootstrap context of a dataset: one weight matrix per subset (shared by every arm, so every contrast is paired)."""

    def __init__(self, ctx: DatasetCtx, n_resamples: int = 2000, seed: int = 0):
        self.ctx = ctx
        self.subsets = {"all": list(range(len(ctx.names))), "dev": ctx.dev_idx, "confirmation": ctx.conf_idx}
        self.weights: dict[str, np.ndarray] = {}
        for name, idx in self.subsets.items():
            labs = [ctx.labels[i] for i in idx]
            P = np.array([lab.sum() for lab in labs], dtype=np.float64)
            N = np.array([len(lab) - lab.sum() for lab in labs], dtype=np.float64)
            self.weights[name] = resample_weights(len(idx), P, N, n_resamples, seed)

    def stats(self, scores_by_seed: list[list[np.ndarray]]) -> SeriesStats:
        s = SeriesStats()
        for name, idx in self.subsets.items():
            s.mats[name] = [pair_matrix(self.ctx.labels, sc, idx) for sc in scores_by_seed]
        return s

    def point(self, s: SeriesStats, subset: str) -> float:
        return 100 * float(np.mean([full_auc(*m) for m in s.mats[subset]]))

    def samples(self, s: SeriesStats, subset: str) -> np.ndarray:
        return np.mean([boot_auc(*m, self.weights[subset]) for m in s.mats[subset]], axis=0)

    def ci(self, s: SeriesStats, subset: str) -> list[float]:
        return percentile_ci(self.samples(s, subset))

    def diff(self, a: SeriesStats, b: SeriesStats, subset: str) -> dict:
        d = self.samples(a, subset) - self.samples(b, subset)
        lo, hi = percentile_ci(d)
        return {"delta": self.point(a, subset) - self.point(b, subset), "ci95": [lo, hi], "excludes_zero": bool(lo > 0 or hi < 0)}


def now() -> str:
    return time.strftime("%H:%M:%S")
