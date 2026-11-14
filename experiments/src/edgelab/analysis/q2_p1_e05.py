"""Q2-P1-E05 — Confirmation of the locked Q2-P1 system (no training).

    uv run --extra pose python -m edgelab.analysis.q2_p1_e05 lock        # plan.md is written by hand first; this pins config.yaml (hashes, cells, arms, rules)
    uv run --extra pose python -m edgelab.analysis.q2_p1_e05 tables      # extra run tables (ML0 + default tracker, stock + default tracker, C1 re-track) + builder checks
    uv run --extra pose python -m edgelab.analysis.q2_p1_e05 run         # 11 arms x 15 cells x 2 datasets (matched cost) + correctness checks
    uv run --extra pose python -m edgelab.analysis.q2_p1_e05 strata      # per-event strata (duration / new entry / motion / crowd)
    uv run --extra pose python -m edgelab.analysis.q2_p1_e05 analyze     # Part A (2,000-resample bootstraps, ladder, ReTrack, guard) + Part B (subgroups)
    uv run --extra pose python -m edgelab.analysis.q2_p1_e05 sustained   # Part C2 long-horizon simulation
    uv run --extra pose python -m edgelab.analysis.q2_p1_e05 vm          # Part C1 real sustained run in the edge-os VM (host driver; worker = q2_p1_e05_vm)
    uv run --extra pose python -m edgelab.analysis.q2_p1_e05 report      # results.csv, events.csv, sustained.csv, decisions.jsonl, summary.json, report.md

Builds on E00 (causal S3/S4 protocol, splits), E01 (locked frontend / tracker), E02 (VM pools, FIFO queue, Fixed-Burst), E03 (run tables, C1, guard)
and E04 (stock NCNN cache; MV / INT8 not adopted). New here: the Guard as a policy for intact-window schedules, a non-oracle C1 (`C1-ReTrack`), event strata,
the 2,000-resample paired bootstraps of the whole ladder, a long-horizon simulation with queue carry-over, and a real sustained run in the VM.

Pure functions (guard, strata, event outcomes, bootstraps, drift metrics, verdict rules) are unit-tested without data in tests/test_q2_p1_e05.py.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import multiprocessing as mp
import pickle
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml

from edgelab.analysis import q2_p1_e00 as e00
from edgelab.analysis import q2_p1_e01 as e01
from edgelab.analysis import q2_p1_e02 as e02
from edgelab.analysis import q2_p1_e03 as e03

DATASETS = e00.DATASETS
SHT, UB = DATASETS
PRIMARY = e03.PRIMARY
FPS = e00.FPS
SEED = e00.SEED
POOL_SEED = e02.POOL_SEED
N_BOOT = 2000
MACHINE_TAG = e02.MACHINE_TAG
EXP_DIR = e00.EXPERIMENTS_ROOT / "Q2-P1-E05_confirmation"
DATA_ROOT = e00.DATA_ROOT
WORK_DIR = DATA_ROOT / "q2-p1-e05"
VM_EXCHANGE = Path("/Users/nanh/thesis-data/vm-exchange")
VM_DIR = VM_EXCHANGE / "q2-p1-e05"
TRACKER_LOCKED = e02.TRACKERS["locked"]
TRACKER_DEFAULT = e02.TRACKERS["default"]
SHED_ON_MS, SHED_OFF_MS = e03.BACKLOG_SHED_ON_MS, e03.BACKLOG_SHED_OFF_MS
CAP_MS = e03.CAP_MS
INF = float("inf")

CELLS_E03 = list(e03.CELLS)
CELLS_NATIVE = [(1.0, 0.25), (1.0, 0.50), (1.0, 1.00)]
CELLS = CELLS_E03 + CELLS_NATIVE
BUDGET_CELLS = [(2.0, 0.33), (2.0, 0.50), (2.0, 0.75)]
OVERLOAD_CELL = (3.0, 1.00)
NATIVE_CELL = (1.0, 1.00)
CELL_GROUPS = {"budget": BUDGET_CELLS, "overload": [OVERLOAD_CELL], "native": [NATIVE_CELL]}
CELL_SETS = {"primary12": CELLS_E03, "native3": CELLS_NATIVE, "all15": CELLS}

# arm -> world (table / cost variant), schedule, reuse level, guard, K=1, hold reference ("own" = its own period, "fixed" = the cell's Fixed period)
ARMS: dict[str, dict] = {
    "Stock-Default-K1": dict(world="stock_default", lvl=0, guard=False, k1=True, ref="own"),
    "ML0-Default-K1": dict(world="ml0_default", lvl=0, guard=False, k1=True, ref="own"),
    "Full-Rate": dict(world="ml0_locked", lvl=0, guard=False, k1=True, ref="own"),
    "Stock-Baseline": dict(world="stock_default", lvl=0, guard=False, k1=False, ref="own"),
    "ML0-Default-Fixed": dict(world="ml0_default", lvl=0, guard=False, k1=False, ref="own"),
    "Fixed": dict(world="ml0_locked", lvl=0, guard=False, k1=False, ref="own"),
    "Fixed+Guard": dict(world="ml0_locked", lvl=0, guard=True, k1=False, ref="fixed"),
    "Cache-C1": dict(world="ml0_locked", lvl=3, guard=False, k1=False, ref="fixed"),
    "Cache-C1+Guard": dict(world="ml0_locked", lvl=3, guard=True, k1=False, ref="fixed"),
    "C1-ReTrack": dict(world="ml0_retrack", lvl=3, guard=False, k1=False, ref="fixed"),
    "C1-ReTrack+Guard": dict(world="ml0_retrack", lvl=3, guard=True, k1=False, ref="fixed"),
}
E03_REF_ARMS = ("Independent-Combo",)  # E03 results re-used as a reference (12 E03 cells only; not re-simulated)
FAMILY = {"Stock-Default-K1": "stock_default", "Stock-Baseline": "stock_default", "ML0-Default-K1": "ml0_default", "ML0-Default-Fixed": "ml0_default"}
SUB_ARMS = ("Full-Rate", "Fixed", "Cache-C1", "Cache-C1+Guard", "C1-ReTrack", "Stock-Baseline")
SUSTAINED_ARMS = ("Fixed-K1", "Cache-C1", "Cache-C1+Guard", "Fixed@b0.50", "Cache-C1+Guard@b0.50")
SUSTAINED_S = (1.0, 2.0, 2.5, 3.0)
SEG_MINUTES = 10.0
VM_RUNS = [("fixed_k1", 1.0), ("cache_c1_guard", 1.0), ("fixed_k1", 3.0), ("cache_c1_guard", 3.0), ("fixed_k1", 3.5), ("cache_c1_guard", 3.5), ("cache_c1", 3.5)]
VM_PASSES = 4  # 4 x 2,880 = 11,520 frames
VM_RSS_MARKS = (0, 2500, 5000, 10000)
D1B_TOL_PT = 0.75  # E3-D2 chain hand-off handicap (-0.45 pt SHT) + margin; see plan Amendment A1
E01_REF_ML0 = {SHT: {"locked": 76.82, "default": 73.78}, UB: {"locked": 61.08, "default": 58.04}}  # E01 S4 all-clip AUC (zero-delay), regression anchor
CONTRASTS = {  # id -> (A, B): A - B
    "ladder.frontend": ("ML0-Default-K1", "Stock-Default-K1"),
    "ladder.tracker": ("Full-Rate", "ML0-Default-K1"),
    "ladder.budget": ("Fixed", "Full-Rate"),
    "ladder.cache": ("Cache-C1", "Fixed"),
    "ladder.guard": ("Cache-C1+Guard", "Cache-C1"),
    "e2e.cache_vs_stock": ("Cache-C1", "Stock-Baseline"),
    "e2e.guard_vs_stock": ("Cache-C1+Guard", "Stock-Baseline"),
    "e2e.fixed_vs_stock": ("Fixed", "Stock-Baseline"),
    "e2e.fixed_vs_default_fixed": ("Fixed", "ML0-Default-Fixed"),
    "cache.vs_fullrate": ("Cache-C1", "Full-Rate"),
    "guard.fixed": ("Fixed+Guard", "Fixed"),
    "retrack.vs_fixed": ("C1-ReTrack", "Fixed"),
    "retrack.vs_oracle": ("C1-ReTrack", "Cache-C1"),
    "retrack_guard.vs_fixed": ("C1-ReTrack+Guard", "Fixed"),
    "ref.guard_vs_independent": ("Cache-C1+Guard", "Independent-Combo"),
    "ref.cache_vs_independent": ("Cache-C1", "Independent-Combo"),
}
SUB_CONTRASTS = {"cache_vs_fixed": ("Cache-C1", "Fixed"), "guard_vs_fixed": ("Cache-C1+Guard", "Fixed"), "cache_vs_fullrate": ("Cache-C1", "Full-Rate"),
                 "retrack_vs_fixed": ("C1-ReTrack", "Fixed")}
SUB_METRICS = ("recall_0.5", "recall_1.0", "recall_2.0", "missed", "delay_p50_ms", "delay_p90_ms", "loc_auc")
NONINF_RECALL_PT = 5.0  # V6 / V7 non-inferiority margin (points of recall)
NONINF_DELAY_MS = 250.0  # V7 delay margin
NONINF_AUC_PT = 0.25  # V5a margin
SHORT_EVENT_S = 2.5
ENTRY_WINDOW_S = 1.0
CROWD_MIN_TRACKS = 4
ENTRY_MIN_POSES = 12
ENTRY_MIN_FIRST_FRAME = 3
LOC_PAD_S = 2.0


def plain(obj):
    return e00.to_plain(obj)


def now() -> str:
    return time.strftime("%H:%M:%S")


def sha(path: Path) -> str:
    return e00.sha256_file(path)


def cell_id(s: float, b: float) -> str:
    return e03.cell_id(s, b)


def cells_ids(cells) -> list[str]:
    return [cell_id(*c) for c in cells]


def family_of(arm: str) -> str:
    return FAMILY.get(arm, "ml0_locked")


# =====================================================================================================================
# 1. Pure: Guard policy (intact-window schedules), non-oracle C1 frames, pattern causality
# =====================================================================================================================
class GuardedFixedPolicy(e03.FixedPolicy):
    """`F(W)` with the Independent-Combo load-shedding guard.

    The guard has hysteresis on the queue wait at the start of a frame job: on when wait > `on_ms`, off when wait < `off_ms`. While on no new run starts
    (burst schedules: the start is *deferred* and lost) and a running run is aborted (its incomplete windows are never scored). A continuous schedule
    (W <= L) restarts its run on the next even frame with room once the guard is off. Whenever the guard never turns on the decisions are *exactly* the
    unguarded `FixedPolicy` ones (bit-identical simulation). The scoring thinning of the Scheduler guard is not ported (it would void the only window of a
    burst)."""

    name = "fixed+guard"

    def __init__(self, w, c, period: float, lvl: int = 0, tau2: float | None = None, on_ms: float = SHED_ON_MS, off_ms: float = SHED_OFF_MS):
        super().__init__(w, c, period, lvl, tau2)
        self.on_ms, self.off_ms = float(on_ms), float(off_ms)
        self.shed = False
        self.resume = False
        self.n_shed = self.n_deferred = self.n_aborted = self.n_resumed = self.n_on = 0

    def _advance_score(self, t: int) -> None:
        if t >= self.next_score:
            self.next_score += self.period

    def decide(self, t, n, g, now, wait, rl, can_start):
        was = self.shed
        if wait > self.on_ms:
            self.shed = True
        elif wait < self.off_ms:
            self.shed = False
        if self.shed and not was:
            self.n_on += 1
        cont = self.period < self.L + e03.START_GRID
        if self.shed:
            self.n_shed += 1
            self.resume = True
            if rl > 0:
                self.n_aborted += 1
            elif not cont:
                while self.bi < len(self.starts) and self.starts[self.bi] < t:
                    self.bi += 1
                if self.bi < len(self.starts) and t == self.starts[self.bi]:
                    self.n_deferred += 1
                    self.bi += 1
            if cont:
                self._advance_score(t)
            return False, False, self.lvl, self.tau2
        if not self.resume:
            return super().decide(t, n, g, now, wait, rl, can_start)
        if not cont:  # burst schedule: the base policy skips to the next start lazily
            self.resume = False
            return super().decide(t, n, g, now, wait, rl, can_start)
        if rl > 0:  # still inside a run that was never aborted (shed switched off before the next frame)
            self.resume = False
            return super().decide(t, n, g, now, wait, rl, can_start)
        if not can_start:
            self._advance_score(t)
            return False, False, self.lvl, self.tau2
        self.resume = False
        self.n_resumed += 1
        score = False
        if t >= self.next_score:
            score = True
            self.next_score += self.period
        return True, score, self.lvl, self.tau2

    def info(self) -> dict:
        return {"shed_frames": self.n_shed, "guard_on_events": self.n_on, "deferred_starts": self.n_deferred, "aborted_run_frames": self.n_aborted,
                "resumed_runs": self.n_resumed}


def make_policy5(w, c, spec: dict, period: float):
    if spec["guard"]:
        return GuardedFixedPolicy(w, c, period, spec["lvl"])
    return e03.FixedPolicy(w, c, period, spec["lvl"])


def retrack_frames(frames: list, reused: np.ndarray, src: np.ndarray) -> list:
    """Frames the tracker sees in a C1-re-track run: a reused frame carries the detections of its source detector frame (never its own)."""
    return [frames[int(src[o])] if reused[o] else frames[o] for o in range(len(frames))]


def retrack_source_ok(reused: np.ndarray, src: np.ndarray, a1: int = e03.A1) -> bool:
    """E5-D4: a reused frame's source is strictly earlier, at most `a1` frames old and itself a detector frame."""
    for o in np.nonzero(reused)[0]:
        s = int(src[o])
        if not (s < o and o - s <= a1 and not reused[s]):
            return False
    return True


# =====================================================================================================================
# 2. Pure: events, strata, per-event outcomes, bootstraps
# =====================================================================================================================
def short_frames(ds: str) -> int:
    return int(round(SHORT_EVENT_S * FPS[ds]))


def first_second(s: int, e: int, fps: float) -> slice:
    return slice(int(s), int(min(e, s + int(round(fps)) - 1)) + 1)


def track_summary(tracks: dict, n: int) -> tuple[np.ndarray, np.ndarray]:
    """From {pid: (frames, keypoints)} of one clip -> (first frames of tracks with >= 12 poses that do not exist at the clip start, simultaneous-track count per frame)."""
    starts, count = [], np.zeros(n, dtype=np.int32)
    for f, _ in tracks.values():
        f = np.asarray(f)
        count[f[f < n]] += 1
        if len(f) >= ENTRY_MIN_POSES and int(f[0]) >= ENTRY_MIN_FIRST_FRAME:
            starts.append(int(f[0]))
    return np.array(sorted(starts), dtype=np.int64), count


def classify_event(ds: str, s: int, e: int, starts: np.ndarray, n_active: np.ndarray, motion: np.ndarray, motion_thr: float) -> dict:
    fps = FPS[ds]
    sl = first_second(s, e, fps)
    w = int(round(ENTRY_WINDOW_S * fps))
    entry = bool(len(starts) and np.any((starts >= s - w) & (starts <= s + w)))
    m = motion[sl]
    m = m[np.isfinite(m)]
    fast = bool(len(m) and np.quantile(m, 0.9) >= motion_thr)
    crowd = bool(len(n_active[sl]) and np.median(n_active[sl]) >= CROWD_MIN_TRACKS)
    return {"short": bool(e - s + 1 <= short_frames(ds)), "entry": entry, "fast": fast, "crowd": crowd}


def per_event_delays(events: list[tuple[int, int]], series: np.ndarray, theta: float, min_len: int = e03.EVENT_MIN_FRAMES) -> np.ndarray:
    """Detection delay (frames) of every event of one clip, inf = missed. Same rule as `e03.event_metrics`: an alert run overlapping the event whose raise
    frame (start + min_len - 1) is <= the event end; delay = max(raise, event start) - event start (earliest alert)."""
    runs = e03.alert_runs(np.asarray(series) > theta, min_len)
    out = np.full(len(events), INF)
    for k, (s, e) in enumerate(events):
        for r0, r1 in runs:
            f = r0 + min_len - 1
            if r1 >= s and f <= e:
                d = max(f, s) - s
                if d < out[k]:
                    out[k] = float(d)
    return out


def window_mask(n: int, events: list[tuple[int, int]], fps: float, pad_s: float = LOC_PAD_S) -> np.ndarray:
    """Union of [s - pad, e + pad] over the events (clipped to the clip)."""
    m = np.zeros(n, dtype=bool)
    pad = int(round(pad_s * fps))
    for s, e in events:
        m[max(0, s - pad):min(n, e + pad + 1)] = True
    return m


def wquantile(vals_sorted: np.ndarray, weights: np.ndarray, q: float) -> np.ndarray:
    """Weighted lower quantile of sorted values. weights: (B, m) -> (B,)."""
    cw = np.cumsum(weights, axis=1)
    thr = q * cw[:, -1:]
    idx = np.minimum((cw < thr).sum(axis=1), len(vals_sorted) - 1)
    out = vals_sorted[idx].astype(np.float64)
    out[cw[:, -1] <= 0] = np.nan
    return out


class EventBoot:
    """Clip-level paired bootstrap of event metrics: the same multinomial clip weights as `e03.AucBoot` (same n, same seed) for every series."""

    def __init__(self, n_clips: int, n_boot: int = N_BOOT, seed: int = 0):
        self.n = n_clips
        self.W = np.random.default_rng(seed).multinomial(n_clips, np.full(n_clips, 1.0 / n_clips), size=n_boot).astype(np.float64)

    def rate(self, clip: np.ndarray, hit: np.ndarray) -> tuple[float, np.ndarray]:
        """Percentage of events with `hit`. -> (point, boot (B,))."""
        num = np.bincount(clip, weights=np.asarray(hit, dtype=np.float64), minlength=self.n)
        den = np.bincount(clip, minlength=self.n).astype(np.float64)
        bd = self.W @ den
        boot = np.where(bd > 0, 100 * (self.W @ num) / np.where(bd > 0, bd, 1.0), np.nan)
        return (100 * num.sum() / den.sum() if den.sum() > 0 else float("nan")), boot

    def quantile(self, clip: np.ndarray, values: np.ndarray, q: float) -> tuple[float, np.ndarray]:
        if len(values) == 0:
            return float("nan"), np.full(len(self.W), np.nan)
        order = np.argsort(values, kind="stable")
        v, c = np.asarray(values, dtype=np.float64)[order], np.asarray(clip)[order]
        pt = wquantile(v, np.ones((1, len(v))), q)[0]
        return float(pt), wquantile(v, self.W[:, c], q)


def ci95(vec: np.ndarray) -> tuple[float, float]:
    v = np.asarray(vec, dtype=np.float64)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return float("nan"), float("nan")
    return float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))


def diff_ci(pa: float, ba: np.ndarray, pb: float, bb: np.ndarray) -> dict:
    d = np.asarray(ba) - np.asarray(bb)
    lo, hi = ci95(d)
    return {"delta": float(pa - pb), "lo": lo, "hi": hi}


# =====================================================================================================================
# 3. Pure: sustained-operation metrics, verdict rules
# =====================================================================================================================
def segment_stats(wait: np.ndarray, dt_ms: float, minutes: float = SEG_MINUTES) -> list[dict]:
    """Queue wait statistics per `minutes` of stream time."""
    n_seg = max(1, int(round(minutes * 60000.0 / dt_ms)))
    out = []
    for k, a in enumerate(range(0, len(wait), n_seg)):
        x = np.asarray(wait[a:a + n_seg], dtype=np.float64)
        out.append({"segment": k, "start_min": a * dt_ms / 60000.0, "frames": int(len(x)), "wait_mean_ms": float(x.mean()), "wait_p50_ms": float(np.quantile(x, 0.5)),
                    "wait_p95_ms": float(np.quantile(x, 0.95)), "wait_p99_ms": float(np.quantile(x, 0.99)), "wait_max_ms": float(x.max())})
    return out


def drift_test(wait: np.ndarray) -> dict:
    """First third vs last third of the stream: mean wait difference (ms) and the least-squares slope of the wait (ms per stream hour)."""
    w = np.asarray(wait, dtype=np.float64)
    k = max(1, len(w) // 3)
    first, last = float(w[:k].mean()), float(w[-k:].mean())
    t = np.arange(len(w), dtype=np.float64)
    tc = t - t.mean()
    slope = float((tc * (w - w.mean())).sum() / max((tc * tc).sum(), 1e-12))
    return {"first_third_mean_ms": first, "last_third_mean_ms": last, "drift_ms": last - first, "slope_ms_per_frame": slope}


def queue_memory_mb(base_mb: float, wait_ms: float, dt_ms: float, size: tuple[int, int]) -> float:
    return base_mb + (wait_ms / dt_ms + 1.0) * size[0] * size[1] * 3 / 2 ** 20


def bounded(wait: np.ndarray, dt_ms: float, size: tuple[int, int], base_mb: float, cap_ms: float = CAP_MS, drift_ms: float = 100.0, mem_mb: float = 200.0) -> dict:
    """V8: max wait <= cap, p99 <= cap, last-third - first-third mean wait <= drift_ms and modelled queue memory (above the E02 RSS) <= mem_mb."""
    w = np.asarray(wait, dtype=np.float64)
    d = drift_test(w)
    mem = queue_memory_mb(base_mb, float(w.max()), dt_ms, size) - base_mb
    ok = bool(w.max() <= cap_ms and np.quantile(w, 0.99) <= cap_ms and d["drift_ms"] <= drift_ms and mem <= mem_mb)
    return {"bounded": ok, "max_ms": float(w.max()), "p99_ms": float(np.quantile(w, 0.99)), "drift_ms": d["drift_ms"], "queue_mem_mb": mem, **{k: d[k] for k in ("first_third_mean_ms", "last_third_mean_ms")}}


def rss_growth(rss_by_frame: dict[int, float], a: int = 2500) -> dict:
    ks = sorted(rss_by_frame)
    end = ks[-1]
    base = rss_by_frame[a] if a in rss_by_frame else rss_by_frame[ks[0]]
    return {"rss_ref_mb": base, "rss_end_mb": rss_by_frame[end], "growth_pct": 100.0 * (rss_by_frame[end] - base) / base}


def verdict_lock(c: dict) -> str:
    if c["lo"] > 0:
        return "confirmed"
    if c["hi"] < 0:
        return "rejected"
    return "directionally consistent" if c["delta"] > 0 else "not supported"


def verdict_replication(c: dict) -> str:
    if c["lo"] > 0:
        return "replicated"
    if c["hi"] < 0:
        return "not replicated"
    return "no significant effect (non-negative)" if c["delta"] >= 0 else "no significant effect (negative point estimate)"


def verdict_confirmed(c: dict) -> str:
    return "confirmed" if c["lo"] > 0 else ("rejected" if c["hi"] < 0 else "not confirmed")


def verdict_retrack(c_rt: dict, c_or: dict) -> str:
    """V4: ReTrack - Fixed CI lower > 0 and delta >= 50 % of the oracle Cache-C1 - Fixed delta -> holds; delta > 0 otherwise -> attenuated; CI upper < 0 -> does not hold."""
    if c_rt["hi"] < 0:
        return "does not hold"
    if c_rt["lo"] > 0 and c_rt["delta"] >= 0.5 * c_or["delta"]:
        return "holds without oracle identity"
    return "attenuated" if c_rt["delta"] > 0 else "does not hold"


def verdict_guard(noninf: dict | None, overload: dict | None) -> dict:
    out = {}
    out["no_harm"] = "not testable" if noninf is None else ("non-inferior" if noninf["lo"] > -NONINF_AUC_PT else "inferior")
    out["overload"] = "not testable" if overload is None else ("guard helps" if overload["lo"] > 0 else ("guard hurts" if overload["hi"] < 0 else "no significant effect"))
    return out


def verdict_noninf_recall(c: dict, margin: float = NONINF_RECALL_PT) -> str:
    return "non-inferior" if c["lo"] > -margin else "not shown non-inferior"


def verdict_delay(c: dict, margin: float = NONINF_DELAY_MS) -> str:
    return "not delayed" if (np.isfinite(c["hi"]) and c["hi"] < margin) else "not shown"


# =====================================================================================================================
# 4. Run tables: ML0 + default tracker, stock + default tracker, C1 re-track
# =====================================================================================================================
_G: dict = {}
VARIANTS = ("ml0_default", "stock_default", "ml0_retrack")


def tb_path(variant: str, ds: str) -> Path:
    return WORK_DIR / f"tables_{variant}_{ds}.npz"


def run_path(ds: str) -> Path:
    return WORK_DIR / f"run_{ds}.pkl"


def analysis_path(ds: str) -> Path:
    return WORK_DIR / f"analysis_{ds}.pkl"


def strata_path(ds: str) -> Path:
    return WORK_DIR / f"strata_{ds}.pkl"


def variant_spec(ds: str, variant: str) -> dict:
    prim = PRIMARY[ds]
    if variant == "ml0_default":
        return {"front": prim, "tracker": "default", "mode": "none", "arm": prim, "sentinel_arm": prim, "cache": None}
    if variant == "stock_default":
        cache = None
        if ds == SHT:  # equal-runtime control: stock YOLOv8n-pose NCNN FP16 @640 (E04 cache)
            from edgelab.analysis import q2_p1_e04 as e04

            cache = e04.CACHE_JOBS["stock_ncnn640"][0]
        return {"front": "stock", "tracker": "default", "mode": "none", "arm": "stock", "sentinel_arm": "stock", "cache": cache}
    if variant == "ml0_retrack":
        return {"front": prim, "tracker": "locked", "mode": "retrack", "arm": prim, "sentinel_arm": prim, "cache": None}
    raise KeyError(variant)


def _tclip5(ci: int) -> dict:
    """Run tables of one clip (E03 `_table_clip` with a configurable tracker and an optional non-oracle C1 re-track mode, level 3 only)."""
    G = _G
    ds = G["ds"]
    L, rmax = e03.SEG_LENS[ds], e03.RMAX[ds]
    frames, ids = G["frames"][ci], G["ids"][ci]
    tracker, mode, thr = G["tracker"], G["mode"], G["thr"]
    roi, ext_stat = (G["roi"][ci], G["ext"][ci]) if mode == "retrack" else (None, None)
    n = len(frames)
    pool: list[np.ndarray] = []
    index: dict[int, int] = {}
    rows_c, rows_e, rows_uid = [], [], []
    starts = list(range(0, n - L + 1, e03.START_GRID))
    bits = np.zeros((len(starts), len(e03.LEVELS)), dtype=np.uint64)
    n_reused = n_run = 0
    ext = e03.EXT[ds]

    def intern(arr: np.ndarray) -> int:
        key = e03._hash64(arr.tobytes())
        got = index.get(key)
        if got is None:
            got = index[key] = len(pool)
            pool.append(arr)
        return got

    for si, c in enumerate(starts):
        end = min(n, c + rmax)
        nrun = end - c
        fr = frames[c:end]
        if mode == "retrack":
            re_mask, src = e03.reuse_pattern(roi, ext_stat, c, nrun, thr["roi"][2], thr["ext"][2])
            bits[si, 2] = np.uint64(e03.pack_bits(re_mask))
            fr = retrack_frames(fr, re_mask, src)
            n_reused += int(re_mask.sum())
            n_run += nrun
        tracks = e01.replay_cell(fr, ids[c:end], tracker)
        for pid, (f, kp) in tracks.items():
            ws = e00.complete_window_starts(f, L)
            if len(ws) == 0:
                continue
            last = f[ws + L - 1]
            e = last - (c + L - 1)
            for st, ee in zip(ws[e <= ext], e[e <= ext]):
                arr = np.array(kp[st:st + L], dtype=np.float64)
                u = intern(arr)
                rows_c.append(si)
                rows_e.append(int(ee))
                rows_uid.append([u] * (1 + len(e03.LEVELS)))
    pool_arr = np.stack(pool) if pool else np.zeros((0, L, 17, 3), dtype=np.float32)
    order = np.lexsort((np.array(rows_e, dtype=np.int64), np.array(rows_c, dtype=np.int64))) if rows_c else np.zeros(0, dtype=np.int64)
    return {"pool": pool_arr, "starts": np.array(starts, dtype=np.int32), "row_c": np.array(rows_c, dtype=np.int32)[order], "row_e": np.array(rows_e, dtype=np.int8)[order],
            "row_uid": (np.array(rows_uid, dtype=np.int32).reshape(-1, 1 + len(e03.LEVELS)) if rows_c else np.zeros((0, 1 + len(e03.LEVELS)), np.int32))[order],
            "bits": bits, "n_reused_frames": n_reused, "n_run_frames": n_run}


def stage_tables(ds: str, ctx, variant: str, procs: int, log=print, force: bool = False) -> dict:
    path = tb_path(variant, ds)
    if path.exists() and not force:
        log(f"[{now()}] tables {variant} {ds}: cached")
        return load_tb(variant, ds)
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    spec = variant_spec(ds, variant)
    names, labels = ctx.names[ds], ctx.labels[ds]
    arm = e01.arm_by_key(ctx.arms)[f"{ds}/{spec['arm']}"]
    cache_dir = spec["cache"] or arm.cache_dir
    tracker = e02.TRACKERS[spec["tracker"]]
    scorer = ctx.scorers[ds]
    st = e03.stage_static(ds, ctx, procs, log) if spec["mode"] == "retrack" else {"roi": None, "ext": None, "thr": None}
    frames_by_clip, ids_by_clip = e01.load_frames_floor(cache_dir, names, e03.GATE_FLOOR)
    for ids, lab in zip(ids_by_clip, labels):
        assert np.array_equal(ids, np.arange(len(lab))), "detector cache frame ids must be 0..n-1"
    n_frames = np.array([len(x) for x in labels])
    offs = np.concatenate([[0], np.cumsum(n_frames)])
    n_in = np.concatenate([[int((f["conf"] >= tracker.gate).sum()) for f in fr] for fr in frames_by_clip]).astype(np.int16)
    roi = [st["roi"][offs[i]:offs[i + 1]] for i in range(len(names))] if st["roi"] is not None else None
    ext = [st["ext"][offs[i]:offs[i + 1]] for i in range(len(names))] if st["ext"] is not None else None
    _G.update(ds=ds, frames=frames_by_clip, ids=ids_by_clip, roi=roi, ext=ext, thr=st["thr"], tracker=tracker, mode=spec["mode"])
    t0 = time.perf_counter()
    cols: dict[str, list] = {k: [] for k in ("row_norm", "row_e", "row_start", "start_frame", "clip_of_start", "bits")}
    n_pool = n_reused = n_run = start_off = 0
    try:
        with mp.get_context("fork").Pool(procs) as pool:
            for ci, res in enumerate(pool.imap(_tclip5, range(len(names)), chunksize=1)):
                norm = scorer.score(res["pool"].astype(np.float64)) if len(res["pool"]) else np.zeros(0)
                uid = res["row_uid"]
                cols["row_norm"].append(norm[uid].astype(np.float32))
                cols["row_e"].append(res["row_e"])
                cols["row_start"].append(res["row_c"] + start_off)
                cols["start_frame"].append(res["starts"])
                cols["clip_of_start"].append(np.full(len(res["starts"]), ci, dtype=np.int32))
                cols["bits"].append(res["bits"])
                start_off += len(res["starts"])
                n_pool += len(res["pool"])
                n_reused += res["n_reused_frames"]
                n_run += res["n_run_frames"]
                if (ci + 1) % 20 == 0 or ci + 1 == len(names):
                    log(f"[{now()}]   tables {variant} {ds}: {ci + 1}/{len(names)} clips, {n_pool} unique windows, {time.perf_counter() - t0:.0f}s")
    finally:
        _G.clear()
    out = {k: np.concatenate(v) for k, v in cols.items()}
    out["row_norm"] = out["row_norm"].reshape(-1, 1 + len(e03.LEVELS))
    counts = np.zeros((start_off, e03.EXT[ds] + 1), dtype=np.int64)
    np.add.at(counts, (out["row_start"], out["row_e"].astype(np.int64)), 1)
    out["row_ptr"] = np.concatenate([[0], np.cumsum(counts.reshape(-1))])
    nrow = len(out["row_norm"])
    out["row_lin"] = np.zeros(nrow, dtype=np.int32)  # C2 columns are unused by Fixed-Burst / C1 policies
    out["row_step"] = np.full(nrow, np.nan, dtype=np.float32)
    out["row_qual"] = np.zeros(nrow, dtype=np.float32)
    out["n_in"] = n_in
    out["n_pool"] = np.array(n_pool)
    out["meta"] = np.array(json.dumps({"variant": variant, "ds": ds, "L": e03.SEG_LENS[ds], "tracker": tracker.label, "frontend": spec["front"], "mode": spec["mode"],
                                       "cache_dir": str(cache_dir), "thr": st["thr"], "rows": nrow, "reused_frame_fraction": (n_reused / n_run) if n_run else 0.0,
                                       "build_s": time.perf_counter() - t0}))
    np.savez_compressed(path, **out)
    log(f"[{now()}] tables {variant} {ds}: {nrow} rows, {n_pool} unique windows, {start_off} starts in {time.perf_counter() - t0:.0f}s")
    return load_tb(variant, ds)


def load_tb(variant: str, ds: str) -> dict:
    z = np.load(tb_path(variant, ds))
    out = {k: z[k] for k in z.files}
    out["meta"] = json.loads(str(out["meta"]))
    return out


def check_tables(ds: str, ctx, procs: int, log=print) -> dict:
    """E5-D3 (builder equivalence on three clips) and E5-D4 (re-track source causality on every start of the clips)."""
    names, labels = ctx.names[ds], ctx.labels[ds]
    arm = e01.arm_by_key(ctx.arms)[f"{ds}/{PRIMARY[ds]}"]
    picks = [0, len(names) // 2, len(names) - 1]
    sub = [names[i] for i in picks]
    frames_by_clip, ids_by_clip = e01.load_frames_floor(arm.cache_dir, sub, e03.GATE_FLOOR)
    st = e03.stage_static(ds, ctx, procs, log)
    offs = np.concatenate([[0], np.cumsum([len(x) for x in labels])])
    roi = [st["roi"][offs[i]:offs[i + 1]] for i in picks]
    ext = [st["ext"][offs[i]:offs[i + 1]] for i in picks]
    # E03 builder
    e03._G.update(ds=ds, frames=frames_by_clip, ids=ids_by_clip, roi=roi, ext=ext, thr=st["thr"])
    try:
        ref = [e03._table_clip(k) for k in range(len(picks))]
    finally:
        e03._G.clear()
    _G.update(ds=ds, frames=frames_by_clip, ids=ids_by_clip, roi=roi, ext=ext, thr=st["thr"], tracker=TRACKER_LOCKED, mode="none")
    try:
        got = [_tclip5(k) for k in range(len(picks))]
        _G["mode"] = "retrack"
        rt = [_tclip5(k) for k in range(len(picks))]
    finally:
        _G.clear()
    eq = []
    for a, b in zip(ref, got):
        # E03's pool also holds the C1-substituted variants of a window, so compare the genuine window of every row rather than the pool arrays
        eq.append(bool(np.array_equal(a["row_c"], b["row_c"]) and np.array_equal(a["row_e"], b["row_e"])
                       and np.array_equal(a["pool"][a["row_uid"][:, 0]], b["pool"][b["row_uid"][:, 0]])))
    # D4 on every start of every clip (SHT) / 40 clips (UB)
    thr = st["thr"]
    n_ok = n_all = 0
    bad = 0
    ci_list = range(len(names)) if ds == SHT else range(0, len(names), max(1, len(names) // 40))
    L, rmax = e03.SEG_LENS[ds], e03.RMAX[ds]
    for i in ci_list:
        n = len(labels[i])
        r_i, x_i = st["roi"][offs[i]:offs[i + 1]], st["ext"][offs[i]:offs[i + 1]]
        for c in range(0, n - L + 1, e03.START_GRID):
            nrun = min(n, c + rmax) - c
            re_mask, src = e03.reuse_pattern(r_i, x_i, c, nrun, thr["roi"][2], thr["ext"][2])
            n_all += 1
            if retrack_source_ok(re_mask, src):
                n_ok += 1
            else:
                bad += 1
    return {"D3_builder_equal_e03": {"clips": sub, "equal": eq, "pass": bool(all(eq))},
            "D3_retrack_rows": [int(len(r["row_c"])) for r in rt], "D3_e03_rows": [int(len(r["row_c"])) for r in ref],
            "D4_pattern_sources": {"starts_checked": n_all, "ok": n_ok, "violations": bad, "pass": bad == 0}}


# =====================================================================================================================
# 5. Worlds, matched-cost runner
# =====================================================================================================================
def make_worlds(ctx, ds: str, cfg03: dict, which=("ml0_locked", "ml0_default", "stock_default", "ml0_retrack")) -> dict:
    K = copy.deepcopy(cfg03["datasets"][ds]["constants"])
    arms = e01.arm_by_key(ctx.arms)
    out: dict = {}
    for v in which:
        if v == "ml0_locked":
            w = e03.World(ctx, ds)
            w.K = copy.deepcopy(K)
        else:
            sp = variant_spec(ds, v)
            pools = e02.build_pools(ds, sp["front"], sp["tracker"])
            w = e03.World.from_parts(ds, ctx.names[ds], ctx.labels[ds], ctx.dev_idx[ds], ctx.conf_idx[ds], e01.arm_sentinel(ctx.proto, arms[f"{ds}/{sp['sentinel_arm']}"]),
                                     True, load_tb(v, ds), pools)
            w.K = copy.deepcopy(K)
            if sp["mode"] == "retrack":
                w.K["c1_retrack"] = True
            else:
                w.rn = [w.rn[0]] * (1 + len(e03.LEVELS))  # no reuse columns: one list serves every level
        out[v] = w
    return out


def run_cell5(w: e03.World, spec: dict, cell: tuple, target: float, w_ref: float | None, clips: list[int], knob: float | None = None, trace: dict | None = None) -> dict:
    """Calibrate the period knob of `spec` to the absolute CPU `target` (labels never read) and simulate once more with full recording."""
    s, beta = cell
    c = w.cost(s)
    mk = lambda x: make_policy5(w, c, spec, x)  # noqa: E731
    if spec["k1"]:
        cal = {"knob": 1.0, "cost": None, "status": "given", "evals": 0}
    elif knob is not None:
        cal = {"knob": knob, "cost": None, "status": "given", "evals": 0}
    else:
        lo, hi = e03.KNOB_RANGE["period"]
        cal = e03.calibrate(lambda x: e03.total_cost(w, e03.simulate(w, c, mk(x), clips, light=True)), lo, hi, target, increasing=False)
    w_eff = cal["knob"] if w_ref is None else w_ref
    out = {"cfg": {"arm": spec.get("name", "?"), **{k: spec[k] for k in ("world", "lvl", "guard", "k1")}}, "s": s, "beta": beta, "target": target, "knob": cal["knob"],
           "status": cal["status"], "evals": cal["evals"], "w_ref": w_eff, "hold": max(0, int(round(w_eff)) - 1)}
    R = e03.simulate(w, c, mk(cal["knob"]), clips, light=False, trace=trace)
    out["cost"] = e03.total_cost(w, R)
    out["ratio"] = out["cost"] / target
    out["R"] = R
    return out


def _job(job: tuple) -> tuple:
    arm, cell, w_ref, target = job
    spec = dict(ARMS[arm], name=arm)
    w = _G["worlds"][spec["world"]]
    r = run_cell5(w, spec, cell, target, w_ref, _G["allc"])
    r["cfg"]["arm"] = arm
    return arm, cell_id(*cell), e03.compact_result(w, r)


def theta_of_family(w: e03.World, R_ideal_series: list[np.ndarray]) -> float:
    devc = list(w.dev_idx)
    normal = np.concatenate([R_ideal_series[i][w.labels[i] == 0] for i in devc])
    return float(np.quantile(normal, 1.0 - e03.ALERT_FPR))


def stage_run(ctx, ds: str, procs: int, log=print, force: bool = False) -> dict:
    path = run_path(ds)
    stamp = sha(EXP_DIR / "config.yaml")
    if path.exists() and not force:
        got = pickle.loads(path.read_bytes())
        if got["stamp"] == stamp:
            log(f"[{now()}] run {ds}: cached")
            return got
    cfg, cfg03 = load_config(), e03.load_config()
    worlds = make_worlds(ctx, ds, cfg03)
    allc = list(range(len(ctx.names[ds])))
    cells = [tuple(c_) for c_ in cfg["cells"]]
    wl = worlds["ml0_locked"]
    target = {cell_id(*c_): c_[1] * e03.c_full(wl, c_[0], allc, "all") for c_ in cells}
    t0 = time.perf_counter()
    _G.update(worlds=worlds, allc=allc)
    results: dict = {}
    try:
        with mp.get_context("fork").Pool(procs) as pool:
            own = [a for a, sp in ARMS.items() if sp["ref"] == "own"]
            res1 = pool.map(_job, [(a, c_, None, target[cell_id(*c_)]) for a in own for c_ in cells], chunksize=1)
            for arm, cid, r in res1:
                results[(arm, cid)] = r
            w_ref = {cell_id(*c_): results[("Fixed", cell_id(*c_))]["knob"] for c_ in cells}
            log(f"[{now()}] run {ds}: Fixed periods " + ", ".join(f"{cell_id(*c_)}:{w_ref[cell_id(*c_)]:.1f}" for c_ in cells))
            dep = [a for a, sp in ARMS.items() if sp["ref"] == "fixed"]
            res2 = pool.map(_job, [(a, c_, w_ref[cell_id(*c_)], target[cell_id(*c_)]) for a in dep for c_ in cells], chunksize=1)
            for arm, cid, r in res2:
                results[(arm, cid)] = r
    finally:
        _G.clear()
    e03run = pickle.loads(e03.run_path(ds).read_bytes())
    for arm in E03_REF_ARMS:
        for c_ in CELLS_E03:
            results[(arm, cell_id(*c_))] = e03run["results"][(arm, cell_id(*c_))]
    log(f"[{now()}] run {ds}: {len(results)} (arm, cell) results in {time.perf_counter() - t0:.0f}s")
    # alert thresholds per family (E03 constant for the locked family; test_dev K=1 ideal series otherwise)
    theta = {"ml0_locked": float(cfg["constants"][ds]["theta"])}
    cut = np.cumsum(wl.n_frames)[:-1]
    for fam, arm in (("ml0_default", "ML0-Default-K1"), ("stock_default", "Stock-Default-K1")):
        ideal = np.split(results[(arm, cell_id(*CELLS[0]))]["ideal"].astype(np.float64), cut)
        theta[fam] = theta_of_family(worlds[ARMS[arm]["world"]], ideal)
    ideal = np.split(results[("Full-Rate", cell_id(*CELLS[0]))]["ideal"].astype(np.float64), cut)
    theta["ml0_locked_rederived"] = theta_of_family(wl, ideal)
    checks = run_checks(ctx, ds, worlds, results, target, w_ref, cells, e03run, theta, log)
    traces = {}
    mid = (2.0, 0.50)
    pick = [wl.conf_idx[0], wl.conf_idx[len(wl.conf_idx) // 2], wl.dev_idx[0]]
    for arm in ("Fixed", "Cache-C1", "Cache-C1+Guard", "C1-ReTrack"):
        sp = dict(ARMS[arm], name=arm)
        w = worlds[sp["world"]]
        r0 = results[(arm, cell_id(*mid))]
        tr = {0: [], 1: [], 2: []}
        run_cell5(w, sp, mid, target[cell_id(*mid)], r0["w_ref"], pick, knob=r0["knob"], trace=tr)
        traces[arm] = {"clips": [wl.names[i] for i in pick], "cell": cell_id(*mid), "frames": {k: v[:400] for k, v in tr.items()},
                       "columns": ["t", "observe", "c1_level", "c1_reused", "forced_detect", "windows_fresh", "windows_reused", "queue_wait_ms", "tokens"]}
    out = {"stamp": stamp, "results": results, "w_ref": w_ref, "targets": target, "theta": theta, "checks": checks, "traces": traces,
           "n_frames": [int(x) for x in wl.n_frames], "build_s": time.perf_counter() - t0, "table_meta": {v: worlds[v].tb["meta"] for v in worlds if v != "ml0_locked"}}
    path.write_bytes(pickle.dumps(out))
    return out


def run_checks(ctx, ds, worlds, results, target, w_ref, cells, e03run, theta, log) -> dict:
    wl = worlds["ml0_locked"]
    cut = np.cumsum(wl.n_frames)[:-1]
    out: dict = {}
    # E5-D1: bit-for-bit replay of E03 (Fixed, Cache-C1) on the 12 E03 cells
    d1 = {}
    for arm in ("Fixed", "Cache-C1"):
        for c_ in CELLS_E03:
            cid = cell_id(*c_)
            a, b = results[(arm, cid)], e03run["results"][(arm, cid)]
            d1[f"{arm}|{cid}"] = bool(np.array_equal(a["series"], b["series"]) and np.array_equal(a["ideal"], b["ideal"]) and a["knob"] == b["knob"] and a["cost"] == b["cost"]
                                      and a["target"] == b["target"])
    out["E5-D1"] = {"pass": all(d1.values()), "n": len(d1), "failed": [k for k, v in d1.items() if not v]}
    # E01 regression anchor: K=1 ideal series of the ML0 arms reproduce E01's S4 all-clip AUC
    anchors = {}
    for arm, key in (("Full-Rate", "locked"), ("ML0-Default-K1", "default")):
        ideal = np.split(results[(arm, cell_id(*CELLS[0]))]["ideal"].astype(np.float64), cut)
        anchors[key] = {"e05": 100 * e00.pooled_auc(wl.labels, ideal, None), "e01": E01_REF_ML0[ds][key]}
    out["E5-D1b"] = {"pass": all(abs(v["e05"] - v["e01"]) <= D1B_TOL_PT for v in anchors.values()), "tolerance_pt": D1B_TOL_PT, "anchors": anchors,
                     "theta_rederived_vs_e03": [theta["ml0_locked_rederived"], theta["ml0_locked"]]}
    # E5-D2: guard identity / boundedness at the knob of Cache-C1
    d2 = {}
    spec_c1, spec_g = dict(ARMS["Cache-C1"], name="Cache-C1"), dict(ARMS["Cache-C1+Guard"], name="Cache-C1+Guard")
    allc = list(range(len(wl.names)))
    for c_ in CELLS:
        cid = cell_id(*c_)
        knob = results[("Cache-C1", cid)]["knob"]
        c = wl.cost(c_[0])
        Ru = e03.simulate(wl, c, make_policy5(wl, c, spec_c1, knob), allc)
        pg = make_policy5(wl, c, spec_g, knob)
        Rg = e03.simulate(wl, c, pg, allc)
        hold = results[("Cache-C1", cid)]["hold"]
        su, _, _ = e03.delivery(wl, Ru, hold)
        sg, _, _ = e03.delivery(wl, Rg, hold)
        never = float(Ru["wait"].max()) <= SHED_ON_MS
        if never:
            ok = bool(all(np.array_equal(x, y) for x, y in zip(su, sg)) and pg.info()["shed_frames"] == 0)
        else:
            ok = bool(Rg["wait"].max() <= Ru["wait"].max() + 1e-6)
        d2[cid] = {"never_exceeds_on": never, "ok": ok, "unguarded_max_wait_ms": float(Ru["wait"].max()), "guarded_max_wait_ms": float(Rg["wait"].max()), **pg.info()}
    out["E5-D2"] = {"pass": all(v["ok"] for v in d2.values()), "cells": d2}
    # E5-D5: queue invariants for the guarded arms at the mid and overload cells
    d5 = {}
    for arm in ("Fixed+Guard", "Cache-C1+Guard", "C1-ReTrack+Guard"):
        sp = dict(ARMS[arm], name=arm)
        w = worlds[sp["world"]]
        for c_ in ((2.0, 0.50), OVERLOAD_CELL):
            cid = cell_id(*c_)
            r = run_cell5(w, sp, c_, target[cid], results[(arm, cid)]["w_ref"], allc, knob=results[(arm, cid)]["knob"])
            R = r["R"]
            ser, st, info = e03.delivery(w, R, r["hold"])
            soff = np.concatenate([[0], np.cumsum(w.n_frames[allc])])
            ck = e02.check_queue_invariants(np.arange(len(R["fin"])) * w.dt, R["svc"], R["fin"], info["landing"], R["w_t"], w.dt, R["w_pos"], soff)
            d5[f"{arm}|{cid}"] = {"ok": bool(all(ck.values()) and (R["wait"] >= 0).all() and np.isfinite(R["cpu"]).all()), **{k: bool(v) for k, v in ck.items()}}
    out["E5-D5"] = {"pass": all(v["ok"] for v in d5.values()), "cells": d5}
    # E5-D6: cost matching
    d6 = {}
    for (arm, cid), r in results.items():
        if arm in E03_REF_ARMS or ARMS[arm]["k1"]:
            continue
        d6[f"{arm}|{cid}"] = {"ratio": r["ratio"], "status": r["status"]}
    bad = [k for k, v in d6.items() if v["ratio"] > 1.005]
    out["E5-D6"] = {"pass": not bad, "violations": bad, "status_counts": {st: sum(1 for v in d6.values() if v["status"] == st) for st in ("ok", "under", "over")},
                    "flagged": sorted(k for k, v in d6.items() if v["status"] != "ok")}
    return out


# =====================================================================================================================
# 6. Lock
# =====================================================================================================================
def load_config() -> dict:
    return yaml.safe_load((EXP_DIR / "config.yaml").read_text())


def stage_lock(log=print) -> dict:
    cfg03 = e03.load_config()
    prior = {"E00": e00.EXP_DIR, "E01": e01.EXP_DIR, "E02": e02.EXP_DIR, "E03": e03.EXP_DIR, "E04": e00.EXPERIMENTS_ROOT / "Q2-P1-E04_optimizations"}
    plan = EXP_DIR / "plan.md"
    assert plan.exists(), "write plan.md first"
    rules = {"V1": "SHT confirmation Cache-C1 - Fixed, 12-cell mean: CI lower > 0 -> confirmed", "V2": "UB replication of V1 (lower > 0 replicated; non-negative otherwise)",
             "V3": "Cache-C1 - Stock-Baseline, confirmation, 12-cell mean, both datasets: CI lower > 0", "V4": "ReTrack - Fixed lower > 0 and >= 50% of the oracle gain",
             "V5": "guard: non-overload cells non-inferior (lower > -0.25 pt); overload cells lower > 0",
             "V6": "short events, budget cells: recall@1s lower > -5 pt vs Fixed", "V7": "new entries: recall@1s lower > -5 pt and delay p50 upper < +250 ms vs Fixed",
             "V8": "C2 bounded: max wait, p99 <= 2000 ms, drift <= 100 ms, queue memory <= 200 MB", "V9": "VM: >= 10,000 frames, RSS growth <= 5%, p99 Pi-equivalent wait <= 2000 ms"}
    cfg = {"experiment": "Q2-P1-E05", "title": "Confirmation", "seed": SEED, "pool_seed": POOL_SEED, "plan_sha256": sha(plan), "bootstrap": {"resamples": N_BOOT, "seed": 0},
           "builds_on": {n: {"summary_sha256": sha(d / "summary.json"), "config_sha256": sha(d / "config.yaml")} for n, d in prior.items()},
           "locked_system": {"frontend": PRIMARY, "frontend_control": {SHT: "stock NCNN FP16 @640 (E04 cache)", UB: "stock MPS @640"}, "tracker": TRACKER_LOCKED.label,
                             "tracker_control": TRACKER_DEFAULT.label, "scorer": "frozen STG-NF (E00 strict causal S3/S4, completion landing)",
                             "primary": "Cache-C1", "overload_safe": "Cache-C1+Guard", "adopted_from_E04": {"MV": False, "INT8": False},
                             "guard": {"shed_on_ms": SHED_ON_MS, "shed_off_ms": SHED_OFF_MS, "thinning_ported": False}, "c1": {"level": 3, "A1": e03.A1}},
           "cells": [list(c_) for c_ in CELLS], "cells_primary12": [list(c_) for c_ in CELLS_E03], "cells_native": [list(c_) for c_ in CELLS_NATIVE],
           "cell_groups": {k: [list(c_) for c_ in v] for k, v in CELL_GROUPS.items()},
           "arms": {a: {k: v for k, v in sp.items()} for a, sp in ARMS.items()}, "e03_reference_arms": list(E03_REF_ARMS), "contrasts": {k: list(v) for k, v in CONTRASTS.items()},
           "constants": {ds: cfg03["datasets"][ds]["constants"] for ds in DATASETS},
           "subgroups": {"short_event_s": SHORT_EVENT_S, "short_frames": {ds: short_frames(ds) for ds in DATASETS}, "entry_window_s": ENTRY_WINDOW_S, "crowd_min_tracks": CROWD_MIN_TRACKS,
                         "motion_threshold": "E03 test_dev motion_q.q85 (p90 of the first second of the event)", "loc_pad_s": LOC_PAD_S, "arms": list(SUB_ARMS),
                         "metrics": list(SUB_METRICS), "noninf_recall_pt": NONINF_RECALL_PT, "noninf_delay_ms": NONINF_DELAY_MS},
           "sustained": {"arms": list(SUSTAINED_ARMS), "s": list(SUSTAINED_S), "segment_minutes": SEG_MINUTES, "sht_loops": 5, "ub_loops": 1, "cap_ms": CAP_MS,
                         "bounded_rule": {"max_wait_ms": CAP_MS, "p99_ms": CAP_MS, "drift_ms": 100.0, "queue_mem_mb": 200.0}},
           "vm": {"runs": [{"mode": m, "s": s} for m, s in VM_RUNS], "passes": VM_PASSES, "frames": VM_PASSES * 2880, "rss_marks": list(VM_RSS_MARKS), "threads": 4,
                  "x264": "libx264 veryfast crf 23 yuv420p g 48", "host_load_limit_pct": 30},
           "verdict_rules": rules, "frozen_numbers_e03": {"theta": {ds: cfg03["datasets"][ds]["constants"]["theta"] for ds in DATASETS}}}
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    (EXP_DIR / "config.yaml").write_text(yaml.safe_dump(plain(cfg), sort_keys=False))
    log(f"[{now()}] lock: wrote {EXP_DIR / 'config.yaml'}")
    return cfg


# =====================================================================================================================
# 7. Strata of the ground-truth events
# =====================================================================================================================
def _replay_k1(ci: int):
    return e01.replay_cell(_G["frames"][ci], _G["ids"][ci], TRACKER_LOCKED)


def stage_strata(ctx, ds: str, procs: int, log=print, force: bool = False) -> dict:
    path = strata_path(ds)
    if path.exists() and not force:
        log(f"[{now()}] strata {ds}: cached")
        return pickle.loads(path.read_bytes())
    names, labels = ctx.names[ds], ctx.labels[ds]
    arms = e01.arm_by_key(ctx.arms)
    arm = arms[f"{ds}/{PRIMARY[ds]}"]
    frames, ids = e01.load_frames_floor(arm.cache_dir, names, e03.GATE_FLOOR)
    _G.update(frames=frames, ids=ids)
    try:
        tracks = e01.pmap(_replay_k1, list(range(len(names))), procs)
    finally:
        _G.clear()
    cfg03 = e03.load_config()
    motion_thr = float(cfg03["datasets"][ds]["constants"]["motion_q"]["q85"])
    motion = e03.load_motion(ds, names)["motion"]
    rel_tracks = e01.read_tracks(arms[f"{ds}/released"].pose_dir, ds, names) if ds == SHT else None
    ev = {k: [] for k in ("clip", "s", "e", "short", "entry", "fast", "crowd", "entry_released", "n_active_med", "motion_p90")}
    for i, lab in enumerate(labels):
        events = e03.label_events(lab)
        if not events:
            continue
        starts, count = track_summary(tracks[i], len(lab))
        rstarts = track_summary(rel_tracks[i], len(lab))[0] if rel_tracks is not None else None
        for s, e in events:
            fl = classify_event(ds, s, e, starts, count, motion[i], motion_thr)
            sl = first_second(s, e, FPS[ds])
            m = motion[i][sl]
            m = m[np.isfinite(m)]
            ev["clip"].append(i), ev["s"].append(s), ev["e"].append(e)
            for k in ("short", "entry", "fast", "crowd"):
                ev[k].append(fl[k])
            ev["entry_released"].append(bool(len(rstarts) and np.any((rstarts >= s - int(round(FPS[ds]))) & (rstarts <= s + int(round(FPS[ds])))))
                                        if rstarts is not None else False)
            ev["n_active_med"].append(float(np.median(count[sl])))
            ev["motion_p90"].append(float(np.quantile(m, 0.9)) if len(m) else float("nan"))
    out = {k: np.array(v) for k, v in ev.items()}
    out["n_events"] = int(len(out["clip"]))
    out["motion_thr"] = motion_thr
    out["has_released"] = rel_tracks is not None
    out["k1_tracks_per_clip"] = [len(t) for t in tracks]
    n_ev = out["n_events"]
    exp = {SHT: 194, UB: 199}[ds]
    out["checks"] = {"E5-D9": {"pass": bool(n_ev == exp and out["short"].dtype == bool), "events": n_ev, "expected": exp, "short_frames": short_frames(ds),
                               "short_frames_expected": {SHT: 60, UB: 75}[ds]}}
    out["counts"] = {k: int(out[k].sum()) for k in ("short", "entry", "fast", "crowd")}
    if out["has_released"]:
        out["entry_agreement_released_pct"] = float(100 * np.mean(out["entry"] == out["entry_released"]))
    path.write_bytes(pickle.dumps(out))
    log(f"[{now()}] strata {ds}: {n_ev} events, {out['counts']}")
    return out


def strata_levels(st: dict) -> list[tuple[str, str, np.ndarray]]:
    n = st["n_events"]
    return [("all", "all", np.ones(n, dtype=bool)),
            ("duration", "short_event", st["short"]), ("duration", "sustained_event", ~st["short"]),
            ("entry", "new_entry_near_onset", st["entry"]), ("entry", "existing_tracks_only", ~st["entry"]),
            ("motion", "sudden_fast_motion", st["fast"]), ("motion", "moderate_motion", ~st["fast"]),
            ("crowd", "high_crowd", st["crowd"]), ("crowd", "low_crowd", ~st["crowd"])]


# =====================================================================================================================
# 8. Analysis (Part A + Part B)
# =====================================================================================================================
_A: dict = {}


class Analysis5:
    def __init__(self, ds: str, w: e03.World, run: dict, cfg: dict):
        self.ds, self.w, self.run, self.cfg = ds, w, run, cfg
        self.results = run["results"]
        self.fps = FPS[ds]
        self.cut = np.cumsum(w.n_frames)[:-1]
        self.theta = run["theta"]
        self.splits = {"dev": list(w.dev_idx), "confirmation": list(w.conf_idx), "all": list(range(len(w.names)))}
        self.boots = {k: e03.AucBoot(w.labels, None if k == "all" else self.splits[k], N_BOOT, 0) for k in ("confirmation", "all")}
        self._ser: dict = {}

    def keys(self):
        return sorted(self.results)

    def has(self, arm: str, cid: str) -> bool:
        return (arm, cid) in self.results

    def theta_of(self, arm: str) -> float:
        return self.theta[family_of(arm)]

    def series(self, arm: str, cid: str, kind: str = "series") -> list[np.ndarray]:
        key = (arm, cid, kind)
        if key not in self._ser:
            self._ser[key] = np.split(self.results[(arm, cid)][kind].astype(np.float64), self.cut)
        return self._ser[key]


def _boot_job(key: tuple):
    arm, cid, sub = key
    A = _A["A"]
    ser = A.series(arm, cid)
    b = A.boots[sub]
    return key, b.aucs([ser[i] for i in b.idx])


def contrast_sets(boots: dict, sub: str, a: str, b: str, cids: list[str]) -> dict | None:
    use = [c for c in cids if (a, c, sub) in boots and (b, c, sub) in boots]
    if not use:
        return None
    pa = np.mean([boots[(a, c, sub)][0] for c in use])
    pb = np.mean([boots[(b, c, sub)][0] for c in use])
    ba = np.mean([boots[(a, c, sub)][1] for c in use], axis=0)
    bb = np.mean([boots[(b, c, sub)][1] for c in use], axis=0)
    d = diff_ci(pa, ba, pb, bb)
    d.update({"n_cells": len(use), "excludes_zero": bool(d["lo"] > 0 or d["hi"] < 0), "mean_a": float(pa), "mean_b": float(pb)})
    return d


def level_metrics(eb: EventBoot, clip: np.ndarray, delays: np.ndarray, fps: float) -> dict:
    """Event metrics of one event set. recall @ tau (%), missed (%), detection delay p50 / p90 (ms, detected events). -> {metric: (point, boot)}."""
    out = {}
    for tau in (0.5, 1.0, 2.0):
        out[f"recall_{tau}"] = eb.rate(clip, delays <= tau * fps)
    det = np.isfinite(delays)
    out["missed"] = eb.rate(clip, ~det)
    ms = delays[det] * 1000.0 / fps
    out["delay_p50_ms"] = eb.quantile(clip[det], ms, 0.5)
    out["delay_p90_ms"] = eb.quantile(clip[det], ms, 0.9)
    return out


def avg_metrics(blocks: list[dict]) -> dict:
    out = {}
    for k in blocks[0]:
        pts = np.array([b[k][0] for b in blocks], dtype=np.float64)
        bts = np.stack([b[k][1] for b in blocks])
        out[k] = (float(np.nanmean(pts)) if np.isfinite(pts).any() else float("nan"), np.nanmean(bts, axis=0) if np.isfinite(bts).any() else np.full(bts.shape[1], np.nan))
    return out


def summarize_metrics(m: dict) -> dict:
    out = {}
    for k, (pt, bt) in m.items():
        lo, hi = ci95(bt)
        out[k] = {"point": float(pt), "lo": lo, "hi": hi}
    return out


def _loc_job(job: tuple):
    li, = job
    S = _A["S"]
    A, level = _A["A"], S["levels"][li]
    sel = level[2]
    fps = A.fps
    labels = A.w.labels
    masks = []
    by_clip: dict[int, list] = {}
    for k in np.nonzero(sel)[0]:
        by_clip.setdefault(int(S["st"]["clip"][k]), []).append((int(S["st"]["s"][k]), int(S["st"]["e"][k])))
    for i in range(len(labels)):
        masks.append(window_mask(len(labels[i]), by_clip.get(i, []), fps) if i in by_clip else np.zeros(len(labels[i]), dtype=bool))
    lab_m = [labels[i][masks[i]] for i in range(len(labels))]
    ab = e03.AucBoot(lab_m, None, N_BOOT, 0)
    out = {}
    for (arm, cid) in S["pairs"]:
        ser = A.series(arm, cid)
        out[(arm, cid)] = ab.aucs([ser[i][masks[i]] for i in range(len(labels))])
    return li, out


def analyze_dataset(ctx, ds: str, procs: int, log=print) -> dict:
    cfg, cfg03 = load_config(), e03.load_config()
    run = pickle.loads(run_path(ds).read_bytes())
    st = pickle.loads(strata_path(ds).read_bytes())
    w = make_worlds(ctx, ds, cfg03, which=("ml0_locked",))["ml0_locked"]
    A = Analysis5(ds, w, run, cfg)
    t0 = time.perf_counter()
    cells = [tuple(c_) for c_ in cfg["cells"]]
    cids = cells_ids(cells)
    arms_all = list(ARMS) + [a for a in E03_REF_ARMS]
    # ---- AUC points (dev / confirmation / all) and bootstraps (confirmation / all) -------------------------------------------------------------
    auc: dict = {}
    for arm in arms_all:
        for cid in cids:
            if not A.has(arm, cid):
                continue
            ser = A.series(arm, cid)
            ide = A.series(arm, cid, "ideal")
            auc.setdefault(arm, {})[cid] = {sub: 100 * e00.pooled_auc(w.labels, ser, None if sub == "all" else A.splits[sub]) for sub in ("dev", "confirmation", "all")}
            auc[arm][cid]["ideal_all"] = 100 * e00.pooled_auc(w.labels, ide, None)
    jobs = [(arm, cid, sub) for arm in arms_all for cid in cids if A.has(arm, cid) for sub in ("confirmation", "all")]
    _A["A"] = A
    try:
        with mp.get_context("fork").Pool(procs) as pool:
            boots = dict(pool.map(_boot_job, jobs, chunksize=4))
    finally:
        _A.clear()
    log(f"[{now()}] analyze {ds}: {len(boots)} AUC bootstraps ({N_BOOT} resamples) in {time.perf_counter() - t0:.0f}s")
    # ---- overload sets ---------------------------------------------------------------------------------------------------------------------------
    qmax = {cid: run["results"][("Cache-C1", cid)]["queue"]["max_wait_ms"] for cid in cids}
    sets = {name: cells_ids(cs) for name, cs in CELL_SETS.items()}
    sets["nonoverload"] = [cid for cid in cids if qmax[cid] <= SHED_ON_MS]
    sets["overload"] = [cid for cid in cids if qmax[cid] > CAP_MS]
    # ---- arm summaries and contrasts ---------------------------------------------------------------------------------------------------------------
    arm_sum: dict = {}
    for arm in arms_all:
        for sname, sc in sets.items():
            for sub in ("confirmation", "all"):
                use = [c for c in sc if (arm, c, sub) in boots]
                if not use:
                    continue
                pt = float(np.mean([boots[(arm, c, sub)][0] for c in use]))
                bt = np.mean([boots[(arm, c, sub)][1] for c in use], axis=0)
                lo, hi = ci95(bt)
                arm_sum.setdefault(arm, {}).setdefault(sname, {})[sub] = {"mean": pt, "lo": lo, "hi": hi, "n_cells": len(use)}
            use = [c for c in sc if arm in auc and c in auc[arm]]
            if use:
                arm_sum.setdefault(arm, {}).setdefault(sname, {})["dev"] = {"mean": float(np.mean([auc[arm][c]["dev"] for c in use])), "n_cells": len(use)}
    contrasts: dict = {}
    for name, (a, b) in CONTRASTS.items():
        for sub in ("confirmation", "all"):
            for sname, sc in sets.items():
                got = contrast_sets(boots, sub, a, b, sc)
                if got is not None:
                    contrasts.setdefault(name, {}).setdefault(sub, {})[sname] = got
            for cid in cids:
                got = contrast_sets(boots, sub, a, b, [cid])
                if got is not None:
                    contrasts.setdefault(name, {}).setdefault(sub, {})[cid] = got
    # ---- event outcomes ---------------------------------------------------------------------------------------------------------------------------------
    n_ev = st["n_events"]
    clip_ev: dict[int, list] = {}
    for k in range(n_ev):
        clip_ev.setdefault(int(st["clip"][k]), []).append(k)
    delays: dict = {}
    clip_tab: dict = {}
    for (arm, cid) in A.keys():
        ser = A.series(arm, cid)
        th = A.theta_of(arm)
        d = np.full(n_ev, INF)
        for i, ks in clip_ev.items():
            d[ks] = per_event_delays([(int(st["s"][k]), int(st["e"][k])) for k in ks], ser[i], th)
        delays[(arm, cid)] = d
        tab = []
        for i in range(len(w.names)):
            m = e03.event_metrics(w.labels[i], ser[i], th, A.fps)
            dl = m["delays_ms"]
            tab.append((m["events"], m["hit_0.5"], m["hit_1.0"], sum(1 for x in dl if x <= 2000.0), float(np.median(dl)) if dl else float("nan"), m["false_alerts"], m["normal_frames"]))
        clip_tab[(arm, cid)] = np.array(tab, dtype=np.float64)
    log(f"[{now()}] analyze {ds}: event outcomes for {len(delays)} (arm, cell) pairs")
    ev_clip = st["clip"].astype(np.int64)
    # ---- Part A event metrics (confirmation / all clips) ------------------------------------------------------------------------------------------------
    ev_part_a: dict = {}
    for sub in ("confirmation", "all"):
        idx = A.splits[sub]
        pos = {c: i for i, c in enumerate(idx)}
        sel = np.array([c in pos for c in ev_clip])
        eb = EventBoot(len(idx), N_BOOT, 0)
        clip_sub = np.array([pos[c] for c in ev_clip[sel]], dtype=np.int64)
        per_cell: dict = {}
        for arm in arms_all:
            for cid in cids:
                if (arm, cid) in delays:
                    per_cell[(arm, cid)] = level_metrics(eb, clip_sub, delays[(arm, cid)][sel], A.fps)
        for arm in arms_all:
            for sname in ("primary12", "native3"):
                use = [per_cell[(arm, c)] for c in sets[sname] if (arm, c) in per_cell]
                if not use:
                    continue
                ev_part_a.setdefault(sub, {}).setdefault(arm, {})[sname] = summarize_metrics(avg_metrics(use))
                ev_part_a[sub][arm][sname]["n_events"] = int(sel.sum())
            for cid in cids:
                if (arm, cid) in per_cell:
                    ev_part_a.setdefault(sub, {}).setdefault(arm, {})[cid] = summarize_metrics(per_cell[(arm, cid)])
        # false alerts per normal hour (point)
        for arm in arms_all:
            for sname in ("primary12", "native3"):
                fa, hrs = [], []
                for c in sets[sname]:
                    if (arm, c) in clip_tab:
                        t = clip_tab[(arm, c)][idx]
                        fa.append(t[:, 5].sum())
                        hrs.append(t[:, 6].sum() / A.fps / 3600)
                if fa:
                    ev_part_a[sub][arm][sname]["false_alerts_per_normal_hour"] = float(np.mean(fa) / np.mean(hrs))
        # paired event contrasts (12-cell mean), recall@1.0 and delay p50
        for name, (a, b) in CONTRASTS.items():
            for sname in ("primary12",):
                ua = [per_cell[(a, c)] for c in sets[sname] if (a, c) in per_cell and (b, c) in per_cell]
                ub = [per_cell[(b, c)] for c in sets[sname] if (a, c) in per_cell and (b, c) in per_cell]
                if not ua:
                    continue
                ma, mb = avg_metrics(ua), avg_metrics(ub)
                ev_part_a.setdefault(sub, {}).setdefault("_contrasts", {}).setdefault(name, {})[sname] = {
                    k: diff_ci(ma[k][0], ma[k][1], mb[k][0], mb[k][1]) for k in ("recall_0.5", "recall_1.0", "recall_2.0", "missed", "delay_p50_ms")}
    # ---- Part B: strata ---------------------------------------------------------------------------------------------------------------------------------------
    levels = strata_levels(st)
    group_cells = {g: cells_ids(cs) for g, cs in CELL_GROUPS.items()}
    pairs = sorted({(arm, cid) for arm in SUB_ARMS for g in group_cells.values() for cid in g})
    _A.update(A=A, S={"levels": levels, "st": st, "pairs": pairs})
    try:
        with mp.get_context("fork").Pool(min(procs, len(levels))) as pool:
            loc = dict(pool.map(_loc_job, [(li,) for li in range(len(levels))], chunksize=1))
    finally:
        _A.clear()
    log(f"[{now()}] analyze {ds}: localised AUC for {len(levels)} levels x {len(pairs)} series in {time.perf_counter() - t0:.0f}s")
    eb_all = EventBoot(len(w.names), N_BOOT, 0)
    sub_res: dict = {}
    sub_con: dict = {}
    for li, (stratum, level, sel) in enumerate(levels):
        n_sel = int(sel.sum())
        for g, gc in group_cells.items():
            blocks_by_arm: dict = {}
            for arm in SUB_ARMS:
                bl = []
                for cid in gc:
                    m = level_metrics(eb_all, ev_clip[sel], delays[(arm, cid)][sel], A.fps) if n_sel else {k: (float("nan"), np.full(N_BOOT, np.nan)) for k in SUB_METRICS[:-1]}
                    m["loc_auc"] = loc[li][(arm, cid)]
                    bl.append(m)
                blocks_by_arm[arm] = avg_metrics(bl)
                sub_res.setdefault(level, {}).setdefault(g, {})[arm] = {"stratum": stratum, "n_events": n_sel, **summarize_metrics(blocks_by_arm[arm])}
            for cname, (a, b) in SUB_CONTRASTS.items():
                sub_con.setdefault(level, {}).setdefault(g, {})[cname] = {k: diff_ci(blocks_by_arm[a][k][0], blocks_by_arm[a][k][1], blocks_by_arm[b][k][0], blocks_by_arm[b][k][1])
                                                                         for k in SUB_METRICS}
    log(f"[{now()}] analyze {ds}: subgroup tables done ({time.perf_counter() - t0:.0f}s)")
    # ---- verdicts (SHT / UB, both reported) ---------------------------------------------------------------------------------------------------------------------
    cs = contrasts
    v: dict = {}
    c = cs["ladder.cache"]["confirmation"]["primary12"]
    v["V1_confirmation"] = {"contrast": c, "verdict": verdict_lock(c)}
    v["V1_all"] = {"contrast": cs["ladder.cache"]["all"]["primary12"], "verdict": verdict_lock(cs["ladder.cache"]["all"]["primary12"])}
    v["V2_replication"] = {"contrast": c, "verdict": verdict_replication(c)}
    c3 = cs["e2e.cache_vs_stock"]["confirmation"]["primary12"]
    v["V3_end_to_end"] = {"contrast": c3, "verdict": verdict_confirmed(c3), "guard_vs_stock": cs["e2e.guard_vs_stock"]["confirmation"]["primary12"]}
    c_rt, c_or = cs["retrack.vs_fixed"]["confirmation"]["primary12"], cs["ladder.cache"]["confirmation"]["primary12"]
    v["V4_retrack"] = {"retrack_vs_fixed": c_rt, "oracle_vs_fixed": c_or, "retrack_vs_oracle": cs["retrack.vs_oracle"]["confirmation"]["primary12"],
                       "retention_pct": float(100 * c_rt["delta"] / c_or["delta"]) if c_or["delta"] else float("nan"), "verdict": verdict_retrack(c_rt, c_or)}
    ni = cs["ladder.guard"]["confirmation"].get("nonoverload")
    ov = cs["ladder.guard"]["confirmation"].get("overload")
    v["V5_guard"] = {"nonoverload_cells": sets["nonoverload"], "overload_cells": sets["overload"], "contrast_nonoverload": ni, "contrast_overload": ov, **verdict_guard(ni, ov)}
    for lv, key in (("short_event", "V6_short_events"), ("new_entry_near_onset", "V7_new_entries")):
        con = sub_con[lv]["budget"]
        d: dict = {"n_events": sub_res[lv]["budget"]["Fixed"]["n_events"]}
        for cname in ("cache_vs_fixed", "guard_vs_fixed"):
            d[cname] = {"recall_1.0": con[cname]["recall_1.0"], "verdict_recall": verdict_noninf_recall(con[cname]["recall_1.0"]),
                        "delay_p50_ms": con[cname]["delay_p50_ms"], "verdict_delay": verdict_delay(con[cname]["delay_p50_ms"]), "loc_auc": con[cname]["loc_auc"]}
        v[key] = d
    return {"ds": ds, "auc": auc, "auc_boot_ci": {f"{a}|{c_}|{s}": ci95(b[1]) for (a, c_, s), b in boots.items()}, "sets": sets, "arm_summary": arm_sum, "contrasts": contrasts,
            "events_part_a": ev_part_a, "subgroups": sub_res, "subgroup_contrasts": sub_con, "verdicts": v, "qmax_cache_c1": qmax, "clip_tab": clip_tab, "delays": delays,
            "n_events": n_ev, "build_s": time.perf_counter() - t0}


def stage_analyze(ctx, procs: int, log=print, force: bool = False) -> None:
    for ds in DATASETS:
        if analysis_path(ds).exists() and not force:
            log(f"[{now()}] analyze {ds}: cached")
            continue
        out = analyze_dataset(ctx, ds, procs, log)
        analysis_path(ds).write_bytes(pickle.dumps(out))


# =====================================================================================================================
# 9. Part C2 - long-horizon simulation
# =====================================================================================================================
def sustained_sim(w: e03.World, spec: dict, period: float, hold: int, s: float, loops: int, u0: np.ndarray) -> dict:
    """The whole test stream `loops` times back to back on one FIFO worker (queue carried across loops via `init_free`, fresh service-time draws per loop)."""
    allc = list(range(len(w.names)))
    free = 0.0
    waits, cpu_sum, n_obs, n_fr, shed = [], 0.0, 0, 0, 0
    auc_loops, stats_sum = [], {"reuse_frames": 0, "run_frames": 0, "detect_frames": 0, "win_fresh": 0}
    pol = None
    peak_wait_loop = []
    try:
        for k in range(loops):
            w.u = u0 if k == 0 else np.random.default_rng(POOL_SEED + 1000 * k).random((w.total, 5))
            w._cost.clear()
            c = w.cost(s)
            if pol is None:
                pol = make_policy5(w, c, spec, period)
            R = e03.simulate(w, c, pol, allc, light=False, init_free=free)
            ser, _, _ = e03.delivery(w, R, hold)
            auc_loops.append(100 * e00.pooled_auc(w.labels, ser, None))
            waits.append(R["wait"].astype(np.float32))
            cpu_sum += float(R["cpu"].sum())
            n_obs += int(R["obs"].sum())
            n_fr += len(R["cpu"])
            shed += int(R["stats"]["shed_frames"])
            for kk in stats_sum:
                stats_sum[kk] += int(R["stats"][kk])
            peak_wait_loop.append(float(R["wait"].max()))
            free = max(0.0, float(R["fin"][-1]) - w.total * w.dt)
    finally:
        w.u = u0
        w._cost.clear()
    wait = np.concatenate(waits)
    out = {"frames": n_fr, "minutes": n_fr * w.dt / 60000.0, "cpu_ms_per_video_s": cpu_sum / (n_fr * w.dt / 1000.0), "observed_fraction": n_obs / n_fr, "shed_frame_fraction": shed / n_fr,
           "auc_per_loop": auc_loops, "peak_wait_per_loop_ms": peak_wait_loop, "segments": segment_stats(wait, w.dt),
           "bounded": bounded(wait, w.dt, w.size, e03.E02_RSS_MB[w.ds]), "wait_quantiles_ms": {q: float(np.quantile(wait, q)) for q in (0.5, 0.95, 0.99, 0.999)},
           "wait_max_ms": float(wait.max()), "c1_reuse_fraction_of_inrun": stats_sum["reuse_frames"] / max(1, stats_sum["run_frames"]),
           "guard": (pol.info() if pol is not None else {}), "final_backlog_ms": free}
    minute = max(1, int(round(60000.0 / w.dt)))
    nm = len(wait) // minute
    out["trace_minute_max_ms"] = wait[:nm * minute].reshape(nm, minute).max(axis=1).astype(np.float64).tolist()
    return out


def stage_sustained(ctx, log=print, force: bool = False) -> dict:
    path = WORK_DIR / "sustained_sim.pkl"
    if path.exists() and not force:
        log(f"[{now()}] sustained: cached")
        return pickle.loads(path.read_bytes())
    cfg03 = e03.load_config()
    out: dict = {"cells": {}, "loops": {SHT: 5, UB: 1}}
    t0 = time.perf_counter()
    for ds in DATASETS:
        wl = make_worlds(ctx, ds, cfg03, which=("ml0_locked",))["ml0_locked"]
        run = pickle.loads(run_path(ds).read_bytes())
        u0 = wl.u
        loops = out["loops"][ds]
        for s in SUSTAINED_S:
            cid = cell_id(s, 0.50)
            w_fixed = run["results"][("Fixed", cid)]["knob"]
            w_guard = run["results"][("Cache-C1+Guard", cid)]["knob"]
            arms = {"Fixed-K1": (ARMS["Full-Rate"], 1.0, 0), "Cache-C1": (ARMS["Cache-C1"], 1.0, 0), "Cache-C1+Guard": (ARMS["Cache-C1+Guard"], 1.0, 0),
                    "Fixed@b0.50": (ARMS["Fixed"], w_fixed, max(0, int(round(w_fixed)) - 1)), "Cache-C1+Guard@b0.50": (ARMS["Cache-C1+Guard"], w_guard, max(0, int(round(w_fixed)) - 1))}
            for name in SUSTAINED_ARMS:
                spec, period, hold = arms[name]
                r = sustained_sim(wl, spec, period, hold, s, loops, u0)
                r.update({"period": period, "hold": hold, "s": s, "arm": name})
                out["cells"][(ds, s, name)] = r
                log(f"[{now()}] sustained {ds} s={s} {name}: {r['minutes']:.0f} min, wait max {r['wait_max_ms']:.0f} ms p99 {r['wait_quantiles_ms'][0.99]:.0f} ms, "
                    f"bounded={r['bounded']['bounded']}, AUC/loop {np.mean(r['auc_per_loop']):.2f}")
        out[f"{ds}_frames"] = int(wl.total)
        out[f"{ds}_minutes_one_pass"] = float(wl.total * wl.dt / 60000.0)
    out["build_s"] = time.perf_counter() - t0
    path.write_bytes(pickle.dumps(out))
    return out


# =====================================================================================================================
# 10. Part C1 - real sustained run in the VM (host driver + post-processing)
# =====================================================================================================================
def vm_name(mode: str, s: float) -> str:
    return f"{mode}_s{s:.1f}"


def vm_stage_h264(log=print) -> list[str]:
    from edgelab.analysis import q2_p1_e02_vm as vm

    d = vm.STAGING / "h264_sht"
    d.mkdir(parents=True, exist_ok=True)
    todo = [c for c in vm.SHT_TIMING_CLIPS if not (d / f"{c}.mp4").exists()]
    for c in todo:
        r = vm.run_vm(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "24", "-i", str(vm.FRAMES / c / "%03d.jpg"), "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                       "-pix_fmt", "yuv420p", "-g", "48", str(d / f"{c}.mp4")])
        if r.returncode:
            raise RuntimeError(f"x264 re-encode of {c} failed: {r.stderr[-300:]}")
    log(f"[{now()}] vm: H.264 staging ready ({len(todo)} clips re-encoded)")
    return list(vm.SHT_TIMING_CLIPS)


def stage_vm(ctx, log=print, force: bool = False, only: list[str] | None = None, max_tries: int = 3) -> list[dict]:
    from edgelab.analysis import q2_p1_e02_vm as vm
    from edgelab.pipeline.host_load import LoadSampler, find_vm_pid, summarise
    from edgelab.pipeline.timing_exp010 import wait_for_quiet_host

    cfg = load_config()
    clips = vm_stage_h264(log)
    arms = e01.arm_by_key(ctx.arms)
    sentinel = e01.arm_sentinel(ctx.proto, arms[f"{SHT}/{PRIMARY[SHT]}"])
    c1 = cfg["constants"][SHT]["c1"]
    out_dir, in_dir = VM_DIR / "out", VM_DIR / "in"
    out_dir.mkdir(parents=True, exist_ok=True)
    in_dir.mkdir(parents=True, exist_ok=True)
    manifest = out_dir / "manifest.jsonl"
    done = set()
    if manifest.exists() and not force:
        for line in manifest.read_text().splitlines():
            row = json.loads(line)
            if row["status"] == "ok" and not row["excluded"]:
                done.add(row["run"])
    vm_pid = find_vm_pid()
    rows = []
    for mode, s in VM_RUNS:
        name = vm_name(mode, s)
        if name in done or (only and not any(o in name for o in only)):
            continue
        conf = {"mode": mode, "s_emul": s, "passes": VM_PASSES, "clips": clips, "period": 1.0, "hold": 0, "sentinel": sentinel, "threads": 4, "cap_ms": CAP_MS,
                "rss_marks": list(VM_RSS_MARKS), "c1": {"roi": c1["roi"][2], "ext": c1["ext"][2], "a1": e03.A1}}
        cfgp = in_dir / f"{name}.json"
        cfgp.write_text(json.dumps(conf, indent=1))
        for attempt in range(max_tries):
            quiet = wait_for_quiet_host(vm_pid, 30.0, max_wait_s=900)
            sampler = LoadSampler(vm_pid).start()
            t0 = time.time()
            prefix = out_dir / name
            log(f"[{now()}] vm run {name} try{attempt} ...")
            r = vm.run_vm(["python", "-m", "edgelab.analysis.q2_p1_e05_vm", "longrun", "--config", str(cfgp), "--out", str(prefix)], 4, timeout=4 * 3600)
            load = summarise(sampler.stop())
            ok = r.returncode == 0 and Path(f"{prefix}.npz").exists()
            outside = load["host_outside_vm_pct"]
            row = {"run": name, "mode": mode, "s": s, "attempt": attempt, "status": "ok" if ok else "failed", "seconds": round(time.time() - t0, 1), "pre_run_host_outside_vm_pct": quiet,
                   **load, "excluded": (outside is None or outside > 30.0), "error": None if ok else (r.stderr[-600:] + r.stdout[-300:]), "start": time.strftime("%Y-%m-%d %H:%M:%S")}
            with open(manifest, "a") as f:
                f.write(json.dumps(row) + "\n")
            rows.append(row)
            log(f"[{now()}] vm run {name}: {row['status']} {row['seconds']}s host_outside_vm={outside if outside is None else round(outside, 1)}%{' EXCLUDED' if row['excluded'] else ''}")
            if ok and not row["excluded"]:
                break
    return rows


def vm_delivery_offline(arr: dict, meta: dict) -> tuple[np.ndarray, dict]:
    """Offline reconstruction of the delivered series of a real run from its window records (same landing / late / end rules as the simulator)."""
    n_clip = np.asarray(arr["n_clip"]).astype(np.int64)
    s_emul = float(meta["s_emul"])
    dt_emul = 1000.0 / 24.0 / s_emul
    win = np.asarray(arr["win_rec"])
    total = int(n_clip.sum())
    fin = np.zeros(total)
    if len(win):
        fin[win[:, 2].astype(np.int64)] = win[:, 3]
    R = {"clips": list(range(len(n_clip))), "w_pos": win[:, 0].astype(np.int64), "w_t": win[:, 1].astype(np.int64), "w_gs": win[:, 2].astype(np.int64), "w_val": win[:, 4], "fin": fin}
    stub = SimpleNamespace(n_frames=n_clip, dt=dt_emul, sentinel=float(meta["sentinel"]))
    ser, stats, _ = e03.delivery(stub, R, int(meta["hold"]), cap_ms=float(meta["cap_ms"]) / s_emul)
    return np.concatenate(ser), stats


def analyze_vm_run(ctx, wl: e03.World, name: str, log=print) -> dict:
    out_dir = VM_DIR / "out"
    z = np.load(out_dir / f"{name}.npz")
    arr = {k: z[k] for k in z.files}
    meta = json.loads((out_dir / f"{name}.json").read_text())
    s = float(meta["s_emul"])
    n = int(meta["frames"])
    dt = 1000.0 / 24.0
    wait_pi = arr["wait_ms"] * s
    svc = arr["svc_ms"]
    obs = arr["obs"].astype(bool)
    reused = arr["reused"].astype(bool)
    marks = {int(k): v for k, v in meta["rss_marks_mb"].items()}
    ser_off, dstats = vm_delivery_offline(arr, meta)
    online = arr["series"]
    clips = meta["clips"]
    pos = {c: i for i, c in enumerate(wl.names)}
    n_first = [int(x) for x in np.asarray(arr["n_clip"])[:len(clips)]]  # the VM clips are the first frames of the SHT test clips
    labels_stream = [wl.labels[pos[c]][:n_c] for c, n_c in zip(clips, n_first)] * int(meta["passes"])
    cut = np.cumsum([len(x) for x in labels_stream])[:-1]
    auc_online = 100 * e00.pooled_auc(labels_stream, list(np.split(online, cut)), None)
    # simulation of the same stream (E02 pools x s), same policy / knob
    spec = dict(ARMS["Full-Rate"] if meta["mode"] == "fixed_k1" else (ARMS["Cache-C1+Guard"] if meta["mode"] == "cache_c1_guard" else ARMS["Cache-C1"]), name=meta["mode"])
    ids = [pos[c] for c in clips] * int(meta["passes"])
    c = wl.cost(s)
    pol = make_policy5(wl, c, spec, float(meta["period"]))
    R = e03.simulate(wl, c, pol, ids, light=False)
    ser_sim, _, _ = e03.delivery(wl, R, int(meta["hold"]))
    ser_cut = [x[:n_c] for x, n_c in zip(ser_sim, n_first * int(meta["passes"]))]  # causal series: the prefix is unaffected by later frames
    auc_sim = 100 * e00.pooled_auc(labels_stream, ser_cut, None)
    full_len = [int(wl.n_frames[i]) for i in ids]
    wait_sim = np.concatenate([x[:n_c] for x, n_c in zip(np.split(R["wait"], np.cumsum(full_len)[:-1]), n_first * int(meta["passes"]))])
    qs = [0.5, 0.95, 0.99]
    th = np.array_split(np.arange(n), 3)
    blocks = meta["blocks"]
    thermal = {"steal_fraction": meta["steal_total_jiffies"] / max(1.0, meta["jiffies_total"]), "cooling_start": meta["cooling_start"], "cooling_end": meta["cooling_end"],
               "sensor": meta["thermal_sensor"], "block_svc_mean_ms": [b["svc_mean_ms"] for b in blocks], "block_cpu_mean_ms": [b["cpu_mean_ms"] for b in blocks],
               "block_steal_pct": [100 * b["steal_jiffies"] / max(1.0, b["total_jiffies"]) for b in blocks], "block_loadavg": [b["loadavg"] for b in blocks],
               "block_rss_mb": [b["rss_mb"] for b in blocks],
               "svc_drift_pct_first_to_last_block": float(100 * (blocks[-1]["svc_mean_ms"] - blocks[0]["svc_mean_ms"]) / blocks[0]["svc_mean_ms"]) if len(blocks) >= 2 else float("nan"),
               "svc_by_pass_ms": [float(svc[i].mean()) for i in np.array_split(np.arange(n), int(meta["passes"]))],
               "cpu_by_pass_ms": [float(arr["cpu_ms"][i].mean()) for i in np.array_split(np.arange(n), int(meta["passes"]))]}
    thermal["svc_drift_pct_first_to_last_pass"] = float(100 * (thermal["svc_by_pass_ms"][-1] - thermal["svc_by_pass_ms"][0]) / thermal["svc_by_pass_ms"][0])
    thermal["cpu_drift_pct_first_to_last_pass"] = float(100 * (thermal["cpu_by_pass_ms"][-1] - thermal["cpu_by_pass_ms"][0]) / thermal["cpu_by_pass_ms"][0])
    cpu_total = float(arr["cpu_ms"].sum())
    res = {"run": name, "mode": meta["mode"], "s": s, "frames": n, "wall_s": meta["wall_total_s"], "offered_fps": 24.0 * s, "achieved_fps": n / meta["wall_total_s"],
           "busy_fps": n / (svc.sum() / 1000.0), "svc_mean_ms": float(svc.mean()), "svc_p95_ms": float(np.quantile(svc, 0.95)),
           "cpu_ms_per_video_s_vm": cpu_total / (n * dt / 1000.0), "cpu_ms_per_video_s_pi_eq": s * cpu_total / (n * dt / 1000.0), "cpu_util_pct_of_wall": 100 * meta["cpu_total_s"] / meta["wall_total_s"] / 4,
           "c1_hit_rate_of_inrun": float(reused.sum() / max(1, obs.sum())), "c1_hit_rate_of_frames": float(reused.mean()), "observed_fraction": float(obs.mean()), "detector_frames": int((obs & ~reused).sum()),
           "wait_pi_ms": {**{f"p{int(q * 100)}": float(np.quantile(wait_pi, q)) for q in qs}, "max": float(wait_pi.max()), "mean": float(wait_pi.mean())},
           "wait_vm_ms": {**{f"p{int(q * 100)}": float(np.quantile(arr["wait_ms"], q)) for q in qs}, "max": float(arr["wait_ms"].max())},
           "wait_thirds_pi_mean": [float(wait_pi[i].mean()) for i in th], "drift_pi_ms": float(wait_pi[th[2]].mean() - wait_pi[th[0]].mean()),
           "wait_segments": segment_stats(wait_pi, dt, 1.0), "rss_marks_mb": marks, "rss_growth": rss_growth(marks, 2500), "peak_rss_mb": meta["peak_rss_mb"],
           "guard_transitions": arr["transitions"].tolist(), "guard": meta["guard"], "shed_frame_fraction": float(arr["shed"].mean()),
           "windows_scored": int(arr["scored"].sum()), "dropped_late": meta["dropped_late"], "dropped_end": meta["dropped_end"],
           "auc_online": auc_online, "auc_sim": auc_sim, "online_equals_offline": bool(np.allclose(online, ser_off, rtol=0, atol=1e-9)), "online_offline_max_abs_diff": float(np.max(np.abs(online - ser_off))),
           "series_finite": bool(np.isfinite(online).all()), "arrival_monotone": bool((np.diff(arr["arrival_ms"]) >= 0).all()), "n_frames_ok": int(n) >= 10000,
           "sim": {"cost_ms_per_video_s_pi_eq": e03.total_cost(wl, R), "wait_mean_ms": float(wait_sim.mean()), "wait_p99_ms": float(np.quantile(wait_sim, 0.99)), "wait_max_ms": float(wait_sim.max()),
                   "note": "simulated on the full-length clips; AUC and waits evaluated on the first n frames of each clip that the VM run decoded",
                   "c1_reuse_fraction_of_inrun": R["stats"]["reuse_frames"] / max(1, R["stats"]["run_frames"]), "observed_fraction": float(R["obs"].mean()), "frames": len(R["wait"])},
           "thermal_proxy": thermal, "affinity": meta["affinity"], "versions": meta.get("versions"), "x264": meta["x264"],
           "trace_wait_pi_100f": [float(wait_pi[a:a + 100].mean()) for a in range(0, n, 100)], "trace_rss": arr["rss_series"].tolist()}
    return res


def stage_vm_analyze(ctx, log=print) -> dict:
    cfg03 = e03.load_config()
    wl = make_worlds(ctx, SHT, cfg03, which=("ml0_locked",))["ml0_locked"]
    man = {}
    mp_ = VM_DIR / "out" / "manifest.jsonl"
    for line in mp_.read_text().splitlines():
        r = json.loads(line)
        if r["status"] == "ok":
            man[r["run"]] = r
    runs = {}
    for mode, s in VM_RUNS:
        name = vm_name(mode, s)
        if name in man and (VM_DIR / "out" / f"{name}.npz").exists():
            runs[name] = analyze_vm_run(ctx, wl, name, log)
            runs[name]["host_load"] = {k: man[name].get(k) for k in ("host_outside_vm_pct", "pre_run_host_outside_vm_pct", "excluded", "attempt", "seconds")}
            log(f"[{now()}] vm analyze {name}: AUC online {runs[name]['auc_online']:.2f} sim {runs[name]['auc_sim']:.2f}, p99 wait {runs[name]['wait_pi_ms']['p99']:.0f} ms (Pi-eq), "
                f"RSS growth {runs[name]['rss_growth']['growth_pct']:.1f}%")
    out = {"runs": runs, "manifest": man}
    (WORK_DIR / "vm_summary.json").write_text(json.dumps(plain(out), indent=1))
    return out


# =====================================================================================================================
# 11. Report stage: results.csv, events.csv, sustained.csv, decisions.jsonl, summary.json, figures, report.md
# =====================================================================================================================
def md_table(headers: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(str(x) for x in r) + " |" for r in rows]
    return "\n".join(out)


def fnum(x, d: int = 2) -> str:
    return "n/a" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:.{d}f}"


def fci(c: dict | None, d: int = 2) -> str:
    """'+2.39 [+1.39, +3.46]*' (* = the 95% CI excludes 0)."""
    if c is None or not all(math.isfinite(c[k]) for k in ("delta", "lo", "hi")):
        return "n/a"
    star = "*" if (c["lo"] > 0 or c["hi"] < 0) else ""
    return f"{c['delta']:+.{d}f} [{c['lo']:+.{d}f}, {c['hi']:+.{d}f}]{star}"


def fpt(m: dict | None, d: int = 1) -> str:
    """'45.2 [40.1, 50.3]' for {'point', 'lo', 'hi'}."""
    if m is None:
        return "n/a"
    return f"{fnum(m['point'], d)} [{fnum(m['lo'], d)}, {fnum(m['hi'], d)}]"


def fmean(m: dict | None, d: int = 2) -> str:
    return "n/a" if m is None else f"{m['mean']:.{d}f} [{m['lo']:.{d}f}, {m['hi']:.{d}f}]"


def load_all() -> dict:
    out = {"run": {}, "ana": {}, "st": {}}
    for ds in DATASETS:
        out["run"][ds] = pickle.loads(run_path(ds).read_bytes())
        out["ana"][ds] = pickle.loads(analysis_path(ds).read_bytes())
        out["st"][ds] = pickle.loads(strata_path(ds).read_bytes())
    out["sus"] = pickle.loads((WORK_DIR / "sustained_sim.pkl").read_bytes())
    vp = WORK_DIR / "vm_summary.json"
    out["vm"] = json.loads(vp.read_text()) if vp.exists() else None
    out["tbl_checks"] = json.loads((WORK_DIR / "checks_tables.json").read_text())
    return out


def verdict_v8(sus: dict) -> dict:
    rows, ok = [], True
    for (ds, s, arm), r in sorted(sus["cells"].items()):
        if arm != "Cache-C1+Guard":
            continue
        rows.append({"dataset": ds, "s": s, **{k: r["bounded"][k] for k in ("bounded", "max_ms", "p99_ms", "drift_ms", "queue_mem_mb")}})
        ok &= bool(r["bounded"]["bounded"])
    diverged = [{"dataset": ds, "s": s, "arm": arm, "max_ms": r["wait_max_ms"], "drift_ms": r["bounded"]["drift_ms"]} for (ds, s, arm), r in sorted(sus["cells"].items())
                if arm in ("Fixed-K1", "Cache-C1") and not r["bounded"]["bounded"]]
    return {"verdict": "bounded" if ok else "not bounded in every (dataset, s)", "cells": rows, "unguarded_unbounded": diverged}


def verdict_v9(vm: dict | None) -> dict:
    if not vm or not vm["runs"]:
        return {"verdict": "not run", "runs": {}}
    rows, ok = {}, True
    for name, r in vm["runs"].items():
        if r["mode"] != "cache_c1_guard":
            continue
        chk = {"frames_ge_10000": r["frames"] >= 10000, "rss_growth_le_5pct": r["rss_growth"]["growth_pct"] <= 5.0, "p99_wait_le_2000": r["wait_pi_ms"]["p99"] <= CAP_MS,
               "host_load_ok": not r["host_load"]["excluded"]}
        rows[name] = {"s": r["s"], "frames": r["frames"], "rss_growth_pct": r["rss_growth"]["growth_pct"], "p99_wait_pi_ms": r["wait_pi_ms"]["p99"], "checks": chk, "pass": all(chk.values())}
        ok &= all(chk.values())
    need = {vm_name("cache_c1_guard", s) for m, s in VM_RUNS if m == "cache_c1_guard"}
    ok &= need <= set(rows)
    return {"verdict": "sustained" if ok else "not sustained / incomplete", "runs": rows}


def correctness_rules(D: dict) -> list[dict]:
    rules = []

    def add(rid, ok, detail, required=True):
        rules.append({"kind": "rule", "id": rid, "pass": bool(ok), "required": required, "detail": detail})

    ck = {ds: D["run"][ds]["checks"] for ds in DATASETS}
    add("E5-D1", all(ck[ds]["E5-D1"]["pass"] for ds in DATASETS), {ds: ck[ds]["E5-D1"] for ds in DATASETS})
    add("E5-D1b", all(ck[ds]["E5-D1b"]["pass"] for ds in DATASETS), {ds: ck[ds]["E5-D1b"] for ds in DATASETS}, required=False)
    add("E5-D2", all(ck[ds]["E5-D2"]["pass"] for ds in DATASETS), {ds: ck[ds]["E5-D2"] for ds in DATASETS})
    tc = D["tbl_checks"]
    add("E5-D3", all(tc[ds]["D3_builder_equal_e03"]["pass"] for ds in DATASETS), {ds: tc[ds]["D3_builder_equal_e03"] for ds in DATASETS})
    add("E5-D4", all(tc[ds]["D4_pattern_sources"]["pass"] for ds in DATASETS), {ds: tc[ds]["D4_pattern_sources"] for ds in DATASETS})
    add("E5-D5", all(ck[ds]["E5-D5"]["pass"] for ds in DATASETS), {ds: ck[ds]["E5-D5"] for ds in DATASETS})
    add("E5-D6", all(ck[ds]["E5-D6"]["pass"] for ds in DATASETS), {ds: {k: v for k, v in ck[ds]["E5-D6"].items() if k != "rows"} for ds in DATASETS})
    vm = D["vm"]
    if vm and vm["runs"]:
        runs = vm["runs"]
        d7 = {n: {"frames": r["frames"], "n_frames_ok": r["n_frames_ok"], "arrival_monotone": r["arrival_monotone"], "series_finite": r["series_finite"],
                  "online_equals_offline": r["online_equals_offline"], "max_abs_diff": r["online_offline_max_abs_diff"]} for n, r in runs.items()}
        add("E5-D7", all(v["n_frames_ok"] and v["arrival_monotone"] and v["series_finite"] and v["online_equals_offline"] for v in d7.values()), d7)
    else:
        add("E5-D7", False, "no VM run")
    ok8, det8 = True, {}
    for ds in DATASETS:
        a = D["ana"][ds]
        for arm, per_set in a["arm_summary"].items():
            for sname in ("primary12", "native3"):
                if sname not in per_set:
                    continue
                for sub in ("confirmation", "all"):
                    if sub not in per_set[sname]:
                        continue
                    cids = a["sets"][sname]
                    pts = [a["auc"][arm][c][sub] for c in cids if c in a["auc"].get(arm, {})]
                    if pts and abs(np.mean(pts) - per_set[sname][sub]["mean"]) > 1e-9:
                        ok8 = False
                        det8[f"{ds}|{arm}|{sname}|{sub}"] = [float(np.mean(pts)), per_set[sname][sub]["mean"]]
    add("E5-D8", ok8 and N_BOOT == 2000, {"resamples": N_BOOT, "mismatches": det8})
    add("E5-D9", all(D["st"][ds]["checks"]["E5-D9"]["pass"] for ds in DATASETS), {ds: D["st"][ds]["checks"]["E5-D9"] for ds in DATASETS})
    return rules


def write_results_csv(D: dict, w_names: dict) -> int:
    cols = ["dataset", "arm", "cell", "s", "beta", "world", "level", "guard", "k1", "period", "hold", "status", "cost_ms_per_s", "target_ms_per_s", "cost_ratio", "auc_dev", "auc_confirmation", "auc_all",
            "auc_all_ci_lo", "auc_all_ci_hi", "auc_confirmation_ci_lo", "auc_confirmation_ci_hi", "ideal_auc_all", "clips", "events", "hit_0.5", "hit_1.0", "hit_2.0", "false_alerts", "normal_frames",
            "wait_mean_ms", "wait_p95_ms", "wait_max_ms", "utilisation", "observed_fraction", "c1_reuse_fraction", "shed_frames", "dropped_late", "mem_mb"]
    n = 0
    with open(EXP_DIR / "results.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=cols + ["video", "video_events", "video_hit_0.5", "video_hit_1.0", "video_hit_2.0", "video_false_alerts", "video_normal_frames", "video_delay_p50_ms",
                                                   "video_cpu_ms_per_s", "video_wait_mean_ms", "video_wait_max_ms"])
        wr.writeheader()
        for ds in DATASETS:
            run, a = D["run"][ds], D["ana"][ds]
            names = w_names[ds]
            for (arm, cid), r in sorted(run["results"].items()):
                tab = a["clip_tab"][(arm, cid)]
                au = a["auc"][arm][cid]
                lo_a, hi_a = a["auc_boot_ci"].get(f"{arm}|{cid}|all", (float("nan"),) * 2)
                lo_c, hi_c = a["auc_boot_ci"].get(f"{arm}|{cid}|confirmation", (float("nan"),) * 2)
                cfg = r.get("cfg", {})
                st = r.get("stats", {})
                q = r.get("queue", {})
                base = {"dataset": ds, "arm": arm, "cell": cid, "s": r["s"], "beta": r["beta"], "world": cfg.get("world", "e03"), "level": cfg.get("lvl", ""), "guard": cfg.get("guard", ""),
                        "k1": cfg.get("k1", ""), "period": r.get("knob"), "hold": r.get("hold"), "status": r.get("status"), "cost_ms_per_s": r.get("cost"), "target_ms_per_s": r.get("target"),
                        "cost_ratio": r.get("ratio"), "auc_dev": au["dev"], "auc_confirmation": au["confirmation"], "auc_all": au["all"], "auc_all_ci_lo": lo_a, "auc_all_ci_hi": hi_a,
                        "auc_confirmation_ci_lo": lo_c, "auc_confirmation_ci_hi": hi_c, "ideal_auc_all": au["ideal_all"], "clips": len(names), "events": int(tab[:, 0].sum()),
                        "hit_0.5": int(tab[:, 1].sum()), "hit_1.0": int(tab[:, 2].sum()), "hit_2.0": int(tab[:, 3].sum()), "false_alerts": int(tab[:, 5].sum()), "normal_frames": int(tab[:, 6].sum()),
                        "wait_mean_ms": q.get("mean_wait_ms"), "wait_p95_ms": q.get("p95_wait_ms"), "wait_max_ms": q.get("max_wait_ms"), "utilisation": q.get("rho"),
                        "observed_fraction": float(np.mean(r["observed"])) if "observed" in r else "", "c1_reuse_fraction": (st["reuse_frames"] / max(1, st["run_frames"])) if st else "",
                        "shed_frames": r.get("policy", {}).get("shed_frames", 0), "dropped_late": r.get("dstats", {}).get("dropped_late", ""), "mem_mb": r.get("mem_mb")}
                for i, nm in enumerate(names):
                    row = dict(base, video=nm, video_events=int(tab[i, 0]), **{"video_hit_0.5": int(tab[i, 1]), "video_hit_1.0": int(tab[i, 2]), "video_hit_2.0": int(tab[i, 3])},
                               video_delay_p50_ms=("" if not np.isfinite(tab[i, 4]) else float(tab[i, 4])), video_false_alerts=int(tab[i, 5]), video_normal_frames=int(tab[i, 6]),
                               video_cpu_ms_per_s=float(r["clip_cpu_ms_per_s"][i]) if "clip_cpu_ms_per_s" in r else "", video_wait_mean_ms=float(r["clip_wait_mean_ms"][i]) if "clip_wait_mean_ms" in r else "",
                               video_wait_max_ms=float(r["clip_wait_max_ms"][i]) if "clip_wait_max_ms" in r else "")
                    wr.writerow(row)
                    n += 1
    return n


def write_events_csv(D: dict, w_names: dict) -> int:
    n = 0
    with open(EXP_DIR / "events.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["dataset", "arm", "cell", "event", "video", "start_frame", "end_frame", "duration_frames", "short", "new_entry_near_onset", "fast_motion", "high_crowd", "delay_ms", "detected_0.5s",
                     "detected_1.0s", "detected_2.0s"])
        for ds in DATASETS:
            st, a = D["st"][ds], D["ana"][ds]
            fps = FPS[ds]
            for (arm, cid), dl in sorted(a["delays"].items()):
                for k in range(st["n_events"]):
                    d = dl[k]
                    det = bool(np.isfinite(d))
                    wr.writerow([ds, arm, cid, k, w_names[ds][int(st["clip"][k])], int(st["s"][k]), int(st["e"][k]), int(st["e"][k] - st["s"][k] + 1), int(st["short"][k]), int(st["entry"][k]),
                                 int(st["fast"][k]), int(st["crowd"][k]), (f"{1000.0 * d / fps:.1f}" if det else ""), int(det and d <= 0.5 * fps), int(det and d <= 1.0 * fps), int(det and d <= 2.0 * fps)])
                    n += 1
    return n


def write_sustained_csv(D: dict) -> int:
    n = 0
    with open(EXP_DIR / "sustained.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["part", "dataset", "arm_or_run", "s", "segment", "start_min", "frames", "wait_mean_ms", "wait_p50_ms", "wait_p95_ms", "wait_p99_ms", "wait_max_ms", "svc_mean_ms", "cpu_mean_ms",
                     "steal_pct", "loadavg", "rss_mb", "c1_reuse"])
        for (ds, s, arm), r in sorted(D["sus"]["cells"].items()):
            for sg in r["segments"]:
                wr.writerow(["C2", ds, arm, s, sg["segment"], f"{sg['start_min']:.2f}", sg["frames"], f"{sg['wait_mean_ms']:.2f}", f"{sg['wait_p50_ms']:.2f}", f"{sg['wait_p95_ms']:.2f}",
                             f"{sg['wait_p99_ms']:.2f}", f"{sg['wait_max_ms']:.2f}", "", "", "", "", "", ""])
                n += 1
        vm = D["vm"]
        for name, r in (vm["runs"].items() if vm else []):
            t = r["thermal_proxy"]
            for sg in r["wait_segments"]:
                wr.writerow(["C1-wait", SHT, name, r["s"], sg["segment"], f"{sg['start_min']:.3f}", sg["frames"], f"{sg['wait_mean_ms']:.2f}", f"{sg['wait_p50_ms']:.2f}", f"{sg['wait_p95_ms']:.2f}",
                             f"{sg['wait_p99_ms']:.2f}", f"{sg['wait_max_ms']:.2f}", "", "", "", "", "", ""])
                n += 1
            for k in range(len(t["block_svc_mean_ms"])):
                wr.writerow(["C1-block", SHT, name, r["s"], k, "", 1000, "", "", "", "", "", f"{t['block_svc_mean_ms'][k]:.3f}", f"{t['block_cpu_mean_ms'][k]:.3f}", f"{t['block_steal_pct'][k]:.3f}",
                             t["block_loadavg"][k], f"{t['block_rss_mb'][k]:.1f}", ""])
                n += 1
    return n


def make_figures(D: dict) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figs = []
    order = ["Stock-Default-K1", "ML0-Default-K1", "Full-Rate", "Stock-Baseline", "ML0-Default-Fixed", "Fixed", "Fixed+Guard", "Cache-C1", "Cache-C1+Guard", "C1-ReTrack", "Independent-Combo"]
    for ds in DATASETS:
        a = D["ana"][ds]
        fig, axs = plt.subplots(1, 2, figsize=(13, 4.2), sharey=True)
        for ax, sname in zip(axs, ("primary12", "native3")):
            arms = [x for x in order if x in a["arm_summary"] and sname in a["arm_summary"][x]]
            m = [a["arm_summary"][x][sname]["confirmation"] for x in arms]
            ax.bar(range(len(arms)), [x["mean"] for x in m], yerr=[[x["mean"] - x["lo"] for x in m], [x["hi"] - x["mean"] for x in m]], color="#4477aa", capsize=3)
            ax.set_xticks(range(len(arms)))
            ax.set_xticklabels(arms, rotation=60, ha="right", fontsize=8)
            ax.set_title(f"{ds} {'12 E03 cells' if sname == 'primary12' else 'native s=1 cells'} (confirmation, 95% CI)")
            lo = min(x["lo"] for x in m)
            ax.set_ylim(max(0, lo - 3), max(x["hi"] for x in m) + 1)
        axs[0].set_ylabel("delivered AUC x 100 (cell mean)")
        fig.tight_layout()
        p = EXP_DIR / f"fig_{ds}_ladder.png"
        fig.savefig(p, dpi=130)
        plt.close(fig)
        figs.append(p.name)
    fig, axs = plt.subplots(2, 4, figsize=(16, 6.5))
    for r_i, ds in enumerate(DATASETS):
        for c_i, s in enumerate(SUSTAINED_S):
            ax = axs[r_i, c_i]
            for arm, col in (("Fixed-K1", "#cc3311"), ("Cache-C1", "#ee7733"), ("Cache-C1+Guard", "#0077bb"), ("Fixed@b0.50", "#999999"), ("Cache-C1+Guard@b0.50", "#009988")):
                r = D["sus"]["cells"].get((ds, s, arm))
                if r:
                    ax.plot(np.arange(len(r["trace_minute_max_ms"])) / 60.0, r["trace_minute_max_ms"], lw=0.9, color=col, label=arm)
            ax.axhline(CAP_MS, color="k", ls=":", lw=0.8)
            ax.set_yscale("symlog", linthresh=100)
            ax.set_title(f"{ds} s={s}", fontsize=9)
            ax.set_xlabel("stream hours")
            if c_i == 0:
                ax.set_ylabel("max queue wait per minute (Pi-eq ms)")
    axs[0, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(EXP_DIR / "fig_c2_queue_traces.png", dpi=120)
    plt.close(fig)
    figs.append("fig_c2_queue_traces.png")
    vm = D["vm"]
    if vm and vm["runs"]:
        fig, axs = plt.subplots(1, 2, figsize=(14, 4.2))
        for name, r in vm["runs"].items():
            axs[0].plot(np.arange(len(r["trace_wait_pi_100f"])) * 100, r["trace_wait_pi_100f"], lw=0.9, label=name)
            tr = np.array(r["trace_rss"])
            if tr.size:
                axs[1].plot(tr[:, 0], tr[:, 1], lw=0.9, label=name)
        axs[0].axhline(CAP_MS, color="k", ls=":", lw=0.8)
        axs[0].set_yscale("symlog", linthresh=50)
        axs[0].set_xlabel("frame")
        axs[0].set_ylabel("mean queue wait per 100 frames (Pi-eq ms)")
        axs[0].legend(fontsize=7)
        axs[1].set_xlabel("frame")
        axs[1].set_ylabel("process RSS (MB)")
        fig.tight_layout()
        fig.savefig(EXP_DIR / "fig_vm_traces.png", dpi=120)
        plt.close(fig)
        figs.append("fig_vm_traces.png")
    return figs


def stage_report(ctx=None, log=print) -> dict:
    from edgelab.analysis import q2_p1_e02 as _e02  # noqa: F401

    D = load_all()
    cfg03 = e03.load_config()
    names = {}
    for ds in DATASETS:
        if ctx is not None:
            names[ds] = list(make_worlds(ctx, ds, cfg03, which=("ml0_locked",))["ml0_locked"].names)
        else:
            names[ds] = [str(i) for i in range(len(D["run"][ds]["n_frames"]))]
    rules = correctness_rules(D)
    v8, v9 = verdict_v8(D["sus"]), verdict_v9(D["vm"])
    n_res = write_results_csv(D, names)
    n_ev = write_events_csv(D, names)
    n_sus = write_sustained_csv(D)
    figs = make_figures(D)
    log(f"[{now()}] report: results.csv {n_res} rows, events.csv {n_ev} rows, sustained.csv {n_sus} rows, figures {figs}")
    cfg = load_config()
    amend = [{"kind": "amendment", "id": a, "text": t} for a, t in (
        ("A0", "UB stream is 92,640 frames = 51.5 min (task text 105,711 / 58.7 min is wrong); SHT 40,791 frames = 28.3 min is correct."),
        ("A1", "Diagnostic E5-D1b tolerance 0.1 -> 0.75 pt after seeing the 0.05-0.56 pt offset to E01 caused by E03's run-table handicap (E3-D2); no verdict depends on it."),
        ("A2", "E5-D3 compares the genuine window of every row (E03's pool also stores C1-substituted variants)."),
        ("A3", "VM clips are the first 240 frames of each SHT clip: real-run AUC and the same-stream simulation are evaluated on those prefixes; pass-over-pass drift added."))]
    summary = {"experiment": "Q2-P1-E05", "title": "Confirmation", "training": False, "seed": SEED, "generated": time.strftime("%Y-%m-%d %H:%M:%S"), "plan_sha256": cfg["plan_sha256"],
               "hardware": {"physical_pi_connected": False, "machine": MACHINE_TAG, "note": "every cost is a VM measurement x s; no Pi number"},
               "verdicts": {ds: D["ana"][ds]["verdicts"] for ds in DATASETS}, "correctness": {"all_pass": all(r["pass"] for r in rules if r["required"]), "rules": {r["id"]: r["pass"] for r in rules}},
               "arm_summary": {ds: D["ana"][ds]["arm_summary"] for ds in DATASETS}, "contrasts": {ds: D["ana"][ds]["contrasts"] for ds in DATASETS},
               "subgroups": {ds: D["ana"][ds]["subgroups"] for ds in DATASETS}, "subgroup_contrasts": {ds: D["ana"][ds]["subgroup_contrasts"] for ds in DATASETS},
               "events_part_a": {ds: D["ana"][ds]["events_part_a"] for ds in DATASETS}, "strata_counts": {ds: D["st"][ds]["counts"] for ds in DATASETS},
               "sustained_sim": {f"{ds}|{s}|{arm}": {k: v for k, v in r.items() if k not in ("trace_minute_max_ms",)} for (ds, s, arm), r in sorted(D["sus"]["cells"].items())},
               "vm": ({n: {k: v for k, v in r.items() if k not in ("trace_wait_pi_100f", "trace_rss")} for n, r in D["vm"]["runs"].items()} if D["vm"] else None), "figures": figs}
    summary["verdicts"]["V8_long_horizon"] = v8
    summary["verdicts"]["V9_vm_sustained"] = v9
    summary["verdicts"] = plain(summary["verdicts"])
    (EXP_DIR / "summary.json").write_text(json.dumps(plain(summary), indent=1))
    rows = [{"kind": "lock", "plan_sha256": cfg["plan_sha256"], "builds_on": cfg["builds_on"]}] + amend + rules
    for ds in DATASETS:
        for k, v in D["ana"][ds]["verdicts"].items():
            rows.append({"kind": "verdict", "dataset": ds, "id": k, "verdict": v.get("verdict", {k2: v[k2] for k2 in ("no_harm", "overload") if k2 in v}), "detail": v})
        rows.append({"kind": "strata", "dataset": ds, "counts": D["st"][ds]["counts"], "events": D["st"][ds]["n_events"], "motion_thr": D["st"][ds]["motion_thr"]})
        rows.append({"kind": "calibration", "dataset": ds, "theta": D["run"][ds]["theta"], "w_ref": D["run"][ds]["w_ref"], "targets": D["run"][ds]["targets"]})
        for (arm, cid), r in sorted(D["run"][ds]["results"].items()):
            rows.append({"kind": "calibration_cell", "dataset": ds, "arm": arm, "cell": cid, "knob": r.get("knob"), "status": r.get("status"), "ratio": r.get("ratio"), "hold": r.get("hold")})
    rows.append({"kind": "verdict", "dataset": "both", "id": "V8_long_horizon", "verdict": v8["verdict"], "detail": v8})
    rows.append({"kind": "verdict", "dataset": SHT, "id": "V9_vm_sustained", "verdict": v9["verdict"], "detail": v9})
    if D["vm"]:
        for n, r in D["vm"]["runs"].items():
            rows.append({"kind": "guard_transitions", "run": n, "n": len(r["guard_transitions"]), "transitions": r["guard_transitions"][:400], "guard": r["guard"]})
    with open(EXP_DIR / "decisions.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(plain(r)) + "\n")
    text = render_report(D, rules, v8, v9, figs)
    (EXP_DIR / "report.md").write_text(text)
    log(f"[{now()}] report: wrote summary.json, decisions.jsonl ({len(rows)} rows), report.md ({len(text.splitlines())} lines)")
    return summary



SUB_ORDER = ("Full-Rate", "Fixed", "Cache-C1", "Cache-C1+Guard", "C1-ReTrack", "Stock-Baseline")
LEVEL_TITLES = {"all": "all events", "short_event": "short (<= 2.5 s)", "sustained_event": "sustained (> 2.5 s)", "new_entry_near_onset": "new person within +-1 s of onset",
                "existing_tracks_only": "no new entry near onset", "sudden_fast_motion": "fast motion (p90 >= E03 q85)", "moderate_motion": "moderate motion",
                "high_crowd": "crowd (>= 4 tracks)", "low_crowd": "low crowd (< 4 tracks)"}
LADDER_ORDER = [("Stock-Default-K1", "1. Stock frontend, default tracker, K=1"), ("ML0-Default-K1", "2. ML0 frontend, default tracker, K=1"), ("Full-Rate", "3. ML0 + locked tracker, K=1 (Full-Rate)"),
                ("Fixed", "4. Fixed-Burst (cost-matched)"), ("Cache-C1", "5a. Cache-C1 (primary, oracle identity)"), ("Cache-C1+Guard", "5b. Cache-C1+Guard"),
                ("Stock-Baseline", "ctl. Stock-Baseline (stock + default tracker, Fixed-Burst)"), ("ML0-Default-Fixed", "ctl. ML0 + default tracker, Fixed-Burst"), ("Fixed+Guard", "ctl. Fixed+Guard"),
                ("C1-ReTrack", "ctl. C1-ReTrack (non-oracle)"), ("C1-ReTrack+Guard", "ctl. C1-ReTrack+Guard"), ("Independent-Combo", "ref. E03 Independent-Combo (12 cells)")]


def c1_saving(D: dict, ds: str) -> list[dict]:
    """CPU of Cache-C1 vs Full-Rate (K=1, no reuse) at the same schedule (W = 1): the beta = 1.0 cells where Cache-C1 sits at W = 1."""
    out = []
    for s_ in (1.0, 2.0, 2.5, 3.0):
        cid = cell_id(s_, 1.0)
        a, b = D["run"][ds]["results"].get(("Full-Rate", cid)), D["run"][ds]["results"].get(("Cache-C1", cid))
        if a and b and abs(b["knob"] - 1.0) < 1e-9:
            out.append({"s": s_, "full_rate": a["cost"], "cache_c1": b["cost"], "saving_pct": 100 * (1 - b["cost"] / a["cost"])})
    return out


def guard_zone(D: dict, ds: str) -> list[dict]:
    """Cells where the guard can act (unguarded Cache-C1 peak wait > 400 ms) but the unguarded queue stays below the 2 s cap."""
    a = D["ana"][ds]
    return [{"cell": c, "peak_ms": a["qmax_cache_c1"][c], "c": a["contrasts"]["ladder.guard"]["confirmation"].get(c)} for c in cells_ids(CELLS) if SHED_ON_MS < a["qmax_cache_c1"][c] <= CAP_MS]


def cooling_txt(t: dict) -> str:
    def st(x):
        return sorted({v.get("cur_state") for v in x.values()}) if isinstance(x, dict) and x else []

    a, b = st(t["cooling_start"]), st(t["cooling_end"])
    return f"{len(t['cooling_start'])} devices, cur_state {a} → {b}"


def render_report(D: dict, rules: list[dict], v8: dict, v9: dict, figs: list[str]) -> str:
    A = D["ana"]
    L: list[str] = []
    add = L.append
    ok_all = all(r["pass"] for r in rules if r["required"])
    n_pass = sum(1 for r in rules if r["pass"])
    vm = D["vm"]

    def V(ds, k):
        return A[ds]["verdicts"][k]

    def c12(ds, name, sub="confirmation", sname="primary12"):
        return A[ds]["contrasts"][name][sub].get(sname)

    add("# Q2-P1-E05 — Confirmation: report\n")
    add("**Hardware / honesty disclosure.** The physical Raspberry Pi 5 is **not connected**. Every CPU cost is an `edge-os-arm64-vm` measurement (4 vCPU ARM64 VM on an Apple M2 Pro host) multiplied by a "
        "Pi multiplier `s`; the real sustained run (Part C1) is a VM run with *arrivals accelerated by `s`* (the dual of a Pi `s` times slower), and waits are reported in Pi-equivalent ms (= VM ms x `s`). "
        "Delivered AUCs are *simulated delivery on replayed detections* (frozen `ML0_ncnn`@640 for ShanghaiTech, the `ML0_mps` proxy for UBnormal, locked BoT-SORT, frozen STG-NF, E00 causal S3/S4 protocol). "
        "**No number here is a Pi measurement. No model was trained.** Plan and config were frozen before any E05 evaluation (`plan.md`, `config.yaml`, hash `" + load_config()["plan_sha256"][:12] +
        "…`); post-lock changes are listed as amendments in §9.\n")
    add(f"*Generated by `edgelab.analysis.q2_p1_e05 report` (seed {SEED}, {time.strftime('%Y-%m-%d')}). Correctness rules: {n_pass}/{len(rules)} pass "
        f"({'all required rules pass' if ok_all else 'SOME REQUIRED RULES FAIL'}).*\n")

    # ------------------------------------------------------------------------------------------ 1. answer
    add("## 1. Answer\n")
    s1, u1 = c12(SHT, "ladder.cache"), c12(UB, "ladder.cache")
    s1a, u1a = c12(SHT, "ladder.cache", "all"), c12(UB, "ladder.cache", "all")
    s3, u3 = c12(SHT, "e2e.cache_vs_stock"), c12(UB, "e2e.cache_vs_stock")
    g3 = {ds: c12(ds, "e2e.guard_vs_stock") for ds in DATASETS}
    add("**Question: does the locked system (`ML0` frontend + locked tracker + Fixed-Burst + C1 detector-output reuse, with an overload guard) hold up on the full protocol, on UBnormal, on short events, "
        "on new entries and under sustained operation?**\n")
    add(f"1. **Lock (V1, ShanghaiTech) — {V(SHT, 'V1_confirmation')['verdict'].upper()}.** At matched CPU, `Cache-C1 − Fixed` = {fci(s1)} pt on `test_confirmation` (12-cell mean, 2,000-resample "
        f"paired video bootstrap) and {fci(s1a)} on all 107 clips. Net CPU saving of C1 at the same schedule (W=1): "
        + ", ".join(f"{x['saving_pct']:.0f}% at s={x['s']}" for x in c1_saving(D, SHT)) + " (§3.3).")
    add(f"2. **Replication (V2, UBnormal) — {V(UB, 'V2_replication')['verdict']}.** `Cache-C1 − Fixed` = {fci(u1)} on the 141 confirmation videos ({fci(u1a)} on all 211). "
        f"E03 already showed that on UBnormal the winning arm was `Independent-Combo`, not `Cache-C1`; the replication is therefore a *no-harm* check, and it is reported as such. "
        f"Per-cell effects are heterogeneous ({min(c['delta'] for k, c in A[UB]['contrasts']['ladder.cache']['confirmation'].items() if k in cells_ids(CELLS)):+.1f} to "
        f"{max(c['delta'] for k, c in A[UB]['contrasts']['ladder.cache']['confirmation'].items() if k in cells_ids(CELLS)):+.1f} pt): at matched *mean* CPU the unguarded `Cache-C1` queue is unstable in the UB cells with `s ≥ 2` "
        f"(peak waits up to {max(A[UB]['qmax_cache_c1'].values()) / 1000:.0f} s), which `Cache-C1+Guard` fixes (V5).")
    add(f"3. **End to end (V3).** `Cache-C1 − Stock-Baseline` (stock frontend + default tracker + Fixed-Burst at the same CPU): SHT {fci(s3)} — **{V(SHT, 'V3_end_to_end')['verdict']}**; "
        f"UB {fci(u3)} — **{V(UB, 'V3_end_to_end')['verdict']}** (UB: E01 had already shown no ML0 advantage on UBnormal).")
    add(f"4. **No oracle (V4).** The non-oracle `C1-ReTrack` keeps {V(SHT, 'V4_retrack')['retention_pct']:.0f}% of the oracle `Cache-C1 − Fixed` gain on SHT "
        f"({fci(V(SHT, 'V4_retrack')['retrack_vs_fixed'])}) — **{V(SHT, 'V4_retrack')['verdict']}**; UB {V(UB, 'V4_retrack')['retention_pct']:.0f}% "
        f"({fci(V(UB, 'V4_retrack')['retrack_vs_fixed'])}) — **{V(UB, 'V4_retrack')['verdict']}**.")
    gs, gu = V(SHT, "V5_guard"), V(UB, "V5_guard")
    add(f"5. **Guard (V5).** SHT: no-harm in {len(gs['nonoverload_cells'])} non-overload cells **{gs['no_harm']}** ({fci(gs['contrast_nonoverload'])}), overload cells (n={len(gs['overload_cells'])}): **{gs['overload']}** "
        f"({fci(gs['contrast_overload'])}). UB: no-harm **{gu['no_harm']}** ({fci(gu['contrast_nonoverload'])}), overload (n={len(gu['overload_cells'])}): **{gu['overload']}** ({fci(gu['contrast_overload'])}). "
        "No-harm cells are those where the unguarded peak wait never exceeds 400 ms (the guard is then bit-identical, E5-D2). "
        "**Cost of the guard in mild overload** (unguarded peak wait 400–2,000 ms, the guard acts but nothing would have been dropped): "
        + "; ".join(f"{'SHT' if ds == SHT else 'UB'} {z['cell']} (peak {z['peak_ms']:.0f} ms) {fci(z['c'])}" for ds in DATASETS for z in guard_zone(D, ds)) + ".")
    v6s, v6u, v7s, v7u = V(SHT, "V6_short_events"), V(UB, "V6_short_events"), V(SHT, "V7_new_entries"), V(UB, "V7_new_entries")
    add(f"6. **Short events and new entries (V6/V7, budget cells).** Short events (SHT n={v6s['n_events']}, UB n={v6u['n_events']}), recall@1 s `Cache-C1 − Fixed`: SHT {fci(v6s['cache_vs_fixed']['recall_1.0'], 1)} "
        f"→ {v6s['cache_vs_fixed']['verdict_recall']}; UB {fci(v6u['cache_vs_fixed']['recall_1.0'], 1)} → {v6u['cache_vs_fixed']['verdict_recall']}. New entries (SHT n={v7s['n_events']}, UB n={v7u['n_events']}): recall@1 s "
        f"SHT {fci(v7s['cache_vs_fixed']['recall_1.0'], 1)} → {v7s['cache_vs_fixed']['verdict_recall']}, delay p50 {fci(v7s['cache_vs_fixed']['delay_p50_ms'], 0)} ms → {v7s['cache_vs_fixed']['verdict_delay']}; "
        f"UB {fci(v7u['cache_vs_fixed']['recall_1.0'], 1)} → {v7u['cache_vs_fixed']['verdict_recall']}, delay {fci(v7u['cache_vs_fixed']['delay_p50_ms'], 0)} ms → {v7u['cache_vs_fixed']['verdict_delay']}. "
        "**Reading:** new entries are *not* hurt by C1 (both datasets). Short events are **inconclusive, not cleared**: the point estimates are negative on both datasets and the CIs are wide because only "
        f"{v6s['n_events']} / {v6u['n_events']} events are short, so the 5-pt non-inferiority margin cannot be established; on SHT short events the detection delay of `Cache-C1` is also longer than `Fixed` "
        f"({fci(A[SHT]['subgroup_contrasts']['short_event']['budget']['cache_vs_fixed']['delay_p50_ms'], 0)} ms p50), while the localised AUC on short events does not drop "
        f"({fci(A[SHT]['subgroup_contrasts']['short_event']['budget']['cache_vs_fixed']['loc_auc'])} pt SHT).")
    add(f"7. **Sustained simulation (V8) — {v8['verdict']}.** `Cache-C1+Guard`: bounded queue in "
        f"{sum(1 for r in v8['cells'] if r['bounded'])}/{len(v8['cells'])} (dataset, s) cells (SHT 2.36 h / UB 51.5 min streams). Unguarded arms unbounded in {len(v8['unguarded_unbounded'])} (arm, dataset, s) cases.")
    if vm and vm["runs"]:
        gr = {n: r for n, r in vm["runs"].items() if r["mode"] == "cache_c1_guard"}
        k1 = {n: r for n, r in vm["runs"].items() if r["mode"] == "fixed_k1"}
        ratios = [r["cpu_ms_per_video_s_pi_eq"] / r["sim"]["cost_ms_per_video_s_pi_eq"] for r in vm["runs"].values() if r["s"] == 1.0]  # s=1.0: no queueing / shedding
        add(f"8. **Real VM run (V9) — {v9['verdict']}** (latency and memory criteria). {len(vm['runs'])} runs x {next(iter(vm['runs'].values()))['frames']:,} frames in `edge-os`: with `Cache-C1+Guard` the p99 Pi-equivalent wait is "
            + ", ".join(f"{r['wait_pi_ms']['p99']:.0f} ms at s={r['s']}" for r in gr.values()) + f" and RSS grows by ≤ {max(r['rss_growth']['growth_pct'] for r in gr.values()):.1f}% from frame 2,500, whereas `Fixed-K1` "
            "collapses at s = 3.0 / 3.5 (p99 wait " + " / ".join(f"{r['wait_pi_ms']['p99'] / 1000:.0f} s" for r in k1.values() if r["s"] > 1.5) + ", " +
            f"{k1[vm_name('fixed_k1', 3.0)]['dropped_late']:,} of {k1[vm_name('fixed_k1', 3.0)]['windows_scored']:,} windows dropped late at s=3.0).")
        add(f"9. **Important caveat from the real run: the simulation under-estimates real cost by ≈ {np.mean(ratios):.2f}x** (real CPU ms per video-second / simulated in the two s = 1.0 runs, which have no queueing or shedding: {ratios[0]:.2f}x and {ratios[1]:.2f}x; busy-time capacity of `Fixed-K1` "
            f"{k1[vm_name('fixed_k1', 1.0)]['busy_fps']:.0f} fps = s* ≈ {k1[vm_name('fixed_k1', 1.0)]['busy_fps'] / 24:.1f} vs E02's 3.08). The *relative* C1 saving transfers "
            f"({100 * (1 - gr[vm_name('cache_c1_guard', 1.0)]['cpu_ms_per_video_s_vm'] / k1[vm_name('fixed_k1', 1.0)]['cpu_ms_per_video_s_vm']):.0f}% real vs "
            f"{100 * (1 - gr[vm_name('cache_c1_guard', 1.0)]['sim']['cost_ms_per_video_s_pi_eq'] / k1[vm_name('fixed_k1', 1.0)]['sim']['cost_ms_per_video_s_pi_eq']):.0f}% simulated at s=1), the absolute levels do not. "
            f"Consequence: the guard keeps the queue bounded but at s = 3.0 / 3.5 the dense `W = 1` schedule sheds {100 * gr[vm_name('cache_c1_guard', 3.0)]['shed_frame_fraction']:.0f}% / {100 * gr[vm_name('cache_c1_guard', 3.5)]['shed_frame_fraction']:.0f}% of the frames and its "
            f"delivered AUC on the looped clips falls to {gr[vm_name('cache_c1_guard', 3.0)]['auc_online']:.1f} / {gr[vm_name('cache_c1_guard', 3.5)]['auc_online']:.1f} (vs {gr[vm_name('cache_c1_guard', 1.0)]['auc_online']:.1f} at s=1; simulation "
            f"{gr[vm_name('cache_c1_guard', 3.0)]['auc_sim']:.1f} / {gr[vm_name('cache_c1_guard', 3.5)]['auc_sim']:.1f}) - i.e. *bounded latency is confirmed, accuracy under overload is not*. A sparser Fixed-Burst operating point "
            "(β ≈ 0.5) was only simulated (C2), not run on the VM; it is the natural next step.")
    else:
        add("8. **Real VM run (V9): not run.**")
    add("")

    # ------------------------------------------------------------------------------------------ 2. ladder recap
    add("## 2. The E00 → E05 ladder in one page\n")
    add("| step | question | result (all from the frozen artifacts; no re-computation here) |\n|---|---|---|")
    add("| E00 audit | Can the existing scores / protocol be reproduced? | Yes for scores, windows, normalisation, missing data; SHT is not uniformly 24 fps (rule R7 failed as pre-registered). Causal S3/S4 protocol, splits and sentinels locked. |")
    add("| E01 quality | What can the frozen frontend + scorer reach with no CPU limit? | SHT S4: ML0 `ML0_ncnn` 73.78 (76.82 with the locked tracker) vs stock@640 66.03; the P2 head (MLP) is not confirmed; UB shows no ML0 advantage (ML0 58.04 / locked tracker 61.08 vs stock 57.72). |")
    add("| E02 fixed baseline | What does an optimised fixed pipeline achieve on the Pi proxy? | K=1 is sustainable up to s* = 3.08 (SHT) / 1.66 (UB); Fixed-Burst is the only fixed schedule that never relabels an old pose as new. |")
    add("| E03 ablation | Which mechanisms pay at matched CPU? | Only C1 detector-output reuse pays (SHT Cache-Only − Fixed ≈ +2.4 pt, CPU −29 %); event-aware scheduling does not beat Fixed; joint controller loses after overhead; C2 dropped. UB: `Independent-Combo` was locked, C1 alone ≈ +0.2 [−1.0, +1.5]. |")
    add("| E04 optimisations | Do codec motion vectors or INT8 pay on top? | **Neither adopted** (MV: ≈ 0 pt at matched cost; INT8: −1.9 pt confirmation despite −22 % CPU). |")
    add(f"| **E05 confirmation** | Does the locked system survive the full protocol, UB, short events, new entries and sustained operation? | V1 {V(SHT, 'V1_confirmation')['verdict']}, V2 {V(UB, 'V2_replication')['verdict']}, V3 SHT {V(SHT, 'V3_end_to_end')['verdict']} / UB {V(UB, 'V3_end_to_end')['verdict']}, "
        f"V4 {V(SHT, 'V4_retrack')['verdict']} / {V(UB, 'V4_retrack')['verdict']}, V8 {v8['verdict']}, V9 {v9['verdict']}. |")
    add("")
    add("Locked system: `ML0_ncnn`@640 (SHT) / `ML0_mps` proxy (UB) → two-stage BoT-SORT (`c=0.20, low=0.05, buffer=30, match=0.85`) → frozen STG-NF with the E00 strict causal completion-landing protocol (hold `H = W_ref − 1`; sentinels SHT `-0.5122`, UB `-0.8273`) "
        "scheduled as Fixed-Burst with C1 level 3 (`A1 = 4`), `Cache-C1+Guard` = + hysteresis load-shedding (on > 400 ms, off < 150 ms). MV and INT8 are not part of the system.\n")

    # ------------------------------------------------------------------------------------------ 3. Part A
    add("## 3. Part A — full-protocol progression (2,000-resample paired video bootstrap, 95 % CI)\n")
    add("Cells: the 12 E03 cells (`s ∈ {2.0, 2.5, 3.0}` x CPU budget `β`) plus the native cells `s = 1.0` x `β ∈ {0.25, 0.50, 1.00}`. All cost-matched arms use the same CPU target per cell "
        "(`β x C_full(ML0-locked K=1, s)`); K=1 arms are *not* cost-matched (they run full cadence and overload for `s ≥ s*`). `*` = the CI excludes 0. Ladder steps are paired contrasts.\n")
    for ds in DATASETS:
        a = A[ds]
        add(f"### 3.{DATASETS.index(ds) + 1} {ds}\n")
        add(f"**Delivered AUC x 100 per arm** (cell mean, bootstrap CI of the mean; `dev` = point on `test_dev`). Mean realised CPU in ms CPU per video-second over the 12 cells in the last column.\n")
        rows = []
        for arm, label in LADDER_ORDER:
            ar = a["arm_summary"].get(arm)
            if not ar:
                continue
            cost = np.mean([D["run"][ds]["results"][(arm, c)]["cost"] for c in a["sets"]["primary12"] if (arm, c) in D["run"][ds]["results"]]) if arm != "Independent-Combo" else float("nan")
            rows.append([label, fnum(ar["primary12"]["dev"]["mean"]), fmean(ar["primary12"].get("confirmation")), fmean(ar["primary12"].get("all")), fmean(ar.get("native3", {}).get("confirmation")), fnum(cost, 0)])
        add(md_table(["arm", "dev (12 cells)", "confirmation (12 cells)", "all (12 cells)", "confirmation (native s=1, 3 cells)", "CPU ms/s"], rows))
        add("")
        add("**Ladder contrasts** (A − B, AUC points; 12-cell mean unless stated; confirmation / all / native cells / overload cell s=3, β=1.0):\n")
        rows = []
        names = {"ladder.frontend": "ML0-Default-K1 − Stock-Default-K1 (frontend)", "ladder.tracker": "Full-Rate − ML0-Default-K1 (locked tracker)", "ladder.budget": "Fixed − Full-Rate (cost of the budget)",
                 "ladder.cache": "**Cache-C1 − Fixed (the lock)**", "ladder.guard": "Cache-C1+Guard − Cache-C1", "e2e.cache_vs_stock": "**Cache-C1 − Stock-Baseline (end to end)**",
                 "e2e.guard_vs_stock": "Cache-C1+Guard − Stock-Baseline", "e2e.fixed_vs_stock": "Fixed − Stock-Baseline", "e2e.fixed_vs_default_fixed": "Fixed − ML0-Default-Fixed",
                 "cache.vs_fullrate": "Cache-C1 − Full-Rate (reuse vs full cadence)", "guard.fixed": "Fixed+Guard − Fixed", "retrack.vs_fixed": "C1-ReTrack − Fixed (non-oracle)",
                 "retrack.vs_oracle": "C1-ReTrack − Cache-C1 (price of identity)", "retrack_guard.vs_fixed": "C1-ReTrack+Guard − Fixed", "ref.guard_vs_independent": "Cache-C1+Guard − Independent-Combo",
                 "ref.cache_vs_independent": "Cache-C1 − Independent-Combo"}
        for k, lab in names.items():
            cs = a["contrasts"].get(k)
            if not cs:
                continue
            rows.append([lab, fci(cs["confirmation"].get("primary12")), fci(cs["all"].get("primary12")), fci(cs["confirmation"].get("native3")), fci(cs["confirmation"].get(cell_id(*OVERLOAD_CELL)))])
        add(md_table(["contrast", "confirmation, 12 cells", "all clips, 12 cells", "confirmation, native 3 cells", "confirmation, s=3.0 β=1.0"], rows))
        add("")
        add(f"**`Cache-C1 − Fixed` per cell (confirmation)**\n")
        rows = []
        for c in cells_ids(CELLS):
            cc = a["contrasts"]["ladder.cache"]["confirmation"].get(c)
            ca = a["contrasts"]["ladder.cache"]["all"].get(c)
            gg = a["contrasts"]["ladder.guard"]["confirmation"].get(c)
            rows.append([c, fnum(D["run"][ds]["w_ref"][c], 1), fnum(a["auc"]["Fixed"][c]["confirmation"]), fnum(a["auc"]["Cache-C1"][c]["confirmation"]), fci(cc), fci(ca), fnum(a["qmax_cache_c1"][c], 0), fci(gg)])
        add(md_table(["cell", "Fixed period W", "Fixed", "Cache-C1", "Cache-C1 − Fixed (conf)", "(all)", "Cache-C1 peak wait ms", "Guard − Cache-C1 (conf)"], rows))
        add("")
        add("**Event metrics, 12-cell mean (confirmation clips; recall in % of events, delay of detected events in ms, bootstrap 95 % CI)**\n")
        rows = []
        for arm in ("Stock-Default-K1", "ML0-Default-K1", "Full-Rate", "Stock-Baseline", "Fixed", "Cache-C1", "Cache-C1+Guard", "C1-ReTrack"):
            e = a["events_part_a"]["confirmation"].get(arm, {}).get("primary12")
            if e:
                rows.append([arm, e["n_events"], fpt(e["recall_0.5"]), fpt(e["recall_1.0"]), fpt(e["recall_2.0"]), fpt(e["missed"]), fpt(e["delay_p50_ms"], 0), fnum(e["false_alerts_per_normal_hour"], 1)])
        add(md_table(["arm", "events", "recall@0.5 s", "recall@1 s", "recall@2 s", "missed %", "delay p50 ms", "false alerts / normal h"], rows))
        add("")
        ec = a["events_part_a"]["confirmation"]["_contrasts"]
        rows = []
        for k in ("ladder.cache", "e2e.cache_vs_stock", "ladder.guard", "retrack.vs_fixed", "cache.vs_fullrate"):
            if k in ec:
                m = ec[k]["primary12"]
                rows.append([names[k], fci(m["recall_0.5"], 1), fci(m["recall_1.0"], 1), fci(m["recall_2.0"], 1), fci(m["missed"], 1), fci(m["delay_p50_ms"], 0)])
        add(md_table(["event contrast (confirmation)", "Δ recall@0.5 s (pt)", "Δ recall@1 s (pt)", "Δ recall@2 s (pt)", "Δ missed (pt)", "Δ delay p50 (ms)"], rows))
        add("")
        add(f"![{ds} ladder](fig_{ds}_ladder.png)\n")

    add("### 3.3 CPU at the same schedule (W = 1, β = 1.0 cells where `Cache-C1` runs at W = 1)\n")
    rows = []
    for ds in DATASETS:
        for x in c1_saving(D, ds):
            rows.append([ds, x["s"], fnum(x["full_rate"], 0), fnum(x["cache_c1"], 0), f"{x['saving_pct']:.1f} %"])
    add(md_table(["dataset", "s", "Full-Rate CPU ms/s", "Cache-C1 CPU ms/s", "saving"], rows))
    add("")
    # ------------------------------------------------------------------------------------------ 4. Part B
    add("## 4. Part B — stratified subgroup analysis (all ground-truth events)\n")
    add("Partitions (each event is in exactly one level per stratum, rule E5-D9): **duration** short ≤ 2.5 s (60 frames SHT / 75 UB) vs sustained; **new-person entry**: a track (≥ 12 poses, first frame ≥ 3) of the K=1 locked "
        "tracker starts within ±1 s of the event onset; **motion**: p90 of the E03 ROI motion signal over the first second ≥ the E03 `test_dev` q85 threshold; **crowd**: median number of simultaneous K=1 tracks over the first second ≥ 4. "
        "Metrics per level: event recall @0.5/1/2 s, missed rate, detection delay p50/p90 (detected events) and the **localised event-window AUC** on `[s − 2 s, e + 2 s]` (delivered series). "
        "Cell group *budget* = `s=2.0` x `β ∈ {0.33, 0.50, 0.75}` (mean of per-cell values); *overload* = `s=3.0, β=1.0`; *native* = `s=1.0, β=1.0`. Clip-level paired bootstrap (all clips), 2,000 resamples; no multiplicity correction.\n")
    for ds in DATASETS:
        st, S = D["st"][ds], A[ds]["subgroups"]
        add(f"### 4.{DATASETS.index(ds) + 1} {ds} ({st['n_events']} events: short {st['counts']['short']}, new entry {st['counts']['entry']}, fast {st['counts']['fast']}, crowd {st['counts']['crowd']})\n")
        add("**Budget cells (s=2.0, β ∈ {0.33, 0.50, 0.75}), per level and arm**\n")
        rows = []
        for lv, title in LEVEL_TITLES.items():
            for arm in SUB_ORDER:
                m = S[lv]["budget"][arm]
                rows.append([title if arm == SUB_ORDER[0] else "", m["n_events"] if arm == SUB_ORDER[0] else "", arm, fpt(m["recall_0.5"]), fpt(m["recall_1.0"]), fpt(m["recall_2.0"]), fpt(m["missed"]),
                             fpt(m["delay_p50_ms"], 0), fpt(m["delay_p90_ms"], 0), fpt(m["loc_auc"])])
        add(md_table(["level", "n", "arm", "recall@0.5 s %", "recall@1 s %", "recall@2 s %", "missed %", "delay p50 ms", "delay p90 ms", "localised AUC"], rows))
        add("")
        for g, gt in (("budget", "budget cells"), ("overload", "overload cell s=3.0 β=1.0"), ("native", "native cell s=1.0 β=1.0")):
            add(f"**Paired subgroup contrasts, {gt}** (Δ recall@1 s in pt | Δ delay p50 in ms | Δ localised AUC in pt)\n")
            rows = []
            for lv, title in LEVEL_TITLES.items():
                row = [title, S[lv][g]["Fixed"]["n_events"]]
                for cn in ("cache_vs_fixed", "guard_vs_fixed", "retrack_vs_fixed", "cache_vs_fullrate"):
                    c = A[ds]["subgroup_contrasts"][lv][g][cn]
                    row.append(f"{fci(c['recall_1.0'], 1)} \\| {fci(c['delay_p50_ms'], 0)} \\| {fci(c['loc_auc'], 2)}")
                rows.append(row)
            add(md_table(["level", "n", "Cache-C1 − Fixed", "Cache-C1+Guard − Fixed", "C1-ReTrack − Fixed", "Cache-C1 − Full-Rate"], rows))
            add("")
        if st.get("has_released") and "entry_agreement_released_pct" in st:
            add(f"Entry-label agreement with the released-pose tracks: {st['entry_agreement_released_pct']:.1f} % of events (E5-D9 partitions exact).\n")
        else:
            add("No released-pose tracks exist for this dataset; the new-entry label comes from the K=1 locked tracker only (E5-D9 partitions exact).\n")
    add("**Pre-registered subgroup verdicts (budget cells):**\n")
    rows = []
    for ds in DATASETS:
        for k, lab in (("V6_short_events", "V6 short events"), ("V7_new_entries", "V7 new entries")):
            v = A[ds]["verdicts"][k]
            for cn in ("cache_vs_fixed", "guard_vs_fixed"):
                rows.append([ds, lab, cn.replace("_vs_", " − ").replace("cache", "Cache-C1").replace("guard", "Cache-C1+Guard").replace("fixed", "Fixed"), v["n_events"], fci(v[cn]["recall_1.0"], 1), v[cn]["verdict_recall"],
                             fci(v[cn]["delay_p50_ms"], 0), v[cn]["verdict_delay"] if k == "V7_new_entries" else "–"])
    add(md_table(["dataset", "verdict", "contrast", "events", "Δ recall@1 s (pt)", "recall rule (lower > −5)", "Δ delay p50 (ms)", "delay rule (upper < +250)"], rows))
    add("")

    # ------------------------------------------------------------------------------------------ 5. C2
    add("## 5. Part C2 — long-horizon simulation (queue carried across loops)\n")
    add(f"SHT: all 107 clips concatenated ({D['sus']['ShanghaiTech_frames']:,} frames = {D['sus']['ShanghaiTech_minutes_one_pass']:.1f} min) looped **5x = {5 * D['sus']['ShanghaiTech_minutes_one_pass'] / 60:.2f} h**; "
        f"UB: all clips once ({D['sus']['UBnormal_frames']:,} frames = {D['sus']['UBnormal_minutes_one_pass']:.1f} min; see amendment A0). Same simulator and service pools as E03 with fresh service-time draws per loop and the FIFO backlog carried across loop boundaries. "
        "Arms: `Fixed-K1` (W=1, no reuse), `Cache-C1` (W=1), `Cache-C1+Guard` (W=1) and `Fixed` / `Cache-C1+Guard` at the `β = 0.50` period of the cell. `bounded` = max wait ≤ 2,000 ms, p99 ≤ 2,000 ms, last-third − first-third mean wait ≤ 100 ms, modelled queue memory ≤ 200 MB (V8, evaluated for `Cache-C1+Guard` at W=1).\n")
    for ds in DATASETS:
        add(f"**{ds}**\n")
        rows = []
        for s in SUSTAINED_S:
            for arm in SUSTAINED_ARMS:
                r = D["sus"]["cells"][(ds, s, arm)]
                b = r["bounded"]
                rows.append([s, arm, fnum(r["period"], 1), f"{r['minutes']:.0f}", fnum(r["wait_quantiles_ms"][0.5], 0), fnum(r["wait_quantiles_ms"][0.99], 0), fnum(r["wait_max_ms"], 0), fnum(b["drift_ms"], 0),
                             fnum(b["queue_mem_mb"], 0), "yes" if b["bounded"] else "**NO**", fnum(np.mean(r["auc_per_loop"])), fnum(100 * r["shed_frame_fraction"], 1), fnum(r["cpu_ms_per_video_s"], 0)])
        add(md_table(["s", "arm", "period W", "minutes", "wait p50 ms", "p99 ms", "max ms", "drift (last − first third) ms", "queue mem MB", "bounded", "AUC / loop", "shed %", "CPU ms/s"], rows))
        add("")
    add("![C2 queue traces](fig_c2_queue_traces.png)\n")
    add(f"**V8 verdict: {v8['verdict']}.** Unguarded arms that violate the bounded rule: " +
        ("; ".join(f"{x['arm']} {x['dataset']} s={x['s']} (max {x['max_ms']:.0f} ms)" for x in v8["unguarded_unbounded"]) if v8["unguarded_unbounded"] else "none") + ".\n")

    # ------------------------------------------------------------------------------------------ 6. C1
    add("## 6. Part C1 — real sustained run in the `edge-os` VM\n")
    if not vm or not vm["runs"]:
        add("Not run.\n")
    else:
        runs = vm["runs"]
        add("Input: the 12 SHT timing clips re-encoded to H.264 (x264 `veryfast`, CRF 23, GOP 48) and looped 4x (**11,520 frames per run**). Pipeline (all real, 4 threads, pinned): `cv2` decode → pixel C1 check → NCNN FP16 `ML0` "
            "(only when C1 does not hold; detection forced every Δ=12 in-run frames) → two-stage BoT-SORT (persistent within a run, fed with the *reused* detections on C1 frames: non-oracle) → window buffers → PyTorch STG-NF → "
            "online `CausalScoreDeliverer`. Frames arrive on a wall-clock schedule at `s x 24 fps`.\n")
        rows = []
        for n, r in runs.items():
            rows.append([n, r["frames"], fnum(r["wall_s"], 0), fnum(r["offered_fps"], 0), fnum(r["achieved_fps"], 1), fnum(r["busy_fps"], 1), fnum(r["svc_mean_ms"], 2), fnum(r["svc_p95_ms"], 2),
                         fnum(r["cpu_ms_per_video_s_vm"], 0), fnum(r["cpu_ms_per_video_s_pi_eq"], 0), fnum(100 * r["c1_hit_rate_of_inrun"], 1), fnum(100 * r["observed_fraction"], 1)])
        add(md_table(["run", "frames", "wall s", "offered fps (arrival)", "achieved fps", "busy-time capacity fps", "service mean ms", "service p95 ms", "CPU ms/video-s (VM)", "CPU ms/video-s (Pi-eq = x s)",
                      "C1 hit rate of in-run frames %", "frames observed %"], rows))
        add("")
        rows = []
        for n, r in runs.items():
            w = r["wait_pi_ms"]
            rm = {int(k): v for k, v in r["rss_marks_mb"].items()}
            rows.append([n, fnum(w["p50"], 0), fnum(w["p95"], 0), fnum(w["p99"], 0), fnum(w["max"], 0), fnum(r["drift_pi_ms"], 0), "/".join(fnum(rm.get(k, float("nan")), 0) for k in (0, 2500, 5000, 10000)),
                         fnum(r["peak_rss_mb"], 0), fnum(r["rss_growth"]["growth_pct"], 2), fnum(100 * r["shed_frame_fraction"], 1), len(r["guard_transitions"]), fnum(r["auc_online"]), fnum(r["auc_sim"])])
        add(md_table(["run", "wait p50 (Pi-eq ms)", "p95", "p99", "max", "drift last − first third (ms)", "RSS MB @0 / 2,500 / 5,000 / 10,000", "peak RSS MB", "RSS growth 2,500 → end %", "shed %", "guard transitions",
                      "delivered AUC (real run)", "AUC (simulation of the same stream)"], rows))
        add("")
        rows = []
        for n, r in runs.items():
            t = r["thermal_proxy"]
            rows.append([n, t["sensor"] or "none exposed by the VM", f"{100 * t['steal_fraction']:.3f}", fnum(t["svc_drift_pct_first_to_last_pass"], 1), fnum(t["cpu_drift_pct_first_to_last_pass"], 1), fnum(t["svc_drift_pct_first_to_last_block"], 1), cooling_txt(t),
                         fnum(max(b for b in t["block_loadavg"] if b is not None) if t["block_loadavg"] else float("nan"), 2), fnum(r["host_load"]["host_outside_vm_pct"], 1), "no" if not r["host_load"]["excluded"] else "**yes**",
                         r["online_equals_offline"], f"{r['dropped_late']}/{r['dropped_end']}"])
        add(md_table(["run", "thermal sensor", "CPU steal %", "service drift pass 1 → pass 4 % (same content)", "CPU drift pass 1 → pass 4 %", "service drift first → last 1,000-frame block % (content differs)", "cooling_device state start → end", "max loadavg", "host CPU outside VM %", "excluded?",
                      "online == offline delivery", "windows dropped (late / end)"], rows))
        add("")
        add("The VM exposes no temperature sensor, so *thermal throttling cannot be measured*: thermal / process stability is reported through proxies (service-time drift across 1,000-frame blocks, `/proc/stat` steal, loadavg, `cooling_device*` states). "
            "A physical Pi 5 run is required for a real thermal statement.\n")
        add("![VM traces](fig_vm_traces.png)\n")
        add("**Real run vs simulation of the same stream.** The simulation (E02 pools x s) is evaluated on the same clip order and policy; differences in AUC, wait and C1 hit rate are in the tables above and `summary.json` (`vm.*.sim`).\n")
        rows = []
        for n, r in runs.items():
            sm = r["sim"]
            rows.append([n, fnum(r["wait_pi_ms"]["mean"], 0), fnum(sm["wait_mean_ms"], 0), fnum(r["wait_pi_ms"]["p99"], 0), fnum(sm["wait_p99_ms"], 0), fnum(r["wait_pi_ms"]["max"], 0), fnum(sm["wait_max_ms"], 0),
                         fnum(100 * r["c1_hit_rate_of_inrun"], 1), fnum(100 * sm["c1_reuse_fraction_of_inrun"], 1), fnum(r["cpu_ms_per_video_s_pi_eq"], 0), fnum(sm["cost_ms_per_video_s_pi_eq"], 0), fnum(r["cpu_ms_per_video_s_pi_eq"] / sm["cost_ms_per_video_s_pi_eq"], 2)])
        add(md_table(["run", "real wait mean", "sim wait mean", "real p99", "sim p99", "real max", "sim max", "real C1 hit %", "sim C1 reuse %", "real CPU ms/s (Pi-eq)", "sim CPU ms/s", "real / sim CPU"], rows))
        add("")
        add(f"**V9 verdict: {v9['verdict']}** (`Cache-C1+Guard`: ≥ 10,000 frames, RSS growth from frame 2,500 ≤ 5 %, p99 Pi-equivalent wait ≤ 2,000 ms, run not excluded for host load).\n")
        rows = [[n, v["s"], v["frames"], fnum(v["rss_growth_pct"], 2), fnum(v["p99_wait_pi_ms"], 0), "pass" if v["pass"] else "**FAIL**"] for n, v in v9["runs"].items()]
        add(md_table(["run", "s", "frames", "RSS growth %", "p99 wait Pi-eq ms", "V9"], rows))
        add("")

    # ------------------------------------------------------------------------------------------ 7. verdicts
    add("## 7. Pre-registered verdicts\n")
    rows = []
    for ds in DATASETS:
        for k, rule in ((("V1_confirmation", "V1 lock: Cache-C1 − Fixed, conf, 12-cell mean, CI lower > 0"), ("V1_all", "V1 (all clips)")) if ds == SHT else
                        (("V2_replication", "V2 replication: same statistic on UB, CI lower > 0"),)) + (("V3_end_to_end", "V3 Cache-C1 − Stock-Baseline lower > 0"),):
            v = A[ds]["verdicts"][k]
            rows.append([ds, k, rule, fci(v["contrast"]), v["verdict"]])
        v = A[ds]["verdicts"]["V4_retrack"]
        rows.append([ds, "V4_retrack", "ReTrack − Fixed lower > 0 and ≥ 50 % of oracle gain", f"{fci(v['retrack_vs_fixed'])}; retention {v['retention_pct']:.0f} %", v["verdict"]])
        v = A[ds]["verdicts"]["V5_guard"]
        rows.append([ds, "V5_guard", "guard: no harm (lower > −0.25) / overload helps (lower > 0)", f"no-harm {fci(v['contrast_nonoverload'])}; overload {fci(v['contrast_overload'])}", f"{v['no_harm']} / {v['overload']}"])
        for k in ("V6_short_events", "V7_new_entries"):
            v = A[ds]["verdicts"][k]
            rows.append([ds, k, "recall@1 s lower > −5 pt" + (" and delay p50 upper < +250 ms" if k == "V7_new_entries" else ""), fci(v["cache_vs_fixed"]["recall_1.0"], 1),
                         v["cache_vs_fixed"]["verdict_recall"] + ("; " + v["cache_vs_fixed"]["verdict_delay"] if k == "V7_new_entries" else "")])
    rows.append(["both", "V8_long_horizon", "Guard bounded in every (dataset, s)", f"{sum(1 for r in v8['cells'] if r['bounded'])}/{len(v8['cells'])} cells", v8["verdict"]])
    rows.append([SHT, "V9_vm_sustained", "VM run: ≥ 10,000 frames, RSS ≤ +5 %, p99 ≤ 2,000 ms", f"{sum(1 for v in v9['runs'].values() if v['pass'])}/{len(v9['runs'])} runs", v9["verdict"]])
    add(md_table(["dataset", "id", "rule", "statistic", "verdict"], rows))
    add("")
    add("**Correctness rules**\n")
    desc = {"E5-D1": "E03 replay of Fixed and Cache-C1 bit-for-bit", "E5-D1b": "K=1 ideal series vs E01 anchors (diagnostic, tolerance 0.75 pt, amendment A1)", "E5-D2": "guard identity when never triggered; guarded wait ≤ unguarded max",
            "E5-D3": "table builder equals E03 on 3 clips / dataset", "E5-D4": "ReTrack causality (source ≤ A1 old, never reused)", "E5-D5": "queue invariants for guarded arms", "E5-D6": "cost ratio ≤ 1.005 (under/over flagged)",
            "E5-D7": "real-run integrity (≥ 10,000 frames, monotone arrivals, finite scores, online == offline)", "E5-D8": "bootstrap point = pooled AUC, 2,000 resamples", "E5-D9": "subgroup partitions and event totals"}
    add(md_table(["rule", "description", "result"], [[r["id"], desc.get(r["id"], ""), "PASS" if r["pass"] else "**FAIL**"] for r in rules]))
    add("")
    st_counts = {ds: {k: sum(1 for (a_, c_), r in D["run"][ds]["results"].items() if r.get("status") == k) for k in ("ok", "under", "over")} for ds in DATASETS}
    add(f"E5-D6 status counts (arm x cell): " + "; ".join(f"{ds}: {v}" for ds, v in st_counts.items()) + ". `under` = the Fixed period reached its floor (W = 1 / all frames) so the arm cannot spend the whole budget, mostly β = 1.0 and the UB cells; these are flagged, not hidden. No arm exceeds its target.\n")

    # ------------------------------------------------------------------------------------------ 8. limitations
    add("## 8. Limitations (honest)\n")
    lim = [
        "**No physical Pi 5.** `s` is a scalar slowdown on VM service times (C2 / Part A) or on the arrival rate (C1). Memory-bandwidth contention, cache behaviour, DVFS and **thermal throttling of a real Pi are not modelled or measured**; a Pi run is still required before any deployment claim.",
        "**UBnormal has no NCNN frontend.** The `ML0_mps` proxy supplies detections, the NCNN `ML0` timing model supplies costs (E02 convention). The UB result is a replication of the *scheduling* claim, not of the frontend gain (E01: no ML0 advantage on UB).",
        "**C1 oracle identity in the headline arm.** `Cache-C1` re-scores genuine-track windows with source poses (identity is an oracle). `C1-ReTrack` and the real VM run feed the *reused* detections to the tracker (non-oracle); V4 reports how much of the gain survives. Both are cost-matched in the same way.",
        "**Detection-event recall is low for every arm** (scorer-limited, E03: ≈ 5–11 % at 1 s on SHT), so subgroup recall differences rest on few events per stratum (see `n`); no multiplicity correction is applied to the ≈ 50 subgroup contrasts, and the subgroup verdicts are non-inferiority checks against a 5-pt margin, not detection of gains.",
        "**C1 staleness is a frame-difference heuristic** (ROI mean-abs-diff + exterior p95 on 24x40 luma thumbnails): it cannot see a person who enters a region that was empty at the source frame and is not modelled by the thumbnail grid. The new-entry stratum (V7) is the direct test of this.",
        "**The real run is a re-encode** of 12 SHT clips (x264, our recipe), looped 4x (the loops repeat the same content, so the C1 hit rate and score distribution are those of those 12 clips, not of a camera stream); thresholds were calibrated on the original JPEG frames. The Pi's hardware decoder / camera path is not exercised.",
        "**Run tables restart the tracker every Δ = 12 in-run frames** (E3-D2, −0.45 pt common handicap on SHT); the real pipeline keeps one tracker per run. The K=1 zero-delay AUCs therefore sit 0.05–0.56 pt from the E01 anchors (amendment A1).",
        "**The simulation under-estimates the real pipeline's cost by ≈ 1.3–1.4x** in the VM run (§6): the E02 service pools were measured for the E02 harness, the E05 real pipeline (H.264 decode → C1 check → NCNN → BoT-SORT → STG-NF → online deliverer in Python, 4 threads, host shared) is slower. Sustainable-`s` limits taken from the simulation (SHT s* = 3.08) should be divided by ≈ 1.4 until a Pi run; relative effects (C1 saving, guard behaviour) transfer, absolute capacities do not.",
        "**The real run only tests the dense W = 1 schedule** (and C1 at W = 1); the cost-matched Fixed-Burst schedules of Part A were simulated, not run on the VM.",
        "**Cost matching uses mean CPU.** Peak behaviour is handled by the queue simulation, not by the budget; at β = 1.0 and on UB several arms cannot spend the budget (`under`).",
        "**UB K=1 is overloaded already at s = 2** (s* = 1.66); UB budget conclusions at s ≥ 2 mostly concern the schedule's graceful degradation, not accuracy at full cadence.",
        "**One seed / one random service-time draw** per simulation cell (E02 pools, seed 7); the bootstraps resample videos, not service-time draws. C2's per-loop fresh draws give 5 independent draws on SHT only.",
    ]
    for i, t in enumerate(lim, 1):
        add(f"{i}. {t}")
    add("")

    # ------------------------------------------------------------------------------------------ 9. amendments / reproduce
    add("## 9. Amendments, deviations and reproduction\n")
    add("* **A0** — The task text states UB = 105,711 frames = 58.7 min; the locked UB test labels give **92,640 frames = 51.5 min** at 30 fps. All UB stream lengths use the measured value. SHT (40,791 frames = 28.3 min) matches.")
    add("* **A1** — Diagnostic E5-D1b (not a section-8 rule) was given a 0.75 pt tolerance after the first run showed 0.05–0.56 pt offsets to E01's S4 AUCs, explained by E03's documented run-table handicap (E3-D2). No verdict depends on it; the bit-for-bit E5-D1 replay of E03 passes.")
    add("* **A2** — E5-D3 compares the genuine window of every row, not the pool arrays (E03's pool also holds the C1-substituted variants).")
    add("* **A3** — The VM clips are the first 240 frames of the 12 SHT clips; the real-run AUC and the same-stream simulation are scored on those prefixes (causal series, so the prefix is unaffected by later frames). Pass-over-pass drift (same content) was added beside the first-vs-last-block drift, which is confounded by clip content. No verdict rule changed.")
    add("* The E03 simulator gained two backward-compatible options (`init_free` for carried queue, `c1_retrack` cost accounting); E5-D1 shows the E03 results are reproduced bit-for-bit.")
    add("\nReproduce (`experiments/`): `uv run --extra pose python -m edgelab.analysis.q2_p1_e05 {lock,tables,run,strata,analyze,sustained,vm,report}`; tests: `uv run pytest tests/test_q2_p1_e05.py tests/test_q2_p1_e0*.py`. "
        "Files: `plan.md`, `config.yaml`, `results.csv` (per video x arm x cell), `events.csv` (per event x arm x cell, with strata), `sustained.csv`, `decisions.jsonl`, `summary.json`, figures.\n")
    return "\n".join(L) + "\n"



def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["lock", "tables", "run", "strata", "analyze", "sustained", "vm", "vm-analyze", "report", "all"])
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--only", nargs="*", default=None, help="vm: only runs whose name contains one of these strings")
    a = ap.parse_args()
    if a.stage == "lock":
        stage_lock()
        return
    ctx = e01.build_ctx(a.procs)
    st = a.stage
    if st == "tables":
        for ds in DATASETS:
            for v in VARIANTS:
                stage_tables(ds, ctx, v, a.procs, force=a.force)
            check_tables(ds, ctx, a.procs)
    if st in ("run", "all"):
        for ds in DATASETS:
            stage_run(ctx, ds, a.procs, force=a.force)
    if st in ("strata", "all"):
        for ds in DATASETS:
            stage_strata(ctx, ds, a.procs, force=a.force)
    if st in ("analyze", "all"):
        stage_analyze(ctx, a.procs, force=a.force)
    if st in ("sustained", "all"):
        stage_sustained(ctx, force=a.force)
    if st == "vm":
        stage_vm(ctx, force=a.force, only=a.only)
    if st in ("vm-analyze", "report", "all"):
        if st == "vm-analyze" or (st == "report" and (VM_DIR / "out" / "manifest.jsonl").exists() and not (WORK_DIR / "vm_summary.json").exists()):
            stage_vm_analyze(ctx)
    if st in ("report", "all"):
        stage_report(ctx)


if __name__ == "__main__":
    main()
