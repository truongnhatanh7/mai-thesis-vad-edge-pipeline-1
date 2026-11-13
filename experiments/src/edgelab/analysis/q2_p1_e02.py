"""Q2-P1-E02 (Fixed baseline): what does an optimised FIXED pipeline achieve on the (proxy) Pi?

    uv run --extra pose python -m edgelab.analysis.q2_p1_e02 lock     # hash everything -> Q2-P1-E02_fixed_baseline/config.yaml
    uv run --extra pose python -m edgelab.analysis.q2_p1_e02 vm       # stage benchmark inside the `edge-os` Lima VM (host must be idle)
    uv run --extra pose python -m edgelab.analysis.q2_p1_e02 run      # schedules x scenarios on all clips -> results.csv, decisions.jsonl, summary.json
    uv run --extra pose python -m edgelab.analysis.q2_p1_e02 report   # summary.json -> report.md

No training. Builds on Q2-P1-E00 (causal S3/S4 protocol, splits, window engine) and Q2-P1-E01 (locked frontend / tracker): this module
*imports* their engines. New here:

  * `q2_p1_e02_vm`  : decode / pose / tracking / scoring / end-to-end profile inside the ARM64 VM, per stage and thread count
  * schedules       : fixed stride K (+ zero-order-hold / constant-velocity propagation, flagged as violating "never relabel an old
                      pose") and fixed window bursts of period W (intact, 100 % genuine windows)
  * true causality  : a FIFO single-worker queue charges collection, queueing and inference to the score's delivery frame
                      `ceil(t_done / dt)`; late work (> 2.0 s) is discarded; unsustainable configurations are rejected
  * sustainable workload under Pi multipliers s and CPU budgets b, from the measured VM service-time pools

Pure functions (schedules, propagation, queue, delivery, service model, selection) are unit-tested without data in
tests/test_q2_p1_e02.py; the I/O drivers need ~/thesis-data, the STG-NF checkpoints and (for `vm`) the running `edge-os` VM.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import multiprocessing as mp
import os
import pickle
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from edgelab.analysis import q2_p1_e00 as e00
from edgelab.analysis import q2_p1_e01 as e01

DATASETS = e00.DATASETS
SEG_LENS = e00.SEG_LENS
FPS = e00.FPS
SEED = e00.SEED
POOL_SEED = 7  # service-time sampling (common random numbers across schedules and scenarios)
N_BOOT = 1000
EXP_DIR = e00.EXPERIMENTS_ROOT / "Q2-P1-E02_fixed_baseline"
E00_DIR, E01_DIR = e00.EXP_DIR, e01.EXP_DIR
DATA_ROOT = e00.DATA_ROOT
WORK_DIR = DATA_ROOT / "q2-p1-e02"
VM_OUT = DATA_ROOT / "vm-exchange/q2-p1-e02/out"
MACHINE_TAG = "edge-os-arm64-vm (comparative proxy; physical Pi 5 pending D8/D10)"

CAP_MS = 2000.0  # the plan's 1-2 s rolling buffer (upper end)
CORES = 4  # Pi 5: 4 x Cortex-A76; the VM has 4 vCPUs
THREADS = (1, 2, 3, 4)
S_GRID = (1.0, 1.5, 2.0, 2.5, 3.0)  # Pi-5 Cortex-A76 multiplier on every VM service time
B_GRID = (1.0, 0.75, 0.50, 0.33, 0.25)  # CPU budget as a fraction of the 4-core device
STRIDES = (1, 2, 3, 4)
FILLS = ("none", "zoh", "cv")
BURST_W = {"ShanghaiTech": (1, 2, 4, 6, 8, 12, 24, 36, 48, 72, 96), "UBnormal": (1, 2, 4, 6, 8, 12, 16, 24, 32, 48, 64)}
HEADLINE = ("T4_s2.0_b1.00", "T4_s3.0_b1.00", "T4_s2.0_b0.50")  # B1 selection scenarios
TIE_BAND = 0.10

# frontends (E01 arm names; UBnormal uses MPS proxies, there is no UB NCNN cache) x trackers
FRONTENDS = {"ShanghaiTech": ("ML0_ncnn", "stock", "stock_ncnn800"), "UBnormal": ("ML0_mps", "stock")}
PRIMARY = {"ShanghaiTech": "ML0_ncnn", "UBnormal": "ML0_mps"}
POSE_TIMING_MODEL = {  # VM pose cell whose service times stand in for a frontend (same architecture / FLOPs; weights do not matter for time)
    ("ShanghaiTech", "ML0_ncnn"): "ML0-640", ("ShanghaiTech", "stock"): "stock-640", ("ShanghaiTech", "stock_ncnn800"): "stock-800",
    ("UBnormal", "ML0_mps"): "ML0-640-ub", ("UBnormal", "stock"): "ML0-640-ub"}
TRACKERS = {"locked": e01.TrackerCell(0.20, 0.05, 30, 0.85), "default": e01.DEFAULT_CELL}
GATE_FLOOR = 0.05


def plain(obj):
    return e00.to_plain(obj)


def now() -> str:
    return time.strftime("%H:%M:%S")


# =====================================================================================================================
# 1. Fixed schedules (pure)
# =====================================================================================================================
@dataclass(frozen=True)
class Schedule:
    """family 'stride': detector on frames f % k == 0 (duty 1/k), skipped frames handled by `fill`;
    family 'burst' : window period k = W, windows of L consecutive observed frames (duty min(1, L/W))."""

    family: str
    k: int
    fill: str = "none"

    @property
    def id(self) -> str:
        if self.family == "stride":
            return "stride1" if self.k == 1 else f"stride{self.k}-{self.fill}"
        return f"burst{self.k}"

    @property
    def alias_of(self) -> str | None:
        return "stride1" if self.family == "burst" and self.k == 1 else None

    @property
    def refresh(self) -> int:
        """frames between two score updates; the score is held for refresh - 1 frames (a-priori rule, `H = 0` is the sensitivity)."""
        return self.k

    @property
    def hold(self) -> int:
        return self.k - 1

    @property
    def propagated(self) -> bool:
        return self.family == "stride" and self.k > 1 and self.fill in ("zoh", "cv")

    @property
    def plan_compliant(self) -> bool:
        """True when every pose in every scored window is a genuine observation of its own frame."""
        return not self.propagated

    def duty(self, seg_len: int) -> float:
        return 1.0 / self.k if self.family == "stride" else min(1.0, seg_len / self.k)

    def observed(self, n_frames: int, seg_len: int) -> np.ndarray:
        """Frames on which the detector is requested. Nothing else is ever read from a detector cache."""
        t = np.arange(n_frames)
        if self.family == "stride":
            return (t % self.k) == 0
        if self.k <= seg_len:
            return np.ones(n_frames, dtype=bool)
        return (t % self.k) < seg_len

    def burst_ids(self, n_frames: int, seg_len: int) -> np.ndarray:
        """Tracker generation per frame: a burst schedule with a gap re-creates the tracker at every burst start."""
        t = np.arange(n_frames)
        return (t // self.k) if (self.family == "burst" and self.k > seg_len) else np.zeros(n_frames, dtype=np.int64)

    def keep_windows(self, last_frame: np.ndarray, seg_len: int) -> np.ndarray:
        """Windows the schedule scores: those completing on its scoring grid."""
        last = np.asarray(last_frame)
        if self.family == "stride":
            return (last % self.k) == 0
        if self.k <= seg_len:
            return (last >= seg_len - 1) & (((last - (seg_len - 1)) % self.k) == 0)
        return (last % self.k) == (seg_len - 1)

    @property
    def trackset(self) -> tuple[str, int]:
        return ("stride", self.k) if self.family == "stride" else ("burst", self.k)


def trackset_key(s: Schedule, seg_len: int) -> tuple[str, int]:
    """Schedules sharing one tracker replay: stride K; bursts with W <= L observe every frame (== stride 1); bursts with a gap reset."""
    if s.family == "stride":
        return ("stride", s.k)
    return ("stride", 1) if s.k <= seg_len else ("burst", s.k)


def schedule_grid(dataset: str) -> list[Schedule]:
    out = [Schedule("stride", 1)]
    for k in STRIDES[1:]:
        out += [Schedule("stride", k, fill) for fill in FILLS]
    out += [Schedule("burst", w) for w in BURST_W[dataset]]
    return out


def propagate_track(frames: np.ndarray, keypoints: np.ndarray, k: int, mode: str, n_frames: int | None = None
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Causal fill of the K-1 frames after every observation of ONE track.

    mode 'zoh': the observed pose is repeated; 'cv': constant velocity of the keypoints (x, y) estimated from the previous observation
    (only if it is at most K frames earlier, otherwise zero-order hold); the confidence channel is held. Frames at or after the next
    observation, and frames beyond `n_frames`, get nothing. Returns (frames, keypoints, genuine) sorted by frame; `genuine` marks real
    observations. mode 'none' (or k == 1) returns the track unchanged.
    """
    frames = np.asarray(frames, dtype=np.int64)
    kp = np.asarray(keypoints, dtype=np.float64)
    genuine = np.ones(len(frames), dtype=bool)
    if mode == "none" or k <= 1 or len(frames) == 0:
        return frames, kp, genuine
    nxt = np.append(frames[1:], np.iinfo(np.int64).max)
    vel = np.zeros((len(frames), 17, 2))
    if mode == "cv" and len(frames) > 1:
        d = np.diff(frames)
        ok = d <= k
        v = (kp[1:, :, :2] - kp[:-1, :, :2]) / d[:, None, None]
        vel[1:][ok] = v[ok]
    parts_f, parts_k, parts_g = [frames], [kp], [genuine]
    for o in range(1, k):
        f = frames + o
        valid = f < nxt
        if n_frames is not None:
            valid &= f < n_frames
        if not valid.any():
            continue
        k2 = kp[valid].copy()
        if mode == "cv":
            k2[:, :, :2] += vel[valid] * o
        parts_f.append(f[valid])
        parts_k.append(k2)
        parts_g.append(np.zeros(int(valid.sum()), dtype=bool))
    all_f = np.concatenate(parts_f)
    order = np.argsort(all_f, kind="stable")
    return all_f[order], np.concatenate(parts_k)[order], np.concatenate(parts_g)[order]


# =====================================================================================================================
# 2. Queue and delivery (pure)
# =====================================================================================================================
def fifo_finish(arrival_ms: np.ndarray, service_ms: np.ndarray) -> np.ndarray:
    """Single worker, FIFO, work-conserving: finish_t = max(arrival_t, finish_{t-1}) + service_t, in closed form
    finish_t = C_t + max_{j <= t}(arrival_j - C_{j-1}) with C = cumsum(service)."""
    arrival_ms = np.asarray(arrival_ms, dtype=np.float64)
    service_ms = np.asarray(service_ms, dtype=np.float64)
    c = np.cumsum(service_ms)
    return c + np.maximum.accumulate(arrival_ms - (c - service_ms))


def fifo_finish_reference(arrival_ms, service_ms) -> np.ndarray:
    out, free = np.zeros(len(arrival_ms)), -math.inf
    for i, (a, s) in enumerate(zip(arrival_ms, service_ms)):
        free = max(a, free) + s
        out[i] = free
    return out


def fifo_finish_segments(arrival_ms: np.ndarray, service_ms: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """Queue that is empty at every segment (clip) start; `offsets` = segment start indices + the end."""
    out = np.empty(len(arrival_ms))
    for a, b in zip(offsets[:-1], offsets[1:]):
        out[a:b] = fifo_finish(arrival_ms[a:b], service_ms[a:b])
    return out


def delivery_frame(finish_ms: np.ndarray, dt_ms: float) -> np.ndarray:
    """First frame whose capture time is >= finish: a score produced at `finish` can only be shown from this frame on."""
    return np.ceil(np.asarray(finish_ms) / dt_ms - 1e-9).astype(np.int64)


def queue_metrics(arrival_ms: np.ndarray, service_ms: np.ndarray, finish_ms: np.ndarray, dt_ms: float, cap_ms: float = CAP_MS) -> dict:
    """Backlog statistics of a (continuous) stream. Sustainable = rho < 1 and max queue wait <= cap and p99 latency <= cap."""
    start = finish_ms - service_ms
    wait = np.maximum(start - arrival_ms, 0.0)
    latency = finish_ms - arrival_ms
    span = float(len(arrival_ms) * dt_ms)
    rho = float(service_ms.sum() / span)
    t_s = arrival_ms / 1000.0
    if len(wait) > 2 and wait.max() > 0:  # closed-form least-squares slope (no LAPACK: it is not fork-safe on macOS Accelerate)
        tc = t_s - t_s.mean()
        slope = float((tc * (wait - wait.mean())).sum() / (tc * tc).sum())
    else:
        slope = 0.0
    out = {"rho": rho, "mean_wait_ms": float(wait.mean()), "p95_wait_ms": float(np.quantile(wait, 0.95)), "max_wait_ms": float(wait.max()),
           "p99_latency_ms": float(np.quantile(latency, 0.99)), "max_latency_ms": float(latency.max()),
           "final_wait_ms": float(wait[-1]) if len(wait) else 0.0, "wait_slope_ms_per_s": slope,
           "late_frame_fraction": float((latency > cap_ms).mean())}
    out["sustainable"] = bool(rho < 1.0 and out["max_wait_ms"] <= cap_ms and out["p99_latency_ms"] <= cap_ms)
    return out


def land_windows(clip_idx: np.ndarray, last_frame: np.ndarray, offsets: np.ndarray, finish_ms: np.ndarray, arrival_ms: np.ndarray,
                 dt_ms: float, n_frames: np.ndarray, cap_ms: float = CAP_MS) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Where does the score of a window completing on frame `last_frame` of clip `clip_idx` become visible?

    -> (landing frame (clip-local), delivered mask, latency ms capture-of-last-pose -> emission, reason code 0 ok / 1 late / 2 after clip end).
    landing >= completion frame always; a score later than `cap_ms` after the window's last frame was captured is discarded (late work)."""
    g = offsets[clip_idx] + last_frame
    latency = finish_ms[g] - arrival_ms[g]
    landing = delivery_frame(finish_ms[g], dt_ms) - offsets[clip_idx]
    landing = np.maximum(landing, last_frame)
    late = latency > cap_ms
    beyond = landing >= n_frames[clip_idx]
    reason = np.where(late, 1, np.where(beyond, 2, 0)).astype(np.int8)
    return landing, reason == 0, latency, reason


# =====================================================================================================================
# 3. Service-time model (pure given pools)
# =====================================================================================================================
STAGES_T = ("grab", "retrieve", "jpeg", "pose")  # thread-count dependent pools; tracker is single-threaded; scoring depends on T and N


def timing_stats(x) -> dict:
    """mean, std, p50, p90, p95, p99, min, max (ms) of a sample."""
    v = np.asarray(x, dtype=np.float64)
    if len(v) == 0:
        return {k: float("nan") for k in ("mean", "std", "p50", "p90", "p95", "p99", "min", "max")} | {"n": 0}
    q = np.quantile(v, [0.5, 0.9, 0.95, 0.99])
    return {"n": int(len(v)), "mean": float(v.mean()), "std": float(v.std()), "p50": float(q[0]), "p90": float(q[1]), "p95": float(q[2]),
            "p99": float(q[3]), "min": float(v.min()), "max": float(v.max())}


def interp_cores(means: dict[int, float], q: float) -> float:
    """Mean service time at `q` effective cores from the measured thread-scaling curve {T: mean}: linear interpolation between measured
    thread counts, flat above the largest, mean(1)/q below one core (CPU quota)."""
    ts = sorted(means)
    if q < ts[0]:
        return means[ts[0]] / q
    if q >= ts[-1]:
        return means[ts[-1]]
    lo = max(t for t in ts if t <= q)
    hi = min(t for t in ts if t >= q)
    if lo == hi:
        return means[lo]
    w = (q - lo) / (hi - lo)
    return (1 - w) * means[lo] + w * means[hi]


class ServiceModel:
    """Empirical per-stage service-time pools (wall, cpu) measured in the VM, with the budget / Pi mapping.

    pools[("pose", T)] = (wall, cpu); ("grab", T), ("retrieve", T), ("jpeg", T); ("track", n); ("score", T, n).
    Frames are mapped to pool entries through FIXED uniform draws u (common random numbers): two scenarios differ only by the pools'
    scale, never by sampling noise."""

    def __init__(self, pools: dict):
        self.pools = pools
        self.threads = sorted({k[1] for k in pools if k[0] == "pose"})
        self.score_n = sorted({k[2] for k in pools if k[0] == "score"})
        self.track_n = sorted({k[1] for k in pools if k[0] == "track"})

    def stage_pool(self, stage: str, q: float, n: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        """(wall, cpu) pool of `stage` at q effective cores. Wall is rescaled so that its mean equals the interpolated scaling curve."""
        if stage == "track":
            n = self._nearest(self.track_n, n)
            return self.pools[("track", n)]
        ts = self.threads
        t0 = max([t for t in ts if t <= q], default=ts[0])
        if stage == "score":
            n = self._nearest(self.score_n, n)
            means = {t: float(self.pools[("score", t, n)][0].mean()) for t in ts}
            wall, cpu = self.pools[("score", t0, n)]
        else:
            means = {t: float(self.pools[(stage, t)][0].mean()) for t in ts}
            wall, cpu = self.pools[(stage, t0)]
        base = means[t0]
        factor = interp_cores(means, q) / base if base > 0 else 1.0
        return wall * factor, cpu

    @staticmethod
    def _nearest(values: list[int], n: int) -> int:
        n = int(n)
        return min(values, key=lambda v: (abs(v - n), v))

    def sample(self, q: float, obs: np.ndarray, n_in: np.ndarray, n_win: np.ndarray, u: np.ndarray, decode: str = "h264"
               ) -> tuple[np.ndarray, np.ndarray, dict]:
        """Per-frame (wall ms, cpu ms) of the job of every frame. `u` = fixed uniforms (frames x 5): grab/jpeg, retrieve, pose, track, score.

        decode 'h264': every frame is decoded (skipped frames: grab only, observed: grab + retrieve); 'jpeg': observed frames only."""
        total = len(obs)
        wall, cpu = np.zeros(total), np.zeros(total)
        parts = {}

        def draw(pool: tuple[np.ndarray, np.ndarray], col: int, mask: np.ndarray):
            idx = np.minimum((u[mask, col] * len(pool[0])).astype(np.int64), len(pool[0]) - 1)
            return pool[0][idx], pool[1][idx]

        if decode == "h264":
            w, c = draw(self.stage_pool("grab", q), 0, np.ones(total, bool))
            wall += w
            cpu += c
            if obs.any():
                w, c = draw(self.stage_pool("retrieve", q), 1, obs)
                wall[obs] += w
                cpu[obs] += c
        else:
            if obs.any():
                w, c = draw(self.stage_pool("jpeg", q), 0, obs)
                wall[obs] += w
                cpu[obs] += c
        parts["decode_ms"] = float(wall.sum())
        if obs.any():
            w, c = draw(self.stage_pool("pose", q), 2, obs)
            wall[obs] += w
            cpu[obs] += c
            parts["pose_ms"] = float(w.sum())
            tw = np.zeros(total)
            tc = np.zeros(total)
            for n in np.unique(n_in[obs]):
                m = obs & (n_in == n)
                tw[m], tc[m] = draw(self.stage_pool("track", q, int(n)), 3, m)
            wall += tw
            cpu += tc
            parts["track_ms"] = float(tw.sum())
        sw = np.zeros(total)
        sc = np.zeros(total)
        for n in np.unique(n_win[n_win > 0]):
            m = n_win == n
            sw[m], sc[m] = draw(self.stage_pool("score", q, int(n)), 4, m)
        wall += sw
        cpu += sc
        parts["score_ms"] = float(sw.sum())
        return wall, cpu, parts


def scenario_list() -> list[dict]:
    """Headline scenarios (plan section 5): id, threads, s, b, effective cores q = CORES * b."""
    out = [{"id": f"T4_s{s:.1f}_b1.00", "s": s, "b": 1.0, "q": float(CORES)} for s in S_GRID]
    out += [{"id": f"T4_s2.0_b{b:.2f}", "s": 2.0, "b": b, "q": CORES * b} for b in B_GRID[1:]]
    return out


# =====================================================================================================================
# 4. Selection rules (pure)
# =====================================================================================================================
def select_fixed_reference(rows: list[dict], tie: float = TIE_BAND, require_compliant: bool = False) -> dict | None:
    """B1: among sustainable schedules the highest dev delivered AUC; schedules within `tie` of it are tied -> lowest CPU ms per video-second.
    rows: dicts with keys schedule, sustainable, dev_auc, cpu_ms_per_s, plan_compliant."""
    ok = [r for r in rows if r["sustainable"] and not math.isnan(r["dev_auc"]) and (r["plan_compliant"] or not require_compliant)]
    if not ok:
        return None
    best = max(r["dev_auc"] for r in ok)
    tied = [r for r in ok if r["dev_auc"] >= best - tie]
    return min(tied, key=lambda r: (r["cpu_ms_per_s"], r["schedule"]))


def pareto_front(points: list[tuple[float, float, str]]) -> list[str]:
    """Names of non-dominated points (cost to minimise, value to maximise)."""
    out = []
    for cost, value, name in points:
        if not any((c <= cost and v >= value) and (c < cost or v > value) for c, v, n in points if n != name):
            out.append(name)
    return out


def h1_decision(diff: dict) -> str:
    """H1 verdict from a paired contrast (burst - stride): CI lower > 0 burst preferred, CI upper < 0 stride preferred, else tie."""
    lo, hi = diff["ci95"]
    return "burst preferred" if lo > 0 else ("stride preferred" if hi < 0 else "tie")


# =====================================================================================================================
# 5. Lock (config.yaml is written BEFORE any VM or simulation number exists)
# =====================================================================================================================
def short(path: Path) -> str:
    return e01.short(path)


def vm_facts() -> dict:
    try:
        r = subprocess.run(["limactl", "shell", "edge-os", "--", "bash", "-c",
                            "nproc; free -m | sed -n 2p; uname -r; lscpu | grep -E 'Vendor ID|Model name|Flags' | cut -c1-120; "
                            "cat /etc/os-release | grep PRETTY_NAME"], capture_output=True, text=True, timeout=60)
        return {"raw": r.stdout.strip().splitlines()}
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}


def build_config(log=print) -> dict:
    from edgelab.analysis import q2_p1_e02_vm as vm

    e00cfg, e00sum = e01.load_e00()
    e01cfg = yaml.safe_load((E01_DIR / "config.yaml").read_text())
    e01sum = json.loads((E01_DIR / "summary.json").read_text())
    lock = e01sum["locked_for_E02"]
    frontends = {f["key"]: f for f in e01cfg["frontends"]}
    arms = {}
    for ds, names in FRONTENDS.items():
        for name in names:
            f = frontends[f"{ds}/{name}"]
            arms[f"{ds}/{name}"] = {"cache_dir": f["cache_dir"], "cache_tree_sha256": f["cache_tree_sha256"], "runtime": f["runtime"],
                                    "imgsz": f["imgsz"], "sentinel_distribution": f["sentinel_distribution"],
                                    "sentinel_value": f["sentinel_value"], "pose_timing_model": POSE_TIMING_MODEL[(ds, name)]}
    log("hashing VM model files ...")
    models = {}
    for mid, (runtime, path) in vm.POSE_MODELS.items():
        files = [path] if path.is_file() else sorted(path.glob("model.ncnn.*"))
        models[mid] = {"runtime": runtime, "path": str(path), "files": {f.name: e00.sha256_file(f) for f in files}}
    cells = vm.cell_list()
    config = {
        "experiment": "Q2-P1-E02_fixed_baseline",
        "question": "What does an optimized fixed pipeline achieve on the actual Pi? Profile decode, pose, tracking and scoring; use the "
                    "intended runtime. Required: stage timings, memory, delivered AUC and sustainable workload.",
        "training": False,
        "seed": SEED,
        "pool_seed": POOL_SEED,
        "bootstrap": {"unit": "video", "resamples": N_BOOT, "seed": 0},
        "hardware_disclosure": {
            "physical_pi5": "NOT connected (owner to-do, DECISIONS D8/D10); no number in E02 is a Pi measurement",
            "machine_tag": MACHINE_TAG, "vm": vm_facts(),
            "host": {"cpu": "Apple M2 Pro (10 cores)", "note": "Lima vz VM edge-os, 4 vCPU, 8 GiB; no DVFS/thermal limit; ~2-2.5x faster single-core than Cortex-A76 @ 2.4 GHz"},
            "pi_multiplier_s": list(S_GRID), "budget_b": list(B_GRID), "device_cores": CORES},
        "depends_on": {
            "E00": {"config_sha256": e00.sha256_file(E00_DIR / "config.yaml"), "summary_sha256": e00.sha256_file(E00_DIR / "summary.json"),
                    "report_sha256": e00.sha256_file(E00_DIR / "report.md")},
            "E01": {"config_sha256": e00.sha256_file(E01_DIR / "config.yaml"), "summary_sha256": e00.sha256_file(E01_DIR / "summary.json"),
                    "report_sha256": e00.sha256_file(E01_DIR / "report.md")},
            "E02_plan_sha256": e00.sha256_file(EXP_DIR / "plan.md")},
        "locked_from_E01": {"frontend": lock["frontend"]["name"], "tracker": lock["tracker"]["label"],
                            "default_tracker": e01.DEFAULT_CELL.label, "runner_up": "stock_ncnn800",
                            "S4_reference_locked_tracker": {"ShanghaiTech": 76.82, "UBnormal_ML0_mps": 61.08},
                            "S4_reference_default_tracker": {"ShanghaiTech": 73.78, "UBnormal_ML0_mps": 58.04}},
        "scoring_protocol": e01sum["protocol"],
        "splits_from_E00": e01cfg["splits_from_E00"],
        "scorers": e01cfg["scorers"],
        "arms": arms,
        "trackers": {k: {"label": v.label, "c": v.c, "low": v.low, "buffer": v.buffer, "match": v.match, "gate": v.gate}
                     for k, v in TRACKERS.items()},
        "schedules": {ds: [{"id": s.id, "family": s.family, "k": s.k, "fill": s.fill, "refresh": s.refresh, "hold": s.hold,
                            "duty": s.duty(SEG_LENS[ds]), "propagated": s.propagated, "alias_of": s.alias_of} for s in schedule_grid(ds)]
                      for ds in DATASETS},
        "scenarios": scenario_list(),
        "delivery": {"cap_ms": CAP_MS, "stream_models": ["continuous", "clip_reset"], "headline": "continuous", "headline_scenarios": list(HEADLINE),
                     "sustainable": "rho < 1 and max queue wait <= cap and p99 latency <= cap (continuous stream)",
                     "late_work": "scores later than cap after the capture of their last pose are discarded",
                     "hold_rule": "hold = refresh interval - 1 frames (a priori); H=0 reported as sensitivity",
                     "landing": "frame ceil(finish / dt), never before the completion frame"},
        "fps": FPS, "seg_len": SEG_LENS,
        "vm_protocol": {"frames": "12 L0 ShanghaiTech clips x 240 = 2,880 frames (natural daylight, un-darkened)", "threads": list(THREADS),
                        "affinity": "cores 0..T-1", "host_load_limit_pct": 30, "max_tries": 4, "conf_floor": vm.GATE,
                        "cells": [c["id"] for c in cells], "models": models,
                        "decode_sources": {"jpg": "12 SHT clips (2,880 frames)", "mpeg1": vm.SHT_AVI, "h264_sht": "x264 re-encode of the 12 SHT clips (ours)",
                                           "h264_ub": vm.UB_MP4}},
        "tolerances": {"P2_repeat_pct": 5.0, "P3_model_vs_e2e_pct": 10.0, "P4_monotone_tol_pct": 3.0, "P5_min_frames_fraction": 0.95,
                       "D1_parity_pt": 0.02, "selection_tie_band_pt": TIE_BAND},
        "rules": ["P1", "P2", "P3", "P4", "P5", "D1", "D2", "D3", "D4", "D5", "D6", "D7", "H1", "H2", "H3", "B1"],
        "environment": e00.env_info(),
    }
    return plain(config)


def cmd_lock(log=print) -> None:
    config = build_config(log)
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    header = ("# Q2-P1-E02 locked configuration (generated by `python -m edgelab.analysis.q2_p1_e02 lock` BEFORE `vm` / `run`; do not hand-edit).\n"
              "# Hashes of the VM models, detector caches and the E00/E01 inputs; schedule and scenario grids; tolerances.\n")
    (EXP_DIR / "config.yaml").write_text(header + yaml.safe_dump(config, sort_keys=False, width=140))
    log(f"wrote {EXP_DIR / 'config.yaml'}")


# =====================================================================================================================
# 6. VM traces -> stage tables, service pools, profiling rules P1-P5
# =====================================================================================================================
POSE_IDS = ("ML0-640", "MLP-640", "stock-640", "stock-800", "ML0-640-ub", "stock-640-ort-fp32", "stock-640-ort-int8", "stock-800-ort-fp32",
            "stock-800-ort-int8")
DECODE_KINDS = ("jpg", "mpeg1", "h264_sht", "h264_ub")


def load_cell(cell_id: str, out_dir: Path | None = None) -> tuple[dict, dict] | None:
    prefix = (out_dir or VM_OUT) / cell_id
    if not Path(f"{prefix}.npz").exists() or not Path(f"{prefix}.json").exists():
        return None
    z = np.load(f"{prefix}.npz")
    return {k: z[k] for k in z.files}, json.loads(Path(f"{prefix}.json").read_text())


def load_manifest(out_dir: Path | None = None) -> dict[str, dict]:
    """Last attempt of every cell (the file on disk belongs to the last attempt)."""
    path = (out_dir or VM_OUT) / "manifest.jsonl"
    rows: dict[str, dict] = {}
    if path.exists():
        for line in path.read_text().splitlines():
            r = json.loads(line)
            rows[r["cell"]] = r
    return rows


def pose_cell_ids(model: str, t: int) -> list[str]:
    return [f"pose_{model}_T{t}_rep{r}" for r in (0, 1)] if model == "ML0-640" else [f"pose_{model}_T{t}"]


def pose_arrays(model: str, t: int) -> dict | None:
    parts = [c for c in (load_cell(i) for i in pose_cell_ids(model, t)) if c is not None]
    if not parts:
        return None
    keys = parts[0][0].keys()
    out = {k: np.concatenate([p[0][k] for p in parts]) for k in keys}
    out["_meta"] = parts[0][1]
    out["_peak_rss"] = max(p[1]["peak_rss_mb"] for p in parts)
    return out


def cell_stats(x) -> dict:
    return timing_stats(x)


def decode_rows() -> list[dict]:
    rows = []
    for kind in DECODE_KINDS:
        for t in THREADS:
            c = load_cell(f"decode_{kind}_T{t}")
            if c is None:
                continue
            a, m = c
            total = a["wall_grab_ms"] + a["wall_retrieve_ms"]
            cpu = a["cpu_grab_ms"] + a["cpu_retrieve_ms"]
            cli = [x for x in (m.get("ffmpeg_cli") or []) if x and "cpu_ms_per_frame" in x]
            rows.append({"kind": kind, "threads": t, "frames": int(len(total)), "grab": cell_stats(a["wall_grab_ms"]),
                         "retrieve": cell_stats(a["wall_retrieve_ms"]), "total": cell_stats(total), "cpu_ms_mean": float(cpu.mean()),
                         "ffmpeg_cli_cpu_ms_per_frame": float(np.mean([x["cpu_ms_per_frame"] for x in cli])) if cli else None,
                         "ffmpeg_cli_wall_ms_per_frame": float(np.mean([x["wall_ms_per_frame"] for x in cli])) if cli else None,
                         "peak_rss_mb": m["peak_rss_mb"], "files": m["files"][:3] if kind != "jpg" else m["files"][:1]})
    return rows


def pose_rows() -> list[dict]:
    rows = []
    base: dict[str, float] = {}
    for model in POSE_IDS:
        for t in THREADS:
            a = pose_arrays(model, t)
            if a is None:
                continue
            w = a["wall_ms"]
            if t == 1:
                base[model] = float(w.mean())
            row = {"model": model, "runtime": a["_meta"]["runtime"], "threads": t, "frames": int(len(w)), "input_hw": a["_meta"]["input_hw"],
                   **{f"wall_{k}": v for k, v in cell_stats(w).items() if k != "n"},
                   "pre_ms": float(a["pre_ms"].mean()), "infer_ms": float(a["infer_ms"].mean()), "post_ms": float(a["post_ms"].mean()),
                   "cpu_ms_mean": float(a["cpu_ms"].mean()), "dets_025_mean": float(a["n_dets_025"].mean()),
                   "dets_floor_mean": float(a["n_dets"].mean()), "peak_rss_mb": a["_peak_rss"],
                   "rss_after_load_mb": a["_meta"].get("rss_after_load_mb")}
            if model in base:
                row["speedup_vs_T1"] = base[model] / row["wall_mean"]
                row["efficiency"] = row["speedup_vs_T1"] / t
            rows.append(row)
    return rows


def track_rows() -> list[dict]:
    c = load_cell("track_T1")
    if c is None:
        return []
    a, m = c
    rows = []
    for name in TRACKERS:
        n_in = a[f"real_{name}_n_in"]
        for n in sorted(set(n_in.tolist())):
            sel = n_in == n
            if sel.sum() >= 20:
                rows.append({"tracker": name, "source": "real SHT ML0 cache", "n_in": int(n), **cell_stats(a[f"real_{name}_ms"][sel])})
        rows.append({"tracker": name, "source": "real SHT ML0 cache", "n_in": "all", **cell_stats(a[f"real_{name}_ms"])})
        sn = a[f"syn_{name}_n"]
        for n in sorted(set(sn.tolist())):
            rows.append({"tracker": name, "source": "synthetic", "n_in": int(n), **cell_stats(a[f"syn_{name}_ms"][sn == n])})
    return rows


def score_rows() -> list[dict]:
    rows = []
    for ds in DATASETS:
        for t in THREADS:
            c = load_cell(f"score_{ds}_T{t}")
            if c is None:
                continue
            a, m = c
            for n in sorted(set(a["n"].tolist())):
                sel = a["n"] == n
                rows.append({"dataset": ds, "threads": t, "n_windows": int(n), "assemble_ms": float(a["assemble_ms"][sel].mean()),
                             "normalize_ms": float(a["normalize_ms"][sel].mean()), "flow_ms": float(a["flow_ms"][sel].mean()),
                             "cpu_ms": float(a["cpu_ms"][sel].mean()), **{f"total_{k}": v for k, v in cell_stats(a["total_ms"][sel]).items()
                                                                           if k in ("mean", "std", "p50", "p90", "p95", "p99")}})
    return rows


E2E_IDS = [("ShanghaiTech", "h264", t) for t in THREADS] + [("ShanghaiTech", "jpg", 4)] + [("UBnormal", "h264", t) for t in THREADS]


def e2e_rows() -> list[dict]:
    rows = []
    for ds, source, t in E2E_IDS:
        c = load_cell(f"e2e_{ds}_{source}_T{t}")
        if c is None:
            continue
        a, m = c
        fps = FPS[ds]
        svc = a["service_ms"]
        rows.append({"dataset": ds, "source": source, "threads": t, "frames": int(len(svc)), "model": m["model"], "input_hw": m["input_hw"],
                     **{f"service_{k}": v for k, v in cell_stats(svc).items() if k != "n"},
                     "decode_ms": float(a["decode_ms"].mean()), "pose_ms": float((a["pre_ms"] + a["infer_ms"] + a["post_ms"]).mean()),
                     "pre_ms": float(a["pre_ms"].mean()), "infer_ms": float(a["infer_ms"].mean()), "post_ms": float(a["post_ms"].mean()),
                     "track_ms": float(a["track_ms"].mean()), "score_ms": float((a["assemble_ms"] + a["score_ms"]).mean()),
                     "cpu_ms_mean": float(a["cpu_ms"].mean()), "n_in_mean": float(a["n_in"].mean()), "windows_per_frame": float(a["n_windows"].mean()),
                     "service_ms_per_video_s": float(svc.mean() * fps), "cpu_ms_per_video_s": float(a["cpu_ms"].mean() * fps),
                     "max_fps": float(1000.0 / svc.mean()), "peak_rss_mb": m["peak_rss_mb"], "wall_total_s": m["wall_total_s"]})
    return rows


def build_pools(dataset: str, frontend: str, tracker: str) -> dict:
    """ServiceModel pools of one (dataset, frontend, tracker) from the VM cells."""
    pools: dict = {}
    pose_model = POSE_TIMING_MODEL[(dataset, frontend)]
    kind = "h264_sht" if dataset == "ShanghaiTech" else "h264_ub"
    for t in THREADS:
        a = pose_arrays(pose_model, t)
        pools[("pose", t)] = (a["wall_ms"].astype(float), a["cpu_ms"].astype(float))
        d = load_cell(f"decode_{kind}_T{t}")[0]
        pools[("grab", t)] = (d["wall_grab_ms"].astype(float), d["cpu_grab_ms"].astype(float))
        pools[("retrieve", t)] = (d["wall_retrieve_ms"].astype(float), d["cpu_retrieve_ms"].astype(float))
        j = load_cell(f"decode_jpg_T{t}")[0]
        pools[("jpeg", t)] = (j["wall_retrieve_ms"].astype(float), j["cpu_retrieve_ms"].astype(float))
        s, _ = load_cell(f"score_{dataset}_T{t}")
        for n in sorted(set(s["n"].tolist())):
            if n > 0:
                sel = s["n"] == n
                pools[("score", t, int(n))] = (s["total_ms"][sel].astype(float), s["cpu_ms"][sel].astype(float))
    tr, _ = load_cell("track_T1")
    real_n = tr[f"real_{tracker}_n_in"]
    sn = tr[f"syn_{tracker}_n"]
    for n in sorted(set(sn.tolist())):
        sel = real_n == n
        if sel.sum() >= 30:
            pools[("track", int(n))] = (tr[f"real_{tracker}_ms"][sel].astype(float), tr[f"real_{tracker}_cpu_ms"][sel].astype(float))
        else:
            pools[("track", int(n))] = (tr[f"syn_{tracker}_ms"][sn == n].astype(float), tr[f"syn_{tracker}_cpu_ms"][sn == n].astype(float))
    return pools


def p3_prediction(threads: int) -> dict:
    """Model-vs-measured mean service of the SHT H.264 end-to-end cell: pool model fed with the e2e run's own n_in / n_windows."""
    a, m = load_cell(f"e2e_ShanghaiTech_h264_T{threads}")
    sm = ServiceModel(build_pools("ShanghaiTech", "ML0_ncnn", "locked"))
    n = len(a["service_ms"])
    rng = np.random.default_rng(POOL_SEED)
    u = rng.random((n, 5))
    wall, cpu, parts = sm.sample(float(threads), np.ones(n, bool), a["n_in"].astype(int), a["n_windows"].astype(int), u, "h264")
    return {"threads": threads, "predicted_mean_ms": float(wall.mean()), "measured_mean_ms": float(a["service_ms"].mean()),
            "rel_error_pct": float(100 * (wall.mean() - a["service_ms"].mean()) / a["service_ms"].mean()),
            "predicted_cpu_ms": float(cpu.mean()), "measured_cpu_ms": float(a["cpu_ms"].mean())}


def profiling_rules(manifest: dict, pose: list[dict], p3: list[dict]) -> dict:
    out = {}
    from edgelab.analysis import q2_p1_e02_vm as vm

    cells = {c["id"] for c in vm.cell_list()}
    missing = sorted(cells - set(manifest))
    excl = sorted(c for c, r in manifest.items() if r.get("excluded"))
    loads = [r["host_outside_vm_pct"] for r in manifest.values() if r.get("host_outside_vm_pct") is not None]
    out["P1"] = {"pass": not excl and not missing, "detail": f"{len(manifest)} cells; host CPU outside the VM mean {np.mean(loads):.1f}% max {np.max(loads):.1f}% "
                                                              f"(limit 30%); excluded after retries: {excl or 'none'}; missing: {missing or 'none'}",
                 "excluded": excl, "missing": missing}
    ml0 = {t: {r: load_cell(f"pose_ML0-640_T{t}_rep{r}") for r in (0, 1)} for t in THREADS}
    diffs = {}
    for t in THREADS:
        if ml0[t][0] and ml0[t][1]:
            a, b = ml0[t][0][0]["wall_ms"].mean(), ml0[t][1][0]["wall_ms"].mean()
            diffs[t] = float(100 * abs(a - b) / min(a, b))
    out["P2"] = {"pass": bool(diffs) and max(diffs.values()) <= 5.0, "detail": "rep0 vs rep1 mean pose ms difference (%): " +
                 ", ".join(f"T{t} {d:.1f}" for t, d in diffs.items()), "diff_pct": diffs}
    out["P3"] = {"pass": bool(p3) and max(abs(r["rel_error_pct"]) for r in p3) <= 10.0,
                 "detail": "pool model vs measured end-to-end mean (SHT H.264, %): " + ", ".join(f"T{r['threads']} {r['rel_error_pct']:+.1f}" for r in p3),
                 "rows": p3}
    means = [r["wall_mean"] for r in pose if r["model"] == "ML0-640"]
    mono = all(means[i + 1] <= means[i] * 1.03 for i in range(len(means) - 1)) and len(means) == len(THREADS)
    out["P4"] = {"pass": mono, "detail": "ML0 NCNN pose mean ms by T: " + ", ".join(f"{m:.1f}" for m in means), "means": means}
    bad = []
    ub_meta = load_cell("decode_h264_ub_T1")
    ub_expected = sum(min(300, f["n_frames"]) for f in ub_meta[1]["files"]) if ub_meta else 900  # the UB clips are shorter than 300 frames
    for cid in cells:
        c = load_cell(cid)
        if c is None:
            bad.append((cid, 0.0))
            continue
        a, m = c
        n = len(next(iter(a.values())))
        expected = {"pose": 2880 if not cid.startswith("pose_ML0-640-ub") else ub_expected, "decode": None,
                    "e2e": 2880 if "UBnormal" not in cid else ub_expected, "score": None, "track": None}[cid.split("_")[0]]
        if cid.startswith("decode_"):
            expected = 2880 if "jpg" in cid or "h264_sht" in cid else sum(min(400, f.get("n_frames", 400)) for f in m["files"])
        if expected and n < 0.95 * expected:
            bad.append((cid, n / expected))
    out["P5"] = {"pass": not bad, "detail": f"every cell processed >= 95% of its requested frames" if not bad else f"short cells: {bad}"}
    return out


# =====================================================================================================================
# 7. Scheduled window tables (host; one tracker replay per trackset, windows scored once)
# =====================================================================================================================
def trackset_mask(ts: tuple[str, int], n_frames: int, seg_len: int) -> np.ndarray:
    kind, k = ts
    t = np.arange(n_frames)
    return (t % k) == 0 if kind == "stride" else (t % k) < seg_len


def schedule_ids(dataset: str, primary: bool, tracker: str) -> list[str]:
    """Which schedules an (arm, tracker) is evaluated under. Primary arm: all; other frontends (locked tracker): K=1, ZOH strides, all bursts."""
    grid = schedule_grid(dataset)
    if primary:
        return [s.id for s in grid]
    return [s.id for s in grid if s.family == "burst" or s.k == 1 or s.fill == "zoh"]


_G: dict = {}


def _replay_clip(ci: int) -> dict:
    frames, ids, cell, L = _G["frames"][ci], _G["ids"][ci], _G["cell"], _G["L"]
    out = {}
    for ts in _G["tracksets"]:
        kind, k = ts
        if kind == "stride":
            sel = np.nonzero(ids % k == 0)[0]
            out[ts] = e01.replay_cell([frames[i] for i in sel], ids[sel], cell)
        else:  # burst with a gap: a fresh tracker for every burst, no pose is carried across the gap
            tracks, bursts = {}, ids // k
            for b in np.unique(bursts):
                sel = np.nonzero((bursts == b) & ((ids % k) < L))[0]
                if len(sel):
                    for pid, v in e01.replay_cell([frames[i] for i in sel], ids[sel], cell).items():
                        tracks[int(b) * 100000 + pid] = v
            out[ts] = tracks
    return out


def replay_tracksets(frames_by_clip: list, ids_by_clip: list, cell: e01.TrackerCell, tracksets: list, seg_len: int, procs: int
                     ) -> dict[tuple, list[dict]]:
    _G.update(frames=frames_by_clip, ids=ids_by_clip, cell=cell, L=seg_len, tracksets=tracksets)
    try:
        per_clip = e01.pmap(_replay_clip, list(range(len(frames_by_clip))), procs)
    finally:
        for k in ("frames", "ids", "cell", "L", "tracksets"):
            _G.pop(k, None)
    return {ts: [pc[ts] for pc in per_clip] for ts in tracksets}


def scheduled_table(tracks_by_clip: list[dict], n_frames: list[int], seg_len: int, scorer, sched: Schedule, chunk: int = 4096) -> dict:
    """Windows of the schedule's scoring grid only (never scores a window the schedule would not run).

    Columns: clip_idx, person, last_frame, genuine_frac (share of the window's poses that are real observations), normality."""
    cols = {"clip_idx": [], "person": [], "last_frame": [], "genuine_frac": []}
    buf, norms, nbuf = [], [], 0

    def flush():
        nonlocal buf, nbuf
        if buf:
            norms.append(scorer.score(np.concatenate(buf)))
        buf, nbuf = [], 0

    for ci, tracks in enumerate(tracks_by_clip):
        for pid, (frames, kp) in tracks.items():
            gen = None
            if sched.propagated:
                frames, kp, gen = propagate_track(frames, kp, sched.k, sched.fill, n_frames[ci])
            starts = e00.complete_window_starts(frames, seg_len)
            if len(starts) == 0:
                continue
            last = frames[starts + seg_len - 1]
            keep = sched.keep_windows(last, seg_len)
            if not keep.any():
                continue
            starts, last = starts[keep], last[keep]
            if gen is None:
                gf = np.ones(len(starts))
            else:
                cum = np.concatenate([[0], np.cumsum(gen)])
                gf = (cum[starts + seg_len] - cum[starts]) / seg_len
            buf.append(e00.gather_windows(kp, starts, seg_len))
            nbuf += len(starts)
            cols["clip_idx"].append(np.full(len(starts), ci, dtype=np.int32))
            cols["person"].append(np.full(len(starts), pid, dtype=np.int64))
            cols["last_frame"].append(last.astype(np.int32))
            cols["genuine_frac"].append(gf)
            if nbuf >= chunk:
                flush()
    flush()
    table = {c: (np.concatenate(v) if v else np.zeros(0)) for c, v in cols.items()}
    table["clip_idx"] = table["clip_idx"].astype(np.int32)
    table["last_frame"] = table["last_frame"].astype(np.int32)
    table["normality"] = np.concatenate(norms) if norms else np.zeros(0)
    return table


def filter_table(table: dict, mask: np.ndarray) -> dict:
    return {k: (v[mask] if isinstance(v, np.ndarray) and v.shape[:1] == mask.shape else v) for k, v in table.items()}


def build_tables(tracks: dict, n_frames: list[int], seg_len: int, scorer, sids: list[str], dataset: str, log=print) -> dict[str, dict]:
    """sid -> window table. K=1 is the full E00/E01 table (all complete windows, E00 columns); bursts with W <= L filter it."""
    by_id = {s.id: s for s in schedule_grid(dataset)}
    base = e01.window_table_from_tracks(tracks[("stride", 1)], seg_len, scorer)
    base["genuine_frac"] = np.ones(len(base["normality"]))
    out = {"stride1": base}
    for sid in sids:
        s = by_id[sid]
        if sid in out:
            continue
        if s.family == "burst" and s.k <= seg_len:
            out[sid] = filter_table(base, s.keep_windows(base["last_frame"], seg_len))
        else:
            t0 = time.perf_counter()
            out[sid] = scheduled_table(tracks[trackset_key(s, seg_len)], n_frames, seg_len, scorer, s)
            log(f"      {sid}: {len(out[sid]['normality'])} windows ({time.perf_counter() - t0:.0f}s)")
    return {sid: out[sid] for sid in sids if sid in out}


def window_counts(table: dict, offsets: np.ndarray, total: int) -> np.ndarray:
    """Windows completing on every (global) frame."""
    if len(table["normality"]) == 0:
        return np.zeros(total, dtype=np.int64)
    return np.bincount(offsets[table["clip_idx"]] + table["last_frame"], minlength=total).astype(np.int64)


# =====================================================================================================================
# 8. Delivery simulation of one (schedule, scenario) on one cell
# =====================================================================================================================
def simulate_delivery(clip_idx: np.ndarray, last_frame: np.ndarray, normality: np.ndarray, n_frames, service_ms: np.ndarray, dt_ms: float,
                      sentinel: float, hold: float, stream: str = "continuous", cap_ms: float = CAP_MS) -> tuple[list[np.ndarray], dict]:
    """Per-frame delivered anomaly scores of a schedule whose frame jobs take `service_ms` (flat over clips).

    Frame g (global index) arrives at g*dt; the FIFO worker serves jobs in frame order; a window completing on frame t of clip c is scored by
    the job of that frame and visible from frame ceil(finish/dt) (never before t); older than `cap_ms` -> discarded."""
    n_frames = np.asarray(n_frames, dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(n_frames)])
    arrival = np.arange(int(offsets[-1])) * dt_ms
    service_ms = np.asarray(service_ms, dtype=np.float64)
    finish = fifo_finish(arrival, service_ms) if stream == "continuous" else fifo_finish_segments(arrival, service_ms, offsets)
    landing, ok, latency, reason = land_windows(np.asarray(clip_idx), np.asarray(last_frame), offsets, finish, arrival, dt_ms, n_frames, cap_ms)
    raw = e00.per_frame_min(np.asarray(clip_idx)[ok], landing[ok], np.asarray(normality)[ok], list(n_frames))
    series, stats = e00.deliver(raw, sentinel, hold)
    return series, {"stats": stats, "arrival": arrival, "finish": finish, "landing": landing, "ok": ok, "latency": latency, "reason": reason,
                    "raw": raw, "offsets": offsets}


def check_queue_invariants(arrival: np.ndarray, service: np.ndarray, finish: np.ndarray, landing: np.ndarray, last_frame: np.ndarray,
                           dt_ms: float, clip_idx: np.ndarray, offsets: np.ndarray) -> dict:
    ref = fifo_finish_reference(arrival, service)
    g = offsets[clip_idx] + last_frame
    return {"fifo_equals_reference": bool(np.allclose(finish, ref, rtol=0, atol=1e-6)),
            "start_ge_arrival": bool(((finish - service) >= arrival - 1e-6).all()),
            "finish_nondecreasing": bool((np.diff(finish) >= -1e-9).all()),
            "landing_ge_completion": bool((landing >= last_frame).all()),
            "landing_ge_ceil_finish": bool((landing + offsets[clip_idx] >= delivery_frame(finish[g], dt_ms)).all())}


def service_for(cd: dict, sid: str, scenario: dict, decode: str = "h264") -> tuple[np.ndarray, np.ndarray, dict]:
    obs = cd["obs"][sid]
    n_in = np.where(obs, cd["n_in_full"], 0)
    wall, cpu, parts = cd["model"].sample(scenario["q"], obs, n_in, cd["n_win"][sid], cd["u"], decode)
    return wall * scenario["s"], cpu * scenario["s"], {k: v * scenario["s"] for k, v in parts.items()}


def per_clip_sum(values: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    return np.add.reduceat(values, offsets[:-1]) if len(values) else np.zeros(len(offsets) - 1)


def run_job(job: dict) -> dict:
    """One delivery simulation. job: sid, scenario (dict | None = ideal), stream, hold ('rule' | 'zero'), decode, keep_series."""
    cd = _G["cd"]
    sid = job["sid"]
    sched = cd["schedules"][sid]
    table = cd["tables"][sid]
    scen = job["scenario"]
    n_frames, offsets, dt = cd["n_frames"], cd["offsets"], cd["dt_ms"]
    total = int(offsets[-1])
    hold = 0 if job["hold"] == "zero" else sched.hold
    if scen is None:
        service = np.zeros(total)
        cpu = np.zeros(total)
        parts = {}
    else:
        service, cpu, parts = service_for(cd, sid, scen, job["decode"])
    series, info = simulate_delivery(table["clip_idx"], table["last_frame"], table["normality"], n_frames, service, dt, cd["sentinel"], hold,
                                     job["stream"])
    auc = e01.auc_subsets(cd["labels"], series, cd["dev_idx"], cd["conf_idx"])
    clip_auc = [100 * e00.clip_auc(cd["labels"][i], series[i]) for i in range(len(series))]
    span_s = total * dt / 1000.0
    out = {"auc": auc, "clip_auc": clip_auc, "stats": info["stats"], "windows": int(len(table["normality"])),
           "windows_delivered": int(info["ok"].sum()), "dropped_late": int((info["reason"] == 1).sum()), "dropped_end": int((info["reason"] == 2).sum()),
           "genuine_frac_mean": float(table["genuine_frac"].mean()) if len(table["normality"]) else float("nan")}
    fresh = np.concatenate([np.isfinite(r) for r in info["raw"]])
    out["coverage"] = float(fresh.mean())
    lat = info["latency"][info["ok"]]
    out["emit_latency_ms_mean"] = float(lat.mean()) if len(lat) else float("nan")
    out["emit_latency_ms_p95"] = float(np.quantile(lat, 0.95)) if len(lat) else float("nan")
    delay = (info["landing"] - table["last_frame"])[info["ok"]]
    out["landing_delay_frames_mean"] = float(delay.mean()) if len(delay) else float("nan")
    if scen is None:
        out.update({"sustainable": True, "rho": 0.0, "max_wait_ms": 0.0, "mean_wait_ms": 0.0, "p95_wait_ms": 0.0, "p99_latency_ms": 0.0,
                    "final_wait_ms": 0.0, "late_frame_fraction": 0.0, "wait_slope_ms_per_s": 0.0, "cpu_ms_per_s": 0.0, "service_ms_per_s": 0.0})
        out["clip_wait_max_ms"] = [0.0] * len(n_frames)
        out["clip_cpu_ms_per_s"] = [0.0] * len(n_frames)
        out["clip_delay_ms"] = [0.0] * len(n_frames)
    else:
        out.update(queue_metrics(info["arrival"], service, info["finish"], dt))  # of the simulated stream model
        out["cpu_ms_per_s"] = float(cpu.sum() / span_s)
        out["service_ms_per_s"] = float(service.sum() / span_s)
        out["parts_ms_per_s"] = {k: float(v / span_s) for k, v in parts.items()}
        wait = np.maximum(info["finish"] - service - info["arrival"], 0.0)
        out["clip_wait_max_ms"] = np.maximum.reduceat(wait, offsets[:-1]).tolist()
        out["clip_cpu_ms_per_s"] = (per_clip_sum(cpu, offsets) / (np.asarray(n_frames) * dt / 1000.0)).tolist()
        sums = np.bincount(table["clip_idx"][info["ok"]], weights=lat, minlength=len(n_frames))
        cnt = np.bincount(table["clip_idx"][info["ok"]], minlength=len(n_frames))
        out["clip_delay_ms"] = np.where(cnt > 0, sums / np.maximum(cnt, 1), np.nan).tolist()
    if job.get("keep_series"):
        out["series"] = np.concatenate(series)
    return out


def max_streams(cd: dict, sid: str, s: float, tracker_model, n_max: int = 16, decode: str = "h264") -> dict:
    """Largest number n of identical independent streams (one process, T=1, each on its own core or a 4/n share) that stay sustainable."""
    best, rows = 0, []
    for n in range(1, n_max + 1):
        q = min(1.0, CORES / n)
        wall, cpu, _ = tracker_model.sample(q, cd["obs"][sid], np.where(cd["obs"][sid], cd["n_in_full"], 0), cd["n_win"][sid], cd["u"], decode)
        service = wall * s
        arrival = np.arange(len(service)) * cd["dt_ms"]
        finish = fifo_finish(arrival, service)
        qm = queue_metrics(arrival, service, finish, cd["dt_ms"])
        rows.append({"n": n, "rho": qm["rho"], "sustainable": qm["sustainable"], "max_wait_ms": qm["max_wait_ms"]})
        if qm["sustainable"]:
            best = n
        elif n > best + 1:
            break
    return {"max_streams": best, "rows": rows}


# =====================================================================================================================
# 9. Simulation stage: (dataset, frontend, tracker) -> every schedule x scenario
# =====================================================================================================================
SIM_VERSION = "sim-v1"
SCEN = {s["id"]: s for s in scenario_list()}
ALL_SCEN = list(SCEN)
SECONDARY_SCEN = list(HEADLINE) + ["T4_s1.0_b1.00"]


def job_key(sid: str, scen: str | None, stream: str = "continuous", hold: str = "rule", decode: str = "h264") -> str:
    return f"{sid}|{scen or 'ideal'}|{stream}|{hold}|{decode}"


def jobs_for(dataset: str, primary: bool, tracker: str, sids: list[str]) -> list[dict]:
    jobs: list[dict] = []

    def add(sid, scen, stream="continuous", hold="rule", decode="h264", keep=False):
        jobs.append({"key": job_key(sid, scen, stream, hold, decode), "sid": sid, "scenario": SCEN[scen] if scen else None, "stream": stream,
                     "hold": hold, "decode": decode, "keep_series": keep})

    for sid in sids:
        add(sid, None, keep=primary or sid == "stride1")
        for sc in (ALL_SCEN if primary else SECONDARY_SCEN):
            add(sid, sc, keep=primary and sc in HEADLINE)
        if primary and tracker == "locked":
            for sc in ALL_SCEN:
                add(sid, sc, stream="clip_reset")
            for sc in (None, *HEADLINE):
                add(sid, sc, hold="zero")
            if dataset == "ShanghaiTech":
                for sc in HEADLINE:
                    add(sid, sc, decode="jpeg")
    return jobs


def check_d7(seg_lens: dict = SEG_LENS) -> dict:
    """The frames fed to a tracker replay are exactly the frames the schedule requests (D7)."""
    bad = []
    for ds, L in seg_lens.items():
        for s in schedule_grid(ds):
            for n in (1, 7, 48, 265, 1000):
                if not np.array_equal(trackset_mask(trackset_key(s, L), n, L), s.observed(n, L)):
                    bad.append((ds, s.id, n))
    return {"pass": not bad, "detail": "tracker input mask == schedule.observed for every schedule and dataset" if not bad else f"mismatch {bad[:3]}"}


def cache_stamp(ds: str, arm: e01.Arm, tracker: str, scorer, config: dict) -> str:
    return json.dumps({"v": SIM_VERSION, "cache": config["arms"][f"{ds}/{arm.name}"]["cache_tree_sha256"], "ckpt": e00.sha256_file(scorer.checkpoint)[:16],
                       "tracker": TRACKERS[tracker].label, "grid": [s.id for s in schedule_grid(ds)]}, sort_keys=True)


def load_tables(path: Path, stamp: str, sids: list[str]) -> dict | None:
    if not path.exists():
        return None
    z = np.load(path, allow_pickle=False)
    if str(z["__stamp__"]) != stamp:
        return None
    out: dict = {}
    for k in z.files:
        if k != "__stamp__":
            sid, col = k.split("|", 1)
            out.setdefault(sid, {})[col] = z[k]
    return out if all(s in out for s in sids) else None


def save_tables(path: Path, stamp: str, tables: dict) -> None:
    flat = {f"{sid}|{col}": v for sid, t in tables.items() for col, v in t.items()}
    np.savez_compressed(path, __stamp__=np.array(stamp), **flat)


def run_checks(cd: dict, tracker_model, log) -> dict:
    """D3 (queue invariants on real services), D4 (prefix invariance on real tables); primary arm, locked tracker."""
    out = {"D3": [], "D4": []}
    rng = np.random.default_rng(SEED)
    for sid in ("stride1", [s.id for s in cd["schedules"].values() if s.family == "stride" and s.k == 2 and s.fill == "zoh"][0]):
        for scen in ("T4_s2.0_b1.00", "T4_s3.0_b1.00"):
            service, _, _ = service_for(cd, sid, SCEN[scen])
            t = cd["tables"][sid]
            series, info = simulate_delivery(t["clip_idx"], t["last_frame"], t["normality"], cd["n_frames"], service, cd["dt_ms"], cd["sentinel"],
                                             cd["schedules"][sid].hold)
            ck = check_queue_invariants(info["arrival"], service, info["finish"], info["landing"], t["last_frame"], cd["dt_ms"], t["clip_idx"],
                                        info["offsets"])
            out["D3"].append({"sid": sid, "scenario": scen, **ck, "pass": all(ck.values())})
    sid, scen = "stride1", SCEN["T4_s2.0_b1.00"]
    service, _, _ = service_for(cd, sid, scen)
    t = cd["tables"][sid]
    full, _ = simulate_delivery(t["clip_idx"], t["last_frame"], t["normality"], cd["n_frames"], service, cd["dt_ms"], cd["sentinel"], 0)
    n_frames = np.asarray(cd["n_frames"])
    for trial in range(6):
        m = int(rng.integers(2, min(40, len(n_frames))))  # keep the first m clips, cut the last one at a random frame
        cut = int(rng.integers(1, n_frames[m - 1]))
        nf = n_frames[:m].copy()
        nf[-1] = cut
        keep = (t["clip_idx"] < m) & ~((t["clip_idx"] == m - 1) & (t["last_frame"] >= cut))
        sub, _ = simulate_delivery(t["clip_idx"][keep], t["last_frame"][keep], t["normality"][keep], nf, service[:int(nf.sum())], cd["dt_ms"],
                                   cd["sentinel"], 0)  # the truncated stream is a prefix of the flat service array
        diff = max(float(np.abs(sub[i][:nf[i]] - full[i][:nf[i]]).max()) for i in range(m))
        out["D4"].append({"clips": m, "cut_frame": cut, "max_abs_diff": diff, "pass": diff == 0.0})
    return out


def sample_traces(cd: dict, sids: list[str], max_frames: int = 400) -> list[dict]:
    """Queue traces of three clips: per frame job requested / started / finished, and where the windows it scored became visible."""
    n_frames, offsets = np.asarray(cd["n_frames"]), cd["offsets"]
    busy = int(np.argmax(per_clip_sum(cd["n_in_full"].astype(float), offsets) / n_frames))
    clips = sorted({0, int(cd["dev_idx"][0]), busy})
    lines = []
    wanted = [s for s in ("stride1", "stride2-zoh") if s in sids] + [f"burst{2 * cd['L']}"]
    for sid in wanted:
        if sid not in sids:
            continue
        t = cd["tables"][sid]
        for scen in ("T4_s2.0_b1.00", "T4_s3.0_b1.00"):
            service, _, _ = service_for(cd, sid, SCEN[scen])
            _, info = simulate_delivery(t["clip_idx"], t["last_frame"], t["normality"], n_frames, service, cd["dt_ms"], cd["sentinel"],
                                        cd["schedules"][sid].hold)
            for ci in clips:
                sel = np.nonzero(t["clip_idx"] == ci)[0]
                by_frame: dict[int, list] = {}
                for w in sel:
                    by_frame.setdefault(int(t["last_frame"][w]), []).append(int(info["landing"][w]) if info["ok"][w] else None)
                for f in range(min(int(n_frames[ci]), max_frames)):
                    g = int(offsets[ci]) + f
                    lines.append({"type": "queue_trace", "dataset_L": cd["L"], "schedule": sid, "scenario": scen, "clip": cd["clips"][ci], "frame": f,
                                  "requested_ms": round(float(info["arrival"][g]), 3), "started_ms": round(float(info["finish"][g] - service[g]), 3),
                                  "finished_ms": round(float(info["finish"][g]), 3), "service_ms": round(float(service[g]), 3),
                                  "observed": bool(cd["obs"][sid][g]), "windows_scored": len(by_frame.get(f, [])),
                                  "visible_from_frame": by_frame.get(f, [])})
    return lines


def stage_sim(ctx, ds: str, arm_name: str, tracker: str, frames_pack: tuple, procs: int, force: bool, log) -> dict:
    primary = arm_name == PRIMARY[ds]
    arm = e01.arm_by_key(ctx.arms)[f"{ds}/{arm_name}"]
    cell = TRACKERS[tracker]
    L, dt = SEG_LENS[ds], 1000.0 / FPS[ds]
    out_path = WORK_DIR / "stages" / f"e02_sim__{ds}__{arm_name}__{tracker}.pkl"
    config = yaml.safe_load((EXP_DIR / "config.yaml").read_text())
    scorer = ctx.scorers[ds]
    stamp = cache_stamp(ds, arm, tracker, scorer, config)
    stage_stamp = stamp + "|" + e00.sha256_file(EXP_DIR / "config.yaml")[:16] + "|" + str(build_pool_fingerprint(ds, arm_name, tracker))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not force:
        res = pickle.loads(out_path.read_bytes())
        if res["stamp"] == stage_stamp:
            log(f"  [{now()}] {ds}/{arm_name}/{tracker}: cached")
            return res
    names, labels = ctx.names[ds], ctx.labels[ds]
    n_frames = [len(x) for x in labels]
    offsets = np.concatenate([[0], np.cumsum(n_frames)])
    total = int(offsets[-1])
    frames_by_clip, ids_by_clip = frames_pack
    sids = schedule_ids(ds, primary, tracker)
    t0 = time.perf_counter()
    tables = load_tables(WORK_DIR / f"e02_tables__{ds}__{arm_name}__{tracker}.npz", stamp, sids)
    if tables is None:
        wanted = {trackset_key(s, L) for s in schedule_grid(ds) if s.id in sids} | {("stride", 1)}
        log(f"  [{now()}] {ds}/{arm_name}/{tracker}: replaying {len(wanted)} tracksets on {len(names)} clips ...")
        tracks = replay_tracksets(frames_by_clip, ids_by_clip, cell, sorted(wanted), L, procs)
        log(f"  [{now()}]   replay {time.perf_counter() - t0:.0f}s; scoring windows ...")
        tables = build_tables(tracks, n_frames, L, scorer, sids, ds, log)
        del tracks
        save_tables(WORK_DIR / f"e02_tables__{ds}__{arm_name}__{tracker}.npz", stamp, tables)
    log(f"  [{now()}]   tables ready ({time.perf_counter() - t0:.0f}s)")
    n_in_full = np.zeros(total, dtype=np.int64)
    for ci, (fr, ids) in enumerate(zip(frames_by_clip, ids_by_clip)):
        n_in_full[offsets[ci] + ids] = [int((f["conf"] >= cell.gate).sum()) for f in fr]
    by_id = {s.id: s for s in schedule_grid(ds)}
    obs = {sid: np.concatenate([by_id[sid].observed(n, L) for n in n_frames]) for sid in sids}
    n_win = {sid: window_counts(tables[sid], offsets, total) for sid in sids}
    model = ServiceModel(build_pools(ds, arm_name, tracker))
    sentinel = e01.arm_sentinel(ctx.proto, arm)
    cd = {"n_frames": n_frames, "offsets": offsets, "labels": labels, "dev_idx": ctx.dev_idx[ds], "conf_idx": ctx.conf_idx[ds], "clips": names,
          "sentinel": sentinel, "dt_ms": dt, "L": L, "tables": tables, "obs": obs, "n_in_full": n_in_full, "n_win": n_win, "model": model,
          "u": np.random.default_rng(POOL_SEED).random((total, 5)), "schedules": {sid: by_id[sid] for sid in sids}}
    # --- reference (E00 / E01 engine on the K=1 table): D1 / D2 -------------------------------------------------------
    stub = e00.Source("e02", ds, Path("."), None, None, None, "")
    d = e00.SourceData(stub, names, labels, tables["stride1"], L, ctx.dev_idx[ds], ctx.conf_idx[ds])
    ref_series, _ = e00.deliver(d.raw_per_frame("completion", "all"), sentinel, 0)
    s0_series, _ = d.step_scores("S0")
    ref = {"S4": e01.auc_subsets(labels, ref_series, ctx.dev_idx[ds], ctx.conf_idx[ds]),
           "S0": e01.auc_subsets(labels, s0_series, ctx.dev_idx[ds], ctx.conf_idx[ds]),
           "S0_clip_auc": [100 * e00.clip_auc(labels[i], s0_series[i]) for i in range(len(labels))],
           "ref_series": np.concatenate(ref_series)}
    jobs = jobs_for(ds, primary, tracker, sids)
    log(f"  [{now()}]   {len(jobs)} delivery simulations on {procs} processes ...")
    _G["cd"] = cd
    try:
        results = e01.pmap(run_job, jobs, procs)
    finally:
        _G.pop("cd", None)
    res = {"stamp": stage_stamp, "dataset": ds, "arm": arm_name, "tracker": tracker, "primary": primary, "sentinel": sentinel, "ref": ref,
           "jobs": {j["key"]: r for j, r in zip(jobs, results)}, "sids": sids, "n_frames": n_frames,
           "table_stats": {sid: {"windows": int(len(tables[sid]["normality"])), "genuine_frac_mean": float(tables[sid]["genuine_frac"].mean())
                                 if len(tables[sid]["normality"]) else float("nan"),
                                 "genuine_frac_min": float(tables[sid]["genuine_frac"].min()) if len(tables[sid]["normality"]) else float("nan"),
                                 "clips_with_windows": int(len(np.unique(tables[sid]["clip_idx"])))} for sid in sids},
           "obs_fraction": {sid: float(obs[sid].mean()) for sid in sids}}
    if primary:
        res["max_streams"] = {sid: {f"{s:.1f}": max_streams(cd, sid, s, model) for s in S_GRID} for sid in sids}
    if primary and tracker == "locked":
        res["checks"] = run_checks(cd, model, log)
        res["traces"] = sample_traces(cd, sids)
    out_path.write_bytes(pickle.dumps(res))
    log(f"  [{now()}]   {ds}/{arm_name}/{tracker} done in {time.perf_counter() - t0:.0f}s")
    return res


def build_pool_fingerprint(ds: str, arm_name: str, tracker: str) -> str:
    pools = build_pools(ds, arm_name, tracker)
    h = hashlib.sha256()
    for k in sorted(pools, key=str):
        h.update(str(k).encode())
        h.update(np.asarray(pools[k][0]).tobytes()[:4096])
        h.update(str(float(np.sum(pools[k][0]))).encode())
    return h.hexdigest()[:16]


def stage_all_sims(procs: int, force: bool, log) -> dict[tuple, dict]:
    ctx = e01.build_ctx(procs, log)
    out: dict[tuple, dict] = {}
    for ds in DATASETS:
        for arm_name in FRONTENDS[ds]:
            arm = e01.arm_by_key(ctx.arms)[f"{ds}/{arm_name}"]
            frames_pack = None
            for tracker in (TRACKERS if arm_name == PRIMARY[ds] else ("locked",)):
                if frames_pack is None:
                    log(f"[{now()}] loading detector cache {short(arm.cache_dir)} ...")
                    frames_pack = e01.load_frames_floor(arm.cache_dir, ctx.names[ds], GATE_FLOOR)
                out[(ds, arm_name, tracker)] = stage_sim(ctx, ds, arm_name, tracker, frames_pack, procs, force, log)
    return out


# =====================================================================================================================
# 10. Analysis: contrasts, rules, selections
# =====================================================================================================================
_BOOT: dict = {}


def split_flat(flat: np.ndarray, n_frames: list[int]) -> list[np.ndarray]:
    offs = np.concatenate([[0], np.cumsum(n_frames)])
    return [flat[offs[i]:offs[i + 1]] for i in range(len(n_frames))]


def jget(R: dict, ds: str, arm: str, tracker: str, sid: str, scen: str | None = None, stream: str = "continuous", hold: str = "rule",
         decode: str = "h264") -> dict:
    return R[(ds, arm, tracker)]["jobs"][job_key(sid, scen, stream, hold, decode)]


def _boot_job(job: tuple):
    kind, ds, a, b, subset, n = job
    labels, idx = _BOOT["labels"][ds], _BOOT["idx"][ds][subset]
    if kind == "ci":
        return job, e00.bootstrap_ci(labels, _BOOT["series"][a], idx, n)
    return job, e00.paired_diff(labels, _BOOT["series"][a], _BOOT["series"][b], idx, n)  # AUC(a) - AUC(b)


def run_bootstrap(jobs: list[tuple], series: dict, ctx, procs: int, log) -> dict:
    _BOOT.update(labels=ctx.labels, series=series, idx={ds: {"all": None, "dev": ctx.dev_idx[ds], "confirmation": ctx.conf_idx[ds]} for ds in DATASETS})
    try:
        out = dict(e01.pmap(_boot_job, jobs, procs))
    finally:
        _BOOT.clear()
    return out


def series_of(R: dict, ds: str, arm: str, tracker: str, sid: str, scen: str | None) -> list[np.ndarray]:
    return split_flat(jget(R, ds, arm, tracker, sid, scen)["series"], R[(ds, arm, tracker)]["n_frames"])


def best_propagated(R: dict, ds_pick: str, k: int) -> str:
    """Propagation variant (zoh / cv) of stride K with the higher ideal-timing dev AUC on `ds_pick` (primary arm, locked tracker)."""
    arm = PRIMARY[ds_pick]
    cands = {f: jget(R, ds_pick, arm, "locked", f"stride{k}-{f}", None)["auc"]["dev"] for f in ("zoh", "cv")}
    return f"stride{k}-" + max(cands, key=cands.get)


def contrast_jobs(R: dict) -> tuple[list[tuple], dict]:
    """(job tuples, series dict). Everything uses the primary arm and series kept by the simulation stage (ideal + headline scenarios)."""
    jobs, series = [], {}

    def put(ds, arm, tracker, sid, scen) -> str:
        name = f"{ds}|{arm}|{tracker}|{sid}|{scen or 'ideal'}"
        if name not in series:
            series[name] = series_of(R, ds, arm, tracker, sid, scen)
        return name

    seen = set()

    def add(kind, ds, a, b, subset):
        j = (kind, ds, a, b, subset, N_BOOT)
        if j not in seen:
            seen.add(j)
            jobs.append(j)

    sht_pick = {k: best_propagated(R, "ShanghaiTech", k) for k in (2, 3, 4)}
    for ds in DATASETS:
        arm, L = PRIMARY[ds], SEG_LENS[ds]
        subsets = ("all", "dev", "confirmation") if ds == "ShanghaiTech" else ("all",)
        for scen in (None, "T4_s2.0_b1.00"):
            for k in (2, 3, 4):
                burst, stride = put(ds, arm, "locked", f"burst{k * L}", scen), put(ds, arm, "locked", sht_pick[k], scen)
                for sub in subsets:
                    add("diff", ds, burst, stride, sub)  # H1: burst - best (SHT-picked) stride
        for scen in (None,):
            for sid in R[(ds, arm, "locked")]["sids"]:
                lo, de = put(ds, arm, "locked", sid, scen), put(ds, arm, "default", sid, scen)
                add("diff", ds, lo, de, "all")  # H2
                if ds == "ShanghaiTech":
                    add("diff", ds, lo, de, "dev")
        for scen in ("T4_s2.0_b1.00", "T4_s3.0_b1.00"):  # delivery cost at K=1 (delivered - ideal)
            add("diff", ds, put(ds, arm, "locked", "stride1", scen), put(ds, arm, "locked", "stride1", None), "all")
    for ds, other in (("ShanghaiTech", "stock"), ("ShanghaiTech", "stock_ncnn800"), ("UBnormal", "stock")):  # frontend contrast at K=1 / ideal
        arm = PRIMARY[ds]
        a, b = put(ds, arm, "locked", "stride1", None), put(ds, other, "locked", "stride1", None)
        for sub in (("all", "dev", "confirmation") if ds == "ShanghaiTech" else ("all",)):
            add("diff", ds, a, b, sub)
    for ds in DATASETS:  # CI of the K=1 reference
        for tr in TRACKERS:
            for scen in (None, "T4_s2.0_b1.00", "T4_s3.0_b1.00"):
                add("ci", ds, put(ds, PRIMARY[ds], tr, "stride1", scen), "", "all")
    return jobs, series


def b1_rows(R: dict, ds: str, scen: str, tracker: str = "locked") -> list[dict]:
    arm = PRIMARY[ds]
    rows = []
    by_id = {s.id: s for s in schedule_grid(ds)}
    for sid in R[(ds, arm, tracker)]["sids"]:
        if by_id[sid].alias_of or R[(ds, arm, tracker)]["table_stats"][sid]["windows"] == 0:
            continue  # burst1 is stride1 itself; schedules that produce no window at all (no-fill stride, D5) are not candidates
        r = jget(R, ds, arm, tracker, sid, scen)
        rows.append({"schedule": sid, "sustainable": r["sustainable"], "dev_auc": r["auc"]["dev"], "all_auc": r["auc"]["all"],
                     "confirmation_auc": r["auc"]["confirmation"], "cpu_ms_per_s": r["cpu_ms_per_s"], "plan_compliant": by_id[sid].plan_compliant,
                     "rho": r["rho"], "max_wait_ms": r["max_wait_ms"], "p99_latency_ms": r["p99_latency_ms"], "family": by_id[sid].family,
                     "duty": by_id[sid].duty(SEG_LENS[ds])})
    return rows


def grid_rows(R: dict) -> list[dict]:
    """Aggregated (dataset, arm, tracker, schedule, scenario, ...) table (no per-video values)."""
    rows = []
    for (ds, arm, tracker), res in R.items():
        by_id = {s.id: s for s in schedule_grid(ds)}
        for k, r in res["jobs"].items():
            sid, scen, stream, hold, decode = k.split("|")
            s = by_id[sid]
            sc = SCEN.get(scen, {"s": 0.0, "b": float("nan"), "q": float("nan")})
            rows.append({"dataset": ds, "arm": arm, "tracker": tracker, "schedule": sid, "family": s.family, "k": s.k, "fill": s.fill,
                         "propagated": s.propagated, "duty": s.duty(SEG_LENS[ds]), "scenario": scen, "s": sc["s"], "b": sc["b"], "stream": stream,
                         "hold_rule": hold, "decode": decode, "auc_all": r["auc"]["all"], "auc_dev": r["auc"]["dev"], "auc_confirmation": r["auc"]["confirmation"],
                         "sustainable": r["sustainable"], "rho": r["rho"], "max_wait_ms": r["max_wait_ms"], "p99_latency_ms": r["p99_latency_ms"],
                         "final_wait_ms": r["final_wait_ms"], "late_frame_fraction": r["late_frame_fraction"], "cpu_ms_per_s": r["cpu_ms_per_s"],
                         "service_ms_per_s": r["service_ms_per_s"], "windows": r["windows"], "windows_delivered": r["windows_delivered"],
                         "dropped_late": r["dropped_late"], "dropped_end": r["dropped_end"], "coverage": r["coverage"],
                         "emit_latency_ms_mean": r["emit_latency_ms_mean"], "emit_latency_ms_p95": r["emit_latency_ms_p95"],
                         "sentinel_fraction": r["stats"]["sentinel_fraction"], "held_fraction": r["stats"]["held_fraction"],
                         "mean_age_frames": r["stats"]["mean_age_frames"], "genuine_frac_mean": r["genuine_frac_mean"]})
    return rows


def evaluate_rules(R: dict, vm_rules: dict, boot: dict, e01sum: dict, checks_d7: dict) -> dict:
    rules = {k: {"status": "PASS" if v["pass"] else "FAIL", **v} for k, v in vm_rules.items()}
    # D1 ------------------------------------------------------------------------------------------------------------
    diffs, lines = [], []
    tv = e01sum["tracker_verification"]["series"]
    for (ds, arm, tracker), res in R.items():
        key = f"{ds}/{arm}"
        label = TRACKERS[tracker].label
        ref = tv.get(key, {}).get("cells", {}).get(label)
        if ref is None:
            continue
        for step in ("S4", "S0"):
            diffs.append(abs(res["ref"][step]["all"] - ref["auc"][step]["all"]))
        lines.append(f"{key} {tracker}: S4 {res['ref']['S4']['all']:.2f} (E01 {ref['auc']['S4']['all']:.2f})")
    rules["D1"] = {"pass": bool(diffs) and max(diffs) <= 0.02,
                   "detail": f"max |dAUC| vs E01 (S4 and S0, all clips) = {max(diffs, default=float('nan')):.4f} pt over {len(diffs)} checks; " + "; ".join(lines[:4]),
                   "max_diff_pt": max(diffs) if diffs else None}
    # D2 ------------------------------------------------------------------------------------------------------------
    eq = {f"{ds}/{arm}/{tr}": bool(np.array_equal(res["ref"]["ref_series"], jget(R, ds, arm, tr, "stride1", None)["series"])) for (ds, arm, tr), res in R.items()}
    rules["D2"] = {"pass": all(eq.values()), "detail": f"zero-service delivered series == E00 completion-landing series (bit-exact) for {sum(eq.values())}/{len(eq)} (dataset, frontend, tracker) cells",
                   "per_cell": eq}
    # D3 / D4 -------------------------------------------------------------------------------------------------------
    d3, d4 = [], []
    for (ds, arm, tr), res in R.items():
        for c in res.get("checks", {}).get("D3", []):
            d3.append({"dataset": ds, **c})
        for c in res.get("checks", {}).get("D4", []):
            d4.append({"dataset": ds, **c})
    rules["D3"] = {"pass": bool(d3) and all(c["pass"] for c in d3), "detail": f"{len(d3)} real-service checks (FIFO == reference loop, start >= arrival, finish monotone, landing >= completion and >= ceil(finish/dt))", "rows": d3}
    rules["D4"] = {"pass": bool(d4) and all(c["pass"] for c in d4), "detail": f"{len(d4)} random prefix cuts: max |delivered score difference| = {max((c['max_abs_diff'] for c in d4), default=float('nan')):.1e}", "rows": d4}
    # D5 / D6 -------------------------------------------------------------------------------------------------------
    nofill = {}
    for (ds, arm, tr), res in R.items():
        for k in (2, 3, 4):
            sid = f"stride{k}-none"
            if sid in res["table_stats"]:
                nofill[f"{ds}/{arm}/{tr}/{sid}"] = res["table_stats"][sid]["windows"]
    rules["D5"] = {"pass": bool(nofill) and all(v == 0 for v in nofill.values()), "detail": f"windows produced by no-fill stride K >= 2: {sorted(set(nofill.values()))} over {len(nofill)} cells", "cells": nofill}
    g = {}
    for (ds, arm, tr), res in R.items():
        for sid, st in res["table_stats"].items():
            if sid.startswith("burst") and int(sid[5:]) > SEG_LENS[ds] or "-zoh" in sid or "-cv" in sid:
                g[f"{ds}/{arm}/{tr}/{sid}"] = st
    burst_ok = all(abs(v["genuine_frac_min"] - 1.0) < 1e-12 for k_, v in g.items() if "burst" in k_ and v["windows"] > 0)
    zoh = {k_: v["genuine_frac_mean"] for k_, v in g.items() if "-zoh" in k_ and "/ML0" in k_ and k_.endswith(("-zoh",)) and "/locked/" in k_}
    rules["D6"] = {"pass": burst_ok, "detail": "burst windows (W > L): genuine fraction min = 1.0 -> " + str(burst_ok) + "; stride-ZOH mean genuine fraction: " +
                   ", ".join(f"{k_.split('/')[0][:3]} {k_.split('/')[-1]} {v:.2f}" for k_, v in sorted(zoh.items())), "zoh": zoh}
    rules["D7"] = {"status": "PASS" if checks_d7["pass"] else "FAIL", **checks_d7}
    for k_ in ("D1", "D2", "D3", "D4", "D5", "D6"):
        rules[k_]["status"] = "PASS" if rules[k_]["pass"] else "FAIL"
    return rules


# =====================================================================================================================
# 11. Summary assembly
# =====================================================================================================================
def contrast(boot: dict, ds: str, a: str, b: str, subset: str) -> dict | None:
    return boot.get(("diff", ds, a, b, subset, N_BOOT))


def sname(ds: str, arm: str, tracker: str, sid: str, scen: str | None) -> str:
    return f"{ds}|{arm}|{tracker}|{sid}|{scen or 'ideal'}"


def build_summary(R: dict, boot: dict, vm: dict, rules: dict, e01sum: dict) -> dict:
    S: dict = {"experiment": "Q2-P1-E02_fixed_baseline", "machine": MACHINE_TAG, "seed": SEED, "environment": e00.env_info(),
               "physical_pi5": "not connected (D8/D10); all service times are VM measurements, Pi numbers are s-multiplier sensitivities"}
    S["vm"] = vm
    S["reference"] = {f"{ds}/{arm}/{tr}": {"S4": res["ref"]["S4"], "S0": res["ref"]["S0"]} for (ds, arm, tr), res in R.items()}
    # --- headline grid (primary frontend, locked tracker) ----------------------------------------------------------------
    S["schedules"] = {ds: [{"id": s.id, "family": s.family, "k": s.k, "fill": s.fill, "duty": s.duty(SEG_LENS[ds]), "propagated": s.propagated,
                            "plan_compliant": s.plan_compliant, "hold": s.hold} for s in schedule_grid(ds)] for ds in DATASETS}
    head: dict = {}
    for ds in DATASETS:
        arm = PRIMARY[ds]
        for tr in TRACKERS:
            res = R[(ds, arm, tr)]
            for sid in res["sids"]:
                row = {"ideal": jget(R, ds, arm, tr, sid, None)["auc"]}
                row["windows"] = res["table_stats"][sid]["windows"]
                row["genuine_frac_mean"] = res["table_stats"][sid]["genuine_frac_mean"]
                for sc in ALL_SCEN:
                    r = jget(R, ds, arm, tr, sid, sc)
                    row[sc] = {"auc": r["auc"], "sustainable": r["sustainable"], "rho": r["rho"], "max_wait_ms": r["max_wait_ms"],
                               "p99_latency_ms": r["p99_latency_ms"], "cpu_ms_per_s": r["cpu_ms_per_s"], "service_ms_per_s": r["service_ms_per_s"],
                               "emit_latency_ms_mean": r["emit_latency_ms_mean"], "coverage": r["coverage"], "dropped_late": r["dropped_late"],
                               "dropped_end": r["dropped_end"], "windows_delivered": r["windows_delivered"], "late_frame_fraction": r["late_frame_fraction"],
                               "sentinel_fraction": r["stats"]["sentinel_fraction"], "mean_age_frames": r["stats"]["mean_age_frames"],
                               "parts_ms_per_s": r.get("parts_ms_per_s", {})}
                head.setdefault(f"{ds}/{arm}/{tr}", {})[sid] = row
    S["headline"] = head
    # --- selections (B1) ---------------------------------------------------------------------------------------------------
    b1, pareto = {}, {}
    for sc in HEADLINE:
        rows = b1_rows(R, "ShanghaiTech", sc)
        ref_all, ref_comp = select_fixed_reference(rows), select_fixed_reference(rows, require_compliant=True)
        rows_ub = b1_rows(R, "UBnormal", sc)
        b1[sc] = {"reference": ref_all, "plan_compliant_reference": ref_comp, "n_sustainable": int(sum(r["sustainable"] for r in rows)),
                  "rows": rows, "ub_rows": rows_ub, "stride1_row": next(r for r in rows if r["schedule"] == "stride1")}
        if ref_all:
            rid = ref_all["schedule"]
            b1[sc]["ub_same_schedule"] = next((r for r in rows_ub if r["schedule"] == rid), None)
    S["B1"] = b1
    for ds in DATASETS:
        for sc in ("T4_s2.0_b1.00", "T4_s3.0_b1.00", "T4_s2.0_b0.50"):
            rows = [r for r in b1_rows(R, ds, sc) if r["sustainable"]]
            names = pareto_front([(r["cpu_ms_per_s"], r["all_auc"], r["schedule"]) for r in rows])
            pareto[f"{ds}|{sc}"] = [r for r in rows if r["schedule"] in names]
    S["pareto"] = pareto
    # --- max streams and headroom ------------------------------------------------------------------------------------------
    S["max_streams"] = {ds: {sid: {s: v["max_streams"] for s, v in per.items()} for sid, per in R[(ds, PRIMARY[ds], "locked")]["max_streams"].items()}
                        for ds in DATASETS}
    S["headroom"] = {ds: {sid: {"rho_s1": head[f"{ds}/{PRIMARY[ds]}/locked"][sid]["T4_s1.0_b1.00"]["rho"],
                                "s_star": (1.0 / head[f"{ds}/{PRIMARY[ds]}/locked"][sid]["T4_s1.0_b1.00"]["rho"]
                                           if head[f"{ds}/{PRIMARY[ds]}/locked"][sid]["T4_s1.0_b1.00"]["rho"] > 0 else float("inf"))}
                         for sid in R[(ds, PRIMARY[ds], "locked")]["sids"]} for ds in DATASETS}
    # --- contrasts -----------------------------------------------------------------------------------------------------------
    h1 = {}
    sht_pick = {k: best_propagated(R, "ShanghaiTech", k) for k in (2, 3, 4)}
    for ds in DATASETS:
        L, arm = SEG_LENS[ds], PRIMARY[ds]
        for scen in (None, "T4_s2.0_b1.00"):
            for k in (2, 3, 4):
                burst, stride = sname(ds, arm, "locked", f"burst{k * L}", scen), sname(ds, arm, "locked", sht_pick[k], scen)
                for sub in (("all", "dev", "confirmation") if ds == "ShanghaiTech" else ("all",)):
                    c = contrast(boot, ds, burst, stride, sub)
                    ja = jget(R, ds, arm, "locked", f"burst{k * L}", scen)
                    jb = jget(R, ds, arm, "locked", sht_pick[k], scen)
                    h1[f"{ds}|{scen or 'ideal'}|K{k}|{sub}"] = {"burst": f"burst{k * L}", "stride": sht_pick[k], "burst_auc": ja["auc"][sub], "stride_auc": jb["auc"][sub],
                                                                 **c, "verdict": h1_decision(c)}
    S["H1"] = h1
    h2 = {}
    for ds in DATASETS:
        arm = PRIMARY[ds]
        for sid in R[(ds, arm, "locked")]["sids"]:
            c = contrast(boot, ds, sname(ds, arm, "locked", sid, None), sname(ds, arm, "default", sid, None), "all")
            h2[f"{ds}|{sid}"] = {**c, "locked_auc": jget(R, ds, arm, "locked", sid, None)["auc"]["all"],
                                 "default_auc": jget(R, ds, arm, "default", sid, None)["auc"]["all"]}
            if ds == "ShanghaiTech":
                h2[f"{ds}|{sid}"]["dev"] = contrast(boot, ds, sname(ds, arm, "locked", sid, None), sname(ds, arm, "default", sid, None), "dev")
    S["H2"] = h2
    k1 = {f"{ds}|{sc or 'ideal'}": {tr: jget(R, ds, PRIMARY[ds], tr, "stride1", sc)["auc"] for tr in TRACKERS} for ds in DATASETS for sc in (None, "T4_s2.0_b1.00")}
    S["k1_auc"] = k1
    S["k1_ci"] = {f"{ds}|{tr}|{sc or 'ideal'}": boot.get(("ci", ds, sname(ds, PRIMARY[ds], tr, "stride1", sc), "", "all", N_BOOT))
                  for ds in DATASETS for tr in TRACKERS for sc in (None, "T4_s2.0_b1.00", "T4_s3.0_b1.00")}
    S["delivery_cost_k1"] = {f"{ds}|{sc}": contrast(boot, ds, sname(ds, PRIMARY[ds], "locked", "stride1", sc), sname(ds, PRIMARY[ds], "locked", "stride1", None), "all")
                             for ds in DATASETS for sc in ("T4_s2.0_b1.00", "T4_s3.0_b1.00")}
    S["frontend_contrasts"] = {}
    for ds, other in (("ShanghaiTech", "stock"), ("ShanghaiTech", "stock_ncnn800"), ("UBnormal", "stock")):
        a, b = sname(ds, PRIMARY[ds], "locked", "stride1", None), sname(ds, other, "locked", "stride1", None)
        S["frontend_contrasts"][f"{ds}|{PRIMARY[ds]}-{other}"] = {sub: contrast(boot, ds, a, b, sub) for sub in (("all", "dev", "confirmation") if ds == "ShanghaiTech" else ("all",))}
    S["frontends_k1"] = {f"{ds}/{arm}": {"ideal": jget(R, ds, arm, "locked", "stride1", None)["auc"],
                                         **{sc: {"auc": jget(R, ds, arm, "locked", "stride1", sc)["auc"], "sustainable": jget(R, ds, arm, "locked", "stride1", sc)["sustainable"],
                                                 "rho": jget(R, ds, arm, "locked", "stride1", sc)["rho"], "cpu_ms_per_s": jget(R, ds, arm, "locked", "stride1", sc)["cpu_ms_per_s"]}
                                            for sc in SECONDARY_SCEN}} for ds in DATASETS for arm in FRONTENDS[ds]}
    # --- H3 / sensitivities ----------------------------------------------------------------------------------------------------
    h3 = {}
    for ds in DATASETS:
        L, arm = SEG_LENS[ds], PRIMARY[ds]
        base = jget(R, ds, arm, "locked", "stride1", "T4_s1.0_b1.00")
        for w in [w for w in BURST_W[ds] if 1 < w <= L]:
            r = jget(R, ds, arm, "locked", f"burst{w}", "T4_s1.0_b1.00")
            h3[f"{ds}|burst{w}"] = {"duty": 1.0, "cpu_ms_per_s": r["cpu_ms_per_s"], "cpu_vs_k1": r["cpu_ms_per_s"] / base["cpu_ms_per_s"], "score_ms_per_s": r["parts_ms_per_s"].get("score_ms"),
                                    "k1_score_ms_per_s": base["parts_ms_per_s"].get("score_ms"), "auc_ideal_all": jget(R, ds, arm, "locked", f"burst{w}", None)["auc"]["all"]}
    S["H3"] = h3
    sens = {}
    for ds in DATASETS:
        arm = PRIMARY[ds]
        for sid in R[(ds, arm, "locked")]["sids"]:
            for sc in HEADLINE:
                base = jget(R, ds, arm, "locked", sid, sc)
                e = {"continuous": base["auc"]["all"], "clip_reset": jget(R, ds, arm, "locked", sid, sc, stream="clip_reset")["auc"]["all"],
                     "hold0": jget(R, ds, arm, "locked", sid, sc, hold="zero")["auc"]["all"],
                     "sustainable_clip_reset": jget(R, ds, arm, "locked", sid, sc, stream="clip_reset")["sustainable"],
                     "sustainable": base["sustainable"]}
                if ds == "ShanghaiTech":
                    j = jget(R, ds, arm, "locked", sid, sc, decode="jpeg")
                    e.update({"jpeg_auc": j["auc"]["all"], "jpeg_sustainable": j["sustainable"], "jpeg_rho": j["rho"], "h264_rho": base["rho"]})
                sens[f"{ds}|{sid}|{sc}"] = e
            sens[f"{ds}|{sid}|ideal_hold0"] = jget(R, ds, arm, "locked", sid, None, hold="zero")["auc"]["all"]
    S["sensitivity"] = sens
    S["traces_n"] = {f"{ds}": len(R[(ds, PRIMARY[ds], "locked")].get("traces", [])) for ds in DATASETS}
    S["rules"] = rules
    S["milestone1_refs"] = {"E01_locked_S4": e01sum["locked_for_E02"]["frontend"]["S4_ShanghaiTech"], "E01_locked_S4_UB_proxy": e01sum["locked_for_E02"]["frontend"]["S4_UBnormal_proxy"]}
    return plain(S)


# =====================================================================================================================
# 12. Outputs and commands
# =====================================================================================================================
CSV_COLUMNS = ["dataset", "video", "split", "frontend", "tracker", "schedule", "family", "k", "fill", "propagated", "detector_duty", "scenario", "s",
               "b", "stream", "machine", "n_frames", "fps", "duration_s", "anomalous_frames", "auc_delivered", "auc_ideal", "auc_offline_S0",
               "delay_ms", "backlog_max_wait_ms", "cpu_ms_per_video_s", "peak_rss_mb", "sustainable"]


def vm_summary(log=print) -> dict:
    manifest = load_manifest()
    pose, e2e = pose_rows(), e2e_rows()
    p3 = [p3_prediction(t) for t in THREADS]
    rules = profiling_rules(manifest, pose, p3)
    loads = [r["host_outside_vm_pct"] for r in manifest.values() if r.get("host_outside_vm_pct") is not None]
    rss = {f"{r['dataset']}|{r['source']}|T{r['threads']}": r["peak_rss_mb"] for r in e2e}
    return {"pose": pose, "decode": decode_rows(), "track": track_rows(), "score": score_rows(), "e2e": e2e, "p3": p3, "rules": rules,
            "manifest": {"cells": len(manifest), "attempts_total": sum(1 for _ in (VM_OUT / "manifest.jsonl").read_text().splitlines()),
                         "host_outside_vm_mean_pct": float(np.mean(loads)), "host_outside_vm_max_pct": float(np.max(loads)),
                         "total_seconds": float(sum(r["seconds"] for r in manifest.values())), "excluded": [c for c, r in manifest.items() if r.get("excluded")]},
            "rss_e2e_mb": rss}


def write_results_csv(R: dict, ctx, vm: dict) -> int:
    rss = {"ShanghaiTech": vm["rss_e2e_mb"].get("ShanghaiTech|h264|T4"), "UBnormal": vm["rss_e2e_mb"].get("UBnormal|h264|T4")}
    n = 0
    with open(EXP_DIR / "results.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        w.writeheader()
        for (ds, arm, tracker), res in R.items():
            primary = arm == PRIMARY[ds]
            by_id = {s.id: s for s in schedule_grid(ds)}
            scens = [None, "T4_s2.0_b1.00", "T4_s3.0_b1.00"] if (primary and tracker == "locked") else [None, "T4_s2.0_b1.00"]
            sids = res["sids"] if primary else ["stride1"]
            dev = set(ctx.dev_idx[ds])
            for sid in sids:
                s = by_id[sid]
                ideal = jget(R, ds, arm, tracker, sid, None)["clip_auc"]
                for sc in scens:
                    r = jget(R, ds, arm, tracker, sid, sc)
                    sci = SCEN.get(sc, {"s": 0.0, "b": 1.0})
                    for i, clip in enumerate(ctx.names[ds]):
                        lab = ctx.labels[ds][i]
                        w.writerow({"dataset": ds, "video": clip, "split": "dev" if i in dev else "confirmation", "frontend": arm, "tracker": tracker,
                                    "schedule": sid, "family": s.family, "k": s.k, "fill": s.fill, "propagated": s.propagated,
                                    "detector_duty": round(s.duty(SEG_LENS[ds]), 4), "scenario": sc or "ideal", "s": sci["s"], "b": sci["b"],
                                    "stream": "continuous", "machine": MACHINE_TAG, "n_frames": len(lab), "fps": FPS[ds],
                                    "duration_s": round(len(lab) / FPS[ds], 3), "anomalous_frames": int(lab.sum()),
                                    "auc_delivered": _r(r["clip_auc"][i]), "auc_ideal": _r(ideal[i]), "auc_offline_S0": _r(res["ref"]["S0_clip_auc"][i]),
                                    "delay_ms": _r(r["clip_delay_ms"][i]), "backlog_max_wait_ms": _r(r["clip_wait_max_ms"][i]),
                                    "cpu_ms_per_video_s": _r(r["clip_cpu_ms_per_s"][i]), "peak_rss_mb": rss[ds], "sustainable": r["sustainable"]})
                        n += 1
    return n


def _r(x, nd: int = 4):
    return "" if x is None or (isinstance(x, float) and math.isnan(x)) else round(float(x), nd)


def write_grid_csv(R: dict) -> int:
    rows = grid_rows(R)
    with open(EXP_DIR / "grid.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (_r(v, 5) if isinstance(v, float) else v) for k, v in r.items()})
    return len(rows)


def write_decisions(R: dict, S: dict) -> int:
    n = 0
    with open(EXP_DIR / "decisions.jsonl", "w") as fh:
        def put(obj):
            nonlocal n
            fh.write(json.dumps(plain(obj)) + "\n")
            n += 1

        for rid, r in S["rules"].items():
            put({"type": "rule", "id": rid, "status": r["status"], "detail": r.get("detail")})
        for sc, b in S["B1"].items():
            put({"type": "selection", "id": "B1", "scenario": sc, "reference": b["reference"], "plan_compliant_reference": b["plan_compliant_reference"],
                 "n_sustainable": b["n_sustainable"], "stride1": b["stride1_row"]})
        for k, v in S["H1"].items():
            put({"type": "H1", "key": k, **v})
        for ds in DATASETS:
            for line in R[(ds, PRIMARY[ds], "locked")].get("traces", []):
                put({**line, "dataset": ds})
    return n


def cmd_vm(quick: bool = False, log=print) -> None:
    from edgelab.analysis import q2_p1_e02_vm as vm

    if not (EXP_DIR / "config.yaml").exists():
        raise SystemExit("run `lock` first (config.yaml must exist before any VM number)")
    vm.prepare_staging(log)
    vm.drive(quick=quick, log=log)


def cmd_run(procs: int, force: bool, log=print) -> dict:
    if not (EXP_DIR / "config.yaml").exists():
        raise SystemExit("run `lock` first")
    t0 = time.perf_counter()
    ctx = e01.build_ctx(procs, log)
    e01sum = json.loads((E01_DIR / "summary.json").read_text())
    log(f"[{now()}] VM traces ...")
    vm = vm_summary(log)
    R = stage_all_sims(procs, force, log)
    log(f"[{now()}] bootstrap contrasts ...")
    jobs, series = contrast_jobs(R)
    boot = run_bootstrap(jobs, series, ctx, procs, log)
    log(f"[{now()}] rules ...")
    rules = evaluate_rules(R, vm["rules"], boot, e01sum, check_d7())
    S = build_summary(R, boot, vm, rules, e01sum)
    S["runtime_seconds"] = time.perf_counter() - t0
    n_csv = write_results_csv(R, ctx, vm)
    n_grid = write_grid_csv(R)
    n_dec = write_decisions(R, S)
    S["outputs"] = {"results_csv_rows": n_csv, "grid_csv_rows": n_grid, "decisions_jsonl_lines": n_dec}
    (EXP_DIR / "summary.json").write_text(json.dumps(S, indent=1, default=str))
    log(f"[{now()}] wrote summary.json, results.csv ({n_csv} rows), grid.csv ({n_grid}), decisions.jsonl ({n_dec}) in {S['runtime_seconds']:.0f}s")
    return S


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("cmd", choices=["lock", "vm", "run", "report", "all"])
    p.add_argument("--procs", type=int, default=max(1, min(8, (os.cpu_count() or 4) - 2)))
    p.add_argument("--force", action="store_true", help="ignore cached simulation stages")
    p.add_argument("--quick", action="store_true", help="vm: only a handful of cells (smoke test)")
    args = p.parse_args()
    if args.cmd in ("lock", "all"):
        cmd_lock()
    if args.cmd in ("vm", "all"):
        cmd_vm(args.quick)
    if args.cmd in ("run", "all"):
        cmd_run(args.procs, args.force)
    if args.cmd in ("report", "all"):
        from edgelab.analysis.q2_p1_e02_report import render_report

        render_report()


if __name__ == "__main__":
    main()
