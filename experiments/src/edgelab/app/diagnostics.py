"""Failure-case analysis behind the Streamlit "Failure Case Inspector" tab.

Everything here READS frozen artefacts that already exist on disk (Q2-P1 result tables, simulation pickles, E01 per-frame series, detector caches,
the physical Raspberry Pi 5 live traces). Nothing is trained or measured and nothing talks to the Pi 5.

Failure modes (one ranking function + one diagnostic template each):

    small_person     `stock` misses small anomalous actors that `ML0` finds            (E01 per-frame S4 series + detector caches)
    fragmentation    default BoT-SORT breaks tracks, two-stage BoT-SORT bridges them   (E05 `ML0-Default-K1` vs `Full-Rate`, E01 gap stats)
    scorer_limited   person tracked, `Full-Rate K=1` still raises no alert             (E05 `Full-Rate` series, per-event alert rule)
    scheduling       Motion-Only / Joint-Controller / Cache-C1 vs Fixed, queue collapse (E03 + E05 pickles, Pi 5 live traces)

Conventions: all scores are anomaly scores (higher = more anomalous); clip AUCs are x100; series of a dataset are the concatenation of its clips in
canonical order (sorted GT files for ShanghaiTech, `test_videos.txt` for UBnormal), so `offsets` slice one clip out.
"""

from __future__ import annotations

import functools
import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from edgelab.app.catalog import (
    DATA_ROOT,
    EXPERIMENTS_ROOT,
    MODEL_BY_ID,
    GtEvent,
    build_gt_events,
    find_cache_npz,
    list_dataset_videos,
    load_ground_truth,
    load_split_map,
    native_fps,
)

DATASETS = ("ShanghaiTech", "UBnormal")
SEG_LEN = {"ShanghaiTech": 24, "UBnormal": 16}  # STG-NF window length (E00)
ALERT_MIN_FRAMES = 6  # an alert = score above theta for >= 6 consecutive frames (E03 `EVENT_MIN_FRAMES`)
K1_CELL = "s1.0_b1.00"  # native-speed cell: K=1 arms are sustainable there, so the delivered series is the arm's own quality
DEFAULT_BUDGET_CELL = "s2.0_b0.50"
KEEP_CELLS = ("s1.0_b0.25", "s1.0_b0.50", "s1.0_b1.00", "s2.0_b0.25", "s2.0_b0.50", "s2.0_b0.75", "s3.0_b0.50", "s3.0_b1.00")
BUDGET_CELLS = tuple(c for c in KEEP_CELLS if c != K1_CELL)

E01_SERIES = DATA_ROOT / "q2-p1-e01/series.npz"
E01_RESULTS = EXPERIMENTS_ROOT / "Q2-P1-E01_quality/results.csv"
RUN_PICKLES = {
    "e05": DATA_ROOT / "q2-p1-e05/run_{dataset}.pkl",
    "e03": DATA_ROOT / "q2-p1-e03/run_{dataset}.pkl",
}
PI5_LIVE_DIR = DATA_ROOT / "pi5_suite/out/q2-p1-e05"
E01_ML0 = {"ShanghaiTech": "ML0_ncnn", "UBnormal": "ML0_mps"}  # frozen ML0 front-end per dataset in Q2-P1 (UB uses the MPS proxy)

FAILURE_MODES: dict[str, str] = {
    "small_person": "1. Small-person misses (stock fails, ML0 succeeds)",
    "fragmentation": "2. Track fragmentation (default tracker fails, 2-stage BoT-SORT succeeds)",
    "scorer_limited": "3. Scorer-limited misses (Full-Rate K=1 also misses)",
    "scheduling": "4. Scheduling / caching failures (Motion-Only, Joint-Controller, Cache-C1, queue collapse)",
}
SCHEDULING_VARIANTS: dict[str, str] = {
    "motion_only": "Motion-Only vs Fixed (quiet anomalies missed)",
    "joint": "Joint-Controller vs Fixed",
    "cache_helps": "Cache-C1 vs Fixed - where C1 helps most",
    "cache_hurts": "Cache-C1 vs Fixed - where C1 hurts most",
    "queue_pi5": "Queue collapse on the physical Pi 5: Fixed K=1 vs Cache-C1+Guard (12 live clips)",
    "queue_sim": "Queue collapse in simulation (s=3.0): Full-Rate K=1 vs Cache-C1+Guard",
}


# ============================================================================================ pure helpers
def clip_offsets(n_frames: list[int] | np.ndarray) -> np.ndarray:
    """Start index of every clip inside a concatenated per-frame series (length n_clips + 1)."""
    return np.concatenate([[0], np.cumsum(np.asarray(n_frames, dtype=np.int64))])


def clip_auc(label: np.ndarray, score: np.ndarray) -> float:
    """ROC-AUC x 100 of one clip (rank formulation with tie handling); NaN when the clip has a single class."""
    label = np.asarray(label).astype(bool)
    score = np.asarray(score, dtype=np.float64)
    n_pos, n_neg = int(label.sum()), int((~label).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    sorted_scores = score[order]
    ranks = np.empty(len(score), dtype=np.float64)
    i = 0
    while i < len(score):  # average ranks over ties
        j = i
        while j + 1 < len(score) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    u = ranks[label].sum() - n_pos * (n_pos + 1) / 2.0
    return float(100.0 * u / (n_pos * n_neg))


def alert_runs(above: np.ndarray, min_len: int = ALERT_MIN_FRAMES) -> list[tuple[int, int]]:
    """Inclusive runs of >= `min_len` consecutive True frames (an alert = score above theta that long)."""
    above = np.asarray(above, dtype=bool)
    if above.size == 0:
        return []
    d = np.diff(np.concatenate([[0], above.astype(np.int8), [0]]))
    starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1) - 1
    return [(int(s), int(e)) for s, e in zip(starts, ends) if e - s + 1 >= min_len]


def event_alert_delay(event: tuple[int, int], runs: list[tuple[int, int]], min_len: int = ALERT_MIN_FRAMES) -> float | None:
    """Frames from event onset to the first alert that overlaps the event and is raised before it ends (E05 rule); None = missed."""
    s, e = event
    best: float | None = None
    for r0, r1 in runs:
        raise_frame = r0 + min_len - 1
        if r1 >= s and raise_frame <= e:
            d = float(max(raise_frame, s) - s)
            best = d if best is None else min(best, d)
    return best


def scorable_mask(series: np.ndarray, sentinel: float) -> np.ndarray:
    """Frames that carry a real window score (everything not filled with the most-normal sentinel)."""
    return np.asarray(series, dtype=np.float64) > float(sentinel) + 1e-6


def evaluate_events(events: list[GtEvent], series: np.ndarray, theta: float, sentinel: float,
                    min_len: int = ALERT_MIN_FRAMES) -> list[dict]:
    """Per-event outcome of one score series: scorable coverage, peak score, peak/theta, alert (bool) and delay in frames."""
    runs = alert_runs(np.asarray(series) > theta, min_len)
    scor = scorable_mask(series, sentinel)
    out = []
    for ev in events:
        seg = np.asarray(series[ev.start_frame:ev.end_frame + 1], dtype=np.float64)
        delay = event_alert_delay((ev.start_frame, ev.end_frame), runs, min_len)
        peak = float(seg.max()) if seg.size else float("nan")
        out.append({
            "coverage": float(scor[ev.start_frame:ev.end_frame + 1].mean()) if seg.size else 0.0,
            "peak": peak,
            "peak_ratio": peak / theta if theta else float("nan"),
            "alerted": delay is not None,
            "delay_frames": delay,
        })
    return out


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU of xyxy boxes `a` (N,4) and `b` (M,4) -> (N,M)."""
    a = np.asarray(a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)


def small_person_miss_frames(ml0_frames: list[dict], stock_frames: list[dict], gate: float = 0.25,
                             max_height: float = 48.0, iou_thr: float = 0.3) -> dict:
    """Per frame: ML0 persons (conf >= gate) that NO stock detection (conf >= gate) overlaps (IoU >= `iou_thr`).

    Returns arrays over the common frame range: `n_small_ml0` (ML0 persons shorter than `max_height` px), `n_unmatched` (those small persons stock misses),
    `n_unmatched_any` (stock misses of ANY height), `heights` (px of the small missed persons, per frame) and `boxes` (their xyxy boxes, per frame).
    """
    n = min(len(ml0_frames), len(stock_frames))
    n_small = np.zeros(n, dtype=np.int32)
    n_miss = np.zeros(n, dtype=np.int32)
    n_miss_any = np.zeros(n, dtype=np.int32)
    heights: list[list[float]] = []
    boxes_out: list[np.ndarray] = []
    all_missed_heights: list[float] = []
    for t in range(n):
        m, s = ml0_frames[t], stock_frames[t]
        mb = m["boxes"][m["conf"] >= gate]
        sb = s["boxes"][s["conf"] >= gate]
        if len(mb) == 0:
            heights.append([])
            boxes_out.append(np.zeros((0, 4)))
            continue
        h = mb[:, 3] - mb[:, 1]
        matched = box_iou(mb, sb).max(axis=1) >= iou_thr if len(sb) else np.zeros(len(mb), dtype=bool)
        small = h < max_height
        n_small[t] = int(small.sum())
        n_miss_any[t] = int((~matched).sum())
        miss_small = small & ~matched
        n_miss[t] = int(miss_small.sum())
        heights.append([round(float(x), 1) for x in h[miss_small]])
        boxes_out.append(mb[miss_small])
        all_missed_heights.extend(float(x) for x in h[~matched])
    return {"n_small_ml0": n_small, "n_unmatched": n_miss, "n_unmatched_any": n_miss_any, "heights": heights, "boxes": boxes_out,
            "missed_heights_any": all_missed_heights}


def gap_bridging_frames(frames: list[dict], high: float = 0.20, low: float = 0.05, reach: int = 5, iou_thr: float = 0.25) -> list[dict]:
    """Frames where a LOW box (low <= conf < high) is the only evidence of a person who has a HIGH box within `reach` frames before and after.

    These are exactly the detections a single-stage tracker (gate 0.25) throws away and the two-stage tracker's second association keeps,
    so they are where an intermittent confidence dip would otherwise break the track.
    """
    n = len(frames)
    hi_boxes = [f["boxes"][f["conf"] >= high] for f in frames]
    out = []
    for t in range(n):
        f = frames[t]
        sel = (f["conf"] >= low) & (f["conf"] < high)
        for box, conf in zip(f["boxes"][sel], f["conf"][sel]):
            if len(hi_boxes[t]) and box_iou(box[None], hi_boxes[t]).max() >= iou_thr:
                continue  # already covered by a high-confidence box on this frame
            before = any(len(hi_boxes[u]) and box_iou(box[None], hi_boxes[u]).max() >= iou_thr for u in range(max(0, t - reach), t))
            after = any(len(hi_boxes[u]) and box_iou(box[None], hi_boxes[u]).max() >= iou_thr for u in range(t + 1, min(n, t + reach + 1)))
            if before and after:
                out.append({"frame": t, "conf": float(conf), "height": float(box[3] - box[1])})
    return out


def runs_of(frames: list[int]) -> list[tuple[int, int]]:
    """Sorted frame indices -> inclusive runs of consecutive frames."""
    out: list[tuple[int, int]] = []
    for f in sorted(set(int(x) for x in frames)):
        if out and f == out[-1][1] + 1:
            out[-1] = (out[-1][0], f)
        else:
            out.append((f, f))
    return out


def window_landing_counts(tracks: dict[int, tuple[np.ndarray, np.ndarray]], n_frames: int, seg_len: int) -> tuple[np.ndarray, np.ndarray]:
    """From `{track id: (frame ids, keypoints)}`: per-frame active-track count and number of complete STG-NF windows landing (ending) on each frame."""
    from edgelab.analysis.q2_p1_e00 import complete_window_starts

    active = np.zeros(n_frames, dtype=np.int32)
    landing = np.zeros(n_frames, dtype=np.int32)
    for frames, _ in tracks.values():
        frames = np.asarray(frames)
        active[frames[frames < n_frames]] += 1
        starts = complete_window_starts(frames, seg_len)
        if len(starts):
            last = frames[starts + seg_len - 1]
            np.add.at(landing, last[last < n_frames], 1)
    return active, landing


# ============================================================================================ artefact loaders
@functools.lru_cache(maxsize=4)
def dataset_clips(dataset: str) -> tuple[list[str], list[np.ndarray]]:
    """Clip names in canonical order and their 0/1 GT masks."""
    names = [v.video_id for v in list_dataset_videos(dataset)]
    return names, [load_ground_truth(dataset, n) for n in names]


@functools.lru_cache(maxsize=4)
def clip_layout(dataset: str) -> tuple[list[str], np.ndarray, np.ndarray]:
    """`(names, offsets, concatenated GT)`."""
    names, labels = dataset_clips(dataset)
    return names, clip_offsets([len(g) for g in labels]), np.concatenate(labels)


def clip_slice(dataset: str, clip: str) -> slice:
    names, offsets, _ = clip_layout(dataset)
    i = names.index(clip)
    return slice(int(offsets[i]), int(offsets[i + 1]))


@functools.lru_cache(maxsize=2)
def load_e01_series(path_str: str = str(E01_SERIES)) -> dict[str, np.ndarray]:
    p = Path(path_str)
    if not p.exists():
        return {}
    d = np.load(p)
    return {k: d[k] for k in d.files}


@functools.lru_cache(maxsize=4)
def load_e01_clip_stats(dataset: str) -> pd.DataFrame:
    """E01 per-clip track statistics of the front-end + stored default tracker (gaps, windows, coverage); indexed by clip."""
    if not E01_RESULTS.exists():
        return pd.DataFrame()
    df = pd.read_csv(E01_RESULTS, low_memory=False)
    df = df[(df["dataset"] == dataset) & (df["frontend"] == E01_ML0[dataset]) & (df["tracker"] == "stored-default")]
    return df.set_index("video")


@dataclass
class ArmRun:
    arm: str
    cell: str
    series: np.ndarray  # delivered anomaly score per frame (all clips concatenated)
    observed: np.ndarray  # 1 = detector observation on that frame
    clip_wait_mean_ms: np.ndarray
    clip_wait_max_ms: np.ndarray
    clip_cpu_ms_per_s: np.ndarray


@dataclass
class RunBundle:
    source: str  # "e05" | "e03"
    dataset: str
    n_frames: list[int]
    theta: dict[str, float] = field(default_factory=dict)
    runs: dict[tuple[str, str], ArmRun] = field(default_factory=dict)
    traces: dict[str, dict] = field(default_factory=dict)

    def arms(self) -> list[str]:
        return sorted({a for a, _ in self.runs})

    def cells(self) -> list[str]:
        return sorted({c for _, c in self.runs})

    def has(self, arm: str, cell: str) -> bool:
        return (arm, cell) in self.runs

    def clip_series(self, arm: str, cell: str, clip_index: int) -> np.ndarray:
        off = clip_offsets(self.n_frames)
        return self.runs[(arm, cell)].series[off[clip_index]:off[clip_index + 1]]

    def clip_observed(self, arm: str, cell: str, clip_index: int) -> np.ndarray:
        off = clip_offsets(self.n_frames)
        return self.runs[(arm, cell)].observed[off[clip_index]:off[clip_index + 1]]


@functools.lru_cache(maxsize=4)
def load_run_bundle(source: str, dataset: str) -> RunBundle | None:
    """Series / observation masks / per-clip queue waits of every (arm, cell) in `KEEP_CELLS` from the frozen E05 or E03 simulation pickle."""
    path = Path(str(RUN_PICKLES[source]).format(dataset=dataset))
    if not path.exists():
        return None
    with open(path, "rb") as fh:
        obj = pickle.load(fh)
    bundle = RunBundle(source=source, dataset=dataset, n_frames=[int(x) for x in obj["n_frames"]],
                       theta={k: float(v) for k, v in (obj.get("theta") or {}).items()})
    for (arm, cell), r in obj["results"].items():
        if cell not in KEEP_CELLS or "series" not in r:
            continue
        bundle.runs[(arm, cell)] = ArmRun(
            arm=arm, cell=cell, series=np.asarray(r["series"], dtype=np.float32),
            observed=np.asarray(r.get("observed", np.zeros(len(r["series"]))), dtype=np.uint8),
            clip_wait_mean_ms=np.asarray(r.get("clip_wait_mean_ms", []), dtype=np.float64),
            clip_wait_max_ms=np.asarray(r.get("clip_wait_max_ms", []), dtype=np.float64),
            clip_cpu_ms_per_s=np.asarray(r.get("clip_cpu_ms_per_s", []), dtype=np.float64),
        )
    for arm, tr in (obj.get("traces") or {}).items():
        bundle.traces[arm] = tr
    return bundle


def trace_frame(bundle: RunBundle, arm: str, clip: str) -> pd.DataFrame | None:
    """Per-frame scheduler trace (observe, C1 level, C1 reuse, fresh/reused windows, queue wait ms) for the clips E05 traced in detail."""
    tr = bundle.traces.get(arm)
    if tr is None or clip not in tr["clips"]:
        return None
    rows = tr["frames"][tr["clips"].index(clip)]
    df = pd.DataFrame(rows, columns=tr["columns"])
    return df.rename(columns={"t": "frame"})


@functools.lru_cache(maxsize=1)
def load_pi5_live() -> dict | None:
    """The physical-Pi-5 sustained E05 run (12 ShanghaiTech clips x first 240 frames): per-frame series, queue wait, C1 reuse, shedding."""
    if not PI5_LIVE_DIR.exists():
        return None
    arms = {}
    labels = {
        "fixed_k1": "Pi 5 Live: Fixed K=1 (no cache)",
        "cache_c1": "Pi 5 Live: Cache-C1",
        "cache_c1_guard": "Pi 5 Live: Cache-C1+Guard",
        "cache_c1_guard_burst32": "Pi 5 Live: Cache-C1+Guard Burst32",
    }
    clips: list[str] | None = None
    for key in labels:
        npz, js = PI5_LIVE_DIR / f"{key}.npz", PI5_LIVE_DIR / f"{key}.json"
        if not npz.exists() or not js.exists():
            continue
        d = np.load(npz, allow_pickle=True)
        meta = json.loads(js.read_text())
        clips = clips or list(meta["clips"])
        arms[key] = {"label": labels[key], "meta": meta, **{k: d[k] for k in d.files}}
    if not arms or clips is None:
        return None
    return {"clips": clips, "arms": arms}


def pi5_clip(pi5: dict, arm_key: str, clip: str) -> dict | None:
    """Slice of the Pi 5 live run belonging to `clip` (clip-local frame numbers)."""
    if pi5 is None or clip not in pi5["clips"] or arm_key not in pi5["arms"]:
        return None
    a = pi5["arms"][arm_key]
    sel = a["clip_of"] == pi5["clips"].index(clip)
    return {
        "frame": a["local_t"][sel], "series": a["series"][sel], "wait_ms": a["wait_ms"][sel], "reused": a["reused"][sel],
        "shed": a["shed"][sel], "obs": a["obs"][sel], "svc_ms": a["svc_ms"][sel],
    }


# ============================================================================================ series registry (timeline)
@dataclass(frozen=True)
class SeriesDef:
    id: str
    label: str
    kind: str  # "e01" | "e05" | "e03" | "pi5"
    key: str  # e01: "<frontend>|<S>" ; e05/e03: arm ; pi5: run key
    world: str  # theta of this series in the E05 pickle
    k1: bool = False  # use the native-speed cell instead of the budget cell


SERIES_DEFS: tuple[SeriesDef, ...] = (
    SeriesDef("stock", "Stock YOLOv8n @640 (causal S4)", "e01", "stock|S4", "stock_default"),
    SeriesDef("ml0_s0", "ML0 Offline S0", "e01", "{ml0}|S0", "ml0_default"),
    SeriesDef("ml0_s4", "ML0 Causal S4", "e01", "{ml0}|S4", "ml0_default"),
    SeriesDef("full_rate", "Full-Rate K=1 (ML0 + 2-stage tracker)", "e05", "Full-Rate", "ml0_locked", True),
    SeriesDef("fixed", "Fixed-Burst", "e05", "Fixed", "ml0_locked"),
    SeriesDef("motion_only", "Motion-Only", "e03", "Motion-Only", "ml0_locked"),
    SeriesDef("joint", "Joint-Controller", "e03", "Joint-Controller", "ml0_locked"),
    SeriesDef("cache_c1", "Cache-C1", "e05", "Cache-C1", "ml0_locked"),
    SeriesDef("cache_c1_guard", "Cache-C1+Guard", "e05", "Cache-C1+Guard", "ml0_locked"),
    SeriesDef("pi5_guard", "Pi 5 Live (Cache-C1+Guard)", "pi5", "cache_c1_guard", "ml0_locked"),
    SeriesDef("pi5_fixed_k1", "Pi 5 Live (Fixed K=1)", "pi5", "fixed_k1", "ml0_locked"),
)
SERIES_BY_ID = {s.id: s for s in SERIES_DEFS}
DEFAULT_TIMELINE_SERIES = ("stock", "ml0_s0", "ml0_s4", "fixed", "cache_c1_guard", "pi5_guard")


def get_clip_series(dataset: str, series_id: str, clip: str, budget_cell: str = DEFAULT_BUDGET_CELL) -> dict | None:
    """`{"frame": ndarray, "score": ndarray, "theta": float | None, "note": str}` for one registered series on one clip, or `None` if unavailable."""
    sd = SERIES_BY_ID[series_id]
    names, _, _ = clip_layout(dataset)
    if clip not in names:
        return None
    ci = names.index(clip)
    n = len(dataset_clips(dataset)[1][ci])
    theta = None
    e05 = load_run_bundle("e05", dataset)
    if e05 is not None:
        theta = e05.theta.get(sd.world)
    if sd.kind == "e01":
        arr = load_e01_series().get(f"{dataset}|{sd.key.format(ml0=E01_ML0[dataset])}")
        if arr is None:
            return None
        return {"frame": np.arange(n), "score": np.asarray(arr[clip_slice(dataset, clip)], dtype=np.float64), "theta": theta,
                "note": "frozen E01 ideal (compute-unconstrained) series, default tracker"}
    if sd.kind in ("e05", "e03"):
        bundle = load_run_bundle(sd.kind, dataset)
        cell = K1_CELL if sd.k1 else budget_cell
        if bundle is None or not bundle.has(sd.key, cell):
            return None
        return {"frame": np.arange(n), "score": bundle.clip_series(sd.key, cell, ci).astype(np.float64), "theta": theta,
                "note": f"{sd.kind.upper()} simulation, cell {cell}"}
    if sd.kind == "pi5":
        if dataset != "ShanghaiTech":
            return None
        pc = pi5_clip(load_pi5_live(), sd.key, clip)
        if pc is None:
            return None
        return {"frame": pc["frame"], "score": pc["series"].astype(np.float64), "theta": theta,
                "note": "physical Raspberry Pi 5 live run (first 240 frames of the clip only)"}
    return None


# ============================================================================================ failure-case rankings
def _split_of(dataset: str, clip: str) -> str:
    return load_split_map(dataset).get(clip, "n/a")


def _clip_meta_rows(dataset: str) -> list[dict]:
    names, labels = dataset_clips(dataset)
    return [{"clip": n, "split": _split_of(dataset, n), "frames": len(g), "anomalous_frames": int(g.sum())} for n, g in zip(names, labels)]


def _per_clip_auc(dataset: str, series: np.ndarray) -> np.ndarray:
    names, offsets, _ = clip_layout(dataset)
    _, labels = dataset_clips(dataset)
    return np.array([clip_auc(labels[i], series[offsets[i]:offsets[i + 1]]) for i in range(len(names))])


def rank_small_person_misses(dataset: str) -> pd.DataFrame:
    """Clips ordered by the `ML0 - stock` clip-AUC gain (E01 causal S4, same default tracker for both)."""
    ser = load_e01_series()
    ks, km = f"{dataset}|stock|S4", f"{dataset}|{E01_ML0[dataset]}|S4"
    if ks not in ser or km not in ser:
        return pd.DataFrame()
    df = pd.DataFrame(_clip_meta_rows(dataset))
    df["auc_stock"] = _per_clip_auc(dataset, ser[ks])
    df["auc_ml0"] = _per_clip_auc(dataset, ser[km])
    df["auc_gain"] = df["auc_ml0"] - df["auc_stock"]
    return df.dropna(subset=["auc_gain"]).sort_values("auc_gain", ascending=False).reset_index(drop=True)


def rank_track_fragmentation(dataset: str) -> pd.DataFrame:
    """Clips ordered by the `two-stage - default tracker` clip-AUC gain (E05 `Full-Rate` vs `ML0-Default-K1`, native-speed cell)."""
    b = load_run_bundle("e05", dataset)
    if b is None or not (b.has("ML0-Default-K1", K1_CELL) and b.has("Full-Rate", K1_CELL)):
        return pd.DataFrame()
    df = pd.DataFrame(_clip_meta_rows(dataset))
    df["auc_default"] = _per_clip_auc(dataset, b.runs[("ML0-Default-K1", K1_CELL)].series)
    df["auc_two_stage"] = _per_clip_auc(dataset, b.runs[("Full-Rate", K1_CELL)].series)
    df["auc_gain"] = df["auc_two_stage"] - df["auc_default"]
    stats = load_e01_clip_stats(dataset)
    for col in ("n_tracks", "gaps_1", "gaps_2", "gaps_ge3", "gap_frames"):
        df[f"default_{col}"] = df["clip"].map(stats[col]) if col in stats else np.nan
    return df.dropna(subset=["auc_gain"]).sort_values("auc_gain", ascending=False).reset_index(drop=True)


def rank_scorer_limited(dataset: str, min_coverage: float = 0.5) -> pd.DataFrame:
    """Every GT event with `Full-Rate K=1` (ML0 + two-stage tracker): coverage, peak/theta, alert outcome; `scorer_limited` = tracked but silent."""
    b = load_run_bundle("e05", dataset)
    if b is None or not b.has("Full-Rate", K1_CELL):
        return pd.DataFrame()
    names, labels = dataset_clips(dataset)
    run = b.runs[("Full-Rate", K1_CELL)]
    sentinel = float(run.series.min())
    theta = b.theta["ml0_locked"]
    seg = SEG_LEN[dataset]
    rows = []
    for ci, clip in enumerate(names):
        events = build_gt_events(dataset, clip, labels[ci])
        res = evaluate_events(events, b.clip_series("Full-Rate", K1_CELL, ci), theta, sentinel)
        for ev, r in zip(events, res):
            rows.append({
                "clip": clip, "split": _split_of(dataset, clip), "event": ev.index, "start_frame": ev.start_frame, "end_frame": ev.end_frame,
                "duration_frames": ev.duration_frames, "duration_s": round(ev.duration_s, 2), "tags": ", ".join(ev.tags),
                "coverage": round(r["coverage"], 3), "peak_score": round(r["peak"], 3), "theta": round(theta, 3),
                "peak_over_theta": round(r["peak_ratio"], 3), "alerted": r["alerted"],
                "delay_s": None if r["delay_frames"] is None else round(r["delay_frames"] / native_fps(dataset), 2),
                "tracked_but_silent": (not r["alerted"]) and r["coverage"] >= min_coverage,
                "shorter_than_warmup": ev.duration_frames < seg,
            })
    return pd.DataFrame(rows)


def rank_scheduling(dataset: str, variant: str, budget_cell: str = DEFAULT_BUDGET_CELL) -> pd.DataFrame:
    """Clip ranking for one scheduling/caching variant (`SCHEDULING_VARIANTS`)."""
    names, labels = dataset_clips(dataset)
    df = pd.DataFrame(_clip_meta_rows(dataset))
    if variant in ("motion_only", "joint"):
        arm = "Motion-Only" if variant == "motion_only" else "Joint-Controller"
        b = load_run_bundle("e03", dataset)
        if b is None or not (b.has(arm, budget_cell) and b.has("Fixed", budget_cell)):
            return pd.DataFrame()
        df["auc_fixed"] = _per_clip_auc(dataset, b.runs[("Fixed", budget_cell)].series)
        df["auc_variant"] = _per_clip_auc(dataset, b.runs[(arm, budget_cell)].series)
        df["auc_delta"] = df["auc_variant"] - df["auc_fixed"]
        off = clip_offsets(b.n_frames)
        for tag, key in (("fixed", "Fixed"), ("variant", arm)):
            obs = b.runs[(key, budget_cell)].observed
            df[f"observed_anom_{tag}"] = [float(obs[off[i]:off[i + 1]][labels[i].astype(bool)].mean()) if labels[i].any() else np.nan for i in range(len(names))]
            df[f"observed_all_{tag}"] = [float(obs[off[i]:off[i + 1]].mean()) for i in range(len(names))]
        return df.dropna(subset=["auc_delta"]).sort_values("auc_delta", ascending=True).reset_index(drop=True)
    if variant in ("cache_helps", "cache_hurts"):
        b = load_run_bundle("e05", dataset)
        if b is None or not (b.has("Cache-C1", budget_cell) and b.has("Fixed", budget_cell)):
            return pd.DataFrame()
        df["auc_fixed"] = _per_clip_auc(dataset, b.runs[("Fixed", budget_cell)].series)
        df["auc_variant"] = _per_clip_auc(dataset, b.runs[("Cache-C1", budget_cell)].series)
        df["auc_delta"] = df["auc_variant"] - df["auc_fixed"]
        off = clip_offsets(b.n_frames)
        for tag, key in (("fixed", "Fixed"), ("variant", "Cache-C1")):
            obs = b.runs[(key, budget_cell)].observed
            df[f"observed_all_{tag}"] = [float(obs[off[i]:off[i + 1]].mean()) for i in range(len(names))]
            df[f"observed_anom_{tag}"] = [float(obs[off[i]:off[i + 1]][labels[i].astype(bool)].mean()) if labels[i].any() else np.nan for i in range(len(names))]
        df["wait_max_ms_cache"] = b.runs[("Cache-C1", budget_cell)].clip_wait_max_ms if len(b.runs[("Cache-C1", budget_cell)].clip_wait_max_ms) == len(names) else np.nan
        return df.dropna(subset=["auc_delta"]).sort_values("auc_delta", ascending=(variant == "cache_hurts")).reset_index(drop=True)
    if variant == "queue_pi5":
        pi5 = load_pi5_live()
        if dataset != "ShanghaiTech" or pi5 is None or not {"fixed_k1", "cache_c1_guard"} <= set(pi5["arms"]):
            return pd.DataFrame()
        rows = []
        for clip in pi5["clips"]:
            fx, gd = pi5_clip(pi5, "fixed_k1", clip), pi5_clip(pi5, "cache_c1_guard", clip)
            ci = names.index(clip)
            gt = labels[ci][:len(fx["frame"])]
            rows.append({
                "clip": clip, "split": _split_of(dataset, clip), "frames_run": len(fx["frame"]), "anomalous_frames": int(gt.sum()),
                "wait_max_s_fixed_k1": round(float(fx["wait_ms"].max()) / 1000.0, 2), "wait_mean_s_fixed_k1": round(float(fx["wait_ms"].mean()) / 1000.0, 2),
                "wait_max_ms_guard": round(float(gd["wait_ms"].max()), 1), "shed_frac_guard": round(float(gd["shed"].mean()), 3),
                "c1_reuse_frac_guard": round(float(gd["reused"].mean()), 3),
                "auc_fixed_k1": clip_auc(gt, fx["series"]), "auc_guard": clip_auc(gt, gd["series"]),
            })
        out = pd.DataFrame(rows)
        out["auc_delta"] = out["auc_guard"] - out["auc_fixed_k1"]
        return out.sort_values("wait_max_s_fixed_k1", ascending=False).reset_index(drop=True)
    if variant == "queue_sim":
        b = load_run_bundle("e05", dataset)
        cell = "s3.0_b1.00"
        if b is None or not (b.has("Full-Rate", cell) and b.has("Cache-C1+Guard", cell)):
            return pd.DataFrame()
        fr, gd = b.runs[("Full-Rate", cell)], b.runs[("Cache-C1+Guard", cell)]
        df["auc_full_rate_k1"] = _per_clip_auc(dataset, fr.series)
        df["auc_guard"] = _per_clip_auc(dataset, gd.series)
        df["auc_delta"] = df["auc_guard"] - df["auc_full_rate_k1"]
        df["wait_max_s_full_rate"] = fr.clip_wait_max_ms / 1000.0 if len(fr.clip_wait_max_ms) == len(names) else np.nan
        df["wait_max_ms_guard"] = gd.clip_wait_max_ms if len(gd.clip_wait_max_ms) == len(names) else np.nan
        return df.dropna(subset=["auc_delta"]).sort_values("wait_max_s_full_rate", ascending=False).reset_index(drop=True)
    raise ValueError(f"unknown scheduling variant {variant!r}")


# ============================================================================================ per-clip replay (trackers on cached detections)
def ml0_spec_id(dataset: str) -> str:
    return "exp020-ML0-ncnn" if dataset == "ShanghaiTech" else "exp020-ML0"


@functools.lru_cache(maxsize=6)
def load_clip_detections(model_id: str, dataset: str, clip: str) -> list[dict] | None:
    """All cached detections (conf >= 0.01) of one clip for one catalog model, one dict per frame; None when no cache exists."""
    from edgelab.pose.detcache import load_detections

    npz = find_cache_npz(MODEL_BY_ID[model_id], dataset, clip)
    if npz is None:
        return None
    frames, _ = load_detections(npz)
    return frames


@functools.lru_cache(maxsize=12)
def replay_track_stats(dataset: str, clip: str, tracker: str) -> dict | None:
    """Replay the default (`c=0.25`, single stage) or locked two-stage (`c=0.20, low=0.05`) BoT-SORT over the ML0 detector cache of a clip.

    Returns per-frame `active` tracks and `landing` windows plus track-length statistics, or None if the cache is missing.
    """
    from edgelab.analysis.q2_p1_eval import BOTSORT_DEFAULT, BOTSORT_LOCKED, replay_clip

    frames = load_clip_detections(ml0_spec_id(dataset), dataset, clip)
    if frames is None:
        return None
    spec = BOTSORT_LOCKED if tracker == "locked" else BOTSORT_DEFAULT
    tracks = replay_clip(frames, np.arange(len(frames)), spec)
    active, landing = window_landing_counts(tracks, len(frames), SEG_LEN[dataset])
    lens = [len(f) for f, _ in tracks.values()]
    return {
        "active": active, "landing": landing, "n_tracks": len(tracks), "longest_track": max(lens) if lens else 0,
        "mean_track_len": float(np.mean(lens)) if lens else 0.0, "n_windows": int(landing.sum()),
    }


# ============================================================================================ queue / cache overlay
def build_overlay_frame(dataset: str, clip: str, budget_cell: str = DEFAULT_BUDGET_CELL) -> tuple[pd.DataFrame, list[str]]:
    """Per-frame scheduling overlay of a clip, long-format columns `frame, series, value, panel`; also returns human-readable provenance notes.

    Priority per quantity: physical Pi 5 live trace (12 clips) > E05 per-frame trace (3 clips) > E05/E03 observation masks (every clip).
    """
    rows: list[dict] = []
    notes: list[str] = []
    names, _ = dataset_clips(dataset)
    ci = names.index(clip)

    def add(frames, values, series, panel):
        for f, v in zip(frames, values):
            rows.append({"frame": int(f), "series": series, "value": float(v), "panel": panel})

    pi5 = load_pi5_live() if dataset == "ShanghaiTech" else None
    if pi5 is not None and clip in pi5["clips"]:
        for key in ("fixed_k1", "cache_c1_guard"):
            pc = pi5_clip(pi5, key, clip)
            if pc is None:
                continue
            tag = pi5["arms"][key]["label"]
            add(pc["frame"], pc["wait_ms"] / 1000.0, f"{tag} - queue wait (s)", "queue wait (s)")
            if key == "cache_c1_guard":
                add(pc["frame"], pc["reused"], "Pi 5 C1 cache hit (1 = reused)", "C1 cache / observation")
                add(pc["frame"], pc["shed"], "Pi 5 guard shed (1 = frame shed)", "C1 cache / observation")
        notes.append("Pi 5 live traces cover only the first 240 frames of this clip (12-clip sustained run).")

    e05 = load_run_bundle("e05", dataset)
    if e05 is not None:
        for arm in ("Cache-C1+Guard", "Cache-C1", "Fixed"):
            tdf = trace_frame(e05, arm, clip)
            if tdf is not None and arm in ("Cache-C1+Guard", "Fixed"):
                add(tdf["frame"], tdf["queue_wait_ms"] / 1000.0, f"Sim {arm} - queue wait (s)", "queue wait (s)")
                if arm == "Cache-C1+Guard":
                    add(tdf["frame"], tdf["c1_reused"], "Sim C1 cache hit (1 = reused)", "C1 cache / observation")
                notes.append(f"E05 per-frame scheduler trace available for this clip ({arm}, cell {e05.traces[arm].get('cell', 's2.0_b0.50')}).")
        for arm in ("Cache-C1+Guard", "Fixed"):
            if e05.has(arm, budget_cell):
                obs = e05.clip_observed(arm, budget_cell, ci)
                add(np.arange(len(obs)), obs, f"Sim {arm} - detector observation", "C1 cache / observation")
    return pd.DataFrame(rows, columns=["frame", "series", "value", "panel"]), sorted(set(notes))


# ============================================================================================ diagnostic cards
@dataclass
class Diagnosis:
    verdict: str  # "failed" | "succeeded" | "mixed" | "info"
    headline: str
    bullets: list[str]
    mechanism: str


def _fmt(x: float | None, nd: int = 1) -> str:
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"


def diagnose_small_person(ctx: dict) -> Diagnosis:
    gain = ctx["auc_ml0"] - ctx["auc_stock"]
    n_gt = max(1, ctx["n_gt_frames"])
    heights = ctx.get("focus_heights") or []
    bullets = [
        f"Clip AUC x100: stock {_fmt(ctx['auc_stock'])} -> ML0 {_fmt(ctx['auc_ml0'])} ({gain:+.1f} pt).",
        f"In {ctx['n_unmatched_gt_frames']}/{ctx['n_gt_frames']} anomalous frames ({100 * ctx['n_unmatched_gt_frames'] / n_gt:.0f}%) ML0 detects at least one "
        f"person shorter than 48 px that no stock box overlaps (IoU >= 0.3, conf >= {ctx.get('gate', 0.25):.2f}).",
    ]
    if ctx.get("mean_det_gt_ml0") is not None:
        bullets.append(
            f"Mean detections per anomalous frame (conf >= {ctx.get('gate', 0.25):.2f}): stock {ctx['mean_det_gt_stock']:.2f} vs ML0 {ctx['mean_det_gt_ml0']:.2f}; "
            f"{ctx['n_unmatched_any_gt_frames']}/{ctx['n_gt_frames']} anomalous frames contain an ML0 person of ANY height that stock misses.")
    if ctx.get("focus_frame") is not None:
        bullets.append(
            f"Frame {ctx['focus_frame']}: stock keeps {ctx['stock_n_at_focus']} detection(s), ML0 keeps {ctx['ml0_n_at_focus']}; "
            f"missed heights {heights} px.")
    mech = ("Stock COCO YOLOv8n-pose is trained with a label converter that drops keypoint-less small people, so recall collapses below ~48 px "
            "(EXP-011). ML0 keeps those box-only persons in the detection loss (partial labels) at zero extra latency, so the small actor exists, "
            "gets a track and STG-NF can score its 24-frame window; with stock the actor never enters a window and the anomaly is invisible.")
    if gain >= 1.0:
        return Diagnosis("succeeded", "ML0 succeeded where stock failed (small anomalous actors)", bullets, mech)
    if gain <= -1.0:
        return Diagnosis("failed", "ML0 did WORSE than stock on this clip", bullets, mech)
    return Diagnosis("mixed", "No clear front-end difference on this clip", bullets, mech)


def diagnose_fragmentation(ctx: dict) -> Diagnosis:
    gain = ctx["auc_two_stage"] - ctx["auc_default"]
    bullets = [f"Clip AUC x100: default tracker {_fmt(ctx['auc_default'])} -> two-stage {_fmt(ctx['auc_two_stage'])} ({gain:+.1f} pt)."]
    if ctx.get("default_gaps") is not None:
        bullets.append(f"E01 default-tracker diagnostics: {_fmt(ctx['default_gaps'], 0)} track gap(s), {_fmt(ctx['default_gap_frames'], 0)} missing pose frame(s).")
    bullets.append(
        f"{ctx['n_bridge_frames']} frame(s) in {ctx['n_bridge_runs']} run(s) (longest {ctx['longest_bridge']} frames) are carried only by LOW boxes "
        f"(0.05 <= conf < 0.20) that have a high-confidence box of the same person before and after.")
    if ctx.get("replay") is not None:
        r = ctx["replay"]
        bullets.append(
            f"Tracker replay on the ML0 cache: default -> {r['default_tracks']} tracks, longest {r['default_longest']} frames, {r['default_windows']} complete windows; "
            f"two-stage -> {r['locked_tracks']} tracks, longest {r['locked_longest']} frames, {r['locked_windows']} complete windows "
            f"({r['locked_windows'] - r['default_windows']:+d}).")
    mech = ("The default tracker (BoT-SORT, gate 0.25, single association) never sees a box whose confidence dips below 0.25, so a person who is "
            "intermittently low-confidence loses the track and gets a new ID; STG-NF only scores windows of 24 consecutive poses of ONE identity "
            "(<= 2 missing), so each ID break discards the window. The locked two-stage tracker (c=0.20, low=0.05) runs a second association with the "
            "0.05..0.20 boxes, bridges the gap, keeps the identity and saves the 24-frame window.")
    if gain >= 1.0:
        return Diagnosis("succeeded", "Two-stage BoT-SORT succeeded where the default tracker fragmented", bullets, mech)
    if gain <= -1.0:
        return Diagnosis("failed", "Two-stage BoT-SORT did WORSE than the default tracker here", bullets, mech)
    return Diagnosis("mixed", "Both trackers behave alike on this clip", bullets, mech)


def diagnose_scorer_limited(ctx: dict) -> Diagnosis:
    ev = ctx["event"]
    bullets = [
        f"Event {ev['event']} f{ev['start_frame']}-{ev['end_frame']} ({ev['duration_frames']} frames = {ev['duration_s']} s), tags: {ev['tags']}.",
        f"A STG-NF window was scored on {100 * ev['coverage']:.0f}% of the event frames, so the person was detected AND tracked (front-end, tracker and "
        f"scheduler are not the problem; this is `Full-Rate K=1`, no scheduling at all).",
        f"Peak anomaly score inside the event = {ev['peak_score']:.2f} vs alert threshold theta = {ev['theta']:.2f} "
        f"(peak/theta = {ev['peak_over_theta']:.2f}); an alert needs >= {ALERT_MIN_FRAMES} consecutive frames above theta."
        + ("" if not ev["alerted"] else f" It DID alert after {ev['delay_s']} s."),
    ]
    reasons = []
    if ev["shorter_than_warmup"]:
        reasons.append(f"the event lasts {ev['duration_frames']} frames, shorter than the {ctx['seg_len']}-frame STG-NF window, so no window can be fully "
                       "inside it and the score is diluted by normal frames before/after")
    elif "short_event" in ev["tags"]:
        reasons.append("it is a short event (<= 2.5 s): the first complete window needs 24 frames of pose, so the causal score arrives late and the "
                       "event ends before >= 6 consecutive frames exceed theta")
    if ev["peak_over_theta"] < 1.0 and not reasons:
        reasons.append("the actor's internal pose dynamics look normal to the frozen flow: for smooth gliding motion (cyclists, skaters, vehicles) the "
                       "per-window mean/std normalisation removes the global translation, leaving a walking-like skeleton sequence")
    elif ev["peak_over_theta"] < 1.0:
        reasons.append("even inside the window the skeleton dynamics stay below theta after per-window mean/std normalisation (smooth gliding actors look normal)")
    if "high_crowd" in ev["tags"]:
        reasons.append("many simultaneous people produce many normal windows per frame, and the frame score is the most anomalous window only")
    mech = ("Scorer-limited miss: detection, tracking and scheduling all succeeded, but the frozen STG-NF normalizing flow assigns the actor's windows a "
            "normal likelihood. Likely cause(s) on this event: " + ("; ".join(reasons) if reasons else "the anomaly is a scene-level event, not a pose-dynamics event")
            + ". Fixing it needs a different scorer or training signal, not a better detector, tracker or scheduler.")
    return Diagnosis("failed" if not ev["alerted"] else "succeeded",
                     "Scorer-limited miss: tracked but never alerted" if not ev["alerted"] else "Detected and alerted (not a miss)", bullets, mech)


def diagnose_scheduling(variant: str, ctx: dict) -> Diagnosis:
    d = ctx.get("auc_delta")
    if variant in ("motion_only", "joint"):
        name = "Motion-Only" if variant == "motion_only" else "Joint-Controller"
        bullets = [f"Clip AUC x100: Fixed {_fmt(ctx['auc_fixed'])} -> {name} {_fmt(ctx['auc_variant'])} ({d:+.1f} pt)."]
        oa, ob = ctx.get("observed_anom_variant"), ctx.get("observed_all_variant")
        if oa is not None and ob is not None and np.isfinite(oa):
            bullets.append(f"{name} observed {100 * oa:.0f}% of the anomalous frames vs {100 * ob:.0f}% of all frames "
                           f"(Fixed: {100 * ctx['observed_anom_fixed']:.0f}% / {100 * ctx['observed_all_fixed']:.0f}%).")
        quiet = oa is not None and ob is not None and np.isfinite(oa) and oa < ob
        mech = (f"{name} spends its CPU budget where motion/uncertainty is high. On this clip the anomaly is 'quiet': motion during the event is low, so the "
                "scheduler starts few detector bursts there and the windows covering the event are stale or missing, while Fixed-Burst observes at a constant cadence "
                "regardless of content." if quiet and d < 0 else
                f"{name} did not lose observation density on the anomalous frames; the AUC difference comes from which windows land inside the event.")
        return Diagnosis("failed" if d <= -1 else "succeeded" if d >= 1 else "mixed",
                         f"{name} {'missed a quiet anomaly' if d <= -1 else 'matched or beat Fixed'}", bullets, mech)
    if variant in ("cache_helps", "cache_hurts"):
        bullets = [f"Clip AUC x100: Fixed {_fmt(ctx['auc_fixed'])} -> Cache-C1 {_fmt(ctx['auc_variant'])} ({d:+.1f} pt) at the same CPU budget.",
                   f"Observed frames: Fixed {100 * ctx['observed_all_fixed']:.0f}% -> Cache-C1 {100 * ctx['observed_all_variant']:.0f}%."]
        mech = ("C1 reuses the previous detector output when a cheap thumbnail check says the scene is static, saving ~27% CPU. At matched budget the savings "
                "buy a denser Fixed-Burst schedule, so more windows complete and the clip scores better. "
                if d >= 0 else
                "Cache-C1 observes MORE frames than Fixed at the same CPU, so on this clip the loss is not a lack of observations. A likely cause is stale reuse: when a "
                "person starts moving between two thumbnail checks the reused boxes/poses lag the real person, which perturbs the window's mean/std normalisation. "
                "Per-clip deltas of this size are also partly noise (E05 12-cell mean: +2.4 pt on ShanghaiTech, +0.2 pt on UBnormal). ")
        return Diagnosis("succeeded" if d >= 1 else "failed" if d <= -1 else "mixed",
                         f"Cache-C1 {'helped' if d >= 1 else 'hurt' if d <= -1 else 'made no difference'} vs Fixed on this clip", bullets, mech)
    if variant == "queue_pi5":
        bullets = [
            f"Unguarded Fixed K=1 on the physical Pi 5 accumulates up to {ctx['wait_max_s_fixed_k1']:.1f} s of queue wait on this clip "
            f"(mean {ctx['wait_mean_s_fixed_k1']:.1f} s); Cache-C1+Guard stays <= {ctx['wait_max_ms_guard']:.0f} ms.",
            f"Guard shed {100 * ctx['shed_frac_guard']:.1f}% of frames and C1 reused {100 * ctx['c1_reuse_frac_guard']:.1f}%.",
            f"Clip AUC x100 on the processed frames: Fixed K=1 {_fmt(ctx['auc_fixed_k1'])} vs Cache-C1+Guard {_fmt(ctx['auc_guard'])}.",
        ]
        mech = ("The Pi 5 needs ~60 ms per frame at 640 px (16.6-17.2 FPS) but frames arrive every 41.7 ms, so utilisation > 1 and an unbounded FIFO queue "
                "builds: each score is delivered tens of seconds after the event, i.e. the alert arrives long after the anomaly. Cache-C1 cuts the mean service "
                "time to ~35 ms and the hysteresis guard sheds frames when wait > 400 ms, bounding p99 wait near 405 ms.")
        return Diagnosis("failed", "Queue collapse without C1/Guard on the Raspberry Pi 5", bullets, mech)
    if variant == "queue_sim":
        bullets = [f"Simulated s=3.0 Pi: Full-Rate K=1 peak wait {ctx['wait_max_s_full_rate']:.1f} s vs Cache-C1+Guard {ctx['wait_max_ms_guard']:.0f} ms.",
                   f"Clip AUC x100: Full-Rate K=1 {_fmt(ctx['auc_full_rate_k1'])} vs Cache-C1+Guard {_fmt(ctx['auc_guard'])} ({d:+.1f} pt)."]
        mech = ("At s=3.0 the K=1 pipeline needs more than one CPU-second per video-second, so its queue builds up (peak wait above) and late windows are dropped, "
                "which is why its delivered score collapses to the sentinel (AUC 50). The guarded cached pipeline bounds the wait (<= ~405 ms) at the price of shedding a few frames.")
        return Diagnosis("failed" if ctx["wait_max_s_full_rate"] > 1 else "mixed", "Queue overload of an unguarded K=1 pipeline", bullets, mech)
    raise ValueError(variant)


def diagnose_case(mode: str, ctx: dict, variant: str | None = None) -> Diagnosis:
    if mode == "small_person":
        return diagnose_small_person(ctx)
    if mode == "fragmentation":
        return diagnose_fragmentation(ctx)
    if mode == "scorer_limited":
        return diagnose_scorer_limited(ctx)
    if mode == "scheduling":
        return diagnose_scheduling(variant or "cache_helps", ctx)
    raise ValueError(f"unknown failure mode {mode!r}")


# ============================================================================================ context builders (one clip / event)
def build_small_person_context(dataset: str, clip: str, gate: float = 0.25) -> dict:
    """Evidence for failure mode 1 on one clip: AUCs, frames where ML0 finds a <48 px person stock misses, best frame to show."""
    names, labels = dataset_clips(dataset)
    ci = names.index(clip)
    gt = labels[ci]
    ser = load_e01_series()
    sl = clip_slice(dataset, clip)
    ctx = {
        "auc_stock": clip_auc(gt, ser[f"{dataset}|stock|S4"][sl]), "auc_ml0": clip_auc(gt, ser[f"{dataset}|{E01_ML0[dataset]}|S4"][sl]),
        "n_gt_frames": int(gt.sum()), "n_unmatched_gt_frames": 0, "gate": gate, "focus_frame": None,
        "miss_frames": np.zeros(len(gt), dtype=np.int32), "available": False,
    }
    ml0 = load_clip_detections(ml0_spec_id(dataset), dataset, clip)
    stock = load_clip_detections("stock-yolov8n-640", dataset, clip)
    if ml0 is None or stock is None:
        return ctx
    miss = small_person_miss_frames(ml0, stock, gate=gate)
    n = len(miss["n_unmatched"])
    gt_n = gt[:n].astype(bool)
    um = np.where(gt_n, miss["n_unmatched"], 0)
    ml0_n = np.array([int((f["conf"] >= gate).sum()) for f in ml0[:n]])
    stock_n = np.array([int((f["conf"] >= gate).sum()) for f in stock[:n]])
    um_any = np.where(gt_n, miss["n_unmatched_any"], 0)
    ctx.update(available=True, miss_frames=miss["n_unmatched"], n_unmatched_gt_frames=int((um > 0).sum()),
               mean_det_gt_stock=float(stock_n[gt_n].mean()) if gt_n.any() else 0.0,
               mean_det_gt_ml0=float(ml0_n[gt_n].mean()) if gt_n.any() else 0.0,
               n_unmatched_any_gt_frames=int((um_any > 0).sum()))
    pool = um if um.max() > 0 else miss["n_unmatched"]
    if pool.max() > 0:
        f = int(np.argmax(pool))
        ctx.update(focus_frame=f, focus_heights=miss["heights"][f], focus_boxes=miss["boxes"][f],
                   stock_n_at_focus=int((stock[f]["conf"] >= gate).sum()), ml0_n_at_focus=int((ml0[f]["conf"] >= gate).sum()))
    return ctx


def build_fragmentation_context(dataset: str, clip: str, row: pd.Series | None = None, run_replay: bool = True) -> dict:
    """Evidence for failure mode 2 on one clip: AUCs, low-confidence gap-bridging frames, optional BoT-SORT replay statistics."""
    names, labels = dataset_clips(dataset)
    ci = names.index(clip)
    gt = labels[ci]
    b = load_run_bundle("e05", dataset)
    ctx = {
        "auc_default": clip_auc(gt, b.clip_series("ML0-Default-K1", K1_CELL, ci)), "auc_two_stage": clip_auc(gt, b.clip_series("Full-Rate", K1_CELL, ci)),
        "n_bridge_frames": 0, "n_bridge_runs": 0, "longest_bridge": 0, "bridge_frames": [], "focus_frame": None, "replay": None,
    }
    stats = load_e01_clip_stats(dataset)
    if clip in stats.index:
        r = stats.loc[clip]
        ctx["default_gaps"] = float(r["gaps_1"] + r["gaps_2"] + r["gaps_ge3"])
        ctx["default_gap_frames"] = float(r["gap_frames"])
    frames = load_clip_detections(ml0_spec_id(dataset), dataset, clip)
    if frames is not None:
        bridges = gap_bridging_frames(frames)
        fr = [x["frame"] for x in bridges]
        runs = runs_of(fr)
        ctx.update(n_bridge_frames=len(set(fr)), n_bridge_runs=len(runs), longest_bridge=max([e - s + 1 for s, e in runs], default=0), bridge_frames=bridges)
        in_gt = [f for f in fr if f < len(gt) and gt[f]]
        pick = in_gt or fr
        if pick:
            ctx["focus_frame"] = int(pick[len(pick) // 2])
        if run_replay:
            d, lk = replay_track_stats(dataset, clip, "default"), replay_track_stats(dataset, clip, "locked")
            if d is not None and lk is not None:
                ctx["replay"] = {"default_tracks": d["n_tracks"], "default_longest": d["longest_track"], "default_windows": d["n_windows"],
                                 "locked_tracks": lk["n_tracks"], "locked_longest": lk["longest_track"], "locked_windows": lk["n_windows"]}
    return ctx


def build_scorer_context(dataset: str, row: pd.Series) -> dict:
    """Evidence for failure mode 3 from one row of `rank_scorer_limited`."""
    ev = row.to_dict()
    ev["event"] = int(ev["event"])
    names, _ = dataset_clips(dataset)
    b = load_run_bundle("e05", dataset)
    series = b.clip_series("Full-Rate", K1_CELL, names.index(ev["clip"]))
    seg = series[int(ev["start_frame"]):int(ev["end_frame"]) + 1]
    focus = int(ev["start_frame"] + int(np.argmax(seg))) if len(seg) else int(ev["start_frame"])
    return {"event": ev, "seg_len": SEG_LEN[dataset], "focus_frame": focus}


def build_scheduling_context(dataset: str, variant: str, row: pd.Series) -> dict:
    ctx = {k: (None if (isinstance(v, float) and not np.isfinite(v)) else v) for k, v in row.to_dict().items()}
    clip = ctx["clip"]
    names, labels = dataset_clips(dataset)
    ci = names.index(clip)
    gt = labels[ci]
    focus = int(np.flatnonzero(gt)[len(np.flatnonzero(gt)) // 2]) if gt.any() else len(gt) // 2
    if variant == "queue_pi5":
        pc = pi5_clip(load_pi5_live(), "fixed_k1", clip)
        if pc is not None and len(pc["wait_ms"]):
            focus = int(pc["frame"][int(np.argmax(pc["wait_ms"]))])
    ctx["focus_frame"] = focus
    return ctx


# ============================================================================================ timeline data + Altair charts
Y_MODES = {
    "raw": "Raw anomaly score",
    "ratio": "Score / alert threshold theta (alert when > 1)",
    "minmax": "Min-max per curve (shape only)",
}


def build_timeline_frame(dataset: str, clip: str, series_ids: list[str] | tuple[str, ...], budget_cell: str = DEFAULT_BUDGET_CELL,
                         y_mode: str = "raw") -> tuple[pd.DataFrame, list[str], dict[str, str]]:
    """Long frame `frame, series, value, raw` of the selected score series of one clip.

    Returns `(frame, labels of unavailable series, {label: provenance note})`. `y_mode`: raw | ratio (score / theta of that series' world) | minmax.
    """
    rows = []
    missing: list[str] = []
    notes: dict[str, str] = {}
    for sid in series_ids:
        sd = SERIES_BY_ID[sid]
        got = get_clip_series(dataset, sid, clip, budget_cell)
        if got is None:
            missing.append(sd.label)
            continue
        raw = np.asarray(got["score"], dtype=np.float64)
        if y_mode == "ratio":
            val = raw / got["theta"] if got["theta"] else np.full_like(raw, np.nan)
        elif y_mode == "minmax":
            lo, hi = float(np.nanmin(raw)), float(np.nanmax(raw))
            val = (raw - lo) / (hi - lo) if hi > lo else np.zeros_like(raw)
        else:
            val = raw
        notes[sd.label] = got["note"]
        rows.append(pd.DataFrame({"frame": got["frame"], "series": sd.label, "value": val, "raw": raw}))
    frame = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["frame", "series", "value", "raw"])
    return frame, missing, notes


def _event_frame(events: list[GtEvent]) -> pd.DataFrame:
    return pd.DataFrame({
        "start": [e.start_frame for e in events], "end": [e.end_frame + 1 for e in events],
        "label": [f"GT event {e.index}: f{e.start_frame}-{e.end_frame} ({e.start_s:.2f}-{e.end_s:.2f} s) [{', '.join(e.tags)}]" for e in events],
    })


def make_timeline_chart(df: pd.DataFrame, events: list[GtEvent], cursor_frame: int | None = None, y_title: str = "anomaly score",
                        threshold: float | None = None, height: int = 300, step: bool = False, n_frames: int | None = None):
    """Interactive (zoom/pan) Altair line chart: red GT-event bands, one line per series, optional alert threshold rule and frame cursor."""
    import altair as alt

    if df is None or df.empty:
        return None
    layers = []
    if events:
        layers.append(alt.Chart(_event_frame(events)).mark_rect(color="#ff4b4b", opacity=0.18).encode(
            x=alt.X("start:Q", title="frame"), x2="end:Q", tooltip=["label:N"]))
    scale_x = alt.Scale(domain=[0, max(1, (n_frames or int(df["frame"].max() + 1)) - 1)], nice=False)
    line = alt.Chart(df).mark_line(strokeWidth=1.8, interpolate="step-after" if step else "linear").encode(
        x=alt.X("frame:Q", title="frame", scale=scale_x),
        y=alt.Y("value:Q", title=y_title),
        color=alt.Color("series:N", legend=alt.Legend(orient="bottom", columns=2, title=None)),
        tooltip=[alt.Tooltip("series:N"), alt.Tooltip("frame:Q"), alt.Tooltip("value:Q", format=".3f")] + ([alt.Tooltip("raw:Q", format=".3f", title="raw score")] if "raw" in df.columns else []),
    )
    layers.append(line)
    if threshold is not None:
        layers.append(alt.Chart(pd.DataFrame({"y": [threshold]})).mark_rule(color="#444", strokeDash=[6, 4]).encode(y="y:Q"))
    if cursor_frame is not None:
        layers.append(alt.Chart(pd.DataFrame({"x": [cursor_frame]})).mark_rule(color="#ff7f0e", strokeWidth=2).encode(x="x:Q"))
    return alt.layer(*layers).properties(height=height).interactive(bind_y=False)


def active_and_window_frame(dataset: str, clip: str, trackers: tuple[str, ...] = ("default", "locked")) -> pd.DataFrame | None:
    """Long frame of per-frame active-track count and scorable (landing) window count under the default and two-stage tracker replays."""
    rows = []
    label = {"default": "default BoT-SORT", "locked": "2-stage BoT-SORT"}
    for tr in trackers:
        st_ = replay_track_stats(dataset, clip, tr)
        if st_ is None:
            return None
        n = len(st_["active"])
        rows.append(pd.DataFrame({"frame": np.arange(n), "series": f"active persons ({label[tr]})", "value": st_["active"]}))
        rows.append(pd.DataFrame({"frame": np.arange(n), "series": f"scorable windows landing ({label[tr]})", "value": st_["landing"]}))
    return pd.concat(rows, ignore_index=True)
