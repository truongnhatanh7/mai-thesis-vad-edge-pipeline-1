"""Q2-P1-E03 (Core ablation): fixed / motion-only / adaptive-only / cache-only / independent / joint at matched CPU cost.

    uv run --extra pose python -m edgelab.analysis.q2_p1_e03 motion   # per-frame cheap observers + static statistics (cached on disk)
    uv run --extra pose python -m edgelab.analysis.q2_p1_e03 tables   # exact run tables (tracker re-created per run start), windows scored once
    uv run --extra pose python -m edgelab.analysis.q2_p1_e03 lock     # constants derived from test_dev -> Q2-P1-E03_ablation/config.yaml
    uv run --extra pose python -m edgelab.analysis.q2_p1_e03 run      # tune on dev, calibrate costs, evaluate -> results.csv, decisions.jsonl, summary.json
    uv run --extra pose python -m edgelab.analysis.q2_p1_e03 report   # summary.json -> report.md

No training. Builds on E00 (causal S3/S4 protocol, splits, window engine), E01 (locked frontend / tracker) and E02 (VM service-time pools,
true time-causal FIFO queue, fixed baselines). New here:

  * run tables      : for every even start frame `c` the locked tracker is run on the genuine detections of the next `L + Delta - 1` frames and every
                      window is scored once; a *run* of detector frames is served by chains of such tables, so every policy that stays in
                      intact windows is replayed exactly (a re-created tracker per run start), without re-running tracker/scorer per policy
  * C1 / C2         : two cross-frame reuse mechanisms (static-validity detector skip; cross-window scorer reuse) with source timestamps,
                      validity, refreshes, numerical error and net CPU
  * controllers     : Fixed burst F(W), Motion-Only, Adaptive-Only, Cache-Only, Independent-Combo, Joint-Controller on one FIFO worker
  * matched cost    : every arm's rate knob is bisected so the realised total CPU cost (observers and cache checks included) equals beta * C_full(s)

Pure functions (patterns, pools, queue, controllers, metrics) are unit-tested without data in tests/test_q2_p1_e03.py.
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
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from edgelab.analysis import q2_p1_e00 as e00
from edgelab.analysis import q2_p1_e01 as e01
from edgelab.analysis import q2_p1_e02 as e02

DATASETS = e00.DATASETS
SEG_LENS = e00.SEG_LENS
FPS = e00.FPS
SEED = e00.SEED
POOL_SEED = e02.POOL_SEED
N_BOOT = 1000
EXP_DIR = e00.EXPERIMENTS_ROOT / "Q2-P1-E03_ablation"
DATA_ROOT = e00.DATA_ROOT
WORK_DIR = DATA_ROOT / "q2-p1-e03"
MOTION_DIR = WORK_DIR / "motion_cache"
SHT_FRAMES_ROOT = DATA_ROOT / "shanghaitech/shanghaitech/testing/frames"
MACHINE_TAG = e02.MACHINE_TAG
PRIMARY = e02.PRIMARY
TRACKER = e02.TRACKERS["locked"]
GATE_FLOOR = e02.GATE_FLOOR
CAP_MS = e02.CAP_MS

GRID_H, GRID_W = 24, 40  # pooled thumbnail of the 4x down-sampled luma frame
A1 = 4  # C1: maximum age (frames) of a reused detector output
A2 = 6  # C2: maximum age (frames) of a reused window score
LEVELS = (1, 2, 3)  # C1 / C2 threshold levels (quantiles below)
C1_QUANTILES = (0.10, 0.25, 0.50)
C2_QUANTILES = (0.15, 0.35, 0.60)
START_GRID = 2  # runs start on even clip-local frames (decision epoch of the run tables)
DELTA = {ds: SEG_LENS[ds] // 2 for ds in DATASETS}  # chain hand-off (frames)
EXT = {ds: DELTA[ds] - 1 for ds in DATASETS}  # extra window end offsets served by one chain table
RMAX = {ds: SEG_LENS[ds] + DELTA[ds] - 1 for ds in DATASETS}  # frames covered by one table

OBS_MS_EXP023 = 0.366  # EXP-023 `observer_cheap_motion` (856x480), ms/frame
OBS_MS = {"ShanghaiTech": 0.366, "UBnormal": 0.370}  # max(EXP-023 figure, measured in `lock`: SHT 0.361, UB 0.368 ms on this host)
C1_CHECK_MS = {"ShanghaiTech": 0.15, "UBnormal": 0.26}  # thumbnail pooling + ROI / exterior check per in-run frame (measured 0.14 / 0.25 + margin)
C2_CHECK_MS = 0.01  # displacement bookkeeping per candidate window (x s)
LATE_CAP_MS = CAP_MS

CELLS = ([(2.0, b) for b in (0.25, 0.33, 0.50, 0.67, 0.75, 1.00)] + [(2.5, b) for b in (0.50, 0.75, 1.00)] +
         [(3.0, b) for b in (0.50, 0.75, 1.00)])
TUNING_CELLS = [(2.0, 0.33), (2.0, 0.75), (3.0, 1.00)]
ARMS = ("Fixed", "Motion-Only", "Adaptive-Only", "Cache-C1", "Cache-C2", "Cache-Only", "Independent-Combo", "Joint-Controller")
HEADLINE_ARMS = ("Fixed", "Motion-Only", "Adaptive-Only", "Cache-Only", "Independent-Combo", "Joint-Controller")


def plain(obj):
    return e00.to_plain(obj)


def now() -> str:
    return time.strftime("%H:%M:%S")


def cell_id(s: float, b: float) -> str:
    return f"s{s:.1f}_b{b:.2f}"


# =====================================================================================================================
# 1. Cheap observers (pure) and per-frame statistics
# =====================================================================================================================
def luma_small(gray: np.ndarray, ds: int = 4) -> np.ndarray:
    return gray[::ds, ::ds]


def motion_observer(prev_small: np.ndarray, curr_small: np.ndarray, band: float = 0.10) -> tuple[float, float, float]:
    """EXP-023 `measure_CheapMotionOnly` statistic on already down-sampled luma frames + a border-band activity.

    -> (motion_score, border_active_fraction, |global luma shift|). motion = q90/255 + 2 * fraction of pixels moving > 12 grey levels after
    median luma-shift compensation; border = the same activity fraction restricted to the outer `band` of the frame (entry cue)."""
    raw = curr_small.astype(np.float32) - prev_small.astype(np.float32)
    shift = float(np.median(raw))
    comp = np.abs(raw - shift)
    q90 = float(np.quantile(comp, 0.90))
    active = comp > 12.0
    motion = q90 / 255.0 + 2.0 * float(active.mean())
    h, w = comp.shape
    bh, bw = max(1, int(round(h * band))), max(1, int(round(w * band)))
    mask = np.zeros((h, w), dtype=bool)
    mask[:bh], mask[-bh:], mask[:, :bw], mask[:, -bw:] = True, True, True, True
    return motion, float(active[mask].mean()), abs(shift)


def thumb(small: np.ndarray) -> np.ndarray:
    import cv2

    return cv2.resize(small, (GRID_W, GRID_H), interpolation=cv2.INTER_AREA)


def box_cells(boxes: np.ndarray, width: float, height: float, dilate: int = 1) -> np.ndarray:
    """Boolean (GRID_H, GRID_W) mask of the thumbnail cells touched by `boxes` (x1, y1, x2, y2 in pixels), dilated by `dilate` cells."""
    mask = np.zeros((GRID_H, GRID_W), dtype=bool)
    for x1, y1, x2, y2 in np.asarray(boxes, dtype=float).reshape(-1, 4):
        c0, c1 = int(math.floor(x1 / width * GRID_W)) - dilate, int(math.floor(x2 / width * GRID_W)) + dilate
        r0, r1 = int(math.floor(y1 / height * GRID_H)) - dilate, int(math.floor(y2 / height * GRID_H)) + dilate
        mask[max(r0, 0):min(r1, GRID_H - 1) + 1, max(c0, 0):min(c1, GRID_W - 1) + 1] = True
    return mask


def static_stats(thumbs: np.ndarray, boxes_by_frame: list[np.ndarray], width: float, height: float, a1: int = A1
                 ) -> tuple[np.ndarray, np.ndarray]:
    """C1 validity statistics. For every source frame s and age k = 1..a1: mean |thumb[s+k] - thumb[s]| inside the person ROI of s
    (boxes of s) and the 95th percentile outside it (a person entering breaks it). NaN where s + k does not exist. -> roi, ext (n, a1)."""
    n = len(thumbs)
    th = thumbs.astype(np.float32)
    roi = np.full((n, a1), np.nan, dtype=np.float32)
    ext = np.full((n, a1), np.nan, dtype=np.float32)
    for s in range(n):
        mask = box_cells(boxes_by_frame[s], width, height) if len(boxes_by_frame[s]) else np.zeros((GRID_H, GRID_W), dtype=bool)
        inside, outside = mask, ~mask
        for k in range(1, a1 + 1):
            if s + k >= n:
                break
            d = np.abs(th[s + k] - th[s])
            roi[s, k - 1] = float(d[inside].mean()) if inside.any() else 0.0
            ext[s, k - 1] = float(np.percentile(d[outside], 95)) if outside.any() else 0.0
    return roi, ext


def reuse_pattern(roi: np.ndarray, ext: np.ndarray, c: int, n_run: int, tau_roi: float, tau_ext: float, a1: int = A1
                  ) -> tuple[np.ndarray, np.ndarray]:
    """C1 pattern of a run chain starting at clip frame `c` over `n_run` frames: -> (reused (n_run,) bool, source offset (n_run,)).

    Offset 0 always runs the detector. Frame `c + o` reuses the source `s` (the last detector frame) iff its age `o - s <= a1` and the ROI /
    exterior statistics of (s, o - s) are at most the thresholds; otherwise the detector runs and becomes the new source. Causal: offset `o`
    depends on frames <= c + o only."""
    reused = np.zeros(n_run, dtype=bool)
    src = np.arange(n_run, dtype=np.int64)
    s = 0
    for o in range(1, n_run):
        k = o - s
        if k <= a1 and roi[c + s, k - 1] <= tau_roi and ext[c + s, k - 1] <= tau_ext:  # NaN compares False -> detector runs
            reused[o], src[o] = True, s
        else:
            s = o
    return reused, src


def pack_bits(mask: np.ndarray) -> int:
    return int(sum(1 << i for i in np.nonzero(mask)[0]))


def unpack_bit(bits: int, i: int) -> bool:
    return bool((bits >> i) & 1)


def thresholds_from_stats(roi1: np.ndarray, ext1: np.ndarray, quantiles=C1_QUANTILES) -> dict:
    """tau_roi / tau_ext per level from the one-frame statistics of the dev frames (finite values only)."""
    r, x = roi1[np.isfinite(roi1)], ext1[np.isfinite(ext1)]
    return {"roi": [float(np.quantile(r, q)) for q in quantiles], "ext": [float(np.quantile(x, q)) for q in quantiles],
            "quantiles": list(quantiles)}


# ---- motion stage ----------------------------------------------------------------------------------------------------------------
def motion_path(ds: str, clip: str) -> Path:
    return MOTION_DIR / ds / f"{clip}.npz"


def iter_gray(ds: str, clip: str):
    import cv2

    if ds == "ShanghaiTech":
        for f in sorted((SHT_FRAMES_ROOT / clip).glob("*.jpg")):
            yield cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        return
    cap = cv2.VideoCapture(str(e00.UB_TEST_VIDEOS / f"{clip}.mp4"))
    while True:
        ok, img = cap.read()
        if not ok:
            break
        yield cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cap.release()


def _motion_clip(job: tuple) -> dict:
    ds, clip = job
    prev, smalls = None, []
    motion, border, shift, thumbs, obs_ms = [], [], [], [], []
    size = None
    for gray in iter_gray(ds, clip):
        size = (gray.shape[1], gray.shape[0])
        t0 = time.perf_counter()
        small = luma_small(gray)
        if prev is None:
            m, b, sh = float("nan"), float("nan"), float("nan")
        else:
            m, b, sh = motion_observer(prev, small)
        obs_ms.append((time.perf_counter() - t0) * 1e3)
        prev = small
        motion.append(m)
        border.append(b)
        shift.append(sh)
        thumbs.append(thumb(small))
    arrays = {"motion": np.array(motion, dtype=np.float32), "border": np.array(border, dtype=np.float32), "shift": np.array(shift, dtype=np.float32),
              "thumb": np.stack(thumbs).astype(np.uint8), "obs_ms": np.array(obs_ms, dtype=np.float32), "size": np.array(size)}
    path = motion_path(ds, clip)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    return {"clip": clip, "n": len(motion), "obs_ms_mean": float(np.mean(obs_ms[1:])) if len(obs_ms) > 1 else float("nan")}


def stage_motion(ds: str, procs: int, log=print, force: bool = False) -> dict:
    names, labels = e00.dataset_clips(ds)
    todo = [(ds, c) for c in names if force or not motion_path(ds, c).exists()]
    log(f"[{now()}] motion cache {ds}: {len(todo)}/{len(names)} clips to compute")
    res = []
    if todo:  # spawn (not fork): OpenCV's ffmpeg decoder deadlocks in forked workers on macOS
        with mp.get_context("spawn").Pool(min(procs, len(todo))) as pool:
            res = pool.map(_motion_clip, todo, chunksize=1)
    bad = []
    for c, lab in zip(names, labels):
        z = np.load(motion_path(ds, c))
        if len(z["motion"]) != len(lab):
            bad.append((c, len(z["motion"]), len(lab)))
    if bad:
        raise AssertionError(f"motion cache frame count != labels: {bad[:3]}")
    return {"computed": len(res), "obs_ms_mean": float(np.mean([r["obs_ms_mean"] for r in res])) if res else None}


def load_motion(ds: str, names: list[str]) -> dict:
    out = {"motion": [], "border": [], "thumb": [], "size": None}
    for c in names:
        z = np.load(motion_path(ds, c))
        out["motion"].append(z["motion"].astype(np.float64))
        out["border"].append(z["border"].astype(np.float64))
        out["thumb"].append(z["thumb"])
        out["size"] = tuple(int(v) for v in z["size"])
    return out


def bench_observer(ds: str, names: list[str], n_frames: int = 400) -> dict:
    """Re-measure the observer / ROI check / displacement bookkeeping on this host (medians, ms per frame)."""
    gray = list(iter_gray(ds, names[0]))[: n_frames + 1]
    smalls = [luma_small(g) for g in gray]
    t_obs, t_thumb, t_roi = [], [], []
    boxes = np.array([[100.0, 80.0, 180.0, 300.0], [400.0, 100.0, 460.0, 330.0]])
    size = (gray[0].shape[1], gray[0].shape[0])
    th = [thumb(s) for s in smalls]
    for i in range(1, len(gray)):
        t0 = time.perf_counter()
        sm = luma_small(gray[i])
        motion_observer(smalls[i - 1], sm)
        t1 = time.perf_counter()
        thumb(sm)
        t2 = time.perf_counter()
        mask = box_cells(boxes, *size)
        d = np.abs(th[i].astype(np.float32) - th[i - 1].astype(np.float32))
        d[mask].mean(), np.percentile(d[~mask], 95)
        t3 = time.perf_counter()
        t_obs.append((t1 - t0) * 1e3), t_thumb.append((t2 - t1) * 1e3), t_roi.append((t3 - t2) * 1e3)
    return {"observer_ms_median": float(np.median(t_obs)), "thumb_ms_median": float(np.median(t_thumb)), "roi_check_ms_median": float(np.median(t_roi)),
            "frames": len(t_obs)}


# =====================================================================================================================
# 2. Static statistics (C1 validity) per dataset
# =====================================================================================================================
_G: dict = {}


def static_path(ds: str) -> Path:
    return WORK_DIR / f"static_{ds}.npz"


def _static_clip(ci: int):
    frames, motion = _G["frames"][ci], _G["motion"]
    width, height = motion["size"]
    boxes = [f["boxes"][f["conf"] >= 0.10] for f in frames]
    return static_stats(motion["thumb"][ci], boxes, width, height)


def stage_static(ds: str, ctx, procs: int, log=print, force: bool = False) -> dict:
    """roi / ext statistics of every clip (concatenated) and the level thresholds from the test_dev one-frame distribution."""
    path = static_path(ds)
    if path.exists() and not force:
        z = np.load(path)
        return {"roi": z["roi"], "ext": z["ext"], "thr": json.loads(str(z["thr"]))}
    arm = e01.arm_by_key(ctx.arms)[f"{ds}/{PRIMARY[ds]}"]
    names = ctx.names[ds]
    frames_by_clip, ids_by_clip = e01.load_frames_floor(arm.cache_dir, names, GATE_FLOOR)
    for ids in ids_by_clip:
        assert np.array_equal(ids, np.arange(len(ids))), "detector cache frame ids must be 0..n-1"
    _G.update(frames=frames_by_clip, motion=load_motion(ds, names))
    try:
        res = e01.pmap(_static_clip, list(range(len(names))), procs)
    finally:
        _G.clear()
    roi = np.concatenate([r for r, _ in res]).astype(np.float32)
    ext = np.concatenate([x for _, x in res]).astype(np.float32)
    n_frames = [len(x) for x in ctx.labels[ds]]
    offs = np.concatenate([[0], np.cumsum(n_frames)])
    dev = np.concatenate([np.arange(offs[i], offs[i + 1]) for i in ctx.dev_idx[ds]])
    thr = thresholds_from_stats(roi[dev, 0], ext[dev, 0])
    np.savez_compressed(path, roi=roi, ext=ext, thr=np.array(json.dumps(thr)))
    log(f"[{now()}] static stats {ds}: thresholds {thr}")
    return {"roi": roi, "ext": ext, "thr": thr}


# =====================================================================================================================
# 3. Run tables (exact): tracker re-created at every even start, every window scored once
# =====================================================================================================================
def tables_path(ds: str) -> Path:
    return WORK_DIR / f"tables_{ds}.npz"


def _hash64(buf: bytes) -> int:
    return int.from_bytes(hashlib.blake2b(buf, digest_size=8).digest(), "little", signed=False)


def window_meta(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """(height-normalised hip-centred last pose (17, 2), velocity over two frames (17, 2), mean keypoint confidence) of a (L, 17, 3) window."""
    last = arr[-1, :, :2].astype(np.float64)
    ok = arr[-1, :, 2] > 0.1
    ys = last[ok, 1] if ok.any() else last[:, 1]
    h = max(float(ys.max() - ys.min()), 1.0)
    hip = last[11:13].mean(axis=0)
    return (last - hip) / h, (last - arr[-3, :, :2].astype(np.float64)) / h, float(arr[:, :, 2].mean())


def window_distance(pose_a, vel_a, pose_b, vel_b) -> float:
    """Pose displacement + velocity change between two windows' last frames (mean over joints of the L2 distance, height-normalised)."""
    return float(np.linalg.norm(pose_a - pose_b, axis=1).mean() + np.linalg.norm(vel_a - vel_b, axis=1).mean())


def _table_clip(ci: int) -> dict:
    G = _G
    ds = G["ds"]
    L, delta, ext, rmax = SEG_LENS[ds], DELTA[ds], EXT[ds], RMAX[ds]
    frames, ids = G["frames"][ci], G["ids"][ci]
    roi, ext_stat = G["roi"][ci], G["ext"][ci]
    thr = G["thr"]
    n = len(frames)
    pool: list[np.ndarray] = []
    index: dict[int, int] = {}
    heads: list[int] = []
    tails: list[int] = []
    rows_c, rows_e, rows_uid = [], [], []
    starts = list(range(0, n - L + 1, START_GRID))
    bits = np.zeros((len(starts), len(LEVELS)), dtype=np.uint64)
    hist = np.zeros((len(LEVELS), 60), dtype=np.int64)  # ZOH pose error (height-normalised) of reused poses, bins of 0.01
    n_reused_pose = np.zeros(len(LEVELS), dtype=np.int64)
    n_unseen = np.zeros(len(LEVELS), dtype=np.int64)

    def intern(arr: np.ndarray) -> int:
        key = _hash64(arr.tobytes())
        got = index.get(key)
        if got is None:
            got = index[key] = len(pool)
            pool.append(arr)
            heads.append(_hash64(arr[:-1].tobytes()))
            tails.append(_hash64(arr[1:].tobytes()))
        return got

    for si, c in enumerate(starts):
        end = min(n, c + rmax)
        tracks = e01.replay_cell(frames[c:end], ids[c:end], TRACKER)
        nrun = end - c
        pats = [reuse_pattern(roi, ext_stat, c, nrun, thr["roi"][li], thr["ext"][li]) for li in range(len(LEVELS))]
        for li, (re_mask, _) in enumerate(pats):
            bits[si, li] = np.uint64(pack_bits(re_mask))
        for pid, (f, kp) in tracks.items():
            ws = e00.complete_window_starts(f, L)
            if len(ws) == 0:
                continue
            last = f[ws + L - 1]
            e = last - (c + L - 1)
            fidx = None
            for st, ee in zip(ws[e <= ext], e[e <= ext]):
                arr = np.array(kp[st:st + L], dtype=np.float64)
                uids = [intern(arr)]
                offs = f[st:st + L] - c
                for li, (re_mask, src) in enumerate(pats):
                    rm = re_mask[offs]
                    if not rm.any():
                        uids.append(uids[0])
                        continue
                    if fidx is None:
                        fidx = {int(x): i for i, x in enumerate(f)}
                    arr2 = arr.copy()
                    for j in np.nonzero(rm)[0]:
                        i_src = fidx.get(int(c + src[offs[j]]))
                        if i_src is None:  # person not seen at the source frame: validity miss (keeps the genuine pose, counted)
                            n_unseen[li] += 1
                            continue
                        arr2[j] = kp[i_src]
                        ok = arr[j, :, 2] > 0.1
                        ys = arr[j, ok, 1] if ok.any() else arr[j, :, 1]
                        h = max(float(ys.max() - ys.min()), 1.0)
                        err = float(np.linalg.norm(arr2[j, :, :2] - arr[j, :, :2], axis=1).mean() / h)
                        hist[li, min(int(err / 0.01), 59)] += 1
                        n_reused_pose[li] += 1
                    uids.append(intern(arr2))
                rows_c.append(si)
                rows_e.append(int(ee))
                rows_uid.append(uids)
    pool_arr = np.stack(pool) if pool else np.zeros((0, L, 17, 3), dtype=np.float32)
    meta_pose, meta_vel, qual = (np.zeros((len(pool), 17, 2)), np.zeros((len(pool), 17, 2)), np.zeros(len(pool)))
    last_pose_frame = np.zeros(len(pool), dtype=np.int64)
    for i, arr in enumerate(pool):
        meta_pose[i], meta_vel[i], qual[i] = window_meta(arr)
    order = np.lexsort((np.array(rows_e, dtype=np.int64), np.array(rows_c, dtype=np.int64))) if rows_c else np.zeros(0, dtype=np.int64)
    return {"pool": pool_arr, "pose": meta_pose.astype(np.float32), "vel": meta_vel.astype(np.float32), "qual": qual.astype(np.float32),
            "head": np.array(heads, dtype=np.uint64), "tail": np.array(tails, dtype=np.uint64), "starts": np.array(starts, dtype=np.int32),
            "row_c": np.array(rows_c, dtype=np.int32)[order], "row_e": np.array(rows_e, dtype=np.int8)[order],
            "row_uid": (np.array(rows_uid, dtype=np.int32).reshape(-1, 1 + len(LEVELS)) if rows_c else np.zeros((0, 1 + len(LEVELS)), np.int32))[order],
            "bits": bits, "hist": hist, "n_reused_pose": n_reused_pose, "n_unseen": n_unseen}


def build_lineage(res: dict, last_pose_frame_of_uid: dict | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Lineage id and one-step displacement of every *base* pool entry of one clip.

    A window `w` continues the window `p` of the same person iff the first L-1 poses of `w` equal the last L-1 poses of `p` (identical
    per-frame cache entries, so this links windows across run tables). Windows are processed in order of their last-pose frame; `step[w]` =
    `window_distance(w, p)` (NaN without predecessor)."""
    pool_n = len(res["qual"])
    lin = np.full(pool_n, -1, dtype=np.int64)
    step = np.full(pool_n, np.nan, dtype=np.float32)
    base = np.unique(res["row_uid"][:, 0]) if len(res["row_uid"]) else np.zeros(0, dtype=np.int64)
    # order by the frame of the last pose: the pool entries were interned in start order, windows with later last frames have later ends
    # in their rows; use (min row e + start) as the key
    key = {}
    for c, e, u in zip(res["row_c"], res["row_e"], res["row_uid"][:, 0]):
        k = int(res["starts"][c]) + int(e)
        if u not in key or k < key[u]:
            key[u] = k
    by_tail: dict[int, int] = {}
    nxt = 0
    for u in sorted(base, key=lambda x: (key[int(x)], int(x))):
        u = int(u)
        p = by_tail.get(int(res["head"][u]))
        if p is not None and p != u:
            lin[u] = lin[p]
            step[u] = window_distance(res["pose"][u], res["vel"][u], res["pose"][p], res["vel"][p])
        else:
            lin[u] = nxt
            nxt += 1
        by_tail.setdefault(int(res["tail"][u]), u)
    return lin, step


def stage_tables(ds: str, ctx, procs: int, log=print, force: bool = False) -> dict:
    path = tables_path(ds)
    if path.exists() and not force:
        log(f"[{now()}] tables {ds}: cached")
        return load_tables_npz(ds)
    names, labels = ctx.names[ds], ctx.labels[ds]
    arm = e01.arm_by_key(ctx.arms)[f"{ds}/{PRIMARY[ds]}"]
    scorer = ctx.scorers[ds]
    st = stage_static(ds, ctx, procs, log)
    frames_by_clip, ids_by_clip = e01.load_frames_floor(arm.cache_dir, names, GATE_FLOOR)
    n_frames = np.array([len(x) for x in labels])
    offs = np.concatenate([[0], np.cumsum(n_frames)])
    n_in = np.concatenate([[int((f["conf"] >= TRACKER.gate).sum()) for f in fr] for fr in frames_by_clip]).astype(np.int16)
    assert len(n_in) == int(offs[-1])
    roi = [st["roi"][offs[i]:offs[i + 1]] for i in range(len(names))]
    ext = [st["ext"][offs[i]:offs[i + 1]] for i in range(len(names))]
    _G.update(ds=ds, frames=frames_by_clip, ids=ids_by_clip, roi=roi, ext=ext, thr=st["thr"])
    t0 = time.perf_counter()
    cols: dict[str, list] = {k: [] for k in ("row_norm", "row_lin", "row_step", "row_qual", "row_e", "row_uid", "row_start", "clip_of_start",
                                              "start_frame", "bits")}
    ptr_counts = []
    hist = np.zeros((len(LEVELS), 60), dtype=np.int64)
    n_reused = np.zeros(len(LEVELS), dtype=np.int64)
    n_unseen = np.zeros(len(LEVELS), dtype=np.int64)
    n_pool = 0
    lin_off = 0
    start_off = 0
    try:
        with mp.get_context("fork").Pool(procs) as pool:
            for ci, res in enumerate(pool.imap(_table_clip, range(len(names)), chunksize=1)):
                norm = scorer.score(res["pool"].astype(np.float64)) if len(res["pool"]) else np.zeros(0)
                lin, step = build_lineage(res)
                uid = res["row_uid"]
                nrow = len(uid)
                cols["row_norm"].append(norm[uid].astype(np.float32))
                cols["row_lin"].append((lin[uid[:, 0]] + lin_off).astype(np.int32))
                cols["row_step"].append(step[uid[:, 0]])
                cols["row_qual"].append(res["qual"][uid[:, 0]])
                cols["row_e"].append(res["row_e"])
                cols["row_start"].append(res["row_c"] + start_off)
                cols["start_frame"].append(res["starts"])
                cols["clip_of_start"].append(np.full(len(res["starts"]), ci, dtype=np.int32))
                cols["bits"].append(res["bits"])
                lin_off += int(lin.max()) + 1 if len(lin) and lin.max() >= 0 else 0
                start_off += len(res["starts"])
                hist += res["hist"]
                n_reused += res["n_reused_pose"]
                n_unseen += res["n_unseen"]
                n_pool += len(res["pool"])
                if (ci + 1) % 10 == 0 or ci + 1 == len(names):
                    log(f"[{now()}]   tables {ds}: {ci + 1}/{len(names)} clips, {n_pool} unique windows, {time.perf_counter() - t0:.0f}s")
    finally:
        _G.clear()
    out = {k: np.concatenate(v) for k, v in cols.items() if k not in ("row_uid",)}
    out["row_norm"] = out["row_norm"].reshape(-1, 1 + len(LEVELS))
    n_starts = start_off
    # CSR over (start, e): rows are sorted by (start, e); ptr[start * (EXT + 1) + e] .. ptr[... + 1]
    counts = np.zeros((n_starts, EXT[ds] + 1), dtype=np.int64)
    np.add.at(counts, (out["row_start"], out["row_e"].astype(np.int64)), 1)
    flat = np.concatenate([[0], np.cumsum(counts.reshape(-1))])
    out["row_ptr"] = flat
    out["hist"], out["n_reused_pose"], out["n_unseen"] = hist, n_reused, n_unseen
    out["n_pool"] = np.array(n_pool)
    out["n_in"] = n_in
    out["meta"] = np.array(json.dumps({"ds": ds, "L": SEG_LENS[ds], "delta": DELTA[ds], "ext": EXT[ds], "rmax": RMAX[ds], "thr": st["thr"],
                                       "tracker": TRACKER.label, "frontend": PRIMARY[ds], "build_s": time.perf_counter() - t0}))
    np.savez_compressed(path, **out)
    log(f"[{now()}] tables {ds}: {len(out['row_norm'])} rows, {n_pool} unique windows, {n_starts} starts in {time.perf_counter() - t0:.0f}s")
    return load_tables_npz(ds)


def load_tables_npz(ds: str) -> dict:
    z = np.load(tables_path(ds))
    out = {k: z[k] for k in z.files}
    out["meta"] = json.loads(str(out["meta"]))
    return out


# =====================================================================================================================
# 4. World (one dataset), per-scenario cost arrays, derived constants
# =====================================================================================================================
BURST_WINDOWS_PER_RUN = 2.0  # expected persons scored by a bare L-frame run (token reservation only; costs are always realised)
RUN_CAP_FACTOR = 4  # longest run (in L) a scheduler may extend to
BACKLOG_SHED_ON_MS, BACKLOG_SHED_OFF_MS, BACKLOG_THIN_MS = 400.0, 150.0, 200.0
BUCKET_CAP_BURSTS = 2.5
EVENT_MIN_FRAMES = 6  # an alert = score above the threshold for >= 6 consecutive frames
ALERT_FPR = 0.05
FORCE_PERIOD_FACTOR = 2  # P_max = 2 max(W_ref, L)
FB_DECAY_MS = 2000.0
A2_DEFAULT = A2


class World:
    """Everything a replay needs for one dataset: labels, run tables (python lists for the hot loop), motion statistics, service pools."""

    def __init__(self, ctx, ds: str, load_service: bool = True, tb: dict | None = None, pools: dict | None = None):
        arm = e01.arm_by_key(ctx.arms)[f"{ds}/{PRIMARY[ds]}"]
        self._setup(ds, ctx.names[ds], ctx.labels[ds], ctx.dev_idx[ds], ctx.conf_idx[ds], e01.arm_sentinel(ctx.proto, arm), load_service, tb, pools)

    @classmethod
    def from_parts(cls, ds: str, names: list, labels: list, dev_idx: list, conf_idx: list, sentinel: float, load_service: bool = True,
                   tb: dict | None = None, pools: dict | None = None) -> "World":
        """Build without an E01 context (tests, quick checks). `tb` / `pools` (E04): alternative run tables / service pools."""
        self = cls.__new__(cls)
        self._setup(ds, names, labels, dev_idx, conf_idx, sentinel, load_service, tb, pools)
        return self

    def _setup(self, ds: str, names, labels, dev_idx, conf_idx, sentinel: float, load_service: bool, tb: dict | None = None,
               pools: dict | None = None) -> None:
        self.ds, self.L, self.delta, self.ext = ds, SEG_LENS[ds], DELTA[ds], EXT[ds]
        self.names = list(names)
        self.labels = [np.asarray(x) for x in labels]
        self.n_frames = np.array([len(x) for x in self.labels], dtype=np.int64)
        self.offsets = np.concatenate([[0], np.cumsum(self.n_frames)])
        self.total = int(self.offsets[-1])
        self.dt = 1000.0 / FPS[ds]
        self.dev_idx, self.conf_idx = list(dev_idx), list(conf_idx)
        self.sentinel = sentinel
        tb = load_tables_npz(ds) if tb is None else tb
        self.tb = tb
        L = self.L
        starts_per_clip = np.array([(n - L) // START_GRID + 1 if n >= L else 0 for n in self.n_frames], dtype=np.int64)
        self.start_base = np.concatenate([[0], np.cumsum(starts_per_clip)])
        assert int(self.start_base[-1]) == len(tb["start_frame"]), "table starts do not match the clips"
        self.row_ptr = tb["row_ptr"].tolist()
        self.rn = [tb["row_norm"][:, i].tolist() for i in range(1 + len(LEVELS))]
        self.row_lin = tb["row_lin"].tolist()
        self.row_step = tb["row_step"].tolist()
        self.row_qual = tb["row_qual"].tolist()
        self.bits = [tb["bits"][:, i].tolist() for i in range(len(LEVELS))]
        self.n_in = tb["n_in"].astype(np.int64)
        mo = load_motion(ds, self.names)
        self.motion = np.concatenate(mo["motion"])
        self.border = np.concatenate(mo["border"])
        assert len(self.motion) == self.total
        self.mot_l, self.bor_l = self.motion.tolist(), self.border.tolist()
        self.size = mo["size"]
        self.K: dict = {}
        self.model = e02.ServiceModel(e02.build_pools(ds, PRIMARY[ds], "locked") if pools is None else pools) if load_service else None
        self.u = np.random.default_rng(POOL_SEED).random((self.total, 5))
        self._cost: dict = {}

    def clip_slice(self, ci: int) -> slice:
        return slice(int(self.offsets[ci]), int(self.offsets[ci + 1]))

    def cost(self, s: float) -> "CostArrays":
        if s not in self._cost:
            self._cost[s] = CostArrays(self, s)
        return self._cost[s]


class CostArrays:
    """Per-frame service (wall) and CPU times of every stage at scenario multiplier `s` (T=4, q=4), python lists for the hot loop.

    Frame g's draws use the same common random numbers as E02; the tracker pool follows the frame's detection count."""

    def __init__(self, world: World, s: float):
        m, u, q = world.model, world.u, float(e02.CORES)
        total = world.total
        self.s = s

        def draw(pool, col, mask=None):
            idx = np.minimum((u[:, col] * len(pool[0])).astype(np.int64), len(pool[0]) - 1)
            return pool[0][idx] * s, pool[1][idx] * s

        self.gw, self.gc = (a.tolist() for a in draw(m.stage_pool("grab", q), 0))
        self.rw, self.rc = (a.tolist() for a in draw(m.stage_pool("retrieve", q), 1))
        self.pw, self.pc = (a.tolist() for a in draw(m.stage_pool("pose", q), 2))
        tw, tc = np.zeros(total), np.zeros(total)
        for n in np.unique(world.n_in):
            sel = world.n_in == n
            w, c = draw(m.stage_pool("track", q, int(n)), 3)
            tw[sel], tc[sel] = w[sel], c[sel]
        self.tw, self.tc = tw.tolist(), tc.tolist()
        self.score_n = list(m.score_n)
        sw, sc = [], []
        for n in self.score_n:
            w, c = draw(m.stage_pool("score", q, n), 4)
            sw.append(w.tolist())
            sc.append(c.tolist())
        self.sw, self.sc = sw, sc
        top = 96
        self.k_of_n = [0] + [min(range(len(self.score_n)), key=lambda i, n=n: (abs(self.score_n[i] - n), self.score_n[i])) for n in range(1, top + 1)]
        self.k_top = len(self.score_n) - 1
        costs = world.K.get("costs", {"obs_ms": OBS_MS[world.ds], "c1_check_ms": C1_CHECK_MS[world.ds], "c2_check_ms": C2_CHECK_MS})
        self.obs_ms = costs["obs_ms"] * s  # the scheduler's own host cost is scaled by the Pi multiplier like every other stage (conservative)
        self.c1_ms = costs["c1_check_ms"] * s
        self.c2_ms = costs["c2_check_ms"] * s
        mv = world.K.get("mv")  # E04 codec-motion-vector accounting (None = the E03 pixel path, unchanged)
        self.skip_ret = bool(mv and mv.get("skip_retrieve"))
        self.mvx_w = mv["export_wall_ms"] * s if mv else 0.0  # MV export overhead of the decoder, every frame
        self.mvx_c = mv["export_cpu_ms"] * s if mv else 0.0
        # reference quantities for token reservations (nominal = no reuse)
        self.det_frame = float(np.mean(np.array(self.rc) + np.array(self.pc) + np.array(self.tc)))
        self.score1 = float(np.mean(self.sc[self.k_of_n[1]]))
        self.burst_cost = world.L * self.det_frame + BURST_WINDOWS_PER_RUN * self.score1

    def k(self, n: int) -> int:
        return self.k_of_n[n] if n < len(self.k_of_n) else self.k_top


def derive_constants(w: World) -> dict:
    """Constants derived from `test_dev` only (written to config.yaml by `lock` before any arm is simulated)."""
    tb = w.tb
    dev = set(w.dev_idx)
    clip_of_start = tb["clip_of_start"]
    dev_start = np.isin(clip_of_start, list(dev))
    row_dev = dev_start[tb["row_start"]]
    # window completion frame label (normal-frame windows): clip-local last frame of the window
    last = tb["start_frame"][tb["row_start"]].astype(np.int64) + w.L - 1 + tb["row_e"].astype(np.int64)
    lab = np.array([w.labels[int(c)][min(l, len(w.labels[int(c)]) - 1)] for c, l in zip(clip_of_start[tb["row_start"]], last)])
    normal = row_dev & (lab == 0)
    nu = tb["row_norm"][normal, 0].astype(np.float64)
    step = tb["row_step"][row_dev & (tb["row_e"] == 0)].astype(np.float64)
    step = step[np.isfinite(step)]
    dev_frames = np.concatenate([np.arange(w.offsets[i], w.offsets[i + 1]) for i in w.dev_idx])
    mot = w.motion[dev_frames]
    mot = mot[np.isfinite(mot)]
    bor = w.border[dev_frames]
    bor = bor[np.isfinite(bor)]
    qual = tb["row_qual"][row_dev]
    q = lambda x, p: float(np.quantile(x, p))  # noqa: E731
    return {
        "nu_safe": q(nu, 0.25), "nu_ref": q(nu, 0.5), "sigma_ref": max(float((q(nu, 0.75) - q(nu, 0.25)) / 1.349), 1e-3),
        "tau2": [q(step, p) for p in C2_QUANTILES], "tau2_quantiles": list(C2_QUANTILES),
        "motion_q": {"q50": q(mot, 0.5), "q70": q(mot, 0.7), "q85": q(mot, 0.85), "q90": q(mot, 0.9)},
        "border_q": {"q50": q(bor, 0.5), "q90": q(bor, 0.9)}, "qual_ref": q(qual, 0.5), "n_normal_dev_windows": int(normal.sum()),
        "c1": tb["meta"]["thr"],
    }


# =====================================================================================================================
# 5. Policies (decision at job start, from information available then) and the simulator
# =====================================================================================================================
INF = float("inf")


class Policy:
    """A scheduler. `decide` is called at the start of every frame job and returns (observe, score, c1_level, c2_tau).

    observe: run the detector pipeline on this frame (must continue the current run or start a new run on an even frame with room for L
    frames); score: deliver the windows completing now; c1_level 0..3 (0 = no C1); c2_tau None = no C2."""

    name = "policy"
    observer = False  # cheap motion observer + decode on every frame
    debit = "none"  # token bucket debit: none | actual | nominal

    def __init__(self, w: World, c: CostArrays):
        self.w, self.c, self.L = w, c, w.L

    def start_clip(self, n: int) -> None:
        pass

    def feedback(self, finish_ms: float, vmin: float, qual: float) -> None:
        pass

    def charge(self, actual: float, nominal: float) -> None:
        pass

    def decide(self, t: int, n: int, g: int, now: float, wait: float, rl: int, can_start: bool):
        raise NotImplementedError

    def info(self) -> dict:
        return {}


def burst_starts(period: float, n: int, L: int) -> list[int]:
    """Fixed-Burst starts on even clip-local frames (E02: multiples of W) with room for a full run."""
    out, k = [], 0
    while True:
        s = START_GRID * int(round(k * period / START_GRID))
        if s > n - L:
            break
        if not out or s > out[-1]:
            out.append(s)
        k += 1
    return out


class FixedPolicy(Policy):
    """F(W): W <= L -> detector duty 1, score every W frames; W > L -> an L-frame run every W frames (windows at run ends only)."""

    name = "fixed"

    def __init__(self, w, c, period: float, lvl: int = 0, tau2: float | None = None):
        super().__init__(w, c)
        self.period, self.lvl, self.tau2 = float(period), lvl, tau2

    def start_clip(self, n):
        self.n = n
        self.next_score = float(self.L - 1)
        self.starts = burst_starts(self.period, n, self.L) if self.period > self.L else []
        self.bi = 0
        self.cur = -1

    def decide(self, t, n, g, now, wait, rl, can_start):
        L = self.L
        if self.period < L + START_GRID:  # gaps shorter than one start epoch = a continuous run
            obs = rl > 0 or (t == 0 and can_start)
            score = False
            if t >= self.next_score:
                score = True
                self.next_score += self.period
            return obs, score and obs, self.lvl, self.tau2
        if rl == 0:
            while self.bi < len(self.starts) and self.starts[self.bi] < t:
                self.bi += 1
            if self.bi < len(self.starts) and t == self.starts[self.bi]:
                self.cur = t
                self.bi += 1
                obs = True
            else:
                obs = False
        else:
            obs = t < self.cur + L
        return obs, obs and t == self.cur + L - 1, self.lvl, self.tau2


def _clip01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


class Scheduler(Policy):
    """Token-bucket scheduler. kind 'motion': start on motion >= theta_m; kind 'adaptive': event/uncertainty priority u (module doc).

    cache: None | 'static' (fixed C1/C2 levels, Independent-Combo) | 'joint' (levels modulated by suspicion / backlog, savings reinvested)."""

    observer = True

    def __init__(self, w, c, kind: str, rate: float, w_ref: float, tau0: float = 1.4, extend: bool = True, theta_m: float | None = None,
                 cache: str | None = None, l1: int = 0, l2: int = 0, p_factor: float = FORCE_PERIOD_FACTOR):
        super().__init__(w, c)
        K = w.K
        self.kind, self.rate, self.tau0, self.extend, self.theta_m = kind, rate, tau0, extend, theta_m
        self.cap = BUCKET_CAP_BURSTS * c.burst_cost
        self.tokens = 0.5 * self.cap
        self.p_max = p_factor * max(w_ref, self.L)
        self.cache, self.l1, self.l2 = cache, l1, l2
        self.debit = "nominal" if cache == "static" else "actual"
        self.adaptive = kind == "adaptive"
        self.shed_on = self.adaptive
        self.ext_flag = extend
        self.m_med, self.b_med = K["motion_q"]["q50"], K["border_q"]["q50"]
        self.m_scale = max(K["motion_q"]["q90"] - K["motion_q"]["q50"], 1e-3)
        self.b_scale = max(K["border_q"]["q90"] - K["border_q"]["q50"], 1e-2)
        self.nu_ref, self.sigma_ref, self.q_ref = K["nu_ref"], K["sigma_ref"], K["qual_ref"]
        self.tau2_levels = K["tau2"]
        self.run_cap = RUN_CAP_FACTOR * self.L
        self.reset_stream()
        self.n_start = self.n_forced = self.n_event = 0

    def reset_stream(self):
        self.fb_norm, self.fb_qual, self.fb_t = None, None, -INF
        self.shed = False
        self.age = 10 ** 9
        self.bm, self.bb = self.m_med, self.b_med

    def start_clip(self, n):
        self.bm, self.bb = self.m_med, self.b_med
        self.age = 10 ** 9  # a new clip = new scene: the periodic full-scene check is due
        self.last_u = 0.0

    def feedback(self, finish_ms, vmin, qual):
        self.fb_norm, self.fb_qual, self.fb_t = vmin, qual, finish_ms

    def charge(self, actual, nominal):
        self.tokens -= actual if self.debit == "actual" else nominal

    def _cache_levels(self, suspect: bool, backlog: bool):
        if self.cache is None:
            return 0, None
        l1, l2 = self.l1, self.l2
        if self.cache == "joint":
            if suspect:
                l1, l2 = (1 if l1 else 0), (1 if l2 else 0)  # tighten
            elif backlog:
                l1, l2 = (3 if l1 else 0), (3 if l2 else 0)  # relax
        return l1, (self.tau2_levels[l2 - 1] if l2 else None)

    def decide(self, t, n, g, now, wait, rl, can_start):
        w = self.w
        L = self.L
        self.tokens = tokens = min(self.cap, self.tokens + self.rate)
        m, b = w.mot_l[g], w.bor_l[g]
        self.age += 1
        cut = t == 0
        um = ub = 0.0
        if not cut:
            um = _clip01((m - self.bm) / self.m_scale)
            ub = _clip01((b - self.bb) / self.b_scale)
            self.bm += 0.03 * (min(m, self.bm + self.m_scale) - self.bm)
            self.bb += 0.03 * (min(b, self.bb + self.b_scale) - self.bb)
        if self.shed_on:
            if wait > BACKLOG_SHED_ON_MS:
                self.shed = True
            elif wait < BACKLOG_SHED_OFF_MS:
                self.shed = False
        shed = self.shed
        ua = uq = 0.0
        if self.adaptive and self.fb_norm is not None:
            decay = math.exp(-(now - self.fb_t) / FB_DECAY_MS)
            ua = _clip01((self.nu_ref - self.fb_norm) / (2.0 * self.sigma_ref)) * decay
            uq = _clip01(4.0 * (self.q_ref - self.fb_qual) / self.q_ref) * decay
        suspect = ua >= 0.3 or ub >= 0.5 or uq >= 0.5 or cut
        lvl, tau2 = self._cache_levels(suspect, wait > BACKLOG_THIN_MS)
        fill = _clip01(tokens / self.cap)
        det = self.c.det_frame
        forced = self.age >= self.p_max
        if self.adaptive:
            u = um + ub + uq + 1.5 * ua + min(1.5, self.age / self.p_max)
            hot = max(um, ub, 1.5 * ua, uq)
            start_ok = forced or u >= self.tau0 * (1.0 - fill)
            ext_ok = self.extend and hot >= 0.5
        else:
            hot = m
            start_ok = forced or (m >= self.theta_m)
            ext_ok = self.extend and m >= self.theta_m
        if rl == 0:
            if not can_start or shed:
                return False, False, lvl, tau2
            reserve = 0.0 if forced else self.c.burst_cost * min(1.0, self.age / self.p_max)
            if tokens >= self.c.burst_cost + reserve and start_ok:
                self.n_start += 1
                self.n_forced += int(forced and not (hot >= 0.5))
                self.age = 0
                return True, True, lvl, tau2
            return False, False, lvl, tau2
        # inside a run
        if shed:
            return False, False, lvl, tau2
        if min(rl, self.age) < L:  # the current logical run (since the latest start) is younger than L frames
            cont = True
        else:
            cont = rl < self.run_cap and ext_ok and tokens >= 2.0 * det
            if not cont and can_start:  # the run would end here and a new one would start at once: it simply continues (counted as an extension)
                reserve = 0.0 if forced else self.c.burst_cost * min(1.0, self.age / self.p_max)
                if tokens >= self.c.burst_cost + reserve and start_ok:
                    cont = True
                    self.n_start += 1
                    self.age = 0
        score = cont
        if cont and self.adaptive and wait > BACKLOG_THIN_MS and (t & 1):
            score = False  # thin scoring under load: every 2nd frame
        return cont, score, lvl, tau2

    def info(self):
        return {"starts": self.n_start, "forced_checks": self.n_forced}


def simulate(w: World, c: CostArrays, pol: Policy, clips: list[int], light: bool = False, trace: dict | None = None, init_free: float = 0.0) -> dict:
    """Replay `pol` on the continuous stream formed by `clips` (queue carried over between clips, never reset).

    E05 additions (defaults reproduce E03 / E04 exactly): `init_free` = backlog (ms after the stream's first arrival) the worker starts with, to chain
    long streams segment by segment; `w.K["c1_retrack"]` = the tracker still runs (and is charged) on C1-reused frames (non-oracle identity).

    Returns per-frame service/cpu/wait/finish lists, one record per scored job (clip position, completion frame, stream index, delivered
    min-normality) and cache / schedule statistics. `trace` = {clip position: [], ...} receives per-frame decision tuples."""
    L, delta, E1 = w.L, w.delta, w.ext + 1
    dt = w.dt
    nu_safe = w.K["nu_safe"]
    ptr, rn, row_lin, row_step, row_qual = w.row_ptr, w.rn, w.row_lin, w.row_step, w.row_qual
    bits = w.bits
    gw, gc, rw, rc, pw, pc, tw, tc, sw, sc = c.gw, c.gc, c.rw, c.rc, c.pw, c.pc, c.tw, c.tc, c.sw, c.sc
    obs_ms, c1_ms, c2_ms = c.obs_ms, c.c1_ms, c.c2_ms
    mvx_w, mvx_c, skip_ret = c.mvx_w, c.mvx_c, c.skip_ret
    observer = pol.observer
    pend: deque = deque()
    fin_l, svc_l, cpu_l, wait_l, obs_l = [], [], [], [], []
    w_pos, w_t, w_gs, w_val = [], [], [], []
    S = dict(runs=0, run_frames=0, detect_frames=0, reuse_frames=0, forced_frames=0, c1_check_ms=0.0, c2_check_ms=0.0, saved_pose_ms=0.0,
             saved_score_ms=0.0, win_fresh=0, win_reuse=0, win_cand=0, c2_err_sum=0.0, c2_err_max=0.0, c2_age_sum=0, c2_age_max=0, c1_age_sum=0, shed_frames=0,
             decode_ms=0.0, pose_ms=0.0, track_ms=0.0, score_ms=0.0, obs_ms=0.0, window_frames=0, scored_frames=0, mv_export_ms=0.0, retrieve_saved_ms=0.0)
    free, gs = float(init_free), 0
    retrack = bool(w.K.get("c1_retrack"))
    for pos, ci in enumerate(clips):
        n, g0, sb = int(w.n_frames[ci]), int(w.offsets[ci]), int(w.start_base[ci])
        pol.start_clip(n)
        tr = trace.get(pos) if trace is not None else None
        a, chains, rl, c2 = -1, [], 0, {}
        for t in range(n):
            g = g0 + t
            arrival = gs * dt
            now = arrival if arrival > free else free
            wait = now - arrival
            while pend and pend[0][0] <= now:
                f, vmin, q = pend.popleft()
                pol.feedback(f, vmin, q)
            can_start = (t & 1) == 0 and t <= n - L
            obs, score, lvl, tau2 = pol.decide(t, n, g, now, wait, rl, can_start)
            if obs and rl == 0:
                if not can_start:
                    raise AssertionError("policy started a run where none is possible")
                a, chains, c2 = t, [], {}
                S["runs"] += 1
            cpu = gc[g]
            wall = gw[g]
            disc = 0.0
            if mvx_w:
                wall += mvx_w
                cpu += mvx_c
                S["mv_export_ms"] += mvx_c
            if observer:
                wall += obs_ms
                cpu += obs_ms
                S["obs_ms"] += obs_ms
                if not skip_ret:
                    wall += rw[g]
                    cpu += rc[g]
                    S["decode_ms"] += rc[g]
            n_fresh = n_reuse = n_cand = 0
            vmin, qsum, rr, forced = INF, 0.0, False, False
            chk = 0.0
            saved = 0.0
            if obs:
                rl += 1
                S["run_frames"] += 1
                if (t - a) % delta == 0 and t <= n - L:
                    chains.append((t, lvl, bits[lvl - 1][sb + (t >> 1)] if lvl else 0))
                forced = (t - a) % delta == 0
                if not observer and not skip_ret:
                    wall += rw[g]
                    cpu += rc[g]
                    S["decode_ms"] += rc[g]
                if not skip_ret:
                    disc += rc[g]
                if chains and chains[-1][1]:
                    wall += c1_ms
                    cpu += c1_ms
                    chk += c1_ms
                    S["c1_check_ms"] += c1_ms
                if t >= a + L - 1:
                    ch = chains[(t - a - L + 1) // delta]
                else:
                    ch = chains[0]
                if ch[1] and not forced and (ch[2] >> (t - ch[0])) & 1:
                    rr = True
                if skip_ret:  # the pixel is only needed when the detector (or a window) must run: BGR conversion on demand
                    if rr:
                        saved += rc[g]
                        S["retrieve_saved_ms"] += rc[g]
                    else:
                        wall += rw[g]
                        cpu += rc[g]
                        disc += rc[g]
                        S["decode_ms"] += rc[g]
                if rr:
                    S["reuse_frames"] += 1
                    if retrack:  # non-oracle C1: only the pose network is skipped, the tracker consumes the reused detections
                        wall += tw[g]
                        cpu += tc[g]
                        disc += tc[g]
                        S["track_ms"] += tc[g]
                        saved += pc[g]
                        S["saved_pose_ms"] += pc[g]
                    else:
                        saved += pc[g] + tc[g]
                        S["saved_pose_ms"] += pc[g] + tc[g]
                    age = 1
                    while (ch[2] >> (t - ch[0] - age)) & 1:
                        age += 1
                    S["c1_age_sum"] += age
                else:
                    wall += pw[g] + tw[g]
                    cpu += pc[g] + tc[g]
                    disc += pc[g] + tc[g]
                    S["detect_frames"] += 1
                    S["pose_ms"] += pc[g]
                    S["track_ms"] += tc[g]
                    if forced:
                        S["forced_frames"] += 1
                if t >= a + L - 1:
                    S["window_frames"] += 1
                    j = (t - a - L + 1) // delta
                    cj, lj, _ = chains[j]
                    k = (sb + (cj >> 1)) * E1 + (t - cj - (L - 1))
                    lo, hi = ptr[k], ptr[k + 1]
                    if hi > lo:
                        nl = rn[lj]
                        if tau2 is None:
                            if score:
                                n_fresh = hi - lo
                                vmin = min(nl[lo:hi])
                                qsum = sum(row_qual[lo:hi]) / n_fresh
                        else:
                            for r in range(lo, hi):
                                lin = row_lin[r]
                                st = row_step[r]
                                ent = c2.get(lin)
                                if st != st:  # no predecessor: drift unknown
                                    ent = None
                                    c2.pop(lin, None)
                                elif ent is not None:
                                    ent[1] += st
                                if not score:
                                    continue
                                n_cand += 1
                                nv = nl[r]
                                if ent is not None and ent[1] <= tau2 and t - ent[2] <= A2 and ent[0] >= nu_safe:
                                    v = ent[0]
                                    n_reuse += 1
                                    e_abs = abs(nv - v)
                                    S["c2_err_sum"] += e_abs
                                    if e_abs > S["c2_err_max"]:
                                        S["c2_err_max"] = e_abs
                                    S["c2_age_sum"] += t - ent[2]
                                    if t - ent[2] > S["c2_age_max"]:
                                        S["c2_age_max"] = t - ent[2]
                                else:
                                    v = nv
                                    n_fresh += 1
                                    c2[lin] = [nv, 0.0, t]
                                if v < vmin:
                                    vmin = v
                                qsum += row_qual[r]
                            if n_cand:
                                qsum /= n_cand
                                x = c2_ms * (hi - lo)
                                wall += x
                                cpu += x
                                chk += x
                                S["c2_check_ms"] += x
                    if n_fresh:
                        kk = c.k(n_fresh)
                        wall += sw[kk][g]
                        cpu += sc[kk][g]
                        disc += sc[kk][g]
                        S["score_ms"] += sc[kk][g]
                    if n_reuse:
                        kt = c.k(n_fresh + n_reuse)
                        sv = sc[kt][g] - (sc[c.k(n_fresh)][g] if n_fresh else 0.0)
                        saved += sv
                        S["saved_score_ms"] += sv
                    S["win_fresh"] += n_fresh
                    S["win_reuse"] += n_reuse
                    S["win_cand"] += n_cand
                    if vmin < INF:
                        S["scored_frames"] += 1
                disc += chk
            else:
                rl = 0
            if wait > BACKLOG_SHED_ON_MS:
                S["shed_frames"] += 1
            finish = now + wall
            free = finish
            pol.charge(disc, disc + saved - chk)
            if vmin < INF:
                pend.append((finish, vmin, qsum))
                if not light:
                    w_pos.append(pos)
                    w_t.append(t)
                    w_gs.append(gs)
                    w_val.append(vmin)
            fin_l.append(finish)
            obs_l.append(1 if obs else 0)
            svc_l.append(wall)
            cpu_l.append(cpu)
            wait_l.append(wait)
            if tr is not None:
                tr.append((t, int(obs), lvl, int(rr), int(forced), n_fresh, n_reuse, round(wait, 2), round(getattr(pol, "tokens", 0.0), 2)))
            gs += 1
    S["n_frames"] = gs
    return {"fin": np.array(fin_l), "obs": np.array(obs_l, dtype=np.uint8), "svc": np.array(svc_l), "cpu": np.array(cpu_l), "wait": np.array(wait_l), "w_pos": np.array(w_pos, dtype=np.int64),
            "w_t": np.array(w_t, dtype=np.int64), "w_gs": np.array(w_gs, dtype=np.int64), "w_val": np.array(w_val), "stats": S, "clips": list(clips),
            "policy": pol.info()}


# =====================================================================================================================
# 6. Delivery, metrics
# =====================================================================================================================
def delivery(w: World, R: dict, hold: float, ideal: bool = False, cap_ms: float = CAP_MS):
    """Per-clip delivered (or ideal: zero delay, completion landing) anomaly series of one simulation. -> series, stats, info."""
    clips = R["clips"]
    nfl = w.n_frames[clips]
    soff = np.concatenate([[0], np.cumsum(nfl)])
    pos, t, gs, val = R["w_pos"], R["w_t"], R["w_gs"], R["w_val"]
    if ideal or len(pos) == 0:
        landing, ok = t.copy(), np.ones(len(pos), dtype=bool)
        late = beyond = np.zeros(len(pos), dtype=bool)
    else:
        fin = R["fin"][gs]
        latency = fin - gs * w.dt
        landing = np.maximum(e02.delivery_frame(fin, w.dt) - soff[pos], t)
        late = latency > cap_ms
        beyond = landing >= nfl[pos]
        ok = ~(late | beyond)
    raw = e00.per_frame_min(pos[ok], landing[ok], val[ok], list(nfl))
    series, stats = e00.deliver(raw, w.sentinel, hold)
    stats = dict(stats, dropped_late=int(late.sum()), dropped_end=int(beyond.sum()), windows=int(len(pos)))
    return series, stats, {"landing": landing, "ok": ok, "raw": raw}


def label_events(label: np.ndarray) -> list[tuple[int, int]]:
    """Inclusive (start, end) frame runs of label == 1."""
    x = np.concatenate([[0], np.asarray(label).astype(int), [0]])
    d = np.diff(x)
    return list(zip(np.nonzero(d == 1)[0].tolist(), (np.nonzero(d == -1)[0] - 1).tolist()))


def alert_runs(above: np.ndarray, min_len: int = EVENT_MIN_FRAMES) -> list[tuple[int, int]]:
    """Inclusive runs of `above` of at least `min_len` frames. An alert is *raised* at start + min_len - 1."""
    return [(a, b) for a, b in label_events(above) if b - a + 1 >= min_len]


def event_metrics(label: np.ndarray, series: np.ndarray, theta: float, fps: float, min_len: int = EVENT_MIN_FRAMES, tol_s: float = 1.0) -> dict:
    """Event recall @0.5 s / 1.0 s, delays (ms) and false alerts of one clip. A detection of event (s, e) is an alert run overlapping it whose
    raise frame is <= e; its delay is max(raise, s) - s. A false alert is a run that does not overlap any event (extended by tol_s)."""
    events = label_events(label)
    runs = alert_runs(np.asarray(series) > theta, min_len)
    tol = int(round(tol_s * fps))
    delays = []
    for s, e in events:
        best = None
        for r0, r1 in runs:
            f = r0 + min_len - 1
            if r1 >= s and f <= e:
                d = max(f, s) - s
                best = d if best is None else min(best, d)
        delays.append(best)
    false_alerts = sum(1 for r0, r1 in runs if not any(r1 >= s and r0 <= e + tol for s, e in events))
    det = [d for d in delays if d is not None]
    return {"events": len(events), "hit_0.5": sum(1 for d in det if d <= 0.5 * fps), "hit_1.0": sum(1 for d in det if d <= 1.0 * fps), "detected": len(det),
            "delays_ms": [1000.0 * d / fps for d in det], "false_alerts": false_alerts, "normal_frames": int((np.asarray(label) == 0).sum())}


def clip_costs(w: World, R: dict) -> dict:
    """Per-clip CPU (ms per video-second), wait statistics of a simulation."""
    clips = R["clips"]
    nfl = w.n_frames[clips]
    off = np.concatenate([[0], np.cumsum(nfl)])
    cpu, wait = R["cpu"], R["wait"]
    sec = nfl * w.dt / 1000.0
    return {"cpu_ms_per_s": np.add.reduceat(cpu, off[:-1]) / sec, "wait_mean_ms": np.add.reduceat(wait, off[:-1]) / nfl,
            "wait_max_ms": np.maximum.reduceat(wait, off[:-1]),
            "service_ms": np.add.reduceat(R["svc"], off[:-1])}


def total_cost(w: World, R: dict) -> float:
    """CPU ms per video-second of the whole stream."""
    return float(R["cpu"].sum() / (len(R["cpu"]) * w.dt / 1000.0))


def peak_memory_mb(w: World, R: dict) -> float:
    """E02 end-to-end RSS (T4, h264) + frames queued behind the worker at the worst moment (decoded BGR, 3 bytes / pixel)."""
    base = E02_RSS_MB[w.ds]
    queued = (float(R["wait"].max()) / w.dt + 1.0) if len(R["wait"]) else 1.0
    return base + queued * w.size[0] * w.size[1] * 3 / 2 ** 20


E02_RSS_MB = {"ShanghaiTech": 896.40234375, "UBnormal": 922.1328125}


# =====================================================================================================================
# 7. Cost matching (label-free bisection of one rate knob)
# =====================================================================================================================
def calibrate(cost_of, lo: float, hi: float, target: float, increasing: bool, iters: int = 12, tol: float = 0.004) -> dict:
    """Largest-spend knob with cost <= target. `cost_of(knob)` -> realised CPU ms/s (labels never read). Geometric bisection on [lo, hi].

    increasing: cost grows with the knob (token rate); otherwise it falls (period W). Returns knob, cost, status ('ok', 'under' = cannot spend
    the budget even at the maximum, 'over' = cannot get below the budget even at the minimum), evaluations."""
    ev = []

    def f(x):
        y = cost_of(x)
        ev.append((x, y))
        return y

    cmax_knob, cmin_knob = (hi, lo) if increasing else (lo, hi)  # knob of the largest / smallest spend
    c_top = f(cmax_knob)
    if c_top <= target:
        return {"knob": cmax_knob, "cost": c_top, "status": "ok" if c_top >= 0.97 * target else "under", "evals": len(ev)}
    c_bot = f(cmin_knob)
    if c_bot > target:
        return {"knob": cmin_knob, "cost": c_bot, "status": "over", "evals": len(ev)}
    good_k, good_c = cmin_knob, c_bot  # feasible (cost <= target)
    bad_k = cmax_knob
    for _ in range(iters):
        mid = math.sqrt(good_k * bad_k)
        cm = f(mid)
        if cm <= target:
            good_k, good_c = mid, cm
            if cm >= (1.0 - tol) * target:
                break
        else:
            bad_k = mid
    return {"knob": good_k, "cost": good_c, "status": "ok" if good_c >= 0.97 * target else "under", "evals": len(ev)}


def cell_target(c_full: dict, s: float, beta: float) -> float:
    return beta * c_full[s]


# =====================================================================================================================
# 8. Arm configurations, policy factory, cell runner, tuning
# =====================================================================================================================
THETA_GRID = ("q50", "q70", "q85")
TAU0_GRID = (0.8, 1.4, 2.0)
P_GRID = (1, 2)
EXT_GRID = (False, True)
KNOB_RANGE = {"period": (1.0, 400.0), "rate": (1.0, 1000.0)}
TIE_PT = 0.10  # selection ties (mean dev AUC within 0.10 pt) -> the simpler value (earlier in the grid)


def make_policy(w: World, c: CostArrays, cfg: dict, knob: float, w_ref: float) -> Policy:
    arm = cfg["arm"]
    lev2 = (lambda l: w.K["tau2"][l - 1] if l else None)  # noqa: E731
    if arm in ("Fixed", "Cache-C1", "Cache-C2", "Cache-Only"):
        return FixedPolicy(w, c, knob, cfg.get("l1", 0), lev2(cfg.get("l2", 0)))
    if arm == "Motion-Only":
        return Scheduler(w, c, "motion", knob, w_ref, extend=cfg["extend"], theta_m=w.K["motion_q"][cfg["theta"]], p_factor=cfg["p"])
    if arm == "Adaptive-Only":
        return Scheduler(w, c, "adaptive", knob, w_ref, tau0=cfg["tau0"], extend=cfg["extend"], p_factor=cfg["p"])
    if arm == "Independent-Combo":
        return Scheduler(w, c, "adaptive", knob, w_ref, tau0=cfg["tau0"], extend=cfg["extend"], p_factor=cfg["p"], cache="static", l1=cfg["l1"], l2=cfg["l2"])
    if arm == "Joint-Controller":
        return Scheduler(w, c, "adaptive", knob, w_ref, tau0=cfg["tau0"], extend=cfg["extend"], p_factor=cfg["p"], cache="joint", l1=cfg["l1"], l2=cfg["l2"])
    raise KeyError(arm)


def knob_kind(cfg: dict) -> str:
    return "period" if cfg["arm"] in ("Fixed", "Cache-C1", "Cache-C2", "Cache-Only") else "rate"


def cfg_label(cfg: dict) -> str:
    return "|".join(f"{k}={v}" for k, v in cfg.items() if k != "arm")


_full_cost: dict = {}


def c_full(w: World, s: float, clips: list[int], tag: str) -> float:
    """CPU ms/s of the K=1 (continuous detector, every window scored) pipeline on this stream: the budget unit (beta = 1)."""
    key = (w.ds, s, tag)
    if key not in _full_cost:
        c = w.cost(s)
        _full_cost[key] = total_cost(w, simulate(w, c, FixedPolicy(w, c, 1.0), clips, light=True))
    return _full_cost[key]


def run_cell(w: World, cfg: dict, s: float, beta: float, clips: list[int], tag: str, w_ref: float | None, full: bool = False,
             trace: dict | None = None, knob: float | None = None) -> dict:
    """Calibrate the rate knob of `cfg` to the cell's cost target on `clips` (labels never read), then (full) simulate once more with
    recording. w_ref: the cell's Fixed period (hold, periodic check); None for the Fixed arm itself (it defines it)."""
    c = w.cost(s)
    target = beta * c_full(w, s, clips, tag)
    kind = knob_kind(cfg)
    lo, hi = KNOB_RANGE[kind]
    mk = (lambda x: make_policy(w, c, cfg, x, x if w_ref is None else w_ref))  # noqa: E731
    if knob is None:
        cal = calibrate(lambda x: total_cost(w, simulate(w, c, mk(x), clips, light=True)), lo, hi, target, increasing=(kind == "rate"))
    else:
        cal = {"knob": knob, "cost": None, "status": "given", "evals": 0}
    out = {"cfg": cfg, "s": s, "beta": beta, "target": target, "knob": cal["knob"], "status": cal["status"], "evals": cal["evals"]}
    if cfg["arm"] == "Fixed":
        w_ref = cal["knob"]
    out["w_ref"] = w_ref
    out["hold"] = max(0, int(round(w_ref)) - 1)
    pol = make_policy(w, c, cfg, cal["knob"], w_ref)
    R = simulate(w, c, pol, clips, light=not full, trace=trace)
    out["cost"] = total_cost(w, R)
    out["ratio"] = out["cost"] / target
    out["R"] = R
    return out


def dev_auc_of(w: World, R: dict, hold: int, clips: list[int]) -> float:
    ser, _, _ = delivery(w, R, hold)
    return 100 * e00.pooled_auc([w.labels[i] for i in clips], ser)


def tune_job(job: tuple) -> dict:
    """One (config, tuning cell) on the dev stream -> dev delivered AUC (selection statistic) and its cost ratio."""
    w, devc = _G["w"], _G["dev"]
    cfg, (s, beta), w_ref = job
    r = run_cell(w, cfg, s, beta, devc, "dev", w_ref, full=True)
    return {"cfg": cfg, "cell": (s, beta), "auc": dev_auc_of(w, r["R"], r["hold"], devc), "ratio": r["ratio"], "status": r["status"], "knob": r["knob"],
            "w_ref": r["w_ref"]}


def tuning_configs(w: World, base: dict) -> dict[str, list[dict]]:
    """Grids of the plan (section 6); `base` carries already selected values for the sequential stages."""
    g: dict[str, list[dict]] = {}
    g["Motion-Only"] = [{"arm": "Motion-Only", "theta": th, "extend": ex, "p": p} for th in THETA_GRID for ex in EXT_GRID for p in P_GRID]
    g["Adaptive-Only"] = [{"arm": "Adaptive-Only", "tau0": t0, "extend": ex, "p": p} for t0 in TAU0_GRID for ex in EXT_GRID for p in P_GRID]
    g["Cache-C1"] = [{"arm": "Cache-C1", "l1": l1, "l2": 0} for l1 in LEVELS]
    g["Cache-C2"] = [{"arm": "Cache-C2", "l1": 0, "l2": l2} for l2 in LEVELS]
    return g


def select_best(rows: list[dict]) -> tuple[dict, list[dict]]:
    """Mean dev AUC over the tuning cells per config; ties within TIE_PT -> the earlier (simpler) config. -> best cfg, table."""
    by: dict[str, dict] = {}
    for r in rows:
        key = cfg_label(r["cfg"])
        d = by.setdefault(key, {"cfg": r["cfg"], "aucs": [], "ratios": [], "status": []})
        d["aucs"].append(r["auc"])
        d["ratios"].append(r["ratio"])
        d["status"].append(r["status"])
    table = [{"cfg": d["cfg"], "mean_auc": float(np.mean(d["aucs"])), "aucs": d["aucs"], "ratios": d["ratios"], "status": d["status"]} for d in by.values()]
    top = max(t["mean_auc"] for t in table)
    best = next(t for t in table if t["mean_auc"] >= top - TIE_PT)  # grid order = simplicity order
    return best["cfg"], table


def tune_dataset(w: World, procs: int, log=print) -> dict:
    """Sequential, dev-stream-only tuning. Returns {'selected': {arm: cfg}, 'tables': {...}, 'tuning_cells': [...], 'w_ref': {cell: W}}."""
    devc = list(w.dev_idx)
    cells = TUNING_CELLS
    _G.update(w=w, dev=devc)
    sel: dict[str, dict] = {}
    tables: dict[str, list] = {}
    try:
        with mp.get_context("fork").Pool(procs) as pool:
            fixed = pool.map(tune_job, [({"arm": "Fixed"}, cell, None) for cell in cells])
            w_ref = {cell: r["w_ref"] for cell, r in zip(cells, fixed)}
            log(f"[{now()}] tuning {w.ds}: dev Fixed periods {{cell: W}} = { {str(k): round(v, 1) for k, v in w_ref.items()} }")

            def stage(name: str, cfgs: list[dict]):
                rows = pool.map(tune_job, [(cfg, cell, w_ref[cell]) for cfg in cfgs for cell in cells])
                best, table = select_best(rows)
                sel[name] = best
                tables[name] = table
                log(f"[{now()}] tuning {w.ds} {name}: best {cfg_label(best)} mean dev AUC {max(t['mean_auc'] for t in table):.2f} "
                    f"(grid {len(table)}; fixed {np.mean([r['auc'] for r in fixed]):.2f})")
                return best

            grids = tuning_configs(w, {})
            stage("Motion-Only", grids["Motion-Only"])
            adaptive = stage("Adaptive-Only", grids["Adaptive-Only"])
            c1 = stage("Cache-C1", grids["Cache-C1"])
            c2 = stage("Cache-C2", grids["Cache-C2"])
            both = {"arm": "Cache-Only", "l1": c1["l1"], "l2": c2["l2"]}
            sel["Cache-Only"] = both
            sel["Independent-Combo"] = {"arm": "Independent-Combo", "tau0": adaptive["tau0"], "extend": adaptive["extend"], "p": adaptive["p"],
                                        "l1": c1["l1"], "l2": c2["l2"]}
            stage("Joint-Controller", [{"arm": "Joint-Controller", "tau0": t0, "extend": adaptive["extend"], "p": adaptive["p"], "l1": c1["l1"],
                                        "l2": c2["l2"]} for t0 in TAU0_GRID])
            rows = pool.map(tune_job, [(both, cell, w_ref[cell]) for cell in cells])
            tables["Cache-Only"] = [{"cfg": both, "mean_auc": float(np.mean([r["auc"] for r in rows])), "aucs": [r["auc"] for r in rows],
                                     "ratios": [r["ratio"] for r in rows], "status": [r["status"] for r in rows]}]
            rows = pool.map(tune_job, [(sel["Independent-Combo"], cell, w_ref[cell]) for cell in cells])
            tables["Independent-Combo"] = [{"cfg": sel["Independent-Combo"], "mean_auc": float(np.mean([r["auc"] for r in rows])),
                                            "aucs": [r["auc"] for r in rows], "ratios": [r["ratio"] for r in rows], "status": [r["status"] for r in rows]}]
            tables["Fixed"] = [{"cfg": {"arm": "Fixed"}, "mean_auc": float(np.mean([r["auc"] for r in fixed])), "aucs": [r["auc"] for r in fixed],
                                "ratios": [r["ratio"] for r in fixed], "status": [r["status"] for r in fixed]}]
    finally:
        _G.clear()
    sel["Fixed"] = {"arm": "Fixed"}
    return {"selected": sel, "tables": tables, "tuning_cells": [list(c_) for c_ in cells], "w_ref_dev": {str(k): v for k, v in w_ref.items()}}


# =====================================================================================================================
# 9. Lock: constants + dev tuning -> config.yaml (before any final run)
# =====================================================================================================================
def sha(path: Path) -> str:
    return e00.sha256_file(path)


def lock_dataset(ctx, ds: str, procs: int, log=print) -> dict:
    w = World(ctx, ds)
    w.K = derive_constants(w)
    names = w.names
    bench = bench_observer(ds, names)
    costs = {"obs_ms": max(OBS_MS[ds], bench["observer_ms_median"]), "c1_check_ms": max(C1_CHECK_MS[ds], bench["thumb_ms_median"] + bench["roi_check_ms_median"]),
             "c2_check_ms": C2_CHECK_MS, "measured": bench}
    w.K["costs"] = costs
    w._cost.clear()
    # alert threshold: 5 % false-positive rate of the K=1 ideal series on test_dev normal frames
    c = w.cost(2.0)
    devc = list(w.dev_idx)
    R = simulate(w, c, FixedPolicy(w, c, 1.0), devc)
    ser, _, _ = delivery(w, R, 0, ideal=True)
    normal = np.concatenate([ser[i][w.labels[devc[i]] == 0] for i in range(len(devc))])
    w.K["theta"] = float(np.quantile(normal, 1.0 - ALERT_FPR))
    log(f"[{now()}] lock {ds}: constants {json.dumps(plain({k: v for k, v in w.K.items() if k != 'costs'}))[:300]} ... costs {costs['obs_ms']:.3f}/{costs['c1_check_ms']:.3f} ms")
    tuned = tune_dataset(w, procs, log)
    cf = {}
    allc = list(range(len(names)))
    for s in sorted({c_[0] for c_ in CELLS}):
        cf[str(s)] = {"all": c_full(w, s, allc, "all"), "dev": c_full(w, s, devc, "dev")}
    return {"constants": plain(w.K), "tuning": plain(tuned), "c_full": cf, "n_clips": len(names), "L": w.L, "delta": w.delta,
            "frames": int(w.total), "dev_frames": int(sum(w.n_frames[i] for i in devc))}


def stage_lock(procs: int, log=print) -> dict:
    ctx = e01.build_ctx(procs, log)
    e02_sum = json.loads((e02.EXP_DIR / "summary.json").read_text())
    cfg = {"experiment": "Q2-P1-E03", "title": "Core ablation", "seed": SEED, "pool_seed": POOL_SEED, "bootstrap": {"resamples": N_BOOT, "seed": 0},
           "builds_on": {n: {"summary_sha256": sha(d / "summary.json"), "config_sha256": sha(d / "config.yaml")}
                         for n, d in (("E00", e00.EXP_DIR), ("E01", e01.EXP_DIR), ("E02", e02.EXP_DIR))},
           "tracker": TRACKER.label, "frontends": PRIMARY, "cells": [list(x) for x in CELLS], "tuning_cells": [list(x) for x in TUNING_CELLS], "arms": list(ARMS),
           "grids": {"theta_m": list(THETA_GRID), "tau0": list(TAU0_GRID), "extend": list(EXT_GRID), "p_factor": list(P_GRID), "c1_levels": list(LEVELS),
                     "c2_levels": list(LEVELS), "tie_pt": TIE_PT},
           "fixed_constants": {"A1": A1, "A2": A2, "grid": [GRID_H, GRID_W], "c1_quantiles": list(C1_QUANTILES), "c2_quantiles": list(C2_QUANTILES),
                               "bucket_cap_bursts": BUCKET_CAP_BURSTS, "shed_on_ms": BACKLOG_SHED_ON_MS, "shed_off_ms": BACKLOG_SHED_OFF_MS,
                               "thin_ms": BACKLOG_THIN_MS, "run_cap_L": RUN_CAP_FACTOR, "force_period_factor": FORCE_PERIOD_FACTOR,
                               "event_min_frames": EVENT_MIN_FRAMES, "alert_fpr": ALERT_FPR, "late_cap_ms": CAP_MS, "chain_delta": DELTA, "start_grid": START_GRID,
                               "fb_decay_ms": FB_DECAY_MS, "burst_windows_per_run": BURST_WINDOWS_PER_RUN},
           "e02_reference_cost": {k: v for k, v in {ds: {f"T4_s{s:.1f}_b1.00": e02_sum["headline"][f"{ds}/{PRIMARY[ds]}/locked"].get("stride1", {}).get(f"T4_s{s:.1f}_b1.00", {})
                                                          .get("cpu_ms_per_s") for s in (2.0, 2.5, 3.0)} for ds in DATASETS}.items()},
           "datasets": {}}
    for ds in DATASETS:
        cfg["datasets"][ds] = lock_dataset(ctx, ds, procs, log)
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    (EXP_DIR / "config.yaml").write_text(yaml.safe_dump(plain(cfg), sort_keys=False))
    log(f"[{now()}] lock: wrote {EXP_DIR / 'config.yaml'}")
    return cfg


def load_config() -> dict:
    return yaml.safe_load((EXP_DIR / "config.yaml").read_text())


# =====================================================================================================================
# 10. Final run: every arm x cell on the continuous all-clips stream (calibrated on `all`, evaluated on dev / confirmation / all)
# =====================================================================================================================
def run_path(ds: str) -> Path:
    return WORK_DIR / f"run_{ds}.pkl"


def compact_result(w: World, r: dict) -> dict:
    """Everything the analysis needs from one (arm, cell) simulation; the heavy simulation record is dropped."""
    R, hold = r["R"], r["hold"]
    ser, dstats, info = delivery(w, R, hold)
    ideal, istats, _ = delivery(w, R, hold, ideal=True)
    cc = clip_costs(w, R)
    n = len(R["fin"])
    qm = e02.queue_metrics(np.arange(n) * w.dt, R["svc"], R["fin"], w.dt)
    nfl = w.n_frames[R["clips"]]
    ok = info["ok"]
    lat = R["fin"][R["w_gs"]] - R["w_gs"] * w.dt
    cnt = np.bincount(R["w_pos"][ok], minlength=len(nfl))
    emit = np.where(cnt > 0, np.bincount(R["w_pos"][ok], weights=lat[ok], minlength=len(nfl)) / np.maximum(cnt, 1), np.nan)
    land_delay = np.where(cnt > 0, np.bincount(R["w_pos"][ok], weights=(info["landing"] - R["w_t"])[ok], minlength=len(nfl)) / np.maximum(cnt, 1), np.nan)
    out = {k: r[k] for k in ("cfg", "s", "beta", "target", "knob", "status", "evals", "w_ref", "hold", "cost", "ratio")}
    out.update({"series": np.concatenate(ser).astype(np.float32), "ideal": np.concatenate(ideal).astype(np.float32), "dstats": plain(dstats), "istats": plain(istats),
                "stats": plain(R["stats"]), "policy": plain(R["policy"]), "queue": plain(qm), "mem_mb": peak_memory_mb(w, R),
                "clip_cpu_ms_per_s": cc["cpu_ms_per_s"], "clip_wait_mean_ms": cc["wait_mean_ms"], "clip_wait_max_ms": cc["wait_max_ms"], "clip_emit_ms": emit,
                "clip_landing_delay_frames": land_delay, "clip_windows": cnt, "observed": R["obs"]})
    return out


def eval_job(job: tuple) -> dict:
    w, allc = _G["w"], _G["all"]
    cfg, (s, beta), w_ref = job
    return compact_result(w, run_cell(w, cfg, s, beta, allc, "all", w_ref, full=True))


def stage_run(ctx, ds: str, procs: int, log=print, force: bool = False) -> dict:
    path = run_path(ds)
    cfg = load_config()
    stamp = sha(EXP_DIR / "config.yaml")
    if path.exists() and not force:
        got = pickle.loads(path.read_bytes())
        if got["stamp"] == stamp:
            log(f"[{now()}] run {ds}: cached")
            return got
    dc = cfg["datasets"][ds]
    w = World(ctx, ds)
    w.K = dc["constants"]
    sel = dc["tuning"]["selected"]
    allc = list(range(len(w.names)))
    cells = [tuple(c_) for c_ in cfg["cells"]]
    t0 = time.perf_counter()
    _G.update(w=w, all=allc)
    try:
        with mp.get_context("fork").Pool(procs) as pool:
            fixed = pool.map(eval_job, [(sel["Fixed"], cell, None) for cell in cells], chunksize=1)
            w_ref = {cell: r["w_ref"] for cell, r in zip(cells, fixed)}
            log(f"[{now()}] run {ds}: Fixed periods " + ", ".join(f"{cell_id(*c_)}:{w_ref[c_]:.1f}" for c_ in cells))
            others = pool.map(eval_job, [(sel[arm], cell, w_ref[cell]) for arm in ARMS if arm != "Fixed" for cell in cells], chunksize=1)
    finally:
        _G.clear()
    results = {}
    for r in fixed + others:
        results[(r["cfg"]["arm"], cell_id(r["s"], r["beta"]))] = r
    log(f"[{now()}] run {ds}: {len(results)} (arm, cell) simulations in {time.perf_counter() - t0:.0f}s")
    # sample decision traces (3 clips replayed on their own stream at the mid cell) for decisions.jsonl
    traces = {}
    mid = (2.0, 0.50)
    pick = [w.conf_idx[0], w.conf_idx[len(w.conf_idx) // 2], w.dev_idx[0]]
    for arm in ("Fixed", "Adaptive-Only", "Cache-Only", "Joint-Controller"):
        r = results[(arm, cell_id(*mid))]
        tr = {0: [], 1: [], 2: []}
        run_cell(w, sel[arm], mid[0], mid[1], pick, "trace", r["w_ref"] if arm != "Fixed" else None, full=True, trace=tr, knob=r["knob"])
        traces[arm] = {"clips": [w.names[i] for i in pick], "cell": cell_id(*mid), "frames": {k: v[:400] for k, v in tr.items()},
                       "columns": ["t", "observe", "c1_level", "c1_reused", "forced_detect", "windows_fresh", "windows_reused", "queue_wait_ms", "tokens"]}
    out = {"stamp": stamp, "results": results, "w_ref": {cell_id(*c_): w_ref[c_] for c_ in cells}, "traces": traces,
           "n_frames": [int(x) for x in w.n_frames], "build_s": time.perf_counter() - t0}
    path.write_bytes(pickle.dumps(out))
    return out


# =====================================================================================================================
# 11. Analysis: bootstrap, contrasts, verdicts (pure functions first)
# =====================================================================================================================
class AucBoot:
    """Exact pooled-AUC over a video subset and its paired video-bootstrap distribution (same resamples for every series).

    Series are rank-compressed once; per-video (positive, negative) histograms over the unique values are combined with the multinomial
    resample weights, so a resample's pooled AUC (ties = 1/2) costs one sparse-dense product instead of a sort."""

    def __init__(self, labels: list, idx: list[int] | None, n_boot: int = N_BOOT, seed: int = 0):
        self.idx = list(range(len(labels))) if idx is None else list(idx)
        n = len(self.idx)
        self.n = n
        self.y = [np.asarray(labels[i]).astype(np.int8) for i in self.idx]
        self.lens = np.array([len(y) for y in self.y])
        self.W = np.random.default_rng(seed).multinomial(n, np.full(n, 1.0 / n), size=n_boot).astype(np.float64)

    def aucs(self, scores: list[np.ndarray]) -> tuple[float, np.ndarray]:
        """-> (point AUC in %, bootstrap AUCs in % (n_boot,)). `scores` aligned with the subset (one array per video)."""
        from scipy.sparse import coo_matrix

        s = np.concatenate([np.asarray(scores[k], dtype=np.float64) for k in range(self.n)])
        y = np.concatenate(self.y).astype(np.float64)
        u, inv = np.unique(s, return_inverse=True)
        vid = np.repeat(np.arange(self.n), self.lens)
        P = coo_matrix((y, (inv, vid)), shape=(len(u), self.n)).tocsr()
        N = coo_matrix((1.0 - y, (inv, vid)), shape=(len(u), self.n)).tocsr()

        def auc_of(Wm: np.ndarray) -> np.ndarray:
            out = np.empty(len(Wm))
            for a in range(0, len(Wm), 100):
                Wc = Wm[a:a + 100].T
                Pb, Nb = P @ Wc, N @ Wc
                cum = np.cumsum(Nb, axis=0) - Nb
                den = Pb.sum(0) * Nb.sum(0)
                out[a:a + 100] = np.where(den > 0, (Pb * (cum + 0.5 * Nb)).sum(0) / np.where(den > 0, den, 1.0), np.nan)
            return out

        return 100 * float(auc_of(np.ones((1, self.n)))[0]), 100 * auc_of(self.W)


def ci_of(vec: np.ndarray) -> list[float]:
    v = vec[np.isfinite(vec)]
    return [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))]


def contrast(a: np.ndarray, b: np.ndarray) -> dict:
    """Paired bootstrap contrast of two (n_cells, n_boot) AUC arrays or two (n_boot,) vectors, averaged over cells."""
    da = (np.asarray(a) - np.asarray(b))
    if da.ndim == 2:
        da = da.mean(0)
    lo, hi = ci_of(da)
    return {"delta": float(np.nanmean(da)), "lo": lo, "hi": hi, "excludes_zero": bool(lo > 0 or hi < 0)}


def verdict_h1(c_af: dict, c_am: dict) -> str:
    """H1: Adaptive - Fixed and Adaptive - Motion (12-cell averages)."""
    if c_af["hi"] < 0 or c_am["hi"] < 0:
        return "rejected"
    n_pos = int(c_af["lo"] > 0) + int(c_am["lo"] > 0)
    return "supported" if n_pos == 2 else ("partially supported" if n_pos == 1 else "not supported")


def verdict_h2(net_saving: float, c_cf: dict, recall_delta: float) -> str:
    """H2: Cache-Only vs Fixed: net CPU saving > 5 %, CI lower > 0, recall @1.0 s not worse by more than 2 points."""
    if net_saving <= 0 or c_cf["hi"] < 0:
        return "rejected"
    if net_saving > 0.05 and c_cf["lo"] > 0 and recall_delta >= -2.0:
        return "supported"
    if net_saving > 0.05 and c_cf["lo"] > -0.25:
        return "partially supported"
    return "not supported"


def mechanism_dropped(net_saving: float, c_xf: dict) -> bool:
    return net_saving < 0.02 or c_xf["hi"] <= 0


def verdict_h3(j_adapt: dict, j_cache: dict, j_indep: dict, j_fixed: dict) -> str:
    if all(c["lo"] > 0 for c in (j_adapt, j_cache, j_indep, j_fixed)):
        return "supported"
    if j_indep["lo"] > 0 and j_fixed["lo"] > 0:
        return "partially supported"
    return "not supported"


def stop_rule(contrasts_vs_simpler: dict[str, dict], vs_adaptive_cache: list[str]) -> dict:
    """Joint is retained iff for every simpler arm A the (Joint - A) CI upper bound >= 0 and Joint beats at least one of Adaptive / Cache-Only (lo > 0)."""
    not_beaten = all(c["hi"] >= 0 for c in contrasts_vs_simpler.values())
    beats_one = any(contrasts_vs_simpler[a]["lo"] > 0 for a in vs_adaptive_cache if a in contrasts_vs_simpler)
    return {"retained": bool(not_beaten and beats_one), "not_beaten_by_any": bool(not_beaten), "beats_adaptive_or_cache": bool(beats_one),
            "beaten_by": [a for a, c in contrasts_vs_simpler.items() if c["hi"] < 0]}


SIMPLICITY = ["Fixed", "Cache-Only", "Motion-Only", "Adaptive-Only", "Independent-Combo", "Joint-Controller"]


def lock_rule(avg_auc: dict[str, float], valid: dict[str, bool], vs_best: dict[str, dict], vs_fixed: dict[str, dict], margin: float = 0.25) -> dict:
    """Lock = simplest arm whose averaged paired contrast to `best` has CI lower > -margin, provided it beats Fixed with CI lower > 0 (else Fixed)."""
    cand = [a for a in SIMPLICITY if valid.get(a, False) and a in avg_auc]
    best = max(cand, key=lambda a: avg_auc[a])
    for a in cand:
        non_inf = a == best or vs_best[a]["lo"] > -margin
        if non_inf and (a == "Fixed" or vs_fixed[a]["lo"] > 0):
            return {"best": best, "lock": a, "note": ""}
    return {"best": best, "lock": "Fixed", "note": "no scheduler or cache beats Fixed with CI lower bound > 0 while being non-inferior to the best arm"}


# =====================================================================================================================
# 12. Per-dataset analysis object
# =====================================================================================================================
SUBSETS = ("dev", "confirmation", "all")
CONTRASTS = {  # name -> (A, B): A - B
    "H1.Adaptive-Fixed": ("Adaptive-Only", "Fixed"), "H1.Adaptive-Motion": ("Adaptive-Only", "Motion-Only"),
    "H2.Cache-Fixed": ("Cache-Only", "Fixed"), "H2.C1-Fixed": ("Cache-C1", "Fixed"), "H2.C2-Fixed": ("Cache-C2", "Fixed"),
    "H3.Joint-Adaptive": ("Joint-Controller", "Adaptive-Only"), "H3.Joint-Cache": ("Joint-Controller", "Cache-Only"),
    "H3.Joint-Independent": ("Joint-Controller", "Independent-Combo"), "H3.Joint-Fixed": ("Joint-Controller", "Fixed"),
    "X.Joint-Motion": ("Joint-Controller", "Motion-Only"), "X.Motion-Fixed": ("Motion-Only", "Fixed"),
    "X.Independent-Fixed": ("Independent-Combo", "Fixed"), "X.Independent-Adaptive": ("Independent-Combo", "Adaptive-Only"),
    "X.Independent-Cache": ("Independent-Combo", "Cache-Only"), "X.Adaptive-Cache": ("Adaptive-Only", "Cache-Only"),
    "X.Motion-Cache": ("Motion-Only", "Cache-Only"),
}


class Analysis:
    """Delivered / ideal series, event metrics and paired bootstraps of one dataset's run."""

    def __init__(self, w: World, run: dict, cfg: dict):
        self.w, self.run, self.cfg = w, run, cfg
        self.ds = w.ds
        self.theta = cfg["datasets"][w.ds]["constants"]["theta"]
        self.cells = [tuple(c_) for c_ in cfg["cells"]]
        self.cids = [cell_id(*c_) for c_ in self.cells]
        self.fps = FPS[w.ds]
        self.splits = {"dev": w.dev_idx, "confirmation": w.conf_idx, "all": None}
        self.idx_of = {k: (list(range(len(w.names))) if v is None else list(v)) for k, v in self.splits.items()}
        self.cut = np.cumsum(w.n_frames)[:-1]
        self._ser: dict = {}
        self._tab: dict = {}
        self._boot: dict = {}
        self._auc: dict = {}
        self.boots = {k: AucBoot(w.labels, v, N_BOOT, 0) for k, v in self.splits.items()}

    def r(self, arm: str, cid: str) -> dict:
        return self.run["results"][(arm, cid)]

    def series(self, arm: str, cid: str, kind: str = "series") -> list[np.ndarray]:
        key = (arm, cid, kind)
        if key not in self._ser:
            self._ser[key] = np.split(self.r(arm, cid)[kind].astype(np.float64), self.cut)
        return self._ser[key]

    def table(self, arm: str, cid: str) -> list[dict]:
        """Event metrics of every clip (delivered series)."""
        key = (arm, cid)
        if key not in self._tab:
            ser = self.series(arm, cid)
            self._tab[key] = [event_metrics(self.w.labels[i], ser[i], self.theta, self.fps) for i in range(len(ser))]
        return self._tab[key]

    def boot(self, arm: str, cid: str, sub: str) -> tuple[float, np.ndarray]:
        key = (arm, cid, sub)
        if key not in self._boot:
            ser = self.series(arm, cid)
            self._boot[key] = self.boots[sub].aucs([ser[i] for i in self.boots[sub].idx])
        return self._boot[key]

    def auc(self, arm: str, cid: str, sub: str, kind: str = "series") -> float:
        key = (arm, cid, sub, kind)
        if key not in self._auc:
            ser = self.series(arm, cid, kind)
            self._auc[key] = 100 * e00.pooled_auc(self.w.labels, ser, None if sub == "all" else self.splits[sub])
        return self._auc[key]

    def events(self, arm: str, cid: str, sub: str) -> dict:
        tab, idx = self.table(arm, cid), self.idx_of[sub]
        ev = sum(tab[i]["events"] for i in idx)
        h05, h10, det = (sum(tab[i][k] for i in idx) for k in ("hit_0.5", "hit_1.0", "detected"))
        delays = [d for i in idx for d in tab[i]["delays_ms"]]
        fa = sum(tab[i]["false_alerts"] for i in idx)
        hours = sum(tab[i]["normal_frames"] for i in idx) / self.fps / 3600.0
        return {"events": ev, "recall_0.5": 100 * h05 / ev if ev else float("nan"), "recall_1.0": 100 * h10 / ev if ev else float("nan"),
                "missed_rate": 100 * (1 - det / ev) if ev else float("nan"), "delay_p50_ms": float(np.median(delays)) if delays else float("nan"),
                "delay_p90_ms": float(np.quantile(delays, 0.9)) if delays else float("nan"), "false_alerts_per_hour": fa / hours if hours else float("nan"),
                "false_alerts": fa}

    def cost_sub(self, arm: str, cid: str, sub: str) -> dict:
        r, idx = self.r(arm, cid), self.idx_of[sub]
        nfl = self.w.n_frames[idx]
        sec = nfl * self.w.dt / 1000.0
        cpu = float((r["clip_cpu_ms_per_s"][idx] * sec).sum() / sec.sum())
        return {"cpu_ms_per_s": cpu, "wait_mean_ms": float((r["clip_wait_mean_ms"][idx] * nfl).sum() / nfl.sum()), "wait_max_ms": float(r["clip_wait_max_ms"][idx].max()),
                "emit_latency_ms": float(np.nanmean(r["clip_emit_ms"][idx])) if np.isfinite(r["clip_emit_ms"][idx]).any() else float("nan")}

    def cell_row(self, arm: str, cid: str, sub: str) -> dict:
        r = self.r(arm, cid)
        point, vec = self.boot(arm, cid, sub)
        lo, hi = ci_of(vec)
        row = {"arm": arm, "cell": cid, "subset": sub, "s": r["s"], "beta": r["beta"], "auc_delivered": self.auc(arm, cid, sub), "auc_ci95": [lo, hi],
               "auc_ideal": self.auc(arm, cid, sub, "ideal"), "target_cpu_ms_per_s": r["target"], "cost_ratio_all": r["ratio"], "status": r["status"],
               "knob": r["knob"], "w_ref": r["w_ref"], "hold": r["hold"], "mem_mb": r["mem_mb"], "queue_p99_latency_ms": r["queue"]["p99_latency_ms"],
               "queue_max_wait_ms": r["queue"]["max_wait_ms"], "queue_mean_wait_ms": r["queue"]["mean_wait_ms"], "late_dropped_windows": r["dstats"]["dropped_late"]}
        row.update(self.cost_sub(arm, cid, sub))
        row.update(self.events(arm, cid, sub))
        return row

    # ---- contrasts -------------------------------------------------------------------------------------------------------------
    def vecs(self, arm: str, sub: str, cids: list[str] | None = None) -> np.ndarray:
        return np.stack([self.boot(arm, c_, sub)[1] for c_ in (cids or self.cids)])

    def contrast_all(self, a: str, b: str, sub: str, cids: list[str] | None = None) -> dict:
        return contrast(self.vecs(a, sub, cids), self.vecs(b, sub, cids))

    def contrast_cells(self, a: str, b: str, sub: str) -> dict:
        return {c_: contrast(self.boot(a, c_, sub)[1], self.boot(b, c_, sub)[1]) | {"a": self.boot(a, c_, sub)[0], "b": self.boot(b, c_, sub)[0]}
                for c_ in self.cids}

    def avg_auc(self, arm: str, sub: str, cids: list[str] | None = None) -> float:
        return float(np.mean([self.boot(arm, c_, sub)[0] for c_ in (cids or self.cids)]))

    def valid(self, arm: str) -> bool:
        """Budget validity: realised cost <= 1.02 x target in >= 10 / 12 cells (saturated arms that cost less are valid)."""
        return sum(self.r(arm, c_)["ratio"] <= 1.02 for c_ in self.cids) >= 10

    def net_saving(self, arm: str, cid: str) -> float:
        r = self.r(arm, cid)
        S = r["stats"]
        span_s = self.w.total * self.w.dt / 1000.0
        chk = S["c1_check_ms"] + S["c2_check_ms"]
        saved = S["saved_pose_ms"] + S["saved_score_ms"]
        nominal = r["cost"] * span_s + saved - chk
        return float((saved - chk) / nominal) if nominal > 0 else 0.0

    def mean_net_saving(self, arm: str) -> float:
        return float(np.mean([self.net_saving(arm, c_) for c_ in self.cids]))

    def recall_delta(self, a: str, b: str, sub: str = "confirmation") -> float:
        return float(np.mean([self.events(a, c_, sub)["recall_1.0"] - self.events(b, c_, sub)["recall_1.0"] for c_ in self.cids]))


# =====================================================================================================================
# 13. Correctness rules (E3-D1 .. D7), failure cases, cache statistics
# =====================================================================================================================
def check_rules(ctx, ds: str, A: Analysis, cfg: dict) -> dict:
    """Pre-registered rules of plan section 9. Needs the service model (a fresh World with pools)."""
    w = World(ctx, ds)
    w.K = cfg["datasets"][ds]["constants"]
    allc = list(range(len(w.names)))
    c = w.cost(2.0)
    e02sum = json.loads((e02.EXP_DIR / "summary.json").read_text())["headline"][f"{ds}/{PRIMARY[ds]}/locked"]
    rules: dict = {}
    # D1 -------------------------------------------------------------------------------------------------------------------
    diffs, rows = [], {}
    for key in e02sum:
        if not key.startswith("burst") or int(key[5:]) <= w.L:
            continue
        W = int(key[5:])
        R = simulate(w, c, FixedPolicy(w, c, W), allc, light=False)
        ser, _, _ = delivery(w, R, W - 1, ideal=True)
        got = e01.auc_subsets(w.labels, ser, w.dev_idx, w.conf_idx)
        ref = e02sum[key]["ideal"]
        rows[key] = {"e03": got, "e02": {k: ref[k] for k in got}, "max_abs_diff": max(abs(got[k] - ref[k]) for k in got)}
        diffs.append(rows[key]["max_abs_diff"])
    rules["E3-D1"] = {"pass": bool(diffs) and max(diffs) <= 0.02, "max_abs_diff_pt": max(diffs) if diffs else None, "bursts": rows,
                      "detail": "F(W > L) at zero delay vs E02 burst{W} (all / dev / confirmation), tolerance 0.02 pt"}
    # D2 -------------------------------------------------------------------------------------------------------------------
    R = simulate(w, c, FixedPolicy(w, c, 1.0), allc)
    ser, _, _ = delivery(w, R, 0, ideal=True)
    got = e01.auc_subsets(w.labels, ser, w.dev_idx, w.conf_idx)
    ref = e02sum["stride1"]["ideal"]
    d2 = {k: got[k] - ref[k] for k in got}
    rules["E3-D2"] = {"pass": abs(d2["all"]) <= 0.5, "diff_pt": d2, "e03": got, "e02": {k: ref[k] for k in got},
                      "detail": "continuous run tables (tracker re-created every Delta frames) vs E02 K=1 persistent tracker, zero delay; pass if |all| <= 0.5 pt"}
    rules["E3-D2"]["cost_e03"] = total_cost(w, R)
    rules["E3-D2"]["cost_e02"] = e02sum["stride1"]["T4_s2.0_b1.00"]["cpu_ms_per_s"]
    # D3: poisoning rows of frames the scheduler did not observe must not change anything; feedback only from finished jobs -------
    sel = cfg["datasets"][ds]["tuning"]["selected"]
    clips = [w.dev_idx[0], w.conf_idx[0], w.conf_idx[1]]
    ref_w = A.r("Fixed", "s2.0_b0.50")["w_ref"]
    d3 = {}
    for arm in ("Adaptive-Only", "Joint-Controller"):
        knob = A.r(arm, "s2.0_b0.50")["knob"]
        tr = {i: [] for i in range(len(clips))}
        pol = make_policy(w, c, sel[arm], knob, ref_w)
        fb_log = []
        orig = pol.feedback
        pol.feedback = lambda f, v, q, _o=orig: fb_log.append(f) or _o(f, v, q)  # noqa: E731
        R1 = simulate(w, c, pol, clips, trace=tr)
        sb = [int(w.start_base[ci]) for ci in clips]
        saved_rn = [list(col) for col in w.rn]
        E1 = w.ext + 1
        poisoned = 0
        for pos, ci in enumerate(clips):
            n = int(w.n_frames[ci])
            obs = {x[0] for x in tr[pos] if x[1]}
            for st in range((n - w.L) // START_GRID + 1):
                for e in range(E1):
                    t = st * START_GRID + w.L - 1 + e
                    if t >= n or t in obs:
                        continue
                    k = (sb[pos] + st) * E1 + e
                    for r_ in range(w.row_ptr[k], w.row_ptr[k + 1]):
                        for col in w.rn:
                            col[r_] = float("nan")
                        poisoned += 1
        try:
            pol2 = make_policy(w, c, sel[arm], knob, ref_w)
            R2 = simulate(w, c, pol2, clips)
        finally:
            w.rn = saved_rn
        same = bool(np.array_equal(R1["fin"], R2["fin"]) and np.allclose(R1["w_val"], R2["w_val"], equal_nan=False))
        starts = np.array(sorted(fb_log))
        d3[arm] = {"poisoned_rows": poisoned, "identical_after_poison": same, "feedback_calls": len(fb_log)}
    rules["E3-D3"] = {"pass": all(v["identical_after_poison"] and v["poisoned_rows"] > 0 for v in d3.values()), "arms": d3,
                      "detail": "rows of frames the policy did not request are set to NaN: decisions, costs and delivered values unchanged; feedback is popped only for jobs with finish <= job start (simulate())"}
    # D4 / D5 --------------------------------------------------------------------------------------------------------------
    inv, trunc, reland = {}, {}, {}
    for arm in ("Fixed", "Adaptive-Only", "Cache-Only", "Joint-Controller"):
        r0 = A.r(arm, "s2.0_b0.50")
        pol = make_policy(w, c, sel[arm], r0["knob"], ref_w if arm != "Fixed" else r0["knob"])
        R = simulate(w, c, pol, allc)
        ser, st, info = delivery(w, R, r0["hold"])
        soff = np.concatenate([[0], np.cumsum(w.n_frames)])
        ck = e02.check_queue_invariants(np.arange(len(R["fin"])) * w.dt, R["svc"], R["fin"], info["landing"], R["w_t"], w.dt, R["w_pos"], soff)
        inv[arm] = ck
        R_z = dict(R, fin=np.arange(len(R["fin"])) * w.dt)  # zero service: finish == arrival
        ser_z, _, _ = delivery(w, R_z, r0["hold"])
        ser_i, _, _ = delivery(w, R, r0["hold"], ideal=True)
        reland[arm] = bool(all(np.array_equal(a, b) for a, b in zip(ser_z, ser_i)))
        k = len(allc) // 2
        pol_t = make_policy(w, c, sel[arm], r0["knob"], ref_w if arm != "Fixed" else r0["knob"])
        Rt = simulate(w, c, pol_t, allc[:k])
        ser_t, _, _ = delivery(w, Rt, r0["hold"])
        trunc[arm] = bool(all(np.array_equal(a, b) for a, b in zip(ser_t[: k - 1], ser[: k - 1])))
    rules["E3-D4"] = {"pass": all(all(v.values()) for v in inv.values()), "arms": inv, "detail": "E02 queue invariants on a full stream of every headline family at (2.0, 0.50)"}
    rules["E3-D5"] = {"pass": all(reland.values()) and all(trunc.values()), "zero_delay_equals_ideal": reland, "truncation_invariant": trunc,
                      "detail": "zero-service landing == ideal series; simulating only the first half of the clips leaves every earlier clip's delivered series unchanged"}
    # D6 -------------------------------------------------------------------------------------------------------------------
    n_ok, flagged, bad = 0, 0, []
    for (arm, cid), r in A.run["results"].items():
        if 0.97 <= r["ratio"] <= 1.005:
            n_ok += 1
        elif r["status"] in ("under", "over"):
            flagged += 1
        else:
            bad.append((arm, cid, r["ratio"]))
    rules["E3-D6"] = {"pass": not bad, "in_range": n_ok, "flagged_under_or_over": flagged, "unflagged_out_of_range": bad,
                      "detail": "realised cost / target in [0.97, 1.005] after calibration, or the cell is flagged under / over-spending"}
    # D7 -------------------------------------------------------------------------------------------------------------------
    bits = w.tb["bits"]
    max_run = 0
    for lv in range(len(LEVELS)):
        for b_ in bits[:: max(1, len(bits) // 3000), lv]:
            run = best = 0
            for i in range(RMAX[ds]):
                if (int(b_) >> i) & 1:
                    run += 1
                    best = max(best, run)
                else:
                    run = 0
            max_run = max(max_run, best)
    c2_age_max = max((r["stats"].get("c2_age_max", 0) for r in A.run["results"].values()), default=0)
    rules["E3-D7"] = {"pass": max_run <= A1 and c2_age_max <= A2, "max_c1_consecutive_reused_frames": int(max_run), "A1": A1, "max_c2_source_age": int(c2_age_max), "A2": A2,
                      "detail": "C1 never reuses a detector output older than A1, C2 never a score older than A2 (state is reset at every clip / run start; the elevated-anomaly "
                                "source test is the nu_safe condition, unit-tested)"}
    return plain(rules)


def cache_stats(A: Analysis) -> dict:
    out = {}
    tb = A.w.tb
    err = {}
    for li in range(len(LEVELS)):
        h = tb["hist"][li].astype(np.float64)
        if h.sum() > 0:
            centers = (np.arange(60) + 0.5) * 0.01
            cdf = np.cumsum(h) / h.sum()
            err[f"level{li + 1}"] = {"mean_pose_err_height": float((h * centers).sum() / h.sum()), "p90_pose_err_height": float(centers[np.searchsorted(cdf, 0.9)]),
                                     "reused_window_poses": int(tb["n_reused_pose"][li]), "validity_misses_unseen": int(tb["n_unseen"][li])}
    out["pose_error_by_level"] = err
    span_s = A.w.total * A.w.dt / 1000.0
    rows = {}
    for arm in ("Cache-C1", "Cache-C2", "Cache-Only", "Independent-Combo", "Joint-Controller"):
        for cid in A.cids:
            r = A.r(arm, cid)
            S = r["stats"]
            rows[f"{arm}|{cid}"] = {
                "net_saving_pct": 100 * A.net_saving(arm, cid), "saved_ms_per_s": (S["saved_pose_ms"] + S["saved_score_ms"]) / span_s,
                "overhead_ms_per_s": (S["c1_check_ms"] + S["c2_check_ms"]) / span_s,
                "c1_hit_rate_pct": 100 * S["reuse_frames"] / max(S["run_frames"], 1), "c1_refreshes": S["detect_frames"] - S["forced_frames"],
                "c1_source_age_mean": S["c1_age_sum"] / max(S["reuse_frames"], 1),
                "c2_hit_rate_pct": 100 * S["win_reuse"] / max(S["win_cand"], 1), "c2_candidates": S["win_cand"], "c2_reused": S["win_reuse"],
                "c2_abs_err_mean": S["c2_err_sum"] / max(S["win_reuse"], 1), "c2_abs_err_max": S["c2_err_max"],
                "c2_source_age_mean": S["c2_age_sum"] / max(S["win_reuse"], 1), "c2_source_age_max": S.get("c2_age_max", 0)}
    out["by_arm_cell"] = rows
    return out


def failure_analysis(A: Analysis, arms: list[str], cids: list[str], k_worst: int = 5) -> dict:
    """Missed-event taxonomy (scorer-limited vs schedule-induced) and the clips where each arm loses most to Fixed."""
    w = A.w
    ref_ser = A.series("Fixed", "s2.0_b1.00", "ideal")  # K=1 ideal series (W = 1)
    out: dict = {"taxonomy": {}, "worst_clips": {}}
    for cid in cids:
        fixed_ser = A.series("Fixed", cid)
        fixed_auc = [100 * e00.clip_auc(w.labels[i], fixed_ser[i]) for i in range(len(w.names))]
        for arm in arms:
            ser = A.series(arm, cid)
            obs = np.split(A.r(arm, cid)["observed"], A.cut)
            tab = A.table(arm, cid)
            tax = {"events": 0, "detected_1.0s": 0, "missed": 0, "missed_scorer_limited": 0, "missed_schedule_never_observed": 0,
                   "missed_schedule_observed_but_late_or_low": 0}
            per_clip = []
            for i in range(len(w.names)):
                events = label_events(w.labels[i])
                ref_runs = alert_runs(ref_ser[i] > A.theta)
                for s_, e_ in events:
                    tax["events"] += 1
                    window = slice(s_, min(e_ + 1, s_ + int(round(A.fps))))
                    det = False
                    for r0, r1 in alert_runs(ser[i] > A.theta):
                        f = r0 + EVENT_MIN_FRAMES - 1
                        if r1 >= s_ and f <= e_ and max(f, s_) - s_ <= A.fps:
                            det = True
                            break
                    if det:
                        tax["detected_1.0s"] += 1
                        continue
                    tax["missed"] += 1
                    ref_det = any(r1 >= s_ and (r0 + EVENT_MIN_FRAMES - 1) <= e_ and max(r0 + EVENT_MIN_FRAMES - 1, s_) - s_ <= A.fps for r0, r1 in ref_runs)
                    if not ref_det:
                        tax["missed_scorer_limited"] += 1
                    elif obs[i][window].sum() == 0:
                        tax["missed_schedule_never_observed"] += 1
                    else:
                        tax["missed_schedule_observed_but_late_or_low"] += 1
                a_ = 100 * e00.clip_auc(w.labels[i], ser[i])
                per_clip.append((a_ - fixed_auc[i] if np.isfinite(a_) and np.isfinite(fixed_auc[i]) else float("nan"), i))
            out["taxonomy"][f"{arm}|{cid}"] = tax
            worst = sorted([p for p in per_clip if np.isfinite(p[0])])[:k_worst]
            out["worst_clips"][f"{arm}|{cid}"] = [{"clip": w.names[i], "split": "dev" if i in w.dev_idx else "confirmation", "auc_delta_vs_fixed": d_,
                                                    "events_missed": (tab[i]["events"] - tab[i]["detected"]),
                                                    "observed_frac": float(obs[i].mean()), "wait_max_ms": float(A.r(arm, cid)["clip_wait_max_ms"][i]),
                                                    "windows_delivered": int(A.r(arm, cid)["clip_windows"][i]),
                                                    "median_motion": float(np.nanmedian(w.motion[w.clip_slice(i)]))} for d_, i in worst]
    return out


# =====================================================================================================================
# 14. Report stage
# =====================================================================================================================
def fmt(x, nd=2):
    return "n/a" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:.{nd}f}"


def md_table(headers: list[str], rows: list[list]) -> str:
    esc = lambda x: str(x).replace("|", "\\|")  # noqa: E731
    out = ["| " + " | ".join(esc(h) for h in headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(esc(c_) for c_ in r) + " |" for r in rows]
    return "\n".join(out)


def ci_txt(c_: dict, nd=2) -> str:
    return f"{c_['delta']:+.{nd}f} [{c_['lo']:+.{nd}f}, {c_['hi']:+.{nd}f}]"


def e02_reference_contrasts(A: Analysis) -> dict:
    """Propagated-regime comparator: E02 Fixed-Stride (K, CV fill) simulated delivery (violates the intact-window rule) vs our arms at matched cost."""
    p = e02.WORK_DIR / "stages" / f"e02_sim__{A.ds}__{PRIMARY[A.ds]}__locked.pkl"
    out: dict = {"available": False}
    if not p.exists():
        return out
    jobs = pickle.loads(p.read_bytes())["jobs"]
    cfull = jobs["stride1|T4_s2.0_b1.00|continuous|rule|h264"]["cpu_ms_per_s"]
    pairs = [(2, "s2.0_b0.50"), (3, "s2.0_b0.33"), (4, "s2.0_b0.25")]
    rows = {}
    for k, cid in pairs:
        for fill in ("cv", "zoh"):
            key = f"stride{k}-{fill}|T4_s2.0_b1.00|continuous|rule|h264"
            j = jobs.get(key)
            if not j or "series" not in j:
                continue
            ser = np.split(j["series"].astype(np.float64), A.cut)
            entry = {"stride": k, "fill": fill, "cpu_ms_per_s": j["cpu_ms_per_s"], "cost_fraction_of_K1": j["cpu_ms_per_s"] / cfull, "matched_cell": cid,
                     "auc": j["auc"], "contrasts": {}}
            for sub in SUBSETS:
                pt, vec = A.boots[sub].aucs([ser[i] for i in A.boots[sub].idx])
                entry.setdefault("boot_auc", {})[sub] = pt
                for arm in HEADLINE_ARMS:
                    entry["contrasts"].setdefault(arm, {})[sub] = contrast(A.boot(arm, cid, sub)[1], vec) | {"arm_auc": A.boot(arm, cid, sub)[0], "stride_auc": pt}
            rows[f"stride{k}-{fill}"] = entry
    return {"available": bool(rows), "rows": rows, "k1_cost_e02": cfull}


def make_figures(analyses: dict[str, Analysis], out_dir: Path) -> list[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # pragma: no cover
        return []
    paths = []
    colors = {"Fixed": "k", "Motion-Only": "tab:orange", "Adaptive-Only": "tab:blue", "Cache-Only": "tab:green", "Independent-Combo": "tab:purple",
              "Joint-Controller": "tab:red"}
    for ds, A in analyses.items():
        cells = [c_ for c_ in A.cells if c_[0] == 2.0]
        betas = [c_[1] for c_ in cells]
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for arm in HEADLINE_ARMS:
            cids = [cell_id(*c_) for c_ in cells]
            axes[0].plot(betas, [A.boot(arm, c_, "confirmation")[0] for c_ in cids], "o-", color=colors[arm], label=arm)
            axes[1].plot(betas, [A.events(arm, c_, "confirmation")["delay_p50_ms"] for c_ in cids], "o-", color=colors[arm], label=arm)
            axes[2].plot(betas, [A.events(arm, c_, "confirmation")["recall_1.0"] for c_ in cids], "o-", color=colors[arm], label=arm)
        for ax, t in zip(axes, ("delivered AUC (%), test_confirmation", "detection delay p50 (ms), test_confirmation", "event recall @1.0 s (%), test_confirmation")):
            ax.set_xlabel("budget beta (x K=1 CPU cost), s = 2.0")
            ax.set_title(f"{ds}: {t}", fontsize=9)
            ax.grid(alpha=0.3)
        axes[0].legend(fontsize=7)
        fig.tight_layout()
        p = out_dir / f"fig_{ds}_matched_cost_curves.png"
        fig.savefig(p, dpi=130)
        plt.close(fig)
        paths.append(p.name)
    return paths


def verdicts_for(A: Analysis, sub: str = "confirmation") -> dict:
    c = {name: A.contrast_all(a, b, sub) for name, (a, b) in CONTRASTS.items()}
    v: dict = {"subset": sub, "contrasts": c}
    v["H1"] = verdict_h1(c["H1.Adaptive-Fixed"], c["H1.Adaptive-Motion"])
    net = {arm: A.mean_net_saving(arm) for arm in ("Cache-C1", "Cache-C2", "Cache-Only")}
    v["net_saving"] = net
    v["recall_delta_cache_fixed_pt"] = A.recall_delta("Cache-Only", "Fixed", sub)
    v["H2"] = verdict_h2(net["Cache-Only"], c["H2.Cache-Fixed"], v["recall_delta_cache_fixed_pt"])
    v["H2_C1"] = {"net_saving": net["Cache-C1"], "dropped": mechanism_dropped(net["Cache-C1"], c["H2.C1-Fixed"]),
                  "verdict": verdict_h2(net["Cache-C1"], c["H2.C1-Fixed"], A.recall_delta("Cache-C1", "Fixed", sub))}
    v["H2_C2"] = {"net_saving": net["Cache-C2"], "dropped": mechanism_dropped(net["Cache-C2"], c["H2.C2-Fixed"]),
                  "verdict": verdict_h2(net["Cache-C2"], c["H2.C2-Fixed"], A.recall_delta("Cache-C2", "Fixed", sub))}
    v["H3"] = verdict_h3(c["H3.Joint-Adaptive"], c["H3.Joint-Cache"], c["H3.Joint-Independent"], c["H3.Joint-Fixed"])
    simpler = {"Fixed": "H3.Joint-Fixed", "Cache-Only": "H3.Joint-Cache", "Motion-Only": "X.Joint-Motion", "Adaptive-Only": "H3.Joint-Adaptive", "Independent-Combo": "H3.Joint-Independent"}
    v["stop_rule"] = stop_rule({a: c[n] for a, n in simpler.items()}, ["Adaptive-Only", "Cache-Only"])
    avg = {arm: A.avg_auc(arm, sub) for arm in HEADLINE_ARMS}
    valid = {arm: A.valid(arm) for arm in HEADLINE_ARMS}
    best = max((a for a in HEADLINE_ARMS if valid[a]), key=lambda a: avg[a])
    vs_best = {a: (contrast(A.vecs(a, sub), A.vecs(best, sub)) if a != best else {"delta": 0.0, "lo": 0.0, "hi": 0.0}) for a in HEADLINE_ARMS}
    vs_fixed = {a: (contrast(A.vecs(a, sub), A.vecs("Fixed", sub)) if a != "Fixed" else {"delta": 0.0, "lo": 0.0, "hi": 0.0}) for a in HEADLINE_ARMS}
    # mechanisms that were dropped cannot be the cache candidate (Cache-Only is evaluated as run: C1 + C2; if C2 is dropped the C1-only arm is the cache candidate)
    cache_name = "Cache-Only"
    if v["H2_C2"]["dropped"] and not v["H2_C1"]["dropped"]:
        cache_name = "Cache-C1"
    elif v["H2_C1"]["dropped"] and v["H2_C2"]["dropped"]:
        cache_name = None
    v["cache_candidate"] = cache_name
    avg_l, valid_l, vb_l, vf_l = dict(avg), dict(valid), dict(vs_best), dict(vs_fixed)
    if cache_name == "Cache-C1":
        avg_l["Cache-Only"] = A.avg_auc("Cache-C1", sub)
        valid_l["Cache-Only"] = A.valid("Cache-C1")
        vb_l["Cache-Only"] = contrast(A.vecs("Cache-C1", sub), A.vecs(best, sub)) if best != "Cache-C1" else {"delta": 0.0, "lo": 0.0, "hi": 0.0}
        vf_l["Cache-Only"] = contrast(A.vecs("Cache-C1", sub), A.vecs("Fixed", sub))
    elif cache_name is None:
        valid_l["Cache-Only"] = False
    v["avg_auc"], v["valid"], v["best_arm"] = avg, valid, best
    v["lock"] = lock_rule(avg_l, valid_l, vb_l, vf_l)
    v["lock"]["resolved_name"] = ("Cache-C1 (C1 only; C2 dropped)" if (v["lock"]["lock"] == "Cache-Only" and cache_name == "Cache-C1") else v["lock"]["lock"])
    v["vs_best"], v["vs_fixed"] = vs_best, vs_fixed
    return v


def write_results_csv(analyses: dict[str, Analysis], path: Path) -> int:
    cols = ["dataset", "video", "split", "arm", "cell", "s", "beta", "target_cpu_ms_per_s", "cpu_ms_per_video_s", "cost_ratio_stream", "status", "knob", "w_ref", "hold",
            "n_frames", "fps", "anomalous_frames", "auc_delivered", "auc_ideal", "events", "hit_0.5s", "hit_1.0s", "delay_p50_ms", "false_alerts", "normal_hours",
            "queue_wait_mean_ms", "queue_wait_max_ms", "emit_latency_ms", "windows_delivered", "observed_fraction", "peak_mem_mb_stream"]
    n = 0
    with path.open("w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(cols)
        for ds, A in analyses.items():
            w = A.w
            dev = set(w.dev_idx)
            for (arm, cid), r in sorted(A.run["results"].items(), key=lambda kv: (ARMS.index(kv[0][0]), kv[0][1])):
                ser, idl = A.series(arm, cid), A.series(arm, cid, "ideal")
                tab = A.table(arm, cid)
                obs = np.split(r["observed"], A.cut)
                for i, name in enumerate(w.names):
                    d = tab[i]["delays_ms"]
                    wr.writerow([ds, name, "dev" if i in dev else "confirmation", arm, cid, r["s"], r["beta"], round(r["target"], 3), round(float(r["clip_cpu_ms_per_s"][i]), 3),
                                 round(r["ratio"], 4), r["status"], round(r["knob"], 4), round(r["w_ref"], 3), r["hold"], int(w.n_frames[i]), A.fps, int(w.labels[i].sum()),
                                 round(100 * e00.clip_auc(w.labels[i], ser[i]), 4), round(100 * e00.clip_auc(w.labels[i], idl[i]), 4), tab[i]["events"], tab[i]["hit_0.5"],
                                 tab[i]["hit_1.0"], round(float(np.median(d)), 2) if d else "", tab[i]["false_alerts"], round(tab[i]["normal_frames"] / A.fps / 3600, 6),
                                 round(float(r["clip_wait_mean_ms"][i]), 3), round(float(r["clip_wait_max_ms"][i]), 3),
                                 round(float(r["clip_emit_ms"][i]), 3) if np.isfinite(r["clip_emit_ms"][i]) else "", int(r["clip_windows"][i]), round(float(obs[i].mean()), 4),
                                 round(r["mem_mb"], 1)])
                    n += 1
    return n


def stage_report(ctx, procs: int, log=print) -> dict:
    cfg = load_config()
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    analyses: dict[str, Analysis] = {}
    summary: dict = {"experiment": "Q2-P1-E03", "title": "Core ablation", "machine": MACHINE_TAG, "seed": SEED,
                     "hardware_disclosure": "physical Pi 5 not connected: VM service-time pools (comparative proxy) x Pi multiplier s; simulated delivery on replayed detections",
                     "config_sha256": sha(EXP_DIR / "config.yaml"), "datasets": {}}
    decisions: list[dict] = []
    t0 = time.perf_counter()
    for ds in DATASETS:
        w = World(ctx, ds, load_service=False)
        w.K = cfg["datasets"][ds]["constants"]
        run = pickle.loads(run_path(ds).read_bytes())
        assert run["stamp"] == summary["config_sha256"], "run stage is stale: config.yaml changed since `run`"
        A = Analysis(w, run, cfg)
        analyses[ds] = A
        d: dict = {"n_clips": len(w.names), "n_dev": len(w.dev_idx), "n_confirmation": len(w.conf_idx), "theta_alert": A.theta, "selected": cfg["datasets"][ds]["tuning"]["selected"],
                   "constants": cfg["datasets"][ds]["constants"], "c_full": cfg["datasets"][ds]["c_full"], "w_ref": run["w_ref"]}
        log(f"[{now()}] report {ds}: cell rows ...")
        d["cells"] = [A.cell_row(arm, cid, sub) for arm in ARMS for cid in A.cids for sub in SUBSETS]
        log(f"[{now()}] report {ds}: contrasts / verdicts ...")
        d["verdicts"] = {sub: verdicts_for(A, sub) for sub in SUBSETS}
        d["contrast_cells_confirmation"] = {name: A.contrast_cells(a, b, "confirmation") for name, (a, b) in CONTRASTS.items() if name.startswith(("H1", "H2.Cache", "H3"))}
        budget_cids = [c_ for c_, cell in zip(A.cids, A.cells) if cell[1] < 1.0]
        d["contrasts_budget_cells_only"] = {sub: {name: A.contrast_all(a, b, sub, budget_cids) for name, (a, b) in CONTRASTS.items()} for sub in SUBSETS}
        stable = [c_ for c_ in A.cids if A.r("Fixed", c_)["queue"]["sustainable"]]
        d["stable_cells"] = stable
        d["contrasts_stable_cells"] = {sub: {name: A.contrast_all(a, b, sub, stable) for name, (a, b) in CONTRASTS.items()} for sub in SUBSETS} if stable else {}
        d["collapsed_cells"] = {f"{arm}": [c_ for c_ in A.cids if not A.r(arm, c_)["queue"]["sustainable"]] for arm in ARMS}
        d["cache"] = cache_stats(A)
        d["calibration"] = {f"{arm}|{cid}": {k: A.r(arm, cid)[k] for k in ("knob", "target", "cost", "ratio", "status", "evals", "w_ref", "hold")} | {"mem_mb": A.r(arm, cid)["mem_mb"]}
                            for arm in ARMS for cid in A.cids}
        d["tuning_tables"] = cfg["datasets"][ds]["tuning"]["tables"]
        d["propagated_reference"] = e02_reference_contrasts(A)
        log(f"[{now()}] report {ds}: failure analysis ...")
        d["failure"] = failure_analysis(A, ["Fixed", "Motion-Only", "Adaptive-Only", "Cache-Only", "Joint-Controller"], ["s2.0_b0.33", "s2.0_b0.50", "s2.0_b0.75"])
        log(f"[{now()}] report {ds}: correctness rules ...")
        d["rules"] = check_rules(ctx, ds, A, cfg)
        summary["datasets"][ds] = d
        for key, rule in d["rules"].items():
            decisions.append({"type": "rule", "dataset": ds, "rule": key, "pass": rule["pass"], "detail": rule.get("detail")})
        for arm, c_ in d["selected"].items():
            decisions.append({"type": "selection", "dataset": ds, "arm": arm, "config": c_})
        for key, v in d["calibration"].items():
            decisions.append({"type": "calibration", "dataset": ds, "arm_cell": key, **v})
        for sub in SUBSETS:
            v = d["verdicts"][sub]
            decisions.append({"type": "verdict", "dataset": ds, "subset": sub, "H1": v["H1"], "H2": v["H2"], "H2_C1": v["H2_C1"], "H2_C2": v["H2_C2"], "H3": v["H3"],
                              "stop_rule": v["stop_rule"], "lock": v["lock"]})
        for arm, tr in run["traces"].items():
            decisions.append({"type": "controller_trace", "dataset": ds, "arm": arm, **tr})
    n_rows = write_results_csv(analyses, EXP_DIR / "results.csv")
    figs = make_figures(analyses, EXP_DIR)
    summary["figures"] = figs
    summary["rows_results_csv"] = n_rows
    summary["headline"] = {"verdict_dataset": "ShanghaiTech", "verdict_subset": "confirmation",
                           **{k: summary["datasets"]["ShanghaiTech"]["verdicts"]["confirmation"][k] for k in ("H1", "H2", "H3", "stop_rule", "lock", "H2_C1", "H2_C2", "best_arm")},
                           "ubnormal_replication": {k: summary["datasets"]["UBnormal"]["verdicts"]["confirmation"][k] for k in ("H1", "H2", "H3", "stop_rule", "lock", "H2_C1", "H2_C2", "best_arm")}}
    summary["rules"] = {ds: {k: v["pass"] for k, v in summary["datasets"][ds]["rules"].items()} for ds in DATASETS}
    summary["runtime_seconds"] = time.perf_counter() - t0
    (EXP_DIR / "summary.json").write_text(json.dumps(plain(summary), indent=1))
    with (EXP_DIR / "decisions.jsonl").open("w") as fh:
        for rec in decisions:
            fh.write(json.dumps(plain(rec)) + "\n")
    write_report(summary, analyses, EXP_DIR / "report.md")
    log(f"[{now()}] report written ({time.perf_counter() - t0:.0f}s)")
    return summary


def _cell_sort(cids: list[str]) -> list[str]:
    return sorted(cids, key=lambda c_: (float(c_.split("_")[0][1:]), float(c_.split("_b")[1])))


def write_report(summary: dict, analyses: dict[str, Analysis], path: Path) -> None:
    S = summary["datasets"]
    L: list[str] = []
    add = L.append
    cfg = load_config()
    head = summary["headline"]
    add("# Q2-P1-E03 — Core ablation: report\n")
    add("**Hardware / honesty disclosure.** The physical Raspberry Pi 5 is **not connected**. Every service time is sampled from the E02 VM pools "
        "(`edge-os-arm64-vm`, a comparative proxy) and scaled by a Pi multiplier `s`; all E03 numbers are *simulated delivery on replayed detections* "
        "(frozen frontend `ML0_ncnn`@640 for ShanghaiTech, `ML0_mps` proxy for UBnormal, locked BoT-SORT tracker, frozen STG-NF scorer, E00 causal S3/S4 "
        "protocol), not Pi measurements. No training was done. All thresholds come from `test_dev` or the E00 train sentinels; **verdicts use `test_confirmation` "
        "of ShanghaiTech (72 videos)**, UBnormal (141 confirmation videos) is the replication.\n")
    add("## 1. Answer in one paragraph\n")
    v = S["ShanghaiTech"]["verdicts"]["confirmation"]
    u = S["UBnormal"]["verdicts"]["confirmation"]
    c = v["contrasts"]
    add(f"At matched total CPU cost (12-cell average, SHT `test_confirmation`, delivered AUC): **cross-frame detector reuse (C1) is the only mechanism that clearly pays** "
        f"(Cache-Only − Fixed {ci_txt(c['H2.Cache-Fixed'])} pt, net CPU saving {100 * v['net_saving']['Cache-Only']:.0f} %), while **event-aware scheduling does not beat a fixed "
        f"burst schedule** (Adaptive − Fixed {ci_txt(c['H1.Adaptive-Fixed'])} pt; Motion-Only is worse than Fixed, {ci_txt(c['X.Motion-Fixed'])}), "
        f"and **the joint controller loses after overhead** (Joint − Cache-Only {ci_txt(c['H3.Joint-Cache'])}, Joint − Independent {ci_txt(c['H3.Joint-Independent'])}). "
        f"Scorer reuse (C2) saves only {100 * v['net_saving']['Cache-C2']:.1f} % CPU on average (scoring is ~4 % of the K=1 CPU and C2 hits <= 18 % of the candidate windows) and is dropped. "
        f"Verdicts — **H1: {v['H1']}**, **H2: {v['H2']}**, **H3: {v['H3']}**; stop rule: Joint-Controller **{'retained' if v['stop_rule']['retained'] else 'rejected'}**; "
        f"**lock for E04/E05: `{v['lock']['resolved_name']}`**. UBnormal replicates only part of this (H1 {u['H1']}, H2 {u['H2']}, H3 {u['H3']}; best arm there: "
        f"`{u['best_arm']}`), because most of its cells are overload cells in which only backlog-aware arms survive (section 7).\n")
    add("## 2. Setup actually used\n")
    sel_rows = []
    for ds in DATASETS:
        for arm, cfg_ in S[ds]["selected"].items():
            sel_rows.append([ds, arm, cfg_label(cfg_) or "(none)"])
    add("Selected on the `test_dev` stream (3 tuning cells (2.0,0.33), (2.0,0.75), (3.0,1.00); ties within 0.10 pt -> simpler); everything else fixed a priori in `config.yaml`:\n")
    add(md_table(["dataset", "arm", "selected hyper-parameters"], sel_rows) + "\n")
    add("Cells: `s = 2.0` with `beta in {0.25, 0.33, 0.50, 0.67, 0.75, 1.00}`; overload cells `s in {2.5, 3.0}` with `beta in {0.50, 0.75, 1.00}`. Target = `beta x C_full(s)` "
        "(`C_full` = CPU of the K=1 pipeline on the same stream; SHT 2276 CPU-ms/s at s=2). Each arm's rate knob is bisected label-free until the realised CPU "
        "(incl. observer / C1-check / C2-check) is within [0.97, 1.0] of the target (flag `under` = cannot spend the budget: full coverage already reached, "
        "`over` = cannot get below it).\n")
    add("## 3. Correctness rules (pre-registered, `summary.json -> rules`)\n")
    rr = []
    for key in ("E3-D1", "E3-D2", "E3-D3", "E3-D4", "E3-D5", "E3-D6", "E3-D7"):
        rr.append([key, S["ShanghaiTech"]["rules"][key]["detail"], "PASS" if S["ShanghaiTech"]["rules"][key]["pass"] else "FAIL", "PASS" if S["UBnormal"]["rules"][key]["pass"] else "FAIL"])
    add(md_table(["rule", "statement", "SHT", "UB"], rr) + "\n")
    r1, r2 = S["ShanghaiTech"]["rules"]["E3-D1"], S["ShanghaiTech"]["rules"]["E3-D2"]
    add(f"E3-D1 max |AUC difference| to E02 burst baselines = {r1['max_abs_diff_pt']:.4f} pt (exact replay). E3-D2: continuous run tables (tracker re-created every Delta frames) differ from "
        f"E02's persistent K=1 tracker by {r2['diff_pt']['all']:+.2f} pt (all clips; dev {r2['diff_pt']['dev']:+.2f}, confirmation {r2['diff_pt']['confirmation']:+.2f}) at "
        f"{r2['cost_e03']:.0f} vs {r2['cost_e02']:.0f} CPU-ms/s — a small, **common** handicap of every continuous-run arm.\n")
    for ds in DATASETS:
        A = analyses[ds]
        d = S[ds]
        add(f"## 4{'a' if ds == 'ShanghaiTech' else 'b'}. Matched-cost accuracy / delay curves — {ds}\n")
        add(f"![matched-cost curves](fig_{ds}_matched_cost_curves.png)\n")
        cells = _cell_sort(A.cids)
        cols = ["cell"] + list(HEADLINE_ARMS)
        for sub in ("confirmation", "dev"):
            rows = []
            for cid in cells:
                rows.append([cid] + [f"{A.boot(a, cid, sub)[0]:.2f}" + ("*" if A.r(a, cid)["status"] == "under" else "") + ("!" if not A.r(a, cid)["queue"]["sustainable"] else "")
                                     for a in HEADLINE_ARMS])
            rows.append(["**mean of 12**"] + [f"**{A.avg_auc(a, sub):.2f}**" for a in HEADLINE_ARMS])
            add(f"**Delivered AUC (%), `{sub}`** (pooled over videos; `*` = arm under-spends its budget, `!` = unsustainable queue: late work discarded)\n")
            add(md_table(cols, rows) + "\n")
        rows = []
        for cid in cells:
            rows.append([cid] + [f"{A.auc(a, cid, 'confirmation', 'ideal'):.2f}" for a in HEADLINE_ARMS])
        add("**Ideal AUC (%), `confirmation`** (same decisions, zero service delay)\n")
        add(md_table(cols, rows) + "\n")
        add("**Event metrics, `confirmation`** (alert = delayed score above the locked dev threshold for >= 6 frames; recall = fraction of events detected within 0.5 / 1.0 s; "
            f"delay p50 / p90 over detected events; FA/h per hour of normal video; {A.events('Fixed', cells[0], 'confirmation')['events']} events)\n")
        rows = []
        for cid in ("s2.0_b0.33", "s2.0_b0.50", "s2.0_b0.75", "s3.0_b1.00"):
            for a in HEADLINE_ARMS:
                e = A.events(a, cid, "confirmation")
                cs = A.cost_sub(a, cid, "confirmation")
                rows.append([cid, a, f"{e['recall_0.5']:.1f}", f"{e['recall_1.0']:.1f}", f"{e['missed_rate']:.1f}", fmt(e["delay_p50_ms"], 0), fmt(e["delay_p90_ms"], 0),
                             f"{e['false_alerts_per_hour']:.0f}", f"{cs['cpu_ms_per_s']:.0f}", f"{A.r(a, cid)['ratio']:.3f}", f"{cs['wait_mean_ms']:.0f}",
                             f"{A.r(a, cid)['queue']['p99_latency_ms']:.0f}", f"{A.r(a, cid)['queue']['max_wait_ms']:.0f}", f"{A.r(a, cid)['mem_mb']:.0f}"])
        add(md_table(["cell", "arm", "recall@0.5s %", "recall@1.0s %", "missed %", "delay p50 ms", "delay p90 ms", "FA/h", "CPU ms/s", "cost/target", "queue mean ms",
                      "p99 latency ms", "max wait ms", "peak mem MB"], rows) + "\n")
    add("## 5. Hypotheses (12-cell average of paired video-bootstrap contrasts, delivered AUC in pt; verdicts on SHT `test_confirmation`)\n")
    for ds in DATASETS:
        vv = S[ds]["verdicts"]
        add(f"**{ds}** — contrasts A − B (`dev` | `confirmation` | `all`; delta [95 % CI]):\n")
        rows = []
        for name, (a, b) in CONTRASTS.items():
            rows.append([name] + [ci_txt(vv[sub]["contrasts"][name]) for sub in SUBSETS])
        add(md_table(["contrast", *SUBSETS], rows) + "\n")
        st = S[ds]["stable_cells"]
        add(f"Sensitivity — only the {len(st)} cells in which Fixed's queue is sustainable ({', '.join(st)}) and the budget cells beta < 1 are in `summary.json` "
            f"(`contrasts_stable_cells`, `contrasts_budget_cells_only`).\n")
        add(f"Verdicts ({ds}, confirmation): **H1 {vv['confirmation']['H1']}**; **H2 {vv['confirmation']['H2']}** (net saving {100 * vv['confirmation']['net_saving']['Cache-Only']:.1f} %, "
            f"recall@1.0 s Cache − Fixed {vv['confirmation']['recall_delta_cache_fixed_pt']:+.2f} pt); C1: {vv['confirmation']['H2_C1']['verdict']} "
            f"(saving {100 * vv['confirmation']['H2_C1']['net_saving']:.1f} %, {'dropped' if vv['confirmation']['H2_C1']['dropped'] else 'kept'}); C2: {vv['confirmation']['H2_C2']['verdict']} "
            f"(saving {100 * vv['confirmation']['H2_C2']['net_saving']:.1f} %, {'dropped' if vv['confirmation']['H2_C2']['dropped'] else 'kept'}); **H3 {vv['confirmation']['H3']}**.\n")
    add("## 6. Cache mechanisms (C1 detector-output reuse, C2 scorer reuse)\n")
    for ds in DATASETS:
        ca = S[ds]["cache"]
        add(f"**{ds}** pose error of reused poses (height-normalised mean / p90) by C1 level: " +
            "; ".join(f"L{k[-1]}: {v_['mean_pose_err_height']:.3f} / {v_['p90_pose_err_height']:.3f}" for k, v_ in ca["pose_error_by_level"].items()) + ".\n")
        rows = []
        for cid in ("s2.0_b0.33", "s2.0_b0.67", "s2.0_b1.00"):
            for arm in ("Cache-C1", "Cache-C2", "Cache-Only", "Independent-Combo", "Joint-Controller"):
                x = ca["by_arm_cell"][f"{arm}|{cid}"]
                rows.append([cid, arm, f"{x['net_saving_pct']:.1f}", f"{x['saved_ms_per_s']:.0f}", f"{x['overhead_ms_per_s']:.1f}", f"{x['c1_hit_rate_pct']:.1f}", x["c1_refreshes"],
                             f"{x['c1_source_age_mean']:.2f}", f"{x['c2_hit_rate_pct']:.1f}", f"{x['c2_abs_err_mean']:.4f}", f"{x['c2_source_age_mean']:.2f}"])
        add(md_table(["cell", "arm", "net saving %", "saved ms/s", "overhead ms/s", "C1 hit %", "C1 refreshes", "C1 src age (frames)", "C2 hit %", "C2 |dNormality| mean",
                      "C2 src age"], rows) + "\n")
    add("C1 logs, per reused frame, the source detector frame (age <= 4), the validity statistics and the forced refresh; every window containing reused poses is **re-scored**, "
        "so the AUC effect already includes the pose error. C1 identity is an oracle (track membership of the run is kept) — slightly favourable to C1. C2 only fires when "
        "consecutive windows of one lineage exist (continuous or extended runs; bursts of L frames give one window per person), and even then STG-NF scoring is only ~4 % of the "
        "K=1 CPU, so a 10-18 % hit rate saves ~1 % before its check overhead.\n")
    add("## 7. Overload cells and failure cases\n")
    for ds in DATASETS:
        d = S[ds]
        add(f"**{ds}** — cells in which an arm's queue is unsustainable (late work > 2.0 s discarded, queue grows): " +
            "; ".join(f"{arm}: {', '.join(cs) if cs else 'none'}" for arm, cs in d["collapsed_cells"].items() if arm in HEADLINE_ARMS) + ".\n")
        tx = d["failure"]["taxonomy"]
        rows = []
        for key, t_ in tx.items():
            if key.split("|")[1] in ("s2.0_b0.33", "s2.0_b0.50"):
                rows.append([key.split("|")[1], key.split("|")[0], t_["events"], t_["detected_1.0s"], t_["missed"], t_["missed_scorer_limited"], t_["missed_schedule_never_observed"],
                             t_["missed_schedule_observed_but_late_or_low"]])
        add("Missed-event taxonomy (all videos; *scorer-limited* = the K=1 ideal series also misses it within 1.0 s):\n")
        add(md_table(["cell", "arm", "events", "detected <=1.0 s", "missed", "scorer-limited", "schedule: never observed in 1st second", "schedule: observed but late/low"], rows) + "\n")
        rows = []
        for key in ("Fixed|s2.0_b0.33", "Adaptive-Only|s2.0_b0.33", "Joint-Controller|s2.0_b0.33", "Cache-Only|s2.0_b0.33"):
            for wc in d["failure"]["worst_clips"][key][:3]:
                rows.append([key.replace("|", " @ "), wc["clip"], wc["split"], f"{wc['auc_delta_vs_fixed']:+.1f}", wc["events_missed"], f"{100 * wc['observed_frac']:.0f}", f"{wc['median_motion']:.3f}"])
        add("Clips where an arm loses most to Fixed (clip-level AUC delta, pt):\n")
        add(md_table(["arm @ cell", "clip", "split", "AUC delta vs Fixed", "events missed", "frames observed %", "median motion"], rows) + "\n")
    add("## 8. Propagated-regime reference (E02 Fixed-Stride, violates the intact-window rule)\n")
    for ds in DATASETS:
        pr = S[ds]["propagated_reference"]
        if not pr.get("available"):
            continue
        rows = []
        for k_, e_ in pr["rows"].items():
            if not k_.endswith("cv"):
                continue
            rows.append([k_, f"{e_['cost_fraction_of_K1']:.3f}", e_["matched_cell"], f"{e_['boot_auc']['confirmation']:.2f}"] +
                        [ci_txt(e_["contrasts"][a]["confirmation"], 1) for a in ("Fixed", "Cache-Only", "Independent-Combo")])
        add(f"**{ds}** E02 stride-K (constant-velocity fill) delivered AUC on `confirmation` and our arm minus stride at the matched-cost cell:\n")
        add(md_table(["E02 arm", "cost / K=1", "matched cell", "AUC", "Fixed − stride", "Cache-Only − stride", "Independent − stride"], rows) + "\n")
    add("## 9. Stop rule and lock\n")
    for ds in DATASETS:
        vv = S[ds]["verdicts"]["confirmation"]
        add(f"**{ds}** — stop rule: Joint {'retained' if vv['stop_rule']['retained'] else 'REJECTED'} (beaten by: {', '.join(vv['stop_rule']['beaten_by']) or 'none'}; "
            f"beats Adaptive-Only or Cache-Only: {vv['stop_rule']['beats_adaptive_or_cache']}). Best 12-cell arm: `{vv['best_arm']}`. Lock rule -> `{vv['lock']['resolved_name']}`"
            f"{' — ' + vv['lock']['note'] if vv['lock']['note'] else ''}.\n")
    lock = head["lock"]
    add(f"**Locked for E04/E05 (decided on SHT `test_confirmation` per the pre-registered rule): `{lock['resolved_name']}`.** Concretely: the Fixed-Burst schedule with C1 level "
        f"{S['ShanghaiTech']['selected']['Cache-C1']['l1']} (static-validity detector skip, `A1 = 4`), no C2, no observer, no controller; the burst period is the cost-matched "
        f"`W` of each cell. E04/E05 inherit the cell grid, the tuned thresholds in `config.yaml` and the hold rule `H = W_ref - 1`.\n")
    interp = path.parent / "interpretation.md"
    if interp.exists():
        add(interp.read_text())
    add("## 10. Reproduction\n")
    add("`uv run --extra pose python -m edgelab.analysis.q2_p1_e03 {motion, tables, lock, run, report, all}`; tests: `uv run pytest tests/test_q2_p1_e03.py`. "
        "Intermediate caches (motion statistics, run tables, run pickles): `~/thesis-data/q2-p1-e03/`.\n")
    path.write_text("\n".join(L))


# =====================================================================================================================
# 15. CLI
# =====================================================================================================================
def main() -> None:
    ap = argparse.ArgumentParser(description="Q2-P1-E03 core ablation")
    ap.add_argument("command", choices=["motion", "tables", "lock", "run", "report", "all"])
    ap.add_argument("--procs", type=int, default=max(2, (os.cpu_count() or 4) - 2))
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dataset", choices=DATASETS, default=None)
    args = ap.parse_args()
    dss = [args.dataset] if args.dataset else list(DATASETS)
    cmds = ["motion", "tables", "lock", "run", "report"] if args.command == "all" else [args.command]
    ctx = None
    for cmd in cmds:
        if cmd == "motion":
            for ds in dss:
                print(stage_motion(ds, args.procs, force=args.force))
            continue
        ctx = ctx or e01.build_ctx(args.procs)
        if cmd == "tables":
            for ds in dss:
                stage_tables(ds, ctx, args.procs, force=args.force)
        elif cmd == "lock":
            stage_lock(args.procs)
        elif cmd == "run":
            for ds in dss:
                stage_run(ctx, ds, args.procs, force=args.force)
        elif cmd == "report":
            stage_report(ctx, args.procs)


if __name__ == "__main__":
    main()
