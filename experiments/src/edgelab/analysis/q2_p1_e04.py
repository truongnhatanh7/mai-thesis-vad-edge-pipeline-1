"""Q2-P1-E04 (Optional optimizations): codec motion vectors (+MV), then pose-model post-training INT8 (+INT8), one change at a time.

    uv run --extra pose python -m edgelab.analysis.q2_p1_e04 prep      # inputs: ML0 ONNX + INT8 (calibration only), H.264 re-encode + MV cache, detector caches
    uv run --extra pose python -m edgelab.analysis.q2_p1_e04 vm        # VM profile: detector T=1..4 (NCNN FP16 / ORT FP32 / ORT INT8) and decode / MV / pixel path
    uv run --extra pose python -m edgelab.analysis.q2_p1_e04 tables    # MV statistics + test_dev thresholds, exact run tables (MV / advection / INT8 / E03 rebuild check)
    uv run --extra pose python -m edgelab.analysis.q2_p1_e04 lock      # config.yaml: hashes, rules, VM costs, MV thresholds from test_dev (BEFORE any metric)
    uv run --extra pose python -m edgelab.analysis.q2_p1_e04 run       # 12-cell matched-cost runs (MV, advection, INT8) + detector quality (S0 / S4, recall by height)
    uv run --extra pose python -m edgelab.analysis.q2_p1_e04 report    # decisions, summary.json, results.csv, decisions.jsonl, report.md

No training. Builds on E00 (causal S3/S4 protocol, splits), E01 (locked frontend / tracker), E02 (VM service pools, true time-causal queue) and
E03 (locked candidate `Cache-C1`: fixed burst schedule + C1 detector-output reuse). Every change is evaluated against the E03 `Cache-C1`
replayed bit-exactly (rule E4-D1); matched cost = the *same absolute CPU target* as E03's cell (so an optimisation that saves CPU can only
show it as a denser schedule, and one that costs CPU as a sparser one).
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import multiprocessing as mp
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import yaml

from edgelab.analysis import q2_p1_e00 as e00
from edgelab.analysis import q2_p1_e01 as e01
from edgelab.analysis import q2_p1_e02 as e02
from edgelab.analysis import q2_p1_e03 as e03
from edgelab.analysis import q2_p1_e04_vm as vmm

DATASETS = e00.DATASETS
SEG_LENS = e00.SEG_LENS
FPS = e00.FPS
SEED = e00.SEED
N_BOOT = 1000
EXP_DIR = e00.EXPERIMENTS_ROOT / "Q2-P1-E04_optimizations"
DATA_ROOT = e00.DATA_ROOT
WORK_DIR = DATA_ROOT / "q2-p1-e04"
MV_DIR = WORK_DIR / "mv_cache"
H264_DIR = WORK_DIR / "h264"
MODEL_DIR = WORK_DIR / "models"
CACHE_DIR = WORK_DIR / "cache"
VM_EXCHANGE = Path(os.environ.get("EDGELAB_EXCHANGE", str(DATA_ROOT / "vm-exchange")))
VM_E04 = VM_EXCHANGE / "q2-p1-e04"
VM_OUT = VM_E04 / "out"
SHT_FRAMES_ROOT = DATA_ROOT / "shanghaitech/shanghaitech/testing/frames"
SHT_TRAIN_VIDEOS = e00.SHT_TRAIN_VIDEOS
ML0_WEIGHTS = e00.EXPERIMENTS_ROOT / "artifacts/EXP-020/ML0.pt"
ML0_FP32_ONNX = MODEL_DIR / "ML0-640-384x640.onnx"
ML0_INT8_ONNX = MODEL_DIR / "ML0-640-384x640-int8.onnx"
MACHINE_TAG = vmm.MACHINE_TAG
SRC = str(e00.EXPERIMENTS_ROOT / "src")

# detector caches evaluated in step 2: key -> (directory, label, runtime, weights group)
CACHE_JOBS = {  # generated here (host CPU, ORT / NCNN, 1 thread per worker, 8 workers)
    "ML0_ort_int8": (CACHE_DIR / "L0_E0-ML0-int8-640", ML0_INT8_ONNX, "ML0-int8-640"),
    "ML0_ort_fp32": (CACHE_DIR / "L0_E0-ML0-ortfp32-640", ML0_FP32_ONNX, "ML0-ortfp32-640"),
    "stock_ncnn640": (CACHE_DIR / "L0_E0-stock-ncnn16-640", VM_EXCHANGE / "models/yolov8n-pose-640-384x640-ncnn", "stock-ncnn16-640"),
}
EXISTING_CACHES = {  # produced by EXP-012 / EXP-020 and E01
    "ML0_ncnn": DATA_ROOT / "exp020-ncnn/cache/L0_E0-ML0",
    "stock_ort_int8_640": DATA_ROOT / "exp012/cache/L0_E0-int8-640",
    "stock_ort_int8_800": DATA_ROOT / "exp012/cache/L0_E0-int8-800",
    "stock_ort_fp32_640": DATA_ROOT / "exp012/cache/L0_E0-ortfp32-640",
    "stock_ncnn800": DATA_ROOT / "exp012/cache/L0_E0-ncnn16-800",
}


def plain(obj):
    return e00.to_plain(obj)


def now() -> str:
    return time.strftime("%H:%M:%S")


def sha(path: Path) -> str:
    return e00.sha256_file(path)


def sha_tree(directory: Path, pattern: str = "*") -> str:
    import hashlib

    h = hashlib.sha256()
    for f in sorted(Path(directory).glob(pattern)):
        if f.is_file():
            h.update(f.name.encode())
            h.update(sha(f).encode())
    return h.hexdigest()


# =====================================================================================================================
# 1. prep: models (calibration only), H.264 re-encode + MV cache, detector caches
# =====================================================================================================================
def stage_models(log=print) -> dict:
    """ML0 @640 (384x640) ONNX FP32 + static QDQ INT8 (200 seeded random SHT *training* frames, seed 0, `--decode fp32`, minmax). No training."""
    from edgelab.pipeline import export_onnx, quantize_int8

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if not ML0_FP32_ONNX.exists():
        log(f"[{now()}] exporting ML0 ONNX {ML0_FP32_ONNX.name}")
        export_onnx.export_static(ML0_WEIGHTS, (384, 640), ML0_FP32_ONNX, name="ML0")
    if not ML0_INT8_ONNX.exists():
        picks = quantize_int8.calibration_frames(SHT_TRAIN_VIDEOS, 200, 0)
        ML0_INT8_ONNX.with_name(ML0_INT8_ONNX.name + ".calib.txt").write_text("\n".join(f"{v} {i}" for v, i in picks) + "\n")
        frames = [quantize_int8.read_frame(SHT_TRAIN_VIDEOS, v, i) for v, i in picks]
        log(f"[{now()}] quantizing ML0 -> INT8 (200 frames, decode fp32, minmax)")
        info = quantize_int8.quantize(ML0_FP32_ONNX, ML0_INT8_ONNX, frames, "fp32", "minmax")
        (MODEL_DIR / "ML0-int8.quantize.json").write_text(json.dumps(plain(info)))
    dst = VM_E04 / "models"
    dst.mkdir(parents=True, exist_ok=True)
    for f in (ML0_FP32_ONNX, ML0_INT8_ONNX):
        if not (dst / f.name).exists() or sha(dst / f.name) != sha(f):
            (dst / f.name).write_bytes(f.read_bytes())
    return {"ml0_fp32_onnx": sha(ML0_FP32_ONNX), "ml0_int8_onnx": sha(ML0_INT8_ONNX), "ml0_weights": sha(ML0_WEIGHTS)}


def encode_jobs() -> list[tuple]:
    jobs = []
    for ds in DATASETS:
        names, _ = e00.dataset_clips(ds)
        for c in names:
            src = SHT_FRAMES_ROOT / c if ds == "ShanghaiTech" else e00.UB_TEST_VIDEOS / f"{c}.mp4"
            jobs.append((ds, c, str(src), str(H264_DIR / ds / f"{c}.mp4"), str(MV_DIR / ds / f"{c}.npz")))
    return jobs


def stage_encode(procs: int, log=print) -> dict:
    jobs = encode_jobs()
    todo = [j for j in jobs if not Path(j[4]).exists()]
    log(f"[{now()}] H.264 IPPP re-encode + MV extraction: {len(todo)}/{len(jobs)} clips ({procs} workers)")
    res = []
    if todo:  # spawn: PyAV and cv2 must not share a process on macOS (duplicate libavdevice)
        with mp.get_context("spawn").Pool(procs) as pool:
            for i, r in enumerate(pool.imap_unordered(vmm.encode_job, todo, chunksize=1)):
                res.append(r)
                if (i + 1) % 25 == 0 or i + 1 == len(todo):
                    log(f"[{now()}]   {i + 1}/{len(todo)} clips")
    # frame-count audit against the labels (every clip must have exactly one MV record per labelled frame)
    bad = []
    for ds in DATASETS:
        names, labels = e00.dataset_clips(ds)
        for c, lab in zip(names, labels):
            z = np.load(MV_DIR / ds / f"{c}.npz")
            if len(z["valid"]) != len(lab):
                bad.append((ds, c, len(z["valid"]), len(lab)))
    if bad:
        raise AssertionError(f"MV cache frame count != labels: {bad[:5]}")
    return {"encoded": len(res), "x264_SHT": vmm.x264_params("ShanghaiTech"), "x264_UB": vmm.x264_params("UBnormal")}


def _cache_shard(job: tuple) -> int:
    key, shard, n_shards = job
    out, det, label = CACHE_JOBS[key]
    names, _ = e00.dataset_clips("ShanghaiTech")
    videos = names[shard::n_shards]
    env = dict(os.environ, OMP_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1", MLFLOW_DISABLE_AGENT_HINT="1")
    cmd = [sys.executable, "-m", "edgelab.pose.detcache", "--frames-root", str(SHT_FRAMES_ROOT), "--level", "L0", "--enhance", "E0", "--out", str(out),
           "--detector", str(det), "--ncnn-fp16", "on", "--label", label, "--threads", "1", "--videos", *videos]
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(str(out) + f".shard{shard}.log", "a") as f:
        return subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env, cwd=str(e00.EXPERIMENTS_ROOT)).returncode


def stage_caches(procs: int, keys: list[str], log=print) -> dict:
    """Full 107-clip ShanghaiTech detection caches (ORT / NCNN on the host, 1 thread per worker, shards over the clips)."""
    names, _ = e00.dataset_clips("ShanghaiTech")
    out = {}
    for key in keys:
        d, det, label = CACHE_JOBS[key]
        have = sum((d / f"{c}.npz").exists() for c in names)
        if have == len(names):
            log(f"[{now()}] cache {key}: complete ({have} clips)")
            continue
        t0 = time.perf_counter()
        log(f"[{now()}] cache {key}: {len(names) - have} clips missing, {procs} shards")
        with mp.get_context("spawn").Pool(procs) as pool:
            codes = pool.map(_cache_shard, [(key, s, procs) for s in range(procs)], chunksize=1)
        have = sum((d / f"{c}.npz").exists() for c in names)
        log(f"[{now()}] cache {key}: {have}/{len(names)} clips, exit codes {codes}, {time.perf_counter() - t0:.0f}s")
        if have != len(names):
            raise RuntimeError(f"cache {key} incomplete: {have}/{len(names)}")
        out[key] = {"seconds": time.perf_counter() - t0}
    return out


def cmd_prep(procs: int, log=print) -> None:
    out = {"models": stage_models(log)}
    out["encode"] = stage_encode(procs, log)
    out["caches"] = stage_caches(procs, list(CACHE_JOBS), log)
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    (WORK_DIR / "prep.json").write_text(json.dumps(plain(out), indent=1))
    log(f"[{now()}] prep done: {out}")


# =====================================================================================================================
# 2. VM profiling (host driver; workers live in q2_p1_e04_vm.py and run inside the edge-os VM)
# =====================================================================================================================
THREADS = (1, 2, 3, 4)
POSE_MODEL_IDS = ("ML0-640-ncnn", "ML0-640-ort-fp32", "ML0-640-ort-int8", "stock-640-ncnn", "stock-640-ort-int8", "stock-800-ncnn", "stock-800-ort-int8")


def vm_stage(log=print) -> None:
    """Copy the profiling clips' H.264 files into the exchange folder the VM reads."""
    import shutil

    for kind, ds in (("sht", "ShanghaiTech"), ("ub", "UBnormal")):
        dst = VM_E04 / "h264" / kind
        dst.mkdir(parents=True, exist_ok=True)
        for c in vmm.PROFILE_CLIPS[kind]:
            if not (dst / f"{c}.mp4").exists():
                shutil.copy(H264_DIR / ds / f"{c}.mp4", dst / f"{c}.mp4")
    VM_OUT.mkdir(parents=True, exist_ok=True)


def vm_cells() -> list[dict]:
    cells = []
    for kind in ("sht", "ub"):
        for t in THREADS:
            cells.append({"id": f"mvprof_{kind}_T{t}", "args": ["mvprof", "--kind", kind, "--threads", str(t)], "threads": t})
    for m in POSE_MODEL_IDS:
        for t in THREADS:
            cells.append({"id": f"pose_{m}_T{t}", "args": ["pose", "--model", m, "--threads", str(t), "--step", "2"], "threads": t})
    return cells


def vm_drive(limit: float = 30.0, max_tries: int = 4, only: list[str] | None = None, log=print) -> list[dict]:
    from edgelab.analysis import q2_p1_e02_vm as vm02
    from edgelab.pipeline.host_load import LoadSampler, find_vm_pid, summarise
    from edgelab.pipeline.timing_exp010 import wait_for_quiet_host

    vm_stage(log)
    manifest = VM_OUT / "manifest.jsonl"
    done = set()
    if manifest.exists():
        for line in manifest.read_text().splitlines():
            row = json.loads(line)
            if row["status"] == "ok" and not row["excluded"]:
                done.add(row["cell"])
    vm_pid = find_vm_pid()
    results = []
    for cell in vm_cells():
        if cell["id"] in done or (only and not any(o in cell["id"] for o in only)):
            continue
        for attempt in range(max_tries):
            quiet = wait_for_quiet_host(vm_pid, limit, max_wait_s=900)
            sampler = LoadSampler(vm_pid).start()
            t0 = time.time()
            prefix = VM_OUT / cell["id"]
            r = vm02.run_vm(["python", "-m", "edgelab.analysis.q2_p1_e04_vm", *cell["args"], "--out", str(prefix)], cell["threads"])
            load = summarise(sampler.stop())
            ok = r.returncode == 0 and Path(f"{prefix}.npz").exists()
            outside = load["host_outside_vm_pct"]
            row = {"cell": cell["id"], "attempt": attempt, "status": "ok" if ok else "failed", "seconds": round(time.time() - t0, 1),
                   "pre_run_host_outside_vm_pct": quiet, **load, "excluded": (outside is None or outside > limit),
                   "error": None if ok else (r.stderr[-600:] + r.stdout[-200:]), "start": time.strftime("%Y-%m-%d %H:%M:%S")}
            with open(manifest, "a") as f:
                f.write(json.dumps(row) + "\n")
            results.append(row)
            log(f"  [{now()}] {cell['id']} try{attempt} {row['status']} {row['seconds']}s host_outside_vm={outside if outside is None else round(outside, 1)}%"
                f"{' EXCLUDED' if row['excluded'] else ''}{'' if ok else ' ' + str(row['error'])[-200:]}")
            if ok and not row["excluded"]:
                break
    return results


def load_cell(cell_id: str) -> tuple[dict, dict] | None:
    prefix = VM_OUT / cell_id
    if not Path(f"{prefix}.npz").exists() or not Path(f"{prefix}.json").exists():
        return None
    z = np.load(f"{prefix}.npz")
    return {k: z[k] for k in z.files}, json.loads(Path(f"{prefix}.json").read_text())


# =====================================================================================================================
# 3. MV statistics (pure), exact run tables (MV / advection / INT8), thresholds
# =====================================================================================================================
E03_DIR = e03.EXP_DIR
PRIMARY = e03.PRIMARY
TRACKER = e03.TRACKER
GATE_FLOOR = e03.GATE_FLOOR
A1 = e03.A1
LEVELS = e03.LEVELS
CELLS = e03.CELLS
_G: dict = {}


def mv_path(ds: str, clip: str) -> Path:
    return MV_DIR / ds / f"{clip}.npz"


def tb_path(tag: str, ds: str) -> Path:
    return WORK_DIR / f"tables_{tag}_{ds}.npz"


def mvstat_path(tag: str, ds: str) -> Path:
    return WORK_DIR / f"mvstat_{tag}_{ds}.npz"


def run_path(ds: str) -> Path:
    return WORK_DIR / f"run_{ds}.pkl"


def load_cfg03() -> dict:
    return yaml.safe_load((E03_DIR / "config.yaml").read_text())


def load_config() -> dict:
    return yaml.safe_load((EXP_DIR / "config.yaml").read_text())


def mv_static_stats(mag: np.ndarray, cov: np.ndarray, valid: np.ndarray, boxes_by_frame: list, width: float, height: float, a1: int = A1
                    ) -> tuple[np.ndarray, np.ndarray]:
    """MV analogue of E03 `static_stats`. For source frame s and age k = 1..a1 the *accumulated* magnitude sum_{j=s+1..s+k} m_j (intra
    macroblocks = INTRA_MAG) on the macroblock grid: mean inside the person boxes of s (dilated one macroblock) and 95th percentile outside.
    NaN when s + k does not exist or any frame in (s, s + k] is invalid (a key frame carries no vectors), so `NaN <= tau` is False and the detector runs."""
    n, gh, gw = mag.shape
    m = np.zeros((n, gh, gw), dtype=np.float32)
    for j in range(n):
        if valid[j]:
            m[j] = vmm.magnitude_grid(mag[j].astype(np.float32), cov[j].astype(np.float32) / 255.0)
    roi = np.full((n, a1), np.nan, dtype=np.float32)
    ext = np.full((n, a1), np.nan, dtype=np.float32)
    empty = np.zeros((gh, gw), dtype=bool)
    for s in range(n):
        b = boxes_by_frame[s]
        mask = vmm.box_cells_grid(b, width, height, gh, gw, 1) if len(b) else empty
        acc = np.zeros((gh, gw), dtype=np.float32)
        for k in range(1, a1 + 1):
            j = s + k
            if j >= n or not valid[j]:
                break
            acc += m[j]
            roi[s, k - 1], ext[s, k - 1] = vmm.roi_check(acc, mask)
    return roi, ext


def mv_max_roi_p90(mag: np.ndarray, cov: np.ndarray, valid: np.ndarray, boxes_by_frame: list, width: float, height: float) -> np.ndarray:
    """`mv_max_roi_p90` (source plan name): per frame the maximum over the person boxes of the 90th percentile of the macroblock magnitude in the box (NaN: invalid frame / no box)."""
    n, gh, gw = mag.shape
    out = np.full(n, np.nan, dtype=np.float32)
    for j in range(n):
        b = boxes_by_frame[j]
        if not valid[j] or len(b) == 0:
            continue
        m = vmm.magnitude_grid(mag[j].astype(np.float32), cov[j].astype(np.float32) / 255.0)
        best = []
        for box in np.asarray(b, dtype=float).reshape(-1, 4):
            mask = vmm.box_cells_grid(box[None], width, height, gh, gw, 0)
            if mask.any():
                best.append(float(np.quantile(m[mask], 0.90)))
        if best:
            out[j] = max(best)
    return out


def carry_forward(x: np.ndarray) -> np.ndarray:
    """Replace NaN (key frames) by the previous value of the same clip; a leading NaN (frame 0) stays NaN (the schedulers treat t = 0 as a cut)."""
    out = x.astype(np.float64).copy()
    for i in range(1, len(out)):
        if not np.isfinite(out[i]):
            out[i] = out[i - 1]
    return out


def mv_constants(motion: np.ndarray, border: np.ndarray, dev_frames: np.ndarray) -> dict:
    """E03 `derive_constants` motion / border quantiles for the MV observer (test_dev, label-free)."""
    mot = motion[dev_frames]
    mot = mot[np.isfinite(mot)]
    bor = border[dev_frames]
    bor = bor[np.isfinite(bor)]
    q = lambda x, p: float(np.quantile(x, p))  # noqa: E731
    return {"motion_q": {"q50": q(mot, 0.5), "q70": q(mot, 0.7), "q85": q(mot, 0.85), "q90": q(mot, 0.9)}, "border_q": {"q50": q(bor, 0.5), "q90": q(bor, 0.9)}}


def adv_shift(pose: np.ndarray, adv: tuple, s: int, j: int) -> tuple[float, float]:
    """Accumulated mean MV displacement (px) inside the source person box (keypoint extent of `pose`, macroblock cells it touches) from frame s to frame j."""
    cdx, cdy, (gh, gw), (width, height) = adv
    ok = pose[:, 2] > 0.1
    pts = pose[ok] if ok.any() else pose
    box = np.array([[pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max()]])
    mask = vmm.box_cells_grid(box, width, height, gh, gw, 0)
    if not mask.any():
        return 0.0, 0.0
    return float((cdx[j] - cdx[s])[mask].mean()), float((cdy[j] - cdy[s])[mask].mean())


def _mvstat_clip(ci: int):
    ds, names, frames = _G["ds"], _G["names"], _G["frames"]
    z = np.load(mv_path(ds, names[ci]))
    width, height = (int(v) for v in z["size"])
    boxes = [f["boxes"][f["conf"] >= 0.10] for f in frames[ci]]
    roi, ext = mv_static_stats(z["mag"], z["cov"], z["valid"], boxes, width, height)
    mrp = mv_max_roi_p90(z["mag"], z["cov"], z["valid"], boxes, width, height)
    return roi, ext, mrp


def _pixstat_clip(ci: int):
    frames, motion = _G["frames"][ci], _G["motion"]
    width, height = motion["size"]
    boxes = [f["boxes"][f["conf"] >= 0.10] for f in frames]
    return e03.static_stats(motion["thumb"][ci], boxes, width, height)


def dev_frame_index(ctx, ds: str) -> np.ndarray:
    n_frames = [len(x) for x in ctx.labels[ds]]
    offs = np.concatenate([[0], np.cumsum(n_frames)])
    return np.concatenate([np.arange(offs[i], offs[i + 1]) for i in ctx.dev_idx[ds]])


def stage_mvstat(ds: str, ctx, tag: str, frames_by_clip: list, procs: int, log=print, thr: dict | None = None) -> dict:
    """MV C1 statistics (roi / ext, n_total x A1) + per-frame MV statistics of every clip. Thresholds: the 10 / 25 / 50 % quantiles of the one-frame
    test_dev statistics (label-free) unless `thr` is given (INT8 detections reuse the locked thresholds)."""
    names = ctx.names[ds]
    _G.update(ds=ds, names=names, frames=frames_by_clip)
    try:
        res = e01.pmap(_mvstat_clip, list(range(len(names))), procs)
    finally:
        _G.clear()
    roi = np.concatenate([r for r, _, _ in res]).astype(np.float32)
    ext = np.concatenate([x for _, x, _ in res]).astype(np.float32)
    mrp = np.concatenate([m for _, _, m in res]).astype(np.float32)
    per = {k: np.concatenate([np.load(mv_path(ds, c))[k] for c in names]).astype(np.float32) for k in ("mv_p90", "mv_border_p90", "mv_motion", "mv_border")}
    dev = dev_frame_index(ctx, ds)
    thr = thr or e03.thresholds_from_stats(roi[dev, 0], ext[dev, 0])
    motion = np.concatenate([carry_forward(np.load(mv_path(ds, c))["mv_motion"]) for c in names])
    border = np.concatenate([carry_forward(np.load(mv_path(ds, c))["mv_border"]) for c in names])
    consts = mv_constants(motion, border, dev)
    np.savez_compressed(mvstat_path(tag, ds), roi=roi, ext=ext, mv_max_roi_p90=mrp, motion_cf=motion.astype(np.float32), border_cf=border.astype(np.float32),
                        thr=np.array(json.dumps(thr)), consts=np.array(json.dumps(consts)), **per)
    log(f"[{now()}] mv statistics {ds}/{tag}: thresholds {json.dumps(plain(thr))}")
    return load_mvstat(tag, ds)


def load_mvstat(tag: str, ds: str) -> dict:
    z = np.load(mvstat_path(tag, ds))
    out = {k: z[k] for k in z.files}
    out["thr"], out["consts"] = json.loads(str(out["thr"])), json.loads(str(out["consts"]))
    return out


def stage_pixstat(ds: str, ctx, frames_by_clip: list, procs: int) -> tuple[np.ndarray, np.ndarray]:
    names = ctx.names[ds]
    _G.update(frames=frames_by_clip, motion=e03.load_motion(ds, names))
    try:
        res = e01.pmap(_pixstat_clip, list(range(len(names))), procs)
    finally:
        _G.clear()
    return np.concatenate([r for r, _ in res]).astype(np.float32), np.concatenate([x for _, x in res]).astype(np.float32)


def _tclip(ci: int) -> dict:
    """E03 `_table_clip` (identical logic, so that the replay of E03 is exact) with (i) arbitrary C1 statistics / thresholds and (ii) optional pose advection."""
    G = _G
    ds = G["ds"]
    L, delta, ext, rmax = SEG_LENS[ds], e03.DELTA[ds], e03.EXT[ds], e03.RMAX[ds]
    frames, ids = G["frames"][ci], G["ids"][ci]
    roi, ext_stat = G["roi"][ci], G["ext"][ci]
    thr = G["thr"]
    n = len(frames)
    adv = None
    if G["advect"]:
        z = np.load(mv_path(ds, G["names"][ci]))
        dx, dy = z["dx"].astype(np.float32), z["dy"].astype(np.float32)
        dx[~z["valid"]] = 0.0
        dy[~z["valid"]] = 0.0
        adv = (np.cumsum(dx, axis=0), np.cumsum(dy, axis=0), dx.shape[1:], tuple(int(v) for v in z["size"]))
    pool: list[np.ndarray] = []
    index: dict[int, int] = {}
    heads: list[int] = []
    tails: list[int] = []
    rows_c, rows_e, rows_uid = [], [], []
    starts = list(range(0, n - L + 1, e03.START_GRID))
    nlev = len(LEVELS)
    bits = np.zeros((len(starts), nlev), dtype=np.uint64)
    hist = np.zeros((nlev, 60), dtype=np.int64)
    n_reused_pose = np.zeros(nlev, dtype=np.int64)
    n_unseen = np.zeros(nlev, dtype=np.int64)

    def intern(arr: np.ndarray) -> int:
        key = e03._hash64(arr.tobytes())
        got = index.get(key)
        if got is None:
            got = index[key] = len(pool)
            pool.append(arr)
            heads.append(e03._hash64(arr[:-1].tobytes()))
            tails.append(e03._hash64(arr[1:].tobytes()))
        return got

    for si, c in enumerate(starts):
        end = min(n, c + rmax)
        tracks = e01.replay_cell(frames[c:end], ids[c:end], TRACKER)
        nrun = end - c
        pats = [e03.reuse_pattern(roi, ext_stat, c, nrun, thr["roi"][li], thr["ext"][li]) for li in range(nlev)]
        for li, (re_mask, _) in enumerate(pats):
            bits[si, li] = np.uint64(e03.pack_bits(re_mask))
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
                        s_abs = int(c + src[offs[j]])
                        i_src = fidx.get(s_abs)
                        if i_src is None:  # person not seen at the source frame: validity miss (keeps the genuine pose, counted)
                            n_unseen[li] += 1
                            continue
                        arr2[j] = kp[i_src]
                        if adv is not None:
                            sx, sy = adv_shift(kp[i_src], adv, s_abs, int(c + offs[j]))
                            arr2[j, :, 0] += sx
                            arr2[j, :, 1] += sy
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
    for i, arr in enumerate(pool):
        meta_pose[i], meta_vel[i], qual[i] = e03.window_meta(arr)
    order = np.lexsort((np.array(rows_e, dtype=np.int64), np.array(rows_c, dtype=np.int64))) if rows_c else np.zeros(0, dtype=np.int64)
    return {"pool": pool_arr, "pose": meta_pose.astype(np.float32), "vel": meta_vel.astype(np.float32), "qual": qual.astype(np.float32),
            "head": np.array(heads, dtype=np.uint64), "tail": np.array(tails, dtype=np.uint64), "starts": np.array(starts, dtype=np.int32),
            "row_c": np.array(rows_c, dtype=np.int32)[order], "row_e": np.array(rows_e, dtype=np.int8)[order],
            "row_uid": (np.array(rows_uid, dtype=np.int32).reshape(-1, 1 + nlev) if rows_c else np.zeros((0, 1 + nlev), np.int32))[order],
            "bits": bits, "hist": hist, "n_reused_pose": n_reused_pose, "n_unseen": n_unseen}


def build_tables(ds: str, ctx, tag: str, frames_by_clip: list, ids_by_clip: list, roi_all: np.ndarray, ext_all: np.ndarray, thr: dict, advect: bool,
                 procs: int, log=print, force: bool = False) -> Path:
    """E03 `stage_tables` with the C1 statistics, thresholds and (optionally) advection of this variant. Saves `tables_{tag}_{ds}.npz`."""
    path = tb_path(tag, ds)
    if path.exists() and not force:
        log(f"[{now()}] tables {tag}/{ds}: cached")
        return path
    names, labels = ctx.names[ds], ctx.labels[ds]
    scorer = ctx.scorers[ds]
    for ids in ids_by_clip:
        assert np.array_equal(ids, np.arange(len(ids))), "detector cache frame ids must be 0..n-1"
    n_frames = np.array([len(x) for x in labels])
    offs = np.concatenate([[0], np.cumsum(n_frames)])
    n_in = np.concatenate([[int((f["conf"] >= TRACKER.gate).sum()) for f in fr] for fr in frames_by_clip]).astype(np.int16)
    assert len(n_in) == int(offs[-1]) == len(roi_all)
    roi = [roi_all[offs[i]:offs[i + 1]] for i in range(len(names))]
    ext = [ext_all[offs[i]:offs[i + 1]] for i in range(len(names))]
    _G.update(ds=ds, names=names, frames=frames_by_clip, ids=ids_by_clip, roi=roi, ext=ext, thr=thr, advect=advect)
    t0 = time.perf_counter()
    cols: dict[str, list] = {k: [] for k in ("row_norm", "row_lin", "row_step", "row_qual", "row_e", "row_start", "clip_of_start", "start_frame", "bits")}
    hist = np.zeros((len(LEVELS), 60), dtype=np.int64)
    n_reused = np.zeros(len(LEVELS), dtype=np.int64)
    n_unseen = np.zeros(len(LEVELS), dtype=np.int64)
    n_pool = lin_off = start_off = 0
    try:
        with mp.get_context("fork").Pool(procs) as pool:
            for ci, res in enumerate(pool.imap(_tclip, range(len(names)), chunksize=1)):
                norm = scorer.score(res["pool"].astype(np.float64)) if len(res["pool"]) else np.zeros(0)
                lin, step = e03.build_lineage(res)
                uid = res["row_uid"]
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
                if (ci + 1) % 20 == 0 or ci + 1 == len(names):
                    log(f"[{now()}]   tables {tag}/{ds}: {ci + 1}/{len(names)} clips, {n_pool} unique windows, {time.perf_counter() - t0:.0f}s")
    finally:
        _G.clear()
    out = {k: np.concatenate(v) for k, v in cols.items()}
    out["row_norm"] = out["row_norm"].reshape(-1, 1 + len(LEVELS))
    counts = np.zeros((start_off, e03.EXT[ds] + 1), dtype=np.int64)
    np.add.at(counts, (out["row_start"], out["row_e"].astype(np.int64)), 1)
    out["row_ptr"] = np.concatenate([[0], np.cumsum(counts.reshape(-1))])
    out["hist"], out["n_reused_pose"], out["n_unseen"], out["n_pool"], out["n_in"] = hist, n_reused, n_unseen, np.array(n_pool), n_in
    out["meta"] = np.array(json.dumps({"ds": ds, "tag": tag, "L": SEG_LENS[ds], "delta": e03.DELTA[ds], "ext": e03.EXT[ds], "rmax": e03.RMAX[ds], "thr": thr,
                                       "advect": advect, "tracker": TRACKER.label, "build_s": time.perf_counter() - t0}))
    np.savez_compressed(path, **out)
    log(f"[{now()}] tables {tag}/{ds}: {len(out['row_norm'])} rows, {n_pool} unique windows, {start_off} starts in {time.perf_counter() - t0:.0f}s")
    return path


def load_tb(tag: str, ds: str) -> dict:
    z = np.load(tb_path(tag, ds))
    out = {k: z[k] for k in z.files}
    out["meta"] = json.loads(str(out["meta"]))
    return out


TB_KEYS = ("row_norm", "row_lin", "row_step", "row_qual", "row_e", "row_start", "start_frame", "clip_of_start", "bits", "row_ptr", "hist", "n_reused_pose",
           "n_unseen", "n_in")


def compare_tables(a: dict, b: dict) -> dict:
    """Exactness of a rebuilt table set against the E03 one (rule E4-D1, table level)."""
    out = {}
    for k in TB_KEYS:
        x, y = np.asarray(a[k]), np.asarray(b[k])
        if x.shape != y.shape:
            out[k] = {"equal": False, "shape": [list(x.shape), list(y.shape)]}
            continue
        eq = bool(np.array_equal(x, y, equal_nan=True)) if x.dtype.kind == "f" else bool(np.array_equal(x, y))
        d = float(np.nanmax(np.abs(x.astype(np.float64) - y.astype(np.float64)))) if (x.size and x.dtype.kind in "fiu") else 0.0
        out[k] = {"equal": eq, "max_abs_diff": d}
    out["all_equal"] = all(v["equal"] for v in out.values())
    return out


def int8_frames(ctx, ds: str, key: str) -> tuple[list, list]:
    return e01.load_frames_floor(CACHE_JOBS[key][0], ctx.names[ds], GATE_FLOOR)


def stage_tables(ctx, procs: int, log=print, force: bool = False) -> dict:
    arms = e01.arm_by_key(ctx.arms)
    cfg03 = load_cfg03()
    report: dict = {}
    for ds in DATASETS:
        names = ctx.names[ds]
        arm = arms[f"{ds}/{PRIMARY[ds]}"]
        frames, ids = e01.load_frames_floor(arm.cache_dir, names, GATE_FLOOR)
        thr03 = cfg03["datasets"][ds]["constants"]["c1"]
        z03 = np.load(e03.static_path(ds))
        # (a) E03 replay: the rebuilt tables must equal the E03 tables
        build_tables(ds, ctx, "pixrebuild", frames, ids, z03["roi"], z03["ext"], thr03, False, procs, log, force)
        report[ds] = {"pixrebuild_vs_e03": compare_tables(load_tb("pixrebuild", ds), e03.load_tables_npz(ds))}
        log(f"[{now()}] tables {ds}: rebuild vs E03 all_equal={report[ds]['pixrebuild_vs_e03']['all_equal']}")
        # (b) +MV: statistics + thresholds from test_dev one-frame statistics, zero-order hold and advection
        if not mvstat_path("mv", ds).exists() or force:
            stage_mvstat(ds, ctx, "mv", frames, procs, log)
        ms = load_mvstat("mv", ds)
        build_tables(ds, ctx, "mv", frames, ids, ms["roi"], ms["ext"], ms["thr"], False, procs, log, force)
        build_tables(ds, ctx, "mvadv", frames, ids, ms["roi"], ms["ext"], ms["thr"], True, procs, log, force)
        # (c) +INT8 (ShanghaiTech only: the UBnormal frontend is the MPS proxy ML0_mps and no UB-geometry INT8 cache exists)
        if ds == "ShanghaiTech":
            f8, i8 = int8_frames(ctx, ds, "ML0_ort_int8")
            roi8, ext8 = stage_pixstat(ds, ctx, f8, procs)
            build_tables(ds, ctx, "int8", f8, i8, roi8, ext8, thr03, False, procs, log, force)
            if not mvstat_path("int8mv", ds).exists() or force:
                stage_mvstat(ds, ctx, "int8mv", f8, procs, log, thr=ms["thr"])
            ms8 = load_mvstat("int8mv", ds)
            build_tables(ds, ctx, "int8mv", f8, i8, ms8["roi"], ms8["ext"], ms["thr"], False, procs, log, force)
    (WORK_DIR / "tables_check.json").write_text(json.dumps(plain(report), indent=1))
    return report


# =====================================================================================================================
# 4. VM costs -> lock (config.yaml, written before any delivered-AUC / quality / decision number exists)
# =====================================================================================================================
KIND_OF = {"ShanghaiTech": "sht", "UBnormal": "ub"}
MV_MARGIN = 1.25
MV_SENS_MARGINS = (1.0, 2.0)


def vm_cost_table() -> dict:
    """Mean wall / CPU (ms per frame) of every profiled ingredient, per clip kind and thread count (NaN-aware)."""
    out: dict = {}
    for kind in ("sht", "ub"):
        row = {}
        for t in THREADS:
            c = load_cell(f"mvprof_{kind}_T{t}")
            if c is None:
                continue
            a, m = c
            row[f"T{t}"] = {meth: {"wall_ms": float(np.nanmean(a[f"{meth}_w"])), "cpu_ms": float(np.nanmean(a[f"{meth}_c"])),
                                   "p95_wall_ms": float(np.nanpercentile(a[f"{meth}_w"], 95)), "n": int(np.isfinite(a[f"{meth}_w"]).sum())} for meth in vmm.METHODS}
            row[f"T{t}"]["_frames"] = int(m["frames"])
            row[f"T{t}"]["_ptype_I_fraction"] = float((a["ptype"] == 1).mean()) if len(a["ptype"]) else float("nan")
        out[kind] = row
    return out


def derive_vm_costs(table: dict, kind: str, margin: float = MV_MARGIN) -> dict:
    """Simulation constants (VM milliseconds, before the Pi multiplier s). Export overhead = decode(MV) - decode(plain) at T = 4 charged on EVERY frame; the
    numpy statistics (single thread) at T = 1. `margin` multiplies every MV cost; the pixel counterparts are reported unscaled."""
    t4, t1 = table[kind]["T4"], table[kind]["T1"]
    both = lambda d: max(d["wall_ms"], d["cpu_ms"])  # noqa: E731  (the simulator charges one number as wall and CPU)
    return {"export_wall_ms": margin * max(t4["decode_mv"]["wall_ms"] - t4["decode_plain"]["wall_ms"], 0.0),
            "export_cpu_ms": margin * max(t4["decode_mv"]["cpu_ms"] - t4["decode_plain"]["cpu_ms"], 0.0),
            "mv_obs_ms": margin * both(t1["mv_obs"]), "mv_c1_ms": margin * both(t1["mv_c1"]),
            "pix_obs_ms_vm": both(t1["pix_obs"]), "pix_c1_ms_vm": both(t1["pix_c1"]), "margin": margin}


def pose_drift() -> dict:
    """E4-D4: the E04-session NCNN FP16 pose pool of ML0 against the E02 pool (same model / geometry, different VM session)."""
    out = {"threads": {}}
    for t in THREADS:
        a = load_cell(f"pose_ML0-640-ncnn_T{t}")
        e = e02.pose_arrays("ML0-640", t)
        if a is None or e is None:
            continue
        a = a[0]
        out["threads"][f"T{t}"] = {"e04_wall_ms": float(a["wall_ms"].mean()), "e02_wall_ms": float(e["wall_ms"].mean()), "e04_cpu_ms": float(a["cpu_ms"].mean()),
                                   "e02_cpu_ms": float(e["cpu_ms"].mean()), "wall_ratio_e02_over_e04": float(e["wall_ms"].mean() / a["wall_ms"].mean()),
                                   "cpu_ratio_e02_over_e04": float(e["cpu_ms"].mean() / a["cpu_ms"].mean())}
    out["max_abs_wall_deviation_pct"] = max(abs(100 * (v["wall_ratio_e02_over_e04"] - 1)) for v in out["threads"].values())
    out["pass"] = bool(out["max_abs_wall_deviation_pct"] <= 10.0)
    return out


def pose_pool_summary() -> dict:
    out = {}
    for m in POSE_MODEL_IDS:
        for t in THREADS:
            c = load_cell(f"pose_{m}_T{t}")
            if c is None:
                continue
            a, meta = c
            out[f"{m}_T{t}"] = {"wall_ms": float(a["wall_ms"].mean()), "cpu_ms": float(a["cpu_ms"].mean()), "p95_wall_ms": float(np.percentile(a["wall_ms"], 95)),
                                "infer_ms": float(a["infer_ms"].mean()), "frames": int(len(a["wall_ms"])), "peak_rss_mb": float(meta["peak_rss_mb"])}
    return out


def stage_lock(log=print) -> dict:
    cfg03 = load_cfg03()
    e03sum = e03.EXP_DIR / "summary.json"
    table = vm_cost_table()
    manifest = e02.load_manifest(VM_OUT)
    loads = [r["host_outside_vm_pct"] for r in manifest.values() if r.get("host_outside_vm_pct") is not None]
    drift = pose_drift()
    cfg = {
        "experiment": "Q2-P1-E04", "title": "Optional optimizations (+MV, then +INT8)", "training": False, "seed": SEED, "pool_seed": e03.POOL_SEED,
        "bootstrap": {"unit": "video", "resamples": N_BOOT, "seed": 0},
        "hardware_disclosure": {"physical_pi5": "NOT connected; every CPU cost is an edge-os VM measurement x the E02/E03 Pi multiplier s; delivered AUC is simulated delivery on replayed detections",
                                "machine_tag": MACHINE_TAG, "isa_note": "the VM exposes asimddp, i8mm, bf16; Cortex-A76 (Pi 5) has dot-product only: INT8 speed-ups are an upper bound",
                                "mv_note": "MV streams are our own x264 IPPP re-encode of the dataset frames (UBnormal: second generation), not camera streams"},
        "builds_on": {"E00": {"summary_sha256": sha(e00.EXP_DIR / "summary.json")}, "E01": {"summary_sha256": sha(e01.EXP_DIR / "summary.json")},
                      "E02": {"summary_sha256": sha(e02.EXP_DIR / "summary.json")},
                      "E03": {"config_sha256": sha(E03_DIR / "config.yaml"), "summary_sha256": sha(e03sum), "run_sha256": {ds: sha(e03.run_path(ds)) for ds in DATASETS}}},
        "plan_sha256": sha(EXP_DIR / "plan.md"),
        "tracker": TRACKER.label, "frontends": dict(PRIMARY), "cells": [list(c) for c in CELLS], "c1_level": 3, "A1": A1,
        "e03_locked": {ds: {"Cache-C1": cfg03["datasets"][ds]["tuning"]["selected"]["Cache-C1"], "Motion-Only": cfg03["datasets"][ds]["tuning"]["selected"]["Motion-Only"],
                            "c1_pixel_thresholds": cfg03["datasets"][ds]["constants"]["c1"], "alert_theta": cfg03["datasets"][ds]["constants"]["theta"]} for ds in DATASETS},
        "inputs": {"models": {"ml0_weights": sha(ML0_WEIGHTS), "ml0_fp32_onnx": sha(ML0_FP32_ONNX), "ml0_int8_onnx": sha(ML0_INT8_ONNX),
                              "int8_calibration_list": sha(ML0_INT8_ONNX.with_name(ML0_INT8_ONNX.name + ".calib.txt")),
                              "int8_quantize_info": json.loads((MODEL_DIR / "ML0-int8.quantize.json").read_text())},
                   "mv_cache_tree_sha256": {ds: sha_tree(MV_DIR / ds, "*.npz") for ds in DATASETS},
                   "h264_tree_sha256": {ds: sha_tree(H264_DIR / ds, "*.mp4") for ds in DATASETS},
                   "detector_caches": {k: {"dir": str(v[0]), "clips": len(list(v[0].glob("*.npz"))), "tree_sha256": sha_tree(v[0], "*.npz")} for k, v in CACHE_JOBS.items()},
                   "x264": {ds: vmm.x264_params(ds) for ds in DATASETS}, "x264_base": vmm.X264_BASE, "mv_semantics": {"intra_mag_px": vmm.INTRA_MAG, "active_px": vmm.ACTIVE_PX,
                                                                                                                 "border_band": vmm.BORDER_BAND, "keyint_s": vmm.KEYINT_S}},
        "mv": {ds: {"c1_thresholds_test_dev": load_mvstat("mv", ds)["thr"], "observer_constants_test_dev": load_mvstat("mv", ds)["consts"]} for ds in DATASETS},
        "vm": {"cost_table": table, "derived_costs": {ds: derive_vm_costs(table, KIND_OF[ds]) for ds in DATASETS},
               "sensitivity_margins": list(MV_SENS_MARGINS), "pose_pools": pose_pool_summary(), "pose_drift_vs_e02": drift,
               "pose_pool_correction": "ML0 ORT INT8 pool x (E02 NCNN pool mean / E04-session NCNN pool mean) per thread count (wall and CPU separately); uncorrected is a sensitivity arm",
               "host_outside_vm_pct": {"mean": float(np.mean(loads)), "max": float(np.max(loads)), "limit": 30.0},
               "cells": len(manifest), "excluded": sorted(c for c, r in manifest.items() if r.get("excluded")),
               "manifest": {c: {k: r.get(k) for k in ("attempt", "status", "seconds", "host_outside_vm_pct", "excluded")} for c, r in sorted(manifest.items())}},
        "tables_check": json.loads((WORK_DIR / "tables_check.json").read_text()),
        "accounting": {"mv_margin": MV_MARGIN, "export_overhead": "decode_mv - decode_plain at T = 4, every frame", "retrieve_skip_credit": True,
                       "pixel_observer_and_thumbnail": "not charged in MV arms", "matched_cost": "E03 absolute target beta * C_full(s) (pixel pipeline) in every cell",
                       "path_b": "schedule period (or controller knob) of the base arm kept; CPU and AUC read directly"},
        "adoption_rule": {"path_a": "cell-averaged delivered-AUC gain at matched absolute CPU > 0 on test_dev with 95% paired video-bootstrap CI excluding 0",
                          "path_b": "total CPU at the identical schedule falls by >= 5 % (cell average)", "degradation_pt": 0.25,
                          "confirmation": "same direction on test_confirmation (A: point estimate >= 0; B: reduction >= 5 %), degradation <= 0.25 pt", "reproducible": "rules E4-D1..D7 pass",
                          "int8_base": "+MV if +MV is adopted for the dataset, else B0 (both reported)", "int8_scope": "ShanghaiTech only (no UB-geometry INT8 cache)"},
        "environment": e00.env_info(),
    }
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    header = ("# Q2-P1-E04 locked configuration (generated by `q2_p1_e04 lock` AFTER `vm` and `tables`, BEFORE `run`; no delivered AUC, recall or decision existed).\n"
              "# Contains the input hashes, VM-measured costs, MV thresholds / observer constants from test_dev and the adoption rule.\n")
    (EXP_DIR / "config.yaml").write_text(header + yaml.safe_dump(plain(cfg), sort_keys=False, width=140))
    log(f"[{now()}] lock: wrote {EXP_DIR / 'config.yaml'} (pose drift max {drift['max_abs_wall_deviation_pct']:.1f}%, host max {np.max(loads):.1f}%)")
    return cfg


# =====================================================================================================================
# 5. Run: matched-cost cells (E03 absolute targets) for every arm
# =====================================================================================================================
POSE_MODEL_IDS_E04 = ("ML0-640-ncnn", "ML0-640-ort-fp32", "ML0-640-ort-int8")
DELIVERED_ARMS_NOTE = ("C1-Pixel", "C1-MV", "C1-MV@W", "C1-MV-Advect", "Motion-Only-Pixel", "Motion-Only+MV", "C1-INT8", "C1-INT8@W", "C1-MV-INT8", "C1-MV-INT8@W")


def k_pixel(cfg03: dict, ds: str) -> dict:
    return copy.deepcopy(cfg03["datasets"][ds]["constants"])


def k_mv(cfg03: dict, cfg: dict, ds: str, margin: float = MV_MARGIN, tag: str = "mv") -> dict:
    """E03 constants with the MV accounting: export overhead on every frame, MV observer / C1 costs, retrieve-skip credit, MV observer quantiles."""
    K = k_pixel(cfg03, ds)
    d = derive_vm_costs(cfg["vm"]["cost_table"], KIND_OF[ds], margin)
    K["costs"] = {"obs_ms": d["mv_obs_ms"], "c1_check_ms": d["mv_c1_ms"], "c2_check_ms": e03.C2_CHECK_MS}
    K["mv"] = {"export_wall_ms": d["export_wall_ms"], "export_cpu_ms": d["export_cpu_ms"], "skip_retrieve": True}
    K["motion_q"], K["border_q"] = cfg["mv"][ds]["observer_constants_test_dev"]["motion_q"], cfg["mv"][ds]["observer_constants_test_dev"]["border_q"]
    return K


def k_pixel_vm(cfg03: dict, cfg: dict, ds: str) -> dict:
    K = k_pixel(cfg03, ds)
    d = derive_vm_costs(cfg["vm"]["cost_table"], KIND_OF[ds], 1.0)
    K["costs"] = {"obs_ms": d["pix_obs_ms_vm"], "c1_check_ms": d["pix_c1_ms_vm"], "c2_check_ms": e03.C2_CHECK_MS}
    return K


def int8_pools(base_pools: dict, cfg: dict, corrected: bool = True) -> dict:
    """E02 service pools with the pose stage replaced by the ML0 ORT INT8 pool of this session (drift-corrected to the E02 NCNN pool by default)."""
    pools = dict(base_pools)
    drift = cfg["vm"]["pose_drift_vs_e02"]["threads"]
    for t in THREADS:
        a = load_cell(f"pose_ML0-640-ort-int8_T{t}")[0]
        fw = drift[f"T{t}"]["wall_ratio_e02_over_e04"] if corrected else 1.0
        fc = drift[f"T{t}"]["cpu_ratio_e02_over_e04"] if corrected else 1.0
        pools[("pose", t)] = (a["wall_ms"].astype(float) * fw, a["cpu_ms"].astype(float) * fc)
    return pools


def make_world(ctx, ds: str, tb: dict, pools: dict, K: dict, mv_obs: dict | None = None) -> e03.World:
    arm = e01.arm_by_key(ctx.arms)[f"{ds}/{PRIMARY[ds]}"]
    w = e03.World.from_parts(ds, ctx.names[ds], ctx.labels[ds], ctx.dev_idx[ds], ctx.conf_idx[ds], e01.arm_sentinel(ctx.proto, arm), True, tb, pools)
    w.K = K
    if mv_obs is not None:  # MV observer replaces the pixel observer (carry-forward over key frames)
        w.motion, w.border = mv_obs["motion_cf"].astype(np.float64), mv_obs["border_cf"].astype(np.float64)
        w.mot_l, w.bor_l = w.motion.tolist(), w.border.tolist()
    return w


def set_costs(w: e03.World, K: dict | None = None, pools: dict | None = None) -> None:
    if K is not None:
        w.K = K
    if pools is not None:
        w.model = e02.ServiceModel(pools)
    w._cost.clear()


def run_cell_t(w: e03.World, cfg: dict, s: float, beta: float, target: float, clips: list[int], w_ref: float, knob: float | None = None, trace: dict | None = None) -> dict:
    """E03 `run_cell` with an explicit CPU target (the E03 absolute target of the cell, never re-derived from this world's own K = 1 cost)."""
    c = w.cost(s)
    kind = e03.knob_kind(cfg)
    lo, hi = e03.KNOB_RANGE[kind]
    mk = lambda x: e03.make_policy(w, c, cfg, x, w_ref)  # noqa: E731
    if knob is None:
        cal = e03.calibrate(lambda x: e03.total_cost(w, e03.simulate(w, c, mk(x), clips, light=True)), lo, hi, target, increasing=(kind == "rate"))
    else:
        cal = {"knob": knob, "cost": None, "status": "given", "evals": 0}
    out = {"cfg": cfg, "s": s, "beta": beta, "target": target, "knob": cal["knob"], "status": cal["status"], "evals": cal["evals"], "w_ref": w_ref,
           "hold": max(0, int(round(w_ref)) - 1)}
    R = e03.simulate(w, c, mk(cal["knob"]), clips, light=False, trace=trace)
    out["cost"] = e03.total_cost(w, R)
    out["ratio"] = out["cost"] / target
    out["R"] = R
    return out


def _eval_job(job: tuple) -> tuple:
    name, cfg, cell, w_ref, target, knob = job
    w, allc = _G["w"], _G["all"]
    r = run_cell_t(w, cfg, cell[0], cell[1], target, allc, w_ref, knob)
    return name, e03.cell_id(*cell), e03.compact_result(w, r)


def run_arms(w: e03.World, specs: list[tuple], procs: int, log=print) -> dict:
    """specs: (arm name, cfg, cell, w_ref, target, knob | None). -> {(arm, cell id): compact result}."""
    _G.update(w=w, all=list(range(len(w.names))))
    t0 = time.perf_counter()
    try:
        with mp.get_context("fork").Pool(procs) as pool:
            res = pool.map(_eval_job, specs, chunksize=1)
    finally:
        _G.clear()
    log(f"[{now()}]   {len(res)} simulations in {time.perf_counter() - t0:.0f}s")
    return {(name, cid): r for name, cid, r in res}


def stage_run_dataset(ctx, ds: str, cfg: dict, cfg03: dict, procs: int, log=print) -> dict:
    sel = cfg03["datasets"][ds]["tuning"]["selected"]
    e03run = pickle.loads(e03.run_path(ds).read_bytes())
    cells = [tuple(c) for c in cfg["cells"]]
    cids = [e03.cell_id(*c) for c in cells]
    pools0 = e02.build_pools(ds, PRIMARY[ds], "locked")
    c1cfg = lambda lvl=3: {"arm": "Cache-C1", "l1": lvl, "l2": 0}  # noqa: E731
    mocfg = sel["Motion-Only"]
    assert sel["Cache-C1"] == c1cfg(3), sel["Cache-C1"]
    results: dict = {}
    allc = list(range(len(ctx.names[ds])))
    t0 = time.perf_counter()
    # --- B0: E03 replay (pixel path, E03 tables, E03 costs) -----------------------------------------------------------------
    w = make_world(ctx, ds, e03.load_tables_npz(ds), pools0, k_pixel(cfg03, ds))
    w_ref = {cid: e03run["w_ref"][cid] for cid in cids}
    target = {e03.cell_id(*c): c[1] * e03.c_full(w, c[0], allc, "all") for c in cells}
    log(f"[{now()}] run {ds}: B0 replay + sensitivities")
    specs = [("C1-Pixel", c1cfg(), c, w_ref[cid], target[cid], None) for c, cid in zip(cells, cids)]
    specs += [("Motion-Only-Pixel", mocfg, c, w_ref[cid], target[cid], None) for c, cid in zip(cells, cids)]
    specs += [(f"C1-Pixel|L{lv}", c1cfg(lv), c, w_ref[cid], target[cid], None) for lv in (1, 2) for c, cid in zip(cells, cids)]
    results.update(run_arms(w, specs, procs, log))
    knob_pix = {cid: results[("C1-Pixel", cid)]["knob"] for cid in cids}
    set_costs(w, k_pixel_vm(cfg03, cfg, ds))
    results.update(run_arms(w, [("C1-Pixel|vmcost", c1cfg(), c, w_ref[cid], target[cid], None) for c, cid in zip(cells, cids)], procs, log))
    set_costs(w, k_pixel(cfg03, ds))
    d1 = {"C1-Pixel": {}, "Motion-Only-Pixel": {}}
    for arm, e03arm in (("C1-Pixel", "Cache-C1"), ("Motion-Only-Pixel", "Motion-Only")):
        for cid in cids:
            a, b = results[(arm, cid)], e03run["results"][(e03arm, cid)]
            d1[arm][cid] = {"series_equal": bool(np.array_equal(a["series"], b["series"])), "ideal_equal": bool(np.array_equal(a["ideal"], b["ideal"])),
                            "cost_equal": bool(a["cost"] == b["cost"]), "knob_equal": bool(a["knob"] == b["knob"]), "target_equal": bool(a["target"] == b["target"]),
                            "max_abs_series_diff": float(np.max(np.abs(a["series"].astype(np.float64) - b["series"].astype(np.float64))))}
    del w
    # --- +MV ------------------------------------------------------------------------------------------------------------------
    ms = load_mvstat("mv", ds)
    log(f"[{now()}] run {ds}: +MV")
    Kmv = k_mv(cfg03, cfg, ds)
    w = make_world(ctx, ds, load_tb("mv", ds), pools0, Kmv, ms)
    specs = [("C1-MV", c1cfg(), c, w_ref[cid], target[cid], None) for c, cid in zip(cells, cids)]
    specs += [("C1-MV@W", c1cfg(), c, w_ref[cid], target[cid], knob_pix[cid]) for c, cid in zip(cells, cids)]
    specs += [("Motion-Only+MV", mocfg, c, w_ref[cid], target[cid], None) for c, cid in zip(cells, cids)]
    specs += [(f"C1-MV|L{lv}", c1cfg(lv), c, w_ref[cid], target[cid], None) for lv in (1, 2) for c, cid in zip(cells, cids)]
    results.update(run_arms(w, specs, procs, log))
    knob_mv = {cid: results[("C1-MV", cid)]["knob"] for cid in cids}
    for mg in MV_SENS_MARGINS:
        set_costs(w, k_mv(cfg03, cfg, ds, mg))
        results.update(run_arms(w, [(f"C1-MV|cost{mg:.1f}", c1cfg(), c, w_ref[cid], target[cid], None) for c, cid in zip(cells, cids)], procs, log))
    set_costs(w, Kmv)
    traces = {}
    mid, pick = (2.0, 0.50), [w.conf_idx[0], w.conf_idx[len(w.conf_idx) // 2], w.dev_idx[0]]
    for arm, cf_, kn in (("C1-MV", c1cfg(), knob_mv), ("Motion-Only+MV", mocfg, {e03.cell_id(*mid): results[("Motion-Only+MV", e03.cell_id(*mid))]["knob"]})):
        cid = e03.cell_id(*mid)
        tr = {0: [], 1: [], 2: []}
        run_cell_t(w, cf_, mid[0], mid[1], target[cid], pick, w_ref[cid], kn[cid], tr)
        traces[arm] = {"clips": [w.names[i] for i in pick], "cell": cid, "frames": {k: v[:400] for k, v in tr.items()},
                       "columns": ["t", "observe", "c1_level", "c1_reused", "forced_detect", "windows_fresh", "windows_reused", "queue_wait_ms", "tokens"]}
    del w
    # --- advection: the same schedule as C1-MV, only the reused poses are shifted ---------------------------------------------
    log(f"[{now()}] run {ds}: +MV advection")
    w = make_world(ctx, ds, load_tb("mvadv", ds), pools0, Kmv, ms)
    results.update(run_arms(w, [("C1-MV-Advect", c1cfg(), c, w_ref[cid], target[cid], knob_mv[cid]) for c, cid in zip(cells, cids)], procs, log))
    del w
    # --- +INT8 (ShanghaiTech) -------------------------------------------------------------------------------------------------
    if ds == "ShanghaiTech":
        log(f"[{now()}] run {ds}: +INT8 (on B0 and on +MV)")
        w = make_world(ctx, ds, load_tb("int8", ds), int8_pools(pools0, cfg), k_pixel(cfg03, ds))
        specs = [("C1-INT8", c1cfg(), c, w_ref[cid], target[cid], None) for c, cid in zip(cells, cids)]
        specs += [("C1-INT8@W", c1cfg(), c, w_ref[cid], target[cid], knob_pix[cid]) for c, cid in zip(cells, cids)]
        results.update(run_arms(w, specs, procs, log))
        set_costs(w, pools=int8_pools(pools0, cfg, corrected=False))
        results.update(run_arms(w, [("C1-INT8|uncorrected", c1cfg(), c, w_ref[cid], target[cid], None) for c, cid in zip(cells, cids)], procs, log))
        del w
        w = make_world(ctx, ds, load_tb("int8mv", ds), int8_pools(pools0, cfg), Kmv, ms)
        specs = [("C1-MV-INT8", c1cfg(), c, w_ref[cid], target[cid], None) for c, cid in zip(cells, cids)]
        specs += [("C1-MV-INT8@W", c1cfg(), c, w_ref[cid], target[cid], knob_mv[cid]) for c, cid in zip(cells, cids)]
        results.update(run_arms(w, specs, procs, log))
        del w
    return {"stamp": sha(EXP_DIR / "config.yaml"), "results": results, "w_ref": w_ref, "targets": target, "d1": d1, "traces": traces,
            "e03_targets_equal": {cid: bool(target[cid] == e03run["results"][("Cache-C1", cid)]["target"]) for cid in cids},
            "n_frames": [len(y) for y in ctx.labels[ds]], "build_s": time.perf_counter() - t0}


# ---- detector quality (offline S0, causal S4, recall by person height) ----------------------------------------------------------
QUALITY_CACHES = {  # key -> (cache dir, sentinel arm, label)
    "ML0_ncnn": (None, "ML0_ncnn", "ML0 NCNN FP16 @640 (deployed baseline)"),
    "ML0_ort_int8": (CACHE_JOBS["ML0_ort_int8"][0], "ML0_ncnn", "ML0 ORT INT8 @640 (+INT8, calibration only)"),
    "ML0_ort_fp32": (CACHE_JOBS["ML0_ort_fp32"][0], "ML0_ncnn", "ML0 ORT FP32 @640 (runtime control)"),
    "stock_ncnn640": (CACHE_JOBS["stock_ncnn640"][0], "stock_ncnn800", "stock YOLOv8n-pose NCNN FP16 @640"),
    "stock_ort_int8_640": (EXISTING_CACHES["stock_ort_int8_640"], "stock_ncnn800", "stock ORT INT8 @640 (EXP-012)"),
    "stock_ort_fp32_640": (EXISTING_CACHES["stock_ort_fp32_640"], "stock_ncnn800", "stock ORT FP32 @640 (EXP-012)"),
    "stock_ncnn800": (EXISTING_CACHES["stock_ncnn800"], "stock_ncnn800", "stock NCNN FP16 @800 (EXP-012)"),
    "stock_ort_int8_800": (EXISTING_CACHES["stock_ort_int8_800"], "stock_ncnn800", "stock ORT INT8 @800 (EXP-012)"),
}
QUALITY_PAIRS = (("ML0_ort_int8", "ML0_ncnn"), ("ML0_ort_int8", "ML0_ort_fp32"), ("ML0_ort_fp32", "ML0_ncnn"), ("stock_ort_int8_640", "stock_ncnn640"),
                 ("stock_ort_int8_800", "stock_ncnn800"))


def recall_counts(cache_dir: Path, tracks: dict, videos: list[str]) -> np.ndarray:
    """(videos, height bins, [persons, found]) of the released reference tracks against a detector cache (E01 matcher: conf >= 0.25, IoU >= 0.3)."""
    from edgelab.analysis.exp011 import BIN_NAMES, height_bin

    counts = np.zeros((len(videos), len(BIN_NAMES), 2), dtype=np.int64)
    for vi, video in enumerate(videos):
        boxes = e01.load_boxes(cache_dir / f"{video}.npz")
        for pid, entries in tracks[video].items():
            hit = e01.hit_flags(e01.best_conf_per_entry(entries, boxes))
            for (_, height, _), h in zip(entries, hit):
                counts[vi, height_bin(height), 0] += 1
                counts[vi, height_bin(height), 1] += int(h)
    return counts


def stage_quality(ctx, procs: int, log=print) -> dict:
    from edgelab.analysis import exp022
    from edgelab.analysis.exp011 import BIN_NAMES

    ds = "ShanghaiTech"
    arms = e01.arm_by_key(ctx.arms)
    names, labels = ctx.names[ds], ctx.labels[ds]
    hold = e01.hold_of(ctx.proto, ds)
    ref = exp022.reference_tracks()
    videos = sorted(ref)
    dev_set = {names[i] for i in ctx.dev_idx[ds]}
    out: dict = {"caches": {}, "pairs": {}, "recall_pairs": {}, "tracker_cells": {k: TRACKERS_E04[k].label for k in TRACKERS_E04}, "hold": hold}
    series: dict = {}
    counts: dict = {}
    for key, (cdir, sent_arm, label) in QUALITY_CACHES.items():
        cdir = arms[f"{ds}/ML0_ncnn"].cache_dir if key == "ML0_ncnn" else cdir
        sentinel = e01.arm_sentinel(ctx.proto, arms[f"{ds}/{sent_arm}"])
        log(f"[{now()}] quality {key}: replay + score ({len(names)} clips)")
        frames, ids = e01.load_frames_floor(cdir, names, GATE_FLOOR)
        entry = {"label": label, "cache_dir": str(cdir), "sentinel": sentinel, "trackers": {}}

        def consume(lab, tracks_by_clip, _key=key, _entry=entry):
            table = e01.window_table_from_tracks(tracks_by_clip, SEG_LENS[ds], ctx.scorers[ds])
            r = e01.evaluate_tracks(tracks_by_clip, table, names, labels, ds, sentinel, hold, ctx.dev_idx[ds], ctx.conf_idx[ds])
            _entry["trackers"][lab] = {"auc": r["auc"], "coverage": r["coverage"], "n_windows": r["n_windows"], "agg_all": {k: v for k, v in r["agg"]["all"].items() if isinstance(v, (int, float))}}
            series[(_key, lab)] = {s: v for s, v in r["series"].items()}

        e01.replay_cells(list(TRACKERS_E04.values()), frames, ids, procs, consume)
        c = recall_counts(Path(cdir), ref, videos)
        counts[key] = c
        tot = c.sum(axis=0)
        grp = lambda bins, cc=c: {"persons": int(cc[:, bins, 0].sum()), "found": int(cc[:, bins, 1].sum()), "recall": float(cc[:, bins, 1].sum() / max(cc[:, bins, 0].sum(), 1))}  # noqa: E731
        entry["recall"] = {"all": grp(list(range(len(BIN_NAMES)))), "lt48": grp(e01.BIN_GROUPS["<48"]), "48-96": grp(e01.BIN_GROUPS["48-96"]), "ge96": grp(e01.BIN_GROUPS[">=96"]),
                           "bins": {n: {"persons": int(tot[i, 0]), "found": int(tot[i, 1])} for i, n in enumerate(BIN_NAMES)}}
        out["caches"][key] = entry
    locked_label = TRACKERS_E04["locked"].label
    for a, b in QUALITY_PAIRS:
        res: dict = {}
        for tlabel in (locked_label, TRACKERS_E04["default"].label):
            for step in ("S0", "S4"):
                for sub, idx in (("all", None), ("dev", ctx.dev_idx[ds]), ("confirmation", ctx.conf_idx[ds])):
                    res[f"{tlabel}|{step}|{sub}"] = e00.paired_diff(labels, series[(a, tlabel)][step], series[(b, tlabel)][step], idx, N_BOOT)
        out["pairs"][f"{a}-{b}"] = res
        out["recall_pairs"][f"{a}-{b}"] = {g: e01.bootstrap_count_diff(counts[a], counts[b], bins) for g, bins in e01.BIN_GROUPS.items()}
    return out


TRACKERS_E04 = {"locked": e02.TRACKERS["locked"], "default": e02.TRACKERS["default"]}


def stage_run(ctx, procs: int, log=print, force: bool = False) -> dict:
    cfg, cfg03 = load_config(), load_cfg03()
    stamp = sha(EXP_DIR / "config.yaml")
    out = {}
    for ds in DATASETS:
        path = run_path(ds)
        if path.exists() and not force and pickle.loads(path.read_bytes())["stamp"] == stamp:
            log(f"[{now()}] run {ds}: cached")
            continue
        run = stage_run_dataset(ctx, ds, cfg, cfg03, procs, log)
        path.write_bytes(pickle.dumps(run))
        log(f"[{now()}] run {ds}: {len(run['results'])} (arm, cell) results in {run['build_s']:.0f}s")
    qpath = WORK_DIR / "quality.json"
    if force or not qpath.exists() or json.loads(qpath.read_text()).get("stamp") != stamp:
        q = stage_quality(ctx, procs, log)
        q["stamp"] = stamp
        qpath.write_text(json.dumps(plain(q), indent=1))
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["prep", "prep-models", "prep-encode", "prep-caches", "vm", "tables", "lock", "run", "report", "all"])
    p.add_argument("--procs", type=int, default=8)
    p.add_argument("--only", nargs="*", default=None)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    if args.cmd == "prep":
        cmd_prep(args.procs)
    elif args.cmd == "prep-models":
        print(stage_models())
    elif args.cmd == "prep-encode":
        print(stage_encode(args.procs))
    elif args.cmd == "prep-caches":
        print(stage_caches(args.procs, args.only or list(CACHE_JOBS)))
    elif args.cmd == "vm":
        vm_drive(only=args.only)
    elif args.cmd == "tables":
        stage_tables(e01.build_ctx(args.procs), args.procs, force=args.force)
    elif args.cmd == "lock":
        stage_lock()
    elif args.cmd == "run":
        stage_run(e01.build_ctx(args.procs), args.procs, force=args.force)
    elif args.cmd == "report":
        from edgelab.analysis import q2_p1_e04_report as rep

        rep.stage_report(e01.build_ctx(args.procs), args.procs)
    else:
        raise SystemExit("`all` is not defined: run prep, vm, tables, lock, run, report in this order")


if __name__ == "__main__":
    main()
