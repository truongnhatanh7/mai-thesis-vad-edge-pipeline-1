"""Q2-P1-E02: stage-by-stage benchmark of the fixed pipeline inside the `edge-os` Lima VM (ARM64 Ubuntu, 4 vCPU).

Workers run INSIDE the VM, one process per cell (peak RSS is per process), pinned to cores 0..T-1:

    python -m edgelab.analysis.q2_p1_e02_vm decode --kind {jpg,mpeg1,h264_sht,h264_ub} --threads T --out <prefix>
    python -m edgelab.analysis.q2_p1_e02_vm pose   --model <id> --threads T --out <prefix>
    python -m edgelab.analysis.q2_p1_e02_vm track  --out <prefix>
    python -m edgelab.analysis.q2_p1_e02_vm score  --dataset {ShanghaiTech,UBnormal} --threads T --out <prefix>
    python -m edgelab.analysis.q2_p1_e02_vm e2e    --dataset ... --source {h264,jpg} --threads T --out <prefix>

The HOST driver (`prepare`, `drive`) stages the inputs in `~/thesis-data/vm-exchange/q2-p1-e02/`, calls the workers through
`limactl shell edge-os`, samples host load outside the VM during every cell (EXP-004 rule: mean > 30 % -> excluded, re-run) and
appends one line per attempt to `out/manifest.jsonl`.

Timing is wall (`perf_counter`) and process CPU (`process_time`: user + system of ALL threads, including OpenMP/ORT spinning).
Every number is a *comparative proxy*: the VM is not a Raspberry Pi 5 (DECISIONS D8/D10).
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import re
import shlex
import shutil
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np

MACHINE_TAG = "edge-os-arm64-vm (comparative proxy; physical Pi 5 pending D8/D10)"
EXCHANGE = Path(os.environ.get("EDGELAB_EXCHANGE", "/Users/nanh/thesis-data/vm-exchange"))
E02 = EXCHANGE / "q2-p1-e02"
OUT = E02 / "out"
STAGING = E02 / "staging"
CACHE = E02 / "cache"
FRAMES = EXCHANGE / "frames"
MODELS = EXCHANGE / "models"
SRC = "/Users/nanh/Documents/RMIT_MscAI/thesis/experiments/src"
GATE = 0.05  # detector confidence floor of the locked two-stage tracker

POSE_MODELS = {  # id -> (runtime, path)
    "ML0-640": ("ncnn", MODELS / "exp020/ML0-640-384x640-ncnn"),
    "MLP-640": ("ncnn", MODELS / "exp020/MLP-640-384x640-ncnn"),
    "stock-640": ("ncnn", MODELS / "yolov8n-pose-640-384x640-ncnn"),
    "stock-800": ("ncnn", MODELS / "yolov8n-pose-800-480x800-ncnn"),
    "ML0-640-ub": ("ncnn", E02 / "models/ML0-640-448x640-ncnn"),  # UBnormal 1080x720 geometry (timing only)
    "stock-640-ort-fp32": ("ort", MODELS / "yolov8n-pose-640-384x640.onnx"),
    "stock-640-ort-int8": ("ort", MODELS / "yolov8n-pose-640-384x640-int8.onnx"),
    "stock-800-ort-fp32": ("ort", MODELS / "yolov8n-pose-800-480x800.onnx"),
    "stock-800-ort-int8": ("ort", MODELS / "yolov8n-pose-800-480x800-int8.onnx"),
}
SHT_TIMING_CLIPS = ["01_0014", "01_0136", "03_0032", "03_0039", "04_0004", "04_0010", "05_0021", "05_0024", "08_0079", "08_0178",
                    "12_0143", "12_0149"]
SHT_AVI = ["01_001", "05_001", "10_001"]  # ShanghaiTech training videos (MPEG-1 in AVI), 856x480 @ 24 fps
UB_MP4 = ["abnormal_scene_1_scenario_1", "abnormal_scene_5_scenario_4", "normal_scene_9_scenario_1"]  # H.264 1080x720 @ 30 fps
SCORE_N = [0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32, 48]
TRACK_N = list(range(0, 11)) + [12, 16]


# =====================================================================================================================
# small utilities (VM side)
# =====================================================================================================================
def status_kb(key: str) -> float:
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith(key + ":"):
            return float(line.split()[1])
    return float("nan")


def rss_mb() -> float:
    return status_kb("VmRSS") / 1024.0


def peak_rss_mb() -> float:
    return status_kb("VmHWM") / 1024.0


def pin(threads: int) -> None:
    """Pin the process and every existing thread to cores 0..threads-1 (call again after models create pools)."""
    from edgelab.pipeline.os_config import apply_os_config

    cpus = set(range(threads))
    os.sched_setaffinity(0, cpus)
    apply_os_config(cpus, None)


def versions() -> dict:
    """Package versions WITHOUT importing anything (importing torch / ultralytics would inflate the peak RSS we report)."""
    from importlib import metadata

    out = {"python": platform.python_version(), "machine": MACHINE_TAG, "kernel": platform.release(), "cpus_visible": os.cpu_count()}
    for mod, dist in (("numpy", "numpy"), ("cv2", "opencv-python-headless"), ("ncnn", "ncnn"), ("onnxruntime", "onnxruntime"),
                      ("torch", "torch"), ("ultralytics", "ultralytics")):
        try:
            out[mod] = metadata.version(dist)
        except Exception:  # noqa: BLE001
            out[mod] = None
    return out


def sha256_short(path: Path, n: int = 16) -> str:
    import hashlib

    digest = hashlib.sha256()
    for f in ([path] if path.is_file() else sorted(path.glob("model.ncnn.*"))):
        digest.update(f.read_bytes())
    return digest.hexdigest()[:n]


def save(prefix: Path, arrays: dict, meta: dict) -> None:
    peak, current = peak_rss_mb(), rss_mb()  # read BEFORE anything else is imported or allocated
    prefix.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(f"{prefix}.npz", **{k: np.asarray(v) for k, v in arrays.items()})
    Path(f"{prefix}.json").write_text(json.dumps({**meta, "versions": versions(), "peak_rss_mb": peak, "rss_end_mb": current,
                                                  "loadavg_end": os.getloadavg(), "finished": time.strftime("%Y-%m-%d %H:%M:%S")},
                                                 indent=1, default=str))


class Clock:
    """wall + process-CPU stopwatch in ms"""

    def __init__(self):
        self.w, self.c = time.perf_counter(), time.process_time()

    def lap(self) -> tuple[float, float]:
        w, c = time.perf_counter(), time.process_time()
        out = ((w - self.w) * 1e3, (c - self.c) * 1e3)
        self.w, self.c = w, c
        return out


def sht_frame_files(clips: list[str] = SHT_TIMING_CLIPS, step: int = 1) -> list[Path]:
    return [f for clip in clips for f in sorted((FRAMES / clip).glob("*.jpg"))[::step]]


def video_paths(kind: str) -> list[Path]:
    if kind == "mpeg1":
        return [STAGING / f"sht_avi/{c}.avi" for c in SHT_AVI]
    if kind == "h264_sht":
        return [STAGING / f"h264_sht/{c}.mp4" for c in SHT_TIMING_CLIPS]
    if kind == "h264_ub":
        return [STAGING / f"ub_mp4/{c}.mp4" for c in UB_MP4]
    raise ValueError(kind)


# =====================================================================================================================
# decode worker
# =====================================================================================================================
def ffmpeg_cli_decode(path: Path, threads: int) -> dict | None:
    """`ffmpeg -benchmark` decode-only pass: utime / stime / rtime / maxrss as reported by ffmpeg itself."""
    if shutil.which("ffmpeg") is None:
        return None
    r = subprocess.run(["ffmpeg", "-hide_banner", "-nostdin", "-threads", str(threads), "-benchmark", "-i", str(path), "-an", "-f", "null", "-"],
                       capture_output=True, text=True)
    m = re.findall(r"bench: utime=([\d.]+)s stime=([\d.]+)s rtime=([\d.]+)s", r.stderr)
    frames = re.findall(r"frame=\s*(\d+)", r.stderr)
    if not m or not frames:
        return None
    u, s, rt = map(float, m[-1])
    n = int(frames[-1])
    return {"frames": n, "cpu_ms_per_frame": (u + s) / n * 1e3, "wall_ms_per_frame": rt / n * 1e3}


def run_decode(kind: str, threads: int, out: Path) -> None:
    import cv2

    pin(threads)
    cv2.setNumThreads(threads)
    wall_grab, cpu_grab, wall_ret, cpu_ret, file_id = [], [], [], [], []
    files_meta = []
    if kind == "jpg":
        paths = sht_frame_files()
        blobs = [np.fromfile(p, dtype=np.uint8) for p in paths[:64]]  # warm-up on a few frames (file bytes read untimed)
        for b in blobs:
            cv2.imdecode(b, cv2.IMREAD_COLOR)
        del blobs
        for i, p in enumerate(paths):
            data = np.fromfile(p, dtype=np.uint8)  # I/O excluded: page cache, then pure JPEG decode
            clock = Clock()
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
            w, c = clock.lap()
            wall_grab.append(0.0)
            cpu_grab.append(0.0)
            wall_ret.append(w)
            cpu_ret.append(c)
            file_id.append(i // 240)
        files_meta = [{"file": str(p), "frames": 240, "size": [int(img.shape[1]), int(img.shape[0])]} for p in paths[::240]]
        cli = None
    else:
        paths = video_paths(kind)
        warm = cv2.VideoCapture(str(paths[0]))
        for _ in range(30):
            warm.read()
        warm.release()
        cli = []
        for fi, p in enumerate(paths):
            cap = cv2.VideoCapture(str(p))
            meta = {"file": p.name, "fps": cap.get(cv2.CAP_PROP_FPS), "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), "n_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                    "fourcc": int(cap.get(cv2.CAP_PROP_FOURCC)).to_bytes(4, "little").decode("latin1"), "size_bytes": p.stat().st_size}
            n = 0
            while n < 400:
                clock = Clock()
                ok = cap.grab()
                wg, cg = clock.lap()
                if not ok:
                    break
                ok, _ = cap.retrieve()
                wr, cr = clock.lap()
                wall_grab.append(wg)
                cpu_grab.append(cg)
                wall_ret.append(wr)
                cpu_ret.append(cr)
                file_id.append(fi)
                n += 1
            cap.release()
            meta["frames_timed"] = n
            files_meta.append(meta)
            cli.append({"file": p.name, **(ffmpeg_cli_decode(p, threads) or {})})
    save(out, {"wall_grab_ms": wall_grab, "cpu_grab_ms": cpu_grab, "wall_retrieve_ms": wall_ret, "cpu_retrieve_ms": cpu_ret,
               "file_id": file_id},
         {"cell": "decode", "kind": kind, "threads": threads, "files": files_meta, "ffmpeg_cli": cli, "frames": len(wall_grab),
          "affinity": sorted(os.sched_getaffinity(0))})


# =====================================================================================================================
# pose worker
# =====================================================================================================================
def make_detector(model_id: str, threads: int):
    runtime, path = POSE_MODELS[model_id]
    if runtime == "ncnn":
        from edgelab.pipeline.ncnn_pose import NcnnPose

        return NcnnPose(path, threads, True)
    from edgelab.pipeline.ort_pose import OrtPose

    return OrtPose(path, threads, True)


def ub_frames(max_frames: int):
    """Frames of the three staged UBnormal clips (decoded untimed, lazily)."""
    import cv2

    for p in video_paths("h264_ub"):
        cap = cv2.VideoCapture(str(p))
        n = 0
        while n < max_frames:
            ok, img = cap.read()
            if not ok:
                break
            yield img
            n += 1
        cap.release()


def run_pose(model_id: str, threads: int, step: int, out: Path) -> None:
    import cv2

    pin(threads)
    cv2.setNumThreads(threads)
    runtime, path = POSE_MODELS[model_id]
    det = make_detector(model_id, threads)
    rss_loaded = rss_mb()
    pin(threads)
    is_ub = model_id.endswith("-ub")

    def image_iter():
        if is_ub:
            yield from ub_frames(300)
        else:
            for f in sht_frame_files(step=step):
                yield cv2.imread(str(f))  # read outside the timed region: decode cost is the decode cell's business

    for _, img in zip(range(10), image_iter()):
        det(img, GATE)  # warm-up (buffers, thread pools)
    images = image_iter()
    rows = []
    t_start = time.perf_counter()
    for img in images:
        c0 = time.process_time()
        boxes, scores, keypoints, ms = det(img, GATE)
        cpu = (time.process_time() - c0) * 1e3
        rows.append((ms["pre_ms"], ms["infer_ms"], ms["post_ms"], ms["pre_ms"] + ms["infer_ms"] + ms["post_ms"], cpu, len(scores),
                     int((scores >= 0.25).sum())))
    a = np.array(rows)
    save(out, {"pre_ms": a[:, 0], "infer_ms": a[:, 1], "post_ms": a[:, 2], "wall_ms": a[:, 3], "cpu_ms": a[:, 4], "n_dets": a[:, 5],
               "n_dets_025": a[:, 6]},
         {"cell": "pose", "model": model_id, "runtime": runtime, "threads": threads, "input_hw": list(det.shape), "frames": len(rows),
          "frame_step": step, "rss_after_load_mb": rss_loaded, "weights_sha256_16": sha256_short(path), "conf_floor": GATE,
          "wall_total_s": time.perf_counter() - t_start, "affinity": sorted(os.sched_getaffinity(0))})


# =====================================================================================================================
# tracking worker (BoT-SORT update, locked two-stage vs default)
# =====================================================================================================================
def tracker_cells():
    from edgelab.analysis.q2_p1_e01 import DEFAULT_CELL, TrackerCell

    return {"locked": TrackerCell(0.20, 0.05, 30, 0.85), "default": DEFAULT_CELL}


def track_update(tracker, frame: dict, gate: float):
    """One frame of the live pipeline: gate, build Detections, BoT-SORT update, keypoint lookup. -> (poses {id: kp}, n_in)."""
    from edgelab.pose.replay import Detections

    keep = frame["conf"] >= gate
    boxes, conf, cls, kp = frame["boxes"][keep], frame["conf"][keep], frame["cls"][keep], frame["keypoints"][keep]
    rows = tracker.update(Detections(boxes, conf, cls), None)
    return {int(r[4]): kp[int(r[7])] for r in rows}, int(len(conf))


def synthetic_frames(n_persons: int, n_frames: int, rng: np.random.Generator) -> list[dict]:
    """`n_persons` random-walk people (box ~ 40x110 px) in an 856x480 scene, conf in [0.3, 0.95]; NCNN-like float32 arrays."""
    pos = rng.uniform([60, 60], [796, 320], size=(n_persons, 2))
    vel = rng.normal(0, 2.0, size=(n_persons, 2))
    out = []
    for _ in range(n_frames):
        pos = pos + vel + rng.normal(0, 0.6, size=pos.shape)
        pos = np.clip(pos, [30, 55], [826, 425])
        w, h = 40.0, 110.0
        boxes = np.stack([pos[:, 0] - w / 2, pos[:, 1] - h / 2, pos[:, 0] + w / 2, pos[:, 1] + h / 2], axis=1).astype(np.float32)
        kp = np.zeros((n_persons, 17, 3), np.float32)
        kp[:, :, 0] = pos[:, :1] + rng.normal(0, 8, (n_persons, 17))
        kp[:, :, 1] = pos[:, 1:] + rng.normal(0, 20, (n_persons, 17))
        kp[:, :, 2] = rng.uniform(0.2, 1.0, (n_persons, 17))
        out.append({"boxes": boxes, "conf": rng.uniform(0.3, 0.95, n_persons).astype(np.float32), "cls": np.zeros(n_persons, np.float32),
                    "keypoints": kp})
    return out


def run_track(out: Path) -> None:
    from edgelab.analysis.q2_p1_e01 import build_tracker_cell
    from edgelab.pose.detcache import load_detections

    pin(1)
    cells = tracker_cells()
    arrays: dict[str, list] = {}
    clips = sorted(CACHE.glob("*.npz"))
    meta = {"cell": "track", "real_clips": [c.stem for c in clips], "synthetic_n": TRACK_N, "threads": 1, "affinity": sorted(os.sched_getaffinity(0))}
    # warm-up: build a tracker and run a few frames so imports / allocations are not timed
    warm_tracker = build_tracker_cell(cells["locked"])
    for fr in synthetic_frames(5, 20, np.random.default_rng(0)):
        track_update(warm_tracker, fr, GATE)
    for name, cell in cells.items():
        ms, cpu, n_in, n_out, clip_id = [], [], [], [], []
        for ci, path in enumerate(clips):
            frames, _ = load_detections(path)
            tracker = build_tracker_cell(cell)
            for fr in frames:
                clock = Clock()
                poses, n = track_update(tracker, fr, cell.gate)
                w, c = clock.lap()
                ms.append(w), cpu.append(c), n_in.append(n), n_out.append(len(poses)), clip_id.append(ci)
        arrays.update({f"real_{name}_ms": ms, f"real_{name}_cpu_ms": cpu, f"real_{name}_n_in": n_in, f"real_{name}_n_out": n_out,
                       f"real_{name}_clip": clip_id})
        s_ms, s_cpu, s_n = [], [], []
        for n_persons in TRACK_N:
            for rep in range(3):
                frames = synthetic_frames(n_persons, 230, np.random.default_rng(1000 * n_persons + rep))
                tracker = build_tracker_cell(cell)
                for i, fr in enumerate(frames):
                    clock = Clock()
                    track_update(tracker, fr, cell.gate)
                    w, c = clock.lap()
                    if i >= 30:
                        s_ms.append(w), s_cpu.append(c), s_n.append(n_persons)
        arrays.update({f"syn_{name}_ms": s_ms, f"syn_{name}_cpu_ms": s_cpu, f"syn_{name}_n": s_n})
    save(out, arrays, meta)


# =====================================================================================================================
# scoring worker (window assembly + per-window normalisation + frozen STG-NF forward)
# =====================================================================================================================
def run_score(dataset: str, threads: int, reps: int, out: Path) -> None:
    import torch

    from edgelab.analysis import q2_p1_e00 as e00

    pin(threads)
    torch.set_num_threads(threads)
    pool = np.load(E02 / f"windows_{dataset}.npy")  # (P, L, 17, 3) real windows from the E01 ML0 poses
    scorer = e00.Scorer(dataset)
    L = scorer.seg_len
    rng = np.random.default_rng(0)
    pin(threads)
    for _ in range(30):  # warm-up
        scorer.score(pool[rng.choice(len(pool), 8)])
    arrays: dict[str, list] = {"n": [], "assemble_ms": [], "normalize_ms": [], "flow_ms": [], "total_ms": [], "cpu_ms": []}
    # per-frame store emulation: every track keeps its last L (x, y, conf) arrays; assembling = stack the lists
    tracks = [[pool[i][t].copy() for t in range(L)] for i in range(len(pool))]
    for n in SCORE_N:
        for _ in range(reps):
            idx = rng.choice(len(pool), size=n, replace=False) if n else []
            clock = Clock()
            cpu_start = time.process_time()
            if n:
                batch = np.stack([np.stack(tracks[i]) for i in idx])
                wa, _ = clock.lap()
                x = e00.normalize_windows(batch)
                wn, _ = clock.lap()
                scorer.score_normalized(x)
                wf, _ = clock.lap()
            else:
                wa = wn = wf = 0.0
            arrays["n"].append(n)
            arrays["assemble_ms"].append(wa)
            arrays["normalize_ms"].append(wn)
            arrays["flow_ms"].append(wf)
            arrays["total_ms"].append(wa + wn + wf)
            arrays["cpu_ms"].append((time.process_time() - cpu_start) * 1e3)
    save(out, arrays, {"cell": "score", "dataset": dataset, "threads": threads, "seg_len": L, "reps_per_n": reps, "scorer_ckpt_sha256_16":
                       sha256_short(scorer.checkpoint), "affinity": sorted(os.sched_getaffinity(0))})


# =====================================================================================================================
# end-to-end worker
# =====================================================================================================================
def is_continuous(frames: list[int], seg_len: int) -> bool:
    return len(set(frames) & set(range(frames[0], frames[0] + seg_len))) >= seg_len - 2


def e2e_clip_sources(dataset: str, source: str):
    """-> list of (clip name, kind, path-or-dir) in a fixed order."""
    if dataset == "UBnormal":
        return [(p.stem, "video", p) for p in video_paths("h264_ub")]
    if source == "jpg":
        return [(c, "jpg", FRAMES / c) for c in SHT_TIMING_CLIPS]
    return [(c, "video", p) for c, p in zip(SHT_TIMING_CLIPS, video_paths("h264_sht"))]


def run_e2e(dataset: str, source: str, threads: int, max_frames: int, out: Path) -> None:
    """decode -> pose (NCNN FP16) -> locked BoT-SORT -> ring buffers -> every complete window scored (E00 S3 workload)."""
    import cv2
    import torch

    from edgelab.analysis import q2_p1_e00 as e00
    from edgelab.analysis.q2_p1_e01 import build_tracker_cell
    from edgelab.pose.replay import Detections

    pin(threads)
    torch.set_num_threads(threads)
    cv2.setNumThreads(threads)
    model_id = "ML0-640-ub" if dataset == "UBnormal" else "ML0-640"
    det = make_detector(model_id, threads)
    scorer = e00.Scorer(dataset)
    L = scorer.seg_len
    cell = tracker_cells()["locked"]
    pin(threads)
    sources = e2e_clip_sources(dataset, source)
    # warm-up (not timed): detector, tracker, scorer on the first frames of the first clip
    name0, kind0, path0 = sources[0]
    img0 = cv2.imread(str(sorted(path0.glob("*.jpg"))[0])) if kind0 == "jpg" else cv2.VideoCapture(str(path0)).read()[1]
    for _ in range(3):
        det(img0, GATE)
    scorer.score(np.random.default_rng(0).uniform(100, 400, (4, L, 17, 3)))
    cols = ["clip", "frame", "decode_ms", "pre_ms", "infer_ms", "post_ms", "track_ms", "assemble_ms", "score_ms", "service_ms", "cpu_ms",
            "n_dets", "n_in", "n_tracks", "n_windows"]
    rows = []
    t_start = time.perf_counter()
    for ci, (name, kind, path) in enumerate(sources):
        tracker = build_tracker_cell(cell)
        buffers: dict[int, deque] = {}
        cap = None if kind == "jpg" else cv2.VideoCapture(str(path))
        files = sorted(path.glob("*.jpg")) if kind == "jpg" else None
        for frame_no in range(max_frames):
            c_start = time.process_time()
            clock = Clock()
            if kind == "jpg":
                if frame_no >= len(files):
                    break
                img = cv2.imread(str(files[frame_no]))
            else:
                ok, img = cap.read()
                if not ok:
                    break
            t_dec, _ = clock.lap()
            boxes, scores, keypoints, ms = det(img, GATE)
            clock.lap()  # detector time is taken from its own pre/infer/post split
            keep = scores >= cell.gate
            tracks = tracker.update(Detections(boxes[keep], scores[keep], np.zeros(int(keep.sum()))), None)
            kp_kept = keypoints[keep]
            poses = {int(r[4]): kp_kept[int(r[7])] for r in tracks}
            t_track, _ = clock.lap()
            ready = []
            for pid, kp in poses.items():
                buf = buffers.setdefault(pid, deque(maxlen=L))
                buf.append((frame_no, kp))
                if len(buf) == L and is_continuous([f for f, _ in buf], L):
                    ready.append(np.stack([k for _, k in buf]))
            batch = np.stack(ready) if ready else None
            t_asm, _ = clock.lap()
            if batch is not None:
                scorer.score(batch)
            t_score, _ = clock.lap()
            cpu = (time.process_time() - c_start) * 1e3
            svc = t_dec + ms["pre_ms"] + ms["infer_ms"] + ms["post_ms"] + t_track + t_asm + t_score
            rows.append((ci, frame_no, t_dec, ms["pre_ms"], ms["infer_ms"], ms["post_ms"], t_track, t_asm, t_score, svc, cpu, len(scores),
                         int(keep.sum()), len(tracks), len(ready)))
        if cap is not None:
            cap.release()
    a = np.array(rows)
    save(out, {c: a[:, i] for i, c in enumerate(cols)},
         {"cell": "e2e", "dataset": dataset, "source": source, "threads": threads, "model": model_id, "clips": [s[0] for s in sources],
          "tracker": cell.label, "frames": len(rows), "wall_total_s": time.perf_counter() - t_start, "affinity": sorted(os.sched_getaffinity(0)),
          "input_hw": list(det.shape)})


# =====================================================================================================================
# HOST side: staging + driver
# =====================================================================================================================
def run_vm(args: list[str], threads: int = 1, timeout: int = 7200) -> subprocess.CompletedProcess:
    env = (f"export PATH=$HOME/edgelab-vm/.venv/bin:$HOME/.local/bin:$PATH EDGELAB_MACHINE=vm PYTHONPATH={SRC} "
           f"OMP_NUM_THREADS={threads} OPENBLAS_NUM_THREADS={threads} MKL_NUM_THREADS={threads} NUMEXPR_NUM_THREADS={threads}")
    quoted = shlex.join(args)
    return subprocess.run(["limactl", "shell", "edge-os", "--", "bash", "-c", f"{env}; cd $HOME; {quoted}"], capture_output=True, text=True,
                          timeout=timeout)


def prepare_staging(log=print) -> dict:
    """Host: copy 3 SHT .avi + 3 UB .mp4 (temporary), the 12 ML0-NCNN detector caches, real windows, the UB-geometry NCNN export; VM: x264
    re-encode of the 12 SHT clips. Idempotent."""
    from edgelab.analysis import q2_p1_e00 as e00

    data = e00.DATA_ROOT
    for d in ("sht_avi", "ub_mp4", "h264_sht"):
        (STAGING / d).mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    for c in SHT_AVI:
        dst = STAGING / f"sht_avi/{c}.avi"
        if not dst.exists():
            shutil.copy(e00.SHT_TRAIN_VIDEOS / f"{c}.avi", dst)
    for c in UB_MP4:
        dst = STAGING / f"ub_mp4/{c}.mp4"
        if not dst.exists():
            shutil.copy(e00.UB_TEST_VIDEOS / f"{c}.mp4", dst)
    src_cache = data / "exp020-ncnn/cache/L0_E0-ML0"
    for c in SHT_TIMING_CLIPS:
        if not (CACHE / f"{c}.npz").exists():
            shutil.copy(src_cache / f"{c}.npz", CACHE / f"{c}.npz")
    # real windows for the scoring benchmark (ML0 poses, E01 default tracker folders)
    for ds, pose_dir, n_clips in (("ShanghaiTech", data / "exp020-ncnn/poses/L0_E0-ML0_c0.25_botsort", 8),
                                  ("UBnormal", data / "exp020-ML0-ub/poses/L0_E0_c0.25_botsort", 14)):
        dst = E02 / f"windows_{ds}.npy"
        if dst.exists():
            continue
        seg = e00.SEG_LENS[ds]
        files = e00.index_pose_dir(pose_dir, ds)
        rng = np.random.default_rng(3)
        picked = []
        for clip in sorted(files)[::max(1, len(files) // n_clips)][:n_clips]:
            for _, (frames, kp) in e00.read_pose_clip(files[clip]).items():
                starts = e00.complete_window_starts(frames, seg)
                if len(starts):
                    picked.append(e00.gather_windows(kp, rng.choice(starts, min(len(starts), 12), replace=False), seg))
        pool = np.concatenate(picked)
        rng.shuffle(pool)
        np.save(dst, pool[:200].astype(np.float32))
        log(f"  windows_{ds}.npy: {min(len(pool), 200)} real windows")
    ub_model = POSE_MODELS["ML0-640-ub"][1]
    if not (ub_model / "model.ncnn.param").exists():
        from edgelab.pipeline.ncnn_pose import export_ncnn

        log("  exporting ML0 NCNN at 448x640 (UBnormal 1080x720 geometry) ...")
        export_ncnn(e00.EXPERIMENTS_ROOT / "artifacts/EXP-020/ML0.pt", (448, 640), ub_model)
    todo = [c for c in SHT_TIMING_CLIPS if not (STAGING / f"h264_sht/{c}.mp4").exists()]
    for c in todo:
        r = run_vm(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "24", "-i", str(FRAMES / c / "%03d.jpg"), "-c:v", "libx264",
                    "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p", "-g", "48", str(STAGING / f"h264_sht/{c}.mp4")])
        if r.returncode:
            raise RuntimeError(f"x264 re-encode of {c} failed: {r.stderr[-300:]}")
    log(f"  staging ready ({len(todo)} clips re-encoded)")
    return {"sht_avi": SHT_AVI, "ub_mp4": UB_MP4, "h264_sht": SHT_TIMING_CLIPS, "x264": "preset veryfast crf 23 yuv420p gop 48 (our re-encode)"}


def cell_list(quick: bool = False) -> list[dict]:
    """Every VM cell, interleaved over thread counts so slow drift cannot masquerade as a thread effect."""
    cells: list[dict] = []
    T = (1, 2, 3, 4)
    for rep in range(2):  # ML0 NCNN is repeated (rule P2)
        for t in T:
            cells.append({"id": f"pose_ML0-640_T{t}_rep{rep}", "args": ["pose", "--model", "ML0-640", "--threads", str(t)], "threads": t})
    for m in ("MLP-640", "stock-640", "stock-800", "stock-640-ort-fp32", "stock-640-ort-int8", "stock-800-ort-fp32", "stock-800-ort-int8"):
        for t in T:
            cells.append({"id": f"pose_{m}_T{t}", "args": ["pose", "--model", m, "--threads", str(t)], "threads": t})
    for t in T:
        cells.append({"id": f"pose_ML0-640-ub_T{t}", "args": ["pose", "--model", "ML0-640-ub", "--threads", str(t), "--step", "1"], "threads": t})
    for kind in ("jpg", "mpeg1", "h264_sht", "h264_ub"):
        for t in T:
            cells.append({"id": f"decode_{kind}_T{t}", "args": ["decode", "--kind", kind, "--threads", str(t)], "threads": t})
    cells.append({"id": "track_T1", "args": ["track"], "threads": 1})
    for ds in ("ShanghaiTech", "UBnormal"):
        for t in T:
            cells.append({"id": f"score_{ds}_T{t}", "args": ["score", "--dataset", ds, "--threads", str(t)], "threads": t})
    for t in T:
        cells.append({"id": f"e2e_ShanghaiTech_h264_T{t}", "args": ["e2e", "--dataset", "ShanghaiTech", "--source", "h264", "--threads", str(t)],
                      "threads": t})
    cells.append({"id": "e2e_ShanghaiTech_jpg_T4", "args": ["e2e", "--dataset", "ShanghaiTech", "--source", "jpg", "--threads", "4"], "threads": 4})
    for t in T:
        cells.append({"id": f"e2e_UBnormal_h264_T{t}", "args": ["e2e", "--dataset", "UBnormal", "--source", "h264", "--threads", str(t),
                                                                  "--max-frames", "300"], "threads": t})
    if quick:
        cells = [c for c in cells if c["id"].endswith(("_T1", "_T4")) or c["id"] == "track_T1"][:6]
    return cells


def drive(limit: float = 30.0, max_tries: int = 4, only: list[str] | None = None, quick: bool = False, log=print) -> list[dict]:
    """Run every cell in the VM with host-load sampling; re-run (<= max_tries) when the host outside the VM was busy."""
    from edgelab.pipeline.host_load import LoadSampler, find_vm_pid, summarise
    from edgelab.pipeline.timing_exp010 import wait_for_quiet_host

    OUT.mkdir(parents=True, exist_ok=True)
    manifest = OUT / "manifest.jsonl"
    done = set()
    if manifest.exists():
        for line in manifest.read_text().splitlines():
            row = json.loads(line)
            if row["status"] == "ok" and not row["excluded"]:
                done.add(row["cell"])
    vm_pid = find_vm_pid()
    results = []
    for cell in cell_list(quick):
        if cell["id"] in done or (only and not any(o in cell["id"] for o in only)):
            continue
        for attempt in range(max_tries):
            quiet = wait_for_quiet_host(vm_pid, limit, max_wait_s=600)
            sampler = LoadSampler(vm_pid).start()
            t0 = time.time()
            prefix = OUT / cell["id"]
            r = run_vm(["python", "-m", "edgelab.analysis.q2_p1_e02_vm", *cell["args"], "--out", str(prefix)], cell["threads"])
            load = summarise(sampler.stop())
            ok = r.returncode == 0 and Path(f"{prefix}.npz").exists()
            outside = load["host_outside_vm_pct"]
            row = {"cell": cell["id"], "attempt": attempt, "status": "ok" if ok else "failed", "seconds": round(time.time() - t0, 1),
                   "pre_run_host_outside_vm_pct": quiet, **load, "excluded": (outside is None or outside > limit),
                   "error": None if ok else (r.stderr[-400:] + r.stdout[-200:]), "start": time.strftime("%Y-%m-%d %H:%M:%S")}
            with open(manifest, "a") as f:
                f.write(json.dumps(row) + "\n")
            results.append(row)
            log(f"  [{time.strftime('%H:%M:%S')}] {cell['id']} try{attempt} {row['status']} {row['seconds']}s host_outside_vm={outside if outside is None else round(outside, 1)}%"
                f"{' EXCLUDED' if row['excluded'] else ''}")
            if ok and not row["excluded"]:
                break
    return results


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("decode")
    d.add_argument("--kind", required=True, choices=["jpg", "mpeg1", "h264_sht", "h264_ub"])
    d.add_argument("--threads", type=int, required=True)
    d.add_argument("--out", type=Path, required=True)
    q = sub.add_parser("pose")
    q.add_argument("--model", required=True, choices=sorted(POSE_MODELS))
    q.add_argument("--threads", type=int, required=True)
    q.add_argument("--step", type=int, default=1, help="frame step over the 2,880 SHT frames")
    q.add_argument("--out", type=Path, required=True)
    t = sub.add_parser("track")
    t.add_argument("--out", type=Path, required=True)
    s = sub.add_parser("score")
    s.add_argument("--dataset", required=True, choices=["ShanghaiTech", "UBnormal"])
    s.add_argument("--threads", type=int, required=True)
    s.add_argument("--reps", type=int, default=100)
    s.add_argument("--out", type=Path, required=True)
    e = sub.add_parser("e2e")
    e.add_argument("--dataset", required=True, choices=["ShanghaiTech", "UBnormal"])
    e.add_argument("--source", default="h264", choices=["h264", "jpg"])
    e.add_argument("--threads", type=int, required=True)
    e.add_argument("--max-frames", type=int, default=240)
    e.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if args.cmd == "decode":
        run_decode(args.kind, args.threads, args.out)
    elif args.cmd == "pose":
        run_pose(args.model, args.threads, args.step, args.out)
    elif args.cmd == "track":
        run_track(args.out)
    elif args.cmd == "score":
        run_score(args.dataset, args.threads, args.reps, args.out)
    elif args.cmd == "e2e":
        run_e2e(args.dataset, args.source, args.threads, args.max_frames, args.out)


if __name__ == "__main__":
    main()
