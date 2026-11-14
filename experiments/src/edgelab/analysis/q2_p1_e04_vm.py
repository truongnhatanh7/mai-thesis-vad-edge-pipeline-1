"""Q2-P1-E04 lightweight helpers: H.264 re-encode, codec motion-vector (MV) extraction and the VM-side profiling workers.

Imports only numpy / PyAV / cv2 / onnxruntime (lazily), so the module runs unchanged inside the `edge-os` VM venv (no torch / ultralytics) and on the
host. Shared between the host driver (`q2_p1_e04.py`) and the VM:

    python -m edgelab.analysis.q2_p1_e04_vm mvprof --kind {sht,ub} --threads T --out <prefix>     # decode / MV / pixel-path profile
    python -m edgelab.analysis.q2_p1_e04_vm pose   --model <id> --threads T --step S --out <prefix> # detector profile (E02 worker, E04 models)

MV semantics (what is and is not claimed)
  * Real H.264 macroblock partitions exported by libavcodec (`flags2 +export_mvs`, PyAV side data MOTION_VECTORS) of an IPPP stream
    (x264 `bframes=0 ref=1 scenecut=0`, key frame every 2 s): every P frame has one reference = the previous frame, so a vector is the
    displacement (dst - src) of a block between consecutive frames. I frames carry no vectors (frame flagged invalid).
  * Per 16x16 macroblock: area-weighted mean displacement (dx, dy), area-weighted mean magnitude, coverage. A macroblock without any exported
    partition is intra coded: its magnitude is set to INTRA_MAG (the content could not be predicted from the previous frame).
  * Vectors describe what the *encoder* found cheapest, not true motion: static noise has zero vectors and a person entering a static scene is
    intra / large-vector. Whether that suffices to validate a cached pose is exactly what E04 measures.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import time
from pathlib import Path

import numpy as np

MB = 16
INTRA_MAG = 16.0  # px: magnitude assigned to an intra-coded macroblock in a P frame
ACTIVE_PX = 1.0  # a macroblock is "moving" when its mean magnitude is >= 1 px / frame
BORDER_BAND = 0.10
KEYINT_S = 2.0  # key frame every 2 s (IP-camera default order of magnitude)
FPS = {"ShanghaiTech": 24, "UBnormal": 30}
X264_BASE = {"preset": "veryfast", "crf": "23", "pix_fmt": "yuv420p"}
MACHINE_TAG = "edge-os-arm64-vm (comparative proxy; physical Pi 5 pending D8/D10)"


def x264_params(ds: str) -> str:
    """IPPP, single reference, no scene-cut key frames, single thread (deterministic), key frame every 2 s."""
    k = int(round(KEYINT_S * FPS[ds]))
    return f"bframes=0:ref=1:scenecut=0:keyint={k}:min-keyint={k}:threads=1"


def mb_shape(width: int, height: int) -> tuple[int, int]:
    return int(math.ceil(height / MB)), int(math.ceil(width / MB))


# =====================================================================================================================
# 1. MV aggregation (pure numpy)
# =====================================================================================================================
def mv_grid(side: np.ndarray | None, gh: int, gw: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Exported partitions of one frame -> per-macroblock (dx, dy, mag, cov), each (gh, gw) float32.

    `side` = structured array with fields w, h, src_x, src_y, dst_x, dst_y (AVMotionVector; (dst_x, dst_y) is the block centre). dx / dy: area
    weighted mean displacement dst - src; mag: area weighted mean of the partition magnitudes; cov: covered fraction of the 256 px macroblock
    (0 = no partition exported = intra). Vectors whose source is a *future* frame (source > 0) are ignored (causality; never produced by IPPP)."""
    z = np.zeros((gh, gw), dtype=np.float32)
    if side is None or len(side) == 0:
        return z, z.copy(), z.copy(), z.copy()
    if "source" in side.dtype.names:
        side = side[side["source"] <= 0]
        if len(side) == 0:
            return z, z.copy(), z.copy(), z.copy()
    area = side["w"].astype(np.float64) * side["h"].astype(np.float64)
    ddx = (side["dst_x"].astype(np.float64) - side["src_x"].astype(np.float64))
    ddy = (side["dst_y"].astype(np.float64) - side["src_y"].astype(np.float64))
    cx = np.clip(side["dst_x"].astype(np.int64) // MB, 0, gw - 1)
    cy = np.clip(side["dst_y"].astype(np.int64) // MB, 0, gh - 1)
    flat = cy * gw + cx
    n = gh * gw
    sa = np.bincount(flat, weights=area, minlength=n)
    safe = np.maximum(sa, 1.0)
    dx = np.bincount(flat, weights=area * ddx, minlength=n) / safe
    dy = np.bincount(flat, weights=area * ddy, minlength=n) / safe
    mag = np.bincount(flat, weights=area * np.hypot(ddx, ddy), minlength=n) / safe
    cov = np.clip(sa / float(MB * MB), 0.0, 1.0)
    return (dx.reshape(gh, gw).astype(np.float32), dy.reshape(gh, gw).astype(np.float32), mag.reshape(gh, gw).astype(np.float32),
            cov.reshape(gh, gw).astype(np.float32))


def magnitude_grid(mag: np.ndarray, cov: np.ndarray, intra_mag: float = INTRA_MAG) -> np.ndarray:
    """Magnitude with intra macroblocks (coverage < 0.5) set to `intra_mag`."""
    return np.where(cov >= 0.5, mag, np.float32(intra_mag)).astype(np.float32)


def border_mask(gh: int, gw: int, band: float = BORDER_BAND) -> np.ndarray:
    bh, bw = max(1, int(round(gh * band))), max(1, int(round(gw * band)))
    m = np.zeros((gh, gw), dtype=bool)
    m[:bh], m[-bh:], m[:, :bw], m[:, -bw:] = True, True, True, True
    return m


def frame_mv_stats(mag: np.ndarray, border: np.ndarray) -> tuple[float, float, float, float]:
    """(mv_p90, mv_border_p90, mv_motion, mv_border_active) of one magnitude grid (intra already mapped to INTRA_MAG).

    mv_motion = p90 / INTRA_MAG + 2 * (fraction of macroblocks >= ACTIVE_PX)  (the analogue of EXP-023's q90/255 + 2 * active fraction);
    mv_border_active = fraction of the border-band macroblocks >= ACTIVE_PX (the analogue of the pixel border activity)."""
    p90 = float(np.quantile(mag, 0.90))
    bp90 = float(np.quantile(mag[border], 0.90))
    active = mag >= ACTIVE_PX
    return p90, bp90, p90 / INTRA_MAG + 2.0 * float(active.mean()), float(active[border].mean())


# =====================================================================================================================
# 2. Encode + extract (PyAV; no cv2 import so the host process never loads two libavdevice copies)
# =====================================================================================================================
def _encoder(container, ds: str, width: int, height: int):
    st = container.add_stream("libx264", rate=FPS[ds])
    st.width, st.height = width, height
    st.pix_fmt = X264_BASE["pix_fmt"]
    st.options = {"preset": X264_BASE["preset"], "crf": X264_BASE["crf"], "x264-params": x264_params(ds)}
    return st


def encode_frames(frames, ds: str, out: Path, width: int, height: int) -> int:
    """frames: iterable of av.VideoFrame | (h, w, 3) uint8 RGB arrays. Writes `out` (.mp4, IPPP). Returns the number of frames."""
    import av

    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.mp4")
    n = 0
    with av.open(str(tmp), "w") as c:
        st = _encoder(c, ds, width, height)
        for f in frames:
            if isinstance(f, np.ndarray):
                f = av.VideoFrame.from_ndarray(f, format="rgb24")
            f.pts = None
            for p in st.encode(f):
                c.mux(p)
            n += 1
        for p in st.encode():
            c.mux(p)
    os.replace(tmp, out)
    return n


def sht_frames(frame_dir: Path):
    from PIL import Image

    for f in sorted(frame_dir.glob("*.jpg")):
        with Image.open(f) as im:
            yield np.asarray(im.convert("RGB"))


def ub_frames_av(path: Path):
    import av

    with av.open(str(path)) as c:
        for f in c.decode(video=0):
            yield f.to_ndarray(format="rgb24")


def open_mv_decoder(path: Path, threads: int = 1, export: bool = True):
    import av

    c = av.open(str(path))
    s = c.streams.video[0]
    if export:
        s.codec_context.flags2 |= av.codec.context.Flags2.export_mvs
    s.thread_type = "AUTO"
    s.thread_count = threads
    return c, s


def side_array(frame):
    sd = frame.side_data.get("MOTION_VECTORS") if hasattr(frame.side_data, "get") else None
    return None if sd is None else sd.to_ndarray()


def extract_clip(path: Path) -> dict:
    """Decode an IPPP file with exported MVs -> per-frame macroblock grids and frame types (all float16 / uint8 / bool arrays)."""
    c, s = open_mv_decoder(path, threads=1)
    width, height = s.codec_context.width, s.codec_context.height
    gh, gw = mb_shape(width, height)
    dxs, dys, mags, covs, ptype, nent = [], [], [], [], [], []
    for f in c.decode(s):
        side = side_array(f)
        dx, dy, mag, cov = mv_grid(side, gh, gw)
        dxs.append(dx), dys.append(dy), mags.append(mag), covs.append(cov)
        ptype.append(int(f.pict_type))
        nent.append(0 if side is None else len(side))
    c.close()
    ptype = np.array(ptype, dtype=np.uint8)
    return {"dx": np.stack(dxs).astype(np.float16), "dy": np.stack(dys).astype(np.float16), "mag": np.stack(mags).astype(np.float16),
            "cov": (np.stack(covs) * 255).round().astype(np.uint8), "ptype": ptype, "valid": ptype == 2, "n_entries": np.array(nent, dtype=np.int32),
            "size": np.array([width, height])}


def clip_scalar_stats(mag: np.ndarray, cov: np.ndarray, valid: np.ndarray) -> dict:
    """Per-frame MV statistics of a whole clip (NaN on invalid = intra / key frames): mv_p90, mv_border_p90, mv_motion, mv_border."""
    n, gh, gw = mag.shape
    border = border_mask(gh, gw)
    out = {k: np.full(n, np.nan, dtype=np.float32) for k in ("mv_p90", "mv_border_p90", "mv_motion", "mv_border")}
    for i in range(n):
        if not valid[i]:
            continue
        m = magnitude_grid(mag[i].astype(np.float32), cov[i].astype(np.float32) / 255.0)
        out["mv_p90"][i], out["mv_border_p90"][i], out["mv_motion"][i], out["mv_border"][i] = frame_mv_stats(m, border)
    return out


def encode_job(job: tuple) -> dict:
    """(dataset, clip, source path, h264 path, mv-cache path) -> re-encode as IPPP H.264 and cache the macroblock MV grids + per-frame statistics.

    ShanghaiTech: the test set only exists as JPEG frame folders; UBnormal: the original .mp4 is decoded and re-encoded (second generation).
    Idempotent: an existing complete mv-cache file is kept."""
    ds, clip, src, h264, mvp = job
    src, h264, mvp = Path(src), Path(h264), Path(mvp)
    if mvp.exists():
        z = np.load(mvp)
        return {"clip": clip, "dataset": ds, "frames": int(len(z["valid"])), "cached": True}
    t0 = time.perf_counter()
    if ds == "ShanghaiTech":
        first = next(sht_frames(src))
        height, width = first.shape[:2]
        frames = sht_frames(src)
    else:
        import av

        with av.open(str(src)) as c:
            s = c.streams.video[0]
            width, height = s.codec_context.width, s.codec_context.height
        frames = ub_frames_av(src)
    n_enc = encode_frames(frames, ds, h264, width, height)
    t1 = time.perf_counter()
    r = extract_clip(h264)
    t2 = time.perf_counter()
    assert len(r["valid"]) == n_enc, (clip, len(r["valid"]), n_enc)
    stats = clip_scalar_stats(r["mag"], r["cov"], r["valid"])
    mvp.parent.mkdir(parents=True, exist_ok=True)
    tmp = mvp.with_suffix(".partial.npz")
    np.savez_compressed(tmp, **r, **stats)
    os.replace(tmp, mvp)
    return {"clip": clip, "dataset": ds, "frames": n_enc, "cached": False, "bytes": h264.stat().st_size, "encode_s": t1 - t0, "extract_s": t2 - t1,
            "intra_cell_frac": float(((r["cov"][r["valid"]] / 255.0) < 0.5).mean()) if r["valid"].any() else float("nan"),
            "zero_cell_frac": float((r["mag"][r["valid"]] == 0).mean()) if r["valid"].any() else float("nan")}


# =====================================================================================================================
# 3. Pixel-path observers (numpy copies of the E03 functions, so the VM worker needs no torch; parity is unit-tested)
# =====================================================================================================================
GRID_H, GRID_W = 24, 40


def luma_small(gray: np.ndarray, ds: int = 4) -> np.ndarray:
    return gray[::ds, ::ds]


def motion_observer(prev_small: np.ndarray, curr_small: np.ndarray, band: float = 0.10) -> tuple[float, float, float]:
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


def box_cells_grid(boxes: np.ndarray, width: float, height: float, gh: int, gw: int, dilate: int = 1) -> np.ndarray:
    """Boolean (gh, gw) mask of the cells touched by `boxes` (x1, y1, x2, y2 px), dilated by `dilate` cells (E03 `box_cells` for any grid)."""
    mask = np.zeros((gh, gw), dtype=bool)
    for x1, y1, x2, y2 in np.asarray(boxes, dtype=float).reshape(-1, 4):
        c0, c1 = int(math.floor(x1 / width * gw)) - dilate, int(math.floor(x2 / width * gw)) + dilate
        r0, r1 = int(math.floor(y1 / height * gh)) - dilate, int(math.floor(y2 / height * gh)) + dilate
        mask[max(r0, 0):min(r1, gh - 1) + 1, max(c0, 0):min(c1, gw - 1) + 1] = True
    return mask


def roi_check(delta: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    """(mean inside the ROI, 95th percentile outside it) of a difference / accumulated-motion grid (the E03 C1 statistic pair)."""
    inside, outside = mask, ~mask
    return (float(delta[inside].mean()) if inside.any() else 0.0, float(np.percentile(delta[outside], 95)) if outside.any() else 0.0)


# =====================================================================================================================
# 4. Profiling workers (VM): decode with / without MV export, MV aggregation, pixel path
# =====================================================================================================================
def status_kb(key: str) -> float:
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith(key + ":"):
            return float(line.split()[1])
    return float("nan")


def pin(threads: int) -> None:
    cpus = set(range(threads))
    if hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, cpus)


class Clock:
    def __init__(self):
        self.w, self.c = time.perf_counter(), time.process_time()

    def lap(self) -> tuple[float, float]:
        w, c = time.perf_counter(), time.process_time()
        out = ((w - self.w) * 1e3, (c - self.c) * 1e3)
        self.w, self.c = w, c
        return out


def versions() -> dict:
    from importlib import metadata

    out = {"python": platform.python_version(), "machine": MACHINE_TAG, "kernel": platform.release(), "cpus_visible": os.cpu_count()}
    for mod, dist in (("numpy", "numpy"), ("av", "av"), ("cv2", "opencv-python-headless"), ("ncnn", "ncnn"), ("onnxruntime", "onnxruntime")):
        try:
            out[mod] = metadata.version(dist)
        except Exception:  # noqa: BLE001
            out[mod] = None
    return out


def save(prefix: Path, arrays: dict, meta: dict) -> None:
    peak = status_kb("VmHWM") / 1024.0
    prefix.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(f"{prefix}.npz", **{k: np.asarray(v) for k, v in arrays.items()})
    Path(f"{prefix}.json").write_text(json.dumps({**meta, "versions": versions(), "peak_rss_mb": peak, "loadavg_end": os.getloadavg(),
                                                  "finished": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=1, default=str))


MV_STAGE_DIR = Path(os.environ.get("EDGELAB_EXCHANGE", "/Users/nanh/thesis-data/vm-exchange")) / "q2-p1-e04"
PROFILE_CLIPS = {"sht": ["01_0014", "01_0136", "03_0032", "03_0039", "04_0004", "04_0010", "05_0021", "05_0024", "08_0079", "08_0178", "12_0143", "12_0149"],
                 "ub": ["abnormal_scene_1_scenario_1", "abnormal_scene_5_scenario_4", "normal_scene_9_scenario_1"]}
PROFILE_MAX_FRAMES = {"sht": 240, "ub": 300}
METHODS = ("decode_plain", "decode_mv", "bgr", "gray", "mv_grid", "mv_c1", "mv_obs", "pix_c1", "pix_obs", "cv2_grab", "cv2_retrieve")
SYN_BOXES = {"sht": np.array([[100.0, 80.0, 180.0, 300.0], [400.0, 100.0, 460.0, 330.0]]),
             "ub": np.array([[200.0, 150.0, 330.0, 560.0], [700.0, 160.0, 820.0, 600.0]])}


def run_mvprof(kind: str, threads: int, out: Path) -> None:
    """Per-frame wall / CPU (ms) of every ingredient, one process, pinned to cores 0..T-1, every clip decoded 3 times (plain, MV, cv2).

    decode_plain  PyAV decode, no MV export                    decode_mv     PyAV decode with export_mvs (side data created, not read)
    bgr / gray    frame.to_ndarray('bgr24' / 'gray') of a decoded frame (the pixel path needs gray; the detector path needs bgr)
    mv_grid       side-data -> ndarray + per-macroblock grids (dx, dy, mag, cov)
    mv_c1         mv_grid + magnitude map + accumulated ROI statistics (mean inside / p95 outside the person boxes) = the C1 check frame cost
    mv_obs        mv_grid + magnitude map + per-frame (p90, border p90, motion, border) statistics = the Motion-Only observer frame cost
    pix_c1        gray + 4x down-sample + 24x40 thumbnail + ROI statistics = the pixel C1 check frame cost (E03 accounts 0.15 / 0.26 ms)
    pix_obs       gray + 4x down-sample + motion observer = the pixel observer frame cost (E03 accounts 0.366 ms)
    cv2_grab / cv2_retrieve  cv2.VideoCapture grab / retrieve of the same IPPP files (comparability with E02's pools)"""
    import cv2

    pin(threads)
    cv2.setNumThreads(threads)
    ds = "ShanghaiTech" if kind == "sht" else "UBnormal"
    files = [MV_STAGE_DIR / "h264" / kind / f"{c}.mp4" for c in PROFILE_CLIPS[kind]]
    boxes = SYN_BOXES[kind]
    rows: dict[str, list] = {f"{m}_{u}": [] for m in METHODS for u in ("w", "c")}
    file_id: list[int] = []
    ptypes: list[int] = []

    def push(name: str, lap: tuple[float, float]) -> None:
        rows[f"{name}_w"].append(lap[0])
        rows[f"{name}_c"].append(lap[1])

    def decode_pass(path: Path, export: bool, body) -> int:
        c, s = open_mv_decoder(path, threads, export)
        it = c.decode(s)
        n = 0
        while n < PROFILE_MAX_FRAMES[kind]:
            clk = Clock()
            try:
                f = next(it)
            except StopIteration:
                break
            lap = clk.lap()
            body(n, f, lap)
            n += 1
        c.close()
        return n

    # warm-up
    decode_pass(files[0], True, lambda n, f, lap: (side_array(f), f.to_ndarray(format="gray")))
    gh = gw = 0
    for fi, path in enumerate(files):
        # (1) plain decode + pixel-path ingredients -------------------------------------------------------------------------
        prev = {"small": None, "thumb": None}
        c0, _s = open_mv_decoder(path, threads, False)
        width, height = _s.codec_context.width, _s.codec_context.height
        c0.close()
        mask_px = box_cells_grid(boxes, width, height, GRID_H, GRID_W)

        def plain_body(n, f, lap):
            push("decode_plain", lap)
            clk = Clock()
            f.to_ndarray(format="bgr24")
            push("bgr", clk.lap())
            clk = Clock()
            gray = f.to_ndarray(format="gray")
            small = luma_small(gray)
            g_lap = clk.lap()
            push("gray", g_lap)
            if prev["small"] is None:
                prev["small"], prev["thumb"] = small, thumb(small)
                for k in ("pix_c1", "pix_obs"):
                    push(k, (float("nan"), float("nan")))
                return
            # pixel observer frame: gray + small (already timed in g_lap) + motion observer
            clk = Clock()
            motion_observer(prev["small"], small)
            a, b = clk.lap()
            push("pix_obs", (g_lap[0] + a, g_lap[1] + b))
            clk = Clock()
            th = thumb(small)
            d = np.abs(th.astype(np.float32) - prev["thumb"].astype(np.float32))
            roi_check(d, mask_px)
            a, b = clk.lap()
            push("pix_c1", (g_lap[0] + a, g_lap[1] + b))
            prev["small"], prev["thumb"] = small, th

        n = decode_pass(path, False, plain_body)
        file_id.extend([fi] * n)
        # (2) decode with MV export + aggregation ---------------------------------------------------------------------------
        gh, gw = mb_shape(width, height)
        mask_mv = box_cells_grid(boxes, width, height, gh, gw)
        border = border_mask(gh, gw)
        acc = {"a": np.zeros((gh, gw), dtype=np.float32)}

        def mv_body(n_, f, lap):
            push("decode_mv", lap)
            if n_ < n:
                ptypes.append(int(f.pict_type))
            clk = Clock()
            side = side_array(f)
            dx, dy, mag, cov = mv_grid(side, gh, gw)
            g_lap = clk.lap()
            push("mv_grid", g_lap)
            clk = Clock()
            m = magnitude_grid(mag, cov)
            acc["a"] = acc["a"] * 0.0 + m
            roi_check(acc["a"], mask_mv)
            a, b = clk.lap()
            push("mv_c1", (g_lap[0] + a, g_lap[1] + b))
            clk = Clock()
            frame_mv_stats(m, border)
            a, b = clk.lap()
            push("mv_obs", (g_lap[0] + a, g_lap[1] + b))

        n2 = decode_pass(path, True, mv_body)
        assert n2 == n, (n2, n)
        # (3) cv2 grab / retrieve of the same file (E02 comparability) -------------------------------------------------------
        cap = cv2.VideoCapture(str(path))
        k = 0
        while k < n:
            clk = Clock()
            ok = cap.grab()
            lg = clk.lap()
            if not ok:
                break
            ok, _ = cap.retrieve()
            lr = clk.lap()
            push("cv2_grab", lg)
            push("cv2_retrieve", lr)
            k += 1
        cap.release()
        while k < n:  # frame-count mismatch between the decoders is a bug, but keep columns aligned
            push("cv2_grab", (float("nan"), float("nan")))
            push("cv2_retrieve", (float("nan"), float("nan")))
            k += 1
    arrays = {k: np.array(v, dtype=np.float64) for k, v in rows.items()}
    arrays["file_id"] = np.array(file_id)
    arrays["ptype"] = np.array(ptypes)
    save(out, arrays, {"cell": "mvprof", "kind": kind, "dataset": ds, "threads": threads, "frames": len(file_id), "files": [p.name for p in files],
                       "grid": [gh, gw], "affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
                       "x264": x264_params(ds), "max_frames_per_clip": PROFILE_MAX_FRAMES[kind]})


# ---- detector profiling: the E02 worker with the E04 models ---------------------------------------------------------------------
E04_MODELS = {  # id -> (runtime, path relative to MV_STAGE_DIR / "models" or absolute)
    "ML0-640-ncnn": ("ncnn", "/Users/nanh/thesis-data/vm-exchange/models/exp020/ML0-640-384x640-ncnn"),
    "ML0-640-ort-fp32": ("ort", "ML0-640-384x640.onnx"),
    "ML0-640-ort-int8": ("ort", "ML0-640-384x640-int8.onnx"),
    "stock-640-ncnn": ("ncnn", "/Users/nanh/thesis-data/vm-exchange/models/yolov8n-pose-640-384x640-ncnn"),
    "stock-640-ort-int8": ("ort", "/Users/nanh/thesis-data/vm-exchange/models/yolov8n-pose-640-384x640-int8.onnx"),
    "stock-800-ncnn": ("ncnn", "/Users/nanh/thesis-data/vm-exchange/models/yolov8n-pose-800-480x800-ncnn"),
    "stock-800-ort-int8": ("ort", "/Users/nanh/thesis-data/vm-exchange/models/yolov8n-pose-800-480x800-int8.onnx"),
}


def run_pose(model_id: str, threads: int, step: int, out: Path) -> None:
    from edgelab.analysis import q2_p1_e02_vm as vm

    runtime, path = E04_MODELS[model_id]
    p = Path(path) if os.path.isabs(path) else MV_STAGE_DIR / "models" / path
    vm.POSE_MODELS[model_id] = (runtime, p)
    vm.run_pose(model_id, threads, step, out)


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("mvprof")
    m.add_argument("--kind", required=True, choices=["sht", "ub"])
    m.add_argument("--threads", type=int, required=True)
    m.add_argument("--out", type=Path, required=True)
    q = sub.add_parser("pose")
    q.add_argument("--model", required=True, choices=sorted(E04_MODELS))
    q.add_argument("--threads", type=int, required=True)
    q.add_argument("--step", type=int, default=2)
    q.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if args.cmd == "mvprof":
        run_mvprof(args.kind, args.threads, args.out)
    else:
        run_pose(args.model, args.threads, args.step, args.out)


if __name__ == "__main__":
    main()
