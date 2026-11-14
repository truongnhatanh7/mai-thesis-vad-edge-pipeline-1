"""Q2-P1-Pi5-Suite: PHYSICAL Raspberry Pi 5 hardware validation of E02 (stage profiling), E04 (+MV / +INT8 on Cortex-A76) and E05 (sustained live operation).

Two halves in one module (imports are lazy so each half only needs its own dependencies):

PI SIDE (runs on `nanh@raspberrypi.local`, `~/edgelab/.venv`; one process at a time, every cell its own process, PMIC power + thermal telemetry):

    python -m edgelab.analysis.q2_p1_pi5_suite drive  --part {e02,e04,all}   # the E02 / E04 cell grid -> ~/thesis-data/pi5_suite/out/
    python -m edgelab.analysis.q2_p1_pi5_suite e05    --configs a.json b.json  # sustained live streams (one process per arm)
    python -m edgelab.analysis.q2_p1_pi5_suite cell   ...                      # (internal) one telemetry-wrapped worker process

    * `drive` re-uses the locked VM workers of E02 (`q2_p1_e02_vm`) and E04 (`q2_p1_e04_vm`) UNCHANGED, so every Pi cell has exactly the file format
      (`<cell>.npz` / `<cell>.json`) and cell naming the E02 / E04 / E05 simulators already read. The child process only adds a wall-clock *timed window*
      (first/last timed work) so power and temperature are averaged over the measured region and not over model loading.
    * The parent samples `vcgencmd pmic_read_adc` (DA9091 rails -> W), SoC temperature, `get_throttled`, cpufreq and the child's RSS every 0.25 s
      (`edgelab.telemetry.pi5_telemetry`), writes one MLflow run per cell (params, metrics, telemetry step series, artifacts, `@mlflow.trace` spans).
    * `e05` is a live end-to-end stream: real H.264 decode -> pixel C1 reuse -> NCNN FP16 ML0 -> two-stage BoT-SORT -> STG-NF -> online causal delivery,
      frames arriving on a 24 fps wall clock; telemetry every 60 frames to MLflow (async), `@mlflow.trace` spans on representative frames.

MAC SIDE (`simulate`, `report`, `sync`): Pi-calibrated delivered-AUC simulations (E02 / E03 / E04 / E05 arms) fed by the *measured Pi stage pools*,
2,000-resample video bootstrap, `results.csv`, `stage_timings_pi5.csv`, `summary.json`, `report.md`.

Honest accounting: every power number is the sum of the PMIC rail V x I products that `vcgencmd pmic_read_adc` exposes (SoC + DDR + I/O rails; not the
wall-socket power), sampled at 4 Hz. Timing is `perf_counter` wall and `process_time` CPU (all threads).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

# =====================================================================================================================
# 0. Paths and constants
# =====================================================================================================================
BOARD_TAG = "Raspberry Pi 5 (physical, 4x Cortex-A76 @ 2.4 GHz, 8 GB, active cooler)"
PI_HOME = Path.home()
PI_SUITE = PI_HOME / "thesis-data" / "pi5_suite"
PI_EXCHANGE = PI_SUITE / "exchange"
PI_OUT = PI_SUITE / "out"
OUT_E02 = PI_OUT / "q2-p1-e02"
OUT_E04 = PI_OUT / "q2-p1-e04"
OUT_E05 = PI_OUT / "q2-p1-e05"
PI_USER_H264 = PI_HOME / "thesis-data" / "h264"
EXP_E02, EXP_E04, EXP_E05 = "Pi5-E02-Stage-Profile", "Pi5-E04-Optimizations", "Pi5-E05-Sustained-Operation"
TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")

E02_POSE_MODELS = ("ML0-640", "MLP-640", "stock-640", "stock-800", "stock-320")  # detector scaling (user list)
E04_POSE_MODELS = ("ML0-640-ncnn", "ML0-640-ort-int8", "ML0-640-ort-fp32")
THREADS_ALL = (1, 2, 3, 4)
POSE_STEP = 6  # 2,880 natural L0 SHT frames / 6 = 480 frames per pose cell
SCORE_N_REPORT = (1, 3, 5, 12, 24)
MV_METHODS = ("decode_plain", "decode_mv", "mv_grid", "mv_c1", "mv_obs", "pix_c1", "pix_obs", "bgr", "gray", "cv2_grab", "cv2_retrieve")
SHT_CLIPS = ["01_0014", "01_0136", "03_0032", "03_0039", "04_0004", "04_0010", "05_0021", "05_0024", "08_0079", "08_0178", "12_0143", "12_0149"]
USER_H264_SHT = ["01_0014", "03_0032"]  # copied from ~/thesis-data/q2-p1-e04/h264/ShanghaiTech/ (our IPPP x264 encodes)
USER_H264_UB = ["abnormal_scene_1_scenario_1", "normal_scene_9_scenario_1"]  # original UBnormal .mp4 (1080x720 / 1280x720)
E05_ARMS = {  # name -> (policy mode, period W, hold)
    "Fixed-K1": ("fixed_k1", 1.0, 0),
    "Cache-C1": ("cache_c1", 1.0, 0),
    "Cache-C1+Guard": ("cache_c1_guard", 1.0, 0),
    "Cache-C1+Guard-Burst32": ("cache_c1_guard", 32.0, 31),
}
E05_FRAMES = 2880
E05_MARKS = (0, 720, 1440, 2880)
E05_TRACE_FRAMES = (0, 720, 1440, 2160, 2879)
TELEMETRY_INTERVAL_S = 0.25
COOL_TARGET_C = 58.0  # the Pi 5 + active cooler idles at 54-56 C


def plain(obj):
    """JSON-safe copy (numpy scalars / arrays, NaN -> None)."""
    if isinstance(obj, dict):
        return {str(k): plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [plain(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return plain(obj.tolist())
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return f if math.isfinite(f) else None
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, Path):
        return str(obj)
    return obj


# =====================================================================================================================
# 1. Pure helpers (unit-tested without a Pi)
# =====================================================================================================================
def stats4(x) -> dict:
    """mean / p50 / p95 / p99 (ms) of a sample; NaN-aware, empty -> NaN."""
    v = np.asarray(x, dtype=np.float64)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return {"n": 0, "mean": float("nan"), "p50": float("nan"), "p95": float("nan"), "p99": float("nan")}
    q = np.quantile(v, [0.5, 0.95, 0.99])
    return {"n": int(len(v)), "mean": float(v.mean()), "p50": float(q[0]), "p95": float(q[1]), "p99": float(q[2])}


def slice_telemetry(samples: list[dict], t0: float | None, t1: float | None) -> list[dict]:
    """Samples whose `timestamp_utc` lies in [t0, t1]; the whole list when the window is unknown or holds no sample."""
    if t0 is None or t1 is None or t1 <= t0:
        return list(samples)
    sel = [s for s in samples if t0 <= s["timestamp_utc"] <= t1]
    return sel if sel else list(samples)


def summarize_telemetry(samples: list[dict]) -> dict:
    """Flat summary of the telemetry samples of one timed region (power W, temperature, clock, RSS, throttle codes)."""

    def col(k):
        v = np.array([s.get(k, float("nan")) for s in samples], dtype=np.float64)
        return v[np.isfinite(v)]

    pw, pc, tc, fr = col("power_total_w"), col("power_core_w"), col("temp_c"), col("cpu_freq_mhz_mean")
    rss, cpu = col("rss_mb"), col("sys_cpu_pct")
    codes = sorted({int(s["throttled_int"]) for s in samples if s.get("throttled_int", -1) >= 0})
    return {"n_samples": len(samples), "power_total_w": float(pw.mean()) if len(pw) else float("nan"), "power_total_w_max": float(pw.max()) if len(pw) else float("nan"),
            "power_core_w": float(pc.mean()) if len(pc) else float("nan"), "temp_c_mean": float(tc.mean()) if len(tc) else float("nan"),
            "temp_c_max": float(tc.max()) if len(tc) else float("nan"), "temp_c_start": float(tc[0]) if len(tc) else float("nan"),
            "temp_c_end": float(tc[-1]) if len(tc) else float("nan"), "freq_mhz": float(fr.mean()) if len(fr) else float("nan"),
            "freq_mhz_min": float(fr.min()) if len(fr) else float("nan"), "rss_mb_peak": float(rss.max()) if len(rss) else float("nan"),
            "sys_cpu_pct": float(cpu.mean()) if len(cpu) else float("nan"), "throttled_codes": [hex(c) for c in codes],
            "throttled_any": any(c != 0 for c in codes)}


def energy_j(power_w: float, wall_ms: float) -> float:
    """Joules of one job: mean board power x job wall time."""
    return float(power_w * wall_ms / 1000.0) if math.isfinite(power_w) and math.isfinite(wall_ms) else float("nan")


def active_energy_j(power_w: float, idle_w: float, wall_ms: float) -> float:
    """Joules above the idle floor (never negative)."""
    return energy_j(max(power_w - idle_w, 0.0), wall_ms) if math.isfinite(power_w) and math.isfinite(idle_w) else float("nan")


def rss_growth_table(marks: dict[int, float], start: int = 0) -> dict:
    """RSS (MB) at the frame marks -> growth in % relative to the start mark and between consecutive marks."""
    ks = sorted(marks)
    base = marks[ks[0]] if start not in marks else marks[start]
    out = {"marks_mb": {str(k): float(marks[k]) for k in ks}, "growth_pct_vs_start": {str(k): float(100.0 * (marks[k] - base) / base) for k in ks}}
    mid = [k for k in ks if k > 0]
    if len(mid) >= 2:
        out["growth_pct_first_mark_to_end"] = float(100.0 * (marks[mid[-1]] - marks[mid[0]]) / marks[mid[0]])
    out["growth_pct_total"] = float(100.0 * (marks[ks[-1]] - base) / base)
    return out


def queue_wait_stats(wait_ms: np.ndarray) -> dict:
    w = np.asarray(wait_ms, dtype=np.float64)
    return {"p50": float(np.quantile(w, 0.5)), "p95": float(np.quantile(w, 0.95)), "p99": float(np.quantile(w, 0.99)), "max": float(w.max()), "mean": float(w.mean())}


def thin(values: list, n: int) -> list:
    """At most `n` evenly spaced elements (keeps the first and last)."""
    if len(values) <= n:
        return list(values)
    idx = np.unique(np.round(np.linspace(0, len(values) - 1, n)).astype(int))
    return [values[i] for i in idx]


def cell_specs(part: str = "all") -> list[dict]:
    """Every Pi cell, interleaved over thread counts (slow drift cannot masquerade as a thread effect).
    part e02: decode (JPEG / H.264 SHT / H.264 UB / user 2+2 clips), pose scaling, tracker, scorer, end-to-end. part e04: ML0 NCNN vs ORT INT8 vs ORT FP32, +MV profile."""
    cells: list[dict] = []
    if part in ("e02", "all"):
        for t in THREADS_ALL:
            for kind in ("jpg", "h264_sht", "h264_ub"):
                cells.append({"id": f"decode_{kind}_T{t}", "part": "e02", "kind": "decode", "args": {"kind": kind}, "threads": t})
        for t in (1, 2, 4):
            for kind in ("h264_user_sht", "h264_user_ub"):
                cells.append({"id": f"decode_{kind}_T{t}", "part": "e02", "kind": "decode", "args": {"kind": kind}, "threads": t})
        for t in THREADS_ALL:
            for m in E02_POSE_MODELS:
                cells.append({"id": f"pose_{m}_T{t}", "part": "e02", "kind": "pose", "args": {"model": m, "step": POSE_STEP}, "threads": t})
            cells.append({"id": f"pose_ML0-640-ub_T{t}", "part": "e02", "kind": "pose", "args": {"model": "ML0-640-ub", "step": 1}, "threads": t})
        cells.append({"id": "track_T1", "part": "e02", "kind": "track", "args": {}, "threads": 1})
        for ds in ("ShanghaiTech", "UBnormal"):
            for t in THREADS_ALL:
                cells.append({"id": f"score_{ds}_T{t}", "part": "e02", "kind": "score", "args": {"dataset": ds, "reps": 100}, "threads": t})
        for t in (1, 4):
            cells.append({"id": f"e2e_ShanghaiTech_h264_T{t}", "part": "e02", "kind": "e2e", "args": {"dataset": "ShanghaiTech", "source": "h264", "max_frames": 120}, "threads": t})
        cells.append({"id": "e2e_UBnormal_h264_T4", "part": "e02", "kind": "e2e", "args": {"dataset": "UBnormal", "source": "h264", "max_frames": 300}, "threads": 4})
    if part in ("e04", "all"):
        for rep in (0, 1):
            for t in THREADS_ALL:
                for m in E04_POSE_MODELS:
                    cells.append({"id": f"pose_{m}_T{t}" + (f"_rep{rep}" if rep else ""), "part": "e04", "kind": "e04pose", "args": {"model": m, "step": POSE_STEP}, "threads": t})
        for kind in ("sht", "ub"):
            for t in THREADS_ALL:
                cells.append({"id": f"mvprof_{kind}_T{t}", "part": "e04", "kind": "mvprof", "args": {"kind": kind}, "threads": t})
    return cells


def derive_rows(spec: dict, arrays: dict, meta: dict, tel: dict, idle_w: float) -> list[dict]:
    """Stage-timing rows (one per stage / configuration / N) of one finished cell. `tel` = summarize_telemetry of the timed window."""
    kind, t = spec["kind"], spec["threads"]
    base = {"part": spec["part"], "cell": spec["id"], "threads": t, "peak_rss_mb": meta.get("peak_rss_mb", float("nan")), "power_total_w": tel["power_total_w"],
            "power_core_w": tel["power_core_w"], "temp_c_mean": tel["temp_c_mean"], "temp_c_max": tel["temp_c_max"], "freq_mhz": tel["freq_mhz"],
            "throttled_any": tel["throttled_any"], "idle_power_w": idle_w}

    def row(stage, config, wall, cpu_ms=float("nan"), n=None, dataset="", note=""):
        s = stats4(wall)
        return {**base, "stage": stage, "config": config, "dataset": dataset, "n": n if n is not None else "", "frames": s["n"], "wall_mean_ms": s["mean"],
                "wall_p50_ms": s["p50"], "wall_p95_ms": s["p95"], "wall_p99_ms": s["p99"], "cpu_ms_mean": float(cpu_ms),
                "energy_j_per_frame": energy_j(tel["power_total_w"], s["mean"]), "active_energy_j_per_frame": active_energy_j(tel["power_total_w"], idle_w, s["mean"]), "note": note}

    rows: list[dict] = []
    if kind == "decode":
        k = spec["args"]["kind"]
        if k == "jpg":
            rows.append(row("decode_jpeg_imdecode", k, arrays["wall_retrieve_ms"], float(np.mean(arrays["cpu_retrieve_ms"])), dataset="ShanghaiTech 856x480"))
        else:
            ds = "UBnormal" if "ub" in k else "ShanghaiTech"
            tot_w = arrays["wall_grab_ms"] + arrays["wall_retrieve_ms"]
            tot_c = float(np.mean(arrays["cpu_grab_ms"] + arrays["cpu_retrieve_ms"]))
            rows.append(row("decode_h264_grab+retrieve", k, tot_w, tot_c, dataset=ds))
            rows.append(row("decode_h264_grab", k, arrays["wall_grab_ms"], float(np.mean(arrays["cpu_grab_ms"])), dataset=ds))
            rows.append(row("decode_h264_retrieve", k, arrays["wall_retrieve_ms"], float(np.mean(arrays["cpu_retrieve_ms"])), dataset=ds))
    elif kind in ("pose", "e04pose"):
        m = spec["args"]["model"]
        note = f"pre {np.mean(arrays['pre_ms']):.2f} / infer {np.mean(arrays['infer_ms']):.2f} / post {np.mean(arrays['post_ms']):.2f} ms; runtime {meta.get('runtime')}; input {meta.get('input_hw')}"
        rows.append(row("pose_detector", m, arrays["wall_ms"], float(np.mean(arrays["cpu_ms"])), dataset="UBnormal" if m.endswith("-ub") else "ShanghaiTech", note=note))
    elif kind == "track":
        for name in ("default", "locked"):
            n_in = arrays[f"real_{name}_n_in"]
            rows.append(row("tracker_botsort", name, arrays[f"real_{name}_ms"], float(np.mean(arrays[f"real_{name}_cpu_ms"])), n="all", note="real SHT ML0 detections"))
            for n in sorted(set(n_in.tolist())):
                sel = n_in == n
                if sel.sum() >= 20:
                    rows.append(row("tracker_botsort", name, arrays[f"real_{name}_ms"][sel], float(np.mean(arrays[f"real_{name}_cpu_ms"][sel])), n=int(n), note="real SHT ML0 detections"))
            sn = arrays[f"syn_{name}_n"]
            for n in sorted(set(sn.tolist())):
                sel = sn == n
                rows.append(row("tracker_botsort", name, arrays[f"syn_{name}_ms"][sel], float(np.mean(arrays[f"syn_{name}_cpu_ms"][sel])), n=int(n), note="synthetic persons"))
    elif kind == "score":
        ds = spec["args"]["dataset"]
        for n in sorted(set(arrays["n"].tolist())):
            sel = arrays["n"] == n
            note = f"assemble {arrays['assemble_ms'][sel].mean():.3f} / normalize {arrays['normalize_ms'][sel].mean():.3f} / flow {arrays['flow_ms'][sel].mean():.3f} ms"
            rows.append(row("scorer_stgnf", "window_assembly+normalize+forward", arrays["total_ms"][sel], float(np.mean(arrays["cpu_ms"][sel])), n=int(n), dataset=ds, note=note))
    elif kind == "e2e":
        rows.append(row("e2e_service", f"{spec['args']['dataset']}/{spec['args']['source']}", arrays["service_ms"], float(np.mean(arrays["cpu_ms"])), dataset=spec["args"]["dataset"],
                        note=f"decode {np.mean(arrays['decode_ms']):.2f} pose {np.mean(arrays['pre_ms'] + arrays['infer_ms'] + arrays['post_ms']):.2f} track {np.mean(arrays['track_ms']):.2f} "
                             f"score {np.mean(arrays['assemble_ms'] + arrays['score_ms']):.2f} ms"))
    elif kind == "mvprof":
        ds = "ShanghaiTech" if spec["args"]["kind"] == "sht" else "UBnormal"
        for meth in MV_METHODS:
            rows.append(row(f"mvprof_{meth}", meth, arrays[f"{meth}_w"], float(np.nanmean(arrays[f"{meth}_c"])), dataset=ds))
    return rows


# =====================================================================================================================
# 2. Child process: one telemetry-wrapped worker
# =====================================================================================================================
class _Rec:
    t0_clock: float | None = None
    t1_clock: float | None = None
    t0_det: float | None = None
    t1_det: float | None = None
    calls = 0


def _patch_workers(rec: type[_Rec]):
    """Hook the locked VM workers (no behaviour change): record the first / last timed work in wall-clock unix time, point the exchange folder paths at the Pi."""
    from edgelab.analysis import q2_p1_e02_vm as vm
    from edgelab.analysis import q2_p1_e04_vm as e04vm

    vm.MACHINE_TAG = e04vm.MACHINE_TAG = BOARD_TAG
    vm.POSE_MODELS["stock-320"] = ("ncnn", vm.MODELS / "yolov8n-pose-320-320x320-ncnn")
    orig_paths = vm.video_paths

    def video_paths(kind: str):
        if kind == "h264_user_sht":
            return [PI_USER_H264 / "ShanghaiTech" / f"{c}.mp4" for c in USER_H264_SHT]
        if kind == "h264_user_ub":
            return [PI_USER_H264 / "UBnormal" / f"{c}.mp4" for c in USER_H264_UB]
        return orig_paths(kind)

    vm.video_paths = video_paths

    def make_clock(base):
        class RecClock(base):
            def __init__(self):
                super().__init__()
                if rec.t0_clock is None:
                    rec.t0_clock = time.time()

            def lap(self):
                r = super().lap()
                rec.t1_clock = time.time()
                return r

        return RecClock

    vm.Clock = make_clock(vm.Clock)
    e04vm.Clock = make_clock(e04vm.Clock)

    class DetProxy:
        warmup = 10  # run_pose warms the detector with 10 calls; the e2e worker with 3 (its Clock window takes precedence)

        def __init__(self, det):
            self._det = det

        def __getattr__(self, k):
            return getattr(self._det, k)

        def __call__(self, *a, **k):
            rec.calls += 1
            if rec.calls == self.warmup + 1:
                rec.t0_det = time.time()
            out = self._det(*a, **k)
            if rec.calls > self.warmup:
                rec.t1_det = time.time()
            return out

    orig_make = vm.make_detector
    vm.make_detector = lambda model_id, threads: DetProxy(orig_make(model_id, threads))
    # E04 detector ids -> Pi model folders (the module hard-codes the Mac VM exchange)
    ex = vm.EXCHANGE
    e04vm.E04_MODELS["ML0-640-ncnn"] = ("ncnn", str(ex / "models/exp020/ML0-640-384x640-ncnn"))
    return vm, e04vm


def child_main(args: argparse.Namespace) -> None:
    rec = _Rec
    vm, e04vm = _patch_workers(rec)
    a = json.loads(args.args_json)
    out = Path(args.out)
    t = int(args.threads)
    kind = args.kind
    if kind == "decode":
        vm.run_decode(a["kind"], t, out)
    elif kind == "pose":
        vm.run_pose(a["model"], t, int(a["step"]), out)
    elif kind == "track":
        vm.run_track(out)
    elif kind == "score":
        vm.run_score(a["dataset"], t, int(a["reps"]), out)
    elif kind == "e2e":
        vm.run_e2e(a["dataset"], a["source"], t, int(a["max_frames"]), out)
    elif kind == "mvprof":
        e04vm.run_mvprof(a["kind"], t, out)
    elif kind == "e04pose":
        e04vm.run_pose(a["model"], t, int(a["step"]), out)
    else:
        raise ValueError(kind)
    t0 = rec.t0_clock if rec.t0_clock is not None else rec.t0_det
    t1 = rec.t1_clock if rec.t0_clock is not None else rec.t1_det
    Path(f"{out}.window.json").write_text(json.dumps({"t0": t0, "t1": t1, "source": "clock" if rec.t0_clock is not None else "detector"}))


# =====================================================================================================================
# 3. Parent driver: telemetry, cool-down, MLflow
# =====================================================================================================================
def _telemetry():
    from edgelab.telemetry import pi5_telemetry as tel

    return tel


def wait_cool(max_temp: float = COOL_TARGET_C, max_wait_s: float = 150.0, log=print) -> dict:
    tel = _telemetry()
    t0 = time.time()
    temp = tel.read_soc_temp_c()
    while temp > max_temp and time.time() - t0 < max_wait_s:
        time.sleep(3.0)
        temp = tel.read_soc_temp_c()
    return {"cooled_to_c": temp, "waited_s": round(time.time() - t0, 1)}


def measure_idle(seconds: float = 20.0) -> dict:
    tel = _telemetry()
    mon = tel.Pi5TelemetryMonitor(interval_s=TELEMETRY_INTERVAL_S).start()
    time.sleep(seconds)
    mon.stop()
    s = summarize_telemetry(mon.samples)
    return {"idle_power_w": s["power_total_w"], "idle_power_core_w": s["power_core_w"], "idle_temp_c": s["temp_c_mean"], "idle_freq_mhz": s["freq_mhz"], "seconds": seconds}


def board_info() -> dict:
    tel = _telemetry()
    b = tel.read_board_info()
    b["cpu_flags"] = next((ln.split(":", 1)[1].strip() for ln in Path("/proc/cpuinfo").read_text().splitlines() if ln.startswith("Features")), "")
    b["kernel"] = os.uname().release
    b["board_tag"] = BOARD_TAG
    try:
        b["arm_clock_hz"] = subprocess.check_output(["vcgencmd", "measure_clock", "arm"], text=True).strip().split("=")[-1]
    except Exception:  # noqa: BLE001
        pass
    return plain(b)


def _mlflow():
    import mlflow

    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    mlflow.set_tracking_uri(TRACKING_URI)
    return mlflow


def log_cell_mlflow(spec: dict, rows: list[dict], summary: dict, samples: list[dict], files: list[Path], board: dict, log=print) -> str | None:
    """One MLflow run per cell: tags / params, summary metrics, telemetry step series (single batched call), artifacts and a hierarchical trace."""
    try:
        mlflow = _mlflow()
        from mlflow.entities import Metric, SpanType

        exp = {"e02": EXP_E02, "e04": EXP_E04}[spec["part"]]
        mlflow.set_experiment(exp)
        with mlflow.start_run(run_name=spec["id"]) as run:
            mlflow.set_tags({"machine": "pi5", "board_model": board.get("board_model", "?"), "cpu_governor": board.get("cpu_governor", "?"), "cell_kind": spec["kind"],
                             "part": spec["part"], "physical_pi5": "true"})
            mlflow.log_params({"cell": spec["id"], "kind": spec["kind"], "threads": spec["threads"], **{f"arg_{k}": v for k, v in spec["args"].items()}})
            met = {"power_total_w": summary["tel"]["power_total_w"], "power_core_w": summary["tel"]["power_core_w"], "temp_c_mean": summary["tel"]["temp_c_mean"],
                   "temp_c_max": summary["tel"]["temp_c_max"], "freq_mhz": summary["tel"]["freq_mhz"], "peak_rss_mb": summary["peak_rss_mb"],
                   "idle_power_w": summary["idle_power_w"], "timed_window_s": summary["timed_window_s"] or 0.0, "throttled_any": float(summary["tel"]["throttled_any"])}
            head = rows[0] if rows else {}
            for k in ("wall_mean_ms", "wall_p50_ms", "wall_p95_ms", "wall_p99_ms", "cpu_ms_mean", "energy_j_per_frame", "active_energy_j_per_frame"):
                if k in head and head[k] == head[k]:
                    met[k] = float(head[k])
            mlflow.log_metrics({k: float(v) for k, v in met.items() if v is not None and v == v})
            client = mlflow.MlflowClient()
            batch = []
            for i, s in enumerate(thin(samples, 240)):
                ts = int(s["timestamp_utc"] * 1000)
                for k in ("temp_c", "power_total_w", "power_core_w", "cpu_freq_mhz_mean", "sys_cpu_pct", "rss_mb", "throttled_int"):
                    v = s.get(k)
                    if v is not None and v == v:
                        batch.append(Metric(f"live_{k}", float(v), ts, i))
            for j in range(0, len(batch), 900):
                client.log_batch(run.info.run_id, metrics=batch[j:j + 900])
            for f in files:
                if f.exists():
                    mlflow.log_artifact(str(f))
            _trace_cell(mlflow, SpanType, spec, rows, summary)
        mlflow.flush_trace_async_logging()
        return run.info.run_id
    except Exception as exc:  # noqa: BLE001  (measurement must never fail because of the tracking server)
        log(f"  [mlflow] logging failed for {spec['id']}: {type(exc).__name__}: {str(exc)[:160]}")
        return None


def _trace_cell(mlflow, SpanType, spec: dict, rows: list[dict], summary: dict) -> None:
    """Hierarchical trace: cell -> timing distribution (per stage row) -> hardware telemetry."""

    @mlflow.trace(name=f"pi5_cell_{spec['kind']}", span_type=SpanType.CHAIN)
    def traced(cell: str, threads: int):
        span = mlflow.get_current_active_span()
        if span:
            span.set_attribute("description", f"Physical Pi 5 profile cell {cell} ({threads} threads)")
        with mlflow.start_span(name="timing_distribution", span_type=SpanType.TOOL) as s_t:
            s_t.set_inputs({"cell": cell, "threads": threads, "args": spec["args"]})
            s_t.set_outputs({"rows": [{k: r[k] for k in ("stage", "config", "n", "wall_mean_ms", "wall_p50_ms", "wall_p95_ms", "wall_p99_ms", "cpu_ms_mean") if k in r}
                                      for r in rows[:12]]})
        with mlflow.start_span(name="pi5_hardware_telemetry", span_type=SpanType.TOOL) as s_h:
            s_h.set_inputs({"sensor_source": "vcgencmd_pmic_and_sysfs", "window": summary.get("window_source")})
            s_h.set_outputs(plain(summary["tel"]))
        return {"cell": cell, "rows": len(rows)}

    traced(spec["id"], spec["threads"])


def run_cell(spec: dict, out_dir: Path, idle_w: float, board: dict, mlflow_on: bool = True, log=print, max_tries: int = 2) -> dict:
    """Parent side of one cell: cool down, spawn the child, sample telemetry, derive rows, log to MLflow, write the summary + manifest line."""
    tel = _telemetry()
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = out_dir / spec["id"]
    env = {**os.environ, "EDGELAB_EXCHANGE": str(PI_EXCHANGE), "OMP_NUM_THREADS": str(spec["threads"]), "OPENBLAS_NUM_THREADS": str(spec["threads"]),
           "MKL_NUM_THREADS": str(spec["threads"]), "NUMEXPR_NUM_THREADS": str(spec["threads"]), "MLFLOW_DISABLE_AGENT_HINT": "1"}
    cmd = [sys.executable, "-m", "edgelab.analysis.q2_p1_pi5_suite", "cell", "--kind", spec["kind"], "--threads", str(spec["threads"]), "--args-json", json.dumps(spec["args"]),
           "--out", str(prefix)]
    last_err = ""
    for attempt in range(max_tries):
        cool = wait_cool(log=log)
        thr0 = tel.read_throttled()["throttled_hex"]
        t_start = time.time()
        proc = subprocess.Popen(cmd, env=env, cwd=str(PI_HOME / "edgelab"), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        mon = tel.Pi5TelemetryMonitor(interval_s=TELEMETRY_INTERVAL_S, pid=proc.pid).start()
        output, _ = proc.communicate()
        mon.stop()
        seconds = time.time() - t_start
        ok = proc.returncode == 0 and Path(f"{prefix}.npz").exists() and Path(f"{prefix}.json").exists()
        if not ok:
            last_err = (output or "")[-600:]
            log(f"  [{time.strftime('%H:%M:%S')}] {spec['id']} try{attempt} FAILED rc={proc.returncode}: {last_err[-200:]}")
            continue
        win = json.loads(Path(f"{prefix}.window.json").read_text()) if Path(f"{prefix}.window.json").exists() else {"t0": None, "t1": None, "source": "none"}
        samples = slice_telemetry(mon.samples, win["t0"], win["t1"])
        tsum = summarize_telemetry(samples)
        z = np.load(f"{prefix}.npz")
        arrays = {k: z[k] for k in z.files}
        meta = json.loads(Path(f"{prefix}.json").read_text())
        rows = derive_rows(spec, arrays, meta, tsum, idle_w)
        window_s = (win["t1"] - win["t0"]) if win["t0"] and win["t1"] else None
        summary = {"cell": spec["id"], "spec": spec, "attempt": attempt, "seconds": round(seconds, 1), "timed_window_s": window_s, "window_source": win["source"], "tel": tsum,
                   "peak_rss_mb": meta.get("peak_rss_mb"), "idle_power_w": idle_w, "cool": cool, "throttled_before": thr0, "throttled_after": tel.read_throttled()["throttled_hex"],
                   "n_telemetry_all": len(mon.samples), "rows": rows}
        (out_dir / f"{spec['id']}.telemetry.json").write_text(json.dumps(plain(samples)))
        (out_dir / f"{spec['id']}.summary.json").write_text(json.dumps(plain(summary), indent=1))
        run_id = log_cell_mlflow(spec, rows, summary, samples, [out_dir / f"{spec['id']}.summary.json", out_dir / f"{spec['id']}.telemetry.json", Path(f"{prefix}.json")], board, log) if mlflow_on else None
        summary["mlflow_run_id"] = run_id
        (out_dir / f"{spec['id']}.summary.json").write_text(json.dumps(plain(summary), indent=1))
        head = rows[0] if rows else {}
        with open(out_dir / "manifest.jsonl", "a") as f:
            f.write(json.dumps({"cell": spec["id"], "attempt": attempt, "status": "ok", "seconds": round(seconds, 1), "excluded": False, "mlflow_run_id": run_id,
                                "power_w": tsum["power_total_w"], "temp_c_max": tsum["temp_c_max"], "throttled_any": tsum["throttled_any"],
                                "start": time.strftime("%Y-%m-%d %H:%M:%S")}) + "\n")
        log(f"  [{time.strftime('%H:%M:%S')}] {spec['id']} ok {seconds:.0f}s wall_mean={head.get('wall_mean_ms', float('nan')):.2f}ms P={tsum['power_total_w']:.2f}W "
            f"T={tsum['temp_c_mean']:.1f}/{tsum['temp_c_max']:.1f}C thr={tsum['throttled_codes']} run={run_id}")
        return summary
    with open(out_dir / "manifest.jsonl", "a") as f:
        f.write(json.dumps({"cell": spec["id"], "status": "failed", "excluded": True, "error": last_err, "start": time.strftime("%Y-%m-%d %H:%M:%S")}) + "\n")
    return {"cell": spec["id"], "failed": True}


def drive_main(args: argparse.Namespace) -> None:
    board = board_info()
    PI_OUT.mkdir(parents=True, exist_ok=True)
    (PI_OUT / "board.json").write_text(json.dumps(board, indent=1))
    log = lambda m: print(m, flush=True)  # noqa: E731
    log(f"[pi5] board: {board['board_model']} governor={board['cpu_governor']} temp={board['initial_temp_c']}C throttled={board['initial_throttled']['throttled_hex']}")
    idle_file = PI_OUT / "idle.json"
    if idle_file.exists():
        idle = json.loads(idle_file.read_text())
    else:
        wait_cool(log=log)
        idle = measure_idle(30.0)
        idle_file.write_text(json.dumps(idle, indent=1))
    log(f"[pi5] idle floor: {idle['idle_power_w']:.3f} W total, {idle['idle_power_core_w']:.3f} W core, {idle['idle_temp_c']:.1f} C")
    done = set()
    for part, d in (("e02", OUT_E02), ("e04", OUT_E04)):
        man = d / "manifest.jsonl"
        if man.exists():
            for line in man.read_text().splitlines():
                r = json.loads(line)
                if r["status"] == "ok":
                    done.add((part, r["cell"]))
    for spec in cell_specs(args.part):
        if (spec["part"], spec["id"]) in done or (args.only and not any(o in spec["id"] for o in args.only)):
            continue
        run_cell(spec, OUT_E02 if spec["part"] == "e02" else OUT_E04, idle["idle_power_w"], board, not args.no_mlflow, log)
    log("[pi5] drive complete")


# =====================================================================================================================
# 4. E05: sustained live stream (worker process; one arm per process)
# =====================================================================================================================
def run_longrun(cfg_path: Path, out: Path) -> None:
    """One arm, one FIFO worker, wall-clock arrivals (`dt = 1000/24` ms), the real locked pipeline on the Pi; telemetry every 60 frames -> MLflow."""
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    import cv2
    import torch

    from edgelab.analysis import q2_p1_e00 as e00
    from edgelab.analysis import q2_p1_e02_vm as vm
    from edgelab.analysis import q2_p1_e03 as e03
    from edgelab.analysis import q2_p1_e05 as e05
    from edgelab.analysis.q2_p1_e01 import build_tracker_cell
    from edgelab.pipeline.refresh_policy import CausalScoreDeliverer
    from types import SimpleNamespace

    tel = _telemetry()
    cfg = json.loads(Path(cfg_path).read_text())
    vm.MACHINE_TAG = BOARD_TAG
    threads = int(cfg.get("threads", 4))
    vm.pin(threads)
    torch.set_num_threads(threads)
    cv2.setNumThreads(threads)
    det = vm.make_detector("ML0-640", threads)
    scorer = e00.Scorer("ShanghaiTech")
    L = scorer.seg_len
    cell = vm.tracker_cells()["locked"]
    vm.pin(threads)
    clips = cfg["clips"]
    paths = [vm.STAGING / f"h264_sht/{c}.mp4" for c in clips]
    passes, mode = int(cfg["passes"]), cfg["mode"]
    dt_ms = 1000.0 / 24.0
    cap_ms = float(cfg.get("cap_ms", e03.CAP_MS))
    use_c1, use_guard = mode in ("cache_c1", "cache_c1_guard"), mode == "cache_c1_guard"
    period = float(cfg.get("period", 1.0))
    tau_roi, tau_ext, a1, delta = float(cfg["c1"]["roi"]), float(cfg["c1"]["ext"]), int(cfg["c1"]["a1"]), L // 2
    stub = SimpleNamespace(L=L)
    lvl = 3 if use_c1 else 0
    pol = (e05.GuardedFixedPolicy(stub, None, period, lvl, None, e03.BACKLOG_SHED_ON_MS, e03.BACKLOG_SHED_OFF_MS) if use_guard
           else e03.FixedPolicy(stub, None, period, lvl, None))
    n_clip = []
    for p in paths:
        cap0 = cv2.VideoCapture(str(p))
        n_clip.append(int(cap0.get(cv2.CAP_PROP_FRAME_COUNT)))
        cap0.release()
    total = passes * sum(n_clip)
    marks = sorted(set(int(m) for m in cfg.get("rss_marks", E05_MARKS)))
    trace_frames = set(int(x) for x in cfg.get("trace_frames", E05_TRACE_FRAMES))
    log_every = int(cfg.get("log_every", 60))
    # warm-up (not timed): decoder, detector, tracker, scorer
    cap0 = cv2.VideoCapture(str(paths[0]))
    ok, img0 = cap0.read()
    cap0.release()
    for _ in range(5):
        det(img0, vm.GATE)
    scorer.score(np.random.default_rng(0).uniform(100, 400, (4, L, 17, 3)))
    wtr = build_tracker_cell(cell)
    for fr in vm.synthetic_frames(4, 12, np.random.default_rng(0)):
        vm.track_update(wtr, fr, vm.GATE)
    vm.pin(threads)
    # MLflow run (async metric logging so the per-60-frame calls do not block the worker)
    mlflow = tracers = None
    run_id = None
    if cfg.get("mlflow", True):
        try:
            mlflow = _mlflow()
            from mlflow.entities import SpanType

            mlflow.set_experiment(cfg.get("experiment", EXP_E05))
            run = mlflow.start_run(run_name=cfg["name"], description=cfg.get("description", ""))
            run_id = run.info.run_id
            board = board_info()
            mlflow.set_tags({"machine": "pi5", "board_model": board.get("board_model", "?"), "cpu_governor": board.get("cpu_governor", "?"), "arm_name": cfg["name"],
                             "physical_pi5": "true"})
            mlflow.log_params({"arm": cfg["name"], "mode": mode, "period_W": period, "hold": cfg["hold"], "c1_reuse": use_c1, "guard": use_guard, "guard_on_ms": e03.BACKLOG_SHED_ON_MS,
                               "guard_off_ms": e03.BACKLOG_SHED_OFF_MS, "threads": threads, "frames": total, "fps": 24.0, "dt_ms": round(dt_ms, 3), "cap_ms": cap_ms,
                               "c1_roi": tau_roi, "c1_ext": tau_ext, "c1_a1": a1, "forced_detect_every": delta, "detector": "ML0 NCNN FP16 @640", "tracker": cell.label})
            tracers = _make_e05_tracer(mlflow, SpanType)
        except Exception as exc:  # noqa: BLE001
            print(f"[e05] mlflow disabled: {type(exc).__name__}: {str(exc)[:160]}", flush=True)
            mlflow = tracers = None
    monitor = tel.Pi5TelemetryMonitor(interval_s=TELEMETRY_INTERVAL_S).start()
    thr_start = tel.read_throttled()["throttled_hex"]
    temp_start = tel.read_soc_temp_c()
    rss_marks: dict[int, float] = {}
    blocks: list[dict] = []
    cols = {k: np.zeros(total, dtype=np.float64) for k in ("arrival_ms", "start_ms", "wait_ms", "svc_ms", "cpu_ms", "decode_ms", "c1_ms", "pose_ms", "track_ms", "score_ms")}
    tele = {k: np.full(total, np.nan, dtype=np.float32) for k in ("power_w", "power_core_w", "temp_c", "freq_mhz")}
    flags = {k: np.zeros(total, dtype=np.int8) for k in ("obs", "reused", "forced", "shed", "scored", "windows")}
    clip_of = np.zeros(total, dtype=np.int32)
    local_t = np.zeros(total, dtype=np.int32)
    series = np.zeros(total, dtype=np.float64)
    win_rec: list[tuple] = []
    transitions: list[tuple] = []
    n_dropped_late = n_dropped_end = 0
    clip_starts: list[int] = []
    T0 = time.perf_counter() + 0.5
    cpu_start = time.process_time()
    wall_epoch = time.time() + 0.5
    last_shed = False
    g = 0
    pos = 0
    pending: list[tuple[int, float, int, int]] = []
    fin_last = 0.0
    for pass_i in range(passes):
        for ci, path in enumerate(paths):
            n = n_clip[ci]
            pol.start_clip(n)
            clip_starts.append(g)
            deliverer = CausalScoreDeliverer(float(cfg["sentinel"]), int(cfg["hold"]))
            n_dropped_end += len(pending)
            pending = []
            cap = None
            tracker = None
            buffers: dict[int, list] = {}
            rl, a = 0, -1
            src = None
            for t in range(n):
                if g in marks:
                    rss_marks[g] = vm.rss_mb()
                arrival = T0 + g * dt_ms / 1000.0
                now = time.perf_counter()
                if now < arrival:
                    if arrival - now > 0.002:
                        time.sleep(arrival - now - 0.001)
                    while time.perf_counter() < arrival:
                        pass
                    now = time.perf_counter()
                start_rel = (now - T0) * 1e3
                wait_ms = max(0.0, now - arrival) * 1e3
                c0 = time.process_time()
                clk = vm.Clock()
                avail = [v for (lf, v, _, _) in pending if lf <= g]
                pending = [p for p in pending if p[0] > g]
                series[g] = deliverer.step_frame(t, avail)
                clip_of[g], local_t[g] = pos, t
                if cap is None:
                    cap = cv2.VideoCapture(str(path))
                if not cap.grab():
                    raise RuntimeError(f"decoder ended early: {path.name} frame {t}/{n}")
                can_start = (t & 1) == 0 and t <= n - L
                obs, score, _, _ = pol.decide(t, n, g, 0.0, wait_ms, rl, can_start)
                shed_now = bool(getattr(pol, "shed", False))
                if shed_now != last_shed:
                    transitions.append((g, int(shed_now), wait_ms))
                    last_shed = shed_now
                reused = forced = False
                t_pose = t_track = t_score = t_c1 = t_dec = 0.0
                vmin = float("inf")
                n_win = 0
                n_dets = n_tracks = 0
                if obs:
                    if rl == 0:
                        if not can_start:
                            raise AssertionError("policy started a run where none is possible")
                        a, src = t, None
                        tracker = build_tracker_cell(cell)
                        buffers = {}
                    rl += 1
                    ok, img = cap.retrieve()
                    if not ok:
                        raise RuntimeError(f"retrieve failed: {path.name} frame {t}")
                    t_dec, _ = clk.lap()
                    cols["decode_ms"][g] = t_dec
                    if use_c1:
                        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        th = e03.thumb(e03.luma_small(gray)).astype(np.float32)
                    forced = (t - a) % delta == 0
                    if use_c1 and not forced and src is not None and (t - src["t"]) <= a1:
                        d = np.abs(th - src["th"])
                        m = src["mask"]
                        roi = float(d[m].mean()) if m.any() else 0.0
                        ext = float(np.percentile(d[~m], 95)) if (~m).any() else 0.0
                        reused = roi <= tau_roi and ext <= tau_ext
                    if use_c1:
                        t_c1, _ = clk.lap()
                    if reused:
                        frame = src["frame"]
                    else:
                        boxes, scores, kps, ms = det(img, vm.GATE)
                        t_pose = ms["pre_ms"] + ms["infer_ms"] + ms["post_ms"]
                        frame = {"boxes": boxes, "conf": scores, "cls": np.zeros(len(scores), dtype=np.float32), "keypoints": kps}
                        if use_c1:
                            hgt, wid = gray.shape
                            keep = scores >= e05_gate_store()
                            mask = e03.box_cells(boxes[keep], wid, hgt) if keep.any() else np.zeros((e03.GRID_H, e03.GRID_W), dtype=bool)
                            src = {"t": t, "th": th, "mask": mask, "frame": frame}
                    clk.lap()
                    poses, n_dets = vm.track_update(tracker, frame, cell.gate)
                    n_tracks = len(poses)
                    t_track, _ = clk.lap()
                    for pid, kp in poses.items():
                        buf = buffers.setdefault(pid, [])
                        buf.append((t, kp))
                        if len(buf) > L:
                            del buf[0]
                    if score:
                        ready = [np.stack([k for _, k in b]) for b in buffers.values() if len(b) == L and b[-1][0] - b[0][0] == L - 1]
                        if ready:
                            clk.lap()
                            norm = scorer.score(np.stack(ready))
                            t_score, _ = clk.lap()
                            vmin = float(np.min(norm))
                            n_win = len(ready)
                else:
                    rl = 0
                    clk.lap()
                t_end = time.perf_counter()
                fin_rel = (t_end - T0) * 1e3
                fin_last = fin_rel
                if vmin < float("inf"):
                    latency = fin_rel - g * dt_ms
                    lf = max(int(e03.e02.delivery_frame(np.array([fin_rel]), dt_ms)[0]), g)
                    late = latency > cap_ms
                    beyond = lf >= clip_starts[-1] + n
                    win_rec.append((pos, t, g, fin_rel, vmin, int(late), int(beyond)))
                    if late:
                        n_dropped_late += 1
                    elif beyond:
                        n_dropped_end += 1
                    else:
                        pending.append((lf, vmin, g, pos))
                cols["arrival_ms"][g], cols["start_ms"][g] = (arrival - T0) * 1e3, start_rel
                cols["wait_ms"][g] = wait_ms
                cols["svc_ms"][g] = (t_end - now) * 1e3
                cols["cpu_ms"][g] = (time.process_time() - c0) * 1e3
                cols["c1_ms"][g], cols["pose_ms"][g], cols["track_ms"][g], cols["score_ms"][g] = t_c1, t_pose, t_track, t_score
                flags["obs"][g], flags["reused"][g], flags["forced"][g] = int(obs), int(reused), int(forced)
                flags["shed"][g], flags["scored"][g], flags["windows"][g] = int(shed_now), int(vmin < float("inf")), n_win
                snap = monitor.samples[-1] if monitor.samples else None  # latest background sample: no sensor access on the worker thread
                if snap is not None:
                    tele["power_w"][g], tele["power_core_w"][g] = snap["power_total_w"], snap["power_core_w"]
                    tele["temp_c"][g], tele["freq_mhz"][g] = snap["temp_c"], snap["cpu_freq_mhz_mean"]
                # ---- live telemetry to MLflow every `log_every` frames (async) and traced sample frames --------------------------------------------
                if mlflow is not None and (g % log_every == 0 or g == total - 1):
                    try:
                        recent = slice(max(0, g - log_every + 1), g + 1)
                        mlflow.log_metrics({"live_service_ms": float(cols["svc_ms"][g]), "live_service_ms_mean60": float(cols["svc_ms"][recent].mean()),
                                            "live_queue_wait_ms": float(wait_ms), "live_queue_wait_ms_max60": float(cols["wait_ms"][recent].max()),
                                            "live_pose_ms": float(t_pose), "live_track_ms": float(t_track), "live_score_ms": float(t_score),
                                            "live_c1_reuse_pct60": float(100.0 * flags["reused"][recent].mean()), "live_guard_shed": float(shed_now),
                                            "live_temp_c": float(snap["temp_c"]) if snap else float("nan"),
                                            "live_power_total_w": float(snap["power_total_w"]) if snap else float("nan"),
                                            "live_power_core_w": float(snap["power_core_w"]) if snap else float("nan"),
                                            "live_cpu_freq_mhz": float(snap["cpu_freq_mhz_mean"]) if snap else float("nan"),
                                            "live_throttled_int": float(snap["throttled_int"]) if snap else float("nan"),
                                            "live_rss_mb": vm.rss_mb(), "live_sys_cpu_pct": float(snap["sys_cpu_pct"]) if snap else float("nan")}, step=g, synchronous=False)
                    except Exception:  # noqa: BLE001
                        pass
                if tracers is not None and g in trace_frames:
                    try:
                        tracers(cfg["name"], g, path.stem, t, float(t_dec), float(t_c1), float(t_pose), float(t_track), float(t_score), int(n_dets), int(n_tracks), int(n_win),
                                (None if vmin == float("inf") else vmin), bool(reused), float(wait_ms), bool(shed_now),
                                {"temp_c": snap["temp_c"] if snap else None, "power_total_w": snap["power_total_w"] if snap else None, "rss_mb": vm.rss_mb(),
                                 "throttled_hex": snap["throttled_hex"] if snap else None})
                    except Exception:  # noqa: BLE001
                        pass
                g += 1
            if cap is not None:
                cap.release()
            pos += 1
    wall_total = (fin_last / 1e3) if fin_last else time.perf_counter() - T0
    wall_epoch_end = time.time()
    rss_marks[g] = vm.rss_mb()
    cpu_total = time.process_time() - cpu_start
    summ = monitor.stop()
    thr_end = tel.read_throttled()["throttled_hex"]
    samples = [s for s in monitor.samples if s["timestamp_utc"] >= wall_epoch - 1.0]
    tsum = summarize_telemetry([s for s in samples if wall_epoch <= s["timestamp_utc"] <= wall_epoch_end] or samples)
    arrays = {**cols, **tele, **flags, "clip_of": clip_of, "local_t": local_t, "series": series, "clip_starts": np.array(clip_starts), "n_clip": np.array(n_clip * passes),
              "win_rec": np.array(win_rec, dtype=np.float64).reshape(-1, 7), "transitions": np.array(transitions, dtype=np.float64).reshape(-1, 3)}
    # temperature trajectory every 360 frames
    traj = [{"frame": int(f), "temp_c": float(tele["temp_c"][f]), "power_w": float(tele["power_w"][f]), "freq_mhz": float(tele["freq_mhz"][f])} for f in range(0, total, 360)] + \
           [{"frame": total - 1, "temp_c": float(tele["temp_c"][total - 1]), "power_w": float(tele["power_w"][total - 1]), "freq_mhz": float(tele["freq_mhz"][total - 1])}]
    wq = queue_wait_stats(cols["wait_ms"])
    j_per_frame = energy_j(tsum["power_total_w"], wall_total * 1e3 / total)
    meta = {"cell": "e05_pi5_live", "name": cfg["name"], "mode": mode, "s_emul": 1.0, "period": period, "frames": int(total), "passes": passes, "clips": clips, "threads": threads,
            "wall_total_s": wall_total, "cpu_total_s": cpu_total, "achieved_fps": total / wall_total, "offered_fps": 24.0, "rss_marks_mb": {str(k): v for k, v in rss_marks.items()},
            "rss_growth": rss_growth_table(rss_marks), "dropped_late": n_dropped_late, "dropped_end": n_dropped_end, "guard": pol.info() if hasattr(pol, "info") else {},
            "sentinel": cfg["sentinel"], "hold": cfg["hold"], "cap_ms": cap_ms, "c1": cfg["c1"], "x264": "preset veryfast crf 23 yuv420p gop 48", "affinity": sorted(os.sched_getaffinity(0)),
            "tracker": cell.label, "forced_detect_every": delta, "queue_wait_ms": wq, "telemetry": tsum, "telemetry_all": summ, "energy_j_per_frame": j_per_frame,
            "power_total_w_mean": tsum["power_total_w"], "power_core_w_mean": tsum["power_core_w"], "temp_trajectory": traj, "throttled_start": thr_start, "throttled_end": thr_end,
            "temp_c_start": temp_start, "board": board_info(), "mlflow_run_id": run_id, "guard_transitions": len(transitions), "obs_fraction": float(flags["obs"].mean()),
            "c1_hit_rate_of_inrun": float(flags["reused"].sum() / max(1, flags["obs"].sum())), "shed_fraction": float(flags["shed"].mean()), "windows_scored": int(flags["scored"].sum())}
    vm.save(out, arrays, meta)
    Path(f"{out}.telemetry.json").write_text(json.dumps(plain(monitor.samples)))
    if mlflow is not None:
        try:
            mlflow.log_metrics({"achieved_fps": meta["achieved_fps"], "queue_wait_ms_p50": wq["p50"], "queue_wait_ms_p95": wq["p95"], "queue_wait_ms_p99": wq["p99"],
                                "queue_wait_ms_max": wq["max"], "service_ms_mean": float(cols["svc_ms"].mean()), "service_ms_p95": float(np.quantile(cols["svc_ms"], 0.95)),
                                "service_ms_p99": float(np.quantile(cols["svc_ms"], 0.99)), "power_total_w_mean": tsum["power_total_w"], "power_core_w_mean": tsum["power_core_w"],
                                "energy_j_per_frame": j_per_frame, "temp_c_start": temp_start, "temp_c_max": tsum["temp_c_max"], "temp_c_end": tsum["temp_c_end"],
                                "c1_hit_rate_of_inrun": meta["c1_hit_rate_of_inrun"], "observed_fraction": meta["obs_fraction"], "shed_fraction": meta["shed_fraction"],
                                "cpu_ms_per_video_s": cpu_total * 1000.0 / (total / 24.0), "rss_start_mb": rss_marks[min(rss_marks)], "rss_end_mb": rss_marks[max(rss_marks)],
                                "rss_growth_pct_total": meta["rss_growth"]["growth_pct_total"], "throttled_any": float(tsum["throttled_any"]),
                                **{f"rss_mb_at_{k}": v for k, v in rss_marks.items()}})
            mlflow.set_tags({"throttled_start": thr_start, "throttled_end": thr_end})
            for suffix in (".json", ".telemetry.json"):
                mlflow.log_artifact(f"{out}{suffix}")
            mlflow.end_run()
            mlflow.flush_trace_async_logging()
        except Exception as exc:  # noqa: BLE001
            print(f"[e05] mlflow finalize failed: {type(exc).__name__}: {str(exc)[:160]}", flush=True)
    print(f"[e05] {cfg['name']}: {meta['achieved_fps']:.3f} FPS wait p50/p95/p99/max = {wq['p50']:.1f}/{wq['p95']:.1f}/{wq['p99']:.1f}/{wq['max']:.1f} ms "
          f"P={tsum['power_total_w']:.2f}W J/frame={j_per_frame:.4f} T={temp_start:.1f}->{tsum['temp_c_end']:.1f}C (max {tsum['temp_c_max']:.1f}) thr {thr_start}->{thr_end} "
          f"RSS {rss_marks}", flush=True)


def e05_gate_store() -> float:
    from edgelab.analysis import q2_p1_e05_vm as e05vm

    return e05vm.GATE_STORE


def _make_e05_tracer(mlflow, SpanType):
    @mlflow.trace(name="pi5_e05_live_frame", span_type=SpanType.CHAIN)
    def traced(arm: str, g: int, clip: str, t: int, dec_ms: float, c1_ms: float, pose_ms: float, track_ms: float, score_ms: float, n_dets: int, n_tracks: int, n_windows: int,
               min_normality, c1_reused: bool, queue_wait_ms: float, guard_shed: bool, snap: dict):
        span = mlflow.get_current_active_span()
        if span:
            span.set_attribute("description", f"Pi 5 sustained live frame ({arm}, clip {clip}, global frame {g})")
        mlflow.update_current_trace(metadata={"mlflow.trace.user": "nanh", "mlflow.trace.session": f"pi5-e05-{arm}"})
        with mlflow.start_span(name="0_queue", span_type=SpanType.TOOL) as s0:
            s0.set_inputs({"global_frame": g, "arrival_ms": round(g * 1000.0 / 24.0, 2)})
            s0.set_outputs({"queue_wait_ms": round(queue_wait_ms, 3), "guard_shed_on": guard_shed})
        with mlflow.start_span(name="1_h264_decode", span_type=SpanType.TOOL) as s1:
            s1.set_inputs({"clip": clip, "clip_frame": t})
            s1.set_outputs({"decode_ms": round(dec_ms, 3)})
        with mlflow.start_span(name="2_c1_static_validity_check", span_type=SpanType.TOOL) as s2:
            s2.set_outputs({"c1_ms": round(c1_ms, 3), "detector_reused": c1_reused})
        with mlflow.start_span(name="3_ncnn_ml0_pose_detector", span_type=SpanType.LLM) as s3:
            s3.set_outputs({"pose_ms": round(pose_ms, 3), "n_detections": n_dets})
        with mlflow.start_span(name="4_botsort_two_stage_tracker", span_type=SpanType.TOOL) as s4:
            s4.set_outputs({"track_ms": round(track_ms, 3), "n_tracks": n_tracks})
        with mlflow.start_span(name="5_stgnf_scorer", span_type=SpanType.LLM) as s5:
            s5.set_inputs({"n_windows": n_windows})
            s5.set_outputs({"score_ms": round(score_ms, 3), "min_normality": min_normality})
        with mlflow.start_span(name="6_pi5_hardware_telemetry", span_type=SpanType.TOOL) as s6:
            s6.set_inputs({"sensor_source": "vcgencmd_pmic_and_sysfs"})
            s6.set_outputs(snap)
        return {"arm": arm, "frame": g, "service_ms": round(dec_ms + c1_ms + pose_ms + track_ms + score_ms, 3)}

    return traced


def e05_main(args: argparse.Namespace) -> None:
    OUT_E05.mkdir(parents=True, exist_ok=True)
    log = lambda m: print(m, flush=True)  # noqa: E731
    for i, cfgp in enumerate(args.configs):
        cfg = json.loads(Path(cfgp).read_text())
        out = OUT_E05 / cfg["name_slug"]
        if Path(f"{out}.npz").exists() and not args.force:
            log(f"[e05] {cfg['name']}: cached")
            continue
        cool = wait_cool(max_temp=float(args.cool_c), max_wait_s=360.0, log=log)
        log(f"[{time.strftime('%H:%M:%S')}] [e05] {cfg['name']}: start (cooled to {cool['cooled_to_c']} C after {cool['waited_s']} s)")
        env = {**os.environ, "EDGELAB_EXCHANGE": str(PI_EXCHANGE), "OMP_NUM_THREADS": str(cfg.get("threads", 4)), "OPENBLAS_NUM_THREADS": str(cfg.get("threads", 4)),
               "MKL_NUM_THREADS": str(cfg.get("threads", 4)), "MLFLOW_DISABLE_AGENT_HINT": "1"}
        r = subprocess.run([sys.executable, "-m", "edgelab.analysis.q2_p1_pi5_suite", "longrun", "--config", str(cfgp), "--out", str(out)], env=env, cwd=str(PI_HOME / "edgelab"),
                           capture_output=True, text=True)
        log((r.stdout or "").strip()[-800:])
        if r.returncode != 0:
            log(f"[e05] {cfg['name']} FAILED rc={r.returncode}: {(r.stderr or '')[-600:]}")


# =====================================================================================================================
# 5. CLI
# =====================================================================================================================
def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("cell")
    c.add_argument("--kind", required=True)
    c.add_argument("--threads", type=int, required=True)
    c.add_argument("--args-json", default="{}")
    c.add_argument("--out", required=True)
    d = sub.add_parser("drive")
    d.add_argument("--part", default="all", choices=["e02", "e04", "all"])
    d.add_argument("--only", nargs="*", default=None)
    d.add_argument("--no-mlflow", action="store_true")
    r = sub.add_parser("longrun")
    r.add_argument("--config", type=Path, required=True)
    r.add_argument("--out", type=Path, required=True)
    e = sub.add_parser("e05")
    e.add_argument("--configs", nargs="+", required=True)
    e.add_argument("--force", action="store_true")
    e.add_argument("--cool-c", type=float, default=COOL_TARGET_C)
    # Mac side
    for name in ("simulate", "report", "sync", "lock", "view", "all-mac"):
        m = sub.add_parser(name)
        m.add_argument("--procs", type=int, default=8)
        m.add_argument("--force", action="store_true")
    args = p.parse_args()
    if args.cmd == "cell":
        child_main(args)
    elif args.cmd == "drive":
        drive_main(args)
    elif args.cmd == "longrun":
        run_longrun(args.config, args.out)
    elif args.cmd == "e05":
        e05_main(args)
    else:
        main_mac(args)  # Mac half (defined below; heavy imports are lazy)


# =====================================================================================================================
# 6. MAC SIDE: sync, simulation view, Pi-calibrated simulations, live analysis, outputs
# =====================================================================================================================
PI_REMOTE = "nanh@raspberrypi.local"
PI_REMOTE_SUITE = "/home/nanh/thesis-data/pi5_suite"
MAC_SUITE = Path.home() / "thesis-data" / "pi5_suite"
MAC_OUT = MAC_SUITE / "out"
SIM_VIEW = MAC_SUITE / "sim_view"
SIM_DIR = MAC_SUITE / "sim"
EXP_ROOT = Path(__file__).resolve().parents[3]
EXP_PI5 = EXP_ROOT / "Q2-P1-Pi5_suite"
N_BOOT = 2000
SHT_DS, UB_DS = "ShanghaiTech", "UBnormal"
DATASETS_ALL = (SHT_DS, UB_DS)
E05_ORDER = ("Fixed-K1", "Cache-C1", "Cache-C1+Guard", "Cache-C1+Guard-Burst32")
E05_SLUG = {a: a.replace("+", "_").replace("-", "_").lower() for a in E05_ARMS}
E05_SIM_SPEC = {"fixed_k1": "Full-Rate", "cache_c1": "Cache-C1", "cache_c1_guard": "Cache-C1+Guard"}
SIM_CONTRASTS = {  # name -> (A, B): A - B (E02 / E03 / E05 arms, native Pi cells)
    "E02 K=1 ladder: tracker (Full-Rate - ML0-Default-K1)": ("Full-Rate", "ML0-Default-K1"),
    "E02 K=1 ladder: frontend (ML0-Default-K1 - Stock-Default-K1)": ("ML0-Default-K1", "Stock-Default-K1"),
    "E03 Cache-C1 - Fixed": ("Cache-C1", "Fixed"),
    "E03 Cache-C1 - Stock-Baseline": ("Cache-C1", "Stock-Baseline"),
    "E05 Cache-C1+Guard - Cache-C1": ("Cache-C1+Guard", "Cache-C1"),
    "E05 Cache-C1+Guard - Fixed": ("Cache-C1+Guard", "Fixed"),
    "E05 Fixed+Guard - Fixed": ("Fixed+Guard", "Fixed"),
    "E05 C1-ReTrack - Fixed": ("C1-ReTrack", "Fixed"),
}
E04_CONTRASTS = {
    "E04 +MV: C1-MV - C1-Pixel (matched CPU)": ("C1-MV", "C1-Pixel"),
    "E04 +MV: C1-MV@W - C1-Pixel (same schedule)": ("C1-MV@W", "C1-Pixel"),
    "E04 +INT8: C1-INT8 - C1-Pixel (matched CPU)": ("C1-INT8", "C1-Pixel"),
    "E04 +INT8: C1-INT8@W - C1-Pixel (same schedule)": ("C1-INT8@W", "C1-Pixel"),
    "E04 +MV+INT8: C1-MV-INT8 - C1-MV (matched CPU)": ("C1-MV-INT8", "C1-MV"),
}
LIMITATIONS = [
    "Power is the sum of the PMIC rail V x I products exposed by `vcgencmd pmic_read_adc` (SoC core + DDR + I/O rails), sampled at 4 Hz; it is not wall-socket power "
    "(no USB-C / fan / peripheral losses). Energy J/frame = mean PMIC power over the timed window x mean wall ms/frame; 'active' J/frame subtracts the idle floor.",
    "The `ondemand` cpufreq governor was left unchanged (changing it needs sudo): the clock ramps within ~10 ms and sat at 2.4 GHz during every timed window "
    "(see `freq_mhz`), but idle-floor frequency is 1.5 GHz.",
    "Cells of about one second (JPEG / H.264 decode, user clips) contain only a few 0.25 s telemetry samples: their power and J/frame are indicative; the timing statistics are exact.",
    "The user's text states '2,880 frames = 2 full passes over the 12 SHT clips = 2.0 minutes'. 2,880 frames at 24 fps is 120 s, and the 12 SHT timing clips hold 240 frames each, "
    "so 2,880 frames is exactly ONE pass of 12 x 240 frames. The live runs use 2,880 frames (2.0 minutes of 24 fps video per arm).",
    "Thread counts T in {1, 2, 3, 4} are measured (a superset of the requested T in {1, 2, 4} for decode and ORT/NCNN, and {1, 2, 3, 4} for pose).",
    "Pi-calibrated delivered AUCs use the measured physical pools at the native cell (s = 1.0, CPU budget beta in {0.25, 0.5, 1.0}); there is no VM-to-Pi multiplier. "
    "Pixel observer / C1 check costs come from the Pi `mvprof` cells (single thread).",
    "E02 pose cells for ML0 @640 are the E02 cell and, as second replicate, the E04 ML0 NCNN FP16 cell (same model, same input, same settings); the E02 simulators pool the two.",
]


def _now() -> str:
    return time.strftime("%H:%M:%S")


def is_cell_file(name: str) -> bool:
    return name.endswith(".npz") or (name.endswith(".json") and not name.endswith((".window.json", ".telemetry.json", ".summary.json")))


def mac_sync(log=print) -> None:
    """Pi -> Mac copy of ~/thesis-data/pi5_suite (raw traces, telemetry, summaries, configs). Absolute remote path (openrsync mangles relative ones)."""
    MAC_SUITE.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(["rsync", "-a", "--exclude", "exchange/", f"{PI_REMOTE}:{PI_REMOTE_SUITE}/", f"{MAC_SUITE}/"], capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(f"rsync failed: {r.stderr[-400:]}")
    n = sum(1 for _ in MAC_SUITE.rglob("*") if _.is_file())
    log(f"[{_now()}] sync: {n} files under {MAC_SUITE}")


def build_view(log=print) -> dict:
    """Symlink view of the Pi cells under the exact file names the E02 / E04 simulators read (ML0-640 gets rep0 = E02 cell, rep1 = the E04 ML0 NCNN cell)."""
    import shutil

    v02, v04 = SIM_VIEW / "q2-p1-e02", SIM_VIEW / "q2-p1-e04"
    for d in (v02, v04):
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True)
    src02, src04 = MAC_OUT / "q2-p1-e02", MAC_OUT / "q2-p1-e04"
    for src, dst in ((src02, v02), (src04, v04)):
        for f in sorted(src.glob("*")):
            if is_cell_file(f.name):
                (dst / f.name).symlink_to(f.resolve())
    missing = []
    for t in THREADS_ALL:
        for ext in (".npz", ".json"):
            for rep, src in ((0, src02 / f"pose_ML0-640_T{t}{ext}"), (1, src04 / f"pose_ML0-640-ncnn_T{t}{ext}")):
                if src.exists():
                    (v02 / f"pose_ML0-640_T{t}_rep{rep}{ext}").symlink_to(src.resolve())
                else:
                    missing.append(str(src))
    log(f"[{_now()}] view: {len(list(v02.glob('*.npz')))} E02 cells, {len(list(v04.glob('*.npz')))} E04 cells, missing={len(missing)}")
    return {"missing": missing}


def load_stage_rows() -> list[dict]:
    """Every stage-timing row of every finished Pi cell (None -> NaN)."""
    rows: list[dict] = []
    for part in ("q2-p1-e02", "q2-p1-e04"):
        for f in sorted((MAC_OUT / part).glob("*.summary.json")):
            s = json.loads(f.read_text())
            for r in s.get("rows", []):
                rows.append({k: (float("nan") if v is None else v) for k, v in r.items()})
    return rows


def write_stage_csv(rows: list[dict]) -> int:
    cols = ["part", "cell", "threads", "stage", "config", "dataset", "n", "frames", "wall_mean_ms", "wall_p50_ms", "wall_p95_ms", "wall_p99_ms", "cpu_ms_mean", "peak_rss_mb",
            "power_total_w", "power_core_w", "energy_j_per_frame", "active_energy_j_per_frame", "temp_c_mean", "temp_c_max", "freq_mhz", "throttled_any", "idle_power_w", "note"]
    EXP_PI5.mkdir(parents=True, exist_ok=True)
    with open(EXP_PI5 / "stage_timings_pi5.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({c: (f"{r.get(c):.6g}" if isinstance(r.get(c), float) else r.get(c, "")) for c in cols})
    return len(rows)


# ---- simulation set-up ----------------------------------------------------------------------------------------------------------------------
def sim_setup():
    from types import SimpleNamespace

    from edgelab.analysis import q2_p1_e00 as e00
    from edgelab.analysis import q2_p1_e01 as e01
    from edgelab.analysis import q2_p1_e02 as e02
    from edgelab.analysis import q2_p1_e03 as e03
    from edgelab.analysis import q2_p1_e04 as e04
    from edgelab.analysis import q2_p1_e05 as e05

    e02.VM_OUT = SIM_VIEW / "q2-p1-e02"  # the simulators now read the PHYSICAL Pi pools
    e04.VM_OUT = SIM_VIEW / "q2-p1-e04"
    assert set(e04.vmm.METHODS) == set(MV_METHODS), "MV method list differs from the E04 worker"
    return SimpleNamespace(e00=e00, e01=e01, e02=e02, e03=e03, e04=e04, e05=e05)


def pi_costs(m, ds: str) -> dict:
    """Pixel observer / C1 check costs measured on the Pi (`mvprof`, single thread; the simulator charges max(wall, cpu))."""
    d = m.e04.derive_vm_costs(m.e04.vm_cost_table(), m.e04.KIND_OF[ds], 1.0)
    return {"obs_ms": d["pix_obs_ms_vm"], "c1_check_ms": d["pix_c1_ms_vm"], "c2_check_ms": m.e03.C2_CHECK_MS}


def set_pi_costs(m, w, ds: str) -> None:
    import copy

    w.K = copy.deepcopy(w.K)
    w.K["costs"] = pi_costs(m, ds)
    w._cost.clear()


def auc_boot_table(m, wl, results: dict, subsets=("all", "confirmation")) -> dict:
    """{(arm, cell, subset): (point AUC %, bootstrap AUCs %)}: exact pooled AUC, video-level paired bootstrap (same resamples for every series)."""
    cut = np.cumsum(wl.n_frames)[:-1]
    idx = {"all": None, "confirmation": list(wl.conf_idx), "dev": list(wl.dev_idx)}
    boots = {s: m.e03.AucBoot(wl.labels, idx[s], N_BOOT, 0) for s in subsets}
    out = {}
    for (arm, cid), r in sorted(results.items()):
        ser = np.split(np.asarray(r["series"], dtype=np.float64), cut)
        for s, b in boots.items():
            out[(arm, cid, s)] = b.aucs([ser[i] for i in b.idx])
    return out


def contrast_table(m, boots: dict, pairs: dict, cids: list[str], subsets=("all", "confirmation")) -> list[dict]:
    rows = []
    for name, (a, b) in pairs.items():
        for sub in subsets:
            per = {cid: m.e05.contrast_sets(boots, sub, a, b, [cid]) for cid in cids}
            mean = m.e05.contrast_sets(boots, sub, a, b, cids)
            if mean is None:
                continue
            rows.append({"contrast": name, "a": a, "b": b, "subset": sub, "mean_over_cells": plain(mean), "per_cell": {c: plain(v) for c, v in per.items() if v}})
    return rows


def pool_fork(procs: int):
    import multiprocessing as mp

    return mp.get_context("fork").Pool(procs)


def sim_e05(m, ctx, ds: str, procs: int, log=print, force: bool = False) -> dict:
    """The E05 arm ladder at the native Pi cells (s = 1.0, beta in {0.25, 0.5, 1.0}) with the physical Pi stage pools (covers the E02 K=1 ladder, E03 and E05 arms)."""
    import pickle

    path = SIM_DIR / f"e05_{ds}.pkl"
    SIM_DIR.mkdir(parents=True, exist_ok=True)
    if path.exists() and not force:
        return pickle.loads(path.read_bytes())
    e03, e05 = m.e03, m.e05
    cfg03 = e03.load_config()
    worlds = e05.make_worlds(ctx, ds, cfg03)
    for w in worlds.values():
        set_pi_costs(m, w, ds)
    allc = list(range(len(ctx.names[ds])))
    cells = list(e05.CELLS_NATIVE)
    wl = worlds["ml0_locked"]
    target = {e05.cell_id(*c): c[1] * e03.c_full(wl, c[0], allc, "all") for c in cells}
    t0 = time.perf_counter()
    e05._G.update(worlds=worlds, allc=allc)
    results: dict = {}
    try:
        with pool_fork(procs) as pool:
            own = [a for a, sp in e05.ARMS.items() if sp["ref"] == "own"]
            for arm, cid, r in pool.map(e05._job, [(a, c, None, target[e05.cell_id(*c)]) for a in own for c in cells], chunksize=1):
                results[(arm, cid)] = r
            w_ref = {e05.cell_id(*c): results[("Fixed", e05.cell_id(*c))]["knob"] for c in cells}
            log(f"[{_now()}] sim {ds}: Fixed periods " + ", ".join(f"{k}:{v:.2f}" for k, v in w_ref.items()))
            dep = [a for a, sp in e05.ARMS.items() if sp["ref"] == "fixed"]
            for arm, cid, r in pool.map(e05._job, [(a, c, w_ref[e05.cell_id(*c)], target[e05.cell_id(*c)]) for a in dep for c in cells], chunksize=1):
                results[(arm, cid)] = r
    finally:
        e05._G.clear()
    log(f"[{_now()}] sim {ds}: {len(results)} (arm, cell) results in {time.perf_counter() - t0:.0f}s")
    out = {"results": results, "w_ref": w_ref, "targets": target, "costs": pi_costs(m, ds), "n_frames": [int(x) for x in wl.n_frames]}
    path.write_bytes(pickle.dumps(out))
    return out


def sim_e04(m, ctx, ds: str, e05run: dict, procs: int, log=print, force: bool = False) -> dict:
    """E04 arms at the native Pi cells: +MV (Pi-measured decode_mv - decode_plain export overhead, MV observer / C1 cost) and +INT8 (physical ORT INT8 pool, Cortex-A76)."""
    import copy
    import pickle

    path = SIM_DIR / f"e04_{ds}.pkl"
    if path.exists() and not force:
        return pickle.loads(path.read_bytes())
    e02, e03, e04 = m.e02, m.e03, m.e04
    cfg = copy.deepcopy(e04.load_config())
    cfg03 = e04.load_cfg03()
    cfg["vm"]["cost_table"] = e04.vm_cost_table()
    cfg["vm"]["pose_drift_vs_e02"] = e04.pose_drift()
    cells = list(m.e05.CELLS_NATIVE)
    cids = [e03.cell_id(*c) for c in cells]
    w_ref, target = e05run["w_ref"], e05run["targets"]
    pools0 = e02.build_pools(ds, e04.PRIMARY[ds], "locked")
    c1cfg = {"arm": "Cache-C1", "l1": 3, "l2": 0}
    results: dict = {}
    t0 = time.perf_counter()

    def specs(name, knobs=None):
        return [(name, c1cfg, c, w_ref[cid], target[cid], None if knobs is None else knobs[cid]) for c, cid in zip(cells, cids)]

    w = e04.make_world(ctx, ds, e03.load_tables_npz(ds), pools0, e04.k_pixel_vm(cfg03, cfg, ds))
    results.update(e04.run_arms(w, specs("C1-Pixel"), procs, log))
    knob_pix = {cid: results[("C1-Pixel", cid)]["knob"] for cid in cids}
    del w
    ms = e04.load_mvstat("mv", ds)
    Kmv = e04.k_mv(cfg03, cfg, ds, 1.0)  # margin 1.0: the Pi measures the MV costs directly (E04's 1.25 VM safety margin is the sensitivity arm)
    w = e04.make_world(ctx, ds, e04.load_tb("mv", ds), pools0, Kmv, ms)
    results.update(e04.run_arms(w, specs("C1-MV") + specs("C1-MV@W", knob_pix), procs, log))
    knob_mv = {cid: results[("C1-MV", cid)]["knob"] for cid in cids}
    e04.set_costs(w, e04.k_mv(cfg03, cfg, ds, 1.25))
    results.update(e04.run_arms(w, specs("C1-MV|cost1.25"), procs, log))
    del w
    if ds == SHT_DS:
        w = e04.make_world(ctx, ds, e04.load_tb("int8", ds), e04.int8_pools(pools0, cfg), e04.k_pixel_vm(cfg03, cfg, ds))
        results.update(e04.run_arms(w, specs("C1-INT8") + specs("C1-INT8@W", knob_pix), procs, log))
        e04.set_costs(w, pools=e04.int8_pools(pools0, cfg, corrected=False))
        results.update(e04.run_arms(w, specs("C1-INT8|uncorrected"), procs, log))
        del w
        w = e04.make_world(ctx, ds, e04.load_tb("int8mv", ds), e04.int8_pools(pools0, cfg), Kmv, ms)
        results.update(e04.run_arms(w, specs("C1-MV-INT8") + specs("C1-MV-INT8@W", knob_mv), procs, log))
        del w
    log(f"[{_now()}] sim e04 {ds}: {len(results)} results in {time.perf_counter() - t0:.0f}s")
    out = {"results": results, "w_ref": w_ref, "targets": target, "cost_table": cfg["vm"]["cost_table"], "pose_drift": cfg["vm"]["pose_drift_vs_e02"],
           "derived_costs": {mg: e04.derive_vm_costs(cfg["vm"]["cost_table"], e04.KIND_OF[ds], mg) for mg in (1.0, 1.25)}}
    path.write_bytes(pickle.dumps(out))
    return out


def arm_cells_summary(results: dict, boots: dict, wl_n: list[int], arms: list[str], cids: list[str], subsets=("all", "confirmation")) -> list[dict]:
    rows = []
    for arm in arms:
        for cid in cids:
            if (arm, cid) not in results:
                continue
            r = results[(arm, cid)]
            row = {"arm": arm, "cell": cid, "cost_cpu_ms_per_video_s": float(r["cost"]), "target": float(r["target"]), "cost_ratio": float(r["ratio"]), "knob": float(r["knob"]),
                   "w_ref": float(r["w_ref"]), "status": r["status"], "observed_fraction": float(np.mean(r["observed"])), "wait_p99_ms": float(r["queue"].get("wait_p99_ms", float("nan"))
                                                                                                                                        ) if isinstance(r.get("queue"), dict) else float("nan")}
            for sub in subsets:
                if (arm, cid, sub) in boots:
                    pt, bv = boots[(arm, cid, sub)]
                    v = bv[np.isfinite(bv)]
                    row[f"auc_{sub}"] = float(pt)
                    row[f"auc_{sub}_lo"], row[f"auc_{sub}_hi"] = float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))
            rows.append(row)
    return rows


# ---- live (physical) E05 analysis -------------------------------------------------------------------------------------------------------------
def analyze_live(m, wl, arm: str, log=print) -> dict | None:
    """Physical live run -> metrics + online delivered AUC (bootstrap over the 12 videos) + the Pi-calibrated simulation of the very same stream."""
    slug = E05_SLUG[arm]
    base = MAC_OUT / "q2-p1-e05" / slug
    if not Path(f"{base}.npz").exists():
        return None
    z = np.load(f"{base}.npz")
    arr = {k: z[k] for k in z.files}
    meta = json.loads(Path(f"{base}.json").read_text())
    e03, e05, e00 = m.e03, m.e05, m.e00
    n = int(meta["frames"])
    dt = 1000.0 / 24.0
    clips = meta["clips"]
    pos = {c: i for i, c in enumerate(wl.names)}
    n_first = [int(x) for x in np.asarray(arr["n_clip"])[:len(clips)]]
    labels_stream = [wl.labels[pos[c]][:n_c] for c, n_c in zip(clips, n_first)] * int(meta["passes"])
    cut = np.cumsum([len(x) for x in labels_stream])[:-1]
    online = arr["series"]
    ser_online = list(np.split(online, cut))
    auc_online = 100 * e00.pooled_auc(labels_stream, ser_online, None)
    boot_pt, boot_v = e03.AucBoot(labels_stream, None, N_BOOT, 0).aucs(ser_online)
    ser_off, _ = e05.vm_delivery_offline(arr, meta)
    spec = dict(e05.ARMS[E05_SIM_SPEC[meta["mode"]]], name=arm)
    ids = [pos[c] for c in clips] * int(meta["passes"])
    c = wl.cost(1.0)
    pol = e05.make_policy5(wl, c, spec, float(meta["period"]))
    R = e03.simulate(wl, c, pol, ids, light=False)
    ser_sim, _, _ = e03.delivery(wl, R, int(meta["hold"]))
    auc_sim = 100 * e00.pooled_auc(labels_stream, [x[:n_c] for x, n_c in zip(ser_sim, n_first * int(meta["passes"]))], None)
    full_len = [int(wl.n_frames[i]) for i in ids]
    wait_sim = np.concatenate([x[:n_c] for x, n_c in zip(np.split(R["wait"], np.cumsum(full_len)[:-1]), n_first * int(meta["passes"]))])
    wait, svc, obs, reused = arr["wait_ms"], arr["svc_ms"], arr["obs"].astype(bool), arr["reused"].astype(bool)
    tel = meta["telemetry"]
    marks = {int(k): float(v) for k, v in meta["rss_marks_mb"].items()}
    ks = sorted(marks)
    ev = [0, 720, 1440, 2880]
    rss_tab = {str(k): marks.get(k) for k in ev}
    base_rss = marks[ks[0]]
    growth = {str(k): 100.0 * (marks[k] - base_rss) / base_rss for k in ks}
    q = lambda x, p: float(np.quantile(x, p))  # noqa: E731
    return {
        "arm": arm, "mode": meta["mode"], "period": meta["period"], "hold": meta["hold"], "frames": n, "stream_seconds": n / 24.0, "wall_s": meta["wall_total_s"],
        "achieved_fps": meta["achieved_fps"], "offered_fps": 24.0, "dt_ms": dt, "real_time_ok": bool(meta["achieved_fps"] >= 23.5),
        "wait_ms": {"p50": q(wait, 0.5), "p95": q(wait, 0.95), "p99": q(wait, 0.99), "max": float(wait.max()), "mean": float(wait.mean())},
        "service_ms": {"mean": float(svc.mean()), "p95": q(svc, 0.95), "p99": q(svc, 0.99)}, "cpu_ms_per_frame": float(arr["cpu_ms"].mean()),
        "cpu_ms_per_video_s": float(arr["cpu_ms"].sum() / (n / 24.0)), "rss_marks_mb": rss_tab, "rss_marks_all": {str(k): v for k, v in marks.items()}, "rss_growth_pct": growth,
        "rss_growth_total_pct": float(100.0 * (marks[ks[-1]] - base_rss) / base_rss), "power_total_w": tel["power_total_w"], "power_core_w": tel["power_core_w"],
        "energy_j_per_frame": meta["energy_j_per_frame"], "energy_j_per_frame_check": float(tel["power_total_w"] * meta["wall_total_s"] / n),
        "temp_c_start": meta["temp_c_start"], "temp_c_max": tel["temp_c_max"], "temp_c_end": tel["temp_c_end"], "freq_mhz": tel["freq_mhz"], "freq_mhz_min": tel["freq_mhz_min"],
        "temp_trajectory": meta["temp_trajectory"], "throttled_start": meta["throttled_start"], "throttled_end": meta["throttled_end"], "throttled_codes": tel["throttled_codes"],
        "throttled_any": tel["throttled_any"], "c1_hit_rate_of_inrun": float(reused.sum() / max(1, obs.sum())), "observed_fraction": float(obs.mean()),
        "shed_fraction": float(arr["shed"].mean()), "guard_transitions": int(meta["guard_transitions"]), "windows_scored": int(meta["windows_scored"]),
        "dropped_late": meta["dropped_late"], "dropped_end": meta["dropped_end"], "auc_online": float(auc_online),
        "auc_online_ci95": [float(np.nanquantile(boot_v, 0.025)), float(np.nanquantile(boot_v, 0.975))], "auc_sim_pi_pools": float(auc_sim),
        "online_equals_offline": bool(np.allclose(online, ser_off, rtol=0, atol=1e-9)), "series_finite": bool(np.isfinite(online).all()),
        "sim": {"wait_p50_ms": q(wait_sim, 0.5), "wait_p95_ms": q(wait_sim, 0.95), "wait_p99_ms": q(wait_sim, 0.99), "wait_max_ms": float(wait_sim.max()),
                "cpu_ms_per_video_s": float(e03.total_cost(wl, R)), "observed_fraction": float(R["obs"].mean()),
                "c1_reuse_fraction_of_inrun": R["stats"]["reuse_frames"] / max(1, R["stats"]["run_frames"])},
        "wait_trace_100f_mean_ms": [float(wait[a:a + 100].mean()) for a in range(0, n, 100)], "mlflow_run_id": meta.get("mlflow_run_id"),
        "svc_by_quarter_ms": [float(svc[i].mean()) for i in np.array_split(np.arange(n), 4)],
    }


def mac_simulate(args, log=print) -> None:
    """Pi-calibrated full-dataset simulations (107 SHT clips, 211 UBnormal videos) + the live analysis; writes `sim/analysis.json`."""
    m = sim_setup()
    SIM_DIR.mkdir(parents=True, exist_ok=True)
    ctx = m.e01.build_ctx(args.procs)
    an: dict = {"datasets": {}, "live": {}, "n_boot": N_BOOT}
    cids = [m.e05.cell_id(*c) for c in m.e05.CELLS_NATIVE]
    for ds in DATASETS_ALL:
        run5 = sim_e05(m, ctx, ds, args.procs, log, args.force)
        run4 = sim_e04(m, ctx, ds, run5, args.procs, log, args.force)
        wl = m.e05.make_worlds(ctx, ds, m.e03.load_config(), which=("ml0_locked",))["ml0_locked"]
        t0 = time.perf_counter()
        boots5 = auc_boot_table(m, wl, run5["results"])
        boots4 = auc_boot_table(m, wl, {**run4["results"], **{k: v for k, v in run5["results"].items() if k[0] in ("Cache-C1", "Fixed")}})
        log(f"[{_now()}] bootstrap {ds}: {len(boots5) + len(boots4)} series x {N_BOOT} resamples in {time.perf_counter() - t0:.0f}s")
        arms5 = list(m.e05.ARMS)
        arms4 = sorted({a for a, _ in run4["results"]})
        an["datasets"][ds] = {
            "n_videos": len(wl.names), "n_frames": int(wl.total), "w_ref": run5["w_ref"], "targets": run5["targets"], "pi_costs": run5["costs"],
            "e05_arms": arm_cells_summary(run5["results"], boots5, wl.n_frames, arms5, cids),
            "e04_arms": arm_cells_summary(run4["results"], boots4, wl.n_frames, arms4, cids),
            "contrasts": contrast_table(m, boots5, SIM_CONTRASTS, cids),
            "e04_contrasts": contrast_table(m, boots4, E04_CONTRASTS, cids),
            "e04_cost_table": run4["cost_table"], "e04_derived_costs": run4["derived_costs"], "e04_pose_drift": run4["pose_drift"],
        }
        if ds == SHT_DS:
            for arm in E05_ORDER:
                r = analyze_live(m, wl, arm, log)
                if r:
                    an["live"][arm] = r
                    log(f"[{_now()}] live {arm}: {r['achieved_fps']:.2f} FPS, wait p99 {r['wait_ms']['p99']:.0f} ms, AUC online {r['auc_online']:.2f} sim {r['auc_sim_pi_pools']:.2f}")
    (SIM_DIR / "analysis.json").write_text(json.dumps(plain(an), indent=1))
    log(f"[{_now()}] wrote {SIM_DIR / 'analysis.json'}")


# ---- lock: plan.md + config.yaml ------------------------------------------------------------------------------------------------------------
PLAN_MD = """# Q2-P1-Pi5-Suite: physical Raspberry Pi 5 validation of E02, E04, E05

**Hardware.** Raspberry Pi 5 (4 x Cortex-A76 @ 2.4 GHz, 8 GB, active cooler), `nanh@raspberrypi.local`, `~/edgelab/.venv` (uv; torch, ultralytics, ncnn, onnxruntime, mlflow, psutil, av).
Only one process runs on the Pi at any time; every cell is its own process; the SoC is cooled to <= {cool} C before every cell and every live arm.

**Method.** The locked VM workers of E02 / E04 / E05 are re-used unchanged (cell file formats and ids identical), so the E02 / E03 / E04 / E05 simulators read the physical
cells through a symlink view and no VM-to-Pi multiplier is applied (native cell s = 1.0). A parent process samples `vcgencmd pmic_read_adc` (PMIC rail power), SoC temperature,
`get_throttled`, cpufreq and RSS every {tel} s and slices the samples to the timed window of the child (first / last timed work).

**Part 1 (E02).** Decode (JPEG `imdecode`; H.264 `grab`+`retrieve` on SHT H.264 re-encodes, UBnormal 1280x720 `.mp4`, plus 2 user SHT clips and 2 user UB clips), T in {{1,2,3,4}};
pose detector scaling (ML0 @640, MLP @640, stock @640 / @800 / @320 NCNN FP16; 480 natural L0 frames), T in {{1,2,3,4}}; BoT-SORT Default vs Locked two-stage (T=1) over person counts;
STG-NF window assembly + per-window normalisation + PyTorch forward, N in {{1,3,5,12,24}}, T in {{1,2,3,4}}; end-to-end service cells.
**Part 2 (E04).** ML0 NCNN FP16 vs ORT INT8 vs ORT FP32, T in {{1,2,3,4}}, 2 replicates (Cortex-A76: `asimddp`, no `i8mm`); H.264 motion-vector export / MV grid / MV C1 cost vs pixel-diff C1 cost (`mvprof`).
**Part 3 (E05).** 2,880-frame live end-to-end streams at 24.0 fps real-time arrival (dt = 41.67 ms), four arms: Fixed-K1; Cache-C1; Cache-C1+Guard (on > 400 ms, off < 150 ms); Cache-C1+Guard-Burst32
(intact 24-frame burst every W = 32). MLflow metrics every 60 frames, `@mlflow.trace` spans on frames 0 / 720 / 1440 / 2160 / 2879.

**Pi-calibrated simulations (Mac).** Full datasets (107 SHT clips, 211 UBnormal videos) with the measured Pi pools: E05 ladder (K=1 ladder = E02, Fixed / Cache-C1 / Guard / Stock = E03 / E05),
E04 arms (+MV, +INT8, +MV+INT8) at the native cells s=1.0, beta in {{0.25, 0.5, 1.0}}; exact pooled AUC, {boot}-resample video-level paired bootstrap.

**Outputs.** `config.yaml`, `results.csv`, `stage_timings_pi5.csv`, `summary.json`, `report.md`; raw traces in `~/thesis-data/pi5_suite/`.
"""


def mac_lock(log=print) -> None:
    import hashlib

    import yaml

    EXP_PI5.mkdir(parents=True, exist_ok=True)
    (EXP_PI5 / "plan.md").write_text(PLAN_MD.format(cool=COOL_TARGET_C, tel=TELEMETRY_INTERVAL_S, boot=N_BOOT))
    board = json.loads((MAC_OUT / "board.json").read_text()) if (MAC_OUT / "board.json").exists() else {}
    idle = json.loads((MAC_OUT / "idle.json").read_text()) if (MAC_OUT / "idle.json").exists() else {}
    cfgs = {p.stem: json.loads(p.read_text()) for p in sorted((MAC_SUITE / "configs").glob("*.json")) if not p.stem.endswith("_smoke")}
    src = Path(__file__).read_bytes()
    cfg = {"experiment": "Q2-P1-Pi5-Suite", "board": board, "idle_floor": idle, "module_sha256": hashlib.sha256(src).hexdigest(),
           "telemetry": {"interval_s": TELEMETRY_INTERVAL_S, "power": "vcgencmd pmic_read_adc, sum of rail V x I (PMIC, not wall)", "cool_target_c": COOL_TARGET_C, "thermal": "/sys/class/thermal/thermal_zone0/temp"},
           "mlflow": {"tracking_uri": TRACKING_URI, "experiments": [EXP_E02, EXP_E04, EXP_E05]}, "threads": list(THREADS_ALL), "pose_step": POSE_STEP, "score_n": list(SCORE_N_REPORT),
           "e05": {"frames": E05_FRAMES, "fps": 24.0, "arms": {k: list(v) for k, v in E05_ARMS.items()}, "rss_marks": list(E05_MARKS), "trace_frames": list(E05_TRACE_FRAMES), "configs": cfgs},
           "cells": [{k: v for k, v in c.items()} for c in cell_specs("all")], "bootstrap": {"unit": "video", "resamples": N_BOOT, "seed": 0},
           "native_cells": [[1.0, 0.25], [1.0, 0.5], [1.0, 1.0]], "limitations": LIMITATIONS}
    (EXP_PI5 / "config.yaml").write_text("# Q2-P1-Pi5-Suite configuration (generated by `q2_p1_pi5_suite lock`)\n" + yaml.safe_dump(plain(cfg), sort_keys=False, width=140))
    log(f"[{_now()}] lock: wrote plan.md and config.yaml in {EXP_PI5}")


# ---- report ---------------------------------------------------------------------------------------------------------------------------------
def _f(x, d: int = 2) -> str:
    return "n/a" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:.{d}f}"


def md(headers: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(str(x) for x in r) + " |" for r in rows]
    return "\n".join(out)


def where(rows: list[dict], **kw) -> list[dict]:
    return [r for r in rows if all(r.get(k) == v for k, v in kw.items())]


def avg_reps(rows: list[dict]) -> list[dict]:
    """Average replicates (`_rep1` cells) of the same (stage, config, threads, n)."""
    groups: dict = {}
    for r in rows:
        groups.setdefault((r["part"], r["stage"], r["config"], r["threads"], str(r.get("n", "")), r.get("dataset", "")), []).append(r)
    out = []
    for _, g in groups.items():
        r = dict(g[0])
        for k in ("wall_mean_ms", "wall_p50_ms", "wall_p95_ms", "wall_p99_ms", "cpu_ms_mean", "power_total_w", "power_core_w", "energy_j_per_frame", "active_energy_j_per_frame",
                  "temp_c_mean", "temp_c_max", "peak_rss_mb"):
            v = [x[k] for x in g if isinstance(x.get(k), (int, float)) and math.isfinite(x[k])]
            r[k] = float(np.mean(v)) if v else float("nan")
        r["reps"] = len(g)
        out.append(r)
    return out


def stage_cols(r: dict) -> list:
    return [_f(r["wall_mean_ms"], 2), _f(r["wall_p50_ms"], 2), _f(r["wall_p95_ms"], 2), _f(r["wall_p99_ms"], 2), _f(r["cpu_ms_mean"], 2), _f(r["peak_rss_mb"], 0), _f(r["power_total_w"], 2),
            _f(r["power_core_w"], 2), _f(r["energy_j_per_frame"], 4), _f(r["temp_c_mean"], 1)]


STAGE_HEAD = ["wall mean ms", "p50", "p95", "p99", "CPU ms", "peak RSS MB", "PMIC W", "core W", "J/frame", "T C"]


def ci_s(row: dict, sub: str) -> str:
    k = f"auc_{sub}"
    return "n/a" if k not in row else f"{row[k]:.2f} [{row[k + '_lo']:.2f}, {row[k + '_hi']:.2f}]"


def fci(c: dict | None) -> str:
    if not c:
        return "n/a"
    star = "*" if (c["lo"] > 0 or c["hi"] < 0) else ""
    return f"{c['delta']:+.2f} [{c['lo']:+.2f}, {c['hi']:+.2f}]{star}"


def render_report(rows: list[dict], an: dict, board: dict, idle: dict) -> str:
    L: list[str] = []
    add = L.append
    add("# Q2-P1-Pi5-Suite: physical Raspberry Pi 5 results (E02 stage profiling, E04 +MV / +INT8, E05 sustained live operation)\n")
    add(f"*Board:* {board.get('board_model', '?')}; kernel {board.get('kernel', '?')}; governor `{board.get('cpu_governor', '?')}`; idle floor {_f(idle.get('idle_power_w'), 3)} W total / "
        f"{_f(idle.get('idle_power_core_w'), 3)} W core at {_f(idle.get('idle_temp_c'), 1)} C. Every number below was measured on the physical Pi unless it is labelled "
        f"*Pi-calibrated simulation* (full datasets replayed with the measured Pi stage pools, no VM multiplier).\n")
    add("## Summary of results\n")
    e02s = avg_reps([r for r in rows if r["part"] == "e02"])
    e04s = avg_reps([r for r in rows if r["part"] == "e04"])

    def _w(rs, **kw):
        w = where(rs, **kw)
        return w[0]["wall_mean_ms"] if w else float("nan")

    p1, p4 = _w(e02s, stage="pose_detector", config="ML0-640", threads=1), _w(e02s, stage="pose_detector", config="ML0-640", threads=4)
    add(f"- **E02 (physical stage profile).** The pose detector dominates: ML0 @640 NCNN FP16 takes {_f(p1)} ms/frame at T=1 and {_f(p4)} ms at T=4 ({_f(p1 / p4)}x speed-up, "
        f"saturating after T=2-3); JPEG decode is ~{_f(_w(e02s, stage='decode_jpeg_imdecode', threads=1))} ms, the tracker ~1 ms and STG-NF 4-20 ms per job. "
        f"Full-rate 24 fps is out of reach at T=1 (pose alone is {_f(1000 / p1 if p1 else float('nan'), 1)} FPS).")
    ort = {r["config"]: r for r in where(e04s, stage="pose_detector", threads=4)}
    if {"ML0-640-ncnn", "ML0-640-ort-int8", "ML0-640-ort-fp32"} <= set(ort):
        n, i8, f32 = (ort[k]["wall_mean_ms"] for k in ("ML0-640-ncnn", "ML0-640-ort-int8", "ML0-640-ort-fp32"))
        add(f"- **E04 (+INT8 / +MV on Cortex-A76).** At T=4: NCNN FP16 {_f(n)} ms, ORT INT8 {_f(i8)} ms ({_f(n / i8)}x), ORT FP32 {_f(f32)} ms ({_f(n / f32)}x); INT8 is slower than NCNN at "
            f"T=1-2 and only matches it at T=3-4, with no J/frame gain. MV-derived C1 / observer cost is 2.6-5x cheaper than the pixel path (see section 2.1).")
    live = an.get("live", {})
    if live:
        add("- **E05 (2,880-frame live streams at 24 fps, T=4, no thermal throttling in any run).** " + "; ".join(
            f"{a}: {_f(live[a]['achieved_fps'])} FPS, wait p99 {live[a]['wait_ms']['p99']:.0f} ms, {_f(live[a]['energy_j_per_frame'], 3)} J/frame, online AUC {_f(live[a]['auc_online'])}%"
            for a in E05_ORDER if a in live) + ". Only the Guard arms keep up with real time on the physical Pi; RSS growth stays within about 3.5 % over 2,880 frames.")
    add("- **Pi-calibrated full-dataset simulation**: see section 4 (exact pooled AUC, 2,000-resample video bootstrap).\n")
    add("## 0. Honest accounting\n")
    for s in LIMITATIONS:
        add(f"- {s}")
    # ---- E02 -------------------------------------------------------------------------------------------------------------
    add("\n## 1. E02 stage profiling (physical Pi 5)\n")
    e02rows = avg_reps([r for r in rows if r["part"] == "e02"])
    add("### 1.1 Decode (per frame; JPEG `imdecode` vs H.264 `grab` + `retrieve`)\n")
    trs = []
    for stage in ("decode_jpeg_imdecode", "decode_h264_grab+retrieve", "decode_h264_grab", "decode_h264_retrieve"):
        for r in sorted(where(e02rows, stage=stage), key=lambda r: (r["config"], r["threads"])):
            trs.append([stage, r["config"], r["dataset"], r["threads"], *stage_cols(r)])
    add(md(["stage", "source", "dataset", "T", *STAGE_HEAD], trs))
    add("\n### 1.2 Pose detector thread scaling (NCNN FP16, 480 natural L0 SHT frames; UB model = 1,000+ UB frames)\n")
    trs = []
    base = {r["config"]: r["wall_mean_ms"] for r in where(e02rows, stage="pose_detector", threads=1)}
    for r in sorted(where(e02rows, stage="pose_detector"), key=lambda r: (r["config"], r["threads"])):
        sp = base.get(r["config"], float("nan")) / r["wall_mean_ms"] if r["wall_mean_ms"] else float("nan")
        trs.append([r["config"], r["threads"], *stage_cols(r), _f(sp, 2) + "x", r.get("note", "")[:70]])
    add(md(["model", "T", *STAGE_HEAD, "speedup vs T1", "pre / infer / post"], trs))
    add("\n### 1.3 Tracker BoT-SORT (T=1): Default (c=0.25, low off) vs Locked two-stage (c=0.20, low=0.05)\n")
    trs = []
    for r in sorted(where(e02rows, stage="tracker_botsort"), key=lambda r: (r["note"], r["config"], str(r["n"]).zfill(4))):
        if r["note"].startswith("real") and r["n"] != "all" and int(r["n"]) not in (1, 3, 5, 8, 12):
            continue
        trs.append([r["config"], r["note"], r["n"], r["frames"], *stage_cols(r)[:5], _f(r["power_total_w"], 2), _f(r["energy_j_per_frame"], 5)])
    add(md(["tracker", "source", "persons", "frames", "wall mean ms", "p50", "p95", "p99", "CPU ms", "PMIC W", "J/frame"], trs))
    add("\n### 1.4 Scorer STG-NF (window assembly + per-window normalisation + PyTorch forward)\n")
    trs = []
    for r in sorted(where(e02rows, stage="scorer_stgnf"), key=lambda r: (r["dataset"], r["threads"], int(r["n"]))):
        if int(r["n"]) in SCORE_N_REPORT and r["threads"] in (1, 4):
            trs.append([r["dataset"], r["threads"], r["n"], *stage_cols(r)[:5], _f(r["power_total_w"], 2), _f(r["energy_j_per_frame"], 5), r["note"]])
    add(md(["dataset", "T", "N windows", "wall mean ms", "p50", "p95", "p99", "CPU ms", "PMIC W", "J/job", "assemble / normalize / flow"], trs))
    add("\n### 1.5 End-to-end service cells\n")
    trs = [[r["config"], r["threads"], *stage_cols(r), r["note"]] for r in sorted(where(e02rows, stage="e2e_service"), key=lambda r: (r["config"], r["threads"]))]
    add(md(["cell", "T", *STAGE_HEAD, "decode / pose / track / score ms"], trs))
    # ---- E04 -------------------------------------------------------------------------------------------------------------
    add("\n## 2. E04 on the Cortex-A76: NCNN FP16 vs ORT INT8 vs ORT FP32 (ML0 @640, 480 frames, mean of 2 replicates)\n")
    e04rows = avg_reps([r for r in rows if r["part"] == "e04"])
    pose4 = where(e04rows, stage="pose_detector")
    ncnn = {r["threads"]: r for r in where(pose4, config="ML0-640-ncnn")}
    trs = []
    for r in sorted(pose4, key=lambda r: (r["threads"], r["config"])):
        n = ncnn.get(r["threads"])
        rel = n["wall_mean_ms"] / r["wall_mean_ms"] if n else float("nan")
        erel = r["energy_j_per_frame"] / n["energy_j_per_frame"] if n else float("nan")
        trs.append([r["threads"], r["config"], *stage_cols(r), _f(rel, 2) + "x", _f(erel, 2) + "x"])
    add(md(["T", "runtime", *STAGE_HEAD, "speed vs NCNN", "J/frame vs NCNN"], trs))
    add("\n### 2.1 H.264 motion-vector export vs pixel-diff C1 (per frame, Pi CPU)\n")
    trs = []
    for ds_kind, dsn in (("sht", "ShanghaiTech 856x480 (x264 IPPP)"), ("ub", "UBnormal 1280x720")):
        for t in (1, 4):
            for r in sorted(where(e04rows, part="e04", threads=t, dataset=SHT_DS if ds_kind == "sht" else UB_DS), key=lambda r: r["config"]):
                if r["stage"].startswith("mvprof_"):
                    trs.append([dsn, t, r["config"], _f(r["wall_mean_ms"], 3), _f(r["wall_p95_ms"], 3), _f(r["cpu_ms_mean"], 3), _f(r["power_total_w"], 2)])
    add(md(["clips", "T", "method", "wall mean ms", "p95", "CPU ms", "PMIC W"], trs))
    for ds in DATASETS_ALL:
        d = an["datasets"].get(ds)
        if d:
            c = d["e04_derived_costs"]["1.0"]
            add(f"\n*{ds}* derived costs (Pi, margin 1.0): MV export overhead (decode_mv - decode_plain, T=4, every frame) {_f(c['export_wall_ms'], 3)} ms wall / {_f(c['export_cpu_ms'], 3)} ms CPU; "
                f"MV observer {_f(c['mv_obs_ms'], 3)} ms, MV C1 {_f(c['mv_c1_ms'], 3)} ms vs pixel observer {_f(c['pix_obs_ms_vm'], 3)} ms, pixel C1 {_f(c['pix_c1_ms_vm'], 3)} ms.")
    # ---- E05 live --------------------------------------------------------------------------------------------------------
    add("\n## 3. E05 sustained live operation (physical, 2,880 frames per arm, 24.0 fps real-time arrival, T=4)\n")
    live = an.get("live", {})
    if live:
        trs = []
        for a in E05_ORDER:
            r = live.get(a)
            if r:
                w = r["wait_ms"]
                trs.append([a, _f(r["achieved_fps"], 2), _f(w["p50"], 1), _f(w["p95"], 1), _f(w["p99"], 1), _f(w["max"], 1), _f(r["power_total_w"], 2), _f(r["power_core_w"], 2),
                            _f(r["energy_j_per_frame"], 4), _f(r["temp_c_start"], 1), _f(r["temp_c_max"], 1), f"{r['throttled_start']} -> {r['throttled_end']}"])
        add(md(["arm", "achieved FPS", "wait p50 ms", "p95", "p99", "max", "PMIC W", "core W", "J/frame", "T start C", "T max C", "throttled"], trs))
        add("\n### 3.1 RSS growth (MB at frame 0 / 720 / 1440 / 2880)\n")
        trs = []
        for a in E05_ORDER:
            r = live.get(a)
            if r:
                g = r["rss_growth_pct"]
                trs.append([a, *[_f(r["rss_marks_mb"].get(k), 1) for k in ("0", "720", "1440", "2880")], *[_f(g.get(k), 2) + " %" for k in ("720", "1440", "2880")]])
        add(md(["arm", "RSS 0", "RSS 720", "RSS 1440", "RSS 2880", "growth @720", "@1440", "@2880"], trs))
        add("\n### 3.2 Pipeline behaviour, online delivered AUC and the Pi-calibrated simulation of the same stream\n")
        trs = []
        for a in E05_ORDER:
            r = live.get(a)
            if r:
                s = r["sim"]
                trs.append([a, _f(r["observed_fraction"], 3), _f(r["c1_hit_rate_of_inrun"], 3), _f(r["shed_fraction"], 3), r["guard_transitions"], r["windows_scored"], r["dropped_late"],
                            f"{r['auc_online']:.2f} [{r['auc_online_ci95'][0]:.1f}, {r['auc_online_ci95'][1]:.1f}]", _f(r["auc_sim_pi_pools"], 2), _f(s["wait_p99_ms"], 0), _f(r["wait_ms"]["p99"], 0),
                            "yes" if r["online_equals_offline"] else "NO"])
        add(md(["arm", "observed frac.", "C1 hit rate (in-run)", "guard shed frac.", "guard on/off", "windows scored", "dropped late", "online AUC % [95% CI, 12 videos]", "sim AUC %",
                "sim wait p99 ms", "live wait p99 ms", "online == offline"], trs))
        add("\n### 3.3 SoC temperature trajectory (every 360 frames: frame, C, W)\n")
        trs = []
        for a in E05_ORDER:
            r = live.get(a)
            if r:
                trs.append([a, "; ".join(f"{p['frame']}: {p['temp_c']:.1f}C {p['power_w']:.2f}W" for p in r["temp_trajectory"])])
        add(md(["arm", "trajectory"], trs))
    else:
        add("_No live run found._")
    # ---- simulations -----------------------------------------------------------------------------------------------------
    add("\n## 4. Pi-calibrated full-dataset delivered AUC (simulation with the measured Pi stage pools; exact pooled AUC, 2,000-resample video bootstrap)\n")
    for ds in DATASETS_ALL:
        d = an["datasets"].get(ds)
        if not d:
            continue
        add(f"### 4.{1 + DATASETS_ALL.index(ds)} {ds}: {d['n_videos']} videos, {d['n_frames']:,} frames; native cells s = 1.0\n")
        add("*K=1 arms (Stock-Default-K1, ML0-Default-K1, Full-Rate) process every frame whatever the CPU budget: their value is identical in the three cells and shows the real-time "
            "delivered AUC of the unthrottled pipeline on the Pi (E02 question). Cost-matched arms (Fixed, Cache-C1, +Guard, ReTrack, Stock-Baseline, ML0-Default-Fixed) are calibrated to "
            "beta x the Full-Rate CPU cost of the cell (E03 / E05 questions).*\n")
        for sub, title in (("all", "all videos"), ("confirmation", "test_confirmation split")):
            add(f"**E02 / E03 / E05 arms, {title}** (AUC %, [95% CI]):\n")
            by = {}
            for r in d["e05_arms"]:
                by.setdefault(r["arm"], {})[r["cell"]] = r
            cids = ["s1.0_b0.25", "s1.0_b0.50", "s1.0_b1.00"]
            trs = [[a, *[ci_s(by[a][c], sub) if c in by[a] else "n/a" for c in cids]] for a in by]
            add(md(["arm", *[f"beta={c.split('_b')[1]}" for c in cids]], trs))
            add("")
        add("**Contrasts (A - B, AUC points, mean over the 3 native cells; * = 95% CI excludes 0):**\n")
        trs = [[c["contrast"], c["subset"], fci(c["mean_over_cells"]), *[fci(c["per_cell"].get(k)) for k in ("s1.0_b0.25", "s1.0_b0.50", "s1.0_b1.00")]] for c in d["contrasts"]]
        add(md(["contrast", "subset", "mean", "beta=0.25", "beta=0.50", "beta=1.00"], trs))
        add(f"\n**E04 arms (Pi pools + Pi-measured MV costs, {'+INT8 on the Cortex-A76 pool' if ds == SHT_DS else 'no INT8 cache for UB'}), all videos:**\n")
        by = {}
        for r in d["e04_arms"]:
            by.setdefault(r["arm"], {})[r["cell"]] = r
        trs = [[a, *[ci_s(by[a][c], "all") if c in by[a] else "n/a" for c in cids], _f(by[a].get("s1.0_b0.50", {}).get("cost_cpu_ms_per_video_s"), 0)] for a in by]
        add(md(["arm", "beta=0.25", "beta=0.50", "beta=1.00", "CPU ms/video-s (beta=0.5)"], trs))
        add("\n**E04 contrasts:**\n")
        trs = [[c["contrast"], c["subset"], fci(c["mean_over_cells"]), *[fci(c["per_cell"].get(k)) for k in ("s1.0_b0.25", "s1.0_b0.50", "s1.0_b1.00")]] for c in d["e04_contrasts"]]
        add(md(["contrast", "subset", "mean", "beta=0.25", "beta=0.50", "beta=1.00"], trs))
        add("")
    return "\n".join(L) + "\n"


def results_rows(rows: list[dict], an: dict) -> list[dict]:
    out = []

    def add(exp, src, ds, arm, cell, sub, metric, v, lo="", hi="", note=""):
        out.append({"experiment": exp, "source": src, "dataset": ds, "arm": arm, "cell": cell, "subset": sub, "metric": metric, "value": v, "ci_lo": lo, "ci_hi": hi, "note": note})

    for r in rows:
        exp = "E02" if r["part"] == "e02" else "E04"
        for k in ("wall_mean_ms", "wall_p50_ms", "wall_p95_ms", "wall_p99_ms", "cpu_ms_mean", "peak_rss_mb", "power_total_w", "power_core_w", "energy_j_per_frame", "temp_c_mean"):
            add(exp, "physical-pi5", r.get("dataset", ""), f"{r['stage']}:{r['config']}", r["cell"], f"T{r['threads']}" + (f"/n{r['n']}" if r.get("n", "") != "" else ""), k, r.get(k))
    for a, r in an.get("live", {}).items():
        for k in ("achieved_fps", "cpu_ms_per_video_s", "power_total_w", "power_core_w", "energy_j_per_frame", "temp_c_start", "temp_c_max", "temp_c_end", "rss_growth_total_pct",
                  "c1_hit_rate_of_inrun", "observed_fraction", "shed_fraction", "auc_sim_pi_pools"):
            add("E05", "physical-pi5-live", SHT_DS, a, "live", "2880f", k, r[k])
        for k, v in r["wait_ms"].items():
            add("E05", "physical-pi5-live", SHT_DS, a, "live", "2880f", f"queue_wait_{k}_ms", v)
        add("E05", "physical-pi5-live", SHT_DS, a, "live", "12 videos", "auc_online", r["auc_online"], r["auc_online_ci95"][0], r["auc_online_ci95"][1])
        for k, v in r["rss_marks_mb"].items():
            add("E05", "physical-pi5-live", SHT_DS, a, "live", f"frame{k}", "rss_mb", v)
    for ds, d in an["datasets"].items():
        for key, exp in (("e05_arms", "E02/E03/E05"), ("e04_arms", "E04")):
            for r in d[key]:
                for sub in ("all", "confirmation"):
                    if f"auc_{sub}" in r:
                        add(exp, "pi5-calibrated-sim", ds, r["arm"], r["cell"], sub, "delivered_auc", r[f"auc_{sub}"], r[f"auc_{sub}_lo"], r[f"auc_{sub}_hi"])
                add(exp, "pi5-calibrated-sim", ds, r["arm"], r["cell"], "all", "cpu_ms_per_video_s", r["cost_cpu_ms_per_video_s"])
        for key in ("contrasts", "e04_contrasts"):
            for c in d[key]:
                for cid, v in [("mean", c["mean_over_cells"]), *c["per_cell"].items()]:
                    add("E04" if key == "e04_contrasts" else "E02/E03/E05", "pi5-calibrated-sim", ds, c["contrast"], cid, c["subset"], "delta_auc", v["delta"], v["lo"], v["hi"])
    return out


def mac_report(log=print) -> None:
    EXP_PI5.mkdir(parents=True, exist_ok=True)
    rows = load_stage_rows()
    n = write_stage_csv(rows)
    an = json.loads((SIM_DIR / "analysis.json").read_text())
    board = json.loads((MAC_OUT / "board.json").read_text())
    idle = json.loads((MAC_OUT / "idle.json").read_text())
    res = results_rows(rows, an)
    cols = ["experiment", "source", "dataset", "arm", "cell", "subset", "metric", "value", "ci_lo", "ci_hi", "note"]
    with open(EXP_PI5 / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in res:
            w.writerow({c: (f"{r[c]:.6g}" if isinstance(r[c], float) else ("" if r[c] is None else r[c])) for c in cols})
    manifest = {p: [json.loads(x) for x in (MAC_OUT / p / "manifest.jsonl").read_text().splitlines()] for p in ("q2-p1-e02", "q2-p1-e04") if (MAC_OUT / p / "manifest.jsonl").exists()}
    summary = {"board": board, "idle_floor": idle, "limitations": LIMITATIONS, "stage_rows": n, "cells": {p: {"ok": sum(1 for r in v if r["status"] == "ok"), "failed": sum(1 for r in v if r["status"] != "ok")}
                                                                                        for p, v in manifest.items()},
               "e05_live": an["live"], "pi_calibrated": {ds: {k: v for k, v in d.items() if k not in ("e04_cost_table",)} for ds, d in an["datasets"].items()}, "bootstrap": {"resamples": N_BOOT, "unit": "video"},
               "mlflow": {"tracking_uri": TRACKING_URI, "experiments": [EXP_E02, EXP_E04, EXP_E05]},
               "throttled_any_cell": sorted({r["cell"] for r in rows if r.get("throttled_any")})}
    (EXP_PI5 / "summary.json").write_text(json.dumps(plain(summary), indent=1))
    (EXP_PI5 / "report.md").write_text(render_report(rows, an, board, idle))
    log(f"[{_now()}] report: {n} stage rows, {len(res)} result rows -> {EXP_PI5}")


def main_mac(args) -> None:
    cmd = args.cmd
    if cmd in ("sync", "all-mac"):
        mac_sync()
    if cmd in ("lock", "all-mac"):
        mac_lock()
    if cmd in ("view", "simulate", "all-mac"):
        build_view()
    if cmd in ("simulate", "all-mac"):
        mac_simulate(args)
    if cmd in ("report", "all-mac"):
        mac_report()


if __name__ == "__main__":
    main()
