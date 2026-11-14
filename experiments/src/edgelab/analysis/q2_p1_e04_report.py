"""Q2-P1-E04 analysis: contrasts, adoption decisions, correctness rules (E4-D1..D7), section-8 gates, results.csv / decisions.jsonl / summary.json / report.md.

Reads `~/thesis-data/q2-p1-e04/run_{dataset}.pkl` (written by `q2_p1_e04 run` against the locked `config.yaml`) and `quality.json`. Nothing is re-tuned here: the
adoption rule is the one written in plan.md section 4 and copied into config.yaml before any delivered-AUC number existed.
"""

from __future__ import annotations

import csv
import json
import math
import multiprocessing as mp
import pickle
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from edgelab.analysis import q2_p1_e00 as e00
from edgelab.analysis import q2_p1_e01 as e01
from edgelab.analysis import q2_p1_e02 as e02
from edgelab.analysis import q2_p1_e03 as e03
from edgelab.analysis import q2_p1_e04 as e04
from edgelab.analysis import q2_p1_e04_vm as vmm

DATASETS = e04.DATASETS
SUBSETS = e03.SUBSETS
DEGRADE_PT = 0.25
CPU_GAIN = 0.05
ARM_ORDER = ["C1-Pixel", "C1-MV", "C1-MV@W", "C1-MV-Advect", "Motion-Only-Pixel", "Motion-Only+MV", "C1-INT8", "C1-INT8@W", "C1-MV-INT8", "C1-MV-INT8@W",
             "C1-MV|cost1.0", "C1-MV|cost2.0", "C1-Pixel|L1", "C1-MV|L1", "C1-Pixel|L2", "C1-MV|L2", "C1-Pixel|vmcost", "C1-INT8|uncorrected"]
plain = e04.plain
fmt, md_table, ci_txt = e03.fmt, e03.md_table, e03.ci_txt


def now() -> str:
    return time.strftime("%H:%M:%S")


# =====================================================================================================================
# 1. Pure decision logic (unit-tested)
# =====================================================================================================================
def component_decision(a_dev: dict, a_conf: dict, b_auc_dev: dict, b_auc_conf: dict, cpu_dev: float, cpu_conf: float,
                       degrade: float = DEGRADE_PT, cpu_gain: float = CPU_GAIN) -> dict:
    """Pre-registered adoption rule of one component on one dataset (plan section 4).

    a_*: matched-cost AUC contrast {delta (point, cell average), lo, hi}; b_auc_*: AUC contrast at the identical schedule; cpu_*: cell-averaged CPU reduction (fraction)
    at the identical schedule. Path A: test_dev gain > 0 with CI excluding 0 and no degradation; confirmation point estimate >= 0 and no degradation. Path B: CPU
    reduction >= 5 % and no degradation on both splits."""
    a_dev_ok = a_dev["delta"] > 0 and a_dev["lo"] > 0 and a_dev["delta"] >= -degrade
    a_conf_ok = a_conf["delta"] >= 0 and a_conf["delta"] >= -degrade
    b_dev_ok = cpu_dev >= cpu_gain and b_auc_dev["delta"] >= -degrade
    b_conf_ok = cpu_conf >= cpu_gain and b_auc_conf["delta"] >= -degrade
    path_a, path_b = bool(a_dev_ok and a_conf_ok), bool(b_dev_ok and b_conf_ok)
    return {"path_a": {"adopt": path_a, "dev_ok": bool(a_dev_ok), "confirmation_ok": bool(a_conf_ok), "dev": a_dev, "confirmation": a_conf},
            "path_b": {"adopt": path_b, "dev_ok": bool(b_dev_ok), "confirmation_ok": bool(b_conf_ok), "auc_dev": b_auc_dev, "auc_confirmation": b_auc_conf,
                       "cpu_reduction_dev": cpu_dev, "cpu_reduction_confirmation": cpu_conf},
            "adopted": bool(path_a or path_b), "path": ("A" if path_a else ("B" if path_b else None))}


def component_verdict(per_ds: dict[str, dict]) -> str:
    """Dataset-conditional wording: adopted everywhere evaluated / on one dataset only / not adopted."""
    ok = [ds for ds, d in per_ds.items() if d["adopted"]]
    if not ok:
        return "not adopted"
    if len(ok) == len(per_ds):
        return "adopted" + ("" if len(per_ds) > 1 else f" (evaluated on {ok[0]} only)")
    return f"dataset-conditional: adopted on {', '.join(ok)} only (not generalised)"


def qat_gate(deg_s0: float, deg_s4: float, deg_delivered: float, cpu_reduction: float, margin: float = DEGRADE_PT) -> dict:
    """Section 8: QAT only if INT8 degrades quality (S0 / S4 or delivered AUC) by more than `margin` pt while saving CPU (a speed-up worth rescuing)."""
    degraded = min(deg_s0, deg_s4, deg_delivered) < -margin
    return {"fires": bool(degraded and cpu_reduction > 0), "degraded": bool(degraded), "cpu_reduction": cpu_reduction,
            "deltas_pt": {"S0": deg_s0, "S4": deg_s4, "delivered": deg_delivered}}


def hist_stats(h: np.ndarray) -> dict:
    h = np.asarray(h, dtype=np.float64)
    if h.sum() <= 0:
        return {"n": 0, "mean": float("nan"), "p90": float("nan")}
    centers = (np.arange(len(h)) + 0.5) * 0.01
    cdf = np.cumsum(h) / h.sum()
    return {"n": int(h.sum()), "mean": float((h * centers).sum() / h.sum()), "p90": float(centers[np.searchsorted(cdf, 0.9)])}


def agreement(a: np.ndarray, b: np.ndarray) -> dict:
    """Agreement of two boolean decisions on the frames where both exist (NaN-free mask passed by the caller)."""
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    n = len(a)
    po = float((a == b).mean()) if n else float("nan")
    pe = float(a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())) if n else float("nan")
    return {"n": int(n), "agree": po, "kappa": float((po - pe) / (1 - pe)) if n and pe < 1 else float("nan"), "pixel_static": float(a.mean()) if n else float("nan"),
            "mv_static": float(b.mean()) if n else float("nan"), "both_static": float((a & b).mean()) if n else float("nan"),
            "mv_static_pixel_moving": float((~a & b).mean()) if n else float("nan"), "pixel_static_mv_moving": float((a & ~b).mean()) if n else float("nan")}


# =====================================================================================================================
# 2. Dataset analysis
# =====================================================================================================================
def build_world(ctx, ds: str) -> e03.World:
    arm = e01.arm_by_key(ctx.arms)[f"{ds}/{e04.PRIMARY[ds]}"]
    return e03.World.from_parts(ds, ctx.names[ds], ctx.labels[ds], ctx.dev_idx[ds], ctx.conf_idx[ds], e01.arm_sentinel(ctx.proto, arm), False, e03.load_tables_npz(ds), None)


class Study:
    """E03 `Analysis` + the E04 contrasts, CPU accounting and decisions of one dataset."""

    def __init__(self, ctx, ds: str, run: dict, cfg03: dict):
        self.ds, self.run = ds, run
        self.w = build_world(ctx, ds)
        self.A = e03.Analysis(self.w, run, cfg03)
        self.cids = self.A.cids
        self.arms = [a for a in ARM_ORDER if (a, self.cids[0]) in run["results"]]

    def has(self, arm: str) -> bool:
        return (arm, self.cids[0]) in self.run["results"]

    def con(self, a: str, b: str, sub: str) -> dict:
        c = self.A.contrast_all(a, b, sub)
        pt = self.A.avg_auc(a, sub) - self.A.avg_auc(b, sub)
        pos = sum(self.A.boot(a, cid, sub)[0] > self.A.boot(b, cid, sub)[0] for cid in self.cids)
        return {"delta": float(pt), "lo": c["lo"], "hi": c["hi"], "excludes_zero": c["excludes_zero"], "cells_positive": int(pos), "cells": len(self.cids),
                "a": self.A.avg_auc(a, sub), "b": self.A.avg_auc(b, sub)}

    def cpu_reduction(self, new: str, base: str, sub: str) -> dict:
        per = {cid: 1.0 - self.A.cost_sub(new, cid, sub)["cpu_ms_per_s"] / self.A.cost_sub(base, cid, sub)["cpu_ms_per_s"] for cid in self.cids}
        v = np.array(list(per.values()))
        return {"mean": float(v.mean()), "min": float(v.min()), "max": float(v.max()), "per_cell": per,
                "ms_per_s_new": float(np.mean([self.A.cost_sub(new, c_, sub)["cpu_ms_per_s"] for c_ in self.cids])),
                "ms_per_s_base": float(np.mean([self.A.cost_sub(base, c_, sub)["cpu_ms_per_s"] for c_ in self.cids]))}

    def decide(self, a_new: str, a_base: str, b_new: str, b_base: str) -> dict:
        d = component_decision(self.con(a_new, a_base, "dev"), self.con(a_new, a_base, "confirmation"), self.con(b_new, b_base, "dev"), self.con(b_new, b_base, "confirmation"),
                               self.cpu_reduction(b_new, b_base, "dev")["mean"], self.cpu_reduction(b_new, b_base, "confirmation")["mean"])
        d["arms"] = {"A": [a_new, a_base], "B": [b_new, b_base]}
        d["all"] = {"auc_A": self.con(a_new, a_base, "all"), "auc_B": self.con(b_new, b_base, "all"), "cpu_B": self.cpu_reduction(b_new, b_base, "all")}
        return d

    def account(self, arm: str, cid: str) -> dict:
        """CPU ms per video second: realised cost, saved work, overheads (stream totals, scenario multiplier included)."""
        r = self.A.r(arm, cid)
        S = r["stats"]
        span = self.w.total * self.w.dt / 1000.0
        saved = S["saved_pose_ms"] + S["saved_score_ms"] + S.get("retrieve_saved_ms", 0.0)
        over = S["c1_check_ms"] + S["c2_check_ms"] + S.get("mv_export_ms", 0.0) + S.get("obs_ms", 0.0)
        return {"cost": r["cost"], "saved_ms_per_s": saved / span, "overhead_ms_per_s": over / span, "export_ms_per_s": S.get("mv_export_ms", 0.0) / span,
                "c1_check_ms_per_s": S["c1_check_ms"] / span, "retrieve_saved_ms_per_s": S.get("retrieve_saved_ms", 0.0) / span, "pose_ms_per_s": S["pose_ms"] / span,
                "decode_ms_per_s": S["decode_ms"] / span, "reuse_rate_pct": 100.0 * S["reuse_frames"] / max(S["run_frames"], 1), "detect_frames": S["detect_frames"],
                "run_frames": S["run_frames"], "knob": r["knob"], "status": r["status"], "ratio": r["ratio"]}

    def cell_mean(self, arm: str, key: str) -> float:
        return float(np.mean([self.account(arm, cid)[key] for cid in self.cids]))


def max_reuse_run(tb: dict, ds: str) -> int:
    best_all = 0
    for lv in range(len(e03.LEVELS)):
        for b_ in tb["bits"][:: max(1, len(tb["bits"]) // 3000), lv]:
            run = best = 0
            for i in range(e03.RMAX[ds]):
                if (int(b_) >> i) & 1:
                    run += 1
                    best = max(best, run)
                else:
                    run = 0
            best_all = max(best_all, best)
    return best_all


# =====================================================================================================================
# 3. Correctness rules E4-D1 .. D7
# =====================================================================================================================
def _cmp_npz(a: Path, b: Path) -> bool:
    """Content equality of two npz files: NaN == NaN, and host wall-clock timing fields (`*_ms`) are excluded (they are measurements, not outputs)."""
    za, zb = np.load(a), np.load(b)
    keys = [k for k in za.files if not k.endswith("_ms")]
    return set(za.files) == set(zb.files) and all(np.array_equal(za[k], zb[k], equal_nan=za[k].dtype.kind == "f") for k in keys)


def rule_d1(runs: dict, tables_check: dict) -> dict:
    rows, ok = {}, True
    for ds in DATASETS:
        d1 = runs[ds]["d1"]
        for arm, cells in d1.items():
            good = all(c["series_equal"] and c["ideal_equal"] and c["cost_equal"] and c["knob_equal"] and c["target_equal"] for c in cells.values())
            rows[f"{ds}/{arm}"] = {"cells": len(cells), "bit_exact": bool(good), "max_abs_series_diff": max(c["max_abs_series_diff"] for c in cells.values())}
            ok &= good
        ok &= all(runs[ds]["e03_targets_equal"].values())
        tc = tables_check[ds]["pixrebuild_vs_e03"]
        tab_ok = all(v["equal"] or (k in ("row_norm", "row_step", "row_qual") and v.get("max_abs_diff", 1) <= 1e-6) for k, v in tc.items() if k != "all_equal")
        rows[f"{ds}/tables_rebuild"] = {"all_equal": tc["all_equal"], "within_1e-6": bool(tab_ok), "max_abs_diff": {k: v.get("max_abs_diff") for k, v in tc.items() if k != "all_equal"}}
        ok &= tab_ok
    return {"pass": bool(ok), "rows": rows, "detail": "E03 Cache-C1 and Motion-Only replayed through the extended simulator in 12 cells x 2 datasets: delivered series, ideal series, cost, "
                                                       "knob and target bit-identical; the rebuilt run tables equal E03's"}


def _translation_check() -> dict:
    """Encode a textured clip with a known global translation (4 px / frame to the right), read the exported vectors back (spawned process: PyAV only)."""
    return mp.get_context("spawn").Pool(1).apply(_translation_job)


def _translation_job() -> dict:
    from pathlib import Path as P
    import tempfile

    rng = np.random.default_rng(0)
    big = (rng.random((144, 400)) * 255).astype(np.uint8)
    big = np.kron(big, np.ones((2, 2), dtype=np.uint8))  # smooth texture (blocky) so that motion search is meaningful
    frames = []
    for i in range(24):
        f = big[:, 4 * i: 4 * i + 256][:288]
        frames.append(np.stack([f, f, f], axis=-1)[:240])
    with tempfile.TemporaryDirectory() as d:
        out = P(d) / "t.mp4"
        vmm.encode_frames(frames, "ShanghaiTech", out, 256, 240)
        r = vmm.extract_clip(out)
    P_ = r["valid"]
    # content moves LEFT in the image when the window slides right: displacement (dst - src) is -4 px/frame in x
    dx = r["dx"][P_].astype(np.float32)
    cov = r["cov"][P_].astype(np.float32) / 255.0
    sel = cov >= 0.9
    return {"median_dx": float(np.median(dx[sel])), "median_dy": float(np.median(r["dy"][P_].astype(np.float32)[sel])), "frames_valid": int(P_.sum()), "frames": int(len(P_))}


def _encode_jobs_repeat(jobs: list) -> list:
    return mp.get_context("spawn").Pool(min(3, len(jobs))).map(vmm.encode_job, jobs, chunksize=1)


def rule_d2(ctx) -> dict:
    # (a) one MV record per labelled frame (audit of the whole cache)
    bad = []
    for ds in DATASETS:
        for c, lab in zip(*e00.dataset_clips(ds)):
            z = np.load(e04.mv_path(ds, c))
            if len(z["valid"]) != len(lab):
                bad.append((ds, c))
    # (b) known translation
    tr = _translation_check()
    ok_tr = abs(abs(tr["median_dx"]) - 4.0) <= 1.0 and abs(tr["median_dy"]) <= 0.5
    # (c) determinism: re-encode 3 clips
    jobs = []
    picks = [("ShanghaiTech", e00.dataset_clips("ShanghaiTech")[0][0]), ("ShanghaiTech", e00.dataset_clips("ShanghaiTech")[0][50]), ("UBnormal", e00.dataset_clips("UBnormal")[0][0])]
    tmp = e04.WORK_DIR / "tmp_d2"
    for ds, c in picks:
        src = e04.SHT_FRAMES_ROOT / c if ds == "ShanghaiTech" else e00.UB_TEST_VIDEOS / f"{c}.mp4"
        jobs.append((ds, c, str(src), str(tmp / ds / f"{c}.mp4"), str(tmp / ds / f"{c}.npz")))
    _encode_jobs_repeat(jobs)
    same = {j[1]: _cmp_npz(Path(j[4]), e04.mv_path(j[0], j[1])) for j in jobs}
    return {"pass": bool(not bad and ok_tr and all(same.values())), "frame_count_mismatches": bad, "translation": tr, "translation_ok": bool(ok_tr), "reencode_identical": same,
            "detail": "one MV record per labelled frame in every clip; a known global translation of 4 px / frame is recovered from the exported vectors; re-encoding 3 clips gives identical grids and statistics (NaN == NaN)"}


def rule_d3() -> dict:
    rng = np.random.default_rng(1)
    ok = {}
    prev, cur = (rng.random((120, 214)) * 255).astype(np.uint8), (rng.random((120, 214)) * 255).astype(np.uint8)
    ok["motion_observer"] = bool(vmm.motion_observer(prev, cur) == e03.motion_observer(prev, cur))
    boxes = np.array([[100.0, 80.0, 180.0, 300.0], [400.0, 100.0, 460.0, 330.0]])
    ok["box_cells"] = bool(np.array_equal(vmm.box_cells_grid(boxes, 856, 480, e03.GRID_H, e03.GRID_W, 1), e03.box_cells(boxes, 856, 480)))
    d = np.abs(rng.random((e03.GRID_H, e03.GRID_W)).astype(np.float32))
    mask = e03.box_cells(boxes, 856, 480)
    ok["roi_check"] = bool(vmm.roi_check(d, mask) == (float(d[mask].mean()), float(np.percentile(d[~mask], 95))))
    ok["luma_small"] = bool(np.array_equal(vmm.luma_small(prev), e03.luma_small(prev)))
    return {"pass": all(ok.values()), "checks": ok, "detail": "numpy copies of the E03 observers used in the VM equal the E03 functions"}


def rule_d4(cfg: dict) -> dict:
    v = cfg["vm"]
    ok = v["pose_drift_vs_e02"]["pass"] and v["host_outside_vm_pct"]["max"] < 30.0 and not v["excluded"]
    return {"pass": bool(ok), "pose_max_wall_deviation_pct": v["pose_drift_vs_e02"]["max_abs_wall_deviation_pct"], "host_outside_vm_pct": v["host_outside_vm_pct"],
            "excluded_cells": v["excluded"], "detail": "E04-session ML0 NCNN pose pool within 10 % of E02's (all T); host CPU outside the VM < 30 %; no cell excluded after retries"}


def rule_d5() -> dict:
    from edgelab.pipeline import quantize_int8

    picks = quantize_int8.calibration_frames(e04.SHT_TRAIN_VIDEOS, 200, 0)
    listed = [tuple(x.split()) for x in e04.ML0_INT8_ONNX.with_name(e04.ML0_INT8_ONNX.name + ".calib.txt").read_text().split("\n") if x.strip()]
    same_list = [(v, str(i)) for v, i in picks] == listed
    train = {p.name for p in e04.SHT_TRAIN_VIDEOS.iterdir()}
    from_train = all(v in train for v, _ in picks)
    tmp = e04.WORK_DIR / "tmp_d5" / "ML0-int8-repro.onnx"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    frames = [quantize_int8.read_frame(e04.SHT_TRAIN_VIDEOS, v, i) for v, i in picks]
    quantize_int8.quantize(e04.ML0_FP32_ONNX, tmp, frames, "fp32", "minmax")
    identical = e04.sha(tmp) == e04.sha(e04.ML0_INT8_ONNX)
    return {"pass": bool(len(picks) == 200 and same_list and from_train and identical), "n_frames": len(picks), "list_reproduced": bool(same_list), "from_training_videos": bool(from_train),
            "requantised_identical_hash": bool(identical), "detail": "200 frames, seed 0, training/videos only; re-quantising reproduces the identical file hash"}


def rule_d6(tables: dict, ds_list=DATASETS) -> dict:
    out, ok = {}, True
    for ds in ds_list:
        for tag in ("mv", "mvadv") + (("int8", "int8mv") if ds == "ShanghaiTech" else ()):
            r = max_reuse_run(e04.load_tb(tag, ds), ds)
            out[f"{ds}/{tag}"] = r
            ok &= r <= e04.A1
    probe = e03.reuse_pattern(np.full((12, 4), np.nan, dtype=np.float32), np.zeros((12, 4), dtype=np.float32), 0, 12, 1.0, 1.0)[0]
    nan_false = not bool(probe.any())
    roi = np.zeros((12, 4), dtype=np.float32)
    _, src = e03.reuse_pattern(roi, roi, 0, 12, 1.0, 1.0)
    causal = bool(all(src[o] <= o for o in range(12)))
    return {"pass": bool(ok and nan_false and causal), "max_consecutive_reused": out, "A1": e04.A1, "nan_compares_false": nan_false, "sources_never_after_current": causal,
            "detail": "C1 never reuses a detector output older than A1 or taken after the current frame; an invalid MV (NaN: key frame) never validates a reuse"}


def rule_d7(ctx) -> dict:
    names = ctx.names["ShanghaiTech"]
    from edgelab.pose.detcache import load_detections

    d = e04.CACHE_JOBS["ML0_ort_int8"][0]
    n_ok, bad = 0, []
    for c in names:
        p = d / f"{c}.npz"
        if not p.exists():
            bad.append(c)
            continue
        _, data = load_detections(p)
        if np.array_equal(data["frame_ids"], np.arange(len(data["frame_ids"]))) and len(data["frame_ids"]) == len(ctx.labels["ShanghaiTech"][names.index(c)]):
            n_ok += 1
        else:
            bad.append(c)
    # re-run of two clips (single thread, same command as the cache stage) is identical
    tmp = e04.WORK_DIR / "tmp_d7" / "cache"
    tmp.mkdir(parents=True, exist_ok=True)
    vids = [names[0], names[60]]
    out, det, label = e04.CACHE_JOBS["ML0_ort_int8"]
    import os

    env = dict(os.environ, OMP_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1", MLFLOW_DISABLE_AGENT_HINT="1")
    cmd = [sys.executable, "-m", "edgelab.pose.detcache", "--frames-root", str(e04.SHT_FRAMES_ROOT), "--level", "L0", "--enhance", "E0", "--out", str(tmp), "--detector", str(det),
           "--ncnn-fp16", "on", "--label", label, "--threads", "1", "--videos", *vids]
    r = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=str(e00.EXPERIMENTS_ROOT))
    same = {v: (tmp / f"{v}.npz").exists() and _cmp_npz(tmp / f"{v}.npz", d / f"{v}.npz") for v in vids}
    return {"pass": bool(n_ok == len(names) and not bad and all(same.values()) and r.returncode == 0), "clips_ok": n_ok, "clips": len(names), "bad": bad, "rerun_identical": same,
            "detail": "INT8 cache: 107 clips, one record per frame, frame ids 0..n-1; re-running 2 clips with the same command reproduces every output field (detections, keypoints, counts, luma) bit for bit; host timing fields (*_ms) are excluded"}


# =====================================================================================================================
# 4. Gates (plan section 8), diagnostics
# =====================================================================================================================
def ideal_reuse_gap(ctx, ds: str, tag: str, cfg: dict, cfg03: dict) -> dict:
    """Scorer error caused by reuse at full coverage: ideal (zero-delay) series of K = 1 with level-3 reuse minus K = 1 without reuse (same tables / pools)."""
    pools = e02.build_pools(ds, e04.PRIMARY[ds], "locked")
    tb = e03.load_tables_npz(ds) if tag == "pix" else e04.load_tb(tag, ds)
    w = e04.make_world(ctx, ds, tb, pools, e04.k_pixel(cfg03, ds))
    allc = list(range(len(w.names)))
    c = w.cost(2.0)
    out = {}
    series = {}
    for name, lvl in (("noreuse", 0), ("reuse3", 3)):
        R = e03.simulate(w, c, e03.FixedPolicy(w, c, 1.0, lvl), allc)
        ser, _, _ = e03.delivery(w, R, 0, ideal=True)
        series[name] = ser
        out[name] = e01.auc_subsets(w.labels, ser, w.dev_idx, w.conf_idx)
    out["delta_pt"] = {k: out["reuse3"][k] - out["noreuse"][k] for k in out["reuse3"]}
    for sub, idx in (("dev", w.dev_idx), ("confirmation", w.conf_idx)):
        out.setdefault("paired", {})[sub] = e00.paired_diff(w.labels, series["reuse3"], series["noreuse"], idx, e04.N_BOOT)
    return out


def md(rows: list, headers: list) -> str:
    return md_table(headers, rows)


# =====================================================================================================================
# 5. Report stage
# =====================================================================================================================
def stage_report(ctx, procs: int, log=print) -> dict:
    t0 = time.perf_counter()
    cfg, cfg03 = e04.load_config(), e04.load_cfg03()
    stamp = e04.sha(e04.EXP_DIR / "config.yaml")
    runs = {ds: pickle.loads(e04.run_path(ds).read_bytes()) for ds in DATASETS}
    for ds in DATASETS:
        assert runs[ds]["stamp"] == stamp, f"run {ds} is stale: config.yaml changed since `run`"
    quality = json.loads((e04.WORK_DIR / "quality.json").read_text())
    assert quality["stamp"] == stamp, "quality stage is stale"
    tables_check = json.loads((e04.WORK_DIR / "tables_check.json").read_text())
    studies = {}
    summary: dict = {"experiment": "Q2-P1-E04", "title": "Optional optimizations", "machine": e04.MACHINE_TAG, "seed": e04.SEED, "config_sha256": stamp,
                     "hardware_disclosure": cfg["hardware_disclosure"], "datasets": {}}
    decisions: list[dict] = []
    for ds in DATASETS:
        log(f"[{now()}] report {ds}: analysis")
        st = Study(ctx, ds, runs[ds], cfg03)
        studies[ds] = st
        A = st.A
        d: dict = {"n_clips": len(st.w.names), "cells": [A.cell_row(arm, cid, sub) for arm in st.arms for cid in A.cids for sub in SUBSETS]}
        # ---- +MV
        d["mv"] = {"decision": st.decide("C1-MV", "C1-Pixel", "C1-MV@W", "C1-Pixel"),
                   "advection": {sub: st.con("C1-MV-Advect", "C1-MV", sub) for sub in SUBSETS},
                   "motion_only": {sub: st.con("Motion-Only+MV", "Motion-Only-Pixel", sub) for sub in SUBSETS},
                   "motion_only_cpu_same_knob": None,
                   "sensitivity": {name: {sub: st.con(a, b, sub) for sub in SUBSETS} for name, (a, b) in
                                   {"cost_x1.0": ("C1-MV|cost1.0", "C1-Pixel"), "cost_x2.0": ("C1-MV|cost2.0", "C1-Pixel"), "level1": ("C1-MV|L1", "C1-Pixel|L1"),
                                    "level2": ("C1-MV|L2", "C1-Pixel|L2"), "pixel_costs_at_vm_values": ("C1-Pixel|vmcost", "C1-Pixel")}.items()},
                   "cells_dev": A.contrast_cells("C1-MV", "C1-Pixel", "dev"), "cells_confirmation": A.contrast_cells("C1-MV", "C1-Pixel", "confirmation")}
        d["mv"]["sensitivity_cpu_path_b"] = {name: {sub: st.cpu_reduction(new, "C1-Pixel", sub)["mean"] for sub in SUBSETS}
                                             for name, new in (("C1-MV@W", "C1-MV@W"),)}
        d["mv"]["accounting"] = {arm: {k: st.cell_mean(arm, k) for k in ("cost", "saved_ms_per_s", "overhead_ms_per_s", "export_ms_per_s", "c1_check_ms_per_s", "retrieve_saved_ms_per_s",
                                                                            "decode_ms_per_s", "pose_ms_per_s", "reuse_rate_pct")} for arm in ("C1-Pixel", "C1-MV@W", "C1-MV")}
        # ---- diagnostics: pixel vs MV statistics, decision agreement, reuse error
        ms = e04.load_mvstat("mv", ds)
        z03 = np.load(e03.static_path(ds))
        thr03 = cfg03["datasets"][ds]["constants"]["c1"]
        mot = st.w.motion
        mvm = ms["mv_motion"].astype(np.float64)
        fin = np.isfinite(mot) & np.isfinite(mvm)
        from scipy.stats import pearsonr, spearmanr

        pix_dec = (z03["roi"][:, 0] <= thr03["roi"][2]) & (z03["ext"][:, 0] <= thr03["ext"][2])
        mv_dec = (ms["roi"][:, 0] <= ms["thr"]["roi"][2]) & (ms["ext"][:, 0] <= ms["thr"]["ext"][2])
        both = np.isfinite(z03["roi"][:, 0]) & np.isfinite(ms["roi"][:, 0])
        d["mv"]["diagnostics"] = {
            "motion_correlation": {"pearson": float(pearsonr(mot[fin], mvm[fin])[0]), "spearman": float(spearmanr(mot[fin], mvm[fin])[0]), "n": int(fin.sum())},
            "border_correlation": {"spearman": float(spearmanr(st.w.border[fin & np.isfinite(st.w.border)], ms["mv_border"][fin & np.isfinite(st.w.border)])[0])},
            "c1_decision_agreement_level3_age1": agreement(pix_dec[both], mv_dec[both]),
            "mv_thresholds_test_dev": ms["thr"], "pixel_thresholds_e03": thr03,
            "invalid_frame_fraction": float(1.0 - np.isfinite(ms["mv_motion"]).mean()),
            "mv_max_roi_p90": {"mean": float(np.nanmean(ms["mv_max_roi_p90"])), "p90": float(np.nanpercentile(ms["mv_max_roi_p90"], 90))},
            "pose_reuse_error_height_norm": {tag: [hist_stats(e04.load_tb(tag, ds)["hist"][li] if tag != "pix" else e03.load_tables_npz(ds)["hist"][li]) for li in range(3)]
                                             for tag in ("pix", "mv", "mvadv")}}
        # ---- INT8
        if ds == "ShanghaiTech":
            mv_ok = d["mv"]["decision"]["adopted"]
            base = "C1-MV" if mv_ok else "C1-Pixel"
            d["int8"] = {"base_used_for_decision": base,
                         "on_B0": st.decide("C1-INT8", "C1-Pixel", "C1-INT8@W", "C1-Pixel"), "on_MV": st.decide("C1-MV-INT8", "C1-MV", "C1-MV-INT8@W", "C1-MV"),
                         "uncorrected_pool": {sub: st.con("C1-INT8|uncorrected", "C1-Pixel", sub) for sub in SUBSETS},
                         "pose_cpu_reduction_B0": 1.0 - st.cell_mean("C1-INT8@W", "pose_ms_per_s") / st.cell_mean("C1-Pixel", "pose_ms_per_s"),
                         "accounting": {arm: {k: st.cell_mean(arm, k) for k in ("cost", "pose_ms_per_s", "saved_ms_per_s", "reuse_rate_pct")} for arm in ("C1-Pixel", "C1-INT8@W", "C1-MV", "C1-MV-INT8@W")}}
            d["int8"]["decision"] = d["int8"]["on_MV" if mv_ok else "on_B0"]
        # ---- verdict-relevant sub-sets only; calibration records
        d["calibration"] = {f"{arm}|{cid}": {k: A.r(arm, cid)[k] for k in ("knob", "target", "cost", "ratio", "status", "evals", "w_ref", "hold")} for arm in st.arms for cid in A.cids}
        d["w_ref"] = runs[ds]["w_ref"]
        summary["datasets"][ds] = d
    # ---- verdicts
    summary["verdicts"] = {"mv": component_verdict({ds: summary["datasets"][ds]["mv"]["decision"] for ds in DATASETS}),
                           "int8": component_verdict({"ShanghaiTech": summary["datasets"]["ShanghaiTech"]["int8"]["decision"]})}
    # ---- quality gate inputs
    q = quality
    locked = e04.TRACKERS_E04["locked"].label
    p = q["pairs"]["ML0_ort_int8-ML0_ncnn"]
    s0, s4 = p[f"{locked}|S0|all"]["delta"], p[f"{locked}|S4|all"]["delta"]
    i8 = summary["datasets"]["ShanghaiTech"]["int8"]
    deliv = i8["on_B0"]["all"]["auc_A"]["delta"]
    cpu_i8 = i8["on_B0"]["all"]["cpu_B"]["mean"]
    log(f"[{now()}] report: gates")
    gap = {tag: ideal_reuse_gap(ctx, "ShanghaiTech", tag if tag != "pix" else "pix", cfg, cfg03) for tag in ("pix", "mv", "mvadv")}
    e03sum = json.loads((e03.EXP_DIR / "summary.json").read_text())
    tax = e03sum["datasets"]["ShanghaiTech"]["failure"]["taxonomy"].get("Cache-Only|s2.0_b0.50")
    rec = q["caches"]["ML0_ncnn"]["recall"]
    summary["gates"] = {
        "qat": qat_gate(s0, s4, deliv, cpu_i8) | {"note": "INT8 vs deployed NCNN FP16 baseline, locked tracker, all clips; delivered = matched-cost cell average on ShanghaiTech (all clips)"},
        "cadence_aware_scorer": {"ideal_reuse_gap": gap, "fires": bool(min(g["delta_pt"]["all"] for g in gap.values()) < -DEGRADE_PT),
                                 "rule": "fires only if reuse at full coverage (K = 1, zero delay) costs more than 0.25 pt of ideal AUC",
                                 "e03_missed_event_taxonomy_cache_only_s2.0_b0.50": tax},
        "longer_pose_finetuning": {"fires": False, "evaluable": False,
                                   "evidence": {"recall_lt48_ml0_ncnn": rec["lt48"], "recall_48_96_ml0_ncnn": rec["48-96"], "recall_all": rec["all"],
                                                "s4_ml0_vs_best_stock": {k: q["caches"][k]["trackers"][locked]["auc"]["S4"] for k in ("ML0_ncnn", "stock_ncnn800")}},
                                   "note": "the oracle-detection (detector-recall-limited vs ideal) series needed for this gate was not built in E04; proxy only: E03 attributes "
                                           f"{(tax['missed_scorer_limited'] / max(tax['missed'], 1) * 100) if tax else float('nan'):.0f} % of missed events to the scorer"}}
    summary["gates"]["gpu_experiment_warranted"] = bool(summary["gates"]["qat"]["fires"] or summary["gates"]["cadence_aware_scorer"]["fires"])
    summary["quality"] = {"caches": q["caches"], "pairs": q["pairs"], "recall_pairs": q["recall_pairs"]}
    # ---- rules
    log(f"[{now()}] report: correctness rules")
    rules = {"E4-D1": rule_d1(runs, tables_check), "E4-D2": rule_d2(ctx), "E4-D3": rule_d3(), "E4-D4": rule_d4(cfg), "E4-D5": rule_d5(), "E4-D6": rule_d6(tables_check),
             "E4-D7": rule_d7(ctx)}
    summary["rules"] = {k: v["pass"] for k, v in rules.items()}
    summary["rules_detail"] = rules
    summary["vm"] = {"derived_costs": cfg["vm"]["derived_costs"], "pose_pools": cfg["vm"]["pose_pools"], "pose_drift": cfg["vm"]["pose_drift_vs_e02"], "cost_table": cfg["vm"]["cost_table"]}
    all_rules = all(rules[k]["pass"] for k in rules)
    summary["reproducible"] = all_rules
    if not all_rules:  # an adopted component is only adopted if the rules pass (plan section 4)
        for key in ("mv", "int8"):
            summary["verdicts"][key] += " [RULES FAILED: not adoptable]"
    # ---- decisions.jsonl
    for key, rule in rules.items():
        decisions.append({"type": "rule", "rule": key, "pass": rule["pass"], "detail": rule.get("detail")})
    for ds in DATASETS:
        decisions.append({"type": "decision", "component": "+MV", "dataset": ds, **summary["datasets"][ds]["mv"]["decision"]})
    decisions.append({"type": "decision", "component": "+INT8", "dataset": "ShanghaiTech", "base": i8["base_used_for_decision"], **i8["decision"]})
    decisions.append({"type": "decision", "component": "+INT8", "dataset": "UBnormal", "adopted": False, "note": "not evaluated: no UB-geometry INT8 cache (frontend ML0_mps proxy)"})
    for g, v in summary["gates"].items():
        decisions.append({"type": "gate", "gate": g, **(v if isinstance(v, dict) else {"value": v})})
    decisions.append({"type": "verdict", **summary["verdicts"]})
    for ds in DATASETS:
        for key, v in summary["datasets"][ds]["calibration"].items():
            decisions.append({"type": "calibration", "dataset": ds, "arm_cell": key, **v})
        for arm, tr in runs[ds]["traces"].items():
            decisions.append({"type": "controller_trace", "dataset": ds, "arm": arm, **tr})
    EXP = e04.EXP_DIR
    with (EXP / "decisions.jsonl").open("w") as fh:
        for rec_ in decisions:
            fh.write(json.dumps(plain(rec_)) + "\n")
    n_rows = write_results_csv(studies, EXP / "results.csv")
    summary["rows_results_csv"] = n_rows
    summary["runtime_seconds"] = time.perf_counter() - t0
    (EXP / "summary.json").write_text(json.dumps(plain(summary), indent=1))
    write_report(summary, studies, cfg, EXP / "report.md")
    log(f"[{now()}] report written ({time.perf_counter() - t0:.0f}s)")
    return summary


def write_results_csv(studies: dict, path: Path) -> int:
    cols = ["dataset", "video", "split", "arm", "cell", "s", "beta", "target_cpu_ms_per_s", "cpu_ms_per_video_s", "cost_ratio_stream", "status", "knob", "w_ref", "hold", "n_frames", "fps",
            "anomalous_frames", "auc_delivered", "auc_ideal", "events", "hit_0.5s", "hit_1.0s", "delay_p50_ms", "false_alerts", "normal_hours", "queue_wait_mean_ms",
            "queue_wait_max_ms", "windows_delivered", "observed_fraction", "peak_mem_mb_stream"]
    n = 0
    with path.open("w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(cols)
        for ds, st in studies.items():
            A, w = st.A, st.w
            dev = set(w.dev_idx)
            for arm in st.arms:
                for cid in A.cids:
                    r = A.r(arm, cid)
                    ser, idl, tab = A.series(arm, cid), A.series(arm, cid, "ideal"), A.table(arm, cid)
                    obs = np.split(r["observed"], A.cut)
                    for i, name in enumerate(w.names):
                        dl = tab[i]["delays_ms"]
                        wr.writerow([ds, name, "dev" if i in dev else "confirmation", arm, cid, r["s"], r["beta"], round(r["target"], 3), round(float(r["clip_cpu_ms_per_s"][i]), 3),
                                     round(r["ratio"], 4), r["status"], round(r["knob"], 4), round(r["w_ref"], 3), r["hold"], int(w.n_frames[i]), A.fps, int(w.labels[i].sum()),
                                     round(100 * e00.clip_auc(w.labels[i], ser[i]), 4), round(100 * e00.clip_auc(w.labels[i], idl[i]), 4), tab[i]["events"], tab[i]["hit_0.5"],
                                     tab[i]["hit_1.0"], round(float(np.median(dl)), 2) if dl else "", tab[i]["false_alerts"], round(tab[i]["normal_frames"] / A.fps / 3600, 6),
                                     round(float(r["clip_wait_mean_ms"][i]), 3), round(float(r["clip_wait_max_ms"][i]), 3), int(r["clip_windows"][i]), round(float(obs[i].mean()), 4),
                                     round(r["mem_mb"], 1)])
                        n += 1
    return n


# =====================================================================================================================
# 6. report.md
# =====================================================================================================================
def ctxt(c: dict, nd: int = 2) -> str:
    return f"{c['delta']:+.{nd}f} [{c['lo']:+.{nd}f}, {c['hi']:+.{nd}f}]"


def yesno(x: bool) -> str:
    return "yes" if x else "no"


def decision_rows(name: str, d: dict) -> list[list]:
    a, b = d["path_a"], d["path_b"]
    return [[name, "A: matched-cost AUC gain", f"dev {ctxt(a['dev'])}", f"conf {a['confirmation']['delta']:+.2f}", yesno(a["dev_ok"]), yesno(a["confirmation_ok"]), yesno(a["adopt"])],
            [name, "B: CPU at identical schedule", f"dev {100 * b['cpu_reduction_dev']:+.1f} % (AUC {b['auc_dev']['delta']:+.2f} pt)",
             f"conf {100 * b['cpu_reduction_confirmation']:+.1f} % (AUC {b['auc_confirmation']['delta']:+.2f} pt)", yesno(b["dev_ok"]), yesno(b["confirmation_ok"]), yesno(b["adopt"])]]


def write_report(summary: dict, studies: dict, cfg: dict, path: Path) -> None:
    S = summary["datasets"]
    sht, ub = S["ShanghaiTech"], S["UBnormal"]
    q = summary["quality"]
    locked = e04.TRACKERS_E04["locked"].label
    L: list[str] = []
    add = L.append
    V = summary["verdicts"]
    add("# Q2-P1-E04 — Optional optimizations (+MV, then +INT8): report\n")
    add("**Hardware / honesty disclosure.** The physical Raspberry Pi 5 is **not connected**. Every CPU cost is an `edge-os-arm64-vm` measurement multiplied by the E02/E03 Pi "
        "multiplier `s`; delivered AUC is *simulated delivery on replayed detections* (frozen frontend `ML0_ncnn`@640 for ShanghaiTech, `ML0_mps` proxy for UBnormal, locked tracker, frozen "
        "STG-NF scorer, E00 causal S3/S4 protocol). Two E04-specific caveats: (i) the VM exposes `asimddp`, `i8mm`, `bf16`, the Pi 5's Cortex-A76 only dot-product, so an INT8 speed-up measured "
        "here is an **upper bound**; (ii) the MV streams are **our own x264 IPPP re-encode** of the dataset frames (UBnormal: second generation), not camera streams. No training was done "
        "(INT8 = post-training static quantisation, 200 calibration frames from training videos only). All thresholds come from `test_dev` / label-free statistics; verdicts follow the "
        "pre-registered rule of `plan.md` section 4 (copied into `config.yaml` before `run`).\n")
    add("## 1. Answer\n")
    mv = sht["mv"]["decision"]
    add(f"* **+MV (codec motion vectors replace the pixel-difference cue):** **{V['mv']}**. ShanghaiTech: {'adopted' if mv['adopted'] else 'not adopted'} "
        f"(path A matched-cost dev {ctxt(mv['path_a']['dev'])} pt, confirmation {mv['path_a']['confirmation']['delta']:+.2f} pt; path B total CPU at the same schedule "
        f"{100 * mv['path_b']['cpu_reduction_dev']:+.1f} % dev / {100 * mv['path_b']['cpu_reduction_confirmation']:+.1f} % confirmation, AUC {mv['path_b']['auc_dev']['delta']:+.2f} / "
        f"{mv['path_b']['auc_confirmation']['delta']:+.2f} pt). UBnormal: {'adopted' if ub['mv']['decision']['adopted'] else 'not adopted'} "
        f"(path A dev {ctxt(ub['mv']['decision']['path_a']['dev'])}, confirmation {ub['mv']['decision']['path_a']['confirmation']['delta']:+.2f}; path B "
        f"{100 * ub['mv']['decision']['path_b']['cpu_reduction_dev']:+.1f} % / {100 * ub['mv']['decision']['path_b']['cpu_reduction_confirmation']:+.1f} %).")
    i8 = sht["int8"]["decision"]
    add(f"* **+INT8 (ML0 ORT static-QDQ INT8 replaces NCNN FP16; ShanghaiTech only):** **{V['int8']}** (evaluated on top of "
        f"{'+MV' if sht['int8']['base_used_for_decision'] == 'C1-MV' else 'B0 = E03 Cache-C1'}). Path A dev {ctxt(i8['path_a']['dev'])} pt, confirmation {i8['path_a']['confirmation']['delta']:+.2f} pt; "
        f"path B total CPU {100 * i8['path_b']['cpu_reduction_dev']:+.1f} % dev / {100 * i8['path_b']['cpu_reduction_confirmation']:+.1f} % confirmation at the identical schedule "
        f"(AUC {i8['path_b']['auc_dev']['delta']:+.2f} / {i8['path_b']['auc_confirmation']['delta']:+.2f} pt). UBnormal: not evaluated (no UB-geometry INT8 cache).")
    g = summary["gates"]
    add(f"* **Section-8 gates:** QAT {'FIRES' if g['qat']['fires'] else 'does not fire'}; cadence-aware scorer {'FIRES' if g['cadence_aware_scorer']['fires'] else 'does not fire'}; "
        f"longer pose fine-tuning: not evaluable with E04 data (proxy: no). **GPU experiment warranted: {'yes' if g['gpu_experiment_warranted'] else 'no'}.**")
    add(f"* **Correctness rules:** " + ", ".join(f"{k} {'pass' if v else 'FAIL'}" for k, v in summary["rules"].items()) + ".\n")
    # ---- VM profile
    add("## 2. VM profile (what each ingredient costs)\n")
    ct = summary["vm"]["cost_table"]
    rows = []
    for kind, label in (("sht", "ShanghaiTech 856x480"), ("ub", "UBnormal 1280x720")):
        t4, t1 = ct[kind]["T4"], ct[kind]["T1"]
        rows.append([label, fmt(t4["decode_plain"]["wall_ms"], 3), fmt(t4["decode_mv"]["wall_ms"], 3), fmt(t4["decode_mv"]["wall_ms"] - t4["decode_plain"]["wall_ms"], 3),
                     fmt(t4["decode_mv"]["cpu_ms"] - t4["decode_plain"]["cpu_ms"], 3), fmt(t4["bgr"]["wall_ms"], 3), fmt(t1["mv_c1"]["wall_ms"], 3), fmt(t1["mv_obs"]["wall_ms"], 3),
                     fmt(t1["pix_c1"]["wall_ms"], 3), fmt(t1["pix_obs"]["wall_ms"], 3)])
    add(md(rows, ["clips", "decode T4 plain", "decode T4 +MV", "export overhead wall", "export overhead CPU", "BGR convert (retrieve credit)", "MV C1 check", "MV observer",
                  "pixel C1 check", "pixel observer"]))
    add("\nms per frame, VM, mean (cost ingredients are multiplied by 1.25 and by the Pi multiplier `s` in the simulation). The export overhead is charged on **every** frame.\n")
    pp = summary["vm"]["pose_pools"]
    rows = []
    for t in (1, 2, 3, 4):
        r = [f"T={t}"]
        for m in ("ML0-640-ncnn", "ML0-640-ort-fp32", "ML0-640-ort-int8"):
            r.append(f"{pp[f'{m}_T{t}']['wall_ms']:.1f} / {pp[f'{m}_T{t}']['cpu_ms']:.1f}")
        r.append(f"{pp[f'ML0-640-ncnn_T{t}']['wall_ms'] / pp[f'ML0-640-ort-int8_T{t}']['wall_ms']:.2f}x")
        r.append(f"{pp[f'ML0-640-ort-fp32_T{t}']['wall_ms'] / pp[f'ML0-640-ort-int8_T{t}']['wall_ms']:.2f}x")
        rows.append(r)
    add(md(rows, ["threads", "ML0 NCNN FP16 wall / CPU ms", "ML0 ORT FP32", "ML0 ORT INT8", "INT8 speed-up vs NCNN FP16", "INT8 speed-up vs ORT FP32 (quantisation only)"]))
    dr = summary["vm"]["pose_drift"]
    add(f"\nSession drift (E4-D4): E04 NCNN pose pool vs E02 pool, max wall deviation {dr['max_abs_wall_deviation_pct']:.1f} % (limit 10 %). The INT8 pool is multiplied by the E02/E04 NCNN "
        f"ratio per thread count (an uncorrected-pool arm is reported as sensitivity).\n")
    add("Stock-model profile (NCNN FP16 vs ORT INT8, wall ms / CPU ms): " + "; ".join(
        f"{m} T4 {pp[f'{m}_T4']['wall_ms']:.1f} / {pp[f'{m}_T4']['cpu_ms']:.1f}" for m in ("stock-640-ncnn", "stock-640-ort-int8", "stock-800-ncnn", "stock-800-ort-int8")) + ".\n")
    # ---- +MV
    add("## 3. Step 1 — +MV\n")
    add("### 3.1 Decision (12-cell averages, paired video bootstrap, E03 absolute CPU targets)\n")
    rows = decision_rows("ShanghaiTech", sht["mv"]["decision"]) + decision_rows("UBnormal", ub["mv"]["decision"])
    add(md(rows, ["dataset", "path", "test_dev", "test_confirmation", "dev ok", "conf ok", "adopt"]))
    add("")
    for ds, d in S.items():
        a = d["mv"]["accounting"]
        add(f"* **{ds}** CPU accounting (cell mean, ms per video-second; scenario multiplier included): Pixel total {a['C1-Pixel']['cost']:.0f} (saved by C1 {a['C1-Pixel']['saved_ms_per_s']:.0f}, "
            f"checks {a['C1-Pixel']['overhead_ms_per_s']:.1f}); MV at the same schedule {a['C1-MV@W']['cost']:.0f} (saved {a['C1-MV@W']['saved_ms_per_s']:.0f} incl. retrieve-skip "
            f"{a['C1-MV@W']['retrieve_saved_ms_per_s']:.1f}, MV export {a['C1-MV@W']['export_ms_per_s']:.1f}, checks {a['C1-MV@W']['c1_check_ms_per_s']:.1f}); C1 reuse rate "
            f"{a['C1-Pixel']['reuse_rate_pct']:.1f} % (pixel) vs {a['C1-MV@W']['reuse_rate_pct']:.1f} % (MV).")
    add("")
    A_rows = []
    for ds, st in studies.items():
        for sub in SUBSETS:
            cm = st.con("C1-MV", "C1-Pixel", sub)
            A_rows.append([ds, sub, fmt(cm["b"]), fmt(cm["a"]), ctxt(cm), f"{cm['cells_positive']}/{cm['cells']}"])
    add(md(A_rows, ["dataset", "subset", "C1-Pixel AUC (avg of 12 cells)", "C1-MV AUC", "MV - Pixel (95 % CI) pt", "cells MV > Pixel"]))
    add("\n### 3.2 Components and sensitivities (matched-cost contrasts, pt of delivered AUC)\n")
    rows = []
    for ds, d in S.items():
        for sub in ("dev", "confirmation"):
            rows.append([ds, sub, ctxt(d["mv"]["advection"][sub]), ctxt(d["mv"]["motion_only"][sub])] + [ctxt(d["mv"]["sensitivity"][k][sub]) for k in ("cost_x1.0", "cost_x2.0", "level1", "level2",
                                                                                                                                             "pixel_costs_at_vm_values")])
    add(md(rows, ["dataset", "subset", "advection (C1-MV-Advect - C1-MV)", "Motion-Only+MV - Motion-Only", "C1-MV cost x1.0 - Pixel", "C1-MV cost x2.0 - Pixel", "level 1 MV - pixel",
                  "level 2 MV - pixel", "Pixel with VM-measured costs - Pixel"]))
    add("\n### 3.3 Diagnostics\n")
    rows = []
    for ds, d in S.items():
        dg = d["mv"]["diagnostics"]
        ag = dg["c1_decision_agreement_level3_age1"]
        pe = dg["pose_reuse_error_height_norm"]
        rows.append([ds, f"{dg['motion_correlation']['pearson']:.2f} / {dg['motion_correlation']['spearman']:.2f}", f"{dg['border_correlation']['spearman']:.2f}", f"{100 * ag['agree']:.1f} % (kappa {ag['kappa']:.2f})",
                     f"{100 * ag['pixel_static']:.1f} / {100 * ag['mv_static']:.1f} %", f"{100 * dg['invalid_frame_fraction']:.1f} %",
                     f"{pe['pix'][2]['mean']:.3f} / {pe['mv'][2]['mean']:.3f} / {pe['mvadv'][2]['mean']:.3f}", f"{pe['pix'][2]['p90']:.2f} / {pe['mv'][2]['p90']:.2f} / {pe['mvadv'][2]['p90']:.2f}"])
    add(md(rows, ["dataset", "motion r (Pearson / Spearman)", "border Spearman", "C1 static decision agreement (level 3, age 1)", "static rate pixel / MV", "key-frame (invalid) share",
                  "reused-pose error mean (px / h) pixel / MV / advected", "p90"]))
    add("\nAdvection: " + "; ".join(f"{ds} dev {ctxt(S[ds]['mv']['advection']['dev'])}, confirmation {ctxt(S[ds]['mv']['advection']['confirmation'])}" for ds in S) + ".\n")
    # ---- +INT8
    add("## 4. Step 2 — +INT8 (ShanghaiTech)\n")
    add("### 4.1 Detector quality (cache-level; locked tracker `c=0.20, low=0.05`, E00 causal S4 with the arm's sentinel, offline S0)\n")
    rows = []
    for key, e in q["caches"].items():
        t = e["trackers"][locked]["auc"]
        rows.append([e["label"], fmt(t["S0"]["all"]), fmt(t["S4"]["all"]), fmt(t["S4"]["dev"]), fmt(t["S4"]["confirmation"]), f"{100 * e['recall']['lt48']['recall']:.1f}", f"{100 * e['recall']['48-96']['recall']:.1f}",
                     f"{100 * e['recall']['ge96']['recall']:.1f}", f"{100 * e['recall']['all']['recall']:.1f}"])
    add(md(rows, ["detector cache", "S0 AUC (all)", "S4 AUC (all)", "S4 dev", "S4 confirmation", "recall <48 px %", "recall 48-96 px %", "recall >=96 px %", "recall all %"]))
    add("")
    rows = []
    for name, res in q["pairs"].items():
        r = [name]
        for step in ("S0", "S4"):
            for sub in ("all", "dev", "confirmation"):
                c = res[f"{locked}|{step}|{sub}"]
                r.append(f"{c['delta']:+.2f} [{c['ci95'][0]:+.2f}, {c['ci95'][1]:+.2f}]")
        rows.append(r)
    add(md(rows, ["pair (A - B), pt AUC", "S0 all", "S0 dev", "S0 confirmation", "S4 all", "S4 dev", "S4 confirmation"]))
    add("")
    rows = []
    for name, res in q["recall_pairs"].items():
        rows.append([name] + [f"{100 * res[g]['delta']:+.1f} [{100 * res[g]['ci95'][0]:+.1f}, {100 * res[g]['ci95'][1]:+.1f}]" for g in ("<48", "48-96", ">=96", "all")])
    add(md(rows, ["recall pair (A - B), pt", "<48 px", "48-96 px", ">=96 px", "all"]))
    add("\n### 4.2 Decision\n")
    iv = sht["int8"]
    rows = decision_rows("on B0 (E03 Cache-C1)", iv["on_B0"]) + decision_rows("on +MV", iv["on_MV"])
    add(md(rows, ["base", "path", "test_dev", "test_confirmation", "dev ok", "conf ok", "adopt"]))
    ac = iv["accounting"]
    add(f"\nPose-stage CPU at the identical schedule (B0): {ac['C1-Pixel']['pose_ms_per_s']:.0f} -> {ac['C1-INT8@W']['pose_ms_per_s']:.0f} ms per video-second "
        f"({100 * iv['pose_cpu_reduction_B0']:+.1f} % reduction); total {ac['C1-Pixel']['cost']:.0f} -> {ac['C1-INT8@W']['cost']:.0f}. Uncorrected-pool sensitivity (matched cost, INT8 - Pixel): "
        f"dev {ctxt(iv['uncorrected_pool']['dev'])}, confirmation {ctxt(iv['uncorrected_pool']['confirmation'])}. Decision base used: **{'+MV' if iv['base_used_for_decision'] == 'C1-MV' else 'B0'}**.\n")
    # ---- matched cost grid
    add("## 5. Per-cell matched-cost AUC (dev / confirmation, `all` stream calibration)\n")
    for ds, st in studies.items():
        arms = [a for a in ("C1-Pixel", "C1-MV", "C1-MV-Advect", "C1-INT8", "C1-MV-INT8", "Motion-Only-Pixel", "Motion-Only+MV") if st.has(a)]
        rows = []
        for cid in st.cids:
            rows.append([cid, f"{st.A.r('C1-Pixel', cid)['target']:.0f}"] + [f"{st.A.boot(a, cid, 'dev')[0]:.2f} / {st.A.boot(a, cid, 'confirmation')[0]:.2f}"
                                                                           + ("*" if st.A.r(a, cid)["status"] == "under" else ("!" if st.A.r(a, cid)["status"] == "over" else "") ) for a in arms])
        add(f"**{ds}** (`*` = cannot spend the budget: saturated schedule, `!` = cannot get below it)\n")
        add(md(rows, ["cell", "target CPU ms/s"] + arms))
        add("")
    # ---- rules
    add("## 6. Correctness rules (E4-D)\n")
    rows = [[k, "pass" if v["pass"] else "**FAIL**", v["detail"]] for k, v in summary["rules_detail"].items()]
    add(md(rows, ["rule", "result", "what was checked"]))
    add("")
    # ---- gates
    add("## 7. Section-8 gates\n")
    add(f"* **QAT:** fires = {g['qat']['fires']} (deltas INT8 - NCNN FP16 in pt: S0 {g['qat']['deltas_pt']['S0']:+.2f}, S4 {g['qat']['deltas_pt']['S4']:+.2f}, delivered (matched cost, ShanghaiTech all) "
        f"{g['qat']['deltas_pt']['delivered']:+.2f}; CPU reduction at identical schedule {100 * g['qat']['cpu_reduction']:+.1f} %). Rule: only if degradation > 0.25 pt while CPU falls.")
    gg = g["cadence_aware_scorer"]["ideal_reuse_gap"]
    add(f"* **Cadence-aware scorer:** fires = {g['cadence_aware_scorer']['fires']}. Reuse at full coverage (K = 1, zero delay, level 3) changes ideal AUC by "
        + ", ".join(f"{k} {v['delta_pt']['all']:+.2f} pt (dev {v['delta_pt']['dev']:+.2f}, confirmation {v['delta_pt']['confirmation']:+.2f})" for k, v in gg.items()) + " (pix / mv / mvadv tables).")
    lf = g["longer_pose_finetuning"]
    add(f"* **Longer pose fine-tuning:** not evaluable with E04 data (no oracle-detection series); proxy: ML0 recall <48 px {100 * lf['evidence']['recall_lt48_ml0_ncnn']['recall']:.1f} %, 48-96 px "
        f"{100 * lf['evidence']['recall_48_96_ml0_ncnn']['recall']:.1f} %. {lf['note']}.")
    add(f"* **GPU experiment:** {'recommended' if g['gpu_experiment_warranted'] else 'no GPU experiment warranted'} (no gate fires with an expected gain above the 0.25 pt margin).\n")
    # ---- limitations
    add("## 8. Known limitations and disclosed order of work\n")
    add("* VM, not Pi; INT8 speed-ups are an upper bound (VM has `i8mm`); MV statistics describe our own x264 IPPP re-encode (UBnormal second generation), not the original capture.\n"
        "* C1 track identity is an oracle (as E03). The MV static test uses the accumulated *magnitude* (not the vector sum), key frames invalidate a reuse, thresholds are test_dev quantiles of the one-frame statistic.\n"
        "* INT8 is evaluated on ShanghaiTech only (the UBnormal frontend is the MPS proxy; no UB INT8 cache). The INT8 pose pool is drift-corrected to the E02 NCNN pool.\n"
        "* Order of work: plan.md and the rule text were written first; ONNX export + INT8 calibration, H.264/MV re-encode and the detector caches (no metric) were produced before `lock`; VM profiling "
        "(a machine measurement) and the MV statistics / thresholds (label-free, test_dev) precede `lock`; `config.yaml` was hashed before `run`; the first full `run` is the reported one (see the "
        "build notes below for code corrections).\n")
    path.write_text("\n".join(L) + "\n")
