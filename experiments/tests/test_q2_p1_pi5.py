"""Q2-P1-Pi5-Suite: pure helpers of the physical Raspberry Pi 5 harness (statistics, telemetry slicing / summaries, energy, RSS growth, queue waits, cell grid,
stage-row derivation on synthetic arrays, view / CSV / report plumbing) and - when the artifacts exist - the generated outputs."""

import json
import math
from pathlib import Path

import numpy as np
import pytest

from edgelab.analysis import q2_p1_pi5_suite as s

OUT = s.EXP_PI5
HAVE_OUT = all((OUT / f).exists() for f in ("results.csv", "stage_timings_pi5.csv", "summary.json", "report.md", "config.yaml", "plan.md"))
needs_out = pytest.mark.skipif(not HAVE_OUT, reason="Pi 5 suite outputs not generated")


def sample(t, p=5.0, core=2.0, temp=60.0, thr=0, freq=2400.0, rss=300.0):
    return {"timestamp_utc": t, "power_total_w": p, "power_core_w": core, "temp_c": temp, "throttled_int": thr, "cpu_freq_mhz_mean": freq, "rss_mb": rss, "sys_cpu_pct": 90.0}


# ---- statistics -------------------------------------------------------------------------------------------------------
def test_stats4_basic_and_nan_handling():
    st = s.stats4(np.arange(1, 101, dtype=float))
    assert st["n"] == 100 and st["mean"] == pytest.approx(50.5) and st["p50"] == pytest.approx(50.5)
    assert st["p95"] == pytest.approx(95.05) and st["p99"] == pytest.approx(99.01)
    st2 = s.stats4([1.0, float("nan"), 3.0, float("inf")])
    assert st2["n"] == 2 and st2["mean"] == pytest.approx(2.0)
    assert s.stats4([])["n"] == 0 and math.isnan(s.stats4([])["mean"])


def test_thin_keeps_ends_and_bounds():
    v = list(range(1000))
    t = s.thin(v, 10)
    assert len(t) <= 10 and t[0] == 0 and t[-1] == 999
    assert s.thin([1, 2, 3], 10) == [1, 2, 3]


# ---- telemetry ---------------------------------------------------------------------------------------------------------
def test_slice_telemetry_window_and_fallbacks():
    smp = [sample(float(t)) for t in range(10)]
    assert [x["timestamp_utc"] for x in s.slice_telemetry(smp, 2.0, 5.0)] == [2.0, 3.0, 4.0, 5.0]
    assert len(s.slice_telemetry(smp, None, None)) == 10  # unknown window -> everything
    assert len(s.slice_telemetry(smp, 5.0, 5.0)) == 10  # empty window -> everything
    assert len(s.slice_telemetry(smp, 20.0, 30.0)) == 10  # window without samples -> everything


def test_summarize_telemetry_values_and_throttle_codes():
    smp = [sample(0, p=4.0, temp=60.0), sample(1, p=6.0, temp=64.0, thr=0x50000), sample(2, p=5.0, temp=62.0, thr=0)]
    t = s.summarize_telemetry(smp)
    assert t["n_samples"] == 3 and t["power_total_w"] == pytest.approx(5.0) and t["power_total_w_max"] == 6.0
    assert t["temp_c_max"] == 64.0 and t["temp_c_start"] == 60.0 and t["temp_c_end"] == 62.0
    assert t["throttled_any"] is True and t["throttled_codes"] == ["0x0", "0x50000"]
    clean = s.summarize_telemetry([sample(0), sample(1)])
    assert clean["throttled_any"] is False and clean["throttled_codes"] == ["0x0"]


def test_summarize_telemetry_ignores_missing_values():
    smp = [sample(0), {"timestamp_utc": 1.0, "power_total_w": float("nan"), "temp_c": 61.0, "throttled_int": -1}]
    t = s.summarize_telemetry(smp)
    assert t["power_total_w"] == pytest.approx(5.0) and t["temp_c_max"] == 61.0 and t["throttled_codes"] == ["0x0"]


# ---- energy / memory / queue ---------------------------------------------------------------------------------------------
def test_energy_and_active_energy():
    assert s.energy_j(5.0, 100.0) == pytest.approx(0.5)
    assert s.active_energy_j(5.0, 2.0, 100.0) == pytest.approx(0.3)
    assert s.active_energy_j(1.0, 2.0, 100.0) == 0.0  # never negative
    assert math.isnan(s.energy_j(float("nan"), 100.0)) and math.isnan(s.active_energy_j(5.0, float("nan"), 1.0))


def test_rss_growth_table():
    g = s.rss_growth_table({0: 800.0, 720: 840.0, 1440: 860.0, 2880: 880.0})
    assert g["growth_pct_vs_start"]["0"] == 0.0 and g["growth_pct_vs_start"]["720"] == pytest.approx(5.0)
    assert g["growth_pct_total"] == pytest.approx(10.0) and g["growth_pct_first_mark_to_end"] == pytest.approx(100 * 40 / 840)
    flat = s.rss_growth_table({0: 500.0, 100: 500.0})
    assert flat["growth_pct_total"] == 0.0


def test_queue_wait_stats_order():
    w = np.concatenate([np.zeros(900), np.linspace(0, 300, 100)])
    q = s.queue_wait_stats(w)
    assert q["p50"] <= q["p95"] <= q["p99"] <= q["max"] == 300.0 and q["mean"] == pytest.approx(w.mean())


def test_plain_json_safe():
    obj = {"a": np.float32(1.5), "b": np.arange(3), "c": float("nan"), "d": (np.int64(2), np.bool_(True)), "e": Path("/x")}
    p = s.plain(obj)
    json.dumps(p)
    assert p["c"] is None and p["b"] == [0, 1, 2] and p["d"] == [2, True] and p["e"] == "/x"


# ---- cell grid ----------------------------------------------------------------------------------------------------------
def test_cell_specs_grid_complete_and_unique():
    cells = s.cell_specs("all")
    ids = [c["id"] for c in cells]
    assert len(ids) == len(set(ids)) == 86
    e02 = {c["id"] for c in s.cell_specs("e02")}
    e04 = {c["id"] for c in s.cell_specs("e04")}
    assert len(e02) == 54 and len(e04) == 32 and not (e02 & e04)
    for t in (1, 2, 3, 4):
        for m in s.E02_POSE_MODELS:
            assert f"pose_{m}_T{t}" in e02  # ML0 / MLP @640, stock @640 / @800 / @320
        assert f"score_ShanghaiTech_T{t}" in e02 and f"score_UBnormal_T{t}" in e02 and f"mvprof_sht_T{t}" in e04 and f"mvprof_ub_T{t}" in e04
        for m in s.E04_POSE_MODELS:
            assert f"pose_{m}_T{t}" in e04 and f"pose_{m}_T{t}_rep1" in e04
    assert {"track_T1", "decode_h264_user_sht_T4", "decode_h264_user_ub_T2", "e2e_UBnormal_h264_T4"} <= e02
    assert all(1 <= c["threads"] <= 4 for c in cells)


def test_cell_specs_use_the_vm_worker_vocabulary():
    kinds = {c["kind"] for c in s.cell_specs("all")}
    assert kinds == {"decode", "pose", "track", "score", "e2e", "e04pose", "mvprof"}
    pose = [c for c in s.cell_specs("e02") if c["kind"] == "pose"]
    assert all(c["args"]["step"] in (1, s.POSE_STEP) for c in pose)
    assert 2880 // s.POSE_STEP == 480  # 480 natural L0 frames per pose cell


def test_e05_arm_definitions_match_the_task():
    assert set(s.E05_ARMS) == set(s.E05_ORDER) and s.E05_FRAMES == 2880 == 12 * 240
    assert s.E05_ARMS["Cache-C1+Guard-Burst32"] == ("cache_c1_guard", 32.0, 31)  # hold = W - 1
    assert s.E05_ARMS["Fixed-K1"][0] == "fixed_k1" and s.E05_ARMS["Cache-C1"][0] == "cache_c1"
    assert 2880 / 24.0 == 120.0  # 2.0 minutes of 24 fps video


# ---- stage rows ---------------------------------------------------------------------------------------------------------
TEL = {"power_total_w": 4.0, "power_core_w": 1.5, "temp_c_mean": 61.0, "temp_c_max": 63.0, "freq_mhz": 2400.0, "throttled_any": False}


def test_derive_rows_pose_energy_and_stats():
    rng = np.random.default_rng(0)
    wall = rng.uniform(60, 80, 480)
    arr = {"wall_ms": wall, "cpu_ms": wall * 3.5, "pre_ms": np.full(480, 5.0), "infer_ms": np.full(480, 60.0), "post_ms": np.full(480, 5.0)}
    rows = s.derive_rows({"id": "pose_ML0-640_T4", "part": "e02", "kind": "pose", "args": {"model": "ML0-640"}, "threads": 4}, arr, {"runtime": "ncnn", "input_hw": [384, 640], "peak_rss_mb": 600.0}, TEL, 1.9)
    assert len(rows) == 1
    r = rows[0]
    assert r["stage"] == "pose_detector" and r["config"] == "ML0-640" and r["frames"] == 480 and r["threads"] == 4
    assert r["wall_mean_ms"] == pytest.approx(wall.mean()) and r["wall_p50_ms"] <= r["wall_p95_ms"] <= r["wall_p99_ms"]
    assert r["energy_j_per_frame"] == pytest.approx(4.0 * wall.mean() / 1000.0)
    assert r["active_energy_j_per_frame"] == pytest.approx((4.0 - 1.9) * wall.mean() / 1000.0) and r["cpu_ms_mean"] == pytest.approx((wall * 3.5).mean())


def test_derive_rows_decode_h264_has_three_rows():
    g, rt = np.full(100, 1.0), np.full(100, 2.0)
    arr = {"wall_grab_ms": g, "wall_retrieve_ms": rt, "cpu_grab_ms": g * 2, "cpu_retrieve_ms": rt * 2}
    rows = s.derive_rows({"id": "decode_h264_sht_T2", "part": "e02", "kind": "decode", "args": {"kind": "h264_sht"}, "threads": 2}, arr, {}, TEL, 1.9)
    by = {r["stage"]: r for r in rows}
    assert set(by) == {"decode_h264_grab+retrieve", "decode_h264_grab", "decode_h264_retrieve"}
    assert by["decode_h264_grab+retrieve"]["wall_mean_ms"] == pytest.approx(3.0) and by["decode_h264_grab+retrieve"]["cpu_ms_mean"] == pytest.approx(6.0)


def test_derive_rows_scorer_by_n_and_mvprof():
    n = np.repeat([1, 3, 5], 10)
    arr = {"n": n, "total_ms": n * 2.0, "assemble_ms": n * 0.1, "normalize_ms": n * 0.2, "flow_ms": n * 1.7, "cpu_ms": n * 6.0}
    rows = s.derive_rows({"id": "score_ShanghaiTech_T4", "part": "e02", "kind": "score", "args": {"dataset": "ShanghaiTech"}, "threads": 4}, arr, {}, TEL, 1.9)
    assert [r["n"] for r in rows] == [1, 3, 5] and [r["wall_mean_ms"] for r in rows] == pytest.approx([2.0, 6.0, 10.0])
    mv = {f"{m}_w": np.full(20, 0.5) for m in s.MV_METHODS} | {f"{m}_c": np.full(20, 0.4) for m in s.MV_METHODS}
    rows = s.derive_rows({"id": "mvprof_sht_T1", "part": "e04", "kind": "mvprof", "args": {"kind": "sht"}, "threads": 1}, mv, {}, TEL, 1.9)
    assert len(rows) == len(s.MV_METHODS) and {r["config"] for r in rows} == set(s.MV_METHODS) and all(r["dataset"] == "ShanghaiTech" for r in rows)


# ---- Mac side plumbing ---------------------------------------------------------------------------------------------------
def test_cell_file_filter():
    assert s.is_cell_file("pose_ML0-640_T1.npz") and s.is_cell_file("pose_ML0-640_T1.json")
    for n in ("a.window.json", "a.telemetry.json", "a.summary.json", "manifest.jsonl"):
        assert not s.is_cell_file(n)


def test_build_view_links_cells_and_rep_aliases(tmp_path, monkeypatch):
    out, view = tmp_path / "out", tmp_path / "view"
    (out / "q2-p1-e02").mkdir(parents=True)
    (out / "q2-p1-e04").mkdir(parents=True)
    for t in (1, 2, 3, 4):
        for ext in (".npz", ".json"):
            (out / "q2-p1-e02" / f"pose_ML0-640_T{t}{ext}").write_text("a")
            (out / "q2-p1-e04" / f"pose_ML0-640-ncnn_T{t}{ext}").write_text("b")
    (out / "q2-p1-e02" / "pose_ML0-640_T1.summary.json").write_text("{}")
    monkeypatch.setattr(s, "MAC_OUT", out)
    monkeypatch.setattr(s, "SIM_VIEW", view)
    info = s.build_view(log=lambda *_: None)
    assert info["missing"] == []
    v02 = view / "q2-p1-e02"
    assert (v02 / "pose_ML0-640_T3_rep0.npz").read_text() == "a" and (v02 / "pose_ML0-640_T3_rep1.npz").read_text() == "b"
    assert not (v02 / "pose_ML0-640_T1.summary.json").exists() and (view / "q2-p1-e04" / "pose_ML0-640-ncnn_T2.json").exists()


def test_write_stage_csv_and_avg_reps(tmp_path, monkeypatch):
    monkeypatch.setattr(s, "EXP_PI5", tmp_path)
    base = {"part": "e04", "cell": "c", "threads": 4, "stage": "pose_detector", "config": "ML0-640-ncnn", "dataset": "ShanghaiTech", "n": "", "frames": 480, "wall_p50_ms": 1.0, "wall_p95_ms": 1.0,
            "wall_p99_ms": 1.0, "cpu_ms_mean": 3.0, "peak_rss_mb": 500.0, "power_total_w": 4.0, "power_core_w": 1.5, "energy_j_per_frame": 0.3, "active_energy_j_per_frame": 0.2, "temp_c_mean": 60.0}
    rows = [{**base, "wall_mean_ms": 70.0}, {**base, "wall_mean_ms": 74.0, "cell": "c_rep1"}]
    avg = s.avg_reps(rows)
    assert len(avg) == 1 and avg[0]["wall_mean_ms"] == pytest.approx(72.0) and avg[0]["reps"] == 2
    assert s.write_stage_csv(rows) == 2
    txt = (tmp_path / "stage_timings_pi5.csv").read_text().splitlines()
    assert txt[0].startswith("part,cell,threads,stage") and len(txt) == 3


def test_markdown_helpers():
    assert s.md(["a", "b"], [[1, 2]]).splitlines() == ["| a | b |", "|---|---|", "| 1 | 2 |"]
    assert s._f(float("nan")) == "n/a" and s._f(None) == "n/a" and s._f(1.234, 1) == "1.2"
    assert s.fci({"delta": 1.0, "lo": 0.5, "hi": 1.5}).endswith("*") and not s.fci({"delta": 1.0, "lo": -0.5, "hi": 1.5}).endswith("*") and s.fci(None) == "n/a"
    assert s.where([{"a": 1}, {"a": 2}], a=2) == [{"a": 2}]


def test_plan_template_formats():
    txt = s.PLAN_MD.format(cool=s.COOL_TARGET_C, tel=s.TELEMETRY_INTERVAL_S, boot=s.N_BOOT)
    assert "2,880-frame" in txt and "2000-resample" in txt
    assert not any(f"{{{k}}}" in txt for k in ("cool", "tel", "boot"))


# ---- generated outputs ---------------------------------------------------------------------------------------------------
@needs_out
def test_outputs_consistent():
    summ = json.loads((OUT / "summary.json").read_text())
    assert set(summ["e05_live"]) == set(s.E05_ORDER)
    for a, r in summ["e05_live"].items():
        assert r["frames"] == 2880 and r["online_equals_offline"] and r["series_finite"]
        assert r["wait_ms"]["p50"] <= r["wait_ms"]["p95"] <= r["wait_ms"]["p99"] <= r["wait_ms"]["max"]
        assert r["energy_j_per_frame"] == pytest.approx(r["energy_j_per_frame_check"], rel=0.02)
        assert set(r["rss_marks_mb"]) == {"0", "720", "1440", "2880"}
    lines = (OUT / "stage_timings_pi5.csv").read_text().splitlines()
    assert len(lines) - 1 == summ["stage_rows"] >= 86  # at least one stage row per Pi cell (54 E02 + 32 E04 = 86 cells; 362 rows in the final run)
    assert sum(v["ok"] for v in summ["cells"].values()) == 86 and sum(v["failed"] for v in summ["cells"].values()) == 0
    for ds in ("ShanghaiTech", "UBnormal"):
        d = summ["pi_calibrated"][ds]
        assert d["n_videos"] == (107 if ds == "ShanghaiTech" else 211)
        for r in d["e05_arms"]:
            assert r["auc_all_lo"] <= r["auc_all"] <= r["auc_all_hi"]
    assert summ["bootstrap"]["resamples"] == 2000
