"""Q2-P1-E04 optional optimizations: MV aggregation / statistics (pure), observer parity with E03, key-frame validity, adoption logic, VM-cost derivation,
CPU accounting in the extended simulator and - when the artifacts exist - the locked config, the run outputs and the rules E4-D1..D7."""

import copy
import json
import math
import pickle
from pathlib import Path

import numpy as np
import pytest
import yaml

from edgelab.analysis import q2_p1_e00 as e00
from edgelab.analysis import q2_p1_e01 as e01
from edgelab.analysis import q2_p1_e02 as e02
from edgelab.analysis import q2_p1_e03 as e03
from edgelab.analysis import q2_p1_e04 as e04
from edgelab.analysis import q2_p1_e04_report as rep
from edgelab.analysis import q2_p1_e04_vm as vmm

SHT = "ShanghaiTech"
HAVE_E03 = e03.tables_path(SHT).exists() and (e03.MOTION_DIR / SHT).exists()
HAVE_RUN = all(e04.run_path(ds).exists() for ds in e04.DATASETS) and (e04.EXP_DIR / "config.yaml").exists()
needs_e03 = pytest.mark.skipif(not HAVE_E03, reason="E03 run tables / motion cache not built")
needs_run = pytest.mark.skipif(not HAVE_RUN, reason="E04 run stage not executed")


def side(entries):
    """entries: (w, h, src_x, src_y, dst_x, dst_y) -> structured array like PyAV's MOTION_VECTORS side data."""
    dt = np.dtype([("source", "i4"), ("w", "u1"), ("h", "u1"), ("src_x", "i2"), ("src_y", "i2"), ("dst_x", "i2"), ("dst_y", "i2")])
    a = np.zeros(len(entries), dtype=dt)
    for i, (w, h, sx, sy, dx, dy) in enumerate(entries):
        a[i] = (-1, w, h, sx, sy, dx, dy)
    return a


# ---- MV aggregation (pure) -----------------------------------------------------------------------------------------------
def test_mv_grid_displacement_magnitude_and_coverage():
    gh, gw = 2, 3
    s = side([(16, 16, 8, 8, 11, 12), (8, 8, 24, 8, 24, 8), (8, 8, 28, 8, 28, 8)])  # MB (0,0): moves (3, 4) -> mag 5; MB (0,1): half covered, still
    dx, dy, mag, cov = vmm.mv_grid(s, gh, gw)
    assert dx[0, 0] == pytest.approx(3.0) and dy[0, 0] == pytest.approx(4.0) and mag[0, 0] == pytest.approx(5.0) and cov[0, 0] == pytest.approx(1.0)
    assert cov[0, 1] == pytest.approx(0.5) and mag[0, 1] == 0.0 and cov[1, 2] == 0.0
    z = vmm.mv_grid(None, gh, gw)
    assert all((x == 0).all() and x.shape == (gh, gw) for x in z)


def test_mv_grid_ignores_future_sources():
    s = side([(16, 16, 8, 8, 12, 8)])
    s["source"][0] = 1  # references a future frame: never produced by IPPP, never used (causality)
    assert (vmm.mv_grid(s, 1, 1)[2] == 0).all()


def test_magnitude_grid_maps_intra_macroblocks():
    mag = np.array([[1.0, 2.0]], dtype=np.float32)
    cov = np.array([[1.0, 0.0]], dtype=np.float32)
    assert vmm.magnitude_grid(mag, cov).tolist() == [[1.0, vmm.INTRA_MAG]]


def test_frame_mv_stats_and_border_mask():
    gh, gw = 20, 30
    m = np.zeros((gh, gw), dtype=np.float32)
    b = vmm.border_mask(gh, gw)
    assert b[0, 0] and b[-1, -1] and not b[10, 15] and b.sum() < gh * gw
    p90, bp90, motion, bact = vmm.frame_mv_stats(m, b)
    assert p90 == 0 and motion == 0 and bact == 0
    m[10, 15] = 5.0
    _, _, motion2, bact2 = vmm.frame_mv_stats(m, b)
    assert motion2 > 0 and bact2 == 0
    m[0, 0] = 5.0
    assert vmm.frame_mv_stats(m, b)[3] > 0


def test_x264_params_are_ippp_single_thread_keyint_2s():
    p = vmm.x264_params("ShanghaiTech")
    assert "bframes=0" in p and "ref=1" in p and "scenecut=0" in p and "threads=1" in p and "keyint=48" in p
    assert "keyint=60" in vmm.x264_params("UBnormal")


# ---- numpy observer parity (E4-D3) ----------------------------------------------------------------------------------------
def test_observers_equal_e03_functions():
    r = rep.rule_d3()
    assert r["pass"], r["checks"]


# ---- MV static statistics -----------------------------------------------------------------------------------------------
def synth_mv(n=12, gh=6, gw=8, intra_at=(0,), still=True):
    mag = np.zeros((n, gh, gw), dtype=np.float16)
    cov = np.full((n, gh, gw), 255, dtype=np.uint8)
    valid = np.ones(n, dtype=bool)
    for i in intra_at:
        valid[i] = False
    return mag, cov, valid


def test_mv_static_stats_accumulates_and_masks_boxes():
    mag, cov, valid = synth_mv()
    mag[3, 2, 2] = 4.0  # a 4 px motion inside the box at frame 3
    boxes = [np.array([[16.0, 16.0, 48.0, 48.0]])] * 12  # covers cells around (1..3, 1..3) (+1 dilation)
    roi, ext = e04.mv_static_stats(mag, cov, valid, boxes, 128.0, 96.0)
    assert roi.shape == (12, e03.A1)
    assert roi[1, 0] == 0 and roi[1, 1] > 0  # source 1: age 1 = frame 2 still, age 2 = frame 3 has the motion
    assert roi[2, 0] > 0 and roi[2, 1] == pytest.approx(roi[2, 0])  # source 2, age 1 = frame 3; accumulated stays (non-decreasing in age)
    assert roi[0, 0] == 0 or np.isfinite(roi[0, 0])
    assert np.isnan(roi[11, 0]) and np.isnan(ext[11]).all()  # no frame after the last one
    mag2, cov2, valid2 = synth_mv()
    mag2[3, 0, 7] = 8.0  # motion outside every box: exterior 95th pct is unaffected for a single cell, inside mean stays 0
    roi2, ext2 = e04.mv_static_stats(mag2, cov2, valid2, boxes, 128.0, 96.0)
    assert roi2[2, 0] == 0


def test_mv_static_stats_key_frame_invalidates_reuse():
    mag, cov, valid = synth_mv(n=12, intra_at=(0, 6))
    boxes = [np.zeros((0, 4))] * 12
    roi, ext = e04.mv_static_stats(mag, cov, valid, boxes, 128.0, 96.0)
    assert np.isnan(roi[5, 0]) and np.isnan(ext[5, 0])  # frame 6 is a key frame: nothing can be validated across it
    assert np.isnan(roi[4, 1]) and np.isfinite(roi[4, 0])  # age 1 (frame 5) fine, age 2 (frame 6) invalid
    assert np.isfinite(roi[6, 0]) and np.isfinite(roi[6, 3])
    pat, _ = e03.reuse_pattern(roi, ext, 0, 12, 1.0, 1.0)
    assert not pat[6] and pat[1]  # NaN compares False: the detector runs on the key-frame boundary


def test_intra_macroblock_counts_as_large_motion():
    mag, cov, valid = synth_mv()
    cov[3, 2, 2] = 0  # intra macroblock inside the box at frame 3
    boxes = [np.array([[16.0, 16.0, 48.0, 48.0]])] * 12
    roi, _ = e04.mv_static_stats(mag, cov, valid, boxes, 128.0, 96.0)
    assert roi[2, 0] > 0.5  # INTRA_MAG = 16 px in one cell of the box


def test_mv_max_roi_p90_nan_without_box_or_valid():
    mag, cov, valid = synth_mv()
    mag[4] = 3.0
    boxes = [np.zeros((0, 4))] * 12
    boxes[4] = np.array([[0.0, 0.0, 50.0, 50.0]])
    out = e04.mv_max_roi_p90(mag, cov, valid, boxes, 128.0, 96.0)
    assert np.isnan(out[0]) and np.isnan(out[5]) and out[4] == pytest.approx(3.0)


def test_carry_forward_keeps_leading_nan_only():
    x = np.array([np.nan, 1.0, np.nan, np.nan, 3.0, np.nan])
    y = e04.carry_forward(x)
    assert np.isnan(y[0]) and y[2] == 1.0 and y[3] == 1.0 and y[5] == 3.0


def test_adv_shift_averages_accumulated_displacement_inside_the_box():
    n, gh, gw = 6, 6, 8
    dx = np.zeros((n, gh, gw), dtype=np.float32)
    dy = np.zeros_like(dx)
    dx[2:, 1:4, 1:4] = 2.0  # content inside cells (1..3, 1..3) moves +2 px / frame from frame 2 on
    adv = (np.cumsum(dx, axis=0), np.cumsum(dy, axis=0), (gh, gw), (128, 96))
    pose = np.zeros((17, 3))
    pose[:, 0], pose[:, 1], pose[:, 2] = np.linspace(20, 44, 17), np.linspace(20, 44, 17), 1.0  # box within cells (1..2)
    sx, sy = e04.adv_shift(pose, adv, 1, 4)  # frames 2, 3, 4 -> 3 x 2 px
    assert sx == pytest.approx(6.0) and sy == pytest.approx(0.0)
    assert e04.adv_shift(pose, adv, 3, 3) == (0.0, 0.0)


# ---- tables comparison / constants ---------------------------------------------------------------------------------------
def test_compare_tables_flags_differences():
    a = {k: np.arange(5, dtype=np.float32) for k in e04.TB_KEYS}
    b = {k: v.copy() for k, v in a.items()}
    assert e04.compare_tables(a, b)["all_equal"]
    b["row_norm"][2] += 1e-3
    c = e04.compare_tables(a, b)
    assert not c["all_equal"] and c["row_norm"]["max_abs_diff"] == pytest.approx(1e-3, rel=1e-3)


def test_mv_constants_quantiles_and_dev_only():
    motion = np.concatenate([np.linspace(0, 1, 101), np.full(50, 99.0)])
    border = np.concatenate([np.linspace(0, 2, 101), np.full(50, 99.0)])
    k = e04.mv_constants(motion, border, np.arange(101))
    assert k["motion_q"]["q50"] == pytest.approx(0.5) and k["border_q"]["q90"] == pytest.approx(1.8)


def table_for_costs():
    def cell(wall, cpu, n=100):
        return {"decode_plain": {"wall_ms": 2.0, "cpu_ms": 5.0}, "decode_mv": {"wall_ms": wall, "cpu_ms": cpu}, "mv_obs": {"wall_ms": 0.4, "cpu_ms": 0.35},
                "mv_c1": {"wall_ms": 0.3, "cpu_ms": 0.31}, "pix_obs": {"wall_ms": 0.37, "cpu_ms": 0.36}, "pix_c1": {"wall_ms": 0.15, "cpu_ms": 0.14}}

    return {"sht": {"T1": cell(0, 0), "T4": cell(2.5, 6.0)}}


def test_derive_vm_costs_uses_export_delta_at_t4_and_margin():
    d = e04.derive_vm_costs(table_for_costs(), "sht", 1.25)
    assert d["export_wall_ms"] == pytest.approx(1.25 * 0.5) and d["export_cpu_ms"] == pytest.approx(1.25 * 1.0)
    assert d["mv_obs_ms"] == pytest.approx(1.25 * 0.4) and d["mv_c1_ms"] == pytest.approx(1.25 * 0.31) and d["pix_c1_ms_vm"] == pytest.approx(0.15)
    t = table_for_costs()
    t["sht"]["T4"]["decode_mv"]["wall_ms"] = 1.0  # noise below the plain decode never gives a negative overhead
    assert e04.derive_vm_costs(t, "sht")["export_wall_ms"] == 0.0


def test_k_mv_overrides_costs_motion_and_credit_only():
    cfg03 = {"datasets": {"ShanghaiTech": {"constants": {"costs": {"obs_ms": 0.366}, "motion_q": {"q50": 1}, "border_q": {"q50": 1}, "nu_safe": -0.9, "c1": {}}}}}
    cfg = {"vm": {"cost_table": table_for_costs()}, "mv": {"ShanghaiTech": {"observer_constants_test_dev": {"motion_q": {"q50": 7}, "border_q": {"q50": 8}}}}}
    K = e04.k_mv(cfg03, cfg, "ShanghaiTech")
    assert K["motion_q"] == {"q50": 7} and K["border_q"] == {"q50": 8} and K["nu_safe"] == -0.9
    assert K["mv"]["skip_retrieve"] and K["mv"]["export_cpu_ms"] == pytest.approx(1.25) and K["costs"]["c1_check_ms"] == pytest.approx(1.25 * 0.31)
    assert cfg03["datasets"]["ShanghaiTech"]["constants"]["costs"] == {"obs_ms": 0.366}  # deep copy: E03 constants untouched


# ---- adoption logic, gates, diagnostics (pure) ---------------------------------------------------------------------------
def ci(delta, lo, hi):
    return {"delta": delta, "lo": lo, "hi": hi}


def test_component_decision_path_a_needs_dev_ci_and_confirmation_direction():
    ok = rep.component_decision(ci(0.5, 0.1, 0.9), ci(0.2, -0.1, 0.5), ci(0.0, -1, 1), ci(0.0, -1, 1), 0.0, 0.0)
    assert ok["adopted"] and ok["path"] == "A"
    ci_includes_zero = rep.component_decision(ci(0.5, -0.1, 0.9), ci(0.2, 0.0, 0.5), ci(0.0, -1, 1), ci(0.0, -1, 1), 0.0, 0.0)
    assert not ci_includes_zero["adopted"]
    wrong_dir = rep.component_decision(ci(0.5, 0.1, 0.9), ci(-0.01, -0.5, 0.5), ci(0.0, -1, 1), ci(0.0, -1, 1), 0.0, 0.0)
    assert not wrong_dir["adopted"]


def test_component_decision_path_b_needs_5pct_and_no_degradation_on_both_splits():
    a0 = ci(0.0, -0.3, 0.3)
    ok = rep.component_decision(a0, a0, ci(-0.2, -0.5, 0.1), ci(-0.1, -0.4, 0.2), 0.06, 0.05)
    assert ok["adopted"] and ok["path"] == "B"
    assert not rep.component_decision(a0, a0, ci(-0.2, -0.5, 0.1), ci(-0.1, -0.4, 0.2), 0.049, 0.06)["adopted"]  # dev reduction < 5 %
    assert not rep.component_decision(a0, a0, ci(-0.26, -0.5, 0.1), ci(0.0, -0.4, 0.2), 0.10, 0.10)["adopted"]  # degraded more than 0.25 pt on dev
    assert not rep.component_decision(a0, a0, ci(0.0, -0.5, 0.1), ci(-0.5, -0.9, 0.2), 0.10, 0.10)["adopted"]  # not verified on confirmation


def test_neutral_component_stays_out():
    z = ci(0.0, -0.2, 0.2)
    assert not rep.component_decision(z, z, z, z, 0.0, 0.0)["adopted"]


def test_component_verdict_wording():
    yes, no = {"adopted": True}, {"adopted": False}
    assert rep.component_verdict({"ShanghaiTech": yes, "UBnormal": yes}) == "adopted"
    assert rep.component_verdict({"ShanghaiTech": no, "UBnormal": no}) == "not adopted"
    assert "dataset-conditional" in rep.component_verdict({"ShanghaiTech": yes, "UBnormal": no})
    assert "ShanghaiTech only" in rep.component_verdict({"ShanghaiTech": yes})


def test_qat_gate_requires_degradation_and_a_speedup():
    assert rep.qat_gate(-0.3, 0.0, 0.0, 0.2)["fires"]
    assert not rep.qat_gate(-0.1, -0.2, -0.24, 0.2)["fires"]
    assert not rep.qat_gate(-0.5, -0.5, -0.5, 0.0)["fires"]  # no CPU saved: nothing worth rescuing


def test_agreement_and_hist_stats():
    a = np.array([1, 1, 0, 0, 1, 0], bool)
    b = np.array([1, 0, 0, 0, 1, 1], bool)
    g = rep.agreement(a, b)
    assert g["agree"] == pytest.approx(4 / 6) and g["both_static"] == pytest.approx(2 / 6) and g["mv_static_pixel_moving"] == pytest.approx(1 / 6)
    assert rep.agreement(a, a)["kappa"] == pytest.approx(1.0)
    h = np.zeros(60)
    h[0], h[10] = 9, 1
    s = rep.hist_stats(h)
    assert s["n"] == 10 and s["mean"] == pytest.approx((9 * 0.005 + 1 * 0.105) / 10) and s["p90"] == pytest.approx(0.005)


# ---- codec: known translation and determinism (PyAV + libx264) --------------------------------------------------------------
def _have_av():
    try:
        import av  # noqa: F401

        return True
    except Exception:  # pragma: no cover
        return False


@pytest.mark.skipif(not _have_av(), reason="PyAV not installed")
def test_known_global_translation_is_recovered_from_exported_vectors():
    tr = rep._translation_check()
    assert tr["frames_valid"] >= 20
    assert abs(abs(tr["median_dx"]) - 4.0) <= 1.0 and abs(tr["median_dy"]) <= 0.5


@pytest.mark.skipif(not _have_av(), reason="PyAV not installed")
def test_encode_and_extraction_are_deterministic(tmp_path):
    rng = np.random.default_rng(3)
    frames = [(rng.random((64, 96, 3)) * 255).astype(np.uint8) for _ in range(10)]
    outs = []
    for k in range(2):
        p = tmp_path / f"c{k}.mp4"
        vmm.encode_frames(frames, "ShanghaiTech", p, 96, 64)
        outs.append(vmm.extract_clip(p))
    for key in ("mag", "dx", "dy", "cov", "ptype", "valid"):
        assert np.array_equal(outs[0][key], outs[1][key])
    assert len(outs[0]["valid"]) == 10 and not outs[0]["valid"][0] and outs[0]["valid"][1:].all()  # one record per frame, only the first frame is intra


# ---- extended simulator: MV accounting (needs E03 tables) ---------------------------------------------------------------------
def _world():
    names, labels = e00.dataset_clips(SHT)
    cfg, _ = e01.load_e00()
    pos = {c: i for i, c in enumerate(names)}
    dev = [pos[c] for c in cfg["datasets"][SHT]["splits"]["test_dev"]]
    conf = [pos[c] for c in cfg["datasets"][SHT]["splits"]["test_confirmation"]]
    pk = e02.WORK_DIR / "stages" / f"e02_sim__{SHT}__{e02.PRIMARY[SHT]}__locked.pkl"
    w = e03.World.from_parts(SHT, names, labels, dev, conf, pickle.loads(pk.read_bytes())["sentinel"])
    w.K = yaml.safe_load((e03.EXP_DIR / "config.yaml").read_text())["datasets"][SHT]["constants"]
    return w


@pytest.fixture(scope="module")
def world():
    if not HAVE_E03:
        pytest.skip("E03 tables not built")
    try:
        return _world()
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"world unavailable: {exc}")


@needs_e03
def test_export_overhead_is_charged_on_every_frame(world):
    c = world.cost(2.0)
    clips = world.dev_idx[:6]
    base = e03.total_cost(world, e03.simulate(world, c, e03.FixedPolicy(world, c, 36.0, 3), clips, light=True))
    K = copy.deepcopy(world.K)
    K["mv"] = {"export_wall_ms": 1.0, "export_cpu_ms": 1.0, "skip_retrieve": False}
    world.K = K
    world._cost.clear()
    try:
        c2 = world.cost(2.0)
        got = e03.total_cost(world, e03.simulate(world, c2, e03.FixedPolicy(world, c2, 36.0, 3), clips, light=True))
    finally:
        world.K = yaml.safe_load((e03.EXP_DIR / "config.yaml").read_text())["datasets"][SHT]["constants"]
        world._cost.clear()
    assert got - base == pytest.approx(1.0 * 2.0 * e00.FPS[SHT], rel=1e-9)  # 1 ms x s per frame x fps, every frame, observed or not


@needs_e03
def test_retrieve_skip_credit_only_saves_on_reused_frames(world):
    clips = world.dev_idx[:6]
    K0 = copy.deepcopy(world.K)
    K1 = copy.deepcopy(world.K)
    K1["mv"] = {"export_wall_ms": 0.0, "export_cpu_ms": 0.0, "skip_retrieve": True}
    try:
        res = {}
        for name, K, lvl in (("base", K0, 3), ("skip", K1, 3), ("skip_nocache", K1, 0), ("base_nocache", K0, 0)):
            world.K = K
            world._cost.clear()
            c = world.cost(2.0)
            R = e03.simulate(world, c, e03.FixedPolicy(world, c, 36.0, lvl), clips, light=True)
            res[name] = (e03.total_cost(world, R), R["stats"])
    finally:
        world.K = K0
        world._cost.clear()
    assert res["skip_nocache"][0] == pytest.approx(res["base_nocache"][0], rel=1e-12)  # without reuse every frame still needs its pixels
    assert res["skip"][0] <= res["base"][0] + 1e-9
    assert res["skip"][1]["retrieve_saved_ms"] >= 0 and res["base"][1]["retrieve_saved_ms"] == 0


@needs_e03
def test_matched_target_calibration_with_explicit_target(world):
    clips = world.dev_idx[:8]
    r = e04.run_cell_t(world, {"arm": "Cache-C1", "l1": 3, "l2": 0}, 2.0, 0.5, 600.0, clips, 40.0)
    assert r["ratio"] <= 1.005 and r["status"] in ("ok", "under") and r["hold"] == 39
    g = e04.run_cell_t(world, {"arm": "Cache-C1", "l1": 3, "l2": 0}, 2.0, 0.5, 600.0, clips, 40.0, knob=r["knob"])
    assert g["cost"] == pytest.approx(r["cost"]) and g["status"] == "given"


# ---- plan / config / outputs ---------------------------------------------------------------------------------------------------
def test_plan_preregisters_rule_and_gates():
    txt = (e04.EXP_DIR / "plan.md").read_text()
    for token in ("Pre-registered adoption rule", "0.25 pt", "5 %", "E4-D", "QAT", "Cadence-aware scorer", "not connected"):
        assert token in txt


@needs_run
def test_config_written_before_run_and_consistent():
    cfg = yaml.safe_load((e04.EXP_DIR / "config.yaml").read_text())
    assert cfg["experiment"] == "Q2-P1-E04" and cfg["training"] is False and cfg["c1_level"] == 3
    stamp = e04.sha(e04.EXP_DIR / "config.yaml")
    for ds in e04.DATASETS:
        assert pickle.loads(e04.run_path(ds).read_bytes())["stamp"] == stamp
    assert cfg["adoption_rule"]["degradation_pt"] == 0.25 and cfg["vm"]["pose_drift_vs_e02"]["pass"]
    assert cfg["inputs"]["models"]["ml0_int8_onnx"] == e04.sha(e04.ML0_INT8_ONNX)
    for ds in e04.DATASETS:
        thr = cfg["mv"][ds]["c1_thresholds_test_dev"]
        assert len(thr["roi"]) == 3 and thr["roi"] == sorted(thr["roi"])


@needs_run
def test_matched_cost_targets_equal_e03_and_cells_complete():
    cfg = yaml.safe_load((e04.EXP_DIR / "config.yaml").read_text())
    for ds in e04.DATASETS:
        run = pickle.loads(e04.run_path(ds).read_bytes())
        assert all(run["e03_targets_equal"].values())
        cells = {e03.cell_id(*c) for c in cfg["cells"]}
        for arm in ("C1-Pixel", "C1-MV", "C1-MV@W", "C1-MV-Advect", "Motion-Only-Pixel", "Motion-Only+MV"):
            assert {cid for (a, cid) in run["results"] if a == arm} == cells
        for cid in cells:
            r = run["results"][("C1-MV@W", cid)]
            assert r["knob"] == run["results"][("C1-Pixel", cid)]["knob"]  # path B keeps the schedule
            assert run["results"][("C1-MV-Advect", cid)]["knob"] == run["results"][("C1-MV", cid)]["knob"]
        if ds == SHT:
            assert {cid for (a, cid) in run["results"] if a == "C1-INT8"} == cells


@needs_run
def test_e03_replay_is_bit_exact_in_all_cells():
    for ds in e04.DATASETS:
        run = pickle.loads(e04.run_path(ds).read_bytes())
        for arm, cells in run["d1"].items():
            assert len(cells) == 12
            for cid, c in cells.items():
                assert c["series_equal"] and c["ideal_equal"] and c["cost_equal"] and c["knob_equal"] and c["target_equal"], (ds, arm, cid)


@needs_run
def test_path_a_arms_respect_the_matched_cost_target_or_are_flagged():
    for ds in e04.DATASETS:
        run = pickle.loads(e04.run_path(ds).read_bytes())
        for (arm, cid), r in run["results"].items():
            if arm.endswith("@W") or arm == "C1-MV-Advect":
                continue
            assert r["ratio"] <= 1.005 or r["status"] == "over", (ds, arm, cid, r["ratio"], r["status"])


@pytest.mark.skipif(not (e04.EXP_DIR / "summary.json").exists(), reason="report not generated")
def test_outputs_exist_and_agree():
    summary = json.loads((e04.EXP_DIR / "summary.json").read_text())
    for f in ("config.yaml", "results.csv", "decisions.jsonl", "summary.json", "report.md", "plan.md"):
        assert (e04.EXP_DIR / f).exists(), f
    assert summary["config_sha256"] == e04.sha(e04.EXP_DIR / "config.yaml")
    assert all(summary["rules"].values()), summary["rules"]
    assert summary["reproducible"]
    recs = [json.loads(line) for line in (e04.EXP_DIR / "decisions.jsonl").read_text().splitlines()]
    kinds = {r["type"] for r in recs}
    assert {"rule", "decision", "gate", "verdict", "calibration"} <= kinds
    # the report states the same verdicts as the summary
    rep_txt = (e04.EXP_DIR / "report.md").read_text()
    for key in ("mv", "int8"):
        assert summary["verdicts"][key] in rep_txt
    with (e04.EXP_DIR / "results.csv").open() as fh:
        header = fh.readline().strip().split(",")
        n = sum(1 for _ in fh)
    assert n == summary["rows_results_csv"] and "auc_delivered" in header
