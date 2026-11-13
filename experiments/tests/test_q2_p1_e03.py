"""Q2-P1-E03 core ablation: cheap observers, C1 / C2 validity rules, run-table replay (parity with E02 bursts), schedulers, queue / causality
invariants, cost matching, event metrics, bootstrap, verdict rules and - when the artifacts exist - the outputs."""

import json
import math
import pickle
from pathlib import Path

import numpy as np
import pytest
import yaml
from sklearn.metrics import roc_auc_score

from edgelab.analysis import q2_p1_e00 as e00
from edgelab.analysis import q2_p1_e01 as e01
from edgelab.analysis import q2_p1_e02 as e02
from edgelab.analysis import q2_p1_e03 as e03

SHT = "ShanghaiTech"
HAVE_TABLES = e03.tables_path(SHT).exists() and (e03.MOTION_DIR / SHT).exists()
needs_tables = pytest.mark.skipif(not HAVE_TABLES, reason="E03 run tables / motion cache not built")


# ---- observers (pure) --------------------------------------------------------------------------------------------------
def test_motion_observer_static_and_changed():
    rng = np.random.default_rng(0)
    a = rng.integers(0, 255, size=(60, 100)).astype(np.uint8)
    m, b, sh = e03.motion_observer(a, a)
    assert m == 0.0 and b == 0.0 and sh == 0.0
    c = a.copy()
    c[20:40, 30:60] = 255 - c[20:40, 30:60]
    m2, b2, _ = e03.motion_observer(a, c)
    assert m2 > 0.05 and b2 == 0.0  # activity in the middle: no border activity
    d = a.copy()
    d[:5, :] = 255 - d[:5, :]
    _, b3, _ = e03.motion_observer(a, d)
    assert b3 > 0.0


def test_motion_observer_compensates_global_luma_shift():
    a = np.full((60, 100), 100, dtype=np.uint8)
    m, _, sh = e03.motion_observer(a, a + 20)
    assert m == 0.0 and sh == 20.0


def test_box_cells_and_static_stats_shape_and_nan_tail():
    mask = e03.box_cells(np.array([[0.0, 0.0, 10.0, 10.0]]), 100.0, 100.0, dilate=0)
    assert mask.shape == (e03.GRID_H, e03.GRID_W) and mask[0, 0] and not mask[-1, -1]
    th = np.zeros((10, e03.GRID_H, e03.GRID_W), dtype=np.uint8)
    roi, ext = e03.static_stats(th, [np.zeros((0, 4))] * 10, 100.0, 100.0)
    assert roi.shape == (10, e03.A1) and np.isnan(roi[9]).all() and np.isfinite(roi[0]).all() and (roi[:6] == 0).all()
    assert np.isnan(roi[8, 1:]).all() and np.isfinite(roi[8, 0])


# ---- C1 reuse pattern --------------------------------------------------------------------------------------------------
def make_stats(n, roi_val, ext_val=0.0):
    return np.full((n, e03.A1), roi_val, dtype=np.float64), np.full((n, e03.A1), ext_val, dtype=np.float64)


def test_reuse_pattern_age_limit_and_forced_detect():
    roi, ext = make_stats(30, 0.0)
    reused, src = e03.reuse_pattern(roi, ext, 0, 30, 0.5, 0.5)
    assert not reused[0]  # offset 0 always runs the detector
    run = best = 0
    for r in reused:
        run = run + 1 if r else 0
        best = max(best, run)
    assert best <= e03.A1  # never older than A1 frames
    assert (np.arange(30) - src)[reused].max() <= e03.A1
    assert (src[~reused] == np.arange(30)[~reused]).all()


def test_reuse_pattern_threshold_is_inclusive_and_nan_forces_detect():
    roi, ext = make_stats(10, 0.5, 1.0)
    reused, _ = e03.reuse_pattern(roi, ext, 0, 10, 0.5, 1.0)  # equal to the threshold -> valid
    assert reused[1]
    roi2 = roi.copy()
    roi2[0, 0] = np.nan
    reused2, _ = e03.reuse_pattern(roi2, ext, 0, 10, 0.5, 1.0)
    assert not reused2[1]
    reused3, _ = e03.reuse_pattern(roi, ext, 0, 10, 0.49, 1.0)
    assert not reused3.any()


def test_reuse_pattern_is_causal():
    rng = np.random.default_rng(1)
    roi = rng.uniform(0, 1, size=(40, e03.A1))
    ext = rng.uniform(0, 1, size=(40, e03.A1))
    base, _ = e03.reuse_pattern(roi, ext, 0, 20, 0.5, 0.5)
    roi2, ext2 = roi.copy(), ext.copy()
    roi2[12:] = 99.0  # statistics of frames >= 12 are only needed by offsets >= 13
    ext2[12:] = 99.0
    alt, _ = e03.reuse_pattern(roi2, ext2, 0, 20, 0.5, 0.5)
    assert np.array_equal(base[:13], alt[:13])


def test_pack_unpack_bits_roundtrip_uint64():
    mask = np.zeros(40, dtype=bool)
    mask[[0, 3, 17, 35, 39]] = True
    packed = e03.pack_bits(mask)
    assert all(e03.unpack_bit(packed, i) == bool(mask[i]) for i in range(40))
    assert int(np.uint64(packed)) == packed


def test_thresholds_from_stats_quantiles():
    r = np.arange(101, dtype=float)
    thr = e03.thresholds_from_stats(r, r)
    assert thr["roi"] == [10.0, 25.0, 50.0] and thr["ext"] == [10.0, 25.0, 50.0]


# ---- schedules and calibration -----------------------------------------------------------------------------------------
def test_burst_starts_even_with_room():
    starts = e03.burst_starts(37.3, 400, 24)
    assert all(s % 2 == 0 and s <= 400 - 24 for s in starts) and starts[0] == 0
    assert all(b > a for a, b in zip(starts, starts[1:]))
    assert e03.burst_starts(48, 100, 24) == [0, 48]
    assert e03.burst_starts(48, 70, 24) == [0]


def test_calibrate_increasing_and_decreasing():
    inc = e03.calibrate(lambda x: 10.0 * x, 1.0, 1000.0, 500.0, increasing=True)
    assert inc["status"] == "ok" and 0.99 * 500 <= inc["cost"] <= 500
    dec = e03.calibrate(lambda x: 1000.0 / x, 1.0, 400.0, 100.0, increasing=False)
    assert dec["status"] == "ok" and 0.99 * 100 <= dec["cost"] <= 100
    under = e03.calibrate(lambda x: 10.0 * x, 1.0, 10.0, 500.0, increasing=True)
    assert under["status"] == "under" and under["knob"] == 10.0
    over = e03.calibrate(lambda x: 10.0 * x, 100.0, 1000.0, 50.0, increasing=True)
    assert over["status"] == "over" and over["knob"] == 100.0


# ---- metrics -----------------------------------------------------------------------------------------------------------
def test_label_events_and_alert_runs():
    lab = np.array([0, 1, 1, 0, 0, 1, 0, 1, 1, 1])
    assert e03.label_events(lab) == [(1, 2), (5, 5), (7, 9)]
    above = np.array([0, 1, 1, 1, 1, 1, 1, 0, 1, 1], dtype=bool)
    assert e03.alert_runs(above, 6) == [(1, 6)] and e03.alert_runs(above, 2) == [(1, 6), (8, 9)]


def test_event_metrics_delay_and_false_alerts():
    fps = 10.0
    lab = np.zeros(100, dtype=int)
    lab[30:50] = 1
    score = np.zeros(100)
    score[33:45] = 2.0  # alert run 33..44 -> raised at 38 -> delay 8 frames = 800 ms
    score[70:80] = 2.0  # alert far from the event -> false alert
    m = e03.event_metrics(lab, score, 1.0, fps)
    assert m["events"] == 1 and m["detected"] == 1 and m["hit_0.5"] == 0 and m["hit_1.0"] == 1
    assert m["delays_ms"] == [800.0] and m["false_alerts"] == 1 and m["normal_frames"] == 80
    score2 = np.zeros(100)
    score2[20:40] = 2.0  # alert already on when the event starts: delay 0
    m2 = e03.event_metrics(lab, score2, 1.0, fps)
    assert m2["delays_ms"] == [0.0] and m2["false_alerts"] == 0
    score3 = np.zeros(100)
    score3[30:34] = 2.0  # too short to alert
    assert e03.event_metrics(lab, score3, 1.0, fps)["detected"] == 0


def test_event_metrics_alert_raised_after_event_does_not_count():
    lab = np.zeros(60, dtype=int)
    lab[10:14] = 1
    score = np.zeros(60)
    score[12:30] = 2.0  # raised at frame 17 > event end 13
    m = e03.event_metrics(lab, score, 1.0, 10.0)
    assert m["detected"] == 0


# ---- bootstrap -----------------------------------------------------------------------------------------------------------
def make_videos(n=12, seed=0, with_ties=True):
    rng = np.random.default_rng(seed)
    labels, scores = [], []
    for _ in range(n):
        m = int(rng.integers(40, 80))
        y = (rng.uniform(size=m) < 0.4).astype(int)
        y[0], y[-1] = 0, 1
        s = rng.normal(size=m) + 0.8 * y
        if with_ties:
            s = np.round(s, 1)
        labels.append(y)
        scores.append(s)
    return labels, scores


def test_aucboot_point_matches_sklearn_with_ties():
    labels, scores = make_videos()
    ab = e03.AucBoot(labels, None, n_boot=50, seed=0)
    point, vec = ab.aucs(scores)
    ref = 100 * roc_auc_score(np.concatenate(labels), np.concatenate(scores))
    assert point == pytest.approx(ref, abs=1e-9)
    assert vec.shape == (50,) and np.isfinite(vec).all()
    sub = e03.AucBoot(labels, [1, 3, 5, 7], n_boot=10)
    p2, _ = sub.aucs([scores[i] for i in sub.idx])
    assert p2 == pytest.approx(100 * roc_auc_score(np.concatenate([labels[i] for i in [1, 3, 5, 7]]), np.concatenate([scores[i] for i in [1, 3, 5, 7]])), abs=1e-9)


def test_aucboot_resample_equals_explicit_weighting():
    labels, scores = make_videos(8, seed=3)
    ab = e03.AucBoot(labels, None, n_boot=5, seed=2)
    _, vec = ab.aucs(scores)
    for b in range(5):
        idx = np.repeat(np.arange(8), ab.W[b].astype(int))
        ref = 100 * roc_auc_score(np.concatenate([labels[i] for i in idx]), np.concatenate([scores[i] for i in idx]))
        assert vec[b] == pytest.approx(ref, abs=1e-9)


def test_contrast_of_identical_series_is_zero_and_signed():
    labels, scores = make_videos()
    ab = e03.AucBoot(labels, None, n_boot=200, seed=0)
    _, v1 = ab.aucs(scores)
    c0 = e03.contrast(np.stack([v1, v1]), np.stack([v1, v1]))
    assert c0["delta"] == 0 and c0["lo"] == 0 and c0["hi"] == 0 and not c0["excludes_zero"]
    rng = np.random.default_rng(5)
    noisy = [s + rng.normal(scale=3.0, size=len(s)) for s in scores]
    _, v2 = ab.aucs(noisy)
    c = e03.contrast(v1, v2)
    assert c["delta"] > 0 and c["lo"] > 0


# ---- verdict rules -------------------------------------------------------------------------------------------------------
def cc(delta, lo, hi):
    return {"delta": delta, "lo": lo, "hi": hi, "excludes_zero": lo > 0 or hi < 0}


def test_verdicts_h1_h2_h3():
    assert e03.verdict_h1(cc(1, 0.2, 2), cc(1, 0.1, 2)) == "supported"
    assert e03.verdict_h1(cc(-0.1, -1, 0.8), cc(1.2, 0.1, 2)) == "partially supported"
    assert e03.verdict_h1(cc(0.1, -1, 1), cc(0.1, -1, 1)) == "not supported"
    assert e03.verdict_h1(cc(-2, -3, -1), cc(1, 0.5, 2)) == "rejected"
    assert e03.verdict_h2(0.2, cc(2, 1, 3), -0.5) == "supported"
    assert e03.verdict_h2(0.2, cc(2, 1, 3), -3.0) == "partially supported"  # recall condition fails but non-inferior
    assert e03.verdict_h2(0.03, cc(2, 1, 3), 0.0) == "not supported"
    assert e03.verdict_h2(-0.01, cc(2, 1, 3), 0.0) == "rejected" and e03.verdict_h2(0.2, cc(-2, -3, -1), 0.0) == "rejected"
    assert e03.mechanism_dropped(0.01, cc(1, 0.5, 2)) and e03.mechanism_dropped(0.2, cc(0, -1, 0)) and not e03.mechanism_dropped(0.2, cc(1, 0.5, 2))
    assert e03.verdict_h3(cc(1, .1, 2), cc(1, .1, 2), cc(1, .1, 2), cc(1, .1, 2)) == "supported"
    assert e03.verdict_h3(cc(1, -.1, 2), cc(1, -.1, 2), cc(1, .1, 2), cc(1, .1, 2)) == "partially supported"
    assert e03.verdict_h3(cc(1, -.1, 2), cc(1, -.1, 2), cc(1, -.1, 2), cc(1, .1, 2)) == "not supported"


def test_stop_rule():
    ok = {a: cc(0.5, 0.1, 1) for a in ("Fixed", "Cache-Only", "Motion-Only", "Adaptive-Only", "Independent-Combo")}
    assert e03.stop_rule(ok, ["Adaptive-Only", "Cache-Only"])["retained"]
    beaten = dict(ok, **{"Cache-Only": cc(-1.5, -2.4, -0.8)})
    r = e03.stop_rule(beaten, ["Adaptive-Only", "Cache-Only"])
    assert not r["retained"] and r["beaten_by"] == ["Cache-Only"]
    no_gain = {a: cc(0.0, -0.5, 0.5) for a in ok}
    assert not e03.stop_rule(no_gain, ["Adaptive-Only", "Cache-Only"])["retained"]  # not beaten but also beats neither


def test_lock_rule_prefers_simplest_non_inferior_that_beats_fixed():
    avg = {"Fixed": 70.0, "Cache-Only": 73.0, "Motion-Only": 69.0, "Adaptive-Only": 70.5, "Independent-Combo": 73.2, "Joint-Controller": 71.0}
    valid = {a: True for a in avg}
    vs_best = {a: cc(avg[a] - 73.2, -3, 0.1) for a in avg}
    vs_best["Cache-Only"] = cc(-0.2, -0.2, 0.1)  # non-inferior (> -0.25)
    vs_fixed = {a: cc(avg[a] - 70.0, 0.5 if avg[a] > 71 else -0.5, 3) for a in avg}
    r = e03.lock_rule(avg, valid, vs_best, vs_fixed)
    assert r["best"] == "Independent-Combo" and r["lock"] == "Cache-Only"
    r2 = e03.lock_rule(avg, valid, vs_best, {a: cc(0, -0.5, 1) for a in avg})
    assert r2["lock"] == "Fixed" and r2["note"]
    r3 = e03.lock_rule(avg, dict(valid, **{"Independent-Combo": False}), vs_best, vs_fixed)
    assert r3["best"] == "Cache-Only"


# ---- replay on the real run tables (needs the caches) ------------------------------------------------------------------------
def _world(ds=SHT):
    names, labels = e00.dataset_clips(ds)
    cfg, _ = e01.load_e00()
    pos = {c: i for i, c in enumerate(names)}
    dev = [pos[c] for c in cfg["datasets"][ds]["splits"]["test_dev"]]
    conf = [pos[c] for c in cfg["datasets"][ds]["splits"]["test_confirmation"]]
    pk = e02.WORK_DIR / "stages" / f"e02_sim__{ds}__{e02.PRIMARY[ds]}__locked.pkl"
    sentinel = pickle.loads(pk.read_bytes())["sentinel"]
    w = e03.World.from_parts(ds, names, labels, dev, conf, sentinel)
    w.K = _constants(w)
    return w


def _constants(w):
    cp = e03.EXP_DIR / "config.yaml"
    if cp.exists():
        return yaml.safe_load(cp.read_text())["datasets"][w.ds]["constants"]
    return e03.derive_constants(w)


@pytest.fixture(scope="module")
def world():
    if not HAVE_TABLES:
        pytest.skip("E03 run tables not built")
    try:
        return _world()
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"world unavailable: {exc}")


@needs_tables
def test_fixed_burst_parity_with_e02_bursts(world):
    """E3-D1: F(W > L) at zero delay reproduces E02's burst{W} ideal AUC (all / dev / confirmation)."""
    ref = json.loads((e02.EXP_DIR / "summary.json").read_text())["headline"][f"{SHT}/ML0_ncnn/locked"]
    c = world.cost(2.0)
    allc = list(range(len(world.names)))
    for W in (36, 72):
        R = e03.simulate(world, c, e03.FixedPolicy(world, c, W), allc)
        ser, _, _ = e03.delivery(world, R, W - 1, ideal=True)
        got = e01.auc_subsets(world.labels, ser, world.dev_idx, world.conf_idx)
        for k in got:
            assert got[k] == pytest.approx(ref[f"burst{W}"]["ideal"][k], abs=0.02)


@needs_tables
def test_full_pipeline_cost_matches_e02_k1(world):
    c = world.cost(2.0)
    R = e03.simulate(world, c, e03.FixedPolicy(world, c, 1.0), list(range(len(world.names))), light=True)
    ref = json.loads((e02.EXP_DIR / "summary.json").read_text())["headline"][f"{SHT}/ML0_ncnn/locked"]["stride1"]["T4_s2.0_b1.00"]["cpu_ms_per_s"]
    assert e03.total_cost(world, R) == pytest.approx(ref, rel=0.01)


@needs_tables
@pytest.mark.parametrize("cfg", [{"arm": "Motion-Only", "theta": "q70", "extend": True, "p": 1}, {"arm": "Adaptive-Only", "tau0": 1.4, "extend": True, "p": 2},
                                 {"arm": "Joint-Controller", "tau0": 1.4, "extend": False, "p": 1, "l1": 3, "l2": 2}])
def test_cost_matching_budget_and_queue_invariants(world, cfg):
    """E3-D4 / E3-D6 on a dev stream: realised cost within [0.97, 1.0] of the target, FIFO invariants, landing >= completion."""
    devc = list(world.dev_idx)
    r = e03.run_cell(world, cfg, 2.0, 0.50, devc, "dev", 40.0, full=True)
    assert 0.97 <= r["ratio"] <= 1.005 and r["status"] == "ok"
    R = r["R"]
    ser, st, info = e03.delivery(world, R, r["hold"])
    soff = np.concatenate([[0], np.cumsum(world.n_frames[devc])])
    ck = e02.check_queue_invariants(np.arange(len(R["fin"])) * world.dt, R["svc"], R["fin"], info["landing"], R["w_t"], world.dt, R["w_pos"], soff)
    assert all(ck.values())
    assert (R["wait"] >= 0).all() and np.isfinite(R["cpu"]).all()


@needs_tables
def test_zero_delay_relanding_equals_ideal_and_truncation_is_prefix_stable(world):
    c = world.cost(2.0)
    devc = list(world.dev_idx)
    cfg = {"arm": "Adaptive-Only", "tau0": 1.4, "extend": False, "p": 1}
    pol = lambda: e03.make_policy(world, c, cfg, 60.0, 40.0)  # noqa: E731
    R = e03.simulate(world, c, pol(), devc)
    R0 = dict(R, fin=np.arange(len(R["fin"])) * world.dt)
    s0, _, _ = e03.delivery(world, R0, 39)
    si, _, _ = e03.delivery(world, R, 39, ideal=True)
    assert all(np.array_equal(a, b) for a, b in zip(s0, si))
    full, _, _ = e03.delivery(world, R, 39)
    k = len(devc) // 2
    Rt = e03.simulate(world, c, pol(), devc[:k])
    part, _, _ = e03.delivery(world, Rt, 39)
    assert all(np.array_equal(a, b) for a, b in zip(part[: k - 1], full[: k - 1]))  # earlier clips unaffected by later frames


@needs_tables
def test_policy_never_reads_tables_of_unrequested_frames(world):
    """E3-D3: poisoning every table row whose completion frame was not observed changes nothing."""
    c = world.cost(2.0)
    clips = [world.dev_idx[0], world.conf_idx[0]]
    cfg = {"arm": "Joint-Controller", "tau0": 1.4, "extend": False, "p": 1, "l1": 3, "l2": 2}
    tr = {0: [], 1: []}
    R1 = e03.simulate(world, c, e03.make_policy(world, c, cfg, 50.0, 40.0), clips, trace=tr)
    saved = [list(col) for col in world.rn]
    E1, poisoned = world.ext + 1, 0
    for pos, ci in enumerate(clips):
        obs = {x[0] for x in tr[pos] if x[1]}
        n, sb = int(world.n_frames[ci]), int(world.start_base[ci])
        for st in range((n - world.L) // e03.START_GRID + 1):
            for e in range(E1):
                t = st * e03.START_GRID + world.L - 1 + e
                if t < n and t not in obs:
                    k = (sb + st) * E1 + e
                    for r in range(world.row_ptr[k], world.row_ptr[k + 1]):
                        for col in world.rn:
                            col[r] = float("nan")
                        poisoned += 1
    try:
        R2 = e03.simulate(world, c, e03.make_policy(world, c, cfg, 50.0, 40.0), clips)
    finally:
        world.rn = saved
    assert poisoned > 0
    assert np.array_equal(R1["fin"], R2["fin"]) and np.array_equal(R1["w_val"], R2["w_val"])


@needs_tables
def test_feedback_only_from_finished_jobs(world):
    c = world.cost(2.0)
    pol = e03.make_policy(world, c, {"arm": "Adaptive-Only", "tau0": 1.4, "extend": False, "p": 1}, 80.0, 40.0)
    seen = []
    orig_feedback, orig_decide = pol.feedback, pol.decide
    pol.feedback = lambda f, v, q: (seen.append(("fb", f)), orig_feedback(f, v, q))
    pol.decide = lambda t, n, g, now, wait, rl, cs: (seen.append(("job", now)), orig_decide(t, n, g, now, wait, rl, cs))[1]
    e03.simulate(world, c, pol, list(world.dev_idx[:3]))
    pending, n_fb = [], 0
    for kind, x in seen:
        if kind == "fb":
            pending.append(x)
            n_fb += 1
        else:  # feedback is delivered right before the decision of the job that starts at `x`: only results finished by then
            assert all(f <= x + 1e-9 for f in pending)
            pending = []
    assert n_fb > 0


@needs_tables
def test_c1_c2_validity_rules_in_simulation(world):
    c = world.cost(2.0)
    devc = list(world.dev_idx)
    cfg = {"arm": "Cache-Only", "l1": 3, "l2": 2}
    r = e03.run_cell(world, cfg, 2.0, 1.0, devc, "dev", 1.0, full=True, knob=1.0)
    S = r["R"]["stats"]
    assert S["reuse_frames"] > 0 and S["win_reuse"] > 0
    assert S["c2_age_max"] <= e03.A2
    assert S["c1_age_sum"] / S["reuse_frames"] <= e03.A1
    # a source with elevated anomaly (normality below nu_safe) is never reused: with an impossible nu_safe C2 never fires
    old = world.K["nu_safe"]
    world.K["nu_safe"] = float("inf")
    try:
        R = e03.simulate(world, c, e03.FixedPolicy(world, c, 1.0, 0, world.K["tau2"][2]), devc, light=True)
    finally:
        world.K["nu_safe"] = old
    assert R["stats"]["win_reuse"] == 0
    # reuse never crosses a clip boundary: C2 state is per clip -> the first window of every clip is scored fresh
    c2 = e03.simulate(world, c, e03.FixedPolicy(world, c, 1.0, 0, world.K["tau2"][2]), devc[:2], light=True)
    assert c2["stats"]["win_cand"] > 0 and c2["stats"]["win_fresh"] > 0


@needs_tables
def test_token_bucket_never_exceeds_capacity_and_runs_start_on_even_frames(world):
    c = world.cost(2.0)
    pol = e03.make_policy(world, c, {"arm": "Adaptive-Only", "tau0": 0.8, "extend": True, "p": 1}, 500.0, 40.0)
    e03.simulate(world, c, pol, list(world.dev_idx[:4]))  # raises AssertionError if a run starts where no full run fits
    assert pol.tokens <= pol.cap + 1e-9


@needs_tables
def test_motion_cache_frame_counts_match_labels(world):
    for ci in (0, 5, len(world.names) - 1):
        z = np.load(e03.motion_path(SHT, world.names[ci]))
        assert len(z["motion"]) == len(world.labels[ci]) and np.isnan(z["motion"][0])


@needs_tables
def test_chain_tables_cover_every_start_and_offset(world):
    tb = world.tb
    assert len(tb["row_ptr"]) == (int(world.start_base[-1]) * (world.ext + 1)) + 1
    assert (np.diff(tb["row_ptr"]) >= 0).all() and tb["row_ptr"][-1] == len(tb["row_norm"])
    assert tb["row_norm"].shape[1] == 1 + len(e03.LEVELS)
    # column 0 is the unmodified score; columns 1..3 are the C1 levels
    diff = tb["row_norm"][:, 1:] != tb["row_norm"][:, :1]
    frac = diff.mean(axis=0)
    assert 0 < frac[0] <= frac[1] <= frac[2] < 1  # more permissive C1 levels modify more windows


# ---- plan / config / outputs --------------------------------------------------------------------------------------------------
def test_cells_and_arms_follow_the_plan():
    assert len(e03.CELLS) == 12 and (2.0, 0.25) in e03.CELLS and (3.0, 1.0) in e03.CELLS
    assert set(e03.HEADLINE_ARMS) <= set(e03.ARMS) and len(e03.HEADLINE_ARMS) == 6
    assert e03.SIMPLICITY.index("Fixed") < e03.SIMPLICITY.index("Cache-Only") < e03.SIMPLICITY.index("Adaptive-Only") < e03.SIMPLICITY.index("Joint-Controller")


def test_plan_preregisters_hypotheses_and_rules():
    text = (e03.EXP_DIR / "plan.md").read_text()
    for token in ("H1", "H2", "H3", "Stop rule", "Lock rule", "E3-D1", "E3-D7", "non-inferiority", "Amendments"):
        assert token in text


def test_config_written_before_run_and_consistent():
    p = e03.EXP_DIR / "config.yaml"
    if not p.exists():
        pytest.skip("lock not run")
    cfg = yaml.safe_load(p.read_text())
    assert [tuple(c_) for c_ in cfg["cells"]] == e03.CELLS
    for ds in e03.DATASETS:
        sel = cfg["datasets"][ds]["tuning"]["selected"]
        assert set(sel) == set(e03.ARMS)
        assert cfg["datasets"][ds]["constants"]["theta"] > 0


def test_outputs_exist_and_agree():
    sp = e03.EXP_DIR / "summary.json"
    if not sp.exists():
        pytest.skip("report not run")
    s = json.loads(sp.read_text())
    for name in ("plan.md", "config.yaml", "results.csv", "decisions.jsonl", "summary.json", "report.md"):
        assert (e03.EXP_DIR / name).exists() and (e03.EXP_DIR / name).stat().st_size > 100
    assert s["config_sha256"] == e00.sha256_file(e03.EXP_DIR / "config.yaml")
    assert all(all(v.values()) for v in s["rules"].values())
    kinds = {json.loads(line)["type"] for line in (e03.EXP_DIR / "decisions.jsonl").read_text().splitlines()}
    assert {"rule", "selection", "calibration", "verdict", "controller_trace"} <= kinds
    h = s["headline"]
    assert h["H1"] in ("supported", "partially supported", "not supported", "rejected")
    n_rows = sum(1 for _ in (e03.EXP_DIR / "results.csv").open()) - 1
    assert n_rows == s["rows_results_csv"] == (107 + 211) * len(e03.ARMS) * len(e03.CELLS)
