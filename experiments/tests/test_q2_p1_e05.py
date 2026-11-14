"""Q2-P1-E05 confirmation: guard policy (identity + shed behaviour), non-oracle C1 frames, event strata / per-event outcomes (parity with E03),
bootstraps, drift / bounded-queue / RSS metrics, verdict rules, offline delivery of a real run, and - when the artifacts exist - the outputs."""

import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from edgelab.analysis import q2_p1_e03 as e03
from edgelab.analysis import q2_p1_e05 as e05

SHT, UB = e05.SHT, e05.UB
OUT = e05.EXP_DIR
HAVE_OUT = all((OUT / f).exists() for f in ("results.csv", "decisions.jsonl", "summary.json", "report.md"))
needs_out = pytest.mark.skipif(not HAVE_OUT, reason="E05 outputs not generated")


# ---- guard policy ------------------------------------------------------------------------------------------------------
def drive(pol, waits, n=240, L=24):
    """Toy single-clip driver: runs that were started are L frames long and end by themselves; an aborted run has rl = 0 at the next frame."""
    pol.start_clip(n)
    rl, out = 0, []
    for t in range(n):
        obs, score, _, _ = pol.decide(t, n, t, 0.0, float(waits[t]), rl, True)
        out.append((bool(obs), bool(score)))
        rl = (rl - 1 if rl > 0 else L - 1) if obs else 0
    return out


def toy_policy(cls, period, **kw):
    return cls(SimpleNamespace(L=24), None, period, **kw)


@pytest.mark.parametrize("period", [60.0, 87.4, 30.1, 12.0, 1.0])
def test_guard_is_identity_when_never_triggered(period):
    waits = np.minimum(np.linspace(0, 390, 240), 390)  # never above the shed threshold (400 ms)
    a = drive(toy_policy(e03.FixedPolicy, period), waits)
    g = toy_policy(e05.GuardedFixedPolicy, period)
    b = drive(g, waits)
    assert a == b and g.info()["shed_frames"] == 0 and g.info()["guard_on_events"] == 0


def test_guard_defers_a_burst_start_and_resumes():
    waits = np.zeros(240)
    waits[40:100] = 500.0
    a = drive(toy_policy(e03.FixedPolicy, 60.0), waits)
    g = toy_policy(e05.GuardedFixedPolicy, 60.0)
    b = drive(g, waits)
    assert [x[0] for x in a[60:84]] == [True] * 24  # the unguarded run at t = 60 happens
    assert not any(x[0] for x in b[60:84]) and not any(x[1] for x in b[60:84])  # the guarded start is deferred (lost)
    assert a[:40] == b[:40] and a[120:] == b[120:]  # before the overload and after it everything is identical
    info = g.info()
    assert info["guard_on_events"] == 1 and info["deferred_starts"] == 1 and info["shed_frames"] == 60 and info["aborted_run_frames"] == 0


def test_guard_aborts_the_running_run_and_never_scores_it():
    waits = np.zeros(240)
    waits[10:31] = 800.0
    a = drive(toy_policy(e03.FixedPolicy, 60.0), waits)
    g = toy_policy(e05.GuardedFixedPolicy, 60.0)
    b = drive(g, waits)
    assert a[23][1] and not b[23][1]  # the window completing at t = 23 is lost
    assert not any(x[0] for x in b[10:31])
    assert g.info()["aborted_run_frames"] == 1 and b[60:] == a[60:]


def test_guard_hysteresis():
    g = toy_policy(e05.GuardedFixedPolicy, 60.0)
    g.start_clip(240)
    for wait, shed in [(100, False), (399, False), (401, True), (300, True), (150, True), (149, False), (300, False)]:
        g.decide(1, 240, 1, 0.0, wait, 0, True)
        assert g.shed is shed, wait


def test_guard_continuous_schedule_resumes_on_the_next_free_frame():
    g = toy_policy(e05.GuardedFixedPolicy, 1.0)
    g.start_clip(100)
    assert g.decide(0, 100, 0, 0.0, 0.0, 0, True)[0]
    assert g.decide(1, 100, 1, 0.0, 500.0, 5, True)[:2] == (False, False)  # shed: the in-flight run is aborted
    assert not g.decide(2, 100, 2, 0.0, 100.0, 0, False)[0]  # guard off but no room yet
    assert g.decide(3, 100, 3, 0.0, 100.0, 0, True)[0] and g.info()["resumed_runs"] == 1


def test_make_policy5_picks_the_guarded_class():
    w, c = SimpleNamespace(L=24), None
    assert type(e05.make_policy5(w, c, e05.ARMS["Cache-C1"], 40.0)) is e03.FixedPolicy
    assert isinstance(e05.make_policy5(w, c, e05.ARMS["Cache-C1+Guard"], 40.0), e05.GuardedFixedPolicy)
    assert e05.ARMS["Cache-C1"]["lvl"] == 3 and e05.ARMS["Fixed"]["lvl"] == 0 and e05.ARMS["Cache-C1+Guard"]["guard"]


# ---- non-oracle C1 -----------------------------------------------------------------------------------------------------
def test_retrack_frames_substitute_the_source_detections():
    frames = [f"f{i}" for i in range(6)]
    reused = np.array([False, True, True, False, True, False])
    src = np.array([0, 0, 0, 3, 3, 5])
    assert e05.retrack_frames(frames, reused, src) == ["f0", "f0", "f0", "f3", "f3", "f5"]
    assert e05.retrack_frames(frames, np.zeros(6, dtype=bool), np.arange(6)) == frames


def test_retrack_source_ok_rules():
    reused = np.array([False, True, True, False, True])
    assert e05.retrack_source_ok(reused, np.array([0, 0, 0, 3, 3]))
    assert not e05.retrack_source_ok(reused, np.array([0, 2, 0, 3, 3]))  # source in the future / reused
    assert not e05.retrack_source_ok(np.array([False] * 7 + [True]), np.array([0] * 7 + [0]))  # older than A1
    assert not e05.retrack_source_ok(np.array([False, True, True]), np.array([0, 0, 1]))  # source is itself a reused frame


# ---- strata ------------------------------------------------------------------------------------------------------------
def test_short_event_thresholds():
    assert e05.short_frames(SHT) == 60 and e05.short_frames(UB) == 75


def test_track_summary_entries_and_crowd():
    tracks = {1: (np.arange(0, 50), None), 2: (np.arange(10, 40), None), 3: (np.arange(20, 25), None), 4: (np.arange(1, 30), None)}
    starts, count = e05.track_summary(tracks, 60)
    assert starts.tolist() == [10]  # pid 1 exists at the clip start, pid 4 starts at frame 1 (< 3), pid 3 has < 12 poses
    assert count[0] == 1 and count[22] == 4 and count[25] == 3 and count[55] == 0


def test_classify_event_strata():
    n = 400
    starts, active = np.array([100]), np.full(n, 2)
    motion = np.zeros(n)
    base = e05.classify_event(SHT, 130, 160, starts, active, motion, 1.0)
    assert base == {"short": True, "entry": False, "fast": False, "crowd": False}  # the entry (frame 100) is 30 frames before the onset: outside +-24
    far = e05.classify_event(SHT, 200, 300, starts, active, motion, 1.0)
    assert far["entry"] is False and far["short"] is False
    near = e05.classify_event(SHT, 110, 200, starts, active, motion, 1.0)
    assert near["entry"] is True
    m2 = motion.copy()
    m2[200:224] = 5.0
    assert e05.classify_event(SHT, 200, 300, starts, active, m2, 1.0)["fast"] is True
    act2 = active.copy()
    act2[200:224] = 4
    assert e05.classify_event(SHT, 200, 300, starts, act2, motion, 1.0)["crowd"] is True


def test_classify_event_duration_boundary():
    z = np.zeros(500)
    for ds, lim in ((SHT, 60), (UB, 75)):
        s = 100
        assert e05.classify_event(ds, s, s + lim - 1, np.array([]), z, z, 1.0)["short"] is True
        assert e05.classify_event(ds, s, s + lim, np.array([]), z, z, 1.0)["short"] is False


# ---- per-event outcomes --------------------------------------------------------------------------------------------------
def synthetic_clip(seed=0, n=300):
    rng = np.random.default_rng(seed)
    label = np.zeros(n, dtype=int)
    label[50:90] = 1
    label[150:158] = 1
    label[220:260] = 1
    series = rng.normal(1.0, 0.1, n)
    series[55:75] = 2.0
    series[150:158] = 2.0
    series[240:245] = 2.0  # only 5 frames: not an alert
    return label, series


def test_per_event_delays_equal_e03_event_metrics():
    label, series = synthetic_clip()
    ev = e03.label_events(label)
    d = e05.per_event_delays(ev, series, 1.5)
    m = e03.event_metrics(label, series, 1.5, 24.0)
    fin = d[np.isfinite(d)]
    assert np.allclose(1000.0 * fin / 24.0, m["delays_ms"]) and len(fin) == m["detected"] == 2
    assert np.isinf(d[2])


def test_per_event_delays_alert_before_event_counts_with_delay_zero():
    label = np.zeros(100, dtype=int)
    label[40:60] = 1
    series = np.zeros(100)
    series[30:50] = 5.0
    d = e05.per_event_delays([(40, 59)], series, 1.0)
    assert d[0] == 0.0


def test_window_mask_pads_and_clips():
    m = e05.window_mask(100, [(30, 40), (95, 99)], 10.0, pad_s=2.0)
    assert m[10] and m[60] and not m[9] and not m[61] and m[75] and m[99]
    assert not e05.window_mask(100, [], 10.0).any()


def test_wquantile_matches_numpy_for_unit_weights():
    v = np.sort(np.random.default_rng(1).normal(size=51))
    q = e05.wquantile(v, np.ones((1, 51)), 0.5)[0]
    assert q == np.median(v)
    w = np.zeros((1, 51))
    assert np.isnan(e05.wquantile(v, w, 0.5)[0])
    assert e05.wquantile(v, np.eye(51)[:1] * 3, 0.9)[0] == v[0]


# ---- bootstraps ------------------------------------------------------------------------------------------------------------
def test_event_boot_rate_point_and_interval():
    eb = e05.EventBoot(10, 500, 0)
    clip = np.repeat(np.arange(10), 4)
    hit = np.zeros(40, dtype=bool)
    hit[:20] = True
    pt, boot = eb.rate(clip, hit)
    assert pt == 50.0 and boot.shape == (500,) and 30 < np.nanmean(boot) < 70
    lo, hi = e05.ci95(boot)
    assert lo < 50 < hi
    pt2, boot2 = eb.rate(clip, np.ones(40, dtype=bool))
    assert pt2 == 100.0 and np.all(boot2 == 100.0)


def test_event_boot_is_paired_and_deterministic():
    a, b = e05.EventBoot(12, 300, 0), e05.EventBoot(12, 300, 0)
    assert np.array_equal(a.W, b.W) and (a.W.sum(axis=1) == 12).all()
    clip = np.arange(12)
    pa, ba = a.rate(clip, np.ones(12, dtype=bool))
    pb, bb = a.rate(clip, np.ones(12, dtype=bool))
    d = e05.diff_ci(pa, ba, pb, bb)
    assert d == {"delta": 0.0, "lo": 0.0, "hi": 0.0}


def test_event_boot_quantile_point_is_the_median():
    eb = e05.EventBoot(5, 200, 0)
    clip = np.array([0, 1, 2, 3, 4, 0, 1])
    vals = np.array([100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0])
    pt, boot = eb.quantile(clip, vals, 0.5)
    assert pt == 400.0 and boot.shape == (200,)
    pt0, b0 = eb.quantile(np.array([], dtype=int), np.array([]), 0.5)
    assert math.isnan(pt0) and np.isnan(b0).all()


def test_ci95_drops_nans_and_handles_empty():
    assert e05.ci95(np.array([np.nan, np.nan])) == (pytest.approx(float("nan"), nan_ok=True), pytest.approx(float("nan"), nan_ok=True))
    lo, hi = e05.ci95(np.arange(101, dtype=float))
    assert lo == pytest.approx(2.5) and hi == pytest.approx(97.5)


# ---- sustained operation ------------------------------------------------------------------------------------------------
def test_segment_stats_blocks_and_values():
    dt = 1000.0 / 24.0
    wait = np.concatenate([np.full(1440, 10.0), np.full(1440, 30.0), np.full(100, 50.0)])
    seg = e05.segment_stats(wait, dt, minutes=1.0)
    assert [s["frames"] for s in seg] == [1440, 1440, 100]
    assert seg[0]["wait_mean_ms"] == 10.0 and seg[1]["wait_max_ms"] == 30.0 and seg[2]["wait_p99_ms"] == 50.0
    assert seg[1]["start_min"] == pytest.approx(1.0)


def test_drift_test_detects_growth_and_stationarity():
    flat = e05.drift_test(np.full(900, 20.0))
    assert flat["drift_ms"] == 0.0 and flat["slope_ms_per_frame"] == pytest.approx(0.0, abs=1e-12)
    ramp = e05.drift_test(np.arange(900, dtype=float))
    assert ramp["drift_ms"] == pytest.approx(600.0) and ramp["slope_ms_per_frame"] == pytest.approx(1.0)


def test_bounded_rules():
    dt, size = 1000.0 / 24.0, (856, 480)
    ok = e05.bounded(np.full(1000, 100.0), dt, size, 900.0)
    assert ok["bounded"] and ok["drift_ms"] == 0.0 and 0 < ok["queue_mem_mb"] < 200
    assert not e05.bounded(np.arange(1000, dtype=float) * 5, dt, size, 900.0)["bounded"]  # drift + max
    spike = np.full(1000, 10.0)
    spike[3] = 2500.0
    assert not e05.bounded(spike, dt, size, 900.0)["bounded"]  # max wait above the 2,000 ms cap
    p99 = np.full(1000, 10.0)
    p99[:20] = 2100.0
    assert not e05.bounded(p99, dt, size, 900.0)["bounded"]


def test_queue_memory_grows_with_the_wait():
    dt, size = 1000.0 / 24.0, (856, 480)
    m0 = e05.queue_memory_mb(900.0, 0.0, dt, size)
    m1 = e05.queue_memory_mb(900.0, 2000.0, dt, size)
    assert m1 > m0 > 900.0
    assert m1 - m0 == pytest.approx(2000.0 / dt * 856 * 480 * 3 / 2 ** 20)


def test_rss_growth():
    g = e05.rss_growth({0: 800.0, 2500: 1000.0, 5000: 1010.0, 10000: 1030.0})
    assert g["rss_ref_mb"] == 1000.0 and g["growth_pct"] == pytest.approx(3.0)
    assert e05.rss_growth({0: 100.0, 10000: 110.0})["growth_pct"] == pytest.approx(10.0)


# ---- verdict rules -------------------------------------------------------------------------------------------------------
def ci(delta, lo, hi):
    return {"delta": delta, "lo": lo, "hi": hi}


def test_verdict_lock_and_replication():
    assert e05.verdict_lock(ci(2.0, 1.0, 3.0)) == "confirmed"
    assert e05.verdict_lock(ci(-2.0, -3.0, -1.0)) == "rejected"
    assert e05.verdict_lock(ci(0.3, -1.0, 1.5)) == "directionally consistent"
    assert e05.verdict_lock(ci(-0.3, -1.0, 1.5)) == "not supported"
    assert e05.verdict_replication(ci(2.0, 1.0, 3.0)) == "replicated"
    assert e05.verdict_replication(ci(-2.0, -3.0, -1.0)) == "not replicated"
    assert e05.verdict_replication(ci(0.2, -1.0, 1.0)).startswith("no significant effect")
    assert "negative" in e05.verdict_replication(ci(-0.2, -1.0, 1.0))
    assert e05.verdict_confirmed(ci(1.0, 0.1, 2.0)) == "confirmed"
    assert e05.verdict_confirmed(ci(-1.0, -2.0, -0.1)) == "rejected"
    assert e05.verdict_confirmed(ci(0.0, -1.0, 1.0)) == "not confirmed"


def test_verdict_retrack():
    oracle = ci(2.0, 1.0, 3.0)
    assert e05.verdict_retrack(ci(1.2, 0.4, 2.0), oracle) == "holds without oracle identity"
    assert e05.verdict_retrack(ci(0.8, 0.1, 1.5), oracle) == "attenuated"  # < 50 % of the oracle gain
    assert e05.verdict_retrack(ci(0.8, -0.2, 1.5), oracle) == "attenuated"
    assert e05.verdict_retrack(ci(-0.5, -1.0, -0.1), oracle) == "does not hold"
    assert e05.verdict_retrack(ci(-0.1, -1.0, 0.5), oracle) == "does not hold"


def test_verdict_guard():
    v = e05.verdict_guard(ci(0.0, -0.1, 0.1), ci(4.0, 1.0, 7.0))
    assert v == {"no_harm": "non-inferior", "overload": "guard helps"}
    v = e05.verdict_guard(ci(-0.5, -0.9, -0.3), ci(-4.0, -7.0, -1.0))
    assert v == {"no_harm": "inferior", "overload": "guard hurts"}
    assert e05.verdict_guard(None, ci(0.0, -1.0, 1.0)) == {"no_harm": "not testable", "overload": "no significant effect"}
    assert e05.verdict_guard(ci(0, -0.1, 0.1), None)["overload"] == "not testable"


def test_verdict_noninferiority_rules():
    assert e05.verdict_noninf_recall(ci(-1.0, -4.9, 2.0)) == "non-inferior"
    assert e05.verdict_noninf_recall(ci(-1.0, -5.1, 2.0)) == "not shown non-inferior"
    assert e05.verdict_delay(ci(10.0, -50.0, 249.0)) == "not delayed"
    assert e05.verdict_delay(ci(10.0, -50.0, 251.0)) == "not shown"
    assert e05.verdict_delay(ci(float("nan"), float("nan"), float("nan"))) == "not shown"


# ---- offline delivery of a real run --------------------------------------------------------------------------------------
def vm_stub(win, s=1.0):
    dt = 1000.0 / 24.0
    arr = {"n_clip": np.array([20, 20]), "win_rec": np.array(win, dtype=np.float64).reshape(-1, 7)}
    meta = {"s_emul": s, "sentinel": -0.5, "hold": 0, "cap_ms": 2000.0}
    return arr, meta, dt


def test_vm_delivery_offline_lands_windows_and_drops_late_ones():
    # rows: pos, t, g, fin_ms, vmin, late, beyond; the delivered series is the causally smoothed anomaly score (-normality), sentinel level = 0.5
    dt = 1000.0 / 24.0
    win = [[0, 5, 5, 5 * dt + 20.0, -2.0, 0, 0], [1, 10, 30, 30 * dt + 5000.0, -3.0, 1, 0]]
    arr, meta, _ = vm_stub(win)
    ser, stats = e05.vm_delivery_offline(arr, meta)
    assert len(ser) == 40 and np.isfinite(ser).all()
    assert stats["dropped_late"] == 1 and stats["windows"] == 2
    assert np.allclose(ser[:6], 0.5)  # nothing has landed before frame 6 (completion at 5, plus 20 ms of latency)
    assert ser[6] > 0.7 and ser[6] > ser[7] > ser[8] > 0.5  # lands on frame 6 (hold 0), then the causal smoothing decays back to the sentinel level
    assert np.allclose(ser[20:], 0.5)  # the late window never lands


def test_vm_delivery_offline_without_windows_is_all_sentinel():
    arr, meta, _ = vm_stub([])
    ser, stats = e05.vm_delivery_offline(arr, meta)
    assert len(ser) == 40 and np.allclose(ser, 0.5) and stats["windows"] == 0


def test_vm_delivery_offline_scales_the_late_cap_with_s():
    dt = 1000.0 / 24.0 / 3.0  # s = 3: frames arrive 3x faster; the 2,000 ms cap is in Pi-equivalent ms (= 667 VM ms)
    win = [[0, 5, 5, 5 * dt + 700.0, -2.0, 0, 0]]
    arr, meta, _ = vm_stub(win, s=3.0)
    _, stats = e05.vm_delivery_offline(arr, meta)
    assert stats["dropped_late"] == 1
    arr, meta, _ = vm_stub([[0, 5, 5, 5 * dt + 600.0, -2.0, 0, 0]], s=3.0)
    assert e05.vm_delivery_offline(arr, meta)[1]["dropped_late"] == 0


# ---- configuration / plan / constants -----------------------------------------------------------------------------------
def test_arm_and_contrast_definitions_are_consistent():
    names = set(e05.ARMS) | set(e05.E03_REF_ARMS)
    assert len(e05.ARMS) == 11
    for a, b in e05.CONTRASTS.values():
        assert a in names and b in names
    for a, b in e05.SUB_CONTRASTS.values():
        assert a in e05.ARMS and b in e05.ARMS
    for arm in e05.SUB_ARMS:
        assert arm in e05.ARMS
    for spec in e05.ARMS.values():
        assert spec["ref"] in ("own", "fixed") and spec["lvl"] in (0, 3)


def test_cells_and_cell_groups():
    assert len(e05.CELLS) == 15 and len(set(e05.cells_ids(e05.CELLS))) == 15
    assert e05.CELLS[:12] == list(e03.CELLS)
    for g in e05.CELL_GROUPS.values():
        assert all(c in e05.CELLS for c in g)
    assert e05.family_of("Stock-Baseline") == "stock_default" and e05.family_of("Cache-C1") == "ml0_locked"
    assert e05.N_BOOT == 2000 and e05.VM_PASSES * 2880 >= 10000


def test_plan_and_config_are_locked():
    plan, conf = OUT / "plan.md", OUT / "config.yaml"
    assert plan.exists() and conf.exists()
    cfg = yaml.safe_load(conf.read_text())
    assert len(cfg["cells"]) == 15 and set(cfg["arms"]) >= set(e05.ARMS)
    assert cfg["locked_system"]["primary"] == "Cache-C1"
    text = plan.read_text()
    for token in ("Amendment", "E5-D1", "V9", "Cache-C1+Guard", "105,711"):
        assert token in text
    assert len(cfg["plan_sha256"]) == 64 and int(cfg["plan_sha256"], 16) >= 0  # hash of the plan at lock time; later changes are logged as amendments


# ---- outputs ---------------------------------------------------------------------------------------------------------------
@needs_out
def test_summary_json_structure_and_verdicts():
    s = json.loads((OUT / "summary.json").read_text())
    assert s["experiment"] == "Q2-P1-E05" and s["training"] is False
    for ds in (SHT, UB):
        v = s["verdicts"][ds]
        for k in ("V1_confirmation", "V2_replication", "V3_end_to_end", "V4_retrack", "V5_guard", "V6_short_events", "V7_new_entries"):
            assert k in v
        c = v["V1_confirmation"]["contrast"]
        assert c["lo"] <= c["delta"] <= c["hi"]
    assert "V8_long_horizon" in s["verdicts"] and "V9_vm_sustained" in s["verdicts"]
    assert s["correctness"]["all_pass"] is True
    assert s["hardware"]["physical_pi_connected"] is False


@needs_out
def test_decisions_jsonl_is_valid_and_complete():
    rows = [json.loads(line) for line in (OUT / "decisions.jsonl").read_text().splitlines() if line.strip()]
    kinds = {r["kind"] for r in rows}
    assert {"rule", "verdict", "calibration", "strata", "amendment"} <= kinds
    rules = {r["id"] for r in rows if r["kind"] == "rule"}
    assert {f"E5-D{i}" for i in range(1, 10)} <= rules
    assert all(r["pass"] for r in rows if r["kind"] == "rule" and r.get("required", True))


@needs_out
def test_results_csv_covers_all_arms_and_cells():
    import csv

    with open(OUT / "results.csv") as f:
        rows = list(csv.DictReader(f))
    got = {(r["dataset"], r["arm"], r["cell"]) for r in rows}
    for ds in (SHT, UB):
        for arm in e05.ARMS:
            for cid in e05.cells_ids(e05.CELLS):
                assert (ds, arm, cid) in got, (ds, arm, cid)
    assert all(math.isfinite(float(r["auc_all"])) for r in rows)


@needs_out
def test_report_has_required_sections():
    import re

    text = (OUT / "report.md").read_text()
    low = text.lower()
    for token in ("verdict", "limitations", "bootstrap", "subgroup", "sustained", "e00", "e05", "ubnormal", "cache-c1+guard", "92,640", "not connected", "amendment"):
        assert token in low, token
    assert "TODO" not in text and not re.search(r"\bnan\b", low)
    for fig in re.findall(r"\]\((fig_[^)]+\.png)\)", text):
        assert (OUT / fig).exists(), fig
