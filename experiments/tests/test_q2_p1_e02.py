"""Q2-P1-E02 fixed baseline: schedules, causal propagation, FIFO queue + delivery landing, service-time model, selection rules,
queue/prefix invariants on synthetic tables, plan/config consistency and - when the artifacts exist - the outputs."""

import json
import math
from pathlib import Path

import numpy as np
import pytest
import yaml

from edgelab.analysis import q2_p1_e00 as e00
from edgelab.analysis import q2_p1_e02 as e02
from edgelab.analysis.q2_p1_e02 import Schedule

L_SHT, L_UB = 24, 16


# ---- helpers -----------------------------------------------------------------------------------------------------------
def make_kp(n, seed=0):
    return np.random.default_rng(seed).uniform(0, 100, size=(n, 17, 3))


class FakeScorer:
    """normality = mean x of the window (any pure function of the window works for the table logic)."""

    seg_len = L_SHT

    def score(self, windows):
        return -np.asarray(windows)[:, :, :, 0].mean(axis=(1, 2))


# ---- schedules ---------------------------------------------------------------------------------------------------------
def test_stride_and_burst_observed_masks():
    n = 100
    assert e02.Schedule("stride", 1).observed(n, 24).all()
    m = e02.Schedule("stride", 3, "zoh").observed(n, 24)
    assert m.sum() == 34 and m[0] and m[3] and not m[1]
    assert e02.Schedule("burst", 12).observed(n, 24).all()  # W <= L: the union of the bursts is every frame
    b = e02.Schedule("burst", 48).observed(n, 24)
    assert b[:24].all() and not b[24:48].any() and b[48:72].all() and not b[72:96].any() and b[96:].all()
    assert math.isclose(e02.Schedule("burst", 48).duty(24), 0.5) and math.isclose(e02.Schedule("burst", 12).duty(24), 1.0)
    assert math.isclose(e02.Schedule("stride", 4, "cv").duty(24), 0.25)


def test_schedule_grid_matches_plan():
    sht = [s.id for s in e02.schedule_grid("ShanghaiTech")]
    assert sht[0] == "stride1" and "stride2-none" in sht and "stride4-cv" in sht
    assert [int(s[5:]) for s in sht if s.startswith("burst")] == [1, 2, 4, 6, 8, 12, 24, 36, 48, 72, 96]
    ub = [s.id for s in e02.schedule_grid("UBnormal")]
    assert [int(s[5:]) for s in ub if s.startswith("burst")] == [1, 2, 4, 6, 8, 12, 16, 24, 32, 48, 64]
    assert len(sht) == len(set(sht))
    # W = K * L has the detector duty 1/K of stride K
    for k in (2, 3, 4):
        assert math.isclose(Schedule("burst", k * 24).duty(24), Schedule("stride", k, "zoh").duty(24))


def test_plan_compliance_flags():
    assert not Schedule("stride", 2, "zoh").plan_compliant and not Schedule("stride", 3, "cv").plan_compliant
    assert Schedule("stride", 2, "none").plan_compliant and Schedule("stride", 1).plan_compliant
    assert all(Schedule("burst", w).plan_compliant for w in (1, 6, 24, 48, 96))
    assert Schedule("burst", 1).alias_of == "stride1"
    assert Schedule("stride", 3, "zoh").hold == 2 and Schedule("burst", 6).hold == 5 and Schedule("stride", 1).hold == 0


def test_keep_windows_only_on_the_scoring_grid():
    last = np.arange(300)
    assert e02.Schedule("stride", 1).keep_windows(last, 24).all()
    assert (last[e02.Schedule("stride", 3, "zoh").keep_windows(last, 24)] % 3 == 0).all()
    k6 = last[e02.Schedule("burst", 6).keep_windows(last, 24)]
    assert k6[0] == 23 and (np.diff(k6) == 6).all()  # windows [kW, kW+L-1]
    k48 = last[e02.Schedule("burst", 48).keep_windows(last, 24)]
    assert list(k48) == [23, 71, 119, 167, 215, 263, 311][:len(k48)] and (k48 % 48 == 23).all()


def test_tracker_input_mask_equals_requested_frames_D7():
    assert e02.check_d7()["pass"]


def test_schedule_never_reads_unrequested_frames(monkeypatch):
    """The replay slices the detector cache with the schedule's mask: a spy list shows no other index is touched."""
    class Spy(list):
        touched: set

        def __getitem__(self, i):
            self.touched.add(i)
            return super().__getitem__(i)

    seg, n = 24, 130
    seen = {}

    def fake_replay(frames, frame_ids, cell):
        seen["ids"] = np.asarray(frame_ids).copy()
        return {}

    monkeypatch.setattr(e02.e01, "replay_cell", fake_replay)
    for ts in (("stride", 1), ("stride", 3), ("stride", 4), ("burst", 36), ("burst", 96)):
        frames = Spy([{"i": i} for i in range(n)])
        frames.touched = set()
        e02._G.update(frames=[frames], ids=[np.arange(n)], cell=e02.TRACKERS["locked"], L=seg, tracksets=[ts])
        try:
            e02._replay_clip(0)
        finally:
            for key in ("frames", "ids", "cell", "L", "tracksets"):
                e02._G.pop(key, None)
        requested = set(np.nonzero(e02.trackset_mask(ts, n, seg))[0].tolist())
        assert frames.touched == requested, ts


def test_burst_tracker_is_recreated_per_burst(monkeypatch):
    calls = []

    def fake_replay(frames, frame_ids, cell):
        calls.append(np.asarray(frame_ids).copy())
        return {1: (np.asarray(frame_ids), make_kp(len(frame_ids)))}

    monkeypatch.setattr(e02.e01, "replay_cell", fake_replay)
    n, seg = 120, 24
    e02._G.update(frames=[[{}] * n], ids=[np.arange(n)], cell=e02.TRACKERS["locked"], L=seg, tracksets=[("burst", 48)])
    try:
        out = e02._replay_clip(0)[("burst", 48)]
    finally:
        for key in ("frames", "ids", "cell", "L", "tracksets"):
            e02._G.pop(key, None)
    assert len(calls) == 3 and all(len(c) <= seg for c in calls) and [c[0] for c in calls] == [0, 48, 96]
    assert len(out) == 3  # one track per burst: nothing is carried across a gap


# ---- causal propagation ---------------------------------------------------------------------------------------------------
def test_propagate_none_and_k1_are_identity():
    f, kp = np.array([0, 3, 6]), make_kp(3)
    for mode, k in (("none", 3), ("zoh", 1), ("cv", 1)):
        pf, pk, g = e02.propagate_track(f, kp, k, mode)
        assert np.array_equal(pf, f) and np.array_equal(pk, kp) and g.all()


def test_zoh_repeats_the_last_observation_and_never_past_the_next():
    f, kp = np.array([0, 3, 9]), make_kp(3)
    pf, pk, g = e02.propagate_track(f, kp, 3, "zoh", n_frames=100)
    assert list(pf) == [0, 1, 2, 3, 4, 5, 9, 10, 11]  # 6-8 is a track gap longer than K - 1: nothing is invented
    assert list(g) == [True, False, False, True, False, False, True, False, False]
    assert np.array_equal(pk[1], kp[0]) and np.array_equal(pk[2], kp[0]) and np.array_equal(pk[4], kp[1])
    assert np.array_equal(pk[7], kp[2])


def test_cv_extrapolates_with_the_previous_velocity_only_causally():
    kp = np.zeros((3, 17, 3))
    kp[0, :, :2], kp[1, :, :2], kp[2, :, :2] = 10.0, 13.0, 16.0  # +3 px per 3 frames -> +1 px per frame
    kp[:, :, 2] = 0.8
    pf, pk, g = e02.propagate_track(np.array([0, 3, 6]), kp, 3, "cv", n_frames=100)
    assert list(pf) == [0, 1, 2, 3, 4, 5, 6, 7, 8]
    assert np.allclose(pk[0, :, :2], 10) and np.allclose(pk[1, :, :2], 10)  # first observation has no velocity yet: hold
    assert np.allclose(pk[4, :, :2], 14.0) and np.allclose(pk[5, :, :2], 15.0)
    assert np.allclose(pk[:, :, 2], 0.8)  # confidence is held
    # causality: a later observation never changes an earlier propagated pose
    pf2, pk2, _ = e02.propagate_track(np.array([0, 3]), kp[:2], 3, "cv", n_frames=100)
    assert np.allclose(pk2[:6], pk[:6])


def test_propagation_respects_the_clip_end():
    f, kp = np.array([0, 3]), make_kp(2)
    pf, _, _ = e02.propagate_track(f, kp, 3, "zoh", n_frames=5)
    assert list(pf) == [0, 1, 2, 3, 4]


def test_no_fill_stride_yields_zero_windows_D5():
    """K >= 2 without filling: a window of L consecutive poses needs K*L frames; STG-NF's missing <= 2 rule cannot accept it."""
    for k in (2, 3, 4):
        frames = np.arange(0, 400, k)
        assert len(e00.complete_window_starts(frames, 24)) == 0
        assert len(e00.complete_window_starts(frames, 16)) == 0
        tbl = e02.scheduled_table([{1: (frames, make_kp(len(frames)))}], [400], 24, FakeScorer(), Schedule("stride", k, "none"))
        assert len(tbl["normality"]) == 0
    assert len(e00.complete_window_starts(np.arange(0, 100), 24)) == 77  # K=1 control


def test_scheduled_table_zoh_genuine_fraction_and_cadence():
    frames = np.arange(0, 120, 3)
    s = Schedule("stride", 3, "zoh")
    tbl = e02.scheduled_table([{7: (frames, make_kp(len(frames)))}], [120], 24, FakeScorer(), s)
    assert len(tbl["normality"]) > 0 and (tbl["last_frame"] % 3 == 0).all()
    assert np.allclose(tbl["genuine_frac"], 8 / 24)  # one real observation in three
    assert tbl["last_frame"].min() == 24  # the first window of consecutive filled poses ends at frame 24 (poses 3..26 are filled 3..26)


def test_burst_windows_are_fully_genuine_D6():
    frames = np.arange(0, 24)
    s = Schedule("burst", 48)
    tbl = e02.scheduled_table([{1: (frames, make_kp(24))}], [200], 24, FakeScorer(), s)
    assert len(tbl["normality"]) == 1 and tbl["genuine_frac"][0] == 1.0 and tbl["last_frame"][0] == 23


def test_scheduled_table_scores_only_the_grid():
    frames = np.arange(0, 100)
    s = Schedule("burst", 6)
    tbl = e02.scheduled_table([{1: (frames, make_kp(100))}], [100], 24, FakeScorer(), s)
    assert list(tbl["last_frame"]) == list(range(23, 100, 6))[:len(tbl["last_frame"])] and (tbl["last_frame"] % 6 == 5).all()


# ---- queue and delivery ---------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("seed", range(5))
def test_vectorised_fifo_equals_reference_D3(seed):
    rng = np.random.default_rng(seed)
    arrival = np.arange(500) * 41.7
    service = rng.gamma(2.0, 20.0, 500) * rng.choice([0.2, 1.0, 3.0])
    f = e02.fifo_finish(arrival, service)
    assert np.allclose(f, e02.fifo_finish_reference(arrival, service))
    assert ((f - service) >= arrival - 1e-9).all() and (np.diff(f) >= 0).all()


def test_fifo_segments_reset_the_backlog():
    arrival = np.arange(20) * 10.0
    service = np.full(20, 25.0)
    cont = e02.fifo_finish(arrival, service)
    seg = e02.fifo_finish_segments(arrival, service, np.array([0, 10, 20]))
    assert np.allclose(seg[:10], cont[:10]) and seg[10] == 100.0 + 25.0 and cont[10] > seg[10]


def test_delivery_frame_is_ceil_of_finish():
    dt = 1000 / 24
    assert list(e02.delivery_frame(np.array([0.0, 1.0, dt, dt + 0.5, 3 * dt]), dt)) == [0, 1, 1, 2, 3]


def test_land_windows_never_before_completion_and_late_work_is_discarded():
    n_frames = np.array([50, 50])
    offsets = np.array([0, 50, 100])
    dt = 40.0
    arrival = np.arange(100) * dt
    service = np.full(100, 10.0)
    service[10] = 3000.0  # one catastrophic job: the 3.0 s inference
    finish = e02.fifo_finish(arrival, service)
    clip, last = np.array([0, 0, 0, 1]), np.array([5, 10, 12, 49])
    landing, ok, latency, reason = e02.land_windows(clip, last, offsets, finish, arrival, dt, n_frames)
    assert (landing >= last).all()
    assert reason[1] == 1 and not ok[1]  # latency 3.0 s > 2.0 s cap -> discarded
    assert reason[2] in (0, 1)
    g = offsets[clip] + last
    assert (landing + offsets[clip] >= e02.delivery_frame(finish[g], dt)).all()


def test_queue_metrics_flags_overload():
    n, dt = 600, 41.7
    arrival = np.arange(n) * dt
    ok_service = np.full(n, 30.0)
    bad_service = np.full(n, 60.0)
    f1, f2 = e02.fifo_finish(arrival, ok_service), e02.fifo_finish(arrival, bad_service)
    assert e02.queue_metrics(arrival, ok_service, f1, dt)["sustainable"]
    m = e02.queue_metrics(arrival, bad_service, f2, dt)
    assert not m["sustainable"] and m["rho"] > 1 and m["max_wait_ms"] > e02.CAP_MS and m["wait_slope_ms_per_s"] > 0


# ---- delivery simulation on synthetic tables (D2, D4) ----------------------------------------------------------------------
def synthetic_stream(seed=0, n_clips=6):
    rng = np.random.default_rng(seed)
    n_frames = rng.integers(60, 120, n_clips)
    cols = {"clip_idx": [], "last_frame": [], "normality": []}
    for c, n in enumerate(n_frames):
        for _ in range(3):  # three tracks per clip
            start = int(rng.integers(0, n // 3))
            last = np.arange(start + 23, n)
            cols["clip_idx"].append(np.full(len(last), c))
            cols["last_frame"].append(last)
            cols["normality"].append(-rng.uniform(0.1, 1.5, len(last)))
    return n_frames, {k: np.concatenate(v) for k, v in cols.items()}


def test_zero_service_equals_ideal_completion_landing_D2():
    n_frames, t = synthetic_stream(1)
    series, _ = e02.simulate_delivery(t["clip_idx"], t["last_frame"], t["normality"], n_frames, np.zeros(int(n_frames.sum())), 1000 / 24, -0.5, 0)
    raw = e00.per_frame_min(t["clip_idx"], t["last_frame"], t["normality"], list(n_frames))
    ref, _ = e00.deliver(raw, -0.5, 0)
    assert all(np.array_equal(a, b) for a, b in zip(series, ref))


def test_delivery_with_latency_is_later_never_earlier():
    n_frames, t = synthetic_stream(2)
    svc = np.full(int(n_frames.sum()), 100.0)  # 100 ms per job at 24 fps: rho = 2.4 -> overload; use 30 ms for a stable queue
    svc[:] = 30.0
    _, info = e02.simulate_delivery(t["clip_idx"], t["last_frame"], t["normality"], n_frames, svc, 1000 / 24, -0.5, 0)
    assert (info["landing"] >= t["last_frame"]).all()
    assert (info["landing"][info["ok"]] - t["last_frame"][info["ok"]] <= 1).all()  # 30 ms < 41.7 ms: at most one frame of lateness


def test_prefix_invariance_D4():
    n_frames, t = synthetic_stream(3)
    rng = np.random.default_rng(0)
    service = rng.gamma(2.0, 15.0, int(n_frames.sum()))
    full, _ = e02.simulate_delivery(t["clip_idx"], t["last_frame"], t["normality"], n_frames, service, 1000 / 24, -0.5, 4)
    for m, cut in ((3, 20), (4, 50), (5, 7)):
        nf = n_frames[:m].copy()
        nf[-1] = min(cut, nf[-1])
        keep = (t["clip_idx"] < m) & ~((t["clip_idx"] == m - 1) & (t["last_frame"] >= nf[-1]))
        sub, _ = e02.simulate_delivery(t["clip_idx"][keep], t["last_frame"][keep], t["normality"][keep], nf, service[:int(nf.sum())], 1000 / 24, -0.5, 4)
        for i in range(m):
            assert np.array_equal(sub[i][:nf[i]], full[i][:nf[i]])


def test_queue_invariant_checker_passes_on_a_real_queue():
    n_frames, t = synthetic_stream(4)
    service = np.random.default_rng(1).gamma(2.0, 12.0, int(n_frames.sum()))
    _, info = e02.simulate_delivery(t["clip_idx"], t["last_frame"], t["normality"], n_frames, service, 1000 / 24, -0.5, 0)
    ck = e02.check_queue_invariants(info["arrival"], service, info["finish"], info["landing"], t["last_frame"], 1000 / 24, t["clip_idx"], info["offsets"])
    assert all(ck.values()), ck


def test_window_counts_per_frame():
    table = {"clip_idx": np.array([0, 0, 1]), "last_frame": np.array([5, 5, 2]), "normality": np.zeros(3)}
    c = e02.window_counts(table, np.array([0, 10, 20]), 20)
    assert c[5] == 2 and c[12] == 1 and c.sum() == 3


# ---- service-time model --------------------------------------------------------------------------------------------------
def toy_pools(scale=1.0):
    rng = np.random.default_rng(0)
    pools = {}
    for t in (1, 2, 3, 4):
        base = {"pose": 40.0 / t ** 0.8, "grab": 2.0, "retrieve": 0.5, "jpeg": 1.3}
        for stage, ms in base.items():
            w = rng.gamma(20, ms * scale / 20, 300)
            pools[(stage, t)] = (w, w * (1 + 0.1 * t))
        for n in (1, 2, 4):
            w = rng.gamma(20, (1 + n) / 20, 100)
            pools[("score", t, n)] = (w, w * 1.2)
    for n in range(0, 5):
        w = rng.gamma(20, (0.2 + 0.05 * n) / 20, 200)
        pools[("track", n)] = (w, w)
    return pools


def test_interp_cores():
    means = {1: 40.0, 2: 24.0, 3: 18.0, 4: 15.0}
    assert e02.interp_cores(means, 1) == 40.0 and e02.interp_cores(means, 4) == 15.0 and e02.interp_cores(means, 8) == 15.0
    assert math.isclose(e02.interp_cores(means, 2.5), 21.0)
    assert math.isclose(e02.interp_cores(means, 0.5), 80.0)  # below one core: CPU quota -> 1 / q


def test_service_model_scales_with_cores_and_s():
    m = e02.ServiceModel(toy_pools())
    obs = np.ones(200, bool)
    n_in = np.full(200, 2)
    n_win = np.zeros(200, dtype=int)
    n_win[::10] = 2
    u = np.random.default_rng(1).random((200, 5))
    w4, c4, parts = m.sample(4.0, obs, n_in, n_win, u)
    w1, _, _ = m.sample(1.0, obs, n_in, n_win, u)
    wq, _, _ = m.sample(0.5, obs, n_in, n_win, u)
    assert w1.mean() > w4.mean() and wq.mean() > 1.7 * w1.mean()
    assert (w4 > 0).all() and (c4 > 0).all() and set(parts) == {"decode_ms", "pose_ms", "track_ms", "score_ms"}
    assert np.allclose(w4, m.sample(4.0, obs, n_in, n_win, u)[0])  # common random numbers: deterministic
    # unobserved frames cost the grab only; JPEG input pays nothing for them
    obs2 = obs.copy()
    obs2[::2] = False
    wh, _, _ = m.sample(4.0, obs2, n_in, n_win, u, "h264")
    wj, _, _ = m.sample(4.0, obs2, n_in, n_win, u, "jpeg")
    assert wh[~obs2][wh[~obs2] > 0].max() < 20 and (wj[~obs2 & (n_win == 0)] == 0).all()


def test_nearest_score_pool_for_unmeasured_window_counts():
    m = e02.ServiceModel(toy_pools())
    assert m._nearest(m.score_n, 3) == 2 and m._nearest(m.score_n, 100) == 4 and m._nearest(m.track_n, 11) == 4


def test_timing_stats():
    s = e02.timing_stats(np.arange(1, 101, dtype=float))
    assert s["n"] == 100 and math.isclose(s["mean"], 50.5) and math.isclose(s["p50"], 50.5) and s["p99"] > 98 and s["max"] == 100
    assert math.isnan(e02.timing_stats([])["mean"])


def test_scenario_list_is_the_plan_grid():
    ids = [s["id"] for s in e02.scenario_list()]
    assert ids == ["T4_s1.0_b1.00", "T4_s1.5_b1.00", "T4_s2.0_b1.00", "T4_s2.5_b1.00", "T4_s3.0_b1.00", "T4_s2.0_b0.75", "T4_s2.0_b0.50",
                   "T4_s2.0_b0.33", "T4_s2.0_b0.25"]
    assert all(h in ids for h in e02.HEADLINE)
    assert math.isclose(e02.scenario_list()[-1]["q"], 1.0)


# ---- selection rules -----------------------------------------------------------------------------------------------------
def test_select_fixed_reference_prefers_accuracy_then_cpu_within_the_tie_band():
    rows = [{"schedule": "a", "sustainable": True, "dev_auc": 70.0, "cpu_ms_per_s": 900, "plan_compliant": True},
            {"schedule": "b", "sustainable": True, "dev_auc": 70.05, "cpu_ms_per_s": 600, "plan_compliant": False},
            {"schedule": "c", "sustainable": True, "dev_auc": 71.0, "cpu_ms_per_s": 1500, "plan_compliant": True},
            {"schedule": "d", "sustainable": False, "dev_auc": 80.0, "cpu_ms_per_s": 100, "plan_compliant": True}]
    assert e02.select_fixed_reference(rows)["schedule"] == "c"  # most accurate sustainable
    rows[2]["dev_auc"] = 70.1
    assert e02.select_fixed_reference(rows)["schedule"] == "b"  # within 0.10 of the best: lowest CPU
    assert e02.select_fixed_reference(rows, require_compliant=True)["schedule"] == "a"  # b is propagated; a and c tie within 0.10
    rows[2]["dev_auc"] = 70.5
    assert e02.select_fixed_reference(rows, require_compliant=True)["schedule"] == "c"
    assert e02.select_fixed_reference([{"schedule": "x", "sustainable": False, "dev_auc": 1.0, "cpu_ms_per_s": 1.0, "plan_compliant": True}]) is None


def test_pareto_front_and_h1_decision():
    pts = [(1.0, 60.0, "a"), (2.0, 70.0, "b"), (2.5, 65.0, "c"), (3.0, 75.0, "d")]
    assert set(e02.pareto_front(pts)) == {"a", "b", "d"}
    assert e02.h1_decision({"ci95": [0.5, 3.0]}) == "burst preferred"
    assert e02.h1_decision({"ci95": [-3.0, -0.5]}) == "stride preferred"
    assert e02.h1_decision({"ci95": [-1.0, 2.0]}) == "tie"


def test_job_list_covers_the_plan_grid():
    sids = [s.id for s in e02.schedule_grid("ShanghaiTech")]
    jobs = e02.jobs_for("ShanghaiTech", True, "locked", sids)
    keys = {j["key"] for j in jobs}
    assert len(keys) == len(jobs)
    assert e02.job_key("stride1", None) in keys and e02.job_key("burst48", "T4_s2.0_b1.00", decode="jpeg") in keys
    assert e02.job_key("burst48", "T4_s2.0_b0.25", stream="clip_reset") in keys
    default_jobs = e02.jobs_for("ShanghaiTech", True, "default", sids)
    assert not any(j["stream"] == "clip_reset" for j in default_jobs)
    assert e02.schedule_ids("ShanghaiTech", False, "locked") == ["stride1", "stride2-zoh", "stride3-zoh", "stride4-zoh"] + [
        f"burst{w}" for w in e02.BURST_W["ShanghaiTech"]]


# ---- plan / config consistency and artifacts --------------------------------------------------------------------------------
def test_machine_tag_is_the_disclosure_string():
    assert e02.MACHINE_TAG == "edge-os-arm64-vm (comparative proxy; physical Pi 5 pending D8/D10)"
    assert tuple(e02.S_GRID) == (1.0, 1.5, 2.0, 2.5, 3.0)


def test_locked_trackers_are_e01s():
    assert e02.TRACKERS["locked"].label == "c0.20|low-0.05|b30|m0.85" and e02.TRACKERS["default"].label == "c0.25|low-off|b30|m0.80"
    assert e02.PRIMARY == {"ShanghaiTech": "ML0_ncnn", "UBnormal": "ML0_mps"}


CONFIG = e02.EXP_DIR / "config.yaml"


@pytest.mark.skipif(not CONFIG.exists(), reason="E02 not locked yet")
def test_config_is_locked_and_hashes_inputs():
    cfg = yaml.safe_load(CONFIG.read_text())
    assert cfg["training"] is False and cfg["hardware_disclosure"]["machine_tag"] == e02.MACHINE_TAG
    assert cfg["locked_from_E01"]["frontend"] == "ML0_ncnn" and cfg["hardware_disclosure"]["pi_multiplier_s"] == [1.0, 1.5, 2.0, 2.5, 3.0]
    assert cfg["depends_on"]["E01"]["config_sha256"] == e00.sha256_file(e02.E01_DIR / "config.yaml")
    assert cfg["depends_on"]["E02_plan_sha256"] == e00.sha256_file(e02.EXP_DIR / "plan.md")
    assert set(cfg["rules"]) >= {"P1", "P5", "D1", "D7", "H1", "B1"}


SUMMARY = e02.EXP_DIR / "summary.json"


@pytest.mark.skipif(not SUMMARY.exists(), reason="E02 outputs not generated yet")
def test_outputs_are_complete_and_consistent():
    S = json.loads(SUMMARY.read_text())
    for name in ("plan.md", "config.yaml", "results.csv", "decisions.jsonl", "summary.json", "report.md"):
        assert (e02.EXP_DIR / name).exists(), name
    assert S["machine"] == e02.MACHINE_TAG
    failed = {k: v["status"] for k, v in S["rules"].items() if v["status"] == "FAIL"}
    assert not failed, failed
    lines = (e02.EXP_DIR / "results.csv").read_text().splitlines()
    assert lines[0].split(",") == e02.CSV_COLUMNS and len(lines) - 1 == S["outputs"]["results_csv_rows"]
    assert all(json.loads(x) for x in (e02.EXP_DIR / "decisions.jsonl").read_text().splitlines()[:50])
    report = (e02.EXP_DIR / "report.md").read_text()
    assert e02.MACHINE_TAG in report and "physical Pi 5" in report
