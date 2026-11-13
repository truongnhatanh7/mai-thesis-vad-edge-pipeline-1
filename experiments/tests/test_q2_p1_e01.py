"""Q2-P1-E01 quality: pure rules (track statistics, tracker grid, selection rules, recall matcher, miss decomposition, paired recall bootstrap),
the tracker replay on toy detections (needs ultralytics), the locked config / plan consistency, and - when the artifacts exist - the outputs."""

import json
import math
from pathlib import Path

import numpy as np
import pytest
import yaml

from edgelab.analysis import q2_p1_e00 as e00
from edgelab.analysis import q2_p1_e01 as e01

ULTRALYTICS = pytest.importorskip("ultralytics", reason="tracker replay tests need the `pose` extra")


def _frames(start, n, step=1):
    return np.arange(start, start + n * step, step, dtype=np.int64)


def _track(frames):
    return np.asarray(frames, dtype=np.int64), np.zeros((len(frames), 17, 3))


# ---- track statistics (part C) ---------------------------------------------------------------------------------------
def test_every_pose_of_a_track_falls_in_exactly_one_class():
    seg = 24
    tracks = {
        1: _track(_frames(0, 10)),  # stranded short
        2: _track(_frames(0, 24)),  # exact seg_len: one window
        3: _track(_frames(0, 60)),  # clean long track
        4: _track(np.r_[_frames(0, 12), _frames(40, 12)]),  # 24 poses but a 28-frame hole -> stranded gappy
        5: _track(np.r_[_frames(0, 30), _frames(32, 30)]),  # 2 missing frames (tolerated), 60 poses
        6: _track(np.r_[_frames(0, 30), _frames(60, 30)]),  # 30 missing frames: windows only inside each half
    }
    c, lengths, gaps = e01.clip_track_stats(tracks, seg)
    assert c["n_tracks"] == 6 and c["n_poses"] == int(lengths.sum()) == 10 + 24 + 60 + 24 + 60 + 60
    assert c["poses_scorable"] + c["poses_unscorable_stranded"] + c["poses_unscorable_initial"] + c["poses_unscorable_post_gap"] == c["n_poses"]
    assert c["n_stranded_short"] == 1 and c["n_stranded_gappy"] == 1 and c["n_exact_seg_len"] == 2  # tracks 2 and 4 both have exactly 24 poses
    assert c["poses_unscorable_stranded"] == 10 + 24
    assert c["n_windows"] == c["poses_scorable"]  # one window per scorable pose
    assert c["gaps_1"] == 0 and c["gaps_2"] == 1 and c["gaps_ge3"] == 2  # track 5: 2 missing frames; tracks 4 and 6: long holes
    assert sorted(gaps.tolist()) == [2, 28, 30] and c["gap_frames"] == 60
    # a fully contiguous track of n poses has n - seg + 1 windows and seg - 1 warm-up poses
    c3, *_ = e01.clip_track_stats({3: tracks[3]}, seg)
    assert c3["poses_scorable"] == 37 and c3["poses_unscorable_initial"] == 23 and c3["poses_unscorable_post_gap"] == 0


def test_gap_tolerance_matches_stgnf_continuity_rule():
    seg = 24
    two_missing = _track(np.r_[_frames(0, 12), _frames(14, 20)])  # 2 missing frames inside: still complete
    three_missing = _track(np.r_[_frames(0, 12), _frames(15, 12)])  # 24 poses, 3 missing in the only window -> not complete
    c2, *_ = e01.clip_track_stats({1: two_missing}, seg)
    c3, *_ = e01.clip_track_stats({1: three_missing}, seg)
    assert c2["gaps_2"] == 1 and c2["n_windows"] > 0
    assert c3["gaps_ge3"] == 1 and c3["n_windows"] == 0 and c3["n_stranded_gappy"] == 1


def test_aggregate_track_rows_pools_additively_and_derives_ratios():
    seg = 24
    a, *_ = e01.clip_track_stats({1: _track(_frames(0, 60)), 2: _track(_frames(0, 5))}, seg)
    b, *_ = e01.clip_track_stats({1: _track(_frames(0, 30))}, seg)
    for row, n_frames, fresh in ((a, 100, 40), (b, 50, 10)):
        row.update(n_frames=n_frames, anomalous_frames=n_frames // 2, n_normal_frames=n_frames - n_frames // 2, scorable_frames=fresh,
                   scorable_normal=fresh // 2, scorable_anomalous=fresh - fresh // 2)
    t = e01.aggregate_track_rows([a, b], seg)
    assert t["n_tracks"] == 3 and t["n_poses"] == 95 and t["n_frames"] == 150 and t["scorable_frames"] == 50
    assert math.isclose(t["stranded_short_share_of_tracks"], 1 / 3)
    assert math.isclose(t["coverage_all"], 50 / 150) and math.isclose(t["windows_per_frame"], t["n_windows"] / 150)
    assert math.isclose(t["scorable_pose_share"] + t["stranded_poses_share"] + t["initial_warmup_pose_share"] + t["post_gap_pose_share"], 1.0)


def test_histograms_cover_every_value_once():
    lengths = np.array([1, 7, 8, 15, 16, 23, 24, 47, 48, 95, 96, 400])
    assert sum(e01.histogram(lengths, e01.LEN_EDGES)) == len(lengths)
    assert e01.histogram(lengths, e01.LEN_EDGES)[0] == 2  # 1, 7
    assert sum(e01.histogram(np.array([1, 2, 3, 5, 6, 11, 12, 23, 24, 100]), e01.GAP_EDGES)) == 10
    assert e01.gap_summary(np.zeros(0, np.int64))["n"] == 0 and e01.length_summary(np.zeros(0, np.int64), 24)["n"] == 0


# ---- tracker grid ----------------------------------------------------------------------------------------------------
def test_grid_is_the_pre_registered_36_cells_with_unique_labels_and_a_default():
    cells = e01.grid_cells()
    assert len(cells) == 36 and len({c.label for c in cells}) == 36
    assert e01.DEFAULT_CELL in cells and e01.DEFAULT_CELL.n_changed() == 0
    assert {c.c for c in cells} == {0.20, 0.25, 0.30} and {c.low for c in cells} == {None, 0.10, 0.05}
    assert {c.buffer for c in cells} == {30, 45} and {c.match for c in cells} == {0.80, 0.85}
    # the detector gate is c when two-stage is disabled and the low threshold otherwise; the gate never exceeds c
    for c in cells:
        assert c.gate == (c.c if c.low is None else c.low) and c.gate <= c.c


def test_default_cell_is_exactly_the_replay_py_tracker():
    from edgelab.pose.replay import build_tracker

    mine, ref = e01.build_tracker_cell(e01.DEFAULT_CELL), build_tracker(0.25)
    for key in ("track_high_thresh", "new_track_thresh", "track_low_thresh", "track_buffer", "match_thresh", "gmc_method", "with_reid", "proximity_thresh",
                "appearance_thresh", "fuse_score"):
        assert getattr(mine.args, key) == getattr(ref.args, key), key


def _det_frames(conf_per_frame, x0=100.0, dx=2.0, h=120.0, w=50.0):
    frames = []
    for t, conf in enumerate(conf_per_frame):
        if conf is None:
            frames.append({"boxes": np.zeros((0, 4), np.float32), "conf": np.zeros(0, np.float32), "cls": np.zeros(0, np.float32),
                           "keypoints": np.zeros((0, 17, 3), np.float32)})
            continue
        x = x0 + dx * t
        frames.append({"boxes": np.array([[x, 50, x + w, 50 + h]], np.float32), "conf": np.array([conf], np.float32), "cls": np.zeros(1, np.float32),
                       "keypoints": np.full((1, 17, 3), t, np.float32)})
    return frames, np.arange(len(conf_per_frame))


def test_replay_default_cell_matches_replay_video_and_drops_low_boxes():
    from edgelab.pose.replay import replay_video

    frames, ids = _det_frames([0.9] * 10 + [0.15] * 3 + [0.9] * 10)
    mine = e01.replay_cell(frames, ids, e01.DEFAULT_CELL)
    poses, _ = replay_video(frames, ids, 0.25)
    assert e01.track_partition(mine) == e01.track_partition(e01.tracks_from_poses(poses))
    (only,) = [f for f, _ in mine.values() if len(f) > 5][:1] or [None]
    assert only is not None
    assert not {10, 11, 12} & {int(f) for f, _ in mine.values() for f in f}  # low boxes never reached the tracker


def test_two_stage_low_boxes_continue_a_live_track_but_cannot_start_one():
    frames, ids = _det_frames([0.9] * 10 + [0.15] * 3 + [0.9] * 10)
    two = e01.replay_cell(frames, ids, e01.TrackerCell(low=0.10))
    covered = {int(f) for fr, _ in two.values() for f in fr}
    assert {10, 11, 12} <= covered  # the 0.15 boxes were used by the second association
    assert len({pid for pid, (fr, _) in two.items() if len(fr) >= 3}) == 1  # one identity bridges the weak frames
    only_low, ids2 = _det_frames([0.15] * 30)
    assert e01.replay_cell(only_low, ids2, e01.TrackerCell(low=0.10)) == {}  # new tracks still need conf >= c
    assert e01.replay_cell(only_low, ids2, e01.TrackerCell(c=0.20, low=0.10)) == {}


def test_lower_gate_c_can_start_a_track_from_the_same_boxes():
    frames, ids = _det_frames([0.22] * 30)
    assert e01.replay_cell(frames, ids, e01.DEFAULT_CELL) == {}
    tracks = e01.replay_cell(frames, ids, e01.TrackerCell(c=0.20))
    assert sum(len(f) for f, _ in tracks.values()) >= 25


# ---- selection rules -------------------------------------------------------------------------------------------------
def _cells():
    return {c.label: c for c in e01.grid_cells()}


def test_tracker_selection_prefers_the_default_inside_the_tie_band_and_the_closest_cell_otherwise():
    cells = _cells()
    base = {k: 70.0 for k in cells}
    d = e01.DEFAULT_CELL.label
    other = e01.TrackerCell(c=0.30).label
    far = e01.TrackerCell(c=0.20, low=0.05, buffer=45, match=0.85).label
    dev = dict(base, **{other: 70.05, far: 70.08})  # all within the 0.10 band of the best
    sel = e01.select_tracker_cell(dev, cells)
    assert sel["winner"] == d and sel["argmax"] == far and set(sel["tied"]) >= {d, other, far}
    dev2 = dict(base, **{other: 70.30, far: 70.42})  # default now 0.42 behind the best, `other` is 0.12 behind: not tied
    sel2 = e01.select_tracker_cell(dev2, cells)
    assert sel2["winner"] == far and sel2["tied"] == [far]
    dev3 = dict(base, **{other: 70.40, far: 70.42})  # `other` (one factor changed) beats `far` (four) inside the band
    assert e01.select_tracker_cell(dev3, cells)["winner"] == other


def test_tracker_adoption_needs_size_ci_and_no_veto():
    d = e01.DEFAULT_CELL.label
    w = e01.TrackerCell(c=0.30).label
    good = {"delta": 0.8, "ci95": [0.2, 1.4]}
    assert e01.adopt_tracker(w, d, good, {}) == {"adopted": True, "locked": w, "reason": "all T2 conditions hold"}
    assert not e01.adopt_tracker(w, d, {"delta": 0.3, "ci95": [0.1, 0.5]}, {})["adopted"]  # below +0.5 pt
    r = e01.adopt_tracker(w, d, {"delta": 0.9, "ci95": [-0.1, 1.9]}, {})
    assert not r["adopted"] and r["locked"] == d and "does not exclude 0" in r["reason"]
    r = e01.adopt_tracker(w, d, good, {"UBnormal/stock": {"delta": -1.0, "ci95": [-2.0, -0.1]}})
    assert not r["adopted"] and "harmful" in r["reason"]
    assert e01.adopt_tracker(w, d, good, {"UBnormal/stock": {"delta": -1.0, "ci95": [-2.0, 0.4]}})["adopted"]  # CI covers 0: no veto
    assert e01.adopt_tracker(d, d, good, {})["locked"] == d


def test_tracker_verification_reverts_only_on_a_clearly_negative_confirmation():
    d = e01.DEFAULT_CELL.label
    w = e01.TrackerCell(c=0.30).label
    assert e01.verify_tracker(d, d, {"delta": 0.0, "ci95": [0.0, 0.0]})["final"] == d
    ok = e01.verify_tracker(w, d, {"delta": -0.2, "ci95": [-0.9, 0.6]})
    assert not ok["failed"] and ok["final"] == w
    bad = e01.verify_tracker(w, d, {"delta": -1.0, "ci95": [-1.8, -0.2]})
    assert bad["failed"] and bad["final"] == d


def test_frontend_lock_walks_up_the_cost_ladder_only_with_margin_ci_and_no_ub_harm():
    cand = [{"name": "ML0_ncnn", "cost": 1.0}, {"name": "stock_ncnn800", "cost": 1.5625}, {"name": "MLP_ncnn", "cost": 1.92}]
    dev = {"ML0_ncnn": 72.4, "stock_ncnn800": 67.6, "MLP_ncnn": 74.4}
    pair = {("stock_ncnn800", "ML0_ncnn"): {"delta": -4.8, "ci95": [-9.0, 0.4]}, ("MLP_ncnn", "ML0_ncnn"): {"delta": 2.0, "ci95": [0.3, 3.6]},
            ("MLP_ncnn", "stock_ncnn800"): {"delta": 6.8, "ci95": [1.0, 12.0]}}
    sel = e01.select_frontend(cand, dev, pair, {"MLP_ncnn": False})
    assert sel["locked"] == "MLP_ncnn" and [s["replaces"] for s in sel["steps"]] == [False, True]
    assert e01.select_frontend(cand, dev, pair, {"MLP_ncnn": True})["locked"] == "ML0_ncnn"  # UB harm veto
    pair2 = {**pair, ("MLP_ncnn", "ML0_ncnn"): {"delta": 2.0, "ci95": [-0.1, 3.6]}}
    assert e01.select_frontend(cand, dev, pair2, {})["locked"] == "ML0_ncnn"  # CI touches 0
    dev2 = dict(dev, MLP_ncnn=73.9)  # +1.5 < +2.0
    assert e01.select_frontend(cand, dev2, pair, {})["locked"] == "ML0_ncnn"


# ---- recall matcher / miss decomposition ------------------------------------------------------------------------------
def test_best_conf_per_entry_hit_flags_and_buckets():
    ref = [(0, 100.0, np.array([0, 0, 50, 100.0])), (1, 100.0, np.array([0, 0, 50, 100.0])), (2, 100.0, np.array([0, 0, 50, 100.0])),
           (3, 100.0, np.array([0, 0, 50, 100.0])), (4, 100.0, np.array([0, 0, 50, 100.0]))]
    boxes = {0: (np.array([[0, 0, 50, 100.0], [300, 300, 350, 400.0]]), np.array([0.60, 0.99])),  # good box + a far one
             1: (np.array([[0, 0, 50, 100.0]]), np.array([0.25])),  # exactly at the gate: found
             2: (np.array([[0, 0, 50, 100.0]]), np.array([0.12])),  # in the 0.10-0.25 band
             3: (np.array([[0, 0, 50, 100.0], [2, 2, 52, 102.0]]), np.array([0.03, 0.07])),  # best of two overlapping low boxes: 0.07
             }  # frame 4: nothing cached
    best = e01.best_conf_per_entry(ref, boxes)
    assert np.allclose(best[:4], [0.60, 0.25, 0.12, 0.07]) and np.isnan(best[4])
    assert e01.hit_flags(best).tolist() == [True, True, False, False, False]
    assert [e01.bucket_of_best_conf(x) for x in (0.12, 0.07, 0.03, float("nan"), 0.0099)] == ["0.10-0.25", "0.05-0.10", "0.01-0.05", "none", "none"]
    # IoU below 0.3 does not count
    far_box = {0: (np.array([[40, 0, 90, 100.0]]), np.array([0.9]))}
    assert np.isnan(e01.best_conf_per_entry(ref[:1], far_box)[0])


def test_miss_breakdown_classes_and_recoverability_shares():
    tall = 100.0
    box = np.array([0, 0, 50, 100.0])
    tracks = {"v": {1: [(f, tall, box) for f in range(4)],  # intermittent: hit, miss(0.12), miss(0.07), miss(no box)
                    2: [(f, tall, box) for f in range(3)],  # never seen: 0.12, 0.2, nothing
                    3: [(0, tall, box)]}}  # fully matched
    best = {"v": {1: np.array([0.5, 0.12, 0.07, np.nan]), 2: np.array([0.12, 0.20, np.nan]), 3: np.array([0.9])}}
    r = e01.miss_breakdown(tracks, best)
    assert r["missed"] == 6
    assert r["by_kind"]["intermittent"] == {"0.10-0.25": 1, "0.05-0.10": 1, "0.01-0.05": 0, "none": 1}
    assert r["by_kind"]["never_seen"] == {"0.10-0.25": 2, "0.05-0.10": 0, "0.01-0.05": 0, "none": 1}
    assert r["in_band_total"] == 4 and math.isclose(r["share_in_band"], 4 / 6)
    assert r["two_stage_recoverable"] == 2 and r["gate_only_recoverable"] == 2
    assert math.isclose(r["share_two_stage_recoverable"] + r["share_gate_only_recoverable"], r["share_in_band"])


def test_bootstrap_count_diff_is_zero_for_equal_arms_and_positive_when_a_is_better():
    rng = np.random.default_rng(3)
    persons = rng.integers(20, 80, size=(40, 6))
    found_b = (persons * 0.5).astype(int)
    a = np.stack([persons, found_b + 10], axis=-1)
    b = np.stack([persons, found_b], axis=-1)
    same = e01.bootstrap_count_diff(b, b, [0, 1, 2, 3, 4, 5], n=300)
    assert same["delta"] == 0 and same["ci95"] == [0.0, 0.0]
    better = e01.bootstrap_count_diff(a, b, [0, 1], n=300)
    assert better["delta"] > 0 and better["ci95"][0] > 0
    again = e01.bootstrap_count_diff(a, b, [0, 1], n=300)
    assert again == better  # seeded


# ---- window table from in-memory tracks (identical tracks are scored once) --------------------------------------------
class _CountingScorer:
    def __init__(self):
        self.calls = 0

    def score(self, windows):
        self.calls += len(windows)
        return windows[..., :2].reshape(len(windows), -1).sum(axis=1)  # deterministic function of the window


def test_window_table_from_tracks_matches_clip_stats_and_reuses_the_cache():
    seg = 4
    rng = np.random.default_rng(0)

    def make(n, start=0):
        fr = _frames(start, n)
        return fr, rng.normal(size=(n, 17, 3))

    clips = [{1: make(10), 2: make(3)}, {5: make(6, 2)}]
    scorer, cache = _CountingScorer(), {}
    t = e01.window_table_from_tracks(clips, seg, scorer, cache)
    stats = [e01.clip_track_stats(c, seg)[0] for c in clips]
    assert len(t["normality"]) == sum(s["n_windows"] for s in stats) == 7 + 3
    assert t["clip_n_tracks"].tolist() == [2, 1] and t["clip_n_poses"].tolist() == [13, 6]
    assert scorer.calls == 10
    again = e01.window_table_from_tracks(clips, seg, scorer, cache)
    assert scorer.calls == 10 and np.array_equal(again["normality"], t["normality"])  # nothing re-scored
    assert t["last_frame"].tolist()[:2] == [3, 4] and t["start_frame"].tolist()[:2] == [0, 1]


def test_track_partition_is_id_agnostic_but_sensitive_to_every_pose():
    a = {7: _track([0, 1, 2]), 9: _track([5, 6])}
    b = {1: _track([5, 6]), 2: _track([0, 1, 2])}
    assert e01.track_partition(a) == e01.track_partition(b)
    kp = np.zeros((3, 17, 3))
    kp[1, 0, 0] = 1e-9
    assert e01.track_partition({1: (np.arange(3), kp)}) != e01.track_partition({1: _track([0, 1, 2])})


# ---- lock: arms, contrasts, protocol -----------------------------------------------------------------------------------
def test_arms_cover_the_requested_frontends_and_contrasts_reference_existing_arms():
    arms = e01.build_arms()
    keys = {a.key for a in arms}
    for name in ("released", "stock", "stock_r480", "stock_r800", "stock_r960", "stock_ncnn800", "M00_mps", "ML0_mps", "ML0_ncnn", "MLP_mps", "MLP_ncnn"):
        assert f"ShanghaiTech/{name}" in keys, name
    for name in ("released", "stock", "stock_r480", "stock_r960", "M00_mps", "ML0_mps", "MLP_mps"):
        assert f"UBnormal/{name}" in keys, name
    assert len(keys) == len(arms)
    for c in e01.contrast_specs():
        assert f"{c['dataset']}/{c['a']}" in keys and f"{c['dataset']}/{c['b']}" in keys, c["id"]
    ids = {c["id"] for c in e01.contrast_specs()}
    for needed in ("sht_ML0mps_stock", "sht_ML0mps_stock800", "sht_ML0mps_M00mps", "sht_MLPmps_ML0mps", "sht_ML0ncnn_ML0mps"):
        assert needed in ids
    assert {a.name: a.cost for a in arms if a.dataset == "ShanghaiTech"}["MLP_ncnn"] == 1.92
    # released arms have no detector cache; every other arm has a pose folder and a cache
    for a in arms:
        assert (a.cache_dir is None) == (a.group == "released")


@pytest.mark.skipif(not (e01.E00_DIR / "summary.json").exists(), reason="E00 outputs not present")
def test_locked_protocol_reads_e00_without_retuning():
    _, summ = e01.load_e00()
    proto = e01.locked_protocol(summ, e01.build_arms())
    for ds in e01.DATASETS:
        assert proto[ds]["quantile"] == summ["selection"][ds]["quantile"] and str(proto[ds]["hold"]) == str(summ["selection"][ds]["hold"])
    for a in e01.build_arms():
        v = e01.arm_sentinel(proto, a)
        assert math.isfinite(v) and v == proto[a.dataset]["sentinel_values"][a.sentinel_dist]
        assert e01.arm_sentinel(proto, a, common=True) == proto[a.dataset]["sentinel_values"]["released_all"]


# ---- config.yaml / plan.md consistency (pre-registration) --------------------------------------------------------------
CONFIG = e01.EXP_DIR / "config.yaml"


@pytest.mark.skipif(not CONFIG.exists(), reason="run `python -m edgelab.analysis.q2_p1_e01 lock` first")
def test_config_locks_the_grid_the_arms_the_splits_and_the_e00_hashes():
    cfg = yaml.safe_load(CONFIG.read_text())
    ts = cfg["tracker_sweep"]
    assert ts["n_cells"] == 36 and ts["dev_only"] is True and ts["default_cell"] == e01.DEFAULT_CELL.label
    assert ts["cells"] == [c.label for c in e01.grid_cells()]
    assert len(cfg["frontends"]) == len(e01.build_arms())
    assert {f["key"] for f in cfg["frontends"]} == {a.key for a in e01.build_arms()}
    e00cfg = yaml.safe_load((e01.E00_DIR / "config.yaml").read_text())
    for ds in e01.DATASETS:  # the splits are E00's, byte for byte
        for split in ("test_dev", "test_confirmation"):
            clips = e00cfg["datasets"][ds]["splits"][split]
            assert cfg["splits_from_E00"][ds][split] == {"n": len(clips), "sha256": e01.split_hash(clips)}
    shared = [f for f in cfg["frontends"] if f["matches_e00_cache_hash"] is not None]
    assert shared and all(f["matches_e00_cache_hash"] for f in shared)
    assert cfg["depends_on_E00"]["config_sha256"] == e00.sha256_file(e01.E00_DIR / "config.yaml")
    r = e01.verify_lock(cfg)
    assert r["passes"], r["problems"]


@pytest.mark.skipif(not (e01.EXP_DIR / "plan.md").exists(), reason="plan.md missing")
def test_plan_preregisters_every_rule_the_code_evaluates():
    plan = (e01.EXP_DIR / "plan.md").read_text()
    for rule in ("T1", "T2", "T3", "F1", "F2", "F3", "QA1", "QA7", "QB1", "QB4", "R1", "R2", "R3", "R4", "R5", "R6"):
        assert f"**{rule}**" in plan or f"**{rule} " in plan, rule
    assert "test_dev" in plan and "test_confirmation" in plan


# ---- outputs (only when the analysis has run) --------------------------------------------------------------------------
SUMMARY = e01.EXP_DIR / "summary.json"


@pytest.mark.skipif(not SUMMARY.exists(), reason="run the analysis first")
def test_outputs_are_complete_consistent_and_every_rule_passes():
    s = json.loads(SUMMARY.read_text())
    for f in ("plan.md", "config.yaml", "results.csv", "decisions.jsonl", "summary.json", "report.md"):
        assert (e01.EXP_DIR / f).exists() and (e01.EXP_DIR / f).stat().st_size > 0, f
    failed = {k: v["detail"] for k, v in s["rules"].items() if v["status"] == "FAIL"}
    assert not failed, failed
    # the dev-only sweep never touched a confirmation clip
    assert s["tracker_sweep"]["dev_only"] is True
    _, e00s = e01.load_e00()
    assert s["tracker_sweep"]["dev_clip_counts"]["ShanghaiTech/ML0_ncnn"] == e00s["R8_splits"]["ShanghaiTech"]["n_dev"]
    assert s["tracker_sweep"]["dev_clip_counts"]["UBnormal/stock"] == e00s["R8_splits"]["UBnormal"]["n_dev"]
    # the locked tracker is the default or was adopted by T2 and survived T3
    lock = s["locked_for_E02"]
    default = e01.DEFAULT_CELL.label
    assert lock["tracker"]["label"] == default or (s["tracker_sweep"]["T2"]["adopted"] and not s["tracker_verification"]["T3"]["failed"])
    # E01's S0/S3/S4 numbers reproduce E00's on every shared arm
    for arm in s["arms"].values():
        if arm["e00_parity"]:
            assert max(abs(v) for st in arm["e00_parity"].values() for v in st.values()) <= 0.01
    # results.csv has one row per video x arm (stored pipeline) for every arm
    import csv

    with open(e01.EXP_DIR / "results.csv") as fh:
        rows = list(csv.DictReader(fh))
    stored = [r for r in rows if r["tracker"] == "stored-default"]
    n_expected = sum(len(e00s["R8_splits"][a["dataset"]] and range(e00s["R8_splits"][a["dataset"]]["n_test"])) for a in s["arms"].values())
    assert len(stored) == n_expected
    lines = [json.loads(x) for x in (e01.EXP_DIR / "decisions.jsonl").read_text().splitlines()]
    assert {"arm_result", "contrast", "recall", "miss_decomposition", "tracker_cell", "tracker_selection", "frontend_selection", "rule", "lock"} <= {x["kind"] for x in lines}
