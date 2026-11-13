"""Q2-P1-E00 audit: pure rules (windows, landing, fills, causal delivery, splits, hashes, cadence) on toy data,
plus parity checks against third_party/STG-NF and the older refresh_policy deliverer."""

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest

from edgelab.analysis import q2_p1_e00 as e00
from edgelab.pose.segments import window_starts
from edgelab.pose.stgnf_eval import STGNF_ROOT, _import_stgnf, assemble_clip_scores
from pose_toy import write_toy_dir

CHECKPOINT = STGNF_ROOT / "checkpoints/ShanghaiTech_85_9.tar"


# ---- hashing ---------------------------------------------------------------------------------------------------
def test_sha256_file_is_the_full_digest_and_tree_hash_detects_changes(tmp_path):
    (tmp_path / "a.npz").write_bytes(b"x" * (5 << 20) + b"tail")  # larger than one read chunk: a prefix hash would miss the tail
    (tmp_path / "b.npz").write_bytes(b"hello")
    assert e00.sha256_file(tmp_path / "a.npz") == hashlib.sha256(b"x" * (5 << 20) + b"tail").hexdigest()
    first = e00.tree_manifest(tmp_path)
    assert first["n_files"] == 2 and first["files"]["b.npz"] == hashlib.sha256(b"hello").hexdigest()
    assert e00.tree_manifest(tmp_path)["tree_sha256"] == first["tree_sha256"]
    (tmp_path / "b.npz").write_bytes(b"hellp")
    assert e00.tree_manifest(tmp_path)["tree_sha256"] != first["tree_sha256"]


def test_verify_lock_flags_changed_missing_and_unchanged_files(tmp_path):
    good, bad = tmp_path / "good.bin", tmp_path / "bad.bin"
    good.write_bytes(b"1")
    bad.write_bytes(b"2")
    config = {"frozen_artifacts": {"good": {"path": str(good), "sha256": e00.sha256_file(good)},
                                   "bad": {"path": str(bad), "sha256": e00.sha256_file(bad)},
                                   "gone": {"path": str(tmp_path / "gone"), "sha256": "00"}},
              "detector_caches": {}}
    bad.write_bytes(b"changed")
    result = e00.verify_lock(config)
    assert not result["passes"] and len(result["problems"]) == 2
    assert any("bad" in p and "changed" in p for p in result["problems"]) and any("gone" in p for p in result["problems"])


# ---- splits ------------------------------------------------------------------------------------------------------
def _toy_items(n_scenes=6, per_scene=(9, 3, 1, 5, 7, 2)):
    items = [f"{s:02d}_{i:04d}" for s in range(n_scenes) for i in range(per_scene[s])]
    return items, {it: it.split("_")[0] for it in items}


def test_split_is_exact_disjoint_deterministic_and_covers_every_big_stratum():
    items, strata = _toy_items()
    dev, conf = e00.stratified_dev_split(items, strata, 9, seed=1)
    assert len(dev) == 9 and not set(dev) & set(conf) and set(dev) | set(conf) == set(items)
    assert e00.stratified_dev_split(items, strata, 9, seed=1) == (dev, conf)
    assert e00.stratified_dev_split(items, strata, 9, seed=2)[0] != dev  # seed matters
    # proportional up to rounding: a stratum of n clips contributes n * 9 / 27 +- 1 to dev
    for scene in {s for s in strata.values()}:
        n = sum(1 for v in strata.values() if v == scene)
        got = sum(1 for it in dev if strata[it] == scene)
        assert abs(got - n * 9 / len(items)) <= 1.0


def test_split_rejects_degenerate_sizes():
    items, strata = _toy_items()
    with pytest.raises(ValueError):
        e00.stratified_dev_split(items, strata, 0)
    with pytest.raises(ValueError):
        e00.stratified_dev_split(items, strata, len(items))


# ---- windows: the omission bug and continuity ----------------------------------------------------------------------
def test_complete_windows_include_the_last_one_that_stgnf_omits():
    frames = np.arange(30)
    starts = e00.complete_window_starts(frames, 24)
    assert list(starts) == list(range(7))  # starts 0..6: n - seg_len + 1 windows
    offline = e00.stgnf_offline_window_mask(starts, len(frames), 24)
    assert offline.tolist() == [True] * 6 + [False]  # exactly the last window is dropped by STG-NF
    assert list(e00.complete_window_starts(np.arange(24), 24)) == [0]  # a track of exactly seg_len poses is a window ...
    assert not e00.stgnf_offline_window_mask(np.array([0]), 24, 24).any()  # ... that STG-NF never scores
    assert len(e00.complete_window_starts(np.arange(23), 24)) == 0


def test_offline_windows_equal_the_existing_stride_one_window_rule():
    rng = np.random.default_rng(0)
    for _ in range(20):
        frames = np.sort(rng.choice(np.arange(200), size=rng.integers(20, 120), replace=False))
        mine = e00.complete_window_starts(frames, 24)
        theirs = window_starts(frames, 24, 1)
        assert list(mine[e00.stgnf_offline_window_mask(mine, len(frames), 24)]) == list(theirs)


def test_continuity_allows_two_missing_frames_but_not_three():
    two_missing = np.delete(np.arange(40), [5, 9])
    three_missing = np.delete(np.arange(40), [5, 9, 11])
    assert 0 in e00.complete_window_starts(two_missing, 24)
    assert 0 not in e00.complete_window_starts(three_missing, 24)


def test_windows_equal_the_ones_stgnf_cuts_on_toy_poses(tmp_path):
    directory = write_toy_dir(tmp_path)
    _import_stgnf()
    from dataset import gen_dataset

    _, meta, *_ = gen_dataset(str(directory), seg_stride=1, seg_len=24, ret_keys=True, ret_global_data=False)
    theirs = {(int(m[0]), int(m[1]), int(m[2]), int(m[3])) for m in meta}
    mine = set()
    for path in sorted(directory.glob("*tracked_person.json")):
        scene, clip = (int(x) for x in path.name.split("_")[:2])
        for pid, (frames, _) in e00.read_pose_clip(path).items():
            starts = e00.complete_window_starts(frames, 24)
            keep = e00.stgnf_offline_window_mask(starts, len(frames), 24)
            mine |= {(scene, clip, pid, int(frames[s])) for s in starts[keep]}
    assert mine == theirs and len(theirs) > 0


# ---- landing, per-frame minimum, global fill (historical rule) -------------------------------------------------------
def test_landing_modes_and_completion_never_precedes_end():
    start, last = np.array([10, 10, 50]), np.array([33, 34, 75])  # window 2 has 2 missing frames (completes 2 frames late)
    assert e00.landing_frames(start, last, 24, "middle").tolist() == [22, 22, 62]
    assert e00.landing_frames(start, last, 24, "end").tolist() == [33, 33, 73]
    assert e00.landing_frames(start, last, 24, "completion").tolist() == [33, 34, 75]
    assert (e00.landing_frames(start, last, 24, "completion") >= e00.landing_frames(start, last, 24, "end")).all()
    assert e00.landing_frames(np.array([0]), np.array([15]), 16, "middle").tolist() == [8]  # UBnormal seg_len 16
    with pytest.raises(ValueError):
        e00.landing_frames(start, last, 24, "bogus")


def test_per_frame_min_takes_the_least_normal_person_and_drops_out_of_range():
    out = e00.per_frame_min(np.array([0, 0, 0, 1]), np.array([2, 2, 9, 1]), np.array([-1.0, -3.0, -5.0, -2.0]), [5, 4])
    assert out[0].tolist() == [np.inf, np.inf, -3.0, np.inf, np.inf]  # two persons on frame 2 -> min; landing 9 is outside the clip
    assert out[1].tolist() == [np.inf, -2.0, np.inf, np.inf]


def test_global_fill_matches_stgnf_assemble_clip_scores(tmp_path):
    rng = np.random.default_rng(3)
    lengths = {"01_0001": 60, "01_0002": 45, "02_0003": 70}
    for name, n in lengths.items():
        np.save(tmp_path / f"{name}.npy", (rng.random(n) > 0.7).astype(int))
    rows, norm = [], []
    for name, n in lengths.items():
        scene, clip = (int(x) for x in name.split("_"))
        for person in (0, 1):
            for start in rng.choice(np.arange(0, n - 24), size=10, replace=False):
                rows.append([scene, clip, person, int(start)])
                norm.append(rng.normal(-1.5, 0.5))
    metadata, normality = np.array(rows), np.array(norm)
    names, _, theirs = assemble_clip_scores(normality, metadata, tmp_path, frame_offset=12, dataset="ShanghaiTech")
    clip_idx = np.array([list(lengths).index(f"{r[0]:02d}_{r[1]:04d}") for r in rows])
    raw = e00.per_frame_min(clip_idx, metadata[:, 3] + 12, normality, list(lengths.values()))
    filled, most_normal = e00.global_most_normal_fill(raw)
    for mine, ref in zip(filled, theirs):
        np.testing.assert_allclose(-mine, ref, atol=0, rtol=0)
    assert most_normal == normality.max()


def test_global_fill_is_look_ahead_but_train_sentinel_is_not():
    a, b = np.array([np.inf, -1.0, -2.0]), np.array([-0.5, np.inf])
    _, with_future = e00.global_most_normal_fill([a, b])
    _, without_future = e00.global_most_normal_fill([a])
    assert with_future == -0.5 and without_future == -1.0  # fill value depends on frames that have not happened yet
    prefix, _, _ = e00.hold_fill(np.array([np.inf, np.inf, -4.0]), sentinel=-0.7, max_hold=0)
    assert prefix[0] == -0.7  # the sentinel is a constant fixed before the video starts


# ---- causal delivery -----------------------------------------------------------------------------------------------
def _random_series(rng, n=300, p_empty=0.4):
    x = rng.normal(-1.0, 0.4, n)
    x[rng.random(n) < p_empty] = np.inf
    return x


@pytest.mark.parametrize("hold", [0, 3, 12, 24])
def test_hold_fill_equals_the_stateful_causal_deliverer(hold):
    from edgelab.pipeline.refresh_policy import CausalScoreDeliverer

    rng = np.random.default_rng(hold)
    raw = _random_series(rng)
    deliverer = CausalScoreDeliverer(train_sentinel_normality=-0.79, max_hold_frames=hold)
    reference = np.array([deliverer.step_frame(t, [raw[t]] if np.isfinite(raw[t]) else []) for t in range(len(raw))])
    mine, stats = e00.deliver([raw], -0.79, hold)
    np.testing.assert_allclose(mine[0], reference, atol=1e-12)
    assert abs(stats["fresh_fraction"] + stats["held_fraction"] + stats["sentinel_fraction"] - 1) < 1e-12


def test_hold_modes_and_ages():
    raw = np.array([np.inf, -1.0, np.inf, np.inf, np.inf, -2.0, np.inf])
    value, mode, age = e00.hold_fill(raw, sentinel=-9.0, max_hold=2)
    assert value.tolist() == [-9.0, -1.0, -1.0, -1.0, -9.0, -2.0, -2.0]
    assert mode.tolist() == [2, 0, 1, 1, 2, 0, 1]  # frame 4 is 3 frames stale > 2 -> sentinel
    assert age.tolist() == [-1, 0, 1, 2, 3, 0, 1]
    forever = e00.hold_fill(raw, -9.0, math.inf)
    assert forever[0].tolist() == [-9.0, -1.0, -1.0, -1.0, -1.0, -2.0, -2.0]
    none = e00.hold_fill(raw, -9.0, 0)
    assert none[1].tolist() == [2, 0, 2, 2, 2, 0, 2]


def test_delivered_score_is_prefix_invariant_but_global_fill_is_not():
    rng = np.random.default_rng(5)
    raw = _random_series(rng, 200)
    full, _ = e00.deliver([raw], -0.8, 12)
    for t in (10, 57, 120, 199):
        prefix, _ = e00.deliver([raw[: t + 1]], -0.8, 12)
        assert abs(prefix[0][-1] - full[0][t]) <= 1e-12
    raw[190] = -0.1  # a very normal frame in the future: the global fill of EARLY frames changes
    filled_a, _ = e00.global_most_normal_fill([raw[:150]])
    filled_b, _ = e00.global_most_normal_fill([raw])
    empty = np.isinf(raw[:150])
    assert filled_a[0][empty][0] != filled_b[0][:150][empty][0]


def test_unscored_frames_are_not_free_normal_a_low_sentinel_penalises_empty_normal_frames():
    """The EXP-023 sentinel finding in miniature: normal frames with nobody in them rank above/below by the sentinel chosen."""
    labels = np.array([0] * 40 + [1] * 40)
    raw = np.concatenate([np.full(40, np.inf), np.full(40, -2.0)])  # normal part: empty scene; anomalous part: poor scores
    low = e00.deliver([raw], sentinel=-3.0, max_hold=0)[0][0]  # sentinel LESS normal than the anomaly -> empty normals look worse
    high = e00.deliver([raw], sentinel=-0.5, max_hold=0)[0][0]
    assert e00.clip_auc(labels, high) > e00.clip_auc(labels, low)


# ---- decomposition on a synthetic window table --------------------------------------------------------------------------
def _synthetic_source(seg_len=24, n=80):
    rng = np.random.default_rng(1)
    labels = [np.r_[np.zeros(40), np.ones(40)].astype(int)]
    starts = np.arange(0, n - seg_len + 1)
    table = {"clip_idx": np.zeros(len(starts), dtype=np.int32), "person": np.zeros(len(starts), dtype=np.int32),
             "start_idx": starts.astype(np.int32), "n_track": np.full(len(starts), n, dtype=np.int32),
             "start_frame": starts.astype(np.int32), "last_frame": (starts + seg_len - 1).astype(np.int32),
             "in_offline": starts < n - seg_len, "normality": rng.normal(-1.0, 0.3, len(starts)),
             "clip_n_tracks": np.array([1]), "clip_n_poses": np.array([n])}
    source = e00.Source("toy", "ShanghaiTech", Path("."), None, None, None, "toy")
    return e00.SourceData(source, ["toy"], labels, table, seg_len, [0], [0])


def test_step_definitions_differ_exactly_where_the_plan_says():
    d = _synthetic_source()
    s1_raw, s2_raw = d.raw_per_frame("end", "offline"), d.raw_per_frame("completion", "all")
    assert np.isfinite(s2_raw[0]).sum() == np.isfinite(s1_raw[0]).sum() + 1  # the omitted last window adds one scored frame
    assert np.isfinite(s2_raw[0][-1]) and not np.isfinite(s1_raw[0][-1])
    paper = d.raw_per_frame("middle", "offline")[0]
    assert np.isfinite(paper).nonzero()[0][0] == 12  # paper score lands 12 frames after the window starts ...
    assert np.isfinite(s1_raw[0]).nonzero()[0][0] == 23  # ... live lands when the 24th pose exists
    for step in e00.STEP_SPECS:
        scores, _ = d.step_scores(step)
        assert scores[0].shape == (80,) and np.isfinite(scores[0]).all()


def test_causality_tests_on_synthetic_source_pass():
    result = e00.causality_tests(_synthetic_source(), sentinel=-0.8)
    assert result["prefix_invariance_max_abs_diff"] <= 1e-12
    assert result["matches_CausalScoreDeliverer_max_abs_diff"] <= 1e-9
    assert result["windows_landing_before_completion_frame"] == 0
    assert result["global_fill_is_look_ahead"] in (True, False)


def test_select_config_prefers_smaller_hold_within_the_tie_band():
    grid = {"a": {"auc": {"dev": 70.00}, "quantile": "max", "hold": 24, "sentinel": -0.5},
            "b": {"auc": {"dev": 69.95}, "quantile": "p90", "hold": 0, "sentinel": -0.8},
            "c": {"auc": {"dev": 69.50}, "quantile": "p50", "hold": 0, "sentinel": -1.0}}
    assert e00.select_config(grid, tie=0.10)["key"] == "b"
    assert e00.select_config(grid, tie=0.01)["key"] == "a"


# ---- cadence ---------------------------------------------------------------------------------------------------------
def test_native_frame_rates_and_timestamps():
    assert e00.FPS == {"ShanghaiTech": 24.0, "UBnormal": 30.0}
    assert e00.frame_time_ms("UBnormal", 30) == pytest.approx(1000.0)
    assert e00.frame_time_ms("ShanghaiTech", 24) == pytest.approx(1000.0)
    np.testing.assert_allclose(e00.frame_time_ms("UBnormal", np.array([0, 3])), [0.0, 100.0])
    # EXP-023's 24 fps assumption would have overstated delay by 25 % and understated false alerts/hour by 20 % on UBnormal
    assert e00.normal_hours("UBnormal", 30 * 3600) == pytest.approx(1.0)
    assert e00.normal_hours("ShanghaiTech", 24 * 3600) == pytest.approx(1.0)
    assert 30 * 16 and e00.SEG_LENS["UBnormal"] == 16 and e00.SEG_LENS["ShanghaiTech"] == 24


def test_exp023_hardcoded_24fps_is_still_there_to_be_documented():
    text = (e00.EXPERIMENTS_ROOT / "src/edgelab/analysis/exp023.py").read_text()
    assert "fps=24.0" in text and "input_fps\": 24.0" in text  # untouched: the earlier experiment is preserved


# ---- normalisation / reuse -----------------------------------------------------------------------------------------------
def _toy_track(n=60, seed=0):
    rng = np.random.default_rng(seed)
    body = rng.uniform(-30, 30, (17, 2))
    t = np.arange(n)[:, None, None]
    xy = 300 + 1.5 * t + 0.4 * t ** 1.2 * np.array([1.0, 0.3]) + body + rng.normal(0, 0.5, (n, 17, 2))
    return np.arange(n), np.concatenate([xy, rng.uniform(0.3, 0.9, (n, 17, 1))], axis=2)


def test_per_frame_pose_reuse_across_overlapping_windows_is_bit_exact():
    frames, kp = _toy_track()
    starts = e00.complete_window_starts(frames, 24)
    shared = e00.gather_windows(kp, starts, 24)
    streamed = np.stack([w for _, w in e00.streaming_windows(frames, kp, 24)])
    independent = np.stack([kp[s:s + 24].copy() for s in starts])
    assert np.array_equal(shared, streamed) and np.array_equal(shared, independent)
    assert float(np.abs(shared - streamed).max()) == 0.0
    # window w+1 reuses 23 of the 24 per-frame poses of window w
    assert np.array_equal(shared[1:, :-1], shared[:-1, 1:])


def test_normalization_is_per_window_so_post_normalization_reuse_is_wrong():
    frames, kp = _toy_track(n=48)
    w = e00.gather_windows(kp, e00.complete_window_starts(frames, 24), 24)
    per_window = e00.normalize_windows(w)
    # normalising each window with ITS OWN stats == explicit stats path
    for i in (0, 5, 12):
        mean, std = e00.window_stats(w[i:i + 1])
        np.testing.assert_allclose(e00.normalize_with_stats(w[i:i + 1], mean[0], float(std[0])), per_window[i:i + 1], atol=1e-12)
    # window 12's frames 1..23 normalised as part of window 11 differ from the same frames normalised in window 12
    assert not np.allclose(per_window[11, 1:], per_window[12, :-1], atol=1e-6)
    # ... but the RAW per-frame poses of those frames are identical
    assert np.array_equal(w[11, 1:], w[12, :-1])
    # reusing window 0's statistics for window 12 is not the same input
    mean0, std0 = e00.window_stats(w[:1])
    assert np.abs(e00.normalize_with_stats(w[12:13], mean0[0], float(std0[0])) - per_window[12:13]).max() > 1e-3


def test_causal_smoother_is_the_reference_implementation():
    from edgelab.pose.smoothing import causal_half_gaussian

    x = np.random.default_rng(0).normal(size=200)
    assert np.array_equal(e00.causal_smooth(x), causal_half_gaussian(x))


# ---- data-dependent parity (skipped when the data / checkpoints are not on this machine) -----------------------------------------
needs_checkpoint = pytest.mark.skipif(not CHECKPOINT.exists(), reason="STG-NF checkpoint not available")


@needs_checkpoint
def test_window_engine_reproduces_stgnf_segments_and_scores_on_toy_poses(tmp_path):
    from edgelab.pose.stgnf_eval import score_segments

    directory = write_toy_dir(tmp_path)
    segments = score_segments(directory, CHECKPOINT, device="cpu", dataset="ShanghaiTech")
    scorer = e00.Scorer("ShanghaiTech")
    files = e00.index_pose_dir(directory, "ShanghaiTech")
    clips = sorted(files)
    table = e00.build_window_table(files, clips, scorer)
    off = table["in_offline"]
    mine = {(clips[c], int(p), int(s)): float(n) for c, p, s, n in zip(table["clip_idx"][off], table["person"][off],
                                                                      table["start_frame"][off], table["normality"][off])}
    theirs = {(f"{int(m[0]):02d}_{int(m[1]):03d}", int(m[2]), int(m[3])): float(n) for m, n in zip(segments.metadata, segments.normality)}
    assert set(mine) == set(theirs)
    assert max(abs(mine[k] - theirs[k]) for k in mine) < 1e-4
    assert (~off).sum() == sum(1 for t in e00.read_pose_clip(files["01_001"]).values() if len(t[0]) >= 24) + \
        sum(1 for t in e00.read_pose_clip(files["02_005"]).values() if len(t[0]) >= 24)  # one omitted window per scorable track


@needs_checkpoint
def test_reuse_audit_on_toy_poses_per_frame_exact_and_negative_controls_fail(tmp_path):
    directory = write_toy_dir(tmp_path)
    scorer = e00.Scorer("ShanghaiTech")
    files = e00.index_pose_dir(directory, "ShanghaiTech")
    result = e00.reuse_audit("ShanghaiTech", files, sorted(files), scorer, max_clips=2)
    assert result["per_frame_reuse"]["input_max_abs_diff_shared_vs_recomputed"] == 0.0
    assert result["per_frame_reuse"]["score_max_abs_diff_same_batch"] == 0.0
    assert result["per_frame_reuse"]["input_bit_identical_shared_vs_streaming"]
    assert result["post_normalization_reuse"]["frozen_first_window_stats"]["input_max_abs_diff"] > 0
    assert result["post_normalization_reuse"]["frozen_first_window_stats"]["score_max_abs_diff"] > 0
    assert result["hidden_state_reuse"]["max_abs_diff_shared_norm_stats"] > 0
    assert result["reuse_factor"] > 5 and result["passes"]


# ---- serialisation -------------------------------------------------------------------------------------------------------------
def test_to_plain_handles_numpy_inf_and_paths():
    plain = e00.to_plain({"a": np.float32(1.5), "b": np.arange(3), "c": math.inf, "d": float("nan"), "e": Path("/x"), "f": np.bool_(True)})
    assert plain == {"a": 1.5, "b": [0, 1, 2], "c": "inf", "d": None, "e": "/x", "f": True}
    json.dumps(plain)
