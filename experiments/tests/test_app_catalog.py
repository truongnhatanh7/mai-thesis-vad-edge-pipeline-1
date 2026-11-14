"""Tests for the Streamlit explorer catalog and inference helpers."""

import numpy as np

from edgelab.app.catalog import (
    MODEL_BY_ID,
    build_model_catalog,
    build_scorer_catalog,
    find_cache_npz,
    find_pose_json,
    find_scores_npz,
    iter_video_frames,
    list_dataset_videos,
    load_dataset_summary_row,
    load_ground_truth,
    read_video_frame,
)
from edgelab.app.inference import (
    LivePipelineSession,
    apply_lighting_and_enhancer,
    draw_poses_and_boxes,
    load_model_handle,
)


def test_model_catalog_has_core_trained_models():
    models = build_model_catalog()
    ids = {m.id for m in models}
    for expected in (
        "stock-yolov8n-640",
        "exp020-ML0",
        "exp020-MLP",
        "exp020-M00",
        "exp020-M0P",
        "exp013-sht-ft",
        "exp014-wise-0.75",
        "exp016-topdown-640",
        "exp016-topdown-960",
        "released-alphapose",
    ):
        assert expected in ids
        spec = MODEL_BY_ID[expected]
        if spec.weights_path is not None:
            assert spec.weights_path.exists(), f"missing weights for {expected}: {spec.weights_path}"


def test_scorer_catalog_lists_existing_checkpoints():
    for ds in ("ShanghaiTech", "UBnormal"):
        scorers = build_scorer_catalog(ds)
        assert len(scorers) >= 1
        for s in scorers:
            assert s.checkpoint.exists()


def test_datasets_and_precomputed_artifacts_resolve():
    sht_videos = list_dataset_videos("ShanghaiTech")
    ub_videos = list_dataset_videos("UBnormal")
    assert len(sht_videos) == 107
    assert len(ub_videos) == 211

    ml0 = MODEL_BY_ID["exp020-ML0"]
    assert find_cache_npz(ml0, "ShanghaiTech", "01_0014", "L0", "E0") is not None
    assert find_pose_json(ml0, "ShanghaiTech", "01_0014", "L0", "E0") is not None
    assert find_scores_npz(ml0, "ShanghaiTech", "L0", "E0", variant="paper") is not None
    row = load_dataset_summary_row(ml0, "ShanghaiTech", "L0", "E0", variant="paper")
    assert row is not None and 0.75 < row["auc"] < 0.85

    ub_vid = ub_videos[0].video_id
    assert find_pose_json(ml0, "UBnormal", ub_vid, "L0", "E0") is not None
    assert find_scores_npz(ml0, "UBnormal", "L0", "E0", variant="paper") is not None


def test_frame_reading_and_drawing_pipeline():
    img = read_video_frame("ShanghaiTech", "01_0014", 0)
    assert img is not None and img.ndim == 3
    gt = load_ground_truth("ShanghaiTech", "01_0014")
    assert len(gt) == 265

    dark, enh, stats = apply_lighting_and_enhancer(img, "01_0014", 0, level="L2", enhance="E2")
    assert dark.shape == img.shape and enh.shape == img.shape
    assert stats["luma_dark"] < stats["luma_orig"]

    boxes = np.array([[100.0, 120.0, 140.0, 170.0]], dtype=np.float32)
    confs = np.array([0.85], dtype=np.float32)
    kpts = np.zeros((1, 17, 3), dtype=np.float32)
    kpts[0, :, 0] = 120.0
    kpts[0, :, 1] = 145.0
    kpts[0, :, 2] = 0.9
    drawn = draw_poses_and_boxes(enh, boxes=boxes, confs=confs, keypoints=kpts, track_ids=[1])
    assert drawn.shape == img.shape


def test_iter_video_frames_and_live_pipeline_session():
    frames_sht = list(iter_video_frames("ShanghaiTech", "01_0014", start_frame=0, end_frame=6, step=2))
    assert [f[0] for f in frames_sht] == [0, 2, 4]
    assert all(f[1].ndim == 3 for f in frames_sht)

    ub_vid = list_dataset_videos("UBnormal")[0].video_id
    frames_ub = list(iter_video_frames("UBnormal", ub_vid, start_frame=0, end_frame=6, step=3))
    assert [f[0] for f in frames_ub] == [0, 3]

    spec = MODEL_BY_ID["exp020-ML0"]
    handle = load_model_handle(spec)
    session = LivePipelineSession(spec, "ShanghaiTech", conf_thresh=0.25, device="cpu", loaded_handle=handle)
    rgb, info = session.step(0, frames_sht[0][1])
    assert rgb.shape == frames_sht[0][1].shape
    assert info["num_persons"] >= 1
    assert info["total_ms"] > 0



# ---------------------------------------------------------------------------------------------------------------------
# Dataset Explorer helpers
# ---------------------------------------------------------------------------------------------------------------------


def test_gt_interval_helpers():
    from edgelab.app.catalog import format_interval, frames_to_seconds, gt_intervals, interval_seconds, native_fps

    assert gt_intervals(np.array([0, 1, 1, 0, 0, 1])) == [(1, 2), (5, 5)]
    assert gt_intervals(np.zeros(5)) == []
    assert gt_intervals(np.array([])) == []
    assert interval_seconds(24, 47, 24.0) == (1.0, 2.0)
    assert format_interval(24, 47, 24.0) == "f24-47 (1.00-2.00 s)"
    assert native_fps("ShanghaiTech") == 24.0 and native_fps("UBnormal") == 30.0
    assert frames_to_seconds(60, 30.0) == 2.0


def test_event_tags_and_events_of_a_real_clip():
    from edgelab.app.catalog import build_gt_events, event_peak_frame, event_tags

    assert event_tags(True, True, False, None) == ("short_event", "new_entry_near_onset")
    assert event_tags(False) == ("sustained_event",)

    events = build_gt_events("ShanghaiTech", "01_0014")
    assert len(events) == 1
    ev = events[0]
    assert (ev.start_frame, ev.end_frame) == (154, 229)
    assert ev.duration_frames == 76 and not ev.is_short
    assert "sustained_event" in ev.tags and "new_entry_near_onset" in ev.tags
    assert abs(ev.start_s - 154 / 24) < 1e-9

    assert event_peak_frame(ev) == ev.mid_frame
    scores = np.zeros(265)
    scores[200] = 5.0
    assert event_peak_frame(ev, scores) == 200
    scores[100] = 99.0  # outside the event: must be ignored
    assert event_peak_frame(ev, scores) == 200


def test_split_config_and_clip_tables_match_the_frozen_split():
    from edgelab.app.catalog import build_clip_table, load_split_map, parse_split_config, summarize_clip_table

    cfg = {"datasets": {"X": {"splits": {"test_dev": ["a", "b"], "test_confirmation": ["c"]}}}}
    assert parse_split_config(cfg, "X") == {"a": "test_dev", "b": "test_dev", "c": "test_confirmation"}

    split = load_split_map("ShanghaiTech")
    assert len(split) == 107
    assert sum(v == "test_dev" for v in split.values()) == 35

    sht = summarize_clip_table(build_clip_table("ShanghaiTech"))
    assert sht["clips"] == 107 and sht["fps"] == 24.0 and sht["scenes"] == 12
    assert sht["events"] == 194 and sht["short_events"] + sht["sustained_events"] == 194
    assert sht["test_dev_clips"] == 35 and sht["test_confirmation_clips"] == 72
    assert set(sht["clips_with_tag"]) >= {"short_event", "sustained_event", "new_entry_near_onset", "sudden_fast_motion", "high_crowd"}

    ub = summarize_clip_table(build_clip_table("UBnormal"))
    assert ub["clips"] == 211 and ub["fps"] == 30.0
    assert ub["events"] == 199 and ub["test_dev_clips"] + ub["test_confirmation_clips"] == 211

    row = build_clip_table("ShanghaiTech")[0]
    assert row["clip"] == "01_0014" and row["frames"] == 265 and row["scene"] == "01" and row["n_events"] == 1
    assert row["events"] == "f154-229 (6.42-9.58 s)"
    assert summarize_clip_table([]) == {"clips": 0}


def test_experiment_catalog_includes_exp_and_q2_ids():
    from edgelab.app.catalog import EXPERIMENT_BY_ID, build_experiment_catalog, load_experiment_artifacts, summarize_results_table

    specs = build_experiment_catalog()
    ids = [s.id for s in specs]
    assert len(ids) == len(set(ids))
    for i in range(24):
        assert f"EXP-{i:03d}" in ids
    for i in range(8):
        assert f"Q2-P1-E{i:02d}" in ids
    assert "Q2-P1-Pi5_huang_repro" in ids and "Q2-P1-Pi5_suite" in ids
    assert set(EXPERIMENT_BY_ID) == set(ids)

    art = load_experiment_artifacts("Q2-P1-E05")
    assert art["spec"].id == "Q2-P1-E05"
    assert art["events"] is not None and len(art["events"]) > 0
    table = summarize_results_table(art["results"])
    assert table is None or len(table) > 0
    assert summarize_results_table(None) is None

    # Experiments without artefacts yet must still load without raising.
    for i in (6, 7):
        out = load_experiment_artifacts(f"Q2-P1-E0{i}")
        assert out["spec"].status


def test_overlay_sources_are_resolvable_for_a_real_clip():
    from edgelab.app.catalog import OVERLAY_BY_KEY, OVERLAY_SOURCES, boxes_from_keypoints, load_overlay_detections, overlay_availability

    assert {s.key for s in OVERLAY_SOURCES} >= {"released", "stock", "ml0_ncnn", "mlp", "yolo26"}
    assert set(OVERLAY_BY_KEY) == {s.key for s in OVERLAY_SOURCES}
    avail = overlay_availability("ShanghaiTech", "01_0014")
    assert avail["stock"] and avail["ml0_ncnn"]

    det = load_overlay_detections("ml0_ncnn", "ShanghaiTech", "01_0014", 180, min_conf=0.05)
    assert det is not None
    n = len(det["boxes"])
    assert det["boxes"].shape == (n, 4) and len(det["conf"]) == n
    assert n == 0 or float(det["conf"].min()) >= 0.05

    kp = np.zeros((1, 17, 3), dtype=np.float32)
    kp[0, :, 0] = np.linspace(10, 30, 17)
    kp[0, :, 1] = np.linspace(20, 80, 17)
    kp[0, :, 2] = 0.9
    b = boxes_from_keypoints(kp)
    assert b.shape == (1, 4) and b[0, 0] == 10 and b[0, 3] == 80
    assert np.all(boxes_from_keypoints(np.zeros((1, 17, 3))) == 0)


def test_height_confidence_overlay_helpers():
    from edgelab.app.inference import confidence_class, draw_height_conf_overlay, height_bin_label, summarize_detections

    assert [height_bin_label(h) for h in (10, 31.9, 32, 47.9, 48, 63.9, 64, 300)] == [
        "<32 px", "<32 px", "32-48 px", "32-48 px", "48-64 px", "48-64 px", ">=64 px", ">=64 px"]
    assert confidence_class(0.20) == "high" and confidence_class(0.9) == "high"
    assert confidence_class(0.05) == "low" and confidence_class(0.199) == "low"
    assert confidence_class(0.049) == "ignored"

    boxes = np.array([[0, 0, 10, 20], [0, 0, 30, 90], [0, 0, 20, 40], [0, 0, 20, 60]], dtype=np.float32)
    confs = np.array([0.5, 0.1, 0.01, 0.3], dtype=np.float32)
    s = summarize_detections(boxes, confs)
    assert s["n_high"] == 2 and s["n_low"] == 1  # the 0.01 box is ignored
    assert s["counts"][("<32 px", "high")] == 1 and s["counts"][(">=64 px", "low")] == 1
    assert s["counts"][("48-64 px", "high")] == 1
    assert s["n_small_lt48"] == 1
    empty = summarize_detections(None, None)
    assert empty["n_high"] == 0 and empty["n_low"] == 0

    img = np.full((120, 160, 3), 90, dtype=np.uint8)
    out = draw_height_conf_overlay(img, boxes, confs, None, title="t", gt_anomalous=True, highlight_boxes=boxes[:2])
    assert out.shape == img.shape and out.dtype == np.uint8
    assert not np.array_equal(out, img) and np.array_equal(img, np.full((120, 160, 3), 90, dtype=np.uint8))
    assert draw_height_conf_overlay(img, None, None, None).shape == img.shape


# ---------------------------------------------------------------------------------------------------------------------
# Failure Case Inspector helpers
# ---------------------------------------------------------------------------------------------------------------------


def test_diagnostic_primitives_on_synthetic_data():
    from edgelab.app import diagnostics as D

    assert list(D.clip_offsets([3, 2, 4])) == [0, 3, 5, 9]
    assert D.clip_auc(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9])) == 100.0
    assert D.clip_auc(np.array([0, 0, 1, 1]), np.array([0.9, 0.8, 0.2, 0.1])) == 0.0
    assert D.clip_auc(np.array([0, 1]), np.array([0.5, 0.5])) == 50.0
    assert np.isnan(D.clip_auc(np.zeros(4), np.arange(4)))

    above = np.array([0, 1, 1, 1, 1, 1, 1, 0, 1, 1], dtype=bool)
    assert D.alert_runs(above) == [(1, 6)]
    assert D.alert_runs(above, min_len=2) == [(1, 6), (8, 9)]
    assert D.alert_runs(np.array([], dtype=bool)) == []
    assert D.event_alert_delay((3, 20), [(5, 10)]) == 7.0  # alert raised after min_len frames from the run start
    assert D.event_alert_delay((3, 20), [(21, 30)]) is None
    assert D.event_alert_delay((3, 20), []) is None

    series = np.array([0.0, 0.0, 0.5, 0.9, 0.0])
    assert list(D.scorable_mask(series, 0.0)) == [False, False, True, True, False]

    assert D.runs_of([5, 3, 4, 9, 10, 10]) == [(3, 5), (9, 10)]
    assert D.runs_of([]) == []
    iou = D.box_iou(np.array([[0, 0, 10, 10]]), np.array([[0, 0, 10, 10], [20, 20, 30, 30], [5, 0, 15, 10]]))
    assert iou.shape == (1, 3)
    assert iou[0, 0] == 1.0 and iou[0, 1] == 0.0 and abs(iou[0, 2] - 1 / 3) < 1e-6


def test_small_person_and_gap_bridging_logic_on_synthetic_detections():
    from edgelab.app import diagnostics as D

    def frame(boxes, confs):
        return {"boxes": np.array(boxes, dtype=np.float32).reshape(-1, 4), "conf": np.array(confs, dtype=np.float32)}

    small = [0, 0, 10, 30]   # 30 px tall
    big = [100, 100, 140, 220]
    ml0 = [frame([small, big], [0.6, 0.9]), frame([big], [0.9])]
    stock = [frame([big], [0.9]), frame([big], [0.9])]
    res = D.small_person_miss_frames(ml0, stock)
    assert list(res["n_small_ml0"]) == [1, 0]
    assert list(res["n_unmatched"]) == [1, 0]  # only frame 0 holds the small person that stock cannot see
    assert res["heights"][0] == [30.0] and res["heights"][1] == []

    # a person present at frames 0 and 10 as HIGH, with a LOW box at frame 5 -> exactly that frame is bridged
    hi = [10, 10, 50, 110]
    frames = [frame([], [])] * 11
    frames[0] = frame([hi], [0.9])
    frames[5] = frame([hi], [0.1])
    frames[10] = frame([hi], [0.9])
    out = D.gap_bridging_frames(frames)
    assert [o["frame"] for o in out] == [5]
    assert D.gap_bridging_frames([frame([], [])] * 3) == []


def test_failure_rankings_on_real_artifacts():
    from edgelab.app import diagnostics as D

    small = D.rank_small_person_misses("ShanghaiTech")
    assert small.iloc[0]["clip"] == "01_0140"
    assert small["auc_gain"].is_monotonic_decreasing
    assert small.iloc[0]["auc_stock"] < 20 < 80 < small.iloc[0]["auc_ml0"]

    frag = D.rank_track_fragmentation("ShanghaiTech")
    assert frag.iloc[0]["clip"] == "01_0016" and frag.iloc[0]["auc_gain"] > 20

    scorer = D.rank_scorer_limited("ShanghaiTech")
    assert len(scorer) == 194
    assert int(scorer["tracked_but_silent"].sum()) > 0
    assert set(scorer["alerted"].unique()) <= {True, False}

    pi5 = D.rank_scheduling("ShanghaiTech", "queue_pi5")
    assert len(pi5) == 12
    top = pi5.iloc[0]
    assert top["clip"] == "12_0149" and top["wait_max_s_fixed_k1"] > 40 and top["wait_max_ms_guard"] < 500

    for variant in D.SCHEDULING_VARIANTS:
        for ds in ("ShanghaiTech", "UBnormal"):
            df = D.rank_scheduling(ds, variant)
            assert df is not None
            if variant == "queue_pi5" and ds == "UBnormal":
                assert df.empty  # the Pi 5 live run only covers ShanghaiTech clips


def test_diagnosis_cards_explain_each_failure_mode():
    from edgelab.app import diagnostics as D

    row = D.rank_small_person_misses("ShanghaiTech").iloc[0]
    ctx = D.build_small_person_context("ShanghaiTech", row["clip"])
    d = D.diagnose_case("small_person", ctx)
    assert d.verdict == "succeeded" and d.headline and d.bullets and d.mechanism

    frag = D.rank_track_fragmentation("ShanghaiTech").iloc[0]
    d = D.diagnose_case("fragmentation", D.build_fragmentation_context("ShanghaiTech", frag["clip"]))
    assert d.verdict in ("succeeded", "failed") and d.bullets

    scorer = D.rank_scorer_limited("ShanghaiTech")
    silent = scorer[scorer["tracked_but_silent"]].iloc[0]
    d = D.diagnose_case("scorer_limited", D.build_scorer_context("ShanghaiTech", silent))
    assert d.verdict == "failed" and d.bullets

    pi5 = D.rank_scheduling("ShanghaiTech", "queue_pi5").iloc[0]
    d = D.diagnose_case("scheduling", D.build_scheduling_context("ShanghaiTech", "queue_pi5", pi5), "queue_pi5")
    assert d.verdict == "failed" and d.headline


def test_timeline_frame_and_chart_for_real_clip():
    from edgelab.app import diagnostics as D
    from edgelab.app.catalog import build_gt_events

    events = build_gt_events("ShanghaiTech", "01_0140")
    assert events
    df, missing, notes = D.build_timeline_frame("ShanghaiTech", "01_0140", D.DEFAULT_TIMELINE_SERIES)
    assert list(df.columns) == ["frame", "series", "value", "raw"]
    assert df["frame"].min() == 0 and df["frame"].max() == 240
    assert df["series"].nunique() >= 4
    assert "Pi 5 Live (Cache-C1+Guard)" in missing  # 01_0140 was not one of the 12 live clips
    assert set(df["series"]) <= set(notes)

    for mode in D.Y_MODES:
        d2, _, _ = D.build_timeline_frame("ShanghaiTech", "01_0140", D.DEFAULT_TIMELINE_SERIES, y_mode=mode)
        assert len(d2) == len(df) and np.isfinite(d2["value"].to_numpy(dtype=float)).all()

    spec = D.make_timeline_chart(df, events, cursor_frame=100).to_dict()
    assert "layer" in spec or "hconcat" in spec or "vconcat" in spec or "mark" in spec

    # the Pi 5 live series is present for a live clip
    live, miss_live, _ = D.build_timeline_frame("ShanghaiTech", "12_0149", ["pi5_guard", "pi5_fixed_k1"])
    assert not miss_live and live["series"].nunique() == 2


def test_streamlit_new_tabs_render_without_exceptions():
    from streamlit.testing.v1 import AppTest

    def explorer():
        from edgelab.app import streamlit_app as app

        app.render_dataset_explorer()

    def inspector():
        from edgelab.app import streamlit_app as app

        app.render_failure_inspector()

    for fn in (explorer, inspector):
        at = AppTest.from_function(fn, default_timeout=180).run()
        assert not at.exception, [e.value for e in at.exception]
