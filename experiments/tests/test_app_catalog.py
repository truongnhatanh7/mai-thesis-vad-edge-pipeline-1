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

