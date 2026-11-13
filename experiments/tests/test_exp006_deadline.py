"""EXP-006: the deadline replay (delivered AUC) with dataset='UBnormal'; ShanghaiTech behaviour is guarded by
test_pose_delivered.py and test_pipeline_deadline_replay.py (unchanged)."""

import json

import numpy as np
import pytest

from edgelab.pose.delivered import delivered_variant_scores, processed_normality
from edgelab.pose.ubnormal import CHECKPOINT

A, B = "abnormal_scene_1_scenario_1", "normal_scene_2_scenario3_1"


def _save_gt(gt_dir, video, normal_flags):
    gt_dir.mkdir(exist_ok=True)
    with open(gt_dir / f"{video}_tracks.txt", "wb") as f:  # NumPy file with a .txt name, 1 = normal
        np.save(f, np.asarray(normal_flags, dtype=float))


def test_processed_normality_accepts_string_metadata():
    metadata = np.array([[1, "abnormal__1", 1, 0], [1, "abnormal__1", 2, 1], [2, "normal_3_1", 1, 0]])  # strings, as STG-NF
    normality = np.array([-1.0, -5.0, -99.0])
    out = processed_normality(normality, metadata, "1", "abnormal__1", n_processed=20, frame_offset=15)
    assert out[15] == -1.0 and out[16] == -5.0 and np.isinf(out[17:]).all()  # other clip ignored


def test_delivered_scores_for_ubnormal_use_segment_16_and_its_gt(tmp_path):
    gt_normal = np.ones(60)
    gt_normal[40:50] = 0
    _save_gt(tmp_path, A, gt_normal)
    frame_map = np.arange(0, 60, 2)  # every second real frame processed: 30 processed frames
    # one person, segments start at processed 0..5, normality = start; live score frame = start + 15 (not 23)
    metadata = np.array([[1, "abnormal__1", 1, s] for s in range(6)])
    names, labels, scores = delivered_variant_scores(np.arange(6, dtype=float), metadata, {A: frame_map}, tmp_path,
                                                     smoother_name="none", dataset="UBnormal", clips=[A])
    assert names == [A]
    assert labels[0].tolist() == [0] * 40 + [1] * 10 + [0] * 10  # flipped: 1 = anomalous
    assert scores[0][30] == scores[0][31] == -0.0  # processed 15 -> real 30, 31
    assert scores[0][32] == scores[0][33] == -1.0
    assert scores[0][0] == -5.0  # no output yet: most normal score


def _write_video(cache, gt_dir, name, n_frames, seed):
    from edgelab.pose.detcache import save_atomic
    rng = np.random.default_rng(seed)
    base = rng.uniform(-60, 60, (17, 2)) + [400, 240]
    walk = np.cumsum(rng.normal(0, 2, (n_frames, 1, 2)), axis=0)
    keypoints = np.concatenate([base[None] + walk, np.full((n_frames, 17, 1), 0.9)], axis=2)
    boxes = np.array([[300 + 2 * t, 150, 360 + 2 * t, 330] for t in range(n_frames)], np.float32)
    save_atomic(cache / f"{name}.npz", {"boxes": boxes, "conf": np.full(n_frames, 0.9, np.float32),
                                        "cls": np.zeros(n_frames, np.float32), "keypoints": keypoints.astype(np.float32),
                                        "counts": np.ones(n_frames, int), "frame_ids": np.arange(n_frames),
                                        "meta": np.array("{}")})
    gt_normal = np.ones(n_frames)
    if name.startswith("abnormal"):
        gt_normal[n_frames // 2: n_frames // 2 + 20] = 0
    _save_gt(gt_dir, name, gt_normal)


@pytest.fixture
def ub_cache(tmp_path):
    cache, gt_dir = tmp_path / "cache" / "L0_E0", tmp_path / "gt"
    cache.mkdir(parents=True)
    _write_video(cache, gt_dir, A, 150, 1)
    _write_video(cache, gt_dir, B, 150, 2)
    return cache, gt_dir


@pytest.mark.skipif(not CHECKPOINT.exists(), reason="UBnormal checkpoint missing")
def test_run_cell_for_ubnormal_end_to_end(tmp_path, ub_cache):
    pytest.importorskip("ultralytics", reason="needs `uv sync --extra pose`")
    from edgelab.pipeline.deadline_replay import run_cell
    cache, gt_dir = ub_cache
    trace = np.full(48, 2.5 * 1000 / 24.0)  # every frame costs 2.5 camera periods: 60 of 150 frames processed
    result = run_cell(cache, trace, 0.25, tmp_path / "out", {"label": "ub"}, n_resamples=10, gt_dir=gt_dir,
                      dataset="UBnormal", videos=[A, B], tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}")
    assert result["drop_rate"] == pytest.approx(0.6, abs=0.02)
    assert result["num_videos"] == 2
    assert result["num_segments"] == 2 * (60 - 16)  # segment length 16 on the re-indexed frames
    poses = sorted(p.name for p in (tmp_path / "out" / "poses").glob("*_tracked_person.json"))
    assert poses[0].startswith(f"{A}_alphapose_")  # STG-NF's UBnormal file-name rule
    scores = np.load(tmp_path / "out" / "delivered_scores_live.npz")
    assert len(scores[f"score_{A}"]) == len(scores[f"label_{A}"]) == 150
    config = json.loads((tmp_path / "out" / "result.json").read_text())["config"]
    assert config["dataset"] == "UBnormal"


@pytest.mark.skipif(not CHECKPOINT.exists(), reason="UBnormal checkpoint missing")
def test_zero_drop_delivered_equals_offline_live_auc(tmp_path, ub_cache):
    """EXP-004's check, for UBnormal: a trace that never drops a frame must give the offline live AUC."""
    pytest.importorskip("ultralytics", reason="needs `uv sync --extra pose`")
    from edgelab.analysis.exp006 import zero_drop_check
    cache, gt_dir = ub_cache
    result = zero_drop_check(cache, [A, B], tmp_path / "check", gt_dir=gt_dir)
    assert result["drop_rate"] == 0.0
    assert result["delivered_auc"] == result["offline_live_auc"]


def test_cache_names_and_default_cells():
    from edgelab.analysis.exp004 import parse_cell
    from edgelab.analysis.exp006 import cache_name, wanted
    assert [cache_name("E0", r) for r in (640, 480, 960)] == ["L0_E0", "L0-480_E0", "L0-960_E0"]
    assert cache_name("E2", 640) == "L0_E2"
    assert wanted(parse_cell("L3-E0-R960-c0.25-O0-B4"))
    assert wanted(parse_cell("L3-E2-R640-c0.25-O1n-B2")) and wanted(parse_cell("L3-E0-R640-c0.25-O0r-B2"))
    assert not wanted(parse_cell("L3-E0-R640-c0.25-O1-B4")) and not wanted(parse_cell("L3-E0-R640-c0.25-O0-B2"))
