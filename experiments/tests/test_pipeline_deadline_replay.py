"""End-to-end check of the deadline replay on a tiny synthetic dataset (2 videos, one walking person each)."""

import json

import numpy as np
import pandas as pd
import pytest

from edgelab.pose.evaluate import CHECKPOINT

pytest.importorskip("ultralytics", reason="needs `uv sync --extra pose`")

FPS = 24.0


def _video(cache, gt_dir, name, n_frames, seed):
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
    gt = np.zeros(n_frames, int)
    gt[n_frames // 2: n_frames // 2 + 20] = 1
    np.save(gt_dir / f"{name}.npy", gt)


def test_load_trace_cuts_each_video_to_whole_blocks(tmp_path):
    from edgelab.pipeline.deadline_replay import load_trace
    pd.DataFrame({"video": ["a"] * 50 + ["b"] * 30, "service_ms": np.arange(80.0)}).to_csv(tmp_path / "t.csv", index=False)
    trace = load_trace([tmp_path / "t.csv"])
    assert len(trace) == 48 + 24 and trace[47] == 47 and trace[48] == 50  # a: 0..47, b: 50..73


@pytest.mark.skipif(not CHECKPOINT.exists(), reason="STG-NF checkpoint missing")
def test_run_cell_end_to_end_on_synthetic_videos(tmp_path):
    from edgelab.pipeline.deadline_replay import run_cell
    cache, gt_dir = tmp_path / "cache" / "L0_E0", tmp_path / "gt"
    cache.mkdir(parents=True), gt_dir.mkdir()
    _video(cache, gt_dir, "01_0001", 150, 1)
    _video(cache, gt_dir, "01_0002", 150, 2)
    trace = np.full(48, 2.5 * 1000 / FPS)  # every frame costs 2.5 camera periods: keep 0, 2, 5, 7, 10, ...
    result = run_cell(cache, trace, 0.25, tmp_path / "out", {"label": "test"}, n_resamples=10, gt_dir=gt_dir,
                      tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}")
    assert result["drop_rate"] == pytest.approx(0.6, abs=0.02)
    assert result["num_videos"] == 2 and 0.0 <= result["auc"] <= 1.0
    # processed frames were re-indexed: 60 processed frames per video -> 60 - 24 = 36 offline windows each
    assert result["num_segments"] == 2 * (60 - 24)
    scores = np.load(tmp_path / "out" / "delivered_scores_live.npz")
    assert len(scores["score_01_0001"]) == len(scores["label_01_0001"]) == 150  # one score per REAL frame
    assert json.loads((tmp_path / "out" / "result.json").read_text())["config"]["label"] == "test"
