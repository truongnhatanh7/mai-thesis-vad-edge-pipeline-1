"""Guards the live pipeline pieces: online STG-NF == offline STG-NF, segment rules, OS config parsing, subset choice, CSV."""

import csv
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from edgelab.pipeline.os_config import parse_cpus
from edgelab.pipeline.stgnf_live import LiveStgnf, is_continuous, load_model
from edgelab.pipeline.timing_subset import select_subset
from edgelab.pose.evaluate import CHECKPOINT

pytest.importorskip("ultralytics", reason="needs `uv sync --extra pose`")
YOLO_ONNX = Path("/Users/nanh/thesis-data/vm-exchange/models/yolov8n-pose-640-384x640.onnx")


def test_parse_cpus():
    assert parse_cpus("1,2,3") == {1, 2, 3} and parse_cpus("1-3") == {1, 2, 3} and parse_cpus("0,2-3") == {0, 2, 3}


def test_continuity_rule_allows_two_missing_slots():
    assert is_continuous(list(range(24)))
    assert is_continuous([f for f in range(26) if f not in (5, 9)])  # 24 poses over 26 slots: 2 missing, accepted
    assert not is_continuous([f for f in range(27) if f not in (5, 9, 13)])  # 3 missing
    assert not is_continuous(list(range(0, 48, 2)))  # every second frame


def test_select_subset_picks_two_per_scene_for_the_six_biggest_scenes_and_is_seeded():
    counts = {f"{s:02d}_{i:04d}": 300 for s, n in [(1, 9), (2, 8), (3, 7), (4, 6), (5, 5), (6, 4), (7, 3)] for i in range(n)}
    counts["01_0000"] = 100  # too short: never eligible
    subset = select_subset(counts)
    assert len(subset) == 12 and "01_0000" not in subset
    assert {v.split("_")[0] for v in subset} == {"01", "02", "03", "04", "05", "06"}  # scene 07 is not among the top 6
    assert subset == select_subset(counts) and subset != select_subset(counts, seed=1)


def test_select_subset_breaks_ties_by_lower_scene_id():
    counts = {f"{s:02d}_{i:04d}": 300 for s in (1, 2, 3, 4, 5, 6, 7) for i in range(3)}
    assert {v.split("_")[0] for v in select_subset(counts)} == {"01", "02", "03", "04", "05", "06"}


def _fake_track(n_frames, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.uniform(-60, 60, (17, 2)) + [400, 240]
    walk = np.cumsum(rng.normal(0, 3, (n_frames, 1, 2)), axis=0)
    xy = base[None] + walk + rng.normal(0, 1, (n_frames, 17, 2))
    return np.concatenate([xy, np.full((n_frames, 17, 1), 0.9)], axis=2)  # (n_frames, 17, 3)


@pytest.mark.skipif(not CHECKPOINT.exists(), reason="STG-NF checkpoint missing")
def test_live_stgnf_scores_equal_the_offline_scores(tmp_path):
    from edgelab.pose.stgnf_eval import score_segments
    track = _fake_track(30)
    clip = {"1": {f"{f:04d}": {"keypoints": track[f].reshape(-1).tolist(), "scores": 0.9} for f in range(30)}}
    (tmp_path / "01_0001_x_tracked_person.json").write_text(json.dumps(clip))
    offline = score_segments(tmp_path, CHECKPOINT, device="cpu").normality  # starts 0..5 (the loader skips the last window)

    live = LiveStgnf(load_model())
    scores = {}
    for frame in range(30):
        ids, normality = live.update(frame, {1: track[frame]})
        if ids:
            scores[frame - 23] = normality[0]  # segment start = frame - 23
    assert sorted(scores) == list(range(7))  # every complete window is scored: 7 windows in 30 frames
    np.testing.assert_allclose([scores[s] for s in range(6)], offline, rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(not CHECKPOINT.exists(), reason="STG-NF checkpoint missing")
def test_live_stgnf_batches_the_new_segments_of_all_tracks_in_one_frame():
    live = LiveStgnf(load_model())
    tracks = {1: _fake_track(30, seed=1), 2: _fake_track(30, seed=2)}
    for frame in range(30):
        ids, normality = live.update(frame, {t: k[frame] for t, k in tracks.items()})
        assert (len(ids) == 2) == (frame >= 23) and len(normality) == len(ids)


@pytest.mark.skipif(not (YOLO_ONNX.exists() and CHECKPOINT.exists()), reason="ONNX model or checkpoint missing")
def test_live_timing_writes_one_csv_row_per_frame_with_all_columns(tmp_path, monkeypatch):
    from edgelab.pipeline import live_timing
    video = tmp_path / "frames" / "01_0001"
    video.mkdir(parents=True)
    for i in range(4):
        cv2.imwrite(str(video / f"{i:03d}.jpg"), np.random.default_rng(i).integers(0, 255, (96, 128, 3), np.uint8))
    (tmp_path / "videos.txt").write_text("# comment\n01_0001\n")
    out = tmp_path / "run.csv"
    monkeypatch.setattr("sys.argv", ["live_timing", "--frames-root", str(tmp_path / "frames"), "--videos-file",
                                     str(tmp_path / "videos.txt"), "--level", "L2", "--enhance", "E1", "--c", "0.25",
                                     "--threads", "1", "--spin", "off", "--imgsz", "640", "--max-frames-per-video", "3", "--out", str(out), "--tag", "t"])
    live_timing.main()
    rows = list(csv.DictReader(out.open()))
    assert [r["frame"] for r in rows] == ["0", "1", "2"]  # --max-frames-per-video 3 of the 4 frames and list(rows[0]) == live_timing.COLUMNS
    assert rows[0]["imgsz"] == "640" and rows[0]["tag"] == "t" and rows[0]["spin"] == "off" and rows[0]["threads"] == "1"
    for r in rows:
        parts = sum(float(r[k]) for k in ("read_ms", "darken_ms", "enhance_ms", "det_pre_ms", "det_infer_ms",
                                          "det_post_ms", "track_ms", "flow_ms"))
        assert float(r["total_ms"]) >= parts - 1e-6  # total also covers small gaps between the timed parts
        assert float(r["service_ms"]) == pytest.approx(float(r["total_ms"]) - float(r["darken_ms"]))
    meta = json.loads(out.with_suffix(".meta.json").read_text())
    assert meta["videos"] == ["01_0001"] and meta["yolo_input_hw"] == [384, 640]
