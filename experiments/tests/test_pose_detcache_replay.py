"""Guards the detector cache file format and the replay step (cache -> gate threshold -> tracker -> STG-NF JSON)."""

import json

import cv2
import numpy as np
import pytest

from edgelab.pose.detcache import WEIGHTS, cache_video, device_check_passes, load_detections, save_atomic

pytest.importorskip("ultralytics", reason="needs `uv sync --extra pose`")


def _fake_cache(path, persons, n_frames=12):
    """persons: list of (x_start, conf, marker). Each person is a 40x100 box moving 4 px/frame.

    The keypoints of a person are all equal to its marker, so we can see which person a track got attached to.
    """
    boxes, conf, keypoints, counts = [], [], [], []
    for t in range(n_frames):
        for x0, c, marker in persons:
            x = x0 + 4 * t
            boxes.append([x, 50, x + 40, 150])
            conf.append(c)
            keypoints.append(np.full((17, 3), marker, np.float32))
        counts.append(len(persons))
    arrays = {"boxes": np.array(boxes, np.float32), "conf": np.array(conf, np.float32),
              "cls": np.zeros(len(conf), np.float32), "keypoints": np.stack(keypoints),
              "counts": np.array(counts), "frame_ids": np.arange(n_frames), "meta": np.array("{}")}
    save_atomic(path, arrays)
    return path


def test_load_detections_splits_flat_arrays_per_frame(tmp_path):
    path = _fake_cache(tmp_path / "v.npz", [(0, 0.9, 1.0), (300, 0.5, 2.0)], n_frames=3)
    frames, data = load_detections(path)
    assert len(frames) == 3 and all(len(f["conf"]) == 2 for f in frames)
    assert frames[1]["boxes"][0, 0] == 4 and frames[1]["keypoints"][1, 0, 0] == 2.0
    assert not list(tmp_path.glob("*.partial.npz"))  # atomic write cleaned up


@pytest.mark.skipif(not WEIGHTS.exists(), reason="YOLO weights missing")
def test_cache_video_writes_one_row_per_frame_with_timings(tmp_path):
    from ultralytics import YOLO
    frames_dir = tmp_path / "01_0001"
    frames_dir.mkdir()
    for i in range(3):
        cv2.imwrite(str(frames_dir / f"{i:03d}.jpg"), np.random.default_rng(i).integers(0, 255, (96, 128, 3), np.uint8))
    arrays = cache_video(YOLO(str(WEIGHTS)), frames_dir, "L2", "E1", "cpu")
    assert arrays["frame_ids"].tolist() == [0, 1, 2]
    assert arrays["counts"].sum() == len(arrays["conf"]) == len(arrays["boxes"]) == len(arrays["keypoints"])
    assert arrays["keypoints"].shape[1:] == (17, 3)
    for key in ("luma_dark", "luma_enhanced", "darken_ms", "enhance_ms", "yolo_infer_ms"):
        assert arrays[key].shape == (3,)
    assert json.loads(str(arrays["meta"]))["level"] == "L2"


def test_replay_gives_each_person_one_track_with_their_own_keypoints(tmp_path):
    from edgelab.pose.replay import replay_video
    path = _fake_cache(tmp_path / "v.npz", [(0, 0.9, 1.0), (300, 0.8, 2.0)])
    frames, data = load_detections(path)
    poses, stats = replay_video(frames, data["frame_ids"], c=0.25)
    assert stats["tracks"] == 2 and stats["frames_with_person"] == 12
    for track_id in {p.track_id for p in poses}:
        markers = {float(p.keypoints[0, 0]) for p in poses if p.track_id == track_id}
        assert len(markers) == 1  # a track never mixes two people's keypoints


def test_gate_threshold_decides_whether_a_weak_person_is_ever_tracked(tmp_path):
    from edgelab.pose.replay import replay_video
    path = _fake_cache(tmp_path / "v.npz", [(0, 0.9, 1.0), (300, 0.12, 2.0)])
    frames, data = load_detections(path)
    strict, _ = replay_video(frames, data["frame_ids"], c=0.25)
    loose, loose_stats = replay_video(frames, data["frame_ids"], c=0.10)
    assert {float(p.keypoints[0, 0]) for p in strict} == {1.0}
    assert {float(p.keypoints[0, 0]) for p in loose} == {1.0, 2.0}
    assert loose_stats["detections_kept"] == 24


def test_keep_mask_hides_frames_from_the_tracker(tmp_path):
    from edgelab.pose.replay import replay_video
    path = _fake_cache(tmp_path / "v.npz", [(0, 0.9, 1.0)])
    frames, data = load_detections(path)
    keep = np.ones(12, bool)
    keep[4:8] = False
    poses, stats = replay_video(frames, data["frame_ids"], c=0.25, keep_mask=keep)
    assert {p.frame for p in poses} == set(range(12)) - {4, 5, 6, 7}
    assert stats["frames_seen_by_tracker"] == 8 and stats["frames_total"] == 12


def test_replay_dir_writes_stgnf_named_json_files(tmp_path):
    from edgelab.pose.replay import replay_dir
    cache = tmp_path / "L2_E1"
    _fake_cache(cache / "01_0014.npz", [(0, 0.9, 1.0)])
    replay_dir(cache, 0.25, tmp_path / "poses")
    (written,) = (tmp_path / "poses").glob("*_tracked_person.json")
    assert written.name == "01_0014_L2-E1-c0.25-botsort_tracked_person.json"
    assert written.name.split("_")[:2] == ["01", "0014"]  # STG-NF reads scene and clip from here
    clip = json.loads(written.read_text())
    assert set(next(iter(clip.values()))) == {f"{i:04d}" for i in range(12)}


def test_bytetrack_option_runs(tmp_path):
    from edgelab.pose.replay import replay_video
    path = _fake_cache(tmp_path / "v.npz", [(0, 0.9, 1.0)])
    frames, data = load_detections(path)
    poses, stats = replay_video(frames, data["frame_ids"], c=0.25, tracker="bytetrack")
    assert stats["tracks"] == 1


def test_device_check_rule_from_plan():
    ok = {"frames": 100, "frames_equal_count": 99, "max_conf_diff": 0.01}
    assert device_check_passes([ok])
    assert not device_check_passes([{**ok, "frames_equal_count": 98}])  # counts differ on more than 1% of frames
    assert not device_check_passes([ok, {**ok, "max_conf_diff": 0.011}])  # one video exceeds the confidence tolerance
