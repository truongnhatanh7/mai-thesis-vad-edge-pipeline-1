"""Guards replay re-indexing (processed frames numbered 0..K-1) and mapping scores back to real frames (hold-last)."""

import json

import numpy as np
import pytest

from edgelab.pose.delivered import delivered_variant_scores, hold_last, processed_normality
from edgelab.metrics.scores import frame_level_auc


def test_hold_last_holds_until_the_next_processed_frame():
    frame_map = np.array([0, 2, 5])
    held = hold_last(np.array([10.0, 20.0, 30.0]), frame_map, 8)
    assert held.tolist() == [10, 10, 20, 20, 20, 30, 30, 30]


def test_hold_last_before_the_first_processed_frame_is_inf():
    held = hold_last(np.array([1.0, 2.0]), np.array([3, 5]), 7)
    assert held.tolist() == [np.inf, np.inf, np.inf, 1, 1, 2, 2]


def test_processed_normality_places_end_frame_and_takes_least_normal_person():
    # two people; segments start at processed frames 0 and 1 (person 1), 1 (person 2). end frame = start + 23.
    metadata = np.array([[1, 7, 1, 0], [1, 7, 1, 1], [1, 7, 2, 1], [2, 9, 1, 0]])
    normality = np.array([-1.0, -2.0, -5.0, -99.0])  # the last row is another clip: must be ignored
    out = processed_normality(normality, metadata, scene=1, clip=7, n_processed=26, frame_offset=23)
    assert out[23] == -1.0 and out[24] == -5.0  # frame 24: min(-2, -5)
    assert np.isinf(out[:23]).all() and np.isinf(out[25])


def _gt_dir(tmp_path, n_frames=60, anomalous=range(40, 50)):
    gt = np.zeros(n_frames, dtype=int)
    gt[list(anomalous)] = 1
    np.save(tmp_path / "01_0001.npy", gt)
    return tmp_path


def test_delivered_scores_hold_and_stay_on_the_real_timeline(tmp_path):
    gt_dir = _gt_dir(tmp_path)
    frame_map = np.arange(0, 60, 2)  # every second real frame processed: 30 processed frames
    # one person, segments starting at processed 0..5, normality = start index; end frame = processed 23..28
    metadata = np.array([[1, 1, 1, s] for s in range(6)])
    normality = np.arange(6, dtype=float)
    names, labels, scores = delivered_variant_scores(normality, metadata, {"01_0001": frame_map}, gt_dir,
                                                     smoother_name="none")
    assert names == ["01_0001"] and len(scores[0]) == 60 == len(labels[0])
    # processed frame 23 -> real frames 46, 47 (held); score = -normality = -0
    assert scores[0][46] == scores[0][47] == -0.0
    assert scores[0][48] == scores[0][49] == -1.0  # processed 24 -> real 48, 49
    assert scores[0][0] == -5.0 == scores[0][45]  # nobody / no output yet: the most normal score (= -max normality)


def test_delivered_live_smoothing_is_causal(tmp_path):
    gt_dir = _gt_dir(tmp_path)
    frame_map = np.arange(0, 60, 2)
    metadata = np.array([[1, 1, 1, s] for s in range(6)])
    base = np.arange(6, dtype=float)
    changed = base.copy()
    changed[4] = -50.0  # segment 4 is delivered at processed 27 -> real frame 54
    _, _, a = delivered_variant_scores(base, metadata, {"01_0001": frame_map}, gt_dir)
    _, _, b = delivered_variant_scores(changed, metadata, {"01_0001": frame_map}, gt_dir)
    np.testing.assert_allclose(a[0][:54], b[0][:54])
    assert not np.allclose(a[0][54:], b[0][54:])


def test_delivered_auc_is_perfect_for_a_perfect_detector_with_nothing_dropped(tmp_path):
    gt_dir = _gt_dir(tmp_path, n_frames=80, anomalous=range(60, 80))
    metadata = np.array([[1, 1, 1, s] for s in range(57)])  # end frames 23..79: every frame from 23 on has a score
    normality = np.where(np.arange(57) + 23 >= 60, -10.0, 0.0)  # anomalous from frame 60 on
    _, labels, scores = delivered_variant_scores(normality, metadata, {"01_0001": np.arange(80)}, gt_dir,
                                                 smoother_name="none")
    assert frame_level_auc(labels, scores) == 1.0


def test_no_scored_segment_is_an_error(tmp_path):
    gt_dir = _gt_dir(tmp_path)
    with pytest.raises(ValueError):
        delivered_variant_scores(np.array([]), np.zeros((0, 4), int), {"01_0001": np.arange(60)}, gt_dir)


# ---- replay with reindex (needs the Ultralytics tracker) -------------------------------------------------------

def _fake_cache(path, n_frames=12):
    from edgelab.pose.detcache import save_atomic
    boxes = [[4 * t, 50, 4 * t + 40, 150] for t in range(n_frames)]
    arrays = {"boxes": np.array(boxes, np.float32), "conf": np.full(n_frames, 0.9, np.float32),
              "cls": np.zeros(n_frames, np.float32), "keypoints": np.ones((n_frames, 17, 3), np.float32),
              "counts": np.ones(n_frames, int), "frame_ids": np.arange(n_frames), "meta": np.array("{}")}
    save_atomic(path, arrays)
    return path


def test_replay_reindex_numbers_processed_frames_consecutively_and_returns_the_map(tmp_path):
    pytest.importorskip("ultralytics")
    from edgelab.pose.detcache import load_detections
    from edgelab.pose.replay import replay_video
    frames, data = load_detections(_fake_cache(tmp_path / "v.npz"))
    keep = np.ones(12, bool)
    keep[[1, 2, 3, 7]] = False
    poses, stats = replay_video(frames, data["frame_ids"], c=0.25, keep_mask=keep, reindex=True)
    assert sorted(p.frame for p in poses) == list(range(8))
    assert stats["frame_map"] == [0, 4, 5, 6, 8, 9, 10, 11]
    plain, plain_stats = replay_video(frames, data["frame_ids"], c=0.25, keep_mask=keep)  # default is unchanged
    assert sorted(p.frame for p in plain) == [0, 4, 5, 6, 8, 9, 10, 11] and "frame_map" not in plain_stats


def test_replay_dir_reindex_writes_consecutive_json_frames_and_frame_maps(tmp_path):
    pytest.importorskip("ultralytics")
    from edgelab.pose.replay import FRAME_MAPS_FILE, replay_dir
    cache = tmp_path / "L2_E1"
    _fake_cache(cache / "01_0014.npz")
    keep = np.arange(12) % 3 == 0  # frames 0, 3, 6, 9
    stats = replay_dir(cache, 0.25, tmp_path / "poses", keep_masks={"01_0014": keep}, reindex=True)
    assert "frame_map" not in stats["01_0014"]  # kept out of replay_stats.json
    (written,) = (tmp_path / "poses").glob("*_tracked_person.json")
    clip = json.loads(written.read_text())
    assert set(next(iter(clip.values()))) == {"0000", "0001", "0002", "0003"}
    assert np.load(tmp_path / "poses" / FRAME_MAPS_FILE)["01_0014"].tolist() == [0, 3, 6, 9]
