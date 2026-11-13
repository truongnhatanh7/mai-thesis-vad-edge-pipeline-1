"""EXP-009 helpers: keypoint boxes, IoU matching of released poses to detector boxes, keyframe masks, coverage bootstrap."""

import numpy as np
import pytest

from edgelab.analysis.exp003 import Scores
from edgelab.analysis.exp009 import iou_with_boxes, keep_matched, keyframe_masks, keypoint_box, paired_coverage_diff


def person(x1, y1, x2, y2):
    """17 keypoints: two corners with conf 1, the rest conf 0 (not visible) at the origin."""
    keypoints = np.zeros((17, 3))
    keypoints[0] = [x1, y1, 1.0]
    keypoints[1] = [x2, y2, 0.5]
    return {"keypoints": keypoints.reshape(-1).tolist(), "scores": 1.0}


def test_keypoint_box_ignores_keypoints_with_zero_confidence():
    keypoints = np.array(person(10, 20, 30, 60)["keypoints"]).reshape(17, 3)
    assert keypoint_box(keypoints).tolist() == [10, 20, 30, 60]  # the conf-0 points at (0, 0) do not stretch the box
    assert keypoint_box(np.zeros((17, 3))) is None


def test_iou_values():
    box = np.array([0.0, 0, 10, 10])
    boxes = np.array([[0.0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30]])
    assert iou_with_boxes(box, boxes) == pytest.approx([1.0, 50 / 150, 0.0])
    assert iou_with_boxes(box, np.zeros((0, 4))).shape == (0,)


def test_keep_matched_needs_a_box_on_the_same_frame_with_iou_at_least_threshold():
    clip = {"1": {"0000": person(0, 0, 10, 10), "0001": person(0, 0, 10, 10), "0002": person(0, 0, 10, 10)},
            "2": {"0000": person(100, 100, 110, 110)}}
    boxes = {0: np.array([[0.0, 0, 10, 10]]),            # frame 0: matches person 1 only
             1: np.array([[0.0, 0, 10, 20]]),            # frame 1: IoU 0.5 -> kept
             2: np.array([[0.0, 0, 10, 40]])}            # frame 2: IoU 0.25 < 0.3 -> dropped
    kept, n_kept, n_total = keep_matched(clip, boxes, iou_min=0.3)
    assert (n_kept, n_total) == (2, 4)
    assert sorted(kept) == ["1"] and sorted(kept["1"]) == ["0000", "0001"]  # track 2 lost all poses: removed


def test_keep_matched_shift_compares_with_the_neighbouring_detector_frame():
    clip = {"1": {"0005": person(0, 0, 10, 10)}}
    boxes = {6: np.array([[0.0, 0, 10, 10]])}
    assert keep_matched(clip, boxes)[1] == 0
    assert keep_matched(clip, boxes, shift=1)[1] == 1


def test_keyframe_mask_is_every_kth_frame_starting_at_frame_zero():
    masks = keyframe_masks({"a": 7}, 3)
    assert masks["a"].tolist() == [True, False, False, True, False, False, True]


def test_paired_coverage_diff_on_a_toy_pair():
    labels = [np.array([0, 0, 1, 1]), np.array([0, 1])]
    a = Scores(["x", "y"], labels, [np.zeros(4), np.zeros(2)], [np.array([1, 0, 0, 0], bool), np.array([1, 0], bool)])
    b = Scores(["x", "y"], labels, [np.zeros(4), np.zeros(2)], [np.array([1, 1, 1, 0], bool), np.array([1, 1], bool)])
    out = paired_coverage_diff(a, b, n_resamples=50)
    assert out["anomalous"]["a"] == 0 and out["anomalous"]["b"] == pytest.approx(2 / 3)
    assert out["normal"]["a"] == pytest.approx(2 / 3) and out["normal"]["b"] == 1.0
