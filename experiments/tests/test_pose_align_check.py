"""Alignment check helpers: box from keypoints, IoU, best match."""

import numpy as np

from edgelab.pose.align_check import best_ious, iou, pose_box


def test_iou_of_half_overlapping_boxes_is_one_third():
    a, b = np.array([0, 0, 10, 10.0]), np.array([5, 0, 15, 10.0])  # intersection 50, union 150
    assert abs(iou(a, b) - 1 / 3) < 1e-9
    assert iou(a, np.array([20, 20, 30, 30.0])) == 0.0


def test_pose_box_uses_only_confident_keypoints():
    kp = np.tile([999.0, 999.0, 0.1], (17, 1))
    kp[0], kp[1], kp[2] = [10, 20, 0.9], [30, 20, 0.9], [10, 60, 0.9]
    assert pose_box(kp).tolist() == [10, 20, 30, 60]
    kp[2, 2] = 0.1
    assert pose_box(kp) is None  # only 2 visible keypoints


def test_best_ious_picks_the_best_detection_and_zero_without_detections():
    released = [np.array([0, 0, 10, 10.0])]
    detected = np.array([[50, 50, 60, 60.0], [0, 0, 10, 10.0]])
    assert best_ious(released, detected) == [1.0]
    assert best_ious(released, np.zeros((0, 4))) == [0.0]
