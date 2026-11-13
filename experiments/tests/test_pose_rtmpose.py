"""RTMPose crop / decode helpers and the EXP-015 oracle re-estimation, on tiny hand-made inputs (no model files needed)."""

import numpy as np

from edgelab.pose.oracle_check import LEFT_RIGHT_SWAP, joint_distance_report
from edgelab.pose.oracle_rtmpose import iou, person_frames, rerun_clip
from edgelab.pose.rtmpose import (INPUT_H, INPUT_W, MEAN, STD, box_to_center_scale, crop_matrix, crop_person,
                                  crop_to_image, decode_simcc, keypoints_to_box, to_network_input)


def test_wide_box_is_made_taller_to_the_192_by_256_aspect():
    center, scale = box_to_center_scale([0, 0, 200, 100])  # 200 wide, 100 high
    assert np.allclose(center, [100, 50])
    assert np.allclose(scale, [200 * 1.25, 200 / (192 / 256) * 1.25])  # height from width, then x1.25


def test_tall_box_is_made_wider():
    _, scale = box_to_center_scale([0, 0, 30, 120])
    assert np.allclose(scale, [120 * 192 / 256 * 1.25, 120 * 1.25])


def test_crop_matrix_sends_the_scaled_box_corners_to_the_crop_corners():
    center, scale = np.array([300.0, 200.0]), np.array([96.0, 128.0])
    matrix = crop_matrix(center, scale)
    top_left = matrix @ [300 - 48, 200 - 64, 1]
    bottom_right = matrix @ [300 + 48, 200 + 64, 1]
    assert np.allclose(top_left, [0, 0]) and np.allclose(bottom_right, [INPUT_W, INPUT_H])


def test_crop_person_puts_a_marked_pixel_where_the_matrix_says():
    img = np.zeros((300, 400, 3), np.uint8)
    img[150, 210] = (255, 255, 255)  # y, x
    center, scale = np.array([200.0, 150.0]), np.array([192.0, 256.0])  # scale = crop size: 1 image pixel = 1 crop pixel
    crop = crop_person(img, center, scale)
    # crop origin is image (200-96, 150-128) = (104, 22), so the marked pixel lands at crop (x, y) = (106, 128)
    assert crop.shape == (INPUT_H, INPUT_W, 3)
    assert crop[128, 106].min() == 255


def test_crop_outside_the_image_is_black():
    img = np.full((50, 50, 3), 200, np.uint8)
    crop = crop_person(img, np.array([0.0, 0.0]), np.array([192.0, 256.0]))
    assert crop[0, 0].max() == 0  # top-left of the crop is left of / above the image
    assert crop[128, 96].min() == 200  # the image centre of the crop sits on the image corner region that exists


def test_crop_to_image_undoes_crop_matrix():
    center, scale = np.array([321.5, 88.0]), np.array([77.0, 102.7])
    crop_xy = np.array([[[10.0, 20.0], [150.0, 240.0]]])  # one person, two points, crop pixels
    image_xy = crop_to_image(crop_xy, center[None], scale[None])[0]
    matrix = crop_matrix(center, scale)
    back = np.array([matrix @ [x, y, 1] for x, y in image_xy])
    assert np.allclose(back, crop_xy[0], atol=1e-4)


def test_decode_simcc_reads_argmax_over_two_and_takes_the_smaller_peak():
    simcc_x, simcc_y = np.zeros((1, 17, 384)), np.zeros((1, 17, 512))
    simcc_x[0, 0, 100], simcc_y[0, 0, 300] = 0.9, 0.6
    keypoints, scores = decode_simcc(simcc_x, simcc_y)
    assert np.allclose(keypoints[0, 0], [50.0, 150.0])  # split ratio 2: index / 2
    assert np.isclose(scores[0, 0], 0.6)


def test_to_network_input_normalises_with_mean_and_std_in_rgb_order():
    crop = np.zeros((INPUT_H, INPUT_W, 3), np.uint8)
    crop[:, :] = (10, 20, 30)  # BGR
    out = to_network_input([crop])
    assert out.shape == (1, 3, INPUT_H, INPUT_W) and out.dtype == np.float32
    assert np.allclose(out[0, 0, 0, 0], (30 - MEAN[0]) / STD[0])  # channel 0 is red = the third BGR value
    assert np.allclose(out[0, 2, 0, 0], (10 - MEAN[2]) / STD[2])


def test_keypoints_to_box_ignores_zero_confidence_points():
    kp = np.zeros((17, 3))
    kp[:, 2] = 0.5
    kp[:, :2] = [[10, 20]] * 17
    kp[3] = [50, 80, 0.7]
    kp[4] = [999, 999, 0.0]  # not seen: must not stretch the box
    assert np.allclose(keypoints_to_box(kp), [10, 20, 50, 80])
    assert keypoints_to_box(np.zeros((17, 3))) is None


def test_iou():
    assert np.isclose(iou(np.array([0, 0, 10, 10.0]), np.array([[5, 0, 15, 10.0]]))[0], 50 / 150)
    assert iou(np.array([0, 0, 10, 10.0]), np.array([[20, 20, 30, 30.0]]))[0] == 0


# ---- oracle re-estimation on a made-up clip ----

def _entry(x0: float, score: float = 2.5) -> dict:
    keypoints = [[x0 + i, 50 + 2 * i, 0.8] for i in range(17)]
    return {"keypoints": [v for kp in keypoints for v in kp], "scores": score}


class ShiftEstimator:
    """Pretends to be a pose model: returns the box centre for every joint, conf 0.3; NaN for boxes whose x1 < 0 is not used."""

    def __init__(self):
        self.calls = []

    def predict(self, img, boxes):
        self.calls.append(np.array(boxes))
        out = np.zeros((len(boxes), 17, 3), np.float32)
        out[:, :, 0] = ((boxes[:, 0] + boxes[:, 2]) / 2)[:, None]
        out[:, :, 1] = ((boxes[:, 1] + boxes[:, 3]) / 2)[:, None]
        out[:, :, 2] = 0.3
        return out


def test_person_frames_accepts_dict_and_list_form():
    clip = {"1": {"0000": _entry(0)}, "2": [{"0003": _entry(5)}, {"0004": _entry(6)}]}
    assert sorted((p, k) for p, k, _ in person_frames(clip)) == [("1", "0000"), ("2", "0003"), ("2", "0004")]


def test_rerun_clip_keeps_ids_frame_keys_and_scores_and_replaces_only_keypoints(monkeypatch):
    import edgelab.pose.oracle_rtmpose as module

    monkeypatch.setattr(module.cv2, "imread", lambda path: np.zeros((200, 200, 3), np.uint8))
    clip = {"7": {"0000": _entry(10, 2.5), "068": _entry(12, 3.5)}, "8": [{"0000": _entry(100, 1.5)}]}
    frames = {0: "f0.jpg", 68: "f68.jpg"}  # '068' is a 3-digit key like in the released 08_0044
    new, stats = rerun_clip(clip, frames, ShiftEstimator())
    assert set(new) == {"7", "8"} and set(new["7"]) == {"0000", "068"}
    assert new["7"]["068"]["scores"] == 3.5 and new["8"]["0000"]["scores"] == 1.5
    assert new["7"]["0000"]["keypoints"][2] == np.float32(0.3)  # conf replaced by the estimator's
    assert np.isclose(new["7"]["0000"]["keypoints"][0], 10 + 8)  # box x centre of x = 10..26
    assert stats["person_frames"] == 3 and stats["no_box"] == 0 and stats["dropped"] == 0


def test_rerun_clip_keeps_released_pose_when_the_box_is_degenerate_and_drops_nan_rows(monkeypatch):
    import edgelab.pose.oracle_rtmpose as module

    monkeypatch.setattr(module.cv2, "imread", lambda path: np.zeros((200, 200, 3), np.uint8))
    flat = {"keypoints": [5.0, 5.0, 0.9] * 17, "scores": 1.0}  # all joints on one point: zero-size box
    clip = {"1": {"0000": flat}, "2": {"0000": _entry(50)}}

    class DropEverything:
        def predict(self, img, boxes):
            return np.full((len(boxes), 17, 3), np.nan, np.float32)

    new, stats = rerun_clip(clip, {0: "f.jpg"}, DropEverything())
    assert stats["no_box"] == 1 and stats["dropped"] == 1
    assert new["1"]["0000"]["keypoints"] == flat["keypoints"]  # degenerate box: released pose kept and counted
    assert "2" not in new  # the estimator found nobody: that person-frame is gone


# ---- keypoint-order check ----

def test_joint_distance_report_prefers_the_matching_order():
    rng = np.random.default_rng(0)
    ref = np.concatenate([rng.uniform(0, 100, (17, 2)), np.ones((17, 1))], axis=1)
    ref[:, 1] = np.linspace(0, 100, 17)  # a box 100 high
    same = ref + 0.0
    assert joint_distance_report([(0, ref, same)])["order_ok"]
    swapped = ref[LEFT_RIGHT_SWAP]
    assert not joint_distance_report([(0, ref, swapped)])["order_ok"]
