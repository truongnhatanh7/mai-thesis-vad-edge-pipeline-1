"""Guards our own YOLO-pose pre/post-processing for ONNX Runtime (letterbox, decode, conf filter, NMS) with synthetic data."""

import numpy as np

from edgelab.pipeline.ort_pose import decode, letterbox


def test_letterbox_856x480_geometry_and_values():
    image = np.full((480, 856, 3), (10, 20, 30), np.uint8)  # BGR
    blob, ratio, (left, top) = letterbox(image)
    assert blob.shape == (1, 3, 640, 640) and blob.dtype == np.float32
    assert ratio == 640 / 856 and (left, top) == (0, 140)  # resized to 640x359, padded 140 above and 141 below
    assert blob[0, 0, 320, 320] == np.float32(30 / 255)  # channel 0 is red: RGB order
    assert blob[0, 2, 320, 320] == np.float32(10 / 255)
    assert blob[0, 0, 10, 320] == np.float32(114 / 255) and blob[0, 0, 635, 320] == np.float32(114 / 255)  # grey padding
    assert blob[0, 0, 140, 320] == np.float32(30 / 255) and blob[0, 0, 139, 320] == np.float32(114 / 255)


def _output(anchors):
    """anchors: list of (cx, cy, w, h, conf, keypoints (17, 3)) in letterbox pixels -> (1, 56, n)."""
    rows = [np.concatenate([[cx, cy, w, h, conf], kp.reshape(-1)]) for cx, cy, w, h, conf, kp in anchors]
    return np.array(rows, dtype=np.float32).T[None]


def _lb(x, y, ratio=640 / 856, pad=(0, 140)):
    return x * ratio + pad[0], y * ratio + pad[1]


def test_decode_maps_boxes_and_keypoints_back_to_original_pixels():
    _, ratio, pad = letterbox(np.zeros((480, 856, 3), np.uint8))
    x1, y1 = _lb(100, 50)
    x2, y2 = _lb(200, 250)
    kp = np.zeros((17, 3), np.float32)
    kp[:, :2] = _lb(150, 100)
    kp[:, 2] = 0.8
    out = _output([((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1, 0.9, kp)])
    boxes, conf, keypoints = decode(out, 0.25, ratio, pad, (480, 856))
    np.testing.assert_allclose(boxes[0], [100, 50, 200, 250], atol=1e-3)
    np.testing.assert_allclose(keypoints[0, 0], [150, 100, 0.8], atol=1e-3)
    assert conf.tolist() == [np.float32(0.9)]


def test_decode_drops_low_confidence_and_suppresses_overlapping_boxes_sorted_by_confidence():
    kp = np.zeros((17, 3), np.float32)
    anchors = [(100, 300, 50, 100, 0.6, kp),  # overlaps the next one almost completely -> suppressed
               (102, 300, 50, 100, 0.9, kp),  # kept (highest of its pair)
               (400, 300, 50, 100, 0.7, kp),  # separate person, kept
               (500, 300, 50, 100, 0.2, kp)]  # below the gate
    boxes, conf, _ = decode(_output(anchors), 0.25, 1.0, (0, 0), (640, 640))
    assert np.allclose(conf, [0.9, 0.7]) and np.allclose(boxes[:, 0], [77, 375])


def test_decode_with_nothing_above_the_gate_returns_empty_arrays():
    kp = np.zeros((17, 3), np.float32)
    boxes, conf, keypoints = decode(_output([(100, 100, 20, 40, 0.1, kp)]), 0.25, 1.0, (0, 0), (640, 640))
    assert boxes.shape == (0, 4) and conf.shape == (0,) and keypoints.shape == (0, 17, 3)


def test_decode_clips_boxes_to_the_image():
    kp = np.zeros((17, 3), np.float32)
    boxes, _, _ = decode(_output([(10, 300, 100, 100, 0.9, kp)]), 0.25, 1.0, (0, 0), (480, 640))
    assert boxes[0].tolist() == [0.0, 250.0, 60.0, 350.0]


def test_letterbox_to_the_rectangles_ultralytics_uses_for_856x480():
    image = np.full((480, 856, 3), (10, 20, 30), np.uint8)
    # (R, shape, pad_top): hand-computed, e.g. R=480: image 480x269, 19 rows of padding split 9 above / 10 below
    for shape, top in [((288, 480), 9), ((384, 640), 12), ((480, 800), 15), ((544, 960), 3)]:
        blob, ratio, pad = letterbox(image, shape)
        assert blob.shape == (1, 3, *shape) and pad == (0, top) and ratio == shape[1] / 856
        assert blob[0, 0, top, 5] == np.float32(30 / 255) and blob[0, 0, top - 1, 5] == np.float32(114 / 255)
