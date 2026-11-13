"""EXP-017: decoding the NMS-free end-to-end YOLO26 output (1, 300, 57) and the layout switch shared with the YOLOv8 decoder."""

import numpy as np
import pytest

from edgelab.pipeline.ort_pose import decode, decode_any, decode_e2e


def e2e_output(rows: list[tuple[float, float, float, float, float]], n: int = 300) -> np.ndarray:
    out = np.zeros((1, n, 57), np.float32)
    for i, (x1, y1, x2, y2, conf) in enumerate(rows):
        out[0, i, :5] = (x1, y1, x2, y2, conf)
        out[0, i, 6:] = np.tile([x1 + 1, y1 + 2, 0.5], 17)
    return out


def test_e2e_decode_filters_by_conf_sorts_and_undoes_the_letterbox():
    # letterbox: ratio 0.5, pad (10, 20): original = (letterbox - pad) / 0.5
    out = e2e_output([(30, 40, 70, 140, 0.4), (110, 60, 150, 160, 0.9), (0, 0, 5, 5, 0.01)])
    boxes, conf, kpts = decode_e2e(out, 0.25, 0.5, (10, 20), (480, 856))
    assert conf.tolist() == pytest.approx([0.9, 0.4])  # sorted, low-conf row dropped
    assert boxes[0].tolist() == pytest.approx([200, 80, 280, 280])
    assert kpts.shape == (2, 17, 3) and kpts[0, 0, :2].tolist() == pytest.approx([202, 84])


def test_e2e_decode_clips_to_the_frame_and_needs_no_nms():
    # two heavily overlapping boxes both survive: the end-to-end head is trusted to have removed duplicates itself
    out = e2e_output([(0, 0, 100, 100, 0.8), (2, 2, 100, 100, 0.7), (-50, -50, 2000, 2000, 0.6)])
    boxes, conf, _ = decode_e2e(out, 0.25, 1.0, (0, 0), (480, 856))
    assert len(conf) == 3
    assert boxes[2].tolist() == [0, 0, 856, 480]


def test_decode_any_picks_the_decoder_by_output_layout():
    v8 = np.zeros((1, 56, 100), np.float32)
    v8[0, :5, 3] = (50, 60, 20, 40, 0.9)
    boxes, conf, _ = decode_any(v8, 0.25, 1.0, (0, 0), (480, 856))
    assert conf.tolist() == pytest.approx([0.9]) and boxes[0].tolist() == pytest.approx([40, 40, 60, 80])
    assert decode_any(e2e_output([(1, 2, 3, 4, 0.5)]), 0.25, 1.0, (0, 0), (480, 856))[1].tolist() == [0.5]
    with pytest.raises(ValueError):
        decode_any(np.zeros((1, 10, 11), np.float32), 0.25, 1.0, (0, 0), (480, 856))
    # decode (v8 path) is unchanged: same answer as through decode_any
    assert np.allclose(decode(v8, 0.25, 1.0, (0, 0), (480, 856))[0], boxes)
