"""NcnnPose against a tiny hand-written NCNN model (one Reshape layer), so no export or weights are needed."""

import numpy as np
import pytest

from edgelab.pipeline.live_timing import make_detector
from edgelab.pipeline.ncnn_pose import NcnnPose

pytest.importorskip("ncnn", reason="needs the ncnn package")

H, W = 28, 32  # 3 * 28 * 32 = 2688 = 56 * 48


@pytest.fixture
def tiny_model(tmp_path):
    """Input in0 (3, 28, 32) -> Reshape -> out0 (56 rows, 48 columns): the same layout as YOLOv8-pose's (56, N) output."""
    (tmp_path / "model.ncnn.param").write_text("7767517\n2 2\nInput in0 0 1 in0\nReshape r 1 1 in0 out0 0=48 1=56 2=-233\n")
    (tmp_path / "model.ncnn.bin").write_bytes(b"")
    (tmp_path / "metadata.yaml").write_text(f"imgsz:\n- {H}\n- {W}\n")
    return tmp_path


def test_input_shape_is_read_from_metadata(tiny_model):
    assert NcnnPose(tiny_model).shape == (H, W)


def test_raw_output_has_the_onnx_layout(tiny_model):
    blob = np.random.default_rng(0).random((1, 3, H, W), dtype=np.float32)
    out = NcnnPose(tiny_model, threads=1, fp16=False).raw(blob)
    assert out.shape == (1, 56, 48)
    np.testing.assert_array_equal(out[0], blob.reshape(56, 48))


def test_call_returns_the_ortpose_tuple_and_conf_filter_applies(tiny_model):
    pose = NcnnPose(tiny_model, threads=1, fp16=False)
    image = np.full((14, 16, 3), 200, np.uint8)
    boxes, conf, keypoints, timing = pose(image, 2.0)  # row 4 is a pixel value in [0, 1]: nothing reaches 2.0
    assert len(conf) == 0 and boxes.shape == (0, 4) and keypoints.shape == (0, 17, 3)
    assert set(timing) == {"pre_ms", "infer_ms", "post_ms"}


def test_make_detector_picks_ncnn_for_a_folder(tiny_model):
    assert isinstance(make_detector(tiny_model, 1, False, ncnn_fp16=False), NcnnPose)
