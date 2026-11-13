"""INT8 quantization helpers (calibration frames, head-node selection, a real tiny QDQ run) and the parity comparison."""

import cv2
import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper

from edgelab.pipeline.parity_variants import compare, iou
from edgelab.pipeline.quantize_int8 import calibration_frames, decode_node_names, quantize, read_frame


def _write_video(path, n_frames=10, size=(48, 32)):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 24, size)
    for i in range(n_frames):
        writer.write(np.full((size[1], size[0], 3), 10 * i, np.uint8))
    writer.release()


@pytest.fixture
def videos(tmp_path):
    for i in range(6):
        _write_video(tmp_path / f"{i:02d}_001.avi")
    return tmp_path


def test_calibration_frames_are_distinct_videos_and_deterministic(videos):
    a, b = calibration_frames(videos, 4, seed=0), calibration_frames(videos, 4, seed=0)
    assert a == b and len({v for v, _ in a}) == 4
    assert all(0 <= i < 10 for _, i in a)
    assert calibration_frames(videos, 4, seed=1) != a


def test_calibration_frames_refuses_when_there_are_too_few_videos(videos):
    with pytest.raises(ValueError):
        calibration_frames(videos, 7)


def test_read_frame_returns_the_asked_frame(videos):
    frame = read_frame(videos, "00_001.avi", 5)
    assert frame.shape == (32, 48, 3) and abs(int(frame.mean()) - 50) <= 3  # MJPG is lossy


def _tiny_model(path):
    """images (1,3,32,32) -> Conv (named like a head conv) -> Mul by 2 (named like a head decode op) -> output0."""
    weight = helper.make_tensor("w", TensorProto.FLOAT, [4, 3, 3, 3], np.random.default_rng(0).normal(size=108).astype(np.float32))
    two = helper.make_tensor("two", TensorProto.FLOAT, [1], [2.0])
    nodes = [helper.make_node("Conv", ["images", "w"], ["c"], name="/model.22/cv2.0/conv", pads=[1, 1, 1, 1]),
             helper.make_node("Mul", ["c", "two"], ["output0"], name="/model.22/Mul_3")]
    graph = helper.make_graph(nodes, "tiny", [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 32, 32])],
                              [helper.make_tensor_value_info("output0", TensorProto.FLOAT, [1, 4, 32, 32])], [weight, two])
    # ir_version 10: onnx 1.23's default IR (14) is newer than the installed ONNX Runtime accepts (max 13); Ultralytics exports are older
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=10)
    onnx.save(model, str(path))
    return model


def test_decode_node_names_keeps_convs_and_inner_blocks_out(tmp_path):
    model = _tiny_model(tmp_path / "tiny.onnx")
    assert decode_node_names(model) == ["/model.22/Mul_3"]


def test_quantize_writes_a_qdq_model_that_runs(tmp_path):
    import onnxruntime as ort

    _tiny_model(tmp_path / "tiny.onnx")
    rng = np.random.default_rng(1)
    frames = [rng.integers(0, 255, (24, 40, 3), dtype=np.uint8) for _ in range(4)]
    summary = quantize(tmp_path / "tiny.onnx", tmp_path / "tiny-int8.onnx", frames, decode="fp32")
    assert summary["input_hw"] == (32, 32) and summary["excluded_nodes"] == 1 and summary["qdq_nodes"] > 0
    types = {n.op_type for n in onnx.load(str(tmp_path / "tiny-int8.onnx")).graph.node}
    assert {"QuantizeLinear", "DequantizeLinear", "Conv"} <= types
    out = ort.InferenceSession(str(tmp_path / "tiny-int8.onnx"), providers=["CPUExecutionProvider"]).run(
        None, {"images": rng.random((1, 3, 32, 32), dtype=np.float32)})[0]
    assert out.shape == (1, 4, 32, 32)


def _det(box, conf, kpt_xy, kpt_conf=0.9):
    keypoints = np.zeros((1, 17, 3), np.float32)
    keypoints[0, :, :2], keypoints[0, :, 2] = kpt_xy, kpt_conf
    return np.array([box], np.float32), np.array([conf], np.float32), keypoints


def test_iou():
    assert iou(np.array([0, 0, 10, 10]), np.array([0, 0, 10, 10])) == 1.0
    assert iou(np.array([0, 0, 10, 10]), np.array([5, 0, 15, 10])) == pytest.approx(1 / 3)
    assert iou(np.array([0, 0, 1, 1]), np.array([5, 5, 6, 6])) == 0.0


def test_compare_reports_conf_and_keypoint_differences():
    ref = _det([0, 0, 10, 20], 0.90, [5, 5])
    other = _det([0, 0, 10, 20], 0.85, [8, 9])  # keypoints moved by (3, 4) = 5 px
    row = compare(ref, other)
    assert (row["n_ref"], row["n_variant"], row["n_matched"]) == (1, 1, 1)
    assert row["max_conf_diff"] == pytest.approx(0.05, abs=1e-6) and row["max_kpt_px"] == pytest.approx(5.0)


def test_compare_counts_unmatched_and_extra_boxes():
    ref = _det([0, 0, 10, 20], 0.9, [5, 5])
    far = _det([100, 100, 110, 120], 0.9, [105, 105])
    row = compare(ref, far)
    assert (row["n_ref"], row["n_variant"], row["n_matched"]) == (1, 1, 0) and row["max_conf_diff"] == 0.0


def test_compare_ignores_keypoints_the_reference_does_not_see():
    ref = _det([0, 0, 10, 20], 0.9, [5, 5], kpt_conf=0.1)
    other = _det([0, 0, 10, 20], 0.9, [50, 50])
    assert compare(ref, other)["max_kpt_px"] == 0.0
