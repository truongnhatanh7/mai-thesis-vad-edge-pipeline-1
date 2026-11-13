"""EXP-020 P2 model: weight-name mapping, shape-checked transfer, and the pre-flight gate (P3-P5 outputs equal stock)."""

import pytest
import torch
from ultralytics.nn.tasks import PoseModel

from edgelab.pose.p2model import P2_YAML, build_p2, stock_key_to_p2, transfer


def _randomised_stock() -> PoseModel:
    """Stock yolov8n-pose graph with non-trivial weights and BatchNorm statistics (so equality tests are not vacuous)."""
    torch.manual_seed(0)
    model = PoseModel("yolov8n-pose.yaml", nc=1, data_kpt_shape=(17, 3), verbose=False)
    with torch.no_grad():
        for name, tensor in model.state_dict().items():
            if tensor.dtype.is_floating_point:
                if "running_var" in name:
                    tensor.copy_(torch.rand_like(tensor) + 0.5)
                else:
                    tensor.add_(0.1 * torch.randn_like(tensor))
    return model.eval()


def test_key_mapping_rules():
    assert stock_key_to_p2("model.0.conv.weight") == "model.0.conv.weight"
    assert stock_key_to_p2("model.21.cv1.bn.running_mean") == "model.21.cv1.bn.running_mean"
    assert stock_key_to_p2("model.22.cv2.0.0.conv.weight") == "model.25.cv2.0.0.conv.weight"
    assert stock_key_to_p2("model.22.cv3.2.2.bias") == "model.25.cv3.2.2.bias"
    assert stock_key_to_p2("model.22.cv4.1.0.bn.weight") == "model.25.cv4.1.0.bn.weight"
    assert stock_key_to_p2("model.22.dfl.conv.weight") == "model.25.dfl.conv.weight"
    assert stock_key_to_p2("model.22.cv2.3.0.conv.weight") is None  # no such stock level
    assert stock_key_to_p2("something.else") is None


def test_every_stock_tensor_is_copied_and_new_tensors_are_the_p2_branch_and_level3_head():
    stock = _randomised_stock()
    p2, report = build_p2(stock)
    assert report["copied"] == report["tensors_stock"] and report["stock_tensors_without_counterpart"] == []
    assert report["new"] == report["tensors_p2"] - report["tensors_stock"] > 0
    assert all(k.startswith(("model.22.", "model.23.", "model.24.")) or
               any(k.startswith(f"model.25.{h}.3.") for h in ("cv2", "cv3", "cv4")) for k in report["new_tensors"])
    p2_state, stock_state = p2.state_dict(), stock.state_dict()
    for key, tensor in stock_state.items():
        assert torch.equal(p2_state[stock_key_to_p2(key)], tensor), key


def test_transfer_fails_loudly_on_a_shape_mismatch():
    stock = _randomised_stock().state_dict()
    p2 = PoseModel(str(P2_YAML), nc=1, data_kpt_shape=(17, 3), verbose=False).state_dict()
    bad = dict(stock)
    bad["model.22.cv3.0.2.weight"] = torch.zeros(2, 64, 1, 1)
    with pytest.raises(ValueError):
        transfer(bad, p2)


def test_p2_model_has_four_levels_with_stride_4_last():
    p2, _ = build_p2(_randomised_stock())
    assert [float(s) for s in p2.stride] == [8.0, 16.0, 32.0, 4.0]
    assert p2.model[-1].nl == 4 and list(p2.kpt_shape) == [17, 3]


def test_preflight_gate_p3_p4_p5_outputs_equal_stock():
    """Same input -> the first 3 levels of the P2 model give the stock model's boxes/scores/keypoints exactly."""
    stock = _randomised_stock()
    p2, _ = build_p2(stock)
    p2.eval()
    x = torch.rand(1, 3, 128, 160)
    with torch.no_grad():
        (decoded_s, raw_s), (decoded_p, raw_p) = stock(x), p2(x)
    n = raw_s["scores"].shape[-1]  # anchors of levels P3, P4, P5
    assert decoded_p.shape[-1] > n
    for key in ("boxes", "scores", "kpts"):
        assert torch.allclose(raw_p[key][..., :n], raw_s[key], atol=1e-6), key
    assert torch.allclose(decoded_p[..., :n], decoded_s, atol=1e-5)  # decoded boxes in pixels + sigmoid scores + keypoints
    assert decoded_p.shape[-1] == n + (128 // 4) * (160 // 4)  # the P2 level adds one anchor per 4 x 4 pixels


def test_saved_checkpoint_loads_as_a_pose_task_model(tmp_path):
    """Regression: a checkpoint without train_args.task is read as task 'detect' and would train without keypoints."""
    from ultralytics import YOLO

    from edgelab.pose.p2model import save_checkpoint

    stock = _randomised_stock()
    p2, _ = build_p2(stock)
    save_checkpoint(p2, tmp_path / "init.pt")
    loaded = YOLO(str(tmp_path / "init.pt"))
    assert loaded.task == "pose" and isinstance(loaded.model, PoseModel) and loaded.model.model[-1].nl == 4
    assert torch.equal(loaded.model.state_dict()["model.25.cv2.0.0.conv.weight"].float().half().float(),
                       stock.state_dict()["model.22.cv2.0.0.conv.weight"].half().float())  # fp16 round trip of a copied tensor
