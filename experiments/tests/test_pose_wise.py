"""EXP-014 weight interpolation: hand-computed tensors, and a round trip through a real yolov8n-pose checkpoint."""

import pytest
import torch

from edgelab.pose.detcache import WEIGHTS
from edgelab.pose.wise import interpolate_state_dicts, make_wise_weights


def test_interpolation_is_a_weighted_mean_of_float_tensors():
    a = {"w": torch.tensor([0.0, 4.0]), "bn.num_batches_tracked": torch.tensor(7)}
    b = {"w": torch.tensor([8.0, 0.0]), "bn.num_batches_tracked": torch.tensor(9)}
    assert interpolate_state_dicts(a, b, 0.25)["w"].tolist() == [2.0, 3.0]  # 0.75 * a + 0.25 * b
    assert interpolate_state_dicts(a, b, 0.0)["w"].tolist() == [0.0, 4.0]
    assert interpolate_state_dicts(a, b, 1.0)["w"].tolist() == [8.0, 0.0]


def test_integer_entries_come_from_the_finetuned_model():
    a = {"bn.num_batches_tracked": torch.tensor(7)}
    b = {"bn.num_batches_tracked": torch.tensor(9)}
    assert interpolate_state_dicts(a, b, 0.5)["bn.num_batches_tracked"].item() == 9


def test_different_architectures_are_rejected():
    with pytest.raises(ValueError, match="different keys"):
        interpolate_state_dicts({"w": torch.zeros(1)}, {"v": torch.zeros(1)}, 0.5)
    with pytest.raises(ValueError, match="shape mismatch"):
        interpolate_state_dicts({"w": torch.zeros(1)}, {"w": torch.zeros(2)}, 0.5)


@pytest.mark.skipif(not WEIGHTS.exists(), reason="yolov8n-pose.pt not present")
def test_real_checkpoint_round_trip(tmp_path):
    from ultralytics import YOLO

    original = YOLO(str(WEIGHTS))
    key = "model.0.conv.weight"
    original.model.state_dict()[key].add_(1.0)  # a second checkpoint that differs by +1 on one tensor
    shifted = tmp_path / "shifted.pt"
    original.save(str(shifted))
    mixed = tmp_path / "mixed.pt"
    make_wise_weights(str(WEIGHTS), str(shifted), 0.5, str(mixed))
    base = YOLO(str(WEIGHTS)).model.state_dict()[key]
    got = YOLO(str(mixed)).model.state_dict()[key]
    assert torch.allclose(got, base + 0.5, atol=2e-2)  # stored as fp16 (half): rounding only
