import torch

from edgelab.models.onnx_export import export_onnx, ort_session, parity
from edgelab.models.salman_cnn_gru import SalmanCnnGru


def test_onnx_matches_pytorch_on_random_weights(tmp_path):
    torch.manual_seed(0)
    model = SalmanCnnGru(num_classes=2).eval()
    path = export_onnx(model, tmp_path / "m.onnx", clip_len=4)
    clips = [torch.rand(1, 4, 3, 256, 256) for _ in range(3)]
    result = parity(model, ort_session(path, 1), clips)
    assert result["parity_max_abs_delta_prob"] <= 1e-4
    assert result["parity_label_flips"] == 0
