import torch

from edgelab.models.salman_cnn_gru import SalmanCnnGru, WithSoftmax, macs_per_frame, parameter_count

HAND_ESTIMATE_PARAMS = 0.18e6
HAND_ESTIMATE_MACS = 0.099e9


def test_output_shape_and_probabilities():
    model = SalmanCnnGru(num_classes=2).eval()
    clip = torch.rand(2, 8, 3, 256, 256)
    assert model(clip).shape == (2, 2)
    probs = WithSoftmax(model)(clip)
    assert torch.allclose(probs.sum(-1), torch.ones(2), atol=1e-6)


def test_encoder_shapes_follow_table3():
    model = SalmanCnnGru().eval()
    expected = {0: (32, 128, 128), 1: (32, 64, 64), 2: (64, 64, 64), 3: (64, 32, 32), 4: (128, 32, 32)}
    x = torch.zeros(1, 3, 256, 256)
    for i, layer in enumerate(model.encoder.layers):
        x = layer(x)
        if i in expected:
            assert tuple(x.shape[1:]) == expected[i], f"layer {i}"


def test_size_within_2x_of_hand_estimate():
    model = SalmanCnnGru(num_classes=2)
    params, macs = parameter_count(model), macs_per_frame(model)
    print(f"params={params} macs/frame={macs / 1e9:.4f} G")
    assert HAND_ESTIMATE_PARAMS / 2 < params < HAND_ESTIMATE_PARAMS * 2
    assert HAND_ESTIMATE_MACS / 2 < macs < HAND_ESTIMATE_MACS * 2


def test_six_class_head():
    assert SalmanCnnGru(num_classes=6).eval()(torch.rand(1, 3, 3, 256, 256)).shape == (1, 6)
