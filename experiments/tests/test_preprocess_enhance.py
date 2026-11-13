"""Guards the enhancers: gamma formula, CLAHE effect, Zero-DCE++ weights load, torch and ONNX Runtime agree."""

import numpy as np
import pytest

from edgelab.preprocess.darken import darken, mean_luma
from edgelab.preprocess.enhance import ENHANCERS, clahe_lab, gamma_adaptive, gamma_for
from edgelab.preprocess.zerodcepp import DEFAULT_WEIGHTS, ZeroDCEpp, ZeroDCEppOnnx, bgr_to_input

needs_weights = pytest.mark.skipif(not DEFAULT_WEIGHTS.exists(), reason="Zero-DCE++ repo not cloned")


def _dark_scene(shape=(120, 160, 3)):
    rng = np.random.default_rng(0)
    base = rng.integers(60, 200, shape, dtype=np.uint8)
    return darken(base, 0.05, seed=1)


def test_gamma_formula_hand_values():
    assert gamma_for(0.4) == pytest.approx(1.0)
    assert gamma_for(0.2) == pytest.approx(np.log(0.4) / np.log(0.2))  # 0.569
    assert gamma_for(0.9) == 1.0  # bright frame: clipped to "no change"
    assert gamma_for(0.001) == 0.2  # very dark: clipped at the strongest gamma


def test_gamma_guards_for_black_and_white_frames():
    assert gamma_for(0.0) == 0.2 and gamma_for(1.0) == 1.0
    black, white = np.zeros((8, 8, 3), np.uint8), np.full((8, 8, 3), 255, np.uint8)
    np.testing.assert_array_equal(gamma_adaptive(black), black)
    np.testing.assert_array_equal(gamma_adaptive(white), white)


def test_gamma_brightens_dark_frame_and_leaves_bright_frame_alone():
    dark = _dark_scene()
    assert mean_luma(gamma_adaptive(dark)) > mean_luma(dark) + 20
    bright = np.full((8, 8, 3), 200, np.uint8)
    np.testing.assert_array_equal(gamma_adaptive(bright), bright)


def test_clahe_keeps_shape_dtype_and_raises_contrast_of_a_dull_image():
    dull = np.random.default_rng(1).integers(100, 120, (96, 128, 3), dtype=np.uint8)
    out = clahe_lab(dull)
    assert out.shape == dull.shape and out.dtype == np.uint8
    assert out.std() > dull.std()


def test_registry_functions_map_uint8_to_uint8_with_same_shape():
    img = _dark_scene()
    for name in ("E0", "E1", "E2"):
        out = ENHANCERS[name](img)
        assert out.shape == img.shape and out.dtype == np.uint8
    assert ENHANCERS["E0"](img) is img


@needs_weights
def test_zerodce_loads_official_weights_and_brightens_dark_frame():
    enhancer = ZeroDCEpp()
    assert sum(p.numel() for p in enhancer.net.parameters()) == 10_561  # the Zero-DCE++ paper's parameter count
    dark = _dark_scene()
    out = enhancer(dark)
    assert out.shape == dark.shape and out.dtype == np.uint8
    assert mean_luma(out) > mean_luma(dark) + 10


@needs_weights
def test_our_network_equals_the_official_code_with_the_same_weights():
    import importlib.util

    import torch
    spec = importlib.util.spec_from_file_location("official_model", DEFAULT_WEIGHTS.parents[1] / "model.py")
    official_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(official_module)
    official = official_module.enhance_net_nopool(1)
    official.load_state_dict(torch.load(DEFAULT_WEIGHTS, map_location="cpu"))
    ours = ZeroDCEpp().net
    x = torch.from_numpy(bgr_to_input(_dark_scene()))
    with torch.no_grad():
        np.testing.assert_allclose(ours(x).numpy(), official.eval()(x)[0].numpy(), atol=1e-6)


@needs_weights
def test_onnx_runtime_matches_torch_on_480x856(tmp_path):
    pytest.importorskip("onnxruntime")
    enhancer = ZeroDCEpp()
    onnx_path = enhancer.export_onnx(tmp_path / "zerodcepp.onnx")
    ort_model = ZeroDCEppOnnx(onnx_path, threads=2)

    image = np.random.default_rng(2).integers(0, 80, (480, 856, 3), dtype=np.uint8)
    import torch
    x = bgr_to_input(image)
    with torch.no_grad():
        torch_out = enhancer.net(torch.from_numpy(x)).numpy()
    float_diff = float(np.abs(torch_out - ort_model.run_float(x)).max())
    uint8_diff = int(np.abs(enhancer(image).astype(int) - ort_model(image).astype(int)).max())
    print(f"torch vs ORT on 480x856: max abs diff {float_diff:.2e} (float, [0,1]), {uint8_diff} LSB (uint8)")
    assert float_diff < 1e-4
    assert uint8_diff <= 1
