"""Low-light enhancers (EXP-003 plan §4.2). Every enhancer is `fn(img_bgr_uint8) -> img_bgr_uint8`."""

import os

import cv2
import numpy as np

from edgelab.preprocess.darken import mean_luma
from edgelab.preprocess.zerodcepp import ZeroDCEpp, ZeroDCEppOnnx  # noqa: F401  (re-exported)

TARGET_MEAN_Y = 0.4


def gamma_for(mean_y: float) -> float:
    """gamma = clip(log 0.4 / log mean_Y, 0.2, 1.0). Black frame -> strongest gamma; frame already >= 1 -> none."""
    if mean_y <= 0.0:
        return 0.2
    if mean_y >= 1.0:
        return 1.0
    return float(np.clip(np.log(TARGET_MEAN_Y) / np.log(mean_y), 0.2, 1.0))


def gamma_adaptive(img_bgr_uint8: np.ndarray) -> np.ndarray:
    gamma = gamma_for(mean_luma(img_bgr_uint8) / 255.0)
    lut = np.round(255.0 * (np.arange(256) / 255.0) ** gamma).astype(np.uint8)
    return cv2.LUT(img_bgr_uint8, lut)


def clahe_lab(img_bgr_uint8: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(img_bgr_uint8, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def identity(img_bgr_uint8: np.ndarray) -> np.ndarray:
    return img_bgr_uint8


_zerodce_cache: dict[str, ZeroDCEpp] = {}


def zerodce(img_bgr_uint8: np.ndarray, device: str | None = None) -> np.ndarray:
    """Zero-DCE++ via PyTorch; the model is loaded on first use. Device: argument, else $EDGELAB_ZERODCE_DEVICE, else cpu."""
    device = device or os.environ.get("EDGELAB_ZERODCE_DEVICE", "cpu")
    if device not in _zerodce_cache:
        _zerodce_cache[device] = ZeroDCEpp(device=device)
    return _zerodce_cache[device](img_bgr_uint8)


ENHANCERS = {"E0": identity, "E1": gamma_adaptive, "E2": clahe_lab, "E3": zerodce}
