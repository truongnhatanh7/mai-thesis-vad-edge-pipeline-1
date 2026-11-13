"""Synthetic low light, sensor style (EXP-003 plan §4.1).

8-bit sRGB -> linear light -> expected electrons (alpha * linear * full_well) -> Poisson shot noise + Gaussian read
noise -> divide by full_well, clip -> sRGB -> round to 8 bit. No camera gain, so dark frames stay dark and noisy.
"""

import hashlib

import cv2
import numpy as np

# level name -> alpha (None = clean frame). "L2-nonoise" has the same alpha as L2 but no noise (mechanism control).
LEVELS = {"L0": None, "L1": 0.3, "L2": 0.1, "L3": 0.03, "L4": 0.01, "L2-nonoise": 0.1,
          # Addendum B: same alpha, second noise realisation (the seed comes from the level name)
          "L2-s1": 0.1, "L3-s1": 0.03,
          # EXP-005 held-out levels (between L1/L2, L2/L3, L3/L4)
          "X1": 0.2, "X2": 0.05, "X3": 0.02}


def srgb_to_linear(srgb: np.ndarray) -> np.ndarray:
    """srgb in [0, 1] -> linear light in [0, 1] (standard piecewise IEC 61966-2-1 formula)."""
    return np.where(srgb <= 0.04045, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(linear: np.ndarray) -> np.ndarray:
    return np.where(linear <= 0.0031308, linear * 12.92, 1.055 * np.power(linear, 1 / 2.4) - 0.055)


def level_seed(video: str, frame: int, level: str) -> int:
    """Stable across runs and machines (Python's built-in hash() is randomised per process)."""
    digest = hashlib.sha256(f"{video}|{frame}|{level}".encode()).digest()
    return int.from_bytes(digest[:8], "little")


# Addendum C: camera gain (auto-exposure) applied to the SAME noisy signal as the base level (same seed), before
# 8-bit quantisation. name -> (base level, gain). Gain 1/alpha = full exposure compensation; L3 capped at 16x (24 dB).
GAIN_LEVELS = {"L2-gain": ("L2", 10.0), "L3-gain": ("L3", 16.0)}


def darken(img_bgr_uint8: np.ndarray, alpha: float, full_well: float = 1000, read_noise: float = 2.0,
           noise: bool = True, seed: int = 0, gain: float = 1.0) -> np.ndarray:
    linear = srgb_to_linear(img_bgr_uint8.astype(np.float64) / 255.0)
    electrons = alpha * linear * full_well
    if noise:
        rng = np.random.default_rng(seed)
        electrons = rng.poisson(electrons) + rng.normal(0.0, read_noise, electrons.shape)
    signal = electrons / full_well
    if gain != 1.0:
        signal = signal * gain
    dark_srgb = linear_to_srgb(np.clip(signal, 0.0, 1.0))
    return np.round(dark_srgb * 255.0).astype(np.uint8)


def darken_level(img_bgr_uint8: np.ndarray, level: str, video: str, frame: int) -> np.ndarray:
    if level in GAIN_LEVELS:
        base, gain = GAIN_LEVELS[level]
        return darken(img_bgr_uint8, LEVELS[base], seed=level_seed(video, frame, base), gain=gain)
    alpha = LEVELS[level]
    if alpha is None:
        return img_bgr_uint8
    return darken(img_bgr_uint8, alpha, noise=(level != "L2-nonoise"), seed=level_seed(video, frame, level))


def mean_luma(img_bgr_uint8: np.ndarray) -> float:
    """Mean of the grey-scale image, in [0, 255]."""
    return float(cv2.cvtColor(img_bgr_uint8, cv2.COLOR_BGR2GRAY).mean())
