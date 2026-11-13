"""Per-frame score smoothing: the paper's offline (centred) filter and causal (past-only) replacements."""

import numpy as np
from scipy.ndimage import gaussian_filter1d

STGNF_SIGMAS = range(1, 7)
# Gaussians compose by adding variances: sigma 1..6 applied in sequence ~ one Gaussian of sqrt(91) ~ 9.54 frames.
STGNF_EQUIVALENT_SIGMA = float(np.sqrt(sum(s * s for s in STGNF_SIGMAS)))


def offline_gaussian(scores: np.ndarray) -> np.ndarray:
    """Exact copy of STG-NF utils/scoring_utils.smooth_scores for one clip (uses future frames)."""
    out = np.asarray(scores, dtype=float)
    for sigma in STGNF_SIGMAS:
        out = gaussian_filter1d(out, sigma=sigma)
    return out


def causal_half_gaussian(scores: np.ndarray, sigma: float = STGNF_EQUIVALENT_SIGMA, truncate: float = 4.0) -> np.ndarray:
    """Weighted mean of the current and past frames only, weights exp(-k^2 / 2 sigma^2) for lag k >= 0."""
    scores = np.asarray(scores, dtype=float)
    lags = np.arange(int(truncate * sigma) + 1)
    weights = np.exp(-(lags ** 2) / (2 * sigma ** 2))
    out = np.empty_like(scores)
    for t in range(len(scores)):
        window = scores[max(0, t - lags[-1]): t + 1][::-1]
        w = weights[: len(window)]
        out[t] = np.dot(w, window) / w.sum()
    return out


def causal_ewma(scores: np.ndarray, time_constant: float = STGNF_EQUIVALENT_SIGMA) -> np.ndarray:
    """Exponentially weighted moving average, alpha = 1 - exp(-1 / time_constant)."""
    scores = np.asarray(scores, dtype=float)
    alpha = 1.0 - np.exp(-1.0 / time_constant)
    out = np.empty_like(scores)
    acc = scores[0] if len(scores) else 0.0
    for t, s in enumerate(scores):
        acc = alpha * s + (1 - alpha) * acc
        out[t] = acc
    return out


SMOOTHERS = {
    "none": lambda s: np.asarray(s, dtype=float),
    "offline_gaussian": offline_gaussian,
    "causal_half_gaussian": causal_half_gaussian,
    "causal_ewma": causal_ewma,
}
