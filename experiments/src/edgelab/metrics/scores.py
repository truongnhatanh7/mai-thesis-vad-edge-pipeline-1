"""Evaluation metrics. Each one is checked against scikit-learn / scipy in tests/test_metrics.py."""

import numpy as np
from sklearn.metrics import roc_auc_score


def frame_level_auc(labels_per_video: list[np.ndarray], scores_per_video: list[np.ndarray]) -> float:
    """Standard VAD protocol: concatenate every test frame of every video, then one ROC-AUC.

    Higher score = more anomalous. Label 1 = anomalous frame.
    """
    labels = np.concatenate(labels_per_video)
    scores = np.concatenate(scores_per_video)
    if labels.shape != scores.shape:
        raise ValueError(f"labels {labels.shape} and scores {scores.shape} differ")
    return float(roc_auc_score(labels, scores))


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def binary_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = np.asarray(y_true).astype(bool), np.asarray(y_pred).astype(bool)
    tp = np.sum(y_true & y_pred)
    precision_denominator, recall_denominator = np.sum(y_pred), np.sum(y_true)
    if tp == 0:
        return 0.0
    precision, recall = tp / precision_denominator, tp / recall_denominator
    return float(2 * precision * recall / (precision + recall))


def bootstrap_ci_over_videos(labels_per_video, scores_per_video, metric=frame_level_auc,
                             n_resamples: int = 2000, confidence: float = 0.95, seed: int = 0):
    """Percentile CI. Videos (not frames) are resampled because frames of one video are not independent."""
    rng = np.random.default_rng(seed)
    n = len(labels_per_video)
    estimates = []
    while len(estimates) < n_resamples:
        idx = rng.integers(0, n, n)
        labels = [labels_per_video[i] for i in idx]
        if len(np.unique(np.concatenate(labels))) < 2:
            continue  # AUC undefined without both classes
        estimates.append(metric(labels, [scores_per_video[i] for i in idx]))
    alpha = (1 - confidence) / 2
    return float(np.quantile(estimates, alpha)), float(np.quantile(estimates, 1 - alpha))


def paired_bootstrap_diff(labels_per_video, scores_a, scores_b, metric=frame_level_auc,
                          n_resamples: int = 2000, confidence: float = 0.95, seed: int = 0):
    """AUC(a) - AUC(b) with a percentile CI. Both arms see the SAME resampled videos in every resample (paired).

    Returns (diff on all videos, ci_low, ci_high).
    """
    rng = np.random.default_rng(seed)
    n = len(labels_per_video)
    diffs = []
    while len(diffs) < n_resamples:
        idx = rng.integers(0, n, n)
        labels = [labels_per_video[i] for i in idx]
        if len(np.unique(np.concatenate(labels))) < 2:
            continue
        diffs.append(metric(labels, [scores_a[i] for i in idx]) - metric(labels, [scores_b[i] for i in idx]))
    alpha = (1 - confidence) / 2
    diff = metric(labels_per_video, scores_a) - metric(labels_per_video, scores_b)
    return float(diff), float(np.quantile(diffs, alpha)), float(np.quantile(diffs, 1 - alpha))
