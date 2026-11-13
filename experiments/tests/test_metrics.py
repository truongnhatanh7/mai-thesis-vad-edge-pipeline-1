import numpy as np
import pytest
from scipy import stats
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from edgelab.metrics.scores import (accuracy, binary_f1, bootstrap_ci_over_videos, frame_level_auc,
                                    paired_bootstrap_diff)

rng = np.random.default_rng(42)


def _toy_videos(n_videos=30, frames=50):
    labels, scores = [], []
    for _ in range(n_videos):
        y = (rng.random(frames) < 0.3).astype(int)
        labels.append(y)
        scores.append(y * 0.5 + rng.random(frames))
    return labels, scores


def test_frame_level_auc_matches_sklearn_on_concatenated_frames():
    labels, scores = _toy_videos()
    expected = roc_auc_score(np.concatenate(labels), np.concatenate(scores))
    assert frame_level_auc(labels, scores) == pytest.approx(expected, abs=1e-12)


def test_frame_level_auc_rejects_misaligned_inputs():
    with pytest.raises(ValueError):
        frame_level_auc([np.array([0, 1, 1])], [np.array([0.1, 0.9])])


def test_accuracy_and_f1_match_sklearn():
    y_true = rng.integers(0, 2, 500)
    y_pred = rng.integers(0, 2, 500)
    assert accuracy(y_true, y_pred) == pytest.approx(accuracy_score(y_true, y_pred), abs=1e-12)
    assert binary_f1(y_true, y_pred) == pytest.approx(f1_score(y_true, y_pred), abs=1e-12)


def test_bootstrap_ci_close_to_scipy():
    labels, scores = _toy_videos(n_videos=40)
    ours = bootstrap_ci_over_videos(labels, scores, n_resamples=2000, seed=1)

    def auc_of_indices(idx):
        idx = idx.astype(int)
        return frame_level_auc([labels[i] for i in idx], [scores[i] for i in idx])

    ref = stats.bootstrap((np.arange(len(labels)),), auc_of_indices, n_resamples=2000, method="percentile",
                          vectorized=False, random_state=1).confidence_interval
    assert ours[0] == pytest.approx(ref.low, abs=0.01)
    assert ours[1] == pytest.approx(ref.high, abs=0.01)


def test_paired_bootstrap_diff_hand_computed_case():
    # Every video is [normal, normal, anomalous, anomalous]. Arm a ranks them perfectly (AUC 1); arm b scores
    # everything equal (all ties, AUC 0.5). Any resample of videos gives the same AUCs, so the CI collapses to 0.5.
    labels = [np.array([0, 0, 1, 1])] * 3
    perfect = [np.array([0.0, 0.1, 0.9, 1.0])] * 3
    ties = [np.zeros(4)] * 3
    assert paired_bootstrap_diff(labels, perfect, ties, n_resamples=50) == (0.5, 0.5, 0.5)
    assert paired_bootstrap_diff(labels, ties, perfect, n_resamples=50) == (-0.5, -0.5, -0.5)
    assert paired_bootstrap_diff(labels, perfect, perfect, n_resamples=50) == (0.0, 0.0, 0.0)


def test_paired_bootstrap_ci_is_narrower_than_unpaired_for_correlated_arms():
    labels, scores_a = _toy_videos(n_videos=40)
    scores_b = [s + 0.05 * rng.normal(size=len(s)) for s in scores_a]  # nearly the same ranking as a
    diff, low, high = paired_bootstrap_diff(labels, scores_a, scores_b, n_resamples=500, seed=3)
    ci_a, ci_b = (bootstrap_ci_over_videos(labels, s, n_resamples=500, seed=3) for s in (scores_a, scores_b))
    assert low <= diff <= high
    assert high - low < (ci_a[1] - ci_a[0]) + (ci_b[1] - ci_b[0])
