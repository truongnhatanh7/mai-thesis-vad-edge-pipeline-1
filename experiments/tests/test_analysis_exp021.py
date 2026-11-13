"""EXP-021: the coverage bound is attained by the best-case scoring and never beaten by random scorings."""

import numpy as np
from sklearn.metrics import roc_auc_score

from edgelab.analysis.exp021 import bound_half, bound_loose, bound_pess, valid_rows


def _best_case(n_a, n_n, ca, cn, covered_normals_on_floor=False):
    """Labels and scores: uncovered frames at the floor (0), covered anomalous on top, covered normal just above the floor."""
    a_cov, n_cov = round(ca * n_a), round(cn * n_n)
    labels = np.r_[np.ones(n_a), np.zeros(n_n)].astype(int)
    scores_a = np.r_[np.full(a_cov, 10.0) + np.arange(a_cov), np.zeros(n_a - a_cov)]
    base = np.zeros(n_cov) if covered_normals_on_floor else 1.0 + np.arange(n_cov) / (n_cov + 1)  # strictly above the floor
    scores_n = np.r_[base, np.zeros(n_n - n_cov)]
    return labels, np.r_[scores_a, scores_n]


def test_best_case_scoring_attains_the_half_bound():
    for ca, cn in [(0.5, 0.5), (0.2, 0.9), (1.0, 1.0), (0.0, 0.0), (0.8, 0.1)]:
        labels, scores = _best_case(100, 200, ca, cn)
        assert abs(roc_auc_score(labels, scores) - bound_half(ca, cn)) < 1e-9
        assert bound_pess(ca, cn) <= bound_half(ca, cn) <= bound_loose(ca, cn)


def test_covered_normals_on_the_floor_attain_the_loose_bound():
    labels, scores = _best_case(100, 200, 0.3, 0.6, covered_normals_on_floor=True)
    assert abs(roc_auc_score(labels, scores) - bound_loose(0.3, 0.6)) < 1e-9


def test_no_random_scoring_beats_the_loose_bound_and_floor_tied_frames_respect_the_half_bound():
    rng = np.random.default_rng(0)
    for _ in range(200):
        n_a, n_n = int(rng.integers(5, 60)), int(rng.integers(5, 60))
        ca, cn = rng.integers(0, n_a + 1) / n_a, rng.integers(0, n_n + 1) / n_n
        labels = np.r_[np.ones(n_a), np.zeros(n_n)].astype(int)
        cov = np.r_[np.arange(n_a) < round(ca * n_a), np.arange(n_n) < round(cn * n_n)]
        scores = np.where(cov, rng.random(n_a + n_n) + 0.1, 0.0)  # covered: random, strictly above the floor; uncovered: floor
        auc = roc_auc_score(labels, scores)
        assert auc <= bound_half(cov[labels == 1].mean(), cov[labels == 0].mean()) + 1e-12
        assert auc <= bound_loose(cov[labels == 1].mean(), cov[labels == 0].mean()) + 1e-12


def test_valid_rows_drops_placeholders_other_gates_invalid_and_keeps_last_rerun():
    base = {"c": 0.25, "variant": "paper", "coverage_anomalous": 0.5, "coverage_normal": 0.4}
    rows = [{**base, "cache": "A", "auc": 0.5, "no_segments": True},          # placeholder, superseded below
            {**base, "cache": "A", "auc": 0.6, "no_segments": False},         # real row
            {**base, "cache": "B", "auc": 0.7, "c": 0.1},                     # other gate
            {**base, "cache": "C", "auc": 0.9, "invalid": "incomplete cache"},
            {**base, "cache": "D", "auc": 0.55}, {**base, "cache": "D", "auc": 0.56}]  # re-run: last wins
    got = {r["config"]: r["auc"] for r in valid_rows(rows, "cache", "src")}
    assert got == {"A": 0.6, "D": 0.56}


def test_effective_coverage_counts_frames_above_the_file_minimum(tmp_path):
    from edgelab.analysis.exp021 import effective_coverage
    np.savez(tmp_path / "s.npz", names=np.array(["v"]), score_v=np.array([0.5, 0.5, 0.6, 0.7]), label_v=np.array([1, 0, 1, 0]))
    assert effective_coverage(tmp_path / "s.npz") == (0.5, 0.5)
    assert effective_coverage(tmp_path / "missing.npz") is None
