import numpy as np

from edgelab.analysis.exp003 import Scores, co_covered_auc, cross_fit, diff, halves


def toy(names, good: bool, seed=0):
    rng = np.random.default_rng(seed)
    labels = [np.r_[np.zeros(20), np.ones(10)].astype(int) for _ in names]
    scores = [l + (0.1 if good else 3.0) * rng.normal(size=l.size) for l in labels]
    covered = [np.ones(l.size, bool) for l in labels]
    return Scores(list(names), labels, scores, covered)


def test_halves_deterministic_and_disjoint():
    names = [f"v{i:02d}" for i in range(11)]
    a, b = halves(names, seed=0)
    assert a == halves(list(reversed(names)), seed=0)[0]
    assert not a & b and len(a | b) == 11


def test_co_covered_uses_only_frames_both_cover():
    names = ["v0", "v1"]
    a, b = toy(names, True), toy(names, True, seed=1)
    b.covered[0][:15] = False  # b loses 15 normal frames of v0
    _, _, frames = co_covered_auc(a, b)
    assert frames == 60 - 15


def test_diff_is_positive_when_second_arm_is_better():
    names = [f"v{i:02d}" for i in range(10)]
    a, b = toy(names, False), toy(names, True)
    result = diff(a, b)
    assert result["diff"] > 0 and result["ci"][0] > 0 and result["p_le_zero"] < 0.05


def test_diff_matches_metrics_paired_bootstrap():
    from edgelab.metrics.scores import paired_bootstrap_diff
    names = [f"v{i:02d}" for i in range(10)]
    a, b = toy(names, False), toy(names, True, seed=3)
    ours = diff(a, b, n_resamples=200)
    theirs = paired_bootstrap_diff(a.labels, b.scores, a.scores, n_resamples=200)
    assert abs(ours["diff"] - theirs[0]) < 1e-12 and abs(ours["ci"][0] - theirs[1]) < 1e-12


def test_cross_fit_picks_the_better_gate_on_each_half():
    names = [f"v{i:02d}" for i in range(10)]
    pooled, chosen = cross_fit({0.25: toy(names, False), 0.10: toy(names, True)})
    assert chosen == {"fit_on_A": 0.10, "fit_on_B": 0.10}
    assert pooled.names == names
