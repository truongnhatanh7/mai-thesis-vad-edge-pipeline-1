"""EXP-013/014: frozen verdict thresholds at their boundaries, alpha selection, and a cell from fake saved scores."""

import json

import numpy as np

from edgelab.analysis.exp013 import (cell, f1_more_people, f2_higher_accuracy, f3_no_transfer_damage, w1_no_in_domain_loss,
                                     w2_robustness_gain)
from edgelab.analysis.exp014 import select_alpha


def test_f1_needs_four_points_more_coverage():
    assert f1_more_people(0.714, 0.674) and not f1_more_people(0.713, 0.674)


def test_f2_needs_one_and_a_half_points_and_a_positive_paired_ci():
    assert f2_higher_accuracy(0.700, 0.685, paired_ci_low=0.001)
    assert not f2_higher_accuracy(0.700, 0.685, paired_ci_low=-0.001)  # big enough gain, CI includes 0
    assert not f2_higher_accuracy(0.6999, 0.685, paired_ci_low=0.01)  # CI fine, gain 1.49


def test_f3_allows_half_a_point_loss_on_ubnormal():
    assert f3_no_transfer_damage(0.5525, 0.5575) and not f3_no_transfer_damage(0.5524, 0.5575)


def test_w1_allows_three_tenths_of_a_point_loss():
    assert w1_no_in_domain_loss(0.697, 0.700) and not w1_no_in_domain_loss(0.6969, 0.700)


def test_w2_needs_one_point_on_L3_or_on_ubnormal():
    assert w2_robustness_gain(0.62, 0.61, 0.55, 0.55)  # L3 +1.0
    assert w2_robustness_gain(0.60, 0.61, 0.56, 0.55)  # UB +1.0 only
    assert not w2_robustness_gain(0.6199, 0.61, 0.5599, 0.55)


def test_select_alpha_takes_the_highest_coverage_among_the_interior_alphas_only():
    assert select_alpha({0.0: 0.5, 0.25: 0.6, 0.5: 0.7, 0.75: 0.65, 1.0: 0.9}) == 0.5  # the end alpha 1.0 is never chosen


def test_select_alpha_ties_go_to_the_alpha_closest_to_half_then_the_smaller():
    assert select_alpha({0.25: 0.7, 0.5: 0.7, 0.75: 0.7}) == 0.5
    assert select_alpha({0.25: 0.7, 0.5: 0.6, 0.75: 0.7}) == 0.25


def _fake_root(root, auc_scores):
    """One setting 'L0_E0': two videos of 4 frames; scores per video given; covered = score != 0."""
    (root / "scores/L0_E0_c0.25_botsort").mkdir(parents=True)
    arrays = {"names": np.array(["v1", "v2"])}
    for name, labels, scores in zip(("v1", "v2"), ([0, 0, 1, 1], [0, 1, 0, 1]), auc_scores):
        arrays |= {f"label_{name}": np.array(labels), f"score_{name}": np.array(scores, float),
                   f"covered_{name}": np.array(scores) != 0}
    np.savez(root / "scores/L0_E0_c0.25_botsort/frame_scores_paper.npz", **arrays)
    row = {"cache": "L0_E0", "c": 0.25, "tracker": "botsort", "variant": "paper", "auc": 0.9, "ci95_low": 0.8,
           "ci95_high": 1.0, "coverage_anomalous": 0.75, "coverage_normal": 0.5}
    (root / "results.jsonl").write_text(json.dumps(row) + "\n")


def test_cell_reports_row_values_all_frame_coverage_and_paired_difference(tmp_path):
    base, arm = tmp_path / "base", tmp_path / "arm"
    _fake_root(base, ([0, 0, 1, 2], [0, 0, 0, 1]))
    _fake_root(arm, ([1, 1, 3, 4], [2, 2, 1, 3]))  # better ranking and everything covered
    result = cell(arm, base, "L0_E0", "paper")
    assert result["coverage_all"] == 1.0 and result["videos"] == 2
    assert result["coverage_anomalous"] == 0.75  # straight from the sweep row
    assert result["vs"]["diff"] > 0  # arm separates the labels better than the baseline
