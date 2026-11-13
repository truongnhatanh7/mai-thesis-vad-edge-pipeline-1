"""EXP-020 evaluation glue: verdict rules, pooled recall, AUC cell with paired difference on synthetic score files."""

import json

import numpy as np

from edgelab.analysis.exp020 import (auc_cell, best_of, p1_label_cause, p2_resolution_cause, p3_general, p4_p2_wins,
                                     p5_combination, pooled_recall, verdicts)


def test_p1_needs_margin_ci_and_mid_height_recall():
    assert p1_label_cause(0.70, 0.68, 0.005, 0.40, 0.30)
    assert not p1_label_cause(0.6999, 0.68, 0.005, 0.40, 0.30)  # +1.99 < 2.0
    assert not p1_label_cause(0.70, 0.68, -0.001, 0.40, 0.30)  # CI includes 0
    assert not p1_label_cause(0.70, 0.68, 0.005, 0.30, 0.30)  # recall not higher


def test_p2_margin_and_ci():
    assert p2_resolution_cause(0.70, 0.68, 0.001) and not p2_resolution_cause(0.70, 0.68, 0.0)
    assert not p2_resolution_cause(0.69, 0.68, 0.01)


def test_p3_best_arm_uses_sht_l0_and_both_conditions():
    assert best_of({"ML0": 0.70, "M0P": 0.72, "MLP": 0.71}) == "M0P"
    assert p3_general(ub_best=0.52, ub_m00=0.50, sht_best=0.696, sht_m00=0.70)
    assert not p3_general(0.52, 0.50, 0.69, 0.70)  # -1.0 point on SHT: below the -0.5 floor
    assert not p3_general(0.50, 0.50, 0.75, 0.70)  # UB point difference must be > 0


def test_p3_floor_is_half_a_point():
    assert not p3_general(0.52, 0.50, 0.694, 0.70)
    assert p3_general(0.52, 0.50, 0.695, 0.70)


def test_p4_wins_if_auc_not_lower_and_ms_within_110_percent():
    assert p4_p2_wins(0.71, 0.70, 20.0, 19.0)  # 1.053x
    assert not p4_p2_wins(0.69, 0.70, 18.0, 19.0)
    assert not p4_p2_wins(0.71, 0.70, 21.0, 19.0)  # 1.105x
    assert p4_p2_wins(0.71, 0.70, 20.9, 19.0)  # 1.0999x


def test_p5_descriptive_comparison():
    assert p5_combination(0.72, 0.71, 0.70) and not p5_combination(0.70, 0.71, 0.69)


def test_pooled_recall_pools_counts_not_rates():
    rec = {"48-64": {"persons": 100, "found": 10}, "64-96": {"persons": 300, "found": 150}}
    assert pooled_recall(rec) == 160 / 400


def _write_arm(root, cache, labels, scores, auc):
    d = root / "scores" / f"{cache}_c0.25_botsort"
    d.mkdir(parents=True)
    names = [f"v{i}" for i in range(len(labels))]
    arrays = {"names": np.array(names)}
    for n, y, s in zip(names, labels, scores):
        arrays[f"label_{n}"], arrays[f"score_{n}"], arrays[f"covered_{n}"] = y, s, np.ones(len(y), bool)
    np.savez(d / "frame_scores_paper.npz", **arrays)
    row = {"cache": cache, "c": 0.25, "tracker": "botsort", "variant": "paper", "auc": auc, "ci95_low": auc - 0.05,
           "ci95_high": auc + 0.05, "coverage_anomalous": 0.9, "coverage_normal": 0.8}
    (root / "results.jsonl").write_text(json.dumps(row) + "\n")


def test_auc_cell_pairs_with_reference_and_signs_the_difference(tmp_path):
    rng = np.random.default_rng(0)
    labels = [np.r_[np.zeros(40), np.ones(40)] for _ in range(12)]
    good = [y * 1.0 + rng.normal(0, 0.5, len(y)) for y in labels]  # separates well
    noise = [rng.normal(0, 1, len(y)) for y in labels]
    _write_arm(tmp_path / "ref", "L0_E0", labels, noise, 0.5)
    _write_arm(tmp_path / "arm", "L0_E0", labels, good, 0.9)
    cell = auc_cell(tmp_path / "arm", "L0_E0", "paper", tmp_path / "ref")
    assert cell["auc"] == 0.9 and cell["vs_ref"]["diff"] > 0.2 and cell["vs_ref"]["ci"][0] > 0
    assert auc_cell(tmp_path / "ref", "L0_E0", "paper")["auc"] == 0.5 and "vs_ref" not in auc_cell(tmp_path / "ref", "L0_E0", "paper")


def test_verdicts_assemble_from_a_report():
    cell = lambda auc, lo=None: {"auc": auc, **({"vs_ref": {"diff": auc - 0.68, "ci": [lo, lo + 0.05]}} if lo is not None else {})}  # noqa: E731
    paper = {"M00": cell(0.68), "ML0": cell(0.71, 0.01), "M0P": cell(0.69, -0.01), "MLP": cell(0.72, 0.02)}
    ub = {"M00": cell(0.50), "ML0": cell(0.49, 0.0), "M0P": cell(0.50, 0.0), "MLP": cell(0.52, 0.0)}
    r = lambda found: {b: {"persons": 100, "found": found} for b in ("48-64", "64-96")}  # noqa: E731
    report = {"arms": ["M00", "ML0", "M0P", "MLP"], "auc": {"SHT L0": {"paper": paper}, "UB L0": {"paper": ub}},
              "recall": {"M00": r(10), "ML0": r(40), "M0P": r(20), "MLP": r(50)}, "timing": {}}
    out = verdicts(report)
    assert out["P1"]["pass"] and not out["P2"]["pass"]
    assert out["P3"]["best"] == "MLP" and out["P3"]["pass"]
    assert out["P5"]["pass"] and "P4" not in out
