"""EXP-019 driver pieces: the seed-paired bootstrap that decides G1, and the timing harness."""

import json

import numpy as np

from edgelab.analysis import exp019
from edgelab.metrics.scores import frame_level_auc


def fake_arm_files(root, labels, strength_of_arm, rng, test="released", variant="paper"):
    by = {}
    for arm, strength in strength_of_arm.items():
        scores = [strength * y + rng.normal(0, 1, len(y)) for y in labels]
        path = root / f"{arm}_{test}_{variant}.npz"
        np.savez(path, names=np.array([f"v{i}" for i in range(len(labels))]),
                 **{f"score_v{i}": s for i, s in enumerate(scores)}, **{f"label_v{i}": y for i, y in enumerate(labels)},
                 **{f"covered_v{i}": np.ones(len(y), bool) for i, y in enumerate(labels)})
        by[(arm, test, variant)] = {"scores_file": str(path), "auc": frame_level_auc(labels, scores)}
    return by


def test_seed_paired_bootstrap_detects_a_real_gain_and_not_a_null_one(tmp_path):
    rng = np.random.default_rng(0)
    labels = [(rng.random(60) < 0.4).astype(int) for _ in range(30)]
    gain = {"released": 1.0, **{f"standard_s{s}": 1.0 for s in range(3)}, **{f"gau_s{s}": 1.6 for s in range(3)}}
    result = exp019.seed_paired_bootstrap("released", "paper", fake_arm_files(tmp_path, labels, gain, rng), n_resamples=60)
    assert result["diff"] > 3 and result["ci_videos"][0] > 0
    assert len(result["seed_diffs"]) == 3 and result["ci_seeds_t"][0] < result["diff"] < result["ci_seeds_t"][1]

    null = {name: 1.0 for name in gain}
    result = exp019.seed_paired_bootstrap("released", "paper", fake_arm_files(tmp_path, labels, null, rng), n_resamples=60)
    assert result["ci_videos"][0] < 0 < result["ci_videos"][1]


def test_timing_reports_both_models_with_their_parameter_counts(tmp_path, monkeypatch):
    monkeypatch.setattr(exp019, "OUT", tmp_path)  # no trained models there: untrained ones are timed (the cost does not depend on weights)
    result = exp019.timing(repeats=2, calls=3)
    assert set(result["summary_median_ms"]) >= {"standard_s0", "gau_s0"}
    assert result["params"]["standard_s0"] == 616 and result["params"]["gau_s0"] == 5808
    assert json.loads((tmp_path / "timing.json").read_text())["threads"] == 1
