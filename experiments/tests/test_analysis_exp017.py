"""EXP-017 verdict rules Y1-Y5 on made-up numbers, checked against the frozen plan (plan §3)."""

import pytest

from edgelab.analysis import exp010, exp017


def test_y2_needs_70_03_and_a_paired_ci_above_zero():
    assert exp017.y2(70.03, 0.01)["pass"]
    assert not exp017.y2(70.02, 1.0)["pass"]  # threshold = 68.53 + 1.5
    assert not exp017.y2(75.0, -0.01)["pass"]  # CI must exclude 0 from below


def test_y3_allows_half_a_point_of_transfer_loss():
    assert exp017.y3(55.26)["pass"] and not exp017.y3(55.25)["pass"]


def test_y4_ratio_limit_is_1_10():
    assert exp017.y4(13.64, 12.4)["pass"] and not exp017.y4(13.65, 12.4)["pass"]


def test_y5_needs_a_win_at_every_slowdown():
    assert exp017.y5({1: 67.0, 3: 66.0}, {1: 66.0, 3: 65.9})["pass"]
    assert not exp017.y5({1: 67.0, 3: 65.9}, {1: 66.0, 3: 65.9})["pass"]  # tie at s = 3 is not a win


def test_y1_both_small_bins_must_improve():
    def table(a, b):
        mk = lambda r: {"persons": 1000, "found": int(1000 * r)}  # noqa: E731
        return {"new": {"48-64": mk(a), "64-96": mk(b)}, "old": {"48-64": mk(0.5), "64-96": mk(0.8)}}

    assert exp017.y1(table(0.6, 0.9), "new", "old")["pass"]
    assert not exp017.y1(table(0.6, 0.8), "new", "old")["pass"]


def test_delivered_uses_exp010_with_seed_1_and_the_two_slowdowns(tmp_path, monkeypatch):
    saved = (exp010.SEED, exp010.OUT, exp010.ROWS, exp010.MLFLOW_EXPERIMENT, exp010.EXPERIMENT_DIR, exp010.GROUPS,
             exp010.trace_for, exp010.SLOWDOWNS)
    try:
        monkeypatch.setattr(exp017, "DELIVERED", tmp_path / "delivered")
        exp017.configure_exp010()
        assert exp010.SEED == 1 and exp010.SLOWDOWNS == (1, 3) and exp010.MLFLOW_EXPERIMENT == "EXP-017"
        assert exp010.ROWS == tmp_path / "delivered" / "rows.jsonl"
        assert set(exp010.GROUPS) == set(exp017.DELIVERED_GROUPS)
        dataset, cache, trace_id = exp010.GROUPS["SHT-L0-26n"]
        assert dataset == "ShanghaiTech" and cache.name == "L0_E0" and "exp017" in str(cache) and trace_id == "yolo26n-ort-fp32-R640"
        assert exp010.GROUPS["SHT-L0-8n"][1].parts[-3:] == ("exp003", "cache", "L0_E0")
    finally:
        (exp010.SEED, exp010.OUT, exp010.ROWS, exp010.MLFLOW_EXPERIMENT, exp010.EXPERIMENT_DIR, exp010.GROUPS,
         exp010.trace_for, exp010.SLOWDOWNS) = saved


def test_score_path_and_paired_names_follow_gate_sweep_layout(tmp_path):
    assert exp017.score_path(tmp_path, "L0_E0-o2m", 0.25, "paper") == tmp_path / "scores/L0_E0-o2m_c0.25_botsort/frame_scores_paper.npz"
    rows = [{"cache": "L0_E0", "c": 0.25, "variant": "paper", "auc": 0.7}]
    assert exp017.row(rows, "L0_E0", 0.25, "paper")["auc"] == pytest.approx(0.7) and exp017.row(rows, "L2_E0", 0.25, "paper") is None
