"""Verdict rules of EXP-015 (O1, O2, model choice) and EXP-016 (T1, T2) on made-up rows, checked against the frozen plans."""

import json

import pytest

from edgelab.analysis import exp015, exp016
from edgelab.pose.topdown import IncompleteCacheError


def rows15(s: float, t: float, y: float = 0.60) -> list[dict]:
    auc = {"released": 0.8594, "rtmpose-s": s, "rtmpose-t": t, "yolov8n-pose": y}
    return [{"arm": a, "variant": "paper", "auc": v} for a, v in auc.items()]


def test_o1_o2_thresholds_are_83_and_80_on_the_paper_variant():
    v = exp015.verdicts(rows15(0.8300, 0.7999))
    assert v["O1_pass"] and not v["O2_pass"]
    v = exp015.verdicts(rows15(0.8299, 0.8000))
    assert not v["O1_pass"] and v["O2_pass"]


def test_exp016_uses_rtmpose_s_unless_o2_passes_then_the_better_of_the_two():
    assert exp015.verdicts(rows15(0.84, 0.79))["exp016_pose_model"] == "rtmpose-s"  # O2 fails
    assert exp015.verdicts(rows15(0.84, 0.82))["exp016_pose_model"] == "rtmpose-s"  # O2 passes, s is better
    assert exp015.verdicts(rows15(0.81, 0.82))["exp016_pose_model"] == "rtmpose-t"  # O2 passes, t is better


def rows16(aucs: dict[tuple[int, int], float], model: str = "rtmpose-s") -> list[dict]:
    return [{"config": exp016.config_id("L0", i, k, model), "variant": "paper", "auc": v} for (i, k), v in aucs.items()]


def test_t1_needs_2_points_over_the_68_53_baseline():
    ok = exp016.verdicts(rows16({(640, 1): 0.69, (640, 3): 0.68, (960, 1): 0.7053, (960, 3): 0.70}), "rtmpose-s")
    assert ok["T1_pass"] and ok["best"] == "n960_K1"
    no = exp016.verdicts(rows16({(640, 1): 0.69, (640, 3): 0.68, (960, 1): 0.7052, (960, 3): 0.70}), "rtmpose-s")
    assert not no["T1_pass"]


def test_t2_k3_may_lose_at_most_one_point_per_detector_size():
    v = exp016.verdicts(rows16({(640, 1): 0.70, (640, 3): 0.6901, (960, 1): 0.72, (960, 3): 0.7099}), "rtmpose-s")
    assert v["T2_pass_by_size"] == {"n640": True, "n960": False} and not v["T2_pass"]


def test_invalid_rows_are_ignored_by_verdicts_and_by_the_resume_check(tmp_path, monkeypatch):
    good = rows16({(640, 1): 0.69, (640, 3): 0.68, (960, 1): 0.70, (960, 3): 0.69})
    bad = [{**r, "auc": 0.49, "invalid": "incomplete"} for r in rows16({(960, 1): 0.49, (960, 3): 0.49})]
    v = exp016.verdicts(good + bad, "rtmpose-s")
    assert v["paper_auc_L0"]["n960_K1"] == pytest.approx(70.0)
    with pytest.raises(ValueError, match="960"):  # only invalid n960 rows -> no verdict
        exp016.verdicts(rows16({(640, 1): 0.69, (640, 3): 0.68}) + bad, "rtmpose-s")
    rows_file = tmp_path / "rows.jsonl"
    rows_file.write_text("".join(json.dumps(r) + "\n" for r in good[:1] + bad))
    monkeypatch.setattr(exp016, "ROWS", rows_file)
    assert [r["config"] for r in exp016.read_rows()] == [good[0]["config"]]
    assert len(exp016.read_rows(include_invalid=True)) == 3


def test_run_config_never_scores_an_incomplete_cache(tmp_path, monkeypatch):
    cache = tmp_path / "cache" / "L0_n960"
    cache.mkdir(parents=True)
    for i in range(4):
        (cache / f"v{i}.npz").write_bytes(b"x")
    monkeypatch.setattr(exp016, "DATA", tmp_path)
    monkeypatch.setattr(exp016, "ROWS", tmp_path / "rows.jsonl")
    called = []
    monkeypatch.setattr(exp016, "poses_for_dir", lambda *a, **k: called.append(a))
    monkeypatch.setattr(exp016, "score_pose_dir", lambda *a, **k: called.append(a))
    with pytest.raises(IncompleteCacheError):
        exp016.run_config("L0", 960, 1, "rtmpose-s", 1)
    assert not called and not (tmp_path / "rows.jsonl").exists()
