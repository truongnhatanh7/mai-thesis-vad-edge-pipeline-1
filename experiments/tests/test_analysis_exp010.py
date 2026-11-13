"""EXP-010 helpers: hold-last scoring of FILL-HOLD (vs FILL-NORMAL), and the 8s trace scaling."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from edgelab.analysis.exp010 import fill_variant_scores, hold_last_finite, load_trace_8s, scale_detector_time
from edgelab.pipeline.deadline_replay import load_trace


def test_hold_last_finite_copies_the_last_finite_value_forward():
    values = np.array([np.nan, np.nan, -2.0, np.nan, np.nan, -5.0, np.nan])
    assert hold_last_finite(values, fallback=-1.0).tolist() == [-1.0, -1.0, -2.0, -2.0, -2.0, -5.0, -5.0]


def test_hold_last_finite_with_nothing_finite_is_all_fallback():
    assert hold_last_finite(np.full(3, np.nan), fallback=7.0).tolist() == [7.0, 7.0, 7.0]


def _toy_segments(tmp_path):
    gt = np.zeros(60, dtype=int)
    gt[40:50] = 1
    np.save(tmp_path / "01_0001.npy", gt)
    # one person; windows start at frames 0 and 1 -> live score frames 23 and 24 (offset 23); normality -1 and -3
    metadata = np.array([[1, 1, 1, 0], [1, 1, 1, 1]])
    return SimpleNamespace(normality=np.array([-1.0, -3.0]), metadata=metadata)


def test_fill_hold_blank_frames_take_the_last_score_and_before_the_first_the_most_normal(tmp_path):
    segments = _toy_segments(tmp_path)
    names, labels, scores, covered = fill_variant_scores(segments, tmp_path, "end", "none", "ShanghaiTech", "hold")
    anomaly = scores[0]  # anomaly score = -normality
    assert names == ["01_0001"] and covered[0].sum() == 2 and covered[0][23] and covered[0][24]
    assert (anomaly[:23] == 1.0).all()          # before the first score: most normal value (-1) -> anomaly 1
    assert anomaly[23] == 1.0 and anomaly[24] == 3.0
    assert (anomaly[25:] == 3.0).all()          # after the last score: held, not reset to normal


def test_fill_normal_blank_frames_take_the_most_normal_score(tmp_path):
    segments = _toy_segments(tmp_path)
    _, _, scores, _ = fill_variant_scores(segments, tmp_path, "end", "none", "ShanghaiTech", "normal")
    anomaly = scores[0]
    assert anomaly[24] == 3.0 and (anomaly[25:] == 1.0).all()  # EXP-009 behaviour: back to normal


def test_unknown_blank_rule_is_an_error(tmp_path):
    with pytest.raises(ValueError):
        fill_variant_scores(_toy_segments(tmp_path), tmp_path, "end", "none", "ShanghaiTech", "other")


def test_scale_detector_time_adds_r_minus_one_times_the_detector_part():
    out = scale_detector_time(np.array([50.0, 40.0]), np.array([30.0, 20.0]), r=2.5)
    assert out.tolist() == [50 + 1.5 * 30, 40 + 1.5 * 20]
    assert scale_detector_time(np.array([50.0]), np.array([30.0]), r=1.0).tolist() == [50.0]


def _write_trace(path, n=48):
    pd.DataFrame({"video": ["a"] * n, "service_ms": np.arange(n) + 100.0, "det_pre_ms": 1.0, "det_infer_ms": 10.0,
                  "det_post_ms": 2.0}).to_csv(path, index=False)


def test_load_trace_8s_with_r_one_is_the_plain_trace_and_with_r_two_adds_the_detector_time(tmp_path):
    path = tmp_path / "t.csv"
    _write_trace(path)
    plain = load_trace([path])
    assert np.array_equal(load_trace_8s([path], 1.0), plain)
    assert np.allclose(load_trace_8s([path], 2.0), plain + 13.0)  # detector = 1 + 10 + 2 ms per frame
