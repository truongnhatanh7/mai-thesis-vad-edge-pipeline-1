import numpy as np

from edgelab.analysis.exp005 import coverage_all_frames, long_run_frames, select


def test_long_run_frames_keeps_only_long_consecutive_runs():
    track = list(range(0, 50)) + list(range(60, 70))  # one 50-frame run, one 10-frame run
    assert long_run_frames(track, min_run=48) == set(range(50))
    assert long_run_frames(list(range(47)), min_run=48) == set()


def test_coverage_is_label_free_share_of_frames():
    covered = [np.array([True, False, True, True]), np.array([False, False])]
    assert coverage_all_frames(covered) == 3 / 6


def test_select_breaks_ties_towards_no_enhancement():
    assert select({"E0": 0.5, "E1": 0.5, "E2": 0.4}) == "E0"
    assert select({"E0": 0.4, "E1": 0.6, "E2": 0.5}) == "E1"
