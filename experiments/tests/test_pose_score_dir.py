"""Guards coverage and the two variants' frame placement in score_dir, using fake STG-NF segment scores."""

from types import SimpleNamespace

import numpy as np

from edgelab.pose.score_dir import coverage, variant_scores


def _fake_segments(tmp_path):
    """One 60-frame clip (frames 30-39 anomalous). One person, segments starting at frames 0..9 (24 frames each)."""
    gt = np.zeros(60, dtype=int)
    gt[30:40] = 1
    np.save(tmp_path / "01_0001.npy", gt)
    metadata = np.array([[1, 1, 1, start] for start in range(10)])
    normality = np.arange(10, dtype=float)  # segment k has normality k
    return SimpleNamespace(normality=normality, metadata=metadata)


def test_coverage_counts_frames_with_a_scored_person():
    covered = [np.array([True, True, False, False])]
    labels = [np.array([0, 1, 1, 0])]
    assert coverage(covered, labels) == (0.5, 0.5)  # 1 of 2 anomalous, 1 of 2 normal


def test_paper_variant_scores_land_on_middle_frame(tmp_path):
    segments = _fake_segments(tmp_path)
    names, labels, scores, covered = variant_scores(segments, tmp_path, "middle", "none")
    assert names == ["01_0001"]
    assert covered[0].nonzero()[0].tolist() == list(range(12, 22))  # start + 12
    assert scores[0][12] == -0.0 and scores[0][21] == -9.0  # score = -normality
    assert scores[0][0] == -9.0  # frame with nobody: the most normal score in the dataset (= -max normality)


def test_live_variant_scores_land_on_last_frame_of_the_segment(tmp_path):
    segments = _fake_segments(tmp_path)
    _, _, _, covered = variant_scores(segments, tmp_path, "end", "none")
    assert covered[0].nonzero()[0].tolist() == list(range(23, 33))  # start + 23


def test_live_smoothing_is_causal(tmp_path):
    segments = _fake_segments(tmp_path)
    _, _, scores, _ = variant_scores(segments, tmp_path, "end", "causal_half_gaussian")
    changed = SimpleNamespace(normality=segments.normality.copy(), metadata=segments.metadata)
    changed.normality[8] = -50.0  # the segment scored on frame 31; the dataset's most normal score (segment 9) is unchanged
    _, _, scores_changed, _ = variant_scores(changed, tmp_path, "end", "causal_half_gaussian")
    np.testing.assert_allclose(scores[0][:31], scores_changed[0][:31])
