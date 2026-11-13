"""Label-free coverage: hand examples, and the window rule against STG-NF's own split_pose_to_segments."""

import sys

import numpy as np

from edgelab.pose.segment_coverage import OFFSETS, scored_frames, video_coverage
from edgelab.pose.stgnf_eval import STGNF_ROOT


def test_a_continuous_track_of_26_poses_scores_two_windows():
    # windows start at poses 0 and 1 (STG-NF stops one window early: 26 - 24 = 2 windows)
    assert scored_frames(list(range(100, 126)), offset=12) == {112, 113}
    assert scored_frames(list(range(100, 126)), offset=23) == {123, 124}


def test_a_track_of_24_poses_or_fewer_scores_nothing():
    assert scored_frames(list(range(24)), offset=12) == set()


def test_two_missing_frames_are_allowed_three_are_not():
    two_missing = [f for f in range(30) if f not in (5, 6)]  # 28 poses; window 0 covers frames 0..25 minus 2 gaps
    assert 12 in scored_frames(two_missing, offset=12)  # window starting at frame 0: 24 poses, 22 in [0, 24)
    three_missing = [f for f in range(30) if f not in (2, 3, 4)]
    assert 12 not in scored_frames(three_missing, offset=12)  # window starting at 0 has only 21 poses in [0, 24)


def test_video_coverage_marks_frames_and_ignores_frames_beyond_the_video():
    mask = video_coverage({1: list(range(26)), 2: list(range(50, 76))}, n_frames=70, offset=12)
    assert mask.nonzero()[0].tolist() == [12, 13, 62, 63]
    assert OFFSETS == {"paper": 12, "live": 23}


def test_window_rule_equals_stgnf_split_pose_to_segments():
    sys.path.insert(0, str(STGNF_ROOT))
    from utils.pose_utils import split_pose_to_segments

    rng = np.random.default_rng(0)
    for _ in range(200):
        frames = sorted(rng.choice(80, size=rng.integers(20, 70), replace=False).tolist())
        poses = np.zeros((len(frames), 17, 3))
        _, meta, _ = split_pose_to_segments(poses, [1, frames[0]], [f"{f:04d}" for f in frames], 0, 1, 24,
                                            scene_id="1", clip_id="1", single_score_np=np.zeros(len(frames)))
        theirs = {m[3] + 12 for m in meta}
        assert scored_frames(frames, offset=12) == theirs
