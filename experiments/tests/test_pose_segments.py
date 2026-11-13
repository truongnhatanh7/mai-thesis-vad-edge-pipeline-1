"""SegmentBank cuts the same windows as third_party/STG-NF (dataset.gen_dataset) and keeps the right context."""

import numpy as np

from edgelab.pose.segments import build_bank, window_starts
from edgelab.pose.stgnf_eval import _import_stgnf
from pose_toy import write_toy_dir

_import_stgnf()
from dataset import gen_dataset  # noqa: E402


def test_windows_are_the_ones_stgnf_cuts(tmp_path):
    directory = write_toy_dir(tmp_path)
    theirs, meta, *_ = gen_dataset(str(directory), seg_stride=6, seg_len=24, ret_keys=True, ret_global_data=False)
    bank = build_bank(directory, seg_len=24, stride=6, ctx=16)
    assert len(bank) == len(meta) == len(theirs)
    # the poses of each window: the first 24 poses of the grid from column ctx on
    for window in range(len(bank)):
        rows = bank.grid[window, bank.ctx:]
        rows = rows[rows >= 0][:24]
        mine = bank.xy[rows]  # (24, 17, 2)
        their_xy = theirs[window, :2]  # (2, 24, 18) after their coco-18 conversion: compare the 17-keypoint part via the neck rule
        assert mine.shape == (24, 17, 2)
        neck = 0.5 * (mine[:, 5] + mine[:, 6])
        order = [0, 17, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3]
        coco18 = np.concatenate([mine, neck[:, None]], axis=1)[:, order]  # (24, 18, 2)
        assert np.allclose(coco18.transpose(2, 0, 1), their_xy, atol=1e-3)


def test_window_start_rule_keeps_at_most_two_missing_frames():
    frames = np.arange(60)
    assert list(window_starts(frames, 24, 6)) == [0, 6, 12, 18, 24, 30]
    assert len(window_starts(np.arange(24), 24, 6)) == 0  # a track of exactly one window length gives none (STG-NF)
    gappy = np.delete(np.arange(60), [10, 11, 12])  # 3 frames missing inside the first windows
    starts = window_starts(gappy, 24, 6)
    assert list(starts) == [12, 18, 24, 30]  # windows 0 and 6 have 3 holes within their 24 frames


def test_grid_has_context_before_the_window_and_marks_absent_frames(tmp_path):
    bank = build_bank(write_toy_dir(tmp_path), seg_len=24, stride=6, ctx=16)
    assert bank.grid.shape[1] == 16 + 24 + 2
    first = bank.grid[0]  # person 1 of clip 01_001, starts at frame 0: no context
    assert (first[:16] == -1).all() and (first[16:] >= 0).all()
    later = next(g for g in bank.grid if (g[:16] >= 0).all())  # a window with a full 16-frame context
    assert np.all(np.diff(later[later >= 0]) == 1)  # context + window are consecutive poses of one track
