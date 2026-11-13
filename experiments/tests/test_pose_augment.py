"""EXP-018 C1 augmentation: STG-NF normalisation, random keep masks, window cutting, provenance channel."""

import numpy as np
import torch

from edgelab.pose.augment import make_batch, random_keep_masks, select_windows, to_flow_input
from edgelab.pose.segments import build_bank
from edgelab.pose.stgnf_eval import _import_stgnf
from pose_toy import write_toy_dir

_import_stgnf()
from dataset import gen_dataset  # noqa: E402
from utils.data_utils import normalize_pose, trans_list  # noqa: E402


def their_inputs(directory, flip: bool):
    segs, *_ = gen_dataset(str(directory), seg_stride=6, seg_len=24, ret_keys=True, ret_global_data=False)
    if flip:
        segs = np.stack([trans_list[1](s) for s in segs])  # PoseTransform: x -> -x
    return normalize_pose(segs.transpose((0, 2, 3, 1)).copy())[..., :2].transpose(0, 3, 1, 2)  # (N, 2, T, 18)


def test_our_batches_equal_stgnf_dataset_inputs_with_and_without_flip(tmp_path):
    directory = write_toy_dir(tmp_path)
    bank = build_bank(directory, seg_len=24, stride=6, ctx=16)
    rng = np.random.default_rng(0)
    for flip in (False, True):
        x, provenance = make_batch(bank, np.arange(len(bank)), None, None, rng, np.full(len(bank), flip))
        assert provenance is None
        assert np.allclose(x.numpy(), their_inputs(directory, flip), atol=1e-4)


def test_keep_masks_drop_more_at_higher_rate_in_bursts_and_are_reproducible():
    masks = random_keep_masks(600, 42, np.random.default_rng(0))
    assert masks.shape == (600, 42) and masks.dtype == bool
    assert np.array_equal(masks, random_keep_masks(600, 42, np.random.default_rng(0)))
    drop = 1 - masks.mean(axis=1)
    assert drop.min() == 0 and drop.max() > 0.7  # rate ~ U(0, .85): from nothing dropped to most dropped
    low, high = drop[drop < 0.3], drop[drop > 0.6]
    assert len(low) > 50 and len(high) > 50
    # bursts: in heavily dropped rows, a dropped frame is usually followed by another dropped frame
    heavy = masks[drop > 0.6]
    assert (~heavy[:, 1:] & ~heavy[:, :-1]).sum() / (~heavy[:, :-1]).sum() > 0.5


def grid_inputs(valid_columns, predicted_columns=(), ctx=2, seg_len=6, length=None):
    length = length or ctx + seg_len + 2
    xy = torch.arange(length, dtype=torch.float32)[None, :, None, None].expand(1, length, 17, 2).clone()
    valid = torch.zeros(1, length, dtype=torch.bool)
    valid[0, list(valid_columns)] = True
    predicted = torch.zeros(1, length, dtype=torch.bool)
    predicted[0, list(predicted_columns)] = True
    return xy, valid, predicted


def test_select_windows_takes_the_first_T_poses_from_the_window_start():
    xy, valid, predicted = grid_inputs(range(0, 10))  # context and window all valid
    windows, provenance, kept = select_windows(xy, valid, predicted, ctx=2, seg_len=6)
    assert kept.tolist() == [0] and windows[0, :, 0, 0].tolist() == [2, 3, 4, 5, 6, 7]  # columns 2..7
    assert provenance.tolist() == [[1.0] * 6]


def test_select_windows_allows_two_missing_frames_but_not_three_and_marks_predicted():
    xy, valid, predicted = grid_inputs([0, 1, 2, 3, 5, 6, 8, 9], predicted_columns=[5])  # columns 4 and 7 missing: 2 holes
    windows, provenance, kept = select_windows(xy, valid, predicted, ctx=2, seg_len=6)
    assert kept.tolist() == [0]
    assert windows[0, :, 0, 0].tolist() == [2, 3, 5, 6, 8, 9]  # compacted: the next 6 poses
    assert provenance[0].tolist() == [1, 1, 0, 1, 1, 1]  # column 5 was predicted
    xy, valid, predicted = grid_inputs([0, 1, 2, 3, 6, 9])  # four holes inside the window range
    assert len(select_windows(xy, valid, predicted, ctx=2, seg_len=6)[2]) == 0


def test_select_windows_starts_at_the_first_valid_frame_after_the_context():
    xy, valid, predicted = grid_inputs([0, 1, 3, 4, 5, 6, 7, 8, 9])  # the window's first frame (column 2) is gone
    windows, _, kept = select_windows(xy, valid, predicted, ctx=2, seg_len=6)
    assert kept.tolist() == [0] and windows[0, :, 0, 0].tolist() == [3, 4, 5, 6, 7, 8]


def test_make_batch_with_fill_gives_a_provenance_channel_that_matches_the_filled_poses(tmp_path):
    bank = build_bank(write_toy_dir(tmp_path), seg_len=24, stride=6, ctx=16)
    rng = np.random.default_rng(1)
    rows = np.tile(np.arange(len(bank)), 40)
    x, provenance = make_batch(bank, rows, "cv", None, rng, np.zeros(len(rows), dtype=bool))
    assert provenance.shape == (len(x), 24) and x.shape[1:] == (2, 24, 18)
    assert set(provenance.unique().tolist()) <= {0.0, 1.0}
    assert 0 < len(x) < len(rows)  # heavy drops remove some windows entirely
    assert (provenance == 0).any() and (provenance == 1).any()
    assert torch.isfinite(x).all()
