"""Guards the deadline simulation: which frames a single-worker, latest-frame system keeps."""

import numpy as np
import pytest

from edgelab.pipeline.deadline_sim import (block_bootstrap_service, keep_masks_for_videos, simulate_latest_frame,
                                           video_seed)

FPS = 24.0
PERIOD_MS = 1000.0 / FPS


def kept(mask):
    return np.flatnonzero(mask).tolist()


def test_fast_service_keeps_every_frame():
    service = np.full(100, 0.9 * PERIOD_MS)
    assert simulate_latest_frame(service, 100, FPS).all()


def test_service_of_two_and_a_half_periods_keeps_a_pattern_checked_by_hand():
    # frame 0: 0 -> 2.5 (frames 1 and 2 arrived; newest = 2)   frame 2: 2.5 -> 5.0 (frame 5 arrives exactly then)
    # frame 5: 5.0 -> 7.5 (newest = 7)                          frame 7: 7.5 -> 10.0 (newest = 10)
    service = np.full(50, 2.5 * PERIOD_MS)
    assert kept(simulate_latest_frame(service, 13, FPS)) == [0, 2, 5, 7, 10, 12]


def test_frame_that_arrived_while_busy_but_was_superseded_is_dropped():
    # Frame 0 takes 3.5 periods. Frames 1, 2, 3 arrive meanwhile; only the newest (3) is taken, 1 and 2 are dropped.
    service = np.array([3.5] + [0.1] * 10) * PERIOD_MS
    assert kept(simulate_latest_frame(service, 8, FPS)) == [0, 3, 4, 5, 6, 7]


def test_worker_waits_for_the_next_frame_when_idle():
    # Frame 0 takes 0.2 periods; nothing newer has arrived, so frame 1 starts at t = 1 (not at 0.2).
    service = np.full(10, 0.2 * PERIOD_MS)
    assert simulate_latest_frame(service, 10, FPS).all()


def test_first_frame_is_always_processed():
    assert simulate_latest_frame(np.full(5, 10 * PERIOD_MS), 5, FPS)[0]


def test_too_short_service_array_is_an_error():
    with pytest.raises(ValueError):
        simulate_latest_frame(np.full(2, 0.5 * PERIOD_MS), 10, FPS)


def test_block_bootstrap_uses_whole_aligned_blocks_and_is_seeded():
    trace = np.arange(72, dtype=float)  # 3 blocks: 0..23, 24..47, 48..71
    out = block_bootstrap_service(trace, 100, block=24, seed=1)
    assert len(out) == 100
    for start in range(0, 96, 24):  # every full chunk of the output is one aligned block of the trace
        chunk = out[start:start + 24]
        assert chunk[0] % 24 == 0 and np.array_equal(chunk, chunk[0] + np.arange(24))
    np.testing.assert_array_equal(out, block_bootstrap_service(trace, 100, block=24, seed=1))
    assert not np.array_equal(out, block_bootstrap_service(trace, 100, block=24, seed=2))


def test_block_bootstrap_needs_one_full_block():
    with pytest.raises(ValueError):
        block_bootstrap_service(np.ones(10), 5, block=24)


def test_video_seed_is_stable_and_depends_on_both_inputs():
    assert video_seed(0, "01_0014") == video_seed(0, "01_0014")
    assert len({video_seed(0, "01_0014"), video_seed(1, "01_0014"), video_seed(0, "01_0015")}) == 3


def test_keep_masks_for_videos_one_mask_per_video_independent_of_dict_order():
    trace = np.concatenate([np.full(24, 0.5 * PERIOD_MS), np.full(24, 3.0 * PERIOD_MS)])
    counts = {"01_0014": 200, "01_0015": 150}
    masks = keep_masks_for_videos(counts, trace, FPS, seed=0)
    assert {v: len(m) for v, m in masks.items()} == counts
    reordered = keep_masks_for_videos(dict(reversed(counts.items())), trace, FPS, seed=0)
    for video in counts:
        np.testing.assert_array_equal(masks[video], reordered[video])
    assert 0.2 < masks["01_0014"].mean() < 1.0  # a mix of fast and slow blocks drops some, not all
