"""Guards frame-score assembly and smoothing: same numbers as STG-NF's own code; causal filters never look ahead."""

import argparse
import os
import sys

import numpy as np
import pytest

from edgelab.pose.smoothing import (STGNF_EQUIVALENT_SIGMA, causal_ewma, causal_half_gaussian, offline_gaussian)
from edgelab.pose.stgnf_eval import STGNF_ROOT, assemble_clip_scores

needs_stgnf = pytest.mark.skipif(not (STGNF_ROOT / "utils/scoring_utils.py").exists(), reason="STG-NF not cloned")


def _synthetic_clip(tmp_path, rng):
    gt_dir = tmp_path / "gt"
    gt_dir.mkdir()
    lengths = {(1, 14): 120, (2, 3): 90, (3, 5): 60}
    metadata, normality = [], []
    for (scene, clip), n in lengths.items():
        gt = np.zeros(n, dtype=int)
        gt[n // 2: n // 2 + 20] = 1
        np.save(gt_dir / f"{scene:02d}_{clip:04d}.npy", gt)
        if clip == 5:
            continue  # a clip in which nobody is ever detected
        for person in (1, 2):
            for start in range(0, n - 24, 1 + person):
                metadata.append([scene, clip, person, start])
                normality.append(rng.normal())
    return gt_dir, np.array(metadata), np.array(normality)


@needs_stgnf
def test_assembly_matches_stgnf_get_dataset_scores(tmp_path):
    gt_dir, metadata, normality = _synthetic_clip(tmp_path, np.random.default_rng(1))
    names, labels, scores = assemble_clip_scores(normality, metadata, gt_dir)

    sys.path.insert(0, str(STGNF_ROOT))
    from utils import scoring_utils
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        os.makedirs("data/ShanghaiTech/gt", exist_ok=True)
        if not os.path.exists("data/ShanghaiTech/gt/test_frame_mask"):
            os.symlink(gt_dir, "data/ShanghaiTech/gt/test_frame_mask")
        gt, theirs = scoring_utils.get_dataset_scores(normality.copy(), metadata,
                                                      args=argparse.Namespace(dataset="ShanghaiTech", seg_len=24))
    finally:
        os.chdir(cwd)
    for ours, their, label, their_gt in zip(scores, theirs, labels, gt):
        np.testing.assert_allclose(ours, -their)
        np.testing.assert_array_equal(label, 1 - their_gt)


@needs_stgnf
def test_offline_gaussian_equals_stgnf_smooth_scores():
    sys.path.insert(0, str(STGNF_ROOT))
    from utils.scoring_utils import smooth_scores
    x = np.random.default_rng(2).normal(size=300)
    np.testing.assert_allclose(offline_gaussian(x), smooth_scores([x.copy()])[0])


@pytest.mark.parametrize("smoother", [causal_half_gaussian, causal_ewma])
def test_causal_filters_ignore_future_frames(smoother):
    x = np.random.default_rng(3).normal(size=200)
    changed = x.copy()
    changed[120:] += 100.0
    np.testing.assert_allclose(smoother(x)[:120], smoother(changed)[:120])


@pytest.mark.parametrize("smoother", [causal_half_gaussian, causal_ewma, offline_gaussian])
def test_smoothers_keep_constant_signal(smoother):
    np.testing.assert_allclose(smoother(np.full(80, 2.5)), 2.5)


def test_equivalent_sigma_is_sqrt_91():
    assert STGNF_EQUIVALENT_SIGMA == pytest.approx(91 ** 0.5)
