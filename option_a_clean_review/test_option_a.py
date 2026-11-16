"""Unit tests for the clean Option A review folder (`step1` -> `step4`).

Run with:
    ../experiments/.venv/bin/pytest test_option_a.py -v
"""

from __future__ import annotations

import numpy as np
import torch

from step1_models import (
    SkeletonNormalizingFlow,
    coco17_to_openpose18,
    ema_normalize_track_n1,
    normalize_window_n0,
)
from step2_tracker_cache import (
    HysteresisQueueGuard,
    OnlineCircularWindowScorer,
    StaticValidityCacheC1,
)
from step3_offline_eval import (
    evaluate_tracks_s0_and_s4,
    smooth_causal_one_sided,
    smooth_offline_two_sided,
)


def test_step1_models_and_causality():
    # 1. Verify COCO-17 -> OpenPose-18 and N0 / N1 normalization shapes
    xy17 = np.random.randn(4, 24, 17, 2).astype(np.float32) * 50.0 + 200.0
    op18 = coco17_to_openpose18(xy17)
    assert op18.shape == (4, 24, 18, 2)

    x_n0 = normalize_window_n0(xy17)
    assert x_n0.shape == (4, 2, 24, 18)

    x_n1 = ema_normalize_track_n1(xy17[0], seg_len=24)
    assert x_n1.shape == (24, 18, 2)

    # 2. Verify LoRaD Causal Flow (`causal=True`) has ZERO future leakage vs. symmetric STG-NF (`causal=False`)
    torch.manual_seed(0)
    lorad = SkeletonNormalizingFlow(n_frames=24, steps=4, causal=True).train()
    _ = lorad(x_n0)
    lorad.eval()
    z_a, nll_a = lorad(x_n0)
    assert nll_a.shape == (4,)

    # Perturb future frames 14..23 and verify causal outputs at frames 0..13 are 100% unchanged (diff == 0.0)
    x_pert = x_n0.clone()
    x_pert[:, :, 14:, :] += 10.0
    z_b, _ = lorad(x_pert)
    assert float(torch.max(torch.abs(z_a[:, :, :14, :] - z_b[:, :, :14, :])).item()) == 0.0


def test_step2_cache_guard_and_circular_window_scorer():
    # 1. Verify C1 Static-Validity Cache skips static frames and refreshes on period
    cache = StaticValidityCacheC1(max_age=3, refresh_period=12)
    static_img = np.full((480, 856, 3), 120, dtype=np.uint8)
    calls = 0

    def fake_detector(img):
        nonlocal calls
        calls += 1
        return (
            np.array([[100.0, 100.0, 150.0, 220.0]], dtype=np.float32),
            np.array([0.85], dtype=np.float32),
            np.ones((1, 17, 3), dtype=np.float32) * 120.0,
            10.0,
        )

    _, _, _, reused1 = cache.check_or_compute(1, static_img, fake_detector)
    _, _, _, reused2 = cache.check_or_compute(2, static_img, fake_detector)
    assert reused1 is False and reused2 is True and calls == 1

    # 2. Verify Hysteresis Queue Guard
    guard = HysteresisQueueGuard(high_ms=400.0, low_ms=150.0)
    assert guard.should_shed(1, 100.0) is False
    assert guard.should_shed(1, 450.0) is True   # Triggers guard on odd frame
    assert guard.should_shed(3, 250.0) is True   # Stays active until < 150 ms
    assert guard.should_shed(5, 100.0) is False  # Disarms below 150 ms

    # 3. Verify OnlineCircularWindowScorer emits scores once 24 frames arrive
    lorad = SkeletonNormalizingFlow(n_frames=24, steps=4, causal=True).train()
    _ = lorad(torch.randn(2, 2, 24, 18))
    scorer = OnlineCircularWindowScorer(lorad.eval(), seg_len=24)
    kp = np.ones((17, 3), dtype=np.float32) * 150.0
    for f in range(23):
        assert len(scorer.step(f, {1: kp})) == 0
    scores_24 = scorer.step(23, {1: kp})
    assert len(scores_24) == 1


def test_step3_offline_s0_vs_causal_s4():
    # Verify causal smoother is prefix-invariant (future changes don't affect past)
    s1 = np.zeros(50, dtype=np.float64)
    s2 = s1.copy()
    s2[30:] = 100.0
    assert np.allclose(smooth_causal_one_sided(s1)[:30], smooth_causal_one_sided(s2)[:30])
    assert not np.allclose(smooth_offline_two_sided(s1)[:30], smooth_offline_two_sided(s2)[:30])

    # Verify evaluate_tracks_s0_and_s4 runs cleanly on a synthetic clip
    lorad = SkeletonNormalizingFlow(n_frames=24, steps=4, causal=True).train()
    _ = lorad(torch.randn(2, 2, 24, 18))
    frames = np.arange(40, dtype=np.int64)
    kp = np.random.randn(40, 17, 3).astype(np.float32) * 10.0 + 200.0
    gt = np.zeros(40, dtype=np.int64)
    gt[25:] = 1
    res = evaluate_tracks_s0_and_s4([{1: (frames, kp)}], [gt], lorad.eval())
    assert 0.0 <= res["auc_s0"] <= 100.0 and 0.0 <= res["auc_s4"] <= 100.0
