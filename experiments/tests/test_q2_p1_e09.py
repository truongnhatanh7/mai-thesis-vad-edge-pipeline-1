"""Unit tests for Q2-P1-E09 (Hypothesis 1 MS-CTFlow-Bottleneck, Hypothesis 2 Two-Tier MPC-LoRaD, Hypothesis 3 Green Kinematic Training)."""

from __future__ import annotations

import numpy as np
import torch

from edgelab.analysis import q2_p1_e09 as e09


def test_msctflow_causality_and_invertibility():
    torch.manual_seed(0)
    model = e09.MSCTFlow(seg_len=24, n_blocks=8, kernel_t=3).train()
    x = torch.randn(4, 2, 24, 18)
    _ = model(x)
    model.eval()
    z, nll = model(x)
    assert z.shape == (4, 2, 24, 18)
    assert nll.shape == (4,)

    # Exact analytical invertibility
    x_rec = model.inverse(z)
    assert float(torch.max(torch.abs(x - x_rec)).item()) < 1e-4

    # Strict temporal causality: perturbing frames >= 12 cannot change latent outputs at frames < 12
    x_pert = x.clone()
    x_pert[:, :, 12:, :] += 10.0
    z_pert, _ = model(x_pert)
    assert float(torch.max(torch.abs(z[:, :, :12, :] - z_pert[:, :, :12, :])).item()) == 0.0


def test_h2_two_tier_mpc_kinematic_extrapolation():
    # Synthetic clip with 100 frames and 1 track present on frames 10..60 with steady constant velocity
    class _StubCtx:
        n_frames = [100]

    frames = np.arange(10, 60, dtype=np.int64)
    kp = np.zeros((len(frames), 17, 3), dtype=np.float32)
    for idx, t in enumerate(frames):
        kp[idx, :, 0] = 100.0 + 1.5 * idx
        kp[idx, :, 1] = np.linspace(100.0, 180.0, 17)
        kp[idx, :, 2] = 0.9
    tracks = [{1: (frames, kp)}]

    mod_tracks, stats = e09.apply_two_tier_mpc_to_tracks(tracks, _StubCtx(), eps_disp_px=0.05)
    assert stats["tier1_sleep_pct"] == 50.0  # 50 empty frames out of 100
    assert stats["tier2_extrap_pct"] > 50.0  # steady constant velocity is extrapolated
    assert stats["total_saved_pct"] > 75.0
    assert len(mod_tracks) == 1


def test_h3_kinematic_losses_and_refiner():
    torch.manual_seed(42)
    x = torch.randn(6, 2, 24, 18)
    refiner = e09.SpatioTemporalKinematicRefiner(num_joints=18, hidden=16)
    x_hat = refiner(x)
    # At initialization, refiner output equals input x
    assert float(torch.max(torch.abs(x_hat - x)).item()) == 0.0
    l_bone, l_smooth, l_rec = e09.compute_kinematic_losses(x_hat, x)
    assert float(l_bone.item()) >= 0.0
    assert float(l_smooth.item()) >= 0.0
    assert float(l_rec.item()) == 0.0
