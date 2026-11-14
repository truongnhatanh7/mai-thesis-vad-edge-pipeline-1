"""Q2-P1-E06 quality: causal flow has zero future leakage, parameter counts, N1 = running normalisation, streaming == batch == online engine,
plan / config consistency and (when produced) the outputs."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from edgelab.models import lorad
from edgelab.models.stgnf_flow import build_flow, count_parameters, frame_nll, random_pose_batch
from edgelab.analysis import q2_p1_e06 as e06

EXP = Path(__file__).resolve().parents[1] / "Q2-P1-E06_lorad_causal_flow"


def _flow(kind, L=24, seed=0):
    torch.manual_seed(seed)
    flow = build_flow(kind, L)
    lorad.mark_actnorm(flow)
    # random (non-degenerate) weights so every path carries signal
    with torch.no_grad():
        for p in flow.parameters():
            p.add_(0.05 * torch.randn_like(p))
    return flow.eval()


def _track(n=70, seed=0):
    rng = np.random.default_rng(seed)
    frames = np.arange(n, dtype=np.int64)
    base = np.stack([rng.uniform(100, 700, 17), rng.uniform(100, 400, 17), np.ones(17)], axis=1)
    kp = np.repeat(base[None], n, 0).astype(np.float32)
    kp[:, :, :2] += np.cumsum(rng.normal(0, 2, (n, 1, 2)), axis=0).astype(np.float32) + rng.normal(0, 1, (n, 17, 2)).astype(np.float32)
    return frames, kp


def test_parameter_counts():
    assert count_parameters(build_flow("graph", 24)) == 616
    assert count_parameters(build_flow("causal", 24)) == 616  # the causal temporal conv keeps STG-NF's size
    assert count_parameters(build_flow("causal-lite", 24)) == 408


@pytest.mark.parametrize("L", [16, 24])
def test_causal_flow_has_no_future_leakage(L):
    causal, acausal = _flow("causal", L), _flow("graph", L)
    x = random_pose_batch(6, L, seed=2)
    x2 = x.clone()
    t0 = L // 2
    x2[:, :, t0:] += 3.0 * torch.randn_like(x2[:, :, t0:])
    with torch.no_grad():
        d_c = (frame_nll(causal, x)[:, :t0] - frame_nll(causal, x2)[:, :t0]).abs().max().item()
        d_a = (frame_nll(acausal, x)[:, :t0] - frame_nll(acausal, x2)[:, :t0]).abs().max().item()
        mean_gap = (frame_nll(causal, x).mean(1) - causal(x)[1]).abs().max().item()
    assert d_c < 1e-5 and d_a > 1e-3  # the symmetric STG-NF does leak, the causal one does not
    assert mean_gap < 1e-4  # the per-frame shares regroup the window NLL exactly


def test_ema_normalisation_is_causal_and_never_changes_a_pose():
    _, kp = _track()
    alpha = lorad.ema_alpha(24)
    assert alpha == pytest.approx(2 / 25)
    full = lorad.ema_normalize_track(kp[:, :, :2], alpha)
    prefix = lorad.ema_normalize_track(kp[:40, :, :2], alpha)
    np.testing.assert_allclose(full[:40], prefix, atol=1e-6)  # a pose's normalised value never depends on later poses
    kp2 = kp.copy()
    kp2[40:, :, :2] += 100
    np.testing.assert_allclose(lorad.ema_normalize_track(kp2[:, :, :2], alpha)[:40], full[:40], atol=1e-6)
    online = lorad.EmaNormalizer(alpha)
    got = np.stack([online.update(k[:, :2]) for k in kp])
    np.testing.assert_allclose(got, full, atol=1e-5)


def test_stream_equals_batch_equals_online_engine():
    L = 24
    flow = _flow("causal", L)
    frames, kp = _track(80, seed=4)
    stream = lorad.FlowScorer(flow, "N1", L, "stream").score_tracks([(frames, kp)])[0]
    online = lorad.OnlineLoRaD(flow, "N1", L, mode="stream")
    got = [v for f, k in zip(frames, kp) if (v := online.update(1, int(f), k)) is not None]
    assert len(got) == len(stream) == len(lorad.starts_of(frames, L)) > 0
    np.testing.assert_allclose(got, stream, atol=1e-4)  # O(1) ring-buffer evaluation == whole-track causal evaluation
    window = lorad.FlowScorer(flow, "N1", L, "window").score_tracks([(frames, kp)])[0]
    assert np.abs(window - stream).max() < 5.0  # window mode (per-window flow pass) differs only by the zero-padding of the first k - 1 frames
    o0 = lorad.OnlineLoRaD(flow, "N0", L, mode="window")
    got0 = [v for f, k in zip(frames, kp) if (v := o0.update(1, int(f), k)) is not None]
    ref0 = lorad.FlowScorer(flow, "N0", L, "window").score_tracks([(frames, kp)])[0]
    np.testing.assert_allclose(got0, ref0, atol=1e-4)


def test_window_rule_matches_e00():
    from edgelab.analysis import q2_p1_e00 as e00

    frames = np.r_[np.arange(0, 30), np.arange(33, 70, 2), np.arange(90, 130)]
    for L in (16, 24):
        np.testing.assert_array_equal(lorad.starts_of(frames, L), e00.complete_window_starts(frames, L))


def test_arms_and_jobs():
    assert lorad.lorad_kind("A2") == ("causal", "N0") and lorad.lorad_kind("A3") == ("causal", "N1") and lorad.lorad_kind("A1") == ("graph", "N0")
    jobs = e06.jobs()
    assert len(jobs) == 20 and len(set(jobs)) == 20 and ("A2L", "ShanghaiTech", 0) in jobs


def test_config_and_plan_consistency():
    cfg = yaml.safe_load((EXP / "config.yaml").read_text())
    assert cfg["seg_len"] == e06.SEG_LEN and cfg["train"]["epochs"] == 10 and cfg["seed_list"] == list(e06.SEEDS)
    assert cfg["train"]["temporal_kernel"] == {"ShanghaiTech": 24 // 2 + 1, "UBnormal": 16 // 2 + 1}
    plan = (EXP / "plan.md").read_text()
    for rule in ("C1", "R1", "R2", "R3", "R4", "R5"):
        assert rule in plan


@pytest.mark.skipif(not (EXP / "summary.json").exists(), reason="E06 not finished")
def test_outputs():
    s = json.loads((EXP / "summary.json").read_text())
    assert s["checks"]["causality"]["C1_pass"] is True
    assert s["selected_for_e07"]["arm"] in ("A2", "A3")
    assert (EXP / "results.csv").exists() and (EXP / "report.md").exists()
