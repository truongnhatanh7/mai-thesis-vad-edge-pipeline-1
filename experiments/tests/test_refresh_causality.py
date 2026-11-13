"""Unit tests for refresh controllers, causal delivery, budget/service gates, and all 10 mandatory
preflight invariants in EXP-023.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from edgelab.pipeline.refresh_policy import (
    CausalLiveStgnf,
    CausalScoreDeliverer,
    DetectorBudgetGate,
    GuardedDetectionCache,
    PredictionValidityController,
    RefreshThresholds,
    TimebasePreservingTracker,
    simulate_video_policy,
)
from edgelab.pipeline.stgnf_live import load_model, score_segments
from edgelab.pose.visual_propagation import (
    PersonTrackState,
    PropagatedPoseRecord,
    measure_visual_observer,
)


def _synthetic_sequence(n_frames: int = 40, seed: int = 123):
    """Build synthetic detection cache dicts and textured grayscale frames for testing."""
    rng = np.random.default_rng(seed)
    h, w = 240, 320
    base_bg = rng.integers(50, 110, size=(h, w), dtype=np.uint8)
    base_bg = cv2.GaussianBlur(base_bg, (5, 5), 1.0)

    frames_det = []
    gray_frames = []
    for f in range(n_frames):
        cx = 90.0 + 2.0 * f
        cy = 120.0 + 0.5 * f
        img = base_bg.copy()
        x1, y1 = int(cx - 18), int(cy - 40)
        x2, y2 = int(cx + 18), int(cy + 40)
        yy, xx = np.mgrid[max(0, y1) : min(h, y2), max(0, x1) : min(w, x2)]
        pat = (((xx - x1) // 4 + (yy - y1) // 4) % 2) * 130 + 60
        img[max(0, y1) : min(h, y2), max(0, x1) : min(w, x2)] = pat.astype(np.uint8)
        gray_frames.append(img)

        kp = np.zeros((1, 17, 3), dtype=np.float32)
        kp[0, :, 0] = cx + np.sin(np.arange(17)) * 5.0
        kp[0, :, 1] = np.linspace(cy - 32.0, cy + 32.0, 17)
        kp[0, :, 2] = 0.92
        frames_det.append(
            {
                "boxes": np.array([[x1, y1, x2, y2]], dtype=np.float32),
                "conf": np.array([0.90], dtype=np.float32),
                "cls": np.array([0.0], dtype=np.float32),
                "keypoints": kp,
            }
        )
    return frames_det, np.arange(n_frames, dtype=np.int64), gray_frames


class _DummyStgnfModel:
    """Fast deterministic mock STG-NF model for unit tests that only depends on input tensor values."""

    def __call__(self, x, label=None):
        # x shape: (N, 2, seg_len, 18)
        # Return deterministic NLL proportional to temporal velocity energy
        vel = x[:, :, 1:, :] - x[:, :, :-1, :]
        nll = (vel**2).mean(dim=(1, 2, 3)) * 10.0 - 1.2
        return x, nll


def test_preflight_1_prefix_invariance():
    """Preflight check #1: Modifying future frames (> k) cannot change already emitted live outputs (<= k)."""
    frames_a, frame_ids, grays_a = _synthetic_sequence(36, seed=11)
    frames_b, _, grays_b = _synthetic_sequence(36, seed=11)

    # Corrupt frames 28..35 wildly in sequence B (huge coordinates, violent motion, extra detections)
    for f in range(28, 36):
        grays_b[f] = np.full_like(grays_b[f], 250)
        frames_b[f]["keypoints"][0, :, :2] += 150.0 * ((-1) ** f)
        frames_b[f]["conf"][0] = 0.99

    dummy_model = _DummyStgnfModel()
    for method in ("S1", "S2", "S3", "S4", "S5", "B"):
        res_a = simulate_video_policy(
            video="vid_test",
            method=method,
            budget_ratio=0.5,
            frames_det=frames_a,
            frame_ids=frame_ids,
            stgnf_model=dummy_model,
            gray_loader=lambda idx, g=grays_a: g[idx],
        )
        res_b = simulate_video_policy(
            video="vid_test",
            method=method,
            budget_ratio=0.5,
            frames_det=frames_b,
            frame_ids=frame_ids,
            stgnf_model=dummy_model,
            gray_loader=lambda idx, g=grays_b: g[idx],
        )
        # Prefix 0..27 must be bit-for-bit identical!
        np.testing.assert_allclose(res_a.delivered_scores[:28], res_b.delivered_scores[:28], atol=0.0, rtol=0.0)
        np.testing.assert_array_equal(res_a.observed_mask[:28], res_b.observed_mask[:28])


def test_preflight_2_b_availability_never_backdated():
    """Preflight check #2: No B-interpolated pose or score can appear before the future anchor completes."""
    scorer = CausalLiveStgnf(_DummyStgnfModel(), dataset="ShanghaiTech")
    kp = np.ones((17, 3), dtype=np.float64) * 50.0
    kp[:, 2] = 0.9
    future_tagged_rec = PropagatedPoseRecord(
        frame_idx=1,
        track_id=1,
        keypoints=kp,
        score=0.9,
        box=np.array([40.0, 20.0, 60.0, 80.0]),
        age=1,
        mode="B-linear",
        valid_joint_fraction=1.0,
        available_at_frame=4,
        available_at_time_ms=180.0,
    )
    # Trying to ingest a B-interpolated pose at frame 1 (before anchor frame 4) must raise ValueError!
    with pytest.raises(ValueError, match="Causality violation"):
        scorer.ingest_poses(current_frame=1, records=[future_tagged_rec])

    # In a full B simulation at K=4 (anchors at 0, 4, 8, ..., 24), frames 21, 22, 23 are only filled when
    # anchor 24 completes, so no window can become scorable on unobserved frames 22 or 23!
    frames_det, frame_ids, _ = _synthetic_sequence(30, seed=5)
    res_b = simulate_video_policy(
        video="vid_b",
        method="B",
        budget_ratio=0.25,
        frames_det=frames_det,
        frame_ids=frame_ids,
        stgnf_model=_DummyStgnfModel(),
    )
    for f in range(len(frames_det)):
        if not res_b.observed_mask[f]:
            # Unobserved frames under B never emit a new segment score before the next anchor arrives
            assert not res_b.scorable_mask[f], f"B-linear emitted a score on unobserved frame {f} before anchor!"


def test_preflight_3_detector_cache_access_guard():
    """Preflight check #3: Decisions cannot inspect detections on frames not selected for inference."""
    frames_det, frame_ids, _ = _synthetic_sequence(10, seed=1)
    guard = GuardedDetectionCache(frames_det, frame_ids, c=0.25)

    with pytest.raises(PermissionError, match="Illegal detector cache access"):
        guard.authorize_and_fetch(frame_idx=1, decision_granted=False)

    det0 = guard.authorize_and_fetch(frame_idx=0, decision_granted=True)
    assert len(det0["boxes"]) == 1
    assert guard.accessed_frames == [0]


def test_preflight_4_and_5_tracker_timebase_and_no_resurrection():
    """Preflight checks #4 & #5: Skipped frames do not compress elapsed time; lost tracks expire in real time."""
    trk = TimebasePreservingTracker(c=0.25, max_confirm_gap=6, max_frames_lost=10)
    frames_det, _, _ = _synthetic_sequence(30, seed=9)

    # Observe frame 0 and frame 2 -> track 1 confirmed and active
    r0 = trk.observe(0, frames_det[0])
    assert len(r0) == 1
    r2 = trk.observe(2, frames_det[2])
    assert len(r2) == 1
    tid = r2[0][0]

    # Now person disappears after frame 2; step unobserved frames up to frame 15 (> max_frames_lost=10 real frames)
    # Even though tracker.observe() was only called twice, 13 real frames have elapsed!
    trk.step_unobserved(15)
    assert trk.tracker.frame_id == 16
    assert len(trk.tracker.tracked_stracks) == 0
    assert len(trk.tracker.lost_stracks) == 0

    # If person reappears at frame 16, it cannot resurrect the expired track_id `tid`
    r16 = trk.observe(16, frames_det[2])
    for new_tid, _, _, _ in r16:
        assert new_tid != tid


def test_preflight_6_and_7_new_person_entry_and_invalidity_trigger():
    """Preflight checks #6 & #7: Exterior person entry and flow invalidity trigger detector refresh."""
    ctrl = PredictionValidityController(stride_k=4, thresholds=RefreshThresholds())
    base = np.full((240, 320), 100, dtype=np.uint8)

    # Check #7: Track exists on untextured blank image -> flow fails completely -> must trigger excessive_flow_invalidity
    kp = np.ones((17, 3), dtype=np.float64) * 120.0
    kp[:, 2] = 0.9
    tracks = {1: PersonTrackState(1, 0, kp, 0.9, np.array([100.0, 80.0, 140.0, 160.0]))}
    obs_invalid = measure_visual_observer(base, base, tracks, frame_idx=1)
    req_inv, reason_inv = ctrl.decide(
        frame_idx=1,
        last_det_frame=0,
        tracks=tracks,
        tokens_available=1.2,
        observer_out=obs_invalid,
    )
    assert req_inv is True
    assert reason_inv == "excessive_flow_invalidity"

    # Check #6: Person enters outside existing box [100..140, 80..160] at right side [240..290, 60..180]
    rng = np.random.default_rng(55)
    tex1 = cv2.GaussianBlur(rng.integers(40, 120, size=(240, 320), dtype=np.uint8), (5, 5), 1.0)
    tex2 = tex1.copy()
    tex2[40:200, 220:300] = 245  # large new entrant outside existing track box
    obs_ext = measure_visual_observer(tex1, tex2, tracks, frame_idx=1)
    req_ext, reason_ext = ctrl.decide(
        frame_idx=1,
        last_det_frame=0,
        tracks=tracks,
        tokens_available=1.2,
        observer_out=obs_ext,
    )
    assert req_ext is True
    assert reason_ext == "exterior_new_person"


def test_preflight_8_cost_integrity_and_busy_until_gate():
    """Preflight check #8: Every stage charges cost, and physical busy_until_ms rejects overlapping requests."""
    gate = DetectorBudgetGate(budget_ratio=0.5, fps=24.0, burst_capacity=2.0, initial_tokens=2.0)
    # Frame 0 arrives at 0 ms, detector takes 50 ms (> 41.67 ms frame interval!) -> busy until 51 ms
    exec0, rej0, _, _, comp0 = gate.try_acquire(frame_idx=0, requested=True, decision_ready_ms=1.0, detector_service_ms=50.0)
    assert exec0 is True
    assert rej0 == "none"
    assert comp0 == pytest.approx(51.0)

    # Frame 1 arrives at 41.67 ms, decision ready at 42.5 ms < 51.0 ms -> MUST be rejected as detector_busy
    # even though tokens >= 1.0!
    exec1, rej1, tok_b1, _, _ = gate.try_acquire(frame_idx=1, requested=True, decision_ready_ms=42.5, detector_service_ms=50.0)
    assert tok_b1 >= 1.0
    assert exec1 is False
    assert rej1 == "detector_busy"
    assert gate.rejected_busy == 1

    # Verify cost charging in simulate_video_policy for S5
    frames_det, frame_ids, grays = _synthetic_sequence(28, seed=21)
    res = simulate_video_policy(
        video="vid_cost",
        method="S5",
        budget_ratio=0.5,
        frames_det=frames_det,
        frame_ids=frame_ids,
        stgnf_model=_DummyStgnfModel(),
        gray_loader=lambda i: grays[i],
    )
    for d in res.decisions:
        assert d.decode_cpu_ms > 0.0
        if d.frame_idx > 0:
            assert d.observer_cpu_ms > 0.0
        if d.executed:
            assert d.detector_cpu_ms > 0.0
            assert d.tracker_cpu_ms > 0.0
        else:
            assert d.detector_cpu_ms == 0.0
            assert d.tracker_cpu_ms == 0.0


def test_preflight_9_score_policy_integrity():
    """Preflight check #9: CausalScoreDeliverer applies identical sentinel, bounded hold, and causal smoother."""
    deliv = CausalScoreDeliverer(train_sentinel_normality=1.5, max_hold_frames=2, smoother_name="none")
    # Frame 0: no score -> sentinel 1.5 -> anomaly -1.5
    assert deliv.step_frame(0, []) == pytest.approx(-1.5)
    # Frame 1: two people scored (-0.5 and 0.8) -> least normal is -0.5 -> anomaly +0.5
    assert deliv.step_frame(1, [0.8, -0.5]) == pytest.approx(0.5)
    # Frame 2 & 3: no new score, within max_hold_frames=2 -> holds +0.5
    assert deliv.step_frame(2, []) == pytest.approx(0.5)
    assert deliv.step_frame(3, []) == pytest.approx(0.5)
    # Frame 4: age = 4 - 1 = 3 > max_hold_frames=2 -> reverts to sentinel -1.5!
    assert deliv.step_frame(4, []) == pytest.approx(-1.5)


def test_preflight_10_normalization_and_scorer_parity():
    """Preflight check #10: CausalLiveStgnf.score_batch has exact numerical parity with stgnf_live.score_segments."""
    real_model = load_model()
    scorer = CausalLiveStgnf(real_model, dataset="ShanghaiTech")
    rng = np.random.default_rng(99)
    segs = rng.uniform(80.0, 350.0, size=(4, 24, 17, 3)).astype(np.float32)
    segs[:, :, :, 2] = rng.uniform(0.3, 0.95, size=(4, 24, 17)).astype(np.float32)

    ref_norm = score_segments(real_model, segs.copy())
    new_norm = scorer.score_batch(segs.copy())
    np.testing.assert_allclose(new_norm, ref_norm, atol=1e-6, rtol=1e-6)
