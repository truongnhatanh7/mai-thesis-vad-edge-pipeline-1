"""Unit tests for causal visual pose propagation (P-CV, P-LK, B-linear delayed) in EXP-023."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from edgelab.pose.fill import DAMPING, fill_B, fill_P
from edgelab.pose.visual_propagation import (
    PersonTrackState,
    interpolate_b_delayed,
    measure_CheapMotionOnly,
    measure_visual_observer,
    propagate_tracks_p_cv,
    propagate_tracks_p_lk,
)


def _make_textured_frame(h: int = 240, w: int = 320, seed: int = 42) -> np.ndarray:
    """Create a synthetic grayscale frame with rich background texture and a person patch."""
    rng = np.random.default_rng(seed)
    base = rng.integers(40, 120, size=(h, w), dtype=np.uint8)
    base = cv2.GaussianBlur(base, (5, 5), 1.0)
    return base


def _draw_person_patch(img: np.ndarray, cx: int, cy: int, pw: int = 36, ph: int = 80) -> np.ndarray:
    """Draw a high-contrast patterned person rectangle centered at (cx, cy) so LK has corner features."""
    out = img.copy()
    x1, y1 = max(2, cx - pw // 2), max(2, cy - ph // 2)
    x2, y2 = min(img.shape[1] - 2, cx + pw // 2), min(img.shape[0] - 2, cy + ph // 2)
    # Checkerboard / stripe pattern inside person box
    yy, xx = np.mgrid[y1:y2, x1:x2]
    pattern = (((xx - x1) // 4 + (yy - y1) // 4) % 2) * 140 + 50
    out[y1:y2, x1:x2] = pattern.astype(np.uint8)
    return out


def _person_keypoints(cx: float, cy: float, ph: float = 80.0) -> np.ndarray:
    """Generate 17 keypoints distributed inside a person box centered at (cx, cy)."""
    kp = np.zeros((17, 3), dtype=np.float64)
    ys = np.linspace(cy - 0.4 * ph, cy + 0.4 * ph, 17)
    xs = cx + np.sin(np.arange(17)) * 6.0
    kp[:, 0] = xs
    kp[:, 1] = ys
    kp[:, 2] = 0.9
    return kp


def test_p_cv_matches_existing_fill_p_and_caps_at_max_gap():
    """Verify P-CV matches existing fill_P math and enforces the max_gap (no resurrection) bound."""
    kp0 = _person_keypoints(100.0, 120.0)
    kp2 = _person_keypoints(108.0, 124.0)  # velocity = (+4, +2) px/frame
    box2 = np.array([82.0, 80.0, 134.0, 168.0])

    tr = PersonTrackState(
        track_id=1,
        last_obs_frame=0,
        last_obs_keypoints=kp0,
        last_obs_score=0.8,
        last_obs_box=box2,
    )
    tr.update_from_observation(2, kp2, 0.85, box2)
    tracks = {1: tr}

    # Build equivalent STG-NF dict for fill_P
    stg_track = {
        "0000": {"keypoints": kp0.reshape(-1).tolist(), "scores": 0.8},
        "0002": {"keypoints": kp2.reshape(-1).tolist(), "scores": 0.85},
    }
    filled_ref, _ = fill_P(stg_track, max_gap=4, damping=DAMPING)

    for step_f in range(3, 7):
        res = propagate_tracks_p_cv(tracks, step_f, current_time_ms=step_f * 41.67, max_gap=4)
        assert 1 in res
        rec = res[1]
        ref_kp = np.asarray(filled_ref[f"{step_f:04d}"]["keypoints"]).reshape(17, 3)
        np.testing.assert_allclose(rec.keypoints, ref_kp, atol=1e-6)
        assert rec.age == step_f - 2

    # Frame 7 is k = 5 > max_gap = 4 -> track must expire (no eternal extrapolation)
    res_expired = propagate_tracks_p_cv(tracks, 7, current_time_ms=7 * 41.67, max_gap=4)
    assert 1 not in res_expired
    assert 1 not in tracks


def test_p_lk_tracks_visual_shift_and_detects_disagreement():
    """Verify P-LK propagates reliable joints along visual motion and reports high-quantile disagreement."""
    base = _make_textured_frame(240, 320, seed=7)
    # Person was stationary at (120, 120) up to frame 1 (so P-CV velocity = 0),
    # then suddenly moves +6 px right, +4 px down on frame 2.
    f1 = _draw_person_patch(base, 120, 120)
    f2 = _draw_person_patch(base, 126, 124)

    kp1 = _person_keypoints(120.0, 120.0)
    box1 = np.array([102.0, 80.0, 138.0, 160.0])
    tr = PersonTrackState(
        track_id=1,
        last_obs_frame=0,
        last_obs_keypoints=kp1,
        last_obs_score=0.9,
        last_obs_box=box1,
    )
    tr.update_from_observation(1, kp1, 0.9, box1)  # velocity_cv == 0
    tracks = {1: tr}

    obs = measure_visual_observer(f1, f2, tracks, frame_idx=2)
    assert obs.any_reliable_motion is True
    assert obs.per_person[1].valid_joints >= 12
    # Since P-CV predicted 0 displacement and person moved sqrt(6^2 + 4^2) = 7.21 px on height 80 px (~0.09),
    # normalized disagreement_q80 should be around 0.08 - 0.10.
    assert obs.max_disagreement_q80 > 0.06

    prop = propagate_tracks_p_lk(tracks, 2, current_time_ms=83.3, observer_out=obs, max_gap=8)
    assert 1 in prop
    moved_kp = prop[1].keypoints
    median_dx = float(np.median(moved_kp[:, 0] - kp1[:, 0]))
    median_dy = float(np.median(moved_kp[:, 1] - kp1[:, 1]))
    assert 4.5 <= median_dx <= 7.5
    assert 2.5 <= median_dy <= 5.5


def test_failure_is_not_calmness_on_untextured_frames():
    """Preflight check #7: No reliable flow points must NOT produce an all-zero reassuring motion cue."""
    blank1 = np.full((240, 320), 128, dtype=np.uint8)
    blank2 = np.full((240, 320), 128, dtype=np.uint8)

    kp = _person_keypoints(160.0, 120.0)
    box = np.array([140.0, 80.0, 180.0, 160.0])
    tr = PersonTrackState(1, 0, kp, 0.9, box)
    tracks = {1: tr}

    obs = measure_visual_observer(blank1, blank2, tracks, frame_idx=1)
    assert obs.any_reliable_motion is False
    assert obs.per_person[1].valid_joints == 0
    assert obs.per_person[1].invalid_fraction == pytest.approx(1.0)
    assert obs.max_invalid_fraction == pytest.approx(1.0)
    # Disagreement must be NaN (never 0.0) when flow fails completely
    assert np.isnan(obs.max_disagreement_q80)
    assert np.isnan(obs.per_person[1].disagreement_q80)


def test_scene_cut_invalidates_tracks():
    """Preflight check #5: Scene cuts invalidate old track states immediately."""
    f1 = _make_textured_frame(240, 320, seed=1)
    # Abrupt bright flash / completely different scene cut
    f2 = np.clip(255 - f1 + 80, 0, 255).astype(np.uint8)

    kp = _person_keypoints(120.0, 120.0)
    box = np.array([100.0, 80.0, 140.0, 160.0])
    tracks = {1: PersonTrackState(1, 0, kp, 0.9, box)}

    obs = measure_visual_observer(f1, f2, tracks, frame_idx=1)
    assert obs.scene_cut is True

    prop = propagate_tracks_p_lk(tracks, 1, current_time_ms=41.67, observer_out=obs, max_gap=8)
    assert prop == {}
    assert len(tracks) == 0


def test_b_linear_delayed_availability_and_interpolation():
    """Preflight check #2: B-linear matches fill_B values and never backdates availability before frame_after."""
    kp0 = _person_keypoints(100.0, 120.0)
    kp4 = _person_keypoints(116.0, 128.0)
    box0 = np.array([80.0, 80.0, 120.0, 160.0])
    box4 = np.array([96.0, 88.0, 136.0, 168.0])

    completion_after_ms = 4 * 41.67 + 14.0
    recs = interpolate_b_delayed(
        track_id=3,
        frame_before=0,
        kp_before=kp0,
        score_before=0.8,
        box_before=box0,
        frame_after=4,
        kp_after=kp4,
        score_after=0.9,
        box_after=box4,
        completion_time_after_ms=completion_after_ms,
        max_gap=8,
    )
    assert [r.frame_idx for r in recs] == [1, 2, 3]
    for r in recs:
        assert r.available_at_frame == 4
        assert r.available_at_time_ms == pytest.approx(completion_after_ms)

    # Check exact numerical agreement with fill_B
    stg_track = {
        "0000": {"keypoints": kp0.reshape(-1).tolist(), "scores": 0.8},
        "0004": {"keypoints": kp4.reshape(-1).tolist(), "scores": 0.9},
    }
    filled_b, _ = fill_B(stg_track, max_gap=8)
    for r in recs:
        ref = np.asarray(filled_b[f"{r.frame_idx:04d}"]["keypoints"]).reshape(17, 3)
        np.testing.assert_allclose(r.keypoints, ref, atol=1e-6)


def test_cheap_motion_compensates_global_lighting_shift():
    """Verify uniform global illumination shift does not create a false high motion score in S3."""
    f1 = _make_textured_frame(240, 320, seed=10)
    # Add uniform +15 brightness shift everywhere without any moving object
    f2 = np.clip(f1.astype(np.int16) + 15, 0, 255).astype(np.uint8)
    score_static, luma_diff, scene_cut = measure_CheapMotionOnly(f1, f2)
    assert luma_diff == pytest.approx(15.0, abs=1.0)
    assert scene_cut is False
    assert score_static < 0.02
