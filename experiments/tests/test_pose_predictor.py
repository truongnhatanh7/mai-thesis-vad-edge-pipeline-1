"""EXP-018 C2: constant-velocity batch filler == fill_P; GRU predictor is causal, capped, resets after long gaps."""

import numpy as np
import torch

from edgelab.pose.fill import fill_P
from edgelab.pose.predictor import (PosePredictor, fill_clip_gru, fill_dir_gru, fill_const_velocity, pose_from_state, pose_state,
                                    rollout)
from pose_toy import toy_track, write_toy_dir


def random_poses(rng, batch=3, length=40):
    walk = np.cumsum(rng.normal(0, 2, size=(batch, length, 1, 2)), axis=1) + rng.uniform(100, 400, size=(batch, 1, 1, 2))
    body = rng.uniform(-30, 30, size=(batch, 1, 17, 2))
    return torch.tensor(walk + body + rng.normal(0, 0.5, size=(batch, length, 17, 2)), dtype=torch.float32)


def trained_like(seed=0) -> PosePredictor:
    """A predictor with non-zero weights everywhere (the real one is trained; the tests only need a non-trivial function)."""
    torch.manual_seed(seed)
    model = PosePredictor()
    torch.nn.init.normal_(model.head.weight, std=0.05)
    return model.eval()


def test_pose_state_roundtrip():
    xy = random_poses(np.random.default_rng(0))[:, 0]
    assert torch.allclose(pose_from_state(*pose_state(xy)), xy, atol=1e-4)


def test_constant_velocity_batch_equals_fill_P_on_random_tracks():
    rng = np.random.default_rng(1)
    for _ in range(5):
        detected_frames = sorted(rng.choice(np.arange(40), size=rng.integers(3, 25), replace=False))
        track = toy_track(rng, detected_frames)
        filled_track, ages = fill_P(track, max_gap=8)
        xy = torch.zeros(1, 60, 17, 2)
        detected = torch.zeros(1, 60, dtype=torch.bool)
        for frame in detected_frames:
            xy[0, frame] = torch.tensor(np.asarray(track[f"{frame:04d}"]["keypoints"]).reshape(-1, 3)[:, :2])
            detected[0, frame] = True
        pose, filled = fill_const_velocity(xy, detected, max_gap=8)
        assert sorted(np.flatnonzero(filled[0].numpy())) == sorted(int(k) for k in ages)
        for key in ages:
            theirs = np.asarray(filled_track[key]["keypoints"]).reshape(-1, 3)[:, :2]
            assert np.allclose(pose[0, int(key)].numpy(), theirs, atol=1e-3)


def test_untrained_gru_holds_the_last_pose_and_fills_the_same_frames_as_constant_velocity():
    xy = random_poses(np.random.default_rng(2), batch=2)
    detected = torch.ones(2, 40, dtype=torch.bool)
    detected[:, [10, 11, 12, 13]] = False
    detected[:, 25:] = False  # then 15 missing frames: only 8 are filled
    pose, filled = rollout(PosePredictor().eval(), xy, detected)
    _, filled_cv = fill_const_velocity(xy, detected)
    assert torch.equal(filled, filled_cv)
    assert filled[:, 25:].sum(dim=1).tolist() == [8, 8] and filled[:, 10:14].all()
    for t in (10, 11, 12, 13):
        assert torch.allclose(pose[:, t], xy[:, 9], atol=1e-3)  # zero head = hold the last detection


def test_gru_is_causal_the_future_cannot_change_the_past():
    rng = np.random.default_rng(3)
    xy = random_poses(rng, batch=4)
    detected = torch.rand(4, 40, generator=torch.Generator().manual_seed(0)) > 0.4
    model = trained_like()
    pose_a, filled_a = rollout(model, xy, detected)
    t = 20
    xy_b, detected_b = xy.clone(), detected.clone()
    xy_b[:, t + 1:] += 50.0
    detected_b[:, t + 1:] = ~detected_b[:, t + 1:]
    pose_b, filled_b = rollout(model, xy_b, detected_b)
    assert torch.equal(filled_a[:, :t + 1], filled_b[:, :t + 1])
    assert torch.allclose(pose_a[:, :t + 1], pose_b[:, :t + 1], atol=1e-6)
    assert not torch.allclose(pose_a[:, t + 1:], pose_b[:, t + 1:])  # sanity: the test can see a change


def test_gru_state_is_reset_after_a_gap_longer_than_the_cap():
    xy = random_poses(np.random.default_rng(4), batch=1, length=60)
    detected = torch.ones(1, 60, dtype=torch.bool)
    detected[:, 10:30] = False  # 20 missing frames, cap 8
    model = trained_like()
    pose_all, filled_all = rollout(model, xy, detected)
    pose_tail, filled_tail = rollout(model, xy[:, 30:], detected[:, 30:])
    assert torch.allclose(pose_all[:, 30:], pose_tail, atol=1e-5)  # behaves as if it started at the new detection
    assert filled_all[:, 10:30].sum() == 8 and filled_all[:, 10:18].all()


def test_predictor_size_and_nothing_is_predicted_before_the_first_detection():
    model = PosePredictor()
    assert sum(p.numel() for p in model.parameters()) <= 20_000
    xy = random_poses(np.random.default_rng(5), batch=1)
    detected = torch.ones(1, 40, dtype=torch.bool)
    detected[:, :6] = False
    _, filled = rollout(model, xy, detected)
    assert not filled[:, :6].any()


def test_gru_json_fill_keeps_detections_decays_confidence_and_writes_ages():
    rng = np.random.default_rng(6)
    frames = [0, 1, 2, 3, 4, 8, 9, 10, 30, 31]
    track = toy_track(rng, frames)
    filled_clip, provenance = fill_clip_gru({"7": track}, PosePredictor().eval(), max_gap=8)
    out, ages = filled_clip["7"], provenance["7"]
    assert all(out[k] == track[k] for k in track)  # detections untouched
    assert sorted(int(k) for k in ages) == [5, 6, 7, 11, 12, 13, 14, 15, 16, 17, 18, 32, 33, 34, 35, 36, 37, 38, 39]
    assert ages["0006"] == 2 and ages["0013"] == 3 and ages["0039"] == 8
    base = np.asarray(track["0004"]["keypoints"]).reshape(-1, 3)
    made = np.asarray(out["0006"]["keypoints"]).reshape(-1, 3)
    assert np.allclose(made[:, 2], base[:, 2] * 0.8 ** 2) and np.allclose(made[:, :2], base[:, :2], atol=1e-3)
    assert out["0006"]["scores"] == track["0004"]["scores"] * 0.8 ** 2


def test_gru_dir_fill_writes_the_same_files_as_the_constant_velocity_filler(tmp_path):
    import json

    from edgelab.pose.fill import fill_dir

    source = write_toy_dir(tmp_path / "in")
    n_frames = {"01_001": 130, "02_005": 100}
    fill_dir(source, tmp_path / "cv", "P", 8, n_frames=n_frames)
    stats = fill_dir_gru(source, tmp_path / "gru", PosePredictor().eval(), 8, n_frames=n_frames)

    assert sorted(p.name for p in (tmp_path / "gru").iterdir()) == sorted(p.name for p in (tmp_path / "cv").iterdir())
    assert sorted(p.name for p in (tmp_path / "gru_prov").iterdir()) == sorted(p.name for p in (tmp_path / "cv_prov").iterdir())
    sidecar = json.loads((tmp_path / "gru_prov" / "01_001_alphapose_tracked_person.provenance.json").read_text())
    cv_sidecar = json.loads((tmp_path / "cv_prov" / "01_001_alphapose_tracked_person.provenance.json").read_text())
    assert sidecar == cv_sidecar  # same poses are made up, same ages: only their positions differ
    assert stats["predicted_poses"] > 0 and stats["mode"] == "GRU"


def test_gru_fill_of_a_video_without_poses_is_empty():
    assert fill_clip_gru({}, PosePredictor().eval()) == ({}, {})
