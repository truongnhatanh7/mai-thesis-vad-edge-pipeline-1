"""Q2-P1-E07 quality: TrajVAD causality + online/batch equivalence, shared bootstrap engine vs sklearn, TrackTrack replay on toy detections,
verdict rule, plan / config consistency, and - when the artifacts exist - the outputs."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from edgelab.analysis import q2_p1_e07 as e07
from edgelab.analysis import q2_p1_eval as ev
from edgelab.models import trajvad as tv
from edgelab.models.lorad import starts_of

EXP = Path(__file__).resolve().parents[1] / "Q2-P1-E07_yolo26_tracktrack_trajvadp"


def _random_track(n=60, seed=0, gaps=((20, 22), (40, 41))):
    rng = np.random.default_rng(seed)
    frames = np.arange(n + 8)
    keep = np.ones(len(frames), bool)
    for a, b in gaps:
        keep[a:b] = False
    frames = frames[keep][:n].astype(np.int64)
    base = np.stack([rng.uniform(100, 700, 17), rng.uniform(100, 400, 17), np.ones(17)], axis=1)
    walk = np.cumsum(rng.normal(0, 3, (len(frames), 1, 2)), axis=0)
    kp = np.repeat(base[None], len(frames), 0).astype(np.float32)
    kp[:, :, :2] += walk.astype(np.float32) + rng.normal(0, 1, kp[:, :, :2].shape).astype(np.float32)
    kp[:, :, 2] = rng.uniform(0.3, 1.0, kp[:, :, 2].shape)
    return frames, kp


# ---- TrajVAD features ------------------------------------------------------------------------------------------------
def test_trajectory_features_are_causal():
    frames, kp = _random_track()
    f1 = tv.trajectory_features(frames, kp)
    kp2 = kp.copy()
    kp2[30:, :, :2] += 50.0 * np.random.default_rng(1).normal(size=kp2[30:, :, :2].shape).astype(np.float32)
    f2 = tv.trajectory_features(frames, kp2)
    assert f1.shape == (len(frames), tv.N_FEATURES) == (len(frames), 27)
    np.testing.assert_array_equal(f1[:30], f2[:30])  # pose t < 30 unchanged by whatever happens later
    assert np.abs(f1[30:] - f2[30:]).max() > 1e-3


def test_online_features_equal_batch_features():
    frames, kp = _random_track(n=80, seed=3)
    batch = tv.trajectory_features(frames, kp)
    st = tv._TrackState(16)
    online = np.stack([st.push(int(f), k) for f, k in zip(frames, kp)])
    scale = np.maximum(np.abs(batch).max(0), 1.0)
    assert np.abs(online - batch).max(0).__truediv__(scale).max() < 1e-3


@pytest.mark.parametrize("kind", ["T", "P"])
def test_online_trajvad_equals_window_scorer(kind):
    torch.manual_seed(0)
    L = 12
    model = tv.build_model(kind, L)
    tv.mark_initialised(model)
    scaler = tv.Scaler(np.zeros(27), np.full(27, 50.0))
    frames, kp = _random_track(n=50, seed=5)
    ref = tv.TrajVadScorer(model, scaler, kind, L).score_tracks([(frames, kp)])[0]
    on = tv.OnlineTrajVAD(model, scaler, kind, L)
    got = []
    for f, k in zip(frames, kp):
        got += [v for _, v in on.update_frame(int(f), {7: k})]
    assert len(ref) == len(got) == len(starts_of(frames, L)) > 0
    np.testing.assert_allclose(got, ref, rtol=1e-3, atol=1e-3)


def test_models_have_the_authors_sizes():
    assert tv.count_parameters(tv.build_model("T", 16)) == 210720
    assert tv.count_parameters(tv.build_model("P", 16)) == 757338


# ---- bootstrap engine ------------------------------------------------------------------------------------------------
def test_pair_matrix_auc_equals_sklearn():
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(0)
    labels = [(rng.random(60) < 0.3).astype(int) for _ in range(9)]
    scores = [np.round(rng.normal(size=60) + 0.8 * lab, 1) for lab in labels]  # rounded: ties
    idx = list(range(9))
    U, P, N = ev.pair_matrix(labels, scores, idx)
    assert ev.full_auc(U, P, N) == pytest.approx(roc_auc_score(np.concatenate(labels), np.concatenate(scores)), abs=1e-12)
    w = np.array([2, 0, 1, 1, 3, 0, 1, 1, 0], dtype=float)  # a resample: clip multiplicities
    y = np.concatenate([np.tile(labels[i], int(w[i])) for i in idx])
    s = np.concatenate([np.tile(scores[i], int(w[i])) for i in idx])
    assert ev.boot_auc(U, P, N, w[None])[0] == pytest.approx(roc_auc_score(y, s), abs=1e-12)


def test_verdict_rule():
    assert e07.verdict([0.2, 3.0]) == "better"
    assert e07.verdict([-3.0, -0.1]) == "worse"
    assert e07.verdict([-0.8, 0.7]) == "equal"
    assert e07.verdict([-0.8, 2.5]) == "inconclusive"
    assert e07.verdict([-4.0, 4.0]) == "inconclusive"


# ---- TrackTrack on toy detections -------------------------------------------------------------------------------------
def test_tracktrack_follows_a_walker_and_keeps_low_confidence_boxes():
    pytest.importorskip("ultralytics")
    from edgelab.pose.replay import Detections

    spec = ev.TrackerSpec("tracktrack", 0.20, 0.05, 30, 0.80)
    frames = []
    for t in range(30):
        x = 100 + 4 * t
        box = np.array([[x, 100, x + 40, 220]], dtype=np.float32)
        conf = np.array([0.5 if t % 5 else 0.08], dtype=np.float32)  # every 5th frame only a low-confidence detection
        kp = np.zeros((1, 17, 3), np.float32)
        kp[0, :, 0], kp[0, :, 1], kp[0, :, 2] = x + 20, 160, 0.9
        frames.append({"boxes": box, "conf": conf, "cls": np.zeros(1, np.float32), "keypoints": kp})
    tracks = ev.replay_clip(frames, np.arange(30), spec)
    assert len(tracks) == 1
    (fr, kp), = tracks.values()
    # confirmed after min_track_len = 3 hits (first emitted frame 3); the low-confidence frames 5, 10, ... are bridged by the second stage
    assert fr[0] <= 3 and len(fr) >= 26 and np.all(np.diff(fr) == 1) and {5, 10, 15, 20, 25} <= set(fr.tolist())
    assert spec.gate == 0.05 and spec.label.startswith("tt_c0.20_low0.05_m0.80")


# ---- plan / config ---------------------------------------------------------------------------------------------------
def test_config_matches_code():
    cfg = yaml.safe_load((EXP / "config.yaml").read_text())
    assert cfg["experiment"] == "Q2-P1-E07"
    assert cfg["trajvad"]["train"]["seeds"] == list(e07.SEEDS) and cfg["trajvad"]["train"]["epochs"] == 10
    assert cfg["datasets"]["ShanghaiTech"]["traj_seg_len"] == tv.SEG_LEN["ShanghaiTech"]
    assert cfg["datasets"]["UBnormal"]["traj_seg_len"] == tv.SEG_LEN["UBnormal"]
    assert cfg["frontends"]["ShanghaiTech"] and set(cfg["registry"]) == {"edgelab-yolo26n-pose", "edgelab-lorad-causal-flow", "edgelab-trajvad-p"}
    grid = cfg["trackers"]["tracktrack_sweep"]
    assert len(grid["track_high"]) * len(grid["track_low"]) * len(grid["match_thresh"]) == len(e07.TT_GRID) == 12
    plan = (EXP / "plan.md").read_text()
    assert "Selection rule T1" in plan and "**T2**" in plan and "Amendments" in plan


def test_tracktrack_selection_is_on_dev_only():
    import inspect

    src = inspect.getsource(e07.cmd_sweep)
    assert "dev_idx" in src and "confirm" not in src.lower().replace("confirmation", "")


# ---- outputs (when produced) -----------------------------------------------------------------------------------------
@pytest.mark.skipif(not (EXP / "summary.json").exists(), reason="E07 not finished")
def test_summary_outputs():
    s = json.loads((EXP / "summary.json").read_text())
    assert "tracktrack_selection" in s and "contrasts" in s and "pi5" in s
    assert (EXP / "results.csv").exists() and (EXP / "report.md").exists()
