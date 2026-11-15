"""Q2-P1-E08 quality: every proposal is causal (prefix invariance), the parameter budgets, the virtual-node graph, context features,
the smoothing recursion, the hybrid loss, online == batch, the subgroup / FA-h machinery, plan / config consistency and (when produced) the outputs."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from edgelab.models import lorad_gaps as lg
from edgelab.models.lorad import mark_actnorm
from edgelab.models.stgnf_flow import build_flow, count_parameters, frame_nll, random_pose_batch

EXP = Path(__file__).resolve().parents[1] / "Q2-P1-E08_lorad_gaps_proposals"


def _flow(arm, L=24, seed=0):
    torch.manual_seed(seed)
    flow = lg.build_gap_flow(arm, L)
    mark_actnorm(flow)
    with torch.no_grad():  # non-degenerate weights so every path carries signal
        for p in flow.parameters():
            p.add_(0.05 * torch.randn_like(p))
    return flow.eval()


def _track(n=70, seed=0, step=2.0):
    rng = np.random.default_rng(seed)
    frames = np.arange(n, dtype=np.int64)
    base = np.stack([rng.uniform(100, 700, 17), rng.uniform(100, 400, 17), rng.uniform(0.05, 1.0, 17)], axis=1)
    kp = np.repeat(base[None], n, 0)
    kp[:, :, :2] += np.cumsum(rng.normal(0, step, (n, 1, 2)), axis=0) + rng.normal(0, 1, (n, 17, 2))
    return frames, kp


# ---------------------------------------------------------------------------------------------------------------------
# architecture
# ---------------------------------------------------------------------------------------------------------------------
def test_parameter_budgets():
    base = count_parameters(build_flow("causal", 24))
    assert base == 616
    assert count_parameters(lg.build_gap_flow("P_B", 24)) == base  # P_B: same runtime network
    assert count_parameters(lg.build_gap_flow("P_C", 24)) == base  # virtual nodes add no parameter (shared 1x1 convs, buffer-only graph)
    assert count_parameters(lg.build_gap_flow("P_D", 24)) == base
    pa, pabc = count_parameters(lg.build_gap_flow("P_A", 24)), count_parameters(lg.build_gap_flow("P_ABC", 24))
    assert pa == pabc and 1000 < pa < 3000
    assert count_parameters(lg.build_gap_flow("P_A", 16)) < pa  # UBnormal kernel (9) is smaller


def test_adjacency_20_nodes_and_hub_distances():
    A = lg.make_adjacency_ctx()
    assert A.shape == (1, 20, 20)
    assert torch.allclose(A[0].sum(0), torch.ones(20), atol=1e-6)  # columns normalised
    hop = lg.hop_distances()
    for node in (18, 19):  # nodes 19 and 20 (1-based): one hop from neck (1) and both hips (8, 11)
        assert all(hop[node, h] == 1 for h in lg.HUB)
    assert np.isfinite(hop).all() and hop.max() <= 8  # STG-NF's max_hop = 8 makes the uniform graph dense
    assert torch.allclose(lg.make_adjacency_ctx(0)[0], build_flow("causal", 24).steps[0].subnet.A[0], atol=1e-6)


@pytest.mark.parametrize("arm", ["P_A", "P_C", "P_ABC"])
def test_prefix_invariance_every_proposal(arm):
    flow = _flow(arm)
    v = 20 if lg.ARMS[arm]["ctx"] else 18
    x = torch.randn(6, 2, 24, v)
    x2 = x.clone()
    t0 = 13
    x2[:, :, t0:] += 3.0 * torch.randn_like(x2[:, :, t0:])
    with torch.no_grad():
        d = (frame_nll(flow, x)[:, :t0] - frame_nll(flow, x2)[:, :t0]).abs().max().item()
        d_future = (frame_nll(flow, x)[:, t0:] - frame_nll(flow, x2)[:, t0:]).abs().max().item()
    assert d < 1e-5 and d_future > 1e-3


def test_dilated_branches_are_each_causal_and_see_the_far_past():
    blk = lg.MSCausalSTGCN(2, 2, (5, 1), residual=True).eval()
    A = torch.ones(1, 18, 18) / 18
    x = torch.randn(2, 2, 40, 18)
    base, _ = blk(x, A)
    for t0 in (10, 30):
        x2 = x.clone()
        x2[:, :, t0] += 1.0
        out, _ = blk(x2, A)
        assert (out[:, :, :t0] - base[:, :, :t0]).abs().max() < 1e-6  # nothing before t0 moves
        assert (out[:, :, t0] - base[:, :, t0]).abs().max() > 1e-6
    # a frame d (k - 1) = 16 steps in the past still reaches t through the d = 4 branch, and 17 steps does not (zero tail of the kernels)
    x2 = x.clone()
    x2[:, :, 0] += 1.0
    out, _ = blk(x2, A)
    assert (out[:, :, 16] - base[:, :, 16]).abs().max() > 1e-6
    assert (out[:, :, 17] - base[:, :, 17]).abs().max() < 1e-6


def test_every_step_is_exactly_invertible_so_the_reverse_term_is_round_off():
    for arm in ("P_B", "P_C", "P_ABC"):
        flow = _flow(arm)
        x = torch.randn(4, 2, 24, 20 if lg.ARMS[arm]["ctx"] else 18)
        with torch.no_grad():
            z, _ = flow(x)
            assert (flow.decode(z) - x).abs().max() < 1e-3
            assert torch.allclose(lg.decode_flow(flow, z), flow.decode(z), atol=1e-6)


# ---------------------------------------------------------------------------------------------------------------------
# P_D smoothing
# ---------------------------------------------------------------------------------------------------------------------
def test_conf_smooth_untouched_when_confident_and_recursion_when_not():
    frames, kp = _track(30)
    kp[:, :, 2] = 1.0
    assert np.allclose(lg.conf_smooth(kp), kp)
    kp[10:14, 3, 2] = 0.0  # joint 3 lost for 4 frames: position held at the last good one
    s = lg.conf_smooth(kp)
    assert np.allclose(s[10:14, 3, :2], kp[9, 3, :2])
    assert np.allclose(s[14, 3, :2], kp[14, 3, :2])
    kp[20, 5, 2] = 0.125  # w = 0.5
    s = lg.conf_smooth(kp)
    assert np.allclose(s[20, 5, :2], 0.5 * kp[20, 5, :2] + 0.5 * kp[19, 5, :2])


def test_conf_smooth_is_causal_and_streams():
    _, kp = _track(40, seed=3)
    full = lg.conf_smooth(kp)
    assert np.allclose(lg.conf_smooth(kp[:25]), full[:25])
    sm = lg.ConfSmoother()
    assert np.allclose(np.stack([sm.update(k) for k in kp]), full)


# ---------------------------------------------------------------------------------------------------------------------
# P_C context
# ---------------------------------------------------------------------------------------------------------------------
def test_box_kinematics_translation_speed_survives_and_aspect_is_scale_free():
    frames, kp = _track(20, step=0.0)
    kp[:, :, :2] = kp[:1, :, :2]  # exactly static
    kp[:, :, :2] -= kp[:, :, :2].mean(axis=(0, 1))
    kp[:, :, :2] += 300.0
    moving = kp.copy()
    moving[:, :, 0] += (np.arange(20) * 40.0)[:, None]  # 40 px / frame to the right
    kin = lg.box_kinematics(frames, moving)
    h = lg.keypoint_box(moving)[:, 3]
    assert kin[0, 0] == 0.0
    assert np.allclose(kin[1:, 0], np.clip(40.0 / np.maximum(h[1:], lg.H_MIN), -0.5, 0.5), atol=1e-6)
    assert np.abs(lg.box_kinematics(frames, kp)[1:, :2]).max() < 1e-9  # static box: no velocity
    big = kp.copy()
    big[:, :, :2] = (big[:, :, :2] - 300) * 2 + 300
    assert np.allclose(lg.box_kinematics(frames, big)[:, 2], lg.box_kinematics(frames, kp)[:, 2], atol=1e-6)  # w / h scale free
    # N0 (per-window mean) erases the ABSOLUTE position of the window ...
    shifted = kp.copy()
    shifted[:, :, :2] += np.array([37.0, -21.0])
    assert torch.allclose(lg.to_input(kp[None, :, :, :2]), lg.to_input(shifted[None, :, :, :2]), atol=1e-4)
    # ... but keeps the drift inside the window; node 19 adds the scale-free per-frame speed on top of it
    nodes = lg.pack_nodes(np.concatenate([lg.box_kinematics(frames, moving), np.zeros((20, 3))], axis=1).reshape(20, 2, 3))
    assert nodes[5, 0, 0] > 1.0  # >= 1 gain unit for a box moving ~0.1 box-heights per frame


def test_neighbour_features_nearest_and_solo():
    f = lg.neighbour_features(np.array([[100.0, 100.0], [140.0, 100.0], [600.0, 300.0]]), np.array([100.0, 100.0, 100.0]))
    assert np.isclose(f[0, 0], 40 / lg.W_VID) and f[0, 1] == 0 and np.isclose(f[0, 2], 1 / 1.4)
    assert np.isclose(f[1, 0], -40 / lg.W_VID)
    assert np.allclose(lg.neighbour_features(np.array([[1.0, 2.0]]), np.array([50.0])), 0.0)
    nodes = lg.pack_nodes(np.stack([np.zeros(3), f[0]])[None])
    assert np.isclose(np.linalg.norm(nodes[0, 1]), f[0, 2]) and nodes[0, 1, 0] > 0


def test_clip_context_is_causal_per_frame_and_matches_hand_value():
    t0 = _track(30, seed=1)
    t1 = _track(30, seed=2)
    t1 = (t1[0] + 5, t1[1])  # frames 5 .. 34
    ctx = lg.clip_context({0: t0, 1: t1})
    assert (ctx[0][:5, 3:] == 0).all() and (ctx[1][-5:, 3:] == 0).all()  # alone: zeros
    assert (ctx[0][5:, 5] > 0).all()
    # the context at frame t does not change when every pose after frame t is cut off (same frame of the other tracks is allowed)
    cut = lg.clip_context({0: (t0[0][:20], t0[1][:20]), 1: (t1[0][:15], t1[1][:15])})  # frames <= 19
    assert np.allclose(cut[0][:20], ctx[0][:20]) and np.allclose(cut[1][:15], ctx[1][:15])


def test_prepare_tracks_shapes():
    out = lg.prepare_tracks([{0: _track(30)}, {}], smooth=True, ctx=True)
    assert out[0][0][1].shape == (30, 19, 3) and out[1] == {}
    assert lg.prepare_tracks([{0: _track(30)}], smooth=False, ctx=False)[0][0][1].shape == (30, 17, 3)


# ---------------------------------------------------------------------------------------------------------------------
# scorer, online == batch
# ---------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("arm", ["P_A", "P_C", "P_D", "P_ABC"])
def test_online_equals_batch(arm):
    cfg = {**lg.ARMS, **lg.EVAL_ONLY}[arm]
    L = 16
    flow = _flow(arm, L)
    tracks = {0: _track(60, 0), 1: (np.arange(10, 60, dtype=np.int64), _track(50, 1)[1]), 2: _track(30, 2)}
    batch_tracks = lg.prepare_tracks([tracks], cfg["smooth"], cfg["ctx"])[0]
    scorer = lg.GapFlowScorer(flow, L, cfg["ctx"])
    want = {pid: v for pid, v in zip(batch_tracks, scorer.score_tracks(list(batch_tracks.values())))}
    online = lg.OnlineGapScorer(flow, L, ctx=cfg["ctx"], smooth=cfg["smooth"])
    got = {pid: [] for pid in tracks}
    for f in range(60):
        poses = {pid: kp[np.searchsorted(fr, f)] for pid, (fr, kp) in tracks.items() if f in fr}
        # the online scorer returns one list per frame in the order of `poses`; recompute per track by index
        res = online.step(f, poses)
        ready = [pid for pid in poses if len(online.tracks[pid]["xy"]) == L and np.count_nonzero(np.fromiter(online.tracks[pid]["frames"], np.int64) < online.tracks[pid]["frames"][0] + L) >= L - 2]
        assert len(res) == len(ready)
        for pid, v in zip(ready, res):
            got[pid].append(v)
    for pid in tracks:
        assert len(got[pid]) == len(want[pid]) > 0
        assert np.allclose(got[pid], want[pid], atol=1e-4), (arm, pid)


# ---------------------------------------------------------------------------------------------------------------------
# P_B hybrid loss
# ---------------------------------------------------------------------------------------------------------------------
def test_perturbations_change_the_window_and_keep_the_shape():
    x = torch.randn(300, 2, 24, 18)
    xp, kind = lg.perturb(x, torch.Generator().manual_seed(0))
    assert xp.shape == x.shape and set(kind.tolist()) == {0, 1, 2}
    assert (xp - x).abs().flatten(1).max(1).values.gt(1e-6).all()
    sh = xp[kind == 0]
    assert torch.allclose(sh.sort(dim=2).values, x[kind == 0].sort(dim=2).values)  # a shuffle only reorders frames


def test_hybrid_loss_terms_and_bn_statistics_untouched_by_the_perturbed_pass():
    flow = lg.build_gap_flow("P_B", 24)
    flow.train()
    x = torch.randn(32, 2, 24, 18)
    flow(x)  # data-dependent ActNorm init
    loss, parts = lg.hybrid_loss(flow, x, torch.Generator().manual_seed(0))
    assert parts["rec"] < 1e-6  # exact inverse: the reverse term is float round-off
    assert abs(loss.item() - (parts["nll"] + lg.HP["lambda1"] * parts["rec"] + lg.HP["lambda2"] * parts["hinge"])) < 1e-4
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in flow.parameters())
    bn = next(m for m in flow.modules() if isinstance(m, torch.nn.BatchNorm2d))
    before = bn.running_mean.clone()
    with lg._FrozenBN(flow):
        flow(torch.randn(32, 2, 24, 18) * 5 + 3)
    assert torch.equal(bn.running_mean, before) and bn.training


def test_margin_term_is_zero_when_the_gap_exceeds_the_margin():
    hp = {**lg.HP, "lambda1": 0.0}
    flow = lg.build_gap_flow("P_B", 24)
    x = torch.randn(16, 2, 24, 18)
    flow.train()
    flow(x)  # ActNorm init
    _, parts = lg.hybrid_loss(flow, x, torch.Generator().manual_seed(1), {**hp, "margin": -1e9})
    assert parts["hinge"] == 0.0


def test_train_smoke_all_arms(tmp_path):
    L = 16
    rng = np.random.default_rng(0)
    xy = rng.uniform(100, 700, (400, 17, 2)).astype(np.float32)
    starts = np.arange(0, 300, 6, dtype=np.int64)
    for arm, spec in lg.ARMS.items():
        nodes = rng.normal(0, 1, (400, 2, 2)).astype(np.float32) if spec["ctx"] else None
        ts = lg.GapTrainSet(xy, nodes, starts, L)
        path = lg.train_gap_flow(arm, "unit", 0, tmp_path / arm, ts, epochs=2, batch_size=32, threads=1, max_batches=3, log=lambda m: None)
        flow, cfg = lg.load_gap_flow(path)
        assert cfg["params"] == count_parameters(flow) and len(cfg["history"]) == 2
        tracks = lg.prepare_tracks([{0: _track(40)}], spec["smooth"], spec["ctx"])[0]
        out = lg.GapFlowScorer(flow, L, spec["ctx"]).score_tracks(list(tracks.values()))[0]
        assert len(out) == 40 - L + 1 and np.isfinite(out).all()
        if spec["rev"]:
            assert "rec" in cfg["history"][-1] and "hinge" in cfg["history"][-1]
    # resumable: a finished run is returned as is
    assert lg.train_gap_flow("P_A", "unit", 0, tmp_path / "P_A", ts, epochs=2, log=lambda m: None) == tmp_path / "P_A" / "final.pt"


# ---------------------------------------------------------------------------------------------------------------------
# analysis driver: events / strata / operating points / verdicts / plan-config consistency / outputs
# ---------------------------------------------------------------------------------------------------------------------
from types import SimpleNamespace  # noqa: E402

from edgelab.analysis import q2_p1_e08 as e08  # noqa: E402


def test_false_alert_counts():
    score = np.array([0, 1, 1, 0, 1, 0, 0, 1], dtype=float)
    lab = np.array([0, 0, 1, 0, 0, 0, 0, 0])
    fa = e08.false_alert_counts(score, lab, np.array([0.5, 1.5]))
    assert fa.tolist() == [2, 0]  # runs {1,2} (contains the anomaly), {4}, {7} -> 2 false alerts at tau 0.5; none at 1.5
    assert e08.false_alert_counts(score, np.zeros(8), np.array([0.5])).tolist() == [3]


def test_masked_labels_and_masked_boot():
    ctx = SimpleNamespace(labels=[np.array([0, 0, 1, 1, 0, 1, 0, 0]), np.zeros(8, dtype=int)])
    events = [{"clip": 0, "a": 2, "b": 4, "short_event": True, "sustained_event": False}, {"clip": 0, "a": 5, "b": 6, "short_event": False, "sustained_event": True}]
    m = e08.masked_labels(ctx, events, "short_event")
    assert m[0].tolist() == [0, 0, 1, 1, 0, 2, 0, 0] and m[1].tolist() == [0] * 8  # other events ignored, normal frames kept
    perfect = [np.where(np.asarray(l) == 1, 1.0, 0.0) for l in ctx.labels]
    mb = e08.MaskedBoot(m, [0, 1], n=40)
    assert mb.valid
    mats = mb.stats([perfect])
    assert mb.point(mats) == pytest.approx(100.0) and mb.samples(mats).shape == (40,)
    only_normal = e08.masked_labels(ctx, events, "sustained_event")[1:]  # the normal clip alone has no positives
    assert not e08.MaskedBoot(only_normal, [0], n=10).valid


def test_event_table_strata(monkeypatch):
    n = 120
    lab = np.zeros(n, dtype=int)
    lab[10:30] = 1   # 20 frames at 30 fps = 0.67 s (short), person moves fast
    lab[60:119] = 1  # 59 frames = 1.97 s -> short as well (<= 2.5 s)
    ctx = SimpleNamespace(labels=[lab], fps=30.0, names=["c0"])
    base = _track(n, seed=1)[1]
    fast = base.copy()
    fast[:, :, 0] += 60.0 * np.arange(n)[:, None]  # ~0.2 box heights / frame
    frames = np.arange(n, dtype=np.int64)
    rel = [{0: (frames, fast), 1: (frames, base.copy()), 2: (frames, base.copy()), 3: (frames, base.copy())}]
    from edgelab.analysis import q2_p1_eval as ev

    monkeypatch.setattr(ev, "released_tracks", lambda ds, names: rel)
    events = e08.event_table("ShanghaiTech", ctx)
    assert len(events) == 2
    e0, e1 = events
    assert e0["short_event"] and e0["fast_motion"] and e0["high_crowd"] and e0["tracks"] == 4
    assert e1["dur_s"] == pytest.approx(59 / 30.0) and e1["short_event"]  # <= 2.5 s
    for e in events:  # the families partition the events
        assert e["short_event"] != e["sustained_event"] and e["fast_motion"] != e["moderate_motion"] and e["high_crowd"] != e["low_crowd"]
    ctx2 = SimpleNamespace(labels=[np.r_[np.zeros(5), np.ones(100), np.zeros(15)].astype(int)], fps=30.0, names=["c0"])
    monkeypatch.setattr(ev, "released_tracks", lambda ds, names: [{0: (frames, base.copy())}])
    (s,) = e08.event_table("ShanghaiTech", ctx2)
    assert s["sustained_event"] and s["moderate_motion"] and s["low_crowd"]


def _contrast(metric, a, b, delta, lo, hi, **kw):
    d = {"pose": "ML0_edge", "subset": "confirmation", "dataset": "mean", "metric": metric, "a": a, "b": b, "a_value": 0.0, "b_value": 0.0,
         "delta": delta, "ci_lo": lo, "ci_hi": hi, "excludes_zero": bool(lo > 0 or hi < 0)}
    d.update(kw)
    return d


def test_verdict_rules():
    P = {("B1", "ShanghaiTech"): 616, ("B1", "UBnormal"): 616, ("P_B", "ShanghaiTech"): 616, ("P_B", "UBnormal"): 616, ("P_C", "ShanghaiTech"): 616, ("TJP", "ShanghaiTech"): 757338}
    c = [_contrast("auc_S4", "P_A", "B1", 1.0, 0.2, 1.8), _contrast("auc_S4|short_event", "P_A", "B1", 2.0, 0.5, 3.5), _contrast("auc_S4|sustained_event", "P_A", "B1", 0.1, -0.5, 0.7),
         _contrast("auc_S4", "P_B", "B1", 0.4, 0.1, 0.7), _contrast("fa_h", "P_B", "B1", -3.0, -5.0, -1.0),
         _contrast("auc_S4", "P_D", "B1", -0.2, -0.4, 0.0), _contrast("fa_h", "P_D", "B1", -2.0, -4.0, 1.0)]
    v = e08.verdicts(c, [], P)
    assert v["H_A"]["verdict"] == "SUPPORTED"
    assert v["H_B"]["verdict"] == "SUPPORTED"
    assert v["H_D"]["verdict"] == "PARTIAL" and v["H_D"]["conditions"] == {"fa_h_ci_hi_lt_0": False, "pooled_ci_lo_ge_-0.5": True}
    assert v["H_C"]["verdict"] == "NOT SUPPORTED" and v["H_Joint"]["verdict"] == "NOT RUN"
    c[0] = _contrast("auc_S4", "P_A", "B1", -1.0, -1.8, -0.2)
    assert e08.verdicts(c, [], P)["H_A"]["verdict"] == "REFUTED"  # pooled CI below zero
    # latency condition of H_B comes from the Pi rows
    pi = [{"scorer": e08.ARM_LABEL["B1"], "variant": "Full-Rate", "score_ms": 1.00}, {"scorer": e08.ARM_LABEL["P_B"], "variant": "Full-Rate", "score_ms": 1.03}]
    assert e08.verdicts(c, pi, P)["H_B"]["conditions"]["pi5_scorer_latency_within_5pct"] is True
    pi[1]["score_ms"] = 1.2
    assert e08.verdicts(c, pi, P)["H_B"]["conditions"]["pi5_scorer_latency_within_5pct"] is False


def test_verdict_h_joint():
    def row(sc, auc, fps=25.0, wait=100.0, thr=False):
        return {"scorer": e08.ARM_LABEL[sc], "variant": "Cache-C1+Guard", "delivered_auc_ontime": auc, "delivered_fps": fps, "queue_wait_p99_ms": wait, "throttled_any": thr,
                "throttled_codes": "", "score_ms": 1.0, "end_to_end_ms": 30.0}
    ok = e08.verdicts([], [row("B1", 80.0), row("P_ABC", 82.0)], {})["H_Joint"]
    assert ok["verdict"] == "SUPPORTED"
    bad = e08.verdicts([], [row("B1", 83.0), row("P_ABC", 82.0, fps=20.0)], {})["H_Joint"]
    assert bad["verdict"] == "NOT SUPPORTED" and not bad["conditions"]["delivered_fps_ge_24"]


def test_plan_and_config_consistent():
    cfg = yaml.safe_load((EXP / "config.yaml").read_text())
    plan = (EXP / "plan.md").read_text()
    assert cfg["seed_list"] == list(e08.SEEDS) == [0, 1, 2]
    assert cfg["train"]["epochs"] == 10 and cfg["train"]["batch_size"] == 256 and cfg["train"]["lr"] == pytest.approx(5e-4)
    assert cfg["seg_len"] == e08.SEG_LEN
    assert cfg["hybrid_loss"]["lambda1_reverse"] == lg.HP["lambda1"] and cfg["hybrid_loss"]["lambda2_margin"] == lg.HP["lambda2"] and cfg["hybrid_loss"]["margin_bits_per_dim"] == lg.HP["margin"]
    assert list(lg.DILATIONS) == cfg["arms"]["P_A_MS_CTFlow"]["dilations"] == [1, 2, 4]
    assert cfg["smoothing"]["conf_threshold"] == lg.CONF_THR == 0.25
    assert cfg["bootstrap"]["resamples"] == e08.N_RESAMPLES == 2000 and cfg["bootstrap"]["seed"] == e08.BOOT_SEED
    assert cfg["strata"]["short_event_max_s"] == e08.SHORT_S and cfg["strata"]["fast_motion_min_box_heights_per_frame"] == e08.FAST_THR and cfg["strata"]["high_crowd_min_tracks"] == e08.CROWD_MIN
    assert set(cfg["arms"]) == {e08.ARM_LABEL[a] for a in (*e08.TRAIN_ARMS, "P_D")}
    for h in ("H_A", "H_B", "H_C", "H_D", "H_Joint"):
        assert h in plan
    assert "A1" in plan  # the parallel-training amendment is on record
    assert set(e08.TRAIN_ARMS) == set(lg.ARMS) and set(lg.EVAL_ONLY) == {"P_D"}


def test_train_jobs_cover_matrix(tmp_path, monkeypatch):
    # hermetic: independent of the DONE markers of the real (already trained) models
    monkeypatch.setattr(e08, "model_dir", lambda arm, ds, seed: tmp_path / f"{arm}_{ds}_s{seed}")
    jobs = e08.train_jobs(list(e08.DATASETS), list(e08.TRAIN_ARMS), list(e08.SEEDS))
    assert len(jobs) == 24 and len(set(jobs)) == 24  # 4 arms x 2 datasets x 3 seeds
    assert jobs[0][0] == "P_ABC"  # longest first
    done = tmp_path / "P_ABC_ShanghaiTech_s0"
    done.mkdir()
    (done / "DONE").write_text("")
    assert len(e08.train_jobs(list(e08.DATASETS), list(e08.TRAIN_ARMS), list(e08.SEEDS))) == 23  # finished models are skipped


def test_pi_arm_list():
    from edgelab.analysis import q2_p1_e08_pi5 as pi8

    arms = pi8.arm_list()
    assert len(arms) == 14 and len({a["name"] for a in arms}) == 14
    assert {a["scorer_key"] for a in arms} == set(e08.PI_ARMS)
    assert sum(a["use_c1"] for a in arms) == 7  # every scorer once full-rate and once Cache-C1+Guard


@pytest.mark.skipif(not (EXP / "summary.json").exists(), reason="E08 outputs not produced yet")
def test_outputs_complete():
    import csv

    for f in ("results.csv", "subgroups.csv", "pi5_benchmark.csv", "summary.json", "report.md"):
        assert (EXP / f).exists() and (EXP / f).stat().st_size > 0, f
    summ = json.loads((EXP / "summary.json").read_text())
    assert set(summ["verdicts"]) == {"H_A", "H_B", "H_C", "H_D", "H_Joint"}
    res = list(csv.DictReader(open(EXP / "results.csv")))
    arms = {r["arm"] for r in res}
    assert {e08.ARM_LABEL[a] for a in e08.EVAL_ARMS} <= arms
    assert {r["protocol"] for r in res} == {"S0", "S4"} and {r["subset"] for r in res} == {"all", "dev", "confirmation"}
    assert {r["pose"] for r in res} == set(e08.POSE_SOURCES) and {r["dataset"] for r in res} == set(e08.DATASETS)
    sub = list(csv.DictReader(open(EXP / "subgroups.csv")))
    assert {r["stratum"] for r in sub} >= set(e08.STRATA) and any(r["metric"] == "fa_h" for r in sub)
    pi = list(csv.DictReader(open(EXP / "pi5_benchmark.csv")))
    assert len(pi) == 14 and all(float(r["power_w"]) > 0 and r["throttled_any"] in ("False", "True") for r in pi)
    report = (EXP / "report.md").read_text()
    for h in ("H_A", "H_B", "H_C", "H_D", "H_Joint"):
        assert h in report
