"""Q2-P1-E06: LoRaD causal pose flow (causal ST-GCN normalising flow + N0 / N1 normalisation) vs the frozen / retrained STG-NF.

    uv run python -m edgelab.analysis.q2_p1_e06 train  [--procs 4]     # 20 flows -> ~/thesis-data/q2-p1-e06/models/<arm>_<ds>_s<seed>
    uv run python -m edgelab.analysis.q2_p1_e06 run                    # evaluate every arm on every pose source -> results.csv, summary.json
    uv run python -m edgelab.analysis.q2_p1_e06 report                 # summary.json -> report.md
    uv run python -m edgelab.analysis.q2_p1_e06 all

Plan: Q2-P1-E06_lorad_causal_flow/plan.md. Builds on the E00 window engine / protocol (S0 offline paper, S4 strict causal) and the E01 locked
tracker; the evaluation engine (`edgelab.analysis.q2_p1_eval`) is shared with E07.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path

from edgelab.models.lorad import build_trainset, train_flow

DATA_ROOT = Path.home() / "thesis-data"
WORK = DATA_ROOT / "q2-p1-e06"
MODELS = WORK / "models"
EXP_DIR = Path(__file__).resolve().parents[3] / "Q2-P1-E06_lorad_causal_flow"
STGNF = Path(__file__).resolve().parents[3] / "third_party/STG-NF"
SEG_LEN = {"ShanghaiTech": 24, "UBnormal": 16}
TRAIN_POSES = {"ShanghaiTech": STGNF / "data/ShanghaiTech/pose/train", "UBnormal": STGNF / "data/UBnormal/pose/train"}
SEEDS = (0, 1, 2)
ARMS = ("A1", "A2", "A3")
ARM_NAMES = {"A0": "A0_STGNF_Frozen", "A1": "A1_STGNF_Retrained", "A2": "A2_LoRaD_CausalConv_WinNorm",
             "A3": "A3_LoRaD_CausalConv_PrefixNorm", "A3s": "A3s_LoRaD_PrefixNorm_Stream", "A2L": "A2L_Lite"}


def model_dir(arm: str, dataset: str, seed: int) -> Path:
    return MODELS / f"{arm}_{dataset}_s{seed}"


def jobs() -> list[tuple[str, str, int]]:
    """(arm, dataset, seed), ShanghaiTech first (longest)."""
    out = [(arm, ds, seed) for ds in ("ShanghaiTech", "UBnormal") for seed in SEEDS for arm in ARMS]
    out += [("A2L", ds, 0) for ds in ("ShanghaiTech", "UBnormal")]
    return out


def _trainset(dataset: str):
    seg = SEG_LEN[dataset]
    return build_trainset(TRAIN_POSES[dataset], seg, cache=WORK / f"trainset_{dataset}_L{seg}.npz")


def _train_job(job: tuple[str, str, int]) -> str:
    arm, dataset, seed = job
    logf = WORK / "train.log"

    def log(msg: str) -> None:
        with open(logf, "a") as f:
            f.write(f"{time.strftime('%H:%M:%S')} {msg}\n")

    trainset = _trainset(dataset)
    path = train_flow(arm, dataset, seed, model_dir(arm, dataset, seed), trainset, epochs=10, threads=2, log=log)
    return str(path)


def cmd_train(procs: int) -> None:
    MODELS.mkdir(parents=True, exist_ok=True)
    for ds in SEG_LEN:  # build / cache the training sets once, before the workers start
        _trainset(ds)
    todo = [j for j in jobs() if not (model_dir(*j) / "DONE").exists()]
    print(f"{len(todo)} models to train on {procs} processes", flush=True)
    with mp.get_context("spawn").Pool(procs) as pool:
        for path in pool.imap_unordered(_train_job, todo):
            print("done", path, flush=True)


# =====================================================================================================================
# Evaluation
# =====================================================================================================================
RUN = WORK / "run"
POSE_SOURCES = ("released", "ML0_edge", "YOLO26_e2e")
EVAL_ARMS = ("A0", "A1", "A2", "A3", "A3s", "A2L")
BOOT_SEED = 0


def pose_cache(ds: str, src: str) -> Path:
    from edgelab.analysis import q2_p1_e07 as e07

    return e07.frontend_cache(ds, {"ML0_edge": "ML0_ncnn" if ds == "ShanghaiTech" else "ML0_mps", "YOLO26_e2e": "yolo26_e2e"}[src])


def pose_tracks(ds: str, src: str, ctx, procs: int) -> list[dict]:
    from edgelab.analysis import q2_p1_eval as ev

    if src == "released":
        return ev.released_tracks(ds, ctx.names)
    return ev.replay_tracks(pose_cache(ds, src), ctx.names, ev.BOTSORT_LOCKED, procs=procs, work=WORK, tag=f"{ds}_{src}")


def arm_scorers(arm: str, ds: str) -> list:
    """[(scorer, id)] one per seed."""
    from edgelab.analysis import q2_p1_eval as ev
    from edgelab.models.lorad import FlowScorer

    if arm == "A0":
        return [(ev.FrozenStgnfScorer(ds), f"A0_{ds}")]
    base, mode = ("A3", "stream") if arm == "A3s" else (arm, "window")
    seeds = (0,) if arm == "A2L" else SEEDS
    return [(FlowScorer.from_checkpoint(model_dir(base, ds, seed) / "final.pt", mode=mode), f"{arm}_{ds}_s{seed}") for seed in seeds]


def cmd_run(procs: int, datasets: list[str], arms: list[str]) -> None:
    import pickle

    from edgelab.analysis import q2_p1_eval as ev

    ctxs = ev.load_ctx()
    RUN.mkdir(parents=True, exist_ok=True)
    for ds in datasets:
        ctx = ctxs[ds]
        for src in POSE_SOURCES:
            tracks = None
            for arm in arms:
                out = RUN / f"{ds}__{src}__{arm}.pkl"
                if out.exists():
                    continue
                need = [model_dir("A3" if arm == "A3s" else arm, ds, sd) for sd in ((0,) if arm == "A2L" else SEEDS)] if arm != "A0" else []
                if any(not (d / "DONE").exists() for d in need):
                    print(f"skip {ds} {src} {arm}: not trained", flush=True)
                    continue
                t0 = time.perf_counter()
                tracks = tracks if tracks is not None else pose_tracks(ds, src, ctx, procs)
                per_seed = []
                for scorer, sid in arm_scorers(arm, ds):
                    sent = ev.train_sentinels(scorer, ds, WORK / "sentinels" / f"{sid}.json")[ev.QUANTILE[ds]]
                    table = ev.build_table(tracks, scorer)
                    res = ev.evaluate_table(table, tracks, ctx, scorer.seg_len, sent)
                    res.pop("rows")
                    res["sentinel"], res["scorer_id"] = sent, sid
                    per_seed.append(res)
                out.write_bytes(pickle.dumps({"dataset": ds, "pose": src, "arm": arm, "seeds": per_seed}))
                print(f"{ev.now()} {ds} {src} {arm}: S0 all {[round(r['auc']['S0']['all'], 2) for r in per_seed]} "
                      f"S4 all {[round(r['auc']['S4']['all'], 2) for r in per_seed]} ({time.perf_counter() - t0:.0f}s)", flush=True)


# =====================================================================================================================
# Hard checks C1 and the O(1) latency R5
# =====================================================================================================================
def check_causality() -> dict:
    import numpy as np
    import torch

    from edgelab.analysis import q2_p1_eval as ev
    from edgelab.models.lorad import FlowScorer, OnlineLoRaD, ema_alpha, load_flow
    from edgelab.models.stgnf_flow import frame_nll, random_pose_batch
    from edgelab.pose.segments import read_tracks

    out = {}
    ds = "ShanghaiTech"
    L = SEG_LEN[ds]
    causal, _, _ = load_flow(model_dir("A2", ds, 0) / "final.pt")
    acausal, _, _ = load_flow(model_dir("A1", ds, 0) / "final.pt")
    x = random_pose_batch(8, L, seed=1)
    x2 = x.clone()
    t0 = 12
    x2[:, :, t0:] += 3.0 * torch.randn_like(x2[:, :, t0:])
    with torch.no_grad():
        d_c = (frame_nll(causal, x)[:, :t0] - frame_nll(causal, x2)[:, :t0]).abs().max().item()
        d_a = (frame_nll(acausal, x)[:, :t0] - frame_nll(acausal, x2)[:, :t0]).abs().max().item()
        x_ = x[:, :, :]
        _, nll = causal(x_)
        d_mean = (frame_nll(causal, x_).mean(1) - nll).abs().max().item()
    out["causal_prefix_max_abs_delta"], out["acausal_prefix_max_abs_delta"], out["frame_nll_mean_vs_window_nll"] = d_c, d_a, d_mean
    path = sorted(TRAIN_POSES[ds].glob("*_tracked_person.json"))[0]
    frames, kp = max(read_tracks(path), key=lambda t: len(t[0]))
    kp = kp.astype(np.float32)
    flow, norm, L = load_flow(model_dir("A3", ds, 0) / "final.pt")
    ref_stream = FlowScorer(flow, norm, L, "stream").score_tracks([(frames, kp)])[0]
    online = OnlineLoRaD(flow, norm, L, mode="stream")
    got = [v for f, k in zip(frames, kp) if (v := online.update(1, int(f), k)) is not None]
    out["stream_online_vs_batch_max_abs_delta"] = float(np.abs(np.array(got) - ref_stream).max()) if len(got) == len(ref_stream) else float("nan")
    out["stream_windows"] = [len(got), len(ref_stream)]
    out["C1_pass"] = bool(d_c < 1e-5 and out["stream_online_vs_batch_max_abs_delta"] < 1e-4 and d_a > 1e-3)
    return out


def check_latency(reps: int = 300) -> dict:
    """R5: per-pose latency (1 thread, CPU) of the streaming evaluator vs L, and of the window evaluator."""
    import numpy as np
    import torch

    from edgelab.models.lorad import OnlineLoRaD, mark_actnorm
    from edgelab.models.stgnf_flow import build_flow

    torch.set_num_threads(1)
    rng = np.random.default_rng(0)
    pose = (np.stack([rng.uniform(100, 700, 17), rng.uniform(100, 400, 17), np.ones(17)], 1)).astype(np.float32)
    out = {"L": [], "stream_ms_k_tied": [], "window_ms": [], "stream_ms_k13": []}
    flows_k13 = {}
    for L in (8, 16, 24, 48, 96):
        for key, mode, norm, kernel_flow in (("stream_ms_k_tied", "stream", "N1", None), ("window_ms", "window", "N0", None)):
            flow = build_flow("causal", L).eval()
            mark_actnorm(flow)
            online = OnlineLoRaD(flow, norm, L, mode=mode)
            for f in range(L + 5):  # warm-up + fill the buffer
                online.update(1, f, pose + rng.normal(0, 1, pose.shape).astype(np.float32))
            t = []
            for f in range(L + 5, L + 5 + reps):
                p = pose + rng.normal(0, 1, pose.shape).astype(np.float32)
                t0 = time.perf_counter()
                online.update(1, f, p)
                t.append((time.perf_counter() - t0) * 1e3)
            out[key].append(float(np.median(t)))
        # fixed kernel (13, the SHT flow) with a longer circular buffer: the L-dependence alone
        flow = build_flow("causal", 24).eval()
        mark_actnorm(flow)
        online = OnlineLoRaD(flow, "N1", L, mode="stream")
        for f in range(L + 5):
            online.update(1, f, pose + rng.normal(0, 1, pose.shape).astype(np.float32))
        t = []
        for f in range(L + 5, L + 5 + reps):
            p = pose + rng.normal(0, 1, pose.shape).astype(np.float32)
            t0 = time.perf_counter()
            online.update(1, f, p)
            t.append((time.perf_counter() - t0) * 1e3)
        out["stream_ms_k13"].append(float(np.median(t)))
        out["L"].append(L)
    r = lambda v: max(v) / min(v)  # noqa: E731
    out["stream_ratio_max_min_k_tied"], out["stream_ratio_max_min_k13"] = r(out["stream_ms_k_tied"]), r(out["stream_ms_k13"])
    out["window_ratio_96_over_8"] = out["window_ms"][-1] / out["window_ms"][0]
    out["R5_pass_k13"] = bool(out["stream_ratio_max_min_k13"] <= 1.15 and out["window_ratio_96_over_8"] >= 3.0)
    out["R5_pass_k_tied"] = bool(out["stream_ratio_max_min_k_tied"] <= 1.15 and out["window_ratio_96_over_8"] >= 3.0)
    return out


# =====================================================================================================================
# Statistics, rules, report
# =====================================================================================================================
def load_cells() -> dict[tuple, dict]:
    import pickle

    return {(d["dataset"], d["pose"], d["arm"]): d for d in (pickle.loads(p.read_bytes()) for p in sorted(RUN.glob("*.pkl")))}


def cmd_stats(n_resamples: int = 2000) -> dict:
    import csv

    import numpy as np

    from edgelab.analysis import q2_p1_e07 as e07
    from edgelab.analysis import q2_p1_eval as ev

    cells, ctxs = load_cells(), ev.load_ctx()
    boots = {ds: ev.AucBoot(ctxs[ds], n_resamples, BOOT_SEED) for ds in ctxs}
    stats, rows = {}, []
    for (ds, src, arm), d in cells.items():
        for proto in ("S0", "S4"):
            series = [[s["series"][proto][i] for i in range(len(ctxs[ds].names))] for s in d["seeds"]]
            st = boots[ds].stats(series)
            stats[(ds, src, arm, proto)] = st
            for subset in ("all", "dev", "confirmation"):
                seed_auc = [s["auc"][proto][subset] for s in d["seeds"]]
                for sd, v in zip(range(len(seed_auc)), seed_auc):
                    rows.append({"dataset": ds, "pose": src, "arm": ARM_NAMES[arm], "seed": sd, "protocol": proto, "subset": subset, "auc": v,
                                 "ci_lo": "", "ci_hi": "", "coverage": d["seeds"][sd]["coverage"]["all"]})
                lo, hi = boots[ds].ci(st, subset)
                rows.append({"dataset": ds, "pose": src, "arm": ARM_NAMES[arm], "seed": "mean", "protocol": proto, "subset": subset,
                             "auc": boots[ds].point(st, subset), "ci_lo": lo, "ci_hi": hi,
                             "coverage": float(np.mean([s["coverage"]["all"] for s in d["seeds"]]))})
    contrasts = []

    def contrast(rule, a, b, src, proto, subset, text, datasets=("ShanghaiTech", "UBnormal")):
        per = {}
        for ds in datasets:
            ka, kb = (ds, src, a, proto), (ds, src, b, proto)
            if ka in stats and kb in stats:
                per[ds] = boots[ds].samples(stats[ka], subset) - boots[ds].samples(stats[kb], subset)
                pt = boots[ds].point(stats[ka], subset) - boots[ds].point(stats[kb], subset)
                lo, hi = ev.percentile_ci(per[ds])
                contrasts.append({"rule": rule, "text": text, "a": a, "b": b, "pose": src, "protocol": proto, "subset": subset, "dataset": ds,
                                  "delta": pt, "ci_lo": lo, "ci_hi": hi})
        if len(per) == len(datasets) > 1:
            samples = np.mean([per[ds] for ds in datasets], axis=0)
            pt = float(np.mean([next(c["delta"] for c in contrasts[::-1] if c["rule"] == rule and c["dataset"] == ds and c["a"] == a and c["b"] == b
                                     and c["pose"] == src and c["protocol"] == proto and c["subset"] == subset) for ds in datasets]))
            lo, hi = ev.percentile_ci(samples)
            contrasts.append({"rule": rule, "text": text, "a": a, "b": b, "pose": src, "protocol": proto, "subset": subset, "dataset": "mean",
                              "delta": pt, "ci_lo": lo, "ci_hi": hi})

    for subset in ("all", "dev", "confirmation"):
        contrast("R1", "A2", "A1", "released", "S0", subset, "A2 - A1: cost of causality (released poses, S0)")
        for src in POSE_SOURCES:
            contrast("R3", "A3", "A2", src, "S4", subset, f"A3 - A2: N1 vs N0 ({src}, S4)")
            contrast("R3s", "A3s", "A3", src, "S4", subset, f"A3s - A3: stream vs window ({src}, S4)")
            contrast("R4", "A1", "A0", src, "S0", subset, f"A1 - A0: retrained vs frozen ({src}, S0)")
            contrast("R4", "A1", "A0", src, "S4", subset, f"A1 - A0: retrained vs frozen ({src}, S4)")
            contrast("X1", "A2", "A0", src, "S4", subset, f"A2 - A0 ({src}, S4)")
            contrast("X1", "A3", "A0", src, "S4", subset, f"A3 - A0 ({src}, S4)")
            contrast("X2", "A2L", "A2", src, "S4", subset, f"A2L - A2 lite vs full temporal conv ({src}, S4)")
    Path(WORK / "report").mkdir(exist_ok=True)
    with open(WORK / "report" / "results.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(WORK / "report" / "contrasts.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(contrasts[0]))
        w.writeheader()
        w.writerows(contrasts)
    return {"rows": rows, "contrasts": contrasts, "stats": stats, "boots": boots, "e07_verdict": e07.verdict}


def decide(res: dict) -> dict:
    """R1, R3 and the LoRaD selection rule, on the contrasts."""
    import numpy as np

    c = res["contrasts"]
    rows = res["rows"]
    pick = lambda **kw: [x for x in c if all(x[k] == v for k, v in kw.items())]  # noqa: E731
    out = {}
    r1 = [x for x in pick(rule="R1", dataset="mean", subset="all")]
    out["R1"] = {"rule": "A2 - A1, released, S0, mean of datasets: free if CI lower >= -1.0", "per_dataset": pick(rule="R1", subset="all", protocol="S0"),
                 "free": bool(r1 and r1[0]["ci_lo"] >= -1.0)}
    r3 = {src: pick(rule="R3", dataset="mean", subset="all", pose=src)[0] for src in ("ML0_edge", "YOLO26_e2e") if pick(rule="R3", dataset="mean", subset="all", pose=src)}
    out["R3"] = {"rule": "A3 - A2 on edge poses, S4, mean over datasets (subset all): N1 adopted if CI lower >= -0.5 for both pose sources",
                 "contrasts": r3, "n1_adopted": bool(len(r3) == 2 and all(v["ci_lo"] >= -0.5 for v in r3.values()))}

    def dev_mean(arm):
        v = [r["auc"] for r in rows if r["arm"] == ARM_NAMES[arm] and r["pose"] == "ML0_edge" and r["protocol"] == "S4" and r["subset"] == "dev"
             and r["seed"] == "mean"]
        return float(np.mean(v)) if len(v) == 2 else float("nan")

    a2, a3 = dev_mean("A2"), dev_mean("A3")
    choose = "A3" if (a3 >= a2 - 0.5 or np.isnan(a2)) else "A2"
    out["selection"] = {"rule": "higher dev-S4 mean AUC (ML0_edge, mean of datasets); ties within 0.5 pt -> A3 (O(1))", "A2_dev_S4": a2, "A3_dev_S4": a3}
    out["selected_for_e07"] = {"arm": choose, "mode": "stream" if choose == "A3" else "window"}
    return out


def cmd_report() -> None:
    import numpy as np

    res = cmd_stats()
    decisions = decide(res)
    checks = {"causality": check_causality(), "latency": check_latency()}
    rows = res["rows"]
    summary = {"experiment": "Q2-P1-E06", "decisions": decisions, "checks": checks, "contrasts": res["contrasts"],
               "selected_for_e07": decisions["selected_for_e07"],
               "mean_rows": [r for r in rows if r["seed"] == "mean"]}
    (EXP_DIR / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    import csv

    with open(EXP_DIR / "results.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    lines = ["# Q2-P1-E06 report (generated; see `report_notes.md` for the narrative)\n"]
    mean = [r for r in rows if r["seed"] == "mean" and r["subset"] == "all"]
    lines.append("## Pooled AUC (all clips), seed mean [95% video-bootstrap CI]\n")
    lines.append("| dataset | pose | arm | S0 | S4 |")
    lines.append("|---|---|---|---|---|")
    keys = sorted({(r["dataset"], r["pose"], r["arm"]) for r in mean})
    for ds, pose, arm in keys:
        cell = {r["protocol"]: r for r in mean if (r["dataset"], r["pose"], r["arm"]) == (ds, pose, arm)}
        f = lambda r: f"{r['auc']:.2f} [{r['ci_lo']:.2f}, {r['ci_hi']:.2f}]"  # noqa: E731
        lines.append(f"| {ds} | {pose} | {arm} | {f(cell['S0'])} | {f(cell['S4'])} |")
    notes = EXP_DIR / "report_notes.md"
    if notes.exists():
        lines = [lines[0], notes.read_text(), *lines[1:]]
    (EXP_DIR / "report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"decisions": decisions, "checks": checks}, indent=1, default=float))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["train", "run", "report", "all"])
    parser.add_argument("--procs", type=int, default=4)
    parser.add_argument("--datasets", default="ShanghaiTech,UBnormal")
    parser.add_argument("--arms", default=",".join(EVAL_ARMS))
    args = parser.parse_args()
    if args.command in ("train", "all"):
        cmd_train(args.procs)
    if args.command in ("run", "all"):
        cmd_run(max(1, args.procs - 1), args.datasets.split(","), args.arms.split(","))
    if args.command in ("report", "all"):
        cmd_report()


if __name__ == "__main__":
    main()
