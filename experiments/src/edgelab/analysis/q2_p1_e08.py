"""Q2-P1-E08: four fixes for LoRaD's limitations (MS-CTFlow, RevMargin, IAC-Flow, ConfSmooth) and the joint MS-IAC-RevFlow.

    uv run python -m edgelab.analysis.q2_p1_e08 train  [--procs 5]    # 24 flows, single-thread worker processes
    uv run python -m edgelab.analysis.q2_p1_e08 run                                       # every arm on every pose source -> WORK/run/*.pkl
    uv run python -m edgelab.analysis.q2_p1_e08 stats                                     # bootstrap, subgroups, FA/h -> WORK/report/*
    uv run python -m edgelab.analysis.q2_p1_e08 pi-prep | pi-report | mlflow | final

Plan: Q2-P1-E08_lorad_gaps_proposals/plan.md (locked before any model was trained). Engine: `q2_p1_eval` (E00 protocol, E01 trackers), baselines =
E06 checkpoints / cells, TrajVAD-P = E07 cells.
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np

THESIS = Path.home() / "thesis-data"
WORK = THESIS / "q2-p1-e08"
MODELS = WORK / "models"
RUN = WORK / "run"
E06_MODELS = THESIS / "q2-p1-e06" / "models"
E06_SENT = THESIS / "q2-p1-e06" / "sentinels"
EXP_DIR = Path(__file__).resolve().parents[3] / "Q2-P1-E08_lorad_gaps_proposals"
STGNF = Path(__file__).resolve().parents[3] / "third_party/STG-NF"
SEG_LEN = {"ShanghaiTech": 24, "UBnormal": 16}
TRAIN_POSES = {"ShanghaiTech": STGNF / "data/ShanghaiTech/pose/train", "UBnormal": STGNF / "data/UBnormal/pose/train"}
DATASETS = ("ShanghaiTech", "UBnormal")
SEEDS = (0, 1, 2)
TRAIN_ARMS = ("P_A", "P_B", "P_C", "P_ABC")
EVAL_ARMS = ("B0", "B1", "P_A", "P_B", "P_C", "P_D", "P_ABC")
ARM_LABEL = {"B0": "B0_STGNF", "B1": "B1_LoRaD", "P_A": "P_A_MS_CTFlow", "P_B": "P_B_RevMargin", "P_C": "P_C_IAC_Flow", "P_D": "P_D_ConfSmooth",
             "P_ABC": "P_ABC_MS_IAC_RevFlow", "TJP": "TrajVAD-P"}
E06_ARM = {"B0": "A1", "B1": "A2"}
POSE_SOURCES = ("released", "ML0_edge", "YOLO26_e2e")
PRIMARY = "ML0_edge"
BOOT_SEED = 20261114
N_RESAMPLES = 2000


def model_dir(arm: str, ds: str, seed: int) -> Path:
    return MODELS / f"{arm}_{ds}_s{seed}"


def arm_flags(arm: str) -> dict:
    from edgelab.models import lorad_gaps as lg

    return {**lg.ARMS, **lg.EVAL_ONLY}.get(arm, dict(ms=False, ctx=False, smooth=False, rev=False))


# =====================================================================================================================
# Training (one process, sequential)
# =====================================================================================================================
def _trainset(ds: str, smooth: bool, ctx: bool):
    from edgelab.models import lorad_gaps as lg

    seg = SEG_LEN[ds]
    return lg.build_gap_trainset(TRAIN_POSES[ds], seg, ctx=ctx, smooth=smooth, cache=WORK / f"gtrain_{ds}_L{seg}_s{int(smooth)}c{int(ctx)}.npz")


COST = {"P_ABC": 1.0, "P_A": 0.4, "P_B": 0.4, "P_C": 0.12}  # relative training cost (1 thread), for longest-first scheduling


def train_jobs(datasets: list[str], arms: list[str], seeds: list[int]) -> list[tuple[str, str, int]]:
    jobs = [(arm, ds, seed) for ds in datasets for seed in seeds for arm in arms if not (model_dir(arm, ds, seed) / "DONE").exists()]
    return sorted(jobs, key=lambda j: (-COST[j[0]] * (1.0 if j[1] == "ShanghaiTech" else 0.35), j[2]))


def _train_job(job: tuple) -> str:
    from edgelab.models import lorad_gaps as lg

    arm, ds, seed, device, threads, max_batches = job
    spec = lg.ARMS[arm]

    def log(msg: str) -> None:
        with open(WORK / "train.log", "a") as f:
            f.write(f"{time.strftime('%H:%M:%S')} {msg}\n")

    ts = _trainset(ds, spec["smooth"], spec["ctx"])
    path = lg.train_gap_flow(arm, ds, seed, model_dir(arm, ds, seed), ts, epochs=10, threads=threads, device=device, max_batches=max_batches, log=log)
    return str(path)


def cmd_train(device: str, threads: int, datasets: list[str], arms: list[str], seeds: list[int], procs: int = 5, max_batches: int | None = None) -> None:
    """24 independent models on `procs` single-thread worker processes of THIS experiment (1 thread is the fastest per model: P_ABC 1.05 s / batch vs 2.4 s at 8 threads)."""
    import multiprocessing as mp

    WORK.mkdir(parents=True, exist_ok=True)
    for ds in datasets:  # build / cache the training windows once, before the workers start
        for smooth, ctx in {(lg_arm["smooth"], lg_arm["ctx"]) for lg_arm in map(arm_flags, arms)}:
            _trainset(ds, smooth, ctx)
    todo = train_jobs(datasets, arms, seeds)
    print(f"{len(todo)} models to train on {procs} single-thread processes", flush=True)
    with mp.get_context("spawn").Pool(procs) as pool:
        for path in pool.imap_unordered(_train_job, [(a, d, s, device, threads, max_batches) for a, d, s in todo]):
            print(time.strftime("%H:%M:%S"), "done", path, flush=True)


# =====================================================================================================================
# Evaluation
# =====================================================================================================================
def pose_tracks(ds: str, src: str, ctx, procs: int) -> list[dict]:
    from edgelab.analysis import q2_p1_e06 as e06
    from edgelab.analysis import q2_p1_e07 as e07

    if src == "YOLO26_e2e":  # E07 T1-selected TrackTrack cell over the YOLO26 e2e cache (same on-disk track cache as E07)
        return e07.tracks_for(ds, "yolo26_e2e", e07.tracker_specs()["tracktrack"], ctx.names, procs)
    return e06.pose_tracks(ds, src, ctx, procs)  # released / ML0_edge (+ E01 locked two-stage BoT-SORT)


def gap_sentinels(scorer, ds: str, smooth: bool, ctx: bool, path: Path) -> dict:
    """E00 sentinel quantiles of the scorer's normality over every complete window of the released-pose training tracks (processed by the arm's own front end)."""
    from edgelab.analysis import q2_p1_e00 as e00
    from edgelab.analysis import q2_p1_eval as ev
    from edgelab.models import lorad_gaps as lg

    if Path(path).exists():
        return json.loads(Path(path).read_text())
    clips = lg.prepare_tracks(ev.train_tracks(ds), smooth, ctx)
    tracks = [t for clip in clips for t in clip.values()]
    parts = []
    for i in range(0, len(tracks), 512):
        parts += scorer.score_tracks(tracks[i:i + 512])
    values = e00.sentinel_values(np.concatenate([p for p in parts if len(p)]))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(values))
    return values


def arm_scorers(arm: str, ds: str) -> list[tuple]:
    """[(scorer, sentinel_cache_path, scorer_id, smooth, ctx)] one per seed."""
    from edgelab.models import lorad_gaps as lg
    from edgelab.models.lorad import FlowScorer, load_flow

    out = []
    for seed in SEEDS:
        if arm in E06_ARM:
            sid = f"{E06_ARM[arm]}_{ds}_s{seed}"
            out.append((FlowScorer.from_checkpoint(E06_MODELS / f"{E06_ARM[arm]}_{ds}_s{seed}" / "final.pt", mode="window"), E06_SENT / f"{sid}.json", sid, False, False))
        elif arm == "P_D":
            flow, _, seg = load_flow(E06_MODELS / f"A2_{ds}_s{seed}" / "final.pt")
            sid = f"P_D_{ds}_s{seed}"
            out.append((lg.GapFlowScorer(flow, seg, ctx=False), WORK / "sentinels" / f"{sid}.json", sid, True, False))
        else:
            flow, cfg = lg.load_gap_flow(model_dir(arm, ds, seed) / "final.pt")
            spec = lg.ARMS[arm]
            sid = f"{arm}_{ds}_s{seed}"
            out.append((lg.GapFlowScorer(flow, cfg["seg_len"], ctx=spec["ctx"]), WORK / "sentinels" / f"{sid}.json", sid, spec["smooth"], spec["ctx"]))
    return out


def trained(arm: str, ds: str) -> bool:
    if arm in E06_ARM:
        return all((E06_MODELS / f"{E06_ARM[arm]}_{ds}_s{s}" / "DONE").exists() for s in SEEDS)
    if arm == "P_D":
        return all((E06_MODELS / f"A2_{ds}_s{s}" / "DONE").exists() for s in SEEDS)
    return all((model_dir(arm, ds, s) / "DONE").exists() for s in SEEDS)


def cmd_run(procs: int, datasets: list[str], arms: list[str], sources: list[str]) -> None:
    from edgelab.analysis import q2_p1_e06 as e06
    from edgelab.analysis import q2_p1_eval as ev
    from edgelab.models import lorad_gaps as lg

    ctxs = ev.load_ctx()
    RUN.mkdir(parents=True, exist_ok=True)
    for ds in datasets:
        ctx = ctxs[ds]
        for src in sources:
            tracks = None
            for arm in arms:
                out = RUN / f"{ds}__{src}__{arm}.pkl"
                if out.exists():
                    continue
                if arm in E06_ARM and src != "YOLO26_e2e":  # baselines of the E06 matrix: identical pipeline, reuse the E06 cell
                    cell = e06.RUN / f"{ds}__{src}__{E06_ARM[arm]}.pkl"
                    if cell.exists():
                        d = pickle.loads(cell.read_bytes())
                        d["arm"], d["source"] = arm, f"E06 cell {cell.name}"
                        out.write_bytes(pickle.dumps(d))
                        print(f"{ev.now()} {ds} {src} {arm}: E06 cell reused S4 all {[round(r['auc']['S4']['all'], 2) for r in d['seeds']]}", flush=True)
                        continue
                if not trained(arm, ds):
                    print(f"skip {ds} {src} {arm}: not trained", flush=True)
                    continue
                t0 = time.perf_counter()
                tracks = tracks if tracks is not None else pose_tracks(ds, src, ctx, procs)
                per_seed = []
                for scorer, sent_path, sid, smooth, use_ctx in arm_scorers(arm, ds):
                    sent = gap_sentinels(scorer, ds, smooth, use_ctx, sent_path)[ev.QUANTILE[ds]]
                    prepared = lg.prepare_tracks(tracks, smooth, use_ctx) if (smooth or use_ctx) else tracks
                    table = ev.build_table(prepared, scorer)
                    res = ev.evaluate_table(table, tracks, ctx, scorer.seg_len, sent)  # track statistics of the pipeline's own tracks
                    res.pop("rows")
                    res["sentinel"], res["scorer_id"] = sent, sid
                    per_seed.append(res)
                out.write_bytes(pickle.dumps({"dataset": ds, "pose": src, "arm": arm, "seeds": per_seed}))
                print(f"{ev.now()} {ds} {src} {arm}: S0 all {[round(r['auc']['S0']['all'], 2) for r in per_seed]} "
                      f"S4 all {[round(r['auc']['S4']['all'], 2) for r in per_seed]} ({time.perf_counter() - t0:.0f}s)", flush=True)


# =====================================================================================================================
# Events, strata, operating-point metrics
# =====================================================================================================================
SHORT_S, FAST_THR, CROWD_MIN = 2.5, 0.06, 4
FAMILIES = {"duration": ("short_event", "sustained_event"), "motion": ("fast_motion", "moderate_motion"), "crowd": ("high_crowd", "low_crowd")}
STRATA = [s for pair in FAMILIES.values() for s in pair]
SUBSETS = ("all", "dev", "confirmation")
GRID_K = 200


def event_table(ds: str, ctx) -> list[dict]:
    """Every event (maximal run of anomalous frames) of the dataset with its stratum flags (from the released poses, pipeline independent)."""
    from edgelab.analysis import q2_p1_eval as ev
    from edgelab.models import lorad_gaps as lg

    rel = ev.released_tracks(ds, ctx.names)
    events = []
    for ci, lab in enumerate(ctx.labels):
        edges = np.diff(np.concatenate([[0], (np.asarray(lab) == 1).astype(int), [0]]))
        starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
        if len(starts) == 0:
            continue
        n = len(lab)
        count = np.zeros(n + 1)
        speed = {}
        for pid, (frames, kp) in rel[ci].items():
            ok = (frames >= 0) & (frames < n)
            np.add.at(count, frames[ok], 1)
            kin = lg.box_kinematics(frames, kp)
            speed[pid] = (frames, np.hypot(kin[:, 0], kin[:, 1]))
        for a, b in zip(starts, ends):
            med = [float(np.median(sp[(fr >= a) & (fr < b)])) for fr, sp in speed.values() if np.count_nonzero((fr >= a) & (fr < b)) >= 3]
            ev_ = {"clip": ci, "a": int(a), "b": int(b), "dur_s": float((b - a) / ctx.fps), "speed": float(max(med)) if med else 0.0,
                   "tracks": float(count[a:b].mean())}
            ev_["short_event"], ev_["sustained_event"] = ev_["dur_s"] <= SHORT_S, ev_["dur_s"] > SHORT_S
            ev_["fast_motion"], ev_["moderate_motion"] = ev_["speed"] >= FAST_THR, ev_["speed"] < FAST_THR
            ev_["high_crowd"], ev_["low_crowd"] = ev_["tracks"] >= CROWD_MIN, ev_["tracks"] < CROWD_MIN
            events.append(ev_)
    return events


def masked_labels(ctx, events: list[dict], stratum: str) -> list[np.ndarray]:
    """Per clip: 1 = frames of the stratum's events, 0 = normal frames, 2 = ignored (other events)."""
    out = [np.where(np.asarray(lab) == 0, 0, 2).astype(np.int8) for lab in ctx.labels]
    for e in events:
        if e[stratum]:
            out[e["clip"]][e["a"]:e["b"]] = 1
    return out


class MaskedBoot:
    """Paired video bootstrap of the pooled AUC with an ignore label (2), on one subset of clips."""

    def __init__(self, labels: list[np.ndarray], idx: list[int], n: int = N_RESAMPLES, seed: int = BOOT_SEED):
        from edgelab.analysis import q2_p1_eval as ev

        self.labels, self.idx = labels, idx
        P = np.array([(labels[i] == 1).sum() for i in idx], dtype=np.float64)
        N = np.array([(labels[i] == 0).sum() for i in idx], dtype=np.float64)
        self.valid = bool(P.sum() > 0 and N.sum() > 0)
        self.W = ev.resample_weights(len(idx), P, N, n, seed) if self.valid else None

    def stats(self, series_by_seed: list[list[np.ndarray]]) -> list[tuple]:
        from edgelab.analysis import q2_p1_eval as ev

        return [ev.pair_matrix(self.labels, sc, self.idx) for sc in series_by_seed]

    def point(self, mats) -> float:
        from edgelab.analysis import q2_p1_eval as ev

        return 100 * float(np.mean([ev.full_auc(*m) for m in mats]))

    def samples(self, mats) -> np.ndarray:
        from edgelab.analysis import q2_p1_eval as ev

        return np.mean([ev.boot_auc(*m, self.W) for m in mats], axis=0)


def false_alert_counts(score: np.ndarray, lab: np.ndarray, taus: np.ndarray) -> np.ndarray:
    """(K,) number of alerts (maximal runs of score >= tau) that contain no anomalous frame, for every tau."""
    above = score[:, None] >= taus[None, :]
    start = above & ~np.vstack([np.zeros((1, above.shape[1]), dtype=bool), above[:-1]])
    total = start.sum(axis=0)
    pos = np.asarray(lab) == 1
    if not pos.any():
        return total
    run_id = np.cumsum(start, axis=0) * above
    r = np.sort(run_id[pos], axis=0)
    with_pos = (r[0] != 0).astype(int) + ((r[1:] != r[:-1]) & (r[1:] != 0)).sum(axis=0)
    return total - with_pos


def op_data(series: list[np.ndarray], labels: list[np.ndarray], events: list[dict], idx_sets: dict[str, list[int]]) -> dict:
    """Per (arm, seed): thresholds = K quantiles of the normal-frame scores (log-spaced tail) + the tau detecting 50 % of the events of each subset."""
    normal = np.concatenate([np.asarray(s)[np.asarray(l) == 0] for s, l in zip(series, labels)])
    levels = 1.0 - np.logspace(np.log10(0.5), -5, GRID_K)
    grid = np.quantile(normal, levels)
    emax = np.array([np.max(series[e["clip"]][e["a"]:e["b"]]) for e in events])
    clip_of = np.array([e["clip"] for e in events])
    extra = []
    for name in SUBSETS:
        m = np.isin(clip_of, idx_sets[name])
        v = np.sort(emax[m])[::-1]
        extra.append(v[int(np.ceil(0.5 * len(v))) - 1] if len(v) else np.inf)
    taus = np.concatenate([grid, extra])
    fa = np.stack([false_alert_counts(np.asarray(s), np.asarray(l), taus) for s, l in zip(series, labels)])  # (clips, K + 3)
    nf = np.array([(np.asarray(l) == 0).sum() for l in labels], dtype=np.float64)
    return {"taus": taus, "fa": fa, "nf": nf, "hit": emax[:, None] >= taus[None, :], "clip_of": clip_of}


def fa_per_hour(od: dict, idx: list[int], col: int, fps: float) -> float:
    return float(od["fa"][idx, col].sum() / (od["nf"][idx].sum() / fps / 3600.0))


def e07_cell_key(ds: str, src: str) -> tuple:
    if src == "ML0_edge":
        return (ds, "ML0_ncnn" if ds == "ShanghaiTech" else "ML0_mps", "botsort", "trajvad_p")
    if src == "YOLO26_e2e":
        return (ds, "yolo26_e2e", "tracktrack", "trajvad_p")
    return None


def load_cells() -> dict[tuple, dict]:
    from edgelab.analysis import q2_p1_e07 as e07

    cells = {}
    for p in sorted(RUN.glob("*.pkl")):
        d = pickle.loads(p.read_bytes())
        cells[(d["dataset"], d["pose"], d["arm"])] = d
    e07cells = e07.load_cells()
    for ds in DATASETS:
        for src in POSE_SOURCES:
            k = e07_cell_key(ds, src)
            if k is not None and k in e07cells:
                cells[(ds, src, "TJP")] = {**e07cells[k], "arm": "TJP", "pose": src}
    return cells


PAIRS = [("P_A", "B1"), ("P_B", "B1"), ("P_C", "B1"), ("P_D", "B1"), ("P_ABC", "B1"), ("B1", "B0"), ("P_ABC", "P_A"), ("P_ABC", "P_B"),
         ("P_ABC", "P_C"), ("P_ABC", "P_D"), ("P_C", "TJP"), ("P_ABC", "TJP"), ("B1", "TJP"), ("P_A", "P_D"), ("B0", "TJP")]


def collect_metrics(ds: str, cells: dict, ctx, events: list[dict], log=print) -> dict:
    """{(src, subset, metric, arm): (point, samples (R,))} for the pooled AUCs (S0, S4), the stratum AUCs (S4), FA/h@R50 and the event recall at B1's FA budget.
    AUC / recall in percent points."""
    from edgelab.analysis import q2_p1_eval as ev

    ab = ev.AucBoot(ctx, N_RESAMPLES, BOOT_SEED)
    idx_sets = {"all": list(range(len(ctx.names))), "dev": ctx.dev_idx, "confirmation": ctx.conf_idx}
    masked = {s: masked_labels(ctx, events, s) for s in STRATA}
    mb = {(s, sub): MaskedBoot(masked[s], idx_sets[sub]) for s in STRATA for sub in SUBSETS}
    out: dict = {}
    clip_of = np.array([e["clip"] for e in events])
    strat_mask = {s: np.array([e[s] for e in events]) for s in STRATA}
    n_clips = len(ctx.names)
    arms_here = lambda src: [a for a in (*EVAL_ARMS, "TJP") if (ds, src, a) in cells]  # noqa: E731
    for src in POSE_SOURCES:
        t0 = time.perf_counter()
        ods = {}
        for arm in arms_here(src):
            d = cells[(ds, src, arm)]
            for proto in ("S0", "S4"):
                series = [[s["series"][proto][i] for i in range(n_clips)] for s in d["seeds"]]
                st = ab.stats(series)
                for sub in SUBSETS:
                    out[(src, sub, f"auc_{proto}", arm)] = (ab.point(st, sub), ab.samples(st, sub))
                if proto == "S4":
                    for s in STRATA:
                        for sub in SUBSETS:
                            b = mb[(s, sub)]
                            if b.valid:
                                mats = b.stats(series)
                                out[(src, sub, f"auc_S4|{s}", arm)] = (b.point(mats), b.samples(mats))
                    ods[arm] = [op_data(sr, ctx.labels, events, idx_sets) for sr in series]
        # FA/h@R50 and event recall at B1's FA/h budget (per subset)
        for sub in SUBSETS:
            idx = idx_sets[sub]
            col50 = GRID_K + SUBSETS.index(sub)
            W = ab.weights[sub]
            in_sub = np.isin(clip_of, idx)
            budget = float(np.mean([fa_per_hour(od, idx, col50, ctx.fps) for od in ods["B1"]])) if "B1" in ods else None
            for arm, odl in ods.items():
                fa_s, rec_s = [], {k: [] for k in ("all", *STRATA)}
                fa_pt, rec_pt = [], {k: [] for k in ("all", *STRATA)}
                for od in odl:
                    nf, fa = od["nf"][idx], od["fa"][idx, col50]
                    fa_s.append((W @ fa) / ((W @ nf) / ctx.fps / 3600.0))
                    fa_pt.append(fa_per_hour(od, idx, col50, ctx.fps))
                    k_star = GRID_K - 1
                    if budget is not None:
                        for k in range(GRID_K):
                            if fa_per_hour(od, idx, k, ctx.fps) <= budget:
                                k_star = k
                                break
                    hit = od["hit"][:, k_star]
                    for key in ("all", *STRATA):
                        m = in_sub & (strat_mask[key] if key != "all" else True)
                        tot = np.bincount(clip_of[m], minlength=n_clips)[idx].astype(float)
                        det = np.bincount(clip_of[m & hit], minlength=n_clips)[idx].astype(float)
                        den = W @ tot
                        rec_s[key].append(np.where(den > 0, (W @ det) / np.maximum(den, 1e-9), np.nan))
                        rec_pt[key].append(100 * det.sum() / tot.sum() if tot.sum() > 0 else np.nan)
                out[(src, sub, "fa_h", arm)] = (float(np.mean(fa_pt)), np.mean(fa_s, axis=0))
                for key in ("all", *STRATA):
                    if np.isfinite(np.nanmean(rec_pt[key])) if len(rec_pt[key]) else False:
                        out[(src, sub, f"recall|{key}", arm)] = (float(np.nanmean(rec_pt[key])), 100 * np.nanmean(rec_s[key], axis=0))
        log(f"  {ds} {src}: metrics done ({time.perf_counter() - t0:.0f}s)")
    return out


def n_events(events: list[dict], ctx, sub: str, stratum: str) -> int:
    idx = {"all": range(len(ctx.names)), "dev": ctx.dev_idx, "confirmation": ctx.conf_idx}[sub]
    return int(sum(1 for e in events if e["clip"] in set(idx) and (stratum == "all" or e[stratum])))


def mean_over_datasets(metrics: dict[str, dict], key: tuple):
    parts = [metrics[ds][key] for ds in DATASETS if key in metrics[ds]]
    if len(parts) != len(DATASETS):
        return None
    return float(np.mean([p[0] for p in parts])), np.mean([p[1] for p in parts], axis=0)


def pct_ci(samples: np.ndarray, metric: str) -> tuple[float, float]:
    """AUC / recall samples are already in percent points; FA/h is in raw units."""
    return float(np.nanquantile(samples, 0.025)), float(np.nanquantile(samples, 0.975))


def contrast_rows(metrics: dict[str, dict]) -> list[dict]:
    rows = []
    keys = {k[:3] for ds in DATASETS for k in metrics[ds]}
    for (src, sub, metric) in sorted(keys):
        for a, b in PAIRS:
            for ds in (*DATASETS, "mean"):
                ka, kb = (src, sub, metric, a), (src, sub, metric, b)
                if ds == "mean":
                    A, B = mean_over_datasets(metrics, ka), mean_over_datasets(metrics, kb)
                else:
                    A, B = metrics[ds].get(ka), metrics[ds].get(kb)
                if A is None or B is None:
                    continue
                d = A[1] - B[1]
                lo, hi = pct_ci(d, metric)
                rows.append({"pose": src, "subset": sub, "metric": metric, "a": a, "b": b, "dataset": ds, "a_value": A[0], "b_value": B[0],
                             "delta": A[0] - B[0], "ci_lo": lo, "ci_hi": hi, "excludes_zero": bool(lo > 0 or hi < 0)})
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    import csv

    with open(path, "w", newline="") as fh:
        fields = list(dict.fromkeys(k for r in rows for k in r))
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def cmd_stats() -> dict:
    """metrics.pkl + results.csv + subgroups.csv + contrasts.csv + per_video.csv (work copies in WORK/report)."""
    from edgelab.analysis import q2_p1_e00 as e00
    from edgelab.analysis import q2_p1_eval as ev

    cells, ctxs = load_cells(), ev.load_ctx()
    (WORK / "report").mkdir(parents=True, exist_ok=True)
    metrics, events_by_ds = {}, {}
    for ds in DATASETS:
        ctx = ctxs[ds]
        events_by_ds[ds] = event_table(ds, ctx)
        print(f"{ds}: {len(events_by_ds[ds])} events; strata counts "
              f"{ {s: sum(e[s] for e in events_by_ds[ds]) for s in STRATA} }", flush=True)
        metrics[ds] = collect_metrics(ds, cells, ctx, events_by_ds[ds], log=lambda m: print(m, flush=True))
    contrasts = contrast_rows(metrics)
    # ---- results.csv (per seed and seed mean with bootstrap CI), per_video.csv
    results, per_video = [], []
    for (ds, src, arm), d in sorted(cells.items()):
        ctx = ctxs[ds]
        for proto in ("S0", "S4"):
            for subset in SUBSETS:
                for sd, s in enumerate(d["seeds"]):
                    results.append({"dataset": ds, "pose": src, "arm": ARM_LABEL[arm], "seed": sd, "protocol": proto, "subset": subset, "auc": s["auc"][proto][subset],
                                    "ci_lo": "", "ci_hi": "", "coverage": s["coverage"]["all"]})
                pt, smp = metrics[ds][(src, subset, f"auc_{proto}", arm)]
                lo, hi = pct_ci(smp, "auc")
                results.append({"dataset": ds, "pose": src, "arm": ARM_LABEL[arm], "seed": "mean", "protocol": proto, "subset": subset, "auc": pt, "ci_lo": lo,
                                "ci_hi": hi, "coverage": float(np.mean([s["coverage"]["all"] for s in d["seeds"]]))})
        dev = set(ctx.dev_idx)
        for i, clip in enumerate(ctx.names):
            lab = np.asarray(ctx.labels[i])
            if lab.min() == lab.max():
                continue
            aucs = [100 * e00.pooled_auc(ctx.labels, [s["series"]["S4"][j] for j in range(len(ctx.names))], [i]) for s in d["seeds"]]
            per_video.append({"dataset": ds, "pose": src, "arm": ARM_LABEL[arm], "video": clip, "split": "dev" if i in dev else "confirmation", "protocol": "S4",
                              "auc_seed_mean": float(np.mean(aucs)), "auc_seed_sd": float(np.std(aucs))})
    sub_rows = []
    for ds in (*DATASETS, "mean"):
        for (src, sub, metric, arm) in sorted({k for d_ in DATASETS for k in metrics[d_]}):
            if metric in ("auc_S0",):
                continue
            if ds == "mean":
                v = mean_over_datasets(metrics, (src, sub, metric, arm))
                ref = mean_over_datasets(metrics, (src, sub, metric, "B1"))
            else:
                v, ref = metrics[ds].get((src, sub, metric, arm)), metrics[ds].get((src, sub, metric, "B1"))
            if v is None:
                continue
            lo, hi = pct_ci(v[1], metric)
            row = {"dataset": ds, "pose": src, "subset": sub, "protocol": "S4", "metric": metric, "arm": ARM_LABEL[arm],
                   "family": next((f for f, ss in FAMILIES.items() if metric.split("|")[-1] in ss), "pooled"),
                   "stratum": metric.split("|")[1] if "|" in metric else "all", "value": v[0], "ci_lo": lo, "ci_hi": hi}
            if ds != "mean":
                row["n_events"] = n_events(events_by_ds[ds], ctxs[ds], sub, row["stratum"])
            if ref is not None and arm != "B1":
                dd = v[1] - ref[1]
                row["delta_vs_B1"], (row["d_ci_lo"], row["d_ci_hi"]) = v[0] - ref[0], pct_ci(dd, metric)
            sub_rows.append(row)
    write_csv(WORK / "report" / "results.csv", results)
    write_csv(WORK / "report" / "subgroups.csv", sub_rows)
    write_csv(WORK / "report" / "contrasts.csv", contrasts)
    write_csv(WORK / "report" / "per_video.csv", per_video)
    (WORK / "report" / "metrics.pkl").write_bytes(pickle.dumps({"contrasts": contrasts, "events": events_by_ds,
                                                                "point": {ds: {k: v[0] for k, v in metrics[ds].items()} for ds in DATASETS}}))
    print(f"results {len(results)} rows, subgroups {len(sub_rows)}, contrasts {len(contrasts)}, per-video {len(per_video)}", flush=True)
    return {"results": results, "subgroups": sub_rows, "contrasts": contrasts, "events": events_by_ds}


# =====================================================================================================================
# Raspberry Pi 5: checkpoints out, results back, delivered AUC
# =====================================================================================================================
PI_HOST = "nanh@raspberrypi.local"
PI_CKPT = WORK / "ckpt_pi"
PI_OUT = THESIS / "q2-p1-e08-pi5"
PI_ARMS = ("B0", "B1", "P_A", "P_B", "P_C", "P_ABC", "TJP")


def cmd_pi_prep() -> None:
    """Seed-0 ShanghaiTech checkpoints of every Pi arm -> ~/thesis-data/q2-p1-e08/ckpt on the Pi; code -> ~/edgelab-e07/src (the tree the E07 Pi runs used)."""
    import shutil
    import subprocess

    src = {"B0": (E06_MODELS / "A1_ShanghaiTech_s0", "e06", False, False), "B1": (E06_MODELS / "A2_ShanghaiTech_s0", "e06", False, False)}
    for arm in TRAIN_ARMS:
        f = arm_flags(arm)
        src[arm] = (model_dir(arm, "ShanghaiTech", 0), "e08", f["ctx"], f["smooth"])
    PI_CKPT.mkdir(parents=True, exist_ok=True)
    for name, (d, source, ctx, smooth) in src.items():
        if not (d / "DONE").exists():
            raise FileNotFoundError(d)
        shutil.copy(d / "final.pt", PI_CKPT / f"{name}.pt")
        (PI_CKPT / f"{name}.json").write_text(json.dumps({"source": source, "arm": name, "ctx": bool(ctx), "smooth": bool(smooth), "src_dir": str(d)}))
    root = Path(__file__).resolve().parents[3]
    run = lambda cmd: subprocess.run(cmd, check=True)  # noqa: E731
    run(["ssh", PI_HOST, "mkdir -p ~/thesis-data/q2-p1-e08/ckpt ~/thesis-data/q2-p1-e08-pi5"])
    run(["rsync", "-a", f"{PI_CKPT}/", f"{PI_HOST}:thesis-data/q2-p1-e08/ckpt/"])
    run(["rsync", "-a", str(root / "src/edgelab/models/lorad_gaps.py"), f"{PI_HOST}:edgelab-e07/src/edgelab/models/"])
    run(["rsync", "-a", str(root / "src/edgelab/analysis/q2_p1_e08_pi5.py"), f"{PI_HOST}:edgelab-e07/src/edgelab/analysis/"])
    print("Pi 5 prepared:", sorted(p.name for p in PI_CKPT.glob("*.pt")), flush=True)


def pi_scorer_sentinel(key: str) -> float:
    """Train-calibrated S4 sentinel (SHT quantile) of the seed-0 checkpoint the Pi runs."""
    from edgelab.analysis import q2_p1_e07 as e07
    from edgelab.analysis import q2_p1_eval as ev

    ds = "ShanghaiTech"
    if key == "TJP":
        scorer, sid = e07.scorers_for("trajvad_p", ds)[0]
        return ev.train_sentinels(scorer, ds, e07.WORK / "sentinels" / f"{sid}.json")[ev.QUANTILE[ds]]
    scorer, path, _, smooth, ctx = arm_scorers(key, ds)[0]
    return gap_sentinels(scorer, ds, smooth, ctx, path)[ev.QUANTILE[ds]]


def cmd_pi_report() -> list[dict]:
    """pi5_benchmark rows: latency / power / thermal per arm + the delivered real-time AUC on the 12 clips (E00 S4 landing, train-calibrated sentinel)."""
    from edgelab.analysis import q2_p1_e00 as e00
    from edgelab.analysis import q2_p1_eval as ev

    ds = "ShanghaiTech"
    ctx = ev.load_ctx()[ds]
    cells = load_cells()
    rows, sentinels = [], {}
    for sf in sorted(PI_OUT.glob("*_summary.json")):
        s = json.loads(sf.read_text())
        key = s.get("scorer")
        key = "TJP" if key == "trajvad_p" else key
        if key not in sentinels:
            sentinels[key] = pi_scorer_sentinel(key)
        z = np.load(PI_OUT / f"{s['arm']}_raw.npz")
        clips = sorted({k.split("__", 1)[1] for k in z.files})
        idx = [ctx.names.index(c) for c in clips]
        labels = [ctx.labels[i][: len(z[f"all__{c}"])] for i, c in zip(idx, clips)]
        tel = s["telemetry"]
        n_proc = s["frames_processed"]
        row = {"arm": s["arm"], "scorer": ARM_LABEL.get(key, key), "variant": "Cache-C1+Guard" if s["use_c1"] else "Full-Rate", "detector": "ML0 YOLOv8n-pose NCNN FP16 384x640",
               "tracker": "BoT-SORT two-stage (E01 locked)", "threads": s["threads"], "frames": s["frames"], "frames_processed": n_proc, "delivered_fps": s["delivered_fps"],
               "throughput_fps_processed": s.get("throughput_fps_processed"), "score_ms": s["score_ms_mean"], "score_ms_p95": s["score_ms_p95"], "end_to_end_ms": s["service_ms_mean"],
               "end_to_end_ms_p95": s["service_ms_p95"], "end_to_end_ms_p99": s["service_ms_p99"], "decode_ms": s["decode_ms_mean"], "pose_ms": s["pose_ms_mean"],
               "track_ms": s["track_ms_mean"], "c1_ms": s["c1_check_ms_mean"], "queue_wait_p99_ms": s["queue_wait_ms_p99"], "queue_wait_max_ms": s["queue_wait_ms_max"],
               "c1_hit_pct": s["c1_hit_rate_pct"], "guard_shed_pct": s["guard_shed_pct"], "sustainable_24fps": s["Sustainable_24fps"], "power_w": tel["power_total_w_mean"],
               "power_w_max": tel["power_total_w_max"], "j_per_processed_frame": s["energy_j_per_processed_frame"],
               "j_per_input_frame": round(tel["power_total_w_mean"] * s["wall_total_s"] / s["frames"], 4), "temp_c_mean": tel["temp_c_mean"], "temp_c_max": tel["temp_c_max"],
               "start_temp_c": s.get("start_temp_c"), "rss_mb_peak": tel["rss_mb_peak"], "sys_cpu_pct_mean": tel["sys_cpu_pct_mean"], "throttled_any": tel.get("throttled_any"),
               "throttled_codes": "|".join(tel.get("throttled_codes_seen", [])), "windows_ontime": s["windows_delivered_ontime"], "windows_all": s["windows_delivered_all"],
               "run_id": s.get("mlflow_run_id", "")}
        for mode in ("ontime", "all"):
            raw = [z[f"{mode}__{c}"] for c in clips]
            series, _ = e00.deliver(raw, sentinels[key], 0)
            row[f"delivered_auc_{mode}"] = 100 * float(e00.pooled_auc(labels, series, list(range(len(clips)))))
            row[f"coverage_{mode}"] = float(np.mean(np.concatenate([np.isfinite(r) for r in raw])))
        cell = cells.get((ds, PRIMARY, key))
        if cell is not None:
            ser = cell["seeds"][0]["series"]["S4"]
            row["offline_ideal_auc_same_clips"] = 100 * float(e00.pooled_auc(labels, [ser[i][: len(labels[j])] for j, i in enumerate(idx)], list(range(len(clips)))))
        rows.append(row)
    order = {a: i for i, a in enumerate(PI_ARMS)}
    rows.sort(key=lambda r: (order.get(next((k for k, v in ARM_LABEL.items() if v == r["scorer"]), "TJP"), 99), r["variant"]))
    if rows:
        write_csv(WORK / "report" / "pi5_benchmark.csv", rows)
    return rows


# =====================================================================================================================
# Verdicts
# =====================================================================================================================
def pick(contrasts: list[dict], **kw):
    return next((c for c in contrasts if all(c.get(k) == v for k, v in kw.items())), None)


def count_params() -> dict:
    out = {}
    for arm in (*E06_ARM, *TRAIN_ARMS):
        for ds in DATASETS:
            d = (E06_MODELS / f"{E06_ARM[arm]}_{ds}_s0") if arm in E06_ARM else model_dir(arm, ds, 0)
            if (d / "config.json").exists():
                out[(arm, ds)] = json.loads((d / "config.json").read_text())["params"]
    out[("P_D", "ShanghaiTech")], out[("P_D", "UBnormal")] = out.get(("B1", "ShanghaiTech")), out.get(("B1", "UBnormal"))
    tj = THESIS / "q2-p1-e07" / "models" / "TrajVAD-P_ShanghaiTech_s0" / "config.json"
    if tj.exists():
        out[("TJP", "ShanghaiTech")] = json.loads(tj.read_text())["params"]
    return out


def verdicts(contrasts: list[dict], pi_rows: list[dict], params: dict) -> dict:
    P = dict(pose=PRIMARY, subset="confirmation", dataset="mean")
    ok = lambda c, f: bool(c is not None and f(c))  # noqa: E731
    out = {}

    def label(support: bool, partial: bool, refuted: bool) -> str:
        return "REFUTED" if refuted else "SUPPORTED" if support else "PARTIAL" if partial else "NOT SUPPORTED"

    pooled = lambda a: pick(contrasts, metric="auc_S4", a=a, b="B1", **P)  # noqa: E731
    stratum = lambda a, st, m="auc_S4": pick(contrasts, metric=f"{m}|{st}", a=a, b="B1", **P)  # noqa: E731
    fa = lambda a: pick(contrasts, metric="fa_h", a=a, b="B1", **P)  # noqa: E731
    # H_A
    pa, sh, su = pooled("P_A"), stratum("P_A", "short_event"), stratum("P_A", "sustained_event")
    c_a = {"pooled_ci_lo_gt_0": ok(pa, lambda c: c["ci_lo"] > 0), "short_ci_lo_gt_0": ok(sh, lambda c: c["ci_lo"] > 0), "sustained_point_gt_0": ok(su, lambda c: c["delta"] > 0)}
    strata_ok = c_a["short_ci_lo_gt_0"] and c_a["sustained_point_gt_0"]
    out["H_A"] = {"verdict": label(c_a["pooled_ci_lo_gt_0"] and strata_ok, c_a["pooled_ci_lo_gt_0"] or strata_ok, ok(pa, lambda c: c["ci_hi"] < 0)), "conditions": c_a,
                  "evidence": {"pooled": pa, "short_event": sh, "sustained_event": su}}
    # H_B
    pb, fb = pooled("P_B"), fa("P_B")
    pi_lat = None
    if pi_rows:
        b1 = next((r for r in pi_rows if r["scorer"] == ARM_LABEL["B1"] and r["variant"] == "Full-Rate"), None)
        pbr = next((r for r in pi_rows if r["scorer"] == ARM_LABEL["P_B"] and r["variant"] == "Full-Rate"), None)
        if b1 and pbr:
            pi_lat = {"B1_score_ms": b1["score_ms"], "P_B_score_ms": pbr["score_ms"], "ratio": pbr["score_ms"] / b1["score_ms"], "within_5pct": abs(pbr["score_ms"] / b1["score_ms"] - 1) <= 0.05}
    c_b = {"pooled_ci_lo_gt_0": ok(pb, lambda c: c["ci_lo"] > 0), "fa_h_ci_hi_lt_0": ok(fb, lambda c: c["ci_hi"] < 0),
           "params_equal_B1": all(params.get(("P_B", ds)) == params.get(("B1", ds)) for ds in DATASETS if ("P_B", ds) in params),
           "pi5_scorer_latency_within_5pct": None if pi_lat is None else pi_lat["within_5pct"]}
    out["H_B"] = {"verdict": label(c_b["pooled_ci_lo_gt_0"] and c_b["fa_h_ci_hi_lt_0"], c_b["pooled_ci_lo_gt_0"] or c_b["fa_h_ci_hi_lt_0"], ok(pb, lambda c: c["ci_hi"] < 0) or ok(fb, lambda c: c["ci_lo"] > 0)),
                  "conditions": c_b, "evidence": {"pooled": pb, "fa_h": fb, "pi5_latency": pi_lat}}
    # H_C
    pc, fm, hc = pooled("P_C"), stratum("P_C", "fast_motion"), stratum("P_C", "high_crowd")
    rf, rh = stratum("P_C", "fast_motion", "recall"), stratum("P_C", "high_crowd", "recall")
    vs_tj = pick(contrasts, metric="auc_S4", a="P_C", b="TJP", **P)
    tj_row = next((r for r in pi_rows if r["scorer"] == ARM_LABEL["TJP"] and r["variant"] == "Full-Rate"), None)
    pc_row = next((r for r in pi_rows if r["scorer"] == ARM_LABEL["P_C"] and r["variant"] == "Full-Rate"), None)
    ratio = (params.get(("P_C", "ShanghaiTech"), np.nan) / params[("TJP", "ShanghaiTech")]) if ("TJP", "ShanghaiTech") in params else None
    speed = (tj_row["score_ms"] / pc_row["score_ms"]) if (tj_row and pc_row) else None
    c_c = {"fast_motion_ci_lo_gt_0": ok(fm, lambda c: c["ci_lo"] > 0), "high_crowd_ci_lo_gt_0": ok(hc, lambda c: c["ci_lo"] > 0),
           "recall_fast_point_gt_0": ok(rf, lambda c: c["delta"] > 0), "recall_crowd_point_gt_0": ok(rh, lambda c: c["delta"] > 0),
           "vs_trajvad_p_ci_lo_ge_-2": ok(vs_tj, lambda c: c["ci_lo"] >= -2.0), "params_ratio_le_1_300": None if ratio is None else bool(ratio <= 1 / 300),
           "pi5_scorer_speedup_ge_4": None if speed is None else bool(speed >= 4.0)}
    acc = all(c_c[k] for k in ("fast_motion_ci_lo_gt_0", "high_crowd_ci_lo_gt_0", "recall_fast_point_gt_0", "recall_crowd_point_gt_0"))
    comp = all(bool(c_c[k]) for k in ("vs_trajvad_p_ci_lo_ge_-2", "params_ratio_le_1_300", "pi5_scorer_speedup_ge_4"))
    out["H_C"] = {"verdict": label(acc and comp, acc or comp, ok(fm, lambda c: c["ci_hi"] < 0) and ok(hc, lambda c: c["ci_hi"] < 0)), "conditions": c_c, "accuracy_part": acc, "competitiveness_part": comp,
                  "evidence": {"pooled": pc, "fast_motion": fm, "high_crowd": hc, "recall_fast": rf, "recall_crowd": rh, "vs_trajvad_p": vs_tj, "params_ratio": ratio, "pi5_speedup": speed}}
    # H_D
    pd_, fd = pooled("P_D"), fa("P_D")
    c_d = {"fa_h_ci_hi_lt_0": ok(fd, lambda c: c["ci_hi"] < 0), "pooled_ci_lo_ge_-0.5": ok(pd_, lambda c: c["ci_lo"] >= -0.5)}
    out["H_D"] = {"verdict": label(all(c_d.values()), any(c_d.values()), ok(fd, lambda c: c["ci_lo"] > 0) and ok(pd_, lambda c: c["ci_hi"] < 0)), "conditions": c_d, "evidence": {"pooled": pd_, "fa_h": fd}}
    # H_Joint
    cache_rows = [r for r in pi_rows if r["variant"] == "Cache-C1+Guard"]
    joint = next((r for r in cache_rows if r["scorer"] == ARM_LABEL["P_ABC"]), None)
    if joint:
        best = max(cache_rows, key=lambda r: r["delivered_auc_ontime"])
        c_j = {"highest_delivered_auc_among_cache_arms": best["scorer"] == joint["scorer"], "delivered_fps_ge_24": joint["delivered_fps"] >= 24.0,
               "queue_wait_p99_lt_500ms": joint["queue_wait_p99_ms"] < 500.0, "throttled_0x0": not joint["throttled_any"]}
        n = sum(c_j.values())
        out["H_Joint"] = {"verdict": label(n == 4, n >= 3 or c_j["highest_delivered_auc_among_cache_arms"], False), "conditions": c_j,
                          "evidence": {"P_ABC": {k: joint[k] for k in ("delivered_auc_ontime", "delivered_fps", "queue_wait_p99_ms", "throttled_codes", "score_ms", "end_to_end_ms")},
                                       "best_cache_arm": {"scorer": best["scorer"], "delivered_auc_ontime": best["delivered_auc_ontime"]},
                                       "ranking": [(r["scorer"], round(r["delivered_auc_ontime"], 2)) for r in sorted(cache_rows, key=lambda r: -r["delivered_auc_ontime"])]}}
    else:
        out["H_Joint"] = {"verdict": "NOT RUN", "conditions": {}}
    return out


# =====================================================================================================================
# Final tables, summary.json, report.md
# =====================================================================================================================
def fmt(x, nd=2, plus=False):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    return f"{x:+.{nd}f}" if plus else f"{x:.{nd}f}"


def contrast_str(c: dict | None, nd=2) -> str:
    if c is None:
        return "n/a"
    return f"{c['delta']:+.{nd}f} [{c['ci_lo']:+.{nd}f}, {c['ci_hi']:+.{nd}f}]{'*' if c['excludes_zero'] else ''}"


def md_table(header: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def train_diag() -> list[dict]:
    rows = []
    for arm in TRAIN_ARMS:
        for ds in DATASETS:
            for seed in SEEDS:
                p = model_dir(arm, ds, seed) / "config.json"
                if not p.exists():
                    continue
                c = json.loads(p.read_text())
                h = c.get("history", c.get("epoch_losses", []))
                last = h[-1] if h else {}
                rows.append({"arm": ARM_LABEL[arm], "dataset": ds, "seed": seed, "params": c["params"], "n_windows": c.get("n_windows"), "minutes": c.get("train_minutes"),
                             **{k: (last.get(k) if isinstance(last, dict) else last) for k in ("nll", "rec", "hinge", "gap")}})
    return rows


def repro_check(cells: dict) -> list[dict]:
    """B1 / B0 cells copied from E06 vs the same arm recomputed through E08's own scorer path (ShanghaiTech released, seed 0): must agree to numerical precision."""
    from edgelab.analysis import q2_p1_eval as ev

    out = []
    ds, src = "ShanghaiTech", "released"
    ctx = ev.load_ctx()[ds]
    for arm in ("B1",):
        scorer, sent_path, sid, smooth, use_ctx = arm_scorers(arm, ds)[0]
        sent = gap_sentinels(scorer, ds, smooth, use_ctx, sent_path)[ev.QUANTILE[ds]]
        tracks = pose_tracks(ds, src, ctx, 4)
        res = ev.evaluate_table(ev.build_table(tracks, scorer), tracks, ctx, scorer.seg_len, sent)
        ref = cells[(ds, src, arm)]["seeds"][0]
        out.append({"arm": arm, "dataset": ds, "pose": src, "e06_S4_all": ref["auc"]["S4"]["all"], "recomputed_S4_all": res["auc"]["S4"]["all"],
                    "abs_diff": abs(ref["auc"]["S4"]["all"] - res["auc"]["S4"]["all"])})
    return out


def cmd_final(skip_repro: bool = False) -> None:
    import shutil

    rep = WORK / "report"
    stats = cmd_stats() if not (rep / "metrics.pkl").exists() else None
    meta = pickle.loads((rep / "metrics.pkl").read_bytes())
    contrasts = meta["contrasts"]
    pi_rows = cmd_pi_report() if list(PI_OUT.glob("*_summary.json")) else []
    params = count_params()
    verd = verdicts(contrasts, pi_rows, params)
    EXP_DIR.mkdir(exist_ok=True)
    for f in ("results.csv", "subgroups.csv", "contrasts.csv", "per_video.csv", "pi5_benchmark.csv"):
        if (rep / f).exists():
            shutil.copy(rep / f, EXP_DIR / f)
    cells = load_cells()
    repro = [] if skip_repro else repro_check(cells)
    diag = train_diag()
    if diag:
        write_csv(EXP_DIR / "training_diagnostics.csv", diag)
    summary = {"experiment": "Q2-P1-E08", "verdicts": {k: {"verdict": v["verdict"], "conditions": v["conditions"]} for k, v in verd.items()}, "evidence": {k: v.get("evidence") for k, v in verd.items()},
               "params": {f"{a}|{d}": v for (a, d), v in params.items()}, "reproduction_check": repro, "pi5": pi_rows, "primary_cell": {"pose": PRIMARY, "protocol": "S4", "subset": "confirmation"},
               "n_resamples": N_RESAMPLES, "seeds": list(SEEDS), "events": {ds: len(meta["events"][ds]) for ds in DATASETS},
               "strata_counts": {ds: {s: int(sum(e[s] for e in meta["events"][ds])) for s in STRATA} for ds in DATASETS}}
    (EXP_DIR / "summary.json").write_text(json.dumps(summary, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    (EXP_DIR / "report.md").write_text(write_report(contrasts, pi_rows, params, verd, repro, diag, meta))
    print((EXP_DIR / "report.md").name, "written;", {k: v["verdict"] for k, v in verd.items()}, flush=True)


def write_report(contrasts, pi_rows, params, verd, repro, diag, meta) -> str:
    import csv

    L: list[str] = ["# Q2-P1-E08: LoRaD gap proposals (P_A, P_B, P_C, P_D, P_ABC)", "",
                    "All numbers are produced by `src/edgelab/analysis/q2_p1_e08.py` (stages `train`, `run`, `stats`, `pi-prep`/`pi-report`, `final`) on the pre-registered `plan.md` / `config.yaml`. "
                    "AUCs are in percent; deltas carry 2,000-resample paired video-bootstrap 95% CIs (`*` = CI excludes 0). The primary cell is ML0_edge + locked two-stage tracker, S4, "
                    "test_confirmation, mean over the two datasets, paired against B1_LoRaD.", ""]
    L += ["## 1. Verdicts", ""]
    rows = []
    for h in ("H_A", "H_B", "H_C", "H_D", "H_Joint"):
        v = verd[h]
        rows.append([h, f"**{v['verdict']}**", "; ".join(f"{k}={'n/a' if x is None else ('yes' if x else 'no')}" for k, x in v["conditions"].items())])
    L += [md_table(["Hypothesis", "Verdict", "Pre-registered conditions"], rows), ""]
    P = dict(pose=PRIMARY, subset="confirmation", dataset="mean")
    L += ["## 2. Primary contrasts (ML0_edge, S4, confirmation, dataset mean)", ""]
    rows = []
    for a in ("P_A", "P_B", "P_C", "P_D", "P_ABC"):
        g = lambda m, b="B1": contrast_str(pick(contrasts, metric=m, a=a, b=b, **P))  # noqa: E731
        rows.append([ARM_LABEL[a], g("auc_S4"), g("auc_S0"), g("fa_h"), g("recall|all"), g("auc_S4", "TJP")])
    L += [md_table(["Arm", "dAUC S4 vs B1", "dAUC S0 vs B1", "dFA/h@R50 vs B1", "dRecall@FA(B1) vs B1", "dAUC S4 vs TrajVAD-P"], rows), ""]
    for ds in DATASETS:
        L += [f"### Pooled AUC (S4 / S0) by pose source, {ds}", ""]
        rows = []
        pts = meta["point"][ds]
        for src in POSE_SOURCES:
            for sub in ("dev", "confirmation", "all"):
                rows.append([src, sub] + [f"{fmt(pts.get((src, sub, 'auc_S4', a)))} / {fmt(pts.get((src, sub, 'auc_S0', a)))}" for a in (*EVAL_ARMS, "TJP")])
        L += [md_table(["Pose", "Subset", *[ARM_LABEL[a] for a in (*EVAL_ARMS, "TJP")]], rows), ""]
    L += ["## 3. Pose-source robustness (S4, confirmation, paired vs B1, dataset mean)", ""]
    rows = []
    for src in POSE_SOURCES:
        for a in ("P_A", "P_B", "P_C", "P_D", "P_ABC"):
            c = pick(contrasts, pose=src, subset="confirmation", dataset="mean", metric="auc_S4", a=a, b="B1")
            rows.append([src, ARM_LABEL[a], contrast_str(c)])
    L += [md_table(["Pose", "Arm", "dAUC vs B1 [95% CI]"], rows), ""]
    L += ["## 4. Subgroups (S4, ML0_edge, confirmation, dataset mean; AUC points, delta vs B1)", ""]
    sub = list(csv.DictReader(open(EXP_DIR / "subgroups.csv")))
    rows = []
    for st in STRATA:
        for a in ("B1", "P_A", "P_B", "P_C", "P_D", "P_ABC"):
            r = next((x for x in sub if x["dataset"] == "mean" and x["pose"] == PRIMARY and x["subset"] == "confirmation" and x["metric"] == f"auc_S4|{st}" and x["arm"] == ARM_LABEL[a]), None)
            rc = next((x for x in sub if x["dataset"] == "mean" and x["pose"] == PRIMARY and x["subset"] == "confirmation" and x["metric"] == f"recall|{st}" and x["arm"] == ARM_LABEL[a]), None)
            if r is None:
                continue
            d = "" if a == "B1" else f"{float(r['delta_vs_B1']):+.2f} [{float(r['d_ci_lo']):+.2f}, {float(r['d_ci_hi']):+.2f}]"
            dr = "" if (a == "B1" or rc is None or "delta_vs_B1" not in rc or rc["delta_vs_B1"] == "") else f"{float(rc['delta_vs_B1']):+.1f}"
            rows.append([st, ARM_LABEL[a], fmt(float(r["value"])), d, "" if rc is None else fmt(float(rc["value"]), 1), dr])
    L += [md_table(["Stratum", "Arm", "AUC", "dAUC vs B1 [CI]", "Recall@FA(B1) %", "dRecall pts"], rows), "",
          f"Event counts (all clips): {json.dumps({ds: {s: int(sum(e[s] for e in meta['events'][ds])) for s in STRATA} for ds in DATASETS})}.", ""]
    L += ["### False alerts per hour at Recall 50%", ""]
    rows = []
    for ds in (*DATASETS, "mean"):
        for a in (*EVAL_ARMS, "TJP"):
            r = next((x for x in sub if x["dataset"] == ds and x["pose"] == PRIMARY and x["subset"] == "confirmation" and x["metric"] == "fa_h" and x["arm"] == ARM_LABEL[a]), None)
            if r:
                rows.append([ds, ARM_LABEL[a], fmt(float(r["value"]), 1), f"[{float(r['ci_lo']):.1f}, {float(r['ci_hi']):.1f}]", "" if a == "B1" or "delta_vs_B1" not in r or r["delta_vs_B1"] == "" else f"{float(r['delta_vs_B1']):+.1f} [{float(r['d_ci_lo']):+.1f}, {float(r['d_ci_hi']):+.1f}]"])
    L += [md_table(["Dataset", "Arm", "FA/h@R50", "95% CI", "delta vs B1 [CI]"], rows), ""]
    L += ["## 5. Parameter budgets", "", md_table(["Arm", "ShanghaiTech", "UBnormal"], [[ARM_LABEL.get(a, a), params.get((a, "ShanghaiTech"), "n/a"), params.get((a, "UBnormal"), "n/a")] for a in (*E06_ARM, *TRAIN_ARMS, "P_D", "TJP")]), ""]
    if diag:
        L += ["## 6. Training diagnostics (last epoch, per run)", "", md_table(["Arm", "Dataset", "Seed", "Params", "NLL (bits/dim)", "Reverse term", "Hinge", "Gap"],
                                                                     [[d["arm"], d["dataset"], d["seed"], d["params"], fmt(d["nll"], 4), fmt(d["rec"], 8), fmt(d["hinge"], 4), fmt(d["gap"], 4)] for d in diag]), "",
              "The reverse-consistency term is the round-off of an exactly invertible flow; P_B's effect is therefore attributable to the margin term.", ""]
    if pi_rows:
        L += ["## 7. Raspberry Pi 5 benchmark (4 threads, ML0 NCNN FP16 + BoT-SORT two-stage)", "",
              md_table(["Scorer", "Variant", "Frames", "Deliv. FPS", "Score ms", "E2E ms", "Wait p99 ms", "W", "J/proc frame", "J/input frame", "Tmax C", "RSS MB", "Throttle", "Deliv. AUC (on-time)", "Deliv. AUC (all)", "Offline AUC same clips"],
                       [[r["scorer"], r["variant"], r["frames"], fmt(r["delivered_fps"]), fmt(r["score_ms"], 3), fmt(r["end_to_end_ms"], 1), fmt(r["queue_wait_p99_ms"], 0), fmt(r["power_w"]), fmt(r["j_per_processed_frame"], 3),
                         fmt(r["j_per_input_frame"], 3), fmt(r["temp_c_max"], 1), fmt(r["rss_mb_peak"], 0), r["throttled_codes"] or "0x0", fmt(r["delivered_auc_ontime"]), fmt(r["delivered_auc_all"]),
                         fmt(r.get("offline_ideal_auc_same_clips"))] for r in pi_rows]), ""]
    L += ["## 8. Evidence behind each verdict", ""]
    for h in ("H_A", "H_B", "H_C", "H_D", "H_Joint"):
        L += [f"- **{h}** {verd[h]['verdict']}: `{json.dumps(verd[h].get('evidence'), default=lambda o: o.item() if hasattr(o, 'item') else str(o))[:1500]}`"]
    L += ["", "## 9. Reproduction check of the baseline pipeline", "", md_table(["Arm", "Dataset", "Pose", "E06 S4 all", "Recomputed S4 all", "abs diff"],
                                                                        [[r["arm"], r["dataset"], r["pose"], fmt(r["e06_S4_all"], 4), fmt(r["recomputed_S4_all"], 4), f"{r['abs_diff']:.2e}"] for r in repro]) if repro else "_skipped_", ""]
    L += ["## 10. Deviations, caveats and limits", "",
          "- **Amendment A1 (training parallelism).** 24 models were trained by 5 single-thread CPU workers inside this experiment's training stage only. All stages and all experiments (Mac vs. e06/e07 and the Pi) ran strictly sequentially; strict one-model-at-a-time training would have taken ~25 h.",
          "- N0 (per-window mean/std_y) removes absolute position but retains within-window drift; the source plan's statement that translation is 'completely erased' is overstated.",
          "- The reverse-consistency loss is zero up to round-off for an exactly invertible flow (see section 6); the P_B hypothesis therefore reduces to the margin term.",
          "- P_C adds no parameters (not ~0.8K): the context nodes are extra graph nodes sharing the same subnet; the dense max_hop=8 adjacency means hub edges do not sparsify A. Node 19 carries velocity only (aspect is not fed to the flow).",
          "- P_A has about 2.5x the baseline's parameters; H_A is a statement about accuracy at that cost.",
          "- The Pi 5 benchmark ran on ShanghaiTech seed-0 checkpoints and 12 clips (same sample as E07); P_D is eval-only (no Pi arm, it shares B1's weights).",
          "- Event strata come from the released poses (duration, per-track median box speed >= 0.06 h/frame, mean concurrency >= 4) and are shared by all pose sources.", ""]
    return "\n".join(L)


# =====================================================================================================================
# MLflow: training / matrix / Pi runs + model registry (edgelab-ms-iac-revflow)
# =====================================================================================================================
def cmd_mlflow() -> None:
    import os
    import tempfile

    os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
    import mlflow
    from mlflow.tracking import MlflowClient

    uri = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment("Q2-P1-E08")
    summary = json.loads((EXP_DIR / "summary.json").read_text())
    with mlflow.start_run(run_name="gap_training") as parent:
        mlflow.set_tags({"stage": "training", "amendment": "A1: 5 single-thread workers"})
        for arm in TRAIN_ARMS:
            for ds in DATASETS:
                for seed in SEEDS:
                    d = model_dir(arm, ds, seed)
                    if not (d / "config.json").exists():
                        continue
                    cfg = json.loads((d / "config.json").read_text())
                    with mlflow.start_run(run_name=d.name, nested=True):
                        mlflow.log_params({"arm": arm, "dataset": ds, "seed": seed, "params": cfg["params"], "seg_len": cfg.get("seg_len"), "epochs": cfg.get("epochs")})
                        for e, h in enumerate(cfg.get("history", [])):
                            mlflow.log_metrics({k: float(v) for k, v in (h.items() if isinstance(h, dict) else [("nll", h)]) if isinstance(v, (int, float))}, step=e + 1)
                        mlflow.log_artifact(str(d / "config.json"))
    with mlflow.start_run(run_name="offline_matrix") as run:
        mlflow.set_tags({"stage": "matrix"})
        for h, v in summary["verdicts"].items():
            mlflow.set_tag(f"verdict_{h}", v["verdict"])
        for c in pickle.loads((WORK / "report" / "metrics.pkl").read_bytes())["contrasts"]:
            if c["dataset"] == "mean" and c["subset"] == "confirmation" and c["pose"] == PRIMARY and c["metric"] in ("auc_S4", "fa_h") and c["b"] == "B1":
                mlflow.log_metric(f"d{c['metric']}__{c['a']}_vs_B1", c["delta"])
        for name in ("results.csv", "subgroups.csv", "contrasts.csv", "pi5_benchmark.csv", "summary.json", "plan.md", "config.yaml", "report.md"):
            if (EXP_DIR / name).exists():
                mlflow.log_artifact(str(EXP_DIR / name))
    traced_clip_demo()
    mlflow.flush_trace_async_logging()
    # ---- registry
    client = MlflowClient(tracking_uri=uri)
    name = "edgelab-ms-iac-revflow"
    desc = ("MS-IAC-RevFlow (P_ABC): LoRaD causal flow + multi-scale causal dilated temporal convolutions (d in {1,2,4}), two virtual context nodes "
            "(box kinematics, nearest-neighbour proximity), reverse-margin training objective and causal confidence-weighted smoothing. Trained here (E08).")
    tags = {"task": "skeleton_anomaly_scoring", "architecture": "MS-IAC-RevFlow", "experiment": "Q2-P1-E08", "verdicts": json.dumps({k: v["verdict"] for k, v in summary["verdicts"].items()})}
    try:
        client.create_registered_model(name=name, description=desc, tags=tags)
    except Exception:
        client.update_registered_model(name=name, description=desc)
        for k, v in tags.items():
            client.set_registered_model_tag(name, k, v)
    have = {v.tags.get("stage_label") for v in client.search_model_versions(f"name='{name}'") if v.tags}
    out = []
    for ds, Lw in (("ShanghaiTech", 24), ("UBnormal", 16)):
        label = f"{ds}-L{Lw}-seed0"
        if label in have:
            continue
        d = model_dir("P_ABC", ds, 0)
        with mlflow.start_run(run_name=f"register-{name}-{label}") as run:
            mlflow.set_tags({"model_name": name, "stage_label": label})
            mlflow.log_params({"model_name": name, "dataset": ds, "seg_len": Lw, "arm": "P_ABC", "params": params_of(d)})
            for f in (d / "final.pt", d / "config.json"):
                mlflow.log_artifact(str(f), artifact_path="model")
            with tempfile.TemporaryDirectory() as tmp:
                card = Path(tmp) / "model_card.json"
                card.write_text(json.dumps({"name": name, "version": label, "verdicts": summary["verdicts"], "pi5": [{k: r.get(k) for k in ("scorer", "variant", "delivered_fps", "score_ms", "delivered_auc_ontime", "power_w", "j_per_processed_frame")} for r in summary["pi5"]]}, indent=2, default=str))
                mlflow.log_artifact(str(card), artifact_path="model")
            mv = client.create_model_version(name=name, source=f"runs:/{run.info.run_id}/model", run_id=run.info.run_id, description=label, tags={"stage_label": label, "dataset": ds, "seg_len": str(Lw)})
            out.append((label, mv.version))
    print("registered", name, out, flush=True)


def params_of(d: Path) -> int:
    return json.loads((d / "config.json").read_text())["params"]


def traced_clip_demo() -> None:
    """One hierarchical trace of the P_ABC streaming path on a real ShanghaiTech test clip (poses -> context nodes -> flow -> normality)."""
    import mlflow
    from mlflow.entities import SpanType

    from edgelab.analysis import q2_p1_eval as ev
    from edgelab.models import lorad_gaps as lg

    flow, cfg = lg.load_gap_flow(model_dir("P_ABC", "ShanghaiTech", 0) / "final.pt")
    sc = lg.OnlineGapScorer(flow, cfg["seg_len"], ctx=True, smooth=True, name="P_ABC")
    ctx = ev.load_ctx()["ShanghaiTech"]
    tracks = pose_tracks("ShanghaiTech", "released", ctx, 2)
    clip = max(range(len(ctx.names)), key=lambda i: sum(len(f) for f, _ in tracks[i].values()))
    frames: dict[int, dict] = {}
    for tid, (fr, kp) in tracks[clip].items():
        for f, k in zip(fr, kp):
            frames.setdefault(int(f), {})[tid] = np.asarray(k, np.float32)
    with mlflow.start_span(name="e08_demo_clip", span_type=SpanType.CHAIN) as root:
        root.set_inputs({"clip": ctx.names[clip], "frames": len(frames)})
        for f in sorted(frames)[:48]:
            with mlflow.start_span(name="frame", span_type=SpanType.CHAIN) as s:
                s.set_inputs({"frame": f, "tracks": len(frames[f])})
                with mlflow.start_span(name="P_ABC_step", span_type=SpanType.LLM) as ss:
                    norms = sc.step(f, frames[f])
                    ss.set_outputs({"windows": len(norms), "min_normality": float(min(norms)) if norms else None})
        root.set_outputs({"scorer": "P_ABC"})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["train", "run", "stats", "final", "pi-prep", "pi-report", "mlflow"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--procs", type=int, default=5)
    ap.add_argument("--datasets", default=",".join(DATASETS))
    ap.add_argument("--arms", default=None)
    ap.add_argument("--sources", default=",".join(POSE_SOURCES))
    ap.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    ap.add_argument("--max-batches", type=int, default=None)
    a = ap.parse_args()
    ds = a.datasets.split(",")
    if a.stage == "train":
        cmd_train(a.device, a.threads, ds, (a.arms or ",".join(TRAIN_ARMS)).split(","), [int(s) for s in a.seeds.split(",")], a.procs, a.max_batches)
    elif a.stage == "run":
        cmd_run(a.procs, ds, (a.arms or ",".join(EVAL_ARMS)).split(","), a.sources.split(","))
    else:
        {"stats": cmd_stats, "final": cmd_final, "pi-prep": cmd_pi_prep, "pi-report": cmd_pi_report, "mlflow": cmd_mlflow}[a.stage]()


if __name__ == "__main__":
    main()
