"""Q2-P1-E07 driver: YOLO26-pose -> TrackTrack -> {STG-NF, LoRaD, TrajVAD-T, TrajVAD-P}.

Stages (each resumable, each a sub-command):
  train   TrajVAD-T / TrajVAD-P on normal training tracks (mps; ~0.07 s / batch vs 2.6 s on cpu)
  sweep   TrackTrack dev-grid on test_dev
  run     factorial matrix + bootstrap
  cache   Cache-C1+Guard on the selected pipelines
  report  results.csv / summary.json / report.md
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

WORK = Path.home() / "thesis-data" / "q2-p1-e07"
MODELS = WORK / "models"
STG = Path(__file__).resolve().parents[3] / "third_party" / "STG-NF" / "data"
DATASETS = ("ShanghaiTech", "UBnormal")
SEEDS = (0, 1, 2)
SCORE_DEVICE = "mps"  # TrajVAD scoring of the offline matrix (train-window sentinels are ~1M windows); the Pi runs the CPU path


def traj_dir(kind: str, ds: str, seed: int) -> Path:
    return MODELS / f"TrajVAD-{kind}_{ds}_s{seed}"


def cmd_train(device: str = "mps") -> None:
    from edgelab.models import trajvad as tv

    WORK.mkdir(parents=True, exist_ok=True)
    log_path = WORK / "train.log"

    def log(msg: str) -> None:
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        print(line, flush=True)
        with open(log_path, "a") as fh:
            fh.write(line + "\n")

    for ds in DATASETS:
        seg = tv.SEG_LEN[ds]
        ts = tv.build_trainset(STG / ds / "pose" / "train", seg, cache=WORK / f"trajtrain_{ds}_L{seg}.npz")
        log(f"{ds}: {len(ts)} train windows, {len(ts.feats)} poses, L={seg}")
        for seed in SEEDS:
            for kind in ("T", "P"):
                out = traj_dir(kind, ds, seed)
                if (out / "DONE").exists():
                    continue
                tv.train_trajvad(kind, ds, seed, out, ts, device=device, log=log)


# =====================================================================================================================
# Frontends and trackers
# =====================================================================================================================
THESIS = Path.home() / "thesis-data"
FRONTENDS = {
    "ShanghaiTech": ("yolo26_e2e", "yolo26_o2m", "stock", "ML0_ncnn"),
    "UBnormal": ("yolo26_e2e", "stock", "ML0_mps"),
}
FRONTEND_LABEL = {"yolo26_e2e": "YOLO26n-pose (e2e)", "yolo26_o2m": "YOLO26n-pose (o2m)", "stock": "stock YOLOv8n-pose @640",
                  "ML0_ncnn": "ML0 YOLOv8n-pose @640 (NCNN)", "ML0_mps": "ML0 YOLOv8n-pose @640 (MPS proxy)"}


def frontend_cache(ds: str, name: str) -> Path:
    if name.startswith("yolo26"):
        root = THESIS / ("exp017-sht" if ds == "ShanghaiTech" else "exp017-ub") / "cache"
        return root / ("L0_E0-o2m" if name.endswith("o2m") else "L0_E0")
    from edgelab.analysis import q2_p1_e01 as e01

    return {a.key: a for a in e01.build_arms()}[f"{ds}/{name}"].cache_dir


def tracks_for(ds: str, frontend: str, spec, clips: list[str], procs: int = 3) -> list[dict]:
    from edgelab.analysis import q2_p1_eval as ev

    return ev.replay_tracks(frontend_cache(ds, frontend), clips, spec, procs=procs, work=WORK, tag=f"{ds}_{frontend}")


TT_GRID = [(c, low, m) for c in (0.20, 0.25, 0.35) for low in (0.05, 0.10) for m in (0.70, 0.80)]


# =====================================================================================================================
# TrackTrack dev sweep (test_dev only)
# =====================================================================================================================
def cmd_sweep(procs: int) -> None:
    import csv
    import json

    from edgelab.analysis import q2_p1_eval as ev

    ds = "ShanghaiTech"
    ctx = ev.load_ctx()[ds]
    dev_names = [ctx.names[i] for i in ctx.dev_idx]
    scorer = ev.FrozenStgnfScorer(ds)
    sent = ev.train_sentinels(scorer, ds, WORK / f"sentinels_stgnf_{ds}.json")[ev.QUANTILE[ds]]
    seg = scorer.seg_len
    specs = [("tracktrack", ev.TrackerSpec("tracktrack", c, low, 30, m)) for c, low, m in TT_GRID]
    specs += [("tracktrack_default", ev.TRACKTRACK_DEFAULT), ("botsort_default", ev.BOTSORT_DEFAULT), ("botsort_locked", ev.BOTSORT_LOCKED)]
    rows, cache = [], {}
    for fe in ("yolo26_e2e", "yolo26_o2m", "ML0_ncnn"):
        for kind, spec in specs:
            t0 = time.perf_counter()
            tr = tracks_for(ds, fe, spec, dev_names, procs)
            full = [{} for _ in ctx.names]
            for i, t in zip(ctx.dev_idx, tr):
                full[i] = t
            table = ev.build_table(full, scorer, cache)
            res = ev.evaluate_table(table, full, ctx, seg, sent)
            a = res["agg"]["dev"]
            row = {"frontend": fe, "kind": kind, "tracker": spec.label, "c": spec.c, "low": spec.low, "match": spec.match,
                   "S4_dev": res["auc"]["S4"]["dev"], "S0_dev": res["auc"]["S0"]["dev"], "n_tracks": a["n_tracks"], "n_poses": a["n_poses"],
                   "n_windows": a["n_windows"], "stranded_short": a["n_stranded_short"], "stranded_poses_share": a["stranded_poses_share"],
                   "scorable_pose_share": a["scorable_pose_share"], "coverage_all": a["coverage_all"],
                   "gaps_per_1000_poses": a["gaps_per_1000_poses"], "tracks_per_frame": a["tracks_per_frame"]}
            rows.append(row)
            print(f"{ev.now()} {fe:11s} {spec.label:34s} S4 {row['S4_dev']:.2f} S0 {row['S0_dev']:.2f} tracks {row['n_tracks']:5d} "
                  f"stranded {row['stranded_short']:5d} cov {row['coverage_all']:.3f} ({time.perf_counter() - t0:.0f}s)", flush=True)
    (WORK / "sweep").mkdir(exist_ok=True)
    with open(WORK / "sweep" / "sweep.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    # T1: highest mean dev S4 over the 3 frontends; ties (<0.3 pt) -> fewer stranded, then more coverage
    cells: dict[str, list] = {}
    for r in rows:
        if r["kind"] == "tracktrack":
            cells.setdefault(r["tracker"], []).append(r)
    summ = [{"tracker": k, "mean_S4_dev": sum(r["S4_dev"] for r in v) / len(v), "mean_stranded": sum(r["stranded_short"] for r in v) / len(v),
             "mean_cov": sum(r["coverage_all"] for r in v) / len(v), "c": v[0]["c"], "low": v[0]["low"], "match": v[0]["match"]} for k, v in cells.items()]
    best_auc = max(s["mean_S4_dev"] for s in summ)
    pool = [s for s in summ if s["mean_S4_dev"] >= best_auc - 0.3]
    pick = sorted(pool, key=lambda s: (s["mean_stranded"], -s["mean_cov"]))[0]
    (WORK / "sweep" / "selected.json").write_text(json.dumps({"rule": "T1", "selected": pick, "all": summ}, indent=1))
    print("T1 selected:", pick, flush=True)


# =====================================================================================================================
# Factorial matrix
# =====================================================================================================================
E06_MODELS = THESIS / "q2-p1-e06" / "models"
RUN = WORK / "run"
SCORERS = ("stgnf", "lorad", "trajvad_t", "trajvad_p")
SCORER_LABEL = {"stgnf": "STG-NF (frozen)", "lorad": "LoRaD (Causal STG-NF)", "trajvad_t": "TrajVAD-T", "trajvad_p": "TrajVAD-P"}


def tracker_specs():
    """The two trackers of the matrix: TrackTrack with the T1-selected dev cell, BoT-SORT E01 locked 2-stage."""
    import json

    from edgelab.analysis import q2_p1_eval as ev

    sel = json.loads((WORK / "sweep" / "selected.json").read_text())["selected"]
    return {"tracktrack": ev.TrackerSpec("tracktrack", sel["c"], sel["low"], 30, sel["match"]), "botsort": ev.BOTSORT_LOCKED}


def lorad_choice() -> dict:
    import json

    return json.loads((Path(__file__).resolve().parents[3] / "Q2-P1-E06_lorad_causal_flow" / "summary.json").read_text())["selected_for_e07"]


def scorers_for(name: str, ds: str) -> list:
    """[(scorer, id)] one per seed (frozen STG-NF: one)."""
    from edgelab.analysis import q2_p1_eval as ev
    from edgelab.models import trajvad as tv
    from edgelab.models.lorad import FlowScorer

    if name == "stgnf":
        return [(ev.FrozenStgnfScorer(ds), f"stgnf_{ds}")]
    if name in ("trajvad_t", "trajvad_p"):
        kind = name[-1].upper()
        out = []
        for seed in SEEDS:
            d = traj_dir(kind, ds, seed)
            if not (d / "DONE").exists():
                raise FileNotFoundError(f"{d} not trained")
            out.append((tv.TrajVadScorer.from_checkpoint(d / "final.pt", device=SCORE_DEVICE), f"TrajVAD-{kind}_{ds}_s{seed}"))
        return out
    if name == "lorad":
        ch = lorad_choice()
        out = []
        for seed in SEEDS:
            d = E06_MODELS / f"{ch['arm']}_{ds}_s{seed}"
            out.append((FlowScorer.from_checkpoint(d / "final.pt", mode=ch["mode"]), f"{ch['arm']}_{ch['mode']}_{ds}_s{seed}"))
        return out
    raise KeyError(name)


def run_cell(ds: str, fe: str, tr_name: str, spec, scorer_name: str, ctx, tracks: list[dict]) -> Path:
    import pickle

    from edgelab.analysis import q2_p1_eval as ev

    RUN.mkdir(parents=True, exist_ok=True)
    out = RUN / f"{ds}__{fe}__{tr_name}__{scorer_name}.pkl"
    if out.exists():
        return out
    per_seed = []
    for scorer, sid in scorers_for(scorer_name, ds):
        sent = ev.train_sentinels(scorer, ds, WORK / "sentinels" / f"{sid}.json")[ev.QUANTILE[ds]]
        table = ev.build_table(tracks, scorer)
        res = ev.evaluate_table(table, tracks, ctx, scorer.seg_len, sent)
        res.pop("rows")
        res["sentinel"], res["scorer_id"] = sent, sid
        per_seed.append(res)
    out.write_bytes(pickle.dumps({"dataset": ds, "frontend": fe, "tracker": tr_name, "tracker_label": spec.label, "scorer": scorer_name,
                                  "seeds": per_seed}))
    return out


def cmd_run(procs: int, scorers: list[str], datasets: list[str]) -> None:
    from edgelab.analysis import q2_p1_eval as ev

    ctxs = ev.load_ctx()
    specs = tracker_specs()
    for ds in datasets:
        ctx = ctxs[ds]
        for fe in FRONTENDS[ds]:
            for tr_name, spec in specs.items():
                tracks = None
                for sc in scorers:
                    if (RUN / f"{ds}__{fe}__{tr_name}__{sc}.pkl").exists():
                        continue
                    t0 = time.perf_counter()
                    if tracks is None:
                        tracks = tracks_for(ds, fe, spec, ctx.names, procs)
                    p = run_cell(ds, fe, tr_name, spec, sc, ctx, tracks)
                    print(f"{ev.now()} {ds} {fe} {tr_name} {sc}: done {p.name} ({time.perf_counter() - t0:.0f}s)", flush=True)


# =====================================================================================================================
# Bootstrap, results.csv, contrasts
# =====================================================================================================================
BOOT_SEED = 20260514
BASELINE = {"ShanghaiTech": ("ML0_ncnn", "botsort", "stgnf"), "UBnormal": ("ML0_mps", "botsort", "stgnf")}  # locked E01 pipeline


def verdict(ci: list[float]) -> str:
    lo, hi = ci
    if lo > 0:
        return "better"
    if hi < 0:
        return "worse"
    if lo >= -1.0 and hi <= 1.0:
        return "equal"
    return "inconclusive"


def load_cells() -> dict[tuple, dict]:
    import pickle

    cells = {}
    for p in sorted(RUN.glob("*.pkl")):
        d = pickle.loads(p.read_bytes())
        cells[(d["dataset"], d["frontend"], d["tracker"], d["scorer"])] = d
    return cells


def cmd_report_stats(n_resamples: int = 2000) -> dict:
    """Pooled / dev / confirmation AUC (seed mean) with bootstrap CI for every cell x protocol + the paired contrasts."""
    import csv
    import json

    import numpy as np

    from edgelab.analysis import q2_p1_eval as ev

    cells, ctxs = load_cells(), ev.load_ctx()
    boots = {ds: ev.AucBoot(ctxs[ds], n_resamples, BOOT_SEED) for ds in ctxs}
    stats: dict[tuple, object] = {}
    rows = []
    for (ds, fe, tr, sc), d in cells.items():
        ab = boots[ds]
        for proto in ("S0", "S4"):
            st = ab.stats([[s["series"][proto][i] for i in range(len(ctxs[ds].names))] for s in d["seeds"]])
            stats[(ds, fe, tr, sc, proto)] = st
            for subset in ("all", "dev", "confirmation"):
                lo, hi = ab.ci(st, subset)
                seed_auc = [s["auc"][proto][subset] for s in d["seeds"]]
                a = [s["agg"][subset] for s in d["seeds"]]
                rows.append({"dataset": ds, "frontend": fe, "tracker": tr, "tracker_label": d["tracker_label"], "scorer": sc, "protocol": proto,
                             "subset": subset, "auc": ab.point(st, subset), "ci_lo": lo, "ci_hi": hi, "n_seeds": len(seed_auc),
                             "auc_seed_sd": float(np.std(seed_auc)) if len(seed_auc) > 1 else 0.0,
                             "coverage_all": float(np.mean([x["coverage_all"] for x in a])), "n_tracks": a[0]["n_tracks"], "n_poses": a[0]["n_poses"],
                             "n_windows": float(np.mean([x["n_windows"] for x in a])), "stranded_short": a[0]["n_stranded_short"],
                             "scorable_pose_share": float(np.mean([x["scorable_pose_share"] for x in a])),
                             "gaps_per_1000_poses": a[0]["gaps_per_1000_poses"]})
    (WORK / "report").mkdir(exist_ok=True)
    with open(WORK / "report" / "results.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    contrasts = []

    def add(kind, ds, a_key, b_key, text):
        for proto in ("S0", "S4"):
            ka, kb = a_key + (proto,), b_key + (proto,)
            if ka not in stats or kb not in stats:
                continue
            for subset in ("all", "dev", "confirmation"):
                dd = boots[ds].diff(stats[ka], stats[kb], subset)
                contrasts.append({"kind": kind, "dataset": ds, "a": "|".join(a_key), "b": "|".join(b_key), "text": text, "protocol": proto,
                                  "subset": subset, "delta": dd["delta"], "ci_lo": dd["ci95"][0], "ci_hi": dd["ci95"][1],
                                  "verdict": verdict(dd["ci95"])})

    for ds in ctxs:
        for fe in FRONTENDS[ds]:
            for sc in SCORERS:
                add("T2", ds, (ds, fe, "tracktrack", sc), (ds, fe, "botsort", sc), f"TrackTrack - BoT-SORT | {fe} | {sc}")
            for tr in ("tracktrack", "botsort"):
                add("V1", ds, (ds, fe, tr, "lorad"), (ds, fe, tr, "stgnf"), f"LoRaD - STG-NF | {fe} | {tr}")
                add("V2", ds, (ds, fe, tr, "trajvad_p"), (ds, fe, tr, "lorad"), f"TrajVAD-P - LoRaD | {fe} | {tr}")
                add("V3", ds, (ds, fe, tr, "trajvad_p"), (ds, fe, tr, "trajvad_t"), f"TrajVAD-P - TrajVAD-T | {fe} | {tr}")
                add("V0", ds, (ds, fe, tr, "trajvad_p"), (ds, fe, tr, "stgnf"), f"TrajVAD-P - STG-NF | {fe} | {tr}")
                add("V0t", ds, (ds, fe, tr, "trajvad_t"), (ds, fe, tr, "stgnf"), f"TrajVAD-T - STG-NF | {fe} | {tr}")
            bfe, btr, bsc = BASELINE[ds]
            for tr in ("tracktrack", "botsort"):
                for sc in SCORERS:
                    add("B", ds, (ds, fe, tr, sc), (ds, bfe, btr, bsc), f"{fe}|{tr}|{sc} - locked E01 ({bfe}|{btr}|{bsc})")
    with open(WORK / "report" / "contrasts.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(contrasts[0]))
        w.writeheader()
        w.writerows(contrasts)
    return {"rows": rows, "contrasts": contrasts}


# =====================================================================================================================
# Pi 5 delivered real-time AUC
# =====================================================================================================================
PI_OUT = THESIS / "q2-p1-e07-pi5"
PI_ARM_FRONTEND = {"yolo26": "yolo26_o2m", "ml0": "ML0_ncnn"}  # the offline matrix row of the same detector


def cmd_pi_report() -> dict:
    import csv
    import json

    import numpy as np

    from edgelab.analysis import q2_p1_e00 as e00
    from edgelab.analysis import q2_p1_eval as ev

    ds = "ShanghaiTech"
    ctx = ev.load_ctx()[ds]
    cells = load_cells()
    sentinels, ch = {}, None
    present = {json.loads(p.read_text())["scorer"] for p in PI_OUT.glob("*_summary.json")}
    for sc in ("lorad", "trajvad_p"):
        if sc not in present:
            continue
        if sc == "lorad":
            ch = lorad_choice()
        scorer, sid = scorers_for(sc, ds)[0]  # seed 0 = the checkpoint the Pi runs
        sentinels[sc] = ev.train_sentinels(scorer, ds, WORK / "sentinels" / f"{sid}.json")[ev.QUANTILE[ds]]
    rows = []
    for sf in sorted(PI_OUT.glob("*_summary.json")):
        s = json.loads(sf.read_text())
        z = np.load(PI_OUT / f"{s['arm']}_raw.npz")
        clips = sorted({k.split("__", 1)[1] for k in z.files})
        idx = [ctx.names.index(c) for c in clips]
        labels = [ctx.labels[i][: len(z[f"all__{c}"])] for i, c in zip(idx, clips)]
        row = {"arm": s["arm"], "detector": s["detector"], "scorer": s["scorer"], "cache_guard": s["use_c1"], "frames": s["frames"],
               "frames_processed": s["frames_processed"], "delivered_fps": s["delivered_fps"], "service_ms_mean": s["service_ms_mean"],
               "service_ms_p95": s["service_ms_p95"], "decode_ms": s["decode_ms_mean"], "pose_ms": s["pose_ms_mean"],
               "track_ms": s["track_ms_mean"], "score_ms": s["score_ms_mean"], "c1_ms": s["c1_check_ms_mean"],
               "queue_wait_p99_ms": s["queue_wait_ms_p99"], "queue_wait_max_ms": s["queue_wait_ms_max"], "c1_hit_pct": s["c1_hit_rate_pct"],
               "guard_shed_pct": s["guard_shed_pct"], "sustainable_24fps": s["Sustainable_24fps"], "power_w": s["telemetry"]["power_total_w_mean"],
               "power_w_max": s["telemetry"]["power_total_w_max"], "temp_c_mean": s["telemetry"]["temp_c_mean"], "temp_c_max": s["telemetry"]["temp_c_max"],
               "rss_mb_peak": s["telemetry"]["rss_mb_peak"], "sys_cpu_pct_mean": s["telemetry"]["sys_cpu_pct_mean"],
               "j_per_processed_frame": s["energy_j_per_processed_frame"], "run_id": s.get("mlflow_run_id", "")}
        sentinel = sentinels[s["scorer"]]
        for mode in ("ontime", "all"):
            raw = [z[f"{mode}__{c}"] for c in clips]
            series, _ = e00.deliver(raw, sentinel, 0)
            row[f"delivered_auc_{mode}"] = 100 * float(e00.pooled_auc(labels, series, list(range(len(clips)))))
            row[f"coverage_{mode}"] = float(np.mean(np.concatenate([np.isfinite(r) for r in raw])))
        ref = cells.get((ds, PI_ARM_FRONTEND[s["detector"]], "tracktrack", s["scorer"]))
        if ref is not None:
            ser = ref["seeds"][0]["series"]["S4"]
            row["offline_ideal_auc_same_clips"] = 100 * float(e00.pooled_auc(labels, [ser[i][: len(labels[j])] for j, i in enumerate(idx)],
                                                                         list(range(len(clips)))))
        rows.append(row)
    if rows:
        with open(WORK / "report" / "pi5_results.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    return {"pi": rows, "lorad": ch}


# =====================================================================================================================
# Final tables / summary.json
# =====================================================================================================================
EXP_DIR = Path(__file__).resolve().parents[3] / "Q2-P1-E07_yolo26_tracktrack_trajvadp"


def cmd_final() -> None:
    import csv
    import json
    import shutil
    from collections import Counter

    import numpy as np

    res = cmd_report_stats()
    pi = cmd_pi_report()
    rows, contrasts = res["rows"], res["contrasts"]
    for name in ("results.csv", "contrasts.csv", "pi5_results.csv"):
        if (WORK / "report" / name).exists():
            shutil.copy(WORK / "report" / name, EXP_DIR / name)
    shutil.copy(WORK / "sweep" / "sweep.csv", EXP_DIR / "tracktrack_dev_sweep.csv")
    sel = json.loads((WORK / "sweep" / "selected.json").read_text())
    get = {(r["dataset"], r["frontend"], r["tracker"], r["scorer"], r["protocol"], r["subset"]): r for r in rows}
    tables = []
    for subset in ("confirmation", "all"):
        for ds in ("ShanghaiTech", "UBnormal"):
            tables.append(f"\n#### {ds} — S4 (strict causal) pooled AUC, {subset} clips, seed mean [95% CI]  (S0 offline in parentheses)\n")
            tables.append("| frontend | tracker | " + " | ".join(SCORER_LABEL[s] for s in SCORERS) + " |")
            tables.append("|---|---|" + "---|" * len(SCORERS))
            for fe in FRONTENDS[ds]:
                for tr in ("tracktrack", "botsort"):
                    cells = []
                    for sc in SCORERS:
                        a, b = get.get((ds, fe, tr, sc, "S4", subset)), get.get((ds, fe, tr, sc, "S0", subset))
                        cells.append("–" if a is None else f"{a['auc']:.2f} [{a['ci_lo']:.1f}, {a['ci_hi']:.1f}] ({b['auc']:.1f})")
                    tables.append(f"| {FRONTEND_LABEL[fe]} | {'TrackTrack' if tr == 'tracktrack' else 'BoT-SORT 2-stage'} | " + " | ".join(cells) + " |")
    verdicts = {}
    for kind in ("T2", "V1", "V2", "V3", "V0", "V0t"):
        sub = [c for c in contrasts if c["kind"] == kind and c["protocol"] == "S4" and c["subset"] == "confirmation"]
        verdicts[kind] = dict(Counter(c["verdict"] for c in sub))
        verdicts[kind]["n"] = len(sub)
    # V4: best pipeline = highest mean (over the two datasets) confirmation S4 AUC among (frontend-name-matched) pipelines
    best = {}
    for ds in ("ShanghaiTech", "UBnormal"):
        for (d, fe, tr, sc, proto, subset), r in get.items():
            if d == ds and proto == "S4" and subset == "confirmation":
                best.setdefault((fe if fe.startswith("yolo26") else ("ML0" if fe.startswith("ML0") else fe), tr, sc), {})[ds] = r["auc"]
    ranked = sorted(((np.mean(list(v.values())), k, v) for k, v in best.items() if len(v) == 2), reverse=True)
    summary = {"experiment": "Q2-P1-E07", "tracktrack_selection": sel, "verdict_counts_S4_confirmation": verdicts,
               "best_pipelines_mean_of_datasets_S4_confirmation": [{"pipeline": "|".join(k), "mean_auc": float(m), "per_dataset": v} for m, k, v in ranked[:8]],
               "contrasts": contrasts, "pi5": pi["pi"], "lorad_choice": pi.get("lorad"), "mean_rows": [r for r in rows if r["subset"] == "confirmation"],
               "train": {p.parent.name: json.loads(p.read_text()) for p in sorted(MODELS.glob("*/config.json"))}}
    (EXP_DIR / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    (WORK / "report" / "tables.md").write_text("\n".join(tables) + "\n")
    notes = EXP_DIR / "report_notes.md"
    if notes.exists():
        (EXP_DIR / "report.md").write_text(notes.read_text() + "\n".join(tables) + "\n")
    print(json.dumps({"verdicts": verdicts, "best": summary["best_pipelines_mean_of_datasets_S4_confirmation"][:5]}, indent=1, default=float))


# =====================================================================================================================
# MLflow: runs, artifacts, traces, model registry
# =====================================================================================================================
def traced_clip_demo(ds: str = "ShanghaiTech", clip: str | None = None) -> dict:
    """One hierarchical MLflow trace of a clip through YOLO26(o2m cache) -> TrackTrack -> LoRaD / TrajVAD-P (Mac, offline replay of the caches)."""
    import mlflow
    import numpy as np
    from mlflow.entities import SpanType

    from edgelab.analysis import q2_p1_eval as ev
    from edgelab.pose.detcache import load_detections
    from edgelab.pose.replay import Detections

    ctx = ev.load_ctx()[ds]
    clip = clip or ctx.names[ctx.conf_idx[0]]
    spec = tracker_specs()["tracktrack"]
    lorad = scorers_for("lorad", ds)[0][0]
    trajp = scorers_for("trajvad_p", ds)[0][0]

    @mlflow.trace(name="e07_clip_pipeline", span_type=SpanType.CHAIN)
    def run() -> dict:
        mlflow.update_current_trace(metadata={"mlflow.trace.session": "e07-offline-demo"})
        with mlflow.start_span(name="1_load_detections", span_type=SpanType.TOOL) as s:
            frames, data = load_detections(frontend_cache(ds, "yolo26_o2m") / f"{clip}.npz")
            s.set_inputs({"clip": clip, "cache": "yolo26_o2m"})
            s.set_outputs({"frames": len(frames)})
        with mlflow.start_span(name="2_tracktrack", span_type=SpanType.TOOL) as s:
            t0 = time.perf_counter()
            tracks = ev.replay_clip(frames, data["frame_ids"], spec)
            s.set_inputs({"tracker": spec.label})
            s.set_outputs({"n_tracks": len(tracks), "n_poses": int(sum(len(v[0]) for v in tracks.values())), "ms": round((time.perf_counter() - t0) * 1e3, 2)})
        out = {}
        for name, scorer in (("3_lorad_causal_flow", lorad), ("4_trajvad_p", trajp)):
            with mlflow.start_span(name=name, span_type=SpanType.LLM) as s:
                t0 = time.perf_counter()
                norms = scorer.score_tracks(list(tracks.values()))
                allv = np.concatenate([n for n in norms if len(n)]) if any(len(n) for n in norms) else np.zeros(0)
                s.set_inputs({"scorer": name, "n_tracks": len(tracks)})
                s.set_outputs({"n_windows": int(len(allv)), "min_normality": float(allv.min()) if len(allv) else None,
                               "ms": round((time.perf_counter() - t0) * 1e3, 2)})
                out[name] = int(len(allv))
        return out

    return run()


def cmd_mlflow() -> None:
    import csv
    import json
    import os
    import tempfile

    os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
    import mlflow
    from mlflow.tracking import MlflowClient

    from edgelab.analysis import q2_p1_eval as ev

    uri = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment("Q2-P1-E07")
    summary = json.loads((EXP_DIR / "summary.json").read_text())
    sweep = list(csv.DictReader(open(EXP_DIR / "tracktrack_dev_sweep.csv")))
    with mlflow.start_run(run_name="tracktrack_dev_sweep") as run:
        mlflow.set_tags({"stage": "sweep", "split": "test_dev (35 SHT clips)"})
        mlflow.log_params({"grid_track_high": "0.20,0.25,0.35", "grid_track_low": "0.05,0.10", "grid_match": "0.70,0.80", "selected": summary["tracktrack_selection"]["selected"]["tracker"]})
        for i, r in enumerate(sweep):
            mlflow.log_metrics({f"S4_dev__{r['frontend']}": float(r["S4_dev"]), f"stranded__{r['frontend']}": float(r["stranded_short"]),
                                f"coverage__{r['frontend']}": float(r["coverage_all"])}, step=i)
        mlflow.log_artifact(str(EXP_DIR / "tracktrack_dev_sweep.csv"))
    with mlflow.start_run(run_name="trajvad_training") as parent:
        for d in sorted(MODELS.glob("*/config.json")):
            cfg = json.loads(d.read_text())
            with mlflow.start_run(run_name=d.parent.name, nested=True):
                mlflow.log_params({k: cfg[k] for k in ("kind", "dataset", "seg_len", "seed", "epochs", "batch_size", "lr", "n_windows", "params", "device")})
                for e, v in enumerate(cfg["epoch_losses"]):
                    mlflow.log_metric("train_nll_bits_per_dim", v, step=e + 1)
                mlflow.log_metric("train_minutes", cfg["train_minutes"])
                mlflow.log_artifact(str(d))
    with mlflow.start_run(run_name="factorial_matrix") as run:
        mlflow.set_tags({"stage": "matrix"})
        for r in summary["mean_rows"]:
            if r["protocol"] == "S4" and r["subset"] == "confirmation":
                mlflow.log_metric(f"S4_conf__{r['dataset'][:3]}__{r['frontend']}__{r['tracker']}__{r['scorer']}", r["auc"])
        for kind, c in summary["verdict_counts_S4_confirmation"].items():
            for k, v in c.items():
                mlflow.log_metric(f"verdict_{kind}_{k}", v)
        for name in ("results.csv", "contrasts.csv", "summary.json", "pi5_results.csv", "plan.md", "config.yaml", "report.md"):
            if (EXP_DIR / name).exists():
                mlflow.log_artifact(str(EXP_DIR / name))
    traced_clip_demo()
    mlflow.flush_trace_async_logging()

    # ---- model registry -----------------------------------------------------------------------------------------------
    client = MlflowClient(tracking_uri=uri)
    ch = lorad_choice()
    sht_auc = {(r["frontend"], r["tracker"], r["scorer"]): r["auc"] for r in summary["mean_rows"] if r["dataset"] == "ShanghaiTech" and r["protocol"] == "S4" and r["subset"] == "confirmation"}
    specs = [
        {"name": "edgelab-yolo26n-pose", "description": "Ultralytics YOLO26n-pose (COCO-pose, NMS-free e2e + one-to-many heads) and its NCNN FP16-ready 384x640 export used on the Pi 5 (E07).",
         "tags": {"task": "pose_estimation", "architecture": "YOLO26n-pose", "paper_reference": "Ultralytics 8.4.201"},
         "versions": [{"label": "PyTorch-weights", "files": [Path(__file__).resolve().parents[3] / "artifacts/EXP-017/yolo26n-pose.pt"], "tags": {"runtime": "pytorch"}},
                      {"label": "NCNN-FP16-384x640", "files": sorted((THESIS / "vm-exchange/models/yolo26n-pose-640-384x640-ncnn").glob("model.ncnn.*")) + [THESIS / "vm-exchange/models/yolo26n-pose-640-384x640-ncnn/metadata.yaml"],
                       "tags": {"runtime": "ncnn_fp16", "imgsz": "384x640", "head": "one-to-many (raw 56 x 5040, NMS in post)"}}]},
        {"name": "edgelab-lorad-causal-flow", "description": f"LoRaD Causal STG-NF (Zhang et al., JRTIP 2026): causal temporal-conv coupling flow, {ch['arm']} ({ch['mode']} mode), 616 params, trained here (E06).",
         "tags": {"task": "skeleton_anomaly_scoring", "architecture": "Causal STG-NF", "arm": ch["arm"], "mode": ch["mode"], "paper_reference": "doi:10.1007/s11554-026-01937-6"},
         "versions": [{"label": f"{ds}-L{L}-seed0", "files": [E06_MODELS / f"{ch['arm']}_{ds}_s0" / "final.pt", E06_MODELS / f"{ch['arm']}_{ds}_s0" / "config.json"],
                       "tags": {"dataset": ds, "seg_len": str(L)}} for ds, L in (("ShanghaiTech", 24), ("UBnormal", 16))]},
        {"name": "edgelab-trajvad-p", "description": "TrajVAD-P (Song & Lee, ECCV 2026) causal edge port: 27 kinematic box-trajectory features + conditional pose flow, 757,338 params, trained here (E07).",
         "tags": {"task": "skeleton_anomaly_scoring", "architecture": "TrajVAD-P (causal-conv couplings)", "paper_reference": "ECCV 2026"},
         "versions": [{"label": f"{ds}-L{L}-seed0", "files": [traj_dir("P", ds, 0) / "final.pt", traj_dir("P", ds, 0) / "config.json"], "tags": {"dataset": ds, "seg_len": str(L)}}
                      for ds, L in (("ShanghaiTech", 16), ("UBnormal", 12))]},
    ]
    out = {}
    for spec in specs:
        name = spec["name"]
        try:
            client.create_registered_model(name=name, description=spec["description"], tags={k: str(v) for k, v in spec["tags"].items()})
        except Exception:
            client.update_registered_model(name=name, description=spec["description"])
            for k, v in spec["tags"].items():
                client.set_registered_model_tag(name, k, str(v))
        have = {v.tags.get("stage_label") for v in client.search_model_versions(f"name='{name}'") if v.tags}
        for ver in spec["versions"]:
            if ver["label"] in have:
                continue
            with mlflow.start_run(run_name=f"register-{name}-{ver['label']}") as run:
                mlflow.set_tags({"model_name": name, "stage_label": ver["label"]})
                mlflow.log_params({"model_name": name, "stage_label": ver["label"], **ver["tags"]})
                for f in ver["files"]:
                    mlflow.log_artifact(str(f), artifact_path="model")
                with tempfile.TemporaryDirectory() as tmp:
                    card = Path(tmp) / "model_card.json"
                    card.write_text(json.dumps({"spec": {k: v for k, v in spec.items() if k != "versions"}, "version": ver["label"], "files": [str(f) for f in ver["files"]],
                                                "sht_S4_confirmation_auc_by_pipeline": {"|".join(k): v for k, v in sht_auc.items()}}, indent=2, default=str))
                    mlflow.log_artifact(str(card), artifact_path="model")
                mv = client.create_model_version(name=name, source=f"runs:/{run.info.run_id}/model", run_id=run.info.run_id, description=ver["label"],
                                                 tags={"stage_label": ver["label"], **{k: str(v) for k, v in ver["tags"].items()}})
                out.setdefault(name, []).append(mv.version)
    print(json.dumps(out, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["train", "sweep", "run", "stats", "pi-report", "final", "mlflow"])
    ap.add_argument("--device", default="mps")
    ap.add_argument("--procs", type=int, default=3)
    ap.add_argument("--scorers", default="stgnf,trajvad_t,trajvad_p")
    ap.add_argument("--datasets", default="ShanghaiTech,UBnormal")
    a = ap.parse_args()
    if a.stage == "train":
        cmd_train(a.device)
    elif a.stage == "sweep":
        cmd_sweep(a.procs)
    elif a.stage == "run":
        cmd_run(a.procs, a.scorers.split(","), a.datasets.split(","))
    elif a.stage == "stats":
        cmd_report_stats()
    elif a.stage == "pi-report":
        cmd_pi_report()
    elif a.stage == "final":
        cmd_final()
    elif a.stage == "mlflow":
        cmd_mlflow()


if __name__ == "__main__":
    main()
