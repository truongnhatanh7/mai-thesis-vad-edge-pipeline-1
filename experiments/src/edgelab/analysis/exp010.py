"""EXP-010 driver: SKIP vs FILL-HOLD vs FILL-NORMAL under CPU deadlines s = 3, 6 on six groups (plan frozen 2026-11-07 17:00).

    uv run --extra pose python -m edgelab.analysis.exp010 run --part SHT      # or UB; resumable
    uv run --extra pose python -m edgelab.analysis.exp010 analyse             # paired bootstraps + verdicts C1-C4

The three policies see identical keep masks (deadline_sim.keep_masks_for_videos, seed 1):
  SKIP         dropped frames are skipped, processed frames re-indexed 0..K-1, scores held to the next processed frame
               (what deadline_replay.run_cell does; here it also gives the paper variant);
  FILL-HOLD    dropped frames stay at real time, P fill G 8, STG-NF, frames with no score get the LAST available score;
  FILL-NORMAL  as FILL-HOLD, but frames with no score get the most normal score (EXP-009).
Rows: ~/thesis-data/exp010/rows.jsonl; per-frame scores: ~/thesis-data/exp010/scores/<condition>/frame_scores_<variant>.npz.
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

from edgelab.analysis.exp003 import Scores, diff
from edgelab.analysis.exp004 import usable_traces
from edgelab.analysis.exp008 import MANIFEST as EXP004_MANIFEST
from edgelab.analysis.exp009 import load_scores_file
from edgelab.metrics.scores import bootstrap_ci_over_videos, frame_level_auc
from edgelab.pipeline.deadline_replay import BLOCK, cache_frame_counts, load_trace
from edgelab.pipeline.deadline_sim import keep_masks_for_videos
from edgelab.pose.delivered import delivered_variant_scores
from edgelab.pose.fill import fill_dir, segment_predicted_shares
from edgelab.pose.replay import FRAME_MAPS_FILE, replay_dir
from edgelab.pose.score_dir import DATASET_DEFAULTS, VARIANTS, coverage
from edgelab.pose.smoothing import SMOOTHERS
from edgelab.pose.stgnf_eval import clip_normality, score_frame_offsets, score_segments
from edgelab.tracking.mlflow_run import tracked_run

DATA = Path.home() / "thesis-data"
OUT = DATA / "exp010"
ROWS = OUT / "rows.jsonl"
R_FILE = DATA / "vm-exchange/results/exp010/r.json"
EXPERIMENT_DIR = Path(__file__).resolve().parents[3] / "EXP-010-filling-confirmation"
TRACE_CELL = "L3-E0-R640-c0.25-O0-B4"
SEED = 1
MLFLOW_EXPERIMENT = "EXP-010"
SLOWDOWNS = (3, 6)
POLICIES = ("SKIP", "FILL-HOLD", "FILL-NORMAL")
FILL_G = 8
N_RESAMPLES = 2000
N_PAIRED = 1000
# group -> (dataset, cache dir, detector "8n" or "8s")
GROUPS = {"SHT-L0-8n": ("ShanghaiTech", DATA / "exp003/cache/L0_E0", "8n"),
          "SHT-L2-8n": ("ShanghaiTech", DATA / "exp003/cache/L2_E0", "8n"),
          "SHT-L3-8n": ("ShanghaiTech", DATA / "exp003/cache/L3_E0", "8n"),
          "UB-L0-8n": ("UBnormal", DATA / "exp006/cache/L0_E0", "8n"),
          "SHT-L0-8s": ("ShanghaiTech", DATA / "exp007-sht/cache/L0_E0", "8s"),
          "UB-L0-8s": ("UBnormal", DATA / "exp007-ub/cache/L0_E0", "8s")}
PARTS = {"SHT": [g for g in GROUPS if g.startswith("SHT")], "UB": [g for g in GROUPS if g.startswith("UB")]}


def configure(seed: int = 1, out_root: Path | None = None, mlflow_experiment: str = "EXP-010", log_dir: Path | None = None):
    """Change the keep-mask seed and where everything is written (EXP-010b reuses this module with seed 2). Defaults = EXP-010."""
    global SEED, OUT, ROWS, MLFLOW_EXPERIMENT, EXPERIMENT_DIR
    SEED, MLFLOW_EXPERIMENT = seed, mlflow_experiment
    if out_root is not None:
        OUT, ROWS = Path(out_root), Path(out_root) / "rows.jsonl"
    if log_dir is not None:
        EXPERIMENT_DIR = Path(log_dir)


# ---------------------------------------------------------------- traces

def scale_detector_time(service_ms: np.ndarray, detector_ms: np.ndarray, r: float) -> np.ndarray:
    """8s service time of each frame: the 8n service time with its detector part (pre + infer + post) multiplied by r."""
    return np.asarray(service_ms, float) + (r - 1.0) * np.asarray(detector_ms, float)


def load_trace_8s(csv_files: list[Path], r: float) -> np.ndarray:
    """Like deadline_replay.load_trace (pooled, each video cut to whole 24-frame blocks), on the r-scaled service times."""
    parts = []
    for path in csv_files:
        frame = pd.read_csv(path, usecols=["video", "service_ms", "det_pre_ms", "det_infer_ms", "det_post_ms"])
        for _, video in frame.groupby("video", sort=False):
            detector = video[["det_pre_ms", "det_infer_ms", "det_post_ms"]].sum(axis=1).to_numpy(float)
            values = scale_detector_time(video["service_ms"].to_numpy(float), detector, r)
            parts.append(values[: len(values) // BLOCK * BLOCK])
    return np.concatenate(parts)


def trace_for(detector: str) -> np.ndarray:
    files = usable_traces(EXP004_MANIFEST)[TRACE_CELL]
    if detector == "8n":
        return load_trace(files)
    return load_trace_8s(files, json.loads(R_FILE.read_text())["r"])


# ---------------------------------------------------------------- scoring rules

def hold_last_finite(values: np.ndarray, fallback: float) -> np.ndarray:
    """Each non-finite entry gets the latest finite value before it; before the first finite value: `fallback`."""
    values = np.asarray(values, float)
    finite = np.isfinite(values)
    latest = np.maximum.accumulate(np.where(finite, np.arange(len(values)), -1))
    return np.where(latest >= 0, values[np.maximum(latest, 0)], fallback)


def fill_variant_scores(segments, gt_dir: Path, offset_name: str, smoother_name: str, dataset: str, blank: str):
    """-> (names, labels, scores, covered). blank = 'hold' (FILL-HOLD) or 'normal' (FILL-NORMAL).

    Per-frame normality of the least-normal person (clip_normality; +inf = nobody scored the frame). 'hold': such a frame
    takes the last earlier frame's normality (before the first scored frame: the dataset's most normal value, as SKIP does);
    'normal': it takes the most normal value (EXP-009). Then the same smoother as every other arm.
    """
    names, labels, normality = clip_normality(segments.normality, segments.metadata, gt_dir,
                                              score_frame_offsets(dataset)[offset_name], dataset)
    covered = [np.isfinite(n) for n in normality]
    flat = np.concatenate(normality)
    most_normal = flat[np.isfinite(flat)].max()
    if blank == "hold":
        filled = [hold_last_finite(np.where(np.isfinite(n), n, np.nan), most_normal) for n in normality]
    elif blank == "normal":
        filled = [np.where(np.isfinite(n), n, most_normal) for n in normality]
    else:
        raise ValueError(blank)
    return names, labels, [SMOOTHERS[smoother_name](-f) for f in filled], covered


# ---------------------------------------------------------------- rows

def read_rows() -> list[dict]:
    return [json.loads(line) for line in ROWS.read_text().splitlines() if line.strip()] if ROWS.exists() else []


def is_done(condition: str) -> bool:
    return {r["variant"] for r in read_rows() if r["condition"] == condition} >= set(VARIANTS)


def log(message: str):
    with open(EXPERIMENT_DIR / "log.md", "a") as f:
        f.write(f"\n### {datetime.now():%Y-%m-%d %H:%M} — driver\n- {message}\n")


def emit(spec: dict, variant: str, names, labels, scores, covered, num_segments: int, extra: dict, dataset: str):
    auc = frame_level_auc(labels, scores)
    low, high = bootstrap_ci_over_videos(labels, scores, n_resamples=N_RESAMPLES)
    cov_a, cov_n = coverage(covered, labels)
    scores_dir = OUT / "scores" / spec["condition"].replace("/", "__")
    scores_dir.mkdir(parents=True, exist_ok=True)
    scores_file = scores_dir / f"frame_scores_{variant}.npz"
    np.savez_compressed(scores_file, names=np.array(names),
                        **{f"score_{n}": s for n, s in zip(names, scores)},
                        **{f"label_{n}": y for n, y in zip(names, labels)},
                        **{f"covered_{n}": c for n, c in zip(names, covered)})
    row = {**spec, **extra, "variant": variant, "auc": auc, "ci": [low, high], "coverage_anomalous": cov_a,
           "coverage_normal": cov_n, "num_segments": num_segments, "scores_file": str(scores_file)}
    checkpoint = DATASET_DEFAULTS[dataset][0]
    with tracked_run(MLFLOW_EXPERIMENT, f"{spec['condition']}-{variant}", {**spec, **extra, "variant": variant},
                     input_files={"checkpoint": checkpoint}):
        mlflow.log_metrics({"auc": auc, "ci_low": low, "ci_high": high, "coverage_anomalous": cov_a,
                            "coverage_normal": cov_n, "num_segments": num_segments})
        mlflow.log_artifact(str(scores_file))
    print(f"{spec['condition']:36s} {variant:5s} AUC={auc * 100:.2f} [{low * 100:.2f},{high * 100:.2f}] "
          f"cov a/n={cov_a:.3f}/{cov_n:.3f} seg={num_segments}", flush=True)
    return row


def run_group_speed(group: str, s: int, threads: int):
    dataset, cache, detector = GROUPS[group]
    checkpoint, gt_dir = DATASET_DEFAULTS[dataset]
    conditions = {p: f"{group}/s{s}/{p}" for p in POLICIES}
    if all(is_done(c) for c in conditions.values()):
        return
    videos = None
    if dataset == "UBnormal":
        from edgelab.pose.ubnormal import load_test_list
        videos = load_test_list()
    counts = cache_frame_counts(cache, videos)
    masks = keep_masks_for_videos(counts, trace_for(detector) * s, 24.0, seed=SEED)
    drop_rate = 1 - sum(int(m.sum()) for m in masks.values()) / sum(counts.values())
    base = OUT / "poses" / f"{group}__s{s}"
    common = {"group": group, "s": s, "detector": detector, "drop_rate": drop_rate, "seed": SEED}

    if not is_done(conditions["SKIP"]):  # skip-and-reindex: processed frames 0..K-1, scores held on the real timeline
        skip_dir = base.with_name(base.name + "__skip")
        replay_dir(cache, 0.25, skip_dir, keep_masks=masks, reindex=True, videos=videos, dataset=dataset)
        segments = score_segments(skip_dir, checkpoint, device="cpu", threads=threads, dataset=dataset)
        frame_maps = dict(np.load(skip_dir / FRAME_MAPS_FILE))
        rows = []
        for variant, (offset_name, smoother_name) in VARIANTS.items():
            names, labels, scores = delivered_variant_scores(segments.normality, segments.metadata, frame_maps, gt_dir,
                                                             offset_name, smoother_name, dataset, videos)
            covered = [np.ones(len(y), bool) for y in labels]  # SKIP delivers a (held) score on every frame it can
            spec = {**common, "condition": conditions["SKIP"], "policy": "SKIP"}
            rows.append(emit(spec, variant, names, labels, scores, covered, segments.num_segments,
                             {"predicted_share_of_scored_windows": 0.0}, dataset))
        append_rows(rows)

    if not (is_done(conditions["FILL-HOLD"]) and is_done(conditions["FILL-NORMAL"])):
        real_dir = base.with_name(base.name + "__realtime")
        replay_dir(cache, 0.25, real_dir, keep_masks=masks, reindex=False, videos=videos, dataset=dataset)
        filled_dir = base.with_name(base.name + "__P8")
        fill_dir(real_dir, filled_dir, "P", FILL_G, n_frames={v: len(m) for v, m in masks.items()})
        rows = []
        try:
            segments = score_segments(filled_dir, checkpoint, device="cpu", threads=threads, dataset=dataset)
        except (ValueError, IndexError) as error:
            # STG-NF crashes when no window survives. Pre-registered (EXP-003, "deviations policy as EXP-003"): constant output,
            # AUC 0.5, coverage 0, row kept. Its scores file is a constant series so the paired bootstrap still works.
            for policy in ("FILL-HOLD", "FILL-NORMAL"):
                for variant in VARIANTS:
                    names, labels, scores, covered = constant_scores(gt_dir, dataset, videos)
                    spec = {**common, "condition": conditions[policy], "policy": policy}
                    rows.append({**emit(spec, variant, names, labels, scores, covered, 0,
                                        {"predicted_share_of_scored_windows": 0.0, "G": FILL_G}, dataset),
                                 "no_segments": True, "error": str(error)[:200], "auc": 0.5, "ci": None})
            log(f"{group} s={s}: no scorable window after P fill -> FILL rows = AUC 0.5 (EXP-003 rule)")
            append_rows(rows)
            return
        shares = segment_predicted_shares(segments.metadata, filled_dir, filled_dir.with_name(filled_dir.name + "_prov"), dataset)
        predicted_share = float(shares.mean()) if len(shares) else 0.0
        for policy, blank in (("FILL-HOLD", "hold"), ("FILL-NORMAL", "normal")):
            for variant, (offset_name, smoother_name) in VARIANTS.items():
                names, labels, scores, covered = fill_variant_scores(segments, gt_dir, offset_name, smoother_name, dataset, blank)
                spec = {**common, "condition": conditions[policy], "policy": policy}
                rows.append(emit(spec, variant, names, labels, scores, covered, segments.num_segments,
                                 {"predicted_share_of_scored_windows": predicted_share, "G": FILL_G}, dataset))
        append_rows(rows)


def constant_scores(gt_dir: Path, dataset: str, clips):
    """Every frame the same score (what a system with no scorable window outputs): AUC 0.5, nothing covered."""
    from edgelab.pose.stgnf_eval import _clips
    names, labels = [], []
    for name, _, _, gt in _clips(dataset, gt_dir, clips):
        names.append(name)
        labels.append(gt)
    return names, labels, [np.zeros(len(y)) for y in labels], [np.zeros(len(y), bool) for y in labels]


def append_rows(rows: list[dict]):
    OUT.mkdir(parents=True, exist_ok=True)
    done = {(r["condition"], r["variant"]) for r in read_rows()}
    with open(ROWS, "a") as f:
        f.write("".join(json.dumps(r) + "\n" for r in rows if (r["condition"], r["variant"]) not in done))


def run_part(part: str, threads: int):
    OUT.mkdir(parents=True, exist_ok=True)
    for s in SLOWDOWNS:
        for group in PARTS[part]:
            run_group_speed(group, s, threads)
    log(f"part {part} finished")


# ---------------------------------------------------------------- analysis (plan §3, exactly)

def paired(group: str, s: int, policy: str, variant: str, by: dict) -> dict:
    skip, other = by[(f"{group}/s{s}/SKIP", variant)], by[(f"{group}/s{s}/{policy}", variant)]
    d = diff(load_scores_file(skip["scores_file"]), load_scores_file(other["scores_file"]), n_resamples=N_PAIRED)
    return {"skip": skip["auc"] * 100, "policy": other["auc"] * 100, "diff": d["diff"] * 100,
            "ci": [d["ci"][0] * 100, d["ci"][1] * 100]}


def analyse():
    rows = read_rows()
    by = {(r["condition"], r["variant"]): r for r in rows}
    table = {}
    for group in GROUPS:
        for s in SLOWDOWNS:
            for policy in ("FILL-HOLD", "FILL-NORMAL"):
                for variant in VARIANTS:
                    table[f"{group}/s{s}/{policy}/{variant}"] = paired(group, s, policy, variant, by)
    significant = lambda key: table[key]["diff"] > 0 and table[key]["ci"][0] > 0  # noqa: E731
    verdicts = {"table": table, "r": json.loads(R_FILE.read_text()) if R_FILE.exists() else None}
    for name, s in (("C1", 3), ("C2", 6)):
        wins = [g for g in GROUPS if significant(f"{g}/s{s}/FILL-HOLD/live")]
        verdicts[name] = {"s": s, "groups_significant": wins, "n": len(wins), "pass": len(wins) >= 4}
    cells = [table[f"{g}/s{s}/FILL-HOLD/live"]["diff"] for g in GROUPS for s in SLOWDOWNS]
    verdicts["C3"] = {"mean_diff_live": float(np.mean(cells)), "n_cells": len(cells), "pass": bool(np.mean(cells) > 0)}
    c4 = {g: table[f"{g}/s3/FILL-HOLD/live"]["diff"] for g in ("SHT-L0-8s", "UB-L0-8s")}
    verdicts["C4"] = {"diff_live_s3": c4, "pass": all(v > 0 for v in c4.values())}
    verdicts["claim_allowed"] = bool(verdicts["C1"]["pass"] and verdicts["C2"]["pass"])
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "verdicts.json").write_text(json.dumps(verdicts, indent=1, default=str))
    print(json.dumps({k: v for k, v in verdicts.items() if k != "table"}, indent=1, default=str))
    (OUT / "results_tables.md").write_text(results_markdown(rows, verdicts))
    if OUT == DATA / "exp010":
        (EXPERIMENT_DIR / "results_tables.md").write_text(results_markdown(rows, verdicts))
    return verdicts


def results_markdown(rows: list[dict], verdicts: dict) -> str:
    table, r = verdicts["table"], verdicts["r"]
    yes = lambda ok: "PASS" if ok else "FAIL"  # noqa: E731
    lines = ["# EXP-010 results (generated by `edgelab.analysis.exp010 analyse`; AUC x 100, paired bootstrap over videos, 1000 resamples)\n",
             f"YOLOv8s cost factor r = {r['r']:.3f} (8s detector {r['s_mean_ms']:.2f} ms vs 8n {r['n_mean_ms']:.2f} ms); keep masks seed 1; G = 8.\n",
             "## Verdicts (plan §3)\n", "| # | rule | result | verdict |", "|---|---|---|---|"]
    c1, c2, c3, c4 = (verdicts[k] for k in ("C1", "C2", "C3", "C4"))
    lines += [f"| C1 | FILL-HOLD > SKIP, CI excludes 0, >= 4 of 6 groups, s = 3, live | {c1['n']} of 6 ({', '.join(c1['groups_significant']) or 'none'}) | {yes(c1['pass'])} |",
              f"| C2 | same at s = 6 | {c2['n']} of 6 ({', '.join(c2['groups_significant']) or 'none'}) | {yes(c2['pass'])} |",
              f"| C3 | mean of the 12 live FILL-HOLD - SKIP differences > 0 | {c3['mean_diff_live']:+.2f} | {yes(c3['pass'])} |",
              f"| C4 | both 8s groups FILL-HOLD - SKIP > 0 at s = 3 (live) | {', '.join(f'{k} {v:+.2f}' for k, v in c4['diff_live_s3'].items())} | {yes(c4['pass'])} |",
              f"\n**Claim 'causal pose prediction rescues delivered accuracy under CPU deadlines' (needs C1 and C2): "
              f"{'ALLOWED' if verdicts['claim_allowed'] else 'NOT allowed; EXP-009 stays exploratory, not replicated'}.**\n"]
    for policy in ("FILL-HOLD", "FILL-NORMAL"):
        lines += [f"## {policy} - SKIP (positive = fill better)\n",
                  "| group | s | drop rate | SKIP live | fill live | diff live [95% CI] | SKIP paper | fill paper | diff paper [95% CI] | fill coverage anom/normal | predicted share |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
        for group in GROUPS:
            for s in SLOWDOWNS:
                lv, pp = table[f"{group}/s{s}/{policy}/live"], table[f"{group}/s{s}/{policy}/paper"]
                row = next(x for x in rows if x["condition"] == f"{group}/s{s}/{policy}" and x["variant"] == "live")
                sig = lambda t: "*" if t["ci"][0] > 0 or t["ci"][1] < 0 else ""  # noqa: E731
                lines.append(f"| {group} | {s} | {row['drop_rate']:.2f} | {lv['skip']:.2f} | {lv['policy']:.2f} | "
                             f"{lv['diff']:+.2f} [{lv['ci'][0]:+.2f}, {lv['ci'][1]:+.2f}]{sig(lv)} | {pp['skip']:.2f} | {pp['policy']:.2f} | "
                             f"{pp['diff']:+.2f} [{pp['ci'][0]:+.2f}, {pp['ci'][1]:+.2f}]{sig(pp)} | "
                             f"{row['coverage_anomalous']:.3f} / {row['coverage_normal']:.3f} | {row['predicted_share_of_scored_windows']:.3f} |")
        lines.append("\n(* = 95% CI excludes 0)\n")
    lines += ["## All arms (AUC [95% CI over videos])\n", "| condition | paper | live | segments |", "|---|---|---|---|"]
    for cond in dict.fromkeys(x["condition"] for x in rows):
        paper = next(x for x in rows if x["condition"] == cond and x["variant"] == "paper")
        live = next(x for x in rows if x["condition"] == cond and x["variant"] == "live")
        fmt = lambda x: f"{x['auc'] * 100:.2f}" + (f" [{x['ci'][0] * 100:.1f}, {x['ci'][1] * 100:.1f}]" if x["ci"] else "")  # noqa: E731
        lines.append(f"| {cond} | {fmt(paper)} | {fmt(live)} | {paper['num_segments']} |")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--part", choices=list(PARTS), required=True)
    run.add_argument("--threads", type=int, default=2)
    analyse_parser = sub.add_parser("analyse")
    for p in (run, analyse_parser):
        p.add_argument("--seed", type=int, default=1, help="keep-mask bootstrap seed (EXP-010: 1)")
        p.add_argument("--out-root", type=Path, default=None, help="where rows/scores/poses go (default ~/thesis-data/exp010)")
    args = parser.parse_args()
    configure(args.seed, args.out_root)
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    run_part(args.part, args.threads) if args.command == "run" else analyse()


if __name__ == "__main__":
    main()
