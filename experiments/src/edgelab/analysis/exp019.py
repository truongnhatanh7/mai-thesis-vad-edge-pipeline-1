"""EXP-019 driver: standard STG-NF vs STG-NF with GAU coupling networks, 3 seeds each (plan + Amendment 1).

    uv run --extra pose python -m edgelab.analysis.exp019 eval [--threads 2]   # score every finished model on every test set; resumable
    uv run --extra pose python -m edgelab.analysis.exp019 timing               # CPU ms per window, 1 thread, batch 1
    uv run --extra pose python -m edgelab.analysis.exp019 analyse              # G1-G3 + tables -> results_tables.md, verdicts.json

Models: ~/thesis-data/exp019/models/<standard|gau>_s<seed>/final.pt (edgelab.pose.stgnf_train), plus the released checkpoint as a
reference row. Test sets: released AlphaPose test poses, YOLOv8n-pose edge poses (EXP-003 L0_E0 c0.25 botsort), top-down poses
(EXP-016 L0 n640 K1 rtmpose-s). Rows: ~/thesis-data/exp019/rows.jsonl, per-frame scores next to them.
"""

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

import mlflow
import numpy as np
import torch
from scipy import stats
from sklearn.metrics import roc_auc_score

from edgelab.analysis.exp009 import load_scores_file
from edgelab.metrics.scores import bootstrap_ci_over_videos, frame_level_auc
from edgelab.pose.evaluate import CHECKPOINT, GT_DIR
from edgelab.pose.score_dir import VARIANTS, coverage, variant_scores
from edgelab.pose.score_model import build_model, load_model, parameter_count, score_segments_model
from edgelab.pose.stgnf_eval import STGNF_ROOT
from edgelab.tracking.mlflow_run import tracked_run

DATA = Path.home() / "thesis-data"
OUT = DATA / "exp019"
ROWS = OUT / "rows.jsonl"
EXPERIMENT_DIR = Path(__file__).resolve().parents[3] / "EXP-019-stgnf-gau"
SEEDS = (0, 1, 2)
KINDS = ("standard", "gau")
TEST_SETS = {"released": STGNF_ROOT / "data/ShanghaiTech/pose/test",
             "edge": DATA / "exp003/poses/L0_E0_c0.25_botsort",
             "topdown": DATA / "exp016/poses/L0_n640_K1_rtmpose-s"}
N_PAIRED = 1000
N_ROW_CI = 300


def arm_names() -> list[str]:
    return ["released"] + [f"{kind}_s{seed}" for kind in KINDS for seed in SEEDS]


def model_path(arm: str) -> Path:
    return CHECKPOINT if arm == "released" else OUT / "models" / arm / "final.pt"


def read_rows() -> list[dict]:
    return [json.loads(line) for line in ROWS.read_text().splitlines() if line.strip()] if ROWS.exists() else []


def log(message: str):
    with open(EXPERIMENT_DIR / "log.md", "a") as f:
        f.write(f"\n### {datetime.now():%Y-%m-%d %H:%M} — driver\n- {message}\n")


def evaluate(threads: int):
    OUT.mkdir(parents=True, exist_ok=True)
    for arm in arm_names():
        if not model_path(arm).exists():
            continue
        for test, pose_dir in TEST_SETS.items():
            done = {r["variant"] for r in read_rows() if r["arm"] == arm and r["test"] == test}
            if done >= set(VARIANTS):
                continue
            model = load_model(model_path(arm))
            segments = score_segments_model(model, pose_dir, threads=threads)
            rows = []
            for variant, (offset_name, smoother_name) in VARIANTS.items():
                names, labels, scores, covered = variant_scores(segments, GT_DIR, offset_name, smoother_name)
                auc = frame_level_auc(labels, scores)
                low, high = bootstrap_ci_over_videos(labels, scores, n_resamples=N_ROW_CI)
                cov_a, cov_n = coverage(covered, labels)
                scores_file = OUT / "scores" / arm / test / f"frame_scores_{variant}.npz"
                scores_file.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(scores_file, names=np.array(names), **{f"score_{n}": s for n, s in zip(names, scores)},
                                    **{f"label_{n}": y for n, y in zip(names, labels)},
                                    **{f"covered_{n}": c for n, c in zip(names, covered)})
                row = {"arm": arm, "test": test, "variant": variant, "auc": auc, "ci": [low, high], "coverage_anomalous": cov_a,
                       "coverage_normal": cov_n, "num_segments": segments.num_segments, "params": parameter_count(model),
                       "scores_file": str(scores_file), "pose_dir": str(pose_dir)}
                with tracked_run("EXP-019", f"{arm}-{test}-{variant}", {k: v for k, v in row.items() if k != "ci"},
                                 input_files={"checkpoint": model_path(arm)}):
                    mlflow.log_metrics({"auc": auc, "ci_low": low, "ci_high": high, "coverage_anomalous": cov_a,
                                        "coverage_normal": cov_n, "num_segments": segments.num_segments})
                rows.append(row)
                print(f"{arm:12s} {test:9s} {variant:5s} AUC={auc * 100:.2f} [{low * 100:.2f},{high * 100:.2f}] seg={segments.num_segments}", flush=True)
            with open(ROWS, "a") as f:
                f.write("".join(json.dumps(r) + "\n" for r in rows))
    log("eval pass finished: " + ", ".join(sorted({r["arm"] for r in read_rows()})))


# ---------------------------------------------------------------- timing (G3)

def time_model(model: torch.nn.Module, calls: int, x: torch.Tensor) -> list[float]:
    model.eval()
    times = []
    with torch.no_grad():
        for _ in range(50):
            model(x)
        for _ in range(calls):
            t0 = time.perf_counter()
            model(x)
            times.append((time.perf_counter() - t0) * 1e3)
    return times


def timing(repeats: int = 5, calls: int = 2000):
    """CPU ms per window (batch 1, 1 torch thread). Standard and GAU interleaved in the same session; each repeat = `calls` forwards."""
    torch.set_num_threads(1)
    models = {}
    for arm in ("released", "standard_s0", "gau_s0"):
        if model_path(arm).exists():
            models[arm] = load_model(model_path(arm))
    if "gau_s0" not in models or "standard_s0" not in models:  # cost does not depend on the weight values
        for kind in ("standard", "gau"):
            if f"{kind}_s0" not in models:
                fresh = build_model(kind, 24)
                fresh.train()
                fresh(torch.randn(8, 2, 24, 18))
                models[f"{kind}_s0"] = fresh.eval()
    x = torch.from_numpy(np.random.default_rng(0).standard_normal((1, 2, 24, 18)).astype(np.float32))
    result = {"threads": 1, "calls_per_repeat": calls, "loadavg_start": os.getloadavg(), "repeats": {}}
    for repeat in range(repeats):
        for arm, model in models.items():
            t = time_model(model, calls, x)
            result["repeats"].setdefault(arm, []).append({"median_ms": float(np.median(t)), "mean_ms": float(np.mean(t)),
                                                          "p95_ms": float(np.percentile(t, 95)), "loadavg": os.getloadavg()[0]})
    result["params"] = {arm: parameter_count(m) for arm, m in models.items()}
    result["summary_median_ms"] = {arm: float(np.median([r["median_ms"] for r in rs])) for arm, rs in result["repeats"].items()}
    (OUT / "timing.json").write_text(json.dumps(result, indent=1))
    print(json.dumps({"median_ms": result["summary_median_ms"], "params": result["params"]}))
    return result


# ---------------------------------------------------------------- analysis

def seed_paired_bootstrap(test: str, variant: str, by: dict, n_resamples: int = N_PAIRED, seed: int = 0) -> dict:
    """Mean over seeds of AUC(GAU s) - AUC(standard s) (x100), bootstrap over videos, one resample for all six models."""
    loaded = {arm: load_scores_file(by[(arm, test, variant)]["scores_file"]) for arm in arm_names()[1:]}
    reference = loaded["standard_s0"]
    labels = reference.labels
    n = len(labels)
    rng = np.random.default_rng(seed)

    def mean_auc(kind: str, idx) -> float:
        y = np.concatenate([labels[i] for i in idx])
        return float(np.mean([roc_auc_score(y, np.concatenate([loaded[f"{kind}_s{s}"].scores[i] for i in idx])) for s in SEEDS]))

    everyone = np.arange(n)
    point = {k: mean_auc(k, everyone) * 100 for k in KINDS}
    diffs, per_kind = [], {k: [] for k in KINDS}
    while len(diffs) < n_resamples:
        idx = rng.integers(0, n, n)
        if len(np.unique(np.concatenate([labels[i] for i in idx]))) < 2:
            continue
        a = {k: mean_auc(k, idx) for k in KINDS}
        for k in KINDS:
            per_kind[k].append(a[k])
        diffs.append(a["gau"] - a["standard"])
    seed_diffs = np.array([by[(f"gau_s{s}", test, variant)]["auc"] - by[(f"standard_s{s}", test, variant)]["auc"] for s in SEEDS]) * 100
    t_half = stats.t.ppf(0.975, df=len(SEEDS) - 1) * seed_diffs.std(ddof=1) / np.sqrt(len(SEEDS))
    return {"mean_standard": point["standard"], "mean_gau": point["gau"], "diff": point["gau"] - point["standard"],
            "ci_videos": [float(np.quantile(diffs, 0.025) * 100), float(np.quantile(diffs, 0.975) * 100)],
            "ci_standard": [float(np.quantile(per_kind["standard"], 0.025) * 100), float(np.quantile(per_kind["standard"], 0.975) * 100)],
            "ci_gau": [float(np.quantile(per_kind["gau"], 0.025) * 100), float(np.quantile(per_kind["gau"], 0.975) * 100)],
            "seed_diffs": seed_diffs.tolist(), "ci_seeds_t": [float(seed_diffs.mean() - t_half), float(seed_diffs.mean() + t_half)]}


def analyse() -> dict:
    rows = read_rows()
    by = {(r["arm"], r["test"], r["variant"]): r for r in rows}
    missing = [(a, t, v) for a in arm_names() for t in TEST_SETS for v in VARIANTS if (a, t, v) not in by]
    if missing:
        raise SystemExit(f"{len(missing)} rows missing, e.g. {missing[:3]}")
    paired = {f"{t}/{v}": seed_paired_bootstrap(t, v, by) for t in TEST_SETS for v in VARIANTS}
    timing_json = json.loads((OUT / "timing.json").read_text()) if (OUT / "timing.json").exists() else None
    ms = timing_json["summary_median_ms"] if timing_json else {}
    g1 = paired["released/paper"]
    g2 = {t: paired[f"{t}/paper"]["diff"] for t in ("edge", "topdown")}
    verdicts = {"paired": paired, "timing": timing_json,
                "G1": {"diff": g1["diff"], "ci_videos": g1["ci_videos"], "ci_seeds_t": g1["ci_seeds_t"],
                       "pass": bool(g1["diff"] >= 1.0 and g1["ci_videos"][0] > 0)},
                "G2": {"diff_vs_standard": g2, "pass": bool(all(d >= -0.5 for d in g2.values()))},
                "G3": {"gau_ms": ms.get("gau_s0"), "standard_ms": ms.get("standard_s0"),
                       "pass": bool(ms.get("gau_s0") is not None and ms["gau_s0"] <= 2.0)}}
    (OUT / "verdicts.json").write_text(json.dumps(verdicts, indent=1, default=str))
    (EXPERIMENT_DIR / "results_tables.md").write_text(markdown(rows, verdicts))
    print(json.dumps({k: verdicts[k] for k in ("G1", "G2", "G3")}, indent=1, default=str))
    return verdicts


def markdown(rows: list[dict], verdicts: dict) -> str:
    by = {(r["arm"], r["test"], r["variant"]): r for r in rows}
    ok = lambda b: "PASS" if b else "FAIL"  # noqa: E731
    g1, g2, g3 = verdicts["G1"], verdicts["G2"], verdicts["G3"]
    lines = ["# EXP-019 results tables (generated by `edgelab.analysis.exp019 analyse`; AUC x 100)\n",
             "## Verdicts (paper variant, mean over 3 seeds)\n", "| # | rule | result | verdict |", "|---|---|---|---|",
             f"| G1 | GAU − standard ≥ +1.0 on released test poses, seed-paired CI excludes 0 | {g1['diff']:+.2f}, CI over videos [{g1['ci_videos'][0]:+.2f}, {g1['ci_videos'][1]:+.2f}]; t-interval over 3 seed differences [{g1['ci_seeds_t'][0]:+.2f}, {g1['ci_seeds_t'][1]:+.2f}] | {ok(g1['pass'])} |",
             f"| G2 | GAU ≥ standard − 0.5 on edge and on top-down poses | edge {g2['diff_vs_standard']['edge']:+.2f}, top-down {g2['diff_vs_standard']['topdown']:+.2f} | {ok(g2['pass'])} |",
             f"| G3 | GAU ≤ 2 ms per window (1 thread, batch 1) | GAU {g3['gau_ms']} ms vs standard {g3['standard_ms']} ms | {ok(g3['pass'])} |\n",
             "## AUC per model and test set (paper variant [live variant])\n",
             "| model | params | released | edge (YOLOv8n-pose) | top-down (rtmpose-s) |", "|---|---|---|---|---|"]
    for arm in arm_names():
        cells = [f"{by[(arm, t, 'paper')]['auc'] * 100:.2f} [{by[(arm, t, 'live')]['auc'] * 100:.2f}]" for t in TEST_SETS]
        lines.append(f"| {arm} | {by[(arm, 'released', 'paper')]['params']} | " + " | ".join(cells) + " |")
    lines += ["\n## Mean over seeds and paired differences (GAU − standard)\n",
              "| test set | variant | standard (mean) | GAU (mean) | difference | CI over videos (seed-paired) | t-interval over 3 seed differences | seed differences |",
              "|---|---|---|---|---|---|---|---|"]
    for key, p in verdicts["paired"].items():
        test, variant = key.split("/")
        lines.append(f"| {test} | {variant} | {p['mean_standard']:.2f} [{p['ci_standard'][0]:.1f}, {p['ci_standard'][1]:.1f}] | "
                     f"{p['mean_gau']:.2f} [{p['ci_gau'][0]:.1f}, {p['ci_gau'][1]:.1f}] | {p['diff']:+.2f} | "
                     f"[{p['ci_videos'][0]:+.2f}, {p['ci_videos'][1]:+.2f}] | [{p['ci_seeds_t'][0]:+.2f}, {p['ci_seeds_t'][1]:+.2f}] | "
                     + ", ".join(f"{d:+.2f}" for d in p["seed_diffs"]) + " |")
    if verdicts["timing"]:
        t = verdicts["timing"]
        lines += ["\n## CPU cost per window (batch 1, 1 thread, this Mac, other jobs running)\n",
                  f"Load average at start {t['loadavg_start'][0]:.1f}; {t['calls_per_repeat']} forwards per repeat, {len(next(iter(t['repeats'].values())))} interleaved repeats.\n",
                  "| model | params | median ms (over repeats) | per-repeat medians |", "|---|---|---|---|"]
        for arm, reps in t["repeats"].items():
            lines.append(f"| {arm} | {t['params'][arm]} | {t['summary_median_ms'][arm]:.3f} | " + ", ".join(f"{r['median_ms']:.2f}" for r in reps) + " |")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("eval")
    run.add_argument("--threads", type=int, default=2)
    sub.add_parser("timing")
    sub.add_parser("analyse")
    args = parser.parse_args()
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    {"eval": lambda: evaluate(args.threads), "timing": timing, "analyse": analyse}[args.command]()


if __name__ == "__main__":
    main()
