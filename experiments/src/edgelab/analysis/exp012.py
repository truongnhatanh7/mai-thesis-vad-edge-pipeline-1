"""EXP-012 ("quantize to buy resolution"): verdicts Q1-Q3 from the VM timing, the offline sweep rows and the delivered replays.

    uv run --extra pose python -m edgelab.analysis.exp012 --delivered --variants ncnn-fp16-R800 ort-int8-R800 --lights L0 L2 \
        --also-fp32-640            # replays (host, slow): measured traces x s in {1, 2} -> ~/thesis-data/exp012/delivered/
    uv run --extra pose python -m edgelab.analysis.exp012 --verdicts     # writes ~/thesis-data/exp012/verdicts.json and prints a table

Comparator (decided in log.md before any result): "ORT FP32 640" = EXP-003 cache L0_E0, offline paper AUC at c 0.25, and the
EXP-008 delivered live AUC row SHT-L0 E0@640 (EXP-004 trace). The same-window FP32-640 trace replayed on the same cache is a
sensitivity row (`--also-fp32-640`).
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from edgelab.analysis.exp003 import diff, load_scores
from edgelab.analysis.exp008 import DATA
from edgelab.metrics.scores import paired_bootstrap_diff
from edgelab.pipeline.deadline_replay import load_trace, run_cell
from edgelab.pipeline.timing_exp012 import RESULTS as TIMING_DIR
from edgelab.pipeline.timing_exp012 import summary

ROOT = DATA / "exp012"
EXP003, EXP008 = DATA / "exp003", DATA / "exp008"
REFERENCE = "ort-fp32-R640"
FAST_800 = ("ort-int8-R800", "ncnn-fp16-R800", "ncnn-fp32-R800")
CACHE_TAG = {"ncnn-fp16-R800": "n16r800", "ncnn-fp32-R800": "n32r800", "ort-int8-R800": "i8r800", "ort-int8-R640": "i8r640",
             "ort-fp32-R800": "o32r800"}
SLOWDOWNS = (1, 2)
Q1_LIMIT, Q2_MARGIN_POINTS = 1.10, 1.0


def best_fast_800(table: pd.DataFrame) -> str:
    """Cheapest (mean detector ms) of the fast 800 variants in the timing summary."""
    rows = table[table.variant.isin(FAST_800)]
    return str(rows.loc[rows.det_ms.idxmin(), "variant"])


def q1_ratio(table: pd.DataFrame, variant: str) -> dict:
    """Q1: mean detector ms of `variant` / mean detector ms of ORT FP32 640 (same window); pass if <= 1.10."""
    ms = table.set_index("variant").det_ms
    ratio = float(ms[variant] / ms[REFERENCE])
    return {"ratio": ratio, "pass": ratio <= Q1_LIMIT}


def cache_name(light: str, variant: str) -> str:
    return f"{light}_E0-{CACHE_TAG[variant]}"


def usable_trace_files(variant: str, timing_dir: Path = TIMING_DIR) -> list[Path]:
    manifest = pd.read_csv(timing_dir / "manifest.csv")
    ok = manifest[(manifest.cell_id == variant) & (manifest.status == "ok") & (~manifest.excluded.astype(bool))]
    return [Path(p) for p in ok.trace_file]


def delivered_dir(light: str, variant: str, s: int, root: Path = ROOT) -> Path:
    return root / "delivered" / f"{light}__{variant}__s{s}"


def replay_delivered(light: str, variant: str, cache: Path, s: int, root: Path = ROOT) -> dict:
    """EXP-008 method: measured service times of `variant` x s, latest-frame deadline sim at 24 fps, replay, STG-NF, live AUC."""
    out = delivered_dir(light, variant, s, root)
    if (out / "result.json").exists():
        return json.loads((out / "result.json").read_text())
    trace = load_trace(usable_trace_files(variant))
    return run_cell(cache, trace * s, 0.25, out, {"label": f"{light}-{variant}-s{s}", "slowdown": s}, n_resamples=200, threads=2,
                    experiment="EXP-012")


def load_delivered(path: Path) -> tuple[list[str], list[np.ndarray], list[np.ndarray]]:
    data = np.load(path)
    names = [str(n) for n in data["names"]]
    return names, [data[f"label_{n}"] for n in names], [data[f"score_{n}"] for n in names]


def delivered_diff(path_a: Path, path_b: Path, n_resamples: int = 1000) -> dict:
    """AUC(a) - AUC(b), paired bootstrap over videos (same videos, same labels in both files)."""
    names_a, labels, scores_a = load_delivered(path_a)
    names_b, _, scores_b = load_delivered(path_b)
    assert names_a == names_b, "the two delivered files do not hold the same videos"
    d, low, high = paired_bootstrap_diff(labels, scores_a, scores_b, n_resamples=n_resamples)
    return {"diff": d, "ci": [low, high]}


def sweep_row(root: Path, cache: str, variant: str, c: float = 0.25) -> dict:
    """The gate-sweep row (c, 'paper' or 'live') of one cache in root/results.jsonl."""
    rows = [json.loads(line) for line in (root / "results.jsonl").read_text().splitlines() if line.strip()]
    return next(r for r in rows if r["cache"] == cache and r["c"] == c and r["variant"] == variant)


def verdicts(root: Path = ROOT, timing_dir: Path = TIMING_DIR) -> dict:
    table = summary(timing_dir)
    best = best_fast_800(table)
    out = {"best_fast_800": best, "timing": table.round(3).to_dict("records"), "Q1": {**q1_ratio(table, best), "variant": best}}
    out["Q1_all_fast_800"] = {v: q1_ratio(table, v) for v in FAST_800 if v in set(table.variant)}
    ref_paper, ref_live = (sweep_row(EXP003, "L0_E0", v)["auc"] for v in ("paper", "live"))
    out["offline_L0"] = {"ort-fp32-R640 (EXP-003)": {"paper": ref_paper, "live": ref_live}}
    for variant in CACHE_TAG:
        name = cache_name("L0", variant)
        if not (root / "scores" / f"{name}_c0.25_botsort").exists():
            continue
        row = {v: sweep_row(root, name, v)["auc"] for v in ("paper", "live")}
        row["paper_diff_vs_fp32_640"] = diff(load_scores(EXP003, "L0_E0", 0.25, "paper"),
                                             load_scores(root, name, 0.25, "paper"))
        out["offline_L0"][variant] = row
    q2 = out["offline_L0"].get(best)
    out["Q2"] = None if q2 is None else {"variant": best, "paper_auc": q2["paper"], "reference_paper_auc": ref_paper,
                                         "gain_points": (q2["paper"] - ref_paper) * 100,
                                         "pass": (q2["paper"] - ref_paper) * 100 >= Q2_MARGIN_POINTS}
    out["delivered"] = {}
    for light, knob in (("L0", "SHT-L0"), ("L2", "SHT-L2")):
        reference_rows = [json.loads(line) for line in (EXP008 / f"rows_{knob}.jsonl").read_text().splitlines()]
        for variant in CACHE_TAG:
            for s in SLOWDOWNS:
                path = delivered_dir(light, variant, s, root) / "result.json"
                if not path.exists():
                    continue
                result = json.loads(path.read_text())
                ref = next(r for r in reference_rows if r["knob"] == "E0@640" and r["s"] == s)
                ref_file = EXP008 / "delivered" / f"{knob}__E0@640__s{s}" / "delivered_scores_live.npz"
                mine = delivered_dir(light, variant, s, root) / "delivered_scores_live.npz"
                out["delivered"][f"{light}|{variant}|s{s}"] = {
                    "auc": result["auc"], "drop_rate": result["drop_rate"], "reference_auc": ref["delivered"],
                    "reference_drop_rate": ref["drop_rate"], "diff_vs_exp008_fp32_640": delivered_diff(mine, ref_file)}
    q3 = {s: out["delivered"].get(f"L0|{best}|s{s}") for s in SLOWDOWNS}
    out["Q3"] = None if None in q3.values() else {"variant": best, "pass": all(q3[s]["auc"] > q3[s]["reference_auc"] for s in SLOWDOWNS),
                                                  **{f"s{s}": {"auc": q3[s]["auc"], "reference_auc": q3[s]["reference_auc"]}
                                                     for s in SLOWDOWNS}}
    (root / "verdicts.json").write_text(json.dumps(out, indent=1, default=float))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--delivered", action="store_true")
    parser.add_argument("--variants", nargs="+", default=[])
    parser.add_argument("--lights", nargs="+", default=["L0"])
    parser.add_argument("--also-fp32-640", action="store_true", help="sensitivity: same-window FP32-640 trace on the EXP-003 cache")
    parser.add_argument("--verdicts", action="store_true")
    args = parser.parse_args()
    if args.delivered:
        for light in args.lights:
            for variant in args.variants:
                for s in SLOWDOWNS:
                    replay_delivered(light, variant, ROOT / "cache" / cache_name(light, variant), s)
            if args.also_fp32_640:
                for s in SLOWDOWNS:
                    replay_delivered(light, REFERENCE, EXP003 / "cache" / f"{light}_E0", s)
    if args.verdicts:
        print(json.dumps(verdicts(), indent=1, default=float))


if __name__ == "__main__":
    main()
