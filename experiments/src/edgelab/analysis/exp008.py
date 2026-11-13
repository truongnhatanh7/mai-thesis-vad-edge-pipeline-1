"""EXP-008: delivered AUC of 6 knobs at simulated device slow-downs s, rank flips, and a delivered-AUC model.

    uv run --extra pose python -m edgelab.analysis.exp008 --group SHT-L2 --out ~/thesis-data/exp008     # replays
    uv run --extra pose python -m edgelab.analysis.exp008 --analyse --out ~/thesis-data/exp008           # verdicts

A slow-down s multiplies every measured per-frame service time (plan §2); s = 1 is the VM.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

from edgelab.analysis.exp003 import load_scores
from edgelab.analysis.exp004 import usable_traces
from edgelab.metrics.scores import frame_level_auc
from edgelab.pipeline.deadline_replay import load_trace, run_cell

DATA = Path.home() / "thesis-data"
MANIFEST = DATA / "vm-exchange/results/matrix/manifest.csv"
KNOBS = [("E0", 480), ("E0", 640), ("E0", 800), ("E0", 960), ("E1", 640), ("E2", 640)]
SLOWDOWNS = (1, 2, 4, 8)
GROUPS = {"SHT-L0": ("ShanghaiTech", "exp003", "L0"), "SHT-L2": ("ShanghaiTech", "exp003", "L2"),
          "SHT-L3": ("ShanghaiTech", "exp003", "L3"), "UB-L0": ("UBnormal", "exp006", "L0")}


def cache_name(group: str, enhance: str, imgsz: int) -> str:
    dataset, _, light = GROUPS[group]
    if imgsz == 640:
        return f"{light}_{enhance}"
    return f"{light}_{enhance}-r{imgsz}" if dataset == "ShanghaiTech" else f"{light}-{imgsz}_{enhance}"


def trace_cell(enhance: str, imgsz: int) -> str:
    return f"L3-{enhance}-R{imgsz}-c0.25-O0-B4"


def offline_auc(group: str, enhance: str, imgsz: int) -> float:
    _, root, _ = GROUPS[group]
    s = load_scores(DATA / root, cache_name(group, enhance, imgsz), 0.25, "live")
    return frame_level_auc(s.labels, s.scores)


def replay_group(group: str, out: Path) -> None:
    dataset, root, _ = GROUPS[group]
    traces = usable_traces(MANIFEST)
    rows_file = out / f"rows_{group}.jsonl"
    done = {(r["knob"], r["s"]) for r in map(json.loads, rows_file.read_text().splitlines())} if rows_file.exists() else set()
    for enhance, imgsz in KNOBS:
        cache = DATA / root / "cache" / cache_name(group, enhance, imgsz)
        if not cache.exists():
            continue
        trace = load_trace(traces[trace_cell(enhance, imgsz)])
        offline = offline_auc(group, enhance, imgsz)
        for s in SLOWDOWNS:
            knob = f"{enhance}@{imgsz}"
            if (knob, s) in done:
                continue
            result = run_cell(cache, trace * s, 0.25, out / "delivered" / f"{group}__{knob}__s{s}",
                              {"label": f"{group}-{knob}-s{s}", "slowdown": s}, n_resamples=200, threads=2,
                              dataset=dataset, experiment="EXP-008")
            with open(rows_file, "a") as f:
                f.write(json.dumps({"group": group, "knob": knob, "s": s, "offline": offline,
                                    "delivered": result["auc"], "drop_rate": result["drop_rate"]}) + "\n")


def analyse(out: Path) -> dict:
    df = pd.concat([pd.read_json(p, lines=True) for p in sorted(out.glob("rows_*.jsonl"))])
    v = {}
    s1 = df[df.s == 1]
    flips = {g: (d.loc[d.offline.idxmax(), "knob"], d.loc[d.delivered.idxmax(), "knob"]) for g, d in s1.groupby("group")}
    v["A1a_offline_vs_delivered_best"] = flips
    v["A1a_pass"] = sum(a != b for a, b in flips.values()) >= 2
    s4 = df[df.s == 4]
    v["A1b_kendall_tau_s4"] = float(kendalltau(s4.offline, s4.delivered)[0])
    v["A1b_kendall_tau_s1"] = float(kendalltau(s1.offline, s1.delivered)[0])
    v["A1b_pass"] = v["A1b_kendall_tau_s4"] <= 0.5
    best_res, monotone, wins480 = {}, True, 0
    for g, d in df[df.knob.str.startswith("E0@")].groupby("group"):
        best = {s: int(d[d.s == s].sort_values("delivered").iloc[-1].knob.split("@")[1]) for s in SLOWDOWNS}
        best_res[g] = best
        monotone &= all(best[a] >= best[b] for a, b in zip(SLOWDOWNS, SLOWDOWNS[1:]))
        d8 = d[d.s == 8].set_index("knob").delivered
        wins480 += int(d8.get("E0@480", -1) > d8.get("E0@640", 2))
    v["A2_best_imgsz_by_s"] = best_res
    v["A2_pass"] = bool(monotone and wins480 >= 3)
    v["A2_480_beats_640_at_s8_groups"] = wins480
    train, test = df[df.group.str.startswith("SHT")], df[df.group.str.startswith("UB")]
    X = lambda d: np.column_stack([np.ones(len(d)), d.offline, d.drop_rate])  # noqa: E731
    coef, *_ = np.linalg.lstsq(X(train), train.delivered, rcond=None)
    pred = X(test) @ coef
    v["A3_coef_a_b_c"] = [float(x) for x in coef]
    v["A3_mae_auc_points"] = float(np.mean(np.abs(pred - test.delivered)) * 100)
    v["A3_spearman"] = float(spearmanr(pred, test.delivered)[0])
    v["A3_pass"] = v["A3_mae_auc_points"] <= 1.5 and v["A3_spearman"] >= 0.7
    df.to_csv(out / "all_rows.csv", index=False)
    (out / "verdicts.json").write_text(json.dumps(v, indent=1, default=str))
    return v


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", choices=list(GROUPS))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--analyse", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.analyse:
        print(json.dumps(analyse(args.out), indent=1, default=str))
    else:
        replay_group(args.group, args.out)


if __name__ == "__main__":
    main()
