"""EXP-020 evaluation glue: AUC tables, recall by height, delivered AUC, verdicts P1-P5 (rules: plan.md + its amendments).

    uv run --extra pose python -m edgelab.analysis.exp020 report --out-dir experiments/artifacts/EXP-020
    uv run --extra pose python -m edgelab.analysis.exp020 delivered [--arms M00 ML0 M0P MLP stock-R640 stock-R800] [--slowdowns 1 3]

Data layout (written by EXP-020-small-person-detector/scripts/run_eval.sh and gate_sweep):
    ~/thesis-data/exp020-<arm>-{sht,ub}/{cache,scores,results.jsonl}   MPS FP32 detections of the trained .pt (c 0.25 replay)
    ~/thesis-data/exp020-ncnn/                                          NCNN FP16 640 detections of each arm (L0), for P4
    ~/thesis-data/exp012/ (cache L0_E0-ncnn16-800)                      stock yolov8n-pose @800 NCNN FP16 reference (EXP-012)
    ~/thesis-data/exp003, exp006                                        the original yolov8n-pose @640 (EXP-003 / EXP-006)
Every difference is AUC(arm) - AUC(reference), paired bootstrap over videos (analysis.exp003.diff, 1000 resamples).
"""

import argparse
import json
import os
from pathlib import Path

from edgelab.analysis import exp010
from edgelab.analysis.exp003 import diff, load_scores
from edgelab.analysis.exp011 import BIN_NAMES, recall, recall_by_bin, reference_people

DATA = Path.home() / "thesis-data"
ARMS = ("M00", "ML0", "M0P", "MLP")
SETTINGS = {"SHT L0": ("sht", "L0_E0"), "SHT L2": ("sht", "L2_E0"), "SHT L3": ("sht", "L3_E0"), "UB L0": ("ub", "L0_E0")}
VARIANTS = ("paper", "live")
MID_BINS = ("48-64", "64-96")  # "recall(48-96 px)" of P1: the two bins pooled
P_MARGIN = 0.02  # P1 / P2: +2.0 AUC points
P3_FLOOR = 0.005  # P3: not worse than M00 - 0.5 points
P4_MS_RATIO = 1.10
EPS = 1e-9
OUT = DATA / "exp020"


def arm_root(arm: str, kind: str) -> Path:
    return DATA / f"exp020-{arm}-{kind}"


def stock800_ncnn() -> tuple[Path, str]:
    return DATA / "exp012", "L0_E0-ncnn16-800"


def ncnn640(arm: str) -> tuple[Path, str]:
    return DATA / "exp020-ncnn", f"L0_E0-{arm}"


# ---------------------------------------------------------------- rules (plain functions; tests/test_analysis_exp020.py)

def pooled_recall(rec: dict, bins=MID_BINS) -> float:
    """Recall pooled over the given height bins of one recall_by_bin result."""
    persons = sum(rec[b]["persons"] for b in bins)
    return sum(rec[b]["found"] for b in bins) / persons if persons else float("nan")


def p1_label_cause(auc_ml0: float, auc_m00: float, ci_low: float, recall_ml0: float, recall_m00: float) -> bool:
    return auc_ml0 - auc_m00 >= P_MARGIN - EPS and ci_low > 0 and recall_ml0 > recall_m00


def p2_resolution_cause(auc_m0p: float, auc_m00: float, ci_low: float) -> bool:
    return auc_m0p - auc_m00 >= P_MARGIN - EPS and ci_low > 0


def best_of(aucs_sht_l0: dict[str, float]) -> str:
    """Amendment 1 reading of "best of {ML0, M0P, MLP}": the arm with the highest SHT L0 paper AUC."""
    return max(aucs_sht_l0, key=aucs_sht_l0.get)


def p3_general(ub_best: float, ub_m00: float, sht_best: float, sht_m00: float) -> bool:
    """Amendment 1: best arm beats M00 on UBnormal L0 (point > 0) and is >= M00 - 0.5 points on ShanghaiTech L0."""
    return ub_best - ub_m00 > 0 and sht_best >= sht_m00 - P3_FLOOR - EPS


def p4_p2_wins(auc_p2: float, auc_ref: float, ms_p2: float, ms_ref: float) -> bool:
    return auc_p2 >= auc_ref - EPS and ms_p2 <= P4_MS_RATIO * ms_ref + EPS


def p5_combination(auc_mlp: float, auc_ml0: float, auc_m0p: float) -> bool:
    return auc_mlp >= max(auc_ml0, auc_m0p) - EPS


# ---------------------------------------------------------------- loading

def sweep_row(root: Path, cache: str, variant: str, c: float = 0.25) -> dict:
    for line in (root / "results.jsonl").read_text().splitlines():
        row = json.loads(line)
        if row["cache"] == cache and row["c"] == c and row["tracker"] == "botsort" and row["variant"] == variant:
            return row
    raise KeyError(f"no row for {cache} c={c} {variant} in {root}")


def auc_cell(root: Path, cache: str, variant: str, ref_root: Path | None = None, ref_cache: str | None = None) -> dict:
    """AUC (paper / live), 95% CI over videos and coverage of one detector; with a reference also the paired difference."""
    row = sweep_row(root, cache, variant)
    out = {"auc": row["auc"], "ci": [row["ci95_low"], row["ci95_high"]], "coverage_anomalous": row["coverage_anomalous"],
           "coverage_normal": row["coverage_normal"]}
    if ref_root is not None:
        a, b = load_scores(ref_root, ref_cache or cache, 0.25, variant), load_scores(root, cache, 0.25, variant)
        assert a.names == b.names, "arms must score the same videos in the same order"
        out["vs_ref"] = diff(a, b)
    return out


def recall_table(caches: dict[str, Path]) -> dict:
    """label -> recall_by_bin on SHT L0 detections (EXP-011 matcher: conf >= 0.25, IoU >= 0.3, reference = released poses)."""
    people = reference_people()
    return {label: recall_by_bin(people, cache) for label, cache in caches.items()}


# ---------------------------------------------------------------- report

def available_arms() -> list[str]:
    return [a for a in ARMS if all((arm_root(a, k) / "results.jsonl").exists() for k in ("sht", "ub"))]


def build_report() -> dict:
    arms = available_arms()
    report = {"arms": arms, "auc": {}, "recall": {}, "verdicts": {}}
    orig = {"sht": DATA / "exp003", "ub": DATA / "exp006"}
    for setting, (kind, cache) in SETTINGS.items():
        report["auc"][setting] = {}
        for v in VARIANTS:
            cells = {"orig@640": auc_cell(orig[kind], cache, v)}
            for arm in arms:
                ref = arm_root("M00", kind) if "M00" in arms and arm != "M00" else None
                cells[arm] = auc_cell(arm_root(arm, kind), cache, v, ref, cache)  # vs M00
                cells[arm]["vs_orig"] = diff(load_scores(orig[kind], cache, 0.25, v), load_scores(arm_root(arm, kind), cache, 0.25, v))
            report["auc"][setting][v] = cells
    ref_root, ref_cache = stock800_ncnn()
    if (ref_root / "results.jsonl").exists():
        for v in VARIANTS:
            report["auc"].setdefault("stock@800 NCNN", {})[v] = {"SHT L0": auc_cell(ref_root, ref_cache, v)}
            for arm in arms:
                root, cache = ncnn640(arm)
                if (root / "results.jsonl").exists() and (root / "scores" / f"{cache}_c0.25_botsort").exists():
                    report["auc"].setdefault("NCNN FP16 640 vs stock@800 NCNN", {}).setdefault(v, {})[arm] = auc_cell(
                        root, cache, v, ref_root, ref_cache)
    caches = {"orig@640": DATA / "exp003/cache/L0_E0", **{a: arm_root(a, "sht") / "cache/L0_E0" for a in arms},
              "stock@800 NCNN": ref_root / "cache" / ref_cache}
    report["recall"] = recall_table({k: v for k, v in caches.items() if v.exists()})
    report["timing"] = timing_table()
    report["delivered"] = delivered_table()
    report["verdicts"] = verdicts(report)
    return report


def timing_table() -> dict:
    """variant -> mean detector ms (pre + infer + post) etc. from the VM timing manifest (empty before the timing ran)."""
    from edgelab.pipeline.timing_exp020 import RESULTS
    from edgelab.pipeline.timing_exp012 import summary

    if not (RESULTS / "manifest.csv").exists():
        return {}
    table = summary(RESULTS)
    return {r["variant"]: {k: float(v) for k, v in r.items() if k != "variant"} for r in table.to_dict("records")}


def verdicts(report: dict) -> dict:
    out = {}
    auc, rec, arms = report["auc"], report["recall"], report["arms"]
    if "M00" not in arms:
        return out
    paper = auc["SHT L0"]["paper"]
    if "ML0" in arms:
        d = paper["ML0"]["vs_ref"]
        out["P1"] = {"pass": p1_label_cause(paper["ML0"]["auc"], paper["M00"]["auc"], d["ci"][0],
                                            pooled_recall(rec["ML0"]), pooled_recall(rec["M00"])),
                     "delta": d["diff"], "ci": d["ci"], "recall_48_96": {a: pooled_recall(rec[a]) for a in ("M00", "ML0")}}
    if "M0P" in arms:
        d = paper["M0P"]["vs_ref"]
        out["P2"] = {"pass": p2_resolution_cause(paper["M0P"]["auc"], paper["M00"]["auc"], d["ci"][0]),
                     "delta": d["diff"], "ci": d["ci"]}
    contenders = [a for a in ("ML0", "M0P", "MLP") if a in arms]
    if contenders:
        best = best_of({a: paper[a]["auc"] for a in contenders})
        ub = auc["UB L0"]["paper"]
        out["P3"] = {"best": best, "pass": p3_general(ub[best]["auc"], ub["M00"]["auc"], paper[best]["auc"], paper["M00"]["auc"]),
                     "ub_delta": ub[best]["auc"] - ub["M00"]["auc"], "sht_delta": paper[best]["auc"] - paper["M00"]["auc"],
                     "any_beats_m00_on_ub": any(ub[a]["auc"] > ub["M00"]["auc"] for a in contenders)}
    p4 = auc.get("NCNN FP16 640 vs stock@800 NCNN", {}).get("paper", {}).get("M0P")
    if p4 is not None and {"M0P", "stock-R800"} <= set(report["timing"]):
        ref = auc["stock@800 NCNN"]["paper"]["SHT L0"]
        ms_p2, ms_ref = report["timing"]["M0P"]["det_ms"], report["timing"]["stock-R800"]["det_ms"]
        out["P4"] = {"pass": p4_p2_wins(p4["auc"], ref["auc"], ms_p2, ms_ref), "delta_auc": p4["vs_ref"]["diff"],
                     "ci": p4["vs_ref"]["ci"], "ms_m0p": ms_p2, "ms_stock800": ms_ref, "ms_ratio": ms_p2 / ms_ref}
    if {"ML0", "M0P", "MLP"} <= set(arms):
        out["P5"] = {"pass": p5_combination(paper["MLP"]["auc"], paper["ML0"]["auc"], paper["M0P"]["auc"]),
                     "mlp": paper["MLP"]["auc"], "ml0": paper["ML0"]["auc"], "m0p": paper["M0P"]["auc"]}
    return out


def markdown(report: dict) -> str:
    pts = lambda x: f"{x * 100:.2f}"  # noqa: E731
    sg = lambda x: f"{x * 100:+.2f}"  # noqa: E731
    lines = []
    for v in VARIANTS:
        lines += [f"### AUC x 100 ({v} variant; 95% CI over videos; Δ vs M00 paired)", "",
                  "| setting | detector | AUC | CI | cov. anomalous | Δ vs M00 [CI] |", "|---|---|---|---|---|---|"]
        for setting in SETTINGS:
            for name, c in report["auc"][setting][v].items():
                d = c.get("vs_ref")
                lines.append(f"| {setting} | {name} | {pts(c['auc'])} | [{pts(c['ci'][0])}, {pts(c['ci'][1])}] | "
                             f"{c['coverage_anomalous']:.3f} | " + (f"{sg(d['diff'])} [{sg(d['ci'][0])}, {sg(d['ci'][1])}]" if d else "—") + " |")
        lines.append("")
    lines += ["### Recall by reference-person height (SHT L0, conf 0.25, IoU 0.3)", "",
              "| detector | " + " | ".join(BIN_NAMES) + " | all | 48-96 pooled |", "|---|" + "---|" * (len(BIN_NAMES) + 2)]
    for name, r in report["recall"].items():
        lines.append(f"| {name} | " + " | ".join(f"{recall(r[b]):.3f}" for b in BIN_NAMES) + f" | {recall(r['all']):.3f} | {pooled_recall(r):.3f} |")
    if report.get("timing"):
        lines += ["", "### NCNN FP16 detector time in the VM (EXP-012 protocol)", "", "| variant | runs | frames | det ms (pre+infer+post) | p95 | host outside VM % |",
                  "|---|---|---|---|---|---|"]
        for name, t in report["timing"].items():
            lines.append(f"| {name} | {t['runs']:.0f} | {t['frames']:.0f} | {t['det_ms']:.2f} | {t['det_p95_ms']:.2f} | {t['host_outside_vm_pct']:.1f} |")
    if report.get("delivered"):
        lines += ["", "### Delivered AUC x 100 (SHT L0, FILL-HOLD vs SKIP, measured VM trace x s)", "",
                  "| detector | s | policy | drop rate | paper | live |", "|---|---|---|---|---|---|"]
        keyed = {(r["group"], r["s"], r["policy"], r["variant"]): r for r in report["delivered"]}
        for group, s, policy in sorted({(g, s, p) for g, s, p, _ in keyed}):
            paper, live = keyed.get((group, s, policy, "paper")), keyed.get((group, s, policy, "live"))
            if paper and live:
                lines.append(f"| {group.removeprefix('SHT-L0-')} | {s} | {policy} | {paper['drop_rate']:.3f} | {paper['auc'] * 100:.2f} | {live['auc'] * 100:.2f} |")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- delivered (EXP-010 method with the measured trace of each detector)

def run_delivered(arms: list[str], slowdowns: tuple[int, ...], threads: int = 2):
    """FILL-HOLD / FILL-NORMAL / SKIP on SHT L0 for each detector: its own VM trace x s, EXP-010 keep masks (seed 1), P fill G 8."""
    from edgelab.pipeline.deadline_replay import load_trace
    from edgelab.pipeline.timing_exp020 import usable_trace_files

    caches = {"M00": arm_root("M00", "sht"), "ML0": arm_root("ML0", "sht"), "M0P": arm_root("M0P", "sht"),
              "MLP": arm_root("MLP", "sht")}
    groups = {}
    for arm in arms:
        if arm in caches:
            groups[f"SHT-L0-{arm}"] = ("ShanghaiTech", caches[arm] / "cache/L0_E0", arm)
        elif arm == "stock-R640":
            groups["SHT-L0-stock-R640"] = ("ShanghaiTech", DATA / "exp003/cache/L0_E0", arm)
        elif arm == "stock-R800":
            ref_root, ref_cache = stock800_ncnn()
            groups["SHT-L0-stock-R800"] = ("ShanghaiTech", ref_root / "cache" / ref_cache, arm)
    exp010.configure(seed=1, out_root=OUT / "delivered", mlflow_experiment="EXP-020",
                     log_dir=Path(__file__).resolve().parents[3] / "EXP-020-small-person-detector")
    exp010.GROUPS.clear()
    exp010.GROUPS.update(groups)
    exp010.trace_for = lambda detector: load_trace(usable_trace_files(detector))
    for s in slowdowns:
        for group in groups:
            exp010.run_group_speed(group, s, threads)


def delivered_table(slowdowns=(1, 3)) -> list[dict]:
    """Rows of the delivered runs: AUC paper/live per (detector, s, policy) with the drop rate."""
    exp010.configure(seed=1, out_root=OUT / "delivered")
    rows = []
    for r in exp010.read_rows():
        rows.append({"group": r["group"], "s": r["s"], "policy": r["policy"], "variant": r["variant"],
                     "auc": r["auc"], "ci": r.get("ci"), "drop_rate": r["drop_rate"]})
    return [r for r in rows if r["s"] in slowdowns]


def main():
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    rep = sub.add_parser("report")
    rep.add_argument("--out-dir", type=Path, required=True)
    dl = sub.add_parser("delivered")
    dl.add_argument("--arms", nargs="+", default=list(ARMS) + ["stock-R640", "stock-R800"])
    dl.add_argument("--slowdowns", type=int, nargs="+", default=[1, 3])
    dl.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.command == "report":
        report = build_report()
        args.out_dir.mkdir(parents=True, exist_ok=True)
        (args.out_dir / "report.json").write_text(json.dumps(report, indent=1, default=float))
        (args.out_dir / "report.md").write_text(markdown(report))
        print(markdown(report))
        print(json.dumps(report["verdicts"], default=float))
    else:
        run_delivered(args.arms, tuple(args.slowdowns), args.threads)


if __name__ == "__main__":
    main()
