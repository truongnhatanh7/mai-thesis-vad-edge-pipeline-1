"""EXP-013 / EXP-014 tables and verdicts from the saved gate-sweep rows and per-frame scores (gate c = 0.25 only).

    uv run --extra pose python -m edgelab.analysis.exp013 --out-dir experiments/artifacts/EXP-013 [--alpha-star 0.5]

Arms (one detector each): baseline (original yolov8n-pose; EXP-003 / EXP-006 results), ft (EXP-013 fine-tuned),
a0.25 / a0.5 / a0.75 (EXP-014 weight interpolations). Settings: SHT L0, L2, L3 (ShanghaiTech test, EXP-003 darkening,
no enhancement) and UB L0 (UBnormal test). Paired differences use the same bootstrap as EXP-003 (analysis.exp003.diff).
Verdict rules are the frozen ones of the two plans; they are plain functions below, tested in tests/test_analysis_exp013.py.
"""

import argparse
import json
from pathlib import Path

from edgelab.analysis.exp003 import Scores, diff, load_scores
from edgelab.analysis.exp005 import coverage_all_frames
from edgelab.metrics.scores import frame_level_auc

DATA = Path.home() / "thesis-data"
ALPHAS = (0.25, 0.5, 0.75)
SETTINGS = {"SHT L0": ("sht", "L0_E0"), "SHT L2": ("sht", "L2_E0"), "SHT L3": ("sht", "L3_E0"), "UB L0": ("ub", "L0_E0")}
VARIANTS = ("paper", "live")


def arm_roots() -> dict[str, dict[str, Path]]:
    roots = {"baseline": {"sht": DATA / "exp003", "ub": DATA / "exp006"},
             "ft": {"sht": DATA / "exp013-sht", "ub": DATA / "exp013-ub"}}
    for alpha in ALPHAS:
        roots[f"a{alpha}"] = {"sht": DATA / f"exp014-a{alpha}-sht", "ub": DATA / f"exp014-a{alpha}-ub"}
    return roots


# ---- verdict rules (frozen in the plans); AUC and coverage are fractions in [0, 1] ----

EPS = 1e-9


def f1_more_people(coverage_ft: float, coverage_base: float) -> bool:
    return coverage_ft >= coverage_base + 0.04 - EPS


def f2_higher_accuracy(auc_ft: float, auc_base: float, paired_ci_low: float) -> bool:
    return auc_ft >= auc_base + 0.015 - EPS and paired_ci_low > 0


def f3_no_transfer_damage(auc_ub_ft: float, auc_ub_base: float) -> bool:
    return auc_ub_ft >= auc_ub_base - 0.005 - EPS


def w1_no_in_domain_loss(auc_star_l0: float, auc_ft_l0: float) -> bool:
    return auc_star_l0 >= auc_ft_l0 - 0.003 - EPS


def w2_robustness_gain(auc_star_l3: float, auc_ft_l3: float, auc_star_ub: float, auc_ft_ub: float) -> bool:
    return auc_star_l3 >= auc_ft_l3 + 0.01 - EPS or auc_star_ub >= auc_ft_ub + 0.01 - EPS


# ---- loading ----

def sweep_row(root: Path, cache: str, variant: str, c: float = 0.25) -> dict:
    for line in (root / "results.jsonl").read_text().splitlines():
        row = json.loads(line)
        if row["cache"] == cache and row["c"] == c and row["tracker"] == "botsort" and row["variant"] == variant:
            return row
    raise KeyError(f"no row for {cache} c={c} {variant} in {root}")


def cell(arm_root: Path, base_root: Path, cache: str, variant: str, reference: tuple[Path, str] | None = None) -> dict:
    """One arm in one setting: AUC (+95% CI over videos), coverage, and the paired difference to the baseline.

    reference = (root, cache) of another arm to pair with instead of the baseline (EXP-014: alpha vs fine-tuned).
    """
    row = sweep_row(arm_root, cache, variant)
    scores = load_scores(arm_root, cache, 0.25, variant)
    ref_root, ref_cache = reference if reference else (base_root, cache)
    base = load_scores(ref_root, ref_cache, 0.25, variant)
    assert scores.names == base.names, "arms must score the same videos in the same order"
    out = {"auc": row["auc"], "ci": [row["ci95_low"], row["ci95_high"]], "coverage_anomalous": row["coverage_anomalous"],
           "coverage_normal": row["coverage_normal"], "coverage_all": coverage_all_frames(scores.covered),
           "auc_check": frame_level_auc(scores.labels, scores.scores), "videos": len(scores.names)}
    out["vs"] = diff(base, scores)  # AUC(arm) - AUC(reference), paired bootstrap over videos
    return out


def available(arm: str, roots: dict[str, dict[str, Path]]) -> bool:
    return all((roots[arm][kind] / "results.jsonl").exists() for kind in ("sht", "ub"))


def build_report(alpha_star: float | None) -> dict:
    roots = arm_roots()
    report = {"alpha_star": alpha_star, "arms": {}}
    for arm in roots:
        if arm == "baseline" or not available(arm, roots):
            continue
        report["arms"][arm] = {}
        for setting, (kind, cache) in SETTINGS.items():
            report["arms"][arm][setting] = {
                v: {"vs_baseline": cell(roots[arm][kind], roots["baseline"][kind], cache, v)} for v in VARIANTS}
            if arm.startswith("a"):
                for v in VARIANTS:
                    report["arms"][arm][setting][v]["vs_ft"] = cell(
                        roots[arm][kind], roots["baseline"][kind], cache, v, reference=(roots["ft"][kind], cache))
    report["baseline"] = {}
    for setting, (kind, cache) in SETTINGS.items():
        report["baseline"][setting] = {}
        for v in VARIANTS:
            base = sweep_row(roots["baseline"][kind], cache, v)
            scores = load_scores(roots["baseline"][kind], cache, 0.25, v)
            report["baseline"][setting][v] = {
                "auc": base["auc"], "ci": [base["ci95_low"], base["ci95_high"]],
                "coverage_anomalous": base["coverage_anomalous"], "coverage_normal": base["coverage_normal"],
                "coverage_all": coverage_all_frames(scores.covered)}
    report["verdicts"] = verdicts(report, alpha_star)
    return report


def verdicts(report: dict, alpha_star: float | None) -> dict:
    out = {}
    if "ft" in report["arms"]:
        ft, base = report["arms"]["ft"], report["baseline"]
        for v in VARIANTS:
            l0, ub = ft["SHT L0"][v]["vs_baseline"], ft["UB L0"][v]["vs_baseline"]
            out[f"F1_{v}"] = f1_more_people(l0["coverage_anomalous"], base["SHT L0"][v]["coverage_anomalous"])
            out[f"F2_{v}"] = f2_higher_accuracy(l0["auc"], base["SHT L0"][v]["auc"], l0["vs"]["ci"][0])
            out[f"F3_{v}"] = f3_no_transfer_damage(ub["auc"], base["UB L0"][v]["auc"])
    key = f"a{alpha_star}"
    if alpha_star is not None and key in report["arms"] and "ft" in report["arms"]:
        star, ft = report["arms"][key], report["arms"]["ft"]
        for v in VARIANTS:
            out[f"W1_{v}"] = w1_no_in_domain_loss(star["SHT L0"][v]["vs_baseline"]["auc"], ft["SHT L0"][v]["vs_baseline"]["auc"])
            out[f"W2_{v}"] = w2_robustness_gain(
                star["SHT L3"][v]["vs_baseline"]["auc"], ft["SHT L3"][v]["vs_baseline"]["auc"],
                star["UB L0"][v]["vs_baseline"]["auc"], ft["UB L0"][v]["vs_baseline"]["auc"])
    return out


# ---- markdown ----

def pts(x: float) -> str:
    return f"{x * 100:.2f}"


def signed(x: float) -> str:
    return f"{x * 100:+.2f}"


def table(report: dict, variant: str, arms: list[str]) -> str:
    lines = [f"| arm | setting | AUC ({variant}) | 95% CI | anomalous cov | normal cov | all-frame cov | Δ AUC vs baseline (paired 95% CI) |",
             "|---|---|---|---|---|---|---|---|"]
    for setting in SETTINGS:
        b = report["baseline"][setting][variant]
        lines.append(f"| baseline | {setting} | {pts(b['auc'])} | [{pts(b['ci'][0])}, {pts(b['ci'][1])}] | "
                     f"{b['coverage_anomalous']:.3f} | {b['coverage_normal']:.3f} | {b['coverage_all']:.3f} | — |")
        for arm in arms:
            c = report["arms"][arm][setting][variant]["vs_baseline"]
            d = c["vs"]
            lines.append(f"| {arm} | {setting} | {pts(c['auc'])} | [{pts(c['ci'][0])}, {pts(c['ci'][1])}] | "
                         f"{c['coverage_anomalous']:.3f} | {c['coverage_normal']:.3f} | {c['coverage_all']:.3f} | "
                         f"{signed(d['diff'])} [{signed(d['ci'][0])}, {signed(d['ci'][1])}] |")
    return "\n".join(lines)


def table_vs_ft(report: dict, variant: str, arms: list[str]) -> str:
    lines = [f"| arm | setting | AUC ({variant}) | Δ AUC vs fine-tuned (paired 95% CI) |", "|---|---|---|---|"]
    for setting in SETTINGS:
        for arm in arms:
            c = report["arms"][arm][setting][variant]["vs_ft"]
            d = c["vs"]
            lines.append(f"| {arm} | {setting} | {pts(c['auc'])} | {signed(d['diff'])} [{signed(d['ci'][0])}, {signed(d['ci'][1])}] |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--alpha-star", type=float, default=None)
    args = parser.parse_args()
    report = build_report(args.alpha_star)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "report.json").write_text(json.dumps(report, indent=1))
    for variant in VARIANTS:
        print(table(report, variant, list(report["arms"])))
    print(json.dumps(report["verdicts"]))


if __name__ == "__main__":
    main()
