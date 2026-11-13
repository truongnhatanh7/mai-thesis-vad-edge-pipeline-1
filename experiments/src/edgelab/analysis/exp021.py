"""EXP-021: analytic upper bound on frame-level ROC-AUC from person coverage (plan frozen 2026-11-08 17:15).

    uv run --extra pose python -m edgelab.analysis.exp021 [--out-dir <dir>]      # reads the existing results.jsonl files, seconds

Frames with no scored person get the most-normal score. If every covered frame were scored perfectly, AUC could not exceed
    bound_half = cA + (1 - cA)(1 - cN) / 2       (ties count 1/2, the sklearn convention)
    bound_pess = cA                              (ties count 0, "pessimistic")
    bound_loose = (1 + cA) / 2                   (covered normal frames may sit exactly on the floor and tie; looser still)
The derivation is in EXP-021-coverage-bound/derivation.md.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

DATA = Path.home() / "thesis-data"
EXPERIMENT_DIR = Path(__file__).resolve().parents[3] / "EXP-021-coverage-bound"

# (source name, results file, dataset, key field of one configuration)
SOURCES = [
    ("exp003", "exp003/results.jsonl", "ShanghaiTech", "cache"),
    ("exp006", "exp006/results.jsonl", "UBnormal", "cache"),
    ("exp007-sht", "exp007-sht/results.jsonl", "ShanghaiTech", "cache"),
    ("exp007-ub", "exp007-ub/results.jsonl", "UBnormal", "cache"),
    ("exp012", "exp012/results.jsonl", "ShanghaiTech", "cache"),
    ("exp013-sht", "exp013-sht/results.jsonl", "ShanghaiTech", "cache"),
    ("exp013-ub", "exp013-ub/results.jsonl", "UBnormal", "cache"),
    ("exp016", "exp016/rows.jsonl", "ShanghaiTech", "config"),
    ("exp017-sht", "exp017-sht/results.jsonl", "ShanghaiTech", "cache"),
    ("exp017-ub", "exp017-ub/results.jsonl", "UBnormal", "cache"),
]
CORE_VARIANT = "paper"


def bound_half(ca, cn):
    """Best-case AUC, ties = 1/2: cA + (1 - cA)(1 - cN) / 2."""
    ca, cn = np.asarray(ca, dtype=float), np.asarray(cn, dtype=float)
    return ca + (1 - ca) * (1 - cn) / 2


def bound_pess(ca, cn):
    """Best-case AUC when a tie counts as a loss: cA (does not depend on cN)."""
    return np.asarray(ca, dtype=float) + 0 * np.asarray(cn, dtype=float)


def bound_loose(ca, cn):
    """Largest AUC any scoring can reach with ties = 1/2: (1 + cA) / 2 (covered normals tied with the floor)."""
    return (1 + np.asarray(ca, dtype=float)) / 2 + 0 * np.asarray(cn, dtype=float)


def valid_rows(rows: list[dict], key: str, source: str, c: float | None = 0.25) -> list[dict]:
    """Rows of one results file that are real measurements.

    Dropped: other gate values (c != 0.25, when the row has a c), `no_segments` placeholder rows (AUC 0.5 by construction,
    superseded by a later row of the same configuration where one exists), rows flagged `invalid` (incomplete caches, EXP-016),
    and exact re-runs (same configuration + variant: the last row wins).
    """
    latest = {}
    for row in rows:
        if "c" in row and c is not None and row["c"] != c:
            continue
        if row.get("no_segments") or row.get("invalid"):
            continue
        latest[(row[key], row["variant"])] = row
    return [{"source": source, "config": cfg, "variant": variant, "auc": row["auc"], "ca": row["coverage_anomalous"],
             "cn": row["coverage_normal"], "num_segments": row.get("num_segments"), "scores_file": row.get("scores_file")} for (cfg, variant), row in latest.items()]


def effective_coverage(scores_file: str | Path) -> tuple[float, float] | None:
    """Post-hoc (not in the frozen plan): coverage measured on the FINAL scores = share of anomalous / normal frames whose
    score is above the file's minimum (the most-normal floor). The temporal smoother spreads covered scores onto
    neighbouring uncovered frames, so this is larger than the logged coverage. None if the file is missing."""
    path = Path(scores_file) if scores_file else None
    if path is None or not path.exists():
        return None
    d = np.load(path)
    names = list(d["names"])
    scores = np.concatenate([d[f"score_{n}"] for n in names])
    labels = np.concatenate([d[f"label_{n}"] for n in names])
    above = scores > scores.min()
    return float(above[labels == 1].mean()), float(above[labels == 0].mean())


def collect(data: Path = DATA) -> list[dict]:
    points = []
    for source, path, dataset, key in SOURCES:
        rows = [json.loads(line) for line in (data / path).read_text().splitlines() if line.strip()]
        for point in valid_rows(rows, key, source):
            points.append({**point, "dataset": dataset})
    return points


def evaluate(points: list[dict], effective: bool = False) -> dict:
    """effective=True: use the post-hoc effective coverage (keys eca, ecn) instead of the logged one."""
    ca = np.array([p["eca" if effective else "ca"] for p in points])
    cn = np.array([p["ecn" if effective else "cn"] for p in points])
    auc = np.array([p["auc"] for p in points])
    bounds = {"half": bound_half(ca, cn), "pess": bound_pess(ca, cn), "loose": bound_loose(ca, cn)}
    out = {}
    for name, b in bounds.items():
        violations = [(p["source"], p["config"], p["variant"], float(p["auc"]), float(x)) for p, x in zip(points, b) if p["auc"] > x + 1e-12]
        out[name] = {"n": len(points), "violations": violations, "max_excess": float(max(0.0, (auc - b).max())),
                     "min_slack": float((b - auc).min()), "median_slack": float(np.median(b - auc))}
        for dataset in sorted({p["dataset"] for p in points}):
            for variant in (None, "paper", "live"):
                mask = np.array([p["dataset"] == dataset and (variant is None or p["variant"] == variant) for p in points])
                rho = float(spearmanr(b[mask], auc[mask])[0]) if mask.sum() > 2 else float("nan")
                out[name][f"spearman_{dataset}_{variant or 'both'}"] = rho
                out[name][f"n_{dataset}_{variant or 'both'}"] = int(mask.sum())
    return out


def plot(points: list[dict], path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, name, fn in zip(axes, ("half", "pess"), (bound_half, bound_pess)):
        for dataset, colour in (("ShanghaiTech", "tab:blue"), ("UBnormal", "tab:orange")):
            sel = [p for p in points if p["dataset"] == dataset]
            ax.scatter(fn([p["ca"] for p in sel], [p["cn"] for p in sel]), [p["auc"] for p in sel], s=14, alpha=0.7, c=colour,
                       label=f"{dataset} (n={len(sel)})")
        ax.plot([0.4, 1], [0.4, 1], "k--", lw=1, label="AUC = bound")
        ax.set_xlabel({"half": "bound, ties = 1/2:  cA + (1-cA)(1-cN)/2", "pess": "bound, ties = 0:  cA"}[name])
        ax.set_ylabel("measured frame AUC")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=130)


def markdown(points: list[dict], result: dict, eff_points: list[dict], eff_result: dict) -> str:
    h, p, lo = result["half"], result["pess"], result["loose"]
    n_cfg = len({(q["source"], q["config"]) for q in points})
    lines = ["# EXP-021 results: coverage upper bound on frame AUC (generated by `edgelab.analysis.exp021`)\n",
             f"{len(points)} points = {n_cfg} configurations x variants (paper, live) from the existing results files "
             "(c = 0.25 rows; `no_segments` placeholders, the incomplete-cache EXP-016 rows and re-run duplicates dropped; "
             "see `valid_rows`). Derivation: `derivation.md`.\n",
             "Bounds (cA, cN = coverage of anomalous / normal frames, as logged in each row):\n",
             "- `half` (primary, ties count 1/2 as in sklearn): cA + (1 - cA)(1 - cN)/2",
             "- `pess` (ties count 0): cA",
             "- `loose` (ties 1/2, covered normals may sit on the floor): (1 + cA)/2\n",
             "## Verdicts (frozen rules, primary bound `half`, logged coverage)\n", "| # | rule | numbers | verdict |", "|---|---|---|---|",
             f"| B1 | AUC <= bound for every configuration | {len(h['violations'])} of {h['n']} points violate (max excess {h['max_excess']:.4f}); "
             f"`loose` {len(lo['violations'])} violations | {'PASS' if not h['violations'] else 'FAIL'} |",
             f"| B2 | Spearman(bound, AUC) >= 0.8 per dataset | ShanghaiTech {h['spearman_ShanghaiTech_both']:.3f} (n={h['n_ShanghaiTech_both']}), "
             f"UBnormal {h['spearman_UBnormal_both']:.3f} (n={h['n_UBnormal_both']}); per variant min "
             f"{min(h[f'spearman_{d}_{v}'] for d in ('ShanghaiTech', 'UBnormal') for v in ('paper', 'live')):.3f} | "
             f"{'PASS' if min(h['spearman_ShanghaiTech_both'], h['spearman_UBnormal_both']) >= 0.8 else 'FAIL'} |\n",
             "## B1: measured AUC <= bound everywhere\n", "| bound | points | violations | max excess (AUC - bound) | min slack | median slack |", "|---|---|---|---|---|---|"]
    for name, r in (("half", h), ("pess", p), ("loose", lo)):
        lines.append(f"| {name} | {r['n']} | {len(r['violations'])} | {r['max_excess']:.4f} | {r['min_slack']:+.4f} | {r['median_slack']:+.4f} |")
    for name, r in (("half", h), ("pess", p)):
        if r["violations"]:
            lines += [f"\nViolations of `{name}` (source, config, variant, AUC, bound):\n", "| source | config | variant | AUC | bound |", "|---|---|---|---|---|"]
            lines += [f"| {s} | {c} | {v} | {a:.4f} | {b:.4f} |" for s, c, v, a, b in r["violations"]]
    lines += ["\n## B2: Spearman(bound, AUC) per dataset (pass: >= 0.8)\n", "| bound | dataset | both variants (n) | paper | live |", "|---|---|---|---|---|"]
    for name, r in (("half", h), ("pess", p), ("loose", lo)):
        for dataset in ("ShanghaiTech", "UBnormal"):
            lines.append(f"| {name} | {dataset} | {r[f'spearman_{dataset}_both']:.3f} ({r[f'n_{dataset}_both']}) | "
                         f"{r[f'spearman_{dataset}_paper']:.3f} | {r[f'spearman_{dataset}_live']:.3f} |")
    lines += ["\n## Post-hoc (not in the frozen plan): why `half` is violated, and the bound on effective coverage\n",
              "The logged coverage counts frames that had a scored person *before* the temporal smoother. The smoother (Gaussian, "
              "applied after uncovered frames are set to the floor) spreads covered scores onto neighbouring uncovered frames, so those "
              "frames are no longer tied at the floor. Effective coverage = share of anomalous / normal frames whose final score is above "
              f"the file's minimum, read from each row's `scores_file` ({len(eff_points)} of {len(points)} points have the file). "
              "On effective coverage the bound's premise holds exactly, so the bound is a theorem there.\n",
              "| bound on effective coverage | points | violations | max excess | median slack | Spearman SHT | Spearman UB |", "|---|---|---|---|---|---|---|"]
    for name in ("half", "pess", "loose"):
        r = eff_result[name]
        lines.append(f"| {name} | {r['n']} | {len(r['violations'])} | {r['max_excess']:.4f} | {r['median_slack']:+.4f} | "
                     f"{r['spearman_ShanghaiTech_both']:.3f} ({r['n_ShanghaiTech_both']}) | {r['spearman_UBnormal_both']:.3f} ({r['n_UBnormal_both']}) |")
    ratios = [(p["eca"] - p["ca"]) for p in eff_points]
    lines.append(f"\nMedian (effective cA - logged cA) = {np.median(ratios):+.3f}; mean = {np.mean(ratios):+.3f}; max = {max(ratios):+.3f}.\n")
    lines += ["`pess` on effective coverage is not a bound for tie-as-half AUC (it counts ties as losses); shown for completeness.",
              "\n`scatter_bound_vs_auc.png`: left, `half` bound vs AUC; right, `pess` bound vs AUC; dashed line AUC = bound.\n",
              "## All points\n", "| dataset | source | config | variant | cA | cN | AUC | half | pess | slack(half) |", "|---|---|---|---|---|---|---|---|---|---|"]
    for q in sorted(points, key=lambda q: (q["dataset"], q["source"], q["config"], q["variant"])):
        bh, bp = float(bound_half(q["ca"], q["cn"])), float(bound_pess(q["ca"], q["cn"]))
        lines.append(f"| {q['dataset']} | {q['source']} | {q['config']} | {q['variant']} | {q['ca']:.3f} | {q['cn']:.3f} | "
                     f"{q['auc']:.4f} | {bh:.4f} | {bp:.4f} | {bh - q['auc']:+.4f} |")
    return "\n".join(lines) + "\n"


def add_effective(points: list[dict]) -> list[dict]:
    """Points that have a scores file, with eca / ecn added."""
    out = []
    for p in points:
        eff = effective_coverage(p["scores_file"])
        if eff is not None:
            out.append({**p, "eca": eff[0], "ecn": eff[1]})
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=EXPERIMENT_DIR)
    args = parser.parse_args()
    points = collect()
    result = evaluate(points)
    eff_points = add_effective(points)
    eff_result = evaluate(eff_points, effective=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "points.json").write_text(json.dumps({"points": points, "evaluation": result, "effective_points": eff_points,
                                                          "effective_evaluation": eff_result}, indent=1))
    (args.out_dir / "results.md").write_text(markdown(points, result, eff_points, eff_result))
    plot(points, args.out_dir / "scatter_bound_vs_auc.png")
    for name in ("half", "pess", "loose"):
        r = eff_result[name]
        print("effective", name, "violations", len(r["violations"]), {k: round(v, 3) for k, v in r.items() if k.startswith("spearman") and "both" in k})
    for name in ("half", "pess", "loose"):
        r = result[name]
        print(name, "violations", len(r["violations"]), "max_excess", round(r["max_excess"], 4),
              {k: round(v, 3) for k, v in r.items() if k.startswith("spearman") and "both" in k})
    print(len(points), "points")


if __name__ == "__main__":
    main()
