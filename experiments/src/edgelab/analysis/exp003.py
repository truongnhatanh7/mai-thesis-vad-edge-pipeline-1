"""EXP-003 analysis on saved per-frame scores (score_dir.py npz files): co-covered AUC (H2), cross-fitted gate (H3),
paired differences between arms (H4-H6, Addendum A).

    uv run --extra pose python -m edgelab.analysis.exp003 --root ~/thesis-data/exp003 --out <md file>
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from edgelab.metrics.scores import bootstrap_ci_over_videos, frame_level_auc, paired_bootstrap_diff  # noqa: F401

GATES = (0.25, 0.15, 0.10, 0.05)


@dataclass
class Scores:
    names: list[str]
    labels: list[np.ndarray]
    scores: list[np.ndarray]
    covered: list[np.ndarray]


def load_scores(root: Path, cache: str, c: float, variant: str, tracker: str = "botsort") -> Scores | None:
    path = root / "scores" / f"{cache}_c{c}_{tracker}" / f"frame_scores_{variant}.npz"
    if not path.exists():
        return None
    data = np.load(path)
    names = [str(n) for n in data["names"]]
    return Scores(names, [data[f"label_{n}"] for n in names], [data[f"score_{n}"] for n in names],
                  [data[f"covered_{n}"].astype(bool) for n in names])


def co_covered_auc(a: Scores, b: Scores) -> tuple[float, float, int]:
    """AUC of a and of b on the frames where both have >= 1 scored person (same videos, same order)."""
    assert a.names == b.names
    masks = [ca & cb for ca, cb in zip(a.covered, b.covered)]
    keep = [i for i, m in enumerate(masks) if m.any()]
    labels = [a.labels[i][masks[i]] for i in keep]
    auc_a = frame_level_auc(labels, [a.scores[i][masks[i]] for i in keep])
    auc_b = frame_level_auc(labels, [b.scores[i][masks[i]] for i in keep])
    return auc_a, auc_b, int(sum(m.sum() for m in masks))


def halves(names: list[str], seed: int = 0) -> tuple[set[str], set[str]]:
    order = np.random.default_rng(seed).permutation(sorted(names))
    return set(order[: len(order) // 2]), set(order[len(order) // 2:])


def subset_auc(s: Scores, keep: set[str]) -> float:
    idx = [i for i, n in enumerate(s.names) if n in keep]
    return frame_level_auc([s.labels[i] for i in idx], [s.scores[i] for i in idx])


def cross_fit(by_gate: dict[float, Scores], seed: int = 0) -> tuple[Scores, dict]:
    """Plan §4.4: pick c on one half of the videos, apply it to the other half, and vice versa; pool all videos."""
    names = next(iter(by_gate.values())).names
    half_a, half_b = halves(names, seed)
    chosen = {}
    pooled = {}
    for fit, apply in ((half_a, half_b), (half_b, half_a)):
        best = max(by_gate, key=lambda c: (subset_auc(by_gate[c], fit), -abs(c - 0.25)))
        chosen["fit_on_A" if fit is half_a else "fit_on_B"] = best
        for i, n in enumerate(names):
            if n in apply:
                pooled[n] = (by_gate[best].labels[i], by_gate[best].scores[i], by_gate[best].covered[i])
    return Scores(names, [pooled[n][0] for n in names], [pooled[n][1] for n in names],
                  [pooled[n][2] for n in names]), chosen


def summarize(s: Scores) -> dict:
    low, high = bootstrap_ci_over_videos(s.labels, s.scores, n_resamples=1000)
    return {"auc": frame_level_auc(s.labels, s.scores), "ci": [low, high]}


def diff(a: Scores, b: Scores, n_resamples: int = 1000, seed: int = 0) -> dict:
    """AUC(b) - AUC(a), paired bootstrap over videos (positive = b is better). Labels are the same GT in both.

    Same resampling as metrics.paired_bootstrap_diff; also returns p_le_zero = share of resamples with diff <= 0.
    """
    rng = np.random.default_rng(seed)
    n, samples = len(a.labels), []
    while len(samples) < n_resamples:
        idx = rng.integers(0, n, n)
        labels = [a.labels[i] for i in idx]
        if len(np.unique(np.concatenate(labels))) < 2:
            continue
        samples.append(frame_level_auc(labels, [b.scores[i] for i in idx]) - frame_level_auc(labels, [a.scores[i] for i in idx]))
    samples = np.array(samples)
    return {"diff": frame_level_auc(a.labels, b.scores) - frame_level_auc(a.labels, a.scores),
            "ci": [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))],
            "p_le_zero": float(np.mean(samples <= 0))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--variant", default="paper")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    caches = sorted({p.name.rsplit("_c", 1)[0] for p in (args.root / "scores").iterdir()})
    report = {}
    base = load_scores(args.root, "L0_E0", 0.25, args.variant)
    for cache in caches:
        by_gate = {c: s for c in GATES if (s := load_scores(args.root, cache, c, args.variant)) is not None}
        if 0.25 not in by_gate:
            continue
        entry = {"default": summarize(by_gate[0.25]), "vs_L0E0": diff(base, by_gate[0.25])}
        if cache.split("_")[0] != "L0":
            auc_l0, auc_lk, frames = co_covered_auc(base, by_gate[0.25])
            entry["co_covered"] = {"auc_L0": auc_l0, "auc_Lk": auc_lk, "frames": frames}
        if len(by_gate) == len(GATES):
            xfit, chosen = cross_fit(by_gate)
            entry["xfit"] = {**summarize(xfit), "chosen": chosen, "vs_default": diff(by_gate[0.25], xfit)}
        report[cache] = entry
        print(cache, json.dumps({k: (round(v["auc"] * 100, 2) if "auc" in v else v) for k, v in entry.items()
                                 if k in ("default", "xfit")}), flush=True)
    args.out.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
