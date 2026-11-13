"""EXP-005 analysis: label-free coverage selectors S and S24 vs oracle and fixed policies (plan §3-4).

    uv run --extra pose python -m edgelab.analysis.exp005 --root ~/thesis-data/exp003 --out <json> [--delivered]
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from edgelab.analysis.exp003 import load_scores
from edgelab.metrics.scores import frame_level_auc

LEVELS = ("X1", "X2", "X3")
CANDIDATES = ("E0", "E1", "E2")
MIN_RUN = 48  # S24: a covering track needs >= 48 consecutive tracked frames (plan §3)


def coverage_all_frames(covered: list[np.ndarray]) -> float:
    """Label-free: share of all test frames with >= 1 scored person."""
    return float(sum(c.sum() for c in covered) / sum(c.size for c in covered))


def long_run_frames(track_frames: list[int], min_run: int = MIN_RUN) -> set[int]:
    """Frames that lie inside a run of >= min_run consecutive frame ids of one track."""
    frames, keep, start = sorted(track_frames), set(), 0
    for i in range(1, len(frames) + 1):
        if i == len(frames) or frames[i] != frames[i - 1] + 1:
            if i - start >= min_run:
                keep.update(frames[start:i])
            start = i
    return keep


def coverage_long_tracks(pose_dir: Path, frame_counts: dict[str, int], min_run: int = MIN_RUN) -> float:
    covered = total = 0
    for video, n in frame_counts.items():
        files = list(pose_dir.glob(f"{video}_*_tracked_person.json"))
        frames = set()
        if files:
            for track in json.loads(files[0].read_text()).values():
                frames |= long_run_frames([int(k) for k in track], min_run)
        covered += len({f for f in frames if f < n})
        total += n
    return covered / total


def select(values: dict[str, float]) -> str:
    best = max(values.values())
    return "E0" if values.get("E0") == best else max(values, key=values.get)  # ties -> E0


def analyse(root: Path, variant: str) -> dict:
    report, all_cov, all_auc = {}, [], []
    for level in LEVELS:
        auc, cov, cov24 = {}, {}, {}
        for enh in CANDIDATES:
            s = load_scores(root, f"{level}_{enh}", 0.25, variant)
            auc[enh] = frame_level_auc(s.labels, s.scores)
            cov[enh] = coverage_all_frames(s.covered)
            counts = {n: len(l) for n, l in zip(s.names, s.labels)}
            cov24[enh] = coverage_long_tracks(root / "poses" / f"{level}_{enh}_c0.25_botsort", counts)
            all_cov.append(cov[enh])
            all_auc.append(auc[enh])
        oracle = max(auc, key=auc.get)
        pick, pick24 = select(cov), select(cov24)
        report[level] = {"auc": auc, "coverage": cov, "coverage_long": cov24, "oracle": oracle, "S": pick, "S24": pick24,
                         "regret_S": auc[oracle] - auc[pick], "regret_S24": auc[oracle] - auc[pick24]}
    means = {f"fixed_{e}": float(np.mean([report[l]["auc"][e] for l in LEVELS])) for e in CANDIDATES}
    means["S"] = float(np.mean([report[l]["auc"][report[l]["S"]] for l in LEVELS]))
    means["S24"] = float(np.mean([report[l]["auc"][report[l]["S24"]] for l in LEVELS]))
    means["oracle"] = float(np.mean([report[l]["auc"][report[l]["oracle"]] for l in LEVELS]))
    verdicts = {
        "F1": sum(report[l]["regret_S"] <= 0.005 for l in LEVELS) >= 2,
        "F2": means["S"] >= max(means[f"fixed_{e}"] for e in CANDIDATES),
        "F3": float(spearmanr(all_cov, all_auc)[0]),
        "F4_regret_S_vs_S24_at_X3": [report["X3"]["regret_S"], report["X3"]["regret_S24"]],
    }
    return {"variant": variant, "levels": report, "means": means, "verdicts": verdicts}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = {v: analyse(args.root, v) for v in ("paper", "live")}
    args.out.write_text(json.dumps(result, indent=1))
    for v, r in result.items():
        print(v, json.dumps({l: {k: r["levels"][l][k] for k in ("oracle", "S", "S24")} for l in LEVELS}))
        print(v, "means", {k: round(x * 100, 2) for k, x in r["means"].items()}, "verdicts", r["verdicts"])


if __name__ == "__main__":
    main()
