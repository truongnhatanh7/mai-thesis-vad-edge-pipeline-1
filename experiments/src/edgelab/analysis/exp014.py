"""EXP-014: choose alpha* with no test data, from the label-free coverage of the held-out TRAINING videos.

    uv run --extra pose python -m edgelab.analysis.exp014 --dir ~/thesis-data/exp014-heldout --out <json>

`--dir` holds `a<alpha>.json` files written by edgelab.pose.segment_coverage (alpha 0 = original, 1 = fine-tuned).
Only the three interior alphas can be chosen; the two ends are reported for context. Rule: the highest all-frame
coverage of the paper variant; ties go to the alpha closest to 0.5, then to the smaller one.
"""

import argparse
import json
from pathlib import Path

CANDIDATES = (0.25, 0.5, 0.75)


def select_alpha(coverage: dict[float, float], candidates: tuple[float, ...] = CANDIDATES) -> float:
    return min(candidates, key=lambda a: (-coverage[a], abs(a - 0.5), a))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    results = {float(p.stem[1:]): json.loads(p.read_text()) for p in sorted(args.dir.glob("a*.json"))}
    paper = {a: r["coverage_paper"] for a, r in results.items()}
    chosen = select_alpha(paper)
    summary = {"alpha_star": chosen, "coverage_paper": paper, "coverage_live": {a: r["coverage_live"] for a, r in results.items()},
               "frames": {a: r["frames"] for a, r in results.items()}}
    args.out.write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
