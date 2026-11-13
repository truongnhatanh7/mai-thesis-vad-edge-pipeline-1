"""EXP-003 gate sweep: every finished detection cache x gate c -> replay -> STG-NF -> one JSONL row per variant.

    uv run --extra pose python -m edgelab.pose.gate_sweep --root ~/thesis-data/exp003 [--wait]
    uv run --extra pose python -m edgelab.pose.gate_sweep --root ~/thesis-data/exp006 --dataset UBnormal --experiment EXP-006 [--wait]

Resumable: a (cache, c, tracker) already in results.jsonl is skipped. With --wait it keeps polling for new caches
until cache/jobs_done.txt says SWEEP_DONE.
"""

import argparse
import json
import time
from pathlib import Path

from edgelab.pose.replay import replay_dir
from edgelab.pose.score_dir import score_pose_dir

GATES = (0.25, 0.15, 0.10, 0.05)  # plan §4.4


def finished_caches(root: Path) -> list[str]:
    done = root / "cache" / "jobs_done.txt"
    if not done.exists():
        return []
    names = []
    for line in done.read_text().splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[2] == "exit=0":
            names.append(f"{parts[0]}_{parts[1]}")
    return names


def done_keys(results: Path) -> set[tuple]:
    if not results.exists():
        return set()
    rows = [json.loads(line) for line in results.read_text().splitlines() if line.strip()]
    return {(r["cache"], r["c"], r["tracker"]) for r in rows}


def run_one(root: Path, cache: str, c: float, tracker: str, threads: int, dataset: str = "ShanghaiTech",
            gt_dir: Path | None = None, experiment: str = "EXP-003") -> list[dict]:
    level, enhance = cache.rsplit("_", 1)
    tag = f"{cache}_c{c}_{tracker}"
    stats = replay_dir(root / "cache" / cache, c, root / "poses" / tag, tracker, dataset=dataset)
    replay_totals = {k: sum(s[k] for s in stats.values()) for k in next(iter(stats.values()))}
    config = {"level": level, "enhance": enhance, "c": c, "tracker": tracker, "device": "mps-cache"}
    base = {"cache": cache, "level": level, "enhance": enhance, "c": c, "tracker": tracker, **replay_totals}
    if dataset != "ShanghaiTech":
        base["dataset"] = dataset  # ShanghaiTech rows keep their EXP-003 format
    try:
        results = score_pose_dir(root / "poses" / tag, root / "scores" / tag, config, threads=threads,
                                 gt_dir=gt_dir, experiment=experiment, dataset=dataset)
    except ValueError as error:
        # pre-registered (log 2026-11-04 23:37): constant output -> AUC 0.5, coverage 0, kept in every table
        return [{**base, "variant": v, "auc": 0.5, "ci95_low": None, "ci95_high": None, "coverage_anomalous": 0.0,
                 "coverage_normal": 0.0, "num_segments": 0, "no_segments": True, "error": str(error)}
                for v in ("paper", "live")]
    return [{**base, "variant": v, **r, "no_segments": False} for v, r in results.items()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--tracker", default="botsort")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--dataset", default="ShanghaiTech", choices=["ShanghaiTech", "UBnormal"])
    parser.add_argument("--gt-dir", type=Path, default=None, help="default: STG-NF's GT folder of the dataset")
    parser.add_argument("--experiment", default="EXP-003", help="MLflow experiment name (EXP-006 for UBnormal)")
    parser.add_argument("--gates", type=float, nargs="*", default=list(GATES), help="gate thresholds c (default: all four)")
    args = parser.parse_args()
    results = args.root / "results.jsonl"
    while True:
        todo = [(cache, c) for cache in finished_caches(args.root) for c in args.gates
                if (cache, c, args.tracker) not in done_keys(results)]
        for cache, c in todo:
            rows = run_one(args.root, cache, c, args.tracker, args.threads, args.dataset, args.gt_dir, args.experiment)
            with open(results, "a") as f:
                for row in rows:
                    f.write(json.dumps(row) + "\n")
            print(json.dumps({k: rows[0][k] for k in ("cache", "c")} | {r["variant"]: round(r["auc"], 4) for r in rows}),
                  flush=True)
        sweep_done = "SWEEP_DONE" in (args.root / "cache" / "jobs_done.txt").read_text()
        if not args.wait or (sweep_done and not todo):
            break
        time.sleep(60)


if __name__ == "__main__":
    main()
