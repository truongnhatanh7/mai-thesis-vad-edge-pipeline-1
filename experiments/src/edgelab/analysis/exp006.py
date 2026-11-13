"""EXP-006 deadline replay (plan §3 U5): delivered AUC of UBnormal caches under the EXP-004 VM timing traces.

    uv run --extra pose python -m edgelab.analysis.exp006 --manifest ~/thesis-data/vm-exchange/results/matrix/manifest.csv \
        --caches ~/thesis-data/exp006/cache --out ~/thesis-data/exp006/deadline [--cells L3-E0-R640-c0.25-O0-B4 ...]

Reuses EXP-004's trace loading and `run_cell`; only the caches (L0 light, UBnormal videos) and the dataset differ.
A cell id names the VM trace (light L3 = the trace's light level); the cache is chosen from its enhancement and
resolution: R640 -> L0_E<x>, R480 -> L0-480_E<x>, R960 -> L0-960_E<x>.
Each cell writes <out>/delivered/L0__<cell>/result.json; finished cells are skipped (resumable).

    ... --zero-drop-check --videos <v1> <v2> ...   # a trace that never drops a frame must give the offline live AUC
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from edgelab.analysis.exp004 import parse_cell, usable_traces
from edgelab.pipeline.deadline_replay import load_trace, run_cell

LIGHT = "L0"  # UBnormal caches are not darkened


def cache_name(enhance: str, imgsz: int) -> str:
    return f"{LIGHT}_{enhance}" if imgsz == 640 else f"{LIGHT}-{imgsz}_{enhance}"


def wanted(spec: dict) -> bool:
    """Default cells: the O0-B4 cells (U5) and the O1n-B2 / O0r-B2 cells at c 0.25."""
    return spec["c"] == 0.25 and ((spec["o"] == "O0" and spec["b"] == "B4") or
                                  (spec["o"] in ("O1n", "O0r") and spec["b"] == "B2"))


def delivered_ubnormal(traces: dict[str, list[Path]], caches: Path, out: Path, cells: list[str] | None = None,
                       videos: list[str] | None = None, n_resamples: int = 1000, threads: int | None = 2) -> pd.DataFrame:
    rows = []
    for cell in cells or sorted(traces):
        spec = parse_cell(cell)
        if cells is None and not wanted(spec):
            continue
        cache = caches / cache_name(spec["enhance"], spec["imgsz"])
        if cell not in traces or not cache.exists():
            print(f"skip {cell}: " + ("no usable trace" if cell not in traces else f"no cache {cache.name}"), flush=True)
            continue
        cell_out = out / "delivered" / f"{LIGHT}__{cell}"
        if not (cell_out / "result.json").exists():
            config = {"label": f"{LIGHT}__{cell}", "level": LIGHT, "enhance": spec["enhance"], "imgsz": spec["imgsz"],
                      "o": spec["o"], "b": spec["b"], "trace_cell": cell}
            run_cell(cache, load_trace(traces[cell]), spec["c"], cell_out, config, n_resamples=n_resamples,
                     threads=threads, dataset="UBnormal", videos=videos)
        result = json.loads((cell_out / "result.json").read_text())
        rows.append({"cell": cell, **spec, "cache": cache.name, "delivered_auc": result["auc"],
                     "ci_low": result["ci95_low"], "ci_high": result["ci95_high"], "drop_rate": result["drop_rate"],
                     "videos": result["num_videos"]})
    return pd.DataFrame(rows)


def zero_drop_check(cache: Path, videos: list[str], workdir: Path, c: float = 0.25, threads: int | None = 2,
                    dataset: str = "UBnormal", gt_dir: Path | None = None) -> dict:
    """Delivered AUC with a trace that never misses a deadline vs the offline live AUC of the same cache and videos.

    Same replay (c), same STG-NF, same live variant, so the two numbers must be equal. MLflow: 'EXP-006-check'.
    """
    from edgelab.pose.replay import replay_dir
    from edgelab.pose.score_dir import score_pose_dir

    workdir.mkdir(parents=True, exist_ok=True)
    tracking_uri = f"sqlite:///{workdir / 'mlflow_check.db'}"
    delivered = run_cell(cache, np.full(48, 1.0), c, workdir / "delivered", {"label": "zero-drop"}, n_resamples=50,
                         threads=threads, tracking_uri=tracking_uri, dataset=dataset, videos=videos,
                         experiment="EXP-006-check", gt_dir=gt_dir)
    replay_dir(cache, c, workdir / "poses", videos=videos, dataset=dataset)
    offline = score_pose_dir(workdir / "poses", workdir / "scores", {"level": LIGHT, "enhance": "check", "c": c},
                             threads=threads, n_resamples=50, experiment="EXP-006-check", tracking_uri=tracking_uri,
                             dataset=dataset, clips=videos, gt_dir=gt_dir)
    return {"delivered_auc": delivered["auc"], "drop_rate": delivered["drop_rate"], "offline_live_auc": offline["live"]["auc"],
            "difference": delivered["auc"] - offline["live"]["auc"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--caches", type=Path, help="folder with the UBnormal caches L0_E0, L0-960_E0, ...")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cells", nargs="*", default=None, help="VM trace cell ids (default: O0-B4, O1n-B2, O0r-B2 cells)")
    parser.add_argument("--videos", nargs="*", default=None, help="only these videos (checks and smoke runs)")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--n-resamples", type=int, default=1000)
    parser.add_argument("--zero-drop-check", action="store_true", help="needs --caches <one cache dir> --videos ...")
    args = parser.parse_args()
    if args.zero_drop_check:
        print(json.dumps(zero_drop_check(args.caches, args.videos, args.out, threads=args.threads), indent=1))
        return
    summary = delivered_ubnormal(usable_traces(args.manifest), args.caches, args.out, args.cells, args.videos,
                                 args.n_resamples, args.threads)
    args.out.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.out / "delivered_summary.csv", index=False)
    print(summary[["cell", "cache", "delivered_auc", "ci_low", "ci_high", "drop_rate", "videos"]].to_string(index=False))


if __name__ == "__main__":
    main()
