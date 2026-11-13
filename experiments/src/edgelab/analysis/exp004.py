"""EXP-004 analysis: timing summary per cell, delivered AUC per (cell, light), H7 / H8 tests (plan §4, Amendment 1).

    uv run --extra pose python -m edgelab.analysis.exp004 --manifest <matrix/manifest.csv> --caches ~/thesis-data/exp003/cache \
        --out ~/thesis-data/exp004 [--timing-only]

Traces of the L3 cells are reused for L0 / L2 delivered AUC (Amendment 1) unless the L0 cost check differs by > 10%.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from edgelab.analysis.exp003 import Scores, diff
from edgelab.pipeline.deadline_replay import load_trace, run_cell
from edgelab.pipeline.deadline_sim import simulate_latest_frame

STAGES = ["read_ms", "enhance_ms", "det_pre_ms", "det_infer_ms", "det_post_ms", "track_ms", "flow_ms", "service_ms"]


def usable_traces(manifest: Path) -> dict[str, list[Path]]:
    m = pd.read_csv(manifest)
    ok = m[(m["status"] == "ok") & (~m["excluded"].astype(bool))]
    return {cell: [Path(p) for p in group["trace_file"]] for cell, group in ok.groupby("cell_id")}


def parse_cell(cell_id: str) -> dict:
    light, enhance, r, c, o, b = cell_id.split("-")
    return {"light": light, "enhance": enhance, "imgsz": int(r[1:]), "c": float(c[1:]), "o": o, "b": b}


def timing_summary(traces: dict[str, list[Path]], fps: float = 24.0) -> pd.DataFrame:
    rows = []
    for cell, files in sorted(traces.items()):
        frames = pd.concat([pd.read_csv(f) for f in files])
        per_rep_p99 = [np.percentile(pd.read_csv(f)["service_ms"], 99) for f in files]
        service = frames["service_ms"].to_numpy(float)
        kept = simulate_latest_frame(service, len(service), fps).mean()  # trace in order, one long stream
        row = {"cell": cell, **parse_cell(cell), "reps": len(files), "frames": len(service),
               "p50": np.percentile(service, 50), "p95": np.percentile(service, 95), "p99": np.percentile(service, 99),
               "p99_rep_min": min(per_rep_p99), "p99_rep_max": max(per_rep_p99), "kept_fraction": kept}
        row.update({f"mean_{s}": frames[s].mean() for s in STAGES if s in frames})
        rows.append(row)
    return pd.DataFrame(rows)


def cache_for(light: str, enhance: str, imgsz: int) -> str:
    return f"{light}_{enhance}" if imgsz == 640 else f"{light}_{enhance}-r{imgsz}"


def load_delivered(out_dir: Path) -> Scores:
    data = np.load(out_dir / "delivered_scores_live.npz")
    names = [str(n) for n in data["names"]]
    labels = [data[f"label_{n}"] for n in names]
    return Scores(names, labels, [data[f"score_{n}"] for n in names], [np.ones(len(l), bool) for l in labels])


def delivered_all(traces: dict[str, list[Path]], caches: Path, out: Path, lights=("L3", "L2", "L0")) -> pd.DataFrame:
    rows = []
    for cell, files in sorted(traces.items()):
        spec = parse_cell(cell)
        if spec["light"] != "L3" or spec["c"] != 0.25:
            continue  # cost checks only
        trace = load_trace(files)
        for light in lights:
            cache = caches / cache_for(light, spec["enhance"], spec["imgsz"])
            if not cache.exists():
                continue
            cell_out = out / "delivered" / f"{light}__{cell}"
            if not (cell_out / "result.json").exists():
                run_cell(cache, trace, 0.25, cell_out, {"label": f"{light}__{cell}", "level": light,
                                                         "enhance": spec["enhance"], "imgsz": spec["imgsz"],
                                                         "o": spec["o"], "b": spec["b"]}, n_resamples=1000, threads=2)
            result = json.loads((cell_out / "result.json").read_text())
            spec_cols = {("trace_light" if k == "light" else k): v for k, v in spec.items()}
            rows.append({"light": light, "cell": cell, **spec_cols, "delivered_auc": result["auc"],
                         "ci_low": result["ci95_low"], "ci_high": result["ci95_high"], "drop_rate": result["drop_rate"]})
    return pd.DataFrame(rows)


def interaction(out: Path, light: str, b: str, p: str = "E3", r: int = 640, tuned: str = "O1", base: str = "O0") -> dict:
    """H8: [AUC(P,tuned) - AUC(E0,tuned)] - [AUC(P,base) - AUC(E0,base)], paired bootstrap over videos.
    H8' (Amendment 2): tuned = O1n, base = O0r at B2."""
    get = lambda enh, o: load_delivered(out / "delivered" / f"{light}__L3-{enh}-R{r}-c0.25-{o}-{b}")  # noqa: E731
    e0_o0, p_o0, e0_o1, p_o1 = get("E0", base), get(p, base), get("E0", tuned), get(p, tuned)
    # interaction as a difference of two paired differences: build "score arms" whose AUC difference is the contrast
    d_o1, d_o0 = diff(e0_o1, p_o1), diff(e0_o0, p_o0)
    rng = np.random.default_rng(0)
    from edgelab.metrics.scores import frame_level_auc
    n, samples = len(e0_o0.labels), []
    while len(samples) < 1000:
        idx = rng.integers(0, n, n)
        labels = [e0_o0.labels[i] for i in idx]
        if len(np.unique(np.concatenate(labels))) < 2:
            continue
        auc = lambda s: frame_level_auc(labels, [s.scores[i] for i in idx])  # noqa: E731
        samples.append((auc(p_o1) - auc(e0_o1)) - (auc(p_o0) - auc(e0_o0)))
    return {"light": light, "b": b, "p": p, "tuned": tuned, "base": base, "effect_O1": d_o1["diff"], "effect_O0": d_o0["diff"],
            "interaction": d_o1["diff"] - d_o0["diff"],
            "ci": [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--caches", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--timing-only", action="store_true")
    parser.add_argument("--lights", nargs="+", default=["L3", "L2", "L0"], help="delivered replays for these caches")
    parser.add_argument("--no-tests", action="store_true", help="replays only (run the tests once all lights exist)")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    traces = usable_traces(args.manifest)
    timing = timing_summary(traces)
    timing.to_csv(args.out / "timing_summary.csv", index=False)
    print(timing[["cell", "reps", "p50", "p99", "kept_fraction"]].to_string(index=False), flush=True)
    if args.timing_only:
        return
    delivered = delivered_all(traces, args.caches, args.out, tuple(args.lights))
    delivered.to_csv(args.out / f"delivered_summary_{'_'.join(args.lights)}.csv", index=False)
    if args.no_tests:
        return
    tests = [interaction(args.out, light, b, p) for light in ("L3", "L2") for b in ("B2", "B4") for p in ("E1", "E2", "E3")]
    if (args.out / "delivered" / "L3__L3-E3-R640-c0.25-O1n-B2").exists():  # Amendment 2 (H8')
        tests += [interaction(args.out, light, "B2", "E3", tuned="O1n", base="O0r") for light in ("L3", "L2")]
        tests += [interaction(args.out, light, "B2", "E2", tuned="O1n", base="O0r") for light in ("L3", "L2")]
    (args.out / "interaction_tests.json").write_text(json.dumps(tests, indent=1))
    for t in tests:
        print(json.dumps({k: (round(v * 100, 2) if isinstance(v, float) else v) for k, v in t.items() if k != "ci"},
                         ) + f"  CI [{t['ci'][0] * 100:+.2f}, {t['ci'][1] * 100:+.2f}]", flush=True)


if __name__ == "__main__":
    main()
