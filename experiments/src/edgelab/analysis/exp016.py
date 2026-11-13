"""EXP-016 driver: top-down configurations -> poses -> frozen STG-NF -> rows; then the results table and verdicts T1-T3.

    uv run --extra pose python -m edgelab.analysis.exp016 run --pose-model rtmpose-s [--levels L0 L2]   # resumable
    uv run --extra pose python -m edgelab.analysis.exp016 report --pose-model rtmpose-s

Config = level (L0 clean, L2 = EXP-003 darkening, no enhancer) x detector input (640, 960) x keyframe step K (1, 3).
Detector caches (person boxes): ~/thesis-data/exp016/cache/<level>_n<imgsz>/ (made by `edgelab.pose.topdown cache`).
Rows: ~/thesis-data/exp016/rows.jsonl ; poses and scores: .../poses/<config>/, .../scores/<config>/.
Baseline (single-stage YOLOv8n-pose at 640, BoT-SORT, c 0.25, same scorer): EXP-003 rows L0_E0 / L2_E0.
"""

import argparse
import json
import time
from pathlib import Path

from edgelab.analysis.exp003 import diff
from edgelab.analysis.exp009 import load_scores_file
from edgelab.pose.score_dir import score_pose_dir
from edgelab.pose.topdown import EXPECTED_VIDEOS, IncompleteCacheError, check_cache_complete, poses_for_dir

DATA = Path.home() / "thesis-data/exp016"
EXP003 = Path.home() / "thesis-data/exp003"
FRAMES_ROOT = Path.home() / "thesis-data/shanghaitech/shanghaitech/testing/frames"
EXPERIMENT_DIR = Path(__file__).resolve().parents[3] / "EXP-016-topdown-detector-sweep"
ROWS = DATA / "rows.jsonl"
SIZES, KS, C = (640, 960), (1, 3), 0.25
BASELINE_PAPER_L0 = 68.53  # plan.md: YOLOv8n-pose 640 baseline
T1_MARGIN, T2_MARGIN = 2.0, 1.0


def config_id(level: str, imgsz: int, k: int, pose_model: str) -> str:
    return f"{level}_n{imgsz}_K{k}_{pose_model}"


def read_rows(include_invalid: bool = False) -> list[dict]:
    """Scored rows. Rows labelled `"invalid"` (incomplete MPS-crashed caches, deviation 2026-11-08) are kept in the file for
    the record but are never used for scoring decisions, verdicts or the main table unless include_invalid is set."""
    rows = [json.loads(line) for line in ROWS.read_text().splitlines() if line.strip()] if ROWS.exists() else []
    return rows if include_invalid else [r for r in rows if not r.get("invalid")]


def run_config(level: str, imgsz: int, k: int, pose_model: str, threads: int) -> list[dict]:
    cid = config_id(level, imgsz, k, pose_model)
    if any(r["config"] == cid for r in read_rows()):
        return []
    cache = DATA / "cache" / f"{level}_n{imgsz}"
    check_cache_complete(cache)  # raises IncompleteCacheError: an incomplete cache is never posed or scored
    t0 = time.perf_counter()
    pose_dir = DATA / "poses" / cid
    stats = poses_for_dir(cache, FRAMES_ROOT, level, pose_model, k, C, pose_dir, threads)
    t1 = time.perf_counter()
    n_json = len(list(pose_dir.glob("*_tracked_person.json")))
    if n_json != EXPECTED_VIDEOS:
        raise IncompleteCacheError(f"{pose_dir}: {n_json} pose files, {EXPECTED_VIDEOS} expected")
    totals = {key: sum(s[key] for s in stats.values()) for key in next(iter(stats.values()))}
    results = score_pose_dir(pose_dir, DATA / "scores" / cid,
                             {"level": level, "enhance": f"n{imgsz}-K{k}-{pose_model}", "c": C, "tracker": "botsort-topdown",
                              "device": "cpu"}, threads=threads, experiment="EXP-016")
    base = {"config": cid, "level": level, "imgsz": imgsz, "k": k, "pose_model": pose_model, **totals,
            "poses_wall_s": round(t1 - t0, 1)}
    rows = [{**base, "variant": v, **r} for v, r in results.items()]
    with open(ROWS, "a") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return rows


def baseline_row(level: str, variant: str) -> dict:
    for line in (EXP003 / "results.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r["cache"] == f"{level}_E0" and r["c"] == C and r["tracker"] == "botsort" and r["variant"] == variant:
            return r
    raise KeyError((level, variant))


def get(rows: list[dict], level: str, imgsz: int, k: int, pose_model: str, variant: str) -> dict | None:
    cid = config_id(level, imgsz, k, pose_model)
    return next((r for r in rows if r["config"] == cid and r["variant"] == variant), None)


def verdicts(rows: list[dict], pose_model: str) -> dict:
    rows = [r for r in rows if not r.get("invalid")]
    missing = [(i, k) for i in SIZES for k in KS if get(rows, "L0", i, k, pose_model, "paper") is None]
    if missing:
        raise ValueError(f"no valid L0 paper row for (imgsz, K) {missing}; verdicts need all four configs")
    auc = {(i, k): 100 * get(rows, "L0", i, k, pose_model, "paper")["auc"] for i in SIZES for k in KS
           if get(rows, "L0", i, k, pose_model, "paper")}
    best = max(auc, key=auc.get)
    t1 = auc[best] >= BASELINE_PAPER_L0 + T1_MARGIN
    t2 = {i: auc[(i, 3)] >= auc[(i, 1)] - T2_MARGIN for i in SIZES if (i, 1) in auc and (i, 3) in auc}
    return {"paper_auc_L0": {f"n{i}_K{k}": v for (i, k), v in auc.items()}, "best": f"n{best[0]}_K{best[1]}",
            "T1_pass": bool(t1), "T1_threshold": BASELINE_PAPER_L0 + T1_MARGIN,
            "T2_pass_by_size": {f"n{i}": bool(v) for i, v in t2.items()}, "T2_pass": bool(all(t2.values()))}


def fmt(row: dict | None) -> str:
    if row is None:
        return "-"
    return f"{100 * row['auc']:.2f} [{100 * row['ci95_low']:.2f}, {100 * row['ci95_high']:.2f}]"


def report(pose_model: str) -> str:
    rows = read_rows()
    invalid = [r for r in read_rows(include_invalid=True) if r.get("invalid") and r["variant"] == "paper"]
    timing_file = DATA.parent / "vm-exchange/results/exp016-timing/summary.json"
    timing = json.loads(timing_file.read_text()) if timing_file.exists() else {}
    head = ("| config | L0 paper AUC [95% CI] | L0 live | L0 anomalous cov. | L0 normal cov. | L2 paper | L2 live | "
            "L2 anomalous cov. | person-frames L0 | VM ms/frame mean (p50 / p99) | boxes/frame in VM subset |")
    lines = [head, "|" + "---|" * 11]
    b = {(lv, v): baseline_row(lv, v) for lv in ("L0", "L2") for v in ("paper", "live")}
    lines.append(f"| baseline YOLOv8n-pose 640 (single stage, EXP-003) | {fmt(b['L0', 'paper'])} | {fmt(b['L0', 'live'])} | "
                 f"{b['L0', 'paper']['coverage_anomalous']:.3f} | {b['L0', 'paper']['coverage_normal']:.3f} | "
                 f"{fmt(b['L2', 'paper'])} | {fmt(b['L2', 'live'])} | {b['L2', 'paper']['coverage_anomalous']:.3f} | "
                 f"{b['L0', 'paper']['poses']} | see EXP-004 | |")
    for i in SIZES:
        for k in KS:
            p0, l0 = (get(rows, "L0", i, k, pose_model, v) for v in ("paper", "live"))
            p2, l2 = (get(rows, "L2", i, k, pose_model, v) for v in ("paper", "live"))
            t = timing.get(f"L0-n{i}-K{k}-{pose_model}-c0.25-O0-B4")
            ms = "-" if t is None else f"{t['mean_ms']:.1f} ({t['p50_ms']:.1f} / {t['p99_ms']:.1f})"
            boxes = "-" if t is None else f"{t['mean_boxes']:.2f}"
            lines.append(f"| {i} px, K={k}, {pose_model} | {fmt(p0)} | {fmt(l0)} | "
                         f"{p0['coverage_anomalous'] if p0 else float('nan'):.3f} | {p0['coverage_normal'] if p0 else float('nan'):.3f} | "
                         f"{fmt(p2)} | {fmt(l2)} | {p2['coverage_anomalous'] if p2 else float('nan'):.3f} | "
                         f"{p0['poses'] if p0 else '-'} | {ms} | {boxes} |")
    verdict = verdicts(rows, pose_model)
    best = verdict["best"]
    best_i, best_k = int(best[1:].split("_")[0]), int(best.split("K")[1])
    base_scores = load_scores_file(EXP003 / "scores/L0_E0_c0.25_botsort/frame_scores_paper.npz")
    best_scores = load_scores_file(DATA / f"scores/{config_id('L0', best_i, best_k, pose_model)}/frame_scores_paper.npz")
    d = diff(base_scores, best_scores, n_resamples=1000)
    lines += ["", f"T1 (best L0 paper AUC >= {verdict['T1_threshold']:.2f}): {'PASS' if verdict['T1_pass'] else 'FAIL'}; best = {best} "
              f"{verdict['paper_auc_L0'][best]:.2f}; paired diff vs the single-stage baseline {100 * d['diff']:+.2f} "
              f"[{100 * d['ci'][0]:+.2f}, {100 * d['ci'][1]:+.2f}]"]
    for i, ok in verdict["T2_pass_by_size"].items():
        lines.append(f"T2 at {i}: K=3 {verdict['paper_auc_L0'][f'{i}_K3']:.2f} vs K=1 {verdict['paper_auc_L0'][f'{i}_K1']:.2f} "
                     f"(rule: K=3 >= K=1 - {T2_MARGIN}): {'PASS' if ok else 'FAIL'}")
    if invalid:
        lines += ["", "INVALID rows (kept for the record; scored on an incomplete MPS-crashed cache, excluded from everything above): "
                  "config, L0/L2 paper AUC, person-frames (poses), reason"]
        for r in invalid:
            lines.append(f"- {r['config']}: paper AUC {100 * r['auc']:.2f}, poses {r['poses']}, {r['invalid']}")
    (EXPERIMENT_DIR / "verdicts.json").write_text(json.dumps(verdict, indent=1))
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--pose-model", required=True, choices=["rtmpose-s", "rtmpose-t"])
    run.add_argument("--levels", nargs="+", default=["L0", "L2"])
    run.add_argument("--threads", type=int, default=4)
    rep = sub.add_parser("report")
    rep.add_argument("--pose-model", required=True, choices=["rtmpose-s", "rtmpose-t"])
    args = parser.parse_args()
    if args.command == "run":
        for level in args.levels:
            for imgsz in SIZES:
                for k in KS:
                    for row in run_config(level, imgsz, k, args.pose_model, args.threads):
                        print(json.dumps({x: row[x] for x in ("config", "variant", "auc", "coverage_anomalous", "poses")}),
                              flush=True)
    else:
        print(report(args.pose_model))


if __name__ == "__main__":
    main()
