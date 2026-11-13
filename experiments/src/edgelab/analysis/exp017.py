"""EXP-017 driver: YOLO26n-pose vs YOLOv8n-pose as the edge front end (plan frozen 2026-11-07 22:20). Verdicts Y1-Y5.

    # accuracy caches (CPU, detcache --weights artifacts/EXP-017/yolo26n-pose.pt --nms-free) -> gate_sweep (see EXP-017/log.md)
    uv run --extra pose python -m edgelab.analysis.exp017 offline        # Y1 (recall by height), Y2, Y3 -> offline.json
    uv run --extra pose python -m edgelab.analysis.exp017 delivered      # Y5 FILL-HOLD replays (needs the Y4 timing traces)
    uv run --extra pose python -m edgelab.analysis.exp017 cost           # Y4 from the VM timing
    uv run --extra pose python -m edgelab.analysis.exp017 report         # verdicts.json + results_tables.md

Baselines (YOLOv8n-pose, same pipeline): EXP-003 caches L0_E0 / L2_E0 / L3_E0 (ShanghaiTech), EXP-006 L0_E0 (UBnormal).
Y5 reuses the EXP-010 driver (`edgelab.analysis.exp010`, keep-mask seed 1, P fill G = 8, FILL-HOLD) with measured VM traces.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from edgelab.analysis import exp010, exp011
from edgelab.analysis.exp003 import diff
from edgelab.analysis.exp009 import load_scores_file
from edgelab.pose.topdown import EXPECTED_VIDEOS, IncompleteCacheError, check_cache_complete

DATA = Path.home() / "thesis-data"
SHT, UB = DATA / "exp017-sht", DATA / "exp017-ub"
OUT = DATA / "exp017"
EXPERIMENT_DIR = Path(__file__).resolve().parents[3] / "EXP-017-yolo26"
BASE = {"ShanghaiTech": DATA / "exp003", "UBnormal": DATA / "exp006"}
# plan §3 pass rules
Y2_MARGIN, Y3_MARGIN, Y4_LIMIT = 1.5, 0.5, 1.10
BASELINE_PAPER_SHT_L0, BASELINE_PAPER_UB_L0 = 68.53, 55.76
LIGHTS = ("L0", "L2", "L3")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def row(rows: list[dict], cache: str, c: float, variant: str) -> dict | None:
    return next((r for r in rows if r["cache"] == cache and r["c"] == c and r["variant"] == variant), None)


def score_path(root: Path, cache: str, c: float, variant: str) -> Path:
    return root / "scores" / f"{cache}_c{c}_botsort" / f"frame_scores_{variant}.npz"


def paired(base_root: Path, base_cache: str, root: Path, cache: str, c: float, variant: str, n_resamples: int = 1000) -> dict:
    d = diff(load_scores_file(score_path(base_root, base_cache, c, variant)), load_scores_file(score_path(root, cache, c, variant)),
             n_resamples=n_resamples)
    return {"diff_points": 100 * d["diff"], "ci": [100 * d["ci"][0], 100 * d["ci"][1]]}


def y2(auc_paper: float, paired_ci_low: float) -> dict:
    """Y2: ShanghaiTech L0 paper AUC >= 68.53 + 1.5 and the paired CI lower bound > 0."""
    return {"auc": auc_paper, "threshold": BASELINE_PAPER_SHT_L0 + Y2_MARGIN, "ci_low": paired_ci_low,
            "pass": bool(auc_paper >= BASELINE_PAPER_SHT_L0 + Y2_MARGIN and paired_ci_low > 0)}


def y3(auc_paper: float) -> dict:
    """Y3: UBnormal L0 paper AUC >= 55.76 - 0.5."""
    return {"auc": auc_paper, "threshold": BASELINE_PAPER_UB_L0 - Y3_MARGIN, "pass": bool(auc_paper >= BASELINE_PAPER_UB_L0 - Y3_MARGIN)}


def y1(table: dict, new: str, old: str) -> dict:
    """Y1: recall in the 48-64 AND 64-96 px bins both higher for `new` than for `old` (recall_by_height tables)."""
    bins = ("48-64", "64-96")
    r = {b: {"new": exp011.recall(table[new][b]), "old": exp011.recall(table[old][b])} for b in bins}
    return {"bins": r, "pass": bool(all(v["new"] > v["old"] for v in r.values()))}


def y4(ms_new: float, ms_old: float) -> dict:
    """Y4: NCNN FP16 detector ms at 640 <= 1.10 x the YOLOv8n-pose value."""
    return {"new_ms": ms_new, "old_ms": ms_old, "ratio": ms_new / ms_old, "pass": bool(ms_new <= Y4_LIMIT * ms_old)}


def y5(auc_new: dict, auc_old: dict) -> dict:
    """Y5: delivered live AUC (FILL-HOLD) of the new detector > the old one's at every slowdown given (s = 1 and 3)."""
    return {"auc_new": auc_new, "auc_old": auc_old, "pass": bool(all(auc_new[s] > auc_old[s] for s in auc_old))}


# ---------------------------------------------------------------- offline (Y1-Y3)

def offline() -> dict:
    out = {"sht": {}, "ub": {}}
    sht_rows, ub_rows = read_jsonl(SHT / "results.jsonl"), read_jsonl(UB / "results.jsonl")
    for kind, root, rows, dataset in (("sht", SHT, sht_rows, "ShanghaiTech"), ("ub", UB, ub_rows, "UBnormal")):
        for r in rows:
            base_cache = r["cache"].split("-o2m")[0]  # L0_E0-o2m is compared with the same baseline
            base = row(read_jsonl(BASE[dataset] / "results.jsonl"), base_cache, r["c"], r["variant"])
            out[kind][f"{r['cache']}|c{r['c']}|{r['variant']}"] = {
                "auc": r["auc"], "ci95": [r.get("ci95_low"), r.get("ci95_high")], "coverage_anomalous": r["coverage_anomalous"],
                "coverage_normal": r["coverage_normal"], "poses": r["poses"], "baseline_auc": base["auc"] if base else None,
                "baseline_poses": base["poses"] if base else None}
    for kind, root, dataset in (("sht", SHT, "ShanghaiTech"), ("ub", UB, "UBnormal")):
        for cache in sorted({k.split("|")[0] for k in out[kind]}):
            for variant in ("paper", "live"):
                key = f"{cache}|c0.25|{variant}"
                if key in out[kind] and score_path(root, cache, 0.25, variant).exists():
                    out[kind][key]["paired_vs_baseline"] = paired(BASE[dataset], cache.split("-o2m")[0], root, cache, 0.25, variant)
    verdict = {}
    k = "L0_E0|c0.25|paper"
    if k in out["sht"] and "paired_vs_baseline" in out["sht"][k]:
        verdict["Y2"] = y2(100 * out["sht"][k]["auc"], out["sht"][k]["paired_vs_baseline"]["ci"][0])
    if k in out["ub"]:
        verdict["Y3"] = y3(100 * out["ub"][k]["auc"])
    people = exp011.reference_people()
    table = {"yolov8n (EXP-003 L0_E0)": exp011.recall_by_bin(people, BASE["ShanghaiTech"] / "cache/L0_E0")}
    for name, cache in (("yolo26n e2e", SHT / "cache/L0_E0"), ("yolo26n o2m+NMS (sensitivity)", SHT / "cache/L0_E0-o2m")):
        try:
            check_cache_complete(cache, EXPECTED_VIDEOS)  # never evaluate an unfinished cache (EXP-016 deviation D-1)
        except IncompleteCacheError:
            continue
        table[name] = exp011.recall_by_bin(people, cache)
    if "yolo26n e2e" in table:
        verdict["Y1"] = y1(table, "yolo26n e2e", "yolov8n (EXP-003 L0_E0)")
    out["recall_by_height"] = table
    out["verdict"] = verdict
    (OUT / "offline.json").parent.mkdir(parents=True, exist_ok=True)
    (OUT / "offline.json").write_text(json.dumps(out, indent=1, default=float))
    return out


# ---------------------------------------------------------------- delivered (Y5)

DELIVERED = OUT / "delivered"
# group -> (cache, trace id). "8n" = EXP-010's own trace (EXP-004 ORT FP32 640); others come from the Y4 timing run (same window).
DELIVERED_GROUPS = {"SHT-L0-8n": (DATA / "exp003/cache/L0_E0", "8n"),
                    "SHT-L0-26n": (SHT / "cache/L0_E0", "yolo26n-ort-fp32-R640"),
                    "SHT-L0-8n-win": (DATA / "exp003/cache/L0_E0", "yolov8n-ort-fp32-R640"),
                    "SHT-L0-8n-ncnn": (DATA / "exp003/cache/L0_E0", "yolov8n-ncnn-fp16-R640"),
                    "SHT-L0-26n-ncnn": (SHT / "cache/L0_E0-o2m", "yolo26n-ncnn-fp16-R640")}
PRIMARY = ("SHT-L0-8n", "SHT-L0-26n")
SLOWDOWNS = (1, 3)


def trace_for(trace_id: str) -> np.ndarray:
    from edgelab.pipeline.deadline_replay import load_trace
    from edgelab.pipeline.timing_exp017 import usable_files

    if trace_id == "8n":
        from edgelab.analysis.exp004 import usable_traces
        from edgelab.analysis.exp008 import MANIFEST

        return load_trace(usable_traces(MANIFEST)[exp010.TRACE_CELL])
    return load_trace(usable_files(trace_id))


def configure_exp010():
    exp010.configure(seed=1, out_root=DELIVERED, mlflow_experiment="EXP-017", log_dir=EXPERIMENT_DIR)
    exp010.GROUPS = {g: ("ShanghaiTech", cache, trace_id) for g, (cache, trace_id) in DELIVERED_GROUPS.items()}
    exp010.trace_for = trace_for  # (detector field of GROUPS = trace id)
    exp010.SLOWDOWNS = SLOWDOWNS


def delivered(groups: list[str], threads: int):
    configure_exp010()
    DELIVERED.mkdir(parents=True, exist_ok=True)
    for group in groups:
        for s in SLOWDOWNS:
            exp010.run_group_speed(group, s, threads)


def delivered_table() -> dict:
    configure_exp010()
    rows = exp010.read_rows()
    by = {(r["condition"], r["variant"]): r for r in rows}
    out = {}
    for group in DELIVERED_GROUPS:
        for s in SLOWDOWNS:
            for policy in exp010.POLICIES:
                r = by.get((f"{group}/s{s}/{policy}", "live"))
                p = by.get((f"{group}/s{s}/{policy}", "paper"))
                if r:
                    out[f"{group}/s{s}/{policy}"] = {"live": r["auc"], "paper": p["auc"] if p else None, "drop_rate": r["drop_rate"],
                                                     "coverage_anomalous": r["coverage_anomalous"], "scores_file": r["scores_file"]}
    return out


def y5_verdicts(table: dict, pair: tuple[str, str] = PRIMARY) -> dict:
    old_g, new_g = pair
    auc_old = {s: table[f"{old_g}/s{s}/FILL-HOLD"]["live"] for s in SLOWDOWNS}
    auc_new = {s: table[f"{new_g}/s{s}/FILL-HOLD"]["live"] for s in SLOWDOWNS}
    verdict = y5({s: 100 * v for s, v in auc_new.items()}, {s: 100 * v for s, v in auc_old.items()})
    verdict["paired"] = {}
    for s in SLOWDOWNS:
        d = diff(load_scores_file(table[f"{old_g}/s{s}/FILL-HOLD"]["scores_file"]), load_scores_file(table[f"{new_g}/s{s}/FILL-HOLD"]["scores_file"]),
                 n_resamples=1000)
        verdict["paired"][f"s{s}"] = {"diff_points": 100 * d["diff"], "ci": [100 * d["ci"][0], 100 * d["ci"][1]]}
    verdict["drop_rate"] = {f"s{s}": {"old": table[f"{old_g}/s{s}/FILL-HOLD"]["drop_rate"], "new": table[f"{new_g}/s{s}/FILL-HOLD"]["drop_rate"]}
                            for s in SLOWDOWNS}
    return verdict


# ---------------------------------------------------------------- cost (Y4)

def cost() -> dict:
    from edgelab.pipeline.timing_exp012 import RESULTS as T12
    from edgelab.pipeline.timing_exp012 import summary as summary12
    from edgelab.pipeline.timing_exp017 import summary

    t17 = summary().set_index("variant")
    t12 = summary12(T12).set_index("variant")
    out = {"timing": t17.round(3).reset_index().to_dict("records"),
           "exp012_ncnn_fp16_R640_det_ms": float(t12.det_ms["ncnn-fp16-R640"]), "exp012_ort_fp32_R640_det_ms": float(t12.det_ms["ort-fp32-R640"])}
    new = float(t17.det_ms["yolo26n-ncnn-fp16-R640"])
    out["Y4"] = {**y4(new, out["exp012_ncnn_fp16_R640_det_ms"]), "comparator": "EXP-012 NCNN FP16 R640 (plan)"}
    if "yolov8n-ncnn-fp16-R640" in t17.index:
        out["Y4_same_window_control"] = y4(new, float(t17.det_ms["yolov8n-ncnn-fp16-R640"]))
    return out


# ---------------------------------------------------------------- report

def report() -> str:
    parts = {}
    for name in ("offline", "cost"):
        f = OUT / f"{name}.json"
        parts[name] = json.loads(f.read_text()) if f.exists() else {}
    verdicts = {**parts["offline"].get("verdict", {})}
    if parts["cost"]:
        verdicts["Y4"] = parts["cost"]["Y4"]
    try:
        table = delivered_table()
        if all(f"{g}/s{s}/FILL-HOLD" in table for g in PRIMARY for s in SLOWDOWNS):
            verdicts["Y5"] = y5_verdicts(table)
        sens = {}
        for pair in (("SHT-L0-8n-win", "SHT-L0-26n"), ("SHT-L0-8n-ncnn", "SHT-L0-26n-ncnn")):
            if all(f"{g}/s{s}/FILL-HOLD" in table for g in pair for s in SLOWDOWNS):
                sens[f"{pair[0]} vs {pair[1]}"] = y5_verdicts(table, pair)
        verdicts["Y5_sensitivity"] = sens
        parts["delivered"] = table
    except FileNotFoundError:
        pass
    (OUT / "verdicts.json").write_text(json.dumps(verdicts, indent=1, default=float))
    (EXPERIMENT_DIR / "verdicts.json").write_text(json.dumps(verdicts, indent=1, default=float))
    return markdown(parts, verdicts)


def markdown(parts: dict, verdicts: dict) -> str:
    lines = ["# EXP-017 tables (generated by `edgelab.analysis.exp017 report`; AUC x 100)\n"]
    off = parts.get("offline", {})
    for kind, title in (("sht", "ShanghaiTech"), ("ub", "UBnormal")):
        if not off.get(kind):
            continue
        lines += [f"## {title}: YOLO26n-pose (end-to-end head unless `-o2m`) vs YOLOv8n-pose baseline, same pipeline\n",
                  "| cache | c | paper | live | baseline paper | baseline live | paired diff paper (c 0.25) [95% CI] | paired diff live (c 0.25) | poses | baseline poses | anomalous cov. |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
        caches = sorted({k.split("|")[0] for k in off[kind]})
        for cache in caches:
            for c in (0.25, 0.15, 0.1, 0.05):
                p, lv = off[kind].get(f"{cache}|c{c}|paper"), off[kind].get(f"{cache}|c{c}|live")
                if not p:
                    continue
                dp = p.get("paired_vs_baseline")
                dl = lv.get("paired_vs_baseline") if lv else None
                fmt = lambda d: "" if not d else f"{d['diff_points']:+.2f} [{d['ci'][0]:+.2f}, {d['ci'][1]:+.2f}]"  # noqa: E731
                lines.append(f"| {cache} | {c} | {100 * p['auc']:.2f} | {100 * lv['auc']:.2f} | {100 * p['baseline_auc']:.2f} | "
                             f"{100 * lv['baseline_auc']:.2f} | {fmt(dp)} | {fmt(dl)} | {p['poses']} | {p['baseline_poses']} | "
                             f"{p['coverage_anomalous']:.3f} |")
        lines.append("")
    rec = off.get("recall_by_height")
    if rec:
        names = exp011.BIN_NAMES + ["all"]
        lines += ["## Recall by reference-person height, ShanghaiTech L0 (EXP-011 code; conf >= 0.25, IoU >= 0.3)\n",
                  "| detector | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
        for name, res in rec.items():
            lines.append(f"| {name} | " + " | ".join(f"{exp011.recall(res[b]):.3f} ({res[b]['found']}/{res[b]['persons']})" for b in names) + " |")
        lines.append("")
    cost_ = parts.get("cost")
    if cost_:
        lines += ["## Detector cost in the VM (pre + infer + post, ms/frame; EXP-012 protocol)\n",
                  "| variant | runs | frames | det ms | per-run sd | p95 | infer | post | service ms | boxes/frame | host load outside VM % |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
        for t in cost_["timing"]:
            lines.append(f"| {t['variant']} | {t['runs']} | {t['frames']} | {t['det_ms']:.2f} | {t['det_ms_per_run_sd']:.2f} | "
                         f"{t['det_p95_ms']:.2f} | {t['infer_ms']:.2f} | {t['post_ms']:.2f} | {t['service_ms']:.2f} | {t['n_dets']:.2f} | "
                         f"{t['host_outside_vm_pct']:.1f} |")
        lines.append(f"\nEXP-012 NCNN FP16 R640 detector ms (plan comparator): {cost_['exp012_ncnn_fp16_R640_det_ms']:.2f}\n")
    dtab = parts.get("delivered")
    if dtab:
        lines += ["## Delivered (24 fps, seed 1, P fill G 8): live AUC by policy\n", "| group | s | drop rate | SKIP | FILL-HOLD | FILL-NORMAL |",
                  "|---|---|---|---|---|---|"]
        for group in DELIVERED_GROUPS:
            for s in SLOWDOWNS:
                cells = [dtab.get(f"{group}/s{s}/{p}") for p in exp010.POLICIES]
                if all(cells):
                    lines.append(f"| {group} | {s} | {cells[1]['drop_rate']:.3f} | " + " | ".join(f"{100 * c['live']:.2f}" for c in cells) + " |")
        lines.append("")
    lines += ["## Verdicts\n", "```json", json.dumps(verdicts, indent=1, default=float), "```"]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("offline")
    d = sub.add_parser("delivered")
    d.add_argument("--groups", nargs="+", default=list(PRIMARY), choices=list(DELIVERED_GROUPS))
    d.add_argument("--threads", type=int, default=2)
    sub.add_parser("cost")
    sub.add_parser("report")
    args = parser.parse_args()
    if args.command == "offline":
        print(json.dumps(offline()["verdict"], indent=1, default=float))
    elif args.command == "delivered":
        delivered(args.groups, args.threads)
    elif args.command == "cost":
        out = cost()
        (OUT / "cost.json").parent.mkdir(parents=True, exist_ok=True)
        (OUT / "cost.json").write_text(json.dumps(out, indent=1, default=float))
        print(json.dumps(out["Y4"], indent=1))
    else:
        text = report()
        (EXPERIMENT_DIR / "results_tables.md").write_text(text)
        print(text)


if __name__ == "__main__":
    main()
