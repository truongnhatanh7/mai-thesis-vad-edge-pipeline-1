"""EXP-018 driver: DAPF (learned GRU pose filler C2 + filling-aware STG-NF C1) under CPU deadlines (plan + Amendments 1-2).

    uv run --extra pose python -m edgelab.analysis.exp018 run --group SHT-L0 [--threads 2]   # grid cells of one group; resumable
    uv run --extra pose python -m edgelab.analysis.exp018 offline                           # no drops, FA models vs released standard
    uv run --extra pose python -m edgelab.analysis.exp018 check                              # SKIP / FILL-HOLD vs EXP-010 / EXP-010b rows
    uv run --extra pose python -m edgelab.analysis.exp018 analyse                            # pooled differences + verdicts D1-D4

A cell = (group, s, seed). In a cell all arms see the same keep mask (EXP-004 latest-frame deadline simulator, EXP-004 trace cell
L3-E0-R640-c0.25-O0-B4 x s, seed 1 or 2 = the EXP-010 / EXP-010b masks):
  SKIP       dropped frames skipped, processed frames re-indexed, released STG-NF, scores held (= EXP-010);
  FILL-HOLD  constant-velocity fill (P, G 8), released STG-NF, frames without a score hold the last score (= EXP-010);
  +C1        constant-velocity fill, STG-NF-FA trained with constant-velocity fills (provenance condition from the sidecars);
  +C2        GRU fill, released STG-NF;
  DAPF       GRU fill, STG-NF-FA trained with GRU fills.
Rows: ~/thesis-data/exp018/rows.jsonl, per-frame scores: ~/thesis-data/exp018/scores/.
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import mlflow
import numpy as np
from sklearn.metrics import roc_auc_score

from edgelab.analysis import exp010
from edgelab.analysis.exp009 import load_scores_file
from edgelab.metrics.scores import bootstrap_ci_over_videos, frame_level_auc
from edgelab.pipeline.deadline_replay import cache_frame_counts
from edgelab.pipeline.deadline_sim import keep_masks_for_videos
from edgelab.pose.delivered import delivered_variant_scores
from edgelab.pose.fill import SUFFIX, fill_dir, merge_parts, segment_predicted_shares, video_of_file
from edgelab.pose.predictor import fill_dir_gru, load_predictor
from edgelab.pose.replay import FRAME_MAPS_FILE, replay_dir
from edgelab.pose.score_dir import DATASET_DEFAULTS, VARIANTS, coverage, variant_scores
from edgelab.pose.score_model import load_model, score_segments_model
from edgelab.pose.stgnf_eval import STGNF_ROOT
from edgelab.pose.stgnf_format import frame_key
from edgelab.tracking.mlflow_run import tracked_run

DATA = Path.home() / "thesis-data"
OUT = DATA / "exp018"
ROWS = OUT / "rows.jsonl"
EXPERIMENT_DIR = Path(__file__).resolve().parents[3] / "EXP-018-dapf"
SLOWDOWNS = (1, 3, 6)
SEEDS = (1, 2)
FILL_G = 8
MIN_WINDOWS = 200
N_ROW_CI = 500
N_PAIRED = 1000
ARMS = ("SKIP", "FILL-HOLD", "+C1", "+C2", "DAPF")  # arm -> (filler, scorer) in ARM_PARTS
ARM_PARTS = {"FILL-HOLD": ("cv", "released"), "+C1": ("cv", "fa-cv"), "+C2": ("gru", "released"), "DAPF": ("gru", "fa-gru")}
COMPARISONS = (("FILL-HOLD", "SKIP"), ("+C1", "FILL-HOLD"), ("+C2", "FILL-HOLD"), ("DAPF", "FILL-HOLD"), ("DAPF", "SKIP"))
# group -> dataset, YOLOv8n-pose detection cache (replay) or finished pose dir, keep masks of which group's cache
GROUPS = {"SHT-L0": {"dataset": "ShanghaiTech", "cache": DATA / "exp003/cache/L0_E0", "mask_cache": DATA / "exp003/cache/L0_E0"},
          "SHT-L2": {"dataset": "ShanghaiTech", "cache": DATA / "exp003/cache/L2_E0", "mask_cache": DATA / "exp003/cache/L2_E0"},
          "UB-L0": {"dataset": "UBnormal", "cache": DATA / "exp006/cache/L0_E0", "mask_cache": DATA / "exp006/cache/L0_E0"},
          "SHT-L0-TD": {"dataset": "ShanghaiTech", "poses": DATA / "exp016/poses/L0_n640_K1_rtmpose-s",
                        "mask_cache": DATA / "exp003/cache/L0_E0"}}
# no-drop sets for D3 and the offline table: name -> (dataset, pose dir)
OFFLINE_SETS = {"SHT-released": ("ShanghaiTech", STGNF_ROOT / "data/ShanghaiTech/pose/test"),
                "SHT-L0": ("ShanghaiTech", DATA / "exp003/poses/L0_E0_c0.25_botsort"),
                "SHT-L2": ("ShanghaiTech", DATA / "exp003/poses/L2_E0_c0.25_botsort"),
                "SHT-L0-TD": ("ShanghaiTech", DATA / "exp016/poses/L0_n640_K1_rtmpose-s"),
                "UB-L0": ("UBnormal", DATA / "exp006/poses/L0_E0_c0.25_botsort")}
OFFLINE_SCORERS = ("released", "fa-cv", "fa-gru")
DATASET_SHORT = {"ShanghaiTech": "SHT", "UBnormal": "UB"}


# ---------------------------------------------------------------- models and paths

def scorer_path(dataset: str, name: str) -> Path:
    if name == "released":
        return DATASET_DEFAULTS[dataset][0]
    return OUT / "models" / f"{DATASET_SHORT[dataset]}_{name}" / "final.pt"


def predictor_path(dataset: str) -> Path:
    return OUT / f"predictor_{dataset}" / "predictor.pt"


def log(message: str):
    with open(EXPERIMENT_DIR / "log.md", "a") as f:
        f.write(f"\n### {datetime.now():%Y-%m-%d %H:%M} — driver\n- {message}\n")


def read_rows() -> list[dict]:
    return [json.loads(line) for line in ROWS.read_text().splitlines() if line.strip()] if ROWS.exists() else []


def is_done(condition: str) -> bool:
    return {r["variant"] for r in read_rows() if r["condition"] == condition} >= set(VARIANTS)


def append_rows(rows: list[dict]):
    OUT.mkdir(parents=True, exist_ok=True)
    done = {(r["condition"], r["variant"]) for r in read_rows()}
    with open(ROWS, "a") as f:
        f.write("".join(json.dumps(r) + "\n" for r in rows if (r["condition"], r["variant"]) not in done))


def emit(spec: dict, variant: str, names, labels, scores, covered, num_segments: int, extra: dict, checkpoint: Path) -> dict:
    auc = frame_level_auc(labels, scores)
    low, high = bootstrap_ci_over_videos(labels, scores, n_resamples=N_ROW_CI)
    cov_a, cov_n = coverage(covered, labels)
    scores_file = OUT / "scores" / spec["condition"].replace("/", "__") / f"frame_scores_{variant}.npz"
    scores_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(scores_file, names=np.array(names), **{f"score_{n}": s for n, s in zip(names, scores)},
                        **{f"label_{n}": y for n, y in zip(names, labels)}, **{f"covered_{n}": c for n, c in zip(names, covered)})
    row = {**spec, **extra, "variant": variant, "auc": auc, "ci": [low, high], "coverage_anomalous": cov_a,
           "coverage_normal": cov_n, "num_segments": num_segments, "scores_file": str(scores_file)}
    with tracked_run("EXP-018", f"{spec['condition']}-{variant}", {**spec, **extra, "variant": variant}, input_files={"checkpoint": checkpoint}):
        mlflow.log_metrics({"auc": auc, "ci_low": low, "ci_high": high, "coverage_anomalous": cov_a,
                            "coverage_normal": cov_n, "num_segments": num_segments})
    print(f"{spec['condition']:40s} {variant:5s} AUC={auc * 100:.2f} [{low * 100:.2f},{high * 100:.2f}] seg={num_segments}", flush=True)
    return row


# ---------------------------------------------------------------- masks and dropping poses

_TRACE = {}


def masks_for(group: str, s: int, seed: int) -> dict[str, np.ndarray]:
    """Keep masks of the group's frames: EXP-010 recipe (trace x s, 24 fps, latest-frame simulator, seed)."""
    info = GROUPS[group]
    videos = None
    if info["dataset"] == "UBnormal":
        from edgelab.pose.ubnormal import load_test_list
        videos = load_test_list()
    if "8n" not in _TRACE:
        _TRACE["8n"] = exp010.trace_for("8n")
    counts = cache_frame_counts(info["mask_cache"], videos)
    return keep_masks_for_videos(counts, _TRACE["8n"] * s, 24.0, seed=seed)


def drop_frames_dir(in_dir: Path, out_dir: Path, masks: dict[str, np.ndarray], reindex: bool) -> dict[str, np.ndarray]:
    """Remove the poses of dropped frames from finished pose files (top-down group: no tracker re-run).

    reindex=True: kept frames are renumbered 0..K-1 (SKIP); returns {video: real frame id of each processed frame}.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    frame_maps = {}
    for path in sorted(Path(in_dir).glob(f"*{SUFFIX}")):
        video = video_of_file(path.name, masks)
        if video is None:
            raise ValueError(f"no keep mask for pose file {path.name}")
        mask = masks[video]
        frame_map = np.flatnonzero(mask)
        rank = {int(f): k for k, f in enumerate(frame_map)}
        clip = {}
        for person, track in json.loads(path.read_text()).items():
            kept = {}
            for key, pose in merge_parts(track).items():
                frame = int(key)
                if frame < len(mask) and mask[frame]:
                    kept[frame_key(rank[frame] if reindex else frame)] = pose
            if kept:
                clip[person] = kept
        (out_dir / path.name).write_text(json.dumps(clip))
        frame_maps[video] = frame_map
    if reindex:
        np.savez_compressed(out_dir / FRAME_MAPS_FILE, **frame_maps)
    return frame_maps


def dropped_dir(info: dict, masks: dict, out_dir: Path, reindex: bool) -> None:
    videos = sorted(masks) if info["dataset"] == "UBnormal" else None
    if "cache" in info:
        replay_dir(info["cache"], 0.25, out_dir, keep_masks=masks, reindex=reindex, videos=videos, dataset=info["dataset"])
    else:
        drop_frames_dir(info["poses"], out_dir, masks, reindex)


# ---------------------------------------------------------------- one cell

def gt_and_videos(dataset: str):
    videos = None
    if dataset == "UBnormal":
        from edgelab.pose.ubnormal import load_test_list
        videos = load_test_list()
    return DATASET_DEFAULTS[dataset][1], videos


def score_fill_arm(spec: dict, common: dict, dataset: str, pose_dir: Path, prov_dir: Path, scorer: str, threads: int) -> list[dict]:
    """Rows of one fill arm: STG-NF test pass on the filled poses, frames without a score hold the last score (FILL-HOLD rule)."""
    gt_dir, videos = gt_and_videos(dataset)
    checkpoint = scorer_path(dataset, scorer)
    model = load_model(checkpoint, dataset)
    rows = []
    try:
        segments = score_segments_model(model, pose_dir, dataset, prov_dir, threads=threads)
        shares = segment_predicted_shares(segments.metadata, pose_dir, prov_dir, dataset)
        extra = {"predicted_share_of_scored_windows": float(shares.mean()) if len(shares) else 0.0}
        for variant, (offset_name, smoother_name) in VARIANTS.items():
            names, labels, scores, covered = exp010.fill_variant_scores(segments, gt_dir, offset_name, smoother_name, dataset, "hold")
            rows.append(emit({**common, **spec}, variant, names, labels, scores, covered, segments.num_segments, extra, checkpoint))
    except (ValueError, IndexError) as error:  # no scorable window: AUC 0.5, nothing covered (EXP-003 rule, as EXP-010)
        for variant in VARIANTS:
            names, labels, scores, covered = exp010.constant_scores(gt_dir, dataset, videos)
            rows.append({**emit({**common, **spec}, variant, names, labels, scores, covered, 0, {}, checkpoint),
                         "no_segments": True, "error": str(error)[:200], "auc": 0.5, "ci": None})
        log(f"{spec['condition']}: no scorable window -> AUC 0.5 (EXP-003 rule)")
    return rows


def run_cell(group: str, s: int, seed: int, threads: int):
    info = GROUPS[group]
    dataset = info["dataset"]
    conditions = {arm: f"{group}/s{s}/seed{seed}/{arm}" for arm in ARMS}
    todo = [arm for arm in ARMS if not is_done(conditions[arm])]
    if not todo:
        return
    gt_dir, videos = gt_and_videos(dataset)
    masks = masks_for(group, s, seed)
    drop_rate = 1 - sum(int(m.sum()) for m in masks.values()) / sum(len(m) for m in masks.values())
    base = OUT / "poses" / f"{group}__s{s}__seed{seed}"
    common = {"kind": "grid", "group": group, "s": s, "seed": seed, "dataset": dataset, "drop_rate": drop_rate}
    released = scorer_path(dataset, "released")

    if "SKIP" in todo:
        skip_dir = base.with_name(base.name + "__skip")
        dropped_dir(info, masks, skip_dir, reindex=True)
        spec = {**common, "condition": conditions["SKIP"], "arm": "SKIP"}
        try:
            segments = score_segments_model(load_model(released, dataset), skip_dir, dataset, threads=threads)
            frame_maps = dict(np.load(skip_dir / FRAME_MAPS_FILE))
            rows = []
            for variant, (offset_name, smoother_name) in VARIANTS.items():
                names, labels, scores = delivered_variant_scores(segments.normality, segments.metadata, frame_maps, gt_dir,
                                                                 offset_name, smoother_name, dataset, videos)
                covered = [np.ones(len(y), bool) for y in labels]
                rows.append(emit(spec, variant, names, labels, scores, covered, segments.num_segments, {}, released))
        except (ValueError, IndexError) as error:
            rows = []
            for variant in VARIANTS:
                names, labels, scores, covered = exp010.constant_scores(gt_dir, dataset, videos)
                rows.append({**emit(spec, variant, names, labels, scores, covered, 0, {}, released),
                             "no_segments": True, "error": str(error)[:200], "auc": 0.5, "ci": None})
        append_rows(rows)

    fills = [arm for arm in todo if arm != "SKIP"]
    if not fills:
        return
    real_dir = base.with_name(base.name + "__realtime")
    dropped_dir(info, masks, real_dir, reindex=False)
    n_frames = {v: len(m) for v, m in masks.items()}
    fill_dirs = {}
    if any(ARM_PARTS[a][0] == "cv" for a in fills):
        fill_dirs["cv"] = base.with_name(base.name + "__P8")
        fill_dir(real_dir, fill_dirs["cv"], "P", FILL_G, n_frames=n_frames)
    if any(ARM_PARTS[a][0] == "gru" for a in fills):
        fill_dirs["gru"] = base.with_name(base.name + "__GRU8")
        fill_dir_gru(real_dir, fill_dirs["gru"], load_predictor(predictor_path(dataset)), FILL_G, n_frames=n_frames)
    for arm in fills:
        filler, scorer = ARM_PARTS[arm]
        pose_dir = fill_dirs[filler]
        spec = {"condition": conditions[arm], "arm": arm, "filler": filler, "scorer": scorer}
        append_rows(score_fill_arm(spec, common, dataset, pose_dir, pose_dir.with_name(pose_dir.name + "_prov"), scorer, threads))


def run_group(group: str, threads: int, slowdowns=SLOWDOWNS, seeds=SEEDS):
    OUT.mkdir(parents=True, exist_ok=True)
    for s in slowdowns:
        for seed in seeds:
            run_cell(group, s, seed, threads)
    log(f"group {group} finished (s {list(slowdowns)}, seeds {list(seeds)})")


# ---------------------------------------------------------------- offline (no drops)

def run_offline(threads: int):
    OUT.mkdir(parents=True, exist_ok=True)
    for name, (dataset, pose_dir) in OFFLINE_SETS.items():
        gt_dir, _ = gt_and_videos(dataset)
        for scorer in OFFLINE_SCORERS:
            if scorer != "released" and not scorer_path(dataset, scorer).exists():
                continue
            condition = f"offline/{name}/{scorer}"
            if is_done(condition):
                continue
            checkpoint = scorer_path(dataset, scorer)
            segments = score_segments_model(load_model(checkpoint, dataset), pose_dir, dataset, threads=threads)
            rows = []
            for variant, (offset_name, smoother_name) in VARIANTS.items():
                names, labels, scores, covered = variant_scores(segments, gt_dir, offset_name, smoother_name, dataset)
                spec = {"kind": "offline", "condition": condition, "group": name, "dataset": dataset, "arm": scorer}
                rows.append(emit(spec, variant, names, labels, scores, covered, segments.num_segments, {}, checkpoint))
            append_rows(rows)
    log("offline pass finished")


# ---------------------------------------------------------------- reproduction check

def reproduction_check() -> list[dict]:
    """SKIP and FILL-HOLD at s 3 and 6 vs the EXP-010 (seed 1) and EXP-010b (seed 2) rows (same masks, released checkpoints)."""
    reference = {}
    for seed, path in ((1, DATA / "exp010/rows.jsonl"), (2, DATA / "exp010b/rows.jsonl")):
        for line in path.read_text().splitlines():
            r = json.loads(line)
            reference[(seed, r["condition"], r["variant"])] = r
    mine = {(r["seed"], r["group"], r["s"], r["arm"], r["variant"]): r for r in read_rows() if r.get("kind") == "grid"}
    out = []
    for (seed, group, s, arm, variant), r in sorted(mine.items()):
        if arm not in ("SKIP", "FILL-HOLD") or s not in (3, 6) or group not in ("SHT-L0", "SHT-L2", "UB-L0"):
            continue
        key = (seed, f"{group}-8n/s{s}/{arm}", variant)
        if key not in reference:
            continue
        out.append({"cell": f"{group}/s{s}/seed{seed}/{arm}/{variant}", "auc_here": r["auc"] * 100, "auc_exp010": reference[key]["auc"] * 100,
                    "abs_diff": abs(r["auc"] - reference[key]["auc"]) * 100,
                    "windows_here": r["num_segments"], "windows_exp010": reference[key]["num_segments"]})
    return out


# ---------------------------------------------------------------- analysis

def auc_diffs_per_resample(cells: list[tuple], by: dict, n_resamples: int, seed: int = 0) -> np.ndarray:
    """(resamples, cells, comparisons) AUC differences x100, live variant. One video resample per dataset and resample,
    shared by every cell of that dataset (plan Amendment 1, point 8)."""
    loaded, datasets = {}, {}
    for cell in cells:
        datasets[cell] = by[(*cell, "SKIP", "live")]["dataset"]
        for arm in ARMS:
            loaded[(cell, arm)] = load_scores_file(by[(*cell, arm, "live")]["scores_file"])
    labels = {}
    for cell in cells:
        sc = loaded[(cell, "SKIP")]
        if datasets[cell] in labels:
            assert [y.shape for y in labels[datasets[cell]]] == [y.shape for y in sc.labels], "videos differ between cells"
        labels[datasets[cell]] = sc.labels
    rng = np.random.default_rng(seed)
    draws = {}
    for dataset, ys in labels.items():
        n, rows = len(ys), []
        while len(rows) < n_resamples:
            idx = rng.integers(0, n, n)
            if len(np.unique(np.concatenate([ys[i] for i in idx]))) == 2:
                rows.append(idx)
        draws[dataset] = rows
    result = np.zeros((n_resamples, len(cells), len(COMPARISONS)))
    for b in range(n_resamples):
        y_of = {d: np.concatenate([labels[d][i] for i in draws[d][b]]) for d in labels}
        for c, cell in enumerate(cells):
            d, idx = datasets[cell], draws[datasets[cell]][b]
            auc = {arm: roc_auc_score(y_of[d], np.concatenate([loaded[(cell, arm)].scores[i] for i in idx])) for arm in ARMS}
            for k, (a, b_arm) in enumerate(COMPARISONS):
                result[b, c, k] = (auc[a] - auc[b_arm]) * 100
    return result


def point_diffs(cells: list[tuple], by: dict) -> np.ndarray:
    return np.array([[(by[(*c, a, "live")]["auc"] - by[(*c, b, "live")]["auc"]) * 100 for a, b in COMPARISONS] for c in cells])


def pooled(diffs: np.ndarray, point: np.ndarray, select: list[int], k: int) -> dict:
    """mean over the selected cells; CI over the resamples."""
    if not select:
        return {"n_cells": 0, "diff": None, "ci": None}
    boot = diffs[:, select, k].mean(axis=1)
    return {"n_cells": len(select), "diff": float(point[select, k].mean()), "ci": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))]}


def analyse(n_resamples: int = N_PAIRED) -> dict:
    rows = read_rows()
    by = {(r["group"], r["s"], r["seed"], r["arm"], r["variant"]): r for r in rows if r.get("kind") == "grid"}
    cells = [(g, s, seed) for g in GROUPS for s in SLOWDOWNS for seed in SEEDS if all((g, s, seed, a, "live") in by for a in ARMS)]
    if len(cells) < len(GROUPS) * len(SLOWDOWNS) * len(SEEDS):
        print(f"WARNING: only {len(cells)} of {len(GROUPS) * len(SLOWDOWNS) * len(SEEDS)} cells are complete")
    degenerate = [min(by[(*c, a, "live")]["num_segments"] for a in ARMS) < MIN_WINDOWS for c in cells]
    diffs = auc_diffs_per_resample(cells, by, n_resamples)
    point = point_diffs(cells, by)
    good = [i for i, d in enumerate(degenerate) if not d]
    every = list(range(len(cells)))
    ki = {c: k for k, c in enumerate(COMPARISONS)}

    def selection(group=None, s=None, base=good):
        return [i for i in base if (group is None or cells[i][0] == group) and (s is None or cells[i][1] == s)]

    summary = {"cells": [list(c) for c in cells], "degenerate": [list(cells[i]) for i, d in enumerate(degenerate) if d],
               "overall": {}, "overall_with_degenerate": {}, "by_group_s": {}, "by_s": {}, "by_group": {}, "per_cell": []}
    for (a, b), k in ki.items():
        name = f"{a} - {b}"
        summary["overall"][name] = pooled(diffs, point, good, k)
        summary["overall_with_degenerate"][name] = pooled(diffs, point, every, k)
        summary["by_s"][name] = {str(s): pooled(diffs, point, selection(s=s), k) for s in SLOWDOWNS}
        summary["by_group"][name] = {g: pooled(diffs, point, selection(group=g), k) for g in GROUPS}
        summary["by_group_s"][name] = {f"{g}/s{s}": pooled(diffs, point, selection(g, s), k) for g in GROUPS for s in SLOWDOWNS}
    for i, c in enumerate(cells):
        summary["per_cell"].append({"cell": list(c), "degenerate": degenerate[i], "drop_rate": by[(*c, "SKIP", "live")]["drop_rate"],
                                    "auc_live": {a: by[(*c, a, "live")]["auc"] * 100 for a in ARMS},
                                    "windows": {a: by[(*c, a, "live")]["num_segments"] for a in ARMS},
                                    "diff": {f"{a} - {b}": float(point[i, k]) for (a, b), k in ki.items()}})
    positive = lambda cmp: float(np.mean([point[i, ki[cmp]] > 0 for i in good])) if good else float("nan")  # noqa: E731
    ov = summary["overall"]
    summary["D1"] = {"fraction_positive": positive(("DAPF", "FILL-HOLD")), "pooled": ov["DAPF - FILL-HOLD"],
                     "pass": bool(good and positive(("DAPF", "FILL-HOLD")) >= 0.75 and ov["DAPF - FILL-HOLD"]["ci"][0] > 0)}
    summary["D2"] = {"+C1": ov["+C1 - FILL-HOLD"], "+C2": ov["+C2 - FILL-HOLD"],
                     "pass": bool(good and ov["+C1 - FILL-HOLD"]["diff"] > 0 and ov["+C2 - FILL-HOLD"]["diff"] > 0)}
    summary["D4"] = {"pooled": ov["DAPF - SKIP"], "pass": bool(good and ov["DAPF - SKIP"]["ci"][0] > 0)}
    summary["D3"] = offline_verdict(rows)
    summary["reproduction"] = reproduction_check()
    (OUT / "verdicts.json").write_text(json.dumps(summary, indent=1, default=str))
    (EXPERIMENT_DIR / "results_tables.md").write_text(markdown(summary, by))
    print(json.dumps({k: summary[k] for k in ("D1", "D2", "D3", "D4")}, indent=1, default=str))
    return summary


def offline_verdict(rows: list[dict]) -> dict:
    """D3: no drops, paper variant: each FA model - released standard >= -1.0 AUC on released test poses and SHT-L0 edge poses (point estimates)."""
    off = {(r["group"], r["arm"], r["variant"]): r for r in rows if r.get("kind") == "offline"}
    table, passed = {}, True
    for group in ("SHT-released", "SHT-L0"):
        for scorer in ("fa-cv", "fa-gru"):
            if (group, scorer, "paper") not in off or (group, "released", "paper") not in off:
                return {"pass": None, "table": table, "note": "offline rows missing"}
            d = (off[(group, scorer, "paper")]["auc"] - off[(group, "released", "paper")]["auc"]) * 100
            table[f"{group}/{scorer}"] = d
            passed = passed and d >= -1.0
    return {"table": table, "pass": bool(passed)}


def fmt_pooled(p: dict) -> str:
    return "n/a" if not p["n_cells"] else f"{p['diff']:+.2f} [{p['ci'][0]:+.2f}, {p['ci'][1]:+.2f}] (n={p['n_cells']})"


def markdown(summary: dict, by: dict) -> str:
    ok = lambda b: "PASS" if b else ("FAIL" if b is not None else "n/a")  # noqa: E731
    d1, d2, d3, d4 = (summary[k] for k in ("D1", "D2", "D3", "D4"))
    lines = ["# EXP-018 results tables (generated by `edgelab.analysis.exp018 analyse`; live variant, AUC x100, "
             "paired bootstrap over videos, 1000 resamples, one resample per dataset shared by all its cells)\n",
             "## Verdicts\n", "| # | rule | result | verdict |", "|---|---|---|---|",
             f"| D1 | DAPF > FILL-HOLD in >= 75% of non-degenerate cells and pooled CI lower bound > 0 | {d1['fraction_positive'] * 100:.0f}% of cells positive; pooled {fmt_pooled(d1['pooled'])} | {ok(d1['pass'])} |",
             f"| D2 | +C1 and +C2 each > FILL-HOLD (pooled point estimate) | +C1 {fmt_pooled(d2['+C1'])}; +C2 {fmt_pooled(d2['+C2'])} | {ok(d2['pass'])} |",
             f"| D3 | no offline cost: FA >= released - 1.0 (released test poses and SHT-L0 edge poses, paper variant) | "
             + ", ".join(f"{k} {v:+.2f}" for k, v in d3["table"].items()) + f" | {ok(d3['pass'])} |",
             f"| D4 | DAPF > SKIP, pooled CI lower bound > 0 | {fmt_pooled(d4['pooled'])} | {ok(d4['pass'])} |\n",
             f"Degenerate cells (any arm < {MIN_WINDOWS} windows, excluded above): " + (", ".join("/".join(map(str, c)) for c in summary["degenerate"]) or "none") + "\n",
             "## Pooled differences (mean over cells [95% CI])\n", "| comparison | all non-degenerate cells | incl. degenerate (sensitivity) | s=1 | s=3 | s=6 |", "|---|---|---|---|---|---|"]
    for name, p in summary["overall"].items():
        lines.append(f"| {name} | {fmt_pooled(p)} | {fmt_pooled(summary['overall_with_degenerate'][name])} | "
                     + " | ".join(fmt_pooled(summary["by_s"][name][str(s)]) for s in SLOWDOWNS) + " |")
    groups = list(GROUPS)
    lines += ["\n## Pooled over the two seeds, per group and slowdown\n", "| comparison | s | " + " | ".join(groups) + " |", "|---|---|" + "---|" * len(groups)]
    for name in summary["overall"]:
        for s in SLOWDOWNS:
            lines.append(f"| {name} | {s} | " + " | ".join(fmt_pooled(summary["by_group_s"][name][f"{g}/s{s}"]) for g in groups) + " |")
    lines += ["\n## Every cell (live AUC; windows = scorable windows of the arm)\n",
              "| group | s | seed | drop rate | degenerate | " + " | ".join(ARMS) + " | windows (" + "/".join(ARMS) + ") |", "|---|---|---|---|---|" + "---|" * (len(ARMS) + 1)]
    for c in summary["per_cell"]:
        g, s, seed = c["cell"]
        lines.append(f"| {g} | {s} | {seed} | {c['drop_rate']:.2f} | {'yes' if c['degenerate'] else ''} | "
                     + " | ".join(f"{c['auc_live'][a]:.2f}" for a in ARMS) + " | " + "/".join(str(c["windows"][a]) for a in ARMS) + " |")
    lines += ["\n## Offline, no drops (AUC; paper / live)\n", "| set | " + " | ".join(OFFLINE_SCORERS) + " |", "|---|" + "---|" * len(OFFLINE_SCORERS)]
    off = {(r["group"], r["arm"], r["variant"]): r for r in read_rows() if r.get("kind") == "offline"}
    for name in OFFLINE_SETS:
        cells = [f"{off[(name, sc, 'paper')]['auc'] * 100:.2f} / {off[(name, sc, 'live')]['auc'] * 100:.2f}" if (name, sc, "paper") in off else "n/a" for sc in OFFLINE_SCORERS]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    rep = summary["reproduction"]
    if rep:
        lines += ["\n## SKIP / FILL-HOLD vs the EXP-010 / EXP-010b rows (s = 3, 6)\n",
                  f"{len(rep)} rows compared; max |AUC difference| = {max(r['abs_diff'] for r in rep):.4f} points; "
                  f"windows equal in {sum(r['windows_here'] == r['windows_exp010'] for r in rep)} of {len(rep)}.\n"]
        worst = sorted(rep, key=lambda r: -r["abs_diff"])[:5]
        lines += ["| cell | here | EXP-010(b) | |diff| |", "|---|---|---|---|"] + [f"| {r['cell']} | {r['auc_here']:.3f} | {r['auc_exp010']:.3f} | {r['abs_diff']:.4f} |" for r in worst]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--group", required=True, choices=list(GROUPS))
    run.add_argument("--threads", type=int, default=2)
    run.add_argument("--s", type=int, nargs="+", default=list(SLOWDOWNS))
    run.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    off = sub.add_parser("offline")
    off.add_argument("--threads", type=int, default=2)
    sub.add_parser("check")
    sub.add_parser("analyse")
    args = parser.parse_args()
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    if args.command == "run":
        run_group(args.group, args.threads, tuple(args.s), tuple(args.seeds))
    elif args.command == "offline":
        run_offline(args.threads)
    elif args.command == "check":
        print(json.dumps(reproduction_check(), indent=1))
    else:
        analyse()


if __name__ == "__main__":
    main()
