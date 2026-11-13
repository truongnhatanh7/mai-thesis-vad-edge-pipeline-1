"""EXP-009 driver: pose filling (B interpolate / P extrapolate) on offline caches, deadline drops and keyframe schedules.

    uv run --extra pose python -m edgelab.analysis.exp009 run --part SHT      # or UB; resumable, one JSONL row per condition x variant
    uv run --extra pose python -m edgelab.analysis.exp009 analyse             # P1-P4, D, R verdicts -> summary.json, results.md

Spec: EXP-009-pose-filling/plan.md (frozen). Rows go to ~/thesis-data/exp009/rows.jsonl; per-frame scores to
~/thesis-data/exp009/scores/<condition>/frame_scores_<variant>.npz (same layout as score_dir.py).
Baseline rows (G = 0, no filling) are copied from the EXP-003 / EXP-006 results (same cache, c 0.25, same scorer).
"""

import argparse
import dataclasses
import json
import os
from pathlib import Path

import mlflow
import numpy as np

from edgelab.analysis.exp003 import Scores, diff
from edgelab.analysis.exp004 import usable_traces
from edgelab.analysis.exp008 import GROUPS, MANIFEST
from edgelab.metrics.scores import bootstrap_ci_over_videos, frame_level_auc
from edgelab.pipeline.deadline_replay import cache_frame_counts, load_trace
from edgelab.pipeline.deadline_sim import keep_masks_for_videos
from edgelab.pose.detcache import load_detections
from edgelab.pose.fill import SUFFIX, fill_dir, merge_parts, segment_predicted_shares
from edgelab.pose.replay import replay_dir
from edgelab.pose.score_dir import DATASET_DEFAULTS, VARIANTS, coverage, variant_scores
from edgelab.pose.stgnf_eval import STGNF_ROOT, _clips, score_frame_offsets, score_segments
from edgelab.tracking.mlflow_run import tracked_run

DATA = Path.home() / "thesis-data"
OUT = DATA / "exp009"
ROWS = OUT / "rows.jsonl"
RELEASED_POSE_DIR = STGNF_ROOT / "data/ShanghaiTech/pose/test"
TRACE_CELL = "L3-E0-R640-c0.25-O0-B4"  # EXP-004 O0-B4 traces for E0 @640
GATE_SHARE = 0.5  # B-gate: windows with more than this share of predicted poses are not scored
FILL_GAPS = (2, 4, 8, 12)
SLOWDOWNS = (1, 2, 4)
KEYFRAMES = (2, 3, 4)
DEADLINE_GROUPS = ("SHT-L0", "SHT-L2", "UB-L0")
KEYFRAME_GROUPS = ("SHT-L0", "UB-L0")
OFFLINE_GROUPS = ("SHT-L0", "SHT-L2", "SHT-L3", "UB-L0")
N_RESAMPLES = 2000


# ---------------------------------------------------------------- D1: keep released poses the edge detector also found

def keypoint_box(keypoints: np.ndarray) -> np.ndarray | None:
    """(17, 3) x, y, conf -> [x1, y1, x2, y2] over the keypoints with conf > 0; None if there is none."""
    visible = keypoints[keypoints[:, 2] > 0]
    if len(visible) == 0:
        return None
    return np.array([visible[:, 0].min(), visible[:, 1].min(), visible[:, 0].max(), visible[:, 1].max()])


def iou_with_boxes(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """IoU of one xyxy box with each row of boxes (N, 4)."""
    if len(boxes) == 0:
        return np.zeros(0)
    width = np.clip(np.minimum(box[2], boxes[:, 2]) - np.maximum(box[0], boxes[:, 0]), 0, None)
    height = np.clip(np.minimum(box[3], boxes[:, 3]) - np.maximum(box[1], boxes[:, 1]), 0, None)
    inter = width * height
    union = (box[2] - box[0]) * (box[3] - box[1]) + (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) - inter
    return np.where(union > 0, inter / np.where(union > 0, union, 1), 0.0)


def keep_matched(clip: dict, boxes_by_frame: dict[int, np.ndarray], iou_min: float = 0.3, shift: int = 0) -> tuple[dict, int, int]:
    """Keep the poses of a released clip whose keypoint box has IoU >= iou_min with a detector box on the same frame.

    shift: compare released frame f with detector frame f + shift (0 for the real run; +-1 is the alignment check).
    Returns (filtered clip, poses kept, poses total). Tracks left with no pose are dropped.
    """
    kept_clip, kept, total = {}, 0, 0
    for person_id, track in clip.items():
        track = merge_parts(track)
        kept_track = {}
        for frame, pose in track.items():
            total += 1
            box = keypoint_box(np.asarray(pose["keypoints"], dtype=float).reshape(-1, 3))
            boxes = boxes_by_frame.get(int(frame) + shift)
            if box is not None and boxes is not None and (iou_with_boxes(box, boxes) >= iou_min).any():
                kept_track[frame] = pose
                kept += 1
        if kept_track:
            kept_clip[person_id] = kept_track
    return kept_clip, kept, total


def detector_boxes(cache_file: Path, conf: float = 0.25) -> dict[int, np.ndarray]:
    frames, data = load_detections(cache_file)
    return {int(f): fr["boxes"][fr["conf"] >= conf] for f, fr in zip(data["frame_ids"], frames)}


def filter_released(pose_dir: Path, cache: Path, out_dir: Path, conf: float = 0.25, iou_min: float = 0.3) -> dict:
    """D1: released poses -> only those matched by the edge detector (cache at conf >= `conf`). Also logs the frame alignment check."""
    out_dir.mkdir(parents=True, exist_ok=True)
    kept = total = 0
    outside, shift_kept = 0, {-1: 0, 0: 0, 1: 0}
    for path in sorted(pose_dir.glob(f"*{SUFFIX}")):
        video = "_".join(path.name.split("_")[:2])
        boxes = detector_boxes(cache / f"{video}.npz", conf)
        clip = json.loads(path.read_text())
        released_frames = {int(f) for t in clip.values() for f in merge_parts(t)}
        outside += len(released_frames - set(boxes))  # released frame ids the cache does not have
        filtered, k, n = keep_matched(clip, boxes, iou_min)
        (out_dir / path.name).write_text(json.dumps(filtered))
        kept, total = kept + k, total + n
        for shift in (-1, 1):
            shift_kept[shift] += keep_matched(clip, boxes, iou_min, shift)[1]
        shift_kept[0] += k
    return {"poses_total": total, "poses_kept": kept, "released_frames_missing_in_cache": outside,
            "kept_by_frame_shift": {str(k): v for k, v in shift_kept.items()}}


# ---------------------------------------------------------------- scoring one pose folder -> rows

def per_frame_winning_share(segments, shares: np.ndarray, gt_dir: Path, offset: int, dataset: str, clips):
    """For every frame: predicted share of the window that gave the frame its score (the least-normal person's window).

    NaN where nobody scored the frame. Same placement rule as stgnf_eval.clip_normality (checked by the caller).
    """
    metadata, normality = segments.metadata, segments.normality
    scene_col, clip_col = metadata[:, 0].astype(str), metadata[:, 1].astype(str)
    out = []
    for _, scene_id, clip_id, gt in _clips(dataset, gt_dir, clips):
        rows_of_clip = (scene_col == scene_id) & (clip_col == clip_id)
        best, best_share = np.full(len(gt), np.inf), np.full(len(gt), np.nan)
        for pid in sorted(set(metadata[rows_of_clip, 2].astype(int))):
            rows = np.where(rows_of_clip & (metadata[:, 2].astype(int) == pid))[0]
            frames = metadata[rows, 3].astype(int) + offset
            inside = frames < len(gt)
            frames, rows = frames[inside], rows[inside]
            better = normality[rows] < best[frames]
            best[frames[better]] = normality[rows][better]
            best_share[frames[better]] = shares[rows][better]
        out.append(best_share)
    return out


def risk_split(segments, shares, gt_dir, dataset, clips, variant: str) -> dict:
    """R: AUC on frames scored from mostly-predicted windows (share > 0.5) vs windows without any predicted pose."""
    offset_name, smoother_name = VARIANTS[variant]
    offset = score_frame_offsets(dataset)[offset_name]
    names, labels, scores, covered = variant_scores(segments, gt_dir, offset_name, smoother_name, dataset, clips)
    winning = per_frame_winning_share(segments, shares, gt_dir, offset, dataset, clips)
    assert all(np.array_equal(np.isfinite(w), c) for w, c in zip(winning, covered)), "winning-share coverage != covered"
    labels, scores, winning = np.concatenate(labels), np.concatenate(scores), np.concatenate(winning)
    result = {}
    for name, mask in (("mostly_predicted", winning > GATE_SHARE), ("no_predicted", winning == 0)):
        y = labels[mask]
        auc = float(frame_level_auc([y], [scores[mask]])) if 0 < y.sum() < len(y) else None
        result[name] = {"frames": int(mask.sum()), "anomalous_frames": int(y.sum()), "auc": auc}
    return result


def score_condition(spec: dict, pose_dir: Path, prov_dir: Path | None, dataset: str, config: dict, threads: int = 2,
                    gate: bool = False, risk: bool = False, extra: dict | None = None, write: bool = True) -> list[dict]:
    """STG-NF on pose_dir once; rows for both variants (and for the B-gated windows when gate=True)."""
    checkpoint, gt_dir = DATASET_DEFAULTS[dataset]
    extra = extra or {}
    try:
        segments = score_segments(pose_dir, checkpoint, device="cpu", threads=threads, dataset=dataset)
    except (ValueError, IndexError) as error:  # STG-NF crashes when no window survives (ValueError: no track of seg_len frames;
        # IndexError: tracks long enough but every window has > 2 missing frames): pre-registered as AUC 0.5, coverage 0 (EXP-003)
        rows = [{**spec, **extra, "variant": v, "auc": 0.5, "ci": None, "coverage_anomalous": 0.0, "coverage_normal": 0.0,
                 "num_segments": 0, "predicted_share_of_scored_windows": 0.0, "no_segments": True, "error": str(error)}
                for v in VARIANTS]
        if write:
            append_rows(rows)
        return rows
    shares = segment_predicted_shares(segments.metadata, pose_dir, prov_dir, dataset)
    arms = [(spec, segments, shares)]
    if gate:
        keep = shares <= GATE_SHARE
        gated = dataclasses.replace(segments, normality=segments.normality[keep], metadata=segments.metadata[keep],
                                    num_segments=int(keep.sum()))
        arms.append(({**spec, "condition": spec["condition"] + "-gate", "mode": "Bgate"}, gated, shares[keep]))
    all_rows = []
    for arm_spec, arm_segments, arm_shares in arms:
        for variant, (offset_name, smoother_name) in VARIANTS.items():
            names, labels, scores, covered = variant_scores(arm_segments, gt_dir, offset_name, smoother_name, dataset)
            auc = frame_level_auc(labels, scores)
            low, high = bootstrap_ci_over_videos(labels, scores, n_resamples=N_RESAMPLES)
            cov_a, cov_n = coverage(covered, labels)
            scores_dir = OUT / "scores" / arm_spec["condition"].replace("/", "__")
            scores_dir.mkdir(parents=True, exist_ok=True)
            scores_file = scores_dir / f"frame_scores_{variant}.npz"
            np.savez_compressed(scores_file, names=np.array(names),
                                **{f"score_{n}": s for n, s in zip(names, scores)},
                                **{f"label_{n}": y for n, y in zip(names, labels)},
                                **{f"covered_{n}": c for n, c in zip(names, covered)})
            row = {**arm_spec, **extra, "variant": variant, "auc": auc, "ci": [low, high], "coverage_anomalous": cov_a,
                   "coverage_normal": cov_n, "num_segments": arm_segments.num_segments,
                   "predicted_share_of_scored_windows": float(arm_shares.mean()) if len(arm_shares) else 0.0,
                   "no_segments": False, "scores_file": str(scores_file)}
            if risk and arm_spec is spec:
                row["risk"] = risk_split(arm_segments, arm_shares, gt_dir, dataset, None, variant)
            if write:
                with tracked_run("EXP-009", f"{arm_spec['condition']}-{variant}", {**config, **arm_spec, "variant": variant,
                                                                                   "pose_dir": str(pose_dir)},
                                 input_files={"checkpoint": checkpoint}):
                    mlflow.log_metrics({"auc": auc, "ci_low": low, "ci_high": high, "coverage_anomalous": cov_a,
                                        "coverage_normal": cov_n, "num_segments": arm_segments.num_segments,
                                        "predicted_share": row["predicted_share_of_scored_windows"]})
                    mlflow.log_artifact(str(scores_file))
            all_rows.append(row)
            print(f"{arm_spec['condition']:34s} {variant:5s} AUC={auc * 100:.2f} [{low * 100:.2f},{high * 100:.2f}] "
                  f"cov a/n={cov_a:.3f}/{cov_n:.3f} seg={arm_segments.num_segments} "
                  f"pred={row['predicted_share_of_scored_windows']:.3f}", flush=True)
    if write:
        append_rows(all_rows)
    return all_rows


def append_rows(rows: list[dict]):
    OUT.mkdir(parents=True, exist_ok=True)
    with open(ROWS, "a") as f:
        f.write("".join(json.dumps(r) + "\n" for r in rows))


def read_rows() -> list[dict]:
    return [json.loads(line) for line in ROWS.read_text().splitlines() if line.strip()] if ROWS.exists() else []


def is_done(condition: str) -> bool:
    variants = {r["variant"] for r in read_rows() if r["condition"] == condition}
    return variants >= set(VARIANTS)


# ---------------------------------------------------------------- the conditions

def group_info(group: str):
    dataset, root, light = GROUPS[group]
    cache = DATA / root / "cache" / f"{light}_E0"
    videos = None
    if dataset == "UBnormal":
        from edgelab.pose.ubnormal import load_test_list
        videos = load_test_list()
    return dataset, DATA / root, cache, videos


def baseline_rows(group: str) -> list[dict]:
    """G = 0 rows: the EXP-003 / EXP-006 result of the same cache at c 0.25 (same scorer), re-labelled."""
    dataset, root, _, _ = group_info(group)
    cache_name = f"{GROUPS[group][2]}_E0"
    rows = []
    for line in (root / "results.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r["cache"] == cache_name and r["c"] == 0.25 and r["tracker"] == "botsort":
            rows.append({"group": group, "condition": f"{group}/offline/none/G0", "kind": "offline", "mode": "none", "G": 0,
                         "variant": r["variant"], "auc": r["auc"], "ci": [r["ci95_low"], r["ci95_high"]],
                         "coverage_anomalous": r["coverage_anomalous"], "coverage_normal": r["coverage_normal"],
                         "num_segments": r["num_segments"], "predicted_share_of_scored_windows": 0.0,
                         "latency_frames_max": 0, "no_segments": False, "source": f"{root.name}/results.jsonl",
                         "scores_file": r["scores_file"]})
    assert len(rows) == 2, f"expected 2 baseline rows for {group}, got {len(rows)}"
    return rows


def ensure_baseline(group: str):
    if not is_done(f"{group}/offline/none/G0"):
        append_rows(baseline_rows(group))


def run_offline(group: str, threads: int):
    dataset, root, cache, videos = group_info(group)
    ensure_baseline(group)
    src = root / "poses" / f"{GROUPS[group][2]}_E0_c0.25_botsort"
    counts = cache_frame_counts(cache, videos)
    for mode in ("B", "P"):
        for gap in FILL_GAPS:
            condition = f"{group}/offline/{mode}/G{gap}"
            gate = mode == "B" and gap == 8
            if is_done(condition) and (not gate or is_done(condition + "-gate")):
                continue
            work = OUT / "poses" / condition.replace("/", "__")
            stats = fill_dir(src, work, mode, gap, n_frames=counts)
            spec = {"group": group, "condition": condition, "kind": "offline", "mode": mode, "G": gap}
            extra = {"latency_frames_max": gap if mode == "B" else 0, **{f"poses_{k}": v for k, v in stats.items() if k.endswith("poses")}}
            score_condition(spec, work, work.with_name(work.name + "_prov"), dataset, {"group": group}, threads, gate=gate,
                            risk=gate and group == "SHT-L0", extra=extra)


def trace_for_cell() -> np.ndarray:
    return load_trace(usable_traces(MANIFEST)[TRACE_CELL])


def replay_with_masks(cache: Path, masks: dict, out: Path, videos, dataset: str):
    if not (out / "replay_stats.json").exists():
        replay_dir(cache, 0.25, out, keep_masks=masks, reindex=False, videos=videos, dataset=dataset)


def run_masked(group: str, tag: str, masks: dict, fills: list[tuple[str, int]], spec_base: dict, threads: int):
    """Replay with `masks` (gaps stay at real time), then score each fill: ('none', 0), ('P', 8), ('B', 8) ..."""
    dataset, _, cache, videos = group_info(group)
    counts = {v: len(m) for v, m in masks.items()}
    drop_rate = 1 - sum(int(m.sum()) for m in masks.values()) / sum(counts.values())
    base_dir = OUT / "poses" / f"{group}__{tag}__replay"
    for mode, gap in fills:
        condition = f"{group}/{spec_base['kind']}/{tag}/{mode}" + (f"/G{gap}" if mode != "none" else "")
        if is_done(condition):
            continue
        replay_with_masks(cache, masks, base_dir, videos, dataset)
        if mode == "none":
            work, prov = base_dir, None
        else:
            work = OUT / "poses" / condition.replace("/", "__")
            fill_dir(base_dir, work, mode, gap, n_frames=counts)
            prov = work.with_name(work.name + "_prov")
        spec = {"group": group, "condition": condition, "mode": mode, "G": gap, **spec_base}
        extra = {"drop_rate": drop_rate, "latency_frames_max": gap if mode == "B" else 0}
        score_condition(spec, work, prov, dataset, {"group": group}, threads, extra=extra,
                        risk=(group == "SHT-L0" and tag == "s2" and mode == "B"))


def run_deadline(group: str, threads: int):
    dataset, _, cache, videos = group_info(group)
    trace = trace_for_cell()
    counts = cache_frame_counts(cache, videos)
    for s in SLOWDOWNS:
        masks = keep_masks_for_videos(counts, trace * s, 24.0, seed=0)  # exactly as deadline_replay.run_cell
        run_masked(group, f"s{s}", masks, [("none", 0), ("P", 8), ("B", 8)], {"kind": "deadline", "s": s}, threads)


def keyframe_masks(counts: dict[str, int], k: int) -> dict[str, np.ndarray]:
    """Detector on every k-th frame (frames 0, k, 2k, ...)."""
    return {video: np.arange(n) % k == 0 for video, n in counts.items()}


def run_keyframe(group: str, threads: int):
    dataset, _, cache, videos = group_info(group)
    counts = cache_frame_counts(cache, videos)
    for k in KEYFRAMES:
        run_masked(group, f"k{k}", keyframe_masks(counts, k), [("B", k)], {"kind": "keyframe", "k": k}, threads)


def run_decomposition(threads: int):
    """ref = released poses; D1 = released poses the edge detector found; D2 = D1 + B fill G 8."""
    group = "SHT-L0"
    cache = DATA / "exp003/cache/L0_E0"
    counts = cache_frame_counts(cache)
    if not is_done("SHT-released/ref"):
        score_condition({"group": "SHT-released", "condition": "SHT-released/ref", "kind": "decomposition", "mode": "none", "G": 0},
                        RELEASED_POSE_DIR, None, "ShanghaiTech", {"group": group}, threads,
                        extra={"latency_frames_max": 0})
    d1_dir = OUT / "poses" / "SHT-released__D1"
    if not (d1_dir / "d1_stats.json").exists():
        stats = filter_released(RELEASED_POSE_DIR, cache, d1_dir)
        (d1_dir / "d1_stats.json").write_text(json.dumps(stats))
        log(f"D1 filter: {stats}")
    if not is_done("SHT-released/D1"):
        score_condition({"group": "SHT-released", "condition": "SHT-released/D1", "kind": "decomposition", "mode": "none", "G": 0},
                        d1_dir, None, "ShanghaiTech", {"group": group}, threads, extra={"latency_frames_max": 0})
    if not is_done("SHT-released/D2"):
        d2_dir = OUT / "poses" / "SHT-released__D2"
        fill_dir(d1_dir, d2_dir, "B", 8, n_frames=counts)
        score_condition({"group": "SHT-released", "condition": "SHT-released/D2", "kind": "decomposition", "mode": "B", "G": 8},
                        d2_dir, d2_dir.with_name(d2_dir.name + "_prov"), "ShanghaiTech", {"group": group}, threads,
                        extra={"latency_frames_max": 8})


def log(message: str):
    from datetime import datetime
    path = Path(__file__).resolve().parents[3] / "EXP-009-pose-filling/log.md"
    with open(path, "a") as f:
        f.write(f"\n### {datetime.now():%Y-%m-%d %H:%M} — driver\n- {message}\n")


def run_part(part: str, threads: int):
    OUT.mkdir(parents=True, exist_ok=True)
    if part == "SHT":
        run_offline("SHT-L0", threads)
        run_decomposition(threads)
        run_deadline("SHT-L0", threads)
        run_keyframe("SHT-L0", threads)
        run_deadline("SHT-L2", threads)
        run_offline("SHT-L2", threads)
        run_offline("SHT-L3", threads)
    else:
        run_offline("UB-L0", threads)
        run_deadline("UB-L0", threads)
        run_keyframe("UB-L0", threads)
    log(f"part {part} finished")


# ---------------------------------------------------------------- analysis

def load_scores_file(path: str | Path) -> Scores:
    data = np.load(path)
    names = [str(n) for n in data["names"]]
    return Scores(names, [data[f"label_{n}"] for n in names], [data[f"score_{n}"] for n in names],
                  [data[f"covered_{n}"].astype(bool) for n in names])


def paired_coverage_diff(a: Scores, b: Scores, n_resamples: int = 1000, seed: int = 0) -> dict:
    """Coverage(b) - coverage(a) of anomalous and of normal frames, resampling videos (same videos in both arms)."""
    def cover(s: Scores, idx, label):
        total = sum(int((s.labels[i] == label).sum()) for i in idx)
        return sum(int((s.covered[i] & (s.labels[i] == label)).sum()) for i in idx) / total if total else np.nan
    rng = np.random.default_rng(seed)
    n, everyone = len(a.labels), np.arange(len(a.labels))
    out = {}
    for name, label in (("anomalous", 1), ("normal", 0)):
        samples = []
        while len(samples) < n_resamples:
            idx = rng.integers(0, n, n)
            if np.isnan(cover(a, idx, label)):
                continue
            samples.append(cover(b, idx, label) - cover(a, idx, label))
        out[name] = {"a": cover(a, everyone, label), "b": cover(b, everyone, label),
                     "diff": cover(b, everyone, label) - cover(a, everyone, label),
                     "ci": [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))]}
    return out


def analyse():
    rows = read_rows()
    by = {(r["condition"], r["variant"]): r for r in rows}
    auc = lambda cond, variant: by[(cond, variant)]["auc"] * 100 if (cond, variant) in by else None  # noqa: E731
    verdicts = {}
    # P1
    p1 = {}
    for variant in VARIANTS:
        a, b = by[("SHT-L0/offline/none/G0", variant)], by[("SHT-L0/offline/B/G8", variant)]
        sa, sb = load_scores_file(a["scores_file"]), load_scores_file(b["scores_file"])
        d = diff(sa, sb)
        p1[variant] = {"auc_G0": a["auc"], "auc_B8": b["auc"], **d, "coverage": paired_coverage_diff(sa, sb)}
    verdicts["P1"] = {**p1, "pass_paper": p1["paper"]["ci"][0] > 0 and p1["paper"]["coverage"]["anomalous"]["diff"] > 0}
    # P2
    p2 = {g: {"B": auc(f"{g}/deadline/s2/B/G8", "live"), "none": auc(f"{g}/deadline/s2/none", "live")} for g in DEADLINE_GROUPS}
    verdicts["P2"] = {"groups": p2, "wins": sum(v["B"] is not None and v["B"] > v["none"] for v in p2.values()),
                      "pass": sum(v["B"] is not None and v["B"] > v["none"] for v in p2.values()) >= 2}
    # P3
    p3 = {v: {"full_rate": auc("SHT-L0/offline/none/G0", v), "k2_B": auc("SHT-L0/keyframe/k2/B/G2", v)} for v in VARIANTS}
    for v in p3.values():
        v["diff"] = v["k2_B"] - v["full_rate"] if v["k2_B"] is not None else None
    verdicts["P3"] = {**p3, "pass": all(v["diff"] is not None and abs(v["diff"]) <= 1.0 for v in p3.values())}
    # P4
    p4 = {g: {"B": auc(f"{g}/offline/B/G8", "paper"), "P": auc(f"{g}/offline/P/G8", "paper")} for g in OFFLINE_GROUPS}
    wins = sum(v["B"] is not None and v["P"] is not None and v["B"] >= v["P"] for v in p4.values())
    verdicts["P4"] = {"groups": p4, "wins": wins, "pass": wins >= 3}
    # D
    d = {}
    for v in VARIANTS:
        ref, d1, d2, e = (auc(c, v) for c in ("SHT-released/ref", "SHT-released/D1", "SHT-released/D2", "SHT-L0/offline/none/G0"))
        if None not in (ref, d1, d2, e):
            d[v] = {"ref": ref, "D1": d1, "D2": d2, "E": e, "ref_minus_E": ref - e, "ref_minus_D1": ref - d1,
                    "share_explained_by_D1": (ref - d1) / (ref - e) if ref != e else None, "recovery_D2_minus_D1": d2 - d1}
    verdicts["D"] = d
    verdicts["R"] = {r["condition"] + "/" + r["variant"]: r["risk"] for r in rows if "risk" in r}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(verdicts, indent=1, default=str))
    print(json.dumps(verdicts, indent=1, default=str))
    (Path(__file__).resolve().parents[3] / "EXP-009-pose-filling/results.md").write_text(results_markdown(rows, verdicts))


def results_markdown(rows: list[dict], verdicts: dict) -> str:
    lines = ["# EXP-009 results (generated by `edgelab.analysis.exp009 analyse`; numbers = AUC x 100)\n",
             "| condition | paper AUC [CI] | live AUC [CI] | cov anom/normal (paper) | segments | predicted share |",
             "|---|---|---|---|---|---|"]
    conds = list(dict.fromkeys(r["condition"] for r in rows))
    for cond in conds:
        paper = next((r for r in rows if r["condition"] == cond and r["variant"] == "paper"), None)
        live = next((r for r in rows if r["condition"] == cond and r["variant"] == "live"), None)
        if not (paper and live):
            continue
        fmt = lambda r: f"{r['auc'] * 100:.2f}" + (f" [{r['ci'][0] * 100:.1f}, {r['ci'][1] * 100:.1f}]" if r["ci"] else "")  # noqa: E731
        lines.append(f"| {cond} | {fmt(paper)} | {fmt(live)} | {paper['coverage_anomalous']:.3f} / {paper['coverage_normal']:.3f} "
                     f"| {paper['num_segments']} | {paper['predicted_share_of_scored_windows']:.3f} |")
    lines += ["", "## Verdicts", "```json", json.dumps(verdicts, indent=1, default=str), "```"]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--part", choices=["SHT", "UB"], required=True)
    run.add_argument("--threads", type=int, default=2)
    sub.add_parser("analyse")
    args = parser.parse_args()
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    if args.command == "run":
        run_part(args.part, args.threads)
    else:
        analyse()


if __name__ == "__main__":
    main()
