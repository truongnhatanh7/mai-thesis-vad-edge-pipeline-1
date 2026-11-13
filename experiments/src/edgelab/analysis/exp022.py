"""EXP-022: free diagnostics on missed people (plan frozen 2026-11-08 17:15).

    uv run --extra pose python -m edgelab.analysis.exp022 [--out-dir <dir>]      # single-threaded, ~1-2 min

Reference people = released AlphaPose test poses (ShanghaiTech), matched with the EXP-011 matcher
(cached box conf >= 0.25, IoU >= 0.3 with the keypoint box on the same frame).
(a) recall by reference-person height, YOLOv8n-pose baseline vs the EXP-013 fine-tuned detector (both L0_E0 caches).
(b) per reference track (video, person id): "never seen" = no frame of the track matched; "intermittent" = matched in some
    frames. Missed person-frames are split by that class. Rule: intermittent >= 30% of missed person-frames -> propose TSM.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from edgelab.analysis.exp009 import RELEASED_POSE_DIR, detector_boxes, iou_with_boxes, keypoint_box
from edgelab.analysis.exp011 import BIN_NAMES, CONF, IOU_MIN, height_bin, recall, wilson
from edgelab.pose.fill import SUFFIX, merge_parts

DATA = Path.home() / "thesis-data"
EXPERIMENT_DIR = Path(__file__).resolve().parents[3] / "EXP-022-miss-diagnostics"
CACHES = {"yolov8n-pose (EXP-003)": DATA / "exp003/cache/L0_E0", "fine-tuned (EXP-013)": DATA / "exp013-sht/cache/L0_E0"}
INTERMITTENT_THRESHOLD = 0.30
LONG_TRACK = 24  # STG-NF segment length: sensitivity restricted to tracks at least this long


def reference_tracks(pose_dir: Path = RELEASED_POSE_DIR) -> dict[str, dict[str, list[tuple[int, float, np.ndarray]]]]:
    """video -> {person id: [(frame, height, box)]} for every released pose that has a keypoint box."""
    result = {}
    for path in sorted(pose_dir.glob(f"*{SUFFIX}")):
        video = "_".join(path.name.split("_")[:2])
        tracks = result.setdefault(video, {})
        for person_id, track in json.loads(path.read_text()).items():
            out = []
            for frame, pose in merge_parts(track).items():
                box = keypoint_box(np.asarray(pose["keypoints"], dtype=float).reshape(-1, 3))
                if box is not None:
                    out.append((int(frame), float(box[3] - box[1]), box))
            if out:
                tracks[person_id] = out
    return result


def match_flags(tracks: dict, cache: Path, conf: float = CONF, iou_min: float = IOU_MIN) -> dict[str, dict[str, np.ndarray]]:
    """video -> {person id: bool array, one entry per reference person-frame: matched by a cached box}."""
    flags = {}
    for video, people in tracks.items():
        boxes = detector_boxes(cache / f"{video}.npz", conf)
        flags[video] = {}
        for person_id, entries in people.items():
            hit = []
            for frame, _, box in entries:
                found = boxes.get(frame)
                hit.append(found is not None and len(found) > 0 and bool((iou_with_boxes(box, found) >= iou_min).any()))
            flags[video][person_id] = np.array(hit, dtype=bool)
    return flags


def recall_from_flags(tracks: dict, flags: dict) -> dict:
    """{bin name: {"persons", "found"}} plus "all" (same layout as exp011.recall_by_bin)."""
    counts = np.zeros((len(BIN_NAMES), 2), dtype=np.int64)
    for video, people in tracks.items():
        for person_id, entries in people.items():
            for (_, height, _), hit in zip(entries, flags[video][person_id]):
                b = height_bin(height)
                counts[b, 0] += 1
                counts[b, 1] += int(hit)
    result = {name: {"persons": int(counts[i, 0]), "found": int(counts[i, 1])} for i, name in enumerate(BIN_NAMES)}
    result["all"] = {"persons": int(counts[:, 0].sum()), "found": int(counts[:, 1].sum())}
    return result


def miss_classes(tracks: dict, flags: dict, min_length: int = 1) -> dict:
    """Split missed person-frames by track class, over tracks with at least `min_length` person-frames.

    never_seen: tracks with no matched frame; intermittent: tracks with some (not all) matched frames; fully matched tracks have no miss.
    """
    out = {"tracks": 0, "tracks_never_seen": 0, "tracks_intermittent": 0, "tracks_fully_matched": 0,
           "person_frames": 0, "missed_person_frames": 0, "missed_in_never_seen": 0, "missed_in_intermittent": 0,
           "missed_in_intermittent_match_rate_ge_half": 0}
    by_bin = {name: {"missed_never_seen": 0, "missed_intermittent": 0} for name in BIN_NAMES}
    for video, people in tracks.items():
        for person_id, entries in people.items():
            hit = flags[video][person_id]
            if len(hit) < min_length:
                continue
            missed = int((~hit).sum())
            out["tracks"] += 1
            out["person_frames"] += len(hit)
            out["missed_person_frames"] += missed
            if hit.sum() == 0:
                kind = "never_seen"
            elif missed > 0:
                kind = "intermittent"
            else:
                kind = "fully_matched"
            out["tracks_" + kind] += 1
            if kind != "fully_matched":
                out["missed_in_" + kind] += missed
                if kind == "intermittent" and hit.mean() >= 0.5:
                    out["missed_in_intermittent_match_rate_ge_half"] += missed
                for (_, height, _), h in zip(entries, hit):
                    if not h:
                        by_bin[BIN_NAMES[height_bin(height)]]["missed_" + kind] += 1
    total = out["missed_person_frames"]
    out["intermittent_share"] = out["missed_in_intermittent"] / total if total else float("nan")
    out["never_seen_share"] = out["missed_in_never_seen"] / total if total else float("nan")
    out["intermittent_share_match_rate_ge_half"] = out["missed_in_intermittent_match_rate_ge_half"] / total if total else float("nan")
    out["missed_by_height_bin"] = by_bin
    return out


def decision(classes: dict) -> dict:
    share = classes["intermittent_share"]
    return {"intermittent_share": share, "threshold": INTERMITTENT_THRESHOLD, "propose_tsm": bool(share >= INTERMITTENT_THRESHOLD)}


def markdown(table: dict, classes: dict, long_classes: dict, decisions: dict) -> str:
    names = list(table)
    lines = ["# EXP-022 results: free diagnostics on missed people (generated by `edgelab.analysis.exp022`)\n",
             "Reference people: released AlphaPose test poses (ShanghaiTech, 107 videos); matcher = EXP-011 (cached box conf >= 0.25, IoU >= 0.3 with the "
             "keypoint box on the same frame). Detectors: " + "; ".join(f"{k} = `{v}`" for k, v in CACHES.items()) + ".\n",
             "## (a) Recall by reference-person height\n", "Cells: recall (found / persons).\n",
             "| detector | " + " | ".join(BIN_NAMES) + " | all |", "|---|" + "---|" * (len(BIN_NAMES) + 1)]
    for name in names:
        lines.append(f"| {name} | " + " | ".join(f"{recall(table[name][b]):.3f} ({table[name][b]['found']}/{table[name][b]['persons']})" for b in BIN_NAMES + ["all"]) + " |")
    lines += ["\nWilson 95% CI of recall:\n", "| detector | " + " | ".join(BIN_NAMES) + " |", "|---|" + "---|" * len(BIN_NAMES)]
    for name in names:
        lines.append(f"| {name} | " + " | ".join("%.3f-%.3f" % wilson(table[name][b]) for b in BIN_NAMES) + " |")
    lines += ["\nRecall gain of fine-tuned over baseline (points): " + ", ".join(
        f"{b} {recall(table[names[1]][b]) - recall(table[names[0]][b]):+.3f}" for b in BIN_NAMES + ["all"]) + "\n",
        "## (b) Missed person-frames: never-seen vs intermittent reference tracks\n",
        "A reference track = one released AlphaPose person id within a video. *Never seen*: no frame of the track is matched. "
        "*Intermittent*: matched in some frames, missed in others. Misses of never-seen tracks cannot be fixed by temporal smoothing; "
        "misses of intermittent tracks are the ones a temporal detector (TSM) could plausibly recover.\n",
        "| detector | tracks | never seen | intermittent | fully matched | person-frames | missed | missed in never-seen | missed in intermittent | **intermittent share** |",
        "|---|---|---|---|---|---|---|---|---|---|"]
    for name in names:
        c = classes[name]
        lines.append(f"| {name} | {c['tracks']} | {c['tracks_never_seen']} | {c['tracks_intermittent']} | {c['tracks_fully_matched']} | {c['person_frames']} | "
                     f"{c['missed_person_frames']} | {c['missed_in_never_seen']} ({c['never_seen_share']:.1%}) | {c['missed_in_intermittent']} | **{c['intermittent_share']:.1%}** |")
    lines += [f"\nSensitivity (descriptive, not the rule): tracks with >= {LONG_TRACK} person-frames only.\n",
              "| detector | tracks | never seen | intermittent | missed | missed in never-seen | missed in intermittent | intermittent share |", "|---|---|---|---|---|---|---|---|"]
    for name in names:
        c = long_classes[name]
        lines.append(f"| {name} | {c['tracks']} | {c['tracks_never_seen']} | {c['tracks_intermittent']} | {c['missed_person_frames']} | "
                     f"{c['missed_in_never_seen']} | {c['missed_in_intermittent']} | {c['intermittent_share']:.1%} |")
    lines += ["\nDescriptive: share of all missed person-frames lying in intermittent tracks that are matched in >= 50% of their frames "
              "(a temporal detector has most to work with there): " + "; ".join(f"{n} {classes[n]['intermittent_share_match_rate_ge_half']:.1%}" for n in names) + ".\n",
              "\nMissed person-frames by height bin and track class:\n", "| detector | class | " + " | ".join(BIN_NAMES) + " |", "|---|---|" + "---|" * len(BIN_NAMES)]
    for name in names:
        for kind in ("never_seen", "intermittent"):
            lines.append(f"| {name} | {kind} | " + " | ".join(str(classes[name]["missed_by_height_bin"][b]["missed_" + kind]) for b in BIN_NAMES) + " |")
    lines += ["\n## Decision (plan rule: intermittent >= 30% of missed person-frames -> propose a TSM experiment)\n", "| detector | intermittent share | threshold | propose TSM |", "|---|---|---|---|"]
    for name in names:
        d = decisions[name]
        lines.append(f"| {name} | {d['intermittent_share']:.1%} | {d['threshold']:.0%} | {'YES' if d['propose_tsm'] else 'NO'} |")
    lines += [f"\nPlan names the YOLOv8n-pose baseline cache for (b); that row decides: **{'propose a TSM experiment' if decisions[names[0]]['propose_tsm'] else 'do not propose a TSM experiment'}**. "
              "The fine-tuned row is reported for context.\n",
              "Caveats: the reference is AlphaPose output, not human annotation (EXP-011 threat); a track id that AlphaPose splits or merges changes the "
              "never-seen / intermittent classification; a track of length 1 can only be fully matched or never seen."]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=EXPERIMENT_DIR)
    args = parser.parse_args()
    tracks = reference_tracks()
    print("videos", len(tracks), "tracks", sum(len(p) for p in tracks.values()), "person-frames", sum(len(e) for p in tracks.values() for e in p.values()), flush=True)
    table, classes, long_classes, decisions = {}, {}, {}, {}
    for name, cache in CACHES.items():
        flags = match_flags(tracks, cache)
        table[name] = recall_from_flags(tracks, flags)
        classes[name] = miss_classes(tracks, flags)
        long_classes[name] = miss_classes(tracks, flags, min_length=LONG_TRACK)
        decisions[name] = decision(classes[name])
        print(name, json.dumps(table[name]["all"]), "intermittent share", round(classes[name]["intermittent_share"], 4), flush=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "miss_diagnostics.json").write_text(json.dumps({"recall": table, "classes": classes, "classes_long_tracks": long_classes, "decision": decisions}, indent=1))
    (args.out_dir / "results.md").write_text(markdown(table, classes, long_classes, decisions))


if __name__ == "__main__":
    main()
