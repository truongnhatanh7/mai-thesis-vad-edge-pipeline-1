"""EXP-015 driver: score the oracle-box pose files with the frozen STG-NF checkpoint, then write results.md.

    uv run --extra pose python -m edgelab.analysis.exp015 score --name rtmpose-s [--threads 4]   # resumable
    uv run --extra pose python -m edgelab.analysis.exp015 report                                  # results.md + verdicts

Arms: `released` (AlphaPose poses, the 85.94 reference), `rtmpose-s`, `rtmpose-t`, `yolov8n-pose` (O3).
Pose files: ~/thesis-data/exp015/poses/<arm>/ ; rows: ~/thesis-data/exp015/rows.jsonl ; per-frame scores: .../scores/<arm>/.
"""

import argparse
import json
from pathlib import Path

from edgelab.analysis.exp003 import diff
from edgelab.analysis.exp009 import load_scores_file
from edgelab.pose.oracle_check import joint_distance_report, paired_poses
from edgelab.pose.oracle_rtmpose import RELEASED_DIR, SUFFIX, person_frames, released_videos
from edgelab.pose.score_dir import score_pose_dir

DATA = Path.home() / "thesis-data/exp015"
ROWS = DATA / "rows.jsonl"
EXPERIMENT_DIR = Path(__file__).resolve().parents[3] / "EXP-015-oracle-rtmpose"
ARMS = ("released", "rtmpose-s", "rtmpose-t", "yolov8n-pose")
O1_THRESHOLD, O2_THRESHOLD = 83.0, 80.0  # plan.md, paper variant, AUC in %


def pose_dir(arm: str) -> Path:
    return RELEASED_DIR if arm == "released" else DATA / "poses" / arm


def count_person_frames(directory: Path) -> int:
    return sum(len(person_frames(json.loads((directory / f"{v}{SUFFIX}").read_text()))) for v in released_videos())


def keypoint_distance(arm: str) -> float | None:
    """Mean over person-frames and joints of |new - released| / box height (released box), all 107 videos."""
    if arm == "released":
        return None
    pairs = [p for v in released_videos() for p in paired_poses(v, pose_dir(arm))]
    return joint_distance_report(pairs)["mean_same"]


def score_arm(arm: str, threads: int | None) -> list[dict]:
    done = {r["arm"] for r in read_rows()}
    if arm in done:
        return []
    results = score_pose_dir(pose_dir(arm), DATA / "scores" / arm,
                             {"level": "L0", "enhance": arm, "c": None, "tracker": "oracle-boxes", "device": "cpu"},
                             threads=threads, experiment="EXP-015")
    rows = [{"arm": arm, "variant": v, **r, "person_frames": count_person_frames(pose_dir(arm)),
             "keypoint_distance": keypoint_distance(arm)} for v, r in results.items()]
    with open(ROWS, "a") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return rows


def read_rows() -> list[dict]:
    return [json.loads(line) for line in ROWS.read_text().splitlines() if line.strip()] if ROWS.exists() else []


def row_of(rows: list[dict], arm: str, variant: str) -> dict:
    return next(r for r in rows if r["arm"] == arm and r["variant"] == variant)


def verdicts(rows: list[dict]) -> dict:
    """O1, O2 pass rules on the paper-variant AUC (in %); O3 is descriptive. Also which RTMPose model EXP-016 uses."""
    auc = {arm: 100 * row_of(rows, arm, "paper")["auc"] for arm in ARMS if any(r["arm"] == arm for r in rows)}
    o1, o2 = auc["rtmpose-s"] >= O1_THRESHOLD, auc["rtmpose-t"] >= O2_THRESHOLD
    # plan.md of EXP-016: "its best RTMPose model if O2 passes, else RTMPose-s"
    chosen = max(("rtmpose-s", "rtmpose-t"), key=lambda a: auc[a]) if o2 else "rtmpose-s"
    return {"auc_paper": auc, "O1_pass": bool(o1), "O2_pass": bool(o2), "exp016_pose_model": chosen}


def report() -> str:
    rows = read_rows()
    verdict = verdicts(rows)
    base = load_scores_file(DATA / "scores/released/frame_scores_paper.npz")
    lines = ["| arm | paper AUC [95% CI] | live AUC [95% CI] | paper diff vs released (paired 95% CI) | anomalous-frame coverage | "
             "person-frames | mean joint distance / box height |", "|---|---|---|---|---|---|---|"]
    for arm in ARMS:
        if not any(r["arm"] == arm for r in rows):
            continue
        p, l = row_of(rows, arm, "paper"), row_of(rows, arm, "live")
        if arm == "released":
            delta = "-"
        else:
            d = diff(base, load_scores_file(DATA / f"scores/{arm}/frame_scores_paper.npz"), n_resamples=1000)
            delta = f"{100 * d['diff']:+.2f} [{100 * d['ci'][0]:+.2f}, {100 * d['ci'][1]:+.2f}]"
        dist = "-" if p["keypoint_distance"] is None else f"{p['keypoint_distance']:.3f}"
        lines.append(f"| {arm} | {100 * p['auc']:.2f} [{100 * p['ci95_low']:.2f}, {100 * p['ci95_high']:.2f}] | "
                     f"{100 * l['auc']:.2f} [{100 * l['ci95_low']:.2f}, {100 * l['ci95_high']:.2f}] | {delta} | "
                     f"{p['coverage_anomalous']:.3f} | {p['person_frames']} | {dist} |")
    auc = verdict["auc_paper"]
    lines += ["", f"O1 (RTMPose-s paper AUC >= {O1_THRESHOLD}): {'PASS' if verdict['O1_pass'] else 'FAIL'} ({auc['rtmpose-s']:.2f})",
              f"O2 (RTMPose-t paper AUC >= {O2_THRESHOLD}): {'PASS' if verdict['O2_pass'] else 'FAIL'} ({auc['rtmpose-t']:.2f})"]
    if "yolov8n-pose" in auc:
        lines.append(f"O3 (descriptive): YOLOv8n-pose on the oracle crops {auc['yolov8n-pose']:.2f} paper AUC, "
                     f"{read_value(rows, 'yolov8n-pose', 'person_frames')} person-frames kept of {read_value(rows, 'released', 'person_frames')}")
    lines.append(f"EXP-016 pose model by the plan's rule: {verdict['exp016_pose_model']}")
    (EXPERIMENT_DIR / "verdicts.json").write_text(json.dumps(verdict, indent=1))
    return "\n".join(lines)


def read_value(rows: list[dict], arm: str, key: str):
    return row_of(rows, arm, "paper")[key]


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    score = sub.add_parser("score")
    score.add_argument("--name", required=True, choices=ARMS)
    score.add_argument("--threads", type=int, default=None)
    sub.add_parser("report")
    args = parser.parse_args()
    if args.command == "score":
        for row in score_arm(args.name, args.threads):
            print(json.dumps({k: row[k] for k in ("arm", "variant", "auc", "coverage_anomalous", "person_frames")}))
    else:
        print(report())


if __name__ == "__main__":
    main()
