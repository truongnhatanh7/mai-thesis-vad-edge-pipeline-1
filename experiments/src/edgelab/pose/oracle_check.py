"""EXP-015 sanity check: does a re-estimated pose file use the same keypoint order as the released one?

    uv run --extra pose python -m edgelab.pose.oracle_check --new <pose dir> --videos 01_0014 06_0144 --png-dir <dir>

(1) per joint: mean distance between new and released keypoints, divided by the box height of the person;
    next to it the same number with the new left/right joints swapped. The first must be clearly smaller than the second.
(2) PNG of a few person-frames: released skeleton in red, new skeleton in green, drawn on the frame.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from edgelab.pose.oracle_rtmpose import FRAMES_ROOT, RELEASED_DIR, SUFFIX, person_frames
from edgelab.pose.rtmpose import keypoints_to_box

COCO_NAMES = ["nose", "l_eye", "r_eye", "l_ear", "r_ear", "l_shoulder", "r_shoulder", "l_elbow", "r_elbow", "l_wrist",
              "r_wrist", "l_hip", "r_hip", "l_knee", "r_knee", "l_ankle", "r_ankle"]
LEFT_RIGHT_SWAP = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
SKELETON = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]


def paired_poses(video: str, new_dir: Path, released_dir: Path = RELEASED_DIR):
    """-> list of (frame, released (17, 3), new (17, 3)) for person-frames present in both files."""
    released = json.loads((released_dir / f"{video}{SUFFIX}").read_text())
    new = json.loads((new_dir / f"{video}{SUFFIX}").read_text())
    new_index = {(pid, key): e for pid, key, e in person_frames(new)}
    pairs = []
    for pid, key, entry in person_frames(released):
        if (pid, key) in new_index:
            pairs.append((int(key), np.asarray(entry["keypoints"], float).reshape(17, 3),
                          np.asarray(new_index[(pid, key)]["keypoints"], float).reshape(17, 3)))
    return pairs


def joint_distance_report(pairs) -> dict:
    """Mean |new - released| / box height per joint, and the same with new left/right swapped."""
    same, swapped = [], []
    for _, ref, new in pairs:
        box = keypoints_to_box(ref)
        if box is None or box[3] - box[1] < 2:
            continue
        height = box[3] - box[1]
        same.append(np.linalg.norm(new[:, :2] - ref[:, :2], axis=1) / height)
        swapped.append(np.linalg.norm(new[LEFT_RIGHT_SWAP, :2] - ref[:, :2], axis=1) / height)
    used = len(same)
    same, swapped = np.mean(same, axis=0), np.mean(swapped, axis=0)
    return {"person_frames": used, "joint_names": COCO_NAMES, "same_order": same.tolist(),
            "left_right_swapped": swapped.tolist(), "mean_same": float(same.mean()), "mean_swapped": float(swapped.mean()),
            "order_ok": bool(same.mean() < swapped.mean())}


def draw_pose(img: np.ndarray, keypoints: np.ndarray, color) -> None:
    for a, b in SKELETON:
        cv2.line(img, tuple(int(v) for v in keypoints[a, :2]), tuple(int(v) for v in keypoints[b, :2]), color, 1)
    for x, y in keypoints[:, :2]:
        cv2.circle(img, (int(x), int(y)), 2, color, -1)


def plot_pairs(video: str, pairs, png: Path, n: int = 3, frames_root: Path = FRAMES_ROOT) -> None:
    """One PNG: n frames spread over the video, each with every pair of that frame drawn (red released, green new)."""
    frames = sorted({f for f, _, _ in pairs})
    chosen = [frames[int(i)] for i in np.linspace(0, len(frames) - 1, n)]
    tiles = []
    for frame in chosen:
        img = cv2.imread(str(frames_root / video / f"{frame:03d}.jpg"))
        for f, ref, new in pairs:
            if f == frame:
                draw_pose(img, ref, (0, 0, 255))
                draw_pose(img, new, (0, 200, 0))
        cv2.putText(img, f"{video} f{frame} red=released green=new", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        tiles.append(img)
    png.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(png), np.concatenate(tiles, axis=0))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--new", type=Path, required=True)
    parser.add_argument("--videos", nargs="+", required=True)
    parser.add_argument("--png-dir", type=Path, default=None)
    args = parser.parse_args()
    pairs = []
    for video in args.videos:
        video_pairs = paired_poses(video, args.new)
        pairs += video_pairs
        if args.png_dir:
            plot_pairs(video, video_pairs, args.png_dir / f"{args.new.name}_{video}.png")
    report = joint_distance_report(pairs)
    print(json.dumps({k: (np.round(v, 3).tolist() if isinstance(v, list) and k != "joint_names" else v)
                      for k, v in report.items()}))


if __name__ == "__main__":
    main()
