"""EXP-013: are the released training poses aligned with the decoded video frames (0-based)?

    uv run --extra pose python -m edgelab.pose.align_check --videos-dir <training/videos> --pose-dir <pose/train> --out <json>

For sampled frames f of some TRAINING videos: detect people with the original yolov8n-pose on decoded frame f + s
(s = -2..+2) and compare with the released poses of frame f. If the released ids are 0-based like OpenCV's decode
order, the mean best IoU peaks at s = 0. Only training videos and training poses are read.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from edgelab.pose.detcache import WEIGHTS, detect
from edgelab.pose.yolo_dataset import VISIBLE_CONF, pose_file, released_poses_by_frame

SHIFTS = (-2, -1, 0, 1, 2)


def pose_box(keypoints: np.ndarray) -> np.ndarray | None:
    """xyxy box of the visible keypoints (conf > 0.3), or None."""
    visible = keypoints[:, 2] > VISIBLE_CONF
    if visible.sum() < 3:
        return None
    xs, ys = keypoints[visible, 0], keypoints[visible, 1]
    return np.array([xs.min(), ys.min(), xs.max(), ys.max()])


def iou(a: np.ndarray, b: np.ndarray) -> float:
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = w * h
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def best_ious(released_boxes: list[np.ndarray], detected_boxes: np.ndarray) -> list[float]:
    """For each released box the best IoU with any detected box (0 if there is no detection)."""
    return [max((iou(r, d) for d in detected_boxes), default=0.0) for r in released_boxes]


def check_video(model, video_path: Path, pose_json: Path, n_frames: int, device: str, rng) -> dict[int, list[float]]:
    poses = released_poses_by_frame(pose_json)
    capture = cv2.VideoCapture(str(video_path))
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    candidates = [f for f in range(3, total - 3) if poses.get(f)]
    chosen = set(rng.choice(candidates, size=min(n_frames, len(candidates)), replace=False).tolist())
    needed = {f + s for f in chosen for s in SHIFTS}
    frames, index = {}, 0
    while index <= max(needed):
        ok, image = capture.read()
        if not ok:
            break
        if index in needed:
            frames[index] = image
        index += 1
    capture.release()
    out = {s: [] for s in SHIFTS}
    for f in sorted(chosen):
        released = [b for kp in poses[f] if (b := pose_box(kp)) is not None]
        for s in SHIFTS:
            boxes, conf, _, _, _ = detect(model, frames[f + s], device, conf=0.25)
            out[s] += best_ious(released, boxes)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos-dir", type=Path, required=True)
    parser.add_argument("--pose-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--every", type=int, default=25, help="check every N-th training video")
    parser.add_argument("--frames", type=int, default=12, help="sampled frames per video")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    from ultralytics import YOLO

    model = YOLO(str(WEIGHTS))
    rng = np.random.default_rng(args.seed)
    videos = sorted(p.stem for p in args.videos_dir.glob("*.avi") if pose_file(args.pose_dir, p.stem).exists())[::args.every]
    pooled = {s: [] for s in SHIFTS}
    for video in videos:
        result = check_video(model, args.videos_dir / f"{video}.avi", pose_file(args.pose_dir, video), args.frames, args.device, rng)
        for s in SHIFTS:
            pooled[s] += result[s]
        print(video, {s: round(float(np.mean(v)), 3) for s, v in result.items()}, flush=True)
    summary = {"videos": videos, "released_boxes_per_shift": len(pooled[0]),
               "mean_best_iou": {s: float(np.mean(v)) for s, v in pooled.items()},
               "share_iou_ge_0.3": {s: float(np.mean(np.array(v) >= 0.3)) for s, v in pooled.items()}}
    summary["peak_shift"] = max(summary["mean_best_iou"], key=summary["mean_best_iou"].get)
    args.out.write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
