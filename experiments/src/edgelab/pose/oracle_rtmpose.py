"""EXP-015: keep the released AlphaPose boxes, track ids and frames; re-estimate only the keypoints with another model.

    uv run --extra pose python -m edgelab.pose.oracle_rtmpose --model rtmpose-s --out <pose dir> [--videos 01_0014 ...]

For every person-frame of a released `<video>_alphapose_tracked_person.json`:
  box = min/max of the released keypoints with conf > 0 -> crop (1.25x padding, aspect 192:256, see rtmpose.py)
  -> new keypoints from `--model` (rtmpose-s | rtmpose-t | yolov8n-pose) -> same JSON layout, same ids, same frame keys,
  same `scores` field. Nothing else changes, so any AUC change comes from the keypoints alone.

yolov8n-pose (O3) is run on the same 256x192 crop and the detection with the highest IoU against the oracle box is taken;
no detection with IoU >= 0.3 means that person-frame is dropped (that is part of what O3 measures).
"""

import argparse
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np

from edgelab.pose.rtmpose import (INPUT_H, INPUT_W, RtmPose, box_to_center_scale, crop_matrix, crop_person, crop_to_image,
                                  keypoints_to_box)
from edgelab.pose.stgnf_eval import EXPERIMENTS_ROOT

RELEASED_DIR = EXPERIMENTS_ROOT / "third_party/STG-NF/data/ShanghaiTech/pose/test"
FRAMES_ROOT = Path.home() / "thesis-data/shanghaitech/shanghaitech/testing/frames"
MODELS_DIR = Path.home() / "thesis-data/exp015/models"
YOLO_POSE_WEIGHTS = EXPERIMENTS_ROOT / "artifacts/EXP-001/models/yolov8n-pose.pt"
SUFFIX = "_alphapose_tracked_person.json"
MIN_BOX_PX = 2.0  # a box thinner than this has no body in it; such a person-frame is kept as-is and counted
YOLO_MATCH_IOU = 0.3


def person_frames(clip: dict) -> list[tuple[str, str, dict]]:
    """Released clip -> [(person id, frame key, entry)]. Some released files store a person as a list of one-frame dicts."""
    rows = []
    for pid, person in clip.items():
        if isinstance(person, list):
            merged = {}
            for part in person:
                merged.update(part)
            person = merged
        rows += [(pid, key, entry) for key, entry in person.items()]
    return rows


def iou(box_a: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    x1, y1 = np.maximum(box_a[0], boxes[:, 0]), np.maximum(box_a[1], boxes[:, 1])
    x2, y2 = np.minimum(box_a[2], boxes[:, 2]), np.minimum(box_a[3], boxes[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return inter / (area_a + area_b - inter + 1e-9)


class YoloOnCrop:
    """O3: YOLOv8n-pose on each oracle crop; keypoints of the detection that best overlaps the oracle box."""

    def __init__(self, weights: Path = YOLO_POSE_WEIGHTS, imgsz: int = INPUT_H, device: str = "cpu"):
        from ultralytics import YOLO

        self.model, self.imgsz, self.device = YOLO(str(weights)), imgsz, device

    def predict(self, img_bgr: np.ndarray, boxes_xyxy: np.ndarray) -> np.ndarray:
        """-> (n, 17, 3) in image pixels; rows of NaN where nothing matched."""
        out = np.full((len(boxes_xyxy), 17, 3), np.nan, np.float32)
        if len(boxes_xyxy) == 0:
            return out
        centers, scales = zip(*(box_to_center_scale(b) for b in boxes_xyxy))
        centers, scales = np.array(centers), np.array(scales)
        crops = [crop_person(img_bgr, c, s) for c, s in zip(centers, scales)]
        results = self.model.predict(crops, imgsz=self.imgsz, conf=0.01, iou=0.7, device=self.device, verbose=False)
        for i, (result, box, center, scale) in enumerate(zip(results, boxes_xyxy, centers, scales)):
            if len(result.boxes) == 0:
                continue
            matrix = crop_matrix(center, scale)
            corners = np.array([[box[0], box[1]], [box[2], box[3]]]) @ matrix[:, :2].T + matrix[:, 2]
            overlaps = iou(corners.reshape(-1), result.boxes.xyxy.cpu().numpy())
            best = int(overlaps.argmax())
            if overlaps[best] < YOLO_MATCH_IOU:
                continue
            keypoints = result.keypoints.data.cpu().numpy()[best]  # (17, 3) in crop pixels
            xy = crop_to_image(keypoints[None, :, :2], center[None], scale[None])[0]
            out[i] = np.concatenate([xy, keypoints[:, 2:]], axis=1)
        return out


def make_estimator(name: str, threads: int):
    if name in ("rtmpose-s", "rtmpose-t"):
        return RtmPose(MODELS_DIR / f"{name}-body7-256x192.onnx", threads=threads)
    if name == "yolov8n-pose":
        return YoloOnCrop()
    raise ValueError(f"unknown model {name!r}")


def frame_paths(frames_dir: Path) -> dict[int, Path]:
    return {int(p.stem): p for p in frames_dir.glob("*.jpg")}


def rerun_clip(clip: dict, frames: dict[int, Path], estimator) -> tuple[dict, dict]:
    """Released clip dict -> (new clip dict with replaced keypoints, stats)."""
    by_frame: dict[int, list] = {}
    for pid, key, entry in person_frames(clip):
        by_frame.setdefault(int(key), []).append((pid, key, entry))
    new: dict[str, dict] = {}
    stats = {"person_frames": 0, "no_box": 0, "dropped": 0, "boxes_px_h": []}
    for frame in sorted(by_frame):
        img = cv2.imread(str(frames[frame]))
        items = by_frame[frame]
        released = [np.asarray(e["keypoints"], dtype=np.float32).reshape(17, 3) for _, _, e in items]
        boxes = [keypoints_to_box(k) for k in released]
        usable = [b is not None and (b[2] - b[0]) >= MIN_BOX_PX and (b[3] - b[1]) >= MIN_BOX_PX for b in boxes]
        estimates = estimator.predict(img, np.array([b for b, u in zip(boxes, usable) if u]).reshape(-1, 4))
        estimates = iter(estimates)
        for (pid, key, entry), ok, box in zip(items, usable, boxes):
            stats["person_frames"] += 1
            if ok:
                keypoints = next(estimates)
                stats["boxes_px_h"].append(float(box[3] - box[1]))
            else:  # no usable oracle box: keep the released pose (counted, expected to be ~0)
                keypoints = np.asarray(entry["keypoints"], dtype=float).reshape(17, 3)
                stats["no_box"] += 1
            if np.isnan(keypoints).any():
                stats["dropped"] += 1
                continue
            new.setdefault(pid, {})[key] = {"keypoints": [float(v) for v in keypoints.reshape(-1)],
                                            "scores": float(entry["scores"])}
    return new, stats


def rerun_video(video: str, estimator, out_dir: Path, released_dir: Path = RELEASED_DIR,
                frames_root: Path = FRAMES_ROOT) -> dict:
    clip = json.loads((released_dir / f"{video}{SUFFIX}").read_text())
    new, stats = rerun_clip(clip, frame_paths(frames_root / video), estimator)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"{video}{SUFFIX}"
    tmp = target.with_name(target.name + ".partial")
    tmp.write_text(json.dumps(new))
    os.replace(tmp, target)  # a killed run never leaves a half-written file that would count as done
    heights = stats.pop("boxes_px_h")
    stats["median_box_px_h"] = float(np.median(heights)) if heights else None
    return stats


def released_videos(released_dir: Path = RELEASED_DIR) -> list[str]:
    return sorted(p.name[: -len(SUFFIX)] for p in released_dir.glob(f"*{SUFFIX}"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["rtmpose-s", "rtmpose-t", "yolov8n-pose"])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--videos", nargs="*", default=None)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    estimator = make_estimator(args.model, args.threads)
    for video in args.videos or released_videos():
        if (args.out / f"{video}{SUFFIX}").exists():
            continue  # resumable
        t0 = time.perf_counter()
        stats = rerun_video(video, estimator, args.out)
        print(json.dumps({"video": video, **stats, "wall_s": round(time.perf_counter() - t0, 1)}), flush=True)


if __name__ == "__main__":
    main()
