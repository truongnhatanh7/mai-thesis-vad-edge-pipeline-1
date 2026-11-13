"""EXP-013: YOLO-pose training set from ShanghaiTech TRAINING videos + their released AlphaPose poses (pseudo-labels).

    uv run --extra pose python -m edgelab.pose.yolo_dataset --videos-dir <training/videos> --pose-dir <pose/train> \
        --out ~/thesis-data/exp013-data [--step 5] [--val-fraction 0.1] [--seed 0]

Leakage rule: only training videos and their released poses are read here; no test frame, test pose or label.

Label of one person (Ultralytics pose format, one line per person, coordinates divided by image width / height):
    0 cx cy w h  x1 y1 v1 ... x17 y17 v17
box = min/max of the VISIBLE keypoints (AlphaPose confidence > 0.3), grown by 10% of its width/height on every side,
clipped to the image. v = 2 for visible keypoints, 0 (and x = y = 0) otherwise. A person with fewer than 3 visible
keypoints, or a box without area, is dropped. A sampled frame with no label keeps an empty label file (background).
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from edgelab.pose.stgnf_format import read_stgnf_json

VISIBLE_CONF = 0.3
BOX_PAD = 0.10
MIN_VISIBLE = 3
FLIP_IDX = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]  # COCO-17 left <-> right, for flip augmentation


def label_line(keypoints: np.ndarray, width: int, height: int) -> str | None:
    """(17, 3) AlphaPose keypoints [x, y, conf] in pixels -> one YOLO-pose label line, or None if the person is dropped."""
    visible = keypoints[:, 2] > VISIBLE_CONF
    if visible.sum() < MIN_VISIBLE:
        return None
    xs, ys = keypoints[visible, 0], keypoints[visible, 1]
    x1, x2, y1, y2 = xs.min(), xs.max(), ys.min(), ys.max()
    pad_x, pad_y = BOX_PAD * (x2 - x1), BOX_PAD * (y2 - y1)
    x1, x2 = max(0.0, x1 - pad_x), min(float(width), x2 + pad_x)
    y1, y2 = max(0.0, y1 - pad_y), min(float(height), y2 + pad_y)
    if x2 - x1 < 1 or y2 - y1 < 1:
        return None
    box = [(x1 + x2) / 2 / width, (y1 + y2) / 2 / height, (x2 - x1) / width, (y2 - y1) / height]
    parts = ["0"] + [f"{v:.6f}" for v in box]
    for (x, y, _), is_visible in zip(keypoints, visible):
        if is_visible:
            parts += [f"{min(max(x / width, 0.0), 1.0):.6f}", f"{min(max(y / height, 0.0), 1.0):.6f}", "2"]
        else:
            parts += ["0", "0", "0"]
    return " ".join(parts)


def released_poses_by_frame(pose_json: Path) -> dict[int, list[np.ndarray]]:
    """STG-NF tracked_person json -> {0-based frame: [(17, 3) keypoints of each person on that frame]}."""
    by_frame: dict[int, list[np.ndarray]] = {}
    for frames, keypoints, _ in read_stgnf_json(pose_json).values():
        for frame, kp in zip(frames, keypoints):
            by_frame.setdefault(int(frame), []).append(kp)
    return by_frame


def split_by_video(videos: list[str], val_fraction: float = 0.1, seed: int = 0) -> tuple[list[str], list[str]]:
    """Whole videos go to train or val. Per scene (name prefix before '_') round(val_fraction * n) videos, at least 1."""
    rng = np.random.default_rng(seed)
    scenes: dict[str, list[str]] = {}
    for video in sorted(videos):
        scenes.setdefault(video.split("_")[0], []).append(video)
    val = []
    for scene in sorted(scenes):
        members = scenes[scene]
        n_val = max(1, round(val_fraction * len(members)))
        val += [members[i] for i in rng.permutation(len(members))[:n_val]]
    return sorted(set(videos) - set(val)), sorted(val)


def pose_file(pose_dir: Path, video: str) -> Path:
    return pose_dir / f"{video}_alphapose_tracked_person.json"


def build_video(video_path: Path, pose_json: Path, out: Path, split: str, step: int) -> dict:
    """Write every `step`-th frame (0, step, 2*step, ...) as jpg + label file. Returns counts for the manifest."""
    video = video_path.stem
    poses = released_poses_by_frame(pose_json)
    (out / "images" / split).mkdir(parents=True, exist_ok=True)
    (out / "labels" / split).mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video_path))
    header_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame, images, persons, empty = 0, 0, 0, 0
    while True:
        ok, image = capture.read()
        if not ok:
            break
        if frame % step == 0:
            height, width = image.shape[:2]
            lines = [line for kp in poses.get(frame, []) if (line := label_line(kp, width, height))]
            name = f"{video}_{frame:04d}"
            cv2.imwrite(str(out / "images" / split / f"{name}.jpg"), image, [cv2.IMWRITE_JPEG_QUALITY, 95])
            (out / "labels" / split / f"{name}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
            images += 1
            persons += len(lines)
            empty += int(not lines)
        frame += 1
    capture.release()
    last_pose_frame = max(poses) if poses else -1
    return {"video": video, "split": split, "frames_decoded": frame, "frames_header": header_frames,
            "last_pose_frame": last_pose_frame, "images": images, "persons": persons, "empty_images": empty}


def write_yaml(out: Path) -> Path:
    path = out / "dataset.yaml"
    path.write_text(f"path: {out}\ntrain: images/train\nval: images/val\nkpt_shape: [17, 3]\nflip_idx: {FLIP_IDX}\n"
                    "names:\n  0: person\n")
    return path


def build_dataset(videos_dir: Path, pose_dir: Path, out: Path, step: int = 5, val_fraction: float = 0.1,
                  seed: int = 0) -> dict:
    videos = sorted(p.stem for p in videos_dir.glob("*.avi") if pose_file(pose_dir, p.stem).exists())
    train, val = split_by_video(videos, val_fraction, seed)
    out.mkdir(parents=True, exist_ok=True)
    (out / "split.json").write_text(json.dumps({"train": train, "val": val, "seed": seed, "step": step}, indent=1))
    rows = []
    for split, names in (("train", train), ("val", val)):
        for video in names:
            rows.append(build_video(videos_dir / f"{video}.avi", pose_file(pose_dir, video), out, split, step))
            print(json.dumps(rows[-1]), flush=True)
    write_yaml(out)
    manifest = {"step": step, "seed": seed, "videos_train": len(train), "videos_val": len(val), "rows": rows,
                "images_train": sum(r["images"] for r in rows if r["split"] == "train"),
                "images_val": sum(r["images"] for r in rows if r["split"] == "val"),
                "persons": sum(r["persons"] for r in rows), "empty_images": sum(r["empty_images"] for r in rows),
                "videos_header_mismatch": [r["video"] for r in rows if r["frames_decoded"] != r["frames_header"]],
                "videos_pose_beyond_video": [r["video"] for r in rows if r["last_pose_frame"] >= r["frames_decoded"]]}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos-dir", type=Path, required=True)
    parser.add_argument("--pose-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--step", type=int, default=5)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    manifest = build_dataset(args.videos_dir, args.pose_dir, args.out, args.step, args.val_fraction, args.seed)
    print(json.dumps({k: v for k, v in manifest.items() if k != "rows"}))


if __name__ == "__main__":
    main()
