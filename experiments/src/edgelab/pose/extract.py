"""YOLO pose + built-in tracker over ShanghaiTech test frame folders -> STG-NF JSON files, on CPU.

    uv run python -m edgelab.pose.extract --frames-root <testing/frames> --out-dir <dir> [--videos 01_0014 ...]

Ultralytics is AGPL-3.0: research use only (DECISIONS.md D11).
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from edgelab.pose.stgnf_format import poses_from_ultralytics_result, write_stgnf_json

DEFAULTS = {"model": "yolov8n-pose.pt", "tracker": "botsort.yaml", "imgsz": 640, "conf": 0.25, "iou": 0.7}


def iter_frames(source: Path):
    """(frame index, BGR image) from a folder of NNN.jpg (test split) or a video file (train split .avi)."""
    if source.is_dir():
        for frame_file in sorted(source.glob("*.jpg")):
            yield int(frame_file.stem), cv2.imread(str(frame_file))
        return
    capture = cv2.VideoCapture(str(source))
    index = 0
    while True:
        ok, image = capture.read()
        if not ok:
            break
        yield index, image
        index += 1
    capture.release()


def extract_video(model, source: Path, tracker: str, imgsz: int, conf: float, iou: float, threads: int | None = None):
    """Returns (poses, per-frame timing dicts in ms). Tracker state is reset per video."""
    poses, timings = [], []
    frames = iter_frames(source)
    while True:
        t0 = time.perf_counter()
        item = next(frames, None)
        if item is None:
            break
        frame, image = item
        t1 = time.perf_counter()
        result = model.track(image, persist=len(timings) > 0, tracker=tracker, imgsz=imgsz, conf=conf, iou=iou,
                             device="cpu", verbose=False)[0]
        t2 = time.perf_counter()
        if threads and len(timings) == 0:
            torch.set_num_threads(threads)  # Ultralytics select_device() resets it to min(8, cpus-1) on setup
        poses += poses_from_ultralytics_result(result, frame)
        timings.append({"read_ms": (t1 - t0) * 1e3, "track_call_ms": (t2 - t1) * 1e3, **result.speed})
    return poses, timings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames-root", type=Path, required=True, help="folder of frame folders or of .avi videos")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--videos", nargs="*", default=None)
    parser.add_argument("--model", default=DEFAULTS["model"])
    parser.add_argument("--tracker", default=DEFAULTS["tracker"])
    parser.add_argument("--imgsz", type=int, default=DEFAULTS["imgsz"])
    parser.add_argument("--conf", type=float, default=DEFAULTS["conf"])
    parser.add_argument("--iou", type=float, default=DEFAULTS["iou"])
    parser.add_argument("--threads", type=int, default=None)
    args = parser.parse_args()

    from ultralytics import YOLO

    if args.threads:
        torch.set_num_threads(args.threads)
    source_tag = Path(args.model).stem + "-" + Path(args.tracker).stem
    videos = args.videos or sorted(p.name for p in args.frames_root.iterdir() if p.is_dir() or p.suffix == ".avi")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    timing_log = args.out_dir / "timings.jsonl"
    for video in videos:
        out_file = args.out_dir / f"{Path(video).stem}_{source_tag}_tracked_person.json"
        if out_file.exists():
            continue  # resumable
        model = YOLO(args.model)  # fresh model per video = fresh tracker state
        t0 = time.perf_counter()
        poses, timings = extract_video(model, args.frames_root / video, args.tracker, args.imgsz, args.conf, args.iou,
                                       args.threads)
        wall = time.perf_counter() - t0
        write_stgnf_json(poses, out_file)
        summary = {"video": Path(video).stem, "frames": len(timings), "poses": len(poses),
                   "tracks": len({p.track_id for p in poses}), "wall_s": wall, "threads": torch.get_num_threads(),
                   **{f"mean_{k}": float(np.mean([t[k] for t in timings])) for k in timings[0]}}
        with open(timing_log, "a") as f:
            f.write(json.dumps(summary) + "\n")
        print(json.dumps(summary))


if __name__ == "__main__":
    main()
