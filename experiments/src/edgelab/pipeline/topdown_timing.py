"""Per-frame timing of the EXP-016 top-down pipeline. Meant to run inside the VM `edge-os` (same harness as EXP-004).

    read jpg -> darken -> [YOLOv8n person detector, ONNX Runtime CPU, only on keyframes] -> BoT-SORT (update or Kalman
    predict) -> RTMPose on every tracked box (ONNX Runtime CPU) -> STG-NF on each new 24-pose segment

    sudo env PYTHONPATH=<experiments>/src ~/edgelab-vm/.venv/bin/python -m edgelab.pipeline.topdown_timing \
        --frames-root <dir with one folder of NNN.jpg per video> --videos-file timing_subset.txt --level L0 --c 0.25 \
        --imgsz 640 --k 3 --pose-model rtmpose-s --out run.csv --tag T3-640-K3

One CSV row per frame, times in ms. service_ms = total_ms - darken_ms (darkening only simulates the sensor).
n_boxes = boxes given to the pose model in that frame (what the pose cost scales with).
"""

import argparse
import csv
import json
import os
import platform
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from edgelab.pipeline.ort_pose import IOU, letterbox, make_session
from edgelab.pipeline.stgnf_live import LiveStgnf, load_model, score_segments
from edgelab.pose.replay import Detections, build_tracker
from edgelab.pose.rtmpose import RtmPose
from edgelab.pose.topdown import clip_boxes, predict_only
from edgelab.preprocess.darken import darken_level

HOST_DATA = Path("/Users/nanh/thesis-data")  # the VM mounts the host home at the same path (its own Path.home() is different)
# The VM only mounts vm-exchange (rw) and experiments/ (ro): the models are copies of ~/thesis-data/exp016/models and
# ~/thesis-data/exp015/models under vm-exchange/models (2026-11-08 fix: the first version pointed at host-only paths).
MODELS = HOST_DATA / "vm-exchange/models"
DET_ONNX = {640: MODELS / "exp016/yolov8n-640-384x640.onnx", 960: MODELS / "exp016/yolov8n-960-544x960.onnx"}
RTM_DIR = MODELS / "exp015"
CONFIG_COLUMNS = ["tag", "machine", "level", "c", "imgsz", "k", "pose_model", "threads"]
TIME_COLUMNS = ["read_ms", "darken_ms", "det_pre_ms", "det_infer_ms", "det_post_ms", "track_ms", "pose_ms", "flow_ms",
                "service_ms", "total_ms"]
COLUMNS = CONFIG_COLUMNS + ["video", "frame", "is_key"] + TIME_COLUMNS + ["n_dets", "n_boxes", "n_new_segments"]


def decode_persons(output: np.ndarray, conf: float, ratio: float, pad: tuple[int, int], shape: tuple[int, int],
                   iou: float = IOU) -> tuple[np.ndarray, np.ndarray]:
    """YOLOv8 detection output (1, 4 + 80 classes, N) -> person boxes xyxy (n, 4) in image pixels, conf (n,), by conf.

    Same rule as Ultralytics with classes=[0]: an anchor counts when its best class is person and the person
    probability is >= conf; then NMS at `iou`.
    """
    rows = output[0].T
    rows = rows[(rows[:, 4:].argmax(axis=1) == 0) & (rows[:, 4] >= conf)]
    if len(rows) == 0:
        return np.zeros((0, 4), np.float32), np.zeros(0, np.float32)
    cx, cy, w, h = rows[:, :4].T
    topleft = np.stack([cx - w / 2, cy - h / 2, w, h], axis=1)
    keep = np.asarray(cv2.dnn.NMSBoxes(topleft.tolist(), rows[:, 4].tolist(), conf, iou)).reshape(-1)
    rows, topleft = rows[keep], topleft[keep]
    order = np.argsort(-rows[:, 4])
    rows, topleft = rows[order], topleft[order]
    height, width = shape
    xyxy = np.stack([topleft[:, 0], topleft[:, 1], topleft[:, 0] + topleft[:, 2], topleft[:, 1] + topleft[:, 3]], axis=1)
    xyxy = (xyxy - [*pad, *pad]) / ratio
    xyxy[:, [0, 2]] = xyxy[:, [0, 2]].clip(0, width)
    xyxy[:, [1, 3]] = xyxy[:, [1, 3]].clip(0, height)
    return xyxy.astype(np.float32), rows[:, 4].astype(np.float32)


class OrtPersonDetector:
    def __init__(self, onnx_path: str | Path, threads: int = 0):
        self.session = make_session(onnx_path, threads)
        shape = self.session.get_inputs()[0].shape
        self.input_name, self.shape = self.session.get_inputs()[0].name, (int(shape[2]), int(shape[3]))

    def __call__(self, img_bgr: np.ndarray, conf: float):
        """-> (boxes, conf, {"pre_ms", "infer_ms", "post_ms"})."""
        t0 = time.perf_counter()
        blob, ratio, pad = letterbox(img_bgr, self.shape)
        t1 = time.perf_counter()
        output = self.session.run(None, {self.input_name: blob})[0]
        t2 = time.perf_counter()
        boxes, scores = decode_persons(output, conf, ratio, pad, img_bgr.shape[:2])
        t3 = time.perf_counter()
        return boxes, scores, {"pre_ms": (t1 - t0) * 1e3, "infer_ms": (t2 - t1) * 1e3, "post_ms": (t3 - t2) * 1e3}


def run_video(video_dir: Path, n_frames: int, config: dict, detector: OrtPersonDetector, pose: RtmPose, stgnf_model) -> list[dict]:
    video, level, c, k = video_dir.name, config["level"], config["c"], config["k"]
    tracker, live = build_tracker(c), LiveStgnf(stgnf_model)
    rows = []
    for i, frame_file in enumerate(sorted(video_dir.glob("*.jpg"))[:n_frames]):
        frame, is_key = int(frame_file.stem), i % k == 0
        t0 = time.perf_counter()
        image = cv2.imread(str(frame_file))
        t1 = time.perf_counter()
        dark = darken_level(image, level, video, frame)
        t2 = time.perf_counter()
        det = {"pre_ms": 0.0, "infer_ms": 0.0, "post_ms": 0.0}
        n_dets = 0
        if is_key:
            boxes, scores, det = detector(dark, c)
            n_dets = len(scores)
            t3 = time.perf_counter()
            tracks = np.asarray(tracker.update(Detections(boxes, scores, np.zeros(len(scores))), None)).reshape(-1, 8)
            ids, track_boxes = tracks[:, 4].astype(int), tracks[:, :4]
        else:
            t3 = time.perf_counter()
            ids, track_boxes, _ = predict_only(tracker)
        t4 = time.perf_counter()
        boxes = clip_boxes(np.asarray(track_boxes).reshape(-1, 4), dark.shape[1], dark.shape[0])
        ok = ((boxes[:, 2] - boxes[:, 0]) >= 2) & ((boxes[:, 3] - boxes[:, 1]) >= 2)
        keypoints = pose.predict(dark, boxes[ok])
        t5 = time.perf_counter()
        new_ids, _ = live.update(frame, {int(t): kp for t, kp in zip(np.asarray(ids)[ok], keypoints)})
        t6 = time.perf_counter()
        ms = lambda a, b: (b - a) * 1e3  # noqa: E731
        rows.append({**config, "video": video, "frame": frame, "is_key": int(is_key), "read_ms": ms(t0, t1),
                     "darken_ms": ms(t1, t2), "det_pre_ms": det["pre_ms"], "det_infer_ms": det["infer_ms"],
                     "det_post_ms": det["post_ms"], "track_ms": ms(t3, t4), "pose_ms": ms(t4, t5), "flow_ms": ms(t5, t6),
                     "service_ms": ms(t0, t6) - ms(t1, t2), "total_ms": ms(t0, t6), "n_dets": n_dets,
                     "n_boxes": int(ok.sum()), "n_new_segments": len(new_ids)})
    return rows


def warm_up(image: np.ndarray, detector: OrtPersonDetector, pose: RtmPose, stgnf_model, c: float):
    """First calls allocate buffers and build thread pools; do them outside the timed loop."""
    boxes = np.array([[100.0, 100.0, 160.0, 260.0]] * 3)
    for _ in range(2):
        detector(image, c)
        pose.predict(image, boxes)
    score_segments(stgnf_model, np.random.default_rng(0).uniform(100, 400, (3, 24, 17, 3)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames-root", type=Path, required=True)
    parser.add_argument("--videos-file", type=Path, required=True)
    parser.add_argument("--level", required=True)
    parser.add_argument("--c", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, required=True, choices=sorted(DET_ONNX))
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--pose-model", required=True, choices=["rtmpose-s", "rtmpose-t"])
    parser.add_argument("--threads", type=int, default=0, help="ORT intra-op and torch threads; 0 = library default (O0)")
    parser.add_argument("--max-frames-per-video", dest="max_frames", type=int, default=240)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tag", default="")
    args = parser.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
        cv2.setNumThreads(args.threads)
    config = {"tag": args.tag, "machine": os.environ.get("EDGELAB_MACHINE", platform.node()), "level": args.level, "c": args.c,
              "imgsz": args.imgsz, "k": args.k, "pose_model": args.pose_model, "threads": args.threads}
    videos = [v.strip() for v in args.videos_file.read_text().splitlines() if v.strip() and not v.startswith("#")]
    detector = OrtPersonDetector(DET_ONNX[args.imgsz], args.threads)
    pose = RtmPose(RTM_DIR / f"{args.pose_model}-body7-256x192.onnx", threads=args.threads)
    stgnf_model = load_model()
    first = sorted((args.frames_root / videos[0]).glob("*.jpg"))[0]
    warm_up(darken_level(cv2.imread(str(first)), args.level, videos[0], int(first.stem)), detector, pose, stgnf_model, args.c)
    meta = {**config, "loadavg_start": os.getloadavg(), "videos": videos, "max_frames": args.max_frames,
            "det_onnx": str(DET_ONNX[args.imgsz]), "det_input_hw": detector.shape, "python": platform.python_version(),
            "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    meta_file = args.out.with_suffix(".meta.json")
    meta_file.write_text(json.dumps(meta, indent=1))
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, COLUMNS)
        writer.writeheader()
        for video in videos:
            writer.writerows(run_video(args.frames_root / video, args.max_frames, config, detector, pose, stgnf_model))
            f.flush()  # a killed run keeps the videos finished so far
    meta["loadavg_end"] = os.getloadavg()
    meta_file.write_text(json.dumps(meta, indent=1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
