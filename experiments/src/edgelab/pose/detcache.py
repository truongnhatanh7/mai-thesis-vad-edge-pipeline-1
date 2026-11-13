"""Detector cache (EXP-003 plan §4.3): frame -> darken -> enhance -> YOLOv8n-pose, all detections kept (conf >= 0.01).

    uv run python -m edgelab.pose.detcache --frames-root <testing/frames | folder of .mp4 files> --level L2 --enhance E1 --out <dir> \
        [--videos 01_0014 ...] [--device mps|cpu] [--threads N]

One compressed `<video>.npz` per video. Detections of all frames are stored flat; `counts[i]` says how many belong to
frame `frame_ids[i]`. Tracking happens later (replay.py), so one cache serves every gate threshold.
Ultralytics is AGPL-3.0: research use only (DECISIONS.md D11).
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from edgelab.pose.extract import iter_frames
from edgelab.preprocess.darken import darken_level, mean_luma
from edgelab.preprocess.enhance import ENHANCERS
from edgelab.pose.stgnf_eval import EXPERIMENTS_ROOT

WEIGHTS = EXPERIMENTS_ROOT / "artifacts/EXP-001/models/yolov8n-pose.pt"
IMGSZ, CONF, IOU = 640, 0.01, 0.7


def detect(model, img, device: str, imgsz: int = IMGSZ, conf: float = CONF, iou: float = IOU, nms_free: bool = False):
    """One frame -> (boxes xyxy (N,4), conf (N,), cls (N,), keypoints (N,17,3), ultralytics speed dict)."""
    extra = {"nms": False} if nms_free else {}  # EXP-017: YOLO26's NMS-free one-to-one head (the default .pt path is one-to-many + NMS)
    result = model.predict(img, imgsz=imgsz, conf=conf, iou=iou, device=device, verbose=False, **extra)[0]
    boxes = result.boxes
    keypoints = result.keypoints.data.cpu().numpy().reshape(-1, 17, 3)
    return (boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy(), boxes.cls.cpu().numpy(), keypoints, result.speed)


MAX_DET = 300  # Ultralytics' default cap on detections per frame


def detect_fast(detector, img, conf: float = CONF):
    """Same tuple as `detect`, from an OrtPose / NcnnPose (EXP-012: ORT FP32, ORT INT8, NCNN). cls is 0 (person) everywhere.

    The `speed` dict uses Ultralytics' key names so cache_video stores the timings in the same columns.
    """
    boxes, scores, keypoints, t = detector(img, conf)
    boxes, scores, keypoints = boxes[:MAX_DET], scores[:MAX_DET], keypoints[:MAX_DET]  # already sorted by confidence
    return (boxes, scores, np.zeros(len(scores), np.float32), keypoints,
            {"preprocess": t["pre_ms"], "inference": t["infer_ms"], "postprocess": t["post_ms"]})


VIDEO_SUFFIXES = (".mp4", ".avi")


def list_videos(root: Path) -> list[str]:
    """Names of what `root` holds: frame folders (ShanghaiTech) and/or video files (UBnormal: <name>.mp4)."""
    return sorted(p.stem if p.is_file() else p.name for p in root.iterdir()
                  if p.is_dir() or p.suffix in VIDEO_SUFFIXES)


def find_source(root: Path, video: str) -> Path:
    """Frame folder `root/video`, else video file `root/video.<suffix>`."""
    if (root / video).is_dir():
        return root / video
    for suffix in VIDEO_SUFFIXES:
        if (root / f"{video}{suffix}").is_file():
            return root / f"{video}{suffix}"
    raise FileNotFoundError(f"no folder or video file named {video!r} in {root}")


def letterbox_shape(model, width: int, height: int, imgsz: int = IMGSZ, device: str = "cpu") -> tuple[int, int]:
    """(h, w) of the padded image Ultralytics feeds the network for a width x height frame at `imgsz` (after one predict)."""
    img = np.zeros((height, width, 3), np.uint8)
    detect(model, img, device, imgsz=imgsz)
    return tuple(model.predictor.pre_transform([img])[0].shape[:2])


def cache_video(model, frames_dir: Path, level: str, enhance: str, device: str, threads: int | None = None,
                imgsz: int = IMGSZ, detector=None, nms_free: bool = False, weights_name: str | None = None) -> dict:
    """model = Ultralytics YOLO; or detector = OrtPose / NcnnPose (then `model`, `device` and `imgsz` are only labels)."""
    video = frames_dir.stem if frames_dir.is_file() else frames_dir.name  # frame folder (ShanghaiTech) or .mp4/.avi file
    enhance_fn = ENHANCERS[enhance]
    detections, counts, frame_ids, rows = [], [], [], []
    for frame, image in iter_frames(frames_dir):
        t0 = time.perf_counter()
        dark = darken_level(image, level, video, frame)
        t1 = time.perf_counter()
        enhanced = enhance_fn(dark)
        t2 = time.perf_counter()
        if detector is not None:
            boxes, conf, cls, keypoints, speed = detect_fast(detector, enhanced)
        else:
            boxes, conf, cls, keypoints, speed = detect(model, enhanced, device, imgsz=imgsz, nms_free=nms_free)
        if threads and not rows:
            torch.set_num_threads(threads)  # Ultralytics resets the torch thread count on first predict
        detections.append((boxes, conf, cls, keypoints))
        counts.append(len(conf))
        frame_ids.append(frame)
        rows.append([mean_luma(dark), mean_luma(enhanced), (t1 - t0) * 1e3, (t2 - t1) * 1e3,
                     speed["preprocess"], speed["inference"], speed["postprocess"]])
    rows = np.array(rows, dtype=np.float32)
    stack = lambda i, shape: np.concatenate([d[i] for d in detections]).astype(np.float32).reshape(shape)  # noqa: E731
    names = ["luma_dark", "luma_enhanced", "darken_ms", "enhance_ms", "yolo_pre_ms", "yolo_infer_ms", "yolo_post_ms"]
    return {"boxes": stack(0, (-1, 4)), "conf": stack(1, (-1,)), "cls": stack(2, (-1,)), "keypoints": stack(3, (-1, 17, 3)),
            "counts": np.array(counts), "frame_ids": np.array(frame_ids),
            **{name: rows[:, i] for i, name in enumerate(names)},
            "meta": np.array(json.dumps({"video": video, "level": level, "enhance": enhance, "device": device,
                                         "imgsz": imgsz, "conf": CONF, "iou": IOU,
                                         **({"nms_free": True} if nms_free else {}),
                                         **({"weights": weights_name} if weights_name else {})}))}


def save_atomic(path: Path, arrays: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".partial.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)  # an interrupted run never leaves a half-written file that would be skipped as "done"


def load_detections(path: Path) -> tuple[list[dict], dict]:
    """-> (one dict per frame with boxes/conf/cls/keypoints, whole-file arrays)."""
    data = dict(np.load(path))
    ends = np.cumsum(data["counts"])
    frames = []
    for i, end in enumerate(ends):
        s = slice(end - data["counts"][i], end)
        frames.append({k: data[k][s] for k in ("boxes", "conf", "cls", "keypoints")})
    return frames, data


def mps_vs_cpu_check(frames_dir: Path, n_frames: int, level: str = "L2", enhance: str = "E0",
                     weights: Path = WEIGHTS) -> dict:
    """Plan §4.3 on one video: compare MPS and CPU detections of the first n frames.

    Returns max |conf diff| (over frames whose box count at conf >= 0.1 agrees, boxes paired by rank of confidence)
    and the fraction of frames whose count at conf >= 0.1 agrees. `device_check_passes` applies the rule.
    """
    from ultralytics import YOLO

    model = YOLO(str(weights))
    per_device = {}
    for device in ("cpu", "mps"):
        confs = []
        for frame, image in iter_frames(frames_dir):
            if len(confs) == n_frames:
                break
            img = ENHANCERS[enhance](darken_level(image, level, frames_dir.name, frame))
            confs.append(np.sort(detect(model, img, device)[1])[::-1])
        per_device[device] = confs
    equal, max_diff = 0, 0.0
    for cpu, mps in zip(per_device["cpu"], per_device["mps"]):
        cpu, mps = cpu[cpu >= 0.1], mps[mps >= 0.1]
        if len(cpu) == len(mps):
            equal += 1
            if len(cpu):
                max_diff = max(max_diff, float(np.abs(cpu - mps).max()))
    return {"video": frames_dir.name, "frames": len(per_device["cpu"]), "frames_equal_count": equal,
            "max_conf_diff": max_diff}


def device_check_passes(stats: list[dict]) -> bool:
    """Rule from plan §4.3, pooled over the checked videos: max conf diff <= 0.01 and equal counts on >= 99% of frames."""
    frames = sum(s["frames"] for s in stats)
    equal = sum(s["frames_equal_count"] for s in stats)
    return max(s["max_conf_diff"] for s in stats) <= 0.01 and equal >= 0.99 * frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames-root", type=Path, required=True)
    parser.add_argument("--level", required=True, help="L0 L1 L2 L3 L4 L2-nonoise")
    parser.add_argument("--enhance", default="E0", help="E0 none, E1 gamma, E2 CLAHE, E3 Zero-DCE++")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--videos", nargs="*", default=None)
    parser.add_argument("--device", default="cpu", choices=["cpu", "mps"])
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--weights", type=Path, default=WEIGHTS)
    parser.add_argument("--imgsz", type=int, default=IMGSZ)
    parser.add_argument("--nms-free", action="store_true",
                        help="EXP-017: Ultralytics predict(nms=False), the end-to-end head of YOLO26 (meta records it)")
    parser.add_argument("--detector", type=Path, default=None,
                        help="EXP-012: an .onnx file (ONNX Runtime, FP32 or INT8) or an NCNN folder instead of the .pt weights")
    parser.add_argument("--ncnn-fp16", choices=["on", "off"], default="on")
    parser.add_argument("--label", default=None, help="written to meta.device when --detector is used (e.g. ort-int8)")
    args = parser.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    detector, model = None, None
    if args.detector is not None:
        from edgelab.pipeline.live_timing import make_detector

        detector = make_detector(args.detector, args.threads or 0, False, args.ncnn_fp16 == "on")
        args.device, args.imgsz = args.label or args.detector.name, detector.shape[1]
    else:
        from ultralytics import YOLO

        model = YOLO(str(args.weights))
    videos = args.videos or list_videos(args.frames_root)
    args.out.mkdir(parents=True, exist_ok=True)
    for video in videos:
        out_file = args.out / f"{video}.npz"
        if out_file.exists():
            continue  # resumable
        t0 = time.perf_counter()
        arrays = cache_video(model, find_source(args.frames_root, video), args.level, args.enhance, args.device, args.threads,
                             args.imgsz, detector, args.nms_free, args.weights.name if args.nms_free else None)
        save_atomic(out_file, arrays)
        print(json.dumps({"video": video, "frames": len(arrays["counts"]), "detections": int(arrays["counts"].sum()),
                          "wall_s": round(time.perf_counter() - t0, 1)}), flush=True)


if __name__ == "__main__":
    main()
