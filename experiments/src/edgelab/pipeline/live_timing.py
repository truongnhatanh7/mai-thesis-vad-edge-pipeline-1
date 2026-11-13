"""Per-frame timing trace of the whole edge pipeline (EXP-004 plan §3.1 step 1). Meant to run inside the VM `edge-os`.

    read jpg -> darken -> enhance -> YOLOv8n-pose (ONNX Runtime CPU) -> BoT-SORT -> STG-NF on each new 24-pose segment

    sudo env PYTHONPATH=<experiments>/src ~/edgelab-vm/.venv/bin/python -m edgelab.pipeline.live_timing \
        --frames-root <dir with one folder of NNN.jpg per video> --videos-file timing_subset.txt --level L2 --enhance E3 \
        --c 0.25 --threads 3 --spin off --affinity 1,2,3 --fifo 10 --out run.csv --tag O1-rep0

One CSV row per frame, all times in ms; config columns repeat on every row; `<out>.meta.json` holds versions and what
the kernel applied. service_ms = everything a live system pays per frame = total_ms - darken_ms (darkening only
simulates the dark sensor; reading the jpg stands in for fetching the camera frame).
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

from edgelab.pipeline.os_config import apply_os_config, cgroup_report, parse_cpus, thread_report
from edgelab.pipeline.ncnn_pose import NcnnPose
from edgelab.pipeline.ort_pose import OrtPose
from edgelab.pipeline.stgnf_live import LiveStgnf, load_model, score_segments
from edgelab.pose.replay import Detections, build_tracker
from edgelab.preprocess.darken import darken_level
from edgelab.preprocess.enhance import ENHANCERS
from edgelab.preprocess.zerodcepp import ZeroDCEppOnnx

EXCHANGE = Path("/Users/nanh/thesis-data/vm-exchange")
CONFIG_COLUMNS = ["tag", "machine", "level", "enhance", "c", "imgsz", "threads", "spin", "affinity", "fifo"]
TIME_COLUMNS = ["read_ms", "darken_ms", "enhance_ms", "det_pre_ms", "det_infer_ms", "det_post_ms", "track_ms",
                "flow_ms", "service_ms", "total_ms"]
COLUMNS = CONFIG_COLUMNS + ["video", "frame"] + TIME_COLUMNS + ["n_dets", "n_tracks", "n_new_segments"]


def yolo_onnx_for(imgsz: int, models_dir: Path = EXCHANGE / "models") -> Path:
    """The static rectangular export for R = imgsz (yolov8n-pose-<R>-<h>x<w>.onnx, see export_onnx.py)."""
    # the FP32 export only: EXP-012's INT8 file (…-int8.onnx) lives in the same folder
    found = sorted(p for p in models_dir.glob(f"yolov8n-pose-{imgsz}-*x*.onnx") if not p.name.endswith("-int8.onnx"))
    if len(found) != 1:
        raise FileNotFoundError(f"expected exactly one yolov8n-pose-{imgsz}-<h>x<w>.onnx in {models_dir}, found {found}")
    return found[0]


def make_detector(path: Path, threads: int, spin: bool, ncnn_fp16: bool = True):
    """A folder is an NCNN model (Ultralytics export), a file is an ONNX model (FP32 or INT8) for ONNX Runtime."""
    return NcnnPose(path, threads, ncnn_fp16) if path.is_dir() else OrtPose(path, threads, spin)


def make_enhancer(name: str, zdce_onnx: Path, threads: int, spin: bool):
    if name == "E3":
        return ZeroDCEppOnnx(zdce_onnx, threads=threads, spin=spin)  # same threads and spinning as the detector
    return ENHANCERS[name]


def run_video(video_dir: Path, n_frames: int, config: dict, enhance, detector: OrtPose, stgnf_model) -> list[dict]:
    video, level, c = video_dir.name, config["level"], config["c"]
    tracker, live = build_tracker(c), LiveStgnf(stgnf_model)
    rows = []
    for frame_file in sorted(video_dir.glob("*.jpg"))[:n_frames]:
        frame = int(frame_file.stem)
        t0 = time.perf_counter()
        image = cv2.imread(str(frame_file))
        t1 = time.perf_counter()
        dark = darken_level(image, level, video, frame)
        t2 = time.perf_counter()
        enhanced = enhance(dark)
        t3 = time.perf_counter()
        boxes, scores, keypoints, det = detector(enhanced, c)
        t4 = time.perf_counter()
        tracks = tracker.update(Detections(boxes, scores, np.zeros(len(scores))), None)
        t5 = time.perf_counter()
        ids, _ = live.update(frame, {int(row[4]): keypoints[int(row[7])] for row in tracks})
        t6 = time.perf_counter()
        ms = lambda a, b: (b - a) * 1e3  # noqa: E731
        rows.append({**config, "video": video, "frame": frame, "read_ms": ms(t0, t1), "darken_ms": ms(t1, t2),
                     "enhance_ms": ms(t2, t3), "det_pre_ms": det["pre_ms"], "det_infer_ms": det["infer_ms"],
                     "det_post_ms": det["post_ms"], "track_ms": ms(t4, t5), "flow_ms": ms(t5, t6),
                     "service_ms": ms(t0, t6) - ms(t1, t2), "total_ms": ms(t0, t6), "n_dets": len(scores),
                     "n_tracks": len(tracks), "n_new_segments": len(ids)})
    return rows


def warm_up(image: np.ndarray, enhance, detector: OrtPose, stgnf_model, c: float):
    """First calls allocate buffers and build thread pools; do them outside the timed loop."""
    for _ in range(2):
        detector(enhance(image), c)
    score_segments(stgnf_model, np.random.default_rng(0).uniform(100, 400, (3, 24, 17, 3)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames-root", type=Path, required=True)
    parser.add_argument("--videos-file", type=Path, required=True, help="one video folder name per line")
    parser.add_argument("--level", required=True)
    parser.add_argument("--enhance", required=True, choices=["E0", "E1", "E2", "E3"])
    parser.add_argument("--c", type=float, required=True)
    parser.add_argument("--threads", type=int, default=0, help="ORT intra-op and torch threads; 0 = library default")
    parser.add_argument("--spin", choices=["on", "off"], default="on")
    parser.add_argument("--affinity", default=None, help="e.g. 1,2,3")
    parser.add_argument("--fifo", type=int, default=None, help="SCHED_FIFO priority (needs root)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tag", default="")
    parser.add_argument("--max-frames-per-video", "--max-frames", dest="max_frames", type=int, default=240)
    parser.add_argument("--imgsz", type=int, default=640, choices=[480, 640, 800, 960],
                        help="R: selects the static rectangular ONNX exported for this imgsz")
    parser.add_argument("--yolo-onnx", type=Path, default=None,
                        help="override the file chosen by --imgsz; a DIRECTORY is an NCNN model (EXP-012), a file is ONNX")
    parser.add_argument("--ncnn-fp16", choices=["on", "off"], default="on", help="NCNN only: half-precision storage/arithmetic")
    parser.add_argument("--zdce-onnx", type=Path, default=EXCHANGE / "models/zerodcepp.onnx")
    args = parser.parse_args()

    affinity = parse_cpus(args.affinity) if args.affinity else None
    apply_os_config(affinity, args.fifo)  # before the models: the threads they create inherit it
    if args.threads:
        torch.set_num_threads(args.threads)
        cv2.setNumThreads(args.threads)
    spin = args.spin == "on"
    config = {"tag": args.tag, "machine": os.environ.get("EDGELAB_MACHINE", platform.node()), "level": args.level,
              "enhance": args.enhance, "c": args.c, "imgsz": args.imgsz, "threads": args.threads, "spin": args.spin,
              "affinity": args.affinity or "", "fifo": args.fifo or 0}

    videos = [v.strip() for v in args.videos_file.read_text().splitlines() if v.strip() and not v.startswith("#")]
    yolo_onnx = args.yolo_onnx or yolo_onnx_for(args.imgsz)
    detector = make_detector(yolo_onnx, args.threads, spin, args.ncnn_fp16 == "on")
    enhance = make_enhancer(args.enhance, args.zdce_onnx, args.threads, spin)
    stgnf_model = load_model()
    first = sorted((args.frames_root / videos[0]).glob("*.jpg"))[0]
    warm_up(darken_level(cv2.imread(str(first)), args.level, videos[0], int(first.stem)), enhance, detector,
            stgnf_model, args.c)
    apply_os_config(affinity, args.fifo)  # again: catches thread pools that started lazily during the warm-up
    meta = {**config, "loadavg_start": os.getloadavg(), "threads_applied": thread_report(),
            "cgroup": cgroup_report() if os.path.exists("/proc/self/cgroup") else None,
            "torch_threads": torch.get_num_threads(), "cv2_threads": cv2.getNumThreads(),
            "videos": videos, "max_frames": args.max_frames, "yolo_onnx": str(yolo_onnx),
            "yolo_input_hw": detector.shape, "detector": type(detector).__name__, "ncnn_fp16": args.ncnn_fp16, "python": platform.python_version(),
            "started": time.strftime("%Y-%m-%d %H:%M:%S")}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    meta_file = args.out.with_suffix(".meta.json")
    meta_file.write_text(json.dumps(meta, indent=1))  # also written at the end, with the final load average
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, COLUMNS)
        writer.writeheader()
        for video in videos:
            writer.writerows(run_video(args.frames_root / video, args.max_frames, config, enhance, detector, stgnf_model))
            f.flush()  # a killed run keeps the videos finished so far
    meta["loadavg_end"] = os.getloadavg()
    meta_file.write_text(json.dumps(meta, indent=1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
