"""EXP-016 top-down pipeline: person detector -> BoT-SORT -> RTMPose on every tracked box -> STG-NF pose JSON.

    # 1. person-box cache (YOLOv8n detection weights, person class only, conf >= 0.01), one npz per video
    uv run --extra pose python -m edgelab.pose.topdown cache --frames-root <testing/frames> --level L0 --imgsz 640 --out <dir>
    # 2. tracking replay + pose, K = 1 (detector every frame) or K = 3 (detector every 3rd frame)
    uv run --extra pose python -m edgelab.pose.topdown poses --cache <dir> --level L0 --pose-model rtmpose-s --k 3 --c 0.25 --out <dir>

Keyframe rule (K > 1): on frame i with i % K == 0 the cached detections go through the tracker as usual. On the other
frames the detector is not used: every active track's Kalman filter is advanced by one step and its predicted box is
used (new people can only appear on keyframes). RTMPose runs on every box of every frame.
Frame i = index in the video (0-based), the same index as the jpg name. Ultralytics is AGPL-3.0: research use only.
"""

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from edgelab.pose.detcache import list_videos
from edgelab.pose.extract import iter_frames
from edgelab.pose.replay import Detections, build_tracker, source_tag
from edgelab.pose.rtmpose import RtmPose
from edgelab.pose.stgnf_eval import EXPERIMENTS_ROOT
from edgelab.pose.stgnf_format import TrackedPose, write_stgnf_json
from edgelab.preprocess.darken import darken_level

WEIGHTS = EXPERIMENTS_ROOT / "artifacts/EXP-016/models/yolov8n.pt"
RTMPOSE_DIR = Path.home() / "thesis-data/exp015/models"
PERSON_CLASS = 0
CONF, IOU = 0.01, 0.7
MIN_BOX_PX = 2.0
EXPECTED_VIDEOS = 107  # ShanghaiTech test split


# ---------- detector cache ----------

def detect_persons(model, img, device: str, imgsz: int):
    """One frame -> (boxes xyxy (n, 4), conf (n,), ultralytics speed dict); person class only."""
    result = model.predict(img, imgsz=imgsz, conf=CONF, iou=IOU, classes=[PERSON_CLASS], device=device, verbose=False)[0]
    return result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy(), result.speed


def cache_video(model, frames_dir: Path, level: str, imgsz: int, device: str, threads: int | None = None) -> dict:
    video = frames_dir.name
    boxes, confs, counts, frame_ids, infer_ms = [], [], [], [], []
    for frame, image in iter_frames(frames_dir):
        b, c, speed = detect_persons(model, darken_level(image, level, video, frame), device, imgsz)
        if threads and not counts:
            import torch

            torch.set_num_threads(threads)  # Ultralytics resets the torch thread count on first predict
        boxes.append(b)
        confs.append(c)
        counts.append(len(c))
        frame_ids.append(frame)
        infer_ms.append(speed["inference"])
    return {"boxes": np.concatenate(boxes).astype(np.float32).reshape(-1, 4),
            "conf": np.concatenate(confs).astype(np.float32), "counts": np.array(counts), "frame_ids": np.array(frame_ids),
            "infer_ms": np.array(infer_ms, dtype=np.float32),
            "meta": np.array(json.dumps({"video": video, "level": level, "imgsz": imgsz, "device": device, "conf": CONF,
                                         "iou": IOU, "weights": WEIGHTS.name}))}


class IncompleteCacheError(RuntimeError):
    """A person-box cache does not hold exactly one finished npz per video (EXP-016 deviation 2026-11-08)."""


def cache_files(cache: Path) -> list[Path]:
    """Finished cache files only (`<video>.npz`; the `.partial.npz` temp files of save_atomic are not caches)."""
    return sorted(p for p in cache.glob("*.npz") if not p.name.endswith(".partial.npz"))


def check_cache_complete(cache: Path, expected: int | list[str] | None = EXPECTED_VIDEOS) -> int:
    """Raise IncompleteCacheError unless `cache` has exactly the expected videos (an int = count only, a list = names too).

    Every consumer that scores a cache calls this first: a crashed (e.g. MPS) cache run must never be scored.
    """
    found = [p.stem for p in cache_files(cache)] if cache.is_dir() else []
    if isinstance(expected, int):
        if len(found) != expected:
            raise IncompleteCacheError(f"{cache}: {len(found)} npz files, {expected} expected")
    elif expected is not None:
        missing = sorted(set(expected) - set(found))
        if missing or len(found) != len(expected):
            raise IncompleteCacheError(f"{cache}: {len(found)} npz files, {len(expected)} expected; missing {missing[:5]}")
    return len(found)


def load_boxes(path: Path) -> tuple[list[tuple[np.ndarray, np.ndarray]], np.ndarray]:
    """-> ([(boxes, conf) per frame], frame_ids)."""
    data = np.load(path)
    ends = np.cumsum(data["counts"])
    per_frame = [(data["boxes"][e - n:e], data["conf"][e - n:e]) for n, e in zip(data["counts"], ends)]
    return per_frame, data["frame_ids"]


def save_atomic(path: Path, arrays: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".partial.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def device_stats(model, frames_dir: Path, n_frames: int, imgsz: int, level: str = "L0") -> dict:
    """MPS-vs-CPU check of plan §4.3 for the detection weights: same keys as detcache.mps_vs_cpu_check."""
    per_device = {}
    for device in ("cpu", "mps"):
        confs = []
        for frame, image in iter_frames(frames_dir):
            if len(confs) == n_frames:
                break
            confs.append(np.sort(detect_persons(model, darken_level(image, level, frames_dir.name, frame), device, imgsz)[1])[::-1])
        per_device[device] = confs
    equal, max_diff = 0, 0.0
    for cpu, mps in zip(per_device["cpu"], per_device["mps"]):
        cpu, mps = cpu[cpu >= 0.1], mps[mps >= 0.1]
        if len(cpu) == len(mps):
            equal += 1
            if len(cpu):
                max_diff = max(max_diff, float(np.abs(cpu - mps).max()))
    return {"video": frames_dir.name, "imgsz": imgsz, "frames": n_frames, "frames_equal_count": equal, "max_conf_diff": max_diff}


# ---------- tracking with keyframes ----------

@dataclass
class FrameTracks:
    frame: int
    track_ids: np.ndarray  # (m,)
    boxes: np.ndarray  # (m, 4) xyxy
    scores: np.ndarray  # (m,) detection score of the track's last matched detection
    is_keyframe: bool


def predict_only(tracker) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A frame without detections: move every active (and lost) track one Kalman step, return the active tracks' boxes.

    The state is what `tracker.update` would see anyway (it also predicts the active and lost tracks first), so the next
    keyframe continues from consistent filters.
    """
    tracker.frame_id += 1
    active = [t for t in tracker.tracked_stracks if t.is_activated]
    tracker.multi_predict(active + list(tracker.lost_stracks))
    if not active:
        return np.zeros(0, int), np.zeros((0, 4)), np.zeros(0)
    return (np.array([t.track_id for t in active]), np.array([t.xyxy for t in active]), np.array([t.score for t in active]))


def track_video(per_frame: list[tuple[np.ndarray, np.ndarray]], frame_ids: np.ndarray, c: float, k: int = 1,
                tracker_name: str = "botsort") -> list[FrameTracks]:
    """Cached person boxes -> tracked boxes for every frame. Frame index i is a keyframe when i % k == 0."""
    tracker = build_tracker(c, tracker_name)
    out = []
    for i, ((boxes, conf), frame_id) in enumerate(zip(per_frame, frame_ids)):
        if i % k == 0:
            above = conf >= c
            rows = tracker.update(Detections(boxes[above], conf[above], np.zeros(above.sum())), None)
            rows = np.asarray(rows).reshape(-1, 8)
            out.append(FrameTracks(int(frame_id), rows[:, 4].astype(int), rows[:, :4], rows[:, 5], True))
        else:
            ids, boxes_pred, scores = predict_only(tracker)
            out.append(FrameTracks(int(frame_id), ids, boxes_pred.reshape(-1, 4), scores, False))
    return out


# ---------- pose on tracked boxes ----------

def clip_boxes(boxes: np.ndarray, width: int, height: int) -> np.ndarray:
    clipped = boxes.copy().astype(np.float64)
    clipped[:, [0, 2]] = clipped[:, [0, 2]].clip(0, width)
    clipped[:, [1, 3]] = clipped[:, [1, 3]].clip(0, height)
    return clipped


def pose_on_tracks(tracks: list[FrameTracks], source: Path, level: str, pose: RtmPose) -> tuple[list[TrackedPose], dict]:
    """Run the pose model on every tracked box (darkened frame, same seed as the detector saw). Returns poses and counts."""
    video = source.name
    by_frame = {t.frame: t for t in tracks}
    poses, stats = [], {"pose_calls": 0, "too_small": 0, "pose_ms": 0.0}
    for frame, image in iter_frames(source):
        t = by_frame[frame]
        if len(t.track_ids) == 0:
            continue
        image = darken_level(image, level, video, frame)
        boxes = clip_boxes(t.boxes, image.shape[1], image.shape[0])
        ok = ((boxes[:, 2] - boxes[:, 0]) >= MIN_BOX_PX) & ((boxes[:, 3] - boxes[:, 1]) >= MIN_BOX_PX)
        stats["too_small"] += int((~ok).sum())
        t0 = time.perf_counter()
        keypoints = pose.predict(image, boxes[ok])
        stats["pose_ms"] += (time.perf_counter() - t0) * 1e3
        stats["pose_calls"] += int(ok.sum())
        poses += [TrackedPose(frame, int(i), kp.astype(float), float(s))
                  for i, kp, s in zip(t.track_ids[ok], keypoints, t.scores[ok])]
    return poses, stats


def poses_for_video(per_frame, frame_ids, source: Path, level: str, pose: RtmPose, c: float, k: int) -> tuple[list[TrackedPose], dict]:
    tracks = track_video(per_frame, frame_ids, c, k)
    poses, stats = pose_on_tracks(tracks, source, level, pose)
    stats.update({"frames": len(tracks), "keyframes": sum(t.is_keyframe for t in tracks),
                  "frames_with_person": sum(len(t.track_ids) > 0 for t in tracks), "poses": len(poses),
                  "tracks": len({p.track_id for p in poses})})
    return poses, stats


def poses_for_dir(cache: Path, frames_root: Path, level: str, pose_model: str, k: int, c: float, out: Path,
                  threads: int = 4, videos: list[str] | None = None, expected: int | list[str] | None = EXPECTED_VIDEOS
                  ) -> dict[str, dict]:
    if videos is None:
        check_cache_complete(cache, expected)  # never pose/score an incomplete cache
    pose = RtmPose(RTMPOSE_DIR / f"{pose_model}-body7-256x192.onnx", threads=threads)
    out.mkdir(parents=True, exist_ok=True)
    all_stats = {}
    for video in videos or [p.stem for p in cache_files(cache)]:
        per_frame, frame_ids = load_boxes(cache / f"{video}.npz")
        poses, stats = poses_for_video(per_frame, frame_ids, frames_root / video, level, pose, c, k)
        write_stgnf_json(poses, out / f"{video}_{source_tag(cache, c, 'botsort')}_tracked_person.json")
        all_stats[video] = stats
    (out / "topdown_stats.json").write_text(json.dumps(all_stats, indent=1))
    return all_stats


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    cache = sub.add_parser("cache")
    cache.add_argument("--frames-root", type=Path, required=True)
    cache.add_argument("--level", required=True)
    cache.add_argument("--imgsz", type=int, required=True)
    cache.add_argument("--out", type=Path, required=True)
    cache.add_argument("--videos", nargs="*", default=None)
    cache.add_argument("--device", default="mps", choices=["cpu", "mps"])
    cache.add_argument("--weights", type=Path, default=WEIGHTS)
    cache.add_argument("--threads", type=int, default=None)
    parity = sub.add_parser("parity", help="plan §4.3 MPS-vs-CPU rule for the detection weights; exit code 1 = use cpu")
    parity.add_argument("--frames-root", type=Path, required=True)
    parity.add_argument("--videos", nargs="+", required=True)
    parity.add_argument("--frames", type=int, default=50)
    parity.add_argument("--weights", type=Path, default=WEIGHTS)
    poses = sub.add_parser("poses")
    poses.add_argument("--cache", type=Path, required=True)
    poses.add_argument("--frames-root", type=Path, required=True)
    poses.add_argument("--level", required=True)
    poses.add_argument("--pose-model", default="rtmpose-s", choices=["rtmpose-s", "rtmpose-t"])
    poses.add_argument("--k", type=int, default=1)
    poses.add_argument("--c", type=float, default=0.25)
    poses.add_argument("--out", type=Path, required=True)
    poses.add_argument("--videos", nargs="*", default=None)
    poses.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.command == "parity":
        from ultralytics import YOLO

        from edgelab.pose.detcache import device_check_passes

        model = YOLO(str(args.weights))
        stats = [device_stats(model, args.frames_root / v, args.frames, imgsz) for v in args.videos for imgsz in (640, 960)]
        print(json.dumps(stats))
        raise SystemExit(0 if device_check_passes(stats) else 1)
    if args.command == "cache":
        from ultralytics import YOLO

        model = YOLO(str(args.weights))
        for video in args.videos or list_videos(args.frames_root):
            target = args.out / f"{video}.npz"
            if target.exists():
                continue  # resumable
            t0 = time.perf_counter()
            arrays = cache_video(model, args.frames_root / video, args.level, args.imgsz, args.device, args.threads)
            save_atomic(target, arrays)
            print(json.dumps({"video": video, "frames": len(arrays["counts"]), "detections": int(arrays["counts"].sum()),
                              "wall_s": round(time.perf_counter() - t0, 1)}), flush=True)
        if args.videos is None:
            check_cache_complete(args.out, list_videos(args.frames_root))  # fails (exit 1) if anything is missing
    else:
        stats = poses_for_dir(args.cache, args.frames_root, args.level, args.pose_model, args.k, args.c, args.out,
                              args.threads, args.videos)
        print(json.dumps({k: sum(s[k] for s in stats.values()) for k in next(iter(stats.values()))}))


if __name__ == "__main__":
    main()
