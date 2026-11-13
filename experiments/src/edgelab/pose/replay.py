"""Replay a detector cache through the Ultralytics tracker at gate threshold `c` -> STG-NF pose JSON (plan §4.3-4.4).

    uv run python -m edgelab.pose.replay --cache <detcache dir> --c 0.25 --out <pose_json_dir> [--tracker botsort]

Gate threshold `c` sets the detector confidence cut and the tracker's start-a-track thresholds together.
Ultralytics is AGPL-3.0: research use only (DECISIONS.md D11).
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from edgelab.pose.detcache import load_detections
from edgelab.pose.stgnf_format import TrackedPose, write_stgnf_json
from edgelab.pose.ubnormal import POSE_FILE_TOKEN


@dataclass
class Detections:
    """The minimum the Ultralytics tracker reads from a detection set: conf, cls, xywh, len and boolean indexing."""
    xyxy: np.ndarray
    conf: np.ndarray
    cls: np.ndarray

    def __len__(self):
        return len(self.conf)

    def __getitem__(self, mask):
        return Detections(self.xyxy[mask], self.conf[mask], self.cls[mask])

    @property
    def xywh(self):
        x1, y1, x2, y2 = self.xyxy.T
        return np.stack([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], axis=1)


def build_tracker(c: float, tracker: str = "botsort"):
    """Ultralytics default yaml for `tracker`, with the gate thresholds set from c (and no GMC / ReID for botsort)."""
    from ultralytics.trackers.track import TRACKER_MAP
    from ultralytics.utils import ROOT, YAML, IterableSimpleNamespace

    cfg = YAML.load(ROOT / "cfg/trackers" / f"{tracker}.yaml")
    cfg.update(track_high_thresh=c, new_track_thresh=c, track_low_thresh=min(0.1, c / 2))
    if cfg["tracker_type"] == "botsort":
        cfg.update(gmc_method="none", with_reid=False)  # no frame is passed to the tracker: GMC would have nothing to see
    return TRACKER_MAP[cfg["tracker_type"]](IterableSimpleNamespace(**cfg))


def replay_video(frames: list[dict], frame_ids: np.ndarray, c: float, tracker: str = "botsort",
                 keep_mask: np.ndarray | None = None, reindex: bool = False) -> tuple[list[TrackedPose], dict]:
    """frames: per-frame detection dicts from load_detections. Frames with keep_mask False never reach the tracker.

    reindex=True: pose frame ids are the consecutive index 0..K-1 of the processed frames (what a live system's
    per-track buffer of the last 24 poses sees); stats["frame_map"][k] = real frame id of processed frame k.
    """
    tracker_obj = build_tracker(c, tracker)
    poses, frames_seen, frames_with_person, detections_kept = [], 0, 0, 0
    frame_map = []
    for i, (frame, frame_id) in enumerate(zip(frames, frame_ids)):
        if keep_mask is not None and not keep_mask[i]:
            continue
        frame_map.append(int(frame_id))
        frames_seen += 1
        above = frame["conf"] >= c
        kept = {k: v[above] for k, v in frame.items()}
        detections_kept += int(above.sum())
        tracks = tracker_obj.update(Detections(kept["boxes"], kept["conf"], kept["cls"]), None)
        # rows: x1 y1 x2 y2 track_id score cls idx, idx = row in `kept`
        for row in tracks:
            idx = int(row[7])
            pose_frame = frames_seen - 1 if reindex else int(frame_id)
            poses.append(TrackedPose(pose_frame, int(row[4]), kept["keypoints"][idx].astype(float),
                                     float(kept["conf"][idx])))
        frames_with_person += int(len(tracks) > 0)
    stats = {"frames_total": len(frames), "frames_seen_by_tracker": frames_seen,
             "frames_with_person": frames_with_person, "detections_kept": detections_kept,
             "poses": len(poses), "tracks": len({p.track_id for p in poses})}
    if reindex:
        stats["frame_map"] = frame_map
    return poses, stats


FRAME_MAPS_FILE = "frame_maps.npz"  # reindex=True: per video, processed index -> real frame id


def replay_dir(cache: Path, c: float, out: Path, tracker: str = "botsort", videos: list[str] | None = None,
               keep_masks: dict[str, np.ndarray] | None = None, reindex: bool = False,
               dataset: str = "ShanghaiTech") -> dict[str, dict]:
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(cache.glob("*.npz")) if not videos else [cache / f"{v}.npz" for v in videos]
    all_stats, frame_maps = {}, {}
    for path in files:
        video = path.stem
        frames, data = load_detections(path)
        keep = None if keep_masks is None else keep_masks.get(video)
        poses, stats = replay_video(frames, data["frame_ids"], c, tracker, keep, reindex)
        if reindex:
            frame_maps[video] = np.array(stats.pop("frame_map"))
        write_stgnf_json(poses, out / f"{video}_{source_tag(cache, c, tracker, dataset)}_tracked_person.json")
        all_stats[video] = stats
    (out / "replay_stats.json").write_text(json.dumps(all_stats, indent=1))
    if reindex:
        np.savez_compressed(out / FRAME_MAPS_FILE, **frame_maps)
    return all_stats


def source_tag(cache: Path, c: float, tracker: str, dataset: str = "ShanghaiTech") -> str:
    # ShanghaiTech: no '_' in the tag, STG-NF takes scene and clip from the first two '_' fields of the file name.
    tag = f"{cache.name.replace('_', '-')}-c{c}-{tracker}"
    if dataset == "UBnormal":
        # STG-NF's UBnormal regex needs '_alphapose_' in the file name (see edgelab.pose.ubnormal.POSE_FILE_TOKEN)
        return f"{POSE_FILE_TOKEN}_{tag}"
    return tag


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--c", type=float, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tracker", default="botsort", choices=["botsort", "bytetrack"])
    parser.add_argument("--videos", nargs="*", default=None)
    parser.add_argument("--dataset", default="ShanghaiTech", choices=["ShanghaiTech", "UBnormal"])
    args = parser.parse_args()
    stats = replay_dir(args.cache, args.c, args.out, args.tracker, args.videos, dataset=args.dataset)
    total = {k: sum(s[k] for s in stats.values()) for k in next(iter(stats.values()))}
    print(json.dumps({"videos": len(stats), **total}))


if __name__ == "__main__":
    main()
