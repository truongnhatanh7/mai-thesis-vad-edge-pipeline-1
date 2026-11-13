"""EXP-006 light strata: one row per UBnormal test video -> strata.csv (plan §2).

    uv run --extra pose python -m edgelab.analysis.exp006_strata --videos ~/thesis-data/ubnormal/test_videos \
        --out EXP-006-ubnormal-replication/strata.csv

Per video: decoded frame count, GT length (must be equal), normal / anomalous frame counts, mean luma of frames sampled
at 1 per second (EXP-000 method: Y of OpenCV YCrCb, every round(fps) frames), flags and the stratum.
"""

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

from edgelab.pose.ubnormal import GT_DIR, is_weather, load_gt, load_test_list, stratum_of

COLUMNS = ["video", "scene", "frames", "gt_frames", "normal_frames", "anomalous_frames", "mean_luma", "dark", "weather",
           "stratum", "width", "height", "fps"]


def video_luma(path: Path, samples_per_second: float = 1.0) -> dict:
    """Decode the whole video once: frame count, size, fps and mean luma of the 1-per-second sample (as EXP-000)."""
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, round(fps / samples_per_second))
    means, frames = [], 0
    while cap.grab():
        if frames % step == 0:
            ok, bgr = cap.retrieve()
            if ok:
                means.append(float(cv2.split(cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb))[0].mean()))
        frames += 1
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    cap.release()
    return {"frames": frames, "mean_luma": float(np.mean(means)), "width": size[0], "height": size[1], "fps": fps}


def strata_row(video: str, probe: dict, labels: np.ndarray, dark_below: float = 60.0) -> dict:
    """One CSV row. labels: 1 = anomalous frame (ubnormal.load_gt)."""
    return {"video": video, "scene": int(video.split("_scene_")[1].split("_")[0]), "frames": probe["frames"],
            "gt_frames": len(labels), "normal_frames": int((labels == 0).sum()), "anomalous_frames": int(labels.sum()),
            "mean_luma": round(probe["mean_luma"], 2), "dark": probe["mean_luma"] < dark_below,
            "weather": is_weather(video), "stratum": stratum_of(video, probe["mean_luma"], dark_below),
            "width": probe["width"], "height": probe["height"], "fps": probe["fps"]}


def summarize(rows: list[dict]) -> dict[str, dict]:
    """Per stratum: videos, videos with any anomalous frame, videos without, normal and anomalous frames."""
    out = {}
    for stratum in ("dark", "weather", "normal"):
        part = [r for r in rows if r["stratum"] == stratum]
        with_anomaly = [r for r in part if r["anomalous_frames"] > 0]
        out[stratum] = {"videos": len(part), "with_anomaly": len(with_anomaly),
                        "without_anomaly": len(part) - len(with_anomaly),
                        "normal_frames": sum(r["normal_frames"] for r in part),
                        "anomalous_frames": sum(r["anomalous_frames"] for r in part)}
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos", type=Path, required=True, help="folder with <video>.mp4 for the test videos")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--gt-dir", type=Path, default=GT_DIR)
    args = parser.parse_args()
    rows = []
    for video in load_test_list():
        path = args.videos / f"{video}.mp4"
        if not path.exists():
            print(f"missing {path}")
            continue
        row = strata_row(video, video_luma(path), load_gt(video, args.gt_dir))
        if row["frames"] != row["gt_frames"]:
            print(f"FRAME COUNT MISMATCH {video}: decoded {row['frames']}, GT {row['gt_frames']}")
        rows.append(row)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} of {len(load_test_list())} test videos -> {args.out}")
    for stratum, stats in summarize(rows).items():
        print(stratum, stats)


if __name__ == "__main__":
    main()
