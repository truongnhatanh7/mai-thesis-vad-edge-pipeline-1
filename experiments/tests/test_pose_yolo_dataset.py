"""EXP-013 pseudo-label dataset: label lines by hand, split by video, and a tiny end-to-end build."""

import json

import cv2
import numpy as np

from edgelab.pose.stgnf_format import TrackedPose, write_stgnf_json
from edgelab.pose.yolo_dataset import build_dataset, label_line, split_by_video


def _person(visible_xy: dict[int, tuple[float, float]]) -> np.ndarray:
    """(17, 3): the listed keypoints have conf 0.9, all others conf 0.1 (invisible) at (500, 500)."""
    kp = np.tile([500.0, 500.0, 0.1], (17, 1))
    for i, (x, y) in visible_xy.items():
        kp[i] = [x, y, 0.9]
    return kp


def test_label_line_box_is_visible_keypoints_plus_ten_percent():
    kp = _person({0: (100, 100), 1: (200, 100), 2: (100, 300)})  # box x 100..200, y 100..300
    fields = label_line(kp, width=1000, height=1000).split()
    assert len(fields) == 5 + 17 * 3
    cx, cy, w, h = (float(v) for v in fields[1:5])
    # width 100 -> pad 10 each side -> x 90..210 ; height 200 -> pad 20 -> y 80..320
    assert (cx, cy, w, h) == (0.15, 0.2, 0.12, 0.24)
    assert fields[0] == "0"


def test_label_line_invisible_keypoints_are_zero_and_visible_have_flag_2():
    kp = _person({0: (100, 100), 1: (200, 100), 2: (100, 300)})
    fields = label_line(kp, 1000, 1000).split()
    assert fields[5:8] == ["0.100000", "0.100000", "2"]  # keypoint 0
    assert fields[5 + 3 * 3: 5 + 3 * 3 + 3] == ["0", "0", "0"]  # keypoint 3: conf 0.1


def test_label_line_box_is_clipped_to_the_image():
    kp = _person({0: (0, 0), 1: (100, 0), 2: (0, 100)})
    cx, cy, w, h = (float(v) for v in label_line(kp, 200, 200).split()[1:5])
    assert (cx - w / 2, cy - h / 2) == (0.0, 0.0)  # padding would go below 0, clipped


def test_label_line_drops_a_person_with_fewer_than_three_visible_keypoints():
    assert label_line(_person({0: (10, 10), 1: (50, 50)}), 100, 100) is None
    assert label_line(_person({0: (10, 10), 1: (10, 10), 2: (10, 10)}), 100, 100) is None  # no area


def test_split_by_video_is_disjoint_per_scene_and_repeatable():
    videos = [f"{scene:02d}_{clip:03d}" for scene in (1, 2, 3) for clip in range(1, 21)]
    train, val = split_by_video(videos, 0.1, seed=0)
    assert not set(train) & set(val) and sorted(train + val) == videos
    assert [v.split("_")[0] for v in val] == ["01", "01", "02", "02", "03", "03"]  # 2 of 20 per scene
    assert split_by_video(videos, 0.1, seed=0) == (train, val)


def test_split_by_video_keeps_at_least_one_validation_video_per_scene():
    train, val = split_by_video(["01_001", "01_002", "02_001"], 0.1, seed=0)
    assert len(val) == 2 and {v.split("_")[0] for v in val} == {"01", "02"}


def test_build_dataset_samples_one_frame_in_step_and_writes_labels(tmp_path):
    videos_dir, pose_dir, out = tmp_path / "videos", tmp_path / "pose", tmp_path / "out"
    videos_dir.mkdir()
    pose_dir.mkdir()
    for video in ("01_001", "01_002", "01_003", "01_004"):
        writer = cv2.VideoWriter(str(videos_dir / f"{video}.avi"), cv2.VideoWriter_fourcc(*"MJPG"), 24, (64, 48))
        for _ in range(12):  # frames 0..11 -> sampled with step 5: 0, 5, 10
            writer.write(np.full((48, 64, 3), 100, np.uint8))
        writer.release()
        # one person on frames 0..11 except frame 5 (pose json uses 0-based frame keys)
        kp = _person({0: (10, 10), 1: (30, 10), 2: (10, 40)})
        poses = [TrackedPose(f, 1, kp, 0.9) for f in range(12) if f != 5]
        write_stgnf_json(poses, pose_dir / f"{video}_alphapose_tracked_person.json")
    manifest = build_dataset(videos_dir, pose_dir, out, step=5, val_fraction=0.5, seed=0)  # scene 01: 4 videos, 2 to val
    assert manifest["images_train"] == 6 and manifest["images_val"] == 6
    assert manifest["persons"] == 8 and manifest["empty_images"] == 4  # frame 5 of every video has nobody
    assert manifest["videos_header_mismatch"] == [] and manifest["videos_pose_beyond_video"] == []
    train_video = json.loads((out / "split.json").read_text())["train"][0]
    assert (out / "images/train" / f"{train_video}_0010.jpg").exists() and not (out / "images/train" / f"{train_video}_0001.jpg").exists()
    assert (out / f"labels/train/{train_video}_0005.txt").read_text() == ""
    assert len((out / f"labels/train/{train_video}_0000.txt").read_text().split()) == 5 + 17 * 3
    assert "kpt_shape: [17, 3]" in (out / "dataset.yaml").read_text()
