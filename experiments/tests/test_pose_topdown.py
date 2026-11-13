"""EXP-016 tracking-with-keyframes and pose-on-tracked-boxes, on synthetic boxes (real Ultralytics tracker, fake pose model)."""

import cv2
import numpy as np
import pytest

pytest.importorskip("ultralytics")

from edgelab.pose.topdown import (  # noqa: E402
    FrameTracks, IncompleteCacheError, check_cache_complete, clip_boxes, load_boxes, poses_for_dir, pose_on_tracks, save_atomic,
    track_video)


def person_box(x: float, y: float = 100.0, w: float = 40.0, h: float = 100.0) -> np.ndarray:
    return np.array([x, y, x + w, y + h], dtype=np.float32)


def frames_with(boxes_by_frame: dict[int, list[np.ndarray]], n_frames: int, conf: float = 0.9):
    """{frame: [boxes]} -> the per-frame (boxes, conf) list a cache would give."""
    out = []
    for i in range(n_frames):
        boxes = boxes_by_frame.get(i, [])
        out.append((np.array(boxes, dtype=np.float32).reshape(-1, 4), np.full(len(boxes), conf, dtype=np.float32)))
    return out, np.arange(n_frames)


def test_k1_follows_one_walking_person_with_one_id():
    per_frame, ids = frames_with({i: [person_box(100 + 3 * i)] for i in range(8)}, 8)
    tracks = track_video(per_frame, ids, c=0.25, k=1)
    assert all(len(t.track_ids) == 1 for t in tracks)
    assert len({int(t.track_ids[0]) for t in tracks}) == 1
    assert all(t.is_keyframe for t in tracks)


def test_k3_uses_detections_only_on_keyframes_and_predicts_boxes_between():
    # detections exist in every frame, but with K = 3 only frames 0, 3, 6 may be read
    walk = {i: [person_box(100 + 5 * i)] for i in range(9)}
    per_frame, ids = frames_with(walk, 9)
    for i in (1, 2, 4, 5, 7, 8):  # poison the non-keyframes: if they were read the person would jump far away
        per_frame[i] = (person_box(600).reshape(1, 4), per_frame[i][1])
    tracks = track_video(per_frame, ids, c=0.25, k=3)
    assert [t.is_keyframe for t in tracks] == [True, False, False] * 3
    assert all(len(t.track_ids) == 1 for t in tracks)  # a box in every frame
    assert len({int(t.track_ids[0]) for t in tracks}) == 1  # one id throughout
    assert all(t.boxes[0, 0] < 300 for t in tracks)  # nobody ever used the poisoned boxes at x = 600


def test_k3_predicted_box_continues_the_motion_of_the_last_keyframes():
    walk = {i: [person_box(100 + 6 * i)] for i in range(12)}
    per_frame, ids = frames_with(walk, 12)
    tracks = track_video(per_frame, ids, c=0.25, k=3)
    x = [t.boxes[0, 0] for t in tracks]
    # keyframes 0, 3, 6, 9 teach the filter a speed of about 6 px per frame; predictions move the box on, not back
    assert x[7] > x[6] and x[8] > x[7]
    assert abs(x[8] - (100 + 6 * 8)) < 12


def test_k3_person_appearing_between_keyframes_is_not_boxed_until_the_filter_has_seen_it_twice():
    per_frame, ids = frames_with({**{i: [person_box(100)] for i in range(12)},
                                  **{i: [person_box(100), person_box(400)] for i in range(4, 12)}}, 12)
    tracks = track_video(per_frame, ids, c=0.25, k=3)
    assert [len(t.track_ids) for t in tracks[:6]] == [1, 1, 1, 1, 1, 1]  # second person not read at frames 4, 5
    assert len(tracks[6].track_ids) == 1  # keyframe 6: new track created but not confirmed yet
    assert len(tracks[9].track_ids) == 2  # keyframe 9: confirmed by the second keyframe match


def test_detections_below_c_never_reach_the_tracker():
    per_frame, ids = frames_with({i: [person_box(100)] for i in range(5)}, 5, conf=0.2)
    assert all(len(t.track_ids) == 0 for t in track_video(per_frame, ids, c=0.25, k=1))


def test_clip_boxes_keeps_boxes_inside_the_frame():
    clipped = clip_boxes(np.array([[-10.0, -5.0, 50.0, 300.0]]), width=100, height=200)
    assert clipped.tolist() == [[0.0, 0.0, 50.0, 200.0]]


class CountingPose:
    def __init__(self):
        self.boxes_seen = []

    def predict(self, img, boxes):
        self.boxes_seen.append(np.array(boxes))
        out = np.zeros((len(boxes), 17, 3), np.float32)
        out[:, :, 2] = 0.5
        return out


def test_pose_on_tracks_runs_on_every_box_skips_tiny_ones_and_keeps_ids_and_frames(tmp_path):
    source = tmp_path / "99_0001"
    source.mkdir()
    for frame in range(3):
        cv2.imwrite(str(source / f"{frame:03d}.jpg"), np.full((200, 300, 3), 90, np.uint8))
    tracks = [
        FrameTracks(0, np.array([4, 5]), np.array([[10, 10, 60, 120], [100, 20, 101, 21]], float), np.array([0.9, 0.8]), True),
        FrameTracks(1, np.array([], int), np.zeros((0, 4)), np.zeros(0), False),
        FrameTracks(2, np.array([4]), np.array([[20, 10, 70, 120]], float), np.array([0.7]), False),
    ]
    pose = CountingPose()
    poses, stats = pose_on_tracks(tracks, source, "L0", pose)
    assert [(p.frame, p.track_id) for p in poses] == [(0, 4), (2, 4)]  # track 5 is a 1 px box: skipped and counted
    assert stats["pose_calls"] == 2 and stats["too_small"] == 1
    assert len(pose.boxes_seen) == 2  # frame 1 has nobody: no call
    assert poses[0].score == 0.9


def test_cache_roundtrip(tmp_path):
    arrays = {"boxes": np.array([[0, 0, 10, 20], [5, 5, 15, 25], [1, 1, 2, 2]], np.float32),
              "conf": np.array([0.5, 0.6, 0.7], np.float32), "counts": np.array([2, 0, 1]), "frame_ids": np.array([0, 1, 2])}
    save_atomic(tmp_path / "v.npz", arrays)
    per_frame, frame_ids = load_boxes(tmp_path / "v.npz")
    assert [len(b) for b, _ in per_frame] == [2, 0, 1] and frame_ids.tolist() == [0, 1, 2]
    assert per_frame[2][1].tolist() == pytest.approx([0.7])


def fake_cache(tmp_path, n: int, partial: bool = False):
    d = tmp_path / "cache"
    for i in range(n):
        save_atomic(d / f"v{i:03d}.npz", {"x": np.zeros(1)})
    if partial:
        (d / "v999.partial.npz").write_bytes(b"x")
    return d


def test_complete_cache_check_counts_107_and_ignores_partial_files(tmp_path):
    assert check_cache_complete(fake_cache(tmp_path, 107, partial=True)) == 107
    with pytest.raises(IncompleteCacheError, match="4 npz files, 107 expected"):
        check_cache_complete(fake_cache(tmp_path / "a", 4))
    with pytest.raises(IncompleteCacheError):
        check_cache_complete(fake_cache(tmp_path / "b", 108))
    with pytest.raises(IncompleteCacheError):
        check_cache_complete(tmp_path / "does-not-exist")


def test_complete_cache_check_by_names(tmp_path):
    d = fake_cache(tmp_path, 3)
    assert check_cache_complete(d, ["v000", "v001", "v002"]) == 3
    with pytest.raises(IncompleteCacheError, match="v003"):
        check_cache_complete(d, ["v000", "v001", "v002", "v003"])


def test_poses_for_dir_refuses_an_incomplete_cache_before_loading_any_model(tmp_path):
    with pytest.raises(IncompleteCacheError):
        poses_for_dir(fake_cache(tmp_path, 4), tmp_path, "L0", "rtmpose-s", 1, 0.25, tmp_path / "out")
    assert not (tmp_path / "out").exists()
