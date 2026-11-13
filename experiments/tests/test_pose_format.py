"""Guards the YOLO -> STG-NF JSON converter: our files must be read by STG-NF exactly like their own."""

import sys
from pathlib import Path

import numpy as np
import pytest

from edgelab.pose.stgnf_eval import STGNF_ROOT
from edgelab.pose.stgnf_format import (TrackedPose, poses_from_stgnf_json, read_stgnf_json, to_stgnf_dict,
                                       write_stgnf_json)

RELEASED_FILE = STGNF_ROOT / "data/ShanghaiTech/pose/test/01_0014_alphapose_tracked_person.json"
needs_stgnf = pytest.mark.skipif(not (STGNF_ROOT / "dataset.py").exists(), reason="third_party/STG-NF not cloned")
needs_released = pytest.mark.skipif(not RELEASED_FILE.exists(), reason="STG-NF data.zip not extracted")


def _stgnf_modules():
    sys.path.insert(0, str(STGNF_ROOT))
    import utils.pose_utils as pose_utils
    return pose_utils


def _synthetic_poses(rng):
    poses = []
    for track_id in (3, 7):
        for frame in range(5, 40):
            kps = np.column_stack([rng.uniform(0, 856, 17), rng.uniform(0, 480, 17), rng.uniform(0, 1, 17)])
            poses.append(TrackedPose(frame, track_id, kps, float(rng.uniform(0.3, 1))))
    return poses


def test_round_trip_preserves_every_value(tmp_path):
    poses = _synthetic_poses(np.random.default_rng(0))
    back = poses_from_stgnf_json(write_stgnf_json(poses, tmp_path / "01_0001_yolo_tracked_person.json"))
    key = lambda p: (p.track_id, p.frame)
    for a, b in zip(sorted(poses, key=key), sorted(back, key=key)):
        assert (a.track_id, a.frame) == (b.track_id, b.frame)
        np.testing.assert_allclose(a.keypoints, b.keypoints, rtol=0, atol=1e-9)
        assert a.score == pytest.approx(b.score)


def test_frame_keys_are_zero_padded_so_string_sort_equals_numeric_sort():
    poses = [TrackedPose(f, 1, np.zeros((17, 3)), 1.0) for f in (9, 10, 100)]
    assert list(to_stgnf_dict(poses)["1"].keys()) == ["0009", "0010", "0100"]


@needs_stgnf
@needs_released
def test_our_parser_reads_released_alphapose_file_like_stgnf():
    pose_utils = _stgnf_modules()
    import json
    clip = json.loads(RELEASED_FILE.read_text())
    ours = read_stgnf_json(RELEASED_FILE)
    assert set(ours) == {int(k) for k in clip}
    for pid in clip:
        their_kps, their_meta, their_keys, their_scores = pose_utils.single_pose_dict2np(clip, pid)
        frames, kps, scores = ours[int(pid)]
        np.testing.assert_array_equal(kps, their_kps)
        np.testing.assert_array_equal(scores, their_scores)
        assert [int(k) for k in their_keys] == list(frames)
        assert their_meta == [int(pid), int(their_keys[0])]


@needs_stgnf
@needs_released
def test_rewritten_released_file_gives_identical_stgnf_segments(tmp_path):
    pose_utils = _stgnf_modules()
    import json
    original = json.loads(RELEASED_FILE.read_text())
    rewritten = json.loads(write_stgnf_json(poses_from_stgnf_json(RELEASED_FILE), tmp_path / "x.json").read_text())

    def segments(clip):
        segs, meta, _, _, _, scores = pose_utils.gen_clip_seg_data_np(clip, 0, 1, 24, scene_id="01", clip_id="0014",
                                                                      ret_keys=True, global_pose_data=[])
        return segs, meta, scores

    a, b = segments(original), segments(rewritten)
    np.testing.assert_allclose(a[0], b[0], rtol=0, atol=1e-9)
    assert a[1] == b[1]
    np.testing.assert_allclose(a[2], b[2])
