"""Guards the EXP-009 fill rules on toy tracks: exact values, gap cap, stop at next detection, provenance sidecar."""

import json

import numpy as np
import pytest

from edgelab.pose.fill import fill_B, fill_clip, fill_dir, fill_P, segment_predicted_shares
from edgelab.pose.stgnf_format import read_stgnf_json


def pose(x, y=0.0, conf=1.0, score=1.0):
    """17 keypoints; keypoint i sits at (x + i, y + 2 i) so a wrong keypoint index shows up in the numbers."""
    keypoints = []
    for i in range(17):
        keypoints += [x + i, y + 2 * i, conf]
    return {"keypoints": keypoints, "scores": score}


def kp(p, i):
    return p["keypoints"][3 * i: 3 * i + 3]


def test_B_interpolates_x_y_conf_and_score_linearly():
    track = {"0000": pose(0, 0, 1.0, 0.8), "0004": pose(8, 4, 0.2, 0.4)}
    filled, predicted = fill_B(track, max_gap=8)
    assert sorted(filled) == ["0000", "0001", "0002", "0003", "0004"]
    # frame 1 is 1/4 of the way, frame 3 is 3/4 of the way
    assert kp(filled["0001"], 0) == pytest.approx([2.0, 1.0, 0.8])
    assert kp(filled["0003"], 0) == pytest.approx([6.0, 3.0, 0.4])
    assert kp(filled["0002"], 5) == pytest.approx([4.0 + 5, 2.0 + 10, 0.6])  # keypoint 5 interpolates its own x, y
    assert filled["0002"]["scores"] == pytest.approx(0.6)
    assert filled["0000"] == track["0000"] and filled["0004"] == track["0004"]  # detections untouched


def test_B_records_age_of_every_made_up_pose_and_nothing_else():
    track = {"0000": pose(0), "0004": pose(8), "0005": pose(9)}
    _, predicted = fill_B(track, max_gap=8)
    assert predicted == {"0001": 1, "0002": 2, "0003": 3}


def test_B_fills_a_gap_of_exactly_G_but_leaves_a_longer_gap_empty():
    track = {"0000": pose(0), "0004": pose(4), "0009": pose(9)}  # gaps of 3 and 4 missing frames
    filled, predicted = fill_B(track, max_gap=3)
    assert {"0001", "0002", "0003"} <= set(filled)
    assert not {"0005", "0006", "0007", "0008"} & set(filled)  # the 4-frame gap is not touched, not even partly
    assert set(predicted) == {"0001", "0002", "0003"}


def test_B_does_not_extend_past_the_last_detection():
    filled, predicted = fill_B({"0000": pose(0), "0001": pose(1)}, max_gap=8)
    assert sorted(filled) == ["0000", "0001"] and predicted == {}


def test_P_damped_velocity_and_confidence_decay():
    track = {"0000": pose(0, conf=0.9, score=0.7), "0001": pose(10, conf=0.9, score=0.7)}  # velocity 10 px / frame
    filled, predicted = fill_P(track, max_gap=3)
    # step k moves velocity * 0.8**k: 10 + 8 = 18, then + 6.4 = 24.4, then + 5.12 = 29.52
    assert kp(filled["0002"], 0)[0] == pytest.approx(18.0)
    assert kp(filled["0003"], 0)[0] == pytest.approx(24.4)
    assert kp(filled["0004"], 0)[0] == pytest.approx(29.52)
    assert kp(filled["0002"], 0)[2] == pytest.approx(0.9 * 0.8)
    assert kp(filled["0004"], 0)[2] == pytest.approx(0.9 * 0.8 ** 3)
    assert filled["0003"]["scores"] == pytest.approx(0.7 * 0.64)
    assert kp(filled["0003"], 4)[1] == pytest.approx(8.0)  # y has no velocity: keypoint 4 stays at y = 2 * 4
    assert predicted == {"0002": 1, "0003": 2, "0004": 3}


def test_P_stops_at_the_next_detection_and_at_the_cap():
    track = {"0000": pose(0), "0001": pose(1), "0004": pose(4)}  # missing 2, 3
    filled, predicted = fill_P(track, max_gap=8, last_frame=4)  # no tail after the last detection
    assert set(predicted) == {"0002", "0003"} and filled["0004"] == track["0004"]
    capped, predicted = fill_P({"0000": pose(0), "0001": pose(1)}, max_gap=2)
    assert set(predicted) == {"0002", "0003"} and "0004" not in capped


def test_P_with_one_detection_holds_position_and_never_goes_past_the_video_end():
    filled, predicted = fill_P({"0005": pose(3, conf=1.0)}, max_gap=4, last_frame=7)
    assert set(predicted) == {"0006", "0007"}
    assert kp(filled["0006"], 0)[:2] == [3.0, 0.0] and kp(filled["0006"], 0)[2] == pytest.approx(0.8)


def test_P_velocity_comes_from_the_last_two_detections_even_across_a_gap():
    track = {"0000": pose(0), "0004": pose(8)}  # 2 px / frame
    filled, _ = fill_P(track, max_gap=2)
    assert kp(filled["0005"], 0)[0] == pytest.approx(8 + 2 * 0.8)


def test_G_zero_changes_nothing():
    track = {"0000": pose(0), "0003": pose(3)}
    assert fill_B(track, 0) == (track, {}) and fill_P(track, 0) == (track, {})


def test_unknown_mode_is_an_error():
    with pytest.raises(ValueError):
        fill_clip({"1": {"0000": pose(0)}}, "X", 4)


def _write_clip(path, clip):
    path.write_text(json.dumps(clip))


def test_fill_dir_keeps_the_stgnf_format_and_writes_provenance_elsewhere(tmp_path):
    src, out = tmp_path / "in", tmp_path / "out"
    src.mkdir()
    _write_clip(src / "01_0001_x_tracked_person.json", {"3": {"0000": pose(0), "0003": pose(3)}})
    stats = fill_dir(src, out, "B", 8)
    assert stats["predicted_poses"] == 2 and stats["detected_poses"] == 2
    written = json.loads((out / "01_0001_x_tracked_person.json").read_text())
    assert all(set(p) == {"keypoints", "scores"} for p in written["3"].values())  # nothing STG-NF could choke on
    frames, keypoints, _ = read_stgnf_json(out / "01_0001_x_tracked_person.json")[3]
    assert frames.tolist() == [0, 1, 2, 3] and keypoints.shape == (4, 17, 3)
    sidecar = json.loads((tmp_path / "out_prov" / "01_0001_x_tracked_person.provenance.json").read_text())
    assert sidecar == {"3": {"0001": 1, "0002": 2}}
    assert not list(out.glob("*provenance*"))


def test_fill_dir_P_stops_at_video_end_from_frame_counts(tmp_path):
    src, out = tmp_path / "in", tmp_path / "out"
    src.mkdir()
    _write_clip(src / "01_0001_x_tracked_person.json", {"1": {"0000": pose(0), "0001": pose(1)}})
    fill_dir(src, out, "P", 8, n_frames={"01_0001": 4})  # frames 0..3 exist
    assert sorted(json.loads((out / "01_0001_x_tracked_person.json").read_text())["1"]) == ["0000", "0001", "0002", "0003"]


def test_predicted_share_of_a_window_counts_predicted_poses_among_its_24(tmp_path):
    src, out = tmp_path / "in", tmp_path / "out"
    src.mkdir()
    track = {f"{f:04d}": pose(f) for f in list(range(0, 5)) + list(range(11, 40))}  # frames 5..10 missing -> predicted by B
    _write_clip(src / "01_0001_x_tracked_person.json", {"7": track})
    fill_dir(src, out, "B", 8)
    metadata = np.array([[1, 1, 7, 0], [1, 1, 7, 10], [1, 1, 7, 11]])  # windows 0..23, 10..33, 11..34
    shares = segment_predicted_shares(metadata, out, tmp_path / "out_prov")
    assert shares.tolist() == pytest.approx([6 / 24, 1 / 24, 0.0])
    assert segment_predicted_shares(metadata, out, None).tolist() == [0, 0, 0]


def test_new_keys_follow_the_width_of_the_tracks_own_keys():
    # one released video writes frames as 3 digits; STG-NF sorts keys as text, so '0001' next to '000' would reorder poses
    track = {"000": pose(0), "003": pose(3)}
    filled_b, predicted_b = fill_B(track, 8)
    assert sorted(filled_b) == ["000", "001", "002", "003"] and set(predicted_b) == {"001", "002"}
    filled_p, _ = fill_P({"000": pose(0), "001": pose(1)}, 2)
    assert sorted(filled_p) == ["000", "001", "002", "003"]
