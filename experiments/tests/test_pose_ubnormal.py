"""EXP-006 guards: UBnormal video names, ground truth, frame-score assembly, per-video strata."""

import argparse
import json
import os
import re
import sys

import cv2
import numpy as np
import pytest

from edgelab.analysis.exp006_strata import strata_row, summarize, video_luma
from edgelab.pose.stgnf_eval import STGNF_ROOT, assemble_clip_scores, clip_normality, score_frame_offsets
from edgelab.pose.ubnormal import (UB_POSE_NAME, load_gt, load_test_list, parse_video_name, stratum_of,
                                   video_of_pose_file)

needs_stgnf = pytest.mark.skipif(not (STGNF_ROOT / "utils/scoring_utils.py").exists(), reason="STG-NF not cloned")

NAMES = ["abnormal_scene_1_scenario_1", "normal_scene_1_scenario1_4", "abnormal_scene_11_scenario_3_fog",
         "normal_scene_29_scenario_2_smoke"]


def _save_gt(gt_dir, video, normal_flags):
    """STG-NF's GT files are NumPy arrays named *_tracks.txt (1 = normal)."""
    gt_dir.mkdir(exist_ok=True)
    with open(gt_dir / f"{video}_tracks.txt", "wb") as f:  # np.save(path) would append '.npy'
        np.save(f, np.asarray(normal_flags, dtype=float))


# ---------- names ----------

def test_parse_video_name_by_hand():
    assert parse_video_name("abnormal_scene_11_scenario_3_fog") == ("11", "abnormal__3_fog")
    assert parse_video_name("normal_scene_1_scenario1_4") == ("1", "normal_1_4")


@pytest.mark.parametrize("video", NAMES)
def test_parse_video_name_matches_stgnf_regex_on_pose_and_gt_file_names(video):
    pose_file = f"{video}_alphapose_tracked_person.json"
    kind, scene, rest = re.findall(r"(abnormal|normal)_scene_(\d+)_scenario(.*)_alphapose_.*", pose_file)[0]
    assert parse_video_name(video) == (scene, f"{kind}_{rest}")
    gt_file = f"{video}_tracks.txt"  # the GT side of STG-NF's evaluation uses the same pattern with '_tracks'
    kind, scene, rest = re.findall(r"(abnormal|normal)_scene_(\d+)_scenario(.*)_tracks.*", gt_file)[0]
    assert parse_video_name(video) == (scene, f"{kind}_{rest}")


def test_unknown_name_is_rejected():
    with pytest.raises(ValueError):
        parse_video_name("01_0014")


def test_video_of_pose_file_needs_the_alphapose_token():
    assert video_of_pose_file("abnormal_scene_1_scenario_1_alphapose_L0-E0-c0.25-botsort_tracked_person.json") == \
        "abnormal_scene_1_scenario_1"
    with pytest.raises(ValueError):
        video_of_pose_file("abnormal_scene_1_scenario_1_L0-E0-c0.25-botsort_tracked_person.json")


def test_replay_writes_ubnormal_files_that_stgnf_parses(tmp_path):
    """The file name our replay writes must give STG-NF the same scene / clip as the plain video name."""
    from edgelab.pose.replay import replay_dir
    from edgelab.pose.detcache import save_atomic
    cache = tmp_path / "L0_E0"
    n = 12  # one person, 40x100 box moving 4 px per frame
    save_atomic(cache / "abnormal_scene_11_scenario_3_fog.npz", {
        "boxes": np.array([[4 * t, 50, 4 * t + 40, 150] for t in range(n)], np.float32),
        "conf": np.full(n, 0.9, np.float32), "cls": np.zeros(n, np.float32),
        "keypoints": np.ones((n, 17, 3), np.float32), "counts": np.ones(n, int), "frame_ids": np.arange(n),
        "meta": np.array("{}")})
    replay_dir(cache, 0.25, tmp_path / "poses", dataset="UBnormal")
    (written,) = (tmp_path / "poses").glob("*_tracked_person.json")
    assert written.name == "abnormal_scene_11_scenario_3_fog_alphapose_L0-E0-c0.25-botsort_tracked_person.json"
    kind, scene, rest = re.findall(UB_POSE_NAME, written.name)[0]
    assert (scene, f"{kind}_{rest}") == parse_video_name("abnormal_scene_11_scenario_3_fog")
    assert video_of_pose_file(written.name) == "abnormal_scene_11_scenario_3_fog"


def test_shanghaitech_file_names_are_unchanged(tmp_path):
    from edgelab.pose.replay import source_tag
    assert source_tag(tmp_path / "L2_E1", 0.25, "botsort") == "L2-E1-c0.25-botsort"


def test_test_list_has_211_unique_videos_that_all_parse():
    videos = load_test_list()
    assert len(videos) == len(set(videos)) == 211
    assert len({parse_video_name(v) for v in videos}) == 211  # (scene, clip id) identifies a video


# ---------- ground truth and frame scores ----------

def test_load_gt_flips_to_1_is_anomalous(tmp_path):
    _save_gt(tmp_path, "abnormal_scene_1_scenario_1", [1, 1, 0, 0, 1])
    assert load_gt("abnormal_scene_1_scenario_1", tmp_path).tolist() == [0, 0, 1, 1, 0]


def test_score_frame_offsets_per_dataset():
    assert score_frame_offsets("ShanghaiTech") == {"middle": 12, "end": 23}
    assert score_frame_offsets("UBnormal") == {"middle": 8, "end": 15}  # STG-NF runs UBnormal with seg_len 16


def _toy(tmp_path):
    """Three videos. A: 80 frames, anomalous 30-49, two people. B: 60 normal frames, one person. C: 50 frames, nobody."""
    gt_dir = tmp_path / "gt"
    gt_a = np.ones(80)
    gt_a[30:50] = 0
    _save_gt(gt_dir, "abnormal_scene_1_scenario_1", gt_a)
    _save_gt(gt_dir, "normal_scene_2_scenario3_1", np.ones(60))
    _save_gt(gt_dir, "abnormal_scene_3_scenario_2_fog", np.ones(50))
    rng = np.random.default_rng(0)
    metadata, normality = [], []
    for scene, clip, person, starts in [(1, "abnormal__1", 1, range(0, 60)), (1, "abnormal__1", 2, range(20, 50, 2)),
                                        (2, "normal_3_1", 7, range(0, 40))]:
        for start in starts:
            metadata.append([scene, clip, person, start])
            normality.append(rng.normal())
    return gt_dir, np.array(metadata), np.array(normality)  # np.array makes the metadata strings, as in STG-NF


def test_clip_normality_places_scores_on_the_middle_frame_and_flips_labels(tmp_path):
    gt_dir, metadata, normality = _toy(tmp_path)
    names, labels, per_clip = clip_normality(normality, metadata, gt_dir, dataset="UBnormal",
                                             clips=["abnormal_scene_1_scenario_1", "normal_scene_2_scenario3_1",
                                                    "abnormal_scene_3_scenario_2_fog"])
    assert names[0] == "abnormal_scene_1_scenario_1"
    assert labels[0].tolist() == [0] * 30 + [1] * 20 + [0] * 30
    # person 1 covers frames 8..67 (start + 8); person 2 covers 28,30,...,56; the clip score is the lower normality
    assert np.isfinite(per_clip[0]).nonzero()[0].tolist() == list(range(8, 68))
    assert per_clip[0][8] == normality[0] and per_clip[0][67] == normality[59]
    both = {28 + 2 * k: normality[60 + k] for k in range(15)}
    for frame, n_two in both.items():
        assert per_clip[0][frame] == min(n_two, normality[frame - 8])
    assert np.isinf(per_clip[2]).all()  # nobody ever detected


def test_assemble_gives_nobody_frames_the_most_normal_score(tmp_path):
    gt_dir, metadata, normality = _toy(tmp_path)
    clips = ["abnormal_scene_1_scenario_1", "normal_scene_2_scenario3_1", "abnormal_scene_3_scenario_2_fog"]
    _, _, scores = assemble_clip_scores(normality, metadata, gt_dir, dataset="UBnormal", clips=clips)
    assert (scores[2] == -normality.max()).all()  # score = -normality, nobody = most normal of the dataset
    assert scores[0][0] == -normality.max()


@needs_stgnf
def test_assembly_matches_stgnf_get_dataset_scores_for_ubnormal(tmp_path):
    gt_dir, metadata, normality = _toy(tmp_path)
    clips = ["abnormal_scene_1_scenario_1", "normal_scene_2_scenario3_1", "abnormal_scene_3_scenario_2_fog"]
    names, labels, scores = assemble_clip_scores(normality, metadata, gt_dir, dataset="UBnormal", clips=clips)

    sys.path.insert(0, str(STGNF_ROOT))
    from utils import scoring_utils
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:  # STG-NF reads the test list and the GT from these fixed relative paths
        os.makedirs("data/UBnormal/pose/test")
        for video in clips:
            open(f"data/UBnormal/pose/test/{video}_alphapose_tracked_person.json", "w").close()
        os.symlink(gt_dir, "data/UBnormal/gt")
        gt, theirs = scoring_utils.get_dataset_scores(normality.copy(), metadata,
                                                      args=argparse.Namespace(dataset="UBnormal", seg_len=16))
    finally:
        os.chdir(cwd)
    order = sorted(clips)  # STG-NF walks the clips in sorted order of their GT file names
    assert len(theirs) == 3
    for name, their_score, their_gt in zip(order, theirs, gt):
        ours = names.index(name)
        np.testing.assert_allclose(scores[ours], -their_score)
        np.testing.assert_array_equal(labels[ours], 1 - their_gt.astype(int))  # their 1 = normal, ours 1 = anomalous


# ---------- strata ----------

def test_stratum_rule():
    assert stratum_of("abnormal_scene_1_scenario_1", 59.9) == "dark"
    assert stratum_of("abnormal_scene_1_scenario_1", 60.0) == "normal"  # dark is strictly below 60
    assert stratum_of("abnormal_scene_11_scenario_1_fog", 120) == "weather"
    assert stratum_of("abnormal_scene_5_scenario_1_smoke", 120) == "weather"
    assert stratum_of("normal_scene_9_scenario_1_rain", 120) == "weather"
    assert stratum_of("abnormal_scene_11_scenario_1_fog", 40) == "dark"  # the plan lists dark first


def _row(video, luma, labels):
    probe = {"frames": len(labels), "mean_luma": luma, "width": 1280, "height": 720, "fps": 30.0}
    return strata_row(video, probe, np.array(labels))


def test_summarize_counts_videos_with_and_without_anomalous_frames():
    rows = [_row("abnormal_scene_1_scenario_1", 100, [0, 1, 1]), _row("normal_scene_1_scenario1_1", 100, [0, 0]),
            _row("abnormal_scene_2_scenario_1", 30, [1, 1, 0, 0]), _row("normal_scene_3_scenario_1_fog", 90, [0] * 5)]
    result = summarize(rows)
    assert result["normal"] == {"videos": 2, "with_anomaly": 1, "without_anomaly": 1, "normal_frames": 3,
                                "anomalous_frames": 2}
    assert result["dark"]["videos"] == 1 and result["dark"]["anomalous_frames"] == 2
    assert result["weather"] == {"videos": 1, "with_anomaly": 0, "without_anomaly": 1, "normal_frames": 5,
                                 "anomalous_frames": 0}
    assert rows[0]["scene"] == 1 and rows[2]["scene"] == 2 and rows[0]["gt_frames"] == 3


def test_video_luma_matches_the_exp000_method(tmp_path):
    from edgelab.data.ucf_crime import lighting_stats
    path = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (64, 48))
    for i in range(35):
        writer.write(np.full((48, 64, 3), 20 + 5 * i, np.uint8))  # a slowly brightening grey clip
    writer.release()
    ours = video_luma(path)
    assert ours["frames"] == 35 and (ours["width"], ours["height"]) == (64, 48)
    assert ours["mean_luma"] == pytest.approx(lighting_stats(path)["mean_luma"])  # same 1-per-second sampling (10 fps)
    assert json.dumps(ours["mean_luma"])  # plain float


# ---------- detector cache on video files ----------

def _write_mp4(path, n_frames=5, size=(128, 96)):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, size)
    for i in range(n_frames):
        writer.write(np.random.default_rng(i).integers(0, 255, (size[1], size[0], 3), np.uint8))
    writer.release()


def test_detcache_lists_and_finds_video_files_and_frame_folders(tmp_path):
    from edgelab.pose.detcache import find_source, list_videos
    _write_mp4(tmp_path / "abnormal_scene_1_scenario_1.mp4", 2)
    (tmp_path / "01_0014").mkdir()
    (tmp_path / "notes.txt").write_text("not a video")
    assert list_videos(tmp_path) == ["01_0014", "abnormal_scene_1_scenario_1"]
    assert find_source(tmp_path, "01_0014") == tmp_path / "01_0014"
    assert find_source(tmp_path, "abnormal_scene_1_scenario_1") == tmp_path / "abnormal_scene_1_scenario_1.mp4"
    with pytest.raises(FileNotFoundError):
        find_source(tmp_path, "missing")


def test_cache_video_accepts_an_mp4_and_names_it_by_its_stem(tmp_path):
    pytest.importorskip("ultralytics", reason="needs `uv sync --extra pose`")
    from edgelab.pose.detcache import WEIGHTS, cache_video
    if not WEIGHTS.exists():
        pytest.skip("YOLO weights missing")
    from ultralytics import YOLO
    _write_mp4(tmp_path / "abnormal_scene_1_scenario_1.mp4", 5)
    arrays = cache_video(YOLO(str(WEIGHTS)), tmp_path / "abnormal_scene_1_scenario_1.mp4", "L0", "E0", "cpu")
    assert arrays["frame_ids"].tolist() == [0, 1, 2, 3, 4]  # one cached row per decoded frame
    assert json.loads(str(arrays["meta"]))["video"] == "abnormal_scene_1_scenario_1"


def test_cache_length_problems(tmp_path):
    from edgelab.pose.ubnormal import cache_length_problems
    gt_dir, cache = tmp_path / "gt", tmp_path / "cache"
    cache.mkdir()
    _save_gt(gt_dir, "abnormal_scene_1_scenario_1", np.ones(5))
    _save_gt(gt_dir, "normal_scene_1_scenario1_1", np.ones(4))
    _save_gt(gt_dir, "abnormal_scene_1_scenario_2", np.ones(3))
    np.savez(cache / "abnormal_scene_1_scenario_1.npz", frame_ids=np.arange(5))
    np.savez(cache / "normal_scene_1_scenario1_1.npz", frame_ids=np.arange(3))  # one frame short
    problems = cache_length_problems(cache, gt_dir, ["abnormal_scene_1_scenario_1", "normal_scene_1_scenario1_1",
                                                     "abnormal_scene_1_scenario_2"])
    assert problems == ["normal_scene_1_scenario1_1: cache has 3 frames, GT has 4",
                        "abnormal_scene_1_scenario_2: no cache file"]


def test_letterbox_shapes_for_the_ubnormal_frame_sizes():
    """Plan §2 assumes 1280x720 gives the same 384x640 input as ShanghaiTech's 856x480 at imgsz 640 (it does)."""
    pytest.importorskip("ultralytics", reason="needs `uv sync --extra pose`")
    from edgelab.pose.detcache import WEIGHTS, letterbox_shape
    if not WEIGHTS.exists():
        pytest.skip("YOLO weights missing")
    from ultralytics import YOLO
    model = YOLO(str(WEIGHTS))
    assert letterbox_shape(model, 1280, 720, 640) == (384, 640)
    assert letterbox_shape(model, 856, 480, 640) == (384, 640)
    assert letterbox_shape(model, 1280, 720, 960) == (544, 960)
    assert letterbox_shape(model, 1080, 720, 640) == (448, 640)  # the narrower UBnormal scenes pad more
