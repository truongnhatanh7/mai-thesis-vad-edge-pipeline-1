"""UBnormal specifics for the STG-NF wrappers: video names, test list, ground truth. (EXP-006)

Everything here follows STG-NF's own UBnormal code (third_party/STG-NF: dataset.py, utils/scoring_utils.py):
 - a test video is called e.g. `abnormal_scene_11_scenario_3_fog`; the pose file is `<video>_alphapose_tracked_person.json`
   and the ground truth `<video>_tracks.txt` (really a NumPy file, one value per frame, 1 = NORMAL, 0 = anomalous);
 - STG-NF reads type / scene / clip out of the file name with a regular expression (UB_NAME below);
 - the test videos are the ones in the released `data/UBnormal/pose/test` folder (list kept in test_videos.txt).
"""

import re
from pathlib import Path

import numpy as np

from edgelab.pose.stgnf_eval import EXPERIMENTS_ROOT, STGNF_ROOT

CHECKPOINT = STGNF_ROOT / "checkpoints/UBnormal_unsupervised_71_8.tar"
GT_DIR = STGNF_ROOT / "data/UBnormal/gt"
TEST_LIST = EXPERIMENTS_ROOT / "EXP-006-ubnormal-replication/test_videos.txt"

# STG-NF's own regex on the pose file name (dataset.py); the GT name uses the same pattern with '_tracks' instead.
UB_POSE_NAME = r"(abnormal|normal)_scene_(\d+)_scenario(.*)_alphapose_.*"

# STG-NF finds the scene and clip of a UBnormal pose file with UB_NAME, which needs the word "alphapose" in the name.
# Our replay writes `<video>_<tag>_tracked_person.json`, so for UBnormal the tag starts with this word. It is only a
# file-name token (the real pose source is recorded in the MLflow run), not a claim that AlphaPose made the poses.
POSE_FILE_TOKEN = "alphapose"


def parse_video_name(name: str) -> tuple[str, str]:
    """'abnormal_scene_11_scenario_3_fog' -> ('11', 'abnormal__3_fog'): the strings STG-NF puts in its segment metadata.

    Uses STG-NF's own regex (clip id = type + '_' + whatever follows 'scenario'), on the name as a pose file would have it.
    """
    found = re.findall(UB_POSE_NAME, f"{name}_{POSE_FILE_TOKEN}_x")
    if not found:
        raise ValueError(f"not a UBnormal video name: {name!r}")
    kind, scene_id, rest = found[0]
    return scene_id, f"{kind}_{rest}"


def video_of_pose_file(filename: str) -> str:
    """'<video>_alphapose_<anything>_tracked_person.json' -> '<video>' (the part before the last '_alphapose_')."""
    stem, token, _ = filename.rpartition(f"_{POSE_FILE_TOKEN}_")
    if not token:
        raise ValueError(f"no '_{POSE_FILE_TOKEN}_' in pose file name {filename!r} (STG-NF's regex would not match)")
    return stem


def load_test_list(path: Path = TEST_LIST) -> list[str]:
    return [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]


def load_gt(video: str, gt_dir: Path = GT_DIR) -> np.ndarray:
    """Frame labels of one test video in OUR convention: 1 = anomalous, 0 = normal (STG-NF's files have 1 = normal)."""
    normal = np.load(Path(gt_dir) / f"{video}_tracks.txt")  # NumPy file with a .txt name
    return (normal == 0).astype(int)


def stratum_of(name: str, mean_luma: float, dark_below: float = 60.0) -> str:
    """Light stratum of one test video (plan §2): dark = mean luma < 60; weather = fog/smoke/rain in the name; else normal.

    The plan lists dark first, so a video that is both goes to dark (strata.csv keeps both flags).
    """
    if mean_luma < dark_below:
        return "dark"
    return "weather" if is_weather(name) else "normal"


def is_weather(name: str) -> bool:
    return any(word in name for word in ("fog", "smoke", "rain"))


def cache_length_problems(cache_dir: Path, gt_dir: Path = GT_DIR, videos: list[str] | None = None) -> list[str]:
    """Check a detector cache against the GT: every video present, frame ids 0..n-1, n = GT length. [] = all fine."""
    problems = []
    for video in videos if videos is not None else load_test_list():
        path = Path(cache_dir) / f"{video}.npz"
        if not path.exists():
            problems.append(f"{video}: no cache file")
            continue
        frame_ids = np.load(path)["frame_ids"]
        gt_length = len(load_gt(video, gt_dir))
        if len(frame_ids) != gt_length or not np.array_equal(frame_ids, np.arange(gt_length)):
            problems.append(f"{video}: cache has {len(frame_ids)} frames, GT has {gt_length}")
    return problems


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="check a detector cache against the UBnormal GT lengths")
    parser.add_argument("--check-cache", type=Path, required=True)
    parser.add_argument("--videos", nargs="*", default=None)
    args = parser.parse_args()
    found = cache_length_problems(args.check_cache, videos=args.videos)
    print("\n".join(found) if found else "cache OK: every video has the GT's frame count")
    raise SystemExit(1 if found else 0)
