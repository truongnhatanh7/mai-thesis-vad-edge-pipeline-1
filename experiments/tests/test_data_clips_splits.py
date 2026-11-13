import shutil
import subprocess

import numpy as np
import pytest

from edgelab.data.splits import PARTITIONS, leaky_clip_split, load_split, save_split, shared_videos, video_level_split
from edgelab.data.video_clips import ClipDataset, ClipItem, clip_starts, decode_video, make_clip_items, to_model_input


def _toy_items(n_videos=100, clips_per_video=4):
    return [ClipItem(f"v{v}", s * 2, v % 2) for v in range(n_videos) for s in range(clips_per_video)]


def test_clip_starts_hockey_and_long_video():
    assert clip_starts(14, 8, 2, 8) == [0, 2, 4, 6]
    long = clip_starts(100, 8, 2, 8)
    assert len(long) == 8 and long[0] == 0 and long[-1] == 92
    assert clip_starts(5, 8, 2, 8) == [0]


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_video_level_split_has_zero_overlap_and_covers_everything(seed):
    items = _toy_items()
    split = video_level_split(items, seed)
    assert shared_videos(items, split) == 0
    assert sorted(i for p in PARTITIONS for i in split[p]) == list(range(len(items)))
    assert 14 <= len({items[i].video_id for i in split["test"]}) <= 16


def test_video_level_split_is_stratified_and_seeded():
    items = _toy_items()
    a, b, c = video_level_split(items, 0), video_level_split(items, 0), video_level_split(items, 1)
    assert a == b and a != c
    test_labels = [items[i].label for i in a["test"]]
    assert abs(np.mean(test_labels) - 0.5) < 0.1


def test_leaky_split_really_leaks():
    items = _toy_items()
    assert shared_videos(items, leaky_clip_split(items, 0)) > 0


def test_split_roundtrip(tmp_path):
    items = _toy_items(20)
    split = video_level_split(items, 3)
    save_split(split, items, tmp_path / "s.json", {"seed": 3})
    assert load_split(tmp_path / "s.json", items) == split


def test_clip_dataset_pads_short_video_and_normalises(tmp_path):
    np.save(tmp_path / "a.npy", np.arange(5, dtype=np.uint8)[:, None, None, None].repeat(4, 1).repeat(4, 2).repeat(3, 3))
    ds = ClipDataset(tmp_path, make_clip_items([{"video_id": "a", "label": 1, "num_frames": 5}]), clip_len=8)
    clip, label, _ = ds[0]
    assert clip.shape == (8, 4, 4, 3) and label == 1
    assert clip[:, 0, 0, 0].tolist() == [0, 1, 2, 3, 4, 4, 4, 4]
    x = to_model_input(clip[None])
    assert x.shape == (1, 8, 3, 4, 4) and float(x.max()) == pytest.approx(4 / 255)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg missing")
def test_decode_video_keeps_every_third_frame(tmp_path):
    video = tmp_path / "t.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=320x240:rate=25:duration=1.2",
                    "-pix_fmt", "yuv420p", str(video)], check=True)
    frames = decode_video(video)
    assert frames.shape == (10, 256, 256, 3) and frames.dtype == np.uint8  # 30 frames -> every 3rd
