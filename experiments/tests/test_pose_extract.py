"""Guards frame indexing: frame N of the source must become JSON frame key N (GT masks are indexed the same way)."""

import cv2
import numpy as np

from edgelab.pose.extract import iter_frames


def test_jpg_folder_uses_file_number_as_frame_index(tmp_path):
    for i in (0, 1, 2, 10):
        cv2.imwrite(str(tmp_path / f"{i:03d}.jpg"), np.full((8, 8, 3), i, np.uint8))
    assert [i for i, _ in iter_frames(tmp_path)] == [0, 1, 2, 10]


def test_video_frames_are_numbered_from_zero(tmp_path):
    path = tmp_path / "01_001.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 24, (16, 16))
    for i in range(5):
        writer.write(np.full((16, 16, 3), i * 40, np.uint8))
    writer.release()
    frames = list(iter_frames(path))
    assert [i for i, _ in frames] == [0, 1, 2, 3, 4]
    assert frames[0][1].shape == (16, 16, 3)
