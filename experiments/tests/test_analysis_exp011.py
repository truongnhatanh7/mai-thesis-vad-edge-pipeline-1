"""EXP-011 helpers: height bins and recall counting on a toy cache."""

import numpy as np

from edgelab.analysis.exp011 import BIN_NAMES, height_bin, pooled, recall, recall_by_bin, wilson


def test_height_bins_have_the_planned_edges():
    assert [height_bin(h) for h in (10, 31.9, 32, 47.9, 48, 63.9, 64, 95.9, 96, 159.9, 160, 400)] == [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
    assert len(BIN_NAMES) == 6


def _toy_cache(tmp_path):
    boxes = np.array([[0.0, 0, 20, 40], [100, 100, 120, 220]], dtype=np.float32)  # frame 0: a 40 px and a 120 px person
    np.savez_compressed(tmp_path / "v.npz", boxes=boxes, conf=np.array([0.9, 0.9], np.float32), cls=np.zeros(2, np.float32),
                        keypoints=np.zeros((2, 17, 3), np.float32), counts=np.array([2, 0]), frame_ids=np.array([0, 1]))
    return tmp_path


def test_recall_counts_found_and_missed_people_per_bin(tmp_path):
    people = {"v": [(0, 40.0, np.array([0.0, 0, 20, 40])),        # matches the first box -> found, bin 32-48
                    (0, 120.0, np.array([100.0, 100, 120, 220])),  # found, bin 96-160
                    (1, 120.0, np.array([100.0, 100, 120, 220])),  # frame 1 has no boxes -> missed
                    (0, 40.0, np.array([300.0, 300, 320, 340]))]}  # no overlap -> missed
    result = recall_by_bin(people, _toy_cache(tmp_path))
    assert result["32-48"] == {"persons": 2, "found": 1}
    assert result["96-160"] == {"persons": 2, "found": 1}
    assert result["all"] == {"persons": 4, "found": 2}
    assert recall(result["32-48"]) == 0.5 and np.isnan(recall(result["<32"]))
    assert pooled(result, ["32-48", "96-160"]) == 0.5
    low, high = wilson(result["32-48"])
    assert low < 0.5 < high
