"""EXP-022: track classification of misses and agreement with the EXP-011 recall counter on a toy cache."""

import numpy as np

from edgelab.analysis.exp011 import recall_by_bin
from edgelab.analysis.exp022 import decision, match_flags, miss_classes, recall_from_flags


def _toy_cache(tmp_path):
    boxes = np.array([[0.0, 0, 20, 40], [100, 100, 120, 220]], dtype=np.float32)  # frame 0: 40 px and 120 px people; frame 1: nothing
    np.savez_compressed(tmp_path / "v.npz", boxes=boxes, conf=np.array([0.9, 0.9], np.float32), cls=np.zeros(2, np.float32),
                        keypoints=np.zeros((2, 17, 3), np.float32), counts=np.array([2, 0]), frame_ids=np.array([0, 1]))
    return tmp_path


def _tracks():
    small, big, nowhere = np.array([0.0, 0, 20, 40]), np.array([100.0, 100, 120, 220]), np.array([300.0, 300, 320, 340])
    return {"v": {"a": [(0, 120.0, big), (1, 120.0, big)],              # frame 0 matched, frame 1 missed -> intermittent (1 miss)
                  "b": [(0, 40.0, nowhere), (1, 40.0, nowhere)],        # never seen (2 misses)
                  "c": [(0, 40.0, small)]}}                              # fully matched


def test_flags_recall_and_miss_classes(tmp_path):
    tracks = _tracks()
    flags = match_flags(tracks, _toy_cache(tmp_path))
    assert [f.tolist() for f in (flags["v"]["a"], flags["v"]["b"], flags["v"]["c"])] == [[True, False], [False, False], [True]]
    result = recall_from_flags(tracks, flags)
    assert result["all"] == {"persons": 5, "found": 2}
    flat = {"v": [(f, h, b) for entries in tracks["v"].values() for f, h, b in entries]}
    assert recall_by_bin(flat, tmp_path) == result  # same counts as the EXP-011 function
    classes = miss_classes(tracks, flags)
    assert (classes["tracks"], classes["tracks_never_seen"], classes["tracks_intermittent"], classes["tracks_fully_matched"]) == (3, 1, 1, 1)
    assert (classes["missed_person_frames"], classes["missed_in_never_seen"], classes["missed_in_intermittent"]) == (3, 2, 1)
    assert abs(classes["intermittent_share"] - 1 / 3) < 1e-12 and decision(classes)["propose_tsm"]
    assert miss_classes(tracks, flags, min_length=2)["tracks"] == 2
