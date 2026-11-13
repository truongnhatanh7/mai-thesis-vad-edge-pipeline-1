"""EXP-017: the detector cache can run Ultralytics' NMS-free end-to-end head (YOLO26) and says so in its meta."""

import json

import cv2
import numpy as np
import pytest

from edgelab.pose.stgnf_eval import EXPERIMENTS_ROOT

pytest.importorskip("ultralytics", reason="needs `uv sync --extra pose`")
WEIGHTS26 = EXPERIMENTS_ROOT / "artifacts/EXP-017/yolo26n-pose.pt"


@pytest.mark.skipif(not WEIGHTS26.exists(), reason="YOLO26 weights missing")
def test_nms_free_cache_uses_the_end_to_end_head_and_records_it(tmp_path):
    from ultralytics import YOLO

    from edgelab.pose.detcache import cache_video

    frames = tmp_path / "v"
    frames.mkdir()
    for i in range(2):
        cv2.imwrite(str(frames / f"{i:03d}.jpg"), np.random.default_rng(i).integers(0, 255, (96, 128, 3), np.uint8))
    model = YOLO(str(WEIGHTS26))
    plain = cache_video(model, frames, "L0", "E0", "cpu")
    free = cache_video(YOLO(str(WEIGHTS26)), frames, "L0", "E0", "cpu", nms_free=True, weights_name=WEIGHTS26.name)
    assert json.loads(str(free["meta"]))["nms_free"] is True and json.loads(str(free["meta"]))["weights"] == WEIGHTS26.name
    assert "nms_free" not in json.loads(str(plain["meta"]))
    assert free["keypoints"].shape[1:] == (17, 3) and (free["counts"] <= 300).all()
    assert model.predictor.model.end2end is False  # default path = one-to-many + NMS, unchanged for all other experiments
