"""EXP-012 plumbing: timing plan/commands/summary, the detector-cache path for ORT/NCNN detectors, the analysis helpers."""

import cv2
import numpy as np
import pandas as pd
import pytest

from edgelab.analysis.exp012 import best_fast_800, q1_ratio
from edgelab.pipeline import timing_exp012 as t12
from edgelab.pose.detcache import MAX_DET, cache_video, detect_fast, load_detections, save_atomic


def test_plan_runs_every_variant_once_per_rep_in_a_seeded_order():
    plan = t12.plan(3)
    assert len(plan) == 3 * len(t12.KINDS) * len(t12.RESOLUTIONS)
    for rep in range(3):
        in_rep = [(k, r) for _, k, r, p in plan if p == rep]
        assert sorted(in_rep) == sorted((k, r) for k in t12.KINDS for r in t12.RESOLUTIONS)
    assert plan == t12.plan(3)
    assert [x[1:3] for x in plan[:8]] != [x[1:3] for x in plan[8:16]]  # a different order in the next repetition


def test_model_paths(tmp_path):
    assert t12.model_path("ort-fp32", 800, tmp_path).name == "yolov8n-pose-800-480x800.onnx"
    assert t12.model_path("ort-int8", 640, tmp_path).name == "yolov8n-pose-640-384x640-int8.onnx"
    assert t12.model_path("ncnn-fp16", 640, tmp_path).name == "yolov8n-pose-640-384x640-ncnn"
    assert t12.model_path("ncnn-fp32", 640, tmp_path) == t12.model_path("ncnn-fp16", 640, tmp_path)


def test_vm_command_passes_the_model_and_the_fp16_switch():
    cmd = t12.vm_command("ncnn-fp32", 800, t12.RESULTS / "x.csv", "tag", max_frames=7)
    assert cmd[cmd.index("--yolo-onnx") + 1].endswith("yolov8n-pose-800-480x800-ncnn")
    assert cmd[cmd.index("--ncnn-fp16") + 1] == "off" and cmd[cmd.index("--max-frames-per-video") + 1] == "7"
    assert cmd[cmd.index("--tag") + 1] == "tag" and "B4" in cmd and "O0" in cmd
    assert "--ncnn-fp16" not in t12.vm_command("ncnn-fp16", 800, t12.RESULTS / "x.csv", "tag")
    assert "--ncnn-fp16" not in t12.vm_command("ort-int8", 800, t12.RESULTS / "x.csv", "tag")


def test_summary_pools_usable_runs_and_skips_excluded_ones(tmp_path):
    def trace(path, ms):
        pd.DataFrame({"det_pre_ms": ms, "det_infer_ms": ms, "det_post_ms": ms, "service_ms": ms}).to_csv(path, index=False)

    trace(tmp_path / "a.csv", [1.0, 1.0])  # detector = 3 ms per frame
    trace(tmp_path / "b.csv", [3.0, 3.0])  # 9 ms
    trace(tmp_path / "c.csv", [100.0])  # excluded
    rows = [("ort-fp32-R640", "a.csv", False, 5.0), ("ort-fp32-R640", "b.csv", False, 7.0), ("ort-fp32-R640", "c.csv", True, 50.0)]
    pd.DataFrame({"cell_id": [r[0] for r in rows], "status": "ok", "excluded": [r[2] for r in rows],
                  "trace_file": [str(tmp_path / r[1]) for r in rows], "host_outside_vm_pct": [r[3] for r in rows]}
                 ).to_csv(tmp_path / "manifest.csv", index=False)
    out = t12.summary(tmp_path).iloc[0]
    assert out.runs == 2 and out.frames == 4 and out.det_ms == 6.0 and out.host_outside_vm_pct == 6.0


def _summary(**det_ms):
    return pd.DataFrame({"variant": list(det_ms), "det_ms": list(det_ms.values())})


def test_best_fast_800_is_the_cheapest_non_reference_variant_at_800():
    table = _summary(**{"ort-fp32-R640": 34.0, "ort-fp32-R800": 52.0, "ort-int8-R800": 19.0, "ncnn-fp16-R800": 18.0,
                        "ncnn-fp32-R800": 27.0, "ncnn-fp16-R640": 12.0})
    assert best_fast_800(table) == "ncnn-fp16-R800"


def test_q1_ratio_against_fp32_640_with_the_pass_rule():
    table = _summary(**{"ort-fp32-R640": 34.0, "ort-int8-R800": 36.0})
    assert q1_ratio(table, "ort-int8-R800") == {"ratio": pytest.approx(36 / 34), "pass": True}
    assert q1_ratio(_summary(**{"ort-fp32-R640": 30.0, "ort-int8-R800": 40.0}), "ort-int8-R800")["pass"] is False


class FakeDetector:
    """Same call signature as OrtPose / NcnnPose: always sees two people, the second less confident."""
    shape = (32, 48)

    def __init__(self, n=2):
        self.n = n

    def __call__(self, img, conf):
        n = self.n
        boxes = np.tile(np.array([[1, 2, 11, 22]], np.float32), (n, 1))
        scores = np.linspace(0.9, 0.1, n).astype(np.float32)
        keypoints = np.full((n, 17, 3), 0.5, np.float32)
        return boxes, scores, keypoints, {"pre_ms": 1.0, "infer_ms": 2.0, "post_ms": 3.0}


def test_detect_fast_returns_the_ultralytics_tuple_and_caps_detections():
    boxes, conf, cls, keypoints, speed = detect_fast(FakeDetector(), np.zeros((32, 48, 3), np.uint8))
    assert boxes.shape == (2, 4) and conf.tolist() == pytest.approx([0.9, 0.1]) and cls.tolist() == [0.0, 0.0]
    assert keypoints.shape == (2, 17, 3) and speed == {"preprocess": 1.0, "inference": 2.0, "postprocess": 3.0}
    assert len(detect_fast(FakeDetector(MAX_DET + 20), np.zeros((32, 48, 3), np.uint8))[1]) == MAX_DET


def test_cache_video_with_a_detector_writes_the_normal_cache_format(tmp_path):
    video = tmp_path / "01_0001"
    video.mkdir()
    for i in range(3):
        cv2.imwrite(str(video / f"{i:03d}.jpg"), np.full((32, 48, 3), 100, np.uint8))
    arrays = cache_video(None, video, "L0", "E0", "ort-int8", imgsz=800, detector=FakeDetector())
    assert arrays["counts"].tolist() == [2, 2, 2] and arrays["frame_ids"].tolist() == [0, 1, 2]
    assert arrays["boxes"].shape == (6, 4) and arrays["keypoints"].shape == (6, 17, 3)
    assert arrays["yolo_infer_ms"].tolist() == [2.0, 2.0, 2.0]
    assert '"device": "ort-int8"' in str(arrays["meta"]) and '"imgsz": 800' in str(arrays["meta"])
    save_atomic(tmp_path / "v.npz", arrays)
    frames, _ = load_detections(tmp_path / "v.npz")  # the replay step reads it like any other cache
    assert len(frames) == 3 and frames[0]["conf"].tolist() == pytest.approx([0.9, 0.1])
