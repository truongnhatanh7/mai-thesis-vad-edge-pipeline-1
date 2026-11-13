"""EXP-017 Y4 timing plumbing: plan, model paths, VM command, summary excluding unusable runs."""

import pandas as pd

from edgelab.pipeline import timing_exp017 as t17
from edgelab.pipeline.timing_matrix import MANIFEST_COLUMNS


def test_plan_runs_every_variant_once_per_rep_in_a_seeded_order():
    plan = t17.plan(3)
    assert len(plan) == 3 * len(t17.VARIANTS)
    for rep in range(3):
        assert sorted(v for _, v, r in plan if r == rep) == sorted(t17.VARIANTS)
    assert plan == t17.plan(3)
    assert [v for _, v, _ in plan[:6]] != [v for _, v, _ in plan[6:12]]


def test_controls_are_yolov8n_models_of_exp012_and_26n_models_are_the_new_ones(tmp_path):
    assert t17.model_path("yolo26n-ort-fp32-R800", tmp_path).name == "yolo26n-pose-800-480x800.onnx"
    assert t17.model_path("yolo26n-ncnn-fp16-R640", tmp_path).name == "yolo26n-pose-640-384x640-ncnn"
    assert t17.model_path("yolov8n-ncnn-fp16-R640", tmp_path).name == "yolov8n-pose-640-384x640-ncnn"
    assert t17.model_path("yolov8n-ort-fp32-R640", tmp_path).name == "yolov8n-pose-640-384x640.onnx"


def test_vm_command_is_the_exp004_cell_with_o0_b4():
    cmd = t17.vm_command("yolo26n-ncnn-fp16-R800", t17.RESULTS / "x.csv", "tag", max_frames=7)
    assert cmd[cmd.index("--yolo-onnx") + 1].endswith("yolo26n-pose-800-480x800-ncnn")
    assert cmd[cmd.index("--level") + 1] == "L3" and cmd[cmd.index("--c") + 1] == "0.25"
    assert cmd[cmd.index("--max-frames-per-video") + 1] == "7" and "B4" in cmd and "O0" in cmd
    assert "--ncnn-fp16" not in cmd


def test_summary_uses_only_ok_unexcluded_runs(tmp_path):
    def trace(name, det_ms):
        path = tmp_path / name
        pd.DataFrame({"det_pre_ms": 1.0, "det_infer_ms": det_ms, "det_post_ms": 1.0, "service_ms": det_ms + 10.0, "n_dets": 2.0},
                     index=range(4)).to_csv(path, index=False)
        return str(path)

    rows = [{"cell_id": "yolo26n-ort-fp32-R640", "rep": 0, "status": "ok", "excluded": False, "host_outside_vm_pct": 5.0,
             "trace_file": trace("a.csv", 10.0)},
            {"cell_id": "yolo26n-ort-fp32-R640", "rep": 1, "status": "ok", "excluded": True, "host_outside_vm_pct": 60.0,
             "trace_file": trace("b.csv", 99.0)},
            {"cell_id": "yolo26n-ort-fp32-R640", "rep": 2, "status": "failed", "excluded": False, "host_outside_vm_pct": 5.0,
             "trace_file": trace("c.csv", 77.0)}]
    pd.DataFrame(rows, columns=MANIFEST_COLUMNS).to_csv(tmp_path / "manifest.csv", index=False)
    table = t17.summary(tmp_path)
    assert table.runs.tolist() == [1] and table.det_ms.tolist() == [12.0]
    assert [p.name for p in t17.usable_files("yolo26n-ort-fp32-R640", tmp_path)] == ["a.csv"]
