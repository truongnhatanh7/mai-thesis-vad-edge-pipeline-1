"""EXP-020 VM timing driver: plan and VM command (no VM is started)."""

from pathlib import Path

from edgelab.pipeline.timing_exp020 import VARIANTS, plan, vm_command


def test_plan_runs_every_variant_once_per_rep_in_seeded_order():
    ids = sorted(VARIANTS)
    p = plan(3, ids)
    assert len(p) == 3 * len(ids)
    for rep in range(3):
        assert sorted(v for _, v, r in p if r == rep) == ids
    assert plan(3, ids) == p  # deterministic
    assert [v for _, v, r in p if r == 0] != [v for _, v, r in p if r == 1]  # orders differ across reps


def test_vm_command_selects_the_ncnn_folder_and_cell_resolution():
    cmd = vm_command("M0P", Path("/tmp/t.csv"), "M0P-rep0", max_frames=40)
    assert cmd[:4] == ["limactl", "shell", "edge-os", "--"]
    assert cmd[cmd.index("--yolo-onnx") + 1].endswith("exp020/M0P-640-384x640-ncnn")
    assert cmd[cmd.index("--imgsz") + 1] == "640" and cmd[cmd.index("--tag") + 1] == "M0P-rep0"
    assert cmd[cmd.index("--max-frames-per-video") + 1] == "40"
    assert vm_command("stock-R800", Path("/tmp/t.csv"), "x")[-1].endswith("yolov8n-pose-800-480x800-ncnn")
    assert "--ncnn-fp16" not in cmd  # FP16 is the default
