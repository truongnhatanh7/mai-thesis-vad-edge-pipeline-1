"""EXP-016 timing helpers: person decoding of a YOLOv8 detection output, and the per-run summary."""

import numpy as np
import pandas as pd

from edgelab.pipeline.timing_exp016 import config_id, summarise_trace
from edgelab.pipeline.topdown_timing import decode_persons


def fake_output(anchors: list[tuple]) -> np.ndarray:
    """anchors: (cx, cy, w, h, {class: prob}) -> (1, 84, N) array like a YOLOv8 detection head."""
    out = np.zeros((1, 84, len(anchors)), np.float32)
    for i, (cx, cy, w, h, probs) in enumerate(anchors):
        out[0, :4, i] = [cx, cy, w, h]
        for cls, p in probs.items():
            out[0, 4 + cls, i] = p
    return out


def test_decode_keeps_persons_only_applies_nms_and_maps_back_to_image_pixels():
    output = fake_output([
        (100, 100, 40, 80, {0: 0.9}),            # person
        (102, 101, 40, 80, {0: 0.8}),            # same person again: removed by NMS
        (300, 100, 40, 80, {0: 0.4, 2: 0.7}),    # best class is car: not a person for classes=[0]
        (400, 100, 40, 80, {0: 0.05}),           # below conf
        (200, 100, 40, 80, {0: 0.6}),            # second person
    ])
    boxes, conf = decode_persons(output, conf=0.25, ratio=0.5, pad=(0, 10), shape=(480, 856))
    assert conf.tolist() == np.float32([0.9, 0.6]).tolist()
    # letterbox: image = (letterbox - pad) / ratio ; first box x 80..120, y 60..140 -> x 160..240, y 100..260
    assert np.allclose(boxes[0], [160, 100, 240, 260], atol=1e-3)
    assert np.allclose(boxes[1], [360, 100, 440, 260], atol=1e-3)


def test_decode_with_nothing_above_conf_returns_empty():
    boxes, conf = decode_persons(fake_output([(10, 10, 5, 5, {0: 0.1})]), 0.25, 1.0, (0, 0), (100, 100))
    assert boxes.shape == (0, 4) and conf.shape == (0,)


def test_summarise_trace_reports_mean_p50_p99_and_budget_share():
    trace = pd.DataFrame({"service_ms": [10.0, 20.0, 30.0, 50.0], "det_pre_ms": 1.0, "det_infer_ms": 4.0, "det_post_ms": 1.0,
                          "pose_ms": [2.0, 4.0, 6.0, 8.0], "n_boxes": [1, 2, 3, 4]})
    s = summarise_trace(trace)
    assert s["frames"] == 4 and s["mean_ms"] == 27.5 and s["p50_ms"] == 25.0
    assert s["mean_det_ms"] == 6.0 and s["mean_pose_ms"] == 5.0 and s["mean_boxes"] == 2.5
    assert s["share_frames_over_budget"] == 0.25  # only the 50 ms frame is over 41.7 ms


def test_config_id_names_the_cell():
    assert config_id(960, 3, "rtmpose-t") == "L0-n960-K3-rtmpose-t-c0.25-O0-B4"


def test_run_on_fresh_manifest_does_not_crash(tmp_path, monkeypatch):
    """Regression: the header of a new manifest was unflushed when done_configs() re-read it (T3 crashed at 16:19)."""
    from pathlib import Path
    from types import SimpleNamespace

    import edgelab.pipeline.timing_exp016 as t

    monkeypatch.setattr(t, "RESULTS", tmp_path)
    monkeypatch.setattr(t, "MANIFEST", tmp_path / "manifest.csv")
    monkeypatch.setattr(t, "CONFIGS", [(640, 1)])
    monkeypatch.setattr(t, "find_vm_pid", lambda: 1)
    monkeypatch.setattr(t, "wait_for_quiet_host", lambda pid, limit: 1.0)
    monkeypatch.setattr(t, "LoadSampler", lambda pid: SimpleNamespace(start=lambda: SimpleNamespace(stop=lambda: [])))
    monkeypatch.setattr(t, "summarise", lambda samples: {"host_outside_vm_pct": 1.0})

    def fake_run(cmd, **kw):
        Path(cmd[cmd.index("--out") + 1]).write_text("x\n")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(t.subprocess, "run", fake_run)
    t.run("rtmpose-s", 2)
    assert t.done_configs() == {t.config_id(640, 1, "rtmpose-s")}
