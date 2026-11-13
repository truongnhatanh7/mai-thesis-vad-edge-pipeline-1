"""EXP-020 training wrapper: arm table, start weights, dataset yaml, summary of results.csv."""

from pathlib import Path

from edgelab.pose.detcache import WEIGHTS
from edgelab.pose.train_coco import ARMS, FIXED_ARGS, P2_INIT, dataset_yaml, start_weights, summarize

CSV = """                  epoch,                  time,         train/box_loss,         train/pose_loss,  metrics/precision(B),  metrics/mAP50(B),  metrics/mAP50-95(B),  metrics/mAP50(P),  metrics/mAP50-95(P)
1,100.5,1.0,2.0,0.5,0.6,0.30,0.5,0.20
2,201.0,0.9,1.8,0.6,0.7,0.35,0.6,0.25
"""


def test_arms_are_the_2x2():
    assert ARMS == {"M00": ("standard", False), "ML0": ("partial", False), "M0P": ("standard", True), "MLP": ("partial", True)}
    assert start_weights("M00") == start_weights("ML0") == WEIGHTS
    assert start_weights("M0P") == start_weights("MLP") == P2_INIT


def test_dataset_yaml_follows_the_label_policy(tmp_path):
    assert dataset_yaml("ML0", tmp_path) == tmp_path / "data/partial/dataset.yaml"
    assert dataset_yaml("M0P", tmp_path) == tmp_path / "data/standard/dataset.yaml"


def test_fixed_schedule_is_identical_for_every_arm_and_has_seed_0_imgsz_640():
    assert FIXED_ARGS["seed"] == 0 and FIXED_ARGS["imgsz"] == 640 and FIXED_ARGS["device"] == "mps"


def test_summarize_reads_last_epoch(tmp_path: Path):
    (tmp_path / "results.csv").write_text(CSV)
    s = summarize(tmp_path, "M00", epochs=2)
    assert s["epochs_run"] == 2 and s["train_seconds_csv"] == 201.0
    assert s["final_val"]["metrics/mAP50-95(P)"] == 0.25 and s["final_train_loss"]["train/pose_loss"] == 1.8
    assert s["per_epoch_val_mAP50-95(B)"] == [0.30, 0.35]
