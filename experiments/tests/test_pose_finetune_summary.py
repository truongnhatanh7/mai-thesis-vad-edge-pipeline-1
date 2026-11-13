import torch

from edgelab.pose.finetune import best_epoch_of, pose_fitness


def row(epoch, b50, b5095, p50, p5095):
    return {"epoch": str(epoch), "metrics/mAP50(B)": str(b50), "metrics/mAP50-95(B)": str(b5095),
            "metrics/mAP50(P)": str(p50), "metrics/mAP50-95(P)": str(p5095)}


def test_stripped_checkpoint_falls_back_to_results_csv(tmp_path):
    path = tmp_path / "best.pt"
    torch.save({"epoch": -1, "best_fitness": None}, path)  # what Ultralytics leaves after a finished run
    rows = [row(1, 0.9, 0.6, 0.7, 0.4), row(2, 0.95, 0.71, 0.81, 0.48), row(3, 0.95, 0.70, 0.80, 0.47)]
    epoch, fitness = best_epoch_of(path, rows)
    assert epoch == 2 and abs(fitness - pose_fitness(rows[1])) < 1e-12


def test_unstripped_checkpoint_is_read_directly(tmp_path):
    path = tmp_path / "best.pt"
    torch.save({"epoch": 4, "best_fitness": 1.23}, path)
    assert best_epoch_of(path, []) == (5, 1.23)
