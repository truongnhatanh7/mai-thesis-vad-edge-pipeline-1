"""EXP-013: fine-tune yolov8n-pose on the pseudo-labelled ShanghaiTech training frames (Ultralytics trainer, MPS).

    uv run --extra pose python -m edgelab.pose.finetune --data ~/thesis-data/exp013-data/dataset.yaml \
        --run-dir ~/thesis-data/exp013-train [--epochs 30 --patience 5 --batch 16 --device mps --resume]

Afterwards the best checkpoint is copied to experiments/artifacts/EXP-013/ and its sha256 and the per-epoch metrics
(results.csv of the run) are written next to it. Ultralytics is AGPL-3.0: research use only (DECISIONS.md D11).
"""

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path

from edgelab.pose.detcache import WEIGHTS
from edgelab.pose.stgnf_eval import EXPERIMENTS_ROOT

ARTIFACT_DIR = EXPERIMENTS_ROOT / "artifacts/EXP-013"
FINETUNED = ARTIFACT_DIR / "yolov8n-pose-sht-ft.pt"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_results_csv(path: Path) -> list[dict]:
    """Ultralytics results.csv -> list of {column: float}; column names are stripped (they are padded in the file)."""
    with open(path) as f:
        return [{k.strip(): float(v) for k, v in row.items()} for row in csv.DictReader(f)]


def summarize_training(rows: list[dict], best_epoch: int) -> dict:
    """Epochs run, the validation metrics of the best epoch and of the last epoch.

    best_epoch: 1-based epoch (the `epoch` column of results.csv) that Ultralytics saved as best.pt.
    """
    best = next(r for r in rows if int(r["epoch"]) == best_epoch)
    return {"epochs_run": len(rows), "best_epoch": best_epoch,
            "best_val": {k: v for k, v in best.items() if k.startswith("metrics/")},
            "last_val": {k: v for k, v in rows[-1].items() if k.startswith("metrics/")}}


def pose_fitness(row: dict) -> float:
    """Ultralytics pose fitness: 0.1·mAP50 + 0.9·mAP50-95, summed over the box and pose metrics."""
    f = lambda k: float(row[k])  # noqa: E731
    return (0.1 * f("metrics/mAP50(B)") + 0.9 * f("metrics/mAP50-95(B)")
            + 0.1 * f("metrics/mAP50(P)") + 0.9 * f("metrics/mAP50-95(P)"))


def best_epoch_of(checkpoint: Path, rows: list[dict] | None = None) -> tuple[int, float]:
    """(1-based epoch, fitness). A finished run's best.pt is stripped by Ultralytics (best_fitness None, epoch -1),
    so then the best epoch is taken from results.csv with the same fitness formula."""
    import torch

    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if ckpt.get("best_fitness") is not None and int(ckpt.get("epoch", -1)) >= 0:
        return int(ckpt["epoch"]) + 1, float(ckpt["best_fitness"])
    best = max(rows, key=pose_fitness)
    return int(float(best["epoch"])), pose_fitness(best)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--weights", type=Path, default=WEIGHTS)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fraction", type=float, default=1.0, help="Ultralytics: share of the training images used")
    parser.add_argument("--name", default="train")
    parser.add_argument("--resume", action="store_true", help="continue an interrupted run from <run-dir>/<name>/weights/last.pt")
    parser.add_argument("--summarize-only", action="store_true", help="no training: copy weights and write the summary")
    args = parser.parse_args()

    from ultralytics import YOLO
    from ultralytics.utils import SETTINGS

    SETTINGS["mlflow"] = False  # in-process only: Ultralytics' own MLflow callback would write runs we do not want
    run = args.run_dir / args.name
    if args.summarize_only:
        pass
    elif args.resume:
        model = YOLO(str(run / "weights/last.pt"))
        model.train(resume=True)
    else:
        model = YOLO(str(args.weights))
        model.train(data=str(args.data), epochs=args.epochs, patience=args.patience, imgsz=args.imgsz, batch=args.batch,
                    workers=args.workers, device=args.device, seed=args.seed, fraction=args.fraction,
                    project=str(args.run_dir), name=args.name, exist_ok=True, plots=False)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy(run / "weights/best.pt", FINETUNED)
    shutil.copy(run / "results.csv", ARTIFACT_DIR / "train_results.csv")
    epoch, fitness = best_epoch_of(run / "weights/best.pt", read_results_csv(run / "results.csv"))
    summary = {**summarize_training(read_results_csv(run / "results.csv"), epoch), "best_fitness": fitness,
               "weights": str(FINETUNED),
               "sha256": sha256_file(FINETUNED), "init_weights": str(args.weights),
               "init_sha256": sha256_file(args.weights)}
    (ARTIFACT_DIR / "train_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
