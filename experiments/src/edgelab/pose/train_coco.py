"""EXP-020: train one arm of the 2 x 2 (M00, ML0, M0P, MLP) with the identical fixed schedule; resumable and idempotent.

    uv run --extra pose python -m edgelab.pose.train_coco --arm M00 --epochs N [--run-root ~/thesis-data/exp020/train]

    arm   labels      head     start weights
    M00   standard    stock    yolov8n-pose.pt
    ML0   partial     stock    yolov8n-pose.pt
    M0P   standard    P2       yolov8n-pose-p2-init.pt (pose/p2model.py: every stock tensor copied)
    MLP   partial     P2       yolov8n-pose-p2-init.pt

The keypoint-loss patch (pose/loss_patch.py) is installed in every arm; it changes nothing for standard labels.
Schedule (plan Amendment 1): same epochs / batch / imgsz 640 / seed 0 / Ultralytics defaults otherwise, no early stopping.
The model evaluated afterwards is the LAST epoch (best.pt would be chosen on a validation set whose labels differ by arm).
Re-running continues an interrupted run (Ultralytics resume from last.pt) and skips a finished one (<run>/train.done).
Ultralytics is AGPL-3.0: research use only (DECISIONS.md D11).
"""

import argparse
import json
import shutil
from pathlib import Path

from edgelab.pose.detcache import WEIGHTS
from edgelab.pose.finetune import read_results_csv, sha256_file
from edgelab.pose.stgnf_eval import EXPERIMENTS_ROOT

DATA_ROOT = Path.home() / "thesis-data/exp020"
ARTIFACT_DIR = EXPERIMENTS_ROOT / "artifacts/EXP-020"
ARMS = {"M00": ("standard", False), "ML0": ("partial", False), "M0P": ("standard", True), "MLP": ("partial", True)}
P2_INIT = DATA_ROOT / "yolov8n-pose-p2-init.pt"
# everything not listed is the Ultralytics default (lrf 0.01, momentum 0.937, weight decay 5e-4, hsv/flip/mosaic/scale
# augmentation). optimizer is pinned: Ultralytics' "auto" picks AdamW lr 0.002 below 10,000 iterations (which wrecks a
# pretrained model in a short run, seen in a 43-iteration smoke test) and SGD lr 0.01 above it (as EXP-013 did).
FIXED_ARGS = {"imgsz": 640, "batch": 16, "workers": 4, "device": "mps", "seed": 0, "patience": 100, "plots": False,
              "exist_ok": True, "optimizer": "SGD", "lr0": 0.01, "warmup_epochs": 1.0, "close_mosaic": 1}


def start_weights(arm: str) -> Path:
    return P2_INIT if ARMS[arm][1] else WEIGHTS


def dataset_yaml(arm: str, data_root: Path = DATA_ROOT) -> Path:
    return data_root / "data" / ARMS[arm][0] / "dataset.yaml"


def summarize(run: Path, arm: str, epochs: int) -> dict:
    rows = read_results_csv(run / "results.csv")
    last = rows[-1]
    return {"arm": arm, "epochs_planned": epochs, "epochs_run": len(rows), "train_seconds_csv": last.get("time"),
            "final_val": {k: v for k, v in last.items() if k.startswith("metrics/")},
            "final_train_loss": {k: v for k, v in last.items() if k.startswith("train/")},
            "per_epoch_val_mAP50-95(B)": [r["metrics/mAP50-95(B)"] for r in rows],
            "per_epoch_val_mAP50-95(P)": [r["metrics/mAP50-95(P)"] for r in rows]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True, choices=sorted(ARMS))
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--run-root", type=Path, default=DATA_ROOT / "train")
    parser.add_argument("--fraction", type=float, default=1.0, help="speed tests only; the real runs use 1.0")
    parser.add_argument("--data", type=Path, default=None, help="dataset yaml override (smoke tests only)")
    parser.add_argument("--name", default=None, help="default: the arm")
    parser.add_argument("--extra", default="{}", help="JSON of extra Ultralytics args (recorded in the summary; real runs use none)")
    args = parser.parse_args()
    name = args.name or args.arm
    run = args.run_root / name
    done = run / "train.done"
    if done.exists():
        print(f"{name}: already done")
        return

    from ultralytics import YOLO
    from ultralytics.utils import SETTINGS

    from edgelab.pose.loss_patch import install

    SETTINGS["mlflow"] = False
    install()
    last = run / "weights/last.pt"
    if last.exists():
        print(f"{name}: resuming from {last}", flush=True)
        YOLO(str(last)).train(resume=True)
    else:
        model = YOLO(str(start_weights(args.arm)))
        model.train(data=str(args.data or dataset_yaml(args.arm)), epochs=args.epochs, fraction=args.fraction, project=str(args.run_root),
                    name=name, **FIXED_ARGS, **json.loads(args.extra))
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    final = ARTIFACT_DIR / f"{name}.pt"
    shutil.copy(run / "weights/last.pt", final)
    summary = {**summarize(run, args.arm, args.epochs), "weights": str(final), "sha256": sha256_file(final),
               "start_weights": str(start_weights(args.arm)), "fixed_args": FIXED_ARGS, "fraction": args.fraction,
               "extra": json.loads(args.extra)}
    (ARTIFACT_DIR / f"{name}_train_summary.json").write_text(json.dumps(summary, indent=1))
    done.write_text(summary["sha256"] + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
