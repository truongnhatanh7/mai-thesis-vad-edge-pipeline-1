"""Train/evaluate the Salman CNN-GRU on cached clips with the paper's hyperparameters (plan.md §5.3).

uv run python -m edgelab.models.train_clips --dataset hockey-fight --protocol video --seed 0
"""

import argparse
import csv
import math
import random
import time
from pathlib import Path

import mlflow
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from edgelab.data.splits import leaky_clip_split, save_split, shared_videos, video_level_split
from edgelab.data.video_clips import ClipDataset, make_clip_items, read_index, to_model_input
from edgelab.metrics.scores import accuracy, binary_f1, bootstrap_ci_over_videos
from edgelab.models.salman_cnn_gru import SalmanCnnGru, macs_per_frame, parameter_count
from edgelab.tracking.mlflow_run import EXPERIMENTS_ROOT, tracked_run

PAPER_HYPERPARAMS = {"lr": 1e-4, "batch_size": 16, "max_epochs": 50, "patience": 10, "dropout": 0.4}
PAPER_AUGMENT = {"flip_p": 0.5, "rotation_deg": 15.0, "brightness": 0.2, "contrast": 0.2, "noise_sigma": 0.01}
EXP_DIR = EXPERIMENTS_ROOT / "EXP-002-anchor-reconstruction"
ARTIFACTS = EXPERIMENTS_ROOT / "artifacts" / "EXP-002"
CACHE_ROOT = EXPERIMENTS_ROOT / "data" / "cache"
IN_MEMORY_LIMIT_BYTES = 6 * 1024**3  # larger caches (RLVS 16 GB) are memory-mapped instead


def augment(x: torch.Tensor, cfg: dict = PAPER_AUGMENT) -> torch.Tensor:
    """x [B, T, 3, H, W] in [0,1]. One random draw per clip, identical for all its frames (Table 2)."""
    b, t, c, h, w = x.shape
    dev = x.device

    def per_clip(values: torch.Tensor) -> torch.Tensor:
        return values.view(b, 1, 1, 1, 1)

    hflip = torch.rand(b, device=dev) < cfg["flip_p"]
    vflip = torch.rand(b, device=dev) < cfg["flip_p"]
    x = torch.where(per_clip(hflip), x.flip(-1), x)
    x = torch.where(per_clip(vflip), x.flip(-2), x)

    angle = (torch.rand(b, device=dev) * 2 - 1) * math.radians(cfg["rotation_deg"])
    cos, sin, zero = angle.cos(), angle.sin(), torch.zeros_like(angle)
    theta = torch.stack([torch.stack([cos, -sin, zero], -1), torch.stack([sin, cos, zero], -1)], 1)
    grid = F.affine_grid(theta.repeat_interleave(t, 0), [b * t, c, h, w], align_corners=False)
    x = F.grid_sample(x.flatten(0, 1), grid, align_corners=False).view(b, t, c, h, w)

    brightness = 1 + (torch.rand(b, device=dev) * 2 - 1) * cfg["brightness"]
    contrast = 1 + (torch.rand(b, device=dev) * 2 - 1) * cfg["contrast"]
    mean = x.mean(dim=(1, 2, 3, 4), keepdim=True)
    x = (x - mean) * per_clip(contrast) + mean
    x = x * per_clip(brightness)
    x = x + torch.randn_like(x) * cfg["noise_sigma"]
    return x.clamp_(0, 1)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def pick_device() -> torch.device:
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


@torch.no_grad()
def predict(model, loader, device) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Returns (P(class 1) per clip, labels, item indices, mean CE loss)."""
    model.eval()
    probs, labels, indices, loss_sum = [], [], [], 0.0
    for clip, label, idx in loader:
        logits = model(to_model_input(clip.to(device)))
        label = label.to(device)
        loss_sum += F.cross_entropy(logits, label, reduction="sum").item()
        probs.append(torch.softmax(logits, -1)[:, 1].cpu())
        labels.append(label.cpu())
        indices.append(idx)
    probs, labels = torch.cat(probs).numpy(), torch.cat(labels).numpy()
    return probs, labels, torch.cat(indices).numpy(), loss_sum / len(labels)


def test_metrics(probs, labels, indices, items) -> dict:
    from sklearn.metrics import roc_auc_score
    preds = (probs >= 0.5).astype(int)
    by_video: dict[str, list[int]] = {}
    for pos, i in enumerate(indices):
        by_video.setdefault(items[i].video_id, []).append(pos)
    videos = sorted(by_video)
    clip_labels = [labels[by_video[v]] for v in videos]
    clip_preds = [preds[by_video[v]] for v in videos]
    video_labels = [np.array([labels[by_video[v][0]]]) for v in videos]
    video_preds = [np.array([int(probs[by_video[v]].mean() >= 0.5)]) for v in videos]

    def acc_metric(l, s):
        return accuracy(np.concatenate(l), np.concatenate(s))

    clip_ci = bootstrap_ci_over_videos(clip_labels, clip_preds, metric=acc_metric, seed=0)
    video_ci = bootstrap_ci_over_videos(video_labels, video_preds, metric=acc_metric, seed=0)
    return {
        "test_clips": len(labels), "test_videos": len(videos),
        "test_clip_acc": accuracy(labels, preds), "test_clip_acc_ci_lo": clip_ci[0], "test_clip_acc_ci_hi": clip_ci[1],
        "test_clip_f1": binary_f1(labels, preds), "test_clip_auc": float(roc_auc_score(labels, probs)),
        "test_video_acc": acc_metric(video_labels, video_preds),
        "test_video_acc_ci_lo": video_ci[0], "test_video_acc_ci_hi": video_ci[1],
    }


def run(dataset: str, protocol: str, seed: int, clip_len: int = 8, max_epochs: int | None = None,
        device: torch.device | None = None) -> dict:
    device = device or pick_device()
    hp = {**PAPER_HYPERPARAMS, **({"max_epochs": max_epochs} if max_epochs else {})}
    cache_dir = CACHE_ROOT / dataset
    items = make_clip_items(read_index(cache_dir), clip_len=clip_len, stride=2, max_clips=8)
    split = (video_level_split if protocol == "video" else leaky_clip_split)(items, seed)
    split_path = EXP_DIR / "splits" / f"{dataset}_{protocol}_seed{seed}.json"
    save_split(split, items, split_path, {"dataset": dataset, "protocol": protocol, "seed": seed, "clip_len": clip_len})

    seed_everything(seed)
    model = SalmanCnnGru(num_classes=2, dropout=hp["dropout"]).to(device)
    cache_bytes = sum(f.stat().st_size for f in cache_dir.glob("*.npy"))
    data = ClipDataset(cache_dir, items, clip_len=clip_len, in_memory=cache_bytes < IN_MEMORY_LIMIT_BYTES)
    gen = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(Subset(data, split["train"]), batch_size=hp["batch_size"], shuffle=True, generator=gen)
    val_loader = DataLoader(Subset(data, split["val"]), batch_size=64)
    test_loader = DataLoader(Subset(data, split["test"]), batch_size=64)
    optimizer = torch.optim.Adam(model.parameters(), lr=hp["lr"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=PAPER_HYPERPARAMS["max_epochs"])

    run_name = f"{dataset}-{protocol}-seed{seed}"
    out_dir = ARTIFACTS / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    config = {"dataset": dataset, "protocol": protocol, "seed": seed, "clip_len": clip_len, "frame_skip": 3,
              "clip_stride": 2, "max_clips_per_video": 8, "device": str(device), "hyperparams": hp,
              "augment": PAPER_AUGMENT, "params": parameter_count(model), "macs_per_frame": macs_per_frame(model.cpu()),
              "split_file": str(split_path.relative_to(EXPERIMENTS_ROOT)),
              "n_clips": {p: len(v) for p, v in split.items()}, "shared_videos": shared_videos(items, split)}
    model.to(device)
    with tracked_run("EXP-002", run_name, config, input_files={"split": split_path}) as mlrun:
        best_val_acc, best_val_loss, epochs_since_best_loss, epoch_seconds = -1.0, math.inf, 0, []
        for epoch in range(hp["max_epochs"]):
            model.train()
            start, train_loss, seen = time.perf_counter(), 0.0, 0
            for clip, label, _ in train_loader:
                x = augment(to_model_input(clip.to(device)))
                label = label.to(device)
                loss = F.cross_entropy(model(x), label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * len(label)
                seen += len(label)
            scheduler.step()
            val_probs, val_labels, _, val_loss = predict(model, val_loader, device)
            seconds = time.perf_counter() - start
            epoch_seconds.append(seconds)
            val_acc = accuracy(val_labels, (val_probs >= 0.5).astype(int))
            mlflow.log_metrics({"train_loss": train_loss / seen, "val_loss": val_loss, "val_acc": val_acc,
                                "epoch_seconds": seconds, "lr": scheduler.get_last_lr()[0]}, step=epoch)
            print(f"[{run_name}] epoch {epoch} train_loss {train_loss / seen:.4f} val_loss {val_loss:.4f} "
                  f"val_acc {val_acc:.4f} {seconds:.1f}s", flush=True)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), out_dir / "best.pt")
                mlflow.log_metric("best_epoch", epoch)
            if val_loss < best_val_loss:
                best_val_loss, epochs_since_best_loss = val_loss, 0
            else:
                epochs_since_best_loss += 1
                if epochs_since_best_loss >= hp["patience"]:
                    break

        model.load_state_dict(torch.load(out_dir / "best.pt", map_location=device))
        probs, labels, indices, _ = predict(model, test_loader, device)
        metrics = {**test_metrics(probs, labels, indices, items), "best_val_acc": best_val_acc,
                   "epochs_run": len(epoch_seconds), "mean_epoch_seconds": float(np.mean(epoch_seconds))}
        mlflow.log_metrics(metrics)
        with open(out_dir / "test_predictions.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["video_id", "start", "label", "prob_violent"])
            for p, lab, i in zip(probs, labels, indices):
                writer.writerow([items[i].video_id, items[i].start, int(lab), f"{p:.6f}"])
        mlflow.log_artifact(str(out_dir / "test_predictions.csv"))
        mlflow.log_artifact(str(out_dir / "best.pt"))
        metrics["run_id"] = mlrun.info.run_id
    print(metrics, flush=True)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="hockey-fight")
    parser.add_argument("--protocol", choices=["video", "leaky"], default="video")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-epochs", type=int, default=None, help="smoke runs only; default = paper's 50")
    args = parser.parse_args()
    run(args.dataset, args.protocol, args.seed, max_epochs=args.max_epochs)


if __name__ == "__main__":
    main()
