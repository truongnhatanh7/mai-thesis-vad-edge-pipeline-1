"""Train the EXP-018 C2 pose predictor (GRU) on released training poses; early stopping on held-out training videos.

    uv run --extra pose python -m edgelab.pose.predictor_train --dataset ShanghaiTech --out-dir ~/thesis-data/exp018/predictor_sht

Every window of the training bank (16 frames of context + the 24-frame window) gets a random keep mask (augment.random_keep_masks);
the GRU rolls through it, reading detections where there are any and its own guess elsewhere, and is trained on the poses it fills
(the same frames the filler emits at test time): error = mean keypoint distance / height of the true pose (a fraction of body height).
Held-out videos = every 10th pose file (sorted); their masks are fixed, so the validation error is comparable between epochs and
with constant velocity (same masks, same frames), reported in history.json. Stop after `patience` epochs without improvement.

Files: predictor.pt + DONE (finished), last.pt (resume), history.json, train.log.
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from edgelab.pose.augment import random_keep_masks
from edgelab.pose.predictor import FILL_G, PosePredictor, fill_const_velocity, rollout
from edgelab.pose.segments import SegmentBank
from edgelab.pose.stgnf_train import TRAIN_POSES, load_bank, log_line

HELD_OUT_EVERY = 10
VAL_WINDOWS = 20000


def grid_poses(bank: SegmentBank, rows: np.ndarray):
    grid = torch.from_numpy(bank.grid[rows].astype(np.int64))
    return torch.from_numpy(bank.xy[grid.clamp(min=0).numpy()]), grid >= 0


def filled_error(pose: torch.Tensor, truth: torch.Tensor, filled: torch.Tensor, present: torch.Tensor) -> tuple[torch.Tensor, int]:
    """(sum of normalised keypoint errors over filled frames that have a true pose, number of such frames)."""
    height = (truth[..., 1].amax(-1) - truth[..., 1].amin(-1)).clamp(min=1.0)
    error = (pose - truth).norm(dim=-1).mean(dim=-1) / height  # (B, L)
    use = filled & present
    return (error * use).sum(), int(use.sum())


def validate(predictor: PosePredictor, bank: SegmentBank, rows: np.ndarray, masks: np.ndarray, batch_size: int = 512) -> dict:
    """Mean normalised error of the GRU and of constant velocity on the same filled frames of the held-out windows."""
    totals = {"gru": 0.0, "cv": 0.0}
    count = 0
    with torch.no_grad():
        for begin in range(0, len(rows), batch_size):
            xy, present = grid_poses(bank, rows[begin:begin + batch_size])
            detected = present & torch.from_numpy(masks[begin:begin + batch_size])
            for name, (pose, filled) in {"gru": rollout(predictor, xy, detected), "cv": fill_const_velocity(xy, detected)}.items():
                total, n = filled_error(pose, xy, filled, present)
                totals[name] += float(total)
            count += n
    return {"gru": totals["gru"] / count, "cv": totals["cv"] / count, "filled_frames": count}


def train(dataset: str, out_dir: Path, seed: int = 0, max_epochs: int = 30, patience: int = 3, batch_size: int = 256,
          lr: float = 2e-3, threads: int = 2, max_batches: int | None = None) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / "DONE").exists():
        return out_dir / "predictor.pt"
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    bank = load_bank(dataset)
    held_out = set(range(0, len(bank.files), HELD_OUT_EVERY))
    val_rows_all = bank.subset(held_out)
    train_rows = np.flatnonzero(~np.isin(bank.file_id, list(held_out)))
    val_rows = np.random.default_rng(123).permutation(val_rows_all)[:VAL_WINDOWS]
    val_masks = random_keep_masks(len(val_rows), bank.grid.shape[1], np.random.default_rng(456))
    predictor = PosePredictor()
    optimizer = torch.optim.Adam(predictor.parameters(), lr=lr)
    history, best, best_state, since_best, start_epoch = [], float("inf"), None, 0, 0
    if (out_dir / "last.pt").exists():
        saved = torch.load(out_dir / "last.pt", map_location="cpu", weights_only=False)
        predictor.load_state_dict(saved["state_dict"])
        optimizer.load_state_dict(saved["optimizer"])
        history, best, best_state, since_best, start_epoch = saved["history"], saved["best"], saved["best_state"], saved["since_best"], saved["epoch"]
        log_line(out_dir, f"resumed after epoch {start_epoch}")
    log_line(out_dir, f"dataset={dataset} train windows={len(train_rows)} held-out windows={len(val_rows_all)} (val {len(val_rows)}) "
                      f"params={sum(p.numel() for p in predictor.parameters())}")
    if not history:
        baseline = validate(predictor, bank, val_rows, val_masks)
        log_line(out_dir, f"untrained (holds the pose still): val error {baseline['gru']:.4f}, constant velocity {baseline['cv']:.4f}")
    t_start = time.perf_counter()
    for epoch in range(start_epoch, max_epochs):
        if since_best >= patience:
            break
        rng = np.random.default_rng([seed, epoch])
        order = rng.permutation(train_rows)
        predictor.train()
        losses = []
        for b, begin in enumerate(range(0, len(order), batch_size)):
            if max_batches is not None and b >= max_batches:
                break
            rows = order[begin:begin + batch_size]
            xy, present = grid_poses(bank, rows)
            detected = present & torch.from_numpy(random_keep_masks(len(rows), xy.shape[1], rng))
            pose, filled = rollout(predictor, xy, detected)
            total, n = filled_error(pose, xy, filled, present)
            if n == 0:
                continue
            loss = total / n
            loss.backward()
            torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            losses.append(loss.item())
        predictor.eval()
        val = validate(predictor, bank, val_rows, val_masks)
        history.append({"epoch": epoch + 1, "train_loss": float(np.mean(losses)), **val})
        improved = val["gru"] < best
        if improved:
            best, best_state, since_best = val["gru"], {k: v.clone() for k, v in predictor.state_dict().items()}, 0
        else:
            since_best += 1
        torch.save({"state_dict": predictor.state_dict(), "optimizer": optimizer.state_dict(), "history": history, "best": best,
                    "best_state": best_state, "since_best": since_best, "epoch": epoch + 1}, out_dir / "last.pt.tmp")
        os.replace(out_dir / "last.pt.tmp", out_dir / "last.pt")
        log_line(out_dir, f"epoch {epoch + 1}: train {history[-1]['train_loss']:.4f} val GRU {val['gru']:.4f} vs const-vel {val['cv']:.4f}"
                          f"{' (best)' if improved else ''}; {(time.perf_counter() - t_start) / 60:.1f} min")
    torch.save({"state_dict": best_state, "hidden": predictor.hidden, "dataset": dataset, "best_val": best,
                "epochs_run": len(history)}, out_dir / "predictor.pt")
    (out_dir / "history.json").write_text(json.dumps({"history": history, "best_val_gru": best, "held_out_files": sorted(bank.files[i] for i in held_out)}, indent=1))
    (out_dir / "DONE").write_text(time.strftime("%Y-%m-%d %H:%M:%S\n"))
    log_line(out_dir, "DONE")
    return out_dir / "predictor.pt"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="ShanghaiTech", choices=list(TRAIN_POSES))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--max-batches", type=int, default=None, help="smoke runs only")
    args = parser.parse_args()
    train(args.dataset, args.out_dir, args.seed, args.max_epochs, threads=args.threads, max_batches=args.max_batches)


if __name__ == "__main__":
    main()
