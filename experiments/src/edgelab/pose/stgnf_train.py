"""Train an STG-NF variant on released training poses (EXP-001 recipe) — resumable, one folder per run.

    uv run --extra pose python -m edgelab.pose.stgnf_train --kind standard|graph|gau|graph-fa|gau-fa --dataset ShanghaiTech \
        --seed 0 --out-dir ~/thesis-data/exp019/models/standard_s0 [--fill cv|gru --predictor <predictor.pt>]

Recipe (third_party/STG-NF args.py defaults, as in EXP-001): 8 epochs, batch 256, Adamax lr 5e-4 (lr x 0.99**epoch after each
epoch), gradient-norm clip 100, windows of 24 poses every 6 poses (UBnormal: 16), 2 augmentations (identity and mirror, so every
window is seen twice per epoch), prior N(3, 1), loss = mean bits-per-dimension NLL. `standard` runs third_party's own STG_NF class,
the other kinds run edgelab.models.stgnf_flow. With --fill, every window is also dropped and filled at random (augment.py) and
the model gets the provenance channel (kinds ending in -fa).

Files in --out-dir: last.pt (after every epoch; a restart continues from it), final.pt + DONE (finished), train.log.
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from edgelab.pose.augment import make_batch
from edgelab.pose.predictor import load_predictor
from edgelab.pose.score_model import build_model, mark_actnorm_initialised, parameter_count
from edgelab.pose.segments import SegmentBank, build_bank
from edgelab.pose.stgnf_eval import SEG_LENS

DATA = Path.home() / "thesis-data"
TRAIN_POSES = {"ShanghaiTech": DATA / "stgnf/data/ShanghaiTech/pose/train", "UBnormal": DATA / "stgnf/data/UBnormal/pose/train"}
STRIDE = 6
CTX = 16
BANK_DIR = DATA / "exp019/banks"


def bank_path(dataset: str) -> Path:
    return BANK_DIR / f"{dataset}_len{SEG_LENS[dataset]}_stride{STRIDE}_ctx{CTX}.npz"


def load_bank(dataset: str) -> SegmentBank:
    """The training windows of `dataset`; built once from the pose JSON (a few minutes), then cached as an npz."""
    path = bank_path(dataset)
    if not path.exists():
        bank = build_bank(TRAIN_POSES[dataset], SEG_LENS[dataset], STRIDE, CTX)
        BANK_DIR.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.stem + ".partial.npz")
        np.savez(tmp, xy=bank.xy, grid=bank.grid, file_id=bank.file_id, files=np.array(json.dumps(bank.files)),
                 ctx=bank.ctx, seg_len=bank.seg_len)
        os.replace(tmp, path)
    data = np.load(path)
    return SegmentBank(data["xy"], data["grid"], data["file_id"], json.loads(str(data["files"])), int(data["ctx"]), int(data["seg_len"]))


def log_line(out_dir: Path, text: str):
    with open(out_dir / "train.log", "a") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {text}\n")


def train(kind: str, dataset: str, seed: int, out_dir: Path, epochs: int = 8, batch_size: int = 256, lr: float = 5e-4,
          fill: str | None = None, predictor_path: Path | None = None, threads: int = 2, max_batches: int | None = None) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / "DONE").exists():
        return out_dir / "final.pt"
    torch.set_num_threads(threads)
    bank = load_bank(dataset)
    seg_len = SEG_LENS[dataset]
    torch.manual_seed(seed)
    model = build_model(kind, seg_len)
    optimizer = torch.optim.Adamax(model.parameters(), lr=lr)
    predictor = load_predictor(predictor_path) if fill == "gru" else None
    start_epoch, seconds_before = 0, 0.0
    if (out_dir / "last.pt").exists():
        saved = torch.load(out_dir / "last.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(saved["state_dict"])
        optimizer.load_state_dict(saved["optimizer"])
        mark_actnorm_initialised(model)
        start_epoch, seconds_before = saved["epoch"], saved["seconds"]
        log_line(out_dir, f"resumed after epoch {start_epoch}")
    n = len(bank)
    log_line(out_dir, f"kind={kind} dataset={dataset} seed={seed} fill={fill} windows={n} params={parameter_count(model)} threads={threads}")
    t_start, last_loss = time.perf_counter(), float("nan")
    for epoch in range(start_epoch, epochs):
        rng = np.random.default_rng([seed, epoch])  # the data order of an epoch depends only on (seed, epoch): restart-safe
        rows = np.concatenate([np.arange(n), np.arange(n)])
        flips = np.concatenate([np.zeros(n, dtype=bool), np.ones(n, dtype=bool)])
        order = rng.permutation(2 * n)
        model.train()
        losses, windows_used = [], 0
        for b, begin in enumerate(range(0, 2 * n, batch_size)):
            if max_batches is not None and b >= max_batches:
                break
            pick = order[begin:begin + batch_size]
            x, provenance = make_batch(bank, rows[pick], fill, predictor, rng, flips[pick])
            if len(x) == 0:
                continue
            _, nll = model(x, provenance)
            loss = nll.mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 100)
            optimizer.step()
            optimizer.zero_grad()
            losses.append(loss.item())
            windows_used += len(x)
            if b % 200 == 0:
                log_line(out_dir, f"epoch {epoch + 1} batch {b} loss {np.mean(losses[-200:]):.4f} windows_used {windows_used}")
        last_loss = float(np.mean(losses)) if losses else float("nan")
        for group in optimizer.param_groups:
            group["lr"] = lr * 0.99 ** epoch
        seconds = seconds_before + time.perf_counter() - t_start
        torch.save({"state_dict": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch + 1, "seconds": seconds},
                   out_dir / "last.pt.tmp")
        os.replace(out_dir / "last.pt.tmp", out_dir / "last.pt")
        log_line(out_dir, f"epoch {epoch + 1} done: mean loss {last_loss:.4f}, windows used {windows_used}, {seconds / 60:.1f} min")
    config = {"kind": kind, "dataset": dataset, "seg_len": seg_len, "seed": seed, "epochs": epochs, "batch_size": batch_size,
              "lr": lr, "fill": fill, "predictor": str(predictor_path) if predictor_path else None, "n_windows": n,
              "params": parameter_count(model), "final_epoch_mean_loss": last_loss, "threads": threads,
              "train_minutes": (seconds_before + time.perf_counter() - t_start) / 60}
    torch.save({"kind": kind, "state_dict": model.state_dict(), "config": config}, out_dir / "final.pt")
    (out_dir / "config.json").write_text(json.dumps(config, indent=1))
    (out_dir / "DONE").write_text(time.strftime("%Y-%m-%d %H:%M:%S\n"))
    log_line(out_dir, "DONE")
    return out_dir / "final.pt"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", required=True, choices=["standard", "graph", "gau", "graph-fa", "gau-fa", "gau-fa-mask"])
    parser.add_argument("--dataset", default="ShanghaiTech", choices=list(TRAIN_POSES))
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--fill", choices=["cv", "gru"], default=None)
    parser.add_argument("--predictor", type=Path, default=None)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--max-batches", type=int, default=None, help="smoke runs only")
    parser.add_argument("--build-bank-only", action="store_true")
    args = parser.parse_args()
    if args.build_bank_only:
        bank = load_bank(args.dataset)
        print(json.dumps({"windows": len(bank), "poses": len(bank.xy), "files": len(bank.files)}))
        return
    train(args.kind, args.dataset, args.seed, args.out_dir, args.epochs, fill=args.fill, predictor_path=args.predictor,
          threads=args.threads, max_batches=args.max_batches)


if __name__ == "__main__":
    main()
