"""POST-HOC (not pre-registered): score a trained checkpoint on every video of the dataset, train+val+test.

Mimics a report over the whole dataset (the anchor's Hockey Table 9 row = all 1000 videos).
uv run python -m edgelab.models.eval_whole_dataset --run hockey-fight-video-seed0
"""

import argparse

import mlflow
import torch
from torch.utils.data import DataLoader

from edgelab.data.splits import load_split
from edgelab.data.video_clips import ClipDataset, make_clip_items, read_index
from edgelab.models.salman_cnn_gru import SalmanCnnGru
from edgelab.models.train_clips import ARTIFACTS, CACHE_ROOT, EXP_DIR, pick_device, predict, test_metrics
from edgelab.tracking.mlflow_run import tracked_run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="hockey-fight-video-seed0")
    args = parser.parse_args()
    dataset, protocol, seed_part = args.run.rsplit("-", 2)
    device = pick_device()
    items = make_clip_items(read_index(CACHE_ROOT / dataset), clip_len=8, stride=2, max_clips=8)
    split = load_split(EXP_DIR / "splits" / f"{dataset}_{protocol}_{seed_part}.json", items)
    model = SalmanCnnGru().to(device)
    model.load_state_dict(torch.load(ARTIFACTS / args.run / "best.pt", map_location=device))
    loader = DataLoader(ClipDataset(CACHE_ROOT / dataset, items), batch_size=64)
    probs, labels, indices, _ = predict(model, loader, device)
    partition_of = {i: p for p, idx in split.items() for i in idx}
    results = {}
    for name, keep in {"all": lambda p: True, "train": lambda p: p == "train", "test": lambda p: p == "test"}.items():
        mask = [keep(partition_of[i]) for i in indices]
        m = test_metrics(probs[mask], labels[mask], indices[mask], items)
        results.update({f"{name}_video_acc": m["test_video_acc"], f"{name}_clip_acc": m["test_clip_acc"],
                        f"{name}_videos": m["test_videos"]})
    with tracked_run("EXP-002", f"posthoc-whole-dataset-{args.run}", {"source_run": args.run, "posthoc": True}) as r:
        mlflow.log_metrics(results)
        results["run_id"] = r.info.run_id
    print(results)


if __name__ == "__main__":
    main()
