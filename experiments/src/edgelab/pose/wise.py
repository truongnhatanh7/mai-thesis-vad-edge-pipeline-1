"""EXP-014: weight-space interpolation (WiSE-FT) between two YOLOv8n-pose checkpoints of the same architecture.

    uv run --extra pose python -m edgelab.pose.wise --original <a.pt> --finetuned <b.pt> --alpha 0.5 --out <c.pt>

theta(alpha) = (1 - alpha) * theta_original + alpha * theta_finetuned, over every floating-point tensor of the state dict
(weights, biases, BatchNorm running mean / variance). Integer entries (BatchNorm `num_batches_tracked`) are copied from
the fine-tuned model. The result is saved with Ultralytics' own `save`, so detcache/YOLO can load it like any weights.
"""

import argparse

import torch


def interpolate_state_dicts(original: dict, finetuned: dict, alpha: float) -> dict:
    if original.keys() != finetuned.keys():
        raise ValueError("state dicts have different keys: the two models are not the same architecture")
    mixed = {}
    for key, tensor_b in finetuned.items():
        tensor_a = original[key]
        if tensor_a.shape != tensor_b.shape:
            raise ValueError(f"shape mismatch at {key}: {tuple(tensor_a.shape)} vs {tuple(tensor_b.shape)}")
        if tensor_b.is_floating_point():
            mixed[key] = (1 - alpha) * tensor_a.float() + alpha * tensor_b.float()
        else:
            mixed[key] = tensor_b.clone()
    return mixed


def recalibrate_batchnorm(net: torch.nn.Module, image_paths: list[str], imgsz: int = 640, batch: int = 16) -> int:
    """Amendment 1: reset BatchNorm running stats and re-estimate them (cumulative average) from unlabeled images.
    Averaged weights + averaged statistics do not match; without this the interpolated models output nothing."""
    import cv2
    import numpy as np
    from ultralytics.data.augment import LetterBox

    letterbox = LetterBox((imgsz, imgsz), auto=False)
    bns = [m for m in net.modules() if isinstance(m, torch.nn.BatchNorm2d)]
    for m in bns:
        m.reset_running_stats()
        m.momentum = None
    dtype = next(net.parameters()).dtype
    net.float().train()
    seen = 0
    with torch.no_grad():
        for i in range(0, len(image_paths), batch):
            imgs = [letterbox(image=cv2.imread(p))[:, :, ::-1] for p in image_paths[i:i + batch]]
            x = torch.from_numpy(np.ascontiguousarray(np.stack(imgs).transpose(0, 3, 1, 2))).float() / 255.0
            net(x)
            seen += len(imgs)
    net.eval().to(dtype)
    return seen


def make_wise_weights(original: str, finetuned: str, alpha: float, out: str, calib_images: list[str] | None = None) -> str:
    from ultralytics import YOLO

    model_a, model_b = YOLO(original), YOLO(finetuned)
    state_a = {k: v.detach().cpu() for k, v in model_a.model.state_dict().items()}
    state_b = {k: v.detach().cpu() for k, v in model_b.model.state_dict().items()}
    model_b.model.load_state_dict(interpolate_state_dicts(state_a, state_b, alpha))  # keeps fine-tuned names / metadata
    if calib_images:
        recalibrate_batchnorm(model_b.model, calib_images)
    model_b.save(out)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", required=True)
    parser.add_argument("--finetuned", required=True)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--calib-dir", default=None, help="Amendment 1: training images for BatchNorm recalibration")
    parser.add_argument("--calib-n", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    torch.set_num_threads(4)
    calib = None
    if args.calib_dir:
        import random
        from pathlib import Path
        files = sorted(str(p) for p in Path(args.calib_dir).glob("*.jpg"))
        calib = random.Random(args.seed).sample(files, min(args.calib_n, len(files)))
    print(make_wise_weights(args.original, args.finetuned, args.alpha, args.out, calib))


if __name__ == "__main__":
    main()
