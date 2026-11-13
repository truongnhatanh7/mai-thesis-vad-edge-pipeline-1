"""EXP-020: yolov8n-pose + P2 head, initialised from yolov8n-pose.pt.

    uv run --extra pose python -m edgelab.pose.p2model --stock artifacts/EXP-001/models/yolov8n-pose.pt \
        --out ~/thesis-data/exp020/yolov8n-pose-p2-init.pt --report artifacts/EXP-020/p2_transfer.json

Architecture: `yolov8n-pose-p2.yaml` (append-only; layers 0-21 as in yolov8-pose.yaml, P2 branch = layers 22-24, Pose head =
layer 25 over [P3, P4, P5, P2]). Weight transfer is by explicit renaming, never by position guessing:
    model.<i>.*  (i = 0..21)            -> model.<i>.*       (backbone, neck, PAN: identical layers)
    model.22.{cv2,cv3,cv4}.<k>.*, k<3   -> model.25.{cv2,cv3,cv4}.<k>.*
    model.22.dfl.*                      -> model.25.dfl.*
Every target tensor must have the shape of its source or the build fails. Everything else (layers 22-24 and the level-3
head modules cv2/cv3/cv4.3) keeps Ultralytics' random initialisation and is reported as "new".
The pre-flight gate (tests): the P3/P4/P5 outputs of the new model equal the stock model's on the same input.
"""

import argparse
import json
import re
from pathlib import Path

import torch

P2_YAML = Path(__file__).with_name("yolov8n-pose-p2.yaml")
STOCK_HEAD_INDEX, P2_HEAD_INDEX = 22, 25
STOCK_LEVELS = 3


def stock_key_to_p2(key: str) -> str | None:
    """Name of the P2-model tensor that receives stock tensor `key`, or None if the stock tensor has no counterpart."""
    m = re.match(r"model\.(\d+)\.(.*)$", key)
    if m is None:
        return None
    layer, rest = int(m.group(1)), m.group(2)
    if layer < STOCK_HEAD_INDEX:
        return key
    if layer != STOCK_HEAD_INDEX:
        return None
    level = re.match(r"(?:cv2|cv3|cv4)\.(\d+)\.", rest)
    if level is not None and int(level.group(1)) < STOCK_LEVELS:
        return f"model.{P2_HEAD_INDEX}.{rest}"
    if rest.startswith("dfl."):
        return f"model.{P2_HEAD_INDEX}.{rest}"
    return None


def transfer(stock_state: dict, p2_state: dict) -> tuple[dict, dict]:
    """-> (new state dict for the P2 model, report). Raises if a mapped tensor does not exist or has another shape."""
    out = dict(p2_state)
    copied, unmapped_stock = [], []
    for key, tensor in stock_state.items():
        target = stock_key_to_p2(key)
        if target is None:
            unmapped_stock.append(key)
            continue
        if target not in p2_state or tuple(p2_state[target].shape) != tuple(tensor.shape):
            raise ValueError(f"{key} {tuple(tensor.shape)} -> {target} {tuple(p2_state[target].shape) if target in p2_state else None}")
        out[target] = tensor.clone().to(p2_state[target].dtype)
        copied.append(target)
    new = sorted(set(p2_state) - set(copied))
    report = {"tensors_stock": len(stock_state), "tensors_p2": len(p2_state), "copied": len(copied),
              "copied_elements": int(sum(p2_state[k].numel() for k in copied)),
              "new": len(new), "new_elements": int(sum(p2_state[k].numel() for k in new)),
              "stock_tensors_without_counterpart": unmapped_stock,
              "new_tensors": new}
    return out, report


def build_p2(stock_model: torch.nn.Module):
    """stock PoseModel (e.g. YOLO('yolov8n-pose.pt').model) -> (P2 PoseModel with transferred weights, report)."""
    from ultralytics.nn.tasks import PoseModel

    p2 = PoseModel(str(P2_YAML), nc=1, data_kpt_shape=(17, 3), verbose=False)
    state, report = transfer(stock_model.float().state_dict(), p2.state_dict())
    p2.load_state_dict(state)
    return p2, report


def save_checkpoint(p2: torch.nn.Module, path: Path):
    """Training checkpoint Ultralytics accepts as `YOLO(path)`. `train_args.task = "pose"` is essential: without it Ultralytics
    guesses "detect" from its default args, builds a box-only dataset and trains a pose model without keypoints."""
    import copy
    from datetime import datetime

    import ultralytics

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": copy.deepcopy(p2).half(), "train_args": {"task": "pose"}, "date": datetime.now().isoformat(),
                "version": ultralytics.__version__, "license": "AGPL-3.0 License (https://ultralytics.com/license)"}, path)


def complexity(model: torch.nn.Module, imgsz) -> dict:
    """Parameters and GFLOPs (Ultralytics' counter) of the fused model at imgsz (int or (h, w))."""
    import copy

    from ultralytics.utils.torch_utils import get_flops

    fused = copy.deepcopy(model).float().eval().fuse(verbose=False)
    return {"params": int(sum(p.numel() for p in fused.parameters())), "gflops": round(float(get_flops(fused, imgsz)), 3)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stock", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="checkpoint to start training from")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    from ultralytics import YOLO

    stock = YOLO(str(args.stock))
    p2, report = build_p2(stock.model)
    p2.eval()
    for name, model in (("stock", stock.model), ("p2", p2)):
        report[name] = {"params_unfused": int(sum(p.numel() for p in model.parameters())),
                        "fused_640x640": complexity(model, 640), "fused_384x640": complexity(model, (384, 640))}
    report["strides"] = [float(s) for s in p2.stride]
    save_checkpoint(p2, args.out)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k not in ("new_tensors", "stock_tensors_without_counterpart")}))


if __name__ == "__main__":
    main()
