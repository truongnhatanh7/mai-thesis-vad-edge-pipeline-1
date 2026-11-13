"""Static ONNX of yolov8n-pose at the exact input rectangle Ultralytics' predictor uses for our frames (EXP-004 Amendment 1).

    uv run --extra pose python -m edgelab.pipeline.export_onnx --imgsz 480 640 800 960 --out-dir <vm-exchange/models>

The shape is read from a real `model.predict` call (the tensor that reaches the network), not computed by hand.
Output: <name>-<R>-<h>x<w>.onnx (name = yolov8n-pose unless --name) and a .sha256 file next to it.
"""

import argparse
import hashlib
import shutil
import tempfile
from pathlib import Path

import numpy as np

from edgelab.pose.detcache import WEIGHTS

FRAME_SHAPE = (480, 856)  # ShanghaiTech test frames (h, w)


def predictor_input_shape(model, imgsz: int, frame_shape: tuple[int, int] = FRAME_SHAPE) -> tuple[int, int]:
    """(h, w) of the tensor Ultralytics feeds to the .pt network for a frame of `frame_shape` at imgsz."""
    seen = []

    def spy(predictor):
        network = predictor.model.model  # the torch module inside Ultralytics' AutoBackend
        original = network.forward

        def forward(x, *args, **kwargs):
            seen.append(tuple(x.shape[2:]))
            return original(x, *args, **kwargs)

        network.forward = forward

    model.add_callback("on_predict_start", spy)
    model.predict(np.full((*frame_shape, 3), 128, np.uint8), imgsz=imgsz, device="cpu", verbose=False)
    model.callbacks["on_predict_start"].remove(spy)
    model.predictor = None  # the next predict builds a fresh predictor
    return seen[0]


def export_static(weights: Path, shape: tuple[int, int], target: Path, name: str = "yolov8n-pose", nms_free: bool = False) -> str:
    """Export at imgsz=[h, w], opset 17, static, no simplify (same settings as the 640x640 export). Returns sha256."""
    from ultralytics import YOLO

    with tempfile.TemporaryDirectory() as tmp:
        local = Path(tmp) / f"{name}.pt"
        shutil.copy(weights, local)
        exported = YOLO(str(local)).export(format="onnx", imgsz=list(shape), opset=17, dynamic=False,
                                           simplify=False, half=False, **({"nms": False} if nms_free else {}))
        shutil.copy(exported, target)
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    target.with_name(target.name + ".sha256").write_text(f"{digest}  {target.name}\n")
    return digest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--imgsz", type=int, nargs="+", default=[480, 640, 800, 960])
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--weights", type=Path, default=WEIGHTS)
    parser.add_argument("--name", default="yolov8n-pose", help="model name in the output file (EXP-010: yolov8s-pose)")
    parser.add_argument("--nms-free", action="store_true", help="EXP-017: end-to-end NMS-free head (YOLO26), output (1, 300, 57)")
    args = parser.parse_args()
    from ultralytics import YOLO

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(args.weights))
    for r in args.imgsz:
        h, w = predictor_input_shape(model, r)
        target = args.out_dir / f"{args.name}-{r}-{h}x{w}.onnx"
        print(f"R={r}: predictor input (h, w) = ({h}, {w}) -> {target.name} sha256 {export_static(args.weights, (h, w), target, args.name, args.nms_free)}",
              flush=True)


if __name__ == "__main__":
    main()
