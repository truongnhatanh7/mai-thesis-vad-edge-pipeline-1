"""YOLOv8n-pose through the NCNN Python package (CPU), same pre/post-processing and call signature as ort_pose.OrtPose.

NCNN model = the folder Ultralytics writes for `YOLO(...).export(format="ncnn", imgsz=[h, w])`
(model.ncnn.param + model.ncnn.bin; input blob "in0", output blob "out0" of shape (56, N), like the ONNX output without batch).

    letterbox (ort_pose.letterbox) -> NCNN extract -> ort_pose.decode (conf filter + NMS)

`fp16=True` lets NCNN store/compute in half precision where the CPU supports it (NCNN's own default on ARM);
`fp16=False` keeps FP32 storage and arithmetic. The input size is read from the folder's metadata.yaml (imgsz: [h, w]).
"""

import time
from pathlib import Path

import numpy as np
import yaml

from edgelab.pipeline.ort_pose import decode_any, letterbox


def export_ncnn(weights: Path, shape: tuple[int, int], out_dir: Path, nms_free: bool = False) -> Path:
    """Ultralytics NCNN export at (h, w) into out_dir (the weights are copied to a scratch dir first). Returns out_dir.
    nms_free (EXP-017): export the end-to-end NMS-free head of YOLO26 (`nms=False`)."""
    import shutil
    import tempfile

    from ultralytics import YOLO

    with tempfile.TemporaryDirectory() as tmp:
        local = Path(tmp) / weights.name
        shutil.copy(weights, local)
        extra = {"nms": False} if nms_free else {}
        exported = Path(YOLO(str(local)).export(format="ncnn", imgsz=list(shape), half=False, **extra))
        if out_dir.exists():
            shutil.rmtree(out_dir)
        shutil.copytree(exported, out_dir)
    return out_dir


class NcnnPose:
    def __init__(self, model_dir: str | Path, threads: int = 0, fp16: bool = True):
        import ncnn

        model_dir = Path(model_dir)
        meta = yaml.safe_load((model_dir / "metadata.yaml").read_text())
        self.shape = (int(meta["imgsz"][0]), int(meta["imgsz"][1]))
        self.net = ncnn.Net()
        self.net.opt.use_vulkan_compute = False
        if threads:
            self.net.opt.num_threads = threads
        self.net.opt.use_fp16_packed = fp16
        self.net.opt.use_fp16_storage = fp16
        self.net.opt.use_fp16_arithmetic = fp16
        self.net.load_param(str(model_dir / "model.ncnn.param"))
        self.net.load_model(str(model_dir / "model.ncnn.bin"))

    def raw(self, blob: np.ndarray) -> np.ndarray:
        """(1, 3, h, w) float32 -> (1, 56, N), the same layout as the ONNX output."""
        import ncnn

        extractor = self.net.create_extractor()
        extractor.input("in0", ncnn.Mat(np.ascontiguousarray(blob[0])).clone())
        _, out = extractor.extract("out0")
        return np.array(out)[None]

    def __call__(self, img_bgr: np.ndarray, conf: float):
        """-> (boxes, conf, keypoints, {"pre_ms", "infer_ms", "post_ms"}), like OrtPose."""
        t0 = time.perf_counter()
        blob, ratio, pad = letterbox(img_bgr, self.shape)
        t1 = time.perf_counter()
        output = self.raw(blob)
        t2 = time.perf_counter()
        boxes, scores, keypoints = decode_any(output, conf, ratio, pad, img_bgr.shape[:2])
        t3 = time.perf_counter()
        return boxes, scores, keypoints, {"pre_ms": (t1 - t0) * 1e3, "infer_ms": (t2 - t1) * 1e3,
                                          "post_ms": (t3 - t2) * 1e3}


def main():
    """Export: python -m edgelab.pipeline.ncnn_pose --weights <pt> --h 384 --w 640 --out <dir>"""
    import argparse

    from edgelab.pose.detcache import WEIGHTS

    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, default=WEIGHTS)
    parser.add_argument("--h", type=int, required=True)
    parser.add_argument("--w", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--nms-free", action="store_true", help="EXP-017: end-to-end YOLO26 head")
    args = parser.parse_args()
    print(export_ncnn(args.weights, (args.h, args.w), args.out, args.nms_free))


if __name__ == "__main__":
    main()
