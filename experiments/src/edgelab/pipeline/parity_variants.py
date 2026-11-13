"""Parity of a detector variant (ORT INT8 or NCNN) against the ORT FP32 model of the same input size (EXP-012 step 1).

    uv run --extra pose python -m edgelab.pipeline.parity_variants --fp32 <fp32.onnx> --variant <int8.onnx | ncnn dir> \
        --frames a.jpg b.jpg c.jpg [--conf 0.25] [--fp16]

Per frame: box counts at --conf, and for FP32 boxes matched to a variant box (greedy, best IoU, IoU >= 0.5):
|confidence difference| and the keypoint distance in pixels (keypoints the FP32 model gives confidence >= 0.5 only).
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

from edgelab.pipeline.ncnn_pose import NcnnPose
from edgelab.pipeline.ort_pose import OrtPose


def iou(a: np.ndarray, b: np.ndarray) -> float:
    """IoU of two xyxy boxes."""
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = w * h
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / union) if union > 0 else 0.0


def compare(ref, other, min_iou: float = 0.5) -> dict:
    """ref, other = (boxes, conf, keypoints) of one frame. Each FP32 box is matched to its best unused variant box."""
    ref_boxes, ref_conf, ref_kpt = ref
    boxes, conf, kpt = other
    used, conf_diffs, kpt_diffs = set(), [], []
    for i in range(len(ref_conf)):
        best, best_iou = None, min_iou
        for j in range(len(conf)):
            value = iou(ref_boxes[i], boxes[j])
            if j not in used and value >= best_iou:
                best, best_iou = j, value
        if best is None:
            continue
        used.add(best)
        conf_diffs.append(abs(float(ref_conf[i]) - float(conf[best])))
        visible = ref_kpt[i, :, 2] >= 0.5
        kpt_diffs.extend(np.linalg.norm(ref_kpt[i, visible, :2] - kpt[best, visible, :2], axis=1).tolist())
    return {"n_ref": len(ref_conf), "n_variant": len(conf), "n_matched": len(conf_diffs),
            "max_conf_diff": max(conf_diffs, default=0.0), "mean_conf_diff": float(np.mean(conf_diffs)) if conf_diffs else 0.0,
            "max_kpt_px": max(kpt_diffs, default=0.0), "mean_kpt_px": float(np.mean(kpt_diffs)) if kpt_diffs else 0.0}


def load_variant(path: Path, fp16: bool = True):
    """A folder is an NCNN model, a file is an ONNX (FP32 or INT8)."""
    return NcnnPose(path, threads=2, fp16=fp16) if path.is_dir() else OrtPose(path, threads=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fp32", type=Path, required=True)
    parser.add_argument("--variant", type=Path, required=True)
    parser.add_argument("--frames", type=Path, nargs="+", required=True)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--fp16", action="store_true", help="NCNN only: half-precision storage/arithmetic")
    args = parser.parse_args()
    reference, variant = OrtPose(args.fp32, threads=2), load_variant(args.variant, args.fp16)
    for frame in args.frames:
        img = cv2.imread(str(frame))
        row = compare(reference(img, args.conf)[:3], variant(img, args.conf)[:3])
        print(frame.name, {k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()})


if __name__ == "__main__":
    main()
