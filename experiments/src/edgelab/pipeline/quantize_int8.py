"""ORT INT8 static quantization (QDQ, per-channel) of the static yolov8n-pose ONNX files (EXP-012 plan).

    uv run --extra pose python -m edgelab.pipeline.quantize_int8 --fp32 <yolov8n-pose-800-480x800.onnx> --out <int8.onnx> \
        --videos-dir ~/thesis-data/shanghaitech/shanghaitech/training/videos [--n 200] [--decode fp32|int8] [--method minmax]

Calibration frames: `n` TRAINING videos picked at random (seed), one random frame each, no darkening, no enhancement,
letterboxed exactly like at inference (ort_pose.letterbox). The same frame list is used for every resolution
(`calibration_frames` is deterministic) and is written next to the output as <out>.calib.txt.
`--decode fp32` leaves the box-decode arithmetic at the end of the head (Add/Sub/Mul/Div/Concat/Sigmoid under /model.22/,
not the convolutions) in float: those ops turn grid offsets into pixel boxes and their values reach 800, which 8 bits cannot hold.
"""

import argparse
import random
from pathlib import Path

import cv2
import numpy as np

from edgelab.pipeline.ort_pose import letterbox

HEAD_PREFIX = "/model.22/"  # the Pose head in the Ultralytics yolov8n-pose graph


def calibration_frames(videos_dir: Path, n: int = 200, seed: int = 0) -> list[tuple[str, int]]:
    """[(video file name, frame index)]: n different videos (seeded sample), one seeded random frame in each."""
    videos = sorted(p for p in videos_dir.iterdir() if p.suffix in (".avi", ".mp4"))
    if len(videos) < n:
        raise ValueError(f"{len(videos)} videos in {videos_dir}, need {n}")
    rng = random.Random(f"calib|{seed}")
    chosen = sorted(rng.sample(videos, n))
    picks = []
    for path in chosen:
        count = int(cv2.VideoCapture(str(path)).get(cv2.CAP_PROP_FRAME_COUNT))
        picks.append((path.name, rng.randrange(count)))
    return picks


def read_frame(videos_dir: Path, video: str, index: int) -> np.ndarray:
    capture = cv2.VideoCapture(str(videos_dir / video))
    capture.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = capture.read()
    if not ok:
        raise ValueError(f"cannot read frame {index} of {video}")
    return frame


def decode_node_names(model) -> list[str]:
    """Names of the head's non-convolution nodes (box decode, sigmoid, concat, reshape): kept in float with --decode fp32."""
    return [n.name for n in model.graph.node
            if n.name.startswith(HEAD_PREFIX) and n.name.count("/") == 2 and n.op_type != "Conv"]


def _make_reader(frames: list[np.ndarray], input_name: str, shape: tuple[int, int]):
    from onnxruntime.quantization import CalibrationDataReader

    class Reader(CalibrationDataReader):
        def __init__(self):
            self.blobs = iter([letterbox(f, shape)[0] for f in frames])

        def get_next(self):
            blob = next(self.blobs, None)
            return None if blob is None else {input_name: blob}

    return Reader()


def quantize(fp32: Path, out: Path, frames: list[np.ndarray], decode: str = "fp32", method: str = "minmax") -> dict:
    """Static QDQ per-channel quantization (weights int8, activations uint8). Returns a small summary dict."""
    import onnx
    from onnxruntime.quantization import CalibrationMethod, QuantFormat, QuantType, quantize_static

    model = onnx.load(str(fp32))
    input_name = model.graph.input[0].name
    shape = (model.graph.input[0].type.tensor_type.shape.dim[2].dim_value,
             model.graph.input[0].type.tensor_type.shape.dim[3].dim_value)
    exclude = decode_node_names(model) if decode == "fp32" else []
    calibrate = {"minmax": CalibrationMethod.MinMax, "percentile": CalibrationMethod.Percentile,
                 "entropy": CalibrationMethod.Entropy}[method]
    out.parent.mkdir(parents=True, exist_ok=True)
    quantize_static(str(fp32), str(out), _make_reader(frames, input_name, shape), quant_format=QuantFormat.QDQ,
                    per_channel=True, weight_type=QuantType.QInt8, activation_type=QuantType.QUInt8,
                    calibrate_method=calibrate, nodes_to_exclude=exclude)
    quantized = onnx.load(str(out))
    n_qdq = sum(n.op_type in ("QuantizeLinear", "DequantizeLinear") for n in quantized.graph.node)
    return {"input_hw": shape, "excluded_nodes": len(exclude), "qdq_nodes": n_qdq, "method": method, "decode": decode,
            "bytes": out.stat().st_size}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fp32", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--videos-dir", type=Path, required=True, help="ShanghaiTech TRAINING videos")
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--decode", choices=["fp32", "int8"], default="fp32")
    parser.add_argument("--method", choices=["minmax", "percentile", "entropy"], default="minmax")
    args = parser.parse_args()
    picks = calibration_frames(args.videos_dir, args.n, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_name(args.out.name + ".calib.txt").write_text("\n".join(f"{v} {i}" for v, i in picks) + "\n")
    frames = [read_frame(args.videos_dir, v, i) for v, i in picks]
    print(quantize(args.fp32, args.out, frames, args.decode, args.method))


if __name__ == "__main__":
    main()
