"""YOLOv8n-pose through ONNX Runtime on CPU, with our own minimal pre/post-processing (EXP-004 plan §3.1).

    letterbox to the model's static input shape (grey 114 padding, centred, like Ultralytics) -> ORT -> decode -> conf filter -> NMS.
The static ONNX of each R is exported at the exact rectangle Ultralytics' predictor uses for 856x480 frames at imgsz=R
(auto=True letterbox, stride 32): R=480 -> 288x480, 640 -> 384x640, 800 -> 480x800, 960 -> 544x960 (see EXP-004 log).
Output rows per anchor: cx, cy, w, h, person confidence, then 17 x (x, y, conf) keypoints, all in letterbox pixels.

Parity check against Ultralytics (host):
    python -m edgelab.pipeline.ort_pose --onnx <file> --pt <weights> --imgsz R --frames a.jpg b.jpg
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

IMGSZ, IOU = 640, 0.7
PAD_VALUE = 114


def make_session(path: str | Path, threads: int = 0, spin: bool = True):
    """threads = 0 means ORT's own default. spin=False turns off the intra-op threads' busy-wait."""
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    if not spin:
        options.add_session_config_entry("session.intra_op.allow_spinning", "0")
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


def letterbox(img_bgr: np.ndarray, shape: tuple[int, int] = (IMGSZ, IMGSZ)):
    """-> (1, 3, h, w) float32 RGB in [0, 1], scale ratio, (pad_left, pad_top). shape = (h, w). Same rounding as Ultralytics."""
    height, width = shape
    h, w = img_bgr.shape[:2]
    ratio = min(height / h, width / w)
    new_w, new_h = round(w * ratio), round(h * ratio)
    pad_w, pad_h = (width - new_w) / 2, (height - new_h) / 2
    resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    top, bottom = round(pad_h - 0.1), round(pad_h + 0.1)
    left, right = round(pad_w - 0.1), round(pad_w + 0.1)
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(PAD_VALUE,) * 3)
    blob = padded[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
    return np.ascontiguousarray(blob), ratio, (left, top)


def decode(output: np.ndarray, conf: float, ratio: float, pad: tuple[int, int], shape: tuple[int, int],
           iou: float = IOU):
    """output (1, 56, N) -> boxes xyxy (n, 4), conf (n,), keypoints (n, 17, 3) in original-image pixels, sorted by conf."""
    rows = output[0].T  # (N, 56)
    rows = rows[rows[:, 4] >= conf]
    if len(rows) == 0:
        return np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros((0, 17, 3), np.float32)
    cx, cy, w, h = rows[:, :4].T
    xywh_topleft = np.stack([cx - w / 2, cy - h / 2, w, h], axis=1)
    keep = np.asarray(cv2.dnn.NMSBoxes(xywh_topleft.tolist(), rows[:, 4].tolist(), conf, iou)).reshape(-1)
    rows, box = rows[keep], xywh_topleft[keep]
    order = np.argsort(-rows[:, 4])
    rows, box = rows[order], box[order]
    height, width = shape
    xyxy = np.stack([box[:, 0], box[:, 1], box[:, 0] + box[:, 2], box[:, 1] + box[:, 3]], axis=1)
    xyxy = (xyxy - [*pad, *pad]) / ratio
    xyxy[:, [0, 2]] = xyxy[:, [0, 2]].clip(0, width)
    xyxy[:, [1, 3]] = xyxy[:, [1, 3]].clip(0, height)
    keypoints = rows[:, 5:].reshape(-1, 17, 3).copy()
    keypoints[:, :, :2] = (keypoints[:, :, :2] - pad) / ratio
    keypoints[:, :, 0] = keypoints[:, :, 0].clip(0, width)
    keypoints[:, :, 1] = keypoints[:, :, 1].clip(0, height)
    return xyxy.astype(np.float32), rows[:, 4].astype(np.float32), keypoints.astype(np.float32)


def decode_e2e(output: np.ndarray, conf: float, ratio: float, pad: tuple[int, int], shape: tuple[int, int]):
    """EXP-017: NMS-free end-to-end head (YOLO26, Ultralytics export `nms=False`). output (1, 300, 57): x1, y1, x2, y2,
    conf, class, then 17 x (x, y, conf), letterbox pixels, no NMS needed; keep conf > `conf` (Ultralytics' own rule).
    Same return format as `decode`: boxes xyxy (n, 4), conf (n,), keypoints (n, 17, 3) in original pixels, sorted by conf."""
    rows = output[0]
    rows = rows[rows[:, 4] > conf]
    rows = rows[np.argsort(-rows[:, 4], kind="stable")]
    height, width = shape
    xyxy = (rows[:, :4] - [*pad, *pad]) / ratio
    xyxy[:, [0, 2]] = xyxy[:, [0, 2]].clip(0, width)
    xyxy[:, [1, 3]] = xyxy[:, [1, 3]].clip(0, height)
    keypoints = rows[:, 6:].reshape(-1, 17, 3).copy()
    keypoints[:, :, :2] = (keypoints[:, :, :2] - pad) / ratio
    keypoints[:, :, 0] = keypoints[:, :, 0].clip(0, width)
    keypoints[:, :, 1] = keypoints[:, :, 1].clip(0, height)
    return xyxy.astype(np.float32), rows[:, 4].astype(np.float32), keypoints.astype(np.float32)


def decode_any(output: np.ndarray, conf: float, ratio: float, pad: tuple[int, int], shape: tuple[int, int], iou: float = IOU):
    """YOLOv8-style (1, 56, N) -> `decode` (conf filter + NMS); end-to-end (1, 300, 57) -> `decode_e2e` (conf filter only)."""
    if output.shape[1] == 56:
        return decode(output, conf, ratio, pad, shape, iou)
    if output.shape[2] == 57:
        return decode_e2e(output, conf, ratio, pad, shape)
    raise ValueError(f"unknown pose output layout {output.shape}")


class OrtPose:
    def __init__(self, onnx_path: str | Path, threads: int = 0, spin: bool = True):
        self.session = make_session(onnx_path, threads, spin)
        model_input = self.session.get_inputs()[0]
        self.input_name = model_input.name
        self.shape = (int(model_input.shape[2]), int(model_input.shape[3]))  # static (h, w) of this export

    def __call__(self, img_bgr: np.ndarray, conf: float):
        """-> (boxes, conf, keypoints, {"pre_ms", "infer_ms", "post_ms"})."""
        t0 = time.perf_counter()
        blob, ratio, pad = letterbox(img_bgr, self.shape)
        t1 = time.perf_counter()
        output = self.session.run(None, {self.input_name: blob})[0]
        t2 = time.perf_counter()
        boxes, scores, keypoints = decode_any(output, conf, ratio, pad, img_bgr.shape[:2])
        t3 = time.perf_counter()
        return boxes, scores, keypoints, {"pre_ms": (t1 - t0) * 1e3, "infer_ms": (t2 - t1) * 1e3,
                                          "post_ms": (t3 - t2) * 1e3}


def compare_with_ultralytics(model_path: str, frames: list[str], pose: OrtPose, confs: list[float],
                             imgsz: int, nms_free: bool = False) -> list[dict]:
    """Same frames through Ultralytics predict (.pt, imgsz, iou 0.7, CPU) and through OrtPose, one row per frame and conf."""
    from ultralytics import YOLO

    model = YOLO(model_path)
    rows = []
    for frame in frames:
        img = cv2.imread(frame)
        for conf in confs:
            result = model.predict(img, imgsz=imgsz, conf=conf, iou=IOU, device="cpu", verbose=False,
                                   **({"nms": False} if nms_free else {}))[0]
            ref_conf = result.boxes.conf.cpu().numpy()
            boxes, scores, keypoints, _ = pose(img, conf)
            row = {"frame": Path(frame).name, "conf": conf, "n_ultralytics": len(ref_conf), "n_ort": len(scores)}
            if len(ref_conf) == len(scores):  # both sorted by confidence
                row["max_conf_diff"] = float(np.abs(ref_conf - scores).max(initial=0))
                row["max_box_diff_px"] = float(np.abs(result.boxes.xyxy.cpu().numpy() - boxes).max(initial=0))
                row["max_kpt_diff_px"] = float(
                    np.abs(result.keypoints.data.cpu().numpy()[:, :, :2] - keypoints[:, :, :2]).max(initial=0))
            rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description="parity: our ORT pre/post-processing vs Ultralytics predict on .pt")
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--pt", required=True, help="the .pt weights")
    parser.add_argument("--imgsz", type=int, required=True, help="R: imgsz given to Ultralytics predict")
    parser.add_argument("--frames", nargs="+", required=True)
    parser.add_argument("--conf", type=float, nargs="+", default=[0.1, 0.25])
    parser.add_argument("--nms-free", action="store_true", help="EXP-017: reference = Ultralytics predict(nms=False)")
    args = parser.parse_args()
    pose = OrtPose(args.onnx, threads=2)
    print(f"onnx input shape (h, w) = {pose.shape}")
    for row in compare_with_ultralytics(args.pt, args.frames, pose, args.conf, args.imgsz, args.nms_free):
        print(row)


if __name__ == "__main__":
    main()
