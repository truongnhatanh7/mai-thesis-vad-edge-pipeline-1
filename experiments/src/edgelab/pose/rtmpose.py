"""RTMPose (SimCC head, official OpenMMLab ONNX export) on person crops: crop -> network -> keypoints in image pixels.

Steps, copied from the model's own `pipeline.json` (rtmpose-*_simcc-body7_pt-body7_420e-256x192):
  1. box -> centre and scale: scale = box size * 1.25, then the box is widened or heightened to the 192:256 aspect;
  2. crop the scaled box out of the image with one affine warp to 192 wide x 256 high (outside the image = black);
  3. RGB, subtract mean [123.675, 116.28, 103.53], divide by std [58.395, 57.12, 57.375];
  4. network -> simcc_x (17, 384), simcc_y (17, 512): one score per half-pixel position on each axis;
  5. keypoint = argmax / 2 in the crop, mapped back to the image; score = min(max of x row, max of y row).

Output order is COCO-17, the same order as the released AlphaPose files (checked in EXP-015).
Models: Apache-2.0 (OpenMMLab MMPose). No dependency on rtmlib or mmpose: the whole thing is this file.
"""

from pathlib import Path

import cv2
import numpy as np

INPUT_W, INPUT_H = 192, 256
PADDING = 1.25
SIMCC_SPLIT = 2.0
MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)


def box_to_center_scale(box_xyxy, padding: float = PADDING) -> tuple[np.ndarray, np.ndarray]:
    """xyxy box -> (centre (2,), scale (2,) = width, height of the region to crop, aspect INPUT_W:INPUT_H)."""
    x1, y1, x2, y2 = (float(v) for v in box_xyxy)
    center = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
    width, height = x2 - x1, y2 - y1
    aspect = INPUT_W / INPUT_H
    if width > height * aspect:  # too wide: make it taller
        height = width / aspect
    else:  # too tall: make it wider
        width = height * aspect
    return center, np.array([width, height]) * padding


def keypoints_to_box(keypoints: np.ndarray) -> np.ndarray | None:
    """(17, 3) x, y, conf -> xyxy box over the keypoints with conf > 0, or None when there is none."""
    seen = keypoints[keypoints[:, 2] > 0]
    if len(seen) == 0:
        return None
    return np.array([seen[:, 0].min(), seen[:, 1].min(), seen[:, 0].max(), seen[:, 1].max()])


def crop_matrix(center: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """2x3 affine matrix: image pixel -> crop pixel, mapping the scaled box onto the whole INPUT_W x INPUT_H crop."""
    sx, sy = INPUT_W / scale[0], INPUT_H / scale[1]
    return np.array([[sx, 0, -(center[0] - scale[0] / 2) * sx],
                     [0, sy, -(center[1] - scale[1] / 2) * sy]], dtype=np.float64)


def crop_person(img_bgr: np.ndarray, center: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """(INPUT_H, INPUT_W, 3) uint8 BGR crop."""
    return cv2.warpAffine(img_bgr, crop_matrix(center, scale), (INPUT_W, INPUT_H), flags=cv2.INTER_LINEAR)


def to_network_input(crops_bgr: list[np.ndarray]) -> np.ndarray:
    """List of BGR crops -> (n, 3, INPUT_H, INPUT_W) float32, normalised, RGB."""
    batch = np.stack(crops_bgr)[:, :, :, ::-1].astype(np.float32)  # BGR -> RGB
    return np.ascontiguousarray(((batch - MEAN) / STD).transpose(0, 3, 1, 2))


def decode_simcc(simcc_x: np.ndarray, simcc_y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(n, 17, 384), (n, 17, 512) -> keypoints (n, 17, 2) in crop pixels, scores (n, 17)."""
    x_pos, y_pos = simcc_x.argmax(axis=2), simcc_y.argmax(axis=2)
    x_val, y_val = simcc_x.max(axis=2), simcc_y.max(axis=2)
    keypoints = np.stack([x_pos, y_pos], axis=2).astype(np.float32) / SIMCC_SPLIT
    return keypoints, np.minimum(x_val, y_val)


def crop_to_image(keypoints_crop: np.ndarray, centers: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """(n, 17, 2) crop pixels -> image pixels; centers and scales are (n, 2)."""
    size = np.array([INPUT_W, INPUT_H], dtype=np.float32)
    return keypoints_crop / size * scales[:, None, :] + centers[:, None, :] - scales[:, None, :] / 2


class RtmPose:
    """One ONNX Runtime session of an RTMPose model. `predict` takes an image and xyxy boxes, returns (n, 17, 3)."""

    def __init__(self, onnx_path: str | Path, threads: int = 0, batch_size: int = 32):
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(onnx_path), options, providers=["CPUExecutionProvider"])
        self.batch_size = batch_size

    def run_crops(self, crops_bgr: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        """-> keypoints in crop pixels (n, 17, 2), scores (n, 17)."""
        points, scores = [], []
        for start in range(0, len(crops_bgr), self.batch_size):
            batch = to_network_input(crops_bgr[start:start + self.batch_size])
            simcc_x, simcc_y = self.session.run(None, {"input": batch})
            p, s = decode_simcc(simcc_x, simcc_y)
            points.append(p)
            scores.append(s)
        if not points:
            return np.zeros((0, 17, 2), np.float32), np.zeros((0, 17), np.float32)
        return np.concatenate(points), np.concatenate(scores)

    def predict(self, img_bgr: np.ndarray, boxes_xyxy: np.ndarray) -> np.ndarray:
        """-> (n, 17, 3): x, y in image pixels and the SimCC score (not limited to 0..1)."""
        if len(boxes_xyxy) == 0:
            return np.zeros((0, 17, 3), np.float32)
        centers, scales = zip(*(box_to_center_scale(b) for b in boxes_xyxy))
        centers, scales = np.array(centers), np.array(scales)
        crops = [crop_person(img_bgr, c, s) for c, s in zip(centers, scales)]
        points, scores = self.run_crops(crops)
        return np.concatenate([crop_to_image(points, centers, scales), scores[:, :, None]], axis=2).astype(np.float32)
