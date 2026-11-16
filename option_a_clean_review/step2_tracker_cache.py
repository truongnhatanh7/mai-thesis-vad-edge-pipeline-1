"""STEP 2: Tracker, Cross-Frame Cache (`C1`), Queue Guard & Online Window Buffer.

Read this file second (~160 lines). It contains the three systems mechanisms of Option A:
  1. Two-Stage `BoT-SORT` Tracker (`build_botsort_tracker`):
     - Default (`two_stage=False`): filters out all boxes < 0.25 BEFORE tracking -> breaks tracks on brief blurs.
     - Locked (`two_stage=True`): passes boxes in [0.05, 0.20) into BoT-SORT's second stage to continue existing
       tracks without spawning false new tracks (+3.04 causal AUC on ShanghaiTech at 0 ms extra detector cost).
  2. Cross-Frame Static-Validity Detector Cache (`StaticValidityCacheC1`):
     - Checks a 40x24 grayscale thumbnail (~1.7 ms on Pi 5). If active person ROIs and borders changed by <= threshold
       and cache age < 3 frames, reuses the previous frame's detections and skips the ~48.9 ms NCNN detector call.
  3. Hysteresis Queue-Admission Guard (`HysteresisQueueGuard`):
     - Prevents unbounded FIFO queue collapse on the Raspberry Pi 5 by shedding alternate frames when queue wait > 400 ms
       and resuming full rate as soon as queue wait drains below 150 ms.
"""

from __future__ import annotations

from collections import deque

import cv2
import numpy as np
import torch

from step1_models import normalize_window_n0


# =====================================================================
# PART A: BoT-SORT Multi-Object Tracker (Default vs. Two-Stage Locked)
# =====================================================================
class DetBoxesWrapper:
    """Minimal Ultralytics Boxes-compatible wrapper required by BoT-SORT's `.update()`."""

    def __init__(self, xyxy: np.ndarray, conf: np.ndarray):
        self.xyxy = xyxy
        self.conf = conf
        self.cls = np.zeros(len(conf), dtype=np.float32)
        cx = 0.5 * (xyxy[:, 0] + xyxy[:, 2])
        cy = 0.5 * (xyxy[:, 1] + xyxy[:, 3])
        w = xyxy[:, 2] - xyxy[:, 0]
        h = xyxy[:, 3] - xyxy[:, 1]
        self.xywh = np.stack([cx, cy, w, h], axis=1) if len(conf) else np.zeros((0, 4), dtype=np.float32)

    def __len__(self) -> int:
        return len(self.conf)

    def __getitem__(self, idx) -> "DetBoxesWrapper":
        return DetBoxesWrapper(self.xyxy[idx], self.conf[idx])


def build_botsort_tracker(two_stage: bool = True):
    """Build BoT-SORT tracker:
    - `two_stage=False`: E00/Huang default (`c=0.25`, no low-conf second stage, `match_thresh=0.80`)
    - `two_stage=True`:  E01 dev-selected lock (`c=0.20`, `track_low_thresh=0.05`, `match_thresh=0.85`)
    """
    from ultralytics.trackers.track import TRACKER_MAP
    from ultralytics.utils import ROOT, YAML, IterableSimpleNamespace

    cfg = YAML.load(ROOT / "cfg/trackers/botsort.yaml")
    if two_stage:
        cfg.update(
            track_high_thresh=0.20,
            new_track_thresh=0.20,
            track_low_thresh=0.05,
            track_buffer=30,
            match_thresh=0.85,
            fuse_score=True,
            gmc_method="none",
            with_reid=False,
        )
    else:
        cfg.update(
            track_high_thresh=0.25,
            new_track_thresh=0.25,
            track_low_thresh=0.25,
            track_buffer=30,
            match_thresh=0.80,
            fuse_score=True,
            gmc_method="none",
            with_reid=False,
        )
    return TRACKER_MAP[cfg["tracker_type"]](IterableSimpleNamespace(**cfg))


def step_tracker(
    tracker,
    boxes: np.ndarray,
    scores: np.ndarray,
    keypoints: np.ndarray,
    img_bgr: np.ndarray | None = None,
    two_stage: bool = True,
) -> dict[int, np.ndarray]:
    """Feed frame detections into BoT-SORT and match output track boxes back to 17-joint keypoints via IoU >= 0.3."""
    min_conf = 0.05 if two_stage else 0.25
    keep = scores >= min_conf
    b_in, s_in, k_in = boxes[keep], scores[keep], keypoints[keep]
    tracks_out = tracker.update(DetBoxesWrapper(b_in, s_in), img_bgr)
    if len(tracks_out) == 0 or len(b_in) == 0:
        return {}

    matched: dict[int, np.ndarray] = {}
    for row in tracks_out:
        tx1, ty1, tx2, ty2, tid = float(row[0]), float(row[1]), float(row[2]), float(row[3]), int(row[4])
        ix1, iy1 = np.maximum(tx1, b_in[:, 0]), np.maximum(ty1, b_in[:, 1])
        ix2, iy2 = np.minimum(tx2, b_in[:, 2]), np.minimum(ty2, b_in[:, 3])
        inter = np.maximum(0.0, ix2 - ix1) * np.maximum(0.0, iy2 - iy1)
        a_t = max(1e-6, (tx2 - tx1) * (ty2 - ty1))
        a_b = np.maximum(1e-6, (b_in[:, 2] - b_in[:, 0]) * (b_in[:, 3] - b_in[:, 1]))
        iou = inter / (a_t + a_b - inter + 1e-6)
        best = int(np.argmax(iou))
        if iou[best] >= 0.3:
            matched[tid] = k_in[best]
    return matched


# =====================================================================
# PART B: Cross-Frame Static-Validity Detector Cache (`C1`)
# =====================================================================
class StaticValidityCacheC1:
    """Skips redundant detector calls when active person ROIs and scene borders are static (up to max_age=3 frames)."""

    def __init__(self, roi_thresh: float = 0.0225, border_thresh: float = 0.0350, max_age: int = 3, refresh_period: int = 12):
        self.roi_thresh = roi_thresh
        self.border_thresh = border_thresh
        self.max_age = max_age
        self.refresh_period = refresh_period
        self.prev_thumb: np.ndarray | None = None
        self.cached_output: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
        self.age = 999

    def reset(self) -> None:
        self.prev_thumb = None
        self.cached_output = None
        self.age = 999

    def check_or_compute(self, frame_idx: int, img_bgr: np.ndarray, detector_fn) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
        """Return (boxes, scores, keypoints, was_reused_from_cache)."""
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        cur_thumb = cv2.resize(gray, (40, 24), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0

        can_reuse = False
        if (
            self.prev_thumb is not None
            and self.cached_output is not None
            and self.age < self.max_age
            and (frame_idx % self.refresh_period != 0)
        ):
            diff = np.abs(cur_thumb - self.prev_thumb)
            border_mask = np.zeros_like(diff, dtype=bool)
            border_mask[:3, :] = border_mask[-3:, :] = border_mask[:, :4] = border_mask[:, -4:] = True
            border_p95 = float(np.percentile(diff[border_mask], 95))

            h_img, w_img = img_bgr.shape[:2]
            roi_max = 0.0
            for box in self.cached_output[0]:
                x1 = max(0, min(39, int(box[0] / w_img * 40)))
                y1 = max(0, min(23, int(box[1] / h_img * 24)))
                x2 = max(x1 + 1, min(40, int(np.ceil(box[2] / w_img * 40))))
                y2 = max(y1 + 1, min(24, int(np.ceil(box[3] / h_img * 24))))
                roi_max = max(roi_max, float(diff[y1:y2, x1:x2].mean()))

            if roi_max <= self.roi_thresh and border_p95 <= self.border_thresh:
                can_reuse = True

        self.prev_thumb = cur_thumb
        if can_reuse and self.cached_output is not None:
            self.age += 1
            boxes, scores, kps = self.cached_output
            return boxes, scores, kps, True

        boxes, scores, kps, _ = detector_fn(img_bgr)
        self.cached_output = (boxes, scores, kps)
        self.age = 0
        return boxes, scores, kps, False


# =====================================================================
# PART C: Hysteresis Queue Guard & Online 24-Frame Window Scorer
# =====================================================================
class HysteresisQueueGuard:
    """Activates load-shedding when FIFO queue wait > high_ms (400 ms) and deactivates when wait < low_ms (150 ms)."""

    def __init__(self, high_ms: float = 400.0, low_ms: float = 150.0):
        self.high_ms = high_ms
        self.low_ms = low_ms
        self.active = False

    def should_shed(self, frame_idx: int, queue_wait_ms: float) -> bool:
        if not self.active and queue_wait_ms > self.high_ms:
            self.active = True
        elif self.active and queue_wait_ms < self.low_ms:
            self.active = False
        return self.active and (frame_idx % 2 == 1)


class OnlineCircularWindowScorer:
    """Maintains per-track 24-pose circular buffers (reusing per-frame poses across overlapping windows bit-exactly)
    and scores all newly completed 24-pose windows (missing <= 2 frames) in a single batched forward pass.
    """

    def __init__(self, flow_model: torch.nn.Module, seg_len: int = 24, max_missing: int = 2):
        self.flow = flow_model.eval()
        self.seg_len = seg_len
        self.max_missing = max_missing
        self.buffers: dict[int, deque] = {}

    def reset(self) -> None:
        self.buffers.clear()

    def step(self, frame_idx: int, tracked_poses: dict[int, np.ndarray]) -> np.ndarray:
        """Append frame's tracked poses and return normality scores (-NLL) for all windows completing on `frame_idx`."""
        ready_windows = []
        for tid, kp17 in tracked_poses.items():
            buf = self.buffers.setdefault(tid, deque(maxlen=self.seg_len))
            buf.append((frame_idx, kp17[:, :2]))
            if len(buf) == self.seg_len:
                f_span = buf[-1][0] - buf[0][0] + 1
                if f_span <= self.seg_len + self.max_missing:
                    ready_windows.append(np.stack([xy for _, xy in buf], axis=0))

        if not ready_windows:
            return np.zeros(0, dtype=np.float64)

        x = normalize_window_n0(np.stack(ready_windows, axis=0))
        with torch.no_grad():
            try:
                _, nll = self.flow(x, label=torch.ones(len(x)))
            except TypeError:
                _, nll = self.flow(x)
        return (-nll).cpu().numpy().reshape(-1).astype(np.float64)
