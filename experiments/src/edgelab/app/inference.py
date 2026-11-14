"""Live inference and visualization helpers for the Streamlit explorer."""

from collections import deque
import time
from pathlib import Path
import cv2
import numpy as np
import torch

from edgelab.app.catalog import COCO17_EDGES, DATA_ROOT, ModelSpec
from edgelab.pose.replay import Detections, build_tracker
from edgelab.pose.stgnf_eval import SEG_LENS, _import_stgnf
from edgelab.preprocess.darken import darken_level, mean_luma
from edgelab.preprocess.enhance import ENHANCERS

# Distinct BGR palette for person track IDs / bounding boxes
PALETTE_BGR = (
    (80, 220, 100),   # vibrant green
    (255, 180, 50),   # sky blue
    (80, 120, 255),   # coral orange
    (220, 100, 240),  # magenta
    (60, 230, 240),   # yellow
    (255, 220, 120),  # cyan
    (130, 90, 255),   # rose
    (180, 255, 120),  # mint
)


def apply_lighting_and_enhancer(img_bgr: np.ndarray, video_id: str, frame_idx: int,
                                level: str = "L0", enhance: str = "E0") -> tuple[np.ndarray, np.ndarray, dict]:
    """Return `(darkened_bgr, enhanced_bgr, stats_dict)` using the exact EXP-003 sensor darkening & enhancer functions."""
    t0 = time.perf_counter()
    dark = darken_level(img_bgr, level, video_id, frame_idx)
    t1 = time.perf_counter()
    enhance_fn = ENHANCERS[enhance]
    enhanced = enhance_fn(dark)
    t2 = time.perf_counter()
    return dark, enhanced, {
        "luma_orig": mean_luma(img_bgr),
        "luma_dark": mean_luma(dark),
        "luma_enhanced": mean_luma(enhanced),
        "darken_ms": (t1 - t0) * 1e3,
        "enhance_ms": (t2 - t1) * 1e3,
    }


def run_live_pose_model(spec: ModelSpec, img_bgr: np.ndarray, conf_thresh: float = 0.25,
                        device: str = "cpu", loaded_handle=None) -> dict:
    """Run live single-frame inference with the chosen trained model (.pt, .onnx, NCNN, or Top-Down RTMPose)."""
    if spec.backend == "released" or spec.weights_path is None:
        return {"boxes": np.zeros((0, 4)), "conf": np.zeros(0), "keypoints": np.zeros((0, 17, 3)), "speed": {}}

    if spec.backend == "ort":
        detector = loaded_handle
        boxes, scores, keypoints, speed = detector(img_bgr, conf_thresh)
        return {"boxes": boxes, "conf": scores, "keypoints": keypoints, "speed": speed}

    if spec.backend == "ncnn":
        detector = loaded_handle
        boxes, scores, keypoints, speed = detector(img_bgr, conf_thresh)
        return {"boxes": boxes, "conf": scores, "keypoints": keypoints, "speed": speed}

    if spec.backend == "topdown":
        yolo_model, rtmpose_model = loaded_handle
        t0 = time.perf_counter()
        res = yolo_model.predict(img_bgr, imgsz=spec.imgsz, conf=conf_thresh, iou=0.7,
                                 classes=[0], device=device, verbose=False)[0]
        boxes = res.boxes.xyxy.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()
        t1 = time.perf_counter()
        if len(boxes) > 0:
            keypoints = rtmpose_model.predict(img_bgr, boxes)
        else:
            keypoints = np.zeros((0, 17, 3), dtype=np.float32)
        t2 = time.perf_counter()
        return {
            "boxes": boxes,
            "conf": confs,
            "keypoints": keypoints,
            "speed": {
                "pre_ms": res.speed.get("preprocess", 0.0),
                "infer_ms": (t1 - t0) * 1e3,
                "post_ms": (t2 - t1) * 1e3,
            },
        }

    # Default: Ultralytics YOLO pose (.pt)
    model = loaded_handle
    extra = {"nms": False} if spec.nms_free else {}
    res = model.predict(img_bgr, imgsz=spec.imgsz, conf=conf_thresh, iou=0.7,
                        device=device, verbose=False, **extra)[0]
    boxes = res.boxes.xyxy.cpu().numpy() if res.boxes is not None else np.zeros((0, 4))
    confs = res.boxes.conf.cpu().numpy() if res.boxes is not None else np.zeros(0)
    if res.keypoints is not None and len(res.keypoints.data) > 0:
        keypoints = res.keypoints.data.cpu().numpy().reshape(-1, 17, 3)
    else:
        keypoints = np.zeros((0, 17, 3), dtype=np.float32)
    return {
        "boxes": boxes,
        "conf": confs,
        "keypoints": keypoints,
        "speed": {
            "pre_ms": res.speed.get("preprocess", 0.0),
            "infer_ms": res.speed.get("inference", 0.0),
            "post_ms": res.speed.get("postprocess", 0.0),
        },
    }


def load_model_handle(spec: ModelSpec):
    """Instantiate the underlying detector/pose model for live inference."""
    if spec.backend == "released" or spec.weights_path is None:
        return None
    if spec.backend == "ort":
        from edgelab.pipeline.ort_pose import OrtPose
        return OrtPose(spec.weights_path, threads=2)
    if spec.backend == "ncnn":
        from edgelab.pipeline.ncnn_pose import NcnnPose
        return NcnnPose(spec.weights_path, threads=2, fp16=True)
    if spec.backend == "topdown":
        from ultralytics import YOLO
        from edgelab.pose.rtmpose import RtmPose
        yolo_det = YOLO(str(spec.weights_path))
        rtm_onnx = DATA_ROOT / "exp015/models/rtmpose-s-body7-256x192.onnx"
        rtm = RtmPose(rtm_onnx, threads=2)
        return (yolo_det, rtm)
    from ultralytics import YOLO
    return YOLO(str(spec.weights_path))


def draw_poses_and_boxes(
    img_bgr: np.ndarray,
    boxes: np.ndarray | None = None,
    confs: np.ndarray | None = None,
    keypoints: np.ndarray | None = None,
    track_ids: list[int] | None = None,
    ref_poses: list[dict] | None = None,
    kpt_conf_thresh: float = 0.2,
    show_boxes: bool = True,
    show_skeleton: bool = True,
    show_height_px: bool = True,
) -> np.ndarray:
    """Render bounding boxes, COCO-17 skeletons, person height (px), and optional reference poses onto a copy of `img_bgr`."""
    canvas = img_bgr.copy()

    # Optional: draw reference AlphaPose skeletons in subtle cyan/white dashed look first
    if ref_poses:
        for p in ref_poses:
            rk = p["keypoints"]
            vis = rk[rk[:, 2] > 0.05]
            if len(vis) > 0:
                x1, y1, x2, y2 = int(vis[:, 0].min()), int(vis[:, 1].min()), int(vis[:, 0].max()), int(vis[:, 1].max())
                cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 255, 0), 1, cv2.LINE_AA)
            for i, j in COCO17_EDGES:
                if rk[i, 2] > 0.05 and rk[j, 2] > 0.05:
                    pt1 = (int(round(rk[i, 0])), int(round(rk[i, 1])))
                    pt2 = (int(round(rk[j, 0])), int(round(rk[j, 1])))
                    cv2.line(canvas, pt1, pt2, (255, 255, 0), 1, cv2.LINE_AA)

    n_people = len(keypoints) if keypoints is not None else (len(boxes) if boxes is not None else 0)
    for idx in range(n_people):
        tid = track_ids[idx] if track_ids is not None and idx < len(track_ids) else idx
        color = PALETTE_BGR[tid % len(PALETTE_BGR)]
        kpts = keypoints[idx] if keypoints is not None and idx < len(keypoints) else None
        box = boxes[idx] if boxes is not None and idx < len(boxes) else None
        score = float(confs[idx]) if confs is not None and idx < len(confs) else None

        # Derive box from keypoints if tracked pose JSON (which has keypoints only) is used
        if box is None and kpts is not None:
            vis = kpts[kpts[:, 2] > 0.05]
            if len(vis) > 0:
                box = np.array([vis[:, 0].min(), vis[:, 1].min(), vis[:, 0].max(), vis[:, 1].max()])

        if show_boxes and box is not None:
            x1, y1, x2, y2 = (int(round(v)) for v in box)
            h_px = max(0, y2 - y1)
            # Highlight small persons (<64px) with a distinct border
            box_color = (50, 200, 255) if h_px < 64 else color
            cv2.rectangle(canvas, (x1, y1), (x2, y2), box_color, 2, cv2.LINE_AA)

            label_parts = []
            if track_ids is not None:
                label_parts.append(f"ID:{tid}")
            if score is not None:
                label_parts.append(f"{score:.2f}")
            if show_height_px:
                label_parts.append(f"{h_px}px")
            label = " ".join(label_parts)
            if label:
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                ly = max(y1 - 4, th + 4)
                cv2.rectangle(canvas, (x1, ly - th - 4), (x1 + tw + 6, ly + 2), box_color, -1)
                cv2.putText(canvas, label, (x1 + 3, ly - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (15, 15, 15), 1, cv2.LINE_AA)

        if show_skeleton and kpts is not None:
            for i, j in COCO17_EDGES:
                if kpts[i, 2] >= kpt_conf_thresh and kpts[j, 2] >= kpt_conf_thresh:
                    pt1 = (int(round(kpts[i, 0])), int(round(kpts[i, 1])))
                    pt2 = (int(round(kpts[j, 0])), int(round(kpts[j, 1])))
                    cv2.line(canvas, pt1, pt2, color, 2, cv2.LINE_AA)
            for k in range(len(kpts)):
                if kpts[k, 2] >= kpt_conf_thresh:
                    pt = (int(round(kpts[k, 0])), int(round(kpts[k, 1])))
                    cv2.circle(canvas, pt, 3, (255, 255, 255), -1, cv2.LINE_AA)
                    cv2.circle(canvas, pt, 2, color, -1, cv2.LINE_AA)

    return canvas


# ---------------------------------------------------------------- Height / confidence overlay (Dataset Explorer)

# (low px inclusive, high px exclusive, label, BGR colour): the person-height bins of EXP-011 / EXP-020 recall tables
HEIGHT_BINS = (
    (0.0, 32.0, "<32 px", (60, 60, 255)),       # red
    (32.0, 48.0, "32-48 px", (0, 140, 255)),    # orange
    (48.0, 64.0, "48-64 px", (0, 225, 255)),    # yellow
    (64.0, float("inf"), ">=64 px", (90, 210, 90)),  # green
)
HIGH_CONF_THRESH = 0.20  # two-stage BoT-SORT first association (locked tracker c = 0.20)
LOW_CONF_THRESH = 0.05   # second association ("low" boxes 0.05 <= conf < 0.20 recover fragmented tracks)


def height_bin_index(height_px: float) -> int:
    """Index into `HEIGHT_BINS` for a person height in pixels (`<32`, `32-48`, `48-64`, `>=64`)."""
    for i, (lo, hi, _, _) in enumerate(HEIGHT_BINS):
        if lo <= height_px < hi:
            return i
    return 0


def height_bin_label(height_px: float) -> str:
    return HEIGHT_BINS[height_bin_index(height_px)][2]


def confidence_class(conf: float, high: float = HIGH_CONF_THRESH, low: float = LOW_CONF_THRESH) -> str:
    """`high` (conf >= 0.20), `low` (0.05 <= conf < 0.20: two-stage recovery box) or `ignored` (< 0.05)."""
    if conf >= high:
        return "high"
    if conf >= low:
        return "low"
    return "ignored"


def summarize_detections(boxes: np.ndarray | None, confs: np.ndarray | None,
                         high: float = HIGH_CONF_THRESH, low: float = LOW_CONF_THRESH) -> dict:
    """Counts of detections per `(height bin label, confidence class)`; `ignored` boxes (< `low`) are not counted."""
    counts = {(b[2], c): 0 for b in HEIGHT_BINS for c in ("high", "low")}
    heights: list[float] = []
    if boxes is not None and confs is not None:
        for box, conf in zip(boxes, confs):
            cls = confidence_class(float(conf), high, low)
            if cls == "ignored":
                continue
            h = float(box[3] - box[1])
            heights.append(round(h, 1))
            counts[(height_bin_label(h), cls)] += 1
    return {
        "counts": counts,
        "n_high": sum(v for (_, c), v in counts.items() if c == "high"),
        "n_low": sum(v for (_, c), v in counts.items() if c == "low"),
        "n_small_lt48": sum(v for (b, _), v in counts.items() if b in ("<32 px", "32-48 px")),
        "heights": heights,
    }


def _draw_dashed_rect(img: np.ndarray, p1: tuple[int, int], p2: tuple[int, int], color, thickness: int = 1, dash: int = 6) -> None:
    (x1, y1), (x2, y2) = p1, p2
    for x in range(x1, x2, dash * 2):
        cv2.line(img, (x, y1), (min(x + dash, x2), y1), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x, y2), (min(x + dash, x2), y2), color, thickness, cv2.LINE_AA)
    for y in range(y1, y2, dash * 2):
        cv2.line(img, (x1, y), (x1, min(y + dash, y2)), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x2, y), (x2, min(y + dash, y2)), color, thickness, cv2.LINE_AA)


def draw_height_conf_overlay(
    img_bgr: np.ndarray,
    boxes: np.ndarray | None,
    confs: np.ndarray | None,
    keypoints: np.ndarray | None = None,
    kpt_conf_thresh: float = 0.2,
    show_boxes: bool = True,
    show_skeleton: bool = True,
    show_labels: bool = True,
    high_thresh: float = HIGH_CONF_THRESH,
    low_thresh: float = LOW_CONF_THRESH,
    title: str | None = None,
    show_legend: bool = True,
    gt_anomalous: bool | None = None,
    highlight_boxes: np.ndarray | None = None,
    highlight_label: str = "MISSED",
) -> np.ndarray:
    """Draw detections colour-coded by person height bin (`<32`, `32-48`, `48-64`, `>=64` px).

    Solid thick boxes/skeletons = high-confidence (`conf >= high_thresh`); dashed thin boxes tagged `LOW` = the 0.05..0.20 boxes that only the
    two-stage tracker's second association can use. Detections below `low_thresh` are not drawn.
    """
    canvas = img_bgr.copy()
    n = len(boxes) if boxes is not None else 0
    for i in range(n):
        conf = float(confs[i]) if confs is not None and i < len(confs) else 1.0
        cls = confidence_class(conf, high_thresh, low_thresh)
        if cls == "ignored":
            continue
        box = boxes[i]
        x1, y1, x2, y2 = (int(round(float(v))) for v in box)
        h_px = max(0, y2 - y1)
        color = HEIGHT_BINS[height_bin_index(h_px)][3]
        thick = 2 if cls == "high" else 1
        if show_boxes:
            if cls == "high":
                cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thick, cv2.LINE_AA)
            else:
                _draw_dashed_rect(canvas, (x1, y1), (x2, y2), color, thick)
            if show_labels:
                label = f"{conf:.2f} {h_px}px" + (" LOW" if cls == "low" else "")
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
                ly = max(y1 - 3, th + 3)
                cv2.rectangle(canvas, (x1, ly - th - 3), (x1 + tw + 4, ly + 2), color, -1)
                cv2.putText(canvas, label, (x1 + 2, ly - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (15, 15, 15), 1, cv2.LINE_AA)
        if show_skeleton and keypoints is not None and i < len(keypoints):
            k = keypoints[i]
            for a, b in COCO17_EDGES:
                if k[a, 2] >= kpt_conf_thresh and k[b, 2] >= kpt_conf_thresh:
                    cv2.line(canvas, (int(round(k[a, 0])), int(round(k[a, 1]))), (int(round(k[b, 0])), int(round(k[b, 1]))),
                             color, thick, cv2.LINE_AA)
            for j in range(len(k)):
                if k[j, 2] >= kpt_conf_thresh:
                    cv2.circle(canvas, (int(round(k[j, 0])), int(round(k[j, 1]))), 2 if cls == "high" else 1, (255, 255, 255), -1, cv2.LINE_AA)

    if show_legend:
        pad, row_h = 6, 16
        entries = [(b[2], b[3]) for b in HEIGHT_BINS]
        w_leg, h_leg = 128, pad * 2 + row_h * (len(entries) + 2)
        overlay = canvas.copy()
        cv2.rectangle(overlay, (4, 4), (4 + w_leg, 4 + h_leg), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.6, canvas, 0.4, 0, canvas)
        for r, (name, col) in enumerate(entries):
            y = 4 + pad + r * row_h + 10
            cv2.rectangle(canvas, (10, y - 8), (24, y + 2), col, -1)
            cv2.putText(canvas, name, (30, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (240, 240, 240), 1, cv2.LINE_AA)
        y = 4 + pad + len(entries) * row_h + 10
        cv2.line(canvas, (10, y - 3), (24, y - 3), (240, 240, 240), 2, cv2.LINE_AA)
        cv2.putText(canvas, f"conf>={high_thresh:.2f}", (30, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (240, 240, 240), 1, cv2.LINE_AA)
        _draw_dashed_rect(canvas, (10, y + row_h - 7), (24, y + row_h - 1), (240, 240, 240), 1, dash=3)
        cv2.putText(canvas, f"{low_thresh:.2f}..{high_thresh:.2f}", (30, y + row_h), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (240, 240, 240), 1, cv2.LINE_AA)

    if highlight_boxes is not None:
        for k, box in enumerate(np.asarray(highlight_boxes).reshape(-1, 4)):
            x1, y1, x2, y2 = (int(round(float(v))) for v in box)
            cv2.rectangle(canvas, (x1 - 4, y1 - 4), (x2 + 4, y2 + 4), (255, 255, 255), 1, cv2.LINE_AA)
            if k == 0:
                cv2.putText(canvas, highlight_label, (x1 - 4, y2 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

    if gt_anomalous:
        cv2.rectangle(canvas, (0, 0), (canvas.shape[1] - 1, canvas.shape[0] - 1), (60, 60, 255), 3)
    if title:
        (tw, th), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        y0 = canvas.shape[0] - 8
        cv2.rectangle(canvas, (4, y0 - th - 6), (4 + tw + 8, y0 + 4), (20, 20, 20), -1)
        cv2.putText(canvas, title, (8, y0 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1, cv2.LINE_AA)
    return canvas


class LivePipelineSession:
    """Stateful live video processor: runs live pose detection -> BoT-SORT tracking -> online STG-NF window scoring."""

    def __init__(self, spec: ModelSpec, dataset: str, conf_thresh: float = 0.25,
                 device: str = "cpu", loaded_handle=None, stgnf_model=None):
        self.spec = spec
        self.dataset = dataset
        self.conf_thresh = conf_thresh
        self.device = device
        self.loaded_handle = loaded_handle
        self.stgnf_model = stgnf_model
        self.seg_len = SEG_LENS.get(dataset, 24)
        self.tracker = build_tracker(conf_thresh, "botsort")
        self.buffers: dict[int, deque] = {}
        self.seq_idx = 0

    def step(self, frame_idx: int, img_bgr: np.ndarray, kpt_thresh: float = 0.2,
             show_boxes: bool = True, show_skeleton: bool = True,
             ref_poses_frame: list[dict] | None = None) -> tuple[np.ndarray, dict]:
        """Run one live frame: detect -> track -> score completed windows -> draw."""
        t0 = time.perf_counter()
        det = run_live_pose_model(
            self.spec, img_bgr, conf_thresh=self.conf_thresh,
            device=self.device, loaded_handle=self.loaded_handle,
        )
        t_det = (time.perf_counter() - t0) * 1e3
        boxes, confs, kpts = det["boxes"], det["conf"], det["keypoints"]

        t1 = time.perf_counter()
        tracks = self.tracker.update(
            Detections(boxes, confs, np.zeros(len(confs), dtype=np.float32)), None
        )
        t_track = (time.perf_counter() - t1) * 1e3

        t_boxes, t_confs, t_kpts, t_ids = [], [], [], []
        ready_segments = []
        for row in tracks:
            idx = int(row[7])
            tid = int(row[4])
            if idx < len(kpts):
                t_boxes.append(row[:4])
                t_confs.append(float(row[5]))
                t_kpts.append(kpts[idx])
                t_ids.append(tid)
                buf = self.buffers.setdefault(tid, deque(maxlen=self.seg_len))
                buf.append((self.seq_idx, kpts[idx]))
                if len(buf) == self.seg_len:
                    frames_in_buf = [f for f, _ in buf]
                    if frames_in_buf[-1] - frames_in_buf[0] <= self.seg_len + 2:
                        ready_segments.append(np.stack([k for _, k in buf]))

        self.seq_idx += 1
        online_score = None
        if ready_segments and self.stgnf_model is not None:
            _import_stgnf()
            from dataset import keypoints17_to_coco18
            from utils.data_utils import normalize_pose

            seg_arr = np.stack(ready_segments).astype(np.float32)
            x = normalize_pose(keypoints17_to_coco18(seg_arr).astype(np.float32))
            x_t = torch.from_numpy(np.ascontiguousarray(x.transpose(0, 3, 1, 2)[:, :2])).float()
            prov = torch.ones((len(x_t), self.seg_len), dtype=torch.float32)
            with torch.no_grad():
                _, nll = self.stgnf_model(x_t, prov)
            normality = -nll.numpy().reshape(-1)
            online_score = float((-normality).max())  # most anomalous person on this frame

        boxes_arr = np.array(t_boxes, dtype=np.float32).reshape(-1, 4) if t_boxes else boxes
        confs_arr = np.array(t_confs, dtype=np.float32) if t_confs else confs
        kpts_arr = np.array(t_kpts, dtype=np.float32).reshape(-1, 17, 3) if t_kpts else kpts
        ids_list = t_ids if t_ids else None

        heights = [round(float(b[3] - b[1]), 1) for b in boxes_arr] if len(boxes_arr) else []
        annotated_bgr = draw_poses_and_boxes(
            img_bgr,
            boxes=boxes_arr,
            confs=confs_arr,
            keypoints=kpts_arr,
            track_ids=ids_list,
            ref_poses=ref_poses_frame,
            kpt_conf_thresh=kpt_thresh,
            show_boxes=show_boxes,
            show_skeleton=show_skeleton,
            show_height_px=show_boxes,
        )
        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
        return annotated_rgb, {
            "frame_idx": frame_idx,
            "num_persons": len(kpts_arr),
            "num_tracks": len(t_ids),
            "small_persons_lt64": sum(1 for h in heights if 0 < h < 64),
            "heights_px": heights,
            "mean_conf": float(np.mean(confs_arr)) if len(confs_arr) else 0.0,
            "det_ms": t_det,
            "track_ms": t_track,
            "total_ms": t_det + t_track,
            "online_score": online_score,
        }

