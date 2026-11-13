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

