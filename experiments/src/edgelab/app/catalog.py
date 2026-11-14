"""Catalog of datasets, video examples, trained models, and precomputed experiment artifacts."""

from dataclasses import dataclass, field
import functools
import json
from pathlib import Path
import re
import cv2
import numpy as np
import pandas as pd

from edgelab.pose.detcache import load_detections
from edgelab.pose.evaluate import CHECKPOINT as SHT_CHECKPOINT, GT_DIR as SHT_GT_DIR
from edgelab.pose.fill import merge_parts
from edgelab.pose.stgnf_eval import EXPERIMENTS_ROOT, STGNF_ROOT
from edgelab.pose import ubnormal

DATA_ROOT = Path.home() / "thesis-data"
SHT_FRAMES_ROOT = DATA_ROOT / "shanghaitech/shanghaitech/testing/frames"
UB_VIDEOS_ROOT = DATA_ROOT / "ubnormal/test_videos"

# COCO-17 skeleton edges (0-indexed)
COCO17_NAMES = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
)

COCO17_EDGES = (
    (0, 1), (0, 2), (1, 3), (2, 4),          # head
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10), # arms & shoulders
    (5, 11), (6, 12), (11, 12),              # torso
    (11, 13), (13, 15), (12, 14), (14, 16),  # legs
)


@dataclass(frozen=True)
class ModelSpec:
    id: str
    name: str
    category: str
    experiment: str
    description: str
    weights_path: Path | None = None
    backend: str = "ultralytics"  # "ultralytics" | "ort" | "ncnn" | "topdown" | "released"
    imgsz: int = 640
    nms_free: bool = False
    sht_root: Path | None = None
    ub_root: Path | None = None
    sht_cache_prefix: str = "L0_E0"
    ub_cache_prefix: str = "L0_E0"
    pose_dir_override: dict[str, Path] = field(default_factory=dict)
    scores_dir_override: dict[str, Path] = field(default_factory=dict)
    cache_dir_override: dict[str, Path] = field(default_factory=dict)


def build_model_catalog() -> list[ModelSpec]:
    """All trained/evaluated pose front-end models across the thesis experiments."""
    a = EXPERIMENTS_ROOT / "artifacts"
    d = DATA_ROOT
    vm_models = d / "vm-exchange/models"

    return [
        ModelSpec(
            id="stock-yolov8n-640",
            name="YOLOv8n-pose @640 (Stock Baseline)",
            category="Baseline",
            experiment="EXP-001 / EXP-003 / EXP-006",
            description="Stock COCO-pretrained YOLOv8n-pose at 640px. Exhibits severe recall cliff (<1% below 48px) because COCO converter drops keypoint-less small people.",
            weights_path=a / "EXP-001/models/yolov8n-pose.pt",
            backend="ultralytics",
            imgsz=640,
            sht_root=d / "exp003",
            ub_root=d / "exp006",
        ),
        ModelSpec(
            id="exp020-ML0",
            name="EXP-020 ML0: Partial-Label YOLOv8n-pose @640 (Practical Winner)",
            category="EXP-020 (COCO 2x2 Ablation)",
            experiment="EXP-020",
            description="Trained on COCO with box-only small persons retained in detection loss (partial labels), stock head @640. +16.24 AUC over control M00 on ShanghaiTech, +5.57 on UBnormal at zero extra latency.",
            weights_path=a / "EXP-020/ML0.pt",
            backend="ultralytics",
            imgsz=640,
            sht_root=d / "exp020-ML0-sht",
            ub_root=d / "exp020-ML0-ub",
        ),
        ModelSpec(
            id="exp020-ML0-ncnn",
            name="EXP-020 ML0: NCNN FP16 @640 (384x640 export, Q2-P1 locked front-end)",
            category="EXP-020 (COCO 2x2 Ablation)",
            experiment="EXP-020 / Q2-P1-E01..E05",
            description="Same ML0 weights exported to NCNN FP16 (384x640 rect). This is the detector replayed by the Q2-P1 experiments on ShanghaiTech (UBnormal uses the `ML0_mps` proxy cache).",
            weights_path=(vm_models / "exp020/ML0-640-384x640-ncnn") if (vm_models / "exp020/ML0-640-384x640-ncnn").exists() else None,
            backend="ncnn",
            imgsz=640,
            sht_root=d / "exp020-ncnn",
            sht_cache_prefix="L0_E0-ML0",
            ub_root=d / "exp020-ML0-ub",
        ),
        ModelSpec(
            id="exp020-MLP",
            name="EXP-020 MLP: Partial-Label + P2 Head YOLOv8n-pose @640 (Best Offline AUC)",
            category="EXP-020 (COCO 2x2 Ablation)",
            experiment="EXP-020",
            description="Partial COCO labels + stride-4 P2 high-resolution detection/pose head. Reaches 79.35 paper AUC on ShanghaiTech L0 (+17.70 vs M00) and +5.49 on UBnormal.",
            weights_path=a / "EXP-020/MLP.pt",
            backend="ultralytics",
            imgsz=640,
            sht_root=d / "exp020-MLP-sht",
            ub_root=d / "exp020-MLP-ub",
        ),
        ModelSpec(
            id="exp020-M00",
            name="EXP-020 M00: Standard-Label Control YOLOv8n-pose @640",
            category="EXP-020 (COCO 2x2 Ablation)",
            experiment="EXP-020",
            description="4-epoch COCO control arm with standard Ultralytics label conversion (drops keypoint-less persons) and stock head.",
            weights_path=a / "EXP-020/M00.pt",
            backend="ultralytics",
            imgsz=640,
            sht_root=d / "exp020-M00-sht",
            ub_root=d / "exp020-M00-ub",
        ),
        ModelSpec(
            id="exp020-M0P",
            name="EXP-020 M0P: Standard-Label + P2 Head YOLOv8n-pose @640",
            category="EXP-020 (COCO 2x2 Ablation)",
            experiment="EXP-020",
            description="Standard COCO labels + stride-4 P2 head. Shows P2 resolution alone (+0.24 AUC vs M00) does not fix small-person blindness without partial labels.",
            weights_path=a / "EXP-020/M0P.pt",
            backend="ultralytics",
            imgsz=640,
            sht_root=d / "exp020-M0P-sht",
            ub_root=d / "exp020-M0P-ub",
        ),
        ModelSpec(
            id="exp013-sht-ft",
            name="EXP-013: ShanghaiTech Pseudo-Label Fine-Tuned YOLOv8n-pose",
            category="Site Adaptation & Weight Averaging",
            experiment="EXP-013",
            description="YOLOv8n-pose fine-tuned on ShanghaiTech training split with AlphaPose pseudo-labels. Reaches 80.28 AUC on SHT L0 (+11.75) at zero extra inference cost, but site-specific (-4.12 on UBnormal).",
            weights_path=a / "EXP-013/yolov8n-pose-sht-ft.pt",
            backend="ultralytics",
            imgsz=640,
            sht_root=d / "exp013-sht",
            ub_root=d / "exp013-ub",
        ),
        ModelSpec(
            id="exp014-wise-0.75",
            name="EXP-014: WiSE-FT (alpha=0.75, BN-Recalibrated)",
            category="Site Adaptation & Weight Averaging",
            experiment="EXP-014",
            description="Weight-space interpolation (0.75 fine-tuned + 0.25 stock) with BatchNorm statistics recalibrated on 512 SHT training frames. Recovers UBnormal transfer (55.67 AUC) while retaining 77.60 on SHT.",
            weights_path=a / "EXP-014/yolov8n-pose-wise-a0.75.pt",
            backend="ultralytics",
            imgsz=640,
            sht_root=d / "exp014-a0.75-sht",
            ub_root=d / "exp014-a0.75-ub",
        ),
        ModelSpec(
            id="exp014-wise-0.5",
            name="EXP-014: WiSE-FT (alpha=0.50, BN-Recalibrated)",
            category="Site Adaptation & Weight Averaging",
            experiment="EXP-014",
            description="50/50 weight-space interpolation between stock COCO and SHT fine-tuned YOLOv8n-pose with BN recalibration.",
            weights_path=a / "EXP-014/yolov8n-pose-wise-a0.5.pt",
            backend="ultralytics",
            imgsz=640,
            sht_root=d / "exp014-a0.5-sht",
            ub_root=d / "exp014-a0.5-ub",
        ),
        ModelSpec(
            id="exp014-wise-0.25",
            name="EXP-014: WiSE-FT (alpha=0.25, BN-Recalibrated)",
            category="Site Adaptation & Weight Averaging",
            experiment="EXP-014",
            description="25% fine-tuned + 75% stock weight-space interpolation with BN recalibration.",
            weights_path=a / "EXP-014/yolov8n-pose-wise-a0.25.pt",
            backend="ultralytics",
            imgsz=640,
            sht_root=d / "exp014-a0.25-sht",
            ub_root=d / "exp014-a0.25-ub",
        ),
        ModelSpec(
            id="exp016-topdown-640",
            name="EXP-016: Top-Down YOLOv8n-det @640 + RTMPose-s (K=1)",
            category="Top-Down Architecture",
            experiment="EXP-016",
            description="Two-stage top-down pipeline: YOLOv8n person detector @640 + BoT-SORT + RTMPose-s (SimCC 256x192 ONNX) on every crop. 80.52 paper AUC on SHT L0 (71.8 ms/frame in VM).",
            weights_path=a / "EXP-016/models/yolov8n.pt",
            backend="topdown",
            imgsz=640,
            pose_dir_override={
                "ShanghaiTech:L0": d / "exp016/poses/L0_n640_K1_rtmpose-s",
                "ShanghaiTech:L2": d / "exp016/poses/L2_n640_K1_rtmpose-s",
            },
            scores_dir_override={
                "ShanghaiTech:L0": d / "exp016/scores/L0_n640_K1_rtmpose-s",
                "ShanghaiTech:L2": d / "exp016/scores/L2_n640_K1_rtmpose-s",
            },
            cache_dir_override={
                "ShanghaiTech:L0": d / "exp016/cache/L0_n640",
                "ShanghaiTech:L2": d / "exp016/cache/L2_n640",
            },
        ),
        ModelSpec(
            id="exp016-topdown-960",
            name="EXP-016: Top-Down YOLOv8n-det @960 + RTMPose-s (K=1)",
            category="Top-Down Architecture",
            experiment="EXP-016",
            description="Two-stage top-down pipeline at 960px detector resolution + RTMPose-s. Reaches 83.11 paper AUC (+14.58 vs baseline) on SHT L0 (111.6 ms/frame in VM).",
            weights_path=a / "EXP-016/models/yolov8n.pt",
            backend="topdown",
            imgsz=960,
            pose_dir_override={
                "ShanghaiTech:L0": d / "exp016/poses/L0_n960_K1_rtmpose-s",
                "ShanghaiTech:L2": d / "exp016/poses/L2_n960_K1_rtmpose-s",
            },
            scores_dir_override={
                "ShanghaiTech:L0": d / "exp016/scores/L0_n960_K1_rtmpose-s",
                "ShanghaiTech:L2": d / "exp016/scores/L2_n960_K1_rtmpose-s",
            },
            cache_dir_override={
                "ShanghaiTech:L0": d / "exp016/cache/L0_n960",
                "ShanghaiTech:L2": d / "exp016/cache/L2_n960",
            },
        ),
        ModelSpec(
            id="exp007-yolov8s-640",
            name="EXP-007: YOLOv8s-pose @640 (Larger Model)",
            category="Architecture & Quantization",
            experiment="EXP-007",
            description="Larger YOLOv8s-pose @640 (2.5x slower). Paradoxically worse coverage (0.569 vs 0.674) and lower AUC (66.25 vs 68.53) on ShanghaiTech due to stricter confidence suppression on small unlabelled people.",
            weights_path=a / "EXP-001/models/yolov8s-pose.pt",
            backend="ultralytics",
            imgsz=640,
            sht_root=d / "exp007-sht",
            ub_root=d / "exp007-ub",
        ),
        ModelSpec(
            id="exp017-yolo26n-640",
            name="EXP-017: YOLO26n-pose @640 (NMS-Free End-to-End)",
            category="Architecture & Quantization",
            experiment="EXP-017",
            description="Ultralytics YOLO26n-pose NMS-free one-to-one head @640. Tested in EXP-017; lower recall and -6.99 AUC vs YOLOv8n-pose on ShanghaiTech.",
            weights_path=a / "EXP-017/yolo26n-pose.pt",
            backend="ultralytics",
            imgsz=640,
            nms_free=True,
            sht_root=d / "exp017-sht",
            ub_root=d / "exp017-ub",
        ),
        ModelSpec(
            id="exp012-ncnn16-800",
            name="EXP-012: YOLOv8n-pose @800 (NCNN FP16)",
            category="Architecture & Quantization",
            experiment="EXP-012",
            description="NCNN FP16 at 800px (480x800 rect). Runs in 18.9 ms (0.57x ORT FP32 @640 cost) while raising SHT L0 paper AUC to 70.65.",
            weights_path=vm_models / "yolov8n-pose-800-480x800-ncnn",
            backend="ncnn",
            imgsz=800,
            sht_root=d / "exp012",
            sht_cache_prefix="L0_E0-ncnn16-800",
        ),
        ModelSpec(
            id="exp012-int8-800",
            name="EXP-012: YOLOv8n-pose @800 (ONNX INT8 Mixed)",
            category="Architecture & Quantization",
            experiment="EXP-012",
            description="ONNX Runtime QDQ INT8 quantized backbone/neck with FP32 pose decode head at 800px (480x800). 69.54 paper AUC on SHT L0.",
            weights_path=vm_models / "yolov8n-pose-800-480x800-int8.onnx",
            backend="ort",
            imgsz=800,
            sht_root=d / "exp012",
            sht_cache_prefix="L0_E0-int8-800",
        ),
        ModelSpec(
            id="stock-yolov8n-960",
            name="YOLOv8n-pose @960 (High-Res Offline Upper Bound)",
            category="Baseline",
            experiment="EXP-003 / EXP-006",
            description="Stock YOLOv8n-pose run at 960px input resolution. Offline paper AUC 71.71 on SHT (+3.18 vs 640), but misses 24 fps CPU deadlines without OS/quantization help.",
            weights_path=a / "EXP-001/models/yolov8n-pose.pt",
            backend="ultralytics",
            imgsz=960,
            sht_root=d / "exp003",
            ub_root=d / "exp006",
            sht_cache_prefix="L0_E0-r960",
            ub_cache_prefix="L0-960_E0",
        ),
        ModelSpec(
            id="released-alphapose",
            name="Released AlphaPose Reference Poses (Oracle Upper Bound)",
            category="Reference / Upper Bound",
            experiment="EXP-001 / EXP-006",
            description="Official STG-NF released AlphaPose + tracker poses (heavy GPU front-end). 85.94 AUC on ShanghaiTech, 71.80 AUC on UBnormal.",
            weights_path=None,
            backend="released",
            pose_dir_override={
                "ShanghaiTech:L0": STGNF_ROOT / "data/ShanghaiTech/pose/test",
                "UBnormal:L0": STGNF_ROOT / "data/UBnormal/pose/test",
            },
            scores_dir_override={
                "ShanghaiTech:L0": d / "exp015/scores/released",
                "UBnormal:L0": a / "EXP-006/R0",
            },
        ),
    ]


MODEL_BY_ID: dict[str, ModelSpec] = {m.id: m for m in build_model_catalog()}


@dataclass(frozen=True)
class ScorerSpec:
    id: str
    name: str
    dataset: str  # "ShanghaiTech" | "UBnormal"
    checkpoint: Path
    description: str


def build_scorer_catalog(dataset: str) -> list[ScorerSpec]:
    """Available STG-NF normalizing flow checkpoints for the given dataset."""
    d = DATA_ROOT
    if dataset == "ShanghaiTech":
        specs = [
            ScorerSpec(
                id="sht-released",
                name="Released STG-NF (85.9 AUC, 616 params)",
                dataset="ShanghaiTech",
                checkpoint=SHT_CHECKPOINT,
                description="Official frozen STG-NF normalizing flow checkpoint (K=8, L=1, R=3.0).",
            ),
            ScorerSpec(
                id="sht-exp019-std-s0",
                name="EXP-019 Retrained Standard STG-NF (seed 0)",
                dataset="ShanghaiTech",
                checkpoint=d / "exp019/models/standard_s0/final.pt",
                description="Our clean re-implementation of standard STG-NF trained on SHT released poses.",
            ),
            ScorerSpec(
                id="sht-exp019-gau-s0",
                name="EXP-019 STG-NF-GAU (Gated Attention Unit, 5,808 params)",
                dataset="ShanghaiTech",
                checkpoint=d / "exp019/models/gau_s0/final.pt",
                description="STG-NF with Gated Attention Unit temporal coupling subnet (EXP-019 negative result: -0.78 AUC vs standard).",
            ),
            ScorerSpec(
                id="sht-exp018-fa-cv",
                name="EXP-018 STG-NF-FA (Filling-Aware, Constant-Velocity)",
                dataset="ShanghaiTech",
                checkpoint=d / "exp018/models/SHT_fa-cv/final.pt",
                description="STG-NF conditioned on pose provenance channel (1=detected, 0=predicted) trained with CV fills.",
            ),
            ScorerSpec(
                id="sht-exp018-fa-gru",
                name="EXP-018 STG-NF-FA (Filling-Aware, Learned GRU)",
                dataset="ShanghaiTech",
                checkpoint=d / "exp018/models/SHT_fa-gru/final.pt",
                description="STG-NF conditioned on pose provenance channel trained with learned GRU fills (DAPF).",
            ),
        ]
    else:
        specs = [
            ScorerSpec(
                id="ub-released",
                name="Released UBnormal STG-NF (71.8 AUC, seg_len=16)",
                dataset="UBnormal",
                checkpoint=ubnormal.CHECKPOINT,
                description="Official frozen UBnormal unsupervised STG-NF checkpoint.",
            ),
            ScorerSpec(
                id="ub-exp018-fa-cv",
                name="EXP-018 UBnormal STG-NF-FA (Filling-Aware, CV)",
                dataset="UBnormal",
                checkpoint=d / "exp018/models/UB_fa-cv/final.pt",
                description="UBnormal STG-NF conditioned on provenance channel with constant-velocity fills.",
            ),
            ScorerSpec(
                id="ub-exp018-fa-gru",
                name="EXP-018 UBnormal STG-NF-FA (Filling-Aware, GRU)",
                dataset="UBnormal",
                checkpoint=d / "exp018/models/UB_fa-gru/final.pt",
                description="UBnormal STG-NF conditioned on provenance channel with learned GRU fills.",
            ),
        ]
    return [s for s in specs if s.checkpoint.exists()]


# ---------------------------------------------------------------- Dataset & Video Discovery

@dataclass(frozen=True)
class VideoExample:
    dataset: str
    video_id: str
    scene_id: str
    num_frames: int
    anomalous_frames: int
    anomalous_ratio: float
    stratum: str  # "normal" | "anomalous" | "fog/smoke/rain"
    source_path: Path

    @property
    def label_summary(self) -> str:
        kind = "ANOMALOUS" if self.anomalous_frames > 0 else "NORMAL"
        extra = f" [{self.stratum}]" if self.stratum not in ("daylight", "normal") else ""
        return f"{self.video_id} — {kind} ({self.anomalous_frames}/{self.num_frames} anom frames){extra}"


def list_dataset_videos(dataset: str) -> list[VideoExample]:
    """Return all test videos for ShanghaiTech (107) or UBnormal (211) with ground-truth stats."""
    examples: list[VideoExample] = []
    if dataset == "ShanghaiTech":
        if not SHT_GT_DIR.exists():
            return []
        for gt_file in sorted(SHT_GT_DIR.glob("*.npy")):
            vid = gt_file.stem
            scene_id = vid.split("_")[0]
            gt = np.load(gt_file).astype(int)
            n_frames = len(gt)
            n_anom = int(gt.sum())
            src = SHT_FRAMES_ROOT / vid
            examples.append(
                VideoExample(
                    dataset="ShanghaiTech",
                    video_id=vid,
                    scene_id=scene_id,
                    num_frames=n_frames,
                    anomalous_frames=n_anom,
                    anomalous_ratio=n_anom / max(1, n_frames),
                    stratum="daylight",
                    source_path=src,
                )
            )
    elif dataset == "UBnormal":
        if not ubnormal.TEST_LIST.exists():
            return []
        for vid in ubnormal.load_test_list():
            scene_id, _ = ubnormal.parse_video_name(vid)
            gt = ubnormal.load_gt(vid)
            n_frames = len(gt)
            n_anom = int(gt.sum())
            stratum = "weather (fog/smoke/rain)" if ubnormal.is_weather(vid) else "normal"
            src = UB_VIDEOS_ROOT / f"{vid}.mp4"
            examples.append(
                VideoExample(
                    dataset="UBnormal",
                    video_id=vid,
                    scene_id=scene_id,
                    num_frames=n_frames,
                    anomalous_frames=n_anom,
                    anomalous_ratio=n_anom / max(1, n_frames),
                    stratum=stratum,
                    source_path=src,
                )
            )
    return examples


def load_ground_truth(dataset: str, video_id: str) -> np.ndarray:
    """1 = anomalous frame, 0 = normal frame."""
    if dataset == "ShanghaiTech":
        return np.load(SHT_GT_DIR / f"{video_id}.npy").astype(int)
    return ubnormal.load_gt(video_id)


def read_video_frame(dataset: str, video_id: str, frame_idx: int) -> np.ndarray | None:
    """Read a single BGR frame (uint8) by 0-based index from ShanghaiTech jpg folder or UBnormal mp4."""
    if dataset == "ShanghaiTech":
        folder = SHT_FRAMES_ROOT / video_id
        # ShanghaiTech test frames are named 000.jpg .. NNN.jpg (sometimes 4 digits for >999 frames)
        for fmt in (f"{frame_idx:03d}.jpg", f"{frame_idx:04d}.jpg", f"{frame_idx}.jpg"):
            candidate = folder / fmt
            if candidate.exists():
                return cv2.imread(str(candidate))
        jpgs = sorted(folder.glob("*.jpg"))
        if 0 <= frame_idx < len(jpgs):
            return cv2.imread(str(jpgs[frame_idx]))
        return None
    else:
        mp4 = UB_VIDEOS_ROOT / f"{video_id}.mp4"
        if not mp4.exists():
            return None
        cap = cv2.VideoCapture(str(mp4))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        cap.release()
        return frame if ok else None


def iter_video_frames(dataset: str, video_id: str, start_frame: int = 0, end_frame: int | None = None, step: int = 1):
    """Yield `(frame_idx, frame_bgr)` sequentially without reopening video files on every frame."""
    step = max(1, int(step))
    if dataset == "ShanghaiTech":
        folder = SHT_FRAMES_ROOT / video_id
        jpgs = sorted(folder.glob("*.jpg"))
        stop = len(jpgs) if end_frame is None else min(len(jpgs), end_frame)
        for f_idx in range(max(0, start_frame), stop, step):
            img = cv2.imread(str(jpgs[f_idx]))
            if img is not None:
                yield f_idx, img
    else:
        mp4 = UB_VIDEOS_ROOT / f"{video_id}.mp4"
        if not mp4.exists():
            return
        cap = cv2.VideoCapture(str(mp4))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        stop = total if end_frame is None else min(total, end_frame)
        cur = max(0, start_frame)
        if cur > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, cur)
        while cur < stop:
            ok, frame = cap.read()
            if not ok:
                break
            if (cur - start_frame) % step == 0:
                yield cur, frame
            cur += 1
        cap.release()


# ---------------------------------------------------------------- Artifact Resolution (Caches, Poses, Scores)

def _resolve_cache_name(spec: ModelSpec, dataset: str, level: str, enhance: str) -> str:
    base = spec.sht_cache_prefix if dataset == "ShanghaiTech" else spec.ub_cache_prefix
    if base != "L0_E0":
        # Special suffix/prefix cache like L0_E0-ncnn16-800 or L0_E0-r960 or L0-960_E0
        if dataset == "ShanghaiTech" and base.startswith("L0_E0"):
            suffix = base.removeprefix("L0_E0")
            return f"{level}_{enhance}{suffix}"
        if dataset == "UBnormal" and base == "L0-960_E0":
            return f"{level}-960_{enhance}" if level == "L0" else f"{level}_{enhance}"
    return f"{level}_{enhance}"


def find_cache_npz(spec: ModelSpec, dataset: str, video_id: str, level: str = "L0", enhance: str = "E0") -> Path | None:
    key = f"{dataset}:{level}"
    if key in spec.cache_dir_override:
        p = spec.cache_dir_override[key] / f"{video_id}.npz"
        return p if p.exists() else None
    root = spec.sht_root if dataset == "ShanghaiTech" else spec.ub_root
    if root is None:
        return None
    cname = _resolve_cache_name(spec, dataset, level, enhance)
    p = root / "cache" / cname / f"{video_id}.npz"
    return p if p.exists() else None


def find_pose_json(spec: ModelSpec, dataset: str, video_id: str, level: str = "L0", enhance: str = "E0",
                   c: float = 0.25, tracker: str = "botsort") -> Path | None:
    key = f"{dataset}:{level}"
    if key in spec.pose_dir_override:
        pdir = spec.pose_dir_override[key]
    else:
        root = spec.sht_root if dataset == "ShanghaiTech" else spec.ub_root
        if root is None:
            return None
        cname = _resolve_cache_name(spec, dataset, level, enhance)
        pdir = root / "poses" / f"{cname}_c{c}_{tracker}"
    if not pdir.is_dir():
        return None
    # Match exact video prefix
    for candidate in pdir.glob(f"{video_id}_*_tracked_person.json"):
        if dataset == "UBnormal":
            if candidate.name.startswith(f"{video_id}_alphapose"):
                return candidate
        else:
            parts = candidate.name.split("_")
            if "_".join(parts[:2]) == video_id:
                return candidate
    return None


def find_scores_npz(spec: ModelSpec, dataset: str, level: str = "L0", enhance: str = "E0",
                    c: float = 0.25, tracker: str = "botsort", variant: str = "paper") -> Path | None:
    key = f"{dataset}:{level}"
    if key in spec.scores_dir_override:
        sdir = spec.scores_dir_override[key]
        # EXP-015 released scores use frame_scores_middle_offline_gaussian.npz or frame_scores_paper.npz
        for fname in (
            f"frame_scores_{variant}.npz",
            "frame_scores_middle_offline_gaussian.npz" if variant == "paper" else "frame_scores_end_causal_half_gaussian.npz",
        ):
            if (sdir / fname).exists():
                return sdir / fname
        return None
    root = spec.sht_root if dataset == "ShanghaiTech" else spec.ub_root
    if root is None:
        return None
    cname = _resolve_cache_name(spec, dataset, level, enhance)
    p = root / "scores" / f"{cname}_c{c}_{tracker}" / f"frame_scores_{variant}.npz"
    return p if p.exists() else None


def load_video_CachedDetections(npz_path: Path, frame_idx: int, conf_thresh: float = 0.25) -> dict:
    """Return boxes (N,4), conf (N,), keypoints (N,17,3) and timing metadata for a single frame from a detcache npz."""
    data = np.load(npz_path)
    counts = data["counts"]
    ends = np.cumsum(counts)
    if frame_idx < 0 or frame_idx >= len(counts):
        return {"boxes": np.zeros((0, 4)), "conf": np.zeros(0), "keypoints": np.zeros((0, 17, 3)), "timing": {}}
    start = int(ends[frame_idx] - counts[frame_idx])
    end = int(ends[frame_idx])
    conf = data["conf"][start:end]
    keep = conf >= conf_thresh
    boxes = data["boxes"][start:end][keep]
    conf_kept = conf[keep]
    kpts = data["keypoints"][start:end][keep] if "keypoints" in data else np.zeros((len(boxes), 17, 3))
    timing = {}
    for k in ("luma_dark", "luma_enhanced", "darken_ms", "enhance_ms", "yolo_pre_ms", "yolo_infer_ms", "yolo_post_ms"):
        if k in data and frame_idx < len(data[k]):
            timing[k] = float(data[k][frame_idx])
    if "infer_ms" in data and frame_idx < len(data["infer_ms"]):
        timing["yolo_infer_ms"] = float(data["infer_ms"][frame_idx])
    return {"boxes": boxes, "conf": conf_kept, "keypoints": kpts, "timing": timing, "all_conf": conf}


def load_video_TrackedPoses(json_path: Path) -> dict[int, list[dict]]:
    """Parse a `*_tracked_person.json` into `{frame_idx: [{"track_id": int, "keypoints": (17,3), "score": float}, ...]}`."""
    raw = json.loads(json_path.read_text())
    by_frame: dict[int, list[dict]] = {}
    for pid_str, track in raw.items():
        track_dict = merge_parts(track)
        pid = int(pid_str)
        for f_str, entry in track_dict.items():
            f_idx = int(f_str)
            kpts = np.asarray(entry["keypoints"], dtype=np.float32).reshape(17, 3)
            score = float(entry.get("scores", 1.0))
            by_frame.setdefault(f_idx, []).append({"track_id": pid, "keypoints": kpts, "score": score})
    return by_frame


def load_video_FrameScores(scores_npz: Path, video_id: str) -> dict | None:
    """Load `score_<vid>`, `label_<vid>`, `covered_<vid>` from `frame_scores_<variant>.npz`."""
    data = np.load(scores_npz)
    s_key, l_key, c_key = f"score_{video_id}", f"label_{video_id}", f"covered_{video_id}"
    if s_key not in data:
        return None
    scores = data[s_key].astype(np.float64)
    labels = data[l_key].astype(int) if l_key in data else None
    covered = data[c_key].astype(bool) if c_key in data else np.ones(len(scores), dtype=bool)
    return {"scores": scores, "labels": labels, "covered": covered}


def load_dataset_summary_row(spec: ModelSpec, dataset: str, level: str = "L0", enhance: str = "E0",
                             c: float = 0.25, variant: str = "paper") -> dict | None:
    """Read the precomputed dataset-wide AUC and coverage row from results.jsonl or rows.jsonl."""
    if spec.backend == "topdown" and dataset == "ShanghaiTech":
        rows_file = DATA_ROOT / "exp016/rows.jsonl"
        if rows_file.exists():
            cid = f"{level}_n{spec.imgsz}_K1_rtmpose-s"
            for line in rows_file.read_text().splitlines():
                if not line.strip():
                    continue
                r = json.loads(line)
                if r.get("config") == cid and r.get("variant") == variant and not r.get("invalid"):
                    return r
        return None

    root = spec.sht_root if dataset == "ShanghaiTech" else spec.ub_root
    if root is None or not (root / "results.jsonl").exists():
        return None
    cname = _resolve_cache_name(spec, dataset, level, enhance)
    for line in (root / "results.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("cache") == cname and r.get("c") == c and r.get("tracker") == "botsort" and r.get("variant") == variant:
            return r
    return None


# ---------------------------------------------------------------- Experiment Catalog (EXP-000..023 + Q2-P1-E00..E07 + Pi 5)

EXP_IDS: tuple[str, ...] = tuple(
    id_ for i in range(0, 24) for id_ in ((f"EXP-{i:03d}", "EXP-010b") if i == 10 else (f"EXP-{i:03d}",))
)
Q2_IDS: tuple[str, ...] = (
    "Q2-P1-E00", "Q2-P1-E01", "Q2-P1-E02", "Q2-P1-E03", "Q2-P1-E04", "Q2-P1-E05",
    "Q2-P1-E06", "Q2-P1-E07", "Q2-P1-Pi5_huang_repro", "Q2-P1-Pi5_suite",
)
EXPERIMENT_IDS: tuple[str, ...] = EXP_IDS + Q2_IDS

_RESULTS_FILES = ("results.csv", "per_video_results.csv")
_EXP_DIR_RE = re.compile(r"^(EXP-\d{3}[a-z]?)-(.+)$")
_Q2_DIR_RE = re.compile(r"^(Q2-P1-(?:E\d{2}|Pi5_[a-z0-9]+(?:_[a-z0-9]+)*?))(?:_([a-z0-9_]+))?$")


def _split_experiment_dir(name: str) -> tuple[str, str] | None:
    """Directory name -> (experiment id, title slug). `Q2-P1-E05_confirmation` -> (`Q2-P1-E05`, `confirmation`)."""
    m = _EXP_DIR_RE.match(name)
    if m:
        return m.group(1), m.group(2)
    if name.startswith("Q2-P1-Pi5_"):
        return name, name.removeprefix("Q2-P1-Pi5_")  # the Pi 5 folders are addressed by their full name
    m = re.match(r"^(Q2-P1-E\d{2})(?:_(.+))?$", name)
    if m:
        return m.group(1), m.group(2) or ""
    return None


@dataclass(frozen=True)
class ExperimentSpec:
    """One experiment folder (EXP-0xx or Q2-P1-*) and which machine-readable artefacts it holds."""

    id: str
    family: str  # "EXP" | "Q2-P1"
    title: str
    path: Path | None
    files: tuple[str, ...] = ()

    @property
    def exists(self) -> bool:
        return self.path is not None

    def _first(self, names: tuple[str, ...]) -> Path | None:
        if self.path is None:
            return None
        for n in names:
            if n in self.files:
                return self.path / n
        return None

    @property
    def results_csv(self) -> Path | None:
        return self._first(_RESULTS_FILES)

    @property
    def events_csv(self) -> Path | None:
        return self._first(("events.csv",))

    @property
    def summary_json(self) -> Path | None:
        return self._first(("summary.json",))

    @property
    def report_md(self) -> Path | None:
        return self._first(("report.md", "results.md"))

    @property
    def plan_md(self) -> Path | None:
        return self._first(("plan.md",))

    @property
    def status(self) -> str:
        if not self.exists:
            return "missing"
        if self.results_csv or self.summary_json or self.events_csv:
            return "results available"
        if self.report_md:
            return "report only"
        return "planned / running (no results yet)"


def build_experiment_catalog(root: Path | None = None) -> list[ExperimentSpec]:
    """`EXP-000`..`EXP-023` followed by `Q2-P1-E00`..`E07`, `Q2-P1-Pi5_huang_repro`, `Q2-P1-Pi5_suite`.

    Every expected id is listed even if its folder is missing (status `missing`), so the UI never silently drops one.
    """
    root = Path(root) if root is not None else EXPERIMENTS_ROOT
    found: dict[str, tuple[str, Path]] = {}
    if root.is_dir():
        for p in sorted(root.iterdir()):
            if not p.is_dir():
                continue
            parsed = _split_experiment_dir(p.name)
            if parsed is not None:
                found[parsed[0]] = (parsed[1], p)
    specs: list[ExperimentSpec] = []
    for exp_id in EXPERIMENT_IDS:
        title, path = found.get(exp_id, ("", None))
        files = tuple(sorted(f.name for f in path.iterdir() if f.is_file())) if path is not None else ()
        specs.append(ExperimentSpec(
            id=exp_id,
            family="Q2-P1" if exp_id.startswith("Q2-P1") else "EXP",
            title=title.replace("_", " "),
            path=path,
            files=files,
        ))
    return specs


EXPERIMENT_BY_ID: dict[str, ExperimentSpec] = {e.id: e for e in build_experiment_catalog()}


@functools.lru_cache(maxsize=16)
def _read_csv_cached(path_str: str) -> pd.DataFrame:
    return pd.read_csv(path_str, low_memory=False)


def load_experiment_artifacts(exp_id: str) -> dict:
    """`{"spec", "results", "events", "summary"}` for one experiment; missing artefacts are `None` (E06/E07 may not have results yet)."""
    spec = EXPERIMENT_BY_ID[exp_id]
    out: dict = {"spec": spec, "results": None, "events": None, "summary": None}
    if spec.results_csv is not None:
        out["results"] = _read_csv_cached(str(spec.results_csv))
    if spec.events_csv is not None:
        out["events"] = _read_csv_cached(str(spec.events_csv))
    if spec.summary_json is not None:
        try:
            out["summary"] = json.loads(spec.summary_json.read_text())
        except (OSError, ValueError):
            out["summary"] = None
    return out


_AUC_COLUMNS = ("auc_all", "auc_confirmation", "auc_delivered", "auc_S4", "auc_S0", "auc_ideal", "auc_offline_S0", "auc_online", "value")
_GROUP_COLUMNS = ("arm", "frontend", "schedule", "metric")


def summarize_results_table(df: pd.DataFrame | None, max_rows: int = 40) -> pd.DataFrame | None:
    """Compact leaderboard of a results table: mean of the first AUC-like column per arm/front-end/schedule (or per metric for the Pi 5 long format)."""
    if df is None or df.empty:
        return None
    group = next((c for c in _GROUP_COLUMNS if c in df.columns), None)
    metric = next((c for c in _AUC_COLUMNS if c in df.columns and pd.api.types.is_numeric_dtype(df[c])), None)
    if group is None or metric is None:
        return None
    g = df.groupby(group, dropna=True)[metric].agg(["count", "mean", "min", "max"]).reset_index()
    g = g.rename(columns={"mean": f"mean {metric}", "min": f"min {metric}", "max": f"max {metric}", "count": "rows"})
    return g.sort_values(f"mean {metric}", ascending=False).head(max_rows).reset_index(drop=True)


# ---------------------------------------------------------------- Dataset Explorer: splits, events, tags

DATASET_FPS: dict[str, float] = {"ShanghaiTech": 24.0, "UBnormal": 30.0}
SHORT_EVENT_S = 2.5  # Q2-P1-E05: short event = at most 2.5 s (60 frames SHT, 75 frames UB)
Q2_AUDIT_CONFIG = EXPERIMENTS_ROOT / "Q2-P1-E00_audit/config.yaml"
Q2_E05_EVENTS = EXPERIMENTS_ROOT / "Q2-P1-E05_confirmation/events.csv"
SPLIT_NAMES = ("test_dev", "test_confirmation")
TAG_ORDER = ("short_event", "sustained_event", "new_entry_near_onset", "sudden_fast_motion", "high_crowd")


def native_fps(dataset: str) -> float:
    return DATASET_FPS[dataset]


def frames_to_seconds(frame: int | float, fps: float) -> float:
    return float(frame) / float(fps)


def gt_intervals(gt: np.ndarray) -> list[tuple[int, int]]:
    """Maximal runs of anomalous frames of a 0/1 mask -> inclusive `(start_frame, end_frame)` pairs."""
    g = np.asarray(gt).astype(bool).astype(np.int8)
    if g.size == 0:
        return []
    d = np.diff(np.concatenate([[0], g, [0]]))
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1) - 1
    return [(int(s), int(e)) for s, e in zip(starts, ends)]


def interval_seconds(start: int, end: int, fps: float) -> tuple[float, float]:
    """Inclusive frame interval -> `[t_start, t_end)` in seconds (the end frame lasts until the next frame boundary)."""
    return start / fps, (end + 1) / fps


def format_interval(start: int, end: int, fps: float) -> str:
    t0, t1 = interval_seconds(start, end, fps)
    return f"f{start}-{end} ({t0:.2f}-{t1:.2f} s)"


@dataclass(frozen=True)
class GtEvent:
    index: int
    start_frame: int  # inclusive
    end_frame: int  # inclusive
    duration_frames: int
    start_s: float
    end_s: float
    duration_s: float
    tags: tuple[str, ...]

    @property
    def mid_frame(self) -> int:
        return (self.start_frame + self.end_frame) // 2

    @property
    def is_short(self) -> bool:
        return "short_event" in self.tags

    def label(self) -> str:
        return f"#{self.index}  f{self.start_frame}-{self.end_frame}  ({self.start_s:.2f}-{self.end_s:.2f} s)  [{', '.join(self.tags)}]"


def parse_split_config(config: dict, dataset: str) -> dict[str, str]:
    """E00 `config.yaml` dict -> `{clip: "test_dev" | "test_confirmation"}` for one dataset."""
    splits = config["datasets"][dataset]["splits"]
    return {str(clip): name for name in SPLIT_NAMES for clip in splits.get(name, [])}


@functools.lru_cache(maxsize=4)
def load_split_map(dataset: str, config_path: Path = Q2_AUDIT_CONFIG) -> dict[str, str]:
    """Clip -> split from `Q2-P1-E00_audit/config.yaml` (empty if the config is not there)."""
    import yaml

    if not Path(config_path).exists():
        return {}
    return parse_split_config(yaml.safe_load(Path(config_path).read_text()), dataset)


@functools.lru_cache(maxsize=4)
def load_event_tag_table(dataset: str, events_path: Path = Q2_E05_EVENTS) -> dict[str, list[dict]]:
    """`{clip: [{start, end, short, new_entry_near_onset, fast_motion, high_crowd}, ...]}` from `Q2-P1-E05_confirmation/events.csv`.

    The event definitions are identical for every arm/cell of that file, so the first (arm, cell) pair is used.
    """
    p = Path(events_path)
    if not p.exists():
        return {}
    cols = ["dataset", "arm", "cell", "video", "start_frame", "end_frame", "short", "new_entry_near_onset", "fast_motion", "high_crowd"]
    df = pd.read_csv(p, usecols=cols)
    df = df[df["dataset"] == dataset]
    if df.empty:
        return {}
    df = df[(df["arm"] == df["arm"].iloc[0]) & (df["cell"] == df["cell"].iloc[0])]
    out: dict[str, list[dict]] = {}
    for r in df.itertuples(index=False):
        out.setdefault(str(r.video), []).append({
            "start": int(r.start_frame), "end": int(r.end_frame), "short": bool(r.short),
            "new_entry_near_onset": bool(r.new_entry_near_onset), "fast_motion": bool(r.fast_motion), "high_crowd": bool(r.high_crowd),
        })
    return out


def event_tags(short: bool, new_entry: bool | None = None, fast: bool | None = None, crowd: bool | None = None) -> tuple[str, ...]:
    """Subgroup tags in the E05 vocabulary (`None` = unknown, tag omitted)."""
    tags = ["short_event" if short else "sustained_event"]
    if new_entry:
        tags.append("new_entry_near_onset")
    if fast:
        tags.append("sudden_fast_motion")
    if crowd:
        tags.append("high_crowd")
    return tuple(tags)


def build_gt_events(dataset: str, video_id: str, gt: np.ndarray | None = None,
                    tag_table: dict[str, list[dict]] | None = None) -> list[GtEvent]:
    """Ground-truth anomaly events of one clip (exact inclusive frame intervals + seconds + E05 subgroup tags)."""
    fps = native_fps(dataset)
    if gt is None:
        gt = load_ground_truth(dataset, video_id)
    if tag_table is None:
        tag_table = load_event_tag_table(dataset)
    known = {(e["start"], e["end"]): e for e in tag_table.get(video_id, [])}
    short_frames = int(round(SHORT_EVENT_S * fps))
    events = []
    for i, (s, e) in enumerate(gt_intervals(gt)):
        dur = e - s + 1
        row = known.get((s, e))
        if row is not None:
            tags = event_tags(row["short"], row["new_entry_near_onset"], row["fast_motion"], row["high_crowd"])
        else:
            tags = event_tags(dur <= short_frames)
        t0, t1 = interval_seconds(s, e, fps)
        events.append(GtEvent(i, s, e, dur, t0, t1, dur / fps, tags))
    return events


def event_peak_frame(event: GtEvent, scores: np.ndarray | None = None) -> int:
    """Peak frame of an event: argmax of `scores` inside the event, or the middle frame when no scores are given."""
    if scores is None or len(scores) <= event.start_frame:
        return event.mid_frame
    seg = np.asarray(scores[event.start_frame:event.end_frame + 1], dtype=float)
    if seg.size == 0 or not np.isfinite(seg).any():
        return event.mid_frame
    return int(event.start_frame + int(np.nanargmax(seg)))


def build_clip_table(dataset: str, videos: list[VideoExample] | None = None) -> list[dict]:
    """One row per test clip: split, scene, frame count, fps, duration, #events, exact intervals and subgroup tags."""
    if videos is None:
        videos = list_dataset_videos(dataset)
    fps = native_fps(dataset)
    splits = load_split_map(dataset)
    tag_table = load_event_tag_table(dataset)
    rows = []
    for v in videos:
        gt = load_ground_truth(dataset, v.video_id)
        events = build_gt_events(dataset, v.video_id, gt, tag_table)
        tags = sorted({t for ev in events for t in ev.tags}, key=lambda t: TAG_ORDER.index(t))
        rows.append({
            "clip": v.video_id,
            "dataset": dataset,
            "split": splits.get(v.video_id, "n/a"),
            "scene": v.scene_id,
            "frames": int(len(gt)),
            "fps": fps,
            "duration_s": round(len(gt) / fps, 2),
            "anomalous_frames": int(gt.sum()),
            "anomalous_pct": round(100.0 * float(gt.mean()), 1) if len(gt) else 0.0,
            "n_events": len(events),
            "n_short": sum(ev.is_short for ev in events),
            "n_sustained": sum(not ev.is_short for ev in events),
            "events": "; ".join(format_interval(ev.start_frame, ev.end_frame, fps) for ev in events) or "-",
            "tags": ", ".join(tags),
            "stratum": v.stratum,
        })
    return rows


def summarize_clip_table(rows: list[dict]) -> dict:
    """Dataset-level statistics of a `build_clip_table` result."""
    if not rows:
        return {"clips": 0}
    frames = sum(r["frames"] for r in rows)
    anomalous = sum(r["anomalous_frames"] for r in rows)
    fps = rows[0]["fps"]
    tag_counts = {t: 0 for t in TAG_ORDER}
    for r in rows:
        for t in r["tags"].split(", "):
            if t:
                tag_counts[t] += 1
    return {
        "clips": len(rows),
        "clips_with_anomaly": sum(r["n_events"] > 0 for r in rows),
        "scenes": len({r["scene"] for r in rows}),
        "frames": frames,
        "hours": frames / fps / 3600.0,
        "fps": fps,
        "anomalous_frames": anomalous,
        "anomalous_ratio": anomalous / max(1, frames),
        "events": sum(r["n_events"] for r in rows),
        "short_events": sum(r["n_short"] for r in rows),
        "sustained_events": sum(r["n_sustained"] for r in rows),
        "test_dev_clips": sum(r["split"] == "test_dev" for r in rows),
        "test_confirmation_clips": sum(r["split"] == "test_confirmation" for r in rows),
        "clips_with_tag": tag_counts,
    }


# ---------------------------------------------------------------- Dataset Explorer: detection / skeleton overlay sources

@dataclass(frozen=True)
class OverlaySource:
    key: str
    label: str
    model_id: str


OVERLAY_SOURCES: tuple[OverlaySource, ...] = (
    OverlaySource("released", "Released AlphaPose (reference poses)", "released-alphapose"),
    OverlaySource("stock", "Stock YOLOv8n-pose @640", "stock-yolov8n-640"),
    OverlaySource("ml0_ncnn", "ML0 NCNN @640", "exp020-ML0-ncnn"),
    OverlaySource("mlp", "MLP @640", "exp020-MLP"),
    OverlaySource("yolo26", "YOLO26n-pose @640", "exp017-yolo26n-640"),
)
OVERLAY_BY_KEY = {s.key: s for s in OVERLAY_SOURCES}
HIGH_CONF = 0.20  # two-stage BoT-SORT first-association gate (Q2-P1 locked tracker: c=0.20, low=0.05)
LOW_CONF = 0.05


@functools.lru_cache(maxsize=8)
def _cached_detection_frames(npz_str: str) -> tuple[list[dict], np.ndarray]:
    frames, data = load_detections(Path(npz_str))
    return frames, np.asarray(data.get("frame_ids", np.arange(len(frames))))


@functools.lru_cache(maxsize=8)
def _cached_tracked_poses(json_str: str) -> dict[int, list[dict]]:
    return load_video_TrackedPoses(Path(json_str))


def boxes_from_keypoints(kpts: np.ndarray, vis_thresh: float = 0.05) -> np.ndarray:
    """(N,17,3) keypoints -> (N,4) xyxy boxes spanning the visible joints (zeros if none visible)."""
    boxes = np.zeros((len(kpts), 4), dtype=np.float32)
    for i, k in enumerate(kpts):
        vis = k[k[:, 2] > vis_thresh]
        if len(vis):
            boxes[i] = (vis[:, 0].min(), vis[:, 1].min(), vis[:, 0].max(), vis[:, 1].max())
    return boxes


def overlay_availability(dataset: str, video_id: str) -> dict[str, str]:
    """`{source key: where its detections come from}` for the sources that have artefacts for this clip."""
    avail: dict[str, str] = {}
    for src in OVERLAY_SOURCES:
        spec = MODEL_BY_ID[src.model_id]
        if spec.backend == "released":
            p = find_pose_json(spec, dataset, video_id)
            if p is not None:
                avail[src.key] = f"released AlphaPose tracked poses ({p.name})"
        else:
            p = find_cache_npz(spec, dataset, video_id)
            if p is not None:
                avail[src.key] = f"detector cache {p.parent.parent.parent.name}/cache/{p.parent.name}"
    return avail


def load_overlay_detections(source_key: str, dataset: str, video_id: str, frame_idx: int,
                            min_conf: float = LOW_CONF) -> dict | None:
    """Boxes (N,4), confidences (N,), keypoints (N,17,3) of one source on one frame, keeping every detection with conf >= `min_conf`.

    Detector caches store all detections with conf >= 0.01, so the 0.05..0.20 'recovery' boxes of the two-stage tracker are included.
    Released AlphaPose has no per-box confidence: its pose score is used. Returns `None` when the artefact does not exist.
    """
    spec = MODEL_BY_ID[OVERLAY_BY_KEY[source_key].model_id]
    if spec.backend == "released":
        p = find_pose_json(spec, dataset, video_id)
        if p is None:
            return None
        entries = _cached_tracked_poses(str(p)).get(frame_idx, [])
        kpts = np.array([e["keypoints"] for e in entries], dtype=np.float32).reshape(-1, 17, 3)
        conf = np.array([e["score"] for e in entries], dtype=np.float32)
        return {"boxes": boxes_from_keypoints(kpts), "conf": conf, "keypoints": kpts,
                "track_ids": [e["track_id"] for e in entries], "origin": "released AlphaPose tracked poses"}
    p = find_cache_npz(spec, dataset, video_id)
    if p is None:
        return None
    frames, _ = _cached_detection_frames(str(p))
    if not 0 <= frame_idx < len(frames):
        return {"boxes": np.zeros((0, 4), np.float32), "conf": np.zeros(0, np.float32), "keypoints": np.zeros((0, 17, 3), np.float32),
                "track_ids": None, "origin": f"{p.parent.parent.parent.name}/cache/{p.parent.name}"}
    f = frames[frame_idx]
    keep = f["conf"] >= min_conf
    return {"boxes": f["boxes"][keep].astype(np.float32), "conf": f["conf"][keep].astype(np.float32),
            "keypoints": f["keypoints"][keep].astype(np.float32), "track_ids": None, "origin": f"{p.parent.parent.parent.name}/cache/{p.parent.name}"}
