"""Populate and document MLflow Model Registry, Experiments, and Traces for the thesis pipeline."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
import mlflow
from mlflow.tracking import MlflowClient

EXPERIMENTS_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")

MODELS_SPEC = [
    {
        "name": "edgelab-yolov8n-pose-stock",
        "description": (
            "Stock Ultralytics YOLOv8n-pose (3.29M params, 9.2 GFLOPs @ 640x640). "
            "Used as the baseline pose estimator in Huang et al. (IEEE ESL 2026) and our E00-E05 control. "
            "Trained on standard COCO-pose labels (where keypoint-less small people are dropped), resulting in "
            "0.4% recall below 48 px height on ShanghaiTech and 68.53% offline / 66.03% causal AUC."
        ),
        "tags": {
            "task": "pose_estimation",
            "architecture": "YOLOv8n-pose",
            "paper_reference": "Huang et al. IEEE ESL 2026 (10.1109/LES.2025.3618635)",
            "sht_offline_auc_640": "68.53",
            "sht_causal_auc_640": "66.03",
            "sht_offline_auc_320": "57.28",
            "sht_recall_under_48px": "0.004",
        },
        "versions": [
            {
                "stage_label": "PyTorch-FP32-640 (Huang RPi baseline)",
                "runtime": "pytorch_cpu",
                "imgsz": "640x640 / 384x640",
                "artifact_path": str(EXPERIMENTS_ROOT / "artifacts/EXP-001/models/yolov8n-pose.pt"),
                "sha256": "c6fa93dd1ee4a2c18c900a45c1d864a1c6f7aba75d84f91648a30b7fb641d212",
                "description": "Stock PyTorch FP32 checkpoint (`yolov8n-pose.pt`). Huang et al. `RPi` config runs this at 640x640 (reported 3.2 FPS).",
            },
            {
                "stage_label": "NCNN-FP16-320 (Huang RPi* baseline)",
                "runtime": "ncnn_fp16",
                "imgsz": "320x320 / 192x320",
                "artifact_path": str(Path("~/thesis-data/vm-exchange/models/yolov8n-pose-320-320x320-ncnn/model.ncnn.bin").expanduser()),
                "sha256": "ncnn_320x320_export",
                "description": "Stock NCNN FP16 320x320 export (`RPi*` config in Huang et al., reported 17.7 FPS; offline AUC drops to 57.28%).",
            },
            {
                "stage_label": "NCNN-FP16-640 (Edge Stock Control)",
                "runtime": "ncnn_fp16",
                "imgsz": "384x640",
                "artifact_path": str(Path("~/thesis-data/vm-exchange/models/yolov8n-pose-640-384x640-ncnn/model.ncnn.bin").expanduser()),
                "sha256": "f374722a7d557486344f1dcc1f21236f4618f5ae6a0e9d173b8b416b49f5fd5f",
                "description": "Stock NCNN FP16 384x640 export for fair runtime comparison against ML0 at 640p.",
            },
        ],
    },
    {
        "name": "edgelab-yolov8n-pose-ML0",
        "description": (
            "Partial-Label COCO Fine-Tuned YOLOv8n-pose (`ML0`, EXP-020 / Q2-P1-E01..E05 locked winner). "
            "Identical architecture and FLOPs (1.00x) to stock YOLOv8n-pose, fine-tuned for 4 epochs on COCO train2017 "
            "(64,115 images) keeping keypoint-less small/crowded people as box-only targets with masked keypoint loss. "
            "Lifts <48px recall from 0.4% to 37.0% (48-96px from 38.9% to 85.6%) and ShanghaiTech causal AUC from 66.03% "
            "to 76.82% (81.62% offline) when paired with two-stage BoT-SORT."
        ),
        "tags": {
            "task": "pose_estimation",
            "architecture": "YOLOv8n-pose (1.00x stock FLOPs)",
            "training_data": "COCO train2017 only (zero ShanghaiTech/UBnormal leakage)",
            "sht_offline_auc_640": "81.62",
            "sht_causal_auc_640": "76.82",
            "sht_recall_under_48px": "0.370",
            "status": "LOCKED_WINNER_Q2_P1",
        },
        "versions": [
            {
                "stage_label": "PyTorch-FP32-640",
                "runtime": "pytorch_cpu_mps",
                "imgsz": "384x640",
                "artifact_path": str(EXPERIMENTS_ROOT / "artifacts/EXP-020/ML0.pt"),
                "sha256": "e8bceab238259fc7883e9070f4ceeec79833a312d096f6a851c2d0ac9d5c52d8",
                "description": "ML0 PyTorch checkpoint (`artifacts/EXP-020/ML0.pt`).",
            },
            {
                "stage_label": "NCNN-FP16-640 (Production Pi 5 Lock)",
                "runtime": "ncnn_fp16",
                "imgsz": "384x640",
                "artifact_path": str(Path("~/thesis-data/vm-exchange/models/exp020/ML0-640-384x640-ncnn/model.ncnn.bin").expanduser()),
                "sha256": "87d55ebf799739ec4d9fc0f34e4391c483bc6c0cd1ca39be121ba71b84a0c8ee",
                "description": "ML0 NCNN FP16 384x640 export deployed on Raspberry Pi 5.",
            },
            {
                "stage_label": "ORT-INT8-640 (E04 QDQ Calibration Control)",
                "runtime": "onnxruntime_int8",
                "imgsz": "384x640",
                "artifact_path": str(Path("~/thesis-data/q2-p1-e04/models/ML0-640-384x640-int8.onnx").expanduser()),
                "sha256": "ml0_ort_int8_qdq_200calib",
                "description": "ML0 ONNX Runtime static QDQ INT8 (200 seeded SHT normal training frames, fp32 box decode). Evaluated in E04.",
            },
        ],
    },
    {
        "name": "edgelab-yolov8n-pose-MLP-p2",
        "description": (
            "Partial-Label + P2 Stride-4 High-Resolution Head YOLOv8n-pose (`MLP`, 3.455M params, 1.92x FLOPs). "
            "Tested in EXP-020 and Q2-P1-E01; rejected for edge deployment because 1.82x slower detector latency yields "
            "only +0.30 pt confirmation AUC gain over ML0."
        ),
        "tags": {
            "task": "pose_estimation",
            "architecture": "YOLOv8n-pose + P2 head (1.92x FLOPs)",
            "sht_offline_auc_640": "79.24",
            "sht_causal_auc_640": "74.69",
            "status": "REJECTED_COST_BENEFIT",
        },
        "versions": [
            {
                "stage_label": "NCNN-FP16-640",
                "runtime": "ncnn_fp16",
                "imgsz": "384x640",
                "artifact_path": str(Path("~/thesis-data/vm-exchange/models/exp020/MLP-640-384x640-ncnn/model.ncnn.bin").expanduser()),
                "sha256": "670d38922e69f6233b1190b57346709f31e57570f0d43ceed4640b149c29a827",
                "description": "MLP NCNN FP16 384x640 export.",
            }
        ],
    },
    {
        "name": "edgelab-stgnf-scorer",
        "description": (
            "Frozen Spatio-Temporal Graph Normalizing Flow (`STG-NF`, Hirschorn & Avidan ICCV 2023, ~0.6K params). "
            "Scores 24-frame (ShanghaiTech) or 16-frame (UBnormal) 18-keypoint human pose trajectories via exact "
            "log-likelihood evaluation. Used as the frozen anomaly scorer across all E00-E05 and Pi 5 experiments."
        ),
        "tags": {
            "task": "skeleton_anomaly_scoring",
            "architecture": "STG-NF (ActNorm + Permutation + Affine ST-GCN coupling)",
            "paper_reference": "Hirschorn & Avidan ICCV 2023 / Huang et al. IEEE ESL 2026 baseline",
            "sht_alphapose_offline_auc": "85.94",
            "ub_alphapose_offline_auc": "71.78",
        },
        "versions": [
            {
                "stage_label": "ShanghaiTech-Unsupervised-L24",
                "runtime": "pytorch_cpu",
                "imgsz": "18x24",
                "artifact_path": str(EXPERIMENTS_ROOT / "third_party/STG-NF/checkpoints/ShanghaiTech_85_9.tar"),
                "sha256": "4950205cb5cdd3c4827e3214ac6c93dc7050c0d059c8325907cbe977360a70ca",
                "description": "Official ShanghaiTech unsupervised STG-NF checkpoint (seg_len=24, 85.94% AUC on AlphaPose).",
            },
            {
                "stage_label": "UBnormal-Unsupervised-L16",
                "runtime": "pytorch_cpu",
                "imgsz": "18x16",
                "artifact_path": str(EXPERIMENTS_ROOT / "third_party/STG-NF/checkpoints/UBnormal_unsupervised_71_8.tar"),
                "sha256": "1e054a5172f118a0ffa03949840961a6ebc6539c7654959910b6e123bcb7ade3",
                "description": "Official UBnormal unsupervised STG-NF checkpoint (seg_len=16, 71.78% AUC on AlphaPose).",
            },
        ],
    },
]


def register_all_models(tracking_uri: str = DEFAULT_URI) -> dict:
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)
    mlflow.set_experiment("Model-Registry-Catalog")
    summary = {}
    for spec in MODELS_SPEC:
        name = spec["name"]
        try:
            client.create_registered_model(name=name, description=spec["description"], tags=spec["tags"])
        except Exception:
            client.update_registered_model(name=name, description=spec["description"])
            for k, v in spec["tags"].items():
                client.set_registered_model_tag(name, k, str(v))

        existing_versions = client.search_model_versions(f"name='{name}'")
        existing_labels = {v.tags.get("stage_label") for v in existing_versions if v.tags}
        created = []
        for ver in spec["versions"]:
            if ver["stage_label"] in existing_labels:
                continue
            with mlflow.start_run(run_name=f"register-{name}-{ver['stage_label']}") as run:
                mlflow.set_tags({
                    "model_name": name,
                    "stage_label": ver["stage_label"],
                    "runtime": ver["runtime"],
                    "sha256": ver["sha256"],
                })
                mlflow.log_params({
                    "model_name": name,
                    "stage_label": ver["stage_label"],
                    "runtime": ver["runtime"],
                    "imgsz": ver["imgsz"],
                    "artifact_path": ver["artifact_path"],
                    "sha256": ver["sha256"],
                })
                with tempfile.TemporaryDirectory() as tmp:
                    meta_path = Path(tmp) / "model_card.json"
                    meta_path.write_text(json.dumps({**spec, "version_info": ver}, indent=2))
                    mlflow.log_artifact(str(meta_path), artifact_path="model_metadata")
                mv = client.create_model_version(
                    name=name,
                    source=f"runs:/{run.info.run_id}/model_metadata",
                    run_id=run.info.run_id,
                    description=ver["description"],
                    tags={
                        "stage_label": ver["stage_label"],
                        "runtime": ver["runtime"],
                        "imgsz": ver["imgsz"],
                        "sha256": ver["sha256"],
                    },
                )
                created.append(mv.version)
        summary[name] = {"created_versions": created, "total_versions": len(client.search_model_versions(f"name='{name}'"))}
    return summary


if __name__ == "__main__":
    print(json.dumps(register_all_models(), indent=2))
