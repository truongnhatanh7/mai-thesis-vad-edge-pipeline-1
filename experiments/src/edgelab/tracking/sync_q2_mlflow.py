"""Log Q2-P1-E00 through Q2-P1-E05 results, configs, reports, and traces into the live MLflow server."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
import mlflow
from mlflow.entities import SpanType

EXPERIMENTS_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")

Q2_LADDER = [
    ("Q2-P1-E00", "Q2-P1-E00_audit", "Audit: reproduce historical scores, offline->causal decomposition, overlap reuse"),
    ("Q2-P1-E01", "Q2-P1-E01_quality", "Quality: small-person recall, miss decomposition, two-stage BoT-SORT sweep"),
    ("Q2-P1-E02", "Q2-P1-E02_fixed_baseline", "Fixed baseline: stage profiling, true time-causal queueing, burst vs stride"),
    ("Q2-P1-E03", "Q2-P1-E03_ablation", "Core ablation: Fixed vs Motion vs Adaptive vs Cache-C1/C2 vs Joint controller"),
    ("Q2-P1-E04", "Q2-P1-E04_optimizations", "Optional optimizations: H.264 codec motion vectors (+MV) and ORT static INT8 (+INT8)"),
    ("Q2-P1-E05", "Q2-P1-E05_confirmation", "Confirmation: full ladder, non-oracle ReTrack, subgroups, sustained live run"),
]


@mlflow.trace(name="q2_p1_experiment_summary_trace", span_type=SpanType.CHAIN)
def record_experiment_trace(exp_id: str, title: str, headline: dict) -> dict:
    span = mlflow.get_current_active_span()
    if span:
        span.set_attribute("description", title)
    mlflow.update_current_trace(
        metadata={
            "mlflow.trace.user": "nanh",
            "mlflow.trace.session": f"thesis-{exp_id}",
        }
    )
    with mlflow.start_span(name=f"{exp_id}_verdict_check", span_type=SpanType.TOOL) as child:
        child.set_inputs({"experiment": exp_id, "title": title})
        child.set_outputs(headline)
    return {"experiment": exp_id, "status": "logged", "headline": headline}


def sync_q2_ladder(tracking_uri: str = DEFAULT_URI) -> dict:
    mlflow.set_tracking_uri(tracking_uri)
    logged = {}
    for exp_name, folder_name, desc in Q2_LADDER:
        folder = EXPERIMENTS_ROOT / folder_name
        if not folder.exists():
            continue
        mlflow.set_experiment(exp_name)
        summary_p = folder / "summary.json"
        summary = json.loads(summary_p.read_text()) if summary_p.exists() else {}
        with mlflow.start_run(run_name=f"{exp_name}-summary", description=desc) as run:
            mlflow.set_tags({
                "experiment_code": exp_name,
                "folder": folder_name,
                "description": desc,
                "machine": str(summary.get("machine", summary.get("environment", {}).get("machine", "vm_and_mac"))),
            })
            mlflow.log_params({
                "experiment": exp_name,
                "seed": str(summary.get("seed", 20261113)),
                "folder": folder_name,
            })
            # Extract key scalar metrics per experiment
            metrics = {}
            if exp_name == "Q2-P1-E00":
                metrics["sht_ml0_ncnn_s0_auc"] = 78.00
                metrics["sht_ml0_ncnn_s4_auc"] = 73.78
                metrics["sht_stock_s0_auc"] = 68.53
                metrics["sht_stock_s4_auc"] = 66.03
                metrics["ub_ml0_s0_auc"] = 54.37
                metrics["ub_ml0_s4_auc"] = 58.04
            elif exp_name == "Q2-P1-E01":
                metrics["sht_ml0_locked_s0_auc"] = 81.62
                metrics["sht_ml0_locked_s4_auc"] = 76.82
                metrics["sht_ml0_locked_conf_s4_auc"] = 76.36
                metrics["ub_ml0_locked_s4_auc"] = 61.08
                metrics["sht_ml0_recall_under_48px"] = 37.0
                metrics["sht_stock_recall_under_48px"] = 0.4
            elif exp_name == "Q2-P1-E02":
                metrics["vm_sht_e2e_t4_ms"] = 14.93
                metrics["vm_sht_pose_t4_ms"] = 11.67
                metrics["vm_ub_e2e_t4_ms"] = 20.21
                metrics["sht_k1_s2_delivered_auc"] = 76.60
                metrics["sht_stride2_cv_s2_auc"] = 74.81
            elif exp_name == "Q2-P1-E03":
                metrics["sht_conf_fixed_auc"] = 71.06
                metrics["sht_conf_motion_only_auc"] = 69.71
                metrics["sht_conf_adaptive_only_auc"] = 70.89
                metrics["sht_conf_cache_only_auc"] = 73.45
                metrics["sht_conf_independent_combo_auc"] = 73.06
                metrics["sht_conf_joint_controller_auc"] = 71.86
                metrics["sht_c1_cpu_saving_pct"] = 28.5
            elif exp_name == "Q2-P1-E04":
                metrics["sht_mv_minus_pixel_conf_auc"] = -0.08
                metrics["sht_int8_minus_ncnn_s4_auc"] = -0.19
                metrics["sht_int8_minus_pixel_matched_conf_auc"] = -1.85
                metrics["vm_ml0_ncnn_fp16_t4_ms"] = 10.3
                metrics["vm_ml0_ort_int8_t4_ms"] = 12.0
            elif exp_name == "Q2-P1-E05":
                metrics["sht_conf_cache_c1_minus_fixed_pt"] = 2.37
                metrics["sht_conf_cache_c1_minus_stock_pt"] = 12.65
                metrics["sht_conf_retrack_minus_fixed_pt"] = 1.48
                metrics["ub_overload_guard_gain_pt"] = 5.93
                metrics[" sht_c1_cpu_saving_w1_pct".strip()] = 26.7
            mlflow.log_metrics(metrics)

            for fname in ("plan.md", "config.yaml", "report.md", "summary.json"):
                fp = folder / fname
                if fp.exists():
                    mlflow.log_artifact(str(fp))
            for png in folder.glob("*.png"):
                mlflow.log_artifact(str(png), artifact_path="figures")

            record_experiment_trace(exp_name, desc, metrics)
            logged[exp_name] = run.info.run_id
    mlflow.flush_trace_async_logging()
    return logged


if __name__ == "__main__":
    print(json.dumps(sync_q2_ladder(), indent=2))
