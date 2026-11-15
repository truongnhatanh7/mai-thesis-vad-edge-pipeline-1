"""Q2-P1-E08 physical Raspberry Pi 5 benchmark: ML0 YOLOv8n-pose (NCNN FP16 384x640) -> two-stage BoT-SORT (E01 locked) -> {B0, B1, P_A, P_B, P_C, P_ABC, TrajVAD-P},
each at full rate and with Cache-C1+Guard.

Reuses the E07 Pi runner (`q2_p1_e07_pi5.run_arm`: decode -> [C1] -> pose -> tracker -> scorer, the Pi5-Huang queue model, PMIC / temperature / RSS telemetry, MLflow metrics + traces)
and replaces three things: the tracker (BoT-SORT two-stage c0.20 / low0.05 / buffer30 / match0.85 instead of TrackTrack), the scorer factory (`OnlineGapScorer`: one flow call per frame for all
completed windows, the same code path for B0, B1 and every proposal so that their scorer times are comparable) and the MLflow experiment / trace names.

    python -m edgelab.analysis.q2_p1_e08_pi5 --out-dir ~/thesis-data/q2-p1-e08-pi5 [--arms a,b] [--local] [--max-clips 12]
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"

import numpy as np

from edgelab.analysis import q2_p1_e07_pi5 as pi

HOME = Path("~/thesis-data").expanduser()
CKPT_E08 = HOME / "q2-p1-e08" / "ckpt"
OUT_DIR = HOME / "q2-p1-e08-pi5"
SCORERS = ("B0", "B1", "P_A", "P_B", "P_C", "P_ABC", "TJP")
SCORER_LABEL = {"B0": "STG-NF (retrained)", "B1": "LoRaD", "P_A": "MS-CTFlow", "P_B": "RevMargin-Flow", "P_C": "IAC-Flow", "P_ABC": "MS-IAC-RevFlow", "TJP": "TrajVAD-P"}
TRACKER = (0.20, 0.05, 0.85)  # c_high, c_low, match (E01 locked two-stage)
_orig_build_scorer = pi.build_scorer


def make_botsort(c_high: float, c_low: float, match: float, buffer: int = 30, min_track_len: int = 3):
    """E01 `build_tracker_cell`: Ultralytics BoT-SORT, no GMC / ReID, new track = c_high, second stage down to c_low."""
    from ultralytics.trackers.track import TRACKER_MAP
    from ultralytics.utils import ROOT, YAML, IterableSimpleNamespace

    cfg = YAML.load(ROOT / "cfg/trackers/botsort.yaml")
    cfg.update(track_high_thresh=c_high, new_track_thresh=c_high, track_low_thresh=c_low, track_buffer=buffer, match_thresh=match, gmc_method="none", with_reid=False)
    return TRACKER_MAP[cfg["tracker_type"]](IterableSimpleNamespace(**cfg))


def build_scorer(kind: str):
    """-> object with `reset()`, `step(frame_id, {track_id: kp17x3}) -> [normality]`, `forget(live)`, `.name`, `.seg_len`."""
    if kind in ("trajvad_p", "TJP"):
        return _orig_build_scorer("trajvad_p")
    from edgelab.models.lorad_gaps import OnlineGapScorer, load_gap_flow
    from edgelab.models.lorad import load_flow

    meta = json.loads((CKPT_E08 / f"{kind}.json").read_text())
    if meta["source"] == "e06":
        flow, _, seg = load_flow(CKPT_E08 / f"{kind}.pt")
    else:
        flow, cfg = load_gap_flow(CKPT_E08 / f"{kind}.pt")
        seg = cfg["seg_len"]
    scorer = OnlineGapScorer(flow, seg, ctx=meta["ctx"], smooth=meta["smooth"], name=SCORER_LABEL[kind])
    rng = np.random.default_rng(0)
    for _ in range(3):  # warm-up: the first calls allocate
        for f in range(seg + 2):
            scorer.step(f, {0: (rng.uniform(100, 700, (17, 3))).astype(np.float32), 1: (rng.uniform(100, 700, (17, 3))).astype(np.float32)})
        scorer.reset()
    return scorer


def arm_list() -> list[dict[str, Any]]:
    model = pi.MODELS_ROOT / "exp020/ML0-640-384x640-ncnn"
    arms = []
    for sc in SCORERS:
        for cache in (False, True):
            arms.append({"name": f"ml0_botsort_{sc}_{'cacheC1guard' if cache else 'full'}", "detector": "ml0", "model_path": model, "scorer": "trajvad_p" if sc == "TJP" else sc,
                         "scorer_key": sc, "use_c1": cache, "use_guard": cache,
                         "description": f"ML0 YOLOv8n-pose (NCNN) -> BoT-SORT two-stage -> {SCORER_LABEL[sc]}" + (" + Cache-C1+Guard" if cache else " (full rate)")})
    return arms


def traced_frame(arm, k, clip, fidx, dec_ms, c1_ms, pose_ms, track_ms, score_ms, n_dets, n_tracks, n_wins, min_norm, reused, wait, scorer_name, tel):
    """One hierarchical MLflow trace per sampled frame: decode -> (C1) -> pose -> BoT-SORT -> scorer -> telemetry."""
    import mlflow
    from mlflow.entities import SpanType

    @mlflow.trace(name="e08_pi5_frame", span_type=SpanType.CHAIN)
    def _inner() -> dict:
        span = mlflow.get_current_active_span()
        if span:
            span.set_attribute("description", f"Pi 5 E08 frame trace ({arm}, clip {clip}, frame {fidx})")
        mlflow.update_current_trace(metadata={"mlflow.trace.session": f"e08-pi5-{arm}"})
        with mlflow.start_span(name="1_decode_frame", span_type=SpanType.TOOL) as s:
            s.set_inputs({"clip": clip, "frame": fidx, "global_index": k})
            s.set_outputs({"decode_ms": round(dec_ms, 3)})
        with mlflow.start_span(name="2_c1_static_check", span_type=SpanType.TOOL) as s:
            s.set_inputs({"queue_wait_ms": round(wait, 1)})
            s.set_outputs({"c1_ms": round(c1_ms, 3), "detections_reused": reused})
        with mlflow.start_span(name="3_pose_detector", span_type=SpanType.LLM) as s:
            s.set_inputs({"arm": arm, "reused": reused})
            s.set_outputs({"pose_ms": round(pose_ms, 3), "n_detections": n_dets})
        with mlflow.start_span(name="4_botsort_two_stage", span_type=SpanType.TOOL) as s:
            s.set_inputs({"n_detections_in": n_dets})
            s.set_outputs({"track_ms": round(track_ms, 3), "n_active_tracks": n_tracks})
        with mlflow.start_span(name=f"5_{scorer_name}_scorer", span_type=SpanType.LLM) as s:
            s.set_inputs({"scorer": scorer_name})
            s.set_outputs({"score_ms": round(score_ms, 3), "n_windows": n_wins, "min_normality": min_norm})
        with mlflow.start_span(name="6_pi5_hardware_telemetry", span_type=SpanType.TOOL) as s:
            s.set_inputs({"source": "vcgencmd pmic + sysfs"})
            s.set_outputs(tel)
        return {"total_ms": round(dec_ms + c1_ms + pose_ms + track_ms + score_ms, 3)}

    return _inner()


def wait_cool(target_c: float = 62.0, max_s: float = 240.0) -> float:
    """Between arms: let the SoC cool to `target_c` (same thermal start for every arm), at most `max_s`."""
    from edgelab.telemetry.pi5_telemetry import read_soc_temp_c

    t0 = time.time()
    temp = read_soc_temp_c()
    while temp > target_c and time.time() - t0 < max_s:
        time.sleep(5.0)
        temp = read_soc_temp_c()
    return float(temp)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--tracking-uri", default=os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    ap.add_argument("--arms", default="")
    ap.add_argument("--local", action="store_true", help="dry run off the Pi (no PMIC, no MLflow)")
    ap.add_argument("--max-clips", type=int, default=None)
    ap.add_argument("--max-frames-per-clip", type=int, default=None)
    ap.add_argument("--rerun", action="store_true")
    ap.add_argument("--frames-root", type=Path, default=None)
    ap.add_argument("--models-root", type=Path, default=None)
    ap.add_argument("--ckpt-root", type=Path, default=None)
    ap.add_argument("--no-cool", action="store_true")
    a = ap.parse_args()
    global CKPT_E08
    CKPT_E08 = a.ckpt_root or CKPT_E08
    pi.FRAMES_ROOT, pi.MODELS_ROOT = a.frames_root or pi.FRAMES_ROOT, a.models_root or pi.MODELS_ROOT
    pi.make_tracktrack, pi.build_scorer, pi.traced_frame = make_botsort, build_scorer, traced_frame
    arms = [x for x in arm_list() if not a.arms or x["name"] in a.arms.split(",")]
    log_mlflow = not a.local
    board = {}
    if log_mlflow:
        import mlflow

        from edgelab.telemetry.pi5_telemetry import read_board_info

        mlflow.set_tracking_uri(a.tracking_uri)
        mlflow.set_experiment("Q2-P1-E08")
        board = read_board_info()
        print(f"[Pi5] board: {json.dumps(board)}", flush=True)
    frames = pi.load_frames(a.max_clips, a.max_frames_per_clip)
    print(f"[Pi5] {len(frames)} frames, {len({f[0] for f in frames})} clips, {len(arms)} arms", flush=True)
    summaries = []
    for arm in arms:
        sf = a.out_dir / f"{arm['name']}_summary.json"
        if sf.exists() and not a.rerun:
            summaries.append(json.loads(sf.read_text()))
            print(f"[Pi5] cached {arm['name']}", flush=True)
            continue
        start_temp = None if (a.local or a.no_cool) else wait_cool()
        print(f"[Pi5] running {arm['name']} (start temp {start_temp})", flush=True)
        if log_mlflow:
            import mlflow

            with mlflow.start_run(run_name=arm["name"], description=arm["description"]) as run:
                mlflow.set_tags({"machine": "pi5", "board_model": board.get("board_model", ""), "cpu_governor": board.get("cpu_governor", ""), "arm_name": arm["name"],
                                 "detector": arm["detector"], "scorer": arm["scorer_key"], "tracker": "botsort_two_stage", "experiment_id": "Q2-P1-E08"})
                mlflow.log_params({"arm": arm["name"], "detector": arm["detector"], "scorer": arm["scorer_key"], "use_c1": arm["use_c1"], "use_guard": arm["use_guard"],
                                   "threads": 4, "frames": len(frames), "tracker_cfg": ",".join(map(str, TRACKER)), "start_temp_c": start_temp})
                res = pi.run_arm(arm, frames, a.out_dir, a.local, TRACKER, True)
                tel = res["telemetry"]
                flat = {k: v for k, v in res.items() if isinstance(v, (int, float)) and not isinstance(v, bool) and np.isfinite(v)}
                flat.update({f"tel_{k}": v for k, v in tel.items() if isinstance(v, (int, float)) and not isinstance(v, bool) and np.isfinite(v)})
                mlflow.log_metrics(flat)
                for suffix in ("summary.json", "telemetry.json", "trace.npz", "raw.npz"):
                    mlflow.log_artifact(str(a.out_dir / f"{arm['name']}_{suffix}"))
                res["mlflow_run_id"], res["start_temp_c"] = run.info.run_id, start_temp
                (a.out_dir / f"{arm['name']}_summary.json").write_text(json.dumps(res, indent=2))
        else:
            res = pi.run_arm(arm, frames, a.out_dir, a.local, TRACKER, False)
        summaries.append(res)
        tel = res["telemetry"]
        print(f"[Pi5] DONE {arm['name']}: processed {res['frames_processed']}/{res['frames']} service {res['service_ms_mean']:.1f} ms "
              f"(pose {res['pose_ms_mean']:.1f} track {res['track_ms_mean']:.2f} score {res['score_ms_mean']:.2f}) wait_p99 {res['queue_wait_ms_p99']:.0f} ms "
              f"P={tel['power_total_w_mean']} W T={tel['temp_c_mean']} C RSS={tel['rss_mb_peak']} MB throttled={tel.get('throttled_codes_seen')}", flush=True)
    if log_mlflow:
        import mlflow

        mlflow.flush_trace_async_logging()
    (a.out_dir / "pi5_e08_all.json").write_text(json.dumps({"board": board, "arms": summaries}, indent=2))


if __name__ == "__main__":
    main()
