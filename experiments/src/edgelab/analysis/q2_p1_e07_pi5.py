"""Q2-P1-E07 physical Raspberry Pi 5 benchmark: {YOLO26n-pose, ML0} (NCNN FP16) -> TrackTrack -> {LoRaD, TrajVAD-P}, with / without Cache-C1+Guard.

Runs ON the Pi (Cortex-A76, 8 GB), streams step metrics, artifacts and MLflow traces to the Mac server through the SSH reverse tunnel
(MLFLOW_TRACKING_URI=http://127.0.0.1:5000). One stream = the 12 ShanghaiTech clips on disk (240 frames each, 2,880 frames) arriving at 24 FPS,
one worker (decode -> [C1 check] -> pose -> TrackTrack -> scorer), the Pi5-Huang queue model (arrival = k / 24 s, start = max(arrival, worker free)).

Per arm the runner saves `<arm>_summary.json`, `<arm>_telemetry.json`, `<arm>_trace.npz` and `<arm>_raw.npz` (per clip, per frame: the minimum
normality of the windows completed at that frame, `ontime` = delivered within 2 s, `all` = regardless of delay) so that the delivered real-time AUC
is computed on the Mac with the E00 S4 landing and the train-calibrated sentinel (`q2_p1_e07 pi-report`).

    python -m edgelab.analysis.q2_p1_e07_pi5 --out-dir ~/thesis-data/q2-p1-e07-pi5 [--arms a,b] [--local] [--max-clips 12]
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"

import cv2
import numpy as np
import torch

HOME = Path("~/thesis-data").expanduser()
FRAMES_ROOT = HOME / "frames"
MODELS_ROOT = HOME / "models"
CKPT_ROOT = HOME / "q2-p1-e07" / "ckpt"
FPS = 24.0
LATE_MS = 2000.0
TRACK_LOW, TRACK_HIGH, TRACK_MATCH = 0.05, 0.20, 0.80  # filled from the E07 T1 selection by `arm_list` (defaults = selected cell)


def make_tracktrack(c_high: float, c_low: float, match: float, buffer: int = 30, min_track_len: int = 3):
    from ultralytics.trackers.track import TRACKER_MAP
    from ultralytics.utils import ROOT, YAML, IterableSimpleNamespace

    cfg = YAML.load(ROOT / "cfg/trackers/tracktrack.yaml")
    cfg.update(track_high_thresh=c_high, new_track_thresh=c_high, track_low_thresh=c_low, track_buffer=buffer, match_thresh=match,
               min_track_len=min_track_len, gmc_method="none", with_reid=False)
    return TRACKER_MAP["tracktrack"](IterableSimpleNamespace(**cfg))


def c1_thumbnail(bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (40, 24), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0


def c1_static_check(prev_thumb: np.ndarray, cur_thumb: np.ndarray, prev_boxes: np.ndarray, h_img: int = 480, w_img: int = 856) -> bool:
    """Locked E03 SHT static-validity rule (level 3, q60) -> True when the previous detections may be reused."""
    diff = np.abs(cur_thumb - prev_thumb)
    mask = np.zeros_like(diff, dtype=bool)
    mask[:3, :] = mask[-3:, :] = True
    mask[:, :4] = mask[:, -4:] = True
    ext_p95 = float(np.percentile(diff[mask], 95))
    roi_max = 0.0
    for box in prev_boxes:
        x1 = max(0, min(39, int(box[0] / w_img * 40)))
        y1 = max(0, min(23, int(box[1] / h_img * 24)))
        x2 = max(x1 + 1, min(40, int(np.ceil(box[2] / w_img * 40))))
        y2 = max(y1 + 1, min(24, int(np.ceil(box[3] / h_img * 24))))
        roi_max = max(roi_max, float(diff[y1:y2, x1:x2].mean()))
    return (roi_max <= 0.0225) and (ext_p95 <= 0.0350)


def load_frames(max_clips: int | None = None, max_frames_per_clip: int | None = None) -> list[tuple[str, int, Path]]:
    items = []
    for clip_dir in sorted(p for p in FRAMES_ROOT.iterdir() if p.is_dir())[:max_clips]:
        jpgs = sorted(clip_dir.glob("*.jpg"), key=lambda p: int(p.stem))[:max_frames_per_clip]
        items += [(clip_dir.name, int(j.stem), j) for j in jpgs]
    return items


def build_scorer(kind: str):
    """-> object with `reset()`, `step(frame_id, {track_id: kp17x3}) -> [normality]`."""
    if kind == "lorad":
        from edgelab.models.lorad import OnlineLoRaD, load_flow

        meta = json.loads((CKPT_ROOT / "lorad.json").read_text())
        flow, norm, seg_len = load_flow(CKPT_ROOT / "lorad.pt")
        online = OnlineLoRaD(flow, norm, seg_len, mode=meta["mode"])
        _L = seg_len

        class _LoRaD:
            name, seg_len = "LoRaD", _L

            def reset(self):
                online.tracks.clear()

            def step(self, frame_id, poses):
                out = []
                for tid, kp in poses.items():
                    v = online.update(tid, frame_id, kp)
                    if v is not None:
                        out.append(v)
                return out

            def forget(self, live):
                online.forget(live)

        return _LoRaD()
    from edgelab.models.trajvad import OnlineTrajVAD

    online = OnlineTrajVAD.from_checkpoint(CKPT_ROOT / "trajvad_p.pt")

    class _TrajP:
        name, seg_len = "TrajVAD-P", online.L

        def reset(self):
            online.tracks.clear()

        def step(self, frame_id, poses):
            return [v for _, v in online.update_frame(frame_id, poses)] if poses else []

        def forget(self, live):
            online.forget(live)

    return _TrajP()


def arm_list() -> list[dict[str, Any]]:
    arms = []
    for det, det_label, model in (("yolo26", "YOLO26n-pose (NCNN o2m head)", MODELS_ROOT / "yolo26n-pose-640-384x640-ncnn"),
                                  ("ml0", "ML0 YOLOv8n-pose (NCNN)", MODELS_ROOT / "exp020/ML0-640-384x640-ncnn")):
        for scorer, sc_label in (("lorad", "LoRaD"), ("trajvadp", "TrajVAD-P")):
            for cache in (False, True):
                arms.append({"name": f"{det}_tracktrack_{scorer}_{'cacheC1guard' if cache else 'full'}", "detector": det, "model_path": model,
                             "scorer": "trajvad_p" if scorer == "trajvadp" else "lorad", "use_c1": cache, "use_guard": cache,
                             "description": f"{det_label} -> TrackTrack -> {sc_label}" + (" + Cache-C1+Guard" if cache else " (full rate)")})
    return arms


def run_arm(arm: dict[str, Any], frames: list[tuple[str, int, Path]], out_dir: Path, local: bool, tracker_cfg: tuple, log_mlflow: bool) -> dict:
    from edgelab.pipeline.ncnn_pose import NcnnPose
    from edgelab.pose.replay import Detections

    threads = int(arm.get("threads", 4))
    os.environ["OMP_NUM_THREADS"] = str(threads)
    torch.set_num_threads(threads)
    c_high, c_low, match = tracker_cfg
    det = NcnnPose(arm["model_path"], threads=threads, fp16=True)
    scorer = build_scorer(arm["scorer"])
    use_c1, use_guard = bool(arm["use_c1"]), bool(arm["use_guard"])
    sample = cv2.imread(str(frames[0][2]))
    for _ in range(5):  # detector warm-up
        det(sample, conf=c_low)
    if arm["scorer"] == "lorad":  # scorer warm-up (first call allocates)
        scorer.step(0, {0: np.zeros((17, 3), np.float32) + 1.0})
        scorer.reset()

    if local:
        class _Mon:
            samples: list = []

            def start(self):
                return self

            def stop(self):
                return {k: float("nan") for k in ("temp_c_mean", "temp_c_max", "power_total_w_mean", "power_total_w_max", "power_core_w_mean",
                                                  "rss_mb_peak", "sys_cpu_pct_mean", "sys_cpu_pct_peak")}

            def sample_once(self):
                import psutil

                return {"temp_c": float("nan"), "power_total_w": float("nan"), "power_core_w": float("nan"),
                        "rss_mb": psutil.Process().memory_info().rss / 2**20, "sys_cpu_pct": float("nan"), "throttled_hex": "n/a"}

        monitor = _Mon()
    else:
        from edgelab.telemetry.pi5_telemetry import Pi5TelemetryMonitor

        monitor = Pi5TelemetryMonitor(interval_s=0.20)
    monitor.start()

    dt_ms = 1000.0 / FPS
    lists: dict[str, list] = {k: [] for k in ("decode", "c1", "pose", "track", "score", "service", "wait")}
    n_dets, n_tracks, n_wins = [], [], []
    clips: dict[str, int] = {}
    for c, f, _ in frames:
        clips[c] = max(clips.get(c, 0), f + 1)
    raw_ontime = {c: np.full(n, np.nan) for c, n in clips.items()}
    raw_all = {c: np.full(n, np.nan) for c, n in clips.items()}
    c1_hits = sheds = late_frames = 0
    cur_clip, tracker, prev_thumb = "", None, None
    prev_boxes, prev_scores, prev_kps = np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros((0, 17, 3), np.float32)
    c1_age, guard_active, worker_free = 999, False, 0.0
    last_seen: dict[int, int] = {}
    wall0, cpu0 = time.perf_counter(), time.process_time()
    for k, (clip, fidx, path) in enumerate(frames):
        if clip != cur_clip:
            cur_clip, tracker, prev_thumb, c1_age = clip, make_tracktrack(c_high, c_low, match), None, 999
            scorer.reset()
            last_seen.clear()
        arrival = k * dt_ms
        start = max(arrival, worker_free)
        wait = start - arrival
        lists["wait"].append(wait)
        if use_guard:
            if not guard_active and wait > 400.0:
                guard_active = True
            elif guard_active and wait < 150.0:
                guard_active = False
            if guard_active and fidx % 2 == 1:  # load shedding: drop every other frame while the queue is long
                sheds += 1
                for key in ("decode", "c1", "pose", "track", "score", "service"):
                    lists[key].append(0.0)
                n_dets.append(0), n_tracks.append(0), n_wins.append(0)
                continue
        t0 = time.perf_counter()
        img = cv2.imread(str(path))
        t1 = time.perf_counter()
        c1_ms, reused = 0.0, False
        if use_c1:
            tc = time.perf_counter()
            thumb = c1_thumbnail(img)
            if prev_thumb is not None and c1_age < 3 and (fidx % 12 != 0):
                reused = c1_static_check(prev_thumb, thumb, prev_boxes, img.shape[0], img.shape[1])
            prev_thumb = thumb
            c1_ms = (time.perf_counter() - tc) * 1e3
        tp = time.perf_counter()
        if reused:
            boxes, scores, kps = prev_boxes, prev_scores, prev_kps
            c1_age += 1
            c1_hits += 1
            pose_ms = 0.0
        else:
            boxes, scores, kps, _ = det(img, conf=c_low)
            pose_ms = (time.perf_counter() - tp) * 1e3
            prev_boxes, prev_scores, prev_kps = boxes, scores, kps
            c1_age = 0
        tt = time.perf_counter()
        keep = scores >= c_low
        kb, ks, kk = boxes[keep], scores[keep], kps[keep]
        rows = tracker.update(Detections(kb, ks, np.zeros(len(ks), np.float32)), None)
        poses = {int(r[4]): kk[int(r[7])].astype(np.float32) for r in rows}
        track_ms = (time.perf_counter() - tt) * 1e3
        ts = time.perf_counter()
        norms = scorer.step(fidx, poses)
        for tid in poses:
            last_seen[tid] = fidx
        if fidx % 60 == 0:
            scorer.forget({t for t, f in last_seen.items() if fidx - f <= 60})
        score_ms = (time.perf_counter() - ts) * 1e3
        dec_ms = (t1 - t0) * 1e3
        service = dec_ms + c1_ms + pose_ms + track_ms + score_ms
        worker_free = start + service
        late = (worker_free - arrival) > LATE_MS
        late_frames += int(late and len(norms) > 0)
        if norms:
            raw_all[clip][fidx] = min(norms)
            if not late:
                raw_ontime[clip][fidx] = min(norms)
        for key, v in (("decode", dec_ms), ("c1", c1_ms), ("pose", pose_ms), ("track", track_ms), ("score", score_ms), ("service", service)):
            lists[key].append(v)
        n_dets.append(int(len(ks))), n_tracks.append(len(poses)), n_wins.append(len(norms))
        if k % 60 == 0 or k == len(frames) - 1:
            snap = monitor.sample_once()
            if log_mlflow:
                import mlflow

                mlflow.log_metrics({"live_service_ms": round(service, 2), "live_pose_ms": round(pose_ms, 2), "live_track_ms": round(track_ms, 2),
                                    "live_score_ms": round(score_ms, 2), "live_queue_wait_ms": round(wait, 2), "live_temp_c": snap["temp_c"],
                                    "live_power_total_w": snap["power_total_w"], "live_rss_mb": snap["rss_mb"],
                                    "live_sys_cpu_pct": snap["sys_cpu_pct"]}, step=k)
                if k in (0, len(frames) // 2, len(frames) - 1) or k % 600 == 0:
                    traced_frame(arm["name"], k, clip, fidx, dec_ms, c1_ms, pose_ms, track_ms, score_ms, len(ks), len(poses), len(norms),
                                 min(norms) if norms else None, reused, wait, scorer.name,
                                 {kk_: snap[kk_] for kk_ in ("temp_c", "power_total_w", "rss_mb", "sys_cpu_pct", "throttled_hex")})
    wall = time.perf_counter() - wall0
    cpu = time.process_time() - cpu0
    tel = monitor.stop()
    n_f = len(frames)
    processed = [s for s in lists["service"] if s > 0]
    mean_proc = float(np.mean(processed)) if processed else float("nan")
    sust = bool(mean_proc < dt_ms and np.percentile(lists["wait"], 99) <= LATE_MS)
    q = lambda key, p: round(float(np.percentile(lists[key], p)), 3)  # noqa: E731
    summary = {
        "arm": arm["name"], "description": arm["description"], "detector": arm["detector"], "scorer": arm["scorer"], "threads": threads,
        "use_c1": use_c1, "use_guard": use_guard, "frames": n_f, "frames_processed": len(processed), "wall_total_s": round(wall, 2),
        "cpu_total_s": round(cpu, 2), "cpu_ms_per_video_s": round(cpu * 1000.0 / (n_f / FPS), 1),
        "throughput_fps_processed": round(1000.0 / mean_proc, 2) if processed else None,
        "delivered_fps": round(len(processed) / wall, 2), "Sustainable_24fps": sust,
        "decode_ms_mean": round(float(np.mean([s for s in lists["decode"] if s > 0])), 3),
        "c1_check_ms_mean": round(float(np.mean(lists["c1"])), 3),
        "pose_ms_mean": round(float(np.mean([s for s in lists["pose"] if s > 0])) if any(lists["pose"]) else 0.0, 3),
        "pose_ms_p95": q("pose", 95), "track_ms_mean": round(float(np.mean([s for s in lists["track"] if s > 0])), 3),
        "track_ms_p95": q("track", 95), "score_ms_mean": round(float(np.mean([s for s in lists["score"] if s > 0])), 3),
        "score_ms_p95": q("score", 95), "service_ms_mean": round(mean_proc, 3), "service_ms_p50": round(float(np.median(processed)), 3),
        "service_ms_p95": round(float(np.percentile(processed, 95)), 3), "service_ms_p99": round(float(np.percentile(processed, 99)), 3),
        "queue_wait_ms_p50": q("wait", 50), "queue_wait_ms_p95": q("wait", 95), "queue_wait_ms_p99": q("wait", 99),
        "queue_wait_ms_max": round(float(np.max(lists["wait"])), 2), "c1_hit_rate_pct": round(100.0 * c1_hits / n_f, 2),
        "guard_shed_pct": round(100.0 * sheds / n_f, 2), "late_frames_with_windows": late_frames,
        "mean_detections_per_frame": round(float(np.mean(n_dets)), 2), "mean_tracks_per_frame": round(float(np.mean(n_tracks)), 2),
        "windows_delivered_ontime": int(sum(np.isfinite(v).sum() for v in raw_ontime.values())),
        "windows_delivered_all": int(sum(np.isfinite(v).sum() for v in raw_all.values())),
        "telemetry": tel, "energy_j_per_processed_frame": round(tel["power_total_w_mean"] * mean_proc / 1000.0, 4),
        "energy_j_per_stream": round(tel.get("energy_total_joules", float("nan")), 2) if "energy_total_joules" in tel else None,
        "tracker": {"kind": "tracktrack", "c_high": c_high, "c_low": c_low, "match": match},
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    name = arm["name"]
    (out_dir / f"{name}_summary.json").write_text(json.dumps(summary, indent=2))
    (out_dir / f"{name}_telemetry.json").write_text(json.dumps(monitor.samples, indent=2))
    np.savez_compressed(out_dir / f"{name}_trace.npz", **{f"{k}_ms": np.array(v, np.float32) for k, v in lists.items()},
                        n_dets=np.array(n_dets, np.int32), n_tracks=np.array(n_tracks, np.int32), n_wins=np.array(n_wins, np.int32))
    np.savez_compressed(out_dir / f"{name}_raw.npz", **{f"ontime__{c}": v for c, v in raw_ontime.items()},
                        **{f"all__{c}": v for c, v in raw_all.items()})
    return summary


def traced_frame(arm, k, clip, fidx, dec_ms, c1_ms, pose_ms, track_ms, score_ms, n_dets, n_tracks, n_wins, min_norm, reused, wait, scorer_name, tel):
    """One hierarchical MLflow trace per sampled frame: decode -> (C1) -> pose -> TrackTrack -> scorer -> telemetry."""
    import mlflow
    from mlflow.entities import SpanType

    @mlflow.trace(name="e07_pi5_frame", span_type=SpanType.CHAIN)
    def _inner() -> dict:
        span = mlflow.get_current_active_span()
        if span:
            span.set_attribute("description", f"Pi 5 E07 frame trace ({arm}, clip {clip}, frame {fidx})")
        mlflow.update_current_trace(metadata={"mlflow.trace.session": f"e07-pi5-{arm}"})
        with mlflow.start_span(name="1_decode_frame", span_type=SpanType.TOOL) as s:
            s.set_inputs({"clip": clip, "frame": fidx, "global_index": k})
            s.set_outputs({"decode_ms": round(dec_ms, 3)})
        with mlflow.start_span(name="2_c1_static_check", span_type=SpanType.TOOL) as s:
            s.set_inputs({"queue_wait_ms": round(wait, 1)})
            s.set_outputs({"c1_ms": round(c1_ms, 3), "detections_reused": reused})
        with mlflow.start_span(name="3_pose_detector", span_type=SpanType.LLM) as s:
            s.set_inputs({"arm": arm, "reused": reused})
            s.set_outputs({"pose_ms": round(pose_ms, 3), "n_detections": n_dets})
        with mlflow.start_span(name="4_tracktrack", span_type=SpanType.TOOL) as s:
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=HOME / "q2-p1-e07-pi5")
    ap.add_argument("--tracking-uri", default=os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    ap.add_argument("--arms", default="")
    ap.add_argument("--local", action="store_true", help="dry run off the Pi (no PMIC, no MLflow)")
    ap.add_argument("--max-clips", type=int, default=None)
    ap.add_argument("--max-frames-per-clip", type=int, default=None)
    ap.add_argument("--tracker", default=f"{TRACK_HIGH},{TRACK_LOW},{TRACK_MATCH}", help="c_high,c_low,match of the T1-selected TrackTrack cell")
    ap.add_argument("--rerun", action="store_true")
    ap.add_argument("--frames-root", type=Path, default=None)
    ap.add_argument("--models-root", type=Path, default=None)
    ap.add_argument("--ckpt-root", type=Path, default=None)
    a = ap.parse_args()
    global FRAMES_ROOT, MODELS_ROOT, CKPT_ROOT
    FRAMES_ROOT, MODELS_ROOT, CKPT_ROOT = a.frames_root or FRAMES_ROOT, a.models_root or MODELS_ROOT, a.ckpt_root or CKPT_ROOT
    tracker_cfg = tuple(float(x) for x in a.tracker.split(","))
    arms = [x for x in arm_list() if not a.arms or x["name"] in a.arms.split(",")]
    log_mlflow = not a.local
    board = {}
    if not a.local:
        import mlflow

        from edgelab.telemetry.pi5_telemetry import read_board_info

        mlflow.set_tracking_uri(a.tracking_uri)
        mlflow.set_experiment("Q2-P1-E07")
        board = read_board_info()
        print(f"[Pi5] board: {json.dumps(board)}", flush=True)
    frames = load_frames(a.max_clips, a.max_frames_per_clip)
    print(f"[Pi5] {len(frames)} frames, {len({f[0] for f in frames})} clips, {len(arms)} arms", flush=True)
    summaries = []
    for arm in arms:
        sf = a.out_dir / f"{arm['name']}_summary.json"
        if sf.exists() and not a.rerun:
            summaries.append(json.loads(sf.read_text()))
            print(f"[Pi5] cached {arm['name']}", flush=True)
            continue
        print(f"[Pi5] running {arm['name']}", flush=True)
        if log_mlflow:
            import mlflow

            with mlflow.start_run(run_name=arm["name"], description=arm["description"]) as run:
                mlflow.set_tags({"machine": "pi5", "board_model": board.get("board_model", ""), "cpu_governor": board.get("cpu_governor", ""),
                                 "arm_name": arm["name"], "detector": arm["detector"], "scorer": arm["scorer"], "tracker": "tracktrack"})
                mlflow.log_params({"arm": arm["name"], "detector": arm["detector"], "scorer": arm["scorer"], "use_c1": arm["use_c1"],
                                   "use_guard": arm["use_guard"], "threads": 4, "frames": len(frames), "tracker_cfg": a.tracker})
                res = run_arm(arm, frames, a.out_dir, a.local, tracker_cfg, True)
                tel = res["telemetry"]
                flat = {k: v for k, v in res.items() if isinstance(v, (int, float)) and not isinstance(v, bool) and np.isfinite(v)}
                flat.update({f"tel_{k}": v for k, v in tel.items() if isinstance(v, (int, float)) and np.isfinite(v)})
                mlflow.log_metrics(flat)
                for suffix in ("summary.json", "telemetry.json", "trace.npz", "raw.npz"):
                    mlflow.log_artifact(str(a.out_dir / f"{arm['name']}_{suffix}"))
                res["mlflow_run_id"] = run.info.run_id
                (a.out_dir / f"{arm['name']}_summary.json").write_text(json.dumps(res, indent=2))
        else:
            res = run_arm(arm, frames, a.out_dir, a.local, tracker_cfg, False)
        summaries.append(res)
        tel = res["telemetry"]
        print(f"[Pi5] DONE {arm['name']}: processed {res['frames_processed']}/{res['frames']} service {res['service_ms_mean']:.1f} ms "
              f"(pose {res['pose_ms_mean']:.1f} track {res['track_ms_mean']:.2f} score {res['score_ms_mean']:.2f}) wait_p99 {res['queue_wait_ms_p99']:.0f} ms "
              f"P={tel['power_total_w_mean']} W T={tel['temp_c_mean']} C RSS={tel['rss_mb_peak']} MB", flush=True)
    if log_mlflow:
        import mlflow

        mlflow.flush_trace_async_logging()
    (a.out_dir / "pi5_e07_all.json").write_text(json.dumps({"board": board, "arms": summaries}, indent=2))


if __name__ == "__main__":
    main()
