"""Q2-P1-E05 C1 - real sustained end-to-end run inside the `edge-os` ARM64 VM (worker; the host driver lives in q2_p1_e05.py).

One process, one FIFO worker, the real locked pipeline on a wall-clock arrival schedule:

    cv2 H.264 decode (grab every frame, retrieve when the frame is inside a run)
      -> pixel C1 check (E03 24x40 luma thumbnails, ROI mean / exterior p95 vs the level-3 thresholds, age <= A1)
      -> NCNN FP16 ML0 detector + pose (only when C1 does not hold)
      -> two-stage BoT-SORT (persistent within a run; fed the *reused* detections on C1 frames = non-oracle identity)
      -> per-track window buffers -> PyTorch STG-NF (frozen) -> online CausalScoreDeliverer (sentinel / hold)

Frames arrive every `dt / s` ms (`s` = Pi-slowdown emulation: arrivals `s` times faster == a machine `s` times slower). Waits, the scheduler's guard
thresholds and the 2,000 ms late-work cap are all expressed in Pi-equivalent ms (VM ms x `s`). Arms: `fixed_k1` (detector every frame, every window
scored), `cache_c1` (W = period with C1), `cache_c1_guard` (the same + the Independent-Combo load-shedding guard).

    python -m edgelab.analysis.q2_p1_e05_vm longrun --config cfg.json --out prefix
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from edgelab.analysis import q2_p1_e02_vm as vm

GATE_STORE = 0.10  # detections of this confidence define the C1 region of interest (E03 `_static_clip`)


def steal_snapshot() -> tuple[float, float, float]:
    """(idle, total, steal) jiffies of the VM from /proc/stat."""
    f = [float(x) for x in Path("/proc/stat").read_text().splitlines()[0].split()[1:]]
    return f[3] + (f[4] if len(f) > 4 else 0.0), sum(f[:8]), (f[7] if len(f) > 7 else 0.0)


def cooling_states() -> dict:
    out = {}
    for p in sorted(Path("/sys/class/thermal").glob("cooling_device*")):
        try:
            out[p.name] = {"type": (p / "type").read_text().strip(), "cur_state": int((p / "cur_state").read_text())}
        except Exception:  # noqa: BLE001
            pass
    return out


def run_longrun(cfg_path: Path, out: Path) -> None:
    import cv2
    import torch

    from edgelab.analysis import q2_p1_e00 as e00
    from edgelab.analysis import q2_p1_e03 as e03
    from edgelab.analysis import q2_p1_e05 as e05
    from edgelab.analysis.q2_p1_e01 import build_tracker_cell
    from edgelab.pipeline.refresh_policy import CausalScoreDeliverer

    cfg = json.loads(Path(cfg_path).read_text())
    threads = int(cfg.get("threads", 4))
    vm.pin(threads)
    torch.set_num_threads(threads)
    cv2.setNumThreads(threads)
    det = vm.make_detector("ML0-640", threads)
    scorer = e00.Scorer("ShanghaiTech")
    L = scorer.seg_len
    cell = vm.tracker_cells()["locked"]
    vm.pin(threads)
    clips = cfg["clips"]
    paths = [vm.STAGING / f"h264_sht/{c}.mp4" for c in clips]
    passes, s_emul, mode = int(cfg["passes"]), float(cfg["s_emul"]), cfg["mode"]
    fps = 24.0
    dt_ms = 1000.0 / fps
    dt_emul = dt_ms / s_emul  # VM ms between two arrivals
    cap_vm = float(cfg.get("cap_ms", e03.CAP_MS)) / s_emul
    use_c1, use_guard = mode in ("cache_c1", "cache_c1_guard"), mode == "cache_c1_guard"
    period = float(cfg.get("period", 1.0))
    tau_roi, tau_ext, a1, delta = float(cfg["c1"]["roi"]), float(cfg["c1"]["ext"]), int(cfg["c1"]["a1"]), L // 2
    stub = SimpleNamespace(L=L)
    lvl = 3 if use_c1 else 0
    pol = (e05.GuardedFixedPolicy(stub, None, period, lvl, None, e03.BACKLOG_SHED_ON_MS, e03.BACKLOG_SHED_OFF_MS) if use_guard
           else e03.FixedPolicy(stub, None, period, lvl, None))
    n_clip = []
    for p in paths:
        cap0 = cv2.VideoCapture(str(p))
        n_clip.append(int(cap0.get(cv2.CAP_PROP_FRAME_COUNT)))
        cap0.release()
    # warm-up (not timed): decoder, detector, tracker, scorer
    cap0 = cv2.VideoCapture(str(paths[0]))
    ok, img0 = cap0.read()
    cap0.release()
    for _ in range(3):
        det(img0, vm.GATE)
    scorer.score(np.random.default_rng(0).uniform(100, 400, (4, L, 17, 3)))
    wtr = build_tracker_cell(cell)
    for fr in vm.synthetic_frames(4, 12, np.random.default_rng(0)):
        vm.track_update(wtr, fr, vm.GATE)
    vm.pin(threads)

    total = passes * sum(n_clip)
    marks = sorted(set(int(m) for m in cfg.get("rss_marks", [0, 2500, 5000, 10000])))
    rss_marks: dict[int, float] = {}
    rss_series: list[tuple[int, float]] = []
    blocks: list[dict] = []
    cols = {k: np.zeros(total, dtype=np.float64) for k in ("arrival_ms", "start_ms", "wait_ms", "svc_ms", "cpu_ms", "decode_ms", "pose_ms", "track_ms", "score_ms")}
    flags = {k: np.zeros(total, dtype=np.int8) for k in ("obs", "reused", "forced", "shed", "scored", "windows")}
    clip_of = np.zeros(total, dtype=np.int32)
    local_t = np.zeros(total, dtype=np.int32)
    series = np.zeros(total, dtype=np.float64)
    win_rec: list[tuple] = []  # (stream clip position, completion frame t, global frame, finish_rel_ms, vmin, dropped_late, dropped_end)
    transitions: list[tuple] = []  # (global frame, shed state, wait Pi ms)
    n_dropped_late = n_dropped_end = 0
    clip_starts = []
    T0 = time.perf_counter() + 0.5
    cpu_start = time.process_time()
    steal0 = steal_snapshot()
    block_t0, block_c0, block_s0 = time.perf_counter(), time.process_time(), steal0
    cool0 = cooling_states()
    last_shed = False
    g = 0
    pos = 0
    pending: list[tuple[int, float, int, int]] = []  # (landing global frame, vmin, completion global frame, stream clip)
    for pass_i in range(passes):
        for ci, path in enumerate(paths):
            n = n_clip[ci]
            pol.start_clip(n)
            clip_starts.append(g)
            deliverer = CausalScoreDeliverer(float(cfg["sentinel"]), int(cfg["hold"]))
            n_dropped_end += len(pending)  # windows still queued for an earlier clip cannot land inside this one
            pending = []
            cap = None
            tracker = None
            buffers: dict[int, list] = {}
            rl, a = 0, -1
            src = None
            for t in range(n):
                if g in marks:
                    rss_marks[g] = vm.rss_mb()
                if g % 250 == 0:
                    rss_series.append((g, vm.rss_mb()))
                if g and g % 1000 == 0:
                    idle1, tot1, st1 = steal_snapshot()
                    wnow, cnow = time.perf_counter(), time.process_time()
                    blocks.append({"end_frame": g, "wall_s": wnow - block_t0, "cpu_s": cnow - block_c0, "steal_jiffies": st1 - block_s0[2],
                                   "total_jiffies": tot1 - block_s0[1], "idle_jiffies": idle1 - block_s0[0], "rss_mb": vm.rss_mb(),
                                   "loadavg": os.getloadavg()[0],
                                   "svc_mean_ms": float(cols["svc_ms"][g - 1000:g].mean()), "cpu_mean_ms": float(cols["cpu_ms"][g - 1000:g].mean()),
                                   "reuse_frac_of_inrun": float(flags["reused"][g - 1000:g].sum() / max(1, flags["obs"][g - 1000:g].sum()))})
                    block_t0, block_c0, block_s0 = wnow, cnow, (idle1, tot1, st1)
                arrival = T0 + g * dt_emul / 1000.0
                now = time.perf_counter()
                if now < arrival:
                    if arrival - now > 0.002:
                        time.sleep(arrival - now - 0.001)
                    while time.perf_counter() < arrival:
                        pass
                    now = time.perf_counter()
                start_rel = (now - T0) * 1e3
                wait_vm = max(0.0, now - arrival) * 1e3
                wait_pi = wait_vm * s_emul
                c0 = time.process_time()
                clk = vm.Clock()
                # ---- emission of frame t's score at its capture time: windows that landed on or before this frame ------------------------------
                avail = [v for (lf, v, _, _) in pending if lf <= g]
                pending = [p for p in pending if p[0] > g]
                series[g] = deliverer.step_frame(t, avail)
                clip_of[g], local_t[g] = pos, t
                # ---- decode + decision ----------------------------------------------------------------------------------------------------------
                if cap is None:
                    cap = cv2.VideoCapture(str(path))
                if not cap.grab():
                    raise RuntimeError(f"decoder ended early: {path.name} frame {t}/{n}")
                can_start = (t & 1) == 0 and t <= n - L
                obs, score, _, _ = pol.decide(t, n, g, 0.0, wait_pi, rl, can_start)
                shed_now = bool(getattr(pol, "shed", False))
                if shed_now != last_shed:
                    transitions.append((g, int(shed_now), wait_pi))
                    last_shed = shed_now
                reused = forced = False
                t_pose = t_track = t_score = 0.0
                vmin = float("inf")
                n_win = 0
                if obs:
                    if rl == 0:
                        if not can_start:
                            raise AssertionError("policy started a run where none is possible")
                        a, src = t, None
                        tracker = build_tracker_cell(cell)
                        buffers = {}
                    rl += 1
                    ok, img = cap.retrieve()
                    if not ok:
                        raise RuntimeError(f"retrieve failed: {path.name} frame {t}")
                    t_dec, _ = clk.lap()
                    cols["decode_ms"][g] = t_dec
                    if use_c1:
                        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        th = e03.thumb(e03.luma_small(gray)).astype(np.float32)
                    forced = (t - a) % delta == 0
                    if use_c1 and not forced and src is not None and (t - src["t"]) <= a1:
                        d = np.abs(th - src["th"])
                        m = src["mask"]
                        roi = float(d[m].mean()) if m.any() else 0.0
                        ext = float(np.percentile(d[~m], 95)) if (~m).any() else 0.0
                        reused = roi <= tau_roi and ext <= tau_ext
                    if reused:
                        frame = src["frame"]
                    else:
                        boxes, scores, kps, ms = det(img, vm.GATE)
                        t_pose = ms["pre_ms"] + ms["infer_ms"] + ms["post_ms"]
                        frame = {"boxes": boxes, "conf": scores, "cls": np.zeros(len(scores), dtype=np.float32), "keypoints": kps}
                        if use_c1:
                            hgt, wid = gray.shape
                            keep = scores >= GATE_STORE
                            mask = e03.box_cells(boxes[keep], wid, hgt) if keep.any() else np.zeros((e03.GRID_H, e03.GRID_W), dtype=bool)
                            src = {"t": t, "th": th, "mask": mask, "frame": frame}
                    clk.lap()
                    poses, _ = vm.track_update(tracker, frame, cell.gate)
                    t_track, _ = clk.lap()
                    for pid, kp in poses.items():
                        buf = buffers.setdefault(pid, [])
                        buf.append((t, kp))
                        if len(buf) > L:
                            del buf[0]
                    if score:
                        ready = [np.stack([k for _, k in b]) for b in buffers.values() if len(b) == L and b[-1][0] - b[0][0] == L - 1]
                        if ready:
                            clk.lap()
                            norm = scorer.score(np.stack(ready))
                            t_score, _ = clk.lap()
                            vmin = float(np.min(norm))
                            n_win = len(ready)
                else:
                    rl = 0
                    clk.lap()
                t_end = time.perf_counter()
                fin_rel = (t_end - T0) * 1e3
                if vmin < float("inf"):
                    latency_pi = (fin_rel - g * dt_emul) * s_emul
                    lf = int(e03.e02.delivery_frame(np.array([fin_rel]), dt_emul)[0])
                    lf = max(lf, g)
                    late = latency_pi > float(cfg.get("cap_ms", e03.CAP_MS))
                    beyond = lf >= clip_starts[-1] + n
                    win_rec.append((pos, t, g, fin_rel, vmin, int(late), int(beyond)))
                    if late:
                        n_dropped_late += 1
                    elif beyond:
                        n_dropped_end += 1
                    else:
                        pending.append((lf, vmin, g, pos))
                cols["arrival_ms"][g], cols["start_ms"][g] = (arrival - T0) * 1e3, start_rel
                cols["wait_ms"][g] = wait_vm
                cols["svc_ms"][g] = (t_end - now) * 1e3
                cols["cpu_ms"][g] = (time.process_time() - c0) * 1e3
                cols["pose_ms"][g], cols["track_ms"][g], cols["score_ms"][g] = t_pose, t_track, t_score
                flags["obs"][g], flags["reused"][g], flags["forced"][g] = int(obs), int(reused), int(forced)
                flags["shed"][g], flags["scored"][g], flags["windows"][g] = int(shed_now), int(vmin < float("inf")), n_win
                g += 1
            if cap is not None:
                cap.release()
            pos += 1
    wall_total = time.perf_counter() - T0
    rss_marks[g] = vm.rss_mb()
    rss_series.append((g, vm.rss_mb()))
    steal1 = steal_snapshot()
    cool1 = cooling_states()
    arrays = {**cols, **flags, "clip_of": clip_of, "local_t": local_t, "series": series, "clip_starts": np.array(clip_starts), "n_clip": np.array(n_clip * passes),
              "win_rec": np.array(win_rec, dtype=np.float64).reshape(-1, 7), "transitions": np.array(transitions, dtype=np.float64).reshape(-1, 3),
              "rss_series": np.array(rss_series, dtype=np.float64)}
    meta = {"cell": "e05_longrun", "mode": mode, "s_emul": s_emul, "period": period, "frames": int(total), "passes": passes, "clips": clips, "threads": threads,
            "wall_total_s": wall_total, "cpu_total_s": time.process_time() - cpu_start, "rss_marks_mb": {str(k): v for k, v in rss_marks.items()},
            "blocks": blocks, "steal_total_jiffies": steal1[2] - steal0[2], "jiffies_total": steal1[1] - steal0[1], "cooling_start": cool0, "cooling_end": cool1,
            "thermal_sensor": "none (VM exposes cooling_device* only)" if not list(Path("/sys/class/thermal").glob("thermal_zone*")) else "thermal_zone present",
            "dropped_late": n_dropped_late, "dropped_end": n_dropped_end, "guard": pol.info(), "sentinel": cfg["sentinel"], "hold": cfg["hold"], "cap_ms": cfg.get("cap_ms", e03.CAP_MS),
            "c1": cfg["c1"], "x264": "preset veryfast crf 23 yuv420p gop 48", "affinity": sorted(os.sched_getaffinity(0)),
            "tracker": cell.label, "persistent_tracker_within_run": True, "forced_detect_every": delta}
    vm.save(out, arrays, meta)


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("longrun")
    r.add_argument("--config", type=Path, required=True)
    r.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if args.cmd == "longrun":
        run_longrun(args.config, args.out)


if __name__ == "__main__":
    main()
