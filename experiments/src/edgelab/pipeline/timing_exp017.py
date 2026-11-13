"""EXP-017 Y4: detector cost of YOLO26n-pose vs YOLOv8n-pose in the VM (EXP-012 protocol). Runs on the HOST, drives the VM.

    uv run --extra pose python -m edgelab.pipeline.timing_exp017 [--reps 3] [--max-frames 40]
    uv run --extra pose python -m edgelab.pipeline.timing_exp017 --summary-only

Cell = EXP-004's `L3-E0-R<R>-c0.25-O0-B4` (12 videos x 40 frames, co-load on, library-default threads), as EXP-012. Variants:
  YOLO26n-pose: ORT FP32 and NCNN FP16 at 384x640 (R640) and 480x800 (R800);
  same-window controls (YOLOv8n-pose, EXP-012 models): ORT FP32 R640 and NCNN FP16 R640.
Every repetition runs all variants once in a seeded random order; a run whose mean host load outside the VM exceeds 30% is
excluded and repeated (waiting for a quiet host first). Output: ~/thesis-data/vm-exchange/results/exp017/.
The ORT YOLO26 model is the NMS-free end-to-end head; Ultralytics' NCNN export of YOLO26 uses the one-to-many head with
external NMS (see EXP-017 log), so the NCNN rows include the NMS post-processing time.
"""

import argparse
import csv
import random
import subprocess
import time
from pathlib import Path

import pandas as pd

from edgelab.pipeline.host_load import LoadSampler, find_vm_pid, summarise
from edgelab.pipeline.timing_exp010 import detector_ms, wait_for_quiet_host
from edgelab.pipeline.timing_matrix import (CELL_SCRIPT, EXCHANGE, FRAMES_PER_VIDEO, MANIFEST_COLUMNS, Cell, cleanup_vm,
                                            is_excluded, live_timing_args)

MODELS = EXCHANGE / "models"
HW = {640: "384x640", 800: "480x800"}
RESULTS = EXCHANGE / "results/exp017"
# variant id -> (model name prefix, kind)
VARIANTS = {"yolo26n-ort-fp32-R640": ("yolo26n-pose", "ort-fp32", 640), "yolo26n-ort-fp32-R800": ("yolo26n-pose", "ort-fp32", 800),
            "yolo26n-ncnn-fp16-R640": ("yolo26n-pose", "ncnn-fp16", 640), "yolo26n-ncnn-fp16-R800": ("yolo26n-pose", "ncnn-fp16", 800),
            "yolov8n-ort-fp32-R640": ("yolov8n-pose", "ort-fp32", 640), "yolov8n-ncnn-fp16-R640": ("yolov8n-pose", "ncnn-fp16", 640)}


def model_path(variant: str, models: Path = MODELS) -> Path:
    name, kind, imgsz = VARIANTS[variant]
    base = f"{name}-{imgsz}-{HW[imgsz]}"
    return models / f"{base}.onnx" if kind == "ort-fp32" else models / f"{base}-ncnn"  # a folder = NCNN


def plan(reps: int, seed: int = 17) -> list[tuple[int, str, int]]:
    """[(order, variant, rep)]: each repetition = every variant once, in a seeded random order."""
    out = []
    for rep in range(reps):
        for variant in random.Random(f"{seed}|{rep}").sample(list(VARIANTS), len(VARIANTS)):
            out.append((len(out), variant, rep))
    return out


def vm_command(variant: str, trace: Path, tag: str, max_frames: int = FRAMES_PER_VIDEO) -> list[str]:
    imgsz = VARIANTS[variant][2]
    cell = Cell("L3", "E0", imgsz, 0.25, "O0", "B4")
    args = live_timing_args(cell, 0, trace)
    args[args.index("--tag") + 1] = tag
    args[args.index("--max-frames-per-video") + 1] = str(max_frames)
    args += ["--yolo-onnx", str(model_path(variant))]  # NCNN FP16 is live_timing's default for a folder
    return ["limactl", "shell", "edge-os", "--", "sudo", "bash", str(CELL_SCRIPT), cell.b, cell.o, *args]


def run(reps: int, max_frames: int, out_dir: Path, max_tries: int, limit: float = 30.0):
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = out_dir / "manifest.csv"
    vm_pid = find_vm_pid()
    new_file = not manifest.exists()
    done = set()
    if not new_file:
        old = pd.read_csv(manifest)
        done = {(r.cell_id, int(r.rep)) for r in old.itertuples() if r.status == "ok" and not r.excluded}
    with open(manifest, "a", newline="") as f:
        writer = csv.DictWriter(f, MANIFEST_COLUMNS)
        if new_file:
            writer.writeheader()
        for order, variant, rep in plan(reps):
            for attempt in range(max_tries):
                if (variant, rep) in done:
                    break
                quiet = wait_for_quiet_host(vm_pid, limit)
                trace = out_dir / f"{variant}_rep{rep}_try{attempt}.csv"
                row = {"cell_id": variant, "rep": rep, "order": order, "start": time.strftime("%Y-%m-%d %H:%M:%S"),
                       "trace_file": str(trace), "error": f"pre-run host outside VM {quiet}"}
                sampler = LoadSampler(vm_pid).start()
                result = subprocess.run(vm_command(variant, trace, f"{variant}-rep{rep}", max_frames), capture_output=True,
                                        text=True, timeout=1800)
                row.update(summarise(sampler.stop()))
                row["end"] = time.strftime("%Y-%m-%d %H:%M:%S")
                row["status"] = "ok" if result.returncode == 0 and trace.exists() else "failed"
                row["excluded"] = is_excluded(row["host_outside_vm_pct"], limit)
                tail = result.stdout.strip().splitlines()[-1:] + result.stderr.strip().splitlines()[-3:]
                row["coload_cpu_s"] = next((t.split("COLOAD_CPU_S=")[1].split()[0] for t in tail if "COLOAD_CPU_S=" in t), "")
                if row["status"] == "failed":
                    row["error"] += " | " + " | ".join(tail)[:300]
                    cleanup_vm()
                writer.writerow(row)
                f.flush()
                print(f"{variant} rep{rep} try{attempt} {row['status']} outside_vm={row['host_outside_vm_pct']} "
                      f"excluded={row['excluded']}", flush=True)
                if row["status"] == "ok" and not row["excluded"]:
                    break


def usable_files(variant: str, out_dir: Path = RESULTS) -> list[Path]:
    manifest = pd.read_csv(out_dir / "manifest.csv")
    ok = manifest[(manifest.cell_id == variant) & (manifest.status == "ok") & (~manifest.excluded.astype(bool))]
    return [Path(p) for p in ok.trace_file]


def summary(out_dir: Path = RESULTS) -> pd.DataFrame:
    """One row per variant: usable runs, frames, mean detector ms (pre+infer+post), infer ms, p95, service ms, host load."""
    manifest = pd.read_csv(out_dir / "manifest.csv")
    ok = manifest[(manifest.status == "ok") & (~manifest.excluded.astype(bool))]
    rows = []
    for cell_id, group in ok.groupby("cell_id"):
        traces = [pd.read_csv(p) for p in group.trace_file]
        det = pd.concat([detector_ms(t) for t in traces])
        per_run = [detector_ms(t).mean() for t in traces]
        rows.append({"variant": cell_id, "runs": len(group), "frames": len(det), "det_ms": det.mean(),
                     "det_ms_per_run_sd": float(pd.Series(per_run).std()) if len(per_run) > 1 else float("nan"),
                     "det_p95_ms": det.quantile(0.95), "infer_ms": pd.concat([t.det_infer_ms for t in traces]).mean(),
                     "post_ms": pd.concat([t.det_post_ms for t in traces]).mean(),
                     "service_ms": pd.concat([t.service_ms for t in traces]).mean(),
                     "n_dets": pd.concat([t.n_dets for t in traces]).mean(),
                     "host_outside_vm_pct": group.host_outside_vm_pct.mean()})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--max-frames", type=int, default=FRAMES_PER_VIDEO)
    parser.add_argument("--out-dir", type=Path, default=RESULTS)
    parser.add_argument("--max-tries", type=int, default=6)
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()
    if not args.summary_only:
        run(args.reps, args.max_frames, args.out_dir, args.max_tries)
    print(summary(args.out_dir).round(2).to_string(index=False))


if __name__ == "__main__":
    main()
