"""EXP-012 step 2: detector variants timed in the VM (plan: EXP-004 protocol). Runs on the HOST, drives the VM.

    uv run --extra pose python -m edgelab.pipeline.timing_exp012 [--reps 3] [--max-frames 40] [--out-dir <dir>]

Cell = EXP-004's `L3-E0-R<R>-c0.25-O0-B4` (12-video x 40-frame subset, co-load on) for 8 detector variants:
{ORT FP32, ORT INT8, NCNN FP16, NCNN FP32} x R {640, 800}. Every repetition runs all 8 variants once in a seeded random order
(same idea as timing_matrix.make_plan). Same host-load rule as EXP-004: a run whose mean host load outside the VM is > 30%
is excluded and repeated. Output: <out-dir>/{manifest.csv, <variant>_rep<k>_try<j>.csv}; `summary` prints mean detector ms.
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
KINDS = ("ort-fp32", "ort-int8", "ncnn-fp16", "ncnn-fp32")
RESOLUTIONS = (640, 800)
RESULTS = EXCHANGE / "results/exp012"


def model_path(kind: str, imgsz: int, models: Path = MODELS) -> Path:
    base = f"yolov8n-pose-{imgsz}-{HW[imgsz]}"
    if kind == "ort-fp32":
        return models / f"{base}.onnx"
    if kind == "ort-int8":
        return models / f"{base}-int8.onnx"
    return models / f"{base}-ncnn"  # a folder: live_timing --yolo-onnx <folder> selects the NCNN detector


def variant_id(kind: str, imgsz: int) -> str:
    return f"{kind}-R{imgsz}"


def plan(reps: int, seed: int = 12) -> list[tuple[int, str, int, int]]:
    """[(order, kind, imgsz, rep)]: each repetition = all variants once, in a seeded random order."""
    variants = [(k, r) for k in KINDS for r in RESOLUTIONS]
    out = []
    for rep in range(reps):
        for kind, imgsz in random.Random(f"{seed}|{rep}").sample(variants, len(variants)):
            out.append((len(out), kind, imgsz, rep))
    return out


def vm_command(kind: str, imgsz: int, trace: Path, tag: str, max_frames: int = FRAMES_PER_VIDEO) -> list[str]:
    cell = Cell("L3", "E0", imgsz, 0.25, "O0", "B4")
    args = live_timing_args(cell, 0, trace)
    args[args.index("--tag") + 1] = tag
    args[args.index("--max-frames-per-video") + 1] = str(max_frames)
    args += ["--yolo-onnx", str(model_path(kind, imgsz))]
    if kind == "ncnn-fp32":
        args += ["--ncnn-fp16", "off"]
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
        for order, kind, imgsz, rep in plan(reps):
            cell_id = variant_id(kind, imgsz)
            for attempt in range(max_tries):
                if (cell_id, rep) in done:
                    break
                quiet = wait_for_quiet_host(vm_pid, limit)
                trace = out_dir / f"{cell_id}_rep{rep}_try{attempt}.csv"
                row = {"cell_id": cell_id, "rep": rep, "order": order, "start": time.strftime("%Y-%m-%d %H:%M:%S"),
                       "trace_file": str(trace), "error": f"pre-run host outside VM {quiet}"}
                sampler = LoadSampler(vm_pid).start()
                result = subprocess.run(vm_command(kind, imgsz, trace, f"{cell_id}-rep{rep}", max_frames),
                                        capture_output=True, text=True, timeout=1800)
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
                print(f"{cell_id} rep{rep} try{attempt} {row['status']} outside_vm={row['host_outside_vm_pct']} "
                      f"excluded={row['excluded']}", flush=True)
                if row["status"] == "ok" and not row["excluded"]:
                    break


def summary(out_dir: Path) -> pd.DataFrame:
    """One row per variant: usable runs, frames, mean detector ms (pre+infer+post), mean infer ms, p95, host load outside VM."""
    manifest = pd.read_csv(out_dir / "manifest.csv")
    ok = manifest[(manifest.status == "ok") & (~manifest.excluded.astype(bool))]
    rows = []
    for cell_id, group in ok.groupby("cell_id"):
        traces = [pd.read_csv(p) for p in group.trace_file]
        det = pd.concat([detector_ms(t) for t in traces])
        rows.append({"variant": cell_id, "runs": len(group), "frames": len(det), "det_ms": det.mean(),
                     "det_p95_ms": det.quantile(0.95), "infer_ms": pd.concat([t.det_infer_ms for t in traces]).mean(),
                     "service_ms": pd.concat([t.service_ms for t in traces]).mean(),
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
