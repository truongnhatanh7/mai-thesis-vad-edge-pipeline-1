"""EXP-020: NCNN FP16 detectors timed in the VM (EXP-012 protocol: EXP-004 cell L3-E0-c0.25-O0-B4, 12 videos x 40 frames,
co-load on, seeded random order per repetition, host-load rule: a run with mean host load outside the VM > 30% is excluded
and repeated). Runs on the HOST, drives the VM; start it only when nothing else heavy runs on the host.

    uv run --extra pose python -m edgelab.pipeline.timing_exp020 [--reps 3] [--out-dir <dir>] [--summary-only]

Variants (id -> NCNN folder written by `edgelab.pipeline.ncnn_pose`, all FP16): stock@640 and stock@800 (EXP-012 folders,
re-timed in the same window), and the four trained models M00, ML0, M0P, MLP at 640 (rect 384x640).
"""

import argparse
import csv
import random
import subprocess
import time
from pathlib import Path

import pandas as pd

from edgelab.pipeline.host_load import LoadSampler, find_vm_pid, summarise
from edgelab.pipeline.timing_exp010 import wait_for_quiet_host
from edgelab.pipeline.timing_exp012 import summary as exp012_summary
from edgelab.pipeline.timing_matrix import (CELL_SCRIPT, EXCHANGE, FRAMES_PER_VIDEO, MANIFEST_COLUMNS, Cell, cleanup_vm,
                                            is_excluded, live_timing_args)

MODELS = EXCHANGE / "models"
EXP020_MODELS = MODELS / "exp020"
RESULTS = EXCHANGE / "results/exp020"
VARIANTS = {  # id -> (NCNN folder, imgsz label of the cell)
    "stock-R640": (MODELS / "yolov8n-pose-640-384x640-ncnn", 640),
    "stock-R800": (MODELS / "yolov8n-pose-800-480x800-ncnn", 800),
    **{arm: (EXP020_MODELS / f"{arm}-640-384x640-ncnn", 640) for arm in ("M00", "ML0", "M0P", "MLP")},
}


def plan(reps: int, variants: list[str], seed: int = 20) -> list[tuple[int, str, int]]:
    """[(order, variant, rep)]: each repetition = every variant once in a seeded random order."""
    out = []
    for rep in range(reps):
        for variant in random.Random(f"{seed}|{rep}").sample(sorted(variants), len(variants)):
            out.append((len(out), variant, rep))
    return out


def vm_command(variant: str, trace: Path, tag: str, max_frames: int = FRAMES_PER_VIDEO) -> list[str]:
    folder, imgsz = VARIANTS[variant]
    cell = Cell("L3", "E0", imgsz, 0.25, "O0", "B4")
    args = live_timing_args(cell, 0, trace)
    args[args.index("--tag") + 1] = tag
    args[args.index("--max-frames-per-video") + 1] = str(max_frames)
    args += ["--yolo-onnx", str(folder)]  # a folder selects the NCNN detector; FP16 is the default
    return ["limactl", "shell", "edge-os", "--", "sudo", "bash", str(CELL_SCRIPT), cell.b, cell.o, *args]


def run(reps: int, max_frames: int, out_dir: Path, max_tries: int, limit: float = 30.0):
    available = [v for v, (folder, _) in VARIANTS.items() if (folder / "model.ncnn.param").exists()]
    missing = sorted(set(VARIANTS) - set(available))
    if missing:
        raise SystemExit(f"NCNN folders missing for {missing}; export first")
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
        for order, variant, rep in plan(reps, available):
            for attempt in range(max_tries):
                if (variant, rep) in done:
                    break
                quiet = wait_for_quiet_host(vm_pid, limit)
                trace = out_dir / f"{variant}_rep{rep}_try{attempt}.csv"
                row = {"cell_id": variant, "rep": rep, "order": order, "start": time.strftime("%Y-%m-%d %H:%M:%S"),
                       "trace_file": str(trace), "error": f"pre-run host outside VM {quiet}"}
                sampler = LoadSampler(vm_pid).start()
                result = subprocess.run(vm_command(variant, trace, f"{variant}-rep{rep}", max_frames),
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
                print(f"{variant} rep{rep} try{attempt} {row['status']} outside_vm={row['host_outside_vm_pct']} "
                      f"excluded={row['excluded']}", flush=True)
                if row["status"] == "ok" and not row["excluded"]:
                    break


def usable_trace_files(variant: str, out_dir: Path = RESULTS) -> list[Path]:
    manifest = pd.read_csv(out_dir / "manifest.csv")
    ok = manifest[(manifest.cell_id == variant) & (manifest.status == "ok") & (~manifest.excluded.astype(bool))]
    return [Path(p) for p in ok.trace_file]


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
    print(exp012_summary(args.out_dir).round(2).to_string(index=False))


if __name__ == "__main__":
    main()
