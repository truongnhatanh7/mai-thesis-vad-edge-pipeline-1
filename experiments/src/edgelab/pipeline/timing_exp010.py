"""EXP-010 step 0: cost of YOLOv8s-pose relative to YOLOv8n-pose in the VM (plan §2). Runs on the HOST, drives the VM.

    uv run --extra pose python -m edgelab.pipeline.timing_exp010 [--reps 3]

Cell = EXP-004's `L3-E0-R640-c0.25-O0-B4` (12-video x 40-frame subset, co-load on), with the YOLOv8s ONNX instead of the n one
(`live_timing --yolo-onnx`). Same host-load rule as EXP-004: a run whose mean host load outside the VM is > 30% is excluded and
repeated (up to --max-tries). `--with-n` also re-times yolov8n in the same window (a sensitivity check only; r uses the EXP-004 traces).
Output: ~/thesis-data/vm-exchange/results/exp010/{manifest.csv, <model>_rep<k>_try<j>.csv}; `r_factor` computes r from them.
"""

import argparse
import csv
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

from edgelab.pipeline.host_load import LoadSampler, find_vm_pid, sample_once, summarise
from edgelab.pipeline.timing_matrix import (CELL_SCRIPT, EXCHANGE, FRAMES_PER_VIDEO, MANIFEST_COLUMNS, SUBSET_FILE, Cell,
                                            cleanup_vm, is_excluded, live_timing_args)

CELL = Cell("L3", "E0", 640, 0.25, "O0", "B4")
MODELS = {"yolov8s": EXCHANGE / "models/yolov8s-pose-640-384x640.onnx", "yolov8n": EXCHANGE / "models/yolov8n-pose-640-384x640.onnx"}
RESULTS = EXCHANGE / "results/exp010"
MANIFEST = RESULTS / "manifest.csv"
DETECTOR_COLUMNS = ["det_pre_ms", "det_infer_ms", "det_post_ms"]


def detector_ms(trace: pd.DataFrame) -> pd.Series:
    """Per-frame detector time = pre + infer + post (what EXP-010 multiplies by r)."""
    return trace[DETECTOR_COLUMNS].sum(axis=1)


def r_factor(s_traces: list[pd.DataFrame], n_traces: list[pd.DataFrame]) -> dict:
    """r = mean detector ms of the 8s runs / mean detector ms of the 8n runs (all frames of all runs pooled)."""
    s_ms = np.concatenate([detector_ms(t).to_numpy() for t in s_traces])
    n_ms = np.concatenate([detector_ms(t).to_numpy() for t in n_traces])
    return {"s_mean_ms": float(s_ms.mean()), "n_mean_ms": float(n_ms.mean()), "r": float(s_ms.mean() / n_ms.mean()),
            "s_frames": len(s_ms), "n_frames": len(n_ms)}


def vm_command(model: str, trace: Path, tag: str) -> list[str]:
    args = live_timing_args(CELL, 0, trace)
    args[args.index("--tag") + 1] = tag
    return ["limactl", "shell", "edge-os", "--", "sudo", "bash", str(CELL_SCRIPT), CELL.b, CELL.o, *args,
            "--yolo-onnx", str(MODELS[model])]


def wait_for_quiet_host(vm_pid: int, limit: float, max_wait_s: int = 1800) -> float:
    """Block until one 3 s host sample outside the VM is <= limit (or give up after max_wait_s). Returns the last sample."""
    waited, last = 0, None
    while waited <= max_wait_s:
        last = sample_once(vm_pid, 10)["outside"]
        if last <= limit:
            return last
        time.sleep(30)
        waited += 33
    return last


def run(reps: int, with_n: bool, max_tries: int, limit: float = 30.0):
    RESULTS.mkdir(parents=True, exist_ok=True)
    vm_pid = find_vm_pid()
    new_file = not MANIFEST.exists()
    done = set()
    if not new_file:
        old = pd.read_csv(MANIFEST)
        done = {(r.cell_id, int(r.rep)) for r in old.itertuples() if r.status == "ok" and not r.excluded}
    models = ["yolov8s", "yolov8n"] if with_n else ["yolov8s"]
    with open(MANIFEST, "a", newline="") as f:
        writer = csv.DictWriter(f, MANIFEST_COLUMNS)
        if new_file:
            writer.writeheader()
        for rep in range(reps):
            for model in models:
                cell_id = f"{model}-{CELL.id}"
                for attempt in range(max_tries):
                    if (cell_id, rep) in done:
                        break
                    quiet = wait_for_quiet_host(vm_pid, limit)
                    trace = RESULTS / f"{model}_rep{rep}_try{attempt}.csv"
                    row = {"cell_id": cell_id, "rep": rep, "order": attempt, "start": time.strftime("%Y-%m-%d %H:%M:%S"),
                           "trace_file": str(trace), "error": f"pre-run host outside VM {quiet}"}
                    sampler = LoadSampler(vm_pid).start()
                    result = subprocess.run(vm_command(model, trace, f"{cell_id}-rep{rep}"), capture_output=True, text=True,
                                            timeout=1800)
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


def load_ok_traces(model: str) -> list[pd.DataFrame]:
    manifest = pd.read_csv(MANIFEST)
    rows = manifest[(manifest.cell_id == f"{model}-{CELL.id}") & (manifest.status == "ok") & (~manifest.excluded.astype(bool))]
    return [pd.read_csv(p) for p in rows.trace_file]


def compute_r() -> dict:
    """r from the finished 8s runs and the EXP-004 8n traces of the same cell (usable = ok and not excluded); saved as r.json."""
    import json

    from edgelab.analysis.exp004 import usable_traces
    from edgelab.analysis.exp008 import MANIFEST as EXP004_MANIFEST

    n_files = usable_traces(EXP004_MANIFEST)[CELL.id]
    result = r_factor(load_ok_traces("yolov8s"), [pd.read_csv(p) for p in n_files])
    result["n_runs_8n_exp004"] = len(n_files)
    result["n_runs_8s"] = len(load_ok_traces("yolov8s"))
    if (RESULTS / "manifest.csv").exists() and load_ok_traces("yolov8n"):  # same-window 8n, sensitivity only
        same = r_factor(load_ok_traces("yolov8s"), load_ok_traces("yolov8n"))
        result["sensitivity_same_window_8n"] = {"n_mean_ms": same["n_mean_ms"], "r": same["r"], "n_runs": len(load_ok_traces("yolov8n"))}
    (RESULTS / "r.json").write_text(json.dumps(result, indent=1))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--with-n", action="store_true", help="also re-time yolov8n in the same window (sensitivity check)")
    parser.add_argument("--max-tries", type=int, default=6)
    parser.add_argument("--report", action="store_true", help="only compute r from the finished runs")
    args = parser.parse_args()
    if not args.report:
        run(args.reps, args.with_n, args.max_tries)
    import json

    print(json.dumps(compute_r(), indent=1))


if __name__ == "__main__":
    main()
