"""EXP-016 T3: VM cost per frame of the top-down configurations, one repetition each (plan: descriptive). Runs on the HOST.

    uv run --extra pose python -m edgelab.pipeline.timing_exp016 --pose-model rtmpose-s [--max-tries 6]
    uv run --extra pose python -m edgelab.pipeline.timing_exp016 --report --pose-model rtmpose-s

Cell = EXP-004's `L0-E0-...-c0.25-O0-B4` (12 videos x 40 frames, library-default threads, 4 vCPU, ffmpeg co-load on) with the
top-down pipeline (`topdown_timing`). Same host-load rule as EXP-004: a run whose mean host load outside the VM is > 30% is
excluded, and repeated after waiting for a quiet host (up to --max-tries). Every try is a row in the manifest.
Output: ~/thesis-data/vm-exchange/results/exp016-timing/{manifest.csv, <config>_try<j>.csv, summary.json}.
"""

import argparse
import csv
import json
import subprocess
import time
from pathlib import Path

import pandas as pd

from edgelab.pipeline.host_load import LoadSampler, find_vm_pid, summarise
from edgelab.pipeline.timing_exp010 import wait_for_quiet_host
from edgelab.pipeline.timing_matrix import EXCHANGE, FRAMES_PER_VIDEO, MANIFEST_COLUMNS, SUBSET_FILE, cleanup_vm, is_excluded

EXPERIMENT_DIR = Path(__file__).resolve().parents[3] / "EXP-016-topdown-detector-sweep"
CELL_SCRIPT = EXPERIMENT_DIR / "scripts/cell_in_vm.sh"
RESULTS = EXCHANGE / "results/exp016-timing"  # inside vm-exchange: the only writable mount of the VM
MANIFEST = RESULTS / "manifest.csv"
CONFIGS = [(imgsz, k) for imgsz in (640, 960) for k in (1, 3)]
LIMIT = 30.0
BUDGET_MS = 1000 / 24  # 41.7 ms: one frame at 24 fps


def config_id(imgsz: int, k: int, pose_model: str) -> str:
    return f"L0-n{imgsz}-K{k}-{pose_model}-c0.25-O0-B4"


def vm_command(imgsz: int, k: int, pose_model: str, trace: Path) -> list[str]:
    args = ["--frames-root", str(EXCHANGE / "frames"), "--videos-file", str(SUBSET_FILE), "--level", "L0", "--c", "0.25",
            "--imgsz", str(imgsz), "--k", str(k), "--pose-model", pose_model, "--max-frames-per-video", str(FRAMES_PER_VIDEO),
            "--out", str(trace), "--tag", config_id(imgsz, k, pose_model)]
    return ["limactl", "shell", "edge-os", "--", "sudo", "bash", str(CELL_SCRIPT), "B4", "O0", *args]


def done_configs() -> set[str]:
    if not MANIFEST.exists():
        return set()
    rows = pd.read_csv(MANIFEST)
    return {r.cell_id for r in rows.itertuples() if r.status == "ok" and not r.excluded}


def run(pose_model: str, max_tries: int):
    RESULTS.mkdir(parents=True, exist_ok=True)
    vm_pid = find_vm_pid()
    new_file = not MANIFEST.exists()
    with open(MANIFEST, "a", newline="") as f:
        writer = csv.DictWriter(f, MANIFEST_COLUMNS)
        if new_file:
            writer.writeheader()
            f.flush()  # done_configs() re-reads the manifest: an unflushed header-less file crashed T3 (16:19)
        for imgsz, k in CONFIGS:
            cell_id = config_id(imgsz, k, pose_model)
            for attempt in range(max_tries):
                if cell_id in done_configs():
                    break
                quiet = wait_for_quiet_host(vm_pid, LIMIT)
                trace = RESULTS / f"{cell_id}_try{attempt}.csv"
                row = {"cell_id": cell_id, "rep": 0, "order": attempt, "start": time.strftime("%Y-%m-%d %H:%M:%S"),
                       "trace_file": str(trace), "error": f"pre-run host outside VM {quiet}"}
                sampler = LoadSampler(vm_pid).start()
                result = subprocess.run(vm_command(imgsz, k, pose_model, trace), capture_output=True, text=True, timeout=1800)
                row.update(summarise(sampler.stop()))
                row["end"] = time.strftime("%Y-%m-%d %H:%M:%S")
                row["status"] = "ok" if result.returncode == 0 and trace.exists() else "failed"
                row["excluded"] = is_excluded(row["host_outside_vm_pct"], LIMIT)
                tail = result.stdout.strip().splitlines()[-1:] + result.stderr.strip().splitlines()[-3:]
                row["coload_cpu_s"] = next((t.split("COLOAD_CPU_S=")[1].split()[0] for t in tail if "COLOAD_CPU_S=" in t), "")
                if row["status"] == "failed":
                    row["error"] += " | " + " | ".join(tail)[:300]
                    cleanup_vm()
                writer.writerow(row)
                f.flush()
                print(f"{cell_id} try{attempt} {row['status']} outside_vm={row['host_outside_vm_pct']} "
                      f"excluded={row['excluded']}", flush=True)


def summarise_trace(trace: pd.DataFrame) -> dict:
    """ms per frame over all frames of one run (service_ms), plus the mean number of boxes the pose model saw."""
    service = trace["service_ms"]
    return {"frames": len(trace), "mean_ms": float(service.mean()), "p50_ms": float(service.quantile(0.5)),
            "p99_ms": float(service.quantile(0.99)), "mean_det_ms": float(trace[["det_pre_ms", "det_infer_ms", "det_post_ms"]].sum(axis=1).mean()),
            "mean_pose_ms": float(trace["pose_ms"].mean()), "mean_boxes": float(trace["n_boxes"].mean()),
            "share_frames_over_budget": float((service > BUDGET_MS).mean())}


def report(pose_model: str) -> dict:
    manifest = pd.read_csv(MANIFEST)
    out = {}
    for imgsz, k in CONFIGS:
        cell_id = config_id(imgsz, k, pose_model)
        rows = manifest[(manifest.cell_id == cell_id) & (manifest.status == "ok") & (~manifest.excluded.astype(bool))]
        if rows.empty:
            continue
        row = rows.iloc[-1]
        out[cell_id] = {**summarise_trace(pd.read_csv(row.trace_file)), "host_outside_vm_pct": float(row.host_outside_vm_pct),
                        "tries": int((manifest.cell_id == cell_id).sum())}
    (RESULTS / "summary.json").write_text(json.dumps(out, indent=1))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pose-model", required=True, choices=["rtmpose-s", "rtmpose-t"])
    parser.add_argument("--max-tries", type=int, default=6)
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()
    if not args.report:
        run(args.pose_model, args.max_tries)
    print(json.dumps(report(args.pose_model), indent=1))


if __name__ == "__main__":
    main()
