"""Timing matrix runner (EXP-004 plan §3.3 + Amendment 1, §6). Runs on the HOST and drives the VM through limactl.

    uv run --extra pose python -m edgelab.pipeline.timing_matrix --gxfit-c 0.15 [--dry-run]

Cells (all L3 unless stated, c 0.25 unless stated, with co-load): P {E0..E3} x O {O0,O1} x B {B4,B2} at R=640 (16);
R {480,800,960} x O x B at E0 (12); cost checks at E0/O0/B4: light L0, and c = G-xfit (2)  -> 30 cells x 5 repetitions.
Order: every repetition is one random permutation of all cells (seed 0), written to timing_plan.csv BEFORE the first run.
Resumable: runs whose (cell, rep) already have status ok in the manifest are skipped.
"""

import argparse
import csv
import random
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from edgelab.pipeline.host_load import LoadSampler, find_vm_pid, summarise

EXCHANGE = Path("/Users/nanh/thesis-data/vm-exchange")
EXPERIMENT_DIR = Path(__file__).resolve().parents[3] / "EXP-004-os-deadlines"
PLAN_FILE = EXPERIMENT_DIR / "timing_plan.csv"
RESULTS_DIR = EXCHANGE / "results" / "matrix"
MANIFEST = RESULTS_DIR / "manifest.csv"
CELL_SCRIPT = EXPERIMENT_DIR / "scripts" / "cell_in_vm.sh"
SUBSET_FILE = EXPERIMENT_DIR / "timing_subset.txt"
REPS, SEED, FRAMES_PER_VIDEO, HOST_LOAD_LIMIT = 5, 0, 40, 30.0
# O0 = library defaults; O1 = tuned (plan §3.2). The co-load placement for O1 is done by cell_in_vm.sh.
O_ARGS = {"O0": [], "O1": ["--threads", "3", "--spin", "off", "--affinity", "1,2,3", "--fifo", "10"],
          # Amendment 2: O1 without SCHED_FIFO (cgroup v2 cpu.max does not throttle RT tasks, so O1 escaped B2)
          "O1n": ["--threads", "3", "--spin", "off", "--affinity", "1,2,3"],
          "O0r": []}  # Amendment 2: O0 re-run in the same window; own id so finished main-block runs are not skipped
INPUT_HW = {480: (288, 480), 640: (384, 640), 800: (480, 800), 960: (544, 960)}  # EXP-004 log 10:40
MANIFEST_COLUMNS = ["cell_id", "rep", "order", "start", "end", "status", "excluded", "host_outside_vm_pct",
                    "host_total_pct", "vm_cpu_pct", "n_samples", "coload_cpu_s", "trace_file", "error"]


@dataclass(frozen=True)
class Cell:
    light: str
    enhance: str
    imgsz: int
    c: float
    o: str
    b: str

    @property
    def id(self) -> str:
        return f"{self.light}-{self.enhance}-R{self.imgsz}-c{self.c}-{self.o}-{self.b}"


def build_cells(gxfit_c: float) -> list[Cell]:
    os_configs = [(o, b) for o in ("O0", "O1") for b in ("B4", "B2")]
    cells = [Cell("L3", p, 640, 0.25, o, b) for p in ("E0", "E1", "E2", "E3") for o, b in os_configs]
    cells += [Cell("L3", "E0", r, 0.25, o, b) for r in (480, 800, 960) for o, b in os_configs]
    cells += [Cell("L0", "E0", 640, 0.25, "O0", "B4"), Cell("L3", "E0", 640, gxfit_c, "O0", "B4")]
    return cells


def amendment2_cells() -> list[Cell]:
    """Plan §7: O1n at B4 and B2, plus O0 at B2 re-run in the same time window, for 3 enhancers at 640 and E0 at 960."""
    knobs = [("E0", 640), ("E2", 640), ("E3", 640), ("E0", 960)]
    configs = [("O1n", "B4"), ("O1n", "B2"), ("O0r", "B2")]
    return [Cell("L3", p, r, 0.25, o, b) for p, r in knobs for o, b in configs]


def make_plan(cells: list[Cell], reps: int = REPS, seed: int = SEED) -> list[tuple[int, Cell, int]]:
    """[(order index, cell, rep)]: repetition r is one random permutation of all cells, so each cell appears once per
    repetition and slow drift of the host cannot line up with one cell."""
    plan = []
    for rep in range(reps):
        for cell in random.Random(f"{seed}|{rep}").sample(cells, len(cells)):
            plan.append((len(plan), cell, rep))
    return plan


def write_plan(plan, path: Path = PLAN_FILE) -> None:
    """Write the planned order once; if the file exists it must be identical (the order is frozen)."""
    lines = ["order,cell_id,rep,light,enhance,imgsz,c,O,B"] + [
        f"{i},{c.id},{rep},{c.light},{c.enhance},{c.imgsz},{c.c},{c.o},{c.b}" for i, c, rep in plan]
    text = "\n".join(lines) + "\n"
    if path.exists():
        if path.read_text() != text:
            raise ValueError(f"{path} exists and differs from the plan just built (different cells or seed)")
        return
    path.write_text(text)


def is_excluded(host_outside_vm_pct: float | None, limit: float = HOST_LOAD_LIMIT) -> bool:
    """Plan §3.3: excluded if the mean host load outside the VM is above 30% of host CPU. Unknown load counts as excluded."""
    return host_outside_vm_pct is None or host_outside_vm_pct > limit


def done_runs(manifest: Path = MANIFEST) -> set[tuple[str, int]]:
    if not manifest.exists():
        return set()
    rows = pd.read_csv(manifest)
    return {(r.cell_id, int(r.rep)) for r in rows.itertuples() if r.status == "ok"}


def live_timing_args(cell: Cell, rep: int, trace: Path) -> list[str]:
    return ["--frames-root", str(EXCHANGE / "frames"), "--videos-file", str(SUBSET_FILE), "--level", cell.light,
            "--enhance", cell.enhance, "--c", str(cell.c), "--imgsz", str(cell.imgsz), "--max-frames-per-video",
            str(FRAMES_PER_VIDEO), "--out", str(trace), "--tag", f"{cell.id}-rep{rep}", *O_ARGS[cell.o]]


def vm_command(cell: Cell, rep: int, trace: Path) -> list[str]:
    return ["limactl", "shell", "edge-os", "--", "sudo", "bash", str(CELL_SCRIPT), cell.b, cell.o,
            *live_timing_args(cell, rep, trace)]


def estimate_seconds(cell: Cell, smoke: dict, frames: int = 12 * FRAMES_PER_VIDEO, overhead_s: float = 20.0) -> float:
    """Lower-bound wall time of one run from the 07:50 smoke means (E0/E3 at 640x640, O0, no co-load, no quota).

    Per frame: smoke total (read + darken + enhance + detect + track + flow) with the detector's inference scaled by
    input pixels relative to 640x640. E1/E2 use the E0 smoke (their cost was not measured). B2/O1 slow-downs are not modelled.
    """
    base = smoke["E3"] if cell.enhance == "E3" else smoke["E0"]
    h, w = INPUT_HW[cell.imgsz]
    per_frame_ms = base["total_ms"] - base["det_infer_ms"] + base["det_infer_ms"] * (h * w) / (640 * 640)
    return frames * per_frame_ms / 1000 + overhead_s


def load_smoke(results: Path = EXCHANGE / "results") -> dict:
    cols = ["total_ms", "det_infer_ms"]
    return {e: pd.read_csv(results / f"smoke_L0_{e}-O0.csv")[cols].mean().to_dict() for e in ("E0", "E3")}


def print_dry_run(plan, smoke: dict, done: set) -> float:
    cells = {c.id: c for _, c, _ in plan}
    per_run = {cid: estimate_seconds(c, smoke) for cid, c in cells.items()}
    todo = [(i, c, r) for i, c, r in plan if (c.id, r) not in done]
    total_s = sum(per_run[c.id] for _, c, _ in todo)
    print(f"{len(cells)} cells x {REPS} reps = {len(plan)} runs; {len(plan) - len(todo)} already done; {len(todo)} to run")
    for i, c, r in plan[:8]:
        print(f"  order {i:3d}  rep {r}  {c.id}  ~{per_run[c.id]:.0f} s")
    print("  ...")
    print(f"lower-bound total {total_s / 3600:.1f} h (smoke means, no B2/O1/co-load slow-down); "
          f"x1.5 bracket {1.5 * total_s / 3600:.1f} h")
    return total_s


def cleanup_vm() -> None:
    """After a failed or timed-out run: stop leftover scopes (pipeline, co-load) and processes in the VM."""
    subprocess.run(["limactl", "shell", "edge-os", "--", "sudo", "bash", "-c",
                    "systemctl stop 'edgelab-*.scope'; pkill -f edgelab.pipeline.live_timing; true"],
                   capture_output=True, text=True, timeout=60)


def run_plan(plan, done: set, cooldown_s: int, timeout_s: int = 1800, max_failures: int = 3) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    vm_pid = find_vm_pid()
    new_file = not MANIFEST.exists()
    failures = 0
    with open(MANIFEST, "a", newline="") as f:
        writer = csv.DictWriter(f, MANIFEST_COLUMNS)
        if new_file:
            writer.writeheader()
        for order, cell, rep in plan:
            if (cell.id, rep) in done:
                continue
            trace = RESULTS_DIR / f"{cell.id}_rep{rep}_o{order}.csv"
            row = {"cell_id": cell.id, "rep": rep, "order": order, "start": time.strftime("%Y-%m-%d %H:%M:%S"),
                   "trace_file": str(trace), "error": ""}
            sampler = LoadSampler(vm_pid).start()
            try:
                result = subprocess.run(vm_command(cell, rep, trace), capture_output=True, text=True, timeout=timeout_s)
            except subprocess.TimeoutExpired:
                result = subprocess.CompletedProcess([], 124, "", f"timeout after {timeout_s} s")
            row.update(summarise(sampler.stop()))
            row["end"] = time.strftime("%Y-%m-%d %H:%M:%S")
            row["status"] = "ok" if result.returncode == 0 and trace.exists() else "failed"
            row["excluded"] = is_excluded(row["host_outside_vm_pct"])
            tail = result.stdout.strip().splitlines()[-1:] + result.stderr.strip().splitlines()[-3:]
            row["coload_cpu_s"] = next((t.split("COLOAD_CPU_S=")[1].split()[0] for t in tail if "COLOAD_CPU_S=" in t), "")
            row["error"] = "" if row["status"] == "ok" else " | ".join(tail)[:300]
            if row["status"] == "failed":
                cleanup_vm()
            writer.writerow(row)
            f.flush()
            print(f"[{order}] {cell.id} rep{rep} {row['status']} host_outside_vm={row['host_outside_vm_pct']} "
                  f"excluded={row['excluded']}", flush=True)
            failures = failures + 1 if row["status"] == "failed" else 0
            if failures >= max_failures:
                raise RuntimeError(f"{failures} failed runs in a row, stopping; see {MANIFEST}")
            time.sleep(cooldown_s)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gxfit-c", type=float, required=True, help="the G-xfit gate chosen in EXP-003 (cost check, H10)")
    parser.add_argument("--dry-run", action="store_true", help="print the plan and the estimated duration, run nothing")
    parser.add_argument("--cooldown-s", type=int, default=5)
    parser.add_argument("--timeout-s", type=int, default=1800, help="per run; a run longer than this is marked failed")
    parser.add_argument("--amendment2", action="store_true", help="run the Amendment 2 cells (plan §7) instead")
    args = parser.parse_args()
    cells = amendment2_cells() if args.amendment2 else build_cells(args.gxfit_c)
    # Amendment 2 uses seed 2 so its order is not a copy of the main block's; the main block keeps seed 0
    plan = make_plan(cells, seed=2) if args.amendment2 else make_plan(cells)
    if args.dry_run:
        print_dry_run(plan, load_smoke(), done_runs())
        return
    write_plan(plan, EXPERIMENT_DIR / "timing_plan_amendment2.csv" if args.amendment2 else PLAN_FILE)
    run_plan(plan, done_runs(), args.cooldown_s, args.timeout_s)


if __name__ == "__main__":
    main()
