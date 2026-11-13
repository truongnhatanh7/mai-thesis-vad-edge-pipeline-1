"""Guards the matrix runner's bookkeeping: cell list, frozen random order, host-load exclusion, resume, host-load parsing."""

from pathlib import Path

import pandas as pd
import pytest

from edgelab.pipeline import timing_matrix as tm
from edgelab.pipeline.host_load import etime_seconds, outside_vm_percent, parse_top, summarise


def test_cell_list_is_the_30_cells_of_amendment_1():
    cells = tm.build_cells(gxfit_c=0.15)
    assert len(cells) == 30 and len({c.id for c in cells}) == 30
    assert sum(c.imgsz == 640 and c.light == "L3" and c.c == 0.25 for c in cells) == 16  # P x O x B
    assert sorted({c.imgsz for c in cells if c.enhance == "E0" and c.light == "L3" and c.c == 0.25}) == [480, 640, 800, 960]
    extra = [c for c in cells if (c.light, c.c) != ("L3", 0.25)]
    assert {(c.light, c.c, c.o, c.b, c.enhance) for c in extra} == {("L0", 0.25, "O0", "B4", "E0"),
                                                                    ("L3", 0.15, "O0", "B4", "E0")}


def test_plan_each_repetition_is_a_permutation_of_all_cells_and_is_seeded():
    cells = tm.build_cells(0.15)
    plan = tm.make_plan(cells, reps=5, seed=0)
    assert [p[0] for p in plan] == list(range(150))
    for rep in range(5):
        assert sorted(c.id for _, c, r in plan if r == rep) == sorted(c.id for c in cells)
    assert plan == tm.make_plan(cells, reps=5, seed=0)
    assert [c for _, c, _ in plan] != [c for _, c, _ in tm.make_plan(cells, reps=5, seed=1)]
    assert [c for _, c, r in plan if r == 0] != [c for _, c, r in plan if r == 1]  # the order differs between repetitions


def test_plan_file_is_written_once_and_must_not_change(tmp_path):
    path = tmp_path / "timing_plan.csv"
    plan = tm.make_plan(tm.build_cells(0.15))
    tm.write_plan(plan, path)
    first = path.read_text()
    tm.write_plan(plan, path)  # same plan again: fine
    assert path.read_text() == first and first.splitlines()[0].startswith("order,cell_id,rep")
    with pytest.raises(ValueError):
        tm.write_plan(tm.make_plan(tm.build_cells(0.2)), path)  # different G-xfit gate -> different plan


def test_exclusion_rule_is_strictly_above_30_percent_and_unknown_load_is_excluded():
    assert not tm.is_excluded(30.0) and tm.is_excluded(30.01) and not tm.is_excluded(0.0)
    assert tm.is_excluded(None)


def test_done_runs_only_counts_status_ok_so_failed_runs_are_retried(tmp_path):
    manifest = tmp_path / "manifest.csv"
    assert tm.done_runs(manifest) == set()
    pd.DataFrame([{"cell_id": "a", "rep": 0, "status": "ok"}, {"cell_id": "a", "rep": 1, "status": "failed"},
                  {"cell_id": "b", "rep": 0, "status": "ok"}]).to_csv(manifest, index=False)
    assert tm.done_runs(manifest) == {("a", 0), ("b", 0)}


def test_vm_command_runs_the_cell_script_with_budget_co_load_and_os_knobs(tmp_path):
    cell = tm.Cell("L3", "E0", 800, 0.25, "O1", "B2")
    command = tm.vm_command(cell, 3, tmp_path / "x.csv")
    assert command[:7] == ["limactl", "shell", "edge-os", "--", "sudo", "bash", str(tm.CELL_SCRIPT)]
    assert command[7:9] == ["B2", "O1"]
    args = command[9:]
    assert args[args.index("--imgsz") + 1] == "800" and args[args.index("--max-frames-per-video") + 1] == "40"
    assert args[args.index("--fifo") + 1] == "10" and args[args.index("--affinity") + 1] == "1,2,3"
    assert "--fifo" not in tm.vm_command(tm.Cell("L3", "E0", 640, 0.25, "O0", "B4"), 0, tmp_path / "y.csv")


def test_estimate_scales_detector_time_with_input_pixels():
    smoke = {"E0": {"total_ms": 100.0, "det_infer_ms": 60.0}, "E3": {"total_ms": 300.0, "det_infer_ms": 80.0}}
    small = tm.estimate_seconds(tm.Cell("L3", "E0", 480, 0.25, "O0", "B4"), smoke, frames=10, overhead_s=0)
    big = tm.estimate_seconds(tm.Cell("L3", "E0", 960, 0.25, "O0", "B4"), smoke, frames=10, overhead_s=0)
    assert small == pytest.approx(10 * (40 + 60 * (288 * 480) / 640 ** 2) / 1000)
    assert small < big
    assert tm.estimate_seconds(tm.Cell("L3", "E3", 640, 0.25, "O0", "B4"), smoke, 10, 0) > big


TOP = """Processes: 5 total
CPU usage: 1.0% user, 2.0% sys, 97.0% idle
PID    COMMAND          %CPU
1      kernel_task      5.0
2      stale            9.0
Processes: 5 total
CPU usage: 40.5% user, 20.0% sys, 39.5% idle
PID    COMMAND          %CPU
1      kernel_task      30.0
13488  com.apple.Virtua 250.5
77     Google Chrome H  12.5
"""


def test_parse_top_reads_the_last_sample_only():
    user, sys, rows = parse_top(TOP)
    assert (user, sys) == (40.5, 20.0)
    assert rows == {1: 30.0, 13488: 250.5, 77: 12.5}  # names with spaces parse; the first sample is ignored


def test_host_load_outside_the_vm_subtracts_the_vm_process_share_of_all_cores():
    assert outside_vm_percent(40.5, 20.0, vm_cpu=250.0, n_cores=10) == pytest.approx(35.5)  # VM = 25% of the host
    assert outside_vm_percent(10.0, 5.0, vm_cpu=400.0, n_cores=10) == 0.0  # never negative


def test_summarise_means_and_empty_case():
    out = summarise([{"total": 40.0, "vm_cpu": 100.0, "outside": 30.0}, {"total": 60.0, "vm_cpu": 200.0, "outside": 50.0}])
    assert out == {"n_samples": 2, "host_total_pct": 50.0, "vm_cpu_pct": 150.0, "host_outside_vm_pct": 40.0}
    assert summarise([])["host_outside_vm_pct"] is None


def test_etime_seconds():
    assert etime_seconds("11:10:35") == 40235 and etime_seconds("46-11:28:51") == 46 * 86400 + 41331 and etime_seconds("05:07") == 307
