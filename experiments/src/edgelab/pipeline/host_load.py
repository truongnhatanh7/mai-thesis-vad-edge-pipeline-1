"""Host CPU load WITHOUT the VM's own usage (EXP-004 plan §3.3 + Amendment 1). macOS only.

Method: `top -l 2 -s 3` (the second sample is a real 3 s interval) gives total user+sys % of ALL cores and the
per-process %CPU (100 = one core). Host-outside-VM load = total user+sys - (CPU of the edge-os VM process) / n_cores.
The edge-os VM process is the `com.apple.Virtualization.VirtualMachine` XPC service started within a minute of the Lima
hostagent (`~/.lima/edge-os/vz.pid`); other VMs on the host (e.g. Docker's) are NOT subtracted. If the VM process is not in
top's first 40 rows its CPU is below the 40th process's CPU and is counted as 0.
"""

import os
import re
import subprocess
import threading
from pathlib import Path

ROW = re.compile(r"^\s*(\d+)\s+(.+?)\s+(\d+(?:\.\d+)?)\s*$")
CPU_LINE = re.compile(r"CPU usage:\s*([\d.]+)% user,\s*([\d.]+)% sys")


def etime_seconds(text: str) -> int:
    """ps etime '[[dd-]hh:]mm:ss' -> seconds."""
    days, _, rest = text.strip().rpartition("-")
    parts = [int(p) for p in rest.split(":")]
    seconds = 0
    for p in parts:
        seconds = seconds * 60 + p
    return seconds + (int(days) * 86400 if days else 0)


def find_vm_pid(lima_instance: str = "edge-os") -> int:
    hostagent = int((Path.home() / ".lima" / lima_instance / "vz.pid").read_text())
    ps = subprocess.run(["ps", "-axo", "pid=,etime=,comm="], capture_output=True, text=True, check=True).stdout
    rows = [line.split(None, 2) for line in ps.splitlines() if line.strip()]
    age = {int(r[0]): etime_seconds(r[1]) for r in rows}
    xpc = [int(r[0]) for r in rows if "com.apple.Virtualization.VirtualMachine" in r[2]]
    best = min(xpc, key=lambda pid: abs(age[pid] - age[hostagent]), default=None)
    if best is None or abs(age[best] - age[hostagent]) > 60:
        raise RuntimeError(f"cannot identify the {lima_instance} VM process among {xpc}")
    return best


def parse_top(text: str) -> tuple[float, float, dict[int, float]]:
    """Last sample of `top -l N` output -> (user %, sys %, {pid: %CPU})."""
    cpu = CPU_LINE.findall(text)[-1]
    last = text[text.rindex("CPU usage:"):]
    rows = {}
    for line in last.splitlines():
        m = ROW.match(line)
        if m:
            rows[int(m.group(1))] = float(m.group(3))
    return float(cpu[0]), float(cpu[1]), rows


def outside_vm_percent(user: float, sys: float, vm_cpu: float, n_cores: int) -> float:
    """Host CPU % (of all cores) used by everything except the VM; never below 0."""
    return max(0.0, user + sys - vm_cpu / n_cores)


def sample_once(vm_pid: int, n_cores: int, interval_s: int = 3) -> dict:
    out = subprocess.run(["top", "-l", "2", "-s", str(interval_s), "-n", "40", "-o", "cpu", "-stats", "pid,command,cpu"],
                         capture_output=True, text=True, check=True).stdout
    user, sys, rows = parse_top(out)
    vm_cpu = rows.get(vm_pid, 0.0)
    return {"total": user + sys, "vm_cpu": vm_cpu, "outside": outside_vm_percent(user, sys, vm_cpu, n_cores)}


class LoadSampler:
    """Background thread that keeps sampling until stop(); stop() returns the samples."""

    def __init__(self, vm_pid: int, n_cores: int | None = None, interval_s: int = 3):
        self.vm_pid, self.n_cores, self.interval_s = vm_pid, n_cores or os.cpu_count(), interval_s
        self.samples: list[dict] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            self.samples.append(sample_once(self.vm_pid, self.n_cores, self.interval_s))

    def start(self):
        self._thread.start()
        return self

    def stop(self) -> list[dict]:
        self._stop.set()
        self._thread.join()
        return self.samples


def summarise(samples: list[dict]) -> dict:
    """Means over the samples of one run: host total %, VM %CPU (one core = 100), host-outside-VM %."""
    if not samples:
        return {"n_samples": 0, "host_total_pct": None, "vm_cpu_pct": None, "host_outside_vm_pct": None}
    mean = lambda key: sum(s[key] for s in samples) / len(samples)  # noqa: E731
    return {"n_samples": len(samples), "host_total_pct": mean("total"), "vm_cpu_pct": mean("vm_cpu"),
            "host_outside_vm_pct": mean("outside")}
