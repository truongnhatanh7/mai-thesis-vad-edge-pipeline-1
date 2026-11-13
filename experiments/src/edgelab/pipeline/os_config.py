"""Process-level OS knobs for the timing runs (EXP-004 plan §3.2): CPU affinity and SCHED_FIFO. Linux only.

Linux applies both per thread, so we set them on every existing thread of the process (library pools created at import
time too); threads started later inherit them from the thread that creates them. Call it again after the models are
built to catch pools that started lazily. SCHED_FIFO needs root (run the whole command under sudo, see EXP-004 log).
"""

import os
from pathlib import Path


def parse_cpus(text: str) -> set[int]:
    """'1,2,3' or '1-3' or '0,2-3' -> {1, 2, 3}."""
    cpus = set()
    for part in text.split(","):
        low, _, high = part.partition("-")
        cpus.update(range(int(low), int(high or low) + 1))
    return cpus


def apply_os_config(affinity: set[int] | None = None, fifo: int | None = None) -> None:
    if not affinity and not fifo:
        return
    for task in Path("/proc/self/task").iterdir():
        tid = int(task.name)
        try:
            if affinity:
                os.sched_setaffinity(tid, affinity)
            if fifo:
                os.sched_setscheduler(tid, os.SCHED_FIFO, os.sched_param(fifo))
        except ProcessLookupError:  # thread ended while we were looking
            continue


def thread_report() -> dict:
    """What the kernel really applied, per thread of this process: {'policies': {policy: n}, 'affinities': {cpus: n}}."""
    if not hasattr(os, "sched_getaffinity"):  # macOS
        return {}
    policies, affinities = {}, {}
    for task in Path("/proc/self/task").iterdir():
        tid = int(task.name)
        try:
            policy = {os.SCHED_OTHER: "OTHER", os.SCHED_FIFO: "FIFO", os.SCHED_RR: "RR"}.get(
                os.sched_getscheduler(tid), "other")
            cpus = ",".join(str(c) for c in sorted(os.sched_getaffinity(tid)))
        except ProcessLookupError:  # thread ended while we were looking
            continue
        policies[policy] = policies.get(policy, 0) + 1
        affinities[cpus] = affinities.get(cpus, 0) + 1
    return {"threads": sum(policies.values()), "policies": policies, "affinities": affinities}


def cgroup_report() -> dict:
    """Which cgroup this process runs in and the CPU quota of its parent (the budget slice), e.g. '200000 100000'."""
    path = Path("/proc/self/cgroup").read_text().strip().split("::")[-1]
    parent = Path("/sys/fs/cgroup") / path.strip("/") / ".."
    cpu_max = (parent / "cpu.max").read_text().strip() if (parent / "cpu.max").exists() else None
    return {"cgroup": path, "parent_cpu_max": cpu_max}
