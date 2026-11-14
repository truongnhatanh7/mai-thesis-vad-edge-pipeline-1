"""Raspberry Pi 5 real-time hardware telemetry collector (PMIC power/rails, thermal zone, throttling, cpufreq, CPU/RAM).

Samples Linux sysfs and `vcgencmd` in a low-overhead background thread (< 0.1% CPU) and computes
exact board power in Watts from the Raspberry Pi 5 DA9091 PMIC ADC rails (`vcgencmd pmic_read_adc`),
plus SoC temperature (°C), throttling bitmask (`vcgencmd get_throttled`), Cortex-A76 per-core clock
frequencies (MHz), system loadavg, CPU utilization %, and process RSS memory (MB).
"""

from __future__ import annotations

import os
import re
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import psutil

# Matching current (A) and voltage (V) rail prefixes on Raspberry Pi 5 DA9091 PMIC
PMIC_RAIL_PAIRS = [
    ("3V7_WL_SW_A", "3V7_WL_SW_V"),
    ("3V3_SYS_A", "3V3_SYS_V"),
    ("1V8_SYS_A", "1V8_SYS_V"),
    ("DDR_VDD2_A", "DDR_VDD2_V"),
    ("DDR_VDDQ_A", "DDR_VDDQ_V"),
    ("1V1_SYS_A", "1V1_SYS_V"),
    ("0V8_SW_A", "0V8_SW_V"),
    ("VDD_CORE_A", "VDD_CORE_V"),
    ("3V3_DAC_A", "3V3_DAC_V"),
    ("3V3_ADC_A", "3V3_ADC_V"),
    ("0V8_AON_A", "0V8_AON_V"),
    ("HDMI_A", "HDMI_V"),
]

_ADC_RE = re.compile(r"^\s*([A-Z0-9_]+)\s+(?:current|volt)\(\d+\)=([0-9.]+)[AV]", re.MULTILINE)


def read_pmic_power() -> dict[str, Any]:
    """Read Raspberry Pi 5 DA9091 PMIC rails via `vcgencmd pmic_read_adc` and compute total & core Watts."""
    try:
        out = subprocess.check_output(["vcgencmd", "pmic_read_adc"], text=True, timeout=1.0)
    except Exception:
        return {"power_total_w": float("nan"), "power_core_w": float("nan"), "ext5v_v": float("nan"), "rails": {}}

    vals: dict[str, float] = {}
    for m in _ADC_RE.finditer(out):
        vals[m.group(1)] = float(m.group(2))

    total_w = 0.0
    rail_watts: dict[str, float] = {}
    for cur_k, volt_k in PMIC_RAIL_PAIRS:
        if cur_k in vals and volt_k in vals:
            w = vals[cur_k] * vals[volt_k]
            rail_name = cur_k[:-2]
            rail_watts[rail_name] = round(w, 4)
            total_w += w

    core_w = rail_watts.get("VDD_CORE", float("nan"))
    return {
        "power_total_w": round(total_w, 4) if rail_watts else float("nan"),
        "power_core_w": round(core_w, 4) if np.isfinite(core_w) else float("nan"),
        "ext5v_v": round(vals.get("EXT5V_V", float("nan")), 4),
        "vdd_core_v": round(vals.get("VDD_CORE_V", float("nan")), 4),
        "vdd_core_a": round(vals.get("VDD_CORE_A", float("nan")), 4),
        "rails_w": rail_watts,
    }


def read_soc_temp_c() -> float:
    """Read SoC temperature in Celsius from sysfs (fast) with vcgencmd fallback."""
    tz = Path("/sys/class/thermal/thermal_zone0/temp")
    if tz.exists():
        try:
            return round(int(tz.read_text().strip()) / 1000.0, 2)
        except Exception:
            pass
    try:
        out = subprocess.check_output(["vcgencmd", "measure_temp"], text=True, timeout=1.0)
        m = re.search(r"([0-9.]+)", out)
        if m:
            return float(m.group(1))
    except Exception:
        pass
    return float("nan")


def read_throttled() -> dict[str, Any]:
    """Read Raspberry Pi throttling bitmask (`0x0` = healthy, no under-voltage or thermal capping)."""
    try:
        out = subprocess.check_output(["vcgencmd", "get_throttled"], text=True, timeout=1.0).strip()
        hex_str = out.split("=")[-1] if "=" in out else out
        code = int(hex_str, 16)
        return {
            "throttled_hex": hex_str,
            "throttled_int": code,
            "under_voltage_now": bool(code & 0x1),
            "freq_capped_now": bool(code & 0x2),
            "throttled_now": bool(code & 0x4),
            "soft_temp_limit_now": bool(code & 0x8),
            "any_throttled_ever": bool(code != 0),
        }
    except Exception:
        return {"throttled_hex": "unknown", "throttled_int": -1, "any_throttled_ever": False}


def read_cpu_freqs_mhz() -> list[float]:
    """Read current scaling frequency (MHz) for all CPU cores."""
    freqs: list[float] = []
    for cpu in range(os.cpu_count() or 4):
        p = Path(f"/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_cur_freq")
        if p.exists():
            try:
                freqs.append(round(int(p.read_text().strip()) / 1000.0, 1))
            except Exception:
                pass
    return freqs


def read_board_info() -> dict[str, Any]:
    """Read static Raspberry Pi hardware model, revision, kernel, and governor info."""
    model = Path("/proc/device-tree/model").read_text(errors="ignore").rstrip("\x00").strip() if Path("/proc/device-tree/model").exists() else "unknown"
    gov_p = Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
    governor = gov_p.read_text().strip() if gov_p.exists() else "unknown"
    vm = psutil.virtual_memory()
    return {
        "board_model": model,
        "cpu_governor": governor,
        "cpu_count": os.cpu_count(),
        "ram_total_gb": round(vm.total / (1024**3), 2),
        "initial_temp_c": read_soc_temp_c(),
        "initial_throttled": read_throttled(),
        "initial_pmic": read_pmic_power(),
    }


class Pi5TelemetryMonitor:
    """Background hardware telemetry sampler for Raspberry Pi 5 experiments."""

    def __init__(self, interval_s: float = 0.25, pid: int | None = None):
        self.interval_s = interval_s
        self.proc = psutil.Process(pid or os.getpid())
        self.samples: list[dict[str, Any]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._t0 = 0.0

    def sample_once(self) -> dict[str, Any]:
        now = time.perf_counter()
        pmic = read_pmic_power()
        temp_c = read_soc_temp_c()
        thr = read_throttled()
        freqs = read_cpu_freqs_mhz()
        try:
            rss_mb = round(self.proc.memory_info().rss / (1024**2), 2)
            proc_cpu_pct = round(self.proc.cpu_percent(interval=None), 1)
        except Exception:
            rss_mb = float("nan")
            proc_cpu_pct = float("nan")
        sys_cpu_pct = round(psutil.cpu_percent(interval=None), 1)
        la1, la5, la15 = os.getloadavg()
        rec = {
            "elapsed_s": round(now - self._t0, 3),
            "timestamp_utc": time.time(),
            "temp_c": temp_c,
            "power_total_w": pmic["power_total_w"],
            "power_core_w": pmic["power_core_w"],
            "ext5v_v": pmic["ext5v_v"],
            "vdd_core_v": pmic["vdd_core_v"],
            "vdd_core_a": pmic["vdd_core_a"],
            "cpu_freq_mhz_mean": round(float(np.mean(freqs)), 1) if freqs else float("nan"),
            "cpu_freq_mhz_min": round(float(np.min(freqs)), 1) if freqs else float("nan"),
            "sys_cpu_pct": sys_cpu_pct,
            "proc_cpu_pct": proc_cpu_pct,
            "rss_mb": rss_mb,
            "loadavg_1m": round(la1, 2),
            "throttled_hex": thr["throttled_hex"],
            "throttled_int": thr["throttled_int"],
        }
        return rec

    def _loop(self) -> None:
        psutil.cpu_percent(interval=None)
        try:
            self.proc.cpu_percent(interval=None)
        except Exception:
            pass
        while not self._stop.is_set():
            self.samples.append(self.sample_once())
            self._stop.wait(self.interval_s)

    def start(self) -> "Pi5TelemetryMonitor":
        self._t0 = time.perf_counter()
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
        if not self.samples:
            self.samples.append(self.sample_once())
        return self.summarize()

    def summarize(self) -> dict[str, Any]:
        temps = [s["temp_c"] for s in self.samples if np.isfinite(s["temp_c"])]
        pw_tot = [s["power_total_w"] for s in self.samples if np.isfinite(s["power_total_w"])]
        pw_core = [s["power_core_w"] for s in self.samples if np.isfinite(s["power_core_w"])]
        rss = [s["rss_mb"] for s in self.samples if np.isfinite(s["rss_mb"])]
        sys_cpu = [s["sys_cpu_pct"] for s in self.samples if np.isfinite(s["sys_cpu_pct"])]
        proc_cpu = [s["proc_cpu_pct"] for s in self.samples if np.isfinite(s["proc_cpu_pct"]) and s["proc_cpu_pct"] > 0]
        freqs = [s["cpu_freq_mhz_mean"] for s in self.samples if np.isfinite(s["cpu_freq_mhz_mean"])]
        thr_codes = [s["throttled_int"] for s in self.samples if s["throttled_int"] >= 0]
        duration_s = self.samples[-1]["elapsed_s"] if self.samples else 0.0
        mean_w = float(np.mean(pw_tot)) if pw_tot else float("nan")
        return {
            "n_samples": len(self.samples),
            "duration_s": round(duration_s, 2),
            "temp_c_mean": round(float(np.mean(temps)), 2) if temps else float("nan"),
            "temp_c_max": round(float(np.max(temps)), 2) if temps else float("nan"),
            "temp_c_min": round(float(np.min(temps)), 2) if temps else float("nan"),
            "power_total_w_mean": round(mean_w, 3) if pw_tot else float("nan"),
            "power_total_w_p95": round(float(np.percentile(pw_tot, 95)), 3) if pw_tot else float("nan"),
            "power_total_w_max": round(float(np.max(pw_tot)), 3) if pw_tot else float("nan"),
            "power_core_w_mean": round(float(np.mean(pw_core)), 3) if pw_core else float("nan"),
            "energy_total_joules": round(mean_w * duration_s, 2) if np.isfinite(mean_w) else float("nan"),
            "rss_mb_mean": round(float(np.mean(rss)), 1) if rss else float("nan"),
            "rss_mb_peak": round(float(np.max(rss)), 1) if rss else float("nan"),
            "sys_cpu_pct_mean": round(float(np.mean(sys_cpu)), 1) if sys_cpu else float("nan"),
            "sys_cpu_pct_peak": round(float(np.max(sys_cpu)), 1) if sys_cpu else float("nan"),
            "proc_cpu_pct_mean": round(float(np.mean(proc_cpu)), 1) if proc_cpu else float("nan"),
            "cpu_freq_mhz_mean": round(float(np.mean(freqs)), 1) if freqs else float("nan"),
            "throttled_any": any(c != 0 for c in thr_codes),
            "throttled_codes_seen": sorted(list({ hex(c) for c in thr_codes })),
        }
