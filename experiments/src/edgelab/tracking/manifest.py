"""Run manifest: everything needed to say later exactly what produced a number."""

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

TRACKED_PACKAGES = ["numpy", "scipy", "scikit-learn", "torch", "onnxruntime", "opencv-python-headless", "mlflow", "ultralytics"]


def sha256_of(path: str | Path, chunk_bytes: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _package_versions() -> dict[str, str]:
    versions = {}
    for name in TRACKED_PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pass
    return versions


def machine_label() -> str:
    """'mac', 'vm' or 'pi' — VM and Pi timings must never be mixed (D8)."""
    if os.environ.get("EDGELAB_MACHINE"):
        return os.environ["EDGELAB_MACHINE"]
    if sys.platform == "darwin":
        return "mac"
    model_file = Path("/proc/device-tree/model")
    if model_file.exists() and "Raspberry Pi" in model_file.read_text(errors="ignore"):
        return "pi"
    return "vm"


def build_manifest(config: dict, input_files: dict[str, str | Path] | None = None) -> dict:
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "machine": machine_label(),
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
        "python": sys.version.split()[0],
        "packages": _package_versions(),
        "git_sha": _git("rev-parse", "HEAD"),
        "git_dirty": _git("status", "--porcelain") not in ("", "unknown"),
        "config": config,
        "input_sha256": {name: sha256_of(p) for name, p in (input_files or {}).items()},
        "env_threads": {k: os.environ[k] for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS") if k in os.environ},
    }


def write_manifest(manifest: dict, out_dir: str | Path) -> Path:
    out = Path(out_dir) / "manifest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return out
