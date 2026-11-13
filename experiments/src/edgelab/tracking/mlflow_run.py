"""One entry point for MLflow so every run logs the same things the same way."""

import tempfile
from contextlib import contextmanager
from pathlib import Path

import mlflow

from edgelab.tracking.manifest import build_manifest, write_manifest

EXPERIMENTS_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_TRACKING_URI = f"sqlite:///{EXPERIMENTS_ROOT / 'mlflow.db'}"


def _flatten(d: dict, prefix: str = "") -> dict:
    flat = {}
    for key, value in d.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(_flatten(value, f"{name}."))
        else:
            flat[name] = value
    return flat


@contextmanager
def tracked_run(experiment: str, run_name: str, config: dict, input_files: dict | None = None,
                tracking_uri: str | None = None):
    """Usage: with tracked_run("EXP-001", "stgnf-cpu-repro", cfg) as run: mlflow.log_metric(...)"""
    mlflow.set_tracking_uri(tracking_uri or DEFAULT_TRACKING_URI)
    mlflow.set_experiment(experiment)
    manifest = build_manifest(config, input_files)
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tags({"machine": manifest["machine"], "git_sha": manifest["git_sha"],
                         "git_dirty": str(manifest["git_dirty"])})
        mlflow.log_params({k: str(v)[:500] for k, v in _flatten(config).items()})
        with tempfile.TemporaryDirectory() as tmp:
            mlflow.log_artifact(str(write_manifest(manifest, tmp)))
        yield run
