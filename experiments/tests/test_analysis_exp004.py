import pandas as pd

from edgelab.analysis.exp004 import cache_for, parse_cell, usable_traces


def test_parse_cell_matches_timing_matrix_ids():
    assert parse_cell("L3-E2-R640-c0.25-O1-B2") == {"light": "L3", "enhance": "E2", "imgsz": 640, "c": 0.25,
                                                    "o": "O1", "b": "B2"}


def test_cache_names_follow_exp003_conventions():
    assert cache_for("L2", "E3", 640) == "L2_E3"
    assert cache_for("L0", "E0", 960) == "L0_E0-r960"


def test_excluded_and_failed_runs_are_not_used(tmp_path):
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame([
        {"cell_id": "A", "status": "ok", "excluded": False, "trace_file": "a0.csv"},
        {"cell_id": "A", "status": "ok", "excluded": True, "trace_file": "a1.csv"},
        {"cell_id": "A", "status": "failed", "excluded": False, "trace_file": "a2.csv"},
        {"cell_id": "B", "status": "ok", "excluded": False, "trace_file": "b0.csv"},
    ]).to_csv(manifest, index=False)
    traces = usable_traces(manifest)
    assert {k: [p.name for p in v] for k, v in traces.items()} == {"A": ["a0.csv"], "B": ["b0.csv"]}
