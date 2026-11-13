"""EXP-018 driver pieces that need no model: dropping poses from finished files, and the pooled-difference statistics."""

import json

import numpy as np

from edgelab.analysis import exp018
from edgelab.metrics.scores import frame_level_auc
from pose_toy import toy_track

ARM_STRENGTH = {"SKIP": 0.3, "FILL-HOLD": 1.0, "+C1": 1.5, "+C2": 1.4, "DAPF": 2.0}


def test_drop_frames_dir_keeps_real_frame_ids_or_reindexes(tmp_path):
    rng = np.random.default_rng(0)
    clip = {"1": toy_track(rng, range(10)), "2": toy_track(rng, [3, 4])}
    (tmp_path / "in").mkdir()
    (tmp_path / "in" / "01_002_x_tracked_person.json").write_text(json.dumps(clip))
    mask = np.ones(10, bool)
    mask[[3, 4]] = False

    exp018.drop_frames_dir(tmp_path / "in", tmp_path / "real", {"01_002": mask}, reindex=False)
    real = json.loads((tmp_path / "real" / "01_002_x_tracked_person.json").read_text())
    assert sorted(real["1"]) == ["0000", "0001", "0002", "0005", "0006", "0007", "0008", "0009"]
    assert "2" not in real  # every pose of person 2 was dropped

    frame_map = exp018.drop_frames_dir(tmp_path / "in", tmp_path / "skip", {"01_002": mask}, reindex=True)
    skip = json.loads((tmp_path / "skip" / "01_002_x_tracked_person.json").read_text())
    assert sorted(skip["1"]) == [f"{k:04d}" for k in range(8)]
    assert list(frame_map["01_002"]) == [0, 1, 2, 5, 6, 7, 8, 9]
    assert skip["1"]["0003"] == clip["1"]["0005"]  # processed frame 3 is real frame 5
    assert list(np.load(tmp_path / "skip" / exp018.FRAME_MAPS_FILE)["01_002"]) == [0, 1, 2, 5, 6, 7, 8, 9]


def write_fake_cell(root, group, s, seed, windows_of_arm, rng, labels):
    """One score file per arm: frame score = strength * label + noise. Returns the rows. (Labels = ground truth: same in every cell.)"""
    n_videos = len(labels)
    rows = []
    for arm, strength in ARM_STRENGTH.items():
        scores = [strength * y + rng.normal(0, 1, len(y)) for y in labels]
        path = root / f"{group}_{s}_{seed}_{arm}.npz"
        np.savez(path, names=np.array([f"v{i}" for i in range(n_videos)]),
                 **{f"score_v{i}": sc for i, sc in enumerate(scores)}, **{f"label_v{i}": y for i, y in enumerate(labels)},
                 **{f"covered_v{i}": np.ones(len(y), bool) for i, y in enumerate(labels)})
        rows.append({"kind": "grid", "group": group, "s": s, "seed": seed, "arm": arm, "variant": "live", "dataset": "ShanghaiTech",
                     "drop_rate": 0.5, "auc": frame_level_auc(labels, scores), "num_segments": windows_of_arm.get(arm, 5000),
                     "scores_file": str(path), "condition": f"{group}/s{s}/seed{seed}/{arm}"})
    return rows


def test_analyse_pools_cells_and_excludes_degenerate_ones(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    labels = [(rng.random(60) < 0.4).astype(int) for _ in range(30)]
    rows = write_fake_cell(tmp_path, "SHT-L0", 3, 1, {}, rng, labels) + write_fake_cell(tmp_path, "SHT-L0", 3, 2, {"+C2": 50}, rng, labels)
    (tmp_path / "rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    monkeypatch.setattr(exp018, "OUT", tmp_path)
    monkeypatch.setattr(exp018, "ROWS", tmp_path / "rows.jsonl")
    monkeypatch.setattr(exp018, "EXPERIMENT_DIR", tmp_path)
    monkeypatch.setattr(exp018, "GROUPS", {"SHT-L0": exp018.GROUPS["SHT-L0"]})
    monkeypatch.setattr(exp018, "SLOWDOWNS", (3,))
    monkeypatch.setattr(exp018, "reproduction_check", lambda: [])

    summary = exp018.analyse(n_resamples=40)

    assert summary["degenerate"] == [["SHT-L0", 3, 2]]  # one arm has 50 windows
    assert summary["overall"]["DAPF - FILL-HOLD"]["n_cells"] == 1
    assert summary["overall_with_degenerate"]["DAPF - FILL-HOLD"]["n_cells"] == 2
    assert summary["D1"]["pass"] and summary["D2"]["pass"] and summary["D4"]["pass"]
    assert summary["D3"]["pass"] is None  # no offline rows written
    assert summary["overall"]["DAPF - FILL-HOLD"]["diff"] > 0
    low, high = summary["overall"]["DAPF - FILL-HOLD"]["ci"]
    assert low <= summary["overall"]["DAPF - FILL-HOLD"]["diff"] <= high
    assert (tmp_path / "results_tables.md").read_text().startswith("# EXP-018 results tables")


def test_offline_verdict_uses_both_fa_models_and_both_sets():
    def row(group, arm, auc):
        return {"kind": "offline", "group": group, "arm": arm, "variant": "paper", "auc": auc}
    rows = [row(g, "released", 0.80) for g in ("SHT-released", "SHT-L0")]
    rows += [row(g, "fa-cv", 0.795) for g in ("SHT-released", "SHT-L0")] + [row(g, "fa-gru", 0.80) for g in ("SHT-released", "SHT-L0")]
    assert exp018.offline_verdict(rows)["pass"] is True
    rows[-1]["auc"] = 0.785  # fa-gru on the edge poses is 1.5 points below the released model
    assert exp018.offline_verdict(rows)["pass"] is False
