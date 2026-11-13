"""The two trainers (STG-NF variants, C2 predictor) on a tiny synthetic training set: they finish, write their files, and resume."""

import json

import torch
from pose_toy import write_toy_dir

from edgelab.pose import predictor_train, stgnf_train
from edgelab.pose.segments import build_bank


def use_toy_bank(monkeypatch, tmp_path):
    poses = write_toy_dir(tmp_path / "poses")
    bank = build_bank(poses, seg_len=24, stride=6, ctx=16)
    monkeypatch.setattr(stgnf_train, "load_bank", lambda dataset: bank)
    monkeypatch.setattr(predictor_train, "load_bank", lambda dataset: bank)
    return bank


def weights_equal(path_a, path_b) -> bool:
    a = torch.load(path_a, weights_only=False)["state_dict"]
    b = torch.load(path_b, weights_only=False)["state_dict"]
    return all(torch.equal(a[k], b[k]) for k in a)


def test_stgnf_training_resumes_to_the_same_weights(tmp_path, monkeypatch):
    use_toy_bank(monkeypatch, tmp_path)
    straight = stgnf_train.train("graph", "ShanghaiTech", seed=0, out_dir=tmp_path / "a", epochs=2, batch_size=16, threads=1)

    stgnf_train.train("graph", "ShanghaiTech", seed=0, out_dir=tmp_path / "b", epochs=1, batch_size=16, threads=1)
    for name in ("DONE", "final.pt", "config.json"):  # as if the process died after epoch 1 of 2
        (tmp_path / "b" / name).unlink()
    resumed = stgnf_train.train("graph", "ShanghaiTech", seed=0, out_dir=tmp_path / "b", epochs=2, batch_size=16, threads=1)

    assert "resumed after epoch 1" in (tmp_path / "b" / "train.log").read_text()
    assert weights_equal(straight, resumed)
    assert json.loads((tmp_path / "b" / "config.json").read_text())["params"] == 616
    assert stgnf_train.train("graph", "ShanghaiTech", seed=0, out_dir=tmp_path / "b", epochs=2) == resumed  # DONE: nothing re-run


def test_fa_training_with_constant_velocity_fill_runs(tmp_path, monkeypatch):
    use_toy_bank(monkeypatch, tmp_path)
    final = stgnf_train.train("graph-fa", "ShanghaiTech", seed=0, out_dir=tmp_path / "fa", epochs=1, batch_size=16, fill="cv", threads=1)
    saved = torch.load(final, weights_only=False)
    assert saved["kind"] == "graph-fa" and saved["config"]["fill"] == "cv" and saved["config"]["params"] == 576


def test_predictor_training_finishes_and_writes_history(tmp_path, monkeypatch):
    use_toy_bank(monkeypatch, tmp_path)
    out = tmp_path / "pred"
    predictor_train.train("ShanghaiTech", out, max_epochs=2, batch_size=16, threads=1)
    assert (out / "predictor.pt").exists() and (out / "DONE").exists()
    history = json.loads((out / "history.json").read_text())
    assert history  # one entry per epoch, with the validation error of the GRU and of constant velocity
