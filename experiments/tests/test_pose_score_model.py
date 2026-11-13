"""score_model reproduces the released scoring path, and reads provenance from the filler's sidecars."""

import json

import numpy as np
import torch

from edgelab.models.stgnf_flow import build_flow
from edgelab.pose.evaluate import CHECKPOINT
from edgelab.pose.fill import fill_dir
from edgelab.pose.score_model import load_model, parameter_count, score_segments_model, segment_provenance
from edgelab.pose.stgnf_eval import score_segments
from pose_toy import write_toy_dir


def test_released_checkpoint_through_score_model_equals_stgnf_eval(tmp_path):
    write_toy_dir(tmp_path)
    ours = score_segments_model(load_model(CHECKPOINT), tmp_path, threads=1)
    theirs = score_segments(tmp_path, CHECKPOINT, threads=1)
    assert ours.num_segments == theirs.num_segments > 0
    assert np.array_equal(ours.metadata, theirs.metadata)
    assert np.allclose(ours.normality, theirs.normality, atol=1e-5)
    assert parameter_count(load_model(CHECKPOINT)) == 616


def test_provenance_is_zero_on_predicted_poses_and_one_everywhere_without_sidecars(tmp_path):
    poses = write_toy_dir(tmp_path / "poses")
    # person 2 of clip 01_001 has holes at 40 and 80-82 (toy_dir); we add 60-62. P fill puts them back as predictions and
    # also predicts 8 frames after the last detection of every track, so every track becomes continuous.
    path = poses / "01_001_alphapose_tracked_person.json"
    clip = json.loads(path.read_text())
    for frame in (60, 61, 62):
        clip["2"].pop(f"{frame:04d}")
    path.write_text(json.dumps(clip))
    filled = tmp_path / "filled"
    fill_dir(poses, filled, "P", 8)
    predicted = {(1, 2): {40, 60, 61, 62, 80, 81, 82, *range(120, 128)}, (1, 1): set(range(100, 108)),
                 (1, 3): set(range(30, 38)), (2, 1): set(range(90, 98))}  # (scene, person): predicted frames
    model = build_flow("graph")
    model.train()
    model(torch.randn(8, 2, 24, 18))
    segments = score_segments_model(model, filled, threads=1)  # any model: we only need the metadata
    prov = segment_provenance(segments.metadata, filled, filled.with_name("filled_prov"))
    assert prov.shape == (segments.num_segments, 24)
    for row, (scene, clip_id, person, start) in enumerate(segments.metadata):
        assert prov[row].tolist() == [0.0 if f in predicted[(scene, person)] else 1.0 for f in range(start, start + 24)]
    assert (prov == 0).any()
    assert (segment_provenance(segments.metadata, filled, None) == 1).all()


def test_flow_with_provenance_scores_through_the_same_path(tmp_path):
    poses = write_toy_dir(tmp_path / "poses")
    model = build_flow("graph-fa")
    model.train()
    model(torch.randn(8, 2, 24, 18), torch.ones(8, 24))
    scores = score_segments_model(model, poses, prov_dir=None, threads=1)
    assert np.isfinite(scores.normality).all() and len(scores.normality) == scores.num_segments
