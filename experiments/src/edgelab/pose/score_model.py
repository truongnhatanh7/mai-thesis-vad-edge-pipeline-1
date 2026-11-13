"""Score a pose folder with any STG-NF-style model (released checkpoint, our retrained standard / GAU / FA models).

    models:  a released `.tar` checkpoint (third_party STG_NF)  or  a `.pt` written by edgelab.pose.stgnf_train
    result:  SegmentScores, the same object edgelab.pose.stgnf_eval.score_segments returns (normality per window + metadata),
             so every existing assembly function (variant_scores, delivered_variant_scores, fill_variant_scores) works on it.

The windows come from STG-NF's own test loader (third_party dataset.py), so they are exactly the ones the released
pipeline scores. Provenance (FA models): per window, 1 = detected pose, 0 = predicted pose, from the filler's sidecar files;
no sidecar folder = nothing was filled = all ones.
"""

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from edgelab.models.stgnf_flow import CouplingFlow, build_flow, count_parameters
from edgelab.pose.fill import SUFFIX, _scene_clip, merge_parts, provenance_path
from edgelab.pose.stgnf_eval import SEG_LENS, STGNF_ROOT, SegmentScores, _import_stgnf

_import_stgnf()
from models.STG_NF.model_pose import STG_NF  # noqa: E402
from models.STG_NF.modules_pose import ActNorm2d  # noqa: E402


class Standard(nn.Module):
    """third_party STG_NF with the recipe's hyper-parameters (K 8, L 1, R 3, hidden 0), called like our flows: forward(x, provenance)."""

    def __init__(self, n_frames: int = 24):
        super().__init__()
        self.model = STG_NF(pose_shape=(2, n_frames, 18), hidden_channels=0, K=8, L=1, R=3.0, actnorm_scale=1.0,
                            flow_permutation="permute", flow_coupling="affine", LU_decomposed=True, learn_top=False,
                            edge_importance=False, temporal_kernel_size=None, strategy="uniform", max_hops=8, device="cpu")

    def forward(self, x, provenance=None):
        return self.model(x, label=torch.ones(x.shape[0]), score=1)


def build_model(kind: str, n_frames: int) -> nn.Module:
    """kind 'standard' (third-party code) or one of stgnf_flow.build_flow's kinds."""
    return Standard(n_frames) if kind == "standard" else build_flow(kind, n_frames)


def uses_provenance(model: nn.Module) -> bool:
    return isinstance(model, CouplingFlow) and model.provenance


def mark_actnorm_initialised(model: nn.Module):
    for module in model.modules():
        if isinstance(module, ActNorm2d):
            module.inited = True


def load_model(path: str | Path, dataset: str = "ShanghaiTech") -> nn.Module:
    """A released `.tar` (standard STG-NF) or our `.pt` checkpoint -> model in eval mode."""
    path = Path(path)
    n_frames = SEG_LENS[dataset]
    if path.suffix == ".tar":
        model = Standard(n_frames)
        state = torch.load(path, map_location="cpu", weights_only=False)["state_dict"]
        model.model.load_state_dict(state, strict=False)
    else:
        saved = torch.load(path, map_location="cpu", weights_only=False)
        n_frames = saved["config"]["seg_len"]
        model = build_model(saved["kind"], n_frames)
        model.load_state_dict(saved["state_dict"])
    mark_actnorm_initialised(model)
    return model.eval()


def parameter_count(model: nn.Module) -> int:
    return count_parameters(model)


def segment_provenance(metadata: np.ndarray, pose_dir: Path, prov_dir: Path | None, dataset: str = "ShanghaiTech") -> np.ndarray:
    """(N, seg_len) float: 1 where the window's pose was detected, 0 where it was predicted. prov_dir None = all detected.

    Same lookup as fill.segment_predicted_shares: the window of a row is the seg_len poses of the person's track from start_frame on.
    """
    seg_len = SEG_LENS[dataset]
    out = np.ones((len(metadata), seg_len), dtype=np.float32)
    if prov_dir is None:
        return out
    index = {}
    for path in Path(pose_dir).glob(f"*{SUFFIX}"):
        index[_scene_clip(path.name, dataset)] = path
    tracks: dict[tuple, dict] = {}
    for row, (scene, clip, person, start) in enumerate(metadata):
        key = (str(int(scene)), str(clip) if dataset == "UBnormal" else str(int(clip)))
        if key not in tracks:
            path = index[key]
            clip_json = json.loads(path.read_text())
            side = json.loads(provenance_path(prov_dir, path).read_text())
            tracks[key] = {pid: (sorted(int(k) for k in merge_parts(t)), {int(f) for f in side.get(pid, {})})
                           for pid, t in clip_json.items()}
        frames, predicted = tracks[key][str(int(person))]
        first = frames.index(int(start))
        window = frames[first:first + seg_len]
        out[row, :len(window)] = [0.0 if f in predicted else 1.0 for f in window]
    return out


def score_segments_model(model: nn.Module, pose_dir: str | Path, dataset: str = "ShanghaiTech", prov_dir: Path | None = None,
                         batch_size: int = 256, threads: int | None = None) -> SegmentScores:
    """STG-NF's test pass (their loader, stride 1) with `model` instead of their trainer. normality = -nll."""
    from args import init_parser, init_sub_args
    from dataset import get_dataset_and_loader
    from utils.data_utils import trans_list

    if threads:
        torch.set_num_threads(threads)
    args = init_parser().parse_args(["--dataset", dataset, "--seg_len", str(SEG_LENS[dataset]), "--device", "cpu",
                                     "--num_workers", "0", "--batch_size", str(batch_size),
                                     "--data_dir", str(STGNF_ROOT / "data")])
    np.random.seed(0)
    args, _ = init_sub_args(args)
    args.pose_path["test"] = str(pose_dir) + "/"
    t0 = time.perf_counter()
    loaded, loader = get_dataset_and_loader(args, trans_list=trans_list, only_test=True)
    t1 = time.perf_counter()
    metadata = np.array(loaded["test"].metadata)
    provenance = segment_provenance(metadata, Path(pose_dir), prov_dir, dataset) if uses_provenance(model) else None
    normality, start = [], 0
    model.eval()
    with torch.no_grad():
        for batch in loader["test"]:
            x = batch[0][:, :2].float()
            prov = None if provenance is None else torch.from_numpy(provenance[start:start + len(x)])
            _, nll = model(x, prov)
            normality.append(-nll.reshape(-1))
            start += len(x)
    t2 = time.perf_counter()
    normality = torch.cat(normality).numpy()
    if dataset == "ShanghaiTech":
        metadata = metadata.astype(int)
    return SegmentScores(normality=np.atleast_1d(normality), metadata=metadata, seconds_model=t2 - t1,
                         seconds_load=t1 - t0, num_segments=len(metadata))
