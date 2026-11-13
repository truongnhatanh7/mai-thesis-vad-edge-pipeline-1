"""Run the frozen STG-NF checkpoint on a folder of *tracked_person.json files and build per-frame scores.

STG-NF code is imported from third_party/STG-NF (pinned commit, see EXP-001 log.md). Their scoring function is
re-implemented here (assemble_clip_scores) so we can move the score to a different frame of the segment for the
causal variant; tests/test_pose_stgnf_scoring.py checks it equals theirs for the default offset.
"""

import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

EXPERIMENTS_ROOT = Path(__file__).resolve().parents[3]
STGNF_ROOT = EXPERIMENTS_ROOT / "third_party" / "STG-NF"
SEG_LEN = 24  # ShanghaiTech
SEG_LENS = {"ShanghaiTech": SEG_LEN, "UBnormal": 16}  # STG-NF README: UBnormal is run with --seg_len 16


def score_frame_offsets(dataset: str = "ShanghaiTech") -> dict[str, int]:
    """Which frame of a segment gets its score: middle = paper (uses future frames), end = live (last frame)."""
    seg_len = SEG_LENS[dataset]
    return {"middle": seg_len // 2, "end": seg_len - 1}


def _import_stgnf():
    if str(STGNF_ROOT) not in sys.path:
        sys.path.insert(0, str(STGNF_ROOT))


@dataclass
class SegmentScores:
    normality: np.ndarray  # one value per segment, higher = more normal (= -NLL)
    metadata: np.ndarray  # rows [scene_id, clip_id, person_id, start_frame]
    seconds_model: float
    seconds_load: float
    num_segments: int


def score_segments(pose_dir: str | Path, checkpoint: str | Path, device: str = "cpu",
                   threads: int | None = None, batch_size: int = 256, dataset: str = "ShanghaiTech") -> SegmentScores:
    """STG-NF test pass, same args as `train_eval.py --dataset <dataset> --checkpoint ...`.

    metadata rows are [scene, clip, person, start_frame]: all ints for ShanghaiTech; for UBnormal the clip id is a
    string like 'abnormal__3_fog' (as in STG-NF), so the whole array is strings there.
    """
    _import_stgnf()
    from args import init_parser, init_sub_args
    from dataset import get_dataset_and_loader
    from models.STG_NF.model_pose import STG_NF
    from models.training import Trainer
    from utils.data_utils import trans_list
    from utils.optim_init import init_optimizer, init_scheduler
    from utils.train_utils import init_model_params

    if threads:
        torch.set_num_threads(threads)
    args = init_parser().parse_args(["--dataset", dataset, "--seg_len", str(SEG_LENS[dataset]),
                                     "--checkpoint", str(checkpoint),
                                     "--device", device, "--num_workers", "0", "--batch_size", str(batch_size),
                                     "--data_dir", str(STGNF_ROOT / "data")])
    np.random.seed(0)
    args, _ = init_sub_args(args)
    args.pose_path["test"] = str(pose_dir) + "/"

    t0 = time.perf_counter()
    loaded, loader = get_dataset_and_loader(args, trans_list=trans_list, only_test=True)
    t1 = time.perf_counter()
    model = STG_NF(**init_model_params(args, loaded))
    trainer = Trainer(args, model, None, loader["test"],
                      optimizer_f=init_optimizer(args.model_optimizer, lr=args.model_lr),
                      scheduler_f=init_scheduler(args.model_sched, lr=args.model_lr, epochs=args.epochs))
    trainer.load_checkpoint(str(checkpoint))
    normality = trainer.test()
    t2 = time.perf_counter()
    metadata = np.array(loaded["test"].metadata)
    if dataset == "ShanghaiTech":
        metadata = metadata.astype(int)
    return SegmentScores(normality=np.atleast_1d(normality), metadata=metadata,
                         seconds_model=t2 - t1, seconds_load=t1 - t0, num_segments=len(metadata))


def _clips(dataset: str, gt_dir: str | Path, clips: list[str] | None):
    """[(name, scene id, clip id, labels)] of the test clips; labels: 1 = anomalous frame (both datasets)."""
    if dataset == "ShanghaiTech":
        out = []
        for gt_file in sorted(Path(gt_dir).glob("*.npy")):
            scene_id, clip_id = gt_file.stem.split("_")
            out.append((gt_file.stem, str(int(scene_id)), str(int(clip_id)), np.load(gt_file).astype(int)))
        return out
    from edgelab.pose.ubnormal import load_gt, load_test_list, parse_video_name
    return [(video, *parse_video_name(video), load_gt(video, Path(gt_dir)))
            for video in (clips if clips is not None else load_test_list())]


def clip_normality(normality: np.ndarray, metadata: np.ndarray, gt_dir: str | Path,
                   frame_offset: int | None = None, dataset: str = "ShanghaiTech", clips: list[str] | None = None):
    """Per-frame normality per test clip, following STG-NF get_dataset_scores/get_clip_score.

    A segment's score lands on frame start + frame_offset (paper: middle frame = seg_len / 2; default). A frame's score is the
    least-normal person present; frames with nobody get +inf. Returns (clip_names, anomaly_labels, normality per clip).
    ShanghaiTech: clips = the *.npy files in gt_dir. UBnormal: clips = the video names (default: test_videos.txt),
    gt_dir holds `<video>_tracks.txt`.
    """
    if frame_offset is None:
        frame_offset = SEG_LENS[dataset] // 2
    # compare as strings so int metadata (ShanghaiTech) and STG-NF's string metadata (UBnormal) work the same way
    scene_col, clip_col = metadata[:, 0].astype(str), metadata[:, 1].astype(str)
    names, labels, normality_per_clip = [], [], []
    for name, scene_id, clip_id, anomaly_gt in _clips(dataset, gt_dir, clips):
        clip_rows = (scene_col == scene_id) & (clip_col == clip_id)
        person_ids = sorted(set(metadata[clip_rows, 2].astype(int)))
        per_person = np.full((max(1, len(person_ids)), len(anomaly_gt)), np.inf)
        for row, pid in enumerate(person_ids):
            rows = np.where(clip_rows & (metadata[:, 2].astype(int) == pid))[0]
            frames = metadata[rows, 3].astype(int) + frame_offset
            inside = frames < len(anomaly_gt)
            per_person[row, frames[inside]] = normality[rows][inside]
        names.append(name)
        labels.append(anomaly_gt)
        normality_per_clip.append(per_person.min(axis=0))
    return names, labels, normality_per_clip


def assemble_clip_scores(normality: np.ndarray, metadata: np.ndarray, gt_dir: str | Path,
                         frame_offset: int | None = None, dataset: str = "ShanghaiTech",
                         clips: list[str] | None = None):
    """Like clip_normality, but frames with nobody get the dataset's most-normal score (STG-NF's rule).

    Returns (clip_names, anomaly_labels, anomaly_scores): label 1 = anomalous, score = -normality.
    """
    names, labels, normality_per_clip = clip_normality(normality, metadata, gt_dir, frame_offset, dataset, clips)
    flat = np.concatenate(normality_per_clip)
    finite_max, finite_min = flat[flat != np.inf].max(), flat[flat != -np.inf].min()
    scores = []
    for clip in normality_per_clip:
        clip = clip.copy()
        clip[clip == np.inf] = finite_max
        clip[clip == -np.inf] = finite_min
        scores.append(-clip)
    return names, labels, scores
