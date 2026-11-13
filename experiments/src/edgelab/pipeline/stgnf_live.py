"""Online STG-NF: score each newly completed 24-pose segment of each track as frames arrive (EXP-004 plan §3.1).

Same model, checkpoint, keypoint conversion and normalisation as the offline test pass (stgnf_eval.score_segments);
tests/test_pipeline_live_timing.py checks the scores equal the offline ones. Differences from offline: a segment is
scored as soon as its 24th pose arrives, and every complete window is scored (the offline loader skips the last one).
"""

from collections import deque

import numpy as np
import torch

from edgelab.pose.evaluate import CHECKPOINT
from edgelab.pose.stgnf_eval import SEG_LEN, _import_stgnf

MAX_MISSING = 2  # STG-NF accepts a segment if at most 2 of its 24 frame slots are missing


def load_model(checkpoint=CHECKPOINT):
    """The frozen STG-NF flow, built with the same arguments as score_segments (ShanghaiTech, CPU)."""
    _import_stgnf()
    from args import init_parser, init_sub_args
    from models.STG_NF.model_pose import STG_NF
    from utils.train_utils import init_model_params

    args = init_parser().parse_args(["--dataset", "ShanghaiTech", "--device", "cpu"])
    args, _ = init_sub_args(args)
    fake_dataset = {"test": [[np.zeros((3, SEG_LEN, 18))]]}  # init_model_params only reads the sample shape
    model = STG_NF(**init_model_params(args, fake_dataset))
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)["state_dict"]
    model.load_state_dict(state, strict=False)  # same call as Trainer.load_checkpoint
    model.set_actnorm_init()
    return model.eval()


def is_continuous(frames: list[int]) -> bool:
    """STG-NF's rule: at most MAX_MISSING of the 24 slots from the first frame on are empty."""
    return len(set(frames) & set(range(frames[0], frames[0] + SEG_LEN))) >= SEG_LEN - MAX_MISSING


def score_segments(model, segments: np.ndarray) -> np.ndarray:
    """segments (n, 24, 17, 3) pixels -> normality (n,), higher = more normal (= -NLL)."""
    _import_stgnf()
    from dataset import keypoints17_to_coco18
    from utils.data_utils import normalize_pose

    x = normalize_pose(keypoints17_to_coco18(segments).astype(np.float32))  # (n, 24, 18, 3)
    x = torch.from_numpy(np.ascontiguousarray(x.transpose(0, 3, 1, 2)[:, :2])).float()  # x, y channels only
    with torch.no_grad():
        _, nll = model(x, label=torch.ones(len(x)))
    return -nll.numpy().reshape(-1)


class LiveStgnf:
    """Keeps the last 24 poses of every track; update() scores the segments completed by the current frame."""

    def __init__(self, model):
        self.model = model
        self.buffers: dict[int, deque] = {}

    def update(self, frame: int, poses: dict[int, np.ndarray]) -> tuple[list[int], np.ndarray]:
        """poses: track id -> (17, 3) keypoints seen on this frame. -> (track ids with a new segment, their normality)."""
        ready, ids = [], []
        for track_id, keypoints in poses.items():
            buffer = self.buffers.setdefault(track_id, deque(maxlen=SEG_LEN))
            buffer.append((frame, keypoints))
            if len(buffer) == SEG_LEN and is_continuous([f for f, _ in buffer]):
                ready.append(np.stack([k for _, k in buffer]))
                ids.append(track_id)
        if not ready:
            return [], np.zeros(0)
        return ids, score_segments(self.model, np.stack(ready))
