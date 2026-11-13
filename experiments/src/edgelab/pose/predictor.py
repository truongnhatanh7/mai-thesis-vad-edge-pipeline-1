"""EXP-018 C2: a tiny causal GRU that predicts the next pose of a track, and filling of dropped frames with it.

A pose is stored as (c, s, shape): c = mean of the 17 keypoints (x, y), s = height of their bounding box (>= 1 px),
shape = (keypoints - c) / s, 34 numbers. At every frame the GRU reads

    [shape (34), (c - c_prev) / s_prev (2), log(s / s_prev) (1), 1 if this pose was detected else 0]

and outputs the change to the next frame: [delta shape (34), delta c / s (2), delta log s (1)]. The pose it reads is the
detection when there is one, otherwise its own previous prediction (so it can roll forward through a gap).

Rules shared with the constant-velocity filler (edgelab.pose.fill.fill_P), so the two are interchangeable:
 - a frame is filled only if it has no detection, there was a detection before it, and at most `max_gap` (G = 8) frames
   have passed since that detection; filling stops at the next detection;
 - keypoint confidences and the detection score of a filled pose = those of the last detection * 0.8 ** age.
After a gap longer than G the GRU state is reset at the next detection (nothing it predicted was used anyway).

Causality: the pose at frame t depends on detections at frames <= t only (tests/test_pose_predictor.py perturbs the future).
Everything is batched: poses (B, L, 17, 2), detected (B, L) bool, one grid column per frame.
"""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from edgelab.pose.fill import DAMPING, SUFFIX, key_format, merge_parts, provenance_path

FILL_G = 8
HIDDEN = 48
FEATURES = 38
MIN_HEIGHT = 1.0


def pose_state(xy: torch.Tensor):
    """(..., 17, 2) -> (c (..., 2), s (..., 1), shape (..., 34))."""
    c = xy.mean(dim=-2)
    s = (xy[..., 1].amax(-1) - xy[..., 1].amin(-1)).clamp(min=MIN_HEIGHT)[..., None]
    shape = ((xy - c[..., None, :]) / s[..., None]).flatten(-2)
    return c, s, shape


def pose_from_state(c: torch.Tensor, s: torch.Tensor, shape: torch.Tensor) -> torch.Tensor:
    return shape.unflatten(-1, (17, 2)) * s[..., None] + c[..., None, :]


class PosePredictor(nn.Module):
    """GRU cell + linear head, ~14 K parameters. The head starts at zero, so an untrained predictor holds the pose still."""

    def __init__(self, hidden: int = HIDDEN):
        super().__init__()
        self.hidden = hidden
        self.cell = nn.GRUCell(FEATURES, hidden)
        self.head = nn.Linear(hidden, 37)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def step(self, h, c, s, shape, c_prev, s_prev, detected):
        """Read the pose at this frame -> (new h, predicted next (c, s, shape))."""
        features = torch.cat([shape, (c - c_prev) / s_prev, torch.log(s / s_prev), detected[:, None].to(shape.dtype)], dim=1)
        h = self.cell(features, h)
        out = self.head(h)
        next_shape = shape + out[:, :34]
        next_c = c + out[:, 34:36] * s
        next_s = (s * torch.exp(0.3 * torch.tanh(out[:, 36:37]))).clamp(min=MIN_HEIGHT)
        return h, next_c, next_s, next_shape


def rollout(predictor: PosePredictor, xy: torch.Tensor, detected: torch.Tensor, max_gap: int = FILL_G):
    """-> (pose (B, L, 17, 2): the detection where detected, else the prediction; filled (B, L) bool).

    `filled` marks the frames the filler would emit (no detection, a detection before, age <= max_gap); at other
    undetected frames `pose` holds the GRU's internal guess and must not be used.
    """
    b, length = detected.shape
    device = xy.device
    h = torch.zeros(b, predictor.hidden, device=device)
    started = torch.zeros(b, dtype=torch.bool, device=device)
    age = torch.zeros(b, dtype=torch.long, device=device)  # frames since the last detection (0 = detected now)
    c_prev, s_prev = torch.zeros(b, 2, device=device), torch.ones(b, 1, device=device)
    guess = (torch.zeros(b, 2, device=device), torch.ones(b, 1, device=device), torch.zeros(b, 34, device=device))
    poses, filled = [], []
    for t in range(length):
        det = detected[:, t]
        c_obs, s_obs, shape_obs = pose_state(xy[:, t])
        sel = lambda a, g: torch.where(det[:, None], a, g)  # noqa: E731  detection if there is one, else our own guess
        c, s, shape = sel(c_obs, guess[0]), sel(s_obs, guess[1]), sel(shape_obs, guess[2])
        new_age = torch.where(det, torch.zeros_like(age), age + 1)
        reset = det & (~started | (age > max_gap))  # first detection, or the gap before it was longer than max_gap frames
        h = torch.where(reset[:, None], torch.zeros_like(h), h)
        c_prev = torch.where(reset[:, None], c, c_prev)
        s_prev = torch.where(reset[:, None], s, s_prev)
        started = started | det
        age = new_age
        filled.append(~det & started & (age <= max_gap))
        poses.append(pose_from_state(c, s, shape))
        h, nc, ns, nshape = predictor.step(h, c, s, shape, c_prev, s_prev, det)
        guess = (nc, ns, nshape)
        c_prev, s_prev = c, s
    return torch.stack(poses, dim=1), torch.stack(filled, dim=1)


def fill_const_velocity(xy: torch.Tensor, detected: torch.Tensor, max_gap: int = FILL_G, damping: float = DAMPING):
    """Batched fill_P (edgelab.pose.fill): damped constant velocity from the last two detections. -> (pose, filled (B, L)).

    Step k after a detection moves the pose by velocity * damping ** k (velocity = per-frame change between the last two
    detections); tests/test_pose_predictor.py checks it against fill_P on random tracks.
    """
    b, length = detected.shape
    position = torch.arange(length, device=xy.device)[None, :].expand(b, length)
    last = torch.cummax(torch.where(detected, position, torch.full_like(position, -1)), dim=1).values  # latest detection <= t
    last_before = torch.cat([torch.full_like(last[:, :1], -1), last[:, :-1]], dim=1)  # latest detection < t
    previous = torch.gather(last_before, 1, last.clamp(min=0))  # latest detection before that detection
    previous = torch.where(last >= 0, previous, torch.full_like(previous, -1))
    gather = lambda index: torch.gather(xy, 1, index.clamp(min=0)[..., None, None].expand(-1, -1, 17, 2))  # noqa: E731
    at_last, at_previous = gather(last), gather(previous)
    gap = (last - previous).clamp(min=1)[..., None, None].to(xy.dtype)
    velocity = torch.where((previous >= 0)[..., None, None], (at_last - at_previous) / gap, torch.zeros_like(at_last))
    age = position - last
    reach = (damping * (1 - damping ** age.to(xy.dtype)) / (1 - damping))[..., None, None]  # sum_{k=1..age} damping ** k
    predicted = at_last + velocity * reach
    filled = ~detected & (last >= 0) & (age <= max_gap)
    return torch.where(detected[..., None, None], xy, predicted), filled


def fill_batch(mode: str, predictor: PosePredictor | None, xy: torch.Tensor, detected: torch.Tensor, max_gap: int = FILL_G):
    """mode 'cv' (constant velocity) or 'gru'. -> (pose, filled)."""
    if mode == "cv":
        return fill_const_velocity(xy, detected, max_gap)
    if mode == "gru":
        with torch.no_grad():
            return rollout(predictor, xy, detected, max_gap)
    raise ValueError(f"mode must be 'cv' or 'gru', got {mode!r}")


# ------------------------------------------------------------------ JSON level (evaluation)

def fill_clip_gru(clip: dict, predictor: PosePredictor, max_gap: int = FILL_G, last_frame: int | None = None):
    """Fill every track of one video's pose JSON with the GRU. -> (filled clip, {person id: {frame key: age}}), like fill_clip."""
    tracks = {pid: merge_parts(track) for pid, track in clip.items() if len(track)}
    if not tracks:  # nobody was detected in this video (every frame dropped)
        return {}, {}
    spans = {}
    for pid, track in tracks.items():
        frames = np.array(sorted(int(k) for k in track))
        spans[pid] = (frames, int(frames[0]), int(frames[-1]) + max_gap)
    persons = list(tracks)
    length = max(span[2] - span[1] + 1 for span in spans.values()) if persons else 0
    xy = np.zeros((len(persons), length, 17, 2), dtype=np.float32)
    detected = np.zeros((len(persons), length), dtype=bool)
    for row, pid in enumerate(persons):
        key = key_format(tracks[pid])
        frames, first, _ = spans[pid]
        for frame in frames:
            xy[row, frame - first] = np.asarray(tracks[pid][key(frame)]["keypoints"]).reshape(-1, 3)[:, :2]
            detected[row, frame - first] = True
    pose, filled = fill_batch("gru", predictor, torch.from_numpy(xy), torch.from_numpy(detected), max_gap)
    pose, filled = pose.numpy(), filled.numpy()
    out, provenance = {}, {}
    for row, pid in enumerate(persons):
        track, key = tracks[pid], key_format(tracks[pid])
        frames, first, last_slot = spans[pid]
        new_track, ages = dict(track), {}
        last_detection = None
        for offset in range(last_slot - first + 1):
            frame = first + offset
            if detected[row, offset]:
                last_detection = frame
            elif filled[row, offset] and (last_frame is None or frame <= last_frame):
                base = track[key(last_detection)]
                age = frame - last_detection
                keypoints = np.asarray(base["keypoints"], dtype=float).reshape(-1, 3).copy()
                keypoints[:, :2] = pose[row, offset]
                keypoints[:, 2] *= DAMPING ** age
                new_track[key(frame)] = {"keypoints": [float(v) for v in keypoints.reshape(-1)],
                                         "scores": float(base["scores"]) * DAMPING ** age}
                ages[key(frame)] = age
        out[pid] = dict(sorted(new_track.items()))
        if ages:
            provenance[pid] = ages
    return out, provenance


def fill_dir_gru(in_dir: Path, out_dir: Path, predictor: PosePredictor, max_gap: int = FILL_G, prov_dir: Path | None = None,
                 n_frames: dict[str, int] | None = None) -> dict:
    """Like edgelab.pose.fill.fill_dir (mode P) but with the GRU. Same file names, same sidecars."""
    from edgelab.pose.fill import video_of_file

    out_dir, in_dir = Path(out_dir), Path(in_dir)
    prov_dir = Path(prov_dir) if prov_dir else out_dir.with_name(out_dir.name + "_prov")
    out_dir.mkdir(parents=True, exist_ok=True)
    prov_dir.mkdir(parents=True, exist_ok=True)
    detected = predicted = 0
    for path in sorted(in_dir.glob(f"*{SUFFIX}")):
        last_frame = None
        if n_frames is not None:
            video = video_of_file(path.name, n_frames)
            if video is None:
                raise ValueError(f"no frame count for pose file {path.name}")
            last_frame = n_frames[video] - 1
        clip = json.loads(path.read_text())
        filled, provenance = fill_clip_gru(clip, predictor, max_gap, last_frame)
        (out_dir / path.name).write_text(json.dumps(filled))
        provenance_path(prov_dir, path).write_text(json.dumps(provenance))
        predicted += sum(len(p) for p in provenance.values())
        detected += sum(len(merge_parts(t)) for t in clip.values())
    stats = {"mode": "GRU", "max_gap": max_gap, "detected_poses": detected, "predicted_poses": predicted}
    (prov_dir / "fill_stats.json").write_text(json.dumps(stats))
    return stats


def load_predictor(path: Path) -> PosePredictor:
    state = torch.load(path, map_location="cpu", weights_only=False)
    model = PosePredictor(state.get("hidden", HIDDEN))
    model.load_state_dict(state["state_dict"])
    return model.eval()
