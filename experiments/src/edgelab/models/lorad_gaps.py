"""Q2-P1-E08: four fixes for the limitations of LoRaD (causal STG-NF) and their joint model MS-IAC-RevFlow.

Baselines (E06, not redefined here)   B0 = A1 (symmetric STG-NF, retrained)   B1 = A2 (LoRaD: causal flow, per-window normalisation N0, window mode)

    P_A    MS-CTFlow     Gap 1  every coupling block's single causal temporal conv -> three parallel causal DILATED convs d in {1, 2, 4}
                                (left padding d (k - 1) only, so frame t reads frames <= t), 1x1 fusion + residual (`MSCausalSTGCN`)
    P_B    RevMargin     Gap 3  same network as B1, trained with  NLL(X) + l1 |X_y - f^-1(f(X))_y|^2 + l2 max(0, m - (NLL(X_pert) - NLL(X)))
                                (`hybrid_loss`; zero parameters and zero inference cost more than B1)
    P_C    IAC-Flow      Gap 2  18 body joints + 2 virtual context nodes (V = 20): node 19 = box kinematics, node 20 = nearest-neighbour
                                (social) proximity. Both are computed from the RAW track BEFORE the per-window mean subtraction of N0
                                erases the translation (`clip_context`, `pack_nodes`, `make_adjacency_ctx`)
    P_D    ConfSmooth    Gap 4  causal confidence-weighted smoothing of low-confidence joints before windowing (`conf_smooth`)
    P_ABC  MS-IAC-RevFlow       P_A + P_B + P_C + P_D

Every temporal operation is causal (frame t depends on frames <= t): the graph convs act on one frame, the temporal convs are left-padded,
the context features use only the current and previous pose of the track and the poses of the other tracks at the SAME frame, the
smoothing is a first-order recursion. tests/test_q2_p1_e08.py checks each of these (prefix invariance) and the online scorer against the batch one.

Notes on the design decisions the plan leaves open (all fixed in Q2-P1-E08_lorad_gaps_proposals/plan.md before any number was seen):
* The flow channels are (x, y), so a graph node carries exactly two numbers. Node 19 carries the bounding-box velocity (dcx / h, dcy / h)
  x VEL_GAIN; the box aspect w / h is computed (and tested) but not fed to the flow: the box is derived from the keypoints, so w / h is already
  readable from the window-normalised skeleton. Node 20 carries prox * (dx, dy) / |(dx, dy)|, i.e. the direction to the nearest neighbour times
  the proximity 1 / (1 + d / h): (direction, proximity) is a bijection onto the plan's (dx_nn / W, dy_nn / H, proximity).
* STG-NF's `uniform` graph with max_hop = 8 is dense (every pair of joints is within 8 hops), so the 20 x 20 adjacency is dense too; the hub
  edges (nodes 19, 20 -> neck, both hips) are in the edge list and the hop distances are tested, but they do not sparsify A.
* f^-1(f(X)) = X for an exactly invertible flow, so the reverse term of `hybrid_loss` is zero up to float round-off. It is implemented as
  specified and its magnitude is logged every epoch (config.json) and reported.
"""

from __future__ import annotations

import json
import math
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from edgelab.models.lorad import STRIDE, mark_actnorm, starts_of
from edgelab.models.stgnf_flow import CouplingFlow, count_parameters, make_adjacency
from edgelab.pose.augment import VID_RES, to_flow_input
from edgelab.pose.segments import read_tracks, window_starts
from edgelab.pose.stgnf_eval import _import_stgnf

_import_stgnf()
from models.STG_NF.graph import Graph, get_hop_distance, normalize_digraph  # noqa: E402
from models.STG_NF.stgcn import ConvTemporalGraphical  # noqa: E402

DILATIONS = (1, 2, 4)
CONF_THR = 0.25  # P_D: joints below this confidence are smoothed
VEL_GAIN = 10.0  # node 19: box velocity in box-heights per frame x gain (walking ~0.3, cycling ~2)
VEL_CLIP = 0.5  # box heights per frame; larger jumps are ID switches / detector glitches
H_MIN = 16.0  # px: floor of the box height (tiny boxes make h-normalised velocities explode)
BOX_CONF = 0.05  # joints below this confidence do not enter the keypoint box
W_VID, H_VID = VID_RES
N_BODY, N_NODES = 18, 20
HUB = (1, 8, 11)  # neck, right hip, left hip in the OpenPose-18 order STG-NF uses
HP = {"lambda1": 1.0, "lambda2": 0.1, "margin": 0.5, "jitter": (0.05, 0.25), "speed_slow": (0.2, 0.5), "speed_fast": (1.8, 3.0)}

ARMS = {  # trained arms
    "P_A": dict(ms=True, ctx=False, smooth=False, rev=False),
    "P_B": dict(ms=False, ctx=False, smooth=False, rev=True),
    "P_C": dict(ms=False, ctx=True, smooth=False, rev=False),
    "P_ABC": dict(ms=True, ctx=True, smooth=True, rev=True),
}
EVAL_ONLY = {"P_D": dict(ms=False, ctx=False, smooth=True, rev=False, weights="B1")}  # smoothing in front of the B1 (LoRaD) weights


# =====================================================================================================================
# Graph with the two virtual context nodes
# =====================================================================================================================
def make_adjacency_ctx(n_extra: int = 2, max_hop: int = 8) -> torch.Tensor:
    """(1, 18 + n_extra, 18 + n_extra) STG-NF `uniform` adjacency of the OpenPose-18 skeleton + virtual nodes hung on neck / both hips."""
    g = Graph(layout="openpose", strategy="uniform", max_hop=max_hop)
    edges = list(g.edge)
    v = N_BODY + n_extra
    for node in range(N_BODY, v):
        edges += [(node, node), *[(node, hub) for hub in HUB]]
    hop = get_hop_distance(v, edges, max_hop=max_hop)
    adj = np.zeros((v, v))
    for h in range(max_hop + 1):
        adj[hop == h] = 1
    return torch.from_numpy(normalize_digraph(adj)[None]).float()


def hop_distances(n_extra: int = 2, max_hop: int = 8) -> np.ndarray:
    g = Graph(layout="openpose", strategy="uniform", max_hop=max_hop)
    edges = list(g.edge)
    v = N_BODY + n_extra
    for node in range(N_BODY, v):
        edges += [(node, node), *[(node, hub) for hub in HUB]]
    return get_hop_distance(v, edges, max_hop=max_hop)


# =====================================================================================================================
# Proposal A: multi-scale causal dilated temporal block
# =====================================================================================================================
class MSCausalSTGCN(nn.Module):
    """Causal ST-GCN block with parallel causal dilated temporal convs (default d = 1, 2, 4), fused by a 1x1 projection with a residual.

    graph conv (1 frame) -> BN -> ReLU = h;  b_d = conv_d(pad_left(h, d (k - 1)));  out = ReLU(BN(proj([b_1, b_2, b_4]) + h) + block residual).
    The left padding of branch d is exactly its receptive field minus one, so no branch ever reads a future frame.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple[int, int], residual: bool = True,
                 dilations: tuple[int, ...] = DILATIONS):
        super().__init__()
        self.k, self.dilations = int(kernel_size[0]), tuple(dilations)
        self.gcn = ConvTemporalGraphical(in_channels, out_channels, kernel_size[1])
        self.bn1, self.relu1 = nn.BatchNorm2d(out_channels), nn.ReLU(inplace=False)
        self.branches = nn.ModuleList(nn.Conv2d(out_channels, out_channels, (self.k, 1), dilation=(d, 1), padding=0) for d in self.dilations)
        self.proj = nn.Conv2d(len(self.dilations) * out_channels, out_channels, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if not residual:
            self.residual = lambda x: 0
        elif in_channels == out_channels:
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1), nn.BatchNorm2d(out_channels))

    def forward(self, x: torch.Tensor, A: torch.Tensor):
        res = self.residual(x)
        h, _ = self.gcn(x, A)
        h = self.relu1(self.bn1(h))
        outs = [conv(F.pad(h, (0, 0, d * (self.k - 1), 0))) for d, conv in zip(self.dilations, self.branches)]
        fused = self.proj(torch.cat(outs, dim=1)) + h
        return F.relu(self.bn2(fused) + res), A


class MSGraphSubnet(nn.Module):
    """Coupling network of P_A: (N, Cin, T, V) -> (N, 2, T, V), frame t of the output sees only frames <= t."""

    def __init__(self, in_channels: int, n_frames: int, first: bool, adjacency: torch.Tensor, dilations: tuple[int, ...] = DILATIONS):
        super().__init__()
        self.register_buffer("A", adjacency.clone())
        self.gcn = MSCausalSTGCN(in_channels, 2, (n_frames // 2 + 1, self.A.size(0)), residual=not first, dilations=dilations)

    def forward(self, x: torch.Tensor, valid: torch.Tensor | None = None) -> torch.Tensor:
        out, _ = self.gcn(x, self.A)
        return out


def build_gap_flow(arm: str, seg_len: int) -> CouplingFlow:
    """B-style causal flow (CausalGraphSubnet, kernel L // 2 + 1) with the arm's changes: multi-scale block and / or the 20-node graph."""
    cfg = {**ARMS, **EVAL_ONLY}[arm]
    flow = CouplingFlow(subnet="causal", n_frames=seg_len)
    adjacency = make_adjacency_ctx() if cfg["ctx"] else None
    for k, step in enumerate(flow.steps):
        if cfg["ms"]:
            step.subnet = MSGraphSubnet(1, seg_len, first=(k == 0), adjacency=adjacency if adjacency is not None else make_adjacency())
        elif adjacency is not None:
            step.subnet.A = adjacency.clone()
    return flow


# =====================================================================================================================
# Proposal D: causal confidence-weighted smoothing
# =====================================================================================================================
def conf_smooth(kp: np.ndarray, thr: float = CONF_THR) -> np.ndarray:
    """(n, 17, 3) keypoints of one track -> same shape, xy smoothed: p^_t = w p_t + (1 - w) p^_{t-1} with w = min(1, c_t / thr) on the
    joints with c_t < thr (w = 1 elsewhere, i.e. the pose is untouched). p^_0 = p_0. First-order causal recursion, O(1) state (17 x 2)."""
    kp = np.asarray(kp, dtype=np.float64)
    out = kp.copy()
    if len(kp) == 0:
        return out
    prev = kp[0, :, :2]
    for t in range(1, len(kp)):
        w = np.minimum(1.0, kp[t, :, 2] / thr)[:, None]
        prev = w * kp[t, :, :2] + (1.0 - w) * prev
        out[t, :, :2] = prev
    return out


class ConfSmoother:
    """The streaming form of `conf_smooth`."""

    def __init__(self, thr: float = CONF_THR):
        self.thr, self.prev = thr, None

    def update(self, kp17x3: np.ndarray) -> np.ndarray:
        kp = np.asarray(kp17x3, dtype=np.float64)
        out = kp.copy()
        if self.prev is None:
            self.prev = kp[:, :2]
        else:
            w = np.minimum(1.0, kp[:, 2] / self.thr)[:, None]
            self.prev = w * kp[:, :2] + (1.0 - w) * self.prev
            out[:, :2] = self.prev
        return out


# =====================================================================================================================
# Proposal C: box kinematics + social proximity
# =====================================================================================================================
def keypoint_box(kp: np.ndarray) -> np.ndarray:
    """(..., 17, 3) -> (..., 4) [cx, cy, w, h] of the box of the joints with confidence >= BOX_CONF (all joints when none passes)."""
    xy, c = kp[..., :2], kp[..., 2]
    ok = c >= BOX_CONF
    ok = np.where(ok.any(axis=-1, keepdims=True), ok, True)
    lo = np.where(ok[..., None], xy, np.inf).min(axis=-2)
    hi = np.where(ok[..., None], xy, -np.inf).max(axis=-2)
    return np.concatenate([(lo + hi) / 2, hi - lo], axis=-1)


def box_kinematics(frames: np.ndarray, kp: np.ndarray) -> np.ndarray:
    """One track -> (n, 3) [vx, vy, aspect] : box-centre velocity in box heights per frame (first pose 0, clipped to +-VEL_CLIP,
    divided by the frame gap, capped at 5) and w / h of the keypoint box. Pose t uses poses t and t - 1 only."""
    box = keypoint_box(np.asarray(kp, dtype=np.float64))
    h = np.maximum(box[:, 3], H_MIN)
    out = np.zeros((len(frames), 3))
    if len(frames) > 1:
        gap = np.clip(np.diff(frames), 1, 5)[:, None]
        out[1:, :2] = np.clip((box[1:, :2] - box[:-1, :2]) / (gap * h[1:, None]), -VEL_CLIP, VEL_CLIP)
    out[:, 2] = box[:, 2] / h
    return out


def neighbour_features(centre: np.ndarray, height: np.ndarray) -> np.ndarray:
    """Poses of ONE frame: centre (m, 2) px, height (m,) px (box heights floored at H_MIN) -> (m, 3) [dx / W, dy / H, prox] to the nearest
    other pose, prox = 1 / (1 + d / h); a lone person gets zeros."""
    m = len(centre)
    out = np.zeros((m, 3))
    if m < 2:
        return out
    d = np.linalg.norm(centre[:, None] - centre[None], axis=-1)
    np.fill_diagonal(d, np.inf)
    j = d.argmin(axis=1)
    rel = centre[j] - centre
    out[:, 0], out[:, 1] = rel[:, 0] / W_VID, rel[:, 1] / H_VID
    out[:, 2] = 1.0 / (1.0 + d[np.arange(m), j] / height)
    return out


def clip_context(tracks: dict) -> dict:
    """{pid: (frames, kp (n, 17, 3))} of one clip -> {pid: (n, 6) [vx, vy, aspect, dx_nn / W, dy_nn / H, prox]} (see module docstring)."""
    ctx, rows = {}, []
    for pid, (frames, kp) in tracks.items():
        ctx[pid] = np.zeros((len(frames), 6))
        if len(frames) == 0:
            continue
        ctx[pid][:, :3] = box_kinematics(frames, kp)
        box = keypoint_box(np.asarray(kp, dtype=np.float64))
        rows.append((pid, np.arange(len(frames)), np.asarray(frames), box[:, :2], np.maximum(box[:, 3], H_MIN)))
    if not rows:
        return ctx
    pid_all = np.concatenate([np.full(len(r[1]), r[0]) for r in rows])
    idx_all = np.concatenate([r[1] for r in rows])
    fr_all = np.concatenate([r[2] for r in rows])
    cen_all = np.concatenate([r[3] for r in rows])
    h_all = np.concatenate([r[4] for r in rows])
    order = np.argsort(fr_all, kind="stable")
    fr_s = fr_all[order]
    cuts = np.flatnonzero(np.diff(fr_s)) + 1
    for grp in np.split(order, cuts):
        feat = neighbour_features(cen_all[grp], h_all[grp])
        for g, f in zip(grp, feat):
            ctx[int(pid_all[g])][idx_all[g], 3:] = f
    return ctx


def augment_clip(tracks: dict) -> dict:
    """{pid: (frames, kp (n, 17, 3))} -> {pid: (frames, kp_aug (n, 19, 3))}: rows 17 = [vx, vy, aspect], 18 = [dx_nn / W, dy_nn / H, prox]
    (so a track stays one hashable array for the shared window-table engine)."""
    ctx = clip_context(tracks)
    out = {}
    for pid, (frames, kp) in tracks.items():
        kp = np.asarray(kp, dtype=np.float64)
        out[pid] = (frames, np.concatenate([kp, ctx[pid].reshape(-1, 2, 3)], axis=1))
    return out


def pack_nodes(ctx_rows: np.ndarray) -> np.ndarray:
    """(..., 2, 3) context rows -> (..., 2 nodes, 2 channels): node 19 = VEL_GAIN (vx, vy); node 20 = prox * unit(dx, dy) (0 when solo)."""
    ctx_rows = np.asarray(ctx_rows, dtype=np.float64)
    v = ctx_rows[..., 0, :2] * VEL_GAIN
    dx, dy, prox = ctx_rows[..., 1, 0] * W_VID, ctx_rows[..., 1, 1] * H_VID, ctx_rows[..., 1, 2]
    norm = np.sqrt(dx ** 2 + dy ** 2)
    unit = np.where(norm[..., None] > 0, np.stack([dx, dy], axis=-1) / np.maximum(norm[..., None], 1e-9), 0.0)
    return np.stack([v, prox[..., None] * unit], axis=-2)


def prepare_tracks(tracks_by_clip: list[dict], smooth: bool, ctx: bool) -> list[dict]:
    """The pipeline's tracks -> what the arm's scorer reads: smoothed keypoints (P_D) and / or the 19-row context-augmented array (P_C).
    Context is computed from the smoothed poses when both are on."""
    out = []
    for tracks in tracks_by_clip:
        cur = {pid: (f, conf_smooth(kp) if smooth else np.asarray(kp, dtype=np.float64)) for pid, (f, kp) in tracks.items()}
        out.append(augment_clip(cur) if ctx else cur)
    return out


# =====================================================================================================================
# Flow input and scorer
# =====================================================================================================================
def to_input(xy_win: np.ndarray, nodes: np.ndarray | None = None, flip: np.ndarray | None = None) -> torch.Tensor:
    """xy_win (B, T, 17, 2) px, nodes (B, T, 2, 2) packed context | None -> (B, 2, T, 18 or 20). N0 normalisation on the body only;
    `flip` mirrors x (the body about the window mean, the nodes' x channel sign)."""
    xy = torch.from_numpy(np.ascontiguousarray(xy_win, dtype=np.float32))
    fl = None if flip is None else torch.from_numpy(np.asarray(flip, dtype=bool))
    body = to_flow_input(xy, fl)
    if nodes is None:
        return body
    nd = torch.from_numpy(np.ascontiguousarray(nodes.transpose(0, 3, 1, 2), dtype=np.float32))  # (B, 2, T, 2 nodes)
    if fl is not None and fl.any():
        nd[fl, 0] = -nd[fl, 0]
    return torch.cat([body, nd], dim=3)


class GapFlowScorer:
    """normality = -NLL of every complete window, window mode, N0 normalisation. Tracks are what `prepare_tracks(.., smooth, ctx)` returns."""

    name = "gap-flow"

    def __init__(self, flow: CouplingFlow, seg_len: int, ctx: bool, chunk: int = 4096):
        self.flow, self.seg_len, self.ctx, self.chunk = flow.eval(), seg_len, ctx, chunk

    def _nll(self, x: torch.Tensor) -> np.ndarray:
        out = []
        with torch.no_grad():
            for i in range(0, len(x), self.chunk):
                out.append(self.flow(x[i:i + self.chunk])[1].numpy().astype(np.float64))
        return np.concatenate(out) if out else np.zeros(0)

    def score_tracks(self, tracks: list[tuple[np.ndarray, np.ndarray]]) -> list[np.ndarray]:
        L, wins, nodes, counts = self.seg_len, [], [], []
        for frames, kp in tracks:
            s = starts_of(frames, L)
            counts.append(len(s))
            if len(s) == 0:
                continue
            idx = s[:, None] + np.arange(L)[None, :]
            wins.append(kp[:, :17, :2][idx])
            if self.ctx:
                nodes.append(pack_nodes(kp[:, 17:, :])[idx])
        if not wins:
            return [np.zeros(0) for _ in tracks]
        nll = self._nll(to_input(np.concatenate(wins), np.concatenate(nodes) if self.ctx else None))
        out, at = [], 0
        for c in counts:
            out.append(-nll[at:at + c])
            at += c
        return out


def load_gap_flow(path: Path) -> tuple[CouplingFlow, dict]:
    saved = torch.load(path, map_location="cpu", weights_only=False)
    cfg = saved["config"]
    flow = build_gap_flow(saved["arm_key"], cfg["seg_len"])
    flow.load_state_dict(saved["state_dict"])
    mark_actnorm(flow)
    return flow.eval(), cfg


# =====================================================================================================================
# Online scorer (Raspberry Pi 5): all tracks of a frame in one flow call
# =====================================================================================================================
class OnlineGapScorer:
    """`step(frame_id, {track_id: keypoints 17x3})` -> normality of every window completed at this frame (window mode, N0).

    Per track: ring buffers (L poses, L frame ids, L packed context rows), the smoother state and the previous box. The windows of all
    tracks that complete on this frame are scored with ONE flow call. Same maths as `GapFlowScorer` on `prepare_tracks` output (tested)."""

    def __init__(self, flow: CouplingFlow, seg_len: int, ctx: bool = False, smooth: bool = False, name: str = "gap-flow"):
        self.flow, self.seg_len, self.ctx, self.smooth, self.name = flow.eval(), seg_len, ctx, smooth, name
        self.tracks: dict[int, dict] = {}

    def reset(self) -> None:
        self.tracks.clear()

    def forget(self, live_ids: set[int]) -> None:
        for tid in [t for t in self.tracks if t not in live_ids]:
            del self.tracks[tid]

    def step(self, frame_id: int, poses: dict[int, np.ndarray]) -> list[float]:
        L = self.seg_len
        cur = {}
        for tid, kp in poses.items():
            st = self.tracks.get(tid)
            if st is None:
                st = {"frames": deque(maxlen=L), "xy": deque(maxlen=L), "ctx": deque(maxlen=L), "smoother": ConfSmoother(), "box": None, "last": None}
                self.tracks[tid] = st
            kp = np.asarray(kp, dtype=np.float64)
            if self.smooth:
                kp = st["smoother"].update(kp)
            box = keypoint_box(kp)
            cur[tid] = (st, kp, box)
        if self.ctx:
            ids = list(cur)
            nn_feat = neighbour_features(np.stack([cur[t][2][:2] for t in ids]) if ids else np.zeros((0, 2)),
                                         np.array([max(cur[t][2][3], H_MIN) for t in ids]))
        batch_xy, batch_nodes = [], []
        for n, (tid, (st, kp, box)) in enumerate(cur.items()):
            vel = np.zeros(3)
            if st["box"] is not None:
                gap = float(np.clip(frame_id - st["last"], 1, 5))
                vel[:2] = np.clip((box[:2] - st["box"][:2]) / (gap * max(box[3], H_MIN)), -VEL_CLIP, VEL_CLIP)
            vel[2] = box[2] / max(box[3], H_MIN)
            st["box"], st["last"] = box, frame_id
            st["frames"].append(frame_id)
            st["xy"].append(kp[:, :2])
            if self.ctx:
                st["ctx"].append(pack_nodes(np.stack([vel, nn_feat[n]])))
            if len(st["xy"]) == L:
                frames = np.fromiter(st["frames"], dtype=np.int64)
                if np.count_nonzero(frames < frames[0] + L) >= L - 2:
                    batch_xy.append(np.stack(st["xy"]))
                    if self.ctx:
                        batch_nodes.append(np.stack(st["ctx"]))
        if not batch_xy:
            return []
        with torch.no_grad():
            _, nll = self.flow(to_input(np.stack(batch_xy), np.stack(batch_nodes) if self.ctx else None))
        return [-float(v) for v in nll]


# =====================================================================================================================
# Training windows
# =====================================================================================================================
@dataclass
class GapTrainSet:
    xy: np.ndarray  # (P, 17, 2) float32 pixel keypoints (smoothed when P_D)
    nodes: np.ndarray | None  # (P, 2, 2) float32 packed context nodes (P_C)
    starts: np.ndarray  # (N,) int64 first pose of each window
    seg_len: int

    def __len__(self) -> int:
        return len(self.starts)

    def batch(self, rows: np.ndarray, flip: np.ndarray) -> torch.Tensor:
        idx = self.starts[rows][:, None] + np.arange(self.seg_len)[None, :]
        return to_input(self.xy[idx], None if self.nodes is None else self.nodes[idx], flip)


def build_gap_trainset(pose_dir: Path, seg_len: int, ctx: bool, smooth: bool, stride: int = STRIDE, cache: Path | None = None) -> GapTrainSet:
    """Training tracks (released AlphaPose, normal clips) cut as STG-NF cuts them; smoothing and context are computed per clip on the raw tracks."""
    if cache is not None and Path(cache).exists():
        z = np.load(cache)
        return GapTrainSet(z["xy"], z["nodes"] if ctx else None, z["starts"], int(z["seg_len"]))
    xy_parts, node_parts, starts, base = [], [], [], 0
    for path in sorted(Path(pose_dir).glob("*_tracked_person.json")):
        tracks = {i: (f, kp) for i, (f, kp) in enumerate(read_tracks(path))}
        tracks = {i: (f, conf_smooth(kp) if smooth else kp) for i, (f, kp) in tracks.items()}
        cx = clip_context(tracks) if ctx else None
        for i, (frames, kp) in tracks.items():
            s = window_starts(frames, seg_len, stride)
            xy_parts.append(kp[:, :, :2].astype(np.float32))
            if ctx:
                node_parts.append(pack_nodes(cx[i].reshape(-1, 2, 3)).astype(np.float32))
            if len(s):
                starts.append(s.astype(np.int64) + base)
            base += len(frames)
    ts = GapTrainSet(np.concatenate(xy_parts), np.concatenate(node_parts) if ctx else None, np.concatenate(starts), seg_len)
    if cache is not None:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, xy=ts.xy, nodes=ts.nodes if ctx else np.zeros(0), starts=ts.starts, seg_len=ts.seg_len)
    return ts


# =====================================================================================================================
# Proposal B: hybrid loss
# =====================================================================================================================
def perturb(x: torch.Tensor, gen: torch.Generator, hp: dict = HP) -> tuple[torch.Tensor, torch.Tensor]:
    """Synthetic pseudo-abnormal windows from normal ones (x: (B, 2, T, V) normalised). Per sample one of
    0 temporal frame shuffle, 1 velocity scaling (x_t -> x_0 + s (x_t - x_0), s in [0.2, 0.5] U [1.8, 3.0]), 2 joint jitter (sigma ~ U(.05, .25)).
    Returns (x_pert, kind (B,))."""
    b, _, t, _ = x.shape
    kind = torch.randint(0, 3, (b,), generator=gen)
    perm = torch.argsort(torch.rand(b, t, generator=gen), dim=1)
    shuffled = torch.gather(x, 2, perm[:, None, :, None].expand_as(x))
    slow = torch.rand(b, generator=gen) < 0.5
    lo, hi = torch.where(slow, torch.tensor(hp["speed_slow"][0]), torch.tensor(hp["speed_fast"][0])), torch.where(slow, torch.tensor(hp["speed_slow"][1]), torch.tensor(hp["speed_fast"][1]))
    s = (lo + (hi - lo) * torch.rand(b, generator=gen))[:, None, None, None]
    scaled = x[:, :, :1] + s * (x - x[:, :, :1])
    sigma = (hp["jitter"][0] + (hp["jitter"][1] - hp["jitter"][0]) * torch.rand(b, generator=gen))[:, None, None, None]
    jittered = x + sigma * torch.randn(x.shape, generator=gen)
    k = kind[:, None, None, None]
    return torch.where(k == 0, shuffled, torch.where(k == 1, scaled, jittered)), kind


class _FrozenBN:
    """Context: BatchNorm modules in eval mode (running statistics) for one forward pass, restored afterwards."""

    def __init__(self, model: nn.Module):
        self.bns = [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
        self.state = [m.training for m in self.bns]

    def __enter__(self):
        for m in self.bns:
            m.eval()

    def __exit__(self, *a):
        for m, s in zip(self.bns, self.state):
            m.train(s)


def decode_flow(flow: CouplingFlow, z: torch.Tensor) -> torch.Tensor:
    """f^-1(z): the analytical inverse of every step (affine coupling, channel permutation, ActNorm), same maths as `CouplingFlow.decode` but
    free of its CPU-allocated log-det so it also runs on MPS."""
    for step in reversed(flow.steps):
        z1, z2 = z[:, 0:1], z[:, 1:2]
        shift, scale = step._shift_scale(z1, None, None)
        z = step.permute(torch.cat([z1, (z2 / scale) - shift], dim=1), reverse=True)
        z = z * torch.exp(-step.actnorm.logs) - step.actnorm.bias
    return z


def hybrid_loss(flow: CouplingFlow, x: torch.Tensor, gen: torch.Generator, hp: dict = HP) -> tuple[torch.Tensor, dict]:
    """L = NLL(X) + l1 |X_y - f^-1(f(X))_y|^2 + l2 max(0, m - (NLL(X_pert) - NLL(X))); the perturbed pass uses the running BN statistics
    (the statistics the scorer uses on test windows) so that the normal batch statistics are not polluted."""
    z, nll = flow(x)
    rec = ((x[:, 1] - decode_flow(flow, z)[:, 1]) ** 2).mean()
    x_pert, _ = perturb(x.detach().cpu(), gen, hp)
    with _FrozenBN(flow):
        _, nll_pert = flow(x_pert.to(x.device))
    gap = nll_pert - nll
    hinge = F.relu(hp["margin"] - gap).mean()
    loss = nll.mean() + hp["lambda1"] * rec + hp["lambda2"] * hinge
    return loss, {"nll": nll.mean().item(), "rec": rec.item(), "hinge": hinge.item(), "gap": gap.mean().item(), "gap_pos_frac": (gap > hp["margin"]).float().mean().item()}


# =====================================================================================================================
# Training
# =====================================================================================================================
def train_gap_flow(arm: str, dataset: str, seed: int, out_dir: Path, trainset: GapTrainSet, epochs: int = 10, batch_size: int = 256,
                   lr: float = 5e-4, threads: int = 4, device: str = "cpu", max_batches: int | None = None, hp: dict = HP, log=print) -> Path:
    """STG-NF / LoRaD recipe (Adamax lr 5e-4 x 0.99^epoch, grad-clip 100, identity + mirror augmentation, prior N(3, 1)), resumable; one folder per run."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / "DONE").exists():
        return out_dir / "final.pt"
    spec = ARMS[arm]
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    seg_len = trainset.seg_len
    model = build_gap_flow(arm, seg_len).to(device)
    optimizer = torch.optim.Adamax(model.parameters(), lr=lr)
    n, t0, by_epoch, start_epoch = len(trainset), time.perf_counter(), [], 0
    if (out_dir / "last.pt").exists():
        saved = torch.load(out_dir / "last.pt", map_location=device, weights_only=False)
        model.load_state_dict(saved["state_dict"])
        optimizer.load_state_dict(saved["optimizer"])
        mark_actnorm(model)
        start_epoch, by_epoch = saved["epoch"], saved["history"]
    for epoch in range(start_epoch, epochs):
        rng = np.random.default_rng([seed, epoch])
        gen = torch.Generator().manual_seed(seed * 1000 + epoch)
        rows = np.concatenate([np.arange(n), np.arange(n)])
        flips = np.concatenate([np.zeros(n, dtype=bool), np.ones(n, dtype=bool)])
        order = rng.permutation(2 * n)
        model.train()
        acc: dict[str, list] = {}
        for b, begin in enumerate(range(0, 2 * n, batch_size)):
            if max_batches is not None and b >= max_batches:
                break
            pick = order[begin:begin + batch_size]
            x = trainset.batch(rows[pick], flips[pick]).to(device)
            if spec["rev"]:
                loss, parts = hybrid_loss(model, x, gen, hp)
            else:
                _, nll = model(x)
                loss = nll.mean()
                parts = {"nll": loss.item()}
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 100)
            optimizer.step()
            optimizer.zero_grad()
            parts["loss"] = loss.item()
            for k, v in parts.items():
                acc.setdefault(k, []).append(v)
        for group in optimizer.param_groups:
            group["lr"] = lr * 0.99 ** epoch
        by_epoch.append({k: float(np.mean(v)) for k, v in acc.items()})
        torch.save({"state_dict": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch + 1, "history": by_epoch}, out_dir / "last.pt.tmp")
        (out_dir / "last.pt.tmp").replace(out_dir / "last.pt")
        log(f"[{arm} {dataset} s{seed}] epoch {epoch + 1}/{epochs} " + " ".join(f"{k} {v:.4g}" for k, v in by_epoch[-1].items()) +
            f" ({(time.perf_counter() - t0) / 60:.1f} min)")
    model = model.cpu()
    config = {"arm": arm, **spec, "dataset": dataset, "seg_len": seg_len, "seed": seed, "epochs": epochs, "batch_size": batch_size, "lr": lr,
              "n_windows": n, "params": count_parameters(model), "history": by_epoch, "train_minutes": (time.perf_counter() - t0) / 60,
              "threads": threads, "device": device, "hp": hp if spec["rev"] else None}
    torch.save({"arm_key": arm, "state_dict": model.state_dict(), "config": config}, out_dir / "final.pt")
    (out_dir / "config.json").write_text(json.dumps(config, indent=1))
    (out_dir / "DONE").write_text(time.strftime("%Y-%m-%d %H:%M:%S\n"))
    return out_dir / "final.pt"
