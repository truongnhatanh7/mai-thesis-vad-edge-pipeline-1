"""STG-NF-style normalizing flow with swappable coupling networks (EXP-019 GAU, EXP-018 provenance channel).

The flow is the STG-NF flow (third_party/STG-NF, untouched): K steps of  ActNorm -> reverse channel order -> affine
coupling on a (2, T, V) input (channels = x, y; T frames; V = 18 joints), prior N(R, 1), loss in bits per dimension.
Only the coupling network (the "subnet" that makes shift and scale from half of the channels) is ours to change:

    GraphSubnet  the original: one st_gcn block (graph conv over joints + temporal conv over frames)
    GauSubnet    that same graph block, then a Gated Attention Unit over the frames of every joint, added on top

Whatever the subnet is, the step stays invertible: the subnet only looks at z1, and shift / scale are applied to z2.

Optional extras (both off by default, so the default model is the plain STG-NF flow):
    provenance    a (N, 1, T) input, 1 = pose was detected, 0 = pose was predicted (EXP-018 C1). It is a *condition*:
                  the subnet sees it, the flow does not model it (the density is still over x, y only).
    mask_missing  GAU only: attention keys of predicted frames (provenance 0) are masked out (EXP-018 GAU-FA).

    CausalGraphSubnet  (Q2-P1-E06, LoRaD) the graph block with a CAUSAL temporal convolution: left-only zero padding of k - 1 frames
                 (F.pad(x, (0, 0, k - 1, 0)), padding=0 inside the conv), so the output at frame t depends only on frames <= t.
                 `causal-lite` swaps the dense (k, 1) temporal conv for a depthwise one (groups = channels).

Shapes: x (N, 2, T, V) -> z (N, 2, T, V), nll (N,). Tensors only, no hidden state besides ActNorm's first-batch init.
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from edgelab.pose.stgnf_eval import _import_stgnf

_import_stgnf()
from models.STG_NF.graph import Graph  # noqa: E402  (third_party/STG-NF, put on sys.path above)
from models.STG_NF.modules_pose import ActNorm2d, Permute2d, gaussian_likelihood  # noqa: E402
from models.STG_NF.stgcn import ConvTemporalGraphical, st_gcn  # noqa: E402

NUM_JOINTS = 18


def make_adjacency() -> torch.Tensor:
    """STG-NF's graph (OpenPose-18 layout, 'uniform' strategy, max_hops 8): shape (1, 18, 18)."""
    return torch.from_numpy(Graph(strategy="uniform", max_hop=8).A).float()


class GraphSubnet(nn.Module):
    """The original coupling network: input (N, Cin, T, V) -> (N, 2, T, V) = [shift, scale] for the other half."""

    def __init__(self, in_channels: int, n_frames: int, first: bool):
        super().__init__()
        self.register_buffer("A", make_adjacency())
        self.gcn = st_gcn(in_channels, 2, (n_frames // 2 + 1, self.A.size(0)), 1, residual=not first)

    def forward(self, x: torch.Tensor, valid: torch.Tensor | None = None) -> torch.Tensor:
        out, _ = self.gcn(x, self.A)
        return out


class CausalSTGCN(nn.Module):
    """`st_gcn` with a causal temporal convolution (LoRaD): graph conv (1-frame kernel, no temporal footprint) -> BN -> ReLU ->
    temporal conv over the k frames ending at t (left zero padding k - 1, no right padding) -> BN, plus the residual path.

    Same parameter count as the original block with `lite=False`; `lite=True` makes the temporal conv depthwise (groups = channels).
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple[int, int], residual: bool = True, lite: bool = False):
        super().__init__()
        assert len(kernel_size) == 2 and kernel_size[0] >= 1
        self.k = int(kernel_size[0])
        self.gcn = ConvTemporalGraphical(in_channels, out_channels, kernel_size[1])
        self.bn1, self.relu1 = nn.BatchNorm2d(out_channels), nn.ReLU(inplace=False)
        self.tconv = nn.Conv2d(out_channels, out_channels, (self.k, 1), padding=0, groups=out_channels if lite else 1)
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
        h = self.tconv(F.pad(h, (0, 0, self.k - 1, 0)))  # pad the time axis on the left only: no future frame is ever read
        return F.relu(self.bn2(h) + res), A


class CausalGraphSubnet(nn.Module):
    """Causal coupling network (LoRaD): input (N, Cin, T, V) -> (N, 2, T, V); frame t of the output sees only frames <= t of the input."""

    def __init__(self, in_channels: int, n_frames: int, first: bool, lite: bool = False, kernel: int | None = None):
        super().__init__()
        self.register_buffer("A", make_adjacency())
        self.gcn = CausalSTGCN(in_channels, 2, (kernel or n_frames // 2 + 1, self.A.size(0)), residual=not first, lite=lite)

    def forward(self, x: torch.Tensor, valid: torch.Tensor | None = None) -> torch.Tensor:
        out, _ = self.gcn(x, self.A)
        return out


class GatedAttentionUnit(nn.Module):
    """Single-head GAU (Hua et al. 2022) over the time axis, one sequence per (sample, joint).

    x (N, Cin, T, V) -> (N, out_channels, T, V). Per joint j and frame t the token is
    Linear(x[:, :, t, j]) + joint_embedding[j]; then, with SiLU activations:
        u = act(W_u t), v = act(W_v t), z = act(W_z t)           (u, v: hidden = expansion * dim; z: qk_dim)
        q = gamma_q * z + beta_q, k = gamma_k * z + beta_k        (cheap per-dim scale and offset, as in the paper)
        A = relu(q k^T / T + b[t - t'])^2                          (b: learned relative-position bias, 2T - 1 values)
        out = W_o (u * (A v))
    mask_missing: valid (N, T) bool; keys of frames that are not valid get attention weight 0 (the queries stay).
    W_o starts at zero, so a freshly built GAU adds nothing and the subnet starts as the plain graph block.
    """

    def __init__(self, in_channels: int, n_frames: int, n_joints: int = NUM_JOINTS, dim: int = 8, qk_dim: int = 8,
                 expansion: int = 2, out_channels: int = 2, mask_missing: bool = False):
        super().__init__()
        hidden = expansion * dim
        self.n_frames, self.hidden, self.mask_missing = n_frames, hidden, mask_missing
        self.embed = nn.Linear(in_channels, dim)
        self.joint_embedding = nn.Parameter(0.1 * torch.randn(n_joints, dim))
        self.norm = nn.LayerNorm(dim)
        self.qk_dim = qk_dim
        self.to_uvz = nn.Linear(dim, 2 * hidden + qk_dim)  # u, v and z from one matrix multiplication
        self.gamma = nn.Parameter(1.0 + 0.02 * torch.randn(2, 1, 1, qk_dim))  # row 0: q, row 1: k
        self.beta = nn.Parameter(torch.zeros(2, 1, 1, qk_dim))
        self.rel_bias = nn.Parameter(torch.zeros(2 * n_frames - 1))
        self.to_out = nn.Linear(hidden, out_channels)
        nn.init.zeros_(self.to_out.weight)
        nn.init.zeros_(self.to_out.bias)

    def bias_matrix(self) -> torch.Tensor:
        """(T, T) with entry [t, t'] = rel_bias[t - t' + T - 1] (a Toeplitz matrix built from the 2T - 1 learned values)."""
        return self.rel_bias.unfold(0, self.n_frames, 1).flip(1)

    def forward(self, x: torch.Tensor, valid: torch.Tensor | None = None) -> torch.Tensor:
        n, _, t, v = x.shape
        tokens = self.embed(x.permute(0, 3, 2, 1)) + self.joint_embedding[None, :, None, :]  # (N, V, T, dim)
        tokens = self.norm(tokens).reshape(n * v, t, -1)
        u, val, z = torch.nn.functional.silu(self.to_uvz(tokens)).split([self.hidden, self.hidden, self.qk_dim], dim=-1)
        q, k = (z[None] * self.gamma + self.beta).unbind(0)  # (N*V, T, qk_dim) each
        scores = torch.baddbmm(self.bias_matrix().expand(n * v, t, t), q, k.transpose(1, 2), alpha=1.0 / t)
        attention = torch.relu(scores)
        attention = attention * attention  # relu(q k^T / T + b)^2
        if self.mask_missing and valid is not None:
            keys = valid.to(attention.dtype)[:, None, None, :].expand(n, v, 1, t).reshape(n * v, 1, t)
            attention = attention * keys
        out = self.to_out(u * (attention @ val))  # (N*V, T, out)
        return out.reshape(n, v, t, -1).permute(0, 3, 2, 1)


class GauSubnet(nn.Module):
    """Graph block (joint mixing + temporal conv, as the original) plus a GAU over time on the same input."""

    def __init__(self, in_channels: int, n_frames: int, first: bool, gau: dict | None = None):
        super().__init__()
        self.graph = GraphSubnet(in_channels, n_frames, first)
        self.gau = GatedAttentionUnit(in_channels, n_frames, **(gau or {}))

    def forward(self, x: torch.Tensor, valid: torch.Tensor | None = None) -> torch.Tensor:
        return self.graph(x) + self.gau(x, valid)


class CouplingStep(nn.Module):
    """ActNorm -> reverse the 2 channels -> affine coupling (z1 makes shift and scale for z2). Same maths as STG-NF's FlowStep."""

    def __init__(self, subnet: nn.Module):
        super().__init__()
        self.actnorm = ActNorm2d(2, 1.0)
        self.permute = Permute2d(2, shuffle=False)
        self.subnet = subnet

    def _shift_scale(self, z1, condition, valid):
        h = self.subnet(z1 if condition is None else torch.cat([z1, condition], dim=1), valid)
        return h[:, 0:1], torch.sigmoid(h[:, 1:2] + 2.0) + 1e-6

    def forward(self, x, logdet, condition=None, valid=None):
        z, logdet = self.actnorm(x, logdet=logdet, reverse=False)
        z = self.permute(z, reverse=False)
        z1, z2 = z[:, 0:1], z[:, 1:2]
        shift, scale = self._shift_scale(z1, condition, valid)
        z2 = (z2 + shift) * scale
        return torch.cat([z1, z2], dim=1), logdet + torch.log(scale).sum(dim=[1, 2, 3])

    def frame_forward(self, x, condition=None, valid=None):
        """Same transform as forward, but the log-determinant is returned per frame (N, T)."""
        z = self.actnorm_frame(x)
        z = self.permute(z, reverse=False)
        z1, z2 = z[:, 0:1], z[:, 1:2]
        shift, scale = self._shift_scale(z1, condition, valid)
        z2 = (z2 + shift) * scale
        logdet = torch.log(scale).sum(dim=(1, 3)) + self.actnorm.logs.sum() * x.shape[3]
        return torch.cat([z1, z2], dim=1), logdet

    def actnorm_frame(self, x):
        return (x + self.actnorm.bias) * torch.exp(self.actnorm.logs)

    def reverse(self, z, condition=None, valid=None):
        z1, z2 = z[:, 0:1], z[:, 1:2]
        shift, scale = self._shift_scale(z1, condition, valid)
        z2 = z2 / scale - shift
        z = self.permute(torch.cat([z1, z2], dim=1), reverse=True)
        z, _ = self.actnorm(z, logdet=torch.zeros(z.shape[0]), reverse=True)
        return z


class CouplingFlow(nn.Module):
    """K coupling steps + the fixed prior N(R, 1). subnet: 'graph' (original) or 'gau'. provenance: add the condition input."""

    def __init__(self, subnet: str = "graph", n_frames: int = 24, steps: int = 8, prior_mean: float = 3.0,
                 provenance: bool = False, mask_missing: bool = False, gau: dict | None = None):
        super().__init__()
        if subnet not in ("graph", "gau", "causal", "causal-lite"):
            raise ValueError(f"subnet must be 'graph', 'gau', 'causal' or 'causal-lite', got {subnet!r}")
        self.n_frames, self.prior_mean, self.provenance = n_frames, prior_mean, provenance
        in_channels = 1 + int(provenance)
        gau = {**(gau or {}), "mask_missing": mask_missing}
        def make(k: int) -> nn.Module:
            if subnet == "graph":
                return GraphSubnet(in_channels, n_frames, first=(k == 0))
            if subnet in ("causal", "causal-lite"):
                return CausalGraphSubnet(in_channels, n_frames, first=(k == 0), lite=subnet == "causal-lite")
            return GauSubnet(in_channels, n_frames, first=(k == 0), gau=gau)

        self.steps = nn.ModuleList(CouplingStep(make(k)) for k in range(steps))

    def _condition(self, provenance: torch.Tensor | None, x: torch.Tensor):
        """(N, T) provenance -> (N, 1, T, V) condition (same value for every joint); None when the model has no condition."""
        if not self.provenance:
            return None, None
        if provenance is None:
            raise ValueError("this model needs a provenance input (N, T)")
        valid = provenance > 0.5
        return provenance.to(x.dtype)[:, None, :, None].expand(-1, 1, -1, x.shape[3]), valid

    def encode(self, x: torch.Tensor, provenance: torch.Tensor | None = None):
        """x (N, 2, T, V) -> (z, logdet (N,))."""
        condition, valid = self._condition(provenance, x)
        logdet = torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)
        z = x
        for step in self.steps:
            z, logdet = step(z, logdet, condition, valid)
        return z, logdet

    def decode(self, z: torch.Tensor, provenance: torch.Tensor | None = None):
        condition, valid = self._condition(provenance, z)
        for step in reversed(self.steps):
            z = step.reverse(z, condition, valid)
        return z

    def forward(self, x: torch.Tensor, provenance: torch.Tensor | None = None, label: torch.Tensor | None = None):
        """-> (z, nll in bits per dimension, per sample). Same normalisation as STG_NF.normal_flow."""
        z, logdet = self.encode(x, provenance)
        mean = torch.full_like(z, self.prior_mean)
        log_p = gaussian_likelihood(mean, torch.zeros_like(z), z) + logdet
        return z, -log_p / (math.log(2.0) * z[0].numel())


def frame_nll(flow: "CouplingFlow", x: torch.Tensor) -> torch.Tensor:
    """(N, 2, T, V) -> (N, T): every frame's share of the window NLL (bits per dimension), so that nll = frame_nll.mean(1).

    Exact for any flow (it is only a regrouping of the sums), but only a *per-frame causal* quantity when the subnets are causal."""
    n, c, t, v = x.shape
    logdet = torch.zeros(n, t, dtype=x.dtype, device=x.device)
    h = x
    for step in flow.steps:  # the steps again, keeping the per-frame log-determinants
        h, d = step.frame_forward(h)
        logdet = logdet + d
    log_p = gaussian_likelihood_frames(torch.full_like(h, flow.prior_mean), h)
    return -(log_p + logdet) / (math.log(2.0) * c * v)


def gaussian_likelihood_frames(mean: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Per-frame log N(x; mean, 1): (N, C, T, V) -> (N, T) (same constants as modules_pose.gaussian_p with logs = 0)."""
    return (-0.5 * (((x - mean) ** 2) / (1.0 + 1e-6) + math.log(2 * math.pi))).sum(dim=(1, 3))


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def build_flow(kind: str, n_frames: int = 24) -> CouplingFlow:
    """kind: 'graph' | 'gau' | 'causal' | 'causal-lite' | 'graph-fa' | 'gau-fa' | 'gau-fa-mask' (fa = with the provenance condition)."""
    options = {"graph": dict(subnet="graph"), "gau": dict(subnet="gau"), "causal": dict(subnet="causal"),
               "causal-lite": dict(subnet="causal-lite"),
               "graph-fa": dict(subnet="graph", provenance=True), "gau-fa": dict(subnet="gau", provenance=True),
               "gau-fa-mask": dict(subnet="gau", provenance=True, mask_missing=True)}
    return CouplingFlow(n_frames=n_frames, **options[kind])


def random_pose_batch(n: int, n_frames: int = 24, seed: int = 0) -> torch.Tensor:
    """Test helper: (n, 2, T, V) standard-normal input."""
    return torch.from_numpy(np.random.default_rng(seed).standard_normal((n, 2, n_frames, NUM_JOINTS)).astype(np.float32))
