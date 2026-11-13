"""Reconstruction of the Salman et al. 2026 (Sci. Rep.) CNN-GRU, Table 3.

Assumptions A1-A9 (strides, BN placement, last-hidden-state readout, T) are listed in
EXP-002-anchor-reconstruction/plan.md §5.1; the paper does not state them.
"""

import torch
from torch import nn

FEATURE_DIM = 128


def _conv_bn_relu(in_ch: int, out_ch: int, kernel: int, stride: int, groups: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel, stride=stride, padding=kernel // 2, groups=groups),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class FrameEncoder(nn.Module):
    """256x256x3 frame -> 128-d vector. Output shapes follow Table 3 row by row."""

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            _conv_bn_relu(3, 32, 3, stride=2),               # 128x128x32
            _conv_bn_relu(32, 32, 3, stride=2, groups=32),   # depthwise -> 64x64x32
            _conv_bn_relu(32, 64, 1, stride=1),              # pointwise -> 64x64x64
            nn.MaxPool2d(2),                                 # 32x32x64
            _conv_bn_relu(64, FEATURE_DIM, 3, stride=1),     # 32x32x128
            nn.AdaptiveAvgPool2d(1),                         # 1x1x128
            nn.Flatten(),
        )

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        return self.layers(frames)


class SalmanCnnGru(nn.Module):
    """Input clip [B, T, 3, 256, 256] in [0, 1]; output logits [B, num_classes]."""

    def __init__(self, num_classes: int = 2, dropout: float = 0.4):
        super().__init__()
        self.encoder = FrameEncoder()
        self.gru = nn.GRU(FEATURE_DIM, FEATURE_DIM, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(FEATURE_DIM, num_classes)

    def forward(self, clip: torch.Tensor) -> torch.Tensor:
        batch, time = clip.shape[:2]
        features = self.encoder(clip.flatten(0, 1)).view(batch, time, FEATURE_DIM)
        _, last_hidden = self.gru(features)
        return self.classifier(self.dropout(last_hidden[-1]))


class WithSoftmax(nn.Module):
    """Deployment head: probabilities instead of logits (what the ONNX file outputs)."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, clip: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.model(clip), dim=-1)


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def macs_per_frame(model: SalmanCnnGru, height: int = 256, width: int = 256) -> int:
    """Multiply-accumulates of conv + linear + GRU per frame, counted with forward hooks (biases/BN/ReLU ignored)."""
    total = 0

    def conv_hook(module: nn.Conv2d, _inputs, output):
        nonlocal total
        kernel_macs = module.in_channels // module.groups * module.kernel_size[0] * module.kernel_size[1]
        total += output.numel() * kernel_macs

    handles = [m.register_forward_hook(conv_hook) for m in model.modules() if isinstance(m, nn.Conv2d)]
    with torch.no_grad():
        model.encoder(torch.zeros(1, 3, height, width))
    for h in handles:
        h.remove()
    gru_macs = 3 * (FEATURE_DIM * FEATURE_DIM * 2)
    return total + gru_macs + model.classifier.in_features * model.classifier.out_features
