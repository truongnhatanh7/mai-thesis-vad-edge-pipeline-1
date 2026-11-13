"""Zero-DCE++ (Li et al., official code https://github.com/Li-Chongyi/Zero-DCE_extension, CC BY-NC 4.0).

The network is re-implemented here (layer names match the official one so the official weights load with strict=True),
only for scale_factor = 1 (no down/up-sampling of the input). It predicts one curve-parameter map `r` and brightens
the image with 8 iterations of  x <- x + r * (x^2 - x).
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

EXPERIMENTS_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_WEIGHTS = EXPERIMENTS_ROOT / "third_party/Zero-DCE_extension/Zero-DCE++/snapshots_Zero_DCE++/Epoch99.pth"
CURVE_ITERATIONS = 8


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.depth_conv = nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch)
        self.point_conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        return self.point_conv(self.depth_conv(x))


class ZeroDCEppNet(nn.Module):
    def __init__(self, width: int = 32):
        super().__init__()
        self.e_conv1 = DepthwiseSeparableConv(3, width)
        self.e_conv2 = DepthwiseSeparableConv(width, width)
        self.e_conv3 = DepthwiseSeparableConv(width, width)
        self.e_conv4 = DepthwiseSeparableConv(width, width)
        self.e_conv5 = DepthwiseSeparableConv(width * 2, width)
        self.e_conv6 = DepthwiseSeparableConv(width * 2, width)
        self.e_conv7 = DepthwiseSeparableConv(width * 2, 3)

    def forward(self, x):
        """x: (N, 3, H, W) RGB in [0, 1] -> brightened image, same shape."""
        x1 = torch.relu(self.e_conv1(x))
        x2 = torch.relu(self.e_conv2(x1))
        x3 = torch.relu(self.e_conv3(x2))
        x4 = torch.relu(self.e_conv4(x3))
        x5 = torch.relu(self.e_conv5(torch.cat([x3, x4], 1)))
        x6 = torch.relu(self.e_conv6(torch.cat([x2, x5], 1)))
        r = torch.tanh(self.e_conv7(torch.cat([x1, x6], 1)))
        for _ in range(CURVE_ITERATIONS):
            x = x + r * (x * x - x)
        return x


def load_net(weights: Path = DEFAULT_WEIGHTS) -> ZeroDCEppNet:
    net = ZeroDCEppNet()
    net.load_state_dict(torch.load(weights, map_location="cpu"), strict=True)
    return net.eval()


def bgr_to_input(img_bgr_uint8: np.ndarray) -> np.ndarray:
    """(H, W, 3) BGR uint8 -> (1, 3, H, W) RGB float32 in [0, 1]."""
    rgb = img_bgr_uint8[:, :, ::-1].astype(np.float32) / 255.0
    return np.ascontiguousarray(rgb.transpose(2, 0, 1)[None])


def output_to_bgr(out: np.ndarray) -> np.ndarray:
    """(1, 3, H, W) RGB float in [0, 1] -> (H, W, 3) BGR uint8."""
    rgb = np.clip(out[0].transpose(1, 2, 0), 0.0, 1.0)
    return np.ascontiguousarray(np.round(rgb[:, :, ::-1] * 255.0).astype(np.uint8))


class ZeroDCEpp:
    """PyTorch inference. device: 'cpu' or 'mps'. Callable: BGR uint8 -> BGR uint8."""

    def __init__(self, weights: Path = DEFAULT_WEIGHTS, device: str = "cpu"):
        self.device = device
        self.net = load_net(weights).to(device)

    def __call__(self, img_bgr_uint8: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            x = torch.from_numpy(bgr_to_input(img_bgr_uint8)).to(self.device)
            return output_to_bgr(self.net(x).cpu().numpy())

    def export_onnx(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        net = self.net.to("cpu")
        dummy = torch.zeros(1, 3, 64, 64)
        torch.onnx.export(net, dummy, str(path), input_names=["image"], output_names=["enhanced"], opset_version=17,
                          dynamic_axes={"image": {2: "h", 3: "w"}, "enhanced": {2: "h", 3: "w"}}, dynamo=False)
        self.net.to(self.device)
        return path


class ZeroDCEppOnnx:
    """Same model through ONNX Runtime on CPU (what the edge device would run)."""

    def __init__(self, path: str | Path, threads: int = 1, spin: bool = True):
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = threads  # 0 = ORT default
        if not spin:
            options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        self.session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])

    def __call__(self, img_bgr_uint8: np.ndarray) -> np.ndarray:
        return output_to_bgr(self.run_float(bgr_to_input(img_bgr_uint8)))

    def run_float(self, x: np.ndarray) -> np.ndarray:
        return self.session.run(None, {"image": x})[0]
