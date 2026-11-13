"""FP32 ONNX export, PyTorch<->ORT parity, ORT-CPU latency (plan.md §5.4).

uv run python -m edgelab.models.onnx_export --run hockey-fight-video-seed0
"""

import argparse
import json
import time
from pathlib import Path

import mlflow
import numpy as np
import onnxruntime as ort
import torch
from torch.utils.data import DataLoader, Subset

from edgelab.data.splits import load_split
from edgelab.data.video_clips import ClipDataset, make_clip_items, read_index, to_model_input
from edgelab.models.salman_cnn_gru import SalmanCnnGru, WithSoftmax
from edgelab.models.train_clips import ARTIFACTS, CACHE_ROOT, EXP_DIR
from edgelab.tracking.mlflow_run import tracked_run

OPSET = 17


def export_onnx(model: torch.nn.Module, out_path: Path, clip_len: int = 8) -> Path:
    deploy = WithSoftmax(model).eval().cpu()
    dummy = torch.rand(1, clip_len, 3, 256, 256)
    torch.onnx.export(deploy, (dummy,), str(out_path), input_names=["clip"], output_names=["probs"],
                      opset_version=OPSET, dynamo=False)
    return out_path


def ort_session(onnx_path: Path, intra_op_threads: int) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.intra_op_num_threads = intra_op_threads
    options.inter_op_num_threads = 1
    return ort.InferenceSession(str(onnx_path), options, providers=["CPUExecutionProvider"])


def parity(model: torch.nn.Module, session: ort.InferenceSession, clips: list[torch.Tensor]) -> dict:
    deploy = WithSoftmax(model).eval().cpu()
    max_delta, flips = 0.0, 0
    with torch.no_grad():
        for clip in clips:
            torch_probs = deploy(clip).numpy()
            ort_probs = session.run(None, {"clip": clip.numpy()})[0]
            max_delta = max(max_delta, float(np.abs(torch_probs - ort_probs).max()))
            flips += int(torch_probs.argmax() != ort_probs.argmax())
    return {"parity_clips": len(clips), "parity_max_abs_delta_prob": max_delta, "parity_label_flips": flips}


def latency(session: ort.InferenceSession, clip: np.ndarray, warmup: int = 10, repeats: int = 100) -> np.ndarray:
    for _ in range(warmup):
        session.run(None, {"clip": clip})
    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        session.run(None, {"clip": clip})
        timings.append(time.perf_counter() - start)
    return np.array(timings) * 1000


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="hockey-fight-video-seed0", help="artifacts/EXP-002/<run>/best.pt")
    parser.add_argument("--threads", default="1,2,4")
    args = parser.parse_args()
    dataset, protocol, seed_part = args.run.rsplit("-", 2)
    seed = int(seed_part.removeprefix("seed"))
    clip_len = 8
    run_dir = ARTIFACTS / args.run
    model = SalmanCnnGru(num_classes=2)
    model.load_state_dict(torch.load(run_dir / "best.pt", map_location="cpu"))
    model.eval()
    onnx_path = export_onnx(model, run_dir / "model_fp32.onnx", clip_len)

    items = make_clip_items(read_index(CACHE_ROOT / dataset), clip_len=clip_len, stride=2, max_clips=8)
    split = load_split(EXP_DIR / "splits" / f"{dataset}_{protocol}_seed{seed}.json", items)
    data = ClipDataset(CACHE_ROOT / dataset, items, clip_len=clip_len, in_memory=False)
    loader = DataLoader(Subset(data, split["test"][:50]), batch_size=1)
    clips = [to_model_input(clip) for clip, _, _ in loader]

    config = {"source_run": args.run, "opset": OPSET, "precision": "fp32", "clip_len": clip_len,
              "ort_version": ort.__version__, "threads": args.threads, "warmup": 10, "repeats": 100,
              "note": "Mac, comparative only"}
    with tracked_run("EXP-002", f"onnx-{args.run}", config, input_files={"onnx": onnx_path}) as mlrun:
        results = parity(model, ort_session(onnx_path, 1), clips)
        mlflow.log_metrics(results)
        for threads in [int(t) for t in args.threads.split(",")]:
            ms = latency(ort_session(onnx_path, threads), clips[0].numpy())
            row = {f"ort_t{threads}_clip_p50_ms": float(np.percentile(ms, 50)),
                   f"ort_t{threads}_clip_p95_ms": float(np.percentile(ms, 95)),
                   f"ort_t{threads}_frame_p50_ms": float(np.percentile(ms, 50)) / clip_len}
            mlflow.log_metrics(row)
            results.update(row)
        results["onnx_bytes"] = onnx_path.stat().st_size
        mlflow.log_metric("onnx_bytes", results["onnx_bytes"])
        results["run_id"] = mlrun.info.run_id
    (run_dir / "onnx_results.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
