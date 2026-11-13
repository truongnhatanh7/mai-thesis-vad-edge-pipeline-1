"""EXP-020: NCNN export parity of one arm. Ultralytics predict (.pt, CPU, imgsz 640, iou 0.7) vs NcnnPose FP32 / FP16 on 12 ShanghaiTech
test frames (every 20th frame of the first 12 videos, original frames, no darkening) at conf 0.1 and 0.25.

    uv run --extra pose python EXP-020-small-person-detector/scripts/ncnn_parity.py <arm>        # from experiments/
Writes artifacts/EXP-020/ncnn_parity_<arm>.json (raw rows + summary).
"""
import json
import sys
from pathlib import Path

import numpy as np

from edgelab.pipeline.ncnn_pose import NcnnPose
from edgelab.pipeline.ort_pose import compare_with_ultralytics

arm = sys.argv[1]
root = Path.home() / "thesis-data"
folder = root / "vm-exchange/models/exp020" / f"{arm}-640-384x640-ncnn"
frames_root = root / "shanghaitech/shanghaitech/testing/frames"
frames = []
for video in sorted(p.name for p in frames_root.iterdir())[:12]:
    files = sorted((frames_root / video).glob("*.jpg"))
    frames.append(str(files[len(files) // 2]))
weights = Path("artifacts/EXP-020") / f"{arm}.pt"
out = {}
for name, fp16 in (("fp32", False), ("fp16", True)):
    rows = compare_with_ultralytics(str(weights), frames, NcnnPose(folder, threads=2, fp16=fp16), [0.1, 0.25], 640)
    same = [r for r in rows if r["n_ultralytics"] == r["n_ort"]]
    out[name] = {"rows": rows, "frames_x_conf": len(rows), "equal_counts": len(same),
                 "max_conf_diff": max((r.get("max_conf_diff", 0) for r in same), default=None),
                 "max_kpt_diff_px": max((r.get("max_kpt_diff_px", 0) for r in same), default=None),
                 "count_ultralytics": int(sum(r["n_ultralytics"] for r in rows)), "count_ncnn": int(sum(r["n_ort"] for r in rows))}
Path("artifacts/EXP-020").mkdir(parents=True, exist_ok=True)
Path(f"artifacts/EXP-020/ncnn_parity_{arm}.json").write_text(json.dumps(out, indent=1))
print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "rows"} for k, v in out.items()}))
