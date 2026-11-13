# EXP-012 log (append-only)

### 2026-11-07 18:24 — lane B started; read plan, EXP-004/010 machinery

### 2026-11-07 18:34 — step 1: variants built
- ORT FP32 exists (yolov8n-pose-{640-384x640,800-480x800}.onnx in vm-exchange/models).
- NCNN: `uv pip install --no-deps ncnn` (1.0.20261020) on host env and in the VM venv (~/edgelab-vm); Ultralytics export format=ncnn (pnnx) at imgsz [384,640] and [480,800], half=False -> models/yolov8n-pose-{640-384x640,800-480x800}-ncnn/ (param/bin FP32; FP16 is a runtime option: net.opt.use_fp16_*). NCNN imports and runs in the VM (aarch64, asimdhp present). New pipeline/ncnn_pose.py (NcnnPose, same call signature as OrtPose, reuses letterbox/decode).
- ORT INT8: new pipeline/quantize_int8.py: quantize_static, QDQ, per_channel, weights int8 / activations uint8, MinMax calibration, 200 TRAINING frames (200 different videos, seed 0, one random frame each, L0, list in <out>.calib.txt). Decision (dev only): the 70 non-Conv nodes of the Pose head (/model.22/<op>: box decode Add/Sub/Mul/Div, Sigmoid, Concat, Reshape) are excluded (stay float) because the decode values reach 800 px. Config choice checked on 10 held-out TRAINING frames (not test): minmax: mean |conf diff| ~0.03, mean kpt diff 1.1-2.6 px, box counts equal on 9/10 frames. Percentile calibration was tried and killed after >2.5 min (histogram too slow); not used.
- Files: models/yolov8n-pose-{640-384x640,800-480x800}-int8.onnx (3.8-3.9 MB).
- Parity on 3 test frames (01_0136/220, 05_0021/200, 03_0032/220, conf 0.25) vs ORT FP32 same size: raw in ~/thesis-data/exp012/build/parity_test3.txt. NCNN FP32: counts equal, conf diff 0, kpt <0.001 px. NCNN FP16: counts equal, max conf diff 0.012, max kpt 10.6 px (640)/1.8 px (800). INT8: counts equal except one extra box (640, frame 2), max conf diff 0.065 (640)/0.19 (800), max kpt 26 px (640)/18 px (800), mean kpt ~2-3 px.
- live_timing: new --ncnn-fp16 on|off; --yolo-onnx may be an NCNN folder (make_detector). Backward compatible. New pipeline/timing_exp012.py driver (8 variants x R640/800, seeded random order per rep, host-load rule). Smoke running (3 frames/video).

### 2026-11-07 18:36 — step 2: VM timing started (tmux e12-time)
- Smoke (1 rep, 3 frames/video, 36 frames/variant, NOT results): all 8 variants run in the VM incl. NCNN FP16/FP32; vm-exchange/results/exp012-smoke. Real run: 3 reps x 8 variants (ORT FP32/INT8, NCNN FP16/FP32 x R640/800), 12 videos x 40 frames, L3-E0 c0.25 O0-B4 co-load, seeded random order per rep, driver waits for a <=30% host sample before each run and excludes runs with mean outside-VM host load > 30%. Other lanes' tmux sessions (e10b, e13-chain, e15) were present at start; their load is captured by the host-load column.
- Decision logged before seeing results: the comparator for Q3 is the existing EXP-008 delivered AUC of SHT-L0 E0@640 (EXP-004 trace); the same-window FP32-640 trace is a sensitivity row.

### 2026-11-07 19:57 — Lead: Cursor crashed (~19:0x); lane B lost. Timing had finished; lock released by lead
- Manifest: 24/24 runs ok (8 variants × 3 reps, 18:36–19:03), 0 excluded, host outside VM 4.0–14.4%.
- Lead summary (detector pre+infer+post mean ms; ratio to ORT FP32 640; kept fraction at 24 fps): NCNN FP16 640 12.4
  (0.38×, 1.000) · NCNN FP16 800 18.9 (0.57×, 1.000) · NCNN FP32 640 18.6 (0.57×) · NCNN FP32 800 28.0 (0.85×) ·
  ORT FP32 640 32.9 (1.00×, 0.967) · ORT FP32 800 51.2 (1.56×, 0.731) · ORT INT8 640 13.9 (0.42×) · ORT INT8 800 20.1 (0.61×).
- Q1 timing part: best fast 800 variant (NCNN FP16, 0.57×) ≤ 1.10× → passes on cost; accuracy (Q2/Q3) pending.
- `touch ~/thesis-data/locks/vm_timing_done` at 19:57 (was never written because the lane died).
