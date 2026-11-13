# EXP-010 log (append-only)

### 2026-11-07 17:02 — Owner go (D20); code lane 4 resumed to implement and run the frozen plan

### 2026-11-07 17:04 — step 0a: YOLOv8s-pose ONNX exported
- Added `--name` (default yolov8n-pose, backward compatible) to pipeline/export_onnx.py. `--imgsz 640 --weights artifacts/EXP-001/models/yolov8s-pose.pt --name yolov8s-pose` -> ~/thesis-data/vm-exchange/models/yolov8s-pose-640-384x640.onnx (predictor input 384x640, opset 17, static, no simplify), sha256 9fea9f3e694bf3ebecc2658b53885f6834c7fb5237aac7100a1b7cfd26472d03. Weights sha256 234314cd...cfb3 verified.
- live_timing already has `--yolo-onnx` (override of the file chosen by --imgsz): no change needed there.

### 2026-11-07 17:05 — step 0b: 8s timing started (tmux e10-time)
- New src/edgelab/pipeline/timing_exp010.py: EXP-004 cell L3-E0-R640-c0.25-O0-B4 via cell_in_vm.sh (co-load on), subset 12 videos x 40 frames, live_timing --yolo-onnx = yolov8s ONNX; LoadSampler host-load logging; run excluded and repeated when mean host load outside VM > 30% (waits for one 3 s sample <= 30% before each try, max 6 tries per rep). 3 reps of 8s; ALSO 3 reps of 8n in the same window as a sensitivity check only (r itself = 8s vs the EXP-004 8n traces, as in plan). ONNX parity vs Ultralytics (3 frames with people, host): max conf diff 1e-6, box/keypoint diff < 0.17 px.
- Host load at start (top sampling, outside VM): 16%, 22%, 42% in three consecutive 3 s samples (load avg 6-9 on 10 cores): variable; per-run means recorded in ~/thesis-data/vm-exchange/results/exp010/manifest.csv.

### 2026-11-07 17:15 — step 0 result: r = 2.507 (YOLOv8s-pose / YOLOv8n-pose detector cost, VM, O0-B4, co-load)
- 3 reps of yolov8s (12 videos x 40 frames = 480 frames each), all 'ok', none excluded: host load outside VM per run 6.7%, 6.8%, 12.0% (limit 30%; total host 28-35%). Mean detector ms (pre+infer+post) over 1,440 frames: 79.81 (8s). 8n from the 5 usable EXP-004 O0-B4 L3-E0-R640 runs (2,400 frames): 31.83. **r = 79.81 / 31.83 = 2.507.**
- Sensitivity (not used): 8n re-timed in the same window, 3 runs (host outside VM 10.5%, 6.9%, one more) mean 31.88 ms -> r = 2.503. Raw: ~/thesis-data/vm-exchange/results/exp010/{manifest.csv, r.json, yolov8{s,n}_rep*_try0.csv}.
- Host was busy at the start (load avg 6-9) but outside-VM mean stayed under 30% in every run; no waiting/retry was needed.
- Driver src/edgelab/analysis/exp010.py + tests/test_analysis_exp010.py (7 tests) written; sessions e10-SHT, e10-UB started (threads 3 each).
- Correction to the line above: same-window 8n runs had host outside VM 10.5%, 6.9%, 11.9%.

### 2026-11-07 17:18 — check: driver's SKIP arm == deadline_replay.run_cell
- SHT-L0-8n, s=3, seed 1: run_cell live AUC 0.5934729195899622 (drop 0.5988, 13,082 segments) = exp010 SKIP live row exactly (same AUC, drop rate, segments). The driver re-implements run_cell's steps only to also produce the paper variant. (MLflow experiment 'EXP-010-check' holds the run_cell check; not part of EXP-010.)
- Note for reading: SKIP's coverage column is 1.0 by construction (it delivers a held score on every frame); only the FILL arms' coverage means 'frames with an own score'.

### 2026-11-07 17:25 — driver
- part SHT finished

### 2026-11-07 17:29 — driver
- part UB finished

### 2026-11-07 17:40 — Lead: review of verdicts; degenerate SHT-L0-8s s6 cell disclosed (results.md, Lead review)

### 2026-11-07 17:40 — all runs finished; verdicts
- 36 conditions x 2 variants = 72 rows (~/thesis-data/exp010/rows.jsonl), no errors, no 'no_segments' fallback needed (SHT-L0-8s s=6 FILL has only 30 windows but STG-NF did score). MLflow experiment 'EXP-010'. Paired bootstrap 1000 resamples (exp003.diff), verdicts in ~/thesis-data/exp010/verdicts.json, tables in EXP-010-filling-confirmation/results.md.
- C1 PASS (5 of 6 at s=3: SHT-L0/L2/L3-8n, SHT-L0-8s, UB-L0-8s; UB-L0-8n not sig.). C2 PASS (4 of 6 at s=6: SHT-L0/L2/L3-8n, SHT-L0-8s). C3 PASS (mean live FILL-HOLD - SKIP = +5.65 over 12 cells). C4 PASS (SHT-L0-8s +8.28, UB-L0-8s +6.67). Claim allowed by the frozen rule.
- Caveat recorded in results.md: C2's 4th group (SHT-L0-8s, s=6) is a degenerate cell (30 windows, coverage 0.000, paper-variant diff -0.39 n.s.); without it C2 = 3 of 6.
- pytest: 177 passed, 10 skipped (170 before + 7 new in tests/test_analysis_exp010.py).
- Files touched: pipeline/export_onnx.py (--name option, backward compatible), new pipeline/timing_exp010.py, analysis/exp010.py, tests/test_analysis_exp010.py. No commits.
