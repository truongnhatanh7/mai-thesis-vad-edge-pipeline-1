# EXP-017 log (append-only)

### 2026-11-08 07:26 — Lane E: weights
- Official YOLO26n-pose weights: https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n-pose.pt (the release asset Ultralytics' own `attempt_download_asset` resolves for the name `yolo26n-pose.pt`; Ultralytics 8.4.201 installed) -> artifacts/EXP-017/yolo26n-pose.pt, 7,878,574 bytes, sha256 eb3bb8268828aeaf515cec23a4bfafd793944a86fe9af94ba7823609c14522a9 (AGPL-3.0, research use only, D11).
- P2: the ultralytics/assets releases (v8.4.0 lists yolo26{n,s,m,l,x}-pose.pt/.onnx only; the only "p2/p6" asset names in any release are `yolov8x-pose-p6.pt` and mobileclip files) contain **no official YOLO26 pose P2 checkpoint**. `yolo26-p2.yaml` exists only as a model config (no weights). Per plan §2: **P2 is out of scope** (no training in this experiment).
- Head mode (found while testing, not in the plan text): Ultralytics 8.4.201 `predict()` on the .pt defaults to the one-to-many head + external NMS (`nms=None`); the NMS-free one-to-one head ("end-to-end output" of plan §2) needs `nms=False` (backend `end2end=True`). Checked on 06_0144 frames: the two heads return different boxes (e.g. 6 vs 2 boxes at conf 0.01). The plan says end-to-end, cache all returned boxes >= 0.01, so the accuracy caches use `nms=False` (new `--nms-free` flag in detcache, default unchanged for all other experiments). Exported ONNX/NCNN models are end-to-end by default (YOLO26 export), i.e. the same head as the caches.

### 2026-11-08 08:30 — Lane E: session crash and resume
- The Cursor session was killed at ~08:2x before any EXP-017 code or cache existed (only the weights file and this log). Resuming: CPU caches (no MPS: EXP-013 is training on MPS), then gate_sweep, recall by height; cost (Y4) and delivered (Y5) only after EXP-013's chain finishes (host quiet) and not overlapping EXP-016 T3.

### 2026-11-08 08:38 — code, exports, parity (CPU only)
- Code (all additive, defaults unchanged for older experiments): `detcache --nms-free` (+ meta `nms_free`, `weights`); `pipeline/ort_pose.py` `decode_e2e` + `decode_any` (layout switch: (1,56,N) YOLOv8 vs (1,300,57) end-to-end; used by OrtPose and NcnnPose), `compare_with_ultralytics(nms_free=)`, `--nms-free` in `export_onnx` and `ncnn_pose`; new `pipeline/timing_exp017.py` (Y4 driver). Tests: test_pipeline_e2e_decode.py (3), test_pose_detcache_nmsfree.py (1), test_pipeline_timing_exp017.py (4).
- Exports into ~/thesis-data/vm-exchange/models: ONNX (opset 17, static, no simplify, `nms=False`, predictor rectangle as EXP-004 Amendment 1): yolo26n-pose-640-384x640.onnx sha256 9a92d053e3b7b2083ae66fbd7c3b0be8f29d07af715f41d7e6f107a02e6c8668; yolo26n-pose-800-480x800.onnx sha256 3ff2ac93ac6d75d27cd628c4903ed11eb2d94a91e3135a38f6193ec2ffbb4b37; output (1, 300, 57) = end-to-end head. NCNN folders yolo26n-pose-{640-384x640,800-480x800}-ncnn (Ultralytics pnnx export).
- **Finding (affects interpretation):** Ultralytics' NCNN export of YOLO26 ignores `nms=False`: the graph ends with the (56, 5040) one-to-many layout (`Reshape 0=5040 1=51`, `out0`), so the NCNN model is the one-to-many head + external NMS (our `decode`, cv2 NMSBoxes), not the end-to-end head. ORT FP32 export honours `nms=False` (end-to-end). So the accuracy caches (end-to-end head, per plan) correspond to the ORT model; the NCNN model is a slightly different detector output. Parity (10 frames x conf 0.25, 0.01, 01_0014/06_0144/03_0032/05_0021/01_0136): ORT e2e vs Ultralytics nms=False: equal counts 20/20, max conf diff 9e-7, max box 0.16 px, max kpt 0.16 px (raw: ~/thesis-data/exp017/parity_ort_e2e_640.txt). NCNN FP32 vs Ultralytics default (one-to-many + NMS): equal counts 10/10, conf diff 2e-6; NCNN FP16: equal counts 10/10, max conf diff 0.024, box 1.25 px, kpt 2.5 px (parity_ncnn_640.txt). Both heads give different boxes on the same frame (e.g. 2 vs 1 boxes at c 0.25 on 06_0144/150).
- Consequence: besides the pre-registered end-to-end caches I add ONE descriptive sensitivity cache, ShanghaiTech L0 with the default (one-to-many + NMS) head = the head NCNN deploys (label `L0_E0-o2m`; not pre-registered, reported separately).
- tmux e17A runs the ShanghaiTech L0 e2e cache (CPU, 2 torch threads) since 08:32; started: yolo26n L0 cache 13 fps at host load ~8. Host-load note: my own CPU jobs and EXP-016's repair jobs count as "outside-VM load", so VM timing (EXP-016 T3, Y4) waits until all CPU jobs and EXP-013's chain (marker ~/thesis-data/exp013-chain/evalwise.done) are finished; T3 and Y4 run in one sequential tmux chain (T3 first, then Y4), so they cannot overlap.

### 2026-11-08 08:42 — VM plumbing smoke for Y4 (not results)
- VM `edge-os` mounts only vm-exchange (rw) and experiments (ro), so all models are under vm-exchange/models (yolo26n ONNX/NCNN there; YOLOv8n controls from EXP-012). Smoke, 3 frames per video x 12 videos, host busy: yolo26n-ort-fp32-R640, yolo26n-ncnn-fp16-R800, yolov8n-ncnn-fp16-R640 all ran end to end in the VM (exit 0; traces in vm-exchange/results/exp017-smoke/, to be ignored).
- Queue (all CPU, 2 torch threads each, started 08:35): e17A (SHT L0 e2e) -> e17A2 (SHT L2, L3 e2e); e17B (UB L0 e2e, after the EXP-016 L2 n960 repair cache finishes); e17C (SHT L0 default-head `L0_E0-o2m`, after UB); e17SweepS / e17SweepU run `gate_sweep --experiment EXP-017` as caches finish (rows ~/thesis-data/exp017-{sht,ub}/results.jsonl).

### 2026-11-08 08:46 — whole suite (CPU, mid-run)
- `pytest tests`: two failures that were not caused by this work, both fixed: (1) `test_live_timing_writes_one_csv_row_per_frame_with_all_columns`: `live_timing.yolo_onnx_for` globbed `yolov8n-pose-640-*x*.onnx` and matched EXP-012's `…-int8.onnx` in the shared models folder -> now excludes `-int8.onnx`; (2) `test_quantize_writes_a_qdq_model_that_runs`: the test's tiny model got onnx 1.23's default IR version 14, newer than the installed ONNX Runtime accepts (13) -> test model built with `ir_version=10` (production code untouched). After the fixes the suite is at 279 passed, 10 skipped (the 10 are the old skips); will be re-run at the end.

- Erratum (08:46): the three entries above were first stamped 08:45 / 08:55 / 09:05 by mistake (clock not checked); corrected to the real times (`date` = 08:46 at this note). Same fix in EXP-016's 'T3 could not have run' entry.

### 2026-11-08 10:56 — Lead: note on the 'L0_E0-o2m' cache
- Lane E queued (exp017-caches2.sh, ~08:35) a ShanghaiTech L0 cache with YOLO26's one-to-many head (needs NMS). This
  variant is **not in the frozen plan**; any result from it is **exploratory** and does not enter Y1–Y5.

### 2026-11-08 10:56 — Lane E: session cut again, resumed
- Cursor disconnect ~10:5x; tmux jobs all kept running (e16fixP, e16e17vm, e17A2, e17C, e17SweepS, e17SweepU). State: EXP-016 repaired caches 107/107 each, L0 n960 K1/K3 scored, L2 posing running; EXP-017 caches: SHT L0 107 (e2e), UB L0 211, SHT L2 103/107, L3 pending, o2m 8/107; VM timing chain (T3 then Y4) waits for EXP-013/014 chain end (evalwise.done) and for Lane E CPU jobs.

### 2026-11-08 13:43 — offline accuracy (Y1-Y3) finished; all caches completeness-checked
- Caches (CPU, `detcache --device cpu --weights artifacts/EXP-017/yolo26n-pose.pt`): ShanghaiTech L0/L2/L3 `--nms-free` (end-to-end head, plan), 107 npz each, 0 partial files; UBnormal L0 `--nms-free`, 211 npz; sensitivity `L0_E0-o2m` (default one-to-many + NMS head of the same weights; NOT in the frozen plan, exploratory), 107 npz (its npz meta carries no `nms_free`/`weights` key because the flag-less path writes none; weights were the same yolo26n-pose.pt). gate_sweep rows: ~/thesis-data/exp017-{sht,ub}/results.jsonl (MLflow EXP-017), 32 + 8 rows.
- c 0.25 paper / live AUC (YOLOv8n-pose baseline in brackets): SHT L0 61.54 / 61.69 (68.53 / 66.17), poses 46,407 (69,895); L2 62.36 / 62.47 (67.05 / 66.11); L3 60.41 / 61.57 (60.43 / 61.71); UB L0 55.64 / 56.06 (55.76 / 56.20); o2m SHT L0 64.77 / 63.12.
- Paired bootstrap (videos, 1000 resamples) paper diff vs baseline: SHT L0 -6.99 [-10.07, -4.40]; L2 -4.68 [-6.77, -2.79]; L3 -0.03 [-2.85, +3.33]; UB L0 -0.12 [-2.72, +2.62]; o2m SHT L0 -3.76 [-5.74, -1.88].
- Recall by height (SHT L0, EXP-011 matcher conf 0.25 IoU 0.3, reference = released poses): 48-64 px: 0.013 (YOLO26 e2e) vs 0.057 (YOLOv8n); 64-96 px: 0.209 vs 0.613; 96-160: 0.901 vs 0.963; >=160: 0.986 vs 0.991. o2m head: 0.022 / 0.339 / 0.944 / 0.991. So YOLO26n-pose finds fewer, not more, small people in this setting (both heads).
- Verdicts so far: Y1 FAIL, Y2 FAIL (61.54 < 70.03; paired CI entirely below 0), Y3 PASS (55.64 >= 55.26). Y4/Y5 wait for VM timing. Raw: ~/thesis-data/exp017/offline.json; tables: EXP-017-yolo26/results_tables.md (regenerated by `exp017 report`).

### 2026-11-08 16:19 — Y4 start
- T3 finished first (no overlap); host load now 2.12 3.34 4.25; running timing_exp017 (6 variants x 3 reps, host-load rule 30%)

### 2026-11-08 16:39 — Y4 done
- manifest ~/thesis-data/vm-exchange/results/exp017/manifest.csv; summary in ~/thesis-data/exp017/cost.json

### 2026-11-08 16:55 — Y5 done
- delivered replays finished (rows ~/thesis-data/exp017/delivered/rows.jsonl)

### 2026-11-08 17:05 — Y5 results, results.md written; erratum
- Erratum: the 16:19 "Y4 start" entry says T3 had finished first; it had in fact crashed at start (EXP-016 log 16:42 entry: unflushed manifest header, fixed, T3 re-run alone and done 16:52). Y4 and T3 never overlapped.
- Y4 (18 VM runs, none excluded, host outside-VM 1.9-15%): YOLO26n NCNN FP16 R640 det 14.28 ms vs EXP-012 12.40 (ratio 1.151) and same-window YOLOv8n control 12.89 (1.108) -> FAIL. Y5 (seed 1, live FILL-HOLD): YOLO26n 64.33 / 63.51 vs YOLOv8n 67.86 / 65.05 at s=1 / s=3 (paired -3.53 [-6.60, -0.96], -1.54 [-3.46, +0.23]) -> FAIL; the YOLOv8n s=3 group reproduces EXP-010 seed 1 (59.35 SKIP, 65.05 FILL-HOLD).
- Final verdicts: Y1 FAIL, Y2 FAIL, Y3 PASS, Y4 FAIL, Y5 FAIL. results.md and results_tables.md/verdicts.json written (`exp017 report`).
