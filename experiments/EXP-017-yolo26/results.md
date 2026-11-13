# EXP-017 results — YOLO26n-pose vs YOLOv8n-pose as the edge front end

Plan: `plan.md` (frozen). Log: `log.md`. Full tables: `results_tables.md`; verdict numbers: `verdicts.json`. All AUC x 100.
Weights: official `yolo26n-pose.pt` (sha256 `eb3bb826...22a9`, in the log). No official YOLO26 pose P2 checkpoint exists, so P2 is out of scope.

## 1. Verdicts

| # | rule (frozen) | measured | verdict |
|---|---|---|---|
| Y1 | ShanghaiTech L0 recall higher than YOLOv8n-pose in both the 48-64 and 64-96 px bins | 48-64 px: 0.013 vs 0.057; 64-96 px: 0.209 vs 0.613 (lower in both) | **FAIL** |
| Y2 | SHT L0 paper AUC >= 68.53 + 1.5 = 70.03 and paired CI lower bound > 0 | 61.54; paired diff -6.99 [-10.07, -4.40] | **FAIL** |
| Y3 | UBnormal L0 AUC >= 55.76 - 0.5 = 55.26 | 55.64 (paired diff -0.12 [-2.72, +2.62]) | **PASS** (no transfer loss, no gain) |
| Y4 | NCNN FP16 detector ms at 640 <= 1.10 x EXP-012's (12.40 ms) | 14.28 ms, ratio 1.151 (control run in the same window: YOLOv8n NCNN 12.89 ms, ratio 1.108) | **FAIL** |
| Y5 | delivered live FILL-HOLD AUC at s=1 and s=3 > YOLOv8n-pose's | s=1: 64.33 vs 67.86 (-3.53 [-6.60, -0.96]); s=3: 63.51 vs 65.05 (-1.54 [-3.46, +0.23]) | **FAIL** |

Only Y3 passes, and it passes as "no worse", not as an improvement. YOLO26n-pose is not a better front end than YOLOv8n-pose for this pipeline.

## 2. Accuracy (c 0.25; paper / live AUC; baseline = YOLOv8n-pose from EXP-003 / EXP-006, same pipeline)

| set | YOLO26n paper / live | baseline paper / live | paired paper diff [95% CI] | poses (YOLO26 / baseline) |
|---|---|---|---|---|
| SHT L0 | 61.54 / 61.69 | 68.53 / 66.17 | -6.99 [-10.07, -4.40] | 46,407 / 69,895 |
| SHT L2 | 62.36 / 62.47 | 67.05 / 66.11 | -4.68 [-6.77, -2.79] | 48,763 / 63,888 |
| SHT L3 | 60.41 / 61.57 | 60.43 / 61.71 | -0.03 [-2.85, +3.33] | 47,595 / 56,914 |
| UB L0 | 55.64 / 56.06 | 55.76 / 56.20 | -0.12 [-2.72, +2.62] | 127,512 / 143,821 |
| SHT L0, one-to-many head + NMS (exploratory, not in the plan) | 64.77 / 63.12 | 68.53 / 66.17 | -3.76 [-5.74, -1.88] | 54,571 / 69,895 |

Gate sweep (0.15, 0.10, 0.05): YOLO26n paper AUC moves by at most 0.5 from the c 0.25 value in every set (largest: SHT L3, +0.51 at c 0.05); the conclusions do not depend on the gate.

Recall by reference-person height, SHT L0 (EXP-011 matcher: reference poses conf >= 0.25, IoU >= 0.3):

| detector | 48-64 px | 64-96 px | 96-160 px | >= 160 px | all |
|---|---|---|---|---|---|
| YOLOv8n-pose (EXP-003) | 0.057 | 0.613 | 0.963 | 0.991 | 0.396 |
| YOLO26n e2e head | 0.013 | 0.209 | 0.901 | 0.986 | 0.263 |
| YOLO26n one-to-many + NMS (exploratory) | 0.022 | 0.339 | 0.944 | 0.991 | 0.309 |

The loss is concentrated where the ShanghaiTech people are small (64-96 px). At >= 160 px the two detectors are equal.
The end-to-end head loses more than the one-to-many head; the o2m variant is exploratory and does not change any verdict.

## 3. Cost in the VM (EXP-012 protocol: 3 reps, O0-B4, ffmpeg co-load; detector pre + infer + post, ms/frame)

18 runs, none excluded (mean host load outside the VM 1.9-15%, limit 30%).

| variant | det ms (sd across runs) | service ms |
|---|---|---|
| YOLO26n NCNN FP16 R640 | 14.28 (0.53) | 16.95 |
| YOLO26n NCNN FP16 R800 | 21.22 (0.71) | 24.22 |
| YOLO26n ORT FP32 R640 | 29.57 (0.10) | 33.67 |
| YOLO26n ORT FP32 R800 | 46.54 (0.70) | 51.87 |
| control: YOLOv8n NCNN FP16 R640 | 12.89 (0.46) | 15.69 |
| control: YOLOv8n ORT FP32 R640 | 32.14 (0.67) | 37.04 |
| EXP-012 reference, YOLOv8n NCNN FP16 R640 | 12.40 | |

On NCNN, YOLO26n is 11-15% slower than YOLOv8n, so Y4 fails on both comparators (the plan comparator gives 1.151 and the same-window control gives 1.108). On ORT FP32 it is 8% faster (29.6 vs 32.1 ms), but ORT FP32 is not the deployed configuration. Raising YOLO26n to 800 px costs +49% (NCNN) and +57% (ORT).

## 4. Delivered (24 fps latest-frame policy, VM service traces x s, seed 1, P fill G=8; live AUC)

| group | s | drop rate | SKIP | FILL-HOLD | FILL-NORMAL |
|---|---|---|---|---|---|
| SHT-L0 YOLOv8n (EXP-004 trace; primary) | 1 | 0.007 | 66.07 | 67.86 | 66.32 |
| SHT-L0 YOLOv8n (EXP-004 trace; primary) | 3 | 0.599 | 59.35 | 65.05 | 63.76 |
| SHT-L0 YOLO26n (ORT FP32 R640 trace; primary) | 1 | 0.000 | 61.69 | 64.33 | 61.99 |
| SHT-L0 YOLO26n (ORT FP32 R640 trace; primary) | 3 | 0.570 | 57.83 | 63.51 | 60.15 |
| sensitivity: YOLOv8n, same-window ORT trace | 1 / 3 | 0.000 / 0.606 | 66.17 / 59.52 | 67.62 / 64.38 | 66.26 / 64.00 |
| sensitivity: YOLOv8n, NCNN trace | 1 / 3 | 0.000 / 0.090 | 66.17 / 65.94 | 67.62 / 67.25 | 66.26 / 65.89 |
| sensitivity: YOLO26n o2m cache, NCNN trace | 1 / 3 | 0.000 / 0.166 | 63.12 / 62.13 | 65.11 / 64.09 | 64.29 / 62.84 |

Check: the YOLOv8n group at s=3 reproduces the EXP-010 seed-1 row (SKIP 59.35, FILL-HOLD 65.05).
Paired FILL-HOLD diffs (YOLO26n minus YOLOv8n): s=1 -3.53 [-6.60, -0.96]; s=3 -1.54 [-3.46, +0.23]. Both sensitivity comparisons point the same way (-3.29 / -0.87 with the same-window trace; -2.51 / -3.16 with NCNN traces, the latter with YOLO26n o2m caches).
At s=3 YOLO26n drops slightly fewer frames (57.0% vs 59.9%) because ORT YOLO26n is a little faster, which is too small to offset the accuracy loss. Under the NCNN traces few frames are dropped at s=3 (9% YOLOv8n vs 17% YOLO26n) and the gap widens.

## 5. Reading and threats

- The accuracy loss is a recall loss for 64-96 px people (0.21 vs 0.61 recall), which is where most ShanghaiTech people are. It is consistent with a nano NMS-free model trained at 640 px being weaker on small objects than the YOLOv8n-pose one. This is a hypothesis; the experiment did not test why.
- The head deployed in NCNN differs from the head in the accuracy caches. Accuracy caches use the end-to-end (NMS-free) head, as the plan says. Ultralytics' NCNN export ignores `nms=False` and produces the one-to-many layout (56 x 5040), which needs NMS; ONNX keeps the end-to-end layout. So the NCNN timing includes NMS and a different head from the accuracy rows. The o2m caches (exploratory) cover the NCNN head's accuracy, and they give the same direction (64.77 vs 68.53).
- Caches were built on CPU (FP32); timing traces are FP16 (NCNN) or FP32 (ORT). Y5's primary group pairs the e2e FP32 cache with the ORT FP32 trace; the NCNN pairing is a sensitivity row. Pose-count differences between FP16 and FP32 were not measured.
- VM timing (Lima, 4 vCPU), not a Pi. Y4's margin is small (ratio 1.151 vs the 1.10 limit; 1.108 against the same-window control): against the same-window control the per-rep ratios are 1.13, 1.09, 1.11 (YOLO26n slower in every rep), so a repeat could land on either side of 1.10 for that comparator.
- The UBnormal baseline (EXP-006) uses the same gate and pipeline, so Y3 is a like-for-like comparison. Y3 is a null result (CI includes 0), not evidence of a benefit.
- Y5 uses one seed (1), as the plan says. Its s=3 CI includes 0, so the s=3 loss is not significant on its own; Y5's rule (both s must be higher) fails regardless, since the point estimate is lower at both.
- Not run: a P2 variant (no official weights), larger YOLO26 sizes, any fine-tuning of YOLO26 pose.

## 6. Reproduction
Weights and sha256 in `log.md`; caches `~/thesis-data/exp017-{sht,ub}/cache`; `edgelab.analysis.exp017 {offline,delivered,cost,report}`; VM traces in `~/thesis-data/vm-exchange/results/exp017/`; tests `tests/test_analysis_exp017.py`, `test_pipeline_timing_exp017.py`, `test_pipeline_e2e_decode.py`, `test_pose_detcache_nmsfree.py`.
