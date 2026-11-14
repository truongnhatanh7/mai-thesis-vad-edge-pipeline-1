# Q2-P1-E04 — Optional optimizations (+MV, then +INT8): report

**Hardware / honesty disclosure.** The physical Raspberry Pi 5 is **not connected**. Every CPU cost is an `edge-os-arm64-vm` measurement multiplied by the E02/E03 Pi multiplier `s`; delivered AUC is *simulated delivery on replayed detections* (frozen frontend `ML0_ncnn`@640 for ShanghaiTech, `ML0_mps` proxy for UBnormal, locked tracker, frozen STG-NF scorer, E00 causal S3/S4 protocol). Two E04-specific caveats: (i) the VM exposes `asimddp`, `i8mm`, `bf16`, the Pi 5's Cortex-A76 only dot-product, so an INT8 speed-up measured here is an **upper bound**; (ii) the MV streams are **our own x264 IPPP re-encode** of the dataset frames (UBnormal: second generation), not camera streams. No training was done (INT8 = post-training static quantisation, 200 calibration frames from training videos only). All thresholds come from `test_dev` / label-free statistics; verdicts follow the pre-registered rule of `plan.md` section 4 (copied into `config.yaml` before `run`).

## 1. Answer

* **+MV (codec motion vectors replace the pixel-difference cue):** **not adopted**. ShanghaiTech: not adopted (path A matched-cost dev -0.03 [-0.61, +0.56] pt, confirmation -0.08 pt; path B total CPU at the same schedule -2.2 % dev / -2.8 % confirmation, AUC +0.21 / +0.03 pt). UBnormal: not adopted (path A dev -2.25 [-3.99, -0.41], confirmation -0.57; path B +7.5 % / +8.8 %).
* **+INT8 (ML0 ORT static-QDQ INT8 replaces NCNN FP16; ShanghaiTech only):** **not adopted** (evaluated on top of B0 = E03 Cache-C1). Path A dev -1.37 [-3.04, +0.46] pt, confirmation -1.85 pt; path B total CPU -22.2 % dev / -22.8 % confirmation at the identical schedule (AUC -0.70 / -1.06 pt). UBnormal: not evaluated (no UB-geometry INT8 cache).
* **Section-8 gates:** QAT does not fire; cadence-aware scorer does not fire; longer pose fine-tuning: not evaluable with E04 data (proxy: no). **GPU experiment warranted: no.**
* **Correctness rules:** E4-D1 pass, E4-D2 pass, E4-D3 pass, E4-D4 pass, E4-D5 pass, E4-D6 pass, E4-D7 pass.

## 2. VM profile (what each ingredient costs)

| clips | decode T4 plain | decode T4 +MV | export overhead wall | export overhead CPU | BGR convert (retrieve credit) | MV C1 check | MV observer | pixel C1 check | pixel observer |
|---|---|---|---|---|---|---|---|---|---|
| ShanghaiTech 856x480 | 0.039 | 0.175 | 0.136 | 0.374 | 0.530 | 0.114 | 0.127 | 0.373 | 0.588 |
| UBnormal 1280x720 | 0.195 | 0.335 | 0.140 | 0.423 | 0.475 | 0.166 | 0.183 | 0.727 | 1.059 |

ms per frame, VM, mean (cost ingredients are multiplied by 1.25 and by the Pi multiplier `s` in the simulation). The export overhead is charged on **every** frame.

| threads | ML0 NCNN FP16 wall / CPU ms | ML0 ORT FP32 | ML0 ORT INT8 | INT8 speed-up vs NCNN FP16 | INT8 speed-up vs ORT FP32 (quantisation only) |
|---|---|---|---|---|---|
| T=1 | 26.4 / 26.4 | 75.7 / 75.7 | 26.4 / 26.4 | 1.00x | 2.86x |
| T=2 | 14.9 / 28.7 | 42.0 / 83.5 | 16.2 / 31.8 | 0.92x | 2.60x |
| T=3 | 13.1 / 35.9 | 31.1 / 92.8 | 13.3 / 39.4 | 0.99x | 2.34x |
| T=4 | 10.3 / 37.5 | 27.4 / 109.5 | 12.0 / 47.6 | 0.86x | 2.28x |

Session drift (E4-D4): E04 NCNN pose pool vs E02 pool, max wall deviation 5.8 % (limit 10 %). The INT8 pool is multiplied by the E02/E04 NCNN ratio per thread count (an uncorrected-pool arm is reported as sensitivity).

Stock-model profile (NCNN FP16 vs ORT INT8, wall ms / CPU ms): stock-640-ncnn T4 10.5 / 37.8; stock-640-ort-int8 T4 12.1 / 48.3; stock-800-ncnn T4 15.9 / 56.3; stock-800-ort-int8 T4 18.6 / 74.7.

## 3. Step 1 — +MV

### 3.1 Decision (12-cell averages, paired video bootstrap, E03 absolute CPU targets)

| dataset | path | test_dev | test_confirmation | dev ok | conf ok | adopt |
|---|---|---|---|---|---|---|
| ShanghaiTech | A: matched-cost AUC gain | dev -0.03 [-0.61, +0.56] | conf -0.08 | no | no | no |
| ShanghaiTech | B: CPU at identical schedule | dev -2.2 % (AUC +0.21 pt) | conf -2.8 % (AUC +0.03 pt) | no | no | no |
| UBnormal | A: matched-cost AUC gain | dev -2.25 [-3.99, -0.41] | conf -0.57 | no | no | no |
| UBnormal | B: CPU at identical schedule | dev +7.5 % (AUC -1.97 pt) | conf +8.8 % (AUC -0.03 pt) | no | yes | no |

* **ShanghaiTech** CPU accounting (cell mean, ms per video-second; scenario multiplier included): Pixel total 1634 (saved by C1 644, checks 7.6); MV at the same schedule 1672 (saved 632 incl. retrieve-skip 15.8, MV export 26.6, checks 7.2); C1 reuse rate 32.7 % (pixel) vs 31.3 % (MV).
* **UBnormal** CPU accounting (cell mean, ms per video-second; scenario multiplier included): Pixel total 3024 (saved by C1 762, checks 15.1); MV at the same schedule 2763 (saved 1058 incl. retrieve-skip 22.6, MV export 37.6, checks 12.0); C1 reuse rate 23.6 % (pixel) vs 32.0 % (MV).

| dataset | subset | C1-Pixel AUC (avg of 12 cells) | C1-MV AUC | MV - Pixel (95 % CI) pt | cells MV > Pixel |
|---|---|---|---|---|---|
| ShanghaiTech | dev | 73.12 | 73.09 | -0.03 [-0.61, +0.56] | 7/12 |
| ShanghaiTech | confirmation | 73.43 | 73.35 | -0.08 [-0.48, +0.32] | 10/12 |
| ShanghaiTech | all | 73.43 | 73.37 | -0.06 [-0.39, +0.26] | 6/12 |
| UBnormal | dev | 54.51 | 52.25 | -2.25 [-3.99, -0.41] | 2/12 |
| UBnormal | confirmation | 56.00 | 55.43 | -0.57 [-1.83, +0.70] | 4/12 |
| UBnormal | all | 55.51 | 54.37 | -1.14 [-2.17, -0.15] | 4/12 |

### 3.2 Components and sensitivities (matched-cost contrasts, pt of delivered AUC)

| dataset | subset | advection (C1-MV-Advect - C1-MV) | Motion-Only+MV - Motion-Only | C1-MV cost x1.0 - Pixel | C1-MV cost x2.0 - Pixel | level 1 MV - pixel | level 2 MV - pixel | Pixel with VM-measured costs - Pixel |
|---|---|---|---|---|---|---|---|---|
| ShanghaiTech | dev | -0.03 [-0.17, +0.14] | +0.53 [-0.98, +2.06] | +0.31 [-0.26, +0.82] | -0.20 [-0.83, +0.48] | +0.09 [-0.98, +1.24] | -0.19 [-0.87, +0.48] | +0.06 [-0.36, +0.48] |
| ShanghaiTech | confirmation | -0.06 [-0.15, +0.04] | +0.77 [+0.17, +1.40] | -0.22 [-0.66, +0.18] | -0.38 [-0.82, +0.06] | -0.19 [-0.68, +0.32] | -0.34 [-0.89, +0.16] | -0.31 [-0.58, -0.07] |
| UBnormal | dev | +0.05 [-0.00, +0.11] | -0.04 [-0.68, +0.59] | -1.82 [-3.74, +0.23] | -2.31 [-4.17, -0.47] | -0.26 [-0.66, +0.11] | -0.34 [-1.36, +0.71] | -0.17 [-0.70, +0.37] |
| UBnormal | confirmation | +0.07 [+0.01, +0.13] | +0.06 [-0.32, +0.46] | -0.66 [-1.89, +0.61] | -0.61 [-1.91, +0.71] | +0.16 [+0.02, +0.29] | -0.43 [-1.04, +0.15] | -0.40 [-0.76, -0.00] |

### 3.3 Diagnostics

| dataset | motion r (Pearson / Spearman) | border Spearman | C1 static decision agreement (level 3, age 1) | static rate pixel / MV | key-frame (invalid) share | reused-pose error mean (px / h) pixel / MV / advected | p90 |
|---|---|---|---|---|---|---|---|
| ShanghaiTech | 0.82 / 0.88 | 0.74 | 84.2 % (kappa 0.68) | 52.0 / 49.3 % | 2.3 % | 0.045 / 0.047 / 0.040 | 0.09 / 0.10 / 0.07 |
| UBnormal | 0.77 / 0.74 | 0.76 | 74.2 % (kappa 0.49) | 36.6 / 50.2 % | 1.8 % | 0.024 / 0.023 / 0.022 | 0.06 / 0.06 / 0.06 |

Advection: ShanghaiTech dev -0.03 [-0.17, +0.14], confirmation -0.06 [-0.15, +0.04]; UBnormal dev +0.05 [-0.00, +0.11], confirmation +0.07 [+0.01, +0.13].

## 4. Step 2 — +INT8 (ShanghaiTech)

### 4.1 Detector quality (cache-level; locked tracker `c=0.20, low=0.05`, E00 causal S4 with the arm's sentinel, offline S0)

| detector cache | S0 AUC (all) | S4 AUC (all) | S4 dev | S4 confirmation | recall <48 px % | recall 48-96 px % | recall >=96 px % | recall all % |
|---|---|---|---|---|---|---|---|---|
| ML0 NCNN FP16 @640 (deployed baseline) | 81.62 | 76.82 | 77.41 | 76.36 | 37.0 | 85.6 | 96.6 | 72.9 |
| ML0 ORT INT8 @640 (+INT8, calibration only) | 81.42 | 76.63 | 77.37 | 76.11 | 37.2 | 86.1 | 96.8 | 73.2 |
| ML0 ORT FP32 @640 (runtime control) | 81.32 | 76.57 | 77.49 | 75.93 | 36.9 | 85.6 | 96.6 | 72.9 |
| stock YOLOv8n-pose NCNN FP16 @640 | 70.70 | 66.99 | 66.95 | 66.72 | 0.4 | 38.9 | 96.9 | 39.6 |
| stock ORT INT8 @640 (EXP-012) | 71.36 | 67.59 | 68.10 | 67.08 | 0.4 | 39.8 | 97.1 | 40.1 |
| stock ORT FP32 @640 (EXP-012) | 70.83 | 67.12 | 66.99 | 66.89 | 0.4 | 38.8 | 96.9 | 39.6 |
| stock NCNN FP16 @800 (EXP-012) | 73.08 | 69.09 | 69.87 | 68.38 | 0.7 | 49.0 | 98.0 | 44.7 |
| stock ORT INT8 @800 (EXP-012) | 72.42 | 68.68 | 69.48 | 67.95 | 0.6 | 47.2 | 98.0 | 43.8 |

| pair (A - B), pt AUC | S0 all | S0 dev | S0 confirmation | S4 all | S4 dev | S4 confirmation |
|---|---|---|---|---|---|---|
| ML0_ort_int8-ML0_ncnn | -0.20 [-0.99, +0.51] | +0.02 [-0.92, +0.82] | -0.28 [-1.31, +0.67] | -0.19 [-0.90, +0.43] | -0.04 [-0.99, +0.87] | -0.25 [-1.12, +0.54] |
| ML0_ort_int8-ML0_ort_fp32 | +0.10 [-0.63, +0.74] | +0.08 [-0.75, +0.75] | +0.15 [-0.75, +1.08] | +0.06 [-0.54, +0.63] | -0.12 [-1.06, +0.77] | +0.17 [-0.57, +0.90] |
| ML0_ort_fp32-ML0_ncnn | -0.30 [-0.78, +0.14] | -0.05 [-0.37, +0.25] | -0.43 [-1.12, +0.22] | -0.25 [-0.65, +0.08] | +0.08 [-0.24, +0.39] | -0.43 [-1.00, +0.07] |
| stock_ort_int8_640-stock_ncnn640 | +0.65 [-0.11, +1.51] | +1.18 [+0.04, +2.40] | +0.39 [-0.56, +1.54] | +0.61 [-0.08, +1.32] | +1.15 [+0.05, +2.18] | +0.36 [-0.39, +1.26] |
| stock_ort_int8_800-stock_ncnn800 | -0.67 [-1.55, +0.12] | -0.55 [-1.57, +0.50] | -0.74 [-1.95, +0.26] | -0.41 [-1.16, +0.22] | -0.39 [-1.32, +0.65] | -0.43 [-1.36, +0.36] |

| recall pair (A - B), pt | <48 px | 48-96 px | >=96 px | all |
|---|---|---|---|---|
| ML0_ort_int8-ML0_ncnn | +0.2 [-0.2, +0.5] | +0.4 [+0.3, +0.6] | +0.1 [+0.1, +0.2] | +0.3 [+0.2, +0.4] |
| ML0_ort_int8-ML0_ort_fp32 | +0.2 [-0.1, +0.6] | +0.5 [+0.3, +0.6] | +0.1 [+0.1, +0.2] | +0.3 [+0.2, +0.4] |
| ML0_ort_fp32-ML0_ncnn | -0.1 [-0.1, +0.0] | -0.0 [-0.1, +0.0] | -0.0 [-0.0, +0.0] | -0.0 [-0.1, +0.0] |
| stock_ort_int8_640-stock_ncnn640 | +0.0 [-0.0, +0.0] | +0.8 [+0.4, +1.2] | +0.1 [+0.1, +0.2] | +0.4 [+0.2, +0.6] |
| stock_ort_int8_800-stock_ncnn800 | -0.0 [-0.1, -0.0] | -1.8 [-2.3, -1.4] | -0.0 [-0.1, +0.1] | -0.9 [-1.1, -0.7] |

### 4.2 Decision

| base | path | test_dev | test_confirmation | dev ok | conf ok | adopt |
|---|---|---|---|---|---|---|
| on B0 (E03 Cache-C1) | A: matched-cost AUC gain | dev -1.37 [-3.04, +0.46] | conf -1.85 | no | no | no |
| on B0 (E03 Cache-C1) | B: CPU at identical schedule | dev -22.2 % (AUC -0.70 pt) | conf -22.8 % (AUC -1.06 pt) | no | no | no |
| on +MV | A: matched-cost AUC gain | dev -1.92 [-3.69, -0.06] | conf -2.29 | no | no | no |
| on +MV | B: CPU at identical schedule | dev -22.3 % (AUC -0.76 pt) | conf -22.7 % (AUC -0.55 pt) | no | no | no |

Pose-stage CPU at the identical schedule (B0): 1320 -> 1680 ms per video-second (-27.2 % reduction); total 1634 -> 1993. Uncorrected-pool sensitivity (matched cost, INT8 - Pixel): dev -0.77 [-2.38, +0.79], confirmation -1.44 [-2.39, -0.53]. Decision base used: **B0**.

## 5. Per-cell matched-cost AUC (dev / confirmation, `all` stream calibration)

**ShanghaiTech** (`*` = cannot spend the budget: saturated schedule, `!` = cannot get below it)

| cell | target CPU ms/s | C1-Pixel | C1-MV | C1-MV-Advect | C1-INT8 | C1-MV-INT8 | Motion-Only-Pixel | Motion-Only+MV |
|---|---|---|---|---|---|---|---|---|
| s2.0_b0.25 | 569 | 66.11 / 66.77 | 65.54 / 67.28 | 65.52 / 67.47 | 68.00 / 66.29 | 65.05 / 65.75 | 69.39 / 61.77 | 68.00 / 64.80 |
| s2.0_b0.33 | 751 | 67.53 / 70.29 | 68.81 / 68.76 | 68.91 / 68.82 | 69.35 / 67.98 | 67.99 / 66.69 | 67.57 / 66.80 | 65.73 / 67.43 |
| s2.0_b0.50 | 1138 | 71.47 / 71.18 | 70.63 / 71.28 | 70.69 / 71.23 | 69.05 / 70.10 | 70.38 / 70.59 | 67.67 / 68.83 | 68.65 / 68.93 |
| s2.0_b0.67 | 1525 | 75.10 / 75.89 | 75.27 / 75.95 | 75.20 / 75.82 | 71.87 / 71.49 | 72.41 / 71.99 | 70.36 / 71.51 | 71.51 / 72.01 |
| s2.0_b0.75 | 1707 | 75.26 / 76.02 | 75.43 / 76.17 | 75.34 / 76.05 | 73.89 / 74.41 | 71.12 / 71.47* | 72.50 / 71.09 | 72.05 / 72.03 |
| s2.0_b1.00 | 2276 | 76.37 / 76.06* | 76.60 / 76.18* | 76.51 / 76.07 | 76.37 / 76.04* | 76.45 / 76.06* | 77.28 / 75.63 | 76.97 / 75.32 |
| s2.5_b0.50 | 1422 | 71.46 / 71.17 | 70.62 / 71.26 | 70.68 / 71.22 | 69.07 / 70.09 | 70.35 / 70.57 | 68.28 / 68.81 | 67.14 / 69.01 |
| s2.5_b0.75 | 2134 | 75.25 / 75.99 | 75.43 / 76.12 | 75.34 / 76.01 | 73.86 / 74.39 | 71.10 / 71.45* | 72.39 / 71.22 | 72.25 / 72.07 |
| s2.5_b1.00 | 2845 | 76.25 / 75.87* | 76.41 / 76.01* | 76.32 / 75.90 | 76.41 / 76.04* | 76.46 / 75.80* | 77.12 / 75.45 | 75.58 / 75.20 |
| s3.0_b0.50 | 1707 | 71.36 / 71.09 | 70.61 / 71.19 | 70.67 / 71.14 | 68.82 / 69.94 | 70.22 / 70.45 | 68.48 / 69.09 | 65.56 / 69.70 |
| s3.0_b0.75 | 2560 | 75.04 / 75.66 | 74.88 / 75.66 | 74.80 / 75.54 | 73.56 / 74.24 | 70.82 / 71.28* | 72.18 / 71.18 | 71.81 / 71.44 |
| s3.0_b1.00 | 3414 | 76.19 / 75.22* | 76.79 / 74.34* | 76.70 / 74.23 | 70.77 / 67.97* | 71.65 / 70.65* | 54.10 / 65.13 | 68.42 / 67.78 |

**UBnormal** (`*` = cannot spend the budget: saturated schedule, `!` = cannot get below it)

| cell | target CPU ms/s | C1-Pixel | C1-MV | C1-MV-Advect | Motion-Only-Pixel | Motion-Only+MV |
|---|---|---|---|---|---|---|
| s2.0_b0.25 | 1016 | 54.79 / 60.26 | 53.26 / 60.23 | 53.15 / 60.33 | 52.66 / 57.86 | 53.21 / 60.69 |
| s2.0_b0.33 | 1341 | 54.20 / 60.41* | 54.18 / 60.45* | 54.33 / 60.64 | 54.97 / 59.26 | 52.57 / 58.30 |
| s2.0_b0.50 | 2032 | 56.15 / 59.98 | 56.36 / 60.90 | 56.44 / 61.01 | 55.36 / 59.45 | 55.41 / 59.05 |
| s2.0_b0.67 | 2723 | 56.82 / 61.69* | 51.57 / 59.64 | 51.71 / 59.69 | 56.86 / 60.18 | 56.38 / 60.33 |
| s2.0_b0.75 | 3048 | 57.64 / 56.10 | 53.44 / 56.12 | 53.49 / 56.18 | 56.52 / 61.08 | 55.87 / 60.96 |
| s2.0_b1.00 | 4064 | 57.75 / 55.98* | 53.24 / 55.77* | 53.31 / 55.79 | 50.18 / 49.91 | 50.18 / 49.91 |
| s2.5_b0.50 | 2540 | 55.66 / 59.96 | 54.44 / 60.75 | 54.55 / 60.87 | 54.59 / 59.40 | 56.27 / 58.70 |
| s2.5_b0.75 | 3810 | 53.04 / 51.64 | 50.20 / 49.97 | 50.20 / 49.97 | 50.16 / 49.88 | 50.09 / 49.88 |
| s2.5_b1.00 | 5080 | 50.08 / 49.98* | 50.20 / 49.98* | 50.20 / 49.98 | 50.00 / 50.00 | 50.00 / 50.00 |
| s3.0_b0.50 | 3048 | 57.71 / 56.04 | 50.14 / 51.56 | 50.22 / 51.74 | 54.30 / 59.11 | 55.20 / 59.00 |
| s3.0_b0.75 | 4572 | 50.20 / 49.99 | 50.00 / 49.88 | 50.00 / 49.88 | 50.00 / 49.94 | 50.00 / 49.94 |
| s3.0_b1.00 | 6096 | 50.04 / 49.96* | 50.00 / 49.88* | 50.00 / 49.88 | 50.00 / 49.98 | 50.00 / 49.98 |

## 6. Correctness rules (E4-D)

| rule | result | what was checked |
|---|---|---|
| E4-D1 | pass | E03 Cache-C1 and Motion-Only replayed through the extended simulator in 12 cells x 2 datasets: delivered series, ideal series, cost, knob and target bit-identical; the rebuilt run tables equal E03's |
| E4-D2 | pass | one MV record per labelled frame in every clip; a known global translation of 4 px / frame is recovered from the exported vectors; re-encoding 3 clips gives identical grids and statistics (NaN == NaN) |
| E4-D3 | pass | numpy copies of the E03 observers used in the VM equal the E03 functions |
| E4-D4 | pass | E04-session ML0 NCNN pose pool within 10 % of E02's (all T); host CPU outside the VM < 30 %; no cell excluded after retries |
| E4-D5 | pass | 200 frames, seed 0, training/videos only; re-quantising reproduces the identical file hash |
| E4-D6 | pass | C1 never reuses a detector output older than A1 or taken after the current frame; an invalid MV (NaN: key frame) never validates a reuse |
| E4-D7 | pass | INT8 cache: 107 clips, one record per frame, frame ids 0..n-1; re-running 2 clips with the same command reproduces every output field (detections, keypoints, counts, luma) bit for bit; host timing fields (*_ms) are excluded |

## 7. Section-8 gates

* **QAT:** fires = False (deltas INT8 - NCNN FP16 in pt: S0 -0.20, S4 -0.19, delivered (matched cost, ShanghaiTech all) -1.68; CPU reduction at identical schedule -22.6 %). Rule: only if degradation > 0.25 pt while CPU falls.
* **Cadence-aware scorer:** fires = False. Reuse at full coverage (K = 1, zero delay, level 3) changes ideal AUC by pix +0.13 pt (dev +0.07, confirmation +0.16), mv +0.24 pt (dev +0.22, confirmation +0.26), mvadv +0.14 pt (dev +0.13, confirmation +0.15) (pix / mv / mvadv tables).
* **Longer pose fine-tuning:** not evaluable with E04 data (no oracle-detection series); proxy: ML0 recall <48 px 37.0 %, 48-96 px 85.6 %. the oracle-detection (detector-recall-limited vs ideal) series needed for this gate was not built in E04; proxy only: E03 attributes 94 % of missed events to the scorer.
* **GPU experiment:** no GPU experiment warranted (no gate fires with an expected gain above the 0.25 pt margin).

## 8. Known limitations and disclosed order of work

* VM, not Pi; INT8 speed-ups are an upper bound (VM has `i8mm`); MV statistics describe our own x264 IPPP re-encode (UBnormal second generation), not the original capture.
* C1 track identity is an oracle (as E03). The MV static test uses the accumulated *magnitude* (not the vector sum), key frames invalidate a reuse, thresholds are test_dev quantiles of the one-frame statistic.
* INT8 is evaluated on ShanghaiTech only (the UBnormal frontend is the MPS proxy; no UB INT8 cache). The INT8 pose pool is drift-corrected to the E02 NCNN pool.
* Order of work: plan.md and the rule text were written first; ONNX export + INT8 calibration, H.264/MV re-encode and the detector caches (no metric) were produced before `lock`; VM profiling (a machine measurement) and the MV statistics / thresholds (label-free, test_dev) precede `lock`; `config.yaml` was hashed before `run`; the first full `run` is the reported one (see the build notes below for code corrections).

