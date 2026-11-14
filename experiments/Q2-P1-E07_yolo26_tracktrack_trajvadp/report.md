# Q2-P1-E07 report — YOLO26-pose → TrackTrack → {STG-NF, LoRaD, TrajVAD-T, TrajVAD-P}

Protocol S4 (strict causal), pooled AUC, seed mean, 2,000-resample paired video bootstrap (seed 20260514). Contrast verdict rule (plan A1): better if CI lo > 0, worse if
CI hi < 0, equal if CI ⊂ [−1, +1], else inconclusive. LoRaD = E06 A2 (window mode, N0), SHT checkpoint seed 0.

## 1. TrackTrack dev sweep (selection rule T1)
12-cell grid on dev clips (YOLO26 e2e, ShanghaiTech + UB mean S4-dev): selected `c0.20 / low0.05 / match0.80` (mean S4-dev 70.17; stranded tracks 71; coverage 0.70).
TrackTrack default (c0.60, low0.25) scores only 60.8 S4-dev on SHT YOLO26 e2e, BoT-SORT default 61.1. Full table: `tracktrack_dev_sweep.csv`.

## 2. Offline / causal matrix — verdict counts (S4, confirmation clips)
| contrast | counts |
|---|---|
| T2 TrackTrack − BoT-SORT | better 10, equal 2, inconclusive 16 (n = 28) |
| V1 LoRaD − STG-NF | equal 3, worse 1, inconclusive 10 (n = 14) |
| V2 TrajVAD-P − LoRaD | worse 11, inconclusive 3 (n = 14) |
| V3 TrajVAD-P − TrajVAD-T | better 6, inconclusive 8 (n = 14) |
| V0 / V0t TrajVAD vs frozen STG-NF | worse 12, inconclusive 2 (n = 14 each) |

Findings
* **TrackTrack never loses to BoT-SORT** and is better in 10 of 28 cells (e.g. SHT YOLO26 e2e + STG-NF S0 +1.53 [0.39, 2.80]); the gain is usually 0.3–1.8 pt and the CIs are wide.
* **LoRaD ≈ frozen STG-NF** (−0.3 to −0.7 pt on SHT, −0.1 to −1.2 on UB; "equal" or inconclusive) — causal retraining is nearly free.
* **TrajVAD-P < LoRaD** by 3.7–5.1 pt on SHT (all front-ends, CI hi < 0) and 0.6–5.6 pt on UB (mostly inconclusive). **TrajVAD-P > TrajVAD-T** by ≈ 1.5 pt on SHT YOLO26/stock
  (better), inconclusive on ML0 and UB. The pose-only port loses the object/appearance cues of the original TrajVAD, so vehicles/bikes are invisible (documented limitation).
* **Front-end dominates everything**: vs the locked E01 baseline (ML0 NCNN + BoT-SORT + STG-NF, SHT S4 76.9), YOLO26 e2e + TrackTrack + STG-NF is −14.2 [−19.1, −9.7],
  YOLO26 o2m −12.0, stock YOLOv8n −7.9; ML0 + TrackTrack + STG-NF is −0.01 (equal). On UB the YOLO26 gaps are inconclusive (−1.0 to −1.7).
* o2m head is 2.2 pt better than e2e on SHT (STG-NF S4 64.4 vs 62.2).

Best pipelines, S4 mean of datasets (confirmation): ML0|TrackTrack|STG-NF 70.33 (SHT 76.36 / UB 64.30); ML0|TrackTrack|LoRaD 69.55; ML0|BoT-SORT|STG-NF 69.38;
ML0|BoT-SORT|LoRaD 68.60; stock|TrackTrack|STG-NF 64.64; ML0|TrackTrack|TrajVAD-P 64.52.
**No YOLO26 pipeline reaches the top 8**: YOLO26n-pose (NCNN-640) is a strictly worse anomaly front-end than the ML0 YOLOv8n-pose on these datasets.

## 3. Raspberry Pi 5 physical benchmark (12 SHT clips × 240 frames = 2,880 frames, 24 fps arrival, 2 s lateness bound)
Run only after `q2_p1_pi5_suite` had finished and `ps` showed zero python processes (no co-loading). PMIC power, SoC temperature and RSS from `Pi5TelemetryMonitor`.
Delivered AUC computed on the Mac from the Pi-logged per-frame min-normality; "ontime" drops windows finishing > 2 s after arrival; "all" keeps late ones.

| arm | FPS | proc/frames | service ms (pose / track / score) | wait p99 ms | W | °C | RSS MB | delivered AUC ontime (cov) | AUC all | offline ideal |
|---|---|---|---|---|---|---|---|---|---|---|
| YOLO26→TT→LoRaD full | 17.3 | 2880/2880 | 56.5 (46.5 / 0.87 / 5.9) | 42,506 | 5.55 | 68.0 | 817 | 62.4 (0.02) | 75.4 | 75.6 |
| YOLO26→TT→LoRaD **+C1+Guard** | **28.7** | 2619/2880 | 33.7 (47.2 / 0.83 / 5.1) | 410 | 5.10 | 72.1 | 821 | **63.6** (0.37) | 63.6 | 75.6 |
| YOLO26→TT→TrajVAD-P full | 14.3 | 2880/2880 | 68.9 (47.3 / 0.88 / 17.4) | 78,169 | 5.49 | 73.7 | 783 | 42.8 (0.01) | 67.8 | 68.0 |
| YOLO26→TT→TrajVAD-P **+C1+Guard** | **25.1** | 2495/2880 | 38.7 (47.6 / 0.81 / 9.4) | 479 | 5.10 | 74.5 | 793 | **61.5** (0.33) | 61.5 | 68.0 |
| ML0→TT→LoRaD full | 14.9 | 2880/2880 | 65.8 (47.1 / 1.55 / 13.9) | 68,319 | 5.51 | 73.8 | 831 | 56.4 (0.03) | 77.6 | 77.1 |
| ML0→TT→LoRaD +C1+Guard | 21.9 | 2378/2880 | 44.6 (47.1 / 1.50 / 11.7) | 440 | 4.98 | 73.0 | 832 | 57.0 (0.37) | 57.0 | 77.1 |
| ML0→TT→TrajVAD-P full | 12.3 | 2880/2880 | 80.2 (47.4 / 1.56 / 27.9) | 109,547 | 5.40 | 75.7 | 810 | 39.7 (0.01) | 63.8 | 65.2 |
| ML0→TT→TrajVAD-P +C1+Guard | 20.0 | 2266/2880 | 48.7 (47.8 / 1.51 / 13.7) | 487 | 5.16 | 75.6 | 815 | 55.3 (0.32) | 55.3 | 65.2 |

Energy per processed frame: 0.17–0.43 J (YOLO26 + C1+Guard LoRaD lowest, 0.172 J). Pose inference (≈ 47 ms NCNN, ARM CPU) is ≈ 85–100 % of the budget; TrackTrack costs 0.8–1.6 ms;
LoRaD 5–14 ms, TrajVAD-P 9–28 ms.  Scorer time is lower with YOLO26 than ML0 (5–9 vs 12–14 ms); the cause (number of scored tracks) was not isolated.

Findings
* **Without the guard no pipeline keeps up with 24 fps**: single-worker service 56–80 ms > 41.7 ms inter-arrival, so wait p99 grows to 42–110 s and the on-time delivered AUC
  collapses to 40–62 with coverage 1–3 %. (The AUC over all, late-delivered windows equals the offline ideal, confirming the scorers are correct.)
* **With Cache-C1+Guard the queue is bounded** (p99 410–490 ms) but only YOLO26 arms exceed the P1 threshold of 25 FPS (28.7 LoRaD, 25.1 TrajVAD-P); ML0 arms reach 20–22 FPS.
* **Acceptance rule C1 is not met** on any pipeline: delivered AUC with the guard is below the offline ideal by 12.0 (YOLO26/LoRaD), 6.5 (YOLO26/TrajVAD-P),
  20.1 (ML0/LoRaD) and 9.9 pt (ML0/TrajVAD-P) — far beyond the −1 pt allowance — because shedding every second frame while the guard is on leaves too few complete windows
  (coverage 0.32–0.37). (Causality is covered by the unit tests; online-vs-batch scorer equality was verified, TrajVAD score diff 7e-6.) The best *delivered* real-time pipeline is **YOLO26 → TrackTrack → LoRaD + C1+Guard: 63.6 AUC at 28.7 FPS, 5.1 W**.
* Temperatures stay 68–76 °C (max 77.7), RSS ≤ 0.83 GB, power 5.0–5.6 W (peaks ≈ 9.8–10 W).

## 4. MLflow
Experiment `Q2-P1-E07` at `http://127.0.0.1:5000`: sweep run, TrajVAD training runs, factorial-matrix run, Pi arms (8 runs), hierarchical `@mlflow.trace` demo; registered models
`edgelab-yolo26n-pose`, `edgelab-lorad-causal-flow`, `edgelab-trajvad-p`.

## 5. Limits
* Pi AUC is measured on 12 SHT clips only (1,044 positive / 2,880 frames); CIs are wide and Pi AUC is not directly comparable with the full-dataset matrix.
* UB uses an MPS proxy for the ML0 front-end (no NCNN cache). The Pi benchmark covers ShanghaiTech clips only.
* The Pi arms use the SHT seed-0 checkpoints of LoRaD (E06 A2) and TrajVAD-P.
* Guard hysteresis (400 / 150 ms) and every-other-frame shedding were taken unchanged from E03/E05, not retuned for the window-completion requirement.
* TrajVAD is a pose-only port (27 kinematic features); the original object/appearance branch is not reproduced.

---
# Generated tables


#### ShanghaiTech — S4 (strict causal) pooled AUC, confirmation clips, seed mean [95% CI]  (S0 offline in parentheses)

| frontend | tracker | STG-NF (frozen) | LoRaD (Causal STG-NF) | TrajVAD-T | TrajVAD-P |
|---|---|---|---|---|---|
| YOLO26n-pose (e2e) | TrackTrack | 62.22 [56.4, 67.4] (64.2) | 61.78 [56.1, 67.0] (63.9) | 56.57 [50.8, 62.1] (58.7) | 58.08 [52.2, 63.6] (60.0) |
| YOLO26n-pose (e2e) | BoT-SORT 2-stage | 61.62 [56.1, 66.6] (62.3) | 60.92 [55.4, 65.9] (62.0) | 55.72 [50.1, 61.0] (57.2) | 57.24 [51.4, 62.6] (58.5) |
| YOLO26n-pose (o2m) | TrackTrack | 64.38 [58.5, 69.7] (67.1) | 63.99 [58.1, 69.4] (66.7) | 58.75 [52.8, 64.3] (60.8) | 60.28 [54.2, 65.9] (62.3) |
| YOLO26n-pose (o2m) | BoT-SORT 2-stage | 63.56 [57.7, 68.9] (65.7) | 63.19 [57.3, 68.5] (65.5) | 57.97 [52.0, 63.4] (60.1) | 59.44 [53.5, 65.0] (61.5) |
| stock YOLOv8n-pose @640 | TrackTrack | 68.48 [62.9, 73.4] (72.3) | 68.18 [62.8, 73.1] (72.0) | 62.03 [56.3, 67.3] (63.8) | 63.67 [57.8, 69.2] (65.5) |
| stock YOLOv8n-pose @640 | BoT-SORT 2-stage | 66.97 [61.4, 71.9] (70.4) | 66.67 [61.2, 71.7] (70.2) | 60.56 [54.7, 65.9] (62.2) | 62.39 [56.6, 67.8] (64.2) |
| ML0 YOLOv8n-pose @640 (NCNN) | TrackTrack | 76.36 [72.9, 79.7] (81.2) | 75.98 [72.5, 79.5] (80.7) | 71.34 [67.1, 75.4] (74.1) | 70.92 [66.5, 75.1] (73.4) |
| ML0 YOLOv8n-pose @640 (NCNN) | BoT-SORT 2-stage | 76.37 [73.0, 79.8] (81.0) | 75.99 [72.5, 79.6] (80.6) | 71.08 [67.0, 74.9] (73.8) | 70.90 [66.7, 74.9] (73.4) |

#### UBnormal — S4 (strict causal) pooled AUC, confirmation clips, seed mean [95% CI]  (S0 offline in parentheses)

| frontend | tracker | STG-NF (frozen) | LoRaD (Causal STG-NF) | TrajVAD-T | TrajVAD-P |
|---|---|---|---|---|---|
| YOLO26n-pose (e2e) | TrackTrack | 61.41 [55.7, 66.9] (58.1) | 61.11 [55.3, 66.8] (57.7) | 56.29 [51.1, 61.5] (54.4) | 56.83 [50.8, 62.7] (56.8) |
| YOLO26n-pose (e2e) | BoT-SORT 2-stage | 60.69 [55.0, 66.3] (57.1) | 60.55 [54.7, 66.2] (56.8) | 55.62 [50.4, 60.9] (53.6) | 55.99 [50.3, 61.8] (55.8) |
| stock YOLOv8n-pose @640 | TrackTrack | 60.80 [55.2, 66.0] (59.3) | 60.74 [55.3, 65.9] (59.5) | 58.40 [53.4, 63.2] (57.8) | 60.19 [54.8, 65.6] (60.1) |
| stock YOLOv8n-pose @640 | BoT-SORT 2-stage | 60.36 [54.8, 65.6] (57.8) | 60.39 [54.9, 65.5] (57.9) | 57.31 [52.2, 62.3] (56.1) | 58.74 [53.2, 64.4] (58.4) |
| ML0 YOLOv8n-pose @640 (MPS proxy) | TrackTrack | 64.30 [59.1, 69.4] (62.1) | 63.12 [57.9, 68.5] (60.7) | 56.22 [51.0, 61.4] (55.7) | 58.11 [52.5, 63.4] (58.2) |
| ML0 YOLOv8n-pose @640 (MPS proxy) | BoT-SORT 2-stage | 62.39 [57.4, 67.4] (59.5) | 61.20 [55.8, 66.3] (57.9) | 54.68 [49.5, 59.7] (54.0) | 55.62 [50.2, 60.6] (55.6) |

#### ShanghaiTech — S4 (strict causal) pooled AUC, all clips, seed mean [95% CI]  (S0 offline in parentheses)

| frontend | tracker | STG-NF (frozen) | LoRaD (Causal STG-NF) | TrajVAD-T | TrajVAD-P |
|---|---|---|---|---|---|
| YOLO26n-pose (e2e) | TrackTrack | 63.51 [59.3, 67.9] (65.7) | 63.13 [59.0, 67.5] (65.4) | 58.04 [53.5, 62.7] (60.3) | 59.60 [55.1, 64.4] (61.8) |
| YOLO26n-pose (e2e) | BoT-SORT 2-stage | 62.69 [58.7, 66.9] (64.1) | 62.11 [58.1, 66.4] (63.8) | 57.00 [52.6, 61.6] (59.1) | 58.56 [54.1, 63.2] (60.5) |
| YOLO26n-pose (o2m) | TrackTrack | 65.69 [61.4, 70.0] (68.7) | 65.27 [61.0, 69.5] (68.3) | 60.06 [55.5, 64.7] (62.5) | 61.90 [57.4, 66.7] (64.3) |
| YOLO26n-pose (o2m) | BoT-SORT 2-stage | 65.09 [60.9, 69.4] (67.5) | 64.71 [60.5, 69.1] (67.2) | 59.40 [54.9, 64.0] (61.6) | 61.17 [56.7, 65.9] (63.4) |
| stock YOLOv8n-pose @640 | TrackTrack | 69.05 [64.8, 73.3] (73.1) | 68.65 [64.5, 72.9] (72.6) | 62.97 [58.4, 67.6] (65.0) | 64.91 [60.3, 69.5] (67.1) |
| stock YOLOv8n-pose @640 | BoT-SORT 2-stage | 67.19 [63.0, 71.4] (70.8) | 66.90 [62.8, 71.2] (70.6) | 61.10 [56.5, 65.6] (63.1) | 63.19 [58.7, 67.8] (65.4) |
| ML0 YOLOv8n-pose @640 (NCNN) | TrackTrack | 76.82 [74.0, 79.6] (81.8) | 76.57 [73.7, 79.4] (81.3) | 72.01 [68.6, 75.3] (75.2) | 72.06 [68.7, 75.5] (74.8) |
| ML0 YOLOv8n-pose @640 (NCNN) | BoT-SORT 2-stage | 76.86 [74.1, 79.5] (81.6) | 76.54 [73.7, 79.3] (81.1) | 71.57 [68.1, 74.9] (74.5) | 71.84 [68.5, 75.2] (74.4) |

#### UBnormal — S4 (strict causal) pooled AUC, all clips, seed mean [95% CI]  (S0 offline in parentheses)

| frontend | tracker | STG-NF (frozen) | LoRaD (Causal STG-NF) | TrajVAD-T | TrajVAD-P |
|---|---|---|---|---|---|
| YOLO26n-pose (e2e) | TrackTrack | 60.77 [56.2, 65.2] (58.1) | 61.14 [56.5, 65.6] (58.2) | 56.50 [52.1, 60.9] (55.5) | 56.51 [51.8, 61.2] (56.5) |
| YOLO26n-pose (e2e) | BoT-SORT 2-stage | 59.75 [55.2, 64.1] (57.0) | 60.19 [55.5, 64.5] (57.2) | 55.56 [51.2, 59.8] (54.4) | 55.50 [50.9, 60.2] (55.4) |
| stock YOLOv8n-pose @640 | TrackTrack | 59.33 [54.8, 63.6] (58.6) | 59.64 [55.3, 64.0] (59.0) | 57.24 [53.1, 61.1] (57.3) | 58.49 [53.8, 62.8] (58.6) |
| stock YOLOv8n-pose @640 | BoT-SORT 2-stage | 59.32 [54.9, 63.6] (58.0) | 59.55 [55.2, 63.9] (58.4) | 57.11 [52.8, 61.1] (56.9) | 58.21 [53.3, 62.9] (58.1) |
| ML0 YOLOv8n-pose @640 (MPS proxy) | TrackTrack | 63.36 [59.3, 67.3] (62.2) | 62.48 [58.3, 66.4] (61.2) | 56.94 [52.6, 61.0] (57.0) | 57.74 [53.3, 61.9] (58.1) |
| ML0 YOLOv8n-pose @640 (MPS proxy) | BoT-SORT 2-stage | 61.08 [56.9, 65.1] (59.2) | 60.26 [56.0, 64.2] (58.0) | 54.80 [50.5, 58.9] (54.6) | 54.89 [50.5, 59.2] (55.0) |
