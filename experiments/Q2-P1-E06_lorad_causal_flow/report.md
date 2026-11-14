# Q2-P1-E06 report (generated; see `report_notes.md` for the narrative)

## Narrative (Q2-P1-E06)

Arms: A0 frozen STG-NF, A1 retrained STG-NF, A2 causal-conv + per-window norm (N0), A3 causal-conv + causal prefix/EMA norm (N1, O(1)/frame),
A3s = A3 evaluated in stream mode (ring buffers), A2L = lite depthwise causal conv (extra). 3 seeds per trained arm; CIs are 2,000-resample paired video bootstrap.
Contrast values below are *released poses, S4, all clips, mean over SHT and UB*, delta in AUC points [95% CI].

### Decisions
| rule | result | outcome |
|---|---|---|
| R1 cost of causality (A2 − A1, released, S0) | +0.24 [+0.01, +0.46] | **free** (CI lower ≥ −1) |
| R3 N1 vs N0 (A3 − A2, S4) | released −2.16 [−2.97, −1.37]; ML0 −1.52 [−2.47, −0.58]; YOLO26 −1.29 [−2.06, −0.53] | **N1 not adopted** (needed CI lower ≥ −0.5 on both edge pose sources) |
| R3s stream vs window (A3s − A3, released, S4) | −1.69 [−2.79, −0.50] (SHT −3.18, UB −0.20) | stream mode costs a further ~1.7 pt, mostly on SHT |
| R4 retrained vs frozen (A1 − A0, released) | S0 −1.15 [−1.82, −0.51]; S4 −1.09 [−1.74, −0.46] | retraining alone slightly worse than the released checkpoint |
| X1 A2 − A0 (released, S4) | −0.87 [−1.54, −0.22] | causal LoRaD within ≈ 1 pt of the frozen STG-NF |
| X2 A2L − A2 (released, S4) | −1.64 [−2.70, −0.59] | lite depthwise conv loses ≈ 1.6 pt (408 vs 616 params) |
| selection for E07 | A2 dev-S4 67.86 vs A3 65.45 | **A2, window mode (N0)** |

### Hard checks
* C1 causality passes: causal prefix perturbation → max |Δ| = 0 for causal conv (5.6 for the acausal control); `frame_nll` mean equals window NLL (1.4e-6);
  online stream vs batch max |Δ| = 4.6e-6 over 383 windows.
* R5 (O(1) per-pose cost) **half-passes**. Stream-mode latency is flat in L ∈ {8…96} (0.62–0.64 ms, max/min 1.02–1.04 ≤ 1.15 → flatness PASS), but the
  window-mode cost grows only 1.62× from L = 8 to 96 (0.79 → 1.29 ms; rule needs ≥ 3× → FAIL) because these tiny flows are dominated by fixed per-call overhead.
  The O(1) property holds, but there is no large measured advantage at L ≤ 96 on the Mac; stream mode is not worth its AUC cost (R3s) at this scale.

### Headline numbers (pooled AUC, S0 / S4)
* SHT released: A0 85.94 / 80.85, A2 85.37 / 80.33, A3 83.95 / 78.77, A3s 80.16 / 75.59.
* UB released: A0 71.78 / 71.58, A2 70.53 / 70.35, A3 67.90 / 67.58, A3s 67.74 / 67.38.
* SHT ML0_edge: A0 81.62 / 76.86, A2 81.12 / 76.54, A3 81.61 / 76.55. SHT YOLO26 e2e: A0 64.14 / 62.69, A2 63.84 / 62.11.

### Limits
* A3/N1 (EMA α = 2/(L+1)) was not tuned; the cause of its 1.3–2.2 pt loss was not isolated.
* S4 sentinel is computed from released-train poses (≈ 0.04 pt difference to E01 numbers).
* Latency measured on the Mac CPU (single process, machine quiet), not on the Pi.

## Pooled AUC (all clips), seed mean [95% video-bootstrap CI]

| dataset | pose | arm | S0 | S4 |
|---|---|---|---|---|
| ShanghaiTech | ML0_edge | A0_STGNF_Frozen | 81.62 [78.38, 84.80] | 76.86 [74.02, 79.56] |
| ShanghaiTech | ML0_edge | A1_STGNF_Retrained | 81.57 [78.50, 84.62] | 76.86 [74.11, 79.51] |
| ShanghaiTech | ML0_edge | A2L_Lite | 80.63 [77.44, 83.93] | 76.05 [73.30, 78.80] |
| ShanghaiTech | ML0_edge | A2_LoRaD_CausalConv_WinNorm | 81.12 [77.89, 84.37] | 76.54 [73.73, 79.32] |
| ShanghaiTech | ML0_edge | A3_LoRaD_CausalConv_PrefixNorm | 81.61 [78.70, 84.49] | 76.55 [73.94, 78.97] |
| ShanghaiTech | ML0_edge | A3s_LoRaD_PrefixNorm_Stream | 76.99 [74.08, 79.87] | 72.89 [70.33, 75.46] |
| ShanghaiTech | YOLO26_e2e | A0_STGNF_Frozen | 64.14 [59.52, 69.05] | 62.69 [58.80, 66.90] |
| ShanghaiTech | YOLO26_e2e | A1_STGNF_Retrained | 63.84 [59.29, 68.77] | 62.58 [58.72, 66.73] |
| ShanghaiTech | YOLO26_e2e | A2L_Lite | 63.34 [58.83, 68.19] | 62.47 [58.72, 66.47] |
| ShanghaiTech | YOLO26_e2e | A2_LoRaD_CausalConv_WinNorm | 63.84 [59.35, 68.67] | 62.11 [58.24, 66.29] |
| ShanghaiTech | YOLO26_e2e | A3_LoRaD_CausalConv_PrefixNorm | 63.54 [59.07, 68.48] | 62.46 [58.70, 66.53] |
| ShanghaiTech | YOLO26_e2e | A3s_LoRaD_PrefixNorm_Stream | 61.12 [56.74, 65.94] | 59.65 [55.76, 63.86] |
| ShanghaiTech | released | A0_STGNF_Frozen | 85.94 [82.75, 88.79] | 80.85 [77.86, 83.49] |
| ShanghaiTech | released | A1_STGNF_Retrained | 85.22 [82.08, 88.18] | 80.20 [77.29, 82.83] |
| ShanghaiTech | released | A2L_Lite | 85.37 [82.24, 88.31] | 80.29 [77.35, 82.95] |
| ShanghaiTech | released | A2_LoRaD_CausalConv_WinNorm | 85.37 [82.14, 88.29] | 80.33 [77.47, 82.99] |
| ShanghaiTech | released | A3_LoRaD_CausalConv_PrefixNorm | 83.95 [80.66, 87.09] | 78.77 [75.86, 81.33] |
| ShanghaiTech | released | A3s_LoRaD_PrefixNorm_Stream | 80.16 [77.31, 82.96] | 75.59 [73.07, 78.00] |
| UBnormal | ML0_edge | A0_STGNF_Frozen | 59.21 [54.55, 63.62] | 61.08 [56.70, 64.98] |
| UBnormal | ML0_edge | A1_STGNF_Retrained | 57.38 [52.86, 61.72] | 59.63 [55.35, 63.68] |
| UBnormal | ML0_edge | A2L_Lite | 53.89 [48.77, 58.64] | 55.88 [50.95, 60.27] |
| UBnormal | ML0_edge | A2_LoRaD_CausalConv_WinNorm | 58.02 [53.39, 62.39] | 60.26 [56.00, 64.26] |
| UBnormal | ML0_edge | A3_LoRaD_CausalConv_PrefixNorm | 55.28 [50.65, 59.85] | 57.20 [52.74, 61.37] |
| UBnormal | ML0_edge | A3s_LoRaD_PrefixNorm_Stream | 55.59 [51.03, 59.99] | 57.73 [53.59, 61.65] |
| UBnormal | YOLO26_e2e | A0_STGNF_Frozen | 57.04 [52.28, 62.05] | 59.75 [55.23, 64.18] |
| UBnormal | YOLO26_e2e | A1_STGNF_Retrained | 56.77 [51.97, 61.78] | 59.58 [55.09, 63.97] |
| UBnormal | YOLO26_e2e | A2L_Lite | 54.12 [49.05, 59.33] | 56.92 [52.21, 61.44] |
| UBnormal | YOLO26_e2e | A2_LoRaD_CausalConv_WinNorm | 57.15 [52.31, 62.22] | 60.19 [55.67, 64.64] |
| UBnormal | YOLO26_e2e | A3_LoRaD_CausalConv_PrefixNorm | 54.94 [50.20, 59.84] | 57.26 [52.69, 61.65] |
| UBnormal | YOLO26_e2e | A3s_LoRaD_PrefixNorm_Stream | 54.46 [49.79, 59.26] | 57.01 [52.74, 61.19] |
| UBnormal | released | A0_STGNF_Frozen | 71.78 [67.37, 75.96] | 71.58 [67.29, 75.57] |
| UBnormal | released | A1_STGNF_Retrained | 70.20 [65.55, 74.54] | 70.04 [65.57, 74.23] |
| UBnormal | released | A2L_Lite | 67.36 [62.61, 71.91] | 67.10 [62.55, 71.48] |
| UBnormal | released | A2_LoRaD_CausalConv_WinNorm | 70.53 [65.84, 74.87] | 70.35 [65.90, 74.55] |
| UBnormal | released | A3_LoRaD_CausalConv_PrefixNorm | 67.90 [63.36, 72.26] | 67.58 [63.19, 71.76] |
| UBnormal | released | A3s_LoRaD_PrefixNorm_Stream | 67.74 [63.39, 72.03] | 67.38 [63.27, 71.41] |
