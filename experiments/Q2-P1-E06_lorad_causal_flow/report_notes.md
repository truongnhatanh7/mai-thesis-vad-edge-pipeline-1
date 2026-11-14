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
