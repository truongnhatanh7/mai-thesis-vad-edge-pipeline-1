# Q2-P1-E06 — LoRaD Causal Pose Flow: pre-registered plan

*Written before any E06 number exists. Anything that changes after the first run goes into "Amendments" at the bottom with its reason.*

**Question.** LoRaD (Zhang et al., JRTIP 23(4), 2026, doi 10.1007/s11554-026-01937-6) claims frame-level AUC 84.8 % (ShanghaiTech) and
71.2 % (UBnormal) with a *causal* ST-GCN flow (causal temporal convolutions + online circular buffer) and a 1.27 M-parameter distilled
MobileNetV3 pose encoder. (a) What does causality cost the flow itself, on the released AlphaPose poses and on our edge poses? (b) Does a
causal running normalisation (N1) keep the accuracy of per-window normalisation (N0) and thereby allow O(1) per-frame circular-buffer evaluation?

**Scope / honesty.** Only the *flow* (Causal STG-NF) is reproduced. The MobileNetV3 distilled pose encoder, its training data and the paper's
exact normalisation are not public in the abstract/DOI record we could access; the paper's 84.8 / 71.2 are therefore a *reference to compare
against*, not a number this lane can claim to reproduce with the identical system. Training: **yes** (flows only; poses are fixed).

## 1. Model (`edgelab.models.stgnf_flow`, `edgelab.models.lorad`)
* Flow = STG-NF flow (K = 8 steps ActNorm → channel reverse → affine coupling, prior N(3, 1), loss in bits/dim). Only the coupling
  subnet changes: **`CausalGraphSubnet`** = `CausalSTGCN` (graph conv over joints, BN, ReLU, temporal conv with kernel `k = L//2 + 1`
  and **left-only zero padding `F.pad(x, (0, 0, k-1, 0))`, conv padding 0**, BN, residual). Frame t of every coupling output depends on
  frames ≤ t; the per-frame NLL share of frame t depends only on poses ≤ t. Parameter count identical to STG-NF (**616**). Optional
  depthwise temporal conv `causal-lite` (408 params) is trained for one seed as an extra.
* **N0** per-window normalisation (STG-NF rule: window mean, std of y). **N1** causal EMA normalisation (span L, α = 2/(L+1), started at
  the first pose of the track; centroid and first/second moment of y): every pose is normalised once and never changes.
* Window score = −NLL of the window (L consecutive poses of one track, ≤ 2 missing frames), landed at the window's last pose (S4) or
  at the middle (S0). **Stream mode (N1 only):** per-pose evaluation with k−1-frame ring buffers per coupling step; window score = mean of the last
  L per-frame NLLs in a circular buffer: O(steps·k) per pose, independent of L.

## 2. Arms (trained on the 330 SHT / 186 UB normal training clips of the released AlphaPose poses; STG-NF recipe: Adamax 5e-4 ×0.99^epoch,
batch 256, **10 epochs**, identity + mirror augmentation, windows every 6 poses, grad-clip 100)
| Arm | Flow | Norm | Seeds |
|---|---|---|---|
| `A0_STGNF_Frozen` | released STG-NF (616 params) | N0 | – |
| `A1_STGNF_Retrained` | symmetric (acausal) STG-NF, our code path | N0 | 0, 1, 2 |
| `A2_LoRaD_CausalConv_WinNorm` | causal | N0 | 0, 1, 2 |
| `A3_LoRaD_CausalConv_PrefixNorm` | causal | N1 (window mode) | 0, 1, 2 |
| `A3s` (same A3 weights) | causal | N1, **stream mode** (O(1)) | 0, 1, 2 |
| `A2L_Lite` (extra) | causal, depthwise | N0 | 0 |

## 3. Evaluation grid
Pose sources (never re-detected; tracks from the locked pipelines of E01): `released` (AlphaPose test poses of the STG-NF authors) ·
`ML0_edge` (E01 locked frontend + tracker: SHT `ML0_ncnn`; UB `ML0_mps` proxy, no UB NCNN cache exists; BoT-SORT two-stage c0.20/low0.05/b30/m0.85) ·
`YOLO26_e2e` (YOLO26n-pose NMS-free cache of EXP-017, same locked tracker). Protocols (E00): **S0** offline paper protocol, **S4** strict causal
(completion landing, causal half-Gaussian, train-calibrated sentinel, hold 0). Datasets: SHT 107 clips (35 dev / 72 confirmation), UB 211 clips
(70 / 141). Metric: pooled frame-level AUC; 2,000-resample paired **video** bootstrap (seed 0), for seed-averaged arms the resample statistic is
the mean over the 3 seeds. Sentinel: train-calibrated quantile (SHT max, UB p50; E00 locked) of the arm's *own* −NLL distribution over the
released-pose training windows (in-sample, as for the frozen model in E00 `released_all`).

## 4. Pre-registered checks and rules
* **C1 causality (hard).** Changing poses after frame t never changes any per-frame NLL share ≤ t (max |Δ| < 1e-5); stream mode equals the
  whole-track batch evaluation (max |Δ| < 1e-4); `OnlineLoRaD` equals the table engine.
* **R1 cost of causality.** `A2 − A1` on released poses, S0, seed mean per dataset: *causality is free* if the CI lower bound ≥ −1.0 pt.
* **R2 reproduction reference.** Report A2/A3 released S0 against the paper's 84.8 (SHT) / 71.2 (UB) and against the frozen A0 (85.94 / 71.78).
  No pass/fail: the pose encoder differs.
* **R3 N1 vs N0.** `A3 − A2` on edge poses (ML0_edge and YOLO26_e2e), S4, mean over datasets: *N1 adopted* if the paired CI lower bound ≥ −0.5 pt.
  Stream-vs-window gap `A3s − A3` reported (the exactness price of true O(1)).
* **R4 retrain vs frozen.** `A1 − A0` under every cell (is the retraining recipe sound).
* **R5 O(1).** Per-pose stream latency (1 thread, CPU) must be flat in L ∈ {8, 16, 24, 48, 96} (max/min ≤ 1.15), whereas window-mode cost grows ≥ 3× from L = 8 → 96.
* **Selection for E07 (LoRaD).** The causal arm (A2 or A3) with the higher dev-S4 mean AUC (mean over datasets, ML0_edge); ties within 0.5 pt → A3 (deployable O(1)).

## 5. Outputs
`results.csv` (one row per arm × seed × pose source × dataset × protocol × split), `summary.json`, `report.md`, `tests/test_q2_p1_e06.py`;
checkpoints in `artifacts/Q2-P1-E06/`; MLflow runs in experiment `Q2-P1-E06` on `http://127.0.0.1:5000`.

## Amendments
* **A0 (before any result).** The task text estimates flow training at "1–2 minutes". Measured: 0.13 s/batch (SHT) / 0.075 s/batch (UB) at 2 threads → 25 / 5
  minutes per 10-epoch model, 20 models → run 4 in parallel (2 threads each) in the background.
