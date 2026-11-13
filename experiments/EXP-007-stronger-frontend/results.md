# EXP-007 results — stronger pose front end (YOLOv8s-pose)

> Same pipeline as EXP-003/006 with YOLOv8s-pose (sha256 `234314cd…cfb3`); c 0.25; scores in
> `~/thesis-data/exp007-{sht,ub}/scores/`, MLflow `EXP-007`. Paired bootstrap over videos (1000). Owner sign-off: pending.

| # | Expectation | Measured (paper / live) | Verdict |
|---|---|---|---|
| V1 | 8s − 8n ≥ +3 at L0 (both datasets) | ShanghaiTech −2.26 [−4.32, −0.43] / −2.17; UBnormal −0.13 [−3.77, +3.35] / −0.42 | **fail** (8s worse or equal) |
| V2 | ShanghaiTech: L2 CLAHE > E0; L3 gamma > E0 | CLAHE +3.24 [+1.49, +4.93] / +3.04 ✔; gamma −1.24 [−2.76, +0.19] / −1.04 ✘ | **half** |
| V3 | UBnormal L3 gamma > E0 | +0.12 [−2.25, +2.21] / +0.29; (CLAHE +2.10 [+0.12, +4.10] / +1.84) | **fail** for gamma; CLAHE helps |
| V4 | Spearman(coverage, AUC) ≥ 0.7 per dataset | ShanghaiTech 0.90 / 0.80; UBnormal 1.00 / 1.00 | **pass** |

8s AUC (paper): ShanghaiTech L0 66.27, L2 69.40 / CLAHE 72.65, L3 67.73 / gamma 66.49; UBnormal L0 55.63, L3 47.83 /
gamma 47.95 / CLAHE 49.92. Coverage: ShanghaiTech L0 **0.569** (8n 0.674), UBnormal L0 0.653 (8n 0.703).

## Observations
1. **A bigger detector is not a better front end here:** at the default gate YOLOv8s-pose tracks *fewer* people than
   8n (coverage 0.569 vs 0.674 on ShanghaiTech) and scores lower — consistent with coverage, not model size,
   deciding AUC (V4 holds). Not explained yet (confidence calibration of 8s vs the 0.25 gate is a candidate).
2. **CLAHE is the one enhancement result that survives across detectors and datasets** (ShanghaiTech 8n L1/L2,
   8s L2, UBnormal 8s L3); **gamma in deep dark does not** (only ShanghaiTech 8n).
3. Darkened ShanghaiTech frames again score above clean with 8s (L2 69.40, L3 67.73 > L0 66.27), the same
   coverage-driven pattern seen with gain/CLAHE in EXP-003.
