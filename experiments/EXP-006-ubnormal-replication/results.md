# EXP-006 results — replication on UBnormal

> UBnormal test (211 videos, 92,640 frames, 30 fps), frozen STG-NF `UBnormal_unsupervised_71_8.tar` (seg_len 16),
> YOLOv8n-pose conf-0.01 cache (MPS) → BoT-SORT replay → STG-NF. Data: HF mirror of the official zip (sha256 and
> 1.37 GB official-prefix checks, log 11:00). Analysis numbers: `artifacts/EXP-006/analysis.json`; deadline replay:
> `~/thesis-data/exp006/deadline/delivered_summary.csv`; MLflow experiment `EXP-006`. Owner sign-off: **pending**.

## 1. Verdicts (plan §3 and Addendum A)

| # | Expectation | Measured (paper / live) | Verdict |
|---|---|---|---|
| U0 | R0 within ±0.5 of 71.8 | 71.78 (= STG-NF's own script) | **pass** |
| U1 | AUC(960) − AUC(640) > 0, CI excludes 0 | +3.25 [−1.64, +7.77] / +2.86 [−1.46, +6.89] | **fail** (positive, n.s.) |
| U2 | G-xfit − default ≤ +0.5 | −0.36 … +0.15 / −0.40 … +0.04 over the 5 configs | **pass** |
| U3 | Spearman(coverage, AUC) over 5 configs ≥ 0.7 | 0.60 / 0.60 | **fail** |
| U4 | light-specific enhancer (dark stratum) | dark stratum empty (min mean luma 64.2) | **not testable** |
| U5 | delivered AUC(960) < AUC(640), O0-B4 | 56.85 vs 56.14: +0.71 [−3.39, +4.74] | **fail** (gain eroded from +2.86 offline, n.s.) |
| UA1 | L2: best of E1/E2 is CLAHE and beats E0 | E2 −0.47 [−2.05, +1.07] / +0.50 [−0.97, +1.98] | **fail** (paper); n.s. point estimate (live) |
| UA2 | L3: best of E1/E2 is gamma and beats E0 | E1 −1.47 [−3.93, +0.67] / −1.25 [−3.50, +0.70] | **fail** (does not replicate) |
| UA3 | Spearman(coverage, AUC) over 6 dark configs ≥ 0.7 | 0.943 / 0.886 | **pass** |

## 2. Main numbers (c 0.25)

| Config | AUC paper | AUC live | label-free coverage |
|---|---|---|---|
| released AlphaPose poses (R0) | 71.78 | 70.32 | — |
| E0 @640 | 55.76 | 56.20 | 0.703 |
| E1 gamma @640 | 55.73 | 56.16 | 0.706 |
| E2 CLAHE @640 | 55.09 | 55.51 | 0.722 |
| E0 @480 | 51.27 (−4.49 [−7.68, −1.47]) | 52.21 (−3.99 [−6.74, −1.38]) | 0.569 |
| E0 @960 | 59.01 | 59.06 | 0.736 |
| L2 (synthetic) E0 / E1 / E2 | 52.75 / 51.76 / 52.28 | 52.52 / 52.79 / 53.03 | 0.576 / 0.571 / 0.601 |
| L3 (synthetic) E0 / E1 / E2 | 48.38 / 46.92 / 48.75 | 49.84 / 48.59 / 49.68 | 0.501 / 0.471 / 0.528 |

Strata (paper, E0 / E1 / E2): weather (29 videos) 36.43 / 36.41 / 34.64; normal (182) 58.65 / 58.63 / 58.26.

Delivered live AUC at 24 fps (EXP-004 traces): 640 px 56.14 (O0-B4), 56.19 (O0r-B2), 56.20 (O1n-B2); 960 px 56.85 (O0-B4,
42% dropped), 55.20 (O0r-B2, 60%), 55.35 (O1n-B2, 58%); 480 px 52.20; E2 55.51 / 55.46 / 55.43.

## 3. Observations

1. **The edge pipeline is close to chance on UBnormal** (55.8 vs 71.8 with the released poses: −16, as on
   ShanghaiTech −17), and *below* chance in fog/smoke/rain (36.4). The pose front end, not the scorer, decides
   accuracy on both datasets.
2. **Replicates:** threshold re-calibration does not help (U2); lowering resolution costs significantly (480 px
   −4.5 / −4.0); coverage tracks AUC across darkened configurations (UA3, ρ 0.89–0.94); deadlines erase most of the
   resolution gain (+2.86 offline → +0.71 delivered at 4 CPUs, −0.99 at a 2-CPU budget; all n.s.).
3. **Does not replicate:** the light-specific enhancer rule. On UBnormal neither gamma in deep dark nor CLAHE in
   moderate dark beats no enhancement; CLAHE slightly hurts on undarkened and weather videos.
4. UBnormal's rendered "night" scenes are not dark (min mean luma 64.2 ≈ ShanghaiTech L1), so natural low light
   could not be tested; the synthetic arm had to carry it.

## 4. Threats

- Near-chance baseline → effects of 1–3 AUC are inside the CIs; the non-replication of the enhancer rule may be
  partly a power problem, not a sign reversal.
- STG-NF's UBnormal checkpoint was trained on AlphaPose poses (style mismatch with YOLO, as on ShanghaiTech).
- Third-party mirror (91% of bytes verified only against the mirror's own hash).
- U5: 24 fps simulated on 30 fps video; traces measured at 384×640 understate the cost of the 448×640 videos.

## 5. Owner sign-off
- [ ] pending
