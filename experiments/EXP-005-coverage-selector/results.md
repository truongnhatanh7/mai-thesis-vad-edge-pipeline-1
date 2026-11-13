# EXP-005 results — label-free coverage selector on unseen light levels

> Held-out levels X1 (α 0.2), X2 (α 0.05), X3 (α 0.02); candidates E0 / E1 gamma / E2 CLAHE at 640 px, c 0.25;
> ShanghaiTech test, frozen STG-NF. Analysis `edgelab.analysis.exp005` → `artifacts/EXP-005/selector_results.json`;
> scores in `~/thesis-data/exp003/scores/X*_E*_c0.25_botsort/`, rows in `results.jsonl` (MLflow `EXP-003`).
> Owner sign-off: **pending**.

## 1. Results (AUC paper / live; label-free all-frame coverage)

| Level | E0 | E1 gamma | E2 CLAHE | oracle | S picks | regret S |
|---|---|---|---|---|---|---|
| X1 α 0.2 | 67.40 / 65.99 (0.571) | 65.94 / 63.23 (0.551) | **70.32 / 66.47 (0.619)** | E2 | E2 | 0 / 0 |
| X2 α 0.05 | 65.63 / 64.85 (0.531) | 65.65 / 65.06 (0.529) | **66.43 / 65.35 (0.592)** | E2 | E2 | 0 / 0 |
| X3 α 0.02 | 53.04 / 54.41 (0.477) | **59.12 / 60.23 (0.311)** | 55.97 / 58.03 (0.518) | E1 | E2 | 3.14 / 2.20 |

Means over X1–X3 (paper / live): fixed E0 62.02 / 61.75 · fixed E1 63.57 / 62.84 · fixed E2 64.24 / 63.28 ·
**S 64.24 / 63.28** · S24 64.24 / 63.28 · oracle 65.29 / 64.01.

| # | Expectation | Measured | Verdict |
|---|---|---|---|
| F1 | regret(S) ≤ 0.5 at ≥ 2 of 3 levels | 0, 0, 3.14 | **pass** |
| F2 | mean S ≥ best fixed policy | 64.24 = 64.24 (live 63.28 = 63.28) | **pass (tie: S ≡ always-CLAHE here)** |
| F3 | Spearman(coverage, AUC) ≥ 0.7 over 9 configs | 0.917 paper / 0.867 live | **pass** |
| F4 | regret(S24) ≤ regret(S) at X3 | 3.14 = 3.14 (same pick) | holds trivially; the guard does not help |

## 2. Observations

1. **Coverage is a validated, label-free ranking signal out of sample** (ρ 0.87–0.92 on unseen light levels), i.e. a
   cheap runtime health metric for an edge pose-VAD pipeline.
2. **But as a selector it adds nothing over "always CLAHE"** on these levels, and it **fails in deep dark** for the
   second time (L3 in EXP-003, X3 here): gamma has the *lowest* coverage yet the best AUC — in deep dark it produces
   fewer but cleaner tracks, while CLAHE/E0 keep more noisy tracks that STG-NF scores as anomalous. Requiring long
   tracks (S24) does not fix it.
3. Across the 8 noisy light levels without gain (L0–L4, X1–X3), the best cheap option is: none in clean light (CLAHE
   −0.43); CLAHE in moderate dark (α 0.05–0.3: L1, X1, L2, X2); gamma in deep dark (α 0.02–0.03: L3 +2.9, X3 +6.1
   paper over E0); nothing reliable at α 0.01 (L4, all |Δ| CIs include 0). A light-level gate between none / CLAHE /
   gamma would match the oracle at every held-out level (descriptive, not tested).

## 3. Threats

- Post-hoc origin of the hypothesis (EXP-003); this test used new levels of the *same* dataset and sensor model.
- One noise realisation per held-out level; 3 levels only; candidates restricted to deadline-feasible knobs.
- Coverage needs STG-NF segments, so it lags by one segment (24 frames) at runtime.

## 4. Next steps
- If a selector is pursued: pre-register a two-signal rule (mean luma gate gamma↔CLAHE + coverage as a health
  check) and test it on another dataset (UBnormal night scenes) and on real dark footage.

## 5. Owner sign-off
- [ ] pending
