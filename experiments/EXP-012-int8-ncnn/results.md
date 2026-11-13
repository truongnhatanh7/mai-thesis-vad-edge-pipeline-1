# EXP-012 results — "quantize to buy resolution" (written by the lead; lane B was lost in the Cursor crash)

> VM timing: 24/24 runs ok (8 variants × 3 reps, 18:36–19:03, host outside VM 4–14%), raw
> `~/thesis-data/vm-exchange/results/exp012/`. Accuracy: caches `~/thesis-data/exp012/cache/`, gate_sweep rows
> `~/thesis-data/exp012/results.jsonl` (MLflow `EXP-012`), delivered `~/thesis-data/exp012/delivered/q3.json`.

## Verdicts
| # | Rule | Measured | Verdict |
|---|---|---|---|
| Q1 | best fast 800 variant detector ms ≤ 1.10 × ORT FP32 640 | NCNN FP16 800: 18.9 ms vs 32.9 ms = **0.57×** (ORT INT8 800: 0.61×) | **pass** |
| Q2 | its offline paper AUC (L0) ≥ ORT FP32 640 + 1.0 | **70.65** vs 68.53 → +2.12 (INT8 800: 69.54, +1.01) | **pass** |
| Q3 | its delivered live AUC at s 1 and s 2 > ORT FP32 640 delivered (L0) | s 1: **67.57** vs 66.19; s 2: **67.13** vs 62.66 (drops 3.3% vs 39.8%) | **pass** |

## Details
- Same-path control (ORT FP32 640 through the new detector path): 68.54 paper / 66.27 live = EXP-003 68.53 / 66.17 → the
  new path is faithful.
- Precision cost: INT8 at 640 = 67.69 (−0.84 vs FP32); NCNN FP16 at 800 (70.65) ≈ ORT FP32 at 800 in EXP-003 (70.70) →
  FP16 is free, INT8 costs ≈ 1 point.
- L2: NCNN FP16 800 69.61, INT8 800 68.90 (vs 67.05 at FP32 640).
- Timing (detector ms; kept at 24 fps): NCNN FP16 640 12.4 (1.000) · NCNN FP16 800 18.9 (1.000) · NCNN FP32 640 18.6 ·
  NCNN FP32 800 28.0 · ORT FP32 640 32.9 (0.967) · ORT FP32 800 51.2 (0.731) · ORT INT8 640 13.9 · ORT INT8 800 20.1.
- Paired CIs for Q2 were not pre-registered (rule is a point threshold) and are not computed here.

## Threats
VM, not a Pi (NCNN's ARM advantage is expected to hold on a Pi 5 but must be measured there); NCNN parity measured by
the lost lane on 3 frames only (log 18:34); single detector family.
