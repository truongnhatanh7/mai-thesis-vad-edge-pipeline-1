# EXP-007 — Does a stronger pose front end change the conclusions?

Pre-registration frozen 2026-11-06 20:45, before any EXP-007 run (D19: execute, review afterwards).

## 1. Why
On both datasets the edge pose front end costs 16–17 AUC (EXP-001, EXP-006) and UBnormal is near chance, which may
have hidden the enhancement effects (EXP-006 threat 1). A stronger but still small pose model tests (a) how much of
the front-end gap a bigger detector closes and (b) whether the light-gated enhancer findings depend on a weak
front end.

## 2. Method
- Front end: **YOLOv8s-pose** (Ultralytics release weights, sha256 logged), otherwise identical to EXP-003/006
  (conf-0.01 cache on MPS, imgsz 640, BoT-SORT gmc none, frozen STG-NF checkpoints, both scoring variants, c 0.25 +
  gate sweep).
- Configs: ShanghaiTech L0 E0, L2 {E0, E2}, L3 {E0, E1}; UBnormal L0 E0, L3 {E0, E1, E2}.
- CPU cost: detector ms/frame from a short VM run later (not part of the pass rules below).

## 3. Expectations

| # | Expectation | Pass rule (paper variant, c 0.25) |
|---|---|---|
| V1 | stronger front end helps | ShanghaiTech L0: AUC(8s) − AUC(8n) ≥ +3; UBnormal L0: ≥ +3 |
| V2 | ShanghaiTech enhancer effects keep their sign | L2: E2 − E0 > 0; L3: E1 − E0 > 0 |
| V3 | UBnormal deep-dark gamma with a stronger front end | L3: E1 − E0 > 0 (descriptive; EXP-006 UA2 failed with 8n) |
| V4 | coverage still tracks AUC | Spearman(label-free coverage, AUC) over all EXP-007 configs per dataset ≥ 0.7 |

## 4. Deviations policy
As EXP-003.
