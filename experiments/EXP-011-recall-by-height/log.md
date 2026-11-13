# EXP-011 log (append-only)

### 2026-11-07 18:25 — implementation and run start
- src/edgelab/analysis/exp011.py + tests/test_analysis_exp011.py (2 tests). Reuses exp009's matcher (keypoint_box conf>0, iou_with_boxes, detector_boxes conf>=0.25, IoU>=0.3). Bins [<32, 32-48, 48-64, 64-96, 96-160, >=160] px by reference box height (y2-y1). Single-threaded (OMP/MKL threads 1) tmux session e11; 6 caches: L0_E0-r480, L0_E0 (640), -r800, -r960, L2_E0, L3_E0.
- Interpretation choices where plan is silent: R1 pools persons of the 2 bins vs the 2 bins; R2 = absolute recall gain; R3 = relative drop as written (bins with zero L0 recall skipped).

### 2026-11-07 18:25 — EXP-011 finished (~2 min single-threaded)
- Persons 187,021 (same as the D1 total). Recall at 640 L0 by bin (<32, 32-48, 48-64, 64-96, 96-160, >=160): 0.001, 0.007, 0.057, 0.613, 0.963, 0.991; all 0.396 (= EXP-009 D1 74,063 kept).
- R1 PASS (0.004 vs 0.969). R2 FAIL under absolute-gain reading (largest gain 64-96: +0.213; <64 bins +0.001/+0.006/+0.126), would pass under relative reading (48-64: 3.2x) — post-hoc note in results.md. R3 PASS (largest relative drop in <32: 0.72).
- Threats recorded in results.md (keypoint-box vs YOLO-box IoU bias for small people, AlphaPose reference). results: EXP-011-recall-by-height/{results.md, recall_by_height.json, recall_by_height.png}.
