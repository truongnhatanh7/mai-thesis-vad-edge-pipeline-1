# EXP-022 log (append-only)

### 2026-11-08 17:51 — implementation and run (Lane H)
- src/edgelab/analysis/exp022.py + tests/test_analysis_exp022.py (1 test; checks agreement with exp011.recall_by_bin on a toy cache). Reuses exp009 matcher (keypoint_box conf>0, detector_boxes conf>=0.25, IoU>=0.3). Reference track = (video, released person id).
- Sanity: 187,021 reference person-frames, 1,415 tracks; baseline found 74,063 = EXP-011 L0_E0 / EXP-009 D1.
- (a) recall <32 / 32-48 / 48-64 / 64-96 / 96-160 / >=160 / all. Baseline: 0.001 / 0.007 / 0.057 / 0.613 / 0.963 / 0.991 / 0.396. Fine-tuned (EXP-013): 0.823 / 0.912 / 0.965 / 0.984 / 0.992 / 0.992 / 0.947.
- (b) baseline: 112,958 missed person-frames; 57,086 (50.5%) in never-seen tracks (715 tracks), 55,872 (49.5%) in intermittent tracks (687). Rule (>= 30%) -> propose TSM. Fine-tuned: 9,954 missed; 1,481 (14.9%) never-seen, 8,473 (85.1%) intermittent.
- Descriptive extra (not the rule): share of missed frames in intermittent tracks matched in >= 50% of their frames: baseline 10.9%, fine-tuned 70.0%. So for the baseline most intermittent misses are in tracks matched only rarely.
- Outputs: results.md, miss_diagnostics.json.
