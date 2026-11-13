# EXP-014 log (append-only)

### 2026-11-07 18:26 — Lane C: start
- EXP-014 shares the EXP-013 fine-tuned weights and the 90/10 video split; code in pose/wise.py. Waiting on EXP-013 training.

### 2026-11-08 10:46 — Lead: naive WiSE collapse found; Amendment 1 (BN recalibration); chain restarted at the WiSE stage
- Held-out training videos: coverage 0.0 at α 0.25/0.5/0.75 (naive) vs 0.546 (α 0) / 0.913 (α 1); detcache logs show 0
  detections at conf ≥ 0.01. Endpoints reproduce the source models (α 1 = fine-tuned: 14 boxes, max conf 0.816 on one
  diagnostic frame; α 0 = original). The chain had chosen α* = 0.5 from all-zero ties and started evaluating it → stopped.
- Fix (plan Amendment 1): BatchNorm recalibration on 512 EXP-013 training images (seed 0). Check on 20 held-out *training*
  frames, people at conf ≥ 0.25: original 12, fine-tuned 20, naive α 0.5 **0**, α 0.5 + BN recal **20**.
- Naive weights/caches kept in artifacts/EXP-014/naive_collapsed and ~/thesis-data/exp014-heldout/naive_collapsed.
