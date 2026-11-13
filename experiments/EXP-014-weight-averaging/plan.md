# EXP-014 — Weight-space averaging (WiSE-FT) between the original and the EXP-013 fine-tuned detector
Frozen 2026-11-07 18:30 before any run. θ(α) = (1−α)·θ_original + α·θ_finetuned, α ∈ {0.25, 0.5, 0.75} (+ α 0 / 1 = the two ends).
α* chosen **without test data**: highest label-free all-frame coverage on the 10% held-out *training* videos of EXP-013.
Evaluation as EXP-013 (SHT L0/L2/L3, UBnormal L0).
| # | Expectation | Pass rule |
|---|---|---|
| W1 | no in-domain loss | AUC(α*) ≥ AUC(fine-tuned) − 0.3 on SHT L0 |
| W2 | robustness gain under shift | AUC(α*) ≥ AUC(fine-tuned) + 1.0 on SHT L3 **or** UBnormal L0 |
If W2 fails, weight-space averaging is reported as not useful here and left out of the paper's method.

## Amendment 1 (frozen 2026-11-08 10:44, before any EXP-014 test-set result)

Finding that forces it: the naively interpolated models (α 0.25 / 0.5 / 0.75, all floating tensors incl. BatchNorm
running mean/variance) produce **zero detections** (held-out training videos: coverage 0.0 at every interior α, vs 0.546
at α 0 and 0.913 at α 1; one diagnostic frame: 0 boxes at conf ≥ 0.01). Endpoints reproduce the two source models
exactly, so the saving path is correct; the collapse is the known BatchNorm-statistics mismatch of weight averaging.
- **Change:** after interpolation, **recalibrate BatchNorm running statistics** (reset, cumulative average) by forward
  passes in train mode over 512 images from the EXP-013 **training** split (no labels used); then α* is chosen as
  planned (highest label-free coverage on the held-out training videos among α ∈ {0.25, 0.5, 0.75}).
- The naive (non-recalibrated) collapse is reported as a result in its own right.
- W1/W2 pass rules unchanged.
