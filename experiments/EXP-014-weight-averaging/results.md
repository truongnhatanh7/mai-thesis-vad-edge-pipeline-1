# EXP-014 results — WiSE-FT weight averaging (original ↔ EXP-013 fine-tuned), with BatchNorm recalibration (Amendment 1)

> α* chosen without test data: highest label-free coverage on held-out *training* videos → **α* = 0.75**
> (`artifacts/EXP-014/alpha_selection.json`). Evaluation as EXP-013 (c 0.25, frozen STG-NF). Rows:
> `~/thesis-data/exp014-a{0.25,0.5,0.75}-{sht,ub}/results.jsonl`. Naive (non-recalibrated) blends output nothing
> (Amendment 1; kept in `artifacts/EXP-014/naive_collapsed`). Written by the lead.

## Verdicts (paper variant)
| # | Rule | Measured | Verdict |
|---|---|---|---|
| W1 | AUC(α*) ≥ AUC(fine-tuned) − 0.3 on SHT L0 | 77.60 vs 80.28 (−2.68) | **fail** |
| W2 | AUC(α*) ≥ AUC(fine-tuned) + 1.0 on SHT L3 **or** UBnormal L0 | UBnormal **55.67 vs 51.63 (+4.04)**; L3 52.43 vs 58.51 | **pass** (via UBnormal) |

## All blends (paper AUC; anomalous coverage)
| α | SHT L0 | SHT L2 | SHT L3 | UBnormal L0 |
|---|---|---|---|---|
| 0 (original) | 68.53 (0.674) | 67.05 | 60.43 | 55.76 |
| 0.25 | 72.35 (0.890) | 60.48 | 51.14 (0.119) | 56.25 |
| 0.5 | 69.46 (0.793) | 66.56 | 50.02 (0.145) | 48.69 |
| **0.75 (α*)** | **77.60 (0.950)** | **72.75** | 52.43 (0.221) | **55.67** |
| 1 (fine-tuned) | 80.28 (0.952) | 76.09 | 58.51 (0.416) | 51.63 |

## Reading (lead)
Weight averaging is a **trade-off, not a free lunch**: α* keeps most of the in-site gain (+9.1 over the original on
ShanghaiTech clean) while undoing the fine-tune's out-of-site damage (UBnormal back to the original's level), but it loses
2.7 points in-site versus the pure fine-tune and **collapses in deep dark** (L3 coverage 0.22). Non-monotonic in α (0.5
worst on UBnormal). Requires BatchNorm recalibration to work at all. Not a method contribution for the paper; reported as a
robustness trade-off.
