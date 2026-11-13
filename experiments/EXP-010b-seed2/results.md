# EXP-010b results — EXP-010 with keep-mask seed 2

> Same pipeline as EXP-010 (6 groups × s 3/6 × SKIP / FILL-HOLD / FILL-NORMAL, G 8, r 2.507), keep-mask seed 2; live
> variant for the tests; paired bootstrap over videos (1000). Full tables: `results_tables.md`; raw:
> `~/thesis-data/exp010b/`. Analysis run by the lead after the code lane was lost in the Cursor crash. Sign-off: pending.

## Verdicts
| # | Rule (live) | Result | Verdict |
|---|---|---|---|
| S1 | FILL-HOLD − SKIP > 0, CI excl. 0, ≥ 4 of 6 groups, s = 3 | 3 of 6 (SHT-L2-8n, SHT-L0-8s, UB-L0-8s) | **fail** |
| S2 | same at s = 6, cells with ≥ 200 windows in both arms | 3 of 5 non-degenerate (SHT-L0/L2/L3-8n); SHT-L0-8s degenerate (24 / 9 windows) | **fail** |

## FILL-HOLD − SKIP, live AUC [95% CI]
| Group | s = 3 | s = 6 |
|---|---|---|
| SHT-L0-8n | +4.34 [−0.38, +9.14] | **+7.63 [+2.21, +13.32]** |
| SHT-L2-8n | **+5.68 [+0.78, +10.62]** | **+6.95 [+1.55, +12.45]** |
| SHT-L3-8n | +4.40 [−0.80, +9.71] | **+5.25 [+0.12, +10.67]** |
| UB-L0-8n | +4.53 [−0.03, +8.53] | +3.67 [−0.42, +7.76] |
| SHT-L0-8s | **+6.00 [+0.48, +11.79]** | degenerate (−2.00; 24 vs 9 windows, both at chance) |
| UB-L0-8s | **+8.63 [+4.61, +12.65]** | +3.81 [−0.47, +8.33] |

## Reading (lead)
- **Direction and size replicate:** all 11 non-degenerate cells favour filling, +3.7 to +8.6 AUC (EXP-010 seed 1:
  +2.2 to +8.3). **Per-cell significance does not:** 6 of 11 cells are significant (EXP-010: 9 of 11), and three
  s = 3 cells miss by a CI lower bound of −0.03 to −0.80.
- Consequence for the paper: EXP-010's formal pass is **not robust to the random choice of dropped frames**. The
  claim should be stated as a consistent effect across two independent mask draws (22 non-degenerate cells, all
  positive, ≈ +3.7 to +8.6 AUC), with per-cell CIs reported, not as "significant in ≥ 4 of 6 groups".
- Per-cell CIs are wide because the bootstrap is over only 107 / 211 test videos; a pre-registered test pooled over
  seeds was not planned and is not computed here (it would be post hoc).
