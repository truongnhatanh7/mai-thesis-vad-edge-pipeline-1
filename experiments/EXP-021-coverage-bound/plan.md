# EXP-021 — An analytic link between person coverage and AUC
Frozen 2026-11-08 17:15. Frames with no scored person receive the most-normal score. Derive an upper bound on frame AUC as a function
of anomalous-frame coverage cA and normal-frame coverage cN (assuming perfect scoring of covered frames and ties broken
pessimistically/optimistically), and test it on all existing configurations (EXP-003…013, 016, 017; ~60 configs).
| # | Expectation | Pass rule |
|---|---|---|
| B1 | bound holds | measured AUC ≤ bound for every configuration |
| B2 | bound explains variation | Spearman(bound, AUC) ≥ 0.8 across configurations, both datasets |
