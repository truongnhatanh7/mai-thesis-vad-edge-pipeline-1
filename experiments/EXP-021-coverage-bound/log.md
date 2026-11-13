# EXP-021 log (append-only)

### 2026-11-08 17:48 — derivation, collection, checks (Lane H)
- Wrote derivation.md (bound_half = cA + (1-cA)(1-cN)/2, bound_pess = cA, bound_loose = (1+cA)/2), src/edgelab/analysis/exp021.py, tests/test_analysis_exp021.py (5 tests).
- Collected 194 points (97 configs x paper/live) from exp003, exp006, exp007-sht/ub, exp012, exp013-sht/ub, exp016, exp017-sht/ub (c 0.25). Dropped: 16 exp003 no_segments placeholders (L4_E1/L4_E3 superseded by later rows), exp016 rows flagged invalid (incomplete MPS-crashed caches), exact re-runs (exp003 L3_E1 x3 identical -> last).
- Primary bound = half (plan left tie choice open; sklearn convention). B1 FAIL: 12/194 violations (all exp003: L4_E0-3, r320, X3_E1; low coverage), max excess 0.051; loose bound 2 violations. B2 PASS: Spearman 0.937 (SHT, n=160), 0.859 (UB, n=34).
- Post-hoc diagnosis (flagged as not in frozen plan): the temporal smoother spreads covered scores onto uncovered neighbours; on effective coverage (final score > file minimum) bound has 0 violations by construction but Spearman only 0.20 / 0.44.
- Outputs: results.md, points.json, scatter_bound_vs_auc.png.
