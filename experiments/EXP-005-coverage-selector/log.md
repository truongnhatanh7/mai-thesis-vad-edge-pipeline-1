# EXP-005 log (append-only)

### 2026-11-06 06:09 — Lead: plan frozen, code written, caches queued
- `plan.md` frozen before any EXP-005 cache. Held-out levels X1/X2/X3 (α 0.2/0.05/0.02) added to
  `darken.LEVELS` (test updated, 11 passed).
- `src/edgelab/analysis/exp005.py` + `tests/test_analysis_exp005.py` (3 passed): selectors S (label-free all-frame
  coverage) and S24 (only tracks with ≥ 48 consecutive frames), oracle, fixed policies, F1–F4 verdicts. Smoke on
  an EXP-003 cache (L3_E1): long-track coverage 0.445.
- Caches: `~/thesis-data/exp003/sweep_x.sh` (9 jobs, X1–X3 × E0–E2 @640, MPS) chained after the EXP-004 final
  analysis (tmux `chain-x`), then gate_sweep scoring.

### 2026-11-06 09:37 — Lead: caches scored, analysis run, results.md written
- 9 caches (X1–X3 × E0–E2) built 07:30–09:2x on MPS, scored by gate_sweep (`gate_sweep_x.log`).
- `edgelab.analysis.exp005` → `artifacts/EXP-005/selector_results.json`. F1 pass, F2 pass (tie with always-CLAHE),
  F3 pass (ρ 0.917 / 0.867), F4 trivial. Deep-dark failure (X3: gamma best with lowest coverage) replicates L3.
