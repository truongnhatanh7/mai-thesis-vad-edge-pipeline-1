# EXP-008 log (append-only)

### 2026-11-07 09:18 — Lead: plan frozen 09:10, driver, 92 replays, verdicts
- `src/edgelab/analysis/exp008.py` (replays per group with traces × s; analysis of A1–A3). 4 tmux sessions
  (e8-SHT-L0/L2/L3, e8-UB-L0), ≈ 10 s per replay, all 92 rows written, no errors.
- Verdicts (verdicts.json): A1a pass, A1b pass (τ 0.16 at s 4; 0.72 at s 1), A2 fail (non-monotone at s 8; 480 > 640
  at s 8 in 3/4), A3 fail (MAE 2.62, ρ 0.61). results.md written.
