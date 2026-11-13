# EXP-010b log (append-only)

### 2026-11-07 18:26 — implementation; waiting for the VM-timing lock
- exp010.py: new configure(seed, out_root, mlflow_experiment, log_dir) and CLI options --seed (default 1) and --out-root (default ~/thesis-data/exp010); EXP-010 behaviour unchanged. New src/edgelab/analysis/exp010b.py (seed 2, root ~/thesis-data/exp010b, MLflow 'EXP-010b', S1/S2 per plan; S2 counts a cell only if both arms have >= 200 scorable windows (STG-NF windows, live row); degenerate cells listed separately and never counted as wins). tests/test_analysis_exp010b.py (2 tests).
- Coordination: tmux 'e10b' polls every 60 s for ~/thesis-data/locks/vm_timing_done (EXP-012 timing needs a quiet host) and only then starts the SHT and UB runs (script ~/thesis-data/exp010b-wait_and_run.sh). No CPU-heavy job of EXP-010b has run yet.
- Interpretation: S1/S2 counting uses FILL-HOLD vs SKIP, live variant, CI excluding 0 and diff > 0 (as in C1/C2); paired bootstrap 1000 resamples seed 0.

### 2026-11-07 18:49 — still waiting for the lock (interim status)
- ~/thesis-data/locks/vm_timing_done does not exist yet; EXP-012 timing is running (tmux e12-time; 25 files in results/exp012 at 18:49). tmux 'e10b' is armed and will start the SHT and UB runs by itself when the lock appears (logs ~/thesis-data/exp010b-run_{SHT,UB}.log, rows ~/thesis-data/exp010b/rows.jsonl). Nothing CPU-heavy of EXP-010b has run; the full pytest suite was also held back so as not to disturb the timing.
- After the runs: `uv run --extra pose python -m edgelab.analysis.exp010b analyse` (writes verdicts.json and results_tables.md; ~10 min of paired bootstraps for FILL-HOLD only), then copy to results.md with threats.

### 2026-11-07 20:28 — driver
- part SHT finished

### 2026-11-07 20:36 — driver
- part UB finished

### 2026-11-07 22:08 — Lead: replays finished (72 rows); analysis + results.md (S1 fail, S2 fail; direction replicates)
