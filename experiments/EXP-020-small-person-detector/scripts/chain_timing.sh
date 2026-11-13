#!/bin/bash
# EXP-020 final stages (tmux e20-timing): VM timing of the six NCNN FP16 detectors (P4), delivered AUC (d), report.
set -u
cd /Users/nanh/Documents/RMIT_MscAI/thesis/experiments || exit 1
export MLFLOW_DISABLE_AGENT_HINT=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
say() { echo "$(date '+%F %T') $*"; }
say "timing start"; uv run --extra pose python -m edgelab.pipeline.timing_exp020 --reps 3 || { say "timing FAILED"; exit 1; }
say "delivered start"; uv run --extra pose python -m edgelab.analysis.exp020 delivered || { say "delivered FAILED"; exit 1; }
say "report start"; uv run --extra pose python -m edgelab.analysis.exp020 report --out-dir artifacts/EXP-020 > /dev/null || { say "report FAILED"; exit 1; }
say "all done"
