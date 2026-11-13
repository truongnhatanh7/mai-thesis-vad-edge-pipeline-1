#!/bin/bash
# Host side. Launch (or resume) the EXP-004 timing matrix; safe to run again: finished (cell, rep) pairs are skipped.
#   tmux new-session -d -s exp004-matrix 'bash <this file> <G-xfit c> [--dry-run]'
# Needs the VM edge-os (started if stopped) and the host idle (the runner records host load per run).
set -u
cd "$(dirname "$0")/../.." || exit 1   # .../experiments
limactl ls --format '{{.Name}} {{.Status}}' | grep -q '^edge-os Running' || limactl start edge-os --tty=false
mkdir -p /Users/nanh/thesis-data/vm-exchange/results/matrix
uv run --extra pose python -m edgelab.pipeline.timing_matrix --gxfit-c "${1:?usage: run_matrix.sh <G-xfit c> [--dry-run]}" "${@:2}" \
  >> /Users/nanh/thesis-data/vm-exchange/results/matrix/runner.log 2>&1
echo "runner exit $?" >> /Users/nanh/thesis-data/vm-exchange/results/matrix/runner.log
