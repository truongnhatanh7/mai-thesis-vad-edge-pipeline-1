#!/bin/bash
# One EXP-016 timing run inside the VM with budget B and co-load placement O (same cgroup set-up as EXP-004's cell_in_vm.sh;
# only the Python module differs).   sudo bash cell_in_vm.sh <B4|B2> <O0|O1> <topdown_timing args...>
# Prints one line "COLOAD_CPU_S=<cpu seconds the co-load used> WALL_S=<seconds>" at the end.
set -u
here=$(dirname "$0")
. "$here/../../EXP-004-os-deadlines/scripts/vm_scopes.sh"
B=$1; O=$2; shift 2
EXPERIMENTS=/Users/nanh/Documents/RMIT_MscAI/thesis/experiments
PY=/home/nanh.linux/edgelab-vm/.venv/bin/python

setup_slice "$B" || exit 2
coload_pid=$(start_coload "$B" "$O")
sleep 2  # let the encoder reach its steady rate
kill -0 "$coload_pid" 2>/dev/null || { echo "co-load did not start"; exit 3; }
coload_scope=$(cg_dir "$B")/edgelab-coload-$$.scope
cpu0=$(awk '/usage_usec/ {print $2}' "$coload_scope/cpu.stat"); t0=$(date +%s)

PYTHONPATH=$EXPERIMENTS/src EDGELAB_MACHINE=vm run_in_slice "$B" "$PY" -m edgelab.pipeline.topdown_timing "$@"
rc=$?

cpu1=$(awk '/usage_usec/ {print $2}' "$coload_scope/cpu.stat"); t1=$(date +%s)
kill "$coload_pid" 2>/dev/null
systemctl stop "edgelab-coload-$$.scope" 2>/dev/null
echo "COLOAD_CPU_S=$(echo "scale=1; ($cpu1-$cpu0)/1000000" | bc) WALL_S=$((t1-t0))"
exit $rc
