#!/bin/bash
# Quick check (about 30 s) that the CPU quota and the cpuset of vm_scopes.sh take effect. Run: sudo bash scope_check.sh
# Busy loops stand in for the pipeline (4 loops) and the co-load (2 loops); CPU seconds are read from cgroup cpu.stat.
set -u
. "$(dirname "$0")/vm_scopes.sh"
usage() { awk '/usage_usec/ {print $2}' "$1/cpu.stat"; }
busy() { sh -c 'while :; do :; done' & sh -c 'while :; do :; done' & wait; }

for B in B4 B2; do for O in O0 O1; do
  slice=$(cg_dir "$B"); setup_slice "$B"
  # co-load stand-in: 2 busy loops (O1: pinned to vCPU 0 by the scope)
  cpid=$(start_coload "$B" "$O" sh -c 'sh -c "while :; do :; done" & sh -c "while :; do :; done" & wait'); sleep 0.5
  # pipeline stand-in: 4 busy loops in a scope in the same slice
  (run_in_slice "$B" sh -c 'for i in 1 2 3 4; do sh -c "while :; do :; done" & done; wait' &) ; sleep 1
  u0=$(usage "$slice"); c0=$(usage "$slice/edgelab-coload-$$.scope"); sleep 5; u1=$(usage "$slice"); c1=$(usage "$slice/edgelab-coload-$$.scope")
  cpus=$(grep Cpus_allowed_list /proc/$cpid/status | cut -f2); ni=$(ps -o ni= -p "$cpid" | tr -d ' ')
  echo "$B $O: slice cpu.max='$(cat $slice/cpu.max)'  slice CPUs used = $(echo "scale=2; ($u1-$u0)/5000000" | bc)  co-load scope CPUs used = $(echo "scale=2; ($c1-$c0)/5000000" | bc)  co-load cpuset.effective='$(cat $slice/edgelab-coload-$$.scope/cpuset.cpus.effective 2>/dev/null)' Cpus_allowed(pid)=$cpus"
  kill "$cpid" 2>/dev/null; sleep 1; systemctl stop "edgelab-pipeline-$$.scope" "edgelab-coload-$$.scope" 2>/dev/null
done; done
systemctl stop edgelab-b2.slice edgelab-b4.slice 2>/dev/null
