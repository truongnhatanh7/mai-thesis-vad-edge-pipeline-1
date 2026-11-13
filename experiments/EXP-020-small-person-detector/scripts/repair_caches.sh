#!/bin/bash
# EXP-020 repair (tmux e20-repair): rebuild MPS detection caches whose job exited non-zero (GPU errors under concurrent
# MPS load), strictly one job at a time, after the e20-chain session has exited; then re-run the gate sweeps.
#   usage: repair_caches.sh <arm> [<arm> ...]
set -u
cd /Users/nanh/Documents/RMIT_MscAI/thesis/experiments || exit 1
export OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MLFLOW_DISABLE_AGENT_HINT=1
D=$HOME/thesis-data; F=$D/shanghaitech/shanghaitech/testing/frames; V=$D/ubnormal/test_videos
say() { echo "$(date '+%F %T') $*"; }
while tmux has-session -t e20-chain 2>/dev/null; do sleep 120; done
say "e20-chain gone -> repairing $*"
for arm in "$@"; do
  W=$PWD/artifacts/EXP-020/$arm.pt
  for spec in "ub $V L0" "sht $F L0" "sht $F L2" "sht $F L3"; do
    set -- $spec; ds=$1; src=$2; lv=$3; root=$D/exp020-$arm-$ds; J=$root/cache/jobs_done.txt
    grep -q "^$lv E0 exit=0" $J && continue
    say "[$arm] rebuild $ds $lv"; rm -rf $root/cache/${lv}_E0; s=$(date +%s)
    uv run --extra pose python -m edgelab.pose.detcache --frames-root $src --level $lv --enhance E0 --weights $W \
      --out $root/cache/${lv}_E0 --device mps --threads 1 > $root/cache/${lv}_E0.log 2>&1
    rc=$?; sed -i '' "/^$lv E0 exit=/d" $J; sed -i '' "/^SWEEP_DONE/d" $J
    echo "$lv E0 exit=$rc seconds=$(( $(date +%s) - s )) $(date +%H:%M:%S)" >> $J; echo SWEEP_DONE >> $J
    say "[$arm] $ds $lv exit=$rc"
  done
  uv run --extra pose python -m edgelab.pose.gate_sweep --root $D/exp020-$arm-sht --gates 0.25 --experiment EXP-020 > $D/exp020-$arm-sht/gate_sweep.log 2>&1
  uv run --extra pose python -m edgelab.pose.gate_sweep --root $D/exp020-$arm-ub --dataset UBnormal --gates 0.25 --experiment EXP-020 > $D/exp020-$arm-ub/gate_sweep.log 2>&1
  say "[$arm] sweeps done"
done
say "repair finished"
