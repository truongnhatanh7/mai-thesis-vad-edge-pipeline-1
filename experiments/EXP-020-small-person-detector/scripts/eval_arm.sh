#!/bin/bash
# EXP-020 evaluation of one arm (run by chain_eval.sh): MPS FP32 caches + gate sweep (run_eval.sh of EXP-013), NCNN FP16 export,
# NCNN FP16 640 detection cache on SHT L0 (for P4) + gate sweep. Resumable via $M/<stage>.done.
#   usage: eval_arm.sh <arm>
set -u
ARM=$1
cd /Users/nanh/Documents/RMIT_MscAI/thesis/experiments || exit 1
export MLFLOW_DISABLE_AGENT_HINT=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
D=$HOME/thesis-data; M=$D/exp020/chain
say() { echo "$(date '+%F %T') [$ARM] $*"; }
stage() { name=$1; shift; [ -f $M/$name.done ] && return; say "stage $name start"; "$@" && touch $M/$name.done || { say "stage $name FAILED"; return 1; }; say "stage $name done"; }
W=artifacts/EXP-020/$ARM.pt
NC=$D/vm-exchange/models/exp020/$ARM-640-384x640-ncnn
mps() { EXP-013-pseudolabel-finetune/run_eval.sh exp020-$ARM $W EXP-020; }
export_ncnn() { mkdir -p $D/vm-exchange/models/exp020 && uv run --extra pose python -m edgelab.pipeline.ncnn_pose --weights $W --h 384 --w 640 --out $NC; }
ncnn_cache() { R=$D/exp020-ncnn; mkdir -p $R/cache; s=$(date +%s)
  uv run --extra pose python -m edgelab.pose.detcache --frames-root $D/shanghaitech/shanghaitech/testing/frames --level L0 --enhance E0 \
    --out $R/cache/L0_E0-$ARM --detector $NC --ncnn-fp16 on --label ncnn16-640 --threads 3 >> $R/cache/L0_E0-$ARM.log 2>&1 || return 1
  echo "L0 E0-$ARM exit=0 seconds=$(( $(date +%s) - s )) $(date +%H:%M:%S)" >> $R/cache/jobs_done.txt; }
ncnn_sweep() { R=$D/exp020-ncnn; grep -q SWEEP_DONE $R/cache/jobs_done.txt || echo SWEEP_DONE >> $R/cache/jobs_done.txt
  uv run --extra pose python -m edgelab.pose.gate_sweep --root $R --gates 0.25 --experiment EXP-020 >> $R/gate_sweep.log 2>&1; }
parity() { uv run --extra pose python EXP-020-small-person-detector/scripts/ncnn_parity.py $ARM > $D/exp020/parity_$ARM.log 2>&1; }
stage ncnn_export_$ARM export_ncnn
stage ncnn_parity_$ARM parity
stage eval_mps_$ARM mps
stage ncnn_cache_$ARM ncnn_cache
stage ncnn_sweep_$ARM ncnn_sweep
