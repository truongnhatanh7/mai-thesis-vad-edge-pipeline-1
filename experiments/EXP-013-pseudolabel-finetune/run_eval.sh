#!/bin/bash
# Evaluate one detector (weights file) the EXP-003/006 way: detection caches (conf 0.01) on ShanghaiTech test L0/L2/L3 and
# UBnormal test L0, then replay at gate c = 0.25 -> frozen STG-NF -> rows in <root>/results.jsonl.
#   usage: run_eval.sh <tag> <weights.pt> <mlflow experiment>      e.g. run_eval.sh exp013 artifacts/EXP-013/yolov8n-pose-sht-ft.pt EXP-013
# Roots: ~/thesis-data/<tag>-sht and ~/thesis-data/<tag>-ub (resumable: finished caches are skipped).
set -u
TAG=$1; W=$(cd "$(dirname "$2")" && pwd)/$(basename "$2"); EXP=$3
cd /Users/nanh/Documents/RMIT_MscAI/thesis/experiments
export OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MLFLOW_DISABLE_AGENT_HINT=1
F=$HOME/thesis-data/shanghaitech/shanghaitech/testing/frames; V=$HOME/thesis-data/ubnormal/test_videos
SHT=$HOME/thesis-data/$TAG-sht; UB=$HOME/thesis-data/$TAG-ub
mkdir -p $SHT/cache $UB/cache
job() { root=$1; src=$2; lv=$3; s=$(date +%s)
  grep -q "^$lv E0 exit=0" $root/cache/jobs_done.txt 2>/dev/null && return
  uv run --extra pose python -m edgelab.pose.detcache --frames-root $src --level $lv --enhance E0 --weights $W --out $root/cache/${lv}_E0 --device mps --threads 1 > $root/cache/${lv}_E0.log 2>&1
  echo "$lv E0 exit=$? seconds=$(( $(date +%s) - s )) $(date +%H:%M:%S)" >> $root/cache/jobs_done.txt; }
job $UB $V L0 & job $SHT $F L0 & job $SHT $F L2 & job $SHT $F L3 &
wait
for r in $SHT $UB; do grep -q SWEEP_DONE $r/cache/jobs_done.txt || echo SWEEP_DONE >> $r/cache/jobs_done.txt; done
uv run --extra pose python -m edgelab.pose.gate_sweep --root $SHT --gates 0.25 --experiment $EXP > $SHT/gate_sweep.log 2>&1 &
uv run --extra pose python -m edgelab.pose.gate_sweep --root $UB --dataset UBnormal --gates 0.25 --experiment $EXP > $UB/gate_sweep.log 2>&1 &
wait
echo "eval $TAG done $(date +%H:%M:%S)"
