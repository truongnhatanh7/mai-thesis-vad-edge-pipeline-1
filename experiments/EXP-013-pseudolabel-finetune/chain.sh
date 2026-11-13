#!/bin/bash
# EXP-013 + EXP-014 chain. Waits for the VM-timing lane (lock file), then: alignment check -> fine-tune -> evaluate ->
# weight interpolation -> held-out coverage -> alpha* -> evaluate every alpha. Each stage leaves a marker in $M and is skipped on re-run.
set -u
cd /Users/nanh/Documents/RMIT_MscAI/thesis/experiments
export MLFLOW_DISABLE_AGENT_HINT=1
D=$HOME/thesis-data; M=$D/exp013-chain; mkdir -p $M artifacts/EXP-013 artifacts/EXP-014
LOCK=$D/locks/vm_timing_done; DATA=$D/exp013-data
FT=artifacts/EXP-013/yolov8n-pose-sht-ft.pt; ORIG=artifacts/EXP-001/models/yolov8n-pose.pt
say() { echo "$(date '+%F %H:%M:%S') $*"; }
stage() { name=$1; shift; [ -f $M/$name.done ] && return; say "stage $name start"; "$@" && touch $M/$name.done || { say "stage $name FAILED"; exit 1; }; say "stage $name done"; }

while [ ! -f $LOCK ]; do say "waiting for $LOCK"; sleep 60; done
say "lock present"
while [ ! -f $DATA/manifest.json ]; do say "waiting for dataset manifest"; sleep 20; done

align() { uv run --extra pose python -m edgelab.pose.align_check --videos-dir $D/shanghaitech/shanghaitech/training/videos \
  --pose-dir $D/stgnf/data/ShanghaiTech/pose/train --out artifacts/EXP-013/alignment_check.json; }
train() { uv run --extra pose python -m edgelab.pose.finetune --data $DATA/dataset.yaml --run-dir $D/exp013-train \
  > $D/exp013-train.log 2>&1; }
evalft() { EXP-013-pseudolabel-finetune/run_eval.sh exp013 $FT EXP-013; }
wise() { for a in 0.25 0.5 0.75; do
  uv run --extra pose python -m edgelab.pose.wise --original $ORIG --finetuned $FT --alpha $a --out artifacts/EXP-014/yolov8n-pose-wise-a$a.pt --calib-dir $DATA/images/train --calib-n 512 --seed 0 || return 1
  done; (cd artifacts/EXP-014 && shasum -a 256 yolov8n-pose-wise-a*.pt > wise_sha256.txt); }
heldout() { H=$D/exp014-heldout; mkdir -p $H
  VAL=$(python3 -c "import json;print(' '.join(json.load(open('$DATA/split.json'))['val']))")
  for a in 0 0.25 0.5 0.75 1; do
    case $a in 0) w=$ORIG;; 1) w=$FT;; *) w=artifacts/EXP-014/yolov8n-pose-wise-a$a.pt;; esac
    ( OMP_NUM_THREADS=1 uv run --extra pose python -m edgelab.pose.detcache --frames-root $D/shanghaitech/shanghaitech/training/videos \
        --videos $VAL --level L0 --enhance E0 --weights $w --out $H/cache/a$a --device mps --threads 1 > $H/a$a.log 2>&1 && \
      uv run --extra pose python -m edgelab.pose.segment_coverage --cache $H/cache/a$a --c 0.25 --out $H/a$a.json >> $H/a$a.log 2>&1 ) &
  done; wait
  uv run --extra pose python -m edgelab.analysis.exp014 --dir $H --out artifacts/EXP-014/alpha_selection.json; }
evalwise() { STAR=$(python3 -c "import json;print(json.load(open('artifacts/EXP-014/alpha_selection.json'))['alpha_star'])")
  for a in $STAR 0.25 0.5 0.75; do
    EXP-013-pseudolabel-finetune/run_eval.sh exp014-a$a artifacts/EXP-014/yolov8n-pose-wise-a$a.pt EXP-014 || return 1
  done; }

stage align align
stage train train
stage evalft evalft
stage wise wise
stage heldout heldout
stage evalwise evalwise
say "chain finished"
