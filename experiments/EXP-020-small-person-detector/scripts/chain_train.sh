#!/bin/bash
# EXP-020 training chain (tmux e20-chain). Resumable: each stage leaves $M/<stage>.done; Ultralytics `resume` continues an
# interrupted run (train_coco.py). Waits for the e13-chain tmux session (EXP-014 evaluation uses MPS) to disappear first.
#   usage: chain_train.sh <epochs>
set -u
EPOCHS=$1
cd /Users/nanh/Documents/RMIT_MscAI/thesis/experiments || exit 1
export MLFLOW_DISABLE_AGENT_HINT=1
D=$HOME/thesis-data; M=$D/exp020/chain; mkdir -p $M $D/exp020/train
say() { echo "$(date '+%F %T') $*"; }
stage() { name=$1; shift; [ -f $M/$name.done ] && return; say "stage $name start"; "$@" && touch $M/$name.done || { say "stage $name FAILED"; exit 1; }; say "stage $name done"; }
while tmux has-session -t e13-chain 2>/dev/null; do say "e13-chain still exists; waiting"; sleep 300; done
say "e13-chain not present"
while [ ! -f $D/coco/extract.done ]; do say "waiting for COCO extraction"; sleep 60; done
train() { uv run --extra pose python -m edgelab.pose.train_coco --arm $1 --epochs $EPOCHS >> $D/exp020/train_$1.log 2>&1; }
for arm in M00 ML0 M0P MLP; do stage train_$arm train $arm; done
say "training chain finished"
