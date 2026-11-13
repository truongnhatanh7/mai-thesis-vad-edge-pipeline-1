#!/bin/bash
# EXP-020 evaluation waiter (tmux e20-eval): as soon as an arm's training is done, evaluate it (eval_arm.sh).
set -u
D=$HOME/thesis-data; S=/Users/nanh/Documents/RMIT_MscAI/thesis/experiments/EXP-020-small-person-detector/scripts
say() { echo "$(date '+%F %T') $*"; }
for arm in M00 ML0 M0P MLP; do
  while [ ! -f $D/exp020/train/$arm/train.done ]; do sleep 120; done
  say "$arm trained -> evaluating"
  bash $S/eval_arm.sh $arm || say "$arm evaluation had a failed stage (re-run to resume)"
done
say "evaluation chain finished"
