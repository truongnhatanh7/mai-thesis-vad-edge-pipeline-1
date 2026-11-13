#!/bin/bash
# EXP-015 full run. Waits for the VM-timing lock, then: re-estimate keypoints on the released boxes (3 models) and score all
# with the frozen STG-NF checkpoint (+ the released poses as the reference). Resumable: pose files and score dirs that exist are skipped.
#   tmux new -d -s e15 'bash run.sh'
set -u
export MLFLOW_DISABLE_AGENT_HINT=1
HERE=$(cd "$(dirname "$0")" && pwd)
EXP=$HERE/..
DATA=$HOME/thesis-data/exp015
LOCK=$HOME/thesis-data/locks/vm_timing_done
cd "$EXP" || exit 1
log() { printf '\n### %s — %s\n- %s\n' "$(date '+%Y-%m-%d %H:%M')" "$1" "$2" >> "$HERE/log.md"; }

while [ ! -e "$LOCK" ]; do sleep 60; done
log "driver" "lock $LOCK present, heavy run starts"

for model in rtmpose-s rtmpose-t yolov8n-pose; do
  uv run --extra pose python -m edgelab.pose.oracle_rtmpose --model "$model" --out "$DATA/poses/$model" --threads 4 \
      >> "$DATA/pose_$model.log" 2>&1
  n=$(ls "$DATA/poses/$model" | grep -c tracked_person.json)
  log "poses $model" "$n pose files in $DATA/poses/$model (107 expected); per-video stats in $DATA/pose_$model.log"
done

for arm in released rtmpose-s rtmpose-t yolov8n-pose; do
  uv run --extra pose python -m edgelab.analysis.exp015 score --name "$arm" --threads 4 >> "$DATA/score_$arm.log" 2>&1
  log "scored $arm" "$(tail -2 "$DATA/score_$arm.log" | tr '\n' ' ')"
done
uv run --extra pose python -m edgelab.analysis.exp015 report > "$HERE/results_table.md" 2>> "$DATA/report.log"
log "driver" "all scored, table in results_table.md; EXP015_DONE"
touch "$DATA/EXP015_DONE"
