#!/bin/bash
# EXP-016 full run (tmux). Waits for the VM-timing lock; MPS/CPU parity check; 4 person-box caches (L0, L2 x 640, 960);
# waits for EXP-015 (pose model choice); 8 configs -> poses -> STG-NF -> rows; table. VM timing (T3) is a separate step (run_timing.sh).
#   tmux new -d -s e16 'bash run.sh'
set -u
export MLFLOW_DISABLE_AGENT_HINT=1
HERE=$(cd "$(dirname "$0")" && pwd)
EXP=$HERE/..
DATA=$HOME/thesis-data/exp016
FRAMES=$HOME/thesis-data/shanghaitech/shanghaitech/testing/frames
LOCK=$HOME/thesis-data/locks/vm_timing_done
cd "$EXP" || exit 1
mkdir -p "$DATA"
log() { printf '\n### %s — %s\n- %s\n' "$(date '+%Y-%m-%d %H:%M')" "$1" "$2" >> "$HERE/log.md"; }

while [ ! -e "$LOCK" ]; do sleep 60; done
log "driver" "lock present, heavy run starts"

if uv run --extra pose python -m edgelab.pose.topdown parity --frames-root "$FRAMES" --videos 01_0014 06_0144 > "$DATA/parity.json" 2> "$DATA/parity.err"; then
  DEVICE=mps; log "parity" "MPS vs CPU rule passed (plan 4.3: max conf diff <= 0.01, equal counts on >= 99% frames): $(cat "$DATA/parity.json"); caches on mps"
else
  DEVICE=cpu; log "parity" "MPS vs CPU rule NOT passed or failed to run: $(cat "$DATA/parity.json") ; caches on cpu"
fi

for level in L0 L2; do
  for sz in 640 960; do
    uv run --extra pose python -m edgelab.pose.topdown cache --frames-root "$FRAMES" --level $level --imgsz $sz \
        --out "$DATA/cache/${level}_n$sz" --device $DEVICE >> "$DATA/cache_${level}_n$sz.log" 2>&1
    log "cache $level n$sz" "$(ls "$DATA/cache/${level}_n$sz" | grep -c '^[0-9].*npz$') npz files (107 expected, device $DEVICE)"
  done
done

while [ ! -e "$HOME/thesis-data/exp015/EXP015_DONE" ]; do sleep 60; done
POSE=$(python3 -c "import json;print(json.load(open('$EXP/EXP-015-oracle-rtmpose/verdicts.json'))['exp016_pose_model'])")
log "pose model" "from EXP-015 verdicts.json by the plan's rule: $POSE"

uv run --extra pose python -m edgelab.analysis.exp016 run --pose-model "$POSE" --levels L0 L2 --threads 4 >> "$DATA/run.log" 2>&1
log "driver" "8 configs scored; rows in $DATA/rows.jsonl; $(tail -1 "$DATA/run.log")"
uv run --extra pose python -m edgelab.analysis.exp016 report --pose-model "$POSE" > "$HERE/results_table.md" 2>> "$DATA/report.log"
echo "$POSE" > "$DATA/POSE_MODEL"
touch "$DATA/EXP016_QUALITY_DONE"
log "driver" "table in results_table.md; EXP016_QUALITY_DONE (VM timing next)"
