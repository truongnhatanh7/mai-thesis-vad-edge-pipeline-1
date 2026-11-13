#!/bin/bash
# EXP-012 step 4a: detection caches of the fast variants on the ShanghaiTech test frames (host CPU, ORT / NCNN).
#   bash run_caches.sh <jobs in parallel> "<level> <tag> <detector path> <ncnn-fp16 on|off>" ...
# Output: ~/thesis-data/exp012/cache/<level>_E0-<tag>/ and one line per job in cache/jobs_done.txt (format gate_sweep reads).
cd /Users/nanh/Documents/RMIT_MscAI/thesis/experiments || exit 1
export MLFLOW_DISABLE_AGENT_HINT=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
F=$HOME/thesis-data/shanghaitech/shanghaitech/testing/frames
OUT=$HOME/thesis-data/exp012/cache
mkdir -p "$OUT"
job() { lv=$1; tag=$2; det=$3; fp16=$4; s=$(date +%s)
  [ -f "$OUT/jobs_done.txt" ] && grep -q "^$lv E0-$tag exit=0" "$OUT/jobs_done.txt" && return 0
  uv run --extra pose python -m edgelab.pose.detcache --frames-root "$F" --level "$lv" --enhance E0 --out "$OUT/${lv}_E0-${tag}" \
     --detector "$det" --ncnn-fp16 "$fp16" --label "$tag" --threads 3 >> "$OUT/${lv}_E0-${tag}.log" 2>&1
  echo "$lv E0-$tag exit=$? seconds=$(( $(date +%s) - s )) $(date +%H:%M:%S)" >> "$OUT/jobs_done.txt"; }
export -f job; export F OUT
parallel=$1; shift
printf '%s\n' "$@" | xargs -P "$parallel" -L 1 bash -c 'job "$0" "$1" "$2" "$3"'
