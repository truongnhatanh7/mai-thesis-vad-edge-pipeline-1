#!/bin/bash
# EXP-020 step 1: download COCO 2017 (official URLs), idempotent/resumable. aria2c -c resumes partial files.
set -u
D=$HOME/thesis-data/coco
cd "$D" || exit 1
LOG=$D/download.log
say() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }
get() {  # name url
  local name=$1 url=$2
  if [ -f "$name.done" ]; then say "$name already done"; return; fi
  say "$name start"
  until aria2c -c -x 8 -s 8 -k 4M --retry-wait=10 --max-tries=0 -o "$name" "$url"; do say "$name aria2c failed, retry"; sleep 20; done
  unzip -tq "$name" >> "$LOG" 2>&1 || { say "$name unzip -t FAILED"; exit 1; }
  { stat -f '%z bytes' "$name"; shasum -a 256 "$name"; } > "$name.done"
  say "$name done: $(cat $name.done | tr '\n' ' ')"
}
get annotations_trainval2017.zip http://images.cocodataset.org/annotations/annotations_trainval2017.zip
get val2017.zip http://images.cocodataset.org/zips/val2017.zip
get train2017.zip http://images.cocodataset.org/zips/train2017.zip
if [ ! -f extract.done ]; then
  say "extract start"
  for z in annotations_trainval2017.zip val2017.zip train2017.zip; do unzip -qn "$z" || { say "unzip $z failed"; exit 1; }; done
  touch extract.done; say "extract done"
fi
say "ALL DONE"
