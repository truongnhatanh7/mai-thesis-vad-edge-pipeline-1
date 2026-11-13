#!/bin/bash
# Run in tmux once the zip download has finished: checks the zip, extracts the 211 test videos, builds strata.csv,
# checks frame counts and letterbox shapes. Everything it prints goes to ~/thesis-data/ubnormal/after_download.log.
cd "$(dirname "$0")/.."   # experiments/
D=$HOME/thesis-data/ubnormal
while ! grep -q ARIA_OK $D/aria.log 2>/dev/null || [ -e $D/UBnormal.zip.aria2 ]; do sleep 30; done
echo "download finished $(date)"
ls -l $D/UBnormal.zip
echo "sha256 (zip):  $(shasum -a 256 $D/UBnormal.zip | cut -d' ' -f1)"
echo "sha256 (HF LFS oid): $(cut -d' ' -f2 $D/hf_lfs_oid.txt)"
unzip -l $D/UBnormal.zip > $D/zip_listing.txt && tail -1 $D/zip_listing.txt
EXP-006-ubnormal-replication/extract_test_videos.sh
uv run --extra pose python -m edgelab.analysis.exp006_strata --videos $D/test_videos --out EXP-006-ubnormal-replication/strata.csv
uv run --extra pose python - <<'PY'
import csv
from ultralytics import YOLO
from edgelab.pose.detcache import WEIGHTS, letterbox_shape
model = YOLO(str(WEIGHTS))
sizes = {}
for row in csv.DictReader(open("EXP-006-ubnormal-replication/strata.csv")):
    sizes.setdefault((int(row["width"]), int(row["height"])), []).append(row["video"])
for (w, h), videos in sorted(sizes.items()):
    print(f"{w}x{h}: {len(videos)} videos;", "; ".join(f"imgsz {z}: {letterbox_shape(model, w, h, z)}" for z in (480, 640, 960)))
PY
echo "AFTER_DONE $(date)"
