#!/bin/bash
# Unzip only the 211 UBnormal TEST videos (names in test_videos.txt) into one flat folder.
# Zip layout: Scene<N>/<video>.mp4 (plus <video>_annotations/ folders of PNGs, which we do not need).
# The zip is ZIP64 (16 GB); macOS Info-ZIP unzip 6.0 cannot read its central directory, so Python's zipfile is used.
set -e
cd "$(dirname "$0")"
ZIP=${1:-$HOME/thesis-data/ubnormal/UBnormal.zip}
OUT=${2:-$HOME/thesis-data/ubnormal/test_videos}
mkdir -p "$OUT"
python3 - "$ZIP" "$OUT" <<'PY'
import re, shutil, sys, zipfile
from pathlib import Path
zip_path, out = sys.argv[1], Path(sys.argv[2])
names = [l.strip() for l in open("test_videos.txt") if l.strip()]
with zipfile.ZipFile(zip_path) as z:
    for name in names:
        scene = re.match(r"(?:abnormal|normal)_scene_(\d+)_", name).group(1)
        with z.open(f"Scene{scene}/{name}.mp4") as src, open(out / f"{name}.mp4", "wb") as dst:
            shutil.copyfileobj(src, dst)
PY
echo "$(ls "$OUT"/*.mp4 | wc -l) videos in $OUT (expected 211)"
