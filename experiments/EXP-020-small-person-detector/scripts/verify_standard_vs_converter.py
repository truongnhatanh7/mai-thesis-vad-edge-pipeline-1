"""EXP-020 step 2 check: our 'standard' label files == files written by Ultralytics' own convert_coco (use_keypoints=True).

Run: uv run --extra pose python EXP-020-small-person-detector/scripts/verify_standard_vs_converter.py   (from experiments/)
Compares, per split, every person image: byte-equal content. Also lists how many files the converter writes that we
do not (images with only crowd / zero-area persons) and vice versa (must be 0).
"""
import json
import shutil
import sys
from pathlib import Path

from ultralytics.data.converter import convert_coco

coco = Path.home() / "thesis-data/coco/annotations"
ours = Path.home() / "thesis-data/exp020/data/standard/labels"
work = Path.home() / "thesis-data/exp020/converter-check"
if work.exists():
    shutil.rmtree(work)
(work / "ann").mkdir(parents=True)
for split in ("train2017", "val2017"):
    (work / "ann" / f"person_keypoints_{split}.json").symlink_to(coco / f"person_keypoints_{split}.json")
convert_coco(str(work / "ann"), str(work / "out"), use_keypoints=True, cls91to80=False)
report = {}
for split in ("train2017", "val2017"):
    theirs = work / "out" / "labels" / f"person_keypoints_{split}"
    t = {p.name for p in theirs.glob("*.txt")}
    o = {p.name for p in (ours / split).glob("*.txt")}
    differ = [n for n in sorted(t & o) if (theirs / n).read_text() != (ours / split / n).read_text()]
    extra_theirs = sorted(t - o)
    extra_nonempty = [n for n in extra_theirs if (theirs / n).read_text()]
    report[split] = {"converter_files": len(t), "our_files": len(o), "common": len(t & o), "content_differs": len(differ),
                     "converter_only_files": len(extra_theirs), "converter_only_nonempty": len(extra_nonempty),
                     "ours_only_files": len(o - t), "first_differences": differ[:5]}
print(json.dumps(report, indent=1))
(Path.home() / "thesis-data/exp020/converter_check.json").write_text(json.dumps(report, indent=1))
sys.exit(0 if all(r["content_differs"] == 0 and r["ours_only_files"] == 0 and r["converter_only_nonempty"] == 0 for r in report.values()) else 1)
