# EXP-006 walkthrough — the UBnormal path

Same pipeline as `EXP-003-lowlight-gates/WALKTHROUGH.md` (cache → replay → STG-NF → AUC). Only what differs for
UBnormal is listed here. Every function keeps its ShanghaiTech default, so EXP-001/003/004/005 code and numbers are unchanged.

| What differs | ShanghaiTech | UBnormal | Code | Guarded by |
|---|---|---|---|---|
| Input | folder of JPG frames per video | one `<video>.mp4` per video (`Scene<N>/` folders in the zip; `extract_test_videos.sh` flattens the 211 test videos) | `detcache.py:list_videos`, `find_source`, `cache_video` | `tests/test_pose_ubnormal.py` (folder + mp4 listing; mp4 cached under its stem, one row per decoded frame) |
| Test videos | 107 GT files | 211 names in `test_videos.txt` (= STG-NF's released `data/UBnormal/pose/test`) | `ubnormal.py:load_test_list` | test: 211 unique names, all parse |
| Video name → STG-NF ids | `01_0014` → scene 1, clip 14 | `abnormal_scene_11_scenario_3_fog` → scene `11`, clip `abnormal__3_fog` (STG-NF's own regex) | `ubnormal.py:parse_video_name` | test vs STG-NF's two regexes (pose file and GT file) |
| Pose file name | `<video>_<tag>_tracked_person.json` | `<video>_alphapose_<tag>_tracked_person.json` — STG-NF's regex needs the literal `_alphapose_`; it is a file-name token only (the pose source is in the MLflow config) | `replay.py:source_tag(dataset=)`, `ubnormal.py:POSE_FILE_TOKEN`, `video_of_pose_file` | test: replay output parses to the same scene/clip as the video name |
| Segment length | 24 | **16** (`--seg_len 16`, STG-NF README; the released checkpoint needs it) → paper variant scores frame start+8, live variant start+15 | `stgnf_eval.py:SEG_LENS`, `score_frame_offsets` | test (12/23 vs 8/15); R0 loads the checkpoint |
| Ground truth | `01_0014.npy`, **1 = anomalous** | `<video>_tracks.txt` (a NumPy file with a .txt name), **1 = normal**; we flip it so 1 = anomalous like everywhere else | `ubnormal.py:load_gt` | test: flip |
| Frame scores | `stgnf_eval.clip_normality` | the same function with `dataset="UBnormal"`; segment metadata has a string clip id, so ids are compared as strings | `stgnf_eval.py:clip_normality`, `assemble_clip_scores` | test: toy example by hand, and equals STG-NF's own `get_dataset_scores` for UBnormal |
| Scoring, sweep | `score_dir`, `gate_sweep` | `--dataset UBnormal --experiment EXP-006` (checkpoint and GT default to UBnormal's) | `score_dir.py:DATASET_DEFAULTS`, `gate_sweep.py` | smoke run (log.md) |
| Light strata | — | per video: mean luma (1 frame/s, EXP-000 method) + name → `strata.csv` | `analysis/exp006_strata.py` | tests: rule, counts, luma equals EXP-000's `lighting_stats` |

## Commands (from `experiments/`)

```
# R0 gate (released poses + released checkpoint) -> MLflow EXP-006
uv run --extra pose python -m edgelab.pose.score_dir --dataset UBnormal --experiment EXP-006 \
  --pose-dir third_party/STG-NF/data/UBnormal/pose/test --out-dir artifacts/EXP-006/R0 --level none --enhance released --tracker alphapose
# STG-NF's own number
cd third_party/STG-NF && uv run --project ../.. --extra pose python train_eval.py --dataset UBnormal --seg_len 16 \
  --checkpoint checkpoints/UBnormal_unsupervised_71_8.tar --device cpu --num_workers 0

# test videos, strata
EXP-006-ubnormal-replication/extract_test_videos.sh
uv run --extra pose python -m edgelab.analysis.exp006_strata --videos ~/thesis-data/ubnormal/test_videos --out EXP-006-ubnormal-replication/strata.csv

# one cache config over all 211 test videos (resumable), then replay and score
uv run --extra pose python -m edgelab.pose.detcache --frames-root ~/thesis-data/ubnormal/test_videos --level L0 --enhance E0 --imgsz 640 --device mps --out ~/thesis-data/exp006/cache/L0_E0
uv run --extra pose python -m edgelab.pose.replay --cache ~/thesis-data/exp006/cache/L0_E0 --c 0.25 --out <pose_dir> --dataset UBnormal
uv run --extra pose python -m edgelab.pose.score_dir --dataset UBnormal --experiment EXP-006 --pose-dir <pose_dir> --out-dir <scores_dir> --level L0 --enhance E0 --c 0.25 --tracker botsort
# gate sweep over finished caches
uv run --extra pose python -m edgelab.pose.gate_sweep --root ~/thesis-data/exp006 --dataset UBnormal --experiment EXP-006 [--wait]
```

`--level L0` = no darkening (identity). Cache directory names are `<level>_<enhance>` (gate_sweep splits on the last `_`),
so resolution variants need a distinct level label in the directory name, e.g. `L0-480_E0`, `L0-960_E0`
(the `--level` given to `detcache` stays `L0`).
