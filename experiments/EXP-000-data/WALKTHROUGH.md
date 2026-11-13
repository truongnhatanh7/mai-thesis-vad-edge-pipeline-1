# WALKTHROUGH — `edgelab.data.ucf_crime` (UCF-Crime test set)

Code: `src/edgelab/data/ucf_crime.py`. Tests: `tests/test_data_ucf_crime.py`.
Download helpers (not part of the package; they live with the data): `/Users/atruong/thesis-data/ucf-crime/tools/`.

## End-to-end flow

```
official CRCV server
  ├─ Temporal_Anomaly_Annotation_For_Testing_Videos.zip ──curl──► raw/…/Txt_formate/Temporal_Anomaly_Annotation.txt
  └─ UCF_Crimes.zip (103 GB, HTTP Range)
        tools/http_zip.py        read the central directory remotely ► raw/UCF_Crimes.zip.listing.tsv (name, size, CRC32, offset)
        tools/extract_members.py for each video: range download → inflate → check size+CRC32 ► UCF_Crimes/Videos/<class>/<video>.mp4
                                                                                             raw/extract_*.tsv (sha256 per video)
            │
            ▼
ucf_crime.load_test_list()      Anomaly_Test.txt → 290 relative paths
ucf_crime.load_annotations()    annotation txt → {name: AnnotatedVideo(class, intervals)}
            │
            ├─ build_test_index()     ffprobe -count_frames per video ► index_test.csv (fps, size, duration, frame counts)
            │                         + class, intervals, n_anomalous_frames
            ├─ test_frame_labels()    index frame count + intervals → {name: uint8[n_frames]} (1 = anomalous)
            └─ build_lighting_table() 1 frame/s → YCrCb → luma/chroma stats ► lighting_test.csv
                    write_labelling_sheet()  ► EXP-000-data/labelling_sheet.csv (owner fills day/night/ir/unclear)
```

Commands (run from `experiments/`):
- `uv run python -m edgelab.data.ucf_crime index`
- `uv run python -m edgelab.data.ucf_crime lighting`
- `uv run python -m edgelab.data.ucf_crime sheet EXP-000-data/labelling_sheet.csv`

Data root: `/Users/atruong/thesis-data/ucf-crime`. Override it with the env var `EDGELAB_UCF_CRIME_ROOT`, e.g. on the
VM or the Pi.

## The functions, in order of use

| Function | In → out | Guarded by test |
|---|---|---|
| `parse_annotation_line` | `"Arson011_x264.mp4  Arson  150  420  680  1267"` → `AnnotatedVideo("Arson011_x264", "Arson", ((150,420),(680,1267)))`; `-1` pairs are dropped | `test_parse_single_and_double_interval_lines` |
| `load_test_list` / `load_train_list` | split txt → list of `Class/Video.mp4` | `test_test_list_and_annotations_agree`, `test_train_list_has_1610_videos_disjoint_from_test` |
| `load_annotations` | annotation txt → dict by video name (290) | same |
| `frame_labels(video, n_frames, end_inclusive=True)` | intervals → 0/1 array. **Convention: 0-based index, end inclusive** (RTFM); `end_inclusive=False` = VadCLIP; clipped to `n_frames` | `test_frame_labels_end_inclusive_vs_exclusive`, `test_frame_labels_clip_interval_past_video_end`, `test_known_videos_label_counts` |
| `violence_like_test_videos` | names whose class ∈ {Abuse, Assault, Fighting, Robbery, Shooting} → 38 | `test_violence_like_subset_has_38_videos` |
| `probe_video` | mp4 → codec, width, height, fps, duration, container frame count, **decoded** frame count | `test_index_covers_all_test_videos_at_30_fps` |
| `test_frame_labels` | `index_test.csv` + annotations → labels for all 290 videos | `test_annotation_intervals_start_inside_video`, `test_published_gt_matches_our_labels_*` |
| `lighting_stats` | mp4 → per-video luma/chroma summary (definitions below) | (descriptive only; spot-checked by eye, see results.md) |

## Why the label convention matters, and how it is checked

The ReadMe only says "starting frame" and "ending frame". Published code disagrees by one frame: RTFM includes the end
frame, VadCLIP does not. The difference is far below anything that moves an AUC, but we still make the choice explicit.

The strongest check is `test_published_gt_matches_our_labels_*`. It rebuilds RTFM's and VadCLIP's released `gt` arrays
from our own labels, by truncating each video to whole 16-frame clips (rule recorded in `log.md`), and compares them.

## Lighting definitions (fixed before looking at the numbers)

Frames are sampled at one per second: every `round(fps)`-th decoded frame, starting at frame 0. Each is converted with
OpenCV `BGR2YCrCb`, so Y, Cr and Cb are on a 0–255 scale with 128 = neutral chroma.

| Column | Definition |
|---|---|
| `mean_luma` | mean over sampled frames of the frame's mean Y |
| `luma_p10/p50/p90` | percentiles of **all sampled pixels** pooled together (exact, from a 256-bin histogram) |
| `frame_mean_luma_min/max` | darkest and brightest sampled frame (shows lights switching / scene changes) |
| `mean_chroma` | mean over frames of mean(\|Cr−128\| + \|Cb−128\|) |
| `greyscale_frac` | fraction of sampled frames with chroma < 3.0 (IR night-mode heuristic; **not** a stratum threshold) |
