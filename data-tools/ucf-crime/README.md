# UCF-Crime local copy

STATUS: COMPLETE (test set COMPLETE; all videos: 1950/1950 present, train 1610/1610; auto-updated 2026-11-03T16:47:11Z)

Machine-readable status is the line above. It is rewritten automatically to `STATUS: COMPLETE (...)` when
`tools/run_rest_and_finalize.sh` finishes and finds all 1,950 videos, including the 1,610 train videos.

- **Test set (290 videos): COMPLETE and verified 2026-11-03.** You can rely on it for evaluation.
- Training set (1,610) and `z_Normal_Videos_event` (50): extraction in progress.
  - Progress: `raw/extract_rest.tsv`. Runner log: `raw/run_rest.log`.

## Source (official only)
- Page: https://www.crcv.ucf.edu/research/real-world-anomaly-detection-in-surveillance-videos/
- Videos and splits: members of `https://www.crcv.ucf.edu/data1/chenchen/UCF_Crimes.zip`
  - 102,957,372,377 B, ETag `629de708-17f8bcd3d9`, Last-Modified 2022-06-06.
  - Fetched member by member over HTTP Range.
  - Each member is verified against the archive's own **CRC32 + size**.
  - sha256 per extracted video in `raw/extract_test.tsv` / `raw/extract_rest.tsv`.
- Annotations: `https://www.crcv.ucf.edu/projects/real-world/Temporal_Anomaly_Annotation_For_Testing_Videos.zip`
  - sha256 `3b11863138b766351ed8893159ca94d8f7465abf1278f2b63c7058463abdfa79`.
- Full audit trail: `/Users/atruong/Documents/thesis/experiments/EXP-000-data/log.md`.

## Layout
```
UCF_Crimes/Anomaly_Detection_splits/Anomaly_Test.txt   290 lines  (sha256 afcb2a6b…48de7)
UCF_Crimes/Anomaly_Detection_splits/Anomaly_Train.txt  1610 lines (sha256 ea91e03d…27744)
UCF_Crimes/Videos/<Class>/<Video>_x264.mp4             same paths as the split files
raw/Temporal_Anomaly_Annotation_For_Testing_Videos/Txt_formate/Temporal_Anomaly_Annotation.txt   (290 lines)
raw/UCF_Crimes.zip.listing.tsv   central directory of the official zip (1,981 entries, CRC32, offsets)
index_test.csv     per test video: codec, w, h, fps, duration, frame counts (container + decoded), class, intervals
lighting_test.csv  per test video: luma/chroma statistics at 1 frame/s (descriptive)
reference/         published gt files used for the cross-check (RTFM @cf938f4, VadCLIP @c41067f)
tools/             http_zip.py, extract_members.py, run_rest_and_finalize.sh, plot_lighting.py
```

## Verified facts (test set)
| Check | Result |
|---|---|
| Videos | 290/290 CRC-verified; 150 normal + 140 anomalous; 38 violence-like (Abuse 2, Assault 3, Fighting 5, Robbery 5, Shooting 23) |
| Format | all H.264, 320×240, 30/1 fps |
| Frames | 1,111,808 decoded (= container count for every video); 10.30 h |
| Anomalous frames | 84,331 (7.59 %), with the 0-based, end-inclusive convention |
| Annotation bounds | every interval starts inside its video. 7 intervals end at or beyond the last frame (Arson011, Arson016, Assault006, Explosion033, Fighting003, Shooting015, Shooting046); they are clipped. |
| Published gt cross-check | VadCLIP `gt_ucf.npy` reproduced **exactly per video**. RTFM `gt-ucf.npy` matches except 46 frames in 4 videos. Details in the EXP-000 log. |

## Use
```python
from edgelab.data import ucf_crime as ucf
labels = ucf.test_frame_labels()          # {video: uint8[n_frames]}, 0-based, end inclusive (RTFM)
subset = ucf.violence_like_test_videos()  # 38 names
```
Override the root with `EDGELAB_UCF_CRIME_ROOT`. Licence: not stated on the page, so treat as research-only. The
content is real crime CCTV.
