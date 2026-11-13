# EXP-000 — lab notebook (append-only)

Machine: MacBook Pro M4 Pro, macOS (Darwin 25.6.0), about 429 GB free at start. Data root:
`/Users/atruong/thesis-data/ucf-crime/` (`experiments/data` symlinks to `/Users/atruong/thesis-data`).
Times are UTC.

## 2026-11-03

### 15:11 — Official source check
- Fetched the official page https://www.crcv.ucf.edu/research/real-world-anomaly-detection-in-surveillance-videos/.
  Its download links:
  - Full archive: `https://www.crcv.ucf.edu/data1/chenchen/UCF_Crimes.zip`
    - HEAD: `200`, `Content-Length: 102957372377`, `Last-Modified: Mon, 06 Jun 2022 11:37:44 GMT`,
      `ETag: "629de708-17f8bcd3d9"`, `Accept-Ranges: bytes`.
  - "In parts": `https://www.dropbox.com/sh/75v5ehq4cdg5g5g/AABvnJSwZI7zXb8_myBA0CLHa?dl=0`. It redirects (302) to
    `https://www.dropbox.com/scl/fo/2aczdnx37hxvcfdo4rq4q/…`, which serves a folder-zip download. **Not used**: the
    range method below gets individual test videos straight from the official archive.
  - Annotations: `https://www.crcv.ucf.edu/projects/real-world/Temporal_Anomaly_Annotation_For_Testing_Videos.zip`
    - HEAD: `Content-Length: 177408`, `Last-Modified: Tue, 27 Feb 2018 16:53:07 GMT`.
- Single-connection throughput test: 50 MB in about 7 s, i.e. **7.48 MB/s**. A full sequential download would take
  about 3.8 h.

### 15:12 — Annotations
- `curl -sSL -o raw/Temporal_Anomaly_Annotation_For_Testing_Videos.zip <url>`
  - 177,408 bytes, sha256 `3b11863138b766351ed8893159ca94d8f7465abf1278f2b63c7058463abdfa79`.
- Unzipped to `raw/Temporal_Anomaly_Annotation_For_Testing_Videos/`. Contents:
  - `ReadMe.docx`
  - `Txt_formate/Temporal_Anomaly_Annotation.txt` (290 lines, 15,574 bytes)
  - `Matlab_formate/*.mat` (290 files)
- ReadMe (read with `textutil`). Each row is: video, event, start1, end1, start2, end2. "-1 means no anomalous event
  instance". Frames are counted after converting the video to images. "Our videos have 30 frames per second."
  **The ReadMe does not say whether frames are 0- or 1-based, nor whether the end is inclusive.**

### 15:13 — Remote zip listing (deviation from "download the full zip first"; reason below)
- The server supports Range requests, so `tools/http_zip.py` (in the data dir) reads the zip **central directory**
  remotely: 1,981 entries, saved as `raw/UCF_Crimes.zip.listing.tsv` with name, sizes, method, offset and CRC32.
  - curl is used instead of Python urllib. Python's TLS verification fails on the CRCV chain ("unable to get local
    issuer certificate"), while curl with the macOS keychain verifies it. **TLS verification is not disabled.**
- Archive facts:
  - 1,950 `.mp4` files, all deflate (method 8), 104.89 GB uncompressed in total.
  - Per folder: Training_Normal_Videos_Anomaly 800; RoadAccidents, Robbery and Testing_Normal_Videos_Anomaly 150 each;
    Burglary and Stealing 100 each; Abuse, Arrest, Arson, Assault, Explosion, Fighting, Shooting, Shoplifting,
    Vandalism and z_Normal_Videos_event 50 each.
- Split files extracted from the archive into `UCF_Crimes/Anomaly_Detection_splits/`:
  - `Anomaly_Test.txt`: 13,056 bytes, sha256 `afcb2a6bffe7286607e08c864425964d8784cbd5b1f263e03b76025e11548de7`,
    290 lines.
  - `Anomaly_Train.txt`: 70,394 bytes, sha256 `ea91e03de7581bfa6b139e663ebb7511ab1651c33427f85741d3f497a2027744`,
    1,610 lines.
  - **Every test and train entry exists in the archive** (0 missing).
  - Test videos total 7.99 GB; train videos 95.81 GB.
- **Deviation and reason.** Instead of downloading 103 GB sequentially and then unzipping (another 105 GB),
  `tools/extract_members.py` downloads each member's compressed byte range, inflates it, and checks **size + CRC32
  against the official central directory**. It writes a sha256 for each extracted video to `raw/extract_*.tsv`.
  - Integrity is the same as unzipping the official file: zip's own CRC32 per member.
  - The 290 test videos arrive first (about 8 GB, under 30 min).
  - No duplicate 103 GB zip sits on disk.
  - Trade-off: we have no sha256 of the whole `UCF_Crimes.zip` file. The ETag + Content-Length above identify the
    exact archive version.
- Smoke test on 2 videos (`Abuse028`, `Abuse030`): OK.

### 15:14 — Test-set extraction started (background, 6 parallel ranges)
`python tools/extract_members.py UCF_Crimes/Anomaly_Detection_splits/Anomaly_Test.txt UCF_Crimes/Videos raw/extract_test.tsv 6`
- Aggregate throughput measured at 4.1–6.4 MB/s. The server seems to cap total bandwidth at about 7 MB/s, not per
  connection.

### 15:15 — Published ground-truth construction (label convention)
Downloaded into `reference/` (sha256 below):
- RTFM `tianyu0207/RTFM` @ `cf938f4c38c908ba19936ecfc60d54de7da0d8b9`
  - `list/make_gt_ucf.py` (`3d91fdb5…ee6`), `list/gt-ucf.npy` (`481e0e5c…f400`): float64, **1,114,144 frames**,
    84,389 positive.
  - The code reads the `.mat` annotations and sets `gt = 1` for `range(start, end + 1)`, i.e. **0-based index, end
    inclusive**. It clips at `num_frame` when `end + 1 > num_frame`.
  - Per video, `num_frame = n_I3D_clips * 16`.
- VadCLIP `nwpu-zxr/VadCLIP` @ `c41067f07d252efcda18008bea367886070c33b0`
  - `list/make_gt_ucf.py` (`1a40941f…`), `list/gt_ucf.npy` (`c6448801…6301`): float32, **1,109,888 frames**,
    84,160 positive.
  - The code uses `gt_vec[start:end] = 1`, i.e. **0-based, end exclusive**.
  - Per video, `lens = (n_clips + 1) * 16`, and then the last 16 frames are dropped (`gt_vec[:-16]`).
  - Its `list/Temporal_Anomaly_Annotation.txt` equals the official file once `.mp4` is stripped and whitespace is
    collapsed (`diff` empty).
  - Note: the released `ucf_CLIP_rgbtest.csv` has no `__0.npy` rows, so the published script cannot be re-run as it
    stands. The gt file was built from a different list.
- **Chosen convention for `edgelab.data.ucf_crime`:** 0-based, end inclusive (RTFM). VadCLIP's variant is available
  via `end_inclusive=False`.

### 15:29 — Test-set extraction finished
- 15:13:47 → 15:29:17 (15.5 min): 290 `ok` (+ 2 `skip-exists` from the smoke test). 0 FAIL, 0 leftover `.part`
  files. 7.85 GB on disk.

### 15:30 — Test index (`uv run python -m edgelab.data.ucf_crime index`, 64 s, 8 processes)
- `index_test.csv` sha256 `c9ab617f222f95127fd8b932c786ee43cfdaac65ae94b9bc33473b3138742e57`.
- All 290 videos: h264, 320×240, `r_frame_rate` 30/1.
  - 1,111,808 decoded frames, equal to the container `nb_frames` for every video.
  - 10.30 h in total.
  - Frames per video: min 155, median 1,788, max 107,997.
- Anomalous frames (0-based, end inclusive): 84,331 (7.585 %). End exclusive: 84,182. The difference is 149 frames.
- Intervals that end at or beyond the last decoded frame are clipped by `frame_labels`:
  - Arson011 (680–1267, n=1266)
  - Arson016 (1000–1796, n=1795)
  - Assault006 (1185–8096, n=8096)
  - Explosion033 (1550–3156, n=3154)
  - Fighting003 (1820–3103, n=3102)
  - Shooting015 (855–1715, n=1713)
  - Shooting046 (4760–5088, n=5088)

### 15:31 — Published gt reconstruction (scratch analysis scripts; final check encoded in tests)
- First attempt assumed the gt files follow the annotation-file order. That order is alphabetical, with "Normal"
  between Fighting and RoadAccidents, and it gave large mismatches. Inferring each video's start offset from the
  positive runs showed that the gt order is **anomalous videos in Anomaly_Test.txt order, then the 150 normal videos**.
  This is RTFM's `ucf-i3d-test.list` order (sha256 `82d581cd…6c2a4d`).
- RTFM:
  - per-video length = ceil(n/16)·16; total 1,114,144 = published ✔
  - labels end inclusive
  - 4 videos differ: Arson011 12, Arson016 11, Explosion033 11 and Shooting015 12 frames (46 in total, all where the
    annotation end overruns n; RTFM labels up to the padded end)
  - every other video is identical
- VadCLIP:
  - per-video length = floor(n/16)·16, except Shooting032 (+16)
  - labels end exclusive
  - **all 290 segments identical**
  - published total = ours + 48 trailing zero frames, i.e. 3 normal videos with one extra clip each; they are all-zero
    so they cannot be located
- Conclusion: our parser and frame counts reproduce both published ground truths. Encoded as
  `test_published_gt_matches_our_labels_per_video[rtfm|vadclip]`.

### 15:33 — Lighting (`uv run python -m edgelab.data.ucf_crime lighting`, 42 s, 8 processes)
- Definitions were fixed in `plan.md` / `WALKTHROUGH.md` before running. The greyscale cut-off is chroma < 3.0.
- `lighting_test.csv` sha256 `1dae21c0e5eab69d112178a7dd7773138410c91e8cce4879d0691d843f4f70a8`. 37,188 sampled
  frames.
- `labelling_sheet.csv` (`uv run python -m edgelab.data.ucf_crime sheet EXP-000-data/labelling_sheet.csv`) sha256
  `58c17b810fbc697590fbf753cb82a723b42aaecd839c723c55ae093665e60e4c`.
- Figure: `lighting_hist.png`, made by `/Users/atruong/thesis-data/ucf-crime/tools/plot_lighting.py`.
- Eye-check: one frame at t = 20 s from 12 videos (4 darkest by mean luma, 4 most greyscale, 4 mixed), via ffmpeg
  into a contact sheet in the session scratchpad (not kept). Findings are in results.md, Observations.

### 15:30–15:36 — Remaining 1,660 videos
- `raw/remaining_members.txt` = all 1,950 archive mp4s minus the 290 test videos = 1,610 train + 50
  `z_Normal_Videos_event`.
- First started 15:29:54 as a session background job. It was stopped at 15:36 because that job type has a 2 h
  limit, below the ETA of about 4–5 h. 194 videos were already done and verified. Stale `.part` files were deleted.
  Files are renamed into place only after the CRC passes, so completed files are trustworthy.
- Relaunched detached at 15:36:28:
  `nohup tools/run_rest_and_finalize.sh > raw/run_rest.log 2>&1 &`
  The script:
  - runs up to 3 resumable passes
  - counts the mp4s (expects 1,950) and the train files present (expects 1,610)
  - rewrites the `STATUS:` line of `README.md` to COMPLETE or PARTIAL
- Throughput is ~6 MB/s, so ETA is ~20:00–21:00 UTC.

### 15:40 — Tests
- `uv run pytest -q`: see the final line below.
- `uv run pytest -q` → **44 passed** (12 in `tests/test_data_ucf_crime.py`, incl. 2 parametrized); 5 warnings come from other modules, not from the UCF-Crime tests.

### 15:42 — Problem for the lead (outside my lane, not changed)
- `experiments/.gitignore` line 1 is `data`. That pattern also ignores **`src/edgelab/data/`**, so
  `ucf_crime.py` and even `__init__.py` are untracked (`git check-ignore -v` confirms it). Suggested fix: change it
  to `/data`.
