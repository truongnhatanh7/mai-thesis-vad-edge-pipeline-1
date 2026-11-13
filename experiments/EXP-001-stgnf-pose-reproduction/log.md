# EXP-001 lab notebook (append-only)

Machine: Mac M4 Pro (10P+4E, 48 GB), macOS Darwin 25.6.0, arm64. All runs CPU only. Repo HEAD at start: `3621863`
(work uncommitted; the lead commits). Env: Python 3.12 via uv; torch 2.15.0, numpy 2.5.4, scipy 1.18.1,
scikit-learn 1.9.1, mlflow 3.17.2, ultralytics 8.4.201, lap 0.5.14, opencv-python-headless 5.0.0.93, gdown 6.5.1.
Times are local (+07).

## 2026-11-03

**22:10 — plan.md written** before any run (pre-registered R1–R4 + stage cost).

**22:11 — clone STG-NF**
```
git clone https://github.com/orhir/STG-NF third_party/STG-NF
```
Pinned commit: `edb5f3220332e160e4d20ce258787d5e2d7e0200` (2023-10-13). Checkpoint `checkpoints/ShanghaiTech_85_9.tar`
ships in the repo.

**22:12 — download STG-NF data.zip** (owner-approved)
```
gdown 1o9h3Kh6zovW4FIHpNBGnYIRSbGCu-qPt -O /Users/atruong/thesis-data/stgnf/data.zip
```
Size 1,333,217,246 B; sha256 `b4ecd5b7b7d79ecd09b45a56295b8dd70914154ce7b1f1fb211bef2d5be0bb02`. Unzipped only
`data/ShanghaiTech/*`: 107 GT masks, 213 test JSONs (107 `*_tracked_person.json` + 106 `*_alphapose-results.json`),
659 train files. The 107 GT masks are byte-identical to the ones in the repo. Linked
`third_party/STG-NF/data/ShanghaiTech/pose -> /Users/atruong/thesis-data/stgnf/data/ShanghaiTech/pose`.

Released JSON format (inspected `01_0014_alphapose_tracked_person.json`):
`{person_id: {"0000": {"keypoints": [x,y,c]*17 (COCO-17, pixels of 856x480), "scores": float}}}`, frames 0-based and
4-digit; `scores` is AlphaPose's detection/proposal score (values > 1 occur, e.g. 2.84). STG-NF's loader adds the neck
(COCO-17 → 18) itself, normalises by 856x480 then per-segment mean/std. The `scores` field is not used by the
shipped checkpoint (`train_seg_conf_th=0`, `model_confidence` off), so YOLO box conf in [0,1] can go there.

**22:14 — ShanghaiTech raw frames source.** The official page (https://svip-lab.github.io/dataset/campus_dataset.html)
has an empty Google Drive href and a OneDrive share (`https://1drv.ms/u/s!AjjUqiJZsj8whLt-1ABerTT-9eH9Ag?e=eJbY6Y`),
which cannot be fetched from a CLI without a browser. `Proactive-LMM/shanghaitech` on HF returns 401 (gated/absent).
Found HF mirror `123zdm/ShanghaiTech`: `shanghaitech.tar.gz.aa..ag` (7 parts, 6,993,962,418 B total). The tar listing
shows the original archive (owner `liuwen`, 2017 timestamps, `shanghaitech/{training/videos, testing/frames,
testing/test_frame_mask, testing/test_pixel_mask}`). Downloaded all parts:
```
curl -sSL --retry 5 -C - -o shanghaitech.tar.gz.$p https://huggingface.co/datasets/123zdm/ShanghaiTech/resolve/main/shanghaitech.tar.gz.$p
```
sha256 (in `/Users/atruong/thesis-data/shanghaitech/hf-123zdm/SHA256SUMS`):
```
e70cc64cf3b9dafb1cbf893c352ce5c078badd60b3526207aed2c1d129dfc816  .aa
82f71132137d6e8eb2aa314a45f4899d4d54832f8593d9a79d1190e72620fc80  .ab
8ec086f62153acf8aec5670ba0d09c9700adc3e10254dcad9eb8a66c775d5cab  .ac
a585cd14ab36f2f71c3431d9459f85cb8a9640acf8232e4b6681b5753b24d2fb  .ad
0b17cb616e4e66dbc6fc9230d25cf0aa0864d43a487ad75f61113172f745a494  .ae
53c5069ca2b02367b131908c51bd6fa9bfd935c8cb172f62bc5e8d0060263f14  .af
3a22b8ed0638f3f267de3bb6fc49aeec17b1dd71f736aa087eeeb961e6175eed  .ag
```
Extracted everything except `test_pixel_mask` to `/Users/atruong/thesis-data/shanghaitech/shanghaitech/`:
107 test videos, 40,791 JPG frames (480x856x3), 330 training videos (.avi, 2.3 GB). Verification: per-video frame
count == GT mask length for all 107 videos; all 107 `test_frame_mask/*.npy` byte-identical to STG-NF's GT. So the
mirror's test part is consistent with the official GT (a third-party mirror is still a threat to validity; noted).

**22:16 — CPU patch of STG-NF** (kept as `stgnf-cpu.patch`, `git diff` of the clone, 5 hunks):
- `dataset.py`, `utils/pose_utils.py`: `np.int` → `int` (removed in numpy ≥ 1.24).
- `models/training.py`: `torch.load(filename, map_location=self.args.device, weights_only=False)` (CUDA-saved
  checkpoint; torch ≥ 2.6 defaults to `weights_only=True`, which fails with UnpicklingError because the checkpoint
  also pickles an `argparse.Namespace` under key `args`; keys = epoch, state_dict, optimizer, args. The file is the
  authors' own checkpoint from the pinned repo, sha256
  `4950205cb5cdd3c4827e3214ac6c93dc7050c0d059c8325907cbe977360a70ca`).
- `utils/pose_utils.py`: matplotlib style `seaborn-ticks` → `seaborn-v0_8-ticks` (renamed in matplotlib ≥ 3.6).
- `train_eval.py`: fall back to a no-op `SummaryWriter` when tensorboard is missing (avoids a new dependency;
  only used for training-loss logging).
Device passed as `--device cpu` (no code change needed; all model code takes the device argument).

**22:18 — R1 official evaluation (their script, unchanged logic)**
```
cd third_party/STG-NF && python train_eval.py --dataset ShanghaiTech --checkpoint checkpoints/ShanghaiTech_85_9.tar --device cpu --num_workers 0
```
Output: `Done with 85.93709088366214% AuC for 40791 samples`. Wall 1:17 (105 s user, 240 s sys, 447% CPU);
the flow forward pass over 144,714 test segments took ~59 s. Stdout saved in `r1_official_stdout.txt`.

**22:20 — our scoring re-implementation** (`src/edgelab/pose/stgnf_eval.py`, `smoothing.py`). Cross-check (scratch
script): our per-frame scores equal STG-NF `get_dataset_scores` exactly (max |Δ| = 0.0); our `frame_level_auc`
on their smoothed scores = 0.8593709088366215 vs theirs 0.8593709088366214 (|Δ| = 1e-16). Unsmoothed AUC = 0.81931.
Our segment counter (`sanity._usable_segments`) reproduces STG-NF's 144,714 test segments on the released poses.

**22:22 — R1 + R4 (released poses), tracked**
```
uv run python -m edgelab.pose.evaluate --source released --pose-dir third_party/STG-NF/data/ShanghaiTech/pose/test
```
8 MLflow runs (2 score-frame positions × 4 smoothers), bootstrap 2000 resamples over videos, seed 0. 4:15 wall.
Per-frame scores in `artifacts/EXP-001/released/`. Numbers → results.md.

R4 design detail (decided before seeing R4 numbers, while writing the code): besides the pre-registered causal
smoother, a second causal element exists: STG-NF writes each 24-frame segment's score onto its *middle* frame
(start+12), which also uses 12 future frames. So R4 is reported for both `score_frame=middle` (smoothing made causal
only) and `score_frame=end` (start+23: score available when the segment is complete — the fully live number).
Causal smoother: one-sided half-Gaussian, σ = √91 ≈ 9.54 frames (the variance-equivalent of their sequential σ=1..6
filters), truncated at 4σ; secondary EWMA with time constant √91 (α = 1−e^(−1/9.54) ≈ 0.10). *Deviation:* plan said
"same σ"; their filter is a cascade of 6 σ, so we used the equivalent single σ.

**22:24 — `uv sync --extra pose`** installed ultralytics 8.4.201 (+ torchvision 0.29.2, polars). First
`model.track` call made Ultralytics **auto-pip-install `lap`** (tracker dependency) outside the lockfile. Fixed
properly: `uv add --optional pose "lap>=0.5.12"` (pyproject `pose` extra + uv.lock changed; lap 0.5.14). Needed by
both BoT-SORT and ByteTrack in Ultralytics.

**22:26 — pose model choice:** `yolov8n-pose.pt` (sha256
`c6fa93dd1ee4a2c18c900a45c1d864a1c6f7aba75d84f91648a30b7fb641d212`, auto-downloaded by Ultralytics to
`artifacts/EXP-001/models/`). Why: nano = edge-realistic, and it is exactly the detector of the Pi 5 skeleton-VAD
system (Huang 2026: YOLOv8n-Pose + BoT-SORT), so our numbers are comparable to that pipeline. yolo11n-pose has similar
COCO accuracy; not run. Tracker: Ultralytics `botsort.yaml` defaults (ReID off), `imgsz=640` (856x480 frames are
letterboxed to 640 wide), `conf=0.25`, `iou=0.7`, fresh model + tracker per video, frames read with OpenCV.

**22:26 — pilot on 2 videos** (01_0014, 05_0017; 698 frames): 34 s wall total → ≈ 49 ms/frame end-to-end with 8
torch threads (inference ≈ 29–34 ms/frame). Estimate for all 40,791 frames: ≈ 33 min. Sanity vs released (same 2
videos): released 10 tracks / 2,017 poses / 1,776 segments; YOLO 10 tracks / 1,358 poses / 1,109 segments. On 01_0014
YOLO finds 322 poses vs 906 — small, distant people are missed at imgsz 640.

**22:28 — full extraction started in background**
```
cd artifacts/EXP-001/models && nohup python -m edgelab.pose.extract --frames-root /Users/atruong/thesis-data/shanghaitech/shanghaitech/testing/frames --out-dir /Users/atruong/thesis-data/stgnf/yolov8n-pose-botsort/test &
```
Per-video timings appended to `.../yolov8n-pose-botsort/test/timings.jsonl`.

**22:49 — full test extraction finished.** 107 JSON files in
`/Users/atruong/thesis-data/stgnf/yolov8n-pose-botsort/test/`. 40,791 frames, 69,984 tracked poses, 876 tracks;
21.0 min of per-video wall time = 30.8 ms/frame end-to-end (read 2.8 ms, track call 28.0 ms of which pose inference
23.9 ms; ultralytics ran with 8 torch threads — see 22:53 note).

**22:50 — R2 + R4 (YOLO poses), tracked**
```
uv run python -m edgelab.pose.evaluate --source yolov8n-pose-botsort --pose-dir /Users/atruong/thesis-data/stgnf/yolov8n-pose-botsort/test
uv run python -m edgelab.pose.sanity --a third_party/STG-NF/data/ShanghaiTech/pose/test --b /Users/atruong/thesis-data/stgnf/yolov8n-pose-botsort/test --out artifacts/EXP-001/sanity_released_vs_yolo.csv
```
Paper variant drops 85.94 → 68.58 (−17.4). **R3 decision (mechanical, pre-registered rule "drop > 3 points"):
R3 is triggered.** Sanity: YOLO has 37% of the released poses (69,984 vs 187,021), 35% of the segments (50,091 vs
144,714), median track length 60.5 vs 117.5 frames; in no video does YOLO have more poses than AlphaPose.

**22:51 — stage benchmark, first attempt (INVALID for thread comparison)**
```
cd artifacts/EXP-001/models && python -m edgelab.pose.stage_benchmark --frames-dir .../testing/frames/01_0014 --num-video-segments 786
```
Runs 59e9cac3…, 936fa9ba…, d05f287e… show no thread effect. Cause found at 22:53: Ultralytics
`utils/torch_utils.py:select_device` calls `torch.set_num_threads(min(8, cpu_count-1))` = 8 on every model setup,
overriding our setting. Runs tagged `invalid` in MLflow. Fix: re-apply the thread count after the first predict
call (`stage_benchmark.py`, `extract.py`), with an assert. Re-run on a quiet machine later. Consequence: the test
extraction (and the train shards below) ran with 8 torch threads regardless of `--threads`. (Also found:
Ultralytics sets `OMP_NUM_THREADS=1` at import if unset, and disables NNPACK on Apple silicon.)
Separate sanity: a bare Conv2d (64→64, 3x3, 96x160) scales 3.15 → 2.16 ms from 1 → 4 threads.

**22:52 — per-scene / luma analysis** (`python -m edgelab.pose.analysis ... --out artifacts/EXP-001/per_video_released_vs_yolo.csv`):
per-video mean luma (every 10th frame, Y of YCrCb) ranges 102.6–137.7 (all 107 test videos are daylight-bright);
Spearman(luma, per-video AUC drop) = −0.06 (p = 0.54, n = 106 videos with both classes). No dark pattern — and no
dark videos to find one in.

**22:52 — coverage diagnostic (exploratory)** (`python -m edgelab.pose.coverage --a artifacts/EXP-001/released --b artifacts/EXP-001/yolov8n-pose-botsort`):
frames with ≥ 1 scored person: anomalous 96.5% (released) vs 67.5% (YOLO); normal 78.0% vs 53.5%. On the 24,236
frames covered by both, unsmoothed AUC 83.9 (released) vs 70.5 (YOLO): so both missed people *and* worse
poses/tracks contribute.

**22:53 — R3 step 1: YOLO poses for the 330 training videos** (needed for the pre-registered retraining). 274,516
frames (24 fps, 856x480 .avi). Added `.avi` support to `extract.py` (`iter_frames`, test `tests/test_pose_extract.py`).
3 shards of 110 videos in parallel:
```
split -l 110 train_videos.txt train_shard_
python -m edgelab.pose.extract --frames-root .../training/videos --out-dir /Users/atruong/thesis-data/stgnf/yolov8n-pose-botsort/train_a{a,b,c} --threads 4 --videos $(cat train_shard_a?)   # x3, background
```
(`--threads 4` was overridden to 8 by Ultralytics in these processes, see above; affects speed only.)

**22:54 — R3 control: retrain on released AlphaPose train poses with the repo recipe** (not pre-registered; it
checks that the recipe reaches ~85.9 on our CPU, so an R3 failure can be attributed to the poses, not the
training setup). MPS attempt failed: `TypeError: Cannot convert a MPS Tensor to float64` (STG-NF data are float64;
fixing needs a code change, so CPU was used instead). Recipe = repo defaults (8 epochs, batch 256, Adamax lr 5e-4,
seg_stride 6, 2 augmentations), `--seed 0`:
```
cd third_party/STG-NF && python train_eval.py --dataset ShanghaiTech --device cpu --num_workers 0 --seed 0 --exp_dir /Users/atruong/thesis-data/stgnf/exp_r3_control_alphapose
```
1,135 batches/epoch, ~3 it/s under CPU contention.

**22:56 — why YOLO misses people (one frame, 01_0140 #155, motorbike anomaly):** overlay shows YOLO keypoints on 1
of 5 people. `model.predict` on that frame: imgsz 640 → 4 boxes with conf 0.26–0.36; imgsz 864 → 6 boxes, conf
0.28–0.62; imgsz 1280 → 6 boxes, conf 0.69–0.81. BoT-SORT defaults (`track_high_thresh 0.25`, `new_track_thresh
0.25`) then start few or no tracks from the 640 detections. People here are ~50 px tall at 856x480; imgsz 640
shrinks them to ~37 px.

**22:57 — exploratory sensitivity (deviation from plan, labelled as such): imgsz 864** (≈ native width 856, rounded
to stride 32), all else identical, test set only:
```
python -m edgelab.pose.extract --frames-root .../testing/frames --out-dir /Users/atruong/thesis-data/stgnf/yolov8n-pose-botsort-imgsz864/test --imgsz 864
```
Pre-registered R2 remains the imgsz-640 number. This only tells the owner how much of the drop is caused by input
resolution (an edge compute trade-off).

**23:07–23:50 — contention.** Load average reached 117 (3 extraction shards + imgsz-864 run + control training,
each with 8–10 OpenMP threads, plus another agent's UCF-Crime download). Per-frame cost rose from 31 to ~70 ms
(train shards) and ~100 ms (imgsz 864). Timings from this window are not used for any claim.

**23:53 — imgsz 864 sensitivity finished** (107 videos; 82.9 ms/frame under contention, not comparable to the 640
run). Evaluated:
```
uv run python -m edgelab.pose.evaluate --source yolov8n-pose-botsort-imgsz864 --pose-dir /Users/atruong/thesis-data/stgnf/yolov8n-pose-botsort-imgsz864/test
uv run python -m edgelab.pose.coverage --a artifacts/EXP-001/released --b artifacts/EXP-001/yolov8n-pose-botsort-imgsz864
uv run python -m edgelab.pose.sanity --a third_party/STG-NF/data/ShanghaiTech/pose/test --b /Users/atruong/thesis-data/stgnf/yolov8n-pose-botsort-imgsz864/test --out artifacts/EXP-001/sanity_released_vs_yolo864.csv
```
Paper variant 71.71 (vs 68.58 at 640); poses 84,399 (vs 69,984); anomalous-frame coverage 75.9% (vs 67.5%).

**23:58 — R3 control finished:** retrained-on-AlphaPose checkpoint gives 85.309 in their script (shipped: 85.937).
Training wall ≈ 64 min under heavy contention. Checkpoint copied to
`artifacts/EXP-001/checkpoints/r3_control_alphapose_seed0.pth.tar` (sha256
`bfb505aa115aee33a11e53083324930bff6444dbcb8eacf312652ee0cb09179d`, + `args.json`) and evaluated with our pipeline:
```
uv run python -m edgelab.pose.evaluate --source r3ctl-alphapose-retrained --checkpoint artifacts/EXP-001/checkpoints/r3_control_alphapose_seed0.pth.tar --pose-dir third_party/STG-NF/data/ShanghaiTech/pose/test
```
**Conclusion: the repo's training recipe works on our CPU stack (−0.6 vs shipped, one seed).**

**00:11 — train extraction finished.** 330 videos, 274,515 frames (OpenCV's CAP_PROP_FRAME_COUNT summed to 274,516;
one video decodes one frame fewer than its header says), 425,529 poses, 5,060 tracks. Merged with symlinks into
`/Users/atruong/thesis-data/stgnf/yolov8n-pose-botsort/train/`.

**00:11 — R3: retrain STG-NF on YOLO poses** (pre-registered, triggered at 22:50). Same recipe and seed as the
control; test = YOLO imgsz-640 test poses:
```
cd third_party/STG-NF && python train_eval.py --dataset ShanghaiTech --device cpu --num_workers 0 --seed 0 --exp_dir /Users/atruong/thesis-data/stgnf/exp_r3_yolo --pose_path_train .../yolov8n-pose-botsort/train/ --pose_path_test .../yolov8n-pose-botsort/test/ --vid_path_train <empty dir> --vid_path_test <empty dir>
```
(vid paths are required by `init_sub_args` to accept custom pose paths but are unused for ShanghaiTech.)
409 batches/epoch (vs 1,135 for AlphaPose: fewer training segments), ~4.7 it/s.

**00:24 — R3 finished:** their script prints 67.589 (YOLO test poses). Training wall ≈ 12 min (8 epochs, CPU, light
load). Checkpoint → `artifacts/EXP-001/checkpoints/r3_yolov8n_botsort_seed0.pth.tar` (sha256
`6ae3260bf116dc25c7f5e2cb7b22ffd0a999bd719da89adbec8128869f6d4d50`, + `args.json`). Evaluated:
```
uv run python -m edgelab.pose.evaluate --source r3-yolo-retrained --checkpoint artifacts/EXP-001/checkpoints/r3_yolov8n_botsort_seed0.pth.tar --pose-dir /Users/atruong/thesis-data/stgnf/yolov8n-pose-botsort/test
```
Pre-registered target ≥ 84 → **R3 fails** (67.59; frozen checkpoint on the same poses: 68.58). One seed only.

**00:26 — stage benchmark, corrected** (machine quiet apart from macOS `mediaanalysisd` ~1 core; load avg ≈ 6):
```
cd artifacts/EXP-001/models && python -m edgelab.pose.stage_benchmark --frames-dir .../testing/frames/01_0014 --num-video-segments 786
```
Runs f30452c0… (1 thread), eee05ad0… (2), fa07201c… (4); the assert confirmed the thread count was applied. 265
frames, first track call excluded. Numbers → results.md.

**00:27 — `uv run pytest -q`: 44 passed** (12 new pose tests in `tests/test_pose_format.py`,
`tests/test_pose_stgnf_scoring.py`, `tests/test_pose_extract.py`). `stgnf-cpu.patch` re-checked = `git diff` of the
clone.

State at end: no background jobs left running. Data under `/Users/atruong/thesis-data/{stgnf,shanghaitech}`
(outside git): data.zip + extracted ShanghaiTech part, HF tar parts (6.5 GB, could be deleted after extraction),
extracted frames (4.3 GB) + train videos (2.3 GB), YOLO poses (test 640, test 864, train 640).

**00:30 — count for results.md:** released AlphaPose training poses = 1,071,322 in 6,742 tracks (330 files), vs
YOLO 425,529 poses / 5,060 tracks.
