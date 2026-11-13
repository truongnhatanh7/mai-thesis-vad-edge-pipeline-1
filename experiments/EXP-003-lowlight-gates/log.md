# EXP-003 log (append-only)

### 2026-11-04 23:25 — DATA lane: environment state, STG-NF code
- Lead's background `uv sync --extra pose` and `brew install ffmpeg` were not running when checked (23:24): no process, `/tmp/uvsync.log` had no `uv exit` line, `which ffmpeg` -> not found. Background children appear to be killed when the launching shell call ends. Re-ran `uv sync --extra pose` myself (appended to /tmp/uvsync.log). ffmpeg is not needed for the data-lane tasks (frames are JPEG; UCF-Crime videos not downloaded).
- ShanghaiTech HF download started 23:22 via `~/thesis-data/shanghaitech/hf-123zdm/dl.sh` (SETUP.md §3 curl loop; `-C -` resume). First two attempts were killed when the launching shell ended (resumed with `-C -`, no data loss).
- STG-NF: `git clone https://github.com/orhir/STG-NF third_party/STG-NF`; `git checkout edb5f3220332e160e4d20ce258787d5e2d7e0200` (HEAD is edb5f32 "Update README.md").
- Patch: SETUP.md's `git -C third_party/STG-NF apply ../EXP-001-stgnf-pose-reproduction/stgnf-cpu.patch` fails ("can't open patch") because `-C` changes the cwd, so `../` points to `third_party/`. Used absolute path `$PWD/EXP-001-stgnf-pose-reproduction/stgnf-cpu.patch` instead -> applied cleanly; modified: dataset.py, models/training.py, train_eval.py, utils/pose_utils.py.
- Repo ships `third_party/STG-NF/data/ShanghaiTech/gt/test_frame_mask/` with 107 .npy files (counted with `ls | wc -l`) and checkpoints/ShanghaiTech_85_9.tar.

### 2026-11-04 23:35 — CODE lane: preprocess, detector cache, replay, scoring implemented
- Env: `uv sync --extra pose` had finished (`/tmp/uvsync.log` "uv exit 0"); no new dependency added (`uv add` not needed: onnx, onnxruntime, torch already in pyproject). ultralytics 8.4.201, torch 2.15.0, onnxruntime 1.31.1.
- New code (all in `src/edgelab`): `preprocess/darken.py`, `preprocess/enhance.py`, `preprocess/zerodcepp.py`, `pose/detcache.py`, `pose/replay.py`, `pose/score_dir.py`; `metrics/scores.py:paired_bootstrap_diff` added; `pose/stgnf_eval.py`: `assemble_clip_scores` split into `clip_normality` (keeps +inf for "nobody") + the old function on top (same output; `tests/test_pose_stgnf_scoring.py` still passes). Tests: `tests/test_preprocess_darken.py`, `test_preprocess_enhance.py`, `test_pose_detcache_replay.py`, `test_pose_score_dir.py`, 2 added to `test_metrics.py`. Walkthrough: `WALKTHROUGH.md`.
- Zero-DCE++: `git clone https://github.com/Li-Chongyi/Zero-DCE_extension third_party/Zero-DCE_extension` (depth 1), commit `09f202b690f82da939b8e6ec8535960ae97ad8bd` (2022-06-19). Weights `Zero-DCE++/snapshots_Zero_DCE++/Epoch99.pth`, sha256 `ca8855b90df9a80fa4195a831f33d3476b1964f787eb70602797c773067f3b84`. Licence: no LICENSE file; README states "academic research purpose only, Attribution-NonCommercial 4.0 International" (CC BY-NC 4.0); GitHub API reports no licence. Research use only.
- Our re-implemented network has 10,561 parameters (= paper's count) and loads the official weights with `strict=True`; its output equals the official `model.py` (scale_factor 1) output with the same weights to 1e-6 (`test_our_network_equals_the_official_code_with_the_same_weights`).
- Torch vs ONNX Runtime (CPU, 2 threads, opset 17, dynamic H/W) on a random 480x856 uint8 image: max abs diff 4.77e-07 on the float output in [0,1]; 1 LSB after rounding to uint8 (`test_onnx_runtime_matches_torch_on_480x856`).
- YOLO weights `artifacts/EXP-001/models/yolov8n-pose.pt` sha256 re-checked = `c6fa93dd1ee4a2c18c900a45c1d864a1c6f7aba75d84f91648a30b7fb641d212`.
- Tests: `uv run --extra pose pytest -q` -> 66 passed, 10 skipped (skips are pre-existing: ffmpeg missing, UCF-Crime annotations / index_test.csv absent).
- Plumbing checks on SYNTHETIC data only (no result, nothing to cite): `detcache` CLI on 2 fake videos with E0 and E3 (resume skips existing files), `replay` CLI, `mps_vs_cpu_check` runs, and `score_pose_dir` through the real STG-NF checkpoint on fake pose JSON (MLflow to a temp db, not `mlflow.db`). Real ShanghaiTech frames were not on disk yet (`~/thesis-data/shanghaitech` still downloading), so no real smoke run.
- Ambiguities and how resolved:
  - Mean luma = OpenCV grey (BGR2GRAY, Rec.601 on the sRGB-coded values), used both for E1's mean_Y and the cached luma columns.
  - E1 guards: mean_Y <= 0 -> gamma 0.2, mean_Y >= 1 -> gamma 1.0 (plan gives only the clip range).
  - Plan §4.3 device rule, "max box-conf |Δ| <= 0.01": boxes cannot be paired when counts differ, so conf values (sorted descending, conf >= 0.1) are compared only on frames whose count agrees; frames with different counts count against the 99% criterion. Pooled over the checked videos by `device_check_passes`. Check frames use level L2, E0 by default.
  - E3 (Zero-DCE++) always runs on CPU PyTorch inside `detcache` (the device flag only moves YOLO), so enhanced images do not depend on MPS.
  - Replay passes `img=None` to the tracker: with `gmc_method: none` the frame is not read, and Ultralytics 8.4.201 skips GMC when img is None. Frames hidden by `keep_mask` do not advance the tracker's frame counter (it never sees them), so a gap does not age lost tracks.
  - Replay output file tag: `<cache dir name with '_'->'-'>-c<c>-<tracker>` (STG-NF reads scene/clip from the first two '_' fields of the file name).
  - G0 caveat: EXP-001 R2 ran detector at conf 0.25 + sparseOptFlow GMC inside `model.track`; replay filters conf >= c from a conf 0.01 NMS output (equivalent boxes, greedy NMS) with GMC off, as the plan specifies.
  - STG-NF's loader raises when no video has a 24-frame track (possible at L4 with high c); `score_pose_dir` re-raises with a clear message instead of inventing a number. The caller must decide (AUC would be undefined/0.5, coverage 0).

### 2026-11-04 23:37 — Lead: clarifications before the first real run (not deviations: no result seen yet)
- **No scorable segment** (e.g. L4 at high c, no 24-frame track in any video): the pipeline's output is constant, so
  record `frame_auc = 0.5`, coverage 0, tag `no_segments=true`; never drop the config from tables. If only some
  videos have no segment, the existing rule applies (frames without a person get the most-normal score).
- **Replay filters at c before the tracker.** This also disables BoT-SORT's second (low-score) association for
  detections below c, exactly as stock Ultralytics `model.track(conf=0.25)` does (its predictor filters at conf
  before the tracker). So replay at c = 0.25 is the faithful equivalent of the EXP-001 R2 pipeline except GMC
  (sparseOptFlow → none, plan §4.3). G0 measures that difference.
- **Device-check pairing** (code lane's interpretation, accepted): compare sorted confidences on frames where the
  box count at conf ≥ 0.1 agrees; frames with different counts count as failures in the 99% rule.
- **E3 (Zero-DCE++) runs on CPU torch inside detcache**: accuracy-only; its cost is measured separately (EXP-004).

### 2026-11-04 23:38 — DATA lane: uv sync, STG-NF data, YOLO weights, pytest, R1 check
- `cd experiments && uv sync --extra pose` (rerun by me, 23:25-23:27) -> `uv exit 0`.
- gdown: `uv run gdown 1o9h3Kh6zovW4FIHpNBGnYIRSbGCu-qPt -O ~/thesis-data/stgnf/data.zip` -> 1,333,217,246 bytes; `shasum -a 256` = `b4ecd5b7b7d79ecd09b45a56295b8dd70914154ce7b1f1fb211bef2d5be0bb02` (MATCHES expected). `unzip -q -o data.zip 'data/ShanghaiTech/*'` into `~/thesis-data/stgnf/`. Result: `pose/test` 213 entries, `pose/train` 659 entries, `gt/test_frame_mask` 107 .npy. Symlink `third_party/STG-NF/data/ShanghaiTech/pose -> ~/thesis-data/stgnf/data/ShanghaiTech/pose` created. `diff -rq` of zip's gt dir vs repo gt dir: identical.
- YOLOv8n-pose: `cd artifacts/EXP-001/models && uv run --project ../../.. python -c "from ultralytics import YOLO; YOLO('yolov8n-pose.pt')"`. Downloaded from https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8n-pose.pt (6,832,633 bytes). sha256 `c6fa93dd1ee4a2c18c900a45c1d864a1c6f7aba75d84f91648a30b7fb641d212` (MATCHES). Ultralytics did not pip-install anything (no output beyond the download bar).
- `EDGELAB_UCF_CRIME_ROOT=~/thesis-data/ucf-crime uv run pytest -q -rs` -> **65 passed, 10 skipped** (12.9 s). (SETUP.md says 44; the test suite has grown since — another agent owns src/tests.) Skips: 1 `ffmpeg missing` (test_data_clips_splits.py:61), 9 UCF-Crime annotations/index_test.csv not present. No failures.
- R1 check: `uv run python -m edgelab.pose.evaluate --source released --pose-dir third_party/STG-NF/data/ShanghaiTech/pose/test --out-dir artifacts/EXP-003/r1_check_m2pro` (190 s). mlflow.db was created fresh (experiment 'EXP-001'). Output:
  - released-middle-none AUC=81.93 CI95=[79.05, 84.63] run=8d0186ae1a064bc8980cb0189b1b2af9
  - released-middle-offline_gaussian AUC=**85.94** CI95=[82.75, 88.79] run=8f8e0dd6c5e44a2eba55ed5a39faaf24  (paper variant; reproduces 85.94)
  - released-middle-causal_half_gaussian AUC=84.32 CI95=[81.28, 87.03] run=23db2a4d865c45c888546397cb393758
  - released-middle-causal_ewma AUC=83.75 CI95=[80.68, 86.48] run=1d9926994d42464da79f689ea6aaadda
  - released-end-none AUC=81.82 CI95=[78.75, 84.55] run=1cd35782eb0d4da3b293341c0fd44896
  - released-end-offline_gaussian AUC=84.36 CI95=[81.26, 87.12] run=8d1405ba941c4f779680a6a2504ae5c2
  - released-end-causal_half_gaussian AUC=80.98 CI95=[78.02, 83.64] run=8df0b9f23aed403183063cd0a78c7956
  - released-end-causal_ewma AUC=80.28 CI95=[77.28, 83.01] run=8e9c47cd2ee341ddb595d63dcd37a17e

### 2026-11-04 23:43 — Lead: ShanghaiTech download restarted
- Sequential `dl.sh` (data lane) ended with error after part `aa` complete (1,073,741,824 B) and `ab` at 191 MB; the
  data lane's parallel replacement `dlp.sh` never launched (`dlp.log`: "no such file or directory").
- Relaunched `nohup bash ./dlp.sh > dlp.log` (parts ab–ag in parallel, curl `-C -` resume, same URLs as SETUP.md §3).
  ≈ 3 MB/s aggregate at 23:42 → ETA ≈ 30–40 min. Integrity is checked afterwards with `shasum -a 256 -c SHA256SUMS`.

### 2026-11-05 00:22 — Lead: data lane ended (connection error); lead takes over the download
- Data lane stopped before returning. Audited its state: STG-NF at `edb5f322…` with `stgnf-cpu.patch` applied
  (4 files modified), released test poses (213 files) and 107 gt masks present; `yolov8n-pose.pt` sha256
  `c6fa93dd…d212` ✔; R1 on M2 Pro = 85.94 paper / 81.93 unsmoothed / 80.98 live (runs `8f8e0dd6…`, `8d0186ae…`,
  `8df0b9f2…`; all 8 variants equal EXP-001 to 0.01).
- The ShanghaiTech download had died with it (0 curl processes, 2.84 GB of ≈ 7 GB, repeated "connection reset by
  peer" from huggingface.co). Relaunched as `dlr.sh` in detached tmux session `sht-dl` (curl `--retry-all-errors`,
  stall detection 20 kB/s for 60 s, infinite outer retry, resume `-C -`). A waiter runs `shasum -a 256 -c SHA256SUMS`
  then extracts (excluding pixel masks) and counts videos/frames.

### 2026-11-05 07:10 — Lead: ShanghaiTech frames verified and extracted
- curl downloads in `hf-123zdm/` completed with correct sizes but **6 of 7 parts failed sha256** (`aa` OK). Likely
  cause: two downloader scripts (data lane's `dlp.sh` and the lead's relaunch) wrote the same files concurrently
  between 23:41 and 00:20. Corrupt parts deleted.
- Clean re-download into `~/thesis-data/shanghaitech/hf-clean/` with `aria2c` (single writer, explicit `out=` names,
  resume; HF serves from `us.aws.cdn.hf.co/xet-bridge-us/...`). `shasum -a 256 -c SHA256SUMS`: **7/7 OK**.
- `cat shanghaitech.tar.gz.a? | tar xz -C ~/thesis-data/shanghaitech --exclude='*test_pixel_mask*'` →
  `~/thesis-data/shanghaitech/shanghaitech/testing/frames`: **107 videos, 40,791 jpg** ✔ (matches SETUP.md).

### 2026-11-05 07:20 — Lead: device check PASS → detection-cache sweep launched (MPS)
- `mps_vs_cpu_check` (plan §4.3), first 150 frames of 01_0014 at L0 and 03_0031 at L2: equal box counts at
  conf ≥ 0.1 on 300/300 frames, max |Δconf| = 2.3e-6 (01_0014) / 0.0 (03_0031) → `device_check_passes` = **PASS**.
- Sweep `~/thesis-data/exp003/sweep_cache.sh` in tmux `exp003-cache`: 24 jobs = {L0..L4, L2-nonoise} × {E0..E3},
  3 in parallel, `--device mps`, out `~/thesis-data/exp003/cache/<level>_<enh>/`, per-job exit/seconds in
  `cache/jobs_done.txt`. Order: L0_E0 (G0) → E0 at L1..L4 → E1..E3 at L0..L4 → L2-nonoise × E0..E3.
- Throughput seen: ≈ 42 frames/s per job (e.g. 01_0051: 337 frames in 7.9 s), i.e. ≈ 16 min per E0 config.

### 2026-11-05 07:45 — Lead: G0 PASS; gate sweep launched
- L0_E0 cache: 107 videos, 926 s on MPS. Replay c 0.25 BoT-SORT (gmc none): 69,895 poses, 880 tracks,
  29,407/40,791 frames with ≥ 1 tracked person; 50,019 segments (EXP-001 R2: 69,984 poses, 876 tracks, 50,091 segs).
- **G0: paper-variant AUC 68.53 [63.79, 73.60]** vs EXP-001 R2 68.58 → |Δ| 0.05 ≤ 1.5 → **PASS**. Live 66.17
  [62.23, 70.34] (R2 live 66.42). Anomalous-frame coverage 0.674 (R2 0.675). Replay is deterministic (sweep rerun
  identical). MLflow experiment `EXP-003` created; first manual run + sweep rerun both logged.
- New module `src/edgelab/pose/gate_sweep.py` (driver only: replay_dir + score_pose_dir per cache × c ∈ {0.25,
  0.15, 0.10, 0.05}; applies the no-segment rule; rows → `~/thesis-data/exp003/results.jsonl`). Running in tmux
  `exp003-gates` with `--wait` (polls for newly finished caches).

### 2026-11-05 07:42 — Lead: clean-light gate result; cache sweep parallelism 3 → 7
- Timestamp correction: the two previous lead entries labelled 07:20 and 07:45 were written at ≈ 07:15 and ≈ 07:30
  (estimated, not read from the clock). From here on entries use `date`.
- **L0 (clean), E0, gate sweep** (paper variant AUC / live AUC / anomalous coverage / detections kept / tracked poses):
  c 0.25: 68.53 / 66.17 / 0.674 / 73,616 / 69,895 · c 0.15: 68.54 / 66.27 / 0.679 / 84,133 / 71,016 ·
  c 0.10: 68.55 / 66.28 / 0.679 / 92,787 / 71,149 · c 0.05: 68.56 / 66.25 / 0.678 / 110,024 / 71,265.
  Rows in `~/thesis-data/exp003/results.jsonl` (MLflow `EXP-003`).
  → In daylight the gate is **not** the bottleneck: lowering c admits +50% detections but only +2% tracked poses
  and +0.5 pp coverage. The people missing in EXP-001 (conf 0.26–0.36) were already above 0.25. Interpretation is
  for results.md; H3 is tested only at L2/L3 as pre-registered.
- Dark caches run at ≈ 7–9 frames/s per job (darken ≈ 131 ms/frame under load, Poisson sampling ≈ 65 ms of it)
  vs ≈ 42 frames/s clean. Relaunched sweep with `xargs -P 7` (resumable per video; no partial files left).
  Aggregate ≈ 45 dark frames/s → remaining ≈ 5 h. Darken code unchanged (changing it mid-sweep would mix RNG
  streams across caches).

### 2026-11-05 09:34 — Lead: interim E0 dose-response; resolution addendum launched
- Interim (paper AUC at c 0.25 → best c; anomalous coverage at c 0.25), all from `results.jsonl`:
  L0 68.53 → 68.56 (0.674) · L1 67.90 → 68.21 (0.656) · L2 67.05 → 67.21 (0.633) · L4 54.30 → 54.04 (0.187;
  c 0.15 raises coverage to 0.231 but not AUC). L3 not yet scored.
- Enhancement, interim: L0_E1 (gamma) ≡ L0_E0 (adaptive γ clipped to 1.0 on bright frames → identity, as the
  formula implies); **L1_E1 65.56 (−2.34 vs L1_E0)**; L0_E2 (CLAHE) 68.10 (−0.43 vs L0_E0).
- Added `--imgsz` to `detcache.py` (default 640 → running jobs unaffected; meta records the value). Plan §7
  Addendum A (resolution arm) frozen before any resolution run. `sweep_res.sh` in tmux `exp003-res`, 3 jobs.

### 2026-11-05 09:58 — Lead: interim H1–H3 (analysis module + sign fix)
- New `src/edgelab/analysis/exp003.py` (+ `tests/test_analysis_exp003.py`, 4 passed): co-covered AUC, 2-fold
  cross-fitted gate (seeded halves), paired diffs. **Bug caught before use:** `paired_bootstrap_diff` returns
  AUC(a) − AUC(b); the first interim JSON (`artifacts/EXP-003/analysis_interim_paper.json`) therefore reports
  "vs_L0E0" as the **drop** (L0 − Lk) and "xfit.vs_default" as default − xfit. Wrapper fixed to AUC(b) − AUC(a)
  with a direction test; final tables will be regenerated. Readings below use the corrected sign.
- **H1 PASS**: paper AUC L0 68.53 · L1 67.90 · L2 67.05 · L3 60.43 · L4 54.30 (non-increasing); L3 − L0 = −8.09
  [−11.66, −5.03] ≤ −5.
- **H2 PASS at L2 and L3**: co-covered AUC (frames scored in both L0 and Lk) L2 75.47 → 75.23 (−0.24 vs −1.48
  overall, 16%); L3 74.18 → 70.84 (−3.34 vs −8.09, 41%); anomalous coverage 0.674 → 0.633 (L2), 0.554 (L3).
  L4 (not in H2): co-covered 75.57 → 53.47 on 5,757 frames — pose quality collapses too at the extreme.
- **H3 FAIL**: G-xfit − default = −0.06 (L2), −1.69 (L3). Chosen c per fold: L2 {A: 0.05, B: 0.15}, L3 {A: 0.10,
  B: 0.25}. Lower gates raise normal-frame coverage more than anomalous (L3: 0.465 → 0.487 vs 0.554 → 0.564), i.e.
  they add noisy poses that STG-NF scores as anomalous.
- Reading: people are lost because the detector does not see them at all (below any gate), not because they sit
  just under the threshold → remedies must raise recall (resolution / enhancement), which Addendum A and E1–E3 test.

### 2026-11-05 10:30 — Lead: resolution result at L0; CPU oversubscription fixed; E3 moved to MPS (deviation)
- **Addendum A, L0, E0, c 0.25** (paper AUC / anomalous coverage / tracked poses): r480 62.24 / 0.536 / 53,895 ·
  r640 68.53 / 0.674 / 69,895 · r800 70.70 / 0.732 / 80,920 · r960 71.73 / 0.765 / 87,193.
  **H12 PASS** (960 − 640 = +3.20 ≥ +3; non-decreasing). Gate sweep at each R moves AUC ≤ 0.5.
- Host load average reached 62–76 (10 cores): every detcache job used PyTorch's default thread count. Restarted both
  sweeps with `--threads 1` and `OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1` (resumable; no cache content change —
  thread count does not change detections).
- **Deviation (E3 device):** Zero-DCE++ on 1 CPU thread ran at ≈ 1.6 frames/s (≈ 7 h per cache). Added
  `EDGELAB_ZERODCE_DEVICE` (default cpu) to `enhance.zerodce`; sweep now sets it to `mps`. Parity on 12 frames
  (levels L0/L2/L3/L4): CPU vs MPS output max 1 LSB, 1.31e-5 of pixels differ. The 38 E3 videos already built on CPU
  (L0_E3 33, L1_E3 5) were **deleted** so every E3 cache uses one device. E3 accuracy only; its CPU cost is EXP-004's.
- Queue reordered (no job dropped): E3 at L2, L3, L0 first (needed by H4–H6), then L4, L1, then L2-nonoise × E0–E3.

### 2026-11-05 22:04 — Lead: all 33 caches scored; STG-NF loader bug found and fixed (deviation)
- Cache sweeps finished (SWEEP_DONE ≈ 15:31, RES_DONE 16:02). `gate_sweep --wait` exited at SWEEP_DONE before the
  three L3 resolution caches were done (driver did not know about RES_DONE); rerun without --wait scored them.
- **Bug:** L4_E1 and L4_E3 were recorded as `no_segments` (AUC 0.5) although 10 / 31 videos contain ≥ 1 track
  with a 24-consecutive-frame run. Cause: STG-NF `utils/pose_utils.py: gen_clip_seg_data_np` concatenates
  `global_pose_data` (a mutable default argument) and raises "need at least one array" whenever the FIRST video
  processed has no person with > seg_len poses. That array is not used for test scoring. Fix: guard the empty case
  (`EXP-003-lowlight-gates/stgnf-empty-global.patch`, one line, on top of `stgnf-cpu.patch`).
- Check: L4_E0 c 0.25 rescored with the patch = 54.30 (identical to before). L4_E1 / L4_E3 rescored at all 4 gates;
  new rows appended to results.jsonl with a `supersedes` field (old rows kept, append-only):
  L4_E1 paper 50.42 / 50.51 / 50.51 / 50.51, L4_E3 51.91 / 51.79 / 51.79 / 51.84 (c 0.25 / 0.15 / 0.10 / 0.05).
- Analysis tables regenerated by `edgelab.analysis.exp003_tables` → `tables_paper.md`, `tables_live.md`.

### 2026-11-05 22:08 — Lead: results.md drafted; Addendum B launched
- `results.md` written from `tables_paper.md` (first generation; L4 E1/E3 values from the rescored rows).
- Plan §8 Addendum B frozen before any run: noise-seed replicate (levels `L2-s1`, `L3-s1` added to
  `darken.LEVELS`; seed derives from the level name, so no other code change) and two enhancement × resolution
  combinations. `sweep_b.sh` in tmux `exp003-b` (6 jobs, ≈ 6.5 frames/s each); gate_sweep queued after B_DONE.

### 2026-11-05 23:49 — Lead: Addendum B results (paper variant, c 0.25; rows in results.jsonl)
- **B1 noise-seed replicate — PASS both:** L2-s1 E0 66.93 / E2 69.48 → Δ +2.55 (seed 0: +3.15; |Δ−Δ0| 0.60 ≤ 1.5, > 0).
  L3-s1 E0 60.21 / E1 63.22 → Δ +3.01 (seed 0: +2.86, same sign). Seed-to-seed change of any config ≤ 0.71 AUC.
  Noisy-dark + CLAHE above clean E0 (68.53) in both seeds (70.19, 69.48).
- **B2 combination:** L2 E2 r960 **72.16** ≥ max(E2@640 70.19, E0@960 71.13) → holds (also above clean@960 71.73).
  L3 E1 r800 62.03 < E1@640 63.29 → does not hold (resolution does not stack with gamma in deep dark).
- Live variant: L2 E2 r960 68.99, L3-s1 E1 63.11 (gate_sweep_b.log).

### 2026-11-06 05:20 — Lead: Addendum C results + exploratory coverage-proxy analysis
- **C2 imgsz 320** (paper / live, c 0.25): L0 57.28 / 59.18 (vs 640: −11.25 / −6.99), L2 56.17 / 58.66; coverage L0
  0.359. Expectation AUC(320) < AUC(480) at L0 (57.28 < 62.24) and L2 (56.17 < 61.54) → **holds**.
- **C1 camera gain**: L2-gain E0 69.77 / 67.60 (L2 no-gain 67.05, L2-nonoise 68.86, clean 68.53) → expectation
  "between no-gain and nonoise" **does not hold** (above both, and above clean); CLAHE gain over E0 shrinks
  (+1.49 vs +3.15) → holds. L3-gain E0 61.39, **L3-gain E2 65.16** (best L3 config). Anomalous coverage explains it:
  L2-gain E0 0.699 (clean 0.674; 72,702 vs 69,895 tracked poses).
- **Exploratory (post-hoc, not pre-registered):** across all 45 c-0.25 configs (superseding rows used),
  Spearman(anomalous coverage, AUC) = 0.949 paper / 0.939 live; **label-free all-frame coverage** (anomalous share
  0.425 of test frames) ρ = 0.926 (paper). Within light level, argmax label-free coverage = argmax AUC at L0, L1, L2,
  L2-nonoise, L4; **not at L3** (ρ 0.05: picks r960 60.64 vs gamma 63.29). Partly mechanical (uncovered frames get
  the most-normal score). Candidate runtime control signal → needs a pre-registered held-out validation.
