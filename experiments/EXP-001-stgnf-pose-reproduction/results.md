# EXP-001 results — STG-NF pose reproduction ladder

> Machine: Mac M4 Pro, CPU only. MLflow experiment `EXP-001` in `experiments/mlflow.db`. Per-frame scores:
> `artifacts/EXP-001/<source>/frame_scores_<score_frame>_<smoothing>.npz`. Frame-level AUC over all 40,791
> ShanghaiTech test frames (107 videos); 95% CI = percentile bootstrap over videos (2000 resamples, seed 0).
> Owner sign-off: **pending**.

## Summary against the pre-registration

| # | Pre-registered | Measured | Verdict |
|---|---|---|---|
| R1 | 85.9 ± 0.5 (≈ 81.9 unsmoothed) | 85.94 (81.93) | **pass** |
| R2 | drop of "a few points" (measurement) | 68.58: **−17.4** | much larger than expected |
| R3 | if drop > 3: retrain, target ≥ 84 | triggered; 67.59 | **fail** (recipe itself verified: 85.31 on AlphaPose) |
| R4 | causal variant (measurement) | live 80.98 (released), 66.42 (YOLO) | −5.0 / −2.2 vs offline |
| C | per-stage cost (Mac) | pose 15–28 ms, tracker 2.3 ms, flow 0.7 ms/segment | pose model dominates |

## Results

### R1 — released AlphaPose poses + released checkpoint, CPU (pass)

| Variant | AUC | 95% CI | MLflow run |
|---|---|---|---|
| Their script (`train_eval.py`, printed) | **85.937** | – | stdout in `r1_official_stdout.txt` |
| Paper variant (middle frame, offline Gaussian), our metric | **85.94** | [82.75, 88.79] | `48db69a39b524b4d9a4de8b8265f9588` |
| No smoothing (middle frame) | **81.93** | [79.05, 84.63] | `0338423980b44f5da0dd28edd0013a10` |

Pre-registered rule 85.9 ± 0.5 → **pass** (|Δ| = 0.04). Unsmoothed 81.93 ≈ the 81.9 in repo issue #21. Our
`frame_level_auc` equals their `roc_auc_score` value to 1e-16; our per-frame score assembly equals theirs exactly
(max |Δ| = 0). CPU runtime of their full evaluation: 77 s wall (flow forward over 144,714 segments ≈ 59 s).

### R2 — same frozen checkpoint on our YOLOv8n-pose + BoT-SORT poses (imgsz 640)

| Variant | AUC | 95% CI | Δ vs released | MLflow run |
|---|---|---|---|---|
| Paper variant (middle, offline Gaussian) | **68.58** | [63.86, 73.68] | **−17.36** | `72619e1ef2d64a76a3b02f1155d93ffb` |
| No smoothing (middle) | 64.41 | [60.38, 68.61] | −17.52 | `a31fb83cf2b241d69dce163987acdbd5` |

Expectation was "a few points"; the measured drop is 17 points. **R3 is triggered** (pre-registered rule: drop > 3).

### R4 — causal (live) scoring, both pose sources

`score_frame=middle` = the segment's score is written on its 12th frame (needs 12 future frames, as in the paper).
`score_frame=end` = written on its last frame (the score exists as soon as the segment is complete: fully causal).

| Pose source | Score frame | Smoothing | AUC | 95% CI | MLflow run |
|---|---|---|---|---|---|
| released | middle | offline Gaussian (paper) | 85.94 | [82.75, 88.79] | `48db69a39b524b4d9a4de8b8265f9588` |
| released | middle | causal half-Gaussian | 84.32 | [81.28, 87.03] | `71620f8ccade4a8eb8b7e08777b6bf1f` |
| released | middle | causal EWMA | 83.75 | [80.68, 86.48] | `3f8eef8aa2cb426dacfc914f7950b11d` |
| released | end | none | 81.82 | [78.75, 84.55] | `26c11dee54d84fdc91d9fd202ec973bf` |
| released | end | offline Gaussian | 84.36 | [81.26, 87.12] | `d043919c0fa84e39bf67f07188944028` |
| released | **end** | **causal half-Gaussian** (live) | **80.98** | [78.02, 83.64] | `e4b41ae1f92e47988e877deeb984a9e7` |
| released | end | causal EWMA | 80.28 | [77.28, 83.01] | `702a32ea47d1454a94a170aba137c6f6` |
| YOLO | middle | offline Gaussian | 68.58 | [63.86, 73.68] | `72619e1ef2d64a76a3b02f1155d93ffb` |
| YOLO | middle | causal half-Gaussian | 67.92 | [63.76, 72.38] | `b36faa58c65b4d02bcfd8307bdb49b8f` |
| YOLO | middle | causal EWMA | 67.65 | [63.38, 72.26] | `c3e43b8b83984cd4a0f90f640a060099` |
| YOLO | end | none | 64.43 | [60.48, 68.62] | `921574428de84c858c2bb30b4b14cf15` |
| YOLO | end | offline Gaussian | 67.76 | [63.18, 72.71] | `af49791ed014409c87e833f991b38723` |
| YOLO | **end** | **causal half-Gaussian** (live) | **66.42** | [62.59, 70.52] | `3cbe4bb360dd4e2eae4a538f6054c3b6` |
| YOLO | end | causal EWMA | 65.64 | [61.49, 70.08] | `f4bf73929b474827821047f49ce65c24` |

Honest live numbers: **80.98** (released poses) and **66.42** (our edge pose pipeline).

### R3 — retrain STG-NF on YOLO poses (triggered; fails the target)

Repo recipe unchanged (8 epochs, batch 256, Adamax lr 5e-4, train seg_stride 6, 2 augmentations), `--seed 0`, CPU.

| Model | Train poses | Test poses | AUC (paper variant) | 95% CI | Live (end + causal) | MLflow runs (paper / live) |
|---|---|---|---|---|---|---|
| Shipped checkpoint | AlphaPose | AlphaPose | 85.94 | [82.75, 88.79] | 80.98 | `48db69a3…` / `e4b41ae1…` |
| **Control:** retrained by us | AlphaPose | AlphaPose | 85.31 | [82.01, 88.33] | 80.35 | `3a4d8379376243d58a759deb881489f9` / `057b10643fae4a2198104d67321feec9` |
| Shipped checkpoint (= R2) | AlphaPose | YOLO 640 | 68.58 | [63.86, 73.68] | 66.42 | `72619e1e…` / `3cbe4bb3…` |
| **R3:** retrained by us | YOLO 640 | YOLO 640 | **67.59** | [62.81, 72.88] | 65.70 | `30502140d1cb4125aab8222262542ffa` / `97690eff50cd4af2bdee5d3c07fe47f8` |

Pre-registered target ≥ 84 → **fail**. The control (not pre-registered, added to make R3 interpretable) shows the
recipe works on our CPU stack (85.31 vs 85.94, one seed), so the R3 failure is about the poses, not the training
setup. Training wall time: 12 min on YOLO poses (409 batches/epoch), 64 min on AlphaPose poses under heavy
contention (1,135 batches/epoch). Checkpoints: `artifacts/EXP-001/checkpoints/`.

### Exploratory sensitivity (not pre-registered): pose input size 864 instead of 640

| Pose source | Poses | Anomalous-frame coverage | AUC paper variant | Live (end + causal) | MLflow runs (paper / live) |
|---|---|---|---|---|---|
| YOLOv8n-pose 640 | 69,984 | 67.5% | 68.58 | 66.42 | `72619e1e…` / `3cbe4bb3…` |
| YOLOv8n-pose 864 | 84,399 | 75.9% | 71.71 [67.10, 76.83] | 68.37 [64.48, 72.42] | `d0a925fa2d8141af8d1da5789500dea6` / `1d72d0ef8d884b17ac75595dc485a5c1` |

Coverage and sanity files: `artifacts/EXP-001/yolov8n-pose-botsort-imgsz864/coverage_vs_a.json`,
`artifacts/EXP-001/sanity_released_vs_yolo864.csv`. +3.1 AUC for ≈ 1.8× the pixels; still 14 points below AlphaPose.

### Pose-source sanity (released AlphaPose vs YOLOv8n-pose + BoT-SORT, 107 test videos)

From `artifacts/EXP-001/sanity_released_vs_yolo.csv` and `artifacts/EXP-001/yolov8n-pose-botsort/coverage_vs_a.json`:

| | Released (AlphaPose + PoseFlow) | YOLOv8n-pose + BoT-SORT, 640 |
|---|---|---|
| Tracks / poses | 1,415 / 187,021 | 876 / 69,984 |
| Test segments (24 frames) | 144,714 | 50,091 |
| Median track length (median over videos) | 117.5 frames | 60.5 frames |
| Anomalous frames with ≥ 1 scored person | 96.5% | 67.5% |
| Normal frames with ≥ 1 scored person | 78.0% | 53.5% |
| Mean keypoint confidence (median over videos) | 0.70 | 0.76 |
| Unsmoothed AUC on the 24,236 frames both sources cover | 83.9 | 70.5 |

### Per-stage CPU cost (Mac M4 Pro, comparative only — not edge evidence)

Video 01_0014 (265 frames, 856x480), YOLOv8n-pose imgsz 640, BoT-SORT defaults, OpenCV 1 thread, mean ms (p95):

| torch threads | Pose model (pre+infer+post) ms/frame | Tracker (+ framework) ms/frame | STG-NF, 1 segment (batch 1) | STG-NF, all 786 segments of the video (one batch) | MLflow run |
|---|---|---|---|---|---|
| 1 | 28.0 (29.3) | 2.26 (2.51) | 0.69 ms (0.72) | 72.0 ms | `f30452c0309b420bb5612a6257ead895` |
| 2 | 19.2 (19.6) | 2.19 (2.45) | 0.93 ms (0.97) | 40.8 ms | `eee05ad072de4b1e86d16b342307b775` |
| 4 | 15.4 (15.9) | 2.28 (2.52) | 1.09 ms (1.14) | 25.5 ms | `fa07201cb56745088cba44eefe853a81` |

The pose model is ≈ 90% of per-frame compute. The flow is ≈ 0.7 ms per person per frame (live: one new segment per
tracked person per frame), and more threads make single segments *slower* (threading overhead on a 616-weight
model). The tracker cost includes BoT-SORT's global motion compensation (`gmc_method: sparseOptFlow`), useless for
a static camera. A first benchmark attempt (runs `59e9cac3…`, `936fa9ba…`, `d05f287e…`, tagged `invalid`) silently
ran at 8 threads because Ultralytics resets torch threads — see Observation 6.

## Observations

1. **The reproduction is exact.** CPU and a 2026 software stack give the paper's 85.9 to the third decimal.
2. **Swapping in an edge pose pipeline loses 17 AUC points, not "a few".** Two separate causes:
   - *Missed people (coverage).* YOLO has 37% of AlphaPose's poses, and in no video does it have more. A third of
     anomalous frames get no person score at all, so they are scored "most normal" — guaranteed misses.
     One example frame (01_0140 #155, motorbike anomaly): YOLO at imgsz 640 gives box conf 0.26–0.36 for the ~50 px
     tall people, BoT-SORT's default thresholds (0.25) then start few tracks; at imgsz 864 conf is 0.28–0.62, at
     1280 0.69–0.81.
   - *Worse poses/tracks.* Even on frames both sources cover, unsmoothed AUC is 70.5 vs 83.9. Tracks are half as
     long (more ID switches/fragmentation), and the 10th-percentile keypoint confidence is lower (0.24 vs 0.39). The
     checkpoint was also trained on AlphaPose's keypoint style.
3. **Most of the drop is concentrated** in a few scenes (scene 12: 83.0 → 57.4; scene 04: 89.8 → 68.9; scene 01:
   85.5 → 72.4), while scene 03 and 11 are unchanged. Per-video median drop is 7 points, and 25 of 106 videos are
   better with YOLO. The worst videos are 01_0140 (per-video AUC 1.00 → 0.16), 01_0132 (1.00 → 0.28) and 05_0020
   (0.92 → 0.24); 01_0140 was inspected (motorbike anomaly, people ~50 px tall), the others were not.
4. **Causal scoring costs ≈ 5 points with released poses** (85.94 → 80.98), and ≈ 2 points with YOLO poses
   (68.58 → 66.42). A one-sided filter adds lag, so with end-of-segment scores the causal smoother is slightly *worse*
   than no smoothing (80.98 vs 81.82): the score is delayed twice (segment end + filter lag). The half-Gaussian
   beats the EWMA in every pair.
5. **No day/dark pattern — ShanghaiTech has no dark videos.** Mean luma of all 107 test videos is 102.6–137.7 (of
   255); Spearman(luma, per-video AUC drop) = −0.06, p = 0.54. ShanghaiTech is a daylight control only, as the README
   assumed; low-light work must come from synthetic darkening or another dataset.
6. **Retraining on YOLO poses does not help** (67.59 vs 68.58 frozen). Retraining can adapt the flow to YOLO's
   keypoint style, but it cannot score people that were never detected or tracked long enough to form a 24-frame
   segment; and the training set itself loses the same people (425,529 YOLO vs 1,071,322 AlphaPose training poses, 40%). Input size helps more
   (+3.1 at 864) than retraining.
7. **Ultralytics silently overrides torch threads** (`select_device` → `torch.set_num_threads(8)`), and auto-installs
   `lap` with pip. Both would have corrupted the OS-knob experiments if unnoticed. Fixed in our code (re-apply the
   thread count after setup, `lap` in the lockfile).

## Discussion

The frozen-checkpoint ladder answers the key question for the thesis workload: STG-NF itself is not the bottleneck
(≈ 0.7 ms per segment on one Mac thread, 616 weights); the **pose front end decides accuracy**. On an edge device the pose model is also
where the compute goes. So the trade-off the thesis wants to study (preprocessing × CPU allocation) sits exactly at
the front end — input resolution, detection threshold and tracker settings move the AUC by many points, and they
are compute knobs. Retraining (R3) does not recover the loss, which locates the problem upstream of the flow: detection recall on
small people and track continuity. This makes the workload a good fit, but it also means the "frozen workload" must be defined
including its pose front end and its operating point (imgsz, conf, tracker), and it must be re-measured whenever
those change.

## Threats to validity

- **Third-party data mirror.** Raw ShanghaiTech frames come from HF `123zdm/ShanghaiTech`, not the official OneDrive
  (not CLI-downloadable). Mitigation: per-video frame counts equal the official GT lengths for all 107 videos, and
  the archive's GT masks are byte-identical to the STG-NF repo's.
- **One pose operating point.** R2 uses Ultralytics defaults (imgsz 640, conf 0.25, BoT-SORT defaults). The drop
  is a property of this operating point, not of "YOLO" in general (see the imgsz sensitivity).
- **Released poses' frame alignment** is assumed equal to the JPG numbering (0-based); the matching GT lengths and
  the reproduced AUC support this.
- **Bootstrap over 107 videos** gives wide CIs (±3 points for released, ±5 for YOLO); the R2 drop is far outside
  them, but small differences among R4 variants are not individually significant (no paired test was pre-registered).
- **Thread counts during extraction** were 8 (Ultralytics override); this affects timing only, not poses.
- **Background load.** Other agents and downloads ran concurrently; all timings here are indicative only.
- **R3 is one seed**, one operating point (imgsz 640). STG-NF training is cheap (12 min), so seed variance can be
  measured; not done here.

## Next steps

1. **Owner decision — how to define the frozen workload.** Options: (a) keep YOLOv8n-pose and accept ≈ 66–69 AUC
   on ShanghaiTech (honest edge number, large headroom); (b) tune the front end (imgsz, conf, tracker thresholds,
   GMC off) as a *pre-registered* operating-point sweep with AUC vs ms/frame — this is already a preprocessing ×
   compute trade-off and could become a thesis result; (c) try a stronger-but-small pose model (yolo11n/s-pose,
   RTMPose-t via ONNX) before freezing.
2. If (b): sweep imgsz {640, 864, 1280} × detection conf {0.1, 0.25} × tracker {BoT-SORT, ByteTrack, GMC off} on the
   test set (≈ 20–60 min per config on the Mac), and report the Pareto front. Retraining (R3) only after the front
   end is fixed.
3. Low light: ShanghaiTech has no dark videos (luma 103–138). Use synthetic darkening (README §3) on ShanghaiTech
   test frames, or UBnormal (has night/fog scenes; STG-NF ships a checkpoint), to get a dark stratum.
4. Re-run the stage benchmark on the Pi 5 (and VM for relative numbers), with thread pinning; the Mac numbers above
   are only a ranking.
5. Optional: verify the Huang 2026 86.2 figure (full text not read in the KB), since our measured YOLOv8n-pose +
   BoT-SORT + STG-NF number is ≈ 68.
