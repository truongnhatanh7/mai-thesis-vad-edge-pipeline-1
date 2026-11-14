# Master Synthesis & Evidence Dossier: Event-Aware Scheduling and Computation Reuse for CPU-Only Skeleton Video Anomaly Detection (`Q2-P1-E00` to `Q2-P1-E05`)

**Date:** 14 November 2026  
**Repository:** `truongnhatanh7/mai-thesis-vad-edge-pipeline-1` (latest commit at E05 completion: `46349e1`)  
**Workspace Root:** `/Users/nanh/Documents/RMIT_MscAI/thesis/` (`experiments/` is the Python package)  
**Purpose:** Single self-contained reference dossier capturing all background, methodologies, locked artifacts, exact empirical results, negative results, statistical intervals, figures, and limitations from `Q2-P1-E00` through `Q2-P1-E05` (plus relevant prior root-cause context from `EXP-020` and `EXP-021`) so another AI agent or author can draft the full Q2 manuscript without fabricating or misinterpreting any number.

---

## 1. Executive Summary & Core Thesis Narrative

### 1.1 The Problem
Edge deployment of skeleton-based Video Anomaly Detection (VAD)—specifically a Huang-inspired CPU-only pipeline consisting of **pose estimation (`YOLOv8n-pose`) $\to$ multi-object tracking (`BoT-SORT`) $\to$ normalizing-flow anomaly scoring (`STG-NF`, Hirschorn & Avidan, ICCV 2023)**—fails on resource-constrained edge CPUs (Raspberry Pi 5 class) for three coupled reasons:
1. **Small-Person Blindness (Frontend Label Artefact + Tracker Drop):** Stock `YOLOv8n-pose` at 640p achieves only **68.53** offline paper AUC (**66.03** strict causal AUC) on ShanghaiTech because it attains **0.4%** recall on people under 48 px tall (`EXP-011`, `E01`). Furthermore, `EXP-020` proved this is primarily a *training-label artefact* (standard COCO-to-YOLO conversion deletes person instances without keypoint annotations, teaching the detector to treat small/crowded people as background), not a feature-stride limitation. Even after fixing labels (`ML0`), **84.2%** of remaining missed person-frames are *intermittent gaps inside active tracks*, **52.3%** of which already exist as low-confidence boxes (`0.05 <= conf < 0.25`) that `BoT-SORT` never receives because the pipeline pre-filters at `conf >= 0.25`.
2. **Pose-Stage CPU Dominance & Queue Collapse:** On a 4-core ARM64 Linux VM (`edge-os`, comparative proxy for Pi 5), end-to-end processing takes **14.93 ms/frame** on ShanghaiTech (**78.2%** in NCNN FP16 pose inference, **16.5%** in STG-NF scoring, **2.8%** tracking, **2.5%** decode) and **20.21 ms/frame** on 1080p/30fps UBnormal. Under a Pi-slowdown sensitivity `s >= 3.0` (ShanghaiTech) or `s >= 2.0` (UBnormal), running full-rate `K=1` saturates the CPU queue, exceeds the 2.0 s rolling buffer, and collapses delivered AUC to chance (`66.59` on SHT, `52.29` on UB).
3. **Evaluation & Naive Subsampling Traps:** Offline VAD evaluation hides failure through two look-ahead leaks (test-set-global `most_normal` fill for unscored frames and two-sided Gaussian smoothing) and omits the final window of every track (`E00`). Meanwhile, naive uniform frame skipping (`stride >= 2` without pose filling) yields **zero** scorable windows because `STG-NF` requires 24 consecutive frames at native cadence (`missing <= 2`), while `STG-NF`'s per-window mean/std normalization invalidates cross-window hidden-state reuse (`E00`).

### 1.2 The Headline Findings (`E00` $\to$ `E05`)
1. **Accuracy Without Enlarging the Model (`E01`):** Retaining box-only COCO persons during training (`ML0` @ 640, NCNN FP16, `1.00x` stock FLOPs) + unlocking two-stage `BoT-SORT` (`c=0.20, track_low_thresh=0.05, buffer=30, match_thresh=0.85`) raises ShanghaiTech offline `S0` AUC from `68.53` $\to$ **`81.62`** (`+13.09 pt`) and strict causal `S4` AUC from `66.03` $\to$ **`76.82`** (`+10.79 pt`; UBnormal `57.72` $\to$ **`61.08`**). By contrast, a P2 stride-4 head (`MLP`, `1.82x` detector latency) adds an unconfirmed `+0.30 pt` on confirmation (`-0.27 pt` under locked tracking) and is rejected.
2. **Exact vs. Invalid Computation Reuse (`E00`, `E03`):**
   - **Per-frame `(x, y, conf)` reuse across overlapping windows** is **100% bit-exact** (`max_abs_diff = 0.0`), reducing pose evaluations by `18.5x` (SHT) / `13.2x` (UB). Reusing per-window normalized inputs (`65.61` vs `73.57`) or flow hidden states (`max_abs_diff = 0.84–2.07`) fails.
   - **Cross-frame static-validity detector output reuse (`C1`)** on a plan-compliant `Fixed-Burst` intact-window schedule saves **26.7% CPU** (SHT) / **18.7% CPU** (UB) at `W=1`, and when matched at total CPU budget (12-cell mean on ShanghaiTech `test_confirmation`, 2,000-resample paired video bootstrap), **`Cache-C1` outperforms `Fixed` by `+2.37 [+1.38, +3.41]*` pt** (`+2.59 [+1.73, +3.47]*` across all 107 clips) and outperforms `Stock-Baseline` by **`+12.65 [+7.96, +17.41]*` pt**. Removing oracle track-ID inheritance (`C1-ReTrack`) retains **62%** of the gain (`+1.48 [+0.24, +2.62]*` pt).
3. **Strong Negative Results (Pre-registered Stop Rules & Rejections):**
   - **Event/Motion Scheduling (`E03`):** `Motion-Only` (`-1.35 [-2.55, -0.26] pt` vs `Fixed`) and `Adaptive-Only` (`-0.17 [-0.99, +0.65] pt`) do **not** beat `Fixed` at matched total CPU cost.
   - **Joint Controller (`E03`):** `Joint-Controller` loses to simple `Cache-C1` by **`-1.60 [-2.43, -0.82] pt`** and to `Independent-Combo` by **`-1.20 [-2.06, -0.34] pt`**, and is **rejected** under the pre-registered Stop Rule.
   - **Cross-Window Scorer Reuse (`C2`, `E03`):** Saves only `0.3%` total CPU (scoring is small relative to detection) and is dropped.
   - **H.264 Codec Motion Vectors (`+MV`, `E04`):** Rejected (`-0.08 pt` on SHT confirmation; `-2.25 [-3.99, -0.41] pt` on UB dev). Per-frame MV export overhead (`0.374 CPU-ms/frame`) cancels the cheaper check (`0.114` vs `0.373 ms`). MV pose advection adds `-0.06` to `+0.07 pt`.
   - **Post-Training Static INT8 (`+INT8`, `E04`):** Rejected. Although ORT INT8 loses negligible detector accuracy (`-0.19 [-0.90, +0.43] pt`) and is `2.28x–2.86x` faster than ORT FP32, **NCNN FP16 (`10.3 ms` wall, `37.5 CPU-ms` at `T=4`) is faster and uses `21%` less CPU than ORT INT8 (`12.0 ms`, `47.6 CPU-ms`)** on ARM64! Switching from NCNN FP16 to ORT INT8 loses `-1.85 pt` at matched cost (`+22.8%` CPU at equal schedule).
   - **No GPU Retraining Needed (`E04` §7 Gate):** All three optional GPU retraining gates returned False (`$0` of `$10` RunPod RTX 6000 Ada budget spent).
4. **Overload Guard & Sustained Operation (`E05`):** Adding a hysteresis queue-wait load-shedding guard (`Cache-C1+Guard`, on `> 400 ms`, off `< 150 ms`) costs `0.00 pt` in non-overload cells, rescues overloaded UBnormal cells by **`+5.93 [+1.73, +9.92]*` pt**, keeps all 8 multi-hour simulated streams bounded (vs 7 unbounded failures for unguarded arms), and runs **11,520 consecutive frames** in the live ARM64 VM with `<= 0.37%` RSS growth (`~900 MB` peak) and `<= 408 ms` p99 Pi-equivalent queue wait (vs `196–307 s` collapse for `Fixed-K1`).

---

## 2. Pipeline, Datasets, Splits & Locked Artifacts (`E00`, `E01`)

### 2.1 System Architecture
1. **Frontend Pose Estimator:** `YOLOv8n-pose` at `384x640` letterboxed input (`640` nominal imgsz).
   - **`stock`:** Ultralytics pretrained `yolov8n-pose.pt` (SHA256 `c6fa93dd...fb641d212`).
   - **`ML0` (Locked Winner):** Fine-tuned for 4 epochs on COCO `train2017` (`64,115` person images, `seed=0`, no ShanghaiTech/UBnormal data) using **partial-label targets** (persons without keypoint annotations are retained as box-only targets with keypoint location and visibility BCE losses masked via `edgelab.pose.loss_patch`). Exported to **NCNN FP16** (`ML0-640-384x640-ncnn`, `model.ncnn.param` SHA256 `d2d16b87...8d434a`, `model.ncnn.bin` SHA256 `87d55ebf...0c8ee`).
   - **`M00` (Ablation Control):** Identical 4-epoch COCO schedule with standard Ultralytics labels.
   - **`MLP` / `M0P` (P2-Head Variants):** Append-only stride-4 P2 head (`3.455M` params, `17.61 GFLOPs` vs `3.290M` / `9.18 GFLOPs` = `1.92x` FLOPs).
2. **Tracker (`BoT-SORT`, `ultralytics==8.4.201`):** `gmc_method=none`, `with_reid=False`, `fuse_score=True`.
   - **Default (`E00`):** pre-filter `conf >= 0.25`, `track_buffer=30`, `match_thresh=0.80` (second-stage association starved).
   - **Locked (`E01`, dev-selected):** `track_high_thresh c = 0.20`, **`track_low_thresh = 0.05`** (boxes in `[0.05, 0.20)` enter second-stage association to continue existing tracks without spawning new tracks), `track_buffer=30`, `match_thresh=0.85`.
3. **Anomaly Scorer (Frozen `STG-NF`, commit `edb5f3220332e160e4d20ce258787d5e2d7e0200`):**
   - **ShanghaiTech checkpoint:** `ShanghaiTech_85_9.tar` (`SHA256 4950205c...77360a70ca`), window length $L=24$ frames (`1.0 s` at 24 fps), `missing <= 2` frames per window tolerated, 18 COCO/AlphaPose-mapped keypoints normalized per window by subtracting window-level $(x, y)$ mean and dividing by $y$-std after dividing by `856x480`.
   - **UBnormal checkpoint:** `UBnormal_unsupervised_71_8.tar` (`SHA256 1e054a51...3bcb7ade3`), window length $L=16$ frames (`0.533 s` at 30 fps), normalized with the original code's `856x480` constant (retained for checkpoint parity).

### 2.2 Datasets & Disjoint Video-Grouped Splits
- **ShanghaiTech Campus:** `107` test clips (`40,791` frames, `856x480`, JPEG sequence, nominal `24.0 fps`; note `E00` R7 found `251` train AVIs are 24 fps and `79` are 25 fps across scenes `02, 04, 07, 09, 10`, so `24.0 fps` on test JPEGs has `<= 4%` timing uncertainty). **194** ground-truth anomalous events.
  - **`test_dev`:** `35` clips (`13,067` frames, `6,041` anomalous frames, 10 scenes).
  - **`test_confirmation`:** `72` clips (`27,724` frames, `11,285` anomalous frames).
  - **`train_calibration`:** `12` normal training clips (`01_001`..`12_001`).
- **UBnormal:** `211` test clips (`92,640` frames = `51.5 min` at native **`30.0 fps`**; `E00` caught and fixed `EXP-023` hardcoding `24.0 fps` in 9 places, which overstated UB delay by `1.25x` and understated FA/h by `0.80x`; `1280x720` resized/letterboxed to `448x640` for YOLO). **199** ground-truth anomalous events. *(Official UB validation set has no frontend caches, so `test_dev`/`test_confirmation` partition the 211 test clips.)*
  - **`test_dev`:** `70` clips (`16,854` anomalous frames).
  - **`test_confirmation`:** `141` clips (`32,796` anomalous frames).
- **Proxy Disclosure:** On UBnormal, no NCNN FP16 cache was generated; `ML0_mps` (MPS FP32) supplies detections (`E01` verified NCNN FP16 vs MPS FP32 on SHT differs by only `+0.20 [-0.04, +0.48] pt`) and VM UB-geometry (`448x640`) NCNN timings supply costs.

### 2.3 Hardware Disclosure
**Physical Raspberry Pi 5 8GB (`DECISIONS.md` D8/D10) was not connected during this study.** Every latency and CPU figure is measured inside the **`edge-os` ARM64 Linux VM** (Lima/VZ, 4 vCPUs, 8 GiB RAM, Ubuntu 24.04 aarch64 on an Apple M2 Pro host, isolated with `taskset`/CPU affinity and host-load monitoring requiring outside-VM host CPU `< 30%`). Pi scaling is modeled via a slowdown multiplier $s \in \{1.0, 1.5, 2.0, 2.5, 3.0, 3.5\}$ ($s=1.0$ is VM speed; Cortex-A76 @ 2.4 GHz is assumed near $s \approx 2.0\text{--}2.5$). Moreover, VM exposes `asimddp`, `i8mm`, `bf16` (upper bound for INT8) and no thermal sensor (proxies only).

---

## 3. Experiment-by-Experiment Methodology & Findings

### 3.1 `Q2-P1-E00` — Audit & Strict Causal Scoring Protocol
- **Goal:** Verify reproducibility (`0.0000 pt`), decompose offline vs. causal delivered AUC, prove valid vs. invalid overlap caching, and lock splits/sentinels.
- **Historical Reproduction (`R1`–`R4` PASS):** All 31 artifact hashes verified; all 10 historical sources reproduced within `0.0000 pt` of stored values (SHT: `released 85.94/80.98`, `stock 68.53/66.17`, `ML0_mps 77.88/73.53`, `ML0_ncnn 78.00/73.77`, `MLP_mps 79.35/74.79`; UB: `released 71.78/70.32`, `stock 55.76/56.20`, `ML0_mps 54.37/54.93`). Cache replays are bit-identical (`0` differing clips).
- **Offline $\to$ Causal Decomposition (`S0` $\to$ `S4`, `ML0_mps`):**
  - **S0 (Offline Paper):** middle landing + two-sided Gaussian smoothing + global test-set `most_normal` fill + last-window omission bug (`ceil((n-L)/1)` drops `0.85%` of SHT windows). SHT = `77.88`, UB = `54.37`.
  - **S1 (Offline Live):** end landing + causal half-Gaussian ($\sigma = \sqrt{\sum_{s=1}^6 s^2} \approx 9.539$, truncated at $4\sigma$). SHT = `73.53` (`-4.35 [-5.96, -2.77]*`), UB = `54.93` (`+0.57 [-0.15, +1.30]`).
  - **S2 (Fix last-window omission & land at actual completion frame `t`):** `7,387` SHT windows complete after `start+23` (up to `41` frames later). SHT = `73.57`, UB = `55.02`.
  - **S3/S4 (Strict Causal Train-Calibrated Sentinel & Hold, selected on `dev` only):** Replacing global test-set look-ahead fill (`-0.532` SHT / `-0.553` UB) with train-only sentinel (`SHT: max = -0.5122` on `ML0_mps_calib12`, `hold H=0`; `UB: p50 = -0.8273` on `released_all`, `hold H=0`):
    - SHT `S3=S4` = **`73.58`** (`dev 72.41`, `conf 74.02`, fresh frame coverage `80.8%`). Note: `EXP-023`'s `-0.7886` sentinel sat at the `83.7th` train percentile, unfairly penalizing `23.5%` of normal test frames.
    - UB `S3=S4` = **`58.04`** (`dev 54.76`, `conf 59.66`; **`+3.02 [+1.12, +5.13]* pt`** better than test-max fill `S2`, because UB has `23.2%` empty frames and test-max fill sits far above normal scored frames).
- **Overlapping-Window Reuse (`R5` PASS):**
  - Caching per-frame `(x, y, conf)`: **`max_abs_diff = 0.0`**, **`18.5x`** (SHT) / **`13.2x`** (UB) fewer pose evaluations.
  - Freezing window normalization stats across shifts: SHT AUC drops `73.57` $\to$ `65.61` (first-window) / `58.88` (whole-track); UB drops `55.02` $\to$ `50.33` / `47.52`.
  - Reusing first flow layer hidden states on shared frames: `max_abs_diff = 0.838` (SHT) / `2.067` (UB). **Invalid.**

### 3.2 `Q2-P1-E01` — Offline Quality, Small-Person Recall & Two-Stage Tracking
- **Frontend Comparison (Causal `S4` / Offline `S0`, all 107 SHT clips, default tracker):**
  - `released` (oracle): `80.85` (`S0 85.94`) | `stock@640`: `66.03` (`S0 68.53`) | `stock@800 NCNN`: `67.47` (`S0 70.65`) | `stock@960`: `68.09` (`S0 71.73`) | `M00`: `61.89` (`S0 61.66`) | **`ML0_ncnn@640`: `73.78` (`S0 78.00`)** | `MLP_ncnn@640`: `74.69` (`S0 79.24`).
  - Paired `S4` contrasts (SHT): `ML0_ncnn - stock@640 = +7.75 [+4.63, +10.47]*`; `ML0_ncnn - stock_ncnn800 = +6.31 [+3.23, +9.17]*`; `ML0_ncnn - M00_ncnn = +11.89 [+8.48, +15.12]*`; `ML0_ncnn - ML0_mps = +0.20 [-0.04, +0.48]` (NCNN FP16 lossless); `MLP_ncnn - ML0_ncnn = +0.91 [-0.02, +1.81]` (`dev +1.96`, missed `+2.0` lock margin, `conf +0.30`).
  - **UBnormal (`S4` MPS proxy):** `released 71.58`, `stock@640 57.72`, `ML0_mps 58.04` (`+0.32 [-3.43, +3.69]`), `MLP_mps 57.00`.
- **Small-Person Recall & Miss Decomposition (against `187,021` AlphaPose reference person-frames):**
  - Recall (`<32 / 32-48 / 48-64 / 64-96 / 96-160 / >=160 / all`):
    - `stock@640`: `0.1% / 0.7% / 5.7% / 61.3% / 96.3% / 99.1% / 39.6%` (`<48 px: 0.4%`).
    - `ML0_ncnn`: `17.2% / 54.1% / 81.2% / 88.6% / 96.1% / 98.5% / 72.9%` (`<48 px: 37.0%`).
  - Miss taxonomy: **`84.2%`** of `ML0_ncnn`'s `50,657` misses are *intermittent gaps within seen tracks* (vs `49.5%` for `stock`), and **`52.3%`** (`48.5%` inside intermittent tracks) have a raw cached box in `[0.05, 0.25)` discarded before BoT-SORT!
  - Track fragmentation (default tracker): `ML0` generates `2,006` tracks (`48%` stranded `< 24` poses) vs `1,415` (`26%`) for `released`.
- **Dev-Only 36-Cell Tracker Sweep (`test_dev` 35 clips):**
  - Locked winner: **`c=0.20, track_low_thresh=0.05, track_buffer=30, match_thresh=0.85`** (reduces tracks `2,006 -> 1,766`, raises scored frame coverage `80.8% -> 84.4%`).
  - Effect (`default -> locked`, `S4`):
    - SHT `ML0_ncnn`: `dev 72.39 -> 77.41` (`+5.01 [+2.64, +7.58]*`), **`confirmation 74.32 -> 76.36` (`+2.04 [+0.21, +3.93]*`)**, `all 73.78 -> 76.82` (`+3.04 [+1.44, +4.71]*`; `S0 all = 81.62`).
    - UB `ML0_mps`: `all 58.04 -> 61.08` (`+3.04 [+0.72, +5.71]*`, `conf +2.73 [+0.44, +5.38]*`).

### 3.3 `Q2-P1-E02` — Fixed Baseline, VM Stage Profiling & Time-Causal Queueing
- **VM Isolated Stage Profile (74 cells, `edge-os` 4 vCPU ARM64, natural L0 frames, host CPU outside VM `17.2%` mean / `22.9%` max):**
  - **End-to-End Locked Pipeline (`T=4`):**
    - SHT (856x480 H.264): Decode `0.38 ms` + Pose `11.67 ms` (`78.2%`) + Track `0.42 ms` + Score `2.46 ms` = **`14.93 ms/frame`** (`p95 16.96`, `p99 22.80`, `67 fps` capacity, **`896 MB` peak RSS**, `1176 CPU-ms/video-s`; `T=1`: `31.62 ms/frame`).
    - UB (1080x720 H.264, `448x640` detector): Decode `0.81 ms` + Pose `16.15 ms` + Track `0.37 ms` + Score `2.88 ms` = **`20.21 ms/frame`** (`p99 40.4`, `49 fps` capacity, **`922 MB` peak RSS**, `1958 CPU-ms/video-s`; `T=1`: `40.53 ms/frame`).
  - **Detector Scaling (`ML0 NCNN FP16`):** `T=1: 27.0 ms` | `T=2: 15.8 ms` | `T=3: 13.7 ms` | `T=4: 10.8 ms` (`137 MB` detector RSS). `MLP` `T=4`: `19.6 ms` (`1.82x`). `stock@800` `T=4`: `16.6 ms` (`1.56x`). `ORT FP32` `T=4`: `27.6 ms` (`2.6x` slower than NCNN FP16). `ORT INT8` `T=4`: `12.4 ms` (`49.3 CPU-ms` vs `38.8 CPU-ms` for NCNN FP16).
  - **Service-Model Validation (`P3`):** Empirical pool model under-predicts measured end-to-end VM frame time by `5.9%–9.6%` (within `+-10%` band) in `E02`, and by `1.36x–1.39x` against the full online Python `E05` loop (`s=1.0`).
- **True Time-Causal Delivery & Sustainable Workload (`2.0 s` max wait cap, continuous stream):**
  - Simulated `s*`: `K=1` stays sustainable up to **`s* = 3.08`** (SHT) and **`s* = 1.66`** (UB). At `s=3.0`, SHT `K=1` collapses (`66.59` vs `76.60` at `s=2.0`); at `s=2.0`, UB `K=1` collapses (`52.29` vs `61.00` at `s=1.0`).
  - **Compliant `Fixed-Burst` vs. Non-Compliant `Fixed-Stride` (`H1`):** Sub-sampling without fill (`stride >= 2`) yields **0 windows**. Propagated strides (`CV`/`ZOH`) violate "never relabel an old pose" (`genuine fraction 0.25–0.50`), whereas `Fixed-Burst` (`L` consecutive real frames every `W` frames, `genuine = 1.0`) loses `4.6–7.2 pt` (SHT) / `2.5–2.8 pt` (UB) to `stride-CV` at equal detector duty. Scoring thinning (`W <= L`, full detector duty) saves at most `15%` CPU (SHT) / `9%` (UB) (`H3`).

### 3.4 `Q2-P1-E03` — Core Ablation (`Fixed` vs. Scheduling vs. Reuse vs. Joint)
- **Design:** 12 cells (`s=2.0` $\times$ `beta in {0.25, 0.33, 0.50, 0.67, 0.75, 1.00}` + overload `s in {2.5, 3.0}` $\times$ `beta in {0.50, 0.75, 1.00}`). All arms are **budget-matched to total CPU** (`target = beta * C_full(s)`, tolerance `[0.97, 1.005]`, including all observer/check overheads). Hyperparameters tuned on `test_dev` only; verdicts on `test_confirmation`.
- **6 Arms (all compliant, intact $L$-frame windows, per-frame `(x,y,conf)` cache shared by all):**
  1. `Fixed`: `Fixed-Burst(W)`.
  2. `Motion-Only`: cheap 4$\times$-downsampled frame-diff observer (`0.37 ms/frame` SHT) triggers windows under token bucket.
  3. `Adaptive-Only`: multi-signal (motion + exterior border motion + score feedback + observation age + queue-backlog guard) scheduler.
  4. `Cache-Only` (`C1 + C2`): `Fixed-Burst` + **C1** (skip detector when ROI mean-abs-diff & exterior p95 are static, reusing previous detections up to `A1=4` frames) + **C2** (skip STG-NF when pose trajectory shift is tiny).
  5. `Independent-Combo`: `Adaptive-Only` + `Cache-Only` combined with independent dev-tuned thresholds.
  6. `Joint-Controller`: dynamically tightens/relaxes C1/C2 validity and trigger thresholds based on anomaly feedback and queue wait, reinvesting saved CPU into fresh windows.
- **Results (`test_confirmation` 12-cell mean AUC, paired video bootstrap):**
  - **SHT:** `Cache-Only` **`73.45`** | `Independent-Combo` `73.06` | `Joint-Controller` `71.86` | `Fixed` `71.06` | `Adaptive-Only` `70.89` | `Motion-Only` `69.71`.
  - **Contrasts (SHT `confirmation`):**
    - `H1`: `Adaptive - Fixed = -0.17 [-0.99, +0.65]`; `Adaptive - Motion = +1.17 [+0.14, +2.33]*` (**H1 partially supported**; neither scheduler beats Fixed).
    - `H2`: `Cache-Only - Fixed = +2.41 [+1.39, +3.48]*`; `C1 - Fixed = +2.39 [+1.39, +3.46]*` (**28.5%** net CPU saving, check overhead `4–9 ms/s` vs `340–640 ms/s` saved); `C2 - Fixed = +0.32 [+0.03, +0.73]` (`0.3%` CPU saving $\to$ **C2 dropped**). (**H2 supported on SHT, not on UB** `+0.18 [-1.03, +1.48]` despite `20%` CPU saving).
    - `H3` & Stop Rule: `Joint - Cache-Only = -1.60 [-2.43, -0.82]*`; `Joint - Independent = -1.20 [-2.06, -0.34]*`. **H3 not supported; Joint-Controller REJECTED; `Cache-C1` locked for E04/E05** (and `Independent-Combo` flagged as UB/overload reference, leading to `Cache-C1+Guard` in E05).

### 3.5 `Q2-P1-E04` — Optional Optimizations (`+MV`, `+INT8`, GPU Gate)
- **Step 1 (`+Codec MV`):** Extracted real H.264 macroblock MVs (`x264 IPPP`, 318/318 clips). VM decode rises `0.039 -> 0.175 ms/frame` (wall) / `0.374 CPU-ms/frame` export overhead charged on *every* frame; MV C1 check `0.114 ms` vs pixel `0.373 ms`. Motion correlation `r = 0.82` (SHT) / `0.77` (UB), static agreement `84.2%` / `74.2%`.
  - **Matched-Cost Decision:** **Not adopted.** SHT `dev -0.03 [-0.61, +0.56]`, `conf -0.08 [-0.48, +0.32]` (Path B CPU `-2.8%`). UB `dev -2.25 [-3.99, -0.41]*`, `conf -0.57 [-1.83, +0.70]`. MV pose advection (`C1-MV-Advect - C1-MV`): `-0.06 [-0.15, +0.04]` (SHT), `+0.07 [+0.01, +0.13]` (UB).
- **Step 2 (`+ORT Static-QDQ INT8`, 200 seeded SHT normal train frames, `--decode fp32`):**
  - **Detector quality:** Lossless (`ML0_ort_int8 - ML0_ncnn`: `S0 -0.20 [-0.99, +0.51]`, `S4 -0.19 [-0.90, +0.43]`, `<48 px` recall `37.2%` vs `37.0%`).
  - **VM Timing (`ML0` @ 640):** `T=1: 26.4 ms` (`NCNN FP16 26.4 ms`, `ORT FP32 75.7 ms`); `T=4: 12.0 ms wall / 47.6 CPU-ms` vs **`NCNN FP16 10.3 ms wall / 37.5 CPU-ms`**. Thus ORT INT8 is `2.28x–2.86x` faster than ORT FP32, **but `21%` higher CPU than NCNN FP16**.
  - **Matched-Cost Decision:** **Not adopted.** On `B0 (Cache-C1)`: Path A `dev -1.37 [-3.04, +0.46]`, `conf -1.85` pt; Path B total CPU `-22.8%` (= 22.8% *more* CPU at equal schedule) and `-1.06 pt`.
- **Section 7 / 8 GPU Training Gates:** QAT = False; Cadence-aware scorer = False (`+0.13` to `+0.24 pt` shift); `$0 / $10` RunPod budget spent.

### 3.6 `Q2-P1-E05` — Full Confirmation, Non-Oracle Tracking, Subgroups & Sustained VM Run
- **Headline Ladder (`test_confirmation`, 12-cell mean, 2,000-resample paired video bootstrap):**
  - **SHT:** `Stock-Baseline` `60.78` $\to$ `Fixed-Burst` `71.06` $\to$ **`Cache-C1` `73.43`** (`Cache-C1+Guard` `73.38`, `C1-ReTrack` `72.54`).
    - `V1` (`Cache-C1 - Fixed`): **`+2.37 [+1.38, +3.41]*`** (`all: +2.59 [+1.73, +3.47]*`) $\to$ **CONFIRMED**.
    - `V3` (`Cache-C1 - Stock-Baseline`): **`+12.65 [+7.96, +17.41]*`** (`all: +12.36 [+9.01, +15.83]*`) $\to$ **CONFIRMED** (`recall@2s +13.9 [+8.6, +19.8]*`, missed events `-16.7 [-23.6, -10.6]*`, delay p50 `-337 ms`).
    - `V4` (`C1-ReTrack - Fixed`, removing oracle identity): **`+1.48 [+0.24, +2.62]*`** (`62%` retention) $\to$ **HOLDS WITHOUT ORACLE IDENTITY**.
  - **UBnormal:** `Stock-Baseline` `53.74` $\to$ `Fixed` `55.82` $\to$ `Cache-C1` `56.00` (`+0.18 [-0.99, +1.39]`, `V2` non-significant/non-negative) $\to$ **`Cache-C1+Guard` `59.45`** (`C1-ReTrack+Guard` `59.69`).
    - `V5` (`Guard`): Non-inferior in non-overload cells (`+0.00 [+0.00, +0.00]`); in 7 overloaded UB cells **`+5.93 [+1.73, +9.92]*`**. Mild-overload cost on SHT `s3.0_b1.00`: `-0.64 [-1.29, -0.14]*`.
- **Event Subgroups (Budget cells `s=2.0, beta in {0.33, 0.50, 0.75}`, `Cache-C1 - Fixed`):**
  - **V7 New Entries (SHT `n=144`, UB `n=100`):** **Non-inferior & not delayed** (`SHT recall@1s -0.2 [-2.8, +2.6]`, `delay -139 [-403, +83] ms`, `localised AUC +3.61 [+1.94, +5.38]*`; `UB Cache-C1+Guard +6.0 [+2.2, +9.8]*`).
  - **V6 Short Events `<= 2.5 s` (SHT `n=81`, UB `n=26`):** **Inconclusive / not cleared** against `-5 pt` non-inferiority margin (`SHT -3.3 [-8.7, +2.2]`, `delay +486 [+0, +1222] ms`, though `localised AUC +1.05 [-1.77, +3.70]`; `UB -6.4 [-14.5, +2.4]`).
  - **Crowd `>= 4` tracks (SHT `n=105`):** Localised AUC `+3.24 [+1.06, +5.52]*` (budget) and `+10.37 [+4.32, +17.00]*`, `recall@1s +3.8 [+0.9, +8.0]*` at `s3.0_b1.00`.
- **Sustained Operation (`V8` Simulation & `V9` Real `edge-os` VM 11,520-frame Runs):**
  - **V8 (2.36 h SHT / 51.5 min UB):** `Cache-C1+Guard` bounded in `8/8` cells; unguarded `Fixed-K1` / `Cache-C1` diverge in 7 cases (max wait up to `2,497 s`).
  - **V9 (Real End-to-End VM, 11,520 frames, 4 threads, non-oracle BoT-SORT):**
    - `Cache-C1+Guard`: `s=1.0`: `24.0 fps` achieved (`62.8 fps` busy capacity vs `47.6 fps` for `Fixed-K1`, **`25.5%` VM CPU reduction** `1182 vs 1587 ms/s`), `p99 wait = 4 ms`, `AUC 82.96`, `RSS growth 0.27%` (`901 MB` peak). `s=3.0` (`72 fps` offered) & `s=3.5` (`84 fps` offered): **`p99 wait = 405 ms / 408 ms`**, `RSS growth = 0.37% / 0.33%` (whereas `Fixed-K1` collapses to `p99 = 195.8 s / 306.6 s`, dropping `9,400+` windows, and unguarded `Cache-C1` at `s=3.5` collapses to `p99 = 115.1 s`).
    - **Honest Overload Caveat:** Real VM Python loop cost is **`1.36x–1.39x`** the E02 simulation pools (`s* ~ 2.0` vs `3.08`), so at `s=3.0/3.5` the dense `W=1` `Cache-C1+Guard` keeps latency bounded (`<= 408 ms`) by shedding `8.9% / 12.7%` of frames, which drops real delivered AUC on the 12 looped clips from `82.96` $\to$ `64.37 / 55.95`.

---

## 4. Direct Q&A for Authoring & Defense

### Q: Does fine-tuning on small people (`ML0`) only inflate AUC by detecting more *normal* people, rather than actually detecting anomalies better?
**Short Answer:** **No on ShanghaiTech; partially yes on UBnormal—and we have direct empirical proof separating both effects.**
1. **Why the question matters (Mechanism):** In frame-level ROC-AUC, missed frames receive the normal fallback score (`most_normal` offline, train sentinel online). As proved in `EXP-021-coverage-bound`, frame-level AUC is mathematically bounded by anomalous-frame coverage $c_A$: $\text{AUC} \le c_A + \frac{1}{2}(1-c_A)(1-c_N)$. In fact, detecting *more normal people* does **not** automatically help AUC—empty frames already receive the `"max"` normal sentinel (`-0.5122` on SHT), so detecting more normal pedestrians actually *increases* false-alarm opportunities (`FA/h` rises from `11.4` in `Stock-Baseline` to `28.5` in `Cache-C1`).
2. **Evidence on ShanghaiTech (Real Anomaly Recovery):**
   - `ML0` raises **anomalous-frame coverage $c_A$ from `67.5%` $\to$ `91.0%`** (`+23.5 pp`), while normal-frame coverage rises `53.4%` $\to$ `72.5%` (`E00`).
   - Crucially, **`E05` evaluated two metrics independent of normal background frames:**
     - **Event-level recall@2s (`confirmation`):** rises from `16.4%` (`Stock-Baseline`) / `26.7%` (`Stock-Default-K1`) to `30.3%` (`Cache-C1`) / `34.9%` (`Full-Rate`), cutting missed anomalous events from `67.9%` $\to$ `51.2%` (`-16.7 [-23.6, -10.6]* pp`) and p50 delay by `-337 ms`.
     - **Localised Event-Window AUC (computed strictly inside `[event_start - 2s, event_end + 2s]`):** jumps from **`51.5`** (`Stock-Baseline`) $\to$ **`62.2`** (`Cache-C1`) $\to$ **`68.0`** (`Full-Rate`). Because background normal clips/segments are excluded, `ML0` is genuinely recovering anomalous actors at small pixel scales.
3. **Evidence on UBnormal (Where the Hypothesis *Does* Bite):** On UBnormal, `ML0` increases *normal* coverage more (`70.5% -> 80.1%`, `+9.6 pp`) than *anomalous* coverage (`70.1% -> 73.3%`, `+3.2 pp`), and noisier small-person detections in synthetic fog/fire scenes trigger higher false alerts without lifting offline AUC (`54.37` vs `55.76` for `stock`), explaining why `ML0` yields no significant advantage over `stock` on UBnormal (`+0.32 pt` in `E01`).

---

## 5. Artifact & Figure Map for Writing
- **Configs & Reports:** `experiments/Q2-P1-E0{0..5}_*/{plan.md,config.yaml,report.md,summary.json,results.csv}`
- **Generated Figures (ready to embed/redraw):**
  - `experiments/Q2-P1-E03_ablation/fig_ShanghaiTech_matched_cost_curves.png`
  - `experiments/Q2-P1-E03_ablation/fig_UBnormal_matched_cost_curves.png`
  - `experiments/Q2-P1-E05_confirmation/fig_ShanghaiTech_ladder.png`
  - `experiments/Q2-P1-E05_confirmation/fig_UBnormal_ladder.png`
  - `experiments/Q2-P1-E05_confirmation/fig_c2_queue_traces.png`
  - `experiments/Q2-P1-E05_confirmation/fig_vm_traces.png`
- **Unit Test Coverage:** `205` passing tests across `experiments/tests/test_q2_p1_e0{0..5}.py`.
