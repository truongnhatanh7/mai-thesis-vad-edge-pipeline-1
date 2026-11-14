# Q2-P1-E04 — Optional optimizations (plan, written before any E04 delivered-AUC or decision number was computed)

**Lane:** `Q2-P1-*`. **Source plan:** `Edge Anomaly Detection Experiment Plan.md`, ladder row **E04 — Optional optimizations**.
**Builds on (read first):** `Q2-P1-E00_audit` (causal S3/S4 protocol, splits `test_dev` / `test_confirmation`), `Q2-P1-E01_quality`
(locked frontend ML0 @640 and locked two-stage tracker `c=0.20, low=0.05`), `Q2-P1-E02_fixed_baseline` (VM stage-timing pools, true
time-causal FIFO queue, 2.0 s late-work cap), `Q2-P1-E03_ablation` (locked candidate **`Cache-C1`**: fixed burst schedule `F(W)` + C1
static-validity detector-output reuse at level 3; 12 budget/overload cells).
**Training:** none. INT8 uses **post-training static quantisation (calibration only)**; QAT / retraining is a separate, gated experiment
(section 8). All thresholds below are derived from `test_dev` and label-free statistics; `test_confirmation` is only used to verify.

> Question (source plan): *Add codec motion vectors, then pose-model INT8, one at a time. Required output: each component's net
> benefit after overhead and accuracy loss.*

### Hardware disclosure (binds every sentence in the report)
The physical Pi 5 is **not connected**. CPU costs are VM measurements (`edge-os-arm64-vm`, Lima/QEMU-class VM on the Apple host) mapped to
a Pi by the E02/E03 multiplier `s in {2.0, 2.5, 3.0}`; delivered AUC is *simulated delivery on replayed detections*. Two caveats specific to
E04: (i) the VM exposes `asimddp`, `i8mm` and `bf16`, the Pi 5's Cortex-A76 only has dot-product: INT8 speed-ups measured in the VM are an
**upper bound** of what a Pi 5 will see; (ii) the MV streams are **our own x264 re-encode** (IPPP) of the dataset frames: the datasets were not
shipped as H.264 camera streams, so MV statistics describe a plausible camera encode, not the original capture.

## 1. The two single changes

| step | change vs the previous locked configuration | everything else |
|---|---|---|
| **B0** | E03 `Cache-C1` (level 3) and E03 `Motion-Only`, exactly replayed (rule E4-D1) | — |
| **+MV** | the *pixel-difference* signal (4x down-sampled luma difference; 24x40 thumbnail ROI / exterior check) is replaced by the **H.264 macroblock motion vectors** exported by the decoder (`export_mvs`) | frontend, tracker, scorer, schedule family `F(W)`, level index (3), `A1 = 4`, quantile definition, queue, pools |
| **+INT8** | the pose network ML0 @640 runs as a **static-QDQ INT8 ONNX Runtime model** instead of NCNN FP16 | the (adopted or not) result of +MV, tracker, scorer, schedule family, level 3 thresholds, queue |

`+INT8` is evaluated on top of **+MV only if +MV is adopted**; otherwise on top of B0. Both are reported on both bases when cheap.

## 2. Step 1 — +MV

### 2.1 MV extraction
* Stream: x264 `ultrafast`-class IPPP (`bframes=0, ref=1, scenecut=0, keyint = 2 s, threads=1`), PyAV `export_mvs`. All **107** ShanghaiTech test
  clips (JPEG frames re-encoded) and all **211** UBnormal test mp4s (decoded, re-encoded: second generation). One MV record per frame.
* Per macroblock (16x16): area-weighted displacement `(dx, dy)` of the content, magnitude `m = |(dx, dy)|`; skip macroblocks are exported (verified on
  `01_0014`: 98.9 % cell coverage), an uncovered macroblock is intra (`m := 16 px`); I frames carry no MVs (**invalid**, statistics NaN).
* Cached statistics per frame (`~/thesis-data/q2-p1-e04/mv_cache/`): `mv_p90` (90th pct of `m`), `mv_max_roi_p90` (see "stat names" in 2.5),
  `mv_border_p90` (90th pct of `m` in the outer 10 % band), and the full grids (for C1 / advection).

### 2.2 Cost profile (VM, `limactl shell edge-os`)
Per frame wall and process-CPU, T in {1, 2, 3, 4}, staged clips: plain H.264 decode; decode + MV export; BGR conversion; gray 4x
down-sample; MV grid build; MV C1 / observer statistics; the pixel counterparts (observer 0.37 ms, C1 check); `cv2` grab / retrieve.
**Charged in the simulation (primary accounting):**
1. *MV export overhead on **every** frame* (the export flag is codec-wide): `decode_mv - decode_plain` at `T = 4`;
2. MV grid + statistic on in-run frames (C1) / on every frame (Motion-Only observer);
3. the pixel observer / thumbnail check is **not** charged for MV arms (the pixels are not needed to decide);
4. **retrieve-skip credit**: on a C1-reused frame the BGR conversion (`retrieve`) is not paid (no pixel is needed), booked as saved work;
5. all MV costs x **1.25** safety margin (E03 used measured + ~5 %; the VM Python path is noisier), scaled by `s` like every stage.
Sensitivity: MV costs x 1.0 and x 2.0; pixel costs at the VM-measured values instead of E03's constants.

### 2.3 Arms (all `test_dev` / `test_confirmation` / `all`, 12 cells, paired video bootstrap 1000 resamples, seed 0)
* `C1-Pixel` = E03 `Cache-C1` (replayed bit-exactly) — baseline.
* `C1-MV` = same fixed-burst schedule, same level 3, validity from MVs: for source frame `s` and age `k = 1..4`, accumulated magnitude
  `sum_{j=s+1..s+k} m_j`; static iff the **mean inside the person ROIs of `s`** (boxes dilated one MB) `<= tau_roi` and the **95th pct outside** `<= tau_ext`;
  invalid (an I frame in `(s, s+k]`) -> the detector runs. Thresholds = the same 10 / 25 / 50 % quantiles of the **`test_dev`** one-frame MV
  statistics (label-free), level 3 locked.
* `C1-MV-Advect` = `C1-MV` plus **pose advection**: the reused pose is shifted by the accumulated mean MV displacement inside the source person box
  (instead of zero-order hold). Compared against `C1-Static` (zero-order hold, = `C1-MV` and `C1-Pixel`).
* `Motion-Only-Pixel` = E03 `Motion-Only` (selected config replayed); `Motion-Only+MV` = identical config and quantile `theta` but the observer is the
  MV statistic (`mv_motion`, border `mv_border`; carry-forward over I frames).
* Diagnostics: correlation `r` (Pearson / Spearman) of pixel vs MV statistics and **agreement** of the C1 static decision (level 3, age 1);
  C1 hit (reuse) rate; pose reuse error (height-normalised ZOH error histogram, advected error); net CPU overhead (ms / video-second);
  sensitivity: levels 1, 2; MV cost x 1.0 / x 2.0.

### 2.4 Matched cost
Every cell's CPU target is **E03's absolute target** `beta * C_full(s)` (E03 `c_full`, pixel pipeline). An optimisation that costs more CPU per
frame must therefore run a sparser schedule, one that saves CPU can run a denser one (**Path A**). **Path B** keeps the schedule period fixed at E03's
calibrated `W` (same knob) and reads the CPU change and the AUC change directly.

### 2.5 Disclosed mapping
The source plan names the cached statistics `mv_p90`, `mv_max_roi_p90`, `mv_border_p90`. `mv_max_roi_p90` is the maximum over the person ROIs of the 90th percentile of
`m` in a ROI; it needs boxes, so it is stored per (clip, frame) by `stats` after the detector cache is read (C1's `roi_mean` / `ext_p95` are the statistics actually used for decisions).

## 3. Step 2 — +INT8
* `quantize_int8.py`: ML0 @640 (`384x640` ONNX FP32 from `artifacts/EXP-020/ML0.pt`) -> static QDQ INT8, **200 seeded random ShanghaiTech *training*
  frames, seed 0, `--decode fp32`, `--method minmax`** (decoder head kept FP32). Calibration frames are disjoint from the test set (rule E4-D5).
* Full 107-clip detection cache `~/thesis-data/q2-p1-e04/cache/L0_E0-ML0-int8-640` (`edgelab.pipeline.ort_pose`, 8 shards, 1 thread each); controls: ML0 ORT FP32 and stock YOLOv8n-pose
  NCNN FP16 (@640), existing stock INT8 caches (`exp012`, @640 and @800).
* VM profile of the pose stage T in {1, 2, 3, 4}: ML0 NCNN FP16 (deployed baseline), ML0 ORT FP32 (isolates quantisation from runtime), ML0 ORT INT8; stock NCNN FP16 vs stock ORT INT8 @640 / @800.
* Quality (locked tracker `c=0.20, low=0.05` and E01 default `c=0.25`): offline S0 AUC, causal S4 AUC, recall by person height (<48 px, 48-96 px); **delivered AUC inside `Cache-C1`** with INT8
  detections (rebuilt run tables, same locked C1 thresholds) and the INT8 pose service pool, 12 cells, same matched-cost definition as 2.4.

## 4. Pre-registered adoption rule (source plan, operationalised here before any result)
A component is **adopted** iff, for the dataset in question,
* **Path A (accuracy):** cell-averaged delivered-AUC change at E03's matched absolute CPU target is `> 0` on `test_dev` with the 95 % paired video-bootstrap CI excluding 0, **or**
* **Path B (cost):** at an identical schedule the cell-averaged CPU cost falls by **>= 5 %**,

**and** (both paths) delivered AUC is **not degraded by more than 0.25 pt** (point estimate, cell average) on `test_dev`; **and** the finding is **verified on `test_confirmation`**: same direction
(point estimate `>= 0` for Path A; CPU reduction `>= 5 %` for Path B) and no degradation worse than 0.25 pt; **and** reproducible (section 6). Otherwise the component is **not adopted** (a
component that is neutral stays out: simplicity). A dataset-conditional result (adopted on one dataset only) is reported as such and not generalised. For INT8 the "CPU cost" is the pose-stage and total
CPU at an identical schedule; for +MV it is total CPU including every overhead in 2.2.
The decision rule is evaluated once per component on the full 12-cell grid and not re-tuned after seeing it; sensitivity runs are reported but cannot flip a decision.

## 5. Correctness rules (E4-D)
* **D1** E03 `Cache-C1` replayed through the (backward-compatible) extended simulator is bit-exact (delivered AUC and cost) in all 12 cells, both datasets.
* **D2** one MV record per labelled frame in every clip; MV extraction recovers a known global translation (unit test); MV encodes are deterministic (re-encode of 3 clips gives identical grids).
* **D3** numpy copies of the E03 observers (used in the VM) equal the E03 functions (unit test).
* **D4** VM pose pool for ML0 NCNN FP16 agrees with E02's pool within 10 % (session drift check); host outside the VM <30 % CPU during profiling.
* **D5** INT8 calibration: 200 frames, seed 0, from `training/videos` only; re-quantising reproduces the identical file hash.
* **D6** C1 reuse never exceeds age `A1`, never uses a frame after the current one (causality), MV `NaN` at invalid frames compares False.
* **D7** INT8 cache: 107 clips, one record per frame, frame ids 0..n-1; INT8 cache re-run of 2 clips is identical.

## 6. Reproducibility
Fixed seeds (`SEED`, `POOL_SEED`, bootstrap seed 0), single-threaded deterministic x264, deterministic ORT (1 thread per shard), all hashes of inputs in `config.yaml`; `uv run pytest tests/test_q2_p1_e04.py`.

## 7. Outputs
`plan.md`, `config.yaml` (inputs, hashes, rules, then the measured costs and `test_dev` thresholds *before* any delivered-AUC run), `results.csv`, `decisions.jsonl`, `summary.json`, `report.md`.
Order of work (disclosed): the plan and the rule text above were written first; input preparation that produces no metric (ONNX export + INT8 calibration, H.264 re-encode + MV extraction, ORT/NCNN detector
caches) was started in parallel with writing this file and finished before `lock`; VM cost profiling (a measurement of the machine, never of accuracy) precedes `lock`; no delivered AUC, recall or decision number existed when `config.yaml` was finalised.

## 8. Section-7 gate check (source plan) — evaluated in the report
* **QAT**: only if INT8 degrades quality (S0/S4 or delivered AUC) beyond 0.25 pt while costing less CPU (i.e. there is a speed-up worth rescuing).
* **Cadence-aware scorer** (train STG-NF on sparse/propagated cadence): only if reuse (C1 errors, burst gaps) shows a measurable scorer error (reuse error -> AUC) that the current scorer cannot absorb.
* **Longer fine-tuning of the pose model**: only if missed people (recall by height) are shown to limit delivered AUC (gap between detector-recall-limited and ideal series).
A RunPod GPU experiment is recommended only if a gate fires **and** the expected gain exceeds the 0.25 pt adoption margin; otherwise "no GPU experiment warranted".

## 9. Known limitations (declared in advance)
VM not Pi; MV re-encode second generation for UBnormal; the VM has ISA extensions the Pi 5 lacks (INT8 upper bound); C1 track identity is an oracle (as E03); one pose model; no training; the MV decode path (PyAV) differs from the
`cv2` decode path used by the E02 pools, so only the **delta** (export + MV statistics) is added to the E02 pools.
