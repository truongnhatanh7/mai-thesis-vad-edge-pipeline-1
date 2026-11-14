# Q2-P1-E07 — YOLO26-pose → TrackTrack → {STG-NF, LoRaD, TrajVAD-T, TrajVAD-P}

Written before any E07 number was looked at.

## Question
Does a 2026 edge front-end (YOLO26n-pose, NMS-free) + a 2025 tracker (TrackTrack, CVPR'25) + a causal pose/trajectory flow
(LoRaD, TrajVAD-P ECCV'26) beat the locked E01 pipeline (YOLOv8n-pose → BoT-SORT 2-stage → frozen STG-NF), and does it keep up on a
Raspberry Pi 5 (real time, with Cache-C1+Guard)?

## Pipeline
`frontend ∈ {YOLO26n-pose e2e, YOLO26n-pose o2m, stock YOLOv8n-pose@640, ML0 YOLOv8n-pose@640}`
→ `tracker ∈ {TrackTrack (tuned), BoT-SORT 2-stage (E01 locked)}`
→ `scorer ∈ {STG-NF (frozen), LoRaD (E06 selected), TrajVAD-T, TrajVAD-P}`.
Datasets: ShanghaiTech (107 test clips: 35 dev / 72 confirmation), UBnormal (211: 70 / 141). Protocols S0 (offline) and S4 (strict causal), E00 definitions.
UB has no o2m cache and no ML0-NCNN cache → UB frontends: YOLO26 e2e, stock, ML0_mps (proxy; stated in the report).

## TrackTrack (Ultralytics 8.4.201 `tracktrack`)
HMIoU cost, iterative assignment, track-aware initialisation. gmc=none, no ReID (same as the BoT-SORT control).
Ultralytics defaults (high 0.6 / new 0.7) starve pose detectors; tuned **on test_dev only (35 SHT clips)**:
`track_high ∈ {0.20,0.25,0.35}`, `track_low ∈ {0.05,0.10}`, `match ∈ {0.70,0.80}` (12 cells) × 3 frontends (YOLO26 e2e, YOLO26 o2m, ML0_ncnn);
`new_track = track_high` (as in the BoT-SORT control), `min_track_len` 3, buffer 30 fixed.
Controls: BoT-SORT default and BoT-SORT E01 locked 2-stage.
**Selection rule T1:** the cell with the highest mean over the 3 frontends of dev S4 AUC (frozen STG-NF); ties (<0.3 pt) → fewer stranded windows, then higher coverage. One cell for all frontends and both datasets, then locked.

## TrajVAD (ECCV'26) causal port — `models/trajvad.py`
Model classes are the authors'. Edge changes (documented): person class only; box = keypoint min/max (no detector box needed);
CV Kalman filter run forward only; first-difference features start at 0; no gap interpolation; every feature of pose t depends on poses ≤ t
(prefix-invariance unit test). 27 kinematic features. TrajVAD-P adds a conditional 51-d pose flow (pelvis-centred, confidence-gated).
Train on the normal train tracks of SHT (330 clips, L=16) / UB (186 clips, L=12): 10 epochs, RMSprop lr 5e-4, batch 256, 3 seeds.
Trained on Apple mps (the 36-layer P model is 2.6 s/batch on CPU, 0.07 s on mps); scored on CPU.

## Evaluation
Scores = seed-mean of the per-window normality for learned scorers. Paired video bootstrap, 2,000 resamples, shared weights across arms (E00 engine).
Per-scorer train-calibrated S4 sentinel. Reported for dev / confirmation / all.

## Pre-registered decisions (confirmation split unless stated)
- **T2** TrackTrack vs BoT-SORT locked, per frontend & dataset, S4 AUC paired difference: "better" iff CI lower bound > 0; "equal" iff CI ⊂ [−1,+1]; else "worse".
- **V1** LoRaD vs STG-NF; **V2** TrajVAD-P vs LoRaD; **V3** TrajVAD-P vs TrajVAD-T; same rule as T2. Reported per tracker × frontend, pooled verdict = number of cells better/equal/worse.
- **V4** Best pipeline = highest confirmation S4 AUC on SHT *and* UB averaged over the two datasets; ties broken by lower latency.
- **C1** Cache-C1+Guard is accepted on a pipeline iff S4 AUC change vs uncached ≥ −1.0 pt (CI lower bound ≥ −2.0) and zero causality violations.
- **P1** Pi 5: a pipeline "keeps up" iff delivered FPS ≥ 25 and guard-off queue wait does not diverge; report PMIC W, °C, RSS, stage latencies (decode / pose / TrackTrack / scorer), delivered real-time AUC.

## Amendments
* **A1 (before any E07 result was read; written while the sweep was running).** Verdict rule T2/V1–V3: "better" iff CI lower bound > 0; "worse" iff upper bound < 0;
  "equal" iff the CI lies inside [−1, +1] and neither of the former; otherwise "inconclusive" (the first draft said "else worse", which mislabels wide CIs).
* **A2 (cost, no result involved).** TrajVAD-P is 2.6 s/batch on the CPU (≈ 6.5 h for SHT) and 0.07 s/batch on Apple mps, so TrajVAD trains on mps (aten::mode runs on the CPU).
  Every train/eval number is from the saved checkpoint scored on the CPU.
* **A3 (design).** The Cache-C1+Guard effect on AUC is measured on the physical Pi 5 on the 12 ShanghaiTech clips on disk (all 2,903 frames), where the guard and the
  detector-reuse act on real latencies; no offline surrogate of the cache is used. Delivered AUC = E00 S4 landing (train-calibrated sentinel, hold 0) of the
  window normalities delivered on the Pi (a window later than 2 s is dropped, as in the Pi5-Huang runner).

## Pi 5 benchmark
Check `ps aux | grep python` on `nanh@raspberrypi.local`; run only when `q2_p1_pi5_suite` is not running (never co-load the CPU).
Copy `yolo26n-pose-640-384x640-ncnn` to `~/thesis-data/models/`, sync checkpoints + runner. Six arms:
YOLO26(NCNN)→TrackTrack→{LoRaD, TrajVAD-P} and ML0(NCNN)→TrackTrack→{LoRaD, TrajVAD-P}, each ± Cache-C1+Guard.
MLflow `http://127.0.0.1:5000`, experiment `Q2-P1-E07`, `@mlflow.trace` spans per stage; registry models
`edgelab-yolo26n-pose`, `edgelab-lorad-causal-flow`, `edgelab-trajvad-p`.

## Deliverables
`results.csv`, `summary.json`, `report.md`, `tests/test_q2_p1_e07.py`, `uv run pytest` green.

## Known limits (stated up front)
- Sentinel/threshold calibration of retrained scorers is in-sample on train tracks.
- UB frontends are proxies (no o2m / NCNN ML0 caches).
- TrajVAD features are heavy-tailed after StandardScaler (|z| up to ~500); we keep the authors' recipe (no clipping) and report it.
