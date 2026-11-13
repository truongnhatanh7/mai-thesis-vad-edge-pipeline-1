# Q2-P1-E00 — Audit (plan, written before any number was computed)

**Lane:** `Q2-P1-*` (new namespace; EXP-009 … EXP-023 are preserved and are only *read*).
**Source plan:** `Edge Anomaly Detection Experiment Plan.md`, ladder row **E00 — Audit**.
**Training:** none. Every checkpoint is frozen and hashed. **Hardware:** Apple-silicon laptop, CPU for STG-NF scoring
(the detector caches were produced earlier on MPS FP32 / NCNN FP16 and are only *replayed* here). No Raspberry-Pi claims.

## 1. The single audit question

> **Can we reproduce the existing scores, timestamps, normalisation and missing-data handling — and which part of the
> offline → causal-delivered AUC gap is real, and which part is a scoring-protocol artefact?**

Required output (from the ladder): *locked configuration, hashes, baseline report.*
This experiment changes **nothing** about the frontends or the scorer. Its only "treatment" is the *scoring protocol*
(how a stream of STG-NF window scores becomes one per-frame score), applied step by step.

## 2. Assumptions (stated up front, each one is verified by a rule in §5)

| # | Assumption | Verified by |
|---|---|---|
| A1 | Every frozen artefact referenced by earlier experiments still exists byte-for-byte. | R1 |
| A2 | The historical numbers (EXP-001/003/006/020) come from `edgelab.pose.score_dir` on the stored pose folders; re-running that code on those folders returns the same AUCs. | R2 |
| A3 | Stored pose folders are the deterministic replay (`c = 0.25`, BoT-SORT) of the stored detector caches. | R2b |
| A4 | STG-NF windows are `seg_len` consecutive **poses** of one track (≤ 2 missing frames allowed), stride 1 at test time; `seg_len = 24` (ShanghaiTech) and `16` (UBnormal). | R3 |
| A5 | STG-NF's test-time window generator drops the last window of every track (and every track with exactly `seg_len` poses). | R3, R4 |
| A6 | A per-frame pose `(x, y, conf)` depends only on that frame (+ the already-fixed track id), so it can be shared by all overlapping windows. STG-NF's `normalize_pose` is a function of *one window's* own mean/std, so anything *after* normalisation cannot be shared. | R5 |
| A7 | ShanghaiTech is 24 fps; UBnormal is 30 fps. EXP-023 hard-coded 24 fps for both. | R7 |
| A8 | Anything marked "causal" may only use information available at frame `t` (training data and the past of the same video). | R6 |

## 3. Data, splits and controls

* **ShanghaiTech test (107 clips, all contain an anomaly).** Video-grouped *development* (35 clips) vs *confirmation*
  (72 clips) split, stratified by scene, deterministic (`sha256(seed|clip)` ordering + systematic sampling, seed `20261113`).
  Normal training clips `01_001 … 12_001` (12 clips; train split only) are used **only** for label-free calibration.
* **UBnormal test (211 clips, 158 anomalous).** Official lists: train (186 normal), validation (64; frontend caches
  *do not exist* → reserved for E01), test (211). The 211 test clips are split video-grouped
  (stratified by `scene × has-anomaly`, 70 dev / 141 confirmation, same recipe).
* **Earlier test sets are already explored**, not untouched hold-outs (experiment plan §8). The dev / confirmation split
  therefore only *disciplines new choices made in this lane*; it does not make the data untouched. All historical
  reproduction numbers are reported on the full test set, exactly as before.
* **Frontends (frozen, all from earlier experiments):** released AlphaPose poses (control, the scorer's own training
  distribution), stock `yolov8n-pose` @ 640, `ML0`, `MLP` (MPS FP32) and `ML0`, `MLP` (NCNN FP16 384×640). `M00` / `M0P`
  are hashed but not scored (not needed for the quality baseline).
* **Scorer:** original STG-NF (`ShanghaiTech_85_9.tar`, `UBnormal_unsupervised_71_8.tar`) — "Huang-*inspired*", not Huang's
  scorer.
* **Controls:** (i) historical code path (`score_segments` + `variant_scores`) is the ground truth for reproduction;
  (ii) a *second, independent* window engine (this module) must agree with it window-by-window; (iii) released AlphaPose
  is the positive control for the scorer; (iv) negative controls for the reuse audit (post-normalisation reuse *must* fail).

## 4. Protocol steps (the decomposition) — one change at a time

| Step | Placement of window score | Smoothing | Windows scored | Empty-frame fill | Look-ahead? |
|---|---|---|---|---|---|
| **S0** offline paper | middle frame (`start + seg_len//2`) | two-sided Gaussian (σ = 1…6) | STG-NF windows (**last window omitted**) | global test-set `most_normal` | yes (future frames, test statistics) |
| **S1** offline "live" | end frame (`start + seg_len − 1`) | causal half-Gaussian | same | global test-set `most_normal` | test statistics only |
| **S2** all windows | **actual completion frame** (frame of the window's last pose) | causal half-Gaussian | **all** complete windows | global test-set `most_normal` | test statistics only |
| **S3** train sentinel | completion frame | causal | all | **train-calibrated sentinel** (p50 / p90 / p99 / max of *training-normal* window normality) | none |
| **S4** hold | completion frame | causal | all | hold last score ≤ `H` frames, then sentinel; `H ∈ {0, 6, 12, 24, ∞}`; coverage/staleness reported | none |

Auxiliary one-factor ablations so that S0→S1 and S1→S2 are attributable: `S0a` (end placement + two-sided Gaussian),
`S0b` (middle placement + causal Gaussian), `S1a` (end placement, all windows), `S1b` (completion placement, omitted-window set).

**Sentinel audit (critical item).** EXP-023 froze `sentinel_normality = −0.7886` = 90-th percentile of normality of the
first two windows of the first 25 released training clips. In normality (= −NLL, higher = more normal) the
most-normal test window is only ≈ −0.53. We quantify, using **training data only** (no test look-ahead), how a sentinel at
p50 / p90 / p99 / max of the training-normal distribution changes delivered AUC, and from which distribution the value
should be taken (released-pose train set, 330 clips; the 12 calibration clips; frontend-matched 12-clip pose sets for
stock and ML0).

**Selection discipline.** Sentinel percentile and hold `H` are chosen **only on the dev split of the ML0-MPS frontend
of each dataset** (criterion: causal delivered AUC; ties within 0.10 AUC points → smaller `H`, then smaller staleness) and then
applied unchanged to every frontend and reported on the confirmation split.

## 5. Pre-registered verification rules (pass/fail is decided by the code, not by the author)

| Rule | Statement | Pass criterion |
|---|---|---|
| **R1 hashes** | Every frozen artefact exists; full SHA-256 recorded; a second pass gives identical digests; `config.yaml` hashes equal the live files. | all present & equal |
| **R2 reproduction** | Re-scoring each stored pose folder with the historical code reproduces its stored AUC (paper & live). | `|ΔAUC| ≤ 0.01` points (and equals the figures quoted in the experiment plan to 2 decimals) |
| **R2b replay** | Replaying each detector cache (`c = 0.25`, BoT-SORT) regenerates the stored pose JSON. | bit-identical dictionaries on 100 % of clips |
| **R3 window parity** | This module's windows = STG-NF's windows (offline set), and normality agrees. | identical key sets; `max|Δ normality| ≤ 1e-4` |
| **R4 decomposition anchors** | S0 and S1 equal the historical *paper* and *live* AUC. | `|Δ| ≤ 0.01` points |
| **R5 reuse** | Re-assembling windows from a shared per-frame `(x, y, conf)` store equals independent re-assembly; post-normalisation reuse and hidden-state reuse differ. | per-frame reuse `max_abs_diff == 0.0` (inputs *and* scores); negative controls `> 0` |
| **R6 causality** | The causal deliverer is prefix-invariant (score at `t` unchanged if the video is truncated at `t`); no window is delivered before its completion frame; the sentinel is computed from training poses only. | `max|Δ| ≤ 1e-12`; 0 violations |
| **R7 cadence** | All ShanghaiTech clips are 24 fps; all UBnormal videos are 30 fps; GT length = cache length = video frame count. | 0 mismatches |
| **R8 split hygiene** | dev ∩ confirmation = ∅, dev ∪ confirmation = test; both contain normal and abnormal frames. | exact |

## 6. Outputs

`plan.md` (this file) · `config.yaml` (locked: splits, weights + SHA-256, cadence, scoring protocol, sentinel rule) ·
`results.csv` (per video × frontend: quality + audit metrics) · `decisions.jsonl` (sample log of window completions,
delivered scores, hold decisions, reuse checks; < 5 MB; git-ignored by `Q2-P1-*/decisions.jsonl`) · `summary.json`
(all aggregates, hashes, CIs, decomposition, sentinel table) · `report.md`.

## 7. Reproducibility specification

* Code: `src/edgelab/analysis/q2_p1_e00.py` (`uv run python -m edgelab.analysis.q2_p1_e00 {lock,run,report,all}`),
  unit tests `tests/test_q2_p1_e00.py`.
* Determinism: `numpy` seed 0 for every bootstrap (`bootstrap_ci_over_videos`, 2000 resamples for the historical
  re-scores, 1000 for new contrasts; videos are the resampling unit); torch CPU, `torch.no_grad`, `float32`.
* Intermediate window tables are cached in `~/thesis-data/q2-p1-e00/` (outside the repo); `run` rebuilds them if missing.
* Environment (python / torch / numpy / ultralytics versions, git commit, `uv.lock` SHA-256) is stored in `summary.json`.
* Stop rule for the lane: if a pre-registered rule fails, the report says so and E01 does not build on the failed item.
