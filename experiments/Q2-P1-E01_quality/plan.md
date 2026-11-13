# Q2-P1-E01 — Quality (plan, written before any E01 number was computed)

**Lane:** `Q2-P1-*`. **Source plan:** `Edge Anomaly Detection Experiment Plan.md`, ladder row **E01 — Quality**.
**Builds on:** `Q2-P1-E00_audit` (locked splits `test_dev` / `test_confirmation`, SHA-256 hashes, causal S3/S4 protocol).
**Training:** none. Every checkpoint and detector cache is frozen and hashed. Detector caches were produced earlier
(MPS FP32, NCNN FP16 384x640) and are only *replayed* here; STG-NF scoring runs on CPU.
**What was visible when this plan was written:** the E00 report (it is the input of this experiment), including E00's
S4 AUC per source and split. Nothing computed by E01 existed. Where an E00 number could tempt a rule choice this is said
explicitly (rule F2).

## 1. Question

> **What accuracy can the frozen frontend/scorer reach before CPU constraints?** Compare stock and existing ML0, inspect
> missed small people and tracking gaps. Required output: an *offline quality reference*; settings are chosen on
> development data. No new training.

The experiment has four parts and one hand-off:

| part | content |
|---|---|
| **A. Frontend & resolution** | released AlphaPose oracle, stock @480/640/800/960 (+ NCNN FP16 @800), M00, ML0, MLP (MPS FP32 and NCNN FP16), scored with the offline paper protocol **S0** and the causal protocol **S3/S4** on ShanghaiTech (107 clips) and UBnormal (211 clips), split `dev` / `confirmation` / `all`. Paired video-bootstrap contrasts. |
| **B. Missed small people** | recall by reference-person height (EXP-011/EXP-022 protocol) and the EXP-022 miss decomposition (never-seen tracks vs intermittent gaps), extended with *how many misses exist in the raw detector cache at lower confidence* (0.05 <= conf < 0.25). |
| **C. Tracking gaps & stranded windows** | tracks, track lengths, stranded short tracks (no STG-NF window), within-track gaps (1-2 frames bridged by STG-NF's `missing <= 2` rule vs >= 3 frames that break windows), scorable-frame coverage, person-window density. |
| **D. Development-split tracker/gate tuning** | training-free sweep of the detection gate `c`, true two-stage BoT-SORT (low-confidence association), `track_buffer`, `match_thresh`; **chosen on `test_dev` only**, verified on `test_confirmation` and `all`. |
| **Hand-off** | `locked_for_E02`: frontend, tracker/gate configuration, scoring protocol. |

## 2. Assumptions (each verified or reported by the code)

| # | Assumption |
|---|---|
| A1 | E00's `config.yaml` is unchanged (its SHA-256 is re-checked); split lists, scorer checkpoints, detector weights and the 8 E00 detector caches still hash to E00's values; the new caches used here are hashed in this experiment's `config.yaml` before any score is computed (R1). |
| A2 | The stored pose folders of the new arms (stock @480/800/960, M00, stock-NCNN@800) are the deterministic BoT-SORT replay (`c=0.25`, default Ultralytics yaml, no GMC/ReID) of their caches. Checked on a clip sample by track-partition equality (R2). |
| A3 | Per-frame (x, y, conf) poses are shared by overlapping windows, normalisation is per window, windows are `seg_len` poses of one track with <= 2 missing frames (`seg_len` 24 SHT, 16 UB). E00 verified these; E01 imports E00's engine (`q2_p1_e00`) instead of re-implementing it. |
| A4 | The causal protocol is exactly E00's: **completion landing**, causal half-Gaussian, train-calibrated sentinel, hold `H=0`: SHT sentinel quantile `max`, UB `p50`, both read from E00's `summary.json` (not re-tuned). S3 == S4 because the selected hold is 0; both are reported. |
| A5 | The reference people for recall are the released AlphaPose test poses (SHT only; UBnormal has no per-person reference in the repo). The reference is a detector's output, not human annotation (EXP-011 threat). |
| A6 | Frontend arms that have no frontend-matched sentinel calibration use E00's rule: stock family (any resolution, NCNN) -> `stock_calib12`; ML0 (MPS, NCNN) -> `ML0_mps_calib12`; M00, MLP, released -> `released_all`; every UB arm -> `released_all`. A sensitivity column re-scores every arm with the common `released_all` sentinel. |
| A7 | The test sets were explored by earlier experiments. `dev` / `confirmation` protect only the choices made in this lane (frontend lock, tracker/gate lock); they do not make the data untouched (experiment plan section 8). |

## 3. Hypotheses and pre-registered rules

All rules are evaluated by code (`summary.json -> rules`) and printed PASS / FAIL / INFO in the report. AUC points = AUC x 100.
CIs are 95 % percentile **video bootstrap** (1000 resamples, seed 0, paired for contrasts).

### Part A

| rule | statement | pass criterion |
|---|---|---|
| **QA1** | Labels help: ML0 beats stock @640 on ShanghaiTech | `ML0_ncnn - stock` S4 on SHT/all > 0 and paired CI lower > 0 |
| **QA2** | Labels beat resolution at equal runtime | `ML0_ncnn - stock_ncnn800` S4 on SHT/all > 0, CI lower > 0 (stock@800 costs 1.56x FLOPs, ML0@640 1.00x) |
| **QA3** | The ML0 gain is the labels, not 4 more fine-tuning epochs | `ML0_mps - M00_mps` S4 on SHT/all > 0, CI lower > 0 |
| **QA4** | NCNN FP16 export is quality-neutral | `|ML0_ncnn - ML0_mps|` S4 on SHT/all <= 0.5 pt and CI contains 0 |
| **QA5** | Stock resolution helps but less than ML0 | `stock_r960 - stock` S4 point estimate > 0 on SHT **and** `ML0_mps - stock_r800` S4 > 0 (point) |
| **QA6** | UBnormal does not show the SHT gain | informational: `ML0_mps - stock` S4 on UB/all with CI; QA6 "supported" if CI lower > 0 |
| **QA7** | P2 head: `MLP - ML0` | informational (CI reported); decided for the lock by F2 |

### Part B (recall, SHT, released-AlphaPose reference, EXP-011 matcher: box conf >= 0.25 and IoU >= 0.3)

| rule | statement | pass criterion |
|---|---|---|
| **QB1** | The small-people gap remains for ML0 | pooled recall(<48 px) of `ML0_ncnn` <= 0.5 x pooled recall(>=96 px) of `ML0_ncnn` |
| **QB2** | ML0 recovers small people vs stock | pooled recall(<48 px): `ML0_ncnn` > `stock`, with non-overlapping Wilson 95 % CIs |
| **QB3** | Temporal bridging is relevant | EXP-022 rule: intermittent-track share of missed person-frames >= 30 % (reported per arm; decided on `ML0_ncnn`) |
| **QB4** | A lower gate / two-stage association has material to work with | share of missed person-frames having a cached box with 0.05 <= conf < 0.25 and IoU >= 0.3 is >= 20 % (descriptive threshold, on `ML0_ncnn`); the **two-stage-recoverable** share = intermittent-track misses with such a box (a low-confidence box can only continue a live track under BoT-SORT, never start one) |

### Part C (descriptive, no pass/fail; definitions are fixed here)

* *Stranded short track*: a track with fewer than `seg_len` poses (SHT < 24, UB < 16) - it produces **no** STG-NF window. *Stranded gappy track*: >= `seg_len` poses but 0 complete windows (too many missing frames). *Exactly-`seg_len` tracks* are dropped by STG-NF's last-window omission but scored by the causal protocol.
* *Gap*: consecutive poses of one track with `frame_diff - 1 = g > 0` missing frames. `g in {1, 2}`: can be bridged by the `missing <= 2` window rule (if the window contains no more than 2 missing frames in total); `g >= 3`: no window can contain the gap, the track needs a full re-warm-up (`seg_len - 1` further poses before the next window completes, 23 on SHT, 15 on UB).
* *Scorable pose*: a pose that is the last pose of >= 1 complete window. Unscorable poses are split into: in a stranded track / initial warm-up (first `seg_len - 1` poses of a scorable track) / post-gap re-warm-up (all others).
* *Scorable-frame coverage*: share of ground-truth frames (all / normal / anomalous) with >= 1 window completing on them. *Person-window density*: windows per frame, and windows per scorable frame (mean number of simultaneously scored persons).

### Part D - tracker / gate tuning (development data only)

**Finding that motivates it (from reading `replay.py`):** the replay filters detections at `conf >= c` *before* the tracker
sees them, so BoT-SORT's second-stage low-confidence association (`track_low_thresh`) never receives a box in
`[track_low_thresh, c)`. The sweep passes boxes down to the low threshold into BoT-SORT: the tracker may use them to *continue*
existing tracks but cannot start tracks from them (`new_track_thresh = c`). The poses written for such frames carry the
low box confidence.

Grid (full factorial, 36 cells; the default cell is the E00 setting, i.e. `c=0.25`, low disabled, `track_buffer=30`, `match_thresh=0.8`):

| factor | levels |
|---|---|
| `c` (= `track_high_thresh` = `new_track_thresh` = gate) | 0.20, **0.25**, 0.30 |
| low-confidence association (`track_low_thresh`; boxes down to it reach the tracker) | **disabled** (= current behaviour), 0.10, 0.05 |
| `track_buffer` | **30**, 45 |
| `match_thresh` | **0.80**, 0.85 |

Series on which cells are evaluated, **`test_dev` clips only** (SHT 35 clips, UB 70 clips), metric = causal S4 AUC with the
locked sentinel/hold (also reported: scorable-frame coverage, tracks, stranded tracks, windows):

* **primary:** `ShanghaiTech/ML0_ncnn` (the deployable frontend; NCNN FP16 cache),
* **veto series:** `ShanghaiTech/stock`, `UBnormal/ML0_mps`, `UBnormal/stock`.

| rule | statement |
|---|---|
| **T1 selection** | winner = highest dev S4 AUC on the primary series; cells within 0.10 pt of the best are tied and resolved in favour of the cell closest to the default (fewest changed factors, then smaller `|c - 0.25|`, then low disabled). |
| **T2 adoption** | a non-default winner is adopted only if **(a)** delta vs default on the primary series >= +0.5 pt, **(b)** the paired dev bootstrap CI of that delta excludes 0, **(c)** on none of the three veto series the paired dev CI of (winner - default) lies entirely below 0. Otherwise the clean default is locked. |
| **T3 verification** | the locked cell, the default cell and the dev arg-max are evaluated on `test_confirmation` and `all` (SHT: ML0_ncnn, stock, MLP_ncnn; UB: ML0_mps, stock). Report-only, never re-selects. If the locked non-default cell is **significantly worse** than the default on confirmation (paired CI entirely below 0 on the primary series), the verification is flagged failed and the default is locked for E02. |
| **T4 honesty** | both dev and confirmation numbers of default, locked cell and dev arg-max are reported even if the default is locked. The winner's dev delta is selection-biased (36 cells); the confirmation delta is the unbiased check. |

Also reported: marginal effect of each factor (mean dev AUC over the other factors) so the answer to "does true two-stage
association help?" does not hinge on the single winning cell.

### Frontend lock for E02

| rule | statement |
|---|---|
| **F1** | deployable candidate set (NCNN FP16 caches exist): `ML0_ncnn@640` (cost 1.00x FLOPs), `stock_ncnn@800` (1.56x), `MLP_ncnn@640` (1.92x; P2 head). Costs from EXP-020 amendment 1.3. |
| **F2** | start from the cheapest candidate; a costlier candidate replaces the incumbent only if its dev S4 AUC on **SHT** is >= incumbent + 2.0 pt **and** the paired dev CI of the difference has lower bound > 0. The +2.0 pt margin with CI-lower > 0 is inherited unchanged from EXP-020's rules P1/P2; it was *not* chosen after looking at E01 (E00 already showed dev S4 ML0_ncnn 72.39 vs MLP_ncnn 74.35, so this margin decides that comparison - the reader should know that). |
| **F3** | UB veto (MPS proxies, there is no UB NCNN cache): MLP does not replace ML0 if `MLP_mps - ML0_mps` dev S4 on UB has a paired CI entirely below 0. |
| **F4** | confirmation and `all` numbers of the locked frontend and of the runner-up are reported; report-only. |

## 4. Controls

(i) `released` AlphaPose poses = positive control of the scorer on every protocol step; (ii) E00's historical re-score is the
ground truth for every arm that E00 already scored: the E01 S0/S3/S4 numbers of those arms must equal E00's (rule R3);
(iii) `M00` isolates the label effect of ML0; (iv) NCNN vs MPS of the same weights isolates quantisation; (v) the default
tracker cell must reproduce the stored pose folders (R2) and E00's AUC (R3); (vi) the sentinel-sensitivity column removes the
frontend-matched sentinel as an explanation of any frontend gap.

| rule | statement | criterion |
|---|---|---|
| **R1** | hashes | E00 `config.yaml` SHA-256 equal; E00 + new caches/pose folders re-hash equal to this `config.yaml`; split lists equal E00's |
| **R2** | replay | my tracker replay at the default cell reproduces the stored pose folders (track-partition equality) on a deterministic sample of clips per arm |
| **R3** | parity with E00 | E01 S0 / S3 / S4 AUC of every E00 source equals E00's `summary.json` within 0.01 pt |
| **R4** | splits | dev/confirmation disjoint, cover the test set; tuning code never touches a confirmation clip before the lock is written (asserted: the sweep gets dev clip indices only) |
| **R5** | causality | the S3/S4 series of every arm is prefix-invariant on a sample (E00's test) and no window lands before its completion frame |
| **R6** | pipeline consistency | the in-memory tracker pipeline at the default cell (replay -> windows -> S0/S4) equals the stored-pose pipeline of part A, and the dev-only sweep AUC of the default cell equals the dev AUC of its full-set evaluation: max abs difference <= 0.02 pt (float32 batch-composition noise of the STG-NF scores) |

## 5. Reproducibility

* Code: `src/edgelab/analysis/q2_p1_e01.py` (`uv run --extra pose python -m edgelab.analysis.q2_p1_e01 {lock,run,report,all}`), tests `tests/test_q2_p1_e01.py`. It imports E00's window engine, scorer wrapper, protocol steps, splits and bootstrap; nothing of E00 is re-implemented.
* `lock` hashes the new caches/pose folders and writes `config.yaml` (frontends, caches, resolution list, tracker grid, rules, protocol, E00 pointers) **before** `run`.
* Determinism: seed 20261113 (splits, from E00), bootstrap seed 0, 1000 resamples; BoT-SORT ids are per tracker instance (reset every video); torch CPU, `no_grad`, float32.
* Intermediate window tables: `~/thesis-data/q2-p1-e01/` (outside the repo); `run` rebuilds them when missing. E00's tables are reused when their stamp (pose folder fingerprint + checkpoint hash) matches.
* `summary.json` stores environment (versions, git commit, `uv.lock` SHA-256) and every aggregate; `results.csv` per video x arm (x tracker cell); `decisions.jsonl` one record per selection / diagnostic (git-ignored by `Q2-P1-*/decisions.jsonl`).

## 6. Outputs

`plan.md` (this file), `config.yaml`, `results.csv`, `decisions.jsonl`, `summary.json`, `report.md` (answers E01 with tables: offline S0 and causal S3/S4 on dev / confirmation / all, recall by height, miss decomposition, tracking-gap statistics, dev-split tracker tuning, the locked frontend/tracker configuration for E02).
