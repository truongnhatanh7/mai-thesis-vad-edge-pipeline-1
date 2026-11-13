# EXP-006 — Do the EXP-003/004/005 findings replicate on a second dataset (UBnormal)?

Pre-registration frozen 2026-11-06 09:50, before any UBnormal frame or score exists (download started 09:40).
Under D13/D19 (execute, review afterwards).

## 1. Background (plain language)

All findings so far come from one dataset (ShanghaiTech) with *synthetic* low light. UBnormal (Acsintoae et al.,
CVPR 2022) is a different benchmark: rendered people in 29 scenes, with some scenes at night and variants with fog,
smoke or rain — real changes of lighting inside the scene, not a sensor model. STG-NF ships a UBnormal checkpoint
(unsupervised, 71.8 AUC) and the released poses, so the same frozen-workload method applies.

## 2. Method

- **Data:** UBnormal official zip (Google Drive id `1KbfdyasribAMbbKoBU1iywAhtoAt9QI0`, sha256 logged), official
  test split (STG-NF's `data/UBnormal/pose/test` file list = the test videos); frame-level GT built the way STG-NF's
  evaluation does (verified by reproducing its number in R0).
- **R0 gate:** released AlphaPose test poses + `UBnormal_unsupervised_71_8.tar` → paper-variant AUC within ±0.5 of
  71.8. If it fails, stop and log.
- **Pipeline:** as EXP-003 (YOLOv8n-pose conf-0.01 cache on MPS, BoT-SORT gmc none replay, frozen STG-NF UBnormal
  checkpoint), both scoring variants. Configurations: E0 / E1 gamma / E2 CLAHE at 640; E0 at 480 and 960; gate sweep
  c ∈ {0.25, 0.15, 0.10, 0.05} with the EXP-003 cross-fitting rule. No synthetic darkening.
- **Light strata (fixed now):** per test video, mean luma of frames sampled at 1/s (EXP-000 method). **dark** = mean
  luma < 60; **weather** = name contains fog / smoke / rain; **normal** = the rest. Strata with < 3 normal or
  < 3 anomalous videos are reported but not tested. Stratum AUC = frame AUC over the videos of that stratum.
- **Deadline replay:** EXP-004 VM traces (O0-B4, O1n-B2, O0r-B2) for the matching knobs; delivered live AUC at 24 fps.
  (Detector input for 1280×720 letterboxes to the same 384×640 at imgsz 640 as 856×480 — checked before replay.)

## 3. Pre-registered expectations (replications of robust findings)

| # | Replicates | Pass rule (all test videos unless stated) |
|---|---|---|
| U0 | STG-NF reproduction | R0 within ±0.5 of 71.8 |
| U1 | resolution helps offline | AUC(960) − AUC(640) > 0, paired CI excludes 0 (paper variant) |
| U2 | gates don't help | G-xfit − default ≤ +0.5 (both variants) |
| U3 | coverage ranks configs | Spearman(label-free all-frame coverage, AUC) over the 5 configs at c 0.25 ≥ 0.7 |
| U4 | enhancement is light-specific | dark stratum: max(E1, E2) − E0 > 0; normal stratum: max(E1, E2) − E0 ≤ +1 |
| U5 | deadlines reverse resolution | delivered AUC(960) < delivered AUC(640) at O0-B4 (live variant) |

## 4. Deviations policy

As EXP-003. If the dark stratum has too few videos, U4 is reported as not testable, not re-defined.

## 5. Clarifications and Addendum A (frozen 2026-11-06 11:06; no EXP-006 accuracy result seen yet except R0)

- **Strata outcome (data fact, not a result):** min mean luma over the 211 test videos = 64.2 → the pre-registered
  dark stratum (< 60) is **empty**; U4 is reported as *not testable* (§4 policy). Weather stratum: 29 videos
  (22 with anomalies); normal: 182.
- **U5 cost input:** UBnormal frames are 1280×720 (letterbox 384×640 at imgsz 640) or 1080–1106×720 (448×640).
  U5 uses the EXP-004 traces measured at 384×640 for every video, which understates the detector cost of the
  448×640 videos by ≈ 17% of pixels → bias *against* finding drops; reported as a threat.
- **Addendum A (exploratory):** synthetic sensor darkening (EXP-003 §4.1 model, seeds from level name) at L2 (α 0.1)
  and L3 (α 0.03) on the 211 test videos × {E0, E1 gamma, E2 CLAHE} @640, c 0.25 + gate sweep.
  Expectations (replication of EXP-003/005): **UA1** at L2 the best of E1/E2 is E2 (CLAHE) and beats E0;
  **UA2** at L3 the best of E1/E2 is E1 (gamma) and beats E0; **UA3** Spearman(label-free coverage, AUC) over the
  6 dark configs ≥ 0.7. Runs after the main caches (CPU-bound darkening).
