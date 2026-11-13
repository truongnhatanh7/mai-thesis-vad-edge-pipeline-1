# EXP-001 — STG-NF pose-skeleton anomaly detection: reproduction ladder

> Status: **pre-registered 2026-11-03** (written before any run; lead-approved scope, owner sign-off pending).
> Type: reproduction + measurement. Machine: Mac M4 Pro 48 GB, **CPU only** (forced, to mimic the edge; accuracy is
> machine-independent). Decision context: D11 (licences research-only), D12 (EXP-001 = this workload).

## Background (plain language)

**Pose-based anomaly detection.** Instead of feeding raw video pixels to a big network, we first run a small *pose
model* that finds each person and 17 body keypoints (nose, shoulders, elbows, knees, …). A *tracker* links the same
person across frames, giving one "stick-figure trajectory" per person. An anomaly detector then only looks at these
stick figures and asks "is this person's motion unusual compared with what we saw in normal training videos?"
(running, fighting, falling, cycling on a walkway…). Skeletons are tiny (18 points × 2 coordinates), so the scorer is
cheap, and they ignore clothing/background.

**Normalizing flow (one paragraph).** A normalizing flow is a neural network that is *invertible*: it maps an input
(here, a 24-frame pose sequence) to a point in a simple space where the data should look like a standard bell curve
(Gaussian). Because the mapping is invertible, the network can compute exactly how *probable* an input is. Trained
only on normal videos, it learns to map normal motion near the centre of the bell curve; an unusual motion lands far
out and gets a low probability → high anomaly score. STG-NF (Hirschorn & Avidan, ICCV 2023) is such a flow built from
spatio-temporal graph convolutions, with only ~1K parameters.

**Why it matters for the thesis.** The full pipeline is *pose model → tracker → flow scorer*. The scorer is nearly
free; the cost on a Raspberry Pi 5 is the pose model, which reads raw pixels — exactly where low light hurts and where
the three stages compete for 4 CPU cores. Before we study preprocessing or OS tuning on this workload, we must know
that (a) we can reproduce the published accuracy, (b) how much we lose when we swap the paper's heavy offline pose
pipeline (AlphaPose + PoseFlow) for an edge-realistic one (YOLO-pose nano + built-in tracker), and (c) what the honest
*live* (causal) number is.

## Problem

The published 85.9 AUC (ShanghaiTech Campus) is computed with (1) AlphaPose poses precomputed offline and (2) a
Gaussian smoothing of scores that uses *future* frames. Neither is available in a live edge pipeline. There is no
public number for STG-NF on YOLO-pose inputs or with causal scoring.

## Objective

Establish a trusted, measured baseline of this workload for later experiments:

- R1: reproduce the paper number on CPU with released data and checkpoint.
- R2: measure the AUC drop when the checkpoint receives our YOLO-pose + tracker poses.
- R3 (conditional): retrain on YOLO poses if the drop is large.
- R4: measure the causal (online) AUC.
- Bonus: per-stage CPU cost on the Mac (comparative only).

## Method

**Data.** ShanghaiTech Campus (437 videos, 13 scenes; unsupervised split: 330 normal train videos, 107 test videos with
frame-level masks). STG-NF `data.zip` (Google Drive id `1o9h3Kh6zovW4FIHpNBGnYIRSbGCu-qPt`, ≈1.2 GB; AlphaPose poses +
ground-truth masks). ShanghaiTech raw test frames from the official page or a verified mirror (test part needed for R2,
training part only for R3). Every download: URL, size, sha256 in `log.md`.

**Model.** STG-NF repo (https://github.com/orhir/STG-NF), pinned commit; shipped checkpoint
`checkpoints/ShanghaiTech_85_9.tar`. Minimal CPU patch kept as `stgnf-cpu.patch`.

**Pose (R2).** Ultralytics YOLO pose, nano variant (edge-realistic; the Pi 5 system of Huang 2026 used YOLOv8n-Pose),
built-in tracker via `model.track`, on CPU. Converter `src/edgelab/pose/` → STG-NF JSON format (COCO-17 keypoints;
the STG-NF loader derives the 18th neck point itself if it does so — to be verified by reading the loader). Unit test:
our parser reads a released AlphaPose file identically to the STG-NF loader.

**Steps.**
1. R1: run STG-NF evaluation (`--dataset ShanghaiTech --checkpoint ...`) on CPU. Cross-check AUC with
   `edgelab.metrics.scores.frame_level_auc` from their per-frame scores; bootstrap 95% CI over videos (2000
   resamples, seed 0). Also report AUC without Gaussian smoothing. Log CPU runtime.
2. R2: YOLO pose + tracker on all test frames → JSON → frozen checkpoint → AUC (smoothed and unsmoothed). Sanity:
   persons/tracks per video and keypoint-score distributions vs released files.
3. R3: decision rule below.
4. R4: replace the centred Gaussian filter with a causal filter (one-sided half-Gaussian with the same σ, past frames
   only; EWMA as secondary) on both pose sources.
5. Per-stage CPU cost on one video: pose ms/frame (input size noted), tracker ms/frame, STG-NF scoring; torch
   intra-op threads 1/2/4. Labelled "Mac, comparative only".

**Metric.** Frame-level ROC-AUC over all concatenated test frames (standard protocol); bootstrap CI over videos.
Every run tracked in MLflow experiment `EXP-001` via `tracked_run`.

## Pre-registered expectations / gates

| # | Expectation | Rule |
|---|---|---|
| R1 | Released poses + released checkpoint on CPU reproduce **85.9 ± 0.5** AUC; ≈ **81.9** without smoothing | Pass if within ±0.5 of 85.9. Our metric must match theirs to 1e-6 |
| R2 | Same checkpoint on our YOLO-pose + tracker poses: drop of "a few points" | Measurement, no pass/fail |
| R3 | Run **only if R2 drop > 3 points**: retrain STG-NF on YOLO poses (train split) with the repo's own recipe (CPU or MPS) | Target ≥ 84 AUC (reference: Huang 2026 reports 86.2 with its own model — figure from the lead's brief, not yet verified in the KB) |
| R4 | Causal smoothing on both pose sources | Measurement; expected to land between unsmoothed and smoothed |
| C  | Per-stage CPU cost | Measurement; Mac only, comparative |

## Deviations policy

- Metric, split, segment length/stride and checkpoint are fixed by this plan; they are not changed after seeing
  results. Any forced change (e.g. a dead download link, a library-version incompatibility) is logged in `log.md` as a
  *deviation* with the reason, and reported in `results.md`.
- The R3 decision is made mechanically from the R2 number and logged.
- If a step exceeds ~2 h wall clock, stop at a clean checkpoint, document state in `log.md`, report.
- No KB or other-experiment files are edited from this experiment.
