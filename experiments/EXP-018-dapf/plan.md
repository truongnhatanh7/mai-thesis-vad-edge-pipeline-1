# EXP-018 — DAPF: Deadline-Aware Pose Filling (a method, not just a policy)
Frozen 2026-11-08 12:20 before any run (owner: "prioritize that over YOLO26"; D22).

## Motivation
EXP-009/010/010b: filling the poses of frames a slow CPU drops beats skipping (+3.7 to +8.6 AUC, consistent; per-cell
significance borderline). Two weaknesses remain: (i) STG-NF treats predicted poses as real and was never trained on
irregular input; (ii) constant-velocity prediction is crude. DAPF fixes both.

## Components
- **C1 filling-aware scorer (STG-NF-FA):** STG-NF retrained (EXP-001 recipe, released ShanghaiTech **training** poses;
  UBnormal analogue for UBnormal) with augmentation: random frame drops (rate ~U(0, 0.85), latest-frame-style bursts),
  dropped frames filled by the active predictor, plus a **provenance input channel** (1 = detected, 0 = predicted).
- **C2 learned predictor (P-learned):** tiny causal GRU (≤ 20 K params) on normalised keypoints, trained on training
  poses to predict the next pose; replaces constant velocity; same cap G 8 and confidence decay.
- **C3 deadline-aware scheduler** (phase 2, only if C1/C2 are done): per frame choose detect vs predict-only from the
  remaining time budget and track uncertainty; separate amendment before any C3 run.
## Evaluation
Groups: SHT-L0, SHT-L2, UB-L0 (YOLOv8n-pose caches) and SHT-L0 top-down (EXP-016 n640 K1 poses). s ∈ {1, 3, 6};
keep-mask seeds 1 and 2 (EXP-010/010b masks, re-generated identically). Arms: SKIP, FILL-HOLD (const-vel + standard
STG-NF, = EXP-010), +C1, +C2, **DAPF = C1+C2**. Live variant; paired bootstrap over videos (1000); degenerate cells
(< 200 scorable windows in an arm) reported separately. Offline (s 1, no drops) also reported.
| # | Expectation | Pass rule |
|---|---|---|
| D1 | DAPF beats FILL-HOLD | DAPF − FILL-HOLD > 0 in ≥ 75% of non-degenerate (group × s × seed) cells, **and** the pooled difference (mean over cells, bootstrap over videos with the same resample shared across cells of a dataset) has CI lower bound > 0 |
| D2 | each component helps | C1-only and C2-only each > FILL-HOLD on the pooled difference (point estimate) |
| D3 | no offline cost | at s 1 without drops, STG-NF-FA ≥ standard STG-NF − 1.0 on released test poses and on edge poses (SHT L0) |
| D4 | still beats SKIP | DAPF − SKIP pooled CI lower bound > 0 |

## Amendment 1 — implementation details (2026-11-08 13:00, written before any EXP-018 result; no rule changed)
1. **Baselines:** FILL-HOLD = EXP-010's policy with the *released* STG-NF checkpoint (UBnormal: the released unsupervised checkpoint), constant-velocity fill G 8. SKIP = EXP-010's skip-and-reindex arm. Both re-computed here for s 1, 3, 6 and seeds 1, 2; at s 3, 6 they must equal the EXP-010/010b rows (checked, reported).
2. **Two FA models per dataset:** "+C1" (constant-velocity filler + STG-NF-FA) uses an FA model trained with constant-velocity fills; "DAPF" (GRU filler + FA) uses an FA model trained with GRU fills ("the active predictor", plan C1). "+C2" = GRU filler + the released standard STG-NF. Same recipe/seed (0) for both FA models. One training seed per FA model (the plan asks for none more).
3. **FA input:** x, y as in STG-NF plus the provenance **condition** (1 detected / 0 predicted, per frame) fed to every coupling subnet (the flow models x, y only: still a valid density). Code: `edgelab.models.stgnf_flow` kind `graph-fa`; same 8 steps, 576 params.
4. **Augmentation (code: `edgelab.pose.augment`):** every training window is seen with a random keep mask over (16 frames of context + the window): rate r ~ U(0, 0.85) per window, made by the EXP-004 latest-frame simulator with log-normal service times (mean 1/(1−r) frame periods); undetected frames inside gaps ≤ 8 frames after a detection are filled by the active predictor; windows are then cut like STG-NF cuts a track (first 24 poses, ≥ 22 within 24 frames) and dropped if they do not qualify. So the training distribution is the set of windows that would also be scored at test time. Slightly stricter than STG-NF at the edge cases (windows are cut at their original start frame, not re-cut on the filled track).
5. **C2 (GRU):** 48 hidden units, ~14 K parameters, features and rollout as in `edgelab.pose.predictor`; trained with backprop through rollouts on masks of the same kind, loss = mean keypoint error / pose height on filled frames; early stopping (patience 3) on held-out training videos (every 10th file); state reset after gaps > G. Trained separately on ShanghaiTech and UBnormal training poses. Reported next to constant velocity on the held-out videos.
6. **Top-down group (SHT-L0-TD):** the EXP-016 L0 n640 K1 rtmpose-s poses are the full-rate poses; the keep mask (same masks as SHT-L0) removes the poses of dropped frames (SKIP: re-indexed) **without re-running the tracker or the pose model** (their tracks come from the full-rate run). This is an approximation: SKIP and the fill arms see identical tracks. Stated as a limitation in results.
7. **Cells and degeneracy:** cell = (group, s, seed) = 4 x 3 x 2 = 24. A cell is *degenerate* if any arm in it has < 200 scorable windows; degenerate cells are excluded from D1/D2/D4 counts and pooled means and listed separately (and with them included, as a sensitivity row).
8. **Pooled difference:** mean over the non-degenerate cells of the AUC difference (live variant, ×100). CI: bootstrap over videos, 1000 resamples; one resample of the 107 ShanghaiTech videos is shared by all ShanghaiTech cells (groups SHT-L0, SHT-L2, SHT-L0-TD), one resample of the 211 UBnormal videos by the UBnormal cells; the pooled value of a resample = mean over cells.
9. **D3 offline:** s = 1 *without drops* (full-rate poses, no mask): STG-NF-FA (provenance all 1) vs the released standard checkpoint, paper variant, released test poses and SHT-L0 YOLOv8n-pose edge poses; paired bootstrap over videos. The D3 pass rule is on the point estimate (as written).
10. **Order/status:** C3 not touched. GAU-FA only after an EXP-019 G1 pass and a separate amendment.

## Amendment 2 — D3 reading and training-set note (2026-11-08 13:00, before any EXP-018 result)
- **D3** is evaluated for *both* STG-NF-FA models (trained with constant-velocity fills, trained with GRU fills); it passes only if both are ≥ standard − 1.0 on both pose sets (stricter than a single-model reading). Standard = the released checkpoint.
- 0.67% of the ShanghaiTech training windows (973/145,162; UBnormal 0.40%) are unusable by the augmentation pipeline (their 24 poses span more than 26 frames); skipped in all trainings of this lane.
