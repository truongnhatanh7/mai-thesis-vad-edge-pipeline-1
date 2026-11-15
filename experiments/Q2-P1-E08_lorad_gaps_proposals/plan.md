# Q2-P1-E08 — Four fixes for LoRaD's limitations (MS-IAC-RevFlow): pre-registered plan

*Written 14 Nov 2026, before any E08 model was trained or any E08 number exists. Anything that changes afterwards goes into "Amendments" at the bottom with its reason.*
Source document: `LORAD_GAPS_EXPERIMENT_PLAN.md`. Builds on E06 (LoRaD flow, window engine) and E07 (TrackTrack, TrajVAD-P, Pi 5 runner).

**Question.** E06/E07 left four limitations of the cheap skeleton flows (STG-NF, LoRaD, ~0.6 K parameters): one temporal scale (Gap 1), isolated single-person
modelling (Gap 2), a pure forward-NLL objective (Gap 3) and fragility to detector / tracker noise (Gap 4). Each is attacked on its own (single-factor ladder) and then jointly,
and each hypothesis is tested against LoRaD (`B1`) with paired video-bootstrap CIs, on the offline matrix and on the physical Raspberry Pi 5.

## 1. Arms
Baselines are **E06 checkpoints, not retrained**: `B0` = `A1_STGNF_Retrained` (symmetric STG-NF, per-window normalisation N0); `B1` = `A2_LoRaD_CausalConv_WinNorm`
(causal flow, N0, window mode; the arm E06 selected as "LoRaD" for E07, so E07's Pi 5 LoRaD numbers are this model). Seeds 0, 1, 2 everywhere.

| Arm | Change vs `B1` | Trained | Params (SHT / UB) |
|---|---|---|---|
| `P_A` MS-CTFlow | each coupling block: single causal conv -> 3 parallel causal **dilated** convs d in {1,2,4} (kernel k = L//2+1, left pad d(k-1)), concat -> 1x1 projection (+ identity skip of the graph features) -> BN -> + block residual | yes | counted at build time (≈ 2.5x) |
| `P_B` RevMargin | same network; loss `NLL(X) + l1 |X_y - f^-1(f(X))_y|^2 + l2 max(0, m - (NLL(X_pert) - NLL(X)))` | yes | = B1 (616 / 616) |
| `P_C` IAC-Flow | V = 20 graph: + node 19 (box kinematics) + node 20 (social proximity), 20x20 `uniform` adjacency, hub = neck + both hips | yes | = B1 (shared 1x1 convs, graph is a buffer) |
| `P_D` ConfSmooth | causal confidence-weighted smoothing of joints with c < 0.25 before windowing; **B1 weights, no retraining** (test-time front end) | no | = B1 |
| `P_ABC` MS-IAC-RevFlow | P_A + P_B + P_C + P_D (smoothing also applied to the training tracks) | yes | = P_A |

**Decisions the source plan leaves open (fixed here):**
* *Node channels.* The flow input has two channels (x, y), so a node carries two numbers. Node 19 = `10 * (dcx/h, dcy/h)` (keypoint-box centre velocity per frame in box heights, clipped to +-0.5 h/frame,
  divided by the frame gap capped at 5, first pose 0). The box aspect `w/h` is computed but **not** fed to the flow (the box comes from the keypoints, so `w/h` is already in the normalised skeleton).
  Node 20 = `prox * (dx, dy)/|(dx, dy)|` with `prox = 1/(1 + d_nn/h)` to the nearest other pose **in the same frame** (0 if solo); (direction, prox) is a bijection onto the source plan's
  `(dx_nn/W, dy_nn/H, prox)`. Context is computed from the raw (or smoothed) track **before** N0; N0 normalises the 18 body joints only.
* *Graph.* STG-NF's `uniform` strategy with `max_hop = 8` is dense (all pairs within 8 hops), so the 20-node A is dense as well; hub edges are in the edge list and tested but do not sparsify A.
* *Perturbations (P_B).* Per training window one of: temporal frame shuffle; velocity scaling `x_t -> x_0 + s(x_t - x_0)`, `s ~ U(0.2,0.5) U U(1.8,3.0)`; joint jitter `sigma ~ U(0.05,0.25)`, all in N0-normalised space.
  `l1 = 1.0`, `l2 = 0.1`, `m = 0.5` bit/dim — **not tuned**. The perturbed pass uses the running BatchNorm statistics (the statistics the scorer uses at test time) and does not update them.
  *Known property, stated up front:* `f^-1(f(X)) = X` for an exactly invertible flow, so the reverse term is zero up to float round-off; it is implemented as specified, its size is logged every epoch and
  reported, and `P_B`'s effect is therefore attributable to the margin term (plus round-off conditioning). No separate `l1`-only / `l2`-only arms are trained (budget).
* *Smoothing (P_D).* `p^_t = w p_t + (1-w) p^_{t-1}`, `w = min(1, c/0.25)` on joints with `c < 0.25`, `w = 1` otherwise, `p^_0 = p_0`, per track, over its whole prefix.
* *Windowing / scoring.* Window mode, N0, 24 (SHT) / 16 (UB) poses, kernel 13 / 9: exactly the B1 path. Score = -NLL; S0/S4 landing and sentinels from E00/E06 (sentinel = train-calibrated, scorer's own, on released train tracks processed by the arm's own front end).

## 2. Training (identical recipe to E06, one process at a time)
Released AlphaPose normal training tracks (SHT 330 clips, L = 24; UB 186 clips, L = 16); Adamax lr 5e-4 x 0.99^epoch, batch 256, **10 epochs**, windows every 6 poses, identity + mirror augmentation
(body x -> -x about the window mean; node x channels change sign), grad-clip 100, prior N(3,1), K = 8 steps, **seeds 0, 1, 2**. 4 arms x 3 seeds x 2 datasets = 24 models. Device is recorded per model in `config.json`.

## 3. Evaluation grid
Pose sources (tracks never re-detected): `released` AlphaPose test poses; `ML0_edge` = **primary edge pipeline** (SHT: `ML0_ncnn` @ 640; UB: `ML0_mps` proxy — no UB NCNN cache exists) + the E01 locked
two-stage BoT-SORT (c 0.20 / low 0.05 / buffer 30 / match 0.85); `YOLO26_e2e` + the E07 T1-selected TrackTrack cell (c 0.20 / low 0.05 / match 0.80). Protocols **S0** offline, **S4** strict causal (E00).
SHT 107 clips (35 dev / 72 confirmation), UB 211 clips (70 / 141). Pooled frame-level AUC, 2,000-resample paired **video** bootstrap (seed 20261114, shared weights across arms; seed-averaged arms use the mean over seeds
inside each resample). Also reported: `TrajVAD-P` (E07 cells, same pipelines; none for `released`).
`dev` is the selection split, `confirmation` the verification split: **all verdicts use `confirmation`**; the dev contrast is reported next to it as the pre-registered selection signal (sign agreement).

**Strata (events = maximal runs of anomalous frames of a clip; 194 SHT / 199 UB per the source plan):**
`short_event` duration <= 2.5 s, else `sustained_event`; `fast_motion` iff the maximum over released-pose tracks of the median per-pose box speed during the event is >= 0.06 box-heights/frame
(walking ~0.03, running ~0.07, cycling ~0.12; the labels carry no object classes, so this kinematic definition stands in for "bikes / skaters / running"); `high_crowd` iff the mean number of concurrent released tracks over
the event's frames >= 4, else `low_crowd`. Strata are computed once from the released poses (pipeline independent). Subgroup AUC: positives = frames of the stratum's events, negatives = all normal frames of the
evaluated clips (frames of other events are ignored), same bootstrap.
**Operating-point metrics (S4):** an alert = a maximal run of frames with score >= tau. `FA/h@R50` = false alerts (runs containing no anomalous frame) per hour of normal video at the tau that detects 50 % of events
(an event is detected when any of its frames reaches tau). `Recall@FA(B1)` = event recall (overall and per stratum) at the largest-recall tau whose FA/h does not exceed B1's FA/h@R50 of the same cell.
CIs: per-clip counts at the arm's own fixed tau, video bootstrap (ratio of sums), paired with B1.

## 4. Pre-registered rules (primary pipeline = ML0_edge, S4, confirmation, mean over the two datasets of the paired contrast, vs `B1`)
Verdict vocabulary: **SUPPORTED / PARTIAL / NOT SUPPORTED / REFUTED** (REFUTED = pooled CI upper bound < 0).
* **H_A (P_A):** SUPPORTED iff pooled AUC contrast CI lower bound > 0 **and** short_event stratum AUC contrast CI lower > 0 **and** sustained_event contrast point estimate > 0. PARTIAL iff exactly one of {pooled, strata} holds.
* **H_B (P_B):** SUPPORTED iff pooled AUC contrast CI lower > 0 **and** FA/h@R50 contrast CI upper < 0; PARTIAL iff one of the two; hard property: parameter count equal to B1 (and Pi 5 scorer latency within 5 %).
* **H_C (P_C):** *accuracy part* holds iff fast_motion **and** high_crowd AUC contrasts both have CI lower > 0 **and** `Recall@FA(B1)` point contrast > 0 in both strata; *competitiveness part* holds iff
  `P_C - TrajVAD-P` pooled contrast CI lower >= -2.0 pt (non-inferiority) **and** params(P_C) <= params(TrajVAD-P)/300 **and** Pi 5 scorer ms/frame(P_C) <= ms/frame(TrajVAD-P)/4 (measured). SUPPORTED iff both parts; PARTIAL iff one.
* **H_D (P_D):** SUPPORTED iff FA/h@R50 contrast CI upper < 0 **and** pooled AUC contrast CI lower >= -0.5 (smoothing may not cost accuracy); PARTIAL iff one.
* **H_Joint (Pi 5, `ML0_ncnn@640` + two-stage BoT-SORT + `P_ABC` + Cache-C1+Guard):** SUPPORTED iff **all** of: its delivered (on-time) AUC is the highest among the Cache-C1+Guard Pi arms {B0, B1, P_A, P_B, P_C, P_ABC, TrajVAD-P};
  delivered FPS >= 24.0; p99 queue wait < 500 ms; `throttled = 0x0` throughout. PARTIAL iff the delivered-AUC condition holds but a real-time condition fails, or vice versa for >= 3 of 4 conditions.
  Offline companion: P_ABC - B1 pooled AUC contrast and P_ABC vs each single-factor arm (reported, not part of the verdict).
* **Dev -> confirmation agreement** is reported for every contrast (sign of the point estimate).

## 5. Physical Raspberry Pi 5 (`nanh@raspberrypi.local`, 4 threads)
Checked idle with `ps aux | grep python` and run only after every Mac job has finished. 12 ShanghaiTech clips on disk (2,880 frames at 24 FPS arrival, one worker, Pi5-Huang queue model), ML0 YOLOv8n-pose NCNN FP16 384x640 -> two-stage BoT-SORT (E01 locked)
-> scorer in {B0, B1, P_A, P_B, P_C, P_ABC, TrajVAD-P}, each Full-Rate and Cache-C1+Guard (guard on 400 ms / off 150 ms, locked E07 rules) = 14 runs, seed-0 SHT checkpoints. Reported per run: decode / pose / track / scorer / service ms,
FPS, p99 wait, PMIC W, J/frame, temperature, RSS, `throttled`, delivered AUC (E00 S4 landing of the on-time windows with the train-calibrated sentinel). Live telemetry and `@mlflow.trace` frame traces to `http://127.0.0.1:5000`
(experiment `Q2-P1-E08`); the registered model `edgelab-ms-iac-revflow` holds the P_ABC checkpoints.
B0/B1/P_* are all run through the same online window scorer (one flow call per frame for all completed windows) so their scorer times are comparable; TrajVAD-P through its E07 online scorer.

## 6. Outputs
`results.csv`, `subgroups.csv`, `pi5_benchmark.csv`, `summary.json`, `report.md`, `tests/test_q2_p1_e08.py` (`uv run pytest tests/test_q2_p1_e08.py` green); extra: `per_video.csv`, `contrasts.csv`.

## 7. Known limits (stated up front)
* Sentinel calibration of every learned scorer is in-sample on released-pose train windows.
* Only the 12 ShanghaiTech clips on the Pi are used for the delivered AUC (E07 A3), so Pi AUCs have wide uncertainty and no CI.
* The subgroup strata come from a kinematic / count proxy, not object-class labels.
* `l1`, `l2`, `m`, the VEL_GAIN and the perturbation ranges are not tuned.
* UBnormal edge poses are the MPS proxy of the ML0 detector.

## Amendments
* **A1 (14 Nov 2026, before any E08 model was trained; cost only, no result involved).** `config.yaml` first said "jobs: sequential". A 20-batch speed probe showed that one thread is the fastest per model
  (P_ABC 1.05 s/batch at 1 thread vs 2.4 s at 8 threads; MPS no faster), so 24 models one-at-a-time would take ~25 h of which 9 of 10 cores idle. The 24 independent trainings therefore run on **5 single-thread
  worker processes of this one experiment** (the E06 pattern). The rule "zero experiments in parallel" is kept between experiments and stages: training starts only after `ps aux | grep -E "q2_p1_e06|q2_p1_e07"`
  is empty, and evaluation, statistics and the Pi 5 benchmark start only after the previous stage has finished. Device = CPU. Numerics do not depend on the number of workers (seeded, per-process).
