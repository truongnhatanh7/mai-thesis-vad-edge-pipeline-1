# 03 · Code audit walkthrough — claim → code → report → test

*This file is generated from a machine-checked table (every `file:line` below was resolved by regex against the working tree, every report anchor string was found in the cited report, every test name exists). Re-check any time: `./reproduce_and_verify.sh verify`. Line numbers are as of 2026-11-16; if code moves, search by the symbol name (`rg -n "def name" experiments/src`).*

All paths are relative to `experiments/` unless they start with the thesis root files. Commands assume `cd /Users/nanh/Documents/RMIT_MscAI/thesis/experiments`.

## 1. How to audit in 30 minutes (do this first)

1. `../HUMAN_3DAY_REVIEW_BUNDLE/reproduce_and_verify.sh audit-tests` — runs the causality/leakage/invariance tests only.
2. Open `src/edgelab/analysis/q2_p1_e00.py`; read in this order: `global_most_normal_fill` (S0 leak) → `hold_fill` (S4 rule) → `landing_frames` → `sentinel_values` → `reuse_audit`. Each has its test in `tests/test_q2_p1_e00.py` (rows A01–A10 below).
3. Open `src/edgelab/models/stgnf_flow.py` lines 59–98 (`CausalSTGCN`, `CausalGraphSubnet`): find `F.pad(h, (0, 0, self.k - 1, 0))` — left-only padding is the whole causality proof; test `test_causal_flow_has_no_future_leakage`.
4. Read §5 below (five warnings). They are the only places I found where a *claim in prose is stronger than what the code does*.
5. Spot-check three numbers yourself: open `Q2-P1-E05_confirmation/summary.json`, find the Cache-C1 SHT value (73.43) and compare with the report; recompute one AUC with `analysis/q2_p1_e00.py::pooled_auc`.

## 2. How to prove "no leakage, no fake numbers" (checklist)

| Question | Where to look | Command / evidence |
|---|---|---|
| Is anything tuned on the test split? | `build_splits` (A11); E00 report §2 (locked inputs); E01 `select_tracker_cell` (A16, dev-only 36-cell grid); E07 `test_tracktrack_selection_is_on_dev_only` | `rg -n "dev" experiments/src/edgelab/analysis/q2_p1_e01.py` |
| Are model/artifact files the ones in the report? | E00 §2 lists the locked inputs with full SHA256 hashes and says the run re-hashed 31 objects (all unchanged); `verify_lock` does the re-hash | `shasum -a 256 <file>` (e.g. `yolov8n-pose.pt` → `c6fa93dd1ee4a2c18c900a45c1d864a1c6f7aba75d84f91648a30b7fb641d212` in E00 line 31) |
| Does the strict-causal protocol really never read the future? | A01–A06 | tests `test_hold_fill_equals_the_stateful_causal_deliverer`, `test_landing_modes_and_completion_never_precedes_end` |
| Is any scorer non-causal? | A32–A34, A38, A39, A44, A45 | `pytest tests/test_q2_p1_e06.py tests/test_q2_p1_e08.py tests/test_q2_p1_e07.py tests/test_q2_p1_e09.py -q` |
| Can a *cache policy* cheat by looking at frames it skipped? | A21 (`GuardedDetectionCache`) | `test_preflight_3_detector_cache_access_guard` |
| Are CIs correct? | A25 | `test_aucboot_point_matches_sklearn_with_ties` |
| Were hypotheses/verdicts fixed before results? | A26; each experiment has `plan.md` + `config.yaml` + `decisions.jsonl`; E03 `test_config_written_before_run_and_consistent`; E05 `test_plan_and_config_are_locked` | `ls Q2-P1-E0*/plan.md Q2-P1-E0*/config.yaml` |
| Is a report number really in the machine output? | every experiment folder has `summary.json` / `results.csv` | `./reproduce_and_verify.sh summary Q2-P1-E05_confirmation` |
| Is the Pi evidence real? | A29–A31; `Q2-P1-Pi5_suite/report.md` header (board, governor, idle floor); `throttled 0x0` in every run | `Q2-P1-Pi5_suite/stage_timings_pi5.csv` |

## 3. Audit table (claim → source → report → test)

Legend: *Src* = implementation, *Report* = file where the headline number lives, *Test* = unit test that would fail if the claim were false.


### Protocol, leakage and reuse audit (the foundation of every other number)

| ID | Claim | Headline (as in report) | Src (file:line, symbol) | Report | Test (file:line, symbol) |
|---|---|---|---|---|---|
| A01 | S0 'paper protocol' uses future information: whole-test-set most-normal fill | S0 77.88 -> S4 73.58 (SHT, ML0_mps) | `src/edgelab/analysis/q2_p1_e00.py:317`<br>`global_most_normal_fill` | `Q2-P1-E00_audit/report.md` | `tests/test_q2_p1_e00.py:158`<br>`test_global_fill_is_look_ahead_but_train_sentinel_is_not` |
| A02 | S0 smoothing is a symmetric (future-peeking) Gaussian; S4 uses a causal half-Gaussian | same ladder S0->S4 | `src/edgelab/pose/smoothing.py:44`<br>`SMOOTHERS` | `Q2-P1-E00_audit/report.md` | `tests/test_q2_p1_e00.py:199`<br>`test_delivered_score_is_prefix_invariant_but_global_fill_is_not` |
| A03 | STG-NF scoring silently omits the last window of every track (offline artefact) | 782 / 91,867 windows lost (0.85%) | `src/edgelab/analysis/q2_p1_e00.py:288`<br>`stgnf_offline_window_mask` | `Q2-P1-E00_audit/report.md` | `tests/test_q2_p1_e00.py:74`<br>`test_complete_windows_include_the_last_one_that_stgnf_omits` |
| A04 | S4 delivers a window score only at its completion frame (never before it ends) | S4 = strict causal | `src/edgelab/analysis/q2_p1_e00.py:295`<br>`landing_frames` | `Q2-P1-E00_audit/report.md` | `tests/test_q2_p1_e00.py:119`<br>`test_landing_modes_and_completion_never_precedes_end` |
| A05 | Missing frames use a TRAIN-calibrated sentinel (not a test-set statistic); hold H=0 | sentinel SHT -0.5122 / UB -0.8273 | `src/edgelab/analysis/q2_p1_e00.py:611`<br>`sentinel_values` | `Q2-P1-E00_audit/report.md` | `tests/test_q2_p1_e00.py:158`<br>`test_global_fill_is_look_ahead_but_train_sentinel_is_not` |
| A06 | Stateful causal deliverer == vectorised hold-fill (two independent implementations agree) | S3/S4 73.58 | `src/edgelab/analysis/q2_p1_e00.py:331`<br>`hold_fill` | `Q2-P1-E00_audit/report.md` | `tests/test_q2_p1_e00.py:175`<br>`test_hold_fill_equals_the_stateful_causal_deliverer` |
| A07 | CausalScoreDeliverer: runtime object that releases a score only from data already seen | queue guard / 24 fps | `src/edgelab/pipeline/refresh_policy.py:548`<br>`CausalScoreDeliverer` | `Q2-P1-E00_audit/report.md` | `tests/test_refresh_causality.py:75`<br>`test_preflight_1_prefix_invariance` |
| A08 | Per-frame (x,y,conf) pose reuse across overlapping windows is bit-exact | max_abs_diff 0.0; 18.5x SHT / 13.2x UB fewer pose evals | `src/edgelab/analysis/q2_p1_e00.py:765`<br>`reuse_audit` | `Q2-P1-E00_audit/report.md` | `tests/test_q2_p1_e00.py:291`<br>`test_per_frame_pose_reuse_across_overlapping_windows_is_bit_exact` |
| A09 | Normalisation is per-window, so POST-normalisation reuse is NOT valid | R7 FAIL (post-norm reuse) | `src/edgelab/analysis/q2_p1_e00.py:412`<br>`normalize_windows` | `Q2-P1-E00_audit/report.md` | `tests/test_q2_p1_e00.py:303`<br>`test_normalization_is_per_window` |
| A10 | Hidden-state reuse is invalid (changes scores) | hidden-state diffs 0.838 / 2.067 | `src/edgelab/analysis/q2_p1_e00.py:846`<br>`hidden_state_reuse` | `Q2-P1-E00_audit/report.md` | `tests/test_q2_p1_e00.py:352`<br>`test_reuse_audit_on_toy_poses` |
| A11 | Splits are deterministic, disjoint, stratified; tuning is dev-only | dev / confirmation split | `src/edgelab/analysis/q2_p1_e00.py:230`<br>`build_splits` | `Q2-P1-E00_audit/report.md` | `tests/test_q2_p1_e00.py:52`<br>`test_split_` |
| A12 | Locked inputs are SHA256-hashed and re-verified (31 artefacts) | yolov8n-pose.pt c6fa93dd...fb641d212 | `src/edgelab/analysis/q2_p1_e00.py:1078`<br>`verify_lock` | `Q2-P1-E00_audit/report.md` | `tests/test_q2_p1_e00.py:32`<br>`test_verify_lock_flags_changed_missing_and_unchanged_files` |

### Direction 1 — perception and tracking recovery

| ID | Claim | Headline (as in report) | Src (file:line, symbol) | Report | Test (file:line, symbol) |
|---|---|---|---|---|---|
| A13 | ML0 label fix: persons WITHOUT keypoint labels become box-only rows | recall <48 px 0.4% -> 37.0% | `src/edgelab/pose/coco_pose_labels.py:32`<br>`person_row` | `Q2-P1-E01_quality/report.md` | `tests/test_pose_coco_labels.py:37`<br>`test_person_row_rules` |
| A14 | Keypoint/visibility BCE losses are MASKED for box-only rows (so no fake keypoint gradient) | EXP-020 P1 +16.24 [+11.99,+20.45] | `src/edgelab/pose/loss_patch.py:21`<br>`masked_calculate_keypoints_loss` | `Q2-P1-E01_quality/report.md` | `tests/test_pose_loss_patch.py:77`<br>`test_gradient_of_patched_loss_ignores_keypoint_less_anchors` |
| A15 | Two-stage BoT-SORT: low-confidence boxes continue a live track but never start one | tracks 2,006 -> 1,766; S4 ML0 73.78 -> 76.82 | `src/edgelab/analysis/q2_p1_e01.py:352`<br>`build_tracker_cell` | `Q2-P1-E01_quality/report.md` | `tests/test_q2_p1_e01.py:128`<br>`test_two_stage_low_boxes_continue_a_live_track_but_cannot_start_one` |
| A16 | Tracker selected on a pre-registered 36-cell DEV-ONLY grid | c=0.20, low=0.05, buffer=30, match=0.85 | `src/edgelab/analysis/q2_p1_e01.py:450`<br>`select_tracker_cell` | `Q2-P1-E01_quality/report.md` | `tests/test_q2_p1_e01.py:83`<br>`test_grid_is_the_pre_registered_36_cells` |
| A17 | Miss decomposition: 84.2% of missed anomalous frames are intermittent gaps | 84.2% intermittent; 52.3% in [0.05,0.25) | `src/edgelab/analysis/q2_p1_e01.py:546`<br>`miss_breakdown` | `Q2-P1-E01_quality/report.md` | `tests/test_q2_p1_e01.py:223`<br>`test_miss_breakdown_classes_and_recoverability_shares` |
| A18 | TrackTrack (CVPR 2025) swapped in; chosen on dev only | ML0|TT|STG-NF 76.36 SHT / 64.30 UB | `src/edgelab/analysis/q2_p1_e07_pi5.py:37`<br>`make_tracktrack` | `Q2-P1-E07_yolo26_tracktrack_trajvadp/report.md` | `tests/test_q2_p1_e07.py:138`<br>`test_tracktrack_selection_is_on_dev_only` |
| A19 | EXP-021 coverage -> AUC bound (B1 FAIL 12/194, B2 PASS) | B1 FAIL (12/194 violations) | `src/edgelab/analysis/exp021.py:38`<br>`bound_half` | `EXP-021-coverage-bound/results.md` | `tests/test_analysis_exp021.py:19`<br>`test_best_case_scoring_attains_the_half_bound` |

### Direction 2 — causal delivery, reuse, Pi 5

| ID | Claim | Headline (as in report) | Src (file:line, symbol) | Report | Test (file:line, symbol) |
|---|---|---|---|---|---|
| A20 | C1 static-validity: reuse a detector output only while the scene is static (A1 = 4 frames) | C1 net CPU saving 28.5% | `src/edgelab/analysis/q2_p1_e03.py:160`<br>`reuse_pattern` | `Q2-P1-E03_ablation/report.md` | `tests/test_q2_p1_e03.py:400`<br>`test_c1_c2_validity_rules_in_simulation` |
| A21 | Policy cannot read detections of frames it did not run (anti-oracle guard) | causality guard | `src/edgelab/pipeline/refresh_policy.py:35`<br>`GuardedDetectionCache` | `Q2-P1-E03_ablation/report.md` | `tests/test_refresh_causality.py:149`<br>`test_preflight_3_detector_cache_access_guard` |
| A22 | Fixed-Burst baseline (equal CPU cost) so the cache is compared FAIRLY | Cache - Fixed +2.41 [+1.39,+3.48]* | `src/edgelab/analysis/q2_p1_e03.py:747`<br>`burst_starts` | `Q2-P1-E03_ablation/report.md` | `tests/test_q2_p1_e03.py:294`<br>`test_fixed_burst_parity_with_e02_bursts` |
| A23 | Matched-cost calibration: every arm is tuned to the same CPU budget | 12-cell confirmation means | `src/edgelab/analysis/q2_p1_e03.py:1235`<br>`calibrate` | `Q2-P1-E03_ablation/report.md` | `tests/test_q2_p1_e03.py:318`<br>`test_cost_matching_budget_and_queue_invariants` |
| A24 | Backlog guard hysteresis: shed ON > 400 ms, OFF < 150 ms | Cache-C1+Guard p99 wait 405 ms | `src/edgelab/analysis/q2_p1_e05.py:152`<br>`GuardedFixedPolicy` | `Q2-P1-E05_confirmation/report.md` | `tests/test_q2_p1_e05.py:71`<br>`test_guard_hysteresis` |
| A25 | Paired video bootstrap (2,000 resamples) == explicit weighting; AUC matches sklearn with ties | [lo, hi]* = CI excludes 0 | `src/edgelab/analysis/q2_p1_e03.py:1576`<br>`AucBoot` | `Q2-P1-E03_ablation/report.md` | `tests/test_q2_p1_e03.py:195`<br>`test_aucboot_resample_equals_explicit_weighting` |
| A26 | Stop rule / verdict logic is pre-registered code, not post-hoc prose | H1/H2/H3 verdicts | `src/edgelab/analysis/q2_p1_e03.py:1651`<br>`verdict_h3` | `Q2-P1-E03_ablation/report.md` | `tests/test_q2_p1_e03.py:238`<br>`test_stop_rule` |
| A27 | E02 fixed baseline: per-stage ms/frame and sustainable slowdown s* | SHT 14.93 ms/frame; s* 3.08 / 1.66 | `src/edgelab/analysis/q2_p1_e02.py:216`<br>`fifo_finish(` | `Q2-P1-E02_fixed_baseline/report.md` | `tests/test_q2_p1_e02.py:186`<br>`test_burst_windows_are_fully_genuine_D6` |
| A28 | E05 live VM run: RSS growth bounded, late windows dropped, queue measured | p99 wait 4 / 405 / 408 ms | `src/edgelab/analysis/q2_p1_e05_vm.py:50`<br>`run_longrun` | `Q2-P1-E05_confirmation/report.md` | `tests/test_q2_p1_e05.py:335`<br>`test_vm_delivery_offline_lands_windows_and_drops_late_ones` |
| A29 | Pi 5 telemetry: PMIC power, SoC temp, throttle codes, per-core clocks | 0x0 throttled; 0.192 J/frame | `src/edgelab/telemetry/pi5_telemetry.py:139`<br>`Pi5TelemetryMonitor` | `Q2-P1-Pi5_huang_repro/report.md` | `tests/test_q2_p1_pi5.py:48`<br>`test_summarize_telemetry_values_and_throttle_codes` |
| A30 | Pi5 suite grid: complete & unique cells; E05 arm definitions match the task | Burst32 24.01 FPS, 250 ms, 0.168 J/frame | `src/edgelab/analysis/q2_p1_pi5_suite.py:172`<br>`cell_specs` | `Q2-P1-Pi5_suite/report.md` | `tests/test_q2_p1_pi5.py:94`<br>`test_cell_specs_grid_complete_and_unique` |
| A31 | Huang et al. reproduction arms on the physical Pi 5 | Huang RPi 6.14 FPS, 162.86 ms | `src/edgelab/analysis/q2_p1_pi5_huang.py:507`<br>`get_arms` | `Q2-P1-Pi5_huang_repro/report.md` | `tests/test_q2_p1_pi5.py:119`<br>`test_e05_arm_definitions_match_the_task` |

### Direction 3 — causal flows, dilation, interaction graph

| ID | Claim | Headline (as in report) | Src (file:line, symbol) | Report | Test (file:line, symbol) |
|---|---|---|---|---|---|
| A32 | Causal temporal conv is LEFT-padded (k-1): output at t depends only on frames <= t | A2 76.54 vs A0 76.86 (S4, ML0) | `src/edgelab/models/stgnf_flow.py:59`<br>`CausalSTGCN` | `Q2-P1-E06_lorad_causal_flow/report.md` | `tests/test_q2_p1_e06.py:46`<br>`test_causal_flow_has_no_future_leakage` |
| A33 | Online engine == batch scorer == streaming (no train/serve skew) | same scores bit-for-bit (to tolerance) | `src/edgelab/models/lorad.py:384`<br>`OnlineLoRaD` | `Q2-P1-E06_lorad_causal_flow/report.md` | `tests/test_q2_p1_e06.py:75`<br>`test_stream_equals_batch_equals_online_engine` |
| A34 | Streaming EMA normaliser uses only the past (N1 vs per-window N0) | A3s 72.89 (streaming norm costs ~3.7 pt) | `src/edgelab/models/lorad.py:81`<br>`EmaNormalizer` | `Q2-P1-E06_lorad_causal_flow/report.md` | `tests/test_q2_p1_e06.py:60`<br>`test_ema_normalisation_is_causal` |
| A35 | Coupling step is exactly invertible; log|det J| is exact | 616 params (SHT) / 488 (UB) | `src/edgelab/models/stgnf_flow.py:164`<br>`CouplingStep` | `Q2-P1-E06_lorad_causal_flow/report.md` | `tests/test_models_stgnf_flow.py:63`<br>`test_flow_keeps_shape_and_is_invertible` |
| A36 | log-determinant equals the autograd Jacobian determinant (numerical check) | bits/dim loss | `src/edgelab/models/stgnf_flow.py:207`<br>`CouplingFlow` | `Q2-P1-E06_lorad_causal_flow/report.md` | `tests/test_models_stgnf_flow.py:76`<br>`test_log_determinant_equals_the_jacobian` |
| A37 | RevMargin reverse-MSE is identically 0 up to float round-off (invertibility, not learning) | P_B: reverse term 0 | `src/edgelab/models/lorad_gaps.py:512`<br>`hybrid_loss` | `Q2-P1-E08_lorad_gaps_proposals/report.md` | `tests/test_q2_p1_e08.py:96`<br>`test_every_step_is_exactly_invertible_so_the_reverse_term_is_round_off` |
| A38 | MS-CTFlow dilated branches are each causal and see the far past | P_A 1,592 params | `src/edgelab/models/lorad_gaps.py:103`<br>`MSCausalSTGCN` | `Q2-P1-E08_lorad_gaps_proposals/report.md` | `tests/test_q2_p1_e08.py:77`<br>`test_dilated_branches_are_each_causal_and_see_the_far_past` |
| A39 | Every E08 proposal is prefix-invariant (score at t unchanged if the future is deleted) | P_A..P_ABC | `src/edgelab/models/lorad_gaps.py:362`<br>`OnlineGapScorer` | `Q2-P1-E08_lorad_gaps_proposals/report.md` | `tests/test_q2_p1_e08.py:64`<br>`test_prefix_invariance_every_proposal` |
| A40 | IAC-Flow adjacency: 20 nodes (18 joints + box-velocity node 19 + social node 20), hub edges | P_C 25.45 FPS / 5.56 ms / 419 ms / 67.51 | `src/edgelab/models/lorad_gaps.py:77`<br>`make_adjacency_ctx` | `Q2-P1-E08_lorad_gaps_proposals/report.md` | `tests/test_q2_p1_e08.py:52`<br>`test_adjacency_20_nodes_and_hub_distances` |
| A41 | Social-proximity feature is computed per frame from other tracks in that frame only | neighbour features | `src/edgelab/models/lorad_gaps.py:223`<br>`neighbour_features` | `Q2-P1-E08_lorad_gaps_proposals/report.md` | `tests/test_q2_p1_e08.py:157`<br>`test_neighbour_features_nearest_and_solo` |
| A42 | Clip context (box kinematics) is causal per frame and matches a hand-computed value | node 19 box velocity | `src/edgelab/models/lorad_gaps.py:239`<br>`clip_context` | `Q2-P1-E08_lorad_gaps_proposals/report.md` | `tests/test_q2_p1_e08.py:166`<br>`test_clip_context_is_causal_per_frame_and_matches_hand_value` |
| A43 | TrajVAD-P: 27 kinematic features, causal; model sizes match the authors' sizes | 757,338 params (test), 27.9 ms on Pi 5 | `src/edgelab/models/trajvad.py:244`<br>`TrajVADP` | `Q2-P1-E07_yolo26_tracktrack_trajvadp/report.md` | `tests/test_q2_p1_e07.py:73`<br>`test_models_have_the_authors_sizes` |
| A44 | TrajVAD-P features use only past frames | N_FEATURES = 27 | `src/edgelab/models/trajvad.py:108`<br>`trajectory_features` | `Q2-P1-E07_yolo26_tracktrack_trajvadp/report.md` | `tests/test_q2_p1_e07.py:36`<br>`test_trajectory_features_are_causal` |
| A45 | E09 H1: MS-CTFlow-Bottleneck (K=3, d in {1,2,4}) is causal and invertible | 608 params; short-event +6.85 pt, pooled -3.7 pt | `src/edgelab/analysis/q2_p1_e09.py:277`<br>`MSCTFlow` | `Q2-P1-E09_hypotheses_suite/report.md` | `tests/test_q2_p1_e09.py:11`<br>`test_msctflow_causality_and_invertibility` |

### Direction 4 — two-tier caching, green training

| ID | Claim | Headline (as in report) | Src (file:line, symbol) | Report | Test (file:line, symbol) |
|---|---|---|---|---|---|
| A46 | H2 Strategy A (accuracy): Tier-2 skip decision | calls saved 16.9-62.7%, |dAUC| <= 0.06 (see WARNING W1) | `src/edgelab/analysis/q2_p1_e09.py:518`<br>`apply_two_tier_mpc_to_tracks` | `Q2-P1-E09_hypotheses_suite/report.md` | `tests/test_q2_p1_e09.py:32`<br>`test_h2_two_tier_mpc_kinematic_extrapolation` |
| A47 | H2 Pi 5 workload run: Tier-1 scene gate + Tier-2 pixel-diff gate (causal gate; no AUC measured) | Heavy / Medium / Sparse / Strategy-B | `src/edgelab/analysis/q2_p1_e09_pi5.py:157`<br>`run_workload_cell` | `Q2-P1-E09_hypotheses_suite/report.md` | — (no dedicated test) |
| A48 | H2 regimes are SYNTHETIC occupancy schedules built by frame index (see WARNING W2) | Medium 40% / Sparse 10% active | `src/edgelab/analysis/q2_p1_e09_pi5.py:127`<br>`build_workload_frames` | `Q2-P1-E09_hypotheses_suite/report.md` | — (no dedicated test) |
| A49 | H3 green training: bone + smoothness + reconstruction regularisers | G2 45 s vs G0 14.6 h (see WARNING W3) | `src/edgelab/analysis/q2_p1_e09.py:602`<br>`compute_kinematic_losses` | `Q2-P1-E09_hypotheses_suite/report.md` | `tests/test_q2_p1_e09.py:52`<br>`test_h3_kinematic_losses_and_refiner` |
| A50 | H3 energy / carbon accounting | g0 hours and TDP are constants in code | `src/edgelab/analysis/q2_p1_e09.py:650`<br>`run_h3_green_training_ablation` | `Q2-P1-E09_hypotheses_suite/report.md` | `tests/test_q2_p1_e09.py:52`<br>`test_h3_kinematic_losses_and_refiner` |
| A51 | E09 evaluator (S0/S4 + short/sustained split); S4 sentinel calibrated on the first 12 EVALUATED clips | see WARNING W4 | `src/edgelab/analysis/q2_p1_e09.py:424`<br>`eval_scorer_on_tracks` | `Q2-P1-E09_hypotheses_suite/report.md` | — (no dedicated test) |

### Direction 5 — FastOS-RL

| ID | Claim | Headline (as in report) | Src (file:line, symbol) | Report | Test (file:line, symbol) |
|---|---|---|---|---|---|
| A52 | 18 OS actions = 3 governors x 3 thread counts x {unpinned/nice0, pinned/nice-10} | 18 configurations | `src/edgelab/analysis/q2_p1_e10_pi5.py:75`<br>`build_action_space` | `Q2-P1-E10_rl_os_tuning/report.md` | `tests/test_q2_p1_e10.py:15`<br>`test_action_space_and_linucb_convergence` |
| A53 | 8-dim surrogate-warm-started LinUCB | tune 4.0-18.7 s per pipeline | `src/edgelab/analysis/q2_p1_e10_pi5.py:189`<br>`FastOSLinUCB` | `Q2-P1-E10_rl_os_tuning/report.md` | `tests/test_q2_p1_e10.py:15`<br>`test_action_space_and_linucb_convergence` |
| A54 | Reward R = -(2.5 Drop + 1.0 Mean/41.67 + 0.4 p95/41.67 + 0.15 J) | reward formula | `src/edgelab/analysis/q2_p1_e10_pi5.py:259`<br>`compute_rl_reward` | `Q2-P1-E10_rl_os_tuning/report.md` | `tests/test_q2_p1_e10.py:40`<br>`test_reward_and_delivered_auc_estimator` |
| A55 | 'Delivered Pi 5 AUC' in E10 is a hand-written FORMULA of measured service/drop (see WARNING W5) | e.g. 74.35 / 69.07 / 62.48 | `src/edgelab/analysis/q2_p1_e10.py:40`<br>`estimate_delivered_auc_from_pi5_service` | `Q2-P1-E10_rl_os_tuning/report.md` | `tests/test_q2_p1_e10.py:40`<br>`test_reward_and_delivered_auc_estimator` |
| A56 | OS knobs applied via sysfs governor, sched_setaffinity, setpriority (<1.5 ms) | ondemand downclock to 1500 MHz on cache skips | `src/edgelab/analysis/q2_p1_e10_pi5.py:86`<br>`apply_os_action` | `Q2-P1-E10_rl_os_tuning/report.md` | `tests/test_q2_p1_e10.py:15`<br>`test_action_space_and_linucb_convergence` |

### Explorer

| ID | Claim | Headline (as in report) | Src (file:line, symbol) | Report | Test (file:line, symbol) |
|---|---|---|---|---|---|
| A57 | Streamlit explorer renders its tabs without exceptions | Dataset Explorer / Failure Case Inspector | `src/edgelab/app/streamlit_app.py:195`<br>`render_model_panel` | `START_HERE_LEARNING_GUIDE.md` | `tests/test_app_catalog.py:386`<br>`test_streamlit_new_tabs_render_without_exceptions` |


## 5. Audit warnings — claims stronger than the code (W1–W6)

These were found by reading the code behind the reports. None of them changes a number the code produced; they change **what you may claim from it**. All are fixable in hours, not weeks.

### W1 — E09 H2 Strategy A decides to skip using the pose it is supposedly skipping (oracle gate)
`experiments/src/edgelab/analysis/q2_p1_e09.py:553` computes `p_true = kp_arr[t, :, :2]` (the *actual detected pose at frame t*) and line 556 extrapolates only `if rel_motion < eps_disp_px`, where `rel_motion` is the distance between the true pose and the constant-velocity prediction. A deployed system would not have `p_true` when deciding to skip pose inference. So the accuracy table "H2 saves 16.9–62.7% pose calls with |ΔAUC| ≤ 0.06" is an **oracle upper bound**, not a deployable result. The Pi-run gate (`q2_p1_e09_pi5.py::run_workload_cell`) *is* causal (pixel-difference inside the previous boxes) but measures time/energy, not AUC.
**Fix:** evaluate AUC with the same pixel-difference gate used on the Pi (replay the cached frames), or relabel Strategy A as "oracle ceiling".

### W2 — E09 H2 Medium/Sparse energy regimes are synthetic and Tier 1 reads the schedule
`q2_p1_e09_pi5.py:144` (`active = (i % 120) < 48`, Medium) and `:147` (`(i % 240) < 24`, Sparse) build occupancy by *frame index*; line 218 sends a frame to Tier-1 sleep whenever `not is_active` (a label of the synthetic schedule), and quiescent frames are replaced by the previous grayscale frame (line 213). The Medium/Sparse energy savings therefore show what happens when the gate is *perfectly right about empty scenes*, not measured savings. The Heavy (100%) and the "StrategyB_Natural_Sparse" regimes are not synthetic.
**Fix:** drive Tier 1 from the pixel-energy test only (drop the `not is_active` shortcut) or call the regimes "ideal-gate power model".

### W3 — E09 H3 "1,150× faster / −99.98% energy" compares a measured 45 s run with assumed constants
`q2_p1_e09.py:665` hard-codes `g0_stage1_hours = 14.5`; line 668 converts to kWh with an assumed 0.250 kW; line 748 uses an assumed 0.045 kW for G1/G2. G0's accuracy is *loaded* from E06's `A2` checkpoint (line 670); it is not retrained here. G1/G2 train with `epochs = 6` (line 333) and `max_batches = 80`, using `MSCTFlow(kernel_t=3)` (line 699) whereas G0 is the K=13 LoRaD flow — so G0-vs-G2 mixes architecture and training recipe. Training data for both is the released STG-NF training poses (`STGNF_TRAIN`), i.e. AlphaPose-extracted skeletons — "teacher-free" is true for the *loss*, not for the *data source*.
The E09 report itself prints both `573x–763x (877.2 min -> 1.15–1.53 min)` (header) and `1,438x faster (0.61 min)` (table): two different runs quoted side by side.
**Fix:** measure wall-clock and energy (PMIC / wall meter) for a *matched architecture* with a real retrain of G0; report seeds and CIs.

### W4 — E09 evaluator differs from the E00/E06 protocol
`q2_p1_e09.py::eval_scorer_on_tracks` (line 424) calibrates the S4 sentinel from `calib_tracks[:12]` (line 451) — the first 12 *evaluated* clips' first two windows per track — instead of the train-calibrated sentinel locked in E00. It is label-free, but it is test-data-derived, and results differ from E06 (e.g. LoRaD SHT ML0_edge S4 76.29 in E09 vs 76.54 in E06). The short/sustained split (line 485) is **clip-level** (normal clips included in both groups), unlike E08's event strata. No bootstrap CIs and a single seed (0).
**Fix:** call `e00.sentinel_values` with the train sentinel; reuse E08's strata + `AucBoot`; add 3 seeds.

### W5 — E10 "Delivered Pi 5 AUC" is a formula, not a measurement
`q2_p1_e10.py:40` (`estimate_delivered_auc_from_pi5_service`) maps measured mean service time and drop rate to an AUC with hand-set constants (`floor_auc = 53.32` at line 53; factors 0.35, 0.25, 0.45; line 192 applies it to every row). The service/FPS/energy/clock columns *are* measured on the Pi; the AUC column and its deltas (+0.00, −5.28, −11.11 …) are model outputs. Also: no static `performance|T4` control arm; one run per cell.
**Fix:** compute delivered AUC from the logged per-frame scores exactly as E07/E08 do (Mac replay), and add a `performance|T4` static baseline + repeats.

### W6 — `Pi5_huang_repro` delivered AUCs are not computed on-device
`q2_p1_pi5_huang.py` stores per-arm *constants* for offline and causal-ideal AUC (e.g. line 519: 68.53) and measures only timing/power/queue on the board. The report's "Delivered Real-Time Pi 5 AUC" values (e.g. 53.32) come from the queue model (report findings item 4: `s_pi = 4.45`). Label them as "queue-model AUC from measured Pi timing".

### Smaller discrepancies in the prose documents (cite the primary table, not the summary)
| Where | Says | Primary source says |
|---|---|---|
| `Q2_P1_SYNTHESIS_FOR_PAPER_WRITING.md` §6 | `273` passing tests in `test_q2_p1_*.py` + `test_app_catalog.py` | `pytest --collect-only` on exactly those files collects **298** tests |
| Synthesis §5.4 | `ML0` outperforms YOLO26 e2e by `+14.14 [-19.1, -9.7]*` | E07 report: YOLO26 e2e + TrackTrack + STG-NF is `−14.2 [−19.1, −9.7]` vs the locked baseline (sign/format inconsistency) |
| E09 report header vs table | `573x–763x` | table `1,438x` (see W3) |
| E10 report §1 vs tables | `10–14 Seconds`, `6 Pipelines` | tables: 8 pipelines, tune time 4.0 s – 18.7 s |
| E08 report §2 | CIs for pooled AUC printed as fractions (e.g. `[-0.02, -0.01]`) while point values are in points | convert ×100 before comparing |
| `PAPER_POSITIONING_AND_CHECKPOINT.md` §2 | IAC-Flow "best of all models" on false alarms | E08 §1 verdict for H_C is **NOT SUPPORTED**; SHT FA/h@R50 improves (49.1 → 35.0) but UB worsens (101.2 → 167.4) and the mean worsens +26.1 [+7.5, +46.5]* |
| Positioning §1.1 | A3s S4 `−3.97` (ML0) | E06 headline lists released A3s S4 75.59 vs A0 80.85 (−5.26); cite the matching pose source |
| `RESEARCH_SYNTHESIS.md` | low-light/resolution framing, ends 2026-11-06 | superseded by label-cause finding (EXP-020); do not use for the paper |

## 6. Where the numbers physically live (for spot checks)

| Need | File |
|---|---|
| Any experiment's machine-readable summary | `experiments/<EXP>/summary.json` (E00–E10, Pi5_*); `results.csv` per video/cell |
| Per-video AUC rows | `Q2-P1-E08_lorad_gaps_proposals/per_video.csv`, `Q2-P1-E07_yolo26_tracktrack_trajvadp/contrasts.csv` |
| Pi 5 stage timings | `Q2-P1-Pi5_suite/stage_timings_pi5.csv`, `Q2-P1-E08_lorad_gaps_proposals/pi5_benchmark.csv`, `Q2-P1-E07_yolo26_tracktrack_trajvadp/pi5_results.csv` |
| E09/E10 raw tables | `Q2-P1-E09_hypotheses_suite/{h1_results,h2_accuracy,h2_pi5_workloads,h3_green_training}.csv`, `Q2-P1-E10_rl_os_tuning/` |
| All MLflow runs (snapshot) | `mlflow_runs.csv` (thesis root); live DB `experiments/mlflow.db` |
| Decisions and amendments | `experiments/DECISIONS.md`, `*/decisions.jsonl` |
