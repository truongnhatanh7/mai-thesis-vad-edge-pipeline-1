# EXP-018-dapf log (append-only, newest last)

### 2026-11-08 12:23 — lane F start
- Read specs, STATUS-NOW, EXP-001/009/010/010b code, STG-NF third_party model. Plan: edgelab/models/stgnf_flow.py (GAU + graph subnets, provenance conditioning), edgelab/pose/{segments,augment,predictor,stgnf_train,score_model}.py, analysis/exp019.py, analysis/exp018.py. No third_party edits.

### 2026-11-08 12:30 — code built + tested (flow, segments, augment, predictor)
- New: src/edgelab/models/stgnf_flow.py (graph subnet == third-party: 616 params, bit-equal nll with copied weights; GAU subnet 5808 params = 9.4x; provenance condition; GAU key mask), pose/segments.py (windows == third-party gen_dataset), pose/augment.py (STG-NF normalisation == normalize_pose incl. flip; random keep masks via latest-frame simulator; window cutting; provenance), pose/predictor.py (GRU 14K params; batched const-velocity == fill_P; causality + reset tests). Tests: tests/test_models_stgnf_flow.py, test_pose_segments.py, test_pose_augment.py, test_pose_predictor.py (+ tests/pose_toy.py helper). No third_party edits.

### 2026-11-08 12:57 — C2 predictor training launched (tmux f18-pred: ShanghaiTech then UBnormal)
- edgelab.pose.predictor_train, 14,485 params, GRU rollouts on random keep masks, held-out = every 10th training pose file (14,589 windows), early stopping patience 3. Smoke (30 batches): val error 0.0482 (GRU) vs 0.0585 (const-vel) vs 0.0663 (hold still), fraction of body height. Outputs ~/thesis-data/exp018/predictor_<dataset>/{predictor.pt,history.json,train.log}.

### 2026-11-08 13:07 — grid/offline/analysis driver written and tested
- src/edgelab/analysis/exp018.py (run --group G / offline / check / analyse), tests/test_analysis_exp018.py (drop_frames_dir for the top-down group, pooled stats with a degenerate cell, D3 rule). Waiting for the C2 predictors (f18-pred) and the FA trainings (need the predictors for the GRU one) before running the grid.

### 2026-11-08 13:08 — C1 (STG-NF-FA) trainings launched
- Smoke runs OK (graph-fa 576 params SHT, 448 params UB; ~96% of windows survive the drop+fill selection). Lanes (CPU, 2 threads each, seed 0): tmux f18-fa-cv (ShanghaiTech then UBnormal, constant-velocity fills) and f18-fa-gru (waits for the predictor DONE file, then ShanghaiTech and UBnormal with GRU fills). Models -> ~/thesis-data/exp018/models/{SHT,UB}_fa-{cv,gru}/. Script ~/thesis-data/exp018/fa_lane.sh.
- C2 predictor (SHT) at epoch 7: held-out val error 0.0392 (GRU) vs 0.0585 (const-vel), still improving.
- 13:09 relaunched f18-fa lanes after fixing a dir-name bug in fa_lane.sh (first launch killed within a minute; no results affected).

### 2026-11-08 13:12 — tests
- Whole suite 319 passed, 10 skipped (baseline before this lane 281 + 10 skipped) before adding tests/test_pose_train.py (3: STG-NF training resumes to bit-identical weights, FA training with cv fill, predictor trainer). Final count at the end.

### 2026-11-08 13:32 — C2 predictors finished (CPU, 14,485 params)
- ShanghaiTech: 16 epochs (early stop, patience 3), 21.4 min. Held-out training videos (every 10th file, fixed masks): normalised keypoint error (fraction of body height on filled frames) GRU 0.0387 vs constant velocity 0.0585 vs hold-still 0.0663 (-34% vs const-vel). ~/thesis-data/exp018/predictor_ShanghaiTech/.
- UBnormal: 25 epochs, 7.5 min. GRU 0.0182 vs constant velocity 0.0235 vs hold-still 0.0275 (-23%). ~/thesis-data/exp018/predictor_UBnormal/.
- FA-gru SHT lane started 13:19 (waited for the predictor); FA-cv SHT running since 13:09.

### 2026-11-08 14:26 — FA models SHT done; grid started
- SHT_fa-cv (57 min?) and SHT_fa-gru (65.7 min) finished; UB_fa-cv finished 14:22; UB_fa-gru running. Grid lanes started: tmux f18-grid-a (SHT-L0 then SHT-L0-TD), f18-grid-b (SHT-L2); UB-L0 after UB_fa-gru. Logs ~/thesis-data/exp018/grid_<group>.log, rows rows.jsonl.
- 14:28 grid first launch crashed in fill_dir_gru (numpy int64 age not JSON-serialisable in the sidecar); fixed in predictor.fill_clip_gru (+ test of fill_dir_gru); relaunched. Rows written before the crash (SKIP etc.) are kept (resumable, deterministic).
- 14:31 second crash: GRU filler on a video where every pose was dropped (empty clip) -> fixed in fill_clip_gru (+ test); relaunched f18-grid-a/b.
- 14:43 UB_fa-gru finished (UB lanes done). Offline pass (no drops; released vs FA-cv vs FA-gru on 5 pose sets) started in tmux f18-offline. First grid cells (s=1,3 SHT-L0/L2): +C1 and DAPF (which use our retrained FA models) are *below* FILL-HOLD / +C2 (e.g. SHT-L0 s3 seed1: FILL-HOLD 65.05, +C1 61.19, +C2 65.24, DAPF 61.67). Not yet understood — checking offline FA accuracy and provenance alignment before concluding.
- 14:52 offline (no drops, released test poses, paper): released 85.94, FA-cv 82.40, FA-gru 83.65 -> FA costs 3.5 / 2.3 points offline (D3 threshold -1.0: will fail on this set).

### 2026-11-08 14:56 — driver
- group SHT-L2 finished (s [1, 3, 6], seeds [1, 2])

### 2026-11-08 14:58 — driver
- group SHT-L0 finished (s [1, 3, 6], seeds [1, 2])

### 2026-11-08 15:07 — driver
- offline pass finished
- 15:32 SHT-L0 and SHT-L2 grid cells finished (14:56/14:58); SHT-L0-TD running. UB-L0 grid started now (tmux f18-grid-c) — should have started at 14:42 when UB_fa-gru finished (my oversight, no effect on results). Offline all sets done (see rows).

### 2026-11-08 15:38 — driver
- group SHT-L0-TD finished (s [1, 3, 6], seeds [1, 2])

### 2026-11-08 16:20 — driver
- group UB-L0 finished (s [1, 3, 6], seeds [1, 2])
- 16:21 all 270 EXP-018 rows done (grid 240 + offline 30); analyse started (tmux f18-check).

### 2026-11-08 16:37 — results.md written
- D1 FAIL (25% cells positive; pooled -1.99 [-3.51,-0.29]); D2 FAIL (+C1 -2.45 [-3.89,-0.84], +C2 +0.51 [+0.16,+0.84]); D3 FAIL (FA -2.3..-4.0 offline); D4 PASS (+2.73 [+0.25,+5.58]). SKIP/FILL-HOLD reproduce EXP-010/010b exactly (48 rows). No degenerate cells. C3 untouched, GAU-FA not run (EXP-019 G1 failed).

### 2026-11-08 16:38 — final test count
- Whole suite: 324 passed, 10 skipped (baseline before this lane 281 + 10 skipped). Lane F finished; no tmux jobs of this lane left running.

### 2026-11-08 16:39 — Lead review
- D1/D2/D3 fail, D4 pass as reported. **Confound:** C1 (STG-NF-FA) is our retrained model, while FILL-HOLD uses the
  released checkpoint; our standard retraining recipe already loses ~0.8–4 points offline (EXP-019 standard mean 85.10 vs
  released 85.94; FA models 2.3–4.0 below released). So "C1 hurts" mixes retraining loss with filling-awareness.
  The fair control — FILL-HOLD scored with the EXP-019 retrained standard models (same recipe, same masks) vs C1 — was not
  run; it needs a dated amendment before use. C2 (learned GRU predictor) is a small but significant gain (+0.51 [+0.16,
  +0.84] pooled). Top-down group rows are optimistic (masks applied to finished poses, no tracker re-run).
