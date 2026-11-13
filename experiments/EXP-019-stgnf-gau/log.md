# EXP-019-stgnf-gau log (append-only, newest last)

### 2026-11-08 12:23 — lane F start
- Read specs, STATUS-NOW, EXP-001/009/010/010b code, STG-NF third_party model. Plan: edgelab/models/stgnf_flow.py (GAU + graph subnets, provenance conditioning), edgelab/pose/{segments,augment,predictor,stgnf_train,score_model}.py, analysis/exp019.py, analysis/exp018.py. No third_party edits.

### 2026-11-08 12:30 — code built + tested (flow, segments, augment, predictor)
- New: src/edgelab/models/stgnf_flow.py (graph subnet == third-party: 616 params, bit-equal nll with copied weights; GAU subnet 5808 params = 9.4x; provenance condition; GAU key mask), pose/segments.py (windows == third-party gen_dataset), pose/augment.py (STG-NF normalisation == normalize_pose incl. flip; random keep masks via latest-frame simulator; window cutting; provenance), pose/predictor.py (GRU 14K params; batched const-velocity == fill_P; causality + reset tests). Tests: tests/test_models_stgnf_flow.py, test_pose_segments.py, test_pose_augment.py, test_pose_predictor.py (+ tests/pose_toy.py helper). No third_party edits.

### 2026-11-08 12:55 — EXP-019 training launched
- (details of the 12:55 launch) Recipe per plan/EXP-001: 8 epochs, batch 256, Adamax 5e-4 (x0.99^epoch), clip 100, stride 6, identity+mirror, 145,162 windows (bank cached ~/thesis-data/exp019/banks). standard = third_party STG_NF class driven by edgelab.pose.stgnf_train; GAU = edgelab.models.stgnf_flow (5,808 params vs 616 = 9.43x). Seeds 0,1,2. CPU only, 2 threads/run (host load avg ~18: other agents' jobs, several on MPS, so no MPS here). Smoke: ~0.4 s/batch (standard), ~1.3 s/batch (GAU) under load => est. 1-1.5 h per standard run, 3-4 h per GAU run. Logs ~/thesis-data/exp019/logs, runs ~/thesis-data/exp019/models/<kind>_s<seed>/{train.log,last.pt,final.pt,DONE}. Restart: ~/thesis-data/exp019/train_lane.sh <threads> kind:seed ... (resumes from last.pt).

### 2026-11-08 13:03 — driver
- eval pass finished: released

### 2026-11-08 13:07 — analysis driver written and tested
- src/edgelab/analysis/exp019.py (eval / timing / analyse), tests/test_analysis_exp019.py (seed-paired bootstrap detects a planted gain, not a null; timing harness). Released checkpoint through the driver: released test poses paper 85.94 / live 80.98 (the EXP-001 number), YOLOv8n-pose edge poses paper 68.53 / live 66.17. Rows: ~/thesis-data/exp019/rows.jsonl (tmux f19-eval ran the released arm only; trained arms are added when their final.pt exists).

### 2026-11-08 13:12 — tests
- Whole suite 319 passed, 10 skipped (baseline before this lane 281 + 10 skipped) before adding tests/test_pose_train.py (3: STG-NF training resumes to bit-identical weights, FA training with cv fill, predictor trainer). Final count at the end.
- 14:13 standard_s0 finished training (45 min under load); eval started (tmux f19-eval2).

### 2026-11-08 14:16 — driver
- eval pass finished: released, standard_s0
- 14:52 standard_s1 finished; eval loop (tmux f19-evalloop) scores every new final.pt until all 6 models are done.

### 2026-11-08 14:57 — driver
- eval pass finished: released, standard_s0, standard_s1

### 2026-11-08 15:02 — driver
- eval pass finished: released, standard_s0, standard_s1

### 2026-11-08 15:07 — driver
- eval pass finished: released, standard_s0, standard_s1

### 2026-11-08 15:12 — driver
- eval pass finished: released, standard_s0, standard_s1

### 2026-11-08 15:18 — driver
- eval pass finished: released, standard_s0, standard_s1

### 2026-11-08 15:23 — driver
- eval pass finished: released, standard_s0, standard_s1

### 2026-11-08 15:28 — driver
- eval pass finished: released, standard_s0, standard_s1

### 2026-11-08 15:33 — driver
- eval pass finished: released, standard_s0, standard_s1

### 2026-11-08 15:40 — driver
- eval pass finished: released, standard_s0, standard_s1, standard_s2

### 2026-11-08 15:58 — driver
- eval pass finished: gau_s0, gau_s1, gau_s2, released, standard_s0, standard_s1, standard_s2
- 15:59 all 6 models trained + scored (eval loop done). Timing run started (tmux f19-timing).
- 16:01 timing (load avg ~4, quiet): standard 0.936 ms, GAU 1.928 ms per window (batch 1, 1 thread). analyse started (tmux f19-analyse).

### 2026-11-08 16:21 — results.md written
- G1 FAIL (-0.78, CI over videos [-2.16, +0.63]), G2 PASS (edge +0.59, top-down +0.04), G3 PASS (GAU 1.93 ms vs standard 0.94 ms). GAU-FA does not enter EXP-018. results.md + results_tables.md in this folder.

### 2026-11-08 16:38 — final test count
- Whole suite: 324 passed, 10 skipped (baseline before this lane 281 + 10 skipped). Lane F finished; no tmux jobs of this lane left running.
