# EXP-009 log (append-only)

### 2026-11-07 15:30 — Lane 4: start
- Read plan.md (frozen), litcheck, replay/score_dir/stgnf_eval/delivered/deadline_*, exp003/exp008. Baseline pytest: 150 passed, 10 skipped (per brief).
- Plan: fill.py (fill_B/fill_P/fill_dir + provenance sidecar in a sibling `*_prov` dir), exp009.py driver, tests, tmux runs (SHT and UB in parallel).
- Frame-index alignment spot check (01_0014): released JSON frames 0..264, cache frame_ids 0..264 (0-based = jpg numbering). Full check over 107 videos is logged with D1.

### 2026-11-07 15:35 — implementation done, runs started
- Added src/edgelab/pose/fill.py (fill_B, fill_P, fill_clip, fill_dir, segment_predicted_shares), src/edgelab/analysis/exp009.py (driver: run --part SHT|UB, analyse), tests/test_pose_fill.py (13), tests/test_analysis_exp009.py (6): 19 new tests pass.
- Conventions fixed where plan is silent: P step k moves velocity*0.8^k (k>=1), velocity from the last two DETECTED frames /frame difference (0 if one detection), P also extrapolates after the last detection of a track but never beyond the last video frame; B only inside gaps <= G; scores (box conf) are filled like keypoint conf; sidecar = {pid: {frame: age}} for predicted poses only, in sibling dir `<out>_prov`.
- G=0 baselines are the existing EXP-003/006 rows (same cache, c 0.25, same scorer) copied as condition '<group>/offline/none/G0'.
- B-gate: same STG-NF segments as B G8, windows with predicted share > 0.5 removed (keep share <= 0.5); rows condition '<group>/offline/B/G8-gate'.
- tmux e9-SHT and e9-UB started (threads 3 each); logs ~/thesis-data/exp009/run_{SHT,UB}.log, rows ~/thesis-data/exp009/rows.jsonl.

### 2026-11-07 15:37 — check: new scorer reproduces EXP-003 baseline
- Re-scoring exp003/poses/L0_E0_c0.25_botsort with exp009.score_condition: paper AUC 0.6852887160860734, coverage_anomalous 0.67413, segments 50019 = identical to results.jsonl (so copied G=0 rows are valid).

### 2026-11-07 15:41 — driver
- D1 filter: {'poses_total': 187021, 'poses_kept': 74063, 'released_frames_missing_in_cache': 0, 'kept_by_frame_shift': {'-1': 73631, '0': 74063, '1': 73762}}

### 2026-11-07 15:45 — problem: KeyError in D2 fill (fixed), SHT session restarted
- e9-SHT crashed in D2: one released video (08_0044) stores frame keys as 3 digits ('068'), the rest 4 digits. Fill code assumed 4 digits. Fix: new keys use the width of the track's own keys (STG-NF sorts keys as text, so mixing widths would reorder poses). Test added (tests/test_pose_fill.py, 14 tests). Rows already written (ref, D1, offline L0 B/P) are unaffected (4-digit tracks); the session resumes from rows.jsonl.
- D1 results so far: ref 85.94 reproduced (paper), D1 69.23. Alignment check (poses kept by IoU at frame shift -1/0/+1): 73631 / 74063 / 73762 of 187021; released frame ids missing in cache: 0 (all 107 videos). Shift 0 keeps the most poses, but only 0.4-0.6% more than +-1 (people move slowly), so this is consistent with, not strong proof of, 0-based alignment; the ids themselves are both 0-based and equal ranges.

### 2026-11-07 15:45 — frame alignment check (stronger version)
- Every 4th released video, all poses with a keypoint box: mean best IoU against cached boxes (conf >= 0.25) at frame shift -2/-1/0/+1/+2 = 0.288 / 0.305 / 0.310 / 0.306 / 0.289; share with best IoU >= 0.3 = 0.491 / 0.504 / 0.506 / 0.505 / 0.496. Peak at shift 0 and symmetric: released frame ids and cache frame ids are aligned (both 0-based, = jpg numbering). The absolute level is low because many released poses have no edge-detector counterpart (that is what D1 measures).

### 2026-11-07 15:49 — problem: no scorable window at s = 2 without fill (handled by the EXP-003 rule), SHT session restarted
- SHT-L0 deadline s = 2, mode none: STG-NF's loader raised IndexError (index 0 out of bounds, size 0): there are tracks >= 24 poses but no window with <= 2 missing frames. Same situation as the EXP-003 'no_segments' rule (log 2026-11-04 23:37: constant output -> AUC 0.5, coverage 0, row kept); score_condition now catches IndexError as well as ValueError and writes that row (no_segments = true). Not a new analysis rule; only the exception type differs. s = 1 rows (none 68.68, P 69.94, B 69.46 paper) are complete.
- 15:50 UB session restarted (e9-UB2) to pick up the IndexError handling; resumable.

### 2026-11-07 16:12 — driver
- part SHT finished

### 2026-11-07 16:14 — driver
- part UB finished

### 2026-11-07 16:17 — all runs finished, analysis written
- 76 conditions x 2 variants = 152 rows in ~/thesis-data/exp009/rows.jsonl (SHT 54 conditions, UB 22); no row missing. MLflow experiment 'EXP-009' (one run per condition x variant, scored conditions only; G=0 rows are copied from EXP-003/006). summary.json + results.md produced by `exp009 analyse`.
- Verdicts (paper variant unless stated): P1 pass (B G8 - G0 = +0.91 AUC pts, paired 95% CI [+0.03, +1.81]; live +0.56 [-0.25, +1.35], not significant; anomalous-frame coverage +4.9 pts [3.3, 7.0], normal +3.8 [2.3, 5.4]). P2 pass but trivial (see below). P3 FAIL (k=2+B 65.81 vs 68.53 = -2.72 paper; live 64.34 vs 66.17 = -1.83). P4 FAIL (B >= P in 0 of 4 groups at G8; P is higher in all four). 
- D: ref 85.94, D1 69.23, D2 71.99, E 68.53 (paper) -> D1 explains 96% of (ref - E); D2 - D1 = +2.76 (live: 91%, +1.53).
- Deadline caveat: at s = 2 and s = 4 the no-fill arm has NO scorable window in any group (drop rate 0.40 / 0.70 leaves every 24-pose window with > 2 gaps), so its AUC is the pre-registered 0.5 and P2 (B > none at s = 2) is satisfied by construction in 3/3 groups. The informative comparison is fill vs fill and vs s = 1.
- R: SHT-L0 B G8: frames scored from mostly-predicted windows (share > 0.5): only 179 frames (63 anomalous), AUC 54.1 (paper) / 54.2 (live); detected-only windows: 21,775 frames, AUC 75.5 / 70.2. s = 2 B: every window contains predicted poses, so 'no predicted' is empty; mostly-predicted subset 1,140 frames, AUC 60.4 (paper) / 50.8 (live). Tiny subsets: descriptive only.
- pytest: full suite 170 passed, 10 skipped (was 150 + 10 skipped).

### 2026-11-07 16:20 — Lead: review + exploratory fair comparison
- Verified G0 = EXP-003 (68.53 / 66.17). Paired fill-vs-skip comparison (live) computed against EXP-008 delivered
  scores (same traces/seeds): 5/6 significant (results.md, Lead review). Labelled exploratory.
