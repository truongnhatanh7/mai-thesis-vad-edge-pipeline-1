# EXP-016 log (append-only)

### 2026-11-07 18:42 — Lane D: code and smoke
- Detector weights: yolov8n.pt (COCO detection, person class 0 only) downloaded from https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt to artifacts/EXP-016/models/yolov8n.pt, sha256 f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36 (AGPL-3.0, research use only, D11). Pose models: the EXP-015 RTMPose files (sha256 in EXP-015/log.md).
- Static ONNX for the VM (host export, opset 17, no simplify, predictor input rectangle for 856x480 frames): ~/thesis-data/exp016/models/yolov8n-640-384x640.onnx sha256 4203cd4b3a23bbf2846f0aa9f03f8cb37fd2fc70b780911cf16ec6595811d668; yolov8n-960-544x960.onnx sha256 0b85126ff1416a2218bf038ea4d6a761ffca15b5900b62c7baa698b6547772cc.
- ORT person decode vs Ultralytics (classes=[0], iou 0.7, 8 frames of 06_0144, both sizes, conf 0.25 and 0.01): equal box counts on 8/8 frames, max box difference 0.16 px (640) / 0.26 px (960), max conf difference 0.
- Code: src/edgelab/pose/topdown.py (person-box cache, tracking with keyframes, pose on tracked boxes), src/edgelab/analysis/exp016.py (driver, table, T1/T2), src/edgelab/pipeline/{topdown_timing,timing_exp016}.py + scripts/cell_in_vm.sh (T3). Tests: tests/test_pose_topdown.py (8), tests/test_pipeline_topdown_timing.py (4), tests/test_analysis_exp015_016.py (4, shared with EXP-015).
- Conventions fixed before any scored run (plan silent): (1) K = 3: frame index i (0-based within the video) is a keyframe when i % 3 == 0; on other frames every active track gets one Kalman step (BoT-SORT `multi_predict`, no detector, no association) and its predicted box is used; tracks that are new on a keyframe need one more keyframe match before the tracker outputs them (Ultralytics' own confirmation rule), so a person who first appears between keyframes gets poses about 3-6 frames later. (2) Boxes for the pose model are the tracker's output boxes clipped to the frame (not the raw detections), RTMPose-standard 1.25 padding. (3) L2 = EXP-003 darkening at level L2, no enhancer (E0); the darkened frame is what both the detector and the pose model see; comparison baseline = EXP-003 L2_E0 (67.05 paper / 66.11 live). (4) conf gate c = 0.25 on the cached person boxes (cache keeps conf >= 0.01); BoT-SORT gmc none, no ReID, same thresholds as replay.py. (5) Pose model = EXP-015 rule: RTMPose-s unless O2 passes, then the better of s and t on the paper AUC. (6) T1/T2 verdicts use L0 paper AUC, the pre-registered threshold line; L2 is reported beside them. (7) T3: EXP-004 cell L0-E0-c0.25-O0-B4 (12 videos x 40 frames, library-default threads, 4 vCPU, ffmpeg co-load), 1 rep per config, host-load rule of EXP-004 (mean outside-VM load <= 30%, else wait and retry); service_ms = read + detect (keyframes) + track + pose + STG-NF update.
- Smoke on 06_0144 (host, L0, 640): K=1 175 poses / 5 tracks; K=3 152 poses / 4 tracks (see tests for the keyframe rules).
- tmux `e16` runs EXP-016/run.sh: waits for the lock, MPS-vs-CPU parity rule for the detection weights, 4 caches, waits for EXP-015, 8 configs, table. VM timing afterwards.

### 2026-11-07 19:57 — driver
- lock present, heavy run starts

### 2026-11-07 19:58 — parity
- MPS vs CPU rule passed (plan 4.3: max conf diff <= 0.01, equal counts on >= 99% frames): [{"video": "01_0014", "imgsz": 640, "frames": 50, "frames_equal_count": 50, "max_conf_diff": 1.8477439880371094e-06}, {"video": "01_0014", "imgsz": 960, "frames": 50, "frames_equal_count": 50, "max_conf_diff": 3.2782554626464844e-06}, {"video": "06_0144", "imgsz": 640, "frames": 50, "frames_equal_count": 50, "max_conf_diff": 7.152557373046875e-07}, {"video": "06_0144", "imgsz": 960, "frames": 50, "frames_equal_count": 50, "max_conf_diff": 1.4901161193847656e-06}]; caches on mps

### 2026-11-07 20:13 — cache L0 n640
- 107 npz files (107 expected, device mps)

### 2026-11-07 20:14 — cache L0 n960
- 4 npz files (107 expected, device mps)

### 2026-11-07 21:59 — cache L2 n640
- 94 npz files (107 expected, device mps)

### 2026-11-07 22:10 — cache L2 n960
- 11 npz files (107 expected, device mps)

### 2026-11-07 22:30 — pose model
- from EXP-015 verdicts.json by the plan's rule: rtmpose-s

### 2026-11-08 01:45 — driver
- 8 configs scored; rows in /Users/nanh/thesis-data/exp016/rows.jsonl; {"config": "L2_n960_K3_rtmpose-s", "variant": "live", "auc": 0.5452165747286237, "coverage_anomalous": 0.06487360036938705, "poses": 12202}

### 2026-11-08 01:46 — driver
- table in results_table.md; EXP016_QUALITY_DONE (VM timing next)

### 2026-11-08 07:25 — DEVIATION D-1 (Lane E): MPS-crashed caches were scored; repair
- What happened: the 2026-11-07 run built the four person-box caches on MPS. Three of them crashed with `torch.AcceleratorError: MPS command buffer execution failed: Impacting Interactivity (kIOGPUCommandBufferCallbackErrorImpactingInteractivity)` under host contention (EXP-013 was also on MPS): L0_n960 after 4 of 107 videos, L2_n960 after 11 of 107, and **L2_n640 after 94 of 107** (the 20:14 / 21:59 / 22:10 log lines above show 4 / 94 / 11 npz files; the driver ignored them and the next step globbed whatever existed). The driver then posed and scored the incomplete caches: the rows for L0_n960 K1/K3 (6,865 / 6,552 person-frames, AUC ~49) and L2_n960 K1/K3 are invalid, and so are the rows for L2_n640 K1/K3 (94 of 107 videos, 36,074 of 40,791 frames; 113,672 / 104,315 poses). The task brief named only the n960 caches; the L2_n640 cache turned out to be incomplete as well (found while checking npz counts) and is repaired the same way. L0_n640 was complete (107/107) and its rows stand.
- Consequence for earlier verdicts: T1 and T2 at n640 were computed on L0_n640 (complete) and are not affected by the L2 problem; T2 at n960 and the L2 columns were computed on incomplete data and are withdrawn. All of them are recomputed after the repair.
- Repair (no change to the protocol, only the device): old caches moved aside as `~/thesis-data/exp016/cache/{L0_n960,L2_n960,L2_n640}_broken_mps`; old poses/scores kept in `poses_pre_repair/`, `scores_pre_repair/`; old rows file copied to `rows_pre_repair.jsonl` and, in `rows.jsonl`, the invalid rows are kept and labelled `"invalid": "incomplete MPS-crashed cache (n/107 videos)"`; they are excluded from scoring, verdicts and the main table, and listed in a separate "invalid" block of the results. The three caches are rebuilt from scratch **on CPU** (torch threads 3, no MPS use, because EXP-013 is training on MPS); L0_n640 stays the complete MPS cache (parity rule passed: max conf diff 1.8e-6, equal counts 100%).
- Guard: `edgelab.pose.topdown.check_cache_complete` (107 finished npz, `.partial.npz` ignored) is called by `poses_for_dir`, by `exp016.run_config`, and at the end of the `cache` command (non-zero exit if anything is missing). Test: tests/test_pose_topdown.py.
- tmux `e16fixA` (L0 n960, then L2 n640) and `e16fixB` (L2 n960) run `run_repair_caches.sh`, status in ~/thesis-data/exp016/repair_status.txt.

### 2026-11-08 07:24 — Lead: completeness audit of the pre-repair caches
- L0_n640 107/107 ✔ (the +11.99 T1 result stands). **L2_n640 94/107, L0_n960 4/107, L2_n960 11/107** — all three crashed
  on MPS ("command buffer execution failed") and were scored incomplete → every L2 and n960 row in the first
  results_table.md is invalid. Lane E rebuilding all three on CPU.

### 2026-11-08 08:04 — repair
- L0_n960 CPU cache complete (107 npz); posing + scoring L0 n960 K1/K3

### 2026-11-08 08:30 — Lane E: session crash and resume
- The Cursor session of Lane E was killed at ~08:2x. The tmux jobs survived: e16fixA (L2 n640 cache rebuild, 24/107 at 08:28), e16fixB (L2 n960, 54/107), e16fixP (waits per cache, now posing/scoring L0 n960 K1: 78 pose files at 08:28). L0_n960 cache finished 08:04 (107/107). Nothing was lost; no step repeated. Code changes done before the crash: topdown.py (`check_cache_complete`, `--threads`), exp016.py (invalid-row handling, guards), tests (17 pass in the two test files).

### 2026-11-08 08:41 — Lane E: T3 could not have run as written; fixed and smoke-tested (smoke numbers are not results)
- The Lima VM `edge-os` mounts only `vm-exchange` (rw) and `experiments/` (ro) (`~/.lima/edge-os/lima.yaml`). `topdown_timing.py` loaded its detector and RTMPose ONNX from `~/thesis-data/exp016/models` and `~/thesis-data/exp015/models`, and `timing_exp016.py` wrote traces to `~/thesis-data/exp016/timing`: none of these exist in the VM. A 3-frames-per-video smoke run confirmed `NoSuchFile` for the detector ONNX.
- Fix: models copied to `~/thesis-data/vm-exchange/models/exp016/` (yolov8n-640-384x640.onnx sha256 4203cd4b…d668, yolov8n-960-544x960.onnx sha256 0b85126f…72cc: both equal to the hashes logged on 2026-11-07 18:42) and `.../exp015/` (rtmpose-s-body7-256x192.onnx sha256 9aeb635b…f5, rtmpose-t sha256 a6c2f6a3…0f); `topdown_timing.py` reads them there; `timing_exp016.py` writes to `vm-exchange/results/exp016-timing/`; `exp016.report` reads the summary from there. No change to what is timed.
- Smoke (3 frames per video, 12 videos, VM B4/O0 with co-load, host busy with my CPU jobs at the time): n640 K=3 and n960 K=1 both ran to the end (36 frames each, `machine=vm`); not used anywhere. T3 proper waits for a quiet host (see EXP-017 log 08:38 for the shared chain).

- Erratum (08:46): the 'T3 could not have run' entry was first stamped 08:50 by mistake; real time ~08:41.

### 2026-11-08 09:13 — repair
- L0 n960 scored; {"config": "L0_n960_K3_rtmpose-s", "variant": "live", "auc": 0.7511223904765163, "coverage_anomalous": 0.9312593789680249, "poses": 151570}

### 2026-11-08 09:51 — repair
- L2_n960 and L2_n640 CPU caches complete (107 npz each); posing + scoring L2 K1/K3 x n640/n960

### 2026-11-08 10:56 — Lane E: session cut again, resumed
- Cursor disconnect ~10:5x; tmux jobs all kept running (e16fixP, e16e17vm, e17A2, e17C, e17SweepS, e17SweepU). State: EXP-016 repaired caches 107/107 each, L0 n960 K1/K3 scored, L2 posing running; EXP-017 caches: SHT L0 107 (e2e), UB L0 211, SHT L2 103/107, L3 pending, o2m 8/107; VM timing chain (T3 then Y4) waits for EXP-013/014 chain end (evalwise.done) and for Lane E CPU jobs.

### 2026-11-08 12:56 — repair
- L2 n640 K1/K3 scored; main loop stopped, L2 n960 K1/K3 running in parallel processes

### 2026-11-08 14:38 — repair
- all 8 configs rescored; results_table.md and verdicts.json regenerated (EXP016_REPAIR_QUALITY_DONE)

### 2026-11-08 14:45 — Lane E: repair scoring finished (all 8 configs on complete caches)
- All eight configs scored on complete caches (107 npz each; 107 pose JSON each, all parse). L0 paper AUC: n640 K1 80.52, K3 76.40; n960 K1 83.11, K3 78.68. L2 paper: 79.34 / 73.78 / 80.96 / 76.47. T1 PASS (best n960_K1 83.11 >= 70.53; paired diff vs the single-stage baseline +14.58 [+10.15, +18.72]); T2 FAIL at both sizes (n640: 76.40 vs 80.52; n960: 78.68 vs 83.11). results_table.md and verdicts.json regenerated by `exp016 report` (invalid rows listed separately).
- Process note: the L2 n960 configs ran in parallel processes (run_repair_poses_parallel.sh, decided when host load dropped); the main loop was stopped at 12:56 right after L2 n640 K3 was scored and may have rewritten the first pose JSON(s) of L2_n960_K1 for a few seconds (identical content; a truncated file would have failed the scoring step; none did: all 107 parse and scoring completed).
- Threat to note in results.md: top-down poses are produced for every tracked box (144,187 person-frames at n640 K1 vs 69,895 for the single-stage baseline) with no pose-confidence gate, so part of the gain is coverage (anomalous coverage 0.92 vs 0.67); the EXP-015 oracle result says what RTMPose alone can do.

### 2026-11-08 16:19 — T3 start
- EXP-013 chain finished (evalwise.done) and Lane E CPU jobs done; host load now 1.95 3.37 4.28; running T3 (timing_exp016, 1 rep per config, host-load rule 30%)

### 2026-11-08 16:19 — T3 done
- cell_id,rep,order,start,end,status,excluded,host_outside_vm_pct,host_total_pct,vm_cpu_pct,n_samples,coload_cpu_s,trace_file,error 

### 2026-11-08 16:42 — T3 crash, fix, relaunch
- The T3 driver in the chain crashed at 16:19 (pandas EmptyDataError): a fresh manifest had its header unflushed when `done_configs()` re-read it. No VM run happened; the manifest was removed (header only). Fix: `f.flush()` after the header + regression test (`test_run_on_fresh_manifest_does_not_crash`, 5 tests pass in that file). The chain then went on to Y4 (ran fine, see EXP-017 log), so order is Y4 then T3 (no overlap, deviation from the planned order, harmless). Relaunching T3 alone in tmux `e16t3` (host-load rule 30%); the chain is still running Y5 (CPU replay, no VM).

### 2026-11-08 16:53 — T3 done, results.md completed
- T3 (rtmpose-s, 1 rep/config, 12 videos x 40 frames, host outside-VM 13.8-15.7%, none excluded) mean VM ms/frame: n640 K1 71.8, n640 K3 44.6, n960 K1 111.6, n960 K3 58.5 (budget 41.7). results_table.md regenerated with the column; results.md placeholders filled. Source: vm-exchange/results/exp016-timing/{manifest.csv,summary.json}.
