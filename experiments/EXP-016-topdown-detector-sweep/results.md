# EXP-016 results — top-down pipeline (YOLOv8n person detector + BoT-SORT + RTMPose-s + frozen STG-NF)

Plan frozen 2026-11-07 18:30 (`plan.md`). Numbers: `results_table.md`, `verdicts.json`, raw `~/thesis-data/exp016/rows.jsonl`.
Pose model = RTMPose-s (EXP-015 rule). ShanghaiTech test, 107 videos, c 0.25, AUC x 100, 95% CI = bootstrap over videos.

## 0. Deviation D-1 (read first)
The first run (2026-11-07) built the person-box caches on MPS. Three of the four crashed (`MPS command buffer execution failed`)
after 4 / 11 / 94 of 107 videos (L0 n960, L2 n960, L2 n640) and the driver scored the incomplete caches. Those six rows
(AUC ~49 at n960 on 4-11 videos; 68.67 / 64.49 for L2 n640 on 94 videos) are **invalid**: they are kept in `rows.jsonl`
labelled `invalid` and listed (not used) at the bottom of `results_table.md`. The three caches were rebuilt from scratch **on CPU**
(old ones kept as `*_broken_mps`); L0 n640 was complete (107/107, MPS, parity rule passed) and its rows stand. The brief named
only the n960 caches; L2 n640 was found incomplete during the audit and repaired too. Guard added: `check_cache_complete`
(exactly 107 finished npz) is called by `poses_for_dir`, by `exp016.run_config` (also checks 107 pose files) and at the end of the
`cache` command; verdicts refuse to run without all four valid L0 rows. Tests: `tests/test_pose_topdown.py`,
`tests/test_analysis_exp015_016.py` (17 tests in the two files). Log: `log.md` 2026-11-08 07:25.

## 1. Corrected table (all rows on complete caches)
| config | L0 paper AUC [95% CI] | L0 live | L0 anom. / normal coverage | L2 paper | L2 live | L2 anom. coverage | person-frames L0 | VM ms/frame mean (p50, p99) |
|---|---|---|---|---|---|---|---|---|
| baseline YOLOv8n-pose 640 (EXP-003) | 68.53 [63.79, 73.60] | 66.17 | 0.674 / 0.535 | 67.05 | 66.11 | 0.633 | 69,895 | see EXP-004 |
| 640 px, K=1 | 80.52 [77.20, 83.71] | 76.23 | 0.915 / 0.737 | 79.34 | 75.71 | 0.871 | 144,187 | 71.8 (p50 73.6, p99 133.8) |
| 640 px, K=3 | 76.40 [72.41, 80.20] | 73.31 | 0.896 / 0.735 | 73.78 | 71.81 | 0.854 | 132,975 | 44.6 (35.9, 130.4) |
| 960 px, K=1 | 83.11 [80.15, 85.92] | 78.42 | 0.941 / 0.763 | 80.96 | 76.92 | 0.898 | 162,679 | 111.6 (113.2, 178.8) |
| 960 px, K=3 | 78.68 [75.15, 82.19] | 75.11 | 0.928 / 0.757 | 76.47 | 73.79 | 0.880 | 151,570 | 58.5 (55.8, 161.9) |

## 2. Verdicts
| # | rule | measured | verdict |
|---|---|---|---|
| T1 | best config L0 paper AUC >= 68.53 + 2.0 = 70.53 | best = 960 px K=1, 83.11; paired diff vs single-stage baseline **+14.58 [+10.15, +18.72]** (640 px K=1: 80.52, +11.99 [+7.71, +16.24]) | **PASS** |
| T2 | K=3 AUC >= K=1 AUC - 1.0 at each detector size | 640: 76.40 vs 80.52 (-4.12); 960: 78.68 vs 83.11 (-4.43) | **FAIL** at both sizes |
| T3 | descriptive: VM ms/frame per config vs the 41.7 ms budget | mean VM ms/frame vs 41.7 ms: 640 K=1 71.8 (74% of frames over budget), 640 K=3 44.6 (46%), 960 K=1 111.6 (100%), 960 K=3 58.5 (58%). **No config meets 24 fps on the 4-vCPU VM at its mean**; 640 K=3 is closest (p50 35.9 ms, under budget). K=3 cuts detector ms/frame by ~65% (34.3→12.1 at 640, 70.9→24.0 at 960) | descriptive |

Changes against the invalid first table: T1 now passes with a larger margin (the 960 px rows were the invalid ones); T2 at 960 px
flips from "PASS" (computed on 4 videos) to **FAIL**; the L2 columns changed from 64-69 to 74-81 at 640 px.

## 3. Reading and threats
- Larger detector input helps (L0 +2.6 at K=1, +2.3 at K=3), K=3 costs about 4 points of AUC at either size: the tracker-predicted
  boxes between keyframes lose people who enter between keyframes and drift on movers (pose count -8% at 640 px).
- The gain over the single-stage baseline is largely **coverage**: top-down poses are produced for every tracked box with no
  pose-confidence gate (144k vs 70k person-frames; anomalous-frame coverage 0.92 vs 0.67), not only a keypoint-quality gain.
  EXP-015 showed RTMPose-s on the correct people gives 86.1 (oracle), so the top-down configs recover most of that gap.
  Part of the gain may also come from RTMPose boxes being tracked boxes (smoothed), which this experiment does not separate.
- L2 uses the EXP-003 darkening without an enhancer; the L2 rows are descriptive (pre-registered verdicts use L0).
- T3 is on the Lima VM (not a Pi), 1 repetition per config, descriptive only; the 4 vCPU VM is not comparable to a Pi 5 core. Run 2026-11-08 16:43-16:52, mean host load outside the VM 13.8-15.7% per run (< 30%, none excluded; a single-core CPU replay of EXP-017 Y5 was running on the host, included in that figure). Pose model costs 23-30 ms/frame (RTMPose-s, 2.2-2.8 boxes/frame), so keyframing only removes the detector share. The first T3 attempt (chain, 16:19) crashed before any VM run (manifest bug, fixed with a test); the VM subset is 12 videos x 40 frames, so ms/frame is a subset estimate. Y4 (EXP-017) ran before T3, not overlapping.
- The 640 px caches differ in device: L0 n640 on MPS (parity-checked: max conf diff 1.8e-6, equal counts 100%), the other
  three on CPU. The device is recorded in each npz meta.
- Single detector family (YOLOv8n, COCO person class) and single seed for the darkening noise.
- The first run's invalid rows should not be cited.
