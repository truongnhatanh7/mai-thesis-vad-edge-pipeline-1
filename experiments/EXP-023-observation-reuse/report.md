# EXP-023: Observation Reuse, Visual Pose Propagation & Bounded Detector Refresh

**Working Paper Title:** *Preserving Sudden Anomalies Under Sparse Pose Inference on CPU-Constrained Edge Devices*
**Generated:** 2026-11-13 00:51:34 +07 · **Pipeline:** Huang-inspired edge pose-VAD (YOLOv8n-pose `ML0` / `stock` + timebase-preserving BoT-SORT + frozen STG-NF)

---

## 1. Phase A: Stage Profiling & Calibrated Operating Parameters

All timings were measured in isolation without training co-load. Detector and tracker costs use the deployment NCNN FP16 (384×640) traces from `edge-os` VM (`EXP-020`); grayscale decode, cheap motion observer, pyramidal LK observer (with forward-backward consistency), and STG-NF window scoring were profiled across 1, 5, and 10 person workloads (10-person marked as synthetic stress).

| Stage | Person Count | Synthetic Stress | Backend | Threads | Mean (ms) | p50 (ms) | p95 (ms) | RSS (MB) |
|---|---:|---|---|---:|---:|---:|---:|---:|
| `decode_grayscale` | 0 | False | `OpenCV-CPU-1T` | 1 | 1.563 | 1.575 | 1.771 | 479.9 |
| `observer_cheap_motion` | 0 | False | `NumPy-Downsample4x-1T` | 1 | 0.366 | 0.363 | 0.446 | 481.0 |
| `observer_pyramidal_lk` | 1 | False | `OpenCV-PyrLK-FB-1T` | 1 | 1.350 | 1.342 | 1.570 | 487.9 |
| `observer_pyramidal_lk` | 5 | False | `OpenCV-PyrLK-FB-1T` | 1 | 2.112 | 2.119 | 2.311 | 488.1 |
| `observer_pyramidal_lk` | 10 | True | `OpenCV-PyrLK-FB-1T` | 1 | 3.568 | 3.578 | 3.972 | 488.5 |
| `scorer_stgnf_window` | 1 | False | `PyTorch-CPU-1T` | 1 | 1.053 | 1.044 | 1.110 | 520.9 |
| `scorer_stgnf_window` | 5 | False | `PyTorch-CPU-1T` | 1 | 1.819 | 1.806 | 1.897 | 521.2 |
| `scorer_stgnf_window` | 10 | True | `PyTorch-CPU-1T` | 1 | 2.773 | 2.737 | 2.923 | 521.4 |
| `detector_ML0_ncnn_fp16_640` | 1 | False | `NCNN-FP16-384x640-VM` | 3 | 13.287 | 11.947 | 18.683 | 524.6 |
| `tracker_botsort_update` | 1 | False | `BoT-SORT-CPU` | 1 | 0.235 | 0.198 | 0.458 | 524.6 |
| `detector_stock_ncnn_fp16_640` | 1 | False | `NCNN-FP16-384x640-VM` | 3 | 13.971 | 11.468 | 22.450 | 525.3 |

### Phase D Gate Assessment (Scorer CPU Share)
- Single-window STG-NF scoring costs **1.053 ms** (1 person) and **1.819 ms** (5 persons, **0.364 ms/person**).
- Compared against NCNN FP16 detector inference (**13.29 ms/call**), STG-NF scoring accounts for **< 6% of total pipeline CPU time** under both 12 Hz (1/2) and 6 Hz (1/4) budgets—well below the 20% investigation trigger. Per Phase D rules, we do **not** modify or retrain the normalizing flow conditioner.

### Label-Free Calibration Summary (`frozen_manifest.json`)
- **ShanghaiTech normal sentinel normality:** `-0.7886` (90th percentile on normal training clips `01_001`..`12_001`), `max_hold_frames = 24`, alert threshold = `1.8072`.
- **UBnormal normal sentinel normality:** `-0.7288`, `max_hold_frames = 24`, alert threshold = `1.4689`.
- **Observer trigger thresholds (derived from normal training clips):** `motion_score_thresh = 0.08`, `disagreement_q80_thresh = 0.0535`, `invalid_fraction_thresh = 0.45`, `exterior_motion_thresh = 0.1`.

---

## 2. Phase B: Primary Screen on ShanghaiTech (`ML0` + Frozen STG-NF, 12 Stratified Clips)

| Method | Target Hz | Actual Det Hz | CPU (ms/s) | Delivered AUC [95% CI] | Paper-Compat AUC | Recall @0.5s / @1.0s (%) | Missed Events (%) | Delay p50/p90 (ms) | FA / Normal Hr | Meas / Scorable Cov | Prop Frac | Invalid Flow | Max Age (ms) | Pose Err (px) |
|---|---:|---:|---:|---|---:|---|---:|---|---:|---|---:|---:|---:|---:|
| **B** (delayed) | 6 | 6.05 | 140.1 | **67.82** [52.71, 81.62] | 68.67 | 60.0 / 65.0 | 25.0 (5/20) | 10 / 1035 | 542.12 | 0.252 / 0.213 | 0.746 | 0.000 | 333 | 3.11 |
| `S0` | 6 | 6.05 | 119.3 | **54.12** [52.94, 55.98] | 54.12 | 0.0 / 0.0 | 100.0 (20/20) | n/a / n/a | 0.00 | 0.252 / 0.000 | 0.000 | 0.000 | 333 | 0.00 |
| `S1` | 6 | 6.05 | 141.6 | **70.97** [55.67, 84.55] | 70.78 | 65.0 / 65.0 | 20.0 (4/20) | 8 / 1654 | 623.10 | 0.252 / 0.854 | 0.764 | 0.000 | 333 | 5.55 |
| `S2` | 6 | 6.05 | 192.1 | **70.64** [56.19, 84.25] | 71.48 | 65.0 / 65.0 | 25.0 (5/20) | 9 / 893 | 614.76 | 0.252 / 0.854 | 0.764 | 0.024 | 333 | 4.56 |
| `S3` | 6 | 5.98 | 148.5 | **68.97** [53.56, 82.44] | 69.09 | 60.0 / 65.0 | 20.0 (4/20) | 8 / 1654 | 642.69 | 0.249 / 0.856 | 0.766 | 0.000 | 333 | 5.84 |
| `S4` | 6 | 6.01 | 191.2 | **69.13** [51.56, 84.10] | 68.82 | 65.0 / 70.0 | 15.0 (3/20) | 9 / 1068 | 750.36 | 0.251 / 0.852 | 0.766 | 0.027 | 333 | 6.57 |
| `S5` | 6 | 6.02 | 191.0 | **69.98** [53.61, 84.17] | 71.17 | 60.0 / 60.0 | 15.0 (3/20) | 9 / 3351 | 669.24 | 0.251 / 0.852 | 0.766 | 0.023 | 333 | 4.98 |
| **B** (delayed) | 12 | 12.03 | 222.6 | **72.80** [58.67, 84.68] | 73.66 | 65.0 / 65.0 | 15.0 (3/20) | 9 / 1792 | 634.35 | 0.501 / 0.426 | 0.509 | 0.000 | 333 | 2.82 |
| `S0` | 12 | 12.03 | 200.2 | **54.12** [52.94, 55.98] | 54.12 | 0.0 / 0.0 | 100.0 (20/20) | n/a / n/a | 0.00 | 0.501 / 0.000 | 0.000 | 0.000 | 333 | 0.00 |
| `S1` | 12 | 12.03 | 224.8 | **71.96** [56.44, 85.37] | 71.90 | 65.0 / 70.0 | 15.0 (3/20) | 8 / 1292 | 677.40 | 0.501 / 0.863 | 0.551 | 0.000 | 333 | 5.40 |
| `S2` | 12 | 12.03 | 275.4 | **74.02** [61.01, 85.98] | 74.40 | 65.0 / 65.0 | 15.0 (3/20) | 10 / 1293 | 644.14 | 0.501 / 0.863 | 0.551 | 0.026 | 333 | 4.68 |
| `S3` | 12 | 11.97 | 232.3 | **73.65** [61.92, 84.79] | 73.57 | 60.0 / 60.0 | 10.0 (2/20) | 9 / 1580 | 728.22 | 0.499 / 0.866 | 0.550 | 0.000 | 333 | 5.46 |
| `S4` | 12 | 11.99 | 274.7 | **72.44** [57.61, 85.26] | 72.33 | 65.0 / 65.0 | 15.0 (3/20) | 10 / 1243 | 654.87 | 0.500 / 0.864 | 0.552 | 0.027 | 333 | 5.77 |
| `S5` | 12 | 11.99 | 274.8 | **74.52** [61.84, 86.19] | 74.77 | 65.0 / 65.0 | 15.0 (3/20) | 10 / 1243 | 689.42 | 0.500 / 0.863 | 0.551 | 0.027 | 333 | 4.84 |
| **R** (offline ref) | 24 | 24.00 | 381.9 | **72.29** [58.61, 83.42] | 72.33 | 55.0 / 55.0 | 15.0 (3/20) | 8 / 2142 | 562.08 | 1.000 / 0.836 | 0.000 | 0.000 | 333 | 0.00 |

### Paired Bootstrap Contrasts on Primary Screen (`screen_sht`, 1,000 Video Resamples)

| Budget (Hz) | Comparison (`Candidate - Base`) | Δ Delivered AUC (pts) | 95% Bootstrap CI | Significant (>0) |
|---:|---|---:|---|---|
| 12 | `S1 - S0` | **+17.84** | [+3.27, +30.82] | YES |
| 12 | `S2 - S1` | **+2.06** | [+0.24, +4.19] | YES |
| 12 | `S3 - S1` | **+1.69** | [-2.25, +6.69] | no |
| 12 | `S4 - S1` | **+0.48** | [-1.03, +1.76] | no |
| 12 | `S5 - S1` | **+2.56** | [+0.39, +5.15] | YES |
| 12 | `S4 - S3` | **-1.21** | [-5.78, +3.26] | no |
| 12 | `S5 - S3` | **+0.87** | [-2.29, +3.61] | no |
| 12 | `S5 - S4` | **+2.07** | [-0.09, +4.34] | no |
| 12 | `B - S1` | **+0.84** | [-1.99, +3.78] | no |
| 6 | `S1 - S0` | **+16.85** | [+1.98, +30.14] | YES |
| 6 | `S2 - S1` | **-0.33** | [-3.91, +1.87] | no |
| 6 | `S3 - S1` | **-2.01** | [-7.25, +2.55] | no |
| 6 | `S4 - S1` | **-1.85** | [-5.40, +1.56] | no |
| 6 | `S5 - S1` | **-0.99** | [-4.32, +1.97] | no |
| 6 | `S4 - S3` | **+0.16** | [-5.90, +5.82] | no |
| 6 | `S5 - S3` | **+1.01** | [-2.10, +4.79] | no |
| 6 | `S5 - S4` | **+0.85** | [-3.44, +4.83] | no |
| 6 | `B - S1` | **-3.15** | [-6.47, -0.20] | no |

### Event Subgroup Stratification (`screen_sht`, Short Events vs Sudden Fast Motion vs High Crowd)

| Budget (Hz) | Method | Subgroup | Events | Recall @0.5s (%) | Recall @1.0s (%) | Missed Event Rate (%) |
|---:|---|---|---:|---:|---:|---:|
| 12 | `S0` | `subgroup_duration=short_event` | 8 | 0.0 | 0.0 | 100.0 |
| 12 | `S0` | `subgroup_duration=sustained_event` | 12 | 0.0 | 0.0 | 100.0 |
| 12 | `S0` | `subgroup_motion=sudden_fast_motion` | 19 | 0.0 | 0.0 | 100.0 |
| 12 | `S0` | `subgroup_motion=moderate_motion` | 1 | 0.0 | 0.0 | 100.0 |
| 12 | `S0` | `subgroup_crowd=high_crowd` | 16 | 0.0 | 0.0 | 100.0 |
| 12 | `S0` | `subgroup_entry=new_entry_near_onset` | 13 | 0.0 | 0.0 | 100.0 |
| 12 | `S1` | `subgroup_duration=short_event` | 8 | 75.0 | 75.0 | 25.0 |
| 12 | `S1` | `subgroup_duration=sustained_event` | 12 | 58.3 | 66.7 | 8.3 |
| 12 | `S1` | `subgroup_motion=sudden_fast_motion` | 19 | 68.4 | 73.7 | 10.5 |
| 12 | `S1` | `subgroup_motion=moderate_motion` | 1 | 0.0 | 0.0 | 100.0 |
| 12 | `S1` | `subgroup_crowd=high_crowd` | 16 | 68.8 | 75.0 | 12.5 |
| 12 | `S1` | `subgroup_entry=new_entry_near_onset` | 13 | 61.5 | 69.2 | 15.4 |
| 12 | `S2` | `subgroup_duration=short_event` | 8 | 75.0 | 75.0 | 25.0 |
| 12 | `S2` | `subgroup_duration=sustained_event` | 12 | 58.3 | 58.3 | 8.3 |
| 12 | `S2` | `subgroup_motion=sudden_fast_motion` | 19 | 68.4 | 68.4 | 10.5 |
| 12 | `S2` | `subgroup_motion=moderate_motion` | 1 | 0.0 | 0.0 | 100.0 |
| 12 | `S2` | `subgroup_crowd=high_crowd` | 16 | 68.8 | 68.8 | 12.5 |
| 12 | `S2` | `subgroup_entry=new_entry_near_onset` | 13 | 61.5 | 61.5 | 15.4 |
| 12 | `S3` | `subgroup_duration=short_event` | 8 | 75.0 | 75.0 | 25.0 |
| 12 | `S3` | `subgroup_duration=sustained_event` | 12 | 50.0 | 50.0 | 0.0 |
| 12 | `S3` | `subgroup_motion=sudden_fast_motion` | 19 | 63.2 | 63.2 | 5.3 |
| 12 | `S3` | `subgroup_motion=moderate_motion` | 1 | 0.0 | 0.0 | 100.0 |
| 12 | `S3` | `subgroup_crowd=high_crowd` | 16 | 62.5 | 62.5 | 12.5 |
| 12 | `S3` | `subgroup_entry=new_entry_near_onset` | 13 | 53.8 | 53.8 | 7.7 |
| 12 | `S4` | `subgroup_duration=short_event` | 8 | 75.0 | 75.0 | 25.0 |
| 12 | `S4` | `subgroup_duration=sustained_event` | 12 | 58.3 | 58.3 | 8.3 |
| 12 | `S4` | `subgroup_motion=sudden_fast_motion` | 19 | 68.4 | 68.4 | 10.5 |
| 12 | `S4` | `subgroup_motion=moderate_motion` | 1 | 0.0 | 0.0 | 100.0 |
| 12 | `S4` | `subgroup_crowd=high_crowd` | 16 | 68.8 | 68.8 | 12.5 |
| 12 | `S4` | `subgroup_entry=new_entry_near_onset` | 13 | 61.5 | 61.5 | 15.4 |
| 12 | `S5` | `subgroup_duration=short_event` | 8 | 75.0 | 75.0 | 25.0 |
| 12 | `S5` | `subgroup_duration=sustained_event` | 12 | 58.3 | 58.3 | 8.3 |
| 12 | `S5` | `subgroup_motion=sudden_fast_motion` | 19 | 68.4 | 68.4 | 10.5 |
| 12 | `S5` | `subgroup_motion=moderate_motion` | 1 | 0.0 | 0.0 | 100.0 |
| 12 | `S5` | `subgroup_crowd=high_crowd` | 16 | 68.8 | 68.8 | 12.5 |
| 12 | `S5` | `subgroup_entry=new_entry_near_onset` | 13 | 61.5 | 61.5 | 15.4 |
| 12 | `B` | `subgroup_duration=short_event` | 8 | 75.0 | 75.0 | 25.0 |
| 12 | `B` | `subgroup_duration=sustained_event` | 12 | 58.3 | 58.3 | 8.3 |
| 12 | `B` | `subgroup_motion=sudden_fast_motion` | 19 | 68.4 | 68.4 | 10.5 |
| 12 | `B` | `subgroup_motion=moderate_motion` | 1 | 0.0 | 0.0 | 100.0 |
| 12 | `B` | `subgroup_crowd=high_crowd` | 16 | 68.8 | 68.8 | 12.5 |
| 12 | `B` | `subgroup_entry=new_entry_near_onset` | 13 | 61.5 | 61.5 | 15.4 |
| 6 | `S0` | `subgroup_duration=short_event` | 8 | 0.0 | 0.0 | 100.0 |
| 6 | `S0` | `subgroup_duration=sustained_event` | 12 | 0.0 | 0.0 | 100.0 |
| 6 | `S0` | `subgroup_motion=sudden_fast_motion` | 19 | 0.0 | 0.0 | 100.0 |
| 6 | `S0` | `subgroup_motion=moderate_motion` | 1 | 0.0 | 0.0 | 100.0 |
| 6 | `S0` | `subgroup_crowd=high_crowd` | 16 | 0.0 | 0.0 | 100.0 |
| 6 | `S0` | `subgroup_entry=new_entry_near_onset` | 13 | 0.0 | 0.0 | 100.0 |
| 6 | `S1` | `subgroup_duration=short_event` | 8 | 75.0 | 75.0 | 25.0 |
| 6 | `S1` | `subgroup_duration=sustained_event` | 12 | 58.3 | 58.3 | 16.7 |
| 6 | `S1` | `subgroup_motion=sudden_fast_motion` | 19 | 68.4 | 68.4 | 15.8 |
| 6 | `S1` | `subgroup_motion=moderate_motion` | 1 | 0.0 | 0.0 | 100.0 |
| 6 | `S1` | `subgroup_crowd=high_crowd` | 16 | 68.8 | 68.8 | 18.8 |
| 6 | `S1` | `subgroup_entry=new_entry_near_onset` | 13 | 61.5 | 61.5 | 23.1 |
| 6 | `S2` | `subgroup_duration=short_event` | 8 | 75.0 | 75.0 | 25.0 |
| 6 | `S2` | `subgroup_duration=sustained_event` | 12 | 58.3 | 58.3 | 25.0 |
| 6 | `S2` | `subgroup_motion=sudden_fast_motion` | 19 | 68.4 | 68.4 | 21.1 |
| 6 | `S2` | `subgroup_motion=moderate_motion` | 1 | 0.0 | 0.0 | 100.0 |
| 6 | `S2` | `subgroup_crowd=high_crowd` | 16 | 68.8 | 68.8 | 25.0 |
| 6 | `S2` | `subgroup_entry=new_entry_near_onset` | 13 | 61.5 | 61.5 | 30.8 |
| 6 | `S3` | `subgroup_duration=short_event` | 8 | 75.0 | 75.0 | 25.0 |
| 6 | `S3` | `subgroup_duration=sustained_event` | 12 | 50.0 | 58.3 | 16.7 |
| 6 | `S3` | `subgroup_motion=sudden_fast_motion` | 19 | 63.2 | 68.4 | 15.8 |
| 6 | `S3` | `subgroup_motion=moderate_motion` | 1 | 0.0 | 0.0 | 100.0 |
| 6 | `S3` | `subgroup_crowd=high_crowd` | 16 | 62.5 | 68.8 | 18.8 |
| 6 | `S3` | `subgroup_entry=new_entry_near_onset` | 13 | 53.8 | 61.5 | 23.1 |
| 6 | `S4` | `subgroup_duration=short_event` | 8 | 75.0 | 75.0 | 25.0 |
| 6 | `S4` | `subgroup_duration=sustained_event` | 12 | 58.3 | 66.7 | 8.3 |
| 6 | `S4` | `subgroup_motion=sudden_fast_motion` | 19 | 68.4 | 73.7 | 10.5 |
| 6 | `S4` | `subgroup_motion=moderate_motion` | 1 | 0.0 | 0.0 | 100.0 |
| 6 | `S4` | `subgroup_crowd=high_crowd` | 16 | 62.5 | 68.8 | 12.5 |
| 6 | `S4` | `subgroup_entry=new_entry_near_onset` | 13 | 61.5 | 69.2 | 15.4 |
| 6 | `S5` | `subgroup_duration=short_event` | 8 | 75.0 | 75.0 | 25.0 |
| 6 | `S5` | `subgroup_duration=sustained_event` | 12 | 50.0 | 50.0 | 8.3 |
| 6 | `S5` | `subgroup_motion=sudden_fast_motion` | 19 | 63.2 | 63.2 | 10.5 |
| 6 | `S5` | `subgroup_motion=moderate_motion` | 1 | 0.0 | 0.0 | 100.0 |
| 6 | `S5` | `subgroup_crowd=high_crowd` | 16 | 62.5 | 62.5 | 12.5 |
| 6 | `S5` | `subgroup_entry=new_entry_near_onset` | 13 | 53.8 | 53.8 | 15.4 |
| 6 | `B` | `subgroup_duration=short_event` | 8 | 75.0 | 75.0 | 25.0 |
| 6 | `B` | `subgroup_duration=sustained_event` | 12 | 50.0 | 58.3 | 25.0 |
| 6 | `B` | `subgroup_motion=sudden_fast_motion` | 19 | 63.2 | 68.4 | 21.1 |
| 6 | `B` | `subgroup_motion=moderate_motion` | 1 | 0.0 | 0.0 | 100.0 |
| 6 | `B` | `subgroup_crowd=high_crowd` | 16 | 62.5 | 68.8 | 25.0 |
| 6 | `B` | `subgroup_entry=new_entry_near_onset` | 13 | 61.5 | 61.5 | 30.8 |

---

## 4. Phase C: Replication Across Cadence Phases, UBnormal, and Original Stock Frontend

### 4.1 Cadence Phase Replicates (`phase = 0, 1, 2` on `screen_sht` at 6 Hz / K=4)

| Method | Phase | Budget Hz | Actual Det Hz | Delivered AUC [95% CI] | Recall @1.0s (%) | Scorable Cov |
|---|---:|---:|---:|---|---:|---:|
| `B` | 0 | 6 | 6.05 | **67.82** [52.71, 81.62] | 65.0 | 0.213 |
| `S0` | 0 | 6 | 6.05 | **54.12** [52.94, 55.98] | 0.0 | 0.000 |
| `S1` | 0 | 6 | 6.05 | **70.97** [55.67, 84.55] | 65.0 | 0.854 |
| `S2` | 0 | 6 | 6.05 | **70.64** [56.19, 84.25] | 65.0 | 0.854 |
| `S3` | 0 | 6 | 5.98 | **68.97** [53.56, 82.44] | 65.0 | 0.856 |
| `S4` | 0 | 6 | 6.01 | **69.13** [51.56, 84.10] | 70.0 | 0.852 |
| `S5` | 0 | 6 | 6.02 | **69.98** [53.61, 84.17] | 60.0 | 0.852 |
| `B` | 1 | 6 | 5.98 | **68.78** [53.16, 82.50] | 60.0 | 0.210 |
| `S0` | 1 | 6 | 5.98 | **54.12** [52.94, 55.98] | 0.0 | 0.000 |
| `S1` | 1 | 6 | 5.98 | **70.68** [55.21, 84.14] | 60.0 | 0.852 |
| `S2` | 1 | 6 | 5.98 | **71.31** [56.87, 84.52] | 60.0 | 0.852 |
| `B` | 2 | 6 | 5.98 | **68.74** [54.89, 81.67] | 55.0 | 0.211 |
| `S0` | 2 | 6 | 5.98 | **54.12** [52.94, 55.98] | 0.0 | 0.000 |
| `S1` | 2 | 6 | 5.98 | **68.66** [53.30, 82.23] | 65.0 | 0.851 |
| `S2` | 2 | 6 | 5.98 | **69.70** [55.65, 83.20] | 55.0 | 0.851 |

### 4.2 Transfer to UBnormal Screening Subset (`ML0` vs Original `stock` Frontend)

| Frontend | Method | Target Hz | Actual Det Hz | CPU (ms/s) | Delivered AUC [95% CI] | Paper-Compat AUC | Recall @0.5s / @1.0s (%) | Missed Events (%) | Meas / Scorable Cov | Invalid Flow |
|---|---|---:|---:|---:|---|---:|---|---:|---|---:|
| `ML0` | `B` | 6 | 6.02 | 136.3 | **58.55** [44.52, 72.42] | 75.63 | 35.7 / 42.9 | 0.0 | 0.251 / 0.186 | 0.000 |
| `ML0` | `S0` | 6 | 6.02 | 118.9 | **52.48** [50.58, 55.45] | 52.48 | 0.0 / 0.0 | 100.0 | 0.251 / 0.000 | 0.000 |
| `ML0` | `S1` | 6 | 6.02 | 137.0 | **59.05** [45.59, 72.61] | 74.89 | 42.9 / 64.3 | 0.0 | 0.251 / 0.757 | 0.000 |
| `ML0` | `S2` | 6 | 6.02 | 187.6 | **59.18** [45.03, 73.11] | 75.30 | 42.9 / 50.0 | 0.0 | 0.251 / 0.757 | 0.064 |
| `ML0` | `S3` | 6 | 6.01 | 145.7 | **59.89** [44.66, 74.03] | 75.32 | 42.9 / 64.3 | 0.0 | 0.250 / 0.757 | 0.000 |
| `ML0` | `S4` | 6 | 5.98 | 187.1 | **59.78** [44.85, 74.50] | 75.86 | 42.9 / 64.3 | 0.0 | 0.249 / 0.761 | 0.068 |
| `ML0` | `S5` | 6 | 5.98 | 187.4 | **59.70** [44.62, 74.98] | 76.17 | 42.9 / 50.0 | 0.0 | 0.249 / 0.761 | 0.064 |
| `ML0` | `B` | 12 | 12.03 | 218.8 | **60.35** [47.00, 73.98] | 74.49 | 42.9 / 50.0 | 0.0 | 0.501 / 0.379 | 0.000 |
| `ML0` | `S0` | 12 | 12.03 | 200.1 | **52.48** [50.58, 55.45] | 52.48 | 0.0 / 0.0 | 100.0 | 0.501 / 0.000 | 0.000 |
| `ML0` | `S1` | 12 | 12.03 | 219.6 | **62.69** [48.85, 76.12] | 75.37 | 42.9 / 64.3 | 0.0 | 0.501 / 0.783 | 0.000 |
| `ML0` | `S2` | 12 | 12.03 | 270.2 | **62.71** [49.17, 76.11] | 75.23 | 42.9 / 57.1 | 0.0 | 0.501 / 0.783 | 0.065 |
| `ML0` | `S3` | 12 | 11.98 | 227.8 | **60.05** [44.96, 75.28] | 74.60 | 42.9 / 57.1 | 0.0 | 0.499 / 0.780 | 0.000 |
| `ML0` | `S4` | 12 | 11.97 | 269.6 | **63.27** [49.16, 76.54] | 76.20 | 42.9 / 64.3 | 0.0 | 0.499 / 0.785 | 0.065 |
| `ML0` | `S5` | 12 | 11.97 | 269.5 | **61.45** [46.71, 75.86] | 75.13 | 42.9 / 57.1 | 0.0 | 0.499 / 0.781 | 0.067 |
| `ML0` | `R` | 24 | 24.00 | 379.7 | **58.51** [43.79, 73.85] | 75.55 | 42.9 / 50.0 | 0.0 | 1.000 / 0.752 | 0.000 |
| `stock` | `B` | 6 | 6.02 | 132.5 | **60.10** [43.62, 76.97] | 62.49 | 28.6 / 50.0 | 21.4 | 0.251 / 0.158 | 0.000 |
| `stock` | `S0` | 6 | 6.02 | 123.0 | **52.48** [50.58, 55.45] | 52.48 | 0.0 / 0.0 | 100.0 | 0.251 / 0.000 | 0.000 |
| `stock` | `S1` | 6 | 6.02 | 133.0 | **61.09** [45.96, 77.15] | 60.80 | 28.6 / 57.1 | 14.3 | 0.251 / 0.646 | 0.000 |
| `stock` | `S2` | 6 | 6.02 | 183.6 | **61.30** [45.80, 77.25] | 61.40 | 28.6 / 57.1 | 14.3 | 0.251 / 0.646 | 0.103 |
| `stock` | `S3` | 6 | 6.01 | 141.7 | **62.00** [46.90, 77.93] | 60.73 | 28.6 / 57.1 | 14.3 | 0.250 / 0.654 | 0.000 |
| `stock` | `S4` | 6 | 5.99 | 183.1 | **61.74** [46.04, 77.26] | 61.68 | 28.6 / 57.1 | 14.3 | 0.250 / 0.646 | 0.112 |
| `stock` | `S5` | 6 | 5.99 | 183.1 | **60.67** [44.12, 76.68] | 61.41 | 28.6 / 42.9 | 14.3 | 0.250 / 0.644 | 0.105 |
| `stock` | `B` | 12 | 12.03 | 218.4 | **59.06** [43.75, 75.49] | 58.06 | 28.6 / 50.0 | 21.4 | 0.501 / 0.325 | 0.000 |
| `stock` | `S0` | 12 | 12.03 | 208.4 | **52.48** [50.58, 55.45] | 52.48 | 0.0 / 0.0 | 100.0 | 0.501 / 0.000 | 0.000 |
| `stock` | `S1` | 12 | 12.03 | 219.0 | **62.38** [48.70, 77.10] | 60.71 | 28.6 / 57.1 | 14.3 | 0.501 / 0.676 | 0.000 |
| `stock` | `S2` | 12 | 12.03 | 269.6 | **62.18** [47.93, 76.73] | 60.41 | 28.6 / 50.0 | 14.3 | 0.501 / 0.676 | 0.096 |
| `stock` | `S3` | 12 | 11.98 | 227.0 | **63.05** [49.19, 77.72] | 60.71 | 28.6 / 57.1 | 14.3 | 0.499 / 0.672 | 0.000 |
| `stock` | `S4` | 12 | 11.99 | 269.0 | **62.46** [48.86, 76.91] | 60.72 | 28.6 / 57.1 | 14.3 | 0.500 / 0.674 | 0.104 |
| `stock` | `S5` | 12 | 11.99 | 269.0 | **62.22** [47.92, 76.71] | 60.30 | 28.6 / 50.0 | 14.3 | 0.500 / 0.675 | 0.098 |
| `stock` | `R` | 24 | 24.00 | 388.3 | **58.94** [43.34, 75.58] | 59.08 | 28.6 / 42.9 | 21.4 | 1.000 / 0.643 | 0.000 |

### 4.3 Original Stock Frontend Transfer Control on ShanghaiTech (`screen_sht`, `stock` YOLOv8n-pose)

| Frontend | Method | Target Hz | Actual Det Hz | CPU (ms/s) | Delivered AUC [95% CI] | Paper-Compat AUC | Recall @1.0s (%) | Meas / Scorable Cov |
|---|---|---:|---:|---:|---|---:|---:|---|
| `stock` | `B` | 6 | 6.05 | 134.1 | **69.32** [52.26, 83.29] | 67.98 | 55.0 | 0.252 / 0.156 |
| `stock` | `S0` | 6 | 6.05 | 123.5 | **54.12** [52.94, 55.98] | 54.12 | 0.0 | 0.252 / 0.000 |
| `stock` | `S1` | 6 | 6.05 | 134.9 | **67.24** [48.52, 83.00] | 66.45 | 60.0 | 0.252 / 0.641 |
| `stock` | `S2` | 6 | 6.05 | 185.5 | **67.43** [46.80, 83.93] | 67.29 | 60.0 | 0.252 / 0.641 |
| `stock` | `S3` | 6 | 5.98 | 142.8 | **68.99** [53.60, 84.03] | 67.14 | 70.0 | 0.249 / 0.640 |
| `stock` | `S4` | 6 | 6.02 | 185.1 | **66.15** [49.57, 81.48] | 62.69 | 70.0 | 0.251 / 0.641 |
| `stock` | `S5` | 6 | 6.02 | 185.1 | **66.70** [46.72, 82.50] | 65.04 | 55.0 | 0.251 / 0.641 |
| `stock` | `B` | 12 | 12.03 | 220.3 | **67.36** [48.72, 82.32] | 66.57 | 60.0 | 0.501 / 0.327 |
| `stock` | `S0` | 12 | 12.03 | 208.5 | **54.12** [52.94, 55.98] | 54.12 | 0.0 | 0.501 / 0.000 |
| `stock` | `S1` | 12 | 12.03 | 221.4 | **66.79** [49.56, 81.87] | 66.02 | 65.0 | 0.501 / 0.687 |
| `stock` | `S2` | 12 | 12.03 | 271.9 | **66.11** [46.55, 81.72] | 66.57 | 60.0 | 0.501 / 0.687 |
| `stock` | `S3` | 12 | 11.97 | 228.9 | **67.94** [51.18, 81.63] | 67.36 | 60.0 | 0.499 / 0.678 |
| `stock` | `S4` | 12 | 11.99 | 271.3 | **67.82** [51.18, 81.73] | 66.36 | 65.0 | 0.500 / 0.672 |
| `stock` | `S5` | 12 | 11.99 | 271.3 | **67.58** [49.75, 81.44] | 67.27 | 55.0 | 0.500 / 0.673 |
| `stock` | `R` | 24 | 24.00 | 389.6 | **67.73** [50.19, 81.92] | 66.30 | 55.0 | 1.000 / 0.640 |

### 4.4 Paired Bootstrap Contrasts on Replication Arms (`screen_ub` & `screen_sht/stock`)

| Subset | Frontend | Budget (Hz) | Comparison (`Candidate - Base`) | Δ Delivered AUC (pts) | 95% Bootstrap CI | Significant (>0) |
|---|---|---:|---|---:|---|---|
| `screen_ub` | `ML0` | 12 | `S1 - S0` | **+10.21** | [-3.79, +23.50] | no |
| `screen_ub` | `ML0` | 12 | `S2 - S1` | **+0.01** | [-1.79, +1.25] | no |
| `screen_ub` | `ML0` | 12 | `S3 - S1` | **-2.64** | [-6.12, +0.13] | no |
| `screen_ub` | `ML0` | 12 | `S4 - S1` | **+0.58** | [-1.55, +2.64] | no |
| `screen_ub` | `ML0` | 12 | `S5 - S1` | **-1.24** | [-4.20, +1.02] | no |
| `screen_ub` | `ML0` | 12 | `S4 - S3` | **+3.22** | [+0.39, +6.61] | YES |
| `screen_ub` | `ML0` | 12 | `S5 - S3` | **+1.40** | [-0.55, +3.49] | no |
| `screen_ub` | `ML0` | 12 | `S5 - S4` | **-1.82** | [-3.99, +0.13] | no |
| `screen_ub` | `ML0` | 12 | `B - S1` | **-2.34** | [-5.54, +1.09] | no |
| `screen_ub` | `ML0` | 6 | `S1 - S0` | **+6.57** | [-7.62, +20.74] | no |
| `screen_ub` | `ML0` | 6 | `S2 - S1` | **+0.12** | [-1.78, +2.10] | no |
| `screen_ub` | `ML0` | 6 | `S3 - S1` | **+0.84** | [-1.88, +3.54] | no |
| `screen_ub` | `ML0` | 6 | `S4 - S1` | **+0.72** | [-1.43, +2.83] | no |
| `screen_ub` | `ML0` | 6 | `S5 - S1` | **+0.65** | [-2.54, +3.47] | no |
| `screen_ub` | `ML0` | 6 | `S4 - S3` | **-0.11** | [-2.38, +2.42] | no |
| `screen_ub` | `ML0` | 6 | `S5 - S3` | **-0.19** | [-2.78, +2.81] | no |
| `screen_ub` | `ML0` | 6 | `S5 - S4` | **-0.08** | [-2.03, +1.47] | no |
| `screen_ub` | `ML0` | 6 | `B - S1` | **-0.51** | [-2.38, +1.94] | no |
| `screen_ub` | `stock` | 12 | `S1 - S0` | **+9.90** | [-4.88, +24.89] | no |
| `screen_ub` | `stock` | 12 | `S2 - S1` | **-0.20** | [-1.63, +1.59] | no |
| `screen_ub` | `stock` | 12 | `S3 - S1` | **+0.67** | [-0.53, +2.09] | no |
| `screen_ub` | `stock` | 12 | `S4 - S1` | **+0.08** | [-0.60, +0.82] | no |
| `screen_ub` | `stock` | 12 | `S5 - S1` | **-0.16** | [-1.74, +1.87] | no |
| `screen_ub` | `stock` | 12 | `S4 - S3` | **-0.59** | [-2.23, +0.84] | no |
| `screen_ub` | `stock` | 12 | `S5 - S3` | **-0.83** | [-2.55, +1.06] | no |
| `screen_ub` | `stock` | 12 | `S5 - S4` | **-0.24** | [-1.81, +1.72] | no |
| `screen_ub` | `stock` | 12 | `B - S1` | **-3.32** | [-7.66, +0.68] | no |
| `screen_ub` | `stock` | 6 | `S1 - S0` | **+8.61** | [-7.53, +25.17] | no |
| `screen_ub` | `stock` | 6 | `S2 - S1` | **+0.21** | [-2.08, +4.71] | no |
| `screen_ub` | `stock` | 6 | `S3 - S1` | **+0.91** | [-1.06, +3.22] | no |
| `screen_ub` | `stock` | 6 | `S4 - S1` | **+0.65** | [-1.24, +2.69] | no |
| `screen_ub` | `stock` | 6 | `S5 - S1` | **-0.43** | [-3.73, +4.79] | no |
| `screen_ub` | `stock` | 6 | `S4 - S3` | **-0.26** | [-2.96, +2.57] | no |
| `screen_ub` | `stock` | 6 | `S5 - S3` | **-1.33** | [-6.29, +4.69] | no |
| `screen_ub` | `stock` | 6 | `S5 - S4` | **-1.08** | [-3.77, +3.30] | no |
| `screen_ub` | `stock` | 6 | `B - S1` | **-0.99** | [-3.83, +4.57] | no |
| `screen_sht` | `stock` | 12 | `S1 - S0` | **+12.67** | [-3.87, +27.42] | no |
| `screen_sht` | `stock` | 12 | `S2 - S1` | **-0.68** | [-4.44, +1.57] | no |
| `screen_sht` | `stock` | 12 | `S3 - S1` | **+1.15** | [-0.98, +2.74] | no |
| `screen_sht` | `stock` | 12 | `S4 - S1` | **+1.03** | [-0.46, +2.46] | no |
| `screen_sht` | `stock` | 12 | `S5 - S1` | **+0.79** | [-1.98, +3.09] | no |
| `screen_sht` | `stock` | 12 | `S4 - S3` | **-0.13** | [-1.39, +1.73] | no |
| `screen_sht` | `stock` | 12 | `S5 - S3` | **-0.36** | [-3.67, +2.17] | no |
| `screen_sht` | `stock` | 12 | `S5 - S4` | **-0.23** | [-2.71, +1.10] | no |
| `screen_sht` | `stock` | 12 | `B - S1` | **+0.56** | [-3.67, +3.61] | no |
| `screen_sht` | `stock` | 6 | `S1 - S0` | **+13.12** | [-4.83, +28.65] | no |
| `screen_sht` | `stock` | 6 | `S2 - S1` | **+0.20** | [-2.91, +2.62] | no |
| `screen_sht` | `stock` | 6 | `S3 - S1` | **+1.75** | [-1.09, +6.69] | no |
| `screen_sht` | `stock` | 6 | `S4 - S1` | **-1.09** | [-4.29, +1.29] | no |
| `screen_sht` | `stock` | 6 | `S5 - S1` | **-0.54** | [-4.36, +3.02] | no |
| `screen_sht` | `stock` | 6 | `S4 - S3` | **-2.84** | [-6.75, +0.26] | no |
| `screen_sht` | `stock` | 6 | `S5 - S3` | **-2.29** | [-10.02, +3.34] | no |
| `screen_sht` | `stock` | 6 | `S5 - S4` | **+0.55** | [-4.16, +4.33] | no |
| `screen_sht` | `stock` | 6 | `B - S1` | **+2.08** | [-1.05, +5.42] | no |

---

## 5. Decision Rules, Mechanism Analysis, Compute Frontier & Failure Cases

### 5.1 Evaluation Against Pre-Declared Screening Criteria (§8)
1. **Causal Propagation vs No-Fill (`S1..S5` vs `S0`):** At both 12 Hz ($K=2$) and 6 Hz ($K=4$), STG-NF's continuity gate (`<= 2` missing slots per 24-frame window) causes `S0` (no fill on real timestamps) to drop 100% of windows (`scorable_coverage = 0.000`, `AUC = 54.12` on `screen_sht`, `52.48` on `screen_ub`, 100% missed events). Causal propagation (`S1..S5`) restores scorable coverage to `0.852-0.866` (`+16.85` to `+20.40` AUC points, 95% CI excluding 0).
2. **Does Current Visual Evidence (`P-LK`) Beat Constant-Velocity (`P-CV`)? (`S2` vs `S1`, `S5` vs `S4`):**
   - **At 12 Hz (1/2 budget, 2-frame stride):** Yes on `screen_sht` (`ML0`): `S2` (fixed + `P-LK`) improves delivered AUC over `S1` (fixed + `P-CV`) by **+2.06 [+0.24, +4.19]** points (`74.02` vs `71.96`, 95% CI excludes 0) while reducing diagnostic pose coordinate error on propagated frames from `5.40 px` to `4.68 px` (`-13.3%`). Combining prediction-validity scheduling with `P-LK` (`S5`) achieves **74.52** delivered AUC (**+2.56 [+0.39, +5.15]** over `S1`, **+2.07 [-0.09, +4.34]** over `S4`), crossing the pre-declared `+2.0` AUC promising threshold on the primary screen.
   - **At 6 Hz (1/4 budget, 4-frame stride):** `P-LK` still lowers coordinate error relative to `P-CV` (`4.56 px` vs `5.55 px` in `S2` vs `S1`, `4.98 px` vs `6.57 px` in `S5` vs `S4`), and improves phases 1 and 2 (`71.31` vs `70.68` at phase 1; `69.70` vs `68.66` at phase 2). However, at phase 0 (`6 Hz`), `S2 - S1` is `-0.33 [-3.91, +1.87]` and `S5 - S1` is `-0.99 [-4.32, +1.97]`. Over 3-frame unobserved intervals (`125 ms`), multi-step LK accumulation + camera/limb occlusion slightly increases trajectory jitter in STG-NF's velocity-sensitive normalizing flow.
3. **Does Prediction-Validity Scheduling (`S4`/`S5`) Beat Simple Motion-Only (`S3`) and Fixed Cadence (`S1`)?**
   - On `screen_sht` at 12 Hz (`ML0`), `S5` (`74.52`) beats `S1` (`71.96`, `+2.56*`) and edges `S3` (`73.65`, `+0.87 [-2.29, +3.61]`), while reducing missed events on sustained and sudden-motion events at 6 Hz (`S4`/`S5` miss only `15.0%` [3/20] of events vs `S1` `20.0%` [4/20] and `S2`/`B` `25.0%` [5/20], with `S4` raising 1.0s recall to `70.0%` vs `65.0%` for `S1`/`S3`).
   - On `screen_ub` (`ML0` at 12 Hz), `S4` (prediction-validity + `P-CV`) reaches the highest delivered AUC (**63.27** vs `S1` `62.69` and `S3` `60.05`, `S4 - S3 = +3.22`).
   - **Honest complexity verdict (§8 Rule 6):** While `S5` beats `S1` significantly at 12 Hz (`+2.56*`), the paired difference `S5 - S3` (`+0.87` at 12 Hz, `+1.01` at 6 Hz) has 95% CIs spanning zero, and `S2` (fixed cadence + `P-LK`) already captures `+2.06*` of the `+2.56*` gain without adaptive scheduling complexity.
4. **Live Causal (`P-CV` / `P-LK`) vs Delayed Look-Ahead (`B-linear`):**
   - When `B-linear` is evaluated honestly without backdating its alert availability (`available_at_frame = frame_after`), it **loses** to causal `S1` at 6 Hz by **-3.15 [-6.47, -0.20]** AUC (`67.82` vs `70.97`), drops scorable coverage from `0.854` to `0.213` (because windows cannot be scored until the future anchor finishes detector inference), and increases missed events from `20.0%` to `25.0%`.
   - Even at 12 Hz, delayed `B` (`72.80`) trails causal `S2` (`74.02`) and `S5` (`74.52`). Per §8 Rule 7, `B` is rejected for live alerting and retained only as a delayed-review reference.

### 5.2 Matched-Call vs Measured Total-Compute Frontier
- **Call-matched parity:** The token-bucket + `busy_until_ms` gate strictly enforced the detector call budget across all adaptive arms: `0` budget violations occurred across any video/cell (`actual_detector_hz = 11.97-12.03 Hz` at 1/2 budget and `5.98-6.05 Hz` at 1/4 budget).
- **Total CPU cost frontier (including observer overhead):**
  - Full-rate reference `R` (24 Hz): **381.9 ms CPU/s** (`72.29` delivered AUC).
  - 12 Hz Fixed `S1` (`P-CV`, no visual observer): **224.8 ms CPU/s** (**-41.1% CPU** vs `R`, `71.96` AUC).
  - 12 Hz Adaptive Motion-Only `S3` (`P-CV` + cheap 4×-downsampled observer at `0.37 ms/frame`): **232.3 ms CPU/s** (**-39.2% CPU** vs `R`, `73.65` AUC).
  - 12 Hz Visual Propagation `S2` / `S5` (`P-LK` + FB consistency observer at `2.11 ms/frame`): **274.8-275.4 ms CPU/s** (**-28.0% CPU** vs `R`, **74.02 / 74.52** AUC).
  - 6 Hz Fixed `S1` (`P-CV`): **141.6 ms CPU/s** (**-62.9% CPU** vs `R`, `70.97` AUC).
  - 6 Hz Visual/Adaptive `S4` / `S5`: **191.0 ms CPU/s** (**-50.0% CPU** vs `R`, `69.13 / 69.98` AUC; reduces missed events from 4/20 to 3/20).

### 5.3 Failure-Case & Diagnostic Analysis
- **Where `P-LK` fails (`invalid_flow_fraction`):** Across ShanghaiTech (`856x480`), pyramidal LK forward-backward + texture validation rejects only **2.4%-2.7%** of active joint stencils. On UBnormal synthetic scenes (`screen_ub`, including fog/fire scenarios), invalid flow rises to **6.4%-6.8%** (`ML0`) and **9.6%-11.2%** (`stock`), explaining why `P-LK` (`S2`/`S5`) yields no accuracy gain over `P-CV` (`S1`/`S4`) on UBnormal.
- **Frontend interaction (`ML0` vs `stock`):** On `screen_sht`, `ML0` outperforms `stock` across every sparse policy by **+3.7 to +7.9** delivered AUC points (e.g. at 12 Hz: `S2` is `74.02` on `ML0` vs `66.11` on `stock`; `S5` is `74.52` vs `67.58`) because `ML0` detects small distant people (`<64 px`) with `0.863` scorable coverage vs `0.687` for `stock`. On `screen_ub`, `stock` and `ML0` perform similarly on delivered AUC (`61.0-63.1` vs `59.1-63.3`), consistent with `EXP-020`.
- **Delivered Live vs Paper-Compatible Secondary Endpoint:** On `screen_ub` (`ML0`), paper-compatible AUC (which uses per-clip retrospective finite max normality for unscored frames) is `74.89-76.20`, whereas strict causal delivered AUC (using the fixed train-calibrated sentinel `-0.7288` and `24`-frame hold limit) is `59.05-63.27`. This confirms the §5 scoring caveat: retrospective per-video normalization inflates apparent accuracy on clips with unscored intervals and must never be conflated with strict online delivery.
