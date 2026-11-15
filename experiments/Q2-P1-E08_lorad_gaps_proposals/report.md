# Q2-P1-E08: LoRaD gap proposals (P_A, P_B, P_C, P_D, P_ABC)

All numbers are produced by `src/edgelab/analysis/q2_p1_e08.py` (stages `train`, `run`, `stats`, `pi-prep`/`pi-report`, `final`) on the pre-registered `plan.md` / `config.yaml`. AUCs are in percent; deltas carry 2,000-resample paired video-bootstrap 95% CIs (`*` = CI excludes 0). The primary cell is ML0_edge + locked two-stage tracker, S4, test_confirmation, mean over the two datasets, paired against B1_LoRaD.

## 1. Verdicts

| Hypothesis | Verdict | Pre-registered conditions |
|---|---|---|
| H_A | **REFUTED** | pooled_ci_lo_gt_0=no; short_ci_lo_gt_0=no; sustained_point_gt_0=no |
| H_B | **REFUTED** | pooled_ci_lo_gt_0=no; fa_h_ci_hi_lt_0=no; params_equal_B1=yes; pi5_scorer_latency_within_5pct=yes |
| H_C | **NOT SUPPORTED** | fast_motion_ci_lo_gt_0=no; high_crowd_ci_lo_gt_0=no; recall_fast_point_gt_0=no; recall_crowd_point_gt_0=no; vs_trajvad_p_ci_lo_ge_-2=yes; params_ratio_le_1_300=yes; pi5_scorer_speedup_ge_4=no |
| H_D | **PARTIAL** | fa_h_ci_hi_lt_0=no; pooled_ci_lo_ge_-0.5=yes |
| H_Joint | **PARTIAL** | highest_delivered_auc_among_cache_arms=yes; delivered_fps_ge_24=no; queue_wait_p99_lt_500ms=yes; throttled_0x0=yes |

## 2. Primary contrasts (ML0_edge, S4, confirmation, dataset mean)

| Arm | dAUC S4 vs B1 | dAUC S0 vs B1 | dFA/h@R50 vs B1 | dRecall@FA(B1) vs B1 | dAUC S4 vs TrajVAD-P |
|---|---|---|---|---|---|
| P_A_MS_CTFlow | -1.51 [-0.02, -0.01]* | -1.80 [-0.03, -0.01]* | +15.11 [-0.02, +30.04] | -3.71 [-6.22, -1.41]* | n/a |
| P_B_RevMargin | -3.48 [-0.06, -0.01]* | -3.79 [-0.06, -0.02]* | +44.98 [+19.29, +74.19]* | -9.77 [-13.42, -6.46]* | n/a |
| P_C_IAC_Flow | -0.71 [-0.02, +0.01] | -0.52 [-0.02, +0.01] | +26.09 [+7.53, +46.48]* | -5.68 [-8.24, -3.42]* | +4.62 [+0.02, +0.07]* |
| P_D_ConfSmooth | +0.01 [-0.00, +0.00] | -0.01 [-0.00, +0.00] | +3.00 [+0.00, +8.04] | -1.00 [-1.71, -0.40]* | n/a |
| P_ABC_MS_IAC_RevFlow | -3.69 [-0.06, -0.01]* | -4.05 [-0.06, -0.02]* | +25.73 [+0.95, +47.95]* | -9.28 [-13.32, -5.54]* | +1.64 [-0.01, +0.04] |

### Pooled AUC (S4 / S0) by pose source, ShanghaiTech

| Pose | Subset | B0_STGNF | B1_LoRaD | P_A_MS_CTFlow | P_B_RevMargin | P_C_IAC_Flow | P_D_ConfSmooth | P_ABC_MS_IAC_RevFlow | TrajVAD-P |
|---|---|---|---|---|---|---|---|---|---|
| released | dev | 79.48 / 85.05 | 79.34 / 85.01 | 78.25 / 83.71 | 75.46 / 80.20 | 78.45 / 84.34 | 78.92 / 84.31 | 75.01 / 79.59 | n/a / n/a |
| released | confirmation | 80.36 / 85.16 | 80.61 / 85.39 | 78.77 / 83.15 | 75.26 / 79.04 | 79.50 / 84.13 | 80.70 / 85.46 | 75.18 / 78.25 | n/a / n/a |
| released | all | 80.20 / 85.22 | 80.33 / 85.37 | 78.71 / 83.42 | 75.48 / 79.55 | 79.27 / 84.29 | 80.26 / 85.21 | 75.35 / 78.87 | n/a / n/a |
| ML0_edge | dev | 77.43 / 82.21 | 77.29 / 81.93 | 76.01 / 80.65 | 73.41 / 77.85 | 74.69 / 79.72 | 77.20 / 81.84 | 70.91 / 75.71 | 73.27 / 76.13 |
| ML0_edge | confirmation | 76.39 / 81.09 | 75.99 / 80.57 | 73.93 / 77.93 | 72.42 / 75.89 | 75.15 / 79.96 | 76.01 / 80.56 | 72.09 / 75.37 | 70.90 / 73.38 |
| ML0_edge | all | 76.86 / 81.57 | 76.54 / 81.12 | 74.70 / 78.91 | 72.82 / 76.59 | 75.08 / 79.95 | 76.53 / 81.09 | 71.86 / 75.62 | 71.84 / 74.43 |
| YOLO26_e2e | dev | 65.62 / 67.67 | 65.23 / 67.64 | 64.71 / 66.26 | 62.77 / 63.17 | 64.44 / 67.04 | 64.72 / 67.05 | 60.87 / 61.15 | 62.32 / 64.87 |
| YOLO26_e2e | confirmation | 62.25 / 63.98 | 61.78 / 63.93 | 61.65 / 63.01 | 60.90 / 61.77 | 60.20 / 62.78 | 60.87 / 62.87 | 59.64 / 60.06 | 58.08 / 60.00 |
| YOLO26_e2e | all | 63.57 / 65.47 | 63.13 / 65.43 | 62.82 / 64.32 | 61.64 / 62.45 | 61.72 / 64.38 | 62.33 / 64.51 | 60.36 / 60.79 | 59.60 / 61.76 |

### Pooled AUC (S4 / S0) by pose source, UBnormal

| Pose | Subset | B0_STGNF | B1_LoRaD | P_A_MS_CTFlow | P_B_RevMargin | P_C_IAC_Flow | P_D_ConfSmooth | P_ABC_MS_IAC_RevFlow | TrajVAD-P |
|---|---|---|---|---|---|---|---|---|---|
| released | dev | 64.74 / 65.04 | 65.22 / 65.48 | 66.68 / 66.96 | 63.84 / 64.03 | 66.46 / 66.34 | 65.11 / 65.39 | 66.65 / 66.58 | n/a / n/a |
| released | confirmation | 72.51 / 72.67 | 72.74 / 72.93 | 72.81 / 73.11 | 70.82 / 70.75 | 72.78 / 72.81 | 72.76 / 72.95 | 70.59 / 70.88 | n/a / n/a |
| released | all | 70.04 / 70.20 | 70.35 / 70.53 | 70.88 / 71.13 | 68.48 / 68.49 | 70.69 / 70.64 | 70.34 / 70.52 | 69.21 / 69.39 | n/a / n/a |
| ML0_edge | dev | 57.80 / 57.78 | 58.43 / 58.27 | 59.62 / 59.09 | 57.48 / 57.21 | 59.59 / 58.86 | 58.44 / 58.27 | 60.28 / 59.06 | 53.31 / 53.84 |
| ML0_edge | confirmation | 60.53 / 57.16 | 61.20 / 57.92 | 60.25 / 56.95 | 57.82 / 55.02 | 60.62 / 57.50 | 61.20 / 57.91 | 57.73 / 55.01 | 55.62 / 55.56 |
| ML0_edge | all | 59.63 / 57.38 | 60.26 / 58.02 | 60.03 / 57.66 | 57.70 / 55.76 | 60.29 / 57.96 | 60.27 / 58.02 | 58.59 / 56.39 | 54.89 / 55.04 |
| YOLO26_e2e | dev | 60.62 / 59.09 | 61.49 / 59.48 | 62.74 / 60.06 | 58.13 / 57.41 | 59.10 / 57.86 | 60.75 / 59.01 | 56.94 / 56.95 | 56.14 / 56.14 |
| YOLO26_e2e | confirmation | 60.39 / 57.11 | 61.11 / 57.69 | 61.07 / 57.94 | 59.29 / 56.00 | 62.91 / 58.07 | 60.73 / 57.26 | 61.24 / 56.97 | 56.83 / 56.82 |
| YOLO26_e2e | all | 60.42 / 57.73 | 61.14 / 58.22 | 61.53 / 58.60 | 58.87 / 56.38 | 61.64 / 57.90 | 60.66 / 57.79 | 59.84 / 56.84 | 56.51 / 56.52 |

## 3. Pose-source robustness (S4, confirmation, paired vs B1, dataset mean)

| Pose | Arm | dAUC vs B1 [95% CI] |
|---|---|---|
| released | P_A_MS_CTFlow | -0.88 [-0.01, -0.00]* |
| released | P_B_RevMargin | -3.63 [-0.05, -0.02]* |
| released | P_C_IAC_Flow | -0.53 [-0.01, +0.00] |
| released | P_D_ConfSmooth | +0.06 [-0.00, +0.00] |
| released | P_ABC_MS_IAC_RevFlow | -3.78 [-0.05, -0.02]* |
| ML0_edge | P_A_MS_CTFlow | -1.51 [-0.02, -0.01]* |
| ML0_edge | P_B_RevMargin | -3.48 [-0.06, -0.01]* |
| ML0_edge | P_C_IAC_Flow | -0.71 [-0.02, +0.01] |
| ML0_edge | P_D_ConfSmooth | +0.01 [-0.00, +0.00] |
| ML0_edge | P_ABC_MS_IAC_RevFlow | -3.69 [-0.06, -0.01]* |
| YOLO26_e2e | P_A_MS_CTFlow | -0.08 [-0.01, +0.00] |
| YOLO26_e2e | P_B_RevMargin | -1.35 [-0.03, +0.00] |
| YOLO26_e2e | P_C_IAC_Flow | +0.11 [-0.01, +0.02] |
| YOLO26_e2e | P_D_ConfSmooth | -0.65 [-0.01, -0.00]* |
| YOLO26_e2e | P_ABC_MS_IAC_RevFlow | -1.00 [-0.03, +0.01] |

## 4. Subgroups (S4, ML0_edge, confirmation, dataset mean; AUC points, delta vs B1)

| Stratum | Arm | AUC | dAUC vs B1 [CI] | Recall@FA(B1) % | dRecall pts |
|---|---|---|---|---|---|
| short_event | B1_LoRaD | 66.18 |  | 39.4 |  |
| short_event | P_A_MS_CTFlow | 65.29 | -0.89 [-0.02, +0.01] | 44.2 | +4.9 |
| short_event | P_B_RevMargin | 64.20 | -1.98 [-0.08, +0.04] | 34.0 | -5.3 |
| short_event | P_C_IAC_Flow | 65.60 | -0.58 [-0.04, +0.03] | 41.5 | +2.2 |
| short_event | P_D_ConfSmooth | 66.21 | +0.03 [-0.00, +0.00] | 38.4 | -0.9 |
| short_event | P_ABC_MS_IAC_RevFlow | 61.15 | -5.03 [-0.10, -0.01] | 26.1 | -13.3 |
| sustained_event | B1_LoRaD | 69.33 |  | 63.4 |  |
| sustained_event | P_A_MS_CTFlow | 67.79 | -1.53 [-0.02, -0.01] | 57.8 | -5.6 |
| sustained_event | P_B_RevMargin | 65.53 | -3.80 [-0.06, -0.02] | 51.7 | -11.7 |
| sustained_event | P_C_IAC_Flow | 68.54 | -0.79 [-0.02, +0.01] | 55.6 | -7.7 |
| sustained_event | P_D_ConfSmooth | 69.34 | +0.01 [-0.00, +0.00] | 62.4 | -1.0 |
| sustained_event | P_ABC_MS_IAC_RevFlow | 65.38 | -3.95 [-0.06, -0.02] | 52.7 | -10.7 |
| fast_motion | B1_LoRaD | 71.75 |  | 67.0 |  |
| fast_motion | P_A_MS_CTFlow | 70.15 | -1.60 [-0.03, -0.01] | 62.9 | -4.1 |
| fast_motion | P_B_RevMargin | 67.09 | -4.66 [-0.07, -0.02] | 56.8 | -10.2 |
| fast_motion | P_C_IAC_Flow | 69.89 | -1.86 [-0.03, -0.01] | 62.2 | -4.7 |
| fast_motion | P_D_ConfSmooth | 71.76 | +0.01 [-0.00, +0.00] | 66.0 | -1.0 |
| fast_motion | P_ABC_MS_IAC_RevFlow | 66.57 | -5.18 [-0.08, -0.03] | 57.8 | -9.2 |
| moderate_motion | B1_LoRaD | 65.55 |  | 38.3 |  |
| moderate_motion | P_A_MS_CTFlow | 63.85 | -1.70 [-0.03, -0.00] | 33.6 | -4.7 |
| moderate_motion | P_B_RevMargin | 62.64 | -2.90 [-0.07, +0.01] | 29.2 | -9.2 |
| moderate_motion | P_C_IAC_Flow | 65.45 | -0.10 [-0.02, +0.02] | 31.6 | -6.8 |
| moderate_motion | P_D_ConfSmooth | 65.56 | +0.01 [-0.00, +0.00] | 37.3 | -1.0 |
| moderate_motion | P_ABC_MS_IAC_RevFlow | 61.95 | -3.60 [-0.07, -0.00] | 27.9 | -10.4 |
| high_crowd | B1_LoRaD | 71.82 |  | 53.5 |  |
| high_crowd | P_A_MS_CTFlow | 69.95 | -1.88 [-0.03, -0.01] | 46.7 | -6.8 |
| high_crowd | P_B_RevMargin | 69.27 | -2.55 [-0.05, +0.00] | 44.4 | -9.1 |
| high_crowd | P_C_IAC_Flow | 72.45 | +0.63 [-0.01, +0.03] | 45.9 | -7.5 |
| high_crowd | P_D_ConfSmooth | 71.84 | +0.02 [-0.00, +0.00] | 52.2 | -1.3 |
| high_crowd | P_ABC_MS_IAC_RevFlow | 71.20 | -0.62 [-0.04, +0.03] | 45.3 | -8.2 |
| low_crowd | B1_LoRaD | 64.41 |  | 50.9 |  |
| low_crowd | P_A_MS_CTFlow | 63.40 | -1.01 [-0.02, +0.00] | 50.1 | -0.9 |
| low_crowd | P_B_RevMargin | 60.25 | -4.17 [-0.07, -0.01] | 41.1 | -9.9 |
| low_crowd | P_C_IAC_Flow | 63.00 | -1.41 [-0.03, -0.00] | 47.4 | -3.5 |
| low_crowd | P_D_ConfSmooth | 64.41 | -0.01 [-0.00, +0.00] | 50.1 | -0.8 |
| low_crowd | P_ABC_MS_IAC_RevFlow | 57.58 | -6.83 [-0.10, -0.04] | 41.0 | -9.9 |

Event counts (all clips): {"ShanghaiTech": {"short_event": 81, "sustained_event": 113, "fast_motion": 125, "moderate_motion": 69, "high_crowd": 114, "low_crowd": 80}, "UBnormal": {"short_event": 26, "sustained_event": 173, "fast_motion": 75, "moderate_motion": 124, "high_crowd": 85, "low_crowd": 114}}.

### False alerts per hour at Recall 50%

| Dataset | Arm | FA/h@R50 | 95% CI | delta vs B1 [CI] |
|---|---|---|---|---|
| ShanghaiTech | B0_STGNF | 38.5 | [13.6, 69.8] | -10.5 [-19.8, -3.5] |
| ShanghaiTech | B1_LoRaD | 49.1 | [22.4, 83.9] |  |
| ShanghaiTech | P_A_MS_CTFlow | 45.6 | [22.8, 72.2] | -3.5 [-16.2, +9.1] |
| ShanghaiTech | P_B_RevMargin | 66.6 | [40.0, 94.3] | +17.5 [-7.1, +42.7] |
| ShanghaiTech | P_C_IAC_Flow | 35.0 | [13.2, 61.0] | -14.0 [-30.1, -3.3] |
| ShanghaiTech | P_D_ConfSmooth | 52.6 | [25.8, 88.6] | +3.5 [+0.0, +11.8] |
| ShanghaiTech | P_ABC_MS_IAC_RevFlow | 54.3 | [33.4, 77.5] | +5.3 [-26.9, +34.1] |
| ShanghaiTech | TrajVAD-P | 168.2 | [121.1, 219.7] | +119.1 [+75.5, +161.0] |
| UBnormal | B0_STGNF | 109.9 | [60.9, 176.2] | +8.7 [-15.2, +37.3] |
| UBnormal | B1_LoRaD | 101.2 | [60.2, 148.8] |  |
| UBnormal | P_A_MS_CTFlow | 134.9 | [86.5, 190.8] | +33.7 [+8.3, +60.8] |
| UBnormal | P_B_RevMargin | 173.6 | [127.9, 224.0] | +72.4 [+27.8, +121.9] |
| UBnormal | P_C_IAC_Flow | 167.4 | [113.6, 233.5] | +66.2 [+32.8, +106.4] |
| UBnormal | P_D_ConfSmooth | 103.7 | [60.5, 154.7] | +2.5 [+0.0, +8.1] |
| UBnormal | P_ABC_MS_IAC_RevFlow | 147.4 | [109.3, 187.0] | +46.2 [+13.0, +78.4] |
| UBnormal | TrajVAD-P | 139.9 | [93.2, 189.2] | +38.7 [-13.8, +90.1] |
| mean | B0_STGNF | 74.2 | [46.8, 109.8] | -0.9 [-13.7, +13.7] |
| mean | B1_LoRaD | 75.1 | [50.8, 103.3] |  |
| mean | P_A_MS_CTFlow | 90.2 | [63.4, 121.0] | +15.1 [-0.0, +30.0] |
| mean | P_B_RevMargin | 120.1 | [94.2, 150.1] | +45.0 [+19.3, +74.2] |
| mean | P_C_IAC_Flow | 101.2 | [71.1, 137.7] | +26.1 [+7.5, +46.5] |
| mean | P_D_ConfSmooth | 78.1 | [52.7, 108.2] | +3.0 [+0.0, +8.0] |
| mean | P_ABC_MS_IAC_RevFlow | 100.8 | [79.1, 124.9] | +25.7 [+0.9, +47.9] |
| mean | TrajVAD-P | 154.0 | [121.5, 190.5] | +78.9 [+45.6, +113.0] |

## 5. Parameter budgets

| Arm | ShanghaiTech | UBnormal |
|---|---|---|
| B0_STGNF | 616 | 488 |
| B1_LoRaD | 616 | 488 |
| P_A_MS_CTFlow | 1592 | 1208 |
| P_B_RevMargin | 616 | 488 |
| P_C_IAC_Flow | 616 | 488 |
| P_ABC_MS_IAC_RevFlow | 1592 | 1208 |
| P_D_ConfSmooth | 616 | 488 |
| TrajVAD-P | 757338 | n/a |

## 6. Training diagnostics (last epoch, per run)

| Arm | Dataset | Seed | Params | NLL (bits/dim) | Reverse term | Hinge | Gap |
|---|---|---|---|---|---|---|---|
| P_A_MS_CTFlow | ShanghaiTech | 0 | 1592 | 0.8533 | n/a | n/a | n/a |
| P_A_MS_CTFlow | ShanghaiTech | 1 | 1592 | 0.8618 | n/a | n/a | n/a |
| P_A_MS_CTFlow | ShanghaiTech | 2 | 1592 | 0.8625 | n/a | n/a | n/a |
| P_A_MS_CTFlow | UBnormal | 0 | 1208 | 0.9389 | n/a | n/a | n/a |
| P_A_MS_CTFlow | UBnormal | 1 | 1208 | 0.8913 | n/a | n/a | n/a |
| P_A_MS_CTFlow | UBnormal | 2 | 1208 | 0.9131 | n/a | n/a | n/a |
| P_B_RevMargin | ShanghaiTech | 0 | 616 | 0.9005 | 0.00000000 | 0.0082 | 1.5402 |
| P_B_RevMargin | ShanghaiTech | 1 | 616 | 0.8885 | 0.00000000 | 0.0050 | 1.7888 |
| P_B_RevMargin | ShanghaiTech | 2 | 616 | 0.9118 | 0.00000000 | 0.0126 | 1.4459 |
| P_B_RevMargin | UBnormal | 0 | 488 | 0.8773 | 0.00000000 | 0.0236 | 0.8740 |
| P_B_RevMargin | UBnormal | 1 | 488 | 0.8863 | 0.00000000 | 0.0088 | 1.3723 |
| P_B_RevMargin | UBnormal | 2 | 488 | 0.9483 | 0.00000000 | 0.0518 | 0.7835 |
| P_C_IAC_Flow | ShanghaiTech | 0 | 616 | 0.9926 | n/a | n/a | n/a |
| P_C_IAC_Flow | ShanghaiTech | 1 | 616 | 1.0110 | n/a | n/a | n/a |
| P_C_IAC_Flow | ShanghaiTech | 2 | 616 | 0.9943 | n/a | n/a | n/a |
| P_C_IAC_Flow | UBnormal | 0 | 488 | 0.9285 | n/a | n/a | n/a |
| P_C_IAC_Flow | UBnormal | 1 | 488 | 0.9184 | n/a | n/a | n/a |
| P_C_IAC_Flow | UBnormal | 2 | 488 | 0.9721 | n/a | n/a | n/a |
| P_ABC_MS_IAC_RevFlow | ShanghaiTech | 0 | 1592 | 0.9894 | 0.00000000 | 0.0031 | 1.5464 |
| P_ABC_MS_IAC_RevFlow | ShanghaiTech | 1 | 1592 | 0.9924 | 0.00000000 | 0.0117 | 1.4262 |
| P_ABC_MS_IAC_RevFlow | ShanghaiTech | 2 | 1592 | 0.9866 | 0.00000000 | 0.0019 | 2.3713 |
| P_ABC_MS_IAC_RevFlow | UBnormal | 0 | 1208 | 0.9539 | 0.00000000 | 0.0108 | 0.9899 |
| P_ABC_MS_IAC_RevFlow | UBnormal | 1 | 1208 | 0.9912 | 0.00000000 | 0.0379 | 0.7861 |
| P_ABC_MS_IAC_RevFlow | UBnormal | 2 | 1208 | 0.9192 | 0.00000000 | 0.0088 | 1.1370 |

The reverse-consistency term is the round-off of an exactly invertible flow; P_B's effect is therefore attributable to the margin term.

## 7. Raspberry Pi 5 benchmark (4 threads, ML0 NCNN FP16 + BoT-SORT two-stage)

| Scorer | Variant | Frames | Deliv. FPS | Score ms | E2E ms | Wait p99 ms | W | J/proc frame | J/input frame | Tmax C | RSS MB | Throttle | Deliv. AUC (on-time) | Deliv. AUC (all) | Offline AUC same clips |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B0_STGNF | Cache-C1+Guard | 2880 | 25.39 | 5.007 | 37.5 | 418 | 5.16 | 0.193 | 0.185 | 76.5 | 841 | 0x0 | 67.12 | 67.12 | 78.99 |
| B0_STGNF | Full-Rate | 2880 | 16.43 | 7.424 | 59.2 | 50005 | 5.48 | 0.325 | 0.334 | 77.1 | 836 | 0x0 | 49.48 | 79.58 | 78.99 |
| B1_LoRaD | Cache-C1+Guard | 2880 | 25.43 | 5.052 | 37.5 | 416 | 5.12 | 0.192 | 0.183 | 77.1 | 841 | 0x0 | 63.48 | 63.48 | 78.77 |
| B1_LoRaD | Full-Rate | 2880 | 16.41 | 7.376 | 59.2 | 49824 | 5.54 | 0.328 | 0.338 | 79.3 | 843 | 0x0 | 56.32 | 79.25 | 78.77 |
| P_A_MS_CTFlow | Cache-C1+Guard | 2880 | 23.71 | 7.967 | 40.3 | 428 | 5.15 | 0.208 | 0.195 | 77.7 | 850 | 0x0 | 66.94 | 66.94 | 77.88 |
| P_A_MS_CTFlow | Full-Rate | 2880 | 15.08 | 12.494 | 64.4 | 64762 | 5.53 | 0.356 | 0.366 | 80.4 | 844 | 0x0 | 57.57 | 78.39 | 77.88 |
| P_B_RevMargin | Cache-C1+Guard | 2880 | 25.60 | 5.046 | 37.3 | 410 | 5.14 | 0.192 | 0.183 | 77.7 | 848 | 0x0 | 66.74 | 66.74 | 77.79 |
| P_B_RevMargin | Full-Rate | 2880 | 16.51 | 7.443 | 58.7 | 48383 | 5.52 | 0.324 | 0.335 | 80.4 | 847 | 0x0 | 53.46 | 78.23 | 77.79 |
| P_C_IAC_Flow | Cache-C1+Guard | 2880 | 25.45 | 5.560 | 37.6 | 419 | 5.31 | 0.200 | 0.191 | 77.7 | 848 | 0x0 | 67.51 | 67.51 | 78.48 |
| P_C_IAC_Flow | Full-Rate | 2880 | 16.44 | 7.947 | 59.0 | 49292 | 5.53 | 0.326 | 0.337 | 80.4 | 848 | 0x0 | 42.18 | 79.44 | 78.48 |
| P_ABC_MS_IAC_RevFlow | Cache-C1+Guard | 2880 | 23.44 | 8.372 | 40.6 | 420 | 5.09 | 0.207 | 0.195 | 77.1 | 853 | 0x0 | 68.02 | 68.02 | 78.93 |
| P_ABC_MS_IAC_RevFlow | Full-Rate | 2880 | 15.13 | 13.153 | 64.2 | 64135 | 5.49 | 0.352 | 0.363 | 80.4 | 850 | 0x0 | 53.73 | 80.47 | 78.93 |
| TrajVAD-P | Cache-C1+Guard | 2880 | 20.04 | 13.868 | 48.0 | 500 | 5.26 | 0.253 | 0.209 | 77.7 | 866 | 0x0 | 55.71 | 55.71 | 65.92 |
| TrajVAD-P | Full-Rate | 2880 | 12.33 | 27.561 | 79.0 | 106204 | 5.29 | 0.418 | 0.429 | 81.5 | 862 | 0x0 | 39.60 | 65.68 | 65.92 |

## 8. Evidence behind each verdict

- **H_A** REFUTED: `{"pooled": {"pose": "ML0_edge", "subset": "confirmation", "metric": "auc_S4", "a": "P_A", "b": "B1", "dataset": "mean", "a_value": 67.08983546006397, "b_value": 68.59811467524871, "delta": -1.5082792151847428, "ci_lo": -0.022841646573486182, "ci_hi": -0.007151411135447225, "excludes_zero": true}, "short_event": {"pose": "ML0_edge", "subset": "confirmation", "metric": "auc_S4|short_event", "a": "P_A", "b": "B1", "dataset": "mean", "a_value": 65.29032294753743, "b_value": 66.18037951594258, "delta": -0.8900565684051429, "ci_lo": -0.02478459607145729, "ci_hi": 0.006340306434911327, "excludes_zero": false}, "sustained_event": {"pose": "ML0_edge", "subset": "confirmation", "metric": "auc_S4|sustained_event", "a": "P_A", "b": "B1", "dataset": "mean", "a_value": 67.79431840988534, "b_value": 69.32767554171048, "delta": -1.5333571318251416, "ci_lo": -0.023236081030482754, "ci_hi": -0.007071945854776726, "excludes_zero": true}}`
- **H_B** REFUTED: `{"pooled": {"pose": "ML0_edge", "subset": "confirmation", "metric": "auc_S4", "a": "P_B", "b": "B1", "dataset": "mean", "a_value": 65.12110613993431, "b_value": 68.59811467524871, "delta": -3.477008535314397, "ci_lo": -0.056940818856682895, "ci_hi": -0.01255894509040214, "excludes_zero": true}, "fa_h": {"pose": "ML0_edge", "subset": "confirmation", "metric": "fa_h", "a": "P_B", "b": "B1", "dataset": "mean", "a_value": 120.08634934998727, "b_value": 75.10813215901284, "delta": 44.97821719097443, "ci_lo": 19.286492169585195, "ci_hi": 74.1891608476558, "excludes_zero": true}, "pi5_latency": {"B1_score_ms": 7.376, "P_B_score_ms": 7.443, "ratio": 1.009083514099783, "within_5pct": true}}`
- **H_C** NOT SUPPORTED: `{"pooled": {"pose": "ML0_edge", "subset": "confirmation", "metric": "auc_S4", "a": "P_C", "b": "B1", "dataset": "mean", "a_value": 67.88494413311165, "b_value": 68.59811467524871, "delta": -0.713170542137064, "ci_lo": -0.02039751385274727, "ci_hi": 0.007477879066615497, "excludes_zero": false}, "fast_motion": {"pose": "ML0_edge", "subset": "confirmation", "metric": "auc_S4|fast_motion", "a": "P_C", "b": "B1", "dataset": "mean", "a_value": 69.89406557576, "b_value": 71.75154952588917, "delta": -1.857483950129179, "ci_lo": -0.032310104870468065, "ci_hi": -0.005033744520554972, "excludes_zero": true}, "high_crowd": {"pose": "ML0_edge", "subset": "confirmation", "metric": "auc_S4|high_crowd", "a": "P_C", "b": "B1", "dataset": "mean", "a_value": 72.45034038950337, "b_value": 71.82232600906633, "delta": 0.6280143804370368, "ci_lo": -0.013099823055290847, "ci_hi": 0.030543876691603564, "excludes_zero": false}, "recall_fast": {"pose": "ML0_edge", "subset": "confirmation", "metric": "recall|fast_motion", "a": "P_C", "b": "B1", "dataset": "mean", "a_value": 62.212539336722344, "b_value": 66.95715323166303, "delta": -4.744613894940684, "ci_lo": -8.870800802590557, "ci_hi": -1.4401953460725967, "excludes_zero": true}, "recall_crowd": {"pose": "ML0_edge", "subset": "confirmation", "metric": "recall|high_crowd", "a": "P_C", "b": "B1", "dataset": "mean", "a_value": 45.93870741131015, "b_value": 53.486125746399715, "delta": -7.547418335089567, "ci_lo": -11.749441885079582, "ci_hi": -3.608792`
- **H_D** PARTIAL: `{"pooled": {"pose": "ML0_edge", "subset": "confirmation", "metric": "auc_S4", "a": "P_D", "b": "B1", "dataset": "mean", "a_value": 68.60834414305992, "b_value": 68.59811467524871, "delta": 0.010229467811214477, "ci_lo": -0.00025820286822051677, "ci_hi": 0.00047013581568182953, "excludes_zero": false}, "fa_h": {"pose": "ML0_edge", "subset": "confirmation", "metric": "fa_h", "a": "P_D", "b": "B1", "dataset": "mean", "a_value": 78.10897941333957, "b_value": 75.10813215901284, "delta": 3.000847254326729, "ci_lo": 0.0, "ci_hi": 8.038175260646375, "excludes_zero": false}}`
- **H_Joint** PARTIAL: `{"P_ABC": {"delivered_auc_ontime": 68.02495742869307, "delivered_fps": 23.44, "queue_wait_p99_ms": 420.061, "throttled_codes": "0x0", "score_ms": 8.372, "end_to_end_ms": 40.591}, "best_cache_arm": {"scorer": "P_ABC_MS_IAC_RevFlow", "delivered_auc_ontime": 68.02495742869307}, "ranking": [["P_ABC_MS_IAC_RevFlow", 68.02], ["P_C_IAC_Flow", 67.51], ["B0_STGNF", 67.12], ["P_A_MS_CTFlow", 66.94], ["P_B_RevMargin", 66.74], ["B1_LoRaD", 63.48], ["TrajVAD-P", 55.71]]}`

## 9. Reproduction check of the baseline pipeline

| Arm | Dataset | Pose | E06 S4 all | Recomputed S4 all | abs diff |
|---|---|---|---|---|---|
| B1 | ShanghaiTech | released | 80.1862 | 80.1862 | 0.00e+00 |

## 10. Deviations, caveats and limits

- **Amendment A1 (training parallelism).** 24 models were trained by 5 single-thread CPU workers inside this experiment's training stage only. All stages and all experiments (Mac vs. e06/e07 and the Pi) ran strictly sequentially; strict one-model-at-a-time training would have taken ~25 h.
- N0 (per-window mean/std_y) removes absolute position but retains within-window drift; the source plan's statement that translation is 'completely erased' is overstated.
- The reverse-consistency loss is zero up to round-off for an exactly invertible flow (see section 6); the P_B hypothesis therefore reduces to the margin term.
- P_C adds no parameters (not ~0.8K): the context nodes are extra graph nodes sharing the same subnet; the dense max_hop=8 adjacency means hub edges do not sparsify A. Node 19 carries velocity only (aspect is not fed to the flow).
- P_A has about 2.5x the baseline's parameters; H_A is a statement about accuracy at that cost.
- The Pi 5 benchmark ran on ShanghaiTech seed-0 checkpoints and 12 clips (same sample as E07); P_D is eval-only (no Pi arm, it shares B1's weights).
- Event strata come from the released poses (duration, per-track median box speed >= 0.06 h/frame, mean concurrency >= 4) and are shared by all pose sources.
