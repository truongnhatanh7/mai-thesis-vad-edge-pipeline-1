| config | L0 paper AUC [95% CI] | L0 live | L0 anomalous cov. | L0 normal cov. | L2 paper | L2 live | L2 anomalous cov. | person-frames L0 | VM ms/frame mean (p50 / p99) | boxes/frame in VM subset |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline YOLOv8n-pose 640 (single stage, EXP-003) | 68.53 [63.79, 73.60] | 66.17 [62.23, 70.34] | 0.674 | 0.535 | 67.05 [62.30, 72.32] | 66.11 [62.28, 70.34] | 0.633 | 69895 | see EXP-004 | |
| 640 px, K=1, rtmpose-s | 80.52 [77.20, 83.71] | 76.23 [73.18, 79.01] | 0.915 | 0.737 | 79.34 [75.99, 82.54] | 75.71 [72.41, 78.96] | 0.871 | 144187 | 71.8 (73.6 / 133.8) | 2.40 |
| 640 px, K=3, rtmpose-s | 76.40 [72.41, 80.20] | 73.31 [69.83, 76.69] | 0.896 | 0.735 | 73.78 [69.84, 77.52] | 71.81 [68.11, 75.53] | 0.854 | 132975 | 44.6 (35.9 / 130.4) | 2.24 |
| 960 px, K=1, rtmpose-s | 83.11 [80.15, 85.92] | 78.42 [75.68, 81.10] | 0.941 | 0.763 | 80.96 [77.95, 83.78] | 76.92 [74.20, 79.71] | 0.898 | 162679 | 111.6 (113.2 / 178.8) | 2.75 |
| 960 px, K=3, rtmpose-s | 78.68 [75.15, 82.19] | 75.11 [71.86, 78.15] | 0.928 | 0.757 | 76.47 [72.87, 80.05] | 73.79 [70.49, 77.31] | 0.880 | 151570 | 58.5 (55.8 / 161.9) | 2.56 |

T1 (best L0 paper AUC >= 70.53): PASS; best = n960_K1 83.11; paired diff vs the single-stage baseline +14.58 [+10.15, +18.72]
T2 at n640: K=3 76.40 vs K=1 80.52 (rule: K=3 >= K=1 - 1.0): FAIL
T2 at n960: K=3 78.68 vs K=1 83.11 (rule: K=3 >= K=1 - 1.0): FAIL

INVALID rows (kept for the record; scored on an incomplete MPS-crashed cache, excluded from everything above): config, L0/L2 paper AUC, person-frames (poses), reason
- L0_n960_K1_rtmpose-s: paper AUC 49.12, poses 6865, incomplete MPS-crashed cache (4/107 videos)
- L0_n960_K3_rtmpose-s: paper AUC 49.11, poses 6552, incomplete MPS-crashed cache (4/107 videos)
- L2_n640_K1_rtmpose-s: paper AUC 68.67, poses 113672, incomplete MPS-crashed cache (94/107 videos)
- L2_n640_K3_rtmpose-s: paper AUC 64.49, poses 104315, incomplete MPS-crashed cache (94/107 videos)
- L2_n960_K1_rtmpose-s: paper AUC 48.45, poses 13584, incomplete MPS-crashed cache (11/107 videos)
- L2_n960_K3_rtmpose-s: paper AUC 48.36, poses 12202, incomplete MPS-crashed cache (11/107 videos)
