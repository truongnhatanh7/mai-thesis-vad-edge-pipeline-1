| arm | paper AUC [95% CI] | live AUC [95% CI] | paper diff vs released (paired 95% CI) | anomalous-frame coverage | person-frames | mean joint distance / box height |
|---|---|---|---|---|---|---|
| released | 85.94 [82.75, 88.79] | 80.98 [78.02, 83.64] | - | 0.965 | 187021 | - |
| rtmpose-s | 86.12 [83.04, 88.91] | 81.26 [78.39, 83.84] | +0.19 [-0.45, +0.81] | 0.965 | 187021 | 0.072 |
| rtmpose-t | 85.91 [82.78, 88.71] | 81.03 [78.12, 83.66] | -0.03 [-0.46, +0.38] | 0.965 | 187021 | 0.070 |
| yolov8n-pose | 84.99 [81.76, 87.93] | 79.77 [76.95, 82.39] | -0.94 [-1.94, +0.12] | 0.963 | 180678 | 0.085 |

O1 (RTMPose-s paper AUC >= 83.0): PASS (86.12)
O2 (RTMPose-t paper AUC >= 80.0): PASS (85.91)
O3 (descriptive): YOLOv8n-pose on the oracle crops 84.99 paper AUC, 180678 person-frames kept of 187021
EXP-016 pose model by the plan's rule: rtmpose-s
