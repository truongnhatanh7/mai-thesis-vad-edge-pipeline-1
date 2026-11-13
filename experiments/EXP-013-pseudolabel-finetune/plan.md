# EXP-013 — Fine-tune YOLOv8n-pose on ShanghaiTech TRAIN frames with released AlphaPose poses as labels
Frozen 2026-11-07 18:30 before any run. **Leakage rule:** only ShanghaiTech *training* videos and their released poses are used for
training/validation; no test frame, test pose or anomaly label is used. Labels: person box = keypoint min/max padded
10%, 17 keypoints with visibility from AlphaPose confidence (> 0.3). ~1 frame in 5 sampled; 90/10 split by *video*
for early stopping. Init: yolov8n-pose.pt; imgsz 640; Ultralytics training on MPS; ≤ 30 epochs, patience 5; seed 0.
Evaluation: EXP-003 pipeline (cache conf 0.01 → replay c 0.25 → frozen STG-NF) on ShanghaiTech test L0, L2, L3 and
UBnormal L0 (transfer, never trained on).
| # | Expectation | Pass rule (paper variant; live reported) |
|---|---|---|
| F1 | more people found | SHT L0 anomalous coverage ≥ baseline + 0.04 |
| F2 | higher accuracy | SHT L0 AUC ≥ baseline + 1.5 with paired CI lower bound > 0 |
| F3 | no transfer damage | UBnormal L0 AUC ≥ baseline − 0.5 |
| F4 | dark robustness (descriptive) | Δ at L2 and L3 reported |
