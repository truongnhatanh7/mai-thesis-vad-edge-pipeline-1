# EXP-016 — Edge top-down pipeline: nano person detector + RTMPose + keyframes
Frozen 2026-11-07 18:30 before any run; runs after EXP-015 (uses its best RTMPose model if O2 passes, else RTMPose-s).
Detector: YOLOv8n (detection weights, person class) at imgsz 640 and 960; BoT-SORT (gmc none) on detections; top-down
pose (RTMPose) on tracked boxes. Keyframe variant K = 3: detector every 3rd frame, tracker-predicted boxes in between,
pose on every frame. ShanghaiTech test L0 (+ L2), c 0.25, frozen STG-NF; VM cost per frame measured once (O0-B4,
1 rep, descriptive).
| # | Expectation | Pass rule (paper variant) |
|---|---|---|
| T1 | top-down beats single-stage | best config AUC ≥ 68.53 + 2.0 (YOLOv8n-pose 640 baseline) |
| T2 | keyframes are cheap | K = 3 AUC ≥ K = 1 AUC − 1.0 (same detector size) |
| T3 | cost (descriptive) | VM ms/frame for each config with ~N people reported vs the 41.7 ms budget |
