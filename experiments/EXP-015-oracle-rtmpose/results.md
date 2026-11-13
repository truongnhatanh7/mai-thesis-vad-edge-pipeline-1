# EXP-015 results — oracle boxes + light pose models (driver output: `results_table.md`)

| Arm (boxes + tracks from released AlphaPose poses) | paper AUC [95% CI] | live AUC | Δ paper vs released [paired CI] |
|---|---|---|---|
| released AlphaPose keypoints | 85.94 [82.75, 88.79] | 80.98 | — |
| RTMPose-s keypoints | **86.12** [83.04, 88.91] | 81.26 | +0.19 [−0.45, +0.81] |
| RTMPose-t keypoints | **85.91** [82.78, 88.71] | 81.03 | −0.03 [−0.46, +0.38] |
| YOLOv8n-pose keypoints on the same crops | 84.99 [81.76, 87.93] | 79.77 | −0.94 [−1.94, +0.12] |

**O1 pass (86.12 ≥ 83), O2 pass (85.91 ≥ 80), O3: YOLOv8n-pose on oracle crops 84.99.**

Reading (lead): with the right people found and tracked, *any* of these small pose models — even YOLOv8n-pose run on
crops — reaches ≈ 85–86 AUC. Together with EXP-009 (96% of the gap = detection) and EXP-011 (blind below ~64 px) this
pins the edge loss on **detection and tracking, not keypoint quality**. Light top-down pose (RTMPose-t) is "good
enough" for reference-level accuracy.
