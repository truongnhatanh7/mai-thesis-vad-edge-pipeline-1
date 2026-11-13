# EXP-015 — Is a light top-down pose model good enough when the people are found? (oracle boxes + RTMPose)
Frozen 2026-11-07 18:30 before any run. Boxes and track ids from the released AlphaPose test poses (oracle detection + tracking);
keypoints re-estimated with RTMPose-s and RTMPose-t (official ONNX via rtmlib, Apache-2.0, sha256 logged) on each crop
(standard 1.25× box padding, 256×192 input); frozen STG-NF ShanghaiTech checkpoint, both variants.
| # | Expectation | Pass rule (paper variant) |
|---|---|---|
| O1 | RTMPose-s ≈ reference | AUC ≥ 83.0 |
| O2 | RTMPose-t good enough | AUC ≥ 80.0 |
| O3 | YOLOv8n-pose keypoints on the same oracle boxes (descriptive) | reported for comparison |
