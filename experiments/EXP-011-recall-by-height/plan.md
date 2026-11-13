# EXP-011 — Where does the edge detector miss people? (recall by person height)
Frozen 2026-11-07 18:30 before any run. Data: ShanghaiTech test, released AlphaPose poses (reference people); EXP-003 caches
(YOLOv8n-pose conf-0.01 cache at imgsz 480/640/800/960, L0; 640 at L2, L3).
Method: reference person box = min/max of keypoints with conf > 0; height bins (px, 856×480 frames):
< 32, 32–48, 48–64, 64–96, 96–160, ≥ 160. Edge recall per bin = share of reference persons matched by a cached YOLO
box with conf ≥ 0.25 and IoU ≥ 0.3 (same matcher as EXP-009 D1).
| # | Expectation | Pass rule |
|---|---|---|
| R1 | recall cliff for small people | at 640 L0, recall(< 48 px bins) ≤ 0.5 × recall(≥ 96 px bins) |
| R2 | resolution helps small people most | recall gain 960 vs 640 is largest in a < 64 px bin |
| R3 | darkness hits small people most | recall drop L0 → L3 at 640 (relative) is largest in a < 64 px bin |
