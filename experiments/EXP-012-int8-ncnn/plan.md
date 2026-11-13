# EXP-012 — "Quantize to buy resolution": INT8 / NCNN detector at 800 px
Frozen 2026-11-07 18:30 before any run.
Variants of YOLOv8n-pose at rect 384×640 and 480×800: (a) ORT FP32 (current), (b) ORT INT8 static quantization (QDQ,
per-channel, calibration on 200 frames from ShanghaiTech **training** videos), (c) NCNN FP32/FP16 via Ultralytics export
(if NCNN runs in the VM; else reported as not available).
Timing: VM edge-os, EXP-004 protocol (L3 subset 12 × 40 frames, co-load on, O0-B4, 3 reps, host-load exclusion 30%).
Accuracy: detection cache with each fast variant at 800 on ShanghaiTech L0 (and L2) test, replay + STG-NF as EXP-003;
delivered AUC at 24 fps (s = 1 and s = 2) as EXP-008.
| # | Expectation | Pass rule |
|---|---|---|
| Q1 | a fast 800 variant costs ≈ FP32 640 | mean detector ms of the best fast 800 variant ≤ 1.10 × ORT FP32 640 |
| Q2 | it keeps the resolution gain offline | its offline paper AUC (L0) ≥ ORT FP32 640 + 1.0 |
| Q3 | it wins delivered | its delivered live AUC at s = 1 and s = 2 > ORT FP32 640 delivered (L0) |
