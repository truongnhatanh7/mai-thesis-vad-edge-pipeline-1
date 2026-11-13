# EXP-017 — YOLO26-pose vs YOLOv8n-pose as the edge front end

Pre-registration frozen 2026-11-07 22:20, before any YOLO26 run (owner: "why don't we use yolo26 since it's better
for edge deployment?"; D19). Starts after the currently running EXP-012/015/016 caches (host contention); VM timing
only on an idle host (EXP-004 exclusion rule).

## 1. Why
YOLOv8n-pose is the frozen front end of EXP-001–016 (and of the closest edge pipeline, Huang 2026). EXP-011 showed it is
blind below ~64 px. Ultralytics' YOLO26 family (present in our installed Ultralytics 8.4.201: `cfg/models/26/
yolo26-pose.yaml`, plus a `yolo26-p2` small-object variant) is marketed as edge-optimised (end-to-end without NMS,
faster CPU inference, small-object-aware training). Those claims are the vendor's; this experiment measures them on
our task.

## 2. Method
- Models: official **YOLO26n-pose** release weights (sha256 logged); if an official YOLO26 pose P2 checkpoint exists,
  include it, otherwise P2 is out of scope (no training in this experiment). Same imgsz 640 (rect letterbox),
  conf-0.01 cache (YOLO26 end-to-end output: cache all returned boxes ≥ 0.01; threshold filtering stays exact without NMS).
- Accuracy: EXP-003/006 pipeline (BoT-SORT gmc none, c 0.25 + gate sweep, frozen STG-NF): ShanghaiTech L0, L2, L3 and
  UBnormal L0; recall by person height as EXP-011 (ShanghaiTech L0).
- Cost: static ONNX (ORT FP32) and NCNN FP16 at 384×640 and 480×800, VM timing as EXP-012 (12 × 40 frames, co-load,
  O0-B4, 3 reps, host-load exclusion 30%).
- Delivered: live AUC at 24 fps at s = 1 and s = 3 (EXP-008/010 method, P-fill G 8 as in EXP-010 FILL-HOLD, seed 1).

## 3. Expectations
| # | Expectation | Pass rule |
|---|---|---|
| Y1 | YOLO26n-pose finds more small people | ShanghaiTech L0 recall in the 48–64 and 64–96 px bins both higher than YOLOv8n-pose (EXP-011) |
| Y2 | higher offline accuracy | ShanghaiTech L0 paper AUC ≥ 68.53 + 1.5 with paired CI lower bound > 0 |
| Y3 | no transfer loss | UBnormal L0 AUC ≥ 55.76 − 0.5 |
| Y4 | at least as cheap | NCNN FP16 detector ms at 640 ≤ 1.10 × YOLOv8n-pose NCNN FP16 at 640 (EXP-012) |
| Y5 | delivered win | delivered live AUC (FILL-HOLD) at s = 1 and s = 3 > YOLOv8n-pose's (same masks/seed) |

## 4. Deviations policy
As EXP-003. If no official YOLO26 pose weights can be obtained, the experiment stops and is reported as not run.
