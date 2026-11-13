# EXP-020 — Why the edge pose detector misses small people: labels vs resolution (2 × 2, COCO only)
Frozen 2026-11-08 17:15 before any run (owner: "train P2 head and benchmark", COCO for a general Q1 claim, full 2×2; D23).

## Hypothesis
YOLO-pose is blind below ~64 px (EXP-011) because (a) the COCO→YOLO conversion drops persons without keypoint labels
(mostly small/crowded people), so they are trained as background (**label cause**), and/or (b) stride-8 features are too
coarse (**resolution cause**). A P2 (stride-4) head addresses (b); **partial-label training** (keep keypoint-less persons
as box-only targets, keypoint loss masked for them) addresses (a).

## Design
All from `yolov8n-pose.pt`, COCO train2017 person images only (no ShanghaiTech/UBnormal data), identical schedule
(fixed epochs, imgsz 640, seed 0, MPS), validation on COCO val2017 person keypoints:
| | stock head | + P2 head (stride 4; pretrained weights copied where shapes match) |
|---|---|---|
| standard labels (Ultralytics converter) | **M00** control | **M0P** |
| partial labels (all persons; keypoint-less = box-only, visibility 0) | **ML0** | **MLP** |
Equal-cost reference: stock model at imgsz 800 via NCNN FP16 (EXP-012).
Evaluation on unseen data: ShanghaiTech test L0/L2/L3 and UBnormal L0 (EXP-003/006 pipeline, c 0.25, frozen STG-NF, both
variants, paired bootstrap); recall by person height (EXP-011); NCNN FP16 detector ms in the VM (EXP-012 protocol, quiet
host); delivered AUC at s 1 and 3 with FILL-HOLD (EXP-010 method).
| # | Expectation | Pass rule (paper variant, ShanghaiTech L0 unless stated) |
|---|---|---|
| P1 | label cause | ML0 − M00 ≥ +2.0 AUC with CI lower > 0, and recall(48–96 px) higher |
| P2 | resolution cause | M0P − M00 ≥ +2.0 with CI lower > 0 |
| P3 | general, not site-specific | best of {ML0, M0P, MLP} beats M00 on UBnormal L0 by > 0 (point) and is ≥ M00 − 0.5 |
| P4 | P2 vs input resolution at equal cost | M0P (640) vs stock@800 NCNN: report Δ AUC and Δ ms; P2 "wins" if AUC ≥ stock@800 and ms ≤ 1.10× |
| P5 | combination | MLP ≥ max(ML0, M0P) (descriptive) |

## Amendment 1 — 2026-11-08 18:05, written before any training run and before any result (lane G)
The plan is silent on the points below; they are implementation fixes, not changes of hypotheses or pass rules.
1. **Image set and labels.** Images = every COCO image with ≥ 1 non-crowd person with a box of positive size (train2017: 64,115;
   val2017: 2,693; identical lists in both datasets). *standard* = Ultralytics' converter output (`convert_coco`, use_keypoints,
   persons with no labelled keypoint are not written; our files are byte-identical to the converter's, checked on all files);
   an image whose persons all lack keypoints keeps an empty label file. *partial* = same rows plus every other non-crowd person as
   box + 17 keypoints (0, 0, 0). Persons by height bin are in artifacts/EXP-020/dataset_manifest.json.
2. **"Keypoint loss masked" needs a patch.** Installed Ultralytics 8.4.201 masks only the keypoint *location* loss for visibility-0
   keypoints (utils/loss.py l.833-834, 349-352); the keypoint *visibility* BCE (l.837) is not masked (it trains "all 17
   invisible") and keypoint-less anchors dilute the location-loss mean. `edgelab.pose.loss_patch` removes anchors whose GT
   person has no visible keypoint from both keypoint terms (box/class/DFL untouched). It equals the stock function when every GT
   person has keypoints (tested), so it is installed in **all four** runs (no difference for the standard-label arms).
3. **P2 head = append-only.** `pose/yolov8n-pose-p2.yaml`: layers 0-21 as yolov8-pose, P2 branch (upsample P3-neck, concat backbone
   P2, C2f 128) as layers 22-24, Pose head over [P3, P4, P5, P2]. Reason: the Pose head sizes its hidden convs from the first level's
   channels, so the stock yolov8-p2 order (P2 first) would make every head tensor a new shape; append-only copies all 397 stock
   tensors (3,306,545 elements); 66 tensors (166,608 elements) are new. Cost: fused 3.455 M params, 17.61 GFLOPs at 640x640
   (stock 3.290 M, 9.18) = 1.92×, i.e. more than stock@800 (1.56×); P4 measures what that costs in ms.
4. **Fixed schedule (all four arms).** 5 epochs [superseded by Amendment 1b: 4 epochs] over the 64,115 training images, batch 16, imgsz 640, seed 0, MPS, SGD lr0 0.01
   (lrf 0.01), warmup 1 epoch, mosaic off in the last epoch, no early stopping, other Ultralytics defaults. Chosen from measured
   speed (stock ≈ 40 min/epoch, P2 slower) so that the slowest run stays ≤ ~8 h. The **last-epoch** weights are evaluated (best.pt
   would be picked on a validation set whose labels differ between standard and partial arms). The Ultralytics val2017 numbers
   (standard labels for M00/M0P, partial for ML0/MLP) are monitoring only and are not compared across label policies.
5. **Evaluation details.** (a) "recall(48-96 px)" of P1 = EXP-011 recall pooled over the 48-64 and 64-96 px bins, SHT L0, conf 0.25,
   IoU ≥ 0.3. (b) P3 "best of {ML0, M0P, MLP}" = highest SHT L0 paper AUC; P3 passes if its UBnormal L0 paper AUC > M00's and its
   SHT L0 paper AUC ≥ M00's − 0.5 points (the "≥ M00 − 0.5" clause is evaluated on SHT, because on UBnormal it is implied by "> 0");
   "any arm beats M00 on UB" is reported next to it. (c) P4 uses the NCNN FP16 640 detections of M0P (same runtime and precision
   as the EXP-012 stock@800 reference cache L0_E0-ncnn16-800) for ΔAUC (paired bootstrap), and detector ms (pre+infer+post) from
   one VM timing session in which stock@640, stock@800 and the four models are timed together (EXP-012 protocol, 3 reps, quiet host);
   the ratio is M0P / stock@800. (d) Delivered AUC (SHT L0, s = 1 and 3): EXP-010 method (keep masks seed 1, P fill G 8, FILL-HOLD
   and SKIP reported) on each model's MPS detection cache, with the service-time trace measured in the VM for that model's NCNN
   FP16 640 export (stock@800: its own trace) multiplied by s; paper and live variants reported, live is the delivered one.
   (e) Primary AUC variant for P1-P5 is "paper" as in the plan; "live" is reported beside it.

### Amendment 1b — 2026-11-08 18:11, still before any training run
Smoke test (43 iterations, SGD, partial labels, host busy with the COCO download/unzip): stock 1.6 it/s (0.62 s/it); P2 model 0.7 it/s (1.4 s/it),
i.e. P2 is ~2.2× slower per iteration, not the 1.5× I assumed. With 4,008 iterations per epoch that is ≈ 42 min/epoch (stock) and
≈ 93 min/epoch (P2). **Epochs: 4** (not 5): P2 runs ≈ 6.5-7 h, stock ≈ 3 h, so every run stays within ~8 h even with the evaluation
of finished arms running alongside. Item 4 of Amendment 1 otherwise unchanged (SGD lr0 0.01, warmup 1 epoch, mosaic off in the last epoch).
Also found in the smoke test: an init checkpoint without `train_args.task = "pose"` is read by Ultralytics as task "detect" (box-only
dataset, no keypoints); `p2model.save_checkpoint` now writes it and a test guards it. A first smoke with Ultralytics' optimizer=auto
(AdamW 0.002 below 10,000 iterations) wrecked the pretrained model (val mAP50-95 0.003), hence the pinned SGD.
