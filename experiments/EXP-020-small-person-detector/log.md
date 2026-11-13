# EXP-020 log (append-only)

### 2026-11-08 17:45 — Lane G: start; step 1 download launched
- Read plan.md (frozen), STATUS-NOW, EXP-013 plan/log/results + finetune.py, RESEARCH-p2-head note, EXP-011 results.
- Disk free 157 GB (> 60 GB). tmux e13-chain does not exist now (no tmux server was running at start).
- Download: scripts/download_coco.sh in tmux e20-dl (aria2c -c, explicit names, unzip -t, size+sha256 in *.done files) -> ~/thesis-data/coco. URLs: images.cocodataset.org/{zips/train2017.zip, zips/val2017.zip, annotations/annotations_trainval2017.zip} (person_keypoints_{train,val}2017.json are inside the annotations zip). COCO publishes no official checksums; recorded sha256 are ours.

### 2026-11-08 17:52 — code so far (no training yet)
- Downloads: annotations_trainval2017.zip (252,907,541 B, sha256 113a836d...), val2017.zip (815,585,330 B, sha256 4f7e2ccb...) done and unzip -t OK; train2017.zip downloaded (19,336,861,798 B), verification/extract in tmux e20-dl.
- Converter check (installed Ultralytics 8.4.201, ultralytics/data/converter.py lines 325-326): `if use_keypoints: if not any((ann.get("keypoints") or [])[2::3]): continue` -> persons without a labelled keypoint are NOT written (confirmed; crowd persons skipped at line 313; boxes with w or h <= 0 at 321).
- Pose loss check (ultralytics/utils/loss.py): line 833 `kpt_mask = gt_kpt[..., 2] != 0`; 834 keypoint location loss (KeypointLoss, 349-352: `kpt_loss_factor * ((1 - exp(-e)) * kpt_mask)).mean()` -> exactly 0 for an all-zero-visibility person, but it still counts in the .mean() over positive anchors); 837 `kpts_obj_loss = bce_pose(pred_kpt[..., 2], kpt_mask.float())` -> NOT masked: trains visibility -> 0 for those persons. => stock Ultralytics does NOT make keypoint-less persons box-only. Patch written (pose/loss_patch.py: drop anchors whose GT person has no visible keypoint from both keypoint terms); equals stock when every GT person has keypoints (test), so it is installed for all four runs. Needs plan amendment (to write before training).
- Augmentation keeps v=0 (data/augment.py apply_keypoints: visibility only ever set to 0) and utils.py verify_image_label accepts the all-zero rows.
- New modules + tests: pose/coco_pose_labels.py (7 tests; standard rows byte-equal to convert_coco on toy data), pose/loss_patch.py (6 tests), pose/p2model.py + pose/yolov8n-pose-p2.yaml (5 tests; pre-flight gate: P3-P5 outputs equal stock).
- P2 model (append-only layout; Pose head over [P3,P4,P5,P2]; the Pose head sizes hidden convs from level-0 channels, so P2 first would change every head tensor's shape): all 397 stock tensors copied (3,306,545 elements), 66 new tensors (166,608 elements: layers 22-24 + head level-3 modules). Params fused 3,289,964 -> 3,455,200; GFLOPs fused @640x640 9.178 -> 17.612 (1.92x), @384x640 5.507 -> 10.567. (Research note's 1.67x was for the standard-order P2 yaml whose head is narrower and transfers only backbone+neck.) artifacts/EXP-020/p2_transfer.json.

### 2026-11-08 17:56 — step 2: datasets built (images symlinked from ~/thesis-data/coco/images; labels ~/thesis-data/exp020/data/{standard,partial})
- Image set = every image with >= 1 non-crowd person (box w,h > 0): train2017 64,115 images, val2017 2,693 (identical lists in both datasets; manifest.json says image_lists_equal = true).
- standard == Ultralytics convert_coco(use_keypoints=True, cls91to80=False) byte-for-byte on all 64,115 + 2,693 label files (scripts/verify_standard_vs_converter.py -> artifacts/EXP-020/converter_check.json: 0 differing, 0 extra).
- Persons by COCO box height (px), train2017 [standard | partial]: <32: 159 | 57,392; 32-48: 2,343 | 29,201; 48-64: 7,720 | 22,587; 64-96: 24,693 | 31,605; 96-160: 36,953 | 37,919; >=160: 77,945 | 78,548; total 149,813 | 257,252 (107,439 keypoint-less persons added). Empty-label (pure background) images: standard 0 train? see manifest; val: standard 347 of 2,693 empty (all persons keypoint-less), partial 0.
- val2017: standard 6,352 persons (<32: 6, 32-48: 107, 48-64: 352, 64-96: 1,040, 96-160: 1,652, >=160: 3,195); partial 10,777 (2,332 / 1,225 / 957 / 1,323 / 1,705 / 3,235).
- Note: the converter's 149,813 train persons equals the known coco-pose count; the 64,115 images include images whose persons all lack keypoints (empty label files in standard).

### 2026-11-08 18:12 — step 3-4 setup; Amendment 1 / 1b written; training chain + eval waiter launched
- Correction to the 17:5x entry: standard train2017 has 7,516 empty-label images (all persons keypoint-less), partial 0.
- Smoke tests (val2017 images as training set, fraction 0.25 = 43 iterations): optimizer=auto -> AdamW 0.002 wrecked the pretrained model (val box mAP50-95 0.003) -> optimizer pinned to SGD 0.01 (stock model after 43 SGD iterations: box mAP50 0.61, pose mAP50 0.31 on partial val, not wrecked). Init checkpoint without train_args.task was read as 'detect' (box-only training of a pose model!) -> p2model.save_checkpoint writes task='pose' + regression test; p2 init rebuilt (same 397/66 tensor counts).
- Speed (host busy with download): stock 1.6 it/s, P2 0.7 it/s (2.2x slower) -> epochs = 4 (plan.md Amendment 1b); P2 about 6.5-7 h, stock about 3 h.
- Scripts: EXP-020-small-person-detector/scripts/{download_coco.sh, chain_train.sh, chain_eval.sh, eval_arm.sh, ncnn_parity.py, verify_standard_vs_converter.py}. tmux e20-chain (waits for: e13-chain absent [it is], COCO extraction; then trains M00, ML0, M0P, MLP sequentially, 4 epochs each, resumable) and e20-eval (waits for each arm's train.done, then NCNN export + parity, MPS caches SHT L0/L2/L3 + UB L0 via run_eval.sh, NCNN FP16 L0 cache, gate sweeps). Logs: ~/thesis-data/exp020/{chain.log, chain_eval.log, train_<arm>.log}.
- Also written (tests pass): analysis/exp020.py (tables, recall by height, verdicts P1-P5, delivered), pipeline/timing_exp020.py (VM timing driver; to run on a quiet host after training+evals).

### 2026-11-09 06:39 — Lead: M0P crash (GPU error) and sequential restart
- train_M0P failed at 00:32 (8 min in, epoch 1 at 6%): torch.AcceleratorError "MPS command buffer execution failed: Discarded
  (victim of GPU error/recovery)"; it overlapped with ML0's evaluation (started 00:25, NCNN + eval_mps stages). The chain
  stopped; MLP never started; GPU idle until 06:36. Crash log kept as train_M0P.crash1.log.
- Fix (no change to hypotheses/schedule): evaluation waiter stopped; one tmux session `e20-chain` now runs training M0P → MLP
  **then** chain_eval (no concurrent GPU jobs). M0P restarted from scratch at 06:37 (nothing was saved before the crash).
  M00/ML0 trainings and their completed eval stages are untouched.
2026-11-10 09:54 Verified: released Ultralytics coco2017labels-pose.zip drops keypoint-less persons (val rows 6,352 = persons with kpts; 4,425/10,777 = 41% of non-crowd persons dropped). Prior art: OpenPose/HigherHRNet mask unannotated persons (ignore, not positive) -> method not novel; novelty = toolchain audit + downstream VAD/deadline impact.
