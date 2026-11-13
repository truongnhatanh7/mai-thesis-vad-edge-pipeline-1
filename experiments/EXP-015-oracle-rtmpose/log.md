# EXP-015 log (append-only)

### 2026-11-07 18:40 — Lane D start: models, code, smoke
- Read plan.md (frozen), RESEARCH note, detcache/replay/score_dir/stgnf_format/stgnf_eval, EXP-009 log (released frame ids 0-based = jpg numbering), EXP-003 walkthrough.
- No new Python dependency: rtmlib not used. Models downloaded with curl from the official OpenMMLab URLs (Apache-2.0, MMPose RTMPose body7 `rtmposev1/onnx_sdk`), 256x192, SimCC head, dynamic batch. The pre/post-processing is `src/edgelab/pose/rtmpose.py` (copied from the model's own pipeline.json: padding 1.25, ImageNet mean/std RGB, SimCC argmax/2, score = min(max x, max y)). Files in ~/thesis-data/exp015/models/ (sha256 also in SHA256SUMS.txt there):
  - rtmpose-t_simcc-body7_pt-body7_420e-256x192-026a1439_20230504.zip  937003a70832d9cc34ea16927f504792f3133e92dda1b9c626236bbbe9e805cb  (https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-t_simcc-body7_pt-body7_420e-256x192-026a1439_20230504.zip)
  - rtmpose-s_simcc-body7_pt-body7_420e-256x192-acd4a1ef_20230504.zip  7673922e531014906ca4f0f239b7e233b740146a10b632deaa2a28d45470d802  (same base URL)
  - rtmpose-t-body7-256x192.onnx (= end2end.onnx inside the t zip)  a6c2f6a3896a4d51131d14d7a80a3d08b50f559af5a58a45d5b098aef510a70f
  - rtmpose-s-body7-256x192.onnx (= end2end.onnx inside the s zip)  9aeb635b83f86aea45cf45d85798f7eba1a162de8e0d721c44e54fe5eebaf47d
- Smoke (light, 2 threads, videos 01_0014 and 06_0144, 1,207 person-frames): all released person-frames have a usable box (0 without), O3 YOLOv8n-pose on crops drops 12 of 1,207 (no detection with IoU >= 0.3).
- Keypoint order check (src/edgelab/pose/oracle_check.py): mean |new - released| / box height over the 17 joints = 0.061 (rtmpose-s), 0.060 (rtmpose-t); the same with left/right swapped = 0.172 / 0.174. Order is COCO-17, same as released. Plot: EXP-015-oracle-rtmpose/figs/rtmpose-s_06_0144.png (red released, green new) overlaps on the people.
- Finding that shapes a convention: the RTMPose score is in 0..1 but lower than AlphaPose confidence (median 0.37 vs 0.75 on 20 person-frames). STG-NF's test pass drops the confidence channel (`--model_confidence` is off, pose_shape = [:2]), and `scores` is only read by a training threshold of 0, so the confidence scale cannot change the AUC. Convention fixed before any scoring: new keypoint conf = the model's own score, `scores` field = released value.
- Heavy runs (full test set) wait for ~/thesis-data/locks/vm_timing_done (tmux waiter, polls every 60 s).

### 2026-11-07 18:40 — code done, waiter running
- Time note: the entry above is stamped 18:40 but was written about 18:32 by the host clock; later entries use the host clock (`date`).
- Code: src/edgelab/pose/{rtmpose,oracle_rtmpose,oracle_check}.py, src/edgelab/analysis/exp015.py; tests/test_pose_rtmpose.py (14 tests pass), tests/test_analysis_exp015_016.py (verdict rules). O3 = YOLOv8n-pose (artifacts/EXP-001/models/yolov8n-pose.pt, sha256 c6fa93dd...d212) at imgsz 256 on the same 256x192 crops, conf 0.01; best IoU >= 0.3 with the oracle box (in crop coordinates), else the person-frame is dropped.
- Scoring path smoke-tested on a 2-video dir (STG-NF + score_dir run, AUC meaningless on 2 videos).
- tmux `e15` runs EXP-015/run.sh: waits for the lock (polls every 60 s), then 3 pose passes (4 threads), then scores released + 3 arms (exp015 score -> rows.jsonl, MLflow experiment EXP-015), then report -> results_table.md + verdicts.json.

### 2026-11-07 19:57 — driver
- lock /Users/nanh/thesis-data/locks/vm_timing_done present, heavy run starts

### 2026-11-07 21:11 — poses rtmpose-s
- 107 pose files in /Users/nanh/thesis-data/exp015/poses/rtmpose-s (107 expected); per-video stats in /Users/nanh/thesis-data/exp015/pose_rtmpose-s.log

### 2026-11-07 21:41 — poses rtmpose-t
- 107 pose files in /Users/nanh/thesis-data/exp015/poses/rtmpose-t (107 expected); per-video stats in /Users/nanh/thesis-data/exp015/pose_rtmpose-t.log

### 2026-11-07 22:21 — poses yolov8n-pose
- 107 pose files in /Users/nanh/thesis-data/exp015/poses/yolov8n-pose (107 expected); per-video stats in /Users/nanh/thesis-data/exp015/pose_yolov8n-pose.log

### 2026-11-07 22:23 — scored released
- {"arm": "released", "variant": "paper", "auc": 0.8593709088366215, "coverage_anomalous": 0.9654853976682443, "person_frames": 187021} {"arm": "released", "variant": "live", "auc": 0.8097943661145235, "coverage_anomalous": 0.9673900496363846, "person_frames": 187021} 

### 2026-11-07 22:25 — scored rtmpose-s
- {"arm": "rtmpose-s", "variant": "paper", "auc": 0.8612262058091633, "coverage_anomalous": 0.9654853976682443, "person_frames": 187021} {"arm": "rtmpose-s", "variant": "live", "auc": 0.8125744478742695, "coverage_anomalous": 0.9673900496363846, "person_frames": 187021} 

### 2026-11-07 22:27 — scored rtmpose-t
- {"arm": "rtmpose-t", "variant": "paper", "auc": 0.8590617387937989, "coverage_anomalous": 0.9654853976682443, "person_frames": 187021} {"arm": "rtmpose-t", "variant": "live", "auc": 0.8102781141887981, "coverage_anomalous": 0.9673900496363846, "person_frames": 187021} 

### 2026-11-07 22:29 — scored yolov8n-pose
- {"arm": "yolov8n-pose", "variant": "paper", "auc": 0.8499327482688118, "coverage_anomalous": 0.9630612951633384, "person_frames": 180678} {"arm": "yolov8n-pose", "variant": "live", "auc": 0.7977051544295688, "coverage_anomalous": 0.9648505136788641, "person_frames": 180678} 

### 2026-11-07 22:29 — driver
- all scored, table in results_table.md; EXP015_DONE
