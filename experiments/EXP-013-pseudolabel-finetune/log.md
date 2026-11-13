# EXP-013 log (append-only)

### 2026-11-07 18:26 — Lane C: start
- Read plan.md (frozen), detcache/replay/score_dir/gate_sweep/stgnf_format/stgnf_eval, exp003/exp005, EXP-003/006 walkthroughs, EXP-009 log.
- Data found: 330 training videos (~/thesis-data/shanghaitech/shanghaitech/training/videos/*.avi, 856x480, 24 fps); released train poses already extracted at ~/thesis-data/stgnf/data/ShanghaiTech/pose/train/<video>_alphapose_tracked_person.json (330 files; first video: frame keys 0..763 = 764 video frames).
- Plan: yolo_dataset.py (labels), finetune.py, wise.py (EXP-014), analysis/exp013.py (tables, verdicts). Training and all heavy jobs wait for ~/thesis-data/locks/vm_timing_done (tmux waiter).

### 2026-11-07 18:33 — dataset built (data preparation only; no training yet)
- Built ~/thesis-data/exp013-data with edgelab.pose.yolo_dataset (every 5th frame: 0, 5, 10, ...; jpg q95; labels from released tracked_person.json; rules in the module docstring: box = visible keypoints (conf > 0.3) +10% each side, v = 2/0, persons with < 3 visible keypoints dropped, frames without a person keep an empty label file).
- Split by video, stratified per scene (round(10% of videos), >= 1 per scene, seed 0): 295 train videos / 35 val videos (all 13 scenes in val). Images: 49,957 train / 5,053 val (55,010 of 274,515 decoded frames). Persons labelled: 211,343 (3,434 sampled frames have none = background). Split list: ~/thesis-data/exp013-data/split.json (copied to artifacts/EXP-013).
- Frame count checks (330 videos): decoded frame count = last released pose frame + 1 for all but two videos: 04_018 header says 649 frames, OpenCV decodes 648 (header off by one; pose max frame 495); 08_001 has released poses on frames 816-817 although the video decodes 816 frames (0..815): those two poses are not used (no image).
- Frame-id alignment of released train poses vs decoded frames: pose frame ids run 0..N-1 = decode order (0-based). IoU check against the detector (pose/align_check.py, shifts -2..+2) is queued behind the lock because it runs the detector; if it peaks away from 0 the labels (not the images) are rebuilt.
- Code + tests so far: pose/yolo_dataset.py (7 tests), pose/wise.py (4), pose/segment_coverage.py (5, window rule checked against STG-NF's own split_pose_to_segments on 200 random tracks), pose/align_check.py (3), analysis/exp013.py + analysis/exp014.py (8); gate_sweep.py got --gates (default unchanged).
- tmux e13-chain waits for ~/thesis-data/locks/vm_timing_done (polls every 60 s), then runs chain.sh: align -> train -> eval -> wise -> held-out coverage -> alpha* -> eval alphas. Log: ~/thesis-data/exp013-chain.log.

### 2026-11-08 09:05 — Lead: training finished; wrapper crash fixed; chain resumed at evaluation
- Ultralytics training completed normally: 25 epochs, early stop (patience 5), best epoch 20; final val box mAP50 0.951,
  mAP50-95 0.711; pose mAP50 0.808, mAP50-95 0.484 (5,053 val images, 19,208 instances).
- Our wrapper `edgelab.pose.finetune` then crashed reading `best_fitness` (None in Ultralytics' stripped final best.pt);
  the chain treated training as failed and stopped (~08:44). Weights had already been copied
  (`artifacts/EXP-013/yolov8n-pose-sht-ft.pt`, 6,813,016 B). Fix: best epoch from results.csv with Ultralytics' pose
  fitness when the checkpoint is stripped; `--summarize-only` mode. Summary written (`train_summary.json`).
- `train.done` marked; chain restarted in tmux `e13-chain` (stages already done are skipped): eval → WiSE → held-out α.
