# EXP-001 walkthrough — from frames to an AUC

Plain-language map of the code used. Every box names the file and function, and the test that guards it.

```
 test frames (JPG, 856x480)                   released AlphaPose poses (data.zip)
        │                                                  │
 [1] YOLOv8n-pose + BoT-SORT  ── per frame: people, 17 keypoints, track id
        │
 [2] converter ── STG-NF JSON  {person: {frame: {keypoints, scores}}}   ◄── same format ──┘
        │
 [3] STG-NF loader: cut each person's track into 24-frame segments (stride 1), add neck point (17→18),
     normalise (÷856x480, centre, ÷ y-std)
        │
 [4] frozen normalizing flow: segment → log-likelihood ("normality"); anomaly score = −normality
        │
 [5] assemble per-frame scores: segment score → one frame of the segment; frame score = least-normal person;
     empty frames = "most normal"
        │
 [6] smoothing over time: none | offline Gaussian (paper) | causal half-Gaussian | causal EWMA
        │
 [7] frame-level AUC over all 40,791 test frames + bootstrap CI over the 107 videos → MLflow
```

| Step | What happens | Code | Guarded by |
|---|---|---|---|
| 1 | Read frames (JPG folder or .avi), run `model.track` (pose model + tracker) on CPU; a fresh tracker per video. Detections without a confirmed track id are dropped | `src/edgelab/pose/extract.py:extract_video`, `iter_frames` | `tests/test_pose_extract.py` (frame N → key N) |
| 2 | Ultralytics `Results` → `TrackedPose` → JSON with 4-digit 0-based frame keys, COCO-17 `[x, y, conf]` in original pixels, box conf as `scores` | `src/edgelab/pose/stgnf_format.py:poses_from_ultralytics_result`, `to_stgnf_dict`, `write_stgnf_json` | `tests/test_pose_format.py`: round trip; our parser reads a released AlphaPose file exactly like STG-NF's `single_pose_dict2np`; a re-written released file gives identical STG-NF segments |
| 3–4 | STG-NF's own loader and model (third-party, pinned commit, CPU patch) | `src/edgelab/pose/stgnf_eval.py:score_segments` → `third_party/STG-NF/dataset.py`, `models/training.py:Trainer.test` | R1 reproduces 85.94 (= their script's printed AUC) |
| 5 | Re-implementation of STG-NF `get_dataset_scores`, with a parameter for which frame of the segment gets the score (12 = paper, 23 = live) | `stgnf_eval.py:assemble_clip_scores` | `tests/test_pose_stgnf_scoring.py::test_assembly_matches_stgnf_get_dataset_scores` |
| 6 | Smoothers | `src/edgelab/pose/smoothing.py` | `test_offline_gaussian_equals_stgnf_smooth_scores`; `test_causal_filters_ignore_future_frames` (changing future frames never changes past outputs) |
| 7 | Metric + CI + MLflow run + per-frame scores saved as `.npz` | `src/edgelab/pose/evaluate.py:evaluate`, `edgelab/metrics/scores.py` | `tests/test_metrics.py` (vs sklearn / scipy) |

Supporting tools:
- `src/edgelab/pose/sanity.py` — per-video tracks, poses, track length, keypoint confidence, usable segments for two
  pose sources (its segment counter reproduces STG-NF's 144,714 test segments).
- `src/edgelab/pose/analysis.py` — per-scene / per-video AUC of two score files and per-video mean luma.
- `src/edgelab/pose/coverage.py` — which frames each pose source scores at all, and AUC on frames both cover.
- `src/edgelab/pose/stage_benchmark.py` — per-stage CPU cost (pose model, tracker, flow) at 1/2/4 torch threads.
  Ultralytics resets torch threads to 8 on model setup; the benchmark re-applies the count after the first call
  and asserts it (`extract.py` does the same for `--threads`).

R3 (retraining) uses STG-NF's own `train_eval.py` unchanged apart from the CPU patch, with
`--pose_path_train/--pose_path_test` pointing at our YOLO JSON folders; the checkpoint is then scored with
`evaluate.py --checkpoint ...` like any other.

Commands (in order) are in `log.md`.

Two "future frames" sources, both removed in the live variant (`score_frame=end` + causal smoother):
1. the segment's score is written on its middle frame (needs 12 frames from the future);
2. the paper's Gaussian smoothing is centred (needs ≈ 30 future frames).
