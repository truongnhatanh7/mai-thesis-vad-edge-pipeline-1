# EXP-003 walkthrough — from frames to an AUC under simulated low light

Plain-language map of the code. Each step names the file, the function, and the test that guards it.
Idea: the expensive part (the detector) runs once per (darkness level × enhancement) and is cached. The cheap part
(tracker gate threshold `c` → STG-NF → AUC) is replayed many times from the cache.

```
 clean test frames (856x480 JPG, 107 video folders)
        │
 [1] darken      level L0..L4 / L2-nonoise: sensor model (light → electrons → noise → 8 bit), seeded per frame
        │
 [2] enhance     E0 none | E1 gamma | E2 CLAHE | E3 Zero-DCE++      (applied to every frame)
        │
 [3] detector    YOLOv8n-pose, conf 0.01  → ALL boxes + 17 keypoints per frame  → one <video>.npz  (cache)
        │                                   (+ luma and ms timings per frame)
        ▼
 [4] replay      keep detections with conf ≥ c → BoT-SORT (no GMC, no ReID) → track ids → STG-NF JSON files
        │
 [5] STG-NF      frozen checkpoint: 24-frame segments → normality   (third-party, unchanged)
        │
 [6] frame scores  paper = middle frame + offline Gaussian | live = last frame + causal half-Gaussian
        │           frames with nobody = "most normal"; coverage = frames with ≥ 1 scored person
        ▼
 [7] AUC + 95% CI over videos (+ paired bootstrap between arms) → MLflow experiment EXP-003
```

| Step | What happens | Code | Guarded by |
|---|---|---|---|
| 1 | sRGB → linear → `α·L·1000` electrons → Poisson + N(0, 2²) → clip → sRGB → uint8. Seed = SHA-256 of `video|frame|level` (never Python's `hash()`) | `src/edgelab/preprocess/darken.py:darken`, `darken_level`, `level_seed`, `LEVELS` | `tests/test_preprocess_darken.py` (α=1 no noise = identity ±1; luma falls with α; same seed = same noise; noise-free is smooth; seed value pinned) |
| 2 | E1: `γ = clip(log 0.4 / log meanY, 0.2, 1)` via lookup table. E2: CLAHE (clip 2.0, 8×8 tiles) on LAB L. E3: Zero-DCE++ official weights, scale 1 (torch or ONNX Runtime) | `src/edgelab/preprocess/enhance.py:gamma_adaptive`, `clahe_lab`, `ENHANCERS`; `zerodcepp.py:ZeroDCEppNet`, `ZeroDCEpp`, `ZeroDCEppOnnx` | `tests/test_preprocess_enhance.py` (gamma hand values and black/white guards; our network = official `model.py` output with the same weights; torch vs ONNX Runtime parity on 480×856) |
| 3 | Per frame: darken → enhance → `model.predict(imgsz 640, conf 0.01, iou 0.7)` (no tracker). One compressed `.npz` per video; existing files are skipped (resumable); written atomically | `src/edgelab/pose/detcache.py:cache_video`, `save_atomic`, `load_detections`, CLI `main` | `tests/test_pose_detcache_replay.py` (flat arrays split back per frame; one row per frame with timings) |
| 3b | MPS-vs-CPU rule from plan §4.3 | `detcache.py:mps_vs_cpu_check`, `device_check_passes` | `test_device_check_rule_from_plan` (rule only; the real check is run by the lead) |
| 4 | Filter `conf ≥ c`; Ultralytics `BOTSORT` (`gmc_method none`, `with_reid False`, `track_high_thresh = new_track_thresh = c`, `track_low_thresh = min(0.1, c/2)`); the tracker's `idx` column links each track row back to its keypoints. Optional `keep_mask` hides frames from the tracker. Output named `<video>_<tag>_tracked_person.json` | `src/edgelab/pose/replay.py:build_tracker`, `replay_video`, `replay_dir`, CLI `main` | `tests/test_pose_detcache_replay.py` (a track never mixes two people's keypoints; weak person tracked only when `c` is low; `keep_mask`; file naming; ByteTrack option) |
| 5 | STG-NF loader + model, unchanged from EXP-001 | `src/edgelab/pose/stgnf_eval.py:score_segments` | EXP-001 R1 (85.94) |
| 6 | Segment score → one frame of the segment; frame score = least-normal person; coverage from the `+inf` ("nobody") frames | `stgnf_eval.py:clip_normality`, `assemble_clip_scores`; `score_dir.py:variant_scores`, `coverage` | `tests/test_pose_stgnf_scoring.py` (equals STG-NF's own scoring); `tests/test_pose_score_dir.py` (frame placement 12 vs 23, causal smoothing, coverage) |
| 7 | Frame-level AUC; percentile bootstrap over videos; paired bootstrap of the AUC difference between two arms | `src/edgelab/metrics/scores.py:frame_level_auc`, `bootstrap_ci_over_videos`, `paired_bootstrap_diff`; `score_dir.py:score_pose_dir` | `tests/test_metrics.py` (vs sklearn/scipy; hand-computed paired case) |

## Commands

```
cd experiments
uv run --extra pose python -m edgelab.pose.detcache --frames-root <testing/frames> --level L2 --enhance E1 --out <cache_dir> [--device mps|cpu] [--threads N]
uv run --extra pose python -m edgelab.pose.replay   --cache <cache_dir> --c 0.25 --out <pose_json_dir> [--tracker botsort|bytetrack]
uv run --extra pose python -m edgelab.pose.score_dir --pose-dir <pose_json_dir> --out-dir <scores_dir> --level L2 --enhance E1 --c 0.25 --tracker botsort --device cpu
```

Not yet exercised: steps 5–7 on real data (frames and STG-NF pose data were not available when this was written);
only the pieces around STG-NF are tested, with synthetic inputs.
