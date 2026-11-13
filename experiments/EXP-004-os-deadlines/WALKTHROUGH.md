# EXP-004 walkthrough — from a timing trace to a delivered AUC

Plain-language map of the code. Each step names the file, the function, and the test that guards it.
Idea: time is measured once, in the VM, on a small subset (how long does one frame take?). Accuracy is NOT re-measured
in the VM: the EXP-003 detection cache is replayed on the host with the frames a live camera box would have skipped.

```
 [1] VM: per-frame timing trace      read jpg → darken → enhance → YOLO (ONNX) → BoT-SORT → STG-NF   → one CSV per run
        │   (12 videos x 240 frames)    each stage timed; OS config (threads, spin, pinning, FIFO) applied in-process
        ▼
 [2] host: service times             CSV rows → service_ms per frame, pooled over repetitions (24-frame blocks)
        │
 [3] deadline sim                    camera at 24 fps, one worker, newest frame wins → keep_mask per video
        │
 [4] replay with the mask            EXP-003 cache → only kept frames reach the tracker → pose JSON with frames
        │                             renumbered 0..K-1 (+ map processed index → real frame)
 [5] STG-NF                          unchanged frozen checkpoint, on the renumbered folder
        │
 [6] delivered scores                score of processed frame k is held on the real frames until the next processed frame;
        │                             causal half-Gaussian over the REAL frames; frames with no output = most normal
        ▼
 [7] delivered AUC (live variant) + bootstrap CI over videos        → MLflow experiment EXP-004
```

| Step | What happens | Code | Guarded by |
|---|---|---|---|
| 0 | Timing subset: the 6 scenes with most test videos (ties → lower scene id), 2 videos each (seed 0, videos with < 240 frames not eligible), first 240 frames. Frames are copied into `vm-exchange/frames/` because the VM cannot see the host's frames | `pipeline/timing_subset.py:select_subset`, `copy_frames`; list in `timing_subset.txt` | `tests/test_pipeline_live_timing.py` (`test_select_subset_*`) |
| 1a | YOLOv8n-pose ONNX through ONNX Runtime CPU. Own letterbox to 640×640, decode of the (1, 56, 8400) output, conf filter at `c`, NMS (OpenCV, IoU 0.7). Threads and spinning set on the session | `pipeline/ort_pose.py:make_session`, `letterbox`, `decode`, `OrtPose` | `tests/test_pipeline_ort_pose.py` (padding geometry, coordinates back to the original image, NMS keeps the best, empty result); parity vs Ultralytics: `ort_pose.py` CLI (numbers in `log.md`) |
| 1b | STG-NF online: last 24 poses per track; every frame, each track that completes a continuous window (≤ 2 missing slots, STG-NF's rule) is scored; new windows of one frame are scored as one batch | `pipeline/stgnf_live.py:LiveStgnf.update`, `score_segments`, `load_model`, `is_continuous` | `tests/test_pipeline_live_timing.py::test_live_stgnf_scores_equal_the_offline_scores` (same numbers as `stgnf_eval.score_segments`), `test_continuity_rule_allows_two_missing_slots`, `test_live_stgnf_batches_*` |
| 1c | OS knobs inside the process: `--threads` (ORT, torch, OpenCV), `--spin on/off`, `--affinity 1,2,3`, `--fifo 10` (affinity and FIFO are set on every existing thread, then again after warm-up; `<out>.meta.json` records what the kernel applied) | `pipeline/os_config.py:apply_os_config`, `thread_report`; `live_timing.py:main` | `tests/test_pipeline_live_timing.py::test_parse_cpus`; real effect checked in the VM smoke (see `log.md`) |
| 1d | The per-frame loop and the CSV (read, darken, enhance, detect pre/infer/post, track, flow, `service_ms = total − darken`, n_dets, n_tracks, n_new_segments) | `pipeline/live_timing.py:run_video`, `main` | `tests/test_pipeline_live_timing.py::test_live_timing_writes_one_csv_row_per_frame_with_all_columns` |
| 1e | Co-load (ffmpeg x264 of a 720p test pattern at real time), optional `--pin` and `--nice` | `EXP-004-os-deadlines/scripts/coload.sh` | tried by hand in the VM (see `log.md`) |
| 2 | Pool the repetitions' service times; each video is cut to whole 24-frame blocks first | `pipeline/deadline_replay.py:load_trace` | `tests/test_pipeline_deadline_replay.py::test_load_trace_cuts_each_video_to_whole_blocks` |
| 3 | Block bootstrap of service times (seeded, aligned 24-frame blocks), then the latest-frame simulation. Per-video seed = SHA-256 of `seed|video` | `pipeline/deadline_sim.py:block_bootstrap_service`, `simulate_latest_frame`, `keep_masks_for_videos`, `video_seed` | `tests/test_pipeline_deadline_sim.py` (all kept when fast; 2.5 periods → kept frames 0, 2, 5, 7, 10, 12 by hand; superseded frames dropped; idle worker waits; seeded and order independent) |
| 4 | `replay_video(..., keep_mask, reindex=True)`: pose frame ids become 0..K-1 and `stats["frame_map"][k]` = real frame of processed frame k; `replay_dir` saves the maps to `frame_maps.npz`. Default (`reindex=False`) is unchanged | `pose/replay.py:replay_video`, `replay_dir`, `FRAME_MAPS_FILE` | `tests/test_pose_delivered.py::test_replay_reindex_*`, `test_replay_dir_reindex_*`; old behaviour: `tests/test_pose_detcache_replay.py` |
| 5 | STG-NF test pass on the renumbered folder | `pose/stgnf_eval.py:score_segments` (unchanged) | EXP-001 / EXP-003 tests |
| 6 | Segment score → processed frame `start + 23` (live variant), least-normal person; hold until the next processed frame; frames with no output or nobody → dataset's most normal score; causal half-Gaussian on real frames | `pose/delivered.py:processed_normality`, `hold_last`, `delivered_variant_scores` | `tests/test_pose_delivered.py` (hold values, before-first-output, least-normal person, causal: later changes never alter earlier scores, perfect detector → AUC 1) |
| 7 | Whole cell: masks → replay → STG-NF → delivered AUC → bootstrap CI over videos → MLflow `EXP-004`; saves `keep_masks.npz`, `delivered_scores_live.npz` (names, score_*, label_*: usable with `paired_bootstrap_diff`), `result.json` | `pipeline/deadline_replay.py:run_cell`, `main` | `tests/test_pipeline_deadline_replay.py::test_run_cell_end_to_end_on_synthetic_videos` (drop rate 0.6 for 2.5-period service, one score per real frame, segments = 36 per video after re-indexing) |

## Decisions made while implementing (not in the frozen plan; the lead should confirm)

- **service_ms excludes darkening.** Darkening only simulates the dark sensor; reading the jpg stays in (stands in for fetching the camera frame). `darken_ms` and `total_ms` are in the CSV, so either choice can be replayed (`--column` is not yet a CLI option; `load_trace(column=...)` takes it).
- **No output latency.** A processed frame's score is attached to the real frame it came from and held from there; the time the worker needed to finish is not added. Latency would only shift the delivered series later by about one service time.
- **Smoothing is on the real-frame series** after hold (time constant stays in real seconds), not on the processed frames.
- **Static 640×640 ONNX vs the cache.** The EXP-003 cache was made with the `.pt` weights, where Ultralytics letterboxes to 640×384 (rectangular); the VM runs the fixed 640×640 ONNX (more compute, slightly different confidences). See the parity numbers in `log.md`.

## Commands

```
# 1. copy the timing subset's frames next to the VM (host)
cd experiments && uv run --extra pose python -m edgelab.pipeline.timing_subset \
    --frames-root ~/thesis-data/shanghaitech/shanghaitech/testing/frames --copy-to ~/thesis-data/vm-exchange/frames

# 2. one timing cell inside the VM (O1 + co-load; for O0 drop --threads/--spin/--affinity/--fifo and the coload options)
limactl shell edge-os -- bash -c '
  E=/Users/nanh/Documents/RMIT_MscAI/thesis/experiments; X=/Users/nanh/thesis-data/vm-exchange
  bash $E/EXP-004-os-deadlines/scripts/coload.sh start --pin 0 --nice 19
  sudo env PYTHONPATH=$E/src EDGELAB_MACHINE=vm $HOME/edgelab-vm/.venv/bin/python -m edgelab.pipeline.live_timing \
    --frames-root $X/frames --videos-file $E/EXP-004-os-deadlines/timing_subset.txt --level L3 --enhance E0 --c 0.25 \
    --threads 3 --spin off --affinity 1,2,3 --fifo 10 --out $X/results/L3_E0_O1_rep0.csv --tag L3-E0-O1-rep0
  bash $E/EXP-004-os-deadlines/scripts/coload.sh stop'

# 3. deadline replay of one config (host)
cd experiments && uv run --extra pose python -m edgelab.pipeline.deadline_replay \
    --cache ~/thesis-data/exp003/cache/L3_E0 --trace ~/thesis-data/vm-exchange/results/L3_E0_O1_rep*.csv \
    --c 0.25 --out ~/thesis-data/exp004/L3_E0_O1_B4_c0.25 --label O1-B4 --level L3 --enhance E0
```

## Amendment 1 additions (timing matrix)

| Piece | Code | Guarded by |
|---|---|---|
| Static rectangular ONNX per R (shape read from a real Ultralytics predict) | `pipeline/export_onnx.py`; letterbox to the model's own (h, w): `ort_pose.py:letterbox`, `OrtPose.shape` | `tests/test_pipeline_ort_pose.py::test_letterbox_to_the_rectangles_*`; parity vs `.pt` in `log.md` 10:40 |
| `--imgsz` picks the ONNX; `--max-frames-per-video` | `pipeline/live_timing.py:yolo_onnx_for`, `main` | `tests/test_pipeline_live_timing.py::test_live_timing_writes_one_csv_row_*` |
| B and O as cgroups (slice with `CPUQuota=200%` shared by pipeline + co-load; co-load scope `AllowedCPUs=0`, nice 19) | `scripts/vm_scopes.sh`, `scripts/cell_in_vm.sh`; check: `scripts/scope_check.sh` | numbers in `log.md` 10:55 |
| Host load outside the VM (`top`, minus the edge-os VM process) | `pipeline/host_load.py` | `tests/test_pipeline_timing_matrix.py` (parse, subtraction, means) |
| Cells, frozen random order (one permutation per repetition, seed 0, `timing_plan.csv`), exclusion > 30%, resume, dry run | `pipeline/timing_matrix.py`; launcher `scripts/run_matrix.sh` | `tests/test_pipeline_timing_matrix.py` |

Launch: `tmux new-session -d -s exp004-matrix 'bash /Users/nanh/Documents/RMIT_MscAI/thesis/experiments/EXP-004-os-deadlines/scripts/run_matrix.sh <G-xfit c>'`;
look at `vm-exchange/results/matrix/{runner.log,manifest.csv}`. A killed runner resumes where it stopped (re-running the same command).
Excluded runs (host load) are kept in the manifest with `excluded=True`; failed runs are retried on the next launch.
