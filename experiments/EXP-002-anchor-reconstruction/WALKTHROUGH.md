# EXP-002 walkthrough: from a video file to a number

```
raw video (.avi/.mp4)
  │  ffmpeg: resample 25 fps, resize 256×256, RGB uint8; keep every 3rd frame
  ▼
cache/<dataset>/<video_id>.npy  [N,256,256,3]  + index.csv (label, N, source sha)
  │  sliding windows: T=8 cached frames, stride 2, ≤8 per video  → ClipItem(video_id, start, label)
  ▼
split JSON (video-level  |  leaky clip-level ablation)   EXP-002-anchor-reconstruction/splits/
  │  DataLoader (batch 16) → uint8 clip → float [0,1], [B,T,3,H,W] → augmentation on GPU (train only)
  ▼
SalmanCnnGru: per-frame CNN (Table 3) → 128-d per frame → GRU(128) → last state → dropout 0.4 → Linear → logits
  │  cross-entropy loss, Adam 1e-4, cosine LR, ≤50 epochs, early stop on val loss (patience 10), keep best val acc
  ▼
best.pt → test clips → P(violent) per clip → accuracy / F1 / AUC, video accuracy, bootstrap CI over videos → MLflow
  │
  ▼
ONNX FP32 (softmax head) → ORT CPU: parity vs PyTorch on 50 clips, latency at 1/2/4 threads → MLflow
```

## Step by step

| Step | What happens | Code | Guarded by |
|---|---|---|---|
| 1. List videos | Hockey: label from the file name (`fi`=1, `no`=0). RLVS: label from the folder (`Violence`/`NonViolence`) | `edgelab/data/video_clips.py:list_hockey`, `list_rlvs` | the 1000 / 500 counts are checked in log.md |
| 2. Decode once | `ffmpeg -vf fps=25,scale=256:256` gives raw RGB bytes; keep `[::3]` | `video_clips.py:decode_video` | `test_decode_video_keeps_every_third_frame` (a synthetic 30-frame video gives 10 frames) |
| 3. Cache | one `.npy` per video plus `index.csv`; done in parallel processes; skipped if the file already exists | `video_clips.py:build_cache`, `read_index` | — |
| 4. Clips | window starts per video; short videos are padded by repeating the last frame | `video_clips.py:clip_starts`, `make_clip_items`, `ClipDataset` | `test_clip_starts_hockey_and_long_video`, `test_clip_dataset_pads_short_video_and_normalises` |
| 5. Split | **video-level**: shuffle videos per label, 70/15/15. **leaky** (ablation): shuffle clips instead. Saved as JSON | `edgelab/data/splits.py:video_level_split`, `leaky_clip_split`, `save_split` | `test_video_level_split_has_zero_overlap…` (0 shared videos), `test_leaky_split_really_leaks`, `test_split_roundtrip`, stratification test |
| 6. Normalise | uint8 → float / 255, channels moved to the position PyTorch expects | `video_clips.py:to_model_input` | same dataset test |
| 7. Augment | per clip: flips, ±15° rotation, ±20 % brightness and contrast, noise σ 0.01 (Table 2) | `edgelab/models/train_clips.py:augment` | — (visual check is a TODO for the owner) |
| 8. Model | Table 3 layers; the assumptions A1–A9 are in plan.md §5.1 | `edgelab/models/salman_cnn_gru.py:SalmanCnnGru` | `test_encoder_shapes_follow_table3`, `test_output_shape_and_probabilities`, `test_size_within_2x_of_hand_estimate` |
| 9. Train | loop over epochs; val loss/acc each epoch; checkpoint on best val acc; early stop | `train_clips.py:run` | the smoke run in log.md |
| 10. Evaluate | clip probabilities → metrics; bootstrap resamples **videos**, not clips | `train_clips.py:test_metrics`, `edgelab/metrics/scores.py` | `tests/test_metrics.py` (vs sklearn / scipy) |
| 11. Export | wraps the model with softmax and exports with `torch.onnx.export` (opset 17) | `edgelab/models/onnx_export.py:export_onnx` | `tests/test_models_onnx.py` |
| 13. Tables | MLflow → markdown tables in results.md; post-hoc whole-dataset scoring | `edgelab/models/summarize_runs.py`, `edgelab/models/eval_whole_dataset.py` | — |
| 12. Parity + latency | same 50 test clips through PyTorch and ORT; timed ORT runs at 1/2/4 threads | `onnx_export.py:parity`, `latency` | gate G-parity in results.md |

## Commands

```bash
uv run pytest -q
uv run python -m edgelab.models.train_clips --dataset hockey-fight --protocol video --seed 0
uv run python -m edgelab.models.train_clips --dataset hockey-fight --protocol leaky --seed 0
uv run python -m edgelab.models.onnx_export --run hockey-fight-video-seed0
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db     # browse runs
```

The cache is built from a Python one-liner (see log.md, 22:17). Run it again for a new dataset with `list_rlvs`.

## Concepts in one line each
- **GRU**: a recurrent layer that reads the 8 frame-vectors in order, keeps a 128-number memory, and outputs its final memory.
- **Depthwise-separable conv**: a 3×3 filter per channel followed by a 1×1 mix of channels. It is much cheaper than a full 3×3 conv.
- **Cosine annealing**: the learning rate falls from 1e-4 toward 0 along half a cosine curve over 50 epochs.
- **Early stopping**: training stops when the validation loss has not improved for 10 epochs.
- **Bootstrap CI over videos**: re-draw the test videos with replacement 2000 times, recompute accuracy each time, and
  take the 2.5 / 97.5 % quantiles. Videos are resampled because clips of the same video are not independent.
