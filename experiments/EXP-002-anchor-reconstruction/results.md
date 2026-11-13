# EXP-002 results: anchor reconstruction (Hockey Fight)

> Session 2026-11-03, 22:10–23:45. Mac M4 Pro, PyTorch 2.15 MPS (training), ORT 1.31.1 CPU (inference).
> Every number below comes from MLflow (experiment `EXP-002`, `mlflow.db`) or a file under `artifacts/EXP-002/`. Tables
> are pasted from `uv run python -m edgelab.models.summarize_runs --dataset hockey-fight`.
> **Status: partial.** Hockey is complete (3 seeds × 2 protocols). RLVS, ShanghaiTech and UCF-Crime were not run (see Next steps).

## 1. Results

### 1.1 Model size (gate G-model): PASS
- Parameters: **177,026**
- MACs/frame: **0.0993 G**

Both match the hand estimate (0.18 M, 0.099 G) from `tests/test_models_salman.py`. ONNX FP32 file: 712,828 bytes.

### 1.2 Hockey Fight test accuracy (T = 8, every 3rd frame, paper hyperparameters)

The CI is a 95 % percentile bootstrap over test videos (2000 resamples). "Video acc" = mean clip probability per video ≥ 0.5.

| protocol | seed | run id | video acc [95 % CI] | clip acc [95 % CI] | clip F1 | clip AUC | epochs | best epoch |
|---|---|---|---|---|---|---|---|---|
| video-level | 0 | `b54dec0434564422bbea2ea435ca3b14` | 0.940 [0.900, 0.973] | 0.938 [0.898, 0.973] | 0.938 | 0.983 | 16 | 13 |
| video-level | 1 | `74e1f195598841acbe4070194b745b22` | 0.940 [0.900, 0.973] | 0.939 [0.899, 0.973] | 0.938 | 0.982 | 33 | 22 |
| video-level | 2 | `c50ba48dd8c141ec95f9308563fcfa65` | 0.933 [0.887, 0.967] | 0.938 [0.900, 0.970] | 0.939 | 0.987 | 37 | 18 |
| leaky clip-level | 0 | `c686ecf138ff43288ff32e2b7fee31b3` | 0.950 [0.929, 0.969] | 0.950 [0.928, 0.969] | 0.949 | 0.991 | 50 | 31 |
| leaky clip-level | 1 | `9a97d6fd6476405b9663ec2c20677729` | 0.936 [0.913, 0.957] | 0.930 [0.904, 0.956] | 0.930 | 0.977 | 17 | 6 |
| leaky clip-level | 2 | `7a2b09ea44ff4d6dbf6f31d0df347ecd` | 0.951 [0.931, 0.969] | 0.953 [0.933, 0.971] | 0.954 | 0.993 | 50 | 45 |

Each video-level test set holds 150 videos (≈ 600 clips). Each leaky test set holds 601 clips drawn from 469–491 different videos.

- **Video-level mean (3 seeds):** video acc 0.938, clip acc 0.9384 (sd 0.0001).
- **Paper (Table 9, Hockey):** 97.6 % printed, or 97.0 % recomputed from its own confusion matrix.

### 1.3 H-A verdict: **not reproduced**
- For every seed, 97.6 % lies **above** the upper 95 % CI bound (0.973, 0.973, 0.967).
- The recomputed 97.0 % lies inside the CIs of seeds 0 and 1 only, near their upper edge.
- Our honest-protocol estimate is ≈ 93.8–94.0 %, about 3.6 pp below the printed number.

### 1.4 H-B, leakage ablation: **no detectable inflation** under the pre-registered rule
- Gap (leaky − video-level, clip accuracy) per seed: **+1.17, −0.84, +1.51 pp**. Mean **+0.61 pp**.
- The pre-registered threshold was ≤ 1 pp, so "no detectable inflation" on Hockey with our clip construction.
- The direction is mostly positive.
- The two leaky runs with the largest gains are the two that never early-stopped (50 epochs).

### 1.5 POST-HOC (not pre-registered): whole-dataset evaluation of the video-level checkpoints

| seed | run id | all 1000 videos | train videos (700) | test videos (150) |
|---|---|---|---|---|
| 0 | `19ec6a360cf946f1a5edaad5cefd377d` | 0.952 | 0.956 | 0.940 |
| 1 | `4d94c338714c4b2a9796179f77762a03` | 0.948 | 0.954 | 0.940 |
| 2 | `3bfd317478394d46afa68f395b859138` | 0.957 | 0.960 | 0.933 |

These are video accuracies.

### 1.6 ONNX parity (gate G-parity): PASS
- MLflow `9a58548ac5ad414282c5b91b7764c0c0`
- 50 test clips (seed-0 video model): max |Δprob| PyTorch-CPU vs ORT-CPU = **1.37e-6**, **0** label flips

### 1.7 ORT CPU latency, FP32. Mac, comparative only, **measured under heavy background load** (see Threats)

The input is one clip of 8 frames, 10 warm-ups + 100 runs. MLflow `9a58548ac5ad414282c5b91b7764c0c0`.

| intra-op threads | clip p50 (ms) | clip p95 (ms) | per frame p50 (ms) |
|---|---|---|---|
| 1 | 25.4 | 34.7 | 3.17 |
| 2 | 12.2 | 13.3 | 1.53 |
| 4 | 6.6 | 8.5 | 0.83 |

### 1.8 Training time (replaces the 1–3 min/epoch, 0.5–2 h/run estimate)
- **Standalone:** 32.2 s/epoch on MPS, for 2,794 train + 602 val Hockey clips at batch 16 (smoke run
  `378fdef9014f45c7ab36295b47e6d697`).
- **With 3 runs in parallel:** 42–90 s/epoch each, ≈ 1.4× aggregate throughput.
- **With 6 runs in parallel:** 150–270 s/epoch, a worse aggregate. Don't do this.
- **Run length:** video-level runs early-stopped at 16–37 epochs, ≈ 9–20 min standalone. A full 50-epoch run is ≈ 27 min
  standalone.
- **RLVS:** epoch 0 took 393 s with 4 processes sharing MPS. The estimate is ≈ 2 min/epoch standalone (3.9× the Hockey
  clips), so ≈ 0.5–1.7 h per run.
- **Data prep:** Hockey cache 10 s (2.6 GB); RLVS cache 2 min (16 GB).

## 2. Observations
1. The rebuilt model learns quickly. After one epoch the val accuracy is already ≈ 0.88–0.90. The val loss bottoms out
   around epochs 5–27.
2. The three video-level seeds agree closely: clip acc 0.938–0.939, AUC 0.982–0.987. Each uses a different split and a
   different initialisation.
3. With early stopping and the paper's augmentation, the model never fits its training set well: training-video
   accuracy is 95.4–96.0 %. So even a "test on everything" protocol (§1.5) gives at most 95.7 %, still below 97.6 %.
4. In the leaky runs, the val loss kept falling (down to 0.11 vs 0.14–0.20 for video-level), so 2 of the 3 runs went to
   all 50 epochs. This is the signature expected when validation clips are near-copies of training clips. However, the
   resulting test gain is small.

## 3. Discussion

**Does the reconstruction reach the paper's numbers?** No. On Hockey Fight, with an honest video-level split, the
Table 3 model as we read it reaches ≈ 94 % (CI upper bounds ≤ 97.3 %), against 97.6 % printed. That is about 3–4 pp lower.

Two readings are possible, and this experiment cannot separate them:
- **(a)** Our reconstruction differs from theirs. Candidates: the clip length T, the strides, the dropout placement, the
  augmentation strength, or the absence of their pruning/fine-tuning.
- **(b)** Their number comes from a different evaluation. Candidates: a whole-dataset test (the 1000-sample matrices),
  frame-level scoring, or a reporting error. Their printed accuracy does not even match their own confusion matrix
  (97.0 vs 97.6).

**What does the leakage ablation show?** Our specific leaky protocol (overlapping 8-frame clips, 4 per video, shuffled
across partitions) inflated accuracy by only ≈ 0.6 pp on average. That is below our pre-registered 1 pp threshold. So
"clip-level leakage" of this mild kind does not explain a 3–4 pp gap on Hockey.

It does not rule out stronger leakage. Examples:
- frame-level splitting at 25 fps (many more near-duplicates)
- testing on training videos
- leakage in combination with no early stopping

The post-hoc whole-dataset evaluation also falls short (≤ 95.7 %). So "they tested on everything" alone does not explain
97.6 % either, unless their model fits the training data much more tightly than ours does (for example, no augmentation
at evaluation, longer training, or a bigger effective T).

**Is the latency consistent with their 67 ms FP32 claim?**
- Our FP32 ORT model needs 3.2 ms/frame on one M4 Pro core and 0.8 ms/frame at 4 threads, even under heavy
  background load.
- Relative to their claim, an inference-only 67.2 ms/frame on a Pi 5 would require the Pi to be ≈ 20× slower than one
  M4 core, or ≈ 80× slower than 4 M4 cores.
- Typical Cortex-A76 vs Apple-core gaps are much smaller than that. This is our expectation, not yet measured: check it on
  the Pi.

This supports the README §2 suspicion: either their deployed model is much larger than Table 3, or their 67 ms
includes capture, resize and display (their §Real-time inference does say the timings are end-to-end). **A Pi 5
measurement of this ONNX file decides it.**

## 4. Threats to validity
- **Reconstruction assumptions A1–A9** (plan §5.1), especially T = 8. The paper's T is unknown. A longer window might help.
- **Dataset copy:**
  - The Kaggle Hockey copy is 360×288, while the paper's Table 1 says 320×240. The frames are resized to 256×256 anyway.
  - The RLVS archive has a duplicated nested folder (identical, deduplicated).
- **Hockey has no official split.** We use our seeded 70/15/15. CIs are wide (150 test videos, ±4 pp).
- **The leaky ablation is one variant.** Clips share 6 of 8 frames with their neighbours, and there are only ≤ 4 clips
  per Hockey video. Stronger leakage (all 25-fps frame offsets, frame-level samples) was not tested.
- **Timing contamination:**
  - Training ran 3–6 processes in parallel, with a SIGSTOP pause for leaky runs. This affects only `epoch_seconds`,
    not the model math.
  - **The latency was measured while other agents' jobs used ≈ 12/14 CPU cores.** Re-measure on an idle machine.
- **MPS non-determinism:** seeds fix the split and the initialisation, but MPS kernels are not bit-deterministic. Reruns
  may differ slightly.
- `git_dirty=True` in every manifest: the code was not yet committed when the runs were made (by instruction: no commit).
- **Best-checkpoint selection:** the checkpoint is chosen on val accuracy, while early stopping watches val loss, as in
  the paper. The best val accuracy can come from an epoch other than the minimum-loss one.

## 5. Next steps
1. **RLVS:** run alone on the GPU (≈ 0.5–1.7 h per seed):
   `uv run python -m edgelab.models.train_clips --dataset rlvs --protocol video --seed 0` (then leaky, then seeds 1 and 2).
   The cache and split file are ready.
2. **Re-measure ORT latency** on an idle Mac, then in the Lima VM (D8), then on the Pi 5. The Pi number is the real test
   of the 67 ms claim.
3. **Sensitivity of A9:** T ∈ {4, 12} on Hockey, 1 seed each. Decide whether the gap to 97.6 depends on T. This needs an
   owner decision because it is a protocol change.
4. **Stronger leakage variant:** frame-level split from the full 25 fps stream. This tests whether *any* leakage reaches 97 %.
5. **ShanghaiTech and UCF-Crime frame-AUC:** follow plan §5.5.
   - ShanghaiTech data now exists (downloaded by EXP-001).
   - UCF-Crime: wait for `STATUS: COMPLETE` (the train split is needed).
6. Owner: send `request_email_draft.md` to the corresponding author.

## 6. Owner sign-off
Pending.
