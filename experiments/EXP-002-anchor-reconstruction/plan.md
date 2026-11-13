# EXP-002 — Anchor reconstruction: Salman et al. 2026 CNN–GRU, retrained on public data

> Status: **v2, 2026-11-03; executed (Hockey) — see results.md**. This replaces the 2026-11-02 draft (see `SUPERSEDED-NOTE.md`; git history has the old text).
> The good parts of the old draft are kept: the Table 3 rebuild, the video-level 70/15/15 split, ONNX parity and the
> latency probe. The old "VM parity" gate moved to a later experiment, because D8 says to train on the Mac and run timings in the VM.
> Machine: MacBook Pro M4 Pro 48 GB. Training uses PyTorch MPS (D8). Evaluation also uses ONNX Runtime CPU. Mac timings are **comparative only**.
> The owner's approval was given in the task brief. Any later change to the protocol is logged in `log.md` as a deviation.

## 1. Background (plain language)

Salman et al. (Sci. Rep. 2026, doi:10.1038/s41598-026-44939-x) describe a small video model: a 3-layer CNN looks at each
frame, and a GRU (a small recurrent network) reads the sequence of frame features and outputs a class. They report ≈ 95–98 %
accuracy on public violence datasets and real-time speed on a Raspberry Pi 5. They released no code, no weights and no
dataset. Our thesis wants a frozen anchor model of this kind for later preprocessing and OS-tuning experiments. Before we
trust it, we must check whether the published accuracy can be reached when we rebuild the model from the paper text alone.

Two terms used below:
- **Video-level split.** Every clip from one video goes to exactly one of train, val or test. This is the honest protocol.
- **Leaky split.** Clips (or frames) are shuffled into train and test without regard to which video they came from.
  Near-identical neighbouring clips of the same video then sit on both sides. Test accuracy measures memorisation of the
  scene, not generalisation. This is a common error in small-video papers.

## 2. Problem

The paper's evaluation cannot be checked from the text:
- The architecture is under-specified (no clip length, no strides).
- The splits are described inconsistently.
- Every Table 9 confusion matrix sums to exactly 1000 samples.

We do not know whether ~97 % on Hockey Fight reflects the model or the protocol.

## 3. Objective

1. Rebuild Table 3 as faithfully as the text allows, and state every assumption.
2. Train it with the paper's hyperparameters on public datasets, using **video-level** splits.
   Report accuracy with 95 % bootstrap CIs, and compare with Table 9.
3. Measure how much a **leaky clip-level split** inflates accuracy (the leakage hypothesis).
4. Export to ONNX. Check PyTorch↔ORT parity. Measure ORT-CPU latency on the Mac (comparative only).
5. Write down the ShanghaiTech / UCF-Crime frame-AUC protocol. Run it once the data is on disk.

## 4. What the paper says (verbatim extracts, fetched 2026-11-03 from nature.com)

Section names follow the article. Tables were fetched from `/tables/N`.

**Architecture — Table 3** ("Summary of the proposed model architecture and layer configuration"):

| Layer type | Filter / units | Kernel | Activation | Output shape |
|---|---|---|---|---|
| Input layer | – | 256 × 256 × 3 | – | 256 × 256 × 3 |
| Convolution + BatchNorm | 32 | 3 × 3 | ReLU | 128 × 128 × 32 |
| Depthwise separable conv | 64 | 3 × 3 | ReLU | 64 × 64 × 64 |
| Max pooling | – | 2 × 2 | – | 32 × 32 × 64 |
| Convolution + BatchNorm | 128 | 3 × 3 | ReLU | 32 × 32 × 128 |
| Global average pooling | – | – | – | 1 × 128 |
| GRU layer | 128 units | – | tanh | 1 × 128 |
| Dropout | 0.4 | – | – | 1 × 128 |
| Dense (softmax) | 6 classes | – | Softmax | 1 × 6 |

- §Model architecture: "the architecture incorporates batch normalization and dropout layers after each convolutional
  block"; "A global average pooling layer is employed before the dense classification head"; "The final fully
  connected layer uses a softmax activation function to predict one of six predefined classes".
- §Model architecture: "Adam optimizer with an adaptive learning rate scheduler and categorical cross-entropy loss."
- §Ablation study: "The sequence length (fixed short temporal window), frame sampling rate (25 fps with temporal
  skipping) … were selected based on empirical stability". The CNN-only baseline "received a short temporal window of
  consecutive frames stacked along the channel dimension (e.g., T = 3 frames, input size: 256 × 256 × 9)".

**Preprocessing — §Data preprocessing and augmentation, Table 2:**
- "decomposed into individual frames at a uniform sampling rate of 25 frames per second"; "each frame was resized to
  256 × 256 pixels … pixel intensity values scaled to the range [0, 1]".
- Augmentation (Table 2):
  - "Horizontal / vertical Flip | Probability = 0.5"
  - "Random rotation | ± 15°"
  - "Brightness / contrast adjust | ± 20%"
  - "Gaussian noise injection | σ = 0.01"
  - "Temporal frame skipping | Every 3rd frame"
  - "Class balancing | Random oversampling of minority classes"
- §Data preprocessing: "temporal frame skipping was applied, selecting every third frame in long sequences".

**Training — §Training and validation strategy, Table 4:**
- Hyperparameters:
  - "Dataset split | Train: 70%, val: 15%, test: 15%"
  - "Optimizer | Adam"
  - "Learning rate | 1 × 10⁻⁴ (with cosine annealing)"
  - "Loss function | Categorical cross-entropy"
  - "Batch size | 16"
  - "Epochs | 50"
  - "Dropout rate | 0.4"
  - "Early stopping patience | 10" ("monitoring the validation loss")
- "The best-performing weights, determined based on validation accuracy, were preserved for final testing".
- §Experimental setup: "Each experiment was repeated three times, and average performance metrics were reported".

**Splits (contradictory):**
- §Experimental setup: "official train/test splits defined in the original dataset publications were strictly followed
  where available (Hockey Fight, Violent Flow, ShanghaiTech, UCF-Crime)". Note: Hockey Fight (Bermejo Nievas et al. 2011)
  defines no official train/test split; its paper used 5-fold cross-validation.
- "The proposed model was trained independently for each dataset using its corresponding training subset".
- Table 8: "Train/val/test split | 70% / 15% / 15%".
- §Quantitative performance evaluation: "All metrics were computed at the frame level and aggregated at the video level";
  for UCF-Crime "a 13-class subset … confusion statistics were computed per class using a one-vs-rest strategy and then
  summed".

**Reported numbers we compare with — Table 9:**

| Dataset | TP | FP | FN | TN | Acc % | F1 % | ROC–AUC |
|---|---|---|---|---|---|---|---|
| Hockey Fight | 482 | 12 | 18 | 488 | 97.6 | 97.0 | 0.982 |
| RLVS | 460 | 24 | 26 | 490 | 95.1 | 94.3 | 0.957 |
| Violent Flows | 472 | 16 | 22 | 490 | 96.4 | 95.8 | 0.965 |
| ShanghaiTech | 448 | 28 | 30 | 494 | 94.8 | 94.0 | 0.951 |
| UCF-Crime | 455 | 21 | 25 | 499 | 95.5 | 94.8 | 0.958 |
| Custom Six-Class | 485 | 15 | 20 | 480 | 96.9 | 96.3 | 0.972 |

- Our observation: Hockey has exactly 1000 videos (500 fight / 500 non-fight), and its row has TP+FN = 500 and
  FP+TN = 500. That is the **whole dataset**, not a 15 % test split (which would be 150).
- From these confusion counts, Hockey recall = 482/500 = 96.4 %, not the 97.2 % printed. Precision = 482/494 = 97.6 %,
  not 96.8 %. Accuracy = (482+488)/1000 = 97.0 %, not the 97.6 % printed. So the headline number we compare against
  cannot itself be recomputed from its own matrix. We compare against the printed 97.6 % and also against 97.0 %.

**Latency — Table 12 (Pi 5):** FP32 67.2 ms/frame, 14.9 FPS. FP16 48.5 ms. INT8 38.4 ms.
- Table 1 gives Hockey resolution "320 × 240". The Kaggle copy we use is 360 × 288 at 25 fps, with 41 frames per video (logged).

**Data availability:** the custom six-class dataset is "available from the corresponding author upon reasonable request".
- A polite request draft for the owner is in `request_email_draft.md`. Nothing has been sent.

## 5. Method

### 5.1 Model — `src/edgelab/models/salman_cnn_gru.py`

The model is a per-frame CNN, shared across T frames, followed by a GRU over time. The shapes follow Table 3.

**Reconstruction assumptions** (none of these are given in the paper):

| # | Assumption | Why |
|---|---|---|
| A1 | Conv1 3×3, **stride 2**, padding 1 | Table 3 output 128×128 from 256×256 |
| A2 | DW-sep: depthwise 3×3 **stride 2** + BN + ReLU, then pointwise 1×1 → 64 + BN + ReLU (MobileNet-v1 style) | output 64×64; text says BN after each conv block |
| A3 | Max-pool 2×2 stride 2 | 64→32 |
| A4 | Conv3 3×3 stride 1, padding 1, + BN + ReLU | output 32×32×128 |
| A5 | No dropout inside the CNN; one dropout (0.4) after the GRU, as in Table 3 | the text's "dropout after each convolutional block" gives no rate; Table 3 places a single dropout layer |
| A6 | GRU: 1 layer, input 128, hidden 128; classify the **last** hidden state | Table 3 GRU output "1 × 128" |
| A7 | Dense → C logits. Softmax is applied in the loss (CE) and in the exported ONNX graph | numerically equivalent |
| A8 | C = 2 (violent / non-violent) for the binary benchmarks; Table 3 has 6 only for the custom set | the binary datasets |
| A9 | **Clip = T = 8 frames** taken from the every-3rd-frame stream (8 × 3 / 25 fps ≈ 0.96 s) | the paper only says "fixed short temporal window". Hockey videos have 41 frames, so every 3rd frame gives 14; T = 8 allows several clips per video |

Hand estimate before coding:
- ≈ 0.18 M parameters (conv1 0.9 k, DW-sep 2.6 k, conv3 74 k, GRU 99 k, dense 0.3 k)
- ≈ 0.099 GMAC/frame (conv1 14.2 M, DW 1.2 M, PW 8.4 M, conv3 75.5 M; the GRU is negligible)

This matches our earlier 0.1 GMAC / 0.2 M estimate.

### 5.2 Data

- **Decode/cache:** `src/edgelab/data/video_clips.py`.
  - `ffmpeg -vf fps=25,scale=256:256:flags=area` decodes to RGB uint8. We keep every 3rd frame.
  - One `.npy` per video (shape N×256×256×3, uint8) goes under `thesis-data/cache/<dataset>/`, with an index CSV
    (video id, label, frame count, source sha256 prefix).
  - Aspect ratio is not preserved (a plain resize, as in the paper).
- **Clips:** sliding windows of T = 8 cached frames, window stride 2.
  - Hockey: 14 frames → starts 0, 2, 4, 6 → 4 clips per video.
  - Longer videos are capped at 8 clips, evenly spaced. This keeps RLVS from being dominated by long videos.
- **Splits:** `src/edgelab/data/splits.py`. Each split is saved as JSON under `EXP-002-anchor-reconstruction/splits/`.
  - **video-level (primary):** stratified by label, 70/15/15, seed s. A test asserts zero video overlap.
  - **clip-level leaky (ablation, clearly labelled):** the same clip items are shuffled and split 70/15/15 at clip
    level, stratified by label, seed s. Clips of one video land in several partitions. A test asserts that overlap
    exists, so the ablation really is leaky.
  - Hockey has no official split (see §4), so we use our own seeded split. RLVS has none either.
- **Augmentation** (training only, on device, one random draw per clip, applied identically to every frame of the clip):
  - h-flip p = 0.5 and v-flip p = 0.5
  - rotation U(−15°, 15°)
  - brightness ×U(0.8, 1.2) and contrast ×U(0.8, 1.2) around the mean
  - Gaussian noise σ = 0.01
  - clamp to [0, 1]

  Class oversampling is a no-op: Hockey and RLVS are balanced.

### 5.3 Training — `src/edgelab/models/train_clips.py`

- Settings:
  - Adam, lr 1e-4
  - CosineAnnealingLR (T_max = 50 epochs)
  - batch 16
  - at most 50 epochs
  - CE loss
  - early stopping on val loss, patience 10
  - keep the checkpoint with the best **val accuracy** (as in the paper)
- Device MPS, fp32.
- Seeds {0, 1, 2}. The seed sets both the split and the initialisation.
- Every run is wrapped in `tracked_run(experiment="EXP-002")`. Each epoch logs train/val loss/acc and seconds/epoch.
- **Test evaluation:**
  - clip-level accuracy, F1 and ROC-AUC on test clips
  - video-level accuracy (mean clip probability per video, threshold 0.5)
  - 95 % bootstrap CI over **test videos** (2000 resamples, `bootstrap_ci_over_videos` with an accuracy metric)

### 5.4 ONNX + latency — `src/edgelab/models/onnx_export.py`

- Export the FP32 model with a softmax head. Input `[1, 8, 3, 256, 256]`, opset 17.
- Parity: 50 test clips, PyTorch CPU vs ORT CPU.
- Latency: ORT CPU EP, intra-op threads ∈ {1, 2, 4}, 10 warm-ups + 100 timed runs. Report p50/p95 per clip and per frame
  (= clip / 8). Tag: "Mac, comparative only".

### 5.5 ShanghaiTech / UCF-Crime protocol (run only when the data is present)

- **Data:**
  - ShanghaiTech Campus, weakly-supervised split of Zhong et al. 2019: 238 train / 199 test videos, video-level labels
    for training, frame-level ground truth for test.
  - UCF-Crime: the official 1610 train / 290 test videos, video-level train labels, the 290 temporally annotated test
    videos.
- **Training (supervised with weak labels, simplest honest variant):** every clip of a training video inherits the video
  label (normal = 0, anomalous = 1). The model is trained as above with CE.
  - This is noisy: most clips of anomalous videos are normal. It is a known weak baseline, and it is roughly what a
    supervised paper must have done.
  - MIL ranking (Sultani 2018) would be the stronger choice. It is **not** what the paper describes, so it is out of
    scope here.
  - The validation set is 15 % of the train videos (video level).
- **Scoring:**
  - Sliding T = 8 clips, stride 1 over the every-3rd-frame stream.
  - Each clip's P(anomalous) is assigned to the original frames it covers. When several clips cover a frame, their scores
    are averaged.
  - Frames that no clip covers take the nearest score.
  - The frame-level ROC-AUC is computed over all test frames (`frame_level_auc`), with a bootstrap CI over videos.
- **Compare with Table 9 accuracy?** Not directly comparable. We report AUC (D2) and note the mismatch.
- **Status:** TODO.
  - `thesis-data/shanghaitech/` is empty on 2026-11-03.
  - The UCF-Crime download (another agent) is not marked complete.

## 6. Pre-registered expectations and gates

| ID | Statement | How it is judged |
|---|---|---|
| G-model | Parameter count and MACs/frame lie within ×2 of the hand estimate (0.18 M, 0.099 GMAC) | unit test; if far off, re-read Table 3 before training |
| G-split | 0 videos shared between the train/val/test of the video-level split; > 0 shared for the leaky split | unit tests |
| G-parity | ORT CPU vs PyTorch on 50 clips: max \|Δprob\| ≤ 1e-4, 0 label flips | script output in MLflow |
| H-A | Hockey video-level accuracy is reported with a 95 % bootstrap CI and compared with 97.6 % | **no pass/fail**; the reproduction verdict is the result. "Reproduced" if 97.6 lies inside the CI of the seed-mean; "not reproduced" if it lies above the CI |
| H-B | The leaky clip-level split gives higher test accuracy than the video-level split (same seeds, same clip pool) | report the gap per seed and its mean. Expected direction: leaky > clean. A gap ≤ 1 pp is "no detectable inflation" on Hockey |
| L | Our measured FP32 latency per frame vs the paper's 67.2 ms (Pi 5) | discussion only; the Mac is not the Pi |

Expected runtime (before measuring): 1–3 min/epoch, 0.5–2 h per run. It will be replaced by the measurement.

## 7. Deviations policy

- Any change to T, the stride, the split rule, the metrics or the hyperparameters after the first training run is logged in
  `log.md` with a reason. If the change affects comparability, it is also added to DECISIONS.md.
- If a run fails or is stopped early (time budget ≈ 2 h wall for this session), that is recorded as is. Partial results are
  marked partial.
- No numbers are tuned toward the paper's values. The paper's protocol is followed where it is stated, and our assumption
  is used where it is not.
