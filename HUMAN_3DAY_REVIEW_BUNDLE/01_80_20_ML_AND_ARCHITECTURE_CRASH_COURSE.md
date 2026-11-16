# 01 · 80/20 ML + architecture crash course (for a Go engineer)

Rule: generic ML explanations are mine; every **thesis number** is quoted from a repo file (path given). If a number has no path, it is arithmetic I show.

Mental model: treat each ML component as a **pure function with a typed signature** plus one piece of state. Most of the thesis is about *which state you are allowed to keep* and *when a result becomes visible* — exactly the questions you ask about a streaming service.

---

## 1. The pipeline as types

```go
// Pseudocode, not repo code.
type Frame   [H][W][3]uint8                       // 24 fps => 41.67 ms deadline per frame
type Pose    [17][3]float32                       // (x, y, confidence) per COCO keypoint
type Track   struct{ ID int; Frames []int; Poses []Pose }

func Decode()    Frame                            // 0.38 ms/frame on the VM (E02)
func Detect(Frame) []Detection                    // YOLOv8n-pose: box + score + 17 keypoints   <- the expensive stage
func Track(dets []Detection) []Track              // BoT-SORT: who is who across frames
func Window(t Track) [N][2][T][18]float32         // last T frames (24 SHT / 16 UB) in 18-joint layout (x,y)
func Score(w Window) float32                      // normalizing flow: how "normal" is this motion? (-NLL)
func Deliver(score, frameIdx) FrameScore          // causal: a score exists only after its window ends
```

Per-frame cost on the VM (E02): decode 0.38 + pose 11.67 + track 0.42 + score 2.46 = **14.93 ms** (SHT) — but on the real Pi 5 pose alone is ~46–74 ms depending on threads (`Q2-P1-Pi5_suite/report.md` §2: NCNN FP16 73.59 ms at T=1 → 46.43 ms at T=4). That gap is the whole story: **the VM said "fine", the Pi said "collapse".**

---

## 2. CNN → pose: single-stage vs top-down

You know: a CNN turns an image into feature maps; each conv layer has a **stride** (how much it downsamples). A feature map at stride 8 has one cell per 8×8 pixels.

```
input 384x640 (derived) ─ backbone ─► P3 stride 8  : 48 x 80  cells   (small people live here)
                                      P4 stride 16 : 24 x 40
                                      P5 stride 32 : 12 x 20   (large people)
                         every cell predicts: box, score, 17 x (x, y, visibility)
```
Arithmetic: 384/8 = 48, 640/8 = 80 (strides are standard YOLOv8; input size 384×640 is the NCNN letterbox used in `Q2-P1-Pi5_huang_repro/report.md`).

* **Single-stage (YOLOv8n-pose):** one network pass produces every person's box *and* skeleton. Cost doesn't grow with the number of people. Weakness here: tiny people.
* **Top-down (AlphaPose, RTMPose):** a person detector, then crop each person and run a pose net per crop. Better quality, cost grows with crowd size. EXP-015: oracle boxes + RTMPose ≈ 85–86 AUC (`EXP-015-oracle-rtmpose/results.md`), EXP-016: top-down 960 K=1 reaches 83.11 but no configuration meets 24 fps in the VM (`EXP-016-topdown-detector-sweep/results.md`).

**Why small people were invisible (the thesis' biggest "aha"):** the COCO → YOLO label conversion keeps a person only if it has keypoint labels. Distant people in COCO often have no keypoints, so training treated them as *background* — the network learned "small person = nothing". The fix **ML0**: keep those persons as **box-only** targets and *mask* the keypoint loss for them (`pose/coco_pose_labels.py::person_row`, `pose/loss_patch.py::masked_calculate_keypoints_loss`). Same network, same FLOPs (1.00×):

| Metric (`Q2-P1-E01_quality/report.md`) | stock | ML0 |
|---|---|---|
| recall `<48 px` | 0.4% | 37.0% |
| S4 AUC (SHT, ML0 vs stock, paired bootstrap) | 66.03 | 73.78; ML0−stock +7.75 [+4.63, +10.47]* |

(`*` means the 95% CI excludes 0.) EXP-020's pre-registered test says it's the **label**, not the stride/resolution: label fix P1 PASS +16.24 [+11.99, +20.45]; adding a stride-4 P2 head (`MLP`, 1.92× FLOPs) gave only +0.91 [−0.02, +1.81] — not significant — and is 108.71 ms vs 46.43 ms at T=4 on the Pi (`Q2-P1-Pi5_suite/report.md`).

---

## 3. MOT: BoT-SORT vs TrackTrack (the thing between detector and scorer)

Tracker = a per-object **Kalman filter** (predict next box with constant velocity) + **IoU matching** (Hungarian assignment between predicted boxes and new detections). Think: a session table keyed by object, with a "time-to-live" (`track_buffer`).

* **Standard BoT-SORT** only associates detections with confidence ≥ a high threshold; low-confidence boxes are discarded → a person flickering below the threshold for 2 frames loses its track → gaps in the skeleton window.
* **Two-stage association** (what the thesis locks): first match high-confidence boxes (`c = 0.20`), then give *low*-confidence boxes (0.05…0.20) a second chance to **continue an existing track but never start one** (`analysis/q2_p1_e01.py::build_tracker_cell`, test `test_two_stage_low_boxes_continue_a_live_track_but_cannot_start_one`). Locked from a 36-cell dev-only grid: `c=0.20, track_low_thresh=0.05, track_buffer=30, match_thresh=0.85`.
* **Why:** of the anomalous frames still missed after ML0, **84.2% are intermittent gaps inside live tracks**, and 52.3% of those already had a cached low-confidence detection in [0.05, 0.25) (`E01 report`). Result: tracks 2,006 → 1,766 (fewer fragments), S4 76.82 (ML0 alone 73.78); two-stage +3.04 [+1.44, +4.71]*.
* **TrackTrack (CVPR 2025)** was also tested (E07): better than BoT-SORT in 10 of 28 cells, equal in 2, inconclusive in 16, never worse; best pipeline `ML0|TrackTrack|STG-NF` 70.33 (SHT 76.36 / UB 64.30).

---

## 4. CNN → GCN: the skeleton is a graph

A skeleton is not a grid; it's 18 nodes (joints) and edges (bones). A **graph convolution** is a convolution where "neighbours" come from the adjacency matrix: each joint's new feature = weighted sum of its neighbours' features. **ST-GCN** = graph conv over joints + ordinary conv over time (a Conv with kernel `(k, 1)` sliding over frames).

```
input   (N, C, T, V)     N windows, C channels, T=24 frames, V=18 joints
 graph conv:  X' = A_norm · X · W        (mixes joints; kernel has NO temporal footprint)
 temporal conv: kernel (k,1) over T      (k = T//2+1: 13 for SHT, 9 for UB)
```
`models/stgnf_flow.py::CausalSTGCN` (line 59) implements exactly this, with left-only padding (see §6).

---

## 5. Normalizing flows (the scorer) — from zero

Goal: assign every 24-frame pose window **X** a likelihood, trained only on *normal* videos (no anomaly labels). Anomaly = low likelihood.

A flow is an **invertible** function *f* mapping X to a latent Z that should look like a simple Gaussian. Change of variables:

```
log p(X) = log p_Z(f(X)) + log |det J_f(X)|        J_f = ∂f/∂X
```
In this repo: prior p_Z = N(3.0, 1) (`CouplingFlow(prior_mean=3.0)`), loss in **bits per dimension** = −log p / (ln 2 · 2·T·18).

**Affine coupling** makes the Jacobian triangular so log|det| is a cheap sum. Split channels (x,y after a channel reversal) into `z1` and `z2`:

```
shift, s_logit = subnet(z1)               # subnet = ST-GCN block, sees ONLY z1
scale          = sigmoid(s_logit + 2) + 1e-6
z2'            = (z2 + shift) * scale     # code: stgnf_flow.py::CouplingStep.forward
log|det J|     = Σ log(scale)
inverse:  z2 = z2' / scale - shift        # CouplingStep.reverse
```
One flow *step* = `ActNorm` (learned per-channel scale+bias) → `Permute2d` (swap x,y roles) → coupling. **8 steps** total: **616 params** (SHT, k=13) / **488** (UB, k=9) — the whole anomaly model is smaller than a single 3×3 conv layer in most CNNs.

```
 X (N,2,T,18) ─► [ActNorm→Permute→Coupling] ×8 ─► Z ;  score = log N(Z; 3,1) + Σ log-dets ;  anomaly ⇔ low score
```

**Why forward NLL alone works without labels:** the model is trained to put high probability on normal motion; abnormal motion lands in a low-density region ⇒ low log p. No negative examples needed.

**Why a "reverse-consistency" loss is worthless here (E08 P_B):** because *f* is exactly invertible by construction (`CouplingStep.reverse`), `f⁻¹(f(X)) = X` up to float round-off, so ‖X − f⁻¹(f(X))‖² ≈ 0 regardless of training. Test: `tests/test_q2_p1_e08.py::test_every_step_is_exactly_invertible_so_the_reverse_term_is_round_off`. Adding a margin term instead: S4 −3.48 [−0.06, −0.01]* and FA/h@R50 +45.0 (E08 report §2/§4).

### The per-frame score trick
`frame_nll` regroups the window NLL into per-frame shares so a score can be attached to a specific frame; it's exact for any flow but only *causal* when every subnet is causal (docstring in `stgnf_flow.py`, line 259–262).

---

## 6. Offline vs strict time-causal evaluation (the systems heart of the thesis)

Think of AUC evaluation as a **replay of a log**. The question: *at the time a frame's score is shown, which inputs existed?*

| Protocol | Rule | Leaks the future? |
|---|---|---|
| **S0** "paper" | symmetric Gaussian smoothing of scores; missing frames filled with the **test set's** most-normal score; windows centred; last window of each track silently dropped | **Yes** (3 ways) |
| **S4** strict-causal | window score lands at its **last frame**; causal half-Gaussian; missing frames get a **train-calibrated sentinel**; hold H=0 | No |

SHT ladder (`Q2-P1-E00_audit/report.md`, ML0_mps): S0 77.88 → S1 73.53 → S2 73.57 → S3/S4 73.58. UB: 54.37 → 54.93 → 55.02 → 58.04 → 58.04. Locked sentinel (train-calibrated): SHT −0.5122, UB −0.8273.

**"Causal conv" ≠ lower latency.** A 24-frame window is complete only at frame 24, whatever padding you use, so left-padding the *inside* of a window gives no latency win under honest delivery (S4). E06: A2 (causal conv, per-window norm) 76.54 vs A0 76.86 (ML0, S4). The O(1) streaming variant (A3s, EMA normalisation) needs *past-only* normalisation: ML0 S4 72.89 vs A2 76.54 — a ≈3.7-point price (`E06 report`, A3s released 75.59 vs A0 80.85).

**What reuse is legal?** Different windows overlap (stride 1), so the same frame's pose appears in up to 24 windows. Reusing the *raw per-frame (x,y,conf)* is bit-exact (max_abs_diff 0.0; 18.5× fewer pose evaluations SHT, 13.2× UB). Reusing *normalised* tensors or *hidden states* is **invalid**, because normalisation is per-window (mean/std change when the window slides): hidden-state diffs 0.838 / 2.067 (E00, rule R7 FAIL). Go analogy: the raw sample is an immutable event; the normalised tensor is a *view* computed over a sliding aggregate — you can cache events, not views.

**Queueing.** Arrivals every 41.67 ms; service time per frame s. If s > 41.67 ms the FIFO queue grows without bound ("queue collapse"); deadline = 2 s lateness cap (`CAP_MS` in E02). Measured on Pi 5 (`Pi5_suite` live sustained, 2,880 frames at 24 fps): Fixed-K1 17.20 FPS, p99 wait 46,824 ms, delivered AUC 47.46% [44.2, 50.2]. With `Cache-C1+Guard`: 24.01 FPS, p99 405 ms, 64.49% [50.3, 78.1]; with `Burst32`: 24.01 FPS, 250 ms, 74.19% [65.9, 82.5] (73 windows scored — wide CI).

**The caches.**
* `C1`: reuse the last detector output while the scene is static (max age A1 = 4 frames). Net CPU saving 28.5% (E03).
* `Guard`: a hysteresis load shedder: start shedding when queue wait > 400 ms, stop when < 150 ms (`GuardedFixedPolicy`, E05).
* `Fixed-Burst`: the *fair baseline* — same CPU budget spent as intact bursts of windows rather than C1. E03 compares at matched cost (12-cell confirmation mean): Cache-Only 73.45 vs Fixed 71.06; Cache − Fixed +2.41 [+1.39, +3.48]*.

---

## 7. Hardware reality in one paragraph

Pi 5 = 4× Cortex-A76 @ 2.4 GHz. The A76 lacks the `i8mm` int8 matmul extension, so ORT INT8 does not beat NCNN FP16: at T=4, NCNN 46.43 ms vs ORT INT8 43.77 ms (1.06×) but 0.2697 vs 0.2981 J/frame (`Q2-P1-Pi5_suite/report.md` §2); INT8 is slower at T=1–2. Earlier E02/E04/E05 timings are **VM** numbers (Lima/VZ ARM64 on an M2 Pro); the physical Pi 5 appears only in `Pi5_huang_repro`, `Pi5_suite`, and E07–E10 — look at the header of each report before quoting a number.

---

## 8. Glossary (the 20% you'll use 80% of the time)

| Term | One line |
|---|---|
| AUC | area under ROC; "probability a random anomalous frame outscores a random normal frame". Pooled over all frames of all test videos here. |
| ML0 / M00 / MLP | fine-tuned YOLOv8n-pose with box-only labels for keypoint-less persons / control with 4 extra epochs / ML0 + stride-4 P2 head. |
| S0…S4 | evaluation protocols from offline (S0) to strict causal (S3/S4). |
| K1 / Fixed-K1 | run the detector on every frame ("full rate"). |
| C1 / C2 | C1 = detector-output reuse on static scenes; C2 = scorer reuse. |
| Guard | backlog hysteresis load shedder (on > 400 ms, off < 150 ms). |
| Delivered AUC | AUC of what the system actually released on time under queueing. |
| `*` after a CI | paired video-bootstrap (2,000 resamples) CI excludes 0. |
| dev / confirmation | tuning split / untouched test split; all selections are dev-only. |
| NLL, bits/dim | negative log-likelihood normalised per dimension and by ln 2. |
| LoRaD | 2026 JRTIP paper: causal temporal conv + online circular buffer for STG-NF. |
| TrajVAD-P | 2026 trajectory+pose flow (27 kinematic features), 757,338 params in the test. |
| FastOS-RL | the thesis' LinUCB OS-knob tuner (governor × threads × affinity/nice). |
