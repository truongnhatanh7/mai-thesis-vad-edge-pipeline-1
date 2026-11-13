# EXP-003 — Low light in an edge pose-VAD pipeline: dose-response, mechanism, remedies

Pre-registration frozen 2026-11-04 23:40 (before any EXP-003 run), under D13 (owner delegated the loop; review
afterwards). Changes after this time go to "Deviations" in `log.md` and `results.md`.

## 1. Background (plain language)

Our edge pipeline finds people with a small pose model (YOLOv8n-pose), links them over time with a tracker
(BoT-SORT), and a tiny flow model (STG-NF) scores how unusual each person's motion is. EXP-001 showed this pipeline
already loses 17 AUC points vs. the paper's heavy pose model, mostly because **small people get low detector
confidence, the tracker refuses to start a track below a confidence threshold, and those people are never scored**.

Darkness also lowers detector confidence. So low light may hurt mainly by closing these confidence "gates", not by
making the poses wrong. If so, the cheapest fix is not image enhancement (which costs CPU) but re-setting the gate
thresholds when it is dark (costs nothing).

## 2. Problem

No published edge low-light study separates *where* the accuracy goes (missed people vs. bad poses) in a tracked
pipeline, and none compares enhancement with threshold re-calibration on equal footing (accuracy and compute).

## 3. Objective

Measure, on ShanghaiTech test (107 videos, 40,791 frames, released STG-NF checkpoint, frozen):
1. dose-response of AUC to synthetic low light;
2. how much of the loss is coverage (no scored person) vs. pose/track quality;
3. AUC recovered by enhancement (gamma, CLAHE, Zero-DCE++) vs. gate re-calibration vs. both, with their CPU cost.

## 4. Method

### 4.1 Synthetic low light (sensor-style, fixed before running)
Per frame, 8-bit sRGB → linear (`srgb_to_linear`) → expected electrons `λ = α · L · K` with full-well `K = 1000` →
`e ~ Poisson(λ) + N(0, σ_r²)`, `σ_r = 2` electrons → `L' = clip(e / K, 0, 1)` (no camera gain) → sRGB → round to
uint8. RNG seed = stable hash of (video, frame, level). Levels:

| Level | α | Approx. mean luma (clean ≈ 120) |
|---|---|---|
| L0 | clean (frame unchanged) | ≈ 103–138 |
| L1 | 0.3 | ≈ 70 |
| L2 | 0.1 | ≈ 37 |
| L3 | 0.03 | ≈ 15 |
| L4 | 0.01 | ≈ 6 |

Real UCF-Crime test videos have mean luma 32.5–170.7 (EXP-000), so L1–L2 is the realistic range, L3–L4 stress.
Mechanism control: **L2-nonoise** (α = 0.1, no Poisson/read noise) separates darkness from noise.

### 4.2 Enhancement arm (P)
E0 none · E1 adaptive gamma LUT (γ = clip(log 0.4 / log mean_Y, 0.2, 1.0), mean_Y of the frame in [0,1]) ·
E2 CLAHE on LAB L channel (clipLimit 2.0, tiles 8×8) · E3 Zero-DCE++ (official weights, scale_factor 1).
Applied after darkening, before the detector, to every frame (no gating in this experiment).

### 4.3 Detection cache and replay
YOLOv8n-pose (sha256 `c6fa93dd…d212`), imgsz 640, iou 0.7, **conf 0.01**, post-NMS boxes + 17 keypoints cached per
frame for each (level × enhancement). Device for the cache: MPS if its detections match CPU on a 2-video check
(max box-conf |Δ| ≤ 0.01 and equal box counts at conf ≥ 0.1 on ≥ 99% of frames); else CPU.
Replay: BoT-SORT (Ultralytics, `gmc_method: none`, no ReID) on cached detections filtered at threshold `c`, then
STG-NF JSON → frozen checkpoint → frame AUC.

### 4.4 Gate arm (G)
One-parameter family `c ∈ {0.25 (default), 0.15, 0.10, 0.05}`: detector conf = `c`, tracker
`track_high_thresh = c`, `new_track_thresh = c`, `track_low_thresh = min(0.1, c/2)`, other tracker params default.
**Selection without test tuning:** 2-fold cross-fitting over videos (seeded random halves, seed 0): the best `c` per
(level, enhancement) is picked on one half by AUC and applied to the other; pooled scores over all 107 videos are
reported as "G-xfit". The full sweep is reported as descriptive (oracle) only.

### 4.5 Metrics
- Primary: frame-level AUC, **paper variant** (middle frame, offline Gaussian) and **live variant** (end frame,
  causal half-Gaussian), as EXP-001. 95% CI = percentile bootstrap over videos (2000). Differences between arms:
  **paired** bootstrap over videos.
- Mechanism: anomalous-/normal-frame coverage (≥ 1 scored person), number of segments, AUC on frames covered in
  both L0 and Lk ("co-covered AUC").
- Cost (Mac M2 Pro CPU, comparative): ms/frame of each enhancement and of the detector, measured idle, 1/2/4 threads.

## 5. Pre-registered expectations / gates

| # | Expectation | Pass rule |
|---|---|---|
| G0 | Replay is faithful | L0 + E0 + c 0.25 within ±1.5 of EXP-001 R2 (68.58, CPU, gmc sparseOptFlow). Outside → deviation, our L0 becomes the baseline |
| H1 | Dose-response | AUC non-increasing L0→L4 (paper variant, E0, c 0.25); L3 − L0 ≤ −5 |
| H2 | Gates, not poses | at L2 and L3: drop in co-covered AUC < ½ of the overall drop, and anomalous coverage falls |
| H3 | Gate re-calibration recovers | G-xfit (E0) recovers ≥ 50% of (L0 − Lk) AUC at k = 2, 3 |
| H4 | Gate ≥ enhancement | at L2, L3: Δ(G-xfit, E0) ≥ Δ(best of E1–E3, c 0.25); paired bootstrap, Holm over 2 levels |
| H5 | Enhancement is not free on clean frames | at L0 every E1–E3 changes AUC by ≤ +1 (no real gain) |
| H6 | Combination | best (E, G-xfit) ≥ max(G-xfit E0, best E at c 0.25) at L2, L3 |

These are expectations, not hopes: any of them failing is a result and will be reported as such.

## 6. Deviations policy

No metric, threshold, split or darkening parameter changes after the first run without a dated log entry and a
results.md "Deviations" row. Extra exploratory analyses are allowed if labelled **exploratory** and kept apart from
the table above.

## 7. Addendum A — input-resolution arm (exploratory, frozen 2026-11-05 09:32, before any run of it)

Why added: interim results (log 09:32) show the gate moves AUC by ≤ 0.3 at L0–L2 and not at all at L4,
while EXP-001 found +3.1 AUC from imgsz 864. The arm tests whether the detector's **input resolution** — a pure
compute knob — is the better place to spend CPU than enhancement. Labelled exploratory: it was added after seeing
H3-relevant interim data (not after seeing any resolution result).

- Factor R = detector imgsz ∈ {480, 640 (existing caches), 800, 960}; E0; levels L0, L2, L3; cache at conf 0.01;
  full gate sweep replay as §4.4. Cache dirs `<level>_E0-r<imgsz>`.
- Cost per R: detector CPU ms/frame measured in the VM (EXP-004 harness), not the MPS times in the cache.
- Expectations:
  - H12: at L0, AUC(960) − AUC(640) ≥ +3 (paper variant, c 0.25), and AUC non-decreasing in R at L0 and L2.
  - H13: at L2 and L3, the best resolution gain over 640 exceeds the best enhancement gain (E1–E3, c 0.25).
  - H14 (descriptive): resolution gain at L3 < at L0 (darkness limits what extra pixels can reveal).

## 8. Addendum B — robustness and combination checks (exploratory, frozen 2026-11-05 22:06, before any run of it)

Why: results.md Obs. 4 and 6 (enhancer sign flips; noisy-dark + CLAHE above clean) rest on one noise realisation;
the condition-aware policy idea needs to know whether enhancement and resolution add up.
- **B1 noise-seed replicate:** levels `L2-s1` (α 0.1) and `L3-s1` (α 0.03) = same sensor model, different seed
  (seed from the level name). Caches: L2-s1 × {E0, E2}, L3-s1 × {E0, E1}; c 0.25 + gate sweep.
  Expectation: Δ(E2 − E0) at L2-s1 within ±1.5 of +3.15 and > 0; Δ(E1 − E0) at L3-s1 same sign as +2.86.
- **B2 combination:** L2 × E2 × imgsz 960 and L3 × E1 × imgsz 800.
  Expectation (descriptive): combination ≥ max(enhancement alone at 640, resolution alone) at each level.

## 9. Addendum C — camera gain and the Huang 2026 operating point (exploratory, frozen 2026-11-05 22:15, before any run)

Runs after the EXP-004 timing matrix (host must stay idle for timing).
- **C1 camera gain:** levels `L2-gain` (α 0.1, gain 10 = full compensation) and `L3-gain` (α 0.03, gain 16, 24 dB
  cap), same noise realisation as L2/L3 (seed from the base level), gain applied before 8-bit quantisation
  (`darken.GAIN_LEVELS`, tested). Caches: × {E0, E2}. Expectation (descriptive): gain-compensated AUC lies between
  the no-gain level and L2-nonoise (gain removes quantisation loss, not noise); CLAHE's gain over E0 shrinks.
- **C2 imgsz 320** at L0 and L2 (E0): the input size Huang et al. 2026 (IEEE ESL, Pi 5, NCNN) deploy to reach
  17.7 FPS without reporting its AUC cost. Expectation: AUC(320) < AUC(480) at both levels.
