# 02 · Every experiment, grouped by paper direction

Conventions
* All numbers are quoted from the report named in the section header (paths relative to `experiments/`). AUC in percent, deltas in AUC points with `[95% paired video-bootstrap CI]`, `*` = CI excludes 0.
* **Machine tags matter.** `VM` = Lima/VZ ARM64 VM on an M2 Pro (E00–E05 and EXP-004…EXP-023 timings). `Pi5` = physical Raspberry Pi 5 (`Pi5_huang_repro`, `Pi5_suite`, E07–E10). Mac = MPS/CPU on the laptop.
* ✅ worked · ❌ failed/negative · ⚠ works but caveat. Warnings **W1–W5** are explained in `03_CODE_AUDIT_WALKTHROUGH.md` §5.

Direction map (your five + one I had to add for experiments that fit none):

| Dir | Theme | Experiments |
|---|---|---|
| D0 | Foundations (data, reproduction, low-light gates, UB replication) | EXP-000, 001, 002, 003, 005, 006, 007 |
| D1 | Perception & tracking recovery at constant FLOPs | EXP-011, 013–017, 020–022, E01, E07 (front-end/tracker) |
| D2 | Causal delivery auditing & computation reuse on Pi 5 | EXP-004, 008–010b, 012, 023, E00, E02–E05, Pi5_huang_repro, Pi5_suite |
| D3 | Causal normalizing flows, multi-scale dilation, interaction-aware graphs | EXP-018, 019, E06, E07 (TrajVAD-P), E08, E09-H1 |
| D4 | Workload-adaptive two-tier caching & green training | E09-H2, E09-H3 |
| D5 | FastOS-RL (OS/CPU tuning with a contextual bandit) | EXP-004 (OS part), E10 |

---

## D0 — Foundations

| Exp | Question | Result (from `results.md`) | Verdict |
|---|---|---|---|
| EXP-000 | Get UCF-Crime | Test set complete & verified; training download partial at time of writing | data only |
| EXP-001 | Reproduce STG-NF (ShanghaiTech) and measure the edge-pose drop | R1 85.94 (81.93 unsmoothed) ✅ passes 85.9 ± 0.5; edge YOLOv8n-pose poses 68.58 = **−17.4** ❌ "much larger than expected"; retraining on edge poses 67.59 ❌ (recipe verified: 85.31 on AlphaPose) | motivates the whole thesis |
| EXP-002 | Anchor reconstruction on Hockey Fight | model size gate PASS: 177,026 params, 0.0993 G MACs; RLVS/ShanghaiTech/UCF-Crime not run | partial |
| EXP-003 | Does low light hurt the edge pipeline, and does a label-free gate fix it? | replay 68.53 [63.94, 73.52] ✅; AUC falls 68.53 → 67.90 → 67.05 → 60.43 → 54.30 with darkness (L3 − L0 = −8.09 [−11.66, −5.03]); gate recovers ≈ nothing: H3 fail, H4 fail (G-xfit − best E = −3.09 [−4.83, −1.63] at L2) | ❌ for the gate |
| EXP-005 | Label-free coverage selector on unseen light levels | picks E2/CLAHE at X1, X2 (regret 0); at X3 picks E2, oracle E1, regret 3.14 / 2.20 | ⚠ |
| EXP-006 | Replicate on UBnormal | R0 71.78 = STG-NF's own script ✅; U1 960 − 640 = +3.25 [−1.64, +7.77] ❌ not significant | ⚠ |
| EXP-007 | Stronger front-end YOLOv8s-pose? | V1 8s − 8n = −2.26 [−4.32, −0.43] on SHT ❌ (8s *worse*); CLAHE +3.24 [+1.49, +4.93] ✅ | ❌ |

**Why D0 matters for the story:** it shows that *every* knob people reach for (bigger detector, brighter image, fancier gate) either fails or is within noise, which is what makes the later *label* explanation (EXP-020) striking.

**Mini-glossary D0:** *L0…L3* = darkening levels; *E0/E1/E2* = no-op / gamma / CLAHE enhancements; *coverage* = fraction of frames with a usable detected person; *paper vs live* = S0-style vs causal scoring variant.

---

## D1 — Perception & tracking recovery at constant FLOPs

**Hypothesis chain.** (1) Edge loss is detection, not keypoint quality (EXP-015). (2) Detection is blind below ~64 px (EXP-011). (3) The cause is label dropping, not resolution (EXP-020). (4) After the fix, what is left are intermittent gaps → two-stage association (E01).

### D1.1 Where the loss lives
| Exp | Table row(s) | Verdict |
|---|---|---|
| EXP-015 (oracle boxes + light pose) | released AlphaPose 85.94 [82.75, 88.79]; RTMPose-s 86.12 (+0.19 [−0.45, +0.81]); RTMPose-t 85.91 (−0.03); YOLOv8n-pose on same crops 84.99 (−0.94 [−1.94, +0.12]) | ✅ with the right people found and tracked, *any* small pose model reaches ≈ 85–86 → edge loss = detection + tracking |
| EXP-011 (recall by height) | stock 640: `<32 px` 0.001, `32–48` 0.007, `48–64` 0.057, `64–96` 0.613, `96–160` 0.963, `≥160` 0.991 (all 0.396) | ✅ diagnostic: blind below ~64 px |
| EXP-022 | fine-tuned (EXP-013) recall `<32` 0.823 vs 0.001, `32–48` 0.912 vs 0.007, all 0.947 vs 0.396 | ✅ fine-tune recovers small people |
| EXP-016 (top-down sweep) | 960 px K=1 83.11 [80.15, 85.92], 640 K=1 80.52; K=3 loses −4.12 / −4.43 (T2 FAIL); **no config meets 24 fps on the 4-vCPU VM at its mean** (640 K=1 71.8 ms, 960 K=1 111.6 ms vs 41.7 ms) | ⚠ accurate, too slow |

### D1.2 What did *not* fix it
| Exp | Result | Verdict |
|---|---|---|
| EXP-017 YOLO26n-pose (NMS-free) | SHT L0 61.54 vs 68.53 baseline (−6.99 [−10.07, −4.40]); recall 64–96 px 0.209 vs 0.613; only "no transfer loss on UB" passes | ❌ |
| EXP-013 fine-tune on SHT-train AlphaPose pseudo-labels | SHT L0 +11.75 [+7.08, +16.40] ✅ but UB −4.12 [−9.44, +1.21] ❌ (site-specific) | ⚠ |
| EXP-014 WiSE-FT blend (α* = 0.75) | SHT 77.60 vs 80.28 (−2.68) ❌; UB 55.67 vs 51.63 (+4.04) ✅ | ⚠ trade-off |
| EXP-020 resolution (M0P stride-4 head, stock labels) | +0.24 [−1.31, +1.76] ❌ (P2 FAIL) | ❌ |
| EXP-021 coverage → AUC bound | B1 FAIL: 12 of 194 points violate (max excess 0.0511); B2 PASS: Spearman 0.937 (SHT, n=160), 0.859 (UB, n=34) | ⚠ bound is "mostly" right |

### D1.3 What fixed it (the contribution)
| Exp | Result | Verdict |
|---|---|---|
| EXP-020 label cause (P1) | ML0 − M00 **+16.24 [+11.99, +20.45]**; recall 48–96 px 0.142 → 0.856; SHT L0 AUC orig@640 68.53, M00 61.65, ML0 77.88, M0P 61.89, MLP 79.35 | ✅ (note: M00 is *below* the original, so quote ML0 − stock for headlines) |
| E01 ML0 vs stock (S4, ShanghaiTech) | stock 66.03 → ML0 73.78; ML0 − stock@640 +7.75 [+4.63, +10.47]*; ML0 − M00 +11.66 [+8.25, +14.82]* (MPS) / +11.89 [+8.48, +15.12]* (NCNN) | ✅ headline |
| E01 P2 head | MLP − ML0 +0.91 [−0.02, +1.81]; MLP = 1.92× FLOPs | ❌ not worth it |
| E01 two-stage tracker | tracks 2,006 → 1,766; ML0 S4 73.78 → **76.82** (+3.04 [+1.44, +4.71]*); S0 81.62 | ✅ |
| E01 diagnosis | 84.2% of remaining misses are intermittent gaps; 52.3% of those have a cached low-confidence box in [0.05, 0.25) | ✅ mechanism |
| E07 front-end/tracker | TrackTrack − BoT-SORT: better 10, equal 2, inconclusive 16 (n = 28); best `ML0|TrackTrack|STG-NF` 70.33 (SHT 76.36 / UB 64.30); YOLO26 e2e + TT + STG-NF −14.2 [−19.1, −9.7] vs locked E01 baseline | ✅/❌ confirms ML0, rules out YOLO26 |

**What failed in D1 (negative results worth a table in the paper):** bigger detector (EXP-007), NMS-free detector (EXP-017), P2 head (EXP-020 P2, E01), resolution (EXP-016 under deadlines), fixed brightness gating (EXP-003).

**Mini-glossary D1:** *recall by height* = fraction of reference persons (from released AlphaPose) matched by a detector box (conf ≥ 0.25, IoU ≥ 0.3); *intermittent track* = reference track found in some frames, missed in others; *M00 / ML0 / M0P / MLP* = {stock labels, ML0 labels} × {no P2, P2}.

---

## D2 — Causal delivery auditing & computation reuse on the Pi 5

### D2.1 Protocol audit (`Q2-P1-E00_audit/report.md`)
| Step | SHT (ML0_mps) | UB (ML0_mps) | Meaning |
|---|---|---|---|
| S0 offline paper | 77.88 | 54.37 | symmetric smoothing + test-set most-normal fill + omit last window |
| S1 | 73.53 | 54.93 | |
| S2 | 73.57 | 55.02 | |
| S3 / S4 | 73.58 | 58.04 / 58.04 | strict causal; train sentinel (SHT −0.5122, UB −0.8273) |

Windows lost by the last-window omission: 782 of 91,867 (0.85%). Reuse: pose reuse bit-exact (18.5× SHT / 13.2× UB fewer pose evals); post-norm reuse invalid (R7 FAIL); hidden-state diffs 0.838 / 2.067. ML0 anomalous/normal frame coverage on SHT: 91.0 / 72.5 vs stock 67.5 / 53.4.

### D2.2 Deadline-aware evaluation (earlier VM experiments)
| Exp | Result |
|---|---|
| EXP-004 (150 timing runs + Amendment 2) | deadlines reverse the offline ranking: 960 px best offline (+2.0 at L0) but worse than 640 delivered by 2.4 / 3.1 (4 CPUs); learned enhancement (Zero-DCE++) net-negative; SCHED_FIFO worsens tails; pinning halves p99 under a 2-CPU budget |
| EXP-008 (92 replays) | A1b: Kendall τ(offline, delivered) 0.72 at s=1 → **0.16** at s=4 ✅; A3 predictive model fails (MAE 2.62, ρ 0.61) ❌ |
| EXP-009 / 010 / 010b | pose filling: FILL-HOLD − SKIP +5.65 mean over 12 cells (EXP-010, PASS C3). EXP-010b (other mask seed) replicates the direction but **fails the per-cell rule** (3 of 6 groups) ⚠ |
| EXP-012 | NCNN FP16 800 px: 18.9 ms vs ORT FP32 640 32.9 ms (0.57×), AUC 70.65 vs 68.53; INT8 costs ≈ 1 pt ✅ |
| EXP-023 | Observation reuse, visual propagation, bounded refresh (`EXP-023-observation-reuse/report.md`); its sentinel/24 fps defects were found and fixed in E00 (see E00 §9) |

### D2.3 The core ablation chain (matched CPU cost)
| Exp | Key rows | Verdict |
|---|---|---|
| E02 fixed baseline (VM) | decode 0.38 + pose 11.67 + track 0.42 + score 2.46 = 14.93 ms/frame (SHT), UB 20.21; s* 3.08 (SHT) / 1.66 (UB); RSS 896 / 922 MB; delivered S4 ideal 76.82 → 76.60 at s=2, collapses at s=3 | ✅ baseline |
| E03 ablation (SHT 12-cell confirmation means) | Cache-Only 73.45, Independent-Combo 73.06, Joint 71.86, Fixed 71.06, Adaptive-Only 70.89, Motion-Only 69.71; Cache − Fixed **+2.41 [+1.39, +3.48]***; Joint − Cache −1.60 [−2.43, −0.82]*; Joint − Independent −1.20 [−2.06, −0.34]*; C1 net CPU saving 28.5% | ✅ cache wins; ❌ "joint controller" loses |
| E04 optimizations | MV not adopted (UB dev −2.25 [−3.99, −0.41]*); INT8 not adopted; "GPU experiment warranted: no" | ❌ (clean negatives) |
| E05 confirmation (VM) | SHT Stock-Baseline 60.78 → Fixed 71.06 → Cache-C1 73.43; V3 +12.65 [+7.96, +17.41]*; V4 ReTrack +1.48 [+0.24, +2.62]*; UB Cache-C1+Guard 59.45; V5 +5.93 [+1.73, +9.92]* on 7 overloaded UB cells; p99 wait 4 / 405 / 408 ms | ✅ |
| E05 event metrics | recall@2s Stock-Baseline 16.4 → Cache-C1 30.3 (Full-Rate 34.9); missed 67.9 → 51.2; FA/h 11.4 → **28.5** (false alarms rise); localised AUC 51.5 → 62.2 (Full-Rate 68.0) | ⚠ trade-off: more recall, more false alarms |

### D2.4 Physical Pi 5 (`Q2-P1-Pi5_huang_repro`, `Q2-P1-Pi5_suite`)
| Config | Result |
|---|---|
| Huang `RPi` (PyTorch 640) | 6.14 FPS, 162.86 ms, 6.04 W, 0.984 J/frame, delivered AUC **53.32** (offline 68.53) — queue-model value, not an on-device AUC (see W6 below) |
| Huang `RPi*` (NCNN 320) | 36.57 FPS; AUC capped by small-person blindness: 57.28 offline / 59.18 causal |
| Stock NCNN 640 full-rate | 17.56 FPS; ML0 full-rate 16.60 FPS, delivered 54.73 |
| ML0 + Cache-C1 + Guard | 28.45 FPS, 35.16 ms, delivered 63.67 at W=1 |
| **Sustained live E05 (2,880 frames @24 fps, measured on Pi)** | Fixed-K1 17.20 FPS, p99 46,824 ms, 47.46 [44.2, 50.2]; Cache-C1 23.00 FPS, p99 9,583 ms, 46.92; Cache-C1+Guard 24.01 FPS, p99 405 ms, 64.49 [50.3, 78.1]; Burst32 24.01 FPS, 250 ms, 0.168 J/frame, 74.19 [65.9, 82.5] (only 73 windows scored) |
| Matched-CPU full-dataset comparison (Pi-calibrated sim.) | Cache-C1 − Fixed +2.56 [+0.77, +4.42]*; with Guard +7.46 [+5.52, +9.35]* |
| E04 on Cortex-A76 | NCNN FP16 46.43 ms (T=4) vs ORT INT8 43.77 ms (1.06×), J/frame 0.2697 vs 0.2981; INT8 slower at T=1–2; ORT FP32 98.83 ms |

**Caveat W6 (read before quoting Pi numbers).** Three kinds of "Pi" AUC exist: (a) *measured on the board* (sustained E05 live table; E07/E08 delivered AUC computed on the Mac from Pi-logged per-frame scores on 12 clips; E09/E10 timing/FPS/power), (b) *Pi-calibrated simulations* with measured Pi stage pools (`Pi5_suite` §4, explicitly labelled), (c) *`Pi5_huang_repro` delivered-AUC columns*: `q2_p1_pi5_huang.py` stores only hard-coded per-arm offline / causal-ideal AUCs (`get_arms`, e.g. 68.53 / 66.03 for stock, 81.62 / 76.82 for ML0) and measures timing/power on-device; the report's delivered AUCs come from the queue model (its findings item 4 states `s_pi = 4.45` on the full 107-clip stream). Only (a) is a direct on-device AUC; label each table accordingly.

**What failed in D2:** motion-only gating (−1.35 pt in the positioning doc; 69.71 in E03 table), joint controller (−1.60), MV codec gating (−0.08 conf / −2.25 UB dev), INT8, learned enhancement, EXP-010b per-cell rule, EXP-008's predictive model.

**Mini-glossary D2:** *K1* = detect every frame; *Fixed-Burst W* = intact windows of length L every W frames at matched cost; *C1/C2* = detector-output / scorer reuse; *Guard* = hysteresis shedder (400 / 150 ms); *s* = VM slowdown factor; *rho* = utilisation (service/arrival); *delivered AUC* = AUC of scores released within 2.0 s.

---

## D3 — Causal normalizing flows, multi-scale dilation, interaction-aware graphs

### D3.1 Is "LoRaD" worth adopting? (`Q2-P1-E06_lorad_causal_flow/report.md`)
| Arm | Mechanism | ML0 S0 / S4 (SHT) | Verdict |
|---|---|---|---|
| A0 | frozen STG-NF | 81.62 / 76.86 | baseline |
| A1 | retrained STG-NF (3 seeds) | 81.57 / 76.86 | retraining alone −1.09 vs released on released poses |
| A2 | causal conv + per-window norm (N0) | 81.12 / 76.54 | ✅ "free" (R1 +0.24 [+0.01, +0.46] at S0), but no latency win |
| A3 | causal conv + prefix/EMA norm (N1) | 81.61 / 76.55 | N1 not adopted: released −2.16 [−2.97, −1.37]* |
| A3s | true O(1) stream mode | released 80.16 / 75.59 | ❌ extra −1.69 [−2.79, −0.50] on top |
| A2L | lite depthwise conv (408 params) | 80.63 / 76.05 | ❌ −1.64 [−2.70, −0.59]* |

Checks: causal prefix perturbation → max |Δ| = 0 (5.6 for the acausal control); online vs batch max |Δ| = 4.6e-6 over 383 windows; R5 half-passes (stream latency flat in L but the O(1) advantage is not measurable at L ≤ 96 on the Mac).

### D3.2 TrajVAD-P and the factorial matrix (E07)
* TrajVAD-P (27 kinematic features + conditional pose flow, 757,338 params): worse than LoRaD by 3.7–5.1 pt on SHT (all front-ends, CI hi < 0). Better than TrajVAD-T by ≈ 1.5 pt on SHT YOLO26/stock.
* Pi 5 arms (12 SHT clips × 240 frames): ML0→TT→TrajVAD-P full 12.3 FPS, p99 109,547 ms, delivered 39.7; +C1+Guard 20.0 FPS, 487 ms, 55.3. YOLO26→TT→LoRaD +C1+Guard 28.7 FPS, 410 ms, delivered 63.6 (the best delivered real-time pipeline in E07). **Acceptance rule C1 is not met on any pipeline.**

### D3.3 E08 — 5 proposals tested against pre-registered rules (`Q2-P1-E08_lorad_gaps_proposals/report.md` §1)
| Proposal | Idea | Primary S4 ΔAUC vs B1 (ML0, confirmation) | Verdict (report's own) |
|---|---|---|---|
| P_A MS-CTFlow | dilations d ∈ {1,2,4}, 1,592 params | −1.51 [−0.02, −0.01]* ; short-event recall@FA(B1) 39.4 → 44.2 (+4.9) | **REFUTED** |
| P_B RevMargin | reverse-consistency + margin | −3.48 [−0.06, −0.01]*; FA/h +45.0 | **REFUTED** (reverse term ≡ 0) |
| P_C IAC-Flow | 20-node graph, node 19 box-velocity, node 20 social | −0.71 [−0.02, +0.01]; FA/h@R50: SHT 49.1 → 35.0 (−14.0 [−30.1, −3.3]*), **UB 101.2 → 167.4**, mean **+26.1 [+7.5, +46.5]*** | **NOT SUPPORTED** |
| P_D ConfSmooth | confidence smoothing | +0.01 [−0.00, +0.00] | **PARTIAL** (no gain) |
| P_ABC | all three | −3.69 [−0.06, −0.01]* | **PARTIAL** (H_Joint) |

Pi 5, Cache-C1+Guard, 12 clips: P_C 25.45 FPS, 5.56 ms scorer, p99 419 ms, delivered 67.51; P_ABC 23.44 FPS, 68.02; B0 STG-NF 67.12; B1 LoRaD 63.48; TrajVAD-P 55.71. The spread between B0 (67.12) and B1 (63.48) at identical cost shows the 12-clip delivered AUC is noise-level (E08 §10: same 12 clips as E07).

### D3.4 E09 H1 — MS-CTFlow-Bottleneck (`Q2-P1-E09_hypotheses_suite/report.md`)
608 params, K=3, d ∈ {1,2,4}. Short-event (≤ 2.5 s, clip-level split) S4: SHT ML0_edge 74.91 → 81.76 (+6.85); pooled S4 drops 76.29 → 72.59. Verdict in report: PARTIALLY SUPPORTED. See W4 (no CIs, single seed, sentinel calibrated on the first 12 evaluated clips).

**What failed in D3:** causal-lite, A3/N1, A3s, RevMargin, MS-CTFlow (pooled), IAC-Flow (mean FA/h), EXP-018 DAPF (D1 FAIL pooled −1.99 [−3.51, −0.29]; +C1 −2.45 [−3.89, −0.84]), EXP-019 GAU (−0.78, CI [−2.16, +0.63]; 5,808 vs 616 params).

**Mini-glossary D3:** *ActNorm* learned per-channel affine; *Permute2d* swaps x/y roles; *coupling* invertible transform where half the channels parameterise the other half; *dilation d* skips d−1 frames between conv taps; *prefix-invariant* = score at t unchanged if the future is deleted; *N0/N1* per-window vs streaming-prefix normalisation; *FA/h@R50* false alerts per hour at 50% event recall.

---

## D4 — Workload-adaptive two-tier caching & green training (E09 H2/H3)

**What the report claims** (`Q2-P1-E09_hypotheses_suite/report.md`): H2 Two-tier MPC-LoRaD (Tier 1 scene deep-sleep + Tier 2 per-person kinematic extrapolation) saves pose calls with `|ΔAUC|` small and energy by workload regime; H3 distillation-free green training (L_NLL + bone + smooth + reconstruction) matches a 100-epoch teacher in 45 s (1,150× faster); UB G2 beats G0 by +3.79 pt S4 (60.92 → 64.71).

**What the code shows (W1–W4, see `03_CODE_AUDIT_WALKTHROUGH.md` §5):**
* **W1** H2 Strategy-A accuracy uses an **oracle skip rule** (`p_true = kp_arr[t]` is read to decide whether to skip) — not deployable as stated.
* **W2** Medium/Sparse energy regimes are **synthetic schedules** (`(i % 120) < 48`, `(i % 240) < 24`), and Tier 1 sleeps whenever the synthetic `is_active` flag is False.
* **W3** H3's G0 cost is **hard-coded** (`g0_stage1_hours = 14.5`, 0.250 kW; G2 0.045 kW); G1/G2 train 6 epochs × ≤ 80 batches; G0 vs G2 also differ in architecture (LoRaD K=13 vs MSCTFlow K=3).
* **W4** The E09 evaluator calibrates the S4 sentinel on the first 12 *evaluated* clips, not on training data (E00 rule), and reports no CIs.

**Honest use:** D4 is a promising *hypothesis section*, not yet evidence. The Pi-measured, causal part is the Heavy (100%) and natural-sparse workload with Tier-2 pixel-difference gating (no AUC measured there).

**Mini-glossary D4:** *MPC* = predictive caching by constant-velocity extrapolation; *Tier 1/2* = scene-level gate / per-person gate; *G0/G1/G2* = teacher distillation / direct training / kinematic-regularised training.

---

## D5 — FastOS-RL (`Q2-P1-E10_rl_os_tuning/report.md`)

* **Setup:** 18 OS configurations (3 governors × T ∈ {2,3,4} × {unpinned nice 0, pinned nice −10}); 8-dim LinUCB warm-started from E02/EXP-004 priors; 12-frame (0.5 s) micro-windows; reward R = −(2.5·Drop + 1.0·Mean/41.67 + 0.4·p95/41.67 + 0.15·J). Header text says 6 pipelines and 10–14 s; the tables show 8 pipelines and tune times 4.0–18.7 s (documented inconsistency).
* **Measured gains (FPS, Pi 5, single run):** 2_LoRaD_Cached 40.28 → 43.79 (+8.7%, `ondemand` was dropping to 1500 MHz); 4_ML0_Cached 32.97 → 34.81 (+5.6%); 6a MSCTFlow Cached 36.98 → 40.46 (+9.4%); 6b 28.80 → 29.85 (+3.6%); full-rate pipelines +1.2% to +3.1%; Huang RPi PyTorch640 +0.5%.
* **The "static EXP-004" baseline (ondemand|T3|pin3) is a strawman:** it is worse than `OS_Default` on most rows (e.g. 6b: 12.64 vs 28.80 FPS).
* **W5:** the "Delivered Pi 5 AUC" column is **not measured**; it is a hand-written formula of measured service time and drop rate (`estimate_delivered_auc_from_pi5_service`, floor 53.32 hard-coded). AUC deltas there (+0.00, −5.28, −11.11 …) are outputs of that formula.
* **Missing control:** no static `performance|T4` arm — a one-line `cpufreq` change may capture most of the gain.

**Mini-glossary D5:** *governor* = Linux CPU-frequency policy (`ondemand`, `schedutil`, `performance`); *affinity/nice* = core pinning and scheduling priority; *LinUCB* = contextual bandit with ridge regression + upper-confidence exploration; *warm start* = prior θ₀ instead of zeros.
