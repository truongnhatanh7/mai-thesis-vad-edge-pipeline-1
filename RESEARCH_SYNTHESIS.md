# Research synthesis — from the experiment loop to a publishable paper (DRAFT, 2026-11-06)

## Headline finding (revised 2026-11-06 20:40 after the UBnormal replication, EXP-006)

> **On a CPU-only edge pose-VAD pipeline, accuracy is limited by the pose front end's coverage — how many people it
> keeps tracked — on both datasets tested (−16 to −17 AUC vs. released poses). Re-calibrating detector/tracker
> confidence thresholds never recovers it (ShanghaiTech and UBnormal, both scoring variants). Input resolution is
> the dominant offline lever (480 px costs 4–6 AUC on both datasets), but under a real-time deadline on a CPU the
> gain of higher resolution is eroded (UBnormal) or reversed (ShanghaiTech) by dropped frames. Illumination
> enhancement effects are small and dataset-specific: the light-gated rule found on ShanghaiTech (none / CLAHE /
> gamma) does not replicate on UBnormal. OS tuning (pinning + co-load isolation, no SCHED_FIFO) cuts tail latency
> but does not change these rankings.**

Cross-dataset status of each claim:

| Claim | ShanghaiTech (EXP-003/004/005) | UBnormal (EXP-006) | Status |
|---|---|---|---|
| front end decides accuracy | −17.4 (EXP-001) | −16.0 | **replicated** |
| thresholds are not a remedy | H3 fail (both variants) | U2 pass | **replicated** |
| lower resolution costs | 480 px −6.3 / −4.5 | −4.5 / −4.0 (sig.) | **replicated** |
| higher resolution helps offline | +3.2 / +2.0 (sig.) | +3.3 / +2.9 (n.s.) | same sign, not sig. on UB |
| deadlines erode/reverse resolution gain | 960 px −2.4 delivered (4 CPU) | +0.71 delivered (from +2.86) | direction replicated, size dataset-dependent |
| coverage tracks AUC | ρ 0.95 (45 configs), 0.87–0.92 out of sample | 0.89–0.94 dark configs; 0.60 clean | mostly replicated |
| light-gated enhancer (CLAHE / gamma) | yes (incl. noise-seed replicate) | **no** (UA1, UA2 fail) | **not replicated** |
| offline rank predicts delivered rank | τ 0.72 (VM) → 0.16 (4× slower, EXP-008) | included in τ | **fails on slow devices** |
| **causal pose prediction of dropped frames (EXP-009, owner's I/P/B idea)** | +2.8 / +5.6 delivered at 2×/4× slower (sig.) | +1.4 (sig.) / +1.8 (n.s.) | **confirmed (EXP-010, pre-registered): +5.7 to +8.3 at 3× slower (5/6 sig.), ShanghaiTech also at 6×; not sig. on UBnormal-8n; no rescue above ~90% drops** |
| OS: no-FIFO pinning halves p99 under budget | H7′ pass | — (same traces) | VM only; Pi needed |

Evidence: EXP-003 (accuracy ledger), EXP-004 (210 VM timing runs, delivered AUC at 24 fps), EXP-005 (out-of-sample
light levels), EXP-006 (second dataset). Weakest links: synthetic light, VM timing, near-chance UBnormal baseline.

Lead agent's synthesis of EXP-003 and EXP-004 (both complete) and EXP-005 (all complete). Every number cites an experiment file; nothing here is new data. Owner review needed before it replaces
anything in LITREVIEW.md or the README RQ.

## 1. What we set out to test, and what happened

| Original bet (RQ v1, LITREVIEW.md) | Outcome | Evidence |
|---|---|---|
| Low light degrades edge public-safety inference | **Only below a cliff.** Down to mean luma ≈ 37 (darkest real UCF-Crime videos ≈ 32) the pose-VAD pipeline loses ≤ 1.5 AUC (live ≈ 0); at luma ≈ 15 / 6 it loses 8 / 14 (live 4.5 / 10.7) | EXP-003 results §2.1, §2.4 |
| Illumination preprocessing restores robustness | **Condition-dependent, sign-flipping.** Gamma hurts at moderate dark (−1.6 to −2.9, both variants); CLAHE is the only moderate-light gain (+1 to +3); Zero-DCE++ ≤ +2 and very expensive; nothing helps at the extreme | EXP-003 §2.1, §2.4, §2.6 |
| "Gates, not pixels": low light acts through detector/tracker confidence thresholds | **Half right.** The loss is coverage (people never scored: 59–84% of the drop), but lowering the thresholds never helps — the people are below any threshold | EXP-003 H2 pass, H3/H4 fail |
| OS tuning changes accuracy only through deadlines; P × O interact | **Deadlines dominate; OS tuning is not a free win.** At 24 fps, 960 px drops 45–63% of frames and turns its +2.0 offline gain into −2.4 / −6.5 delivered; Zero-DCE++ (100–190 ms) keeps 14–29% of frames and is net-negative everywhere; cheap CLAHE/gamma keep their gains. SCHED_FIFO escapes cgroup v2 CPU quotas and worsens tail latency; without it, pinning + co-load isolation halves p99 under a 2-CPU budget (H7′ pass) and keeps 76% more frames for Zero-DCE++, but the pre-registered OS × enhancement interaction is not significant (H8′ +2.3 [−0.5, +5.2]); only cheap CLAHE gains significantly (+0.86 at L3) | EXP-004 results §1–6 |
| (new) Coverage as the common currency | AUC tracks anomalous-frame coverage across 45 configurations (ρ 0.95 / 0.94 live); label-free all-frame coverage ranks configurations on **unseen** light levels too (ρ 0.92 / 0.87, EXP-005 F3 pass), but as a selector it only ties "always CLAHE" and fails in deep dark (gamma wins with the lowest coverage — replicated at L3 and X3) | EXP-003 log; EXP-005 results |
| (new) Detector input resolution | **The largest single lever** in the realistic range: 480 → 960 px = +9.5 AUC (live +6.5); vanishes in deep dark (+0.2): resolution × light interaction; combines with CLAHE at moderate dark (72.16, best of all) | EXP-003 Addendum A, B |

## 2. Refined research question (proposal)

> **RQ:** *In a CPU-only edge pose-based anomaly-detection pipeline, how should a fixed CPU budget be split between
> input resolution and illumination preprocessing under different lighting conditions, and how does OS-level CPU
> allocation change which split is best once frames must meet a real-time deadline?*

Sub-questions:
- **SQ1 (mechanism):** Through which pipeline stage does low light lose accuracy — detection coverage, pose
  quality, or decision thresholds? → answered (EXP-003): coverage; thresholds are not a lever.
- **SQ2 (accuracy ledger):** What is each knob's accuracy effect per lighting condition? → answered (EXP-003, both
  scoring variants), with replication across noise seeds.
- **SQ3 (cost ledger):** What does each knob cost in CPU time on an edge-class CPU, and how many frames does it drop
  at 24 fps? → answered on the VM (EXP-004 §2); Pi 5 confirmation outstanding.
- **SQ4 (joint):** Does OS-level allocation change the delivered accuracy of each knob? → it changes the *cost*
  strongly (p99 −44%, frames kept +76% for costly knobs under a 2-CPU budget, without real-time priority) but the
  delivered-accuracy interaction is small: significant only for CLAHE (+0.86); not enough to rescue costly knobs.
- **SQ5 (new, control):** Can a label-free pipeline statistic (all-frame person coverage) select the
  configuration at runtime? → as a ranking/health signal yes (out of sample); as a selector no better than the best
  fixed policy, and wrong in deep dark (EXP-005).

## 3. Hypotheses for the paper (re-stated; each maps to pre-registered tests)

- **H-A (coverage mechanism):** low light reduces delivered accuracy mainly by removing people from the tracked set,
  not by corrupting the poses of people still tracked. *Supported* (EXP-003 H2).
- **H-B (thresholds are not a remedy):** illumination-adaptive detector/tracker thresholds do not recover the loss.
  *Supported* as a negative result (H3, H4 fail in the predicted-remedy direction; robust in both variants).
- **H-C (resolution × light):** extra input resolution recovers coverage only where the image has signal; its gain
  shrinks to zero in deep dark. *Supported* (H12 pass, H14 true, robust).
- **H-D (enhancement is condition-specific):** no enhancer is beneficial across conditions; the best choice changes
  sign with the lighting level. *Supported* (§2.6); the mechanism (noise vs darkness) is *not* established.
- **H-E (budget):** under a real-time deadline, knobs whose accuracy gain is smaller than their deadline-drop cost
  become net-negative. *Supported* (resolution and Zero-DCE++ reverse; CLAHE/gamma survive). OS allocation shifts
  the cost boundary (H7′) but not enough to move Zero-DCE++ or 960 px above the cheap options (H8′ fail).
- **H-F (coverage proxy, new):** label-free person coverage ranks configurations like AUC. *Supported out of
  sample* (EXP-005 F3); the selector built on it is near-oracle only where it coincides with CLAHE and fails in
  deep dark (F1 pass at 2/3 levels, F2 tie).

## 4. Candidate contributions (ordered by strength, provided EXP-004 confirms H-E)

1. **A measured accuracy-and-cost ledger** for an edge pose-VAD pipeline (YOLOv8n-pose → BoT-SORT → STG-NF) across
   a physically-modelled light sweep × {thresholds, 3 enhancers, 5 resolutions}, with paired CIs, two scoring
   variants and a noise-seed replicate — the first such decomposition for pose-based VAD (LITCHECK-2026-11-05 Q2).
2. **A negative result with a mechanism:** threshold re-calibration does not rescue low-light coverage (not found in
   prior work; LITCHECK-2026-11-05 claim 2).
3. **The resolution × light interaction** and its practical consequence: the closest edge deployment (Huang 2026,
   IEEE ESL) halves resolution to 320 px for 17.7 FPS on a Pi 5 without reporting the accuracy cost; our ledger
   prices that choice (offline vs 640: 480 px −6.3, 320 px −11.3 paper / −7.0 live AUC; Addendum C). Under a Pi's
   tighter budget the delivered optimum may still move toward low resolution — only Pi timing can say.
4. **Offline rankings reverse under deadlines** (EXP-004): the best offline knob (resolution) becomes the worst
   delivered one at 24 fps; learned enhancement is net-negative on a CPU; a condition-aware rule (none / CLAHE /
   gamma by light level, 640 px) beats every fixed configuration on delivered AUC (+0.9 at L2, +1.6 at L3).
5. **A deployment rule ("framework") with its evidence, after EXP-006:** (i) price every knob by *delivered*
   accuracy under the device's frame deadline, not offline accuracy (resolution gains erode/reverse on both
   datasets); (ii) do not tune confidence thresholds for low light (two datasets); (iii) monitor label-free person
   coverage as the health metric (tracks AUC on both datasets in the dark); (iv) treat illumination enhancement as
   a per-site, measured choice — no general rule survived the second dataset; (v) pinning + co-load isolation,
   never SCHED_FIFO inside CPU-quota'd containers.

## 5. What changes in the literature review

- Section "Illumination-aware preprocessing": add the evidence that enhancement for frozen detectors is neutral or
  harmful (LITCHECK-2026-11-04 Q3), and our sign-flip result; Zero-DCE++ cost on a CPU is far from the 1.3 ms GPU
  figure of Abu Awwad 2024 (EXP-004 trace: ≈ 100–185 ms on the VM CPU — **to confirm** with all repetitions).
- New subsection "Input resolution as a compute knob": AdaScale / Chameleon / RAMS (LITCHECK-2026-11-05 Q3) and
  Huang 2026's unreported 320-px trade-off.
- Section "Public-safety surveillance on edge": correct the Huang 2026 description — its 86.2 is the scoring model's
  AUC with the pose source unstated, not an end-to-end edge-pipeline AUC (PAPERS.md, read in full 2026-11-05).
- Gap statement: from "vision and systems optimised separately" to "no work prices the accuracy of vision knobs
  against their CPU/deadline cost for pose-based public-safety analytics, per lighting condition".

## 6. Threats that a Q1 reviewer will raise (and the planned answer)

| Threat | Answer / plan |
|---|---|
| Synthetic low light only | EXP-005: real night footage with person boxes (e.g. LLVIP visible) for the coverage-level claims; camera-gain variant (Addendum C) |
| One dataset (ShanghaiTech), one detector | second detector size (yolov8s-pose) and UBnormal (night/fog scenes, STG-NF checkpoint available) as replication |
| VM timing is not edge timing | all timing claims re-run on the Raspberry Pi 5 (D3, D10) before submission |
| Many comparisons | pre-registered tests listed with verdicts (EXP-003 §1); exploratory addenda labelled |
| Scoring variant sensitivity | report both; claims restricted to findings robust in both (§2.6) |

## 7. Suggested paper outline (Scientific Reports / Springer JRTIP style)

1. Introduction — edge public-safety analytics; low light; CPU budgets; the question of where to spend CPU.
2. Related work — edge VAD pipelines; low-light enhancement for detection; resolution adaptation; OS/CPU allocation.
3. Pipeline and testbed — YOLOv8n-pose/BoT-SORT/STG-NF; light model; enhancers; resolutions; OS configs; VM + Pi 5.
4. Accuracy ledger (EXP-003) — dose-response, coverage mechanism, thresholds, enhancers, resolution, interactions.
5. Cost ledger and delivered accuracy (EXP-004) — per-stage CPU cost, drop rates, delivered AUC, P × O interaction.
6. A budget-aware configuration rule — derived from 4–5; evaluated against fixed configurations.
7. Discussion, threats, Pi 5 confirmation; 8. Conclusion.
