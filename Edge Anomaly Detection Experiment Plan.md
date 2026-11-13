# Fresh Q2 experiment lane: Huang-inspired CPU-only VAD

**Date:** 13 November 2026  
**Status:** Initial plan, not executed. Preserve all previous experiments.  
**Working title:** *Event-Aware Scheduling and Computation Reuse for CPU-Only Skeleton Video Anomaly Detection*

## 1. What we want to do

Reproduce the pipeline, measure its accuracy and CPU bottlenecks, then test **adaptive inference + useful caching** at the same computation budget. Add motion vectors and quantization only after this works.

**Deployment:** Raspberry Pi 5 CPU only; no Hailo/Jetson in the main comparison.  
**Main models:** frozen pose estimator + tracker + frozen anomaly scorer.  
**Target:** ShanghaiTech clean-data AUC x 100 **above 75**, ideally **80+**; prioritize scores actually delivered on time, not only offline scores.

Paper type: **applied edge-AI systems/methods**, targeting Q2. Integration or an AUC threshold alone does not establish publishability.

## 2. Starting point: keep three different results separate

| Evidence | Result | Meaning |
|---|---|---|
| **Huang et al.** | Reported unsupervised AUC: ShanghaiTech 86.2; UBnormal 72.2. | Published reference, not our reproduction. Its pipeline is YOLOv8n-Pose + BoT-SORT + a proposed multi-scale stGCN/normalizing-flow scorer. [H1] |
| **Our EXP-020, offline/paper variant** | ShanghaiTech: stock 68.53; ML0 77.88; MLP 79.35. | ML0 is the existing label-corrected frontend; MLP also changes the architecture. These are historical experiment results. [E1] |
| **Our EXP-020, live replay** | ML0: 73.97 in the s=1 FILL-HOLD setting. | Different scoring/delivery protocol; VM-based timing, not physical Pi performance. UBnormal ML0 was only 54.37 offline. [E1] |

Our code uses **original STG-NF**, not Huang's exact scorer: call it **Huang-inspired** unless faithfully reproduced. [E2]

**Implication:** 75 offline is already achieved; 80 does not beat Huang's 86.2. Our claim concerns resource-constrained delivery. UBnormal remains a separate generalization test.

## 3. Research question and hypotheses

> How should a CPU-only skeleton-VAD pipeline allocate fresh observations and reuse valid computation to improve delivered detection quality under a fixed resource budget?

**H1 - Scheduling:** at matched total CPU cost, event/uncertainty-aware scheduling beats fixed scheduling and a simple motion-only gate.

**H2 - Reuse:** valid computation reuse reduces redundant work without hiding short events; the saved budget can fund fresher or higher-quality observations.

**H3 - Combined system:** the joint policy improves the accuracy-delay-compute trade-off beyond applying each component independently.

Do not start by asserting that Huang is inadequate. Measure where its reproduction succeeds or fails under a declared workload.

## 4. Proposed pipeline, in plain terms

```text
Video -> bounded 1-2 s rolling buffer -> cheap activity/quality checks
                                             |
                                  budget-aware controller
                                  /                    \
                        request fresh window       defer expensive work
                                  |
                     reuse valid cached results
                     compute missing observations
                                  |
                    frozen pose -> tracking -> frozen STG-NF
                                  |
                   timestamped score + freshness/alert status
```

Use scene change, new-person evidence, track quality, observation age and compute budget. Motion schedules work; it does not label anomalies. Keep periodic full-scene checks.

**Preserve temporal meaning:** schedule intact windows at the checkpoint's expected cadence; keep timestamps and tracking order. Do not compress skipped time, introduce learned pose filling or use B-frame interpolation.

Buffering is not a latency guarantee: include collection, queueing and inference. Reject configurations whose processing creates an unbounded backlog.

### What "caching" means here

| Cache | Initial treatment |
|---|---|
| **Real per-frame pose outputs / reusable features** | Reuse the same frame's results across overlapping windows. All strong baselines receive this exact optimization. Never relabel an old pose as a new observation. |
| **Cross-frame neural features** | Separate branch: record source timestamps, validity, refreshes, numerical error and real savings. Unchanged pixels do not guarantee reusable deep features. |
| **Research replay cache** | Precomputed detections speed experiments only. Charge real inference costs; policies cannot inspect unrequested detections. |

STG-NF's per-window normalization can invalidate hidden-state reuse. Require equivalence tests or label a model change. [E2]

## 5. Experiment ladder: one question at a time

Use a new namespace such as `Q2-P1-*`; do not overwrite EXP-009 to EXP-023.

| ID | Question / change | Required output | Training? |
|---|---|---|---|
| **E00 - Audit** | Can we reproduce existing scores, timestamps, normalization and missing-data handling? | Locked configuration, hashes and baseline report. | No |
| **E01 - Quality** | What accuracy can the frozen frontend/scorer reach before CPU constraints? Compare stock and existing ML0; inspect missed small people and tracking gaps. | Offline quality reference; choose settings on development data. | No new training |
| **E02 - Fixed baseline** | What does an optimized fixed pipeline achieve on the actual Pi? Profile decode, pose, tracking and scoring; use the intended runtime. | Stage timings, memory, delivered AUC and sustainable workload. | No |
| **E03 - Core ablation** | Compare fixed, motion-only, adaptive-only, cache-only, independent combination and joint controller. | Matched-cost accuracy/delay curves and failure cases. | No |
| **E04 - Optional optimizations** | Add codec motion vectors, then pose-model INT8, one at a time. | Each component's net benefit after overhead and accuracy loss. | Calibration only for post-training INT8; retraining is separate |
| **E05 - Confirmation** | Lock the best candidate; test full protocol, UBnormal, short events, new entries and sustained Pi operation. | Reproducible results, confidence intervals and limitations. | No |

E03 cache-only means **additional cross-frame reuse**, not a deliberately uncached baseline. Drop caching if no useful stage or net benefit is found.

**First milestone is E00-E02 only.** Read their report before implementing a large controller or launching a parameter sweep.

## 6. How we pursue 75 / 80 without misleading ourselves

Improve observation coverage before enlarging the scorer. Tune resolution and tracking only on development data. Reinvest savings into useful fresh observations, not simply more skipped frames.

Report **offline AUC** and **causal delivered AUC** separately. Alarm-threshold tuning does not improve AUC. Never backdate scores or normalize using future video statistics.

Use identical score-hold/unknown rules. Unscored is not normal: report coverage/staleness and retain difficult frames.

Success is a better accuracy-delay-compute trade-off than the strongest simple baseline; 80+ is a stretch goal, not a promised result.

## 7. Optional training: separate RunPod lane

**Freeze weights within each comparison.** Disclose ML0's previous training and retain stock weights as a control.

Open a GPU branch only when a measured failure justifies it:

| Failure | Smallest training experiment |
|---|---|
| Missed people limit achievable AUC | Longer/gentler frontend fine-tuning, compared with matched stock-label training. |
| Scorer mismatches the chosen input cadence | Retrain a cadence-aware scorer on normal training data; compare a matched retrained baseline. |
| Post-training INT8 damages quality | Quantization-aware training, compared with the same unquantized checkpoint. |

RunPod is for training/extraction, **not Pi latency claims**. Pin versions; smoke-test; repeat promising training comparisons across seeds. Persist checkpoints/logs and export them before termination; storage lifecycles differ. [R1]

Frozen policy sweeps need no rented GPU unless bulk extraction warrants it.

## 8. Evidence needed for the paper

Use video-grouped splits. Earlier test sets are already explored, not untouched holdouts; disclose this and reserve independent confirmation.

Measure **delivered AUC, event recall/misses, delay, false alerts/hour, CPU cost, memory, queue age and measured energy**. Include all overhead; match total cost, not only detector calls. Use paired video-level confidence intervals.

Start with one stream at its recorded cadence, then add load. A freshness rule is not an event-detection guarantee.

For every experiment, save:

```text
Q2-P1-E##_name/
  plan.md          # question, single change, controls, success rule
  config.yaml      # data split, weights, cadence, budgets, runtime
  results.csv      # per-video quality + actual costs
  decisions.jsonl  # requested/completed work, cache use, score timestamps
  report.md        # result, failure cases, next decision
```

**Stop rule:** if the joint controller loses after overhead, simplify or reject it.

## 9. Novelty boundary and sources

Adaptive filtering, video feature reuse and compressed-video pose inference already exist: [Reducto][H2], [DeepCache][H3], [Motion Adaptive Pose Estimation][H4], and [background-feature caching for VAD][H5]. The candidate contribution is **event-sensitive coordination plus controlled end-to-end evidence**. Confirm nearest prior work before claiming novelty.

**[E1] Internal evidence:** `mai-thesis-23-exp020-labels-vs-resolution.md`, final results 10 November 2026; sections "Results: accuracy", "Cost, and what a deadline does to it", and "Caveats". Historical results were read, not rerun for this plan.

**[E2] Internal evidence:** `OVERNIGHT_PLAN.md`, 12 November 2026; baseline distinction, normalization/cache cautions and causal-delivery checks. Its pose-filling proposal is not adopted as the main mechanism here.

[H1]: https://scholar.lib.ntnu.edu.tw/en/publications/deploying-a-skeleton-based-video-anomaly-detection-system-on-edge/ "Huang et al., IEEE Embedded Systems Letters; DOI 10.1109/LES.2025.3618635"
[H2]: https://github.com/reducto-sigcomm-2020/reducto "Reducto, SIGCOMM 2020"
[H3]: https://www.microsoft.com/en-us/research/publication/deepcache-principled-cache-for-mobile-deep-vision/ "DeepCache, MobiCom 2018"
[H4]: https://research.lancaster-university.uk/en/publications/motion-adaptive-pose-estimation-from-compressed-videos/ "Motion Adaptive Pose Estimation from Compressed Videos, ICCV 2021"
[H5]: https://doi.org/10.1145/3805622.3810815 "Efficient Video Anomaly Detection for Edge Devices via Background Feature Caching, ICMR 2026"
[R1]: https://docs.runpod.io/pods/manage-pods "RunPod Pod lifecycle documentation"
