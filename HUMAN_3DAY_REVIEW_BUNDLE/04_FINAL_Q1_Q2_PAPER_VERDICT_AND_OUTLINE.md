# 04 · Final verdict — what to publish, how to frame it, what reviewers will attack

Evidence base: the 57-row audit in `03_CODE_AUDIT_WALKTHROUGH.md` plus the reports. Where I use my own judgement (estimates, venue suggestions), it is marked **[mine]**; everything else is quoted from a repo file. Quartile rankings are **not** in the repo: check SCImago/JCR for the current year before submitting.

---

## 1. The verdict (one paragraph)

**Write Option A (the flagship edge-systems + perception paper) and aim it at a Q2 venue now; Q1 is a credible stretch only if the five fixes in §6 are done.** Option C (negative-results / benchmark paper) is the cheapest second paper and uses the strongest remaining evidence. Option B (LoRaD / MPC / Green-AI) should **not** be written yet: its own pre-registered verdicts are mostly REFUTED / NOT SUPPORTED (E08), and the energy/green/RL parts rest on the five code-level caveats W1–W5. Treat B as "future work" until those are fixed.

Evidence-strength ranking (my call): **A (1st) > C (2nd) > B (3rd)**. Option labels follow the order you gave (A flagship, B LoRaD/MPC/Green-AI, C negatives).

| | Option A — Flagship | Option B — LoRaD/MPC/Green-AI | Option C — Negative results |
|---|---|---|---|
| Core claim | "Offline skeleton-VAD collapses on a real edge CPU; an honest protocol + label-fix front-end + valid reuse + queue guard fixes it" | "Causal multi-scale/interaction flows + two-tier caching + green training" | "What does *not* work for edge VAD, with CIs" |
| Evidence strength | strong (pre-registered, many `*` CIs, Pi 5 measured) | weak/mixed (E08 refuted; W1–W5) | strong (clean negatives, CIs) |
| Novelty type | protocol + mechanism + systems integration | method | empirical / benchmark |
| Risk | "engineering, not novel" ; UBnormal does not replicate the front-end gain | reviewers will find W1–W5 in minutes | "negative results" venue scope |
| Verdict | **Do now (Q2); Q1 if §6 done** | Hold | Spin-off after A |

---

## 2. Option A — flagship systems + perception paper

**Working title [mine]:** *From Offline AUC to Delivered AUC: Making Skeleton-Based Video Anomaly Detection Real-Time on a Raspberry Pi 5.*

### 2.1 Why it wins
1. Every headline has a pre-registered rule, a unit test and a paired-bootstrap CI (audit rows A01–A31).
2. It contains a *mechanism* result, not just a recipe: EXP-020 isolates label dropping as the cause (P1 PASS +16.24 [+11.99, +20.45]; resolution P2 FAIL +0.24 [−1.31, +1.76]).
3. It contains *physical* evidence: Pi 5 sustained live run + power/energy + throttle codes (`0x0`).
4. Negative results make it credible (INT8 on A76, motion vectors, GAU, joint controller).

### 2.2 Headline claims with exact numbers (copy these into the abstract only after re-checking the source)
| # | Claim | Number (source) |
|---|---|---|
| 1 | Offline protocol overstates by ≈ 4 pt (SHT) | S0 77.88 → S4 73.58 (ML0_mps), S0→S4 gap −4.31 [−5.99, −2.65] (`E00` report §1/§4) |
| 2 | Published edge recipes collapse in real time | Huang `RPi` 6.14 FPS, 162.86 ms, queue wait p99 28,682.6 ms; Stock/ML0 640p full-rate 17.56 / 16.60 FPS (`Pi5_huang_repro`) |
| 3 | Small-person blindness is a label artefact | recall `<48 px` 0.4% → 37.0% at 1.00× FLOPs; ML0 − stock@640 +7.75 [+4.63, +10.47]* (`E01`) |
| 4 | Two-stage association recovers intermittent gaps | tracks 2,006 → 1,766; S4 73.78 → 76.82 (+3.04 [+1.44, +4.71]*) (`E01`) |
| 5 | Valid reuse is bit-exact; invalid reuse is quantified | max_abs_diff 0.0, 18.5× / 13.2× fewer pose evals; hidden-state diffs 0.838 / 2.067 (`E00`) |
| 6 | Cache beats fixed schedule at matched cost | Cache − Fixed +2.41 [+1.39, +3.48]* (E03, 12-cell SHT means); +2.56 [+0.77, +4.42]* full-dataset, +7.46 [+5.52, +9.35]* with Guard (`Pi5_suite`) |
| 7 | Real Pi 5 operating point | Fixed-K1 17.20 FPS, p99 46,824 ms, 47.46 [44.2, 50.2] → Burst32 24.01 FPS, 250 ms, 74.19 [65.9, 82.5]; 0.327 → 0.168 J/frame (`Pi5_suite` sustained) |
| 8 | Clean negatives | ORT INT8 vs NCNN FP16 at T=4: 43.77 vs 46.43 ms but 0.2981 vs 0.2697 J/frame; MV gating UB dev −2.25 [−3.99, −0.41]*; joint controller −1.60 [−2.43, −0.82]* (E03/E04, `Pi5_suite`) |

### 2.3 Venues
* Named in the repo: *JRTIP* (venue of LoRaD, Zhang et al., July 2026) and *IEEE Embedded Systems Letters* (venue of Huang et al., 2026), per `PAPER_POSITIONING_AND_CHECKPOINT.md`. Their quartiles are **not** stated in the repo.
* **[mine, unverified scope/quartile — check before use]:** an edge-computing / real-time-vision journal in the IEEE or ACM families; a video-technology journal for the perception half.

### 2.4 Section-by-section outline
1. **Introduction** — the "offline-to-edge illusion"; contributions (protocol, label cause, valid-reuse taxonomy, Pi 5 system). Figure: S0→S4 bars + Huang RPi collapse.
2. **Related work** — STG-NF, LoRaD (JRTIP 2026), Huang et al. (ESL 2026), TrajVAD (ECCV 2026), BoT-SORT/TrackTrack, YOLOv8/26-pose (see `PAPERS.md`, `LITREVIEW.md`).
3. **Problem & protocol** — S0…S4 definition, sentinel, completion landing; E00 decomposition table; tests as evidence. (A01–A12)
4. **Front-end** — label cause (EXP-020), ML0, P2 negative, YOLO26 negative; recall-by-height figure (EXP-011/022). (A13–A14)
5. **Tracking** — two-stage association, 84.2% / 52.3% diagnosis, dev-only 36-cell grid. (A15–A18)
6. **Reuse & scheduling** — valid vs invalid reuse, C1, Guard, Burst, matched-cost methodology, bootstrap. (A08–A10, A20–A26)
7. **Hardware study** — Pi 5 methods (telemetry, throttle, idle floor), stage profile, sustained run, INT8/NCNN. (A27–A31)
8. **Results** — headline table (§2.2), ablation ladder (E05), event metrics incl. **FA/h 11.4 → 28.5**, UB results and limits.
9. **Negative results & limitations** — INT8, MV, GAU, joint controller; UB does not replicate; simulation vs on-device labels (W6).
10. **Conclusion** — what to deploy: ML0 + two-stage + C1 + Guard (+ Burst); what not to.

### 2.5 Reviewer-attack defenses
| Attack | Defense (with evidence) | Residual risk |
|---|---|---|
| "ML0 just detects more normal people → inflated AUC" | `EXP-021` bound AUC ≤ c_A + ½(1−c_A)(1−c_N); ML0 raises anomalous-frame coverage 67.5% → 91.0% (normal 53.4% → 72.5%, E00); event recall@2s 16.4 → 30.3 (Full-Rate 34.9), missed events 67.9 → 51.2; localised AUC 51.5 → 62.2 (Full-Rate 68.0) (E05; synthesis §4 Q&A) | FA/h rises 11.4 → 28.5 — report it; EXP-021 B1 FAIL (12 of 194 points violate the half bound) |
| "UBnormal doesn't show it" | State it: ML0 − stock@640 +0.32 [−3.43, +3.69], vs stock@960 −3.52 [−6.93, −0.43]* (E01 item 4) | real limit; frame as dataset-dependent |
| "VM numbers are not edge numbers" | E02–E05 are labelled VM; Pi 5 sustained live + Pi-calibrated sim + `Pi5_suite` header; INT8/NCNN measured on A76 | Pi sustained AUC CIs wide (73 windows); single board |
| "Simulation, not measurement" | W6: label every delivered-AUC table as measured / Pi-calibrated sim / queue model | needs consistent labelling |
| "Tuned on test" | dev-only 36-cell grid (A16), locked config before run (E03/E05 tests), SHA256 lock (A12) | none seen |
| "Seeds" | E06 uses 3 seeds per trained arm; front-end fine-tune seed count: **check** (`E01` report) | possible single-seed front-end |
| "Engineering only" | mechanism experiment (EXP-020), validity taxonomy, protocol with tests | novelty is integrative |
| "Why not INT8/GPU?" | A76 lacks `i8mm`; INT8 1.06× faster but +11% J/frame (`Pi5_suite` §2); E04: "GPU experiment warranted: no" | scope: one SoC |

### 2.6 Q1 vs Q2 — what separates them here
* **Q2-ready now:** protocol audit + label cause + valid-reuse taxonomy + Pi 5 evidence, clearly labelled.
* **For Q1 [mine]:** a second hardware platform or at least a second Pi run with repeats; UB evaluated on-device with a real NCNN cache (E07 used an MPS proxy for UB ML0); a false-alarm-per-hour operating-point analysis; ≥ 3 seeds for the front-end fine-tune; a deployable (causal) gate for any adaptive claim (W1).

---

## 3. Option B — LoRaD / MPC / Green-AI paper (hold)

**What it would claim:** causal multi-scale dilated flow (MS-CTFlow), interaction-aware graph (IAC-Flow, 616 params), two-tier MPC caching, distillation-free green training, FastOS-RL.

**What the evidence says today:**
| Claim | Evidence | Status |
|---|---|---|
| MS-CTFlow helps | E08 P_A: **REFUTED** (S4 −1.51 [−0.02, −0.01]*; short-event recall +4.9 pp). E09 H1: short-event +6.85 but pooled −3.7; no CIs, clip-level split (W4) | mixed |
| RevMargin | **REFUTED**; reverse term ≡ 0 (A37) | clean negative (→ Option C) |
| IAC-Flow | **NOT SUPPORTED**; SHT FA/h 49.1 → 35.0, UB 101.2 → 167.4, mean +26.1 [+7.5, +46.5]*; ties TrajVAD-P class at 5.56 ms | selective-positive only |
| Two-tier MPC (H2) | Strategy A oracle gate (W1); Medium/Sparse synthetic (W2) | not deployable evidence |
| Green training (H3) | constants hard-coded, truncated training, arch mismatch (W3) | not evidence yet |
| FastOS-RL (E10) | FPS +0.5% … +9.4% single run; AUC column is a formula (W5); no `performance|T4` control | promising engineering note |

**If you still want B:** fix W1–W5 (≈ 1–2 weeks **[mine]**), then target *JRTIP* or *IEEE ESL* (both named in the repo as LoRaD/Huang venues) with a **narrower** claim: "workload-adaptive reuse + OS tuning on Pi 5". Reviewer attack list: oracle gate; synthetic workloads; assumed power; formula AUC; single seed; strawman static baseline.

---

## 4. Option C — negative-results / benchmark angle

**Claim:** "Seven things that look like they should help edge skeleton-VAD, and don't (with CIs)."

| Item | Result | Source |
|---|---|---|
| Bigger / NMS-free detector | YOLOv8s −2.26 [−4.32, −0.43]*; YOLO26n −6.99 [−10.07, −4.40] | EXP-007, EXP-017 |
| P2 stride-4 head | +0.24 [−1.31, +1.76] (stock labels); MLP − ML0 +0.91 [−0.02, +1.81]; 108.71 vs 46.43 ms | EXP-020, E01, `Pi5_suite` |
| Resolution under deadlines | 960 px best offline, worse delivered (−2.4 / −3.1) | EXP-004 |
| Pose filling filler (DAPF) | pooled −1.99 [−3.51, −0.29]* | EXP-018 |
| GAU coupling | −0.78 (CI [−2.16, +0.63]); 5,808 vs 616 params | EXP-019 |
| INT8 on Cortex-A76 / MV gating | 43.77 vs 46.43 ms, +J/frame; MV UB dev −2.25 [−3.99, −0.41]* | E04, `Pi5_suite` |
| LoRaD streaming + RevMargin | A3s ≈ −3.7 pt (ML0 S4 72.89 vs 76.54); reverse term ≡ 0 | E06, E08 |

**Why it's attractive:** all rows have CIs, pre-registered rules and tests. **Risk:** venue scope; combine with the protocol audit (Option A §3) to make it a *benchmark & audit* paper.

---

## 5. What NOT to claim (mine, from the audit)
* "H2 saves 16.9–62.7% pose calls at no AUC cost" — oracle gate (W1).
* "Up to −94% energy on sparse scenes" — synthetic schedule + schedule-driven gate (W2).
* "1,150× faster / −99.98% carbon" — assumed constants, truncated training (W3).
* "FastOS-RL lifts delivered AUC" — formula (W5).
* "IAC-Flow is the best model" — E08 verdict NOT SUPPORTED.
* "Pi 5 delivered AUC 53.32 measured" — queue-model value (W6).

## 6. To-do before submission (ordered by value per hour **[mine]**)
1. **Label every delivered-AUC table** as on-device / Pi-calibrated sim / queue model (W6). *(hours)*
2. **FA/h operating-point table** for A (11.4 → 28.5 is the first thing a reviewer will ask). *(hours; numbers exist in E05 `events.csv`)*
3. **Repeat the Pi 5 sustained runs** (n ≥ 3) or promote the Pi-calibrated full-dataset simulation (with its 2,000-resample CI) as the primary and the live run as validation. *(1 day compute)*
4. **UBnormal NCNN cache + Pi run** so the UB column is not an MPS proxy. *(1 day)*
5. **Check front-end seed count** and, if single seed, retrain ML0 with 2 more seeds (4 epochs on 64,115 images). *(compute-bound)*
6. If you want B later: fix W1–W5 (see `03` §5).

## 7. Suggested figures/tables for A
* Fig 1: pipeline + where the future leaks (S0 vs S4).
* Fig 2: recall by height (EXP-011/022) with ML0 overlay.
* Fig 3: ablation ladder (E05) with CIs.
* Fig 4: Pi 5 latency/queue traces (`fig_c2_queue_traces.png`, `fig_vm_traces.png` in E05).
* Table 1: protocol ladder (E00). Table 2: front-end/tracker contrasts (E01). Table 3: matched-cost ablation (E03). Table 4: Pi 5 sustained (`Pi5_suite`). Table 5: negatives.
