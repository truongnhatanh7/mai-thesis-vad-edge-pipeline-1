# Research loop — orchestration record (started 2026-11-04, M2 Pro)

This file is the lead agent's notebook for the autonomous experiment loop the owner asked for on 2026-11-04
("put yourself in the experiment loop with sub-agents ... end with a clear winning finding that refines the lit
review, hypothesis and RQ"). Append-only. Each iteration: what was asked, which sub-agent lane did it, what came
back, what was decided. Numbers live in the EXP folders and cite run ids / artifact files; this file only links.

## 0. Starting point (from HANDOFF.md, verified by reading every results.md)

- Frozen candidate workload = **edge pose-VAD pipeline**: YOLOv8n-pose → BoT-SORT tracker → STG-NF flow
  (ShanghaiTech). Released poses 85.94 AUC; our edge poses 68.58 (EXP-001).
- EXP-001's key mechanism: the drop is mostly **missed people** — small people get detector confidence 0.26–0.36,
  below the tracker's start threshold, so tracks never start and 1/3 of anomalous frames have no score.
- Anchor CNN-GRU (Salman 2026) not reproducible (≈94 vs 97.6 on Hockey) — EXP-002.
- ShanghaiTech has no dark videos (EXP-001 obs. 5) → low light must be synthetic here; UCF-Crime has real night.

## 1. Research bet (why this, why it is "novel but easy")

Low light lowers detector confidence. In a **tracking-based** pipeline, confidence is not just a score: it is a
**gate** (detector conf threshold, tracker start/match thresholds). So we bet that in edge pose-VAD:

1. Low light hurts mainly by **closing the gates** (coverage/track-continuity loss), not by corrupting poses.
2. Therefore a **zero-compute remedy** — re-calibrating the gate thresholds by lighting — recovers much of the loss,
   while image enhancement (gamma / CLAHE / Zero-DCE++) recovers some of it **at a CPU cost**.
3. Dropped frames (deadline misses on a CPU) attack the **same currency** (track continuity), so preprocessing
   cost and OS/thread allocation interact through one mechanism. This is what makes the P × O factorial
   non-trivial and gives the "framework": an illumination- and budget-aware configuration of the pipeline.

Novelty claim to test: prior low-light edge work (Abu Awwad 2024, Megalingam 2026, Kang 2026 gate, Turbo 2022) treats
low light as an *input-quality* problem fixed by enhancement; none analyses the *decision-threshold / tracking* path
or couples it with CPU deadlines. Easy because: one frozen public model, public data, cached detections let us
replay trackers/thresholds/drops in minutes instead of re-running the detector.

Efficiency trick (key to doing this cheaply): run YOLO once per (darkening × enhancement) at conf 0.01 and cache all
post-NMS detections. Any threshold ≥ 0.01 is then an exact filter of that cache (greedy NMS keeps a box only if no
higher-scoring box suppresses it, and higher-scoring boxes survive the same filter). Tracker settings and
deadline-drop patterns are replayed on the cache in ~1–2 min per config.

## 2. Experiment map

| Exp | Question | State |
|---|---|---|
| EXP-003 | Synthetic low light on ShanghaiTech: dose-response, mechanism (coverage vs pose quality), remedies (enhancement vs gate re-calibration) | planned |
| EXP-004 | CPU budget: per-stage cost on Mac under OS configs; drop-on-late replay → delivered AUC; P × O interaction | planned |
| EXP-005 | Real low light check (real dark footage, detector-level) — only if EXP-003 shows a clear effect | candidate |

## 3. Sub-agent lanes

Sub-agents never commit, never touch another lane's files, write everything to their EXP `log.md` and return a
≤ 300-word summary. Model: Claude Sonnet 5.5 (owner suggested "Grok 4.7"/"Sonnet 5.5"; Grok 4.7 is not available
in this environment, Grok 4.6 is).

## 4. Iteration log

### Iteration 0 — 2026-11-04 23:20 (lead)
- Read HANDOFF, SETUP, README, DECISIONS, all results.md, LITREVIEW, PAPERS.
- Machine is **M2 Pro (6P + 4E cores, 32 GB)**, not the handoff's M4 Pro → all timings tagged `machine=mac-m2pro`;
  not comparable with the M4 Pro numbers in EXP-001/002.
- Started `uv sync --extra pose`, `brew install ffmpeg`; data root `~/thesis-data` (symlink `experiments/data`).
- Wrote this file, DECISIONS D13–D15, EXP-003 plan (pre-registration, frozen 23:40) before any run.
- Launched 3 background lanes (Claude Sonnet 5.5):
  - **data lane**: ShanghaiTech test frames (HF mirror, sha-checked), STG-NF repo + released poses, YOLO weights,
    pytest, R1 reproduction check on the M2 Pro.
  - **code lane**: `preprocess/darken.py`, `preprocess/enhance.py` (gamma, CLAHE, Zero-DCE++ torch+ONNX),
    `pose/detcache.py` (conf-0.01 detection cache), `pose/replay.py` (tracker replay with gate c and drop mask),
    scoring + paired bootstrap, tests, WALKTHROUGH.
  - **literature lane**: novelty check of the "gates, not pixels" claim → `LITCHECK-2026-11-04.md`.
- OS knobs available on macOS (checked `man taskpolicy`): thread counts, ORT spinning, and **QoS clamp**
  (`taskpolicy -c background` restricts a process to the 4 efficiency cores). Plan for EXP-004: P-cores vs
  E-cores (E-core cluster = a slower, Pi-class proxy where deadlines bind), threads, and isolating a co-load
  (ffmpeg re-encode as "recording") onto E-cores. No pinning/SCHED_FIFO/cgroups on macOS → those stay for the Pi.

### Iteration 0b — 2026-11-04 23:25 (owner input)
- Owner: OS knobs must be tested in **Lima VMs**, not on the Mac host. → D16. The macOS `taskpolicy` plan above is
  dropped. EXP-004 OS arm = Linux knobs in a new VM `edge-os` (threads, spinning, pinning, SCHED_FIFO/nice,
  cgroup `cpuset`/`cpu.max`, co-load isolation). `cpu.max` doubles as a way to emulate a Pi-class CPU budget so
  deadlines actually bind. Accuracy stays on the host (detection cache); timing traces come from the VM.
- Existing VMs `default`, `ebpf-lb` (owner's, stopped) are left untouched.
- Launched **VM lane** (Sonnet 5.5): create `edge-os`, install env, verify each knob works, smoke-time one model.

### Iteration 1 — 2026-11-04 23:26 (literature lane returned → `LITCHECK-2026-11-04.md`)
- Novelty verdict: **open as far as found** for (a) illumination-adaptive detector/tracker thresholds, (b) low-light
  or detector-threshold effects on pose-VAD AUC, (c) a CPU-only factorial of preprocessing × OS under deadlines.
  Known/established: detector miscalibration under shift; scene-adaptive ByteTrack threshold (arXiv 2312.01650);
  low-light MOT (LTrack/LMOT, arXiv 2405.06600); enhancement neutral/harmful for frozen detectors
  (arXiv 2409.14461); frame-rate/deadline harms tracking.
- Positioning consequence: the contribution must be the **mechanism + cost-aware remedy in a full pipeline**
  (gates → coverage → AUC, and drops → track continuity → AUC), not "adaptive thresholds" per se.
- Red flags carried into analysis (no plan change needed — already covered by H2 and the full c sweep):
  (1) pose quality may dominate instead of gates → H2 tests it; (2) STG-NF scores bad poses as anomalous, so a
  lower gate can raise coverage but lower AUC → report coverage and AUC side by side, plus normal-frame coverage;
  (3) Huang 2026 (IEEE ESL) read from abstract only → **owner to fetch PDF via RMIT library**.

### Iteration 2 — 2026-11-04 23:37 (code lane returned)
- Implemented darken / enhance (gamma, CLAHE, Zero-DCE++ re-implemented, 10,561 params, strict-load of official
  weights, torch–ORT max |Δ| 4.8e-7) / detcache / replay / score_dir / paired bootstrap. pytest 66 passed, 10 skipped.
  Details + Zero-DCE++ commit/sha/licence (CC BY-NC 4.0, research only) in EXP-003 `log.md`.
- Lead review of `darken.py` and `replay.py`: matches plan §4.1/§4.3. Clarifications logged in EXP-003 log 23:37.
- Waiting on data lane (frames + STG-NF + R1) before the G0 check and the cache sweep.

### Iteration 3 — 2026-11-05 06:46–07:42 (lead, after owner said "continue")
- Sub-agents for data and VM died on "connection stalled" (long blocking commands). Lead finished their work;
  new rule for lanes: no foreground command > 2 min, long jobs in tmux, log after each step.
- ShanghaiTech: first download corrupt (6/7 sha fail; two concurrent writers) → clean aria2c re-download, 7/7 OK,
  107 videos / 40,791 frames. VM: Ubuntu unattended-upgrades was burning a core → disabled.
- EXP-003: MPS device check PASS; **G0 PASS (68.53 vs 68.58)** — caching + replay is exact enough and 25× cheaper.
- **First real finding (clean light):** lowering the gate 0.25 → 0.05 admits +50% detections but only +2% tracked
  poses, AUC 68.53 → 68.56. In daylight the gate is not what loses people; detector recall at 640 px is
  (consistent with EXP-001's +3.1 at imgsz 864). The "gates" bet now hinges on the dark levels (H3 at L2/L3).
- Wrote EXP-004 plan (frozen 07:40 equivalent; timing subset, F = 24 fps verified from training videos, O0/O1 with
  ffmpeg co-load, cgroup budgets B4/B2, randomised interleaving, host-load exclusion rule).
- Code lane 2 (Sonnet 5.5) building deadline simulator, re-indexed replay, VM live-timing harness.
- Candidate pivot if H3 fails: the budget knob that matters may be **input resolution** (imgsz), not the gate —
  then the framework becomes "spend the CPU that OS tuning frees on resolution vs. enhancement". Would be a new,
  separately pre-registered exploratory arm, not a silent change.

### Iteration 3 — 2026-11-05 06:46–07:42 (lead, after owner said "continue")
- Sub-agents for data and VM died on "connection stalled" (long blocking commands). Lead finished their work;
  new rule for lanes: no foreground command > 2 min, long jobs in tmux, log after each step.
- ShanghaiTech: first download corrupt (6/7 sha fail; two concurrent writers) → clean aria2c re-download, 7/7 OK,
  107 videos / 40,791 frames. VM: Ubuntu unattended-upgrades was burning a core → disabled.
- EXP-003: MPS device check PASS; **G0 PASS (68.53 vs 68.58)** — caching + replay is exact enough and 25× cheaper.
- **First real finding (clean light):** lowering the gate 0.25 → 0.05 admits +50% detections but only +2% tracked
  poses, AUC 68.53 → 68.56. In daylight the gate is not what loses people; detector recall at 640 px is
  (consistent with EXP-001's +3.1 at imgsz 864). The "gates" bet now hinges on the dark levels (H3 at L2/L3).
- Wrote EXP-004 plan (frozen 07:40 equivalent; timing subset, F = 24 fps verified from training videos, O0/O1 with
  ffmpeg co-load, cgroup budgets B4/B2, randomised interleaving, host-load exclusion rule).
- Code lane 2 (Sonnet 5.5) building deadline simulator, re-indexed replay, VM live-timing harness.
- Candidate pivot if H3 fails: the budget knob that matters may be **input resolution** (imgsz), not the gate —
  then the framework becomes "spend the CPU that OS tuning frees on resolution vs. enhancement". Would be a new,
  separately pre-registered exploratory arm, not a silent change.

### Iteration 4 — 2026-11-05 09:40 (lead)
- Interim EXP-003: light costs little in the realistic range (L1 −0.6, L2 −1.5 AUC), then a cliff at L4 (−14,
  coverage 0.67 → 0.19). Gate re-calibration ≤ +0.3 everywhere → **the "gates, not pixels" bet is failing** (H3).
  Gamma hurts at L1 (−2.3, noise amplification); CLAHE hurts on clean (−0.4).
- Pivot (pre-registered as exploratory Addendum A before any data): input resolution as the compute knob.
  Emerging thesis: *in edge pose-VAD the accuracy bottleneck is the detector's pixel budget, not illumination
  within the CCTV-realistic range; enhancement is net-negative there, so CPU freed by OS tuning should buy
  resolution, and enhancement only pays below a luma cliff.* To be confirmed by L3, E3, resolution, EXP-004.

### Iteration 5 — 2026-11-05 10:30 (lead)
- EXP-003 interim verdicts (log 09:58–10:30): **H1 PASS** (L3 −8.1), **H2 PASS** at L2/L3 (coverage loss explains
  59–84% of the drop), **H3 FAIL** (cross-fitted gate −0.06 / −1.69: lower gates add noisy poses that raise
  false alarms). **H12 PASS**: imgsz 480 → 960 = 62.2 → 71.7 AUC on clean frames (+9.5, larger than L0→L3 darkness).
- Working thesis: *the edge pose-VAD pipeline is limited by the detector's pixel budget, and low light acts mostly
  by shrinking the people it can find (coverage), not by fooling thresholds; enhancement amplifies noise (gamma
  −2.3 at L1).* Pending: E2/E3 at L2/L3 (H4–H6), resolution at L2/L3 (H13/H14), EXP-004 timing (cost of
  each knob in CPU ms) → "accuracy per CPU millisecond" ledger and the P × O interaction.
- Ops fixes: thread oversubscription (load 70) → 1 thread/job; Zero-DCE++ moved to MPS (deviation, parity 1 LSB).
- 10:35 Code lane 2 returned (EXP-004 toolchain, 99 tests). Lead caught an input-shape mismatch between timing ONNX
  (640×640) and accuracy caches (640×384) → EXP-004 Amendment 1 (rect ONNX per resolution, 30 cells, systemd
  scopes for co-load isolation and 2-CPU budget). Lane 2 resumed to implement + dry-run.

### Iteration 6 — 2026-11-05 20:16–22:15 (lead)
- All 33 EXP-003 caches done and scored (driver stopped early on SWEEP_DONE; rerun). Found and fixed an STG-NF
  loader bug (mutable default `global_pose_data`) that had mislabelled L4_E1/L4_E3 as unscorable; fix verified not
  to change any previously scored config. `results.md` drafted.
- **Final EXP-003 picture (paper variant):** G0, H1, H2, H5, H12 pass; H3, H4, H6, H13 fail; H14 true.
  Strongest new facts: (1) darkness alone costs nothing (+0.33), sensor noise costs −1.81 at α 0.1;
  (2) enhancer effect flips sign with noise (CLAHE +3.2 with noise, −1.5 without); (3) resolution +9.5 AUC
  (480→960) in clean/moderate light but +0.3 at L3; (4) gates never help.
- **Candidate headline:** "Edge pose-VAD under low light is limited by noise-driven loss of detector coverage;
  the right remedy depends on the condition (noise level) and the CPU budget — gates no, CLAHE under moderate noise,
  resolution when there is signal, nothing below a cliff." → framework = condition- and budget-aware choice of
  {enhancer, resolution}; EXP-004 supplies the costs and the OS lever.
- Addendum B (noise-seed replicate + combinations) launched; EXP-004 matrix waiting for an idle host (owner's
  `node`/vtsls language-server processes ≈ 2.8 cores exceed the 30% exclusion rule).

### Iteration 7 — 2026-11-05 22:12 (literature lane 2 → `LITCHECK-2026-11-05.md`)
- Claim status: (1) noise-not-darkness known in spirit (PoseBench, ExLPose, Hong BMVC'21) — new only as a paired,
  CI-backed arm in a full pose-VAD pipeline; (2) coverage loss + gates don't help: nothing found → most novel;
  (3) enhancement sign flip with noise on/off: controlled test new, "enhancement amplifies noise" known;
  (4) resolution gain known (AdaScale, RAMS); resolution × light interaction only hinted; (5) condition- and
  budget-aware selection for CPU pose-VAD not found, incremental until EXP-004 has numbers.
- Closest: physics-based light sweep for person detection (arXiv 2605.22455), RAMS resolution tiers on a Pi 5
  (arXiv 2606.14716), brightness × noise grid (PMC10856852). Synthesis method is NOT a contribution.
- Reviewer risk: PoseBench says resolution does not improve *robustness* — consistent with our H14 (resolution gain
  vanishes at L3); frame (4) as "resolution buys coverage only where there is signal", not as a robustness fix.
- Action: claim (1) holds only for a no-gain sensor model → add a **camera-gain (auto-exposure) variant** to make it
  robust (Addendum C, after Addendum B results), and keep EXP-005 real-dark validation on the list.

### Iteration 8 — 2026-11-05 22:20 (owner supplied Huang 2026 PDF; read in full)
- Huang et al. 2026, IEEE ESL 18(3):178–181, doi 10.1109/LES.2025.3618635: multi-scale stGCN normalizing flow +
  hybrid NLL/reverse-matching loss; **86.2 AUC on ShanghaiTech is the scoring model's AUC; the paper does not state
  it was computed on its own YOLOv8n-Pose + BoT-SORT poses** (compared in a table against STG-NF etc., which use the
  released poses) → resolves the EXP-001 "86.2 vs our ≈ 68" concern: likely different measurements.
- Pi 5: default YOLO 640×640 = 3.2 FPS; NCNN + **320×320 = 17.7 FPS**, "some degradation in object detection quality"
  but **no AUC reported at 320**; YOLO + MOT dominate latency; 6.3 / 6.07 W, 51 / 50.5 °C. No low light, no OS
  tuning. Tables III/IV are images (values not extracted).
- Use in paper: the closest edge pose-VAD deployment trades resolution for frame rate blind; our resolution arm
  (480 = −6.3 AUC vs 640) quantifies that cost → added imgsz 320 to Addendum C. Also: their 3.2 FPS at 640 means
  deadline drops are the norm on a Pi 5 → EXP-004's delivered-AUC question is central, not marginal.
- Darken gain variant implemented + tested (11 passed). Addendum C frozen (gain variant + imgsz 320).

### Iteration 9 — 2026-11-06 00:15 (lead)
- Addendum B: noise-seed replicate holds for both key contrasts (configs move ≤ 0.71); CLAHE + 960 at L2 = 72.16
  (best of all, above clean@960); gamma + 800 at L3 does not stack.
- **Live-variant check changed the claim set:** robust in both scoring variants = realistic-light flatness + cliff,
  gates never help, resolution × light interaction, gamma hurts at moderate dark, CLAHE best moderate-light
  enhancer (small in live). **Not robust:** "noise not darkness" (live: both n.s., opposite signs) → dropped as a
  claim. results.md §2.4–2.6 rewritten accordingly.
- EXP-004 matrix running since 23:52; first reps valid (host outside VM 15–17% < 30%). ETA ≈ 05:00.

### Iteration 10 — 2026-11-06 04:15 (lead) — EXP-004 main results
- Matrix 150/150 valid. Cost ledger (VM): gamma 0.7 ms, CLAHE 3 ms, Zero-DCE++ ≈ 100–190 ms, detector 18/30/49/65 ms
  at 480/640/800/960.
- **Winning finding candidate:** at 24 fps the offline ranking reverses — 960 px (+2.0 offline) delivers −2.4 (4 CPU)
  / −6.5 (2 CPU) because drops break tracks; cheap condition-specific enhancement keeps its gain (CLAHE at L2 67.06,
  gamma at L3 63.41); Zero-DCE++ net-negative everywhere (−3 to −12). Condition-aware choice at 640/O0-B4:
  L0 none 66.19, L2 CLAHE 67.06, L3 gamma 63.41 vs fixed "none": 66.19 / 66.13 / 61.78.
- OS: SCHED_FIFO escaped cgroup v2 cpu.max (O1-B2 ≡ O1-B4) → Amendment 2 (O1n) queued. With 4 CPUs the tuned bundle
  is slower (p50 41 vs 34 ms) and makes costly enhancement worse (interaction −1.1 to −2.1).
- Two driver bugs caught before use (light label overwrite; earlier sign). Queue: Addendum C → A2 timing → H8′.

### Iteration 11 — 2026-11-06 05:25 (lead)
- Addendum C: Huang's 320-px Pi 5 setting costs −11.3 (paper) / −7.0 (live) AUC vs 640 offline; camera gain makes
  noisy dark frames score *above* clean (coverage 0.699 vs 0.674); L3 gain + CLAHE 65.16 = best deep-dark config.
- **New unifying observation:** AUC tracks anomalous-frame coverage across all 45 configurations (ρ 0.95) and even
  label-free all-frame coverage (ρ 0.93), which picks the best knob at 5 of 6 light levels → candidate label-free
  runtime selector (framework). Post-hoc → becomes EXP-005's pre-registered hypothesis.
- 05:40 EXP-005 (coverage selector, out-of-sample on new light levels X1–X3) pre-registered; analysis code + tests
  written before data; caches chained after EXP-004 final analysis.
- 05:40 EXP-005 (coverage selector, out-of-sample on new light levels X1–X3) pre-registered; analysis code + tests
  written before data; caches chained after EXP-004 final analysis.

### Iteration 12 — 2026-11-06 09:45 (lead) — EXP-004 complete, EXP-005 complete
- EXP-004 Amendment 2: SCHED_FIFO was the harmful part of the tuned bundle; pinning + co-load isolation halves p99
  under a 2-CPU budget (H7′ pass) but the delivered-AUC interaction with Zero-DCE++ is n.s. (H8′ fail, +2.3);
  significant only for CLAHE (+0.86).
- EXP-005 (out-of-sample, pre-registered): coverage ranks configs on unseen light (ρ 0.87–0.92, F3 pass); selector
  near-oracle at 2/3 levels (F1 pass) but ≡ always-CLAHE (F2 tie); deep-dark failure replicates (gamma wins with the
  lowest coverage).
- Loop closes here: all planned experiments done; remaining work needs new hardware (Pi 5) or new data (real dark).

### Iteration 13 — 2026-11-06 09:40 (owner: "execute plans without waiting") → D19
- Next step chosen without asking: replication on a second dataset. **EXP-006 UBnormal** (rendered people, 29
  scenes, real in-scene night/fog/smoke/rain) pre-registered (U0–U5) before any data; official 16 GB zip
  downloading (tmux `ub-dl`); code lane 3 (Sonnet 5.5) adapting the STG-NF wrappers + R0 reproduction (71.8).

### Iteration 14 — 2026-11-06 20:40 (lead) — EXP-006 UBnormal replication done
- Google Drive quota → HF mirror (sha256 + official-prefix match); ZIP64 needed Python extraction.
- Edge pipeline near chance on UBnormal (55.8 vs 71.8 released poses; weather 36.4). Replicated: thresholds don't
  help, lower resolution costs, coverage tracks AUC in the dark, deadlines erode the resolution gain. **Not
  replicated: the light-gated enhancer rule** (gamma/CLAHE no better than none on UBnormal). Headline in
  RESEARCH_SYNTHESIS.md narrowed to the cross-dataset claims.
- 20:50 EXP-007 (stronger front end, YOLOv8s-pose, both datasets) pre-registered and launched under D19.

### Iteration 15 — 2026-11-07 07:20 (lead) — EXP-007 done
- YOLOv8s-pose is *worse* than 8n on ShanghaiTech (−2.3, lower coverage 0.569 vs 0.674) and equal on UBnormal →
  coverage, not model size, decides accuracy (V4 ρ 0.8–1.0). CLAHE is the only enhancer effect robust across
  detectors and datasets; the gamma-in-deep-dark rule is not. Next (D19): offline-vs-delivered rank analysis,
  slower-device simulation, delivered-AUC model (owner's novelty ideas 1–3).

### Iteration 16 — 2026-11-07 09:30 (lead) — EXP-008 (no-hardware analyses)
- Offline vs delivered rank agreement τ 0.72 on the VM → **0.16 on a 4× slower device**; offline-best (960 px) is
  never the delivered winner on ShanghaiTech and loses on UBnormal from 2× slower. At 2× slower the winners are
  the cheap options (640 px / CLAHE / gamma). Above ≈ 75% drops everything collapses (A2 fail on monotonicity).
  Linear delivered-AUC model does not transfer across datasets (A3 fail).

### Iteration 17 — 2026-11-07 13:35 (owner idea: codec-style I/P/B pose prediction) — literature lane 3
- `LITCHECK-2026-11-07-pose-propagation.md`: keyframe detection + propagation, adaptive keyframes and
  compressed-domain pose propagation are established (closest: compressed-video pose estimation with I-frame
  detection + P-frame joint warping, IEEE TMM 2022; AdaHPE adaptive keyframes for edge pose, IEEE IoT-J 2025,
  abstract only; Key Frame Proposal Network, ECCV 2020). Not found: I/P/B pose prediction inside a pose-based
  VAD pipeline, evaluated as delivered accuracy under a deadline, nor a test of whether predicted poses hide
  anomalies → novelty is the application + evaluation, not the propagation algorithm.
- Lead verified in our STG-NF fork: `utils/pose_utils.py: is_seg_continuous(..., missing_th=2)` — a window is kept
  only if ≤ 2 of its frames are missing; no imputation. So track gaps > 2 frames remove every window spanning them,
  which is the mechanism pose filling would repair.
- Standing by for the owner's go on EXP-009 (decomposition + I/P/B pose filling).
- 15:35 Owner: "run the experiment now" → EXP-009 plan frozen (decomposition D1/D2; B/P pose filling offline at
  G 2–12; deadline rescue at s 1/2/4; keyframe schedule k 2–4; risk R). Code lane 4 (Sonnet 5.5) implementing + running.

### Iteration 18 — 2026-11-07 16:30 (lead) — EXP-009 (owner's I/P/B pose-filling idea) done
- Decomposition: 96% of the 17-pt edge gap = which people/frames are detected (D1 69.23 vs ref 85.94); pose quality ≈ 0.7.
- Pre-registered: P1 pass (offline B +0.91 [+0.03, +1.81]), P2 pass only by construction, P3 fail (keyframe k2 −2.7),
  P4 fail (causal P ≥ B).
- **Exploratory fair test: causal pose prediction of dropped frames beats skip-and-reindex by +1.4 to +5.6 delivered
  AUC (live), significant in 5/6, on both datasets** — the first method that improves delivered accuracy on both.
  Needs a pre-registered confirmation (e.g. new slow-down factors / seeds / L3 / 8s front end) before claiming.

### Iteration 19 — 2026-11-07 17:00 (lead) — EXP-010 planned, on hold
- EXP-010 (confirmation of EXP-009's fill-vs-skip result) pre-registered: 6 groups (SHT L0/L2/L3 8n, UB L0 8n,
  SHT/UB L0 8s), new slow-downs s 3/6, new seed, primary comparison FILL-HOLD vs SKIP with the **same** hold-last
  scoring (removes EXP-009's confound), pass = ≥ 4/6 significant at both s. **Not run — waiting for owner (D20).**

### Iteration 20 — 2026-11-07 17:35 (research lane: squeezing accuracy/speed) → `RESEARCH-squeeze-performance-2026-11-07.md`
- Ranked: (1) P-fill with provenance (in-house evidence), (2) fine-tune YOLOv8n-pose on AlphaPose pseudo-labels for
  small people (only item that adds small-person information; est. +1–3, could be 0), (3) NCNN/INT8 to afford
  rect 800 px within the deadline (cannot buy 960), (4) WiSE-FT / soups as polish (< 1 AUC expected),
  (5) provenance-gated scoring. **Weight-space verdict: a polish/safety wrapper around a fine-tune, not a lever** —
  averaging models that never saw labelled tiny people cannot make them detect tiny people. Hypothesis to check:
  COCO has no keypoint labels for people < 32² px → recall cliff by person height (cheap check from caches).
- Proposed next experiments (not planned/frozen yet): recall-vs-height + INT8/NCNN timing; pseudo-label fine-tune;
  WiSE-FT/soup. Awaiting owner.

### Iteration 21 — 2026-11-07 17:45 (lead) — EXP-010 confirmation done
- r (8s/8n detector cost, VM) = 2.507. All C1–C4 pass by rule; claim formally allowed. Lead review: one of C2's four
  passes (SHT-L0-8s s6, 91% drops, ~30 windows) is degenerate → honest reading: robust at 3× (5/6), robust at 6× on
  ShanghaiTech-8n (3/3), not significant on UBnormal-8n, no rescue above ~90% drops. Lesson: pre-register a
  degeneracy rule (minimum scorable windows) next time.
- 18:10 Research lane started: AlphaPose-class (top-down) pose on edge CPU + OS orchestration →
  `RESEARCH-alphapose-on-edge-2026-11-07.md`. Obsidian notes updated (new note 17; index, scorecard, discussions,
  next steps).
- 18:20 AlphaPose-on-edge lane returned: RTMPose(-s/-t) is the best light top-down pose model (Pi 5 RTMPose-s
  16.7 ms/crop, one measured number); bottleneck = detector (YOLOX-X ~282 GFLOPs), consistent with the 96% finding;
  lead's "cascade" idea corrected (needs its own small-person detection pass). Proposed: oracle-boxes + RTMPose
  (no Pi), nano-detector + keyframe sweep (no Pi), Pi bake-off. Obsidian note 17 §C updated.

### Iteration 22 — 2026-11-07 18:35 (owner: "do all experiments like your suggestions") → D21
- Plans frozen 18:30 for EXP-010b (seed 2 + degeneracy rule ≥ 200 windows), EXP-011 (recall by person height),
  EXP-012 (INT8/NCNN to buy 800 px), EXP-013 (pseudo-label fine-tune, train split only), EXP-014 (WiSE-FT, α chosen on
  held-out training videos), EXP-015 (oracle boxes + RTMPose-s/t), EXP-016 (nano person detector + RTMPose + K=3).
- Lanes: code lane 4 (010b, 011), lane B (012, VM timing first), lane C (013→014), lane D (015→016). Coordination:
  heavy host jobs wait for ~/thesis-data/locks/vm_timing_done (written by lane B after its timing).
- 18:50 EXP-011 done (code lane 4): recall by person height at 640 L0 = 0.1% (<32 px), 0.7% (32–48), 5.7% (48–64),
  61% (64–96), 96% (96–160), 99% (≥160) → the edge detector is blind below ~64 px (R1 pass); darkness hits the
  smallest most (R3 pass); R2 fail under the pre-chosen absolute reading (gain largest at 64–96 px). EXP-010b code
  ready, waiter armed for the VM-timing lock; lead will run its analysis after the replays.

### Iteration 23 — 2026-11-07 19:56 (Cursor process crashed ~19:0x; lead recovering)
- All sub-agent lanes (4, B, C, D) lost. tmux survived: waiters/chains for EXP-010b (e10b), EXP-013→014 (e13-chain),
  EXP-015 (e15), EXP-016 (e16). EXP-012 VM timing had **finished** (24/24 runs ok, 18:36–19:03) but its lane died
  before writing the lock → everything waited ~1 h. Lead summarised timing (NCNN FP16 @800 = 18.9 ms = 0.57× ORT FP32
  @640; INT8 @800 0.61×), released the lock 19:57, and launched EXP-012's accuracy caches (tmux e12-cache: NCNN-FP16-800
  and INT8-800 at L0/L2, INT8-640, plus a same-path ORT-FP32-640 control).
- Remaining after the chains finish: verdicts + results.md for EXP-010b, 012, 013, 014, 015, 016 (lead, or resumed lanes).
- 22:20 Owner asked about YOLO26. Installed Ultralytics 8.4.201 contains YOLO26 configs (pose, p2). EXP-017 (YOLO26n-pose
  vs YOLOv8n-pose: recall by height, AUC SHT/UB, NCNN/ORT timing, delivered AUC) pre-registered; queued after the
  running caches (host contention), VM timing on an idle host. EXP-010b written up (S1/S2 fail; direction replicates
  in all 11 non-degenerate cells). EXP-012 scoring started (tmux e12-score).

### Iteration 24 — 2026-11-08 07:30 (lead) — overnight results
- **EXP-012 all pass:** NCNN FP16 @800 = 0.57× ORT FP32 @640 cost, 70.65 vs 68.53 offline, delivered 67.57 / 67.13 vs
  66.19 / 62.66 at s 1 / 2. Same-path control 68.54 = baseline.
- **EXP-015 O1/O2 pass:** with oracle boxes+tracks, RTMPose-s 86.12, RTMPose-t 85.91, even YOLOv8n-pose crops 84.99 → the
  edge loss is detection/tracking, not keypoints.
- **EXP-016 T1 pass:** YOLOv8n *detection* @640 + RTMPose-s top-down = **80.52 paper / 76.23 live (+11.99 [+7.71, +16.24])**,
  coverage 0.915 (vs 0.674); in the dark (L2) only 68.67 (+1.6). K=3 loses 4.1 (T2 fail at 640). **n960 rows invalid**
  (MPS crash after 4/107 videos, driver scored incomplete caches) → lane E rebuilding on CPU + completeness check; T3
  timing pending.
- EXP-013 training at epoch ~23/30 (≈ 27 min/epoch). EXP-017 (YOLO26) had NOT been auto-queued (lead's error) → started
  now in lane E. EXP-010b written up earlier (direction replicates, significance rule fails).
- 07:30 Correction to iteration 24: EXP-016's L2 numbers ("68.67, +1.6 in the dark") are INVALID — the L2_n640 cache had
  only 94/107 videos (MPS crash). Valid: L0_n640 (107/107, +11.99). L2_n640, L0_n960, L2_n960 being rebuilt on CPU.

### Iteration 25 — 2026-11-08 09:10 (lead) — status + recovery
- EXP-013 training finished (25 epochs, early stop, best epoch 20); our wrapper crashed on Ultralytics' stripped
  best.pt → chain had stopped at 08:44. Fixed (fallback to results.csv, `--summarize-only`, test), chain resumed at
  evaluation 09:05.
- EXP-016 CPU rebuild: L0_n960 done, L2_n640 56/107, L2_n960 79/107. EXP-017 (YOLO26): code/exports/parity done, caches
  running (SHT L0 80/107). VM timing for EXP-016 T3 and EXP-017 Y4 waits for a quiet host (after EXP-013/014).
- Estimate (no further crashes): all running experiments done ≈ 15:00–16:00.

### Iteration 26 — 2026-11-08 10:50 (lead)
- **EXP-013:** F1 pass, **F2 pass (+11.75 [+7.08, +16.40] → 80.28)**, F3 fail (UBnormal −4.12), L2 +9.0 sig, L3 −1.9 n.s.
  → per-site fine-tune ≈ top-down RTMPose (80.5) at zero extra inference cost; site adaptation, not general.
- **EXP-014:** naive WiSE interpolation collapsed (0 detections at α 0.25–0.75; BatchNorm-statistics mismatch); the chain
  had picked α* = 0.5 from all-zero ties → stopped, Amendment 1 (BN recalibration on 512 training images) frozen before any
  test result, verified on held-out training frames (0 → 20 people), chain restarted at the WiSE stage.
- EXP-016 repaired caches complete (L2_n640, L2_n960 107/107); scoring in lane E. EXP-017 caches: SHT L0 done, L2 96/107,
  UB 206/211 (L3 pending).
- 11:40 Context near limit → lead's working memory written to `STATUS-NOW.md` (running jobs, owners, remaining steps,
  pointers). EXP-014 evaluation running (e13-chain); EXP-016 scoring + EXP-017 caches (L3 39/107) running; lane E resumed.

### Iteration 27 — 2026-11-08 12:25 (owner: prioritise the DAPF framework; add Gated Attention Units to STG-NF) → D22
- EXP-018 (DAPF: filling-aware STG-NF with provenance channel + learned GRU predictor; scheduler later; pooled test over
  cells/seeds to fix EXP-010b's power issue) and EXP-019 (STG-NF-GAU, 3 seeds vs standard; G1 ≥ +1.0 on released poses)
  frozen 12:20; lane F launched. EXP-017 deprioritised (timing/delivered deferred).

### Iteration 28 — 2026-11-08 16:40 (lane F done)
- **EXP-019 G1 fail:** STG-NF-GAU (5,808 params) −0.78 vs standard (616) on released poses; G2/G3 pass (1.93 ms/window).
  Fancier coupling subnet does not help → GAU not used in DAPF.
- **EXP-018 D1–D3 fail, D4 pass:** DAPF − FILL-HOLD −1.99 pooled; C1 −2.45 (confounded: retrained vs released
  checkpoint); C2 learned predictor +0.51 [+0.16, +0.84]; FILL-HOLD − SKIP +4.72 [+2.13, +7.28] pooled over 24 cells
  (strongest evidence yet for plain filling). Proposed control: C1 vs retrained-standard under the same masks (amendment needed).

### Iteration 29 — 2026-11-08 17:05 (research lane → RESEARCH-p2-head-and-tsm-2026-11-08.md)
- P2 on YOLO-pose: not novel as such (Ultralytics issue #15689 + 3 application papers), but nobody measures small-person
  benefit (COCO has no small-person keypoint labels) or VAD AUC under deadlines. Installed Ultralytics: no *-pose-p2* config
  or weights. P2 adds +67% FLOPs at 384×640 (5.58 → 9.34 G) ≈ stock at 800 px (8.71 G). Key mechanism found: the COCO converter
  drops keypoint-less persons → small people are trained as background.
- TSM: not useful inside STG-NF (2-channel coupling subnet; 13-tap temporal conv is ~70% of 616 params). Possible use: online
  TSM in the detector trained with random frame strides — only if misses are intermittent (free diagnostic B0 first).
- Proposed: A0/B0 free diagnostics, A1 label-policy ablation, A2 P2 vs stock-640 vs stock-800 at equal cost, B1 conditional.

### Iteration 30 — 2026-11-08 17:20 (owner: novelty discussion → D23)
- Paper direction agreed: diagnosis + causal label/resolution evidence + a general small-person-aware edge pose detector +
  deadline-aware evaluation + pose filling + coverage→AUC bound + negatives. DAPF, GAU, YOLO26 closed (Obsidian note 18).
- Frozen: EXP-020 (COCO-only 2×2: standard vs partial labels × stock vs P2 head; + stock@800 NCNN reference; tested on unseen
  ShanghaiTech + UBnormal), EXP-021 (coverage→AUC bound), EXP-022 (miss diagnostics; TSM only if ≥ 30% misses intermittent).
- Lanes: G 71e21d3b… (EXP-020), H 7e113816… (EXP-021/022).

### Iteration 31 — 2026-11-08 17:50 (lead) — EXP-014, EXP-016 (final), EXP-017, HTML explainer done
- **EXP-014:** α* 0.75; W1 fail (77.60 vs FT 80.28), W2 pass (UBnormal 55.67 vs FT 51.63); L3 collapses → trade-off only.
- **EXP-016 final (complete caches):** top-down YOLOv8n-det + RTMPose-s: 640 K1 80.52 / L2 79.34; **960 K1 83.11 (+14.58
  [+10.15, +18.72]) / L2 80.96**; K3 ≈ −4. VM mean ms/frame: 640 K1 71.8, 960 K1 111.6, 640 K3 44.6 → none meets 41.7 ms
  (deadline cost to be priced with pose filling). Top-down also helps in the dark (L2 +12.3), correcting the earlier invalid row.
- **EXP-017 final:** YOLO26n worse on all but UBnormal (Y2 −6.99 [−10.07, −4.40]).
- **HTML explainer:** docs/pose-stgnf-explainer.html (238 KB offline, 27 demos, 31 quizzes, 87-term glossary).
- 17:55 **EXP-021** (lane H): bound AUC ≤ cA + (1−cA)(1−cN)/2 (ties ½). B1 fail: 12/194 points exceed it by ≤ 0.051 (low-
  coverage configs; post-hoc cause: temporal smoothing spreads scores onto uncovered frames). **B2 pass:** Spearman 0.937
  (ShanghaiTech, n 160), 0.859 (UBnormal, n 34). **EXP-022:** fine-tuned detector recall <32 px 0.823 (baseline 0.001),
  32–48 0.912 (0.007), 48–64 0.965 (0.057) → strong evidence the blind spot is learnable (labels). Baseline misses 50.5%
  never-seen / 49.5% intermittent → pre-registered rule says propose a TSM experiment (caveat: only 10.9% of misses are in
  tracks matched in ≥ half their frames). Not launched; awaiting owner.
2026-11-10 07:03 EXP-020 verdicts P1 PASS / P2 FAIL / P3 PASS / P5 PASS; label cause confirmed, resolution not; P4+delivered pending VM timing.
2026-11-10 08:40 EXP-020 FINAL: P4 FAIL (M0P vs stock@800 NCNN −8.69, 1.13× ms). ML0 = practical winner: 13.29 ms (=stock@640), +7.35 vs stock@800 NCNN; delivered live s=3 FILL-HOLD 73.49 vs stock@640 66.61 / stock@800 68.50. Obsidian update delegated to Sonnet 5.5 subagent.
