# EXP-004 — CPU deadlines × preprocessing × OS tuning in the edge pose-VAD pipeline (Lima VM)

Pre-registration frozen 2026-11-05 07:40, before any EXP-004 timing run used for results (the VM smoke runs in
`log.md` were all lost or contaminated and are not used). Under D13 (review afterwards) and D16 (OS knobs in VM).
Depends on EXP-003: its detection caches give accuracy; this experiment adds time.

## 1. Background (plain language)

On a real camera box, frames arrive at a fixed rate. If one frame takes too long, the next frames are skipped
("drop-on-late"). Skipped frames hurt this pipeline twice: the tracker loses people between the frames it sees, and
the flow model sees jerkier motion. Image enhancement costs CPU time, so it can cause more skipped frames, which
can cancel its benefit in the dark. OS tuning (threads, pinning, priority, isolating background work) can buy that
time back. This is exactly where the two arms of the thesis should interact.

## 2. Problem / objective

Measure **delivered** accuracy (accuracy of what the live system would actually output) for preprocessing × gate ×
OS configuration under a fixed camera rate, and test whether the effect of enhancement depends on the OS config.

## 3. Method

### 3.1 Two-step design (timing in the VM, accuracy replayed on the host)
1. **Timing traces (VM `edge-os`, 4 vCPU / 8 GB, `machine=vm`)**: run the real per-frame pipeline
   (read frame → darken → enhance → YOLOv8n-pose ONNX (`yolov8n-pose-640.onnx`, sha256 `0fb73008…5ddf`), ORT CPU →
   BoT-SORT → STG-NF on each new 24-pose segment) on a fixed **timing subset** (12 test videos, 2 per scene for the
   6 scenes with most videos, first 240 frames each, chosen by a seeded script before timing), and record per-frame
   service time per stage. One trace per (P, G, O, light).
2. **Deadline replay (host)**: camera at **F = 24 fps** (ShanghaiTech training videos: 251 at 24 fps, 79 at 25 fps,
   OpenCV, log 2026-11-05). Single worker, latest-frame policy: when the worker becomes free it takes the newest
   arrived frame; older waiting frames are dropped. Service times are resampled per frame from the matching trace
   (seeded block bootstrap of 24-frame blocks), giving a per-video `keep_mask` for all 107 test videos. The EXP-003
   detection cache is replayed with that mask; processed frames are re-indexed consecutively per video (what a live
   system's per-track ring buffer of the last 24 poses sees); each output score holds until the next processed frame.

### 3.2 Factors
- **P** (enhancement): E0, E1 gamma, E2 CLAHE, E3 Zero-DCE++ (ONNX, ORT CPU, same threads as the detector).
- **G** (gate): c = 0.25 (default) and c = G-xfit chosen per (light, P) in EXP-003.
- **O** (OS config), all with a **co-load**: ffmpeg x264 re-encode of a 1280×720 test pattern at real time (`-re`,
  "recording to disk"), as on an NVR-style edge box.
  - **O0 default**: ORT default threads, spinning on, no pinning, co-load at default priority, unpinned.
  - **O1 tuned**: co-load pinned to vCPU 0 at nice 19 (cgroup cpuset); pipeline pinned to vCPUs 1–3, ORT intra-op 3,
    spinning off, SCHED_FIFO priority 10 for the pipeline process.
- **Budget** (device class): whole-VM cgroup `cpu.max` = 4 CPUs (B4) and 2 CPUs (B2, a slower box). B2 is a
  stand-in for a weaker CPU, not a model of a Pi 5; Pi numbers come later on real hardware.
- **Light**: L0, L2, L3 (from EXP-003).

### 3.3 Measurement hygiene (host is noisy: load avg 7–10 on 2026-11-05, see log)
- All (P, G, O, B) cells for one light level are run **interleaved in a randomised order** (seeded), 5 repetitions
  per cell, each repetition = the 12-video subset once. VM auto-updates disabled (log 07:10).
- Host CPU (user + sys) is sampled every 3 s; a repetition is **excluded if mean host load outside the VM > 30%**
  of host CPU during it. Exclusions are counted and reported. Absolute ms are never compared with EXP-001/002.

### 3.4 Metrics
- Timing: per-frame service time p50 / p95 / p99 (pooled over repetitions, CI by bootstrap over repetitions),
  drop rate (1 − processed / arrived) at F = 24.
- Accuracy: **delivered AUC, live variant** (end frame, causal half-Gaussian; EXP-001 definition) over all 107 videos,
  bootstrap CI over videos; paired differences by paired bootstrap over videos.

## 4. Pre-registered expectations

| # | Expectation | Pass rule |
|---|---|---|
| H7 | OS tuning protects the pipeline from co-load | O1 vs O0: p99 service time lower and drop rate lower at B2, E0, c 0.25 (CIs over repetitions not overlapping) |
| H8 | Enhancement cost interacts with OS config | at L3, B2: interaction contrast [AUC(E3,O1) − AUC(E0,O1)] − [AUC(E3,O0) − AUC(E0,O0)] > 0, paired bootstrap 95% CI excludes 0 |
| H9 | Gate gain is OS-independent (zero compute) | at L2, L3: |interaction (G × O)| < 1 AUC point |
| H10 | Lower gates cost little CPU | tracker + STG-NF time at G-xfit ≤ 10% of detector time (E0, L3) |
| H11 | A best configuration exists that is not dominated | at L3, B2 the (P, G, O) with highest delivered AUC uses O1 and G-xfit |

If EXP-003 fails H2/H3 (gates do not explain the loss), H9–H11 are dropped and recorded as not tested.

## 5. Deviations policy

As EXP-003. The timing subset, F, repetition count, exclusion rule and seeds are fixed here. If a cell cannot be
run (e.g. SCHED_FIFO starves the VM), it is reported as failed with the error, not silently replaced.

## 6. Amendment 1 (frozen 2026-11-05 10:32, before any timing run used for results; only VM smoke runs exist)

Reasons (log, code lane 2 entries): (a) the pre-registered matrix at 240 frames/video is ≈ 77 h of VM time at the
smoke-measured 130–330 ms/frame; (b) the ONNX used for timing was 640×640 while the EXP-003 caches used Ultralytics'
letterboxed rectangular input (640×384 for 856×480 frames) — timing must use the **same input shape** as accuracy;
(c) the resolution arm (EXP-003 Addendum A) needs timing too.
- **Detector models for timing:** one static ONNX per R ∈ {480, 640, 800, 960}, exported at the exact rectangular
  letterbox shape Ultralytics uses for 856×480 frames at that imgsz; parity with the `.pt` predictions checked.
- **Frames:** 40 frames per subset video (first 40 of the 240 already copied) → 480 frames per repetition; 5 reps.
- **Cells (all with co-load, c = 0.25 unless stated, light L3):** P {E0, E1, E2, E3} × O {O0, O1} × B {B4, B2} = 16;
  plus R {480, 800, 960} × O × B at E0 = 12; plus two cost checks at E0/O0/B4: light L0 (does content change
  tracker/flow cost?) and c = G-xfit chosen value (H10). Traces of L3 are reused for L0/L2 delivered AUC unless the
  L0 check differs by > 10% in mean service time (then L0/L2 cells are added).
- **Co-load isolation (O1):** co-load in its own systemd scope with `AllowedCPUs=0`, nice 19 (cgroup cpuset).
  **B2:** pipeline and co-load inside one systemd scope with `CPUQuota=200%`.
- **Output latency:** a processed frame's score is assigned from its own arrival frame (no extra output delay);
  reported as a threat.

## 7. Amendment 2 (frozen 2026-11-06 03:41, after the main matrix, before any delivered-AUC result)

Finding that forces it (log 03:41): O1 timings are the same at B2 and B4 in every cell (e.g. E0@640 p50
41.07 vs 41.17 ms; E3 148.2 vs 148.6), while O0 slows strongly under B2 (E0@960 72.0 → 113.8). cgroup v2
`cpu.max` throttles only normal (CFS/EEVDF) tasks; O1's SCHED_FIFO pipeline is not subject to it, so **O1-B2 did not
run under the 2-CPU budget**. Consequences: every O1-B2 vs O0-B2 comparison (H7, H8 at B2) is a different-budget
comparison and is reported as such, not as an OS-tuning effect. B4 comparisons are unaffected (no quota).
- New config **O1n** = O1 without SCHED_FIFO (threads 3, spin off, pipeline pinned to vCPUs 1–3, co-load in its own
  scope on vCPU 0 at nice 19).
- Cells (L3, c 0.25, co-load): {E0@640, E2@640, E3@640, E0@960} × {O1n-B4, O1n-B2, O0-B2} = 12 cells × 5 reps,
  randomised interleaved (seed 2), same exclusion rule; O0-B2 is re-run so the B2 comparison is within one window.
- **H7′** (replaces H7 at B2): O1n vs O0 at B2, E0@640: p99 lower and kept fraction higher, per-rep ranges not
  overlapping. **H8′**: H8 recomputed at B2 with O1n in place of O1 (delivered AUC, L3 and L2 caches).
- Runs after EXP-003 Addendum C and the delivered-AUC replays (host idle).
- Amendment 2 fix before launch: O0 re-run cells renamed O0r (same settings) so the runner does not skip them as already-done main-block (cell, rep) ids.
