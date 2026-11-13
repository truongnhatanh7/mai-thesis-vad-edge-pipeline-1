# EXP-004 results — CPU deadlines × preprocessing × resolution × OS configuration (Lima VM)

> Timing: VM `edge-os` (vz, Ubuntu 24.04 arm64, 4 vCPU / 8 GB) on the M2 Pro host, `machine=vm`, comparative only.
> Accuracy: EXP-003 detection caches replayed with per-video keep masks (latest-frame policy, camera 24 fps, 24-frame
> block bootstrap of the measured traces), frozen STG-NF, **live variant** (end frame + causal half-Gaussian).
> Raw: `~/thesis-data/vm-exchange/results/matrix/` (manifest + 150 traces), `~/thesis-data/exp004/`
> (`timing_summary.csv`, `delivered_summary_*.csv`, `delivered/<light>__<cell>/{result.json, delivered_scores_live.npz,
> keep_masks.npz}`, `interaction_tests.json`); MLflow experiment `EXP-004`. Owner sign-off: **pending**.
> **Status:** complete (main matrix 150 runs + Amendment 2 60 runs, all valid).

## 1. Measurement validity

- 150/150 runs ok, **0 excluded** (host CPU outside the VM 14–17% per run, limit 30%); order randomised per repetition
  (seed 0, `timing_plan.csv`). Per-repetition p99 ranges are tight (e.g. E0@640 O0-B4 59.6–64.4 ms).
- L0 cost check: mean service L0 36.57 vs L3 36.37 ms (+0.6%, < 10%) → L3 traces are reused for L0/L2 replays.
- Replay check: with ≈ 0 drops, delivered AUC equals EXP-003's offline live AUC (L0 r480 61.65; L2 E2 67.06).
- **Budget escape:** cgroup v2 `cpu.max` does not throttle SCHED_FIFO tasks. O1 (FIFO priority 10) has the same
  timings at B2 and B4 in all 7 knob cells (E0@640 p50 41.07 vs 41.17 ms; E3 148.2 vs 148.6), while O0 slows under B2
  (E0@960 72.0 → 113.8). **O1-B2 is therefore not a 2-CPU run**; every O1-B2 vs O0-B2 comparison is a
  different-budget comparison. Amendment 2 adds O1n (O1 without FIFO).

## 2. Cost ledger (L3, co-load on, O0-B4 unless stated; mean ms per frame, 5 reps × 480 frames)

| Knob | enhance | detector (pre+infer+post) | track + flow | service p50 / p99 | frames kept at 24 fps (B4 / B2) |
|---|---|---|---|---|---|
| E0 @480 | 0.1 | 18.4 infer | 0.1 + 1.1 | 21.8 / 44.6 | 1.00 / 1.00 |
| E0 @640 | 0.1 | 29.9 infer | 0.2 + 1.7 | 34.4 / 62.7 | 0.97 / 0.95 |
| E0 @800 | 0.1 | 48.5 infer | 0.2 + 3.5 | 53.8 / 88.7 | 0.73 / 0.53 |
| E0 @960 | 0.0 | 64.6 infer | 0.2 + 4.4 | 72.0 / 104.8 | 0.55 / 0.37 |
| E1 gamma @640 | 0.7 | 29.9 | 0.2 + 2.1 | 35.0 / 60.2 | 0.97 / 0.89 |
| E2 CLAHE @640 | 2.9 | 30.0 | 0.2 + 2.4 | 37.7 / 64.1 | 0.96 / 0.91 |
| E3 Zero-DCE++ @640 | 100.9 (B2: 186.2) | 37.7 | 0.2 + 3.5 | 143.1 / 179.8 | 0.29 / 0.14 |

- **H10 pass:** at the lowest gate (c 0.10) tracker + flow = 2.08 ms = 6.6% of the detector (≤ 10%).
- The Zero-DCE++ CPU cost (≈ 100–190 ms) is ~2 orders of magnitude above the 1.3 ms GPU figure quoted in the
  literature (Abu Awwad 2024) — it alone exceeds the 41.7 ms frame period 2.4–4.5×.

## 3. Delivered accuracy (live AUC at 24 fps; offline = EXP-003 live, no deadline)

| Light | Knob | offline | O0-B4 | O0-B2 | O1-B4 | O1-B2† |
|---|---|---|---|---|---|---|
| L0 | E0 @480 | 61.65 | 61.48 | 61.65 | 61.61 | 61.66 |
| L0 | E0 @640 | 66.17 | **66.19** | 66.15 | 65.98 | 65.78 |
| L0 | E0 @800 | 67.77 | 65.70 | 61.15 | 63.92 | 63.64 |
| L0 | E0 @960 | 68.19 | 63.80 | 59.70 | 61.04 | 60.27 |
| L0 | E3 @640 | 66.16 | 58.57 | 53.62 | 54.83 | 57.09 |
| L2 | E0 @640 | 66.11 | 66.13 | 66.10 | 65.69 | 65.60 |
| L2 | E2 CLAHE @640 | 67.06 | **67.06** | 66.55 | 67.37 | 66.93 |
| L2 | E0 @960 | 67.89 | 63.08 | 59.63 | 61.70 | 60.30 |
| L2 | E3 @640 | 66.89 | 58.02 | 53.80 | 55.52 | 58.97 |
| L3 | E0 @640 | 61.71 | 61.78 | 61.74 | 61.21 | 61.34 |
| L3 | E1 gamma @640 | 63.41 | **63.41** | 62.75 | 61.72 | 63.37 |
| L3 | E0 @960 | 61.95 | 59.89 | 58.52 | 59.52 | 59.08 |
| L3 | E3 @640 | 60.47 | 58.51 | 54.89 | 56.73 | 57.64 |

† O1-B2 escaped the 2-CPU budget (§1). Full table incl. E1/E2 at every level and 95% CIs: `delivered_summary_*.csv`.

## 4. Pre-registered expectations (plan §4, Amendment 1)

| # | Expectation | Measured | Verdict |
|---|---|---|---|
| H7 | O1 vs O0 at B2, E0@640: p99 lower and drop lower (rep ranges not overlapping) | p99 87.1 [85.6–88.0] vs 99.3 [89.6–105.1] (lower); kept per rep 0.940–0.958 vs 0.931–0.958 (overlap) | **fail** by rule; and confounded (budget escape) → H7′ |
| H8 | L3, B2, E3: interaction contrast > 0, CI excludes 0 | +3.15 [+0.33, +6.01] | passes by rule, **invalid** (different budgets) → H8′ |
| H9, H11 | gate-based | — | **not tested** (EXP-003 H3 failed; plan rule) |
| H10 | tracker + flow ≤ 10% of detector at G-xfit | 6.6% | **pass** |
| H7′ | O1n vs O0r at B2, E0@640: p99 lower and kept higher, rep ranges not overlapping | p99 55.6 vs 99.7 ms; kept 0.980 [0.967–0.985] vs 0.943 [0.925–0.950] | **pass** |
| H8′ | L3, B2, E3: [Δ(E3) under O1n] − [Δ(E3) under O0r] > 0, CI excludes 0 | +2.34 [−0.46, +5.20] (L2: +2.44 [−0.51, +5.63]) | **fail** (positive, CI includes 0) |

Valid (B4) interaction contrasts, O1 vs O0 (paired bootstrap): L3 E1 −1.13 [−1.91, −0.28], E2 +0.55 [−0.34, +1.41],
E3 −1.22 [−3.06, +0.61]; L2 E1 −0.39 [−1.06, +0.32], E2 +0.76 [−0.06, +1.56], E3 −2.06 [−3.90, −0.24].

## 5. Amendment 2 — same-budget OS comparison (O1n = O1 without SCHED_FIFO; O0r = O0 re-run in the same window)

Timing (L3, co-load; p50 / p99 ms; kept fraction at 24 fps, mean [per-rep range]):

| Knob | O0-B4 | O1-B4 (FIFO) | O1n-B4 | O0r-B2 | O1n-B2 |
|---|---|---|---|---|---|
| E0 @640 | 34.4 / 62.7; 0.972 | 41.2 / 88.1; 0.945 | 41.3 / 56.0; 0.977 | 35.3 / 99.7; 0.943 [0.925–0.950] | 41.1 / **55.6**; **0.980** [0.967–0.985] |
| E2 @640 | 37.7 / 64.1; 0.958 | 44.0 / 91.5; 0.892 | 44.0 / 68.8; 0.907 | 38.3 / 103.7; 0.912 | 44.0 / 58.5; 0.915 |
| E3 @640 | 143.1 / 179.8; 0.288 | 148.6 / 261.3; 0.271 | 147.1 / 162.9; 0.282 | 293.8 / 380.5; 0.146 | 164.7 / 191.6; **0.257** |
| E0 @960 | 72.0 / 104.8; 0.552 | 81.5 / 168.8; 0.483 | 81.8 / 130.0; 0.505 | 112.0 / 190.9; 0.376 | 102.4 / 136.3; 0.428 |

- O0r reproduces O0 (E0@640 p50 35.3 vs 35.1; E3 293.8 vs 294.7 ms) → no drift across the night.
- O1n respects the quota (E3 147 → 165 ms under B2), so O1n-B2 vs O0r-B2 is a same-budget comparison.
- **SCHED_FIFO was the harmful part of the tuned bundle**: at B4, FIFO raises p99 (88.1 vs 56.0 ms for O1n).
  Pinning + co-load isolation without FIFO cuts p99 by ≈ 44% under the 2-CPU budget and keeps 76% more frames for
  Zero-DCE++ — but with 4 CPUs the 4-thread default still keeps more frames for heavy knobs (960 px 0.552 vs 0.505).

Delivered live AUC (24 fps):

| Light | Knob | O0-B4 | O1n-B4 | O0r-B2 | O1n-B2 |
|---|---|---|---|---|---|
| L2 | E0 @640 | 66.13 | 66.06 | 66.13 | 66.10 |
| L2 | E2 @640 | 67.06 | 67.34 | 67.07 | **67.66** |
| L2 | E3 @640 | 58.02 | 57.69 | 56.49 | 58.91 |
| L2 | E0 @960 | 63.08 | 60.49 | 58.54 | 60.54 |
| L3 | E0 @640 | 61.78 | 61.70 | 61.73 | 61.72 |
| L3 | E2 @640 | 61.55 | 62.02 | 61.63 | 62.47 |
| L3 | E3 @640 | 58.51 | 57.25 | 56.37 | 58.69 |

Interaction contrasts O1n vs O0r at B2 (paired bootstrap): E3 +2.34 [−0.46, +5.20] (L3), +2.44 [−0.51, +5.63] (L2);
E2 **+0.86 [+0.29, +1.47]** (L3), +0.62 [−0.03, +1.27] (L2).
Replay noise: two timing-identical cells (O0-B2 vs O0r-B2, E3, kept 0.146) give delivered 53.80 vs 56.49 at L2 →
at > 80% drops the trace-resampling noise of delivered AUC is ≈ 2–3 points, which widens the E3 contrasts.

## 6. Observations

1. **Deadlines reverse the offline ranking of resolution.** Offline, 960 px is the best knob (+2.0 at L0, +1.8 at
   L2); delivered at 24 fps it is worse than 640 px by 2.4 / 3.1 (4 CPUs) and 6.5 / 6.5 (2 CPUs), because 45–63% of
   frames are dropped and tracks break. 800 px also loses delivered accuracy. Only 480 px never drops (but costs
   4.5 offline).
2. **Cheap, condition-specific enhancement survives deadlines intact:** CLAHE at L2 (+0.93 over E0, 67.06) and gamma
   at L3 (+1.63, 63.41) keep their whole offline effect at 4 CPUs (≤ 4% drops).
3. **Learned enhancement is net-negative under a CPU deadline at every light level:** Zero-DCE++ keeps 14–29% of
   frames and loses 3–12 points delivered, even where it helped offline.
4. **OS tuning helps only in the right form and the right budget.** Real-time priority (SCHED_FIFO) worsens tail
   latency and silently escapes cgroup CPU quotas (a deployment pitfall). Without it, pinning + co-load isolation
   halves p99 under a 2-CPU budget (H7′ pass) and recovers 76% more frames for costly enhancement — but this does not
   make Zero-DCE++ worthwhile (still −3 below no enhancement at L3) and with 4 CPUs plain 4-thread defaults are as good
   or better. The only significant OS × preprocessing interaction is small and positive for cheap CLAHE (+0.86 at L3).
5. **A condition-aware rule beats every fixed configuration on delivered AUC** (O0-B4, 640 px): none at L0 (66.19),
   CLAHE at L2 (67.06), gamma at L3 (63.41), vs. always-none 66.19 / 66.13 / 61.78, always-CLAHE 65.93 / 67.06 /
   61.55, always-960 63.80 / 63.08 / 59.89. (Descriptive; gains +0.9 at L2, +1.6 at L3; CIs in the summary files.)

## 7. Threats to validity

- VM timing (vCPUs float over host P/E cores; host 14–17% busy); not edge timing. The Pi 5 is expected to be several
  times slower (Huang 2026: 3.2 FPS at 640×640 PyTorch), which would move the drop boundary to lower resolutions.
- Trace reuse across light levels (validated for cost: +0.6%); single camera rate (24 fps); single-worker
  latest-frame policy (no pipelining of stages); scores held from their own arrival frame (no output delay).
- Budget emulation by cgroup quota is not a slower CPU (quota throttling is bursty) and does not bind RT tasks.
- Accuracy side inherits EXP-003's threats (synthetic light, ShanghaiTech only).

## 8. Owner sign-off
- [ ] pending
