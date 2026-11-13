# EXP-008 — Deadline-aware evaluation: rank flips, slower devices, a predictive model (no new hardware)

Pre-registration frozen 2026-11-07 09:10, before any EXP-008 computation (D19). Analysis-only: reuses the EXP-003 /
EXP-006 detection caches and the EXP-004 VM timing traces. Live scoring variant throughout (what a device outputs).

## 1. Why
The candidate headline is "offline accuracy misleads on edge devices". This experiment quantifies it (A1),
extends it to slower devices than the VM without new hardware (A2), and tests whether delivered accuracy can be
predicted from offline accuracy and the drop rate (A3).

## 2. Method
- **Knobs** (EXP-004 O0-B4 traces, 5 reps pooled): E0 @480 / @640 / @800 / @960, E1 gamma @640, E2 CLAHE @640.
- **Slower devices (A2):** multiply every per-frame service time of a trace by s ∈ {1, 2, 4, 8} (s = 1 is the VM;
  s ≈ 3–5 is the expected Raspberry Pi 5 range — an assumption, to be checked on hardware). Camera 24 fps,
  latest-frame policy, block bootstrap seed 0, exactly as EXP-004. Because only service × fps matters, s also stands
  for higher camera rates (s = 2 at 24 fps ≡ s = 1 at 48 fps).
- **Data:** ShanghaiTech L0, L2, L3 caches (EXP-003) and UBnormal L0 caches (EXP-006), all 6 knobs where a cache exists.
- **Offline reference:** EXP-003 / EXP-006 live AUC of the same cache (no deadline).

## 3. Pre-registered expectations

| # | Expectation | Pass rule |
|---|---|---|
| A1a | offline ranking ≠ delivered ranking | at s = 1, for ≥ 2 of the 4 (dataset, light) groups the offline-best knob is not the delivered-best knob |
| A1b | weak rank agreement on slow devices | Kendall τ between offline and delivered AUC over all (group × knob) cells at s = 4 ≤ 0.5 |
| A2 | the delivered optimum moves to cheaper knobs as the device slows | per group, the imgsz of the delivered-best E0 knob is non-increasing in s; at s = 8, 480 px beats 640 px in ≥ 3 of 4 groups |
| A3 | delivered AUC is predictable | linear model delivered = a + b·offline + c·drop_rate fitted on ShanghaiTech cells (all s), applied to UBnormal: MAE ≤ 1.5 AUC and Spearman(pred, actual) ≥ 0.7 |

## 4. Deviations policy
As EXP-003. No new knobs, speeds or model terms after the first replay.
