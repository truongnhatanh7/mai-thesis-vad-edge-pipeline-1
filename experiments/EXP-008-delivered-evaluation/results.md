# EXP-008 results — deadline-aware evaluation without new hardware

> 92 replays: 4 groups (ShanghaiTech L0/L2/L3, UBnormal L0) × up to 6 knobs × slow-down s ∈ {1, 2, 4, 8}; EXP-004
> O0-B4 traces scaled by s; 24 fps; live variant. Driver `edgelab.analysis.exp008`; rows `~/thesis-data/exp008/
> all_rows.csv`, verdicts `verdicts.json`, MLflow `EXP-008`. s = 1 is the VM; a Pi 5 is *assumed* to be s ≈ 3–5.
> Owner sign-off: pending.

## 1. Verdicts

| # | Expectation | Measured | Verdict |
|---|---|---|---|
| A1a | offline-best ≠ delivered-best in ≥ 2 of 4 groups (s = 1) | SHT-L0 960 → **640**; SHT-L2 960 → **CLAHE@640**; SHT-L3 gamma = gamma; UB 960 = 960 | **pass** (2/4) |
| A1b | Kendall τ(offline, delivered) at s = 4 ≤ 0.5 | **τ = 0.16** (s = 1: 0.72) | **pass** |
| A2 | delivered-best imgsz non-increasing in s; 480 > 640 at s = 8 in ≥ 3 groups | 480 > 640 at s = 8 in 3/4 ✔; not monotone (s = 8 picks 960/800 in SHT-L0/L2 at 90%+ drops) | **fail** |
| A3 | delivered = a + b·offline + c·drop (fit SHT, test UB): MAE ≤ 1.5, ρ ≥ 0.7 | MAE 2.62, ρ 0.61 (a 0.43, b 0.34, c −0.11) | **fail** |

## 2. Delivered live AUC by slow-down (offline in the first column)

| Group / knob | offline | s = 1 | s = 2 | s = 4 | s = 8 |
|---|---|---|---|---|---|
| SHT-L0 480 | 61.7 | 61.5 | 61.0 | 57.0 | 56.2 |
| SHT-L0 640 | 66.2 | **66.2** | **62.7** | 57.6 | 52.5 |
| SHT-L0 960 | **68.2** | 63.8 | 58.7 | 53.9 | 56.2 |
| SHT-L2 640 | 66.1 | 66.1 | 61.2 | 56.5 | 54.1 |
| SHT-L2 CLAHE@640 | 67.1 | **67.1** | **62.2** | 57.1 | 53.3 |
| SHT-L2 960 | **67.9** | 63.1 | 56.5 | 52.6 | 53.6 |
| SHT-L3 gamma@640 | **63.4** | **63.4** | **60.2** | **58.0** | 55.8 |
| SHT-L3 960 | 62.0 | 59.9 | 58.1 | 56.2 | 56.4 |
| UB-L0 640 | 56.2 | 56.1 | **54.7** | **54.0** | **53.0** |
| UB-L0 960 | **59.1** | **56.9** | 53.1 | 51.7 | 52.3 |

Drop rates: s = 1 0–42%, s = 2 7–71%, s = 4 53–85%, s = 8 77–93%.

## 3. Observations
1. **Offline accuracy stops predicting deployed accuracy as the device slows:** rank agreement τ falls from 0.72
   (VM) to 0.16 (4× slower). The most accurate offline setting (960 px) is never the delivered winner on
   ShanghaiTech and loses on UBnormal from s = 2 on.
2. **At s = 2** (plausibly the lower end of a Pi 5) the delivered winners are the *cheap* options: 640 px (L0),
   CLAHE@640 (L2), gamma@640 (L3), 640 px (UBnormal) — 960 px is the worst or second-worst everywhere.
3. **Above ≈ 75% dropped frames (s = 8) every configuration collapses to 52–57 AUC** and the ranking becomes noisy
   (replay noise ≈ 2–3 points, EXP-004) — this is why A2's monotonicity failed. Practical reading: there is a
   drop-rate ceiling beyond which no knob choice rescues the pipeline; the device must be fast enough to stay below it.
4. A simple linear model (offline AUC + drop rate) does not transfer across datasets (A3 fail): the cost of drops
   depends on the dataset (UBnormal's near-chance baseline has less to lose).

## 4. Threats
- s is a uniform slow-down of VM traces, not a measured Pi profile (memory/thermal effects differ).
- Replay noise grows at high drop rates; single bootstrap seed per cell.
- O0-B4 traces only (no co-load isolation / budget variants).
