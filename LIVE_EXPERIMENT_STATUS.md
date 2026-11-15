# Live Experiment Status: `Q2-P1-E09` (Hypotheses 1, 2, and 3 — COMPLETED & ANALYZED)

- **Last Updated:** `2026-11-15 18:55 +07`
- **Status:** **COMPLETED (10/10 Milestones)**
- **Progress:** `[████████████████████] 100%` (`10/10` milestones)
- **Full Detailed Report:** [`experiments/Q2-P1-E09_hypotheses_suite/report.md`](./experiments/Q2-P1-E09_hypotheses_suite/report.md)
- **MLflow Live Dashboard:** `http://127.0.0.1:5000` (Experiment: `Q2-P1-E09-Hypotheses-H1-H2-H3`)

---

## Executive Summary of Verdicts

1. **Hypothesis 1 (`MS-CTFlow-Bottleneck`, $d \in \{1, 2, 4\}$, $K=3$, `608` params, `202.5 KB` RAM) — PARTIALLY SUPPORTED:**
   - **Short-Burst Anomalies (`<= 2.5 s` jumping, tripping, falling):** **Strongly confirmed!** Lifts short-event causal `S4` AUC by **`+6.85 pt` on ShanghaiTech `ML0_edge` (`74.91% -> 81.76%`)**, **`+0.79 pt` on ShanghaiTech `released` (`78.51% -> 79.30%`)**, **`+2.51 pt` on UBnormal `released` (`67.11% -> 69.62%`)**, and **`+1.40 pt` on UBnormal `ML0_edge` (`74.06% -> 75.46%`)**.
   - **Sustained Anomalies (`> 2.5 s`):** Sparse $K=3$ dilated taps leave lag holes on long walking trajectories (`76.29% -> 72.59%` on SHT `ML0_edge`), proving that $d \in \{1, 2, 4\}$ is a specialized impulse filter whereas dense $K=13$ is optimal for sustained gait.

2. **Hypothesis 2 (`Two-Tier MPC-LoRaD`: Tier 1 Scene Deep-Sleep + Tier 2 Per-Person Kinematic Extrapolation) — STRONGLY CONFIRMED:**
   - **Strategy A (Untouched Benchmark Accuracy on 107 SHT & 211 UB Clips):** Across all 12 cells, `MPC-LoRaD` saves **`16.9%–44.0%`** of full pose calls on ShanghaiTech and **`46.1%–62.7%`** on UBnormal while keeping **`|\Delta \text{AUC}| \le 0.06\%` (`-0.06` to `+0.02 pt`, well inside the $\pm 0.3\%$ target!)**.
   - **Physical Raspberry Pi 5 Workload-Dependent Energy & Thermal Profiling:**
     - **Heavy (`100%` occupancy):** `-23.6%` compute energy/frame (`0.319 -> 0.244 J`), throughput `16.4 -> 20.1 FPS`.
     - **Medium (`40%` occupancy):** **`-43.1%` board power (`5.34 W -> 3.04 W`; core power `-55.6%`)**, **`-85.2%` compute energy (`0.331 -> 0.049 J`; `-61.8%` stream energy `0.331 -> 0.127 J`)**, temperature **`-4.4°C` (`67.5°C -> 63.1°C`)**, throughput **`16.1 -> 62.2 FPS`**.
     - **Sparse (`10%` occupancy):** **`-55.4%` board power (`5.35 W -> 2.38 W`; core power `3.78 W -> 1.06 W`)**, **`-94.1%` compute energy (`0.332 -> 0.020 J`; `-70.2%` stream energy `0.333 -> 0.099 J`)**, temperature **`-6.0°C` (`67.7°C -> 61.7°C`)**, throughput **`16.1 -> 121.5 FPS`**.
     - **Strategy B (Natural Sparse Clips):** **`-22.8%` compute energy/frame (`0.339 -> 0.262 J`)**, zero throttling (`0x0`).

3. **Hypothesis 3 (`Distillation-Free Green Kinematic Training`: $\mathcal{L}_{\text{NLL}} + \lambda_{\text{bone}}\mathcal{L}_{\text{bone}} + \lambda_{\text{smooth}}\mathcal{L}_{\text{smooth}} + \lambda_{\text{rec}}\mathcal{L}_{\text{rec}}$) — CONFIRMED ON UBNORMAL (`+3.79 pt` over Teacher) & PARTIALLY CONFIRMED ON SHT:**
   - **Unregularized Collapse (`G1`):** Training a direct pose adapter with $\mathcal{L}_{\text{NLL}}$ alone causes coordinate contraction (`pose_mse = 0.599–0.699`), dropping AUC by `-5.87 pt` on SHT (`70.42%`) and `-5.38 pt` on UBnormal (`55.54%`).
   - **Proposed Green Kinematic Regularizer (`G2`):** Prevents collapse (`pose_mse = 0.0012–0.0471`), **beats the 83M-AlphaPose Teacher-Distilled baseline (`G0`) on UBnormal by `+3.79 pt` causal `S4` (`60.92% -> 64.71%`) and `+2.97 pt` offline `S0` (`58.37% -> 61.34%`)** (and reaches `77.80%` `S0` / `73.80%` `S4` / `76.98%` short-event AUC on ShanghaiTech), while cutting training time by **`1,150x–1,438x` (`877 min -> 0.61–0.76 min`)** and GPU energy/CO2 by **`99.98%` (`3.655 kWh -> 0.0005 kWh`, `$10.82 -> $0.0025`)**!
