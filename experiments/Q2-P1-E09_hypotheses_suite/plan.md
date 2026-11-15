# Q2-P1-E09 — Pre-Registered Plan: MS-CTFlow (H1), Two-Tier MPC-LoRaD (H2), and Green Kinematic Training (H3)

**Date:** 15 November 2026  
**Experiment Namespace:** `Q2-P1-E09` (`experiments/Q2-P1-E09_hypotheses_suite/`)  
**Live Status File:** `/Users/nanh/Documents/RMIT_MscAI/thesis/LIVE_EXPERIMENT_STATUS.md` (auto-updated by the runner after every sub-step)  
**MLflow Tracking Server:** `http://127.0.0.1:5000` (Experiment: `Q2-P1-E09-Hypotheses-H1-H2-H3`)

---

## 1. Overview of the Three Hypotheses

### Hypothesis 1 (H1): Channel-Bottlenecked Multi-Scale Causal Temporal Dilated Graph Flow (`MS-CTFlow-Bottleneck`)
- **Problem:** Standard `LoRaD` (`B1`) uses a single $K=3, d=1$ causal temporal convolution per coupling block (effective receptive field of 9 frames across 4 blocks), missing multi-timescale transitions. Conversely, naive 3-branch concatenation (`E08 P_A`) increased parameters by `2.58x` (`616 -> 1,592`) and overfitted sustained trajectories.
- **Formulation:** Inside each affine coupling layer, split $x \to (x_A, x_B)$ and pass the spatial GCN feature through a **Bottlenecked Multi-Scale Causal Temporal Convolution Unit (`MS-CTC`)** with parallel branches $d \in \{1, 2, 4\}$ and strict left-only padding $(K-1)\cdot d$:
  $$y_t^{(d)} = \sum_{k=0}^{K-1} w_k^{(d)} \cdot x_{t - k \cdot d}, \qquad \forall \tau > t,\; \frac{\partial y_t^{(d)}}{\partial x_\tau} = 0$$
  To prevent parameter blowup and overfitting, each branch uses depthwise-separable temporal filtering followed by a $1\times 1$ bottleneck fusion + residual gate:
  $$y_t = x_t + \alpha \cdot \text{Conv}_{1\times 1}\!\left(\left[y_t^{(1)},\, y_t^{(2)},\, y_t^{(4)}\right]\right)$$
  Buffer size remains strictly $T=24$ (`~52 KB` RAM), expanding the receptive field from 9 to 21 frames with only `+168` parameters (`784` vs `616` params).
- **Target & Evaluation:** Micro-AUC (`S0` offline and `S4` strict causal) and per-class/stratum breakdown (`short_event <= 2.5s` jumping/tripping vs `sustained_event > 2.5s` loitering/wandering) on **ShanghaiTech** (`107` clips) and **UBnormal** (`211` clips) against `STG-NF`, `LoRaD`, and `E08 P_A`.

---

### Hypothesis 2 (H2): Two-Tier Motion-Triggered Predictive Caching (`MPC-LoRaD`) & Workload-Dependent Energy Profiling on Raspberry Pi 5
- **Formulation (Two-Tier Hierarchical Cache):**
  - **Tier 1 (Scene-Level Fast Motion/Presence Gate):** Ultralight 4$\times$-downsampled grayscale frame difference $\mathcal{E}_{\text{scene}, t}$. When $\mathcal{E}_{\text{scene}, t} < \epsilon_{\text{scene}}$ and no active tracks are present $\to$ **`[DEEP SLEEP]`** (bypasses Detector, Pose Encoder, Tracker, and Flow; advances frame clock at idle power).
  - **Tier 2 (Per-Person Kinematic Pose Caching):** When motion/tracks are present, for each tracked individual $i$, compute centroid displacement $\Delta p_t^{(i)} = \|c_t^{(i)} - c_{t-1}^{(i)}\|_2$ and normalized bounding-box crop pixel-difference energy:
    $$\mathcal{E}_t^{(i)} = \frac{1}{|B_t|} \sum_{(u,v) \in B_t} |I_t(u,v) - I_{t-1}(u,v)|$$
    - If $\mathcal{E}_t^{(i)} < \epsilon_{\text{motion}}$ (steady gait / quasi-static): **Bypass pose estimation** and extrapolate keypoints via first-order damped constant-velocity kinematics:
      $$\hat{P}_t^{(i)} = P_{t-1}^{(i)} + \gamma \left(P_{t-1}^{(i)} - P_{t-2}^{(i)}\right)$$
    - If $\mathcal{E}_t^{(i)} \ge \epsilon_{\text{motion}}$ (rapid limb change): Execute full pose estimator.
- **Strategy A (Dual-Evaluation Protocol) & Strategy B (Naturally Sparse Clips):**
  1. **Detection Accuracy Preservation:** Evaluate across the **untouched ShanghaiTech (`107` clips) and UBnormal (`211` clips)** benchmarks (both on `AlphaPose`/`RTMPose` top-down crops and `ML0` edge poses) across threshold sweep $\epsilon_{\text{motion}}$ to verify that `MPC-LoRaD` stays within $\pm 0.3\%$ of uncached `LoRaD` (`85.37%` S0 / `80.33%` S4 on SHT AlphaPose; `81.12%` S0 / `76.54%` S4 on SHT `ML0`).
  2. **Workload-Dependent Hardware Energy & Thermal Profiling on Physical Raspberry Pi 5:** Profile instantaneous power (`Watts` via `vcgencmd pmic_read_adc`), energy per frame (`Joules/frame`), SoC temperature (`°C` via `vcgencmd measure_temp`), and ARM core clock (`MHz` via `vcgencmd measure_clock arm`) across:
     - **Workload Regime 1 — Heavy / Rush-Hour (`100%` occupancy):** Dense ShanghaiTech crowd clips.
     - **Workload Regime 2 — Medium / Intermittent (`40%` occupancy):** Alternating active pedestrian intervals and quiet intervals.
     - **Workload Regime 3 — Sparse / Night Surveillance (`<10%` occupancy):** Long quiescent intervals with occasional passing pedestrians, plus **Strategy B naturally sparse public clips** (untouched test segments with sparse/zero pedestrian traffic).
     - **Sustained Thermal Resistance Profile:** Continuous multi-pass streaming on Raspberry Pi 5 logging temperature (`°C`), clock frequency (`MHz`), throttling flags (`0x0`), and cumulative energy (`Joules`).

---

### Hypothesis 3 (H3): Distillation-Free Green Training via Kinematic Latent Anchors
- **Problem:** `LoRaD`'s Stage 1 requires distilling from an 83M-parameter AlphaPose teacher across 100 epochs of COCO (`~64,115` images $\times 100$ epochs = `6.4M` teacher+student forward passes), incurring massive GPU energy and carbon cost. However, naively training a pose refinement adapter + flow directly on surveillance videos with $\mathcal{L}_{\text{NLL}}$ alone causes degenerate pose collapse (the encoder shrinks poses toward a constant to trivially maximize flow likelihood!).
- **Formulation (Distillation-Free Kinematic-Regularized Direct Training):**
  Initialize with off-the-shelf lightweight pose weights (`ML0` / `RTMPose-t`) + a lightweight Spatio-Temporal Kinematic Pose Refiner $g_\phi$ and `LoRaD` flow $f_\theta$, and train directly on normal target surveillance clips using:
  $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{NLL}}\big(f_\theta(\hat{P})\big) + \lambda_{\text{bone}} \mathcal{L}_{\text{bone}}(\hat{P}) + \lambda_{\text{smooth}} \mathcal{L}_{\text{smooth}}(\hat{P}) + \lambda_{\text{rec}} \mathcal{L}_{\text{rec}}(\hat{P}, P_0)$$
  where:
  - **Bone Length Invariance Loss:** $\mathcal{L}_{\text{bone}} = \sum_{(u,v) \in \mathcal{E}_{\text{skeleton}}} \big| \|\hat{p}_t^{(u)} - \hat{p}_t^{(v)}\|_2 - \bar{L}_{(u,v)} \big|^2$
  - **Kinematic Smoothness (Acceleration) Loss:** $\mathcal{L}_{\text{smooth}} = \sum_{v=1}^{18} \|\hat{p}_t^{(v)} - 2\hat{p}_{t-1}^{(v)} + \hat{p}_{t-2}^{(v)}\|_2^2$
  - **Kinematic Anchor & Flow Inverse Consistency:** $\mathcal{L}_{\text{rec}} = \|\hat{P} - P_0\|_2^2 + \| \hat{P} - f_\theta^{-1}(f_\theta(\hat{P})) \|_2^2$
- **3-Arm Ablation Matrix & Green AI Accounting:**
  1. **`G0_Teacher_Distill_Baseline`:** Full 2-stage AlphaPose teacher distillation baseline (100 epochs COCO teacher distillation + Stage 2 flow training).
  2. **`G1_Direct_Unregularized` ($\lambda_{\text{bone}} = \lambda_{\text{smooth}} = \lambda_{\text{rec}} = 0$):** Jointly optimizing $g_\phi + f_\theta$ with $\mathcal{L}_{\text{NLL}}$ only (demonstrating pose collapse!).
  3. **`G2_Green_Kinematic_Regularized` (Proposed):** Jointly optimizing $g_\phi + f_\theta$ with $\mathcal{L}_{\text{total}}$ (`5x–20x` faster training, zero teacher, measured `kWh`, `kg CO2e`, and `$` cloud GPU cost, while matching or improving downstream SHT & UBnormal AUC!).
