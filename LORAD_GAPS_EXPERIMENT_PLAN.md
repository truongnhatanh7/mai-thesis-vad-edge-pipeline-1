# Experimentation Plan: Addressing LoRaD's Core Limitations (`Q2-P1-E08`)

**Date:** 14 November 2026  
**Experiment ID:** `Q2-P1-E08` (`experiments/Q2-P1-E08_lorad_gaps_proposals/`)  
**Target Venues:** *Journal of Real-Time Image Processing (JRTIP)*, *IEEE Embedded Systems Letters (ESL)*, *Signal Processing: Image Communication*, *Pattern Recognition Letters*  
**Hardware Target:** Physical Raspberry Pi 5 Model B Rev 1.1 (`4x Cortex-A76 @ 2.4 GHz`, 8 GB RAM) + Live MLflow Tracking & Traces (`http://127.0.0.1:5000`)

---

## 1. Motivation & Problem Statement

In `Q2-P1-E06` and `Q2-P1-E07`, our empirical audit of **LoRaD** (*Lightweight Causal Pose Flow for Online Real-Time Video Anomaly Detection*, Zhang et al., *JRTIP* 2026), **STG-NF** (Hirschorn & Avidan, *ICCV* 2023), **Huang et al.** (*IEEE ESL* 2026), and **TrajVAD-P** (Song et al., *ECCV* 2026) revealed a sharp architectural trade-off on edge CPUs:

1. **Ultra-lightweight skeleton flows (`STG-NF`, `LoRaD`, `< 1K–5K` params, `~4 ms` on Pi 5)** are fast to score, but suffer from **four core limitations**:
   - **Gap 1 (Fixed Single Temporal Receptive Field):** Operating a single kernel scale over a fixed buffer ($T=24$ on ShanghaiTech, $T=16$ on UBnormal) cannot simultaneously resolve brief impulsive anomalies (tripping, punching, sudden jumps: `<= 2.5 s`) and prolonged trajectory anomalies (loitering, chasing).
   - **Gap 2 (Isolated Single-Person Modeling & Translation Blindness):** Per-window coordinate normalization (`normalize_pose`) subtracts the window mean $(x, y)$ and divides by height $\sigma_y$. This completely erases **bounding-box translation velocity, scale change, and human-object interactions** (e.g., riding a bicycle or pushing a cart smoothly yields normal relative limb posture) as well as **inter-person social proximity** (fighting, chasing, group gathering). Meanwhile, `TrajVAD-P` fixes this via a separate `757K`-parameter conditional flow, but increases scorer latency on the Raspberry Pi 5 by **`4x–6x` (`17.4–27.9 ms/frame`)**!
   - **Gap 3 (Pure Forward NLL Objective):** Minimizing forward negative log-likelihood ($\mathcal{L}_{\text{NLL}}$) alone under a unimodal Gaussian prior allows out-of-distribution abnormal motions to map into moderate-likelihood regions. Adding a **bidirectional reverse reconstruction term** $\mathcal{L}_{\text{rec}}(X, f^{-1}(f(X)))$ (inspired by Huang et al.) and a **contrastive latent margin term** $\mathcal{L}_{\text{margin}}$ on synthetic kinematic perturbations sharpens the anomaly boundary with **0% runtime inference overhead**.
   - **Gap 4 (Tracking Fragility & Cold-Start/Gap Discontinuities):** Upstream MOT track breaks or low-confidence joint jitter inject sudden discontinuities into the circular buffer, causing false-positive NLL spikes.

---

## 2. Pre-Registered Experimental Design (`Q2-P1-E08`)

We test each research gap **one factor at a time, then jointly**, on **ShanghaiTech Campus** (`107` test clips: `35 dev / 72 confirmation`, `194` events stratified by `short_event <= 2.5s` vs `sustained_event`, `fast_motion` [bikes/skaters/running] vs `moderate_motion`, `high_crowd >= 4 tracks` vs `low_crowd`) and **UBnormal** (`211` test clips: `70 dev / 141 confirmation`, `199` events), across both `released` AlphaPose skeletons and our locked edge frontend/tracker (`ML0_ncnn @ 640 + Two-Stage BoT-SORT / TrackTrack`).

### 2.1 Model Arms (3 Seeds `0, 1, 2` per Trained Scorer)

| Arm ID | Name | Architectural / Training Change (Single Factor Ladder) | Params | Inference Overhead vs `LoRaD` |
| :--- | :--- | :--- | :---: | :---: |
| **`B0`** | `STG-NF` (Baseline) | Standard symmetric ST-GCN coupling flow, forward $\mathcal{L}_{\text{NLL}}$, 18-joint graph | `~0.6K` | `1.00x` |
| **`B1`** | `LoRaD` (Causal Base) | Left-padded causal temporal conv ($d=1$) + online circular buffer + EMA persistence | `~0.6K` | `1.00x` |
| **`P_A`** | **`MS-CTFlow` (Proposal A — Gap 1)** | Replace single-scale causal TCN with **Multi-Scale Causal Dilated TCN** (parallel causal 1D branches with dilation rates $d \in \{1, 2, 4\}$ fused via $1\times 1$ projection + residual) inside each coupling block | `~1.8K` | `~1.05x` (`< +0.5 ms` on Pi 5) |
| **`P_B`** | **`RevMargin-Flow` (Proposal B — Gap 3)** | `LoRaD` architecture trained with **Hybrid Bidirectional + Contrastive Margin Loss**: $\mathcal{L} = \mathcal{L}_{\text{NLL}} + \lambda_1 \|X_y - f^{-1}(f(X))_y\|_2^2 + \lambda_2 \max(0, m - (\text{NLL}(X_{\text{pert}}) - \text{NLL}(X)))$ using synthetic temporal speed/jitter perturbations $X_{\text{pert}}$ | `~0.6K` | **`1.00x` (`0.0%` overhead!)** |
| **`P_C`** | **`IAC-Flow` (Proposal C — Gap 2)** | Augment the 18 human body joints with **2 Virtual Context Nodes** (`V = 20` graph):<br>• **Node 19 (Object / Box-Kinematics Node):** encodes normalized bounding-box velocity $(\Delta c_x / h, \Delta c_y / h)$ and aspect/scale state $(w/h, \Delta h / h)$ BEFORE window mean-subtraction erases translation speed!<br>• **Node 20 (Social Proximity Node):** encodes relative vector $(\Delta x_{\text{nn}}, \Delta y_{\text{nn}})$ and inverse distance to the nearest neighboring tracked person in the frame! | `~0.8K` | `~1.05x` (`~600x` smaller than `TrajVAD-P`!) |
| **`P_D`** | **`ConfSmooth-Buffer` (Gap 4)** | Confidence-weighted kinematic interpolation inside the circular buffer (weighting noisy low-confidence joints $c_{t,j} < 0.25$ with causal EMA velocity prediction before window normalization) | `~0.6K` | `~1.01x` |
| **`P_ABC`** | **`MS-IAC-RevFlow` (Joint Proposed Scorer)** | Combines **Multi-Scale Causal Dilated TCN (`P_A`) + Bidirectional Reverse & Margin Loss (`P_B`) + 20-Node Interaction Graph (`P_C`) + Confidence-Weighted Buffer (`P_D`)** | `~2.2K` | `~1.10x` (`< 5 ms` on Pi 5 vs `28 ms` for `TrajVAD-P`) |

---

## 3. Hypotheses & Pre-Registered Success Criteria

- **H_A (Multi-Scale Temporal Receptive Field — Proposal A):** `P_A (MS-CTFlow)` improves strict causal `S4` AUC and localized AUC on **short-duration events (`<= 2.5 s`)** and **sustained events** over single-scale `B1 (LoRaD)` (`test_dev` selection, `test_confirmation` verification with 2,000-resample paired video bootstrap).
- **H_B (Reverse Consistency & Contrastive Margin — Proposal B):** `P_B (RevMargin-Flow)` improves `S4` AUC and reduces false alarms per normal hour (`FA/h`) over `B1 (LoRaD)` at **exactly zero additional inference latency or parameter count**.
- **H_C (Interaction-Aware Context Nodes — Proposal C):** `P_C (IAC-Flow)` significantly improves `S4` AUC and event recall on **fast-moving human-object anomalies (`fast_motion`: cyclists, skaters, carts)** and **crowded multi-person scenes (`high_crowd >= 4 tracks`)** where pure pose normalization (`B0`/`B1`) is blind, achieving accuracy competitive with `TrajVAD-P` at **`< 1/300th` the parameter count (`~0.8K` vs `757K`) and `4x–5x` faster scoring on the Raspberry Pi 5**.
- **H_Joint (Combined Edge System on Physical Raspberry Pi 5):** Pairing **`ML0_ncnn @ 640 + Two-Stage Tracker + P_ABC (MS-IAC-RevFlow) + Cache-C1+Guard`** achieves the highest real-time delivered AUC on the physical **Raspberry Pi 5** (`>= 24.0 FPS`, `p99 wait < 500 ms`, `throttled = 0x0`).

---

## 4. Required Outputs & Reproducibility

All artifacts will be written to `experiments/Q2-P1-E08_lorad_gaps_proposals/`:
- `plan.md` & `config.yaml` (locked before evaluation)
- `results.csv` (per-video and per-seed S0/S4/delivered AUCs across all arms)
- `subgroups.csv` (short vs. sustained, fast-motion/bike vs. moderate, crowd vs. low-crowd breakdown)
- `pi5_benchmark.csv` (real Raspberry Pi 5 latency, FPS, PMIC Watts, Joules/frame, Temp °C, RSS MB, and live delivered AUC)
- `summary.json` & `report.md`
- Unit tests in `experiments/tests/test_q2_p1_e08.py` and full registration in MLflow (`http://127.0.0.1:5000`).
