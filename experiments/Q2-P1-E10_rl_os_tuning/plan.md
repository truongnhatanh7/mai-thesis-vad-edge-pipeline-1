# Q2-P1-E10 — Fast-Feedback Contextual Bandit / RL OS Tuning for CPU-Only Edge VAD (`FastOS-RL`)

**Date:** 15 November 2026  
**Experiment Namespace:** `Q2-P1-E10` (`experiments/Q2-P1-E10_rl_os_tuning/`)  
**Hardware Target:** Physical Raspberry Pi 5 Model B Rev 1.1 (`4x Cortex-A76 @ 2.4 GHz`, 8 GB RAM)  
**Live Status File:** `/Users/nanh/Documents/RMIT_MscAI/thesis/LIVE_EXPERIMENT_STATUS.md`  
**MLflow Experiment:** `Q2-P1-E10-RL-OS-Tuning` (`http://127.0.0.1:5000`)

---

## 1. Research Problem & Why Fast-Feedback RL Is Needed

Prior edge-inference OS/DVFS tuners (`zTT` MobiSys 2021, `EdgeEngine` SEC 2023, `EcoEdgeInfer` SEC 2024, `AGFT` 2025) require hundreds of inference requests or full-video evaluations per step, taking hours to converge. On a Raspberry Pi 5 running skeleton-based video anomaly detection, OS-level CPU behavior differs fundamentally between **uncached continuous workloads** and **cached bursty workloads**:

1. **Cached Bursty Workloads (`LoRaD + Cache`, `Ours ML0 + Cache-C1+Guard`):**
   - Skipping 30–45% of frames on static scenes causes CPU load to oscillate every `40–120 ms`.
   - Under Raspberry Pi OS's default `ondemand` governor (`sampling_rate = 100 ms`), brief 40 ms cache-hit intervals cause the kernel to downclock the Cortex-A76 cores from `2400 MHz` to `1500 MHz`. When the next full detector frame arrives, it suffers a **frequency ramp-up latency penalty**, increasing p95/p99 service time and frame drop rate!
2. **Uncached Continuous Workloads (`LoRaD Full-Rate`, `Ours Fine-Tuned ML0 Full-Rate`, `Huang et al.`):**
   - Running 4 OpenMP threads unpinned across all 4 cores (`OS-Default`) causes thread-wakeup contention with background OS tasks, Linux `kworker`/SSH interrupts on Core 0, and L2 cache thrashing across the 4$\times$512 KB private L2 caches, inflating p99 latency and frame drops.

---

## 2. Fast-Feedback Surrogate-Warm-Started LinUCB Contextual Bandit (`FastOS-RL`)

To converge in **`< 15 seconds` per pipeline (`24 micro-window steps x 15 frames = 360 frames`)** directly on the physical Raspberry Pi 5 without waiting for 30-minute full-dataset runs:

### 2.1 Action Space ($\mathcal{A}$, 18 Runtime-Switchable OS Knobs)
Each action $a = (g, T, p) \in \mathcal{A}$ configures three OS/CPU layers in `< 1.5 ms` without restarting the process:
1. **CPU Frequency Governor ($g \in \{\text{ondemand}, \text{schedutil}, \text{performance}\}$):** Written directly to `/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor`.
2. **Thread Concurrency ($T \in \{2, 3, 4\}$):** Applied to NCNN (`net.opt.num_threads = T`) and PyTorch (`torch.set_num_threads(T)`).
3. **Core Affinity Pinning & Scheduler Priority ($p \in \{\text{unpinned\_nice0},\; \text{pinned\_isolated\_nice-10}\}$):**
   - `unpinned_nice0`: Default Linux CFS placement across cores `0-3`, `nice = 0`.
   - `pinned_isolated_nice-10`: Pins inference threads strictly to isolated cores (`1..3` for $T=3$, `2..3` for $T=2$, `0..3` for $T=4$, keeping Core 0 free for OS/IO interrupts when $T < 4$) via `/proc/self/task` `os.sched_setaffinity` and elevates CFS scheduling priority (`nice = -10` via `os.setpriority`), avoiding the `SCHED_FIFO` quota-escape/starvation bug discovered in `EXP-004`.

### 2.2 Context State, Surrogate Warm-Start, and Micro-Window Reward
- **Context Vector $\mathbf{s}_k \in \mathbb{R}^5$:** Captures pipeline workload characteristics (`[1, is_cached_bursty, c1_skip_ratio, duty_cycle_load, res_scale]`).
- **Surrogate Warm-Start ($\hat{\theta}_0$):** Initialized from our `E02` thread-scaling curve and `EXP-004` affinity priors so the bandit starts from an informed prior rather than cold random exploration (which we compare against `Random-Search` and `Cold-UCB`).
- **Micro-Window Reward ($W_{\text{step}} = 15$ frames $\approx 0.6\text{ s}$):**
  $$R_k = -\left( 2.5 \cdot \text{DropRate}_k + 1.0 \cdot \frac{\text{MeanServiceMs}_k}{41.67} + 0.5 \cdot \frac{\text{p95ServiceMs}_k}{41.67} + 0.15 \cdot \text{JoulesPerFrame}_k \right)$$

---

## 3. The 5 Evaluated Pipelines on Physical Raspberry Pi 5

For each of the 5 pipelines, we run:
- **(A) `OS-Default`:** `ondemand` governor, `T=4`, unpinned `0-3`, `nice 0`.
- **(B) `OS-Static-EXP004`:** Static heuristic from `EXP-004` (`ondemand`, `T=3`, pinned `1-3`, `nice 0`).
- **(C) `FastOS-RL (Ours)`:** The LinUCB-converged OS configuration $(g^*, T^*, p^*)$ learned in the 24-step fast-feedback loop.

Across the 5 pipelines:
1. **`Pipeline 1: LoRaD (Full-Rate)`** (`Stock NCNN 640` + `BoT-SORT` + `LoRaD Causal Flow`)
2. **`Pipeline 2: LoRaD + Cache`** (`Stock NCNN 640` + `Two-Tier MPC / C1 Cache` + `LoRaD Causal Flow`)
3. **`Pipeline 3: Ours Fine-Tuned ML0 (Full-Rate)`** (`ML0 NCNN 640` + `2-Stage BoT-SORT` + `STG-NF`)
4. **`Pipeline 4: Ours Fine-Tuned ML0 + Cache`** (`ML0 NCNN 640` + `2-Stage BoT-SORT` + `Cache-C1+Guard` + `STG-NF`)
5. **`Pipeline 5: Huang et al. Pipeline`** (Tested on both `5a: Huang RPi* NCNN 320` and `5b: Huang RPi PyTorch 640` + `BoT-SORT` + `STG-NF`)

### Metrics Collected per Pipeline & OS Policy:
- **Inference Speed:** Mean service latency (`ms/frame`), `p50`, `p95`, `p99` tail latency (`ms`), and achieved throughput (`FPS`).
- **Drop / Late Rate:** Percentage of frames exceeding real-time deadline / dropped (`%`) and `p99` queue wait (`ms`).
- **CPU & Clock Utilization:** Process CPU %, System CPU %, and live ARM clock frequency (`MHz`, mean & min—capturing `ondemand` frequency dips).
- **Hardware Telemetry:** Mean PMIC Power (`Watts`), Energy (`Joules/frame`), and SoC Temperature (`°C`).
- **End-to-End Accuracy:** Offline `S0` AUC, Ideal Causal `S4` AUC, and **Real-Time Delivered Pi 5 AUC (%)** across all 107 ShanghaiTech clips.
