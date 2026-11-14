# Paper Positioning & Experimental Checkpoint (`E00`–`E08` + Physical Raspberry Pi 5)

**Saved Checkpoint Date:** 14 November 2026  
**Purpose:** Preserves the exact positioning argument for why our findings form a strong Q2 (or Q1) edge-AI systems & methods paper, alongside the verified numbers from `E00`–`E07` and the physical Raspberry Pi 5 runs.

---

## 1. Why `LoRaD` (2026) and `Huang et al.` (2026) Did NOT Solve Real-Time Edge VAD

### 1.1 What `Q2-P1-E06` Proved About `LoRaD` (Zhang et al., *JRTIP* July 2026)
`LoRaD` proposed (a) replacing `STG-NF`'s symmetric temporal convolutions with **causal (left-padded) temporal convolutions** and (b) using an **online circular buffer** for $O(1)$ per-frame streaming updates.

Our `Q2-P1-E06` ablation (`experiments/Q2-P1-E06_lorad_causal_flow/report.md`) uncovered a fundamental mathematical and systems flaw in that claim:

| Scorer Variant on ShanghaiTech (`ML0` Edge Poses / `Released` AlphaPose) | `ML0` Offline `S0` | `ML0` Causal `S4` | `Released` Offline `S0` | `Released` Causal `S4` | Can It Use an $O(1)$ Circular Buffer? |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **`A0`: Original `STG-NF` (Frozen)** | `81.62` | **`76.86`** | `85.94` | **`80.85`** | **No** (Must re-normalize & re-score all 24 frames) |
| **`A1`: Original `STG-NF` (Retrained, 3 seeds)** | `81.57` | **`76.86`** | `85.22` | `80.20` | **No** |
| **`A2`: `LoRaD` Causal Conv + Window Norm (`N0`)** | `81.12` | **`76.54`** (`-0.32 pt`) | `85.37` | `80.33` | **NO!** Per-window mean/std changes on every step, invalidating all 23 cached conv states! |
| **`A3`: `LoRaD` Causal Conv + Prefix Norm (Batch)** | `81.61` | `76.55` | `83.95` | `78.77` (`-2.08 pt`) | Partial |
| **`A3s`: `LoRaD` True $O(1)$ Circular Buffer (`N1`)** | `76.99` | **`72.89`** (**`-3.97 pt`!**) | `80.16` | **`75.59`** (**`-5.26 pt`!**) | **Yes, but AUC crashes by `-4.0` to `-5.3` points!** |

**Three Takeaways on `LoRaD`:**
1. **Causal convs inside a window (`A2`) give zero benefit under honest completion-time delivery (`S4`):** A 24-frame window only finishes at frame 24 anyway. Using left-only padding inside that 24-frame window shrinks the effective temporal receptive field and slightly reduces accuracy (`76.54` vs `76.86` on SHT; `60.26` vs `61.08` on UBnormal).
2. **True $O(1)$ circular-buffer streaming (`A3s`) loses `3.97` to `5.26` AUC points:** Reusing hidden convolutional states across steps requires freezing each frame's normalized coordinates as it enters the buffer (`N1` prefix/EMA normalization), which breaks window shift/scale invariance and degrades AUC by `4–5` points.
3. **The scorer is only `4–8%` of Raspberry Pi 5 runtime anyway:** Even if the scorer took `0.0 ms`, the pose detector (`46–49 ms` on Pi 5) dominates `82%+` of frame latency, so `LoRaD` without frontend caching and queue guards still runs below `24 FPS` (`~17 FPS`) and suffers queue collapse on the Raspberry Pi 5!

---

## 2. What Makes Your Paper a Strong Q2 (or even Q1) Paper Now?

Instead of claiming a single incremental tweak, our paper presents a **complete, end-to-end systems and algorithmic investigation** that exposes and resolves the failure modes of **three 2025–2026 edge/trajectory VAD paradigms (`Huang et al. 2026`, `LoRaD 2026`, and `TrajVAD-P 2026`)** on a **physical Raspberry Pi 5**:

### Contribution 1 — Exposing the "Offline-to-Edge Illusion" on Physical Raspberry Pi 5 (`E00`, `Pi5-Huang-Repro`)
Published edge-VAD papers report offline desktop AUCs (`84.8–86.2%` on pre-extracted AlphaPose skeletons) while omitting real-time delivered accuracy on the embedded device. On a physical **Raspberry Pi 5 (4× Cortex-A76 @ 2.4 GHz)** streaming 24 FPS video:
- **`Huang RPi` (`640×640` PyTorch YOLO):** Runs at **`6.14 FPS`** (`162.9 ms/frame`, `6.04 W`). Queue wait explodes to `28.7 s` in 10 seconds, collapsing delivered AUC from `68.53%` offline to **`53.32%`**.
- **`Huang RPi*` (`320×320` NCNN YOLO):** Runs at **`36.57 FPS`** (`27.3 ms/frame`, `5.38 W`), keeping up with 24 FPS, **but `320×320` downscaling blinds the detector to small/distant pedestrians (`0.2%` recall `<48 px`), capping AUC at `57.28%` offline / `59.18%` causal**.
- **Even `640p` NCNN FP16 (`Stock` or `ML0` Full-Rate `K=1`):** Takes **`56.4–60.2 ms/frame` (`16.6–17.7 FPS`)** on the Pi 5—slower than the `41.67 ms` (`24 FPS`) deadline—causing the queue to grow to **`46.8 seconds`** over a 2-minute stream and collapsing live delivered AUC to **`47.46%`**.

### Contribution 2 — Fixing Frontend & Tracker Blindness at `1.00×` Detector FLOPs (`E01`, `E07`)
- **Label Cause vs. Resolution Cause:** We prove `YOLOv8n-pose`'s blindness below `64 px` is a **COCO label-conversion artefact** (dropping keypoint-less small people as background). Retaining them as box-only targets with masked keypoint loss (`ML0` @ 640 NCNN FP16, `1.00×` FLOPs) raises `<48 px` recall from `0.4% → 37.0%` (`48–96 px` `38.9% → 85.6%`) and boosts causal AUC by **`+7.75 [+4.63, +10.47]`** points—outperforming `800p` (`1.56×` cost), `P2`-head `MLP` (`1.92×` FLOPs, `2.35×` slower on Pi 5), and NMS-free `YOLO26n-pose` (`62.69%` causal AUC).
- **Two-Stage Track Continuity:** We show `84.2%` of remaining `ML0` misses are intermittent gaps inside active tracks, `52.3%` of which already have a low-confidence cached detection (`0.05 <= conf < 0.25`). Unlocking two-stage association (`BoT-SORT` `c=0.20, low=0.05` and `TrackTrack` CVPR 2025) adds **`+3.04 [+1.44, +4.71]`** causal AUC (`76.82%` causal / `81.62%` offline) at zero extra detector cost (`0.78 ms/frame` on Pi 5).

### Contribution 3 — Auditing Computation Reuse & Queue-Guarded Scheduling on Real Cortex-A76 Silicon (`E00`, `E02`–`E05`, `Pi5_suite`)
- **Valid vs. Invalid Reuse:** Sharing raw per-frame `(x, y, conf)` across overlapping windows is bit-exact (`18.5×` fewer pose evaluations), whereas reusing post-normalization tensors (`-8.0 pt`) or `LoRaD` streaming prefix states (`-4.0 pt`) degrades accuracy.
- **`Cache-C1+Guard` Crosses the Real-Time Pi 5 Barrier:** Skipping redundant NCNN calls on static frames (`C1`) with a hysteresis queue guard (`on > 400 ms, off < 150 ms`) and intact-window burst scheduling (`Burst32`) achieves **`24.01 FPS`** (`p99 wait = 250 ms` vs `46,824 ms` for `Fixed-K1`), cuts Pi 5 energy per frame by **`-49%` (`0.327 J → 0.168 J`)**, and raises live Pi 5 delivered AUC from **`47.46%` $\to$ `74.19%`** (`+26.73` points!). Across all 107 ShanghaiTech clips at matched CPU budgets, `Cache-C1` beats `Fixed` by **`+2.56 [+0.77, +4.42]*`** pt (`+7.46 [+5.52, +9.35]*` pt with `Guard`).
- **Rigorous Negative Results:** On the Pi 5's Cortex-A76 (which lacks `i8mm`), **ONNX Runtime static INT8 is slower than NCNN FP16 at `T=1..2` (`88.2 vs 73.6 ms`) and consumes `+11%` more energy per frame at `T=4` (`0.298 J` at `6.81 W` vs `0.270 J` at `5.81 W`)**. Motion-only gating (`-1.35 pt`), complex joint controllers (`-1.60 pt`), and H.264 codec motion vectors (`-0.08 pt`) also fail to beat `Cache-C1` at matched CPU cost.

### Contribution 4 — Overcoming Scorer Blind Spots (`E07` `TrajVAD-P` & `E08` `MS-CTFlow` / `Reverse-Consistent` / `IAC-Flow`)
- **Why Pure Pose Flows Plateaus (`E07` & `E08`):** Even with perfect tracking, `STG-NF` and `LoRaD` have a single temporal scale ($T=24$), use forward NLL only, and erase bounding-box velocity/scale (`normalize_pose`), blinding them to cyclists, carts, and multi-person interactions.
- **`E07` (`TrajVAD-P` ECCV 2026 on Pi 5):** We benchmark trajectory+pose conditional flows (`757K` params, `17–28 ms` scoring on Pi 5) and show how `Cache-C1+Guard` rescues `TrajVAD-P` from `109.5 s` queue collapse down to `487 ms` wait on the Pi 5.
- **`E08` (Addressing `LoRaD`'s 4 Research Gaps Directly):** We test **Proposal A (`MS-CTFlow` Multi-Scale Dilated Causal TCN $d \in \{1, 2, 4\}$)**, **Proposal B (Reverse-Consistent + Contrastive Margin Causal Flow, `0%` inference overhead)**, and **Proposal C (`IAC-Flow` Interaction-Aware 20-Node Causal Graph with Virtual Object/Box-Velocity & Social Proximity Nodes, `<10K` params)**—giving us an ultra-lightweight (`~1–5 ms` on Pi 5) scorer that captures multi-scale timing, human-object speed, and social interactions without `TrajVAD-P`'s `757K`-param / `28 ms` CPU penalty!
