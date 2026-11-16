# Option A — Clean, Minimal Codebase for Review (Huang et al. vs. LoRaD vs. Ours, Offline & Online on Pi 5)

Everything you need to review **Option A (Flagship Systems + Perception Paper)** is distilled into **4 small, self-contained Python files (~160–180 lines each)** in this folder (`option_a_clean_review/`). Nothing in the original `experiments/` folder was modified.

---

## 1. Read These 4 Files in Order (~25 Minutes Total)

| File | Lines | What It Does & What to Look For |
| :--- | :---: | :--- |
| **[`step1_models.py`](./step1_models.py)** | `182` | **The Neural Networks (Pose + Scorer):**<br>• `PoseDetector`: wraps `YOLOv8n-pose` across Huang `RPi` (`PyTorch 640`), Huang `RPi*` (`NCNN 320`), `Stock NCNN 640`, and our `ML0 NCNN 640`.<br>• `SkeletonNormalizingFlow`: shows the exact code difference between Original `STG-NF` (symmetric temporal padding `(k-1)//2`) and `LoRaD` (`CausalSTGCN`, left-only padding `F.pad(h, (0, 0, k-1, 0))`).<br>• `normalize_window_n0` vs. `ema_normalize_track_n1`: shows per-window normalization (`N0`) vs. streaming prefix EMA normalization (`N1`). |
| **[`step2_tracker_cache.py`](./step2_tracker_cache.py)** | `188` | **The 3 Systems Mechanisms (Tracker + Cache + Guard):**<br>• `build_botsort_tracker`: shows default `BoT-SORT` (`c=0.25`, no low-conf second stage) vs. our locked two-stage `BoT-SORT` (`c=0.20, track_low_thresh=0.05` keeping weak boxes alive to bridge brief blurs).<br>• `StaticValidityCacheC1`: 40×24 grayscale thumbnail check (`~1.7 ms` on Pi 5) that skips the `~48.9 ms` NCNN pose call when active person ROIs and borders are static (up to `max_age=3` frames).<br>• `HysteresisQueueGuard`: sheds alternate frames when FIFO queue wait `> 400 ms` and disarms when `< 150 ms`.<br>• `OnlineCircularWindowScorer`: per-track 24-pose ring buffer that shares per-frame `(x, y)` across overlapping windows bit-exactly. |
| **[`step3_offline_eval.py`](./step3_offline_eval.py)** | `168` | **Offline (`S0`) vs. Strict Causal (`S4`) Evaluation on All 107 ShanghaiTech Clips:**<br>• `smooth_offline_two_sided` vs. `smooth_causal_one_sided`: shows why `S0` (paper protocol) looks into the future (`middle` frame landing + two-sided Gaussian + test-set-max fill) whereas `S4` (strict causal) lands at window completion (`frame 24`), fills empty frames with a fixed training-only sentinel (`-0.5122`), and smooths over past frames `<= t` only. |
| **[`step4_pi5_online.py`](./step4_pi5_online.py)** | `192` | **Live Online Runner on Physical Raspberry Pi 5 (`nanh@raspberrypi.local`):**<br>• Feeds frames on a real-time 24.0 FPS wall-clock schedule (`dt = 41.67 ms`), tracks FIFO queue wait (`start_ms = max(arrival_ms, worker_free_ms)`), drops any window completing `> 2000 ms` after arrival (`windows_dropped_late`), and samples live Raspberry Pi 5 DA9091 PMIC power (`Watts`, `Joules/frame`), SoC temperature (`°C`), and throttling flags (`0x0`). |

---

## 2. Three Commands to Run & Verify Everything

```bash
cd /Users/nanh/Documents/RMIT_MscAI/thesis/option_a_clean_review

# 1. Run the fast unit tests (verifies causality, C1 cache, queue guard, and S0/S4 scoring in 3 seconds):
../experiments/.venv/bin/pytest test_option_a.py -v

# 2. Run the full 107-clip ShanghaiTech Offline (S0) vs. Strict Causal (S4) comparison (~3 mins):
../experiments/.venv/bin/python -u step3_offline_eval.py

# 3. Run the Online Stream Benchmark locally (or on the Raspberry Pi 5 over SSH):
../experiments/.venv/bin/python -u step4_pi5_online.py --max-frames 240
```

---

## 3. Verified Output Summary (Offline Mac + Live Physical Raspberry Pi 5)

### Table A: Offline (`S0`) vs. Strict Causal (`S4`) Audit (`step3_offline_eval.py`, All 107 ShanghaiTech Clips)

| Pipeline Configuration | Offline `S0` Paper AUC | Strict Causal `S4` AUC | Scored Frame Coverage | What This Proves |
| :--- | :---: | :---: | :---: | :--- |
| **1a. Oracle `AlphaPose` + `STG-NF`** *(Paper Ref)* | `85.95%` | `80.83%` | `86.1%` | What Huang et al. (`86.2%`) & LoRaD (`84.8%`) benchmarked on (desktop oracle poses). |
| **1b. Oracle `AlphaPose` + `LoRaD` (`N0` Window)** | `85.18%` | `80.16%` | `86.1%` | Reproduces LoRaD's `84.8%` paper claim when using per-window normalization (`N0`). |
| **1c. Oracle `AlphaPose` + `LoRaD` (`N1` Stream)** | `83.70%` | `78.51%` *(75.59% stream)* | `86.1%` | Streaming prefix normalization (`N1`) loses `1.7–5.3` AUC points vs. `N0`. |
| **2. Huang `Stock YOLOv8n-pose @640` + Default Track** | `68.51%` | `65.33%` *(57.28% @320p)* | `59.7%` | Stock YOLO misses `<48 px` people (`0.4%` recall), dropping coverage to `59.7%`. |
| **3a. Ours `ML0 @640` + Default Track + `STG-NF`** | `78.01%` | `73.75%` | `80.8%` | Fixing the COCO box-only label bug (`ML0`) adds **`+8.42` causal AUC** at `1.00x` FLOPs. |
| **3b. Ours `ML0 @640` + 2-Stage Track + `STG-NF`** | **`81.67%`** | **`76.81%`** | **`84.4%`** | Enabling 2-stage `BoT-SORT` (`low=0.05`) bridges gaps and adds **`+3.06` causal AUC**. |
| **3c. Ours `ML0 @640` + 2-Stage Track + `LoRaD`** | **`80.81%`** | **`76.28%`** | **`84.4%`** | `ML0` + 2-stage tracking works equally well with `LoRaD` Causal Flow. |

---

### Table B: Live Online Execution on Physical Raspberry Pi 5 (`step4_pi5_online.py`, 4× Cortex-A76 @ 2.4 GHz, 24.0 FPS Stream)

| Pipeline on Physical Raspberry Pi 5 | Detector & Scorer | Mean Service (`ms/frame`) | Pi 5 Speed (`FPS`) | Queue Wait `p99` (`ms`) | Power (`W`) / Energy (`J/frame`) | Keeps Up With 24 FPS? | Live On-Time Delivered Pi 5 AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Huang `RPi`** *(Default PyTorch 640)* | Stock PyTorch `640` / `STG-NF` | `162.86 ms` | `6.14 FPS` | `28,683 ms` *(28.7 s!)* | `6.04 W` / `0.984 J` | **NO** *(87.5% dropped late)* | `53.32%` *(Collapsed)* |
| **2. Huang `RPi*`** *(NCNN 320×320)* | Stock NCNN `320` / `STG-NF` | `27.35 ms` | `36.57 FPS` | `18.0 ms` | `5.38 W` / `0.147 J` | **YES** | `59.18%` *(Blind to `<48px`)* |
| **3. Stock NCNN 640 (`Full-Rate`)** | Stock NCNN `640` / `STG-NF` | `56.95 ms` | `17.56 FPS` | `10,816 ms` *(10.8 s!)* | `5.64 W` / `0.321 J` | **NO** *(> 41.67 ms)* | `53.44%` *(Collapsed)* |
| **4. `LoRaD` (`Full-Rate`, No Cache)** | `ML0` NCNN `640` / `LoRaD` | `59.20 ms` | `16.41 FPS` | `49,824 ms` *(49.8 s!)* | `5.54 W` / `0.328 J` | **NO** *(> 41.67 ms)* | `56.32%` *(Collapsed)* |
| **5. `LoRaD` + `Cache-C1+Guard`** | `ML0` NCNN `640` / `LoRaD` | **`37.50 ms`** | **`25.43 FPS`** | **`416.0 ms`** *(0 late!)* | `5.12 W` / **`0.192 J`** (`-41%`) | **YES (`> 24 FPS`)** | **`63.48%`** |
| **6. Ours `ML0` (`Full-Rate`, No Cache)** | `ML0` NCNN `640` / `STG-NF` | `59.20 ms` | `16.43 FPS` | `46,824 ms` *(46.8 s!)* | `5.63 W` / `0.327 J` | **NO** *(> 41.67 ms)* | `47.46%` *(Collapsed)* |
| **7. Ours `ML0` + `Cache-C1+Guard` (`W=1`)** | `ML0` NCNN `640` / `STG-NF` | **`37.50 ms`** | **`25.39 FPS`** | **`405.5 ms`** *(0 late!)* | `4.89 W` / **`0.193 J`** (`-41%`) | **YES (`> 24 FPS`)** | **`64.49%–67.12%`** |
| **8. Ours `ML0` + `Cache-C1+Guard` (`Burst32`)** | `ML0` NCNN `640` / `STG-NF` | **`35.16 ms`** | **`24.01 FPS`** | **`250.3 ms`** *(0 late!)* | **`4.04 W` / `0.168 J` (`-49%`)** | **YES (`> 24 FPS`)** | **`74.19%` (`+26.7 pt`!)** |
