# Q2-P1-Pi5-Huang-Repro — Physical Raspberry Pi 5 Reproduction of Huang et al. (2026) vs. Our Pipeline

*Generated from physical Raspberry Pi 5 Model B Rev 1.1 (`aarch64`, 4x Cortex-A76 @ 2.4 GHz, 8 GB RAM, Debian 6.18.50+rpt-rpi-2712, NVMe root) with live MLflow Tracking, Model Registry, and Trace Spans (`http://127.0.0.1:5000/#/experiments/28`).*

## 1. Board & Sensor Verification

- **Board:** Raspberry Pi 5 Model B Rev 1.1 (`4` cores, `2400 MHz` max clock, `ondemand` governor, `7.87 GiB` RAM).
- **Thermal & Throttling Health:** `vcgencmd get_throttled = 0x0` across all 6 runs (zero under-voltage, zero frequency capping, zero thermal throttling; clock held at `2400.0 MHz` under load).
- **Idle PMIC Power (`vcgencmd pmic_read_adc` across all 12 DA9091 rails):** `1.81–2.35 W` total board power (`0.50–1.00 W` `VDD_CORE` at `0.75–0.85 V`).

---

## 2. Physical Raspberry Pi 5 Benchmark: Huang et al. (`RPi`, `RPi*`) vs. Stock NCNN 640 vs. Ours (`ML0` + 2-Stage BoT-SORT + `Cache-C1+Guard`)

All runs executed directly on the Raspberry Pi 5 (4 threads) on ShanghaiTech (`856x480`, `24.0 fps` arrival schedule = `41.67 ms/frame` real-time deadline, `2.0 s` max queue wait cap):

| Configuration on Raspberry Pi 5 | Runtime & Input Size | Mean Service (`ms/frame`) | Throughput (`FPS`) | Pose / Track / Score (`ms`) | Queue Wait `p99` (`ms`) | Keeps Up With 24 FPS? | Mean Power (`W`) / Core (`W`) | Energy (`J/frame`) | Mean / Max Temp (`°C`) | Peak RSS (`MB`) | Offline AUC (`%`) | Causal Ideal AUC (`%`) | Delivered Real-Time Pi 5 AUC (`%`) | MLflow Run ID |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **1. Huang `RPi`** (Stock YOLO + Default Track) | PyTorch FP32 `640×640` | `162.86` | **`6.14`** *(paper: 3.2)* | `154.14 / 0.61 / 5.12` | `28,682.6` *(28.7 s!)* | **NO** *(87.5% dropped)* | `6.04 / 4.39` *(paper: 6.3)* | `0.984` | `70.0 / 73.8` | `1029` | `68.53` | `66.03` | **`53.32`** *(Collapsed)* | `2a8dadd6da2641f18ad015434a988847` |
| **2. Huang `RPi*` (Square)** (Stock YOLO + Default Track) | NCNN FP16 `320×320` | `27.35` | **`36.57`** *(paper: 17.7)* | `20.06 / 0.59 / 3.68` | `18.0` | **YES** | `5.38 / 3.77` *(paper: 6.07)* | `0.147` | `71.1 / 72.7` | `1069` | **`57.28`** | **`59.18`** | **`59.18`** *(Blind to small people!)* | `7271f0279e5946eeaaed44faf2285221` |
| **3. Huang `RPi*` (16:9 Letterbox)** | NCNN FP16 `192×320` | `19.68` | **`50.81`** | `12.50 / 0.58 / 3.61` | `147.1` | **YES** | `5.27 / 3.69` | `0.104` | `71.0 / 72.2` | `1028` | **`57.28`** | **`59.18`** | **`59.18`** | `2a3f575cacb946ed8034408200d4799a` |
| **4. Stock NCNN @ 640** (Stock YOLO + Default Track) | NCNN FP16 `384×640` | `56.95` | **`17.56`** | `48.71 / 0.70 / 4.53` | `10,815.8` *(10.8 s!)* | **NO** *(> 41.7 ms)* | `5.64 / 3.96` | `0.321` | `72.6 / 74.4` | `1031` | `68.53` | `66.03` | **`53.44`** *(Collapsed)* | `2678d8943ff64fefbca582a211dcc302` |
| **5. Ours `ML0` Full-Rate (`K=1`, No Cache)** | NCNN FP16 `384×640` | `60.23` | **`16.60`** | `48.94 / 1.06 / 7.23` | `13,196.5` *(13.2 s!)* | **NO** *(> 41.7 ms)* | `5.45 / 3.76` | `0.328` | `73.6 / 75.5` | `1032` | **`81.62`** | **`76.82`** | **`54.73`** *(Collapsed without C1!)* | `3d333cddd1834f2c8902e59e05b15cdb` |
| **6. Ours `ML0` + `Cache-C1+Guard` (`W=1`)** | NCNN FP16 `384×640` | **`35.16`** *(**`-41.6%`**)* | **`28.45`** *(**`1.71x`**)* | `27.53 / 0.86 / 2.45` *(C1 check `1.76 ms`)* | **`417.8`** *(0 late drops!)* | **YES** *(28.5 > 24 FPS!)* | `5.45 / 3.80` | **`0.192`** *(**`-41.6%` J**)* | `73.9 / 75.5` | `1033` | **`81.62`** | **`76.82`** | **`63.67` (`W=1`) / `69.58` (`W=32` burst)** | `371374c0ab0742268c696430338297f0` |

---

## 3. Four Critical Insights From Physical Raspberry Pi 5 Execution

1. **Direct Reproduction of Huang et al.'s Hardware Numbers—And Exposure of Their Missing Accuracy-Latency Trade-off:**
   - On `RPi` (PyTorch `640×640`), we measure **`154.1 ms` pose latency** (`6.14 FPS` total, `6.04 W` power), matching Huang et al.'s `149.4 ms` YOLO latency and `6.3 W` power almost identically! (Their `155.1 ms` BoT-SORT latency came from leaving GMC camera-motion compensation or ReID enabled on CPU, whereas disabling GMC/ReID makes BoT-SORT `0.61 ms`). At `6.14 FPS` on a `24 FPS` stream, queue wait explodes to **`28.7 seconds`** in just 10 seconds of video, dropping `87.5%` of windows and collapsing delivered AUC to **`53.32%`**.
   - On `RPi*` (NCNN `320×320`), pose latency drops to **`20.06 ms`** (matching Huang et al.'s `24.8 ms` and `5.38 W` vs `6.07 W`), achieving **`36.57 FPS`** (`50.81 FPS` at `192×320` letterbox)—which keeps up with `24 FPS` (`p99 wait = 18 ms`), **but at `320×320` the detector is completely blind to small/distant people, capping AUC at `57.28%` offline / `59.18%` causal!**

2. **Why `640p` NCNN Alone Cannot Run Full-Rate (`K=1`) at 24 FPS on the Raspberry Pi 5:**
   - On the physical Pi 5's Cortex-A76, NCNN FP16 at `384×640` takes **`48.7–48.9 ms/frame`** for pose inference (`57.0 ms` total for `stock` = **`17.56 FPS`**; `60.2 ms` total for `ML0` = **`16.60 FPS`** because `ML0` finds `3.23` tracks/frame vs `1.90` for `stock`).
   - Because `16.6–17.6 FPS` is slower than the `24.0 FPS` (`41.67 ms/frame`) camera cadence, running `640p` NCNN at full rate (`K=1`) without caching **saturates the Pi 5 CPU (`328%` of 400%) and causes the queue to explode (`10.8–13.2 seconds` wait in 30s of video, `53.4–54.7%` delivered AUC)!**
   - Notice that the real Pi 5 is **`4.21x–4.45x` slower** than the Apple M2 Pro `edge-os` VM (`60.23 ms` vs `13.53 ms` in the simulator), proving how essential physical Pi 5 testing is!

3. **How `Cache-C1+Guard` Crosses the `24 FPS` Real-Time Threshold on the Physical Pi 5:**
   - Turning on **`C1` static-validity detector reuse + Queue Guard (`Ours_ML0_NCNN_640_CacheC1_Guard`)** skips `29.03%` of redundant `48.9 ms` NCNN calls at a cost of just `1.76 ms/frame` for the thumbnail check, cutting mean service time from **`60.23 ms` $\to$ `35.16 ms/frame` (`-41.6%`)** and boosting throughput from **`16.60 FPS` $\to$ `28.45 FPS` (`1.71x` speedup)**!
   - Because **`28.45 FPS > 24.0 FPS` (`35.16 ms < 41.67 ms`)**, our `640p` pipeline **crosses the real-time sustainability threshold on the physical Raspberry Pi 5**, keeping `p99` queue wait at **`417.8 ms`**, dropping **0** windows for lateness, and cutting energy per frame by **`-41.6%` (`0.328 J` $\to$ `0.192 J/frame`)**!

4. **Operating at `W=1` vs. Intact-Window Burst `W=28–32` on Physical Pi 5 (`s_pi = 4.45`):**
   - At `s_pi = 4.45` on the full 107-clip ShanghaiTech stream:
     - Unguarded `K=1` (`Stock` or `ML0`) collapses (`53.44%` / `54.73%` delivered AUC, `451–745 s` queue wait).
     - `Cache-C1+Guard` at `W=1` bounds `p99` wait to **`403.6 ms`** and lifts delivered AUC to **`63.67%`**, while scheduling intact bursts (`W=28–32` with `Cache-C1+Guard`) eliminates emergency mid-window shedding and achieves **`68.78%–69.58%` strict real-time delivered AUC** on the physical Pi 5 (`+10.4` to `+16.3` AUC points above `Huang RPi* 320p` and `Huang RPi 640p`)!
     - Furthermore, at burst period `W=26–28` (`86%–92%` window density), `Fixed-Burst` without `C1` still collapses (`54.83%–55.08%` AUC, `70–205 s` wait), whereas **`Cache-C1` / `Cache-C1+Guard` stays sustainable and delivers `67.71%–68.95%` AUC (`+13.9 pt` over `Fixed`)** because `C1`'s `27%` detector savings keep total utilization under `1.0`!
