# Q2-P1-Pi5-Suite: physical Raspberry Pi 5 results (E02 stage profiling, E04 +MV / +INT8, E05 sustained live operation)

*Board:* Raspberry Pi 5 Model B Rev 1.1; kernel 6.18.50+rpt-rpi-2712; governor `ondemand`; idle floor 1.910 W total / 0.535 W core at 54.1 C. Every number below was measured on the physical Pi unless it is labelled *Pi-calibrated simulation* (full datasets replayed with the measured Pi stage pools, no VM multiplier).

## Summary of results

- **E02 (physical stage profile).** The pose detector dominates: ML0 @640 NCNN FP16 takes 73.90 ms/frame at T=1 and 46.29 ms at T=4 (1.60x speed-up, saturating after T=2-3); JPEG decode is ~2.89 ms, the tracker ~1 ms and STG-NF 4-20 ms per job. Full-rate 24 fps is out of reach at T=1 (pose alone is 13.5 FPS).
- **E04 (+INT8 / +MV on Cortex-A76).** At T=4: NCNN FP16 46.43 ms, ORT INT8 43.77 ms (1.06x), ORT FP32 98.83 ms (0.47x); INT8 is slower than NCNN at T=1-2 and only matches it at T=3-4, with no J/frame gain. MV-derived C1 / observer cost is 2.6-5x cheaper than the pixel path (see section 2.1).
- **E05 (2,880-frame live streams at 24 fps, T=4, no thermal throttling in any run).** Fixed-K1: 17.20 FPS, wait p99 46824 ms, 0.327 J/frame, online AUC 47.46%; Cache-C1: 23.00 FPS, wait p99 9583 ms, 0.238 J/frame, online AUC 46.92%; Cache-C1+Guard: 24.01 FPS, wait p99 405 ms, 0.204 J/frame, online AUC 64.49%; Cache-C1+Guard-Burst32: 24.01 FPS, wait p99 250 ms, 0.168 J/frame, online AUC 74.19%. Only the Guard arms keep up with real time on the physical Pi; RSS growth stays within about 3.5 % over 2,880 frames.
- **Pi-calibrated full-dataset simulation**: see section 4 (exact pooled AUC, 2,000-resample video bootstrap).

## 0. Honest accounting

- Power is the sum of the PMIC rail V x I products exposed by `vcgencmd pmic_read_adc` (SoC core + DDR + I/O rails), sampled at 4 Hz; it is not wall-socket power (no USB-C / fan / peripheral losses). Energy J/frame = mean PMIC power over the timed window x mean wall ms/frame; 'active' J/frame subtracts the idle floor.
- The `ondemand` cpufreq governor was left unchanged (changing it needs sudo): the clock ramps within ~10 ms and sat at 2.4 GHz during every timed window (see `freq_mhz`), but idle-floor frequency is 1.5 GHz.
- Cells of about one second (JPEG / H.264 decode, user clips) contain only a few 0.25 s telemetry samples: their power and J/frame are indicative; the timing statistics are exact.
- The user's text states '2,880 frames = 2 full passes over the 12 SHT clips = 2.0 minutes'. 2,880 frames at 24 fps is 120 s, and the 12 SHT timing clips hold 240 frames each, so 2,880 frames is exactly ONE pass of 12 x 240 frames. The live runs use 2,880 frames (2.0 minutes of 24 fps video per arm).
- Thread counts T in {1, 2, 3, 4} are measured (a superset of the requested T in {1, 2, 4} for decode and ORT/NCNN, and {1, 2, 3, 4} for pose).
- Pi-calibrated delivered AUCs use the measured physical pools at the native cell (s = 1.0, CPU budget beta in {0.25, 0.5, 1.0}); there is no VM-to-Pi multiplier. Pixel observer / C1 check costs come from the Pi `mvprof` cells (single thread).
- E02 pose cells for ML0 @640 are the E02 cell and, as second replicate, the E04 ML0 NCNN FP16 cell (same model, same input, same settings); the E02 simulators pool the two.

## 1. E02 stage profiling (physical Pi 5)

### 1.1 Decode (per frame; JPEG `imdecode` vs H.264 `grab` + `retrieve`)

| stage | source | dataset | T | wall mean ms | p50 | p95 | p99 | CPU ms | peak RSS MB | PMIC W | core W | J/frame | T C |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| decode_jpeg_imdecode | jpg | ShanghaiTech 856x480 | 1 | 2.89 | 2.86 | 3.18 | 3.22 | 2.89 | 69 | 3.17 | 1.76 | 0.0091 | 60.0 |
| decode_jpeg_imdecode | jpg | ShanghaiTech 856x480 | 2 | 2.94 | 2.92 | 3.20 | 3.24 | 2.94 | 69 | 3.22 | 1.80 | 0.0095 | 60.6 |
| decode_jpeg_imdecode | jpg | ShanghaiTech 856x480 | 3 | 2.91 | 2.89 | 3.17 | 3.21 | 2.91 | 69 | 3.25 | 1.81 | 0.0094 | 60.4 |
| decode_jpeg_imdecode | jpg | ShanghaiTech 856x480 | 4 | 2.94 | 2.91 | 3.24 | 3.26 | 2.94 | 69 | 3.23 | 1.83 | 0.0095 | 59.9 |
| decode_h264_grab+retrieve | h264_sht | ShanghaiTech | 1 | 2.46 | 2.25 | 3.22 | 8.29 | 2.46 | 74 | 3.44 | 1.95 | 0.0085 | 60.8 |
| decode_h264_grab+retrieve | h264_sht | ShanghaiTech | 2 | 1.74 | 1.63 | 2.92 | 6.95 | 2.69 | 78 | 3.92 | 2.39 | 0.0068 | 61.2 |
| decode_h264_grab+retrieve | h264_sht | ShanghaiTech | 3 | 1.45 | 1.20 | 2.55 | 6.48 | 2.96 | 79 | 4.45 | 2.82 | 0.0064 | 61.7 |
| decode_h264_grab+retrieve | h264_sht | ShanghaiTech | 4 | 1.38 | 1.13 | 2.40 | 6.32 | 3.47 | 82 | 4.54 | 2.91 | 0.0063 | 62.5 |
| decode_h264_grab+retrieve | h264_ub | UBnormal | 1 | 8.58 | 8.01 | 15.06 | 17.48 | 8.58 | 85 | 3.29 | 1.84 | 0.0282 | 60.3 |
| decode_h264_grab+retrieve | h264_ub | UBnormal | 2 | 5.77 | 4.27 | 13.20 | 16.30 | 9.22 | 93 | 3.87 | 2.37 | 0.0223 | 61.0 |
| decode_h264_grab+retrieve | h264_ub | UBnormal | 3 | 4.53 | 3.20 | 9.65 | 13.18 | 10.19 | 95 | 4.34 | 2.82 | 0.0197 | 61.6 |
| decode_h264_grab+retrieve | h264_ub | UBnormal | 4 | 4.10 | 2.67 | 10.44 | 13.92 | 11.02 | 103 | 5.05 | 3.35 | 0.0207 | 62.7 |
| decode_h264_grab+retrieve | h264_user_sht | ShanghaiTech | 1 | 2.35 | 2.25 | 2.66 | 6.50 | 2.35 | 70 | 3.42 | 1.94 | 0.0080 | 59.8 |
| decode_h264_grab+retrieve | h264_user_sht | ShanghaiTech | 2 | 1.64 | 1.25 | 2.74 | 5.65 | 2.54 | 74 | 4.38 | 2.80 | 0.0072 | 59.7 |
| decode_h264_grab+retrieve | h264_user_sht | ShanghaiTech | 4 | 1.25 | 0.97 | 2.24 | 4.36 | 3.02 | 78 | 5.86 | 4.25 | 0.0073 | 57.9 |
| decode_h264_grab+retrieve | h264_user_ub | UBnormal | 1 | 8.83 | 8.22 | 15.79 | 17.68 | 8.83 | 84 | 3.33 | 1.84 | 0.0294 | 60.0 |
| decode_h264_grab+retrieve | h264_user_ub | UBnormal | 2 | 5.93 | 4.09 | 13.79 | 17.15 | 9.39 | 95 | 3.92 | 2.40 | 0.0233 | 60.7 |
| decode_h264_grab+retrieve | h264_user_ub | UBnormal | 4 | 4.25 | 2.45 | 11.32 | 14.44 | 11.20 | 99 | 4.58 | 2.96 | 0.0195 | 60.8 |
| decode_h264_grab | h264_sht | ShanghaiTech | 1 | 1.42 | 1.20 | 2.18 | 7.21 | 1.42 | 74 | 3.44 | 1.95 | 0.0049 | 60.8 |
| decode_h264_grab | h264_sht | ShanghaiTech | 2 | 0.62 | 0.06 | 1.93 | 6.16 | 1.16 | 78 | 3.92 | 2.39 | 0.0024 | 61.2 |
| decode_h264_grab | h264_sht | ShanghaiTech | 3 | 0.22 | 0.06 | 0.93 | 5.16 | 0.48 | 79 | 4.45 | 2.82 | 0.0010 | 61.7 |
| decode_h264_grab | h264_sht | ShanghaiTech | 4 | 0.19 | 0.06 | 0.14 | 5.02 | 0.44 | 82 | 4.54 | 2.91 | 0.0008 | 62.5 |
| decode_h264_grab | h264_ub | UBnormal | 1 | 7.16 | 6.61 | 13.77 | 16.19 | 7.17 | 85 | 3.29 | 1.84 | 0.0236 | 60.3 |
| decode_h264_grab | h264_ub | UBnormal | 2 | 3.86 | 2.29 | 11.93 | 14.94 | 6.43 | 93 | 3.87 | 2.37 | 0.0149 | 61.0 |
| decode_h264_grab | h264_ub | UBnormal | 3 | 2.37 | 0.12 | 7.83 | 11.45 | 5.73 | 95 | 4.34 | 2.82 | 0.0103 | 61.6 |
| decode_h264_grab | h264_ub | UBnormal | 4 | 1.73 | 0.10 | 8.70 | 12.13 | 4.64 | 103 | 5.05 | 3.35 | 0.0087 | 62.7 |
| decode_h264_grab | h264_user_sht | ShanghaiTech | 1 | 1.30 | 1.20 | 1.60 | 5.47 | 1.30 | 70 | 3.42 | 1.94 | 0.0045 | 59.8 |
| decode_h264_grab | h264_user_sht | ShanghaiTech | 2 | 0.51 | 0.06 | 1.63 | 4.84 | 0.96 | 74 | 4.38 | 2.80 | 0.0022 | 59.7 |
| decode_h264_grab | h264_user_sht | ShanghaiTech | 4 | 0.14 | 0.06 | 0.10 | 3.47 | 0.37 | 78 | 5.86 | 4.25 | 0.0008 | 57.9 |
| decode_h264_grab | h264_user_ub | UBnormal | 1 | 7.33 | 6.87 | 14.48 | 16.38 | 7.33 | 84 | 3.33 | 1.84 | 0.0244 | 60.0 |
| decode_h264_grab | h264_user_ub | UBnormal | 2 | 4.00 | 1.55 | 12.60 | 15.54 | 6.57 | 95 | 3.92 | 2.40 | 0.0157 | 60.7 |
| decode_h264_grab | h264_user_ub | UBnormal | 4 | 1.92 | 0.10 | 9.41 | 11.98 | 5.13 | 99 | 4.58 | 2.96 | 0.0088 | 60.8 |
| decode_h264_retrieve | h264_sht | ShanghaiTech | 1 | 1.04 | 1.03 | 1.10 | 1.21 | 1.04 | 74 | 3.44 | 1.95 | 0.0036 | 60.8 |
| decode_h264_retrieve | h264_sht | ShanghaiTech | 2 | 1.12 | 1.10 | 1.94 | 2.62 | 1.54 | 78 | 3.92 | 2.39 | 0.0044 | 61.2 |
| decode_h264_retrieve | h264_sht | ShanghaiTech | 3 | 1.23 | 1.00 | 2.14 | 2.81 | 2.48 | 79 | 4.45 | 2.82 | 0.0055 | 61.7 |
| decode_h264_retrieve | h264_sht | ShanghaiTech | 4 | 1.20 | 1.05 | 2.10 | 2.65 | 3.03 | 82 | 4.54 | 2.91 | 0.0054 | 62.5 |
| decode_h264_retrieve | h264_ub | UBnormal | 1 | 1.42 | 1.30 | 2.09 | 2.14 | 1.42 | 85 | 3.29 | 1.84 | 0.0047 | 60.3 |
| decode_h264_retrieve | h264_ub | UBnormal | 2 | 1.91 | 1.42 | 4.28 | 5.13 | 2.78 | 93 | 3.87 | 2.37 | 0.0074 | 61.0 |
| decode_h264_retrieve | h264_ub | UBnormal | 3 | 2.16 | 1.76 | 4.38 | 5.47 | 4.46 | 95 | 4.34 | 2.82 | 0.0094 | 61.6 |
| decode_h264_retrieve | h264_ub | UBnormal | 4 | 2.38 | 1.96 | 4.60 | 5.91 | 6.38 | 103 | 5.05 | 3.35 | 0.0120 | 62.7 |
| decode_h264_retrieve | h264_user_sht | ShanghaiTech | 1 | 1.05 | 1.04 | 1.09 | 1.20 | 1.05 | 70 | 3.42 | 1.94 | 0.0036 | 59.8 |
| decode_h264_retrieve | h264_user_sht | ShanghaiTech | 2 | 1.13 | 1.10 | 2.04 | 2.24 | 1.57 | 74 | 4.38 | 2.80 | 0.0049 | 59.7 |
| decode_h264_retrieve | h264_user_sht | ShanghaiTech | 4 | 1.10 | 0.90 | 2.04 | 2.33 | 2.65 | 78 | 5.86 | 4.25 | 0.0065 | 57.9 |
| decode_h264_retrieve | h264_user_ub | UBnormal | 1 | 1.50 | 1.31 | 2.11 | 2.14 | 1.50 | 84 | 3.33 | 1.84 | 0.0050 | 60.0 |
| decode_h264_retrieve | h264_user_ub | UBnormal | 2 | 1.93 | 1.42 | 4.49 | 5.17 | 2.82 | 95 | 3.92 | 2.40 | 0.0076 | 60.7 |
| decode_h264_retrieve | h264_user_ub | UBnormal | 4 | 2.33 | 2.08 | 4.16 | 5.23 | 6.07 | 99 | 4.58 | 2.96 | 0.0107 | 60.8 |

### 1.2 Pose detector thread scaling (NCNN FP16, 480 natural L0 SHT frames; UB model = 1,000+ UB frames)

| model | T | wall mean ms | p50 | p95 | p99 | CPU ms | peak RSS MB | PMIC W | core W | J/frame | T C | speedup vs T1 | pre / infer / post |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ML0-640 | 1 | 73.90 | 73.97 | 74.65 | 74.90 | 73.91 | 135 | 4.10 | 2.51 | 0.3032 | 62.8 | 1.00x | pre 4.06 / infer 69.50 / post 0.34 ms; runtime ncnn; input [384, 640] |
| ML0-640 | 2 | 51.65 | 51.55 | 52.54 | 52.90 | 98.61 | 136 | 4.69 | 3.06 | 0.2425 | 65.3 | 1.43x | pre 3.96 / infer 47.35 / post 0.34 ms; runtime ncnn; input [384, 640] |
| ML0-640 | 3 | 46.72 | 46.54 | 48.05 | 48.57 | 128.24 | 136 | 5.41 | 3.71 | 0.2526 | 66.4 | 1.58x | pre 3.94 / infer 42.45 / post 0.33 ms; runtime ncnn; input [384, 640] |
| ML0-640 | 4 | 46.29 | 45.12 | 52.00 | 62.11 | 163.34 | 137 | 5.34 | 3.68 | 0.2473 | 67.5 | 1.60x | pre 4.09 / infer 41.85 / post 0.34 ms; runtime ncnn; input [384, 640] |
| ML0-640-ub | 1 | 90.41 | 90.36 | 91.19 | 91.44 | 90.56 | 151 | 4.08 | 2.50 | 0.3689 | 64.4 | 1.00x | pre 5.63 / infer 84.45 / post 0.33 ms; runtime ncnn; input [448, 640] |
| ML0-640-ub | 2 | 65.74 | 64.65 | 73.03 | 79.24 | 129.18 | 163 | 5.24 | 3.55 | 0.3442 | 66.4 | 1.38x | pre 5.84 / infer 59.57 / post 0.33 ms; runtime ncnn; input [448, 640] |
| ML0-640-ub | 3 | 59.51 | 58.83 | 63.33 | 65.70 | 166.91 | 163 | 5.65 | 3.94 | 0.3363 | 67.3 | 1.52x | pre 5.90 / infer 53.29 / post 0.32 ms; runtime ncnn; input [448, 640] |
| ML0-640-ub | 4 | 58.27 | 56.47 | 66.52 | 77.20 | 208.64 | 175 | 5.97 | 4.24 | 0.3480 | 68.6 | 1.55x | pre 5.55 / infer 52.40 / post 0.32 ms; runtime ncnn; input [448, 640] |
| MLP-640 | 1 | 151.17 | 151.03 | 151.76 | 155.19 | 151.34 | 182 | 4.11 | 2.47 | 0.6208 | 64.2 | 1.00x | pre 4.53 / infer 146.24 / post 0.40 ms; runtime ncnn; input [384, 640] |
| MLP-640 | 2 | 117.25 | 117.11 | 118.60 | 118.91 | 223.66 | 182 | 4.81 | 3.09 | 0.5643 | 66.3 | 1.29x | pre 4.47 / infer 112.38 / post 0.40 ms; runtime ncnn; input [384, 640] |
| MLP-640 | 3 | 109.55 | 109.14 | 111.22 | 111.70 | 300.49 | 185 | 5.24 | 3.52 | 0.5741 | 67.2 | 1.38x | pre 4.38 / infer 104.78 / post 0.40 ms; runtime ncnn; input [384, 640] |
| MLP-640 | 4 | 108.71 | 107.24 | 114.10 | 122.65 | 389.99 | 180 | 5.69 | 3.92 | 0.6184 | 68.6 | 1.39x | pre 4.58 / infer 103.73 / post 0.40 ms; runtime ncnn; input [384, 640] |
| stock-320 | 1 | 31.20 | 31.22 | 31.62 | 31.77 | 31.22 | 119 | 3.87 | 2.30 | 0.1207 | 62.8 | 1.00x | pre 1.74 / infer 29.23 / post 0.22 ms; runtime ncnn; input [320, 320] |
| stock-320 | 2 | 20.90 | 20.87 | 21.44 | 21.84 | 40.20 | 119 | 4.77 | 3.16 | 0.0997 | 63.9 | 1.49x | pre 1.75 / infer 18.92 / post 0.23 ms; runtime ncnn; input [320, 320] |
| stock-320 | 3 | 19.31 | 19.29 | 19.94 | 20.47 | 54.62 | 119 | 5.24 | 3.60 | 0.1011 | 64.3 | 1.62x | pre 1.78 / infer 17.30 / post 0.22 ms; runtime ncnn; input [320, 320] |
| stock-320 | 4 | 18.90 | 18.49 | 20.72 | 30.64 | 69.00 | 120 | 5.63 | 3.99 | 0.1065 | 66.1 | 1.65x | pre 1.86 / infer 16.82 / post 0.22 ms; runtime ncnn; input [320, 320] |
| stock-640 | 1 | 73.94 | 73.95 | 74.53 | 74.75 | 73.95 | 136 | 4.12 | 2.53 | 0.3043 | 63.4 | 1.00x | pre 4.06 / infer 69.60 / post 0.27 ms; runtime ncnn; input [384, 640] |
| stock-640 | 2 | 51.51 | 51.45 | 52.42 | 52.73 | 98.39 | 136 | 4.80 | 3.15 | 0.2470 | 65.1 | 1.44x | pre 3.97 / infer 47.28 / post 0.27 ms; runtime ncnn; input [384, 640] |
| stock-640 | 3 | 46.96 | 46.78 | 48.14 | 48.78 | 129.00 | 136 | 5.57 | 3.87 | 0.2616 | 66.8 | 1.57x | pre 3.97 / infer 42.72 / post 0.27 ms; runtime ncnn; input [384, 640] |
| stock-640 | 4 | 45.95 | 45.10 | 48.56 | 57.95 | 162.48 | 137 | 5.73 | 4.01 | 0.2633 | 67.6 | 1.61x | pre 4.11 / infer 41.57 / post 0.27 ms; runtime ncnn; input [384, 640] |
| stock-800 | 1 | 123.36 | 123.31 | 123.96 | 124.19 | 123.58 | 136 | 4.10 | 2.50 | 0.5055 | 64.3 | 1.00x | pre 7.56 / infer 115.51 / post 0.29 ms; runtime ncnn; input [480, 800] |
| stock-800 | 2 | 87.41 | 87.28 | 88.45 | 88.71 | 165.77 | 135 | 5.01 | 3.31 | 0.4377 | 65.9 | 1.41x | pre 7.29 / infer 79.83 / post 0.30 ms; runtime ncnn; input [480, 800] |
| stock-800 | 3 | 80.14 | 79.87 | 81.69 | 82.16 | 214.88 | 136 | 5.31 | 3.65 | 0.4259 | 66.9 | 1.54x | pre 7.10 / infer 72.74 / post 0.30 ms; runtime ncnn; input [480, 800] |
| stock-800 | 4 | 76.74 | 75.62 | 81.60 | 90.49 | 267.25 | 149 | 5.92 | 4.16 | 0.4542 | 68.5 | 1.61x | pre 6.14 / infer 70.31 / post 0.29 ms; runtime ncnn; input [480, 800] |

### 1.3 Tracker BoT-SORT (T=1): Default (c=0.25, low off) vs Locked two-stage (c=0.20, low=0.05)

| tracker | source | persons | frames | wall mean ms | p50 | p95 | p99 | CPU ms | PMIC W | J/frame |
|---|---|---|---|---|---|---|---|---|---|---|
| default | real SHT ML0 detections | 1 | 543 | 0.40 | 0.40 | 0.44 | 0.48 | 0.40 | 3.28 | 0.00131 |
| default | real SHT ML0 detections | 3 | 893 | 0.59 | 0.58 | 0.64 | 0.67 | 0.59 | 3.28 | 0.00192 |
| default | real SHT ML0 detections | 5 | 717 | 0.75 | 0.75 | 0.81 | 0.85 | 0.75 | 3.28 | 0.00247 |
| default | real SHT ML0 detections | 8 | 96 | 1.00 | 1.00 | 1.05 | 1.09 | 1.00 | 3.28 | 0.00329 |
| default | real SHT ML0 detections | all | 4908 | 0.65 | 0.66 | 0.99 | 1.12 | 0.65 | 3.28 | 0.00213 |
| locked | real SHT ML0 detections | 1 | 299 | 0.38 | 0.38 | 0.43 | 0.45 | 0.38 | 3.28 | 0.00126 |
| locked | real SHT ML0 detections | 3 | 298 | 0.56 | 0.56 | 0.62 | 0.65 | 0.56 | 3.28 | 0.00184 |
| locked | real SHT ML0 detections | 5 | 527 | 0.72 | 0.71 | 0.81 | 0.85 | 0.72 | 3.28 | 0.00235 |
| locked | real SHT ML0 detections | 8 | 383 | 0.90 | 0.90 | 1.02 | 1.09 | 0.90 | 3.28 | 0.00296 |
| locked | real SHT ML0 detections | 12 | 195 | 1.18 | 1.18 | 1.30 | 1.38 | 1.18 | 3.28 | 0.00388 |
| locked | real SHT ML0 detections | all | 4908 | 0.78 | 0.77 | 1.27 | 1.44 | 0.78 | 3.28 | 0.00256 |
| default | synthetic persons | 10 | 600 | 1.10 | 1.10 | 1.12 | 1.16 | 1.10 | 3.28 | 0.00362 |
| default | synthetic persons | 12 | 600 | 1.26 | 1.26 | 1.28 | 1.34 | 1.26 | 3.28 | 0.00413 |
| default | synthetic persons | 16 | 600 | 1.57 | 1.57 | 1.59 | 1.65 | 1.57 | 3.28 | 0.00515 |

### 1.4 Scorer STG-NF (window assembly + per-window normalisation + PyTorch forward)

| dataset | T | N windows | wall mean ms | p50 | p95 | p99 | CPU ms | PMIC W | J/job | assemble / normalize / flow |
|---|---|---|---|---|---|---|---|---|---|---|
| ShanghaiTech | 1 | 1 | 4.35 | 4.33 | 4.42 | 4.71 | 4.36 | 3.43 | 0.01494 | assemble 0.063 / normalize 0.166 / flow 4.122 ms |
| ShanghaiTech | 1 | 3 | 6.84 | 6.81 | 7.17 | 7.38 | 6.85 | 3.43 | 0.02349 | assemble 0.123 / normalize 0.252 / flow 6.467 ms |
| ShanghaiTech | 1 | 5 | 7.96 | 7.93 | 8.14 | 8.57 | 7.97 | 3.43 | 0.02733 | assemble 0.172 / normalize 0.322 / flow 7.467 ms |
| ShanghaiTech | 1 | 12 | 11.90 | 11.86 | 12.20 | 12.36 | 11.91 | 3.43 | 0.04085 | assemble 0.348 / normalize 0.568 / flow 10.983 ms |
| ShanghaiTech | 1 | 24 | 19.85 | 19.79 | 20.30 | 20.41 | 19.86 | 3.43 | 0.06817 | assemble 0.654 / normalize 0.973 / flow 18.228 ms |
| ShanghaiTech | 4 | 1 | 4.61 | 4.56 | 4.66 | 5.89 | 15.22 | 5.06 | 0.02332 | assemble 0.065 / normalize 0.169 / flow 4.372 ms |
| ShanghaiTech | 4 | 3 | 8.66 | 8.54 | 9.45 | 10.59 | 29.57 | 5.06 | 0.04386 | assemble 0.126 / normalize 0.257 / flow 8.281 ms |
| ShanghaiTech | 4 | 5 | 9.12 | 9.03 | 10.12 | 10.41 | 34.42 | 5.06 | 0.04617 | assemble 0.180 / normalize 0.326 / flow 8.614 ms |
| ShanghaiTech | 4 | 12 | 11.06 | 10.91 | 11.92 | 13.45 | 40.29 | 5.06 | 0.05597 | assemble 0.357 / normalize 0.572 / flow 10.126 ms |
| ShanghaiTech | 4 | 24 | 14.64 | 14.40 | 15.95 | 17.59 | 48.98 | 5.06 | 0.07413 | assemble 0.673 / normalize 0.990 / flow 12.979 ms |
| UBnormal | 1 | 1 | 4.13 | 4.12 | 4.21 | 4.46 | 4.13 | 3.39 | 0.01399 | assemble 0.052 / normalize 0.155 / flow 3.922 ms |
| UBnormal | 1 | 3 | 6.16 | 6.14 | 6.31 | 6.58 | 6.17 | 3.39 | 0.02088 | assemble 0.100 / normalize 0.223 / flow 5.840 ms |
| UBnormal | 1 | 5 | 6.82 | 6.79 | 7.07 | 7.36 | 6.83 | 3.39 | 0.02311 | assemble 0.137 / normalize 0.272 / flow 6.412 ms |
| UBnormal | 1 | 12 | 9.15 | 9.11 | 9.57 | 9.71 | 9.16 | 3.39 | 0.03100 | assemble 0.267 / normalize 0.446 / flow 8.438 ms |
| UBnormal | 1 | 24 | 14.31 | 14.26 | 14.74 | 14.91 | 14.31 | 3.39 | 0.04846 | assemble 0.491 / normalize 0.726 / flow 13.089 ms |
| UBnormal | 4 | 1 | 4.37 | 4.31 | 4.60 | 5.58 | 14.81 | 4.90 | 0.02140 | assemble 0.056 / normalize 0.159 / flow 4.152 ms |
| UBnormal | 4 | 3 | 8.06 | 7.97 | 8.73 | 9.91 | 25.98 | 4.90 | 0.03953 | assemble 0.103 / normalize 0.225 / flow 7.736 ms |
| UBnormal | 4 | 5 | 8.61 | 8.50 | 9.29 | 10.47 | 29.27 | 4.90 | 0.04222 | assemble 0.142 / normalize 0.275 / flow 8.195 ms |
| UBnormal | 4 | 12 | 9.70 | 9.57 | 10.43 | 12.24 | 36.01 | 4.90 | 0.04757 | assemble 0.271 / normalize 0.445 / flow 8.987 ms |
| UBnormal | 4 | 24 | 11.57 | 11.42 | 12.53 | 14.09 | 40.92 | 4.90 | 0.05674 | assemble 0.498 / normalize 0.730 / flow 10.347 ms |

### 1.5 End-to-end service cells

| cell | T | wall mean ms | p50 | p95 | p99 | CPU ms | peak RSS MB | PMIC W | core W | J/frame | T C | decode / pose / track / score ms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ShanghaiTech/h264 | 1 | 82.14 | 83.09 | 87.39 | 90.44 | 82.32 | 865 | 4.08 | 2.49 | 0.3353 | 63.4 | decode 2.58 pose 74.30 track 0.97 score 4.28 ms |
| ShanghaiTech/h264 | 4 | 56.40 | 56.06 | 65.56 | 102.36 | 188.74 | 883 | 5.54 | 3.87 | 0.3122 | 69.1 | decode 1.28 pose 48.37 track 0.97 score 5.77 ms |
| UBnormal/h264 | 4 | 70.09 | 68.04 | 83.03 | 140.36 | 236.26 | 910 | 5.71 | 4.00 | 0.4004 | 68.4 | decode 2.35 pose 59.61 track 0.87 score 7.26 ms |

## 2. E04 on the Cortex-A76: NCNN FP16 vs ORT INT8 vs ORT FP32 (ML0 @640, 480 frames, mean of 2 replicates)

| T | runtime | wall mean ms | p50 | p95 | p99 | CPU ms | peak RSS MB | PMIC W | core W | J/frame | T C | speed vs NCNN | J/frame vs NCNN |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | ML0-640-ncnn | 73.59 | 73.53 | 74.22 | 74.90 | 73.60 | 135 | 4.10 | 2.53 | 0.3018 | 63.3 | 1.00x | 1.00x |
| 1 | ML0-640-ort-fp32 | 247.63 | 247.63 | 248.48 | 248.83 | 247.62 | 157 | 3.98 | 2.51 | 0.9854 | 63.9 | 0.30x | 3.27x |
| 1 | ML0-640-ort-int8 | 88.15 | 88.10 | 88.58 | 89.23 | 88.18 | 125 | 3.71 | 2.24 | 0.3274 | 62.5 | 0.83x | 1.08x |
| 2 | ML0-640-ncnn | 50.95 | 50.82 | 52.06 | 52.56 | 96.42 | 136 | 4.87 | 3.25 | 0.2481 | 61.4 | 1.00x | 1.00x |
| 2 | ML0-640-ort-fp32 | 137.98 | 137.89 | 139.02 | 142.01 | 275.92 | 157 | 5.35 | 3.83 | 0.7377 | 65.5 | 0.37x | 2.97x |
| 2 | ML0-640-ort-int8 | 55.40 | 55.29 | 56.05 | 57.50 | 110.85 | 125 | 4.81 | 3.28 | 0.2665 | 61.2 | 0.92x | 1.07x |
| 3 | ML0-640-ncnn | 48.13 | 48.09 | 49.74 | 50.99 | 129.73 | 136 | 5.48 | 3.81 | 0.2639 | 65.7 | 1.00x | 1.00x |
| 3 | ML0-640-ort-fp32 | 106.46 | 104.63 | 115.52 | 117.69 | 316.85 | 157 | 6.55 | 4.97 | 0.6968 | 69.9 | 0.45x | 2.64x |
| 3 | ML0-640-ort-int8 | 48.29 | 47.79 | 52.40 | 57.63 | 144.19 | 125 | 5.75 | 4.17 | 0.2778 | 66.7 | 1.00x | 1.05x |
| 4 | ML0-640-ncnn | 46.43 | 45.23 | 50.89 | 68.58 | 163.81 | 137 | 5.81 | 4.09 | 0.2697 | 67.9 | 1.00x | 1.00x |
| 4 | ML0-640-ort-fp32 | 98.83 | 97.29 | 105.50 | 109.24 | 392.22 | 157 | 7.36 | 5.75 | 0.7272 | 73.5 | 0.47x | 2.70x |
| 4 | ML0-640-ort-int8 | 43.77 | 43.05 | 47.25 | 51.09 | 173.72 | 125 | 6.81 | 5.19 | 0.2981 | 69.5 | 1.06x | 1.11x |

### 2.1 H.264 motion-vector export vs pixel-diff C1 (per frame, Pi CPU)

| clips | T | method | wall mean ms | p95 | CPU ms | PMIC W |
|---|---|---|---|---|---|---|
| ShanghaiTech 856x480 (x264 IPPP) | 1 | bgr | 0.971 | 0.998 | 0.972 | 3.29 |
| ShanghaiTech 856x480 (x264 IPPP) | 1 | cv2_grab | 1.282 | 1.575 | 1.283 | 3.29 |
| ShanghaiTech 856x480 (x264 IPPP) | 1 | cv2_retrieve | 1.041 | 1.075 | 1.041 | 3.29 |
| ShanghaiTech 856x480 (x264 IPPP) | 1 | decode_mv | 1.338 | 1.645 | 1.339 | 3.29 |
| ShanghaiTech 856x480 (x264 IPPP) | 1 | decode_plain | 1.380 | 1.675 | 1.381 | 3.29 |
| ShanghaiTech 856x480 (x264 IPPP) | 1 | gray | 0.878 | 0.904 | 0.878 | 3.29 |
| ShanghaiTech 856x480 (x264 IPPP) | 1 | mv_c1 | 0.443 | 0.478 | 0.444 | 3.29 |
| ShanghaiTech 856x480 (x264 IPPP) | 1 | mv_grid | 0.263 | 0.290 | 0.263 | 3.29 |
| ShanghaiTech 856x480 (x264 IPPP) | 1 | mv_obs | 0.447 | 0.476 | 0.448 | 3.29 |
| ShanghaiTech 856x480 (x264 IPPP) | 1 | pix_c1 | 1.170 | 1.207 | 1.171 | 3.29 |
| ShanghaiTech 856x480 (x264 IPPP) | 1 | pix_obs | 1.767 | 1.862 | 1.768 | 3.29 |
| ShanghaiTech 856x480 (x264 IPPP) | 4 | bgr | 1.256 | 2.257 | 2.224 | 4.17 |
| ShanghaiTech 856x480 (x264 IPPP) | 4 | cv2_grab | 0.166 | 0.094 | 0.331 | 4.17 |
| ShanghaiTech 856x480 (x264 IPPP) | 4 | cv2_retrieve | 1.096 | 2.018 | 2.728 | 4.17 |
| ShanghaiTech 856x480 (x264 IPPP) | 4 | decode_mv | 0.210 | 0.139 | 0.387 | 4.17 |
| ShanghaiTech 856x480 (x264 IPPP) | 4 | decode_plain | 0.131 | 0.131 | 0.150 | 4.17 |
| ShanghaiTech 856x480 (x264 IPPP) | 4 | gray | 1.526 | 1.625 | 3.099 | 4.17 |
| ShanghaiTech 856x480 (x264 IPPP) | 4 | mv_c1 | 0.578 | 0.669 | 1.554 | 4.17 |
| ShanghaiTech 856x480 (x264 IPPP) | 4 | mv_grid | 0.350 | 0.423 | 0.783 | 4.17 |
| ShanghaiTech 856x480 (x264 IPPP) | 4 | mv_obs | 0.563 | 0.644 | 1.256 | 4.17 |
| ShanghaiTech 856x480 (x264 IPPP) | 4 | pix_c1 | 1.841 | 1.956 | 3.433 | 4.17 |
| ShanghaiTech 856x480 (x264 IPPP) | 4 | pix_obs | 2.570 | 2.706 | 4.166 | 4.17 |
| UBnormal 1280x720 | 1 | bgr | 1.060 | 1.755 | 1.061 | 3.40 |
| UBnormal 1280x720 | 1 | cv2_grab | 2.033 | 2.364 | 2.034 | 3.40 |
| UBnormal 1280x720 | 1 | cv2_retrieve | 1.439 | 2.077 | 1.438 | 3.40 |
| UBnormal 1280x720 | 1 | decode_mv | 2.237 | 2.593 | 2.239 | 3.40 |
| UBnormal 1280x720 | 1 | decode_plain | 2.227 | 2.594 | 2.228 | 3.40 |
| UBnormal 1280x720 | 1 | gray | 1.773 | 1.876 | 1.774 | 3.40 |
| UBnormal 1280x720 | 1 | mv_c1 | 0.659 | 0.707 | 0.661 | 3.40 |
| UBnormal 1280x720 | 1 | mv_grid | 0.428 | 0.469 | 0.429 | 3.40 |
| UBnormal 1280x720 | 1 | mv_obs | 0.646 | 0.694 | 0.648 | 3.40 |
| UBnormal 1280x720 | 1 | pix_c1 | 2.189 | 2.298 | 2.191 | 3.40 |
| UBnormal 1280x720 | 1 | pix_obs | 3.353 | 3.529 | 3.356 | 3.40 |
| UBnormal 1280x720 | 4 | bgr | 1.846 | 3.173 | 3.821 | 4.14 |
| UBnormal 1280x720 | 4 | cv2_grab | 0.401 | 0.115 | 0.661 | 4.14 |
| UBnormal 1280x720 | 4 | cv2_retrieve | 2.070 | 3.301 | 5.481 | 4.14 |
| UBnormal 1280x720 | 4 | decode_mv | 0.475 | 0.206 | 0.729 | 4.14 |
| UBnormal 1280x720 | 4 | decode_plain | 0.250 | 0.139 | 0.324 | 4.14 |
| UBnormal 1280x720 | 4 | gray | 2.384 | 2.617 | 5.428 | 4.14 |
| UBnormal 1280x720 | 4 | mv_c1 | 1.088 | 1.261 | 2.823 | 4.14 |
| UBnormal 1280x720 | 4 | mv_grid | 0.754 | 0.916 | 1.711 | 4.14 |
| UBnormal 1280x720 | 4 | mv_obs | 1.015 | 1.179 | 2.443 | 4.14 |
| UBnormal 1280x720 | 4 | pix_c1 | 2.799 | 3.053 | 5.870 | 4.14 |
| UBnormal 1280x720 | 4 | pix_obs | 4.074 | 4.407 | 7.191 | 4.14 |

*ShanghaiTech* derived costs (Pi, margin 1.0): MV export overhead (decode_mv - decode_plain, T=4, every frame) 0.079 ms wall / 0.238 ms CPU; MV observer 0.448 ms, MV C1 0.444 ms vs pixel observer 1.768 ms, pixel C1 1.171 ms.

*UBnormal* derived costs (Pi, margin 1.0): MV export overhead (decode_mv - decode_plain, T=4, every frame) 0.225 ms wall / 0.406 ms CPU; MV observer 0.648 ms, MV C1 0.661 ms vs pixel observer 3.356 ms, pixel C1 2.191 ms.

## 3. E05 sustained live operation (physical, 2,880 frames per arm, 24.0 fps real-time arrival, T=4)

| arm | achieved FPS | wait p50 ms | p95 | p99 | max | PMIC W | core W | J/frame | T start C | T max C | throttled |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Fixed-K1 | 17.20 | 22602.8 | 44531.9 | 46824.0 | 47392.7 | 5.63 | 4.04 | 0.3271 | 60.0 | 76.5 | 0x0 -> 0x0 |
| Cache-C1 | 23.00 | 4602.2 | 8668.0 | 9582.6 | 9656.0 | 5.48 | 3.91 | 0.2381 | 61.1 | 74.9 | 0x0 -> 0x0 |
| Cache-C1+Guard | 24.01 | 89.5 | 357.4 | 405.5 | 421.2 | 4.89 | 3.35 | 0.2036 | 61.1 | 73.8 | 0x0 -> 0x0 |
| Cache-C1+Guard-Burst32 | 24.01 | 7.4 | 190.9 | 250.3 | 321.1 | 4.04 | 2.57 | 0.1685 | 62.2 | 71.0 | 0x0 -> 0x0 |

### 3.1 RSS growth (MB at frame 0 / 720 / 1440 / 2880)

| arm | RSS 0 | RSS 720 | RSS 1440 | RSS 2880 | growth @720 | @1440 | @2880 |
|---|---|---|---|---|---|---|---|
| Fixed-K1 | 859.2 | 886.1 | 886.6 | 889.3 | 3.13 % | 3.19 % | 3.51 % |
| Cache-C1 | 861.5 | 887.4 | 888.6 | 891.9 | 3.01 % | 3.15 % | 3.53 % |
| Cache-C1+Guard | 861.4 | 886.3 | 888.1 | 889.6 | 2.89 % | 3.10 % | 3.28 % |
| Cache-C1+Guard-Burst32 | 861.4 | 886.3 | 887.2 | 888.4 | 2.90 % | 3.00 % | 3.14 % |

### 3.2 Pipeline behaviour, online delivered AUC and the Pi-calibrated simulation of the same stream

| arm | observed frac. | C1 hit rate (in-run) | guard shed frac. | guard on/off | windows scored | dropped late | online AUC % [95% CI, 12 videos] | sim AUC % | sim wait p99 ms | live wait p99 ms | online == offline |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Fixed-K1 | 1.000 | 0.000 | 0.000 | 0 | 2372 | 2272 | 47.46 [44.2, 50.2] | 58.38 | 65997 | 46824 | yes |
| Cache-C1 | 1.000 | 0.338 | 0.000 | 0 | 2369 | 2107 | 46.92 [43.5, 53.1] | 59.59 | 12446 | 9583 | yes |
| Cache-C1+Guard | 0.906 | 0.370 | 0.075 | 62 | 1433 | 0 | 64.49 [50.3, 78.1] | 73.73 | 400 | 405 | yes |
| Cache-C1+Guard-Burst32 | 0.700 | 0.338 | 0.000 | 0 | 73 | 0 | 74.19 [65.9, 82.5] | 79.19 | 183 | 250 | yes |

### 3.3 SoC temperature trajectory (every 360 frames: frame, C, W)

| arm | trajectory |
|---|---|
| Fixed-K1 | 0: 60.0C 2.23W; 360: 68.3C 6.61W; 720: 69.4C 4.94W; 1080: 71.1C 6.92W; 1440: 71.1C 6.13W; 1800: 72.2C 5.24W; 2160: 73.2C 5.45W; 2520: 74.9C 5.43W; 2879: 73.8C 3.39W |
| Cache-C1 | 0: 61.7C 2.30W; 360: 69.4C 3.80W; 720: 69.4C 4.71W; 1080: 71.1C 5.95W; 1440: 71.6C 5.32W; 1800: 72.7C 7.17W; 2160: 73.8C 3.48W; 2520: 73.2C 7.04W; 2879: 73.8C 5.00W |
| Cache-C1+Guard | 0: 61.7C 2.29W; 360: 68.3C 6.79W; 720: 67.2C 2.36W; 1080: 69.4C 5.09W; 1440: 69.4C 4.87W; 1800: 70.5C 5.40W; 2160: 71.1C 3.77W; 2520: 72.2C 4.63W; 2879: 70.5C 3.36W |
| Cache-C1+Guard-Burst32 | 0: 61.2C 2.28W; 360: 67.2C 6.55W; 720: 65.6C 2.35W; 1080: 67.2C 3.74W; 1440: 65.0C 2.32W; 1800: 68.3C 7.23W; 2160: 65.6C 7.37W; 2520: 70.5C 3.97W; 2879: 65.6C 2.35W |

## 4. Pi-calibrated full-dataset delivered AUC (simulation with the measured Pi stage pools; exact pooled AUC, 2,000-resample video bootstrap)

### 4.1 ShanghaiTech: 107 videos, 40,791 frames; native cells s = 1.0

*K=1 arms (Stock-Default-K1, ML0-Default-K1, Full-Rate) process every frame whatever the CPU budget: their value is identical in the three cells and shows the real-time delivered AUC of the unthrottled pipeline on the Pi (E02 question). Cost-matched arms (Fixed, Cache-C1, +Guard, ReTrack, Stock-Baseline, ML0-Default-Fixed) are calibrated to beta x the Full-Rate CPU cost of the cell (E03 / E05 questions).*

**E02 / E03 / E05 arms, all videos** (AUC %, [95% CI]):

| arm | beta=0.25 | beta=0.50 | beta=1.00 |
|---|---|---|---|
| Stock-Default-K1 | 50.14 [50.00, 50.48] | 50.14 [50.00, 50.48] | 50.14 [50.00, 50.48] |
| ML0-Default-K1 | 49.90 [49.65, 50.00] | 49.90 [49.65, 50.00] | 49.90 [49.65, 50.00] |
| Full-Rate | 49.93 [49.76, 50.00] | 49.93 [49.76, 50.00] | 49.93 [49.76, 50.00] |
| Stock-Baseline | 55.15 [50.83, 59.46] | 60.14 [55.98, 64.40] | 50.14 [50.00, 50.48] |
| ML0-Default-Fixed | 61.21 [57.67, 64.66] | 67.72 [64.54, 70.85] | 49.90 [49.65, 50.00] |
| Fixed | 63.39 [59.82, 66.78] | 69.58 [66.72, 72.16] | 49.93 [49.76, 50.00] |
| Fixed+Guard | 63.39 [59.82, 66.78] | 69.58 [66.72, 72.16] | 64.15 [62.18, 65.95] |
| Cache-C1 | 65.75 [62.58, 68.77] | 70.71 [67.84, 73.36] | 50.21 [47.23, 53.36] |
| Cache-C1+Guard | 65.75 [62.58, 68.77] | 70.57 [67.65, 73.29] | 66.77 [63.53, 69.73] |
| C1-ReTrack | 65.07 [61.93, 68.22] | 69.80 [66.78, 72.60] | 50.38 [47.48, 53.42] |
| C1-ReTrack+Guard | 65.07 [61.93, 68.22] | 69.91 [67.00, 72.60] | 65.88 [62.69, 68.98] |

**E02 / E03 / E05 arms, test_confirmation split** (AUC %, [95% CI]):

| arm | beta=0.25 | beta=0.50 | beta=1.00 |
|---|---|---|---|
| Stock-Default-K1 | 50.22 [50.00, 50.76] | 50.22 [50.00, 50.76] | 50.22 [50.00, 50.76] |
| ML0-Default-K1 | 49.88 [49.53, 50.00] | 49.88 [49.53, 50.00] | 49.88 [49.53, 50.00] |
| Full-Rate | 49.93 [49.67, 50.00] | 49.93 [49.67, 50.00] | 49.93 [49.67, 50.00] |
| Stock-Baseline | 53.44 [48.24, 58.76] | 59.21 [53.73, 64.62] | 50.22 [50.00, 50.76] |
| ML0-Default-Fixed | 59.22 [54.36, 63.85] | 68.23 [64.12, 72.21] | 49.88 [49.53, 50.00] |
| Fixed | 61.35 [56.82, 65.88] | 69.23 [65.87, 72.57] | 49.93 [49.67, 50.00] |
| Fixed+Guard | 61.35 [56.82, 65.88] | 69.23 [65.87, 72.57] | 64.03 [61.34, 66.63] |
| Cache-C1 | 65.43 [61.24, 69.59] | 70.80 [67.27, 74.20] | 51.94 [47.80, 56.01] |
| Cache-C1+Guard | 65.43 [61.24, 69.59] | 70.53 [66.90, 74.06] | 66.94 [63.02, 70.61] |
| C1-ReTrack | 64.18 [60.10, 68.25] | 69.94 [66.17, 73.36] | 51.64 [47.63, 55.70] |
| C1-ReTrack+Guard | 64.18 [60.10, 68.25] | 69.56 [65.83, 73.04] | 65.90 [62.01, 69.56] |

**Contrasts (A - B, AUC points, mean over the 3 native cells; * = 95% CI excludes 0):**

| contrast | subset | mean | beta=0.25 | beta=0.50 | beta=1.00 |
|---|---|---|---|---|---|
| E02 K=1 ladder: tracker (Full-Rate - ML0-Default-K1) | all | +0.04 [+0.00, +0.11] | +0.04 [+0.00, +0.11] | +0.04 [+0.00, +0.11] | +0.04 [+0.00, +0.11] |
| E02 K=1 ladder: tracker (Full-Rate - ML0-Default-K1) | confirmation | +0.05 [+0.00, +0.15] | +0.05 [+0.00, +0.15] | +0.05 [+0.00, +0.15] | +0.05 [+0.00, +0.15] |
| E02 K=1 ladder: frontend (ML0-Default-K1 - Stock-Default-K1) | all | -0.24 [-0.78, +0.00] | -0.24 [-0.78, +0.00] | -0.24 [-0.78, +0.00] | -0.24 [-0.78, +0.00] |
| E02 K=1 ladder: frontend (ML0-Default-K1 - Stock-Default-K1) | confirmation | -0.34 [-1.14, +0.00] | -0.34 [-1.14, +0.00] | -0.34 [-1.14, +0.00] | -0.34 [-1.14, +0.00] |
| E03 Cache-C1 - Fixed | all | +1.26 [-0.30, +2.90] | +2.37 [-0.31, +4.94] | +1.13 [-0.55, +2.86] | +0.28 [-2.72, +3.42] |
| E03 Cache-C1 - Fixed | confirmation | +2.56 [+0.77, +4.42]* | +4.08 [+1.08, +7.21]* | +1.57 [-0.47, +3.51] | +2.01 [-2.04, +6.08] |
| E03 Cache-C1 - Stock-Baseline | all | +7.08 [+4.39, +9.83]* | +10.60 [+6.51, +14.62]* | +10.57 [+7.13, +14.22]* | +0.07 [-3.00, +3.25] |
| E03 Cache-C1 - Stock-Baseline | confirmation | +8.43 [+4.91, +12.07]* | +12.00 [+7.11, +16.90]* | +11.59 [+6.91, +16.24]* | +1.72 [-2.59, +5.84] |
| E05 Cache-C1+Guard - Cache-C1 | all | +5.47 [+4.16, +6.78]* | +0.00 [+0.00, +0.00] | -0.14 [-0.62, +0.15] | +16.56 [+12.68, +20.53]* |
| E05 Cache-C1+Guard - Cache-C1 | confirmation | +4.91 [+3.28, +6.48]* | +0.00 [+0.00, +0.00] | -0.27 [-0.93, +0.05] | +15.00 [+10.22, +19.56]* |
| E05 Cache-C1+Guard - Fixed | all | +6.73 [+5.11, +8.30]* | +2.37 [-0.31, +4.94] | +0.99 [-0.73, +2.73] | +16.83 [+13.61, +19.82]* |
| E05 Cache-C1+Guard - Fixed | confirmation | +7.46 [+5.52, +9.35]* | +4.08 [+1.08, +7.21]* | +1.30 [-0.89, +3.34] | +17.01 [+13.15, +20.68]* |
| E05 Fixed+Guard - Fixed | all | +4.74 [+4.08, +5.35]* | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | +14.22 [+12.24, +16.04]* |
| E05 Fixed+Guard - Fixed | confirmation | +4.70 [+3.79, +5.56]* | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | +14.10 [+11.38, +16.69]* |
| E05 C1-ReTrack - Fixed | all | +0.79 [-0.83, +2.47] | +1.69 [-1.14, +4.58] | +0.22 [-1.60, +2.15] | +0.45 [-2.46, +3.48] |
| E05 C1-ReTrack - Fixed | confirmation | +1.75 [-0.23, +3.80] | +2.83 [-0.83, +6.44] | +0.71 [-1.27, +2.62] | +1.71 [-2.28, +5.76] |

**E04 arms (Pi pools + Pi-measured MV costs, +INT8 on the Cortex-A76 pool), all videos:**

| arm | beta=0.25 | beta=0.50 | beta=1.00 | CPU ms/video-s (beta=0.5) |
|---|---|---|---|---|
| C1-INT8 | 66.96 [63.46, 70.00] | 68.93 [65.93, 71.60] | 54.66 [50.75, 58.86] | 2289 |
| C1-INT8@W | 66.10 [62.56, 69.45] | 71.33 [68.57, 73.86] | 54.66 [50.75, 58.86] | 2428 |
| C1-INT8|uncorrected | 66.96 [63.46, 70.00] | 68.93 [65.94, 71.60] | 54.84 [50.90, 59.11] | 2290 |
| C1-MV | 65.68 [62.37, 68.69] | 69.70 [66.90, 72.21] | 58.77 [54.45, 63.19] | 2292 |
| C1-MV-INT8 | 66.44 [63.04, 69.45] | 70.00 [66.95, 72.69] | 59.03 [54.56, 63.63] | 2288 |
| C1-MV-INT8@W | 66.12 [62.46, 69.46] | 70.88 [68.18, 73.33] | 59.03 [54.56, 63.63] | 2435 |
| C1-MV@W | 65.82 [62.70, 68.83] | 70.71 [67.83, 73.30] | 58.77 [54.45, 63.19] | 2315 |
| C1-MV|cost1.25 | 65.67 [62.36, 68.68] | 69.60 [66.83, 72.19] | 57.57 [53.40, 62.01] | 2281 |
| C1-Pixel | 65.75 [62.58, 68.77] | 70.71 [67.84, 73.36] | 50.21 [47.23, 53.36] | 2291 |

**E04 contrasts:**

| contrast | subset | mean | beta=0.25 | beta=0.50 | beta=1.00 |
|---|---|---|---|---|---|
| E04 +MV: C1-MV - C1-Pixel (matched CPU) | all | +2.49 [+0.59, +4.39]* | -0.07 [-0.88, +0.72] | -1.01 [-2.14, +0.19] | +8.56 [+3.18, +14.00]* |
| E04 +MV: C1-MV - C1-Pixel (matched CPU) | confirmation | +1.61 [-0.64, +3.89] | -0.30 [-1.41, +0.71] | -1.28 [-2.62, +0.02] | +6.41 [+0.08, +13.40]* |
| E04 +MV: C1-MV@W - C1-Pixel (same schedule) | all | +2.88 [+1.04, +4.67]* | +0.06 [-0.34, +0.50] | +0.00 [-0.44, +0.46] | +8.56 [+3.18, +14.00]* |
| E04 +MV: C1-MV@W - C1-Pixel (same schedule) | confirmation | +2.10 [-0.04, +4.39] | -0.18 [-0.68, +0.33] | +0.07 [-0.50, +0.64] | +6.41 [+0.08, +13.40]* |
| E04 +INT8: C1-INT8 - C1-Pixel (matched CPU) | all | +1.29 [-0.19, +2.86] | +1.20 [-1.35, +3.73] | -1.78 [-3.20, -0.36]* | +4.45 [+1.17, +7.84]* |
| E04 +INT8: C1-INT8 - C1-Pixel (matched CPU) | confirmation | +1.80 [+0.05, +3.60]* | +2.31 [-0.50, +5.15] | -2.08 [-3.93, -0.26]* | +5.18 [+0.99, +9.69]* |
| E04 +INT8: C1-INT8@W - C1-Pixel (same schedule) | all | +1.81 [+0.53, +3.17]* | +0.35 [-1.01, +1.79] | +0.62 [-0.36, +1.56] | +4.45 [+1.17, +7.84]* |
| E04 +INT8: C1-INT8@W - C1-Pixel (same schedule) | confirmation | +2.14 [+0.53, +3.87]* | +0.63 [-0.75, +1.97] | +0.61 [-0.51, +1.79] | +5.18 [+0.99, +9.69]* |
| E04 +MV+INT8: C1-MV-INT8 - C1-MV (matched CPU) | all | +0.44 [-0.94, +1.84] | +0.76 [-1.79, +3.31] | +0.31 [-1.27, +1.93] | +0.26 [-2.26, +2.94] |
| E04 +MV+INT8: C1-MV-INT8 - C1-MV (matched CPU) | confirmation | +1.23 [-0.33, +2.74] | +3.06 [+0.57, +5.67]* | +0.32 [-1.46, +2.12] | +0.32 [-3.06, +3.95] |

### 4.2 UBnormal: 211 videos, 92,640 frames; native cells s = 1.0

*K=1 arms (Stock-Default-K1, ML0-Default-K1, Full-Rate) process every frame whatever the CPU budget: their value is identical in the three cells and shows the real-time delivered AUC of the unthrottled pipeline on the Pi (E02 question). Cost-matched arms (Fixed, Cache-C1, +Guard, ReTrack, Stock-Baseline, ML0-Default-Fixed) are calibrated to beta x the Full-Rate CPU cost of the cell (E03 / E05 questions).*

**E02 / E03 / E05 arms, all videos** (AUC %, [95% CI]):

| arm | beta=0.25 | beta=0.50 | beta=1.00 |
|---|---|---|---|
| Stock-Default-K1 | 49.98 [49.92, 50.00] | 49.98 [49.92, 50.00] | 49.98 [49.92, 50.00] |
| ML0-Default-K1 | 49.98 [49.92, 50.00] | 49.98 [49.92, 50.00] | 49.98 [49.92, 50.00] |
| Full-Rate | 49.98 [49.92, 50.00] | 49.98 [49.92, 50.00] | 49.98 [49.92, 50.00] |
| Stock-Baseline | 54.80 [50.36, 58.94] | 49.52 [48.95, 49.91] | 49.98 [49.92, 50.00] |
| ML0-Default-Fixed | 55.70 [51.32, 60.32] | 49.86 [49.38, 50.26] | 49.98 [49.92, 50.00] |
| Fixed | 56.30 [52.25, 60.49] | 49.84 [49.29, 50.27] | 49.98 [49.92, 50.00] |
| Fixed+Guard | 50.10 [49.36, 50.85] | 49.99 [49.78, 50.20] | 49.99 [49.87, 50.12] |
| Cache-C1 | 57.73 [53.79, 61.81] | 53.93 [52.12, 55.96] | 49.91 [49.70, 50.03] |
| Cache-C1+Guard | 55.23 [51.41, 59.06] | 56.51 [52.77, 60.54] | 55.68 [52.09, 59.49] |
| C1-ReTrack | 57.49 [53.59, 61.55] | 53.87 [52.02, 55.89] | 49.91 [49.69, 50.02] |
| C1-ReTrack+Guard | 55.09 [51.27, 59.07] | 56.47 [52.75, 60.33] | 55.70 [52.16, 59.45] |

**E02 / E03 / E05 arms, test_confirmation split** (AUC %, [95% CI]):

| arm | beta=0.25 | beta=0.50 | beta=1.00 |
|---|---|---|---|
| Stock-Default-K1 | 49.96 [49.88, 50.00] | 49.96 [49.88, 50.00] | 49.96 [49.88, 50.00] |
| ML0-Default-K1 | 49.96 [49.88, 50.00] | 49.96 [49.88, 50.00] | 49.96 [49.88, 50.00] |
| Full-Rate | 49.96 [49.88, 50.00] | 49.96 [49.88, 50.00] | 49.96 [49.88, 50.00] |
| Stock-Baseline | 55.30 [49.46, 60.63] | 49.38 [48.57, 49.95] | 49.96 [49.88, 50.00] |
| ML0-Default-Fixed | 57.89 [52.81, 63.09] | 49.69 [49.03, 50.19] | 49.96 [49.88, 50.00] |
| Fixed | 58.44 [53.17, 63.52] | 49.63 [48.87, 50.18] | 49.96 [49.88, 50.00] |
| Fixed+Guard | 50.14 [49.21, 51.05] | 49.99 [49.68, 50.29] | 49.99 [49.81, 50.17] |
| Cache-C1 | 59.45 [54.39, 64.24] | 53.89 [51.81, 56.37] | 49.87 [49.55, 50.05] |
| Cache-C1+Guard | 55.44 [50.97, 59.73] | 56.95 [52.52, 61.18] | 56.30 [52.13, 60.50] |
| C1-ReTrack | 59.29 [54.27, 64.08] | 53.72 [51.65, 56.19] | 49.86 [49.54, 50.04] |
| C1-ReTrack+Guard | 55.63 [51.17, 59.98] | 57.15 [52.89, 61.36] | 56.50 [52.38, 60.63] |

**Contrasts (A - B, AUC points, mean over the 3 native cells; * = 95% CI excludes 0):**

| contrast | subset | mean | beta=0.25 | beta=0.50 | beta=1.00 |
|---|---|---|---|---|---|
| E02 K=1 ladder: tracker (Full-Rate - ML0-Default-K1) | all | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] |
| E02 K=1 ladder: tracker (Full-Rate - ML0-Default-K1) | confirmation | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] |
| E02 K=1 ladder: frontend (ML0-Default-K1 - Stock-Default-K1) | all | +0.00 [-0.00, +0.00] | +0.00 [-0.00, +0.00] | +0.00 [-0.00, +0.00] | +0.00 [-0.00, +0.00] |
| E02 K=1 ladder: frontend (ML0-Default-K1 - Stock-Default-K1) | confirmation | +0.00 [-0.00, +0.00] | +0.00 [-0.00, +0.00] | +0.00 [-0.00, +0.00] | +0.00 [-0.00, +0.00] |
| E03 Cache-C1 - Fixed | all | +1.82 [+0.90, +2.79]* | +1.43 [-0.73, +3.60] | +4.09 [+2.36, +6.09]* | -0.06 [-0.30, +0.10] |
| E03 Cache-C1 - Fixed | confirmation | +1.73 [+0.65, +2.86]* | +1.01 [-1.59, +3.42] | +4.26 [+2.25, +6.64]* | -0.09 [-0.45, +0.15] |
| E03 Cache-C1 - Stock-Baseline | all | +2.43 [+0.90, +4.05]* | +2.93 [-1.12, +6.97] | +4.42 [+2.65, +6.36]* | -0.06 [-0.30, +0.10] |
| E03 Cache-C1 - Stock-Baseline | confirmation | +2.85 [+1.07, +4.70]* | +4.14 [-0.34, +8.52] | +4.51 [+2.49, +6.92]* | -0.09 [-0.45, +0.15] |
| E05 Cache-C1+Guard - Cache-C1 | all | +1.95 [-1.23, +5.18] | -2.50 [-6.59, +1.53] | +2.58 [-1.19, +6.21] | +5.76 [+2.14, +9.63]* |
| E05 Cache-C1+Guard - Cache-C1 | confirmation | +1.83 [-1.72, +5.43] | -4.01 [-8.73, +0.68] | +3.07 [-1.38, +7.44] | +6.43 [+2.27, +10.62]* |
| E05 Cache-C1+Guard - Fixed | all | +3.76 [+0.54, +7.08]* | -1.07 [-5.16, +2.95] | +6.67 [+2.94, +10.69]* | +5.70 [+2.09, +9.51]* |
| E05 Cache-C1+Guard - Fixed | confirmation | +3.56 [-0.28, +7.23] | -3.00 [-7.93, +1.64] | +7.33 [+2.86, +11.78]* | +6.34 [+2.13, +10.53]* |
| E05 Fixed+Guard - Fixed | all | -2.01 [-3.42, -0.63]* | -6.20 [-10.35, -2.21]* | +0.15 [-0.35, +0.74] | +0.02 [-0.10, +0.15] |
| E05 Fixed+Guard - Fixed | confirmation | -2.64 [-4.32, -0.89]* | -8.30 [-13.32, -3.26]* | +0.36 [-0.29, +1.18] | +0.03 [-0.15, +0.21] |
| E05 C1-ReTrack - Fixed | all | +1.72 [+0.78, +2.67]* | +1.19 [-0.95, +3.34] | +4.03 [+2.27, +5.98]* | -0.07 [-0.30, +0.10] |
| E05 C1-ReTrack - Fixed | confirmation | +1.61 [+0.55, +2.72]* | +0.85 [-1.69, +3.31] | +4.09 [+2.11, +6.45]* | -0.10 [-0.46, +0.14] |

**E04 arms (Pi pools + Pi-measured MV costs, no INT8 cache for UB), all videos:**

| arm | beta=0.25 | beta=0.50 | beta=1.00 | CPU ms/video-s (beta=0.5) |
|---|---|---|---|---|
| C1-MV | 57.62 [53.61, 61.75] | 49.84 [49.59, 50.03] | 49.99 [49.94, 50.03] | 3611 |
| C1-MV@W | 57.54 [53.54, 61.67] | 50.55 [47.52, 53.65] | 49.99 [49.94, 50.03] | 3136 |
| C1-MV|cost1.25 | 57.61 [53.61, 61.73] | 50.01 [49.81, 50.29] | 49.99 [49.94, 50.03] | 3610 |
| C1-Pixel | 57.73 [53.79, 61.81] | 53.93 [52.12, 55.96] | 49.91 [49.70, 50.03] | 3562 |

**E04 contrasts:**

| contrast | subset | mean | beta=0.25 | beta=0.50 | beta=1.00 |
|---|---|---|---|---|---|
| E04 +MV: C1-MV - C1-Pixel (matched CPU) | all | -1.37 [-2.32, -0.44]* | -0.11 [-2.06, +1.73] | -4.09 [-6.14, -2.26]* | +0.08 [-0.00, +0.27] |
| E04 +MV: C1-MV - C1-Pixel (matched CPU) | confirmation | -1.42 [-2.56, -0.37]* | -0.23 [-2.42, +1.89] | -4.16 [-6.62, -2.08]* | +0.12 [-0.00, +0.39] |
| E04 +MV: C1-MV@W - C1-Pixel (same schedule) | all | -1.17 [-2.24, -0.17]* | -0.19 [-0.75, +0.39] | -3.38 [-6.40, -0.57]* | +0.08 [-0.00, +0.27] |
| E04 +MV: C1-MV@W - C1-Pixel (same schedule) | confirmation | -0.83 [-2.14, +0.42] | -0.14 [-0.85, +0.55] | -2.48 [-6.20, +1.18] | +0.12 [-0.00, +0.39] |

