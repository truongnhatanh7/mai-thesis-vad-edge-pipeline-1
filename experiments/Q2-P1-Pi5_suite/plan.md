# Q2-P1-Pi5-Suite: physical Raspberry Pi 5 validation of E02, E04, E05

**Hardware.** Raspberry Pi 5 (4 x Cortex-A76 @ 2.4 GHz, 8 GB, active cooler), `nanh@raspberrypi.local`, `~/edgelab/.venv` (uv; torch, ultralytics, ncnn, onnxruntime, mlflow, psutil, av).
Only one process runs on the Pi at any time; every cell is its own process; the SoC is cooled to <= 58.0 C before every cell and every live arm.

**Method.** The locked VM workers of E02 / E04 / E05 are re-used unchanged (cell file formats and ids identical), so the E02 / E03 / E04 / E05 simulators read the physical
cells through a symlink view and no VM-to-Pi multiplier is applied (native cell s = 1.0). A parent process samples `vcgencmd pmic_read_adc` (PMIC rail power), SoC temperature,
`get_throttled`, cpufreq and RSS every 0.25 s and slices the samples to the timed window of the child (first / last timed work).

**Part 1 (E02).** Decode (JPEG `imdecode`; H.264 `grab`+`retrieve` on SHT H.264 re-encodes, UBnormal 1280x720 `.mp4`, plus 2 user SHT clips and 2 user UB clips), T in {1,2,3,4};
pose detector scaling (ML0 @640, MLP @640, stock @640 / @800 / @320 NCNN FP16; 480 natural L0 frames), T in {1,2,3,4}; BoT-SORT Default vs Locked two-stage (T=1) over person counts;
STG-NF window assembly + per-window normalisation + PyTorch forward, N in {1,3,5,12,24}, T in {1,2,3,4}; end-to-end service cells.
**Part 2 (E04).** ML0 NCNN FP16 vs ORT INT8 vs ORT FP32, T in {1,2,3,4}, 2 replicates (Cortex-A76: `asimddp`, no `i8mm`); H.264 motion-vector export / MV grid / MV C1 cost vs pixel-diff C1 cost (`mvprof`).
**Part 3 (E05).** 2,880-frame live end-to-end streams at 24.0 fps real-time arrival (dt = 41.67 ms), four arms: Fixed-K1; Cache-C1; Cache-C1+Guard (on > 400 ms, off < 150 ms); Cache-C1+Guard-Burst32
(intact 24-frame burst every W = 32). MLflow metrics every 60 frames, `@mlflow.trace` spans on frames 0 / 720 / 1440 / 2160 / 2879.

**Pi-calibrated simulations (Mac).** Full datasets (107 SHT clips, 211 UBnormal videos) with the measured Pi pools: E05 ladder (K=1 ladder = E02, Fixed / Cache-C1 / Guard / Stock = E03 / E05),
E04 arms (+MV, +INT8, +MV+INT8) at the native cells s=1.0, beta in {0.25, 0.5, 1.0}; exact pooled AUC, 2000-resample video-level paired bootstrap.

**Outputs.** `config.yaml`, `results.csv`, `stage_timings_pi5.csv`, `summary.json`, `report.md`; raw traces in `~/thesis-data/pi5_suite/`.
