# Experiments — context baseline (DRAFT v1, 2026-10-31)

> Status: **draft**. Decisions D1–D7 are recorded in [`DECISIONS.md`](DECISIONS.md). That file wins if the two disagree.

## 1. Research question (from the lit review)

*How do lightweight vision preprocessing and OS-level optimization, individually and jointly, affect the robustness
and real-time performance of CPU-only public-safety inference on edge devices?*

Operationally, this means a 2×2 factorial (plus levels) on one frozen public-safety model:

| | No OS tuning (O0) | OS tuning (O1+) |
|---|---|---|
| **No preprocessing (P0)** | baseline | system-only |
| **Illumination preprocessing (P1+)** | vision-only | combined |

"Jointly" = the **P × O interaction**: does the effect of preprocessing depend on the OS configuration?
Important mechanism: OS tuning does **not** change a frozen model's outputs. It can only change accuracy through
**deadlines**: in a live stream, frames that finish late are dropped. So the joint claim must be measured as
*delivered* accuracy under a fixed frame rate with drop-on-late, not offline accuracy.

## 2. Anchor paper and why we cannot replicate it as-is

Salman et al. 2026, Sci. Rep., doi:10.1038/s41598-026-44939-x (`kb/papers/salman2026-embedded-violence-detection.md`).
Audit findings (2026-10-31, verified against the article):

- **No code, no weights, no training scripts.** Custom six-class dataset only "upon reasonable request".
- **Architecture under-specified:** Table 3 gives the layers (Conv32 → DWSep64 → Conv128 → GAP → GRU128 → 6-way softmax,
  256×256 input). It does not give the clip length T, the strides, the pruning ratio or the INT8 calibration set. Our
  estimate of ~0.1 GMAC/frame should run in a few ms on a Cortex-A76, which does not fit the reported 67 ms FP32.
- **Evaluation does not follow benchmark protocols:**
  - It reports accuracy instead of frame-level AUC on UCF-Crime and ShanghaiTech.
  - The ShanghaiTech link points to the *crowd-counting* dataset, not ShanghaiTech Campus.
  - It claims "official splits" while training supervised on a split that has no anomalies in training.
- **Numerical red flags:**
  - Every Table 9 confusion matrix sums to exactly 1000, even Violent Flows (246 videos), which suggests frame-level
    leakage.
  - Several metrics cannot be recomputed from their own confusion matrices.
  - The "average 96.9 %" is the custom-dataset row.
  - The t-tests are impossible with n = 3.
  - INT8 accuracy is higher than FP32.

**Consequence:** "replicate the paper" becomes **Phase 0, a bounded reproduction study**: rebuild the Table 3 model, train
it on public data with leakage-free video-level splits, and test whether ~96 % is reachable. A failed reproduction is a
citable finding in its own right. The frozen model for the main experiments is chosen in Phase 0 (D1; open-model choice made in Phase 0).

## 3. Benchmarks and the low-light stratum

No major violence or anomaly benchmark labels lighting or viewpoint, so we must build the stratum ourselves.

| Dataset | Use | Facts (verified 2026-10-31) |
|---|---|---|
| UCF-Crime | main benchmark | full archive ≈ 103 GB, official page alive. Test set: 290 videos (150 normal / 140 anomalous), frame-level temporal annotations, ≈ 1.11 M frames (≈ 37 k at 1 fps). Has real day **and night** footage, unlabelled. Violence-like test videos (Abuse, Assault, Fighting, Robbery, Shooting): **38**. |
| ShanghaiTech Campus | daylight control, synthetic darkening | 437 videos, 13 scenes. Weakly-supervised split (Zhong 2019): 238 train / 199 test. Probably mostly daylight (unverified). |
| XD-Violence | secondary, violence-specific | test set 800 videos, 10.9 GB; mostly movies/YouTube. |
| RLVS, Hockey Fight | Phase 0 reproduction | public, small; Hockey has no low light. RWF-2000 videos currently not downloadable. |

**Low-light stratum (proposal):**
1. Per-video mean luma, with a pre-registered threshold, defines *dark* / *normal* videos.
2. Owner hand-labels the 290 UCF-Crime test videos as day / night / IR-greyscale (~2–3 h). We report how well the
   luma rule agrees with the hand labels.
3. Controlled experiment: synthetic sensor-style darkening (`x·α` in linear space + Poisson-Gaussian noise, 3–4
   severities) applied to well-lit videos. A dose-response curve, explicitly not the same as real night.
4. Stratify **per video**, because frame-AUC inside a stratum needs both classes. Use bootstrap CIs over videos. The
   expected dark ∩ violent group is small (~10 videos), which is a power risk.

Camera angle: no labels anywhere. Dropped (D5).

## 4. Hardware and environment

| Machine | Role | Notes |
|---|---|---|
| MacBook Pro 16" M4 Pro (10P + 4E cores, 48 GB) | development, Phase 0 training (PyTorch MPS), all offline accuracy runs | target machine (D3; AGENTS.md updated) |
| Lima VM `limavm` (vz, Ubuntu 25.04 arm64, 4 vCPU, 8 GB) | Linux harness, eBPF/perf CPU inspection, relative OS-knob experiments | no guest cpufreq/DVFS, no thermals. vCPUs float across host P/E cores, and host jitter leaks in. **Absolute latency and OS claims are weak here.** |
| Raspberry Pi 5 (owner will buy) | credible edge timing + energy | real DVFS, thermals, USB-C power meter; ≈ $273 BOM (see idea file) |

OS knobs that actually exist for a single DNN on a few CPU cores:
- intra-op thread count, thread spinning on/off
- CPU affinity/pinning, SCHED_FIFO/nice priority
- cgroup `cpuset`/`cpu.max`
- governor and DVFS (Pi only)
- sched_ext (kernel ≥ 6.12, optional)

The STUN-style CFS sysctls are gone since Linux 5.13/6.6.

CPU inspection while benchmarking:
- per-stage timestamps
- `pidstat`/psutil sampling
- `perf stat`
- eBPF (`runqlat`, `offcputime`, `cpudist` via bcc/bpftrace) inside the VM or on the Pi

## 5. Phased plan (proposal)

| Phase | Goal | Exit gate |
|---|---|---|
| 0. Reproduce | Rebuild the Salman Table 3 CNN-GRU; train on RLVS/Hockey with video-level splits; also run 1 open pretrained violence model | Documented reproduction verdict; one **frozen anchor model** chosen with credible daylight performance |
| 1. Harness | Video → preprocess → infer → score pipeline; per-stage timing; CPU telemetry; run manifests | Same detections on Mac and VM for 100 clips; full logs for a 10-min run |
| 2. Baseline characterisation | Anchor model on UCF-Crime test (+ShanghaiTech) by lighting stratum; CPU-load profile | Per-stratum AUC with CIs; error analysis of dark videos; CPU profile showing where time goes |
| 3. Preprocessing arm (P) | gamma, CLAHE, Zero-DCE++ (+ luma gate) | Effect on dark vs well-lit strata, cost per frame |
| 4. OS arm (O) | thread/spin/pinning/priority configurations | Latency p50/p95/p99, throughput, CPU util, (energy on the Pi) |
| 5. Joint (P × O) | factorial under fixed-fps streaming with drop-on-late | Interaction contrast with CI; Pareto plots |

### Decisions (see DECISIONS.md)
- Anchor: Salman-style reconstruction + one open pretrained violence model (D1).
- Main metric: UCF-Crime frame-AUC by lighting stratum. Violence accuracy/F1 is secondary (D2).
- Hardware: M4 Pro → Lima VM subset → Pi 5 (D3). Camera angle dropped (D5).
- Remaining open: which open model (MoBiLSTM vs X3D-ONNX, decided in Phase 0); the luma threshold; Phase 0 gates.

## 6. AI-human workflow (D6, D7)

Goal: the owner can trust and explain every number without reading AI code line by line blind.

- **Pre-registration per experiment:** `experiments/EXP-NNN-<slug>/plan.md` states the question, hypothesis, metric,
  pass/fail rule, configs and expected runtime. The owner approves it **before** any run. The AI may not change the
  metric or threshold after seeing results.
- **Decision log:** `experiments/DECISIONS.md`, dated, one line per decision with its reason.
- **Small reviewable slices:** each code change does one thing, is ≤ ~200 lines and comes with a plain-language
  walkthrough (data in → transform → data out) and a diagram for the end-to-end flow.
- **Trust anchors (tests the owner can read):**
  - metric code is checked against sklearn on toy data
  - the harness reproduces a published number before we trust it (e.g. a released VAD checkpoint's UCF-Crime AUC)
  - invariant checks: frame counts, label alignment, no train/test video overlap
- **Run manifests:** every run writes `manifest.json` (git sha, config hash, model hash, dataset hash, machine, kernel,
  thread settings, start/end time). Results are append-only (`results/*.jsonl`) and never hand-edited. Tables and
  plots are generated from them by script.
- **Results review:** `results.md` per experiment = numbers + plots + the AI's interpretation + the owner's sign-off.
  KB note links back.

## 7. Candidate-experiment backlog (borrowed from the KB, D4)

These are candidates, not commitments. Each one becomes an `EXP-NNN` only after its plan is approved.

| # | Candidate | Source in kb/ | Phase |
|---|---|---|---|
| B1 | Preprocessing levels: none / gamma-LUT / CLAHE-on-luma / Zero-DCE++ (ONNX) | idea-deadline-aware-preprocessing-cpu-allocation-edge | 3 |
| B2 | CPU-allocation configs: O0 default (ORT intra-op = all cores, spinning on); O1 spinning off + OpenCV 1 thread; O2 partitioned (inference pinned to 3 cores + SCHED_FIFO, preprocessing/IRQs on core 0) | same | 4 |
| B3 | Sequential vs **pipelined** stages: does concurrent preprocessing inflate inference p99 (super-additive tail)? | hyp-pipelined-contention-superadditive-p99 | 4–5 |
| B4 | **Drop-on-late streaming** at fixed fps; delivered accuracy = the joint metric; sign flip of enhancement's effect across O configs | hyp-enhancement-sign-flip-cpu-allocation | 5 |
| B5 | **Luma + slack gate**: enhance only dark frames *and* only when there is time left before the deadline; compare against never / always / luma-only / oracle | hyp-luma-slack-gate-nondominated | 5 |
| B6 | Does CLAHE help a *frozen* model at night at all? (published frozen-model evidence says often not) | hyp-clahe-frozen-night-recall | 3 |
| B7 | Background co-load (ffmpeg H.264 re-encode as "recording/upload") as a factor | idea-deadline-aware… | 4–5 |
| B8 | Second workload: frozen YOLO11n person detection on LLVIP (CCTV-view low-light), if violence models turn out too weak in the dark | idea-deadline-aware… | optional |
| B9 | DVFS / governor + frame-skip under a latency SLO (Pi only); energy per frame | idea-slo-aware-dvfs-video-pipeline | 4 (Pi) |
| B10 | sched_ext policies (scx_lavd / layered) vs EEVDF / SCHED_DEADLINE with co-located load (Pi) | idea-schedext-multi-dnn-slo-pi | stretch |
| B11 | eBPF telemetry (runqlat, offcputime, cpudist) as the CPU-load inspection layer, and as features for any adaptive policy | idea-ebpf-tuned-edge-inference | 2 onward |
| B12 | CLIP-prompt training-free VAD on UCF-Crime as an alternative frozen anomaly scorer (with AUC baseline from the literature) | idea-energy-vlm-anomaly-scoring-interfaces | optional |

Method rules borrowed as defaults:
- thermal gating
- randomised run order
- per-run (not per-frame) statistics
- paired bootstrap CIs
- Holm correction across pre-registered tests
- equivalence margins for "no effect" claims

## 8. Tech stack (proposal)

| Layer | Choice | Why |
|---|---|---|
| Language | **Python 3.12**, managed by **uv** (lockfile) | The ML ecosystem lives here. Owner knows basic Python. |
| Training (Phase 0 only) | PyTorch + MPS on the M4 Pro | tiny model; no CUDA needed |
| Inference (everywhere) | **ONNX Runtime** CPU EP | same model file on Mac / VM / Pi; exposes thread and spinning knobs |
| Video I/O | **ffmpeg** CLI for offline decode/resampling; OpenCV/PyAV in the live pipeline | no custom C/C++ unless profiling proves Python is the bottleneck (ORT and OpenCV are already C++ inside) |
| Preprocessing | OpenCV (gamma LUT, CLAHE), Zero-DCE++ exported to ONNX | |
| Metrics / stats | numpy, scikit-learn (AUC), own bootstrap (tested against scipy) | |
| Tracking | **MLflow** (local SQLite) + `plan.md` / `results.md` | D7 |
| OS knobs | `os.sched_setaffinity`, `os.sched_setscheduler`, ORT session options, cgroup v2 files, `cpupower` (Pi) | all from Python; no kernel code |
| Telemetry | psutil sampling, `perf stat`, bpftrace/bcc tools | |
| Glue | `justfile` or Makefile with few, short recipes; minimal bash | owner is a bash novice |
| Optional Go | experiment orchestrator (ssh to VM/Pi, collect manifests) or a low-overhead telemetry sampler | owner's expert language; only if Python glue gets messy |
| Tests | pytest | |

## 9. Folder layout (D9)

```
experiments/
  README.md  DECISIONS.md          context + decision log (this file wins over chat)
  pyproject.toml  uv.lock          one Python env, same lockfile on Mac / VM / Pi
  justfile                         short named commands (setup, test, train, run, mlflow-ui)
  src/edgelab/                     the importable package — all reusable code lives here
    data/        dataset indexes, splits, decode-to-cache (ffmpeg)
    models/      model definitions (salman_cnn_gru.py), ONNX export
    preprocess/  gamma, CLAHE, Zero-DCE++ wrappers, luma gate
    pipeline/    offline evaluator + live stream runner (fixed fps, drop-on-late)
    osknobs/     ORT thread/spin options, affinity, priority, cgroups
    telemetry/   psutil sampler, perf/bpftrace wrappers
    metrics/     AUC, accuracy/F1, bootstrap CIs
    tracking/    MLflow helpers + run manifest
  tests/                           pytest: metric checks vs sklearn, invariants, golden outputs
  tools/                           bpftrace scripts, VM/Pi setup scripts (kept short)
  EXP-NNN-<slug>/                  one per experiment
    plan.md                        pre-registration (owner approves before any run)
    config.yaml                    exact parameters used
    results.md                     numbers + plots + interpretation + owner sign-off
  data/      → gitignored; symlink to datasets on the Mac (shared into the VM read-only)
  artifacts/ → gitignored; trained weights, ONNX files (logged to MLflow too)
  mlruns/  mlflow.db → gitignored; MLflow store
```

Rule: experiment folders hold **no reusable code**. They hold configs and documents only. Code lives in `src/edgelab`
and gets tests.

## 10. Experiment record format (for human audit)

Every `EXP-NNN-<slug>/` folder contains:

| File | Content | When |
|---|---|---|
| `plan.md` | **Background** (plain language), **Problem**, **Objective**, **Method** (data, model, steps, metrics), **Pre-registered expectations / gates**, **Deviations policy** | before any run; owner approves |
| `log.md` | Dated lab notebook: every command run, every download (URL, size, sha256), every code patch, every deviation from the plan and why, every failure | during the work, append-only |
| `results.md` | **Results** (tables with run ids), **Observations**, **Discussion**, **Threats to validity**, **Next steps**, **Owner sign-off** | after the runs |
| `WALKTHROUGH.md` | High-level end-to-end flow of the code used: data in → each step → numbers out, with file:function pointers and which tests guard each step | with the code |
| `config.yaml` | exact parameters | per run |

Rules:
- Numbers in `results.md` must cite an MLflow run id or a file under `artifacts/`.
- Third-party code goes in `third_party/<repo>` (gitignored), pinned to a commit sha recorded in `log.md`. Our
  changes are kept as a `.patch` file in the experiment folder.
- Never silently change a metric, threshold or split. Log it as a deviation, and add a decision line if it matters.
