# Q2-P1-E02 — Fixed baseline (plan, written before any E02 number was computed)

**Lane:** `Q2-P1-*`. **Source plan:** `Edge Anomaly Detection Experiment Plan.md`, ladder row **E02 — Fixed baseline**.
**Builds on:** `Q2-P1-E00_audit` (causal S3/S4 protocol, sentinels `max` -0.5122 SHT / `p50` -0.8273 UB, splits) and
`Q2-P1-E01_quality` (locked frontend `ML0_ncnn`@640, locked tracker `c0.20|low-0.05|b30|m0.85`, default tracker
`c0.25|low-off|b30|m0.80`, runner-up `stock_ncnn800`).
**Training:** none. All weights are frozen (hashed in `config.yaml`).
**What was visible when this plan was written:** the E00 and E01 reports (they are the inputs). Earlier VM traces
(EXP-004/012/020: 40 frames of L3-darkened clips, detector found ~1.1 persons/frame instead of 2.3-3.5) were *read*, not reused.

## 1. Question

> **What does an optimized fixed pipeline achieve on the actual Pi?** Profile decode, pose, tracking and scoring; use the
> intended runtime (NCNN FP16). Required output: *stage timings, memory, delivered AUC and sustainable workload.* No training.

### Hardware disclosure (binding for every sentence in the report)

* The physical Raspberry Pi 5 8 GB (owner to-do, `DECISIONS.md` D8/D10) is **not connected**. **No number in E02 is a Pi
  measurement.** Every timing is tagged `machine: edge-os-arm64-vm (comparative proxy; physical Pi 5 pending D8/D10)`.
* Proxy: Lima VM `edge-os` (Apple vz, 4 vCPU ARM64 Ubuntu 24.04, 8 GiB) on an M2 Pro host. Its cores are ~2-2.5x faster
  single-core than a Cortex-A76 @ 2.4 GHz, with no DVFS/thermal limit. Pi sensitivity = multiplier `s in {1.0, 1.5, 2.0, 2.5, 3.0}`
  on **every** service time (s=1 is the VM itself; the A76-equivalent point is expected near s=2-2.5 but is an assumption).
* The pipeline is the Python research pipeline (OpenCV + NCNN + Ultralytics BoT-SORT + PyTorch STG-NF), not a tuned C++ build.

## 2. What is new relative to EXP-004/012/020 and E00/E01

1. **Fresh VM profile** on natural **L0** ShanghaiTech frames: 12 clips x 240 frames = 2,880 frames (not 40 darkened frames),
   per stage, in isolated processes (peak RSS per process), threads in {1, 2, 3, 4}, CPU-affinity pinned to cores 0..T-1,
   host load outside the VM sampled during every cell (the EXP-004 rule: mean > 30 % -> excluded and re-run).
2. **Decode measured in the VM**: JPEG (`imdecode`), MPEG-1 `.avi` (ShanghaiTech training videos, 3 clips), H.264 `.mp4` (UBnormal
   test videos, 3 clips, 1080x720 @ 30 fps) and, for like-for-like content, the 12 SHT clips re-encoded by us to H.264 (x264,
   labelled as such: the SHT *test* set only exists as JPEG). Per frame `grab` (decode only) vs `retrieve` (colour conversion).
3. **True time-causality** (fixes E00 defect #6): `score[t]` may only be emitted for frames `t' >= ceil(t_done / dt)`; a job
   starts at `max(t_arrival, worker_free)`; the queue, collection and inference are charged to the delivered timestamp.
4. **Two fixed-schedule families on all 107 SHT + 211 UB clips** (`dev`, `confirmation`, `all`), see section 4.

## 3. Controls and single changes

| factor | levels |
|---|---|
| frontend (cache replay) | `ML0_ncnn` (SHT; UB uses the `ML0_mps` proxy: no UB NCNN cache exists, E01 showed NCNN == MPS on SHT), `stock` @640 (MPS proxy of stock-NCNN@640), `stock_ncnn800` (SHT only) |
| tracker | **locked** `c0.20|low-0.05|b30|m0.85` and **default** `c0.25|low-off|b30|m0.80` (both always reported) |
| scoring protocol | E00 S3/S4, completion landing, causal half-Gaussian, train sentinel (SHT `max`, UB `p50`; stock arms: E00/E01 sentinel map) |
| schedule | stride (A) or burst (B), see section 4 |
| service-time scenario | thread count T, Pi multiplier s, CPU budget b, see section 5 |

Only one thing changes at a time: the schedule changes the *observations that exist*, the scenario changes *when they are
delivered*. Detector caches are replayed, so policies could in principle see unrequested detections: the code only ever
reads detections on frames the schedule requests (asserted by test `test_schedule_never_reads_unrequested_frames`), and
charges the measured cost of every requested frame.

## 4. The two fixed-schedule families (the E03 baselines)

`L` = `seg_len` (SHT 24, UB 16); `dt` = 1/fps (SHT 24 fps, UB 30 fps, E00 R7: SHT's 24 fps is an approximation).

**A. Fixed-stride K in {1, 2, 3, 4}.** The detector runs on frames `f % K == 0` (duty 1/K); the tracker sees only those frames.
STG-NF expects `L` frames at native cadence, so skipped frames must be handled:

| fill | meaning | status w.r.t. the experiment plan |
|---|---|---|
| `none` | no pose on skipped frames; STG-NF's `missing <= 2` gate decides | plan-compliant; **predicted: 0 windows for K >= 2** (a window of L consecutive poses spans K*L frames) - verified (D5) |
| `zoh` | causal zero-order hold: the track's last observed pose repeated for up to K-1 frames after its observation | **relabels an old pose as a new observation** - violates "never relabel an old pose as a new observation"; run only as the *simplest fixed-cadence comparator*, flagged `propagated` everywhere |
| `cv` | causal constant-velocity extrapolation from the last two observed poses (<= K-1 frames) | same violation (extrapolated pose, not an observation); no learned filling |

Scores are produced only for windows completing on an *observed* frame (cadence 24/K Hz); the score is held for `K-1` frames
(hold = refresh interval - 1, fixed a priori, not tuned; `H=0` is reported as sensitivity).

**B. Fixed-window-burst with period `W`** (window length `L`, burst = `L` consecutive observed frames):

* `W <= L`: windows `[kW, kW + L - 1]` overlap by `L - W` frames; the *union* of the bursts is every frame, so the detector runs on
  every frame (duty 1) - a per-frame pose store makes the overlap free (E00: bit-exact reuse) - and only **scoring** is thinned
  (one score per track per `W` frames). This family therefore saves scoring cost only.
* `W > L`: the detector runs on `L` of every `W` frames (duty `L/W`); frames `[kW + L, (k+1)W - 1]` are unobserved. The tracker is
  re-created at every burst start (no pose is carried across the gap), so every scored window is 100 % genuine 24-fps
  observations of one burst. A person must be tracked through all `L` frames of a burst (the scorer's `missing <= 2` rule applies,
  but a track of fewer than `L` poses yields no window).
* Grid: SHT `W in {1, 2, 4, 6, 8, 12, 24, 36, 48, 72, 96}`, UB `{1, 2, 4, 6, 8, 12, 16, 24, 32, 48, 64}`
  (`W = K*L` has the same detector duty `1/K` as stride K; `W = 1` is the K=1 pipeline).
* Score hold `W - 1` frames (same rule as A). Windows are scored when they complete on the burst grid
  (`last_frame = kW + L - 1`).

No B-frame interpolation, no learned pose filling, no compressed-domain tricks (E04).

## 5. Service-time model and delivery simulation

**Per-frame job** (single worker, FIFO, strictly in frame order): decode (video: *every* frame must be decoded, `grab` for skipped
frames, `grab + retrieve` for observed ones; JPEG: observed frames only) + for observed frames: letterbox + NCNN + decode/NMS
(+ tracker update with `n_in` boxes) + for windows completing on this frame: assemble + per-window normalise + STG-NF forward.
Service times are *empirical pools* measured in the VM (per stage, per thread count; tracker by `n_in`, scoring by number of
windows `N`), sampled i.i.d. with a fixed seed. The pools of one process are validated against the measured end-to-end
frame time (rule P3).

**Scenarios (headline, one stream):**

| id | T | s | b (budget of the 4-core device) | effective cores `q = 4 b` |
|---|---|---|---|---|
| `T4_s{1.0,1.5,2.0,2.5,3.0}_b1.00` | 4 | 5 values | 1.00 | 4 |
| `T4_s2.0_b{0.75,0.50,0.33,0.25}` | 4 | 2.0 | 0.75 / 0.50 / 0.33 / 0.25 | 3 / 2 / 1.33 / 1 |

Service at `q` cores: stage pools at `T = floor(q)` rescaled so that their mean equals the linear interpolation of the measured
thread-scaling curve at `q` (`q < 1`: pool(1) / q). Pi multiplier: `service *= s`. `ideal` = zero service time (E00 landing).

**Delivery.** Frame `t` arrives at `t * dt`. Job `t` starts at `max(arrival_t, finish_{t-1})` and ends at `finish_t`. A score
produced by job `t` is emitted at `finish_t` and lands on frame `d = ceil(finish_t / dt)` (>= t). A score whose
capture-to-emission latency exceeds the **2.0 s rolling-buffer cap** is discarded (late work). Frames then follow E00's rule:
least-normal window landing on a frame, hold for `H` frames, otherwise the E00 train sentinel; causal half-Gaussian smoothing.
Two stream models: **continuous** (all clips of a dataset concatenated into one stream; the backlog carries over - this exposes
divergence) and **clip-reset** (queue empty at each clip start). Headline numbers use *continuous*.

**Sustainable** (per schedule x scenario x dataset): utilisation `rho = sum(service) / sum(dt) < 1` **and** max queue wait
<= 2.0 s **and** p99 emission latency <= 2.0 s on the continuous stream. Unsustainable configurations are reported but flagged
**rejected** (plan: "reject configurations whose processing creates an unbounded backlog"). *Frame-dropping load shedding is
E03's controller and is not simulated here*; the fixed baselines either keep up or are rejected.

**Workload.** Max streams per device = largest `n` such that `n` copies at one core each (`T=1` pools, `q = 4/n` for `n > 4`) are
sustainable; also reported as `cpu_ms_per_video_s` (measured process CPU time per second of video, including OpenMP spinning).

## 6. Pre-registered rules (evaluated by code; `summary.json -> rules`)

### Profiling
| rule | statement | pass criterion |
|---|---|---|
| **P1** | every VM cell ran on a quiet host | mean host CPU outside the VM <= 30 % within 4 tries; otherwise the cell is listed as excluded |
| **P2** | repeatability | ML0 NCNN pose mean per-frame ms of two repetitions differ <= 5 % at every T |
| **P3** | service model validity | pool-model mean per-frame time within +-10 % of the measured end-to-end mean at every T (SHT, H.264 input) |
| **P4** | thread scaling is monotone | NCNN ML0 pose mean ms non-increasing in T (3 % tolerance); speed-up and efficiency are reported |
| **P5** | frame coverage | every cell processed >= 95 % of its requested frames |

### Simulation correctness
| rule | statement | pass criterion |
|---|---|---|
| **D1** | parity with E01 | ideal-timing K=1 S4 AUC (all clips) of `ML0_ncnn` with locked and default tracker (SHT) and `ML0_mps` (UB) equals E01 within 0.02 pt |
| **D2** | zero-service identity | with service = 0 the delivered series equals the ideal (E00) series exactly |
| **D3** | queue invariants | vectorised FIFO == reference loop; `start >= arrival`; `finish` non-decreasing; every score lands at or after `ceil(finish/dt)` and at or after its completion frame |
| **D4** | prefix invariance | truncating a clip at random cut points never changes delivered scores of earlier frames |
| **D5** | no-fill stride breaks the scorer | stride K >= 2, fill `none`: zero windows on both datasets |
| **D6** | genuine observations | burst windows (`W > L`) 100 % genuine; stride-ZOH windows genuine fraction reported (~ 1/K) |
| **D7** | no unrequested detections | observed-frame mask == frames the schedule requests (unit test) |

### Hypotheses (decided on SHT dev; confirmation and all reported alongside)
| rule | statement | decision |
|---|---|---|
| **H1** | at equal detector duty (1/2, 1/3, 1/4) a plan-compliant burst (`W = K*L`) is at least as accurate as the best propagated stride | paired video-bootstrap CI of (burst - best stride) on SHT/all, ideal timing: `burst preferred` if CI lower > 0, `stride preferred` if CI upper < 0, else `tie`; **no direction is pre-registered** |
| **H2** | the E01 locked tracker still beats the default under every schedule | paired CI per schedule (informational) |
| **H3** | `W <= L` saves scoring cost only | detector duty = 1 and CPU/s lower than K=1 (informational, from the service model) |

### Hand-off to E03 (selection on SHT **dev** only; ties within 0.10 pt resolved toward lower CPU ms/s)
**B1.** For each headline scenario (`T4_s2.0_b1.00`, `T4_s3.0_b1.00`, `T4_s2.0_b0.50`) the *fixed reference* is the sustainable
schedule with the highest SHT-dev delivered AUC (ML0_ncnn, locked tracker). A plan-compliant reference (no propagation) is
also reported. Confirmation and UB are reported, never used for the choice. The sustainable workload and the fixed
accuracy-vs-CPU frontier (Pareto set over `cpu_ms_per_video_s`) are the matched-cost curves E03 must beat.

## 7. Outputs

`plan.md` (this file), `config.yaml` (written by `lock` before any run), `results.csv` (per video x schedule x scenario: delivered
AUC, ideal AUC, offline S0 reference, delay, backlog, CPU ms/s, memory), `decisions.jsonl` (rule results, selections, queue
traces of sample clips: requested / started / finished / delivered per job), `summary.json`, `report.md`.
Raw VM traces: `~/thesis-data/vm-exchange/q2-p1-e02/out/` (npz + json per cell, git-ignored); the report quotes the summaries.

## 8. Reproducibility

`uv run --extra pose python -m edgelab.analysis.q2_p1_e02 {lock, vm, run, report, all}`; tests `tests/test_q2_p1_e02.py`.
`vm` drives `limactl shell edge-os` and needs the host idle (it samples host load); `run` is host-only and deterministic
(seed 20261113, service-pool sampling seed 7, bootstrap seed 0, 1000 resamples).
