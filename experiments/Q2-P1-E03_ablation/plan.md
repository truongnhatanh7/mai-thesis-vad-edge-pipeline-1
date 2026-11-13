# Q2-P1-E03 — Core ablation (plan, written before any E03 arm number was computed)

**Lane:** `Q2-P1-*`. **Source plan:** `Edge Anomaly Detection Experiment Plan.md`, ladder row **E03 — Core ablation**.
**Builds on (read first):** `Q2-P1-E00_audit` (causal S3/S4 protocol, train sentinels, splits `test_dev` / `test_confirmation`,
bit-exact per-frame `(x, y, conf)` reuse), `Q2-P1-E01_quality` (locked frontend `ML0_ncnn`@640 / UB `ML0_mps` proxy, locked tracker
`c0.20|low-0.05|b30|m0.85`), `Q2-P1-E02_fixed_baseline` (VM stage pools, true time-causal FIFO queue + 2.0 s late-work cap,
fixed stride/burst baselines and their accuracy-vs-CPU frontier).
**Training:** none. All weights frozen; every threshold is derived from `test_dev` or from the E00 train-calibration sentinels.
**What was visible when this plan was written:** the three reports above, E02's `summary.json` / stage caches, and the EXP-023
controller code (`refresh_policy.py`, `visual_propagation.py`: *read*, not reused; EXP-023's single-clip results are not evidence here).

## 1. Question

> Compare **fixed, motion-only, adaptive-only, cache-only, independent combination and joint controller**.
> Required output: **matched-cost accuracy/delay curves and failure cases.** Training: no.
> "Cache-only" means **additional cross-frame reuse**, not a deliberately uncached baseline. Drop caching if no useful stage or
> net benefit is found.

### Hardware disclosure (binds every sentence in the report)
Exactly as E02: the physical Pi 5 is **not connected**; every service time is sampled from the E02 VM pools
(`edge-os-arm64-vm`, comparative proxy) scaled by a Pi multiplier `s`; E03 numbers are *simulated delivery on replayed detections*,
not Pi measurements. The scheduler's own observer cost is charged at the EXP-023 figure (0.366 ms/frame at 856x480, scaled by
pixel count for UBnormal) **and** re-measured in `lock` (`config.yaml`); the larger of the two is used.

## 2. Two regimes (E02's structural finding) and what E03 evaluates in each

* **Compliant intact-window regime (primary).** Every pose in every scored window is a genuine detector observation of its own
  frame ("never relabel an old pose as a new observation"). The detector therefore runs in *runs of consecutive frames* (a burst of
  `L` frames gives one intact window per tracked person; a longer run gives one window per extra frame), the tracker is re-created
  at every run start, and a **periodic full-scene check** guarantees that quiet entries are never missed forever. Cheap per-frame
  observers (0.37 ms) are always on for the arms that use them. All arms receive the overlapping-window pose reuse of E00 (each
  frame's `(x, y, conf)` is evaluated once) — nothing is a strawman.
* **Sub-sampled reuse / propagation regime (reference).** E02's `Fixed-Stride` (`stride-K` + ZOH/CV fill, *violates* the rule) is
  carried over from E02's stage files as a fixed comparator (paired CIs vs the compliant arms at the nearest matched cost). The
  adaptive counterpart of a propagated stride that E03 *can* evaluate while staying causal is mechanism **C1** below (reuse of a
  pose only while a pixel-static validity check holds, source timestamp logged).

## 3. Controls, single changes, factorial

Fixed controls for every arm: frontend/tracker/scorer/protocol/hold-rule/queue model of E00-E02; decode model `h264`; `T = 4`
threads (`q = 4` cores); one stream at recorded cadence; window length `L` (SHT 24, UB 16).

| arm | scheduling | cross-frame reuse | budget accounting |
|---|---|---|---|
| **Fixed** | periodic window bursts `F(W)`: `W <= L` thins scoring at detector duty 1, `W > L` runs `L` frames every `W` (E02 family, real-valued `W`) | none | none (open loop) |
| **Motion-Only** | burst on `motion >= theta_m` (4x down-sampled luma-compensated frame difference, EXP-023 statistic), run extended while motion stays high, periodic full-scene check | none | token bucket, expected cost |
| **Adaptive-Only** | event/uncertainty-aware priority `u` (scene motion, border/entry motion, track-quality drop, last delivered window anomaly/uncertainty, observation age) + backlog-aware shedding/thinning | none | token bucket, expected cost |
| **Cache-Only** | `Fixed` schedule + reuse: **C1** (static-validity detector skip), **C2** (cross-window scorer reuse), measured separately (`C1`, `C2`) and together (`C1+C2` = the arm) | C1 + C2, fixed thresholds | none; schedule re-chosen so *post-reuse* cost equals the target |
| **Independent-Combo** | `Adaptive-Only` scheduler + `Cache-Only` (fixed thresholds from the Cache-Only tuning), running side by side | C1 + C2, static | scheduler debits **nominal** cost (does not know about savings) |
| **Joint-Controller** | `Adaptive-Only` scheduler + C1 + C2 coordinated: cache thresholds tighten when anomaly/uncertainty/entry is suspected and relax when the queue backlog grows; scoring/detector savings are **reinvested** into the bucket | C1 + C2, dynamic | scheduler debits **actual** cost (savings re-enter the bucket) |

### Cache mechanisms (cross-frame reuse; records source timestamp, validity, refreshes, numerical error, net CPU)
* **C1 — static-validity detector skip** (ZOH). After a detector frame `s`, frame `t` (`t - s <= A1 = 4`) *reuses* the poses of `s`
  iff the mean 4x-down-sampled, 24x40-pooled |frame difference| inside the person ROIs of `s` (boxes of `s`, dilated one cell) is at most
  `tau_roi[l]` **and** the 95th percentile outside the ROIs is at most `tau_ext[l]` (a person entering the scene breaks the exterior
  check). Otherwise the detector runs (refresh, `s := t`). Frames at chain starts always run the detector. Levels `l = 1, 2, 3` use
  the 10 / 25 / 50 % quantiles of the `test_dev` distribution of the one-frame statistics (computed in `lock`, written to
  `config.yaml`). A reused pose is flagged `reused`, not a fresh observation; windows containing reused poses are **re-scored** (STG-NF
  forward on the substituted window), so numerical error and AUC effect are measured, not assumed. Saved: pose + tracker time of the
  skipped frame; charged: observer + ROI check on every in-run frame. P-CV / P-LK (EXP-023) are not re-run: E02 already showed that
  constant-velocity gives <= 0.9 pt over ZOH and LK needs an extra flow stage that has no VM timing pool.
* **C2 — cross-window scorer reuse.** When a lineage's window shifts by one frame, the cumulative L2 displacement of the
  height-normalised last-frame pose plus velocity change since the last *scored* window (path length, an upper bound of the true
  distance) is below `tau2[l]` (quantiles 15 / 35 / 60 % of the `test_dev` one-step distribution), the source window is at most
  `A2 = 6` frames old and its normality is at least the `test_dev` 25th percentile of normal-frame windows (previous score looks
  normal), then the previous normality is **re-delivered** (source timestamp kept, flagged `reused`) and the `~2.5 ms` STG-NF forward is
  skipped; otherwise the window is scored fresh. Charged: 0.01 ms (x s) per candidate window. Hidden-state reuse is *not* attempted
  (E00 R5: per-window normalisation breaks it).

## 4. Cost model, budgets and *matched-cost* comparison

* **Cost unit:** CPU ms per video-second, per job sampled from the E02 VM pools (`cpu` column, OpenMP spin included) x `s`, **plus
  all observer / ROI-check / cache-check costs**. `C_full(s)` = E02's measured K=1 pipeline cost at the same `s`
  (`summary.json`, `T4_s{s}_b1.00`, SHT 2278 CPU-ms/s at `s = 2`).
* **Cells:** `s = 2.0` with `beta in {0.25, 0.33, 0.50, 0.67, 0.75, 1.00}` and the overload cells `s in {2.5, 3.0}` with
  `beta in {0.50, 0.75, 1.00}` (12 cells). Target cost = `beta * C_full(s)`.
* **Matching:** every arm has one *rate knob* (Fixed/Cache-Only: the period `W`; scheduler arms: the token-bucket refill rate).
  Label-free **bisection** sets the knob so the realised total cost on the evaluated stream equals the target (within 1 %, never above).
  Realised costs are reported; an arm that cannot spend its budget is reported as under-spending. The scheduler's bucket debits the
  job's realised CPU (Independent: the counterfactual no-cache cost).
* Calibration of the knob uses costs only (no labels). Hyper-parameter *selection* uses `test_dev` AUC only (section 6).
* Queue: frame `t` arrives at `t dt`, single FIFO worker, job starts at `max(arrival, finish_prev)`; scheduler decisions are taken at
  job start; feedback (delivered window scores, track quality) is visible only for jobs with `finish <= start`; windows later than
  2.0 s after their last frame are discarded; a delivered score lands on `ceil(finish / dt)`, never before its completion frame.
  Overload protection of Adaptive/Joint: shed (abort run, refuse starts) when the queue wait exceeds 0.40 s until it falls below
  0.15 s; thin scoring to every 2nd frame above 0.20 s. Fixed / Motion-Only / Cache-Only do not shed (simple baselines).
* **Hold rule (a priori, not tuned):** `hold = W_ref - 1` frames, where `W_ref` is the period of the dev-selected Fixed arm of the
  same cell (identical for all arms of a cell; `H = 0` and `H = 2 W_ref` reported as sensitivities). Periodic full-scene check period
  `P_max = 2 max(W_ref, L)` frames for the scheduler arms.

## 5. Exactness of the replay (what is exact, what is declared)

* Detections are replayed from the E01/E02 caches; the policy only reads detector outputs of frames it requested and, for the
  controller, only results whose job has finished (tests `test_*causal*`, `test_*unrequested*`).
* **Run tables (exact).** For every even clip-local start frame `c` the locked BoT-SORT tracker is run on the genuine detections of the
  next `L + Delta - 1` frames (`Delta = L/2`: SHT 12, UB 8), every window of every track is scored once (hashed, identical windows
  scored once). A run is served by *chains*: chain `j` starts at `a + j Delta`; windows ending at `t` come from the chain that
  covers `t`. A one-run table therefore equals a re-created tracker exactly (parity with E02 `burst{W>L}`, rule **E3-D1**); a
  *continuous* run is a tracker re-created every `Delta` frames (an evaluation device: its cost is charged as one persistent tracker;
  the AUC difference to E02's K=1 is reported as **E3-D2**).
* **C1 identity is an oracle**: reused frames keep the genuine track membership of the run (static scenes make association trivial);
  numerical effects are on the pose values only. This favours C1 slightly; if C1 still loses the conclusion stands.
* Fixed-Stride (propagated regime) is E02's own simulation (not re-run).

## 6. Tuning discipline (test_dev only)

Hyper-parameter grids (each arm; selection = mean dev delivered AUC over the three tuning cells `(s=2.0, beta=0.33)`, `(2.0, 0.75)`,
`(3.0, 1.00)` with knob-matched cost, simulated on the continuous `test_dev` stream; ties within 0.10 pt -> the simpler value = earlier in the
grid): Motion-Only `theta_m` in dev quantiles {50, 70, 85 %} x extension {off, on} x periodic-check factor `p in {1, 2}`
(`P_max = p max(W_ref, L)`); Adaptive `tau0 in {0.8, 1.4, 2.0}` x extension {off, on} x `p in {1, 2}` (both schedulers receive the same
tuning effort); Cache C1 level {1, 2, 3} and C2 level {1, 2, 3} chosen separately (each on the Fixed schedule), then combined; Joint
`tau0 in {0.8, 1.4, 2.0}` with extension / `p` inherited from the Adaptive-Only selection and the cache levels from the cache tuning
(Independent-Combo = the same scheduler, extension, `p`, `tau0` as Adaptive-Only + the same cache levels). Everything else (weights, caps, thresholds on
queue wait, `A1`, `A2`, bucket capacity 2.5 bursts, hold rule) is fixed a priori in `config.yaml`. UBnormal is tuned on its own dev split.
The alert threshold (frame-level false-positive rate 5 % of the K=1 ideal series on `test_dev` normal frames) is locked in `lock` and
shared by every arm. The *locked policies* are then evaluated on `test_dev`, **`test_confirmation` (verdicts)** and `all`.

## 7. Metrics

Causal **delivered AUC** (true queue) and **ideal AUC** (same decisions, zero service delay, completion landing); event-level recall
at 0.5 s / 1.0 s, missed-event rate, detection delay p50 / p90 (ms), false alerts per hour of normal video (alert = score above the
locked threshold for >= 6 frames), total CPU ms per video-second (including observer/cache overhead), peak-memory estimate
(E02 end-to-end RSS + queued frames), queue wait mean / p99 / max, cache statistics (hit rate, refreshes, source age, numerical
error, saved vs overhead CPU). Paired video-level bootstrap (1000 resamples, seed 0) per cell and averaged over the 12 cells (same
resamples), on `dev`, `confirmation`, `all`.

## 8. Pre-registered hypotheses and decision rules (verdicts on SHT `test_confirmation`; UB is replication)

**H1 — Scheduling.** At matched total CPU cost `Adaptive-Only` beats `Fixed` and `Motion-Only`.
*Supported* iff the 12-cell-average paired contrasts (Adaptive - Fixed, Adaptive - Motion) both have CI lower bound > 0; *partially
supported* iff exactly one has; *not supported* if neither CI excludes 0; *rejected* if an averaged CI upper bound < 0.

**H2 — Reuse.** Valid reuse removes redundant work without hiding short events, and the saving funds fresher observations.
*Supported* iff (a) the net CPU saving after overhead of the Cache-Only operating point is > 5 % **and** (b) the matched-cost average
contrast (Cache-Only - Fixed) has CI lower bound > 0 **and** (c) event recall @1.0 s is not lower than Fixed's by more than 2 points.
*Partially supported* iff (a) holds and the CI lower bound > -0.25 pt (non-inferior) . *Rejected* iff the averaged CI upper bound < 0
or the net saving is <= 0. Verdicts are also given **per mechanism** (C1, C2); a mechanism with net saving < 2 % or averaged CI upper
bound <= 0 is **dropped** ("drop caching if no useful stage or net benefit").

**H3 — Combined system.** The joint policy improves the accuracy-delay-compute trade-off beyond applying each component
independently. *Supported* iff Joint - {Adaptive-Only, Cache-Only, Independent-Combo} all have averaged CI lower bound > 0 (and
Joint - Fixed > 0); *partially supported* iff Joint beats Independent-Combo and Fixed but not every simpler arm; otherwise *not
supported*.

**Stop rule (plan section 8): "if the joint controller loses after overhead, simplify or reject it."** The Joint-Controller is
retained iff it is not beaten by any simpler arm: for every simpler arm `A`, the averaged confirmation contrast (Joint - A) has CI
upper bound >= 0 **and** Joint has CI lower bound > 0 against at least one of them (`Adaptive-Only`, `Cache-Only`). Otherwise it is
rejected and the lock moves to the simplest arm that is non-inferior to the best arm.

**Lock rule (candidate for E04/E05).** Order of simplicity: Fixed < Cache-Only (surviving mechanisms) < Motion-Only < Adaptive-Only <
Independent-Combo < Joint-Controller. Let `best` be the arm with the highest 12-cell-average confirmation delivered AUC among arms
that pass the budget-validity check (realised cost <= 1.02 x target in >= 10/12 cells). The lock is the simplest arm whose averaged
paired contrast to `best` has CI lower bound > -0.25 pt (non-inferiority margin), **provided** it beats Fixed with CI lower bound
> 0; otherwise the lock is `Fixed` (and the report says that no scheduler or cache is warranted at the proxy). E04/E05 inherit the lock,
the tuned hyper-parameters and the cell grid.

## 9. Pre-registered correctness rules (code, `summary.json -> rules`)

| rule | statement |
|---|---|
| **E3-D1** | F(W > L) at zero service delay reproduces E02's `burst{W}` AUC (SHT, all/dev/confirmation) within 0.02 pt for W in {36, 48, 72, 96} |
| **E3-D2** | continuous run tables vs E02 K=1 (ideal timing, all clips): difference reported, pass if <= 0.5 pt |
| **E3-D3** | scheduler never reads detector output of a frame it did not request; feedback only from jobs finished at decision time |
| **E3-D4** | queue invariants of E02 hold for every simulated job set (FIFO == reference, landing >= completion) on a sample of runs |
| **E3-D5** | zero-delay re-landing of the same decisions equals the "ideal" series; truncating a stream never changes earlier delivered scores |
| **E3-D6** | realised cost of every reported arm/cell is within [0.97, 1.0] x target after calibration, or the cell is flagged under/over-spending |
| **E3-D7** | C1/C2 never reuse across a clip boundary, never reuse older than `A1` / `A2`, never reuse a source with elevated anomaly (tests) |

## 10. Outputs

`plan.md` (this file), `config.yaml` (written by `lock` before any arm is simulated), `results.csv` (per video x arm x cell: delivered /
ideal AUC, events, delay, queue, CPU, memory, cache statistics), `decisions.jsonl` (rules, selections, per-run controller decisions of
sample clips incl. cache source timestamps), `summary.json`, `report.md`. Intermediate caches (motion statistics, run tables):
`~/thesis-data/q2-p1-e03/`.

## 11. Reproducibility

`uv run --extra pose python -m edgelab.analysis.q2_p1_e03 {motion, tables, lock, run, report, all}`; tests `tests/test_q2_p1_e03.py`.
Seeds: experiment 20261113, service-pool sampling 7 (common random numbers, as E02), bootstrap 0 (1000 resamples).

## 12. Amendments and disclosures (all made before `lock` wrote `config.yaml` and before any final `run`)

* `lock` = dev-derived constants **and** the dev-only tuning above; `config.yaml` (constants, selected configurations, `C_full`) is written
  by `lock` and is the only input of `run`.
* **Disclosure of exploratory looks.** While debugging the simulator I printed, for one cell `(s = 2.0, beta = 0.5)` on SHT and for a few
  hand-picked configurations, all-split AUCs (including `test_confirmation`) of Fixed / Motion-Only / Adaptive-Only. No selection was made
  from them; every selection in `config.yaml` comes from the dev-stream tuning. Three simulator bugs found by these checks were fixed before
  `lock`: a Fixed-Burst start that coincided with the end of the previous run stalled the schedule; a scheduler's forced start at the end
  of a run was treated as a continuation that then ended one frame later; and C1's strict `<` threshold test rejected all frames whose
  statistic equals a quantised quantile (now `<=`).
* The plan's Motion-Only grid originally had no extension axis and no periodic-check axis; both were added so that the two schedulers get the same
  tuning effort (E00-E02 were unaffected).
* A scheduler run that would end and, in the same frame, satisfy a new start condition simply continues (counted as an extension); the cost is
  charged exactly (scoring per window-frame).
* Ordering note for the report: results for `test_confirmation` are produced only by `run`/`report`; verdicts use them, selections never do.
