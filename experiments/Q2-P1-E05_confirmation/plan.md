# Q2-P1-E05 — Confirmation: pre-registered plan

*Written and frozen before any E05 evaluation number exists. `config.yaml` (written by `q2_p1_e05 lock`) pins the locked system, the
cells, the arms, the verdict rules and the hash of this file. Anything that changes after the first E05 run is logged in "Amendments" with
its reason; nothing below is edited retroactively.*

**Ladder row (plan):** *Lock the best candidate; test full protocol, UBnormal, short events, new entries and sustained Pi operation.*
**Output:** reproducible results, confidence intervals, limitations. **Training: no.**

**Hardware disclosure (unchanged since E02).** The physical Raspberry Pi 5 is **not connected**. Everything labelled "Pi" is the
`edge-os` ARM64 VM (4 vCPU, 8 GB; comparative proxy) with service times multiplied by a Pi factor `s`, or - for the real run in section C1 -
the real VM pipeline driven with arrivals accelerated by `s` (the dual of slowing the machine down by `s`). No claim is made about absolute Pi 5
numbers.

## 1. Locked system (E00 - E04 results, nothing re-tuned)

| Component | Locked value | Origin |
|---|---|---|
| Frontend (ShanghaiTech) | `ML0_ncnn`, NCNN FP16 @640 (frozen detector) | E01 / E02 |
| Frontend (UBnormal) | `ML0_mps` @640 (proxy; no UB NCNN cache) | E01 / E02 |
| Control frontend | `stock` YOLOv8n-pose @640 (SHT: NCNN FP16 cache of E04, UB: MPS) | E00 / E04 |
| Tracker | two-stage BoT-SORT `c=0.20, low=0.05, buffer=30, match=0.85` | E01 |
| Control tracker | E00 default `c=0.25, low=off, buffer=30, match=0.80` | E00 |
| Scorer | frozen STG-NF (`ShanghaiTech_85_9.tar`, `UBnormal_unsupervised_71_8.tar`), E00 strict causal S3/S4 completion-landing | E00 |
| Sentinel / hold | train-calibrated sentinel (SHT max = -0.5122, UB p50 = -0.8273); hold `H = W_ref - 1` | E00 |
| Schedule | Fixed-Burst intact windows `F(W)` (`W <= L`: detector every frame, scoring thinned; `W > L`: an `L`-frame run every `W` frames, tracker re-created per run) | E02 |
| Reuse | C1 = cross-frame static-validity detector-output reuse, level 3, `A1 = 4`, thresholds = dev 50 % quantile of the one-frame statistics (SHT roi 1.8267 / ext 1.0; UB in E03 `config.yaml`) | E03 |
| Guard | queue-backlog load shedding of Independent-Combo: shed on at wait > 400 ms, off at wait < 150 ms; while shed no new run starts and the running run is aborted | E03 |
| +MV, +INT8 | **not adopted** (E04) | E04 |

**Primary lock: `Cache-C1`** (Fixed-Burst + C1). **Overload-safe variant: `Cache-C1+Guard`.** Both are pre-registered against `Fixed`
(compliant Fixed-Burst), `Full-Rate` (K = 1, the locked tracker, every frame) and `Stock-Baseline` (`stock@640 + default tracker + Fixed`).

*Guard port.* In Independent-Combo the guard also thins scoring on odd frames when wait > 200 ms. In an intact-window schedule a window
completes exactly once (at the run end), so thinning would void it; only the shedding half (400 / 150 ms hysteresis, deferral of starts, abort of
the running run) is ported. When the guard never triggers `Cache-C1+Guard` is bit-identical to `Cache-C1` (rule E5-D2).

## 2. Arms (E05 simulation family)

| Arm | Frontend / tracker | Schedule | Reuse | Guard | Role |
|---|---|---|---|---|---|
| `Stock-Default-K1` | stock / default | K = 1 | - | - | ladder step 0 (E00 control) |
| `ML0-Default-K1` | ML0 / default | K = 1 | - | - | ladder step 1 (E00 system) |
| `Full-Rate` (= `ML0-Locked-K1`) | ML0 / locked | K = 1 | - | - | ladder step 2 (E01) |
| `Stock-Baseline` | stock / default | `F(W)` | - | - | baseline at matched cost |
| `ML0-Default-Fixed` | ML0 / default | `F(W)` | - | - | matched-cost control |
| `Fixed` | ML0 / locked | `F(W)` | - | - | ladder step 3 (E02) |
| `Fixed+Guard` | ML0 / locked | `F(W)` | - | yes | isolates the guard |
| `Cache-C1` | ML0 / locked | `F(W)` | C1, oracle identity (E03 convention) | - | **primary lock** |
| `Cache-C1+Guard` | ML0 / locked | `F(W)` | C1, oracle identity | yes | **overload-safe lock** |
| `C1-ReTrack` | ML0 / locked | `F(W)` | C1, **no oracle** (re-track) | - | non-oracle test |
| `C1-ReTrack+Guard` | ML0 / locked | `F(W)` | C1, no oracle | yes | non-oracle test |

*Non-oracle C1 (`C1-ReTrack`).* In E03, windows of reused frames were rescored with the *source* pose of the *same track* (`§9b point 6a`:
the track identity is taken from the genuine tracker = oracle). In `C1-ReTrack` the tracker never sees a genuine detection on a C1-skipped
frame: the source frame's detections (boxes, confidences, keypoints) are fed to BoT-SORT at that frame, and windows come from the tracks that
result. The decision pattern (which frames reuse) is the unchanged causal E03 pattern. Costs: the tracker update is still charged on reused
frames (only the pose network and its decode share are saved) via the optional simulator flag `K["c1_retrack"]`.

## 3. Cells, cost matching, thresholds

* **Cells (15):** the 12 E03 cells `(s, beta)` = `s=2.0` x {0.25, 0.33, 0.50, 0.67, 0.75, 1.00}; `s=2.5` x {0.50, 0.75, 1.00}; `s=3.0` x {0.50, 0.75, 1.00}
  plus the native-VM-speed cells `s=1.0` x {0.25, 0.50, 1.00}. The **12 E03 cells are the primary headline** (identical to E03); the three `s=1.0`
  cells are reported separately and in a 15-cell mean.
* **Cost matching (unchanged):** the period knob `W` of every non-K=1 arm is bisected so the realised stream CPU (observer / C1 checks included, tracker
  charged on reused frames for ReTrack) is the largest value <= `beta * C_full(ML0-Locked-K1, s)`. One absolute target per cell for *all* arms, so
  `Stock-Baseline` and `ML0-Default-Fixed` spend the same CPU as `Fixed`. Status flags `ok / under / over`.
* **Hold:** `H = round(W_ref) - 1`; `W_ref` = the cell's `Fixed` period for ML0-locked arms, the arm's own period for `Stock-Baseline` and
  `ML0-Default-Fixed`, 1 for K = 1 arms.
* **Alert threshold (event metrics):** `theta` = the 95th percentile of the K = 1 *ideal* series on `test_dev` normal frames of the arm's own family
  (ML0-locked family: the E03 constant, SHT 1.3774, UB 1.3030; ML0-default and stock families: derived the same way in the `run` stage from
  `test_dev` only). Alert = score > theta for >= 6 consecutive frames.
* **Streams:** one continuous FIFO stream of all clips (queue carried across clips), service times from the E02 VM pools x `s`, late work > 2000 ms
  dropped by the delivery (E00 / E03 convention).

## 4. Part A - full-protocol progression and confirmation

* Datasets: ShanghaiTech (107 clips; 35 dev / 72 confirmation) and UBnormal (211 clips; 70 / 141). Subsets: `test_dev`, `test_confirmation`, `all`.
* Metric: pooled delivered (causal, completion-landing) frame-level AUC, in %.
* Paired video-level bootstrap, **2,000 resamples**, seed 0, shared multinomial weights for every series of a subset; cell-averaged contrasts
  use the mean of the per-cell bootstrap differences (same as E03).
* Ladder contrasts (each on dev / confirmation / all, per cell and 12-cell mean): `ML0-Default-K1 - Stock-Default-K1` (frontend),
  `Full-Rate - ML0-Default-K1` (tracker), `Fixed - Full-Rate` (price of the budget), `Cache-C1 - Fixed`, `Cache-C1+Guard - Cache-C1`,
  `Cache-C1 - Stock-Baseline` (end-to-end gain at equal cost), `C1-ReTrack - Fixed`, `C1-ReTrack - Cache-C1` (oracle penalty).
* Event metrics: recall @0.5 / 1.0 / 2.0 s, missed-event rate, detection-delay p50 / p90, false alerts per normal hour.

## 5. Part B - stratified subgroups

Scope: all ground-truth anomalous events of the 107 SHT clips and 211 UB clips. For every event `(s, e)` (inclusive frames):

| Stratum | Definition | Levels |
|---|---|---|
| Duration | `e - s + 1 <= 2.5 s` (60 frames SHT / 75 frames UB) | `short_event` / `sustained_event` |
| Entry | a **locked-tracker K = 1** track (full-clip replay, >= 12 poses, first frame >= 3) *starts* within [s - 1.0 s, s + 1.0 s] | `new_entry_near_onset` / `existing_tracks_only` |
| Motion | p90 of the per-frame luma motion statistic over the first second of the event >= the `test_dev` 85th percentile of the clip frames (E03 constant `motion_q.q85`) | `sudden_fast_motion` / `moderate_motion` |
| Crowd | median number of simultaneous K = 1 locked tracks over the first second of the event >= 4 | `high_crowd` / `low_crowd` |

(Sensitivity for *entry*: the same rule on the released AlphaPose tracks of SHT.) Every event belongs to exactly one level of each stratum
(rule E5-D9).

Per subgroup and arm (`Full-Rate`, `Fixed`, `Cache-C1`, `Cache-C1+Guard`, `C1-ReTrack`, `Stock-Baseline`): number of events, event recall @0.5 / 1.0 / 2.0 s,
missed-event rate (never alerted before the event end), detection delay p50 / p90 (ms, detected events), and the **localised event-window AUC**
(pooled over the frames in `[s - 2 s, e + 2 s]` of the subgroup's events, delivered series). Paired clip-level bootstrap, 2,000 resamples.
Cell groups: *budget* = `s=2.0` x `beta` in {0.33, 0.50, 0.75} (mean of per-cell values), *overload* = `s=3.0, beta=1.00`, *native* = `s=1.0, beta=1.00`.
Contrasts: `Cache-C1 - Fixed`, `Cache-C1+Guard - Fixed`, `Cache-C1 - Full-Rate`, `C1-ReTrack - Fixed`.

## 6. Part C - sustained operation

**C1 - real end-to-end run inside the `edge-os` VM** (`limactl shell edge-os`).
Input: the 12 ShanghaiTech timing clips re-encoded to H.264 (x264 `veryfast`, CRF 23, yuv420p, GOP 48) and **looped 4x = 11,520 frames (>= 10,000)**.
Pipeline (all real, T = 4, pinned): `cv2` H.264 decode -> pixel C1 check (E03 24x40 luma thumbnails, level-3 thresholds, age <= 4) -> NCNN FP16 `ML0`
(only when C1 does not hold; a detection is forced every Delta = 12 in-run frames as in the run tables) -> two-stage BoT-SORT (persistent within a run, fed
with the reused detections on C1 frames, i.e. non-oracle) -> window buffers -> PyTorch STG-NF -> online `CausalScoreDeliverer`.
Arms: `Fixed-K1` (the locked tracker, detector every frame) vs `Cache-C1+Guard` (`W = 1`, C1 + guard; the densest schedule, overload handled by the
guard). Frames arrive on a wall-clock schedule at `s x 24 fps` with `s` in {1.0 (native), 3.0, 3.5} (arrival acceleration = a Pi `s` times slower;
waits and the 2,000 ms cap are reported in Pi-equivalent ms = VM ms x `s`).
Measured: wall fps (offered and busy-time throughput), CPU (ms CPU per video-second, VM and Pi-equivalent), C1 hit rate, RSS at 0 / 2,500 / 5,000 / 10,000 /
end frames, queue-wait distribution (p50 / p95 / p99 / max), per-1,000-frame-block service, CPU/frame, loadavg and `/proc/stat` steal as the
thermal / process-stability proxy (the VM exposes no temperature sensor; only `cooling_device*` state files, recorded if readable), delivered AUC of the 12 clips,
and agreement with the simulation of the same stream. Host load outside the VM is sampled (limit 30 %).

**C2 - long-horizon simulation** (same simulator, fresh service-time draws per loop, queue carried across loops via `init_free`):
SHT all 107 clips concatenated (40,791 frames = 28.3 min) and the **5x loop (203,955 frames = 2.36 h)**; UB all 211 clips concatenated once
(**measured length 92,640 frames = 51.5 min at 30 fps**, see Amendment A0). `s` in {1.0, 2.0, 2.5, 3.0}. Arms: `Fixed-K1`, `Cache-C1` (`W = 1`),
`Cache-C1+Guard` (`W = 1`), `Fixed` and `Cache-C1+Guard` at the `beta = 0.50` period of the cell. Reported per 10-min segment: wait mean / p95 / p99 / max,
CPU, modelled memory (E02 RSS + queued decoded frames), shed fraction; plus a first-third vs last-third drift test.

## 7. Hypotheses and verdict rules (pre-registered)

All CIs: 2,000-resample paired bootstrap, 95 %. "12-cell mean" = mean of the per-cell paired differences over the 12 E03 cells.

* **V1 (lock confirmed, SHT).** `Cache-C1 - Fixed`, `test_confirmation`, 12-cell mean: CI lower > 0 -> *confirmed*; delta > 0 with CI containing 0 -> *directionally consistent*; CI upper < 0 -> *rejected*. Same statistic on `all`.
* **V2 (UBnormal replication).** The same statistic on UB `test_confirmation`: CI lower > 0 -> *replicated*; delta >= 0 and CI contains 0 -> *no significant effect (non-negative)*; CI upper < 0 -> *not replicated*.
* **V3 (end-to-end).** `Cache-C1 - Stock-Baseline`, `test_confirmation`, 12-cell mean, both datasets: CI lower > 0 -> *confirmed*.
* **V4 (no oracle).** SHT `test_confirmation`: `C1-ReTrack - Fixed` CI lower > 0 **and** delta >= 50 % of the oracle `Cache-C1 - Fixed` delta -> *holds without oracle identity*; delta > 0 with CI containing 0 or < 50 % -> *attenuated*; CI upper < 0 -> *does not hold*. Same on UB.
* **V5 (guard).** (a) non-overload cells (those where the unguarded `Cache-C1` never exceeds 400 ms of queue wait): `Cache-C1+Guard - Cache-C1` CI lower > -0.25 pt (non-inferiority) - in these cells the guard is bit-identical by E5-D2; (b) overload cells (peak wait of `Cache-C1` > 2,000 ms): CI lower > 0 -> *guard helps*; no overload cell -> *not testable*.
* **V6 (short events).** `short_event`, *budget* cell group, `Cache-C1 - Fixed`: recall @1.0 s lower CI bound > -5 pt -> *no hiding of short events* (non-inferiority margin 5 pt); the same for `Cache-C1+Guard - Fixed`.
* **V7 (new entries).** `new_entry_near_onset`, *budget* group: recall @1.0 s CI lower > -5 pt **and** detection-delay p50 difference (Cache-C1 - Fixed) CI upper < +250 ms (6 frames SHT) -> *entries not delayed*.
* **V8 (sustained simulation, C2).** For `Cache-C1+Guard` and every `(dataset, s)`: max wait <= 2,000 ms, p99 <= 2,000 ms, last-third mean wait - first-third mean wait <= 100 ms and modelled queue memory <= 200 MB -> *bounded*. `Fixed-K1` is expected to diverge where `s >= s*` (E02: SHT 3.08, UB 1.66); this is recorded, not a pass criterion.
* **V9 (real VM run, C1).** `Cache-C1+Guard`: >= 10,000 frames; RSS(end) - RSS(2,500) <= 5 % of RSS(2,500); at every `s` the p99 Pi-equivalent wait <= 2,000 ms; no process fault. `Fixed-K1` at `s = 3.5` is expected to exceed that (recorded).

## 8. Correctness rules

* **E5-D1** E03 replay: `Fixed` and `Cache-C1` at the 12 E03 cells reproduce the E03 delivered series, knobs and costs bit-for-bit (same tables, same pools).
* **E5-D2** Guard identity: at the knob of `Cache-C1`, wherever the unguarded run never exceeds the shed threshold, `Cache-C1+Guard` delivers the identical series; guarded queue wait never exceeds the unguarded one's maximum.
* **E5-D3** Table builder equivalence: the E05 table builder in "no reuse, locked tracker" mode reproduces the E03 windows (pool, row starts / ends, uids) on three clips per dataset.
* **E5-D4** Re-track causality: a C1 frame feeds the tracker only a detector frame at or before itself, never older than `A1 = 4` frames and never a frame that was itself reused.
* **E5-D5** Queue invariants (FIFO, completion <= landing, late cap) of E02 hold for every guarded arm at the mid cell, both datasets.
* **E5-D6** Cost matching: every calibrated arm has realised cost <= 1.005 x target; arms with status `under` / `over` are flagged, not hidden.
* **E5-D7** Real run integrity: >= 10,000 frames per run, arrival times monotone, no frame skipped by the decoder, every delivered score finite, windows landing >= completion; the online `CausalScoreDeliverer` series equals the offline reconstruction of the same window records.
* **E5-D8** Bootstrap: 2,000 resamples; the point AUC equals the pooled AUC of the series.
* **E5-D9** Subgroup partitions: each event is in exactly one level per stratum; event totals 194 (SHT) and 199 (UB); short thresholds 60 / 75 frames.

## 9. Outputs

`plan.md`, `config.yaml`, `results.csv` (per video x arm x cell), `events.csv` (per event x arm x cell group, with strata), `sustained.csv` (C2 segments and C1 blocks),
`decisions.jsonl` (locks, rules, verdicts, calibrations, guard transitions of the real run), `summary.json`, `report.md`, figures. Code:
`src/edgelab/analysis/q2_p1_e05.py` (+ `q2_p1_e05_vm.py` for the VM worker), tests `tests/test_q2_p1_e05.py`.

## 10. Known limitations (declared in advance)

1. No physical Pi 5; `s` is a scalar slowdown applied to VM service times (or to the arrival rate in C1). Cache effects / memory-bandwidth contention of a real Pi are not modelled.
2. UBnormal has no NCNN frontend; the `ML0_mps` proxy supplies detections and the NCNN ML0 timing model supplies costs (E02 convention).
3. C1 detection of staleness is a frame-difference heuristic; it cannot see a person entering a region that was empty at the source frame and not modelled by the thumbnail grid (exterior p95).
4. Event recall is low for all arms (E03: 5-11 % at 1 s on SHT, mostly scorer-limited); subgroup counts are small for some strata - CIs are reported, no multiplicity correction.
5. The real run uses a re-encode of 12 SHT clips (x264, our recipe), which is not the original dataset encoding; thresholds were calibrated on the original JPEG frames.
6. E03's table construction restarts the tracker every Delta frames inside a long run; the real pipeline keeps one tracker per run (it forces a detection every Delta frames only to keep costs comparable).

## Amendments

* **A0 (before any E05 run).** The task text states "UB: all 211 clips = 105,711 frames = 58.7 min". The locked UB test labels give 92,640 frames (51.5 min at 30 fps); the measured figure is used for every stream length. SHT (40,791 frames = 28.3 min) matches the task text.
* **A1 (after the first `run`, before any analysis).** Diagnostic E5-D1b (not in section 8): the zero-delay K = 1 series of the two ML0 arms are compared with E01's S4 all-clip AUC. The code tolerance of 0.1 pt was set without accounting for E03's documented run-table handicap (E3-D2: tracker re-created every Delta frames, -0.45 pt on SHT all clips). Observed offsets were -0.47 / +0.05 (SHT locked / default) and -0.56 / -0.47 pt (UB), i.e. the known common handicap; the tolerance is now 0.75 pt. No verdict rule depends on it; the E5-D1 bit-for-bit replay of E03 is unchanged and passes.
* **A2 (before any analysis).** E5-D3 compares the genuine window of every row (not the pool arrays: E03's pool also stores the C1-substituted variants of a window). The first formulation compared pool arrays and reported a false mismatch.
* **A3 (after the VM runs, before the report).** The VM clips are the first 240 frames of the 12 SHT test clips (not the full clips). The post-processing first scored the real series against the full-length labels and failed; it now truncates the labels to the decoded frames and evaluates the simulation of the same stream on the same 240-frame prefixes (the series is causal, so the prefix is unaffected by later frames; waits are taken from the same prefixes). Added pass-over-pass service / CPU drift (same content) next to the first-vs-last-block drift, which is confounded by clip content. No verdict rule changed.
