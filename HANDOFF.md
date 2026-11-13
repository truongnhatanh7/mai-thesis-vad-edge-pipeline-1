# HANDOFF — start here (state as of 2026-11-04)

> **Update 2026-11-06 (M2 Pro, autonomous loop):** read `RESEARCH_SYNTHESIS.md` first (headline finding, refined RQ,
> hypotheses, contributions, paper outline), then `RESEARCH_LOOP.md` (iteration-by-iteration record). New
> experiments with plan / log / results: `experiments/EXP-003-lowlight-gates/`, `EXP-004-os-deadlines/` (Lima VM
> `edge-os`, recipe in `vm-setup.md`), `EXP-005-coverage-selector/`. Literature checks: `LITCHECK-2026-11-04.md`,
> `LITCHECK-2026-11-05.md`. Owner actions: review D13–D18 in `experiments/DECISIONS.md`, sign off the three
> results.md files. Data and raw outputs: `~/thesis-data/{exp003,exp004,vm-exchange}/`.

You are an AI agent continuing a master's thesis experiment programme on a new Mac (Apple chip).
Read in this order: **this file → `SETUP.md` → `experiments/README.md` (design baseline) →
`experiments/DECISIONS.md` → each `experiments/EXP-*/results.md`**. `LITREVIEW.md` and `PAPERS.md` give background.

## 1. The research

**RQ:** *How do lightweight vision preprocessing and OS-level optimization, individually and jointly, affect the
robustness and real-time performance of CPU-only public-safety inference on edge devices?*

- Designed as a **2×2 (P × O) factorial**: P = illumination preprocessing (none vs gamma/CLAHE/Zero-DCE++,
  maybe luma-gated), O = OS/CPU tuning (default vs threads/spin/pinning/SCHED_FIFO/cgroups/governor).
  "Jointly" = the **interaction effect**.
- OS tuning changes accuracy only through deadlines: stream at fixed fps with drop-on-late, then measure
  *delivered* accuracy. Preprocessing improves dark-frame accuracy but costs CPU → the two interact.
- Main metric (D2): **UCF-Crime test frame-level AUC, stratified by lighting** (day / night / IR).
  Bootstrap 95% CIs **over videos**, not frames. Pre-register expectations in `plan.md` before running.
- Hardware (D3, D8, D10): train/offline accuracy on the Mac (MPS); Linux subset tests in a Lima VM
  (4 vCPU / 8 GB, timings comparative only); **every timing/OS/energy claim on a Raspberry Pi 5 8 GB**
  (owner is buying it — not yet available).
- Anchor paper: *"An embedded deep learning framework for real-time violence detection and alert
  generation"* (Salman et al., Scientific Reports 2026, https://doi.org/10.1038/s41598-026-44939-x).
  CNN-GRU on Pi 5, ONNX Runtime CPU. No code, no weights, custom dataset only on request.

## 2. Owner (calibrate to this)

Software engineer (Go expert, basic Python, novice bash) at a public-safety tech company; ML beginner, basic
stats. Has trust issues with AI code → every experiment needs a **high-level WALKTHROUGH.md + tests**, and every
number must cite an MLflow run id. Explain in **plain language; describe a paper by what it does first, author
name only as a trailing tag**. Owner prefixes shell commands with `rtk` (token-filter wrapper; optional for you).
Owner said: don't shrink scope to toy size; publishable target (workshop / mid-tier venue).

## 3. Rules you must keep

- **Never fabricate** numbers, citations, URLs. A number in a doc must come from a run (cite run id) or a
  verified source.
- Each experiment = folder `experiments/EXP-NNN-<slug>/` with `plan.md` (Background, Problem, Objective, Method,
  Pre-registered expectations, Deviations policy) → `log.md` (append-only, timestamped notebook) →
  `results.md` (Results w/ run ids, Observations, Discussion, Threats, Next steps, Owner sign-off) +
  `WALKTHROUGH.md` + `config.yaml`. Format in README §10.
- Decisions: append a line to `DECISIONS.md` (never edit old lines). Protocol changes need owner approval.
- Third-party code in `experiments/third_party/` (gitignored), pinned to a commit, our changes as a `.patch`.
- **Licences research-only (D11):** STG-NF CC BY-NC 4.0, Ultralytics AGPL-3.0. Never put in employer code.
- Timings: tag `machine` (mac/vm/pi), never mix. Don't run timing while other heavy jobs run (it happened —
  see EXP-001/002 logs). Ultralytics silently resets torch threads to 8 on model load — re-set after load.
- Commit only when the owner asks. Sub-agents get non-overlapping lanes and don't commit.

## 4. Status and findings

| Exp | What | Result | State |
|---|---|---|---|
| EXP-000 | Get UCF-Crime (official), verify, labels, lighting stats | 290 test videos verified (1,111,808 frames, 7.6% anomalous). Frame labels == VadCLIP gt exactly. All 1,950 videos now downloaded. | done |
| EXP-001 | Pose pipeline (STG-NF: tiny normalizing flow over tracked skeletons, Hirschorn & Avidan ICCV'23) on ShanghaiTech | Released poses + checkpoint: **85.94 AUC** (paper 85.9) ✅; no smoothing 81.93 ✅. With edge poses (YOLOv8n-pose + BoT-SORT, imgsz 640): **68.58**. Retrain on those poses: 67.59 (control retrain on released poses 85.31 → recipe OK). Causal/live scoring: 80.98 (released) / 66.42 (YOLO). imgsz 864: 71.71. | done, owner sign-off pending |
| EXP-002 | Rebuild anchor CNN-GRU (Table 3, 177K params, 0.099 GMAC/frame), train on Hockey Fight | Video-level split, 3 seeds: **0.940 / 0.940 / 0.933** video acc (CI upper ≤ 0.973) vs paper 97.6 → **not reproduced**. Leaky split only +0.6 pp. ONNX parity OK. ORT CPU 3.2 ms/frame (1 thr) vs paper 67 ms (contaminated machine, re-measure). | Hockey done; RLVS, ShanghaiTech, UCF-Crime not run |

Key observations:
- **Paper audit (EXP-002):** Table 9 Hockey confusion matrix sums to all 1000 videos (not a test split);
  matrix accuracy = 97.0, printed 97.6 = its precision. Claims an "official split" Hockey doesn't have.
- **Why the pose pipeline drops 17 pts:** small (~50 px) people get conf 0.26–0.36 at imgsz 640 → tracker
  doesn't start tracks → only 67.5% of anomalous frames get any score (96.5% with released poses).
  At imgsz 1280 conf 0.69–0.81. This is the same mechanism low light should trigger → good thesis hook.
- **Per-stage CPU (Mac, comparative):** pose 28.0/19.2/15.4 ms/frame at 1/2/4 threads (~90% of cost),
  tracker ~2.3 ms, STG-NF ~0.7 ms/segment.
- **Lighting (EXP-000):** mean luma misses IR footage (e.g. Burglary079 luma 121 but IR); black borders in
  192 videos skew luma; 32 mostly-greyscale (IR) videos, 26 of them anomalous → **lighting confounded with
  class**, night stratum will have few normals and wide CIs. ShanghaiTech has no dark videos (daylight control).

## 5. Concerns / threats

- Anchor not reproducible; its custom dataset unreleased (request email drafted, owner hasn't sent).
- A Pi 5 skeleton-VAD paper (Huang 2026, IEEE ESL, only abstract read) reports 86.2 with an apparently similar
  pipeline; we measure ~68 → read the full paper before trusting either.
- ShanghaiTech raw frames come from a third-party HF mirror (verified against official GT, still a threat).
- R3 retrain: one seed, one setting. Mac timings so far were taken under load.
- STG-NF training fails on MPS (float64) → CPU only (fine, it's tiny).

## 6. Open decisions (waiting on owner)

1. **Freeze the pose workload:** (a) accept ~68, (b) **sweep imgsz {640,864,1280} × det conf × tracker
   thresholds, report AUC vs ms/frame** — recommended, ~few hours, could be a result itself, (c) try a
   stronger small pose model.
2. Run RLVS for EXP-002 alone on the GPU (~0.5–1.7 h/seed) — recommended.
3. EXP-002 protocol additions: clip length T ∈ {4,12} sensitivity; stronger frame-level leakage variant.
4. Lighting threshold — after owner hand-labels `EXP-000-data/labelling_sheet.csv` (day/night/ir/unclear).

## 7. Next steps (after decisions)

1. EXP-002: RLVS; then UCF-Crime frame-AUC protocol (plan §5.5) and ShanghaiTech.
2. Pose sweep → freeze workload → **Phase 2 baseline**: frozen workload on UCF-Crime test by lighting stratum,
   CPU-load profile.
3. Phase 3 P arm (gamma, CLAHE, Zero-DCE++, luma gate) → Phase 4 O arm → Phase 5 P × O factorial
   (README §5, backlog §7). Then Lima VM subset → Pi 5.

## 8. Owner's own to-dos (remind them)

Buy Pi 5 8 GB · send `EXP-002-anchor-reconstruction/request_email_draft.md` · hand-label lighting sheet
(~2–3 h) · confirm thesis is not employer-sponsored (licences) · sign off results.md files.

## 9. Bundle contents

```
HANDOFF.md  SETUP.md  LITREVIEW.md (owner's draft)  PAPERS.md (47 key papers, condensed)
mlflow_runs.csv                 all 58 runs (ids, params, metrics) — run ids cited in results.md
experiments/                    code + records, git-tracked state of commit listed in SETUP.md
  src/edgelab/ tests/ EXP-000-data/ EXP-001-*/ EXP-002-*/ README.md DECISIONS.md pyproject.toml uv.lock
  artifacts/                    our small trained models (see SETUP.md §4)
data-tools/ucf-crime/           UCF-Crime fetch/verify scripts + README + index/lighting CSVs
```
