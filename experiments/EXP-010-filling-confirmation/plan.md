# EXP-010 — Confirmation: does causal pose prediction beat skipping under a CPU deadline?

Pre-registration frozen 2026-11-07 17:00. **Status: frozen, NOT run — waiting for the owner's explicit go (D20).**

## 1. Why
EXP-009 found (exploratory, chosen after seeing the data) that predicting the poses of frames a slow device drops
("P-fill") beats the skip-and-reindex policy of EXP-004/008 by +1.4 to +5.6 delivered AUC (live), significant in
5 of 6 cases, on both datasets. Two problems keep it from being a claim: (1) the comparison was not pre-registered;
(2) the two policies also scored the remaining blank frames differently (skip-and-reindex holds the last score;
EXP-009's fill path gave blank frames the "normal" score). This experiment fixes both, on conditions not yet seen.

## 2. Method (all frozen models; no new detector runs except one timing measurement)
- **Groups (6):** ShanghaiTech L0, L2, L3 with YOLOv8n-pose (EXP-003 caches); UBnormal L0 with YOLOv8n-pose
  (EXP-006 cache); ShanghaiTech L0 and UBnormal L0 with YOLOv8s-pose (EXP-007 caches). E0, imgsz 640, c 0.25.
- **New slow-downs:** s ∈ {3, 6} (EXP-008/009 used 1, 2, 4, 8). Camera 24 fps, latest-frame policy.
- **Timing traces:** EXP-004 VM trace `L3-E0-R640-c0.25-O0-B4` × s for YOLOv8n-pose. For YOLOv8s-pose, step 0
  measures its cost once: YOLOv8s-pose ONNX (384×640, same export procedure as Amendment 1) timed in the VM on the
  same 12-video × 40-frame subset, O0-B4, co-load on, 3 repetitions, host-load rule as EXP-004; the 8s trace =
  the 8n trace with each frame's detector time multiplied by r = mean(8s detector ms) / mean(8n detector ms).
- **New randomness:** keep-mask bootstrap **seed 1** (EXP-004/008/009 used seed 0).
- **Policies compared on identical keep masks:**
  1. **SKIP** — skip-and-reindex (EXP-004 `run_cell`): processed frames re-indexed; scores held to the next
     processed frame.
  2. **FILL-HOLD (primary)** — dropped frames stay at real time; P-fill with G = 8 (EXP-009 rule: damped constant
     velocity, ×0.8 per frame, confidence ×0.8^k, stop at the next detection); STG-NF scoring; frames that still
     have no score get the **last available score** (same rule as SKIP).
  3. **FILL-NORMAL (continuity)** — as 2 but blank frames get the "normal" score (EXP-009 behaviour).
- Live variant for the tests; paper variant reported. Paired bootstrap over videos (1000 resamples).

## 3. Pre-registered expectations

| # | Expectation | Pass rule |
|---|---|---|
| C1 | FILL-HOLD beats SKIP at s = 3 | AUC(FILL-HOLD) − AUC(SKIP) > 0 with 95% CI excluding 0 in **≥ 4 of 6** groups (live) |
| C2 | FILL-HOLD beats SKIP at s = 6 | same rule at s = 6 |
| C3 | effect is not the scoring rule | mean over the 12 (group × s) cells of [FILL-HOLD − SKIP] > 0 (descriptive companion to C1/C2) |
| C4 | bigger detector | both YOLOv8s groups have FILL-HOLD − SKIP > 0 at s = 3 (descriptive) |

The claim "causal pose prediction rescues delivered accuracy under CPU deadlines" is made in the paper only if C1
**and** C2 pass. If either fails, EXP-009's result is reported as exploratory and not replicated.

## 4. Expected cost
≈ 36 replays (6 groups × 2 s × 3 policies) + one short VM timing run; ≈ 1 hour including write-up. No new
detector caches.

## 5. Deviations policy
As EXP-003. No new groups, slow-downs, policies or rules after the first computation.
