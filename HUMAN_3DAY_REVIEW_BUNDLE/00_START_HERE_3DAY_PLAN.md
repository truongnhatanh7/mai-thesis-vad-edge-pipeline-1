# 00 · START HERE — 3-day plan to own this thesis

> Audience: a Go engineer who knows basic NNs/CNNs. Goal after 3 days: you can (a) explain the whole pipeline on one whiteboard, (b) point at the exact code that proves each claim is not leaked/faked, (c) defend a Q1/Q2 paper package and know its weak spots.
>
> **Ground rule of this bundle:** every number comes from a file in this repo; `reproduce_and_verify.sh verify` re-checks every path, line number and number anchor automatically (see §6). Where the repo's own prose is stronger than its code/tables, I flag it as a **WARNING (W1–W5)** — read `index.html` → tab 6 first if you are short on time.

Paths below are relative to the thesis root `/Users/nanh/Documents/RMIT_MscAI/thesis/`; this bundle lives in `HUMAN_3DAY_REVIEW_BUNDLE/`.

---

## 0. The 5-minute version (read this if you read nothing else)

```
 24 fps camera ──► [C1 gate: skip static frames] ──► YOLOv8n-pose (ML0 fine-tune) ──► 2-stage BoT-SORT
                                                                                          │ tracks of 17x3 keypoints
   score ◄── causal deliverer + queue guard ◄── normalizing-flow scorer (STG-NF/LoRaD) ◄── 24-frame window buffer
```

* **Problem.** Published skeleton-VAD papers report *offline* AUC (e.g. 85.94 on ShanghaiTech) using future frames and pre-extracted AlphaPose skeletons. On a Raspberry Pi 5 streaming 24 fps (41.67 ms per frame) the same recipe collapses: Huang et al.'s `RPi` config runs 6.14 FPS and its delivered AUC falls to 53.32 (`Q2-P1-Pi5_huang_repro/report.md`).
* **What the thesis contributes (strongest → weakest evidence):**
  1. **Honest protocol** `S0→S4` (E00): offline 77.88 → strict-causal 73.58 (SHT, `ML0_mps`), fixed in code and unit-tested.
  2. **Front-end fix at equal FLOPs** (E01): a COCO *label* artefact, not resolution, blinds YOLOv8n-pose to small people: recall `<48 px` 0.4% → 37.0%; S4 AUC stock 66.03 → ML0 73.78 → +two-stage tracker 76.82.
  3. **Valid computation reuse + queue guard on real Pi 5** (E00, E03, E05, Pi5_suite): live Pi 5 delivered AUC Fixed-K1 47.46% → Burst32 74.19% at 24.01 FPS.
  4. **Negative results** (E04, E06, E08, EXP-018/019): INT8, motion vectors, GAU, RevMargin, streaming-prefix norm — all fail or tie.
  5. **Scorer innovations / green / RL** (E08, E09, E10): mostly refuted or *not yet rigorous* — see W1–W5.

---

## Day 1 — "What is the machine?" (concepts + architecture)

| Block | Time | Open (in this order) | You are done when you can… |
|---|---|---|---|
| **1 AM** | 3 h | 1. `index.html` → **Tab 1** (roadmap + 5 concept cards).<br>2. `01_80_20_ML_AND_ARCHITECTURE_CRASH_COURSE.md` (all).<br>3. `../docs/pose-stgnf-explainer.html` §1, §3, §5, §8 only (open in browser; theme toggle top-right). | …draw the pipeline above from memory and explain (in Go terms) what a tracker, a window buffer and a normalizing flow each *do*. |
| **1 PM** | 3 h | 1. `index.html` → **Tab 2**, click every block of diagrams ① ② ③ (full pipeline, YOLOv8n-pose, coupling step).<br>2. `../PAPER_POSITIONING_AND_CHECKPOINT.md` §1 (LoRaD audit) and §2 (contributions).<br>3. `../Q2_P1_SYNTHESIS_FOR_PAPER_WRITING.md` §1 and §2.1–2.3. | …state what `ML0`, `S0…S4`, `C1`, `Cache-C1+Guard`, `A2/A3/A3s` mean and give 3 headline numbers each. |

Checkpoint question (Day 1 end): *"Why is a 24-frame causal window not a latency win, and why does reusing normalised tensors break?"* → crash course §6 and `Q2-P1-E00_audit/report.md` (R7 FAIL).

## Day 2 — "Can I trust the numbers?" (audit + experiments D1/D2)

| Block | Time | Open | You are done when you can… |
|---|---|---|---|
| **2 AM** | 3 h | 1. `03_CODE_AUDIT_WALKTHROUGH.md` §1–§4 (protocol/leak audit).<br>2. Run `./reproduce_and_verify.sh audit-tests` (≈ seconds).<br>3. Read in the editor, in this order: `q2_p1_e00.py::global_most_normal_fill` → `hold_fill` → `landing_frames` → `reuse_audit`; then the 4 matching tests. | …show with your own eyes where S0 leaks the future and where S4 does not. |
| **2 PM** | 3 h | 1. `index.html` → **Tab 3**, directions **D1** and **D2** (tables + negative results).<br>2. `02_EXPERIMENTS_BY_PAPER_DIRECTION.md` §D1, §D2.<br>3. `../experiments/Q2-P1-E05_confirmation/report.md` §1 (answers) and the Pi 5 sustained table in `Q2-P1-Pi5_suite/report.md`. | …defend "Cache-C1 > Fixed at matched cost" with the CI and name two arms that *lose*. |

## Day 3 — "What do I publish and what will reviewers attack?"

| Block | Time | Open | You are done when you can… |
|---|---|---|---|
| **3 AM** | 3 h | 1. `index.html` → **Tab 3** D3, D4, D5 and **Tab 6** (evidence ledger W1–W5).<br>2. `03_CODE_AUDIT_WALKTHROUGH.md` §5 (E09/E10 audit findings).<br>3. `../experiments/Q2-P1-E08_lorad_gaps_proposals/report.md` §1 (pre-registered verdict table). | …explain why E08's own verdicts are REFUTED/NOT SUPPORTED while the positioning doc praises IAC-Flow. |
| **3 PM** | 3 h | 1. `index.html` → **Tab 5** (verdict).<br>2. `04_FINAL_Q1_Q2_PAPER_VERDICT_AND_OUTLINE.md` (all) — fill in the "to-do before submission" list. | …pick Option #1, write its 5 headline claims from the table, and list the 4 fixes that cost < 1 day each. |

---

## 6. Verify the bundle yourself

```bash
cd HUMAN_3DAY_REVIEW_BUNDLE
./reproduce_and_verify.sh verify        # checks every file path, def/class line, number anchor and test name used here
./reproduce_and_verify.sh audit-tests   # runs only the leakage/causality unit tests
./reproduce_and_verify.sh pytest-q2     # the 298 tests in tests/test_q2_p1_*.py + tests/test_app_catalog.py
./reproduce_and_verify.sh pytest        # the entire tests/ folder (slow; collects 687 tests)
./reproduce_and_verify.sh open          # opens index.html
```

## 7. Files in this bundle

| File | What it is |
|---|---|
| `index.html` | 6-tab offline visual graph (double-click; no server, no network). |
| `01_80_20_ML_AND_ARCHITECTURE_CRASH_COURSE.md` | Plain-English ML for a Go engineer, ASCII diagrams, the math you need. |
| `02_EXPERIMENTS_BY_PAPER_DIRECTION.md` | Every experiment grouped into D1–D5 (+ D0 foundations) with tables, CIs, negatives, glossary. |
| `03_CODE_AUDIT_WALKTHROUGH.md` | Claim → file:line → report → test, plus the 5 audit warnings. |
| `04_FINAL_Q1_Q2_PAPER_VERDICT_AND_OUTLINE.md` | The decision, outline, headline tables, reviewer Q&A, fix list. |
| `reproduce_and_verify.sh` | Menu/CLI: pytest, summary.json viewer, verify, open page, Streamlit, MLflow. |
