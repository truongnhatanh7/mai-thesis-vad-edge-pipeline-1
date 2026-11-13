# EXP-005 — Does label-free person coverage pick the right configuration on unseen light levels?

Pre-registration frozen 2026-11-06 05:40 (before any EXP-005 cache exists), under D13. The hypothesis comes from
a **post-hoc** observation in EXP-003 (log 2026-11-06 05:2x); this experiment is its out-of-sample test.

## 1. Background (plain language)

In EXP-003/004 the anomaly-detection accuracy of every configuration followed one number: how many frames contain at
least one tracked person long enough to be scored ("coverage"). Coverage of *all* frames needs no labels, so a camera
box could measure it while running and switch to the setting that keeps the most people in view. If that works on
lighting it has never seen, it is a practical, cheap controller — the "framework" of the thesis.

## 2. Hypothesis

**H-F:** Choosing, per light level, the configuration with the highest label-free coverage gives (almost) the same
AUC as choosing with labels (oracle), and beats every fixed configuration.

## 3. Method

- **Held-out light levels** (never used before; seeds from the level names): X1 α = 0.2, X2 α = 0.05, X3 α = 0.02
  (between L1/L2, L2/L3, L3/L4), same sensor model as EXP-003 §4.1, no gain. Added to `darken.LEVELS`.
- **Candidate configurations** = the ones that meet the 24 fps deadline on the VM (EXP-004 §2, kept ≥ 0.89 at B4):
  E0, E1 gamma, E2 CLAHE, all at imgsz 640, c 0.25. (Zero-DCE++ and ≥ 800 px are excluded: they drop frames.)
- Caches + replay + STG-NF exactly as EXP-003; both scoring variants.
- **Selector S (fixed now):** coverage = fraction of all test frames with ≥ 1 person scored by STG-NF (computed from
  the per-frame `covered` arrays; no labels). S picks argmax coverage per level; ties → E0.
- **Selector S24 (pre-specified deep-dark guard):** same, but a frame counts as covered only if the covering track
  has ≥ 48 consecutive tracked frames (2 segments), to discount short noisy tracks.
- **Baselines:** fixed E0, fixed E1, fixed E2; oracle = argmax AUC per level (uses labels).
- **Deadline check:** each selected configuration is also replayed through the EXP-004 O0-B4 traces (delivered AUC).

## 4. Pre-registered expectations

| # | Expectation | Pass rule |
|---|---|---|
| F1 | S is near-oracle | regret = AUC(oracle) − AUC(S) ≤ 0.5 at ≥ 2 of 3 held-out levels (paper variant) |
| F2 | S beats fixed policies | mean AUC of S over X1–X3 ≥ mean of the best fixed policy (paper and live) |
| F3 | Coverage ranks configs | Spearman(label-free coverage, AUC) over the 9 held-out (level × E) configs ≥ 0.7 |
| F4 | Deep-dark guard | at X3, regret(S24) ≤ regret(S) (descriptive) |

## 5. Deviations policy

As EXP-003. No new candidate configurations, levels or selector rules after the first EXP-005 cache is built.
