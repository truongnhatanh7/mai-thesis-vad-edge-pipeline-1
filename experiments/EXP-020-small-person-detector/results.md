# EXP-020 — results (labels × P2 head, COCO only)
Status 2026-11-10 08:45: all verdicts final (VM timing + delivered finished 07:59, `scripts/chain_timing.sh`). Full tables: `artifacts/EXP-020/report.md` / `report.json`
(`python -m edgelab.analysis.exp020 report`). All four arms: 4 epochs on 64,115 COCO person images, no ShanghaiTech/UBnormal data.

## Verdicts (paper variant, Δ vs M00 paired bootstrap over videos)
| # | rule | result | verdict |
|---|---|---|---|
| P1 label cause | ML0 − M00 ≥ +2.0, CI > 0, recall(48–96) higher | **+16.24 [+11.99, +20.45]**; recall 48–96 px 0.142 → 0.856 | **PASS** |
| P2 resolution cause | M0P − M00 ≥ +2.0, CI > 0 | +0.24 [−1.31, +1.76] | **FAIL** |
| P3 general | best (MLP) > M00 on UB L0; SHT ≥ M00 − 0.5 | UB +5.49 [+1.42, +9.99]; SHT +17.70 | **PASS** (ML0 also: UB +5.57) |
| P4 P2 vs stock@800 | AUC ≥ and ms ≤ 1.10× | NCNN FP16: −8.69 [−11.98, −5.70]; 21.92 vs 19.44 ms (1.13×) | **FAIL** |
| P5 combination | MLP ≥ max(ML0, M0P) (descriptive) | 79.35 vs 77.88 / 61.89 | **PASS** (descriptive; MLP − ML0 untested) |

## AUC × 100 (paper; live in report.md)
| setting | orig@640 | M00 | ML0 | M0P | MLP |
|---|---|---|---|---|---|
| SHT L0 | 68.53 | 61.65 | 77.88 | 61.89 | **79.35** |
| SHT L2 | 67.05 | 61.00 | 74.39 | 58.40 | **76.48** |
| SHT L3 | 60.43 | 58.58 | **62.71** | 54.50 | 62.05 |
| UB L0 | **55.76** | 48.79 | 54.37 | 50.45 | 54.28 |

Recall by reference-person height (SHT L0): ML0 <32 px 0.171, 32–48 0.540, 48–64 0.812 (M00: 0.000 / 0.002 / 0.013;
orig@640: 0.001 / 0.007 / 0.057; stock@800 NCNN: 0.003 / 0.010 / 0.109). Coverage (anomalous, SHT L0) 0.533 → 0.904.

## Cost and delivered (VM, NCNN FP16, 3 reps; SHT L0)
Detector ms (pre+infer+post): stock@640 13.97, **ML0 13.29**, M00 13.16, stock@800 19.44, M0P 21.92, MLP 24.13.
NCNN FP16 640 vs stock@800 NCNN (paper): **ML0 +7.35 [+3.44, +10.70]** at 0.68× the ms; M0P −8.69; M00 −8.99.
Delivered (live AUC, measured trace × s, FILL-HOLD / SKIP):
| detector | s=1 | s=3 (drop rate) |
|---|---|---|
| stock@640 | 67.63 / 66.14 | 66.61 / 65.47 (0.149) |
| stock@800 | 70.30 / 67.57 | 68.50 / 63.51 (0.368) |
| **ML0** | **73.97** / 73.53 | **73.49** / 72.71 (0.170) |
| MLP | 72.77 / 73.77 | 72.98 / 66.56 (0.485) |
FILL-NORMAL is best for ML0 (75.32 / 74.90 live). Filling rescues the slow P2 arms (MLP s=3 SKIP 66.56 → FILL-HOLD 72.98).
**Practical winner: ML0** — same cost as stock@640, +6.34 live delivered at s=1 and +6.88 at s=3 over stock@640 (FILL-HOLD vs FILL-HOLD), beats stock@800.

## Reading
- The blindness to small people is a **label artefact**: the COCO→YOLO converter drops keypoint-less persons, so they are
  trained as background. Keeping them as box-only targets fixes recall below 64 px with no site data (+16 AUC on SHT).
- **Resolution (P2 head) is not the cause**: same recall and AUC as the control, worse at dark levels (L2 −2.60, L3 −4.09, CI < 0).
- **Caveat (honest):** the 4-epoch COCO schedule itself degrades the control (M00 −6.9 vs orig on SHT, −7.0 on UB). The
  causal contrast (vs M00) is clean; against the original model the gain is +9.35 (ML0) / +10.82 (MLP) on SHT L0, but on
  UBnormal ML0/MLP are ~1.4 below orig (54.4 vs 55.8). A longer / lower-LR schedule (or partial labels from the start) is the
  obvious follow-up to remove the schedule penalty; not pre-registered here.
- Dark L3: the label gain shrinks (+4.1) — low light removes the small-person signal the labels let the model use.
