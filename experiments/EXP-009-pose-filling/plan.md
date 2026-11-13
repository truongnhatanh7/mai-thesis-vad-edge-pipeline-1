# EXP-009 — Where the edge front end loses accuracy, and codec-style pose filling (I/P/B)

Pre-registration frozen 2026-11-07 15:35, before any EXP-009 computation (owner: "run the experiment now"; D19).
Owner's idea (2026-11-07): run the pose detector on some frames ("I") and fill the others by prediction —
**P** = causal forward extrapolation, **B** = interpolation between two detected frames — to repair track gaps and
deadline drops. Literature: `LITCHECK-2026-11-07-pose-propagation.md` (propagation itself is known; its use in
pose-VAD under deadlines is not found). STG-NF keeps a 24-frame window only if ≤ 2 of its frames are missing
(`is_seg_continuous`, missing_th 2), so gaps > 2 frames delete every window that spans them.

## 1. Methods (all frozen models; no retraining)

**Filling rules** (applied per track to STG-NF pose JSON, after the tracker):
- **B (interpolate):** for a gap of g missing frames between two detected frames of the same track, with
  g ≤ G, linearly interpolate every keypoint's x, y and confidence.
- **P (extrapolate, causal):** after a detected frame, if the next frames of that track are missing, predict up to
  G frames with damped constant velocity (velocity from the last two detected frames, × 0.8 per frame); confidence =
  last detected confidence × 0.8^k. Stop at the next detection.
- Every pose carries provenance (detected / predicted) and age (frames since last detection) — automatic.
- **Gated variant (B-gate):** windows whose predicted share > 50% are not scored.
- G ∈ {2, 4, 8, 12}; G = 0 is the unfilled baseline.

**Decomposition (ShanghaiTech, released AlphaPose poses = reference 85.94 paper):**
- **D1 edge recall:** keep a released pose only on frames where the edge detector (EXP-003 cache L0_E0, conf ≥ 0.25)
  has a box with IoU ≥ 0.3 to the released pose's keypoint box. = reference pose quality with edge recall.
- **D2:** D1 + B filling (G = 8). = how much of the recall loss filling can recover with good poses.
- Edge pipeline E (YOLOv8n-pose + BoT-SORT) = 68.53. Gap attribution: (ref − D1) = recall/gaps; (D1 − E) = the rest
  (pose/track quality).

**Offline filling on the edge pipeline:** EXP-003/006 caches, c 0.25: ShanghaiTech L0, L2, L3 (E0 @640) and
UBnormal L0 (E0 @640); B and P at G ∈ {2, 4, 8, 12}; B-gate at G 8.

**Deadline rescue:** EXP-004 O0-B4 traces for E0 @640 scaled by s ∈ {1, 2, 4} (EXP-008 method), groups SHT-L0,
SHT-L2, UB-L0. Dropped frames are left as gaps at their real time (no re-indexing), then (a) no fill, (b) P fill,
(c) B fill, with G = 8. **Keyframe schedule:** detector on every k-th frame (k = 2, 3, 4) + B fill (G = k), SHT-L0 and
UB-L0. Scored with the standard STG-NF assembly (frames without a pose get the "normal" score), both variants.
B adds up to G frames of latency (reported).

## 2. Pre-registered expectations

| # | Expectation | Pass rule (paper variant unless stated) |
|---|---|---|
| P1 | filling repairs offline track gaps | SHT L0: AUC(B, G 8) − AUC(G 0) > 0 with paired 95% CI excluding 0; coverage rises |
| P2 | filling rescues deadline drops | at s = 2, delivered AUC (B fill) > delivered AUC (no fill) in ≥ 2 of 3 groups, live variant |
| P3 | keyframe schedule is nearly free | SHT L0, k = 2 + B: AUC within 1.0 of the full-rate offline AUC (both variants) |
| P4 | B beats P | at G 8, AUC(B) ≥ AUC(P) in ≥ 3 of 4 offline groups |
| D | decomposition | descriptive: share of (ref − E) explained by D1; recovery D2 − D1 |
| R | hidden-anomaly risk | descriptive: AUC on frames whose score comes from mostly-predicted windows vs detected-only windows |

## 3. Deviations policy
As EXP-003. No new rules, G values or groups after the first computation.
