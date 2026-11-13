# EXP-019 — STG-NF with Gated Attention Unit coupling networks (STG-NF-GAU)
Frozen 2026-11-08 12:20 before any run (owner request; D22).

## Idea
STG-NF's affine coupling layers use a spatio-temporal graph-conv subnet to produce scale/shift; any subnet keeps the flow
invertible. Replace it with a **Gated Attention Unit** (GAU; single-head gated attention, Hua et al. 2022) over the time
axis (per joint, with joint embeddings) combined with the existing graph conv over joints. Attention can mask missing
frames natively (relevant to DAPF).
## Method
Same data, normalisation, flow depth, optimiser and epochs as the EXP-001 retraining recipe; ShanghaiTech released
training poses; **3 seeds** for both STG-NF (standard, retrained) and STG-NF-GAU. Parameter budget: GAU variant ≤ 10×
standard. Test: released test poses (paper and live variants), edge poses (YOLOv8n-pose 640, EXP-003 L0_E0 replay) and
top-down poses (EXP-016 n640 K1). CPU cost per segment (batch 1, 1 thread) reported.
| # | Expectation | Pass rule (mean over 3 seeds, paper variant) |
|---|---|---|
| G1 | GAU beats standard on reference poses | AUC(GAU) − AUC(standard) ≥ +1.0 on released test poses, seed-paired CI excluding 0 |
| G2 | no loss on edge poses | GAU ≥ standard − 0.5 on YOLOv8n-pose edge poses and on top-down poses |
| G3 | still edge-cheap | GAU CPU ms/segment ≤ 2 ms (VM or Mac, 1 thread), reported vs standard |
If G1 passes, a GAU-FA variant (provenance channel + attention mask for missing frames) is added to EXP-018 by amendment.

## Amendment 1 — implementation details (2026-11-08 13:00, written before any EXP-019 result; no rule changed)
- **GAU configuration (fixed now, never tuned):** token dim 8, qk dim 8, expansion 2 (hidden 16), SiLU, LayerNorm on the tokens, learned joint embedding (18 x 8), learned relative-position bias over frames (2T - 1 values), single head, attention = relu(q k^T / T + b)^2, output projection zero-initialised. Coupling subnet = the original st_gcn block (graph conv over joints + temporal conv) **plus** the GAU output added to its shift/scale output. Parameters: 5,808 vs 616 (9.43x, within the 10x budget). Code: `src/edgelab/models/stgnf_flow.py`; the graph-only variant of the same code has the same 616 parameters and gives bit-equal NLL to `third_party` STG_NF with copied weights (tests).
- **Training code:** the EXP-001 recipe is re-implemented in `edgelab.pose.stgnf_train` (own epoch loop; windows from `edgelab.pose.segments`, inputs equal to third_party's dataset incl. mirror and normalisation: tests). The *standard* arm uses third_party's own `STG_NF` class. Same code path trains both arms, so differences come from the model only. Not bit-identical to EXP-001's run (different data order / seed streams); the seed-0 standard model is compared to EXP-001's control (85.31) as a sanity check.
- **G1 statistic:** mean over the 3 seeds of AUC(GAU seed s) − AUC(standard seed s); CI = percentile bootstrap over the 107 videos (1000 resamples, the same resample for all six models, paired by seed); the t-interval over the three seed differences is reported next to it. G1 passes if the mean difference ≥ +1.0 and the bootstrap CI lower bound > 0.
- **G2:** same mean-over-seeds paper-variant difference on the two other pose sets (YOLOv8n-pose L0_E0 c0.25 botsort; EXP-016 L0 n640 K1 rtmpose-s).
- **G3:** CPU ms per window = median of 2000 forward passes, batch 1, torch.set_num_threads(1), on this Mac (M4 Pro) while other jobs run (load average noted in results), standard and GAU measured interleaved in the same session. Not a Pi/VM number.
- **Note (training set, found 13:00 while the runs were in progress; affects standard and GAU identically):** 973 of the 145,162 training windows (0.67%) are not used by either arm: STG-NF's rule keeps a window if ≥ 22 of its first 24 frames are present, but its last poses may then lie far later in time; our window grid reaches 26 frames past the start (needed by EXP-018), so these windows are skipped. 
