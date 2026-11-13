# STATUS-NOW — lead agent's compact working memory (overwrite; history in RESEARCH_LOOP.md)

Updated 2026-11-08 17:50.

## Pointers
Story: `RESEARCH_SYNTHESIS.md` · history: `RESEARCH_LOOP.md` · decisions: `experiments/DECISIONS.md` (D13–D23) · per
experiment `experiments/EXP-0xx-*/{plan,log,results}.md` · owner notes: Obsidian `mai-thesis-*` (00 index, glossary,
note 18 = failed attempts) · research files in repo root (LITCHECK-*, RESEARCH-*).

## Paper direction (owner-agreed 17:10)
"Why edge pose-based VAD misses people, and how to fix it under CPU deadlines": (1) diagnosis + causal label/resolution
evidence (EXP-011, EXP-020); (2) general small-person-aware edge pose detector (partial labels ± P2, COCO-only, tested on
unseen ShanghaiTech + UBnormal); (3) deadline-aware evaluation (EXP-008) + pose filling (EXP-009/010/010b; +4.72 pooled
in EXP-018 data); (4) coverage→AUC bound (EXP-021); (5) negative results.
**Closed / failed (only in Obsidian note 18, do not pursue): DAPF (EXP-018), STG-NF-GAU (EXP-019), YOLO26 (EXP-017),
threshold re-calibration, light-gated enhancer rule.**

## Done
EXP-003…013, 010b, 012, 013, 015, 017 (YOLO26 worse), 018/019 (closed). Key numbers in RESEARCH_SYNTHESIS / results files.

## Running
| Exp | owner | notes |
|---|---|---|
| EXP-020 labels × P2 (COCO) | lane G 71e21d3b-efb0-4e65-b84f-30cc16871996 | COCO download, datasets, 4 trainings (start after e13-chain ends), eval |
| EXP-021 coverage bound, EXP-022 miss diagnostics | lane H 7e113816-a2fe-4fe2-9678-84235069c9e7 | light CPU |

## After jobs finish (lead)
EXP-014 results; review lanes' results; fill glossary Pending; Obsidian notes (11–17 experiments, link note 18); update
RESEARCH_SYNTHESIS headline. Owner to-dos: Pi 5, real night data, review D13–D23.

17:50: EXP-014 (W1 fail, W2 pass; trade-off), EXP-016 final (960 K1 83.11, +14.58; 111.6 ms/frame VM), EXP-017 (YOLO26 worse) and docs/pose-stgnf-explainer.html done. Running: EXP-020 (lane G), EXP-021/022 (lane H).
17:55 EXP-021 (B1 fail small, B2 pass) + EXP-022 (fine-tune recall <32px 0.82 vs 0.001; TSM rule triggered, awaiting owner) done. Running: EXP-020 only.
21:05 Cursor disconnect: EXP-020 unaffected (tmux e20-chain training, e20-eval waiter). M00 epoch 4/4 at 53% (~44 min/epoch;
4 epochs per arm, Amendment 1b). Expected: M00 ~21:30, ML0 ~00:30, M0P and MLP (P2, slower) ~05:30 and ~10:30 tomorrow, evals
chained → final ~12:00–13:00 Nov 9. Lane G (71e21d3b…) probably disconnected: resume it after training ends to write
verdicts/results (chains do the heavy work).
06:40 Nov 9: M0P crashed 00:32 (MPS GPU error during concurrent ML0 eval); chain stopped overnight. Restarted sequentially in
tmux e20-chain: M0P (06:37) → MLP → chain_eval. ETA: M0P ~10:30, MLP ~14:30, evals → ~17:00–18:00 Nov 9. Never run another GPU
job while e20-chain trains.
06:58 loop armed (every 30 min, PID 72091).
17:58 Nov 9: M0P trained 13:02; MLP epoch 3/4 (ETA ~19:30); evals after → ~22:00. Check loop had died (disconnect) — re-armed (PID 94720).
19:31 Nov 9: MLP epoch 4/4 at 96%; evals next (~22:00). Duplicate check loop (PID 72091) killed; loop PID 94720 kept.
2026-11-09 22:17 EXP-020: M00/ML0 had 6 MPS caches crash (exit=1; chain marked stage done anyway). Queued tmux e20-repair (scripts/repair_caches.sh M00 ML0): waits for e20-chain to exit, rebuilds sequentially, re-sweeps. Log ~/thesis-data/exp020/repair.log.
2026-11-10 07:03 EXP-020: repair done 03:21; report run. P1 PASS (ML0−M00 +16.24 [+11.99,+20.45]); P2 FAIL (+0.24); P3 PASS (MLP UB +5.49); P5 PASS desc (MLP 79.35). Caveat: 4-ep schedule degrades control (M00 61.65 vs orig 68.53; UB ML0 54.37 < orig 55.76). P4 + delivered pending: tmux e20-timing (scripts/chain_timing.sh, log exp020/timing_chain.log). results.md written.
2026-11-10 08:40 EXP-020 FINAL: P4 FAIL (M0P vs stock@800 NCNN −8.69, 1.13× ms). ML0 = practical winner: 13.29 ms (=stock@640), +7.35 vs stock@800 NCNN; delivered live s=3 FILL-HOLD 73.49 vs stock@640 66.61 / stock@800 68.50. Obsidian update delegated to Sonnet 5.5 subagent.
2026-11-10 08:52 Correction: ML0 vs stock@640 delivered live s=1 = +6.34 (FILL-HOLD vs FILL-HOLD), not +5.9. Obsidian notes updated (Sonnet): new mai-thesis-19..24, index/scorecard/glossary refreshed. RESEARCH_SYNTHESIS.md still stale (stops 2026-11-06).
