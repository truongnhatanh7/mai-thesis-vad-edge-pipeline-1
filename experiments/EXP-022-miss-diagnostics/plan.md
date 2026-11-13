# EXP-022 — Free diagnostics on missed people
Frozen 2026-11-08 17:15. (a) Recall by person height (EXP-011 method) for the EXP-013 fine-tuned detector. (b) For each reference
(released AlphaPose) track, classify misses by the YOLOv8n-pose cache: "never seen" (no frame of the track matched) vs
"intermittent" (matched in some frames). Descriptive; decides whether a temporal detector (TSM) is worth testing:
**if intermittent misses ≥ 30% of missed person-frames, propose a TSM experiment; otherwise not.**
