# EXP-009 walkthrough (plain language)

## The question
The edge pose pipeline (YOLOv8n-pose + tracker) loses people for a few frames at a time. STG-NF only scores a
24-frame window if at most 2 of its frames are missing, so a gap of 3+ frames deletes every window across it.
Idea: fill the gaps, like video codecs do. **B** = linear interpolation between two detections (needs the future, adds
up to G frames of delay). **P** = keep moving the last pose forward with shrinking speed (causal, no delay).

## The code (small, plain)
- `src/edgelab/pose/fill.py`
  - `fill_B(track, max_gap)`: for a gap of at most `max_gap` missing frames between two detections, every keypoint's x, y
    and confidence (and box score) is interpolated in a straight line. Longer gaps stay empty.
  - `fill_P(track, max_gap)`: after a detection, up to `max_gap` frames. Step k moves each keypoint by
    `velocity * 0.8**k`, velocity = change per frame between the last two detections (0 if there is only one);
    confidence = last confidence `* 0.8**k`. Stops at the next detection and at the last video frame.
  - `fill_dir`: runs one of them on every `*tracked_person.json` of a folder. The output has the same JSON fields STG-NF
    reads. Which poses were made up (and their age) goes into a separate file per video in a sibling `<name>_prov` folder.
  - `segment_predicted_shares`: for every scored window, the share of its poses that were predicted. Used for the B-gate
    (drop windows with share > 50%) and for the risk analysis R.
- `src/edgelab/analysis/exp009.py`: runs every condition of `plan.md`, appends one row per condition x variant to
  `~/thesis-data/exp009/rows.jsonl`, skips what is already there (resumable), logs to MLflow experiment `EXP-009`.
  `analyse` computes P1-P4, D, R and writes `results.md`.
- Tests: `tests/test_pose_fill.py` (toy tracks with exact numbers), `tests/test_analysis_exp009.py` (box, IoU, matcher,
  keyframe mask, coverage bootstrap).

## What was run
1. **Offline filling** on the edge pipeline's poses (ShanghaiTech L0, L2, L3; UBnormal L0): B and P at G = 2, 4, 8, 12,
   plus B-gate at G = 8 (same windows as B G8, minus those with predicted share > 50%). G = 0 rows are the existing EXP-003/006 results
   (re-scoring SHT-L0 gave the identical AUC to 16 digits).
2. **Decomposition**: released AlphaPose poses (85.94) -> keep only poses with an edge-detector box (conf >= 0.25) of IoU >= 0.3
   (D1) -> add B G8 (D2). Frame ids of released poses and cache were checked: same range, 0-based, IoU matching peaks at shift 0.
3. **Deadline**: keep masks exactly as `deadline_replay.run_cell` (EXP-004 trace, times s = 1, 2, 4); dropped frames stay as gaps
   at their real time; none / P / B (G 8). **Keyframes**: detector on every k-th frame (k = 2, 3, 4), B with G = k.

## Things to know when reading the numbers
- Without filling at s = 2 and s = 4 no window survives (40% / 70% of frames dropped), so "no fill" = AUC 0.5 by the EXP-003 rule. A fill
  "beats" it trivially; compare fill arms with each other and with s = 1.
- One released video stores frame keys with 3 digits. STG-NF sorts keys as text, so the filler copies the track's own key width.
- Released ids and cache ids matching does not prove the two detectors saw the same person, only that the frames line up.
- The R subsets are tiny (179 frames for SHT-L0 B G8): descriptive only.
- No git commits were made, nothing was tuned after the first computation, no condition was added or removed.

## Where things are
`~/thesis-data/exp009/{rows.jsonl, summary.json, scores/, poses/, run_*.log}`, `EXP-009-pose-filling/{log.md, results.md}`.
