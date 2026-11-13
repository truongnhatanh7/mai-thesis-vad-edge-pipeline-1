# EXP-000 — UCF-Crime data acquisition and lighting description (plan)

Status: **draft, written 2026-11-03 at the start of the work.** This is a data task, not a hypothesis test, so it has no
pass/fail statistics. It does have **verification gates**, listed below. Owner approval is pending.

## Background (plain language)

**UCF-Crime** (Sultani, Chen, Shah, CVPR 2018) has 1,900 untrimmed real CCTV videos (about 128 h) in 13 crime
classes plus normal footage. The standard *anomaly detection* protocol is:

- Training is **weakly supervised**: each training video only carries a label saying whether it contains an anomaly
  (`Anomaly_Train.txt`, 1,610 videos).
- Testing uses **290 videos** (150 normal, 140 anomalous) from `Anomaly_Test.txt`. For these, the authors give
  **frame-level temporal annotations**: start and end frames of up to two anomalous intervals per video.
- The reported metric is **frame-level ROC-AUC**. All frames of all 290 test videos are concatenated and one AUC is
  computed over them.

So the test set plus its frame annotations is the part we cannot do without. It is our main benchmark (DECISIONS D2).

Every published number depends on the exact label construction: 0- or 1-based frames, inclusive or exclusive end,
truncation to 16-frame feature clips. We must therefore pin the convention and check it against a published loader.

**Low-light stratum.** No benchmark labels lighting (README §3). UCF-Crime contains real day, night and IR
(greyscale night-mode) footage, but none of it is labelled. To build a lighting stratum, we first *describe*
lighting per test video (luma and chroma statistics at 1 frame/s). The owner then hand-labels each video as
day / night / IR / unclear. A luma rule is pre-registered **later**, in a separate step, and checked against those
hand labels. **This experiment does not choose any dark threshold.**

## Problem

We need a verified local copy of the UCF-Crime test set (videos, splits, temporal annotations), a tested loader for
per-frame labels, and descriptive lighting statistics. All of it must be auditable: official sources, checksums,
counts, and a published-loader cross-check.

## Objective

1. Download from the **official CRCV source** the following: annotations, split files, the 290 test videos, and then
   the rest of the archive (training videos).
2. Verify the test set:
   - 290 videos present (150 normal / 140 anomalous)
   - every video decodes
   - per-video fps, resolution, duration and frame count recorded
   - annotation intervals lie within the frame counts
3. `edgelab.data.ucf_crime`: test list, per-frame labels with a documented convention, violence-like subset, plus
   unit tests.
4. Lighting table + histogram + a labelling sheet for the owner.

## Method

- **Source.** Official page https://www.crcv.ucf.edu/research/real-world-anomaly-detection-in-surveillance-videos/.
  - Full archive `UCF_Crimes.zip` (~103 GB).
  - Annotation zip `Temporal_Anomaly_Annotation_For_Testing_Videos.zip`.
  - Optional per-part Dropbox folder linked from the same page.
  - The server supports HTTP Range. So we read the zip's central directory remotely and extract the test members
    first. Each member is checked against its **CRC32 and size** as recorded in the official zip. The remaining
    members are extracted the same way in the background. Unofficial mirrors are used only if the official source
    fails.
- **Index.** `ffprobe -count_frames` for every test video gives codec, width, height, fps, duration, container frame
  count and decoded frame count. Output: `index_test.csv`.
- **Labels.** Annotation numbers are read as 0-based indices with the end inclusive (RTFM convention); `-1` means no
  interval. The VadCLIP variant (end exclusive) is available as an option.
- **Cross-check.** For every test video, compare `Σ floor(n_frames/16)·16` with the lengths of the published
  `gt-ucf.npy` (RTFM) and `gt_ucf.npy` (VadCLIP).
- **Lighting.** Decode every frame and keep one per second (`round(fps)` step). Convert to OpenCV YCrCb. Per video,
  record:
  - mean of frame-mean luma
  - pixel-pooled luma p10/p50/p90
  - min/max frame-mean luma
  - mean chroma `|Cr−128|+|Cb−128|`
  - fraction of sampled frames whose chroma < 3.0 (near-greyscale; IR heuristic, descriptive only)
- **Labelling sheet.** `labelling_sheet.csv` with an empty `owner_label` column: day / night / ir / unclear.

## Expectations / verification gates (pre-registered)

| Gate | Expected |
|---|---|
| Test list | 290 lines; 150 `Testing_Normal_Videos_Anomaly/…`, 140 anomalous |
| Annotation file | 290 lines, same names as the test list |
| Violence-like subset (Abuse, Assault, Fighting, Robbery, Shooting) | 38 videos (README §3) |
| All test videos CRC-verified and decodable | 290 / 290 |
| fps | ≈ 30 (ReadMe: "Our videos have 30 frames per second") |
| Total test frames | ≈ 1.11 M (KB note: VadCLIP gt length 1,109,888) |
| Annotation intervals within frame counts | all, or documented exceptions |
| Train list | 1,610 entries, all present in the official archive |

## Deviations policy

Any change to source, convention or definitions is logged in `log.md` with its reason. Lighting definitions are
fixed before any statistics are looked at. The greyscale chroma cut-off (3.0) is a descriptive heuristic. It is not
the stratum threshold.
