# EXP-013 results — fine-tuning YOLOv8n-pose on ShanghaiTech TRAINING frames with released AlphaPose poses as labels

> Training: ShanghaiTech training videos only (no test frame / pose / anomaly label), ~1 in 5 frames, split by video;
> 25 epochs (early stop), best epoch 20; val box mAP50 0.951 / pose mAP50 0.808 (`artifacts/EXP-013/train_summary.json`).
> Weights `artifacts/EXP-013/yolov8n-pose-sht-ft.pt`. Evaluation: EXP-003/006 pipeline, c 0.25, frozen STG-NF; paired
> bootstrap over videos (1000). Rows: `~/thesis-data/exp013-{sht,ub}/results.jsonl`, MLflow `EXP-013`. Written by the lead.

## Verdicts
| # | Rule | Measured | Verdict |
|---|---|---|---|
| F1 | SHT L0 anomalous coverage ≥ baseline + 0.04 | 0.674 → **0.952** | **pass** |
| F2 | SHT L0 AUC ≥ baseline + 1.5, CI lower > 0 | **+11.75 [+7.08, +16.40]** (80.28 vs 68.53); live +9.52 [+5.65, +13.28] | **pass** |
| F3 | UBnormal L0 AUC ≥ baseline − 0.5 | **−4.12 [−9.44, +1.21]** (51.63 vs 55.76); live −3.39 | **fail** |
| F4 | dark conditions (descriptive) | L2 **+9.04 [+4.20, +13.88]** (76.09); L3 −1.93 [−7.23, +3.69] (58.51; coverage 0.554 → 0.416) | — |

## Reading (lead)
- Teaching the small pose model the people of these cameras (with free, leakage-free labels) closes most of the edge
  gap on ShanghaiTech: 80.3 vs the 85.9 reference — about the same as the top-down RTMPose pipeline (EXP-016, 80.5) but
  at **zero extra inference cost** (same network, same speed).
- It is **site adaptation, not general improvement**: on UBnormal (other scenes, rendered people) it is worse (−4.1,
  more but noisier detections: coverage 0.701 → 0.786 with lower AUC), and it does not help in deep dark (L3), where it
  finds fewer people — the training frames are daylight.
- Practical framing for the paper: a per-camera fine-tune from the site's own normal footage is cheap and very
  effective; a general-purpose edge model is not obtained this way.

## Threats
Same 13 cameras in train and test (site adaptation by design); pseudo-labels from AlphaPose (its own misses are
inherited); one training seed; UBnormal transfer is the only out-of-domain check.
