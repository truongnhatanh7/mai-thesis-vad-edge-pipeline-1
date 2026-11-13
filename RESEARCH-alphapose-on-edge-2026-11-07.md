# RESEARCH 2026-11-07: running an AlphaPose-class (top-down) pose pipeline on an edge CPU (Raspberry Pi 5)

Research lane output. Web search/fetch only. No experiments were run. No repo file was touched except this one.
Question (owner): how can an AlphaPose-class, top-down, high-accuracy pose pipeline run effectively on an edge CPU, including OS tuning, so that accuracy is as good as possible?

## Read this first (three paragraphs)

**Tags used everywhere.** `[M]` = a number copied from a source (measured by that source). `[D]` = arithmetic I did on `[M]` numbers (the formula is shown). `[E]` = my estimate or assumption, not measured by anyone I found. Anything with no tag is a plain statement from the cited page.

**Read-level key.** `full` = the fetch tool returned the page text and I checked the passage I cite (not necessarily every line). `abstract` = abstract or search excerpt only. `card` = model card, docs page or README. `table` = a table that I read as extracted text (the layout of PDF-extracted tables can be garbled, so I say where that matters). The search tool also returns a "Synthesis" paragraph. I did NOT use that paragraph as evidence, only the quoted page excerpts and the fetched pages. Venue/year is the one printed on the fetched page. Where I give a venue from memory it says so. Web search only (no Scholar/Scopus), so "not found" is not "does not exist".

**The headline in plain language.** AlphaPose is top-down: a big person detector first, then a pose network on each person crop. Our measured diagnosis (EXP-009) says 96% of the 17-point AUC gap to the released AlphaPose poses is *which people are detected*, and only ~0.7 AUC is *pose quality*. So the honest answer to "how do we run an AlphaPose-class pipeline on a Pi?" is: the expensive part you cannot afford is the **detector** (YOLOX-X, 281.9 GFLOPs), not the pose network. The pose network can be replaced by something 8-40x cheaper (RTMPose-t/s/m) with little loss, but that alone will not recover the missing people. What top-down gives you is a way to **decouple** how often you look for people (expensive, high resolution, every K frames) from how often you pose them (cheap, every frame or staggered). Sections 6 and 7 turn that into a design and five experiments. Experiment 1 (oracle boxes + cheap pose) is nearly free and tells us whether any of this is worth building.

---

## 0. What our own results already say (from the five files read first)

- Reference (released AlphaPose) poses give 85.94 AUC on ShanghaiTech (paper variant); our edge front end gives 68.53; with reference-quality poses restricted to the people our detector finds, 69.23 (D1). 96% (paper) / 91% (live) of the gap is detection coverage (EXP-009, `D`).
- Our front end today: YOLOv8n-pose, rect letterbox 384x640 (245,760 px), VM (M2 Pro, 4 vCPU, ORT): 29.9 ms infer, 34.4 ms service p50, 62.7 ms p99 (EXP-004 §2). Offline gains from more pixels: 800 px +1.6, 960 px +2.0 AUC (live) but under a 24 fps deadline on the VM 960 px already loses (63.8 vs 66.2 delivered, EXP-004 §3).
- Causal pose prediction (P-fill) recovers delivered AUC when frames are dropped: +5.7 to +8.3 live at 3x slower (5 of 6 groups significant), +6.1 to +7.4 at 6x slower on ShanghaiTech-8n; no rescue above ~90% drops (EXP-010).
- OS: SCHED_FIFO escaped the cgroup v2 `cpu.max` quota and worsened p99 (88 vs 56 ms at B4); pinning + co-load isolation without FIFO cut p99 by ~44% under a 2-CPU budget (EXP-004 §5).
- A keyframe schedule (detector every 2nd frame + interpolation fill of poses) lost 1.8 to 2.7 AUC (EXP-009 P3). Important for this document: that test filled the in-between frames by *interpolating poses*; it did not run a pose network on tracker-propagated boxes. The top-down design below does the latter, which is a different thing and is untested.

---

## 1. What AlphaPose actually runs

**Plain language.** AlphaPose is a pipeline with three swappable parts. (1) A person detector draws a box around each person in the frame. (2) A pose network looks at the picture inside each box, resized to a fixed size (256x192 pixels), and outputs the body joints for that one person. (3) A tracker (PoseFlow, or a re-identification network) links the same person across frames. Cost = one big detector call per frame + one pose-network call per person. More people in the frame means more pose calls, which is the main weakness of top-down methods on small CPUs.

### Sources (what it does -> finding -> venue year; URL; read)

- AlphaPose paper: whole-body multi-person pose + tracking; modules are replaceable: detectors YOLOX, YOLOv3-SPP, EfficientDet, JDE; pose nets SimplePose, HRNet and the authors' FastPose (with DCN/DUC variants); trackers: re-ID based or PoseFlow; a five-stage multi-process pipeline with FIFO queues between stages so stages run in parallel; top-down cost grows with the number of persons; tested on one Nvidia 2080Ti GPU -> IEEE TPAMI 2022 (venue from the README citation; the arXiv page does not print it). https://arxiv.org/abs/2211.03375 ; read: full for the module and pipeline passages.
- AlphaPose README: v0.2.0 runs "20 fps on COCO validation set (4.6 people per image on average)", 71 mAP (GPU not stated); the MXNet version ran at 23 fps -> GitHub README. https://github.com/MVIG-SJTU/AlphaPose (raw: https://raw.githubusercontent.com/MVIG-SJTU/AlphaPose/master/README.md) ; read: card (full).
- AlphaPose Model Zoo: COCO AP with YOLOv3 detections and flip test: FastPose-R50 72.0, HRNet-W32 72.5, FastPose-DUC-R152 73.3; all at 256x192 input; speed numbers are batch-64 iterations on a TITAN XP with offline detections (not per-image latency) -> GitHub docs. https://raw.githubusercontent.com/MVIG-SJTU/AlphaPose/master/docs/MODEL_ZOO.md ; read: card (full).
- RTMPose paper reproduces the AlphaPose rows with compute: FastPose-R50 5.91 GFLOPs, FastPose-DUC-R50 9.71, FastPose-DUC-R152 15.99 GFLOPs, all 256x192, detector YOLOv3 at 608x608; COCO val AP 71.2 / 71.7 / 72.6 (YOLOv3) and 69.7 / 70.3 / 71.3 (Faster-RCNN boxes); flip test not used -> arXiv 2023. https://arxiv.org/abs/2303.07399 ; read: table (extracted; I trust the GFLOPs and AP values, not the "extra data" column layout).
- YOLOX paper (the detector AlphaPose offers as "yolox-x"): YOLOX-X 51.2 AP, 99.1 M params, 281.9 GFLOPs at 640x640; the paper's COCO table gives AP_S 31.2 for YOLOX-X; YOLOX-Tiny 32.8 AP, 5.06 M, 6.45 GFLOPs at 416; YOLOX-Nano 25.3 AP, 0.91 M, 1.08 GFLOPs -> arXiv 2021 (technical report). https://arxiv.org/abs/2107.08430 ; read: table.
- STG-NF README (our scorer): for custom data it provides `gen_data.py` using AlphaPose with `fast_421_res152_256x192.pth` (FastPose-DUC ResNet-152) and "recommends" the YOLOX-X detector; the released ShanghaiTech poses come from a Google Drive link and the README does **not** say which detector/pose config produced them -> ICCV 2023. https://github.com/orhir/STG-NF (raw README) ; read: card (full).
- The only edge number for an AlphaPose derivative I found: a journal paper that rebuilds AlphaPose for Jetson Nano and reports 9.8 FPS for "AlphaPose-opt" (TensorRT, Jetson Nano GPU, 1.5x faster than the original AlphaPose) -> Applied Computer Science-style journal 2023 (DOI prefix `acs`; journal name not on the page). https://doi.org/10.35784/acs-2023-01 ; read: abstract/excerpt. It is a GPU number, not CPU.

### Reported CPU / ARM / Raspberry Pi speed of AlphaPose

**None found.** All official AlphaPose speeds are GPU (TITAN XP, RTX 3090, 2080Ti). The Jetson Nano paper above is GPU. I found no Raspberry Pi or CPU-only AlphaPose benchmark.

### Assessment for us

- Per-frame cost of the config STG-NF's own script implies (YOLOX-X + FastPose-DUC-R152), at N = 5 people: 281.9 + 5 x 15.99 = **362 GFLOPs** `[D]`. Our YOLOv8n-pose at rect 384x640 is 9.2 x 0.6 = **5.5 GFLOPs** `[D]` (9.2 GFLOPs at 640x640 from the Ultralytics table; 245,760 / 409,600 = 0.6). Ratio ~66x `[D]`. Caution: Ultralytics counts 2 operations per multiply-add and OpenMMLab tools usually report multiply-adds (from memory, not verified), so cross-family GFLOPs can be off by up to 2x. Use them only for orders of magnitude.
- A rough Pi 5 time for that AlphaPose config: at the ~40-90 GFLOP/s effective rates seen in section 6 this is several seconds per frame (`[E]`: 362 / 90 = 4 s to 362 / 40 = 9 s). The point is the order of magnitude: it is not an option on the Pi. The detector is 78% of those FLOPs.
- The released ShanghaiTech poses' detector is unverified. If they used YOLOX-X, our 68.5 vs 85.9 gap is largely "nano detector vs 282-GFLOP detector". That is a hypothesis for Experiment 2 (detector recall race), not a finding.

---

## 2. Lightweight top-down alternatives (and what is not top-down)

**Plain language.** "Near-AlphaPose accuracy at edge cost" is already solved on the *pose-network* half: RTMPose-m reaches a higher COCO AP than AlphaPose's FastPose-DUC-R152 with about 12% of its FLOPs. Most of the other names on the list (MoveNet, BlazePose) are single-person models, and LitePose and YOLO-NAS-Pose are not top-down at all (they find all people in one pass). None of the papers below report accuracy for people ~50 px tall; the small-people evidence is indirect (last bullet group).

### Sources

- **RTMPose (t/s/m/l)**: top-down, CSPNeXt backbone + SimCC-style coordinate-classification head. Pose-only compute and COCO val AP (no flip, 256x192, AIC+COCO training): t 0.36 GFLOPs 67.1 AP; s 0.68 / 71.2; m 1.93 / 75.3; l 4.16 / 76.3. Speed: Intel i7-11700, ONNX Runtime, **1 thread**: t 3.20 ms, s 4.48, m 11.06, l 18.85. Snapdragon 865 (ncnn, thread count not stated in the table): FP32 t 15.84, s 25.01, m 49.46; FP16 t 9.02, s 13.89, m 26.44. Pipeline with RTMDet-nano 320x320 detector (one person, skip-frame detection) on the i7: 12.4 / 16.7 / 26.6 / 36.3 ms for t/s/m/l and pipeline AP 64.4 / 68.5 / 73.2 / 74.2. Versus HRNet-w32+UDP 7.7 GFLOPs 75.1 AP 37.7 ms and LiteHRNet-30 0.42 GFLOPs 66.3 AP 22.75 ms on the same CPU -> arXiv 2023. https://arxiv.org/abs/2303.07399 ; read: full for Tables 10-12 and the pipeline text.
- **RTMPose / MMPose model zoo README**: Body8-trained ONNX checkpoints (AP COCO, no flip? the README does not say): t 65.9, s 69.7, m 74.9; person detector RTMDet-nano 320x320 person-only, 0.99 M params, 0.31 GFLOPs, person AP 40.3; larger detector RTMDet-m 640x640 62.5 AP, 24.66 M params, 38.95 GFLOPs; pipeline latency tested with detection every 5 frames -> GitHub docs. https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose (raw README) ; read: card (full).
- **RTMDet** (the detector family): real-time detector with large-kernel depthwise convolutions, tiny/s/m/l/x sizes; RTMDet-tiny 41.1 AP with 4.8 M params -> arXiv 2022. https://arxiv.org/abs/2212.07784 ; read: abstract + one passage.
- **Detector choice matters less on COCO than on our data**: RTMPose-m pipeline AP is 73.2 with RTMDet-nano 320, 75.3 with Faster-RCNN, 75.7 with RTMDet-m 640 (RTMPose paper Table 4, same source as above). COCO keypoint evaluation ignores tiny people (prior lane: "Small objects (area < 32^2) do not contain keypoint annotations"), so this spread says nothing about 50 px people.
- **Lite-HRNet**: shuffle-block high-resolution network; Lite-HRNet-30 at 256x192: 1.8 M params, 0.31 GFLOPs, 67.2 COCO val AP; "+2.6 AP over MobileNetV2 with only 20% of its GFLOPs and parameters" (so MobileNetV2 SimpleBaseline = 67.2 - 2.6 = 64.6 AP `[D]`) -> CVPR 2021. https://arxiv.org/abs/2104.06403 ; read: full for the results text. Key lesson with RTMPose's numbers: LiteHRNet-30 has the same FLOPs as RTMPose-t but 7x the latency on the i7 (22.75 vs 3.20 ms), so FLOPs are a poor predictor of speed; choose by measured latency.
- **MoveNet** (Lightning/Thunder): single-person (one pose per crop; crops follow the previous pose), 17 keypoints. mAP on a COCO subset filtered to one person: Lightning 63.0 FP16 / 57.4 INT8; Thunder 72.0 / 68.9. Latency, 4 CPU threads: Pixel 5 Lightning FP16 60 ms; **Raspberry Pi 4** Lightning FP16 186 ms, INT8 95 ms; Thunder FP16 594 ms, INT8 251 ms -> TensorFlow Blog 2021 (Google). https://blog.tensorflow.org/2021/08/pose-estimation-and-classification-on-edge-devices-with-MoveNet-and-TensorFlow-Lite.html ; read: full (table). RTMPose Table 5 lists MoveNet Lightning 53.6 and Thunder 64.8 AP on COCO-SinglePerson (double inference) vs RTMPose-t 72.1 / s 77.1 / m 82.4 on the same set (same arXiv source).
- **BlazePose**: detector-tracker design for one person on a phone; the detector is a face detector used as a person proxy ("head should always be visible"); Full model 3.5 M params, ~6.9 MFLOPs as printed in the paper; 10 FPS on a Pixel 2 single core (Full); PCK@0.2 on in-house datasets, not COCO AP -> arXiv 2020 (workshop paper; venue from memory, not on page). https://arxiv.org/abs/2006.10204 ; read: full (short paper). Not suitable for CCTV crowds (single person, face assumption).
- **LitePose**: *bottom-up* multi-person model found by neural architecture search; CrowdPose AP / latency: LitePose-S 448x448 2.7 M params 5.0 G MACs, AP 58.3, **Raspberry Pi 4B+ 420 ms** (TVM AutoScheduler); LitePose-XS 256x256 1.7 M, 1.2 G MACs, AP 49.5, Pi 4B+ 109 ms -> CVPR 2022. https://arxiv.org/abs/2205.01271 ; read: excerpt of Table 1.
- **YOLO-NAS-Pose**: single-stage; COCO val AP / T4 GPU latency: N 59.68 / 2.35 ms, S 64.15 / 3.29 (the LearnOpenCV write-up also plots accuracy vs latency on an Intel Xeon 4th-gen CPU; no numbers read); no ARM-CPU number found -> Deci/super-gradients docs. https://github.com/Deci-AI/super-gradients/blob/master/YOLONAS-POSE.md ; read: card.
- **Ultralytics YOLOv8n-pose and its successor**: YOLOv8n-pose 50.4 mAP pose, 3.3 M params, 9.2 GFLOPs, CPU ONNX 131.8 ms (Ultralytics benchmark CPU); YOLO26n-pose (newer docs) 57.2 mAP, 2.9 M, 7.6 GFLOPs, CPU ONNX 40.3 ms (machines may differ between the two doc versions) -> Ultralytics docs. https://docs.ultralytics.com/tasks/pose (YOLOv8 numbers via https://wiki.seeedstudio.com/YOLOv8-TRT-Jetson/) ; read: card. A newer single-stage nano model is a free baseline upgrade to test (small-person recall unknown).

### Accuracy on small people (indirect evidence only)

- SSA Net built a "Tiny Validation" subset of COCO val (person area < 80^2 px, 361 images; the paper does not say whether it uses GT boxes for the top-down row, so treat the top-down vs bottom-up contrast as indicative). AP drops versus full COCO val: bottom-up HrHRNet 46.8 (-19.7), SWAHR 49.7 (-18.2), DEKR 51.8 (-20.5); **top-down coordinate-classification SimCC (HRNet-W48, 256x192) 62.1 (-13.8)**, the best of the baselines; RTMPose uses SimCC-style coordinate classification -> Sensors 2023. https://pmc.ncbi.nlm.nih.gov/articles/PMC10489956/ ; read: full for Table 2.
- Low-resolution pose paper: heatmap HRNet is "surprisingly sensitive" to input resolution; offset-based UDP beats it by +8.8 AP at 128x96; the method (CAL) is built for low-resolution inputs -> arXiv 2021 preprint (venue not on page). https://arxiv.org/abs/2109.09090 ; read: abstract/intro.
- Super-resolution before the pose model helps people with a small initial pixel area and hurts larger ones, so it needs a size threshold -> arXiv 2021. https://arxiv.org/abs/2107.02108 ; read: abstract.
- KeyTrack states that "top-down methods work poorly on small objects" and cites HRNet for that (a claim, no new number) -> CVPR 2020. https://openaccess.thecvf.com/content_CVPR_2020/papers/Snower_15_Keypoints_Is_All_You_Need_CVPR_2020_paper.pdf ; read: full for the sentence.

### Assessment for us

- **Best pose-network choice: RTMPose-s as default, RTMPose-t as the fast fallback, RTMPose-m if budget allows.** Evidence: highest AP per measured millisecond among the options (RTMPose-t 67.1 AP in 3.2 ms vs LiteHRNet-30 66.3 AP in 22.75 ms on the same CPU; MoveNet and BlazePose are single-person and far lower on COCO-SinglePerson); the only measured Pi 5 number I found is RTMPose-s FP16 TFLite/XNNPACK 4 threads **16.7 ms** median (section 4); SimCC-style heads degrade least on small crops among the families with a small-person measurement.
- Because pose quality is worth only ~0.7 AUC in our diagnosis, **RTMPose-t is probably enough**; Experiment 1 tests that directly.
- Caveat 1: no source reports RTMPose accuracy on ~50 px people. A 50 px person in a 256x192 crop is upscaled ~4-5x `[E]`.
- Caveat 2: top-down only helps coverage if the detector finds the person. A pose network cannot fix a missed box. This is why Section 1's detector cost, not the pose network, is the real issue.

---

## 3. Workload-structure tricks that make top-down affordable

**Plain language.** Top-down cost = (detector, once per frame) + (pose network, once per person per frame). You can cut both: run the detector only every K-th frame and move the boxes with a tracker in between; run the pose network on only some of the people each frame; reuse the previous frame's pose to place the next crop; skip or lower the resolution of crops that do not need it. Everything you skip, you fill with P-fill (our causal pose prediction), which EXP-010 showed is positive even when 60-80% of frames are missing.

### Sources

- **Detector-tracker with skipped detection**: BlazePose runs the detector only when the tracker says "no person"; RTMPose's pipeline detects every K frames (default 5) and "in the interval frames the bounding boxes are generated from the last pose estimation results", plus pose NMS and a OneEuro smoothing filter -> arXiv 2020 / arXiv 2023. https://arxiv.org/abs/2006.10204 ; https://arxiv.org/abs/2303.07399 ; read: full (both passages).
- **LightTrack**: top-down tracking where detection runs only at keyframes (fixed interval, or adaptive: re-detect when a target is lost); between keyframes the box comes from the previous keypoints and IDs persist for free; on PoseTrack'18 val, tracking MOTA with ground-truth keyframe detections falls only modestly as the interval grows (CPN101, SC+GCN: about 73.5 at 2-frame, 69.2 at 5-frame, 68.9 at 8-frame; the table extraction is partly garbled, trust the direction not the decimals); with a real detector MOTA is ~10 points lower; a CPN101 pose pass costs 140 ms per candidate in their setup -> CVPR Workshops 2020. https://openaccess.thecvf.com/content_CVPRW_2020/papers/w70/Ning_LightTrack_A_Generic_Framework_for_Online_Top-Down_Human_Pose_Tracking_CVPRW_2020_paper.pdf ; read: full for method + Table 4 (extracted).
- **KeyTrack**: for each person, also run the pose network on the *previous* frame's box pasted into the current frame, then keep the best by OKS (keypoint similarity), which fixes missed or bad boxes and is "faster than a detector ensemble" -> CVPR 2020. https://openaccess.thecvf.com/content_CVPR_2020/papers/Snower_15_Keypoints_Is_All_You_Need_CVPR_2020_paper.pdf ; read: full for the TOKS passage.
- **Adaptive keyframes beat fixed ones**: Detect-or-Track (a scheduler decides per frame whether to detect or track; abstract only here, venue AAAI 2019 per LITCHECK-2026-11-07) https://arxiv.org/abs/1811.05340 ; Yao, Fang, Shen, Wan, Yang (ECCV 2020; title not shown in the fetched text): detection on sparse keyframes for new objects/occlusion/fast motion, trackers in between, adaptive keyframe scheduling, "real-time on CPU" on ImageNet VID -> https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123590154.pdf ; read: abstract/intro.
- **Prior lane (not re-read now)**: LITCHECK-2026-11-07-pose-propagation.md already covers Deep Feature Flow, MARLIN, Glimpse, Reducto, NoScope and Streaming Perception (Kalman forecasting for deadline gaps). Its conclusions apply unchanged: keyframe schedules trade accuracy against keyframe distance; failures are at new or abruptly changing people.
- **Crop-resolution adaptation**: RTMPose-m 384x288 vs 256x192: COCO AP 77.0 vs 75.8 (Body zoo, AIC+COCO), CPU latency 24.78 vs 11.06 ms (2.2x), so a bigger crop costs about 2x for ~1 AP on COCO (MMPose README table; read: card). A public repo proposes size-dependent crop padding (1.15x small/medium, 1.25x large) -> GitHub, unreviewed. https://github.com/ChenJie1029/RTMPose-SpatialEnhancement ; read: card. Resolution as a deadline knob is also the idea of EMERALD (section 5).
- **Batching crops across people**: on ARM Cortex-A CPUs throughput rises with batch size only up to a plateau and for MobileNet-type models "tends to get worse" at larger batches -> arXiv 2021. https://arxiv.org/abs/2107.10047 (read: excerpt). OpenVINO's own docs say pinned "streams" (several requests in parallel, 1-4 threads each) perform "much better ... than batching" on multi-core CPUs -> OpenVINO docs. https://docs.openvino.ai/2026/openvino-workflow/running-inference/optimize-inference/optimizing-low-level-implementation.html ; read: card. On GPU batching helps; on CPU it mostly does not.
- **Cascade (cheap single-stage everywhere + top-down refinement)**: I found no paper for exactly this on CPU. KAPAO (single-stage) + a refinement stage appears in my search only as the search tool's own suggestion; I did not verify it. SHaRPose (AAAI 2024) does coarse-to-fine inside one network (abstract only). Treat the cascade as our own design (an `[E]`).

### Assessment for us (what to run when)

| Stage | Every frame? | Why |
|---|---|---|
| Camera grab, tracker predict | yes | trivially cheap (EXP-004: tracker+flow 2.1 ms on the VM) |
| Detector (expensive, high res) | **every K frames, K = 3 to 5, plus on a trigger** (track lost, new blob, low box-IoU(predicted, detected)) | cost amortised by K; LightTrack/RTMPose/DorT precedent |
| Pose network per person | every frame for few people; **staggered** (each person every 2nd frame) when N is large | cost grows with N; stagger caps work per frame |
| P-fill (our causal predictor) | for every person-frame not posed, and for dropped frames | EXP-010 +5.7 to +8.3 delivered AUC at s = 3 |
| Re-detect only small people | ROI second pass at low rate on the far-field strip | prior lane idea #6; small people are where single-stage fails |

**Important correction to the requested "cascade (c)".** A cascade that runs YOLOv8n-pose everywhere and then re-poses small/low-confidence people with RTMPose can only improve *pose quality of people already detected*. Pose quality is ~0.7 AUC of our gap. The people the single-stage detector misses are not in the candidate list. To attack coverage you need a **second detector pass aimed at small people** (ROI at higher zoom), and RTMPose is optional there. I call that variant (c').

---

## 4. Runtime and compiler options on an ARM CPU

**Plain language.** The same trained network can run 2x faster or slower depending on which "engine" executes it. On a Raspberry Pi 5 (4x Cortex-A76 at 2.4 GHz, 512 KB L2 per core, 2 MB shared L3, LPDDR4X-4267 per the Pi 5 product brief), several independent benchmarks agree that NCNN is the fastest CPU engine for YOLO-style models and ONNX Runtime is 1.6-2.1x slower. 8-bit integer (INT8) math can add another 1.4-2x on this CPU, but the pose output layer is fragile under INT8. Half precision (FP16) is supported by the A76 hardware, but whether the engine uses it is engine-specific.

### Measured Pi 5 numbers `[M]` from sources (all FP32 unless stated; "ms" are per image)

| Model | Setting | PyTorch | ONNX Runtime | OpenVINO | LiteRT/TFLite | MNN | NCNN | Source |
|---|---|---|---|---|---|---|---|---|
| YOLO26n detect, 640x640 | inference only, Ultralytics 8.4.108 | 299.1 | 126.0 | 104.6 | 123.3 | 91.9 | **67.0** | Ultralytics Pi guide https://docs.ultralytics.com/guides/raspberry-pi (card) |
| YOLO11n detect, 640x640 | inference only, Ultralytics 8.3.39 | 405.2 | 168.1 | **81.2** | 317.4 | 112.6 | 88.0 | blog copy of Ultralytics docs https://bubao.github.io/posts/4b1e9c2a.html (card; secondary copy) |
| YOLOv8n-**pose**, 640x512 | per-image incl. pre/post | 209 | 172 | - | - | - | **83** | thermal-image Pi 5 study, Table 2 https://pmc.ncbi.nlm.nih.gov/articles/PMC13517955/ (full for Table 2) |
| YOLOv8n-pose, 320x256 | same | 71 | 45 | - | - | - | 20 | same |
| YOLOv8n detect, 640, 1080p video | 899 frames | 314.1 | 219.9 | - | - | - | 137.9 | community benchmark https://kadirmertabatay.github.io/rpi-yolo-benchmark/benchmarks/yolov8/yolov8n/ (full) |
| RTMPose-**s** (256x192) | LiteRT 2.2 XNNPACK, FP16, 4 threads, median of 3x50 runs, throttle state 0x0 | - | - | - | **16.7** | - | - | model card https://huggingface.co/litert-community/RTMPose-s-LiteRT (card, full) |

Notes: the pose study's per-image time includes pre/post-processing and a single-person thermal dataset, so NMS cost is tiny; its power column shows 7.4-8.6 W for the n model. A community NCNN RTMPose repo claims "FPS 30+" on a Pi 5 without saying which model, input, thread count or whether the detector is included; I treat that as unverified (https://github.com/zhouweigogogo/RTMPose-NCNN, card).

### Other sources

- **ONNX Runtime**: default intra-op threads = number of physical cores; if you set `intra_op_num_threads` explicitly **no thread affinity is applied**; `session.intra_op_thread_affinities` and a global thread pool exist; thread-pool spinning on by default (`session.intra_op.allow_spinning`), and a bounded spin (`spin_duration_us`, `spin_backoff_max`) is available; "it is normally best to not set thread affinity" except when several sessions run in parallel (then separate cores avoid contention) -> ORT docs. https://onnxruntime.ai/docs/performance/tune-performance/threading.html ; read: full.
- **NCNN**: uses OpenMP; threads default to the core count; OpenMP workers **spin-wait up to 200 ms** by default so all cores look busy and "if there are other threads or processes that require more cpu resources, the running speed of the program will drop severely"; remedies: bind cores, fewer threads (<= 4 recommended for non-clang OpenMP), shorter blocktime or `OMP_WAIT_POLICY=PASSIVE`, `OMP_THREAD_LIMIT` -> NCNN docs. https://raw.githubusercontent.com/Tencent/ncnn/master/docs/how-to-use-and-FAQ/openmp-best-practice.md ; read: full. This matters for a multi-stage pipeline on 4 cores (section 5).
- **MNN**: engine with per-operator "pre-inference" cost model that picks the algorithm per backend and caches memory plans -> MLSys 2020. https://arxiv.org/abs/2002.12418 ; read: abstract/intro. Measured on Pi 5 above: second to NCNN.
- **TVM / Ansor**: auto-generated tensor programs; evaluated on a Raspberry Pi 3b+ (4x Cortex-A53), not on A76 -> OSDI 2020 (venue from memory; page shows arXiv). https://arxiv.org/abs/2006.06762 ; read: abstract + setup text. LitePose used TVM AutoScheduler for its Pi 4B+ numbers (above). No Pi 5 TVM pose number found.
- **OpenVINO on ARM**: the CPU plugin officially supports Arm64 Linux; FP32 and FP16 on Arm (FP16 "enables the `armv8.2-a+fp16` extension ... doubled vector capacity"); the **quantized INT8/uINT8 types are listed for Intel x86-64 only** -> OpenVINO docs https://docs.openvino.ai/2025/openvino-workflow/running-inference/inference-devices-and-modes/cpu-device.html ; read: card. So on a Pi, OpenVINO means FP16/FP32, not INT8. One Pi 5 (A76) FP16 classifier port reports ~40 ms vs ~85 ms for ORT FP32 (PR description, not peer reviewed) https://github.com/tphakala/birdnet-go/pull/3547 ; read: card. On YOLO26n above OpenVINO (104.6) is slower than NCNN (67.0) but faster than ORT (126.0); on YOLO11n it is the fastest. So it is model dependent.
- **FP16 on the A76**: ORT has Armv8.2 FP16 GEMM kernels (`HalfGemmKernelNeon`, flag `armv8.2-a+fp16`) but an open issue (microsoft/onnxruntime #28505) reports that FP16 *model* inference can still call the FP32 SGEMM path because of unsupported ops/casts -> https://github.com/microsoft/onnxruntime/issues/28505 ; read: excerpt (user report). I did not find that NCNN FP16 on A76 is measured for pose.
- **INT8 gains on Pi 5 (prior lane, not re-read)**: ORT ResNet-50 QDQ 144.95 -> 79.08 ms (1.83x) on Pi 5 (arXiv 2609.16085, preprint); Arm's YOLO11n-pose INT8 (ExecuTorch+XNNPACK, pose head kept out of INT8) 335.84 -> 234.0 ms (1.44x), mAP50-95 47.37 -> 46.38 (-0.99) (Arm model card). Source: RESEARCH-squeeze-performance-2026-11-07.md §B.
- **INT8 accuracy for pose models**: MoveNet INT8 loses 5.6 mAP (Lightning 63.0 -> 57.4) and 3.1 mAP (Thunder 72.0 -> 68.9) but gains 1.96x / 2.37x latency on the Pi 4 (186 -> 95 ms, 594 -> 251 ms) (TF blog above). For RTMPose, plain post-training INT8 breaks the coordinate-classification (SimCC) output: a journal study reports a fully INT8 PTQ baseline at 0.207 vs 0.729 FP32 accuracy on its own metric (not COCO AP; platform OpenVINO, x86 per the OpenVINO docs, not confirmed) and a mixed-precision method that keeps 88.8% of FP32 accuracy -> International Journal of Internet, Broadcasting and Communication 2025. https://doi.org/10.7236/IJIBC.2025.17.4.328 (koreascience page https://koreascience.kr/article/JAKO202535150427436.page) ; read: abstract + results excerpt. MMDeploy issue threads report large RTMPose accuracy loss after TensorRT conversion (anecdotal) https://github.com/open-mmlab/mmdeploy/issues/2913 ; read: first post only.
- **Pi 5 vs Pi 4**: "2-3x the CPU performance" (Raspberry Pi product brief) https://pip-assets.raspberrypi.com/categories/892-raspberry-pi-5/documents/RP-008348-DS-6-raspberry-pi-5-product-brief.pdf ; read: excerpt. Use this to translate the Pi 4 numbers (MoveNet, LitePose) to Pi 5 as a rough 2-3x `[D]`.
- **Threads vs processes**: community benchmark: NCNN single stream 7.25 FPS (64.5% CPU) vs three parallel streams 9.04 FPS (74.8% CPU), i.e. +25% throughput at 1.9x latency per frame (kadirmertabatay page above); on an 8-core ARM Jetson CPU, 8 single-core PyDTNN instances gave 16.56 img/s vs 14.34 for one 8-thread instance (+15%) -> arXiv 2021. https://arxiv.org/abs/2105.09187 ; read: excerpt. A Pi 5 LLM test found 4 threads the slowest because that workload is memory-bandwidth bound (Arm Learning Path; an LLM, not a CNN, so do not generalise) https://learn.arm.com/learning-paths/cross-platform/rpi5-energy-per-token/clock-and-threads/ ; read: excerpt.

### Assessment for us

- **Default runtime: NCNN** (fastest in 3 of 4 independent Pi 5 comparisons, 1.6-2.1x faster than ORT). Keep ONNX Runtime as the development and parity runtime (it matches the VM pipeline). Test OpenVINO FP16 as the second candidate (model dependent). TFLite/LiteRT is the slowest in the Ultralytics Pi 5 tables for YOLO but has the only measured RTMPose-s Pi 5 number.
- **INT8**: attractive for the detector only if the pose output stays in FP32 (Arm's recipe, -1 mAP); for RTMPose do not use naive INT8. Pose crops are small, so FP16 is the lower-risk lever.
- **Threads**: 4 threads per model is probably not the best use of 4 cores for small models (RTMPose crops, nano detectors); 1-2 threads per stage with stages on separate cores is the thing to measure (section 5).
- **FLOP caveat**: the Pi 5 effective rates I can derive are ~89 GFLOP/s (YOLOv8n-pose NCNN, 7.36 Ultralytics-GFLOPs in 83 ms) and ~41 GFLOP/s (RTMPose-s, 0.68 MMPose-GFLOPs in 16.7 ms) `[D]`; the two use different FLOP conventions, so do not trust a ratio between them.

---

## 5. OS / system orchestration for a multi-stage pipeline on 4 cores

**Plain language.** A Pi 5 has four identical cores that all share one clock speed setting. A multi-stage pipeline (detector, tracker, pose, scorer) can either run stage after stage using all 4 cores for each (simple, but small models cannot use 4 cores well and the cores idle at synchronisation points) or run stages on separate cores at the same time like an assembly line. Throughput improves with the assembly line, per-frame latency does not. The common pitfalls are the ones we already met: real-time priority that escapes CPU quotas, runtime threads that spin and steal cycles, a CPU that slows down when it is hot or when the governor is slow to speed it up.

### Sources

- **Pipelining across cores**: Pipe-it splits a CNN into layer-level pipeline stages mapped to core groups of an ARM big.LITTLE chip (Hi3670) and processes different frames in parallel, because kernel-level parallelism across clusters "fails to scale"; the Pi 5 is homogeneous, so the cross-cluster part does not apply but the stage-pipeline idea does -> IEEE TCAD 2019. https://arxiv.org/abs/1903.05898 (DOI 10.1109/TCAD.2019.2944584) ; read: abstract + intro excerpt.
- **AlphaPose itself is a pipeline**: five stages in separate processes/threads with FIFO queues, "following the principle that each module consumes similar processing time" (AlphaPose paper above; full).
- **Deadline-aware scheduling of several DNNs**: DART (pipeline stages, data parallelism, deadline-monotonic priority, admission control, runtime enforcement; up to 98.5% shorter worst-case response time on x86 Xeon + Nvidia ARM GPU platforms; not CPU-only) -> RTSS 2019. https://intra.ece.ucr.edu/~hyoseung/pdf/rtss19-dart.pdf ; read: abstract. EMERALD (scales input resolution to meet 30 fps deadlines on heterogeneous edge MPSoCs; 11x and 12.3x fewer missed deadlines than two baselines; argues dropping frames loses continuity) -> NSF PAR copy (conference name not on the page). https://par.nsf.gov/servlets/purl/10574058 ; read: abstract + intro. DynaMIX (profile-based accuracy/latency optimisation for several DNN apps) -> arXiv 2023. https://arxiv.org/abs/2302.01568 ; read: abstract. DualRT (batch size, resolution, frame rate and CPU concurrency chosen per task by RL) -> Concurrency and Computation 2025. https://doi.org/10.1002/cpe.70174 ; read: abstract. None is a CPU-only Pi-class pipeline; they support the ideas (admission control, resolution as a knob), not any number.
- **Latest-frame vs queue**: Streaming Perception shows stale outputs must be scored (offline AP 38.0 -> 6.2 under streaming evaluation, scheduling + Kalman forecast recover 17.8) and our EXP-004 uses a latest-frame policy (per LITCHECK-2026-11-07; not re-read). Design rule used below: queue depth 1, drop-oldest, never produce a late result.
- **Real-time priority and cgroups (independent confirmation of EXP-004)**: the Linux kernel's cgroup v2 documentation states "cgroup2 cpu controller doesn't yet support the (bandwidth) control of realtime processes", with a `CONFIG_RT_GROUP_SCHED` caveat -> kernel docs. https://docs.kernel.org/admin-guide/cgroup-v2.html ; read: full for that passage (fetched with curl; I searched the page text). This matches our EXP-004 observation that SCHED_FIFO escaped `cpu.max`.
- **Core isolation**: cgroup v2 `cpuset.cpus.partition` can create an "isolated" partition (no load balancing), and the `isolcpus` boot option pre-isolates CPUs (same kernel doc; full for the passage).
- **Governor, frequency, thermal on the Pi 5**: default governor `ondemand`; thermal management throttles Arm cores progressively between 80 and 85 C and further at 85 C; frequency steps 1500/1000/750/600 MHz under throttling -> Raspberry Pi docs https://github.com/raspberrypi/documentation/blob/master/documentation/asciidoc/computers/raspberry-pi/frequency-management.adoc ; read: card (excerpt). With the Active Cooler the CPU stays at 60-63 C under sustained load; without it, throttling begins about 200 s into a heavy load -> Raspberry Pi news https://www.raspberrypi.com/news/heating-and-cooling-raspberry-pi-5/ ; read: excerpt. All four Pi 5 cores share one cpufreq policy (`policy0`), `ondemand` jumps to `scaling_max_freq` under load; check `vcgencmd get_throttled` = 0x0 -> Arm Learning Path https://learn.arm.com/learning-paths/cross-platform/rpi5-energy-per-token/clock-and-threads/ ; read: excerpt. 2.4 GHz is the most efficient clock on that silicon (Jeff Geerling blog, not peer reviewed; overclock figures not verified) https://www.jeffgeerling.com/blog/2024/important-consideration-about-pi-5-overclocking/ ; read: excerpt.
- **Memory/cache**: Pi 5 has 512 KB L2 per core and a 2 MB shared L3 (product brief). Four models running at once share L3 and memory bandwidth; I found no source measuring cache interference for CNN pipelines on the Pi 5, so this is a risk, not a number.
- **Runtime-level traps**: ORT explicit thread count disables its automatic affinity; NCNN OpenMP spin-waits 200 ms by default (section 4). Both can make a "pinned" stage steal time from its neighbour.

### Assessment for us (settings to use, with confidence)

| Setting | Recommendation | Basis |
|---|---|---|
| Scheduling policy | normal CFS, no SCHED_FIFO; lower the nice value of the scorer/output stage instead | EXP-004 + kernel doc |
| Cores | `cpuset` partitions: detector stage on cores 0-1, pose stage on cores 2-3; capture/tracker/P-fill/scorer small threads allowed on core 3 and a non-isolated OS core is not available on 4 cores, so test both "isolated 2+2" and "2+1+1" | EXP-004 H7' (pinning + isolation halves p99 under a 2-CPU budget); the exact split is an experiment |
| Threads per stage | 2 per stage; NCNN `OMP_WAIT_POLICY=PASSIVE` or short blocktime; ORT `allow_spinning=0` or bounded spin and explicit `intra_op_thread_affinities` | ORT and NCNN docs |
| Governor | `performance` (pin 2.4 GHz) to remove ramp-up delay on a duty cycle of ~40-100% per 41.7 ms period; default `ondemand` is the control | `[E]` hypothesis: no source measures the effect on this workload; Experiment 3 measures it |
| Cooling | Active Cooler mandatory; log `vcgencmd get_throttled` and temperature every run; discard throttled runs | Raspberry Pi news (throttle after ~200 s without it) |
| Frame queues | depth 1, drop-oldest at the camera and between stages; deadline stamp per frame; skip a person's pose if older than one frame period | Streaming Perception; EXP-004 policy |
| Batching across people | none; run crops as parallel 1-thread jobs | section 3 sources |
| Expected benefit of a 2-stage pipeline | +15% to +25% throughput `[D]` from the two parallel-instance measurements (+25% NCNN 1 vs 3 streams; +15% PyDTNN 1 vs 8 instances), not 2x | section 4 sources |

---

## 6. End-to-end feasibility on a Pi 5 (856x480, typically 2-10 people)

**Plain language.** On the target CPU the detector alone already takes about 1.5x the 41.7 ms frame period even with the fastest engine. So whatever we build runs at about 1.5x to 3x too slow every frame and *must* rely on P-fill for the rest, which is the regime where EXP-010 measured its largest gains. The design question is therefore not "can we process every frame" but "which frames, and which people in them, buy the most coverage per millisecond".

### Cost model and every number's provenance

Frame budget at 24 fps: 41.7 ms; four cores = 167 core-ms per frame. All Pi numbers below are for 4x A76 at 2.4 GHz with active cooling, NCNN unless stated.

| Item | Value | Tag | How obtained |
|---|---|---|---|
| YOLOv8n-pose at 640x512, NCNN | 83 ms/img incl. pre/post | `[M]` | PMC13517955 Table 2 |
| YOLOv8n-pose at 640x512, ORT | 172 ms/img | `[M]` | same |
| Our rect size 384x640 vs 640x512 | pixels 245,760 / 327,680 = 0.75 | `[D]` | arithmetic |
| YOLOv8n-pose rect 384x640, NCNN | **~62 ms** | `[D]` | 83 x 0.75, assumes cost linear in pixels |
| YOLOv8n-pose rect 384x640, ORT | **~129 ms** | `[D]` | 172 x 0.75; ratio to our VM ORT infer 29.9 ms = 4.3x; to the VM service p50 34.4 ms = 3.8x; consistent with the prior lane's s ~ 2-4 |
| tracker + flow + scorer on Pi | ~4-6 ms | `[E]` | VM 1.9-3.8 ms (EXP-004 §2) x 2-3 for A76 vs M2 Pro |
| RTMPose-s per crop, 4 threads, FP16 | **16.7 ms** | `[M]` | LiteRT card, Pi 5 |
| RTMPose-t per crop, 4 threads | ~12 ms (range 8-15) | `[D]` | 16.7 x (3.20 / 4.48) from the i7 ratio |
| RTMPose-m per crop, 4 threads | ~41 ms | `[D]` | 16.7 x (11.06 / 4.48) |
| RTMPose crop at 1 thread on 1 core | ~18-30 ms for t | `[E]` | assumes ~50-65% parallel efficiency of 4 threads on a tiny net; unmeasured |
| RTMDet-nano person detector at 320x320 | published 0.31 GFLOPs; i7 pipeline total 12.4 ms with RTMPose-t | `[M]` | RTMPose paper Table 12 |
| RTMDet-nano at 640x384 | ~0.74 GFLOPs (x2.4 pixels) -> **15-40 ms** | `[D]`/`[E]` | pixel scaling of FLOPs; latency range is wide because depthwise-heavy nets run at 30-41 GFLOP/s-class rates; unmeasured |
| RTMDet-nano at 960x544 | ~1.6 GFLOPs (x5.1) -> **35-80 ms** | `[D]`/`[E]` | same method; unmeasured |

Why 320x320 is not an option for our people: scaling 856x480 to 320 wide shrinks a 50 px person to ~19 px `[D]` (320/856 = 0.37), half of what YOLOv8n sees at 640 (~37 px). The published RTMDet-nano pipeline numbers (12.4 ms on the i7) are therefore not our setting.

### Per-frame cost at N = 5 people (mid-case; ranges in brackets)

| Config | Compute per frame on the Pi | s = cost / 41.7 ms | Delivered rate | Tag |
|---|---|---|---|---|
| (a) YOLOv8n-pose 640 rect, NCNN | 62 + ~5 = **~67 ms** (range 60-75) | **~1.6** | ~15 fps | `[D]` from `[M]` |
| (a) same, ORT | 129 + ~5 = **~134 ms** | **~3.2** | ~7.5 fps | `[D]` |
| (b) RTMDet-nano at 640x384 every frame + RTMPose-t per crop, 4 threads, sequential | det 15-40 + 5 x 8-15 + ~6 = **~61-121 ms** (mid ~85) | ~1.5-2.9 | 8-16 fps | `[E]` |
| (b) at N = 2 | 15-40 + 2 x 8-15 + 6 = 37-76 ms | ~0.9-1.8 | | `[E]` |
| (b) at N = 10 | 15-40 + 10 x 8-15 + 6 = 101-196 ms | ~2.4-4.7 | | `[E]` |
| (c) as requested: YOLOv8n-pose + RTMPose-t only for 1-3 small/low-conf people | 67 + 1-3 x 8-15 = **~75-112 ms** | ~1.8-2.7 | | `[E]` |
| (c') coverage cascade: YOLOv8n-pose (NCNN) every 2nd frame (~31 amortised) + far-field ROI person pass at 2x zoom every 4th frame (ROI ~30% of area x 4 = ~1.2x a native pass `[E]` ~75 ms / 4 = ~19) + RTMPose-t for ROI-found people (2 people every frame ~20) + ~6 | **~76 ms** | ~1.8 | | `[E]`, heavy assumptions |
| (d) decoupled design of section 7: detector 960x544 every 3rd frame (35-80 / 3 = 12-27) + RTMPose-t staggered (5 people every 2nd frame = 20-38 with 4 cores; ~10-19 if crop-parallel on 2 cores works) + tracker/P-fill/scorer ~6 | **~38-71 ms** (mid ~50) | ~0.9-1.7 | 14-24 fps processed + P-fill | `[E]`, the most uncertain row |

Reading the table:

1. **Top-down every frame is not cheaper than single-stage on a Pi at N ~ 5.** Break-even is about N = 2-3. Its advantage is structure (rows (c') and (d)), not the raw cost.
2. **(c) as requested cannot raise coverage** (section 3); at best it recovers part of the ~0.7 AUC pose-quality gap, for ~10-45 ms extra.
3. Every row is > 1.0 in s except possibly (d), so **P-fill is part of the operating point** (EXP-010: +5.7 to +8.3 delivered at s = 3, +6.1 to +7.4 at s = 6 on ShanghaiTech-8n).
4. The decisive unknowns are the **detector latency at 640x384 / 960x544** (15-80 ms range) and the **crop-parallel RTMPose cost**. Both are Pi measurements (Experiment 3).
5. Measured-vs-estimated ledger: the only Pi-measured numbers used are YOLOv8n-pose (83/172 ms at 640x512), RTMPose-s (16.7 ms), and the YOLO detector tables. Everything about RTMDet-nano at our resolutions, RTMPose-t on a Pi, crop-parallelism, tracker/scorer time on the Pi, ROI cost and the governor effect is `[E]`.

---

## 7. Proposed "edge top-down pipeline design" (for pre-registration, nothing here is validated)

```
                 camera 24 fps, 856x480
                         |
                 [S0 capture + frame stamp]  core 3 (shared, nice -5, CFS, no SCHED_FIFO)
                         |  latest-frame slot, depth 1, drop-oldest
        ---------------------------------------------------------------------
        |  EVERY K=3 FRAMES (+ trigger: lost track / new motion blob / box-IoU drop)
        v
 [S1 DETECTOR]  cores 0-1 (cpuset isolated, 2 threads, NCNN, OMP PASSIVE)
   person-only nano detector at 960x544 (RTMDet-nano / YOLOX-class, fine-tuned on
   AlphaPose train pseudo-labels if Exp 2 helps); optional: far-field ROI strip at 2x zoom
        |  boxes + scores (async, may be 2-3 frames old when it lands)
        v
 [S2 TRACKER / BOX PROPAGATION]  core 3, microseconds-ms
   Kalman box predict every frame; box for frame t = padded (1.15-1.25x) bbox of the
   previous pose when available; match to detections when S1 lands (IoU/OKS, KeyTrack-style)
        |  per-person crop jobs, each stamped with age since last pose
        v
 [S3 POSE]  cores 2-3 (isolated, 1 thread per crop worker, RTMPose-t FP16, 256x192, NCNN)
   budget rule: at most M=4 crops per frame, chosen by staleness x uncertainty;
   new people and people with low keypoint confidence first; each person >= every 2nd frame
        |  poses with provenance flag {detected-pose, P-fill, held}, age, confidence
        v
 [S4 P-FILL + SCORER]  core 3 (light)
   causal pose prediction (EXP-009/010 P, G=8) for people/frames not posed or dropped;
   STG-NF windows (24 frames, <= 2 missing), provenance reported, no smoothing of detected poses
        |
     anomaly score per frame (held on blank frames as in EXP-010 FILL-HOLD)
```

| Question | Setting |
|---|---|
| What runs every frame | capture, tracker predict, P-fill, scorer, pose on up to M people |
| What runs on keyframes | the detector (every 3rd frame by default; K swept in Exp 2); ROI second pass every 4th keyframe |
| Which cores | detector 0-1; pose 2-3; light stages share core 3; to be compared with 4-thread sequential and 2+1+1 |
| Which runtime | NCNN (FP16 where it works); ORT as the parity/dev runtime; OpenVINO FP16 as the challenger; no INT8 on RTMPose |
| How P-fill fits | fills every missing (person, frame), including people not posed this frame, with causal prediction; K and M are chosen so that predicted share stays <= ~60-70% (EXP-010 limit: no rescue above ~90% drops) |
| OS settings | CFS only; cpuset isolated partitions; no FIFO; governor performance; Active Cooler; spin-wait off; log throttling |
| Fallback when overloaded | lower M first, then raise K, then lower detector resolution (EMERALD-style knob order), never queue |

---

## 8. Ranked experiments (cheapest first; each is pre-registrable)

All use the existing machinery (paired bootstrap over videos, paper and live variants both reported, claims restricted to results that hold in both). Detector/pose swaps are offline first (VM/Mac), Pi only where stated. "Baseline" = current YOLOv8n-pose 640 front end, SHT paper AUC 68.53, live 66.17.

### Experiment 1 (free, Mac/VM, hours): oracle boxes + cheap pose = how much of 85.9 does a cheap pose network recover when detection is perfect?
- Setup: take person boxes from the released AlphaPose poses (box = keypoint extent + 15-25% margin), run RTMPose-t, -s, -m (and YOLOv8n-pose on the same crop for contrast) on those boxes for ShanghaiTech test (and UBnormal if cheap), re-use the released tracks/IDs, score with frozen STG-NF.
- **H1:** RTMPose-s on oracle boxes reaches paper AUC >= 83.0 (within 3 points of 85.94) and RTMPose-t >= 80.0. **Pass:** both hold with the paired-bootstrap lower bound of (AUC - 80) > 0 for t. **Reading:** a pass proves detection is the whole story and a cheap pose network is enough once people are found (consistent with the 96% decomposition); a fail (t below 80) means pose quality matters more than D1 suggested and RTMPose-s/m is needed.
- Needs the Pi: no. Cost: a few hundred thousand crops at ~5 ms each on the VM `[E]`, under an hour per model.

### Experiment 2 (VM/Mac, 1-2 days): which detector at which resolution reaches AlphaPose coverage, and does K-frame detection survive?
- Arms (detector -> RTMPose-t per box -> our tracker -> STG-NF): person-only detectors RTMDet-nano, YOLOX-tiny/nano, YOLOv8n detect at 640 and 960 rect; baselines YOLOv8n-pose at 640/960. Report per-height recall against AlphaPose boxes, anomalous-frame coverage, detector ms and GFLOPs on the VM.
- **H2a:** the best arm with detector VM cost <= 1.3 x YOLOv8n-pose-640 (<= 45 ms incl. pose crops at N = 5) gains >= +2.0 paper AUC over 68.53, lower CI bound > 0. **Fail rule:** if no arm reaches +1.0 point estimate, record "nano-class top-down does not close the coverage gap", which supports the unaffordability of the YOLOX-X-class detector. Then fine-tuning the detector (the prior lane's Exp 2) is the only lever.
- **H2b (keyframe detection with real pose per frame):** detector every K = 3 (960 px) + RTMPose-t on tracker-propagated boxes every frame achieves AUC >= (same detector every frame) - 1.0 at <= 0.6x detector cost, i.e. it does **not** repeat the 1.8-2.7 point loss of EXP-009 P3. **Pass:** both bounds, paired CI. Sweep K in {2, 3, 5}.
- Needs the Pi: no (VM scaled), final cost confirmed in Exp 3.

### Experiment 3 (Pi 5 required, 1-2 days): runtime + threads + governor bake-off, producing the real cost table for section 6
- Measure per-stage latency (p50/p99), temperature and `get_throttled` for: YOLOv8n-pose rect 384x640 in NCNN vs ORT vs OpenVINO FP16; RTMPose-t/s FP16 in NCNN vs XNNPACK vs ORT with 1, 2 and 4 threads and 4 concurrent 1-thread workers; the Exp 2 detector at 640x384 and 960x544; governor `ondemand` vs `performance`; with Active Cooler; 5 repetitions in alternating order (Arm Learning Path advice).
- **H3:** (i) NCNN <= 0.55 x ORT latency for YOLOv8n-pose rect (the sources imply 0.48-0.60); (ii) RTMPose-t aggregated throughput with 4 concurrent 1-thread workers >= 1.3x one 4-thread worker; (iii) `performance` reduces p99 of the per-frame service time by >= 15% vs `ondemand` at the real duty cycle. **Pass rule:** each reported separately; (i) and (ii) decide the design, (iii) decides the OS recommendation. Any throttled run is discarded.
- Replace every `[E]` in section 6 with a measured value.

### Experiment 4 (Pi 5 required, 1-2 days): does assembly-line orchestration beat 4-thread sequential, and does isolation matter on the Pi?
- Arms: (A) sequential, all stages 4 threads; (B) pipeline 2+2 cpuset-isolated; (C) pipeline 2+1+1; (D) B without isolation; (E) B with SCHED_FIFO on the pose stage (negative control, expected to hurt per EXP-004). Camera replay at 24 fps from ShanghaiTech videos, N distribution as in the test set. Feed measured per-frame traces into the EXP-004/010 replay with P-fill.
- **H4:** B beats A by >= +0.15 absolute in fraction of person-frames posed (or kept frames) at equal detector/pose settings, p99 service time not worse by > 10%, and delivered live AUC (with P-fill) >= A + 1.0 with lower CI > 0; E has worse p99 than B (replicating EXP-004 on the Pi). **Fail rule:** if B <= A + 0.05 kept, report "stage pipelining does not help at 4 homogeneous cores" and use A.

### Experiment 5 (VM then Pi, 2-4 days, only if Exp 1-2 pass): the coverage cascade (c') and the full design vs baselines at the Pi's measured cost
- Compare at the Pi trace speed s from Exp 3, all with P-fill, live and paper variants: baseline YOLOv8n-pose NCNN 640; best Exp 2 detector every frame; design (d) with K, M from Exp 2; (c') ROI cascade (far-field strip built from the *train* tracks).
- **H5:** design (d) or (c') exceeds the baseline's delivered AUC at the same s by >= +2.0 (lower CI > 0) on ShanghaiTech and is >= 0 on UBnormal. **Kill rule:** if the best arm is within +1.0 of baseline, the conclusion for the paper is "on a Pi, P-fill + single-stage is as good as top-down structure", which is itself a result.
- Also report AUC with and without predicted-dominated windows (anomaly-hiding check, as in EXP-009 R).

Ordering rationale: Exp 1 and 2 need no Pi and decide whether top-down can add accuracy at all; Exp 3 and 4 need the Pi and turn estimates into measurements; Exp 5 integrates.

---

## 9. Risks (in order)

1. **Biggest: the detector, not the pose network, is the bottleneck, and a nano-class detector may not reach AlphaPose-class coverage.** AlphaPose's detector is 282 GFLOPs (YOLOX-X); we can afford ~1-2 GFLOPs-class models. If Exp 2 fails, top-down gives structure benefits (decoupled rates) but no coverage gain, and the only lever left is fine-tuning the small detector on pseudo-labels (prior lane) plus P-fill.
2. **The released ShanghaiTech poses' detector/pose config is unverified**, so "AlphaPose-class" in our 85.9 may mean something other than YOLOX-X + FastPose-R152.
3. **Pi costs are mostly estimates.** RTMDet-nano at 640x384 / 960x544, RTMPose-t on the Pi, crop-parallel efficiency, tracker/scorer time, ROI cost and the governor effect are `[E]`. Cross-toolchain FLOP counts can differ by 2x.
4. **Top-down cost scales with the number of people** (N up to 10 doubles the pose stage); the M-crops-per-frame budget plus P-fill must cap it, and crowded scenes may break track continuity.
5. **No small-person evidence for RTMPose on ~50 px people**; SimCC-style heads degrade less than heatmaps (-13.8 vs -18 to -20 AP on SSA Net's tiny subset) but still degrade.
6. **INT8 on RTMPose can collapse** (SimCC output); do not rely on it. FP16 may silently run FP32 kernels in ORT.
7. **Thermal throttling** if the Active Cooler is absent; throttling after ~200 s of heavy load is documented for the Pi 5.
8. **Interaction with P-fill:** more real poses reduce the predicted share but also change the pose-noise distribution fed to the frozen STG-NF (0.7 AUC measured for pose quality); anomaly-hiding checks are mandatory.
9. **Licensing (not verified in this lane):** check AlphaPose, MMPose/RTMPose and YOLO-NAS-Pose checkpoint licenses before publication; a Sixfab page lists Ultralytics models as AGPL-3.0.
10. Source quality: several Pi numbers are community or vendor pages (Kadir benchmark, blog copy, LiteRT card, BirdNET PR, a conference-slide PDF I did not use as evidence); the PMC13517955 study and the Ultralytics docs are the strongest Pi 5 anchors.

---

## 10. What I did not verify

- Which exact detector/pose model produced the released ShanghaiTech/UBnormal poses.
- Any AlphaPose or RTMPose number on ~50 px people; any RTMDet-nano latency on a Pi 5; any RTMPose-t Pi 5 latency; NCNN FP16 speed of RTMPose on an A76.
- The "FLOPs" convention of each toolchain (Ultralytics vs MMPose).
- Abstract-only items: Pipe-it, DART, EMERALD, DynaMIX, DualRT, Detect-or-Track, Yao et al. ECCV 2020, Low-resolution HPE (partly), super-resolution-for-HPE, MNN, Ansor (partly), RTMDet, Korean RTMPose INT8 paper (excerpt), SHaRPose, KAPAO, HF/Ultralytics doc pages (cards).
- Preprints (not peer reviewed as far as the pages show): arXiv 2303.07399 (RTMPose; I believe it is a technical report, venue not on the page), 2212.07784, 2107.08430, 2109.09090, 2107.02108, 2006.10204 (workshop paper per memory), 2105.09187, 2302.01568, 1811.05340 (published AAAI 2019 per LITCHECK), 2609.16085 (prior lane).
- The search tool's own "Synthesis" text was not used as evidence; only quoted excerpts and fetched pages.
