# Start Here: Your Step-by-Step Reading & Learning Guide

If your context feels overloaded, **close every other tab and start here**. You do not need to read 50 files. Follow this exact 5-step order—each step builds naturally on the one before it.

---

## Step 1: The 4 Core Concepts You Need to Know First (10 mins)

Before reading any experiment numbers, make sure you have a mental picture of these 4 building blocks:

1. **The 3-Stage Pipeline (`Pose -> Track -> Score`):**
   - **Stage 1 — Pose Detector (`YOLOv8n-pose` or `YOLO26n-pose`):** Looks at **1 video frame** and outputs a bounding box + 17 body joints `(x, y, confidence)` for each person.
   - **Stage 2 — Multi-Object Tracker (`BoT-SORT` or `TrackTrack`):** Links the boxes across frames so we know *"Person #7 in frame 1 is the same Person #7 in frame 24."*
   - **Stage 3 — Skeleton Anomaly Scorer (`STG-NF`, `LoRaD`, or `TrajVAD-P`):** Takes a **24-frame clip (1 second)** of Person #7's skeleton moving and outputs a **normality score** (using a Normalizing Flow: high probability = normal walking; low probability = abnormal action like fighting, falling, or skating).
2. **Offline vs. Causal ("Real-Time") Scoring:**
   - **Offline (`S0` in papers):** Cheats by looking into the *future* (smooths frame $t$ using future frames $t+1 \dots t+24$ and fills empty frames using the best score from the whole future test set).
   - **Strict Causal (`S4` on a real camera):** Can only use frames $\le t$, and only delivers a score *after* the Raspberry Pi 5 finishes computing it (`decode + detect + track + score + queue_wait`).
3. **Why Queue Collapse Happens on a Raspberry Pi 5:**
   - A 24 FPS camera sends a new frame every **`41.67 ms`**.
   - If your pipeline takes **`60 ms/frame`** on the Pi 5 (`16.6 FPS`), every frame falls `18.3 ms` further behind. After 2 minutes, alerts are **47 seconds late** (queue collapse!).
4. **Why Pure Skeletons Miss Bikes/Carts & Distant People:**
   - **Small-person blindness:** Stock YOLO-pose was trained on COCO where small people without keypoint labels were deleted (treated as background). Our **`ML0`** fix keeps their bounding boxes during training.
   - **Normalization blindness:** `STG-NF` subtracts each window's average position $(x, y)$ and divides by height—which erases *how fast* a person is moving across the screen (making a smooth cyclist/skater look like someone standing still!).

---

## Step 2: Visual Exploration First (15 mins — Interactive!)

Before reading long tables, **see the data and failures with your own eyes** in the Streamlit app (`http://localhost:8501`, or run `./run_app.sh`):

1. Open the **`Dataset Explorer`** tab:
   - Pick a **ShanghaiTech** clip (e.g., `01_0014` or `01_0140`) and click **"Jump to Event Start"**.
   - Toggle between **`stock YOLOv8n-pose`** and **`ML0 NCNN`**: watch how `stock` completely misses `<48 px` background people while `ML0` + dashed two-stage recovery boxes (`0.05 <= conf < 0.20`) track them cleanly.
2. Open the **`Failure Case Inspector`** tab:
   - Switch between the 4 failure filters:
     1. *Small-Person Misses* (`stock` fails, `ML0` succeeds)
     2. *Track Fragmentation* (`default tracker` drops the track, `2-stage BoT-SORT` bridges it)
     3. *Scorer-Limited Misses* (person *is* tracked, but `STG-NF` fails because of normalization/window limits)
     4. *Pi 5 Queue Collapse* (`Fixed-K1` falls 46.8s behind vs. `Cache-C1+Guard` staying under `405 ms`)

---

## Step 3: Read Our Project Checkpoints in This Exact Order (30 mins)

Read these 4 markdown files in the repo root and `experiments/` from top to bottom:

1. **[`PAPER_POSITIONING_AND_CHECKPOINT.md`](./PAPER_POSITIONING_AND_CHECKPOINT.md)** *(5 mins)*
   - The high-level summary of why published papers (`Huang et al. 2026`, `LoRaD 2026`) fail on a real Pi 5 and how our contributions fit together into a Q2/Q1 paper.
2. **[`experiments/Q2-P1-Pi5_huang_repro/report.md`](./experiments/Q2-P1-Pi5_huang_repro/report.md)** *(5 mins)*
   - Our direct reproduction of `Huang et al. (2026)` on your **physical Raspberry Pi 5** (`6.14 FPS` @ 640p PyTorch = queue collapse; `36.6 FPS` @ 320p NCNN = `57.3%` AUC blind to small people; vs. our `ML0 + Cache-C1+Guard` hitting `28.45 FPS` @ 640p!).
3. **[`experiments/Q2-P1-Pi5_suite/report.md`](./experiments/Q2-P1-Pi5_suite/report.md)** *(10 mins)*
   - All 86 stage-timing cells and 4 sustained 2,880-frame live runs on your physical Pi 5 (`E02` stage breakdown, `E04` why NCNN FP16 beats ORT INT8 in energy on Cortex-A76, and `E05` sustained `74.19%` live Pi 5 AUC at `0.168 J/frame`).
4. **[`Q2_P1_SYNTHESIS_FOR_PAPER_WRITING.md`](./Q2_P1_SYNTHESIS_FOR_PAPER_WRITING.md)** & **`~/Desktop/DRAFT_PAPER_Q2_Edge_Skeleton_VAD.md`** *(10 mins)*
   - The complete evidence dossier (`E00` through `E05`) and the full Q2 manuscript draft.

---

## Step 4: Deep-Dive into Individual Experiments (Only When Needed)

Whenever you want to check a specific proof or table, open its folder:
- `experiments/Q2-P1-E00_audit/report.md` — Proof of offline look-ahead leaks & bit-exact overlap pose caching.
- `experiments/Q2-P1-E01_quality/report.md` — Proof of `ML0` small-person recall (`0.4% -> 37.0%`) & 36-cell two-stage `BoT-SORT` sweep (`+3.04` AUC).
- `experiments/Q2-P1-E02_fixed_baseline/report.md` — Fixed burst vs. stride schedules & queueing model.
- `experiments/Q2-P1-E03_ablation/report.md` — Why `Cache-C1` (`+2.41 pt`, `-28.5%` CPU) beat `Motion-Only`, `Adaptive-Only`, and `Joint-Controller`.
- `experiments/Q2-P1-E04_optimizations/report.md` — Why H.264 Motion Vectors (`+MV`) and static `INT8` were rejected.
- `experiments/Q2-P1-E05_confirmation/report.md` — Held-out confirmation, non-oracle `C1-ReTrack`, short-event/new-entry subgroups, and sustained runs.
- `experiments/Q2-P1-E06_lorad_causal_flow/report.md` — Reproduction of `LoRaD` (JRTIP 2026) showing why streaming prefix normalization drops AUC by `-4.0` to `-5.3` points.
- `experiments/Q2-P1-E07_yolo26_tracktrack_trajvadp/report.md` — `YOLO26-pose` + `TrackTrack` (CVPR 2025) + `TrajVAD-P` (ECCV 2026) factorial & Pi 5 benchmark.
- `experiments/Q2-P1-E08_lorad_gaps_proposals/report.md` *(see [`LORAD_GAPS_EXPERIMENT_PLAN.md`](./LORAD_GAPS_EXPERIMENT_PLAN.md))* — Multi-Scale Causal Dilated Flow (`MS-CTFlow`), Reverse-Consistent Margin Flow, and Interaction-Aware 20-Node Graph (`IAC-Flow`).

---

## Step 5: Live Observability in MLflow (`http://127.0.0.1:5000`)

Open **`http://127.0.0.1:5000`** in your browser:
- Click **`Models`** (top bar) to see all registered models (`stock`, `ML0`, `MLP`, `STG-NF`, `LoRaD`, `TrajVAD-P`) with SHA256 hashes and model cards.
- Click **`Experiments`** $\to$ `Pi5-Huang-Repro-and-Comparison`, `Pi5-E02-Stage-Profile`, `Pi5-E04-Optimizations`, `Pi5-E05-Sustained-Operation`, `Q2-P1-E00..E08`, and click the **`Traces`** tab to inspect per-stage latency and Pi 5 PMIC power/temperature spans.
