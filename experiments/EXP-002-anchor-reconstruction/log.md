# EXP-002 lab notebook (append-only)

Times are local (2026-11-03). Machine: MacBook Pro M4 Pro 48 GB, macOS (Darwin 25.6.0).
Library versions: torch 2.15.0, onnxruntime 1.31.1, onnx 1.23.2, Python 3.12 (uv env in `experiments/`). MPS is available.

## 2026-11-03 22:10 — Read context, fetch paper
- Read: AGENTS.md, experiments/README.md, DECISIONS.md, the old plan.md, SUPERSEDED-NOTE.md, and the KB paper note.
- Fetched the article and tables:

  ```
  curl -L -A "Mozilla/5.0" https://www.nature.com/articles/s41598-026-44939-x
  curl -L -A "Mozilla/5.0" https://www.nature.com/articles/s41598-026-44939-x/tables/{1,2,3,4,5,8,9,11,12,14}
  ```

  HTML was stripped with a short Python regex script in the scratchpad (not kept; it is not reusable code). The verbatim
  extracts are in plan.md §4.
- New findings while reading Table 9 (added to plan.md §4):
  - The Hockey row TP+FN = 500 and FP+TN = 500 is the entire 1000-video dataset.
  - Accuracy recomputed from the matrix is 97.0 %, not the printed 97.6 %. Recall is 96.4 %, not 97.2 %. Precision is 97.6 %, not 96.8 %.
- The paper claims "official splits" for Hockey Fight. The dataset paper (Bermejo Nievas 2011) defines none (it uses 5-fold CV).

## 22:12 — Data availability
- `thesis-data/` had: `shanghaitech/` (empty), `ucf-crime/` (download in progress by the other agent, no README
  marking it complete), `stgnf/`. **ShanghaiTech and UCF-Crime runs are deferred** (plan §5.5 TODO).
- No aria2c or transmission, so the Academic Torrents mirror of Hockey was not usable. Kaggle allows anonymous download
  through `https://www.kaggle.com/api/v1/datasets/download/<owner>/<slug>` (302 to signed GCS). No credentials were
  needed.

## 22:13 — Download Hockey Fight
- URL: `https://www.kaggle.com/api/v1/datasets/download/yassershrief/hockey-fight-vidoes` (Kaggle metadata: 171,604,215 bytes)
- Saved to `thesis-data/hockey-fight/archive.zip`, 171,604,215 bytes.
- sha256 `a381924f46c5d9e550f8945a748a042fadfb5d70d4459a65d31e006439947a72`
- Unzipped to `thesis-data/hockey-fight/raw/data/`: 1000 `.avi` files, 500 `fi*`, 500 `no*`. Three no-fight files have odd
  names (`no142_`, `no238mpg`, `no65__`), but the `no` prefix and label are fine.
- ffprobe: 360×288, 25/1 fps, 41 frames (fi1, no1, fi500). Table 1 of the paper says 320×240, so this copy (or the
  paper) differs. Logged as a threat to validity.

## 22:13 — Start RLVS download (background)
- URL: `https://www.kaggle.com/api/v1/datasets/download/mohamedmustafa/real-life-violence-situations-dataset`
- Target `thesis-data/rlvs/archive.zip`. The size and sha256 are recorded when it finishes.

## 22:14 — plan.md v2 written (record format, README §10)
- Choices made before any training:
  - clip length **T = 8** cached frames (assumption A9)
  - window stride 2, at most 8 clips per video
  - seeds {0, 1, 2}, where the seed sets both the split and the initialisation
- Why T = 8 rather than the old draft's T = 16: Hockey has 41 frames, and every 3rd frame gives 14. T = 16 cannot be
  formed without padding.

## 22:15 — Model code + tests
- `src/edgelab/models/salman_cnn_gru.py`, `tests/test_models_salman.py`.
- `uv run pytest -q -s tests/test_models_salman.py` → 4 passed. **params = 177,026; MACs/frame = 0.0993 G**. The hand
  estimate was 0.18 M / 0.099 G, so G-model passes.

## 22:16 — Data code + tests
- `src/edgelab/data/video_clips.py`, `src/edgelab/data/splits.py`, `tests/test_data_clips_splits.py`.
- First run: 3 failures, a test bug. With 50 videos per class, the 15 % test share rounds to 7 per class (Python
  banker's rounding of 7.5 gives 8 val), so the test set holds 14 videos, not 15. The assertion was changed to 14–16.
  Split code unchanged. All 17 tests pass.

## 22:17 — Hockey cache
- `build_cache(list_hockey(raw), thesis-data/cache/hockey-fight, workers=10)` took 9.8 s wall, 2.6 GB on disk.
- Frame counts after every-3rd: 14 (993 videos), 17 (7 videos, which are longer).
- Clip items: 4 per video (starts 0, 2, 4, 6) for the 14-frame videos.

## 22:17 — Trainer smoke run
- `uv run python -m edgelab.models.train_clips --dataset hockey-fight --protocol video --seed 99 --max-epochs 1`
- MLflow run `378fdef9014f45c7ab36295b47e6d697` (SMOKE, seed 99, 1 epoch; not a result).
- **32.2 s/epoch** on MPS, standalone: 2,794 train clips + 602 val clips, batch 16.
- After 1 epoch: val acc 0.897, test video acc 0.900. This shows the pipeline learns. It is not a result.

## 22:18 — Main Hockey runs launched
- Three background workers. Worker s runs `--protocol video --seed s`, then `--protocol leaky --seed s`, for s ∈ {0, 1, 2}.
- Logs in `artifacts/EXP-002/logs/hockey-{video,leaky}-seed{s}.log`.
- **Deviation (timing only):** three processes share the MPS GPU, so the logged `epoch_seconds` of these runs are
  inflated by contention. The standalone 32.2 s/epoch above is the reference number for time/epoch. The training math
  is unaffected.

## 22:19 — ONNX code + test
- `src/edgelab/models/onnx_export.py` (TorchScript-based exporter `dynamo=False`, opset 17),
  `tests/test_models_onnx.py`: random-weight parity test passes.

## 22:20 — Monitoring mistake (no effect on results)
- A shell wait matched the smoke run's folder `hockey-fight-video-seed99` with a glob. That folder was renamed to
  `artifacts/EXP-002/SMOKE-hockey-fight-video-seed99` to avoid confusion. Its MLflow run id is unchanged (`378fdef9…`).
  `summarize_runs.py` excludes seed 99.

## 22:22 — RLVS downloaded
- `thesis-data/rlvs/archive.zip`, 3,847,870,446 bytes. This equals the Kaggle `totalBytes`.
- sha256 `4fb2933dae16cf10ec441d7165fb8e31fc64676493fe8f66c55e359e1c211c74`
- The archive holds two nested copies (`raw/Real Life Violence Dataset/` and
  `raw/real life violence situations/Real Life Violence Dataset/`). `diff -rq` shows they are **identical**. We use the first.
- 1000 Violence + 1000 NonViolence videos, 1951 `.mp4` + 49 `.avi`.

## 22:25 — RLVS cache
- `build_cache(list_rlvs(...), thesis-data/cache/rlvs, workers=8)` took 2 min wall, **16 GB**.
- Cached frames per video: min 9, median 42, max 3131. No video is shorter than T = 8.
- Code patch: `train_clips.run` memory-maps caches larger than 6 GB (`IN_MEMORY_LIMIT_BYTES`). The data is identical, and
  only the load path changes.

## 22:26 — Scheduling change
- The 3 parent shells (which would have started the leaky runs after the video runs) were killed with `kill -9`. Their
  Python children kept running.
- The 3 leaky runs were launched directly in parallel, 6 processes in total.

## 22:33 — Contention is worse with 6 processes; leaky runs paused
- With 6 processes on MPS, epochs took 150–270 s each. Aggregate throughput was ≈ 0.03 epochs/s, against ≈ 0.042 with 3
  processes and 0.031 standalone (32.2 s/epoch).
- The 3 leaky processes (pids 41327–41329) were paused with `kill -STOP` after their epoch 1. They resume with
  `kill -CONT` when the video runs finish.
- Effect: the `epoch_seconds` for the epoch spanning the pause is inflated by the pause. Model math is unaffected.

## 22:51 — Resumed leaky seed 0 (pid 41327) early
- video seed 0 finished (early stop, 16 epochs), so only 2 processes were running. Leaky seeds 1 and 2 resume when the
  video runs end.

## 22:57 — Correction to the 22:51 entry
- pid 41327 was **leaky seed 2**, not seed 0. It is the one that has been running since 22:51. Seeds 0 and 1 resumed at
  22:57:13, when all video runs had finished.

## 22:57 — Hockey video-level runs finished (all three early-stopped)
- seed 0 `b54dec0434564422bbea2ea435ca3b14` (16 epochs), seed 1 `c50ba48dd8c141ec95f9308563fcfa65` (37 epochs),
  seed 2 `74e1f195598841acbe4070194b745b22` (33 epochs). The numbers are in results.md.

## 22:57 — RLVS video-level seed 0 launched ("if time permits" item, one seed only)
- `uv run python -m edgelab.models.train_clips --dataset rlvs --protocol video --seed 0` → `logs/rlvs-video-seed0.log`.
  It runs alongside the 3 leaky Hockey runs (4 processes).

## 23:04 — RLVS run stopped (time budget), deferred
- Epoch 0 of `rlvs-video-seed0`: train loss 0.584, val loss 0.574, val acc 0.685. It took **393 s** with 4 processes
  sharing MPS. It has 10,8xx train clips, about 3.9× Hockey.
- Estimate for a standalone run: ≈ 32.2 s × 3.9 ≈ 2 min/epoch, so ≈ 0.5–1.7 h to early stop or 50 epochs. That does not
  fit this session's ~2 h budget next to the Hockey leaky runs.
- Killed with `pkill`. MLflow run `2ef0ad00fabf4e53a52bb0799e82b7c7` was set to KILLED. It is **not a result**.
- The split file `splits/rlvs_video_seed0.json` and the 16 GB cache are kept for the rerun.
- To rerun: `uv run python -m edgelab.models.train_clips --dataset rlvs --protocol video --seed 0` (alone on the GPU).

## 23:06 — Correction to the 22:57 run-id mapping (MLflow params are authoritative)
- video seed 1 = `74e1f195598841acbe4070194b745b22` (33 epochs). video seed 2 = `c50ba48dd8c141ec95f9308563fcfa65` (37 epochs).
  The 22:57 entry had these two swapped.
- Tables in results.md are generated by `uv run python -m edgelab.models.summarize_runs --dataset hockey-fight`.

## 23:16–23:39 — Hockey leaky runs finished
- seed 0 `c686ecf138ff43288ff32e2b7fee31b3` ran 50 epochs (no early stop). seed 1 `9a97d6fd6476405b9663ec2c20677729`
  early-stopped at 17 epochs. seed 2 `7a2b09ea44ff4d6dbf6f31d0df347ecd` ran 50 epochs.
- Seed 1's `mean_epoch_seconds` (173 s) includes the 22:33–22:57 SIGSTOP pause.

## 23:30 — POST-HOC analysis (not pre-registered): whole-dataset evaluation
- `src/edgelab/models/eval_whole_dataset.py` scores each video-level checkpoint on all 1000 Hockey videos.
- Motivation: the Table 9 Hockey row sums to the whole dataset.
- MLflow runs: seed 0 `19ec6a360cf946f1a5edaad5cefd377d`, seed 1 `4d94c338714c4b2a9796179f77762a03`,
  seed 2 `3bfd317478394d46afa68f395b859138`.

## 23:39 — ONNX export, parity, latency
- `uv run python -m edgelab.models.onnx_export --run hockey-fight-video-seed0` → MLflow `9a58548ac5ad414282c5b91b7764c0c0`.
  The file is `artifacts/EXP-002/hockey-fight-video-seed0/model_fp32.onnx` (712,828 bytes).
- torch.onnx printed a deprecation warning for the TorchScript exporter (`dynamo=False`). The export works.
- **Threat:** during the latency measurement, other agents' jobs were using ≈ 12 of 14 cores:
  - EXP-001 `edgelab.pose.extract` (4 processes)
  - STG-NF `train_eval.py`
  - the UCF-Crime extractor

  The latencies are therefore taken **under heavy background CPU load**. They are an upper bound, not a clean number.
  They should be re-measured on an idle machine.
- The smoke-run timing (22:17, 32.2 s/epoch) was taken before those jobs started (their elapsed time at 23:39 was
  ≈ 45–47 min).

## 23:40 — Data status for §5.5
- `thesis-data/shanghaitech/shanghaitech/{training,testing}` now exists, downloaded by the EXP-001 agent.
- The UCF-Crime README says `STATUS: PARTIAL`: the test set is complete, and the train set is still extracting.
- The ShanghaiTech / UCF-Crime runs were not started: the time budget is spent, and UCF train data is incomplete.
  Left as TODO.

## 23:45 — Wrap-up
- Added `config.yaml` and results.md. WALKTHROUGH.md gained the summarize/post-hoc row.
- Renamed the smoke split to `splits/SMOKE-hockey-fight_video_seed99.json`.
- `uv run pytest -q` → 44 passed (whole suite, including other lanes' tests).
- Not committed, by instruction.
