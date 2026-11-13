# SETUP — rebuild the environment on a new Mac (Apple chip)

Bundle = git-tracked state of `experiments/` at thesis repo commit `f3cb95c` (2026-11-04), plus our small model
files. Everything big (datasets, venv, public models) is re-downloaded with the commands below. Every sha256 here is
from our own download. Re-check it after downloading.

## 1. Tools and Python env

```bash
brew install uv ffmpeg git            # uv manages Python 3.12; ffmpeg is needed for video decoding
cd experiments
uv sync --extra pose                  # exact versions from uv.lock (torch, onnxruntime, mlflow, ultralytics, lap…)
uv run pytest -q                      # expect 44 passed; data-dependent tests skip until data exists
```

Optional: put the experiment history back into MLflow (`mlflow_runs.csv` is the reference copy of all 58 runs):
`uv run mlflow ui --backend-store-uri sqlite:///mlflow.db` starts a fresh, empty database. New runs go there.

## 2. Data location

All data lives outside git. `experiments/data` is a symlink to the data root:

```bash
mkdir -p ~/thesis-data && ln -s ~/thesis-data experiments/data
export EDGELAB_UCF_CRIME_ROOT=~/thesis-data/ucf-crime     # code default is the old machine's path
```

Old absolute paths `/Users/atruong/...` appear in logs and in `data-tools/ucf-crime/tools/run_rest_and_finalize.sh`.
Edit those to your home path.

## 3. Datasets (download only what the next step needs)

| Dataset | Size | Used by | How |
|---|---|---|---|
| UCF-Crime (official) | 103 GB zip; test set alone ≈ 290 videos | EXP-000, main benchmark | below |
| STG-NF released poses + gt | 1.33 GB | EXP-001 R1 | below |
| ShanghaiTech frames | 7.0 GB (7 parts) | EXP-001 pose extraction | below |
| Hockey Fight | 172 MB | EXP-002 | below |
| RLVS | 3.85 GB (cache 16 GB) | EXP-002 next | below |

**UCF-Crime.** Source page: https://www.crcv.ucf.edu/research/real-world-anomaly-detection-in-surveillance-videos/.
Zip: `https://www.crcv.ucf.edu/data1/chenchen/UCF_Crimes.zip`. We don't download the whole 103 GB zip. We fetch
single members by HTTP Range, and each one is checked against the zip's own CRC32 and size.
- Copy `data-tools/ucf-crime/*` to `~/thesis-data/ucf-crime/`. It contains the zip listing
  (`raw/UCF_Crimes.zip.listing.tsv`), the per-video sha256 lists (`raw/extract_*.tsv`), the member lists, and the
  scripts.
- Test set only. Run these from `~/thesis-data/ucf-crime`:
  ```bash
  tail -n +2 raw/extract_test.tsv | cut -f2 > raw/test_members.txt          # the 290 test video paths
  uv run --project ~/<path>/experiments python tools/extract_members.py raw/test_members.txt UCF_Crimes/Videos raw/extract_test_new.tsv 6
  ```
  Then compare the sha256 column with `raw/extract_test.tsv`. The split files are in the bundle (`UCF_Crimes/Anomaly_Detection_splits/`).
- Train set: `raw/remaining_members.txt`, run through `tools/run_rest_and_finalize.sh`. It resumes where it stopped.
- Annotations: `https://www.crcv.ucf.edu/projects/real-world/Temporal_Anomaly_Annotation_For_Testing_Videos.zip`
  (sha256 `3b11863138b766351ed8893159ca94d8f7465abf1278f2b63c7058463abdfa79`). The bundle already includes them.
- Reference ground truth used by the tests (optional):
  - `gt-ucf.npy` from github.com/tianyu0207/RTFM @cf938f4.
  - `gt_ucf.npy` from github.com/nwpu-zxr/VadCLIP @c41067f.
  - Put them in `reference/{RTFM,VadCLIP}/`. The `make_gt_*.py` scripts are in the bundle.
- Then build the derived files: `uv run python -m edgelab.data.ucf_crime index` and `uv run python -m edgelab.data.ucf_crime lighting`.
  `index_test.csv` and `lighting_test.csv` are already in the bundle.
- Full audit trail: `experiments/EXP-000-data/log.md`.

**STG-NF code + released poses.**
```bash
cd experiments && mkdir -p third_party && git clone https://github.com/orhir/STG-NF third_party/STG-NF
git -C third_party/STG-NF checkout edb5f3220332e160e4d20ce258787d5e2d7e0200
git -C third_party/STG-NF apply ../EXP-001-stgnf-pose-reproduction/stgnf-cpu.patch
uv run gdown 1o9h3Kh6zovW4FIHpNBGnYIRSbGCu-qPt -O ~/thesis-data/stgnf/data.zip
# sha256 b4ecd5b7b7d79ecd09b45a56295b8dd70914154ce7b1f1fb211bef2d5be0bb02; unzip only data/ShanghaiTech/*
ln -s ~/thesis-data/stgnf/data/ShanghaiTech/pose third_party/STG-NF/data/ShanghaiTech/pose
```
The checkpoint `checkpoints/ShanghaiTech_85_9.tar` ships with the repo. R1 check:
`uv run python -m edgelab.pose.evaluate --source released --pose-dir third_party/STG-NF/data/ShanghaiTech/pose/test`.
It should give 85.94.

**ShanghaiTech raw frames.** The official page (https://svip-lab.github.io/dataset/campus_dataset.html) only links a
OneDrive share, which a script can't download. We used the Hugging Face mirror instead:
```bash
cd ~/thesis-data/shanghaitech/hf-123zdm
for p in aa ab ac ad ae af ag; do curl -sSL --retry 5 -C - -o shanghaitech.tar.gz.$p \
  https://huggingface.co/datasets/123zdm/ShanghaiTech/resolve/main/shanghaitech.tar.gz.$p; done
shasum -a 256 -c SHA256SUMS      # SHA256SUMS is in the bundle (data-tools/shanghaitech/)
cat shanghaitech.tar.gz.* | tar xz -C .. --exclude='*test_pixel_mask*'
```
Expected result: 107 test videos, 40,791 frames (856×480). Each video's frame count equals its gt mask length.

**Hockey Fight / RLVS.** Kaggle allows anonymous download:
```bash
curl -L -o ~/thesis-data/hockey-fight/archive.zip https://www.kaggle.com/api/v1/datasets/download/yassershrief/hockey-fight-vidoes
# sha256 a381924f46c5d9e550f8945a748a042fadfb5d70d4459a65d31e006439947a72 → unzip to hockey-fight/raw/
curl -L -o ~/thesis-data/rlvs/archive.zip https://www.kaggle.com/api/v1/datasets/download/mohamedmustafa/real-life-violence-situations-dataset
# sha256 4fb2933dae16cf10ec441d7165fb8e31fc64676493fe8f66c55e359e1c211c74 → unzip to rlvs/raw/
uv run python -c "from pathlib import Path; from edgelab.data.video_clips import build_cache, list_hockey; \
build_cache(list_hockey(Path('data/hockey-fight/raw')), Path('data/cache/hockey-fight'), workers=10)"
# same pattern with list_rlvs → data/cache/rlvs
```

## 4. Models

**Public models.** Downloaded automatically; we don't ship them.
- `yolov8n-pose.pt`: Ultralytics fetches it on first use. Our copy's sha256 is
  `c6fa93dd1ee4a2c18c900a45c1d864a1c6f7aba75d84f91648a30b7fb641d212`. EXP-001 ran it from `artifacts/EXP-001/models/`
  (`cd` there first, so the file lands there).
- STG-NF `ShanghaiTech_85_9.tar`: inside the STG-NF repo.

**Our trained models.** These are in the bundle under `experiments/artifacts/`:
- `EXP-002/hockey-fight-video-seed0/best.pt` and `model_fp32.onnx`: the rebuilt anchor CNN-GRU, 0.940 video
  accuracy (MLflow run `b54dec04…`), with `test_predictions.csv` and `onnx_results.json`.
- `EXP-001/checkpoints/r3_yolov8n_botsort_seed0.pth.tar`: STG-NF retrained on YOLO poses, 67.59 (run `30502140…`).
- `EXP-001/checkpoints/r3_control_alphapose_seed0.pth.tar`: STG-NF retrained on the released poses, 85.31
  (run `3a4d8379…`).
- Each checkpoint has its `.args.json` next to it.

Not shipped, regenerate if needed:
- The other seeds. Each takes about 10–20 min to retrain on MPS, e.g.
  `uv run python -m edgelab.models.train_clips --dataset hockey-fight --protocol video --seed 1`.
- The YOLO pose JSONs (494 MB). Regenerate with `python -m edgelab.pose.extract --frames-root
  ~/thesis-data/shanghaitech/shanghaitech/testing/frames --out-dir ~/thesis-data/stgnf/yolov8n-pose-botsort/test`.
  This takes about 30 min on the test set; see EXP-001 log 22:26 and 23:xx for the train shards.
- The per-frame score files. Rerun `edgelab.pose.evaluate`.

## 5. Optional: Lima VM (Linux subset tests before the Pi)

`limactl create --name edge --vm-type vz --cpus 4 --memory 8 template://ubuntu-lts`. In the VM, install uv, then
`uv sync` and copy the ONNX models in. Set `EDGELAB_MACHINE=vm`. VM timings are comparative only (D8).
