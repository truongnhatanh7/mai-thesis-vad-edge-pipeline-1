# EXP-004 log (append-only)

### 2026-11-04 23:26 — VM lane: create VM `edge-os`
- Read D16 (DECISIONS.md), README §4, SETUP.md §5. VMs `default` and `ebpf-lb` (owner's) left Stopped and untouched.
- Host: `limactl 2.0.3` (/opt/homebrew/bin/limactl). `mkdir -p /Users/nanh/thesis-data/vm-exchange/models`.
- `template://ubuntu-lts` = `_images/ubuntu-24.04` + `_default/mounts`; the latter mounts all of `~` read-only, which I do not want. So I used the same image template via `base: template:_images/ubuntu-24.04`, with explicit mounts. YAML saved at `/Users/nanh/thesis-data/vm-exchange/edge-os.yaml` (copy in `vm-setup.md`):
  ```yaml
  base: template:_images/ubuntu-24.04
  vmType: vz
  cpus: 4
  memory: 8GiB
  disk: 40GiB
  mountType: virtiofs
  mounts:
    - location: /Users/nanh/Documents/RMIT_MscAI/thesis/experiments
      mountPoint: /Users/nanh/Documents/RMIT_MscAI/thesis/experiments
      writable: false
    - location: /Users/nanh/thesis-data/vm-exchange
      mountPoint: /Users/nanh/thesis-data/vm-exchange
      writable: true
  ```
- Command: `limactl create --name edge-os --tty=false /Users/nanh/thesis-data/vm-exchange/edge-os.yaml` (took ~5 min, mostly the 620 MB Ubuntu 24.04 arm64 image download). Result: `edge-os Stopped 4 CPUs 8GiB 40GiB`.

### 2026-11-04 23:31 — Host env versions
- Command: `cd experiments && uv run python -c "import sys,torch,onnxruntime,cv2,ultralytics,numpy,onnx; ..."` (/tmp/uvsync.log ended with `uv exit 0`).
- Host (macOS arm64): python 3.12.11, torch 2.15.0, onnxruntime 1.31.1, opencv(cv2) 5.0.0, ultralytics 8.4.201, numpy 2.5.4, onnx 1.23.2.

### 2026-11-04 23:33 — VM started and checked
- `limactl start edge-os --tty=false` → READY (~1 min).
- `limactl shell edge-os -- ...`: kernel `6.8.0-90-generic aarch64` (Ubuntu 24.04), 4 CPUs (`lscpu`: Vendor Apple, 1 cluster of 4, no hybrid info), 7.7 GiB RAM, `/` 38 GB (36 GB free), user `nanh`, `$HOME=/home/nanh.linux`, passwordless sudo OK.
- Mounts (`mount`): `mount0 ... experiments type virtiofs (ro)`, `mount1 /Users/nanh/thesis-data/vm-exchange type virtiofs (rw)`. `touch experiments/x` → "Read-only file system" (good); `touch vm-exchange/.t` works. Host `~` is NOT mounted (`ls /Users/nanh` in VM shows only `Documents`, `thesis-data` parent dirs).
- cgroup: `stat -fc %T /sys/fs/cgroup` = `cgroup2fs`; root `cgroup.controllers` = `cpuset cpu io memory hugetlb pids rdma misc`.

### 2026-11-04 23:35 — Packages in the VM
- `sudo apt-get install -y build-essential curl git ffmpeg stress-ng linux-tools-$(uname -r) linux-tools-common bpftrace bpfcc-tools sysstat libgl1 libglib2.0-0t64` → OK. `perf` = `/usr/bin/perf` (6.8.12), `bpftrace` v0.20.2, `ffmpeg` 6.1.1-3ubuntu5.
- `curl -LsSf https://astral.sh/uv/install.sh | sh` → uv 0.13.7 (aarch64-linux) in `~/.local/bin`.

### 2026-11-04 23:41 — VM-local venv (`~/edgelab-vm`)
- Chosen approach: copy `experiments/pyproject.toml` + `uv.lock` to `~/edgelab-vm` (VM disk), then `uv sync --frozen --extra pose --no-install-project` (took 6m42s; venv 6.3 GB). The project itself is NOT installed (the mount is read-only and hatchling would write `src/*.egg-info`); `import edgelab` works via `PYTHONPATH=/Users/nanh/Documents/RMIT_MscAI/thesis/experiments/src`.
- Versions (VM, Linux aarch64, system python 3.12.3): torch 2.15.0+cu132 (`cuda.is_available()=False`; the lockfile resolves the CUDA wheel on Linux, so the venv carries unused `nvidia-*` packages but runs CPU-only), onnxruntime 1.31.1 (providers: `AzureExecutionProvider`, `CPUExecutionProvider`), cv2 5.0.0, ultralytics 8.4.201, numpy 2.5.4, onnx 1.23.2. `edgelab` imported from the mount.
- Host (macOS, py 3.12.11): torch 2.15.0, onnxruntime 1.31.1, cv2 5.0.0, ultralytics 8.4.201, numpy 2.5.4, onnx 1.23.2. → identical except python patch (3.12.3 vs 3.12.11) and torch build (+cu132 vs macOS CPU/MPS build).
- Note: `torch.get_num_threads()` printed 1 right after import in the VM (before any set; not investigated further — always set threads explicitly).

### 2026-11-04 23:42 — ONNX export (host)
- Weights `experiments/artifacts/EXP-001/models/yolov8n-pose.pt` (sha256 `c6fa93dd1ee4a2c18c900a45c1d864a1c6f7aba75d84f91648a30b7fb641d212`, 6.8 MB) copied to a temp dir; `YOLO('yolov8n-pose.pt').export(format='onnx', imgsz=640, opset=17, dynamic=False, simplify=False, half=False)` via `uv run` on the host.
- Copied to `/Users/nanh/thesis-data/vm-exchange/models/yolov8n-pose-640.onnx` (13,484,674 bytes), sha256 `0fb7300839cab09ba18d98bbc2fb17d5b82a32738ae1ba31a5b51a3537685ddf` (also in `yolov8n-pose-640.onnx.sha256`).

### 2026-11-04 23:4x — Knob (a)(b)(c): threads, affinity, SCHED_FIFO, nice
- Script `vm-exchange/scripts/knobcheck.py`, run in VM as user (euid 501) and under `sudo` (euid 0):
  - (a) `torch.set_num_threads(2)` → `get_num_threads()` = 2. ORT `SessionOptions.intra_op_num_threads` = 1 / 4 with `add_session_config_entry("session.intra_op.allow_spinning","0")` → sessions run, config reads back "0". Works as user.
  - (b) `os.sched_setaffinity(0,{0,1})` → [0,1]; `{2}` → [2]; works as user. `taskset -c 3 ffmpeg ...` also works.
  - (c) SCHED_FIFO prio 10 via `os.sched_setscheduler`: as user `PermissionError(1, Operation not permitted)`; as root OK (policy 1). `chrt -f 10`: as user "failed to set pid 0's policy: Operation not permitted" (rc=1); `sudo chrt -f 10 ...` shows SCHED_FIFO prio 10. `nice +5` works as user; negative nice (-10) as user → PermissionError; as root works (-5 after the +5). `/proc/sys/kernel/sched_rt_runtime_us` = 950000 (RT throttling 95%, default).

### 2026-11-04 23:4x — Knob (d): cgroup v2 cpuset + cpu.max
- Script `vm-exchange/scripts/cgcheck.sh` (sudo; creates `/sys/fs/cgroup/edgetest`, enables `+cpuset +cpu` in root subtree_control, 4 busy `sh` loops for 5 s, measures `usage_usec` delta in `cpu.stat`):

  | cpuset.cpus | cpu.max | CPU-s used in 5.0 s | ≈ CPUs |
  |---|---|---|---|
  | 0-3 | max 100000 | 19.98 | 4.00 |
  | 0-1 | max 100000 | 10.00 | 2.00 |
  | 0 | max 100000 | 5.01 | 1.00 |
  | 0-3 | 100000 100000 | 5.01 | 1.00 (throttled) |
  | 0-3 | 50000 100000 | 2.56 | 0.51 (throttled) |
  | 0-3 | 200000 100000 | 10.07 | 2.01 (throttled) |
  (`nr_throttled` in cpu.stat is cumulative: 50 → 101 → 151 across the last three rows.)
- `sudo systemd-run --scope -q -p CPUQuota=50% -p AllowedCPUs=0-1 ...` → scope under `/system.slice/run-*.scope`, `cpu.max`=`50000 100000`, `cpuset.cpus.effective`=`0-1`; 2 busy loops for 4 s used 2.05 CPU-s (expected 0.5×4 = 2.0). Works (needs sudo; user-level scope not tried).

### 2026-11-04 23:4x — Knobs (e)(f)(g)
- (e) cpufreq: `/sys/devices/system/cpu/cpu0/cpufreq` does not exist; `/sys/devices/system/cpu/cpufreq` is empty; no `scaling_governor`. `cpupower` binary exists but there is nothing to set. `/sys/class/thermal/cooling_device0/type` = `Processor` (no thermal zones inspected). → absent, as expected.
- (f) perf: `perf_event_paranoid` = 4 → as user perf stat refuses; with `sudo perf stat -e task-clock,context-switches,cpu-migrations,page-faults` works (software events). `cycles`, `instructions`, `cache-misses` = `<not supported>` (no PMU exposed by the vz guest). bpftrace v0.20.2: `sudo bpftrace -e 'tracepoint:sched:sched_switch { @[comm]=count(); } interval:s:2 { exit(); }'` works (printed per-comm counts, e.g. kworker, migration/N, rcu_preempt). bcc: `sudo runqlat-bpfcc 2 1` works (run-queue latency histogram, most samples 2–15 µs on an idle VM). All need root.
- (g) ffmpeg co-load: `ffmpeg -f lavfi -i testsrc2=size=1280x720:rate=30 -t 10 -c:v libx264 -preset veryfast -f null -` → 300 frames encoded in 1.19 s wall at 280% CPU (4 threads). Pinned: `taskset -c 3 ... -threads 1` → 99% CPU, 2.22 s wall. Runs and pins fine (it runs faster than real time, so use `-re` or a loop to hold a constant load; or `-stream_loop`/long `-t`).

### 2026-11-04 23:44 — First timing attempt (ABORTED, contaminated)
- Scripts `scripts/bench_ort.py` (20 warm-up + 200 timed, ORT CPU, random 1x3x640x640 float32, seed 0, inter_op=1) and `scripts/hostload.sh` (samples host `top -l` CPU every ~4 s). Host load at the time: `uptime` load avg 6–8; top said ~35–55% user + 10–15% sys (idle 35–60%); `ps` top process: a node process at ~310% CPU, `sysmond` 96%. Host busy ≈ 4–6 cores of 10 → contaminated.
- Partial (from a run that got interrupted by a tool/connection failure; **contaminated**, not to be used): intra=1 spin on p50 131.1 / p95 140.5 / p99 154.9 ms; intra=1 spin off p50 127.5 / 137.7 / 145.0; intra=2 spin on 75.5 / 99.7 / 157.0; intra=2 spin off 78.6 / 182.8 / 205.3. The rest of that run was lost (stall); the result files were deleted and the run restarted.
- A second run (`bench_all.sh run1`, started 00:02:21) was also lost: the VM rebooted during the connection stall (guest uptime 16 min at 00:21), so no timings were produced (`run1-bench.csv` holds only its header). Host load average at that start: 11.06.

### 2026-11-05 00:58 — Lead: VM lane stalled twice; lead takes over. Host noise audit
- VM lane ended twice with "connection stalled". On disk: knob checks (a)–(g) above ✔, ONNX model ✔,
  `results/run2-bench.csv` = header only (run2 lost too), `run2-hostload.csv` mean host 44% user + 13% sys (n = 4).
- Host at 00:58: load avg 7.21 / 8.22 / 9.65, uptime 74 days. Top CPU: short-lived `/opt/homebrew/.../node`
  processes re-spawned every few seconds at ≈ 200% (owner tooling, not ours), `sysmond` 74%, `mediaanalysisd` 66%,
  XProtect 55%, two `curl` (ShanghaiTech download). → **≈ 3–5 of 10 host cores busy with non-experiment work.**
- Consequence for EXP-004 timing protocol (to pre-register): (1) never trust absolute VM ms; compare configs
  **interleaved in randomised order within the same time window** (ordering-trap, Duplyakin ATC'23), many short
  repetitions, paired differences; (2) log host load per repetition and exclude repetitions above a pre-set host-load
  limit; (3) ask the owner to reboot / quit background apps before the final timing block.

### 2026-11-05 07:40 — Code lane 2: modules written (no timing matrix run)
- New: `src/edgelab/pipeline/{deadline_sim,ort_pose,stgnf_live,os_config,live_timing,timing_subset,deadline_replay}.py`, `src/edgelab/pose/delivered.py`, `EXP-004-os-deadlines/scripts/coload.sh`, `WALKTHROUGH.md`. Changed (backward compatible): `pose/replay.py` (`reindex=False` option, `FRAME_MAPS_FILE`), `preprocess/zerodcepp.py` (`ZeroDCEppOnnx(spin=True)` option).
- Tests: `tests/test_pipeline_{deadline_sim,ort_pose,live_timing,deadline_replay}.py`, `tests/test_pose_delivered.py`. Whole suite: `uv run --extra pose pytest -q` → 99 passed, 10 skipped (all skips are old: UCF-Crime data and ffmpeg missing on host), 0 failed. Live STG-NF scores equal the offline `score_segments` scores (rtol 1e-4) on a synthetic 30-frame track.
- Import in the VM: no install needed; `PYTHONPATH=/Users/nanh/Documents/RMIT_MscAI/thesis/experiments/src ~/edgelab-vm/.venv/bin/python -m edgelab.pipeline.live_timing ...` works (as in log 23:41). Under `sudo` pass it with `sudo env PYTHONPATH=... python`.
- Zero-DCE++ ONNX exported on the host with `ZeroDCEpp().export_onnx` → `vm-exchange/models/zerodcepp.onnx` (opset 17, dynamic H, W; torch-vs-ORT parity is tested in `test_preprocess_enhance.py`).

### 2026-11-05 07:41 — Timing subset (seed 0) and frame copy
- `python -m edgelab.pipeline.timing_subset --frames-root .../testing/frames --copy-to .../vm-exchange/frames`. Scene sizes among the 107 test videos: 01:34, 08:12, 03:10, 12:10, 04:9, then 05:8 and 07:8 tied → tie broken by lower scene id (05); not changed after seeing results. Videos shorter than 240 frames (01_0063, 01_0133, 01_0139, 01_0162, 06_0145, 06_0153, 08_0078, 12_0173; 06_0144 etc. not in the top 6) are not eligible.
- Chosen (12, in `timing_subset.txt`): 01_0014 01_0136 03_0032 03_0039 04_0004 04_0010 05_0021 05_0024 08_0079 08_0178 12_0143 12_0149. Copied 2880 jpgs (323 MB) to `/Users/nanh/thesis-data/vm-exchange/frames/<video>/NNN.jpg`.

### 2026-11-05 07:43 — ONNX (own pre/post-processing) vs Ultralytics parity, host, 3 frames
- Frames: 01_0014 #160 (L0), 03_0031 #190 (L0), 04_0010 #20 (L2), each re-saved as JPEG q100 after darkening, chosen because the .pt model finds ≥ 3 people; `python -m edgelab.pipeline.ort_pose --onnx yolov8n-pose-640.onnx --pt <weights> --frames ...`, iou 0.7, CPU.
- Against Ultralytics running **the same ONNX file** (letterbox 640×640): box counts equal on 3/3 frames at conf 0.25 and 0.1 (3, 4, 3); max |conf diff| 6e-7; max box diff 0.16 px; max keypoint diff 0.15 px. → our pre/post-processing is equivalent.
- Against Ultralytics running **the `.pt` weights** (what the EXP-003 cache used; letterbox auto → 640×384 rectangle for 856×480): box counts equal at conf 0.25 (3, 4, 3); max |conf diff| 0.022; max box diff 0.45 px; keypoint 0.56 px. At conf 0.1 one frame differs: 4 (.pt) vs 3 (ONNX) boxes. This difference is the padding shape (static 640×640 export), not our code. Consequence: VM timing is for the 640×640 ONNX (≈1.67× the pixels of the cache's 640×384); accuracy comes from the .pt cache.

### 2026-11-05 07:50 — VM smoke of live_timing (SMOKE ONLY, noisy host: load avg 7–10, cache sweep running; not for results)
- Script `vm-exchange/scripts/smoke_live_timing.sh` (tmux, log `results/smoke3.log`); 1 video (04_0010) x 60 frames, L0, c 0.25, VM idle otherwise, no co-load. Files `results/smoke_L0_*.csv` (+ `.meta.json`). First attempts on 01_0014 had no person in the first 60 frames (0 tracks, flow never exercised) and L2 gave 0 tracks; so 04_0010 L0.
- Smoke means (ms): E0/O0: read 5.5, detect infer 99, pre 4.1, post 0.2, track 0.6, flow 23.5 (38 on the 37 frames with new segments, 92 segments), service 133 (p50 127, p95 230). E3/O0: enhance 173 (p50 161), infer 126, service 329 (p50 301, p95 501). E0 with O1 knobs (threads 3, spin off, cpus 1–3, FIFO 10, under sudo): infer 118, service 143 (p50 137). darken ≈ 90 ms (numpy float64, excluded from service_ms).
- Knobs verified by `<out>.meta.json` (`threads_applied`): O1 → 22 of 22 threads FIFO and affinity 1,2,3. An earlier version set them only on the main thread: 12 of 22 threads (pools created at import) stayed OTHER / 0–3; fixed by applying to every thread in /proc/self/task, before and after the warm-up. torch threads: default 4 in VM (so the "1" noted at 23:41 was transient).
- Checked the command from WALKTHROUGH (co-load start pinned to vCPU 0 nice 19 → sudo live_timing O1 → stop): ran, co-load 28–37% of one CPU, ffmpeg NI 19 PSR 0; test files deleted.
- Not run: the timing matrix, B2 (cgroup cpu.max) cells, host-load sampling (lead).

### 2026-11-05 07:37 — Correction to the three code-lane-2 entries above
- Their headings say 07:40/07:41/07:43/07:50; they were written at 07:36–07:37 wall clock (headings are approximate, ordering is right).
- In the subset entry, the videos below 240 frames that were actually ineligible in the top-6 scenes are 01_0063, 01_0133, 01_0139, 01_0162, 08_0078, 12_0173 (06_0145 and 06_0153 are scene 06, not in the top 6).

### 2026-11-05 10:33 — Lead: review of code lane 2; Amendment 1 frozen
- Reviewed lane return: 99 passed / 10 skipped. Smoke (noisy, not results): VM service ≈ 133 ms/frame E0-O0, 329 ms
  E3-O0 (Zero-DCE++ ≈ 173 ms) → ≈ 3× the 41.7 ms period at 24 fps; deadlines bind even at B4.
- Found a validity issue before any result: timing ONNX was 640×640 while EXP-003 accuracy used Ultralytics'
  letterboxed 640×384 (`.pt` vs ONNX conf diff up to 0.022). Fix = per-R rectangular ONNX (Amendment 1).
- Plan §6 Amendment 1 frozen (cell reduction 96 → 30 cells, 40 frames/video, cgroup scopes for O1/B2).
  Lane 2 resumed to implement it (no matrix run until the host is idle).

### 2026-11-05 10:40 — Amendment 1, task 1: rectangular static ONNX per R, parity (host, nice 10)
- Letterbox shape read from a real `model.predict` (hook on the torch network's forward; code `pipeline/export_onnx.py:predictor_input_shape`; `LetterBox(auto=True, stride=32)`: pad = remainder of (R − resized) mod 32, centred). For an 856×480 frame at imgsz=R, the tensor shape (h, w) fed to the `.pt` network: R=480 → 288×480, R=640 → 384×640, R=800 → 480×800, R=960 → 544×960 (the 960 value was printed by the export log "input shape (1, 3, 544, 960)"; the others are the shapes of the exported files' ONNX inputs, read back by `OrtPose.shape`, and match hand computation: 480×269+19 pad rows, 640×359+25, 800×449+31, 960×538+6).
- Export: `python -m edgelab.pipeline.export_onnx` (imgsz=[h, w], opset 17, static, no simplify; weights `artifacts/EXP-001/models/yolov8n-pose.pt`) → `vm-exchange/models/`:
  - `yolov8n-pose-480-288x480.onnx` sha256 `a40ec7da206619f780f41c9cd06801b4ece40588e8c9f0f90d6f113e70859192`
  - `yolov8n-pose-640-384x640.onnx` sha256 `93b8fbdbbd7fa5e45d09bf223b8fb2aca350c1743cdcb16778ab45349260d726`
  - `yolov8n-pose-800-480x800.onnx` sha256 `cd17c200301ce919c510277a8705d00f172f67739c5ce40df172b2183f72b9b0`
  - `yolov8n-pose-960-544x960.onnx` sha256 `3258c6737a1453fda9513832960dbee8766ec920e3a334b3968596fe93e80916`
  - (old square `yolov8n-pose-640.onnx` kept, not used any more for results.)
- `ort_pose.letterbox` now takes the target (h, w); `OrtPose` reads (h, w) from the ONNX input, so the matching letterbox is automatic. Same 3 frames as before (01_0014 #160 L0, 03_0031 #190 L0, 04_0010 #20 L2), ours vs Ultralytics `.pt` predict at the same imgsz, iou 0.7, CPU, conf 0.1 and 0.25 (`result2` printed by `python -m edgelab.pipeline.ort_pose --onnx ... --pt ... --imgsz R`):
  | R | box counts equal (conf 0.1 / 0.25) | max conf diff | max box diff px | max keypoint diff px |
  |---|---|---|---|---|
  | 480 | 3/3 / 3/3 (counts 2,4,3 / 1,3,2) | 2.0e-6 | 0.28 | 0.27 |
  | 640 | 3/3 / 3/3 (3,4,4 / 3,4,3) | 5.4e-7 | 0.16 | 0.15 |
  | 800 | 3/3 / 3/3 (3,2,4 / 3,2,4) | 7.2e-7 | 0.42 | 0.41 |
  | 960 | 3/3 / 3/3 (3,3,4 / 3,3,4) | 7.7e-7 | 0.28 | 0.27 |
  → the 640×640 vs `.pt` mismatch of 07:43 (conf diff 0.022, one count differing) is gone with the rectangular export.

### 2026-11-05 10:55 — Amendment 1, tasks 2–3: `--imgsz`, cgroup scripts, scope checks (VM, short busy loops)
- `live_timing`: `--imgsz {480,640,800,960}` picks `yolov8n-pose-<R>-*x*.onnx` from `vm-exchange/models` (exactly one match required); `--max-frames-per-video` (alias `--max-frames`, default 240; matrix uses 40); `imgsz` is a CSV config column; `<out>.meta.json` records the onnx file, its input (h, w), and the process's cgroup + the parent's `cpu.max`.
- Scripts in `EXP-004-os-deadlines/scripts/`: `vm_scopes.sh` (library), `cell_in_vm.sh <B4|B2> <O0|O1> <live_timing args>` (run with `sudo bash` in the VM), `scope_check.sh`, and the older `coload.sh` (kept for the smoke). Layout: slice `edgelab-b{4,2}.slice` (systemd nests it as `/sys/fs/cgroup/edgelab.slice/edgelab-b2.slice`; B2 → `CPUQuota=200%`, B4 → none) holds both the pipeline scope (`systemd-run --scope --slice=…`) and the co-load scope, so the quota is shared. O0: co-load scope with no restriction; O1: co-load scope with `AllowedCPUs=0` and `nice -n 19` (`Nice=` is not valid for scopes, so nice is applied inside the scope).
- `scope_check.sh` (systemd 255, busy loops: 4 "pipeline" + 2 "co-load", 5 s window, CPU-seconds from cgroup `cpu.stat`):
  | B | O | slice `cpu.max` | slice CPUs used | co-load scope CPUs used | co-load cpuset / Cpus_allowed |
  |---|---|---|---|---|---|
  | B4 | O0 | max 100000 | 3.74 | 1.31 | 0-3 |
  | B4 | O1 | max 100000 | 3.87 | 0.02 (nice 19 loses to 4 normal loops on vCPU 0) | 0 / 0 |
  | B2 | O0 | 200000 100000 | 2.00 | 0.94 | 0-3 |
  | B2 | O1 | 200000 100000 | 2.03 | 0.49 | 0 / 0 |
  → the quota caps the slice (pipeline + co-load together) at 2.0 CPUs; the cpuset keeps the co-load on vCPU 0. (The VM sees noisy-host steal, so B4 slice < 4.0 is expected.)
- End-to-end check of `cell_in_vm.sh B2 O1` (1 video × 15 frames, R=480, L3, E0; test files deleted): pipeline cgroup `/edgelab.slice/edgelab-b2.slice/edgelab-pipeline-<pid>.scope`, parent `cpu.max` `200000 100000`, 22/22 threads FIFO + cpus 1,2,3, input (h, w) = [288, 480]; script printed `COLOAD_CPU_S=14.2 WALL_S=22` (the real ffmpeg co-load used ≈ 0.65 CPU on average in this noisy VM; earlier ps estimate was ≈ 0.3).

### 2026-11-05 11:10 — Amendment 1, task 4: matrix runner, host-load method, dry run (nothing was run)
- New: `src/edgelab/pipeline/{timing_matrix,host_load}.py`, `scripts/run_matrix.sh`, `tests/test_pipeline_timing_matrix.py` (11 tests). Cells per plan §6: 16 (P × O × B at R 640) + 12 (R 480/800/960 × O × B at E0) + 2 (L0; c = G-xfit, required argument `--gxfit-c`) = 30 cells × 5 reps = 150 runs, 40 frames × 12 videos each. Order: each repetition is one random permutation of the 30 cells (seed 0) so every repetition covers every cell; written to `timing_plan.csv` before the first run (the runner refuses to continue if an existing plan file differs). Manifest: `vm-exchange/results/matrix/manifest.csv` (cell, rep, order, start/end, status, excluded, host load, VM %CPU, co-load CPU-s, trace file, error); resume skips `status=ok`; 3 failures in a row stop the runner; a failed/timed-out run triggers cleanup of leftover scopes in the VM.
- Host-load method (macOS): `top -l 2 -s 3 -n 40 -o cpu` repeatedly during a run (second sample = a real 3 s interval); total user+sys % of all 10 cores minus the edge-os VM process's %CPU / 10. The VM process = the `com.apple.Virtualization.VirtualMachine` XPC service whose age is within 60 s of the Lima hostagent (`~/.lima/edge-os/vz.pid`); today pid 13488 (pid 5906 is Docker's VM, not subtracted). If the VM is not in top's first 40 rows it counts as 0. Run is excluded if the mean outside-VM figure > 30% (or unknown). Check right now: total 100.7%, VM 0.0% (idle) → outside-VM 100.7%: every run would be excluded on the current host.
- Dry run (`python -m edgelab.pipeline.timing_matrix --gxfit-c 0.15 --dry-run`; 0.15 is a placeholder, not the EXP-003 value): 150 runs; estimate from the 07:50 smoke means (E0/E3 totals incl. read+darken, detector inference scaled by input pixels vs the 640×640 smoke, +20 s start-up measured in the cell check, E1/E2 = E0): **lower bound 3.4 h**, ×1.5 bracket 5.1 h (B2/O1/co-load slow-downs and the E1/E2 cost are not modelled).
- Whole suite after all changes: 115 passed, 10 skipped (same old skips), 0 failed.

### 2026-11-05 10:47 — Lead: G-xfit value for timing; H9–H11 status; matrix launch deferred
- EXP-003 cross-fit at L3_E0 chose c = 0.10 (fit on half A) and 0.25 (half B): no single value. For the H10 cost
  check the **lowest chosen value, c = 0.10**, is used (most detections → upper bound on tracker/flow cost).
  Launch argument: `run_matrix.sh 0.10`.
- Plan §4 rule "if EXP-003 fails H2/H3, H9–H11 are dropped": EXP-003 interim H3 = FAIL → **H9, H11 recorded as not
  tested**; H10 (gate CPU cost) is kept as a descriptive cost check because it informs why the gate is not a remedy.
  Any resolution-based replacement for H11 will be added as a labelled exploratory addendum before timing results.
- Matrix **not launched**: host load outside the VM is ≈ 100% (EXP-003 cache sweep), so every run would be
  excluded by the §3.3 rule. Launch after the sweep finishes and the owner idles/reboots the host.

### 2026-11-05 22:12 — Lead: matrix auto-launch chained
- Owner asked to free the host. At 22:10 the owner's `node` processes were already gone; remaining load was our own
  (6 EXP-003 Addendum B detcache jobs + table generator). Nothing killed.
- `~/thesis-data/exp003/launch_matrix_when_idle.sh` (tmux `launcher`, log `launcher.log`): waits for B_DONE and no
  EXP-003 process, then for 4 consecutive 30 s host samples with user+sys < 20%, then starts
  `run_matrix.sh 0.10` in tmux `exp004-matrix`. The runner's own per-repetition exclusion rule (§3.3) still applies.

### 2026-11-05 22:18 — Lead: analysis driver ready; overnight chain
- `src/edgelab/analysis/exp004.py` (+ `tests/test_analysis_exp004.py`, 3 passed): usable traces (status ok and not
  excluded), timing summary per cell (stage means, p50/p95/p99, per-rep p99 range, kept fraction at 24 fps),
  delivered AUC for every L3 c-0.25 cell replayed at L3/L2/L0 caches (Amendment 1 trace reuse), H8 interaction
  contrasts (paired bootstrap over videos) for P ∈ {E1, E2, E3} × B × light ∈ {L3, L2}.
- Overnight chain (owner away, host left idle): Addendum B caches → B scoring → launcher (idle check) →
  `run_matrix.sh 0.10` → chain-c (Addendum C caches + scoring). Launcher fixed to also wait for 24 B score rows.
- Known runner limitation: excluded repetitions are not retried automatically; to be handled after the matrix.

### 2026-11-05 22:59 — Lead: analysis driver ready; overnight chain
- `src/edgelab/analysis/exp004.py` (+ `tests/test_analysis_exp004.py`, 3 passed): usable traces (status ok and not
  excluded), timing summary per cell (stage means, p50/p95/p99, per-rep p99 range, kept fraction at 24 fps),
  delivered AUC for every L3 c-0.25 cell replayed at L3/L2/L0 caches (Amendment 1 trace reuse), H8 interaction
  contrasts (paired bootstrap over videos) for P ∈ {E1, E2, E3} × B × light ∈ {L3, L2}.
- Overnight chain (owner away, host left idle): Addendum B caches → B scoring → launcher (idle check) →
  `run_matrix.sh 0.10` → chain-c (Addendum C caches + scoring). Launcher fixed to also wait for 24 B score rows.
- Known runner limitation: excluded repetitions are not retried automatically; to be handled after the matrix.

### 2026-11-06 03:45 — Lead: main matrix done; budget-escape finding; Amendment 2; delivered replays
- Matrix 23:52–03:39: **150/150 runs ok, 0 excluded** (host outside VM 14–17%). Timing summary:
  `~/thesis-data/exp004/timing_summary.csv` (per cell: stage means, p50/p95/p99, per-rep p99 range, kept fraction).
- Key timing facts (5 reps, VM, L3, co-load): detector infer ≈ 30 ms @640 (O0-B4), 18 ms @480, 49 ms @800,
  65 ms @960; enhance: gamma 0.7 ms, CLAHE 3.0 ms, Zero-DCE++ 101 ms (O0-B4) / 186 ms (O0-B2); track ≈ 0.2 ms;
  flow 1–8 ms. Kept fraction at 24 fps (O0-B4): 640 0.97, 800 0.73, 960 0.55, Zero-DCE++ 0.29; (O0-B2): 0.95, 0.53,
  0.37, 0.14.
- **Budget escape:** O1 (SCHED_FIFO) timings are identical at B2 and B4 in all 7 knob cells → cgroup v2 cpu.max did not
  throttle the RT pipeline. O1 vs O0 at B4 (valid): O1 slower (E0@640 p50 41.2 vs 34.4 ms; p99 88.1 vs 62.7).
- Plan §7 Amendment 2 frozen (O1n = O1 without FIFO; 12 cells × 5 reps, seed 2; `--amendment2`, plan file
  `timing_plan_amendment2.csv`; `vm_scopes.sh` treats O1n like O1 for co-load isolation). Dry run 60 runs,
  1.1–1.6 h. timing_matrix tests 11 passed. Queued after Addendum C + replays.
- Delivered replays launched (tmux deliv-L3/L2/L0, `--no-tests`). Check: zero-drop cells reproduce EXP-003's offline
  live AUC exactly (L0 r480 61.65, L2 r480 62.38).

### 2026-11-06 04:01 — Lead: delivered AUC (84 replays) + a labelling bug fixed
- Bug (driver, caught before use): `delivered_all` rows let the cell's `light` (L3 = trace level) overwrite the replay
  level, so all summary rows said L3. Fixed (`trace_light` column); summaries rebuilt from the saved result.json
  files (no replay rerun). Assert: one row per (light, cell).
- Delivered live AUC, 24 fps (`~/thesis-data/exp004/delivered_summary_{L0,L2,L3}.csv`, MLflow `EXP-004`):
  O0-B4: L0 640 66.19 · 800 65.70 · 960 63.80 · E3 58.57 | L2 640 66.13 · E2 67.06 · 960 63.08 · E3 58.02 |
  L3 640 61.78 · E1 63.41 · 960 59.89 · E3 58.51. O0-B2: L0 960 59.70, E3 53.62; L2 960 59.63; L3 960 58.52.
  Zero-drop cells equal EXP-003 offline live AUC (e.g. L2 E2 67.06, L3 E1 63.41).
- **Main reading:** offline resolution gains (+2.0 at 960) turn into delivered losses (−2.4 at B4, −6.5 at B2) at
  24 fps; cheap enhancers keep their offline gains; Zero-DCE++ is net-negative at every light level.
- H8 interaction tests running (tmux h8); B2 tests involve O1 = budget escape → reported as different-budget.

### 2026-11-06 04:10 — Lead: H8 interaction tests (`~/thesis-data/exp004/interaction_tests.json`)
- Contrast = [Δ(P vs E0) under O1] − [Δ(P vs E0) under O0], delivered live AUC, paired bootstrap over videos (1000).
- **Pre-registered H8 (L3, B2, E3): +3.15 [+0.33, +6.01] → passes by rule, but O1-B2 escaped the 2-CPU quota
  (Amendment 2) → not a same-budget OS effect; H8′ (O1n) decides.**
- B4 (valid, no quota): L3 E1 −1.13 [−1.91, −0.28], E2 +0.55 [−0.34, +1.41], E3 −1.22 [−3.06, +0.61];
  L2 E1 −0.39 [−1.06, +0.32], E2 +0.76 [−0.06, +1.56], E3 −2.06 [−3.90, −0.24]. → with 4 CPUs the tuned config
  makes costly enhancement worse, not better.
- B2 (different budget): L2 E3 +5.67 [+2.67, +9.01], E1 +0.95 [+0.36, +1.58]; L3 E1 +1.02 [+0.31, +1.73].

### 2026-11-06 07:31 — Lead: Amendment 2 done (60/60 ok, 0 excluded); final analysis
- Timing: H7′ PASS (O1n-B2 p99 55.6 vs O0r-B2 99.7 ms; kept 0.980 [0.967–0.985] vs 0.943 [0.925–0.950]); O1n is
  throttled by the quota (E3 147 → 165 ms), O0r reproduces O0 (35.3 vs 35.1 ms).
- Delivered + H8′ (`final.log`, `interaction_tests.json`, `delivered_summary_L3_L2_L0.csv`): H8′ L3 E3 +2.34
  [−0.46, +5.20] → FAIL (CI includes 0); L3 E2 +0.86 [+0.29, +1.47]. Replay noise ≈ 2–3 AUC at > 80% drops (O0 vs O0r).
- results.md completed (§4 verdicts, §5 Amendment 2, Obs. 4 rewritten).
