# Rebuild the `edge-os` VM (Lima) from scratch

Reconstructed by the lead from `log.md` (2026-11-04 23:26–23:4x entries) after the VM lane stopped. Every step below
was run once; commands are copied from the log. Host: macOS on Apple silicon, `limactl` 2.0.3.

## 1. Create the VM (host)

```bash
mkdir -p ~/thesis-data/vm-exchange/{models,results,frames,scripts}
# config: same Ubuntu 24.04 image as template://ubuntu-lts, but WITHOUT Lima's default "mount all of ~"
cat > ~/thesis-data/vm-exchange/edge-os.yaml <<'EOF'
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
EOF
limactl create --name edge-os --tty=false ~/thesis-data/vm-exchange/edge-os.yaml   # ~5 min (image download)
limactl start edge-os --tty=false
```

The code is mounted **read-only**; only `vm-exchange` is writable (results, models, frames come back to the host there).

## 2. Inside the VM: packages, uv, Python env

```bash
limactl shell edge-os
sudo apt-get install -y build-essential curl git ffmpeg stress-ng linux-tools-$(uname -r) linux-tools-common \
  bpftrace bpfcc-tools sysstat libgl1 libglib2.0-0t64
curl -LsSf https://astral.sh/uv/install.sh | sh
mkdir -p ~/edgelab-vm && cp /Users/nanh/Documents/RMIT_MscAI/thesis/experiments/{pyproject.toml,uv.lock} ~/edgelab-vm/
cd ~/edgelab-vm && ~/.local/bin/uv sync --frozen --extra pose --no-install-project   # ~7 min, 6.3 GB
# stop automatic updates (one burned a core for hours on 2026-11-05)
sudo systemctl disable --now unattended-upgrades apt-daily.timer apt-daily-upgrade.timer
```

The package itself is not installed (read-only mount); run code with
`PYTHONPATH=/Users/nanh/Documents/RMIT_MscAI/thesis/experiments/src ~/edgelab-vm/.venv/bin/python -m ...`.

## 3. Models and frames (host)

```bash
cd ~/Documents/RMIT_MscAI/thesis/experiments
uv run --extra pose python -m edgelab.pipeline.export_onnx --imgsz 480 640 800 960 --out-dir ~/thesis-data/vm-exchange/models
uv run --extra pose python -c "from edgelab.preprocess.zerodcepp import ZeroDCEpp; ZeroDCEpp().export_onnx('$HOME/thesis-data/vm-exchange/models/zerodcepp.onnx')"
uv run --extra pose python -m edgelab.pipeline.timing_subset --frames-root ~/thesis-data/shanghaitech/shanghaitech/testing/frames \
  --copy-to ~/thesis-data/vm-exchange/frames
```

## 4. Run the timing matrix (host, VM running, host idle)

```bash
tmux new-session -d -s exp004-matrix 'bash EXP-004-os-deadlines/scripts/run_matrix.sh 0.10'                # main block
tmux new-session -d -s exp004-a2     'bash EXP-004-os-deadlines/scripts/run_matrix.sh 0.10 --amendment2'   # Amendment 2
```

Resumable; progress in `~/thesis-data/vm-exchange/results/matrix/{runner.log,manifest.csv}`.

## 5. Which OS knobs work in this VM (checked 2026-11-04, log entries "Knob (a)–(g)")

| Knob | Works? | How | Caveat |
|---|---|---|---|
| intra-op threads, ORT spinning | yes | `SessionOptions`, `torch.set_num_threads` | — |
| CPU pinning | yes | `os.sched_setaffinity`, `taskset` | vCPUs float over host P/E cores |
| SCHED_FIFO / negative nice | yes (root) | `os.sched_setscheduler`, `chrt -f` | **not throttled by cgroup v2 `cpu.max`** (EXP-004 Amendment 2) |
| cgroup v2 cpuset / cpu.max | yes (root) | `systemd-run --scope -p AllowedCPUs= -p CPUQuota=` | quota throttling is bursty; not a slower CPU |
| cpufreq / governor | **no** | — | Pi only |
| perf (software events), bpftrace, bcc | yes (root) | `perf stat`, `bpftrace`, `runqlat-bpfcc` | no hardware PMU counters in the guest |
| co-load (ffmpeg x264) | yes | `EXP-004-os-deadlines/scripts/coload.sh` | — |
