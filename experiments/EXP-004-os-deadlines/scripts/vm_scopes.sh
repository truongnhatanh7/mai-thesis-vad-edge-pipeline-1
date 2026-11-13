# Sourced by cell_in_vm.sh and scope_check.sh. Run as root inside the VM `edge-os`.
#
# B (budget):  B4 = slice with no quota (4 vCPUs),  B2 = slice with CPUQuota=200% shared by pipeline AND co-load.
# O (co-load): O0 = co-load in the slice, any CPU, default nice;
#              O1 = co-load in its own scope inside the slice, AllowedCPUs=0 (cgroup cpuset), nice 19.
# `systemd-run --scope` registers a scope and then execs the command, so $! of a backgrounded call is the command's pid.

slice_name() { echo "edgelab-$(echo "$1" | tr 'A-Z' 'a-z').slice"; }   # B2 -> edgelab-b2.slice

cg_dir() { echo "/sys/fs/cgroup/edgelab.slice/$(slice_name "$1")"; }   # '-' in a slice name means nesting: edgelab.slice/edgelab-b2.slice

setup_slice() { # B
  local slice; slice=$(slice_name "$1")
  systemctl start "$slice"
  case "$1" in
    B2) systemctl set-property --runtime "$slice" CPUQuota=200% ;;
    B4) systemctl set-property --runtime "$slice" CPUQuota= ;;
    *) echo "unknown budget $1"; return 1 ;;
  esac
}

# prints the pid of the co-load command; the command keeps running until killed
start_coload() { # B O [command...]  (default command = ffmpeg x264 re-encode of a 720p test pattern in real time)
  local slice; slice=$(slice_name "$1"); local o=$2; shift 2
  local cmd=("$@")
  [ ${#cmd[@]} -gt 0 ] || cmd=(ffmpeg -loglevel error -re -f lavfi -i testsrc2=size=1280x720:rate=30 -c:v libx264 -preset veryfast -f null -)
  case "$o" in
    O0|O0r) systemd-run --scope -q --slice="$slice" --unit="edgelab-coload-$$" "${cmd[@]}" >/dev/null 2>&1 & ;;
    O1|O1n) systemd-run --scope -q --slice="$slice" --unit="edgelab-coload-$$" -p AllowedCPUs=0 nice -n 19 "${cmd[@]}" >/dev/null 2>&1 & ;;
    *) echo "unknown O $o" >&2; return 1 ;;
  esac
  echo $!
}

# runs the pipeline command in its own scope inside the slice (same quota as the co-load), waits for it
run_in_slice() { # B command...
  local slice; slice=$(slice_name "$1"); shift
  systemd-run --scope -q --slice="$slice" --unit="edgelab-pipeline-$$" "$@"
}
