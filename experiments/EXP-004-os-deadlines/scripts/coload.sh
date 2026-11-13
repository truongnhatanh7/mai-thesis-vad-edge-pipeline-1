#!/bin/bash
# Co-load for EXP-004 (plan §3.2): ffmpeg x264 re-encode of a 1280x720 test pattern at real time ("recording to disk").
# Runs forever until stopped. Run INSIDE the VM (the experiments folder is mounted read-only: use `bash <path>`).
#
#   bash coload.sh start                  # O0: default priority, all CPUs
#   bash coload.sh start --pin 0 --nice 19  # O1: pinned to vCPU 0 (taskset), nice 19
#   bash coload.sh stop
#   bash coload.sh status
set -u
PIDFILE=/tmp/edgelab-coload.pid
cmd=${1:-}; shift || true
pin=""; nice_level=""
while [ $# -gt 0 ]; do
  case "$1" in
    --pin) pin=$2; shift 2 ;;
    --nice) nice_level=$2; shift 2 ;;
    *) echo "unknown option $1"; exit 2 ;;
  esac
done

running() { [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; }

case "$cmd" in
  start)
    running && { echo "already running, pid $(cat "$PIDFILE")"; exit 1; }
    prefix=()
    [ -n "$pin" ] && prefix+=(taskset -c "$pin")
    [ -n "$nice_level" ] && prefix+=(nice -n "$nice_level")
    "${prefix[@]}" ffmpeg -loglevel error -re -f lavfi -i testsrc2=size=1280x720:rate=30 \
      -c:v libx264 -preset veryfast -f null - >/tmp/edgelab-coload.log 2>&1 &
    echo $! >"$PIDFILE"
    sleep 1
    running && echo "co-load started, pid $(cat "$PIDFILE") (pin='${pin}' nice='${nice_level}')" \
      || { echo "co-load failed to start:"; cat /tmp/edgelab-coload.log; exit 1; }
    ;;
  stop)
    running && kill "$(cat "$PIDFILE")" && echo "stopped" || echo "not running"
    rm -f "$PIDFILE"
    ;;
  status)
    running && ps -o pid,ni,psr,pcpu,cmd -p "$(cat "$PIDFILE")" || echo "not running"
    ;;
  *) echo "usage: coload.sh start [--pin CPUS] [--nice N] | stop | status"; exit 2 ;;
esac
