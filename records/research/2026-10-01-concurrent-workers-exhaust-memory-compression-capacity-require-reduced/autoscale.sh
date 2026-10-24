#!/bin/bash
# TAG opus-trainer (2026-10-02): worker count of a run by box load, so the run speeds up by itself when the other lanes pause.
# Every 2 minutes: 5-minute load average below 12 -> <run>/WORKER_COUNT = 2; above 20 -> 1; in between unchanged.
# (With two workers on an overloaded box the second adds a quarter more gradients for twice the memory and thrashes the compressor.)
# The supervisor reads the file; its gate and swap rules still apply. Ends when <run>/STOP exists.
# usage: nohup bash autoscale.sh <run dir> >> <run dir>/autoscale.log 2>&1 &
run=$1
while [ ! -e "$run/STOP" ]; do
  load=$(sysctl -n vm.loadavg | awk '{print $3}')
  now=$(cat "$run/WORKER_COUNT" 2>/dev/null || echo 2)
  want=$now
  if awk -v l="$load" 'BEGIN {exit !(l < 12)}'; then want=2; fi
  if awk -v l="$load" 'BEGIN {exit !(l > 20)}'; then want=1; fi
  if [ "$want" != "$now" ]; then
    echo "$want" > "$run/WORKER_COUNT"
    echo "$(date -u +%FT%TZ) 5-minute load $load: worker count $now -> $want"
  fi
  sleep 120
done
