#!/bin/bash
# TAG opus-reexam machine gate: block until a heavy launch is allowed: fewer than 6 python processes above 3 GB RSS on the box,
# free+inactive memory >= <own peak GB> + 4 GB, and memory_pressure free >= 30%. Re-checks every 60 s and prints why it waits.
# usage: gate.sh <own_peak_GB> [label]
need=$1; label=${2:-job}
while true; do
  heavy=$(ps -axo rss=,command= | awk '$1 > 3145728 && /python/ {c++} END {print c+0}')
  fi=$(vm_stat | awk '/page size of/ {ps=$8} /Pages free/ {gsub("\\.","",$3); f=$3} /Pages inactive/ {gsub("\\.","",$3); i=$3} END {printf "%.1f", (f+i)*ps/1073741824}')
  mp=$(memory_pressure | tail -1 | grep -o '[0-9]*%' | tr -d '%')
  if [ "$heavy" -lt 6 ] && awk -v a="$fi" -v b="$need" 'BEGIN {exit !(a >= b + 4)}' && [ "$mp" -ge 30 ]; then
    echo "$(date +%T) gate OPEN for $label: heavy=$heavy free+inactive=${fi}GB need $(awk -v b=$need 'BEGIN{print b+4}')GB mp=${mp}%"; exit 0
  fi
  echo "$(date +%T) gate WAIT for $label: heavy=$heavy (<6) free+inactive=${fi}GB (need $(awk -v b=$need 'BEGIN{print b+4}')) mp=${mp}% (>=30)"; sleep 60
done
