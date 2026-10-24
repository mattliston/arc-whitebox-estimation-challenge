#!/bin/bash
# TAG opus-reexam (lead, 2026-10-02): CPU gate by the machine's IDLE SHARE instead of the load average. At 13:15Z the 1-minute load read 51 while top showed
# 58% idle (16 cores): the load average counts short-lived runnable threads of other sessions' jobs and of the browser and stopped every lane's screens for nothing.
# usage: idle_gate.sh [min idle percent, default 30]   -> exit 0 when the idle share over a 2-second sample is at least that, 1 otherwise. Prints the idle share.
# Memory rules are NOT replaced by this (heavy_gate.sh, the swap rules of the lanes stay as they are).
need=${1:-30}
idle=$(top -l 2 -n 0 -s 2 | awk '/CPU usage/ {v=$7} END {gsub("%","",v); print v}')
echo "$idle"
awk -v i="$idle" -v n="$need" 'BEGIN {exit !(i + 0 >= n + 0)}'
