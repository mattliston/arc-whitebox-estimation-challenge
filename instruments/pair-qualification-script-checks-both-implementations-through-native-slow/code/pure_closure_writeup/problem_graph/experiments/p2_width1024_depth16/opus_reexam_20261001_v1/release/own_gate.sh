#!/bin/bash
# TAG opus-release: the lead's gate.sh plus a stricter look at swap. After the gate opens, system swap in use must stay flat for
# 30 s (a growing swap means another lane is still filling memory; the guard would kill our job at +256 MiB anyway) and free +
# file-backed memory must be at least 4 GB (the job's pages then come from the file cache, not from other lanes' compressed
# pages). v2 (2026-10-01 21:32 UTC): v1 asked for 1 GB free outright, which this machine almost never has while four lanes run
# (free pages sit at 0.01-1.3 GB with 15 GB inactive), so the queue stalled for no gain. Loops until both hold.
# usage: own_gate.sh <own peak GB> [label]
R="$(cd "$(dirname "$0")" && pwd)"
used() { sysctl -n vm.swapusage | sed 's/.*used = \([0-9.]*\)M.*/\1/'; }
avail_gb() { vm_stat | awk '/Pages free/ {gsub("\\.","",$3); f=$3} /File-backed pages/ {gsub("\\.","",$3); b=$3} END {printf "%.2f", (f+b)*16384/1073741824}'; }
while true; do
  bash "$R/../gate.sh" "$1" "${2:-release}" || exit 1
  a=$(used); sleep 30; b=$(used); f=$(avail_gb)
  if [ "$a" = "$b" ] && awk -v f="$f" 'BEGIN {exit !(f >= 4.0)}'; then echo "$(date +%T) own gate OPEN: swap flat at ${b}M for 30 s, free + file-backed ${f} GB"; exit 0; fi
  echo "$(date +%T) own gate WAIT: swap ${a}M -> ${b}M, free + file-backed ${f} GB"; sleep 30
done
