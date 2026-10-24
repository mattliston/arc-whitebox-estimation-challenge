#!/bin/bash
# TAG fable-lead 2026-10-19: kill ONE lane cleanly = the driver pid and all its DESCENDANTS (walk down with pgrep -P), never by
# ancestry heuristics (which crashed three lanes on 10-19). usage: kill_tree_v1.sh <driver pid>
set -u
kids() { pgrep -P "$1" 2>/dev/null; }
all=()
walk() { all+=("$1"); for c in $(kids "$1"); do walk "$c"; done; }
walk "$1"
echo "killing tree of $1: ${all[*]}"
for p in "${all[@]}"; do kill -TERM "$p" 2>/dev/null; done
sleep 3
for p in "${all[@]}"; do kill -0 "$p" 2>/dev/null && kill -KILL "$p" 2>/dev/null && echo "KILL $p"; done
exit 0
