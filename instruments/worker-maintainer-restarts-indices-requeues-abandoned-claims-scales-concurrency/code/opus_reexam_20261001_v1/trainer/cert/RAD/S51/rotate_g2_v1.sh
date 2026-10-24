#!/bin/bash
# TAG opus-trainer RAD S51 (10-12 08:2xZ, release 08:18Z: guard_footprint_v2 ready; lead 08:0xZ): moves S51's workers that still run under guard
# v1 (one 3-s ps timeout retires the worker; ps every 0.05 s) to guard v2, ONE AT A TIME: stop the worker of index i (SIGTERM; its claim stays
# and is re-served first by the same index), wait until maintain_workers_v2.sh has relaunched index i under its g2 launcher (at most 6 min), wait
# 30 s, next. Indices 2 3 4 5 0 1 (6 / 7 lie above the maintainer's first target and move at its target 8 or at their 4-h limit). Stops on STOP.
# usage (from O/trainer): nohup bash cert/RAD/S51/rotate_g2_v1.sh > cert/RAD/S51/rotate_g2_v1.log 2>&1 &
cd "$(dirname "$0")/../../.."
RUN=run_RAD_S51_gen; O=..
pid_of() { ps -axo pid=,command= | awk -v r="$RUN" -v i=$1 '$3 == "-B" && $4 ~ /^worker_generated_v[0-9]+\.py$/ && $5 == r && $6 == i {print $1}'; }
g2_of() { ps -axo pid=,command= | awk -v r="$RUN" -v i=$1 '$4 ~ /guard_footprint_v2\.py$/ && index($0, "run_RAD_S51_gen") && $0 ~ ("worker_generated_v12.py " r " " i "$") {print $1}'; }
moved=""
for i in 2 3 4 5 0 1; do
  [ -e $RUN/STOP ] && break
  [ -n "$(g2_of $i)" ] && { echo "$(date -u +%H:%M:%SZ) index $i already under guard v2"; continue; }
  P=$(pid_of $i); [ -n "$P" ] || { echo "$(date -u +%H:%M:%SZ) index $i not alive; the maintainer relaunches it under v2"; continue; }
  kill -TERM $P; echo "$(date -u +%H:%M:%SZ) index $i (pid $P) stopped"
  k=0; until [ -n "$(g2_of $i)" ] || [ $k -ge 360 ]; do sleep 10; k=$((k + 10)); done
  [ -n "$(g2_of $i)" ] && { moved="$moved $i"; echo "$(date -u +%H:%M:%SZ) index $i back under guard v2 after ${k} s"; } || echo "$(date -u +%H:%M:%SZ) index $i NOT back under v2 in 6 min (maintainer gate?)"
  sleep 30
done
echo "[trainer] $(date -u +%Y-%m-%dT%H:%M:%SZ) S51 workers moved to guard_footprint_v2 one at a time (cert/RAD/S51/rotate_g2_v1.sh): indices [$moved ] now under v2 (relaunched by maintain_workers_v2.sh with launch_gen_v7_t2_g2.sh, claims re-served by the same index); 6 / 7 follow at the maintainer's target 8 or their 4-h limit." >> $O/TEAM_LOG.md
