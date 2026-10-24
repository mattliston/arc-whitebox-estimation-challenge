#!/bin/bash
# TAG opus-trainer RAD S51 (10-12 14:5xZ; lead 14:5xZ: more workers did not raise throughput): A/B of the worker count WITHIN S51's phase 2, with index
# w3 on TRAINER_RELIEF=0 (r0) in both windows. Window A: cap 8 (cert/RAD/S51/maintain_v5/TMAX; maintain_workers_v5.sh stops 11..8, one a minute,
# claims requeued), 5 min to settle, 30 min measured. Window B: cap 12 again, the maintainer relaunches 8..11 (one a minute, under the gate), 5 min
# to settle, 30 min measured. Then cert/RAD/S51/ab_analyze_v1.py (rates from results mtimes; the pre-declared decisions). Stops early on STOP.
# usage (from O/trainer): nohup bash cert/RAD/S51/ab_count_v1.sh > cert/RAD/S51/ab_count_v1.log 2>&1 &
cd "$(dirname "$0")/../../.."
M=cert/RAD/S51/maintain_v5; RUN=run_RAD_S51_gen; PY=/Users/matt/anaconda3/bin/python
alive() { ps -axo pid=,command= | awk -v r="$RUN" '$3 == "-B" && $4 ~ /^worker_generated_v[0-9]+\.py$/ && $5 == r' | wc -l | tr -d ' '; }
say() { echo "$(date -u +%H:%M:%SZ) $*"; }
echo 8 > $M/TMAX; say "window A: cap 8"
k=0; until [ "$(alive)" -le 8 ] || [ $k -ge 900 ]; do sleep 20; k=$((k + 20)); done; say "$(alive) alive"
sleep 300; A0=$(date +%s); say "window A measured from now (alive $(alive))"; sleep 1800; A1=$(date +%s); say "window A ends (alive $(alive))"
[ -e $RUN/STOP ] && { say "STOP: no window B"; exit 0; }
echo 12 > $M/TMAX; say "window B: cap 12"
k=0; until [ "$(alive)" -ge 12 ] || [ $k -ge 900 ]; do sleep 20; k=$((k + 20)); done; say "$(alive) alive"
sleep 300; B0=$(date +%s); say "window B measured from now (alive $(alive))"; sleep 1800; B1=$(date +%s); say "window B ends (alive $(alive))"
rm -f $M/TMAX
$PY -B cert/RAD/S51/ab_analyze_v1.py $A0 $A1 $B0 $B1
