#!/bin/bash
# TAG g22-launch-mech15b (T-116g): honest oracle rows (tape built from the one-step mean), three drivers = six workers
D=$(cd "$(dirname "$0")" && pwd); cd "$D"
for rng in "35 100" "101 167" "168 234"; do
  set -- $rng
  G22_OUT=$D/rows_meanleak_mech15_k22 nohup nice -n 10 /Users/matt/arc-white-box-estimation/.venv/bin/python run_gen_meanleak_mech15_k22.py $1 $2 > gen_mech15b_$1.log 2>&1 &
  sleep 1
done
