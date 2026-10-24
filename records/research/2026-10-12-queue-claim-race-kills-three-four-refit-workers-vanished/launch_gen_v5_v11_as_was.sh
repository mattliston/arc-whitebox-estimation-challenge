#!/bin/bash
# TAG opus-trainer RAD (10-11 23:15Z): one guarded no-plan worker (worker_generated_v11.py, 8 threads, cert/Nh/launch_worker_v11.sh) on
# run_RAD_<name>_gen with the allow list cert/RAD/gen_allow_v5.txt (A36 + A39 + A41 + A42 check 7100648-679; sealed sets NOT allowed).
# usage (from O/trainer): bash cert/RAD/launch_gen_v5.sh <index> <name>
IDX=$1; NAME=$2; STAMP=$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p cert/RAD/$NAME
exec bash cert/Nh/launch_worker_v11.sh $IDX cert/RAD/gen_allow_v5.txt "" "" cert/RAD/$NAME/worker_gen${IDX}_guard_$STAMP.json 14400 run_RAD_${NAME}_gen \
  > cert/RAD/$NAME/worker_gen${IDX}_$STAMP.log 2>&1
