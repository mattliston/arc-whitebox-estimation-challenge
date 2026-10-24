#!/bin/bash
# TAG opus-trainer RAD (10-12 07:5xZ, S54): one guarded no-plan worker (worker_generated_v12.py, 8 threads, 8M truths, cert/Nh/launch_worker_v12_8m.sh) on
# run_RAD_<name>_gen with the allow list cert/RAD/gen_allow_v10.txt (A36 + A54 + A56 + A57 + A59 + A55 check 7101480-511; sealed sets and the A47 / A58 reserves NOT allowed).
# usage (from O/trainer): bash cert/RAD/launch_gen_v10.sh <index> <name>
IDX=$1; NAME=$2; STAMP=$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p cert/RAD/$NAME
exec bash cert/Nh/launch_worker_v12_8m.sh $IDX cert/RAD/gen_allow_v10.txt "" "" cert/RAD/$NAME/worker_gen${IDX}_guard_$STAMP.json 14400 run_RAD_${NAME}_gen \
  > cert/RAD/$NAME/worker_gen${IDX}_$STAMP.log 2>&1
