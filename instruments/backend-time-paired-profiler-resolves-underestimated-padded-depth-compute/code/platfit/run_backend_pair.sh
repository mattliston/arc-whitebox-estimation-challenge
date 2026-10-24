#!/bin/bash
# engineer2: CONCURRENT in-process backend-time pair (the local analog of the platform's kernel wall): S7 and S6b run the same 6 nets
# side by side (same load), per-op backend seconds per net -> platfit/optime_<tag>.json. usage: run_backend_pair.sh <tagA> <fileA> <tagB> <fileB>
P2=/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16; E=$P2/fable_engineer2_v1
PY=$P2/astra_phase2_reexamination_v1/openblas_runtime_env_v1/bin/python; NETS=35,100,37,39,42,48
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 nice -n 10 $PY -W ignore $E/platfit/optime_split.py $2 $NETS $1 > $E/platfit/optime_$1.log 2>&1 & q1=$!
sleep 1
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 nice -n 10 $PY -W ignore $E/platfit/optime_split.py $4 $NETS $3 > $E/platfit/optime_$3.log 2>&1 & q2=$!
wait $q1 $q2; echo "$(date +%T) BACKEND_PAIR_DONE $1 $3"
