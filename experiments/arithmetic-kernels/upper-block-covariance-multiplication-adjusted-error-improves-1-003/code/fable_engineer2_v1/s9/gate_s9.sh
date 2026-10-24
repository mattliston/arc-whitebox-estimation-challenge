#!/bin/bash
# S9 light gate (engineer2; lead 09-23): S9 = fired S7 + refit parameter values only. (1) in-process 35,100,37,39 on S9 vs the fired S7's
# saved predictions/log (par5_S7F, same OpenBLAS env): bill must be IDENTICAL to S7 per net, MSE change per net; (2) paired 6-net
# client/server run S9 vs S7 (concurrent): server CPU, residual, round trips, handles, live peak identical within noise.
# usage: gate_s9.sh <S9.py abs> <tag>
P2=/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16; E=$P2/fable_engineer2_v1
PY=$P2/astra_phase2_reexamination_v1/openblas_runtime_env_v1/bin/python; V=$1; TAG=$2
S7=$P2/fable_fire_v1/fire_candS7/candS7_full_cb3_mem.py; NETS=35,100,37,39,42,48
waitq() { while pgrep -f "[q]uiet_pairs_v2" >/dev/null; do sleep 10; done; }
cd $E
waitq; echo "$(date +%T) in-process start $TAG ($(shasum -a 256 $V | cut -c1-16)) vs S7 $(shasum -a 256 $S7 | cut -c1-16)"
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 nice -n 10 $PY -W ignore $E/parity.py $V 35,100,37,39 par4_$TAG par5_S7F > $E/s9/par4_$TAG.log 2>&1
echo "$(date +%T) in-process done"
waitq; echo "$(date +%T) 6-net pair start"
bash $E/run_mem_caches.sh $V $NETS g6$TAG > $E/g6$TAG.out 2>&1 &
q1=$!; sleep 1
bash $E/run_mem_caches.sh $S7 $NETS g6ref$TAG > $E/g6ref$TAG.out 2>&1 &
q2=$!
wait $q1 $q2
echo "$(date +%T) GATE_S9_DONE $TAG"
