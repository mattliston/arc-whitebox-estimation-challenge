#!/usr/bin/env bash
# TAG fable-porter (PT-16): paired in-process USER-CPU census (lead + engineer2): fable_engineer2_v1/platfit/optime_split_v3.py (per-net user/sys
# CPU, page faults, per-op backend time) on candidate A and reference B CONCURRENTLY (same load window), OpenBLAS env, 1 thread each, nice 10,
# nets NETS (default 35,100,37,39,42,48; first cold). Output: fable_engineer2_v1/platfit/optime_<tag>.json (engineer2's project_wall.py input);
# logs in fable_porter_v1/stack/. Takes /tmp/fable_paired.lock. usage: run_usercpu_pair.sh <tagA> <fileA> <tagB> <fileB>
set -u
P2=/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16; E=$P2/fable_engineer2_v1; H=$P2/fable_porter_v1
PY=$P2/astra_phase2_reexamination_v1/openblas_runtime_env_v1/bin/python; NETS=${NETS:-35,100,37,39,42,48}
for t in "$1" "$3"; do [ -e "$E/platfit/optime_$t.json" ] && { echo "refusing to overwrite optime_$t.json"; exit 1; }; done
while ! mkdir /tmp/fable_paired.lock 2>/dev/null; do sleep 30; done
echo "$(date +%T) user-CPU pair start $1 vs $3 | $(memory_pressure | tail -1) | $(sysctl -n vm.swapusage)"
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 nice -n 10 $PY -W ignore $E/platfit/optime_split_v3.py "$2" $NETS "$1" > $H/stack/optime_$1.log 2>&1 & q1=$!
sleep 1
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 nice -n 10 $PY -W ignore $E/platfit/optime_split_v3.py "$4" $NETS "$3" > $H/stack/optime_$3.log 2>&1 & q2=$!
wait $q1 $q2; rmdir /tmp/fable_paired.lock; echo "$(date +%T) USERCPU_PAIR_DONE $1 $3"
