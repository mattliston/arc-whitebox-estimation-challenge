#!/bin/bash
# [opus-memgrowth] one worker-like session of N networks (release boundary_run, normal --short-setup), guard v3 with a the standard footprint
# limit (default 11 GB) so the growth past the platform's 8 GB can be seen; admission: memory_pressure free >= 55% for 3 checks in a row.
# usage: run_mn20_v1.sh <cand.py> <outdir> <ranks> [footprint_limit_bytes]
O=/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1
PY=/Users/matt/arc-white-box-estimation/.venv/bin/python; CAND=$1; OUT=$2; RANKS=$3; LIM=${4:-7500000000}; mkdir $OUT || exit 4
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 VECLIB_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 BLIS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
unset PB_ALL MODE DRY STATS SETUP_ONLY CAPTURED_SETUP_MANIFEST
ok=0; while [ $ok -lt 3 ]; do f=$(memory_pressure | sed -n 's/.*free percentage: \([0-9]*\)%.*/\1/p'); if [ "${f:-0}" -ge 55 ]; then ok=$((ok+1)); else ok=0; fi; sleep 10; done
cd $O/release
$PY -B $O/release/guard_footprint_v3.py --record $OUT/guard.json --seconds 7200 --footprint-limit $LIM --rss-backstop 10000000000 nice -n 10 $PY -B $O/release/boundary_run.py $CAND $RANKS $OUT --short-setup > $OUT.out 2>&1
echo "MN20_DONE exit=$?" >> $OUT.out
