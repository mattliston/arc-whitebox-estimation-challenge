#!/bin/bash
# TAG fable-porter v3 (census_boundary_setup_v2.py: + n_out/nonfinite per net; v2 kept) (setup-calling client; lead 18:1x: smoke/residual with setup() called). Base: copy of fable_engineer2_v1/run_census.sh: outputs (rss_<tag>.txt, <tag>_census.json) land in fable_porter_v1/ and the
# census script is the local copy census_boundary.py (unchanged logic). Counts requests AND hidden free trips; first rank = cold.
# usage: run_census_porter.sh <cand.py abs> <ranks e.g. 35,100,35> <tag>   (one client+server pair, 1 BLAS thread each, nice 10)
P2=/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16; A=$P2/astra_phase2_reexamination_v1; E=$P2/fable_porter_v1; C=$P2/data_gate/.client-venv/bin/python
URL="ipc:///tmp/porter_$$.sock"; rm -f /tmp/porter_$$.sock
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 FLOPSCOPE_MAX_ARRAY_BYTES=4294967296 nice -n 10 $A/openblas_runtime_env_v1/bin/python -m flopscope_server --url "$URL" --timeout 1800 > $E/server_$3.log 2>&1 &
SPID=$!; sleep 4
( while kill -0 $SPID 2>/dev/null; do ps -o rss= -p $SPID; sleep 0.25; done ) > $E/rss_$3.txt &
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 FLOPSCOPE_SERVER_URL="$URL" E2_SERVER_PID=$SPID nice -n 10 $C -B -u $E/census_boundary_setup_v2.py $1 $2 $3 2>&1 | grep -v Warning
kill $SPID; rm -f /tmp/porter_$$.sock; sleep 1
echo "server peak RSS MB: $(sort -n $E/rss_$3.txt | tail -1 | awk '{print $1/1024}')"
