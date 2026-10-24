#!/bin/bash
# usage: run_mem_caches.sh <cand.py abs> <ranks> <tag> : census (requests/free trips) with the memory-accounting server wrapper
P2=/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16; A=$P2/astra_phase2_reexamination_v1; E=$P2/fable_engineer2_v1; C=$P2/data_gate/.client-venv/bin/python
URL="ipc:///tmp/e2m_$$.sock"; rm -f /tmp/e2m_$$.sock
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 FLOPSCOPE_MAX_ARRAY_BYTES=4294967296 E2_MEM_OUT=$E/mem_$3.jsonl E2_MEM_CTL=$E/memctl_$3.txt E2_MEM_WIN=$E/memwin_$3.jsonl nice -n 10 $A/openblas_runtime_env_v1/bin/python $E/server_mem.py --url "$URL" --timeout 1800 > $E/server_mem_$3.log 2>&1 &
SPID=$!; sleep 4
( while kill -0 $SPID 2>/dev/null; do ps -o rss= -p $SPID; sleep 0.25; done ) > $E/rssm_$3.txt &
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 FLOPSCOPE_SERVER_URL="$URL" E2_SERVER_PID=$SPID E2_MEM_CTL=$E/memctl_$3.txt nice -n 10 $C -B -u $E/census_e2_caches.py $1 $2 $3 2>&1 | grep -v Warning
kill $SPID; rm -f /tmp/e2m_$$.sock; sleep 1
cat $E/memwin_$3.jsonl
echo "server peak RSS MB: $(sort -n $E/rssm_$3.txt | tail -1 | awk '{print $1/1024}')"
python3 - "$E/mem_$3.jsonl" <<'PY'
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
print('server live-array peak MB', max(r[2] for r in rows), 'final live MB', rows[-1][1], 'max handles', max(r[3] for r in rows))
PY
