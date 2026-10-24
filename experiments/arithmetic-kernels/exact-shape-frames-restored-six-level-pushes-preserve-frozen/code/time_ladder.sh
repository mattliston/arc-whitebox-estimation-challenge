#!/bin/bash
# boundary timing of the E ladder on a quiet box: numpy-free client + OpenBLAS server, 1 thread, 5 contexts each
P2=/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16; A=$P2/astra_phase2_reexamination_v1; F=$P2/fable_wall_repair_v1; C=$P2/data_gate/.client-venv/bin/python
URL="ipc:///tmp/fwr_flops.sock"; rm -f /tmp/fwr_flops.sock
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 FLOPSCOPE_MAX_ARRAY_BYTES=4294967296 nice -n 10 $A/openblas_runtime_env_v1/bin/python -m flopscope_server --url "$URL" --timeout 1800 > $F/server_ladder.log 2>&1 &
sleep 4
cd $F
for c in variantE6_d6push variantE5_d5squares variantE_contig_cached; do
  rm -f remote_${c}_quiet.json
  env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 FLOPSCOPE_SERVER_URL="$URL" nice -n 10 $C -B -u remote_series.py $F/$c.py remote_${c}_quiet 2>&1 | grep -v Warning | grep -E "wall_s|repeat" | sed -E 's/"timing".*"residual_wall_time_s": ([0-9.]+)[^}]*\}/residual \1/; s/"output_sha256": "[0-9a-f]*"/sha/; s/"shape": \[[0-9, ]*\]//' | sed "s/^/$c: /" | cut -c1-150
done
pkill -f "flopscope_server --url $URL"; rm -f /tmp/fwr_flops.sock; echo LADDER_DONE
