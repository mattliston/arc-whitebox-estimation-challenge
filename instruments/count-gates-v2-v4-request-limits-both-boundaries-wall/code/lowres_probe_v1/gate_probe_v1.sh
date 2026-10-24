#!/bin/bash
# [opus-lowres] after a boundary probe finishes: official120 early gates (normal, slow) + count_gates_v2. usage: gate_probe_v1.sh <cand.py> <probe_dir>
O=/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1
C=$1; B=$2; OFF=$O/lead/runs/astra_smirnov_rpe_delivery_v1/official120_v1; PY=/Users/matt/arc-white-box-estimation/.venv/bin/python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 VECLIB_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 BLIS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
until grep -q "BOUNDARY_DONE" $B.out 2>/dev/null; do sleep 10; done
cd $O/release
$PY -B $OFF/early_count_gate_official120_v1.py $C $B/normal normal $B/OFFICIAL_NORMAL_v1 > $B/gate_normal.log 2>&1; echo "early normal rc=$?"
$PY -B $OFF/early_count_gate_official120_v1.py $C $B/slow slow $B/OFFICIAL_SLOW_v1 $B/OFFICIAL_NORMAL_v1 > $B/gate_slow.log 2>&1; echo "early slow rc=$?"
cd $O/lead/runs/opus_passrel_v1 && python3 count_gates_v2.py $B $O/lead/runs/opus_release_v1/rel_s2tr_20261016_opus_v1 $O/fire/s2tr_20261016_opus_v1/eval_336412.json
