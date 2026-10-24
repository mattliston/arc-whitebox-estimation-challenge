#!/bin/bash
# TAG fable-engineer2 (lead 10:5x priority: which bill scales with the old-tier rank): bill_census_layers_e2.py (setup first; nets 35 then
# 100 warm) on the porter's stack_S27r_mem.py (320 at L10+L13) and stack_S27s_mem.py (320 at L13 only), one process each, behind machine_gate.sh.
P2=/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16; E=$P2/fable_engineer2_v1; W=$P2/fable_porter_v1/stack
bash $P2/fable_porter_v1/machine_gate.sh 8 e2-bill-census-rank
for t in S27r S27s; do
  cd $E/bill27 && env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 nice -n 10 /Users/matt/arc-white-box-estimation/.venv/bin/python bill_census_layers_e2.py $W/stack_${t}_mem.py 35,100 $E/bill27/census_${t}_raw.json 2>&1 | grep -v "Warning\|^  \|configure("
  echo "CENSUS_${t}_DONE"
done
