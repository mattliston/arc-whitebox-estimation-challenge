#!/bin/bash
# TAG fable-engineer2 (E2-S1LR v2 gates, lead 21:47): s1 r32 on S21 alone (S21_s1lr_mem.py). 1) validate / einsum / imports vs S21;
# 2) parity on 35,100,37,39 vs S21 (in-situ bill, MSE); 3) VALID40 through the porter's harness vs S21 rows (yielding to the porter's
# S22 chain): bar adjusted <= -1.2% with the CI excluding 0, pooled raw <= +0.8%; 4) paired residual gate vs S21 (cold delta <= +0.010 s).
set -u
P2=/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16; E=$P2/fable_engineer2_v1; H=$P2/fable_porter_v1; L=$E/lrroute
PY=$P2/astra_phase2_reexamination_v1/openblas_runtime_env_v1/bin/python; S21=$P2/fable_fire_v1/fire_candS21/candS21_S18ep2_dials_d6x_mem.py
C=$L/S21_s1lr_mem.py; J21=$H/paired_valid40_stack_S21_round1_mem.json; T=e2_S21s1lr
echo "$(date +%T) s1 r32 on S21: $(shasum -a 256 $C | cut -c1-8) vs S21 $(shasum -a 256 $S21 | cut -c1-8)"
echo "validate: $(/Users/matt/arc-white-box-estimation/.venv/bin/whest validate --estimator $C 2>&1 | grep -o 'Validation passed.*\|Validation failed.*')"
echo "einsum gate vs S21: $(python3 $H/einsum_site_diff.py $C $S21 | tail -1)"
echo "import lines new vs S21: $(diff <(grep -E '^\s*(import|from) ' $S21 | sort -u) <(grep -E '^\s*(import|from) ' $C | sort -u) | grep '^>' | tr '\n' ' ')(empty = none)"
cd $E && OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 nice -n 10 $PY -W ignore parity.py $C 35,100,37,39 par4_S21s1only > $E/read6/par4_S21s1only.log 2>&1
python3 $E/read6/parity_compare.py S21 S21s1only | tail -2
echo "$(date +%T) PARITY_DONE"
while pgrep -f "[f]ire_chain_S22.sh" > /dev/null; do sleep 60; done
[ -e $H/paired_valid40_$T.json ] && { echo "refusing to overwrite paired_valid40_$T.json"; exit 1; }
echo "$(date +%T) VALID40 queued on the lock"
bash $H/run_valid40_locked.sh $C $T
python3 $H/valid40_vs_ref_ci.py $H/paired_valid40_$T.json $J21 | tee $L/valid40_${T}_vs_S21.txt
python3 $H/valid40_bar.py $H/paired_valid40_$T.json $J21 -1.2 0 | tee $L/valid40_${T}_bar.json
echo "$(date +%T) VALID40_DONE"
while pgrep -f "[f]ire_chain_S22.sh" > /dev/null; do sleep 60; done
bash $L/resid_gate_v1.sh $C S21s1lr $S21
echo "$(date +%T) S1ONLY_DONE"
