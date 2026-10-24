#!/bin/bash
# TAG fable-engineer2 (v2 of run_after_S22.sh, same steps; yield pattern fixed per the porter) (re-queue 22:28, S22 priority): wait until NO fire_chain_S22* process runs (the S22 chain has lock priority;
# run_s1zero.sh's check missed the renamed fire_chain_S22s1_v2.sh), then, each through heavy_gate.sh:
#  1) VALID40 of S21s1 + --s1zero (a49da660) vs the S21s1 rows; bar adjusted <= -1.5% CI excl 0 (raw read from the CI report);
#  2) paired residual gate vs S21s1;  3) the residual-floor probe (setup-calling) on S21s1 nets 35 (cold), 100.
set -u
P2=/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16; E=$P2/fable_engineer2_v1; H=$P2/fable_porter_v1; L=$E/lrroute
REF=$P2/fable_fire_v1/fire_candS21s1/candS21s1_S21_s1lr_mem.py; C=$L/S21s1_s1zero_mem.py; JREF=$H/paired_valid40_e2_S21s1lr.json; T=e2_S21s1z
yield() { while pgrep -f "bash fire_chain_S2[2]" > /dev/null; do sleep 60; done; }   # v2 (porter 22:5x): match the chain invocation only, never watchers
yield; [ -e $H/paired_valid40_$T.json ] && { echo "refusing to overwrite paired_valid40_$T.json"; exit 1; }
bash $E/heavy_gate.sh 9 "VALID40 s1zero"
bash $H/run_valid40_locked.sh $C $T
python3 $H/valid40_vs_ref_ci.py $H/paired_valid40_$T.json $JREF | tee $L/valid40_${T}_vs_S21s1.txt
python3 $H/valid40_bar.py $H/paired_valid40_$T.json $JREF -1.5 0 | tee $L/valid40_${T}_bar.json
echo "$(date +%T) VALID40_DONE"
yield; bash $E/heavy_gate.sh 12 "residual gate s1zero"
bash $L/resid_gate_v1.sh $C S21s1z $REF
yield; bash $E/heavy_gate.sh 7 "residual probe S21s1"
bash $E/resid/run_probe_v6s.sh $REF 35,100 S21s1_r35r100
echo "$(date +%T) AFTER_S22_DONE"
