#!/bin/bash
# TAG fable-analyst (AN-69L): after r320_chain5.sh, gated: (a) torch HELD-OUT raw for the VALID40 winner 384/320/384 (S27t; native bills
# picked it: only the L10 cut pays), then (b) the two bill-map arms 384/352/384, 352/384/384 on VALID40 (decision-irrelevant: native bills
# -0.620% / -0.183% cannot beat -0.70% even at zero raw; run for the table). Supersedes r320_chain6.sh (stopped before launch).
cd "$(dirname "$0")"
until grep -q "R320 CHAIN5 DONE" r320_chain5.out 2>/dev/null; do sleep 30; done
T="nice -n 10 env OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 python3 r320_features.py"
date; bash ../fable_porter_v1/machine_gate.sh 8 analyst_r320_ho_s27t && $T heldout40 3 r320_features_ho40_sched2.jsonl sched384-320-384 > r320_features_ho40_sched2.log 2>&1
date; echo "R320 CHAIN7 HELDOUT DONE"
date; bash ../fable_porter_v1/machine_gate.sh 8 analyst_r320_billmap_arms && $T valid40 3 r320_features_v40_sched4.jsonl sched384-352-384,sched352-384-384 > r320_features_v40_sched4.log 2>&1
date; echo "R320 CHAIN7 DONE"
