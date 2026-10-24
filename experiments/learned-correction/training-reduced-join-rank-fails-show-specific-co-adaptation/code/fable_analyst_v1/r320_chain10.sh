#!/bin/bash
# TAG fable-analyst (AN-70d): separates co-adaptation from general training. loopR7s epoch 1 on the ALL-384 card (B384), VALID40, torch
# float64. If B384 beats A (loopR7 epoch 1 at all-384) by as much as B beats C on the cut card, the loopR7s gain is general training, not
# co-adaptation to the L10 cut. Declared peak 5 GB (measured 3.0-3.35 GB).
cd "$(dirname "$0")"
date; bash ../fable_porter_v1/machine_gate.sh 5 analyst_loopR7s_ep1_all384 && R320_PARAMS=fable_physicist_v1/refit6b/s6b_loopR7s_joint_3ep R320_EPOCH=1 \
  nice -n 10 env OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 \
  python3 r320_features.py valid40 3 r320_refit_v40_loopR7s_ep1_all384.jsonl fixed384 > r320_refit_v40_loopR7s_ep1_all384.log 2>&1
date; echo "R320 CHAIN10 DONE"
