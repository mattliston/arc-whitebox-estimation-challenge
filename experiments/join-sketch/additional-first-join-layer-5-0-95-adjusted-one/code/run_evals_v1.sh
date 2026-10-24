#!/bin/bash
# dissect337259 agent 2026-10-20: ONE in-process evaluation at a time (4 BLAS threads). DMT arms (5 settings) then the J5 join arm, seed 7100005.
cd "$(dirname "$0")/.." || exit 1
PY=/Users/matt/arc-white-box-estimation/.venv/bin/python
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
date -u
$PY -B eval_fresh_arm_v2.py agent_ideas_v1/candidate_dmt_sci_v1.py 7100005 1 '[{"_DMT_S": 0.0, "_DMT_HOST": 0}, {"_DMT_S": 1.0, "_DMT_HOST": 0}, {"_DMT_S": 0.5, "_DMT_HOST": 0}, {"_DMT_S": -0.5, "_DMT_HOST": 0}, {"_DMT_S": 0.0, "_DMT_HOST": 1}]' > agent_ideas_v1/eval_dmt_v2.log 2>&1
date -u
$PY -B eval_fresh_arm_v2.py agent_ideas_v1/candidate_j5join_sci_v1.py 7100005 1 '[{}]' > agent_ideas_v1/eval_j5join_v1.log 2>&1
date -u; echo RUN_EVALS_DONE
