#!/bin/bash
# dissect337259 agent: the J5 join arm alone (one process, 4 BLAS threads), seed 7100005 then 7100006 if the first is clean.
cd "$(dirname "$0")/.." || exit 1
PY=/Users/matt/arc-white-box-estimation/.venv/bin/python
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
date -u
$PY -B eval_fresh_arm_v3.py agent_ideas_v1/candidate_j5join_sci_v1.py 7100005 2 '[{}]' > agent_ideas_v1/eval_j5join_v1.log 2>&1
date -u; echo RUN_J5_DONE
