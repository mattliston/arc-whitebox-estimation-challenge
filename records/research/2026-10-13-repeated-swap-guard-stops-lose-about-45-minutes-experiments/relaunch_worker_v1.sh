#!/bin/bash
# TAG scout-card (2026-10-12 23:4xZ, lead: workers killed by guard_footprint_v2's swap-growth limit at 23:39Z): relaunch ONE guarded worker (index 0)
# on an existing run dir whose chain is still alive in its watcher loop (that chain still touches STOP when the queue empties). The new guard
# re-baselines the system swap at its own start. The worker first re-serves the claim its killed predecessor left unanswered.
# usage: bash relaunch_worker_v1.sh <run dir name> <worker script name, e.g. worker_sc7.py>
O=/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1
RUN=$1; WS=$2; SECS=${3:-14400}
cd $O/trainer
until [ "$(memory_pressure | tail -1 | grep -o '[0-9]*%' | tr -d '%')" -ge 55 ]; do echo "$(date -u +%T) wait memory_pressure < 55%"; sleep 60; done
bash $O/gate.sh 4 scout_card_${RUN}_w0_relaunch
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
nice -n 10 /Users/matt/anaconda3/bin/python -B ../release/guard_footprint_v2.py --record $O/scout_card/$RUN/guard_w0_$STAMP.json --seconds $SECS \
  /usr/bin/env -u TRAINER_TRUTH_PLAN -u TRAINER_TRUTH_PLAN_SHA256 TRAINER_CARD=M TRAINER_PL=1 TRAINER_F4C=1 TRAINER_THREADS=2 \
  TRAINER_GENERATED_ALLOW_FILE=cert/RAD/gen_allow_v7.txt TRAINER_GENERATED_MIN_N=15990784 TRAINER_RELIEF=1 TRAINER_STAR_ATTACH=0 TRAINER_BIRTH_ATTACH=0 \
  TRAINER_RSS_LIMIT=7500000000 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 \
  PYTHONPATH=$O/trainer /Users/matt/anaconda3/bin/python -B $O/scout_card/$WS $O/scout_card/$RUN 0 >> $O/scout_card/$RUN/worker_w0.log 2>&1
