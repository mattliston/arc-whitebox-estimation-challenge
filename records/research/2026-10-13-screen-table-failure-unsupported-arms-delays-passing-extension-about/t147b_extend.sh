#!/bin/bash
# TAG opus-physicist T147b (lead 18:4xZ URGENT): T147's decide step crashed on the errored (j) / (k) arms, so (h)'s PRE-DECLARED extension (17:10:15Z rule:
# 8-network mean log <= -0.15% -> 7100592-607) never ran. This queues nothing itself (t147_make.py extend h was run by hand just before); it serves
# physicist/t147_bets with NW worker_sc8 processes (default 3; the run's old STOP archived as STOP_archived_<stamp>), waits for the 16 extension results,
# touches STOP, waits for the workers, runs t147_table.py -> t147_table_v2.out, posts to t116_chain.log and TEAM_LOG.
O=/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1
P=$O/physicist; LOG=$P/t116_chain.log; R=$P/t147_bets; NW=${1:-3}; PY=/Users/matt/anaconda3/bin/python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
cd $O/trainer
echo "$(date -u +%H:%M:%SZ) T147b start ($NW workers; (h) K31L4 age 4 at 0.6 on 7100592-607, the pre-declared extension)" >> $LOG
for w in $(seq 0 $((NW - 1))); do
  until [ "$(memory_pressure | tail -1 | grep -o '[0-9]*%' | tr -d '%')" -ge 55 ]; do sleep 30; done
  bash $O/gate.sh 4 physicist_t147b_w$w > /dev/null
  env -u TRAINER_TRUTH_PLAN -u TRAINER_TRUTH_PLAN_SHA256 TRAINER_CARD=M TRAINER_PL=1 TRAINER_F4C=1 TRAINER_THREADS=2 TRAINER_RELIEF=1 TRAINER_STAR_ATTACH=0 \
    TRAINER_BIRTH_ATTACH=0 TRAINER_RSS_LIMIT=7500000000 TRAINER_GENERATED_MIN_N=15990784 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
    VECLIB_MAXIMUM_THREADS=2 TRAINER_GENERATED_ALLOW_FILE=$R/allowed_seeds.txt PYTHONPATH=$O/trainer \
    nice -n 10 $PY -B $O/scout_card/worker_sc8.py $R 1$w >> $R/worker_w1$w.log 2>&1 &
  sleep 15
done
until [ $(ls $R/results | grep -cE '^ph147xh_r.*\.(json|error)$') -ge 16 ]; do sleep 30; done
touch $R/STOP
wait
echo "$(date -u +%H:%M:%SZ) T147b done ($(ls $R/results | grep -c '^ph147xh.*json$') (h) extension results, $(ls $R/results | grep -c '^ph147xh.*error$') errors)" >> $LOG
(cd $P && $PY -B t147_table.py > t147_table_v2.out 2>&1; cat t147_table_v2.out >> $LOG)
echo "[physicist] $(date -u +%Y-%m-%dT%H:%M:%SZ) T147b AUTO TABLE ((h) K31L4 age 4 at 0.6 extended to 24 as pre-declared 17:10:15Z; torch worker_sc8 at the S51 + gk point vs T141's C): $(tr '\n' ' ' < $P/t147_table_v2.out)" >> $O/TEAM_LOG.md
echo "$(date -u +%H:%M:%SZ) T147b ALL DONE" >> $LOG
