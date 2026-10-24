#!/bin/bash
# TAG opus-physicist T134 chain (lead 10-12 23:4xZ GO: the learned-OFF screen runs on my single worker once T132's table lands): waits for t133e's
# 'T133e ALL DONE' (or its stop) in t116_chain.log, then serves physicist/t134_refit_screen (25 pre-claimed requests: control + base_off / so2m / so2t
# on 7100584-591) with the trainer's worker_generated_v13.py (weights regenerated in memory: no disk), index 0, nice 10, 2 threads, launched at
# memory_pressure free >= 55% and through the machine gate; then t134_table.py -> t134_table.out and t116_chain.log. Stop: STOP_T134 (before launch).
O=/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1
P=$O/physicist; LOG=$P/t116_chain.log
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
until grep -q "T133e ALL DONE\|T133e STOP" $LOG; do [ -f $P/STOP_T134 ] && exit 0; sleep 60; done
[ -f $P/STOP_T134 ] && { echo "$(date -u +%H:%M:%SZ) T134 STOP before launch" >> $LOG; exit 0; }
cd $O/trainer
until [ "$(memory_pressure | tail -1 | grep -o '[0-9]*%' | tr -d '%')" -ge 55 ]; do sleep 60; done
bash $O/gate.sh 4 physicist_t134 > /dev/null
echo "$(date -u +%H:%M:%SZ) T134 t134_refit_screen start (learned-OFF screen: so2m / so2t vs base_off, S51 + gk, 7100584-591)" >> $LOG
env -u TRAINER_TRUTH_PLAN -u TRAINER_TRUTH_PLAN_SHA256 TRAINER_CARD=M TRAINER_PL=1 TRAINER_F4C=1 TRAINER_THREADS=2 TRAINER_RELIEF=1 TRAINER_STAR_ATTACH=0 \
  TRAINER_BIRTH_ATTACH=0 TRAINER_RSS_LIMIT=7500000000 TRAINER_GENERATED_MIN_N=15990784 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  VECLIB_MAXIMUM_THREADS=2 TRAINER_GENERATED_ALLOW_FILE=../physicist/t134_refit_screen/allowed_seeds.txt \
  nice -n 10 /Users/matt/anaconda3/bin/python -B worker_generated_v13.py ../physicist/t134_refit_screen 0 >> ../physicist/t134_refit_screen/worker_w0.log 2>&1
echo "$(date -u +%H:%M:%SZ) T134 done ($(ls ../physicist/t134_refit_screen/results | grep -c 'json$') results, $(ls ../physicist/t134_refit_screen/results | grep -c 'error$') errors)" >> $LOG
(cd $P && /Users/matt/anaconda3/bin/python -B t134_table.py > t134_table.out 2>&1; cat t134_table.out >> $LOG)
echo "$(date -u +%H:%M:%SZ) T134 ALL DONE" >> $LOG
