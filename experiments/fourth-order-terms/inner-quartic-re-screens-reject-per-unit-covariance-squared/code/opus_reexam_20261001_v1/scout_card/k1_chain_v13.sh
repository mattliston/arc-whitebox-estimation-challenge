# v13 (2026-10-13 10:5xZ): = k1_chain_v12.sh running O/scout_card/worker_sc11.py (otr_m_sc7: + family c2k).
# v12 (2026-10-13 06:4xZ): = k1_chain_v10.sh running O/scout_card/worker_sc10.py (otr_m_sc6: + family e34).
# v110 (00:3xZ): = k1_chain_v8.sh under release/guard_footprint_v3.py (lead 00:3xZ), record names guard_v3_*.
#!/bin/bash
# v8: = k1_chain_v2.sh running O/scout_card/worker_sc8.py (runtime sc4: every O/scout_card/cutsim_sc<k>).
# TAG scout-card K1 (2026-10-12): two guarded workers on O/scout_card/<run dir> = the trainer's launch_worker_v13_t2_g2.sh line with my run dir,
# my guard records and an explicit wait for memory_pressure free >= 55% (lead's rule for scouts); the trainer's worker_generated_v13.py is used
# READ-ONLY (cwd = O/trainer, as its imports need). Allow list = trainer/cert/RAD/gen_allow_v7.txt (A36 + A39 + A41 + A42 check; no sealed set).
# usage: bash k1_chain_v1.sh <run dir name under O/scout_card> <workers 1|2> <guard seconds>.  Stop: touch O/scout_card/<run>/STOP
O=/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1
RUN=$1; NW=${2:-2}; SECS=${3:-10800}
cd $O/trainer
for W in $(seq 0 $((NW - 1))); do
  [ -f $O/scout_card/$RUN/STOP ] && exit 0
  until [ "$(memory_pressure | tail -1 | grep -o '[0-9]*%' | tr -d '%')" -ge 55 ]; do echo "$(date -u +%T) wait memory_pressure < 55%"; sleep 60; done
  bash $O/gate.sh 4 scout_card_${RUN}_w$W
  STAMP=$(date -u +%Y%m%dT%H%M%SZ)
  nice -n 10 /Users/matt/anaconda3/bin/python -B ../release/guard_footprint_v3.py --record $O/scout_card/$RUN/guard_v3_w${W}_$STAMP.json --seconds $SECS \
    /usr/bin/env -u TRAINER_TRUTH_PLAN -u TRAINER_TRUTH_PLAN_SHA256 TRAINER_CARD=M TRAINER_PL=1 TRAINER_F4C=1 TRAINER_THREADS=2 \
    TRAINER_GENERATED_ALLOW_FILE=cert/RAD/gen_allow_v7.txt TRAINER_GENERATED_MIN_N=15990784 TRAINER_RELIEF=1 TRAINER_STAR_ATTACH=0 TRAINER_BIRTH_ATTACH=0 \
    TRAINER_RSS_LIMIT=7500000000 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 \
    PYTHONPATH=$O/trainer /Users/matt/anaconda3/bin/python -B $O/scout_card/worker_sc11.py $O/scout_card/$RUN $W >> $O/scout_card/$RUN/worker_w$W.log 2>&1 &
  sleep 20
done
# the workers idle on an empty queue until STOP: touch it once every queued request has a result or an error
until [ -f $O/scout_card/$RUN/STOP ] || [ -z "$(ls $O/scout_card/$RUN/queue)" -a $(ls $O/scout_card/$RUN/results | grep -cE 'json$|error$') -ge $(ls $O/scout_card/$RUN/claimed | grep -c 'json$') ]; do sleep 30; done
touch $O/scout_card/$RUN/STOP
wait
echo "$(date -u +%FT%TZ) $RUN done: $(ls $O/scout_card/$RUN/results | grep -c 'json$') results, $(ls $O/scout_card/$RUN/results | grep -c 'error$') errors" >> $O/scout_card/chain.log
