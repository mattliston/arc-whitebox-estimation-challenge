#!/bin/bash
# TAG fable-fire (FF-1): QUIET-WINDOW boundary census. Pauses (SIGSTOP) every other heavy python job of this user, runs engineer2's
# run_census.sh (numpy-free client + OpenBLAS server, 1 thread) on the bank and on the fire file(s) over the same nets, then resumes
# (SIGCONT) everything, even on error. usage: quiet_census.sh <ranks e.g. 35,100,37> <cand1.py> [cand2.py ...]
P2=/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16
E=$P2/fable_engineer2_v1; OUT=$P2/fable_fire_v1; RANKS=$1; shift
BANK=$P2/codex_resume_20260919_v1/current_p_message_covariance_release_v1/candidate.py
STOPPED=$(ps -axo pid=,nice=,rss=,command= | awk '$2>=10 && $3>300000 && ($0 ~ /python/) {print $1}' | tr '\n' ' ')
echo "$(date '+%H:%M:%S') pausing: $STOPPED" | tee -a $OUT/quiet_census.log
resume() { for p in $STOPPED; do kill -CONT $p 2>/dev/null; done; echo "$(date '+%H:%M:%S') resumed" | tee -a $OUT/quiet_census.log; }
trap resume EXIT
for p in $STOPPED; do kill -STOP $p 2>/dev/null; done
sleep 5; uptime | tee -a $OUT/quiet_census.log
cd $E
for C in "$BANK" "$@"; do
  TAG=quiet_$(basename $C .py)
  echo "=== $(date '+%H:%M:%S') $C" | tee -a $OUT/quiet_census.log
  bash ./run_census.sh "$C" $RANKS $TAG 2>&1 | tee -a $OUT/quiet_census.log | tail -12
  cp -f $E/${TAG}_census.json $OUT/ 2>/dev/null
done
