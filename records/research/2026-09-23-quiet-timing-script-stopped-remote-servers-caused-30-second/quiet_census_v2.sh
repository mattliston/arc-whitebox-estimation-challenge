#!/bin/bash
# TAG fable-fire (FF-2): QUIET-WINDOW boundary census, v2 (v1 = quiet_census.sh, kept). Pauses (SIGSTOP) heavy python jobs of this
# user EXCEPT flopscope lanes: a paused flopscope server (or its client) dies, because the client aborts any request after 30 s.
# Excluded from the pause: every process with the ZeroMQ library loaded (flopscope servers AND clients; in-process flopscope runs
# have no server and no timeout, so they may pause). Lanes left running are logged as load context. usage: quiet_census_v2.sh <ranks> <cand.py>...
P2=/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16
E=$P2/fable_engineer2_v1; OUT=$P2/fable_fire_v1; RANKS=$1; shift
BANK=$P2/codex_resume_20260919_v1/current_p_message_covariance_release_v1/candidate.py
CAND=$(ps -axo pid=,nice=,rss=,comm= | awk '$2>=10 && $3>300000 && $4 ~ /python/ {print $1}')
STOPPED=""; LANES=""
for p in $CAND; do if lsof -p $p 2>/dev/null | grep -qi zmq; then LANES="$LANES $p"; else STOPPED="$STOPPED $p"; fi; done
SRV="$LANES"
echo "$(date '+%H:%M:%S') v2 pausing: $STOPPED | flopscope lanes left running (servers): $SRV" | tee -a $OUT/quiet_census.log
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
