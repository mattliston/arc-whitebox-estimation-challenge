#!/bin/bash
# TAG scout-card (2026-10-13 00:3xZ, lead: liveness check after two guard cascades): every 60 s, for each live scout chain 'bash k1_chain_v<N>.sh <run>',
# if the run has already started a worker once (a "start" event in <run>/worker_w0.log), has no STOP, and no worker process (worker_sc<k>.py <run>)
# and no relaunch in progress for 5 consecutive checks, relaunch ONE worker under guard_footprint_v3 (relaunch_worker_v2.sh, worker script from the
# chain version) and post one TEAM_LOG line. bash 3.2 (no associative arrays): missing-counters are files in O/scout_card/liveness/.
O=/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1
S=$O/scout_card; L=$S/liveness; mkdir -p $L
post() { echo "[scout-card] $(date -u +%FT%TZ) $1" >> $O/TEAM_LOG.md; }
wsfor() { case $1 in v5) echo worker_sc5.py;; v6) echo worker_sc6.py;; v7) echo worker_sc7.py;; v8|v10) echo worker_sc8.py;; v9|v11) echo worker_sc9.py;; *) echo "";; esac; }
while [ ! -f $S/NO_LIVENESS ]; do
  ps -axo command | grep -E '^bash k1_chain_v[0-9]+\.sh run_[a-z0-9]+ ' | while read -r _ chain run _rest; do
    v=$(echo $chain | sed -n 's/k1_chain_\(v[0-9]*\)\.sh/\1/p'); ws=$(wsfor $v); [ -z "$ws" ] && continue
    if [ -f $S/$run/STOP ] || ! grep -q '"event": "start"' $S/$run/worker_w0.log 2>/dev/null \
       || pgrep -f "worker_sc[0-9]+\.py $S/$run" > /dev/null || pgrep -f "relaunch_worker_v[0-9]\.sh $run " > /dev/null; then
      echo 0 > $L/$run; continue; fi
    c=$(( $(cat $L/$run 2>/dev/null || echo 0) + 1 )); echo $c > $L/$run
    if [ $c -ge 5 ]; then
      nohup bash $S/relaunch_worker_v2.sh $run $ws > $S/$run/relaunch_liveness_$(date -u +%H%M).out 2>&1 &
      post "LIVENESS: $run's chain ($chain) had no worker for 5 min; relaunched one $ws under guard_footprint_v3 (O/scout_card/liveness_v1.sh)."
      echo -10 > $L/$run
    fi
  done
  sleep 60
done
