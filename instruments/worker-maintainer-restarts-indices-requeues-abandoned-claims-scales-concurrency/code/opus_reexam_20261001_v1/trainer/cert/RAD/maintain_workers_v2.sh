#!/bin/bash
# v2 (10-12 07:5xZ, for S52 onward): = v1 with (1) LAUNCHER (env; default cert/RAD/launch_gen_v7_t2.sh; S52: launch_gen_v8_t2.sh, 8M truths)
# and (2) HOLD (env, a shell command): while it fails the target is 2 (the round's start, lead 07:1xZ: S51's sealed stage 2 and port start
# on the same verdict file); once it succeeds and memory_pressure free has been >= 60% for 5 minutes in a row, the target goes to 6, then
# to 8 as in v1. No HOLD = v1 (target 6 from the start). Log lines name maintain_workers_v2.sh.
# TAG opus-trainer RAD (10-12 07:5xZ; lead 07:3xZ profile: each worker is bound by its main thread at ~0.8 core and the box has 12 performance
# cores, so a round's eval rate is a worker-COUNT limit): keeps the worker indices 0 .. T-1 of run_RAD_<NAME>_gen alive at 2 threads
# (cert/RAD/launch_gen_v7_t2.sh). T = 6 at the start; T = 8 once memory_pressure free has stayed >= 50% with system swap within +64 MiB
# for 15 min. At most one launch a minute, each behind gate.sh 6 (5-minute timeout, then retried) and memory_pressure free >= 50%.
# Why: guard_footprint stops every worker at 4 h (and at +256 MiB of swap). A dead index strands its claimed request: only a worker of the
# SAME index re-serves it, so the round's wait would never complete. A relaunch of that index serves the orphan first; an index that stays
# dead 10 min (gate shut, cooldown) has its open claim put back in the queue for the others.
# Yield: another lane's gate.sh seen waiting in two polls in a row -> stop the highest live index >= 4 (open claim requeued); no index >= 4
# is relaunched until no other lane's gate has been seen for 10 min. Swap: +150 MiB within 15 min -> stop the highest live index >= 4, that
# index cools 15 min. Indices 0 / 1 are left to the round's chain while fewer than 2 workers are alive (its own relaunch). Exits on the
# run's STOP file. First launches of an index, stops and the move to 8 are TEAM_LOG lines; relaunches go to the maintainer's log only.
# usage (from O/trainer): [LAUNCHER=...] [HOLD='...'] nohup bash cert/RAD/maintain_workers_v2.sh <NAME> > cert/RAD/<NAME>/maintain_v2.log 2>&1 &
cd "$(dirname "$0")/../.."
NAME=$1; RUN=run_RAD_${NAME}_gen; O=..; M=cert/RAD/$NAME/maintain_v2; mkdir -p $M; LAUNCHER=${LAUNCHER:-cert/RAD/launch_gen_v7_t2.sh}
log() { echo "[trainer] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*" >> $O/TEAM_LOG.md; echo "$(date -u +%H:%M:%SZ) $*"; }
say() { echo "$(date -u +%H:%M:%SZ) $*"; }
pid_of() { ps -axo pid=,command= | awk -v r="$RUN" -v i=$1 '$3 == "-B" && $4 ~ /^worker_generated_v[0-9]+\.py$/ && $5 == r && $6 == i {print $1}'; }
alive() { ps -axo pid=,command= | awk -v r="$RUN" '$3 == "-B" && $4 ~ /^worker_generated_v[0-9]+\.py$/ && $5 == r' | wc -l | tr -d ' '; }
swapmb() { sysctl -n vm.swapusage | sed -E 's/.*used = ([0-9]+)(\.[0-9]+)?M.*/\1/'; }
mp() { memory_pressure | tail -1 | grep -o '[0-9]*%' | tr -d '%'; }
foreign() { ps -axo pid=,command= | awk '$2 ~ /(^|\/)bash$/ && $3 ~ /gate\.sh$/ && $5 !~ /^trainer_/ {print $1}' | sort | tr '\n' ' '; }
requeue() {      # index $1's claims without a result go back to the queue
  for c in $RUN/claimed/*.w$1.json; do
    [ -e "$c" ] || continue
    id=$(basename "$c" | sed -E 's/\.w[0-9]+\.json$//')
    [ -e "$RUN/results/$id.pt" ] || [ -e "$RUN/results/$id.json" ] || [ -e "$RUN/results/$id.error" ] || { mv "$c" "$RUN/queue/$id.json" && say "requeued $id (index $1)"; }
  done
}
stop_top() {     # stop the highest live index >= 4: $1 = reason, $2 = cooldown seconds
  for i in 7 6 5 4; do
    P=$(pid_of $i); [ -n "$P" ] || continue
    kill -TERM $P; sleep 5; requeue $i; echo $(( $(date +%s) + $2 )) > $M/cool_$i
    log "$NAME workers: index $i stopped ($1), its open claim requeued; $(alive) workers left"; return 0
  done
  return 1
}
T=6; [ -n "$HOLD" ] && T=2; OKM=0; RAMP0=$(date +%s); RAMPS=$(swapmb); BASE=$RAMPS; BASET=$RAMP0; PREV=""; LASTF=0
say "maintainer v2 armed on $RUN: target $T (hold: ${HOLD:-none}), launcher $LAUNCHER, swap ${BASE}M, $(alive) workers alive"
while [ ! -e $RUN/STOP ]; do
  NOW=$(date +%s); F=$(foreign); S=$(swapmb)
  W=""; for p in $F; do case " $PREV " in *" $p "*) W="$W $p" ;; esac; done; PREV="$F"
  [ -n "$F" ] && LASTF=$NOW
  if [ -n "$W" ] && stop_top "yield: another lane's gate.sh waits, pid$W" 600; then sleep 60; continue; fi
  if [ $(( S - BASE )) -gt 150 ]; then stop_top "system swap ${BASE}M -> ${S}M within 15 min" 900; BASE=$S; BASET=$NOW; sleep 60; continue; fi
  [ $(( NOW - BASET )) -ge 900 ] && { BASE=$S; BASET=$NOW; }
  if [ $T -eq 2 ]; then
    if eval "$HOLD" > /dev/null 2>&1 && [ "$(mp)" -ge 60 ]; then OKM=$((OKM + 1)); else OKM=0; fi
    [ $OKM -ge 5 ] && { T=6; RAMP0=$NOW; RAMPS=$S; log "$NAME workers: hold over ($HOLD) and memory_pressure free >= 60% for 5 min: target 2 -> 6; cert/RAD/maintain_workers_v2.sh"; }
  elif [ $T -eq 6 ]; then
    if [ "$(mp)" -lt 50 ] || [ $(( S - RAMPS )) -gt 64 ]; then RAMP0=$NOW; RAMPS=$S
    elif [ $(( NOW - RAMP0 )) -ge 900 ]; then T=8; log "$NAME workers: target 6 -> 8 (memory_pressure free >= 50% and swap ${RAMPS}M -> ${S}M over 15 min); cert/RAD/maintain_workers_v2.sh"; fi
  fi
  for i in 0 1 2 3 4 5 6 7; do
    [ -n "$(pid_of $i)" ] && { rm -f $M/dead_$i; continue; }
    [ -e $M/dead_$i ] || echo $NOW > $M/dead_$i
    [ $(( NOW - $(cat $M/dead_$i) )) -ge 600 ] && requeue $i
  done
  for i in 0 1 2 3 4 5 6 7; do
    [ $i -lt $T ] || break
    [ -n "$(pid_of $i)" ] && continue
    [ $i -le 1 ] && [ "$(alive)" -lt 2 ] && continue
    [ -e $M/cool_$i ] && [ "$(cat $M/cool_$i)" -gt $NOW ] && continue
    [ $i -ge 4 ] && [ $(( NOW - LASTF )) -lt 600 ] && continue
    [ "$(mp)" -ge 50 ] || break
    bash $O/gate.sh 6 trainer_${NAME}_w$i > /dev/null & G=$!
    k=0; while kill -0 $G 2>/dev/null && [ $k -lt 300 ]; do sleep 5; k=$((k + 5)); done
    if kill -0 $G 2>/dev/null; then kill $G 2>/dev/null; say "gate for index $i shut for 5 min; retry next minute"; break; fi
    wait $G || break
    [ -e $RUN/STOP ] && break
    (nohup bash $LAUNCHER $i $NAME > /dev/null 2>&1 &)
    if [ -e $M/launched_$i ]; then say "index $i relaunched (target $T; $(alive) were alive)"
    else touch $M/launched_$i; log "$NAME workers: index $i launched at 2 threads (target $T; $(alive) were alive; memory_pressure free $(mp)%, swap ${S}M; $(basename $LAUNCHER)); cert/RAD/maintain_workers_v2.sh"; fi
    break
  done
  sleep 60
done
say "STOP on $RUN: maintainer exits"
