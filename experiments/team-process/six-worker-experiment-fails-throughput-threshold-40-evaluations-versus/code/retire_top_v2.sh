#!/bin/bash
# TAG opus-trainer RAD (written 2026-10-12T22:06Z; lead 22:0xZ): = cert/RAD/S51/retire_top_v1.sh for any round and cap: lowers cert/RAD/WORKER_CAP
# to <cap> with the round's live indices >= <cap> retiring AFTER their current eval (maintain_workers_v5 alone would kill them mid-eval):
# (1) cool_<i> = +24 h in the round's maintainer dir for every index >= <cap> (v5 relaunches no cooled index); (2) per live index >= <cap>: wait for
# the result of its open claim, stop the worker, put the claim it took next back in the queue at once; (3) archive cert/RAD/WORKER_CAP, write <cap>;
# one TEAM_LOG line. Builtins only in the polls.
# usage (from O/trainer): nohup bash cert/RAD/retire_top_v2.sh <round, e.g. S52> <cap> "<reason>" > cert/RAD/<round>/retire_top_v2.log 2>&1 &
cd "$(dirname "$0")/../.."
O=..; NAME=$1; CAP=$2; WHY=$3; RUN=run_RAD_${NAME}_gen; M=cert/RAD/$NAME/maintain_v5
log() { echo "[trainer] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*" >> $O/TEAM_LOG.md; echo "$(date -u +%H:%M:%SZ) $*"; }
pid_of() { ps -axo pid=,command= | awk -v r="$RUN" -v i=$1 '$3 == "-B" && $4 ~ /^worker_generated_v[0-9]+\.py$/ && $5 == r && $6 == i {print $1}'; }
answered() { [ -e $RUN/results/$1.json ] || [ -e $RUN/results/$1.pt ] || [ -e $RUN/results/$1.error ]; }
open_claims() { for c in $RUN/claimed/*.w$1.json; do [ -e "$c" ] || continue; c=${c##*/}; id=${c%.w$1.json}; answered $id || echo $id; done; }
FAR=$(( $(date +%s) + 86400 )); TOP=""
for i in $(seq $CAP 11); do echo $FAR > $M/cool_$i; [ -n "$(pid_of $i)" ] && TOP="$i $TOP"; done
for i in $TOP; do
  (
    P=$(pid_of $i); [ -n "$P" ] || exit 0
    C0=$(open_claims $i | head -1)
    while [ -n "$C0" ] && kill -0 $P 2>/dev/null && ! answered $C0; do sleep 2; done
    sleep 2; kill -TERM $P 2>/dev/null; sleep 5
    for id in $(open_claims $i); do mv $RUN/claimed/$id.w$i.json $RUN/queue/$id.json && echo "index $i: requeued $id"; done
    echo "$(date -u +%H:%M:%SZ) index $i retired after ${C0:-no open claim}"
  ) &
  sleep 1
done
wait
OLD=$(head -1 cert/RAD/WORKER_CAP 2>/dev/null | tr -dc 0-9)
[ -e cert/RAD/WORKER_CAP ] && cp cert/RAD/WORKER_CAP cert/RAD/WORKER_CAP.archived_$(date -u +%Y%m%dT%H%M%SZ)_cap$OLD
echo $CAP > cert/RAD/WORKER_CAP
alive=$(ps -axo pid=,command= | awk -v r="$RUN" '$3 == "-B" && $4 ~ /^worker_generated_v[0-9]+\.py$/ && $5 == r' | wc -l | tr -d ' ')
log "WORKER_CAP $OLD -> $CAP ($WHY): $NAME indices [${TOP% }] retired after their current evals (the claim each took next went back to the queue at once; cool files keep maintain_workers_v5 from relaunching them), $alive $NAME workers left; the old cap file archived. cert/RAD/retire_top_v2.sh."
