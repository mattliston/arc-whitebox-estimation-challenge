#!/bin/bash
# v2 (10-12 04:58Z): RESUMES S51 after the 04:50Z / 04:55Z swap events (each worker's guard_footprint stops it at +256 MiB of system swap: with 4
# workers + other lanes' processes all of them were stopped twice): TWO workers (0 / 1, 8 threads), launched only when fewer than 2 are alive; no S50 waits;
# the config is written only if it does not exist. Everything after the gradients is the v1 chain.
# TAG opus-trainer RAD S51 (lead 10-12 ~03:0xZ): the UNATTENDED chain of round S51 (rules in cert/RAD/S51/make_s51.py, receipts written before any arm
# result). Waits for S50's chain to end -> config (start point by S50's verdict) -> 64 gradients at the start on A36 -> arms by rule -> base + arms on
# A39 + A41 (128) -> freeze -> (if not null) check on A42 7100648-679 -> verdict -> stop. 4 no-plan v11 workers (2 x 8 + 2 x 4 threads), swap guards;
# dead workers relaunched (at most 3 times); an .error result stops the chain. Each transition is a TEAM_LOG line stamped from date -u.
# usage (from O/trainer): nohup bash cert/RAD/S51/chain.sh > cert/RAD/S51/chain.log 2>&1 &
cd "$(dirname "$0")/../../.."
O=..; PY=/Users/matt/anaconda3/bin/python; D=cert/RAD/S51; CFG=$D/config.json; RUN=run_RAD_S51_gen
log() { echo "[trainer] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*" >> $O/TEAM_LOG.md; echo "$(date -u +%H:%M:%SZ) $*"; }
alive() { ps -axo pid=,command= | awk -v r="$RUN" '$3 == "-B" && $4 ~ /^worker_generated_v[0-9]+\.py$/ && $5 == r' | wc -l | tr -d ' '; }
launch() {      # v2: workers 0 / 1 (8 threads) only, each launched only if it is not alive
  for i in 0 1; do
    if [ $(ps -axo pid=,command= | awk -v r="$RUN" -v i=$i '$3 == "-B" && $4 ~ /^worker_generated_v[0-9]+\.py$/ && $5 == r && $6 == i' | wc -l) -eq 0 ]; then
      bash $O/gate.sh 9 trainer_S51_w$i > /dev/null && (nohup bash cert/RAD/launch_gen_v5.sh $i S51 > /dev/null 2>&1 &); sleep 20
    fi
  done
}
RELAUNCH=0
wait_for() {   # wait_for <grep pattern of result files> <count>
  until [ $(ls $RUN/results | grep -c "$1") -ge $2 ]; do
    if ls $RUN/results | grep -q '\.error$'; then log "S51 CHAIN STOPPED: error results in trainer/$RUN/results; workers stopped"; date -u > $RUN/STOP; exit 1; fi
    if [ "$(alive)" -lt 2 ]; then      # v2: keep two workers alive (the swap guards stop single workers)
      RELAUNCH=$((RELAUNCH + 1)); [ $RELAUNCH -gt 12 ] && { log "S51 CHAIN STOPPED: workers keep dying (12 relaunches)"; exit 1; }
      [ "$(alive)" -eq 0 ] && log "S51 chain: no worker alive, relaunch $RELAUNCH of 12"; launch
    fi
    sleep 30
  done
}
mkdir -p $D
[ -e $CFG ] || { $PY -B $D/make_s51.py config > $D/config_v1.out 2>&1 || { log "S51 CHAIN STOPPED: config failed (trainer/$D/config_v1.out)"; exit 1; }; }
$PY -B cert/RAD/rad.py grads $CFG || exit 1
launch
log "S51 CHAIN v2 RESUMED (trainer/cert/RAD/S51/chain_v2.sh) after the swap events: TWO workers x 8 threads; $(tail -1 $D/config_v1.out); gradients $(ls $RUN/results | grep -c '^a_RS51g_.*\.pt$') of 64 done"
wait_for '^a_RS51g_.*\.pt$' 64
$PY -B $D/make_s51.py arms > $D/arms_v1.out 2>&1 || { log "S51 CHAIN STOPPED: arm build failed (trainer/$D/arms_v1.out)"; date -u > $RUN/STOP; exit 1; }
log "S51 ARMS by rule (keep_v1.json / RULES_v1.json written before any arm result): $(tail -1 $D/arms_v1.out | cut -c1-600)"
$PY -B cert/RAD/rad.py base $CFG || exit 1
DIRS=$D/arms_grad_v1; [ -d $D/arms_fixed_v1 ] && DIRS=$DIRS,$D/arms_fixed_v1
$PY -B cert/RAD/rad.py arms $CFG $D/arms_grad_v1 || exit 1
[ -d $D/arms_fixed_v1 ] && { $PY -B cert/RAD/rad.py arms $CFG $D/arms_fixed_v1 || exit 1; }
NK=$($PY -c "import json; print(len(json.load(open('$D/keep_v1.json'))['keep']))")
N=$(( (NK + 1) * 128 ))
log "S51 SIZING queued: base + $NK arms on A39 + A41 (128 networks) = $N evals"
wait_for '^a_RS51_.*\.json$' $N
$PY -B cert/RAD/rad.py freeze $CFG $DIRS $D/keep_v1.json > $D/freeze_v1.out 2>&1 || { log "S51 FREEZE FAILED (trainer/$D/freeze_v1.out); workers stopped"; date -u > $RUN/STOP; exit 1; }
SUMMARY=$($PY -c "import json; f = json.load(open('$D/FROZEN_POINT_v1.json')); w = {k: round(v, 2) for k, v in f['weights'].items() if v}; print('LOO %+.3f%%, model %+.3f%%, set %s, ridge %s, null %s, weights %s' % (f['loo_pct'], f['model_pct'], f['arm_set'], f['ridge'], f['null'], w))")
log "S51 FROZEN before any check (trainer/cert/RAD/S51/FROZEN_POINT_v1.json): $SUMMARY"
if grep -q '"null": true' $D/FROZEN_POINT_v1.json; then log "S51 NULL by the pre-declared rule (LOO >= 0): no check; workers stopped"; date -u > $RUN/STOP; exit 0; fi
$PY -B cert/RAD/rad.py check $CFG || exit 1
wait_for '^a_RS51C_.*\.json$' 64
$PY -B cert/RAD/rad.py verdict $CFG > $D/verdict_v1.out 2>&1
log "S51 CHECK (trainer/cert/RAD/S51/CHECK_VERDICT_v1.json): $(tail -1 $D/verdict_v1.out)"
date -u > $RUN/STOP
log "S51 CHAIN DONE: workers stopped. If ACCEPTED: stage 2 = the physicist's sealed A42 7100680-711 (requests run_RAD_S51_gen/claimed/a_RS51C_ab_r7100648 / a_RS51C_pt_r7100648)."
