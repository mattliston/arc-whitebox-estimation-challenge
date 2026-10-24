#!/bin/bash
# TAG opus-trainer RAD S54 (lead 10-12 08:0xZ: APPROVED as proposed; = S53's chain_s53_v3.sh one round later): the UNATTENDED chain of round S54
# (rules in cert/RAD/S54/make_s54.py = make_s53_v3.py's template and routes; keep_v1.json + RULES_v1.json before any arm result; port rule
# cert/RAD/S54/PORT_RULE_v1.json). Waits for S53's chain (chain_s53[_vN].sh) to end with run_RAD_S53_gen/STOP written (S53 really ran) and S53's
# workers to stop -> config (start = latest accepted of S53 / S52 / S51 / S50) -> 64 gradients at the start on A36 -> arms by rule -> phased
# sizing (phase 1 = A54 7101416-479 + A56 7101544-607; phase 2 = A57 7101608-671 + A59 7101736-799 only if the freeze projected from phase 1's
# measured rate lands within 12 h of the round's start; every range queued on its own: A55 / A58 between them are the check / sealed set and
# the reserve) -> port rule -> freeze (rad_v6: truth-noise-corrected %) -> export (jmS54_v1, exporter v6) -> check on A55 7101480-511 -> verdict
# -> stop. Workers: 0 / 1 by the chain (launch_gen_v10.sh, 8M truths, allow list gen_allow_v10.txt), the rest by cert/RAD/maintain_workers_v2.sh
# (launch_gen_v10_t2.sh): 2 while cert/RAD/S54/hold_s54.sh holds, then 6 -> 8; an .error result stops the chain.
# usage (from O/trainer): nohup bash cert/RAD/S54/chain_s54.sh > cert/RAD/S54/chain_s54.log 2>&1 &
cd "$(dirname "$0")/../../.."
O=..; PY=/Users/matt/anaconda3/bin/python; RAD=cert/RAD/rad_v6.py; D=cert/RAD/S54; CFG=$D/config.json; RUN=run_RAD_S54_gen
log() { echo "[trainer] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*" >> $O/TEAM_LOG.md; echo "$(date -u +%H:%M:%SZ) $*"; }
alive() { ps -axo pid=,command= | awk -v r="$RUN" '$3 == "-B" && $4 ~ /^worker_generated_v[0-9]+\.py$/ && $5 == r' | wc -l | tr -d ' '; }
up() { ps -axo pid=,command= | awk -v r="$RUN" -v i=$1 '$3 == "-B" && $4 ~ /^worker_generated_v[0-9]+\.py$/ && $5 == r && $6 == i' | wc -l | tr -d ' '; }
launch() {
  for i in 0 1; do [ "$(up $i)" -eq 0 ] && { bash $O/gate.sh 9 trainer_S54_w$i > /dev/null && (nohup bash cert/RAD/launch_gen_v10.sh $i S54 > /dev/null 2>&1 &); sleep 20; }; done
}
RELAUNCH=0
wait_for() {   # wait_for <grep pattern of result files> <count>
  until [ $(ls $RUN/results | grep -c "$1") -ge $2 ]; do
    if ls $RUN/results | grep -q '\.error$'; then log "S54 CHAIN STOPPED: error results in trainer/$RUN/results; workers stopped"; date -u > $RUN/STOP; exit 1; fi
    if [ "$(alive)" -lt 2 ]; then
      RELAUNCH=$((RELAUNCH + 1)); [ $RELAUNCH -gt 12 ] && { log "S54 CHAIN STOPPED: workers keep dying (12 relaunches)"; exit 1; }
      [ "$(alive)" -eq 0 ] && log "S54 chain: no worker alive, relaunch $RELAUNCH of 12"; launch
    fi
    sleep 30
  done
}
mkdir -p $D
until [ $(ps -axo command= | awk '$1 == "bash" && $2 ~ /^cert\/RAD\/S53\/chain_s53(_v[0-9]+)?\.sh$/' | wc -l) -eq 0 ] && [ -e run_RAD_S53_gen/STOP ]; do sleep 60; done      # S53 ran and ended
until [ $(ps -axo pid=,command= | awk '$3 == "-B" && $4 ~ /^worker_generated_v[0-9]+\.py$/ && $5 == "run_RAD_S53_gen"' | wc -l) -eq 0 ]; do sleep 20; done
T_START=$(date +%s)
[ -e $CFG ] || { $PY -B $D/make_s54.py config > $D/config_v1.out 2>&1 || { log "S54 CHAIN STOPPED: config failed (trainer/$D/config_v1.out)"; exit 1; }; }
$PY -B $RAD grads $CFG || exit 1
launch
(bash cert/RAD/swap_guard_v2.sh $RUN >> $D/swap_guard_v2.log 2>&1 &); (LAUNCHER=cert/RAD/launch_gen_v10_t2.sh HOLD="bash $D/hold_s54.sh $T_START" nohup bash cert/RAD/maintain_workers_v2.sh S54 > $D/maintain_v2.log 2>&1 &)
log "S54 CHAIN STARTED (trainer/cert/RAD/S54/chain_s54.sh; rules in make_s54.py; workers: maintain_workers_v2.sh, 2 until hold_s54.sh, then 6 -> 8): $(tail -1 $D/config_v1.out); 64 gradients at the start on A36; $(alive) workers"
wait_for '^a_RS54g_.*\.pt$' 64
$PY -B $D/make_s54.py arms > $D/arms_v1.out 2>&1 || { log "S54 CHAIN STOPPED: arm build failed (trainer/$D/arms_v1.out)"; date -u > $RUN/STOP; exit 1; }
log "S54 ARMS (keep_v1.json / RULES_v1.json written before any arm result): $(tail -1 $D/arms_v1.out | cut -c1-500)"
DIRS=$D/arms_grad_v1; [ -d $D/arms_fixed_v1 ] && DIRS=$DIRS,$D/arms_fixed_v1
NK=$($PY -c "import json; print(len(json.load(open('$D/keep_v1.json'))['keep']))")
queue_set() { $PY -B $RAD base $CFG $1 || exit 1; $PY -B $RAD arms $CFG $D/arms_grad_v1 $1 || exit 1; [ -d $D/arms_fixed_v1 ] && { $PY -B $RAD arms $CFG $D/arms_fixed_v1 $1 || exit 1; }; }
T0=$(date +%s); queue_set 7101416-7101479; queue_set 7101544-7101607; N1=$(( (NK + 1) * 128 ))
log "S54 SIZING PHASE 1 queued: base + $NK arms on A54 7101416-479 + A56 7101544-607 (128) = $N1 evals"
wait_for '^a_RS54_.*\.json$' $N1
T1=$(date +%s); PROJ=$(( T1 + (T1 - T0) + 900 ))
if [ $PROJ -lt $(( T_START + 43200 )) ]; then
  queue_set 7101608-7101671; queue_set 7101736-7101799; FCFG=$CFG
  log "S54 PHASE 1 DONE ($N1 evals in $(( (T1 - T0) / 60 )) min): projected 256-network freeze $(date -u -r $PROJ +%H:%MZ) within 12 h of the start -> PHASE 2 on A57 + A59 queued"
  wait_for '^a_RS54_.*\.json$' $(( (NK + 1) * 256 ))
else
  FCFG=$D/config_p1.json
  $PY -c "import json; c = json.load(open('$CFG')); c['sizing'] = [[7101416, 7101479], [7101544, 7101607]]; c['note'] += ' | config_p1: phase-1 sizing only (12 h rule).'; open('$FCFG', 'x').write(json.dumps(c, indent=1))"
  log "S54 PHASE 1 DONE ($N1 evals in $(( (T1 - T0) / 60 )) min): projected 256-network freeze $(date -u -r $PROJ +%H:%MZ) beyond 12 h -> freeze on A54 + A56 (128)"
fi
FAILED=$(grep '^\[engineer\] ' $O/TEAM_LOG.md | grep -oE 'S54 PORT PARITY [A-Za-z0-9_]+: FAILED' | sed -E 's/S54 PORT PARITY ([A-Za-z0-9_]+): FAILED/\1/' | sort -u | tr '\n' ' ')
$PY -c "import json, os; k = json.load(open('$D/keep_v1.json')); bad = '$FAILED'.split(); json.dump(dict(keep=[a for a in k['keep'] if a not in bad], excluded=bad, utc=os.popen('date -u +%FT%TZ').read().strip(), rule='PORT_RULE_v1.json'), open('$D/keep_v2.json', 'x'), indent=1)" || { log "S54 CHAIN STOPPED: keep_v2.json not written (exists already?); workers stopped"; date -u > $RUN/STOP; exit 1; }
log "S54 PORT RULE at the freeze (PORT_RULE_v1.json): excluded = [${FAILED}] (keep_v2.json)"
$PY -B $RAD freeze $FCFG $DIRS $D/keep_v2.json > $D/freeze_v1.out 2>&1 || { log "S54 FREEZE FAILED (trainer/$D/freeze_v1.out); workers stopped"; date -u > $RUN/STOP; exit 1; }
SUMMARY=$($PY -c "import json; f = json.load(open('$D/FROZEN_POINT_v1.json')); s = json.load(open('$D/solve_v1.json')); w = {k: round(v, 2) for k, v in f['weights'].items() if v}; print('LOO %+.3f%% (noise-corrected %+.3f%%), model %+.3f%%, set %s, ridge %s, null %s, weights %s' % (f['loo_pct'], s.get('loo_pct_nc', float('nan')), f['model_pct'], f['arm_set'], f['ridge'], f['null'], w))")
log "S54 FROZEN before any check (trainer/cert/RAD/S54/FROZEN_POINT_v1.json, config $(basename $FCFG)): $SUMMARY"
if grep -q '"null": true' $D/FROZEN_POINT_v1.json; then log "S54 NULL by the pre-declared rule (LOO >= 0): no check; workers stopped"; date -u > $RUN/STOP; exit 0; fi
$PY -B cert/RAD/make_exportable.py $D/point_v1/params.pt $D/point_v1/params_exportable_v1.pt > $D/export_v1.out 2>&1 && \
  $PY -B export_dials_s46_v6.py $D/point_v1/params_exportable_v1.pt candidates/jmS54_v1 "RAD round S54 (frozen; check pending): strengths / dials / arrays per FROZEN_POINT_v1.json; k4 + ssb + k21n + rsy read from the .pt; kappa dropped" >> $D/export_v1.out 2>&1 && \
  HEAD=$($PY -c "import json, hashlib; u = json.load(open('$D/point_v1/hu.json')); assert len(u) >= 31 and all(x == 0 for r in u[31:] for x in r), 'head rows >= 31 not zero'; t = json.dumps(u[:31]); open('candidates/jmS54_v1/head_weights_31x128.json', 'x').write(t); print('sha %s, %s' % (hashlib.sha256(t.encode()).hexdigest()[:12], 'rows = Nd / S50 head (03794864)' if t == open('candidates/jmNd/head_weights_31x128.json').read() else 'rows MOVED vs Nd / S50: ' + ' '.join(str(i) for i in range(31) if u[i] != json.load(open('candidates/jmNd/head_weights_31x128.json'))[i])))" 2>>$D/export_v1.out) && \
  log "S54 EXPORTED for the port (before the check): trainer/candidates/jmS54_v1/ from trainer/cert/RAD/S54/point_v1/params_exportable_v1.pt (export_dials_s46_v6: k4 / ssb / k21n / rsy read_from_pt, K4L age 2 by layer in the sidecar if it varies, s_carry by layer from the .pt, kappa dropped; the file's k21n literal is the base -0.028, the port adds the .pt's k21n shift per layer); head_weights_31x128.json = rows 0-30 of the frozen point_v1/hu.json ($HEAD)" || \
  log "S54 EXPORT FAILED (trainer/cert/RAD/S54/export_v1.out); the check continues"
$PY -B $RAD check $FCFG || exit 1
wait_for '^a_RS54C_.*\.json$' 64
$PY -B $RAD verdict $FCFG > $D/verdict_v1.out 2>&1
log "S54 CHECK on A55 7101480-511 (trainer/cert/RAD/S54/CHECK_VERDICT_v1.json): $(tail -1 $D/verdict_v1.out)"
date -u > $RUN/STOP
log "S54 CHAIN DONE: workers stopped. If ACCEPTED: stage 2 = the physicist's sealed A55 7101512-543 (requests run_RAD_S54_gen/claimed/a_RS54C_ab_r7101480 / a_RS54C_pt_r7101480)."
