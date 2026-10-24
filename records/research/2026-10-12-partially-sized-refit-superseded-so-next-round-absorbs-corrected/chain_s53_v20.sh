#!/bin/bash
# chain_s53_v20.sh (written 2026-10-12T22:44:49Z, engineer2 22:4xZ): = chain_s53_v19.sh with the exporter export_dials_s46_v8.py (k31l instances from the round's config
# carrier list, e.g. an absorbed K31L4 age 4; byte for byte v7 for the default 3-age list). cert/RAD/fix_chain_v18.py. Nothing else changed.
# chain_s53_v19.sh (written 2026-10-12T18:46:21Z, lead 18:3xZ): = chain_s53_v18.sh with the make file make_s53_v11.py (= make_s53_v10.py + the ABSORB hook, cert/RAD/absorb_v1.py, inert while
# cert/RAD/ABSORB.json is absent, + the fagg fix of the strength-arm name parse). cert/RAD/fix_chain_v17.py. Nothing else changed.
# chain_s53_v18.sh (written 2026-10-12T18:34:36Z, lead 18:1xZ): = chain_s53_v17.sh with the make file make_s53_v10.py (= make_s53_v9.py + the ZERO-ARM GUARD, cert/RAD/zero_arms_v1.py:
# arms equal to the start point bit for bit leave arms.json 'live' before any is queued). cert/RAD/fix_chain_v16.py. Nothing else changed.
# chain_s53_v17.sh (written 2026-10-12T17:38:38Z, engineer 17:3xZ): = chain_s53_v16.sh with the port rule and the hold (hold_s53_v3.sh) matching TEAM_LOG lines
# only at the line start. cert/RAD/fix_chain_v15.py.
# chain_s53_v16.sh (written 2026-10-12T17:26:51Z, engineer 17:22:55Z): = chain_s53_v15.sh with the exporter export_dials_s46_v7.py (per-layer x2a1 to the port).
# cert/RAD/fix_chain_v14.py.
# chain_s53_v15.sh (written 2026-10-12T17:13:47Z, radlab): = chain_s53_v14.sh with the make file make_s53_v9.py (radlab/ptload.py pinned too).
# cert/RAD/fix_chain_v13.py.
# chain_s53_v14.sh (written 2026-10-12T17:08:32Z, radlab 17:0xZ): = chain_s53_v13.sh with the make file make_s53_v8.py (rule (d) from radlab/aggregate_spec_v1.py).
# cert/RAD/fix_chain_v12.py.
# chain_s53_v13.sh (written 2026-10-12T17:01:48Z, lead 17:0xZ): = chain_s53_v12.sh with the make file make_s53_v7.py (rule (d): one aggregate weak-band arm).
# cert/RAD/fix_chain_v11.py.
# chain_s53_v12.sh (written 2026-10-12T16:52:12Z, lead / radlab 16:5xZ): = chain_s53_v11.sh with the make file make_s53_v6.py (rule (b) routes from
# cert/RAD/ROUTES.json: k4 bands 1-7 / 8-11 / 12-14 / 15-15 / all). cert/RAD/fix_chain_v10.py.
# chain_s53_v11.sh (written 2026-10-12T16:44:53Z, lead 16:5xZ): = chain_s53_v10.sh with the make file make_s53_v5.py (start point by cert/RAD/CONTINUE_RULE; default
# = the -SE rule as before). cert/RAD/fix_chain_v9.py.
# chain_s53_v10.sh (written 2026-10-12T16:32:46Z, lead 16:5xZ): = chain_s53_v9.sh with the make file make_s53_v4.py (radlab's design: rule (c) off, escal out) and
# phase 2 only if $D/PHASE2_ON exists. cert/RAD/fix_chain_v8.py.
# chain_s53_v9.sh (written 2026-10-12T16:12:22Z, lead 15:2xZ): = chain_s53_v8.sh with its workers on worker_generated_v13.py (no weight cache on disk).
# cert/RAD/fix_chain_v6.py.
# chain_s53_v8.sh (written 2026-10-12T14:51:52Z, lead 14:5xZ): = chain_s53_v7.sh with the maintainer maintain_workers_v5.sh (cap and launcher-flavor files).
# cert/RAD/fix_chain_v5.py.
# chain_s53_v7.sh (written 2026-10-12T13:24:09Z, lead 13:3xZ): = chain_s53_v6.sh with the maintainer maintain_workers_v4.sh (up to 12 workers) and the hold script hold_s53_v2.sh (rule in cert/RAD/HOLD_RULE).
# cert/RAD/fix_chain_v4.py.
# chain_s53_v6.sh (written 2026-10-12T11:09:46Z, lead 11:0xZ): = chain_s53_v5.sh with the maintainer maintain_workers_v3.sh (up to 10 workers) and phase 2 only
# when all its truths already exist. cert/RAD/fix_chain_v3.py.
# chain_s53_v5.sh (written 2026-10-12T08:28:49Z, release 08:18Z / lead 08:0xZ): = chain_s53_v4.sh with the relaunch cap 12 -> 1000 and every worker under
# guard_footprint_v2 (launch_gen_v<K>_g2.sh / _t2_g2.sh). cert/RAD/fix_chain_v2.py.
# chain_s53_v4.sh (written 2026-10-12T08:06:41Z, lead 08:0xZ): = chain_s53_v3.sh with every failure exit writing STOP (fail()), so the next round always follows and starts
# from the last ACCEPTED point; a generated round starts no later than 10-17 03:00Z. cert/RAD/fix_chain_v1.py.
# v3 (10-12 07:5xZ): rules cert/RAD/S53/make_s53_v3.py (s_carry per layer: the engineer's certified DLY route) and export_dials_s46_v6.py
# (accepts a per-layer s_carry). Otherwise = v2.
# v2 (10-12 08:1xZ, lead 07:5xZ): the rules are cert/RAD/S53/make_s53_v2.py (dial families and bands by the engineer's native routes; strength
# arms whose S52 port failed are left out) and the export is export_dials_s46_v5.py (K4L age 2 by layer). Otherwise = v1.
# TAG opus-trainer RAD S53 (lead 10-12 07:4xZ: APPROVED as proposed 07:3xZ; = S52's chain_s52_v6.sh with S53's data and rules): the UNATTENDED chain of
# round S53 (rules in cert/RAD/S53/make_s53.py; keep_v1.json + RULES_v1.json before any arm result; port rule cert/RAD/S53/PORT_RULE_v1.json).
# Waits for S52's chain (any chain_s52_v<N>.sh) to end with run_RAD_S52_gen/STOP written (S52 really ran) and S52's workers to stop -> config (start = latest accepted of S52 / S51 / S50) -> 64
# gradients at the start on A36 -> arms by rule -> phased sizing (phase 1 = A48 7101032-095 + A50 7101160-223; phase 2 = A51 7101224-287 + A53
# 7101352-415 only if the freeze projected from phase 1's measured rate lands within 12 h of the round's start; every range queued on its own: the
# blocks between them are S52's / S53's check and sealed sets) -> port rule -> freeze (rad_v6: truth-noise-corrected %) -> export (jmS53_v1) ->
# check on A52 7101288-319 -> verdict -> stop. Workers: 0 / 1 by the chain (launch_gen_v9.sh, 8M truths, allow list gen_allow_v9.txt), the rest by
# cert/RAD/maintain_workers_v2.sh (launch_gen_v9_t2.sh): 2 while cert/RAD/S53/hold_s53.sh holds, then 6 -> 8; an .error result stops the chain.
# usage (from O/trainer): nohup bash cert/RAD/S53/chain_s53.sh > cert/RAD/S53/chain_s53.log 2>&1 &
cd "$(dirname "$0")/../../.."
O=..; PY=/Users/matt/anaconda3/bin/python; RAD=cert/RAD/rad_v6.py; D=cert/RAD/S53; CFG=$D/config.json; RUN=run_RAD_S53_gen
log() { echo "[trainer] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*" >> $O/TEAM_LOG.md; echo "$(date -u +%H:%M:%SZ) $*"; }
fail() { log "S53 CHAIN STOPPED: $1; STOP written, the next round follows"; mkdir -p $RUN; date -u > $RUN/STOP; exit 1; }
truths() { for r in "$@"; do for x in $(seq -f %.0f ${r%-*} ${r#*-}); do [ -e $O/analyst/fresh/truth_seed$x.npz ] || return 1; done; done; }      # v3 fix
alive() { ps -axo pid=,command= | awk -v r="$RUN" '$3 == "-B" && $4 ~ /^worker_generated_v[0-9]+\.py$/ && $5 == r' | wc -l | tr -d ' '; }
up() { ps -axo pid=,command= | awk -v r="$RUN" -v i=$1 '$3 == "-B" && $4 ~ /^worker_generated_v[0-9]+\.py$/ && $5 == r && $6 == i' | wc -l | tr -d ' '; }
launch() {
  for i in 0 1; do [ "$(up $i)" -eq 0 ] && { bash $O/gate.sh 9 trainer_S53_w$i > /dev/null && (nohup bash cert/RAD/launch_gen_v9_g2_m.sh $i S53 > /dev/null 2>&1 &); sleep 20; }; done
}
RELAUNCH=0
wait_for() {   # wait_for <grep pattern of result files> <count>
  until [ $(ls $RUN/results | grep -c "$1") -ge $2 ]; do
    if ls $RUN/results | grep -q '\.error$'; then log "S53 CHAIN STOPPED: error results in trainer/$RUN/results; workers stopped"; date -u > $RUN/STOP; exit 1; fi
    if [ "$(alive)" -lt 2 ]; then
      RELAUNCH=$((RELAUNCH + 1)); [ $RELAUNCH -gt 1000 ] && fail "workers keep dying (1000 relaunches)"
      [ "$(alive)" -eq 0 ] && log "S53 chain: no worker alive, relaunch $RELAUNCH of 1000"; launch
    fi
    sleep 30
  done
}
mkdir -p $D
until [ $(ps -axo command= | awk '$1 == "bash" && $2 ~ /^cert\/RAD\/S52\/chain_s52_v[0-9]+\.sh$/' | wc -l) -eq 0 ] && [ -e run_RAD_S52_gen/STOP ]; do sleep 60; done      # S52 ran and ended
until [ $(ps -axo pid=,command= | awk '$3 == "-B" && $4 ~ /^worker_generated_v[0-9]+\.py$/ && $5 == "run_RAD_S52_gen"' | wc -l) -eq 0 ]; do sleep 20; done
T_START=$(date +%s)
[ -e $CFG ] || { $PY -B $D/make_s53_v11.py config > $D/config_v1.out 2>&1 || fail "config failed (trainer/$D/config_v1.out)"; }
$PY -B $RAD grads $CFG || fail "gradient queue failed"
launch
(bash cert/RAD/swap_guard_v2.sh $RUN >> $D/swap_guard_v2.log 2>&1 &); (LAUNCHER=cert/RAD/launch_gen_v9_t2_g2_m.sh HOLD="bash $D/hold_s53_v3.sh $T_START" nohup bash cert/RAD/maintain_workers_v5.sh S53 > $D/maintain_v5.log 2>&1 &)
log "S53 CHAIN STARTED (trainer/cert/RAD/S53/chain_s53_v3.sh; rules in make_s53_v3.py; workers: maintain_workers_v2.sh, 2 until hold_s53.sh, then 6 -> 8): $(tail -1 $D/config_v1.out); 64 gradients at the start on A36; $(alive) workers"
wait_for '^a_RS53g_.*\.pt$' 64
$PY -B $D/make_s53_v11.py arms > $D/arms_v1.out 2>&1 || { log "S53 CHAIN STOPPED: arm build failed (trainer/$D/arms_v1.out)"; date -u > $RUN/STOP; exit 1; }
log "S53 ARMS (keep_v1.json / RULES_v1.json written before any arm result): $(tail -1 $D/arms_v1.out | cut -c1-500)"
DIRS=$D/arms_grad_v1; [ -d $D/arms_fixed_v1 ] && DIRS=$DIRS,$D/arms_fixed_v1
NK=$($PY -c "import json; print(len(json.load(open('$D/keep_v1.json'))['keep']))")
queue_set() { $PY -B $RAD base $CFG $1 || fail "queue_set $1 failed"; $PY -B $RAD arms $CFG $D/arms_grad_v1 $1 || fail "queue_set $1 failed"; [ -d $D/arms_fixed_v1 ] && { $PY -B $RAD arms $CFG $D/arms_fixed_v1 $1 || fail "queue_set $1 failed"; }; }
T0=$(date +%s); queue_set 7101032-7101095; queue_set 7101160-7101223; N1=$(( (NK + 1) * 128 ))
log "S53 SIZING PHASE 1 queued: base + $NK arms on A48 7101032-095 + A50 7101160-223 (128) = $N1 evals"
wait_for '^a_RS53_.*\.json$' $N1
T1=$(date +%s); PROJ=$(( T1 + (T1 - T0) + 900 ))
if [ -e $D/PHASE2_ON ] && [ $PROJ -lt $(( T_START + 43200 )) ] && truths 7101224-7101287 7101352-7101415; then
  queue_set 7101224-7101287; queue_set 7101352-7101415; FCFG=$CFG
  log "S53 PHASE 1 DONE ($N1 evals in $(( (T1 - T0) / 60 )) min): projected 256-network freeze $(date -u -r $PROJ +%H:%MZ) within 12 h of the start -> PHASE 2 on A51 + A53 queued"
  wait_for '^a_RS53_.*\.json$' $(( (NK + 1) * 256 ))
else
  FCFG=$D/config_p1.json
  $PY -c "import json; c = json.load(open('$CFG')); c['sizing'] = [[7101032, 7101095], [7101160, 7101223]]; c['note'] += ' | config_p1: phase-1 sizing only (12 h rule).'; open('$FCFG', 'x').write(json.dumps(c, indent=1))"
  log "S53 PHASE 1 DONE ($N1 evals in $(( (T1 - T0) / 60 )) min): projected 256-network freeze $(date -u -r $PROJ +%H:%MZ) phase 2 off (no PHASE2_ON file), or beyond 12 h (or past 10-17 12:00Z, or phase-2 truths not all on disk) -> freeze on A48 + A50 (128)"
fi
FAILED=$(grep -oE '^\[engineer\] [^ ]+ S53 PORT PARITY [A-Za-z0-9_]+: FAILED' $O/TEAM_LOG.md | sed -E 's/.* S53 PORT PARITY ([A-Za-z0-9_]+): FAILED/\1/' | sort -u | tr '\n' ' ')      # anchored (fix_chain_v15)
$PY -c "import json, os; k = json.load(open('$D/keep_v1.json')); bad = '$FAILED'.split(); json.dump(dict(keep=[a for a in k['keep'] if a not in bad], excluded=bad, utc=os.popen('date -u +%FT%TZ').read().strip(), rule='PORT_RULE_v1.json'), open('$D/keep_v2.json', 'x'), indent=1)" || { log "S53 CHAIN STOPPED: keep_v2.json not written (exists already?); workers stopped"; date -u > $RUN/STOP; exit 1; }
log "S53 PORT RULE at the freeze (PORT_RULE_v1.json): excluded = [${FAILED}] (keep_v2.json)"
$PY -B $RAD freeze $FCFG $DIRS $D/keep_v2.json > $D/freeze_v1.out 2>&1 || { log "S53 FREEZE FAILED (trainer/$D/freeze_v1.out); workers stopped"; date -u > $RUN/STOP; exit 1; }
SUMMARY=$($PY -c "import json; f = json.load(open('$D/FROZEN_POINT_v1.json')); s = json.load(open('$D/solve_v1.json')); w = {k: round(v, 2) for k, v in f['weights'].items() if v}; print('LOO %+.3f%% (noise-corrected %+.3f%%), model %+.3f%%, set %s, ridge %s, null %s, weights %s' % (f['loo_pct'], s.get('loo_pct_nc', float('nan')), f['model_pct'], f['arm_set'], f['ridge'], f['null'], w))")
log "S53 FROZEN before any check (trainer/cert/RAD/S53/FROZEN_POINT_v1.json, config $(basename $FCFG)): $SUMMARY"
if grep -q '"null": true' $D/FROZEN_POINT_v1.json; then log "S53 NULL by the pre-declared rule (LOO >= 0): no check; workers stopped"; date -u > $RUN/STOP; exit 0; fi
$PY -B cert/RAD/make_exportable.py $D/point_v1/params.pt $D/point_v1/params_exportable_v1.pt > $D/export_v1.out 2>&1 && \
  $PY -B export_dials_s46_v8.py $D/point_v1/params_exportable_v1.pt candidates/jmS53_v1 "RAD round S53 (frozen; check pending): strengths / dials / arrays per FROZEN_POINT_v1.json; k4 + ssb + k21n + rsy read from the .pt; kappa dropped" >> $D/export_v1.out 2>&1 && \
  HEAD=$($PY -c "import json, hashlib; u = json.load(open('$D/point_v1/hu.json')); assert len(u) >= 31 and all(x == 0 for r in u[31:] for x in r), 'head rows >= 31 not zero'; t = json.dumps(u[:31]); open('candidates/jmS53_v1/head_weights_31x128.json', 'x').write(t); print('sha %s, %s' % (hashlib.sha256(t.encode()).hexdigest()[:12], 'rows = Nd / S50 head (03794864)' if t == open('candidates/jmNd/head_weights_31x128.json').read() else 'rows MOVED vs Nd / S50: ' + ' '.join(str(i) for i in range(31) if u[i] != json.load(open('candidates/jmNd/head_weights_31x128.json'))[i])))" 2>>$D/export_v1.out) && \
  log "S53 EXPORTED for the port (before the check): trainer/candidates/jmS53_v1/ from trainer/cert/RAD/S53/point_v1/params_exportable_v1.pt (export_dials_s46_v8: k4 / ssb / k21n / rsy read_from_pt, K4L age 2 by layer in the sidecar if it varies, s_carry by layer from the .pt, kappa dropped; the file's k21n literal is the base -0.028, the port adds the .pt's k21n shift per layer); head_weights_31x128.json = rows 0-30 of the frozen point_v1/hu.json ($HEAD)" || \
  log "S53 EXPORT FAILED (trainer/cert/RAD/S53/export_v1.out); the check continues"
$PY -B $RAD check $FCFG || fail "check queue failed"
wait_for '^a_RS53C_.*\.json$' 64
$PY -B $RAD verdict $FCFG > $D/verdict_v1.out 2>&1
log "S53 CHECK on A52 7101288-319 (trainer/cert/RAD/S53/CHECK_VERDICT_v1.json): $(tail -1 $D/verdict_v1.out)"
date -u > $RUN/STOP
log "S53 CHAIN DONE: workers stopped. If ACCEPTED: stage 2 = the physicist's sealed A52 7101320-351 (requests run_RAD_S53_gen/claimed/a_RS53C_ab_r7101288 / a_RS53C_pt_r7101288)."
