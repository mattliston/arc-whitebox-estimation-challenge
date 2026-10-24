#!/bin/bash
# TAG opus-trainer (2026-10-10): the JOINT REFIT (lead 10-10: learned arrays + the batch-6 head columns as zero-initialised trainable inputs at layers
# 1-14, 10-20 updates' worth, early stopping on the 16 line-search ranks, judged ONCE on the 50 against S42's arithmetic, handed over with parity
# predictions on 651 / 378), without anyone awake. Starts when the S43 driver has finished. Per step k (cert/refit/jmix.py):
#   gradients of 36 training networks at the current point (step 1: the 36 runs a_0z6g at S42's point; later steps: fresh batches of the training split)
#   -> four arms (learned arrays, column 6's weights, the weights of m2 / eold / qb, those of qold) at trial sizes, final predictions kept, 16 line-search ranks
#   -> sizes by least squares -> the new point run for real on the same ranks (it is the next step's base)
#   -> STOP when the new point is not better than the current one by two standard errors (the current point stands).
# Then: the final point ONCE on the 50 judging ranks against S42's arithmetic (a_0hS41_u2) and S43's (a_0s43c_j), with and without qold (lead 03:30Z:
# qold must buy -0.6% to be carried), predictions of all 16 layers on 651 / 378 for both, the export.
# v2 (10-10 04:20Z): starts from S43's point (cert/s43/s43c.pt; bsk 0.34 given as the extra pair-map entry, k31l age 3 in the file's dials) so that the
# final point is S44's card; qold its own arm; qold / eold no longer enter at layer 15 (otr_m v24: the native columns are layers 1-14).
# jmix5.sh (10-10 05:30Z): a fifth arm h20 = column 20 (K4L value on the response rows, layers 1-14; zero-weight table 21 of 128 beyond 3 se, mean t^2 4.6)
# from step 2 on; a restart at step K1 > 1 counts steps 1..K1-1 as accepted (the point it is given is the last accepted one).
# jmix7.sh (10-10 05:45Z, tag C = jmB continued): + arm h17 = column 17 at layer 15 (kept rows), started from the physicist's offline weights
# (lead 04:15Z), + arm kc = the dial kappa_C at layers 1-14 (lead 05:32Z); the arm list is ARMS below.
# jmix8.sh (10-10 06:17Z): per step only the LIVE arms run (cert/refit/jmix.py arms writes them to arms.json: a head arm needs 3 x the chance count
# of weights beyond 2.5 standard errors; arrays and kappa_C always run).
# jmix9.sh (10-10 06:50Z): the gradient batches leave out the 16 line-search ranks and the 50 judging ranks (until now they were drawn from the whole
# training split: step 2's batch held judging ranks 277 and 480; step 3's held 460 / 439 548 599 651, parked before they ran); a step waits for its
# batch's own size.
# usage (from O/trainer): nohup bash cert/refit/jmix.sh <tag> <params.pt> <hu.json> <first step> <last step> [<gradient label of the first step>] [<extra ncar entries, JSON>] >> cert/refit/jmix_<tag>.log 2>&1 &
PY=/Users/matt/anaconda3/bin/python

TAG=$1; P=$2; H=$3; K1=$4; K2=$5; G1=$6; export JM_EXTRA=${7:-[]}
window_open() { [ "$(grep -E -o '^\[lead\] [^ ]+ QUIET WINDOW (OPEN|CLOSED)' ../TEAM_LOG.md | grep -E -o 'QUIET WINDOW (OPEN|CLOSED)' | tail -1)" = "QUIET WINDOW OPEN" ]; }
log() { printf '%s\n' "[trainer] $(date -u '+%Y-%m-%d %H:%M') UTC — $1" >> ../TEAM_LOG.md; echo "$(date -u +%FT%TZ) logged: $(echo "$1" | cut -c1-300)"; }
n() { ls run_M/results | grep -E -c "$1"; }
queue() {   # id prefix, params, hu json, mode (eval / evalkeep / evalall / grad), ranks (comma list)
  $PY -B - "$@" <<'PYEOF'
import json, os, sys
from pathlib import Path
sys.path.insert(0, '.')
from otr_common import enqueue
prefix, params, hu, mode, ranks = sys.argv[1], str(Path(sys.argv[2]).resolve()), json.loads(Path(sys.argv[3]).read_text()), sys.argv[4], [int(x) for x in sys.argv[5].split(',')]
K = [dict(module='k31ly', age=1, k31=0.0, leaf='nkl1'), dict(module='k31ly', age=2, k31=1.0, leaf='nkl2'), dict(module='k31ly', age=3, k31=0.0, leaf='nkl3')]
req = dict(f4c=1, ss=0, hf=2, b6=1, ncar=[dict(module='n6y', dfit=2)] + K + json.loads(os.environ['JM_EXTRA']), hu=hu)
extra = dict(grad=dict(mode='grad', scale=1e8), eval=dict(mode='eval'), evalkeep=dict(mode='eval', keep_final=1), evalall=dict(mode='eval', keep_all=1))[mode]
print(prefix, sum(enqueue(Path('run_M').resolve(), dict(id='a_%s_r%04d' % (prefix, r), rank=r, params=params, **extra, **req)) for r in ranks), 'queued')
PYEOF
}
while pgrep -qf "cert/s43/driver.sh"; do sleep 30; done
LS16=$(cat cert/ls16_ranks.txt)
BATCHES=$($PY -c "
import sys, random; sys.path.insert(0, '.')
from otr_common import splits
used = {597, 165, 667, 25, 309, 65, 642, 125, 164, 288, 297, 330, 124, 662, 603, 50, 182, 323, 381, 90, 201, 208, 17, 563, 421, 154, 85, 594, 411, 115, 605, 678, 384, 647, 91, 60}
r = [x for x in splits()['train'] if x not in used]; random.Random(20261010).shuffle(r)
held = {int(x) for x in open('cert/ls16_ranks.txt').read().split(',')} | {int(x) for x in open('cert/runM_val50.txt').read().split(',')}
print(' '.join(','.join(str(x) for x in r[i:i + 36] if x not in held) for i in range(0, 36 * 12, 36)))")
read -a BATCH <<< "$BATCHES"
while window_open; do sleep 20; done
queue 0m${TAG}${K1}_base $P $H evalkeep $LS16
ACCEPTED=$((K1 - 1))
for k in $(seq $K1 $K2); do
  D=cert/refit/jm${TAG}_d$k; mkdir -p $D
  if [ "$k" = "$K1" ] && [ -n "$G1" ]; then GL=$G1; else GL=0m${TAG}${k}g; queue $GL $P $H grad ${BATCH[$((k - 1))]}; fi
  if [ "$GL" = "$G1" ]; then NB=36; else NB=$(echo ${BATCH[$((k - 1))]} | tr ',' '\n' | wc -l | tr -d ' '); fi
  until [ "$(n "^a_${GL}_r.*\.pt$")" -ge $NB ] && [ "$(n "^a_0m${TAG}${k}_base_r.*\.json$")" -ge 16 ]; do
    ls run_M/results | grep -E -q "^a_(${GL}|0m${TAG}${k}_)[^ ]*\.error$" && { log "JOINT REFIT $TAG STEP $k: A RUN FAILED: $(ls run_M/results | grep -E "^a_(${GL}|0m${TAG}${k}_).*\.error$" | head -1)"; exit 1; }
    sleep 20
  done
  [ -s $D/arms.json ] || $PY -B cert/refit/jmix.py arms $GL $P $H $D > $D/arms.out 2>&1 || { log "JOINT REFIT $TAG STEP $k: ARMS FAILED: $(tail -1 $D/arms.out | cut -c1-300)"; exit 1; }
  while window_open; do sleep 20; done
  ARMS=$($PY -c "import json; print(' '.join(json.load(open('$D/arms.json'))['live']))"); NA=$(echo $ARMS | wc -w); ARX=$(echo $ARMS | tr ' ' '|')
  for a in $ARMS; do queue 0m${TAG}${k}_$a $D/arm_$a.pt $D/arm_${a}_hu.json evalkeep $LS16; done
  until [ "$(n "^a_0m${TAG}${k}_(${ARX})_r.*\.json$")" -ge $((16 * NA)) ]; do sleep 20; done
  $PY -B cert/refit/jmix.py solve $D 0m${TAG}${k} > $D/solve.out 2>&1
  NP=cert/refit/jm${TAG}_p$k.pt; NH=cert/refit/jm${TAG}_p${k}_hu.json
  $PY -B cert/refit/jmix.py point $D $NP $NH > $D/point.out 2>&1 || { log "JOINT REFIT $TAG STEP $k: POINT FAILED: $(tail -1 $D/point.out | cut -c1-300)"; exit 1; }
  queue 0m${TAG}$((k + 1))_base $NP $NH evalkeep $LS16
  until [ "$(n "^a_0m${TAG}$((k + 1))_base_r.*\.json$")" -ge 16 ]; do sleep 20; done
  REAL=$($PY -B cert/fold/pairs.py cert/ls16_ranks.txt 0m${TAG}$((k + 1))_base:0m${TAG}${k}_base | sed 's/^[^ ]* *//')
  GO=$($PY -c "
import re; m = re.search(r': ([+-][0-9.]+)% \(se ([0-9.]+)', '''$REAL'''); print('yes' if m and float(m.group(1)) < -2 * float(m.group(2)) else 'no')")
  SOLVE=$(grep -v Warning $D/solve.out | tr '\n' ';' | tr -s ' ')
  if [ "$GO" != "yes" ]; then
    log "JOINT REFIT $TAG STEP $k: NOT ACCEPTED (early stop): the solved point on the 16 line-search ranks against the current one: $REAL. Solve: $SOLVE The point of step $((k - 1)) stands: $P."
    break
  fi
  log "JOINT REFIT $TAG STEP $k ACCEPTED (torch card of the start point's arithmetic; gradients $GL; live arms of: learned arrays / column 6 / m2 eold qb / qold / column 20 / column 17 / kappa_C; sizes by least squares on the 16 line-search ranks): $SOLVE the new point for real on the 16 against the current one: $REAL. Point: O/trainer/$NP + $NH."
  P=$NP; H=$NH; ACCEPTED=$k
done
[ "$ACCEPTED" = "0" ] && { log "JOINT REFIT $TAG: NO STEP ACCEPTED; S42 stands, nothing to judge."; exit 0; }
while window_open; do sleep 20; done
HQ=cert/refit/jm${TAG}_final_without_qold_hu.json
$PY -c "import json; U = json.load(open('$H')); U[27] = [0.0] * len(U[27]); json.dump(U, open('$HQ', 'w'))"
queue 0m${TAG}J $P $H eval $(cat cert/runM_val50.txt)
queue 0m${TAG}Jq $P $HQ eval $(cat cert/runM_val50.txt)
queue 0m${TAG}par $P $H evalall 651,378
queue 0m${TAG}parq $P $HQ evalall 651,378
until [ "$(n "^a_0m${TAG}Jq?_r.*\.json$")" -ge 100 ] && [ "$(n "^a_0m${TAG}parq?_r.*\.json$")" -ge 4 ]; do sleep 20; done
J50=$($PY -B cert/fold/pairs.py cert/runM_val50.txt 0m${TAG}J:0hS41_u2 | sed 's/^[^ ]* *//')
J25=$($PY -B cert/fold/pairs.py cert/runM_val25.txt 0m${TAG}J:0hS41_u2 | sed 's/^[^ ]* *//')
J43=$($PY -B cert/fold/pairs.py cert/runM_val50.txt 0m${TAG}J:0s43c_j | sed 's/^[^ ]* *//')
JQ=$($PY -B cert/fold/pairs.py cert/runM_val50.txt 0m${TAG}Jq:0hS41_u2 | sed 's/^[^ ]* *//')
QV=$($PY -B cert/fold/pairs.py cert/runM_val50.txt 0m${TAG}J:0m${TAG}Jq | sed 's/^[^ ]* *//')
J48=$($PY -B cert/fold/pairs.py cert/runM_val48_clean.txt 0m${TAG}J:0hS41_u2 | sed 's/^[^ ]* *//')
$PY -B export_dials.py $P candidates/jm${TAG} "JOINT REFIT $TAG (cert/refit/jmix.sh): the learned arrays after $ACCEPTED accepted steps on the S42 arithmetic; head-input weights in $H. Judged once on the 50 judging ranks against S42's arithmetic: $J50" > cert/refit/jm${TAG}_export.out 2>&1
cp $H candidates/jm${TAG}/head_weights_31x128.json 2>/dev/null
cp $HQ candidates/jm${TAG}/head_weights_31x128_without_qold.json 2>/dev/null
log "JOINT REFIT $TAG JUDGED ONCE ON THE 50 JUDGING RANKS (torch, forward only, final-layer MSE): against S42's arithmetic $J50 (first 25: $J25; on the 48 that were never in a gradient batch, i.e. without 277 and 480: $J48); against S43's candidate $J43; WITHOUT QOLD (row 27 zeroed) against S42's arithmetic $JQ; with qold against without: $QV. Hand-over: O/trainer/candidates/jm${TAG}/ (candidate.py = the RRPI source with the refitted learned arrays and S42's card-side literals; head_weights_31x128.json = the head-input weights in the row order of O/trainer/otr_m.py HEAD_COLUMNS: row 6 = the diagonal column, rows 20 / 21 / 27 / 28 / 29 = K4L value / m2 / qold / eold / qb, all at layers 1-14 on the response rows; row 17 = the law diagonal at layer 15 on the kept rows; head_weights_31x128_without_qold.json = the same with row 27 zeroed); bsk 0.34 (pbase) and k31l age 3 at 0.64 as in S43; kappa_C per layer in physics_dials.json; parity predictions of all 16 layers on 651 / 378: run_M/results/a_0m${TAG}par_r0651.json, _r0378.json (with qold), a_0m${TAG}parq_r0651.json, _r0378.json (without) (field 'prediction')."
