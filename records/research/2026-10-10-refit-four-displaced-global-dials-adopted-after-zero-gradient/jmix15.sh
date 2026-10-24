#!/bin/bash
# TAG opus-trainer (2026-10-10 08:05Z): jmix10.sh = the joint refit driver for a CONTINUATION from S44's torch point (jmD), after jmC (jmix9.sh):
#   sizing set = <sizing ranks file> (a fresh 16: never in a gradient batch, not the line-search 16, not the 50; the 16 line-search ranks sized
#   jmB / jmC and many earlier rounds), exported as JM_SIZING to cert/refit/jmix.py; kappa_C as two arms (layers 1-9 / 10-14, JM_KC_SPLIT=1);
#   gradient batches = the seed-20261010 batches from index <first batch> on (0-2 were used by jmB / jmC), minus the line-search 16, the 50 and
#   the sizing set; per step: gradients -> live arms on the sizing set -> least squares -> the point for real -> accepted only if better by two
#   standard errors (early stop). Then the final point ONCE on the 50 judging ranks (primary: the 48 never in a gradient batch), against S42's
#   arithmetic (a_0hS41_u2) and S44's torch point (a_0mCJq = jmC without qold), its predictions of all 16 layers on 651 / 378, the export.
# jmix11.sh (10-10 09:20Z, jmE): + env JM_HSPLIT (JSON {row: [first layer, new row]}) goes into every request as 'hsplit' (otr_m HSPLIT: banded weights);
# the final comparison is against jmD's runs on the 50 (a_0mDJ) as well.
# jmix12.sh (10-10 09:40Z, jmF): + env JM_REQ (JSON) merged into every request (e.g. {"hbands": 1}: otr_m HBANDS).
# jmix13.sh (10-10 10:30Z, jmG): + env JM_DROP = head rows (comma list) zeroed in a second final variant judged beside the full point (fold costs
# +1.34% of bill natively); env JM_PREV = the label of the start point's judging runs on the 50 (the increment is reported against it).
# jmix14.sh (10-10 10:35Z): ACCEPTANCE ON AN INDEPENDENT SET. jmF's step 1 read -0.94% on its sizing 16 and +0.20% against jmD on the 48 clean: a check on
# the set the sizes were fitted on is biased. Env JM_CHECK = a second fresh ranks file: the current point and each new point are run on it (forward only)
# and a step is accepted when the new point is better there by at least ONE standard error (independent set, so no 2-se allowance for the fit).
# jmix15.sh (10-10 13:45Z, jmH): + env JM_G1 = an existing gradient label for the first step (36 runs at the start point); env JM_DIALS (read by
# cert/refit/jmix.py: dial arms); batches past the 12 are reused from index 0 (modulo 12).
# usage (from O/trainer): JM_REQ='{...}' JM_PREV=<label> JM_CHECK=<ranks file> JM_G1=<label> JM_DIALS=X1,bscale,c0,k31 nohup bash cert/refit/jmix15.sh <tag> <params.pt> <hu.json> <first step> <last step> <first batch> <extra ncar JSON> <sizing ranks file> >> cert/refit/jmix_<tag>.log 2>&1 &
PY=/Users/matt/anaconda3/bin/python
TAG=$1; P=$2; H=$3; K1=$4; K2=$5; B0=$6; export JM_EXTRA=${7:-[]}; export JM_SIZING=$8; export JM_KC_SPLIT=1
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
if json.loads(os.environ.get('JM_HSPLIT') or '{}'):
    req['hsplit'] = json.loads(os.environ['JM_HSPLIT'])
req.update(json.loads(os.environ.get('JM_REQ') or '{}'))
extra = dict(grad=dict(mode='grad', scale=1e8), eval=dict(mode='eval'), evalkeep=dict(mode='eval', keep_final=1), evalall=dict(mode='eval', keep_all=1))[mode]
print(prefix, sum(enqueue(Path('run_M').resolve(), dict(id='a_%s_r%04d' % (prefix, r), rank=r, params=params, **extra, **req)) for r in ranks), 'queued')
PYEOF
}
SZ=$(cat $JM_SIZING); NS=$(echo $SZ | tr ',' '\n' | wc -l | tr -d ' ')
BATCHES=$($PY -c "
import sys, random; sys.path.insert(0, '.')
from otr_common import splits
used = {597, 165, 667, 25, 309, 65, 642, 125, 164, 288, 297, 330, 124, 662, 603, 50, 182, 323, 381, 90, 201, 208, 17, 563, 421, 154, 85, 594, 411, 115, 605, 678, 384, 647, 91, 60}
r = [x for x in splits()['train'] if x not in used]; random.Random(20261010).shuffle(r)
held = {int(x) for f in ('cert/ls16_ranks.txt', 'cert/runM_val50.txt', '$JM_SIZING') for x in open(f).read().split(',')}
print(' '.join(','.join(str(x) for x in r[i:i + 36] if x not in held) for i in range(0, 36 * 12, 36)))")
read -a BATCH <<< "$BATCHES"
while window_open; do sleep 20; done
queue 0m${TAG}${K1}_base $P $H evalkeep $SZ
CK=$(cat $JM_CHECK); NC=$(echo $CK | tr ',' '\n' | wc -l | tr -d ' ')
queue 0m${TAG}${K1}_chk $P $H eval $CK
ACCEPTED=$((K1 - 1))
for k in $(seq $K1 $K2); do
  D=cert/refit/jm${TAG}_d$k; mkdir -p $D
  if [ "$k" = "$K1" ] && [ -n "$JM_G1" ]; then GL=$JM_G1; NB=36
  else BI=$(((B0 + k - K1) % 12)); GL=0m${TAG}${k}g; queue $GL $P $H grad ${BATCH[$BI]}; NB=$(echo ${BATCH[$BI]} | tr ',' '\n' | wc -l | tr -d ' '); fi
  until [ "$(n "^a_${GL}_r.*\.pt$")" -ge $NB ] && [ "$(n "^a_0m${TAG}${k}_base_r.*\.json$")" -ge $NS ]; do
    ls run_M/results | grep -E -q "^a_(${GL}|0m${TAG}${k}_)[^ ]*\.error$" && { log "JOINT REFIT $TAG STEP $k: A RUN FAILED: $(ls run_M/results | grep -E "^a_(${GL}|0m${TAG}${k}_).*\.error$" | head -1)"; exit 1; }
    sleep 20
  done
  [ -s $D/arms.json ] || $PY -B cert/refit/jmix.py arms $GL $P $H $D > $D/arms.out 2>&1 || { log "JOINT REFIT $TAG STEP $k: ARMS FAILED: $(tail -1 $D/arms.out | cut -c1-300)"; exit 1; }
  while window_open; do sleep 20; done
  ARMS=$($PY -c "import json; print(' '.join(json.load(open('$D/arms.json'))['live']))"); NA=$(echo $ARMS | wc -w); ARX=$(echo $ARMS | tr ' ' '|')
  for a in $ARMS; do queue 0m${TAG}${k}_$a $D/arm_$a.pt $D/arm_${a}_hu.json evalkeep $SZ; done
  until [ "$(n "^a_0m${TAG}${k}_(${ARX})_r.*\.json$")" -ge $((NS * NA)) ]; do sleep 20; done
  $PY -B cert/refit/jmix.py solve $D 0m${TAG}${k} > $D/solve.out 2>&1
  NP=cert/refit/jm${TAG}_p$k.pt; NH=cert/refit/jm${TAG}_p${k}_hu.json
  $PY -B cert/refit/jmix.py point $D $NP $NH > $D/point.out 2>&1 || { log "JOINT REFIT $TAG STEP $k: POINT FAILED: $(tail -1 $D/point.out | cut -c1-300)"; exit 1; }
  queue 0m${TAG}$((k + 1))_chk $NP $NH eval $CK
  queue 0m${TAG}$((k + 1))_base $NP $NH evalkeep $SZ
  until [ "$(n "^a_0m${TAG}$((k + 1))_base_r.*\.json$")" -ge $NS ] && [ "$(n "^a_0m${TAG}$((k + 1))_chk_r.*\.json$")" -ge $NC ] && [ "$(n "^a_0m${TAG}${k}_chk_r.*\.json$")" -ge $NC ]; do sleep 20; done
  FIT=$($PY -B cert/fold/pairs.py $JM_SIZING 0m${TAG}$((k + 1))_base:0m${TAG}${k}_base | sed 's/^[^ ]* *//')
  REAL=$($PY -B cert/fold/pairs.py $JM_CHECK 0m${TAG}$((k + 1))_chk:0m${TAG}${k}_chk | sed 's/^[^ ]* *//')
  GO=$($PY -c "
import re; m = re.search(r': ([+-][0-9.]+)% \(se ([0-9.]+)', '''$REAL'''); print('yes' if m and float(m.group(1)) < -1 * float(m.group(2)) else 'no')")
  SOLVE=$(grep -v Warning $D/solve.out | tr '\n' ';' | tr -s ' ')
  if [ "$GO" != "yes" ]; then
    log "JOINT REFIT $TAG STEP $k: NOT ACCEPTED (early stop): the solved point on the INDEPENDENT check set ($JM_CHECK) against the current one: $REAL (on the sizing set, biased: $FIT). Solve: $SOLVE The point of step $((k - 1)) stands: $P."
    break
  fi
  log "JOINT REFIT $TAG STEP $k ACCEPTED (torch, from S44's point; gradients $GL ($NB networks); live arms of: learned arrays / column 6 / m2 eold qb / column 20 / column 17 / kappa_C at layers 1-9 and 10-14; sizes by least squares on the sizing set $JM_SIZING): $SOLVE the new point on the INDEPENDENT check set ($JM_CHECK) against the current one: $REAL (on the sizing set, biased: $FIT). Point: O/trainer/$NP + $NH."
  P=$NP; H=$NH; ACCEPTED=$k
done
[ "$ACCEPTED" = "$((K1 - 1))" ] && { log "JOINT REFIT $TAG: NO STEP ACCEPTED; the start point stands ($P), nothing to judge."; exit 0; }
while window_open; do sleep 20; done
HD=cert/refit/jm${TAG}_final_drop_hu.json
$PY -c "import json; U = json.load(open('$H')); [U.__setitem__(int(r), [0.0] * len(U[int(r)])) for r in '${JM_DROP}'.split(',') if r]; json.dump(U, open('$HD', 'w'))"
queue 0m${TAG}J $P $H eval $(cat cert/runM_val50.txt)
queue 0m${TAG}par $P $H evalall 651,378
[ -n "$JM_DROP" ] && { queue 0m${TAG}Jd $P $HD eval $(cat cert/runM_val50.txt); queue 0m${TAG}pard $P $HD evalall 651,378; }
NJ=50; NP=2; [ -n "$JM_DROP" ] && { NJ=100; NP=4; }
until [ "$(n "^a_0m${TAG}Jd?_r.*\.json$")" -ge $NJ ] && [ "$(n "^a_0m${TAG}pard?_r.*\.json$")" -ge $NP ]; do sleep 20; done
P48=$($PY -B cert/fold/pairs.py cert/runM_val48_clean.txt 0m${TAG}J:$JM_PREV | sed 's/^[^ ]* *//')
J48=$($PY -B cert/fold/pairs.py cert/runM_val48_clean.txt 0m${TAG}J:0hS41_u2 | sed 's/^[^ ]* *//')
J44=$($PY -B cert/fold/pairs.py cert/runM_val48_clean.txt 0m${TAG}J:0mCJq | sed 's/^[^ ]* *//')
J50=$($PY -B cert/fold/pairs.py cert/runM_val50.txt 0m${TAG}J:0hS41_u2 | sed 's/^[^ ]* *//')
DROP="no variant"
if [ -n "$JM_DROP" ]; then
  D48=$($PY -B cert/fold/pairs.py cert/runM_val48_clean.txt 0m${TAG}Jd:$JM_PREV | sed 's/^[^ ]* *//')
  DV=$($PY -B cert/fold/pairs.py cert/runM_val48_clean.txt 0m${TAG}J:0m${TAG}Jd | sed 's/^[^ ]* *//')
  DROP="WITHOUT rows $JM_DROP against the start point $D48; full against without: $DV"
fi
$PY -B export_dials.py $P candidates/jm${TAG} "JOINT REFIT $TAG (cert/refit/jmix13.sh): the learned arrays after $ACCEPTED accepted steps; head-input weights in $H. Judged once on the 48 clean judging ranks against the start point ($JM_PREV): $P48" > cert/refit/jm${TAG}_export.out 2>&1
cp $H candidates/jm${TAG}/head_weights_42x128.json 2>/dev/null
[ -n "$JM_DROP" ] && cp $HD candidates/jm${TAG}/head_weights_42x128_without_rows_${JM_DROP}.json 2>/dev/null
log "JOINT REFIT $TAG JUDGED ONCE (torch, forward only, final-layer MSE, 48 judging ranks never in a gradient batch): against the start point ($JM_PREV) $P48; against S44's torch point $J44; against S42's arithmetic $J48 (all 50: $J50). $DROP. Hand-over: O/trainer/candidates/jm${TAG}/ (candidate.py, physics_dials.json with kappa_C per layer, head_weights_42x128.json in the row order of O/trainer/otr_m.py HEAD_COLUMNS; request extras: $JM_REQ); parity predictions of all 16 layers on 651 / 378: run_M/results/a_0m${TAG}par_r0651.json, _r0378.json (without rows $JM_DROP: a_0m${TAG}pard_*)."
