#!/bin/bash
# TAG opus_engineer a62 (file written 2026-10-12T08:24Z; lead 08:0xZ: route parity for each round's NEW families before its freeze; trainer 07:4xZ: S52's
# fk4la2_1_7 / fk4la2_8_15 need K4L age 2 by layer, fragment_k4la2_v1.py, no-ops bit for bit 07:39Z): UNATTENDED parity of those two arms.
#   1. waits for cert/RAD/S52/config.json -> the start round M (its params), then for the port loop's own files of M (s49/jmS<M>_v1_pbs, the safe,
#      K4L v6; s49/jmS<M>_v1tw_x, the twin, K4L v7) AND its READY / STOPPED / SKIPPED line for S<M> (so the two never overlap)
#   2. builds four arm files: the start point's K4L age-2 strength by layer (1 + its pl['k4la2'][l], l = 2..15) + 0.05 at 1-7 or at 8-15 (the
#      trainer's FIXED arms k4la2:1:7:0.05 / k4la2:8:15:0.05), on the safe and on the twin
#   3. waits for the trainer's torch records a_RS52_base / a_RS52_fk4la2_1_7 / a_RS52_fk4la2_8_15 on 7100712 and 7100713 (sizing A43's first two)
#   4. native runs, two processes (safe slot, twin slot), gate.sh first, guard_footprint_v2
#   5. parity_verdict_v2.py (rule + floor of 07:3xZ) per arm and diet; an arm PASSES only if both diets pass; posts the trainer's exact line
#      'S52 PORT PARITY <arm>: PASSED' or '...: FAILED' (NOT COMPARABLE: no line, the arm is kept). Clean stop: touch a62_modules/STOP_S52_PARITY.
set -u
P2=/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16
O=$P2/opus_reexam_20261001_v1; E=$O/engineer; A=$E/a62_modules; PY=/Users/matt/arc-white-box-estimation/.venv/bin/python
C=$O/trainer/cert/RAD/S52; R=$O/trainer/run_RAD_S52_gen/results; LOG=$O/TEAM_LOG.md; STOP=$A/STOP_S52_PARITY; S1=7100712; S2=7100713
say() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }
verdict_line() { echo "[engineer] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*" >> $LOG; say "POSTED: $*"; }
run1() {
  bash $O/gate.sh 7 engineer || return 3
  cd $P2 && OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
    /Users/matt/arc-white-box-estimation/.venv/bin/python -B $O/release/guard_footprint_v2.py \
    --record $A/gate_rdz_$2_guard.json --seconds 3600 --rss-backstop 7500000000 \
    $P2/astra_phase2_reexamination_v1/openblas_runtime_env_v1/bin/python -B $A/gate_rdz_v1.py $1 none none $3 $A/gate_rdz_$2.json \
    > $A/gate_rdz_$2.log 2>&1
}
run() {
  [ -e $A/gate_rdz_$2.json ] && return 0
  run1 "$1" "$2" "$3" && return 0
  sleep 300; run1 "$1" "$2r" "$3" && { for x in .json _pred.npz; do [ -e $A/gate_rdz_$2$x ] || cp $A/gate_rdz_$2r$x $A/gate_rdz_$2$x; done; return 0; }
  return 1
}
waitfor() { while ! eval "$1"; do [ -e $STOP ] && { say "STOP"; exit 0; }; sleep 120; done; }
say "armed"
waitfor '[ -f $C/config.json ]'
M=$($PY -B -c "import json, re; print(re.search(r'/cert/RAD/(S\d+)/', json.load(open('$C/config.json'))['params']).group(1))")
PT=$($PY -B -c "import json; print(json.load(open('$C/config.json'))['params'])")
say "S52 starts from $M ($PT)"
SB=$A/s49/jm${M}_v1_pbs/candidate.py; TB=$A/s49/jm${M}_v1tw_x/candidate.py
waitfor '[ -f $SB ] && [ -f $TB ] && grep -q -E "^\[engineer\] .* ${M} (FILES READY FOR RELEASE|PORT STOPPED|PORT SKIPPED)" $LOG || [ "$M" = S50 ]'
say "bases: safe $SB, twin $TB"
cd $A
for band in 1_7 8_15; do
  lo=${band%_*}; hi=${band#*_}
  f=$A/fragment_k4la2v_s52arm_${band}.py
  [ -e $f ] || $PY -B -c "
import sys
from pathlib import Path
import numpy as np
src = (Path('$E') / 'a54_jmiv/read_pt.py').read_text().split('\na = load(sys.argv[1])')[0]
ns = {}; exec(compile(src, 'read_pt.py', 'exec'), ns)
eps = np.asarray(ns['load']('$PT').get('pl', {}).get('k4la2', np.zeros(16)), dtype=float).ravel()
t = {l: 1.0 + float(eps[l]) + (0.05 if $lo <= l <= $hi else 0.0) for l in range(2, 16)}
Path('$f').write_text('# opus-engineer a62 chain_s52parity_v1: S52 arm fk4la2_${band} on the start point $M (1 + its k4la2 shift by layer, + 0.05 at $lo-$hi)\n_K4LA2_BY_LAYER = %r\n' % t)
print('$band', t)"
  for d in safe twin; do
    base=$SB; [ $d = twin ] && base=$TB
    route=$A/fragment_k4la2_v1.py; grep -q 'BEGIN OPUS-ENGINEER a62 K4L AGE 2 BY LAYER' $base && route=''   # a base that carries the route gets the values only (no second wrapper)
    [ -e $A/s49/s52k4a_${d}_${band} ] || $PY -B $A/make_s49_v1.py $base s52k4a_${d}_${band} $route $f > /dev/null
  done
done
recs_ok() { for s in $S1 $S2; do for a in base fk4la2_1_7 fk4la2_8_15; do [ -f $R/a_RS52_${a}_r$s.json ] || return 1; done; done; return 0; }
waitfor 'recs_ok'
say "torch records present"
( run $SB s52pb_safe_base $S1,$S2; run $A/s49/s52k4a_safe_1_7/candidate.py s52pb_safe_1_7 $S1,$S2; run $A/s49/s52k4a_safe_8_15/candidate.py s52pb_safe_8_15 $S1,$S2 ) &
sleep 5
( run $TB s52pb_twin_base $S1,$S2; run $A/s49/s52k4a_twin_1_7/candidate.py s52pb_twin_1_7 $S1,$S2; run $A/s49/s52k4a_twin_8_15/candidate.py s52pb_twin_8_15 $S1,$S2 ) &
wait
for band in 1_7 8_15; do
  arm=fk4la2_$band; vs=""
  for d in safe twin; do
    out=$(PREFIX=a_RS52 $PY -B $A/parity_verdict_v2.py $R $arm $A/gate_rdz_s52pb_${d}_base.json $A/gate_rdz_s52pb_${d}_${band}.json $S1 $S2 2>&1)
    echo "$out" > $A/parity_s52_${arm}_${d}.out
    v=$(echo "$out" | grep '^VERDICT' | awk '{print $NF}'); vs="$vs $d:$v"
    say "$arm $d: $(echo "$out" | tr '\n' ' ' | cut -c1-400)"
  done
  case "$vs" in
    *FAIL*) verdict_line "S52 PORT PARITY $arm: FAILED (chain_s52parity_v1, rule of 07:3xZ; K4L age 2 by layer, fragment_k4la2_v1; diets$vs; numbers in a62_modules/parity_s52_${arm}_safe.out / _twin.out)";;
    *"safe:PASS"*"twin:PASS"*) verdict_line "S52 PORT PARITY $arm: PASSED (chain_s52parity_v1, rule of 07:3xZ; K4L age 2 by layer, fragment_k4la2_v1, both diets; numbers in a62_modules/parity_s52_${arm}_safe.out / _twin.out)";;
    *) say "$arm: no verdict line ($vs: not comparable on a diet), the arm is kept";;
  esac
done
say "done"
