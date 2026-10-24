#!/bin/bash
# TAG opus_engineer2 a62 chain_x2a_v3 (file written 2026-10-12T22:36:43Z; lead 22:1xZ: x2a1_7 by option (b), the trainer's LARGER arm) = chain_x2a_v2.sh with:
#   - the torch arm a_RS50_px2a1p1_7_7 (PARITY/s50pt_v2/arm_px2a1p1_7_7.pt: x2a1 +1.0 at L7; the +0.05 arm moved the final prediction only 2.3e-7
#     of its size, under the 2e-6 floor) in trainer/run_trainer_parity_v2/results, base a_RS51_base in run_RAD_S51_gen/results
#     (parity_verdict_v4.py, BASEDIR); native arm s49/x2a7p1s50 (make_x2a_v1.py on fb277a8a with fragment_x2av_a7p1s50.py = x2a_from_pt_v1.py of
#     that .pt: {6: ((0.0, 1.45912846881), (0.0, 0.258157425747))}; the converter checked on the +0.05 arm = x2a7s50's values);
#   - the empty-table no-op judged on the PREDICTION (bill reported): the bill moves with the box load through the setup prebind's 3 s wall
#     guard (chain_routes_v4, 22:1xZ); a different prediction still fails;
#   - it waits for chain_routes_v5.sh to exit.
# v2 header: (file written 2026-10-12T21:55:41Z; lead 21:4xZ: route certifications serial, ONE native process, BLAS threads 1)
# = chain_x2a_v1.sh (never ran: held at 17:48Z) with:
#   - it waits for chain_routes_v3.sh (not v2) to exit;
#   - every native run SERIAL; each launch waits for slot() (fewer than 2 native processes of the engineer lane, chain_rad_port_v9's check and
#     regex) and gate.sh; guard_footprint_v2 at 7200 s (v1: 5400 s); one retry after 5 min under tag <tag>r; a run counts only when its JSON
#     exists;
#   - a no-op that is DIFFERENT fails the route even when the arm is not comparable; a run that fails twice posts one [engineer2] note
#     without the parity prefix (v1 posted it as [engineer]);
#   - OMP / OPENBLAS / MKL / VECLIB / NUMEXPR threads = 1 for every python of the chain; clean stop: touch a62_modules/STOP_X2A.
# Verdict line as v1, as [engineer] (read anchored by the trainer's routes_update_v3.py and the port's family check pt_families_v8.py).
# v1 header: (file written 2026-10-12T17:18Z; lead 16:4xZ item (3) 'x2a1 L7 if time allows'; radlab t -3.27): certify the
# native route x2a1 BY LAYER (make_x2a_v1.py: F4c exact-term strengths by birth layer; torch's x2a<age> at layer l = the cohort born at
# l - age) against the trainer's torch arm a_RS50_px2a1_7_7 (trainer/cert/RAD/PARITY/s50pt_v1/arm_px2a1_7_7.pt: x2a1 +0.05 at L7; read by
# x2a_from_pt_v1.py it gives exactly the native arm's values {6: ((0.0, 0.509128468809), (0.0, 0.258157425747))}) vs a_RS51_base on
# 7100456 / 457 (trainer/run_RAD_S51_gen/results).
#   no-ops on the S50 files: s49/x2az0s50 (empty table) BIT FOR BIT (prediction and bill) vs fb277a8a on 7100521 (cd478a30fbbe56d9);
#     s49/x2af0s50 (all 14 births listed at the S50 strengths: the by-birth path, v2) and s49/x2af0tw (same on the twin f04fa779, v3)
#     prediction bit for bit (7100521 / 7100136; bill reported);
#   arm s49/x2a7s50 on 7100456 / 457 vs the native base s50pbs_a39; parity_verdict_v3.py (corr >= 0.70, size 1/3 .. 3, floor 2e-6).
# One TEAM_LOG line 'ROUTE PARITY x2a1_7: PASSED (...)' or '...: FAILED (...)' (the trainer's format); not comparable: no line.
set -u
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
P2=/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16
O=$P2/opus_reexam_20261001_v1; A=$O/engineer/a62_modules; LOG=$O/TEAM_LOG.md; PY=/Users/matt/arc-white-box-estimation/.venv/bin/python
R=$O/trainer/run_trainer_parity_v2/results; RB=$O/trainer/run_RAD_S51_gen/results; STOP=$A/STOP_X2A
NATIVE="^($P2/astra_phase2_reexamination_v1/openblas_runtime_env_v1/bin/python -B .*/engineer/|$P2/data_gate/.client-venv/bin/python -B -u .*boundary_client[.]py .*/engineer/)"
say() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }
verdict_line() { echo "[engineer] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*" >> $LOG; say "POSTED: $*"; }
note() { case "$*" in *PARITY*|*"FILES READY"*) say "REFUSED a note with a trigger string: $*";; *) echo "[engineer2] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*" >> $LOG; say "NOTE: $*";; esac; }
stop() { [ -e $STOP ] && { say "STOP file: exit"; exit 0; }; return 0; }
slot() { until [ $(ps -axo command= | grep -c -E "$NATIVE") -lt 2 ]; do stop; sleep 60; done; }
run1() {   # <dir under s49> <out tag> <seeds>
  stop; slot; bash $O/gate.sh 7 engineer2
  cd $P2 && $PY -B $O/release/guard_footprint_v2.py \
    --record $A/gate_rdz_$2_guard.json --seconds 7200 --rss-backstop 7500000000 \
    $P2/astra_phase2_reexamination_v1/openblas_runtime_env_v1/bin/python -B $A/gate_rdz_v1.py $A/s49/$1/candidate.py none none $3 $A/gate_rdz_$2.json \
    > $A/gate_rdz_$2.log 2>&1
  [ -e $A/gate_rdz_$2.json ]
}
run() {    # one retry under a new tag; the retry's results stand in for the first tag's
  [ -e $A/gate_rdz_$2.json ] && { say "$2 already done"; return 0; }
  say "$2 start ($1 on $3)"
  run1 "$1" "$2" "$3" && { say "$2 ok"; return 0; }
  say "$2 FAILED, retry in 5 min as $2r"; sleep 300
  run1 "$1" "$2r" "$3" && { say "$2r ok"; for x in .json _pred.npz; do [ -e $A/gate_rdz_$2$x ] || cp $A/gate_rdz_$2r$x $A/gate_rdz_$2$x; done; return 0; }
  say "$2r FAILED"; return 1
}
same() { # <run tag> <reference run tag>
  $PY -B -c "
import json
a = json.load(open('$A/gate_rdz_$2.json'))['rows'][0]; b = json.load(open('$A/gate_rdz_$1.json'))['rows'][0]
print(('BIT FOR BIT' if a['flops'] == b['flops'] else 'PREDICTION BIT FOR BIT, bill %+d flops' % (b['flops'] - a['flops'])) if a['pred_sha'] == b['pred_sha'] else 'DIFFERENT (%s vs %s)' % (b['pred_sha'], a['pred_sha']))" 2>&1 | tail -1; }
say "armed (serial, one native process); waiting for chain_routes_v5.sh to exit"
while pgrep -f 'chain_routes_v5[.]sh' > /dev/null; do stop; sleep 120; done
say "chain_routes_v5 done; runs start"
for spec in "x2az0s50 x2az0_521 7100521" "x2af0s50 x2af0_521 7100521" "x2af0tw x2af0tw_136 7100136" "x2a7p1s50 x2a7p1_a39 7100456,7100457"; do
  set -- $spec
  run $1 $2 $3 || { note "x2a1 L7 route check: native run $2 failed twice (a62_modules/gate_rdz_$2*.log), no verdict"; exit 1; }
done
n0=$(same x2az0_521 s50pbs_521); nf=$(same x2af0_521 s50pbs_521); nt=$(same x2af0tw_136 s50tw_x)
say "no-ops: empty $n0; forced v2 $nf; forced v3 twin $nt"
until [ -f $R/a_RS50_px2a1p1_7_7_r7100456.json ] && [ -f $R/a_RS50_px2a1p1_7_7_r7100457.json ]; do stop; sleep 60; done
out=$(PREFIX=a_RS50 BASEPREFIX=a_RS51 BASEDIR=$RB $PY -B $A/parity_verdict_v4.py $R px2a1p1_7_7 $A/gate_rdz_s50pbs_a39.json $A/gate_rdz_x2a7p1_a39.json 7100456 7100457 2>&1)
echo "$out" > $A/parity_route_x2a1_7.out
v=$(echo "$out" | grep '^VERDICT' | sed 's/^VERDICT [^:]*: //')
nums=$(echo "$out" | grep -v '^VERDICT' | sed -e 's/ of the prediction (floor 2e-06)//' -e "s/^px2a1p1_7_7 //" | tr '\n' ';' | cut -c1-420)
what="make_x2a_v1.py (F4c exact-term strengths by birth layer), x2a1 +1.0 at L7 = the cohort born at 6, age 1 (PARITY/s50pt_v2)"
[ -n "$v" ] || { note "x2a1 L7 route check: parity_verdict_v4 gave no verdict (a62_modules/parity_route_x2a1_7.out), no verdict"; exit 1; }
for n in "$n0" "$nf" "$nt"; do
  case "$n" in *"BIT FOR BIT"*|DIFFERENT*) ;; *) note "x2a1 L7 route check: a no-op comparison did not run ($n), no verdict"; exit 1;; esac
done
case "$v|$n0|$nf|$nt" in
  "PASS|"*"BIT FOR BIT"*"|"*"BIT FOR BIT"*"|"*"BIT FOR BIT"*) verdict_line "ROUTE PARITY x2a1_7: PASSED ($what, on the S50 safe fb277a8a vs the trainer's a_RS50_px2a1p1_7_7 / a_RS51_base, rule of 07:3xZ; no-ops: empty table $n0 on 7100521, all 14 births listed $nf on 7100521, twin (F4c v3) all births listed $nt on 7100136; chain_x2a_v3; $nums)";;
  *DIFFERENT*) verdict_line "ROUTE PARITY x2a1_7: FAILED ($what; verdict $v; no-ops: empty $n0, all births $nf, twin $nt; chain_x2a_v3; $nums)";;
  "NOT COMPARABLE|"*) say "x2a1_7: not comparable on both networks, no line ($nums)";;
  *) verdict_line "ROUTE PARITY x2a1_7: FAILED ($what; verdict $v; no-ops: empty $n0, all births $nf, twin $nt; chain_x2a_v3; $nums)";;
esac
say "done"
