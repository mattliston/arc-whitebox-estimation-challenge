#!/bin/bash
# TAG opus_engineer a62 chain_x2a_v1 (file written 2026-10-12T17:18Z; lead 16:4xZ item (3) 'x2a1 L7 if time allows'; radlab t -3.27): certify the
# native route x2a1 BY LAYER (make_x2a_v1.py: F4c exact-term strengths by birth layer; torch's x2a<age> at layer l = the cohort born at
# l - age) against the trainer's torch arm a_RS50_px2a1_7_7 (trainer/cert/RAD/PARITY/s50pt_v1/arm_px2a1_7_7.pt: x2a1 +0.05 at L7; read by
# x2a_from_pt_v1.py it gives exactly the native arm's values {6: ((0.0, 0.509128468809), (0.0, 0.258157425747))}) vs a_RS51_base on
# 7100456 / 457 (trainer/run_RAD_S51_gen/results). Runs AFTER chain_routes_v2.sh exits (lead's order: S51's port, then bscale / X1, then
# k22 / x2a1), at most 2 native processes of this lane at any time (a launch waits for a free slot), gate.sh, guard_footprint_v2.
#   no-ops on the S50 files: s49/x2az0s50 (empty table) BIT FOR BIT (prediction and bill) vs fb277a8a on 7100521 (cd478a30fbbe56d9);
#     s49/x2af0s50 (all 14 births listed at the S50 strengths: the by-birth path, v2) and s49/x2af0tw (same on the twin f04fa779, v3)
#     prediction bit for bit (7100521 / 7100136; bill reported);
#   arm s49/x2a7s50 on 7100456 / 457 vs the native base s50pbs_a39; parity_verdict_v3.py (corr >= 0.70, size 1/3 .. 3, floor 2e-6).
# One TEAM_LOG line 'ROUTE PARITY x2a1_7: PASSED (...)' or '...: FAILED (...)' (the trainer's format); not comparable: no line.
set -u
P2=/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16
O=$P2/opus_reexam_20261001_v1; A=$O/engineer/a62_modules; LOG=$O/TEAM_LOG.md; PY=/Users/matt/arc-white-box-estimation/.venv/bin/python
R=$O/trainer/run_RAD_S51_gen/results; NATIVE="^$P2/astra_phase2_reexamination_v1/openblas_runtime_env_v1/bin/python -B .*/engineer/"
say() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }
line() { echo "[engineer] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*" >> $LOG; say "POSTED: $*"; }
run() {
  [ -e $A/gate_rdz_$2.json ] && return 0
  until [ $(ps -axo command= | grep -c -E "$NATIVE") -lt 2 ]; do sleep 60; done
  bash $O/gate.sh 7 engineer || return 3
  cd $P2 && /Users/matt/arc-white-box-estimation/.venv/bin/python -B $O/release/guard_footprint_v2.py \
    --record $A/gate_rdz_$2_guard.json --seconds 5400 --rss-backstop 7500000000 \
    $P2/astra_phase2_reexamination_v1/openblas_runtime_env_v1/bin/python -B $A/gate_rdz_v1.py $A/s49/$1/candidate.py none none $3 $A/gate_rdz_$2.json \
    > $A/gate_rdz_$2.log 2>&1
  say "$2 exit=$?"
}
same() { # <run tag> <reference run tag>
  $PY -c "
import json
a = json.load(open('$A/gate_rdz_$2.json'))['rows'][0]; b = json.load(open('$A/gate_rdz_$1.json'))['rows'][0]
print(('BIT FOR BIT' if a['flops'] == b['flops'] else 'PREDICTION BIT FOR BIT, bill %+d flops' % (b['flops'] - a['flops'])) if a['pred_sha'] == b['pred_sha'] else 'DIFFERENT (%s vs %s)' % (b['pred_sha'], a['pred_sha']))" 2>&1 | tail -1; }
say "armed; waiting for chain_routes_v2.sh to exit"
while pgrep -f 'chain_routes_v2[.]sh' > /dev/null; do sleep 120; done
say "chain_routes_v2 done; runs start"
( run x2az0s50 x2az0_521 7100521; run x2af0s50 x2af0_521 7100521 ) &
sleep 30
( run x2af0tw x2af0tw_136 7100136; run x2a7s50 x2a7_a39 7100456,7100457 ) &
wait
for spec in "x2az0s50 x2az0_521 7100521" "x2af0s50 x2af0_521 7100521" "x2af0tw x2af0tw_136 7100136" "x2a7s50 x2a7_a39 7100456,7100457"; do
  set -- $spec
  [ -e $A/gate_rdz_$2.json ] && continue
  say "$2 has no result: one retry in 5 min"; sleep 300; mv $A/gate_rdz_$2.log $A/gate_rdz_$2.log.try1 2>/dev/null; run $1 $2 $3
  [ -e $A/gate_rdz_$2.json ] || { line "x2a1 L7 route: native run $2 failed twice (a62_modules/gate_rdz_$2.log), no parity verdict"; exit 1; }
done
n0=$(same x2az0_521 s50pbs_521); nf=$(same x2af0_521 s50pbs_521); nt=$(same x2af0tw_136 s50tw_x)
say "no-ops: empty $n0; forced v2 $nf; forced v3 twin $nt"
out=$(PREFIX=a_RS50 BASEPREFIX=a_RS51 $PY -B $A/parity_verdict_v3.py $R px2a1_7_7 $A/gate_rdz_s50pbs_a39.json $A/gate_rdz_x2a7_a39.json 7100456 7100457 2>&1)
echo "$out" > $A/parity_route_x2a1_7.out
v=$(echo "$out" | grep '^VERDICT' | sed 's/^VERDICT [^:]*: //')
nums=$(echo "$out" | grep -v '^VERDICT' | sed -e 's/ of the prediction (floor 2e-06)//' -e "s/^px2a1_7_7 //" | tr '\n' ';' | cut -c1-420)
what="make_x2a_v1.py (F4c exact-term strengths by birth layer), x2a1 +0.05 at L7 = the cohort born at 6, age 1"
[ -n "$v" ] || { say "parity_verdict_v3 gave no verdict (a62_modules/parity_route_x2a1_7.out), no line"; exit 1; }
case "$v|$n0|$nf|$nt" in
  "PASS|BIT FOR BIT|"*"BIT FOR BIT"*"|"*"BIT FOR BIT"*) line "ROUTE PARITY x2a1_7: PASSED ($what, on the S50 safe fb277a8a vs the trainer's a_RS50_px2a1_7_7 / a_RS51_base, rule of 07:3xZ; no-ops: empty table $n0 on 7100521, all 14 births listed $nf on 7100521, twin (F4c v3) all births listed $nt on 7100136; $nums)";;
  "NOT COMPARABLE|"*) say "x2a1_7: not comparable on both networks, no line ($nums)";;
  *) line "ROUTE PARITY x2a1_7: FAILED ($what; verdict $v; no-ops: empty $n0, all births $nf, twin $nt; $nums)";;
esac
say "done"
