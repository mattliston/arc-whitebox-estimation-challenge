#!/bin/bash
# TAG opus_engineer a62 chain_routes_v2 (file written 2026-10-12T16:51Z; = chain_routes_v1.sh, never run, with the trainer's 16:5xZ names and verdict format):
# certify per-layer native routes against the trainer's single-layer torch arms at the S50 frozen point (trainer/run_RAD_S51_gen/results:
# arms a_RS50_pbscale_0_0 / a_RS50_pX1_14_14 / a_RS50_pk22_14_14, base a_RS51_base, on 7100456 / 457). Native files on the S50 safe fb277a8a:
# no-ops s49/pldbx0s50 (the 10-10 DLY edits for bscale / X1) and s49/k22z0s50 (make_k22_v1) on 7100521 (bit for bit vs fb277a8a's run there,
# pred cd478a30fbbe56d9); arms s49/pldb0s50 (bscale +0.05 at L0), s49/pldx14s50 (X1 +0.05 at L14), s49/k22a14s50 (k22 x1.05 at L14) on
# 7100456 / 457 against the native base s50pbs_a39. Starts after the port loop's S51 line (lead: S51's port first); two processes, gate.sh,
# guard_footprint_v2. Verdicts: parity_verdict_v3.py (the rule of 07:3xZ: corr >= 0.70, size 1/3 .. 3, floor 2e-6) AND the no-op bit for bit;
# one TEAM_LOG line per family: 'ROUTE PARITY <fam>_<layer>: PASSED (...)' or '...: FAILED (...)' (the trainer's format); not comparable: no line.
P2=/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16
O=$P2/opus_reexam_20261001_v1; A=$O/engineer/a62_modules; LOG=$O/TEAM_LOG.md; PY=/Users/matt/arc-white-box-estimation/.venv/bin/python
R=$O/trainer/run_RAD_S51_gen/results
say() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }
line() { echo "[engineer] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*" >> $LOG; say "POSTED: $*"; }
run() {
  [ -e $A/gate_rdz_$2.json ] && return 0
  bash $O/gate.sh 7 engineer || return 3
  cd $P2 && /Users/matt/arc-white-box-estimation/.venv/bin/python -B $O/release/guard_footprint_v2.py \
    --record $A/gate_rdz_$2_guard.json --seconds 5400 --rss-backstop 7500000000 \
    $P2/astra_phase2_reexamination_v1/openblas_runtime_env_v1/bin/python -B $A/gate_rdz_v1.py $A/s49/$1/candidate.py none none $3 $A/gate_rdz_$2.json \
    > $A/gate_rdz_$2.log 2>&1
  say "$2 exit=$?"
}
noop_of() { python3 -c "
import json
a = json.load(open('$A/gate_rdz_s50pbs_521.json'))['rows'][0]; b = json.load(open('$A/gate_rdz_$1.json'))['rows'][0]
print('BIT FOR BIT' if (a['pred_sha'] == b['pred_sha'] and a['flops'] == b['flops']) else 'DIFFERENT (%s vs %s)' % (b['pred_sha'], a['pred_sha']))"; }
say "armed; waiting for the port loop's S51 line"
until grep -q -E '^\[engineer\] [0-9T:-]+Z S51 (FILES READY FOR RELEASE|PORT STOPPED|PORT SKIPPED)' $LOG; do sleep 120; done
( run pldbx0s50 pldbx0_521 7100521; run pldx14s50 pldx14_a39 7100456,7100457; run k22z0s50 k22z0_521 7100521 ) &
sleep 5
( run pldb0s50 pldb0_a39 7100456,7100457; run k22a14s50 k22a14_a39 7100456,7100457 ) &
wait
for spec in "bscale_0:pbscale_0_0:pldb0_a39:pldbx0_521:the 10-10 DLY builder edit (make_pld_bx_v1.py), bscale +0.05 at L0" \
            "X1_14:pX1_14_14:pldx14_a39:pldbx0_521:the 10-10 DLY builder edit (make_pld_bx_v1.py), X1 +0.05 at L14" \
            "k22_14:pk22_14_14:k22a14_a39:k22z0_521:make_k22_v1.py, k22 x1.05 at L14"; do
  IFS=: read fl arm tag nooptag what <<< "$spec"
  until [ -f $R/a_RS50_${arm}_r7100456.json ] && [ -f $R/a_RS50_${arm}_r7100457.json ]; do sleep 60; done
  noop=$(noop_of $nooptag)
  out=$(PREFIX=a_RS50 BASEPREFIX=a_RS51 $PY -B $A/parity_verdict_v3.py $R $arm $A/gate_rdz_s50pbs_a39.json $A/gate_rdz_$tag.json 7100456 7100457 2>&1)
  echo "$out" > $A/parity_route_${fl}.out
  v=$(echo "$out" | grep '^VERDICT' | sed 's/^VERDICT [^:]*: //')
  nums=$(echo "$out" | grep -v '^VERDICT' | sed -e 's/ of the prediction (floor 2e-06)//' -e "s/^$arm //" | tr '\n' ';' | cut -c1-420)
  case "$v|$noop" in
    "PASS|BIT FOR BIT") line "ROUTE PARITY $fl: PASSED ($what on the S50 safe fb277a8a vs the trainer's a_RS50_$arm / a_RS51_base, rule of 07:3xZ; no-op $noop on 7100521; $nums)";;
    "NOT COMPARABLE|"*) say "$fl: not comparable on both networks, no line ($nums)";;
    *) line "ROUTE PARITY $fl: FAILED ($what; verdict $v; no-op $noop; $nums)";;
  esac
done
say "done"
