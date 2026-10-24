#!/bin/bash
# TAG opus_engineer2 a62 chain_routes_v5 (file written 2026-10-12T22:36:43Z; lead 22:1xZ: X1_14 dropped; bscale_0 by option (b), the trainer's LARGER arm)
# = chain_routes_v4.sh for bscale_0 only, against the torch arm a_RS50_pbscale15_0_0 (PARITY/s50pt_v2/arm_pbscale15_0_0.pt: bscale +1.5 at L0,
# the +0.05 arm moved the final prediction only 1.7e-7 of its size, under the 2e-6 floor) in trainer/run_trainer_parity_v2/results, base
# a_RS51_base in trainer/run_RAD_S51_gen/results (parity_verdict_v4.py, BASEDIR); native arm s49/pldb15s50 (make_pld_bx_v1.py on fb277a8a with
# bscale[0] = 2.5348966470937038 = dly_from_pt_v3.py's value of that .pt, the converter checked on the +0.05 arm: 1.0848966470937038 = pldb0s50's).
# Same no-op (pldbx0s50 on 7100521, judged on the prediction), rule, floor and line formats as v4. usage: chain_routes_v5.sh
# v4 header: (file written 2026-10-12T22:17:53Z) = chain_routes_v3.sh with the no-op judged on the PREDICTION: v3 (and v2) also
# required the no-op's bill bit for bit, which holds only on a quiet box: the setup prebind binds request keys until a 3 s wall guard, so a
# native session's bill and op count move with the box load (release 10-12: ops +2,000, bill -0.13 .. +0.6% on byte-identical files). At
# 22:14:35Z v3 posted the k22 L14 verdict as failed on exactly that (prediction bit for bit, bill +0.55% at load ~100, while the arm matched
# torch at corr +1.000 / size 1.00). Now: 'BIT FOR BIT' (prediction and bill) or 'PREDICTION BIT FOR BIT, bill <+-n> flops' both pass the
# no-op; a different prediction fails it. Runs already done are reused (their JSON exists). usage: chain_routes_v4.sh [family ...]
# v3 header: (file written 2026-10-12T21:55:41Z; lead 21:4xZ: route certifications serial, ONE native process, BLAS threads 1)
# = chain_routes_v2.sh (never ran: held at 17:48Z) with:
#   - every native run SERIAL (never two of this chain at once); each launch waits for slot() (fewer than 2 native processes of the engineer
#     lane: chain_rad_port_v9's own check and regex, so this chain and the port loop share the lane's 2 slots) and gate.sh;
#   - guard_footprint_v2 at 7200 s (v9's slow-box size; v2 had 5400 s); one retry after 5 min under tag <tag>r (v9's run()); a run counts
#     only when its JSON exists (gate_rdz_v1.py writes it at the end; exit codes are not trusted);
#   - per family: its no-op, its arm, then its verdict at once (bscale_0 first);
#   - a verdict line only when every run it needs has a result and parity_verdict_v3.py prints a verdict; a no-op that is DIFFERENT fails the
#     family even when the arm is not comparable; a run that fails twice posts one [engineer2] note without the parity prefix and the chain
#     goes on to the next family;
#   - OMP / OPENBLAS / MKL / VECLIB / NUMEXPR threads = 1 for every python of the chain; clean stop: touch a62_modules/STOP_ROUTES;
#   - usage: chain_routes_v3.sh [family ...] (default bscale_0 X1_14 k22_14). The trainer's S50 single-layer arms for bscale_0 / X1_14 move
#     the final prediction by 1.7e-7 / 5.4-6.0e-7 of its size, under the 2e-6 floor (k22_14: 5.9e-6 / 6.4e-6): those two can only come out
#     'not comparable' or as a noise-driven failure, never pass, so they wait for larger torch arms.
# Verdict lines as v2 (the trainer's routes_update_v3.py reads them anchored at the line start, as [engineer]):
#   'ROUTE PARITY <fam>_<layer>: PASSED (...)' or '...: FAILED (...)'; not comparable: no line.
# v2 header: (file written 2026-10-12T16:51Z; = chain_routes_v1.sh, never run, with the trainer's 16:5xZ names and verdict format):
# certify per-layer native routes against the trainer's single-layer torch arms at the S50 frozen point (trainer/run_RAD_S51_gen/results:
# arms a_RS50_pbscale_0_0 / a_RS50_pX1_14_14 / a_RS50_pk22_14_14, base a_RS51_base, on 7100456 / 457). Native files on the S50 safe fb277a8a:
# no-ops s49/pldbx0s50 (the 10-10 DLY edits for bscale / X1) and s49/k22z0s50 (make_k22_v1) on 7100521 (bit for bit vs fb277a8a's run there,
# pred cd478a30fbbe56d9); arms s49/pldb0s50 (bscale +0.05 at L0), s49/pldx14s50 (X1 +0.05 at L14), s49/k22a14s50 (k22 x1.05 at L14) on
# 7100456 / 457 against the native base s50pbs_a39. Verdicts: parity_verdict_v3.py (the rule of 07:3xZ: corr >= 0.70, size 1/3 .. 3,
# floor 2e-6) AND the no-op bit for bit.
set -u
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
P2=/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16
O=$P2/opus_reexam_20261001_v1; A=$O/engineer/a62_modules; LOG=$O/TEAM_LOG.md; PY=/Users/matt/arc-white-box-estimation/.venv/bin/python
R=$O/trainer/run_trainer_parity_v2/results; RB=$O/trainer/run_RAD_S51_gen/results; STOP=$A/STOP_ROUTES; FAMS=" bscale_0 "
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
noop_of() { $PY -B -c "
import json
a = json.load(open('$A/gate_rdz_s50pbs_521.json'))['rows'][0]; b = json.load(open('$A/gate_rdz_$1.json'))['rows'][0]
print(('BIT FOR BIT' if a['flops'] == b['flops'] else 'PREDICTION BIT FOR BIT, bill %+d flops (box load: the setup prebind wall guard)' % (b['flops'] - a['flops'])) if a['pred_sha'] == b['pred_sha'] else 'DIFFERENT (pred %s vs %s)' % (b['pred_sha'], a['pred_sha']))" 2>&1 | tail -1; }
say "armed for:$FAMS(serial, one native process; lane slots shared with the port loop); waiting for the port loop's S51 line"
until grep -q -E '^\[engineer\] [0-9T:-]+Z S51 (FILES READY FOR RELEASE|PORT STOPPED|PORT SKIPPED)' $LOG; do stop; sleep 120; done
for spec in "bscale_0:pbscale15_0_0:pldbx0s50:pldbx0_521:pldb15s50:pldb15_a39:the 10-10 DLY builder edit (make_pld_bx_v1.py), bscale +1.5 at L0 (PARITY/s50pt_v2)"; do
  IFS=: read fl arm noopdir nooptag armdir tag what <<< "$spec"
  case "$FAMS" in *" $fl "*) ;; *) continue;; esac
  stop
  { run $noopdir $nooptag 7100521 && run $armdir $tag 7100456,7100457; } || { note "route check $fl: a native run failed twice (a62_modules/gate_rdz_${nooptag}*.log, gate_rdz_${tag}*.log), no verdict"; continue; }
  until [ -f $R/a_RS50_${arm}_r7100456.json ] && [ -f $R/a_RS50_${arm}_r7100457.json ]; do stop; sleep 60; done
  noop=$(noop_of $nooptag)
  out=$(PREFIX=a_RS50 BASEPREFIX=a_RS51 BASEDIR=$RB $PY -B $A/parity_verdict_v4.py $R $arm $A/gate_rdz_s50pbs_a39.json $A/gate_rdz_$tag.json 7100456 7100457 2>&1)
  echo "$out" > $A/parity_route_${fl}.out
  v=$(echo "$out" | grep '^VERDICT' | sed 's/^VERDICT [^:]*: //')
  nums=$(echo "$out" | grep -v '^VERDICT' | sed -e 's/ of the prediction (floor 2e-06)//' -e "s/^$arm //" | tr '\n' ';' | cut -c1-420)
  say "$fl: verdict '$v', no-op '$noop'"
  case "$noop" in *"BIT FOR BIT"*|DIFFERENT*) ;; *) note "route check $fl: the no-op comparison did not run ($noop), no verdict"; continue;; esac
  [ -n "$v" ] || { note "route check $fl: parity_verdict_v4 gave no verdict (a62_modules/parity_route_${fl}.out), no verdict"; continue; }
  case "$v|$noop" in
    "PASS|"*"BIT FOR BIT"*) verdict_line "ROUTE PARITY $fl: PASSED ($what on the S50 safe fb277a8a vs the trainer's a_RS50_$arm / a_RS51_base, rule of 07:3xZ; no-op $noop on 7100521; chain_routes_v5; $nums)";;
    "NOT COMPARABLE|"*"BIT FOR BIT"*) say "$fl: not comparable on both networks, no line ($nums)";;
    *) verdict_line "ROUTE PARITY $fl: FAILED ($what; verdict $v; no-op $noop; chain_routes_v5; $nums)";;
  esac
done
say "done"
