#!/bin/bash
# TAG scout-native run_first_stack_v3 (2026-10-12; = v2 with announcement wording that quotes no trigger or verdict text (memory
# p2-1012-never-quote-trigger-strings: watchers match TEAM_LOG unanchored); v2 header follows; lead 19:3xZ / 19:5xZ: FIRST STACK = S51 + gk (card) + SN3 (native), runner scout-native,
# the engineer's scripts read-only; = v1 reordered per the lead's 19:5xZ GO). Every native run SERIAL, one process, BLAS threads = 1,
# through O/scout_native/run_cert_v2.py with PB_ALL=1 (every prebind key bound, so bills compare bit for bit across serial runs), gate.sh +
# guard_footprint_v2 (7200 s), outputs in O/scout_native/cert_fs_*.json; one '[scout-native]' TEAM_LOG line per result.
#   P.  gk torch parity anchor NOW: S50 safe fb277a8a vs s49/gk11s50pbs on 7100584, parity_verdict_v3.py vs parity_gk_v1/ (corr >= 0.70,
#       size 1/3..3x, floor 2e-6). FAIL -> gk out: the stack becomes S51 + SN3 (stack_deltas_sn3only_v1.json).
#   NS. gk strength-0 no-op on S51's SAFE SOURCE NOW: make_gsk_v1.py jmS51_v1 -> s49/gsk0s51s; pred sha + bill bit for bit on 7100521.
#   --- wait for '[engineer] <stamp> S51 FILES READY FOR RELEASE' ---
#   NT. gk strength-0 no-op on S51's TWIN FINAL: s49/gsk0s51twx vs jmS51_v1tw_x on 7100136, bit for bit.
#   SN. SN3 on S51's twin final (copy O/scout_native/sn3_s51tw = jmS51_v1tw_x + fragment_sn3_v1.py) on 7100136: pred sha = the parent's.
#   C.  the engineer's stack_build_v2.sh s51k1 51 <deltas> (unmodified): sources, census, refresh path, counts, FASTGATE line, seal.
# Stop file: O/scout_native/STOP_FIRST_STACK (checked between runs).
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
set -u
P2=/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16
O=$P2/opus_reexam_20261001_v1; E=$O/engineer; A=$E/a62_modules; S=$O/scout_native; PY=/Users/matt/arc-white-box-estimation/.venv/bin/python
LOG=$O/TEAM_LOG.md; N=51; H=jmS${N}_v1; TAG=${TAG:-s51k1}
say() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }
post() { echo "[scout-native] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*" >> $LOG; say "POSTED: $*"; }
halt() { [ -e $S/STOP_FIRST_STACK ] && { say "stop file"; exit 2; }; return 0; }
run1() {  # <candidate.py> <out tag> <seeds>: one serial native run
  [ -e $S/cert_fs_$2.json ] && return 0
  halt; bash $O/gate.sh 5 scout-native || return 3
  until [ $(memory_pressure | tail -1 | grep -o '[0-9]*%' | tr -d '%') -ge 55 ]; do sleep 60; done
  cd $P2 && PB_ALL=1 nice -n 10 $PY -B $O/release/guard_footprint_v2.py \
    --record $S/cert_fs_$2_guard.json --seconds 7200 --rss-backstop 7500000000 \
    $P2/astra_phase2_reexamination_v1/openblas_runtime_env_v1/bin/python -B $S/run_cert_v2.py $1 $3 $S/cert_fs_$2.json > $S/cert_fs_$2.log 2>&1
}
same() { $PY -c "
import json
a = json.load(open('$1'))['rows'][0]; b = json.load(open('$2'))['rows'][0]
print('bit for bit (pred %s, bill %d)' % (a['pred_sha'], a['flops']) if (a['pred_sha'], a['flops']) == (b['pred_sha'], b['flops']) else 'DIFFERENT (pred %s vs %s, bill %d vs %d)' % (b['pred_sha'], a['pred_sha'], b['flops'], a['flops']))" 2>&1 | tail -1; }
foot() { $PY -c "
import json
g = json.load(open('$S/cert_fs_$1_guard.json')); print('peak footprint %.2f GB, rss %.2f GB' % (g['max_footprint_bytes'] / 1e9, g['max_rss_bytes'] / 1e9))" 2>&1 | tail -1; }
say "start (tag $TAG)"
# P. parity anchor
run1 $A/s49/jmS50_v1_pbs/candidate.py gkp_base_584 7100584 || { post "FIRST STACK STOPPED: parity base run failed"; exit 1; }
run1 $A/s49/gk11s50pbs/candidate.py gkp_arm_584 7100584 || { post "FIRST STACK STOPPED: parity arm run failed"; exit 1; }
par=$(cd $A && PREFIX=a_RS51 BASEPREFIX=a_RS51 $PY -B $A/parity_verdict_v3.py $A/parity_gk_v1 gk $S/cert_fs_gkp_base_584.json $S/cert_fs_gkp_arm_584.json 7100584 2>&1)
echo "$par" > $S/parity_gk_s50_584.out
pv=$(echo "$par" | grep '^VERDICT' | sed 's/^VERDICT [^:]*: //'); pn=$(echo "$par" | grep -v '^VERDICT' | tr '\n' ';' | cut -c1-300)
pw=failed; [ "$pv" = PASS ] && pw=ok
post "FIRST STACK step P, gk torch match on 7100584 (S50 safe fb277a8a vs engineer s49/gk11s50pbs, the engineer's verdict tool v3 against parity_gk_v1/, corr >= 0.70, size 1/3..3x): $pw; details in O/scout_native/parity_gk_s50_584.out"
SPEC=$S/stack_deltas_first_v1.json
if [ "$pv" != PASS ]; then post "FIRST STACK: gk OUT on its torch match (lead 19:5xZ); the stack becomes S51 + SN3"; SPEC=$S/stack_deltas_sn3only_v1.json; fi
# NS. no-op on S51's safe source
if [ "$SPEC" = $S/stack_deltas_first_v1.json ]; then
  [ -e $A/s49/gsk0s${N}s ] || $PY -B $A/make_gsk_v1.py $A/s49/$H/candidate.py gsk0s${N}s > $S/make_gsk0s${N}s.out 2>&1 || { post "FIRST STACK STOPPED: make_gsk_v1 on $H failed"; exit 1; }
  run1 $A/s49/$H/candidate.py gkref_s${N}_521 7100521 || exit 1
  run1 $A/s49/gsk0s${N}s/candidate.py gsk0_s${N}_521 7100521 || exit 1
  n1=$(same $S/cert_fs_gkref_s${N}_521.json $S/cert_fs_gsk0_s${N}_521.json)
  post "FIRST STACK step NS, gk strength-0 no-op on S$N's SAFE SOURCE (s49/gsk0s${N}s vs s49/$H) on 7100521: $n1"
  case "$n1" in "bit for bit"*) ;; *) post "FIRST STACK: gk OUT on its no-op certificate; the stack becomes S51 + SN3"; SPEC=$S/stack_deltas_sn3only_v1.json;; esac
fi
say "waiting for S$N's FILES READY line"
until grep -q -E "^\[engineer\] [0-9T:-]+Z S$N FILES READY FOR RELEASE" $LOG; do halt; sleep 120; done
# NT. no-op on S51's twin final
if [ "$SPEC" = $S/stack_deltas_first_v1.json ]; then
  [ -e $A/s49/gsk0s${N}twx ] || $PY -B $A/make_gsk_v1.py $A/s49/${H}tw_x/candidate.py gsk0s${N}twx > $S/make_gsk0s${N}twx.out 2>&1 || { post "FIRST STACK STOPPED: make_gsk_v1 on ${H}tw_x failed"; exit 1; }
  run1 $A/s49/${H}tw_x/candidate.py s${N}twx_136 7100136 || exit 1
  run1 $A/s49/gsk0s${N}twx/candidate.py gsk0tw_s${N}_136 7100136 || exit 1
  n2=$(same $S/cert_fs_s${N}twx_136.json $S/cert_fs_gsk0tw_s${N}_136.json)
  post "FIRST STACK step NT, gk strength-0 no-op on S$N's TWIN FINAL (s49/gsk0s${N}twx vs s49/${H}tw_x) on 7100136: $n2"
  case "$n2" in "bit for bit"*) ;; *) post "FIRST STACK: gk OUT on its twin no-op certificate; the stack becomes S51 + SN3"; SPEC=$S/stack_deltas_sn3only_v1.json;; esac
fi
# SN. SN3 on S51's twin final
mkdir -p $S/sn3_s51tw
[ -s $S/sn3_s51tw/candidate.py ] || cat $A/s49/${H}tw_x/candidate.py $S/fragment_sn3_v1.py > $S/sn3_s51tw/candidate.py
run1 $A/s49/${H}tw_x/candidate.py s${N}twx_136 7100136 || exit 1
run1 $S/sn3_s51tw/candidate.py sn3_s${N}twx_136 7100136 || exit 1
n3=$($PY -c "
import json
a = json.load(open('$S/cert_fs_s${N}twx_136.json'))['rows'][0]; b = json.load(open('$S/cert_fs_sn3_s${N}twx_136.json'))['rows'][0]
print(('pred UNCHANGED %s' % a['pred_sha'] if a['pred_sha'] == b['pred_sha'] else 'pred CHANGED %s vs %s' % (b['pred_sha'], a['pred_sha'])) + ', bill %+.4f%%, ops %+d' % (100 * (b['flops'] / a['flops'] - 1), b['ops'] - a['ops']))" 2>&1 | tail -1)
post "FIRST STACK step SN, SN3 on S$N's TWIN FINAL (O/scout_native/sn3_s51tw = s49/${H}tw_x + fragment_sn3_v1.py sha 5e4a01b1970b) on 7100136: $n3; $(foot sn3_s${N}twx_136) vs base $(foot s${N}twx_136); override check: check_sn3_parent_v1 'append SN3' on jmS51_v1 / jmS51_v1tw"
case "$n3" in "pred UNCHANGED"*) ;; *) post "FIRST STACK STOPPED: SN3 changed S$N's predictions"; exit 1;; esac
# C. the stack (the engineer's builder, unmodified)
halt
post "FIRST STACK step C, starting the engineer's stack_build_v2.sh $TAG $N $(basename $SPEC) (deltas: $($PY -c "import json; print(' + '.join(d['id'] + ' (' + d['carry'] + ')' for d in json.load(open('$SPEC'))['deltas']))"))"
bash $A/stack_build_v2.sh $TAG $N $SPEC
say "stack builder exit=$?"
