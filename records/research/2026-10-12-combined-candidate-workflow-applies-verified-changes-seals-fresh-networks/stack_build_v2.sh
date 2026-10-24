#!/bin/bash
# TAG opus_engineer a62 stack_build_v2 (file written 2026-10-12T17:40Z; lead 17:3xZ decisions 2-3, release 17:3xZ) = stack_build_v1.sh with:
#   - stack_deltas_v2.py (carry native / card, ABSORBED lines, the ledger stack_ledger_v1.jsonl; SKIPPED / DROPPED ids in DETAILS);
#   - the request-key comparison on key uses only (release: the prebind binds keys until a 3 s WALL guard, so census ops / flops depend on
#     the box load; key uses do not);
#   - waits for the port loop to be idle before its first native launch (round ports keep first claim on the slots);
#   - the SEAL after the trigger line, in parallel with release's fastgate: seal_judge_v1.py alloc 8 fresh networks of the analyst's block
#     (seal_block_v1.json), pre-register, native S vs S' (gate_rdz, final MSE) + the final twins' bill ratio on 35 / 100 from the count
#     sessions, radlab's rule (FIRE iff mean < 0; 0 .. +1 se -> 8 more), one line 'STACK <tag> SEALED: <x>% (se, k of n lower) vs ...'.
# v1 header: (file written 2026-10-12T17:31Z; lead 17:0xZ cadence + release 17:2xZ fastgate_v1): the FAST INTEGRATION PATH.
# A stack = round N's port outputs (s49/jmS<N>_v1 safe source, _pbs safe final, tw twin source, tw_x twin final) + a list of verified deltas
# (stack_deltas_v1.py: literals / fragments / edit tools, checked against BOTH physics files). Steps:
#   1. sources S' = S + D, TW' = TW + D (edit tools first, then the fragments by make_s49_v1.py);
#   2. census of S' and TW' on list a (two processes): S' is release's fastgate base (same physics as the twin); their request keys vs the
#      round's own censuses (quick_keys_s<N>s_a / s<N>tw_a, every network: key uses and ops) decide the path:
#        same keys and no edit -> the round's prebind lists are still exact: finals SP' = SP + D, TX' = TX + D (fragments after the round's
#        prebind + XSG);
#        else (REFRESH)        -> census 660-669 of both, own prebind lists (prebind_targeted_v2), XSG on the twin, byte identity of both
#                                 finals against their sources (7100136), as the port loop does;
#   3. counts of TX' (normal ps_<tag>twx_n --short-setup / slow ps_<tag>twx_s --setup-slowdown 1.3, ranks 35,100, release/boundary_run.py)
#   4. fastgate_post_v1.py checks release's pin and posts '[engineer] <stamp> FASTGATE <tag>tw READY: twin <path> <sha8>, base <quick json>,
#      normal <dir>, slow <dir>'; then one 'STACK <tag> DETAILS' line (deltas, path, list-a MSE of S' vs S on the 10 networks, bill on 35/100).
# Any failure: one 'STACK <tag> STOPPED: ...' line. Native launches wait for a free slot (this lane: at most 2 native processes) and gate.sh.
# usage: stack_build_v1.sh <tag, e.g. s51k1> <round N> <deltas.json>
set -u
P2=/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16
O=$P2/opus_reexam_20261001_v1; E=$O/engineer; A=$E/a62_modules; PY=/Users/matt/arc-white-box-estimation/.venv/bin/python; LOG=${STACK_LOG:-$O/TEAM_LOG.md}   # STACK_LOG: a rehearsal posts elsewhere
NATIVE="^($P2/astra_phase2_reexamination_v1/openblas_runtime_env_v1/bin/python -B .*/engineer/|$P2/data_gate/.client-venv/bin/python -B -u .*boundary_client[.]py .*/engineer/)"   # gate / census processes and count-session clients
tag=$1; N=$2; spec=$3; H=jmS${N}_v1
say() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }
post() { case "$*" in *"PARITY"*) say "REFUSED to post a line with a parity prefix";; *) echo "[engineer] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*" >> $LOG; say "POSTED: $*";; esac; }
stop() { post "STACK $tag STOPPED: $*"; exit 1; }
slot() { until [ $(ps -axo command= | grep -c -E "$NATIVE") -lt 2 ]; do sleep 60; done; }
census() { # <candidate.py> <name> <part a|660>: one process, one retry
  local nets=651,378,424,328,287,641,548,557,316,390
  [ $3 = 660 ] && nets=660,661,662,663,664,665,666,667,668,669
  [ -e $E/logs/quick_keys_$2_$3.json ] && return 0
  slot; bash $E/run_quick_v3.sh "$1" $nets keys_$2_$3 --keys > $A/census_$2_$3.out 2>&1 && return 0
  sleep 300; slot; bash $E/run_quick_v3.sh "$1" $nets keys_$2_$3r --keys > $A/census_$2_$3r.out 2>&1 || return 1
  cp $E/logs/quick_keys_$2_$3r.json $E/logs/quick_keys_$2_$3.json
  for f in $E/logs/quick_keys_$2_$3r_pred_*.npy; do cp $f ${f/_$3r_pred_/_$3_pred_}; done
}
gate1() {  # <candidate.py> <out tag> <seed>: one native gate run
  [ -e $A/gate_rdz_$2.json ] && return 0
  slot; bash $O/gate.sh 7 engineer || return 3
  cd $P2 && OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 $PY -B $O/release/guard_footprint_v2.py \
    --record $A/gate_rdz_$2_guard.json --seconds 7200 --rss-backstop 7500000000 \
    $P2/astra_phase2_reexamination_v1/openblas_runtime_env_v1/bin/python -B $A/gate_rdz_v1.py $1 none none $3 $A/gate_rdz_$2.json > $A/gate_rdz_$2.log 2>&1
}
S=$A/s49/$H/candidate.py; SP=$A/s49/${H}_pbs/candidate.py; TW=$A/s49/${H}tw/candidate.py; TX=$A/s49/${H}tw_x/candidate.py
for f in $S $SP $TW $TX $E/logs/quick_keys_s${N}s_a.json $E/logs/quick_keys_s${N}tw_a.json; do [ -f $f ] || stop "round $N file missing: $f"; done
[ -e $A/s49/${tag}s ] && stop "tag $tag used already"
out=$($PY -B $A/stack_deltas_v2.py "$spec" $tag $N $S $TW 2>&1) || stop "deltas: $(echo "$out" | tail -1 | cut -c1-300)"
frags=$(echo "$out" | sed -n 's/^FRAGMENTS //p'); edits=$(echo "$out" | sed -n 's/^EDITS //p'); refresh=$(echo "$out" | sed -n 's/^REFRESH //p')
skipped=$(echo "$out" | sed -n 's/^SKIPPED //p'); dropped=$(echo "$out" | sed -n 's/^DROPPED //p')
ids=$($PY -c "import json, sys; print(' '.join(d['id'] for d in json.load(open(sys.argv[1]))['deltas']))" "$spec")
say "stack $tag on round $N: deltas $ids; fragments [$frags]; edits [$edits]; refresh declared $refresh"
pS=$S; pT=$TW; i=0
for e in $edits; do
  i=$((i + 1)); tool=${e%%:*}; vals=${e#*:}
  $PY -B $A/$tool $pS ${tag}s_e$i $vals > $A/stack_${tag}_e$i.out 2>&1 || stop "edit $tool on the safe source (a62_modules/stack_${tag}_e$i.out)"
  $PY -B $A/$tool $pT ${tag}tw_e$i $vals >> $A/stack_${tag}_e$i.out 2>&1 || stop "edit $tool on the twin source (a62_modules/stack_${tag}_e$i.out)"
  pS=$A/s49/${tag}s_e$i/candidate.py; pT=$A/s49/${tag}tw_e$i/candidate.py
done
$PY -B $A/make_s49_v1.py $pS ${tag}s $frags > /dev/null || stop "safe source build"
$PY -B $A/make_s49_v1.py $pT ${tag}tw $frags > /dev/null || stop "twin source build"
S2=$A/s49/${tag}s/candidate.py; T2=$A/s49/${tag}tw/candidate.py
until tail -1 $(ls -t $A/chain_rad_port_v*.out | head -1) | grep -q -E "waiting for its verdict file|rounds done|loop done"; do sleep 120; done
say "port loop idle: native work starts"
census $S2 ${tag}s a &
sleep 5
census $T2 ${tag}tw a &
wait
for f in ${tag}s_a ${tag}tw_a; do [ -f $E/logs/quick_keys_$f.json ] || stop "census $f failed (a62_modules/census_$f.out)"; done
cmp_keys() { $PY -c "
import json, sys
a, b = (json.load(open(p))['rows'] for p in sys.argv[1:3])
k = lambda rows: {r['rank']: r['key_uses'] for r in rows}
print('SAME' if k(a) == k(b) else 'CHANGED on ' + ' '.join(str(r) for r in sorted(k(a)) if k(a)[r] != k(b).get(r)))" $E/logs/quick_keys_$1.json $E/logs/quick_keys_$2.json; }
ks=$(cmp_keys s${N}s_a ${tag}s_a); kt=$(cmp_keys s${N}tw_a ${tag}tw_a)
say "request keys vs round $N on list a: safe $ks; twin $kt"
if [ -z "$edits" ] && [ "$refresh" = no ] && [ "$ks" = SAME ] && [ "$kt" = SAME ]; then
  path="no refresh (request keys unchanged on list a)"
  $PY -B $A/make_s49_v1.py $SP ${tag}s_pbs $frags > /dev/null || stop "safe final build"
  $PY -B $A/make_s49_v1.py $TX ${tag}tw_x $frags > /dev/null || stop "twin final build"
else
  path="REFRESH (keys: safe $ks, twin $kt; edits [$edits])"
  census $S2 ${tag}s 660 &
  sleep 5
  census $T2 ${tag}tw 660 &
  wait
  for f in ${tag}s_660 ${tag}tw_660; do [ -f $E/logs/quick_keys_$f.json ] || stop "census $f failed"; done
  cd $A
  $PY -B $A/prebind_targeted_v2.py $S2 $O/release/runs/c_s50s3_safe/C0/manifest.json $A/fragment_prebind_${tag}s_v1.py \
    $E/logs/quick_keys_${tag}s_a.json $E/logs/quick_keys_${tag}s_660.json > $A/prebind_${tag}s_v1.out 2>&1 || stop "safe prebind list"
  $PY -B $A/make_s49_v1.py $S2 ${tag}s_pbs $A/fragment_prebind_${tag}s_v1.py > /dev/null || stop "safe final build"
  $PY -B $A/prebind_targeted_v2.py $T2 $O/release/runs/c_s50tw_safe/C0/manifest.json $A/fragment_prebind_${tag}tw_v1.py \
    $E/logs/quick_keys_${tag}tw_a.json $E/logs/quick_keys_${tag}tw_660.json > $A/prebind_${tag}tw_v1.out 2>&1 || stop "twin prebind list"
  $PY -B $A/make_s49_v1.py $T2 ${tag}tw_pbt $A/fragment_prebind_${tag}tw_v1.py > /dev/null || stop "twin prebind build"
  $PY -B $A/make_xsg_any_v1.py $A/s49/${tag}tw_pbt/candidate.py ${tag}tw_x > $A/xsg_${tag}tw.out 2>&1 || stop "XSG on the twin"
  ( gate1 $S2 ${tag}s_136 7100136; gate1 $A/s49/${tag}s_pbs/candidate.py ${tag}spbs_136 7100136 ) &
  sleep 5
  ( gate1 $T2 ${tag}tw_136 7100136; gate1 $A/s49/${tag}tw_x/candidate.py ${tag}twx_136 7100136 ) &
  wait
  ident=$($PY -c "
import json, sys
r = lambda t: json.load(open('$A/gate_rdz_%s.json' % t))['rows'][0]['pred_sha']
print(' '.join('%s %s' % (b, 'BYTE-IDENTICAL' if r(a) == r(b) else 'DIFFERENT') for a, b in (('${tag}s_136', '${tag}spbs_136'), ('${tag}tw_136', '${tag}twx_136'))))" 2>&1 | tail -1)
  case "$ident" in *DIFFERENT*|*rror*) stop "identity of the finals vs their sources on 7100136: $ident";; esac
  path="$path; identity on 7100136: $ident"
fi
SP2=$A/s49/${tag}s_pbs/candidate.py; TX2=$A/s49/${tag}tw_x/candidate.py
for k in "n 35,100" "s 35,100 --setup-slowdown 1.3"; do
  set -- $k; slot
  bash $A/run_boundary_counts_v5.sh $TX2 ps_${tag}twx_$1 $2 ${3:+"$3 $4"} > $A/counts_${tag}twx_$1.out 2>&1 &
  sleep 5
done
wait
line=$($PY -B $A/fastgate_post_v1.py ${tag}tw $TX2 $E/logs/quick_keys_${tag}s_a.json $E/a56_req/runs/ps_${tag}twx_n $E/a56_req/runs/ps_${tag}twx_s 2>&1 | tail -1)
case "$line" in "FASTGATE "*" READY: "*) post "$line";; *) stop "fastgate checks: $line";; esac
mse=$($PY -c "
import json, math, sys
a, b = ({r['rank']: r for r in json.load(open(p))['rows']} for p in sys.argv[1:3])
d = [math.log(b[k]['mse'] / a[k]['mse']) for k in sorted(a) if k in b]
m = sum(d) / len(d); se = (sum((x - m) ** 2 for x in d) / (len(d) - 1) / len(d)) ** 0.5
print('%+.2f%% (mean log ratio, se %.2f%%, lower on %d of %d)' % (100 * m, 100 * se, sum(x < 0 for x in d), len(d)))" $E/logs/quick_keys_s${N}s_a.json $E/logs/quick_keys_${tag}s_a.json 2>&1 | tail -1)
bill=$($PY -c "
import json, sys
f = lambda d: {json.loads(l)['rank']: json.loads(l)['flops'] for l in open(d + '/rows.jsonl') if '\"rank\"' in l}
a, b = f(sys.argv[1]), f(sys.argv[2])
print(', '.join('%d %+.3f%%' % (r, 100 * (b[r] / a[r] - 1)) for r in sorted(b) if r in a))" $E/a56_req/runs/ps_s${N}twx_n $E/a56_req/runs/ps_${tag}twx_n 2>&1 | tail -1)
post "STACK $tag DETAILS (stack_build_v1, round $N): deltas $ids (skipped as absorbed: [$skipped]; DROPPED since the last build: [$dropped]); path: $path; list-a final MSE of the stack safe vs S$N's safe: $mse; twin bill vs S$N's twin on 35 / 100: $bill; safe final for a safe fire: $SP2 $(shasum -a 256 $SP2 | cut -c1-8) (counts not run)"
until [ -f $A/seal_block_v1.json ]; do sleep 120; done
ratio=$($PY -c "
import json, sys
f = lambda d: {json.loads(l)['rank']: json.loads(l)['flops'] for l in open(d + '/rows.jsonl') if '\"rank\"' in l}
a, b = f(sys.argv[1]), f(sys.argv[2])
print(sum(b[r] / a[r] for r in (35, 100)) / 2)" $E/a56_req/runs/ps_s${N}twx_n $E/a56_req/runs/ps_${tag}twx_n) || stop "seal: no bill ratio"
seeds=$($PY -B $A/seal_judge_v1.py alloc $tag 8) || stop "seal: no fresh networks left in seal_block_v1.json"
$PY -B $A/seal_judge_v1.py prereg $tag $S $S2 $seeds $ratio > /dev/null || stop "seal pre-registration"
for x in ${seeds//,/ }; do until [ -f $O/analyst/fresh/truth_seed$x.npz ]; do sleep 120; done; done   # the analyst's truths (block 7190000-031)
gate1 $S seal_${tag}_b1 $seeds &
sleep 5
gate1 $S2 seal_${tag}_s1 $seeds &
wait
bj=$A/gate_rdz_seal_${tag}_b1.json; sj=$A/gate_rdz_seal_${tag}_s1.json
[ -f $bj ] && [ -f $sj ] || stop "seal runs failed (a62_modules/gate_rdz_seal_${tag}_b1.log / _s1.log)"
v=$($PY -B $A/seal_judge_v1.py judge $tag $bj $sj 2>&1 | tail -1)
case "$v" in *EXTEND)
  seeds2=$($PY -B $A/seal_judge_v1.py alloc ${tag}x 8) || stop "seal: no networks for the extension"
  for x in ${seeds2//,/ }; do until [ -f $O/analyst/fresh/truth_seed$x.npz ]; do sleep 120; done; done
  gate1 $S seal_${tag}_b2 $seeds2 &
  sleep 5
  gate1 $S2 seal_${tag}_s2 $seeds2 &
  wait
  [ -f $A/gate_rdz_seal_${tag}_b2.json ] && [ -f $A/gate_rdz_seal_${tag}_s2.json ] || stop "seal extension runs failed"
  v=$($PY -B $A/seal_judge_v1.py judge $tag $bj,$A/gate_rdz_seal_${tag}_b2.json $sj,$A/gate_rdz_seal_${tag}_s2.json 2>&1 | tail -1);;
esac
case "$v" in MEAN*) ;; *) stop "seal judge: $v";; esac
post "STACK $tag SEALED: $(echo "$v" | sed -E 's/^MEAN ([^ ]+) SE ([^ ]+) LOWER ([0-9]+) OF ([0-9]+) -> (.*)$/\1 (se \2, \3 of \4 lower) vs S'$N' round files (safe source '$(shasum -a 256 $S | cut -c1-8)'), networks '$seeds${seeds2:+,$seeds2}'; pre-registered rule (a62_modules\/seal_'$tag'_PREREG.json): \5/')"
say "done"
