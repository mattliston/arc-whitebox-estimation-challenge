#!/bin/bash
# TAG scout-native chain_round_stack_v4 (2026-10-13 10:2xZ, lead 10:2xZ: K4L-inner joins s54k1, its layer set fixed before T141's verdict)
# = v3 where the 6th argument may be @<decision file>: before the build the chain waits for that file, written by the lead, whose first
# line picks the new fragments: L8-14 -> fragment_k4li_v1.py; L12-14 -> fragment_k4li_v1.py + fragment_k4li_l1214_v1.py; none -> none;
# anything else STOPs. A plain comma list works as in v3. v3 header:
# TAG scout-native chain_round_stack_v3 (2026-10-13 10:1xZ, lead 09:5xZ: a new card fragment may join the round stack) = v2 + an optional
# 6th argument: a comma list of NEW fragments appended after ASZ (e.g. fragment_k4li_v1.py: K4L ages 3-4 at the inner layers). They are
# physics (not in the byte-identical subset) and new (not compared with the fired twin by the context check); the 5th argument may be '-'
# for the default seal rule. v2 header:
# TAG scout-native chain_round_stack_v2 (2026-10-13 08:2xZ) = v1 with two yield rules from the lanes whose slots it shares:
#   - engineer2: the census (step 1) starts only after the port loop's own census is done (engineer/logs/quick_keys_s<N>s_660.json and
#     quick_keys_s<N>tw_660.json both on disk), because the port loop's slot() counts every native whose command line holds /engineer/ (our
#     candidates live in a62_modules/s49) and the port census is on the fire path;
#   - release: with seal_<tag>_PAR the stack side runs on this lane's slot at once, and the base side (release's lent slot) starts only while
#     no release gate runs (pgrep fastgate_v*.sh / s47_safe_v*.sh / twin_pipeline_v*.sh); if the stack side ends first, the base side runs
#     on this lane's slot instead.
# v1 header: (2026-10-13; lead 07:40Z: "S53's point carries into S54 ..., whose stack will carry everything")
# = chain_s53k1_v2.sh for any round N: the round's twin source + every carried delta, as one unattended job.
#   0. waits for s49/jmS<N>_v1tw (the port loop builds it first); idc1 regenerated from it (make_idc_sn_v1.py); s49/<tag>tw = source +
#      SN3, rows2, npx1, Astra's block, idc1, k4lf34h, HZT, ASZ (the fired twins' order); XSG -> s49/<tag>tw_xs; context check
#      (check_stack_vs_reference_v2.py) against the fired twin: any DIFF in an override or copy of a VERBATIM carried fragment -> STOP
#      (idc1 is regenerated per round, so its DIFFs are the round's own changes);
#   1. census on list a and 660-669 (--keys) -> list-a final MSE vs release's quick_<fired>_a.json (info line);
#   2. own prebind list -> s49/<tag>tw_x;
#   3. after the port loop's line for S<N>'s files: subset s49/<tag>b_x = jmS<N>_v1tw_x + the byte-identical deltas; run_cert_v2 PB_ALL=1 on
#      7100136 of jmS<N>_v1tw_x, the subset and the stack: the subset must give the same predictions (pred sha, 16/16 layers) or STOP;
#   4. count sessions, fastgate_post_v1.py (NORMAL mode, base release's quick_<fired>_a.json), details line, trigger line (engineer's shape);
#   5. seal vs the fired twin: seal_alloc_sn_v1.py from <block json>, pre-registration BEFORE any seal run, run_cert_v2 PB_ALL=1 base and
#      stack (both at once if O/scout_native/seal_<tag>_PAR exists), seal_judge_sn_v2.py: with <fire at %> FIRE iff mean <= it (no extension),
#      else FIRE iff mean < 0 with one 8-network extension when 0 <= mean < +1 se.
# ONE native process at a time unless noted (guard v3, BLAS 1, nice 10, gate.sh + memory_pressure >= 50%). Any failure: one
# 'STACK <tag> STOPPED: ...' line.
# usage: chain_round_stack_v3.sh <round N> <tag, e.g. s54k1> <fired twin tag, e.g. s51k5tw> <block json> [<fire at %, e.g. -0.10, or ->] [<new fragments, comma list>]
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
set -u
N=$1; TAG=$2; FT=$3; BLOCK=$4; FIRE_AT=${5:-}; [ "$FIRE_AT" = - ] && FIRE_AT=
X6=${6:-}; DEC=; [ "${X6:0:1}" = @ ] && DEC=${X6:1}
EXTRA=; [ -z "$DEC" ] && EXTRA=$(echo "$X6" | tr ',' ' ')
XNAMES=$(for f in $EXTRA; do basename $f .py; done | tr '\n' ' ')
P2=/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16
O=$P2/opus_reexam_20261001_v1; E=$O/engineer; A=$E/a62_modules; S=$O/scout_native; PY=/Users/matt/arc-white-box-estimation/.venv/bin/python
LOG=$O/TEAM_LOG.md; UP=$(echo $TAG | tr a-z A-Z)
H=$A/s49/jmS${N}_v1tw/candidate.py; PT=$A/s49/jmS${N}_v1tw_x/candidate.py
XS=$A/s49/${TAG}tw_xs/candidate.py; TX=$A/s49/${TAG}tw_x/candidate.py; SUB=$A/s49/${TAG}b_x/candidate.py
FIRED=$A/s49/${FT}_x/candidate.py; BASEQ=$O/release/runs/quick_${FT}_a.json; IDC=$S/fragment_idc_idc1s${N}tw.py
VERBATIM="$S/fragment_sn3_v1.py $S/fragment_rows_v2.py $S/fragment_npx_v1.py $S/fragment_astra_stack1_v1.py $S/fragment_k4lf_v1.py $S/fragment_astra_hzt_v2.py $S/fragment_astra_asz_v1.py"
say() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }
post() { echo "[scout-native] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*" >> $LOG; say "POSTED: $*"; }
stop() { post "STACK $TAG STOPPED: $*"; exit 1; }
run1() {  # <candidate.py> <out tag> <seeds>
  [ -e $S/cert_fs_$2.json ] && return 0
  local n=$(echo "$3" | tr ',' '\n' | wc -l | tr -d ' ')
  bash $O/gate.sh 5 scout-native || return 3
  until [ $(memory_pressure | tail -1 | grep -o '[0-9]*%' | tr -d '%') -ge 50 ]; do sleep 60; done
  say "run $2 ($1, $3)"
  cd $P2 && PB_ALL=1 nice -n 10 $PY -B $O/release/guard_footprint_v3.py \
    --record $S/cert_fs_$2_guard.json --seconds $((7200 + 900 * n)) --rss-backstop 7500000000 \
    $P2/astra_phase2_reexamination_v1/openblas_runtime_env_v1/bin/python -B $S/run_cert_v2.py $1 $3 $S/cert_fs_$2.json > $S/cert_fs_$2.log 2>&1
  say "done $2 (exit $?)"
}
say "start: round $N, tag $TAG, fired $FT, block $BLOCK, fire at ${FIRE_AT:-<0 with extension}, new fragments: ${DEC:+from the decision file $DEC}${DEC:-${XNAMES:-none}}"
if [ -n "$DEC" ]; then
  until [ -f $DEC ]; do sleep 30; done
  d=$(head -1 $DEC | tr -d ' \r')
  case "$d" in
    L8-14) EXTRA="$S/fragment_k4li_v1.py";;
    L12-14) EXTRA="$S/fragment_k4li_v1.py $S/fragment_k4li_l1214_v1.py";;
    none) EXTRA=;;
    *) stop "decision file $DEC: unknown first line '$d' (L8-14 / L12-14 / none)";;
  esac
  XNAMES=$(for f in $EXTRA; do basename $f .py; done | tr '\n' ' ')
  say "decision '$d' -> new fragments: ${XNAMES:-none}"
fi
for f in $EXTRA; do [ -f $f ] || stop "new fragment missing: $f"; done
[ -f $FIRED ] || stop "fired twin missing: $FIRED"
# 0. build
until [ -f $H ] && [ -f $(dirname $H)/build.json ]; do sleep 60; done; sleep 30   # build.json is written after the candidate
[ -e $IDC ] || $PY -B $S/make_idc_sn_v1.py $H idc1s${N}tw > $S/make_idc_s${N}tw.out 2>&1 || stop "idc1 for S$N (O/scout_native/make_idc_s${N}tw.out)"
FRAGS="$S/fragment_sn3_v1.py $S/fragment_rows_v2.py $S/fragment_npx_v1.py $S/fragment_astra_stack1_v1.py $IDC $S/fragment_k4lf_v1.py $S/fragment_astra_hzt_v2.py $S/fragment_astra_asz_v1.py $EXTRA"
[ -e $A/s49/${TAG}tw ] || $PY -B $A/make_s49_v1.py $H ${TAG}tw $FRAGS > $S/make_${TAG}tw.out 2>&1 || stop "source build (O/scout_native/make_${TAG}tw.out)"
[ -e $A/s49/${TAG}tw_xs ] || $PY -B $A/make_xsg_any_v1.py $A/s49/${TAG}tw/candidate.py ${TAG}tw_xs > $S/xsg_${TAG}tw.out 2>&1 || stop "XSG (O/scout_native/xsg_${TAG}tw.out)"
nice -n 10 $PY -B $S/check_stack_vs_reference_v2.py $XS $FIRED > $S/context_${TAG}tw.out 2>&1 || stop "context check failed to run (O/scout_native/context_${TAG}tw.out)"
bad=$($PY -c "
import sys
names = [p.rsplit('/', 1)[1][:-3] for p in sys.argv[2:]]
bad = []
for line in open(sys.argv[1]):
    parts = line.split(' | ')
    if line.split(' ')[0] in names and ('DIFF' in parts[0] or 'DIFF' in parts[1]):
        bad.append(line.split(' ')[0] + ': ' + ' | '.join(parts[:2]).split(None, 1)[1].strip())
missing = [n for n in names if not any(l.startswith(n + ' ') for l in open(sys.argv[1]))]
print('; '.join(bad + ['%s not in both files' % n for n in missing]))" $S/context_${TAG}tw.out $VERBATIM 2>&1)
[ -z "$bad" ] || stop "context check vs $FT: $bad (O/scout_native/context_${TAG}tw.out)"
say "built: $(shasum -a 256 $XS | cut -c1-8); context check vs $FT clean"
# 1. census, after the port loop's own census (engineer2)
until [ -f $E/logs/quick_keys_s${N}s_660.json ] && [ -f $E/logs/quick_keys_s${N}tw_660.json ]; do
  grep -qE "^\[engineer2?\] [0-9T:Z-]+ S$N PORT STOPPED" $LOG && stop "the S$N port stopped (port loop line)"
  sleep 60
done
say "the port loop's census of S$N is done: census starts"
for part in "a 651,378,424,328,287,641,548,557,316,390" "660 660,661,662,663,664,665,666,667,668,669"; do
  set -- $part
  k=$S/logs/quick_keys_${TAG}tw_$1.json
  if [ ! -e $k ]; then
    say "census $1"
    bash $S/run_quick_v4s.sh $XS $2 keys_${TAG}tw_$1 --keys > $S/census_${TAG}tw_$1.out 2>&1
    if [ ! -e $k ]; then
      sleep 300
      bash $S/run_quick_v4s.sh $XS $2 keys_${TAG}tw_$1r --keys > $S/census_${TAG}tw_$1r.out 2>&1
      [ -e $S/logs/quick_keys_${TAG}tw_$1r.json ] && cp $S/logs/quick_keys_${TAG}tw_$1r.json $k
    fi
    [ -e $k ] || stop "census $1 of ${TAG}tw_xs failed (O/scout_native/census_${TAG}tw_$1.out)"
    if [ $1 = a ] && [ -e $BASEQ ]; then
      r=$($PY -c "
import json, math
a = {r['rank']: r for r in json.load(open('$BASEQ'))['rows']}; b = {r['rank']: r for r in json.load(open('$k'))['rows']}
d = [math.log(b[x]['mse'] / a[x]['mse']) for x in b if x in a]
m = sum(d) / len(d); se = (sum((x - m) ** 2 for x in d) / (len(d) - 1) / len(d)) ** 0.5
print('final-layer MSE vs the fired $FT (release native ten) %+.3f%% (mean log, se %.3f%%, lower on %d of %d); per network %s' % (
    100 * m, 100 * se, sum(x < 0 for x in d), len(d), ' '.join('%+.2f' % (100 * x) for x in d)))" 2>&1 | tail -1)
      post "$UP FIRST NATIVE READ (list a, census of s49/${TAG}tw_xs $(shasum -a 256 $XS | cut -c1-8) = S$N port + SN3, rows2, npx1, Astra's block, idc1 for S$N, k4lf34h, HZT, ASZ${XNAMES:+, $XNAMES}; quick.py, load-dependent bills not used): $r"
    fi
  fi
done
# 2. own prebind list
if [ ! -e $TX ]; then
  [ -e $S/fragment_prebind_${TAG}tw_v1.py ] || nice -n 10 $PY -B $A/prebind_targeted_v2.py $XS $O/release/runs/c_s50tw_safe/C0/manifest.json $S/fragment_prebind_${TAG}tw_v1.py \
    $S/logs/quick_keys_${TAG}tw_a.json $S/logs/quick_keys_${TAG}tw_660.json > $S/prebind_${TAG}tw_v1.out 2>&1 || stop "prebind list (O/scout_native/prebind_${TAG}tw_v1.out)"
  $PY -B $A/make_s49_v1.py $XS ${TAG}tw_x $S/fragment_prebind_${TAG}tw_v1.py > $S/make_${TAG}tw_x.out 2>&1 || stop "final build (O/scout_native/make_${TAG}tw_x.out)"
fi
sha8=$(shasum -a 256 $TX | cut -c1-8)
say "final ${TAG}tw_x $sha8; prebind: $(tail -3 $S/prebind_${TAG}tw_v1.out | tr '\n' ' ')"
# 3. the byte-identical subset on the round's own twin final; the stack on 7100136
until grep -qE "^\[engineer2?\] [0-9T:Z-]+ S$N FILES READY" $LOG && [ -f $PT ]; do
  grep -qE "^\[engineer2?\] [0-9T:Z-]+ S$N PORT STOPPED" $LOG && stop "the S$N port stopped (port loop line)"
  sleep 60
done
say "S$N port files ready"
[ -e $SUB ] || $PY -B $A/make_s49_v1.py $PT ${TAG}b_x $S/fragment_sn3_v1.py $S/fragment_rows_v2.py $S/fragment_npx_v1.py $IDC \
  $S/fragment_astra_hzt_v2.py $S/fragment_astra_asz_v1.py > $S/make_${TAG}b_x.out 2>&1 || stop "subset build (O/scout_native/make_${TAG}b_x.out)"
run1 $FIRED ${FT}x_136 7100136 || stop "run of the fired twin did not start"
run1 $PT s${N}twx_136 7100136 || stop "run of jmS${N}_v1tw_x did not start"
run1 $SUB ${TAG}bx_136 7100136 || stop "run of ${TAG}b_x did not start"
run1 $TX ${TAG}twx_136 7100136 || stop "run of ${TAG}tw_x did not start"
for t in ${FT}x_136 s${N}twx_136 ${TAG}bx_136 ${TAG}twx_136; do [ -e $S/cert_fs_$t.json ] || stop "run $t failed (O/scout_native/cert_fs_$t.log)"; done
ident=$($PY -c "
import json, numpy as np
S = '$S'
def load(t):
    r = json.load(open(S + '/cert_fs_%s.json' % t))['rows'][0]
    return r, np.load(S + '/cert_fs_%s_pred.npz' % t)['7100136'].astype(np.float64)[15]
(p, pp), (b, pb), (f, pf), (k5, p5) = load('s${N}twx_136'), load('${TAG}bx_136'), load('${TAG}twx_136'), load('${FT}x_136')
same = sum(p['mse_layers'][i] == b['mse_layers'][i] for i in range(16))
d = pf - pb
print('%s predictions: subset %s vs S$N final %s, %d/16 layer MSEs equal, bill %+.4f%%, requests %+d; full stack vs subset |d|/|pred| %.2e (d^2/MSE %.4f%%), MSE %+.4f%%, bill %+.4f%%; full stack vs the fired ${FT}_x: MSE %+.4f%%, bill %+.4f%%, requests %+d' % (
    'SAME' if p['pred_sha'] == b['pred_sha'] and same == 16 else 'DIFFERENT', b['pred_sha'], p['pred_sha'], same, 100 * (b['flops'] / p['flops'] - 1), b['ops'] - p['ops'],
    np.linalg.norm(d) / np.linalg.norm(pb), 100 * float(np.mean(d * d)) / b['mse_final'], 100 * (f['mse_final'] / b['mse_final'] - 1), 100 * (f['flops'] / b['flops'] - 1),
    100 * (f['mse_final'] / k5['mse_final'] - 1), 100 * (f['flops'] / k5['flops'] - 1), f['ops'] - k5['ops']))" 2>&1 | tail -1)
say "7100136: $ident"
case "$ident" in SAME*) ;; *) stop "byte-identical subset (SN3, rows2, npx1, idc1, HZT, ASZ on jmS${N}_v1tw_x) differs on 7100136: $ident";; esac
post "$UP CERTIFIED ON 7100136 (run_cert_v2 PB_ALL=1, guard v3): the byte-identical deltas on S$N's own twin final (s49/${TAG}b_x = jmS${N}_v1tw_x + SN3 + rows2 + npx1 + idc1 for S$N + HZT + ASZ) give the $ident; twin s49/${TAG}tw_x $sha8 (own prebind list; context check vs $FT clean)"
# 4. counts, fastgate trigger
until [ -e $BASEQ ]; do sleep 60; done
for k in "n 35,100" "s 35,100 --setup-slowdown 1.3"; do
  set -- $k
  bash $S/run_boundary_counts_v6.sh $TX ps_${TAG}twx_$1 $2 ${3:+"$3 $4"} > $S/counts_${TAG}twx_$1.out 2>&1 || stop "count session $1 (O/scout_native/counts_${TAG}twx_$1.out)"
  grep -q "exit=0" $S/counts_${TAG}twx_$1.out || stop "count session $1 did not exit 0"
  say "counts $1: $(tail -3 $S/counts_${TAG}twx_$1.out | tr '\n' ' ')"
done
line=$($PY -B $A/fastgate_post_v1.py ${TAG}tw $TX $BASEQ $E/a56_req/runs/ps_${TAG}twx_n $E/a56_req/runs/ps_${TAG}twx_s 2>&1 | tail -1)
case "$line" in "FASTGATE "*" READY: "*) ;; *) stop "fastgate checks: $line";; esac
cnt=$($PY -c "
import json
f = lambda d: {json.loads(l)['rank']: json.loads(l) for l in open(d + '/rows.jsonl') if '\"rank\"' in l}
out = []
for kind in ('n', 's'):
    a, b = f('$E/a56_req/runs/ps_${FT}x_' + kind), f('$E/a56_req/runs/ps_${TAG}twx_' + kind)
    out.append(kind + ' ' + ', '.join('%d: bill %+.4f%%, requests %+d' % (r, 100 * (b[r]['flops'] / a[r]['flops'] - 1), b[r]['requests'] - a[r]['requests']) for r in sorted(b) if r in a))
print('; '.join(out))" 2>&1 | tail -1)
post "$UP TWIN COUNTED: s49/${TAG}tw_x ($sha8) = S$N port + every carried delta${XNAMES:+ + $XNAMES}; counts vs the fired $FT (ps_${FT}x): $cnt; release's fastgate in NORMAL mode against quick_${FT}_a.json; the seal vs ${FT}_x starts now"
echo "[engineer] $(date -u +%Y-%m-%dT%H:%M:%SZ) $line" >> $LOG; say "POSTED: [engineer] $line"
# 5. seal
PRE=$S/seal_${TAG}_PREREG.json
if [ ! -e $PRE ]; then
  seeds=$($PY -B $S/seal_alloc_sn_v1.py $TAG 8 $BLOCK) || stop "seal: no fresh networks left in $BLOCK"
  $PY -c "
import json, hashlib
sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
fire_at = '$FIRE_AT'
d = dict(tag='$TAG', utc='$(date -u +%Y-%m-%dT%H:%M:%SZ)', base='$FIRED', base_sha256=sha('$FIRED'), stack='$TX', stack_sha256=sha('$TX'),
    stack_parts='S$N port twin (jmS${N}_v1tw) + SN3 + rows2 + npx1 + Astra block 6562cc7e + idc1 for S$N (make_idc_sn_v1) + k4lf34h + HZT + ASZ${XNAMES:+ + $XNAMES}, XSG, own prebind list',
    seeds=[int(x) for x in '$seeds'.split(',')], allocation='seal_alloc_sn_v1.py $TAG 8 $BLOCK (ledger a62_modules/seal_alloc_v1.jsonl)',
    runner='O/scout_native/run_cert_v2.py, PB_ALL=1, guard v3, BLAS threads 1',
    score='per network log(final MSE stack / base) + log(flops stack / flops base) from the same runs (bill x MSE)',
    rule=('FIRE iff the 8-network mean <= %s%%; otherwise skipped, no extension' % fire_at) if fire_at else '8 networks: FIRE iff mean < 0; 0 <= mean < +1 se -> 8 more, then FIRE iff the 16-network mean < 0; else NO FIRE')
if fire_at:
    d['fire_at_or_below_pct'] = float(fire_at)
json.dump(d, open('$PRE', 'w'), indent=1)"
fi
SEEDS=$($PY -c "import json; print(','.join(str(s) for s in json.load(open('$PRE'))['seeds']))")
seal_side() {  # <seeds> <suffix>
  for x in ${1//,/ }; do until [ -f $O/analyst/fresh/truth_seed$x.npz ]; do sleep 120; done; done
  if [ -e $S/seal_${TAG}_PAR ]; then
    run1 $TX seal_${TAG}_stack$2 $1 &
    local sp=$!
    sleep 90
    while kill -0 $sp 2> /dev/null && pgrep -f "fastgate_v[0-9].sh|s47_safe_v[0-9].sh|twin_pipeline_v[0-9].sh" > /dev/null; do sleep 30; done   # release: never start the lent side during a release gate
    say "base side starts ($(kill -0 $sp 2> /dev/null && echo "release's lent slot, the stack side still running" || echo "this lane's slot, the stack side done"))"
    run1 $FIRED seal_${TAG}_base$2 $1
    wait $sp
  else
    run1 $FIRED seal_${TAG}_base$2 $1
    run1 $TX seal_${TAG}_stack$2 $1
  fi
  [ -e $S/cert_fs_seal_${TAG}_base$2.json ] && [ -e $S/cert_fs_seal_${TAG}_stack$2.json ] || stop "seal runs incomplete (O/scout_native/cert_fs_seal_${TAG}_*$2.log)"
}
seal_side $SEEDS ""
BJ=$S/cert_fs_seal_${TAG}_base.json; SJ=$S/cert_fs_seal_${TAG}_stack.json
$PY -B $S/seal_judge_sn_v2.py $PRE $BJ $SJ > $S/seal_${TAG}_judge.out 2>&1
v=$(tail -1 $S/seal_${TAG}_judge.out)
case "$v" in *EXTEND)
  seeds2=$($PY -B $S/seal_alloc_sn_v1.py ${TAG}x 8 $BLOCK) || stop "seal: the extension has no fresh networks left in $BLOCK"
  $PY -c "
import json
d = json.load(open('$PRE')); d['extension_seeds'] = [int(x) for x in '$seeds2'.split(',')]
json.dump(d, open('$PRE', 'w'), indent=1)"
  seal_side $seeds2 _x
  $PY -B $S/seal_judge_sn_v2.py $PRE $BJ,$S/cert_fs_seal_${TAG}_base_x.json $SJ,$S/cert_fs_seal_${TAG}_stack_x.json > $S/seal_${TAG}_judge.out 2>&1
  v=$(tail -1 $S/seal_${TAG}_judge.out)
  SEEDS="$SEEDS,$seeds2";;
esac
cat $S/seal_${TAG}_judge.out
case "$v" in MEAN*) ;; *) stop "seal judge: $v";; esac
post "STACK $TAG SEALED: $(echo "$v" | sed -E 's/^MEAN ([^ ]+) SE ([^ ]+) LOWER ([0-9]+) OF ([0-9]+) -> (.*)$/\1 (se \2, \3 of \4 lower)/') vs the fired ${FT}_x on ${SEEDS//,/ } (pre-registered O/scout_native/seal_${TAG}_PREREG.json; score = log MSE ratio + log bill ratio per network, run_cert_v2 PB_ALL=1; twin s49/${TAG}tw_x $sha8 = S$N port + SN3, rows2, npx1, Astra's block, idc1 for S$N, k4lf34h, HZT, ASZ${XNAMES:+, $XNAMES}) -> ${v##*-> }"
say end
