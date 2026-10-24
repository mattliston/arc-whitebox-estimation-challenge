#!/bin/bash
# TAG scout-native build_byteid_stack_v1 (2026-10-13, lead 01:54Z: integrate Astra's measured byte-identical bill cuts as byte-identical stacks,
# like s51k2) = build_s51k2_byteid_v1.sh + build_s51k2_counts_v2.sh generalized:
#   1. make_s49_v1.py <parent twin> -> s49/<tag>tw_x with the given fragments appended (after the parent's prebind list and XSG);
#   2. identity of the final twin on 7100137 against the parent's run (cert_fs_<parent cert>_137, same runner run_cert_v2.py, PB_ALL=1)
#      and of the SAFE s49/s51k1s_pbs + the same fragments (copy O/scout_native/<tag>_safe/) on 7100136 against cert_fs_s51k1spbs_136;
#   3. the two count sessions (35, 100; normal and slow 1.3) under guard v3 (run_boundary_counts_v6.sh), one after the other;
#   4. fastgate_post_v1.py (base = the s51k1 stack's safe census, the same physics); once release's mode file runs/fastgate_mode_<tag>tw.json
#      exists, one [scout-native] summary line and the trigger line in the engineer's shape (as stack_build_v2 posts it).
# One native process at a time, guard v3, BLAS threads 1. Any failure: one '[scout-native] ... <TAG> STOPPED' line.
# SAFE_FRAGS (env, optional): the fragment list for the safe copy when a fragment is bound to its parent (idc1 has a twin and a safe version).
# usage: build_byteid_stack_v1.sh <tag, e.g. s51k3> <parent twin dir under s49, e.g. s51k2tw_x> <parent cert key, e.g. s51k2twx> <ids text> <fragment.py> [...]
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
set -u
P2=/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16
O=$P2/opus_reexam_20261001_v1; E=$O/engineer; A=$E/a62_modules; S=$O/scout_native; PY=/Users/matt/arc-white-box-estimation/.venv/bin/python
LOG=$O/TEAM_LOG.md
TAG=$1; PARENT=$2; PCERT=$3; IDS=$4; shift 4; FRAGS="$*"
UP=$(echo $TAG | tr a-z A-Z)
say() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }
post() { echo "[scout-native] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*" >> $LOG; say "POSTED: $*"; }
stop() { post "$UP STOPPED: $*"; exit 1; }
run1() {  # <candidate.py> <out tag> <seed>
  [ -e $S/cert_fs_$2.json ] && return 0
  bash $O/gate.sh 5 scout-native || return 3
  until [ $(memory_pressure | tail -1 | grep -o '[0-9]*%' | tr -d '%') -ge 55 ]; do sleep 60; done
  say "run $2 ($1, $3)"
  cd $P2 && PB_ALL=1 nice -n 10 $PY -B $O/release/guard_footprint_v3.py \
    --record $S/cert_fs_$2_guard.json --seconds 7200 --rss-backstop 7500000000 \
    $P2/astra_phase2_reexamination_v1/openblas_runtime_env_v1/bin/python -B $S/run_cert_v2.py $1 $3 $S/cert_fs_$2.json > $S/cert_fs_$2.log 2>&1
}
same() {  # <parent cert> <new cert>
  $PY -c "
import json
a = json.load(open('$S/cert_fs_$1.json'))['rows'][0]; b = json.load(open('$S/cert_fs_$2.json'))['rows'][0]
same = sum(a['mse_layers'][k] == b['mse_layers'][k] for k in range(16))
print('%s %s, %d/16 layers, bill %+.4f%%, requests %+d' % ('same' if a['pred_sha'] == b['pred_sha'] else 'DIFFERENT', b['pred_sha'], same, 100 * (b['flops'] / a['flops'] - 1), b['ops'] - a['ops']))"; }
say "start $TAG = $PARENT + $IDS"
[ -e $A/s49/${TAG}tw_x ] && stop "s49/${TAG}tw_x exists already"
post "$UP BUILD STARTED (lead 01:54Z: Astra integration, byte-identical bill stack): s49/$PARENT + $IDS, appended by make_s49_v1.py; identity on 7100137 (twin) and 7100136 (safe s51k1s_pbs + the same fragments), then the two count sessions on guard v3, then the trigger line for release's byte-identical gate in the engineer's shape; no MSE seal"
$PY -B $A/make_s49_v1.py $A/s49/$PARENT/candidate.py ${TAG}tw_x $FRAGS > $S/make_${TAG}tw_x.out 2>&1 || stop "make_s49_v1.py (O/scout_native/make_${TAG}tw_x.out)"
TX2=$A/s49/${TAG}tw_x/candidate.py; sha8=$(shasum -a 256 $TX2 | cut -c1-8)
mkdir -p $S/${TAG}_safe && cat $A/s49/s51k1s_pbs/candidate.py ${SAFE_FRAGS:-$FRAGS} > $S/${TAG}_safe/candidate.py   # SAFE_FRAGS (env): the safe's own copies of parent-bound fragments (e.g. idc1)
say "twin $TX2 $sha8"
run1 $TX2 ${TAG}twx_137 7100137 || stop "identity run 7100137 did not start"
[ -e $S/cert_fs_${TAG}twx_137.json ] || stop "identity run 7100137 failed (O/scout_native/cert_fs_${TAG}twx_137.log)"
r1=$(same ${PCERT}_137 ${TAG}twx_137); say "identity twin 7100137: $r1"
case "$r1" in same*) ;; *) stop "twin predictions on 7100137 differ from $PARENT: $r1";; esac
run1 $S/${TAG}_safe/candidate.py ${TAG}s_136 7100136 || stop "safe identity run did not start"
[ -e $S/cert_fs_${TAG}s_136.json ] || stop "safe identity run failed (O/scout_native/cert_fs_${TAG}s_136.log)"
r2=$(same s51k1spbs_136 ${TAG}s_136); say "identity safe 7100136: $r2"
case "$r2" in same*) ;; *) stop "safe predictions on 7100136 differ from s51k1s_pbs: $r2";; esac
for k in "n 35,100" "s 35,100 --setup-slowdown 1.3"; do
  set -- $k
  bash $S/run_boundary_counts_v6.sh $TX2 ps_${TAG}twx_$1 $2 ${3:+"$3 $4"} > $S/counts_${TAG}twx_$1.out 2>&1 || stop "count session $1 (O/scout_native/counts_${TAG}twx_$1.out)"
  grep -q "exit=0" $S/counts_${TAG}twx_$1.out || stop "count session $1 did not exit 0 (O/scout_native/counts_${TAG}twx_$1.out)"
  say "counts $1: $(tail -3 $S/counts_${TAG}twx_$1.out | tr '\n' ' ')"
done
line=$($PY -B $A/fastgate_post_v1.py ${TAG}tw $TX2 $E/logs/quick_keys_s51k1s_a.json $E/a56_req/runs/ps_${TAG}twx_n $E/a56_req/runs/ps_${TAG}twx_s 2>&1 | tail -1)
case "$line" in "FASTGATE "*" READY: "*) ;; *) stop "fastgate checks: $line";; esac
cnt=$($PY -c "
import json
f = lambda d: {json.loads(l)['rank']: json.loads(l) for l in open(d + '/rows.jsonl') if '\"rank\"' in l}
out = []
for kind, pk in (('n', 'n2'), ('s', 's2')):
    a, b = f('$E/a56_req/runs/ps_s51k2twx_' + pk), f('$E/a56_req/runs/ps_${TAG}twx_' + kind)
    out.append(kind + ' ' + ', '.join('%d: bill %+.4f%%, requests %+d' % (r, 100 * (b[r]['flops'] / a[r]['flops'] - 1), b[r]['requests'] - a[r]['requests']) for r in sorted(b) if r in a))
print('; '.join(out))" 2>&1 | tail -1)
until [ -e $O/release/runs/fastgate_mode_${TAG}tw.json ]; do say "waiting for release's runs/fastgate_mode_${TAG}tw.json"; sleep 60; done
post "$UP TWIN CHECKED: s49/${TAG}tw_x ($sha8) = $PARENT + $IDS; identity twin 7100137: $r1; safe 7100136: $r2; counts vs s51k2tw_x: $cnt; the trigger line for release's byte-identical gate follows in the engineer's shape (as stack_build_v2 posts it)"
echo "[engineer] $(date -u +%Y-%m-%dT%H:%M:%SZ) $line" >> $LOG; say "POSTED: [engineer] $line"
say end
