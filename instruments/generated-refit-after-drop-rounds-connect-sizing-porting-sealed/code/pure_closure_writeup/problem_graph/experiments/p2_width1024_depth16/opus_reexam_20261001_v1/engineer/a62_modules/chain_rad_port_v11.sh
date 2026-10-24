#!/bin/bash
# TAG opus_engineer2 a62 v11 (file written 2026-10-12T22:26:53Z; lead 21:5xZ MANDATORY GUARD: from the absorb round on the stack builder skips gk11, so
# native files carry gk ONLY through the port, and a silent loss costs about +0.3%) = chain_rad_port_v10.sh with gk_guard_v1.py at three points
# of every round, each failure = one 'S<N> PORT STOPPED: gk guard ...' line and NO 'FILES READY' line:
#   G1 after the verdict and the hand-over, before any build: the round's config.json patch.ncar carries pbase gsk / gk4 / gbk = the absorbed
#      strengths (ABSORB.json's merge for round N; gsk = gk4 = 1.0 by the lead's 21:5xZ spec) for every round >= the absorb round (the earliest
#      of ABSORB.json's from_round and the anchored trainer 'ABSORBED gk11 at S<k>' line, the stack builder's own skip rule);
#   G2 on the built safe and twin sources, G3 on the final safe and twin (before the identity runs, counts and READY): the gk edit present
#      (def _gq + six marker lines) and the last module-level _GSK / _GK4 / _GBK = those strengths;
# plus port_rad_v1p.sh / port_twin_v5.sh (= v1o / v4 with the K31 converters pluggable through LAWCONV / K3LCONV, unset here: identical
# commands). Rounds before the absorb round: the guard answers n/a and the round ports exactly as under v10 (S52).
# v10 header: (file written 2026-10-12T22:11:12Z; lead 21:4xZ task 3; trainer 21:48:23Z: gk11 ABSORBED at S53, every RAD request from S53
# on carries pbase gsk = gk4 = 1 through the round's config patch 'ncar' and the exporter carries no request field) = chain_rad_port_v9.sh with:
#   (1) port_rad_v1o.sh <hand-over> <pt> trainer/cert/RAD/S<N>/config.json: the round's absorbed gk applied natively (gk_from_config_v1.py:
#       pbase gsk / gk4 / gbk; any other carrier change stops the round), make_gsk_v1.py LAST on the S49 file; port_twin_v4.sh gives the
#       twin the safe's gk the same way; a round whose config has no gk strengths ports exactly as in v9 (S52);
#   (2) the parity base: when the round carries gk and the start round's native file does not (S53 from S52), the base is that file + the
#       round's gk (make_gsk_v1.py -> s49/s<N>base_gk), since the torch check runs both its points with the round's patch; the continued-never-
#       ported start round's safe source is built with its own config (port_rad_v1o);
#   (3) PORT DETAILS names v10 / v1o / v4 and the gk carried; OMP / OPENBLAS / MKL / VECLIB / NUMEXPR threads = 1 for every side job (the
#       engineer's 18:31Z TODO; native sessions were already pinned by their guards);
#   READY / DETAILS / STOPPED / SKIPPED line formats, the 'waiting for its verdict file' text (stack_build_v2 reads it), the 2 native slots,
#   guards, retries and the clean stop (touch a62_modules/STOP_RAD_PORT) are v9's.
# v9 header: (file written 2026-10-12T17:47Z; lead 17:4xZ URGENT, release: the box is power-capped to about 8 cores, about 6 min per
# network) = chain_rad_port_v8.sh with every native guard sized for a slow box, not only for hangs: censuses through run_quick_v3.sh
# (900 s per network + 120 s, was 240), counts through run_boundary_counts_v5.sh (7200 s, was 3600), gate runs 7200 s (was 3600).
# v8 header: (file written 2026-10-12T17:33Z; lead 17:0xZ cadence: stack builds share this lane's 2 native slots) = chain_rad_port_v7.sh
# where every native launch (gate runs, census parts, count sessions) first waits until fewer than 2 native processes of this lane run
# (slot(), the same check as stack_build_v1.sh and chain_x2a_v1.sh); everything else as v7.
# v7 header: (file written 2026-10-12T17:21Z) = chain_rad_port_v6.sh with port_rad_v1n.sh (F4c exact-term strengths by birth layer,
# x2a1 per layer once its route parity PASSED) and port_twin_v3.sh (the twin gets that edit too); everything else as v6.
# v6 header: (file written 2026-10-12T17:07Z; trainer 16:58Z LEAD_PROMOTE, CONTINUE rule 16:49Z) = chain_rad_port_v4.sh for rounds ${FIRST:-51} .. ${LAST:-66}
# in ONE loop (replaces v4 = 51 .. 63 and v5 = 64 .. 66, both stopped while still waiting for S51's verdict), plus:
#   (1) every skipped round stays on a list that is re-checked every 10 min (while waiting, between rounds, and up to MAXWAIT h after the last
#       round): a verdict file that turns 'accepted' later (lead promotion = RESERVE_VERDICT_v1.json accepted true) is ported then, in full;
#   (2) a NULL freeze (FROZEN_POINT 'null': true, no check is run) or a round with no verdict file whose next round has started is skipped
#       at once, instead of holding the loop for MAXWAIT h;
#   (3) the parity base is the native file of the round's START point; a start round that was continued but never ported (CONTINUE rule
#       plus1se) gets its safe source built first (port_rad_v1m of its own hand-over, text edits only). No fallback to the last fired
#       file when the start round is known (that compares a different delta): parity 'not computed' instead.
# v4 header: (file written 2026-10-12T16:54Z) = chain_rad_port_v2.sh (rounds 51 .. 63) with port_rad_v1m.sh (k22 by layer; per-layer
# bscale / X1 / s_carry read from LAYER 0 by dly_from_pt_v3) and port_twin_v2.sh (the twin gets the safe file's parent edits too).
# v2 header: (file written 2026-10-12T08:20Z) = chain_rad_port_v1.sh where every native session (gate runs, censuses through
# run_quick_v2.sh, counts through run_boundary_counts_v4.sh) runs under release/guard_footprint_v2.py (release 08:3xZ). v1 header: (file written 2026-10-12T08:06Z; lead 08:0xZ): ONE versioned generic port loop for the trainer's RAD rounds, N = ${FIRST:-51} .. ${LAST:-63}
# in order (replaces the S51-only chain_s51port_v2.sh, which it covers), each round = chain_s51port_v2's steps:
#   wait: trainer/cert/RAD/S<N>/CHECK_VERDICT_v1.json or RESERVE_VERDICT_v1.json with "accepted": true -> port; a round is OVER (skipped, one line)
#         when its CHECK says false and the next round's directory has a config / RULES file newer than that CHECK (the trainer runs rounds back
#         to back), or after MAXWAIT hours from the loop reaching it
#   port: port_rad_v1m.sh jmS<N>_v1 cert/RAD/S<N>/point_v1/params_exportable_v1.pt (stops the round on any family without a native route)
#   parity on the check block's FIRST seed with both a_RS<N>C_ab / _pt records: native S<N> vs the native file of the round's START point
#         (FROZEN_POINT_v1.json start_params -> round M -> s49/jmS<M>_v1_pbs, else the last fired safe file named in a lead's FIRED line)
#   censuses of safe and twin (port_twin_v2.sh) on list a + 660-669 -> own prebind lists (costs: release's c_s50s3 / c_s50tw traces)
#   -> twin + XSG -> byte identity (safe on the parity seed, twin on 7100136) -> counts normal / 1.3x slow on 35 / 100
#   -> '[engineer] <stamp> S<N> FILES READY FOR RELEASE: safe <path> <sha8>, twin <path> <sha8>' (release keys its gates on it), then one
#      'S<N> PORT DETAILS' line (parity, identity, counts). Any failure: one 'S<N> PORT STOPPED' line and the loop goes on to N + 1.
# At most 2 native processes, gate.sh before every launch, one retry after 5 min of a failed run. Clean stop: touch a62_modules/STOP_RAD_PORT.
set -u
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
P2=/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16
O=$P2/opus_reexam_20261001_v1; E=$O/engineer; A=$E/a62_modules; PY=/Users/matt/arc-white-box-estimation/.venv/bin/python
FIRST=${FIRST:-51}; LAST=${LAST:-66}; MAXWAIT=${MAXWAIT:-30}; LOG=$O/TEAM_LOG.md; STOP=$A/STOP_RAD_PORT; SKIPPED=""
say() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }
NATIVE="^($P2/astra_phase2_reexamination_v1/openblas_runtime_env_v1/bin/python -B .*/engineer/|$P2/data_gate/.client-venv/bin/python -B -u .*boundary_client[.]py .*/engineer/)"   # gate / census processes and count-session clients
slot() { until [ $(ps -axo command= | grep -c -E "$NATIVE") -lt 2 ]; do sleep 60; done; }
post() { case "$*" in *"PORT PARITY"*) say "REFUSED to post a line with the parity prefix";; *) echo "[engineer] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*" >> $LOG; say "POSTED: $*";; esac; }
run1() {   # <candidate.py> <out tag> <seeds>
  slot; bash $O/gate.sh 7 engineer || return 3
  cd $P2 && OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
    /Users/matt/arc-white-box-estimation/.venv/bin/python -B $O/release/guard_footprint_v2.py \
    --record $A/gate_rdz_$2_guard.json --seconds 7200 --rss-backstop 7500000000 \
    $P2/astra_phase2_reexamination_v1/openblas_runtime_env_v1/bin/python -B $A/gate_rdz_v1.py $1 none none $3 $A/gate_rdz_$2.json \
    > $A/gate_rdz_$2.log 2>&1
}
run() {    # one retry under a new tag; the retry's results stand in for the first tag's
  [ -e $A/gate_rdz_$2.json ] && { say "$2 already done"; return 0; }
  run1 "$1" "$2" "$3" && { say "$2 ok"; return 0; }
  say "$2 FAILED, retry in 5 min as $2r"; sleep 300
  run1 "$1" "$2r" "$3" && { say "$2r ok"; for x in .json _pred.npz; do [ -e $A/gate_rdz_$2$x ] || cp $A/gate_rdz_$2r$x $A/gate_rdz_$2$x; done; return 0; }
  say "$2r FAILED"; return 1
}
census() { # <candidate.py> <name>: list a then 660-669, one process, one retry each
  local cand=$1 nm=$2
  for part in "a 651,378,424,328,287,641,548,557,316,390" "660 660,661,662,663,664,665,666,667,668,669"; do
    set -- $part
    [ -e $E/logs/quick_keys_${nm}_$1.json ] && continue
    slot; bash $E/run_quick_v3.sh "$cand" "$2" keys_${nm}_$1 --keys > $A/census_${nm}_$1.out 2>&1 || { sleep 300; slot; bash $E/run_quick_v3.sh "$cand" "$2" keys_${nm}_$1r --keys > $A/census_${nm}_$1r.out 2>&1 || return 1; cp $E/logs/quick_keys_${nm}_$1r.json $E/logs/quick_keys_${nm}_$1.json; }
  done
}
counts() { # <candidate.py> <name>
  slot; bash $A/run_boundary_counts_v5.sh "$1" ps_$2_n 35,100 > $A/counts_$2_n.out 2>&1 &
  sleep 5
  slot; bash $A/run_boundary_counts_v5.sh "$1" ps_$2_s 35,100 "--setup-slowdown 1.3" > $A/counts_$2_s.out 2>&1 &
  wait
}
verdict() { # <cert dir> -> ACCEPTED <file> | REJECTED | wait
  $PY -B -c "
import json, os, sys
C = sys.argv[1]; nxt = sys.argv[2]
acc = None
for f in ('RESERVE_VERDICT_v1.json', 'CHECK_VERDICT_v1.json'):
    try:
        d = json.load(open(os.path.join(C, f)))
    except Exception:
        continue
    if d.get('accepted') is True:
        print('ACCEPTED ' + f); sys.exit()
chk = os.path.join(C, 'CHECK_VERDICT_v1.json'); res = os.path.join(C, 'RESERVE_VERDICT_v1.json')
try:
    if json.load(open(os.path.join(C, 'FROZEN_POINT_v1.json'))).get('null') is True:
        print('REJECTED (NULL freeze)'); sys.exit()
except Exception:
    pass
if os.path.exists(res):
    print('REJECTED (reserve not accepted)'); sys.exit()
later = [os.path.getmtime(os.path.join(nxt, f)) for f in ('config.json', 'RULES_v1.json') if os.path.exists(os.path.join(nxt, f))]
if os.path.exists(chk):
    if later and max(later) > os.path.getmtime(chk):
        print('REJECTED (check not accepted, next round started)'); sys.exit()
elif later:
    print('REJECTED (no verdict file, next round started)'); sys.exit()
print('wait')" "$1" "$2"
}
base_for() { # <frozen point json> -> the native file of the round's start point, or 'BUILD <hand-over> <.pt>' (continued, never ported)
  $PY -B -c "
import json, re, sys, glob, os, hashlib
A, LOG, T = sys.argv[2], sys.argv[3], sys.argv[4]
d = json.load(open(sys.argv[1]))
m = re.search(r'/cert/RAD/(S\d+|MS)/', d.get('start_params', ''))
cands = []
if m and m.group(1) != 'MS':
    cands += [A + '/s49/jm%s_v1_pbs/candidate.py' % m.group(1), A + '/s49/jm%s_v1/candidate.py' % m.group(1)]
if m and m.group(1) == 'MS':
    cands += [A + '/s48/s48b/candidate.py']
for c in cands:
    if os.path.exists(c):
        print(c); sys.exit()
if m and m.group(1) != 'MS':
    ptx = os.path.join(os.path.dirname(d['start_params']), 'params_exportable_v1.pt')
    if os.path.exists(ptx) and os.path.exists('%s/jm%s_v1/physics_dials.json' % (T, m.group(1))):
        print('BUILD jm%s_v1 %s' % (m.group(1), ptx))
    else:
        print('')
    sys.exit()
fired = re.findall(r'FIRED THE S\d+ SAFE.*?estimator ([0-9a-f]{8})', open(LOG).read())
if fired:
    for c in glob.glob(A + '/s49/*/candidate.py'):
        if hashlib.sha256(open(c, 'rb').read()).hexdigest().startswith(fired[-1]):
            print(c); sys.exit()
print('')" "$1" "$A" "$LOG" "$O/trainer/candidates"
}
recheck() { # port every skipped round whose verdict file has turned 'accepted' since its skip (lead promotion / reserve)
  local M w
  for M in $SKIPPED; do
    w=$(verdict $O/trainer/cert/RAD/S$M $O/trainer/cert/RAD/S$(( M + 1 )))
    case "$w" in ACCEPTED*) ;; *) continue;; esac
    SKIPPED=$(echo " $SKIPPED " | sed "s/ $M / /; s/^ *//; s/ *$//")
    post "S$M VERDICT TURNED ACCEPTED AFTER ITS PORT SKIP ($w: lead promotion or reserve): porting S$M now"
    port_round $M || say "S$M stopped (posted)"
  done
}
port_round() {
  local N=$1 C=$O/trainer/cert/RAD/S$1 NX=$O/trainer/cert/RAD/S$(( $1 + 1 )) R=$O/trainer/run_RAD_S$1_gen/results HAND=jmS$1_v1
  local t0=$(date +%s) v polls=0
  say "S$N: waiting for its verdict file"
  while :; do
    [ -e $STOP ] && { say "STOP file: exit"; exit 0; }
    v=$(verdict $C $NX)
    case "$v" in ACCEPTED*) say "S$N $v"; break;; REJECTED*) say "S$N $v"; SKIPPED="$SKIPPED $N"; post "S$N PORT SKIPPED: $v (no files; re-checked every 10 min for a later 'accepted')"; return 0;; esac
    [ $(( $(date +%s) - t0 )) -gt $(( MAXWAIT * 3600 )) ] && { SKIPPED="$SKIPPED $N"; post "S$N PORT SKIPPED: no verdict within ${MAXWAIT} h (re-checked every 10 min for a later 'accepted')"; return 0; }
    polls=$(( polls + 1 )); [ $(( polls % 5 )) -eq 0 ] && [ -n "$SKIPPED" ] && recheck
    sleep 120
  done
  local pt ptx S B seed par TW SP TX ident cnt
  pt=$($PY -B -c "import json; print(json.load(open('$C/FROZEN_POINT_v1.json'))['params'])")
  ptx=$(dirname $pt)/params_exportable_v1.pt
  until [ -f $ptx ] && [ -f $O/trainer/candidates/$HAND/physics_dials.json ]; do
    [ -e $STOP ] && exit 0
    [ $(( $(date +%s) - t0 )) -gt $(( MAXWAIT * 3600 )) ] && { post "S$N PORT STOPPED: accepted, but no hand-over $HAND / $ptx in time"; return 1; }
    sleep 60
  done
  say "S$N point $pt, exportable $ptx ($(shasum -a 256 $ptx | cut -c1-8)), hand-over $HAND"
  local g1 g2 g3
  g1=$($PY -B $A/gk_guard_v1.py $N $C/config.json 2>&1 | tail -1)
  case "$g1" in "GK GUARD OK"*|"GK GUARD n/a"*) say "S$N config: $g1";; *) post "S$N PORT STOPPED: gk guard on the round's config (gk_guard_v1.py): $(echo "$g1" | cut -c1-300)"; return 1;; esac
  if [ ! -e $A/s49/$HAND/candidate.py ]; then
    cd $A && bash $A/port_rad_v1p.sh $HAND $ptx $C/config.json > $A/port_${HAND}.out 2>&1 || { post "S$N PORT STOPPED at port_rad_v1p (a62_modules/port_${HAND}.out): $(tail -1 $A/port_${HAND}.out | cut -c1-200)"; return 1; }
  fi
  S=$A/s49/$HAND/candidate.py
  local gkv="" gkd="none (no gk strengths in the round's config)"
  if [ -d $A/s49/${HAND}_pregk ]; then
    gkv=$($PY -B -c "import json; print(json.load(open('$A/s49/$HAND/build.json'))['values'])")
    gkd="$(basename $gkv) ($(grep -E '^_GSK =|^_GK4 =|^_GBK =' $gkv | tr '\n' ' '))"
  fi
  B=$(base_for $C/FROZEN_POINT_v1.json)
  case "$B" in BUILD*)
    local bh bp; read -r _ bh bp <<< "$B"
    say "S$N start point = continued round $bh (never ported): building its safe source for the parity base"
    [ -e $A/s49/$bh/candidate.py ] || (cd $A && bash $A/port_rad_v1p.sh $bh $bp $O/trainer/cert/RAD/$(echo $bh | sed -E 's/^jm(S[0-9]+)_v1$/\1/')/config.json > $A/port_${bh}.out 2>&1) || say "base build failed (a62_modules/port_${bh}.out)"
    B=$A/s49/$bh/candidate.py; [ -e $B ] || B="";;
  esac
  if [ -n "$B" ] && [ -n "$gkv" ] && ! grep -q '^def _gq(' $B; then   # v10: the torch check runs start and frozen point with this round's gk
    [ -e $A/s49/s${N}base_gk/candidate.py ] || (cd $A && $PY -B $A/make_gsk_v1.py $B s${N}base_gk $gkv > $A/port_s${N}base_gk.out 2>&1) || say "parity base + gk failed (a62_modules/port_s${N}base_gk.out)"
    B=$A/s49/s${N}base_gk/candidate.py; [ -e $B ] || B=""
    say "S$N parity base = the start round's file + this round's gk: s49/s${N}base_gk"
  fi
  seed=$($PY -B -c "
import glob, os, re, sys
R, N = sys.argv[1], sys.argv[2]
s = sorted(int(m.group(1)) for f in glob.glob(R + '/a_RS%sC_pt_r*.json' % N) for m in [re.search(r'_r(\d+)\.json$', f)] if m and os.path.exists(R + '/a_RS%sC_ab_r%s.json' % (N, m.group(1))))
print(s[0] if s else '')" $R $N)
  if [ -n "$seed" ] && [ -n "$B" ]; then
    run $S s${N}_$seed $seed &
    sleep 5
    run $B s${N}base_$seed $seed &
    wait
    $PY -B $A/parity_v2.py $R/a_RS${N}C_ab_r$seed.json $R/a_RS${N}C_pt_r$seed.json $A/gate_rdz_s${N}base_$seed.json $A/gate_rdz_s${N}_$seed.json $seed > $A/parity_s${N}_$seed.out 2>&1
    par="network $seed, native base $(basename $(dirname $B)): $(sed -n '1p' $A/parity_s${N}_$seed.out | cut -c1-140); $(grep '^L15' $A/parity_s${N}_$seed.out | cut -c1-120)"
  else
    par="not computed (check seed '$seed', native base '$B')"
    [ -n "$seed" ] || seed=7100136
    run $S s${N}_$seed $seed
  fi
  say "S$N parity: $par"
  if [ ! -e $A/s49/${HAND}tw/candidate.py ]; then
    cd $A && bash $A/port_twin_v5.sh $HAND $ptx $HAND ${HAND}tw > $A/port_${HAND}tw.out 2>&1 || { post "S$N PORT STOPPED: twin build failed (a62_modules/port_${HAND}tw.out); the safe source s49/$HAND is built"; return 1; }
  fi
  TW=$A/s49/${HAND}tw/candidate.py
  g2=$($PY -B $A/gk_guard_v1.py $N $C/config.json $S $TW 2>&1 | tail -1)
  case "$g2" in "GK GUARD OK"*|"GK GUARD n/a"*) say "S$N sources: $g2";; *) post "S$N PORT STOPPED: gk guard on the safe and twin sources (gk_guard_v1.py): $(echo "$g2" | cut -c1-300)"; return 1;; esac
  census $S s${N}s &
  sleep 5
  census $TW s${N}tw &
  wait
  for f in s${N}s_a s${N}s_660 s${N}tw_a s${N}tw_660; do [ -f $E/logs/quick_keys_$f.json ] || { post "S$N PORT STOPPED: census $f failed (a62_modules/census_$f.out)"; return 1; }; done
  cd $A
  [ -e $A/fragment_prebind_s${N}s_v1.py ] || $PY -B $A/prebind_targeted_v2.py $S $O/release/runs/c_s50s3_safe/C0/manifest.json $A/fragment_prebind_s${N}s_v1.py \
    $E/logs/quick_keys_s${N}s_a.json $E/logs/quick_keys_s${N}s_660.json > $A/prebind_s${N}s_v1.out 2>&1 || { post "S$N PORT STOPPED: safe prebind list failed"; return 1; }
  [ -e $A/s49/${HAND}_pbs ] || $PY -B $A/make_s49_v1.py $S ${HAND}_pbs $A/fragment_prebind_s${N}s_v1.py > /dev/null || { post "S$N PORT STOPPED: safe build failed"; return 1; }
  [ -e $A/fragment_prebind_s${N}tw_v1.py ] || $PY -B $A/prebind_targeted_v2.py $TW $O/release/runs/c_s50tw_safe/C0/manifest.json $A/fragment_prebind_s${N}tw_v1.py \
    $E/logs/quick_keys_s${N}tw_a.json $E/logs/quick_keys_s${N}tw_660.json > $A/prebind_s${N}tw_v1.out 2>&1 || { post "S$N PORT STOPPED: twin prebind list failed"; return 1; }
  [ -e $A/s49/${HAND}tw_pbt ] || $PY -B $A/make_s49_v1.py $TW ${HAND}tw_pbt $A/fragment_prebind_s${N}tw_v1.py > /dev/null || { post "S$N PORT STOPPED: twin build failed"; return 1; }
  [ -e $A/s49/${HAND}tw_x ] || $PY -B $A/make_xsg_any_v1.py $A/s49/${HAND}tw_pbt/candidate.py ${HAND}tw_x > $A/xsg_${HAND}tw.out 2>&1 || { post "S$N PORT STOPPED: XSG failed on the twin"; return 1; }
  SP=$A/s49/${HAND}_pbs/candidate.py; TX=$A/s49/${HAND}tw_x/candidate.py
  g3=$($PY -B $A/gk_guard_v1.py $N $C/config.json $SP $TX 2>&1 | tail -1)
  case "$g3" in "GK GUARD OK"*|"GK GUARD n/a"*) say "S$N finals: $g3";; *) post "S$N PORT STOPPED: gk guard on the final safe and twin (gk_guard_v1.py): $(echo "$g3" | cut -c1-300)"; return 1;; esac
  run $SP s${N}pbs_$seed $seed &
  sleep 5
  ( run $TW s${N}tw_136 7100136 && run $TX s${N}twx_136 7100136 ) &
  wait
  ident=$($PY -B -c "
import json, sys
A, N, seed = sys.argv[1], sys.argv[2], sys.argv[3]
def r(t): return json.load(open('%s/gate_rdz_%s.json' % (A, t)))['rows'][0]
out = []
for a, b, what in (('s%s_%s' % (N, seed), 's%spbs_%s' % (N, seed), 'safe + own list vs safe on %s' % seed), ('s%stw_136' % N, 's%stwx_136' % N, 'twin + own list + XSG vs twin source on 7100136')):
    try:
        x, y = r(a), r(b)
        out.append('%s: %s, bill %+.4f%%, ops %+d' % (what, 'BYTE-IDENTICAL' if x['pred_sha'] == y['pred_sha'] else 'DIFFERENT', 100 * (y['flops'] / x['flops'] - 1), y['ops'] - x['ops']))
    except Exception as e:
        out.append('%s: no result (%s)' % (what, e))
print('; '.join(out))" $A $N $seed)
  say "S$N identity: $ident"
  case "$ident" in *DIFFERENT*|*"no result"*) post "S$N PORT STOPPED: identity check not clean ($ident)"; return 1;; esac
  counts $SP s${N}pbs
  counts $TX s${N}twx
  cnt=$(for n in s${N}pbs_n s${N}pbs_s s${N}twx_n s${N}twx_s; do printf '%s %s; ' $n "$(tail -2 $A/counts_$n.out 2>/dev/null | awk '{printf "%s %s ", $1, $3}')"; done)
  post "S$N FILES READY FOR RELEASE: safe $SP $(shasum -a 256 $SP | cut -c1-8), twin $TX $(shasum -a 256 $TX | cut -c1-8)"
  post "S$N PORT DETAILS (chain_rad_port_v11): safe = port_rad_v1p of trainer/candidates/$HAND + $(basename $ptx) ($(shasum -a 256 $ptx | cut -c1-8)) + its own prebind list ($(grep -c 'BEGIN OPUS-RELEASE FRAGMENT' $SP) release fragments: FRAGS=none); twin = port_twin_v5 + own list + XSG ($(grep -c 'BEGIN OPUS-RELEASE FRAGMENT' $TX) release fragments). gk (round config): $gkd; $g3. Parity: $par. Identity: $ident. Counts (network requests, 35 then 100): $cnt"
}
say "loop armed: rounds $FIRST .. $LAST"
for N in $(seq $FIRST $LAST); do
  [ -e $STOP ] && { say "STOP file: exit"; exit 0; }
  port_round $N || say "S$N stopped (posted)"
  [ -n "$SKIPPED" ] && recheck
done
say "rounds done; re-checking skipped rounds ($SKIPPED) every 10 min for up to $MAXWAIT h"
t1=$(date +%s)
while [ -n "$SKIPPED" ] && [ $(( $(date +%s) - t1 )) -lt $(( MAXWAIT * 3600 )) ]; do
  [ -e $STOP ] && { say "STOP file: exit"; exit 0; }
  recheck; sleep 600
done
say "loop done"
