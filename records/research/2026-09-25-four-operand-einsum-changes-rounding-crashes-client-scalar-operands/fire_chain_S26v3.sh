#!/usr/bin/env bash
# TAG fable-porter (PT-56c = fire_chain_S26v2.sh revised 04:0x per the engineer + engineer2 + lead; v2 never ran): S26 = S24 fire copy ->
# j3cut (engineer2's build_j3cut.py, fragment 4874216a) -> k-split (arg 2: v3 / v3w / v4; it carries _PB_WALL_S = 3.5 and _PB_BATCHED = (), the
# single source, only VERIFIED here) -> [group 5 route B = engineer2's build_g5_B_v1.py (arg 4) on that exact pre-treuse file; its fragment must be
# byte-identical to the one engineer2 gated (arg 6 sha16), so the gates carry over] -> key list re-derived on that final pre-treuse composition
# (census in-chain; chunk keys stay in the list, engineer 03:5x; arg 7 phase3|none) = S26pre -> + treuse LAST = S26. Gates: einsum
# (S16/S23/S24) + validate + literals -> treuse identity (e73_parity.py, setup(), 35/100/37/39: S26pre vs S26) -> k-split shape check (net 35,
# info) -> smoke with setup() (<= 3.6 s) -> VALID40 vs the S24 rows (bar adj <= BAR, CI excl 0; raw separate; free attribution vs the S25
# j3+treuse rows) -> paired residual vs S24 (setup walls under concurrent load <= 3.6 s) -> staging fable_fire_v1/fire_candS26/.
# (04:3x: the informational k-split check uses ks_shape_check_v2.py = chunk-key coverage by _PB_SINGLE, for k-split v4 which has no _KS_SHAPES)
# usage: fire_chain_S26v3.sh <BAR> <ksplit frag> <sha16> <build_g5_B_v1.py | none> <builder sha16 | none> <gated fragment sha16 | none>
#        <phase3 | none>   (log fire_chain_S26v3.out)
set -u
P2=/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16; H=$P2/fable_porter_v1; TL=$P2/fable_refit_v1/TEAM_LOG.md; W=$H/stack
BAR=${1:--0.6}; KSF=$2; KSS=$3; G5=$4; G5S=$5; G5F=$6; EXM=$7
PT=$P2/fable_physicist_v1/refit6b/s6b_loopR3_joint_s23/params_epoch0.pt; NPY=/Users/matt/arc-white-box-estimation/.venv/bin/python
S24F=$P2/fable_fire_v1/fire_candS24/candS24_S23_loopR3_lvf2_lrnys_mem.py; S23F=$P2/fable_fire_v1/fire_candS23/candS23_S21s1_loopR1_s1z_jn_mem.py; S16F=$P2/fable_fire_v1/fire_candS16/candS16_S15_bb6_mem.py
TR=$P2/fable_engineer_v1/treuse_fragment_v2.py; J3B=$P2/fable_engineer2_v1/j3cut/build_j3cut.py; J3F=$P2/fable_engineer2_v1/j3cut/S23_j3cut_mem_fragment.py
J24=$H/paired_valid40_stack_S24c_mem.json; JNR=$H/paired_valid40_s25r_noR320_mem.json
J3O=$W/s26w_j3.py; KO=$W/s26w_j3_ks.py; GA=$W/s26w_j3_ks_g5.py; PRE=$W/s26w_pre.py; C=$W/stack_S26w_mem.py; FD=$P2/fable_fire_v1/fire_candS26
CEN=$H/ipt_key_census_S26w.json; P3=$W/s26w_ks_phase3.json
CHAIN_DONE=0; trap 'rc=$?; [ $CHAIN_DONE = 1 ] || echo "$(date "+%Y-%m-%d %H:%M") porter: S26 CHAIN DIED rc=$rc (fable_porter_v1/fire_chain_S26v3.out) last: $(tail -2 $H/fire_chain_S26v3.out 2>/dev/null | tr "\n" " " | cut -c1-200)" >> $TL' EXIT
fail() { echo "$(date '+%Y-%m-%d %H:%M') porter: S26 NOT staged: $1" >> $TL; echo "$(date +%T) FAIL: $1"; CHAIN_DONE=1; exit 1; }
for f in $J3O $KO $GA $PRE $C $FD $CEN $P3; do [ -e $f ] && fail "$f already exists (never overwrite)"; done
[ "$(shasum -a 256 $S24F | cut -c1-16)" = 30810da3b302c5ec ] || fail "S24 fire copy sha changed"
[ "$(shasum -a 256 $TR | cut -c1-16)" = 75aa92335f4b1de9 ] || fail "treuse fragment sha changed"
[ "$(shasum -a 256 $J3F | cut -c1-16)" = 4874216ad2a28990 ] || fail "j3cut fragment sha changed"
[ "$(shasum -a 256 $KSF | cut -c1-16)" = "$KSS" ] || fail "k-split fragment sha != $KSS"
[ "$G5" = none ] || [ "$(shasum -a 256 $G5 | cut -c1-16)" = "$G5S" ] || fail "group 5 builder sha != $G5S"
[ "$(shasum -a 256 $PT | cut -c1-16)" = 0750b69ba263cd9d ] || fail "loopR3 params sha changed"
case "$EXM" in phase3|none) ;; *) fail "arg 7 must be phase3 or none";; esac
[ "$G5" = none ] && [ "$G5F" != none ] && fail "gated fragment sha given without a builder"
[ "$G5" != none ] && [ "$G5F" = none ] && fail "group 5 needs the gated fragment sha (arg 6)"
python3 -c "import json,sys; [(lambda d: len([k for k in d if k.startswith('cand:')]) == 40 or sys.exit(1))(json.load(open(f))['results']) for f in sys.argv[1:]]" $J24 $JNR || fail "S24 / S25-attribution VALID40 rows missing or not 40"
echo "$(date +%T) [1] j3cut on S24"
python3 $J3B $S24F $J3O > /dev/null || fail "build_j3cut"
[ "$(shasum -a 256 ${J3O%.py}_fragment.py | cut -c1-16)" = 4874216ad2a28990 ] && [ "$(shasum -a 256 $J3O | cut -c1-16)" = "$(cat $S24F $J3F | shasum -a 256 | cut -c1-16)" ] || fail "j3cut rebuilt on S24 differs"
echo "$(date +%T) [2] + k-split ($(basename $KSF) $KSS)"
cat $J3O $KSF > $KO; A=$KO; BDESC="S24 + j3cut + $(basename $KSF)"
if [ "$G5" != none ]; then
  echo "$(date +%T) [3] group 5 route B on the exact pre-treuse file ($(basename $G5))"
  python3 $G5 $KO $GA > $W/s26w_g5_build.txt 2>&1 || { cat $W/s26w_g5_build.txt; fail "group 5 builder"; }
  cat $W/s26w_g5_build.txt
  GFS=$(shasum -a 256 ${GA%.py}_fragment.py | cut -c1-16); [ "${GFS:0:${#G5F}}" = "$G5F" ] || fail "group 5 fragment $GFS != the gated fragment $G5F (engineer2 must re-gate)"
  python3 -c "import sys; b, f, o = (open(x).read() for x in sys.argv[1:4]); assert o in (b + f, b.rstrip(chr(10)) + chr(10) + f)" $KO ${GA%.py}_fragment.py $GA || fail "build_g5 output != base + fragment"
  A=$GA; BDESC="$BDESC + group 5 route B (fragment $GFS)"
fi
BDESC="$BDESC + key list + treuse (last)"
echo "$(date +%T) [4] key census of the S26 base (300..580 + 35/100/37/39)"
bash $H/machine_gate.sh 8 S26-key-census
(cd $H && OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 nice -n 10 $NPY ipt_key_census_porter.py $A 300,340,380,420,460,500,540,580,35,100,37,39 S26w) 2>&1 | grep -v "Warning\|^  \|configure(" > $W/s26w_census.log; tail -3 $W/s26w_census.log
[ -s $CEN ] || fail "key census"
EXA=""
if [ "$EXM" = phase3 ]; then
  (cd $H && OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 nice -n 10 $NPY ks_phase3_keys_v2.py $A $P3) 2>&1 | grep -v "Warning\|^  \|configure(" | tee $W/s26w_phase3.log
  [ -s $P3 ] && python3 -c "import json,sys; assert len(json.load(open(sys.argv[1]))['keys']) > 0" $P3 || fail "k-split phase-3 keys empty"
  EXA="--exclude=$P3"
fi
echo "$(date +%T) [5] key list on the final pre-treuse composition (porter rule$( [ -n "$EXA" ] && echo ', k-split phase-3 keys excluded')), then + treuse LAST"
python3 $H/build_pb_keylist_v2.py $CEN $A $PRE $EXA | tee $W/s26w_keylist.txt; grep -q "self-test PASS" $W/s26w_keylist.txt && [ -s $PRE ] || fail "key list builder"
DP=$(diff $A $PRE | grep '^[<>]' | grep -v -E '^[<>] (    \([0-9].*\),|_PB_SINGLE = \(.*|\))$' | wc -l | tr -d ' '); [ "$DP" = 0 ] || fail "key-list step changed lines outside the key table ($DP)"
cat $PRE $TR > $C
WS=$(grep -E '^_PB_WALL_S = ' $C | tail -1); BT=$(grep -E '^_PB_BATCHED = ' $C | tail -1); echo "effective (last) settings: $WS | $BT"
echo "$WS" | grep -qE '^_PB_WALL_S = 3\.5\b' && echo "$BT" | grep -qE '^_PB_BATCHED = \(\)' || fail "k-split settings not effective in S26 (_PB_WALL_S 3.5, _PB_BATCHED ())"
echo "$(date +%T) [7] gates on the composed file"
E16=$(python3 $H/einsum_site_diff.py $C $S16F | tail -1); E23=$(python3 $H/einsum_site_diff.py $C $S23F | tail -1); E24=$(python3 $H/einsum_site_diff.py $C $S24F | tail -1)
V=$(/Users/matt/arc-white-box-estimation/.venv/bin/whest validate --estimator $C 2>&1 | grep -o 'Validation passed.*\|Validation failed.*')
echo "$(basename $C): einsum vs S16: $E16 | vs S23: ${E23:0:14} | vs S24: ${E24:0:14} | $V"
for e in "$E16" "$E23" "$E24"; do echo "$e" | grep -q "VERDICT PASS" || fail "einsum gate"; done
echo "$V" | grep -q "Validation passed" || fail "validate"
(cd $H && ./verify_literals.py $C $PT) | tee $W/S26w_literals.txt; grep -q "ALL 28 BIT-EQUAL" $W/S26w_literals.txt || fail "literals"
echo "$(date +%T) [8] treuse identity: S26pre vs S26 (e73_parity.py, setup(), 35,100,37,39)"
bash $H/machine_gate.sh 8 S26-treuse-identity
(cd $P2 && OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 10 $NPY fable_engineer_v1/e73_parity.py $PRE $C 35,100,37,39) 2>&1 | grep -v "Warning\|^  \|configure(" > $W/S26w_treuse_identity.txt; cat $W/S26w_treuse_identity.txt
python3 - $W/S26w_treuse_identity.txt <<'PY' || fail "treuse identity (stack/S26w_treuse_identity.txt)"
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1]) if l.startswith('{')]
par = {r['rank']: r for r in rows if 'all16_identical' in r}; tr = {r['rank']: r['tr_stats'] for r in rows if 'tr_stats' in r}
assert sorted(par) == [35, 37, 39, 100] and sorted(tr) == [35, 37, 39, 100], (sorted(par), sorted(tr))
for n, r in par.items():
    assert r['all16_identical'] is True and r['max_abs_diff'] == 0.0 and r['flops_B'] < r['flops_A'], r
    assert tr[n]['reused'] > 0, (n, tr[n])
print('treuse identity PASS: ' + '; '.join(f"r{n} bill {100 * (par[n]['flops_B'] / par[n]['flops_A'] - 1):+.3f}% reuses {tr[n]['reused']}" for n in sorted(par)))
PY
echo "$(date +%T) [8b] k-split chunk-key coverage (net 35, setup(); informational)"
(cd $H && OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 nice -n 10 $NPY ks_shape_check_v2.py $C 35) 2>&1 | grep -v "Warning\|^  \|configure(" > $W/S26w_ks_shape_check.txt; tail -1 $W/S26w_ks_shape_check.txt
echo "$(date '+%Y-%m-%d %H:%M') porter: S26 ($(shasum -a 256 $C | cut -c1-8)) built = $BDESC + key list ($(grep -o 'old [0-9]* keys.*added [0-9]*' $W/s26w_keylist.txt | head -1)); k-split v3 settings effective; treuse identity PASS; einsum/validate/literals PASS; $(tail -1 $W/S26w_ks_shape_check.txt); smoke next." >> $TL
echo "$(date +%T) [9] smoke with setup() (census v3, 35,100)"
bash $H/machine_gate.sh 7 S26-smoke
while ! mkdir /tmp/fable_paired.lock 2>/dev/null; do sleep 30; done
LOAD="$(sysctl -n vm.loadavg)"; bash $H/run_census_porter_v3.sh $C 35,100 cz_S26w > $W/cz_S26w.out 2>&1
rmdir /tmp/fable_paired.lock
python3 - $H/cz_S26w_census.json $W/cz_S26w.out "$LOAD" > $W/smoke_S26w.txt 2>&1 <<'PY'
import json, sys
rows = json.load(open(sys.argv[1])); assert [r['rank'] for r in rows] == [35, 100], rows
setup = [l for l in open(sys.argv[2]) if 'setup_wall_s' in l]; assert len(setup) == 1, setup
s = json.loads(setup[0].split(': ', 1)[1]); log = s['setup_log']
assert log and not any(str(e[0]).startswith('restored') for e in log), ('setup error (restored)', s)
bound = [e for e in log if e[0] == 'bound']; assert bound, ('no bound entry', s)
assert s['setup_wall_s'] <= 3.6, ('setup wall > 3.6 s', s)
print(f"smoke setup(): wall {s['setup_wall_s']} s at load {sys.argv[3]}, requests {s['setup_requests']:,}, bound {bound[-1][1]} entries, log {[list(e) for e in log]}")
for r in rows:
    assert r['requests'] > 0 and r['flops'] > 0, r
    assert r.get('n_out') == 16 * 1024 and r.get('nonfinite') == 0, ('outputs', r.get('n_out'), r.get('nonfinite'))
    print(f"smoke r{r['rank']}: OK requests {r['requests']:,} server CPU {r['server_cpu_s']} s residual {r['residual_s']} s util {r['util']} outputs {r['n_out']} all finite")
PY
[ $? -eq 0 ] || { cat $W/smoke_S26w.txt; tail -20 $W/cz_S26w.out; fail "SMOKE FAILED (stack/smoke_S26w.txt)"; }
cat $W/smoke_S26w.txt
echo "$(date '+%Y-%m-%d %H:%M') porter: S26 smoke OK ($(grep -o 'wall [0-9.]* s at load [^,]*' $W/smoke_S26w.txt | head -1); $(grep -o 'bound [0-9]* entries' $W/smoke_S26w.txt | head -1)); VALID40 vs S24 rows STARTED (bar adj <= ${BAR}%, CI excl 0)." >> $TL
echo "$(date +%T) [10] VALID40 vs the S24 rows"
bash $H/machine_gate.sh 8 S26-VALID40
bash $H/run_valid40_locked.sh $C stack_S26w_mem
python3 $H/valid40_vs_ref_ci.py $H/paired_valid40_stack_S26w_mem.json $J24 > $H/gate_S26w_valid40_vs_S24.txt; cat $H/gate_S26w_valid40_vs_S24.txt
python3 $H/valid40_vs_ref_ci.py $H/paired_valid40_stack_S26w_mem.json $JNR > $H/gate_S26w_vs_S24j3treuse.txt; cat $H/gate_S26w_vs_S24j3treuse.txt
python3 $H/valid40_bar.py $H/paired_valid40_stack_S26w_mem.json $J24 $BAR 0 | tee $H/gate_S26w_bar.json
if ! grep -q '"verdict": "PASS"' $H/gate_S26w_bar.json; then
  echo "$(date '+%Y-%m-%d %H:%M') porter: S26 NOT staged (VALID40 bar adj <= ${BAR}% vs S24, CI excl 0): $(sed -n 4p $H/gate_S26w_valid40_vs_S24.txt | cut -c1-140) | raw $(sed -n 2p $H/gate_S26w_valid40_vs_S24.txt | cut -c1-90) | k-split part (vs S24+j3+treuse rows) $(sed -n 4p $H/gate_S26w_vs_S24j3treuse.txt | cut -c1-110)" >> $TL
  echo "$(date +%T) S26 CHAIN DONE (not staged)"; CHAIN_DONE=1; exit 0
fi
echo "$(date '+%Y-%m-%d %H:%M') porter: S26 PASSES the VALID40 bar vs S24: $(sed -n 4p $H/gate_S26w_valid40_vs_S24.txt | cut -c1-150) | raw $(sed -n 2p $H/gate_S26w_valid40_vs_S24.txt | cut -c1-90); residual gate next." >> $TL
echo "$(date +%T) [11] paired residual gate vs S24 (setup client)"
bash $H/machine_gate.sh 13 S26-residual-gate
while ! mkdir /tmp/fable_paired.lock 2>/dev/null; do sleep 30; done
echo "$(date +%T) residual gate S24 vs S26 start | load $(sysctl -n vm.loadavg)"
for rep in 1 2 3; do
  if [ $((rep % 2)) = 1 ]; then AA=$S24F; TA=rz26w_S24_r$rep; BB=$C; TB=rz26w_S26_r$rep; else AA=$C; TA=rz26w_S26_r$rep; BB=$S24F; TB=rz26w_S24_r$rep; fi
  bash $H/run_census_porter_v3.sh $AA 35,100,37 $TA > $W/$TA.out 2>&1 & a=$!
  sleep 1
  bash $H/run_census_porter_v3.sh $BB 35,100,37 $TB > $W/$TB.out 2>&1 & b=$!
  wait $a $b; echo "$(date +%T) residual rep $rep done"
done
rmdir /tmp/fable_paired.lock
python3 - $H $W <<'PY' | tee $W/residual_gate_S26w.txt
import json, sys, statistics as st
H, W = sys.argv[1], sys.argv[2]
def load(tag): return {r['rank']: r for r in json.load(open(f'{H}/{tag}_census.json'))}
def setup_wall(tag):
    l = [x for x in open(f'{W}/{tag}.out') if 'setup_wall_s' in x]; s = json.loads(l[0].split(': ', 1)[1]); return s['setup_wall_s'], [e for e in s['setup_log'] if e[0] == 'bound'][-1][1]
R = {c: [load(f'rz26w_{c}_r{k}') for k in (1, 2, 3)] for c in ('S24', 'S26')}
print('TAG fable-porter PT-56: paired residual gate S24 vs S26 (setup-calling client, concurrent pairs, 3 reps, nets 35 cold,100,37)')
cold = {}
for n in (35, 100, 37):
    a = [r[n]['residual_s'] for r in R['S24']]; b = [r[n]['residual_s'] for r in R['S26']]
    rq = (st.mean(r[n]['requests'] for r in R['S24']), st.mean(r[n]['requests'] for r in R['S26']))
    if n == 35: cold = dict(A=st.mean(a), B=st.mean(b))
    print(f"  r{n}{' (cold)' if n == 35 else ''}: S24 mean {st.mean(a):.3f} max {max(a):.3f} s | S26 mean {st.mean(b):.3f} max {max(b):.3f} s | delta {st.mean(b)-st.mean(a):+.3f} s | requests {rq[0]:,.0f} -> {rq[1]:,.0f}")
sw = [setup_wall(f'rz26w_S26_r{k}') for k in (1, 2, 3)]; sw24 = [setup_wall(f'rz26w_S24_r{k}') for k in (1, 2, 3)]
print(f"  setup under concurrent load: S26 walls {[w for w, _ in sw]} s, bound {[b for _, b in sw]} (gate <= 3.6 s: {'PASS' if max(w for w, _ in sw) <= 3.6 else 'FAIL'}) | S24 walls {[w for w, _ in sw24]} s, bound {[b for _, b in sw24]}")
d = cold['B'] - cold['A']
print(f"  S26 cold delta vs S24 {d:+.3f} s (rule <= +0.010 s) -> RESIDUAL GATE {'PASS' if d <= 0.010 else 'FLAG'}")
PY
echo "$(date '+%Y-%m-%d %H:%M') porter: S26 residual gate vs S24: $(tail -1 $W/residual_gate_S26w.txt | sed 's/^ *//') | $(grep -o 'setup under concurrent load.*' $W/residual_gate_S26w.txt | cut -c1-160)" >> $TL
grep -q 'RESIDUAL GATE' $W/residual_gate_S26w.txt || fail "residual verdict missing (stack/residual_gate_S26w.txt)"
grep -q 'gate <= 3.6 s: PASS' $W/residual_gate_S26w.txt || fail "setup wall under concurrent load > 3.6 s (stack/residual_gate_S26w.txt)"
echo "$(date +%T) [12] staging"
[ -e $FD ] && fail "fire dir exists; not touching it"
mkdir $FD; F=$FD/candS26_S24_j3_ks$( [ "$G5" = none ] || echo _g5B)_kl_treuse_mem.py; cp -p $C $F
{ echo "FIRE CHECKS S26 = $BDESC; key list re-derived on the S26 structure; k-split v3 settings: $WS | $BT; $(date '+%Y-%m-%d %H:%M')"
  echo "file $F"; echo "sha256 $(shasum -a 256 $F | cut -c1-64)"
  echo "inputs: S24 30810da3; j3cut 4874216a (rebuilt on S24 byte-identical); $(basename $KSF) $KSS; group 5 ${G5##*/} ${G5S} (gated fragment ${G5F}); treuse_fragment_v2.py 75aa9233; key census $(basename $CEN) $(shasum -a 256 $CEN | cut -c1-16); phase-3 exclusion $EXM"
  echo "key list: $(grep -v '^self-test' $W/s26w_keylist.txt | tr '\n' ' ' | cut -c1-500)"
  echo "literals (fire copy): $(cd $H && ./verify_literals.py $F $PT) | validate: $(/Users/matt/arc-white-box-estimation/.venv/bin/whest validate --estimator $F 2>&1 | grep -o 'Validation passed.*\|Validation failed.*')"
  echo "einsum gate (fire copy) vs fired S16: $(python3 $H/einsum_site_diff.py $F $S16F | tail -1) | vs S23: $(python3 $H/einsum_site_diff.py $F $S23F | tail -1 | cut -c1-14) | vs S24: $(python3 $H/einsum_site_diff.py $F $S24F | tail -1 | cut -c1-14)"
  echo "k-split shape check (net 35): $(tail -1 $W/S26w_ks_shape_check.txt)"
  echo "treuse identity (e73_parity.py, setup(); S26 minus treuse vs S26): $(grep 'PASS' $W/S26w_treuse_identity.txt | head -1) $(python3 -c "import json,sys; rows=[json.loads(l) for l in open(sys.argv[1]) if l.startswith('{')]; print('; '.join(f\"r{r['rank']} identical {r['all16_identical']} maxdiff {r['max_abs_diff']} flops {100*(r['flops_B']/r['flops_A']-1):+.3f}%\" for r in rows if 'all16_identical' in r))" $W/S26w_treuse_identity.txt)"
  echo "client/server smoke with setup(): $(grep -c 'OK requests' $W/smoke_S26w.txt) of 2 nets OK; $(grep 'smoke setup' $W/smoke_S26w.txt | cut -c1-400)"
  echo "VALID40 vs S24 rows: $(sed -n 2,4p $H/gate_S26w_valid40_vs_S24.txt | tr '\n' ' ')"
  echo "bar (adjusted <= ${BAR}% vs S24, CI excl 0): $(cat $H/gate_S26w_bar.json)"
  echo "attribution: S26 vs S24 + j3 + treuse (S25 attribution rows): $(sed -n 2,4p $H/gate_S26w_vs_S24j3treuse.txt | tr '\n' ' ')"
  echo "residual gate vs S24 (setup-calling client): $(sed -n 2,6p $W/residual_gate_S26w.txt | tr '\n' ' ')"
} > $FD/FIRE_CHECKS.txt 2>&1
cat $FD/FIRE_CHECKS.txt
echo "$(date '+%Y-%m-%d %H:%M') porter: S26 STAGED: fable_fire_v1/fire_candS26/$(basename $F) ($(shasum -a 256 $F | cut -c1-8); FIRE_CHECKS.txt; VALID40 $(sed -n 4p $H/gate_S26w_valid40_vs_S24.txt | cut -c1-90); residual $(tail -1 $W/residual_gate_S26w.txt | grep -o 'PASS\|FLAG'))." >> $TL
echo "$(date +%T) S26 CHAIN DONE (fire copy staged)"; CHAIN_DONE=1
