#!/usr/bin/env bash
# TAG fable-porter (PT-54, lead 02:4x GO "S25 = S24 + treuse + j3cut + R320 with a prebind key list MATCHED to rank 320; VALID40 vs the S24 rows, bar
# adj <= -0.5%, CI excl 0"). Built from the S24 fire copy in four checked steps: (1) engineer2's build_j3cut.py (fragment = 4874216a, file = S24 +
# fragment); (2) build_tier_rank.py 320 (diff = the 3 builder edits); (3) build_pb_keylist.py (engineer2's d6x key rule, self-tested on S24's live list)
# on the R320 key census ipt_key_census_S25R320.json (diff = key-table lines only) = S25pre; (4) + treuse_fragment_v2.py LAST (engineer's rule 1).
# Gates: treuse identity (engineer's e73_parity.py with setup(), nets 35/100/37/39: all 16 layers identical, max diff 0, flops lower, reuses > 0)
# -> einsum (S16/S23/S24) + validate + literals -> smoke with setup() (setup wall, bound count, load) -> VALID40 vs the S24 rows (bar adj <= BAR,
# default -0.5, CI excl 0; on FAIL: attribution VALID40 of S24 + j3 + treuse without R320, then stop) -> paired residual vs S24 -> staging
# fable_fire_v1/fire_candS25/ (refused unless every verdict exists).   usage: fire_chain_S25r.sh [BAR]   (log fire_chain_S25r.out)
set -u
P2=/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16; H=$P2/fable_porter_v1; TL=$P2/fable_refit_v1/TEAM_LOG.md; W=$H/stack
BAR=${1:--0.5}; PT=$P2/fable_physicist_v1/refit6b/s6b_loopR3_joint_s23/params_epoch0.pt; NPY=/Users/matt/arc-white-box-estimation/.venv/bin/python
S24F=$P2/fable_fire_v1/fire_candS24/candS24_S23_loopR3_lvf2_lrnys_mem.py; S23F=$P2/fable_fire_v1/fire_candS23/candS23_S21s1_loopR1_s1z_jn_mem.py; S16F=$P2/fable_fire_v1/fire_candS16/candS16_S15_bb6_mem.py
TR=$P2/fable_engineer_v1/treuse_fragment_v2.py; J3B=$P2/fable_engineer2_v1/j3cut/build_j3cut.py; J3F=$P2/fable_engineer2_v1/j3cut/S23_j3cut_mem_fragment.py; CEN=$H/ipt_key_census_S25R320.json
J24=$H/paired_valid40_stack_S24c_mem.json; J3O=$W/s25r_j3.py; R3O=$W/s25r_j3_R320.py; PRE=$W/s25r_pre.py; C=$W/stack_S25r_mem.py; NR=$W/s25r_noR320_mem.py; FD=$P2/fable_fire_v1/fire_candS25
CHAIN_DONE=0; trap 'rc=$?; [ $CHAIN_DONE = 1 ] || echo "$(date "+%Y-%m-%d %H:%M") porter: S25 CHAIN DIED rc=$rc (fable_porter_v1/fire_chain_S25r.out) last: $(tail -2 $H/fire_chain_S25r.out 2>/dev/null | tr "\n" " " | cut -c1-200)" >> $TL' EXIT
fail() { echo "$(date '+%Y-%m-%d %H:%M') porter: S25 NOT staged: $1" >> $TL; echo "$(date +%T) FAIL: $1"; CHAIN_DONE=1; exit 1; }
for f in $J3O $R3O $PRE $C $NR $FD; do [ -e $f ] && fail "$f already exists (never overwrite)"; done
[ "$(shasum -a 256 $S24F | cut -c1-16)" = 30810da3b302c5ec ] || fail "S24 fire copy sha changed"
[ "$(shasum -a 256 $TR | cut -c1-16)" = 75aa92335f4b1de9 ] || fail "treuse fragment sha changed"
[ "$(shasum -a 256 $J3F | cut -c1-16)" = 4874216ad2a28990 ] || fail "j3cut fragment sha changed"
[ "$(shasum -a 256 $CEN | cut -c1-16)" = 645ac0f99fac5a8f ] || fail "R320 key census sha changed"
[ "$(shasum -a 256 $PT | cut -c1-16)" = 0750b69ba263cd9d ] || fail "loopR3 params sha changed"
python3 -c "import json,sys; d=json.load(open(sys.argv[1]))['results']; assert len([k for k in d if k.startswith('cand:')]) == 40" $J24 || fail "S24 VALID40 rows missing or not 40"
echo "$(date +%T) [1] j3cut on S24 (engineer2's builder)"
python3 $J3B $S24F $J3O || fail "build_j3cut"
[ "$(shasum -a 256 ${J3O%.py}_fragment.py | cut -c1-16)" = 4874216ad2a28990 ] && [ "$(shasum -a 256 $J3O | cut -c1-16)" = "$(cat $S24F $J3F | shasum -a 256 | cut -c1-16)" ] || fail "j3cut rebuilt on S24 differs from engineer2's fragment"
echo "$(date +%T) [2] old-tier frame rank 320 (build_tier_rank.py)"
python3 $H/build_tier_rank.py $J3O $R3O 320 || fail "build_tier_rank"
D2=$(diff $J3O $R3O | grep '^[<>]' | grep -v -E '^[<>] (_SLR_RANK = |_TIER_RANK = 320 |    if int\(w32.shape\[0\]\) != 1024 or int\(previous.shape\[0\]\) < |    rank = )' | wc -l | tr -d ' ')
[ "$D2" = 0 ] && [ "$(diff $J3O $R3O | grep -c '^[<>]')" = 7 ] || fail "R320 diff is not exactly the 3 builder edits ($D2 stray lines)"
echo "$(date +%T) [3] prebind key list matched to R320 (engineer2's d6x rule on the R320 census)"
python3 $H/build_pb_keylist.py $CEN $R3O $PRE | tee $W/s25r_keylist.txt; grep -q "self-test PASS" $W/s25r_keylist.txt && [ -s $PRE ] || fail "key list builder"
D3=$(diff $R3O $PRE | grep '^[<>]' | grep -v -E '^[<>] (    \([0-9].*\),|_PB_SINGLE = \(.*|_PB_BATCHED = \(.*|\))$' | wc -l | tr -d ' ')
[ "$D3" = 0 ] || fail "key-list step changed lines outside the key tables ($D3)"
echo "$(date +%T) [4] + treuse LAST"
cat $PRE $TR > $C; cat $J3O $TR > $NR
for f in $C $NR; do
  E16=$(python3 $H/einsum_site_diff.py $f $S16F | tail -1); E23=$(python3 $H/einsum_site_diff.py $f $S23F | tail -1); E24=$(python3 $H/einsum_site_diff.py $f $S24F | tail -1)
  V=$(/Users/matt/arc-white-box-estimation/.venv/bin/whest validate --estimator $f 2>&1 | grep -o 'Validation passed.*\|Validation failed.*')
  echo "$(basename $f): einsum vs S16: $E16 | vs S23: ${E23:0:14} | vs S24: ${E24:0:14} | $V"
  for e in "$E16" "$E23" "$E24"; do echo "$e" | grep -q "VERDICT PASS" || fail "$(basename $f) einsum gate"; done
  echo "$V" | grep -q "Validation passed" || fail "$(basename $f) validate"
done
(cd $H && ./verify_literals.py $C $PT) | tee $W/S25r_literals.txt; grep -q "ALL 28 BIT-EQUAL" $W/S25r_literals.txt || fail "composed literals"
echo "$(date +%T) [5] treuse identity: S25pre vs S25 (engineer's e73_parity.py, setup(), 35,100,37,39)"
bash $H/machine_gate.sh 8 S25r-treuse-identity
(cd $P2 && OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 10 $NPY fable_engineer_v1/e73_parity.py $PRE $C 35,100,37,39) 2>&1 | grep -v "Warning\|^  \|configure(" > $W/S25r_treuse_identity.txt; cat $W/S25r_treuse_identity.txt
python3 - $W/S25r_treuse_identity.txt <<'PY' || fail "treuse identity (stack/S25r_treuse_identity.txt)"
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1]) if l.startswith('{')]
par = {r['rank']: r for r in rows if 'all16_identical' in r}; tr = {r['rank']: r['tr_stats'] for r in rows if 'tr_stats' in r}
assert sorted(par) == [35, 37, 39, 100] and sorted(tr) == [35, 37, 39, 100], (sorted(par), sorted(tr))
for n, r in par.items():
    assert r['all16_identical'] is True and r['max_abs_diff'] == 0.0 and r['flops_B'] < r['flops_A'], r
    assert tr[n]['reused'] > 0, (n, tr[n])
print('treuse identity PASS: ' + '; '.join(f"r{n} bill {100 * (par[n]['flops_B'] / par[n]['flops_A'] - 1):+.3f}% reuses {tr[n]['reused']}" for n in sorted(par)))
PY
echo "$(date '+%Y-%m-%d %H:%M') porter: S25 ($(shasum -a 256 $C | cut -c1-8)) built = S24 + j3cut + R320 (3 edits) + matched key list + treuse last; treuse identity PASS (35/100/37/39); einsum/validate/literals PASS; smoke next." >> $TL
echo "$(date +%T) [6] smoke with setup() (census v3, 35,100)"
bash $H/machine_gate.sh 7 S25r-smoke
while ! mkdir /tmp/fable_paired.lock 2>/dev/null; do sleep 30; done
LOAD="$(sysctl -n vm.loadavg)"; bash $H/run_census_porter_v3.sh $C 35,100 cz_S25r > $W/cz_S25r.out 2>&1
rmdir /tmp/fable_paired.lock
python3 - $H/cz_S25r_census.json $W/cz_S25r.out "$LOAD" > $W/smoke_S25r.txt 2>&1 <<'PY'
import json, sys
rows = json.load(open(sys.argv[1])); assert [r['rank'] for r in rows] == [35, 100], rows
setup = [l for l in open(sys.argv[2]) if 'setup_wall_s' in l]; assert len(setup) == 1, setup
s = json.loads(setup[0].split(': ', 1)[1]); log = s['setup_log']
assert log and not any(str(e[0]).startswith('restored') for e in log), ('setup error (restored)', s)
bound = [e for e in log if e[0] == 'bound']; assert bound, ('no bound entry', s)
assert s['setup_wall_s'] <= 3.1, ('setup wall > 3.1 s', s)
guard = [e for e in log if e[0] == 'wall guard']; lvf = [e for e in log if e[0] == 'lvfuse']
print(f"smoke setup(): wall {s['setup_wall_s']} s at load {sys.argv[3]}, requests {s['setup_requests']:,}, bound {bound[-1][1]} entries, lvfuse {lvf[-1][1:] if lvf else None}" + (f" (WALL GUARD held at {guard[0][1]})" if guard else ''))
for r in rows:
    assert r['requests'] > 0 and r['flops'] > 0, r
    assert r.get('n_out') == 16 * 1024 and r.get('nonfinite') == 0, ('outputs', r.get('n_out'), r.get('nonfinite'))
    print(f"smoke r{r['rank']}: OK requests {r['requests']:,} server CPU {r['server_cpu_s']} s residual {r['residual_s']} s util {r['util']} outputs {r['n_out']} all finite")
PY
[ $? -eq 0 ] || { cat $W/smoke_S25r.txt; tail -20 $W/cz_S25r.out; fail "SMOKE FAILED (stack/smoke_S25r.txt)"; }
cat $W/smoke_S25r.txt
echo "$(date '+%Y-%m-%d %H:%M') porter: S25 smoke OK ($(grep -o 'wall [0-9.]* s at load [^,]*' $W/smoke_S25r.txt | head -1); $(grep -o 'bound [0-9]* entries' $W/smoke_S25r.txt | head -1)); VALID40 vs S24 rows STARTED (bar adj <= ${BAR}%, CI excl 0)." >> $TL
echo "$(date +%T) [7] VALID40 vs the S24 rows"
bash $H/machine_gate.sh 8 S25r-VALID40
bash $H/run_valid40_locked.sh $C stack_S25r_mem
python3 $H/valid40_vs_ref_ci.py $H/paired_valid40_stack_S25r_mem.json $J24 > $H/gate_S25r_valid40_vs_S24.txt; cat $H/gate_S25r_valid40_vs_S24.txt
python3 $H/valid40_bar.py $H/paired_valid40_stack_S25r_mem.json $J24 $BAR 0 | tee $H/gate_S25r_bar.json
if ! grep -q '"verdict": "PASS"' $H/gate_S25r_bar.json; then
  echo "$(date '+%Y-%m-%d %H:%M') porter: S25 FAILS the VALID40 bar (adj <= ${BAR}% vs S24, CI excl 0): $(sed -n 4p $H/gate_S25r_valid40_vs_S24.txt | cut -c1-150); running the attribution VALID40 (S24 + j3 + treuse, no R320)." >> $TL
  bash $H/machine_gate.sh 8 S25r-attribution
  bash $H/run_valid40_locked.sh $NR s25r_noR320_mem
  python3 $H/valid40_vs_ref_ci.py $H/paired_valid40_s25r_noR320_mem.json $J24 > $H/gate_S25r_noR320_vs_S24.txt; cat $H/gate_S25r_noR320_vs_S24.txt
  python3 $H/valid40_vs_ref_ci.py $H/paired_valid40_stack_S25r_mem.json $H/paired_valid40_s25r_noR320_mem.json > $H/gate_S25r_R320part.txt; cat $H/gate_S25r_R320part.txt
  echo "$(date '+%Y-%m-%d %H:%M') porter: S25 attribution vs S24 rows: j3+treuse $(sed -n 4p $H/gate_S25r_noR320_vs_S24.txt | cut -c1-120) | R320 part (package vs j3+treuse) $(sed -n 4p $H/gate_S25r_R320part.txt | cut -c1-120). S25 NOT staged (held)." >> $TL
  echo "$(date +%T) S25 CHAIN DONE (not staged; attribution written)"; CHAIN_DONE=1; exit 0
fi
echo "$(date '+%Y-%m-%d %H:%M') porter: S25 PASSES the VALID40 bar vs S24: $(sed -n 4p $H/gate_S25r_valid40_vs_S24.txt | cut -c1-160); residual gate next." >> $TL
echo "$(date +%T) [8] paired residual gate vs S24 (setup client)"
bash $H/machine_gate.sh 13 S25r-residual-gate
while ! mkdir /tmp/fable_paired.lock 2>/dev/null; do sleep 30; done
echo "$(date +%T) residual gate S24 vs S25 start | load $(sysctl -n vm.loadavg)"
for rep in 1 2 3; do
  if [ $((rep % 2)) = 1 ]; then A=$S24F; TA=rz25r_S24_r$rep; BB=$C; TB=rz25r_S25_r$rep; else A=$C; TA=rz25r_S25_r$rep; BB=$S24F; TB=rz25r_S24_r$rep; fi
  bash $H/run_census_porter_v3.sh $A 35,100,37 $TA > $W/$TA.out 2>&1 & a=$!
  sleep 1
  bash $H/run_census_porter_v3.sh $BB 35,100,37 $TB > $W/$TB.out 2>&1 & b=$!
  wait $a $b; echo "$(date +%T) residual rep $rep done"
done
rmdir /tmp/fable_paired.lock
python3 - $H <<'PY' | tee $W/residual_gate_S25r.txt
import json, sys, statistics as st
H = sys.argv[1]
def load(tag): return {r['rank']: r for r in json.load(open(f'{H}/{tag}_census.json'))}
R = {c: [load(f'rz25r_{c}_r{k}') for k in (1, 2, 3)] for c in ('S24', 'S25')}
print('TAG fable-porter PT-54: paired residual gate S24 vs S25 (setup-calling client, concurrent pairs, 3 reps, nets 35 cold,100,37)')
cold = {}
for n in (35, 100, 37):
    a = [r[n]['residual_s'] for r in R['S24']]; b = [r[n]['residual_s'] for r in R['S25']]
    rq = (st.mean(r[n]['requests'] for r in R['S24']), st.mean(r[n]['requests'] for r in R['S25']))
    if n == 35: cold = dict(A=st.mean(a), B=st.mean(b))
    print(f"  r{n}{' (cold)' if n == 35 else ''}: S24 mean {st.mean(a):.3f} max {max(a):.3f} s | S25 mean {st.mean(b):.3f} max {max(b):.3f} s | delta {st.mean(b)-st.mean(a):+.3f} s | requests {rq[0]:,.0f} -> {rq[1]:,.0f}")
d = cold['B'] - cold['A']
print(f"  S25 cold delta vs S24 {d:+.3f} s (rule <= +0.010 s) -> RESIDUAL GATE {'PASS' if d <= 0.010 else 'FLAG'}")
PY
echo "$(date '+%Y-%m-%d %H:%M') porter: S25 residual gate vs S24: $(tail -1 $W/residual_gate_S25r.txt | sed 's/^ *//')" >> $TL
grep -q 'RESIDUAL GATE' $W/residual_gate_S25r.txt || fail "residual verdict missing (stack/residual_gate_S25r.txt)"
echo "$(date +%T) [9] staging"
[ -e $FD ] && fail "fire dir exists; not touching it"
mkdir $FD; F=$FD/candS25_S24_j3_R320kl_treuse_mem.py; cp -p $C $F
{ echo "FIRE CHECKS S25 = S24 (fired 01:40, 30810da3) + j3cut (engineer2) + old-tier rank 320 with a matched prebind key list (porter) + treuse (engineer E-73, last), $(date '+%Y-%m-%d %H:%M')"
  echo "file $F"; echo "sha256 $(shasum -a 256 $F | cut -c1-64)"
  echo "inputs: j3cut fragment 4874216a (rebuilt on S24 byte-identical); treuse_fragment_v2.py 75aa9233; R320 key census $(basename $CEN) $(shasum -a 256 $CEN | cut -c1-16)"
  echo "R320 diff (vs S24 + j3): the 3 build_tier_rank.py edits only; key list: $(grep -v '^self-test' $W/s25r_keylist.txt | tr '\n' ' ' | cut -c1-500)"
  echo "literals (fire copy): $(cd $H && ./verify_literals.py $F $PT) | validate: $(/Users/matt/arc-white-box-estimation/.venv/bin/whest validate --estimator $F 2>&1 | grep -o 'Validation passed.*\|Validation failed.*')"
  echo "einsum gate (fire copy) vs fired S16: $(python3 $H/einsum_site_diff.py $F $S16F | tail -1) | vs S23: $(python3 $H/einsum_site_diff.py $F $S23F | tail -1 | cut -c1-14) | vs S24: $(python3 $H/einsum_site_diff.py $F $S24F | tail -1 | cut -c1-14)"
  echo "treuse identity (engineer's e73_parity.py, setup(); S25pre vs S25): $(grep -v '^{' $W/S25r_treuse_identity.txt | tr '\n' ' ' | cut -c1-300) $(python3 -c "import json,sys; rows=[json.loads(l) for l in open(sys.argv[1]) if l.startswith('{')]; print('; '.join(f\"r{r['rank']} identical {r['all16_identical']} maxdiff {r['max_abs_diff']} flops {100*(r['flops_B']/r['flops_A']-1):+.3f}%\" for r in rows if 'all16_identical' in r))" $W/S25r_treuse_identity.txt)"
  echo "client/server smoke with setup(): $(grep -c 'OK requests' $W/smoke_S25r.txt) of 2 nets OK; $(grep 'smoke setup' $W/smoke_S25r.txt)"
  echo "VALID40 vs S24 rows: $(sed -n 2,4p $H/gate_S25r_valid40_vs_S24.txt | tr '\n' ' ')"
  echo "bar (adjusted <= ${BAR}% vs S24, CI excl 0): $(cat $H/gate_S25r_bar.json)"
  echo "residual gate vs S24 (setup-calling client): $(sed -n 2,5p $W/residual_gate_S25r.txt | tr '\n' ' ')"
} > $FD/FIRE_CHECKS.txt 2>&1
cat $FD/FIRE_CHECKS.txt
echo "$(date '+%Y-%m-%d %H:%M') porter: S25 STAGED: fable_fire_v1/fire_candS25/$(basename $F) ($(shasum -a 256 $F | cut -c1-8); FIRE_CHECKS.txt; VALID40 $(sed -n 4p $H/gate_S25r_valid40_vs_S24.txt | cut -c1-90); residual $(tail -1 $W/residual_gate_S25r.txt | grep -o 'PASS\|FLAG'))." >> $TL
echo "$(date +%T) S25 CHAIN DONE (fire copy staged)"; CHAIN_DONE=1
