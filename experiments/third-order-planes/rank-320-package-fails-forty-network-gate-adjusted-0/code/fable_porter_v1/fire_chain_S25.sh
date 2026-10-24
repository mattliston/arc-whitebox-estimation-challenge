#!/usr/bin/env bash
# TAG fable-porter (PT-52, lead 02:1x "S25 inputs"): S25 = S24 (fired 01:40, 30810da3) + the engineer's treuse_fragment_v2.py (E-73: bit-identical,
# bill down) + engineer2's j3cut fragment (an approximation) appended last. Optional arg 2 = a new dials json (the physicist's c0 line search): the
# loopR3 S23-structure base is rebuilt with it first (prebuild_S23struct.sh; diff vs S24cpre = dial constants only), then lvfuse2 + star-Nystrom
# (base diff vs S24 = dial lines only). Gates: j3cut applies (the base's _bd_update body = the S23 body it was generated from) -> treuse identity
# (parity_layers_setup_v2.py with setup(), nets 35/100: predictions identical, flops not higher, reuses > 0) -> composed einsum (vs S16, S23, S24) +
# validate + literals -> smoke with setup() -> VALID40 vs the S24 rows (bar adj <= BAR, default -0.3, CI excl 0) -> paired residual vs S24 ->
# staging fable_fire_v1/fire_candS25/ (refused unless every verdict exists).   usage: fire_chain_S25.sh [BAR] [DIALS json | none]  (log fire_chain_S25.out)
set -u
P2=/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16; H=$P2/fable_porter_v1; TL=$P2/fable_refit_v1/TEAM_LOG.md
BAR=${1:--0.3}; DN=${2:-none}; R3=$P2/fable_physicist_v1/refit6b/s6b_loopR3_joint_s23; PT=$R3/params_epoch0.pt
LVF=$P2/fable_engineer_v1/lvfuse_fragment_v2.py; LRN=$H/stack/lrnys_fragment_q1.py; TR=$P2/fable_engineer_v1/treuse_fragment_v2.py; J3=$P2/fable_engineer2_v1/j3cut/S23_j3cut_mem_fragment.py
S24F=$P2/fable_fire_v1/fire_candS24/candS24_S23_loopR3_lvf2_lrnys_mem.py; S24PRE=$H/stack/stack_S24cpre_mem.py; S23F=$P2/fable_fire_v1/fire_candS23/candS23_S21s1_loopR1_s1z_jn_mem.py
S16F=$P2/fable_fire_v1/fire_candS16/candS16_S15_bb6_mem.py; J24=$H/paired_valid40_stack_S24c_mem.json; NPY=/Users/matt/arc-white-box-estimation/.venv/bin/python
T=$H/stack/stack_S25_tr_mem.py; C=$H/stack/stack_S25_mem.py; FD=$P2/fable_fire_v1/fire_candS25
CHAIN_DONE=0; trap 'rc=$?; [ $CHAIN_DONE = 1 ] || echo "$(date "+%Y-%m-%d %H:%M") porter: S25 CHAIN DIED rc=$rc (fable_porter_v1/fire_chain_S25.out) last: $(tail -2 $H/fire_chain_S25.out 2>/dev/null | tr "\n" " " | cut -c1-200)" >> $TL' EXIT
fail() { echo "$(date '+%Y-%m-%d %H:%M') porter: S25 NOT staged: $1" >> $TL; echo "$(date +%T) FAIL: $1"; CHAIN_DONE=1; exit 1; }
DIALRX='^[<>] (_BD_J1_PHYS|_S19_BD_ADD|_S19_BSCALE|_S19_LAM|_S19_K4BAND|_K31R_TABLE) = '
for f in $T $C $FD; do [ -e $f ] && fail "$f already exists (never overwrite)"; done
[ "$(shasum -a 256 $S24F | cut -c1-16)" = 30810da3b302c5ec ] || fail "S24 fire copy sha changed"
[ "$(shasum -a 256 $TR | cut -c1-16)" = 75aa92335f4b1de9 ] || fail "treuse fragment sha changed"
[ "$(shasum -a 256 $J3 | cut -c1-16)" = 4874216ad2a28990 ] || fail "j3cut fragment sha changed"
[ "$(shasum -a 256 $PT | cut -c1-16)" = 0750b69ba263cd9d ] || fail "loopR3 params sha changed"
[ "$(shasum -a 256 $S24PRE | cut -c1-16)" = c396ba73e0e7b30b ] || fail "S24cpre sha changed"
python3 -c "import json,sys; d=json.load(open(sys.argv[1]))['results']; assert len([k for k in d if k.startswith('cand:')]) == 40" $J24 || fail "S24 VALID40 rows missing or not 40"
if [ "$DN" = none ]; then
  B=$S24F; BASEDESC="S24 fire copy (no dial change)"
else
  [ "$(shasum -a 256 $LVF | cut -c1-16)" = 8771edba106b515e ] && [ "$(shasum -a 256 $LRN | cut -c1-16)" = de2111a99bd127b6 ] || fail "lvfuse2 / star-Nystrom fragment sha changed"
  python3 -c "import json,math,sys; d=json.load(open(sys.argv[1])); K={'X1','X0','X2','X3','bscale','s_carry','s_new','c0','c1','k4a','k4b','k4c'}; assert set(d)==K and all(isinstance(v,float) and math.isfinite(v) for v in d.values()), d" $DN || fail "dials $DN are not the 12 finite flat keys"
  X=$H/stack/stack_S25pre_mem.py; B=$H/stack/stack_S25base_mem.py; for f in $X $B; do [ -e $f ] && fail "$f already exists"; done
  cp -p $DN $H/stack/s25_dials_used.json
  echo "$(date +%T) [0] rebuild the loopR3 base with the new dials"; bash $H/machine_gate.sh 4 S25-build
  bash $H/prebuild_S23struct.sh $PT $H/stack/s25_dials_used.json S25pre > $H/stack/prebuild_S25pre_chain.txt 2>&1; cat $H/stack/prebuild_S25pre_chain.txt
  [ -s $X ] && grep -q "ALL 28 BIT-EQUAL" $H/stack/prebuild_S25pre_chain.txt && grep -q "VERDICT PASS" $H/stack/prebuild_S25pre_chain.txt && grep -q "Validation passed" $H/stack/prebuild_S25pre_chain.txt || fail "S25pre build checks"
  DL=$(diff $S24PRE $X | grep '^[<>]' | grep -v -E "$DIALRX" | wc -l | tr -d ' '); [ "$DL" = 0 ] || fail "S25pre differs from S24cpre beyond the dial constants ($DL lines)"
  cat $X $LVF $LRN > $B
  DB=$(diff $S24F $B | grep '^[<>]' | grep -v -E "$DIALRX" | wc -l | tr -d ' '); [ "$DB" = 0 ] || fail "S25 base differs from S24 beyond the dial constants ($DB lines)"
  BASEDESC="loopR3 base rebuilt with $(basename $DN) ($(shasum -a 256 $DN | cut -c1-16)); diff vs S24 by name: $(diff $S24F $B | grep '^[<>]' | sed -E 's/^([<>]) ([A-Za-z_0-9]+) = .*/\1 \2/' | sort | uniq -c | tr '\n' ' ')"
fi
echo "base: $BASEDESC"
python3 - $B $S23F <<'PY' || fail "the base's _bd_update body differs from S23's (rebuild j3cut with build_j3cut.py)"
import sys
def body(path, name):
    lines = open(path).read().splitlines(); i = max(k for k, l in enumerate(lines) if l.startswith(f'def {name}('))
    out = [lines[i]]
    for l in lines[i + 1:]:
        if l and not l[0].isspace() and not l.startswith('#'): break
        out.append(l)
    return '\n'.join(out).rstrip()
assert body(sys.argv[1], '_bd_update') == body(sys.argv[2], '_bd_update'); print('j3cut applies: the base _bd_update body = S23 body')
PY
echo "$(date +%T) [1] compose: base + treuse, + j3cut last"
cat $B $TR > $T; cat $T $J3 > $C
for f in $T $C; do
  E16=$(python3 $H/einsum_site_diff.py $f $S16F | tail -1); E23=$(python3 $H/einsum_site_diff.py $f $S23F | tail -1); E24=$(python3 $H/einsum_site_diff.py $f $S24F | tail -1)
  V=$(/Users/matt/arc-white-box-estimation/.venv/bin/whest validate --estimator $f 2>&1 | grep -o 'Validation passed.*\|Validation failed.*')
  echo "$(basename $f): einsum vs S16: $E16 | vs S23: ${E23:0:14} | vs S24: ${E24:0:14} | $V"
  for e in "$E16" "$E23" "$E24"; do echo "$e" | grep -q "VERDICT PASS" || fail "$(basename $f) einsum gate"; done
  echo "$V" | grep -q "Validation passed" || fail "$(basename $f) validate"
done
(cd $H && ./verify_literals.py $C $PT) | tee $H/stack/S25_literals.txt; grep -q "ALL 28 BIT-EQUAL" $H/stack/S25_literals.txt || fail "composed literals"
echo "$(date +%T) [2] treuse identity: base vs base+treuse, in process with setup(), nets 35,100"
bash $H/machine_gate.sh 8 S25-treuse-identity
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 nice -n 10 $NPY $H/parity_layers_setup_v2.py $B $T 35,100 2>&1 | grep -v "Warning\|^  \|configure(" > $H/stack/S25_treuse_identity.txt; cat $H/stack/S25_treuse_identity.txt
python3 - $H/stack/S25_treuse_identity.txt <<'PY' || fail "treuse not bit-identical, bill higher, or no reuses (stack/S25_treuse_identity.txt)"
import re, sys
t = open(sys.argv[1]).read(); rows = re.findall(r'r(\d+): all-16-layer predictions identical (\w+) .*?flops A ([\d,]+) B ([\d,]+)', t)
assert len(rows) == 2 and all(p == 'True' for _, p, _, _ in rows), rows
fl = [(int(a.replace(',', '')), int(b.replace(',', ''))) for _, _, a, b in rows]; assert all(b <= a for a, b in fl), fl
reused = [int(x) for x in re.findall(r"tr A None B \{'reused': (\d+)", t)]; assert reused and reused[-1] > 0, reused
print('treuse identity PASS: ' + '; '.join(f'r{n} bill {100 * (b / a - 1):+.3f}%' for (n, _, _, _), (a, b) in zip(rows, fl)) + f'; reuses {reused[-1]}')
PY
echo "$(date '+%Y-%m-%d %H:%M') porter: S25 ($(shasum -a 256 $C | cut -c1-8)) built on $BASEDESC; treuse bit-identical with setup() on 35/100; j3cut applies; einsum (S16/S23/S24) + validate + literals PASS; smoke next." >> $TL
echo "$(date +%T) [3] smoke with setup() (census v3, 35,100)"
bash $H/machine_gate.sh 7 S25-smoke
while ! mkdir /tmp/fable_paired.lock 2>/dev/null; do sleep 30; done
bash $H/run_census_porter_v3.sh $C 35,100 cz_S25 > $H/stack/cz_S25.out 2>&1
rmdir /tmp/fable_paired.lock
python3 - $H/cz_S25_census.json $H/stack/cz_S25.out > $H/stack/smoke_S25.txt 2>&1 <<'PY'
import json, sys
rows = json.load(open(sys.argv[1])); assert [r['rank'] for r in rows] == [35, 100], rows
setup = [l for l in open(sys.argv[2]) if 'setup_wall_s' in l]; assert len(setup) == 1, setup
s = json.loads(setup[0].split(': ', 1)[1]); log = s['setup_log']
assert log and not any(str(e[0]).startswith('restored') for e in log), ('setup error (restored)', s)
bound = [e for e in log if e[0] == 'bound']; assert bound, ('no bound entry', s)
assert s['setup_wall_s'] <= 3.1, ('setup wall > 3.1 s', s)
guard = [e for e in log if e[0] == 'wall guard']; lvf = [e for e in log if e[0] == 'lvfuse']
print(f"smoke setup(): wall {s['setup_wall_s']} s, requests {s['setup_requests']:,}, bound {bound[-1][1]} entries, lvfuse {lvf[-1][1:] if lvf else None}" + (f" (WALL GUARD held at {guard[0][1]} under load; accepted per lead 22:2x)" if guard else ''))
for r in rows:
    assert r['requests'] > 0 and r['flops'] > 0, r
    assert r.get('n_out') == 16 * 1024 and r.get('nonfinite') == 0, ('outputs', r.get('n_out'), r.get('nonfinite'))
    print(f"smoke r{r['rank']}: OK requests {r['requests']:,} server CPU {r['server_cpu_s']} s residual {r['residual_s']} s util {r['util']} outputs {r['n_out']} all finite")
PY
[ $? -eq 0 ] || { cat $H/stack/smoke_S25.txt; tail -20 $H/stack/cz_S25.out; fail "SMOKE FAILED (stack/smoke_S25.txt)"; }
cat $H/stack/smoke_S25.txt
echo "$(date '+%Y-%m-%d %H:%M') porter: S25 smoke OK ($(grep -o 'wall [0-9.]* s' $H/stack/smoke_S25.txt | head -1) setup); VALID40 vs S24 rows STARTED (bar adj <= ${BAR}%, CI excl 0)." >> $TL
echo "$(date +%T) [4] VALID40 vs the S24 rows"
bash $H/machine_gate.sh 8 S25-VALID40
bash $H/run_valid40_locked.sh $C stack_S25_mem
python3 $H/valid40_vs_ref_ci.py $H/paired_valid40_stack_S25_mem.json $J24 > $H/gate_S25_valid40_vs_S24.txt; cat $H/gate_S25_valid40_vs_S24.txt
python3 $H/valid40_bar.py $H/paired_valid40_stack_S25_mem.json $J24 $BAR 0 | tee $H/gate_S25_bar.json
if ! grep -q '"verdict": "PASS"' $H/gate_S25_bar.json; then
  echo "$(date '+%Y-%m-%d %H:%M') porter: S25 NOT staged (VALID40 bar adj <= ${BAR}% vs S24, CI excl 0): $(cut -c1-200 $H/gate_S25_bar.json)" >> $TL; echo "$(date +%T) S25 CHAIN DONE (not staged)"; CHAIN_DONE=1; exit 0
fi
echo "$(date '+%Y-%m-%d %H:%M') porter: S25 PASSES the VALID40 bar vs S24: $(sed -n 4p $H/gate_S25_valid40_vs_S24.txt | cut -c1-160); residual gate next." >> $TL
echo "$(date +%T) [5] paired residual gate vs S24 (setup client)"
bash $H/machine_gate.sh 13 S25-residual-gate
while ! mkdir /tmp/fable_paired.lock 2>/dev/null; do sleep 30; done
echo "$(date +%T) residual gate S24 vs S25 start | load $(sysctl -n vm.loadavg)"
for rep in 1 2 3; do
  if [ $((rep % 2)) = 1 ]; then A=$S24F; TA=rz25_S24_r$rep; BB=$C; TB=rz25_S25_r$rep; else A=$C; TA=rz25_S25_r$rep; BB=$S24F; TB=rz25_S24_r$rep; fi
  bash $H/run_census_porter_v3.sh $A 35,100,37 $TA > $H/stack/$TA.out 2>&1 & a=$!
  sleep 1
  bash $H/run_census_porter_v3.sh $BB 35,100,37 $TB > $H/stack/$TB.out 2>&1 & b=$!
  wait $a $b; echo "$(date +%T) residual rep $rep done"
done
rmdir /tmp/fable_paired.lock
python3 - $H <<'PY' | tee $H/stack/residual_gate_S25.txt
import json, sys, statistics as st
H = sys.argv[1]
def load(tag): return {r['rank']: r for r in json.load(open(f'{H}/{tag}_census.json'))}
R = {c: [load(f'rz25_{c}_r{k}') for k in (1, 2, 3)] for c in ('S24', 'S25')}
print('TAG fable-porter PT-52: paired residual gate S24 vs S25 (setup-calling client, concurrent pairs, 3 reps, nets 35 cold,100,37)')
cold = {}
for n in (35, 100, 37):
    a = [r[n]['residual_s'] for r in R['S24']]; b = [r[n]['residual_s'] for r in R['S25']]
    rq = (st.mean(r[n]['requests'] for r in R['S24']), st.mean(r[n]['requests'] for r in R['S25']))
    if n == 35: cold = dict(A=st.mean(a), B=st.mean(b))
    print(f"  r{n}{' (cold)' if n == 35 else ''}: S24 mean {st.mean(a):.3f} max {max(a):.3f} s | S25 mean {st.mean(b):.3f} max {max(b):.3f} s | delta {st.mean(b)-st.mean(a):+.3f} s | requests {rq[0]:,.0f} -> {rq[1]:,.0f}")
d = cold['B'] - cold['A']
print(f"  S25 cold delta vs S24 {d:+.3f} s (rule <= +0.010 s) -> RESIDUAL GATE {'PASS' if d <= 0.010 else 'FLAG'}")
PY
echo "$(date '+%Y-%m-%d %H:%M') porter: S25 residual gate vs S24: $(tail -1 $H/stack/residual_gate_S25.txt | sed 's/^ *//')" >> $TL
grep -q 'RESIDUAL GATE' $H/stack/residual_gate_S25.txt || fail "residual verdict missing (stack/residual_gate_S25.txt)"
echo "$(date +%T) [6] staging"
[ -e $FD ] && fail "fire dir exists; not touching it"
mkdir $FD; F=$FD/candS25_S24_treuse_j3$( [ "$DN" = none ] || echo _dials)_mem.py; cp -p $C $F
{ echo "FIRE CHECKS S25 = S24 (fired 01:40, 30810da3) + treuse_fragment_v2.py (engineer E-73) + j3cut (engineer2), base: $BASEDESC; $(date '+%Y-%m-%d %H:%M')"
  echo "file $F"; echo "sha256 $(shasum -a 256 $F | cut -c1-64)"
  echo "inputs: treuse_fragment_v2.py 75aa9233, S23_j3cut_mem_fragment.py 4874216a$( [ "$DN" = none ] || echo "; dials $(tr -d '\n ' < $DN | cut -c1-300)")"
  echo "composition: cat <base> treuse_fragment_v2.py S23_j3cut_mem_fragment.py; literals (fire copy): $(cd $H && ./verify_literals.py $F $PT)"
  echo "validate (fire copy): $(/Users/matt/arc-white-box-estimation/.venv/bin/whest validate --estimator $F 2>&1 | grep -o 'Validation passed.*\|Validation failed.*')"
  echo "einsum gate (fire copy) vs fired S16: $(python3 $H/einsum_site_diff.py $F $S16F | tail -1) | vs S23: $(python3 $H/einsum_site_diff.py $F $S23F | tail -1 | cut -c1-14) | vs S24: $(python3 $H/einsum_site_diff.py $F $S24F | tail -1 | cut -c1-14)"
  echo "treuse identity (in process, setup(); base vs base+treuse): $(grep -v '^TAG' $H/stack/S25_treuse_identity.txt | tr '\n' ' ' | cut -c1-700)"
  echo "client/server smoke with setup(): $(grep -c 'OK requests' $H/stack/smoke_S25.txt) of 2 nets OK; $(grep 'smoke setup' $H/stack/smoke_S25.txt)"
  echo "VALID40 vs S24 rows: $(sed -n 2,4p $H/gate_S25_valid40_vs_S24.txt | tr '\n' ' ')"
  echo "bar (adjusted <= ${BAR}% vs S24, CI excl 0): $(cat $H/gate_S25_bar.json)"
  echo "residual gate vs S24 (setup-calling client): $(sed -n 2,5p $H/stack/residual_gate_S25.txt | tr '\n' ' ')"
} > $FD/FIRE_CHECKS.txt 2>&1
cat $FD/FIRE_CHECKS.txt
echo "$(date '+%Y-%m-%d %H:%M') porter: S25 STAGED: fable_fire_v1/fire_candS25/$(basename $F) ($(shasum -a 256 $F | cut -c1-8); FIRE_CHECKS.txt; VALID40 $(sed -n 4p $H/gate_S25_valid40_vs_S24.txt | cut -c1-90); residual $(tail -1 $H/stack/residual_gate_S25.txt | grep -o 'PASS\|FLAG'))." >> $TL
echo "$(date +%T) S25 CHAIN DONE (fire copy staged)"; CHAIN_DONE=1
