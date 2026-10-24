#!/usr/bin/env bash
# TAG fable-porter (PT-62, lead 09:3x "(4) S28 = S27 + the best refit checkpoint + the cap raise (verdict PASS) + k-split v5w if parity passes
# + group 5 v2 (b769aa28, before treuse) + the fixed schedule if it passes + treuse last; VALID40 vs the S27 rows, adj <= -0.5%, CI excl 0;
# staged and residual-gated by 16:30"). Build from the S24 fire copy: (1) j3cut (engineer2 build_j3cut.py; fragment 4874216a); (2) + k-split
# fragment; (3) + cap fragment [optional]; (4) refit checkpoint: prebuild_S23struct.sh <params> <dials> -> the 5 literal statements and the
# whole S19 dials block are transplanted (diff = those lines only; literals then bit-equal to the .pt); (5) + per-join schedule
# (build_r320_router_v2.py <rule>) [optional]; (6) group 5 v2 built on that exact file (engineer2 build_g5_B_v2.py; fragment pinned) [optional];
# (7) key census of that pre-treuse file -> key list (v2 rule); (8) + treuse LAST. Gates: einsum (S16/S23/S24/S27) + validate + literals +
# k-split settings -> treuse identity (e73_parity, verdict teed) -> chunk-key coverage (info) -> smoke with setup() (<= 3.6 s) -> VALID40 vs the
# S27 rows (bar adj <= BAR, CI excl 0; raw separate) -> paired residual vs the S27 fire copy (setup walls under load <= 3.6 s) -> staging.
# 11:5x (engineer2 + lead): key list by build_pb_keylist_v4.py (candidates >= 1 of 8 nets, value = freq x bind bill / 8 ms, per side 480,
# total <= 460 (lead 12:0x: 460 + 22 leaf buffers ~ 3.86 s at S27's 8.0 ms/key platform rate, under the 4.0 s guard)), prebind guard _PB_WALL_S = 4.0 (one override line before treuse), setup gates <= 4.1 s (smoke + concurrent).
# 10:4x: after the key list, the k-split LEAF-BUFFERS-FIRST setup fix (build_setup_variants.py ksfirst), as in S27s.
# 10:0x: the smoke also requires ALL intended prebind keys bound (list + batched + k-split leaf buffers), since the 3.6 s wall gate cannot fail at a 3.5 s guard.
# usage: fire_chain_S28.sh <BAR> <ksplit frag> <sha16> <cap frag|none> <sha16|none> <params.pt> <sha16> <dials.json> <sha16>
#                          <rule.json|none> <build_g5_B_v2.py|none> <builder sha16|none> <gated fragment sha prefix|none>   (log fire_chain_S28.out)
set -u
P2=/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16; H=$P2/fable_porter_v1; TL=$P2/fable_refit_v1/TEAM_LOG.md; W=$H/stack
BAR=$1; KSF=$2; KSS=$3; CPF=$4; CPS=$5; RP=$6; RPS=$7; RD=$8; RDS=$9; RULE=${10}; G5=${11}; G5S=${12}; G5F=${13}; NPY=/Users/matt/arc-white-box-estimation/.venv/bin/python
S24F=$P2/fable_fire_v1/fire_candS24/candS24_S23_loopR3_lvf2_lrnys_mem.py; S27F=$P2/fable_fire_v1/fire_candS27/candS27_S26_capA_mem.py; S23F=$P2/fable_fire_v1/fire_candS23/candS23_S21s1_loopR1_s1z_jn_mem.py
S16F=$P2/fable_fire_v1/fire_candS16/candS16_S15_bb6_mem.py; TR=$P2/fable_engineer_v1/treuse_fragment_v2.py; J3B=$P2/fable_engineer2_v1/j3cut/build_j3cut.py; J3F=$P2/fable_engineer2_v1/j3cut/S23_j3cut_mem_fragment.py
J27=$H/paired_valid40_stack_S27_mem.json; A1=$W/s28_j3.py; A2=$W/s28_ks.py; A3=$W/s28_cap.py; Y=$W/stack_S28lit_mem.py; A4=$W/s28_lit.py; A5=$W/s28_sched.py; A6=$W/s28_g5.py
PKL=$W/s28_prekl.py; KSFR=$W/s28_ksfirst_fragment.py; PRE=$W/s28_pre.py; C=$W/stack_S28_mem.py; CEN=$H/ipt_key_census_S28.json; FD=$P2/fable_fire_v1/fire_candS28
CHAIN_DONE=0; trap 'rc=$?; [ $CHAIN_DONE = 1 ] || echo "$(date "+%Y-%m-%d %H:%M") porter: S28 CHAIN DIED rc=$rc (fable_porter_v1/fire_chain_S28.out) last: $(tail -2 $H/fire_chain_S28.out 2>/dev/null | tr "\n" " " | cut -c1-200)" >> $TL' EXIT
fail() { echo "$(date '+%Y-%m-%d %H:%M') porter: S28 NOT staged: $1" >> $TL; echo "$(date +%T) FAIL: $1"; CHAIN_DONE=1; exit 1; }
for f in $A1 $A2 $A3 $Y $A4 $A5 $A6 $PKL $KSFR $PRE $C $CEN $FD; do [ -e $f ] && fail "$f already exists (never overwrite)"; done
[ "$(shasum -a 256 $S24F | cut -c1-16)" = 30810da3b302c5ec ] || fail "S24 fire copy sha changed"
[ "$(shasum -a 256 $S27F | cut -c1-16)" = 9ff80a12de5a5fb8 ] || fail "S27 fire copy sha changed"
[ "$(shasum -a 256 $TR | cut -c1-16)" = 75aa92335f4b1de9 ] || fail "treuse fragment sha changed"
[ "$(shasum -a 256 $J3F | cut -c1-16)" = 4874216ad2a28990 ] || fail "j3cut fragment sha changed"
[ "$(shasum -a 256 $KSF | cut -c1-16)" = "$KSS" ] || fail "k-split fragment sha != $KSS"
[ "$CPF" = none ] || [ "$(shasum -a 256 $CPF | cut -c1-16)" = "$CPS" ] || fail "cap fragment sha != $CPS"
[ "$(shasum -a 256 $RP | cut -c1-16)" = "$RPS" ] || fail "refit params sha != $RPS"
[ "$(shasum -a 256 $RD | cut -c1-16)" = "$RDS" ] || fail "refit dials sha != $RDS"
[ "$G5" = none ] || [ "$(shasum -a 256 $G5 | cut -c1-16)" = "$G5S" ] || fail "group 5 builder sha != $G5S"
python3 -c "import json,math,sys; d=json.load(open(sys.argv[1])); d=d.get('dials', d); K={'X1','X0','X2','X3','bscale','s_carry','s_new','c0','c1','k4a','k4b','k4c'}; assert set(d)==K and all(isinstance(v,float) and math.isfinite(v) for v in d.values()), d" $RD || fail "refit dials are not the 12 finite keys"
python3 -c "import json,sys; d=json.load(open(sys.argv[1]))['results']; assert len([k for k in d if k.startswith('cand:')]) == 40" $J27 || fail "S27 VALID40 rows missing or not 40"
DESC="S24"
echo "$(date +%T) [1] j3cut on S24"
python3 $J3B $S24F $A1 > /dev/null || fail "build_j3cut"
[ "$(shasum -a 256 ${A1%.py}_fragment.py | cut -c1-16)" = 4874216ad2a28990 ] && [ "$(shasum -a 256 $A1 | cut -c1-16)" = "$(cat $S24F $J3F | shasum -a 256 | cut -c1-16)" ] || fail "j3cut rebuilt on S24 differs"
DESC="$DESC + j3cut"
echo "$(date +%T) [2] + k-split $(basename $KSF)"; cat $A1 $KSF > $A2; DESC="$DESC + $(basename $KSF .py)"; B=$A2
if [ "$CPF" != none ]; then echo "$(date +%T) [3] + cap $(basename $CPF)"; cat $B $CPF > $A3; B=$A3; DESC="$DESC + $(basename $CPF .py)"; fi
echo "$(date +%T) [4] refit checkpoint literals + dials (prebuild_S23struct.sh, transplant)"
cp -p $RD $W/s28_dials_used.json; python3 -c "import json,sys; d=json.load(open(sys.argv[1])); d=d.get('dials', d); json.dump(d, open(sys.argv[2], 'w'), indent=1)" $RD $W/s28_dials_flat.json
bash $H/machine_gate.sh 4 S28-build
bash $H/prebuild_S23struct.sh $RP $W/s28_dials_flat.json S28lit > $W/prebuild_S28lit_chain.txt 2>&1; cat $W/prebuild_S28lit_chain.txt
[ -s $Y ] && grep -q "ALL 28 BIT-EQUAL" $W/prebuild_S28lit_chain.txt && grep -q "VERDICT PASS" $W/prebuild_S28lit_chain.txt && grep -q "Validation passed" $W/prebuild_S28lit_chain.txt || fail "refit prebuild checks"
python3 - $Y $B $A4 <<'PY' || fail "literal/dials transplant"
import sys
src, tgt = open(sys.argv[1]).read(), open(sys.argv[2]).read()
names = ('_BD_COEFFICIENT_LITERAL', '_BDNH_A_LITERAL', '_EDGE_LITERAL', '_GNN_LITERAL', '_MESSAGE_EDGE_LITERAL')
def lit(text, n):
    L = [l for l in text.split('\n') if l.startswith(n + ' = ')]; assert len(L) == 1, (n, len(L)); return L[0]
def block(text):
    h = '# PT-33 S19 dials (covhunter learned-ON joint fit, s6bcard_joint semantics)\n'; assert text.count(h) == 1, text.count(h)
    i = text.index(h); j = text.index('\n_K31R_TABLE = {l: (', i); k = text.index('\n', j + 1); return text[i:k + 1]
out = tgt
for n in names: out = out.replace(lit(tgt, n), lit(src, n))
out = out.replace(block(tgt), block(src)); assert out.count(block(src)) == 1
open(sys.argv[3], 'w').write(out); print('transplanted: 5 literal statements + the S19 dials block')
PY
DT=$(diff $B $A4 | grep '^[<>]' | grep -v -E '^[<>] (_BD_COEFFICIENT_LITERAL|_BDNH_A_LITERAL|_EDGE_LITERAL|_GNN_LITERAL|_MESSAGE_EDGE_LITERAL|_BD_J1_PHYS|_S19_BD_ADD|_S19_BSCALE|_S19_LAM|_S19_K4BAND|_K31R_TABLE) = ' | wc -l | tr -d ' ')
[ "$DT" = 0 ] || fail "transplant changed lines outside the literal statements and dial constants ($DT)"
(cd $H && ./verify_literals.py $A4 $RP) | grep -q "ALL 28 BIT-EQUAL" || fail "transplanted literals are not bit-equal to the refit .pt"
B=$A4; DESC="$DESC + refit $(basename $(dirname $RP))/$(basename $RP) ($RPS)"
if [ "$RULE" != none ]; then
  echo "$(date +%T) [5] per-join schedule $(tr -d ' \n' < $RULE)"; cp -p $RULE $W/s28_rule_used.json
  python3 $H/build_r320_router_v2.py $B $A5 $RULE > $W/s28_router_build.txt || fail "router build"; B=$A5; DESC="$DESC + schedule $(tr -d ' \n' < $RULE)"
fi
if [ "$G5" != none ]; then
  echo "$(date +%T) [6] group 5 on the exact pre-treuse file ($(basename $G5))"
  python3 $G5 $B $A6 > $W/s28_g5_build.txt 2>&1 || { cat $W/s28_g5_build.txt; fail "group 5 builder"; }; cat $W/s28_g5_build.txt
  GFS=$(shasum -a 256 ${A6%.py}_fragment.py | cut -c1-16); [ "${GFS:0:${#G5F}}" = "$G5F" ] || fail "group 5 fragment $GFS != the gated fragment $G5F (engineer2 must re-gate)"
  python3 -c "import sys; b, f, o = (open(x).read() for x in sys.argv[1:4]); assert o in (b + f, b.rstrip(chr(10)) + chr(10) + f)" $B ${A6%.py}_fragment.py $A6 || fail "group 5 output != base + fragment"
  B=$A6; DESC="$DESC + group 5 ($GFS)"
fi
echo "$(date +%T) [7] key census of the final pre-treuse file (300..580 + 35/100/37/39)"
bash $H/machine_gate.sh 8 S28-key-census
(cd $H && OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 nice -n 10 $NPY ipt_key_census_porter.py $B 300,340,380,420,460,500,540,580,35,100,37,39 S28) 2>&1 | grep -v "Warning\|^  \|configure(" > $W/s28_census.log; tail -2 $W/s28_census.log
[ -s $CEN ] || fail "key census"
python3 $H/build_pb_keylist_v4.py $CEN $B $PKL --max-total=460 | tee $W/s28_keylist.txt; grep -q "self-test PASS" $W/s28_keylist.txt && [ -s $PKL ] || fail "key list builder"
DP=$(diff $B $PKL | grep '^[<>]' | grep -v -E '^[<>] (    \([0-9].*\),|_PB_SINGLE = \(.*|\))$' | wc -l | tr -d ' '); [ "$DP" = 0 ] || fail "key-list step changed lines outside the key table ($DP)"
python3 $H/build_setup_variants.py ksfirst $PKL $PRE $KSFR || fail "k-split leaf-buffers-first setup fix"
[ "$(diff $PKL $PRE | grep -c '^<')" = 0 ] || fail "ksfirst changed existing lines"; DESC="$DESC + k-split leaf buffers first (setup)"
printf '\n\n_PB_WALL_S = 4.0   # S28 (engineer2 + lead 11:5x): prebind guard 4.0 s (platform setup limit 5 s; the value-ordered list fits 4.0 s at 8 ms/key)\n' >> $PRE; DESC="$DESC + guard 4.0 s"
echo "$(date +%T) [8] + treuse LAST"; cat $PRE $TR > $C; DESC="$DESC + key list + treuse (last)"
E16=$(python3 $H/einsum_site_diff.py $C $S16F | tail -1); E23=$(python3 $H/einsum_site_diff.py $C $S23F | tail -1); E24=$(python3 $H/einsum_site_diff.py $C $S24F | tail -1); E27=$(python3 $H/einsum_site_diff.py $C $S27F | tail -1)
V=$(/Users/matt/arc-white-box-estimation/.venv/bin/whest validate --estimator $C 2>&1 | grep -o 'Validation passed.*\|Validation failed.*')
echo "einsum vs S16: $E16 | S23 ${E23:0:14} | S24 ${E24:0:14} | S27 ${E27:0:14} | $V"
for e in "$E16" "$E23" "$E24" "$E27"; do echo "$e" | grep -q "VERDICT PASS" || fail "einsum gate"; done
echo "$V" | grep -q "Validation passed" || fail "validate"
(cd $H && ./verify_literals.py $C $RP) | tee $W/S28_literals.txt; grep -q "ALL 28 BIT-EQUAL" $W/S28_literals.txt || fail "literals vs the refit .pt"
WS=$(grep -E '^_PB_WALL_S = ' $C | tail -1); BT=$(grep -E '^_PB_BATCHED = ' $C | tail -1); echo "k-split settings: $WS | $BT"
echo "$WS" | grep -qE '^_PB_WALL_S = 4\.0\b' && echo "$BT" | grep -qE '^_PB_BATCHED = \(\)' || fail "prebind settings not effective (_PB_WALL_S 4.0, _PB_BATCHED ())"
bash $H/machine_gate.sh 8 S28-treuse-identity
(cd $P2 && OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 10 $NPY fable_engineer_v1/e73_parity.py $PRE $C 35,100,37,39) 2>&1 | grep -v "Warning\|^  \|configure(" > $W/S28_treuse_identity.txt
python3 - $W/S28_treuse_identity.txt <<'PY' | tee $W/S28_treuse_identity_verdict.txt
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1]) if l.startswith('{')]
par = {r['rank']: r for r in rows if 'all16_identical' in r}; tr = {r['rank']: r['tr_stats'] for r in rows if 'tr_stats' in r}
ok = sorted(par) == [35, 37, 39, 100] and sorted(tr) == [35, 37, 39, 100] and all(r['all16_identical'] is True and r['max_abs_diff'] == 0.0 and r['flops_B'] < r['flops_A'] and tr[n]['reused'] > 0 for n, r in par.items())
print(('treuse identity PASS: ' if ok else 'treuse identity FAIL: ') + '; '.join(f"r{n} identical {par[n]['all16_identical']} bill {100 * (par[n]['flops_B'] / par[n]['flops_A'] - 1):+.3f}% reuses {tr.get(n, {}).get('reused')}" for n in sorted(par)))
PY
grep -q "treuse identity PASS" $W/S28_treuse_identity_verdict.txt || fail "treuse identity (stack/S28_treuse_identity.txt)"
(cd $H && OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 nice -n 10 $NPY ks_shape_check_v2.py $C 35) 2>&1 | grep -v "Warning\|^  \|configure(" > $W/S28_ks_coverage.txt; tail -1 $W/S28_ks_coverage.txt
echo "$(date '+%Y-%m-%d %H:%M') porter: S28 ($(shasum -a 256 $C | cut -c1-8)) built = $DESC; literals bit-equal to the refit .pt; k-split settings effective; einsum/validate + treuse identity PASS; $(tail -1 $W/S28_ks_coverage.txt); smoke next." >> $TL
echo "$(date +%T) [9] smoke with setup() (census v3, 35,100)"
bash $H/machine_gate.sh 7 S28-smoke
while ! mkdir /tmp/fable_paired.lock 2>/dev/null; do sleep 30; done
LOAD="$(sysctl -n vm.loadavg)"; bash $H/run_census_porter_v3.sh $C 35,100 cz_S28 > $W/cz_S28.out 2>&1
rmdir /tmp/fable_paired.lock
python3 - $H/cz_S28_census.json $W/cz_S28.out "$LOAD" $C > $W/smoke_S28.txt 2>&1 <<'PY'
import json, sys
rows = json.load(open(sys.argv[1])); assert [r['rank'] for r in rows] == [35, 100], rows
setup = [l for l in open(sys.argv[2]) if 'setup_wall_s' in l]; assert len(setup) == 1, setup
s = json.loads(setup[0].split(': ', 1)[1]); log = s['setup_log']
assert log and not any(str(e[0]).startswith('restored') for e in log), ('setup error (restored)', s)
bound = [e for e in log if e[0] == 'bound']; assert bound, ('no bound entry', s)
assert s['setup_wall_s'] <= 4.1, ('setup wall > 4.1 s', s)
import re
src = open(sys.argv[4]).read()
def table(name):
    i = src.rindex('\n' + name + ' = (') + 1; j0 = src.index('(', i); depth = 0
    for j in range(j0, len(src)):
        if src[j] == '(': depth += 1
        elif src[j] == ')':
            depth -= 1
            if depth == 0: break
    return eval(re.sub(r'#[^\n]*', '', src[j0:j + 1]))
intended = len(table('_PB_SINGLE')) + len(table('_PB_BATCHED')) + len(table('_KS_ACC_MN'))
fused = [e[1] for e in log if e[0] == 'lvfuse']
print(f"intended keys: {intended} (list {len(table('_PB_SINGLE'))} + batched {len(table('_PB_BATCHED'))} + k-split leaf buffers {len(table('_KS_ACC_MN'))}); bound {bound[-1][1]}; fused {fused}")
assert bound[-1][1] == intended, ('NOT all intended keys bound (lead 10:0x gate)', bound[-1][1], intended)
print(f"smoke setup(): wall {s['setup_wall_s']} s at load {sys.argv[3]}, requests {s['setup_requests']:,}, bound {bound[-1][1]} entries, log {[list(e) for e in log]}")
for r in rows:
    assert r['requests'] > 0 and r['flops'] > 0, r
    assert r.get('n_out') == 16 * 1024 and r.get('nonfinite') == 0, ('outputs', r.get('n_out'), r.get('nonfinite'))
    print(f"smoke r{r['rank']}: OK requests {r['requests']:,} server CPU {r['server_cpu_s']} s residual {r['residual_s']} s util {r['util']} outputs {r['n_out']} all finite")
PY
[ $? -eq 0 ] || { cat $W/smoke_S28.txt; tail -20 $W/cz_S28.out; fail "SMOKE FAILED (stack/smoke_S28.txt)"; }
cat $W/smoke_S28.txt
echo "$(date '+%Y-%m-%d %H:%M') porter: S28 smoke OK ($(grep -o 'wall [0-9.]* s at load [^,]*' $W/smoke_S28.txt | head -1); $(grep -o 'bound [0-9]* entries' $W/smoke_S28.txt | head -1)); VALID40 vs S27 rows STARTED (bar adj <= ${BAR}%, CI excl 0)." >> $TL
echo "$(date +%T) [10] VALID40 vs the S27 rows"
bash $H/machine_gate.sh 8 S28-VALID40
bash $H/run_valid40_locked.sh $C stack_S28_mem
python3 $H/valid40_vs_ref_ci.py $H/paired_valid40_stack_S28_mem.json $J27 > $H/gate_S28_valid40_vs_S27.txt; cat $H/gate_S28_valid40_vs_S27.txt
python3 $H/valid40_bar.py $H/paired_valid40_stack_S28_mem.json $J27 $BAR 0 | tee $H/gate_S28_bar.json
if ! grep -q '"verdict": "PASS"' $H/gate_S28_bar.json; then
  echo "$(date '+%Y-%m-%d %H:%M') porter: S28 NOT staged (VALID40 bar adj <= ${BAR}% vs S27, CI excl 0): $(sed -n 4p $H/gate_S28_valid40_vs_S27.txt | cut -c1-140) | raw $(sed -n 2p $H/gate_S28_valid40_vs_S27.txt | cut -c1-90)" >> $TL
  echo "$(date +%T) S28 CHAIN DONE (not staged)"; CHAIN_DONE=1; exit 0
fi
echo "$(date '+%Y-%m-%d %H:%M') porter: S28 PASSES the VALID40 bar vs S27: $(sed -n 4p $H/gate_S28_valid40_vs_S27.txt | cut -c1-150) | raw $(sed -n 2p $H/gate_S28_valid40_vs_S27.txt | cut -c1-90); residual gate next." >> $TL
echo "$(date +%T) [11] paired residual gate vs the S27 fire copy (setup client)"
bash $H/machine_gate.sh 13 S28-residual-gate
while ! mkdir /tmp/fable_paired.lock 2>/dev/null; do sleep 30; done
for rep in 1 2 3; do
  if [ $((rep % 2)) = 1 ]; then AA=$S27F; TA=rz28_S27_r$rep; BB=$C; TB=rz28_S28_r$rep; else AA=$C; TA=rz28_S28_r$rep; BB=$S27F; TB=rz28_S27_r$rep; fi
  bash $H/run_census_porter_v3.sh $AA 35,100,37 $TA > $W/$TA.out 2>&1 & a=$!
  sleep 1
  bash $H/run_census_porter_v3.sh $BB 35,100,37 $TB > $W/$TB.out 2>&1 & b=$!
  wait $a $b; echo "$(date +%T) residual rep $rep done"
done
rmdir /tmp/fable_paired.lock
python3 - $H $W <<'PY' | tee $W/residual_gate_S28.txt
import json, sys, statistics as st
H, W = sys.argv[1], sys.argv[2]
def load(tag): return {r['rank']: r for r in json.load(open(f'{H}/{tag}_census.json'))}
def setup_wall(tag):
    l = [x for x in open(f'{W}/{tag}.out') if 'setup_wall_s' in x]; s = json.loads(l[0].split(': ', 1)[1]); return s['setup_wall_s'], [e for e in s['setup_log'] if e[0] == 'bound'][-1][1]
R = {c: [load(f'rz28_{c}_r{k}') for k in (1, 2, 3)] for c in ('S27', 'S28')}
print('TAG fable-porter PT-62: paired residual gate S27 vs S28 (setup-calling client, concurrent pairs, 3 reps, nets 35 cold,100,37)')
cold = {}
for n in (35, 100, 37):
    a = [r[n]['residual_s'] for r in R['S27']]; b = [r[n]['residual_s'] for r in R['S28']]
    rq = (st.mean(r[n]['requests'] for r in R['S27']), st.mean(r[n]['requests'] for r in R['S28']))
    if n == 35: cold = dict(A=st.mean(a), B=st.mean(b))
    print(f"  r{n}{' (cold)' if n == 35 else ''}: S27 mean {st.mean(a):.3f} max {max(a):.3f} s | S28 mean {st.mean(b):.3f} max {max(b):.3f} s | delta {st.mean(b)-st.mean(a):+.3f} s | requests {rq[0]:,.0f} -> {rq[1]:,.0f}")
sw = [setup_wall(f'rz28_S28_r{k}') for k in (1, 2, 3)]; sw0 = [setup_wall(f'rz28_S27_r{k}') for k in (1, 2, 3)]
print(f"  setup under concurrent load: S28 walls {[w for w, _ in sw]} s, bound {[b for _, b in sw]} (gate <= 4.1 s: {'PASS' if max(w for w, _ in sw) <= 4.1 else 'FAIL'}) | S27 walls {[w for w, _ in sw0]} s, bound {[b for _, b in sw0]}")
d = cold['B'] - cold['A']
print(f"  S28 cold delta vs S27 {d:+.3f} s (rule <= +0.010 s) -> RESIDUAL GATE {'PASS' if d <= 0.010 else 'FLAG'}")
PY
echo "$(date '+%Y-%m-%d %H:%M') porter: S28 residual gate vs S27: $(tail -1 $W/residual_gate_S28.txt | sed 's/^ *//') | $(grep -o 'setup under concurrent load.*' $W/residual_gate_S28.txt | cut -c1-150)" >> $TL
grep -q 'RESIDUAL GATE' $W/residual_gate_S28.txt || fail "residual verdict missing (stack/residual_gate_S28.txt)"
grep -q 'gate <= 4.1 s: PASS' $W/residual_gate_S28.txt || fail "setup wall under concurrent load > 4.1 s"
echo "$(date +%T) [12] staging"
[ -e $FD ] && fail "fire dir exists; not touching it"
mkdir $FD; F=$FD/candS28_S27_refit_ks$( [ "$CPF" = none ] || echo _cap)$( [ "$RULE" = none ] || echo _sched)$( [ "$G5" = none ] || echo _g5)_mem.py; cp -p $C $F
{ echo "FIRE CHECKS S28 = $DESC, $(date '+%Y-%m-%d %H:%M')"
  echo "file $F"; echo "sha256 $(shasum -a 256 $F | cut -c1-64)"
  echo "inputs: $(basename $KSF) $KSS; cap ${CPF##*/} $CPS; refit params $RPS, dials $RDS $(python3 -c "import json,sys; d=json.load(open(sys.argv[1])); d=d.get('dials', d); print(json.dumps(d, separators=(',', ':'))[:300])" $RD); schedule $( [ "$RULE" = none ] && echo none || tr -d ' \n' < $RULE); group 5 ${G5##*/} $G5S (gated $G5F); key census $(basename $CEN) $(shasum -a 256 $CEN | cut -c1-16)"
  echo "key list: $(grep -v '^self-test' $W/s28_keylist.txt | tr '\n' ' ' | cut -c1-400)"
  echo "literals (fire copy): $(cd $H && ./verify_literals.py $F $RP) | validate: $(/Users/matt/arc-white-box-estimation/.venv/bin/whest validate --estimator $F 2>&1 | grep -o 'Validation passed.*\|Validation failed.*') | k-split settings: $WS | $BT"
  echo "einsum gate (fire copy) vs fired S16: $(python3 $H/einsum_site_diff.py $F $S16F | tail -1) | S23 $(python3 $H/einsum_site_diff.py $F $S23F | tail -1 | cut -c1-14) | S24 $(python3 $H/einsum_site_diff.py $F $S24F | tail -1 | cut -c1-14) | S27 $(python3 $H/einsum_site_diff.py $F $S27F | tail -1 | cut -c1-14)"
  echo "$(cat $W/S28_treuse_identity_verdict.txt)"
  echo "chunk-key coverage (net 35): $(tail -1 $W/S28_ks_coverage.txt)"
  echo "client/server smoke with setup(): $(grep -c 'OK requests' $W/smoke_S28.txt) of 2 nets OK; $(grep 'smoke setup' $W/smoke_S28.txt | cut -c1-300)"
  echo "VALID40 vs S27 rows: $(sed -n 2,4p $H/gate_S28_valid40_vs_S27.txt | tr '\n' ' ')"
  echo "bar (adjusted <= ${BAR}% vs S27, CI excl 0): $(cat $H/gate_S28_bar.json)"
  echo "residual gate vs S27 (setup-calling client): $(sed -n 2,6p $W/residual_gate_S28.txt | tr '\n' ' ')"
} > $FD/FIRE_CHECKS.txt 2>&1
cat $FD/FIRE_CHECKS.txt
echo "$(date '+%Y-%m-%d %H:%M') porter: S28 STAGED: fable_fire_v1/fire_candS28/$(basename $F) ($(shasum -a 256 $F | cut -c1-8); FIRE_CHECKS.txt; VALID40 $(sed -n 4p $H/gate_S28_valid40_vs_S27.txt | cut -c1-90); residual $(tail -1 $W/residual_gate_S28.txt | grep -o 'PASS\|FLAG'))." >> $TL
echo "$(date +%T) S28 CHAIN DONE (fire copy staged)"; CHAIN_DONE=1
