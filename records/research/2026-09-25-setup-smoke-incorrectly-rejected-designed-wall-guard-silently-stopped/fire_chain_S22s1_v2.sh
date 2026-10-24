#!/usr/bin/env bash
# TAG fable-porter (PT-39d, lead 21:5x "S22 may need the s1 fragment layered"): S22s1 = S21 + engineer2 s1 (s1lr_fragment_v2.py 67bc7260, the
# fired-or-staged S21+s1 = fable_engineer2_v1/lrroute/S21_s1lr_mem.py 30aea66e) + ROUND-2 params + ROUND-2 dials: prebuild_dials_stack.sh <pt>
# <dials> <tag> <s1 fragment> (EXTRA appended after d6x; (ep2 + round-1 dials + s1) reproduces 30aea66e byte-exactly). Gates: as fire_chain_S22.sh
# but VALID40 vs the S21+s1 rows (paired_valid40_e2_S21s1lr.json, waited until complete), bar adjusted <= -0.8% CI excl 0; residual vs S21+s1.
# usage: fire_chain_S22s1.sh <params.pt> <dials.json> <tag>
# 22:1x (lead): round-2 dials FAILED -> S22 = S21s1 + loopR1 params + ROUND-1 dials; bar adjusted <= -0.5% vs S21+s1 rows, CI excl 0; byte-check:
# (ep2 + round-1 + s1) must rebuild 30aea66e; parity (informational) vs the physicist's loopR1 VALID16 rows (stack/loopR1_physicist_rows.json).
set -u
# v2 (22:2x, lead): smoke accepts a prebind WALL-GUARD stop (no 'restored' entry, 'bound' present, setup_wall_s <= 3.1 s, every net OK with
# n_out = 16384 finite outputs; bound count recorded); run_census_porter_v3.sh (records n_out/nonfinite); EXIT trap logs a chain death to TEAM_LOG.
P2=/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16; H=$P2/fable_porter_v1; TL=$P2/fable_refit_v1/TEAM_LOG.md
PT=$1; DJ=$2; TAG=$3; S21=$P2/fable_engineer2_v1/lrroute/S21_s1lr_mem.py; J21=$H/paired_valid40_e2_S21s1lr.json; J21PLAIN=$H/paired_valid40_stack_S21_round1_mem.json
S1F=$P2/fable_engineer2_v1/lrroute/s1lr_fragment_v2.py; [ "$(shasum -a 256 $S1F | cut -c1-16)" = 67bc726048c50b52 ] || { echo "s1 fragment sha changed"; exit 1; }
S16F=$P2/fable_fire_v1/fire_candS16/candS16_S15_bb6_mem.py
CHAIN_DONE=0; trap 'rc=$?; [ $CHAIN_DONE = 1 ] || echo "$(date "+%Y-%m-%d %H:%M") porter: S22 CHAIN DIED rc=$rc (fable_porter_v1/fire_chain_S22_v2.out) last: $(tail -2 $H/fire_chain_S22_v2.out 2>/dev/null | tr "\n" " " | cut -c1-200)" >> $TL' EXIT
[ "$(shasum -a 256 $S21 | cut -c1-16)" = 30aea66ef5afa24a ] || { echo "S21+s1 file sha changed"; exit 1; }
until python3 -c "import json,sys; d=json.load(open('$J21')); r=d['results']; sys.exit(0 if sum(1 for k,v in r.items() if k.startswith('cand:') and v.get('mse')) >= 40 else 1)" 2>/dev/null; do sleep 30; done   # engineer2's S21+s1 VALID40 complete
bash $H/prebuild_dials_stack.sh $P2/fable_physicist_v1/refit6b/s6b_j05_ep2/params_epoch0.pt $H/stack/s21_dials_used_round1.json S22s1byte_$TAG $S1F > $H/stack/prebuild_S22s1byte_$TAG.txt 2>&1
BC=$(shasum -a 256 $H/stack/stack_S22s1byte_${TAG}_mem.py | cut -c1-64); echo "byte-check (ep2 + round-1 dials + s1 -> must be 30aea66e): $BC"
[ "${BC:0:16}" = 30aea66ef5afa24a ] || { echo "$(date '+%Y-%m-%d %H:%M') porter: S22s1 STOPPED: byte-check $BC != 30aea66e" >> $TL; exit 1; }
cp $DJ $H/stack/s22s1_dials_used_$TAG.json
bash $H/prebuild_dials_stack.sh $PT $H/stack/s22s1_dials_used_$TAG.json S22s1_$TAG $S1F > $H/stack/prebuild_S22s1_$TAG.txt 2>&1; cat $H/stack/prebuild_S22s1_$TAG.txt
C=$H/stack/stack_S22s1_${TAG}_mem.py; SHA=$(shasum -a 256 $C | cut -c1-64)
[ "$(grep -c 'ALL 28 BIT-EQUAL' $H/stack/prebuild_S22s1_$TAG.txt)" = 2 ] && grep -q 'VERDICT PASS' $H/stack/prebuild_S22s1_$TAG.txt && grep -q 'Validation passed' $H/stack/prebuild_S22s1_$TAG.txt \
  || { echo "$(date '+%Y-%m-%d %H:%M') porter: S22s1 ($TAG) STOPPED at prebuild checks (stack/prebuild_S22s1_$TAG.txt)" >> $TL; exit 1; }
bash $H/machine_gate.sh 7 S22s1-smoke   # lead 21:5x machine rule: < 4 heavy python, free+inactive >= peak + 4 GB, memory free >= 35%
while ! mkdir /tmp/fable_paired.lock 2>/dev/null; do sleep 30; done
echo "$(date +%T) smoke start (client/server with setup(), 35,100)"
bash $H/run_census_porter_v3.sh $C 35,100 cz_S22s1_$TAG > $H/stack/cz_S22s1_$TAG.out 2>&1
rmdir /tmp/fable_paired.lock
python3 - $H/cz_S22s1_${TAG}_census.json $H/stack/cz_S22s1_$TAG.out > $H/stack/smoke_S22s1_$TAG.txt 2>&1 <<'PY'
import json, sys, math
rows = json.load(open(sys.argv[1])); assert [r['rank'] for r in rows] == [35, 100], rows
setup = [l for l in open(sys.argv[2]) if 'setup_wall_s' in l]; assert len(setup) == 1, setup
s = json.loads(setup[0].split(': ', 1)[1]); log = s['setup_log']
assert log and not any(str(e[0]).startswith('restored') for e in log), ('setup error (restored)', s)
bound = [e for e in log if e[0] == 'bound']; assert bound, ('no bound entry', s)
assert s['setup_wall_s'] <= 3.1, ('setup wall > 3.1 s', s)
guard = [e for e in log if e[0] == 'wall guard']
print(f"smoke setup(): wall {s['setup_wall_s']} s, requests {s['setup_requests']:,}, bound {bound[-1][1]} entries" + (f" (WALL GUARD held at {guard[0][1]} of 362 under load; accepted per lead 22:2x)" if guard else '') + f", log {log}")
for r in rows:
    assert r['requests'] > 0 and r['flops'] > 0, r
    assert r.get('n_out') == 16 * 1024 and r.get('nonfinite') == 0, ('outputs', r.get('n_out'), r.get('nonfinite'))
    print(f"smoke r{r['rank']}: OK requests {r['requests']:,} server CPU {r['server_cpu_s']} s residual {r['residual_s']} s util {r['util']} outputs {r['n_out']} all finite")
PY
if [ $? -ne 0 ]; then echo "SMOKE FAILED"; cat $H/stack/smoke_S22s1_$TAG.txt; tail -20 $H/stack/cz_S22s1_$TAG.out; echo "$(date '+%Y-%m-%d %H:%M') porter: S22s1 ($TAG) SMOKE FAILED" >> $TL; exit 1; fi
cat $H/stack/smoke_S22s1_$TAG.txt
echo "$(date '+%Y-%m-%d %H:%M') porter: S22s1 ($TAG, ${SHA:0:8}: S21+s1 recipe + round-2 params + round-2 dials) smoke OK; VALID40 vs S21+s1 rows STARTED." >> $TL
bash $H/machine_gate.sh 8 S22s1-VALID40
bash $H/run_valid40_locked.sh $C stack_S22s1_${TAG}_mem
python3 $H/valid40_vs_ref_ci.py $H/paired_valid40_stack_S22s1_${TAG}_mem.json $J21 > $H/gate_S22s1_${TAG}_valid40_vs_S21.txt; cat $H/gate_S22s1_${TAG}_valid40_vs_S21.txt
python3 $H/valid40_bar.py $H/paired_valid40_stack_S22s1_${TAG}_mem.json $J21 -0.5 0 | tee $H/gate_S22s1_${TAG}_bar.json
python3 $H/valid40_vs_ref_ci.py $H/paired_valid40_stack_S22s1_${TAG}_mem.json $J21PLAIN > $H/gate_S22s1_${TAG}_vs_S20_info.txt   # informational: vs plain S21 rows
if ! grep -q '"verdict": "PASS"' $H/gate_S22s1_${TAG}_bar.json; then
  echo "$(date '+%Y-%m-%d %H:%M') porter: S22s1 ($TAG) NOT staged (VALID40 bar -0.5% vs S21+s1): $(cut -c1-160 $H/gate_S22s1_${TAG}_bar.json)" >> $TL; echo "$(date +%T) S22s1 CHAIN DONE (not staged)"; CHAIN_DONE=1; exit 0
fi
bash $H/machine_gate.sh 13 S22s1-residual-gate
while ! mkdir /tmp/fable_paired.lock 2>/dev/null; do sleep 30; done
echo "$(date +%T) residual gate S21+s1 vs S22s1 (setup-calling client) start | load $(sysctl -n vm.loadavg)"
for rep in 1 2 3; do
  if [ $((rep % 2)) = 1 ]; then A=$S21; TA=rz22s1${TAG}_S21_r$rep; B=$C; TB=rz22s1${TAG}_S22_r$rep; else A=$C; TA=rz22s1${TAG}_S22_r$rep; B=$S21; TB=rz22s1${TAG}_S21_r$rep; fi
  bash $H/run_census_porter_v3.sh $A 35,100,37 $TA > $H/stack/$TA.out 2>&1 & a=$!
  sleep 1
  bash $H/run_census_porter_v3.sh $B 35,100,37 $TB > $H/stack/$TB.out 2>&1 & b=$!
  wait $a $b; echo "$(date +%T) residual rep $rep done"
done
rmdir /tmp/fable_paired.lock
python3 - $H $TAG <<'PY' | tee $H/stack/residual_gate_S22s1_$TAG.txt
import json, sys, statistics as st
H, T = sys.argv[1], sys.argv[2]
def load(tag): return {r['rank']: r for r in json.load(open(f'{H}/{tag}_census.json'))}
R = {c: [load(f'rz22{T}_{c}_r{k}') for k in (1, 2, 3)] for c in ('S21', 'S22')}
print(f'TAG fable-porter PT-39b: paired residual gate S21+s1 vs S22s1 ({T}; setup-calling client, concurrent pairs, 3 reps, nets 35 cold,100,37)')
cold = {}
for n in (35, 100, 37):
    a = [r[n]['residual_s'] for r in R['S21']]; b = [r[n]['residual_s'] for r in R['S22']]
    rq = (st.mean(r[n]['requests'] for r in R['S21']), st.mean(r[n]['requests'] for r in R['S22']))
    if n == 35: cold = dict(A=st.mean(a), B=st.mean(b))
    print(f"  r{n}{' (cold)' if n == 35 else ''}: S21 mean {st.mean(a):.3f} max {max(a):.3f} s | S22 mean {st.mean(b):.3f} max {max(b):.3f} s | delta {st.mean(b)-st.mean(a):+.3f} s | requests {rq[0]:,.0f} -> {rq[1]:,.0f}")
d = cold['B'] - cold['A']
print(f"  S22 cold delta vs S21 {d:+.3f} s (rule <= +0.010 s) -> RESIDUAL GATE {'PASS' if d <= 0.010 else 'FLAG'}")
PY
echo "$(date '+%Y-%m-%d %H:%M') porter: S22s1 ($TAG) residual gate vs S21+s1: $(tail -1 $H/stack/residual_gate_S22s1_$TAG.txt | sed 's/^ *//')" >> $TL
RW=$H/stack/loopR1_physicist_rows.json   # physicist loopR1 VALID16 (base = ep2 + round-1, fit = loopR1); informational
python3 $H/s19_parity_vs_torch.py $H/paired_valid40_stack_S22s1_${TAG}_mem.json $J21 $RW > $H/gate_S22s1_${TAG}_native_vs_torch.txt 2>&1; cat $H/gate_S22s1_${TAG}_native_vs_torch.txt
FD=$P2/fable_fire_v1/fire_candS22; [ -e $FD ] && { echo "fire dir exists; not touching it"; exit 1; }
mkdir $FD; F=$FD/candS22_S21s1_r2_mem.py; cp -p $C $F
{ echo "FIRE CHECKS S22 = S21s1 (#332236, 30aea66e) recipe with the physicist loopR1 params ($PT, sha $(shasum -a 256 $PT | cut -c1-16)) + covhunter ROUND-1 dials (round-2 fit failed; --gn=shared; stack/s22s1_dials_used_$TAG.json) + engineer2 d6x, $(date '+%Y-%m-%d %H:%M')"
  echo "file $F"; echo "sha256 $(shasum -a 256 $F | cut -c1-64)"
  echo "dials: $(tr -d '\n ' < $H/stack/s22s1_dials_used_$TAG.json | cut -c1-400)"
  echo "byte-check: $(grep -o "byte-check.*" $H/fire_chain_S22_v2.out 2>/dev/null | head -1)"
  echo "prebuild (= the recipe: S18 recipe --refit=<round-2 .pt> -> build_s19_dials --gn=shared -> cat d6x_fragment_v2; prebuild_dials_stack.sh reproduces the fired S21 byte-exactly): $(grep -v '^$' $H/stack/prebuild_S22s1_$TAG.txt | tr '\n' ' ' | cut -c1-900)"
  echo "validate (fire copy): $(/Users/matt/arc-white-box-estimation/.venv/bin/whest validate --estimator $F 2>&1 | grep -o 'Validation passed.*\|Validation failed.*')"
  echo "einsum gate vs fired S16 (fire copy): $(python3 $H/einsum_site_diff.py $F $S16F | tail -1)"
  echo "client/server smoke with setup(): $(grep -c 'OK requests' $H/stack/smoke_S22s1_$TAG.txt) of 2 nets OK; $(grep 'smoke setup' $H/stack/smoke_S22s1_$TAG.txt)"
  echo "VALID40 vs S21+s1 rows (paired_valid40_e2_S21s1lr.json): $(sed -n 2,4p $H/gate_S22s1_${TAG}_valid40_vs_S21.txt | tr '\n' ' ')"
  echo "bar (adjusted <= -0.5% vs S21+s1, CI excl 0): $(cat $H/gate_S22s1_${TAG}_bar.json)"
  echo "VALID40 vs plain S21 rows (informational): $(sed -n 2,4p $H/gate_S22s1_${TAG}_vs_S20_info.txt 2>/dev/null | tr '\n' ' ')"
  echo "native vs torch (physicist loopR1 VALID16 rows, informational): $(sed -n 2,3p $H/gate_S22s1_${TAG}_native_vs_torch.txt | tr '\n' ' ')"
  echo "residual gate vs S21+s1 (setup-calling client): $(sed -n 2,5p $H/stack/residual_gate_S22s1_$TAG.txt | tr '\n' ' ')"
  echo "engineer2 d6x gates: $(grep -v '^$' $P2/fable_engineer2_v1/d6x/gates_S18d6x.out | tail -4 | tr '\n' ' ' | cut -c1-500)"
} > $FD/FIRE_CHECKS.txt 2>&1
cat $FD/FIRE_CHECKS.txt
echo "$(date '+%Y-%m-%d %H:%M') porter: S22s1 ($TAG) PASSES the VALID40 bar vs S21+s1 (-0.5%); fire copy staged: fable_fire_v1/fire_candS22/candS22_S21s1_r2_mem.py (FIRE_CHECKS.txt; residual $(tail -1 $H/stack/residual_gate_S22s1_$TAG.txt | grep -o 'PASS\|FLAG'))." >> $TL
echo "$(date +%T) S22s1 CHAIN DONE (fire copy staged)"; CHAIN_DONE=1
