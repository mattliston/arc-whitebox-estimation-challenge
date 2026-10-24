#!/usr/bin/env bash
# TAG fable-porter (PT-45c, lead 22:3x "S23 ... launch after my GO"): S23 = stack/stack_S23_mem.py (9247a212; prebuild_S23.sh: S21s1 recipe + loopR1
# + round-1 dials + d6x + s1 + s1=0 + Nystrom join; partials byte-rebuild 30aea66e / a49da660 / 23c3abc2). Lock order (lead): S22, engineer2's s1=0
# VALID40, S21jn residual, THEN S23 -> each heavy step first waits while those run, then machine_gate.sh. Smoke (setup, v2 rule, finiteness) ->
# VALID40 vs the S21s1 rows (paired_valid40_e2_S21s1lr.json), bar adjusted <= BAR (arg 1, default -3.0), CI excl 0 -> residual vs S21s1 (setup
# client) -> stage fable_fire_v1/fire_candS23/. EXIT trap logs a death to TEAM_LOG.   usage: fire_chain_S23.sh [BAR]
set -u
P2=/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16; H=$P2/fable_porter_v1; TL=$P2/fable_refit_v1/TEAM_LOG.md
BAR=${1:--3.0}; C=$H/stack/stack_S23_mem.py; SHA=$(shasum -a 256 $C | cut -c1-64); [ "${SHA:0:16}" = 9247a212c5a74212 ] || { echo "S23 sha changed"; exit 1; }
S21S1=$P2/fable_fire_v1/fire_candS21s1/candS21s1_S21_s1lr_mem.py; J21S1=$H/paired_valid40_e2_S21s1lr.json; J21=$H/paired_valid40_stack_S21_round1_mem.json
S16F=$P2/fable_fire_v1/fire_candS16/candS16_S15_bb6_mem.py; LR1=$P2/fable_physicist_v1/refit6b/s6b_loopR1/params_epoch0.pt
CHAIN_DONE=0; trap 'rc=$?; [ $CHAIN_DONE = 1 ] || echo "$(date "+%Y-%m-%d %H:%M") porter: S23 CHAIN DIED rc=$rc (fable_porter_v1/fire_chain_S23.out) last: $(tail -2 $H/fire_chain_S23.out 2>/dev/null | tr "\n" " " | cut -c1-200)" >> $TL' EXIT
E2OUT=$P2/fable_engineer2_v1/lrroute/run_after_S22.out; E2BAR=$P2/fable_engineer2_v1/lrroute/valid40_e2_S21s1z_bar.json
e2_valid40_pending() { pgrep -f "run_after_S2[2]" >/dev/null && ! grep -q VALID40_DONE $E2OUT 2>/dev/null; }
yield_all() { while pgrep -f "bash fire_chain_S2[2]" >/dev/null || pgrep -f "bash fire_chain_S21j[n].sh" >/dev/null || e2_valid40_pending; do sleep 60; done; bash $H/machine_gate.sh $1 $2; }   # lead 22:4x order: S22, then engineer2 s1=0 VALID40, then S23
yield_all 7 S23-smoke
while ! mkdir /tmp/fable_paired.lock 2>/dev/null; do sleep 30; done
echo "$(date +%T) smoke start (client/server with setup(), 35,100)"
bash $H/run_census_porter_v3.sh $C 35,100 cz_S23 > $H/stack/cz_S23.out 2>&1
rmdir /tmp/fable_paired.lock
python3 - $H/cz_S23_census.json $H/stack/cz_S23.out > $H/stack/smoke_S23.txt 2>&1 <<'PY'
import json, sys
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
if [ $? -ne 0 ]; then echo "SMOKE FAILED"; cat $H/stack/smoke_S23.txt; tail -20 $H/stack/cz_S23.out; echo "$(date '+%Y-%m-%d %H:%M') porter: S23 SMOKE FAILED" >> $TL; CHAIN_DONE=1; exit 1; fi
cat $H/stack/smoke_S23.txt
echo "$(date '+%Y-%m-%d %H:%M') porter: S23 (${SHA:0:8}) smoke OK; VALID40 vs S21s1 rows STARTED (bar adj <= ${BAR}%, CI excl 0)." >> $TL
if ! grep -q '"verdict": "PASS"' $E2BAR 2>/dev/null; then   # lead: if engineer2's s1=0 VALID40 fails its bar, hold before S23's VALID40
  echo "$(date +%T) HOLD: engineer2 s1=0 VALID40 not PASS ($(cut -c1-160 $E2BAR 2>/dev/null || echo 'bar file missing')); waiting for $H/stack/S23_go.flag (GO|ABORT)"
  echo "$(date '+%Y-%m-%d %H:%M') porter: S23 HOLDING before its VALID40: engineer2 s1=0 VALID40 not PASS; lead decides (stack/S23_go.flag GO|ABORT)" >> $TL
  until [ -s $H/stack/S23_go.flag ] && grep -qxE 'GO|ABORT' $H/stack/S23_go.flag; do sleep 30; done
  grep -qx ABORT $H/stack/S23_go.flag && { echo "$(date +%T) S23 ABORTED by the lead"; CHAIN_DONE=1; exit 0; }
fi
yield_all 8 S23-VALID40
bash $H/run_valid40_locked.sh $C stack_S23_mem
python3 $H/valid40_vs_ref_ci.py $H/paired_valid40_stack_S23_mem.json $J21S1 > $H/gate_S23_valid40_vs_S21s1.txt; cat $H/gate_S23_valid40_vs_S21s1.txt
python3 $H/valid40_bar.py $H/paired_valid40_stack_S23_mem.json $J21S1 $BAR 0 | tee $H/gate_S23_bar.json
python3 $H/valid40_vs_ref_ci.py $H/paired_valid40_stack_S23_mem.json $J21 > $H/gate_S23_vs_S21_info.txt
if ! grep -q '"verdict": "PASS"' $H/gate_S23_bar.json; then
  echo "$(date '+%Y-%m-%d %H:%M') porter: S23 NOT staged (VALID40 bar adj <= ${BAR}% vs S21s1, CI excl 0): $(cut -c1-160 $H/gate_S23_bar.json)" >> $TL; echo "$(date +%T) S23 CHAIN DONE (not staged)"; CHAIN_DONE=1; exit 0
fi
yield_all 13 S23-residual-gate
while ! mkdir /tmp/fable_paired.lock 2>/dev/null; do sleep 30; done
echo "$(date +%T) residual gate S21s1 vs S23 (setup-calling client) start | load $(sysctl -n vm.loadavg)"
for rep in 1 2 3; do
  if [ $((rep % 2)) = 1 ]; then A=$S21S1; TA=rz23_S21s1_r$rep; B=$C; TB=rz23_S23_r$rep; else A=$C; TA=rz23_S23_r$rep; B=$S21S1; TB=rz23_S21s1_r$rep; fi
  bash $H/run_census_porter_v3.sh $A 35,100,37 $TA > $H/stack/$TA.out 2>&1 & a=$!
  sleep 1
  bash $H/run_census_porter_v3.sh $B 35,100,37 $TB > $H/stack/$TB.out 2>&1 & b=$!
  wait $a $b; echo "$(date +%T) residual rep $rep done"
done
rmdir /tmp/fable_paired.lock
python3 - $H <<'PY' | tee $H/stack/residual_gate_S23.txt
import json, sys, statistics as st
H = sys.argv[1]
def load(tag): return {r['rank']: r for r in json.load(open(f'{H}/{tag}_census.json'))}
R = {c: [load(f'rz23_{c}_r{k}') for k in (1, 2, 3)] for c in ('S21s1', 'S23')}
print('TAG fable-porter PT-45c: paired residual gate S21s1 vs S23 (setup-calling client, concurrent pairs, 3 reps, nets 35 cold,100,37)')
cold = {}
for n in (35, 100, 37):
    a = [r[n]['residual_s'] for r in R['S21s1']]; b = [r[n]['residual_s'] for r in R['S23']]
    rq = (st.mean(r[n]['requests'] for r in R['S21s1']), st.mean(r[n]['requests'] for r in R['S23']))
    if n == 35: cold = dict(A=st.mean(a), B=st.mean(b))
    print(f"  r{n}{' (cold)' if n == 35 else ''}: S21s1 mean {st.mean(a):.3f} max {max(a):.3f} s | S23 mean {st.mean(b):.3f} max {max(b):.3f} s | delta {st.mean(b)-st.mean(a):+.3f} s | requests {rq[0]:,.0f} -> {rq[1]:,.0f}")
d = cold['B'] - cold['A']
print(f"  S23 cold delta vs S21s1 {d:+.3f} s (rule <= +0.010 s) -> RESIDUAL GATE {'PASS' if d <= 0.010 else 'FLAG'}")
PY
echo "$(date '+%Y-%m-%d %H:%M') porter: S23 residual gate vs S21s1: $(tail -1 $H/stack/residual_gate_S23.txt | sed 's/^ *//')" >> $TL
FD=$P2/fable_fire_v1/fire_candS23; [ -e $FD ] && { echo "fire dir exists; not touching it"; CHAIN_DONE=1; exit 1; }
mkdir $FD; F=$FD/candS23_S21s1_loopR1_s1z_jn_mem.py; cp -p $C $F
{ echo "FIRE CHECKS S23 = S21s1 (#332236, 30aea66e) recipe + physicist loopR1 params + round-1 dials + d6x + engineer2 s1 + s1=0 + porter Nystrom join, $(date '+%Y-%m-%d %H:%M')"
  echo "file $F"; echo "sha256 $(shasum -a 256 $F | cut -c1-64)"
  echo "prebuild (prebuild_S23.sh; partial byte-rebuilds + composition checks): $(grep -v '^$' $H/stack/prebuild_S23_record.txt | tr '\n' ' ' | cut -c1-1400)"
  echo "validate (fire copy): $(/Users/matt/arc-white-box-estimation/.venv/bin/whest validate --estimator $F 2>&1 | grep -o 'Validation passed.*\|Validation failed.*')"
  echo "einsum gate vs fired S16 (fire copy): $(python3 $H/einsum_site_diff.py $F $S16F | tail -1)"
  echo "literals (fire copy) vs loopR1 params: $(cd $H && ./verify_literals.py $F $LR1)"
  echo "in-process vs S21s1 (r35/r100): $(grep '^r' $H/stack/S23_vs_S21s1_inprocess.txt | tr '\n' ' ')"
  echo "client/server smoke with setup(): $(grep -c 'OK requests' $H/stack/smoke_S23.txt) of 2 nets OK; $(grep 'smoke setup' $H/stack/smoke_S23.txt)"
  echo "VALID40 vs S21s1 rows: $(sed -n 2,4p $H/gate_S23_valid40_vs_S21s1.txt | tr '\n' ' ')"
  echo "bar (adjusted <= ${BAR}% vs S21s1, CI excl 0): $(cat $H/gate_S23_bar.json)"
  echo "VALID40 vs S21 rows (informational): $(sed -n 2,4p $H/gate_S23_vs_S21_info.txt | tr '\n' ' ')"
  echo "residual gate vs S21s1 (setup-calling client): $(sed -n 2,5p $H/stack/residual_gate_S23.txt | tr '\n' ' ')"
  echo "TORCH MIRROR for refits: the Nystrom join = fable_analyst_v1/s2card.py make_push join block with Omega = E_rows (join_sketch_test.py arm 'C'); s1=0 = the star's s1 term dropped in _flms_cov_star"
} > $FD/FIRE_CHECKS.txt 2>&1
cat $FD/FIRE_CHECKS.txt
echo "$(date '+%Y-%m-%d %H:%M') porter: S23 PASSES the VALID40 bar vs S21s1 (${BAR}%); fire copy staged: fable_fire_v1/fire_candS23/candS23_S21s1_loopR1_s1z_jn_mem.py (FIRE_CHECKS.txt; residual $(tail -1 $H/stack/residual_gate_S23.txt | grep -o 'PASS\|FLAG'))." >> $TL
echo "$(date +%T) S23 CHAIN DONE (fire copy staged)"; CHAIN_DONE=1
