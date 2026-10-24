#!/usr/bin/env bash
# TAG fable-porter (PT-32, lead 18:2x "Queue S18 right behind S17p's staging"): S18 = S17p (stack_S17p_j05_prebind_mem.py 9a4ccc1a) + cat
# fable_engineer_v1/lrstar2_k48_q1_r32.py (E-62 low-rank star: Gaussian sketch k=48, 1 power step, rank 32; replaces _flms_cov_star, _flms_star,
# _flms_star_diagonal) = stack/stack_S18_S17p_lrq1r32_mem.py (ab8c8ba8). Order: wait for fire_chain_S17p.sh to finish; require fire_candS17 staged;
# PRE-GATE torch VALID40 star rank 32 on the j05 card (star_valid40.jsonl, 40 nets) pooled raw <= +0.8%; recipe rebuild sha; einsum; validate;
# smoke with setup() (35,100); VALID40 vs S17p rows, bar adjusted <= -3.0%, CI excl 0, >= 30/40, AND pooled raw within +-1.0%; paired residual
# gate vs S17p (setup-calling client, cold <= +0.010 s; FLAG but stage); stage fable_fire_v1/fire_candS18/ (FIRE_CHECKS quotes the engineer's
# wall pair log usercpu_pair_S18.log and identity gate log chain_S18gate.log if present; both are the engineer's gates).
# 18:28: relaunched (was idle in its first wait) to quote the engineer's S18 logs instead of the withdrawn usercpu_pair_lrq1r32.log.
set -u
P2=/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16; H=$P2/fable_porter_v1; TL=$P2/fable_refit_v1/TEAM_LOG.md
C=$H/stack/stack_S18_S17p_lrq1r32_mem.py; SHA=ab8c8ba866534491; B17=$H/stack/stack_S17p_j05_prebind_mem.py; S16F=$P2/fable_fire_v1/fire_candS16/candS16_S15_bb6_mem.py
J17=$H/paired_valid40_stack_S17p_mem.json; PH=$P2/fable_physicist_v1/refit6b; EN=$P2/fable_engineer_v1; LR=$EN/lrstar2_k48_q1_r32.py
[ "$(shasum -a 256 $C | cut -c1-16)" = $SHA ] || { echo "candidate sha changed"; exit 1; }
[ "$(shasum -a 256 $B17 | cut -c1-16)" = 9a4ccc1a845b1b6a ] || { echo "S17p base sha changed"; exit 1; }
until grep -q "S17p CHAIN DONE" $H/fire_chain_S17p.out 2>/dev/null; do sleep 30; done
if [ ! -s $P2/fable_fire_v1/fire_candS17/FIRE_CHECKS.txt ]; then
  echo "$(date '+%Y-%m-%d %H:%M') porter: S18 NOT started: S17p was not staged (see fire_chain_S17p.out)" >> $TL; echo "S17p not staged: S18 stops"; exit 0; fi
echo "$(date +%T) S17p staged; S18 pre-gate (torch star VALID40, rank 32 on the j05 card)"
until [ "$(grep -c '"card": "j05", "arm": "r32"' $H/stack/star_valid40.jsonl)" -ge 40 ]; do sleep 30; done
python3 $H/star_valid40_report.py $H/stack/star_valid40.jsonl > $H/stack/star_valid40_report_S18pregate.txt; cat $H/stack/star_valid40_report_S18pregate.txt
python3 - $H/stack/star_valid40.jsonl $PH/s6b_B2ep1_j05_1ep_valid40/eval_n40.json > $H/stack/S18_pregate.txt <<'PY'
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1])]; base = {int(k): v for k, v in json.load(open(sys.argv[2]))['valid'].items()}
R = {r['rank']: r['mse'] for r in rows if r['card'] == 'j05' and r['arm'] == 'r32'}; assert len(R) == 40, len(R)
pooled = sum(R.values()) / sum(base[k] for k in R) - 1
print(f"S18 pre-gate: torch star rank 32 on the j05 card, VALID40 pooled raw {100*pooled:+.3f}% (bar <= +0.8%) -> {'PASS' if pooled <= 0.008 else 'FAIL'}")
PY
cat $H/stack/S18_pregate.txt
if ! grep -q "PASS" $H/stack/S18_pregate.txt; then echo "$(date '+%Y-%m-%d %H:%M') porter: S18 STOPPED at the torch pre-gate: $(cat $H/stack/S18_pregate.txt)" >> $TL; exit 0; fi
R=$H/scratch_rank_order/repro_S18_$(date +%H%M%S); mkdir -p $R; cd $P2
bash fable_porter_v1/build_stack.sh $R/s.py --gn=fable_porter_v1/gn_knobs/v3.json --refit=$PH/s6b_B2ep1_j05_1ep/params_epoch0.pt --l15cut --join=v2 -- --holdelim --bigarena --depthrule fable_engineer_v1/depthrule_fragment_push5p6.py --f32 third --evict5 --caps --caps2 --batch --bindhold --coldbind3 --symout --lean > /dev/null
python3 fable_engineer2_v1/build_e2mem.py $R/s.py $R/s_mem.py mem,yright,arena,hold2 > /dev/null
python3 fable_engineer_v1/build_colpack.py $R/s_mem.py $R/a_cp.py > /dev/null
python3 fable_porter_v1/build_deadcut.py $R/a_cp.py $R/b_dc64.py --round=64 > /dev/null
python3 fable_engineer_v1/build_m64.py $R/b_dc64.py $R/c_m64.py > /dev/null
python3 fable_engineer2_v1/read6/build_read6.py $R/c_m64.py $R/d_r6.py q > /dev/null
python3 fable_engineer_v1/build_ipt_v3.py $R/d_r6.py $R/e_ipt.py > /dev/null
cat $R/e_ipt.py fable_engineer_v1/ipt_read6_fragment.py > $R/f.py
python3 fable_engineer2_v1/altbasis/build_altbasis.py $R/f.py $R/g.py > /dev/null
cat $R/g.py fable_engineer_v1/bb6_fragment.py fable_engineer_v1/resid_fragment_v3b.py > $R/h.py
python3 fable_porter_v1/build_j05.py $R/h.py $R/i.py > /dev/null
cat $R/i.py fable_engineer_v1/prebind_fragment_v2b.py $LR > $R/s_final.py
RS=$(shasum -a 256 $R/s_final.py | cut -c1-64); cd $H
echo "recipe rebuild sha $RS"
[ "$(echo $RS | cut -c1-16)" = $SHA ] || { echo "$(date '+%Y-%m-%d %H:%M') porter: S18 STOPPED: recipe rebuild sha $RS != candidate" >> $TL; exit 1; }
EG=$(python3 $H/einsum_site_diff.py $C $S16F | tail -1); VA=$(/Users/matt/arc-white-box-estimation/.venv/bin/whest validate --estimator $C 2>&1 | grep -o 'Validation passed.*\|Validation failed.*')
echo "einsum: $EG | validate: $VA"
echo "$EG" | grep -q "VERDICT PASS" && echo "$VA" | grep -q "passed" || { echo "$(date '+%Y-%m-%d %H:%M') porter: S18 STOPPED: einsum/validate: $EG | $VA" >> $TL; exit 1; }
while ! mkdir /tmp/fable_paired.lock 2>/dev/null; do sleep 30; done
echo "$(date +%T) smoke start (client/server with setup(), 35,100)"
bash $H/run_census_porter_v2.sh $C 35,100 cz_S18 > $H/stack/cz_S18.out 2>&1
rmdir /tmp/fable_paired.lock
python3 - $H/cz_S18_census.json $H/stack/cz_S18.out > $H/stack/smoke_S18.txt 2>&1 <<'PY'
import json, sys
rows = json.load(open(sys.argv[1])); assert [r['rank'] for r in rows] == [35, 100], rows
setup = [l for l in open(sys.argv[2]) if 'setup_wall_s' in l]; assert len(setup) == 1, setup
s = json.loads(setup[0].split(': ', 1)[1]); assert s['setup_log'] and s['setup_log'][0][0] == 'bound', s
print(f"smoke setup(): wall {s['setup_wall_s']} s, requests {s['setup_requests']:,}, log {s['setup_log']}")
for r in rows:
    assert r['requests'] > 0 and r['flops'] > 0, r
    print(f"smoke r{r['rank']}: OK requests {r['requests']:,} server CPU {r['server_cpu_s']} s residual {r['residual_s']} s util {r['util']}")
PY
if [ $? -ne 0 ]; then echo "SMOKE FAILED"; cat $H/stack/smoke_S18.txt; tail -20 $H/stack/cz_S18.out; echo "$(date '+%Y-%m-%d %H:%M') porter: S18 SMOKE FAILED (stack/cz_S18.out)" >> $TL; exit 1; fi
cat $H/stack/smoke_S18.txt
echo "$(date '+%Y-%m-%d %H:%M') porter: S18 (S17p + lrstar2 k48 q1 r32, ab8c8ba8) pre-gate PASS, smoke with setup() OK; VALID40 vs S17p rows STARTED." >> $TL
bash $H/run_valid40_locked.sh $C stack_S18_mem
python3 $H/valid40_vs_ref_ci.py $H/paired_valid40_stack_S18_mem.json $J17 > $H/gate_S18_valid40_vs_S17p.txt; cat $H/gate_S18_valid40_vs_S17p.txt
python3 $H/valid40_bar.py $H/paired_valid40_stack_S18_mem.json $J17 -3.0 30 | tee $H/gate_S18_bar.json
python3 - $H/paired_valid40_stack_S18_mem.json $J17 > $H/gate_S18_raw.txt <<'PY'
import json, sys
C, R = json.load(open(sys.argv[1]))['results'], json.load(open(sys.argv[2]))['results']
ks = [k for k in C if k.startswith('cand:') and C[k].get('mse') and R.get(k, {}).get('mse')]
raw = sum(C[k]['mse'] for k in ks) / sum(R[k]['mse'] for k in ks) - 1
print(f"S18 vs S17p pooled raw {100*raw:+.3f}% over {len(ks)} nets (bar within +-1.0%) -> {'PASS' if abs(raw) <= 0.010 else 'FAIL'}")
PY
cat $H/gate_S18_raw.txt
if ! grep -q '"verdict": "PASS"' $H/gate_S18_bar.json || ! grep -q "PASS" $H/gate_S18_raw.txt; then
  echo "$(date '+%Y-%m-%d %H:%M') porter: S18 NOT staged (VALID40 bar -3.0%/30 + raw +-1%): $(cut -c1-160 $H/gate_S18_bar.json) | $(cat $H/gate_S18_raw.txt)" >> $TL; echo "$(date +%T) S18 CHAIN DONE (not staged)"; exit 0
fi
while ! mkdir /tmp/fable_paired.lock 2>/dev/null; do sleep 30; done
echo "$(date +%T) residual gate S17p vs S18 (setup-calling client) start | load $(sysctl -n vm.loadavg)"
for rep in 1 2 3; do
  if [ $((rep % 2)) = 1 ]; then A=$B17; TA=rz18_S17p_r$rep; B=$C; TB=rz18_S18_r$rep; else A=$C; TA=rz18_S18_r$rep; B=$B17; TB=rz18_S17p_r$rep; fi
  bash $H/run_census_porter_v2.sh $A 35,100,37 $TA > $H/stack/$TA.out 2>&1 & a=$!
  sleep 1
  bash $H/run_census_porter_v2.sh $B 35,100,37 $TB > $H/stack/$TB.out 2>&1 & b=$!
  wait $a $b; echo "$(date +%T) residual rep $rep done"
done
rmdir /tmp/fable_paired.lock
python3 - $H <<'PY' | tee $H/stack/residual_gate_S18.txt
import json, sys, statistics as st
H = sys.argv[1]
def load(tag): return {r['rank']: r for r in json.load(open(f'{H}/{tag}_census.json'))}
R = {c: [load(f'rz18_{c}_r{k}') for k in (1, 2, 3)] for c in ('S17p', 'S18')}
print('TAG fable-porter PT-32: paired residual gate S17p vs S18 (setup-calling client, concurrent pairs, 3 reps, nets 35 cold,100,37)')
cold = {}
for n in (35, 100, 37):
    a = [r[n]['residual_s'] for r in R['S17p']]; b = [r[n]['residual_s'] for r in R['S18']]
    rq = (st.mean(r[n]['requests'] for r in R['S17p']), st.mean(r[n]['requests'] for r in R['S18']))
    cpu = (st.mean(r[n]['server_cpu_s'] for r in R['S17p']), st.mean(r[n]['server_cpu_s'] for r in R['S18']))
    if n == 35: cold = dict(A=st.mean(a), B=st.mean(b))
    print(f"  r{n}{' (cold)' if n == 35 else ''}: S17p mean {st.mean(a):.3f} max {max(a):.3f} s | S18 mean {st.mean(b):.3f} max {max(b):.3f} s | delta {st.mean(b)-st.mean(a):+.3f} s | requests {rq[0]:,.0f} -> {rq[1]:,.0f} | server CPU {cpu[0]:.1f} -> {cpu[1]:.1f} s")
d = cold['B'] - cold['A']
print(f"  S18 cold delta vs S17p {d:+.3f} s (rule <= +0.010 s) -> RESIDUAL GATE {'PASS' if d <= 0.010 else 'FLAG'}")
PY
echo "$(date '+%Y-%m-%d %H:%M') porter: S18 residual gate vs S17p: $(tail -1 $H/stack/residual_gate_S18.txt | sed 's/^ *//')" >> $TL
FD=$P2/fable_fire_v1/fire_candS18; [ -e $FD ] && { echo "fire dir exists; not touching it"; exit 1; }
mkdir $FD; F=$FD/candS18_S17_lrq1r32_mem.py; cp -p $C $F
{ echo "FIRE CHECKS S18 = S17 (fire_candS17, 9a4ccc1a) + engineer low-rank star E-62 lrstar2 k48 q1 r32 (APPROXIMATION; replaces _flms_cov_star/_flms_star/_flms_star_diagonal), $(date '+%Y-%m-%d %H:%M')"
  echo "file $F"; echo "sha256 $(shasum -a 256 $F | cut -c1-64) (porter stack_S18_S17p_lrq1r32_mem.py ab8c8ba8...; fragment lrstar2_k48_q1_r32.py $(shasum -a 256 $LR | cut -c1-16))"
  echo "validate: $(/Users/matt/arc-white-box-estimation/.venv/bin/whest validate --estimator $F 2>&1 | grep -o 'Validation passed.*\|Validation failed.*')"
  echo "einsum gate vs fired S16: $(python3 $H/einsum_site_diff.py $F $S16F | tail -1)"
  echo "diff vs S17 fire file: $(diff $P2/fable_fire_v1/fire_candS17/candS17_S16r3bp2b_j05_mem.py $F | grep -c '^>') appended lines (= the lrstar fragment, $(wc -l < $LR) lines), $(diff $P2/fable_fire_v1/fire_candS17/candS17_S16r3bp2b_j05_mem.py $F | grep -c '^<') removed"
  echo "recipe rebuild sha $RS (= S17 recipe + cat lrstar2_k48_q1_r32.py; must equal the fire file)"
  echo "torch pre-gate: $(cat $H/stack/S18_pregate.txt) | full report: $(sed -n 2,4p $H/stack/star_valid40_report_S18pregate.txt | tr '\n' ' ')"
  echo "bill census (bill_by_function_v2, r100 cold, lrstar on S16r3b = the same kernel change): 338.51 GF vs S16 358.49 GF (-5.57%): cov+star 72.25 -> 46.05 GF, _lr_factor +6.75 GF, L15 star diagonal 0.87 -> 0.10 GF (stack/billfn2_S16r3b_lrq1r32_r100.txt)"
  echo "client/server smoke with setup(): $(grep -c 'OK requests' $H/stack/smoke_S18.txt) of 2 nets OK; $(grep 'smoke setup' $H/stack/smoke_S18.txt); $(grep 'smoke r' $H/stack/smoke_S18.txt | tr '\n' ' ')"
  echo "VALID40 vs S17p rows: $(sed -n 2,4p $H/gate_S18_valid40_vs_S17p.txt | tr '\n' ' ')"
  echo "bar (adjusted <= -3.0%, CI excl 0, >= 30/40): $(cat $H/gate_S18_bar.json) ; $(cat $H/gate_S18_raw.txt)"
  echo "residual gate vs S17p (setup-calling client): $(sed -n 2,5p $H/stack/residual_gate_S18.txt | tr '\n' ' ')"
  echo "engineer wall pair (the engineer's gate; e_S18lr vs e_S17pref, nets 35,100,37,39,42,48): $( [ -s $EN/usercpu_pair_S18.log ] && tail -4 $EN/usercpu_pair_S18.log | tr '\n' ' ' || echo 'not yet reported in fable_engineer_v1/usercpu_pair_S18.log')"
  echo "engineer identity gate (S18 client/server with setup() vs in-process candS17j05_k48_q1_r32_mem.py on 35/100): $( [ -s $EN/chain_S18gate.log ] && tail -4 $EN/chain_S18gate.log | tr '\n' ' ' || echo 'not yet reported in fable_engineer_v1/chain_S18gate.log')"
} > $FD/FIRE_CHECKS.txt 2>&1
cat $FD/FIRE_CHECKS.txt
echo "$(date '+%Y-%m-%d %H:%M') porter: S18 (S17 + low-rank star k48 q1 r32) PASSES the VALID40 bar; fire copy staged: fable_fire_v1/fire_candS18/candS18_S17_lrq1r32_mem.py (FIRE_CHECKS.txt; residual $(tail -1 $H/stack/residual_gate_S18.txt | grep -o 'PASS\|FLAG'); engineer wall pair pending unless quoted)." >> $TL
echo "$(date +%T) S18 CHAIN DONE (fire copy staged)"
