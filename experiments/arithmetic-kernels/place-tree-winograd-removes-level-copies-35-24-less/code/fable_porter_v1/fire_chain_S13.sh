#!/usr/bin/env bash
# TAG fable-porter (PT-19, lead 12:48: "the porter runs S13's VALID40 (identity vs S12 rows on all 40 nets + bar), stages fire_candS13, the lead
# fires"). S13 = S12 (c20d67c7) + engineer --ipt v3 (build_ipt_v3.py, post-stage LAST) = stack/stack_S13_ipt3_mem.py (2a5dd324 = engineer's
# candS12ipt3_mem.py). Steps: client/server smoke (35,100) -> VALID40 vs S11 rows -> per-net MSE identity vs S12's VALID40 rows -> bar
# (valid40_bar.py -0.8 26 = S12's relaxed bar; -1.0 verdict and the 3e-9 flag reported too) -> on PASS + identity, stage fable_fire_v1/fire_candS13/.
set -u
P2=/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16; H=$P2/fable_porter_v1; TL=$P2/fable_refit_v1/TEAM_LOG.md
C=$H/stack/stack_S13_ipt3_mem.py; SHA=2a5dd324d78c4db6
S11=$P2/fable_fire_v1/fire_candS11/candS11_S10L_B2e1_mem.py; S11J=$H/paired_valid40_stack_S11_so3_lean_B2e1_mem.json; S12J=$H/paired_valid40_stack_S12liteR6_mem.json
[ "$(shasum -a 256 $C | cut -c1-16)" = $SHA ] || { echo "candidate sha changed"; exit 1; }
while ! mkdir /tmp/fable_paired.lock 2>/dev/null; do sleep 30; done
echo "$(date +%T) smoke start (client/server, 35,100)"
bash $H/run_census_porter.sh $C 35,100 cz_S13 > $H/stack/cz_S13.out 2>&1
rmdir /tmp/fable_paired.lock
python3 - $H/cz_S13_census.json > $H/stack/smoke_S13.txt 2>&1 <<'PY'
import json, sys
rows = json.load(open(sys.argv[1])); assert [r['rank'] for r in rows] == [35, 100], rows
for r in rows:
    assert r['requests'] > 0 and r['flops'] > 0, r
    print(f"smoke r{r['rank']}: OK requests {r['requests']:,} server CPU {r['server_cpu_s']} s residual {r['residual_s']} s util {r['util']}")
PY
if [ $? -ne 0 ]; then echo "SMOKE FAILED"; cat $H/stack/smoke_S13.txt; tail -20 $H/stack/cz_S13.out; exit 1; fi
cat $H/stack/smoke_S13.txt
echo "$(date '+%Y-%m-%d %H:%M') porter: S13 (2a5dd324) client/server smoke OK; VALID40 vs S11 rows STARTED (identity vs S12 rows + bar)." >> $TL
bash $H/run_valid40_locked.sh $C stack_S13_mem
python3 $H/valid40_vs_ref_ci.py $H/paired_valid40_stack_S13_mem.json $S11J > $H/gate_S13_valid40_vs_S11.txt; cat $H/gate_S13_valid40_vs_S11.txt
python3 - $H/paired_valid40_stack_S13_mem.json $S12J > $H/gate_S13_identity_vs_S12.txt <<'PY'
import json, sys
C, R = json.load(open(sys.argv[1]))['results'], json.load(open(sys.argv[2]))['results']
ranks = [k.split(':')[1] for k in C if k.startswith('cand:')]
same = [r for r in ranks if C[f'cand:{r}'].get('mse') is not None and C[f'cand:{r}']['mse'] == R.get(f'cand:{r}', {}).get('mse')]
util = sorted(100 * (C[f'cand:{r}']['util'] / R[f'cand:{r}']['util'] - 1) for r in ranks)
print(f"S13 vs S12 VALID40 rows: final MSE identical on {len(same)}/{len(ranks)} nets; per-net util change {util[0]:+.3f}..{util[-1]:+.3f}% (median {util[len(util)//2]:+.3f}%)")
print('IDENTITY ' + ('PASS' if len(same) == len(ranks) == 40 else 'FAIL'))
PY
cat $H/gate_S13_identity_vs_S12.txt
python3 $H/valid40_bar.py $H/paired_valid40_stack_S13_mem.json $S11J -0.8 26 | tee $H/gate_S13_bar.json
python3 $H/valid40_bar.py $H/paired_valid40_stack_S13_mem.json $S11J -1.0 26 > $H/gate_S13_bar_strict.json; cat $H/gate_S13_bar_strict.json
if ! grep -q '"verdict": "PASS"' $H/gate_S13_bar.json || ! grep -q 'IDENTITY PASS' $H/gate_S13_identity_vs_S12.txt; then
  echo "$(date '+%Y-%m-%d %H:%M') porter: S13 NOT staged: $(cut -c1-160 $H/gate_S13_bar.json) | $(tail -1 $H/gate_S13_identity_vs_S12.txt)" >> $TL; echo "$(date +%T) S13 CHAIN DONE (not staged)"; exit 0
fi
FD=$P2/fable_fire_v1/fire_candS13; [ -e $FD ] && { echo "fire dir exists; not touching it"; exit 1; }
mkdir $FD; F=$FD/candS13_S12_ipt3_mem.py; cp -p $C $F
{ echo "FIRE CHECKS S13 = S12 (c20d67c7 = S11 #332182 + colpack + deadcut35 + read6q) + --ipt v3 (engineer E-53, in-place tree Winograd), $(date '+%Y-%m-%d %H:%M')"
  echo "file $F"; echo "sha256 $(shasum -a 256 $F | cut -c1-64) (engineer candS12ipt3_mem.py 2a5dd324...)"
  echo "validate: $(/Users/matt/arc-white-box-estimation/.venv/bin/whest validate --estimator $F 2>&1 | grep -o 'Validation passed.*\|Validation failed.*')"
  echo "einsum gate vs fired S11: $(python3 $H/einsum_site_diff.py $F $S11 | tail -1)"
  echo "import lines new vs S11: $(diff <(grep -E '^\s*(import|from) ' $S11 | sort -u) <(grep -E '^\s*(import|from) ' $F | sort -u) | grep '^>' | tr '\n' ' ' | cut -c1-300)(empty = none)"
  R=$H/scratch_rank_order/fire_repro_S13_$(date +%H%M%S); mkdir -p $R; cd $P2
  bash fable_porter_v1/build_stack.sh $R/s.py --gn=fable_porter_v1/gn_knobs/v3.json --refit=fable_physicist_v1/refit6b/s6b_B2_fullnet_lr1e4_trunk5e5/params_epoch1.pt --l15cut --join=v2 -- --holdelim --bigarena --depthrule fable_engineer_v1/depthrule_fragment_push5p6.py --f32 third --evict5 --caps --caps2 --batch --bindhold --coldbind3 --symout --lean > /dev/null
  python3 fable_engineer2_v1/build_e2mem.py $R/s.py $R/s_mem.py mem,yright,arena,hold2 > /dev/null
  python3 fable_engineer_v1/build_colpack.py $R/s_mem.py $R/s_cp.py > /dev/null
  python3 fable_porter_v1/build_deadcut.py $R/s_cp.py $R/s_cp_dc.py > /dev/null
  python3 fable_engineer2_v1/read6/build_read6.py $R/s_cp_dc.py $R/s_r6.py q > /dev/null
  python3 fable_engineer_v1/build_ipt_v3.py $R/s_r6.py $R/s_final.py > /dev/null
  echo "recipe rebuild sha $(shasum -a 256 $R/s_final.py | cut -c1-64) (must equal the fire file)"
  echo "recipe: build_stack.sh --gn=v3 --refit=B2 ep1 --l15cut --join=v2 -- <S11 kernel flags incl --symout --lean> -> build_e2mem mem,yright,arena,hold2 -> build_colpack -> build_deadcut (z 3.5, round 32) -> build_read6 q -> build_ipt_v3"
  echo "client/server smoke: $(grep -c 'OK requests' $H/stack/smoke_S13.txt) of 2 nets OK (fable_porter_v1/cz_S13_census.json)"
  echo "identity vs S12: $(tr '\n' ' ' < $H/gate_S13_identity_vs_S12.txt)"
  echo "VALID40 vs S11 rows: $(sed -n 2,4p $H/gate_S13_valid40_vs_S11.txt | tr '\n' ' ')"
  echo "bar (S12's relaxed): $(cat $H/gate_S13_bar.json)"; echo "strict -1.0: $(cat $H/gate_S13_bar_strict.json)"
} > $FD/FIRE_CHECKS.txt 2>&1
cat $FD/FIRE_CHECKS.txt
echo "$(date '+%Y-%m-%d %H:%M') porter: S13 VALID40 + identity PASS; fire copy staged: fable_fire_v1/fire_candS13/candS13_S12_ipt3_mem.py (FIRE_CHECKS.txt)." >> $TL
echo "$(date +%T) S13 CHAIN DONE (fire copy staged)"
