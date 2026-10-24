#!/usr/bin/env bash
# TAG fable-porter (PT-22, lead 14:5x): S15 = engineer candS14ab_mem.py (9d898520) = S14 (9003e536) + --ipab (alternative-basis tree kernel, E-58).
# Recipe: S14's recipe (fired S11 -> build_colpack -> build_deadcut --round=64 -> build_m64.py -> build_read6.py q -> build_ipt_v3.py -> cat
# ipt_read6_fragment.py) -> cat ipab_fragment.py. Runs after fire_chain_S14.sh (needs S14's VALID40 rows). Steps: client/server smoke (35,100) ->
# VALID40 vs S14 rows -> bar (adjusted <= -0.7%, CI excl 0; pooled raw within +-0.15%; wins reported) -> stage fable_fire_v1/fire_candS15/.
set -u
P2=/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16; H=$P2/fable_porter_v1; TL=$P2/fable_refit_v1/TEAM_LOG.md
C=$H/stack/stack_S15_mem.py; SHA=9d8985204914987e
S14F=$H/stack/stack_S14_mem.py; S14J=$H/paired_valid40_stack_S14_mem.json; S13=$P2/fable_fire_v1/fire_candS13/candS13_S12_ipt3_mem.py; S11=$P2/fable_fire_v1/fire_candS11/candS11_S10L_B2e1_mem.py
until grep -q "S14 CHAIN DONE" $H/fire_chain_S14.out 2>/dev/null; do sleep 30; done
[ -e $S14J ] || { echo "no S14 VALID40 rows"; exit 1; }
[ "$(shasum -a 256 $C | cut -c1-16)" = $SHA ] || { echo "candidate sha changed"; exit 1; }
while ! mkdir /tmp/fable_paired.lock 2>/dev/null; do sleep 30; done
echo "$(date +%T) smoke start (client/server, 35,100)"
bash $H/run_census_porter.sh $C 35,100 cz_S15 > $H/stack/cz_S15.out 2>&1
rmdir /tmp/fable_paired.lock
python3 - $H/cz_S15_census.json > $H/stack/smoke_S15.txt 2>&1 <<'PY'
import json, sys
rows = json.load(open(sys.argv[1])); assert [r['rank'] for r in rows] == [35, 100], rows
for r in rows:
    assert r['requests'] > 0 and r['flops'] > 0, r
    print(f"smoke r{r['rank']}: OK requests {r['requests']:,} server CPU {r['server_cpu_s']} s residual {r['residual_s']} s util {r['util']}")
PY
if [ $? -ne 0 ]; then echo "SMOKE FAILED"; cat $H/stack/smoke_S15.txt; tail -20 $H/stack/cz_S15.out; exit 1; fi
cat $H/stack/smoke_S15.txt
echo "$(date '+%Y-%m-%d %H:%M') porter: S15 (9d898520) client/server smoke OK; VALID40 vs S14 rows STARTED." >> $TL
bash $H/run_valid40_locked.sh $C stack_S15_mem
python3 $H/valid40_vs_ref_ci.py $H/paired_valid40_stack_S15_mem.json $S14J > $H/gate_S15_valid40_vs_S14.txt; cat $H/gate_S15_valid40_vs_S14.txt
python3 $H/valid40_bar.py $H/paired_valid40_stack_S15_mem.json $S14J -0.7 0 | tee $H/gate_S15_bar.json
python3 - $H/paired_valid40_stack_S15_mem.json $S14J > $H/gate_S15_raw.txt <<'PY'
import json, sys
C, R = json.load(open(sys.argv[1]))['results'], json.load(open(sys.argv[2]))['results']
ranks = [k.split(':')[1] for k in C if k.startswith('cand:') and C[k].get('mse') is not None and f'cand:{k.split(":")[1]}' in R]
raw = 100 * (sum(C[f'cand:{r}']['mse'] for r in ranks) / sum(R[f'cand:{r}']['mse'] for r in ranks) - 1)
print(f"pooled raw vs S14 {raw:+.3f}% on {len(ranks)} nets -> RAW {'PASS' if abs(raw) <= 0.15 and len(ranks) == 40 else 'FAIL'} (|raw| <= 0.15%)")
PY
cat $H/gate_S15_raw.txt
if ! grep -q '"verdict": "PASS"' $H/gate_S15_bar.json || ! grep -q 'RAW PASS' $H/gate_S15_raw.txt; then
  echo "$(date '+%Y-%m-%d %H:%M') porter: S15 NOT staged: $(cut -c1-160 $H/gate_S15_bar.json) | $(cat $H/gate_S15_raw.txt)" >> $TL; echo "$(date +%T) S15 CHAIN DONE (not staged)"; exit 0
fi
FD=$P2/fable_fire_v1/fire_candS15; [ -e $FD ] && { echo "fire dir exists; not touching it"; exit 1; }
mkdir $FD; F=$FD/candS15_S14_ipab_mem.py; cp -p $C $F
{ echo "FIRE CHECKS S15 = S14 (9003e536) + --ipab (engineer E-58, alternative-basis tree kernel; rounding-level), $(date '+%Y-%m-%d %H:%M')"
  echo "file $F"; echo "sha256 $(shasum -a 256 $F | cut -c1-64) (engineer candS14ab_mem.py 9d898520...)"
  echo "validate: $(/Users/matt/arc-white-box-estimation/.venv/bin/whest validate --estimator $F 2>&1 | grep -o 'Validation passed.*\|Validation failed.*')"
  echo "einsum gate vs fired S13: $(python3 $H/einsum_site_diff.py $F $S13 | tail -1)"
  echo "import lines new vs S13: $(diff <(grep -E '^\s*(import|from) ' $S13 | sort -u) <(grep -E '^\s*(import|from) ' $F | sort -u) | grep '^>' | tr '\n' ' ' | cut -c1-300)(empty = none)"
  R=$H/scratch_rank_order/fire_repro_S15_$(date +%H%M%S); mkdir -p $R; cd $P2
  python3 fable_engineer_v1/build_colpack.py $S11 $R/a_cp.py > /dev/null
  python3 fable_porter_v1/build_deadcut.py $R/a_cp.py $R/b_dc64.py --round=64 > /dev/null
  python3 fable_engineer_v1/build_m64.py $R/b_dc64.py $R/c_m64.py > /dev/null
  python3 fable_engineer2_v1/read6/build_read6.py $R/c_m64.py $R/d_r6.py q > /dev/null
  python3 fable_engineer_v1/build_ipt_v3.py $R/d_r6.py $R/e_ipt.py > /dev/null
  cat $R/e_ipt.py fable_engineer_v1/ipt_read6_fragment.py > $R/f_s14.py
  cat $R/f_s14.py fable_engineer_v1/ipab_fragment.py > $R/s_final.py
  echo "recipe rebuild sha $(shasum -a 256 $R/s_final.py | cut -c1-64) (must equal the fire file); S14 intermediate $(shasum -a 256 $R/f_s14.py | cut -c1-16) (S14 9003e536)"
  echo "recipe: fired S11 -> build_colpack -> build_deadcut --round=64 -> build_m64.py -> build_read6.py q -> build_ipt_v3.py -> cat ipt_read6_fragment.py -> cat ipab_fragment.py"
  echo "client/server smoke: $(grep -c 'OK requests' $H/stack/smoke_S15.txt) of 2 nets OK (fable_porter_v1/cz_S15_census.json)"
  echo "VALID40 vs S14 rows: $(sed -n 2,4p $H/gate_S15_valid40_vs_S14.txt | tr '\n' ' ')"
  echo "bar: $(cat $H/gate_S15_bar.json) | $(cat $H/gate_S15_raw.txt)"
} > $FD/FIRE_CHECKS.txt 2>&1
cat $FD/FIRE_CHECKS.txt
echo "$(date '+%Y-%m-%d %H:%M') porter: S15 PASSES (VALID40 vs S14 + raw); fire copy staged: fable_fire_v1/fire_candS15/candS15_S14_ipab_mem.py (FIRE_CHECKS.txt)." >> $TL
echo "$(date +%T) S15 CHAIN DONE (fire copy staged)"
