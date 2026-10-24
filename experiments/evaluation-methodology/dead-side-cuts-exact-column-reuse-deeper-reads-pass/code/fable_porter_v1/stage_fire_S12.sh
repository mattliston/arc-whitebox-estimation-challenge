#!/usr/bin/env bash
# TAG fable-porter (PT-17b, lead 12:25: S12 fire bar relaxed to pooled adjusted <= -0.8%, CI excluding 0, >= 26/40). Runs after fire_chain_S12.sh
# (whose own bar check used -1.0%): evaluates the S12 VALID40 vs S11 rows at -0.8 and, on PASS, stages fable_fire_v1/fire_candS12/ with the same
# checks as fire_chain_S12.sh step (3) (sha, validate, einsum gate vs fired S11, no new imports vs S11, byte-rebuild from the recipe, smoke).
set -u
P2=/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16; H=$P2/fable_porter_v1; TL=$P2/fable_refit_v1/TEAM_LOG.md
C=$H/stack/stack_S11_cp_dc35_r6q_mem.py; SHA=c20d67c7c1e64c361c02e6362339703be5565eebe27e3065299888f73c5d1e11
S11=$P2/fable_fire_v1/fire_candS11/candS11_S10L_B2e1_mem.py; S11J=$H/paired_valid40_stack_S11_so3_lean_B2e1_mem.json; J=$H/paired_valid40_stack_S12liteR6_mem.json
until grep -q "CHAIN DONE" $H/fire_chain_S12.out 2>/dev/null; do sleep 30; done
python3 $H/valid40_bar.py $J $S11J -0.8 26 | tee $H/gate_S12liteR6_bar_relaxed.json
grep -q '"verdict": "PASS"' $H/gate_S12liteR6_bar_relaxed.json || { echo "$(date '+%Y-%m-%d %H:%M') porter: S12 FAILS the relaxed bar (-0.8%): $(cut -c1-200 $H/gate_S12liteR6_bar_relaxed.json)" >> $TL; echo "$(date +%T) STAGE DONE (relaxed bar FAIL)"; exit 0; }
FD=$P2/fable_fire_v1/fire_candS12
if [ -e $FD/FIRE_CHECKS.txt ]; then echo "fire copy already staged by fire_chain_S12.sh"; cat $FD/FIRE_CHECKS.txt; exit 0; fi
[ -e $FD ] && { echo "fire dir exists without FIRE_CHECKS.txt; not touching it"; exit 1; }
[ "$(shasum -a 256 $C | cut -c1-64)" = $SHA ] || { echo "candidate sha changed"; exit 1; }
mkdir $FD; F=$FD/candS12_S11_cp_dc35_r6q_mem.py; cp -p $C $F
{ echo "FIRE CHECKS S12 = S11 (#332182) + colpack (engineer E-49) + deadcut35 (porter PT-15, current kernels) + read6q (engineer2), $(date '+%Y-%m-%d %H:%M')"
  echo "bar (lead 12:25, relaxed for S12): pooled adjusted <= -0.8%, CI excluding 0, >= 26/40 -> $(cat $H/gate_S12liteR6_bar_relaxed.json)"
  echo "file $F"; echo "sha256 $(shasum -a 256 $F | cut -c1-64) (expected $SHA)"
  echo "validate: $(/Users/matt/arc-white-box-estimation/.venv/bin/whest validate --estimator $F 2>&1 | grep -o 'Validation passed.*\|Validation failed.*')"
  echo "einsum gate vs fired S11: $(python3 $H/einsum_site_diff.py $F $S11 | tail -1)"
  echo "import lines new vs S11: $(diff <(grep -E '^\s*(import|from) ' $S11 | sort -u) <(grep -E '^\s*(import|from) ' $F | sort -u) | grep '^>' | tr '\n' ' ' | cut -c1-300)(empty = none)"
  R=$H/scratch_rank_order/fire_repro_S12_$(date +%H%M%S); mkdir -p $R; cd $P2
  bash fable_porter_v1/build_stack.sh $R/s.py --gn=fable_porter_v1/gn_knobs/v3.json --refit=fable_physicist_v1/refit6b/s6b_B2_fullnet_lr1e4_trunk5e5/params_epoch1.pt --l15cut --join=v2 -- --holdelim --bigarena --depthrule fable_engineer_v1/depthrule_fragment_push5p6.py --f32 third --evict5 --caps --caps2 --batch --bindhold --coldbind3 --symout --lean > /dev/null
  python3 fable_engineer2_v1/build_e2mem.py $R/s.py $R/s_mem.py mem,yright,arena,hold2 > /dev/null
  python3 fable_engineer_v1/build_colpack.py $R/s_mem.py $R/s_cp.py > /dev/null
  python3 fable_porter_v1/build_deadcut.py $R/s_cp.py $R/s_cp_dc.py > /dev/null
  python3 fable_engineer2_v1/read6/build_read6.py $R/s_cp_dc.py $R/s_final.py q > /dev/null
  echo "recipe rebuild sha $(shasum -a 256 $R/s_final.py | cut -c1-64) (must equal the fire file)"
  echo "recipe: build_stack.sh --gn=v3 --refit=B2 ep1 --l15cut --join=v2 -- <S11 kernel flags incl --symout --lean> -> build_e2mem mem,yright,arena,hold2 -> build_colpack -> build_deadcut (z 3.5, round 32) -> build_read6 q"
  echo "client/server smoke: $(grep -c 'OK requests' $H/stack/smoke_S12liteR6.txt 2>/dev/null) of 2 nets OK (fable_porter_v1/cz_S12liteR6_census.json)"
  echo "VALID40 vs S11 rows: $(sed -n 2,4p $H/gate_S12liteR6_valid40_vs_S11.txt | tr '\n' ' ')"
} > $FD/FIRE_CHECKS.txt 2>&1
cat $FD/FIRE_CHECKS.txt
echo "$(date '+%Y-%m-%d %H:%M') porter: S12 PASSES the relaxed bar (-0.8%); fire copy staged: fable_fire_v1/fire_candS12/candS12_S11_cp_dc35_r6q_mem.py (FIRE_CHECKS.txt)." >> $TL
echo "$(date +%T) STAGE DONE (fire copy staged)"
