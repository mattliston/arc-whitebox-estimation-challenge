#!/usr/bin/env bash
# TAG fable-porter (PT-46, covhunter 22:5x heads-up "round 3 on the S23 STRUCTURE"): builds any candidate of the S23 structure =
# S18 recipe + <params.pt> literals + <dials.json> (--gn=shared) + d6x + engineer2 s1 + s1=0 (stack/extra_s1_s1zero_fragment.py) + Nystrom join.
# (loopR1 params + round-1 dials) must rebuild S23 9247a212 (byte-check this script with those inputs first). No launch; build + static checks only.
# usage: prebuild_S23struct.sh <params.pt> <dials.json> <tag>   -> stack/stack_<tag>_mem.py
set -eu
P2=/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16; H=$P2/fable_porter_v1
PT=$1; DJ=$2; TAG=$3; EX=$H/stack/extra_s1_s1zero_fragment.py; S16F=$P2/fable_fire_v1/fire_candS16/candS16_S15_bb6_mem.py
[ "$(shasum -a 256 $EX | cut -c1-16)" = "$(cat $P2/fable_engineer2_v1/lrroute/s1lr_fragment_v2.py $P2/fable_engineer2_v1/lrroute/s1zero_fragment.py | shasum -a 256 | cut -c1-16)" ] || { echo "extra fragment != s1 + s1zero"; exit 1; }
bash $H/prebuild_dials_stack.sh $PT $DJ ${TAG}_pre $EX > $H/stack/prebuild_${TAG}.txt 2>&1
python3 $H/build_join_nystrom.py $H/stack/stack_${TAG}_pre_mem.py $H/stack/stack_${TAG}_mem.py > /dev/null
C=$H/stack/stack_${TAG}_mem.py
echo "$TAG: $(shasum -a 256 $C | cut -c1-64) | params $(shasum -a 256 $PT | cut -c1-16) dials $(shasum -a 256 $DJ | cut -c1-16)"
echo "  $(cd $H && ./verify_literals.py $C $PT) | $(python3 $H/einsum_site_diff.py $C $S16F | tail -1 | cut -c1-40) | $(/Users/matt/arc-white-box-estimation/.venv/bin/whest validate --estimator $C 2>&1 | grep -o 'Validation passed\|Validation failed')"
