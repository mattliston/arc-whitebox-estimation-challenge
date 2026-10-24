#!/usr/bin/env bash
# TAG fable-porter (PT-8): ONE-COMMAND fire-stack builder with optional layers. Every step is count-asserted or AST-located and refuses to
# overwrite; intermediates and a sha list are kept in <out stem>_build/.
#   bank (sha pinned)
#   -> fable_engineer2_v1/build_e2.py: --e2=v7 (default; the fired S2/S3 lineage: every stage except hold2, output pinned 737bdcc9...)
#                                      --e2=v8 (final default stages incl. hold2, output pinned 1e33bace...)
#   -> build_physics.py K31 row (--k31row=c0,c1; default 0.85,0.5) at L1-14 -> build_bfix_v2.py (--bfix=gamma; default 0.1)
#   -> [--gn=<knobs.json>]   build_gn_knobs.py: covhunter's covariance-channel knobs          (physics layer: native vs torch per net)
#   -> [--refit=<params.pt>] build_refit_candidate_v2.py: learned literals from a refit .pt   (physics layer: native vs torch per net)
#   -> [--l15cut[=zc]]      fable_physicist_v1/leadergap/build_l15_rowcut.py (zc default 3.5): L15 k3 row cut   (physics layer: native vs torch)
#   -> [-- <kernel flags>]   fable_engineer_v1/build_eng.py <flags>, e.g. --holdelim | --holdelim --sq6 [--push64]
#                                                                     (kernel layer: MSE bit-identical, or rounding-level for sq6/push64)
#   -> every3 join schedule: --join=v1 (default; build_join_schedule_v1_fired_S2.py, the code of the graded S2 and of S4) or --join=v2
#      (build_join_schedule_v2.py: same predictions, plus the terminal join-count assert)
#   -> [--rank=R]          build_tier_rank.py (PT-14): old-tier frame rank R as one literal (R <= 384, multiple of 32); applied after the join;
#                          R=384 edits are prediction- and bill-identical; rank-then-diet == diet-then-rank (same final sha). PT-8b, 2026-09-24.
# usage: build_stack.sh <out.py> [--e2=v7|v8] [--k31row=c0,c1] [--bfix=gamma] [--gn=<knobs.json>] [--refit=<params.pt>] [--l15cut[=zc]]
#                        [--join=v1|v2] [--rank=R] [-- <build_eng flags>]   (no new options = byte-identical to build_stack_v1_S4.sh; no --rank =
#                        byte-identical to build_stack_v2_S11.sh)
# e.g.   build_stack.sh candX.py -- --holdelim                                     (S3's math: S2 + the HE kernel layer)
#        build_stack.sh candY.py --gn=gn_knobs/v3.json -- --holdelim --sq6
# Parity: check_layer.sh <with.py> <without.py> <ranks> compares native MSE and flops of two builds that differ by one layer.
set -euo pipefail
P2=/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16; H=$P2/fable_porter_v1
BANK=$P2/codex_resume_20260919_v1/current_p_message_covariance_release_v1/candidate.py
BANK_SHA=d79463f389a6b6e4172415b183601d327bc13b560fd7096ddbf781dce7a94d0e
OUT="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"; shift
E2=v7; GN=""; REFIT=""; ENG=(); K31ROW=0.85,0.5; BFIX=0.1; L15CUT=""; JOIN=v1; RANK=""
while [ $# -gt 0 ]; do
  case "$1" in
    --e2=*) E2="${1#--e2=}";;
    --k31row=*) K31ROW="${1#--k31row=}";;
    --bfix=*) BFIX="${1#--bfix=}";;
    --l15cut) L15CUT=3.5;;
    --l15cut=*) L15CUT="${1#--l15cut=}";;
    --join=*) JOIN="${1#--join=}";;
    --rank=*) RANK="${1#--rank=}";;
    --gn=*) GN="$(cd "$(dirname "${1#--gn=}")" && pwd)/$(basename "${1#--gn=}")";;
    --refit=*) REFIT="$(cd "$(dirname "${1#--refit=}")" && pwd)/$(basename "${1#--refit=}")";;
    --) shift; ENG=("$@"); break;;
    *) echo "unknown option $1 (kernel flags go after --)"; exit 1;;
  esac; shift
done
case "$E2" in
  v7) E2_PATCHES=pair,bd,learned,loop,fuse,vec,share,mem,hold; E2_SHA=737bdcc966ebbe0b;;
  v8) E2_PATCHES=pair,bd,learned,loop,fuse,vec,share,mem,hold,hold2; E2_SHA=1e33bace0c3a8d3c;;
  *) echo "--e2 must be v7 or v8"; exit 1;;
esac
[ -e "$OUT" ] && { echo "refusing to overwrite $OUT"; exit 1; }
B="${OUT%.py}_build"; mkdir "$B"
sha() { shasum -a 256 "$1" | cut -c1-"${2:-64}"; }
[ "$(sha "$BANK")" = "$BANK_SHA" ] || { echo "bank sha changed"; exit 1; }
python3 "$P2/fable_engineer2_v1/build_e2.py" "$BANK" "$B/1_e2.py" "$E2_PATCHES" > "$B/1_e2.log"
[ "$(sha "$B/1_e2.py" 16)" = "$E2_SHA" ] || { echo "build_e2 $E2 output changed (sha $(sha "$B/1_e2.py" 16)); re-verify before use"; exit 1; }
python3 "$H/build_physics.py" "$B/1_e2.py" "$B/2_phys.py" "$K31ROW@1-14" none '' > "$B/2_phys.log"
python3 "$H/build_bfix_v2.py" "$B/2_phys.py" "$BFIX" "$B/3_bfix.py" > "$B/3_bfix.log"
LAST="$B/3_bfix.py"
if [ -n "$GN" ]; then python3 "$H/build_gn_knobs.py" "$LAST" "$GN" "$B/4_gn.py" > "$B/4_gn.log"; LAST="$B/4_gn.py"; fi
if [ -n "$REFIT" ]; then /Users/matt/anaconda3/bin/python "$H/build_refit_candidate_v2.py" "$REFIT" "$LAST" "$B/5_refit.py" > "$B/5_refit.log"; LAST="$B/5_refit.py"; fi
if [ -n "$L15CUT" ]; then python3 "$P2/fable_physicist_v1/leadergap/build_l15_rowcut.py" "$LAST" "$B/5b_l15cut.py" "$L15CUT" > "$B/5b_l15cut.log"; LAST="$B/5b_l15cut.py"; fi
if [ ${#ENG[@]} -gt 0 ]; then python3 "$P2/fable_engineer_v1/build_eng.py" "$LAST" "$B/6_eng.py" "${ENG[@]}" > "$B/6_eng.log"; LAST="$B/6_eng.py"; fi
case "$JOIN" in v1) JB=build_join_schedule_v1_fired_S2.py;; v2) JB=build_join_schedule_v2.py;; *) echo "--join must be v1 or v2"; exit 1;; esac
if [ -n "$RANK" ]; then
  python3 "$P2/fable_join_v1/$JB" "$LAST" "$B/7_join.py" every3 > "$B/7_join.log"
  python3 "$H/build_tier_rank.py" "$B/7_join.py" "$OUT" "$RANK" > "$B/8_rank.log"
else
  python3 "$P2/fable_join_v1/$JB" "$LAST" "$OUT" every3 > "$B/7_join.log"
fi
{ echo "built $(date '+%Y-%m-%d %H:%M'): e2 $E2; k31row $K31ROW; bfix $BFIX; l15cut ${L15CUT:-none}; join $JOIN; gn ${GN:-none}$( [ -n "$GN" ] && echo " (sha $(sha "$GN" 16))"); refit ${REFIT:-none}$( [ -n "$REFIT" ] && echo " (sha $(sha "$REFIT" 16))"); kernel flags: ${ENG[*]:-none}"
  [ -n "$RANK" ] && echo "tier rank $RANK (build_tier_rank.py)"
  for f in "$B"/[1-7]_*.py "$B"/5b_*.py "$OUT"; do [ -e "$f" ] || continue; echo "$(sha "$f") $(basename "$f")"; done; } > "$B/CHAIN.txt"
cat "$B/CHAIN.txt"
