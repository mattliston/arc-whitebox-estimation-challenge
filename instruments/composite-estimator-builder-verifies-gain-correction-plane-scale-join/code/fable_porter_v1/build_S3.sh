#!/usr/bin/env bash
# TAG fable-porter (PT-6): ONE-COMMAND builder for the S-line fire candidate:
#   bank (pinned) -> fable_engineer2_v1/build_e2.py (final default patches; output pinned to the announced v8 sha)
#   -> fable_porter_v1/build_physics.py K31 row 0.85,0.5@1-14 -> fable_porter_v1/build_bfix_v2.py gamma 0.1
#   -> [fable_engineer_v1/build_eng.py <kernel flags>]   (skipped when no flags: that is S2's math on e2 v8)
#   -> fable_join_v1/build_join_schedule_v1_fired_S2.py every3 (the version that built the fired S2; the 23:11 revision raises at L8 with every3)
# Every step is count-asserted / refuses to overwrite; intermediates are kept in <out stem>_build/ for audit.
# usage: build_S3.sh <out.py> [build_eng flags ...]    e.g.  build_S3.sh candS3_v9.py --grouped --read5   |   build_S3.sh candS3_H.py --holdonly
set -euo pipefail
P2=/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16
BANK=$P2/codex_resume_20260919_v1/current_p_message_covariance_release_v1/candidate.py
BANK_SHA=d79463f389a6b6e4172415b183601d327bc13b560fd7096ddbf781dce7a94d0e
E2_SHA=1e33bace0c3a8d3c   # engineer2's announced final build_e2 output (v8) on the bank; a change here needs a new bit-exactness check
OUT="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"; shift; FLAGS=("$@")
[ -e "$OUT" ] && { echo "refusing to overwrite $OUT"; exit 1; }
B="${OUT%.py}_build"; mkdir "$B"
sha() { shasum -a 256 "$1" | cut -c1-"${2:-64}"; }
[ "$(sha "$BANK")" = "$BANK_SHA" ] || { echo "bank sha changed"; exit 1; }
python3 "$P2/fable_engineer2_v1/build_e2.py" "$BANK" "$B/1_e2.py" > "$B/1_e2.log"
[ "$(sha "$B/1_e2.py" 16)" = "$E2_SHA" ] || { echo "build_e2 output changed from the announced v8 (sha $(sha "$B/1_e2.py" 16)); re-verify before use"; exit 1; }
python3 "$P2/fable_porter_v1/build_physics.py" "$B/1_e2.py" "$B/2_phys.py" 0.85,0.5@1-14 none '' > "$B/2_phys.log"
python3 "$P2/fable_porter_v1/build_bfix_v2.py" "$B/2_phys.py" 0.1 "$B/3_bfix.py" > "$B/3_bfix.log"
LAST="$B/3_bfix.py"
if [ ${#FLAGS[@]} -gt 0 ]; then
  python3 "$P2/fable_engineer_v1/build_eng.py" "$LAST" "$B/4_eng.py" "${FLAGS[@]}" > "$B/4_eng.log"
  LAST="$B/4_eng.py"
fi
python3 "$P2/fable_join_v1/build_join_schedule_v1_fired_S2.py" "$LAST" "$OUT" every3 > "$B/5_join.log"
{ echo "built $(date '+%Y-%m-%d %H:%M'): kernel flags: ${FLAGS[*]:-none}"; for f in "$B"/[1-4]_*.py "$OUT"; do echo "$(sha "$f") $(basename "$f")"; done; } > "$B/CHAIN.txt"
cat "$B/CHAIN.txt"
