#!/bin/bash
# TAG fable-lead 2026-10-19: the mandatory pre-fire gates, in order, for one candidate (Matt 10-19: fire verified improvements through
# the mandatory gates only): (1) official five-start setup gate under the 7.5 GB guard, (2) stage A package (proven = 95c),
# (3) prefire check against the last fired archive (the only accepted FAIL is the approved gc import). Stops at the first failure.
# usage: release_min_v1.sh <candidate.py> <tag> <qualification dir (new)> <reference archive = last fired submission.tar.gz>
set -u
O=/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1
PY=/Users/matt/arc-white-box-estimation/.venv/bin/python
CAND="$1"; TAG="$2"; Q="$3"; REF="$4"
PROVEN=$O/lead/runs/astra_oct18_accuracy_v1/setup_memo_diet_v1/candidate_compact10_capacity35_memodiet_HELD_v1.py
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
[ -e "$Q" ] && { echo "qualification dir exists: $Q"; exit 2; }
mkdir -p "$Q"
stamp() { date -u +%Y-%m-%dT%H:%M:%SZ; }
echo "$(stamp) SETUP5 START $CAND"
$PY -B $O/lead/runs/astra_crunch_physics_v1/guard_memory50_v1.py --record "$Q/SETUP_GUARD_v1.json" --seconds 120 --footprint-limit 7500000000 --rss-backstop 7500000000 \
  $O/lead/runs/astra_crunch_runtime_v1/failure336631_v1/client310_v1/bin/python -B $O/lead/runs/astra_oct18_accuracy_v1/official_setup_five_v2.py "$CAND" "$Q/setup5" --setup-timeout 5 > "$Q/setup5.log" 2>&1
S=$($PY -c "import json;print(json.load(open('$Q/setup5/SUMMARY_v1.json'))['status'])" 2>/dev/null || echo MISSING)
E=$($PY -c "import json;d=json.load(open('$Q/setup5/SUMMARY_v1.json'));print(max(r['start_envelope_s'] for r in d['records']))" 2>/dev/null || echo NA)
echo "$(stamp) SETUP5 $S max start envelope $E s"
[ "$S" = "PASS" ] || exit 3
echo "$(stamp) STAGE A START tag $TAG"
(cd $O && $PY -B release/stage_a_package.py "$CAND" "$TAG" --proven "$PROVEN") > "$Q/stage_a.log" 2>&1
tail -1 "$Q/stage_a.log"
grep -q '^\[A package\] PASS' "$Q/stage_a.log" || { echo "$(stamp) STAGE A FAILED"; exit 4; }
SHA8=$(shasum -a 256 "$CAND" | cut -c1-8)
echo "$(stamp) PREFIRE START"
(cd $O && $PY -B fire/prefire_check_v1.py release/runs/$TAG/submission.tar.gz "$SHA8" "$REF") > "$Q/prefire.log" 2>&1
grep -v '^  eye' "$Q/prefire.log" | tail -4
# accepted: the only FAIL line is the gc import; no new einsum signatures
BAD=$(grep '^FAIL' "$Q/prefire.log" | grep -v "imports outside the allowed set: \['gc'\]" | wc -l | tr -d ' ')
NEW=$(grep -o 'new signatures: [0-9]*' "$Q/prefire.log" | awk '{print $3}')
echo "$(stamp) PREFIRE other FAIL lines: $BAD, new einsum signatures: ${NEW:-NA}"
[ "$BAD" = "0" ] && [ "${NEW:-1}" = "0" ] && echo "$(stamp) GATES CLEAR: archive $(shasum -a 256 $O/release/runs/$TAG/submission.tar.gz | cut -c1-12) ready for fire_once" || { echo "$(stamp) PREFIRE NOT CLEAR"; exit 5; }
