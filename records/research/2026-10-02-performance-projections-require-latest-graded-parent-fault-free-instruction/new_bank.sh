#!/bin/bash
# TAG opus-release: after a file has been GRADED with no network lost, make it the bank of the release stages. One command for the
# four steps done by hand for S38 on 2026-10-02 08:36-09:00 UTC (each step was run and checked then; this script as a whole has
# NOT been run yet: read its output the first time).
#   1. check_projection.py: what stage C of its chain projected against what the platform measured (log the lines);
#   2. set_bank.py: calibration/bank.json = this file with its own platform record (kernel, overhead, residual, wall, and the
#      memory-bound share carried over through its stage C record against the bank in force);
#   3. make_reference.py: its 20-network worker run of the chain becomes stage D's reference (calibration/bank_stage_d_paced.json;
#      the reference it replaces is moved to _stale/);
#   4. its two candidate legs of the chain's stage C are copied to runs/c_<name>_bank_legs/P0, P1 and named in a new
#      calibration/bank_stage_c_paced_<UTC stamp>.txt (the stored parent legs of quick.sh and --parent-from; use them only when
#      the window is as quiet as the chain's was).
# Nothing is uploaded. usage: new_bank.sh <name> <file.py> <eval_<id>.json> <chain tag>     (the chain's runs/c_<tag>, runs/d_<tag>)
set -eu
R="$(cd "$(dirname "$0")" && pwd)"; PY=/Users/matt/arc-white-box-estimation/.venv/bin/python
[ $# -eq 4 ] || { awk 'NR > 1 && /^#/ {print; next} NR > 1 {exit}' "$0"; exit 2; }
NAME="$1"; FILE="$2"; EVAL="$3"; TAG="$4"; C="$R/runs/c_${TAG}"; D="$R/runs/d_${TAG}/worker"; LEGS="$R/runs/c_${NAME}_bank_legs"
[ -f "$FILE" ] && [ -f "$EVAL" ] && [ -f "$C/stage_c.json" ] && [ -f "$D/rows.jsonl" ] || { echo "missing: the file, its evaluation, $C/stage_c.json or $D/rows.jsonl"; exit 2; }
[ -e "$LEGS" ] && { echo "$LEGS exists already"; exit 2; }
"$PY" -B "$R/check_projection.py" "$EVAL" "$C/stage_c.json"
"$PY" -B "$R/set_bank.py" "$NAME" "$FILE" "$EVAL" "$C/stage_c.json"
REF="$("$PY" -B "$R/make_reference.py" "$D" "$EVAL" | tail -n 1)"
STAMP="$(date -u +%Y%m%dT%H%MZ)"; mkdir -p "$R/_stale"
[ -e "$R/calibration/bank_stage_d_paced.json" ] && mv "$R/calibration/bank_stage_d_paced.json" "$R/_stale/bank_stage_d_paced_replaced_${STAMP}.json"
cp -n "$REF" "$R/calibration/bank_stage_d_paced.json"
mkdir "$LEGS"; i=0
for leg in "$C"/C*; do [ -f "$leg/rows.jsonl" ] && cp -R "$leg" "$LEGS/P$i" && i=$((i + 1)); done
echo "copies of the C legs of $C (the file's sessions of its chain) under P names: stored parent legs of the bank $NAME" > "$LEGS/README.txt"
echo "$LEGS" > "$R/calibration/bank_stage_c_paced_${STAMP}.txt"
"$PY" -B -c "import sys; sys.path.insert(0, '$R'); import common; print('bank in force:', common.BANK_NAME, '#%d' % common.PLATFORM['sid'], common.BANK)"
echo "stage D reference: $REF -> calibration/bank_stage_d_paced.json; stored parent legs: $LEGS ($i sessions)"
