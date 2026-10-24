#!/bin/bash
# TAG opus-release: run the whole release qualification of one candidate, stage by stage, one heavy process at a time.
# Nothing here uploads or submits; the pipeline ends with a packaged, validated archive and four verdict lines.
# usage: qualify.sh <candidate.py> <tag> [parent.py for stage B, default = the bank S35] [extra stage-B gate options...]
#   e.g. a CPU variant that must keep the predictions:  qualify.sh cand.py fastA parent.py --max-adjusted-pct 2 --max-raw-pct 0.05 --min-wins 0 --ci-upper 100
# Order: A packaging (seconds) -> setup margin (seconds, light) -> C boundary census paired with the bank, setup under the
# platform's clock (about 6 min quiet) -> C again as a stress run, one session set up at the pace of a worker 1.3x slower (2 min)
# -> D 20-network worker (about 12 min quiet) -> B native 40 networks vs the parent (about 35 min quiet for two fresh arms;
# stored arms are reused). Under load: x1.5-3.
set -u
R="$(cd "$(dirname "$0")" && pwd)"; PY=/Users/matt/arc-white-box-estimation/.venv/bin/python
CAND="$1"; TAG="$2"; PARENT="${3:-$R/../../codex_resume_20260926_v2/hill_climb_prefix128_fusion_v1/candidate.py}"; shift; shift; shift 2>/dev/null
mkdir -p "$R/runs"; LOG="$R/runs/${TAG}_qualify.log"
echo "$(date -u +%FT%TZ) qualify $TAG candidate=$CAND parent=$PARENT" | tee -a "$LOG"
"$PY" -B "$R/stage_a_package.py" "$CAND" "${TAG}_a" 2>&1 | tee -a "$LOG"
"$PY" -B "$R/setup_margin.py" "candidate=$CAND" 2>&1 | tee -a "$LOG"
"$PY" -B "$R/stage_c_boundary.py" "$CAND" "${TAG}_c" --short-setup 2>&1 | tee -a "$LOG"
"$PY" -B "$R/stage_c_boundary.py" "$CAND" "${TAG}_c_slow" --setup-slowdown 1.3 --legs C --parent-from "$R/runs/${TAG}_c" 2>&1 | tee -a "$LOG"
"$PY" -B "$R/stage_d_serving.py" "$CAND" "${TAG}_d" --short-setup 2>&1 | tee -a "$LOG"
"$PY" -B "$R/stage_b_native40.py" "$CAND" "${TAG}_b" --parent "$PARENT" "$@" 2>&1 | tee -a "$LOG"
echo "$(date -u +%FT%TZ) ---- verdicts for $TAG (NOT uploaded; the lead submits) ----" | tee -a "$LOG"
grep -E '^\[(A|B|C|D) ' "$LOG" | tail -5
