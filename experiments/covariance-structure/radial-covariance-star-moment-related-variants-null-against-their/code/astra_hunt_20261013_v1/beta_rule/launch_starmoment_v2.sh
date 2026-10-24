#!/bin/bash
# Root alone launches under the shared sole-process lock. Never started by beta lane.
# Usage: bash launch_starmoment_v1.sh /absolute/private/run /absolute/fresh/guard.json [seconds]
set -euo pipefail
H=/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_hunt_20261013_v1/beta_rule
O=/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1
RUN=$1
RECORD=$2
SECONDS_LIMIT=${3:-7200}
case "$RUN" in "$H"/run_starmoment_control_v2|"$H"/run_starmoment_arm_v2) ;; *) exit 10 ;; esac
case "$RECORD" in "$H"/*) ;; *) exit 11 ;; esac
test ! -e "$RECORD"
FREE=$(/usr/bin/memory_pressure | /usr/bin/awk '/System-wide memory free percentage/ {gsub(/%/, "", $NF); print $NF}')
test "$FREE" -ge 50
mkdir -p "$H/tmp" "$H/cache" "$H/torch_cache" "$H/matplotlib_cache"
cd "$H"
ENTRY_ARGS=("$H/worker_starmoment_v1.py" "$RUN" 0)
case "${ASTRA_COMPARE_MODE:-}" in
  "") ;;
  control|arm)
    COMPARISON=${ASTRA_COMPARE_ARTIFACT:-"$RECORD.comparison.json"}
    case "$COMPARISON" in "$H"/*) ;; *) exit 12 ;; esac
    test ! -e "$COMPARISON"
    if test "$ASTRA_COMPARE_MODE" = control; then
      COMPARE_ARG=control
    else
      COMPARE_ARG="${COMPARISON%.*}.worker.log"
    fi
    ENTRY_ARGS=("$H/screen_entry_starmoment_v1.py" --worker "$H/worker_starmoment_v1.py"
      --run "$RUN" --index 0 --comparator "$H/compare_starmoment_v2.py"
      --artifact "$COMPARISON" --compare-arg "$COMPARE_ARG")
    ;;
  *) exit 13 ;;
esac
exec /usr/bin/nice -n 10 /Users/matt/arc-white-box-estimation/.venv/bin/python -B "$O/release/guard_footprint_v3.py" \
  --record "$RECORD" --seconds "$SECONDS_LIMIT" --footprint-limit 4000000000 --rss-backstop 4000000000 \
  /usr/bin/env -u TRAINER_TRUTH_PLAN -u TRAINER_TRUTH_PLAN_SHA256 \
  PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX="$H/cache" TMPDIR="$H/tmp" \
  ASTRA_PRESSURE_RECORD="$RECORD.pressure.jsonl" \
  XDG_CACHE_HOME="$H/cache" TORCH_HOME="$H/torch_cache" MPLCONFIGDIR="$H/matplotlib_cache" \
  CUDA_VISIBLE_DEVICES=-1 TRAINER_CARD=M TRAINER_PL=1 TRAINER_F4C=1 TRAINER_THREADS=2 \
  TRAINER_GENERATED_ALLOW_FILE="$H/allow_starmoment_v1.txt" TRAINER_GENERATED_MIN_N=15990784 \
  TRAINER_RELIEF=1 TRAINER_STAR_ATTACH=0 TRAINER_BIRTH_ATTACH=0 TRAINER_RSS_LIMIT=4000000000 \
  OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 VECLIB_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 BLIS_NUM_THREADS=2 \
  PYTHONPATH="$O/trainer:$H" /Users/matt/anaconda3/bin/python -B "${ENTRY_ARGS[@]}"
