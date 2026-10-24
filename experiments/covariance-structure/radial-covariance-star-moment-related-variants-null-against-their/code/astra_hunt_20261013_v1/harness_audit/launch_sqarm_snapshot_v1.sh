#!/bin/bash
# Run only by the parent holding the sole Astra compute slot. No numerical work at preparation.
# Usage: bash launch_private_v1.sh /absolute/private/run /absolute/fresh/guard.json [seconds]
# Optional ASTRA_COMPARE_MODE=control|arm runs Sq comparator in the worker's same process.
set -euo pipefail
A=/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_hunt_20261013_v1
O=/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1
H=$A/harness_audit
RUN=$1
RECORD=$2
SECONDS_LIMIT=${3:-7200}
case "$RUN" in "$A"/*) ;; *) exit 10 ;; esac
case "$RECORD" in "$A"/*) ;; *) exit 11 ;; esac
test ! -e "$RECORD"
FREE=$(/usr/bin/memory_pressure | /usr/bin/awk '/System-wide memory free percentage/ {gsub(/%/, "", $NF); print $NF}')
test "$FREE" -ge 50
mkdir -p "$H/tmp" "$H/cache" "$H/torch_cache" "$H/matplotlib_cache"
ENTRY=("$H/worker_astra_v1.py" "$RUN" 0)
case "${ASTRA_COMPARE_MODE:-}" in
  '') ;;
  control|arm)
    ENTRY=("$H/sq_arm_entry_v1.py" --worker "$H/worker_astra_v1.py" --run "$RUN" --index 0
      --comparator "$H/compare_sq_v1.py" --artifact "$RECORD.comparison.json")
    if [ "$ASTRA_COMPARE_MODE" = control ]; then ENTRY+=(--compare-arg control); fi
    ;;
  *) exit 12 ;;
esac
cd "$H"
exec /usr/bin/nice -n 10 /Users/matt/arc-white-box-estimation/.venv/bin/python -B "$O/release/guard_footprint_v3.py" \
  --record "$RECORD" --seconds "$SECONDS_LIMIT" --footprint-limit 4000000000 --rss-backstop 4000000000 \
  /usr/bin/env -u TRAINER_TRUTH_PLAN -u TRAINER_TRUTH_PLAN_SHA256 \
  PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX="$H/cache" TMPDIR="$H/tmp" \
  ASTRA_PRESSURE_RECORD="$RECORD.pressure.jsonl" \
  XDG_CACHE_HOME="$H/cache" TORCH_HOME="$H/torch_cache" MPLCONFIGDIR="$H/matplotlib_cache" \
  CUDA_VISIBLE_DEVICES=-1 TRAINER_CARD=M TRAINER_PL=1 TRAINER_F4C=1 TRAINER_THREADS=2 \
  TRAINER_GENERATED_ALLOW_FILE="$H/allow_sq_v1.txt" TRAINER_GENERATED_MIN_N=15990784 \
  TRAINER_RELIEF=1 TRAINER_STAR_ATTACH=0 TRAINER_BIRTH_ATTACH=0 TRAINER_RSS_LIMIT=4000000000 \
  OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 VECLIB_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 BLIS_NUM_THREADS=2 \
  PYTHONPATH="$O/trainer:$H" /Users/matt/anaconda3/bin/python -B "${ENTRY[@]}"
