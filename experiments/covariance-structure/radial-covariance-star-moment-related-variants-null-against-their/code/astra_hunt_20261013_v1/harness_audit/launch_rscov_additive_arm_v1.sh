#!/bin/bash
# Usage: bash launch_rscov_additive_arm_v1.sh /absolute/fresh/guard.json /absolute/completed/control.comparison.json [seconds]
set -euo pipefail
A=/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_hunt_20261013_v1
O=/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1
H=$A/harness_audit
RECORD=$1
CONTROL_COMPARISON=$2
SECONDS_LIMIT=${3:-7200}
case "$RECORD" in "$H"/*) ;; *) exit 11 ;; esac
case "$CONTROL_COMPARISON" in "$H"/*) ;; *) exit 12 ;; esac
test ! -e "$RECORD"
test -f "$CONTROL_COMPARISON"
FREE=$(/usr/bin/memory_pressure | /usr/bin/awk '/System-wide memory free percentage/ {gsub(/%/, "", $NF); print $NF}')
test "$FREE" -ge 50
mkdir -p "$H/tmp_rscov_additive_arm_v1" "$H/cache_rscov_additive_arm_v1" "$H/torch_cache_rscov_additive_arm_v1" "$H/matplotlib_cache_rscov_additive_arm_v1"
cd "$H"
exec /usr/bin/nice -n 10 /Users/matt/arc-white-box-estimation/.venv/bin/python -B "$O/release/guard_footprint_v3.py" \
  --record "$RECORD" --seconds "$SECONDS_LIMIT" --footprint-limit 4000000000 --rss-backstop 4000000000 \
  /usr/bin/env -u TRAINER_TRUTH_PLAN -u TRAINER_TRUTH_PLAN_SHA256 \
  PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX="$H/cache_rscov_additive_arm_v1" TMPDIR="$H/tmp_rscov_additive_arm_v1" \
  ASTRA_PRESSURE_RECORD="$RECORD.pressure.jsonl" \
  XDG_CACHE_HOME="$H/cache_rscov_additive_arm_v1" TORCH_HOME="$H/torch_cache_rscov_additive_arm_v1" MPLCONFIGDIR="$H/matplotlib_cache_rscov_additive_arm_v1" \
  CUDA_VISIBLE_DEVICES=-1 TRAINER_CARD=M TRAINER_PL=1 TRAINER_F4C=1 TRAINER_THREADS=2 \
  TRAINER_GENERATED_ALLOW_FILE="$H/allow_rscov_v1.txt" TRAINER_GENERATED_MIN_N=15990784 \
  TRAINER_RELIEF=1 TRAINER_STAR_ATTACH=0 TRAINER_BIRTH_ATTACH=0 TRAINER_RSS_LIMIT=4000000000 \
  OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 VECLIB_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 BLIS_NUM_THREADS=2 \
  PYTHONPATH="$O/trainer:$H" /Users/matt/anaconda3/bin/python -B "$H/screen_entry_rscov_additive_arm_v1.py" \
  --control-comparison "$CONTROL_COMPARISON" --artifact "$RECORD.comparison.json"
