#!/bin/bash
# TAG opus_engineer2 a62 preflight_gk_v2 (file written 2026-10-12T22:28:07Z; lead 21:5xZ guard) = preflight_gk_v1.sh + the port loop v11's config guard
# (gk_guard_v1.py <N> <config.json>, its G1): as soon as round N's config.json and its first a_RS<N>_base_r*.json torch record exist, hours
# before the verdict, it runs the reader the port uses (gk_from_config_v1.py) and the guard, and posts one [engineer2] line with both answers
# (either one failing would stop the unattended port at port time). Gives up after 30 h. usage: preflight_gk_v2.sh <round N>
# v1 header: (file written 2026-10-12T22:18:33Z; lead 21:4xZ task 3): the port loop v10 reads round N's gk from its config at
# port time (after the verdict); this runs the same reader (gk_from_config_v1.py, text only) as soon as round N's config.json and its first
# a_RS<N>_base_r*.json torch record exist, hours before the verdict, and posts one [engineer2] line with the answer (a STOP there would
# otherwise stop the unattended port). Gives up after 30 h. usage: preflight_gk_v1.sh <round N>
set -u
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
O=/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1
A=$O/engineer/a62_modules; PY=/Users/matt/arc-white-box-estimation/.venv/bin/python; N=$1
C=$O/trainer/cert/RAD/S$N/config.json; R=$O/trainer/run_RAD_S${N}_gen/results; t0=$(date +%s)
say() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }
say "waiting for S$N's config and first base record"
until [ -f $C ] && ls $R/a_RS${N}_base_r*.json > /dev/null 2>&1; do
  [ $(( $(date +%s) - t0 )) -gt 108000 ] && { say "gave up after 30 h"; exit 1; }
  sleep 300
done
out=$($PY -B $A/gk_from_config_v1.py $C 2>&1 | tail -1)
g=$($PY -B $A/gk_guard_v1.py $N $C 2>&1 | tail -1)
say "$out"; say "$g"
echo "[engineer2] $(date -u +%FT%TZ) S$N GK PREFLIGHT (the port loop v11 runs both at port time; a62_modules/preflight_gk_v2.sh): reader: $(echo "$out" | sed "s|$A/||"); guard: $g" >> $O/TEAM_LOG.md
