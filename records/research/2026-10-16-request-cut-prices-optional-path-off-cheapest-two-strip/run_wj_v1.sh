#!/bin/bash
# [opus-wallx] warm_judge (warm-up 7103248 -> target 7103256), serial per lane, guard v3, nice 10, BLAS 1. usage: run_wj_v1.sh TAG...
O=/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1
D=$O/lead/runs/opus_lowres_v1; NPY=$O/../astra_phase2_reexamination_v1/openblas_runtime_env_v1/bin/python
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 VECLIB_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 BLIS_NUM_THREADS=1 PB_ALL=1 PYTHONDONTWRITEBYTECODE=1
mkdir -p $D/wj
for tag in "$@"; do
  cand=$D/cand_lowres_${tag}_v1.py; out=$D/wj/${tag}_7103256.json; [ -e $out ] && continue
  while true; do f=$(memory_pressure | awk '/free percentage/{gsub("%","",$NF);print $NF}'); h=$(ps -axo rss=,comm= | awk '$1>1048576 && /[Pp]ython/' | wc -l | tr -d ' '); [ "${f:-0}" -ge 60 ] && [ "$h" -lt 7 ] && break; sleep 20; done
  export CANDIDATE_SHA256=$(shasum -a 256 $cand | cut -d' ' -f1)
  /Users/matt/arc-white-box-estimation/.venv/bin/python -B $O/release/guard_footprint_v3.py --record $D/wj/${tag}_guard.json --seconds 900 --footprint-limit 7500000000 --rss-backstop 10000000000 nice -n 10 $NPY -B $O/lead/runs/opus_pushsm_v1/warm_judge_v1.py $cand 7103248 7103256 $out > $D/wj/${tag}.log 2>&1
  echo "$(date -u +%H:%M:%SZ) wj $tag exit=$?" >> $D/wj/jobs.log
done
