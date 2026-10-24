#!/bin/bash
# sequential seven-rank judges (box otherwise quiet); usage: run_judges_v1.sh <cand1> [<cand2> ...]
O=/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
for C in "$@"; do
  SHA=$(shasum -a 256 "$C" | cut -c1-64); OUT="${C%.py}_science7"
  echo "$(date -u +%H:%M:%SZ) JUDGE START $(basename $C)"
  /Users/matt/arc-white-box-estimation/.venv/bin/python -B $O/lead/runs/astra_oct18_small_stack_qualification_v1/run_current7_authorized_v1.py "$C" "$SHA" "$OUT" > "${OUT}.log" 2>&1
  echo "$(date -u +%H:%M:%SZ) JUDGE END $(basename $C) rc=$?"; grep 'MEAN of 7' "${OUT}.log" | tail -1
done
echo ALL_JUDGES_DONE
