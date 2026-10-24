#!/bin/bash
# TAG opus-reexam E33: per-unit k4 += (6 b / n) (C^2)_ii (the C^2 part of the norm coupling), held-out ranks, 4 threads, nice 15, starts only at load < 14.
cd /Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1
mkdir -p e33_c2k; B="f4=0.6,age=2,c=0.75"
for r in "$@"; do
  f=e33_c2k/rank$(printf %04d $r).log
  [ -s $f ] && grep -q '"arm": "b050"' $f && continue
  until [ "$(uptime | sed 's/.*load averages: //' | cut -d' ' -f1 | cut -d. -f1)" -lt 14 ]; do sleep 20; done
  OMP_NUM_THREADS=4 nice -n 15 /Users/matt/anaconda3/bin/python -B e16b_k4_closed_loop.py $r 4 e33 S37:$B b012:$B,c2k=0.125 b025:$B,c2k=0.25 b050:$B,c2k=0.5 > $f 2>&1
done
echo E33_DONE > e33_c2k/done_$1.txt
