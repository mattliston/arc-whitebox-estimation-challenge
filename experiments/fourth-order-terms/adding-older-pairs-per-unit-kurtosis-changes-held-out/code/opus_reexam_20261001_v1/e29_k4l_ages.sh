#!/bin/bash
# TAG opus-reexam E29: per-unit kurtosis term with older young pairs (ages 3-4), held-out ranks, 4 threads, nice 15.
# Base: F4 0.6 + K4L ages 1-2 at 0.75. Arms: ages 3-4 at relative weight 0.5 / 1.0 at the FINAL layer only; 0.3 / 0.6 at all layers.
cd /Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1
mkdir -p e29_k4l_ages; B="f4=0.6,age=2,c=0.75"
for r in "$@"; do
  f=e29_k4l_ages/rank$(printf %04d $r).log
  [ -s $f ] && grep -q '"arm": "all06"' $f && continue
  nice -n 15 /Users/matt/anaconda3/bin/python -B e16b_k4_closed_loop.py $r 4 e29 S37:$B fin05:$B,a3=0.5,a3min=15 fin10:$B,a3=1.0,a3min=15 all03:$B,a3=0.3 all06:$B,a3=0.6 > $f 2>&1
done
echo E29_DONE > e29_k4l_ages/done_$1.txt
