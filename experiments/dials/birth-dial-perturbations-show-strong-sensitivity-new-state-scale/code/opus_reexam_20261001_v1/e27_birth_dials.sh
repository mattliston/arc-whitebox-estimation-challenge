#!/bin/bash
# TAG opus-reexam E27: newborn-weight dials on the S37 card (F4 0.6 + K4L), held-out newer ranks, 4 threads, nice 15.
# Arms: S37; s_new -0.01 / -0.02; X1 +0.05 / +0.10; birth coefficient d (1 + x radial / 3) with x = 0 and 0.5 (card: 1).
cd /Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1
mkdir -p e27_birth_dials; B="f4=0.6,age=2,c=1"
for r in "$@"; do
  f=e27_birth_dials/rank$(printf %04d $r).log
  [ -s $f ] && grep -q '"arm": "br05"' $f && continue
  nice -n 15 /Users/matt/anaconda3/bin/python -B e16b_k4_closed_loop.py $r 4 e27 S37:$B sn1:$B,d_s_new=-0.01 sn2:$B,d_s_new=-0.02 x1a:$B,d_X1=0.05 x1b:$B,d_X1=0.1 br0:$B,br=0 br05:$B,br=0.5 > $f 2>&1
done
echo E27_DONE > e27_birth_dials/done_$1.txt
