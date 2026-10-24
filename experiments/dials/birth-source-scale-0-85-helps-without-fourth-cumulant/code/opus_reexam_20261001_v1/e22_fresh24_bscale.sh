#!/bin/bash
# TAG opus-reexam E22: the carried-b scale (dial bscale, 1.0383 in the bank) UP, on the S37 card (F4 0.6 + K4L c 1), 24 held-out newer
# ranks (after E21). The trainer found the old refit gradient of this dial had the wrong sign (detached star basis). 4 threads, nice 15.
cd /Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1
mkdir -p e22_fresh24_bscale
for r in 651 378 424 328 287 641 548 557 316 390 293 277 320 312 550 672 333 310 530 259 623 459 16 473; do
  f=e22_fresh24_bscale/rank$(printf %04d $r).log
  [ -s $f ] && grep -q '"arm": "b03"' $f && continue
  nice -n 15 /Users/matt/anaconda3/bin/python -B e16b_k4_closed_loop.py $r 3 e22 S37:f4=0.6,c=1,age=2 b01:f4=0.6,c=1,age=2,d_bscale=0.01 b02:f4=0.6,c=1,age=2,d_bscale=0.02 b03:f4=0.6,c=1,age=2,d_bscale=0.03 > $f 2>&1
done
echo E22_DONE > e22_fresh24_bscale/done.txt
