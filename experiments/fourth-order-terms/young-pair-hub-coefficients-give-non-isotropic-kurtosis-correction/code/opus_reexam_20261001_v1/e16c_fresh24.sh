#!/bin/bash
# TAG opus-reexam E16c: K4L (+F4) on the 24 held-out newer ranks of the physicist's fresh24 list (first 24 of O/trainer/cert/runB_val.txt).
# One process at a time, 3 threads, nice 15. Arms: base, F4 (theta 0.4), F4 + K4L age<=2, F4 + K4L age<=4.
cd /Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1
for r in 651 378 424 328 287 641 548 557 316 390 293 277 320 312 550 672 333 310 530 259 623 459 16 473; do
  f=e16c_fresh24/rank$(printf %04d $r).log; mkdir -p e16c_fresh24
  [ -s $f ] && grep -q '"arm": "F4K4a4"' $f && continue
  nice -n 15 /Users/matt/anaconda3/bin/python -B e16b_k4_closed_loop.py $r 3 fresh24 base:c=0 F4:f4=0.4 F4K4a2:f4=0.4,c=1,age=2 F4K4a4:f4=0.4,c=1,age=4 > $f 2>&1
done
echo E16C_DONE > e16c_fresh24/done.txt
