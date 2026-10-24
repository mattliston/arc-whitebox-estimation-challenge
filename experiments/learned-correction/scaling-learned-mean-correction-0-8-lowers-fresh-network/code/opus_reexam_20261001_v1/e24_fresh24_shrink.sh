#!/bin/bash
# TAG opus-reexam E24: generalisation dials on the S37 card (F4 0.6 + K4L), 24 held-out newer ranks, 4 threads, nice 15.
# Arms: S37; learned mean correction x 0.8 and x 0.7; learned covariance edge off; x 0.5; a combination (correction 0.8, K4L c 0.75, edge off).
cd /Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1
mkdir -p e24_fresh24_shrink; B="f4=0.6,age=2"
for r in 651 378 424 328 287 641 548 557 316 390 293 277 320 312 550 672 333 310 530 259 623 459 16 473; do
  f=e24_fresh24_shrink/rank$(printf %04d $r).log
  [ -s $f ] && grep -q '"arm": "combo"' $f && continue
  nice -n 15 /Users/matt/anaconda3/bin/python -B e16b_k4_closed_loop.py $r 4 e24 S37:$B,c=1 ds08:$B,c=1,ds=0.8 ds07:$B,c=1,ds=0.7 edge0:$B,c=1,edge=0 edge05:$B,c=1,edge=0.5 combo:$B,c=0.75,ds=0.8,edge=0 > $f 2>&1
done
echo E24_DONE > e24_fresh24_shrink/done.txt
