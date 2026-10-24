#!/bin/bash
# TAG opus-reexam E21: no-refit dial values on the S37 card (F4 0.6 + K4L), 24 held-out newer ranks, one process, 4 threads, nice 15.
# Arms: base (card without the two terms, parity with E16c), S37 = F4 0.6 + K4L c 1, + isotropic b source x 0.85, K4L c 0.75, both.
cd /Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1
mkdir -p e21_fresh24_dials
for r in 651 378 424 328 287 641 548 557 316 390 293 277 320 312 550 672 333 310 530 259 623 459 16 473; do
  f=e21_fresh24_dials/rank$(printf %04d $r).log
  [ -s $f ] && grep -q '"arm": "c75bs85"' $f && continue
  nice -n 15 /Users/matt/anaconda3/bin/python -B e16b_k4_closed_loop.py $r 4 e21 base:c=0 S37:f4=0.6,c=1,age=2 bs85:f4=0.6,c=1,age=2,bs=0.85 c75:f4=0.6,c=0.75,age=2 c75bs85:f4=0.6,c=0.75,age=2,bs=0.85 > $f 2>&1
done
echo E21_DONE > e21_fresh24_dials/done.txt
