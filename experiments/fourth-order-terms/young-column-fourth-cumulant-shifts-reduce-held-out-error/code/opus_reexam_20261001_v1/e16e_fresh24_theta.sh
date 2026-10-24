#!/bin/bash
# TAG opus-reexam E16e: strength scan of the physicist's F4 on the 24 held-out newer ranks (after E16c; base and theta 0.4 are in e16c_fresh24/).
# One process, 3 threads, nice 15. Arms: base (parity with E16c), F4 theta 0.6, F4 theta 0.8.
cd /Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1
until [ -f e16c_fresh24/done.txt ]; do sleep 20; done
mkdir -p e16e_fresh24_theta
for r in 651 378 424 328 287 641 548 557 316 390 293 277 320 312 550 672 333 310 530 259 623 459 16 473; do
  f=e16e_fresh24_theta/rank$(printf %04d $r).log
  [ -s $f ] && grep -q '"arm": "F4t08"' $f && continue
  nice -n 15 /Users/matt/anaconda3/bin/python -B e16b_k4_closed_loop.py $r 3 fresh24theta base:c=0 F4t06:f4=0.6 F4t08:f4=0.8 > $f 2>&1
done
echo E16E_DONE > e16e_fresh24_theta/done.txt
