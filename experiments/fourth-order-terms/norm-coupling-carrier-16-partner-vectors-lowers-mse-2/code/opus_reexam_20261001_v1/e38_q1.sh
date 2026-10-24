#!/bin/bash
# TAG opus-reexam E38: mode 1 of the carrier from the bank's trace vector (ncq1), closed loop on network 35 and held-out 651; 4 threads, load-gated.
cd /Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1
mkdir -p e38_q1; B="f4=0.6,age=2,c=0.75"
for r in "$@"; do
  f=e38_q1/rank$(printf %04d $r).log
  until [ "$(uptime | sed 's/.*load averages: //' | cut -d' ' -f1 | cut -d. -f1)" -lt 14 ]; do sleep 15; done
  OMP_NUM_THREADS=4 nice -n 15 /Users/matt/anaconda3/bin/python -B e16b_k4_closed_loop.py $r 4 e38 S37:$B k10:$B,nc=1.0 q05:$B,nc=1.0,ncq1=0.4 q075:$B,nc=1.0,ncq1=0.75 q10:$B,nc=1.0,ncq1=1.0 q075d:$B,nc=1.0,ncq1=0.75,ncdump=1 > $f 2>&1
done
echo done > e38_q1/done.txt
