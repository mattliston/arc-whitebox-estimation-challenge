#!/bin/bash
# TAG opus-reexam E36: the free-running vector carrier (O/ncar.py, law 2) as an extra K31 plane, closed loop, final-layer MSE against the base arm.
# usage: e36_ncar_closed.sh <tag> <rank> [<rank> ...]   (4 threads, nice 15, each network starts at load < 14)
cd /Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1
TAG=$1; shift; mkdir -p e36_ncar_$TAG; B="f4=0.6,age=2,c=0.75"
for r in "$@"; do
  f=e36_ncar_$TAG/rank$(printf %04d $r).log
  [ -s $f ] && grep -q '"arm": "k10m1"' $f && continue
  until [ "$(uptime | sed 's/.*load averages: //' | cut -d' ' -f1 | cut -d. -f1)" -lt 14 ]; do sleep 15; done
  OMP_NUM_THREADS=4 nice -n 15 /Users/matt/anaconda3/bin/python -B e16b_k4_closed_loop.py $r 4 e36 S37:$B k05:$B,nc=0.5 k10:$B,nc=1.0 k15:$B,nc=1.5 k20:$B,nc=2.0 k10m1:$B,nc=1.0,ncm1=1 > $f 2>&1
done
echo E36_DONE > e36_ncar_$TAG/done.txt
