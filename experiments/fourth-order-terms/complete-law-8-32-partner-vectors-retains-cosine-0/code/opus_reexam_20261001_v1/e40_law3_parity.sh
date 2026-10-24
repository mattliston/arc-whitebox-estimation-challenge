#!/bin/bash
# TAG opus-reexam E40: the complete law on K vectors (O/ncar3.py) against the physicist's research carrier (O/physicist/ncarry.py) on the same base card.
# usage: e40_law3_parity.sh <rank> ...   (4 threads, nice 15, starts at load < ${GATE:-22})
cd /Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1
mkdir -p e40_law3; B="f4=0.6,age=2,c=0.75"
for r in "$@"; do
  f=e40_law3/rank$(printf %04d $r).log
  until [ "$(uptime | sed 's/.*load averages: //' | cut -d' ' -f1 | cut -d. -f1)" -lt ${GATE:-22} ]; do sleep 15; done
  OMP_NUM_THREADS=4 nice -n 15 /Users/matt/anaconda3/bin/python -B e16b_k4_closed_loop.py $r 4 e40 S37:$B \
     l3K16:$B,nc=1.0,nclaw=3,nck=16,ncdump=1 l3K16full:$B,nc=1.0,nclaw=3,nck=16,ncplane=1 l3K8:$B,nc=1.0,nclaw=3,nck=8 l3K32:$B,nc=1.0,nclaw=3,nck=32 \
     phP16:$B,ncy=1.0,ncyp=16,ncyforms=1 phFull:$B,ncy=1.0,ncyp=0,ncyforms=4 > $f 2>&1
done
echo done >> e40_law3/done.txt
