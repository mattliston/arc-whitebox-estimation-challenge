#!/bin/bash
# TAG opus-reexam E19: depth-dependent selection floors on the S37 card (F4 0.6 + K4L), 6 fresh ranks, one process, 3 threads, nice 15.
cd /Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1
mkdir -p e19_depth
B="f4=0.6,c=1,age=2"
for r in 651 378 424 328 287 641; do
  f=e19_depth/rank$(printf %04d $r).log
  [ -s $f ] && grep -q '"arm": "bfl2"' $f && continue
  nice -n 15 /Users/matt/anaconda3/bin/python -B e19_depth_provisioning.py $r 3 e19 base:$B rfe05:$B,rfe=0.005 rfm2:$B,rfm=0.02 rfl2:$B,rfl=0.02 rfl05:$B,rfl=0.005 bfe05:$B,bfe=0.005 bfl2:$B,bfl=0.02 > $f 2>&1
done
echo E19_DONE > e19_depth/done.txt
