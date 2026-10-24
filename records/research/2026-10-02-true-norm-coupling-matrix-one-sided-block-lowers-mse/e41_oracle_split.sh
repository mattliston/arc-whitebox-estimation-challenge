#!/bin/bash
# TAG opus-reexam E41: which part of the TRUE network-specific norm-coupling matrix carries the K31 gain? Oracle arms on the lead's base card.
# usage: e41_oracle_split.sh <rank> ...   (4 threads, nice 15, starts at load < ${GATE:-22})
cd /Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1
mkdir -p e41_oracle_split; B="f4=0.6,age=2,c=0.75"
for r in "$@"; do
  f=e41_oracle_split/rank$(printf %04d $r).log
  until [ "$(uptime | sed 's/.*load averages: //' | cut -d' ' -f1 | cut -d. -f1)" -lt ${GATE:-22} ]; do sleep 15; done
  OMP_NUM_THREADS=4 nice -n 15 /Users/matt/anaconda3/bin/python -B e16b_k4_closed_loop.py $r 4 e41 S37:$B whole:$B,ora=1 one16:$B,ora=1,orapart=1 two16:$B,ora=1,orapart=2 comp16:$B,ora=1,orapart=3 \
     one16no1:$B,ora=1,orapart=4 mode1:$B,ora=1,orapart=5 one64:$B,ora=1,orapart=1,orak=64 comp64:$B,ora=1,orapart=3,orak=64 r32:$B,ora=1,orarank=32 > $f 2>&1
done
echo done >> e41_oracle_split/done.txt
