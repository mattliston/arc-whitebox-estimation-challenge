#!/bin/bash
# TAG opus-reexam E16g: CHECK SET nobody tunes on = the analyst's generated fresh networks (seeds 7100000+, 64M-sample truths).
# One process, 2 threads, nice 15. Arms: base, F4 0.4, F4 0.6, F4 0.6 + K4L, F4 0.8 + K4L (age <= 2, c = 1).
cd /Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1
for k in $(seq 1 40); do
  s=$((7100000 + k)); f=e16g_generated/seed$s.log
  [ -f analyst/fresh/truth_seed$s.npz ] || continue
  [ -s $f ] && grep -q '"arm": "F8K4"' $f && continue
  nice -n 15 /Users/matt/anaconda3/bin/python -B e16b_k4_closed_loop.py $s 2 gen base:c=0 F4:f4=0.4 F6:f4=0.6 F6K4:f4=0.6,c=1,age=2 F8K4:f4=0.8,c=1,age=2 > $f 2>&1
done
echo E16G_DONE > e16g_generated/done_$(date -u +%H%M).txt
