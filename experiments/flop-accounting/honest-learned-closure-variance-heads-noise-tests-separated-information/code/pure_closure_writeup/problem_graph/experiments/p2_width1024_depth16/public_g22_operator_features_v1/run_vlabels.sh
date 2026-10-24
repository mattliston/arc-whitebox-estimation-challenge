#!/bin/bash
# TAG g22-vlabels (T-116o): per-unit univariate truths (pre v, k3, k4, post mean/var) for all 200 observer_fit nets by MC at 1M samples on MPS
# (public_g19_attribution_v1/mc_truth_mps.py; ~15 s/net; rel noise of v ~0.14% vs the 0.5-0.85% card error) -> vlabels/mps_truth_rank{r}_P1M.npz
D=/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_g22_operator_features_v1; cd $D
for r in $(seq 35 234); do
  f=vlabels/mps_truth_rank$(printf %04d $r)_P1M.npz
  [ -f $f ] && continue
  nice -n 5 /Users/matt/anaconda3/bin/python /Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_g19_attribution_v1/mc_truth_mps.py $r 1 $f 2>&1 | grep -v Warning | tail -1 | cut -c1-160
done
echo VLABELS_DONE
