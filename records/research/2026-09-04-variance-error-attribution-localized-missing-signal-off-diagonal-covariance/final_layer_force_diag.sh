#!/bin/bash
# TAG g22-final-force-diag (T-115c): oracle recursion (exact upstream means, consistent covariance) + truth state forced at the FINAL layer only
R=/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_g20_emission_retrofit_v1; D=/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_g22_operator_features_v1
for F in none v k3 k4 vk3k4; do
  RETRO_FORCE=$F RETRO_FORCE_LAYERS=15-15 RETRO_FORCE_TRUTH=/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_g19_attribution_v1/mps_truth_rank035_P32M.npz RETRO_MEANLEAK_TRUTH=$D/truthpost/truthpost_rank0035.npz RETRO_MEANLEAK_MEAN=1 RETRO_MEANLEAK_ROWS=pre \
    TAG=finalforce_$F THREADS=4 /Users/matt/arc-white-box-estimation/.venv/bin/python $R/run_retro_screen.py 35 2>&1 | grep -o '"mse_nc_f2j_off": [0-9.e-]*' | sed "s/^/r35 oracle-recursion + truth $F at layer 15 only: /"
done
echo FINALFORCE_DONE
