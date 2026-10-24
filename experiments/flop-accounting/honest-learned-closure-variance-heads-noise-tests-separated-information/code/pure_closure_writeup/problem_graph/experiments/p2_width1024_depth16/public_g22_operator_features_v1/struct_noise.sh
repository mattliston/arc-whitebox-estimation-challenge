#!/bin/bash
# TAG g22-struct-noise (T-116y): does the loop tolerate STRUCTURED (mean-aligned rank-2) covariance noise better than iid diagonal noise?
# (A) iid diagonal relative noise 0.1% / 0.3% at layers 12-15, no truth (RETRO_FORCE=v with MIX=1.0 keeps the card's v);
# (B) rank-2 random factor a = s N(0,1) at the post side of layers 11-14 with s = 8e-5 / 2.5e-4 (induced relative dv ~0.1% / ~0.3%). r35, 1e9 ranking.
D=/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_g22_operator_features_v1; R=/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_g20_emission_retrofit_v1; PY=/Users/matt/arc-white-box-estimation/.venv/bin/python; cd $D
for NZ in 0.001 0.003; do
  RETRO_CLOSED_LOOP=1 RETRO_CL_MECH=1 RETRO_CL_MECH_FNP=1 RETRO_FORCE=v RETRO_FORCE_LAYERS=12-15 RETRO_FORCE_TRUTH=/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_g19_attribution_v1/mps_truth_rank035_P32M.npz RETRO_FORCE_MIX=1.0 RETRO_FORCE_NOISE=v:$NZ RETRO_DUMP_FEAT=1 RETRO_MEANLEAK_OFF=1 RETRO_GNN_PAYLOAD=$D/payload_gnn_hv_f1.npz \
    TAG=closedloop_diagnoise$NZ THREADS=4 $PY $R/run_retro_screen.py 35 2>&1 | grep -o '"mse_nc_f2j_off": [0-9.e-]*' | sed "s/^/r35 iid DIAGONAL v noise $NZ at 12-15 (no truth) CLOSED LOOP: /"
done
for S in 8e-5 2.5e-4; do
  RETRO_CLOSED_LOOP=1 RETRO_CL_MECH=1 RETRO_CL_MECH_FNP=1 RETRO_CL_ACOV_NOISE=$S RETRO_CL_VHEAD_LAYERS=12-15 RETRO_DUMP_FEAT=1 RETRO_MEANLEAK_OFF=1 RETRO_GNN_PAYLOAD=$D/payload_gnn_hv_f1.npz \
    TAG=closedloop_structnoise$S THREADS=4 $PY $R/run_retro_screen.py 35 2>&1 | grep -o '"mse_nc_f2j_off": [0-9.e-]*' | sed "s/^/r35 STRUCTURED rank-2 noise a=$S (pre 12-15) CLOSED LOOP: /"
done
$PY score_vs_truthpost.py closedloop_diagnoise0.001 closedloop_diagnoise0.003 closedloop_structnoise8e-5 closedloop_structnoise2.5e-4 closedloop_HV_mix0.0
echo STRUCTNOISE_DONE
