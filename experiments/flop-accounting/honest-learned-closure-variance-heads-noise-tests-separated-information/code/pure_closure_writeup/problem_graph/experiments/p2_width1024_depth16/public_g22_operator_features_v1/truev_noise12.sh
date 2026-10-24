#!/bin/bash
# TAG g22-truev-noise12 (T-116v): TRUE v at half, layers 12-15 only, without noise and with iid relative noise 0.1% / 0.3% (paired), r35
D=/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_g22_operator_features_v1; R=/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_g20_emission_retrofit_v1; PY=/Users/matt/arc-white-box-estimation/.venv/bin/python; cd $D
for NZ in 0 0.001 0.003; do
  RETRO_CLOSED_LOOP=1 RETRO_CL_MECH=1 RETRO_CL_MECH_FNP=1 RETRO_FORCE=v RETRO_FORCE_LAYERS=12-15 RETRO_FORCE_TRUTH=/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_g19_attribution_v1/mps_truth_rank035_P32M.npz RETRO_FORCE_MIX=0.5 RETRO_FORCE_NOISE=v:$NZ RETRO_DUMP_FEAT=1 RETRO_MEANLEAK_OFF=1 RETRO_GNN_PAYLOAD=$D/payload_gnn_hv_f1.npz \
    TAG=closedloop_truev12_half_noise$NZ THREADS=4 $PY $R/run_retro_screen.py 35 2>&1 | grep -o '"mse_nc_f2j_off": [0-9.e-]*' | sed "s/^/r35 TRUE v half at layers 12-15 + iid noise $NZ CLOSED LOOP: /"
done
$PY score_vs_truthpost.py closedloop_truev12_half_noise0 closedloop_truev12_half_noise0.001 closedloop_truev12_half_noise0.003 closedloop_HV_mix0.0
echo TRUEVNOISE12_DONE
