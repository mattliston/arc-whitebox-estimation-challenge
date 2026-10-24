#!/bin/bash
# TAG g22-plane-in-loop (T-117): a DETERMINISTIC structured supply for the variance channel - the frozen per-layer fourth-order plane scale
# (T-71 table, K22/K31, beta and ratio modes, zero cost) inside the honest closed loop (HV mean head), r35/r100, 1e9 ranking.
D=/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_g22_operator_features_v1; R=/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_g20_emission_retrofit_v1; PY=/Users/matt/arc-white-box-estimation/.venv/bin/python; cd $D
for MODE in beta ratio; do for r in 35 100; do
  RETRO_CLOSED_LOOP=1 RETRO_CL_MECH=1 RETRO_CL_MECH_FNP=1 RETRO_PLANE_SCALE=/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_g29_bond_kernel_v1/fourth_plane_scale_table.json RETRO_PLANE_SCALE_WHICH=K22,K31 RETRO_PLANE_SCALE_MODE=$MODE RETRO_DUMP_FEAT=1 RETRO_MEANLEAK_OFF=1 RETRO_GNN_PAYLOAD=$D/payload_gnn_hv_f1.npz \
    TAG=closedloop_planescale_$MODE THREADS=4 $PY $R/run_retro_screen.py $r 2>&1 | grep -o '"mse_nc_f2j_off": [0-9.e-]*' | sed "s/^/r$r honest loop + plane scale ($MODE) CLOSED LOOP: /"
done; done
$PY score_vs_truthpost.py closedloop_planescale_beta closedloop_planescale_ratio closedloop_HV_mix0.0
echo PLANE_DONE
