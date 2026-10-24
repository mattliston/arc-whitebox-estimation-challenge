#!/bin/bash
# TAG g22-k22chain-in-loop (T-117b): the engine's fourth-order plane chain IN-CARD (RETRO_K22_CHAIN=1, GEMM-class cost) inside the honest closed loop, r35/r100; raw first, util noted
D=/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_g22_operator_features_v1; R=/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_g20_emission_retrofit_v1; PY=/Users/matt/arc-white-box-estimation/.venv/bin/python; cd $D
until grep -q PLANE_DONE plane_in_loop.log 2>/dev/null; do sleep 60; done
for r in 35 100; do
  RETRO_CLOSED_LOOP=1 RETRO_CL_MECH=1 RETRO_CL_MECH_FNP=1 RETRO_K22_CHAIN=1 RETRO_DUMP_FEAT=1 RETRO_MEANLEAK_OFF=1 RETRO_GNN_PAYLOAD=$D/payload_gnn_hv_f1.npz \
    TAG=closedloop_k22chain THREADS=4 $PY $R/run_retro_screen.py $r 2>&1 | grep -o '"u_local": [0-9.]*\|"mse_nc_f2j_off": [0-9.e-]*' | tr '\n' ' ' | sed "s/^/r$r honest loop + engine K22 chain in-card CLOSED LOOP: /"; echo
done
$PY score_vs_truthpost.py closedloop_k22chain closedloop_planescale_beta closedloop_HV_mix0.0
echo K22CHAIN_DONE
