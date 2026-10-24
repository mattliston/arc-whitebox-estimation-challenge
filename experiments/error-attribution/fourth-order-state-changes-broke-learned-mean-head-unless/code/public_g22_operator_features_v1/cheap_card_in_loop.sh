#!/bin/bash
# TAG g22-cheap-card (T-118): utilization is the structural gap to the competitor (0.61 vs 0.22). The honest loop's mean head compensates errors
# the card makes; test a CHEAPER card inside the loop (CP3 cap halved / quartered, then CP4 cap), un-adapted head first: raw AND util (adjusted = raw x util).
D=/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_g22_operator_features_v1; R=/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_g20_emission_retrofit_v1; PY=/Users/matt/arc-white-box-estimation/.venv/bin/python; cd $D
until grep -q K4RESCALE_DONE k4rescale_in_loop.log 2>/dev/null; do sleep 60; done
for CAP in 16384 8192; do
  CP3_CAP=$CAP RETRO_CLOSED_LOOP=1 RETRO_CL_MECH=1 RETRO_CL_MECH_FNP=1 RETRO_DUMP_FEAT=1 RETRO_MEANLEAK_OFF=1 RETRO_GNN_PAYLOAD=$D/payload_gnn_hv_f1.npz \
    TAG=closedloop_cp3cap$CAP THREADS=4 $PY $R/run_retro_screen.py 35 2>&1 | grep -o '"u_local": [0-9.]*\|"mse_nc_f2j_off": [0-9.e-]*\|"dep_mse_nc": [0-9.e-]*' | tr '\n' ' ' | sed "s/^/r35 honest loop with CP3 cap $CAP: /"; echo
done
for CAP in 1024 512; do
  CP4_CAP=$CAP RETRO_CLOSED_LOOP=1 RETRO_CL_MECH=1 RETRO_CL_MECH_FNP=1 RETRO_DUMP_FEAT=1 RETRO_MEANLEAK_OFF=1 RETRO_GNN_PAYLOAD=$D/payload_gnn_hv_f1.npz \
    TAG=closedloop_cp4cap$CAP THREADS=4 $PY $R/run_retro_screen.py 35 2>&1 | grep -o '"u_local": [0-9.]*\|"mse_nc_f2j_off": [0-9.e-]*' | tr '\n' ' ' | sed "s/^/r35 honest loop with CP4 cap $CAP: /"; echo
done
$PY score_vs_truthpost.py closedloop_cp3cap16384 closedloop_cp3cap8192 closedloop_cp4cap1024 closedloop_cp4cap512 closedloop_HV_mix0.0
echo CHEAP_DONE
