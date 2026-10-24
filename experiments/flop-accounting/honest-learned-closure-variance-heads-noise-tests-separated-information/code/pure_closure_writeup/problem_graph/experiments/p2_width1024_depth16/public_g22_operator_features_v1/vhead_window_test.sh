#!/bin/bash
# TAG g22-vhead-window (T-116r): HVAGG head applied only at layers 12-15 (where it is 56-65% accurate), mix 0.5 and 1.0, r35/r100; 1e9 ranking
D=/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_g22_operator_features_v1; R=/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_g20_emission_retrofit_v1; PY=/Users/matt/arc-white-box-estimation/.venv/bin/python; cd $D
for MIX in 0.5 1.0; do for r in 35 100; do
  RETRO_CLOSED_LOOP=1 RETRO_CL_MECH=1 RETRO_CL_MECH_FNP=1 RETRO_CL_VHEAD=1 RETRO_CL_VHEAD_MIX=$MIX RETRO_CL_VHEAD_LAYERS=12-15 RETRO_DUMP_FEAT=1 RETRO_MEANLEAK_OFF=1 RETRO_GNN_PAYLOAD=$D/payload_gnn_hvagg.npz \
    TAG=closedloop_HVAGG_w12_mix$MIX THREADS=4 $PY $R/run_retro_screen.py $r 2>&1 | grep -o '"mse_nc_f2j_off": [0-9.e-]*' | sed "s/^/r$r HVAGG head at layers 12-15 only, mix $MIX CLOSED LOOP: /"
done; done
$PY score_vs_truthpost.py closedloop_HVAGG_w12_mix0.5 closedloop_HVAGG_w12_mix1.0 closedloop_HV_mix0.0 closedloop_HF1
echo VHEAD_WINDOW_DONE
