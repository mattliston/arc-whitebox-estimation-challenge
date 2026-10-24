#!/bin/bash
# TAG g22-chain-f1 (T-116f): after the FIRST=1 training, export (first=1) and run the closed loop on r35/r100
D=/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_g22_operator_features_v1; R=/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_g20_emission_retrofit_v1; cd $D
until grep -q '"tag"\|Traceback' train_gnn_m15_f1.log 2>/dev/null; do sleep 30; done
grep -E "^epoch 40" train_gnn_m15_f1.log | cut -c1-200
EXPORT_FIRST=1 /Users/matt/anaconda3/bin/python export_gnn_payload.py GNN_GNN_M15_F1_MODEL.pt payload_gnn_m15_f1.npz 2>&1 | grep -v Warning | cut -c1-100
for r in 35 100; do
  RETRO_CLOSED_LOOP=1 RETRO_CLOSED_LOOP_ROWS=pre RETRO_CL_MECH=1 RETRO_CL_MECH_FNP=1 RETRO_DUMP_FEAT=1 RETRO_MEANLEAK_OFF=1 RETRO_GNN_PAYLOAD=$D/payload_gnn_m15_f1.npz \
    TAG=closedloop_F1 THREADS=4 /Users/matt/arc-white-box-estimation/.venv/bin/python $R/run_retro_screen.py $r 2>&1 | grep -o '"mse_nc_f2j_off": [0-9.e-]*' | sed "s/^/r$r closedloop_F1 (first corrected layer 1) CLOSED LOOP: /"
done
echo F1_DONE
