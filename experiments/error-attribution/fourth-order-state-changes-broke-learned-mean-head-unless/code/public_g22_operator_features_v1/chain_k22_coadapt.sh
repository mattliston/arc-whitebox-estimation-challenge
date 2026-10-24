#!/bin/bash
# TAG g22-k22-coadapt (T-117g): co-adapt the mean head to the engine K22 plane chain: honest oracle rows WITH the chain active
# (generator variant keeps RETRO_K22_CHAIN), retrain the mean head (init HV_F1), export, loop WITH the chain, 1e9 ranking.
D=/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_g22_operator_features_v1; R=/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_g20_emission_retrofit_v1; PY=/Users/matt/arc-white-box-estimation/.venv/bin/python; AP=/Users/matt/anaconda3/bin/python; cd $D
export RETRO_K22_CHAIN=1
./launch_mech15_k22.sh
sleep 120
while [ "$(ls -d rows_meanleak_mech15_k22/row_*/result.json 2>/dev/null | wc -l | tr -d ' ')" -lt 200 ]; do sleep 60; done
while pgrep -f "gen_rows_meanleak_mech15_k22.py [0-9]" > /dev/null; do sleep 20; done
unset RETRO_K22_CHAIN
GNN_FIRST=1 GNN_INIT=GNN_GNN_HV_F1_MODEL.pt GNN_LR=5e-4 GNN_LR_MIN=5e-5 $AP train_gnn_oracle.py $D/rows_meanleak_mech15_k22 GNN_HK22 30 64 2>&1 | grep -v Warning > train_gnn_hk22.log
grep -E "^epoch (10|30)" train_gnn_hk22.log | cut -c1-200
EXPORT_FIRST=1 $AP export_gnn_payload.py GNN_GNN_HK22_MODEL.pt payload_gnn_hk22.npz 2>&1 | grep -v Warning | cut -c1-80
for r in 35 100; do
  RETRO_K22_CHAIN=1 RETRO_CLOSED_LOOP=1 RETRO_CL_MECH=1 RETRO_CL_MECH_FNP=1 RETRO_DUMP_FEAT=1 RETRO_MEANLEAK_OFF=1 RETRO_GNN_PAYLOAD=$D/payload_gnn_hk22.npz \
    TAG=closedloop_HK22 THREADS=4 $PY $R/run_retro_screen.py $r 2>&1 | grep -o '"mse_nc_f2j_off": [0-9.e-]*' | sed "s/^/r$r closedloop_HK22 (mean head co-adapted to the K22 chain) CLOSED LOOP: /"
done
$PY score_vs_truthpost.py closedloop_HK22 closedloop_k22chain closedloop_HV_mix0.0
echo K22COADAPT_DONE
