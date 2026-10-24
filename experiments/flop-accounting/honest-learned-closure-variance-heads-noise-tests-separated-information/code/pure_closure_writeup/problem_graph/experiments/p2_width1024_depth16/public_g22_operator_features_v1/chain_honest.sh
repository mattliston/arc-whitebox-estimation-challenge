#!/bin/bash
# TAG g22-chain-honest (T-116g): when the honest oracle rows complete, train the learned closure on them (no leak), export, closed loop r35/r100, every-layer profile
D=/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_g22_operator_features_v1; R=/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_g20_emission_retrofit_v1; cd $D
while [ "$(ls -d rows_meanleak_mech15b/row_*/result.json 2>/dev/null | wc -l | tr -d ' ')" -lt 200 ]; do sleep 60; done
while pgrep -f "gen_rows_meanleak_mech15.py [0-9]" > /dev/null; do sleep 20; done
/Users/matt/anaconda3/bin/python train_gnn_oracle.py $D/rows_meanleak_mech15b GNN_H_C64_E40 40 64 2>&1 | grep -v Warning > train_gnn_h.log
grep -E "^epoch (10|20|30|40)" train_gnn_h.log | cut -c1-200
/Users/matt/anaconda3/bin/python export_gnn_payload.py GNN_GNN_H_C64_E40_MODEL.pt payload_gnn_h.npz 2>&1 | grep -v Warning | cut -c1-100
for r in 35 100; do
  RETRO_CLOSED_LOOP=1 RETRO_CL_MECH=1 RETRO_CL_MECH_FNP=1 RETRO_DUMP_FEAT=1 RETRO_MEANLEAK_OFF=1 RETRO_GNN_PAYLOAD=$D/payload_gnn_h.npz \
    TAG=closedloop_H THREADS=4 /Users/matt/arc-white-box-estimation/.venv/bin/python $R/run_retro_screen.py $r 2>&1 | grep -o '"mse_nc_f2j_off": [0-9.e-]*' | sed "s/^/r$r closedloop_H (honest rows) CLOSED LOOP: /"
done
echo "== H loop profile r35"; THREADS=3 /Users/matt/arc-white-box-estimation/.venv/bin/python closedloop_layers_gnn.py 35 $D/payload_gnn_h.npz none 2>&1 | grep -v Warning | grep carried
echo HONEST_DONE
