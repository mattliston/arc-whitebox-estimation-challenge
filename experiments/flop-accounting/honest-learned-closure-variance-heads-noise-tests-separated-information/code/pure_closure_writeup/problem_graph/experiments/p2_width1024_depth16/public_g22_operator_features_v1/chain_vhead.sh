#!/bin/bash
# TAG g22-chain-vhead (T-116o): when the 1M-sample variance labels exist for all 200 nets, train the learned closure WITH the next-layer
# variance head (GNN_HV, FIRST=1), export (v-head included), and run the loop on r35/r100 with the variance correction at mix 0 (control),
# 0.5 and 1.0; rank in the 1e9 frame.
D=/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_g22_operator_features_v1; R=/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_g20_emission_retrofit_v1; PY=/Users/matt/arc-white-box-estimation/.venv/bin/python; AP=/Users/matt/anaconda3/bin/python; cd $D
until grep -q "VLABELS_DONE" run_vlabels.log 2>/dev/null; do sleep 120; done
GNN_FIRST=1 GNN_VLABELS=$D/vlabels GNN_VLOSS=1.0 $AP train_gnn_oracle.py $D/rows_meanleak_mech15b GNN_HV_F1 40 64 2>&1 | grep -v Warning > train_gnn_hv.log
grep -E "^epoch (20|40)|variance head" train_gnn_hv.log | cut -c1-220
EXPORT_FIRST=1 $AP export_gnn_payload.py GNN_GNN_HV_F1_MODEL.pt payload_gnn_hv_f1.npz 2>&1 | grep -v Warning | cut -c1-120
for MIX in 0.0 0.5 1.0; do
  for r in 35 100; do
    RETRO_CLOSED_LOOP=1 RETRO_CL_MECH=1 RETRO_CL_MECH_FNP=1 RETRO_CL_VHEAD=1 RETRO_CL_VHEAD_MIX=$MIX RETRO_DUMP_FEAT=1 RETRO_MEANLEAK_OFF=1 RETRO_GNN_PAYLOAD=$D/payload_gnn_hv_f1.npz \
      TAG=closedloop_HV_mix$MIX THREADS=4 $PY $R/run_retro_screen.py $r 2>&1 | grep -o '"mse_nc_f2j_off": [0-9.e-]*' | sed "s/^/r$r closedloop_HV variance mix $MIX CLOSED LOOP: /"
  done
done
$PY score_vs_truthpost.py closedloop_HV_mix0.0 closedloop_HV_mix0.5 closedloop_HV_mix1.0 closedloop_HF1
echo VHEAD_DONE
