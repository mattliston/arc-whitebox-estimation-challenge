#!/bin/bash
# TAG g22-chain-vhead-bil (T-116q): variance head with 8 bilinear pushed-product channels (the mean-aligned low-rank covariance-residual
# structure, T-100/T-115j) + the wider MLP head, loop-consistent labels (aggregate incl. alpha-0 rows), init from HV_F1; export; loop at
# mix 0.5 and 1.0; rank in the 1e9 frame.
D=/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_g22_operator_features_v1; R=/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_g20_emission_retrofit_v1; PY=/Users/matt/arc-white-box-estimation/.venv/bin/python; AP=/Users/matt/anaconda3/bin/python; cd $D
GNN_FIRST=1 GNN_VLABELS=$D/vlabels GNN_VLOSS=1.0 GNN_VCM=48 GNN_VHID=128 GNN_VBILIN=8 GNN_EXTRA_ROWS=$D/rows_gnnmixed_h0,$D/rows_gnnmixed_h GNN_INIT=GNN_GNN_HV_F1_MODEL.pt GNN_LR=5e-4 GNN_LR_MIN=5e-5 \
  $AP train_gnn_oracle.py $D/rows_meanleak_mech15b GNN_HVBIL 30 64 2>&1 | grep -v Warning > train_gnn_hvbil.log
grep -E "^epoch (10|20|30)|variance head" train_gnn_hvbil.log | tail -4 | cut -c1-220
EXPORT_FIRST=1 $AP export_gnn_payload.py GNN_GNN_HVBIL_MODEL.pt payload_gnn_hvbil.npz 2>&1 | grep -v Warning | cut -c1-100
for MIX in 0.5 1.0; do for r in 35 100; do
  RETRO_CLOSED_LOOP=1 RETRO_CL_MECH=1 RETRO_CL_MECH_FNP=1 RETRO_CL_VHEAD=1 RETRO_CL_VHEAD_MIX=$MIX RETRO_DUMP_FEAT=1 RETRO_MEANLEAK_OFF=1 RETRO_GNN_PAYLOAD=$D/payload_gnn_hvbil.npz \
    TAG=closedloop_HVBIL_mix$MIX THREADS=4 $PY $R/run_retro_screen.py $r 2>&1 | grep -o '"mse_nc_f2j_off": [0-9.e-]*' | sed "s/^/r$r closedloop_HVBIL variance mix $MIX CLOSED LOOP: /"
done; done
$PY score_vs_truthpost.py closedloop_HVBIL_mix0.5 closedloop_HVBIL_mix1.0 closedloop_HV_mix0.0 closedloop_HF1
echo VHEADBIL_DONE
