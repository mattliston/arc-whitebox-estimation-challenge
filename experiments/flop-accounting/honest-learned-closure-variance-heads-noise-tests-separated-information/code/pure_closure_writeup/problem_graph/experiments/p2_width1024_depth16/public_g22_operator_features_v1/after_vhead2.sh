#!/bin/bash
# TAG g22-after-vhead2 (T-116p co-adaptation): ORACLE rows with the HV_F1 variance head active at mix 1.0 (closed loop, truth share 1.0
# = the oracle mean correction; tape pre-correction), then retrain the MEAN head on them with the variance head FROZEN (the mean head
# stops double-counting the variance error), export, loop at mix 1.0 (and 0.5), rank in the 1e9 frame, gate.
D=/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_g22_operator_features_v1; R=/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_g20_emission_retrofit_v1; BAR=1.65e-8; PY=/Users/matt/arc-white-box-estimation/.venv/bin/python; AP=/Users/matt/anaconda3/bin/python; cd $D
while pgrep -f "gen_rows_gnnmixed.py [0-9]|gen_rows_meanleak_mech15.py [0-9]" > /dev/null; do sleep 60; done
RETRO_CL_VHEAD=1 RETRO_CL_VHEAD_MIX=1.0 ./launch_gnnmixed.sh $D/payload_gnn_hv_f1.npz 1.0 $D/rows_oracle_vh
sleep 120
while [ "$(ls -d rows_oracle_vh/row_*/result.json 2>/dev/null | wc -l | tr -d ' ')" -lt 200 ]; do sleep 60; done
while pgrep -f "gen_rows_gnnmixed.py [0-9]" > /dev/null; do sleep 20; done
GNN_FIRST=1 GNN_VLABELS=$D/vlabels GNN_VLOSS=0.0 GNN_VFREEZE=1 GNN_INIT=GNN_GNN_HV_F1_MODEL.pt GNN_LR=5e-4 GNN_LR_MIN=5e-5 $AP train_gnn_oracle.py $D/rows_oracle_vh GNN_HV2 30 64 2>&1 | grep -v Warning > train_gnn_hv2.log
grep -E "^epoch (10|20|30)|frozen" train_gnn_hv2.log | cut -c1-200 >> after_vhead2.log
EXPORT_FIRST=1 $AP export_gnn_payload.py GNN_GNN_HV2_MODEL.pt payload_gnn_hv2.npz 2>&1 | grep -v Warning | cut -c1-100 >> after_vhead2.log
for MIX in 1.0 0.5; do for r in 35 100; do
  RETRO_CLOSED_LOOP=1 RETRO_CL_MECH=1 RETRO_CL_MECH_FNP=1 RETRO_CL_VHEAD=1 RETRO_CL_VHEAD_MIX=$MIX RETRO_DUMP_FEAT=1 RETRO_MEANLEAK_OFF=1 RETRO_GNN_PAYLOAD=$D/payload_gnn_hv2.npz \
    TAG=closedloop_HV2_mix$MIX THREADS=4 $PY $R/run_retro_screen.py $r 2>&1 | grep -o '"mse_nc_f2j_off": [0-9.e-]*' | sed "s/^/r$r closedloop_HV2 co-adapted mean head, variance mix $MIX CLOSED LOOP: /" >> after_vhead2.log
done; done
$PY score_vs_truthpost.py closedloop_HV2_mix1.0 closedloop_HV2_mix0.5 closedloop_HV_mix0.0 closedloop_HF1 >> after_vhead2.log
h35=$(grep "^r35 closedloop_HV2 co-adapted mean head, variance mix 1.0" after_vhead2.log | grep -o "[0-9.e-]*$"); h100=$(grep "^r100 closedloop_HV2 co-adapted mean head, variance mix 1.0" after_vhead2.log | grep -o "[0-9.e-]*$")
if $PY -c "import sys; sys.exit(0 if float('${h35:-1}') <  and float('${h100:-1}') <  else 1)"; then
  echo "HV2 mix 1.0 passes the bar-equivalent -> gate" >> after_vhead2.log; BUILD_VHEAD_MIX=1.0 ./gate_and_fire.sh $D/payload_gnn_hv2.npz hv2 0; cat gate_hv2.log >> after_vhead2.log
fi
echo "AFTER_VHEAD2_DONE $(date)" >> after_vhead2.log
