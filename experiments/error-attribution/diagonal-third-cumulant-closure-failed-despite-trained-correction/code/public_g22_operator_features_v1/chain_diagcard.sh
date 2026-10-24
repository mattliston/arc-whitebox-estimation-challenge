#!/bin/bash
# TAG g22-chain-diagcard (T-120): co-adapt the closure to the DIAGONAL card (CP3 cap 1024, K21 diag-only, CP4 off, no quartic): rows, train (init HNQ), loops with util, 1e9 ranking; gate if raw x util < 0.9e-8 (8M) on both nets
D=/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_g22_operator_features_v1; R=/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_g20_emission_retrofit_v1; PY=/Users/matt/arc-white-box-estimation/.venv/bin/python; AP=/Users/matt/anaconda3/bin/python; REPO=/Users/matt/arc-white-box-estimation; cd $D
export CP3_CAP=1024 RETRO_K21_DIAGONLY=1 CP4_CAP=1 RETRO_NO_QUARTIC=1
sed 's#rows_meanleak_mech15_cap#rows_meanleak_mech15_diag#g' launch_mech15_cap.sh > launch_mech15_diag.sh; chmod +x launch_mech15_diag.sh; ./launch_mech15_diag.sh
sleep 120
while [ "$(ls -d rows_meanleak_mech15_diag/row_*/result.json 2>/dev/null | wc -l | tr -d ' ')" -lt 200 ]; do sleep 60; done
while pgrep -f "gen_rows_meanleak_mech15_cap.py [0-9]" > /dev/null; do sleep 20; done
unset CP3_CAP RETRO_K21_DIAGONLY CP4_CAP RETRO_NO_QUARTIC
GNN_FIRST=1 GNN_INIT=GNN_GNN_HNQ_MODEL.pt GNN_LR=5e-4 GNN_LR_MIN=5e-5 $AP train_gnn_oracle.py $D/rows_meanleak_mech15_diag GNN_HDIAG 40 64 2>&1 | grep -v Warning > train_gnn_hdiag.log
grep -E "^epoch (10|20|40)" train_gnn_hdiag.log | cut -c1-200
EXPORT_FIRST=1 $AP export_gnn_payload.py GNN_GNN_HDIAG_MODEL.pt payload_gnn_hdiag.npz 2>&1 | grep -v Warning | cut -c1-80
for r in 35 100; do
  CP3_CAP=1024 RETRO_K21_DIAGONLY=1 CP4_CAP=1 RETRO_NO_QUARTIC=1 RETRO_CLOSED_LOOP=1 RETRO_CL_MECH=1 RETRO_CL_MECH_FNP=1 RETRO_DUMP_FEAT=1 RETRO_MEANLEAK_OFF=1 RETRO_GNN_PAYLOAD=$D/payload_gnn_hdiag.npz \
    TAG=closedloop_HDIAG THREADS=4 $PY $R/run_retro_screen.py $r 2>&1 | grep -o '"u_local": [0-9.]*\|"mse_nc_f2j_off": [0-9.e-]*' | tr '\n' ' ' | sed "s/^/r$r closedloop_HDIAG (co-adapted DIAGONAL card) CLOSED LOOP: /"; echo
done
$PY score_vs_truthpost.py closedloop_HDIAG closedloop_HV_mix0.0
echo DIAGCARD_DONE
