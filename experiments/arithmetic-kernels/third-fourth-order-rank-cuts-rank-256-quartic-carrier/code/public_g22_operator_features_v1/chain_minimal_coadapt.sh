#!/bin/bash
# TAG g22-minimal-coadapt (T-118h): the MINIMAL card - CP3 cap 1024 + CP4 off (cap 1) + no quartic + no cubic push - rows, co-adapted closure (init HNQ), loops with util
D=/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_g22_operator_features_v1; R=/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_g20_emission_retrofit_v1; PY=/Users/matt/arc-white-box-estimation/.venv/bin/python; AP=/Users/matt/anaconda3/bin/python; cd $D
until grep -q CP3LOW_DONE chain_cp3low_coadapt.log 2>/dev/null; do sleep 120; done
while pgrep -f "gen_rows_meanleak_mech15.*py [0-9]" > /dev/null; do sleep 60; done
export CP3_CAP=1024 CP4_CAP=1 RETRO_NO_QUARTIC=1 RETRO_NO_CUBIC=1
sed 's#rows_meanleak_mech15_cap#rows_meanleak_mech15_min#g' launch_mech15_cap.sh > launch_mech15_min.sh; chmod +x launch_mech15_min.sh; ./launch_mech15_min.sh
sleep 120
while [ "$(ls -d rows_meanleak_mech15_min/row_*/result.json 2>/dev/null | wc -l | tr -d ' ')" -lt 200 ]; do sleep 60; done
while pgrep -f "gen_rows_meanleak_mech15_cap.py [0-9]" > /dev/null; do sleep 20; done
unset CP3_CAP CP4_CAP RETRO_NO_QUARTIC RETRO_NO_CUBIC
GNN_FIRST=1 GNN_INIT=GNN_GNN_HNQ_MODEL.pt GNN_LR=5e-4 GNN_LR_MIN=5e-5 $AP train_gnn_oracle.py $D/rows_meanleak_mech15_min GNN_HMIN 30 64 2>&1 | grep -v Warning > train_gnn_hmin.log
grep -E "^epoch (10|30)" train_gnn_hmin.log | cut -c1-200
EXPORT_FIRST=1 $AP export_gnn_payload.py GNN_GNN_HMIN_MODEL.pt payload_gnn_hmin.npz 2>&1 | grep -v Warning | cut -c1-80
for r in 35 100; do
  CP3_CAP=1024 CP4_CAP=1 RETRO_NO_QUARTIC=1 RETRO_NO_CUBIC=1 RETRO_CLOSED_LOOP=1 RETRO_CL_MECH=1 RETRO_CL_MECH_FNP=1 RETRO_DUMP_FEAT=1 RETRO_MEANLEAK_OFF=1 RETRO_GNN_PAYLOAD=$D/payload_gnn_hmin.npz \
    TAG=closedloop_HMIN THREADS=4 $PY $R/run_retro_screen.py $r 2>&1 | grep -o '"u_local": [0-9.]*\|"mse_nc_f2j_off": [0-9.e-]*' | tr '\n' ' ' | sed "s/^/r$r closedloop_HMIN (co-adapted MINIMAL card) CLOSED LOOP: /"; echo
done
$PY score_vs_truthpost.py closedloop_HMIN closedloop_HCP3LOW closedloop_HNQ closedloop_HV_mix0.0
echo MINIMAL_DONE
