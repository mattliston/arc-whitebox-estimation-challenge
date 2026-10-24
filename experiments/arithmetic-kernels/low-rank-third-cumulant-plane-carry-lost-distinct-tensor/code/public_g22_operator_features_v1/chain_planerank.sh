#!/bin/bash
# TAG g22-chain-planerank (T-119): (a) plain host (no closure) with the K21 plane truncated to rank 256/128 at every layer; (b) co-adaptation at
# rank 128: rows with RETRO_PLANE_RANK=128 (generator passes RETRO_* through) -> train (init HV_F1) -> loops with the truncation -> 1e9 ranking.
D=/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_g22_operator_features_v1; R=/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_g20_emission_retrofit_v1; PY=/Users/matt/arc-white-box-estimation/.venv/bin/python; AP=/Users/matt/anaconda3/bin/python; cd $D
for RK in 256 128; do
  RETRO_PLANE_RANK=$RK RETRO_DUMP_FEAT=1 RETRO_MEANLEAK_OFF=1 TAG=plain_planerank$RK THREADS=4 $PY $R/run_retro_screen.py 35 2>&1 | grep -o '"mse_nc_f2j_off": [0-9.e-]*' | sed "s/^/r35 PLAIN host (no closure), plane rank $RK: /"
done
export RETRO_PLANE_RANK=128
sed 's#rows_meanleak_mech15b#rows_meanleak_mech15_pr128#g' launch_mech15b.sh > launch_mech15_pr128.sh; chmod +x launch_mech15_pr128.sh; ./launch_mech15_pr128.sh
sleep 120
while [ "$(ls -d rows_meanleak_mech15_pr128/row_*/result.json 2>/dev/null | wc -l | tr -d ' ')" -lt 200 ]; do sleep 60; done
while pgrep -f "gen_rows_meanleak_mech15.py [0-9]" > /dev/null; do sleep 20; done
unset RETRO_PLANE_RANK
GNN_FIRST=1 GNN_INIT=GNN_GNN_HV_F1_MODEL.pt GNN_LR=5e-4 GNN_LR_MIN=5e-5 $AP train_gnn_oracle.py $D/rows_meanleak_mech15_pr128 GNN_HPR128 30 64 2>&1 | grep -v Warning > train_gnn_hpr128.log
grep -E "^epoch (10|30)" train_gnn_hpr128.log | cut -c1-200
EXPORT_FIRST=1 $AP export_gnn_payload.py GNN_GNN_HPR128_MODEL.pt payload_gnn_hpr128.npz 2>&1 | grep -v Warning | cut -c1-80
for r in 35 100; do
  RETRO_PLANE_RANK=128 RETRO_CLOSED_LOOP=1 RETRO_CL_MECH=1 RETRO_CL_MECH_FNP=1 RETRO_DUMP_FEAT=1 RETRO_MEANLEAK_OFF=1 RETRO_GNN_PAYLOAD=$D/payload_gnn_hpr128.npz \
    TAG=closedloop_HPR128 THREADS=4 $PY $R/run_retro_screen.py $r 2>&1 | grep -o '"mse_nc_f2j_off": [0-9.e-]*' | sed "s/^/r$r closedloop_HPR128 (closure co-adapted to the rank-128 plane) CLOSED LOOP: /"
done
$PY score_vs_truthpost.py closedloop_HPR128 closedloop_HV_mix0.0
echo PLANERANK_DONE
