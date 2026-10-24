#!/bin/bash
# TAG g22-k22-small (T-118e): 60 honest oracle rows WITH the engine K22 plane chain active, fine-tune the mean head (init HV_F1), loop with the chain
D=/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_g22_operator_features_v1; R=/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_g20_emission_retrofit_v1; PY=/Users/matt/arc-white-box-estimation/.venv/bin/python; AP=/Users/matt/anaconda3/bin/python; cd $D
while pgrep -f "gen_rows_meanleak_mech15_cap.py [0-9]" > /dev/null; do sleep 60; done     # after the cap rows (memory allowance)
export RETRO_K22_CHAIN=1
for rng in "35 54" "101 120" "168 187"; do set -- $rng; G22_OUT=$D/rows_meanleak_mech15_k22 nohup nice -n 10 $PY run_gen_meanleak_mech15_k22.py $1 $2 > gen_k22_$1.log 2>&1 & sleep 1; done
sleep 300
while [ "$(ls -d rows_meanleak_mech15_k22/row_*/result.json 2>/dev/null | wc -l | tr -d ' ')" -lt 60 ]; do sleep 120; done
while pgrep -f "gen_rows_meanleak_mech15_k22.py [0-9]" > /dev/null; do sleep 30; done
unset RETRO_K22_CHAIN
GNN_FIRST=1 GNN_INIT=GNN_GNN_HV_F1_MODEL.pt GNN_LR=5e-4 GNN_LR_MIN=5e-5 $AP train_gnn_oracle.py $D/rows_meanleak_mech15_k22 GNN_HK22S 20 64 2>&1 | grep -v Warning > train_gnn_hk22s.log
grep -E "rows:|^epoch (10|20)" train_gnn_hk22s.log | cut -c1-200
EXPORT_FIRST=1 $AP export_gnn_payload.py GNN_GNN_HK22S_MODEL.pt payload_gnn_hk22s.npz 2>&1 | grep -v Warning | cut -c1-80
for r in 35 100; do
  RETRO_K22_CHAIN=1 RETRO_CLOSED_LOOP=1 RETRO_CL_MECH=1 RETRO_CL_MECH_FNP=1 RETRO_DUMP_FEAT=1 RETRO_MEANLEAK_OFF=1 RETRO_GNN_PAYLOAD=$D/payload_gnn_hk22s.npz \
    TAG=closedloop_HK22S THREADS=4 $PY $R/run_retro_screen.py $r 2>&1 | grep -o '"mse_nc_f2j_off": [0-9.e-]*' | sed "s/^/r$r closedloop_HK22S (mean head co-adapted to the K22 chain, 60 rows) CLOSED LOOP: /"
done
$PY score_vs_truthpost.py closedloop_HK22S closedloop_k22chain closedloop_HV_mix0.0
echo K22SMALL_DONE
