#!/bin/bash
# TAG g22-after-honest (T-116 orchestrator on the HONEST rows): after chain_honest.sh, (1) gate if both nets < 1.65e-8; (2) FIRST=1 variant
# (rows correct layers 1-15; the loop must match) -> loop; (3) scheduled sampling from the best honest payload (alpha 0.5 rows from its
# own loop, 6 workers) -> aggregate fine-tune -> loop -> gate. Submission iff panel8 AND virgin6 adjusted < 1e-8 (preauthorized).
D=/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_g22_operator_features_v1; R=/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_g20_emission_retrofit_v1; BAR=1.65e-8; PY=/Users/matt/arc-white-box-estimation/.venv/bin/python; AP=/Users/matt/anaconda3/bin/python; cd $D
until grep -q "HONEST_DONE" chain_honest.log 2>/dev/null; do sleep 60; done
r35=$(grep "^r35 closedloop_H" chain_honest.log | grep -o "[0-9.e-]*$"); r100=$(grep "^r100 closedloop_H" chain_honest.log | grep -o "[0-9.e-]*$")
echo "H loop (FIRST=2): r35 $r35 r100 $r100 $(date)" >> after_honest.log
if $PY -c "import sys; sys.exit(0 if float('${r35:-1}') <  and float('${r100:-1}') <  else 1)"; then
  echo "H passes the bar-equivalent -> gate" >> after_honest.log; ./gate_and_fire.sh $D/payload_gnn_h.npz h 0; cat gate_h.log >> after_honest.log
fi
# (2) FIRST=1 variant
GNN_FIRST=1 $AP train_gnn_oracle.py $D/rows_meanleak_mech15b GNN_H_F1 40 64 2>&1 | grep -v Warning > train_gnn_h_f1.log
grep -E "^epoch 40" train_gnn_h_f1.log | cut -c1-200 >> after_honest.log
EXPORT_FIRST=1 $AP export_gnn_payload.py GNN_GNN_H_F1_MODEL.pt payload_gnn_h_f1.npz 2>&1 | grep -v Warning | cut -c1-80 >> after_honest.log
for r in 35 100; do
  RETRO_CLOSED_LOOP=1 RETRO_CL_MECH=1 RETRO_CL_MECH_FNP=1 RETRO_DUMP_FEAT=1 RETRO_MEANLEAK_OFF=1 RETRO_GNN_PAYLOAD=$D/payload_gnn_h_f1.npz \
    TAG=closedloop_HF1 THREADS=4 $PY $R/run_retro_screen.py $r 2>&1 | grep -o '"mse_nc_f2j_off": [0-9.e-]*' | sed "s/^/r$r closedloop_HF1 (honest, first=1) CLOSED LOOP: /" >> after_honest.log
done
f35=$(grep "^r35 closedloop_HF1" after_honest.log | grep -o "[0-9.e-]*$"); f100=$(grep "^r100 closedloop_HF1" after_honest.log | grep -o "[0-9.e-]*$")
if $PY -c "import sys; sys.exit(0 if float('${f35:-1}') <  and float('${f100:-1}') <  else 1)"; then
  echo "HF1 passes the bar-equivalent -> gate" >> after_honest.log; ./gate_and_fire.sh $D/payload_gnn_h_f1.npz hf1 0; cat gate_hf1.log >> after_honest.log
fi
# pick the better honest payload by the r35+r100 sum
if $PY -c "import sys; sys.exit(0 if float('${f35:-1}')+float('${f100:-1}') < float('${r35:-1}')+float('${r100:-1}') else 1)"; then BEST=payload_gnn_h_f1.npz; BM=GNN_GNN_H_F1_MODEL.pt; BF=1; else BEST=payload_gnn_h.npz; BM=GNN_GNN_H_C64_E40_MODEL.pt; BF=2; fi
echo "best honest payload: $BEST (first=$BF)" >> after_honest.log
# (3) scheduled sampling from the best honest model
./launch_gnnmixed.sh $D/$BEST 0.5 $D/rows_gnnmixed_h
sleep 120
while [ "$(ls -d rows_gnnmixed_h/row_*/result.json 2>/dev/null | wc -l | tr -d ' ')" -lt 200 ]; do sleep 60; done
while pgrep -f "gen_rows_gnnmixed.py [0-9]" > /dev/null; do sleep 20; done
GNN_FIRST=$BF GNN_EXTRA_ROWS=$D/rows_gnnmixed_h GNN_INIT=$BM GNN_LR=3e-4 GNN_LR_MIN=3e-5 $AP train_gnn_oracle.py $D/rows_meanleak_mech15b GNN_H_SS05AGG 12 64 2>&1 | grep -v Warning > train_gnn_h_ss05agg.log
grep -E "^epoch 10" train_gnn_h_ss05agg.log | cut -c1-200 >> after_honest.log
EXPORT_FIRST=$BF $AP export_gnn_payload.py GNN_GNN_H_SS05AGG_MODEL.pt payload_gnn_h_ss05agg.npz 2>&1 | grep -v Warning | cut -c1-80 >> after_honest.log
for r in 35 100; do
  RETRO_CLOSED_LOOP=1 RETRO_CL_MECH=1 RETRO_CL_MECH_FNP=1 RETRO_DUMP_FEAT=1 RETRO_MEANLEAK_OFF=1 RETRO_GNN_PAYLOAD=$D/payload_gnn_h_ss05agg.npz \
    TAG=closedloop_HSS05AGG THREADS=4 $PY $R/run_retro_screen.py $r 2>&1 | grep -o '"mse_nc_f2j_off": [0-9.e-]*' | sed "s/^/r$r closedloop_HSS05AGG CLOSED LOOP: /" >> after_honest.log
done
s35=$(grep "^r35 closedloop_HSS05AGG" after_honest.log | grep -o "[0-9.e-]*$"); s100=$(grep "^r100 closedloop_HSS05AGG" after_honest.log | grep -o "[0-9.e-]*$")
if $PY -c "import sys; sys.exit(0 if float('${s35:-1}') <  and float('${s100:-1}') <  else 1)"; then
  echo "HSS05AGG passes the bar-equivalent -> gate" >> after_honest.log; ./gate_and_fire.sh $D/payload_gnn_h_ss05agg.npz hss05agg 0; cat gate_hss05agg.log >> after_honest.log
fi
echo "AFTER_HONEST_DONE $(date)" >> after_honest.log
