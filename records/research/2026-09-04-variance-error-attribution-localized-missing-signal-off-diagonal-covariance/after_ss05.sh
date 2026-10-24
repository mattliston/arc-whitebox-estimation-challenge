#!/bin/bash
# TAG g22-after-ss05 (T-116 orchestrator, Matt asleep): when the scheduled-sampling chain (SS05) reports its closed-loop numbers,
# (1) if both nets are below the bar-equivalent (raw < 1.65e-8 board-equivalent at util ~0.61 -> adjusted < 1e-8) run the gate
#     (package, validate, panel + virgin, submit iff both means < 1e-8);
# (2) regardless, run the next control round: rows from the SS05 model's own loop at truth share 0.25, fine-tune from the SS05
#     model on the AGGREGATE of all rows (oracle + alpha 0.5 + alpha 0.25), export, closed loop, gate if it passes.
D=$(cd "$(dirname "$0")" && pwd); cd "$D"; BAR=1.65e-8
L=chain_gnnmixed_ss05.log
until grep -q "GNN_M15_SS05_DONE\|Traceback" $L 2>/dev/null; do sleep 60; done
grep -q Traceback $L && { echo "SS05 chain failed" >> after_ss05.log; }
r35=$(grep "^r35 closedloop_GNN_M15_SS05" $L | grep -o "[0-9.e-]*$"); r100=$(grep "^r100 closedloop_GNN_M15_SS05" $L | grep -o "[0-9.e-]*$")
echo "SS05 loop: r35 $r35 r100 $r100 $(date)" >> after_ss05.log
if /Users/matt/arc-white-box-estimation/.venv/bin/python -c "import sys; sys.exit(0 if float('${r35:-1}') < $BAR and float('${r100:-1}') < $BAR else 1)"; then
  echo "SS05 passes the bar-equivalent on both nets -> gate" >> after_ss05.log
  ./gate_and_fire.sh $D/payload_gnn_m15_ss05.npz ss05 0
  cat gate_ss05.log >> after_ss05.log
fi
# next control round: alpha 0.25 rows from the SS05 model, aggregate fine-tune
[ -f payload_gnn_m15_ss05.npz ] || { echo "no SS05 payload; stop" >> after_ss05.log; exit 1; }
./launch_gnnmixed.sh $D/payload_gnn_m15_ss05.npz 0.25 $D/rows_gnnmixed025
sleep 120
while [ "$(ls -d rows_gnnmixed025/row_*/result.json 2>/dev/null | wc -l | tr -d ' ')" -lt 200 ]; do sleep 60; done
while pgrep -f "gen_rows_gnnmixed.py [0-9]" > /dev/null; do sleep 20; done
GNN_EXTRA_ROWS=$D/rows_gnnmixed,$D/rows_gnnmixed025 GNN_INIT=GNN_GNN_M15_SS05_MODEL.pt GNN_LR=3e-4 GNN_LR_MIN=3e-5 \
  /Users/matt/anaconda3/bin/python train_gnn_oracle.py $D/rows_meanleak_mech15 GNN_M15_SS025AGG 12 64 2>&1 | grep -v Warning > train_gnn_m15_ss025agg.log
/Users/matt/anaconda3/bin/python export_gnn_payload.py GNN_GNN_M15_SS025AGG_MODEL.pt payload_gnn_m15_ss025agg.npz 2>&1 | grep -v Warning | cut -c1-120 >> after_ss05.log
R=$D/../public_g20_emission_retrofit_v1
for r in 35 100; do
  RETRO_CLOSED_LOOP=1 RETRO_CLOSED_LOOP_ROWS=pre RETRO_CL_MECH=1 RETRO_CL_MECH_FNP=1 RETRO_DUMP_FEAT=1 RETRO_MEANLEAK_OFF=1 RETRO_GNN_PAYLOAD=$D/payload_gnn_m15_ss025agg.npz \
    TAG=closedloop_SS025AGG THREADS=4 /Users/matt/arc-white-box-estimation/.venv/bin/python $R/run_retro_screen.py $r 2>&1 | grep -o '"mse_nc_f2j_off": [0-9.e-]*' | sed "s/^/r$r closedloop_SS025AGG CLOSED LOOP: /" >> after_ss05.log
done
r35=$(grep "^r35 closedloop_SS025AGG" after_ss05.log | grep -o "[0-9.e-]*$"); r100=$(grep "^r100 closedloop_SS025AGG" after_ss05.log | grep -o "[0-9.e-]*$")
if /Users/matt/arc-white-box-estimation/.venv/bin/python -c "import sys; sys.exit(0 if float('${r35:-1}') < $BAR and float('${r100:-1}') < $BAR else 1)"; then
  echo "SS025AGG passes the bar-equivalent on both nets -> gate" >> after_ss05.log
  ./gate_and_fire.sh $D/payload_gnn_m15_ss025agg.npz ss025agg 0
  cat gate_ss025agg.log >> after_ss05.log
fi
echo "AFTER_SS05_DONE $(date)" >> after_ss05.log
