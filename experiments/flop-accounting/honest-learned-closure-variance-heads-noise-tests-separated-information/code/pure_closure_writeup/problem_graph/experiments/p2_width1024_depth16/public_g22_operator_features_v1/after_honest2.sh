#!/bin/bash
# TAG g22-after-honest2 (T-116j): after after_honest.sh, the pure on-policy round: alpha 0 rows from the best payload so far, aggregate
# fine-tune (oracle + 0.5 + 0 rows), loop, gate; then an exposure-state variant (GNN_EXPOSURE=1) trained on the same aggregate, looped with
# RETRO_CL_EXPOSURE_FEAT=1 (instrument only; the candidate builder does not carry exposure columns yet).
D=/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_g22_operator_features_v1; R=/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_g20_emission_retrofit_v1; BAR=1.65e-8; PY=/Users/matt/arc-white-box-estimation/.venv/bin/python; AP=/Users/matt/anaconda3/bin/python; cd $D
until grep -q "AFTER_HONEST_DONE" after_honest.log 2>/dev/null; do sleep 60; done
best() { $PY - <<'PYEOF'
import re
from pathlib import Path
L = (Path('after_honest.log').read_text() + Path('chain_honest.log').read_text())
c = {}
for tag, pay, first, model in (("closedloop_H ", "payload_gnn_h.npz", 2, "GNN_GNN_H_C64_E40_MODEL.pt"), ("closedloop_HF1", "payload_gnn_h_f1.npz", 1, "GNN_GNN_H_F1_MODEL.pt"), ("closedloop_HSS05AGG", "payload_gnn_h_ss05agg.npz", None, "GNN_GNN_H_SS05AGG_MODEL.pt")):
    v = [float(m) for m in re.findall(r"^r(?:35|100) " + re.escape(tag) + r".*?: ([0-9.e-]+)$", L, flags=re.M)]
    if len(v) == 2: c[pay] = (sum(v), first, model)
pay = min(c, key=lambda k: c[k][0]); first = c[pay][1]
if first is None:
    first = 1 if "best honest payload: payload_gnn_h_f1" in L else 2
print(pay, first, c[pay][2])
PYEOF
}
read BEST BF BM <<< "$(best)"; echo "after_honest2: best so far $BEST (first=$BF, model $BM)" >> after_honest2.log
./launch_gnnmixed.sh $D/$BEST 0.0 $D/rows_gnnmixed_h0
sleep 120
while [ "$(ls -d rows_gnnmixed_h0/row_*/result.json 2>/dev/null | wc -l | tr -d ' ')" -lt 200 ]; do sleep 60; done
while pgrep -f "gen_rows_gnnmixed.py [0-9]" > /dev/null; do sleep 20; done
GNN_FIRST=$BF GNN_EXTRA_ROWS=$D/rows_gnnmixed_h,$D/rows_gnnmixed_h0 GNN_INIT=$BM GNN_LR=3e-4 GNN_LR_MIN=3e-5 $AP train_gnn_oracle.py $D/rows_meanleak_mech15b GNN_H_DAGGER 12 64 2>&1 | grep -v Warning > train_gnn_h_dagger.log
grep -E "^epoch 10" train_gnn_h_dagger.log | cut -c1-200 >> after_honest2.log
EXPORT_FIRST=$BF $AP export_gnn_payload.py GNN_GNN_H_DAGGER_MODEL.pt payload_gnn_h_dagger.npz 2>&1 | grep -v Warning | cut -c1-80 >> after_honest2.log
for r in 35 100; do
  RETRO_CLOSED_LOOP=1 RETRO_CL_MECH=1 RETRO_CL_MECH_FNP=1 RETRO_DUMP_FEAT=1 RETRO_MEANLEAK_OFF=1 RETRO_GNN_PAYLOAD=$D/payload_gnn_h_dagger.npz \
    TAG=closedloop_HDAGGER THREADS=4 $PY $R/run_retro_screen.py $r 2>&1 | grep -o '"mse_nc_f2j_off": [0-9.e-]*' | sed "s/^/r$r closedloop_HDAGGER CLOSED LOOP: /" >> after_honest2.log
done
d35=$(grep "^r35 closedloop_HDAGGER" after_honest2.log | grep -o "[0-9.e-]*$"); d100=$(grep "^r100 closedloop_HDAGGER" after_honest2.log | grep -o "[0-9.e-]*$")
if $PY -c "import sys; sys.exit(0 if float('${d35:-1}') <  and float('${d100:-1}') <  else 1)"; then
  echo "HDAGGER passes the bar-equivalent -> gate" >> after_honest2.log; ./gate_and_fire.sh $D/payload_gnn_h_dagger.npz hdagger 0; cat gate_hdagger.log >> after_honest2.log
fi
# exposure-state variant on the same aggregate (instrument-only deploy)
GNN_FIRST=$BF GNN_EXPOSURE=1 GNN_EXTRA_ROWS=$D/rows_gnnmixed_h,$D/rows_gnnmixed_h0 $AP train_gnn_oracle.py $D/rows_meanleak_mech15b GNN_H_EXPO 40 64 2>&1 | grep -v Warning > train_gnn_h_expo.log
grep -E "^epoch 40" train_gnn_h_expo.log | cut -c1-200 >> after_honest2.log
EXPORT_FIRST=$BF $AP export_gnn_payload.py GNN_GNN_H_EXPO_MODEL.pt payload_gnn_h_expo.npz 2>&1 | grep -v Warning | cut -c1-80 >> after_honest2.log
for r in 35 100; do
  RETRO_CLOSED_LOOP=1 RETRO_CL_MECH=1 RETRO_CL_MECH_FNP=1 RETRO_CL_EXPOSURE_FEAT=1 RETRO_DUMP_FEAT=1 RETRO_MEANLEAK_OFF=1 RETRO_GNN_PAYLOAD=$D/payload_gnn_h_expo.npz \
    TAG=closedloop_HEXPO THREADS=4 $PY $R/run_retro_screen.py $r 2>&1 | grep -o '"mse_nc_f2j_off": [0-9.e-]*' | sed "s/^/r$r closedloop_HEXPO (exposure input) CLOSED LOOP: /" >> after_honest2.log
done
echo "AFTER_HONEST2_DONE $(date)" >> after_honest2.log
