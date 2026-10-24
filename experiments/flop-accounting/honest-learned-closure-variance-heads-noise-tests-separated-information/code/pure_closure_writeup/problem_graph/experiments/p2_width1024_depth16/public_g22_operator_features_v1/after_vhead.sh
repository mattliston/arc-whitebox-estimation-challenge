#!/bin/bash
# TAG g22-after-vhead (T-116p): after chain_vhead.sh, if the variance head improved the loop at some mix (1e9 frame vs HF1), regenerate
# ORACLE rows with the variance correction active (closed loop with truth share 1.0 -> the oracle mean correction, plus the v-head's
# correction at the pre side; tape pre-correction), retrain mean + variance heads on them (co-adaptation), loop at the same mix, gate.
D=/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_g22_operator_features_v1; R=/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_g20_emission_retrofit_v1; BAR=1.65e-8; PY=/Users/matt/arc-white-box-estimation/.venv/bin/python; AP=/Users/matt/anaconda3/bin/python; cd $D
until grep -q "VHEAD_DONE" chain_vhead.log 2>/dev/null; do sleep 120; done
best=$($PY - <<'PYEOF'
import re, numpy as np
from pathlib import Path
L = Path('chain_vhead.log').read_text()
sc = {}
for tag in ("closedloop_HV_mix0.0", "closedloop_HV_mix0.5", "closedloop_HV_mix1.0", "closedloop_HF1"):
    m = re.search(r"^" + re.escape(tag) + r": r35 ([0-9.e-]+) \| r100 ([0-9.e-]+)$", L, flags=re.M)
    if m: sc[tag] = float(m.group(1)) + float(m.group(2))
if not sc: print("none"); raise SystemExit
b = min(sc, key=sc.get); ref = sc.get("closedloop_HF1", 1.0)
print(b.split("mix")[-1] if "mix" in b and sc[b] < ref else "none")
PYEOF
)
echo "after_vhead: best mix = $best $(date)" >> after_vhead.log
[ "$best" = "none" ] && { echo "variance head did not beat HF1 in the loop; stop" >> after_vhead.log; echo AFTER_VHEAD_DONE >> after_vhead.log; exit 0; }
# gate the best mix directly if it is already under the bar-equivalent
r35=$(grep "^r35 closedloop_HV variance mix $best" chain_vhead.log | grep -o "[0-9.e-]*$"); r100=$(grep "^r100 closedloop_HV variance mix $best" chain_vhead.log | grep -o "[0-9.e-]*$")
if $PY -c "import sys; sys.exit(0 if float('${r35:-1}') <  and float('${r100:-1}') <  else 1)"; then
  echo "HV mix $best passes the bar-equivalent -> gate" >> after_vhead.log; BUILD_VHEAD_MIX=$best ./gate_and_fire.sh $D/payload_gnn_hv_f1.npz hv_mix$best 0; cat gate_hv_mix$best.log >> after_vhead.log
fi
# co-adaptation round: oracle rows with the v-head active
while pgrep -f "gen_rows_gnnmixed.py [0-9]|gen_rows_meanleak_mech15.py [0-9]" > /dev/null; do sleep 60; done
RETRO_CL_VHEAD=1 RETRO_CL_VHEAD_MIX=$best ./launch_gnnmixed.sh $D/payload_gnn_hv_f1.npz 1.0 $D/rows_oracle_vh
sleep 120
while [ "$(ls -d rows_oracle_vh/row_*/result.json 2>/dev/null | wc -l | tr -d ' ')" -lt 200 ]; do sleep 60; done
while pgrep -f "gen_rows_gnnmixed.py [0-9]" > /dev/null; do sleep 20; done
GNN_FIRST=1 GNN_VLABELS=$D/vlabels GNN_VLOSS=1.0 GNN_INIT=GNN_GNN_HV_F1_MODEL.pt GNN_LR=5e-4 GNN_LR_MIN=5e-5 $AP train_gnn_oracle.py $D/rows_oracle_vh GNN_HV2 25 64 2>&1 | grep -v Warning > train_gnn_hv2.log
grep -E "^epoch 25|variance head" train_gnn_hv2.log | tail -2 | cut -c1-220 >> after_vhead.log
EXPORT_FIRST=1 $AP export_gnn_payload.py GNN_GNN_HV2_MODEL.pt payload_gnn_hv2.npz 2>&1 | grep -v Warning | cut -c1-80 >> after_vhead.log
for r in 35 100; do
  RETRO_CLOSED_LOOP=1 RETRO_CL_MECH=1 RETRO_CL_MECH_FNP=1 RETRO_CL_VHEAD=1 RETRO_CL_VHEAD_MIX=$best RETRO_DUMP_FEAT=1 RETRO_MEANLEAK_OFF=1 RETRO_GNN_PAYLOAD=$D/payload_gnn_hv2.npz \
    TAG=closedloop_HV2 THREADS=4 $PY $R/run_retro_screen.py $r 2>&1 | grep -o '"mse_nc_f2j_off": [0-9.e-]*' | sed "s/^/r$r closedloop_HV2 (co-adapted, mix $best) CLOSED LOOP: /" >> after_vhead.log
done
$PY score_vs_truthpost.py closedloop_HV2 closedloop_HV_mix$best closedloop_HF1 >> after_vhead.log
h35=$(grep "^r35 closedloop_HV2" after_vhead.log | grep -o "[0-9.e-]*$"); h100=$(grep "^r100 closedloop_HV2" after_vhead.log | grep -o "[0-9.e-]*$")
if $PY -c "import sys; sys.exit(0 if float('${h35:-1}') <  and float('${h100:-1}') <  else 1)"; then
  echo "HV2 passes the bar-equivalent -> gate" >> after_vhead.log; BUILD_VHEAD_MIX=$best ./gate_and_fire.sh $D/payload_gnn_hv2.npz hv2 0; cat gate_hv2.log >> after_vhead.log
fi
echo "AFTER_VHEAD_DONE $(date)" >> after_vhead.log
