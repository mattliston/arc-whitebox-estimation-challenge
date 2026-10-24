#!/bin/bash
# TAG g22-profile-f1 (T-116f): per-layer carried error of the FIRST=1 model's loop on r35 (every layer), after the F1 chain
D=$(cd "$(dirname "$0")" && pwd); cd "$D"
until grep -q "F1_DONE" chain_f1.log 2>/dev/null; do sleep 30; done
[ -f payload_gnn_m15_f1.npz ] || { echo "no F1 payload"; echo PROFILE_F1_DONE; exit 0; }
echo "== F1 closed loop, r35 (every layer)"; THREADS=3 /Users/matt/arc-white-box-estimation/.venv/bin/python closedloop_layers_gnn.py 35 $D/payload_gnn_m15_f1.npz none 2>&1 | grep -v Warning
echo PROFILE_F1_DONE
