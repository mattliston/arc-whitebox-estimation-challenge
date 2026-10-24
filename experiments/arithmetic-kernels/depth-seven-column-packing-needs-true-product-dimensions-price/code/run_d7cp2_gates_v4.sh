#!/bin/bash
# TAG fable-engineer2 (d7cp v2 64210b64 = v1 + the k = 864/992 colpack products; lead's 12:08 GO on the colpack slice; 2026-09-25): gates on
# S28's pre-treuse file s28_pre.py (27d8e622) + d7cp_fragment_v2.py + treuse v2 (75aa9233) LAST (= s28d7/cand2_T.py), vs base = S28 (a559b073).
# (1) in-process parity (setup first, nets 35/100/37/39; the base npz is the v3 chain's par4_base.npz from the SAME base file): bill and
#     final-layer MSE ratio vs the 32M truth; FAIL if any net's MSE ratio > 1.01. (2) numpy-free client census of cand2 and base (35/100).
# (3) engineer's paired user-CPU pair (6 nets) and (4) my paired residual gate; each waits for no fire chain / VALID40 / held-out runner,
#     then the machine gate (gate before lock).
set -u
P2=/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16; E=$P2/fable_engineer2_v1; D=$E/depth7; W=$D/s28d7
SPY=$P2/astra_phase2_reexamination_v1/openblas_runtime_env_v1/bin/python; PS=$E/diet5/proto; C=$W/cand2_T.py
YIELD="bash fire_chain_S2[0-9]|run_valid40_locke[d]|heldout_R320_ru[n]|heldout40_v[3]"
echo "$(date +%T) gates v4 | base_T $(shasum -a 256 $W/base_T.py | cut -c1-16) | cand2_T $(shasum -a 256 $C | cut -c1-16) | base npz $(shasum -a 256 $W/par4_base.npz | cut -c1-16)"
[ -e $W/par4_d7cp2.npz ] && { echo "refusing to overwrite par4_d7cp2.npz"; exit 1; }
bash $P2/fable_porter_v1/machine_gate.sh 6 e2-d7cp2-parity
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 nice -n 10 $SPY $PS/parity_setup_sep.py $C 35,100,37,39 $W/par4_d7cp2.npz 2>&1 | grep -v "Warning\|^  "
$SPY $PS/parity_setup_cmp.py $W/par4_base.npz $W/par4_d7cp2.npz | tee $W/parity_d7cp2.txt
python3 - $W/parity_d7cp2.txt <<'PY' | tee $W/parity_d7cp2_verdict.txt
import json, sys, statistics as st
rows = [json.loads(l) for l in open(sys.argv[1]) if l.startswith('{')]
print(f"d7cp v2: bill {100*st.mean(r['flops_B']/r['flops_A']-1 for r in rows):+.3f}% (mean of 4 nets; per net " + ", ".join(f"{r['rank']} {100*(r['flops_B']/r['flops_A']-1):+.3f}%" for r in rows) + ") | MSE ratio " + ", ".join(f"{r['mse_ratio']:.5f}" for r in rows))
print('FAIL' if max(r['mse_ratio'] for r in rows) > 1.01 or len(rows) != 4 else 'PARITY_OK')
PY
grep -q "^PARITY_OK" $W/parity_d7cp2_verdict.txt || { echo "FAIL: parity verdict"; exit 1; }
echo "$(date +%T) PARITY_DONE"
bash $P2/fable_porter_v1/machine_gate.sh 8 e2-d7cp2-census
for t in cand2:$C base:$W/base_T.py; do
  bash $P2/fable_porter_v1/run_census_porter_v3.sh ${t#*:} 35,100 e2d7cp2_${t%%:*} > $W/smoke2_${t%%:*}.txt 2>&1
  grep -v "^  \|Warning" $W/smoke2_${t%%:*}.txt | tail -5 | cut -c1-300
  if grep -q "Traceback\|Error" $W/smoke2_${t%%:*}.txt; then echo "FAIL: client smoke ${t%%:*}"; exit 1; fi
done
echo "$(date +%T) CLIENT_SMOKE_PASS"
until ! pgrep -f "$YIELD" >/dev/null; do sleep 30; done
bash $P2/fable_engineer_v1/run_usercpu_pair_e71.sh e2d7cp2_base $W/base_T.py e2d7cp2_cand $C
echo "$(date +%T) USERCPU_DONE"
until ! pgrep -f "$YIELD" >/dev/null; do sleep 30; done
bash $P2/fable_porter_v1/machine_gate.sh 8 e2-d7cp2-resid
bash $E/lrroute/resid_gate_v1.sh $C d7cp2S28 $W/base_T.py
echo "$(date +%T) D7CP2_GATES_ALL_DONE"
