#!/bin/bash
# TAG g22-chain-cp3-8k (T-140): the combined cheaper card CP4_CAP=256 + no dense quartic push, closure CO-ADAPTED: honest oracle rows
# (PSD-projected consistent update) -> closure (64 ch, first=1, init HCAP256) -> export -> closed loop r35/r100 -> 1e9 rank -> build (v-head 0) -> validate vs instrument.
D=$(cd "$(dirname "$0")" && pwd); cd "$D"; R=$D/../public_g20_emission_retrofit_v1; AP=/Users/matt/anaconda3/bin/python; PY=/Users/matt/arc-white-box-estimation/.venv/bin/python
export CP4_CAP=256 CP3_CAP=8192 F32_COV=1 RETRO_MEANLEAK_PSD=1
./launch_mech15_cp3_8k.sh
while [ "$(ls -d rows_meanleak_mech15_cp3_8k/row_*/result.json 2>/dev/null | wc -l | tr -d ' ')" -lt 200 ]; do sleep 60; done
while pgrep -f "gen_rows_meanleak_mech15_cap.py [0-9]" > /dev/null; do sleep 20; done
unset RETRO_MEANLEAK_PSD
$PY - <<'PYEOF'
import numpy as np, glob
D='.'; acc=np.zeros(16); n=0
for d in sorted(glob.glob('rows_meanleak_mech15_cp3_8k/row_*'))[:12]:
    r=int(d.split('_')[-1]); h=np.load(d+'/host.npy').astype(float); t=np.load(f'truthpost/truthpost_rank{r:04d}.npz')['mc_post'].astype(float); acc+=np.mean((h-t)**2,axis=1); n+=1
acc/=n; print('ROW HEALTH cp3_8k one-step L10 %.2e L14 %.2e L15 %.2e ratio L15/L10 %.1fx' % (acc[10], acc[14], acc[15], acc[15]/acc[10]))
PYEOF
GNN_FIRST=1 GNN_INIT=GNN_GNN_HCAP256_MODEL.pt GNN_LR=5e-4 GNN_LR_MIN=5e-5 $AP train_gnn_oracle.py $D/rows_meanleak_mech15_cp3_8k GNN_HCP3_8K 40 64 2>&1 | grep -v Warning > train_gnn_hcp3_8k.log
grep -E "^epoch (10|20|40)" train_gnn_hcp3_8k.log | cut -c1-200
EXPORT_FIRST=1 EXPORT_VHEAD=0 $AP export_gnn_payload.py GNN_GNN_HCP3_8K_MODEL.pt payload_gnn_hcp3_8k.npz 2>&1 | grep -v Warning | cut -c1-80
for r in 35 100; do
  CP4_CAP=256 CP3_CAP=8192 F32_COV=1 RETRO_CLOSED_LOOP=1 RETRO_CL_MECH=1 RETRO_CL_MECH_FNP=1 RETRO_DUMP_FEAT=1 RETRO_MEANLEAK_OFF=1 RETRO_GNN_PAYLOAD=$D/payload_gnn_hcp3_8k.npz \
    TAG=closedloop_HCP3_8K THREADS=4 $PY $R/run_retro_screen.py $r 2>&1 | grep -o '"u_local": [0-9.]*\|"mse_nc_f2j_off": [0-9.e-]*' | tr '\n' ' ' | sed "s/^/r$r closedloop_HCP3_8K (CP4 256 + CP3 8192 + f32 cov, co-adapted) CLOSED LOOP: /"; echo
done
$PY score_vs_truthpost.py closedloop_HCP3_8K closedloop_HCAP256_f32cov closedloop_HCAP256
BUILD_CP4_CAP=256 BUILD_CP3_CAP=8192 BUILD_F32_COV=1 $AP build_gnn_candidate.py $D/payload_gnn_hcp3_8k.npz $D/cand_gnn_hcp3_8k.py 2>&1 | grep -v Warning | tail -1 | cut -c1-100
THREADS=6 $PY cand_vs_instrument.py cand_gnn_hcp3_8k.py 35 dbgdump_lean2_retro-35.npz 2>&1 | grep "final row" | cut -c1-120
echo CP3_8K_DONE
