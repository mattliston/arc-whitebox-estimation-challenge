#!/bin/bash
# stacking-probe chain: deltas at L4/L8, augmented dvhat, consumption
cd "$(dirname "$0")"
set -e
echo "=== fits:"; tail -1 sweep_L4.log; tail -1 sweep_L8.log
for L in 4 8; do
  OPENBLAS_NUM_THREADS=4 nice -n 15 python3 run_g3_engine_delta.py 35 $L \
    ncqc_corefitJ_rank0035_L${L}_r32.npz 2>&1 | tail -1
done
OPENBLAS_NUM_THREADS=4 python3 - <<'PY'
import numpy as np, json, sys
sys.path.insert(0,'.')
from run_chaos1024_shallow import load_weights
W=load_weights(35)
st=np.load('states_rank0035.npz'); tr=np.load('mps_cov2_rank0035.npz')
base=np.load('dvhat_rank0035.npz')
out={k: base[k].copy() for k in base.files}
for L in [4,5,8,11]:
    d=np.load(f'g3delta_planes_L{L}.npz')
    core=np.load(f'ncqc_corefitJ_rank0035_L{L}_r32.npz')
    sel=core['sel']
    dP=d['K21r'].astype(np.float64)-d['K21e'].astype(np.float64)
    Wp=np.asarray(W[L+1],dtype=np.float64)[sel,:]
    dv_c=np.einsum('ac,ab,bc->c',Wp,dP,Wp,optimize=True)
    tv=tr['mc_pre_v'][L+1].astype(np.float64); hv=st['host_v'][L].astype(np.float64)
    dv_true=tv-hv
    g=float(np.dot(dv_c,dv_true)/np.dot(dv_c,dv_c))
    key=f'L{L+1}'
    out[key]=(base[key].astype(np.float64)+g*dv_c).astype(base[key].dtype)
    print(f'L{L}: gamma {g:.4f} addcorr {np.corrcoef(dv_c,dv_true)[0,1]:.3f}')
np.savez('dvhat_rank0035_g3stack.npz',**out)
print('augmented dvhat saved')
PY
echo "=== consumption (base then stacked):"
OPENBLAS_NUM_THREADS=4 nice -n 15 python3 run_estimated_consumption.py 35 2>&1 | tail -1
DV_SUFFIX=_g3stack OPENBLAS_NUM_THREADS=4 nice -n 15 python3 run_estimated_consumption.py 35 2>&1 | tail -1
