"""Actual scalar-final inputs on each branch's frozen, co-adapted prefix."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from threadpoolctl import threadpool_limits
from radial_features import features
from radial_gnn_numpy import gnn_step
from compact_scalar_final_v1 import final_scalar_features
from compact_final_head_v1 import final_input,final_delta

HERE=Path(__file__).resolve().parent
ap=argparse.ArgumentParser();ap.add_argument('--branch',required=True,choices=('control','positive'));ap.add_argument('--rank',required=True,type=int);a=ap.parse_args()
assert a.rank in range(35,59)
if a.branch=='positive':from positive_path_final_reference_v1 import predict
else:from compact_fullpair_final_reference_v1 import predict
out=HERE/f'rows_positive_path_pilot_final_{a.branch}_v1'/f'row_{a.rank:04d}'
assert not out.exists();out.mkdir(parents=True)
payload=HERE/f'payload_positive_path_pilot_prefix_{a.branch}_v1.npz'
P=dict(np.load(payload));Q=dict(np.load(HERE/'payload_fullpair_final_head_v1.npz'))
store=HERE.parent/'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
entry=json.loads((store/'manifest.json').read_text())['roles']['observer_fit']['row_manifests'][a.rank]
W=np.load(store/Path(entry['path']).parent/'weights.npy',mmap_mode='r')
state=np.zeros((1024,P['head'].shape[1]),np.float32);saved={};start=time.time()
def hook(l,m,C,K21,aux,w):
    global state
    x=features(l,m,C,K21,aux,w);dm,state=gnn_step(l,x,aux['sigma'],w,state,P)
    return dm
def final_hook(l,m,aux,w):
    x=final_scalar_features(l,m,aux,w);h=final_input(x,w,state,P)
    saved.update(h=h.copy(),host=m.copy(),sigma=aux['sigma'].copy(),x=x.copy(),old_delta=final_delta(h,aux['sigma'],Q))
    return np.zeros_like(m)
with threadpool_limits(limits=4):
    mean,_=predict(W,hook=hook,final_hook=final_hook,gate_floor=.01,cov32=True,metric='covariance')
# All features and recurrent states above are fixed before targets open.
truth=np.load(HERE.parent/'public_g22_operator_features_v1/truthpost'/f'truthpost_rank{a.rank:04d}.npz')['mc_post'].astype(np.float64)
saved.update(target=truth[-1]-saved['host'],mean_prefix=mean)
assert all(np.all(np.isfinite(v)) for v in saved.values())
np.savez(out/'data.npz',**saved)
record=dict(rank=a.rank,branch=a.branch,wall_s=time.time()-start,prefix_payload_sha256=hashlib.sha256(payload.read_bytes()).hexdigest(),
            frozen_Q_mse=float(np.mean((saved['target']-saved['old_delta'])**2)),host_mse=float(np.mean(saved['target']**2)),
            qualification='No truth state feedback. Actual scalar final input, before final correction; fixed branch prefix.')
(out/'result.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record),flush=True)
