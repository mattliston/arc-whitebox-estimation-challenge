"""Actual three-way scalar-final replay on a fixed pilot holdout."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from threadpoolctl import threadpool_limits
from compact_fullpair_final_reference_v1 import predict as parent_predict
from positive_path_final_reference_v1 import predict as positive_predict
from radial_features import features
from radial_gnn_numpy import gnn_step
from compact_scalar_final_v1 import final_scalar_features
from compact_final_head_v1 import final_input,final_delta

HERE=Path(__file__).resolve().parent
ap=argparse.ArgumentParser();ap.add_argument('--rank',required=True,type=int);a=ap.parse_args();assert a.rank in (37,39,42,48)
OUT=HERE/f'positive_path_pilot_judge_r{a.rank}_v1.json';assert not OUT.exists() and not OUT.with_suffix('.npz').exists()
store=HERE.parent/'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
entry=json.loads((store/'manifest.json').read_text())['roles']['observer_fit']['row_manifests'][a.rank]
W=np.load(store/Path(entry['path']).parent/'weights.npy',mmap_mode='r')
start=time.time();predictions={};parity={};hashes={}
with threadpool_limits(limits=4):
    for branch in ('bank','control','positive'):
        pp=HERE/('payload_compact_feed_fullpair_v1.npz' if branch=='bank' else f'payload_positive_path_pilot_prefix_{branch}_v1.npz')
        qp=HERE/('payload_fullpair_final_head_v1.npz' if branch=='bank' else f'payload_positive_path_pilot_final_{branch}_v1.npz')
        P=dict(np.load(pp));Q=dict(np.load(qp));hashes[branch]=dict(P=hashlib.sha256(pp.read_bytes()).hexdigest(),Q=hashlib.sha256(qp.read_bytes()).hexdigest())
        state=np.zeros((1024,P['head'].shape[1]),np.float32);saved={}
        def hook(l,m,C,K21,aux,w):
            global state
            dm,state=gnn_step(l,features(l,m,C,K21,aux,w),aux['sigma'],w,state,P)
            return dm
        def final_hook(l,m,aux,w):
            h=final_input(final_scalar_features(l,m,aux,w),w,state,P)
            saved.update(h=h.copy(),host=m.copy(),sigma=aux['sigma'].copy())
            return final_delta(h,aux['sigma'],Q)
        fn=positive_predict if branch=='positive' else parent_predict
        mean,_=fn(W,hook=hook,final_hook=final_hook,gate_floor=.01,cov32=True,metric='covariance')
        predictions[branch]=np.maximum(mean,0)
        if branch!='bank':
            frozen=np.load(HERE/f'rows_positive_path_pilot_final_{branch}_v1'/f'row_{a.rank:04d}/data.npz')
            parity[branch]={k:bool(np.array_equal(saved[k],frozen[k])) for k in ('h','host','sigma')}
            assert all(parity[branch].values()),(a.rank,branch,parity[branch])
truth=np.load(HERE.parent/'public_g22_operator_features_v1/truthpost'/f'truthpost_rank{a.rank:04d}.npz')['mc_post'].astype(np.float64)
layer={k:np.mean((v-truth)**2,axis=1).tolist() for k,v in predictions.items()};fin={k:v[-1] for k,v in layer.items()}
np.savez(OUT.with_suffix('.npz'),**predictions)
record=dict(rank=a.rank,final_mse=fin,layer_mse=layer,positive_control_ratio=fin['positive']/fin['control'],
            positive_bank_ratio=fin['positive']/fin['bank'],final_inputs_byte_exact=parity,payload_hashes=hashes,wall_s=time.time()-start,
            qualification='Actual NumPy full closed-loop scalar-final trajectories; four-net pilot, existing1e9 truths. Not metered, adjusted, panel or official score.')
OUT.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps({k:v for k,v in record.items() if k not in ('layer_mse','payload_hashes')},indent=2),flush=True)
