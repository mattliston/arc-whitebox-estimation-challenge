"""Actual carrier verdict for the fixed suffix payload, four held-out nets."""
import json
from pathlib import Path
import time
import numpy as np
from threadpoolctl import threadpool_limits
from radial_compact_feed_fullpair_v1 import predict
from radial_features import features
from radial_gnn_numpy import gnn_step

here=Path(__file__).resolve().parent
out=here/'late_transport_actual_verdict_v1.json'
assert not out.exists()
training=json.loads((here/'late_transport_training_pilot_v1.json').read_text())
assert training['epochs']==5
P0=dict(np.load(here/'payload_compact_feed_fullpair_v1.npz'))
P1=dict(np.load(here/'payload_late_transport_pilot_v1.npz'))
start=time.time();rows=[]
with threadpool_limits(limits=4):
    for rank in (37,39,42,48):
        folder=here/'rows_current_transport_shadow_v2'/f'row_{rank:04d}'
        W=np.load(folder/'weights.npy')
        baseline=np.load(folder/'baseline.npy')
        state=np.zeros((1024,64),np.float32)
        def hook(l,m,C,K21,aux,w):
            global state
            x=features(l,m,C,K21,aux,w)
            delta,state=gnn_step(l,x,aux['sigma'],w,state,P1 if l>=9 else P0)
            return delta
        mean,_=predict(W,hook=hook,gate_floor=.01,cov32=True,metric='covariance',collect_states=False)
        assert np.array_equal(mean[:9],baseline[:9]),'prefix drift'
        truth=np.load(here.parent/'public_g22_operator_features_v1/truthpost'/f'truthpost_rank{rank:04d}.npz')['mc_post']
        e=np.mean((np.maximum(mean,0)-truth)**2,axis=1)
        e0=np.mean((np.maximum(baseline,0)-truth)**2,axis=1)
        pred_path=here/f'late_transport_actual_r{rank}_v1.npz'
        assert not pred_path.exists()
        np.savez(pred_path,mean=mean)
        r=dict(rank=rank,prefix_byte_exact=True,final_mse=float(e[-1]),parent_mse=float(e0[-1]),
               ratio=float(e[-1]/e0[-1]),layer_mse=e.tolist(),parent_layer_mse=e0.tolist())
        rows.append(r);print(json.dumps({k:v for k,v in r.items() if 'layer_mse' not in k}),flush=True)
passed=all(r['ratio']<=.92 for r in rows)
receipt=dict(rows=rows,actual_heldout_gate_pass=passed,wall_s=time.time()-start,
             qualification='Actual full dynamic carrier; four fixed held-out public pilot nets, stored 1e9 targets. Not the 14-net adjusted gate or an official score.')
out.write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(dict(actual_heldout_gate_pass=passed)),flush=True)
