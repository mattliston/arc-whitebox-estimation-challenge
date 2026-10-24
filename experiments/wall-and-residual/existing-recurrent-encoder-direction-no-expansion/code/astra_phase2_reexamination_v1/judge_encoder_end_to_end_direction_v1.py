"""Independent actual NumPy judgement, same model/masks/frozen scalar head."""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from threadpoolctl import threadpool_limits
from compact_fullpair_final_reference_v1 import predict
from radial_features import features
from radial_gnn_numpy import gnn_step
from compact_scalar_final_v1 import final_scalar_features
from compact_final_head_v1 import final_input,final_delta

HERE=Path(__file__).resolve().parent
OUT=HERE/'encoder_end_to_end_actual_verdict_v1.json'
ART=OUT.with_suffix('.npz')
assert not OUT.exists() and not ART.exists()
probe=json.loads((HERE/'encoder_end_to_end_direction_v1.json').read_text())
cpu=np.load(HERE/'encoder_end_to_end_actual_torch_v1.npz')
P=dict(np.load(HERE/'payload_encoder_end_to_end_direction_v1.npz'))
oldP=dict(np.load(HERE/'payload_compact_feed_fullpair_v1.npz'))
Q=dict(np.load(HERE/'payload_fullpair_final_head_v1.npz'))
assert all(np.array_equal(P[k],oldP[k]) for k in oldP if k not in probe['keys'])
start=time.time();records=[];saves={}
with threadpool_limits(limits=3):
    for rank in probe['valid']:
        W=np.load(HERE/'rows_current_transport_shadow_v2'/f'row_{rank:04d}'/'weights.npy')
        state=np.zeros((1024,P['head'].shape[1]),np.float32)
        def hook(l,m,C,K21,aux,w):
            global state
            dm,state=gnn_step(l,features(l,m,C,K21,aux,w),aux['sigma'],w,state,P)
            return dm
        def final_hook(l,m,aux,w):
            h=final_input(final_scalar_features(l,m,aux,w),w,state,P)
            return final_delta(h,aux['sigma'],Q)
        pred,_=predict(W,hook=hook,final_hook=final_hook,gate_floor=.01,
                       cov32=True,metric='covariance')
        pred=np.maximum(pred,0)
        # The old three-way pilot already froze this exact NumPy bank
        # prediction with hashes of both unchanged bank payloads.
        receipt=json.loads((HERE/f'positive_path_pilot_judge_r{rank}_v1.json').read_text())
        assert receipt['payload_hashes']['bank']['P']==hashlib.sha256((HERE/'payload_compact_feed_fullpair_v1.npz').read_bytes()).hexdigest()
        assert receipt['payload_hashes']['bank']['Q']==hashlib.sha256((HERE/'payload_fullpair_final_head_v1.npz').read_bytes()).hexdigest()
        base=np.load(HERE/f'positive_path_pilot_judge_r{rank}_v1.npz')['bank']
        label=np.load(HERE.parent/'public_g22_operator_features_v1/truthpost'/f'truthpost_rank{rank:04d}.npz')['mc_post']
        mse=np.mean((pred-label)**2,axis=1)
        old=np.mean((base-label)**2,axis=1)
        diff=float(np.max(np.abs(pred-cpu[f'mean_{rank}'])))
        assert np.isfinite(pred).all() and diff<2e-6,(rank,diff)
        row=dict(rank=rank,parent_raw=float(old[-1]),raw=float(mse[-1]),ratio=float(mse[-1]/old[-1]),
                 layer_mse=mse.tolist(),parent_layer_mse=old.tolist(),
                 torch_actual_max_difference=diff,wall_s=time.time()-start)
        records.append(row);saves[f'mean_{rank}']=pred
        print(json.dumps({k:v for k,v in row.items() if k not in ('layer_mse','parent_layer_mse')}),flush=True)
ratio=sum(r['raw'] for r in records)/sum(r['parent_raw'] for r in records)
wins=sum(r['ratio']<1 for r in records)
np.savez(ART,**saves)
record=dict(records=records,pooled_raw_ratio=ratio,holdout_wins=wins,
            expansion_gate=ratio<=.90 and wins>=3,
            applied_train_only_step=probe['applied_step'],
            wall_s=time.time()-start,payload_sha256=probe['payload_sha256'],
            qualification='Four pilot-heldout actual NumPy final-layer MSEs against existing1B labels, no head/physics/capacity changes and no deployment FLOP claim. Not the independent14-net panel, metered candidate or official adjusted score.')
OUT.write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps({k:v for k,v in record.items() if k!='records'},indent=2),flush=True)
