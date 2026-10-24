"""Fixed two-net final-mean screen, not a tensor self-error gate."""
import hashlib
import json
from pathlib import Path
import time
import resource
import numpy as np
from threadpoolctl import threadpool_limits
from cp_only_one_high_reference_v1 import predict
from radial_features import features
from radial_gnn_numpy import gnn_step
from compact_scalar_final_v1 import final_scalar_features
from compact_final_head_v1 import final_input,final_delta

HERE=Path(__file__).resolve().parent;OUT=HERE/'cp_only_one_high_screen_v1.json';assert not OUT.exists()
price=json.loads((HERE/'cp_only_one_high_cost_v2.json').read_text());assert price['passed']
old=json.loads((HERE/'positive_path_closedloop_screen_v1.json').read_text())['records']
P=dict(np.load(HERE/'payload_compact_feed_fullpair_v1.npz'));Q=dict(np.load(HERE/'payload_fullpair_final_head_v1.npz'))
records=[];start=time.time()
with threadpool_limits(limits=4):
    for rank in (35,100):
        W=np.load(HERE/'rows_current_transport_shadow_v2'/f'row_{rank:04d}/weights.npy')
        for learned in (False,True):
            state=np.zeros((1024,64),np.float32)
            def hook(l,m,C,K21,aux,w):
                global state
                dm,state=gnn_step(l,features(l,m,C,K21,aux,w),aux['sigma'],w,state,P)
                return dm
            def final_hook(l,m,aux,w):
                h=final_input(final_scalar_features(l,m,aux,w),w,state,P)
                return final_delta(h,aux['sigma'],Q)
            mean,states=predict(W,hook=hook if learned else None,final_hook=final_hook if learned else None,collect_states=True)
            assert np.isfinite(mean).all()
            truth=np.load(HERE.parent/'public_g22_operator_features_v1/truthpost'/f'truthpost_rank{rank:04d}.npz')['mc_post']
            mse=np.mean((np.maximum(mean,0)-truth)**2,axis=1)
            baseline=next(r['parent_raw_final_mse'] for r in old if r['rank']==rank and r['frozen_bank_heads']==learned)
            saving=next(r['projected_fraction_saved'] for r in price['records'] if r['rank']==rank)
            r=dict(rank=rank,frozen_bank_heads=learned,parent_raw_final_mse=baseline,raw_final_mse=float(mse[-1]),
                   raw_ratio=float(mse[-1]/baseline),shape_only_adjusted_ratio=float(mse[-1]/baseline*(1-saving)),states=states,layer_mse=mse.tolist())
            records.append(r)
            artifact=HERE/f'cp_only_one_high_r{rank}_learned{int(learned)}_v1.npz';assert not artifact.exists();np.savez(artifact,mean=mean)
            print(json.dumps({k:v for k,v in r.items() if k not in ('states','layer_mse')}),flush=True)
passed=all(r['raw_ratio']<=1.15 and r['shape_only_adjusted_ratio']<=.85 for r in records if r['frozen_bank_heads'])
record=dict(records=records,passed=passed,wall_s=time.time()-start,peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            source_sha256=hashlib.sha256((HERE/'cp_only_one_high_reference_v1.py').read_bytes()).hexdigest(),
            qualification='New CP-only q4 recursion, f32 large products, exact separate coordinate star and trace. Fixed4 power/QR basis steps. Two-net 1e9 raw means; adjusted ratio is shape-only price projection, NOT measured bill, panel or official score.')
OUT.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(dict(passed=passed,wall_s=time.time()-start)),flush=True)
