"""Oracle-state, source-only covariance labels with strict current-feature order."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from scipy.special import ndtr
from threadpoolctl import threadpool_limits
import radial_compact_feed_fullpair_v1 as base

here=Path(__file__).resolve().parent
g19=here.parent/'public_g19_attribution_v1'
store=here.parent/'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
ap=argparse.ArgumentParser();ap.add_argument('--rank',type=int,required=True);args=ap.parse_args()
assert args.rank in (35,100,150,200,250,300,400,500)
out=here/'rows_covariance_innovation_probe_v1'/f'row_{args.rank:04d}'
assert not out.exists();out.mkdir(parents=True)
man=json.loads((store/'manifest.json').read_text())['roles']['observer_fit']['row_manifests']
W=np.load(store/Path(man[args.rank]['path']).parent/'weights.npy',mmap_mode='r')
mean_truth=np.load(here.parent/'public_g22_operator_features_v1/truthpost'/f'truthpost_rank{args.rank:04d}.npz')['mc_post'].astype(np.float64)
cov_truth=np.load(g19/f'mps_cov2_rank{args.rank:04d}.npz')
post_cov=cov_truth['mc_post_cov'].astype(np.float64)
pre_v=cov_truth['mc_pre_v'].astype(np.float64)
source=base.PREDICT_SOURCE
assert source.count('gate_floor=.001):')==1
source=source.replace('gate_floor=.001):','gate_floor=.001, cov_hook=None):')
assert source.count('            m = m+dm')==1
source=source.replace('            m = m+dm','            m = m+dm\n        if cov_hook is not None:\n            C = cov_hook(l,C)')
assert source.count('        radial = radial_new')==1
source=source.replace('        radial = radial_new','        radial = radial_new\n        if cov_hook is not None:\n            C = cov_hook(l,C)')
env=dict(base.env);exec(source,env);predict=env['predict']
saved={k:[] for k in ('basis','rho','sigma','target','weight','response')}
host_error=[];cov_error=[]

def hook(l,m,C,K21,aux,w):
    host_error.append(float(np.mean((m-mean_truth[l])**2)))
    if 8<=l<=14:
        a=aux['alpha'];p=aux['pdf'];g=aux['gate'];s=aux['sigma']
        skew=aux['pre_k3']/s**3;kurt=aux['pre_k4']/s**4;ms=m/s
        F=np.stack((g,p,a*p,(a*a-1)*p,ms,a*g,skew*g,skew*p,skew*a*p,skew*ms,
                    kurt*g,kurt*p,kurt*a*p,kurt*ms),axis=1).astype(np.float32)
        R=(aux['pre_cov']/np.outer(s,s)).astype(np.float32)
        # Freeze inputs before reading the current covariance label below.
        saved['basis'].append(F.copy());saved['rho'].append(R.copy());saved['sigma'].append(s.astype(np.float32))
        E=post_cov[l]-C
        E=(E+E.T)/2
        wn=W[l+1].astype(np.float64)
        next_mu=wn.T@mean_truth[l]
        next_sigma=np.sqrt(np.maximum(pre_v[l+1],1e-20))
        response=np.exp(-.5*(next_mu/next_sigma)**2)/(np.sqrt(2*np.pi)*2*next_sigma)
        saved['target'].append(E.astype(np.float32));saved['weight'].append(wn.astype(np.float32));saved['response'].append(response.astype(np.float32))
        cov_error.append(dict(layer=l,frobenius_mse=float(np.mean(E*E)),
            next_mean_response_mse=float(np.mean((np.sum(wn*(E@wn),axis=0)*response)**2))))
    return mean_truth[l]-m if l>=1 else np.zeros_like(m)

def cov_hook(l,C):
    return post_cov[l].copy()

start=time.time()
with threadpool_limits(limits=4):
    predict(W,hook=hook,cov_hook=cov_hook,gate_floor=.01,cov32=True,metric='covariance',collect_states=False)
data={k:np.stack(v) for k,v in saved.items()}
assert all(np.isfinite(v).all() for v in data.values())
assert data['basis'].shape==(7,1024,14) and data['target'].shape==(7,1024,1024)
np.savez(out/'data.npz',**data)
record=dict(rank=args.rank,samples=int(cov_truth['P']),layers=list(range(8,15)),
    host_one_step_mse=host_error,covariance_error=cov_error,wall_s=time.time()-start,
    provenance='Current input matrices frozen before current mean/covariance labels; true previous covariance and mean supplied. Oracle-state source diagnostic, not deployment.')
(out/'result.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(dict(rank=args.rank,wall_s=record['wall_s'],late_covariance_mse=float(np.mean([r['frobenius_mse'] for r in cov_error])))),flush=True)
