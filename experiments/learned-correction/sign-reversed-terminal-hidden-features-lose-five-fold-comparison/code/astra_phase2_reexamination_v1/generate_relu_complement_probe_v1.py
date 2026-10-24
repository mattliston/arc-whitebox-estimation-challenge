"""Local sign-complement head read, exact same positive teacher trajectory."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from threadpoolctl import threadpool_limits
from radial_compact_feed_fullpair_v1 import predict
from radial_features import features
from radial_gnn_numpy import gnn_step
from radial_mehler import make_nonlinear

here=Path(__file__).resolve().parent
store=here.parent/'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
ap=argparse.ArgumentParser();ap.add_argument('--rank',type=int,required=True);args=ap.parse_args()
assert 35<=args.rank<59
out=here/'rows_relu_complement_probe_v1'/f'row_{args.rank:04d}'
assert not out.exists();out.mkdir(parents=True)
man=json.loads((store/'manifest.json').read_text())['roles']['observer_fit']['row_manifests']
W=np.load(store/Path(man[args.rank]['path']).parent/'weights.npy',mmap_mode='r')
truth=np.load(here.parent/'public_g22_operator_features_v1/truthpost'/f'truthpost_rank{args.rank:04d}.npz')['mc_post'].astype(np.float64)
P=dict(np.load(here/'payload_compact_feed_fullpair_v1.npz'))
nonlinear=make_nonlinear(6,2,True,True)
state=np.zeros((1024,P['head'].shape[1]),np.float32)
xs=[];hosts=[];sigmas=[];deltas=[];negative=[];identity=[]

def hook(l,m,C,K21,aux,w):
    global state
    x=features(l,m,C,K21,aux,w)
    delta,next_state=gnn_step(l,x,aux['sigma'],w,state,P)
    if l>=10:
        mu=aux['alpha']*aux['sigma'];v=aux['pre_v'];pre=aux['pre_cov']
        radial=float(np.mean(aux['pre_k4']/(v*v)))
        k22=radial*(np.outer(v,v)+2*pre*pre)/3
        mn,cn,_,_,an=nonlinear(-mu,pre,-K21,aux['pre_k4'],k22,'covariance')
        xn=features(l,mn,cn,-K21,an,-w)
        dn,_=gnn_step(l,xn,an['sigma'],-w,state,P)
        identity.append(float(np.max(np.abs((m-mn)-mu))))
        negative.append(dn.copy())
    state=next_state
    xs.append(x.copy());hosts.append(m.copy());sigmas.append(aux['sigma'].copy());deltas.append(delta.copy())
    return truth[l]-m if l>=1 else np.zeros_like(m)

start=time.time()
with threadpool_limits(limits=4):
    predict(W,gate_floor=.01,hook=hook,cov32=True,metric='covariance',collect_states=False)
x=np.stack(xs);host=np.stack(hosts);sigma=np.stack(sigmas);delta=np.stack(deltas)
parent=here/'rows_compact_feed_fullpair_v1'/f'row_{args.rank:04d}'
assert np.array_equal(host,np.load(parent/'host.npy'))
assert np.array_equal(sigma,np.load(parent/'sigma.npy'))
assert np.array_equal(x[:,:,:20],np.load(parent/'raw20.npy'))
assert np.array_equal(x[:,:,20:],np.load(parent/'mech.npy'))
old=np.load(here/'rows_joint_plane_probe_v1'/f'row_{args.rank:04d}'/'data.npz')
assert np.array_equal(delta,old['delta'])
assert max(identity)<1e-12,max(identity)
dn=np.stack(negative);dp=delta[10:];target=(truth-host)[10:]
sym=(dp+dn)/2
assert np.isfinite(dn).all()
data=dict(target=target,positive=dp,negative=dn,symmetric=sym)
np.savez(out/'data.npz',**data)
receipt=dict(rank=args.rank,parent_byte_exact=True,mean_complement_max_abs=max(identity),
    layer_base_mse=np.mean((target-dp)**2,axis=1).tolist(),
    layer_symmetric_mse=np.mean((target-sym)**2,axis=1).tolist(),
    layer_difference_energy=np.mean((dp-dn)**2,axis=1).tolist(),wall_s=time.time()-start)
(out/'result.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(dict(rank=args.rank,parent_byte_exact=True,complement_error=max(identity),
    mean_base=float(np.mean((target-dp)**2)),mean_symmetric=float(np.mean((target-sym)**2)))),flush=True)
