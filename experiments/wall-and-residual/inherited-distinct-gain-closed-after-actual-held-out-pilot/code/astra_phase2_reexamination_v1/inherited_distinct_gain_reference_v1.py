"""Offline NumPy action: alter inherited CP only; preserve repeated plane."""
from pathlib import Path
import numpy as np
import compact_fullpair_final_reference_v1 as parent
from radial_features import features
from radial_gnn_numpy import gnn_step
from compact_scalar_final_v1 import final_scalar_features
from compact_final_head_v1 import final_input, final_delta

HERE=Path(__file__).resolve().parent
source=parent.source
edits=(
    ('gate_floor=.01, final_hook=None):', 'gate_floor=.01, final_hook=None, carry_theta=None):'),
    ('        cp_post = (inherited+born)*live',
     '        gain = 1+carry_theta[l]\n        cp_post = (inherited*gain**3+born)*live'),
    ('g.astype(np.float32)[:, None]*pp,', '(g*gain).astype(np.float32)[:, None]*pp,'),
)
for before,after in edits:
    assert source.count(before)==1,before
    source=source.replace(before,after)
env=dict(parent.env)
exec(source,env)
predict=env['predict']


def run(rank,theta=None,original=False):
    W=np.load(HERE/'rows_current_transport_shadow_v2'/f'row_{rank:04d}'/'weights.npy')
    P=dict(np.load(HERE/'payload_compact_feed_fullpair_v1.npz'))
    Q=dict(np.load(HERE/'payload_fullpair_final_head_v1.npz'))
    state=np.zeros((1024,P['head'].shape[1]),np.float32)
    def hook(l,m,C,K21,aux,w):
        nonlocal state
        dm,state=gnn_step(l,features(l,m,C,K21,aux,w),aux['sigma'],w,state,P)
        return dm
    def final_hook(l,m,aux,w):
        h=final_input(final_scalar_features(l,m,aux,w),w,state,P)
        return final_delta(h,aux['sigma'],Q)
    kwargs=dict(hook=hook,final_hook=final_hook,gate_floor=.01,cov32=True,metric='covariance')
    if original:
        out,_=parent.predict(W,**kwargs)
    else:
        theta=np.zeros(15) if theta is None else np.asarray(theta,np.float64)
        assert theta.shape==(15,) and theta[0]==0
        out,_=predict(W,carry_theta=theta,**kwargs)
    return np.maximum(out,0)
