"""Frozen current dynamic prefix with optional target-blind source reads."""
from pathlib import Path
import numpy as np
import compact_fullpair_final_reference_v1 as parent
from radial_features import features
from radial_gnn_numpy import gnn_step
from compact_scalar_final_v1 import final_scalar_features
from compact_final_head_v1 import final_input,final_delta
from final_variance_sources_v1 import covariance_parts,masked_with_feedback,variance_reads

HERE=Path(__file__).resolve().parent


def run(rank,collect=False):
    W=np.load(HERE/'rows_current_transport_shadow_v2'/f'row_{rank:04d}/weights.npy')
    P=dict(np.load(HERE/'payload_compact_feed_fullpair_v1.npz'))
    Q=dict(np.load(HERE/'payload_fullpair_final_head_v1.npz'))
    state=np.zeros((1024,64),np.float32);saved={};deltas={};counter=0
    original_nonlinear=parent.env['nonlinear']
    def nonlinear(mu,C,K21,k4,k22,metric):
        nonlocal counter
        output=original_nonlinear(mu,C,K21,k4,k22,metric)
        if collect and counter==14:
            saved['pair_parts']=covariance_parts(mu,C,K21,k4,k22,output[1])
            saved['pair_relative']=float(np.linalg.norm(saved['pair_parts'].sum(axis=0)-output[1])/np.linalg.norm(output[1]))
        counter+=1
        return output
    def hook(l,m,C,K21,aux,w):
        nonlocal state
        dm,state=gnn_step(l,features(l,m,C,K21,aux,w),aux['sigma'],w,state,P)
        if collect and l==14:deltas[l]=dm.copy()
        return dm
    def post_probe(l,t):
        if not collect or l!=14:return
        parts=masked_with_feedback(saved.pop('pair_parts'),t['host'],deltas[l],t['alive'])
        actual=t['C']
        saved['post_covariance_relative']=float(np.linalg.norm(parts.sum(axis=0)-actual)/np.linalg.norm(actual))
        saved['variance_parts']=variance_reads(parts,W[15])
    def final_hook(l,m,aux,w):
        h=final_input(final_scalar_features(l,m,aux,w),w,state,P)
        delta=final_delta(h,aux['sigma'],Q)
        saved.update(h=h.copy(),host=m.copy(),sigma=aux['sigma'].copy(),delta=delta.copy(),
                     variance=aux['pre_v'].copy())
        return delta
    source=parent.source
    if collect:
        needle='        previous_alive = alive'
        assert source.count(needle)==1
        source=source.replace(needle,'        post_probe(l,locals())\n'+needle)
    env=dict(parent.env,nonlinear=nonlinear if collect else original_nonlinear,post_probe=post_probe)
    exec(source,env)
    prediction,_=env['predict'](W,hook=hook,final_hook=final_hook,gate_floor=.01,cov32=True,metric='covariance')
    saved['prediction']=np.maximum(prediction,0)
    if collect:
        saved['variance_sum_relative']=float(np.linalg.norm(saved['variance_parts'].sum(axis=1)-saved['variance'])/np.linalg.norm(saved['variance']))
        saved['sources']=saved['variance_parts']/np.maximum(saved['variance'][:,None],1e-10)
    return saved
