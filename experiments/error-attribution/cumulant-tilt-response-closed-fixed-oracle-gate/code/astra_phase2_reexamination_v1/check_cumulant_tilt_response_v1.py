"""One fixed oracle-mean covariance-response opportunity test."""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from scipy.linalg import eigvalsh,solve
from threadpoolctl import threadpool_limits
import radial_compact_feed_fullpair_v1 as parent
from cumulant_tilt_response_v1 import variance_response,certificate

HERE=Path(__file__).resolve().parent;OUT=HERE/'cumulant_tilt_response_r35_l12_v1.json';ART=OUT.with_suffix('.npz')
assert not OUT.exists() and not ART.exists();small=certificate()
W=np.load(HERE/'rows_current_transport_shadow_v2/row_0035/weights.npy')
t=np.load(HERE.parent/'public_g19_attribution_v1/mps_cov2_rank0035.npz')
mt=t['mc_post'].astype(np.float64);ct=t['mc_post_cov'].astype(np.float64)
f=np.load(HERE/'residual_one_high_inputs_r35_l12_v1.npz');saved={};start=time.time()
class Done(Exception):pass
def hook(l,m,C,K21,aux,w):
    if l==12:
        saved.update(mean=m.copy(),cov=(C+C.T)/2)
        return np.zeros_like(m)
    return mt[l]-m if l>=1 else np.zeros_like(m)
def cov_hook(l,C):return ct[l].copy() if l<12 else C
def probe(l,v):
    if l!=12:return
    saved.update(p=v['p'].astype(np.float64),lam=v['lam'].astype(np.float64),star=v['star'].astype(np.float64),b=v['b'].copy(),B=v['Bpost'].copy(),indices=v['live_indices'].copy())
    raise Done
src=parent.PREDICT_SOURCE
for a,b in (('gate_floor=.001):','gate_floor=.001, cov_hook=None):'),
            ('            m = m+dm','            m = m+dm\n        if cov_hook is not None:\n            C = cov_hook(l,C)'),
            ('        radial = radial_new','        radial = radial_new\n        if cov_hook is not None:\n            C = cov_hook(l,C)'),
            ('        previous_alive = alive','        probe(l,locals())\n        previous_alive = alive')):
    assert src.count(a)==1;src=src.replace(a,b)
env=dict(parent.env,probe=probe);exec(src,env)
with threadpool_limits(limits=4):
    try:env['predict'](W,hook=hook,cov_hook=cov_hook,gate_floor=.01,cov32=True,metric='covariance')
    except Done:pass
    assert np.array_equal(saved['cov'],f['host_covariance'])
    truth=np.load(HERE.parent/'public_g22_operator_features_v1/truthpost/truthpost_rank0035.npz')['mc_post'][12].astype(np.float64)
    idx=saved['indices'];C=saved['cov'];Ca=C[np.ix_(idx,idx)]
    minimum=float(eigvalsh(Ca,subset_by_index=(0,0))[0]);assert minimum>0,minimum
    delta=truth-saved['mean'];eta=np.zeros(1024);eta[idx]=solve(Ca,delta[idx],assume_a='pos')
    solve_error=float(np.linalg.norm(Ca@eta[idx]-delta[idx])/np.linalg.norm(delta[idx]));assert solve_error<1e-8
    energy=float(.5*delta[idx]@eta[idx]);wn=f['next_weight']
    dv=variance_response(*(saved[k] for k in ('p','lam','star','b','B')),eta,wn)
    variance=np.sum(wn*(C@wn),axis=0);max_relative=float(np.max(np.abs(dv)/variance))
    sigma=np.sqrt(np.diag(f['next_pre_covariance']));alpha=f['next_pre_shift']/sigma
    response=np.exp(-alpha*alpha/2)/(2*np.sqrt(2*np.pi)*sigma)
    errors={}
    for name in ('A','B'):
        z=np.load(HERE/f'residual_one_high_truth_r35_l12_{name}_v1.npz');P=int(z['P']);cov=z['post_covariance']*(P/(P-1))
        errors[name]=response*np.sum(wn*((cov-C)@wn),axis=0)
    a=errors['A'];b=errors['B'];c=response*dv
    signal=float(np.mean(a*b));dot=float(.5*np.mean((a+b)*c));power=float(np.mean(c*c))
    result=dict(signal_error_ratio=(signal-2*dot+power)/signal,signal_cosine=dot/np.sqrt(signal*power),
                A_ratio=float(np.mean((a-c)**2)/np.mean(a*a)),B_ratio=float(np.mean((b-c)**2)/np.mean(b*b)))
passed=result['signal_error_ratio']<=.75 and result['A_ratio']<=.85 and result['B_ratio']<=.85 and energy<=.01 and max_relative<=.1
np.savez(ART,eta=eta,delta_mean=delta,delta_variance=dv,value_correction=c)
record=dict(rank=35,layer=12,next_layer=13,certificate=small,parent_host_byte_exact=True,active=len(idx),minimum_active_cov_eigenvalue=minimum,
            solve_relative_error=solve_error,quadratic_tilt_energy=energy,max_relative_variance_change=max_relative,results=result,
            opportunity_gate=bool(passed),wall_s=time.time()-start,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            qualification='Oracle1e9 active mean correction and current modeled third cumulant, exact next variance query. Same two independent8M covariance labels. Not a deployed state change, full cost, confidence bound, raw or adjusted score.')
OUT.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2),flush=True)
