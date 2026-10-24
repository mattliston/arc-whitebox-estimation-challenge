"""Fixed covariance/next-value transfer diagnostic with independent labels."""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from threadpoolctl import threadpool_limits
import positive_path_birth_reference_v1 as changed
import radial_compact_feed_fullpair_v1 as parent

HERE=Path(__file__).resolve().parent;OUT=HERE/'positive_path_value_transfer_r35_l12_v1.json';ART=OUT.with_suffix('.npz')
assert not OUT.exists() and not ART.exists()
W=np.load(HERE/'rows_current_transport_shadow_v2/row_0035/weights.npy')
t=np.load(HERE.parent/'public_g19_attribution_v1/mps_cov2_rank0035.npz')
mt=t['mc_post'].astype(np.float64);ct=t['mc_post_cov'].astype(np.float64)
f=np.load(HERE/'residual_one_high_inputs_r35_l12_v1.npz');start=time.time()
class Done(Exception):pass


def run(module):
    saved={}
    def hook(l,m,C,K21,aux,w):
        if l==12:
            saved.update(mean=m.copy(),cov=(C+C.T)/2);raise Done
        return mt[l]-m if l>=1 else np.zeros_like(m)
    def cov_hook(l,C):return ct[l].copy()
    src=module.source if module is changed else module.PREDICT_SOURCE
    old='path_mode=True):' if module is changed else 'gate_floor=.001):'
    new='path_mode=True, cov_hook=None):' if module is changed else 'gate_floor=.001, cov_hook=None):'
    for a,b in ((old,new),
                ('            m = m+dm','            m = m+dm\n        if cov_hook is not None:\n            C = cov_hook(l,C)'),
                ('        radial = radial_new','        radial = radial_new\n        if cov_hook is not None:\n            C = cov_hook(l,C)')):
        assert src.count(a)==1;src=src.replace(a,b)
    env=dict(module.env);exec(src,env)
    try:env['predict'](W,hook=hook,cov_hook=cov_hook,gate_floor=.01,cov32=True,metric='covariance')
    except Done:pass
    return saved


def metrics(Ea,Eb,delta):
    signal=float(np.mean(Ea*Eb));observed=float(.5*np.mean(Ea*Ea+Eb*Eb))
    dot=float(.5*np.mean((Ea+Eb)*delta));energy=float(np.mean(delta*delta))
    return dict(parent_signal_mse=signal,parent_observed_mse=observed,
                observed_noise_fraction=float(.25*np.mean((Ea-Eb)**2)/observed),
                signal_error_ratio=(signal-2*dot+energy)/signal,
                signal_cosine=dot/np.sqrt(max(signal*energy,1e-60)),correction_relative_norm=np.sqrt(energy/max(signal,1e-60)),
                A_ratio=float(np.sum((Ea-delta)**2)/np.sum(Ea*Ea)),B_ratio=float(np.sum((Eb-delta)**2)/np.sum(Eb*Eb)))


with threadpool_limits(limits=4):
    old=run(parent);new=run(changed)
    assert np.array_equal(old['cov'],f['host_covariance'])
    truth=np.load(HERE.parent/'public_g22_operator_features_v1/truthpost/truthpost_rank0035.npz')['mc_post'][12].astype(np.float64)
    covs={}
    for name in ('A','B'):
        v=np.load(HERE/f'residual_one_high_truth_r35_l12_{name}_v1.npz');P=int(v['P'])
        covs[name]=v['post_covariance']*(P/(P-1))
    wn=f['next_weight'];sigma=np.sqrt(np.diag(f['next_pre_covariance']));alpha=f['next_pre_shift']/sigma
    response=np.exp(-alpha*alpha/2)/(2*np.sqrt(2*np.pi)*sigma)
    def value(C):return response*np.sum(wn*(C@wn),axis=0)
    results={};artifacts={}
    for arm in ('before_mean_update','consistent_1e9_mean_update'):
        a=old['cov'].copy();b=new['cov'].copy()
        if arm=='consistent_1e9_mean_update':
            fixed=np.outer(truth,truth)
            a+=np.outer(old['mean'],old['mean'])-fixed
            b+=np.outer(new['mean'],new['mean'])-fixed
        ea=covs['A']-a;eb=covs['B']-a;dc=b-a
        results[arm]=dict(covariance=metrics(ea,eb,dc),next_gaussian_mean_linear_response=metrics(value(ea),value(eb),value(dc)))
        artifacts[arm+'_correction']=dc
    mean_mse={name:float(np.mean((data['mean']-truth)**2)) for name,data in (('parent',old),('positive',new))}
np.savez(ART,**artifacts,parent_mean=old['mean'],positive_mean=new['mean'])
record=dict(rank=35,layer=12,next_layer=13,parent_host_byte_exact=True,results=results,current_one_step_mean_mse_1e9=mean_mse,
            wall_s=time.time()-start,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            qualification='Two independent8M covariance labels; covariance and specified next-Gaussian-mean linear response, not final MSE. Fixed1e9-mean correction arm is diagnostic only; no new fit or deployed truth feedback.')
OUT.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2),flush=True)
