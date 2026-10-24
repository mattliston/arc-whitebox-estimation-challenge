"""Fixed current-host source audit, using already-acquired independent targets."""
import hashlib
import json
from pathlib import Path
import resource
import time
import numpy as np
from threadpoolctl import threadpool_limits
import radial_compact_feed_fullpair_v1 as parent
from linear_transport_residual_source_v1 import source as residual_source,certificate

HERE=Path(__file__).resolve().parent
OUT=HERE/'linear_transport_residual_source_r35_l12_v1.json';ART=OUT.with_suffix('.npz')
assert not OUT.exists() and not ART.exists()
small=certificate();f=np.load(HERE/'residual_one_high_inputs_r35_l12_v1.npz')
W=np.load(HERE/'rows_current_transport_shadow_v2/row_0035/weights.npy')
t=np.load(HERE.parent/'public_g19_attribution_v1/mps_cov2_rank0035.npz')
mt=t['mc_post'].astype(np.float64);ct=t['mc_post_cov'].astype(np.float64)
pre={};answers={};start=time.time()
class Done(Exception):pass


def pre_probe(l,v):
    if l==12:
        pre.update(star=v['star'].astype(np.float64),W=v['w'].copy(),bpre=v['bpre'].copy(),Bpre=v['Bpre'].copy(),alive=v['alive'].copy())


def hook(l,m,C,K21,aux,w):
    if l==12:
        assert np.array_equal((C+C.T)/2,f['host_covariance'])
        pre['g']=aux['gate'].copy()*pre['alive']
    return mt[l]-m if l>=1 else np.zeros_like(m)


def cov_hook(l,C):return ct[l].copy()


def post_probe(l,v):
    if l==12:
        args={k:pre[k] for k in ('star','W','g','bpre','Bpre')}
        answers.update(residual_source(**args,Bpost=v['Bpost'],U=f['U'],next_W=f['next_weight'],indices=f['indices']))
        raise Done


src=parent.PREDICT_SOURCE
for a,b in (
    ('gate_floor=.001):','gate_floor=.001, cov_hook=None):'),
    ('        m, C, target, radial_new, aux = nonlinear','        pre_probe(l,locals())\n        m, C, target, radial_new, aux = nonlinear'),
    ('            m = m+dm','            m = m+dm\n        if cov_hook is not None:\n            C = cov_hook(l,C)'),
    ('        radial = radial_new','        radial = radial_new\n        if cov_hook is not None:\n            C = cov_hook(l,C)'),
    ('        previous_alive = alive','        post_probe(l,locals())\n        previous_alive = alive')):
    assert src.count(a)==1;src=src.replace(a,b)
env=dict(parent.env,pre_probe=pre_probe,post_probe=post_probe);exec(src,env)
with threadpool_limits(limits=4):
    try:env['predict'](W,hook=hook,cov_hook=cov_hook,gate_floor=.01,cov32=True,metric='covariance')
    except Done:pass
    assert answers
    E={name:np.load(HERE/f'residual_one_high_truth_r35_l12_{name}_v1.npz')['next_K21']-f['model_next_K21'] for name in ('A','B')}
    def metrics(correction,weight):
        a=E['A']*weight;b=E['B']*weight;c=correction*weight
        s=float(np.mean(a*b));dot=float(np.mean((a+b)*c)/2);norm=float(np.mean(c*c))
        return dict(signal_error_ratio=(s-2*dot+norm)/s,signal_cosine=dot/np.sqrt(s*norm),
                    correction_relative_norm=np.sqrt(norm/s),A_ratio=float(np.sum((a-c)**2)/np.sum(a*a)),B_ratio=float(np.sum((b-c)**2)/np.sum(b*b)))
    result={arm:dict(unweighted=metrics(answers[arm],1.),weighted=metrics(answers[arm],f['consumer_weight'])) for arm in ('full','projected')}
v=result['projected']['weighted'];exact=result['full']['weighted']
retained=(1-v['signal_error_ratio'])/max(1-exact['signal_error_ratio'],1e-30)
gate=(v['signal_error_ratio']<=.95 and v['signal_cosine']>=.4 and v['A_ratio']<1 and v['B_ratio']<1
      and result['projected']['unweighted']['signal_error_ratio']<=1
      and (exact['signal_error_ratio']>=1 or retained>=.8))
np.savez(ART,**answers)
record=dict(rank=35,layer=12,certificate=small,original_host_byte_exact=True,results=result,
            projected_retained_full_gain=retained,source_gate=bool(gate),
            source_slice_norm=float(np.linalg.norm(answers['slices'])),star_slice_norm=float(answers['star_slice_norm']),metric_slice_norm=float(answers['metric_slice_norm']),
            wall_s=time.time()-start,peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            qualification='Exact same-state linear-third closure defect, natural coefficient1. Frozen q4 residual projection and32 next-K21 rows. Two independent8M targets, fixed source-v2 oracle previous m/C. No new samples, fitted coefficients, recursive estimator or score.')
OUT.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2),flush=True)
